"""Train two independent models: binary/ranking, then native CLEAN EC."""
from __future__ import annotations
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from .data import Dataset, FeatureStore, ROOT, DATA, FEATURE_MATRICES, PLM
from .model import BinaryModel, ECModel, MODEL_POLICY
from .native_ec import (set_seed, train_memberships, encode, centers_from_embeddings,
                       neighbor_pool, sample_triplets)
from .training_defaults import (ADAM_BETAS, ADAPTIVE_RATE, LEARNING_RATE, MAX_EPOCHS,
    SEED, TRIPLET_MARGIN, GLOBAL_GROUP_BATCH, BINARY_EPOCHS,
    EC_BATCH_SIZE, EC_EMBED_BATCH_SIZE, HARD_NEGATIVE_K)

def losses_for_group(logits: torch.Tensor, unmarked: torch.Tensor,
                     group_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    # logits: [n_groups, 4] in P, D0, D5, D10 order.
    p, d0, d5, d10 = logits.unbind(dim=1)
    bce = (group_scale * (0.5 * F.binary_cross_entropy_with_logits(p, torch.ones_like(p))
                          + sum(F.binary_cross_entropy_with_logits(d, torch.zeros_like(d))
                                for d in (d0, d5, d10)) / 12)
           + 0.25 * F.binary_cross_entropy_with_logits(unmarked, torch.zeros_like(unmarked)))
    rank = group_scale * (F.relu(1 - p + d0) + F.relu(1 - d0 + d5)
                          + F.relu(1 - d5 + d10)).mean() / 3
    return bce, rank


def shuffled_queue(values: np.ndarray, rng: np.random.Generator):
    values = values.copy()
    while True:
        rng.shuffle(values)
        yield from values



def _save_model(dest, name, model, config, epochs):
    temporary = dest / f"{name}_checkpoint.pt.tmp"
    torch.save({"state_dict": model.state_dict(), "config": config, "epoch": epochs}, temporary)
    temporary.replace(dest / f"{name}_checkpoint.pt")
    (dest / f"{name}_training_done.json").write_text(json.dumps(
        {"model_policy": MODEL_POLICY, "completed_epochs": epochs,
         "checkpoint_epoch": epochs, "checkpoint_policy": "final"}, indent=2))


def _already_done(dest, name, config, overwrite):
    marker = dest / f"{name}_training_done.json"
    if overwrite or not marker.exists():
        return False
    previous = json.loads((dest / f"{name}_config.json").read_text())
    if previous != config:
        raise ValueError(f"Existing {name} settings differ; choose another results directory or use --overwrite")
    if not (dest / f"{name}_checkpoint.pt").is_file():
        raise FileNotFoundError(f"Completed {name} model lacks its checkpoint")
    return True


def train(split: int, fold: int, device_name: str,
          max_epochs: int = MAX_EPOCHS, seed: int = SEED,
          group_batch: int = GLOBAL_GROUP_BATCH, adaptive_rate: int = ADAPTIVE_RATE,
          results_dir: Path | None = None, binary_epochs: int = BINARY_EPOCHS,
          overwrite: bool = False, data_dir: Path = DATA,
          cache_dir: Path = FEATURE_MATRICES) -> Path:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Run independent jobs with decoyopt train")
    if min(max_epochs, binary_epochs, group_batch, adaptive_rate) < 1:
        raise ValueError("Epoch counts, group batch and adaptive rate must be positive")
    data = Dataset(split, fold, data_dir)
    store = FeatureStore(data, cache_dir)
    device = torch.device(device_name)
    dest = (results_dir or ROOT / "results") / str(split) / f"fold_{fold}"
    dest.mkdir(parents=True, exist_ok=True)
    common = {"model_policy": MODEL_POLICY, "split": split, "fold": fold, "plm": PLM,
              "input_dim": store.dim, "seed": seed, "frozen_plm": True,
              "feature_source": str(store.data_path), "optimizer": "Adam",
              "lr": LEARNING_RATE, "adam_betas": list(ADAM_BETAS), "weight_decay": 0.0,
              "lr_schedule": "constant", "early_stopping": False,
              "checkpoint_policy": "final", "gradient_clip_norm": None}
    binary_config = {**common, "model": "binary", "max_epochs": binary_epochs,
                     "group_batch": group_batch, "optimizer_steps_per_epoch": 1,
                     "rank_margin": 1.0, "rank_weight": 0.5,
                     "bce_weights": {"P": 0.5, "U": 0.25, "D0": 1/12, "D5": 1/12, "D10": 1/12},
                     "dropout": 0.0, "excluded_decoys_without_split_parent": data.excluded_decoys}
    ec_config = {**common, "model": "native_clean_ec", "max_epochs": max_epochs,
                 "batch_size": EC_BATCH_SIZE, "triplets_per_epoch": len(data.ecs),
                 "optimizer_steps_per_epoch": (len(data.ecs) + EC_BATCH_SIZE - 1) // EC_BATCH_SIZE,
                 "adaptive_rate": adaptive_rate, "hard_negative_k": HARD_NEGATIVE_K,
                 "triplet_margin": TRIPLET_MARGIN, "dropout": 0.1,
                 "architecture": "CLEAN.LayerNormNet(input_dim,512,512,128)",
                 "singleton_positive": "two independent dropout views"}
    # Evaluation outputs must not survive a training invocation.
    for filename in ("metrics.json", "predictions.csv", "thresholds.json"):
        (dest / filename).unlink(missing_ok=True)
    (dest.parent / "five_fold_summary.csv").unlink(missing_ok=True)
    # A marker is written only when both separately trained models are complete.
    (dest / "training_done.json").unlink(missing_ok=True)
    if not _already_done(dest, "binary", binary_config, overwrite):
        (dest / "binary_training_done.json").unlink(missing_ok=True)
        (dest / "binary_config.json").write_text(json.dumps(binary_config, indent=2))
        set_seed(seed)
        rng = np.random.default_rng(seed)
        model = BinaryModel(store.dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
        parents = shuffled_queue(np.arange(len(data.groups["train"]), dtype=np.int32), rng)
        if not len(data.unmarked["train"]) or not len(data.groups["train"]):
            raise ValueError("Binary training needs parent groups and unmarked sequences")
        unmarked = shuffled_queue(data.unmarked["train"], rng)
        with (dest / "binary_history.jsonl").open("w") as history:
            epochs = tqdm(range(1, binary_epochs + 1), desc=f"Binary {split}/fold_{fold}", unit="epoch", mininterval=1)
            model.train()
            for epoch in epochs:
                chosen = np.fromiter((next(parents) for _ in range(group_batch)), dtype=np.int32, count=group_batch)
                quad = data.groups["train"][chosen]
                u = np.fromiter((next(unmarked) for _ in range(group_batch)), dtype=np.int32, count=group_batch)
                indices = np.concatenate((quad.reshape(-1), u))
                scores = model(store.batch(indices, device))
                bce, rank = losses_for_group(scores[:quad.size].reshape(-1, 4), scores[quad.size:])
                loss = bce + 0.5 * rank
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                history.write(json.dumps({"epoch": epoch, "lr": LEARNING_RATE,
                    "train": {"total": float(loss.detach()), "bce": float(bce.detach()), "rank": float(rank.detach())}}) + "\n")
                history.flush()
                epochs.set_postfix(loss=f"{float(loss.detach()):.4f}")
        _save_model(dest, "binary", model, binary_config, binary_epochs)
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not _already_done(dest, "ec", ec_config, overwrite):
        (dest / "ec_training_done.json").unlink(missing_ok=True)
        (dest / "ec_config.json").write_text(json.dumps(ec_config, indent=2))
        # Reset every RNG: binary training cannot affect EC initialization/sampling.
        set_seed(seed)
        records = [(data.ids[int(i)], data.labels[int(i)]) for i in data.positives["train"]]
        vocabulary, members, labels = train_memberships(records)
        features = store.batch(data.positives["train"], device)
        model = ECModel(store.dim, device=device)
        criterion = torch.nn.TripletMarginLoss(margin=TRIPLET_MARGIN, reduction="mean")
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
        neighbors, weights = neighbor_pool(centers_from_embeddings(features, members), HARD_NEGATIVE_K)
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        with (dest / "ec_history.jsonl").open("w") as history:
            epochs = tqdm(range(1, max_epochs + 1), desc=f"CLEAN EC {split}/fold_{fold}", unit="epoch", mininterval=1)
            for epoch in epochs:
                if epoch % adaptive_rate == 0:
                    current = encode(model, features, EC_EMBED_BATCH_SIZE)
                    neighbors, weights = neighbor_pool(centers_from_embeddings(current, members), HARD_NEGATIVE_K)
                    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
                a, p, n = sample_triplets(members, labels, vocabulary, neighbors, weights, rng, np_rng)
                model.train()
                loss_total, seen = 0.0, 0
                for start in range(0, len(a), EC_BATCH_SIZE):
                    ai, pi, ni = a[start:start + EC_BATCH_SIZE], p[start:start + EC_BATCH_SIZE], n[start:start + EC_BATCH_SIZE]
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(features[ai]), model(features[pi]), model(features[ni]))
                    loss.backward()
                    optimizer.step()
                    loss_total += float(loss.detach()) * len(ai)
                    seen += len(ai)
                mean_loss = loss_total / seen
                history.write(json.dumps({"epoch": epoch, "train_triplet_loss": mean_loss}) + "\n")
                history.flush()
                epochs.set_postfix(loss=f"{mean_loss:.4f}")
        np.save(dest / "ec_centers.npy", centers_from_embeddings(encode(model, features, EC_EMBED_BATCH_SIZE), members).cpu().numpy())
        (dest / "ec_vocabulary.json").write_text(json.dumps(vocabulary, indent=2))
        _save_model(dest, "ec", model, ec_config, max_epochs)
    if all((dest / f"{name}_training_done.json").exists() for name in ("binary", "ec")):
        config = {"model_policy": MODEL_POLICY, "binary": json.loads((dest / "binary_config.json").read_text()),
                  "ec": json.loads((dest / "ec_config.json").read_text())}
        (dest / "config.json").write_text(json.dumps(config, indent=2))
        (dest / "training_done.json").write_text(json.dumps({"model_policy": MODEL_POLICY,
            "binary_completed_epochs": config["binary"]["max_epochs"],
            "ec_completed_epochs": config["ec"]["max_epochs"], "checkpoint_policy": "final"}, indent=2))
    return dest
