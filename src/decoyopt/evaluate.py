"""Validation-MCC calibration and held-out binary, FPR and EC evaluation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .data import DATA, FEATURE_MATRICES, ROOT, Dataset, FeatureStore
from .metrics import (NON_ENZYME_LABEL, binary_metrics, choose_mcc_threshold,
                      decoy_metrics, multilabel_metrics)
from .model import MODEL_POLICY, BinaryModel, ECModel

EC_PREDICTION_POLICY = "independent_clean_all_test_non_enzyme_and_positive_ec_v3"


def maxsep_count(distances: np.ndarray) -> int:
    """CLEAN maximum separation with first_grad=True and use_max_grad=False."""
    d = np.asarray(distances)
    if len(d) == 1:
        return 1
    gamma = np.concatenate((d[1:], np.repeat(d[-1], 10)))
    gradients = np.abs(np.diff(np.abs(d - gamma.mean())))
    large = np.flatnonzero(gradients > gradients.mean())
    idx = int(large[0]) if len(large) else 0
    return 1 if idx >= 5 else idx + 1


def ec_candidates(embeds, centers, ecs, device, batch_size=512):
    center = torch.from_numpy(centers).to(device)
    center_norm = (center * center).sum(1)
    result = []
    for start in range(0, len(embeds), batch_size):
        block = torch.from_numpy(embeds[start:start + batch_size]).to(device)
        sq = (block * block).sum(1, keepdim=True) + center_norm[None, :] - 2 * block @ center.T
        distances = sq.clamp_min_(0).sqrt_()
        d, ix = torch.topk(distances, k=min(10, len(ecs)), largest=False, sorted=True)
        for row_d, row_ix in zip(d.cpu().numpy(), ix.cpu().numpy()):
            result.append({ecs[int(k)] for k in row_ix[:maxsep_count(row_d)]})
    return result


def load_models(result_dir: Path, device, split: int, fold: int):
    models = []
    for name, cls in (("binary", BinaryModel), ("ec", ECModel)):
        checkpoint = torch.load(result_dir / f"{name}_checkpoint.pt", map_location=device, weights_only=True)
        config = checkpoint["config"]
        if (config.get("model_policy") != MODEL_POLICY or config.get("input_dim") != 1024
                or config.get("split") != split or config.get("fold") != fold
                or config.get("plm") != "ProtT5-XL-UniRef50"):
            raise ValueError(f"Incompatible {name} checkpoint for split={split}, fold={fold}")
        model = cls(1024).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        models.append(model.eval())
    return models


@torch.no_grad()
def predict(data, store, binary, ec, centers, ecs, indices, device, threshold=None):
    logits = np.full(len(data.ids), np.nan, dtype=np.float32)
    candidates = [set() for _ in indices]
    embeds, selected_rows = [], []
    for start in tqdm(range(0, len(indices), 2048), desc="Inference", unit="batch", leave=False):
        batch = indices[start:start + 2048]
        features = store.batch(batch, device)
        scores = binary(features).cpu().numpy()
        logits[batch] = scores
        if threshold is not None:
            mask = np.asarray([data.source[int(i)] == "P" for i in batch]) | (scores >= threshold)
            if mask.any():
                embeds.append(ec(features[torch.from_numpy(mask).to(device)]).cpu().numpy())
                selected_rows.extend(start + np.flatnonzero(mask))
    if embeds:
        predictions = ec_candidates(np.concatenate(embeds), centers, ecs, device)
        for row, labels in zip(selected_rows, predictions):
            candidates[row] = labels
    return logits, candidates


def evaluate(split: int, fold: int, device_name: str, results_dir: Path = ROOT / "results",
             data_dir: Path = DATA, cache_dir: Path = FEATURE_MATRICES,
             checkpoint_dir: Path | None = None):
    data = Dataset(split, fold, data_dir)
    store = FeatureStore(data, cache_dir)
    device = torch.device(device_name)
    source = (checkpoint_dir or results_dir) / str(split) / f"fold_{fold}"
    dest = results_dir / str(split) / f"fold_{fold}"
    binary, ec = load_models(source, device, split, fold)
    centers = np.load(source / "ec_centers.npy")
    ecs = json.loads((source / "ec_vocabulary.json").read_text())
    if centers.shape != (len(ecs), 128) or not len(ecs) or not np.isfinite(centers).all():
        raise ValueError("Invalid EC centers or vocabulary")
    val_idx = data.indices("val")
    val_logits, _ = predict(data, store, binary, ec, centers, ecs, val_idx, device)
    val_y = np.asarray([int(data.source[int(i)] == "P") for i in val_idx])
    threshold = choose_mcc_threshold(val_y, val_logits[val_idx])
    test_idx = data.indices("test")
    logits, direct = predict(data, store, binary, ec, centers, ecs, test_idx, device, threshold)
    sources = np.asarray([data.source[int(i)] for i in test_idx])
    accepted = logits[test_idx] >= threshold
    non_enzyme = {NON_ENZYME_LABEL}
    all_truth = [data.labels[int(i)] if s == "P" else non_enzyme for i, s in zip(test_idx, sources)]
    all_pred = [p if keep else non_enzyme for p, keep in zip(direct, accepted)]
    if any(not labels for labels in all_pred):
        raise ValueError("Accepted proteins must have EC predictions")
    pos_rows = np.flatnonzero(sources == "P")
    truth = [data.labels[int(test_idx[row])] for row in pos_rows]
    gated = [direct[row] if accepted[row] else set() for row in pos_rows]
    by_source = {s: test_idx[sources == s] for s in ("P", "U", "D0", "D5", "D10")}
    metrics = {
        "binary": binary_metrics((sources == "P").astype(np.int8), logits[test_idx], threshold),
        "decoy": decoy_metrics(data.groups["test"], by_source, logits, threshold),
        "thresholds": {"binary_mcc_logit": threshold},
        "ec_all": {str(k): multilabel_metrics(all_truth, all_pred, k, True) for k in range(1, 5)},
        "ec_end_to_end": {str(k): multilabel_metrics(truth, gated, k) for k in range(1, 5)},
        "ec_direct": {str(k): multilabel_metrics(truth, [direct[r] for r in pos_rows], k) for k in range(1, 5)},
        "ec_gate": {"threshold_logit": threshold, "n_positive": len(pos_rows),
                    "n_accepted": int(accepted[pos_rows].sum()), "n_rejected": int((~accepted[pos_rows]).sum())},
        "ec_prediction_policy": EC_PREDICTION_POLICY, "model_policy": MODEL_POLICY,
        "threshold_policy": "Maximum validation MCC; frozen before test inference.",
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    (dest / "thresholds.json").write_text(json.dumps(metrics["thresholds"], indent=2))
    with (dest / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["enzyme_id", "source", "true_ec", "binary_logit", "binary_at_mcc_threshold",
                         "predicted_ec_direct", "predicted_ec_end_to_end", "true_ec_all", "predicted_ec_all"])
        for row, idx in enumerate(test_idx):
            writer.writerow([data.ids[idx], sources[row], ";".join(sorted(data.labels[idx])),
                             float(logits[idx]), int(accepted[row]),
                             ";".join(sorted(direct[row])) if sources[row] == "P" else "",
                             ";".join(sorted(direct[row])) if accepted[row] else "",
                             ";".join(sorted(all_truth[row])), ";".join(sorted(all_pred[row]))])
    print(dest / "metrics.json", flush=True)
    return dest
