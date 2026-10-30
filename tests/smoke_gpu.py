"""Small real-sequence cache/train/test check; temporary artifacts are removed."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from decoyopt.cache import generate_cache
from decoyopt.data import Dataset, FeatureStore, PLM
from decoyopt.evaluate import load_models
from decoyopt.summary import write_summary
from decoyopt.train import train


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_cli(*args):
    subprocess.run([sys.executable, "-m", "decoyopt", *map(str, args)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "Use physical GPU 1"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    print(f"Physical GPU 1: {torch.cuda.get_device_name(0)}", flush=True)
    with tempfile.TemporaryDirectory(prefix=".smoke-", dir=root) as temporary:
        temp = Path(temporary)
        small = temp / "data"
        with (root / "data/five_fold/graphpart_40/fold_1.csv").open() as handle:
            original_rows = list(csv.DictReader(handle))
        rows = []
        for part in ("train", "val", "test"):
            # Distinct positive labels guarantee disjoint EC negatives.
            used_labels = set()
            for row in original_rows:
                if row["split"] == part and row["label"] == "1":
                    labels = set(row["EC number"].split(";"))
                    if labels.isdisjoint(used_labels):
                        rows.append(row)
                        used_labels.update(labels)
                        if sum(r["split"] == part and r["label"] == "1" for r in rows) == 4:
                            break
            rows.extend([r for r in original_rows if r["split"] == part and r["label"] == "0"][:2])
        parent_ids = {r["enzyme_id"] for r in rows if r["label"] == "1"}
        with (root / "data/decoys.csv").open() as handle:
            decoys = [r for r in csv.DictReader(handle) if r["parent_enzyme_id"] in parent_ids]
        keep = {r["enzyme_id"] for r in rows + decoys}
        with (root / "data/sequences.csv").open() as handle:
            seqs = [r for r in csv.DictReader(handle) if r["enzyme_id"] in keep]
        assert len(seqs) == len(keep) == 54
        write_rows(small / "five_fold/graphpart_40/fold_1.csv", rows)
        write_rows(small / "decoys.csv", decoys)
        write_rows(small / "sequences.csv", seqs)
        cache, results = temp / "cache", temp / "results"
        run_cli("cache", "--data-dir", small, "--cache-dir", cache, "--model", args.model,
                "--batch-size", 2, "--device", "cuda:0")
        digest = hashlib.sha256((cache / f"{PLM}.f32").read_bytes()).hexdigest()
        generate_cache(small, cache, "cuda:0", str(args.model), batch_size=2)
        assert digest == hashlib.sha256((cache / f"{PLM}.f32").read_bytes()).hexdigest()
        run_cli("train", "--data-dir", small, "--cache-dir", cache, "--results-dir", results,
                "--splits", 40, "--folds", 1, "--binary-epochs", 2, "--ec-epochs", 2, "--device", "cuda:0")
        train(40, 1, "cuda:0", max_epochs=2, binary_epochs=2, adaptive_rate=1,
              results_dir=temp / "refresh", data_dir=small, cache_dir=cache)
        run_cli("test", "--data-dir", small, "--cache-dir", cache, "--results-dir", results,
                "--splits", 40, "--folds", 1, "--device", "cuda:0")
        pretrained = temp / "pretrained"
        run_cli("test", "--data-dir", small, "--cache-dir", cache, "--results-dir", pretrained,
                "--checkpoint-dir", root / "checkpoints", "--splits", 40, "--folds", 1, "--device", "cuda:0")
        metrics = json.loads((pretrained / "40/fold_1/metrics.json").read_text())
        assert metrics["binary"]["n_positive"] == 4 and metrics["binary"]["n_negative"] == 14
        assert metrics["ec_all"]["4"]["n_samples"] == 18
        assert metrics["ec_end_to_end"]["4"]["n_samples"] == 4
        # Check compatibility of all 30 supplied state dictionaries on the GPU.
        data = Dataset(40, 1, small)
        store = FeatureStore(data, cache)
        vector = store.batch([0], torch.device("cuda:0"))
        for split in (40, 60, 80):
            for fold in range(1, 6):
                models = load_models(root / "checkpoints" / str(split) / f"fold_{fold}", "cuda:0", split, fold)
                with torch.no_grad():
                    assert all(torch.isfinite(m(vector)).all() for m in models)
        # Compare the compact evaluator with the original on the same features.
        sys.path.insert(0, str(args.source / "CLEAN_OPT"))
        original_data = importlib.import_module("data")
        original_test = importlib.import_module("test")
        original_data.DATA = small
        original_data.FEATURES = temp / "registry"
        (original_data.FEATURES / PLM).mkdir(parents=True)
        (small / "swissprot_2026_03_enzyme_esm3_decoys.csv").write_bytes((small / "decoys.csv").read_bytes())
        original = original_data.Dataset(40, 1)
        original_store = original_data.FeatureStore(original, PLM, cache)
        assert original.ids == data.ids and original.labels == data.labels
        binary, ec = load_models(root / "checkpoints/40/fold_1", "cuda:0", 40, 1)
        cp = root / "checkpoints/40/fold_1"
        centers, vocabulary = np.load(cp / "ec_centers.npy"), json.loads((cp / "ec_vocabulary.json").read_text())
        from decoyopt.evaluate import predict
        from decoyopt.metrics import choose_mcc_threshold
        from distributed import setup
        group = setup("cuda:0")
        val = original_test.distributed_predict_partition(original, original_store, binary, ec, centers, vocabulary, "val", group)
        val_idx, val_logits, _ = val
        threshold = choose_mcc_threshold(np.array([int(data.source[i] == "P") for i in val_idx]), val_logits[val_idx])
        test_idx, logits, candidates = original_test.distributed_predict_partition(
            original, original_store, binary, ec, centers, vocabulary, "test", group, binary_threshold=threshold)
        new_logits, new_candidates = predict(data, store, binary, ec, centers, vocabulary, test_idx, torch.device("cuda:0"), threshold)
        np.testing.assert_array_equal(logits[test_idx], new_logits[test_idx])
        assert [{ec for ec, _ in c} if c is not None else set() for c in candidates] == new_candidates
        assert threshold == metrics["thresholds"]["binary_mcc_logit"]
        # Exercise the five-fold aggregator using the same real smoke result.
        for fold in range(2, 6):
            dest = pretrained / "40" / f"fold_{fold}"
            dest.mkdir()
            (dest / "metrics.json").write_text(json.dumps(metrics))
        summaries = write_summary(pretrained)
        assert len(summaries) == 1
        with summaries[0].open() as handle:
            assert all(float(r["sd"]) < 1e-12 for r in csv.DictReader(handle))
        print(json.dumps({"n_sequences": 54, "train_epochs_per_model": 2,
                          "checkpoint_pairs_loaded": 15, "original_inference_parity": "exact",
                          "smoke_metrics": metrics["binary"]}, indent=2), flush=True)
    print("Temporary smoke data, caches and outputs removed.", flush=True)


if __name__ == "__main__":
    main()
