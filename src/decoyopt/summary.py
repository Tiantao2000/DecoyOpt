"""Five-fold mean and sample standard deviation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .data import FOLDS, SPLITS
from .evaluate import EC_PREDICTION_POLICY


def flatten(metrics):
    result = {}
    selections = {
        "binary": ("auroc", "overall_ap", "macro_f1", "mcc", "overall_cen"),
        "decoy": ("fpr_u", "fpr_d0", "fpr_d5", "fpr_d10"),
    }
    for scope, names in selections.items():
        result.update({f"{scope}.{name}": metrics[scope][name] for name in names})
    for scope in ("ec_all", "ec_end_to_end", "ec_direct"):
        for level in range(1, 5):
            for name in ("weighted_precision", "weighted_recall", "weighted_f1", "macro_f1", "overall_mcc"):
                result[f"{scope}.ec{level}.{name}"] = metrics[scope][str(level)][name]
    return result


def write_summary(results_dir: Path):
    outputs = []
    for split in SPLITS:
        paths = [results_dir / str(split) / f"fold_{fold}" / "metrics.json" for fold in FOLDS]
        output = results_dir / str(split) / "five_fold_summary.csv"
        if not all(path.is_file() for path in paths):
            output.unlink(missing_ok=True)
            continue
        metrics = [json.loads(path.read_text()) for path in paths]
        if any(item.get("ec_prediction_policy") != EC_PREDICTION_POLICY for item in metrics):
            output.unlink(missing_ok=True)
            continue
        folds = [flatten(item) for item in metrics]
        with output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["split", "metric", "n_folds", "mean", "sd", "mean_pm_sd"])
            for metric in folds[0]:
                values = np.asarray([item[metric] for item in folds])
                mean, sd = float(values.mean()), float(values.std(ddof=1))
                writer.writerow([split, metric, 5, mean, sd, f"{mean:.4f} ± {sd:.4f}"])
        outputs.append(output)
        print(output)
    return outputs
