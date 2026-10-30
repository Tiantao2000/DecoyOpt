"""Binary, decoy, ranking, and four-level multi-label EC metrics."""

from __future__ import annotations

from collections import Counter

import numpy as np
from pycm import ConfusionMatrix
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score, f1_score

NON_ENZYME_LABEL = "non_enzyme"


def ec_prefixes(labels: set[str] | frozenset[str], level: int,
                include_non_enzyme: bool = False) -> set[str]:
    prefixes = {".".join(ec.split(".")[:level]) for ec in labels
                if ec != NON_ENZYME_LABEL and len(ec.split(".")) >= level}
    if include_non_enzyme and NON_ENZYME_LABEL in labels:
        if len(labels) != 1:
            raise ValueError("non_enzyme must be mutually exclusive with EC labels")
        prefixes.add(NON_ENZYME_LABEL)
    return prefixes


def multilabel_metrics(true: list[set[str]], pred: list[set[str]], level: int,
                       include_non_enzyme: bool = False) -> dict:
    """Five metrics over EC labels, optionally including an explicit non-enzyme class."""
    n = len(true)
    if n != len(pred) or n == 0 or level not in (1, 2, 3, 4):
        raise ValueError("Invalid EC evaluation arrays")
    tp, fp, fn, support = Counter(), Counter(), Counter(), Counter()
    for a, b in zip(true, pred):
        a = ec_prefixes(a, level, include_non_enzyme)
        b = ec_prefixes(b, level, include_non_enzyme)
        support.update(a)
        tp.update(a & b)
        fp.update(b - a)
        fn.update(a - b)
    classes = set(support) | set(fp)
    if include_non_enzyme:
        classes.add(NON_ENZYME_LABEL)
    p, r, f = [], [], []
    weights = []
    for ec in sorted(classes):
        t, x, y = tp[ec], fp[ec], fn[ec]
        pc = t / (t + x) if t + x else 0.0
        rc = t / (t + y) if t + y else 0.0
        fc = 2 * t / (2 * t + x + y) if 2 * t + x + y else 0.0
        p.append(pc); r.append(rc); f.append(fc)
        weights.append(support[ec])
    weights = np.asarray(weights)
    weight_sum = int(weights.sum())
    T, Fp, Fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    # Overall MCC is the binary MCC of all N x K label indicators.
    tn = n * len(classes) - T - Fp - Fn
    den = (T + Fp) * (T + Fn) * (tn + Fp) * (tn + Fn)
    overall_mcc = (T * tn - Fp * Fn) / np.sqrt(float(den)) if den > 0 else 0.0
    return {
        "weighted_precision": float(np.average(p, weights=weights)) if weight_sum else 0.0,
        "weighted_recall": float(np.average(r, weights=weights)) if weight_sum else 0.0,
        "weighted_f1": float(np.average(f, weights=weights)) if weight_sum else 0.0,
        "macro_f1": float(np.mean(f)) if f else 0.0,
        "overall_mcc": float(overall_mcc),
        "n_samples": n,
        "n_ec_classes": len(classes - {NON_ENZYME_LABEL}),
        "n_classes": len(classes),
        "n_true_assignments": weight_sum,
        "n_true_positives": T,
        "n_false_positives": Fp,
        "n_false_negatives": Fn,
        "n_true_negatives": tn,
    }


def binary_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    pred = (scores >= threshold).astype(np.int8)
    cm = ConfusionMatrix(actual_vector=y.tolist(), predict_vector=pred.tolist(), classes=[0, 1])
    return {
        "overall_ap": float(average_precision_score(y, scores)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "auroc": float(roc_auc_score(y, scores)),
        "overall_cen": float(cm.overall_stat["Overall CEN"]),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "enzyme_recall": float(((pred == 1) & (y == 1)).sum() / (y == 1).sum()),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "threshold_logit": float(threshold),
    }


def choose_mcc_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Exact sweep over validation logits; higher thresholds win tied MCC."""
    order = np.argsort(-scores, kind="stable")
    sorted_scores = np.asarray(scores)[order]
    sorted_y = np.asarray(y)[order]
    tp = np.cumsum(sorted_y == 1)
    fp = np.cumsum(sorted_y == 0)
    positives = int(tp[-1]); negatives = int(fp[-1])
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(y) - 1]
    t, f = tp[ends].astype(float), fp[ends].astype(float)
    den = (t + f) * positives * negatives * (positives + negatives - t - f)
    mcc = np.divide(t * (negatives - f) - f * (positives - t), np.sqrt(den),
                    out=np.zeros_like(t), where=den > 0)
    return float(sorted_scores[ends[int(np.argmax(mcc))]])


def decoy_metrics(groups: np.ndarray, source_indices: dict[str, np.ndarray],
                  all_logits: np.ndarray, threshold: float) -> dict:
    out = {}
    for source in ("D0", "D5", "D10", "U"):
        indices = source_indices[source]
        out[f"fpr_{source.lower()}"] = float(np.mean(all_logits[indices] >= threshold))
    all_decoys = np.concatenate([source_indices[s] for s in ("D0", "D5", "D10")])
    out["fpr_decoy_all"] = float(np.mean(all_logits[all_decoys] >= threshold))
    s = all_logits[groups]
    out["strict_order_accuracy"] = float(np.mean(np.all(s[:, :-1] > s[:, 1:], axis=1)))
    out["adjacent_order_accuracy"] = [float(np.mean(s[:, j] > s[:, j + 1])) for j in range(3)]
    return out
