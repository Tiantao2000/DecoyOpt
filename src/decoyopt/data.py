"""Dataset indexing and read-only access to prepared PLM feature matrices."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FEATURE_MATRICES = ROOT / "cache"
PLM = "ProtT5-XL-UniRef50"
SPLITS = (40, 60, 80)
FOLDS = (1, 2, 3, 4, 5)
LEVELS = ("0", "5", "10")


def labels_of(value: str) -> frozenset[str]:
    return frozenset(x.strip() for x in value.split(";") if x.strip())


class Dataset:
    def __init__(self, split: int, fold: int, data_dir: Path = DATA):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if fold not in FOLDS:
            raise ValueError(f"fold must be one of {FOLDS}")
        self.split = split
        self.fold = fold
        path = data_dir / "five_fold" / f"graphpart_{split}" / f"fold_{fold}.csv"
        self.ids: list[str] = []
        self.labels: list[frozenset[str]] = []
        self.partition: list[str] = []
        self.source: list[str] = []
        self.index: dict[str, int] = {}
        self.parents: dict[str, list[int]] = {}
        self.positives: dict[str, list[int]] = defaultdict(list)
        self.unmarked: dict[str, list[int]] = defaultdict(list)
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                sample_id = row["enzyme_id"]
                part = row["split"]
                if part not in ("train", "val", "test"):
                    raise ValueError(f"Invalid partition {part!r} for {sample_id}")
                positive = row["label"] == "1"
                labels = labels_of(row["EC number"]) if positive else frozenset()
                if positive and not labels:
                    raise ValueError(f"Positive {sample_id} has no EC")
                idx = self._add(sample_id, labels, part, "P" if positive else "U")
                (self.positives if positive else self.unmarked)[part].append(idx)
                if positive:
                    self.parents[sample_id] = [idx]
        # Decoys inherit the parent partition. Their other columns are metadata only.
        seen = set()
        self.excluded_decoys = 0
        with (data_dir / "decoys.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                parent = row["parent_enzyme_id"]
                if parent not in self.parents:
                    # A split may omit a parent present in the shared decoy CSV.
                    self.excluded_decoys += 1
                    continue
                category = row["decoy_category"]
                level = next((x for x in LEVELS if category.endswith(f"_{x}A")), None)
                if level is None:
                    raise ValueError(f"Unknown decoy category {category}")
                key = (parent, level)
                if key in seen:
                    raise ValueError(f"Duplicate decoy {key}")
                seen.add(key)
                part = self.partition[self.parents[parent][0]]
                idx = self._add(row["enzyme_id"], frozenset(), part, f"D{level}")
                self.parents[parent].append(idx)
        for parent, group in self.parents.items():
            if len(group) != 4 or [self.source[i] for i in group] != ["P", "D0", "D5", "D10"]:
                # CSV order is expected but does not define the intended ordering.
                d = {self.source[i]: i for i in group}
                if set(d) != {"P", "D0", "D5", "D10"}:
                    raise ValueError(f"Incomplete parent group {parent}")
                self.parents[parent] = [d[x] for x in ("P", "D0", "D5", "D10")]
        self.groups = {part: np.asarray([group for group in self.parents.values()
                                          if self.partition[group[0]] == part], dtype=np.int32)
                       for part in ("train", "val", "test")}
        self.positives = {part: np.asarray(self.positives[part], dtype=np.int32)
                          for part in ("train", "val", "test")}
        self.unmarked = {part: np.asarray(self.unmarked[part], dtype=np.int32)
                         for part in ("train", "val", "test")}
        self.ec_to_train: dict[str, list[int]] = defaultdict(list)
        for i in self.positives["train"]:
            for ec in self.labels[i]:
                self.ec_to_train[ec].append(int(i))
        self.ecs = sorted(self.ec_to_train)

    def _add(self, sample_id: str, labels: frozenset[str], part: str, source: str) -> int:
        if sample_id in self.index:
            raise ValueError(f"Duplicate sample ID: {sample_id}")
        idx = len(self.ids)
        self.index[sample_id] = idx
        self.ids.append(sample_id)
        self.labels.append(labels)
        self.partition.append(part)
        self.source.append(source)
        return idx

    def indices(self, part: str) -> np.ndarray:
        groups = self.groups[part].reshape(-1)
        return np.concatenate((groups, self.unmarked[part]))


class FeatureStore:
    """Read a prepared shared PLM matrix without modifying or building it."""

    def __init__(self, dataset: Dataset, feature_dir: Path = FEATURE_MATRICES):
        self.dataset = dataset
        if (feature_dir / "completed.npy").exists():
            raise ValueError("Feature generation is incomplete; finish decoyopt cache first")
        manifest = feature_dir / "shared_ids.txt"
        if not manifest.is_file():
            raise FileNotFoundError(f"Feature manifest is missing: {manifest}; run decoyopt cache first")
        self.cache_ids = manifest.read_text().splitlines()
        if not self.cache_ids:
            raise ValueError(f"Feature manifest is empty: {manifest}")
        self.data_path = feature_dir / f"{PLM}.f32"
        if not self.data_path.is_file():
            raise FileNotFoundError(f"Prepared feature matrix is missing for {PLM}; run decoyopt cache first")
        row_bytes = len(self.cache_ids) * 4
        size = self.data_path.stat().st_size
        if size < row_bytes or size % row_bytes:
            raise ValueError(f"Incomplete feature matrix: {self.data_path}")
        self.dim = size // row_bytes
        if self.dim != 1024 or len(set(self.cache_ids)) != len(self.cache_ids):
            raise ValueError("Expected unique IDs and 1024-dimensional ProtT5 features")
        self.matrix = np.memmap(self.data_path, dtype=np.float32, mode="r",
                                shape=(len(self.cache_ids), self.dim))
        if dataset.ids == self.cache_ids:
            self.row_index = None
        else:
            lookup = {sample_id: i for i, sample_id in enumerate(self.cache_ids)}
            self.row_index = np.fromiter((lookup.get(sample_id, -1) for sample_id in dataset.ids),
                                         dtype=np.int32, count=len(dataset.ids))
            missing = np.flatnonzero(self.row_index < 0)
            if len(missing):
                examples = [dataset.ids[int(i)] for i in missing[:3]]
                raise ValueError(f"{len(missing)} IDs are absent from the prepared feature manifest: {examples}")

    def raw_batch(self, indices: np.ndarray | list[int]) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if self.row_index is None:
            return np.asarray(self.matrix[indices], dtype=np.float32)
        return np.asarray(self.matrix[self.row_index[indices]], dtype=np.float32)

    def batch(self, indices: np.ndarray | list[int], device: torch.device) -> torch.Tensor:
        return torch.from_numpy(self.raw_batch(indices)).to(device=device, non_blocking=True)
