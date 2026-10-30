"""Build the data and pretrained-checkpoint downloads from a CLEAN_OPT project."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path

SPLITS = (40, 60, 80)
PLM = "ProtT5-XL-UniRef50"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def package(source: Path, output: Path):
    original = source / "EC_number_prediction" / "clean_data"
    data, checkpoints = output / "data", output / "checkpoints"
    if data.exists() or checkpoints.exists():
        raise ValueError("Choose an output without existing data/ and checkpoints/ directories")
    sequences, folds, source_hashes = {}, {}, {}
    for split in SPLITS:
        fixed_test, fixed_members = None, None
        for fold in range(1, 6):
            path = original / "five_fold" / f"graphpart_{split}" / f"fold_{fold}.csv"
            source_hashes[str(path.relative_to(source))] = sha256(path)
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            current = {r["enzyme_id"] for r in rows}
            if len(current) != len(rows):
                raise ValueError(f"Duplicate IDs in {path}")
            test = {r["enzyme_id"] for r in rows if r["split"] == "test"}
            if fixed_test is not None and (fixed_test != test or fixed_members != current):
                raise ValueError(f"Fold membership or fixed test set changed in {path}")
            fixed_test, fixed_members = test, current
            counts = Counter((r["split"], r["label"]) for r in rows)
            folds[f"{split}/fold_{fold}"] = {f"{p}.{'P' if y == '1' else 'U'}": n
                                              for (p, y), n in sorted(counts.items())}
            for row in rows:
                sample_id, sequence = row["enzyme_id"], row["enzyme_sequence"]
                if sequences.setdefault(sample_id, sequence) != sequence:
                    raise ValueError(f"Conflicting sequences for {sample_id}")
            write_csv(data / "five_fold" / f"graphpart_{split}" / f"fold_{fold}.csv",
                      ["enzyme_id", "label", "EC number", "split"], rows)
    decoys_path = original / "swissprot_2026_03_enzyme_esm3_decoys.csv"
    source_hashes[str(decoys_path.relative_to(source))] = sha256(decoys_path)
    with decoys_path.open(newline="") as handle:
        decoys = list(csv.DictReader(handle))
    for row in decoys:
        sample_id, sequence = row["enzyme_id"], row["enzyme_sequence"]
        if sample_id in sequences:
            raise ValueError(f"Duplicate decoy ID {sample_id}")
        sequences[sample_id] = sequence
    write_csv(data / "decoys.csv", ["enzyme_id", "parent_enzyme_id", "decoy_category"], decoys)
    write_csv(data / "sequences.csv", ["enzyme_id", "enzyme_sequence"],
              ({"enzyme_id": i, "enzyme_sequence": s} for i, s in sequences.items()))
    manifest = {"n_sequences": len(sequences), "n_decoys": len(decoys), "fold_counts": folds,
                "source_sha256": source_hashes,
                "files": {str(p.relative_to(data)): sha256(p) for p in sorted(data.rglob("*.csv"))}}
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2))
    filenames = ["binary_checkpoint.pt", "ec_checkpoint.pt", "ec_centers.npy", "ec_vocabulary.json",
                 "binary_config.json", "ec_config.json", "config.json", "thresholds.json",
                 "binary_training_done.json", "ec_training_done.json", "training_done.json"]
    checkpoint_sources = {}
    for split in SPLITS:
        for fold in range(1, 6):
            src = source / "CLEAN_OPT" / "results" / str(split) / PLM / f"fold_{fold}"
            dest = checkpoints / str(split) / f"fold_{fold}"
            dest.mkdir(parents=True)
            for filename in filenames:
                checkpoint_sources[str((src / filename).relative_to(source))] = sha256(src / filename)
                shutil.copy2(src / filename, dest / filename)
            shutil.copy2(src / "metrics.json", dest / "reference_metrics.json")
    (checkpoints / "manifest.json").write_text(json.dumps({
        "plm": PLM, "n_models": 30, "n_split_fold_pairs": 15,
        "source_sha256": checkpoint_sources,
        "files": {str(p.relative_to(checkpoints)): sha256(p) for p in sorted(checkpoints.rglob("*")) if p.is_file()},
    }, indent=2))
    dist = output / "dist"
    dist.mkdir(exist_ok=True)
    for directory, name in ((data, "decoyopt_data.tar.gz"), (checkpoints, "decoyopt_checkpoints.tar.gz")):
        archive = dist / name
        with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
            tar.add(directory, arcname=directory.name)
        print(f"{archive}: {archive.stat().st_size / 1024**2:.1f} MiB", flush=True)
    (dist / "SHA256SUMS.txt").write_text("".join(
        f"{sha256(p)}  {p.name}\n" for p in sorted(dist.glob("decoyopt_*.tar.gz"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original project containing CLEAN_OPT and EC_number_prediction")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    package(args.source, args.output)
