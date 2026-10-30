"""Generate one shared 1024-dimensional ProtT5 feature matrix."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .data import DATA, FEATURE_MATRICES, PLM

MODEL_ID = "Rostlab/prot_t5_xl_uniref50"


def read_sequences(path: Path):
    records, seen = [], set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = row["enzyme_id"]
            sequence = re.sub(r"\s+", "", row["enzyme_sequence"]).upper()
            if not sample_id or sample_id in seen or not sequence:
                raise ValueError(f"Duplicate ID or empty sequence: {sample_id}")
            seen.add(sample_id)
            records.append((sample_id, sequence))
    if not records:
        raise ValueError("The sequence table is empty")
    return records


def embed_batch(records, tokenizer, model, device, max_tokens=1024):
    # Match the original feature producer, including U/Z/O/B/J normalization,
    # 1024-token truncation and reduction in the encoder's inference dtype.
    formatted = [" ".join(re.sub(r"[UZOBJ]", "X", seq)) for _, seq in records]
    encoded = tokenizer(formatted, add_special_tokens=True, padding=True,
                        truncation=True, max_length=max_tokens, return_tensors="pt",
                        return_special_tokens_mask=True)
    special = encoded.pop("special_tokens_mask")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].bool() & ~special.to(device).bool()
        counts = mask.sum(1)
        if torch.any(counts == 0):
            raise ValueError("Sequence has no residue tokens")
        pooled = ((hidden * mask.unsqueeze(-1)).sum(1) / counts.unsqueeze(-1)).float().cpu().numpy()
    if pooled.shape != (len(records), 1024) or not np.isfinite(pooled).all():
        raise ValueError("Invalid ProtT5 features")
    return pooled


def generate_cache(data_dir: Path = DATA, cache_dir: Path = FEATURE_MATRICES,
                   device_name="cuda:0", model_path=MODEL_ID, batch_size=8,
                   dtype_name="auto", hf_cache: Path | None = None):
    from transformers import T5EncoderModel, T5Tokenizer

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = torch.device(device_name)
    if dtype_name == "auto":
        dtype_name = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if device.type == "cuda" else "float32"
    dtype = getattr(torch, dtype_name)
    source = data_dir / "sequences.csv"
    records = read_sequences(source)
    ids = [sample_id for sample_id, _ in records]
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = {"model": str(model_path), "plm": PLM, "dtype": dtype_name,
                "sequence_sha256": fingerprint, "n_sequences": len(ids), "dim": 1024,
                "max_tokens": 1024, "normalization": "uppercase; UZOBJ -> X",
                "pooling": "mean over residues excluding special tokens and padding"}
    cache_dir.mkdir(parents=True, exist_ok=True)
    ids_path, meta_path = cache_dir / "shared_ids.txt", cache_dir / "metadata.json"
    matrix_path = cache_dir / f"{PLM}.f32"
    progress_path = cache_dir / "completed.npy"
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != metadata or ids_path.read_text().splitlines() != ids:
            raise ValueError("Cache settings or sequences changed; choose another --cache-dir")
        if matrix_path.stat().st_size != len(ids) * 1024 * 4:
            raise ValueError("Incomplete feature matrix; choose another --cache-dir")
        if not progress_path.exists():
            print(f"Cache complete: {matrix_path}")
            return
        completed = np.load(progress_path)
    else:
        if any(p.exists() for p in (ids_path, matrix_path, progress_path)):
            raise ValueError("Partial cache metadata; choose another --cache-dir")
        matrix = np.memmap(matrix_path, dtype=np.float32, mode="w+", shape=(len(ids), 1024))
        matrix.flush()
        del matrix
        ids_path.write_text("\n".join(ids) + "\n")
        completed = np.zeros(len(ids), dtype=bool)
        np.save(progress_path, completed)
        meta_path.write_text(json.dumps(metadata, indent=2))
    matrix = np.memmap(matrix_path, dtype=np.float32, mode="r+", shape=(len(ids), 1024))
    pending = sorted(np.flatnonzero(~completed), key=lambda i: len(records[i][1]))
    if pending:
        model_args = {"cache_dir": str(hf_cache or cache_dir / "models")}
        tokenizer = T5Tokenizer.from_pretrained(str(model_path), do_lower_case=False, **model_args)
        model = T5EncoderModel.from_pretrained(str(model_path), torch_dtype=dtype,
                                               **model_args).eval().to(device)
        for start in tqdm(range(0, len(pending), batch_size), desc="ProtT5", unit="batch"):
            rows = pending[start:start + batch_size]
            matrix[rows] = embed_batch([records[i] for i in rows], tokenizer, model, device)
            matrix.flush()
            completed[rows] = True
            temporary = progress_path.with_suffix(".tmp.npy")
            np.save(temporary, completed)
            temporary.replace(progress_path)
    del matrix
    progress_path.unlink()
    print(f"Cache complete: {matrix_path}")
