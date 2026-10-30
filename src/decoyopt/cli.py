"""Command-line entry point for the ProtT5-only workflow."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import DATA, FEATURE_MATRICES, FOLDS, ROOT, SPLITS
from .training_defaults import BINARY_EPOCHS, MAX_EPOCHS


def main():
    parser = argparse.ArgumentParser(prog="decoyopt")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("cache", "train", "test", "summary"):
        p = commands.add_parser(command)
        if command != "summary":
            p.add_argument("--data-dir", type=Path, default=DATA)
            p.add_argument("--cache-dir", type=Path, default=FEATURE_MATRICES)
            p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
        if command != "cache":
            p.add_argument("--results-dir", type=Path, default=ROOT / "results")
        if command in ("train", "test"):
            p.add_argument("--splits", type=int, nargs="+", choices=SPLITS, default=list(SPLITS))
            p.add_argument("--folds", type=int, nargs="+", choices=FOLDS, default=list(FOLDS))
        if command == "cache":
            p.add_argument("--model", default="Rostlab/prot_t5_xl_uniref50", help="Hugging Face model ID or local model directory")
            p.add_argument("--hf-cache", type=Path)
            p.add_argument("--batch-size", type=int, default=8)
            p.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
        elif command == "train":
            p.add_argument("--binary-epochs", type=int, default=BINARY_EPOCHS)
            p.add_argument("--ec-epochs", type=int, default=MAX_EPOCHS)
            p.add_argument("--overwrite", action="store_true")
        elif command == "test":
            p.add_argument("--checkpoint-dir", type=Path, help="Root containing <split>/fold_<n>; defaults to --results-dir")
    args = parser.parse_args()
    if args.command == "cache":
        from .cache import generate_cache
        generate_cache(args.data_dir, args.cache_dir, args.device, args.model,
                       args.batch_size, args.dtype, args.hf_cache)
    elif args.command == "summary":
        from .summary import write_summary
        if not write_summary(args.results_dir):
            parser.error("No complete five-fold results found")
    else:
        from .evaluate import evaluate
        from .summary import write_summary
        from .train import train
        if len(set(args.splits)) != len(args.splits) or len(set(args.folds)) != len(args.folds):
            parser.error("Splits and folds must be unique")
        for split in args.splits:
            for fold in args.folds:
                print(f"{args.command}: split={split}, fold={fold}", flush=True)
                if args.command == "train":
                    train(split, fold, args.device, binary_epochs=args.binary_epochs,
                          max_epochs=args.ec_epochs, results_dir=args.results_dir,
                          overwrite=args.overwrite, data_dir=args.data_dir, cache_dir=args.cache_dir)
                else:
                    evaluate(split, fold, args.device, args.results_dir,
                             args.data_dir, args.cache_dir, args.checkpoint_dir)
        if args.command == "test":
            write_summary(args.results_dir)
