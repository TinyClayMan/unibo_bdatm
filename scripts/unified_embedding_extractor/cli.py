"""Argument parsing and the top-level main() dispatcher."""

from __future__ import annotations

import argparse
import os

import torch

from .runners import run_all, run_experimental
from .utils import ensure_dir, log, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["all", "experimental"])
    p.add_argument("--encoder", required=True, choices=["gearnet", "esm_gearnet"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--esm_weight_dir", default=None)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max_length",
        type=int,
        default=0,
        help="truncate residues per protein; 0 = no truncation",
    )
    p.add_argument("--save_pt", action="store_true")
    p.add_argument("--save_npy", action="store_true")

    # mode = all
    p.add_argument("--pdb_root")
    p.add_argument("--train_terms_tsv")
    p.add_argument("--train_taxonomy_tsv")
    p.add_argument("--bpo_top_labels", type=int, default=1100)
    p.add_argument("--mfo_top_labels", type=int, default=450)
    p.add_argument("--cco_top_labels", type=int, default=300)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--test_frac", type=float, default=0.10)
    p.add_argument("--splits", default="train,valid,test")

    # mode = experimental
    p.add_argument("--reference_embed_dir")
    p.add_argument("--download_root")
    p.add_argument("--valid_manifest")
    p.add_argument("--test_manifest")
    p.add_argument("--valid_pt")
    p.add_argument("--test_pt")
    return p.parse_args()


def validate(args) -> None:
    if not args.save_pt and not args.save_npy:
        args.save_pt = True
    if args.encoder == "esm_gearnet" and not args.esm_weight_dir:
        raise ValueError("--encoder esm_gearnet requires --esm_weight_dir")
    if args.mode == "all":
        for k in ("pdb_root", "train_terms_tsv"):
            if not getattr(args, k):
                raise ValueError(f"--mode all requires --{k}")
    elif args.mode == "experimental":
        for k in ("reference_embed_dir", "download_root"):
            if not getattr(args, k):
                raise ValueError(f"--mode experimental requires --{k}")


def main() -> None:
    # PyTorch 2.6+ safe-loading for old ESM checkpoints (no-op problem on older Torch).
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([argparse.Namespace])

    args = parse_args()
    validate(args)
    ensure_dir(args.output_dir)
    set_seed(args.seed)
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR", os.path.join(args.output_dir, "torch_extensions")
    )
    if args.mode == "all":
        run_all(args)
    else:
        run_experimental(args)
    log("done.")
