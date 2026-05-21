import os
import re
import gc
import json
import glob
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
import torch

# PyTorch 2.6+ compatibility for old ESM checkpoints
torch.serialization.add_safe_globals([argparse.Namespace])

from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from torchdrug import core, data, transforms, layers
from torchdrug.layers import geometry

# make local repo modules importable
REPO_ROOT = "/content/ESM-GearNet"
if REPO_ROOT not in os.sys.path:
    os.sys.path.append(REPO_ROOT)

import gearnet.model
import gearnet.dataset
import gearnet.task
#import gearnet.protbert


from gearnet_common import (
    set_seed,
    log,
    ensure_dir,
    accession_from_path,
    find_all_pdbs,
    LabelArtifacts,
    build_labels,
    load_taxonomy_whitelist,
    build_samples,
    add_multihot_columns,
    multilabel_iterative_split_fallback,
    make_split,
    PDBDataset,
    make_collate_fn,
    safe_iter_loader,
    save_split_embeddings,
    load_download_manifest,
    build_eval_df,
)

# -----------------------------


# -----------------------------



# -----------------------------
# ESM-GearNet model
# -----------------------------
class ESMGearNetEmbeddingModel(nn.Module):
    def __init__(self, checkpoint_path: str, esm_weight_dir: str):
        super().__init__()

        self.graph_construction_model = layers.GraphConstruction(
            node_layers=[geometry.AlphaCarbonNode()],
            edge_layers=[
                geometry.SequentialEdge(max_distance=2),
                geometry.SpatialEdge(radius=10.0, min_distance=5),
                geometry.KNNEdge(k=10, min_distance=5),
            ],
            edge_feature="gearnet",
        )

        cfg = {
            "class": "FusionNetwork",
            "sequence_model": {
                "class": "ESM",
                "path": esm_weight_dir,
                "model": "ESM-2-650M",
            },
            "structure_model": {
                "class": "GearNet",
                "input_dim": 1280,
                "hidden_dims": [512, 512, 512, 512, 512, 512],
                "batch_norm": True,
                "concat_hidden": True,
                "short_cut": True,
                "readout": "sum",
                "num_relation": 7,
            }
        }

        self.model = core.Configurable.load_config_dict(cfg)

        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        log(f"Loaded ESM-GearNet checkpoint. missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            log(f"First missing keys: {missing[:10]}")
        if unexpected:
            log(f"First unexpected keys: {unexpected[:10]}")

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, packed_protein):
        packed_protein.view = "residue"
        graph = self.graph_construction_model(packed_protein)

        if not hasattr(graph, "residue_feature") or graph.residue_feature is None:
            raise RuntimeError("graph.residue_feature missing after graph construction")

        residue_feature = graph.residue_feature.float()
        out = self.model(graph, residue_feature)
        return out["graph_feature"]


# -----------------------------



# -----------------------------
# mode: all
# -----------------------------
def run_all_mode(args):
    ensure_dir(args.output_dir)

    aspect_top_counts = {
        "BPO": int(args.bpo_top_labels),
        "MFO": int(args.mfo_top_labels),
        "CCO": int(args.cco_top_labels),
    }

    labels = build_labels(args.train_terms_tsv, aspect_top_counts)
    num_labels = len(labels.idx_to_term)
    log(f"num_labels={num_labels}")

    df = build_samples(
        pdb_root=args.pdb_root,
        labels=labels,
        taxonomy_path=args.train_taxonomy_tsv,
        require_label=True,
    )
    df, y = add_multihot_columns(df, num_labels)
    df = make_split(df, y, seed=args.seed)

    split_path = os.path.join(args.output_dir, "splits.csv")
    df.to_csv(split_path, index=False)
    log(f"wrote {split_path}")

    with open(os.path.join(args.output_dir, "label_vocab.json"), "w") as f:
        json.dump(
            {
                "idx_to_term": labels.idx_to_term,
                "idx_to_aspect": labels.idx_to_aspect,
                "aspect_top_counts": labels.aspect_top_counts,
            },
            f,
            indent=2,
        )

    for split_name in ["train", "valid", "test"]:
        sdf = df[df["split"] == split_name].reset_index(drop=True)
        mean_labels = sdf["num_pos"].mean() if len(sdf) else 0.0
        log(f"{split_name}: n={len(sdf)} mean_labels={mean_labels:.2f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}")

    model = ESMGearNetEmbeddingModel(
        checkpoint_path=args.checkpoint,
        esm_weight_dir=args.esm_weight_dir,
    ).to(device)
    model.eval()

    collate_fn = make_collate_fn()
    wanted_splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split_name in wanted_splits:
        sdf = df[df["split"] == split_name].reset_index(drop=True)
        if len(sdf) == 0:
            log(f"[{split_name}] empty split, skipping")
            continue

        ds = PDBDataset(
            sdf,
            num_labels=num_labels,
            max_length=args.max_length,
            use_target_col=False,
            kwargs={"atom_feature": None, "bond_feature": None, "residue_feature": "default"},
        )

        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(args.num_workers > 0),
            drop_last=False,
        )

        save_split_embeddings(
            model=model,
            loader=loader,
            device=device,
            split_name=split_name,
            output_dir=args.output_dir,
            save_pt=args.save_pt,
            save_npy=args.save_npy,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log("Done.")


# -----------------------------
# mode: experimental
# -----------------------------
def run_experimental_mode(args):
    ensure_dir(args.output_dir)

    valid_manifest = args.valid_manifest or os.path.join(
        args.download_root, "valid", "valid_experimental_structures.csv"
    )
    test_manifest = args.test_manifest or os.path.join(
        args.download_root, "test", "test_experimental_structures.csv"
    )

    valid_pt = args.valid_pt or os.path.join(args.reference_embed_dir, "valid_embeddings.pt")
    test_pt = args.test_pt or os.path.join(args.reference_embed_dir, "test_embeddings.pt")

    label_vocab_path = os.path.join(args.reference_embed_dir, "label_vocab.json")
    with open(label_vocab_path) as f:
        vocab = json.load(f)
    with open(os.path.join(args.output_dir, "label_vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)

    num_labels = len(vocab["idx_to_term"])
    log(f"num_labels={num_labels}")

    valid_df = build_eval_df(valid_pt, valid_manifest, "valid")
    test_df = build_eval_df(test_pt, test_manifest, "test")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}")

    model = ESMGearNetEmbeddingModel(
        checkpoint_path=args.checkpoint,
        esm_weight_dir=args.esm_weight_dir,
    ).to(device)
    model.eval()

    collate_fn = make_collate_fn()

    for split_name, sdf in [("valid", valid_df), ("test", test_df)]:
        ds = PDBDataset(
            sdf,
            num_labels=num_labels,
            max_length=args.max_length,
            use_target_col=True,
            kwargs={"atom_feature": None, "bond_feature": None, "residue_feature": "default"},
        )

        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(args.num_workers > 0),
            drop_last=False,
        )

        save_split_embeddings(
            model=model,
            loader=loader,
            device=device,
            split_name=split_name,
            output_dir=args.output_dir,
            save_pt=args.save_pt,
            save_npy=args.save_npy,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log("Done.")
    log(f"Saved experimental ESM-GearNet embeddings to: {args.output_dir}")


# -----------------------------
# main
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True, choices=["all", "experimental"])

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--esm_weight_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=550)

    parser.add_argument("--save_pt", action="store_true")
    parser.add_argument("--save_npy", action="store_true")

    # mode = all
    parser.add_argument("--pdb_root", type=str, default=None)
    parser.add_argument("--train_terms_tsv", type=str, default=None)
    parser.add_argument("--train_taxonomy_tsv", type=str, default=None)
    parser.add_argument("--bpo_top_labels", type=int, default=1100)
    parser.add_argument("--mfo_top_labels", type=int, default=450)
    parser.add_argument("--cco_top_labels", type=int, default=300)
    parser.add_argument("--splits", type=str, default="train,valid,test")

    # mode = experimental
    parser.add_argument("--reference_embed_dir", type=str, default=None)
    parser.add_argument("--download_root", type=str, default=None)
    parser.add_argument("--valid_manifest", type=str, default=None)
    parser.add_argument("--test_manifest", type=str, default=None)
    parser.add_argument("--valid_pt", type=str, default=None)
    parser.add_argument("--test_pt", type=str, default=None)

    return parser.parse_args()


def validate_args(args):
    if not args.save_pt and not args.save_npy:
        args.save_pt = True
        args.save_npy = True

    if args.mode == "all":
        required = ["pdb_root", "train_terms_tsv"]
        missing = [k for k in required if not getattr(args, k)]
        if missing:
            raise ValueError(f"--mode all requires: {missing}")

    if args.mode == "experimental":
        required = ["reference_embed_dir", "download_root"]
        missing = [k for k in required if not getattr(args, k)]
        if missing:
            raise ValueError(f"--mode experimental requires: {missing}")


def main():
    args = parse_args()
    validate_args(args)

    ensure_dir(args.output_dir)
    set_seed(args.seed)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(args.output_dir, "torch_extensions"))

    if args.mode == "all":
        run_all_mode(args)
    elif args.mode == "experimental":
        run_experimental_mode(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()