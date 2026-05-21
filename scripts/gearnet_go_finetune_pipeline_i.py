# GearNet embedding extraction pipeline for custom PDB shards + GO labels
# Per-aspect top-K labels:
# - BPO / MFO / CCO selected independently
# - no classification head
# - encoder only, eval mode only
# - saves embeddings + targets + accession names for easy downstream training

import os
import re
import gc
import json
import glob
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from torchdrug import data, models, layers
from torchdrug.layers import geometry

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
)

# -----------------------------
# model
# -----------------------------
class GearNetEmbeddingModel(nn.Module):
    def __init__(self, checkpoint_path: str):
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

        self.encoder = models.GearNet(
            input_dim=21,
            hidden_dims=[512, 512, 512, 512, 512, 512],
            num_relation=7,
            edge_input_dim=59,
            num_angle_bin=8,
            batch_norm=True,
            concat_hidden=True,
            short_cut=True,
            readout="sum",
        )

        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
            if "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            elif "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]

        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        log(f"Loaded encoder checkpoint. missing={len(missing)} unexpected={len(unexpected)}")

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, packed_protein):
        packed_protein.view = "residue"
        graph = self.graph_construction_model(packed_protein)
        node_input = graph.node_feature.float()
        out = self.encoder(graph, node_input)
        return out["graph_feature"]


# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_root", type=str, required=True)
    parser.add_argument("--train_terms_tsv", type=str, required=True)
    parser.add_argument("--train_taxonomy_tsv", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--bpo_top_labels", type=int, default=1100)
    parser.add_argument("--mfo_top_labels", type=int, default=450)
    parser.add_argument("--cco_top_labels", type=int, default=300)

    parser.add_argument("--save_pt", action="store_true")
    parser.add_argument("--save_npy", action="store_true")
    parser.add_argument("--splits", type=str, default="train,valid,test")

    args = parser.parse_args()

    ensure_dir(args.output_dir)
    set_seed(args.seed)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(args.output_dir, "torch_extensions"))

    if not args.save_pt and not args.save_npy:
        args.save_pt = True
        args.save_npy = True

    aspect_top_counts = {
        "BPO": int(args.bpo_top_labels),
        "MFO": int(args.mfo_top_labels),
        "CCO": int(args.cco_top_labels),
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}")

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
        log(f"{split_name}: n={len(sdf)} mean_labels={sdf['num_pos'].mean():.2f}")

    kwargs = {
        "atom_feature": "default",
        "bond_feature": "default",
        "residue_feature": "default",
    }

    collate_fn = make_collate_fn()
    model = GearNetEmbeddingModel(args.checkpoint).to(device)
    model.eval()

    wanted_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split_name in wanted_splits:
        sdf = df[df["split"] == split_name].reset_index(drop=True)
        if len(sdf) == 0:
            log(f"[{split_name}] empty split, skipping")
            continue

        ds = PDBDataset(sdf, num_labels, kwargs=kwargs, use_target_col=True)
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


if __name__ == "__main__":
    main()