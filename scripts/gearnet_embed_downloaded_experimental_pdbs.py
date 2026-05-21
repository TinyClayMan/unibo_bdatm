import os
import gc
import csv
import json
import argparse
from typing import Dict, List

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
    make_collate_fn,
    PDBDataset,
    build_eval_df,
    save_split_embeddings,
)


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
      if isinstance(state, dict):
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





def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--reference_embed_dir", type=str, required=True)
  parser.add_argument("--download_root", type=str, required=True)
  parser.add_argument("--checkpoint", type=str, required=True)
  parser.add_argument("--output_dir", type=str, required=True)

  parser.add_argument("--valid_manifest", type=str, default=None)
  parser.add_argument("--test_manifest", type=str, default=None)
  parser.add_argument("--valid_pt", type=str, default=None)
  parser.add_argument("--test_pt", type=str, default=None)

  parser.add_argument("--batch_size", type=int, default=2)
  parser.add_argument("--num_workers", type=int, default=0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--save_pt", action="store_true")
  parser.add_argument("--save_npy", action="store_true")

  args = parser.parse_args()

  ensure_dir(args.output_dir)
  set_seed(args.seed)
  os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(args.output_dir, "torch_extensions"))

  if not args.save_pt and not args.save_npy:
      args.save_pt = True
      args.save_npy = True

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

  kwargs = {
      "atom_feature": "default",
      "bond_feature": "default",
      "residue_feature": "default",
  }

  device = "cuda" if torch.cuda.is_available() else "cpu"
  log(f"device={device}")

  collate_fn = make_collate_fn()
  model = GearNetEmbeddingModel(args.checkpoint).to(device)
  model.eval()

  for split_name, sdf in [("valid", valid_df), ("test", test_df)]:
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
  log(f"Saved experimental GearNet embeddings to: {args.output_dir}")


if __name__ == "__main__":
  main()