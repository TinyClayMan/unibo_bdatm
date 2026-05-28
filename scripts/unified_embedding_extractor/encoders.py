"""GearNet and ESM-GearNet encoders, plus the shared graph construction."""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn

from torchdrug import core, layers, models
from torchdrug.layers import geometry

from .utils import log


# Path to the ESM-GearNet repo (only needed for --encoder esm_gearnet).
_REPO_ROOT = "/content/ESM-GearNet"
if os.path.isdir(_REPO_ROOT) and _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)


def _load_state(model: nn.Module, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict):
        if "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        elif "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    log(f"checkpoint loaded: missing={len(missing)} unexpected={len(unexpected)}")


def _graph_construction() -> layers.GraphConstruction:
    """Same node/edge construction for both encoders."""
    return layers.GraphConstruction(
        node_layers=[geometry.AlphaCarbonNode()],
        edge_layers=[
            geometry.SequentialEdge(max_distance=2),
            geometry.SpatialEdge(radius=10.0, min_distance=5),
            geometry.KNNEdge(k=10, min_distance=5),
        ],
        edge_feature="gearnet",
    )


class GearNetEncoder(nn.Module):
    def __init__(self, checkpoint: str):
        super().__init__()
        self.graph_construction = _graph_construction()
        self.encoder = models.GearNet(
            input_dim=21,
            hidden_dims=[512] * 6,
            num_relation=7,
            edge_input_dim=59,
            num_angle_bin=8,
            batch_norm=True,
            concat_hidden=True,
            short_cut=True,
            readout="sum",
        )
        _load_state(self.encoder, checkpoint)

    def forward(self, packed):
        packed.view = "residue"
        graph = self.graph_construction(packed)
        return self.encoder(graph, graph.node_feature.float())["graph_feature"]


class ESMGearNetEncoder(nn.Module):
    def __init__(self, checkpoint: str, esm_weight_dir: str):
        super().__init__()
        self.graph_construction = _graph_construction()
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
                "hidden_dims": [512] * 6,
                "batch_norm": True,
                "concat_hidden": True,
                "short_cut": True,
                "readout": "sum",
                "num_relation": 7,
            },
        }
        self.model = core.Configurable.load_config_dict(cfg)
        _load_state(self.model, checkpoint)

    def forward(self, packed):
        packed.view = "residue"
        graph = self.graph_construction(packed)
        if getattr(graph, "residue_feature", None) is None:
            raise RuntimeError("graph.residue_feature missing after graph construction")
        return self.model(graph, graph.residue_feature.float())["graph_feature"]


def build_encoder(name: str, checkpoint: str, esm_weight_dir: Optional[str]) -> nn.Module:
    if name == "gearnet":
        enc: nn.Module = GearNetEncoder(checkpoint)
    elif name == "esm_gearnet":
        if not esm_weight_dir:
            raise ValueError("--esm_weight_dir is required for --encoder esm_gearnet")
        import gearnet.model  
        enc = ESMGearNetEncoder(checkpoint, esm_weight_dir)
    else:
        raise ValueError(f"unknown encoder: {name}")
    for p in enc.parameters():
        p.requires_grad = False
    return enc.eval()
