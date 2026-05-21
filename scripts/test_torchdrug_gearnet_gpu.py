import os
import glob
import sys
import subprocess

import torch
from torch.utils import data as torch_data

from torchdrug import data, models, layers
from torchdrug.layers import geometry

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

EXTRACT_DIR = "/content/pdb_scratch/shard0"
CKPT_PATH = "/content/checkpoints/mc_gearnet_edge.pth"
CKPT_URL = "https://zenodo.org/record/7593637/files/mc_gearnet_edge.pth"

BATCH_SIZE = 2
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg):
    print(msg, flush=True)


class LazyPDBDataset(data.ProteinDataset):
    def get_item(self, index):
        if getattr(self, "lazy", False):
            protein = data.Protein.from_pdb(self.pdb_files[index], **self.kwargs)
        else:
            protein = self.data[index].clone()

        if hasattr(protein, "residue_feature") and protein.residue_feature is not None:
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()

        item = {"graph": protein, "pdb_file": self.pdb_files[index]}
        if self.transform:
            item = self.transform(item)
        return item


def ensure_checkpoint():
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    if os.path.exists(CKPT_PATH):
        log(f"Checkpoint already exists: {CKPT_PATH}")
        return
    log(f"Downloading checkpoint to {CKPT_PATH}")
    subprocess.run(
        ["wget", "-O", CKPT_PATH, CKPT_URL],
        check=True
    )
    log("Checkpoint downloaded")


def build_graph_construction_model():
    return layers.GraphConstruction(
        node_layers=[geometry.AlphaCarbonNode()],
        edge_layers=[
            geometry.SequentialEdge(max_distance=2),
            geometry.SpatialEdge(radius=10.0, min_distance=5),
            geometry.KNNEdge(k=10, min_distance=5),
        ],
        edge_feature="gearnet"
    )


def build_model():
    # Matches the commonly used GearNet-Edge setup used in the official repo/examples
    model = models.GearNet(
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
    return model


def extract_state_dict(obj):
    if isinstance(obj, dict):
        # common checkpoint wrappers
        for key in ["model", "state_dict", "model_state_dict"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        # maybe already a raw state_dict
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise RuntimeError("Could not extract a state_dict from checkpoint")


def collate_fn(batch):
    proteins = [item["graph"] for item in batch]
    pdb_files = [item["pdb_file"] for item in batch]
    packed = data.Protein.pack(proteins)
    packed.view = "residue"
    return {
        "graph": packed,
        "pdb_files": pdb_files,
    }


def main():
    log(f"Using device: {DEVICE}")
    if DEVICE != "cuda":
        log("WARNING: CUDA is not available, so this is not a GPU test.")

    ensure_checkpoint()

    pdb_files = sorted(glob.glob(os.path.join(EXTRACT_DIR, "**", "*.pdb"), recursive=True))
    if not pdb_files:
        raise RuntimeError(f"No .pdb files found under {EXTRACT_DIR}")

    log(f"Found {len(pdb_files)} pdb files")
    log(f"First few: {pdb_files[:3]}")

    dataset = LazyPDBDataset()
    dataset.load_pdbs(
        pdb_files,
        lazy=True,
        atom_feature="default",
        bond_feature="default",
        residue_feature="default",
    )
    log(f"Dataset size: {len(dataset)}")

    loader = torch_data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    batch = next(iter(loader))
    protein = batch["graph"]
    log(f"Packed raw protein batch_size: {protein.batch_size}")
    log(f"PDBs in batch: {batch['pdb_files']}")

    graph_construction_model = build_graph_construction_model()
    model = build_model()

    ckpt_obj = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = extract_state_dict(ckpt_obj)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    log(f"Checkpoint loaded from {CKPT_PATH}")
    log(f"Missing keys: {len(missing)}")
    log(f"Unexpected keys: {len(unexpected)}")
    if missing:
        log(f"First missing keys: {missing[:10]}")
    if unexpected:
        log(f"First unexpected keys: {unexpected[:10]}")

    graph_construction_model = graph_construction_model.to(DEVICE)
    model = model.to(DEVICE)
    model.eval()

    protein = protein.to(DEVICE)

    with torch.no_grad():
        graph = graph_construction_model(protein)
        node_input = graph.node_feature.float()
        output = model(graph, node_input)

    graph_feature = output["graph_feature"]
    node_feature = output["node_feature"]

    log("GearNet forward pass succeeded")
    log(f"Constructed graph: {graph}")
    log(f"Constructed graph batch_size: {graph.batch_size}")
    log(f"Constructed graph num_node: {graph.num_node}")
    log(f"Constructed graph num_edge: {graph.num_edge}")
    log(f"Input node_feature shape: {tuple(graph.node_feature.shape)}")
    log(f"Output graph_feature shape: {tuple(graph_feature.shape)}")
    log(f"Output node_feature shape: {tuple(node_feature.shape)}")
    log(f"Output graph_feature dtype: {graph_feature.dtype}")
    log(f"Output node_feature dtype: {node_feature.dtype}")

    if DEVICE == "cuda":
        mem = torch.cuda.memory_allocated() / 1024**2
        log(f"CUDA memory allocated: {mem:.2f} MiB")


if __name__ == "__main__":
    main()