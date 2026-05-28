"""PyTorch Dataset + collate that wrap a sample table into TorchDrug Protein graphs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from torchdrug import data, transforms

from .utils import log


class PDBDataset(Dataset):
    """
    Each row of `df` must have: accession (str), pdb_path (str),
    target (np.ndarray[num_labels], float32).
    """

    def __init__(self, df: pd.DataFrame, encoder_name: str, max_length: int = 0):
        self.df = df.reset_index(drop=True).copy()
        self.max_length = int(max_length)

        self.pdb_files = self.df["pdb_path"].astype(str).tolist()
        self.accessions = self.df["accession"].astype(str).tolist()
        self.targets = np.stack(self.df["target"].tolist()).astype(np.float32)

        # GearNet's edge_feature="gearnet" needs atom-level features, and ESM-GearNet
        # works off residue-level features only.
        if encoder_name == "gearnet":
            self.from_pdb_kwargs = dict(
                atom_feature="default",
                bond_feature="default",
                residue_feature="default",
            )
        else:  # esm_gearnet
            self.from_pdb_kwargs = dict(
                atom_feature=None,
                bond_feature=None,
                residue_feature="default",
            )

        steps = [transforms.ProteinView("residue")]
        if self.max_length > 0:
            steps.append(transforms.TruncateProtein(self.max_length))
        self.transform = transforms.Compose(steps)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        protein = data.Protein.from_pdb(self.pdb_files[index], **self.from_pdb_kwargs)
        item = self.transform({"graph": protein})
        protein = item["graph"]

        feat = getattr(protein, "residue_feature", None)
        if feat is not None and hasattr(feat, "to_dense"):
            with protein.residue():
                protein.residue_feature = feat.to_dense()

        return {
            "graph": protein,
            "target": torch.from_numpy(self.targets[index]).float(),
            "accession": self.accessions[index],
            "pdb_file": self.pdb_files[index],
        }


def collate(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    packed = data.Protein.pack([x["graph"] for x in batch])
    packed.view = "residue"
    return {
        "graph": packed,
        "target": torch.stack([x["target"] for x in batch], dim=0),
        "accession": [x["accession"] for x in batch],
        "pdb_file": [x["pdb_file"] for x in batch],
    }


def safe_iter(loader: DataLoader, desc: str, max_logs: int = 20):
    """Iterate a DataLoader while skipping safely per-batch crashes (bad PDB files etc.)."""
    skipped = 0
    it = iter(loader)
    pbar = tqdm(total=len(loader), desc=desc, leave=False)
    while True:
        try:
            batch = next(it)
        except StopIteration:
            break
        except Exception as e:
            pbar.update(1)
            skipped += 1
            if skipped <= max_logs:
                log(f"[{desc}] skip bad batch: {e!r}")
            continue
        pbar.update(1)
        if batch is None:
            skipped += 1
            continue
        yield batch
    pbar.close()
    if skipped:
        log(f"[{desc}] skipped batches: {skipped}")
