"""Build the (accession, pdb_path, label_indices) sample table and split it."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .labels import Labels
from .utils import accession_from_path, find_all_pdbs, log


def build_samples(pdb_root: str, labels: Labels, whitelist: Optional[set] = None) -> pd.DataFrame:
    paths = find_all_pdbs(pdb_root)
    log(f"found {len(paths)} pdb files under {pdb_root}")

    rows = []
    for path in tqdm(paths, desc="indexing"):
        acc = accession_from_path(path)
        if whitelist is not None and acc not in whitelist:
            continue
        idxs = labels.entry_to_idx.get(acc)
        if not idxs:
            continue
        rows.append({"accession": acc, "pdb_path": path, "label_indices": idxs})

    if not rows:
        raise RuntimeError("no labeled PDB samples found")
    df = pd.DataFrame(rows)
    log(f"labeled samples: {len(df)}")
    return df


def add_multihot(df: pd.DataFrame, num_labels: int):
    y = np.zeros((len(df), num_labels), dtype=np.float32)
    for i, idxs in enumerate(df["label_indices"]):
        y[i, idxs] = 1.0
    df = df.copy()
    df["num_pos"] = y.sum(axis=1).astype(int)
    return df, y


def random_split(
    df: pd.DataFrame, seed: int, val_frac: float, test_frac: float
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(df)
    perm = rng.permutation(n)
    n_test = max(1, int(round(test_frac * n)))
    n_val = max(1, int(round(val_frac * n)))
    splits = np.array(["train"] * n, dtype=object)
    splits[perm[:n_test]] = "test"
    splits[perm[n_test : n_test + n_val]] = "valid"
    df = df.copy().reset_index(drop=True)
    df["split"] = splits
    return df
