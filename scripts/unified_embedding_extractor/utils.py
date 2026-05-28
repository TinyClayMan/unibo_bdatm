"""Small helpers shared across the package: seeding, paths, accession parsing."""

from __future__ import annotations

import glob
import os
import random
import re
from typing import List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# AlphaFold filenames look like AF-Q12345-F1-model_v4.pdb or AF-12345.pdb.
# Anything else is treated as already being the accession.
_ACC_RE_F = re.compile(r"AF-([A-Z0-9]+)-F\d+$")
_ACC_RE_N = re.compile(r"AF-(\d+)$")


def accession_from_path(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _ACC_RE_F.match(stem) or _ACC_RE_N.match(stem)
    return m.group(1) if m else stem


def find_all_pdbs(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "**", "*.pdb"), recursive=True))


def load_split_pt(path: str):
    """Read one of our `<split>_embeddings.pt` files and return (targets, ids)."""
    obj = torch.load(path, map_location="cpu")
    y = obj["targets"]
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    y = np.asarray(y, dtype=np.float32)
    ids = [str(a) for a in obj["accessions"]]
    return y, ids
