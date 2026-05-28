"""GO label vocabulary built from the CAFA-5 train_terms.tsv file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .utils import log


@dataclass
class Labels:
    entry_to_idx: Dict[str, List[int]]
    idx_to_term: List[str]
    idx_to_aspect: List[str]
    aspect_top_counts: Dict[str, int]


def build_labels(train_terms_tsv: str, top_per_aspect: Dict[str, int]) -> Labels:
    """Keep the top-N most frequent terms per GO aspect (BPO / MFO / CCO)."""
    df = pd.read_csv(train_terms_tsv, sep="\t")
    needed = {"EntryID", "term", "aspect"}
    if not needed.issubset(df.columns):
        raise ValueError(f"train_terms.tsv missing columns: {needed - set(df.columns)}")
    df = df.copy()
    df["EntryID"] = df["EntryID"].astype(str)

    term_to_idx: Dict[str, int] = {}
    idx_to_term: List[str] = []
    idx_to_aspect: List[str] = []
    for aspect, n in top_per_aspect.items():
        sub = df[df["aspect"] == aspect]
        if sub.empty:
            log(f"warn: no rows for aspect={aspect}")
            continue
        for term in sub["term"].value_counts().head(int(n)).index:
            if term in term_to_idx:
                continue
            term_to_idx[term] = len(idx_to_term)
            idx_to_term.append(term)
            idx_to_aspect.append(aspect)

    counts = ", ".join(
        f"{a}:{sum(1 for x in idx_to_aspect if x == a)}" for a in top_per_aspect
    )
    log(f"labels per aspect -> {counts} (total={len(idx_to_term)})")

    sub = df[df["term"].isin(term_to_idx)].copy()
    sub["term_idx"] = sub["term"].map(term_to_idx)
    entry_to_idx = (
        sub.groupby("EntryID")["term_idx"]
        .apply(lambda s: sorted({int(x) for x in s}))
        .to_dict()
    )
    return Labels(
        entry_to_idx=entry_to_idx,
        idx_to_term=idx_to_term,
        idx_to_aspect=idx_to_aspect,
        aspect_top_counts=top_per_aspect,
    )


def load_whitelist(path: Optional[str]) -> Optional[set]:
    """Optional EntryID whitelist (e.g. from train_taxonomy.tsv)."""
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t")
    if "EntryID" not in df.columns:
        return None
    return set(df["EntryID"].astype(str).tolist())
