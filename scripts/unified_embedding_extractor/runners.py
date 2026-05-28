"""Top-level run modes: --mode all (random split) and --mode experimental."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .embed import embed_all_splits
from .labels import build_labels, load_whitelist
from .samples import add_multihot, build_samples, random_split
from .utils import load_split_pt, log


# ---------- mode: all ----------
def run_all(args) -> None:
    top_per_aspect = {
        "BPO": args.bpo_top_labels,
        "MFO": args.mfo_top_labels,
        "CCO": args.cco_top_labels,
    }
    labels = build_labels(args.train_terms_tsv, top_per_aspect)
    num_labels = len(labels.idx_to_term)
    log(f"num_labels={num_labels}")

    whitelist = load_whitelist(args.train_taxonomy_tsv)
    df = build_samples(args.pdb_root, labels, whitelist=whitelist)
    df, y = add_multihot(df, num_labels)
    df = random_split(df, seed=args.seed, val_frac=args.val_frac, test_frac=args.test_frac)
    df["target"] = [y[i] for i in range(len(df))]

    df[["accession", "pdb_path", "num_pos", "split"]].to_csv(
        os.path.join(args.output_dir, "splits.csv"), index=False
    )
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

    for s in ["train", "valid", "test"]:
        sub = df[df["split"] == s]
        mean_lab = sub["num_pos"].mean() if len(sub) else 0.0
        log(f"{s}: n={len(sub)} mean_labels={mean_lab:.2f}")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    embed_all_splits(args, df, splits=splits)


# ---------- mode: experimental ----------
def _load_manifest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"accession", "status", "download_path"}
    if not needed.issubset(df.columns):
        raise ValueError(f"manifest missing columns: {needed - set(df.columns)}")
    df = df[df["status"] == "downloaded"].copy()
    df = df[df["download_path"].notna()]
    df = df[df["download_path"].astype(str).str.lower().str.endswith(".pdb")]
    df = df[df["download_path"].map(os.path.exists)]
    df["accession"] = df["accession"].astype(str)
    return df.drop_duplicates("accession").reset_index(drop=True)


def _build_experimental_df(ref_pt: str, manifest_csv: str, split_name: str) -> pd.DataFrame:
    y_ref, ids_ref = load_split_pt(ref_pt)
    manifest = _load_manifest(manifest_csv)
    ref = {a: i for i, a in enumerate(ids_ref)}

    rows = []
    missing = 0
    for _, m in manifest.iterrows():
        acc = m["accession"]
        i = ref.get(acc)
        if i is None:
            missing += 1
            continue
        rows.append(
            {
                "accession": acc,
                "pdb_path": m["download_path"],
                "target": y_ref[i].astype(np.float32),
                "num_pos": int(y_ref[i].sum()),
            }
        )
    if not rows:
        raise RuntimeError(f"[{split_name}] no overlap between manifest and reference split")
    df = pd.DataFrame(rows)
    log(f"[{split_name}] manifest={len(manifest)} overlap={len(df)} missing_from_ref={missing}")
    return df


def run_experimental(args) -> None:
    valid_pt = args.valid_pt or os.path.join(args.reference_embed_dir, "valid_embeddings.pt")
    test_pt = args.test_pt or os.path.join(args.reference_embed_dir, "test_embeddings.pt")
    valid_csv = args.valid_manifest or os.path.join(
        args.download_root, "valid", "valid_experimental_structures.csv"
    )
    test_csv = args.test_manifest or os.path.join(
        args.download_root, "test", "test_experimental_structures.csv"
    )

    with open(os.path.join(args.reference_embed_dir, "label_vocab.json")) as f:
        vocab = json.load(f)
    log(f"num_labels={len(vocab['idx_to_term'])}")
    with open(os.path.join(args.output_dir, "label_vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)

    dfs = {
        "valid": _build_experimental_df(valid_pt, valid_csv, "valid"),
        "test": _build_experimental_df(test_pt, test_csv, "test"),
    }
    df = pd.concat([d.assign(split=s) for s, d in dfs.items()], ignore_index=True)
    embed_all_splits(args, df, splits=["valid", "test"])
