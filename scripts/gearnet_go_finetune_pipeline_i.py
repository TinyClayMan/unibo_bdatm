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
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from torchdrug import data, models, layers
from torchdrug.layers import geometry


# -----------------------------
# basic utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log(msg: str):
    print(msg, flush=True)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def accession_from_path(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]

    # Main case: AF-A0A0S2Z5D6-F1(.pdb) -> A0A0S2Z5D6
    m = re.match(r"AF-([A-Z0-9]+)-F\d+$", stem)
    if m:
        return m.group(1)

    # Fallback for numeric AFDB-style names
    m = re.match(r"AF-(\d+)$", stem)
    if m:
        return m.group(1)

    return stem


def find_all_pdbs(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "**", "*.pdb"), recursive=True))


# -----------------------------
# labels and sample building
# -----------------------------
@dataclass
class LabelArtifacts:
    entry_to_label_idxs: Dict[str, List[int]]
    idx_to_term: List[str]
    idx_to_aspect: List[str]
    term_to_idx: Dict[str, int]
    aspect_top_counts: Dict[str, int]


def build_labels(train_terms_path: str, aspect_top_counts: Dict[str, int]) -> LabelArtifacts:
    """
    Select top-frequency GO terms separately per aspect.
    Example:
        {'BPO': 1100, 'MFO': 450, 'CCO': 300}
    """
    train_terms = pd.read_csv(train_terms_path, sep="\t")
    required = {"EntryID", "term", "aspect"}
    missing = required - set(train_terms.columns)
    if missing:
        raise ValueError(f"Missing columns in train_terms.tsv: {missing}")

    train_terms = train_terms.copy()
    train_terms["EntryID"] = train_terms["EntryID"].astype(str)

    idx_to_term = []
    idx_to_aspect = []
    term_to_idx = {}
    term_aspect_dict = {}
    idx_counter = 0

    for aspect, n in aspect_top_counts.items():
        aspect_terms = train_terms[train_terms["aspect"] == aspect]
        if aspect_terms.empty:
            log(f"Warning: no rows found for aspect={aspect}")
            continue

        top_for_aspect = (
            aspect_terms["term"]
            .value_counts()
            .head(int(n))
            .index
            .tolist()
        )

        for term in top_for_aspect:
            if term not in term_to_idx:
                term_to_idx[term] = idx_counter
                term_aspect_dict[term] = aspect
                idx_to_term.append(term)
                idx_to_aspect.append(aspect)
                idx_counter += 1

    log(
        "Selected labels per aspect -> "
        f"BPO: {sum(1 for a in idx_to_aspect if a == 'BPO')}, "
        f"MFO: {sum(1 for a in idx_to_aspect if a == 'MFO')}, "
        f"CCO: {sum(1 for a in idx_to_aspect if a == 'CCO')} "
        f"(total={len(idx_to_term)})"
    )

    top_df = train_terms[train_terms["term"].isin(term_to_idx)].copy()
    top_df["term_idx"] = top_df["term"].map(term_to_idx)

    entry_to_label_idxs = (
        top_df.groupby("EntryID")["term_idx"]
        .apply(lambda s: sorted(set(int(x) for x in s.tolist())))
        .to_dict()
    )

    return LabelArtifacts(
        entry_to_label_idxs=entry_to_label_idxs,
        idx_to_term=idx_to_term,
        idx_to_aspect=idx_to_aspect,
        term_to_idx=term_to_idx,
        aspect_top_counts=aspect_top_counts,
    )


def load_taxonomy_whitelist(path: Optional[str]):
    if not path or not os.path.exists(path):
        return None, None
    df = pd.read_csv(path, sep="\t")
    whitelist = None
    if "EntryID" in df.columns:
        whitelist = set(df["EntryID"].astype(str).tolist())
    return df, whitelist


def build_samples(
    pdb_root: str,
    labels: LabelArtifacts,
    taxonomy_path: Optional[str] = None,
    require_label: bool = True,
):
    taxonomy_df, whitelist = load_taxonomy_whitelist(taxonomy_path)
    tax_cols = [] if taxonomy_df is None else list(taxonomy_df.columns)
    tax_map = {}
    if taxonomy_df is not None and "EntryID" in taxonomy_df.columns:
        taxonomy_df = taxonomy_df.copy()
        taxonomy_df["EntryID"] = taxonomy_df["EntryID"].astype(str)
        tax_map = taxonomy_df.set_index("EntryID").to_dict(orient="index")

    all_pdbs = find_all_pdbs(pdb_root)
    log(f"Discovered {len(all_pdbs)} pdb files under {pdb_root}")

    rows = []
    matched_labels = 0
    matched_whitelist = 0
    both = 0

    for path in tqdm(all_pdbs, desc="building sample index"):
        accession = accession_from_path(path)
        in_labels = accession in labels.entry_to_label_idxs
        in_whitelist = whitelist is None or accession in whitelist

        if in_labels:
            matched_labels += 1
        if in_whitelist:
            matched_whitelist += 1
        if in_labels and in_whitelist:
            both += 1

        if whitelist is not None and accession not in whitelist:
            continue

        label_idxs = labels.entry_to_label_idxs.get(accession, [])
        if require_label and not label_idxs:
            continue

        row = {
            "accession": accession,
            "pdb_path": path,
            "label_indices": label_idxs,
        }
        if accession in tax_map:
            row.update(tax_map[accession])
        rows.append(row)

    log(f"label-matched pdbs: {matched_labels}")
    log(f"whitelist-matched pdbs: {matched_whitelist}")
    log(f"usable labeled pdbs: {both}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No labeled PDB samples found")
    for c in tax_cols:
        if c not in df.columns:
            df[c] = None
    return df


def add_multihot_columns(df: pd.DataFrame, num_labels: int):
    y = np.zeros((len(df), num_labels), dtype=np.float32)
    for i, idxs in enumerate(df["label_indices"].tolist()):
        y[i, idxs] = 1.0
    df = df.copy()
    df["num_pos"] = y.sum(axis=1).astype(int)
    return df, y


def multilabel_iterative_split_fallback(df: pd.DataFrame, seed: int = 42):
    work = df.copy()
    work = work.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    work["label_count"] = work["label_indices"].map(len)
    work = work.sort_values(["label_count", "accession"], ascending=[False, True]).reset_index(drop=True)

    n = len(work)
    n_test = max(1, int(round(0.10 * n)))
    n_val = max(1, int(round(0.10 * n)))

    split = np.array(["train"] * n, dtype=object)
    order = np.arange(n)
    test_idx = order[::10][:n_test]
    val_candidates = [i for i in order[1::10] if i not in set(test_idx)]
    val_idx = np.array(val_candidates[:n_val], dtype=int)
    split[test_idx] = "test"
    split[val_idx] = "valid"
    work["split"] = split
    return work


def make_split(df: pd.DataFrame, y: np.ndarray, seed: int = 42):
    work = df.copy().reset_index(drop=True)
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        train_idx, temp_idx = next(msss1.split(np.zeros(len(work)), y))
        train_df = work.iloc[train_idx].copy().reset_index(drop=True)
        temp_df = work.iloc[temp_idx].copy().reset_index(drop=True)
        y_temp = y[temp_idx]
        msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
        valid_rel, test_rel = next(msss2.split(np.zeros(len(temp_df)), y_temp))
        valid_df = temp_df.iloc[valid_rel].copy()
        test_df = temp_df.iloc[test_rel].copy()
        train_df["split"] = "train"
        valid_df["split"] = "valid"
        test_df["split"] = "test"
        out = pd.concat([train_df, valid_df, test_df], ignore_index=True)
        return out
    except Exception:
        log("iterative stratification unavailable; using fallback split")
        return multilabel_iterative_split_fallback(work, seed=seed)


# -----------------------------
# dataset
# -----------------------------
class CustomGODataset(Dataset):
    def __init__(self, df: pd.DataFrame, num_labels: int, kwargs=None):
        self.df = df.reset_index(drop=True).copy()
        self.num_labels = num_labels
        self.kwargs = kwargs or {}
        self.transform = None
        self.pdb_files = self.df["pdb_path"].tolist()
        self.accessions = self.df["accession"].astype(str).tolist()
        self.label_indices = self.df["label_indices"].tolist()
        self.targets = np.zeros((len(self.df), num_labels), dtype=np.float32)
        for i, idxs in enumerate(self.label_indices):
            self.targets[i, idxs] = 1.0

    def __len__(self):
        return len(self.df)

    def get_item(self, index):
        protein = data.Protein.from_pdb(self.pdb_files[index], **self.kwargs)
        if hasattr(protein, "residue_feature") and protein.residue_feature is not None:
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {
            "graph": protein,
            "target": torch.from_numpy(self.targets[index]).float(),
            "accession": self.accessions[index],
            "pdb_file": self.pdb_files[index],
        }
        if self.transform:
            item = self.transform(item)
        return item

    def __getitem__(self, index):
        return self.get_item(index)


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
# safe batch collation
# -----------------------------
def make_collate_fn():
    def collate(batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None
        proteins = [x["graph"] for x in batch]
        targets = torch.stack([x["target"] for x in batch], dim=0)
        accessions = [x["accession"] for x in batch]
        pdb_files = [x["pdb_file"] for x in batch]
        packed = data.Protein.pack(proteins)
        packed.view = "residue"
        return {
            "graph": packed,
            "target": targets,
            "accession": accessions,
            "pdb_file": pdb_files,
        }
    return collate


# -----------------------------
# robust loader iteration
# -----------------------------
def safe_iter_loader(loader, desc: str, dataset_name: str, max_skip_logs: int = 20):
    skipped = 0
    it = iter(loader)
    pbar = tqdm(total=len(loader), desc=desc, leave=False)
    while True:
        try:
            batch = next(it)
            pbar.update(1)
            if batch is None:
                skipped += 1
                if skipped <= max_skip_logs:
                    log(f"[{dataset_name}] skipped empty batch after filtering bad samples")
                continue
            yield batch
        except StopIteration:
            break
        except Exception as e:
            pbar.update(1)
            skipped += 1
            if skipped <= max_skip_logs:
                log(f"[{dataset_name}] skipped failing batch: {repr(e)}")
            continue
    pbar.close()
    if skipped:
        log(f"[{dataset_name}] total skipped batches: {skipped}")


# -----------------------------
# embedding extraction
# -----------------------------
def save_split_embeddings(
    model,
    loader,
    device,
    split_name: str,
    output_dir: str,
    save_pt: bool = True,
    save_npy: bool = True,
):
    model.eval()

    all_embeddings = []
    all_targets = []
    all_accessions = []
    all_pdb_files = []

    with torch.no_grad():
        for batch in safe_iter_loader(loader, desc=f"embed:{split_name}", dataset_name=split_name):
            graph = batch["graph"].to(device)
            emb = model(graph)

            all_embeddings.append(emb.cpu())
            all_targets.append(batch["target"].cpu())
            all_accessions.extend(batch["accession"])
            all_pdb_files.extend(batch["pdb_file"])

    if len(all_embeddings) == 0:
        log(f"[{split_name}] no embeddings produced")
        return

    embeddings = torch.cat(all_embeddings, dim=0).contiguous()
    targets = torch.cat(all_targets, dim=0).contiguous()

    log(
        f"[{split_name}] saved rows={embeddings.shape[0]} "
        f"embedding_dim={embeddings.shape[1]}"
    )

    if save_pt:
        pt_path = os.path.join(output_dir, f"{split_name}_embeddings.pt")
        torch.save(
            {
                "embeddings": embeddings,
                "targets": targets,
                "accessions": all_accessions,
                "pdb_files": all_pdb_files,
                "split": split_name,
            },
            pt_path,
        )
        log(f"[{split_name}] wrote {pt_path}")

    if save_npy:
        np.save(os.path.join(output_dir, f"{split_name}_embeddings.npy"), embeddings.numpy())
        np.save(os.path.join(output_dir, f"{split_name}_targets.npy"), targets.numpy())
        np.save(
            os.path.join(output_dir, f"{split_name}_accessions.npy"),
            np.asarray(all_accessions, dtype=object),
            allow_pickle=True,
        )
        np.save(
            os.path.join(output_dir, f"{split_name}_pdb_files.npy"),
            np.asarray(all_pdb_files, dtype=object),
            allow_pickle=True,
        )
        log(f"[{split_name}] wrote npy files")

    pd.DataFrame(
        {
            "row_idx": np.arange(len(all_accessions)),
            "accession": all_accessions,
            "pdb_file": all_pdb_files,
        }
    ).to_csv(os.path.join(output_dir, f"{split_name}_index.csv"), index=False)


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

        ds = CustomGODataset(sdf, num_labels, kwargs=kwargs)
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