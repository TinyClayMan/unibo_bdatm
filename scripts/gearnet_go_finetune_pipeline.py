# Full GearNet GO fine-tuning pipeline for custom PDB shards + multi-hot GO labels
# Robust version:
# - correct AlphaFold accession parsing: AF-<EntryID>-F1.pdb -> <EntryID>
# - recursive shard discovery
# - skips rare bad / unreadable PDBs instead of crashing
# - tqdm progress bars for sample discovery, train / valid / test loops
# - streamed logging friendly for Colab subprocess execution

import os
import re
import gc
import json
import math
import glob
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# metrics
# -----------------------------

def f1_from_counts(tp: float, fp: float, fn: float, eps: float = 1e-12) -> float:
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    return float(2 * p * r / (p + r + eps))


def micro_f1_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> float:
    y_pred = (y_score >= thr).astype(np.int64)
    tp = (y_pred * y_true).sum()
    fp = (y_pred * (1 - y_true)).sum()
    fn = ((1 - y_pred) * y_true).sum()
    return f1_from_counts(tp, fp, fn)


def macro_f1_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> float:
    y_pred = (y_score >= thr).astype(np.int64)
    f1s = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        tp = (yp * yt).sum()
        fp = (yp * (1 - yt)).sum()
        fn = ((1 - yp) * yt).sum()
        if yt.sum() == 0 and yp.sum() == 0:
            continue
        f1s.append(f1_from_counts(tp, fp, fn))
    if not f1s:
        return 0.0
    return float(np.mean(f1s))


def fmax_micro_macro(y_true: np.ndarray, y_score: np.ndarray, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    best_micro = (0.0, 0.5)
    best_macro = (0.0, 0.5)
    for thr in thresholds:
        micro = micro_f1_at_threshold(y_true, y_score, thr)
        macro = macro_f1_at_threshold(y_true, y_score, thr)
        if micro > best_micro[0]:
            best_micro = (micro, float(thr))
        if macro > best_macro[0]:
            best_macro = (macro, float(thr))
    return {
        "f1_micro": best_micro[0],
        "f1_micro_thr": best_micro[1],
        "f1_macro": best_macro[0],
        "f1_macro_thr": best_macro[1],
    }


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


# -----------------------------
# labels and sample building
# -----------------------------
@dataclass
class LabelArtifacts:
    entry_to_label_idxs: Dict[str, List[int]]
    idx_to_term: List[str]
    idx_to_aspect: List[str]
    term_to_idx: Dict[str, int]
    aspect_masks: Dict[str, np.ndarray]


def build_labels(train_terms_path: str, num_labels: int) -> LabelArtifacts:
    train_terms = pd.read_csv(train_terms_path, sep="\t")
    required = {"EntryID", "term", "aspect"}
    missing = required - set(train_terms.columns)
    if missing:
        raise ValueError(f"Missing columns in train_terms.tsv: {missing}")

    train_terms = train_terms.copy()
    train_terms["EntryID"] = train_terms["EntryID"].astype(str)

    top = train_terms["term"].value_counts().head(num_labels)
    idx_to_term = list(top.index)
    term_to_idx = {t: i for i, t in enumerate(idx_to_term)}

    top_df = train_terms[train_terms["term"].isin(term_to_idx)].copy()
    top_df["term_idx"] = top_df["term"].map(term_to_idx)

    term_aspect = (
        top_df[["term", "aspect"]]
        .drop_duplicates()
        .groupby("term")["aspect"]
        .first()
        .to_dict()
    )
    idx_to_aspect = [str(term_aspect[t]) for t in idx_to_term]

    entry_to_label_idxs = (
        top_df.groupby("EntryID")["term_idx"]
        .apply(lambda s: sorted(set(int(x) for x in s.tolist())))
        .to_dict()
    )

    aspect_masks = {}
    for aspect in ["BPO", "MFO", "CCO", "BP", "MF", "CC"]:
        mask = np.array([a == aspect for a in idx_to_aspect], dtype=bool)
        if mask.any():
            aspect_masks[aspect] = mask

    return LabelArtifacts(
        entry_to_label_idxs=entry_to_label_idxs,
        idx_to_term=idx_to_term,
        idx_to_aspect=idx_to_aspect,
        term_to_idx=term_to_idx,
        aspect_masks=aspect_masks,
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
class GearNetGOFinetuner(nn.Module):
    def __init__(
        self,
        num_labels: int,
        checkpoint_path: str,
        mlp_hidden: int = 1024,
        dropout: float = 0.2,
    ):
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

        graph_dim = 512 * 6
        self.head = nn.Sequential(
            nn.Linear(graph_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_labels),
        )

    def forward(self, packed_protein):
        packed_protein.view = "residue"
        graph = self.graph_construction_model(packed_protein)
        node_input = graph.node_feature.float()
        out = self.encoder(graph, node_input)
        logits = self.head(out["graph_feature"])
        return logits


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
# evaluation / training
# -----------------------------
def evaluate(model, loader, device, aspect_masks: Dict[str, np.ndarray], split_name: str):
    model.eval()
    ys = []
    ps = []
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for batch in safe_iter_loader(loader, desc=f"eval:{split_name}", dataset_name=split_name):
            graph = batch["graph"].to(device)
            target = batch["target"].to(device)
            logits = model(graph)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            prob = torch.sigmoid(logits)
            bs = target.size(0)
            total_loss += loss.item() * bs
            total_n += bs
            ys.append(target.cpu().numpy())
            ps.append(prob.cpu().numpy())

    if total_n == 0:
        return {"loss": float("nan"), "f1_micro": 0.0, "f1_micro_thr": 0.5, "f1_macro": 0.0, "f1_macro_thr": 0.5}

    y_true = np.concatenate(ys, axis=0)
    y_score = np.concatenate(ps, axis=0)

    metrics = {"loss": total_loss / max(total_n, 1)}
    metrics.update(fmax_micro_macro(y_true, y_score))
    for aspect, mask in aspect_masks.items():
        if mask.sum() == 0:
            continue
        sub = fmax_micro_macro(y_true[:, mask], y_score[:, mask])
        metrics[f"{aspect}_f1_micro"] = sub["f1_micro"]
        metrics[f"{aspect}_f1_macro"] = sub["f1_macro"]
    return metrics


def train_one_epoch(model, loader, optimizer, device, pos_weight=None, grad_clip=None, epoch: int = 0):
    model.train()
    total_loss = 0.0
    total_n = 0

    for step, batch in enumerate(safe_iter_loader(loader, desc=f"train:epoch{epoch}", dataset_name=f"train-e{epoch}"), start=1):
        graph = batch["graph"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(graph)
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = target.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        if step % 50 == 0:
            log(f"train epoch={epoch} step={step}/{len(loader)} loss={loss.item():.4f}")

    return {"loss": total_loss / max(total_n, 1)}


def format_metrics(prefix, metrics):
    keys = [
        "loss", "f1_micro", "f1_macro",
        "BP_f1_micro", "BP_f1_macro", "MF_f1_micro", "MF_f1_macro", "CC_f1_micro", "CC_f1_macro",
        "BPO_f1_micro", "BPO_f1_macro", "MFO_f1_micro", "MFO_f1_macro", "CCO_f1_micro", "CCO_f1_macro",
    ]
    parts = []
    for k in keys:
        if k in metrics:
            parts.append(f"{prefix}_{k}={metrics[k]:.4f}")
    return " ".join(parts)


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
    parser.add_argument("--num_labels", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr_encoder", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--freeze_encoder_epochs", type=int, default=0)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    set_seed(args.seed)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(args.output_dir, "torch_extensions"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}")

    labels = build_labels(args.train_terms_tsv, args.num_labels)
    df = build_samples(
        pdb_root=args.pdb_root,
        labels=labels,
        taxonomy_path=args.train_taxonomy_tsv,
        require_label=True,
    )
    df, y = add_multihot_columns(df, args.num_labels)
    df = make_split(df, y, seed=args.seed)

    split_path = os.path.join(args.output_dir, "splits.csv")
    df.to_csv(split_path, index=False)
    with open(os.path.join(args.output_dir, "label_vocab.json"), "w") as f:
        json.dump(
            {
                "idx_to_term": labels.idx_to_term,
                "idx_to_aspect": labels.idx_to_aspect,
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
    train_ds = CustomGODataset(df[df["split"] == "train"], args.num_labels, kwargs=kwargs)
    valid_ds = CustomGODataset(df[df["split"] == "valid"], args.num_labels, kwargs=kwargs)
    test_ds = CustomGODataset(df[df["split"] == "test"], args.num_labels, kwargs=kwargs)

    collate_fn = make_collate_fn()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
    )

    model = GearNetGOFinetuner(args.num_labels, args.checkpoint).to(device)

    if args.freeze_encoder_epochs > 0:
        for p in model.encoder.parameters():
            p.requires_grad = False

    y_train = train_ds.targets
    pos = y_train.sum(axis=0)
    neg = len(y_train) - pos
    pos_weight = torch.tensor((neg + 1.0) / (pos + 1.0), dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr_encoder},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.6, patience=3
    )

    start_epoch = 1
    best_key = "f1_macro"
    best_score = -1.0
    best_path = os.path.join(args.output_dir, "best_model.pt")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt.get("best_score", -1.0)
        log(f"Resumed from epoch {ckpt['epoch']}")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        if args.freeze_encoder_epochs > 0 and epoch == args.freeze_encoder_epochs + 1:
            for p in model.encoder.parameters():
                p.requires_grad = True
            log("Encoder unfrozen")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            pos_weight=pos_weight, grad_clip=args.grad_clip, epoch=epoch
        )
        valid_metrics = evaluate(model, valid_loader, device, labels.aspect_masks, split_name="valid")
        test_metrics = evaluate(model, test_loader, device, labels.aspect_masks, split_name="test")

        scheduler.step(valid_metrics[best_key])

        lr_enc = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]
        dt = time.time() - epoch_start
        log(
            f"epoch={epoch} seconds={dt:.1f} lr_enc={lr_enc:.2e} lr_head={lr_head:.2e} "
            f"{format_metrics('train', train_metrics)} {format_metrics('valid', valid_metrics)} {format_metrics('test', test_metrics)}"
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_score": best_score,
        }
        torch.save(state, os.path.join(args.output_dir, "last_model.pt"))

        if valid_metrics[best_key] > best_score:
            best_score = valid_metrics[best_key]
            state["best_score"] = best_score
            torch.save(state, best_path)
            log(f"Saved best model at epoch {epoch} with valid_{best_key}={best_score:.4f}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best = torch.load(best_path, map_location="cpu")
    model.load_state_dict(best["model"])
    model.to(device)
    final_valid = evaluate(model, valid_loader, device, labels.aspect_masks, split_name="valid-best")
    final_test = evaluate(model, test_loader, device, labels.aspect_masks, split_name="test-best")
    log("Final(best) valid: " + json.dumps(final_valid, indent=2))
    log("Final(best) test: " + json.dumps(final_test, indent=2))


if __name__ == "__main__":
    main()