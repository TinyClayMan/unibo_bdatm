import os
import re
import csv
import json
import glob
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from torchdrug import data, transforms, layers
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


def normalize_id(x) -> str:
    return str(x).strip()


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
    aspect_masks: Optional[Dict[str, np.ndarray]] = None
    aspect_top_counts: Optional[Dict[str, int]] = None


def build_labels(
    train_terms_path: str,
    num_labels: Optional[int] = None,
    aspect_top_counts: Optional[Dict[str, int]] = None
) -> LabelArtifacts:
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
    
    if num_labels is not None:
        top = train_terms["term"].value_counts().head(num_labels)
        idx_to_term = list(top.index)
        term_to_idx = {t: i for i, t in enumerate(idx_to_term)}
        
        term_aspect = (
            train_terms[["term", "aspect"]]
            .drop_duplicates()
            .groupby("term")["aspect"]
            .first()
            .to_dict()
        )
        idx_to_aspect = [str(term_aspect[t]) for t in idx_to_term]
        
    elif aspect_top_counts is not None:
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
    else:
        raise ValueError("Must provide either num_labels or aspect_top_counts")

    top_df = train_terms[train_terms["term"].isin(term_to_idx)].copy()
    top_df["term_idx"] = top_df["term"].map(term_to_idx)

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
class PDBDataset(Dataset):
    def __init__(self, df: pd.DataFrame, num_labels: int, kwargs=None, max_length: int = None, use_target_col: bool = False):
        self.df = df.reset_index(drop=True).copy()
        self.num_labels = num_labels
        self.kwargs = kwargs or {
            "atom_feature": "default",
            "bond_feature": "default",
            "residue_feature": "default",
        }
        
        self.accessions = self.df["accession"].astype(str).tolist()
        
        if "pdb_path" in self.df.columns:
            self.pdb_files = self.df["pdb_path"].tolist()
        else:
            self.pdb_files = self.df["pdb_file"].tolist()
            
        if use_target_col:
            self.targets = np.stack(self.df["target"].tolist()).astype(np.float32)
        else:
            self.targets = np.zeros((len(self.df), num_labels), dtype=np.float32)
            for i, idxs in enumerate(self.df["label_indices"].tolist()):
                self.targets[i, idxs] = 1.0

        self.transform = None
        if max_length is not None:
            self.transform = transforms.Compose([
                transforms.ProteinView("residue"),
                transforms.TruncateProtein(max_length),
            ])

    def __len__(self):
        return len(self.df)

    def get_item(self, index):
        # We handle ESM logic where features can be None
        af = self.kwargs.get("atom_feature", "default")
        bf = self.kwargs.get("bond_feature", "default")
        rf = self.kwargs.get("residue_feature", "default")
        
        protein = data.Protein.from_pdb(
            self.pdb_files[index],
            atom_feature=af,
            bond_feature=bf,
            residue_feature=rf,
        )
        
        item = {
            "graph": protein,
            "target": torch.from_numpy(self.targets[index]).float(),
            "accession": self.accessions[index],
            "pdb_file": self.pdb_files[index],
        }
        
        if self.transform:
            item = self.transform(item)
            protein = item["graph"]

        if hasattr(protein, "residue_feature") and protein.residue_feature is not None:
            with protein.residue():
                if hasattr(protein.residue_feature, "to_dense"):
                    protein.residue_feature = protein.residue_feature.to_dense()

        return item

    def __getitem__(self, index):
        return self.get_item(index)


# -----------------------------
# collation / robust iteration
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
# data loading
# -----------------------------
def load_split_pt(path):
    obj = torch.load(path, map_location="cpu")

    x = obj["embeddings"]
    y = obj["targets"]
    ids = obj["accessions"]

    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    ids = np.asarray(ids).astype(str).tolist()

    return x, y, ids


# -----------------------------
# experimental helpers
# -----------------------------
def load_download_manifest(csv_path: str):
    df = pd.read_csv(csv_path)
    required = {"accession", "status", "download_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")

    df = df.copy()
    df["accession"] = df["accession"].astype(str)
    df = df[df["status"] == "downloaded"].copy()
    df = df[df["download_path"].notna()].copy()
    df = df[df["download_path"].map(os.path.exists)].copy()
    df = df[df["download_path"].astype(str).str.lower().str.endswith(".pdb")].copy()
    df = df.drop_duplicates(subset=["accession"], keep="first").reset_index(drop=True)
    return df


def build_eval_df(reference_split_pt: str, manifest_csv: str, split_name: str):
    x_ref, y_ref, ids_ref = load_split_pt(reference_split_pt)
    manifest = load_download_manifest(manifest_csv)

    acc_to_row = {normalize_id(acc): i for i, acc in enumerate(ids_ref)}
    acc_to_pdb = {
        normalize_id(row["accession"]): row["download_path"]
        for _, row in manifest.iterrows()
    }

    rows = []
    missing_from_ref = 0
    for acc, pdb_path in sorted(acc_to_pdb.items()):
        i = acc_to_row.get(acc)
        if i is None:
            missing_from_ref += 1
            continue
        rows.append({
            "accession": acc,
            "pdb_file": pdb_path,
            "target": y_ref[i].astype(np.float32),
        })

    if not rows:
        raise RuntimeError(f"[{split_name}] no overlapping accessions between reference split and manifest")

    df = pd.DataFrame(rows)
    log(f"[{split_name}] downloaded structures in manifest: {len(manifest)}")
    log(f"[{split_name}] overlap with reference split: {len(df)}")
    if missing_from_ref:
        log(f"[{split_name}] manifest accessions absent from reference split: {missing_from_ref}")
    return df


# -----------------------------
# embedding extraction
# -----------------------------
def run_model_on_single(model, protein, device):
    packed = data.Protein.pack([protein])
    packed.view = "residue"
    packed = packed.to(device)
    emb = model(packed)
    return emb.detach().cpu()


def try_embed_batch(model, batch, device):
    graph = batch["graph"].to(device)
    emb = model(graph)
    return emb.detach().cpu()


def save_bad_rows(bad_rows, output_dir, split_name):
    if not bad_rows:
        return
    bad_path = os.path.join(output_dir, f"{split_name}_bad_samples.csv")
    pd.DataFrame(bad_rows).to_csv(bad_path, index=False)
    log(f"[{split_name}] wrote bad sample log to {bad_path}")


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
    bad_rows = []

    with torch.no_grad():
        for batch in safe_iter_loader(loader, desc=f"embed:{split_name}", dataset_name=split_name):
            try:
                emb = try_embed_batch(model, batch, device)

                all_embeddings.append(emb)
                all_targets.append(batch["target"].cpu())
                all_accessions.extend(batch["accession"])
                all_pdb_files.extend(batch["pdb_file"])
                continue

            except Exception as e:
                log(f"[{split_name}] batch forward failed, retrying sample-by-sample: {repr(e)}")

            # fallback: process one sample at a time
            batch_size = len(batch["accession"])
            recovered_embeddings = []
            recovered_targets = []
            recovered_accessions = []
            recovered_pdb_files = []

            for i in range(batch_size):
                accession = batch["accession"][i]
                pdb_file = batch["pdb_file"][i]
                target = batch["target"][i:i+1].cpu()

                try:
                    protein = batch["graph"][i]
                    emb_i = run_model_on_single(model, protein, device)

                    recovered_embeddings.append(emb_i)
                    recovered_targets.append(target)
                    recovered_accessions.append(accession)
                    recovered_pdb_files.append(pdb_file)

                except Exception as e_single:
                    log(f"[{split_name}] skipping bad sample accession={accession} pdb={pdb_file} err={repr(e_single)}")
                    bad_rows.append({
                        "accession": accession,
                        "pdb_file": pdb_file,
                        "error": repr(e_single),
                    })

            if recovered_embeddings:
                all_embeddings.append(torch.cat(recovered_embeddings, dim=0))
                all_targets.append(torch.cat(recovered_targets, dim=0))
                all_accessions.extend(recovered_accessions)
                all_pdb_files.extend(recovered_pdb_files)

    if len(all_embeddings) == 0:
        save_bad_rows(bad_rows, output_dir, split_name)
        raise RuntimeError(f"[{split_name}] no embeddings produced")

    embeddings = torch.cat(all_embeddings, dim=0).contiguous()
    targets = torch.cat(all_targets, dim=0).contiguous()

    log(f"[{split_name}] saved rows={embeddings.shape[0]} embedding_dim={embeddings.shape[1]}")
    log(f"[{split_name}] bad / skipped samples={len(bad_rows)}")

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

    save_bad_rows(bad_rows, output_dir, split_name)
