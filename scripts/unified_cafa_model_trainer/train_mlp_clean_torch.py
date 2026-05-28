
import os
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from cafa import (
    load_split_pt,
    build_ground_truth,
    precompute_prediction_maps,
    cafa_weighted_fmax_cached,
    CafaApproxCallback,
    Graph,
    obo_parser,
    ia_parser,
    set_seed,
)


TASKS = ["bp", "mf", "cc"]

TASK_ASPECT = {
    "bp": "BPO",
    "mf": "MFO",
    "cc": "CCO",
}

TOPK = {
    "bp": 1100,
    "mf": 450,
    "cc": 300,
}


# ============================================================
# Basic
# ============================================================

def norm_id(x):
    return str(x).strip()


def normalize_aspect(x):
    x = str(x).strip()
    return {
        "P": "BPO",
        "F": "MFO",
        "C": "CCO",
        "bp": "BPO",
        "mf": "MFO",
        "cc": "CCO",
        "biological_process": "BPO",
        "molecular_function": "MFO",
        "cellular_component": "CCO",
    }.get(x, x)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def load_train_terms(path):
    df = pd.read_csv(path, sep="\t")

    required = {"EntryID", "term", "aspect"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train_terms.tsv missing columns: {missing}")

    df["EntryID"] = df["EntryID"].astype(str).map(norm_id)
    df["term"] = df["term"].astype(str)
    df["aspect"] = df["aspect"].map(normalize_aspect)

    return df


# ============================================================
# Split IDs
# ============================================================

def load_npz_ids(path):
    z = np.load(path, allow_pickle=True)
    return [norm_id(x) for x in z["ids"]]


def load_pt_ids(path):
    _, _, ids = load_split_pt(path)
    return [norm_id(x) for x in ids]


def load_csv_ids(path):
    df = pd.read_csv(path)

    for col in ["accession", "EntryID", "protein_id", "id"]:
        if col in df.columns:
            return [norm_id(x) for x in df[col].tolist()]

    raise ValueError(f"No accession column in {path}. Columns: {list(df.columns)}")


def load_reference_ids(args):
    """
    Returns only split accession IDs.

    split_mode=same:
        use mini split from GGN-GO npz IDs.

    split_mode=all:
        use full split IDs from CSV if available, otherwise .pt accessions.
    """
    if args.split_mode == "same":
        ref_task = args.ref_task

        train_npz = os.path.join(args.ref_dir, f"{ref_task}_train_ggngo_embeddings.npz")
        valid_npz = os.path.join(args.ref_dir, f"{ref_task}_val_ggngo_embeddings.npz")

        return load_npz_ids(train_npz), load_npz_ids(valid_npz)

    if args.split_mode == "all":
        train_csv = os.path.join(args.full_split_dir, "train_index.csv")
        valid_csv = os.path.join(args.full_split_dir, "valid_index.csv")

        if os.path.exists(train_csv) and os.path.exists(valid_csv):
            return load_csv_ids(train_csv), load_csv_ids(valid_csv)

        small_train_csv = os.path.join(args.full_split_dir, "small_train_index.csv")

        if os.path.exists(small_train_csv) and os.path.exists(valid_csv):
            return load_csv_ids(small_train_csv), load_csv_ids(valid_csv)

        train_pt = os.path.join(args.full_split_dir, "train_embeddings.pt")
        valid_pt = os.path.join(args.full_split_dir, "valid_embeddings.pt")

        if os.path.exists(train_pt) and os.path.exists(valid_pt):
            return load_pt_ids(train_pt), load_pt_ids(valid_pt)

        raise FileNotFoundError(
            "Could not find split accession files. Checked:\n"
            f"  {train_csv}\n"
            f"  {valid_csv}\n"
            f"  {small_train_csv}\n"
            f"  {train_pt}\n"
            f"  {valid_pt}"
        )

    raise ValueError(args.split_mode)


# ============================================================
# Embedding loading
# ============================================================

def gather_by_ids(want_ids, source_ids, source_x, split_name):
    """
    Correct accession-safe gather.

    source_ids[i] corresponds to source_x[i].
    want_ids defines the requested split/order.
    """
    source_ids = [norm_id(x) for x in source_ids]

    if len(source_ids) != source_x.shape[0]:
        raise ValueError(
            f"{split_name}: IDs/embeddings mismatch: "
            f"{len(source_ids)} IDs vs {source_x.shape[0]} rows"
        )

    if len(set(source_ids)) != len(source_ids):
        vc = pd.Series(source_ids).value_counts()
        dup = vc[vc > 1].head(20)
        raise ValueError(f"{split_name}: duplicate source IDs:\n{dup}")

    pos = {pid: i for i, pid in enumerate(source_ids)}

    rows = []
    kept_ids = []
    missing = []

    for pid in want_ids:
        pid = norm_id(pid)
        j = pos.get(pid)

        if j is None:
            missing.append(pid)
        else:
            rows.append(j)
            kept_ids.append(pid)

    if not rows:
        raise RuntimeError(f"{split_name}: matched zero embeddings")

    x = source_x[np.asarray(rows, dtype=np.int64)].astype(np.float32)

    print(f"{split_name}: matched {len(kept_ids)}/{len(want_ids)} embeddings; missing={len(missing)}")
    if missing:
        print(f"{split_name}: first missing IDs: {missing[:10]}")

    for k in range(min(5, len(kept_ids))):
        assert source_ids[rows[k]] == kept_ids[k]

    return x, kept_ids


def load_t5_embeddings(args, train_ref_ids, valid_ref_ids):
    if args.ids_npy is None or args.embeds_npy is None:
        raise ValueError("--source t5 needs --ids_npy and --embeds_npy")

    source_ids = np.load(args.ids_npy, allow_pickle=True).astype(str).tolist()
    source_ids = [norm_id(x) for x in source_ids]

    source_x = np.load(args.embeds_npy).astype(np.float32)

    x_train, train_ids = gather_by_ids(train_ref_ids, source_ids, source_x, "train")
    x_valid, valid_ids = gather_by_ids(valid_ref_ids, source_ids, source_x, "valid")

    return x_train, train_ids, x_valid, valid_ids


def load_pt_embeddings(path):
    x, _, ids = load_split_pt(path)
    return x.astype(np.float32), [norm_id(z) for z in ids]


def load_split_pt_embeddings(args, train_ref_ids, valid_ref_ids):
    if args.embed_dir is None and (args.train_pt is None or args.valid_pt is None):
        raise ValueError("--source pt needs --embed_dir or explicit --train_pt/--valid_pt")

    train_pt = args.train_pt or os.path.join(args.embed_dir, "train_embeddings.pt")
    valid_pt = args.valid_pt or os.path.join(args.embed_dir, "valid_embeddings.pt")

    x_train_all, train_ids_all = load_pt_embeddings(train_pt)
    x_valid_all, valid_ids_all = load_pt_embeddings(valid_pt)

    x_train, train_ids = gather_by_ids(train_ref_ids, train_ids_all, x_train_all, "train")
    x_valid, valid_ids = gather_by_ids(valid_ref_ids, valid_ids_all, x_valid_all, "valid")

    return x_train, train_ids, x_valid, valid_ids


def load_npz_embeddings(path):
    z = np.load(path, allow_pickle=True)
    x = z["embeddings"].astype(np.float32)
    ids = [norm_id(x) for x in z["ids"]]
    return x, ids


def load_split_npz_embeddings(args, train_ref_ids, valid_ref_ids):
    """
    For GGN-GO embeddings, task='all' still needs one embedding source.

    Use --npz_task bp/mf/cc to choose which GGN-GO backbone embeddings
    to train the all-task head on.

    Default: bp.
    """
    if args.embed_dir is None and (args.train_npz is None or args.valid_npz is None):
        raise ValueError("--source npz needs --embed_dir or explicit --train_npz/--valid_npz")

    emb_task = args.npz_task if args.task == "all" else args.task

    train_npz = args.train_npz or os.path.join(
        args.embed_dir,
        f"{emb_task}_train_ggngo_embeddings.npz",
    )
    valid_npz = args.valid_npz or os.path.join(
        args.embed_dir,
        f"{emb_task}_val_ggngo_embeddings.npz",
    )

    x_train_all, train_ids_all = load_npz_embeddings(train_npz)
    x_valid_all, valid_ids_all = load_npz_embeddings(valid_npz)

    x_train, train_ids = gather_by_ids(train_ref_ids, train_ids_all, x_train_all, "train")
    x_valid, valid_ids = gather_by_ids(valid_ref_ids, valid_ids_all, x_valid_all, "valid")

    return x_train, train_ids, x_valid, valid_ids


def load_features(args, train_ref_ids, valid_ref_ids):
    if args.source == "t5":
        return load_t5_embeddings(args, train_ref_ids, valid_ref_ids)

    if args.source == "pt":
        return load_split_pt_embeddings(args, train_ref_ids, valid_ref_ids)

    if args.source == "npz":
        return load_split_npz_embeddings(args, train_ref_ids, valid_ref_ids)

    raise ValueError(args.source)

def standardize_features(x_train, x_valid, eps=1e-6):
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)

    x_train = (x_train - mean) / std
    x_valid = (x_valid - mean) / std

    return x_train.astype(np.float32), x_valid.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)

# ============================================================
# Direct labels
# ============================================================

def select_terms_for_task(train_terms, train_ids, task, top_k):
    rows = train_terms[
        (train_terms["EntryID"].isin(train_ids))
        & (train_terms["aspect"] == TASK_ASPECT[task])
    ]

    counts = rows["term"].value_counts()
    terms = counts.head(top_k).index.astype(str).tolist()

    print(f"\n{task}: selected {len(terms)} labels")
    print(counts.head(10))

    return terms


def build_label_space(train_terms, train_ids, task, top_labels):
    if task != "all":
        k = top_labels or TOPK[task]
        terms = select_terms_for_task(train_terms, train_ids, task, k)
        return terms, [task] * len(terms)

    idx_to_term = []
    idx_to_task = []

    for t in TASKS:
        k = top_labels or TOPK[t]
        terms = select_terms_for_task(train_terms, train_ids, t, k)
        idx_to_term.extend(terms)
        idx_to_task.extend([t] * len(terms))

    print("\nall-task label space")
    print("  total:", len(idx_to_term))
    print("  bp:", idx_to_task.count("bp"))
    print("  mf:", idx_to_task.count("mf"))
    print("  cc:", idx_to_task.count("cc"))

    return idx_to_term, idx_to_task


def make_direct_labels(ids, train_terms, idx_to_term, idx_to_task):
    """
    Direct labels only.

    y[i, j] = 1 iff ids[i] has direct GO annotation idx_to_term[j]
    in train_terms.tsv with matching ontology.
    """
    y = np.zeros((len(ids), len(idx_to_term)), dtype=np.float32)

    id_to_row = {pid: i for i, pid in enumerate(ids)}
    term_task_to_col = {
        (term, task): j
        for j, (term, task) in enumerate(zip(idx_to_term, idx_to_task))
    }

    rows = train_terms[
        train_terms["EntryID"].isin(id_to_row)
    ][["EntryID", "term", "aspect"]]

    used = 0

    for pid, term, aspect in rows.itertuples(index=False):
        if aspect == "BPO":
            task = "bp"
        elif aspect == "MFO":
            task = "mf"
        elif aspect == "CCO":
            task = "cc"
        else:
            continue

        col = term_task_to_col.get((term, task))
        if col is None:
            continue

        y[id_to_row[pid], col] = 1.0
        used += 1

    print("direct labels used:", used)
    print("positive cells:", int(y.sum()))
    print("zero-label proteins:", int((y.sum(axis=1) == 0).sum()), "/", len(ids))
    print("empty label columns:", int((y.sum(axis=0) == 0).sum()), "/", y.shape[1])

    return y


# ============================================================
# Heads
# ============================================================

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),

            nn.Linear(input_dim, 1024, bias=False),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(1024, 1024, bias=False),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(1024, 512, bias=False),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class GGNHead(nn.Module):
    """
    Original GGN-GO style prediction head, without final Sigmoid.

    Original:
        Linear(input, 1024) -> ReLU -> Dropout(0.2)
        -> Linear(1024, out_dim) -> Sigmoid

    Here:
        no Sigmoid because we use BCEWithLogitsLoss.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Train / predict
# ============================================================

def train_epoch(model, loader, optimizer, loss_fn, device, scheduler=None, grad_clip=5.0):
    model.train()
    total = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = loss_fn(logits, y)

        loss.backward()

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        bs = x.shape[0]
        total += float(loss.item()) * bs
        n += bs

    return total / max(n, 1)


@torch.no_grad()
def eval_loss(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()

        loss = loss_fn(model(x), y)

        bs = x.shape[0]
        total += float(loss.item()) * bs
        n += bs

    return total / max(n, 1)


@torch.no_grad()
def predict(model, x, device, batch_size):
    model.eval()
    preds = []

    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i:i + batch_size]).to(device).float()
        prob = torch.sigmoid(model(xb)).cpu().numpy()
        preds.append(prob)

    return np.concatenate(preds, axis=0).astype(np.float32)


# ============================================================
# CAFA eval
# ============================================================

def evaluate_cafa(pred, valid_ids, idx_to_term, train_terms, args):
    ia_dict = ia_parser(args.ia_txt)

    ontologies = {
        ns: Graph(ns, terms, ia_dict)
        for ns, terms in obo_parser(args.go_obo).items()
    }

    tau_arr = np.arange(0.01, 1.00, 0.01, dtype=np.float32)

    gts = build_ground_truth(valid_ids, train_terms, ontologies)
    pred_maps = precompute_prediction_maps(valid_ids, idx_to_term, ontologies, gts)

    return cafa_weighted_fmax_cached(
        pred,
        ontologies,
        gts,
        pred_maps,
        tau_arr,
    )


def save_loss_plot(history, output_dir, title="loss"):
    try:
        import matplotlib.pyplot as plt

        hist = pd.DataFrame(history)
        if not len(hist):
            return

        plt.figure(figsize=(7, 4))
        plt.plot(hist["epoch"], hist["train_loss"], label="train")
        plt.plot(hist["epoch"], hist["valid_loss"], label="valid")

        if "best_epoch" in hist.columns:
            best_epoch = int(hist["best_epoch"].iloc[-1])
            plt.axvline(best_epoch, linestyle="--", color="black", label=f"best {best_epoch}")

        plt.xlabel("epoch")
        plt.ylabel("BCEWithLogitsLoss")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=200)
        plt.close()

    except Exception as e:
        print("Could not save loss plot:", e)


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--task", choices=["bp", "mf", "cc", "all"], required=True)
    p.add_argument("--source", choices=["t5", "pt", "npz"], required=True)
    p.add_argument("--split_mode", choices=["same", "all"], default="same")

    p.add_argument("--train_terms_tsv", required=True)
    p.add_argument("--go_obo", required=True)
    p.add_argument("--ia_txt", required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--ref_dir", default="/content/features/ggngo_embeddings")
    p.add_argument("--ref_task", choices=["bp", "mf", "cc"], default="bp")

    p.add_argument("--full_split_dir", default="/content/gearnet_embeds")
    p.add_argument("--embed_dir", default=None)

    p.add_argument("--ids_npy", default=None)
    p.add_argument("--embeds_npy", default=None)

    p.add_argument("--train_pt", default=None)
    p.add_argument("--valid_pt", default=None)

    p.add_argument("--train_npz", default=None)
    p.add_argument("--valid_npz", default=None)
    p.add_argument("--npz_task", choices=["bp", "mf", "cc"], default="bp")

    p.add_argument("--top_labels", type=int, default=None)
    p.add_argument("--head", choices=["mlp", "ggn"], default="ggn")

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=5120)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--optimizer", choices=["adamw", "adam"], default="adamw")
    p.add_argument("--scheduler", choices=["none", "onecycle", "plateau"], default="onecycle")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=5.0)

    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--min_delta", type=float, default=1e-5)
    p.add_argument("--eval_every", type=int, default=0)

    p.add_argument("--standardize", action="store_true")

    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    train_terms = load_train_terms(args.train_terms_tsv)

    train_ref_ids, valid_ref_ids = load_reference_ids(args)

    x_train, train_ids, x_valid, valid_ids = load_features(
        args,
        train_ref_ids,
        valid_ref_ids,
    )
    scaler_mean = None
    scaler_std = None

    if args.standardize:
        print("\nStandardizing features using train mean/std...")
        x_train, x_valid, scaler_mean, scaler_std = standardize_features(x_train, x_valid)
        print("  after standardization train mean/std:", float(x_train.mean()), float(x_train.std()))
        print("  after standardization valid mean/std:", float(x_valid.mean()), float(x_valid.std()))

    idx_to_term, idx_to_task = build_label_space(
        train_terms=train_terms,
        train_ids=train_ids,
        task=args.task,
        top_labels=args.top_labels,
    )

    y_train = make_direct_labels(
        ids=train_ids,
        train_terms=train_terms,
        idx_to_term=idx_to_term,
        idx_to_task=idx_to_task,
    )

    y_valid = make_direct_labels(
        ids=valid_ids,
        train_terms=train_terms,
        idx_to_term=idx_to_term,
        idx_to_task=idx_to_task,
    )

    print("\nData")
    print("  x_train:", x_train.shape)
    print("  y_train:", y_train.shape, "positives:", int(y_train.sum()))
    print("  x_valid:", x_valid.shape)
    print("  y_valid:", y_valid.shape, "positives:", int(y_valid.sum()))

    print("\nSanity")
    print("  first train IDs:", train_ids[:5])
    print("  first valid IDs:", valid_ids[:5])
    print("  x_train finite:", bool(np.isfinite(x_train).all()))
    print("  x_valid finite:", bool(np.isfinite(x_valid).all()))
    print("  x_train mean/std:", float(x_train.mean()), float(x_train.std()))
    print("  y positive rate:", float(y_train.mean()))

    if not np.isfinite(x_train).all():
        raise ValueError("x_train has NaN/Inf")
    if y_train.sum() == 0:
        raise ValueError("y_train is all zero")

    if args.head == "mlp":
        model = MLP(x_train.shape[1], y_train.shape[1])
    else:
        model = GGNHead(x_train.shape[1], y_train.shape[1])

    model = model.to(device)

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    valid_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid)),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    step_scheduler = None
    plateau_scheduler = None

    if args.scheduler == "onecycle":
        step_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.08,
            div_factor=25.0,
            final_div_factor=1000.0,
        )
    elif args.scheduler == "plateau":
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )

    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_path = os.path.join(args.output_dir, "best.pt")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            scheduler=step_scheduler,
            grad_clip=args.grad_clip,
        )

        valid_loss = eval_loss(model, valid_loader, loss_fn, device)

        if plateau_scheduler is not None:
            plateau_scheduler.step(valid_loss)

        lr_now = float(optimizer.param_groups[0]["lr"])

        improved = valid_loss < best_loss - args.min_delta
        if improved:
            best_loss = float(valid_loss)
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1

        epoch_fmax = None
        epoch_wfmax = None

        if args.eval_every > 0 and (epoch == 1 or epoch % args.eval_every == 0):
            tmp_pred = predict(model, x_valid, device, args.batch_size)
            tmp_df, tmp_f, tmp_wf = evaluate_cafa(
                pred=tmp_pred,
                valid_ids=valid_ids,
                idx_to_term=idx_to_term,
                train_terms=train_terms,
                args=args,
            )
            epoch_fmax = float(tmp_f)
            epoch_wfmax = float(tmp_wf)
            print(f"    CAFA valid Fmax={epoch_fmax:.6f} WFmax={epoch_wfmax:.6f}")

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "valid_loss": float(valid_loss),
            "best_loss": float(best_loss),
            "best_epoch": int(best_epoch),
            "bad_epochs": int(bad_epochs),
            "lr": lr_now,
            "valid_fmax": epoch_fmax,
            "valid_wfmax": epoch_wfmax,
        })

        print(
            f"{epoch:03d} "
            f"train={train_loss:.6f} "
            f"valid={valid_loss:.6f} "
            f"best={best_loss:.6f}@{best_epoch} "
            f"lr={lr_now:.2e} "
            f"bad={bad_epochs}/{args.patience}"
        )

        if bad_epochs >= args.patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch: {best_epoch}, best valid loss: {best_loss:.6f}"
            )
            break

    model.load_state_dict(torch.load(best_path, map_location=device))

    valid_pred = predict(model, x_valid, device, args.batch_size)

    df, mean_f, mean_wf = evaluate_cafa(
        pred=valid_pred,
        valid_ids=valid_ids,
        idx_to_term=idx_to_term,
        train_terms=train_terms,
        args=args,
    )

    print("\nFINAL VALID")
    if len(df):
        print(df.to_string(index=False))
    else:
        print("No CAFA rows.")

    print("mean_Fmax :", round(float(mean_f), 6))
    print("mean_WFmax:", round(float(mean_wf), 6))
    print("best_epoch:", best_epoch)
    print("best_valid_loss:", round(float(best_loss), 6))

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(args.output_dir, "history.csv"), index=False)

    save_loss_plot(
        history,
        args.output_dir,
        title=f"{args.source} {args.task} {args.head} {args.scheduler}",
    )

    np.save(os.path.join(args.output_dir, "valid_pred.npy"), valid_pred)
    np.save(
        os.path.join(args.output_dir, "valid_ids.npy"),
        np.asarray(valid_ids, dtype=object),
        allow_pickle=True,
    )

    with open(os.path.join(args.output_dir, "label_vocab.json"), "w") as f:
        json.dump(
            {
                "task": args.task,
                "idx_to_term": idx_to_term,
                "idx_to_task": idx_to_task,
                "idx_to_aspect": [TASK_ASPECT[t] for t in idx_to_task],
            },
            f,
            indent=2,
        )

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "task": args.task,
                "source": args.source,
                "split_mode": args.split_mode,
                "head": args.head,
                "scheduler": args.scheduler,
                "optimizer": args.optimizer,
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "num_labels": int(len(idx_to_term)),
                "train_shape": list(x_train.shape),
                "valid_shape": list(x_valid.shape),
                "train_positives": int(y_train.sum()),
                "valid_positives": int(y_valid.sum()),
                "best_epoch": int(best_epoch),
                "best_valid_loss": float(best_loss),
                "mean_Fmax": float(mean_f),
                "mean_WFmax": float(mean_wf),
                "per_namespace": df.to_dict("records"),
            },
            f,
            indent=2,
        )

    print("\nsaved:", args.output_dir)


if __name__ == "__main__":
    main()