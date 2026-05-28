
import os
import argparse
import numpy as np
import pandas as pd

from cafa import (
    build_ground_truth,
    precompute_prediction_maps,
    cafa_weighted_fmax_cached,
    Graph,
    obo_parser,
    ia_parser,
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


# -----------------------
# ID utils
# -----------------------
def norm_id(x):
    return str(x).strip()


def normalize_aspect(x):
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
    }.get(str(x).strip(), x)


# -----------------------
# FULL SPLIT LOADER
# -----------------------
def load_full_split(full_split_dir):
    train_csv = os.path.join(full_split_dir, "train_index.csv")
    valid_csv = os.path.join(full_split_dir, "valid_index.csv")

    if not (os.path.exists(train_csv) and os.path.exists(valid_csv)):
        raise FileNotFoundError("Missing full split CSVs")

    train = pd.read_csv(train_csv)
    valid = pd.read_csv(valid_csv)

    def pick(df):
        for c in ["EntryID", "accession", "protein_id", "id"]:
            if c in df.columns:
                return df[c].astype(str).map(norm_id).tolist()
        raise ValueError("No ID column found")

    return pick(train), pick(valid)


# -----------------------
# MINI SPLIT LOADER (.npz)
# -----------------------
def load_mini_split(ref_dir, task, ref_task="bp"):
    task_file = ref_task if task == "all" else task

    train_path = os.path.join(ref_dir, f"{task_file}_train_ggngo_embeddings.npz")
    valid_path = os.path.join(ref_dir, f"{task_file}_val_ggngo_embeddings.npz")

    def load(path):
        z = np.load(path, allow_pickle=True)
        return [norm_id(x) for x in z["ids"]]

    return load(train_path), load(valid_path)


# -----------------------
# SMART SPLIT SELECTOR
# -----------------------
def load_split(args):
    """
    One function that supports BOTH modes safely.
    """

    if args.split_mode == "same":
        print("[mode] MINI (ggngo npz split)")
        return load_mini_split(args.ref_dir, args.task, args.ref_task)

    elif args.split_mode == "all":
        print("[mode] FULL (CAFA index split)")
        return load_full_split(args.full_split_dir)

    else:
        raise ValueError(args.split_mode)


# -----------------------
# LABEL SELECTION
# -----------------------
def select_freq_labels(train_terms, train_ids, task, top_labels):
    tasks = [task] if task != "all" else TASKS

    idx_to_term, idx_to_task, freqs = [], [], []

    train_set = set(train_ids)

    for t in tasks:
        k = top_labels or TOPK[t]

        rows = train_terms[
            (train_terms["EntryID"].isin(train_set)) &
            (train_terms["aspect"] == TASK_ASPECT[t])
        ]

        counts = rows["term"].value_counts()

        terms = counts.head(k).index.astype(str).tolist()
        vals = (counts.head(k).values.astype(np.float32) / max(len(train_ids), 1)).tolist()

        idx_to_term.extend(terms)
        idx_to_task.extend([t] * len(terms))
        freqs.extend(vals)

        print(f"{t}: {len(terms)} labels")

    return idx_to_term, idx_to_task, np.asarray(freqs, dtype=np.float32)


# -----------------------
# DATA LOADER
# -----------------------
def load_train_terms(path):
    df = pd.read_csv(path, sep="\t")
    df["EntryID"] = df["EntryID"].astype(str).map(norm_id)
    df["term"] = df["term"].astype(str)
    df["aspect"] = df["aspect"].map(normalize_aspect)
    return df


# -----------------------
# MAIN
# -----------------------
def main():
    p = argparse.ArgumentParser()

    p.add_argument("--task", choices=["bp", "mf", "cc", "all"], required=True)
    p.add_argument("--split_mode", choices=["same", "all"], required=True)

    p.add_argument("--ref_dir", default="/content/features/ggngo_embeddings")
    p.add_argument("--ref_task", default="bp")

    p.add_argument("--full_split_dir", default="/content/gearnet_embeds")

    p.add_argument("--train_terms_tsv", required=True)
    p.add_argument("--go_obo", required=True)
    p.add_argument("--ia_txt", required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--top_labels", type=int, default=None)

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------
    # SPLIT (UNIFIED)
    # -----------------------
    train_ids, valid_ids = load_split(args)

    print("\nSplit stats")
    print("train:", len(train_ids))
    print("valid:", len(valid_ids))

    train_terms = load_train_terms(args.train_terms_tsv)

    # IMPORTANT SAFETY CHECK
    overlap = len(set(valid_ids) & set(train_terms["EntryID"]))
    print("valid ∩ annotations:", overlap)

    if overlap == 0:
        raise RuntimeError(
            "No overlap between valid IDs and train_terms.tsv.\n"
            "You are mixing datasets."
        )

    # -----------------------
    # LABELS
    # -----------------------
    idx_to_term, idx_to_task, freqs = select_freq_labels(
        train_terms, train_ids, args.task, args.top_labels
    )

    # -----------------------
    # PREDICTION
    # -----------------------
    pred = np.tile(freqs[None, :], (len(valid_ids), 1)).astype(np.float32)

    # -----------------------
    # CAFA EVAL
    # -----------------------
    ia_dict = ia_parser(args.ia_txt)

    ontologies = {
        ns: Graph(ns, terms, ia_dict)
        for ns, terms in obo_parser(args.go_obo).items()
    }

    tau_arr = np.arange(0.01, 1.0, 0.01, dtype=np.float32)

    gts = build_ground_truth(valid_ids, train_terms, ontologies)

    pred_maps = precompute_prediction_maps(valid_ids, idx_to_term, ontologies, gts)

    df_ns, mean_f, mean_wf = cafa_weighted_fmax_cached(
        pred,
        ontologies,
        gts,
        pred_maps,
        tau_arr,
    )

    print("\nRESULTS")
    print(df_ns.to_string(index=False) if len(df_ns) else "No CAFA rows")

    print("Fmax :", float(mean_f))
    print("WFmax:", float(mean_wf))

    np.save(os.path.join(args.output_dir, "valid_pred.npy"), pred)
    np.save(os.path.join(args.output_dir, "valid_ids.npy"), np.array(valid_ids, dtype=object))

    print("saved:", args.output_dir)


if __name__ == "__main__":
    main()