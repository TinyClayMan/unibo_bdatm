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


def load_npz_ids(path):
    z = np.load(path, allow_pickle=True)
    return [norm_id(x) for x in z["ids"]]


def load_small_split_ids(ref_dir, task, ref_task="bp"):
    task_file = ref_task if task == "all" else task

    train_path = os.path.join(ref_dir, f"{task_file}_train_ggngo_embeddings.npz")
    valid_path = os.path.join(ref_dir, f"{task_file}_val_ggngo_embeddings.npz")

    return load_npz_ids(train_path), load_npz_ids(valid_path)


def load_train_terms(path):
    df = pd.read_csv(path, sep="\t")
    df["EntryID"] = df["EntryID"].astype(str).map(norm_id)
    df["term"] = df["term"].astype(str)
    df["aspect"] = df["aspect"].map(normalize_aspect)
    return df


def select_freq_labels(train_terms, train_ids, task, top_labels):
    if task != "all":
        tasks = [task]
    else:
        tasks = TASKS

    idx_to_term = []
    idx_to_task = []
    freqs = []

    train_id_set = set(train_ids)

    for t in tasks:
        k = top_labels if top_labels is not None else TOPK[t]

        rows = train_terms[
            (train_terms["EntryID"].isin(train_id_set))
            & (train_terms["aspect"] == TASK_ASPECT[t])
        ]

        counts = rows["term"].value_counts()
        n_train = len(train_ids)

        terms = counts.head(k).index.astype(str).tolist()
        vals = (counts.head(k).values.astype(np.float32) / float(n_train)).tolist()

        idx_to_term.extend(terms)
        idx_to_task.extend([t] * len(terms))
        freqs.extend(vals)

        print(f"{t}: selected {len(terms)} labels")
        print(counts.head(10))

    return idx_to_term, idx_to_task, np.asarray(freqs, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--task", choices=["bp", "mf", "cc", "all"], required=True)
    p.add_argument("--ref_dir", default="/content/features/ggngo_embeddings")
    p.add_argument("--ref_task", choices=["bp", "mf", "cc"], default="bp")

    p.add_argument("--train_terms_tsv", required=True)
    p.add_argument("--go_obo", required=True)
    p.add_argument("--ia_txt", required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--top_labels", type=int, default=None)

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_ids, valid_ids = load_small_split_ids(
        args.ref_dir,
        args.task,
        args.ref_task,
    )

    train_terms = load_train_terms(args.train_terms_tsv)

    idx_to_term, idx_to_task, freqs = select_freq_labels(
        train_terms=train_terms,
        train_ids=train_ids,
        task=args.task,
        top_labels=args.top_labels,
    )

    # Same prediction for every validation protein:
    # score(term) = frequency in train split.
    pred = np.tile(freqs[None, :], (len(valid_ids), 1)).astype(np.float32)

    ia_dict = ia_parser(args.ia_txt)
    ontologies = {
        ns: Graph(ns, terms, ia_dict)
        for ns, terms in obo_parser(args.go_obo).items()
    }

    tau_arr = np.arange(0.01, 1.00, 0.01, dtype=np.float32)

    gts = build_ground_truth(valid_ids, train_terms, ontologies)
    pred_maps = precompute_prediction_maps(
        valid_ids,
        idx_to_term,
        ontologies,
        gts,
    )

    df_ns, mean_f, mean_wf = cafa_weighted_fmax_cached(
        pred,
        ontologies,
        gts,
        pred_maps,
        tau_arr,
    )

    print("\nFINAL NAIVE FREQUENCY BASELINE")
    if len(df_ns):
        print(df_ns.to_string(index=False))
    else:
        print("No CAFA rows.")

    print("mean_Fmax :", round(float(mean_f), 6))
    print("mean_WFmax:", round(float(mean_wf), 6))

    np.save(os.path.join(args.output_dir, "valid_pred.npy"), pred)
    np.save(
        os.path.join(args.output_dir, "valid_ids.npy"),
        np.asarray(valid_ids, dtype=object),
        allow_pickle=True,
    )

    pd.DataFrame({
        "term": idx_to_term,
        "task": idx_to_task,
        "frequency": freqs,
    }).to_csv(os.path.join(args.output_dir, "freq_labels.csv"), index=False)

    df_ns.to_csv(os.path.join(args.output_dir, "metrics_per_namespace.csv"), index=False)

    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write(f"task={args.task}\n")
        f.write(f"num_labels={len(idx_to_term)}\n")
        f.write(f"mean_Fmax={float(mean_f)}\n")
        f.write(f"mean_WFmax={float(mean_wf)}\n")

    print("saved:", args.output_dir)


if __name__ == "__main__":
    main()