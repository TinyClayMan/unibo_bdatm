
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cafa import (
    build_ground_truth,
    precompute_prediction_maps,
    cafa_weighted_fmax_cached,
    Graph,
    obo_parser,
    ia_parser,
)

# The prediction heads are defined once in the trainer and imported here
from train_mlp_clean_torch import MLP, GGNHead


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


def task_to_aspect(task):
    return {
        "bp": "BPO",
        "mf": "MFO",
        "cc": "CCO",
    }[task]


def aspect_to_task(aspect):
    return {
        "BPO": "bp",
        "MFO": "mf",
        "CCO": "cc",
    }.get(aspect, None)


def load_train_terms(path):
    df = pd.read_csv(path, sep="\t")
    df["EntryID"] = df["EntryID"].astype(str).map(norm_id)
    df["term"] = df["term"].astype(str)
    df["aspect"] = df["aspect"].map(normalize_aspect)
    return df


def load_pt(path):
    obj = torch.load(path, map_location="cpu")

    x = obj["embeddings"]
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    x = np.asarray(x, dtype=np.float32)

    ids = [norm_id(x) for x in obj["accessions"]]

    if x.shape[0] != len(ids):
        raise ValueError(f"{path}: embeddings/accessions mismatch")

    return x, ids


def gather(ids_wanted, x_source, ids_source):
    pos = {pid: i for i, pid in enumerate(ids_source)}
    rows = [pos[pid] for pid in ids_wanted]
    return x_source[np.asarray(rows, dtype=np.int64)].astype(np.float32)


def make_frequency_baseline(train_terms, train_ids, idx_to_term, idx_to_task):
    """
    One constant prediction vector for all eval proteins.

    score[j] = fraction of training proteins annotated with idx_to_term[j]
               in the matching ontology.
    """
    train_ids = [norm_id(x) for x in train_ids]
    train_set = set(train_ids)

    rows = train_terms[train_terms["EntryID"].isin(train_set)]

    n_train = max(len(train_ids), 1)
    scores = np.zeros(len(idx_to_term), dtype=np.float32)

    for j, (term, task) in enumerate(zip(idx_to_term, idx_to_task)):
        aspect = task_to_aspect(task)
        n_pos = rows[
            (rows["term"] == term)
            & (rows["aspect"] == aspect)
        ]["EntryID"].nunique()

        scores[j] = n_pos / n_train

    return scores


@torch.no_grad()
def predict(model, x, device, batch_size):
    model.eval()
    out = []

    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i:i + batch_size]).to(device).float()
        out.append(torch.sigmoid(model(xb)).cpu().numpy())

    return np.concatenate(out, axis=0).astype(np.float32)


def evaluate_cafa(pred, ids, idx_to_term, train_terms, go_obo, ia_txt):
    ia_dict = ia_parser(ia_txt)
    ontologies = {
        ns: Graph(ns, terms, ia_dict)
        for ns, terms in obo_parser(go_obo).items()
    }

    tau_arr = np.arange(0.01, 1.00, 0.01, dtype=np.float32)

    gts = build_ground_truth(ids, train_terms, ontologies)
    pred_maps = precompute_prediction_maps(ids, idx_to_term, ontologies, gts)

    return cafa_weighted_fmax_cached(
        pred,
        ontologies,
        gts,
        pred_maps,
        tau_arr,
    )


def print_result(title, df, fmax, wfmax):
    print(f"\n{title}")
    print(df.to_string(index=False) if len(df) else "No CAFA rows.")
    print("mean_Fmax :", round(float(fmax), 6))
    print("mean_WFmax:", round(float(wfmax), 6))


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--run_dir", required=True)
    p.add_argument("--train_pt", required=True)
    p.add_argument("--alphadb_valid_pt", required=True)
    p.add_argument("--experimental_valid_pt", required=True)

    p.add_argument("--train_terms_tsv", required=True)
    p.add_argument("--go_obo", required=True)
    p.add_argument("--ia_txt", required=True)

    p.add_argument("--head", choices=["ggn", "mlp"], default=None)
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--batch_size", type=int, default=65536)
    p.add_argument("--output_prefix", default="experimental_vs_alphadb_valid")

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    label_vocab_path = os.path.join(args.run_dir, "label_vocab.json")
    best_path = os.path.join(args.run_dir, "best.pt")
    metrics_path = os.path.join(args.run_dir, "metrics.json")

    with open(label_vocab_path) as f:
        vocab = json.load(f)

    idx_to_term = vocab["idx_to_term"]

    if "idx_to_task" in vocab:
        idx_to_task = vocab["idx_to_task"]
    elif "idx_to_aspect" in vocab:
        idx_to_task = [aspect_to_task(x) for x in vocab["idx_to_aspect"]]
    else:
        raise KeyError("label_vocab.json needs idx_to_task or idx_to_aspect")

    output_dim = len(idx_to_term)

    if args.head is None:
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                args.head = json.load(f).get("head", "ggn")
        else:
            args.head = "ggn"

    x_train, train_ids = load_pt(args.train_pt)
    x_alpha, alpha_ids = load_pt(args.alphadb_valid_pt)
    x_exp, exp_ids = load_pt(args.experimental_valid_pt)

    alpha_set = set(alpha_ids)
    overlap_ids = [pid for pid in exp_ids if pid in alpha_set]

    print("\nOverlap")
    print("  AlphaDB valid:", len(alpha_ids))
    print("  Experimental valid:", len(exp_ids))
    print("  overlap:", len(overlap_ids))

    if not overlap_ids:
        raise RuntimeError("zero overlap")

    x_alpha_sub = gather(overlap_ids, x_alpha, alpha_ids)
    x_exp_sub = gather(overlap_ids, x_exp, exp_ids)

    if args.standardize:
        mean = x_train.mean(axis=0, keepdims=True)
        std = np.maximum(x_train.std(axis=0, keepdims=True), 1e-6)

        x_alpha_sub = ((x_alpha_sub - mean) / std).astype(np.float32)
        x_exp_sub = ((x_exp_sub - mean) / std).astype(np.float32)

    input_dim = x_train.shape[1]

    if args.head == "mlp":
        model = MLP(input_dim, output_dim)
    else:
        model = GGNHead(input_dim, output_dim)

    model.load_state_dict(torch.load(best_path, map_location=device))
    model = model.to(device)

    print("\nModel")
    print("  run_dir:", args.run_dir)
    print("  head:", args.head)
    print("  input_dim:", input_dim)
    print("  output_dim:", output_dim)
    print("  standardize:", args.standardize)

    print("\nFeatures")
    print("  AlphaDB subset:", x_alpha_sub.shape, "mean/std:", float(x_alpha_sub.mean()), float(x_alpha_sub.std()))
    print("  Experimental :", x_exp_sub.shape, "mean/std:", float(x_exp_sub.mean()), float(x_exp_sub.std()))

    train_terms = load_train_terms(args.train_terms_tsv)

    pred_alpha = predict(model, x_alpha_sub, device, args.batch_size)
    pred_exp = predict(model, x_exp_sub, device, args.batch_size)

    freq_scores = make_frequency_baseline(
        train_terms=train_terms,
        train_ids=train_ids,
        idx_to_term=idx_to_term,
        idx_to_task=idx_to_task,
    )
    pred_freq = np.tile(freq_scores[None, :], (len(overlap_ids), 1)).astype(np.float32)

    df_alpha, f_alpha, wf_alpha = evaluate_cafa(
        pred_alpha,
        overlap_ids,
        idx_to_term,
        train_terms,
        args.go_obo,
        args.ia_txt,
    )

    df_exp, f_exp, wf_exp = evaluate_cafa(
        pred_exp,
        overlap_ids,
        idx_to_term,
        train_terms,
        args.go_obo,
        args.ia_txt,
    )

    df_freq, f_freq, wf_freq = evaluate_cafa(
        pred_freq,
        overlap_ids,
        idx_to_term,
        train_terms,
        args.go_obo,
        args.ia_txt,
    )

    print_result("AlphaDB matched valid subset", df_alpha, f_alpha, wf_alpha)
    print_result("Experimental valid subset", df_exp, f_exp, wf_exp)
    print_result("Naive frequency baseline", df_freq, f_freq, wf_freq)

    out_json = os.path.join(args.run_dir, f"{args.output_prefix}.json")
    out_csv_alpha = os.path.join(args.run_dir, f"{args.output_prefix}_alphadb_cafa.csv")
    out_csv_exp = os.path.join(args.run_dir, f"{args.output_prefix}_experimental_cafa.csv")
    out_csv_freq = os.path.join(args.run_dir, f"{args.output_prefix}_frequency_baseline_cafa.csv")

    df_alpha.to_csv(out_csv_alpha, index=False)
    df_exp.to_csv(out_csv_exp, index=False)
    df_freq.to_csv(out_csv_freq, index=False)

    np.save(
        os.path.join(args.run_dir, f"{args.output_prefix}_ids.npy"),
        np.asarray(overlap_ids, dtype=object),
        allow_pickle=True,
    )
    np.save(os.path.join(args.run_dir, f"{args.output_prefix}_alphadb_pred.npy"), pred_alpha)
    np.save(os.path.join(args.run_dir, f"{args.output_prefix}_experimental_pred.npy"), pred_exp)
    np.save(os.path.join(args.run_dir, f"{args.output_prefix}_frequency_baseline_pred.npy"), pred_freq)

    with open(out_json, "w") as f:
        json.dump(
            {
                "run_dir": args.run_dir,
                "head": args.head,
                "standardize": bool(args.standardize),
                "num_overlap": int(len(overlap_ids)),
                "alphadb": {
                    "shape": list(x_alpha_sub.shape),
                    "mean_Fmax": float(f_alpha),
                    "mean_WFmax": float(wf_alpha),
                    "per_namespace": df_alpha.to_dict("records"),
                },
                "experimental": {
                    "shape": list(x_exp_sub.shape),
                    "mean_Fmax": float(f_exp),
                    "mean_WFmax": float(wf_exp),
                    "per_namespace": df_exp.to_dict("records"),
                },
                "frequency_baseline": {
                    "shape": list(pred_freq.shape),
                    "mean_Fmax": float(f_freq),
                    "mean_WFmax": float(wf_freq),
                    "per_namespace": df_freq.to_dict("records"),
                },
            },
            f,
            indent=2,
        )

    print("\nSaved:")
    print(" ", out_json)
    print(" ", out_csv_alpha)
    print(" ", out_csv_exp)
    print(" ", out_csv_freq)


if __name__ == "__main__":
    main()