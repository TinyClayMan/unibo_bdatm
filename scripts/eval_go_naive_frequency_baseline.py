import os
import json
import argparse
import numpy as np
import pandas as pd

from cafa import (
    load_split_pt,
    build_ground_truth,
    precompute_prediction_maps,
    cafa_weighted_fmax_cached,
    Graph,
    obo_parser,
    ia_parser,
    set_seed,
)


def evaluate_split(pred, ids, idx_to_term, train_terms, ontologies, tau_arr, split_name):
    gts = build_ground_truth(ids, train_terms, ontologies)
    pred_maps = precompute_prediction_maps(ids, idx_to_term, ontologies, gts)

    df_ns, mean_f, mean_wf = cafa_weighted_fmax_cached(
        pred,
        ontologies,
        gts,
        pred_maps,
        tau_arr,
    )

    print(f"\n{split_name.upper()} RESULTS")
    if len(df_ns):
        print(df_ns.to_string(index=False))
    print("mean_Fmax :", round(float(mean_f), 6))
    print("mean_WFmax:", round(float(mean_wf), 6))

    return df_ns, mean_f, mean_wf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_terms_tsv", type=str, required=True)
    parser.add_argument("--go_obo", type=str, required=True)
    parser.add_argument("--ia_txt", type=str, required=True)
    parser.add_argument("--embed_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_pt", type=str, default=None)
    parser.add_argument("--valid_pt", type=str, default=None)
    parser.add_argument("--test_pt", type=str, default=None)
    parser.add_argument("--eval_splits", type=str, default="valid,test")

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_pt = args.train_pt or os.path.join(args.embed_dir, "train_embeddings.pt")
    valid_pt = args.valid_pt or os.path.join(args.embed_dir, "valid_embeddings.pt")
    test_pt = args.test_pt or os.path.join(args.embed_dir, "test_embeddings.pt")

    x_train, y_train, train_ids = load_split_pt(train_pt)
    x_valid, y_valid, valid_ids = load_split_pt(valid_pt)
    x_test, y_test, test_ids = load_split_pt(test_pt)

    print("train", x_train.shape, y_train.shape)
    print("valid", x_valid.shape, y_valid.shape)
    print("test ", x_test.shape, y_test.shape)

    num_labels = y_train.shape[1]
    if y_valid.shape[1] != num_labels or y_test.shape[1] != num_labels:
        raise ValueError("Label dimension mismatch across train/valid/test")

    label_vocab_path = os.path.join(args.embed_dir, "label_vocab.json")
    with open(label_vocab_path) as f:
        vocab = json.load(f)

    idx_to_term = vocab["idx_to_term"]
    idx_to_aspect = vocab["idx_to_aspect"]

    if len(idx_to_term) != num_labels:
        raise ValueError(
            f"label_vocab.json has {len(idx_to_term)} labels but tensors have {num_labels}"
        )

    train_terms = pd.read_csv(args.train_terms_tsv, sep="\t")
    train_terms["EntryID"] = train_terms["EntryID"].astype(str)

    ia_dict = ia_parser(args.ia_txt)
    ontologies = {
        ns: Graph(ns, terms, ia_dict)
        for ns, terms in obo_parser(args.go_obo).items()
    }

    tau_arr = np.arange(0.01, 1.00, 0.01, dtype=np.float32)

    # Naive frequency baseline:
    # one constant score per label = prevalence in training set
    label_freq = y_train.mean(axis=0).astype(np.float32)
    print("label_freq shape:", label_freq.shape)
    print("label_freq min/max/mean:", float(label_freq.min()), float(label_freq.max()), float(label_freq.mean()))

    metrics = {
        "num_labels": int(num_labels),
        "train_shape": list(x_train.shape),
        "valid_shape": list(x_valid.shape),
        "test_shape": list(x_test.shape),
        "label_frequency_stats": {
            "min": float(label_freq.min()),
            "max": float(label_freq.max()),
            "mean": float(label_freq.mean()),
        },
    }

    wanted = {s.strip() for s in args.eval_splits.split(",") if s.strip()}

    if "valid" in wanted:
        valid_pred = np.broadcast_to(label_freq, (len(valid_ids), num_labels)).copy()
        df_ns, mean_f, mean_wf = evaluate_split(
            valid_pred, valid_ids, idx_to_term, train_terms, ontologies, tau_arr, "valid"
        )
        metrics["valid"] = {
            "mean_Fmax": float(mean_f),
            "mean_WFmax": float(mean_wf),
            "per_namespace": df_ns.to_dict(orient="records"),
        }
        np.save(os.path.join(args.output_dir, "valid_pred.npy"), valid_pred)

    if "test" in wanted:
        test_pred = np.broadcast_to(label_freq, (len(test_ids), num_labels)).copy()
        df_ns, mean_f, mean_wf = evaluate_split(
            test_pred, test_ids, idx_to_term, train_terms, ontologies, tau_arr, "test"
        )
        metrics["test"] = {
            "mean_Fmax": float(mean_f),
            "mean_WFmax": float(mean_wf),
            "per_namespace": df_ns.to_dict(orient="records"),
        }
        np.save(os.path.join(args.output_dir, "test_pred.npy"), test_pred)

    np.save(os.path.join(args.output_dir, "label_frequency_prior.npy"), label_freq)
    np.save(
        os.path.join(args.output_dir, "idx_to_term.npy"),
        np.asarray(idx_to_term, dtype=object),
        allow_pickle=True,
    )
    np.save(
        os.path.join(args.output_dir, "idx_to_aspect.npy"),
        np.asarray(idx_to_aspect, dtype=object),
        allow_pickle=True,
    )

    with open(os.path.join(args.output_dir, "naive_frequency_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()