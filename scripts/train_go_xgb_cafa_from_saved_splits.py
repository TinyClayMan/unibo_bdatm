import os
import json
import argparse
import pickle
import numpy as np
import pandas as pd

from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

from cafa import (
    set_seed,
    ia_parser,
    obo_parser,
    Graph,
    build_ground_truth,
    precompute_prediction_maps,
    cafa_weighted_fmax_cached,
    load_split_pt,
)

def get_namespace_slices(idx_to_aspect):
    out = {}
    for i, ns in enumerate(idx_to_aspect):
        out.setdefault(ns, []).append(i)
    return {ns: np.asarray(v, dtype=np.int32) for ns, v in out.items()}


# -------------------------
# model helpers
# -------------------------
def build_xgb_classifier(args):
    base = XGBClassifier(
        objective="binary:logistic",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        min_child_weight=args.min_child_weight,
        tree_method=args.tree_method,
        eval_metric="logloss",
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    return MultiOutputClassifier(base, n_jobs=1)


def predict_multioutput_proba(model, x):
    probs = model.predict_proba(x)
    cols = []
    for p in probs:
        if p.ndim == 2 and p.shape[1] >= 2:
            cols.append(p[:, 1])
        else:
            cols.append(np.asarray(p).reshape(-1))
    return np.stack(cols, axis=1).astype(np.float32)


def train_one_namespace(ns, idx, x_train, y_train, x_valid, y_valid, args):
    print(f"\n=== Training namespace: {ns} ===")
    print(f"labels: {len(idx)}")

    model = build_xgb_classifier(args)
    model.fit(x_train, y_train[:, idx])

    valid_pred = predict_multioutput_proba(model, x_valid)
    train_pred = predict_multioutput_proba(model, x_train)

    return {
        "model": model,
        "train_pred": train_pred,
        "valid_pred": valid_pred,
        "idx": idx,
    }


def merge_namespace_predictions(models_by_ns, x_shape0, num_labels, split="valid"):
    y_score = np.zeros((x_shape0, num_labels), dtype=np.float32)
    for ns, obj in models_by_ns.items():
        idx = obj["idx"]
        pred = obj[f"{split}_pred"]
        y_score[:, idx] = pred
    return y_score


# -------------------------
# main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_terms_tsv", type=str, required=True)
    parser.add_argument("--go_obo", type=str, required=True)
    parser.add_argument("--ia_txt", type=str, required=True)
    parser.add_argument("--embed_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--train_pt", type=str, default=None)
    parser.add_argument("--valid_pt", type=str, default=None)
    parser.add_argument("--test_pt", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)

    # XGBoost params
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--reg_lambda", type=float, default=2.0)
    parser.add_argument("--min_child_weight", type=float, default=1.0)
    parser.add_argument("--tree_method", type=str, default="hist")
    parser.add_argument("--n_jobs", type=int, default=-1)

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

    if y_train.shape[1] != y_valid.shape[1] or y_train.shape[1] != y_test.shape[1]:
        raise ValueError("Label dimension mismatch across train/valid/test")

    num_labels = y_train.shape[1]

    label_vocab_path = os.path.join(args.embed_dir, "label_vocab.json")
    with open(label_vocab_path) as f:
        vocab = json.load(f)

    idx_to_term = vocab["idx_to_term"]
    idx_to_aspect = vocab["idx_to_aspect"]

    if len(idx_to_term) != num_labels:
        raise ValueError(f"label_vocab.json has {len(idx_to_term)} labels but tensors have {num_labels}")

    ns_to_idx = get_namespace_slices(idx_to_aspect)
    print("aspect counts:")
    print(pd.Series(idx_to_aspect).value_counts().to_string())

    train_terms = pd.read_csv(args.train_terms_tsv, sep="\t")
    train_terms["EntryID"] = train_terms["EntryID"].astype(str)

    ia_dict = ia_parser(args.ia_txt)
    ontologies = {
        ns: Graph(ns, terms, ia_dict)
        for ns, terms in obo_parser(args.go_obo).items()
    }

    tau_arr = np.arange(0.01, 1.00, 0.01, dtype=np.float32)

    valid_gts = build_ground_truth(valid_ids, train_terms, ontologies)
    valid_pred_maps = precompute_prediction_maps(valid_ids, idx_to_term, ontologies, valid_gts)

    test_gts = build_ground_truth(test_ids, train_terms, ontologies)
    test_pred_maps = precompute_prediction_maps(test_ids, idx_to_term, ontologies, test_gts)

    models_by_ns = {}
    for ns in ["biological_process", "molecular_function", "cellular_component"]:
        if ns not in ns_to_idx:
            continue
        idx = ns_to_idx[ns]
        models_by_ns[ns] = train_one_namespace(
            ns, idx, x_train, y_train, x_valid, y_valid, args
        )

    valid_pred = merge_namespace_predictions(models_by_ns, len(x_valid), num_labels, split="valid")
    df_valid, mean_f_valid, mean_wf_valid = cafa_weighted_fmax_cached(
        valid_pred, ontologies, valid_gts, valid_pred_maps, tau_arr
    )

    print("\nVALID")
    if len(df_valid):
        print(df_valid.to_string(index=False))
    print("mean_Fmax :", round(float(mean_f_valid), 6))
    print("mean_WFmax:", round(float(mean_wf_valid), 6))

    # Refit each namespace model on train+valid for final test
    x_trainvalid = np.concatenate([x_train, x_valid], axis=0)
    y_trainvalid = np.concatenate([y_train, y_valid], axis=0)

    final_models = {}
    for ns in ["biological_process", "molecular_function", "cellular_component"]:
        if ns not in ns_to_idx:
            continue
        idx = ns_to_idx[ns]
        print(f"\n=== Refit namespace on train+valid: {ns} ===")
        model = build_xgb_classifier(args)
        model.fit(x_trainvalid, y_trainvalid[:, idx])
        final_models[ns] = {
            "model": model,
            "idx": idx,
        }

        with open(os.path.join(args.output_dir, f"xgb_{ns}.pkl"), "wb") as f:
            pickle.dump(model, f)

    test_pred = np.zeros((len(x_test), num_labels), dtype=np.float32)
    for ns, obj in final_models.items():
        idx = obj["idx"]
        pred = predict_multioutput_proba(obj["model"], x_test)
        test_pred[:, idx] = pred

    df_ns, mean_f, mean_wf = cafa_weighted_fmax_cached(
        test_pred, ontologies, test_gts, test_pred_maps, tau_arr
    )

    print("\nFINAL TEST")
    if len(df_ns):
        print(df_ns.to_string(index=False))
    print("mean_Fmax :", round(float(mean_f), 6))
    print("mean_WFmax:", round(float(mean_wf), 6))

    with open(os.path.join(args.output_dir, "final_test_metrics.json"), "w") as f:
        json.dump(
            {
                "mean_Fmax": float(mean_f),
                "mean_WFmax": float(mean_wf),
                "valid_mean_Fmax": float(mean_f_valid),
                "valid_mean_WFmax": float(mean_wf_valid),
                "per_namespace": df_ns.to_dict(orient="records"),
                "valid_per_namespace": df_valid.to_dict(orient="records"),
                "num_labels": int(num_labels),
                "train_shape": list(x_train.shape),
                "valid_shape": list(x_valid.shape),
                "test_shape": list(x_test.shape),
                "xgb_params": {
                    "n_estimators": args.n_estimators,
                    "max_depth": args.max_depth,
                    "learning_rate": args.learning_rate,
                    "subsample": args.subsample,
                    "colsample_bytree": args.colsample_bytree,
                    "reg_lambda": args.reg_lambda,
                    "min_child_weight": args.min_child_weight,
                    "tree_method": args.tree_method,
                },
            },
            f,
            indent=2,
        )

    np.save(os.path.join(args.output_dir, "test_pred.npy"), test_pred)
    np.save(
        os.path.join(args.output_dir, "test_ids.npy"),
        np.asarray(test_ids, dtype=object),
        allow_pickle=True,
    )

    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()