import os
import json
import random
import argparse
import pickle
import numpy as np
import pandas as pd
import torch

from scipy.sparse import csr_matrix
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier


# -------------------------
# basic
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# -------------------------
# ontology
# -------------------------
def ia_parser(path):
    d = {}
    with open(path) as f:
        for line in f:
            term, ia = line.strip().split()
            d[term] = float(ia)
    return d


def obo_parser(path):
    term_dict = {}
    term_id = namespace = name = term_def = None
    alt_id, rel = [], []
    obsolete = True

    with open(path) as f:
        for raw in f:
            line = raw.strip().split(": ")
            if not line or len(line) <= 1:
                continue
            k, v = line[0], ": ".join(line[1:])

            if k == "id":
                if term_id is not None and (not obsolete) and namespace is not None:
                    term_dict.setdefault(namespace, {})[term_id] = {
                        "name": name,
                        "namespace": namespace,
                        "def": term_def,
                        "alt_id": alt_id,
                        "rel": rel,
                    }
                term_id = v
                namespace = None
                name = None
                term_def = None
                alt_id, rel = [], []
                obsolete = False
            elif k == "alt_id":
                alt_id.append(v)
            elif k == "name":
                name = v
            elif k == "namespace" and v != "external":
                namespace = v
            elif k == "def":
                term_def = v
            elif k == "is_obsolete":
                obsolete = True
            elif k == "is_a":
                rel.append(v.split("!")[0].strip())
            elif k == "relationship" and v.startswith("part_of"):
                rel.append(v.split()[1].strip())

    if term_id is not None and (not obsolete) and namespace is not None:
        term_dict.setdefault(namespace, {})[term_id] = {
            "name": name,
            "namespace": namespace,
            "def": term_def,
            "alt_id": alt_id,
            "rel": rel,
        }

    return term_dict


class Graph:
    def __init__(self, namespace, terms_dict, ia_dict):
        self.namespace = namespace
        self.terms_dict = {}
        self.primary_ids = []
        rels = []

        for idx, (term_id, term) in enumerate(terms_dict.items()):
            self.primary_ids.append(term_id)
            self.terms_dict[term_id] = {"index": idx}
            for a in term["alt_id"]:
                self.terms_dict[a] = {"index": idx}
            for r in term["rel"]:
                rels.append((term_id, r))

        self.n = len(self.primary_ids)

        child_idx = []
        parent_idx = []

        for c, p in rels:
            if c in self.terms_dict and p in self.terms_dict:
                i = self.terms_dict[c]["index"]
                j = self.terms_dict[p]["index"]
                child_idx.append(i)
                parent_idx.append(j)

        if len(child_idx):
            data = np.ones(len(child_idx), dtype=np.uint8)
            self.child_parent_csr = csr_matrix((data, (child_idx, parent_idx)), shape=(self.n, self.n))
        else:
            self.child_parent_csr = csr_matrix((self.n, self.n), dtype=np.uint8)

        self.children_of = [None] * self.n
        self.parents_of = [None] * self.n

        c2p = self.child_parent_csr
        p2c = c2p.transpose().tocsr()

        for i in range(self.n):
            s, e = c2p.indptr[i], c2p.indptr[i + 1]
            self.parents_of[i] = c2p.indices[s:e].astype(np.int32, copy=False)
            s, e = p2c.indptr[i], p2c.indptr[i + 1]
            self.children_of[i] = p2c.indices[s:e].astype(np.int32, copy=False)

        self.order = self.top_sort()

        self.ia = np.zeros(self.n, dtype=np.float32)
        for term_id in self.primary_ids:
            if term_id in ia_dict:
                self.ia[self.terms_dict[term_id]["index"]] = ia_dict[term_id]
        np.nan_to_num(self.ia, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        self.toi = np.where(self.ia > 0)[0].astype(np.int32)

    def top_sort(self):
        indeg = np.diff(self.child_parent_csr.transpose().tocsr().indptr).astype(np.int32)
        q = list(np.where(indeg == 0)[0])
        out = []
        while q:
            x = q.pop(0)
            out.append(x)
            indeg[x] -= 1
            for parent in self.parents_of[x]:
                indeg[parent] -= 1
                if indeg[parent] == 0:
                    q.append(parent)
        return np.asarray(out, dtype=np.int32)


def propagate_scores(mat, ont, mode="fill"):
    if mat.shape[0] == 0:
        return mat
    order = ont.order
    if order.size == 0:
        return mat
    nonzero_cols = np.where(np.sum(mat[:, order], axis=0) > 0)[0]
    if len(nonzero_cols) == 0:
        return mat
    order_ = order[nonzero_cols[0]:]
    for parent in order_:
        children = ont.children_of[parent]
        if children.size == 0:
            continue
        child_max = mat[:, children].max(axis=1)
        if mode == "max":
            mat[:, parent] = np.maximum(mat[:, parent], child_max)
        else:
            rows = np.where(mat[:, parent] == 0)[0]
            if rows.size:
                mat[rows, parent] = child_max[rows]
    return mat


# -------------------------
# CAFA approx evaluator
# -------------------------
def build_ground_truth(valid_ids, train_terms, ontologies):
    gts = {}
    use = train_terms[train_terms["EntryID"].isin(valid_ids)][["EntryID", "term"]].copy()

    for ns, ont in ontologies.items():
        rows = use[use["term"].isin(ont.terms_dict.keys())]
        if rows.empty:
            continue
        pids = rows["EntryID"].drop_duplicates().tolist()
        pid_to_i = {p: i for i, p in enumerate(pids)}
        mat = np.zeros((len(pids), ont.n), dtype=bool)
        for pid, term in rows.itertuples(index=False):
            mat[pid_to_i[pid], ont.terms_dict[term]["index"]] = 1
        mat = propagate_scores(mat, ont, mode="max")
        gts[ns] = (pid_to_i, mat)
    return gts


def precompute_prediction_maps(protein_ids, label_terms, ontologies, gts):
    term_to_col = {t: i for i, t in enumerate(label_terms)}
    maps = {}
    for ns, ont in ontologies.items():
        if ns not in gts:
            continue
        pid_to_i, gt = gts[ns]
        pred_rows = []
        ont_rows = []
        for r, pid in enumerate(protein_ids):
            i = pid_to_i.get(pid)
            if i is not None:
                pred_rows.append(r)
                ont_rows.append(i)
        if not pred_rows:
            continue
        keep_terms = [t for t in label_terms if t in ont.terms_dict]
        if not keep_terms:
            continue
        model_cols = np.array([term_to_col[t] for t in keep_terms], dtype=np.int32)
        go_cols = np.array([ont.terms_dict[t]["index"] for t in keep_terms], dtype=np.int32)
        maps[ns] = {
            "pred_rows": np.array(pred_rows, dtype=np.int32),
            "ont_rows": np.array(ont_rows, dtype=np.int32),
            "model_cols": model_cols,
            "go_cols": go_cols,
            "gt_shape": gt.shape,
        }
    return maps


def build_prediction_mats(y_score, ontologies, gts, pred_maps):
    out = {}
    for ns, m in pred_maps.items():
        pred = np.zeros(m["gt_shape"], dtype=np.float32)
        block = y_score[np.ix_(m["pred_rows"], m["model_cols"])]
        pred[np.ix_(m["ont_rows"], m["go_cols"])] = block
        pred = propagate_scores(pred, ontologies[ns], mode="fill")
        out[ns] = pred
    return out


def _f(pr, rc):
    d = pr + rc
    return np.divide(2 * pr * rc, d, out=np.zeros_like(pr, dtype=float), where=d != 0)


def cafa_weighted_fmax_cached(y_score, ontologies, gts, pred_maps, tau_arr):
    preds = build_prediction_mats(y_score, ontologies, gts, pred_maps)
    rows = []
    for ns, ont in ontologies.items():
        if ns not in gts or ns not in preds:
            continue
        _, g = gts[ns]
        p_score = preds[ns]
        toi = ont.toi
        if len(toi) == 0:
            continue
        g = g[:, toi].astype(bool)
        p_score = p_score[:, toi]
        ia = ont.ia[toi]

        n_gt = g.sum(axis=1)
        wn_gt = (g * ia).sum(axis=1)

        best_wf, best_tau = 0.0, 0.5
        best_f = 0.0

        for tau in tau_arr:
            p = p_score > tau
            cov = (p.sum(axis=1) > 0)
            inter = p & g
            n_pred = p.sum(axis=1)
            n_inter = inter.sum(axis=1)

            pr = np.divide(n_inter, n_pred, out=np.zeros_like(n_inter, dtype=float), where=n_pred > 0)
            rc = np.divide(n_inter, n_gt, out=np.zeros_like(n_inter, dtype=float), where=n_gt > 0)
            pr = pr[cov].mean() if cov.sum() > 0 else 0.0
            rc = rc.mean()
            f = _f(np.array([pr]), np.array([rc]))[0]

            wn_pred = (p * ia).sum(axis=1)
            wn_inter = (inter * ia).sum(axis=1)
            wpr = np.divide(wn_inter, wn_pred, out=np.zeros_like(wn_inter, dtype=float), where=wn_pred > 0)
            wrc = np.divide(wn_inter, wn_gt, out=np.zeros_like(wn_gt, dtype=float), where=wn_gt > 0)
            wpr = wpr[cov].mean() if cov.sum() > 0 else 0.0
            wrc = wrc.mean()
            wf = _f(np.array([wpr]), np.array([wrc]))[0]

            if wf > best_wf:
                best_wf, best_tau = float(wf), float(tau)
            if f > best_f:
                best_f = float(f)

        rows.append((ns, best_f, best_wf, best_tau))

    df = pd.DataFrame(rows, columns=["ns", "fmax", "wfmax", "best_tau"])
    mean_f = df["fmax"].mean() if len(df) else 0.0
    mean_wf = df["wfmax"].mean() if len(df) else 0.0
    return df, mean_f, mean_wf


# -------------------------
# saved embeddings
# -------------------------
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