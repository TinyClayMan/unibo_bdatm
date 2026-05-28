import os
import json
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from scipy.sparse import csr_matrix


# -------------------------
# basic
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

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
# load saved split embeddings
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


# -------------------------
# callback
# -------------------------
class CafaApproxCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_valid, valid_ids, idx_to_term, train_terms, ontologies, tau_arr, batch_size):
        super().__init__()
        self.x_valid = x_valid
        self.valid_ids = valid_ids
        self.idx_to_term = idx_to_term
        self.train_terms = train_terms
        self.ontologies = ontologies
        self.tau_arr = tau_arr
        self.batch_size = batch_size

        self.gts = build_ground_truth(valid_ids, train_terms, ontologies)
        self.pred_maps = precompute_prediction_maps(valid_ids, idx_to_term, ontologies, self.gts)

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.x_valid, batch_size=self.batch_size, verbose=0)
        df_ns, mean_f, mean_wf = cafa_weighted_fmax_cached(
            pred,
            self.ontologies,
            self.gts,
            self.pred_maps,
            self.tau_arr,
        )
        print(f"\nEpoch {epoch + 1}: CAFA approx Fmax={mean_f:.5f} WFmax={mean_wf:.5f}")
        if len(df_ns):
            print(df_ns.to_string(index=False))