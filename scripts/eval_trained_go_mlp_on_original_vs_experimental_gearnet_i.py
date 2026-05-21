import os
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

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


def normalize_id(x):
    return str(x).strip()


def load_split_dict(pt_path):
    x, y, ids = load_split_pt(pt_path)
    ids = [normalize_id(z) for z in ids]
    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "ids": ids,
        "id_to_row": {acc: i for i, acc in enumerate(ids)},
    }


def intersect_two_splits(ref_split, new_split, split_name):
    common_ids = sorted(set(ref_split["ids"]).intersection(new_split["ids"]))
    if not common_ids:
        raise RuntimeError(f"[{split_name}] no overlapping accessions between original and experimental embeddings")

    ref_idx = np.asarray([ref_split["id_to_row"][acc] for acc in common_ids], dtype=np.int64)
    new_idx = np.asarray([new_split["id_to_row"][acc] for acc in common_ids], dtype=np.int64)

    x_ref = ref_split["x"][ref_idx]
    y_ref = ref_split["y"][ref_idx]
    x_new = new_split["x"][new_idx]
    y_new = new_split["y"][new_idx]

    if y_ref.shape != y_new.shape:
        raise ValueError(f"[{split_name}] label shape mismatch after intersection")
    if not np.array_equal(y_ref, y_new):
        # should match if both were built from same reference labels
        print(f"[{split_name}] warning: y_ref and y_new differ on some rows; using reference labels")

    return common_ids, x_ref, x_new, y_ref


def evaluate_split(
    model,
    split_name,
    ids,
    x_eval,
    train_terms,
    idx_to_term,
    ontologies,
    tau_arr,
    batch_size,
):
    gts = build_ground_truth(ids, train_terms, ontologies)
    pred_maps = precompute_prediction_maps(ids, idx_to_term, ontologies, gts)

    pred = model.predict(x_eval, batch_size=batch_size, verbose=1)
    df_ns, mean_f, mean_wf = cafa_weighted_fmax_cached(
        pred,
        ontologies,
        gts,
        pred_maps,
        tau_arr,
    )

    return {
        "split": split_name,
        "n": int(len(ids)),
        "df_ns": df_ns,
        "mean_fmax": float(mean_f),
        "mean_wfmax": float(mean_wf),
        "pred": pred,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_terms_tsv", type=str, required=True)
    parser.add_argument("--go_obo", type=str, required=True)
    parser.add_argument("--ia_txt", type=str, required=True)

    parser.add_argument("--original_embed_dir", type=str, required=True,
                        help="Original AlphaDB/GearNet embeddings dir, e.g. /content/gearnet_embeds")
    parser.add_argument("--experimental_embed_dir", type=str, required=True,
                        help="Experimental PDB GearNet embeddings dir from step 1")
    parser.add_argument("--trained_model_path", type=str, required=True,
                        help="Path to trained keras classifier, e.g. best_model.keras or last_model.keras")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--valid_original_pt", type=str, default=None)
    parser.add_argument("--test_original_pt", type=str, default=None)
    parser.add_argument("--valid_experimental_pt", type=str, default=None)
    parser.add_argument("--test_experimental_pt", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    valid_original_pt = args.valid_original_pt or os.path.join(args.original_embed_dir, "valid_embeddings.pt")
    test_original_pt = args.test_original_pt or os.path.join(args.original_embed_dir, "test_embeddings.pt")
    valid_experimental_pt = args.valid_experimental_pt or os.path.join(args.experimental_embed_dir, "valid_embeddings.pt")
    test_experimental_pt = args.test_experimental_pt or os.path.join(args.experimental_embed_dir, "test_embeddings.pt")

    label_vocab_path = os.path.join(args.original_embed_dir, "label_vocab.json")
    with open(label_vocab_path) as f:
        vocab = json.load(f)

    idx_to_term = vocab["idx_to_term"]
    idx_to_aspect = vocab["idx_to_aspect"]

    train_terms = pd.read_csv(args.train_terms_tsv, sep="\t")
    train_terms["EntryID"] = train_terms["EntryID"].astype(str)

    ia_dict = ia_parser(args.ia_txt)
    ontologies = {
        ns: Graph(ns, terms, ia_dict)
        for ns, terms in obo_parser(args.go_obo).items()
    }
    tau_arr = np.arange(0.01, 1.00, 0.01, dtype=np.float32)

    model = tf.keras.models.load_model(args.trained_model_path)

    results_summary = {}

    for split_name, orig_pt, exp_pt in [
        ("valid", valid_original_pt, valid_experimental_pt),
        ("test", test_original_pt, test_experimental_pt),
    ]:
        print(f"\n===== {split_name.upper()} =====")

        orig = load_split_dict(orig_pt)
        exp = load_split_dict(exp_pt)

        common_ids, x_orig, x_exp, y_common = intersect_two_splits(orig, exp, split_name)

        print(f"[{split_name}] common accessions: {len(common_ids)}")
        print(f"[{split_name}] original x shape: {x_orig.shape}")
        print(f"[{split_name}] experimental x shape: {x_exp.shape}")

        if x_orig.shape[1] != x_exp.shape[1]:
            raise ValueError(f"[{split_name}] embedding dim mismatch: {x_orig.shape[1]} vs {x_exp.shape[1]}")

        res_orig = evaluate_split(
            model=model,
            split_name=split_name,
            ids=common_ids,
            x_eval=x_orig,
            train_terms=train_terms,
            idx_to_term=idx_to_term,
            ontologies=ontologies,
            tau_arr=tau_arr,
            batch_size=args.batch_size,
        )

        res_exp = evaluate_split(
            model=model,
            split_name=split_name,
            ids=common_ids,
            x_eval=x_exp,
            train_terms=train_terms,
            idx_to_term=idx_to_term,
            ontologies=ontologies,
            tau_arr=tau_arr,
            batch_size=args.batch_size,
        )

        print(f"\n[{split_name}] ORIGINAL AlphaDB GearNet")
        if len(res_orig["df_ns"]):
            print(res_orig["df_ns"].to_string(index=False))
        print("mean_Fmax :", round(res_orig["mean_fmax"], 6))
        print("mean_WFmax:", round(res_orig["mean_wfmax"], 6))

        print(f"\n[{split_name}] EXPERIMENTAL PDB GearNet")
        if len(res_exp["df_ns"]):
            print(res_exp["df_ns"].to_string(index=False))
        print("mean_Fmax :", round(res_exp["mean_fmax"], 6))
        print("mean_WFmax:", round(res_exp["mean_wfmax"], 6))

        delta_f = res_exp["mean_fmax"] - res_orig["mean_fmax"]
        delta_wf = res_exp["mean_wfmax"] - res_orig["mean_wfmax"]

        print(f"\n[{split_name}] DELTA experimental - original")
        print("delta_Fmax :", round(float(delta_f), 6))
        print("delta_WFmax:", round(float(delta_wf), 6))

        res_orig["df_ns"].to_csv(os.path.join(args.output_dir, f"{split_name}_original_metrics.csv"), index=False)
        res_exp["df_ns"].to_csv(os.path.join(args.output_dir, f"{split_name}_experimental_metrics.csv"), index=False)

        np.save(os.path.join(args.output_dir, f"{split_name}_common_ids.npy"),
                np.asarray(common_ids, dtype=object), allow_pickle=True)
        np.save(os.path.join(args.output_dir, f"{split_name}_original_pred.npy"), res_orig["pred"])
        np.save(os.path.join(args.output_dir, f"{split_name}_experimental_pred.npy"), res_exp["pred"])

        pd.DataFrame({
            "accession": common_ids,
        }).to_csv(os.path.join(args.output_dir, f"{split_name}_common_ids.csv"), index=False)

        results_summary[split_name] = {
            "n_common": int(len(common_ids)),
            "original": {
                "mean_Fmax": float(res_orig["mean_fmax"]),
                "mean_WFmax": float(res_orig["mean_wfmax"]),
                "per_namespace": res_orig["df_ns"].to_dict(orient="records"),
            },
            "experimental": {
                "mean_Fmax": float(res_exp["mean_fmax"]),
                "mean_WFmax": float(res_exp["mean_wfmax"]),
                "per_namespace": res_exp["df_ns"].to_dict(orient="records"),
            },
            "delta_experimental_minus_original": {
                "mean_Fmax": float(delta_f),
                "mean_WFmax": float(delta_wf),
            },
        }

    with open(os.path.join(args.output_dir, "comparison_summary.json"), "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nSaved comparison outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()