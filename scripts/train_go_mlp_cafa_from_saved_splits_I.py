import os
import json
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from scipy.sparse import csr_matrix
from cafa import load_split_pt, build_ground_truth, precompute_prediction_maps, cafa_weighted_fmax_cached, CafaApproxCallback, Graph, obo_parser, ia_parser, set_seed

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

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=5120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_cafa_callback", action="store_true")

    parser.add_argument("--train_pt", type=str, default=None)
    parser.add_argument("--valid_pt", type=str, default=None)
    parser.add_argument("--test_pt", type=str, default=None)

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

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.LayerNormalization(),

        tf.keras.layers.Dense(1024, use_bias=False),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Activation("gelu"),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Dense(1024, use_bias=False),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Activation("gelu"),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Dense(512, use_bias=False),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Activation("gelu"),
        tf.keras.layers.Dropout(0.20),

        tf.keras.layers.Dense(num_labels, activation="sigmoid"),
    ])


    model.compile(
        optimizer=tf.keras.optimizers.AdamW(args.lr),
        loss="binary_crossentropy",
        metrics=["binary_accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.output_dir, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(args.output_dir, "history.csv")),
    ]

    if args.use_cafa_callback:
        callbacks.append(
            CafaApproxCallback(
                x_valid=x_valid,
                valid_ids=valid_ids,
                idx_to_term=idx_to_term,
                train_terms=train_terms,
                ontologies=ontologies,
                tau_arr=tau_arr,
                batch_size=args.batch_size,
            )
        )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(os.path.join(args.output_dir, "last_model.keras"))

    test_gts = build_ground_truth(test_ids, train_terms, ontologies)
    test_pred_maps = precompute_prediction_maps(test_ids, idx_to_term, ontologies, test_gts)

    test_pred = model.predict(x_test, batch_size=args.batch_size, verbose=1)
    df_ns, mean_f, mean_wf = cafa_weighted_fmax_cached(
        test_pred,
        ontologies,
        test_gts,
        test_pred_maps,
        tau_arr,
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
                "per_namespace": df_ns.to_dict(orient="records"),
                "num_labels": int(num_labels),
                "train_shape": list(x_train.shape),
                "valid_shape": list(x_valid.shape),
                "test_shape": list(x_test.shape),
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