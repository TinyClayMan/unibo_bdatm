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
    CafaApproxCallback,
    Graph,
    obo_parser,
    ia_parser,
    set_seed,
    normalize_id,
    build_x_for_ids,
    align_rows_to_found_ids,
)

# -------------------------------------------------------
# model
# -------------------------------------------------------
class ModalityDropout(tf.keras.layers.Layer):
    def __init__(self, drop_prob=0.10, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = float(drop_prob)

    def call(self, inputs, training=False):
        xa, xb = inputs
        if (not training) or self.drop_prob <= 0.0:
            return xa, xb

        bsz = tf.shape(xa)[0]
        r = tf.random.uniform([bsz, 1], 0.0, 1.0, dtype=xa.dtype)

        # keep both / drop a / drop b
        keep_both = tf.cast(r >= 2.0 * self.drop_prob, xa.dtype)
        drop_a = tf.cast((r >= self.drop_prob) & (r < 2.0 * self.drop_prob), xa.dtype)
        drop_b = tf.cast(r < self.drop_prob, xa.dtype)

        xa_out = xa * (keep_both + drop_b)
        xb_out = xb * (keep_both + drop_a)
        return xa_out, xb_out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"drop_prob": self.drop_prob})
        return cfg


class GatedLowRankBilinearFusion(tf.keras.Model):
    def __init__(
        self,
        dim_a,
        dim_b,
        num_labels,
        hidden_dim=512,
        bilinear_rank=128,
        dropout=0.20,
        modality_drop_prob=0.10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim_a = int(dim_a)
        self.dim_b = int(dim_b)
        self.num_labels = int(num_labels)
        self.hidden_dim = int(hidden_dim)
        self.bilinear_rank = int(bilinear_rank)
        self.dropout_rate = float(dropout)
        self.modality_drop_prob = float(modality_drop_prob)

        self.norm_a = tf.keras.layers.LayerNormalization()
        self.norm_b = tf.keras.layers.LayerNormalization()

        self.proj_a = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, use_bias=False),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(dropout),
        ])

        self.proj_b = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, use_bias=False),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(dropout),
        ])

        self.mod_drop = ModalityDropout(drop_prob=modality_drop_prob)

        # low-rank bilinear interaction
        self.bilin_a = tf.keras.layers.Dense(bilinear_rank, use_bias=False)
        self.bilin_b = tf.keras.layers.Dense(bilinear_rank, use_bias=False)
        self.bilin_out = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, use_bias=False),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(dropout),
        ])

        # gate chooses how much to trust each source / interaction
        self.gate = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="gelu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(4, activation="softmax"),
        ])

        self.head = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(1024, activation="gelu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(512, activation="gelu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(num_labels, activation="sigmoid"),
        ])

    def call(self, inputs, training=False):
        xa, xb = inputs

        xa = self.norm_a(xa)
        xb = self.norm_b(xb)

        sa = self.proj_a(xa, training=training)   # [B, H]
        sb = self.proj_b(xb, training=training)   # [B, H]

        sa, sb = self.mod_drop((sa, sb), training=training)

        prod = sa * sb
        diff = tf.abs(sa - sb)

        ba = self.bilin_a(sa)                     # [B, R]
        bb = self.bilin_b(sb)                     # [B, R]
        bilinear = ba * bb                        # low-rank bilinear interaction
        bilinear = self.bilin_out(bilinear, training=training)  # [B, H]

        gates = self.gate(tf.concat([sa, sb, prod, diff, bilinear], axis=-1), training=training)
        fused = (
            gates[:, 0:1] * sa +
            gates[:, 1:2] * sb +
            gates[:, 2:3] * prod +
            gates[:, 3:4] * bilinear
        )

        final_rep = tf.concat([sa, sb, prod, diff, bilinear, fused], axis=-1)
        return self.head(final_rep, training=training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "dim_a": self.dim_a,
            "dim_b": self.dim_b,
            "num_labels": self.num_labels,
            "hidden_dim": self.hidden_dim,
            "bilinear_rank": self.bilinear_rank,
            "dropout": self.dropout_rate,
            "modality_drop_prob": self.modality_drop_prob,
        })
        return cfg


# -------------------------------------------------------
# main
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_terms_tsv", type=str, required=True)
    parser.add_argument("--go_obo", type=str, required=True)
    parser.add_argument("--ia_txt", type=str, required=True)

    # modality A: saved split embeddings (e.g. GearNet)
    parser.add_argument("--split_embed_dir", type=str, required=True)
    parser.add_argument("--train_pt", type=str, default=None)
    parser.add_argument("--valid_pt", type=str, default=None)
    parser.add_argument("--test_pt", type=str, default=None)

    # modality B: aligned npy embeddings (e.g. T5)
    parser.add_argument("--other_ids_npy", type=str, required=True)
    parser.add_argument("--other_embeds_npy", type=str, required=True)

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--strict_other_ids", action="store_true")
    parser.add_argument("--use_cafa_callback", action="store_true")

    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--bilinear_rank", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--modality_drop_prob", type=float, default=0.10)

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_pt = args.train_pt or os.path.join(args.split_embed_dir, "train_embeddings.pt")
    valid_pt = args.valid_pt or os.path.join(args.split_embed_dir, "valid_embeddings.pt")
    test_pt  = args.test_pt  or os.path.join(args.split_embed_dir, "test_embeddings.pt")
    label_vocab_path = os.path.join(args.split_embed_dir, "label_vocab.json")

    x_train_a, y_train_ref, train_ids_all = load_split_pt(train_pt)
    x_valid_a, y_valid_ref, valid_ids_all = load_split_pt(valid_pt)
    x_test_a,  y_test_ref,  test_ids_all  = load_split_pt(test_pt)

    other_ids = np.load(args.other_ids_npy, allow_pickle=True).astype(str)
    other_embeds = np.load(args.other_embeds_npy).astype(np.float32)

    x_train_b, train_ids, missing_train = build_x_for_ids(
        train_ids_all, other_ids, other_embeds, "train", strict=args.strict_other_ids
    )
    x_valid_b, valid_ids, missing_valid = build_x_for_ids(
        valid_ids_all, other_ids, other_embeds, "valid", strict=args.strict_other_ids
    )
    x_test_b, test_ids, missing_test = build_x_for_ids(
        test_ids_all, other_ids, other_embeds, "test", strict=args.strict_other_ids
    )

    x_train_a = align_rows_to_found_ids(train_ids_all, x_train_a, train_ids).astype(np.float32)
    x_valid_a = align_rows_to_found_ids(valid_ids_all, x_valid_a, valid_ids).astype(np.float32)
    x_test_a  = align_rows_to_found_ids(test_ids_all,  x_test_a,  test_ids).astype(np.float32)

    y_train = align_rows_to_found_ids(train_ids_all, y_train_ref, train_ids).astype(np.float32)
    y_valid = align_rows_to_found_ids(valid_ids_all, y_valid_ref, valid_ids).astype(np.float32)
    y_test  = align_rows_to_found_ids(test_ids_all,  y_test_ref,  test_ids).astype(np.float32)

    with open(label_vocab_path) as f:
        vocab = json.load(f)

    idx_to_term = vocab["idx_to_term"]
    idx_to_aspect = vocab["idx_to_aspect"]
    num_labels = len(idx_to_term)

    if num_labels != y_train.shape[1]:
        raise ValueError(
            f"label_vocab.json has {num_labels} labels but tensors have {y_train.shape[1]}"
        )

    print("A train", x_train_a.shape, "B train", x_train_b.shape, "y", y_train.shape)
    print("A valid", x_valid_a.shape, "B valid", x_valid_b.shape, "y", y_valid.shape)
    print("A test ", x_test_a.shape,  "B test ", x_test_b.shape,  "y", y_test.shape)

    train_terms = pd.read_csv(args.train_terms_tsv, sep="\t")
    train_terms["EntryID"] = train_terms["EntryID"].astype(str)

    ia_dict = ia_parser(args.ia_txt)
    ontologies = {
        ns: Graph(ns, terms, ia_dict)
        for ns, terms in obo_parser(args.go_obo).items()
    }
    tau_arr = np.arange(0.01, 1.00, 0.01, dtype=np.float32)

    model = GatedLowRankBilinearFusion(
        dim_a=x_train_a.shape[1],
        dim_b=x_train_b.shape[1],
        num_labels=num_labels,
        hidden_dim=args.hidden_dim,
        bilinear_rank=args.bilinear_rank,
        dropout=args.dropout,
        modality_drop_prob=args.modality_drop_prob,
    )

    # build model once so weight checkpointing works cleanly
    _ = model([x_train_a[:2], x_train_b[:2]], training=False)

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "binary_accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )

    best_weights_path = os.path.join(args.output_dir, "best.weights.h5")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_weights_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(args.output_dir, "history.csv")),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=False,
            verbose=1,
        ),
    ]

    if args.use_cafa_callback:
        callbacks.append(
            CafaApproxCallback(
                x_valid=[x_valid_a, x_valid_b],
                valid_ids=valid_ids,
                idx_to_term=idx_to_term,
                train_terms=train_terms,
                ontologies=ontologies,
                tau_arr=tau_arr,
                batch_size=args.batch_size,
            )
        )

    history = model.fit(
        [x_train_a, x_train_b],
        y_train,
        validation_data=([x_valid_a, x_valid_b], y_valid),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    if os.path.exists(best_weights_path):
        model.load_weights(best_weights_path)

    # optional full export for later reloading
    model.save_weights(os.path.join(args.output_dir, "last.weights.h5"))

    test_gts = build_ground_truth(test_ids, train_terms, ontologies)
    test_pred_maps = precompute_prediction_maps(test_ids, idx_to_term, ontologies, test_gts)

    test_pred = model.predict([x_test_a, x_test_b], batch_size=args.batch_size, verbose=1)
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
                "train_shape_a": list(x_train_a.shape),
                "valid_shape_a": list(x_valid_a.shape),
                "test_shape_a": list(x_test_a.shape),
                "train_shape_b": list(x_train_b.shape),
                "valid_shape_b": list(x_valid_b.shape),
                "test_shape_b": list(x_test_b.shape),
                "missing_other_ids": {
                    "train": len(missing_train),
                    "valid": len(missing_valid),
                    "test": len(missing_test),
                },
                "config": vars(args),
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