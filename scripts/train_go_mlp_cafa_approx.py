import os
import json
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

# -------------------------
# config
# -------------------------
TRAIN_TERMS_TSV = "train_terms.tsv"
GO_OBO = "go-basic.obo"
IA_TXT = "IA.txt"

# T5 / Prot embedding source
TRAIN_IDS_NPY = "train_ids.npy"
TRAIN_EMBEDS_NPY = "train_embeds.npy"

# GearNet saved splits: source of accession IDs + labels + split membership
GEARNET_SPLIT_DIR = "/content/gearnet_embeds"

TRAIN_PT = os.path.join(GEARNET_SPLIT_DIR, "train_embeddings.pt")
VALID_PT = os.path.join(GEARNET_SPLIT_DIR, "valid_embeddings.pt")
TEST_PT  = os.path.join(GEARNET_SPLIT_DIR, "test_embeddings.pt")
LABEL_VOCAB_JSON = os.path.join(GEARNET_SPLIT_DIR, "label_vocab.json")

BATCH_SIZE = 5120
EPOCHS = 60
SEED = 42
TAU_ARR = np.arange(0.01, 1.00, 0.01, dtype=np.float32)


# -------------------------
# build dataset from GearNet splits + T5 embeddings
# -------------------------
set_seed(SEED)

# GearNet splits provide accession membership and labels
xg_train, yg_train, train_ids_all = load_split_pt(TRAIN_PT)
xg_valid, yg_valid, valid_ids_all = load_split_pt(VALID_PT)
xg_test,  yg_test,  test_ids_all  = load_split_pt(TEST_PT)

# We only use GearNet x arrays for reading ids/labels, not as model input
del xg_train, xg_valid, xg_test

# Load T5 embeddings
embed_ids = np.load(TRAIN_IDS_NPY, allow_pickle=True).astype(str)
embeds = np.load(TRAIN_EMBEDS_NPY).astype(np.float32)

# Rebuild each split using the SAME accession IDs
x_train, train_ids, missing_train = build_x_for_ids(train_ids_all, embed_ids, embeds, "train")
x_valid, valid_ids, missing_valid = build_x_for_ids(valid_ids_all, embed_ids, embeds, "valid")
x_test,  test_ids,  missing_test  = build_x_for_ids(test_ids_all,  embed_ids, embeds, "test")

# Align labels to the matched IDs
y_train = align_rows_to_found_ids(train_ids_all, yg_train, train_ids)
y_valid = align_rows_to_found_ids(valid_ids_all, yg_valid, valid_ids)
y_test  = align_rows_to_found_ids(test_ids_all,  yg_test,  test_ids)

if y_train.shape[1] != y_valid.shape[1] or y_train.shape[1] != y_test.shape[1]:
    raise ValueError("Label dimension mismatch across train/valid/test")

with open(LABEL_VOCAB_JSON) as f:
    vocab = json.load(f)

idx_to_term = vocab["idx_to_term"]
idx_to_aspect = vocab["idx_to_aspect"]
NUM_LABELS = len(idx_to_term)

if NUM_LABELS != y_train.shape[1]:
    raise ValueError(
        f"label_vocab.json has {NUM_LABELS} labels but tensors have {y_train.shape[1]}"
    )

print("train", x_train.shape, y_train.shape)
print("valid", x_valid.shape, y_valid.shape)
print("test ", x_test.shape, y_test.shape)

# -------------------------
# ontology / eval inputs
# -------------------------
train_terms = pd.read_csv(TRAIN_TERMS_TSV, sep="\t")
train_terms["EntryID"] = train_terms["EntryID"].astype(str)

ia_dict = ia_parser(IA_TXT)
ontologies = {
    ns: Graph(ns, terms, ia_dict)
    for ns, terms in obo_parser(GO_OBO).items()
}

# -------------------------
# model
# -------------------------
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

    tf.keras.layers.Dense(NUM_LABELS, activation="sigmoid"),
])

model.compile(
    optimizer=tf.keras.optimizers.AdamW(1e-3),
    loss="binary_crossentropy",
    metrics=["binary_accuracy", tf.keras.metrics.AUC(name="auc")],
)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_valid, y_valid),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    # callbacks=[
    #     CafaApproxCallback(
    #         x_valid=x_valid,
    #         valid_ids=valid_ids,
    #         idx_to_term=idx_to_term,
    #         train_terms=train_terms,
    #         ontologies=ontologies,
    #         tau_arr=TAU_ARR,
    #         batch_size=BATCH_SIZE,
    #     )
    # ],
    verbose=1,
)

# -------------------------
# final test
# -------------------------
test_gts = build_ground_truth(test_ids, train_terms, ontologies)
test_pred_maps = precompute_prediction_maps(test_ids, idx_to_term, ontologies, test_gts)

test_pred = model.predict(x_test, batch_size=BATCH_SIZE, verbose=1)
df_ns, mean_f, mean_wf = cafa_weighted_fmax_cached(
    test_pred,
    ontologies,
    test_gts,
    test_pred_maps,
    TAU_ARR,
)

print("\nFINAL TEST")
print(df_ns.to_string(index=False))
print("mean_Fmax :", round(float(mean_f), 6))
print("mean_WFmax:", round(float(mean_wf), 6))