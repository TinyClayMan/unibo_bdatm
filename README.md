# CAFA-5 Protein Function Prediction - Project Overview

This repository contains notebooks and scripts for predicting protein Gene Ontology (GO) function annotations on the [CAFA-5](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction) dataset. The project explores multiple neural architectures and embedding strategies and evaluates them with the official CAFA metric suite (Fmax / weighted Fmax).

---

## Repository Structure

```
.
├── README.md
├── scripts/
│   └── setup_env.sh                          # One-shot environment bootstrap (micromamba + PyTorch + TorchDrug)
├── v7_CAFA_GNN_AlphaFold_Big_Data_Project     # Notebook 1 - data pipeline, GearNet & ESM-GearNet embeddings, MLP baselines
│   .ipynb
└── GGN_GO_CAFA5_v3.ipynb                     # Notebook 2 - GGN-GO graph network, feature generation, fine-tuning
```

---

## What the Project Does

The project addresses **multi-label protein function prediction**: given a protein's 3D structure and/or sequence, predict which GO terms it is annotated with across three ontologies - Biological Process (BPO), Molecular Function (MFO), and Cellular Component (CCO).

The overall pipeline is:

```
AlphaFold PDB structures
        │
        ├─► GearNet graph embeddings  ──┐
        ├─► ESM-GearNet embeddings    ──┤
        ├─► T5-ProtTrans embeddings   ──┼─► MLP / GGN-GO head ─► GO term scores ─► CAFA Fmax
        ├─► ESM2 embeddings           ──┤
        └─► DSSP secondary structure  ──┘
```

### Notebook 1 - `v7_CAFA_GNN_AlphaFold_Big_Data_Project.ipynb`

This notebook handles **data ingestion, structural processing, embedding extraction, MLP training, and GCS data management**. Key steps:

1. **GCS data download** - pulls CAFA-5 annotation files (`train_terms.tsv`, `train_taxonomy.tsv`), GO ontology (`go-basic.obo`, `IA.txt`), and AlphaFold PDB shards from a GCS bucket.
2. **Shard processing** - extracts per-protein `.pdb` files from `.tar` shards, organises them into an evaluation pack (`ggngo_eval_pack.tar`) with defined train / validation / test splits.
3. **GearNet embedding extraction** - runs `gearnet_go_finetune_pipeline.py` under a `micromamba`-managed `torchdrug38` environment to produce per-protein graph embeddings (saved as `.pt` split files).
4. **ESM-GearNet embedding extraction** - runs `esm_gearnet_unified_embedding_pipeline.py` with the `mc_esm_gearnet.pth` checkpoint; handles both AlphaFold and experimental PDB structures.
5. **MLP baselines** - trains multi-label MLP heads on T5-ProtTrans, GearNet, and ESM-GearNet embeddings via `train_go_mlp_cafa_from_saved_splits*.py`; evaluates with CAFA Fmax/WFmax.
6. **Bilinear fusion** - trains a cross-attention bilinear fusion of GearNet + T5 embeddings.
7. **Structure-swap experiment** - compares model performance on AlphaFold vs experimental PDB structures for the same proteins.
8. **GCS sync** - uploads trained model checkpoints and result tarballs back to the GCS bucket.

### Notebook 2 - `GGN_GO_CAFA5_v3.ipynb`

This notebook implements the full **GGN-GO graph neural network pipeline** for structure-aware protein function prediction. Key steps:

1. **Configuration** - a single cell sets the runtime (`colab` / `local`), dataset mode (`demo` / `full`), task (`bp` / `mf` / `cc`), and hyperparameters.
2. **GGN-GO repository setup** - clones [MiJia-ID/GGN-GO](https://github.com/MiJia-ID/GGN-GO) and installs PyG, `fair-esm`, `transformers`, `biopython`, and `obonet`.
3. **Data acquisition** - downloads annotation files from GCS or falls back to Kaggle / OBO Foundry instructions.
4. **Feature generation** - per-residue features for each protein:
   - **Backbone coordinates** (N, CA, C, O) from AlphaFold PDBs via BioPython
   - **ProtTrans embeddings** (1024-d) from `Rostlab/prot_t5_xl_half_uniref50-enc`
   - **ESM2 embeddings** (1280-d) from `esm2_t33_650M_UR50D`
   - **DSSP features** (14-d: sin/cos φ/ψ, relative ASA, 9-class SS one-hot) via `mkdssp` installed through `micromamba`
5. **Graph dataset** - a `CAFA5ProteinGraphDataset` builds k-NN residue graphs (k=30) with scalar node features (2324-d total) and vector node/edge features for GVP layers.
6. **GGN-GO model** - loads the pretrained `model_bp/mf/cc.pt` checkpoint; replaces the final readout head for the CAFA-5 label vocabulary; optionally freezes the encoder.
7. **Zero-shot evaluation** - runs the unmodified pretrained GGN-GO checkpoint on the CAFA-5 validation split for BP, MF, and CC separately.
8. **Fine-tuning** - trains with `BCELoss` + optional NT-Xent contrastive loss, with early stopping on validation loss.
9. **Cross-embedding MLP training** (`train_mlp_clean_torch.py`) - trains lightweight MLP or GGN-style heads on top of T5, GearNet (`.pt`), or GGN-GO (`.npz`) embeddings; supports `same` split (GGN-GO mini-split) and `all` split (full CAFA-5 train/val); evaluates with full CAFA Fmax/WFmax.
10. **Naive frequency baseline** (`eval_naive_freq_cafa_small.py`) - computes a term-frequency baseline for comparison.

### `scripts/setup_env.sh`

A self-contained bash script that bootstraps the full `torchdrug38` conda environment via `micromamba`. It:

- Detects the GPU generation (Blackwell vs older) and selects the correct PyTorch wheel (2.7+cu128 or 1.13.1+cu117).
- Installs PyTorch Geometric extensions (`pyg_lib`, `torch_scatter`, `torch_sparse`, `torch_cluster`, `torch_spline_conv`) from the matching PyG index.
- Installs TorchDrug, `webdataset`, `fair-esm`, and `tokenizers`/`transformers` (with Rust for tokenizers).
- Runs a verification script to confirm all critical imports.

Called from the notebook via:
```bash
!bash {SCRIPTS_DIR}/setup_env.sh
```

---

## Setup Guide
### Option A: Google Colab (recommended for first-time use)

1. **Upload the notebooks** to Google Colab (`File → Upload notebook`).
2. Set the runtime to **GPU** (`Runtime → Change runtime type → T4 GPU` or better).
3. **Mount Google Drive** if you want to persist checkpoints between sessions (the environment detection cell handles this automatically when `IN_COLAB = True`).
4. Run the environment setup cell in CNN notebook:
   ```python
   !bash {SCRIPTS_DIR}/setup_env.sh
   ```
   This will install micromamba and build the `torchdrug38` environment (~10–20 min on first run).
5. In GGN_GO notebook, set `RUNTIME = "colab"` in the configuration cell, then run all cells in order.

#### GCS credentials (Colab)
The notebooks authenticate against the GCS bucket using a service account key. Store your own service account credentials in the `gcs_key.json` in the root folder, or set `USE_GCS = False` and supply the data files manually.

#### Without GCS access
Download the required files from the [CAFA-5 Kaggle competition](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction) and place them in the `cafa_data/` directory:
- `train_terms.tsv`
- `train_taxonomy.tsv`
- `go-basic.obo` (also available at `http://purl.obolibrary.org/obo/go/go-basic.obo`)
- `IA.txt`

AlphaFold structures are downloaded automatically in `demo` mode from the [AlphaFold database API](https://alphafold.ebi.ac.uk). For the full dataset (~120,000 proteins), use the GCS shards or pre-download all structures.

---

### Option B: Local Linux Environment

1. **Clone or copy** the notebooks and the `scripts/` directory to your working directory.
2. **Run the environment setup script:**
   ```bash
   bash scripts/setup_env.sh
   ```
   This creates a `micromamba` environment at `/content/micromamba/envs/torchdrug38` (path is hardcoded for Colab compatibility; adjust `MAMBA_ROOT_PREFIX` in the script if needed for local use).
3. **Activate the environment** for standalone script runs:
   ```bash
   export MAMBA_ROOT_PREFIX=/content/micromamba
   /content/bin/micromamba run -n torchdrug38 python your_script.py
   ```
4. **Set `RUNTIME = "local"`** in the GGN_GO notebook configuration cell. This sets `BASE_DIR` to `./ggn_go_workspace` and `ROOT_DIR` to the current working directory instead of `/content`.
5. **Place data files** in the paths expected by each notebook (or adjust `ROOT_DIR` / `DATA_DIR` at the top of each notebook).

#### Local pip-only environment (alternative)

If you do not need TorchDrug / GearNet (i.e., you only want to run GGN_GO notebook), you can skip `setup_env.sh` and install directly:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter torch-cluster torch-geometric \
    -f https://data.pyg.org/whl/torch-2.x.x+cu118.html
pip install ml-collections biopython fair-esm transformers sentencepiece obonet
```

Then install DSSP via conda-forge or system packages:
```bash
conda install -c conda-forge dssp
# or on Debian/Ubuntu:
sudo apt-get install dssp
```

---

## Data Files Reference

| File | Source | Used in |
|------|--------|---------|
| `train_terms.tsv` | CAFA-5 Kaggle / GCS | Both notebooks |
| `train_taxonomy.tsv` | CAFA-5 Kaggle / GCS | GNN otebook, GGN_GO notebook |
| `go-basic.obo` | OBO Foundry / GCS | Both notebooks |
| `IA.txt` | GCS (`gs://protein_shards/IA.txt`) | Both notebooks |
| `shards/*.tar` | GCS (`gs://protein_shards/shards/`) | GNN otebook |
| `ggngo_eval_pack.tar` | Produced by GNN notebook / GCS | GGN_GO notebook (`full` mode) |
| `gearnet_embeds.zip` / `.tar` | Produced by GNN notebook / GCS | GGN_GO notebook (MLP training) |
| `esm_gearnet_all_pdbs_embeds.tar` | Produced by GNN notebook / GCS | GGN_GO notebook (MLP training) |
| `t5_prot_baseline_train_embeds.npy.zip` | GCS | GNN otebook, GGN_GO notebook (T5 baseline) |
| `mc_gearnet_edge.pth` | [TorchDrug model zoo](https://torchdrug.ai) | GNN otebook (GearNet) |
| `mc_esm_gearnet.pth` | [Zenodo 10034578](https://zenodo.org/records/10034578) | GNN otebook (ESM-GearNet) |
| `GGN-GO/Model/model_bp/mf/cc.pt` | Cloned from [MiJia-ID/GGN-GO](https://github.com/MiJia-ID/GGN-GO) | GGN_GO notebook |

---

## Evaluation

All models are evaluated using the CAFA competition metrics implemented in `cafa.py` (written inline in GGN_GO notebook):

- **Fmax** - the maximum F-score over all decision thresholds τ ∈ [0.01, 0.99], computed separately per ontology namespace and averaged.
- **Weighted Fmax (WFmax)** - same as Fmax but GO term contributions are weighted by their Information Accretion (IA) values from `IA.txt`.

Results are reported for BPO, MFO, and CCO independently, plus mean Fmax and mean WFmax across all three.

---

## References

- [GGN-GO: Graph neural network for Gene Ontology annotation](https://github.com/MiJia-ID/GGN-GO)
- [CAFA-5: Critical Assessment of Protein Function Annotation](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction)
- [TorchDrug / GearNet](https://torchdrug.ai)
- [ESM-GearNet (Zenodo)](https://zenodo.org/records/10034578)
- [ProtTrans (Rostlab/prot_t5_xl_half_uniref50-enc)](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc)
- [ESM2 (facebookresearch/esm)](https://github.com/facebookresearch/esm)
- [GVP: Geometric Vector Perceptrons (Jing et al., 2021)](https://arxiv.org/abs/2106.03843)
