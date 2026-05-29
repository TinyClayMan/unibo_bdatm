# CAFA-5 Protein Function Prediction - Project Overview

This repository contains notebooks and scripts for predicting protein Gene Ontology (GO) function annotations on the [CAFA-5](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction) dataset. The project explores multiple neural architectures and embedding strategies and evaluates them with the official CAFA metric suite (Fmax / weighted Fmax).

---

## Repository Structure

```
.
├── 00_Download_AlphaDB_PDBs_for_CAFA_v5.ipynb            # Step 0 - download AlphaFold PDBs for CAFA-5
├── 01_Extract_Embeddings_GearNet_ESM_GearNet.ipynb      # Step 1 - GearNet & ESM-GearNet embeddings (+ experimental PDBs)
├── 02_Extract_Embeddings_GGN_GO.ipynb                   # Step 2 - GGN-GO embeddings (inference-only, mini split)
├── 03_Training_of_all_our_models_GGN_GO_CAFA5_v3.ipynb  # Step 3 - train all heads, evaluate, structure-swap, baseline
├── scripts/
│   ├── setup_env.sh                                     # Environment bootstrap (micromamba + PyTorch + TorchDrug)
│   ├── run_embed_pipeline.py                            # Entry point -> unified_embedding_extractor.cli
│   ├── download_best_experimental_pdbs.py               # UniProt -> RCSB experimental-PDB selector / downloader
│   ├── unified_embedding_extractor/                     # GearNet / ESM-GearNet embedding package (notebook 01)
│   └── unified_cafa_model_trainer/                      # CAFA trainer + evaluator (notebook 03)
├── README.md
└── README_new.md
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

### Notebook `00` - `00_Download_AlphaDB_PDBs_for_CAFA_v5.ipynb`

Downloads AlphaFold-DB PDB structures for the ~142k UniProt accessions listed in CAFA-5 `train_taxonomy.tsv`, organises them into shards, and uploads the shards to the GCS bucket (`gs://protein_shards/`). Each shard's output is kept in the notebook - it is not necessary to rerun to be able to launch subsequent steps.

### Notebook `01` - `01_Extract_Embeddings_GearNet_ESM_GearNet.ipynb`

Extracts per-protein embeddings with two **frozen pretrained encoders** - GearNet (`mc_gearnet_edge.pth`) and ESM-GearNet (`mc_esm_gearnet.pth`) - by running our `scripts/unified_embedding_extractor/` package (via `scripts/run_embed_pipeline.py`) over the PDB shards. It then downloads matching wet-lab PDB structures for the valid / test proteins (`scripts/download_best_experimental_pdbs.py`: UniProt -> RCSB, resolution-filtered) and re-embeds them with the same encoders for the structure-swap experiment. Outputs are `<split>_embeddings.pt` files per encoder, saved to GCS.

### Notebook `02` - `02_Extract_Embeddings_GGN_GO.ipynb`

Does **GGN-GO inference** on the mini split. Clones [MiJia-ID/GGN-GO](https://github.com/MiJia-ID/GGN-GO), builds per-residue multi-modal features (ProtTrans T5 + ESM-2 + DSSP + backbone coordinates), loads the authors' pretrained `model_bp/mf/cc.pt`, and extracts the 512-d pooled protein embedding before the task head. There is no GGN-GO fine-tuning - doing it live is too computationally costly, so this runs on the mini split only.

### Notebook `03` - `03_Training_of_all_our_models_GGN_GO_CAFA5_v3.ipynb`

The training and evaluation notebook. Pulls every embedding tar from GCS, then uses `scripts/unified_cafa_model_trainer/` to:

1. train small `ggn` / `mlp` heads on each embedding source - T5, GearNet, ESM-GearNet, GGN-GO (`train_mlp_clean_torch.py`), with `BCEWithLogitsLoss` and early stopping;
2. run the AlphaFold-vs-experimental structure-swap test (`eval_experimental_vs_alphadb.py`);
3. compute the naive class-frequency baseline (`eval_naive_freq_cafa.py`).

All heads sit on top of frozen encoders (the backbones are never fine-tuned). Every result table in the notebook is backed by the cell output directly below it.

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
4. Run the environment setup cell (only notebook 01, embedding extraction, needs TorchDrug):
   ```python
   !bash {SCRIPTS_DIR}/setup_env.sh
   ```
   This will install micromamba and build the `torchdrug38` environment (~10-20 min on first run).
5. In each notebook's configuration cell set `RUNTIME = "colab"` (or `IN_COLAB = True`), then run notebooks `00 -> 01 -> 02 -> 03` in order.

#### GCS credentials (Colab)
The notebooks authenticate against the GCS bucket using a service account key. Store your own service account credentials in the `gcs_key.json` in the root folder, or set `USE_GCS = False` and supply the data files manually.

#### Without GCS access
Download the required files from the [CAFA-5 Kaggle competition](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction) and place them in the `cafa_data/` directory:
- `train_terms.tsv`
- `train_taxonomy.tsv`
- `go-basic.obo` (also available at `http://purl.obolibrary.org/obo/go/go-basic.obo`)
- `IA.txt`

AlphaFold structures are downloaded automatically in `demo` mode from the [AlphaFold database API](https://alphafold.ebi.ac.uk). For the full dataset (~142,000 proteins), use the GCS shards or pre-download all structures.

---

### Option B: Local Linux Environment (less recommended, untested because of CUDA)

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
4. **Set `RUNTIME = "local"`** in the notebook configuration cell. This sets `BASE_DIR` to `./ggn_go_workspace` and `ROOT_DIR` to the current working directory instead of `/content`.
5. **Place data files** in the paths expected by each notebook (or adjust `ROOT_DIR` / `DATA_DIR` at the top of each notebook).

#### Local pip-only environment (alternative)

If you do not need TorchDrug / GearNet (i.e., you only want to run notebook `02` and/or `03`), you can skip `setup_env.sh` and install directly:

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
| `train_terms.tsv` | CAFA-5 Kaggle / GCS | Notebooks 01, 03 (labels + eval) |
| `train_taxonomy.tsv` | CAFA-5 Kaggle / GCS | Notebook 00 (accession list) |
| `go-basic.obo` | OBO Foundry / GCS | Notebooks 02, 03 |
| `IA.txt` | GCS (`gs://protein_shards/IA.txt`) | Notebook 03 (WFmax weights) |
| `shards/*.tar` | GCS (`gs://protein_shards/`) | Produced by 00, consumed by 01 |
| `gearnet_embeds.zip` / `.tar` | Produced by notebook 01 / GCS | Notebook 03 (head training) |
| `esm_gearnet_all_pdbs_embeds.tar` | Produced by notebook 01 / GCS | Notebook 03 (head training) |
| `t5_prot_baseline_train_embeds.npy.zip` | GCS | Notebook 03 (T5 baseline) |
| `mc_gearnet_edge.pth` | [TorchDrug model zoo](https://torchdrug.ai) | Notebook 01 (GearNet) |
| `mc_esm_gearnet.pth` | [Zenodo 10034578](https://zenodo.org/records/10034578) | Notebook 01 (ESM-GearNet) |
| `GGN-GO/Model/model_bp/mf/cc.pt` | Cloned from [MiJia-ID/GGN-GO](https://github.com/MiJia-ID/GGN-GO) | Notebook 02 |

---

## Evaluation

All models are evaluated using the CAFA competition metrics implemented in `scripts/unified_cafa_model_trainer/cafa.py`:

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
