#!/usr/bin/env bash
set -euxo pipefail

# --- pip: crcmod ---
pip -q install -U crcmod
# --- pip: mdtraj (torch-scatter/cluster/geometric left commented) ---
#!pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
#!pip install torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
#!pip install torch_geometric -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
!pip install mdtraj

# --- pip: webdataset ---
!pip install webdataset

# --- GCS: download prebuilt wheels ---
!gsutil -m cp -r gs://protein_shards/wheels/* /content/wheels/

# --- pip: install from local wheels ---
!pip install --no-index --find-links=/content/wheels/torch2.10.0-cu128-py312 \
  torch-scatter torch-cluster
!pip install torch-geometric

# --- micromamba: bootstrap + PyTorch + PyG + TorchDrug ---
set -euxo pipefail

cd /content

# -----------------------------
# micromamba bootstrap
# -----------------------------
if [ ! -x /content/bin/micromamba ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
fi

export MAMBA_ROOT_PREFIX=/content/micromamba
MICROMAMBA=/content/bin/micromamba
ENV_NAME=torchdrug38

# -----------------------------
# detect GPU generation
# -----------------------------
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || true)"
echo "Detected GPU: ${GPU_NAME}"

IS_BLACKWELL=0

# Name-based detection first
shopt -s nocasematch
if [[ "${GPU_NAME}" == *"RTX 50"* ]] || \
   [[ "${GPU_NAME}" == *"5090"* ]] || \
   [[ "${GPU_NAME}" == *"5080"* ]] || \
   [[ "${GPU_NAME}" == *"5070"* ]] || \
   [[ "${GPU_NAME}" == *"5060"* ]] || \
   [[ "${GPU_NAME}" == *"5050"* ]] || \
   [[ "${GPU_NAME}" == *"Blackwell"* ]] || \
   [[ "${GPU_NAME}" == *"GB200"* ]] || \
   [[ "${GPU_NAME}" == *"B200"* ]] || \
   [[ "${GPU_NAME}" == *"B100"* ]] || \
   [[ "${GPU_NAME}" == *"RTX PRO 6000 Blackwell"* ]]; then
  IS_BLACKWELL=1
fi
shopt -u nocasematch

echo "IS_BLACKWELL=${IS_BLACKWELL}"

# -----------------------------
# system build tools
# -----------------------------
apt-get update
apt-get install -y build-essential

# -----------------------------
# recreate env
# -----------------------------
rm -rf "/content/micromamba/envs/${ENV_NAME}"

if [ "${IS_BLACKWELL}" = "1" ]; then
  # ---------------------------------
  # Blackwell path
  # Python 3.10 because current stable PyTorch requires 3.10+,
  # and TorchDrug compatibility advertises support through 3.10.
  # ---------------------------------
  $MICROMAMBA create -y -n "${ENV_NAME}" -c conda-forge \
    python=3.10 \
    "numpy<2" \
    python-lmdb \
    cffi \
    pip \
    rdkit \
    pandas \
    tqdm \
    scipy \
    matplotlib \
    ninja
else
  # ---------------------------------
  # Older / pre-Blackwell path
  # Matches your working Torch 1.13.1 + cu117 stack
  # ---------------------------------
  $MICROMAMBA create -y -n "${ENV_NAME}" -c conda-forge \
    python=3.8 \
    python-lmdb \
    cffi \
    pip \
    ninja
fi

PY="/content/micromamba/envs/${ENV_NAME}/bin/python"
PIP="$PY -m pip"

# Put env binaries first on PATH
export PATH="/content/micromamba/envs/${ENV_NAME}/bin:${PATH}"

# -----------------------------
# core packaging tools
# -----------------------------
$PIP install --no-cache-dir -U pip setuptools wheel

# -----------------------------
# install torch / pyg / torchdrug
# -----------------------------
if [ "${IS_BLACKWELL}" = "1" ]; then
  # PyTorch 2.7 + CUDA 12.8 for Blackwell
  $PIP install --no-cache-dir \
    torch==2.7.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

  # remove anything mismatched first
  $PIP uninstall -y \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric torchdrug || true

  # matching PyG compiled wheels
  $PIP install --no-cache-dir \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

  $PIP install --no-cache-dir torch_geometric

  # TorchDrug without letting pip "helpfully" rebuild deps
  $PIP install --no-cache-dir --no-deps torchdrug

  # extras commonly needed around ESM-GearNet / TorchDrug
  $PIP install --no-cache-dir --no-deps webdataset
  $PIP install --no-cache-dir decorator networkx jinja2 fair-esm
else
  # PyTorch 1.13.1 + CUDA 11.7 for older GPUs
  $PIP install --no-cache-dir \
    torch==1.13.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

  $PIP uninstall -y \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric || true

  $PIP install --no-cache-dir \
    pyg_lib==0.3.1+pt113cu117 \
    torch_scatter==2.1.1+pt113cu117 \
    torch_sparse==0.6.17+pt113cu117 \
    torch_cluster==1.6.1+pt113cu117 \
    torch_spline_conv==1.2.2+pt113cu117 \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

  $PIP install --no-cache-dir torch_geometric==2.3.1
  $PIP install --no-cache-dir rdkit-pypi torchdrug webdataset pandas tqdm fair-esm
fi

# -----------------------------
# verification
# -----------------------------
MPLBACKEND=Agg $PY - <<'PYCODE'

print("python:", os.sys.version)
print("python_executable:", os.sys.executable)
print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("ninja_on_path:", shutil.which("ninja"))

if torch.cuda.is_available():
    print("gpu_name:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))

mods = [
    "lmdb",
    "pyg_lib",
    "torch_scatter",
    "torch_sparse",
    "torch_cluster",
    "torch_spline_conv",
    "torch_geometric",
    "rdkit",
    "torchdrug",
]

for name in mods:
    try:
        mod = __import__(name)
        print(f"{name}: OK", getattr(mod, "__version__", ""))
    except Exception as e:
        print(f"{name}: FAIL {repr(e)}")

try:
    from torchdrug import models
    print("GearNet exists:", hasattr(models, "GearNet"))
except Exception as e:
    print("torchdrug models import failed:", repr(e))
PYCODE

# --- micromamba: install rdkit + ninja ---
set -euxo pipefail

export MAMBA_ROOT_PREFIX=/content/micromamba
MICROMAMBA=/content/bin/micromamba
PY=/content/micromamba/envs/torchdrug38/bin/python

$MICROMAMBA install -y -n torchdrug38 -c conda-forge "rdkit=2023.09.*" ninja

MPLBACKEND=Agg $PY - <<'PY'
print("rdkit", rdkit.__version__)
print("ninja", shutil.which("ninja"))
print("mplCanvas OK")
print("torchdrug OK", torchdrug.__version__)
PY

# --- h5py, atom3d, ESM-GearNet ---
/content/bin/micromamba install -y -p /content/micromamba/envs/torchdrug38 h5py
/content/micromamba/envs/torchdrug38/bin/python -m pip install atom3d
git clone https://github.com/DeepGraphLearning/ESM-GearNet.git

# --- rust + tokenizers/transformers ---
set -euxo pipefail

curl https://sh.rustup.rs -sSf | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"

/content/micromamba/envs/torchdrug38/bin/python -m pip install --no-cache-dir \
  "tokenizers==0.10.3" "transformers==4.14.1"
