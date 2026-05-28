#!/usr/bin/env bash
# setup_env.sh
# End-to-end environment bootstrap for TorchDrug / ESM-GearNet on Colab-style hosts.
# Run with:  bash setup_env.sh
#
# Notes:
#   * No Jupyter "!" magics. Plain shell only.
#   * Assumes you are root (Colab is). If not, prefix apt-get with sudo.
#   * Assumes /content exists and is writable (Colab convention).

set -euxo pipefail

# ------------------------------------------------------------------
# 0. Paths / constants
# ------------------------------------------------------------------
WORKDIR=/content
WHEEL_DIR="${WORKDIR}/wheels"
MAMBA_BIN_DIR="${WORKDIR}/bin"
MAMBA_ROOT_PREFIX="${WORKDIR}/micromamba"
ENV_NAME=torchdrug38
ENV_PREFIX="${MAMBA_ROOT_PREFIX}/envs/${ENV_NAME}"

mkdir -p "${WORKDIR}" "${WHEEL_DIR}" "${MAMBA_BIN_DIR}"
cd "${WORKDIR}"

# ------------------------------------------------------------------
# 1. System-pip preliminaries (these live in the host Python, NOT the
#    micromamba env. Only keep what the host actually needs, e.g.
#    gsutil-adjacent stuff. If you don't need any of these in the host
#    Python, delete this whole block.)
# ------------------------------------------------------------------
python3 -m pip install -q -U crcmod
python3 -m pip install -q mdtraj webdataset

# ------------------------------------------------------------------
# 2. Pull prebuilt wheels from GCS
# ------------------------------------------------------------------
gsutil -m cp -r gs://protein_shards/wheels/* "${WHEEL_DIR}/"

# ------------------------------------------------------------------
# 3. System build tools (need this BEFORE conda envs that compile)
# ------------------------------------------------------------------
apt-get update
apt-get install -y build-essential curl ca-certificates git

# ------------------------------------------------------------------
# 4. micromamba bootstrap
# ------------------------------------------------------------------
if [ ! -x "${MAMBA_BIN_DIR}/micromamba" ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C "${WORKDIR}" bin/micromamba
fi

export MAMBA_ROOT_PREFIX
MICROMAMBA="${MAMBA_BIN_DIR}/micromamba"

# ------------------------------------------------------------------
# 5. Detect GPU generation (Blackwell vs older)
# ------------------------------------------------------------------
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
echo "Detected GPU: ${GPU_NAME:-<none>}"

IS_BLACKWELL=0
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

# ------------------------------------------------------------------
# 6. (Re)create the conda env
# ------------------------------------------------------------------
rm -rf "${ENV_PREFIX}"

if [ "${IS_BLACKWELL}" = "1" ]; then
  # Blackwell needs CUDA 12.8 wheels, which require Python >= 3.10
  "${MICROMAMBA}" create -y -n "${ENV_NAME}" -c conda-forge \
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
    ninja \
    h5py
else
  # Older GPUs: stay on the known-good torch 1.13.1 + cu117 stack
  "${MICROMAMBA}" create -y -n "${ENV_NAME}" -c conda-forge \
    python=3.8 \
    python-lmdb \
    cffi \
    pip \
    ninja \
    h5py
fi

PY="${ENV_PREFIX}/bin/python"
PIP="${PY} -m pip"

# Put env binaries first on PATH for the rest of the script
export PATH="${ENV_PREFIX}/bin:${PATH}"

# ------------------------------------------------------------------
# 7. Core packaging tools inside the env
# ------------------------------------------------------------------
${PIP} install --no-cache-dir -U pip setuptools wheel

# ------------------------------------------------------------------
# 8. Torch + PyG + TorchDrug
# ------------------------------------------------------------------
if [ "${IS_BLACKWELL}" = "1" ]; then
  # PyTorch 2.7 + CUDA 12.8
  ${PIP} install --no-cache-dir \
    torch==2.7.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

  # remove anything mismatched before installing PyG companions
  ${PIP} uninstall -y \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    torch_geometric torchdrug || true

  ${PIP} install --no-cache-dir \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

  ${PIP} install --no-cache-dir torch_geometric

  # TorchDrug pinned without deps so pip doesn't downgrade torch
  ${PIP} install --no-cache-dir --no-deps torchdrug
  ${PIP} install --no-cache-dir --no-deps webdataset
  ${PIP} install --no-cache-dir decorator networkx jinja2 fair-esm
else
  # PyTorch 1.13.1 + CUDA 11.7
  ${PIP} install --no-cache-dir \
    torch==1.13.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

  ${PIP} uninstall -y \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    torch_geometric || true

  ${PIP} install --no-cache-dir \
    pyg_lib==0.3.1+pt113cu117 \
    torch_scatter==2.1.1+pt113cu117 \
    torch_sparse==0.6.17+pt113cu117 \
    torch_cluster==1.6.1+pt113cu117 \
    torch_spline_conv==1.2.2+pt113cu117 \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

  ${PIP} install --no-cache-dir torch_geometric==2.3.1
  ${PIP} install --no-cache-dir rdkit-pypi torchdrug webdataset pandas tqdm fair-esm
fi

# ------------------------------------------------------------------
# 9. atom3d (needed by ESM-GearNet examples)
# ------------------------------------------------------------------
${PIP} install --no-cache-dir atom3d

# ------------------------------------------------------------------
# 10. Pin rdkit + ninja via conda (Blackwell branch only - older branch
#     already got rdkit-pypi via pip)
# ------------------------------------------------------------------
if [ "${IS_BLACKWELL}" = "1" ]; then
  "${MICROMAMBA}" install -y -n "${ENV_NAME}" -c conda-forge \
    "rdkit=2023.09.*" ninja
fi

# ------------------------------------------------------------------
# 11. Rust toolchain + pinned tokenizers/transformers
#     (transformers 4.14.1 needs to build tokenizers 0.10.3 from src)
# ------------------------------------------------------------------
if ! command -v rustc >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
fi
export PATH="${HOME}/.cargo/bin:${PATH}"

${PIP} install --no-cache-dir "tokenizers==0.10.3" "transformers==4.14.1"

# ------------------------------------------------------------------
# 12. Clone ESM-GearNet (idempotent)
# ------------------------------------------------------------------
if [ ! -d "${WORKDIR}/ESM-GearNet" ]; then
  git clone https://github.com/DeepGraphLearning/ESM-GearNet.git "${WORKDIR}/ESM-GearNet"
fi

# ------------------------------------------------------------------
# 13. Verification
# ------------------------------------------------------------------
MPLBACKEND=Agg "${PY}" - <<'PYCODE'
import os, sys, shutil, importlib

print("python:", sys.version)
print("python_executable:", sys.executable)

try:
    import torch
    print("torch:", torch.__version__)
    print("torch_cuda:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu_name:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
except Exception as e:
    print("torch import failed:", repr(e))

print("ninja_on_path:", shutil.which("ninja"))

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
    "atom3d",
    "h5py",
    "tokenizers",
    "transformers",
]
for name in mods:
    try:
        m = importlib.import_module(name)
        print(f"{name}: OK", getattr(m, "__version__", ""))
    except Exception as e:
        print(f"{name}: FAIL {repr(e)}")

try:
    from torchdrug import models
    print("GearNet exists:", hasattr(models, "GearNet"))
except Exception as e:
    print("torchdrug models import failed:", repr(e))
PYCODE

echo "setup_env.sh finished OK"
