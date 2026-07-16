#!/usr/bin/env bash
# Setup for SBI_shortcomings. Source this file: source setup.sh
#
# Environment: Apptainer container + NVIDIA GPU (see asimov-aymptotics-check/container.def)
# No ATLAS Athena required.

export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ANALYSIS_CODE="$PROJECT_ROOT/asimov-aymptotics-check"

# ── Dataset location ─────────────────────────────────────────────────────────
# Local copy of Tei's CERNBox share (https://cernbox.cern.ch/s/AeKNoWslVzwmAdt,
# folder zz4l/ggzz4l_sm/). See docs/context.md for normalisation instructions.
export DATASET_DIR="$PROJECT_ROOT/data/zz4l/ggzz4l_sm"

# ── Container launch (run these manually, not via source) ────────────────────
# First time — build the container (needs Apptainer and ~10 min):
#   apptainer build "$ANALYSIS_CODE/ml4fp-sbi-tutorial.sif" "$ANALYSIS_CODE/container.def"
#
# Every session — launch with GPU access. PYTHONNOUSERSITE=1 is REQUIRED: apptainer binds $HOME,
# and stray ~/.local packages otherwise shadow the container (a corrupt one broke mplhep,
# 2026-07-10 — see docs/hist-reproduction.md):
#   apptainer shell --nv --env PYTHONNOUSERSITE=1 "$ANALYSIS_CODE/ml4fp-sbi-tutorial.sif"
#
# Inside the container — install the package (first time only):
#   pip install -e "$ANALYSIS_CODE"

# ── Notebook virtualenv (run the notebooks WITHOUT the container) ─────────────
# A venv layered on the cvmfs LCG_109_swan view: torch / jupyter / scikit-hep come
# from cvmfs (immutable → reproducible), only `lightning` is pip-added on top. The
# "sbi-notebooks" Jupyter kernel lives in .venv/share/jupyter, so VS Code and
# `jupyter lab` both find it; its kernel.json carries the view's LD_LIBRARY_PATH so
# it launches without sourcing anything. Built once with:
#   source "$LCG_VIEW"
#   python -m venv --system-site-packages "$NB_VENV"
#   source "$NB_VENV/bin/activate" && python -m pip install "lightning==2.5.*"
#   python -m ipykernel install --sys-prefix --name sbi-notebooks \
#                               --display-name "SBI notebooks (LCG_109_swan)"
export LCG_VIEW="/cvmfs/sft.cern.ch/lcg/views/LCG_109_swan/x86_64-el9-gcc13-opt/setup.sh"
export NB_VENV="$PROJECT_ROOT/.venv"
# Keep Jupyter's data/config/runtime + IPython history OFF the quota-limited $HOME
# (write to project fs; $HOME/user quota is exhausted, breaks sqlite/history writes):
export JUPYTER_DATA_DIR="$NB_VENV/share/jupyter"
export JUPYTER_CONFIG_DIR="$NB_VENV/etc/jupyter"
export JUPYTER_RUNTIME_DIR="$NB_VENV/share/jupyter/runtime"
export IPYTHONDIR="$NB_VENV/share/ipython"

# Every session — activate the notebook env in this shell (source view, then venv):
notebooks_env() {
    source "$LCG_VIEW" && source "$NB_VENV/bin/activate" \
      && echo "notebook env ready → 'jupyter lab notebooks/' or pick the 'sbi-notebooks' kernel in VS Code"
}

echo "Project root : $PROJECT_ROOT"
echo "Analysis code: $ANALYSIS_CODE"
echo "To start: apptainer shell --nv --env PYTHONNOUSERSITE=1 $ANALYSIS_CODE/ml4fp-sbi-tutorial.sif"
echo "Notebooks (no container): run 'notebooks_env', then 'jupyter lab notebooks/'  (kernel: sbi-notebooks)"
[[ -f "$ANALYSIS_CODE/ml4fp-sbi-tutorial.sif" ]] || echo "NOTE: container .sif not built yet — run the apptainer build command above"
