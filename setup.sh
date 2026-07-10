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

echo "Project root : $PROJECT_ROOT"
echo "Analysis code: $ANALYSIS_CODE"
echo "To start: apptainer shell --nv --env PYTHONNOUSERSITE=1 $ANALYSIS_CODE/ml4fp-sbi-tutorial.sif"
[[ -f "$ANALYSIS_CODE/ml4fp-sbi-tutorial.sif" ]] || echo "NOTE: container .sif not built yet — run the apptainer build command above"
