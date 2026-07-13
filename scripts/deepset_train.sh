#!/bin/bash
# Deep-set prototype training (Phase-3 prong 3, v0). Runs INSIDE the container via
# condor +SingularityImage. Usage: deepset_train.sh <out_subdir> [steps]
set -euo pipefail
OUT_SUB=${1:-v0}
STEPS=${2:-20000}

ROOT=/project/atlas/Users/mfernand/SBI_shortcomings
SUB=$ROOT/asimov-aymptotics-check

export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/output/pyextra:$SUB${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/cache"

echo "host=$(hostname) out=$OUT_SUB steps=$STEPS gpu=${CUDA_VISIBLE_DEVICES:-unset}"
mkdir -p "$ROOT/output/deepset/$OUT_SUB"
cd "$SUB"
exec python models/deepset.py \
    --steps "$STEPS" --batch-size 64 --rows 200000 --workers 6 \
    --out-dir "$ROOT/output/deepset/$OUT_SUB"
