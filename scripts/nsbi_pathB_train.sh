#!/bin/bash
# CARL path-B training wrapper (HTCondor GPU job) — one classifier per invocation.
# The job itself runs INSIDE ml4fp-sbi-tutorial.sif (condor +SingularityImage in the .sub),
# so this script calls python directly — no nested apptainer.
# Usage: nsbi_pathB_train.sh <run_name> <numerator_csv>
# Layout satisfies carl.utils.load_results(<RUN>, <run_name>): data_dir and the
# lightning_logs save_dir must both be <RUN>/<run_name> (see docs/nsbi-repro-prep.md §1.3).
set -euo pipefail
RUN_NAME=$1
NUM_CSV=$2

ROOT=/project/atlas/Users/mfernand/SBI_shortcomings
SUB=$ROOT/asimov-aymptotics-check
DATA=$ROOT/data/zz4l/ggzz4l_sm
RUN=$ROOT/output/nsbi-pathB/h4l
FEATS="[l1_pt,l1_eta,l1_phi,l1_energy,l2_pt,l2_eta,l2_phi,l2_energy,l3_pt,l3_eta,l3_phi,l3_energy,l4_pt,l4_eta,l4_phi,l4_energy]"

export PYTHONNOUSERSITE=1                       # keep ~/.local from shadowing the container
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/cache"

echo "host=$(hostname)  run=$RUN_NAME  gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi not available"

mkdir -p "$RUN/$RUN_NAME"
cd "$SUB"   # nsbi/, models/, datasets/ are importable only from the sub-repo root

exec python -m nsbi.carl fit \
    --seed_everything 42 \
    --model.n_layers 3 --model.n_nodes 128 --model.learning_rate 1e-3 \
    --data.numerator_events "$NUM_CSV" \
    --data.denominator_events "$DATA/ggzz4l_bkg.csv" \
    --data.features "$FEATS" \
    --data.batch_size 1024 \
    --data.data_dir "$RUN/$RUN_NAME" \
    --trainer.accelerator gpu --trainer.devices 1 \
    --trainer.max_epochs 200 \
    --trainer.logger.init_args.save_dir "$RUN/$RUN_NAME"
