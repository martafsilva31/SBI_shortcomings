#!/bin/bash
# CARL path-B training wrapper (HTCondor GPU job) — one classifier per invocation.
# The job itself runs INSIDE ml4fp-sbi-tutorial.sif (condor +SingularityImage in the .sub),
# so this script calls python directly — no nested apptainer.
# Usage: nsbi_pathB_train.sh <run_name> <numerator_csv> [n_layers] [n_nodes] [lr]
# Hyperparameters default to the original path-B settings; pass overrides for retrains.
# Early-stop patience via env CARL_EARLYSTOP_PATIENCE (default 20; models/carl.py).
# Layout satisfies carl.utils.load_results(<RUN>, <run_name>): data_dir and the
# lightning_logs save_dir must both be <RUN>/<run_name> (see docs/nsbi-repro-prep.md §1.3).
set -euo pipefail
RUN_NAME=$1
NUM_CSV=$2
N_LAYERS=${3:-3}
N_NODES=${4:-128}
LR=${5:-1e-3}

ROOT=/project/atlas/Users/mfernand/SBI_shortcomings
SUB=$ROOT/asimov-aymptotics-check
DATA=$ROOT/data/zz4l/ggzz4l_sm
RUN=$ROOT/output/nsbi-pathB/h4l
FEATS="[l1_pt,l1_eta,l1_phi,l1_energy,l2_pt,l2_eta,l2_phi,l2_energy,l3_pt,l3_eta,l3_phi,l3_energy,l4_pt,l4_eta,l4_phi,l4_energy]"

export PYTHONNOUSERSITE=1                       # keep ~/.local from shadowing the container
export PYTHONPATH="$ROOT/output/pyextra${PYTHONPATH:+:$PYTHONPATH}"   # jsonargparse[signatures] for LightningCLI (not in container.def)
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/cache"

echo "host=$(hostname)  run=$RUN_NAME  gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi not available"

mkdir -p "$RUN/$RUN_NAME"
cd "$SUB"   # nsbi/, models/, datasets/ are importable only from the sub-repo root

echo "hyperparams: n_layers=$N_LAYERS n_nodes=$N_NODES lr=$LR patience=${CARL_EARLYSTOP_PATIENCE:-20}"
exec python -m nsbi.carl fit \
    --seed_everything 42 \
    --model.n_layers $N_LAYERS --model.n_nodes $N_NODES --model.learning_rate $LR \
    --data.numerator_events "$NUM_CSV" \
    --data.denominator_events "$DATA/ggzz4l_bkg.csv" \
    --data.features "$FEATS" \
    --data.batch_size 1024 \
    --data.data_dir "$RUN/$RUN_NAME" \
    --trainer.accelerator gpu --trainer.devices 1 \
    --trainer.max_epochs 200 \
    --trainer.logger.init_args.save_dir "$RUN/$RUN_NAME"
