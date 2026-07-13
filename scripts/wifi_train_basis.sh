#!/bin/bash
# wifi-ensemble basis training (HTCondor GPU job) — one basis network f_i per invocation.
# Trains a sig_over_bkg CARL replica on its own Poisson bootstrap of the training split
# (physics/simulation/mcfm.py Process.resample, selected by CARL_BOOTSTRAP_SEED).
# Same container/GPU pattern as nsbi_pathB_train.sh: the job runs INSIDE ml4fp-sbi-tutorial.sif.
# Usage: wifi_train_basis.sh <member_index>
# Hyperparameters are the proven path-B sig_over_bkg recipe (val_loss 0.419) — deliberately
# unchanged, so a member differs from the 2026-07-10 keeper only by its bootstrap.
set -euo pipefail
MEMBER=$1

ROOT=/project/atlas/Users/mfernand/SBI_shortcomings
SUB=$ROOT/asimov-aymptotics-check
DATA=$ROOT/data/zz4l/ggzz4l_sm
RUN=$ROOT/output/wifi/h4l
FEATS="[l1_pt,l1_eta,l1_phi,l1_energy,l2_pt,l2_eta,l2_phi,l2_energy,l3_pt,l3_eta,l3_phi,l3_energy,l4_pt,l4_eta,l4_phi,l4_energy]"

export PYTHONNOUSERSITE=1                       # keep ~/.local from shadowing the container
export PYTHONPATH="$ROOT/output/pyextra${PYTHONPATH:+:$PYTHONPATH}"   # jsonargparse[signatures] for LightningCLI
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/cache"

# the bootstrap that makes this member distinct
export CARL_BOOTSTRAP_SEED=$MEMBER

# GPU nodes here have their CPUs fully subscribed by other users; a small core request is what
# makes these jobs matchable at all (see docs/phase3-fixes.md item 8e). 0 = load in the main process:
# the dataset is already an in-memory numpy array, so worker processes buy nothing and cost a core.
export CARL_NUM_WORKERS=0

RUN_NAME="member_$(printf '%02d' "$MEMBER")"

echo "host=$(hostname)  member=$MEMBER  bootstrap_seed=$CARL_BOOTSTRAP_SEED  gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi not available"

mkdir -p "$RUN/$RUN_NAME"
cd "$SUB"   # nsbi/, models/, datasets/ are importable only from the sub-repo root

# NOTE: --seed_everything is held FIXED across members on purpose. The ensemble's diversity must
# come from the bootstrap (the thing whose uncertainty wifi propagates), not from initialisation
# noise; see docs/phase3-fixes.md item 8e.
exec python -m nsbi.carl fit \
    --seed_everything 42 \
    --model.n_layers 3 --model.n_nodes 128 --model.learning_rate 1e-3 \
    --data.numerator_events "$DATA/ggzz4l_sig.csv" \
    --data.denominator_events "$DATA/ggzz4l_bkg.csv" \
    --data.features "$FEATS" \
    --data.batch_size 1024 \
    --data.data_dir "$RUN/$RUN_NAME" \
    --trainer.accelerator gpu --trainer.devices 1 \
    --trainer.max_epochs 200 \
    --trainer.logger.init_args.save_dir "$RUN/$RUN_NAME"
