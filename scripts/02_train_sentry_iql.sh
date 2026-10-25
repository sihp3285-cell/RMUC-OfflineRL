#!/usr/bin/env bash
# Train the sentry IQL policy on 4 GPUs with DDP.
set -e
NGPU="${NGPU:-4}"

torchrun --standalone --nproc_per_node="$NGPU" \
  -m rm_rl.train.train_offline \
  --config configs/sentry_iql.yaml

# Baselines / alternatives:
#   torchrun --standalone --nproc_per_node=$NGPU -m rm_rl.train.train_offline --config configs/sentry_bc.yaml
#   torchrun --standalone --nproc_per_node=$NGPU -m rm_rl.train.train_dt      --config configs/sentry_dt.yaml
