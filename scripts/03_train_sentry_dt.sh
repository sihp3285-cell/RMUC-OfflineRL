#!/usr/bin/env bash
# Train the sentry Decision Transformer on 4 GPUs with DDP.
set -e
NGPU="${NGPU:-4}"

torchrun --standalone --nproc_per_node="$NGPU" \
  -m rm_rl.train.train_dt \
  --config configs/sentry_dt.yaml
