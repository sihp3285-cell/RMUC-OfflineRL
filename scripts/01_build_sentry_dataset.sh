#!/usr/bin/env bash
# Build the sentry offline-RL dataset from the referee SQLite file.
# NOTE: data/sentry/ is already shipped pre-built, so you normally DON'T need
# this. Only run it to rebuild, or to build a different --agent. First extract
# the raw DB:   7z x dataset/rmuc_2026_region_dataset.7z -o dataset/
set -e
DB="${DB:-dataset/rmuc_2026_region_dataset.sqlite}"
OUT="${OUT:-data/sentry}"

python -m rm_rl.data.build_dataset \
  --db "$DB" \
  --out "$OUT" \
  --agent 哨兵 \
  --config configs/sentry_iql.yaml \
  --val-frac 0.1 \
  --min-len 30

echo "Dataset written to $OUT"
