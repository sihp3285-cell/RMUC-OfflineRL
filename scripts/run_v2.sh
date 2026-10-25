#!/usr/bin/env bash
# obs-v2 pipeline: 161-D base + 22 appended slots (183-D total).
#
# This script NEVER touches the v1 artifacts:
#   data/infantry_tactical        (old 161-D shards)
#   rm_runs/infantry_*_tactical   (old checkpoints)
# It writes only:
#   data/infantry_tactical_v2
#   rm_runs/infantry_iql_tactical_v2 (and _bc_/_dt_ if you run those configs)
#
# Usage:
#   NPROC=4 bash scripts/run_v2.sh              # build + check + train IQL
#   SKIP_TRAIN=1 bash scripts/run_v2.sh         # build + check only
#   LIMIT_GAMES=20 bash scripts/run_v2.sh       # smoke
#   PY="$(which python)" bash scripts/run_v2.sh  # after activating the env
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-python}"
DB="${DB:-dataset/rmuc_2026_region_dataset.sqlite}"
DATA="${DATA:-data/infantry_tactical_v2}"
VIS_MAP="${VIS_MAP:-data/vis_map.npz}"
TEAM_PRIOR="${TEAM_PRIOR:-data/team_prior.json}"
CONFIG="${CONFIG:-configs/infantry_iql_tactical_v2.yaml}"
RUN="${RUN:-rm_runs/infantry_iql_tactical_v2}"

# Fail before any build/train work if PY is not a usable training interpreter.
# `/path/to/train/python` in the docs is a placeholder, not a real path.
if [[ ! -x "$PY" ]] && ! command -v "$PY" >/dev/null 2>&1; then
  echo "PY is not executable: $PY" >&2
  echo "Set PY to the training interpreter, e.g.  PY=\"\$(which python)\"  (after activating the env)" >&2
  exit 1
fi
if ! "$PY" -c 'import torch' >/dev/null 2>&1; then
  echo "PY cannot import torch: $PY" >&2
  echo "Activate the training env, or pass PY=/path/to/that/python" >&2
  exit 1
fi

NPROC="${NPROC:-$("$PY" -c 'import torch;print(torch.cuda.device_count() or 1)')}"
LIMIT_GAMES="${LIMIT_GAMES:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

[[ -f "$DB" ]] || { echo "SQLite not found: $DB" >&2; exit 1; }
[[ -f "$VIS_MAP" ]] || { echo "vis_map not found: $VIS_MAP" >&2; exit 1; }
[[ -f "$TEAM_PRIOR" ]] || { echo "team_prior not found: $TEAM_PRIOR" >&2; exit 1; }

if [[ -f "$DATA/meta.json" ]] && "$PY" - "$DATA" <<'PY'
import json, sys
m = json.load(open(sys.argv[1] + "/meta.json", encoding="utf-8"))
raise SystemExit(0 if m.get("obs_version") == 2 and m.get("obs_dim") == 183 else 1)
PY
then
  log "v2 dataset exists and matches (obs_version=2, obs_dim=183), skipping build"
else
  log "building v2 dataset -> $DATA"
  "$PY" -m rm_rl.data.build_dataset \
    --db "$DB" --out "$DATA" --agent infantry \
    --action-mode tactical --goal-horizon 5 \
    --vis-map "$VIS_MAP" --team-prior "$TEAM_PRIOR" \
    --obs-version 2 --limit-games "$LIMIT_GAMES"
fi

log "checking v2 dataset"
"$PY" -m rm_rl.data.check_dataset --data "$DATA"

if [[ "$SKIP_TRAIN" == "1" ]]; then
  log "SKIP_TRAIN=1, stopping after data check"
  exit 0
fi

if [[ -e "$RUN/best.pt" || -e "$RUN/final.pt" ]]; then
  echo "refusing to train: $RUN already has a checkpoint." >&2
  echo "choose a new RUN=..., or move the existing run away first." >&2
  exit 1
fi

log "training $(basename "$CONFIG") -> $RUN  (NPROC=$NPROC)"
if [[ "$NPROC" -gt 1 ]]; then
  "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    -m rm_rl.train.train_offline --config "$CONFIG"
else
  "$PY" -m rm_rl.train.train_offline --config "$CONFIG"
fi

log "done. best checkpoint: $RUN/best.pt"
echo "BC baseline:  $PY -m rm_rl.train.train_offline --config configs/infantry_bc_tactical_v2.yaml"
echo "DT control:   $PY -m rm_rl.train.train_offline --config configs/infantry_dt_tactical_v2.yaml"
