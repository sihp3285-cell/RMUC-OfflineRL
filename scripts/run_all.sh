#!/usr/bin/env bash
# One-shot pipeline for the tactical rebuild. Run from the repo root:
#
#     bash scripts/run_all.sh                       # all GPUs, full dataset
#     NPROC=2 bash scripts/run_all.sh               # pin GPU count
#     LIMIT_GAMES=40 bash scripts/run_all.sh        # quick smoke test (~minutes)
#     SKIP_DT=1 bash scripts/run_all.sh             # skip the Decision Transformer
#
# Stages: engagement map -> team priors -> dataset -> IQL/BC/DT -> OPE -> export.
# Every stage is skipped if its output already exists, so a re-run resumes.
set -euo pipefail

cd "$(dirname "$0")/.."

DB="${DB:-dataset/rmuc_2026_region_dataset.sqlite}"
# Train on the HUMAN-DRIVEN infantry, not the sentry.  Offline RL cannot exceed
# its demonstrator, and the logged sentry is each team's own autonomous script
# (2.9x less map coverage, half the positional entropy of infantry).  Infantry
# are structurally identical to the sentry (400 HP / 260 heat / 17 mm cap) and
# human-operated, so they are the right teacher for a sentry policy.
# Override with AGENT=sentry to reproduce the old sentry-on-sentry run.
AGENT="${AGENT:-infantry}"
DATA="${DATA:-data/infantry_tactical}"
NPROC="${NPROC:-$(python -c 'import torch;print(torch.cuda.device_count() or 1)')}"
LIMIT_GAMES="${LIMIT_GAMES:-0}"
GOAL_HORIZON="${GOAL_HORIZON:-5}"
SKIP_DT="${SKIP_DT:-0}"

VIS_MAP="data/vis_map.npz"
TEAM_PRIOR="data/team_prior.json"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# A dataset directory left over from an earlier feature version silently has the
# wrong obs width, and you only find out when a matmul fails deep inside OPE.
# "Exists" is therefore not a good enough reason to reuse one — check that its
# recorded dims still match what the current code builds.
dataset_ok() {
  [[ -f "$1/meta.json" && -f "$1/train.npz" ]] || return 1
  python - "$1" <<'PY' 2>/dev/null
import json, sys
from rm_rl.data.features import obs_dim
from rm_rl.algos.action_spec import get_spec
d = sys.argv[1]
m = json.load(open(d + "/meta.json", encoding="utf-8"))
agent = (m.get("agent_types") or [m.get("agent_type", "哨兵")])[0].split(",")[0]
want_o = obs_dim(agent)
want_a = get_spec(m.get("action_mode", "velocity"), m.get("action_dim")).dim
if m.get("obs_dim") != want_o:
    print(f"  stale: obs_dim {m.get('obs_dim')} != {want_o}", file=sys.stderr)
    sys.exit(1)
if m.get("action_dim") != want_a:
    print(f"  stale: action_dim {m.get('action_dim')} != {want_a}", file=sys.stderr)
    sys.exit(1)
PY
}

# ---------------------------------------------------------------- 0. sanity
if [[ ! -f "$DB" ]]; then
  echo "SQLite not found at $DB" >&2
  ARCHIVE="dataset/rmuc_2026_region_dataset.7z"
  if [[ -f "$ARCHIVE" ]]; then
    log "extracting $ARCHIVE"
    command -v 7z >/dev/null || { echo "install p7zip-full first: sudo apt install -y p7zip-full" >&2; exit 1; }
    7z x -y -odataset "$ARCHIVE"
  else
    exit 1
  fi
fi
mkdir -p data rm_runs
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'gpus',torch.cuda.device_count())"

# --------------------------------------------- 1. empirical engagement map
if [[ -f "$VIS_MAP" ]]; then
  log "vis_map exists, skipping ($VIS_MAP)"
else
  log "building empirical engagement / line-of-sight map"
  python -m rm_rl.data.vis_map --db "$DB" --out "$VIS_MAP" --limit-games "$LIMIT_GAMES"
fi

# ------------------------------------------------ 2. leak-safe team priors
if [[ -f "$TEAM_PRIOR" ]]; then
  log "team_prior exists, skipping ($TEAM_PRIOR)"
else
  log "building leak-safe historical team priors"
  python -m rm_rl.data.team_prior --db "$DB" --out "$TEAM_PRIOR"
fi

# ------------------------------------------------------------- 3. dataset
if dataset_ok "$DATA"; then
  log "dataset exists and matches the current feature version, skipping ($DATA)"
else
  [[ -e "$DATA" ]] && { log "REBUILDING $DATA (missing or stale)"; rm -rf "$DATA"; }
  log "building tactical dataset for agent=$AGENT"
  python -m rm_rl.data.build_dataset \
    --db "$DB" --out "$DATA" --agent "$AGENT" \
    --config configs/infantry_iql_tactical.yaml \
    --action-mode tactical --goal-horizon "$GOAL_HORIZON" \
    --vis-map "$VIS_MAP" --team-prior "$TEAM_PRIOR" \
    --limit-games "$LIMIT_GAMES"
fi

# ---------------------------------------------------------- 4. sanity check
log "dataset self-check"
python -m rm_rl.data.check_dataset --data "$DATA"

# -------------------------------------------------------------- 5. training
launch() {  # launch <module> <config>
  if [[ "$NPROC" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="$NPROC" -m "$1" --config "$2"
  else
    python -m "$1" --config "$2"
  fi
}

log "training IQL (tactical)   [$NPROC proc]"
launch rm_rl.train.train_offline configs/infantry_iql_tactical.yaml

log "training BC  (tactical)   [$NPROC proc]"
launch rm_rl.train.train_offline configs/infantry_bc_tactical.yaml

if [[ "$SKIP_DT" != "1" ]]; then
  log "training Decision Transformer (tactical)   [$NPROC proc]"
  launch rm_rl.train.train_dt configs/infantry_dt_tactical.yaml
fi

# ------------------------------------------------------------------ 6. OPE
log "offline policy evaluation (FQE) — vs the human infantry it learned from"
python -m rm_rl.eval.ope_fqe --data "$DATA" --run rm_runs/infantry_iql_tactical | tee rm_runs/ope_iql_tactical.txt
python -m rm_rl.eval.ope_fqe --data "$DATA" --run rm_runs/infantry_bc_tactical  | tee rm_runs/ope_bc_tactical.txt
if [[ "$SKIP_DT" != "1" ]]; then
  python -m rm_rl.eval.ope_dt --data "$DATA" --run rm_runs/infantry_dt_tactical | tee rm_runs/ope_dt_tactical.txt
fi

# ------------------------------------------------- 6b. CROSS-ROLE evaluation
# The question this whole project actually asks: does a policy taught by human
# infantry play the sentry role better than the teams' own scripted sentries?
# Answer it by scoring the infantry-trained policy against SENTRY behaviour.
# Valid because the observation layout is role-agnostic (fixed 6-slot team
# block, no role one-hot) and the sentry is structurally a maxed-out infantry.
SENTRY_DATA="data/sentry_tactical"
if dataset_ok "$SENTRY_DATA"; then
  log "sentry eval dataset up to date, skipping ($SENTRY_DATA)"
else
  [[ -e "$SENTRY_DATA" ]] && { log "REBUILDING $SENTRY_DATA (missing or stale)"; rm -rf "$SENTRY_DATA"; }
  log "building sentry dataset (for cross-role evaluation only, not training)"
  python -m rm_rl.data.build_dataset \
    --db "$DB" --out "$SENTRY_DATA" --agent sentry \
    --config configs/infantry_iql_tactical.yaml \
    --action-mode tactical --goal-horizon "$GOAL_HORIZON" \
    --vis-map "$VIS_MAP" --team-prior "$TEAM_PRIOR" \
    --limit-games "$LIMIT_GAMES"
fi
log "CROSS-ROLE: infantry-trained policy scored against scripted-sentry behaviour"
python -m rm_rl.eval.ope_fqe --data "$SENTRY_DATA" --run rm_runs/infantry_iql_tactical \
  | tee rm_runs/ope_iql_infantry_on_sentry.txt
echo "   ^ Delta > 0 here is the headline result: learning from humans beats the script."

# --------------------------------------------------------------- 7. export
log "exporting deployable bundles"
python scripts/export_policy.py rm_runs/infantry_iql_tactical
python scripts/export_policy.py rm_runs/infantry_bc_tactical

log "ALL DONE"
echo "runs:    rm_runs/"
echo "OPE:     rm_runs/ope_*_tactical.txt"
echo "bundles: rm_runs/*/export/"
