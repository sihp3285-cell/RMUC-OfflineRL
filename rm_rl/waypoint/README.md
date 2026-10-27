# waypoint distillation (MVP)

Teacher: `gpt61_2027_adaptive_2772` sentry `goal`.
Student: fixed 10-robot/team/control features → candidate anchors (classifier) or 2D regression.
Output: one field coordinate `(x, y)`; low-level drive/aim/fire stays with the teacher/sentry stack.

## Data

```bash
PY=/home/delphine/miniconda3/envs/dl_env/bin/python
$PY -m rm_rl.waypoint.collect --matches 16 --seed-start 0 --workers 8 \
  --opponent hold --out data/waypoint_mvp_hold
$PY -m rm_rl.waypoint.collect --matches 8 --seed-start 100 --workers 8 \
  --opponent adaptive_2731 --out data/waypoint_mvp_adaptive
```

Each `match-*.npz` contains:

- `features` `[N, 307]` — fixed 2027 contract;
- `labels` `[N, 2]` — teacher goal normalized `(x/28, y/15)`;
- `goals` `[N, 2]` — metres;
- `ego` `[N, 2]` — sentry position;
- `time` `[N]`, `meta` `[N, 4]` (seed, opponent, side, tick).

## Train

```bash
$PY -m rm_rl.waypoint.train_cls --data data/waypoint_mvp_hold data/waypoint_mvp_adaptive \
  --out rm_runs/waypoint_cls --epochs 50 --batch 1024
```

`best.pt` stores the anchor set, normalization, and classifier weights.
`metrics.json` stores top-1, coordinate error, and constant baselines.

## Evaluate

```bash
$PY -m rm_rl.waypoint.eval_closed_loop --mode teacher --ckpt rm_runs/waypoint_cls24/best.pt \
  --matches 2 --duration 240 --workers 2 --seed-start 1000 --out /tmp/teacher.json
$PY -m rm_rl.waypoint.eval_closed_loop --mode student --ckpt rm_runs/waypoint_cls24/best.pt \
  --matches 2 --duration 240 --workers 2 --seed-start 1000 --out /tmp/student.json
```

## Export

```bash
$PY -m rm_rl.waypoint.export_onnx --ckpt rm_runs/waypoint_cls24/best.pt \
  --out rm_runs/waypoint_cls24/waypoint.onnx
```

`waypoint.json` carries `mean/std`, anchors, base dimension and output semantics.
`infer_onnx.WaypointRuntime` is the reference Python runtime; the radar-station C++
node must reproduce `build_features → rule_features → anchor_features → normalize`.

## First run (2026-10-25)

- Data: 24 matches (16 vs `hold`, 8 vs `adaptive_2731`), 97,234 samples; 4 teacher anchors.
- Model: candidate classifier, `335 -> 256 -> 256 -> 4`; teacher goal top-1 **91.97%** on the held-out 5 matches.
- Coordinate error: mean **0.516 m**, median **0.002 m**, p90 **0.097 m**, p99 **10.35 m** (wrong-anchor tail).
- ONNX: **603 KB**, CPU p99 **3.1 ms**, PyTorch/ONNX max diff **8e-7**.
- Closed-loop 240 s × 4 seeds: student 2 wins, teacher 1 win; mixed, small sample, not statistically conclusive.
- Full report: `docs/WAYPOINT_MVP_RESULTS.md`.

## Current caveats

- Teacher 2772 uses onboard vision; the student uses the fixed radio/engine contract.
- The first MVP is BC only; no DAgger/AWR/PPO yet.
- Hard seeds still confuse adjacent anchors; history/latch features and more seeds are the next step.
- Radar station integration needs a ROS 2 node + serial bridge + fallback; ONNX alone is the integration.
