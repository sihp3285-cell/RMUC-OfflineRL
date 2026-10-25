# Windows: rebuild a dataset (only needed to change reward weights or agent).
# The sentry data is already shipped in data\sentry, so you usually skip this.
# Extract the raw DB first: 7z x dataset\rmuc_2026_region_dataset.7z -o dataset\
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\win_build.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\win_build.ps1 infantry3
# Agent alias: sentry / hero / engineer / infantry3 / infantry4 / aerial
param([string]$Agent = "sentry")

$env:PYTHONUTF8 = "1"
python -m rm_rl.data.build_dataset `
  --db dataset\rmuc_2026_region_dataset.sqlite `
  --out data\sentry --agent $Agent --config configs\sentry_iql.yaml
