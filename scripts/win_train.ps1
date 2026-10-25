# Windows single-GPU launcher (e.g. RTX 4070). No torchrun / DDP needed.
# Usage (from the rm_rl/ folder):
#   powershell -ExecutionPolicy Bypass -File scripts\win_train.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\win_train.ps1 configs\sentry_bc.yaml
#   powershell -ExecutionPolicy Bypass -File scripts\win_train.ps1 configs\sentry_dt.yaml
param([string]$Config = "configs\sentry_iql.yaml")

$env:PYTHONUTF8 = "1"          # UTF-8 mode: safe Chinese printing + file I/O

if ($Config -like "*dt*") {
    python -m rm_rl.train.train_dt --config $Config
} else {
    python -m rm_rl.train.train_offline --config $Config
}
