#!/usr/bin/env bash
# Bundle just the results worth pulling off the server.
#
#     bash scripts/collect_results.sh              # logs + best models (~tens of MB)
#     LOGS_ONLY=1 bash scripts/collect_results.sh  # logs + OPE only (~1 MB)
#
# Deliberately excluded: ckpt_*.pt (intermediate, reproducible), final.pt (the
# overfit end-of-run model — best.pt is the one to keep), tb/ (redundant with
# the CSV logs), and data/ (regenerable from the .sqlite, and huge).
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-rm_results_$(date +%Y%m%d_%H%M).tar.gz}"
LOGS_ONLY="${LOGS_ONLY:-0}"

files=()
# --- always: the small, high-value stuff -----------------------------------
while IFS= read -r f; do files+=("$f"); done < <(
  find rm_runs -maxdepth 2 \( \
       -name '*.csv' -o -name 'meta.json' -o -name '*.txt' \
    \) 2>/dev/null | sort)

# --- unless LOGS_ONLY: the deployable model --------------------------------
if [[ "$LOGS_ONLY" != "1" ]]; then
  while IFS= read -r f; do files+=("$f"); done < <(
    find rm_runs -maxdepth 2 \( \
         -name 'best.pt' -o -name 'norm_stats.npz' \
      \) 2>/dev/null | sort)
fi

if [[ ${#files[@]} -eq 0 ]]; then
  echo "nothing found under rm_runs/ — did the run finish?" >&2
  exit 1
fi

tar -czf "$OUT" "${files[@]}"
echo
echo "packed ${#files[@]} files -> $OUT  ($(du -h "$OUT" | cut -f1))"
echo
printf '%s\n' "${files[@]}" | sed 's/^/  /'
echo
echo "Pull it with, from your local machine:"
echo "  scp <user>@<host>:$(pwd)/$OUT ."
