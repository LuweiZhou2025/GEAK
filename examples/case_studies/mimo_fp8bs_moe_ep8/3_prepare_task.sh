#!/usr/bin/env bash
# Copy aiter/fused_moe.py into the task as the editable baseline, then run the harness once.
# Usage: GPU=0 bash 3_prepare_task.sh      (FORCE=1 re-copies aiter_fused_moe.py)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$HERE/mimo_fp8bs_moe_ep8"
export HIP_VISIBLE_DEVICES="${GPU:-0}"

if [[ ! -f "$TASK_DIR/aiter_fused_moe.py" || "${FORCE:-0}" == 1 ]]; then
  src="$(python3 -c 'import os, aiter; print(os.path.join(os.path.dirname(aiter.__file__), "fused_moe.py"))' 2>/dev/null | tail -1)"
  cp "$src" "$TASK_DIR/aiter_fused_moe.py"
  printf 'source: %s\naiter commit: %s\n' "$src" \
    "$(git -C "$(dirname "$src")" rev-parse HEAD 2>/dev/null || echo unknown)" > "$TASK_DIR/AITER_SOURCE.txt"
fi
cat "$TASK_DIR/AITER_SOURCE.txt"

cd "$TASK_DIR"
python3 scripts/task_runner.py compile
python3 scripts/task_runner.py correctness
for run in 1 2 3; do
  echo "--- performance run $run (GPU $HIP_VISIBLE_DEVICES)"
  python3 scripts/task_runner.py performance | grep '^Perf:'
done
