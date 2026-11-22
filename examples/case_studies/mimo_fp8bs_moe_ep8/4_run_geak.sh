#!/usr/bin/env bash
# Launch GEAK kernel_workflow on the task. Prereqs: 0_install_geak.sh, `source 1_env.sh`, 3_prepare_task.sh.
# Env knobs: GPU_IDS (default 0,1,2,3)  BUDGET (8)  MODE (optimize | bakeoff)  HEADLESS (0 | 1)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEAK_HOME="${GEAK_HOME:-/sgl-workspace/GEAK}"
AITER_ROOT="${AITER_ROOT:-/sgl-workspace/aiter}"
TASK_DIR="$HERE/mimo_fp8bs_moe_ep8"
: "${ANTHROPIC_CUSTOM_HEADERS:?run 'source $HERE/1_env.sh' first}"
[[ -f "$TASK_DIR/aiter_fused_moe.py" ]] || { echo "run $HERE/3_prepare_task.sh first" >&2; exit 1; }
[[ -f "$GEAK_HOME/kernel_workflow/kernel_workflow.js" ]] || { echo "GEAK not found at $GEAK_HOME" >&2; exit 1; }

PROMPT="$(GEAK_HOME="$GEAK_HOME" AITER_ROOT="$AITER_ROOT" TASK_DIR="$TASK_DIR" python3 - <<'EOF'
import json
import os

wf = os.environ["GEAK_HOME"] + "/kernel_workflow"
mode = os.environ.get("MODE", "optimize")
task = (
    "Rank-local routed-expert MoE of MiMo-V2.6-Flash-MOPD under EP8 (SGLang MoriEPMoE + MORI fp8 dispatch): "
    "hidden 4096, moe_intermediate 2048, 256 experts (32 local), top-8, SiLU gate/up, W8A8 fp8 block-scale "
    "(128x128 weights, 1x128 activations), bf16 out; cases tokens_per_rank 2048/4096 in an 8*4096-row "
    "receive buffer with num_local_tokens valid rows. Baseline: aiter fused_moe (verbatim copy "
    "aiter_fused_moe.py, called from moe_ep.py), which on gfx942 runs the prebuilt asm "
    "fmoe_fp8_blockscale_g1u1 (.co, no source). Editable: moe_ep.py, aiter_fused_moe.py, new files in the "
    "task dir. Levers: per-shape kernel/config choice (1-stage asm vs 2-stage asm/CK/FlyDSL, block_m, ksplit; "
    f"tuner {os.environ['AITER_ROOT']}/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py, keep tuned CSVs in "
    "the task dir), valid-row handling (num_local_tokens, moe_buf init, sorting), weight layout in "
    "prepare_weights (untimed), or a new kernel. Never edit scripts/task_runner.py or the installed "
    "aiter/sglang sources. write a new flydsl kernel can achieve compute roofline."
)
args = {
    "kernel_path": os.environ["TASK_DIR"],
    "workflow_dir": wf,
    "mode": mode,
    "budget": int(os.environ.get("BUDGET", "8")),
    "gpu_ids": os.environ.get("GPU_IDS", "0,1,2,3"),
    "gpus_per_job": 1,
    "use_expert_skills": "true",
    "apply_to_original": "false",
    "task": task,
    "target_language": "flydsl"
}
if mode == "bakeoff":
    args["backends"] = ["triton", "flydsl"]
print(
    f'Run the Workflow tool with scriptPath "{wf}/kernel_workflow.js" and args set to the JSON object below '
    "(pass args as an object, not a JSON string), then report the final result.\n"
    + json.dumps(args, indent=2)
)
EOF
)"

# Root needs IS_SANDBOX=1 for --dangerously-skip-permissions; the freeze re-restores the harness before every measurement.
export IS_SANDBOX=1 GEAK_FREEZE_HARNESS=1
echo "$PROMPT"
cd "$GEAK_HOME"
if [[ "${HEADLESS:-0}" == 1 ]]; then
  log="$HERE/geak_$(date +%Y%m%d_%H%M%S).log"
  nohup claude -p "$PROMPT" --dangerously-skip-permissions --verbose --output-format stream-json > "$log" 2>&1 &
  echo "pid $!  log $log  results under $GEAK_HOME/exp/"
else
  exec claude --dangerously-skip-permissions "$PROMPT"
fi
