# COMMANDMENT — mimo_fp8bs_moe_ep8 (rank-local fp8 block-scale fused MoE, MiMo-V2.6-Flash-MOPD EP8)

This is the IMMUTABLE measurement contract. Every agent (Engineer / Verify / Profile / later waves)
MUST use these exact commands. Do not modify this file, the harness (`scripts/task_runner.py`), or any
file outside the workspace. Benchmark output is the single source of truth.

## OP / REGIME
Rank-0-of-8 EP simulation of SGLang MoriEPMoE: aiter `fused_moe(quant_type=per_128x128)` on the MORI
fp8-dispatched receive buffer. W8A8 fp8 block-scale (128x128 weights, 1x128 activations), SiLU gate/up,
bf16 out. HIDDEN=4096, INTER=2048, 256 experts total / 32 local, topk=8, recv buffer 32768 rows.
Two timed cases: `tpr2048` (2048 tokens/rank, 10841 valid rows) and `tpr4096` (4096 tokens/rank, 21575
valid rows). GPU arch: **gfx942** (fp8 = float8_e4m3fnuz).

WARM-START: this run continues from a prior committed best of **1.5397x geomean** (tpr2048 1.1178 ms /
tpr4096 2.0447 ms). That 1.5397x is the bar to BEAT; the numbers below are the ORIGINAL aiter baseline
(the speedup denominator), NOT the bar.

## MODIFIABLE FILES
`moe_ep.py`, `aiter_fused_moe.py`, `flydsl_moe_fp8bs_s1v2.py` (to be (re)created by Engineer), and any
NEW files created inside the workspace task dir. Everything else is frozen. `prepare_weights` is untimed
(runs once per weight set); only `forward` is timed.

## SETUP
```
cd /sgl-workspace/GEAK/exp/team_mimo_fp8bs_moe_ep8_20261119_100217_515084_14709/mimo_fp8bs_moe_ep8/workspace
```
Each workspace is a fresh artifact-free copy; nothing to clear. No `rm`. If you suspect a stale JIT
cache after editing, move it aside: `mv .torch_ext .torch_ext.stale_$(date +%s)_$$ 2>/dev/null || true`.

## COMPILE (optional sanity; just imports the impl)
```
cd /sgl-workspace/GEAK/exp/team_mimo_fp8bs_moe_ep8_20261119_100217_515084_14709/mimo_fp8bs_moe_ep8/workspace && bash /sgl-workspace/GEAK/kernel_workflow/scripts/gpu_lock.sh 0,1,2,3 python3 scripts/task_runner.py compile
```

## CORRECTNESS (run BEFORE any benchmark)
```
cd /sgl-workspace/GEAK/exp/team_mimo_fp8bs_moe_ep8_20261119_100217_515084_14709/mimo_fp8bs_moe_ep8/workspace && bash /sgl-workspace/GEAK/kernel_workflow/scripts/gpu_lock.sh 0,1,2,3 python3 scripts/task_runner.py correctness
```
Prints one `PASS/FAIL <case>:` line per case plus a final `Correctness: PASS|FAIL`; exit 0 only when
both cases pass. Tolerances (frozen in the oracle): global rel-L2 <= 2e-2, worst per-token rel-L2 <= 5e-2.
A result is only valid when correctness PASSes first.

## BENCHMARK (quick; same harness as full — 10 warmup / 100 iters)
```
cd /sgl-workspace/GEAK/exp/team_mimo_fp8bs_moe_ep8_20261119_100217_515084_14709/mimo_fp8bs_moe_ep8/workspace && bash /sgl-workspace/GEAK/kernel_workflow/scripts/gpu_lock.sh 0,1,2,3 python3 scripts/task_runner.py performance
```

## FULL_BENCHMARK (authoritative — identical command; 10 warmup / 100 iters, two alternating input sets)
```
cd /sgl-workspace/GEAK/exp/team_mimo_fp8bs_moe_ep8_20261119_100217_515084_14709/mimo_fp8bs_moe_ep8/workspace && bash /sgl-workspace/GEAK/kernel_workflow/scripts/gpu_lock.sh 0,1,2,3 python3 scripts/task_runner.py performance
```
Prints two lines: `Perf: <ms> ms (tpr2048)` and `Perf: <ms> ms (tpr4096)`.

## PROFILE
```
bash /sgl-workspace/GEAK/kernel_workflow/scripts/profile_kernel.sh 0,1,2,3 "cd /sgl-workspace/GEAK/exp/team_mimo_fp8bs_moe_ep8_20261119_100217_515084_14709/mimo_fp8bs_moe_ep8/workspace && python3 scripts/task_runner.py profile" <out_dir>
```
Read `<out_dir>/profile_report.txt`. On this host rocprof-compute degrades and the wrapper falls back to
rocprofv3 (kernel-trace CSV stats) automatically — that is EXPECTED, not a failure; use the rocprofv3
`Name,Calls,TotalDurationNs,AverageNs,Percentage,...` table. Baseline profile: the aiter asm kernel
`aiter::fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256` dominates (~59% of device time) — compute/
issue-bound, confirming the ANALYSIS. If a genuine `!!! PROFILER FAILED` block appears, follow
`kernel_workflow/knowledge/profiling_guide.md` (override the named env var / degrade and say so).

## PARSE
Per-case latency: for each `Perf:` line, the float before `ms` is the per-call latency and the token in
parentheses is the case id. Regex: `^Perf:\s*([0-9.]+)\s*ms\s*\(([a-z0-9]+)\)$` → group(1)=latency_ms,
group(2)=case_id ∈ {`tpr2048`,`tpr4096`}. Correctness: `^(PASS|FAIL)\s+(tpr2048|tpr4096):` per case and
the final `Correctness: PASS|FAIL`.

## METRIC (PRIMARY)
No workload-weighted oracle and no WORKLOAD_SPEC → PRIMARY = **unweighted geomean of per-case speedups**:
`speedup = geomean_i( baseline_ms[i] / candidate_ms[i] )` over the two cases, where `baseline_ms` is the
FROZEN ORIGINAL aiter latency recorded below. (Equivalently `geomean(baseline_ms)/geomean(candidate_ms)`.)
This single number is what the round-winner gate and the final result use. The prior best is 1.5397x;
BEAT that, not 1.0x. Secondary diagnostics: the two per-case speedups individually (confirm no case
regresses). Baseline is the real production aiter asm path — NOT a naive strawman.

## BASELINE (FROZEN ORIGINAL aiter path — speedup denominator)
Measured 3x via FULL_BENCHMARK; stable within ~0.5%. Reliable.

| case     | baseline_ms (mean) | run1   | run2   | run3   |
|----------|--------------------|--------|--------|--------|
| tpr2048  | 1.6747             | 1.6790 | 1.6717 | 1.6733 |
| tpr4096  | 3.2135             | 3.2031 | 3.2178 | 3.2196 |

Baseline geomean_ms = sqrt(1.6747 * 3.2135) = **2.3198 ms**. A candidate's geomean speedup =
2.3198 / geomean(candidate tpr2048, tpr4096). See `baseline_timing.json`.

## RULES
- Never modify this COMMANDMENT, `scripts/task_runner.py`, or any file outside the workspace.
- Always run CORRECTNESS (and require PASS) before trusting any BENCHMARK number.
- Always invoke every GPU command via `gpu_lock.sh 0,1,2,3` from inside the workspace dir.
- FULL_BENCHMARK output is the source of truth; report the PRIMARY geomean speedup vs the baseline above.
