# Baseline Profiling Summary — mimo_fp8bs_moe_ep8 (ROUND 0, PHASE=baseline)

## Device
AMD Instinct **MI300X / gfx942 / CDNA3**, 304 CU (confirmed via rocminfo + agent_info.csv:
Cu_Count=304, Wave_Front_Size=64, Max_Waves_Per_Simd=8, Lds 64KB, Num_Xcc=8), fp8 = FNUZ.

## Profiler used + degrade note
`rocprof-compute` (first in the CDNA priority list) exited 1 — its analyze/TUI python deps are missing
in this image (`dash-svg`, `dash>=3.0.0`, `kaleido==0.2.1`, `textual`, `sqlalchemy>=2.0.42`, ...; see
`profile_output/rocprof-compute_profile_raw.log`). The wrapper auto-degraded to **rocprofv3
`--kernel-trace --stats`**, exactly as the COMMANDMENT PROFILE note predicts. Consequence: I have
per-kernel durations, call counts, VGPR/Accum-VGPR/LDS/grid from the trace, and the dispatch shape, but
**no Speed-of-Light counters** (VALU/VMEM/MFMA/L1/L2/HBM). Classification below is from kernel
dominance + the dominant kernel's register/LDS/grid footprint + the prior-run MFMA/occupancy evidence in
analysis.json/roadmap.md. No `!!! PROFILER FAILED` self-heal was needed beyond the expected degrade.

## What is actually in the workspace
The timed path is the **original aiter asm** fused-MoE (`fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256`),
NOT the committed 1.5397x FlyDSL best — `flydsl_moe_fp8bs_s1v2.py` is absent and is "(re)created by
Engineer". Profiled forward latency 1.85/3.29 ms ≈ the frozen aiter baseline (1.67/3.21 ms), well off the
1.12/2.04 ms prior best. So this baseline measures the slow fallback path that the engineer must replace.

## Kernel breakdown (rocprofv3 kernel_stats.csv, 886 dispatches total)
| kernel | % device | calls | avg | note |
|---|---|---|---|---|
| aiter fmoe fp8 blockscale GEMM (SiLU g1u1) | **59.29%** | 12 | 2.455 ms | dominant; min 1.54ms (tpr2048) / max 3.30ms (tpr4096) |
| elementwise Div (act fp8 blockscale quant) | 6.56% | 72 | 45 us | eager activation quant |
| Abs (act amax) | 5.43% | 68 | 40 us | eager activation quant |
| Mul (act scale) | 4.67% | 64 | 36 us | eager activation quant |
| reduce MaxNan (act amax) | 4.33% | 68 | 32 us | eager activation quant |
| float8_copy cast | 3.39% | 68 | 25 us | eager activation quant |
| MoeSorting P0/P1/P23 + ClearWorkspace + partial_transpose | ~3% | 12 each | small | per-forward routing |

Eager activation fp8 block-scale quant (abs->reduce-max->div->clamp->float8_copy->mul) sums to **~20-26%**
of profiled device time across many small dispatches; it accounts for the ~0.3ms (~17%) gap between the
fmoe kernel (1.54ms) and the measured tpr2048 forward (1.85ms). (normal/uniform/random kernels are
input/setup generation, not the timed forward.)

## Dominant kernel — fill & occupancy (from kernel_trace.csv)
- Workgroup 256 threads, Grid 77824 threads => **304 workgroups = exactly 304 CU => 1 WG/CU. GPU IS
  FILLED** (CTAs == CU count); split-K / finer grid is NOT the lever.
- VGPR 128 + **Accum_VGPR 384 = 512 combined**, SGPR 112, **LDS 64KB/WG** (the full per-CU LDS). Both the
  512 combined VGPR and the 64KB LDS pin the kernel to **1 wave/SIMD (12.5% occupancy)** — there are no
  spare resident waves to hide MFMA-issue / data-dependency latency. This is the limiter.

## Classification: COMPUTE-BOUND (issue/occupancy-limited MFMA)
- The op is a dense fp8 MFMA grouped GEMM; the single GEMM kernel is 59% of device time and ~100% of the
  tpr4096 per-case latency. The GPU is filled (304/304 CU) but each CU runs only 1 wave/SIMD, so the MFMA
  pipe cannot hide its own issue/data-dependency latency — classic compute/issue-bound, matching the
  COMMANDMENT hint and the prior run's measured MFMA util 33% and 1 wave/SIMD at baseline.
- NOT memory-bound: analysis.json records HBM <20% under the 750W power cap (sclk throttled ~1.2GHz), and
  there is no counter evidence of high VMEM/HBM. Do not mislabel the small per-FLOP AI as memory-bound.
- Latency sub-case: **C2 (too few resident waves)** — occupancy ceiling (1 wave/SIMD from 512 VGPR +
  64KB LDS) dominates, so the fix is to raise waves/SIMD (fewer VGPR/LDS per wave), the opposite of C1's
  "shorten the dependency chain". The FlyDSL replacement already demonstrated this (1->2 waves/SIMD).

## Top opportunities (ranked)
1. **Replace aiter asm with the FlyDSL native-fp8 two-stage MoE** (recreate `flydsl_moe_fp8bs_s1v2.py`) —
   raise occupancy 1->2 waves/SIMD + MFMA util 33%->60%; this was the committed 1.5397x path and the
   workspace currently runs the slow fallback.
2. **Stage1 128-row transposed tile** (weights=src-A, tokens=src-B, one token/lane => per-lane scalar
   activation block-scale): shares/halves the streamed B (weights) operand per FLOP — the issue-bound
   limiter. Confirm engagement by live grid shape, not wall-clock (prior "winner" silently ran fallback).
3. **VGPR trim to <=170** on the winning stage1 to unlock a 3rd wave/SIMD for latency hiding.
4. **Fuse the eager activation fp8 block-scale quant** (~20-26% of device time, ~0.3ms/~17% of tpr2048)
   into the kernel / fewer ops.

## Bar
Beat prior best geomean **1.5397x** (not the 1.0x frozen aiter baseline). Baseline geomean_ms = 2.3198.
