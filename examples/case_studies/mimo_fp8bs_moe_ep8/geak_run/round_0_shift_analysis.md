# Round 0 Reprofile — mimo_fp8bs_moe_ep8 (PHASE=reprofile, warm-start adopted state)

## Device
AMD Instinct **MI300X / gfx942 / CDNA3**, 304 CU, wave64, 64KB LDS/CU, fp8 = FNUZ.

## Profiler used + degrade note
`rocprof-compute` (first in the CDNA priority list) failed the same way as baseline — its analyze/TUI
python deps are missing in this image — so the wrapper auto-degraded to **rocprofv3
`--kernel-trace --stats`** exactly as the COMMANDMENT PROFILE note predicts. I have per-kernel
durations, call counts, VGPR/Accum-VGPR/LDS/grid from the trace, and dispatch shape, but **no
Speed-of-Light counters** (VALU/VMEM/MFMA/L1/L2/HBM). Classified from kernel dominance + the dominant
kernels' register/LDS/grid footprint. No `!!! PROFILER FAILED` self-heal needed beyond the expected
degrade.

## What is actually in the workspace now (changed vs baseline)
Correctness **PASS** (tpr2048 rel_l2 2.3e-3, tpr4096 rel_l2 2.3e-3; both well under tol). The timed path
is now the **FlyDSL native-fp8 two-stage MoE** (`moe_ep._forward_fbs` -> `moe_sorting` +
`flydsl_moe_fp8bs_s1v2.compile_stage1_t` + `flydsl_moe_fp8bs_s2.stage2_fp8bs`). The aiter asm kernel
`fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256` is **gone**. Measured forward **1.1206 / 2.0562 ms**
(tpr2048/tpr4096) = **1.528x geomean** vs the frozen aiter baseline (2.3198 geomean_ms), i.e. the
committed ~1.54x warm-start best is reproduced here. This is the state to BEAT.

## Kernel breakdown (rocprofv3 kernel_stats.csv)
| kernel | % device | calls | avg | note |
|---|---|---|---|---|
| moe_fp8bs_gemm1_0 (stage1 gate/up MFMA + SiLU + fp8 quant) | **31.60%** | 12 | 1.055 ms | the leader, ~2x stage2 |
| moe_s2_fp8bs_0 (stage2 down MFMA + weighted scatter) | **17.82%** | 12 | 0.595 ms | |
| DivFunctor (act fp8 quant) | 8.21% | 72 | 46 us | INPUT-GEN / reference, NOT the forward |
| AbsFunctor (act amax) | 6.86% | 68 | 40 us | INPUT-GEN / reference |
| MulFunctor (act scale) | 6.02% | 64 | 38 us | INPUT-GEN / reference |
| reduce MaxNan (act amax) | 5.13% | 68 | 30 us | INPUT-GEN / reference |
| float8_copy cast | 4.19% | 68 | 25 us | INPUT-GEN / reference |
| MoeSorting P0/P1/P23 + ClearWorkspace | ~3.8% | 12 each | small | per-forward routing |

**Important correction vs the baseline summary:** the activation fp8 block-scale quant chain
(Abs->ReduceMax->Div->clamp->float8_copy->Mul) is **input generation / the reference path**, not part of
the FBS timed forward. `_forward_fbs` launches only moe_sorting + stage1 + stage2 — stage1 already fuses
SiLU and the intermediate fp8 block-scale quant (it emits `a2` uint8 + `a2s` scales directly). So the
timed forward's device time is essentially the two FlyDSL GEMMs + routing (dispatch_count ~6 per call).

## Dominant kernels — fill & occupancy (from kernel_trace.csv)
- **stage1 `moe_fp8bs_gemm1_0`**: WG=256, grid=17825792 => **69632 CTAs >> 304 CU (GPU massively
  oversubscribed)**. VGPR 112 + Accum 128 = **240 combined**, SGPR 112, **LDS 32KB/WG**, scratch 0.
  waves/SIMD = min(8, 512/240) = **2**; LDS 32KB/WG also caps at 2 WG/CU (64KB/CU). => **2 waves/SIMD**.
- **stage2 `moe_s2_fp8bs_0`**: WG=256, grid=2228224 => **8704 CTAs >> 304 CU**. VGPR 128 + Accum 128 =
  **256 combined**, SGPR 112, **LDS 28KB/WG**, scratch 0. waves/SIMD = min(8, 512/256) = **2**; LDS
  28KB caps at 2 WG/CU. => **2 waves/SIMD**.

## Classification: COMPUTE-BOUND (MFMA issue/occupancy-limited), occupancy 1 -> 2 waves/SIMD
The two dense fp8 MFMA grouped GEMMs are 49.42% of device time and ~all of the timed forward. Both now
run at **2 waves/SIMD** with no scratch spill — the register/LDS footprint halved vs baseline (240/256
combined VGPR vs 512; 32/28KB LDS vs 64KB), which is exactly the 1->2 wave occupancy doubling that
delivered the ~1.54x. GPU is heavily oversubscribed (tens of thousands of CTAs), so fill / split-K is
NOT the lever. Latency sub-case remains **C2 (too few resident waves)** but materially relieved (2 vs 1).

## BEFORE -> AFTER
```
BEFORE: compute-bound - aiter asm fmoe 59.29%, 1 wave/SIMD, 512 combined VGPR + 64KB LDS, 1.85/3.29 ms
AFTER:  compute-bound - FlyDSL gemm1 31.60% + s2 17.82%, 2 waves/SIMD, 240/256 VGPR + 32/28KB LDS, 1.12/2.06 ms (1.528x)
SHIFT:  aiter asm path replaced by the FlyDSL two-stage kernels -> occupancy doubled 1->2 waves/SIMD
        (combined VGPR 512->240/256, LDS 64KB->32/28KB), no spill; still MFMA/compute-bound, no longer
        starved to a single resident wave.
NEXT:   Push stage1 gemm1 (the 31.60% leader) from 2->3 waves/SIMD. The limiter is now BOTH LDS and VGPR:
        32KB LDS/WG pins 2 WG/CU, so a VGPR trim alone does nothing until LDS drops. Need LDS <=21KB/WG
        AND combined VGPR <=170 together to fit a 3rd wave (33% occupancy) for deeper MFMA latency hiding.
        Stage2 s2 sits exactly at 256 combined VGPR (the 2-wave edge) + 28KB LDS -> same lever.
        Secondary: fuse stage1->stage2 to drop the a2/a2s fp8-intermediate round-trip through HBM.
```
