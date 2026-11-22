# Tech Lead Final Report — mimo_fp8bs_moe_ep8

## Summary

| item | value |
|---|---|
| Kernel | Rank-local routed-expert fused MoE for MiMo-V2.6-Flash-MOPD (EP8, rank 0 of 8). W8A8 fp8 block-scale: 128x128 weights, 1x128 activations. SiLU gate/up, bf16 out. HIDDEN 4096, INTER 2048, 32 of 256 experts local, topk 8, 32,768-row receive buffer. |
| Hardware | gfx942 (MI300X, 304 CU). 750 W socket power cap. fp8 = e4m3fnuz. |
| Type | e2e. The Python wrapper `moe_ep.py` drives two FlyDSL MFMA GEMMs (`moe_fp8bs_gemm1_0`, `moe_s2_fp8bs_0`) and a 2-kernel HIP sort (`moe_sort_fast.py`, load_inline). Operator `fused_moe_grouped_gemm`, language `flydsl`. |
| Baseline (speedup denominator) | The production aiter path: `aiter::fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256` plus CK MoeSorting. 1.6747 / 3.2135 ms; geomean 2.3198 ms. |
| **Final result** | **1.6273x geomean**, the headline (the run is not workload-aligned). Arithmetic 1.6279x. Weighted 1.6268x (equal weights, which equals the harmonic mean of the per-case speedups). |
| Final code | Canonical commit `31c4cb1`, the R3 integrated winner. Per case: tpr2048 1.0558 ms (1.5862x), tpr4096 1.9248 ms (1.6695x). The R3 reprofile FULL_BENCHMARK x3 corroborates it: 1.6256x (1.0603 / 1.9207 ms). |
| Gain within this run | Warm start from a prior run's FlyDSL best (bar 1.5397x; this run's R0 reprofile measured 1.528x). +6.5% over the R0 reprofile (1.5279 -> 1.6273); +5.7% over the warm-start bar. |
| Rounds / budget | 5 rounds. 12 of 12 directions used (R1 3, R2 3, R3 2, R4 3, R5 1). The loop ended with the budget exhausted and noImprove at 2 of 2. |
| Correctness (final) | COMMANDMENT PASS: rel_l2 2.315e-3 / 2.307e-3, against a tolerance of 2e-2 global and 5e-2 per token. `ext_correct.py` passes 24 of 24. Every fallback passes: `FBS_PADSKIP=0`, `S2_PADSKIP=0`, `MOE_FAST_SORT=0`. |
| Verified, not banked | The R4 stack, `round_5/engineer_0/best_patch.diff`. It verified at **1.6575x** and is +1.76% to +1.92% over `31c4cb1` in the same session, with all correctness checks passing. It missed the 2% commit gate by 0.14%. See "Verified but not banked" below. **Bank it first in any follow-up.** |

## Round-by-round

All "verified" numbers come from the independent verifier. "Same-session" means paired, interleaved COMMANDMENT runs
against canonical. Cross-session drift is about 1%, and the GPU-to-GPU spread is up to 2.7%, so the same-session deltas
are the trustworthy ones.

### Round 0 — warm-start adopt (commit `a977b33`)
- The aiter asm fmoe kernel ran at 1 wave/SIMD (512 combined VGPR, 64 KB LDS per WG). The prior run had replaced it with
  a FlyDSL 2-stage, native-fp8 MoE running at 2 waves/SIMD:
  - **gemm1:** gate/up GEMM with fused SiLU and fp8 requant of a2; 240 VGPR, 32 KB LDS.
  - **s2:** down GEMM with topk-weighted atomic scatter; 256 VGPR, 28 KB LDS.
- R0 reprofile: **1.528x** (1.1206 / 2.0562 ms), with 6 dispatches (CK sort, ClearWorkspace and the 2 GEMMs).
- Bottleneck: compute. The fp8 MFMA GEMMs dominate the forward.

### Round 1 — no winner (0 of 3 verified)
| id | specialty | strategy | verified | result |
|---|---|---|---|---|
| r1_d0 | memory | Engage the 128-token transposed stage1 (s1v2/s1t). It fixed the SyntaxError and tuned XCD swizzle, sched and halfskip | none (`apply_failed`) | **Fail.** s1t only ties gemm1 in stage1. The forward was 0.976-0.981x / 0.992-0.994x of gemm1. The engineer reported 1.5337x late (default path unchanged). |
| r1_d1 | host_runtime | Delete the satellite zero-fill and empty-CTA waste. Custom 2-kernel sort, fill moved into gemm1 | none (`apply_failed`) | **Partial, late.** +0.62% same-session (1.5447x self-reported). Re-shipped in R2 as r2_d1. Sub-experiments: a gemm1 device loop lost 4.5%; sort `dispatch_policy=1` ran at about 0.35x. |
| r1_d2 | compute | s2 at 3 workgroups per CU | none (`apply_failed`) | **Fail, late.** 3 WG/CU was reached at 168 VGPR, but s2 got 4-6% slower. The fallback s2 setprio gave +0.7% (1.5459x self-reported) and was re-shipped in R2. |

- Every engineer ran past the roughly 60-minute verify window. Their `best_patch.diff` files were written 12:02-12:18,
  after verify, so the round had no integrate step and no winner. The cumulative stayed at 1.5279x.
- Bottleneck shift: none, since nothing was banked. The round's lesson: the kernel is compute-bound at a **power-capped
  clock** (about 1.1-1.16 GHz at 750 W), so L2-miss energy and executed work set the clock. Occupancy does not.

### Round 2 — improved: 1.5710x (commit `f3f9493`)
| id | specialty | strategy | same-session | verified | result |
|---|---|---|---|---|---|
| r2_d0 | memory | XCD-aware CTA remap of gemm1 | gn=4 0.9988x; expert-run order 0.9863x | none (dead end by design) | **Fail.** XCD tiling trades replicated-X misses for W-panel refetches. The order's dependent prologue costs 2-3.5% of stage1. |
| r2_d1 | host_runtime | Fast 2-kernel sort (k_count/k_place) on by default, with a sort-owned zero-fill | 1.5331 -> 1.5431x (+0.65%, 5/5) | **1.5535** | **Success.** Dispatches went 6 -> 4, and sort + fill went 67.7 / 110.8 -> 42.3 / 54.5 us. |
| r2_d2 | compute | s2 scheduling recipe: setprio, fpre, epi3, 1 VALU per MFMA slot | 1.5355 -> 1.5634x (+1.82%, 5/5) | **1.5629** | **Success.** s2 -8.9% / -8.8% in the reprofile. De-loop and s2 XCD tiling lost and stay off by default. |
| integrate | — | r2_d2 + r2_d1 (disjoint files) + the k_place `sort_grid` stride fix | +0.61% / +0.70% over r2_d2 (9/10 and 10/10 wins) | **1.5710** | **Winner, committed `f3f9493`.** Reprofile 1.5717x (1.0942 / 1.9910 ms). |

Bottleneck shift: 4 dispatches. gemm1 is 62.7% / 64.2% of the forward, s2 33.4% / 33.0%, sort 3.9% / 2.8%. Sustained
clock is 1.07 GHz at 750 W. gemm1 became the target.

### Round 3 — improved: 1.6273x (commit `31c4cb1`, the final canonical)
| id | specialty | strategy | same-session | verified | result |
|---|---|---|---|---|---|
| r3_d0 | compute | gemm1 recipe: per-element scalar-FMA rescale (`FBS_RS=1`, 753 pk_fma -> 0), `waves_per_eu=2`, whole-K-loop setprio, maxnumf epilogue | 1.5679 -> 1.6115x (+2.78%, 5/5) | **1.6192** | **Success.** gemm1 -6.6% / -6.7% in the trace. |
| r3_d1 | algorithm | Sort-emitted per-block valid-row count, used to skip all-padding 16-row MFMA subtiles in s2 (variant set 4,2,1). gemm1 skip priced only | 1.5693 -> 1.5805x (+0.71%) | **1.5814** | **Success.** s2 -3.6% / -2.0%, 1:1 with its MFMA cut. |
| integrate | — | r3_d0 + r3_d1, plus r3_d1's gemm1 pad-skip (`FBS_PADSKIP=4,2,1`) hand-merged onto r3_d0's gemm1 | +0.61% (22/22) for the 2-stack; the gemm1 skip added +0.31% (24/28) | **1.6273** | **Winner, committed `31c4cb1`.** Pooled median of 28 runs on GPUs 0-3: 1.0558 / 1.9248 ms. |

Bottleneck shift: the clock fell from 1.07 to 1.00 GHz. Cycles fell 9.9% / 9.5%, but time fell only 3.9% / 3.4%:
**time tracks energy per forward**. gemm1 is 60.9% / 62.1% of the forward. s2 is 34.8% / 34.9% and absorbed a
clock carry-over of +1.7% / +3.7%.

### Round 4 — +1.4% measured, NOT banked (below the 2% commit gate)
| id | specialty | strategy | same-session | verified | result |
|---|---|---|---|---|---|
| r4_d0 | compute | s2 per-element scalar-FMA rescale (`S2_RS=1`, NV=2; 457 pk_fma -> 0) | about +0.2% (pooled 26 rounds) | **1.629** | **Partial.** s2 -1.6% / -1.9%, but the next gemm1 handed back +0.3% / +1.0% through the power cap. |
| r4_d1 | algorithm | Single-local-expert rows (60% of tokens) get plain nt bf16 stores in s2; k_place zero-fills only multi-expert rows | +1.33% to +1.62% (3 sessions, 17/17) | **1.6462** | **Success.** Fill writes 89 -> 36 / 177 -> 70 MB; s2 memory-side atomics -41%. Wrapper graph capture closed at 0.9963x. |
| r4_d2 | memory | gemm1 pad-skip repair: per-variant prologue (`FBS_PS_PRO`, 35 -> 0 spills), `s_sleep` 8,12 tail pacing, PADZERO | +0.25% (15/18) | none (`apply_failed`) | **Late.** It arrived 5 minutes after integrate started. Carried into R5. |
| integrate | — | r4_d1 + r4_d0 hand-merge (disjoint regions of `flydsl_moe_fp8bs_s2.py`) | +1.60% vs R3 (30/30); +0.26% over r4_d1 (25/30) | **1.6505** | **Round winner, not committed:** it needed more than 1.6273 x 1.02 = 1.6598. |

Bottleneck shift (unbanked merge): kernel sum 1036-1042 / 1880-1897 us, with no host gaps. s2 is at its floors: 0
pk_fma, 1 irreducible FMA per MFMA, and epilogue bytes at the `[M, hidden]` contract floor.

### Round 5 (last; 1 direction, no integrate) — verified 1.6575x, NOT banked
| id | specialty | strategy | same-session | verified | result |
|---|---|---|---|---|---|
| r5_d0 | memory | Bank the R4 stack (R4 merge + r4_d2) and add a table-driven, expert-run, XCD-affine gemm1 CTA order (k_place emits a per-CTA table; gemm1 reads it with one `s_load`) | Stack +1.85% vs `31c4cb1` (12/12; +1.96% / +1.97%). The order increment gave +0.04% at best (8/14) | **1.6575** | **Partial.** The stack is confirmed, but 0.14% under the 1.6598 gate. The order lane is closed (see What didn't work). |

- Verifier, same session: +1.76% paired median (4/4) to +1.92% pooled (1.0365 / 1.8902 ms vs 1.0568 / 1.9255 ms).
- Engineer, absolute stack geomean per GPU: GPU0 1.6571x, GPU1 1.6663x, GPU2 1.6568x, GPU3 1.6222x. Only GPU1 clears
  the gate, and the verifier drew GPU0.
- Bottleneck shift: none in code. The measurement finding: the 15-19% of gemm1 L2 misses the order removed are
  MALL-served cross-XCD re-fetches, and they bought no cycles. Count bytes at HBM, not at L2.

## Final per-test-case table (canonical `31c4cb1`)

| case | params | baseline ms | optimized ms | speedup | time saved |
|---|---|---|---|---|---|
| tpr2048 | 2048 tokens/rank, 10,841 valid rows | 1.6747 | 1.0558 | **1.5862x** | -37.0% |
| tpr4096 | 4096 tokens/rank, 21,575 valid rows | 3.2135 | 1.9248 | **1.6695x** | -40.1% |
| **geomean** | | 2.3198 | 1.4256 | **1.6273x** | |
| arithmetic mean of speedups | | | | 1.6279x | |
| weighted (equal weights; the harmonic mean) | | | | 1.6268x | |

- Weights are 1 and 1 (unweighted COMMANDMENT geomean; no workload spec).
- Corroboration, R3 reprofile FULL_BENCHMARK x3: 1.0603 / 1.9207 ms = 1.6256x.
- For reference, the unbanked stack measured 1.0365 / 1.8902 ms (verifier pooled), or 1.6157x / 1.7001x. That is
  **1.6575x verified**.

## Key optimizations applied (in the final `31c4cb1`)

| # | what | origin | impact |
|---|---|---|---|
| 1 | Replace the aiter asm fmoe (1 wave/SIMD) with the FlyDSL 2-stage native-fp8 MoE (gemm1 + s2, 2 waves/SIMD) | warm start (`a977b33`) | 1.0x -> 1.528x |
| 2 | Custom bit-identical 2-kernel sort (k_count + k_place) replaces CK MoeSorting multi-phase + ClearWorkspace; 304 sort-owned fill blocks; `MOE_S1_IMPL` defaults to gemm1 | r2_d1 | +0.65% same-session. 6 -> 4 dispatches. Sort + fill -25 / -56 us |
| 3 | s2 scheduling: whole-loop setprio, fpre (needs `waves_per_eu=2`), epi3, 1 VALU per MFMA slot | r2_d2 | +1.82% same-session; s2 -9% |
| 4 | gemm1 scalar-FMA rescale (`_fma_sc`; packed FP32 stalls the MFMA pipe on gfx942), `waves_per_eu=2`, whole-K-loop setprio, maxnumf epilogue | r3_d0 | +2.78% same-session; gemm1 -6.6% / -6.7% |
| 5 | Sort-emitted `blk_cnt`, plus all-padding 16-row subtile skip (4,2,1) in s2 | r3_d1 | +0.71%; s2 -3.6% / -2.0% |
| 6 | gemm1 pad-skip (`FBS_PADSKIP=4,2,1`) hand-merged | R3 integrate | +0.31% (24/28) |

Cumulative: 1.5279x (R0/R1) -> 1.5710x (R2) -> **1.6273x (R3)**.

## Verified but not banked: the R4 stack

- **Patch:** `round_5/engineer_0/best_patch.diff`, relative to `31c4cb1`. It changes 4 files: `flydsl_moe_fp8bs.py`,
  `flydsl_moe_fp8bs_s2.py`, `moe_ep.py` and `moe_sort_fast.py`.
  - sha256 `98b5a13bee32d9cd2c3b939d797c249e1c57418f1862538d7d93ec88cdb8c8f9`.
  - It is byte-identical to `tech_lead_r5/r4_stack_vs_31c4cb1.diff` and to the verifier's `verify/applied_best_patch.diff`.
  - `git apply --check` was re-run clean against `31c4cb1` during this report phase.
- **Contents:** r4_d1 (single-expert plain stores and multi-only fill), r4_d0 (s2 scalar rescale) and r4_d2 (gemm1
  pad-skip repair).
  - Defaults ON: `MOE_SSTORE=1`, `S2_SST_POL=1`, `S2_RS=1`, `S2_RS_NV=2`, `FBS_PS_PRO=1`, `FBS_PS_SLEEP=8,12`,
    `FBS_PADZERO=1`.
  - Each knob at 0 restores the R3 ISA byte-identically.
- **Evidence:**
  - Verified 1.6575x on GPU0.
  - Same session vs `31c4cb1`: +1.76% to +1.92% (verifier) and +1.85%, 12/12 (engineer).
  - The parts add linearly: R4 merge +1.47%, r4_d2 another +0.36% (10/12).
  - Correctness: COMMANDMENT PASS (2.315e-3 / 2.307e-3); `ext_correct` 24/24; stress PASS (`sort_single` and
    `s2_sstore` engaged 24/24, with no fallback).
  - ISA: gemm1 246 VGPR, 0 spills; s2 234 VGPR, 0 spills, 0 `v_pk_fma_f32`.
- **Why it was not committed:** the gate compares one session's absolute geomean with a cross-session cumulative, and
  the result landed 0.14% under 1.6598. The gain is real but sub-gate.
- **To bank it:** in the workspace repo at `31c4cb1`, run `git apply <EVAL_DIR>/round_5/engineer_0/best_patch.diff`, then
  run COMMANDMENT correctness and FULL_BENCHMARK.
  - Use the workspace's own git repo or `patch -p1`. `git apply` run from inside the GEAK superproject subtree silently
    skips paths.

## What didn't work

The kernel ends **fp8-MFMA-bound at the 750 W socket cap**: time per forward tracks energy per forward. That one fact
explains most of the dead ends below. Under the cap, removed work converts to time as follows:

| what is removed | share converted to forward time |
|---|---|
| MFMA work | about 1:1 |
| HBM writes and memory-side atomics | about 1:1 |
| VALU or stall cycles in gemm1 | about 45% |
| VALU or stall cycles in the second kernel | about 0, because the next kernel inherits the clock |
| L2 misses served by MALL | about 0 |

- **Stage1 structure:**
  - The 128-row transposed stage1 loses to the 64-row gemm1: it doubles each XCD's working set and adds unit-128
    padding.
  - A device loop around gemm1 costs 4.5%.
  - Fusing stage1 into stage2 was never funded, because the a2 round-trip is compulsory traffic.
- **CTA order and XCD remaps (all closed):**
  - gemm1: plain XCD tiling, the gn=16 M-partition, and the expert-run order, both with a prologue and with an `s_load`
    table (gated, ungated, chunked and balanced).
  - s2: every XCD tiling.
  - W13 sibling sharing in L2 is worth about a fifth of the forward. The X misses these remaps remove are MALL-served
    and do not convert.
- **s2 structure:** de-looping the degenerate grid loop loses, because s2's empty CTAs are not free. A 3rd wave/WG per
  CU loses at the power cap.
- **Instruction-level:**
  - Packing FP32 near MFMAs is inverted on gfx942.
  - VALU-slot scheduling and nop removal, address-VALU trims, fenced prologue fmuls and in-asm execz skips all measured
    as noise.
  - Per-tile setprio and MF32 lose.
- **Skip and pacing:**
  - s2 variant sets other than 4,2,1 perturb register allocation.
  - gemm1 pacing beyond `s_sleep` 8,12 (self-timed pacing, global lockstep) loses.
  - Cache-policy bits never help; nt on W13 costs 20-24%.
- **Host and runtime:**
  - Wrapper CUDA-graph capture loses 0.37%, since there are no host gaps.
  - CK sort `dispatch_policy=1` runs at about 0.35x, and opus sorting equals CK.
  - Every fill-placement variant ties or loses: the cost tracks HBM bytes, not placement.
- **Not pursued (no evidence either way):** r1_d1's in-gemm1 fill at `zf_pos=3` (about +1%, weak and unverified). It was
  superseded by the sort-owned fill and then by r4_d1's multi-only fill.

<!-- dead-ends:yaml -->
```yaml
- idea: "128-row transposed stage1 (s1v2/s1t, MOE_S1_IMPL=v2) instead of the 64-row gemm1"
  measured: "forward 0.976-0.981x / 0.992-0.994x vs gemm1 (engineer, unverified)"
  mechanism: "halves the B stream per FLOP but doubles each XCD's working set (L2 hit 69% vs 77%); unit-128 sort padding adds 2-4% MFMA work and about 15 us of s2"
- idea: "dynamic device grid-stride loop around the gemm1 body"
  measured: "0.955x forward (harness n=3)"
  mechanism: "the scf.for perturbs gemm1 codegen and scheduling; the same M-slot bound without the loop is neutral"
- idea: "aiter moe_sorting dispatch_policy=1 (single kernel) and opus_moe_sorting"
  measured: "dispatch_policy=1 about 0.35x (forward 3.13 / 6.15 ms); opus about 1.00x vs CK"
  mechanism: "the single MoeSortingKernel takes 1.7-3.9 ms; only the custom k_count/k_place sort wins"
- idea: "3rd co-resident workgroup per CU on s2 (waves_per_eu=3, 21.5 KB LDS ring, v3 restructure)"
  measured: "s2 4-6% slower (engineer, reported after the round 1 deadline, unverified)"
  mechanism: "reached at 168 VGPR, but per-wave memory waits doubled and the power-capped clock dropped; occupancy is not the limiter at the 750 W cap"
- idea: "plain XCD tiling of gemm1 (gn=4) and the gn=16 M-partition"
  measured: "gn=4 0.9988x forward; gn=16 stage1 -4.9% / -10.9%"
  mechanism: "trades replicated-X misses for W13-panel refetches; X misses are MALL hits, so the X-hot ceiling is only 3.2% / 5.5% of stage1"
- idea: "gemm1 expert-run CTA order computed in a kernel prologue (seid window load + ballot + division)"
  measured: "0.9863x forward"
  mechanism: "cut gemm1 L2 misses 25.5% / 9.5%, but the dependent prologue costs 2-3.5% of stage1"
- idea: "gemm1 expert-run XCD-affine CTA order from a k_place-emitted per-CTA table read with one s_load (FBS_ORD=1, FBS_ORD_GATE=1)"
  measured: "+0.04% forward (8/14 paired wins; tpr2048 +0.27%, tpr4096 -0.14%)"
  mechanism: "bit-exact and off the critical path, but the 15-19% of gemm1 L2 misses it removes at tpr2048 are MALL-served cross-XCD X re-fetches; active cycles flat to +3%"
- idea: "ungated per-CTA order table (expert-run and identity orders)"
  measured: "expert-run about 0.998x (ord0 won 13/16); identity about 0.997x"
  mechanism: "about 65K empty CTAs each s_load a cold, never-written table line (about 0.3%); bounding the load by num_valid first cuts that to about 0.06%"
- idea: "expert-run order with a run chunk cap of 8 (FBS_ORD_C=8), or an exactly balanced XCD partition"
  measured: "chunk cap 0.9917x (0/12); balanced 0.997x (5/24)"
  mechanism: "re-streaming W13 per chunk costs more than the X reuse it buys; the tpr4096 loss is not XCD imbalance"
- idea: "zero-fill placement variants (side stream, serial pre-s2, fused into gemm1 zf_pos=1, AnyOrderLaunch, grid-barrier single-kernel sort, zsplit, 176 blocks, NT stores)"
  measured: "best variant 1.00x (ties the sort-owned fill); side stream -0.3% to -6%; fused into gemm1 adds 62 us of gemm1 at tpr4096"
  mechanism: "fill cost tracks the 89 / 177 MB of HBM writes, not where they run; only writing fewer bytes helped (r4_d1 multi-only fill)"
- idea: "remove s2's degenerate device grid loop (S2_DELOOP=1)"
  measured: "s2 -3.6% / -1.4% (isolated, final kernel)"
  mechanism: "s2's ~61K empty guarded CTAs are not free, even though VGPR fell 253 -> 220"
- idea: "XCD-tile the s2 grid (S2_XCD_GN=2..16, chunked N-major)"
  measured: "s2 -1.4% / -0.9% (gn=2) to -11.5% / -12.9% (gn=16)"
  mechanism: "the plain (16 N, M) grid keeps each output column range's atomics on one XCD and keeps W2-panel sharers co-resident"
- idea: "pack FP32 rescale FMAs into v_pk_fma_f32 inside the MFMA loop (gemm1 and s2)"
  measured: "inverted: MFMA + 1 v_pk_fma = 30.1 cycles vs 17.1 for MFMA + 2 scalar FMAs (micro-bench)"
  mechanism: "on gfx942 V_PK_*_F32 waits for the matrix pipe to drain; the scalar form shipped (FBS_RS=1 in 31c4cb1, S2_RS=1 in the unbanked stack)"
- idea: "VALU-slotted sched_group_barrier on gemm1 (FBS_SCHED=5) and nop removal"
  measured: "gemm1 within the null arm (+0.1..+0.9% / -1.2..+0.5%)"
  mechanism: "K-loop s_nop cycles fell 1325 -> 425 with no time change; stall cycles cost no energy at the power cap"
- idea: "per-tile s_setprio placement in gemm1 (FBS_G1_PRIO=2)"
  measured: "gemm1 -5.4% / -4.8%"
  mechanism: "s_setprio is a scheduling-region boundary; per-tile toggles cut each tile's sched_group_barrier pattern off its MFMAs; only whole-K-loop placement helps"
- idea: "MF32 32x32x16 MFMA in gemm1, and a BPF / DEFER / SCHED_G1 / FBS_PRO retune"
  measured: "MF32 gemm1 -3.9% / -5.3% (0/7); retunes within the +-0.5% null arm"
  mechanism: "the 16x16 schedule with the scalar rescale is better; defaults stay SCHED=3, SCHED_G1=8, DEFER=1, BPF=2"
- idea: "s2 pad-skip variant sets other than 4,2,1 (4,1 / 4,2 / 4,3 / 4,3,2,1)"
  measured: "s2 -1.6%/-1.0% (4,1), -0.5%/0.0% (4,2), -0.3%/0.0% (4,3,2,1) vs -2.2%/-1.2% for 4,2,1"
  mechanism: "extra variant bodies perturb the full body through the global register allocator (extra v_mov or 14-36 spills)"
- idea: "s2 fenced scalar prologue fmuls (S2_PRO_SC, llvm.arithmetic.fence)"
  measured: "about 1.00x"
  mechanism: "packed ops outside the K loop cost nothing measurable"
- idea: "s2 B-load immediate-offset addressing (S2_BIMM) and soffset address trims"
  measured: "about 1.00x (noise)"
  mechanism: "removing 92 VALU per tile is not an energy lever at the power cap"
- idea: "in-asm s_cbranch_execz over the dead s2 store/atomic pair (S2_SST_SKIP)"
  measured: "+0.30% vs a +0.24% null arm (about 1.00x)"
  mechanism: "exec=0 memory ops already cost about nothing"
- idea: "wrapper-level torch.cuda.CUDAGraph capture/replay of the forward (MOE_GRAPH)"
  measured: "0.9963x (graph won 0/6 and 2/6)"
  mechanism: "4 dispatches with span equal to the kernel sum leave no host gaps to remove"
- idea: "gemm1 pad-skip pacing beyond s_sleep 8,12 (self-timed tail pacing, global lockstep speed limit)"
  measured: "global lockstep -0.1% to -2.0% forward; self-timed pacing no better than 8,12"
  mechanism: "the residual 1-1.4M MALL-hit sibling re-fetches are worth less than any enforced pace"
- idea: "gfx942 cache-policy bits on gemm1 W13 / X loads and a2 / a2-scale stores"
  measured: "W13 nt 0.76x / 0.80x forward; X nt about 0.93x; other bits within +-0.4%"
  mechanism: "sibling-CTA W13 reuse in L2 is worth about a fifth of the forward; never weaken it"
```

## Notes for a follow-up run

1. **Bank the verified stack first.**
   - What: `round_5/engineer_0/best_patch.diff` on `31c4cb1`.
   - Expected: about +1.8% same-session, which takes the result to about 1.657-1.666x depending on the GPU.
2. **Then fund only HBM-byte or MFMA-op removal.** Each lever below is estimated at most +1% and none was measured
   above that:
   - the gemm1 epilogue (ceiling +5.3% / +3.8% of gemm1, mostly compulsory a2 stores);
   - sort + fill HBM bytes (a 13-19 us gap between the in-forward and isolated sort);
   - residual partial-subtile padding (2.29% / 1.29% of MFMA rows).

   The kernel is near its energy floor for this algorithm at the 750 W cap. Do not re-fund anything in the dead-ends
   block.
3. **Gate planning.**
   - Per-direction gains here are 0.2-1.6%, while the commit gate needs +2% against a cross-session cumulative.
   - Keep 2 or more directions for the final round so integrate can stack increments, or reach the final round at least
     0.5% above the gate.
   - The verifier's GPU alone moves the absolute geomean by up to 2.7%.
4. **Measurement protocol** that resolves 0.2-0.3% effects:
   - 12-30 interleaved COMMANDMENT rounds with rotating arm order, under one gpu_lock lease;
   - pooled over 2-4 GPUs, with a null arm;
   - every arm run from a frozen snapshot, because FlyDSL traces its source at first call.

   Tools for this are in `round_4/integrate/tools/` and `round_5/engineer_0/tools/`.
5. **A dead file ships in the final patch.**
   - `flydsl_moe_fp8bs_s1v2.py` arrived with the warm-start commit `a977b33`. It still fails to parse: line 159, an
     unclosed paren.
   - It is inert: `moe_ep.py` imports it only under `MOE_S1_IMPL=v2`, inside `try/except Exception`, and falls back to
     gemm1. The default is gemm1.
   - A follow-up may delete it or leave it. It does not affect the measured result.
6. **Knowledge-base defect (not edited here):** `kernel_workflow/knowledge/amd_instinct.md` line 49 says the L2 is
   "large, shared across CUs (256 MB on MI300X-class)".
   - On this gfx942 box the L2 is 4 MB per XCD (8 XCDs; XCD == linear CTA id % 8), and the 256 MB is the device-wide
     Infinity Cache (MALL).
   - This run's CTA-order, pad-skip and cache-policy conclusions depend on that split.
7. **Expert skills (advisory):** `perf_knowledge/expert_skills/index.yaml` is unchanged since Nov 18 11:22.
   - The only validated, kernel-scope, gfx942 fp8_e4m3_fnuz match for `fused_moe_grouped_gemm` is
     `flydsl_rewrite_quantized_moe`. It is a triton->flydsl migration recipe, and this run is FlyDSL-native, so there is
     nothing to reproduce.
   - Its live constraint is respected: stage-2 output stays `[M, hidden]` with in-kernel accumulate.
   - The blkmap and stage2_fp8partial skills are gfx950-only.

## Artifacts

| artifact | path (under EVAL_DIR) | notes |
|---|---|---|
| Final cumulative patch | `final_patch.diff` | Root `0c5bd5f`..`31c4cb1`, 5 files, +3640/-1. sha256 `e06533da...`, identical to `current_best.diff`. |
| Optimized sources (canonical) | `optimized/` | `moe_ep.py`, `flydsl_moe_fp8bs.py`, `flydsl_moe_fp8bs_s2.py`, `moe_sort_fast.py`, `flydsl_moe_fp8bs_s1v2.py` (inert), and `aiter_fused_moe.py` (unchanged fallback) |
| Unbanked verified stack | `round_5/engineer_0/best_patch.diff` (identical copy at `tech_lead_r5/r4_stack_vs_31c4cb1.diff`) | |
| Closed-lane reference diffs | `round_2/engineer_0/xcd_remap_explore_NOT_A_CANDIDATE.diff`, `round_3/engineer_0/slot_sched_prologue_knobs_NOT_A_CANDIDATE.diff`, `round_3/engineer_1/gemm1_padskip_NOT_A_CANDIDATE.diff`, `round_4/engineer_0/{b_imm_offset,prologue_pksc}_NOT_A_CANDIDATE.diff`, `round_4/engineer_1/{graph_capture,sst_execz_skip}_NOT_A_CANDIDATE.diff`, `round_4/engineer_2/r4d2_selftimed_pace_globalpace_cachepolicy_dbgts_NOT_A_CANDIDATE.diff`, `round_5/engineer_0/order_NOT_A_CANDIDATE.diff` | |
| Per-round memory | `insight_log.md`; `round_{0,2,3}_metrics.json`; `round_{0,2,3}_shift_analysis.md` | |
