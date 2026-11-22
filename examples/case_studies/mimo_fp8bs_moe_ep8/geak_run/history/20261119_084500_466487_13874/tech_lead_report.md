# TechLead Report — mimo_fp8bs_moe_ep8 (fp8 block-scale fused-MoE, EP8, gfx942)

## Summary
- **Kernel**: `mimo_fp8bs_moe_ep8` — fp8 block-scale fused Mixture-of-Experts, expert-parallel 8-way
  dispatch, two-stage (gate/up GEMM1 + down-proj GEMM2). Backend: FlyDSL native-fp8 two-stage rewrite.
- **Kernel type**: composable / DSL (FlyDSL), gfx942 (CDNA3), power-capped throttled clock.
- **Final speedup (headline, unweighted geomean)**: **1.534×** vs the frozen aiter asm baseline.
  - Unweighted arithmetic mean: **1.5345×**
  - Ratio-of-sums (time-weighted, both cases weight 1): **1.546×**
  - (Run is not workload-aligned; geomean is the headline metric.)
- **Rounds**: 2 optimization rounds (plus round-0 reprofile). **Budget used**: 4 engineer directions
  (2 per round).
- **Net result**: No round improved on the warm-started cumulative. The adopted state remains the
  warm-started FlyDSL two-stage kernel at **1.534×**. The compute axis is now near-exhausted and all
  cheap levers are closed dead-ends (see below).

## Starting point
Warm-started from the committed FlyDSL native-fp8 two-stage rewrite
(`warm_start:mimo_fp8bs_moe_ep8__flydsl__gfx942`, geomean 1.534×). The kernel is **compute / MFMA-issue
bound**, running 2 waves/SIMD at 240/256 combined VGPR, with HBM traffic < 20% of peak. The ~1.54×
bar sits against an estimated ~1.8× roofline; the power-capped throttled clock caps fp8 compute
headroom above ~1.54×.

## Round-by-round

### Round 1
- **r1_d0 — algorithm — "Engage the 128-row transposed-T stage1 (gate/up GEMM)"**:
  **FAIL (regression, verified 1.3746× vs adopted 1.527×)**. The 128-row transposed-T stage1 is
  fundamentally a *memory* optimization (halves the weight-B stream per FLOP) applied to a
  *compute-bound* stage. It engages and is bit-exact parity-passing but runs ~10% slower than the
  adopted 64-row gemm1 fallback — the one-token-per-lane layout lowers MFMA packing efficiency
  (engaged T avg GEMM ~1075us > 64-row ~1020us). The claimed ~263KB LDS was wrong (actual 32KB, fits,
  occupancy unchanged). Not shipped. The `s1v2.py:161` unclosed-paren fix is real but auto-enables
  `use_v2`, which silently regresses the adopted path — only land it gated OFF.
- **r1_d1 — compute — "Stage2 down-proj: MFMA issue/dependency + LDS-bank-phase co-tune"**:
  **FAIL (apply_failed, all knobs within run-to-run noise ~0.5–0.7%)**. At tile_m=64/tile_n=256 the
  stage2 compute/issue + epilogue + cache axis is near-closed: manual scheduler sched=1 already best;
  deepening the B pipeline (2-tile-ahead) regresses; bf16 epilogue is off the critical path
  (epi_mode=2 perf-neutral); L2 hints don't help (already L2-friendly).
- **Integrate**: none. **Round winner**: none. **IMPROVED=false.**
- **Bottleneck shift**: unchanged — compute / MFMA-issue bound.

### Round 2
- **r2_d0 — compute — "Stage1 gemm1: push to a 3rd resident wave (accumulator tiling + MFMA schedule)"**:
  **FAIL (apply_failed, nothing beat current-best)**. Confirmed dead-end with hard register data:
  engaged `moe_fp8bs_gemm1` is VGPR=112 / Accum_VGPR=128 (combined 240 → 2 waves/SIMD). A 3rd resident
  wave needs combined ≤168; no in-file knob lowers it (FBS_DEFER=0 goes *higher* at 132 accum; every
  scheduler/prefetch/setprio variant flat-to-worse). Defaults SCHED=3/G1=8/DEFER=1/BPF=2/SETPRIO=0 are
  already occupancy/issue-optimal. The only route (N/M accumulator-subpass tiling) breaks the fused
  1×128 requant epilogue, still misses 168, adds HBM re-stream traffic, and cannot issue more MFMAs on
  an issue-bound kernel. Occupancy is not the stage1 lever.
- **r2_d1 — algorithm — "Stage2 down-proj: structural 32×32×16 MFMA tile-shape remap (+ LDS pad / split-K eval)"**:
  **FAIL (apply_failed, no variant beat 1.5397×)**. Split-K / KBatch is a closed dead-end — the grid
  is already filled (8704 WG ≫ 304 CU); split-K adds atomic contention and regresses monotonically
  (k_batch=2 → 1.27×). The lone untried lever is the 16×16×32 → 32×32×16 fp8 MFMA tile remap: feasible
  in-lane but a large bit-exact rewrite for an issue-count-only gain, with 32×32's higher
  per-instruction latency risking dependency stalls in the hand-scheduled inner loop. Not attempted.
- **Integrate**: none. **Round winner**: none. **IMPROVED=false.**
- **Bottleneck shift**: unchanged — compute. (No SoL/MFMA% counters this round; rocprof-compute
  missing-deps env failure persists — analysis rests on VGPR/accum occupancy + isolated A/B latency.)
- **Expert skills**: no validated expert skill applies — the one kernel-scope FlyDSL MoE stage1 skill
  (`flydsl_decode_moe_stage1_blkmap`) is gated to gfx950; this box is gfx942.

## Final per-test-case table
| case     | params                                      | baseline ms | optimized ms | speedup |
|----------|---------------------------------------------|-------------|--------------|---------|
| tpr2048  | tokens_per_rank=2048, max_dispatch=4096, rows=32768, valid~10841 | 1.6751      | 1.1192       | 1.4967× |
| tpr4096  | tokens_per_rank=4096, max_dispatch=4096, rows=32768, valid~21575 | 3.2118      | 2.0429       | 1.5722× |

- **Geomean (headline)**: 1.534×
- **Arithmetic mean**: 1.5345×
- **Ratio-of-sums (weighted, both weight 1)**: (1.6751+3.2118)/(1.1192+2.0429) = **1.546×**
- Baseline geomean: 2.3195 ms → optimized geomean: 1.5122 ms.

## Key optimizations applied (the adopted warm-start kernel)
- **FlyDSL native-fp8 two-stage rewrite** of the fused MoE, replacing the aiter asm baseline: a
  64-row `moe_fp8bs_gemm1` (gate/up) with a fused 1×128 per-row fp8 requant epilogue, feeding a
  tile_m=64/tile_n=256 fp8 down-proj GEMM2.
- **Fused gate/up MFMA interleave + deferred accumulator flush** on stage1, which already hides the
  inter-MFMA dependency bubbles a 3rd resident wave would otherwise fill — this is why occupancy
  tuning yields nothing.
- **Tuned stage-local schedules** (stage1 SCHED=3/G1=8/DEFER=1/BPF=2; stage2 sched=1, 1-tile-ahead B
  pipeline, epi_mode register-packed epilogue) — all confirmed at their optima this run.

## What didn't work (closed dead-ends)
- 128-row transposed-T stage1 (memory opt on a compute-bound stage) — regresses ~10%.
- Stage1 3rd-resident-wave push (accumulator tiling / all occupancy knobs) — unreachable; combined
  240 VGPR, needs ≤168, no knob lowers it; N/M subpass tiling breaks the requant epilogue.
- Stage2 split-K / KBatch — filled grid, atomic contention, monotonic regression.
- Stage2 compute/issue/epilogue/cache micro-tuning — all inside run-to-run noise at
  tile_m=64/tile_n=256.
- `s1v2.py:161` paren fix — real bug, but auto-enables `use_v2` which regresses the adopted path;
  only land gated OFF.

The only untried lever is the stage2 16×16×32 → 32×32×16 fp8 MFMA tile remap — a high-cost bit-exact
rewrite for an issue-count-only gain on a power-capped clock that caps headroom above ~1.54× (~1.8×
roofline). Do NOT re-fund stage1 occupancy, transposed-T weight-B stream, or stage2
split-K/scheduler/epilogue/cache micro-tuning.

<!-- dead-ends:yaml -->
```yaml
- idea: 128-row transposed-T stage1 (gate/up GEMM), weight-B stream halving
  measured: 1.3746x
  mechanism: memory opt on a compute/MFMA-issue-bound stage; one-token-per-lane layout lowers MFMA
    packing efficiency, ~10% slower than the adopted 64-row gemm1 fallback (1.527x). Bit-exact but not shipped.
- idea: stage1 gemm1 3rd-resident-wave push (accumulator tiling + occupancy/issue knobs)
  measured: no knob beat current-best; FBS_DEFER=0 profiled HIGHER (Accum=132)
  mechanism: combined 240 VGPR (arch 112 + accum 128), a 3rd wave needs <=168; no in-file knob lowers
    it, and N/M accum-subpass tiling breaks the fused 1x128 requant epilogue, still misses 168, adds HBM
    re-stream, and cannot issue more MFMAs on an issue-bound kernel. Occupancy is not the stage1 lever.
- idea: stage2 down-proj split-K / KBatch
  measured: k_batch=2 = 1.27x (regresses from 1.53x)
  mechanism: grid already filled (8704 WG >> 304 CU); split-K adds atomic contention and regresses
    monotonically. k_batch=1 is bit-identical to current-best.
- idea: stage2 down-proj compute/issue + epilogue (epi_mode) + L2 cache + B-pipeline depth micro-tune
  measured: all within run-to-run noise (~0.5-0.7%)
  mechanism: near-closed at tile_m=64/tile_n=256; sched=1 and 1-tile-ahead already optimal, bf16
    epilogue off the critical path. Needs a structural MFMA-tile-shape change, not micro-tuning.
- idea: s1v2.py:161 unclosed-paren fix (ungated)
  measured: regresses adopted path 1.527x -> 1.388x
  mechanism: the fix auto-enables use_v2 (128%64==0), silently selecting the slower 128-row path. Only
    land gated use_v2=OFF.
```
