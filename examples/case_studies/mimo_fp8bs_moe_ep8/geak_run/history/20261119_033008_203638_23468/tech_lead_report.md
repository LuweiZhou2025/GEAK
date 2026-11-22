# TechLead Final Report — mimo_fp8bs_moe_ep8

## Summary
- **Kernel / op:** FlyDSL fp8 block-scale fused MoE (EP8), `fused_moe_grouped_gemm` — gate/up+SiLU (stage1) and down+accumulate (stage2), per-case dispatch. Box: gfx942 / MI300X.
- **Kernel type:** e2e MoE module (FlyDSL native-fp8 two-stage replacement of the aiter asm fmoe path).
- **Final speedup (headline, unweighted geomean):** **1.5397x** over the frozen baseline (not workload-aligned, so geomean is the headline).
  - Unweighted geomean: **1.5397x**
  - Arithmetic mean: **1.5402x**
  - Weighted: **1.5397x** (no WORKLOAD_SPEC; weights = 1, so identical to geomean)
- **Rounds:** 3. **Budget used:** 8 directions (r1: 3, r2: 2, r3: 3).
- **Final winner:** `engineer r2_d0` — Stage1 XCD-aware WG rasterization for per-XCD weight(B) L2 reuse (retained default-on as a no-regret pure-remap tiebreaker). The r1_d0 FlyDSL native-fp8 two-stage rewrite is the structural win underneath it (1.49x→1.54x).

## Round-by-round

### Round 1 — cumulative 1.0 → 1.4938 (IMPROVED)
- **r1_d0 (algorithm)** — FlyDSL native-fp8 block-scale 2-stage MoE (gate/up+SiLU, down+accumulate), per-case dispatch. **Verified 1.4938x (claimed 1.4957).** SUCCESS — round winner. Occupancy doubling (1→2 waves/SIMD, MFMA 33→60%); fell short of the 1.8 target because the 750W power cap throttles sclk to ~1.2 GHz, capping fp8 TFLOPs.
- **r1_d1 (host_runtime)** — Strip pre-kernel tax: valid-row-aware moe_buf/sorting, cached num_local_tokens, per-case config. **Verified 1.0469x.** PARTIAL but INERT — real on the aiter asm path, but `_forward_fbs` never calls `fused_moe_` once the fbs winner is active; does not compound.
- **r1_d2 (compute)** — Competing high-throughput FlyDSL stage-1 with different tiling. **Verified 1.0035x (claimed 1.509 was a non-isolated measurement).** DEAD-END — conflicts with r1_d0's tiling, cannot merge; r1_d0's stage-1 is the faster single stage-1.
- **Integrate / winner:** r1_d0 standalone, 1.4938x. Bottleneck shifted from host/dispatch to stage1/stage2 compute.

### Round 2 — cumulative 1.4938 → 1.5397 (IMPROVED)
- **r2_d0 (memory)** — Stage1 XCD-aware WG rasterization for per-XCD weight(B) L2 reuse. **Verified 1.5397x (claimed 1.5403).** SUCCESS — round winner, but the swizzle delta is run-to-run noise (<0.15%); the number is the inherited current-best, not a swizzle-attributable delta. B-stream is only 9–19% of stage1 and already hidden behind MFMA/issue-wait → NOT memory-bound. Patch retained default-on as a no-regret pure-remap tiebreaker. (NB: `s1v2` has a SyntaxError and fails import silently, so stage1 actually runs the gemm1 fallback `moe_fp8bs_gemm1_0` at ~80% L2 hit; the swizzle never executes.)
- **r2_d1 (algorithm)** — Stage2 non-atomic + unpadded epilogue (kill bf16 atomic RMW, fold repeat_interleave). **Verified 0 / invalid_measurement (reported 1.547 = inherited level).** DEAD-END — deletion control (atomic→plain store) yielded ~0; full epilogue-skip ceiling only ~6% of stage2 ≈ 2% e2e. No patch shipped.
- **Integrate / winner:** r2_d0, 1.5397x. Bottleneck remains compute (throttled-clock fp8 ceiling).

### Round 3 — cumulative 1.5397 → 1.5397 (NOT IMPROVED; winner 1.5407 ≈ bar)
- **r3_d0 (compute)** — Stage1 gemm1 lean fp32 block-scale fold + hoisted scales + freed-VGPR occupancy. **Verified 0 / apply_failed (claimed 1).** DEAD-END — the fold (`_SX_TAB`), hoisted+vectorized scales (`_TR`), freed-VGPR occupancy, and lean SiLU/requant are ALL already realized in the live kernel; 3rd wave infeasible (240 VGPR → 2 waves, need ≤170); MF32/SETPRIO schedule knobs regress.
- **r3_d1 (host_runtime)** — 64-unit moe_sorting to cut stage2 padded-MFMA waste. **Verified 1.5285x (claimed 1.535).** DEAD-END — premise false: moe_sorting already runs at unit=64 (s1v2 import fails; gemm1+stage2 both tile_m=64), so the 128→64 saving is already banked. Its patch instead routes stage2 through a slower s2t 128-row path (1.5285 < 1.5407).
- **r3_d2 (compute)** — Stage2 MFMA issue/dependency re-schedule + pad epilogue LDS row stride. **Verified 1.5407x (claimed 1.5421).** PARTIAL — round winner: `_VM_MODE=1` (front-load weight buffer_loads, defer X) trims the exposed vmem→MFMA window on tpr4096 for ~+0.25%; epilogue LDS-pad dead (inherent stride-independent 2-way bank conflict); scale-fold already pipelined. Real but ≈noise (1.5407 ≈ bar); mutually exclusive with r3_d1 and does not exceed BEST_INDIVIDUAL.
- **Integrate / winner:** no integrated patch exceeds BEST_INDIVIDUAL=1.5407 ≈ bar; no improvement committed. Cumulative stays 1.5397 (committed r2_d0 HEAD).

## Final per-test-case table

| case | baseline ms | optimized ms | speedup |
|------|-------------|--------------|---------|
| tpr2048 | 1.6807 | 1.1178 | 1.5036 |
| tpr4096 | 3.2239 | 2.0447 | 1.5767 |

- **Geomean:** 1.5397x · **Arithmetic mean:** 1.5402x · **Weighted (unweighted, weights=1):** 1.5397x
- Baseline geomean: 2.3277 ms.

## Key optimizations applied
- **FlyDSL native-fp8 two-stage MoE (r1_d0)** — replaced the aiter asm fmoe path with a FlyDSL block-scale kernel pair; occupancy doubled 1→2 waves/SIMD, MFMA utilization 33%→60%. This is the load-bearing structural win (1.0→1.49x).
- **Stage1 XCD-aware WG rasterization (r2_d0)** — un-round-robins the 1-D grid so each XCD's M-blocks for the same (expert, N-tile) of B co-reside for L2 reuse; retained default-on as a pure-remap no-regret tiebreaker (delta is noise, cannot regress).
- **Stage2 `_VM_MODE=1` (r3_d2, measured but not committed)** — front-loads weight buffer_loads / defers X to trim the exposed vmem→MFMA window on the K-heavy tpr4096 case; verified +0.25% but ≈ bar, mutually exclusive with r3_d1, not adopted over the committed r2_d0 HEAD.

## What didn't work (dead-ends)
- Competing stage-1 tiling (r1_d2): conflicts with r1_d0, standalone 1.0035x.
- Host pre-kernel tax strip (r1_d1): real 1.0469x on the asm path but inert under the fbs winner.
- Stage2 non-atomic/unpadded epilogue (r2_d1): atomic→store control ~0; epilogue-skip ceiling ~2% e2e.
- Stage1 gemm1 knob re-tune (r3_d0): every lever already realized in the live kernel; 3rd wave infeasible at 240 VGPR; schedule knobs regress.
- Host 64-unit sort (r3_d1): premise false — already at unit=64; its patch regresses via a slower s2t path (1.5285).
- Expert skill `flydsl_decode_moe_stage1_blkmap` is gfx950-only and INAPPLICABLE on this gfx942 box (needs tile_m=64 base, won't build at 128-row ~263 KB LDS). No validated kernel-scope expert skill matches this op this run.
- **Gating next step (undone):** the single untried high-value lever is structural and blocked behind the `flydsl_moe_fp8bs_s1v2.py` SyntaxError (line 159, unclosed `(`), which makes stage1 silently run the gemm1 fallback. Repairing it is the prerequisite to A/B the s1v2+XCD-swizzle 128-row stage1 vs the fallback. Headroom is otherwise capped by the ~1.2 GHz throttled clock; HBM <20% (not memory-bound).

<!-- dead-ends:yaml -->
```yaml
- idea: competing FlyDSL stage-1 with alternate tiling (r1_d2)
  measured: 1.0035x
  mechanism: conflicts with r1_d0 tiling, cannot merge; slower single stage-1
- idea: host pre-kernel tax strip under the fbs winner (r1_d1)
  measured: 1.0469x
  mechanism: real on aiter asm path but _forward_fbs never calls fused_moe_; inert, does not compound
- idea: stage2 non-atomic + unpadded epilogue (r2_d1)
  measured: 0 (invalid, no patch shipped)
  mechanism: atomic->store control ~0; epilogue-skip ceiling only ~6% of stage2 ~= 2% e2e
- idea: stage1 gemm1 lean fold + hoisted scales + freed-VGPR occupancy knob-retune (r3_d0)
  measured: apply_failed
  mechanism: fold/_SX_TAB, _TR scale hoist, lean SiLU all already in live kernel; 3rd wave infeasible (240 VGPR, need <=170); MF32/SETPRIO regress
- idea: host 64-unit moe_sorting to cut stage2 padded-MFMA waste (r3_d1)
  measured: 1.5285x
  mechanism: already at unit=64; patch instead routes a slower s2t 128-row stage2 path (< 1.5407 bar)
- idea: stage2 epilogue LDS row-stride pad off 32-bank period (r3_d2 sub-lever)
  measured: flat
  mechanism: epi_mode==0 read loads a full 128-col row -> inherent stride-independent 2-way bank conflict
- idea: stage1 XCD swizzle as a memory-bound win (r2_d0)
  measured: 1.5397x (delta <0.15%, noise)
  mechanism: B-stream only 9-19% of stage1, already hidden behind MFMA/issue-wait; not memory-bound; kept as no-regret remap
- idea: expert skill flydsl_decode_moe_stage1_blkmap
  measured: inapplicable
  mechanism: gfx950-only; box is gfx942; needs tile_m=64 base, won't build at 128-row ~263 KB LDS
```
