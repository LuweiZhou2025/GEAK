# Roadmap — mimo_fp8bs_moe_ep8 (WARM-START, bar to BEAT = 1.5397x)

## Summary
Rank-local EP8 fp8 block-scale fused MoE. The prior run already replaced the aiter asm `fmoe` path
with a FlyDSL native-fp8 two-stage MoE (stage1 gate/up+SiLU, stage2 down+bf16 accum) and committed
**1.5397x geomean** (tpr2048 1.1178 ms, tpr4096 2.0447 ms vs baseline 1.6807 / 3.2239 ms). The win is
occupancy (1->2 waves/SIMD) + MFMA util 33%->60%, plus an XCD-aware WG remap. This continuation treats
1.5397x as the bar, NOT baseline. The op is compute/issue-bound under a 750W power cap that throttles
sclk to ~1.2 GHz (realized fp8 ~19-20% of nameplate; HBM <20%), so the realistic FlyDSL roofline is
~45-60% of peak (~0.6-0.7 ms tpr2048 / ~1.2-1.4 ms tpr4096). Headroom is MFMA issue efficiency and a
possible 3rd wave (needs <=170 VGPR, currently ~240).

## Bottleneck hypothesis
Compute/issue-bound MFMA. The streamed B operand (weights) per FLOP is the limiter in stage1; raising
the M tile to share B through LDS/registers cuts B requests per FLOP. Occupancy is the second axis
(240 VGPR -> 2 waves/SIMD; a 170-VGPR variant would give 3 waves and more latency hiding).

## Round strategy (sketch, mapped to specialties)

### Round 1 (highest value) — repair & engage the real 128-row transposed stage1
- **compute/memory (primary):** Fix the SyntaxError at `flydsl_moe_fp8bs_s1v2.py:159`
  (missing closing paren on `int(_os.environ.get("FBS_XCD_N","8")`). Make s1v2 import and build cleanly
  at tile_m=128 on gfx942, watching LDS (~263KB budget) and VGPR. Then isolated A/B of the real
  transposed 'T' 128-row stage1 (+ XCD swizzle) vs the live 64-row gemm1 fallback, per case, against
  the FROZEN baseline. CONFIRM ENGAGEMENT by the live grid shape (swizzled (G,1,1) vs 2-D fallback
  grid), never by the headline wall-clock — the prior "winner" silently ran the fallback.
  Rationale: this transposed stage1 (weights=src-A, tokens=src-B, one token/lane -> per-lane scalar
  activation block-scale, halving B stream per FLOP) is the one lever round 3 identified and it has
  NEVER actually executed. This is the single most likely source of real gain over 1.5397x.

### Round 2 (reserve) — occupancy / issue-efficiency on the winning stage1
- **compute:** If round 1 engages the 128-row T stage1, push toward a 3rd wave: trim VGPR to <=170
  (accumulator layout, scale-promotion scheduling) without regressing MFMA issue. Measure MFMA util
  and waves/SIMD from the trace, not just wall-clock. Keep orthogonal to the stage1 tiling change.
- **memory (alt):** retune the XCD swizzle / WG remap for the 128-row grid specifically (the prior
  swizzle was tuned against — and only ever ran with — the 64-row fallback grid).

### Round 3 (reserve) — stage2 or per-case dispatch tuning only if stage1 saturates
- Re-profile after the stage1 change: if stage2 (down GEMM) becomes the new dominant cost, tune its
  tile/accum; otherwise stop. Per-case dispatch (tpr2048 vs tpr4096) may want different tile_m.

## Compounding / integration notes
- The stage1 tiling change (R1) and the occupancy VGPR trim (R2) are on the SAME kernel and may
  conflict — sequence them (R2 builds on R1's winner), do not run as parallel merge candidates.
- Two different stage1 tilings CANNOT merge (dead-end #6) — pick one by isolated A/B.
- Host-side levers do NOT compound: `_forward_fbs` bypasses `fused_moe_`, so a win on the replaced
  asm path adds 0 (dead-end #1). Keep host work out of the plan unless it lives on the FlyDSL path.

## perf_knowledge levers (reference hypotheses to measure, not decisions)
- `fused_moe_grouped_gemm` + flydsl: preshuffled-B / native-fp8 MFMA src layout, nonkdim choice,
  XCD de-interleave for L2 reuse, per-128-K fp32 scale promotion in-loop (numerics.md). The gfx950
  32x32-MFMA-remap and e2m1-weight cards are architecture-adjacent priors (NOT portable verbatim to
  gfx942) — use only to seed the issue-efficiency direction, decide by on-box A/B.

## Dead-ends (do not re-fund) — see codebase_context.md for the full list
Host tax strip (inert), stage2 atomic->plain-store (~0), gemm1 lean/scale/VGPR retune (already live;
schedule knobs regress), host 128->64 moe_sorting (slower s2t), epilogue LDS stride pad (inherent
2-way conflict), conflicting alternate stage1 tiling, gfx950-only blkmap skill, fp16 weight dequant
(memory-bound), CK 2-stage (slower), asm block_m 16/128 (crash), ksplit (no effect on 1-stage).
