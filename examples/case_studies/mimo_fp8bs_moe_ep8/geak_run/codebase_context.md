# Codebase context — mimo_fp8bs_moe_ep8 (WARM-START continuation)

## Operator
Rank-local routed-expert MoE of MiMo-V2.6-Flash-MOPD under EP8 (SGLang MoriEPMoE + MORI fp8 dispatch).
- hidden 4096, moe_intermediate 2048, 256 experts (32 local per rank), top-8, SiLU gate/up.
- W8A8 fp8 block-scale: weights 128x128 blocks, activations 1x128 groups. bf16 output.
- Receive buffer: 8*4096 = 32768 rows; only `num_local_tokens` rows valid (rest stale).
- Two timed cases: tokens_per_rank 2048 and 4096.

## Entry point (editable)
`moe_ep.py::forward(weights, x, x_scale, topk_weights, topk_ids, expert_mask, num_local_tokens)`.
- Baseline forwards to `aiter_fused_moe.py::fused_moe(...)` with `quant_type=per_128x128`,
  `activation=Silu`, which on gfx942 dispatches the prebuilt asm `fmoe_fp8_blockscale_g1u1` (.co, no source).
- `prepare_weights` (untimed) shuffles w13/w2 with `shuffle_weight(.,(16,16))`.

## Editable files
- `moe_ep.py` (thin wrapper, ~48 lines) — routes forward dispatch.
- `aiter_fused_moe.py` (verbatim 63KB copy of aiter/fused_moe.py) — editable.
- `flydsl_moe_fp8bs_s1v2.py` — the WARM-START FlyDSL two-stage kernel module (seeded from KB `best/`;
  has a SyntaxError at line 159 that must be repaired — see below).
- New files in the task dir. NEVER edit `scripts/task_runner.py` or installed aiter/sglang.

## Constraints
- Never edit `scripts/task_runner.py` or installed aiter/sglang sources.
- Keep native fp8 (in-loop per-128-K fp32 scale promotion). Do NOT dequant weights to fp16.

## Current state (prior committed result on THIS box)
- COMMITTED winner = **1.5397x geomean**: tpr2048 1.1178 ms, tpr4096 2.0447 ms
  (vs baseline 1.6807 / 3.2239 ms).
- Mechanism: FlyDSL native-fp8 two-stage MoE REPLACES the aiter asm fmoe path.
  - stage1 = gate/up GEMM + SiLU/requant (`moe_fp8bs_gemm1_0`); stage2 = down GEMM + bf16 accum
    (`moe_s2_fp8bs_0`); per-case dispatch; XCD-aware WG remap.
  - Win is occupancy: asm path ran 1 wave/SIMD; FlyDSL 2-stage reaches 2 waves/SIMD
    (112 arch + 128 accum = 240 VGPR on 512-wide gfx942 unified file). MFMA util ~33% -> ~60%.
  - Round-1 algorithm replacement carried ~1.49x of it; rounds 2-3 added <1% (run-to-run noise).

## Ceiling reality (do NOT chase nameplate)
- 750W power cap throttles sclk to ~1.2 GHz -> realized fp8 TFLOPs ~19-20% of 2615T nameplate.
- HBM <20% of peak: NOT memory-bound. This is compute/issue-bound.
- Realistic FlyDSL roofline here: ~45-60% of peak = ~0.6-0.7 ms (tpr2048) / ~1.2-1.4 ms (tpr4096).
- Occupancy lever: kernel runs 2 waves/SIMD @ ~240 VGPR; a 3rd wave needs <=170 VGPR.

## HIGHEST-VALUE UNTRIED LEVER (do first)
`flydsl_moe_fp8bs_s1v2.py:159` has a SyntaxError: `compile_stage1_t`'s signature ends with
`xcd_n: int = int(_os.environ.get("FBS_XCD_N", "8")` — missing the closing `)` of `int(...)`.
Because the module fails to import (wrapped in try/except), stage1 SILENTLY falls back to the
64-row gemm1 tile, so the real **128-row transposed 'T' stage1 HAS NEVER EXECUTED**. In the T stage1,
weights are MFMA src-A and tokens are src-B, one token per lane, so the activation block-scale is a
per-lane scalar; it halves the weight/B stream per FLOP vs the 64-row gemm1 tile (bigger M tile shares
B through LDS/registers -> fewer B requests per FLOP). This is the remaining headroom round 3 named.
Repair: add the one missing paren, make s1v2 import and build cleanly at tile_m=128 on gfx942
(watch LDS ~263KB and VGPR), then isolated A/B of s1v2 128-row + XCD swizzle vs the gemm1 fallback,
per case, against the frozen baseline. Confirm ENGAGEMENT by the live grid, not the headline number.

## PROVEN DEAD-ENDS (do NOT re-run)
1. Host pre-kernel tax strip / valid-row config under fbs winner — INERT (`_forward_fbs` never calls
   `fused_moe_`); 1.0469x, does not compound.
2. stage2 non-atomic + unpadded epilogue — atomic->plain-store ~0; epilogue-skip ceiling only ~2% e2e.
3. stage1 gemm1 lean fold + hoisted scales + freed-VGPR retune — `_SX_TAB` fold, `_TR` scale hoist,
   lean SiLU ALREADY live; MF32/SETPRIO schedule knobs regress.
4. host 64-unit moe_sorting — already runs unit=64; 128->64 routes slower s2t stage2 (1.5285 < bar).
5. epilogue LDS row-stride pad off 32-bank period — inherent stride-independent 2-way bank conflict
   on the full-row read.
6. competing stage1 with alternate tiling that conflicts with the committed stage1.
7. expert skill `flydsl_decode_moe_stage1_blkmap` is gfx950-ONLY; will not build at 128-row on gfx942.
8. fp16-dequant FlyDSL seed ran ~11/13.5 ms (memory-bound) — stay native fp8.
9. CK 2-stage blkscale slower (1.86/3.37 ms); asm block_m 16/128 crash; ksplit no effect on 1-stage.

## perf_knowledge pointer
- operator `fused_moe_grouped_gemm`, language `flydsl`.
- refs: backends/flydsl.md, tuning.md, numerics.md; languages/flydsl/{authoring_gemm_levers,
  authoring_tile_programming, debugging}.md.

## Expert skills screen
- `flydsl_decode_moe_stage1_blkmap`: matches operator but gens=gfx950 only -> does NOT match gfx942
  (dead-end #7). No validated gfx942 kernel-scope MoE authoring skill applies.
