# TechLead Report — mimo_fp8bs_moe_ep8

Rank-local routed-expert MoE of MiMo-V2.6-Flash-MOPD under EP8 (SGLang MoriEPMoE + MORI fp8
dispatch). hidden 4096, moe_intermediate 2048, 256 experts (32 local), top-8, SiLU gate/up,
W8A8 fp8 block-scale (128x128 weights / 1x128 activations), bf16 out.

## Summary
- **Kernel type:** e2e wrapper over a prebuilt vendor asm grouped-GEMM (aiter `fused_moe`); the timed
  forward calls the gfx942 prebuilt `.co` asm `fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256`.
- **Device:** AMD Instinct MI300X / gfx942 / CDNA3, 304 CU, fp8 = FNUZ.
- **Target language requested:** flydsl (new from-scratch block-scale grouped-GEMM).
- **Final committed speedup (geomean): 1.000x** — no patch cleared the commit gate; the committed
  workspace is the baseline. (FINAL_WINNER = null, cumulative = 1.0.)
- **Best VERIFIED-in-isolation result: ~1.016x geomean** (tpr2048 ~1.00x neutral, tpr4096 ~1.03x),
  not banked — the gain was confined to one case and sat below the commit threshold.
- **Rounds:** 2. **Budget used:** 3 of 8 directions (DEEP round costs 2).
- Metric is unweighted (`workload_aligned=false`); geomean is the headline.

## Bottleneck (baseline profile)
COMPUTE-BOUND on a single kernel. The grouped fp8 block-scale MFMA GEMM is ~96% of timed forward;
MoE sorting + `partial_transpose` ~84 us/call ~3%. HBM is only 6-11% utilized — memory bandwidth is
NOT the lever. The asm GEMM runs at only ~19-20% of fp8 MFMA peak, i.e. ~5x below the compute
roofline (tpr2048 1.56ms vs 0.317ms roofline; tpr4096 3.35ms vs 0.634ms). That ~5x headroom lives
INSIDE the closed `.co` asm and is reachable ONLY by a new kernel — no config/tuning lever can touch
it (confirmed: best config moved roofline% only ~20%->~20.3% on tpr4096).

## Round-by-round

### Round 1 (2 directions)
- **r1_d0 — algorithm — FlyDSL fp8 block-scale 2-stage fused-MoE kernel (roofline push).**
  Result: **FAILED (1.0x, apply_failed).** Blocked by the installed aiter build: the
  `moe_blockscale_2stage` primitive the expert skill/operator card points at is MISSING; the one std
  `moe_gemm_2stage` kernel models fp8 scales as per-token/per-channel SCALARS applied outside the
  K-reduction (cannot represent 128x128-weight / 1x128-activation block-scale that varies with
  `k//128` inside the sum) AND does not compile for any non-fp4 dtype due to an undefined bare `i32`
  (should be `T.i32`) at `moe_gemm_2stage.py:1079` (confirmed via isolated bf16 + f16 probe). Only the
  fp4 `mixed_moe_gemm_2stage` path compiles. No regression — asm baseline retained.
- **r1_d1 — compute — Valid-row-aware per-shape config + asm/CK tuning.**
  Result: **SUCCESS, verified 1.01503x.** Keys `get_2stage_cfgs` on the TRUE `num_local_tokens`
  (not the padded M=32768 buffer, which collapses both cases into one bucket) and sets `block_m=64`
  for the large tpr4096 bucket, 32 for tpr2048. `num_local_tokens` read via a `data_ptr`-keyed cache
  so the host sync happens ~once per distinct buffer (a naive `.item()` per forward costs ~0.11ms and
  alone regresses tpr2048). tpr2048 0.9998x (neutral), tpr4096 1.0301x.
- **Integrate:** none (single improving direction). **Winner:** r1_d1, 1.01503x.
  **Committed:** NO — `improved=false`, cumulative stayed 1.0. The ~1.5% geomean came entirely from
  tpr4096; tpr2048 was flat, so the net did not clear the commit gate.

### Round 2 (1 direction — dedicated DEEP round, cost 2)
- **r2_d0 — deep_explore — From-scratch FlyDSL fp8 block-scale grouped-GEMM, 3x / ~90%-roofline
  mandate.** Result: **PARTIAL, verified 1.0159x.** The DEEP round could not build the from-scratch
  kernel within one round, so it re-landed exactly the known r1_d1 lever (`block_m=64` for tpr4096,
  `data_ptr`-cached `num_local_tokens`). tpr2048 1.0017x, tpr4096 1.0305x, correctness PASS. No net
  new gain over round 1.
  - Net learning: the FlyDSL AUTHORING API IS available on this box (round-1 "blocked" was too strong
    — the `mixed_moe_gemm_2stage` fp4 path, 4546 lines, compiles and the tile-programming API is
    usable). The from-scratch block-scale grouped GEMM is authorable, just a multi-round build, not a
    one-round win.
- **Integrate:** none. **Winner:** r2_d0, 1.0159x. **Committed:** NO — `improved=false`, cumulative
  stayed 1.0 (same reason: one-case gain below the gate).

## Final per-test-case table (committed state = baseline)
| case    | baseline ms | optimized ms | speedup |
|---------|-------------|--------------|---------|
| tpr2048 | 1.6737      | 1.6737       | 1.000   |
| tpr4096 | 3.2115      | 3.2115       | 1.000   |
| **geomean** | **2.3184** | **2.3184** | **1.000** |
| arithmetic  |            |            | 1.000   |

Best isolated-but-unbanked candidate (for reference, not committed): tpr2048 ~1.001x,
tpr4096 ~1.030x, geomean ~1.016x. Patch at
`round_2/engineer_0/best_patch.diff` (equivalently `round_1/engineer_1/best_patch.diff`).

## Key optimizations applied
None banked. The one correct, non-regressing lever found — valid-row-aware per-shape `block_m`
selection (`block_m=64` on tpr4096, keyed on `data_ptr`-cached `num_local_tokens`) — verified
~1.016x in isolation but did not clear the commit gate because the entire gain sits on one case
while the other is neutral. The committed workspace is unchanged from baseline.

## What didn't work (dead-ends)
- **New FlyDSL block-scale grouped GEMM (the user's explicit mandate):** not deliverable within the
  budget spent. The std `moe_gemm_2stage` path is numerically unable to represent 128x128/1x128
  block-scale (scalar scales outside the K-sum) and is uncompilable for all non-fp4 dtypes (`i32`
  bug at line 1079); the `moe_blockscale_2stage` primitive is absent. The authorable route is a
  from-scratch build on the fp4 `mixed_moe_gemm_2stage` skeleton (swap per-1x32 e8m0 dequant for an
  fp32 128x128-weight/1x128-activation factor per 128-wide K sub-block, fp32 accumulate, E4M3FNUZ,
  fused SiLU+requant stage1->stage2 reduce, untimed `prepare_weights` preshuffle) — a genuine
  multi-round effort that one DEEP round could not land.
- **asm config lever is EXHAUSTED at ~1.016x** — the whole geomean is `block_m=64` on tpr4096 alone;
  tpr2048 has no headroom (`block_m=32` already optimal).

<!-- dead-ends:yaml -->
```yaml
- idea: FlyDSL std moe_gemm_2stage fp8 block-scale path
  measured: fails-to-compile / oracle-fail
  mechanism: scalar per-token/per-channel scales cannot represent 128x128w/1x128a block-scale inside the K-sum; undefined bare i32 (should be T.i32) at moe_gemm_2stage.py:1079 makes all non-fp4 dtypes uncompilable; moe_blockscale_2stage primitive absent
- idea: 2-stage CK path (run_1stage=False)
  measured: 1.86/3.37 and 1.92/3.59 ms (slower both cases)
  mechanism: asm 1-stage faster than CK 2-stage on gfx942; the "CK-better" comment does not hold here
- idea: asm block_m 16 or 128
  measured: crash
  mechanism: prebuilt asm kernel only has 32/64 M-tile variants
- idea: ksplit on 1-stage asm
  measured: no effect
  mechanism: the 1-stage asm path takes no ksplit
- idea: non-temporal load toggle
  measured: within noise
  mechanism: not load-bearing on this compute-bound path
- idea: naive num_local_tokens .item() per forward
  measured: regresses tpr2048 below baseline (~0.11ms/call)
  mechanism: host sync serializes host launch with GPU compute; must cache by data_ptr
- idea: bf16-dequant FlyDSL route
  measured: not pursued (analysis)
  mechanism: halves MFMA peak vs fp8, needs out-of-lane weight prep, hits the same i32 bug
- idea: block_m=64 valid-row-aware config (the only paying lever)
  measured: 1.016x isolated, not banked
  mechanism: gain confined to tpr4096 (~1.03x), tpr2048 neutral; below the commit gate so cumulative stayed 1.0
```

## Recommendation
The only path to the user's "new flydsl kernel at compute roofline" mandate is a from-scratch,
multi-round FlyDSL block-scale grouped GEMM built on the fp4 `mixed_moe_gemm_2stage` skeleton using
the de-risked recipe above, with a correctness-first milestone before any perf claim. The asm
config lever is exhausted; no further gain is available from tuning. With the budget spent this run,
no net improvement was committable.
