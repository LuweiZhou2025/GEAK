# Insight log — mimo_fp8bs_moe_ep8 (warm-start, bar = 1.5397x)

## Round 1 — update_memory

**Outcome:** no improvement. The committed best is still the warm-start FlyDSL 2-stage path
(`moe_fp8bs_gemm1_0` + `moe_s2_fp8bs_0`, about 1.535-1.540x in this round's own base A/Bs). All 3 directions
ended `apply_failed`. No engineer returned a result or left a `best_patch.diff` within the about 60 min
round (10:55 to about 11:57), so verify had nothing to apply. **Every number below was measured by an
engineer and recovered from on-disk logs, PMC CSVs, ISA dumps and code comments. None of it has been
re-measured by the verifier.**

### Per-direction (engineer-measured, unverified)
| id | specialty | strategy | expected | engineer-measured | verdict |
|---|---|---|---|---|---|
| r1_d0 | memory | Engage 128-row transposed stage1 (s1v2/s1t) | 1.08 | forward 0.976x / 0.994x vs live gemm1 (about 0.985 geomean) | dead_end |
| r1_d1 | host_runtime | Delete the P23 zero-fill, right-size the gemm1 grid | 1.035 | zf3+fast-sort +1.31% / +0.63% (about 1.010), in-process | partial |
| r1_d1 sub | host_runtime | gemm1 on-device grid-stride loop (`gl`) | — | 0.955 (-4.5%, harness n=3) | dead_end |
| r1_d1 sub | host_runtime | moe_sorting dispatch_policy=1 / opus sorting | — | dp1 0.35 (fwd 3.13/6.15 ms); opus about 1.00 | dead_end |
| r1_d2 | compute | Stage2 to 3 WG/CU | 1.04 | untimed; 3rd wave not reachable without spills | dead_end |

### Insight blackboard (durable)
1. **Process.** All three engineers ran out the round without returning a result or writing `best_patch.diff`.
   The evidence sat only in scratch logs, and env-gated knobs (default OFF) are invisible to verify.
   Next round, every engineer must checkpoint a `best_patch.diff` with the winning knobs default-ON, plus a
   one-line measured verdict, as soon as any variant beats the bar.
2. **CLOSED: the 128-row transposed stage1 (s1v2/s1t), the roadmap's #1 lever.** The SyntaxError was repaired
   and the kernel engaged (`moe_fp8bs_s1t_0`: 256 VGPR / 0 AGPR, 32 KB LDS, 2 waves/SIMD). The best tune
   (xcd_swz=3, xcd_gn=4, sched=0, halfskip=0, mloop=1, eid_out=1) only ties gemm1 in stage1 (0.99x/1.01x).
   Unit-128 sorting adds about 2-4% padded MFMA work and about 15 us of stage2, so the forward lands at
   0.976x/0.994x. The halved B-stream per FLOP is cancelled because each XCD's working set doubles
   (L2 hit 69% vs 77% at the same mapping). pair64 (+30% L2 misses) and halfskip (+17% L2 misses, 6-15 VGPR
   epilogue spills) are worse.
3. **TRAP in canonical `moe_ep.py`.** `MOE_S1_IMPL` defaults to `"v2"`, but `flydsl_moe_fp8bs_s1v2.py:159` has a
   SyntaxError, so gemm1 runs silently. A patch that fixes the syntax must also set the default to gemm1, or
   the forward silently drops about 1.5%. Confirm stage1 engagement by kernel name in the trace.
4. **Bottleneck refined: the clock under the power cap, not MFMA issue.** PMC MFMA-busy is about 64% on gemm1 and
   about 49-54% on stage2, at sclk about 1.16 GHz under the 750 W cap. A timing-only what-if (B and X forced
   L2-hot) cut stage1 by 18% and raised sclk from about 1160 to about 1370 MHz. L2-miss (MALL/HBM) energy is
   what costs clock. Cutting L2 misses is therefore a compute lever here, and adding HBM traffic inside a GEMM
   slows the GEMM. gemm1 waves spend about 58% of cycles in SQ_WAIT_INST_ANY.
5. **XCD mapping is untested on the live kernels.** gemm1 and stage2 launch a plain (16 N tiles, M) 2-D grid.
   Hardware round-robin pins N tiles {x, x+8} to XCD x, so all 8 XCDs fetch every M block's activation rows.
   On s1t, XCD tiling (each XCD owns 4 N tiles of one M partition, so X is fetched by 4 XCDs instead of 8)
   cut L2 misses by 21%. The N-major swizzle (the warm-start "XCD remap", xcd_swz=1) was the worst at 47%
   hit, and M-partition (xcd_swz=2) was also worse. Live gemm1 is at 77-80% L2 hit (13.4M misses); stage2 is
   at 68-71% hit.
6. **Codegen hazard.** Wrapping gemm1's body in an on-device grid-stride `scf.for` (same 544 M-slot bound)
   lost 4.5%, while capping the grid at the same bound without a loop was neutral. The loop is what costs.
   Empty CTAs (about 88-94% of gemm1's 69,632) are effectively free, so right-sizing the grid gains 0%.
7. **Cost structure** (rocprofv3 medians). The forward is GPU-bound: span equals the kernel sum, 6 kernels per
   forward, no launch gaps, so graph capture would gain about 0.
   tpr2048 = sort 67.7 + gemm1 680.5 + s2 398.8 us (span 1151);
   tpr4096 = sort 110.8 + gemm1 1233.1 + s2 713.5 us (span 2060).
   Sorting is 5.4-5.9% of the forward. Its P23 moe_buf zero-fill (89 / 177 MB of bf16 writes) accounts for
   about 16 / 33 us of that.
8. **Sort / zero-fill lever (about +1%, not shipped, recoverable).** A bit-identical 2-kernel align&sort
   (`k_count` + `k_place`, `moe_sort_fast.py`) costs 30-32 us per forward, replacing the 4-kernel CK chain.
   Putting the fill in gemm1 right after the prologue (zf_pos=1) adds 62 us to gemm1 at tpr4096, a net of
   about 0. Spreading one masked 16-B zero store per K tile (zf_pos=3) plus the fast sort nets -14.4 / -12.9 us
   (+1.31% / +0.63%). That comes from an in-process interleaved run, n=4, with the profiler on.
   Deleting the fill outright (incorrect, diagnostic only) would gain at most +1.1%.
   Code: `round_1/engineer_1/ws_dev2/workspace/{moe_sort_fast.py, flydsl_moe_fp8bs.py, moe_ep.py}`.
9. **Sort dead-ends.** aiter `dispatch_policy=1` (a single CK MoeSortingKernel) takes 1.7-3.9 ms, making the
   forward 3.13 / 6.15 ms (about 2.8-3.0x slower). `opus_moe_sorting` performs the same as CK.
10. **Stage2 3-wave wall.** The v3 s2 loop (running-accumulator rescale, 2-stage A ring, B per 64-K half-step)
    reached 21,504 B LDS, which allows 3 WG/CU by LDS. But VGPR floored at 187 / 205 / 221 (b_depth 2/3/4),
    which still means 2 waves/SIMD. Forcing waves_per_eu=3 (168 VGPR) spills 19-33 VGPRs.
    m2 (8-wave CTA) uses 243 VGPR + 43 KB LDS. tile_n=128 uses 512 VGPR with 349 spills. No timing was
    captured. Code: `round_1/engineer_2/workspace/flydsl_moe_fp8bs_s2.py` (v3=0 default).
11. **Measurement noise.** Harness outliers reach about 5% on the same config (1.1106 vs 1.1786 ms). Repeated base
    A/Bs this round gave 1.5332-1.5385x. Use at least 3 interleaved rounds and take the median. An in-process
    interleaved A/B (`inproc_ab2.py`) resolves effects of about 1%.
12. **Expert skills.** `flydsl_rewrite_quantized_moe` (kernel, validated, gfx942, fp8_e4m3_fnuz) matches, but it is a
    triton->flydsl recipe. Its fp8-blockscale primitive (`moe_blockscale_2stage`) is not installed on this box,
    and its `[M, hidden]` atomic-stage2 memory contract is already met. There is nothing to reproduce.
    The blkmap and stage2_fp8partial skills are gfx950-only.

### Bottleneck now
Compute, limited by the power-capped clock. MFMA-busy is about 64% on gemm1 and about 50% on stage2, at about
1.16 GHz. L2-miss HBM/MALL energy is what limits the clock. Both GEMMs run at 2 waves/SIMD.

### Suggested next (round 2)
- **memory:** port s1t's XCD tiling to the LIVE gemm1 and stage2. Use a block-id remap with an if-guard and no
  device loop.
- **host_runtime:** ship the `ws_dev2` fast sort with the zf_pos=3 fill (about +1%).
- **compute:** stage2 has the most MFMA-busy headroom (about 50%). Amortize its short-K per-CTA overhead
  without a dynamic loop.
- **Do NOT re-fund:** s1v2 / 128-row stage1, the gemm1 grid-loop, sort dispatch_policy=1, or the s2 3rd wave
  via VGPR trim.

---

## Round 2 — update_memory

**Outcome: improved.** The winner is the integrated r2_d2 + r2_d1 patch (commit `f3f9493`, correctness PASS, rel_l2
2.315e-3 / 2.307e-3).
- Integrate measured **1.5710x** geomean (n=10 interleaved ABBA rounds; 1.0968 / 1.9880 ms).
- The reprofile's FULL_BENCHMARK x3 measured **1.5717x** (1.0942 / 1.9910 ms).
- That is +2.8% over R1's 1.5279x and +2.0% over the warm-start bar of 1.5397x.

2 of 3 directions verified (R1: 0 of 3). r2_d0 closed as a measured dead end and left a reference diff.

### Per-direction
| id | specialty | strategy | expected | same-session measured | verified | verdict |
|---|---|---|---|---|---|---|
| r2_d0 | memory | XCD-aware CTA remap of the live gemm1 | 1.03 | gn=4: 0.9988x. Expert-run order: 0.9863x | no patch (by design) | dead_end |
| r2_d1 | host_runtime | Fast 2-kernel sort default-ON, with a sort-owned zero-fill | 1.012 | 1.5331 -> 1.5431x (+0.65%, 5/5 rounds) | 1.5535 | confirmed (shipped) |
| r2_d2 | compute | s2 setprio + fpre + epi3 + 1-VALU sched slot. De-loop and XCD tiling measured negative | 1.02 | 1.5355 -> 1.5634x (+1.82%, 5/5 rounds) | 1.5629 | confirmed (shipped) |
| integrate | - | r2_d2 + r2_d1 (disjoint files) plus the k_place stride fix | ~1.025 | +0.61% / +0.70% over r2_d2 alone (9/10 and 10/10 wins) | 1.5710 | confirmed (winner) |

Sub-experiments closed this round:
- gemm1 expert-run order: 0.9863x.
- s2 de-loop: s2 -3.6% / -1.4%.
- s2 XCD tiling: s2 -0.9% to -12.9%.
- Fill placement and tuning variants: the best one ties; the side-stream fill loses 0.3% to 6%.

### Insight blackboard (supersedes the R1 list; the R1 items that still hold are carried in item 14)
1. **State after R2.** The live path is:
   - the fast sort: `k_count` + `k_place`, which also does the moe_buf zero-fill;
   - `moe_fp8bs_gemm1_0`, **unchanged since the warm start**;
   - `moe_s2_fp8bs_0` with setprio, fpre, epi3 and the 1-VALU sched slot.

   The forward is 4 dispatches (was 6), and the span equals the kernel sum, so there are no host gaps.
2. **Cost structure** (steady-state trace, 110 forwards, medians):

   | case | k_count | k_place | gemm1 | s2 | span |
   |---|---|---|---|---|---|
   | tpr2048 | 6.9 us | 35.4 us | 683.4 us | 363.2 us | 1089.0 us |
   | tpr4096 | 6.8 us | 47.7 us | 1267.3 us | 650.7 us | 1974.1 us |

   Shares of the forward: gemm1 **62.7% / 64.2%**, s2 33.4% / 33.0%, sort 3.9% / 2.8%. A 1% forward gain now needs
   about -1.6% of gemm1 or about -3% of s2.
3. **Power-cap economics (the key R2 lesson).** Socket power holds at the 750 W cap, with sclk around 1.07 GHz
   (51% of 2.1 GHz).
   - Only reductions in work or energy turn into time. r2_d2 cut s2 cycles per XCD by 13.5%, but time fell only
     about 5%, because sclk dropped as the work got denser.
   - Removing non-MFMA VALU pays twice. fpre plus epi3 raised MFMA-busy and clock together: 53.5% at 0.97 GHz
     became 61.8% at 1.04 GHz.
   - So rank levers by work or bytes removed (VALU ops, padding MFMAs, written bytes), not by latency hidden or
     misses moved.
4. **L2-miss steer corrected.** R1's B+X L2-hot what-if (-18% of stage1) is almost all W.
   - On gemm1, W-hot alone saves the same ~88 us in both cases. That is the compulsory 512 MB W13 HBM read,
     which cannot be reduced at tile_m=64.
   - X-hot alone is worth only 3.2% / 5.5% of stage1, because replicated-X misses are mostly cheap MALL hits.
   - Remaps only move MALL traffic. Nowhere in R2 did fewer L2 misses mean less time.
5. **CLOSED: XCD remaps on the live kernels** (gemm1 by r2_d0, s2 by r2_d2).
   - **gemm1, plain XCD tiling (gn=4):** 0.9988x, i.e. noise. It trades X misses for W-panel refetches: the W miss
     probability rises about 2.6% per position of dispatch distance, and net misses fall at best 1% / 4.6%.
   - **gemm1, pure M-partition (gn=16):** loses 4.9% / 10.9% of stage1.
   - **gemm1, expert-run order:** cuts misses 25% at tpr2048 but measures 0.9863x. Its dependent seid vector
     load + ballot + division prologue costs 2-3.5% of stage1.
   - **s2, XCD tiling** (s2 time change):

     | gn | tpr2048 | tpr4096 |
     |---|---|---|
     | 2 | -1.4% | -0.9% |
     | 4 | -2.7% | -0.4% |
     | 8 | -7.2% | -3.6% |
     | 16 | -11.5% | -12.9% |

   - s2 needs the plain (16 N, M) grid. It keeps each output column range's atomics on one XCD and keeps the CTAs
     that share a w2 panel co-resident.
   - The only untested variant is a critical-path-free remap: a sort-emitted (m,n) table read via s_load, combined
     with the expert-run order. Its upside is at most about +1% of stage1, unmeasured.
6. **The s2 recipe that worked** (r2_d2, all default-ON; cumulative same-session COMMANDMENT verdicts):

   | step | cumulative gain | notes |
   |---|---|---|
   | setprio(1) around the K loop | +0.63% | |
   | fpre | +1.40% | a2_scale*w2_scale table per 128-col block in LDS (28 -> 32 KB); rescale becomes a bare FMA |
   | epi_mode 3 | +1.27% (GPU1); in-process +1.29% / +2.00% | routing weight folded into the f-table, v_perm_b32 bf16 packing, branchless atomics; epilogue VALU ~426 -> ~240 per wave |
   | sched_group_barrier at 1 VALU per MFMA slot (was 2) | +1.71% | |
   | final | **+1.82%** | |

   **fpre requires waves_per_eu=2.** Without it the kernel needs 272 VGPR, drops to 1 wave/SIMD and loses 16%.
7. **gemm1 has not received any of this.** In the committed `flydsl_moe_fp8bs.py`:
   - Knobs: `FBS_SETPRIO` defaults to "0" (line 61); `FBS_SCHED`/`SCHED2` are 3; `SCHED_G1`/`G2` are 8;
     `FBS_PK_SPLIT=1` packs only the main rescale.
   - Body: 2048 MFMA, 753 `v_pk_fma_f32` and **542 scalar `v_fma_f32`** (0.98 non-MFMA VALU per MFMA), 436
     `s_waitcnt`, about 45 KB of fully unrolled code.
   - Resources: 240 VGPR + 32 KB LDS (16 KB X ping-pong + 16 KB scale tables). That is 2 waves/SIMD on both limits,
     so any added table or VGPR must be paid for.
   - `FBS_VALID_TOK` only bounds the X buffer. It does not skip any MFMA work.
8. **s2 dead ends and ceilings** (r2_d2).
   - Dead ends: soffset addressing; vm_mode 1/2/3; VALU start slot 0/4; 4 VALU per slot; max-ilp (noise);
     iterative-ilp (-2% / -0.4%); the igrouplp exact solver (compile hang); sc0 (+-0.2%); B prefetch 2 K-tiles
     ahead (-5.7% / -4% at 2 ku); uniform-branch atomics (521 VGPR spill).
   - Ceilings:
     - Rescale is 8.4% / 9.6% of s2. It is unavoidable on gfx942, but s2 still has 668 scalar `v_fma_f32` vs 178
       `v_pk_fma_f32` per 1024 MFMA, which is room to pack.
     - Atomics are 3.6% / 4.2% of s2.
     - The tail costs only 3-4 us, so split-K is not worth it.
9. **Grid/loop asymmetry.**
   - gemm1 must not get a device loop: -4.5% in R1, and its empty CTAs are free.
   - s2 is the opposite. Replacing its existing degenerate loop with a full if-guarded grid (61k empty CTAs) loses
     -3.6% / -1.4% of the final s2.
   - Keep each kernel's grid/loop structure as it is.
10. **Sort + fill shipped (r2_d1).**
    - The bit-identical `k_count` + `k_place` sort runs with 304 sort-owned zero-fill blocks.
      Defaults: `MOE_FAST_SORT=1`, `MOE_ZF_SORT_BLOCKS=304`, with a CK fallback.
    - Same-session gain is +0.48% / +0.79%, and +0.61% / +0.70% again on top of r2_d2.
    - In the forward, sort + fill went from 67.7 / 110.8 us to 42.3 / 54.5 us. Only about 50% of the micro-bench
      savings reach the forward.
11. **The fill's cost does not depend on where it runs.** It costs about 11 / 27 us per forward wherever it is
    placed, and it tracks the bytes written (89 / 177 MB of HBM writes under the cap).
    - Measured dead ends:
      - side stream: -0.3% to -6%;
      - serial before s2: ties;
      - `hipExtAnyOrderLaunch`: ignored on gfx942;
      - single-kernel grid-barrier sort: 34 / 50 us;
      - zsplit: no gain;
      - 176 blocks: forward-neutral;
      - NT stores: did not replicate.
    - The remaining +1.0% / +1.4% needs **fewer bytes written**. 60% of received tokens have exactly one local
      expert (6505 / 10841 and 13037 / 21575). s2 could plain-store those rows and k_place would only zero
      multi-expert rows, for an estimated 1-1.5%.
12. **Padding work** (both stages pay it).
    - gemm1 executes 17,408 / 33,600 MFMA rows for 16,415 / 32,536 routed pairs.
    - **3.8-4.2% / 2.2-2.3% of MFMA rows are whole all-padding 16-row subtiles.** The padding is a suffix of each
      expert's last block, so the skip decision is wave-uniform.
    - MFMA ignores EXEC, so masking saves nothing. A skip needs a uniform branch, a tail path or a tail launch.
    - Avoid a dependent-load or ballot prologue and avoid a device loop. Watch the I-cache (the unrolled body is
      about 45 KB).
    - Estimate: about +3% geomean if done in both GEMMs. Price it first with a deletion control.
13. **Profile** (R2 PMC from six rocprofv3 `--pmc` passes, because rocprof-compute still fails its dependency
    check).

    | metric | gemm1 | s2 |
    |---|---|---|
    | MFMA busy | 61.0 / 65.7% | 58.0 / 65.4% |
    | waves resident (of a 2-wave ceiling) | 1.79 / 1.88 | 1.76 / 1.85 |
    | L2 hit | 77 / 81% | 71 / 77% |
    | LDS bank conflicts | 12.7% | 0% |

    - Wave cycles split into about 23-28% active, 53-57% issue-wait and 18-21% dep-wait. The issue-wait is two
      waves contending for one MFMA pipe.
    - FP8 throughput is 743-908 TFLOPS, 28-35% of nameplate.
    - EA bandwidth is about 30% of peak.
    - Not memory-, LDS- or launch-bound. Stage1->stage2 fusion has a ceiling under 3% of EA traffic.
14. **Still CLOSED from R1:**
    - 128-row s1v2/s1t stage1 (0.976x / 0.994x);
    - gemm1 device loop (-4.5%);
    - sort dispatch_policy=1 (about 0.35x), and opus sorting (equal to CK);
    - a 3rd s2 wave via VGPR trim.

    The `MOE_S1_IMPL` trap is **defused**: the default is now "gemm1" (`moe_ep.py:30`).
15. **Verify and integrate hazards.**
    - `git apply` inside the GEAK subtree silently skips paths. Use `patch -p1` or a standalone repo.
    - The k_place padding loop must stride by `sort_grid`, not `gridDim.x`. With `gridDim.x` it corrupts the sort
      and faults at M < 8192. This is fixed and committed.
    - Role-split grids need small-M tests (M = 256..32768).
16. **Measurement.** Cross-session drift is about 1% (canonical measured 1.5281-1.5408x within R2), so trust only
    same-session paired ratios.
    - At least 5 interleaved COMMANDMENT rounds on one GPU, with medians and win counts, resolve +0.5% effects.
    - In-process paired A/B is the tool for sub-1% knobs: r2_d1 `tools/inproc_final.py`, r2_d2 `tools/ab.py`.
17. **Process.** Early checkpointing worked in R2: `best_patch.diff` with winners default-ON, plus a measured
    verdict per step. Keep requiring it. A dead-end direction should leave a NOT_A_CANDIDATE diff plus its tables.
18. **Expert skills.** Re-checked `index.yaml`. The only kernel-scope validated match is
    `flydsl_rewrite_quantized_moe`, a triton->flydsl recipe.
    - This run is already FlyDSL-native.
    - `moe_blockscale_2stage` is absent; there is no `/opt/flydsl` checkout either.
    - The skill's memory contract is already met.

    Nothing to reproduce. The other MoE/GEMM skills are gfx950-only.

### Hypothesis ledger (cumulative; "actual" = same-session ratio vs canonical unless noted)
| direction | specialty | expected | actual | verdict | lesson |
|---|---|---|---|---|---|
| r1_d0 | memory | 1.08 | 0.985 | dead_end | The 128-row s1t ties gemm1; unit-128 padding costs more. Never re-fund. MOE_S1_IMPL now defaults to gemm1 |
| r1_d1 | host_runtime | 1.035 | 1.01 | partial | Fast sort + fill; R2 shipped it as r2_d1 (+0.65%). The in-gemm1 zf_pos=3 fill was not pursued |
| r1_d1:gemm1_grid_loop | host_runtime | 1.035 | 0.955 | dead_end | Never put a device loop in gemm1 (s2 is the opposite case) |
| r1_d1:sort_dispatch_policy | host_runtime | 1.035 | 0.35 | dead_end | Single CK MoeSortingKernel takes 1.7-3.9 ms; opus = CK |
| r1_d2 | compute | 1.04 | n/a | dead_end | 3rd s2 wave not reachable by VGPR trim; R2 won at 2 waves by removing work |
| r2_d0 | memory | 1.03 | 0.9988 | dead_end | Plain XCD tiling trades X misses for W refetch; X-hot ceiling 3.2% / 5.5% |
| r2_d0:expert_run_order | memory | 1.03 | 0.9863 | dead_end | -25% misses but its prologue costs 2-3.5% of stage1; retry only if the remap is free |
| r2_d1 | host_runtime | 1.012 | 1.0065 (verified 1.5535) | confirmed | Shipped. About half the expected gain; the placement lane is exhausted |
| r2_d1:fill_placement | host_runtime | 1.012 | 1.0 (best ties) | dead_end | Fill cost ~11 / 27 us is invariant to placement; only fewer bytes help |
| r2_d2 | compute | 1.02 | 1.0182 (verified 1.5629) | confirmed | Shipped. Removing work converts to time under the cap; port the recipe to gemm1 |
| r2_d2:s2_deloop | compute | 1.02 | s2 0.975 | dead_end | s2's 61k empty guarded CTAs are not free; keep its loop |
| r2_d2:s2_xcd_tiling | compute | 1.02 | s2 0.989 (gn=2, best) | dead_end | The plain grid keeps atomics per XCD; tiling only loses |
| r2_integrate | integrate | 1.025 | 1.028 cumulative (verified 1.5710) | confirmed | Disjoint-file patches add; k_place stride fix required |

### Bottleneck now
**Compute:** fp8 MFMA-bound at a power-capped clock (750 W, about 1.07 GHz).
- gemm1 is unchanged since the warm start and is 62.7% / 64.2% of the forward.
- Both GEMMs sit at the 2-wave ceiling.
- Time tracks energy per useful FLOP: non-MFMA VALU (0.98 / 1.28 per MFMA), padding MFMAs (3.8-4.2% /
  2.2-2.3%), and written bytes.

### Suggested next (round 3)
Aim at gemm1.
- **(compute, `flydsl_moe_fp8bs.py` only)** Port the s2 recipe:
  1. A/B `FBS_SETPRIO=1` first.
  2. Retune the VALU-per-MFMA sched slot.
  3. Pack the 542 scalar `v_fma_f32`.
  4. Hold 2 waves/SIMD with `waves_per_eu=2`.
- **(algorithm)** Skip the all-padding 16-row MFMA subtiles in both GEMMs (about +3%).
  - Price it first with a deletion control.
  - Use no new dependent prologue load or ballot, and no device loop.
  - Its gemm1 part collides with the compute direction's file, so assign file ownership explicitly.
- **(lower priority)** Plain-store the single-local-expert rows in s2 so k_place zero-fills only the multi-expert
  rows (about 1-1.5%), and pack s2's 668 scalar rescale FMAs (about 1%).
- **Do NOT re-fund:**
  - plain XCD remaps or the expert-run order;
  - s2 de-loop;
  - fill placement, side-stream fill, NT stores or block-count tuning;
  - 128-row stage1;
  - the gemm1 device loop;
  - sort dispatch_policy=1;
  - a 3rd wave via VGPR trim.
- **Not a stop:** measured, unported gemm1 levers remain.

---

## Round 3 — update_memory

**Outcome: improved.** The winner is the integrated r3_d0 + r3_d1 patch plus r3_d1's gemm1 pad-skip, hand-merged onto
r3_d0's gemm1. It is commit `31c4cb1`, correctness PASS (rel_l2 2.315e-3 / 2.307e-3, unchanged).
- Integrate measured **1.6273x** geomean: pooled median of 28 interleaved COMMANDMENT runs on GPUs 0-3, 1.0558 / 1.9248 ms,
  per case 1.5862x / 1.6695x.
- The reprofile's FULL_BENCHMARK x3 measured **1.6256x** (1.0603 / 1.9207 ms).
- That is +3.4% over R2's 1.5717x and +5.6% over the warm-start bar of 1.5397x.

2 of 2 directions verified (R2: 2 of 3, R1: 0 of 3).

### Per-direction
| id | specialty | strategy | expected | same-session measured | verified | verdict |
|---|---|---|---|---|---|---|
| r3_d0 | compute | gemm1: scalar-FMA rescale (`FBS_RS=1`), `waves_per_eu=2`, whole-K-loop setprio, maxnumf epilogue | 1.02 | 1.5679 -> 1.6115x (+2.78%, 5/5 rounds) | 1.6192 (verifier +2.79% vs same-session 1.5752) | confirmed (shipped) |
| r3_d1 | algorithm | sort-emitted `blk_cnt` + all-padding 16-row subtile skip in s2 (4,2,1); gemm1 skip priced only | 1.01 | 1.5693 -> 1.5805x (+0.71%, 6 rounds). Pooled 10 rounds: tpr2048 +1.29% (10/10), tpr4096 +0.38% (9/10) | 1.5814 (verifier +0.67% vs same-session 1.5708) | confirmed (shipped) |
| integrate | - | r3_d0 + r3_d1 (disjoint files) + gemm1 skip hand-merge | ~1.03 | 2-stack +0.61% over r3_d0 (22/22). Final +0.76% over r3_d0 (12/12), +0.31% over the 2-stack (24/28) | 1.6273 | confirmed (winner) |

Sub-experiments closed this round:
- gemm1 fully packed rescale: inverted. Packing is the regression on gfx942.
- gemm1 VALU-slotted `sched_group_barrier` (`FBS_SCHED=5`): noise.
- gemm1 per-tile setprio (`FBS_G1_PRIO=2`): -5.4% / -4.8% of gemm1.
- gemm1 MF32 with the scalar rescale: -3.9% / -5.3%. The BPF / DEFER / SCHED_G1 / FBS_PRO retune: noise.
- s2 skip variant sets other than 4,2,1: they lose most of the gain.
- gemm1 pad-skip: shipped, but delivers only 20-30% of its MFMA cut (partial).
- s2 rescale packing: not built. The live lever is the opposite.

### Insight blackboard (supersedes the R2 list)
1. **State after R3.** The live path is still 4 dispatches, with span equal to the kernel sum (no host gaps):
   - `k_count` + `k_place`: the fast sort, which zero-fills moe_buf and now also emits `blk_cnt[m]`;
   - `moe_fp8bs_gemm1_0` with `FBS_RS=1`, `FBS_WPE=2`, `FBS_G1_PRIO=1`, `FBS_EPI_MAXNUM=1`, `FBS_PADSKIP=4,2,1`;
   - `moe_s2_fp8bs_0` with the R2 recipe plus `S2_PADSKIP=4,2,1`.

   The fallbacks `FBS_PADSKIP=0`, `S2_PADSKIP=0` and `MOE_FAST_SORT=0` (CK sort, no skips) all PASS correctness.
2. **Cost structure** (110-forward steady-state trace, medians):

   | case | k_count | k_place | gemm1 | s2 | span |
   |---|---|---|---|---|---|
   | tpr2048 | 7.3 us | 37.8 us | 645.5 us | 369.3 us | 1059.8 us |
   | tpr4096 | 7.3 us | 48.8 us | 1199.9 us | 674.9 us | 1931.3 us |

   - Shares of the forward: gemm1 60.9% / 62.1%, s2 34.8% / 34.9%, sort 4.3% / 2.9%.
   - vs R2: gemm1 -5.5% / -5.3%, s2 +1.7% / +3.7% (clock carry-over, see item 3).
   - tpr2048 is the weaker case (1.586x vs 1.670x) and carries more padding (5.7% vs 3.2% of MFMA rows before the skip).
   - Wrapper-level graph capture has never been tried. With span equal to the kernel sum it should gain about 0.
3. **The governing rule, now measured.** Source: a same-session, same-GPU sustained R2-vs-R3 A/B at a pinned 750 W
   (`profile_output_r3_pmc/power_ab.log`).
   - R3 removed 9.9% / 9.5% of clock cycles per forward. But sclk fell 6.3% / 6.4% (1077 -> 1010 and 1069 -> 1001 MHz),
     so time fell only 3.9% / 3.4%. **At the cap, time per forward is proportional to energy per forward.**
   - How each kind of change converts to time:

     | change | example | conversion |
     |---|---|---|
     | remove work without adding bytes | s2 pad-skip: MFMA -3.49% / -1.90% gave s2 -3.4% / -1.8% | about 1:1 |
     | remove stall cycles | pk->scalar swap, setprio | about 45%; the NEXT kernel inherits the lower clock (byte-identical s2 ran +4.6% / +4.8% slower) |
     | remove work but add bytes | gemm1 pad-skip | about 20-30% of the MFMA cut |

   - Rank levers by energy removed: MFMA ops, EA/HBM bytes, VALU ops.
   - Judge every kernel change at the forward level. An isolated-kernel A/B overstated r3_d0 by about 2x: gemm1 was
     -6.6% in the trace, the forward only +2.8%.
4. **gfx942 packed-FP32 rule.** Source: r3_d0's micro-bench, `round_3/engineer_0/tools/coissue`. Cycles per
   `v_mfma_f32_16x16x32_fp8` slot:

   | co-issued with the MFMA | cycles |
   |---|---|
   | nothing | 16.6 |
   | 1 `v_pk_fma_f32` | 30.1 |
   | 2 scalar `v_fma_f32` | 17.1 (free) |
   | 2 `v_pk_mul_f32` | 34.1 |

   - A wave issuing packed FP32 makes about zero progress while a co-resident wave streams MFMAs. `V_PK_*_F32` waits for
     the SIMD's matrix pipe to drain, whichever wave owns the MFMA.
   - Transcendentals (exp, rcp) and `cvt_pk_fp8` co-issue, but slow the MFMA wave to about 21.7 cycles.
   - LLVM's `SIPreEmitPeephole::performF32Unpacking` deliberately unpacks packed FP32 inside MFMA shadows. That is where
     gemm1's 542 and s2's 668 "scalar" FMAs came from.
   - **R2's steer to pack them was INVERTED.** Emit rescales as per-element scalar `llvm.fma` (`_fma_sc`,
     `flydsl_moe_fp8bs.py:160`) and keep `V_PK_*_F32` out of MFMA loops.
5. **The gemm1 recipe that worked** (r3_d0; all default-ON in `flydsl_moe_fp8bs.py`):

   | knob | what it does | measured |
   |---|---|---|
   | `FBS_RS=1` | scalar `v_fmac` rescale: 753 `v_pk_fma_f32` -> 0, 542 -> 2048 scalar FMAs | gemm1 -3.5% / -3.8% isolated; forward +2.30% (5/5) |
   | `FBS_WPE=2` | `rocdl.waves_per_eu=2`, VGPR 240 -> 230 | required to hold 2 waves/SIMD |
   | `FBS_EPI_MAXNUM=1` | maxnumf row-max: `v_cmp_o` 56 -> 0, `v_cndmask` 60 -> 4, -127 VALU per wave | gemm1 +0.38% / +0.45% |
   | `FBS_G1_PRIO=1` | `s_setprio(1)` around the WHOLE K loop | +0.2% / +0.5% alone; +1.21% / +1.06% with maxnum |

   - Final: +2.78% forward same-session (5/5). The verifier measured +2.79% (1.5752 -> 1.6192x).
   - gemm1 trace -6.6% / -6.7%; PMC cycles -10.7% / -11.8%; MFMA-busy 61 -> 65% and 66 -> 74%; issue-wait 56-57% -> 49%.
     a2/a2s are bit-identical.
   - `s_setprio` is a scheduling-region boundary. Per-tile placement (`FBS_G1_PRIO=2`) cut each tile's
     `sched_group_barrier` pattern off its MFMAs and lost 5.4% / 4.8%. Place setprio only around whole loops.
6. **gemm1 dead ends and ceilings** (r3_d0; reference diff
   `round_3/engineer_0/slot_sched_prologue_knobs_NOT_A_CANDIDATE.diff`).
   - Dead ends:
     - VALU-slotted `sched_group_barrier` (`FBS_SCHED=5`) cut K-loop `s_nop` cycles from 1325 to 425 (VGPR 250), but
       timing was noise. Stall cycles cost no energy under the cap.
     - BPF=1, DEFER=2, SCHED_G1=4/16 and the slot-schedule combinations stayed within the ±0.5% null arm.
     - `FBS_PRO` (prologue X/B loads before the scale-table LDS writes) was noise.
     - MF32 (32x32x16) with the scalar rescale lost 3.9% / 5.3% of gemm1.
   - Wrong-output what-ifs on the step-1 build:

     | what-if | gemm1 gain | reading |
     |---|---|---|
     | `DBG_BK0` (B always L2-hot) | +14.9% / +10.0% | MFMA-busy unchanged but sclk 1.165 vs 1.05 GHz: weight traffic costs clock, not latency |
     | `DBG_NOSCALE` | +6% | the 2048-FMA rescale is irreducible on gfx942 (no scaled MFMA; exactly 1 scalar FMA per MFMA) |
     | `DBG_NOEPI` | +5.3% / +3.8% | |
     | `DBG_NOBAR` | +1.7% / +0.8% | |
7. **Pad-skip mechanism** (r3_d1; shipped in both GEMMs).
   - `k_place` emits `blk_cnt[m] = min(64, base + cnt - m*64)` inside its `sort_grid`-strided loop. k_place time is
     unchanged and the sorted ids stay bit-identical to CK.
   - Each GEMM s_loads the count next to the expert id (no new dependent vector load, no ballot) and takes a uniform SCC
     branch:
     - more than 32 valid rows: the unchanged full body;
     - 17-32 rows: a 2-subtile copy;
     - 1-16 rows: a 1-subtile copy.

     Skipped subtiles keep zero accumulators, so every a2/a2s row stays defined.
   - **Only the 4,2,1 variant set keeps s2's full-body ISA identical** (246 VGPR, 0 spills). 4,2 / 4,3 / 4,3,2,1 perturb
     the full body through the global register allocator (16 extra `v_mov` or 14-36 spills) and lose most of the gain.
     Check full-body ISA stats (`round_3/engineer_1/tools/isa_cmp.py`) for every variant set.
   - s2 result: s2 -3.6% / -2.0% by kernel name; forward +0.71% to +0.93% same-session; the verifier measured +0.67%.
8. **A subtile skip is not a block deletion** (r3_d1's grid-cap staircase).
   - gemm1's marginal cost is linear at 2.2-2.5 us per 64-row M block, with no tail cliff.
   - The 4,2,1 skip removes 9.5 / 10 block-equivalents of MFMA work (ideal -22 / -23 us). Isolated, it measured only
     -10.5 / -8.3 us (36-48%), because each CTA still streams its full W panel (about 1 MB), its X rows and the epilogue.
   - Residual padding under 4,2,1 is 2.29% / 1.29% of MFMA rows. A 3-subtile variant would remove only 0.76% / 0.39% more;
     the rest is partial subtiles.
   - Each forward has about 8 blocks each with 1, 2 and 3 active subtiles, out of 272 / 525 M blocks.
9. **The gemm1 pad-skip under-delivers.** Sources: the integrate A/B and the reprofile's l2 attribution
   (`profile_output_r3_pmc/l2_attrib/summary.txt`).
   - It cuts gemm1 MFMA by 3.49% / 1.90%, worth about -2.1% / -1.2% of the forward at 1:1. It delivered only +0.31% paired
     over the 2-stack (24/28).
   - With the skip, gemm1 makes +1.7M / +1.2M more L2 misses per call: +16% / +8%, about +215 / +155 MB of EA reads, on
     the same 45.1M requests. It also adds +15% EA writes (spill write-back).
   - With `FBS_PADSKIP=0` the misses return exactly to R2's level. Setprio, RS and WPE are not the cause.
   - VGPR went 230 -> 256 with 35 spills (120 B/lane scratch), all in the 1- and 2-subtile paths. The full body is
     spill-free and matches r3_d0's.
   - Spills are too small to explain the misses. In the 4,1 arm the extra writes are about 3 MB, yet misses rise 1.16M:
     about 1.2 MB per fast tail CTA, which is all of its W + X reads.
   - **Inferred, not measured:** a tail CTA that runs 2-4x shorter falls out of step with the same-expert siblings that
     share its W panel, and loses its L2 hits.
   - Code grew 40 -> 71 KB (s2 24 -> 45 KB) at no cost: I-cache hit stays at or above 99.9%.
10. **Integration R3.** Disjoint-file patches added linearly again.
    - r3_d0 + r3_d1 was +0.61% over r3_d0 (22/22 wins).
    - r3_d1's s2 skip also absorbed r3_d0's clock carry-over on s2: 381 / 687 -> 368 / 674 us.
    - The hand-merged gemm1 skip added +0.31% (24/28). Its structure: `compute_tile(m_eff)` plus one `kbody(m_eff)` copy
      per variant, each keeping r3_d0's setprio, `_fma_sc` and maxnumf.
    - Fresh-tree `patch -p1`, the JIT build and the 24-case extended correctness tool all PASS. The tool is
      `round_3/integrate/tools/ext_correct.py`: rows 256-32768, 4-21632 valid tokens, 2 seeds, with the frozen generator,
      reference and tolerances. Reuse it for any sort, skip or store-path change.
11. **Bytes cost time; remaps that only move MALL traffic do not** (refines R2's L2 steer).
    - `DBG_BK0` (+10-15% of gemm1) and the gemm1 pad-skip's +215 / +155 MB of EA reads both show that EA bytes cost clock
      under the cap.
    - R2's XCD remaps were neutral because they traded X misses for W refetches, leaving net misses flat.
    - The compulsory 512 MB W13 HBM read cannot be cut at tile_m=64.
    - Levers that remove bytes: the single-local-expert plain store (item 15), repairing the gemm1 skip's lost L2 sharing
      (item 9), and avoiding spill traffic.
12. **CLOSED: XCD remaps on the live kernels** (R2).
    - gemm1:
      - plain XCD tiling gn=4: 0.9988x (noise);
      - the gn=16 M-partition: -4.9% / -10.9% of stage1;
      - the expert-run order (gn=16, ord=3): 0.9863x, because its dependent seid load + ballot + division prologue costs
        2-3.5% of stage1.
    - s2: XCD tiling gn=2/4/8/16 loses 1.4/0.9, 2.7/0.4, 7.2/3.6 and 11.5/12.9% of s2. s2 needs the plain (16 N, M) grid.
    - Only untested: a critical-path-free remap, using a sort-emitted per-CTA (m,n) table read by s_load. Its ceiling is
      about +1% of stage1. R3's `blk_cnt` s_load shows that sort-emitted per-block metadata is free to read.
13. **The s2 state** (R2 recipe + R3 skip; `flydsl_moe_fp8bs_s2.py`, all default-ON).
    - Knobs: `S2_PRIO=1`, `S2_FPRE=1`, `S2_WPE=2`, `S2_EPI=3`, `S2_VA_NV=1`, `S2_PADSKIP=4,2,1`.
    - fpre needs `waves_per_eu=2`. Without it the kernel needs 272 VGPR, drops to 1 wave/SIMD and loses 16%.
    - Full body: 1024 MFMA, 668 scalar `v_fma_f32`, **178 `v_pk_fma_f32`** (170 in the K loop, 8 in the epilogue),
      246 VGPR, 0 spills, 32 KB LDS. The reduced bodies are pk-heavier: 123 pk per 256 MFMA, 156 per 512.
    - The rescale is `fmath.fma` on f32x4 vectors (around lines 600 and 638), the same form gemm1 replaced with `_fma_sc`.
    - R2 dead ends:
      - soffset addressing; vm_mode 1/2/3; VALU start slot 0/4; 4 VALU per slot;
      - max-ilp; iterative-ilp; the igrouplp exact solver (compile hang);
      - sc0; B prefetch 2 K-tiles ahead; uniform-branch atomics (521 VGPR);
      - de-loop (-3.6% / -1.4%); XCD tiling.
    - Ceilings: the rescale is 8.4% / 9.6% of s2, atomics 3.6% / 4.2%, the tail 3-4 us (so no split-K).
14. **Grid/loop asymmetry** (R1/R2, still holds).
    - Never put a device loop around the gemm1 body (-4.5%). Its empty CTAs are free: 69,632 launched, 4,352 / 8,400
      doing work.
    - s2 is the opposite: keep its existing degenerate grid loop (`S2_DELOOP=0`). A full if-guarded grid lost
      3.6% / 1.4%.
    - Never change either kernel's loop or grid structure without a forward-level A/B.
15. **The bytes lever left: single-local-expert plain stores.** 60.0-60.6% of received tokens have exactly one local
    expert: 6505 / 10841 and 13037 / 21575 (`profile_output_r3_pmc/routing_stats.log`, stable across both seeds).
    - For those rows, s2 could plain-store bf16 instead of atomic-adding.
      - It is race-free: each output column range of such a token is written by exactly one CTA.
      - It is numerically identical, since 0 + x = x.
      - Padding/dummy entries must stay on their current path.
    - `k_place` would then zero-fill only multi-expert rows. That cuts about 53 / 106 MB of fill writes; k_place's
      37.8 / 48.8 us is HBM-write-bound.
    - It also cuts the matching s2 atomic RMW line fetches. s2's EA fetch is 598 / 850 MB, and atomics were 3.6% / 4.2%
      of s2 in R2.
    - `k_count` already knows each token's local count.
    - Estimated -1% to -1.5% of the forward, converting about 1:1 because it removes bytes.
    - It must keep the `[M, hidden]` in-kernel accumulate output (the expert skill's memory contract): no expanded
      `[M*top_k, hidden]` buffer.
16. **Fill and sort** (R2, still holds).
    - The k_count + k_place sort with 304 sort-owned fill blocks shipped in R2: +0.48% / +0.79%, with sort + fill going
      from 67.7 / 110.8 to 42.3 / 54.5 us. Only about 50% of micro-bench sort savings reach the forward.
    - The fill costs about 11 / 27 us wherever it runs, and tracks the bytes written.
    - Closed:
      - side-stream fill (-0.3% to -6%);
      - serial pre-s2 fill (ties);
      - `hipExtAnyOrderLaunch` (ignored on gfx942);
      - the single-kernel grid-barrier sort (34 / 50 us);
      - zsplit; 176 blocks; NT stores.
    - k_place's padding and M-block loops must stride by `sort_grid`, not `gridDim.x`, or M < 8192 faults.
17. **Profile R3** (rocprofv3: 8 `--pmc` passes plus an 11-arm l2 attribution; rocprof-compute still fails its 13-package
    dependency check).

    | metric | gemm1 | s2 |
    |---|---|---|
    | MFMA busy | 65.1 / 74.2% | 55.1 / 65.7% |
    | wave cycles: active / issue-wait / dep-wait | 28 / 49 / 22% | 25-29 / 50-53 / 18-21% |
    | non-MFMA VALU per MFMA | 1.27 (FMA_F32 1.00, all scalar; TRANS 0.033, MUL 0.069, CVT 0.008) | 1.29 (FMA_F32 0.82) |
    | L2 hit | 73.4 / 79.6% | 72.4 / 77.2% |
    | EA fetch | 1502 / 2214 MB | 598 / 850 MB |
    | resident waves (of a 2-wave cap) | 1.75 / 1.86 | 1.76 / 1.83 |
    | useful FP8 rate | 853 / 910 TFLOPS (+5.8% / +5.6%) | 746 / 809 TFLOPS (down: clock carry-over) |

    - Both kernels: EA traffic at 35-40% of peak, VMEM active about 3%, LDS util 26-33% (bank conflicts 12.9% in gemm1,
      0% in s2), I-cache at or above 99.9%.
    - Not memory-, LDS-, I-cache- or launch-bound.
    - A 3rd wave is out: gemm1 is at 256 VGPR with spills and s2 at 246, and the issue-wait is MFMA-pipe sharing at a
      power-pinned clock.
18. **Still CLOSED** (R1/R2):
    - the 128-row transposed stage1 s1v2/s1t (forward 0.976x / 0.994x);
    - the gemm1 device grid-stride loop (-4.5%);
    - sort dispatch_policy=1 (about 0.35x), and opus_moe_sorting (equal to CK);
    - a 3rd s2 wave via VGPR trim (it floors at 187-221 VGPR unspilled);
    - stage1->stage2 fusion (the a2 round-trip is under 3% of EA bytes).

    `MOE_S1_IMPL` defaults to gemm1. Confirm `moe_fp8bs_gemm1_0` and `moe_s2_fp8bs_0` by name in any trace. The names are
    stable because padskip is in the lru key and the FlyDSL disk cache is off.
19. **Measurement.** Cross-session drift is about 1%, and GPU3 runs about 2% slower for every arm, so only same-session,
    same-GPU paired ratios count.
    - The protocol that resolved +0.3-0.8% effects in R3: at least 5 interleaved COMMANDMENT rounds with rotating arm
      order under one gpu_lock lease, reporting medians and win counts (r3_d0 5/5, r3_d1 10/10 pooled, integrate 12/12
      and 24/28).
    - For sub-1% knobs, use an in-process paired A/B with a null arm (canonical against itself, about ±0.5%):
      `round_3/engineer_0/tools/ab_g1.py`, `round_3/engineer_1/tools/{ab.py,g1ab.py}`.
    - For per-kernel attribution, use a same-process interleaved rocprofv3 kernel trace by name
      (`round_3/engineer_1/tools/{trace_ab.sh,trace_kern.py}`). Always read the NEXT kernel too, because of the clock
      carry-over.
    - Power and clock A/B: `profile_output_r3_pmc/tools/power_ab.sh`. L2 attribution: `profile_output_r3_pmc/tools/l2_attrib.sh`.
20. **Process.** R3 verified 2 of 2 (R2 2 of 3, R1 0 of 3).
    - What worked: `checkpoints.log` with one measured verdict per step, `best_patch.diff` with winners default-ON, and
      fresh-tree `patch -p1` checks.
    - The subagent harness blocks `report.md`, so `checkpoints.log` plus the worker notes are the report.
    - NOT_A_CANDIDATE diffs paid off: integrate shipped r3_d1's priced gemm1 skip from `gemm1_padskip_NOT_A_CANDIDATE.diff`.
    - When two directions must touch one file (s2 in R4), assign regions explicitly and plan a hand-merge at integrate.
21. **Expert skills.** Re-checked `index.yaml`; it is unchanged since Nov 18.
    - The only kernel-scope validated match for gfx942 / fp8_e4m3_fnuz / fused_moe_grouped_gemm is
      `flydsl_rewrite_quantized_moe`, a triton->flydsl recipe.
    - There is nothing to reproduce: this run is FlyDSL-native, and `moe_blockscale_2stage` is absent.
    - Its one live constraint is the mandatory memory contract: stage-2 output must stay `[M, hidden]` with in-kernel
      accumulate. That bounds the plain-store direction.
    - The blkmap and stage2_fp8partial skills are gfx950-only.

### Hypothesis ledger (cumulative, deduplicated; "actual" = same-session ratio vs canonical unless noted)
| direction | specialty | expected | actual | verdict | lesson |
|---|---|---|---|---|---|
| r1_d0 | memory | 1.08 | 0.985 | dead_end | The 128-row s1t ties gemm1, and unit-128 padding costs more. Never re-fund. MOE_S1_IMPL defaults to gemm1 |
| r1_d1 | host_runtime | 1.035 | 1.01 | partial | Fast sort + fill; R2 shipped it as r2_d1 (+0.65%). The in-gemm1 zf_pos=3 fill was not pursued |
| r1_d1:gemm1_grid_loop | host_runtime | 1.035 | 0.955 | dead_end | Never put a device loop in gemm1 (s2 is the opposite case) |
| r1_d1:sort_dispatch_policy | host_runtime | 1.035 | 0.35 | dead_end | The single CK MoeSortingKernel takes 1.7-3.9 ms; opus = CK |
| r1_d2 | compute | 1.04 | n/a | dead_end | A 3rd s2 wave is not reachable by VGPR trim. R2 won at 2 waves by removing work |
| r2_d0 | memory | 1.03 | 0.9988 | dead_end | Plain XCD tiling trades X misses for W refetch. X-hot ceiling 3.2% / 5.5% |
| r2_d0:expert_run_order | memory | 1.03 | 0.9863 | dead_end | -25% misses, but its prologue costs 2-3.5% of stage1. Retry only if the remap is free (an s_load of sort-emitted data is free, per R3) |
| r2_d1 | host_runtime | 1.012 | 1.0065 (verified 1.5535) | confirmed | Shipped. About half the expected gain; the placement lane is exhausted |
| r2_d1:fill_placement | host_runtime | 1.012 | 1.0 (best ties) | dead_end | Fill cost (~11 / 27 us) does not depend on placement; only fewer bytes help |
| r2_d2 | compute | 1.02 | 1.0182 (verified 1.5629) | confirmed | Shipped. Removing work converts to time under the cap |
| r2_d2:s2_deloop | compute | 1.02 | s2 0.975 | dead_end | s2's 61k empty guarded CTAs are not free; keep its loop |
| r2_d2:s2_xcd_tiling | compute | 1.02 | s2 0.989 (gn=2, best) | dead_end | The plain grid keeps atomics per XCD; tiling only loses |
| r2_integrate | integrate | 1.025 | 1.028 cumulative (verified 1.5710) | confirmed | Disjoint-file patches add; the k_place stride fix is required |
| r3_d0 | compute | 1.02 | 1.0278 (verified 1.6192) | confirmed | Scalar-FMA rescale (packed FP32 blocks the MFMA pipe on gfx942), whole-loop setprio, maxnumf. Only ~45% of the cycle cut reaches the forward; s2 inherits the lower clock |
| r3_d0:packed_rescale | compute | 1.02 | inverted (the scalar form won) | dead_end | Never pack FP32 near MFMAs on gfx942. LLVM unpacks them in MFMA shadows on purpose |
| r3_d0:valu_slot_sched | compute | 1.01 | ~1.00 (noise) | dead_end | Nop cycles 1325 -> 425 with no time change: stalls cost no energy at the cap |
| r3_d0:per_tile_setprio | compute | 1.006 | gemm1 0.949 | dead_end | s_setprio is a sched-region boundary; only whole-loop placement helps |
| r3_d0:mf32_and_retune | compute | 1.01 | MF32 gemm1 0.954; knobs noise | dead_end | Defaults stay SCHED=3, G1=8, DEFER=1, BPF=2 |
| r3_d1 | algorithm | 1.01 | 1.0071 (verified 1.5814) | confirmed | Sort-emitted blk_cnt plus a uniform branch. s2 -3.6% / -2.0%, 1:1 with its MFMA cut. Only 4,2,1 keeps the full-body ISA |
| r3_d1:s2_variant_sets | algorithm | 1.01 | s2 -1.6% to 0% (vs 4,2,1 at -2.2% / -1.2%) | dead_end | Other sets perturb the full body's register allocation; run isa_cmp.py first |
| r3_d1:gemm1_padskip | algorithm | 1.016 | 1.0031 over the 2-stack | partial | Shipped, but +16% / +8% L2 misses and 35 spills eat 70-80% of the MFMA cut. Repair it |
| r3_d1:s2_rescale_packing | algorithm | 1.01 | not built | dead_end | Inverted by r3_d0's finding; scalarize s2's 178 pk_fma instead |
| r3_integrate | integrate | 1.03 | 1.0354 cumulative (verified 1.6273) | confirmed | Disjoint files add; the s2 skip absorbs the clock carry-over; ext_correct.py and all fallbacks PASS |

### Bottleneck now
**Compute:** fp8 MFMA-bound at the 750 W cap, now pinned harder (sustained sclk about 1.00 GHz, down from about 1.07 GHz in R2).
- Time per forward tracks energy per forward. R3 cut 9.9% / 9.5% of cycles, but the clock fell 6.3%, so time fell
  3.9% / 3.4%.
- gemm1 is 60.9% / 62.1% of the 1.060 / 1.931 ms forward: 0 pk_fma, MFMA-busy 65% / 74%, 256 VGPR with 35 tail-path
  spills, and +16% / +8% L2 misses from the skip.
- s2 is 34.8% / 34.9% and still has 178 `v_pk_fma_f32` blocking the matrix pipe.

Energy still removable:
- s2's pk_fma pipe blocks;
- the 89 / 177 MB zero-fill plus s2's atomic RMW bytes;
- the gemm1 skip's lost L2 sharing and its spills;
- the epilogues;
- residual padding (2.29% / 1.29% of MFMA rows).

### Suggested next (round 4)
Three energy-removal directions, with explicit file and region ownership:
1. **(compute; `flydsl_moe_fp8bs_s2.py` rescale sites only)** Port `_fma_sc` to s2's 178 `v_pk_fma_f32`.
   - That covers the K loop, the epilogue and the reduced bodies.
   - Hold 246 VGPR, 0 spills and 2 waves, and keep outputs bit-identical.
   - Estimated -0.7% to -0.9% of the forward.
2. **(algorithm or host_runtime; `moe_sort_fast.py` + `moe_ep.py` + the s2 epilogue store path)** Plain-store the 60%
   single-local-expert rows, and have k_place zero-fill only multi-expert rows.
   - Estimated -1% to -1.5% of the forward.
   - Keep the `[M, hidden]` contract and run `ext_correct.py`.
   - It shares the s2 file with direction 1, so plan a hand-merge at integrate.
3. **(algorithm or memory; `flydsl_moe_fp8bs.py` only)** Repair the gemm1 pad-skip.
   - First make the tail paths spill-free.
   - Then restore L2 sharing for fast tail CTAs; an `s_sleep` pacing what-if is one way to test it.
   - Judge it with `l2_attrib.sh` plus a forward A/B. The ceiling is up to about 1.9% / 0.9% of the forward, realistically
     under 1%.

Lower priority: trim gemm1 epilogue VALU (`DBG_NOEPI` is worth +5.3% / +3.8% of gemm1).

Judge every kernel change at the forward level, and read the next kernel's time.

**Do NOT re-fund:**
- packing FP32 near MFMAs;
- VALU-slot scheduling or nop removal;
- per-tile setprio;
- MF32;
- more skip variants;
- a 3rd wave;
- plain XCD remaps or the expert-run order;
- s2 de-loop;
- fill placement, side-stream, NT or block-count tuning;
- the 128-row stage1;
- the gemm1 device loop;
- sort dispatch_policy=1.

**Stop gate (a) is not formally met:** wrapper-level graph capture was never tried. Expect about 0, but a cheap A/B closes
the gate.

**Not a stop:** about 2-3% of estimated, measured-but-unbuilt energy levers remain.

## Round 4 — update_memory

**Outcome: a +1.4% win measured, but NOT banked.**
- Winner: the integrated r4_d1 + r4_d0 hand-merge (`round_4/integrate/integrated_patch.diff`).
  - **1.6505x** geomean: pooled 30 interleaved COMMANDMENT runs on GPUs 1-3, 1.0442 / 1.8919 ms.
  - +1.60% over R3 same-session (30/30 wins). Correctness PASS: rel_l2 2.315e-3 / 2.307e-3, unchanged.
- The workflow commits a winner only above cumulative x (1 + MIN_IMPROVE 0.02) = 1.6273 x 1.02 = **1.6598**. So
  IMPROVED = false, nothing was committed, and the canonical workspace is still `31c4cb1` (R3, 1.6273x).
- Run state (budget 12 per the run args, MAX_NO_IMPROVE = 2 by default):
  - noImprove is now 1 of 2.
  - 11 directions have been dispatched, so R5 gets exactly 1 direction, has no integrate step (that needs 2 or more
    verified patches), and is the last round.
  - Unless R5's single verified patch is above 1.6598, the run ends at 1.6273x and R4's gain is lost.

2 of 3 directions verified (R3: 2 of 2, R2: 2 of 3, R1: 0 of 3). r4_d2 returned after integrate had started.

### Per-direction
| id | specialty | strategy | expected | same-session measured | verified | verdict |
|---|---|---|---|---|---|---|
| r4_d0 | compute | s2 rescale as per-element scalar `llvm.fma` (`S2_RS=1`, `_fma_sc4`, 2 FMAs per MFMA slot) | 1.008 | pooled 26 COMMANDMENT rounds: +0.05% / +0.35% (about +0.2% geo). Trace: s2 -1.58% / -1.90%, the next gemm1 +0.34% / +0.99% | 1.629 (+0.10% over 1.6273) | partial (shipped in the R4 merge, where it adds +0.26%) |
| r4_d1 | algorithm | single-local-expert rows plain-store in s2; k_place zero-fills only multi-expert rows; graph-capture A/B | 1.01 | +1.42% (6/6), +1.62% (6/6), +1.33% (5/5) geo in 3 sessions; in-process +1.54% (9/9) | 1.6462 (+1.16%) | confirmed (shipped in the R4 merge) |
| r4_d2 | memory | gemm1 skip repair: per-variant prologue (0 spills), `s_sleep` tail pacing, PADZERO | 1.01 | pooled 18 rounds: +0.29% / +0.23% (+0.25% geo, 15/18; null -0.04%) | not verified (`apply_failed`: result arrived at 18:20, integrate started at 18:15) | partial (carry into R5) |
| integrate | - | r4_d1 + r4_d0 hand-merge (3 adjacent hunks in the s2 file) | - | +0.26% over r4_d1 (25/30); +1.60% over R3 (30/30) | 1.6505 | confirmed, NOT committed (below the 1.6598 gate) |

Sub-experiments closed this round:
- r4_d0:
  - `S2_PRO_SC`: fenced scalar prologue fmuls; neutral.
  - `S2_BIMM`: B-load immediate offsets; noise.
  - `S2_RS_NV=1`: worse than NV=2.
- r4_d1:
  - wrapper CUDA-graph capture: -0.37% geo.
  - in-asm `s_cbranch_execz` over the dead store/atomic pair: noise.
  - store policy: `nt` is best (+0.35% over none); sc1 and sc0sc1 are no better.
- r4_d2:
  - `s_sleep` sweep 4,6 through 16,24: 8,12 is best.
  - self-timed tail pacing: no better than 8,12.
  - 4,1-only plus pacing: -0.14%.
  - global lockstep pacing: -0.1% to -2.0% of the forward.
  - cache-policy bits: W13 nt -24.4% / -19.9% and X nt -7% of the forward; everything else is noise.
  - LLVM hoist-common off, reduced-body BPF=1: VGPR-only changes.

### Insight blackboard (supersedes the R3 list)
1. **State after R4: the win is not banked.**
   - Canonical is still `31c4cb1` (R3, cumulative 1.6273x).
   - R4's 1.6505x missed the 2% commit gate (1.6598).
   - R5 has 1 direction, no integrate step, and is the last round (budget 12, 11 used; noImprove 1 of 2).
   - R5's one verified patch must beat 1.6598, or the run ends at 1.6273x.
2. **The R4 stack to carry into R5.**
   - Its parts:
     - `round_4/integrate/integrated_patch.diff`: `flydsl_moe_fp8bs_s2.py`, `moe_ep.py`, `moe_sort_fast.py`;
     - `round_4/engineer_2/best_patch.diff`: `flydsl_moe_fp8bs.py` only; +0.25% same-session, never verified.
   - Disjoint files. I dry-ran `patch -p1` of both onto a copy of canonical: both apply cleanly and all four files parse.
     **The combined forward has never been measured**, and r4_d2 has not been through the verifier.
   - Defaults ON in the stack:
     - s2: `S2_RS=1`, `S2_RS_NV=2`, `MOE_SSTORE=1`, `S2_SST_POL=1` (nt);
     - gemm1: `FBS_PS_PRO=1`, `FBS_PS_SLEEP=8,12`, `FBS_PADZERO=1`.

     Each knob set to 0 restores its R3 ISA byte-identically.
   - Expected stack: about 1.6505 x 1.0025 = about 1.655 (estimate). That is still just under 1.6598, so R5 needs a real
     +0.5% increment on top.
3. **Cost structure** of the R4 merge (integrate trace, rocprofv3 medians of 220 dispatches, 4 arms on 2 GPUs):

   | case | k_count | k_place | gemm1 | s2 | kernel sum |
   |---|---|---|---|---|---|
   | tpr2048 | 6.8-7.5 us | 32.5-32.9 us | 643.5-647.8 us | 352.3-354.1 us | 1036-1042 us |
   | tpr4096 | 6.8-7.5 us | 38.4-39.0 us | 1190-1201 us | 644.0-649.8 us | 1880-1897 us |

   - Shares of the forward: gemm1 about 62% / 63%, s2 34% / 34%, sort 3.8% / 2.4%. Still 4 dispatches, with span equal to
     the kernel sum.
   - Against the R3 reprofile (cross-session): k_place -5 / -10 us, s2 -16 / -27 us, gemm1 flat.
   - tpr2048 is still the weaker case (1.6038x vs 1.6986x).
4. **The governing rule held again** (750 W cap: time per forward tracks energy per forward).
   - Removing bytes converts about 1:1. r4_d1 removed 53 / 107 MB of fill writes and 41% of s2's memory-side atomics,
     for +1.33% to +1.62% (17/17). k_place fell 5 / 10 us and s2 1.3% / 1.0%; the next gemm1 was unchanged.
   - Removing pipe stalls in the SECOND kernel converts about 0 at the forward.
     - r4_d0: s2 -1.6% / -1.9%, but the next gemm1 ran +0.34% / +0.99% slower. Kernel sum -0.21% / -0.02%.
     - Sustained-power test: forward -0.4% / -0.5%, with the capped clock 0.5-0.8% lower.
     - COMMANDMENT: about +0.2% geo.
   - Stall removal still adds in combination. r4_d0 on top of r4_d1 gave +0.26% (25/30): s2 -1.9% to -2.4%, gemm1 +0.3%
     to +1.2%.
5. **gfx942 packed-FP32 rule** (R3 micro-bench).
   - `V_PK_*_F32` waits for the SIMD's matrix pipe to drain: MFMA + 1 `v_pk_fma` takes 30.1 cycles, MFMA + 2 scalar FMAs
     17.1.
   - Emit rescales as per-element scalar `llvm.fma`: `_fma_sc` in gemm1, `_fma_sc4` in s2.
   - New in R4:
     - LLVM's SLP vectorizer re-packs plain scalar fmuls that feed a vector store into `v_pk_mul_f32`. Wrapping each
       product in `llvm.arithmetic.fence` keeps them scalar.
     - pk ops outside the K loop measured neutral (s2's 6 prologue `v_pk_mul_f32`).
   - Left in the stack: s2 has 6 `v_pk_mul_f32` (prologue) and gemm1 has 8 (outside the K loop).
6. **s2 in the R4 stack** (`flydsl_moe_fp8bs_s2.py`).
   - Knobs: the R2/R3 set (`S2_PRIO=1`, `S2_FPRE=1`, `S2_WPE=2`, `S2_EPI=3`, `S2_VA_NV=1`, `S2_PADSKIP=4,2,1`) plus
     `S2_RS=1`, `S2_RS_NV=2`, `MOE_SSTORE=1`, `S2_SST_POL=1`.
   - ISA: 234 VGPR, 0 spills, 32 KB LDS, 2 waves/SIMD, 0 `v_pk_fma_f32`, 1792 scalar FMAs (= MFMAs). The epilogue has
     64 nt `global_store` + 64 `global_atomic_pk_add_bf16` + 64 `s_cmov`.
   - **The scale-math lane is exhausted.** There is exactly 1 FMA per MFMA, irreducible bit-exactly (gfx942 has no scaled
     MFMA).
   - Address VALU is not an energy lever. `S2_BIMM` removed 92 VALU per tile (`v_or` 120 -> 0) at 250 VGPR: noise, like
     R2's soffset.
   - NV=2 beats NV=1 (isolated s2 -1.59% / -1.16% vs -1.15% / -1.03%).
   - Remaining s2 levers are bytes only. The a2 read pattern and W2 are as cheap as the plain grid allows (R2 closed s2
     XCD tiling).
7. **Single-local-expert plain stores** (r4_d1, in the stack).
   - Mechanism:
     - `k_count` writes `tsingle[t]` = (local entries == 1).
     - `k_place` sets bit 31 of single tokens' sorted ids; gemm1's `& 0xFFFFFF` decode is untouched.
     - Its 304 fill blocks zero only non-single rows (wave-ballot compaction into LDS, one wave per row).
     - The s2 epilogue emits one straight-line inline-asm block per row. `kind = readfirstlane(valid ? (bit31 ? 1 : 2) : 0)`,
       and `s_cmp`/`s_cmov` build exec.
     - Single rows get 2 nt `global_store_dword`, multi rows 2 `global_atomic_pk_add_bf16`, and padding rows get nothing
       (the padding zero-atomics are gone).
   - Pitfall: an inline-asm `"s"` operand fed by an i64 select was silently given VGPRs and failed to assemble. Pass a
     readfirstlane'd i32 and build exec inside the asm.
   - Bytes:
     - k_place EA writes 89.0 -> 35.7 / 177.3 -> 70.4 MB;
     - s2 EA atomics 2.167M -> 1.268M / 4.235M -> 2.496M (-41%).
     - `TCC_ATOMIC == TCC_EA0_ATOMIC`: every bf16 device atomic is an L2-uncached, memory-side read-modify-write.
     - Plain stores live in L2 as dirty lines and raise s2 EA reads +9.5 / +17 MB. nt recovers 6.5 / 7 MB.
   - Single rows are bit-identical to R3; multi rows differ only at atomic-order noise.
   - The remaining multi-row fill (35.7 / 70.4 MB) and the atomics are the floor under the `[M, hidden]` contract.
8. **The gemm1 pad-skip leak, measured and closed** (r4_d2).
   - Mechanism, from per-CTA timelines (`FBS_DBG_TS`):
     - XCD == linear CTA id % 8 for 100% of CTAs.
     - A full CTA takes 80-83K cycles at any clock (issue/MFMA-bound).
     - Unpaced 1- and 2-subtile tails run 0.71-0.80x as long and finish about 14 us before their siblings. They widen
       the median same-panel sibling start spread from 2 us to 6-10 us: in the one-shot grid, an early exit perturbs
       the dispatch rhythm.
     - The 35 spills explain about 0% of the tpr2048 miss excess and about 30% at tpr4096, but all of the +15% EA writes.
   - The repair (r4_d2 patch):

     | knob | what it does |
     |---|---|
     | `FBS_PS_PRO=1` | branch before the prologue: 0 spills, 246 VGPR, full body = the no-skip census |
     | `FBS_PS_SLEEP=8,12` | `s_sleep` per K tile in the 2- and 1-subtile tails; tails run 0.85-0.9x of a full CTA |
     | `FBS_PADZERO=1` | padding rows gather X out of range as zeros; at most +0.15% alone |

     Results: L2-miss excess -33% / -55%; EA writes back to the no-skip level; forward +0.25% (15/18); gemm1 trace -0.1% / -0.2%.
   - Pacing harder does not buy the rest. A global self-timed speed limit removed the whole excess but cost 0.1-2.0% of
     the forward. About 1-1.4M MALL-hit sibling re-fetches (130-180 MB) are worth less than any enforced pace costs.
     **Close this lane.**
   - Inferred, not measured: s2's grid loop absorbs its tails inside one CTA, which would explain why its skip converted
     1:1.
9. **W13 L2 sharing is load-bearing; cache-policy bits are not a lever** (r4_d2, in-process forward A/B, 9 rounds).
   - nt on W13 loads cost 24.4% / 19.9% of the FORWARD; nt on X loads cost 7.0% / 6.9%.
   - W sc0 / sc1, X sc1 and a2/a2s store nt or sc0|sc1 all stayed within ±0.4% (null +0.23% / -0.16%).
   - Never weaken sibling-CTA reuse of W panels.
10. **A subtile skip is not a block deletion** (R3, still holds).
    - The 4,2,1 skip removes MFMA work, but each CTA still streams its full W panel (about 1 MB), its X rows and the
      epilogue.
    - Residual padding is 2.29% / 1.29% of MFMA rows, mostly partial subtiles.
    - Only 4,2,1 keeps s2's full-body ISA. Run `isa_cmp.py` / `body_census.py` before timing any variant set.
11. **The gemm1 recipe** (R3, default-ON):
    - `FBS_RS=1`: scalar rescale, 0 pk_fma, 1 FMA per MFMA (irreducible);
    - `FBS_WPE=2`;
    - `FBS_G1_PRIO=1`: setprio around the WHOLE K loop. setprio is a scheduling-region boundary; per-tile placement
      lost 5%;
    - `FBS_EPI_MAXNUM=1`;
    - `FBS_PADSKIP=4,2,1`.

    Dead ends: `FBS_SCHED=5`, BPF / DEFER / SCHED_G1 retunes, `FBS_PRO`, MF32 (-3.9% / -5.3%).

    Wrong-output ceilings (share of gemm1):

    | what-if | gemm1 |
    |---|---|
    | `DBG_BK0` (W always L2-hot) | +14.9% / +10.0%, via clock |
    | `DBG_NOSCALE` | +6% |
    | `DBG_NOEPI` | +5.3% / +3.8% |
    | `DBG_NOBAR` | +1.7% / +0.8% |
12. **The only untested gemm1 bytes lever: a critical-path-free expert-run / XCD-affine CTA order.**
    - R2's expert-run order (gn=16, ord=3) cut gemm1 L2 misses 25.5% / 9.5%. It lost 1.4% only because its dependent
      seid load + ballot + division prologue cost 2-3.5% of stage1.
    - R3 showed that an `s_load` of sort-emitted per-block metadata is free. A `k_place`-emitted per-CTA
      (m_block, n_tile) table read by `s_load` would remove that prologue.
    - My estimate: about 0 to +0.7% of the forward (unmeasured).
    - Cautions:
      - MALL-hit re-fetches turned out cheap (r4_d2);
      - the order changes the dispatch rhythm that W sharing depends on;
      - it must keep `blk_cnt` / pad-skip semantics and add no device loop.
13. **Grid/loop asymmetry** (R1/R2).
    - Never put a device loop around the gemm1 body (-4.5%). Its empty CTAs are free.
    - Keep s2's degenerate grid loop. Its full if-guarded grid lost 3.6% / 1.4%.
14. **Fill and sort** (R4).
    - k_count + k_place take 40 / 46 us in the forward. r4_d1's isolated sort micro-bench (k_count + k_place):
      24.6 / 24.7 us without a fill, 25.4 / 36.9 with R3's fill, 27.3 / 26.8 with the multi-only fill.
    - In the forward the sort runs 13-19 us slower than isolated. Likely cause (inferred): the fill pays real HBM writes
      there.
    - Closed: fill placement variants, the grid-barrier sort, `dispatch_policy=1`, opus sorting.
    - k_place loops must stride by `sort_grid`.
15. **Wrapper-level graph capture is CLOSED** (r4_d1).
    - `torch.cuda.CUDAGraph` per input-pointer key: -0.37% geo (graph won 0/6 and 2/6); in-process -0.16%.
    - There are no host gaps to remove. **Stop gate (a) is met.**
    - Reference: `round_4/engineer_1/graph_capture_NOT_A_CANDIDATE.diff`.
16. **Integration R4.**
    - The R4 region ownership split one file: r4_d0 owned the s2 K-loop rescale, and r4_d1 owned the s2 epilogue, the
      sort and the wrapper.
    - `git apply` rejected 3 adjacent hunks: the knob block, the compile signature and the compile kwargs. Keeping both
      sides resolved them.
    - Per-change lru-keyed knobs made the merge checkable. Merged `S2_RS=0` is byte-identical to r4_d1's s2, and 6
      single-knob fallbacks PASS.
    - The patches added about linearly.
17. **Profile** (R3 PMC; R4 had no reprofile).
    - gemm1: MFMA-busy 65 / 74%; issue-wait 49%; L2 hit 73.4 / 79.6%; EA fetch 1502 / 2214 MB.
    - s2: MFMA-busy 55 / 66%; L2 hit 73.9 / 78.8% after r4_d1.
    - Both: EA at 35-40% of peak, VMEM about 3%, LDS 26-33%, I-cache 99.9% or more.
    - My arithmetic, not measured:
      - gemm1's EA fetch is about 2.5x / 3.3x its compulsory bytes (W13 537 MB plus gathered X 67 / 133 MB). That fits X
        fetched by all 8 XCDs plus W13 re-fetched about 2x.
      - s2's 600 / 860 MB fits a2 read once per XCD plus W2 (268 MB) once.
18. **Still closed (R1-R4):**
    - 128-row stage1; gemm1 device loop; s2 de-loop; gemm1 plain XCD tiling and all s2 XCD tiling;
    - a 3rd wave; stage1->stage2 fusion; FP32 packing near MFMAs; VALU-slot scheduling and nop removal;
    - per-tile setprio; MF32; more skip variants; gemm1 skip pacing beyond 8,12 and global lockstep;
    - gemm1 cache-policy bits; s2 address-VALU trims (BIMM, soffset); PRO_SC; in-asm execz skips;
    - wrapper graph capture; fill placement variants; sort `dispatch_policy=1`.

    Confirm `moe_fp8bs_gemm1_0` and `moe_s2_fp8bs_0` by name in any trace.
19. **Measurement.**
    - Cross-session drift is about 1%, and GPU3 runs about 2% slower. Only same-session paired ratios count.
    - The protocol that resolved 0.2-0.3% effects in R4:
      - 18-30 interleaved COMMANDMENT rounds, with rotating arm order, under one lease per session;
      - pooled over 2-3 GPUs;
      - a null arm (within ±0.05-0.25%).
    - Run every A/B from a FROZEN snapshot. FlyDSL traces source at first call, and editing during an A/B corrupts it.
    - Verifier offset vs engineer-pooled: r4_d1 -0.24%, r4_d0 +0.13%.
    - Tools:
      - `round_4/integrate/tools/{abl.sh, ab_fwd.py, trace_ab.sh, trace_kern.py, summ.py, ext_correct.py,
        bitexact_compare.py, body_census.py, isa_diff.py}`;
      - `round_4/engineer_2/tools/tsdump.py` (per-CTA timelines);
      - `profile_output_r3_pmc/tools/{power_ab.sh, l2_attrib.sh}`.
20. **Process.**
    - r4_d2 never reached the verifier. After its decisive 8-round A/B (done 18:07) it ran a second 10-round A/B until
      18:19. It wrote its result at 18:20, but integrate had started at 18:15, so it was recorded `apply_failed`. This is
      the R1 failure mode again.
    - **Return the moment the decisive same-session A/B is in.**
    - What kept working: `checkpoints.log`, default-ON `best_patch.diff`, knob-off = R3 ISA, fresh-tree `patch -p1`, and
      `NOT_A_CANDIDATE` diffs.
21. **Expert skills** (`index.yaml` unchanged since Nov 18).
    - The only kernel-scope validated gfx942 / fp8 / fused_moe_grouped_gemm match is `flydsl_rewrite_quantized_moe`,
      a triton->flydsl recipe. There is nothing to reproduce.
    - Its live constraint is the `[M, hidden]` in-kernel accumulate contract. r4_d1 complies: plain stores go into the
      same `[M, hidden]` buffer, and the only new buffer is a per-token byte flag.
    - The blkmap and stage2_fp8partial skills are gfx950-only.

### Hypothesis ledger (cumulative, deduplicated; "actual" = same-session ratio vs canonical unless noted)
| direction | specialty | expected | actual | verdict | lesson |
|---|---|---|---|---|---|
| r1_d0 | memory | 1.08 | 0.985 | dead_end | The 128-row s1t ties gemm1, and unit-128 padding costs more. Never re-fund |
| r1_d1 | host_runtime | 1.035 | 1.01 | partial | Fast sort + fill; shipped in R2 as r2_d1 |
| r1_d1:gemm1_grid_loop | host_runtime | 1.035 | 0.955 | dead_end | Never put a device loop in gemm1 |
| r1_d1:sort_dispatch_policy | host_runtime | 1.035 | 0.35 | dead_end | The single CK MoeSortingKernel takes 1.7-3.9 ms; opus = CK |
| r1_d2 | compute | 1.04 | n/a | dead_end | A 3rd s2 wave is not reachable by VGPR trim |
| r2_d0 | memory | 1.03 | 0.9988 | dead_end | Plain XCD tiling trades X misses for W refetch |
| r2_d0:expert_run_order | memory | 1.03 | 0.9863 | dead_end | -25.5% / -9.5% misses, killed by its prologue. The sort-emitted s_load table variant is the only untested remap |
| r2_d1 | host_runtime | 1.012 | 1.0065 (verified 1.5535) | confirmed | Shipped fast sort + sort-owned fill |
| r2_d1:fill_placement | host_runtime | 1.012 | 1.0 | dead_end | Fill cost tracks bytes, not placement |
| r2_d2 | compute | 1.02 | 1.0182 (verified 1.5629) | confirmed | Shipped the s2 recipe |
| r2_d2:s2_deloop | compute | 1.02 | s2 0.975 | dead_end | Keep s2's loop |
| r2_d2:s2_xcd_tiling | compute | 1.02 | s2 0.989 | dead_end | s2 needs the plain grid |
| r2_integrate | integrate | 1.025 | 1.028 (verified 1.5710) | confirmed | Disjoint files add; k_place must stride by sort_grid |
| r3_d0 | compute | 1.02 | 1.0278 (verified 1.6192) | confirmed | Scalar-FMA rescale, whole-loop setprio, maxnumf; about 45% cycle-to-time conversion |
| r3_d0:packed_rescale | compute | 1.02 | inverted | dead_end | Never pack FP32 near MFMAs on gfx942 |
| r3_d0:valu_slot_sched | compute | 1.01 | ~1.00 | dead_end | Stall cycles cost no energy at the cap |
| r3_d0:per_tile_setprio | compute | 1.006 | gemm1 0.949 | dead_end | setprio is a sched-region boundary |
| r3_d0:mf32_and_retune | compute | 1.01 | gemm1 0.954 / noise | dead_end | Defaults stay SCHED=3, G1=8, DEFER=1, BPF=2 |
| r3_d1 | algorithm | 1.01 | 1.0071 (verified 1.5814) | confirmed | Sort-emitted blk_cnt + uniform branch; s2 skip converts 1:1 |
| r3_d1:s2_variant_sets | algorithm | 1.01 | below 4,2,1 | dead_end | Other sets perturb the full body's register allocation |
| r3_d1:gemm1_padskip | algorithm | 1.016 | 1.0031 | partial | Shipped; leak measured and partly repaired by r4_d2; the rest is dispatch rhythm |
| r3_d1:s2_rescale_packing | algorithm | 1.01 | not built | dead_end | Inverted; the scalar form (r4_d0) shipped instead |
| r3_integrate | integrate | 1.03 | 1.0354 (verified 1.6273) | confirmed | Committed 31c4cb1, the current canonical |
| r4_d0 | compute | 1.008 | ~1.002 (verified 1.629) | partial | s2 pk 457 -> 0 and s2 -1.6% / -1.9%, but the next gemm1 gives it back. +0.26% in combination. The s2 scale-math lane is exhausted |
| r4_d0:prologue_pk_fence | compute | 1.002 | ~1.00 | dead_end | llvm.arithmetic.fence defeats SLP re-packing, but pk outside the K loop costs nothing |
| r4_d0:b_imm_offset | compute | 1.003 | ~1.00 | dead_end | -92 VALU per tile is noise: address VALU is not an energy lever |
| r4_d1 | algorithm | 1.01 | 1.0133-1.0162 (verified 1.6462) | confirmed | Bytes convert 1:1: fill 89 -> 36 / 177 -> 70 MB, EA atomics -41%. nt plain stores. Multi rows are the contract floor |
| r4_d1:graph_capture | host_runtime | 1.0 | 0.9963 | dead_end | No host gaps; replay slightly slower. Stop gate (a) met |
| r4_d1:execz_skip | compute | 1.002 | ~1.00 (+0.30% vs null +0.24%) | dead_end | Skipping dead exec=0 memory ops is free already |
| r4_d2 | memory | 1.01 | 1.0025 (not verified) | partial | 0 spills, -33% / -55% miss excess, +0.25% (15/18). Carry into R5 |
| r4_d2:pacing_and_lockstep | memory | 1.01 | 0.98-0.999 (lockstep) / = 8,12 (self-timed) | dead_end | The excess is dispatch rhythm; about 1M MALL-hit misses are worth less than any enforced pace |
| r4_d2:cache_policy | memory | 1.005 | W nt 0.76-0.80, X nt 0.93, others ±0.4% | dead_end | W13 L2 sharing is worth about 20% of the forward; policy bits are not a lever |
| r4_integrate | integrate | 1.02 | 1.016 vs R3 (1.6505) | confirmed (not committed) | +1.43% over cumulative is below the 2% commit gate; R5 must carry it |

### Bottleneck now
**Compute:** fp8 MFMA-bound at the 750 W cap. The sustained clock is about 0.97-1.0 GHz, and time per forward tracks energy per forward.
- The R4 stack (uncommitted) is 1.044 / 1.892 ms: gemm1 about 62% / 63%, s2 34%, sort 3.8% / 2.4%. 4 dispatches, no host gaps.
- s2 is at its floors: 0 pk_fma, irreducible scale math, and epilogue bytes at the `[M, hidden]` contract floor.
- gemm1 has 0 pk_fma and 1 irreducible FMA per MFMA. W13 L2 sharing is load-bearing. Its EA fetch is about 2.5-3.3x the
  compulsory bytes (estimate).

Energy left:
- gemm1 W/X re-fetches (the table-driven expert-run order is untested);
- the gemm1 epilogue (wrong-output ceiling +5.3% / +3.8% of gemm1);
- residual partial-subtile padding (2.29% / 1.29% of MFMA rows).

### Suggested next (round 5, the last: 1 direction, no integrate step, commit gate verified > 1.6598)
**One bank-and-extend direction.** focus_files: `flydsl_moe_fp8bs.py`, `flydsl_moe_fp8bs_s2.py`, `moe_ep.py`,
`moe_sort_fast.py`.
1. **Build the stack.** `patch -p1` both `round_4/integrate/integrated_patch.diff` and `round_4/engineer_2/best_patch.diff` onto
   a fresh copy of canonical `31c4cb1`.
   - Correctness: re-run COMMANDMENT, `ext_correct.py` 24/24 and every fallback knob.
   - ISA gates: s2 234 VGPR, 0 spills, 0 pk_fma; gemm1 246 VGPR, 0 spills.
   - A/B: same-session vs `31c4cb1`; expect about +1.6% to +1.9%.
   - Checkpoint the stack as `best_patch.diff` right away.
2. **Add ONE increment worth at least +0.5%.** Ship it only on a same-session win: 10 or more interleaved rounds plus a null
   arm, reading the next kernel's time.
   - Preferred: the table-driven expert-run / XCD-affine gemm1 CTA order.
     - `k_place` emits a per-CTA (m_block, n_tile) table next to `blk_cnt`; gemm1 `s_load`s it.
     - No dependent vector load, no ballot, no device loop.
     - Judge with `l2_attrib.sh` plus a forward A/B.
   - Alternative: trim gemm1 epilogue ops and bytes. Only removing ops or bytes converts.
3. **Return early.** Leave the verifier time (r4_d2's lesson). The final `best_patch.diff` must be relative to `31c4cb1`
   and contain the whole stack.

Stop gates:
- (a) is MET: graph capture -0.37%.
- (c) is MET: the best verified gain was +1.16% individual / +1.43% integrated, under 3%.
- (b) is formally unmet: both cases are compute-bound at 1-2 ms.

The budget remains and R4's gain is unbanked, so **issue the last direction; do not stop.**

## Round 5 — update_memory (the last round)

**Outcome: the run ends at the committed R3 result. The best verified candidate is not banked.**
- r5_d0 shipped the R4 stack alone: `round_5/engineer_0/best_patch.diff`, sha256 `98b5a13b...`, 4 files, relative to
  `31c4cb1`. It is byte-identical to `tech_lead_r5/r4_stack_vs_31c4cb1.diff`.
- The verifier measured **1.6575x** on GPU0. That is **0.14% under** the commit gate (1.6273 x 1.02 = 1.6598), so
  IMPROVED = false and nothing was committed.
- Same session, the stack beats `31c4cb1` by +1.76% (paired median, 4/4 wins) to +1.92% (pooled medians: 1.0365 / 1.8902
  ms vs 1.0568 / 1.9255 ms). The engineer measured +1.85% (12/12).
- Correctness:
  - COMMANDMENT PASS: rel_l2 2.315e-3 / 2.307e-3;
  - `ext_correct` 24/24;
  - verifier stress PASS: `sort_single` and `s2_sstore` engaged 24/24, with no `fused_moe` fallback.
- The increment meant to clear the gate was the table-driven gemm1 CTA order. Its best variant measured +0.04%, so it
  was not shipped.
- Run state: budget 12/12 used (R1 3, R2 3, R3 2, R4 3, R5 1) and noImprove 2/2. The loop ends, and the report goes out
  with canonical `31c4cb1` = 1.6273x (R3 reprofile FULL_BENCHMARK x3: 1.6256x, 1.0603 / 1.9207 ms).

### Per-direction
| id | specialty | strategy | expected | same-session measured | verified | verdict |
|---|---|---|---|---|---|---|
| r5_d0 | memory | bank the R4 stack (R4 merge + r4_d2); add an expert-run, XCD-affine gemm1 CTA order from a per-CTA table that k_place emits and gemm1 reads with one `s_load` | 1.025 | stack vs `31c4cb1` +1.85% (12/12); the order increment +0.04% at best (gated, 8/14) | 1.6575 (verifier same-session +1.76% to +1.92%) | partial: the stack is confirmed, the increment is a dead end, nothing committed (gate) |

Inside the stack, same session (S1 GPU0 + S2 GPU1, 12 interleaved rounds):
- R4 merge vs `31c4cb1`: +1.47% (12/12).
- Full stack vs `31c4cb1`: +1.85% (12/12); tpr2048 +1.96%, tpr4096 +1.97%.
- r4_d2's gemm1 knobs: +0.36% inside the stack (10/12), against +0.25% alone in R4.

CTA-order sub-experiments, all closed. Each is paired against `FBS_ORD=0` and run interleaved from frozen snapshots on 2-4
GPUs.

| variant | result vs ORD=0 | reading |
|---|---|---|
| expert-run order, ungated | about -0.2% (ord0 won 13/16 to 15/18; tpr4096 -0.4%) | the order gains about 0; the mechanism costs about 0.3% |
| identity table, ungated | about -0.3% (ord0 won 12/13) | the mechanism cost: about 65K empty CTAs `s_load` cold, never-written table lines |
| expert-run order, gated (`FBS_ORD_GATE=1`) | +0.04% (8/14; tpr2048 +0.27%, tpr4096 -0.14%) | below the ship rule set before the A/B (at least +0.2%, at least 65% wins, no case below -0.3%) |
| identity table, gated | -0.06% (4/11) | the gate removes most of the mechanism cost |
| run chunk cap 8 (`FBS_ORD_C=8`) | -0.83% (0/12; tpr4096 -1.2%) | re-streaming W per chunk costs far more than the X reuse it buys |
| exactly balanced XCD partition | -0.30% (5/24; tpr4096 -0.44%) | the tpr4096 loss is not caused by XCD imbalance |

PMC data: rocprofv3 TCC counters on GPU3 for the balanced-order build. These are my re-summary of `round_5/engineer_0/l2`
(medians per dispatch group):

| case | gemm1 L2 misses (ord0 -> ord1) | gemm1 EA read requests | gemm1 GRBM_GUI_ACTIVE | s2 |
|---|---|---|---|---|
| tpr2048 | 11.6M -> 9.4-9.9M (hit 74% -> 78-79%) | 11.3-11.5M -> 9.1-9.6M | 5.21-5.24M -> 5.22-5.38M | unchanged |
| tpr4096 | 17.1-17.2M -> 16.6-17.5M | 16.6-16.7M -> 16.1-17.0M | 9.56-10.12M -> 9.79-10.51M | unchanged |

- EA write requests are identical across arms.
- The identity order (ord2) matches ord0 on every counter.
- So 15-19% fewer misses at tpr2048 bought no reduction in active cycles.

### Insight blackboard (supersedes the R4 list)
1. **Final state.** Canonical stays `31c4cb1` (R3, cumulative 1.6273x).
   - The run's best verified candidate is uncommitted: `round_5/engineer_0/best_patch.diff`, the R4 merge + r4_d2.
     - It verified at 1.6575x.
     - Same session it is +1.76% to +1.92% over `31c4cb1`.
     - Every correctness check passes, and `git apply --check` is clean on `31c4cb1`.
   - **It is ready to bank in any follow-up run, or by hand.**
2. **Why the win was lost.**
   - The gate compares one session's absolute geomean with a cross-session cumulative, while this kernel's directions
     return 0.2-1.6% each. R4 (+1.43%) and R5 (stack +1.8% to +1.9%) both landed under +2%.
   - Absolute stack geomean by GPU:

     | GPU | geomean | n |
     |---|---|---|
     | GPU0 | 1.6571x | 21 |
     | GPU1 | 1.6663x | 17 |
     | GPU2 | 1.6568x | 19 |
     | GPU3 | 1.6222x | 12 |

     Only GPU1 clears 1.6598, and the verifier drew GPU0.
   - The verifier's offset from the engineer's pooled number was -0.24%, the same as r4_d1's.
   - **Planning lesson.** Once per-direction gains fall to about 1%, bank a sub-gate win with margin:
     - keep 2 or more directions for the final round, so that integrate can stack two increments;
     - or reach the last round with the stack at least 0.5% above the gate.

     A one-direction last round with a near-gate stack is a coin flip on the verifier's GPU.
3. **The gemm1 CTA-order lane is closed** (`order_NOT_A_CANDIDATE.diff`, which applies on `31c4cb1`).
   - How it works: k_place writes one packed i32 per linear gemm1 CTA (`m | n<<13 | cnt<<17 | expert<<24`) into a
     separate table, passed through gemm1's `arg_blk_cnt` slot. gemm1 `s_load`s its entry next to `num_valid`, with no
     loop, vector load or ballot.
   - It is bit-exact: a2 and a2-scale are identical across ORD 0/1/2, and the table is a bijection and XCD-affine.
     ORD=0 is byte-identical to the stack; ORD=1 uses 242 VGPR with 0 spills.
   - Best result: +0.04% (gated). Every other variant lost: chunked -0.83%, balanced -0.30%.
   - With R2's prologue-bound version (-1.4%), every gemm1 remap is now closed:
     - plain XCD tiling;
     - the gn=16 M-partition;
     - the expert-run order, with a prologue or with a table;
     - chunked runs;
     - the balanced partition.
4. **MALL-served misses do not convert.**
   - At tpr2048 the order removed 15-19% of gemm1 L2 misses: about 2.2M EA read requests, roughly 140 MB at 64 B per
     request (my arithmetic).
   - It cost no energy saving: active cycles were flat to +3%, and the forward gained at most +0.27%.
   - At tpr4096, about 16-block expert runs put about 4 MB of X on each XCD, the whole L2, so misses moved only -2.4% to
     +1.6%.
   - The removed misses are cross-XCD X re-fetches that MALL already serves. This is inferred; it matches R2's small
     X-hot ceiling and r4_d2's cheap MALL-hit sibling re-fetches.
   - **Refined energy rule at the 750 W cap:**

     | what is removed | share converted to forward time |
     |---|---|
     | MFMA work | about 1:1 |
     | HBM writes and memory-side atomics | about 1:1 (r4_d1) |
     | stall / VALU cycles, first kernel | about 45% |
     | stall / VALU cycles, second kernel | about 0 |
     | L2 misses served by MALL | about 0 |

     **Count bytes at HBM, not at L2.**
5. **Empty gemm1 CTAs are free only while they touch nothing new.**
   - One extra per-CTA `s_load` from a k_place-written table cost about 0.3% of the forward. The about 65K empty CTAs (of
     69,632) each loaded a cold, never-written line.
   - Bounding the load by `num_valid` first, so that out-of-region CTAs read entry 0, cut the cost to about 0.06%.
   - Grid-shape changes need a forward-level A/B: the gemm1 device loop lost 4.5%, s2 de-loop 3.6%.
6. **W13 sharing dominates X reuse.**
   - Chunked runs re-stream W and lost 0.83% (0/12). With R4's results (nt on W13: -24.4% / -19.9% of the forward; nt on
     X: -7%), sibling-CTA W13 reuse in L2 is worth about a fifth of the forward.
   - Never weaken it. Cache-policy bits are not a lever.
7. **The stack adds linearly** (see the Per-direction section).
   - ISA: gemm1 246 VGPR, 0 spills; s2 234 VGPR, 0 spills, 0 `v_pk_fma_f32`.
   - Defaults ON: `S2_RS=1`, `S2_RS_NV=2`, `MOE_SSTORE=1`, `S2_SST_POL=1`, `FBS_PS_PRO=1`, `FBS_PS_SLEEP=8,12`,
     `FBS_PADZERO=1`. Each knob at 0 restores its R3 ISA byte-identically.
8. **Cost structure** (R4 merge integrate trace, rocprofv3 medians; the R5 stack adds r4_d2's gemm1 -0.1% / -0.2%):

   | case | k_count | k_place | gemm1 | s2 | kernel sum |
   |---|---|---|---|---|---|
   | tpr2048 | 6.8-7.5 us | 32.5-32.9 us | 643.5-647.8 us | 352.3-354.1 us | 1036-1042 us |
   | tpr4096 | 6.8-7.5 us | 38.4-39.0 us | 1190-1201 us | 644.0-649.8 us | 1880-1897 us |

   - 4 dispatches, span equal to the kernel sum, no host gaps.
   - tpr2048 stays the weaker case: the stack is 1.618x vs 1.706x.
9. **gfx942 packed-FP32 rule.**
   - `V_PK_*_F32` waits for the matrix pipe to drain: MFMA + 1 pk FMA = 30.1 cycles, vs 17.1 for MFMA + 2 scalar FMAs.
   - Emit rescales as per-element scalar `llvm.fma`. Use `llvm.arithmetic.fence` against SLP re-packing.
   - pk ops outside the K loop are neutral.
10. **s2 is at its floors.**
    - 0 pk_fma, and 1 scalar FMA per MFMA, which is irreducible bit-exactly.
    - The epilogue bytes are at the `[M, hidden]` contract floor: multi-row fill 35.7 / 70.4 MB plus 1.27M / 2.50M
      memory-side bf16 atomics.
    - Address-VALU trims are noise. The plain grid with its degenerate loop is required.
11. **Single-local-expert plain stores** (r4_d1, in the stack).
    - Single rows get nt `global_store`, multi rows get `global_atomic_pk_add_bf16`, padding rows get nothing. exec is
      built in asm from `readfirstlane(kind)`.
    - Results: fill writes -60%, s2 EA atomics -41%.
    - Every bf16 device atomic is a memory-side read-modify-write.
12. **gemm1 pad-skip** (closed).
    - Subtile skips leave W/X streaming and the epilogue in place.
    - Early-exiting tails desync same-panel siblings in the one-shot grid.
    - r4_d2's 0-spill prologue + `s_sleep` 8,12 + PADZERO cut the miss excess by a third to a half. The rest is not
      worth any pace.
    - Residual padding is 2.29% / 1.29% of MFMA rows.
13. **gemm1 recipe** (R3): `FBS_RS=1`, `FBS_WPE=2`, `FBS_G1_PRIO=1` (whole-loop setprio), `FBS_EPI_MAXNUM=1`,
    `FBS_PADSKIP=4,2,1`.

    Wrong-output ceilings, as a share of gemm1:

    | what-if | gemm1 |
    |---|---|
    | `DBG_BK0` | +14.9% / +10.0% |
    | `DBG_NOSCALE` | +6% |
    | `DBG_NOEPI` | +5.3% / +3.8% (includes the compulsory a2 stores) |
    | `DBG_NOBAR` | +1.7% / +0.8% |
14. **Fill and sort.**
    - k_count + k_place take 40 / 46 us in the forward, vs about 27 us isolated with the multi-only fill. The gap is
      inferred to be the fill's real HBM writes.
    - Closed: placement variants, the grid-barrier sort, `dispatch_policy=1`, opus sorting.
    - k_place loops must stride by `sort_grid`.
15. **Wrapper graph capture is closed** (-0.37%). There are no host gaps. Stop gate (a) is met.
16. **Profile.**
    - gemm1: MFMA-busy 65% / 74%; L2 hit 74% / 80%; per call 11.6M / 17.2M misses and 0.57M / 1.12M EA write requests.
    - s2: MFMA-busy 55% / 66%; L2 hit 74% / 79%.
    - EA bandwidth is 35-40% of peak and LDS 26-33%: not memory-, LDS- or launch-bound.
    - gemm1's EA fetch is about 2.5x / 3.3x the compulsory bytes (my arithmetic). Most of the excess is MALL-served
      re-fetch (R5).
17. **Still closed (R1-R5):**
    - every gemm1 CTA remap (item 3); all s2 XCD tiling;
    - the 128-row stage1; the gemm1 device loop; s2 de-loop;
    - a 3rd wave; stage1->stage2 fusion;
    - FP32 packing near MFMAs; VALU-slot scheduling and nop removal; per-tile setprio; MF32;
    - more skip variants; pacing beyond 8,12;
    - cache-policy bits; address-VALU trims; PRO_SC; in-asm execz skips;
    - graph capture; fill placement; sort `dispatch_policy=1` and opus sorting.
18. **What is left** (all estimates, none measured above +1%):
    - (a) bank the verified stack (+1.8% to +1.9%);
    - (b) gemm1 epilogue op and byte removal;
    - (c) sort + fill HBM bytes;
    - (d) residual partial-subtile padding.

    The kernel is near its energy floor for this algorithm at the 750 W cap.
19. **Measurement.**
    - Cross-session drift is about 1%, and the absolute geomean varies up to 2.7% between GPUs. Only same-session paired
      ratios count.
    - Protocol: 12-30 interleaved rounds, pooled over 2-4 GPUs, with a null arm and frozen snapshots.
    - Verifier offset vs engineer pooled: r4_d1 -0.24%, r4_d0 +0.13%, r5_d0 -0.24%.
    - New tools: `round_5/engineer_0/tools/{summn.py, ord_check.py, l2_ord.sh, pergpu.py, final_check.sh}`.
20. **Process.** R5 fixed R4's late-return failure.
    - `best_patch.diff` was on disk at minute 6, every step went to `checkpoints.log`, the GPUs stopped at minute 36, and
      the verifier finished.
    - The tech-lead-prepared stack diff let the engineer bank the carry-over in 6 minutes.
    - Verified directions by round: R1 0/3, R2 2/3, R3 2/2, R4 2/3, R5 1/1.
21. **KB defect (not edited here; out of scope).** `kernel_workflow/knowledge/amd_instinct.md` line 49 says the L2 is
    "shared across CUs (256 MB on MI300X-class)".
    - On this gfx942 box, rocminfo reports L2 = 4096 KB (one per XCD, 8 XCDs) and L3 = 262144 KB. The 256 MB is the
      device-wide Infinity Cache (MALL).
    - Items 3, 4, 6 and 12 depend on that split. The maintainer should fix the file.
22. **Expert skills** (`index.yaml` unchanged since Nov 18 11:22).
    - `flydsl_rewrite_quantized_moe` is the only gfx942 match. It is a triton->flydsl recipe, so there is nothing to
      reproduce.
    - Its `[M, hidden]` in-kernel accumulate contract is respected by the stack.

### Hypothesis ledger (cumulative, deduplicated; "actual" = same-session ratio vs canonical unless noted)
| direction | specialty | expected | actual | verdict | lesson |
|---|---|---|---|---|---|
| r1_d0 | memory | 1.08 | 0.985 | dead_end | The 128-row s1t ties gemm1, and unit-128 padding costs more |
| r1_d1 | host_runtime | 1.035 | 1.01 | partial | Fast sort + fill; shipped in R2 as r2_d1 |
| r1_d1:gemm1_grid_loop | host_runtime | 1.035 | 0.955 | dead_end | Never put a device loop in gemm1 |
| r1_d1:sort_dispatch_policy | host_runtime | 1.035 | 0.35 | dead_end | The single CK sort kernel takes 1.7-3.9 ms |
| r1_d2 | compute | 1.04 | n/a | dead_end | A 3rd s2 wave is unreachable |
| r2_d0 | memory | 1.03 | 0.9988 | dead_end | Plain XCD tiling trades X misses for W refetch |
| r2_d0:expert_run_order | memory | 1.03 | 0.9863 | dead_end | Lost to its prologue; R5's table variant closed the lane |
| r2_d1 | host_runtime | 1.012 | 1.0065 (verified 1.5535) | confirmed | Shipped the fast sort + sort-owned fill |
| r2_d1:fill_placement | host_runtime | 1.012 | 1.0 | dead_end | Fill cost tracks bytes, not placement |
| r2_d2 | compute | 1.02 | 1.0182 (verified 1.5629) | confirmed | Shipped the s2 recipe |
| r2_d2:s2_deloop | compute | 1.02 | s2 0.975 | dead_end | Keep s2's loop |
| r2_d2:s2_xcd_tiling | compute | 1.02 | s2 0.989 | dead_end | s2 needs the plain grid |
| r2_integrate | integrate | 1.025 | 1.028 (verified 1.5710) | confirmed | Disjoint files add |
| r3_d0 | compute | 1.02 | 1.0278 (verified 1.6192) | confirmed | Scalar-FMA rescale and whole-loop setprio |
| r3_d0:packed_rescale | compute | 1.02 | inverted | dead_end | Never pack FP32 near MFMAs on gfx942 |
| r3_d0:valu_slot_sched | compute | 1.01 | ~1.00 | dead_end | Stall cycles cost no energy at the cap |
| r3_d0:per_tile_setprio | compute | 1.006 | gemm1 0.949 | dead_end | setprio is a scheduling-region boundary |
| r3_d0:mf32_and_retune | compute | 1.01 | gemm1 0.954 | dead_end | MF32 loses; the retunes are noise |
| r3_d1 | algorithm | 1.01 | 1.0071 (verified 1.5814) | confirmed | Sort-emitted blk_cnt + subtile skip |
| r3_d1:s2_variant_sets | algorithm | 1.01 | below 4,2,1 | dead_end | Other sets perturb register allocation |
| r3_d1:gemm1_padskip | algorithm | 1.016 | 1.0031 | partial | Leak measured; partly repaired by r4_d2 |
| r3_d1:s2_rescale_packing | algorithm | 1.01 | not built | dead_end | Inverted; the scalar form shipped as r4_d0 |
| r3_integrate | integrate | 1.03 | 1.0354 (verified 1.6273) | confirmed | Committed `31c4cb1`: the FINAL canonical |
| r4_d0 | compute | 1.008 | ~1.002 (verified 1.629) | partial | s2 pk 457 -> 0; the next gemm1 gives it back; +0.26% in combination |
| r4_d0:prologue_pk_fence | compute | 1.002 | ~1.00 | dead_end | pk outside the K loop is free |
| r4_d0:b_imm_offset | compute | 1.003 | ~1.00 | dead_end | Address VALU is not an energy lever |
| r4_d1 | algorithm | 1.01 | 1.0133-1.0162 (verified 1.6462) | confirmed | HBM bytes convert 1:1 |
| r4_d1:graph_capture | host_runtime | 1.0 | 0.9963 | dead_end | No host gaps |
| r4_d1:execz_skip | compute | 1.002 | ~1.00 | dead_end | exec=0 memory ops are already free |
| r4_d2 | memory | 1.01 | 1.0025 alone; +0.36% inside the stack (10/12) | confirmed (inside the R5 stack, verified 1.6575, uncommitted) | 0 spills; the residual is dispatch rhythm |
| r4_d2:pacing_and_lockstep | memory | 1.01 | 0.98-0.999 | dead_end | No enforced pace pays |
| r4_d2:cache_policy | memory | 1.005 | W nt 0.76-0.80 | dead_end | W13 sharing is worth about a fifth of the forward |
| r4_integrate | integrate | 1.02 | 1.016 (1.6505) | confirmed, not committed | Below the 2% gate; carried into R5 |
| r5_d0 | memory | 1.025 | 1.0185 (verified 1.6575) | partial, not committed | Stack confirmed, 0.14% under the gate; the increment failed |
| r5_d0:cta_order_table | memory | 1.005 | 1.0004 gated (8/14) | dead_end | -15% to -19% gemm1 misses were MALL hits; no cycles saved. Lane closed |
| r5_d0:cta_order_ungated | memory | 1.0 | ~0.997-0.998 | dead_end | Empty CTAs loading cold table lines cost about 0.3%; bound per-CTA loads |
| r5_d0:cta_order_chunk_balanced | memory | 1.003 | 0.9917 (chunk 8) / 0.997 (balanced) | dead_end | W sharing beats X reuse; the tpr4096 loss is not imbalance |

### Bottleneck now
**Compute:** fp8 MFMA-bound at the 750 W cap. Time per forward tracks energy per forward.
- Final, committed: `31c4cb1` at 1.6273x.
- Best verified but uncommitted: the stack at 1.6575x. Shares: gemm1 62% / 63%, s2 34%, sort 3.8% / 2.4%. 4 dispatches,
  no host gaps.
- s2 is at its floors.
- gemm1's excess fetch is mostly MALL-served, and R5 showed that does not convert.
- The energy left is small: the gemm1 epilogue, sort + fill bytes and residual padding.

### For the report and any follow-up run
- **The run is over** (budget 12/12, noImprove 2/2).
- Report `31c4cb1` = 1.6273x as the final result.
- In the report, list `round_5/engineer_0/best_patch.diff` as the verified-correct, unbanked stack:
  - 1.6575x verified;
  - +1.76% to +1.92% same-session over `31c4cb1`;
  - `git apply --check` is clean on `31c4cb1`.

  Recommend banking it first.
- A follow-up should then fund only HBM-byte or MFMA-op removal (the gemm1 epilogue, sort + fill bytes), each estimated
  at most +1%. Do not re-fund any CTA order, MALL-served miss work or anything else on the closed list.
