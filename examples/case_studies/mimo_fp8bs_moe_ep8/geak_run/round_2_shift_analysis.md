# Round 2 Reprofile — mimo_fp8bs_moe_ep8 (PHASE=reprofile, round-2 integrated winner)

## Device
AMD Instinct **MI300X / gfx942 / CDNA3**, 304 CU, wave64, 64 KB LDS/CU, fp8 = FNUZ (rocminfo + rocm-smi).
Socket power cap **750 W**, sclk max 2100 MHz.

## Which profiler ran, and the degrade
- **rocprof-compute failed** with the same error as the baseline and R0: its Python dependency check is
  missing 13 packages (plotext, textual, dash, ...). No `RPC_PROFILE_ARGS` override can fix a missing
  package, so I degraded on purpose to **rocprofv3**, as the COMMANDMENT PROFILE note expects.
- The COMMANDMENT PROFILE run used `--kernel-trace --stats`. The output is in
  `profile_output_r2/profile_report.txt`. The CSVs are one directory deeper than the wrapper's artifact
  list searches.
- Unlike R0, I also ran **six rocprofv3 `--pmc` passes** on the same `task_runner.py profile` workload:
  mfma, stall, l2, bytes, inst and lds. Output is in `profile_output_r2_pmc/`. Together these stand in for
  Speed-of-Light.
- I also ran these diagnostics:
  - a long steady-state kernel trace of `performance` mode (110 forwards per case);
  - a sustained power and clock run (`amd-smi`, 8 s per case);
  - an exact count of routing padding.
- Counter note: `SQ_INSTS_VALU` includes the MFMA instructions. So non-MFMA VALU per MFMA is
  `SQ_INSTS_VALU / SQ_INSTS_MFMA - 1`. This matches the static ISA ratios exactly (0.984 and 1.278).

## What is in the workspace now
Commit `f3f9493`, "round 2 winner: integrated". **Correctness PASS**: rel_l2 2.315e-3 and 2.307e-3.
Compared with R0:
- **Sort.** `moe_sort_fast.py` (2 HIP kernels, `k_count` + `k_place`) replaces the CK 4-kernel
  ClearWorkspace/P0/P1/P23 chain. Its output is bit-identical. `k_place` now owns the moe_buf zero-fill.
- **s2.** `flydsl_moe_fp8bs_s2.py` gains:
  - `s_setprio` around the K-loop;
  - **fpre**: an a2s×w2s scale table in LDS, which grows LDS from 28 to 32 KB;
  - **epi_mode 3**: weight folded into the f-table, `v_perm` packing, branchless atomics;
  - a sched_group_barrier with 1 VALU per MFMA slot.
- **gemm1 (`moe_fp8bs_gemm1_0`) is unchanged.**

### Per-case latency (COMMANDMENT FULL_BENCHMARK, 3 runs on GPU0, very stable)
| case | runs (ms) | median | vs R0 | speedup vs frozen baseline |
|---|---|---|---|---|
| tpr2048 | 1.0942 / 1.0944 / 1.0916 | **1.0942** | 1.1206 -> -2.4% | 1.5305x |
| tpr4096 | 1.9926 / 1.9910 / 1.9877 | **1.9910** | 2.0562 -> -3.2% | 1.6140x |

**Geomean = 1.5717x.** R0 was 1.528x and the warm-start bar was 1.5397x.

## Dispatch shape: 6 -> 4 kernels per forward, no launch gaps
In the steady-state trace (110 forwards per case, medians), the forward span equals the kernel sum, so
there are no host gaps. The forward is not overhead-bound.

| kernel | tpr2048 us | % fwd | tpr4096 us | % fwd | what changed |
|---|---|---|---|---|---|
| moe_fp8bs_gemm1_0 | **683.4** | **62.7%** | **1267.3** | **64.2%** | unchanged (within cross-session noise of +/-3%) |
| moe_s2_fp8bs_0 | 363.2 | 33.4% | 650.7 | 33.0% | 398.8 -> 363.2 / 713.5 -> 650.7 (**-8.9% / -8.8%**) |
| k_place (sort + zero-fill) | 35.4 | 3.2% | 47.7 | 2.4% | CK chain 67.7 / 110.8 us -> 42.3 / 54.5 us |
| k_count | 6.9 | 0.6% | 6.8 | 0.3% | |
| **span** | **1089.0** | | **1974.1** | | R1 span: 1151 / 2060 |

In the COMMANDMENT profile's device-time view, which includes input generation, gemm1 is **32.58%**
(R0: 31.60%) and s2 is **16.70%** (R0: 17.82%). The Abs, ReduceMax, Div, float8_copy and Mul kernels
(about 41% of device time) are input generation or reference work. They are not in the timed forward.

## PMC results (per kernel; median of 6 dispatches per case, merged over the passes)
| metric | gemm1 tpr2048 | gemm1 tpr4096 | s2 tpr2048 | s2 tpr4096 |
|---|---|---|---|---|
| sclk from GRBM_GUI_ACTIVE / duration | 1.05 GHz | 1.11 GHz | 1.03 GHz | 1.09 GHz |
| **MFMA busy** (SQ_VALU_MFMA_BUSY / cycles·SIMD) | **61.0%** | **65.7%** | **58.0%** | **65.4%** |
| FP8 achieved (MOPS×512 / time) | 794 TFLOPS | 908 TFLOPS | 743 TFLOPS | 888 TFLOPS |
| … % of the 2615 TFLOPS nameplate (2.1 GHz) | 30.4% | 34.7% | 28.4% | 34.0% |
| mean resident waves/SIMD (cap 2) | 1.79 | 1.88 | 1.76 | 1.85 |
| wave cycles: active / issue-wait / dep-wait | 23 / **56** / 20% | 24 / **57** / 19% | 26 / **53** / 21% | 28 / **54** / 18% |
| L2 hit | 77.0% | 81.1% | 71.2% | 76.6% |
| EA fetch (MALL+HBM) | 1290 MB, 1.75 TB/s | 2041 MB, 1.64 TB/s | 626 MB, 1.59 TB/s | 880 MB, 1.39 TB/s |
| EA write | 37 MB | 71 MB | 143 MB | 275 MB |
| non-MFMA VALU / MFMA | 0.98 | 0.98 | 1.28 | 1.28 |
| LDS / MFMA, LDS util, bank conflict | 0.20, 27%, 12.7% | 0.20, 29%, 12.7% | 0.24, 27%, 0% | 0.24, 31%, 0% |
| VMEM active | 2.5% | 2.7% | 2.4% | 2.7% |

**Power and clock (sustained 8 s per case on GPU0):** socket power has a median of **750 W** (max 751 W),
which is the cap. The gfx clock has a median of **1075 / 1065 MHz**, which is **51% of 2100 MHz**. The
GRBM-derived clocks above (1.03 to 1.11 GHz) agree.

**s2 before and after R2** (R2 engineer_2's PMC, same counters):
- Old canonical s2 (128+128 VGPR, 28 KB): about 53.5-56% MFMA busy at 0.97-1.10 GHz.
- New s2: 58-66%.
- The fpre and epi3 variant raised MFMA busy and clock together (53.5% at 0.97 GHz -> 61.8% at
  1.04 GHz). Removing non-MFMA work pays twice under a power cap.

## Cheap checks
- **Peaks:** no efficiency above 100%. The MOPS-derived FP8 rate matches MFMA busy at the measured
  clock exactly, so the calibration is consistent.
- **Fill:** gemm1 launches 69,632 CTAs, of which 4,352 / 8,400 do work. s2 launches 8,704 CTAs. There are
  608 resident CTA slots (2 per CU). That is 7.2 / 13.8 CTA waves, so the GPU is filled.
  - Tail at tpr2048: 0.16 of a wave, about 2%.
- **Occupancy:** both GEMMs are capped at 2 waves/SIMD by VGPR AND by LDS:
  - gemm1: 240 combined VGPR, 32 KB LDS;
  - s2: 232 combined VGPR, 32 KB LDS.
  - Achieved is 1.76-1.88. They sit at the ceiling.
- **Spill:** 0 scratch.
- **LDS bank conflicts:** 12.7% on gemm1, below the 20% threshold. 0% on s2.
- **MoE padding inflates work.** The counters confirm that gemm1 executes 17,408 / 33,600 MFMA rows for
  16,415 / 32,536 routed pairs. That is **5.2-5.7% / 2.9-3.2% padding** across both input sets.
  - Of the MFMA rows, **3.8-4.2% / 2.2-2.3% are whole all-padding 16-row MFMA subtiles.**
  - Both stages pay this.
- **EA traffic breakdown** (tpr2048, from bytes):
  - gemm1: about 512 MB of W13, about 570 MB of X replicated across 8 XCDs, and about 200 MB of refetch.
  - s2: about 256 MB of W2, about 285 MB of a2 replicated ×8, and about 89 MB of moe_buf RMW reads.
  - The replicated X and a2 reads are mostly MALL hits.

## Classification: COMPUTE (fp8 MFMA-bound, clock pinned by the 750 W power cap)
- **Not memory-bound.** EA bandwidth is 1.4-1.8 TB/s, about 30% of 5.3 TB/s. VMEM active is about 2.5%.
- **Not LDS-bound.** LDS util is 27-31% and conflicts are at or below 12.7%.
- **Not overhead-bound.** There are no launch gaps, and latency scales about 1.8x when tokens double.
- The matrix pipe is the busiest unit, at 58-66%, and the chip sits at its power limit. Throughput is
  therefore MFMA work × (1 / MFMA utilization) / clock, with the clock set by **energy per useful FLOP**.
- **The guide's wavefront tree reads issue-wait dominant (53-57% vs 18-21% dep-wait), which looks like
  "C2".** At 1.76-1.88 of the 2-wave cap with MFMA 58-66% busy, though, this issue-wait is mostly the two
  resident waves arbitrating for one MFMA pipe. It is not empty issue slots.
- The C2 remedy (more waves) has low value here, for three reasons:
  - it needs a tile redesign;
  - a power-capped chip cannot clock higher with more waves;
  - R1 measured the s2 3rd wave as unreachable without spills.
- The remaining true latency is the **C1-type dependency wait of about 20%**: rescale VALU waiting on
  MFMA results, plus LDS/VMEM `s_waitcnt`. gemm1 has 436 `s_waitcnt` in its fully unrolled 2048-MFMA body.

## BEFORE -> AFTER
```
BEFORE (R0): compute-bound - gemm1 31.60% + s2 17.82% of device time, 6 dispatches/forward (CK 4-kernel sort),
             2 waves/SIMD (gemm1 240 VGPR+32KB, s2 256 VGPR+28KB), no PMC; 1.1206/2.0562 ms = 1.528x
AFTER  (R2): compute-bound (MFMA, power-capped) - gemm1 62.7/64.2% of forward (unchanged kernel), s2 33.4/33.0%
             (-8.9%; MFMA busy ~54-56% -> 58-66%), sort 3.9/2.8% (2 kernels, -37%/-51%); 4 dispatches/forward;
             750 W / ~1.07 GHz sustained; 1.0942/1.9910 ms = 1.5717x
SHIFT:  The class did not move (compute -> compute), but the leverage did. R2 cut the two non-gemm1 parts:
        - sort: 4->2 kernels with the zero-fill fused;
        - s2: less VALU per MFMA (fpre/epi3), better MFMA issue (setprio/sched).
        gemm1 is now ~2/3 of every forward. PMC now shows the clock is pinned at the 750 W cap (~51% of max
        sclk), so non-MFMA energy (VALU, padding MFMAs, zero-fill/RMW bytes) costs clock as well as issue slots.
NEXT:   Target gemm1 (62.7-64.2%):
        - port the s2 R2 recipe;
        - skip the 3.8-4.2% / 2.2-2.3% of all-padding MFMA subtiles in both stages;
        - turn single-local-expert tokens (60%) into plain stores so the zero-fill and atomics shrink.
        Leave the 3rd wave and plain XCD remaps alone.
```

## Ranked opportunities for the TechLead
1. **gemm1 is the leader at 62.7% / 64.2% of the forward, and R2 did not touch it.** Port the s2 recipe
   that is already measured to work:
   - `FBS_SETPRIO` is **default-OFF** on gemm1. s2's setprio gave +0.63% geomean.
   - Retune gemm1's sched_group_barrier VALU-per-MFMA slot (`FBS_SCHED*`). s2 gained -1.3 to -1.6% at
     1 VALU per slot.
   - Pack the **542 remaining scalar `v_fma_f32`**. The body has 2048 MFMA, 753 `v_pk_fma_f32` and 542
     `v_fma_f32`, about 0.98 non-MFMA VALU per MFMA. `FBS_PK_SPLIT` already packs the main rescale, so
     these come from other paths.
   - Under the power cap, fewer VALU ops also lift the clock, as the s2 R2 evidence shows.
2. **Skip padded MFMA work in both GEMMs.** The counters confirm that tile_m=64 executes 5.2-5.7% /
   2.9-3.2% padding rows, and **3.8-4.2% / 2.2-2.3% of MFMA rows are whole all-padding 16-row subtiles.**
   - **Layout constraint.** gemm1 splits N across its 4 waves, and each wave runs m_repeat=4 16-row
     subtiles. MFMA ignores EXEC, so masking saves nothing.
   - **What it needs:** a uniform per-subtile skip, a tail-block code path, or a separate tail launch.
   - **Watch out for:**
     - I-cache size: the unrolled gemm1 body is about 45 KB of code;
     - prologue cost: R2 measured ballot/div at 2-3.5% of stage1. Have `moe_sort_fast` emit the
       per-block valid-row counts (or a tail list) for an `s_load` instead.
   - **Estimate:** about -3.5 to -4% at tpr2048 and about -2% at tpr4096, about +3% geomean. Under the
     power cap, the saved MFMA energy converts to time about 1:1.
3. **Single-local-expert tokens are 60%** of the received tokens (6505 / 10841 and 13037 / 21575). They
   have exactly one writer per output row, so s2 can plain-store them and `k_place` only needs to zero
   rows with 2 or more local experts. This removes about 60% of:
   - the 89 / 177 MB zero-fill. `k_place` is HBM-write-bound at 3.4-4.3 TB/s, taking 35 / 48 us.
   - the matching moe_buf RMW line reads in s2, about 89 MB at tpr2048.
   - part of the atomic cost, which is 3.6-4.2% of s2.
   The sort already knows each token's local count from `k_count`. Estimate: about 1-1.5% of the forward.
4. **s2 rescale VALU.** s2 runs 1.28 non-MFMA VALU per MFMA, 30% more than gemm1. Its body has
   **668 scalar `v_fma_f32` and only 178 `v_pk_fma_f32` per 1024 MFMA**.
   - Packing cuts about 330 VALU instructions, about 26% of s2's non-MFMA VALU.
   - Rescale is 8.4-9.6% of s2 time (R2's no-rescale control).
   - Estimate: -2 to -4% of s2, about 1% of the forward.
5. **Deprioritize these:**
   - **A 3rd wave.** It needs combined VGPR at or below 168 AND LDS at or below 21 KB together. gemm1's
     32 KB is 16 KB of X ping-pong plus 16 KB of full-K scale tables, so the LDS half is solvable with a
     K-ring, but the VGPR half is not proven. Occupancy is already 1.76-1.88 of 2, and the power cap
     limits the payoff.
   - **Plain XCD remaps.** R2 found them a dead end: L2 misses fell 1-25% but time was flat or worse. The
     X and a2 replication is mostly MALL traffic.
   - **HBM or LDS tuning.** HBM runs at about 30% of peak, and LDS conflicts are at or below 12.7%.
   - **Stage1 -> stage2 fusion.** The a2 round-trip is about 36 / 71 MB, under 3% of EA traffic.

## Artifacts
- COMMANDMENT profile: `profile_output_r2/profile_report.txt` and `profile_output_r2/rocprofv3/*/737844_kernel_{stats,trace}.csv`
- PMC passes: `profile_output_r2_pmc/{mfma,stall,l2,bytes,inst,lds}/`. The merged result is
  `profile_output_r2_pmc/pmc_summary.json`.
  - Tools: `profile_output_r2_pmc/tools/{run_pmc_passes.sh,parse_pmc.py,parse_trace.py}`
- Steady-state trace: `profile_output_r2_pmc/long_trace/`
- Power and clock: `profile_output_r2_pmc/sustained_power.log`
- Benchmarks: `profile_output_r2_pmc/bench.log`
- Routing and padding: `profile_output_r2_pmc/routing_stats.log`
- Static ISA used for the instruction mix: `round_2/engineer_2/isa/best_final/{moe_fp8bs_gemm1_0,moe_s2_fp8bs_0}/21_final_isa.s`.
  Its VGPR/LDS figures match the live kernels in the trace:
  - gemm1: 240 VGPR and 32 KB, matching exactly;
  - s2: 228 VGPR in the ISA, which shows as 100+132 = 232 in the trace after allocation granularity, and 32 KB.
  The static non-MFMA VALU/MFMA ratios (0.984 / 1.278) match the PMC-derived 0.98 / 1.28.
