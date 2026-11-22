# Round 3 Reprofile — mimo_fp8bs_moe_ep8 (PHASE=reprofile, round-3 integrated winner)

## Device
AMD Instinct **MI300X / gfx942 / CDNA3**, 304 CU, wave64, 64 KB LDS/CU, fp8 = FNUZ (rocminfo + rocm-smi).
Socket power cap **750 W**, sclk max 2100 MHz. All 8 GPUs idle at start.

## Which profiler ran, and the degrade
- **rocprof-compute failed** the same way as in every earlier round: its Python dependency check is
  missing 13 packages (plotext, textual, dash, ...). No `RPC_PROFILE_ARGS` override can fix a missing
  package, so I degraded on purpose to **rocprofv3**, as the COMMANDMENT PROFILE note expects.
- COMMANDMENT PROFILE (`--kernel-trace --stats`): `profile_output_r3/profile_report.txt`.
- Additional rocprofv3 work, all on the `task_runner.py profile` workload via `gpu_lock.sh 0,1,2,3`:
  - **8 `--pmc` passes** on GPU0: the R2 set (mfma, stall, l2, bytes, inst, lds) plus two new ones,
    **icache** (SQC_ICACHE_*) and **valu** (FMA_F32 / MFMA_F8 / TRANS / CVT / MUL / FLAT).
  - **An 11-arm `l2` attribution pass** on GPU0, comparing R2 and R3 code with R3's env knobs toggled.
    It uses env knobs only and edits no source.
- Other diagnostics:
  - a 110-forward steady-state kernel trace of `performance`;
  - the `amd-smi` sustained power/clock run;
  - a **same-session, same-GPU sustained power/clock A/B of R3 against R2**, using
    `round_3/integrate/canon`, whose md5 equals `f3f9493`;
  - an ISA dump of the live kernels (`FLYDSL_DUMP_IR`) plus `llvm-mc` code sizes;
  - exact routing and padding counts under the shipped 4/2/1 skip.

## What is in the workspace now
Commit `31c4cb1`, "round 3 winner: integrated". **Correctness PASS** (rel_l2 2.315e-3 and 2.307e-3, unchanged).
Compared with R2 (`f3f9493`):
- **gemm1** (`flydsl_moe_fp8bs.py`), with all knobs default-ON:
  - `FBS_RS=1`: scalar `v_fmac` rescale. The ISA now has 0 `v_pk_fma_f32` (R2 had 753) and 2048 scalar FMAs.
  - `FBS_WPE=2`.
  - `FBS_G1_PRIO=1`: `s_setprio 1` around the whole K loop.
  - `FBS_EPI_MAXNUM=1`.
  - **`FBS_PADSKIP=4,2,1`**: skips all-padding 16-row MFMA subtiles using the sort's `blk_cnt`.
- **s2** (`flydsl_moe_fp8bs_s2.py`): `S2_PADSKIP=4,2,1`. The full m_repeat=4 body is unchanged; its
  opcode sequence is 99.2% identical to R2's.
- **sort** (`moe_sort_fast.py`): `k_place` also emits `blk_cnt[m]`.
- There are still **4 dispatches per forward**. Span equals the kernel sum, so there are no host gaps.

### Per-case latency (COMMANDMENT FULL_BENCHMARK, 3 runs on GPU0)
| case | runs (ms) | median | vs R2 (1.0942 / 1.9910) | speedup vs frozen baseline |
|---|---|---|---|---|
| tpr2048 | 1.0603 / 1.0586 / 1.0610 | **1.0603** | -3.10% | 1.5795x |
| tpr4096 | 1.9202 / 1.9207 / 1.9263 | **1.9207** | -3.53% | 1.6731x |

**Geomean = 1.6256x.** R2 was 1.5717x, so this is +3.43%. The warm-start bar was 1.5397x.

## Dispatch shape and steady-state cost (110 forwards per case, medians)
| kernel | R2 2048 us | **R3 2048 us** | delta | R2 4096 us | **R3 4096 us** | delta | R3 % of fwd |
|---|---|---|---|---|---|---|---|
| moe_fp8bs_gemm1_0 | 683.4 | **645.5** | **-5.5%** | 1267.3 | **1199.9** | **-5.3%** | 60.9 / 62.1 |
| moe_s2_fp8bs_0 | 363.2 | **369.3** | **+1.7%** | 650.7 | **674.9** | **+3.7%** | 34.8 / 34.9 |
| k_place | 35.4 | 37.8 | +2.4 us | 47.7 | 48.8 | +1.1 us | 3.6 / 2.5 |
| k_count | 6.9 | 7.3 | +0.4 us | 6.8 | 7.3 | +0.5 us | 0.7 / 0.4 |
| **span** | 1089.0 | **1059.8** | -2.7% | 1974.1 | **1931.3** | -2.2% | |

- In the COMMANDMENT device-time view, gemm1 is **30.76%** and s2 is **16.98%**. The other 50.9% is input
  generation and reference work, which is outside the timed forward.
- **s2 got slower in the forward even though its MFMA work fell 3.5% / 1.9%.** The integrate round's
  same-session traces (220 dispatches each, GPU2) isolate why:
  - With s2 byte-identical, swapping in the faster gemm1 (arm `d0`) made **s2 +4.6% / +4.8% slower**:
    381 / 687 us against canon's 365 / 655 us.
  - Adding the s2 pad-skip then recovered -3.4% / -1.8% of s2. That matches its MFMA reduction 1:1.
  - The denser gemm1 pulls the power-capped clock down, and s2, which runs next, inherits the lower clock.

## Power and clock: the decisive measurement (same session, same GPU1, 2 x 6 s per case per arm)
| | R2 (`f3f9493`) | R3 (`31c4cb1`) | change |
|---|---|---|---|
| tpr2048 ms/fwd | 1.0972 / 1.0987 | 1.0541 / 1.0568 | **-3.9%** |
| tpr2048 gfx clock | 1077 / 1078 MHz | 1011 / 1008 MHz | **-6.3%** |
| tpr4096 ms/fwd | 2.0070 / 2.0088 | 1.9388 / 1.9419 | **-3.4%** |
| tpr4096 gfx clock | 1069 / 1069 MHz | 1002 / 1000 MHz | **-6.4%** |
| socket power | 750 W (cap) | 750 W (cap) | 0 |

- R3 removed **9.9% / 9.5% of the clock cycles per forward**, but the clock fell 6.3%, so time fell only
  3.9% / 3.4%. **About 60-65% of the cycle savings went into the lower clock.**
- At a pinned 750 W, energy per forward equals 750 W × time, so it also fell only 3.4-3.9%.
- How each kind of R3 change converted to time:

  | change | what it removed | conversion to time |
  |---|---|---|
  | s2 pad-skip | MFMA work (and so energy) | **1:1** in s2 time |
  | scalar-FMA swap + setprio | stall cycles (raised power density) | **~45%** of the cycle cut reached forward time; the next kernel inherits the clock loss |
  | gemm1 pad-skip | MFMA work, but it added bytes (below) | only **~20-30%** of its expected gain |

## PMC (rocprofv3 `--pmc`, serialized dispatches, GPU0; medians; R2 values from the R2 reprofile on GPU0)
| metric | gemm1 2048 | gemm1 4096 | s2 2048 | s2 4096 |
|---|---|---|---|---|
| MFMA busy (R2 -> R3) | 61.0 -> **65.1%** | 65.7 -> **74.2%** | 58.0 -> 55.1% | 65.4 -> 65.7% |
| SQ_INSTS_MFMA change (pad-skip) | **-3.49%** | **-1.90%** | **-3.49%** | **-1.90%** |
| wave cycles: active / issue-wait / dep-wait | 28 / **49** / 22% (R2 23/56/20) | 30 / **49** / 21% (R2 24/57/19) | 25 / 50 / 21% | 29 / 53 / 18% |
| mean resident waves/SIMD (cap 2) | 1.75 | 1.86 | 1.76 | 1.83 |
| non-MFMA VALU per MFMA | **1.27** (R2 0.98) | 1.27 | 1.29 | 1.29 |
| FMA_F32 per MFMA | **1.00** (all scalar) | 1.00 | 0.82 (668 scalar + 178 pk) | 0.82 |
| L2 hit (R2 -> R3) | 77.0 -> **73.4%** | 81.1 -> **79.6%** | 71.2 -> 72.4% | 76.6 -> 77.2% |
| EA fetch (MALL+HBM) (R2 -> R3) | 1290 -> **1502 MB** | 2041 -> **2214 MB** | 626 -> 598 MB | 880 -> 850 MB |
| EA write (R2 -> R3) | 37 -> **42 MB** | 71 -> **77 MB** | 143 -> 139 MB | 275 -> 271 MB |
| I-cache hit | 99.93% | 99.96% | 99.91% | 99.95% |
| LDS util / bank conflict | 30% / 12.9% | 33% / 12.8% | 26% / 0% | 31% / 0% |
| VMEM active | 2.7% | 3.1% | 2.4% | 2.8% |

**Notes on these numbers**
- **MFMA counts.** The SQ_INSTS_MFMA reductions match the exact routing prediction (17408 -> 16800 and
  33600 -> 32960 MFMA rows). The pad-skip works as designed in both kernels.
- **Wave-cycle split.** The 49% issue-wait in gemm1 is the signature of two waves sharing one MFMA pipe.
  It fell from 56-57% because the `v_pk_fma_f32` stalls that blocked the pipe are gone.
- **PMC-mode clock noise.** Per-dispatch clocks swing 0.79-1.35 GHz, so per-kernel cycle deltas below about
  5% (for example s2 2048) cannot be resolved from PMC. Instruction, MFMA-busy and L2 counts are stable.
- **The FMA counter.** `SQ_INSTS_VALU_FMA_F32` counts packed and scalar FMAs alike. TRANS is 0.033,
  MUL 0.069 and CVT 0.008 per MFMA (gemm1 epilogue).

### Static ISA (live kernels, `profile_output_r3_pmc/isa/`)
- **gemm1 is now at 256 VGPR with 35 VGPR spills and 120 B/lane scratch.** R2 had 240 VGPR and no spills.
  - All spill code sits in the 1- and 2-subtile pad-skip paths. The full m_repeat=4 body (2048 MFMA,
    2048 scalar FMA, 0 pk, 375 `s_nop`, 302 VMEM loads) is spill-free.
  - Its VMEM load positions are identical to the no-skip build.
  - **Code size went from 40.1 KB to 71.1 KB**: the full path is about 37 KB, the 2-subtile body
    20.4 KB and the 1-subtile body 12.9 KB.
- **s2 is at 246 VGPR with no spills, and code went from 24.4 KB to 45.3 KB.**
  - The full body is still **1024 MFMA / 668 `v_fma` / 178 `v_pk_fma_f32`**.
  - The reduced bodies are pk-heavier: 123 pk per 256 MFMA, and 156 per 512.
- I-cache hit is 99.9% or better, so the code growth costs nothing measurable.

## NEW: gemm1's pad-skip under-delivers, and the counters say why
The gemm1 skip should behave like s2's skip. MFMA falls -3.49% / -1.90%, which at the 1:1 conversion s2
showed would be **about -2.1% / -1.2% of the forward**. Measured, it delivered only **-0.23% to -0.42% /
-0.33% to -0.39%** (integrate paired A/B, g1skip vs combo, n=16 and n=12). The same-session `l2` pass
(GPU0, medians of 6 dispatches) shows where it went:

| gemm1 build | VGPR (alloc) | scratch B/lane | L2 misses (M) 2048 / 4096 | EA write req (M) |
|---|---|---|---|---|
| R2 canon (x2) | 240 | 0 | 10.37 / 16.49 | 0.575 / 1.109 |
| R3, `FBS_PADSKIP=0` (x2) | 232 | 0 | **10.43 / 16.62** | 0.575 / 1.109 |
| **R3 default, 4,2,1** (x2) | 256 | 120 | **12.11 / 17.83 (+16% / +8%)** | 0.664 / 1.203 |
| R3, `FBS_PADSKIP=4,1` | 256 | 100 | 11.60 / 17.51 | 0.624 / 1.159 |
| R3, `FBS_PADSKIP=4,2` | 256 | 260 | 12.72 / 19.50 | **2.137 / 3.979** |
| R3, `FBS_G1_PRIO=0` | 256 | 120 | 12.45 / 18.00 | 0.664 / 1.203 |
| R3, `FBS_RS=0` | 256 | 116 | 11.84 / 17.11 | 0.633 / 1.168 |
| R3, `FBS_WPE=0` (280 VGPR, 1 wave/SIMD) | 280 | 0 | 10.78 / 16.87 | 0.575 / 1.110 |

**Findings**
- **The pad-skip alone causes the regression.** With it, gemm1 makes **+1.7M / +1.2M more L2 misses per
  call, about +215 / +155 MB of EA reads**, with the same 45.1M L2 requests. It also adds +15% EA writes,
  which is spill write-back. Setprio, RS and WPE are not the cause.
- **Spill traffic is too small to explain the misses.** In `4,1` the extra EA writes are only about 3 MB,
  yet misses rise 1.16M.
- **Size.** The +1.16M misses in `4,1` come from 8 one-subtile tail blocks × 16 CTAs. That is about 1.2 MB
  per fast tail CTA, which is all of its W-panel + X reads (1.25 MB).
- **Likely mechanism (inferred, not measured directly).** A tail CTA that runs 2-4x shorter falls out of step
  with the CTAs of the same expert that share its W panel. In R2 it trailed them in lockstep and hit in L2.
- **Why this costs time.** Under the cap, each extra EA byte costs clock: R3's `DBG_BK0` L2-hot-W what-if
  was worth +10-15% of gemm1. That is why the skip's MFMA-energy saving largely disappears.

## Cheap checks
- **Peaks.** No efficiency exceeds 100%. The MOPS-derived FP8 rate equals MFMA-busy at the measured
  clock exactly.
- **Fill.** gemm1 launches 69,632 CTAs (4,352 / 8,400 do work) and s2 launches 8,704 CTAs, against 608
  resident slots. That is 7.2 / 13.8 CTA waves, so the GPU is filled.
- **Occupancy.** Both GEMMs sit at the 2-wave ceiling, set by VGPR (256 / 248) and by LDS (32 KB).
  Achieved occupancy is 1.75-1.86.
- **Spill.** gemm1 has 120 B/lane scratch, new in R3 and confined to the tail paths (see above).
  s2 has 0.
- **LDS bank conflicts.** gemm1 is at 12.9%, under the 20% threshold. s2 is at 0%.
- **MoE padding.**
  - Under the 4/2/1 skip, executed MFMA rows still carry **2.29% / 1.29% padding**, down from 5.70% / 3.17%.
  - A 3-subtile variant would remove only **0.76% / 0.39%** more MFMA rows. The rest are partial subtiles.
  - The data repeats across both seeds per case.
- **Steady FP8 rate (useful FLOPs).**
  - gemm1: 806 -> **853** TFLOPS at 2048 and 862 -> **910** at 4096 (+5.8% / +5.6%).
  - s2: 758 -> 746 and 839 -> 809, because of the clock carry-over.

## Classification: COMPUTE (fp8 MFMA-bound at a 750 W power-capped clock): unchanged class, sharper mechanism
- **Not memory-bound.** EA traffic runs at 1.8-2.1 TB/s, about 35-40% of 5.3 TB/s, and VMEM active is about 3%.
- **Not LDS-bound.** LDS util is 26-33%.
- **Not overhead-bound.** There are 4 dispatches and no gaps, and latency scales about 1.8x when tokens double.
- **What changed in gemm1.**
  - The MFMA pipe is busier: 65-74%, up from 61-66%.
  - Issue-wait is 49%, which is MFMA-pipe sharing between the 2 resident waves.
  - The scalar-FMA swap removed the `v_pk_fma_f32` pipe blocks.
- **The chip is now pinned harder against the cap.** The sustained clock fell from about 1.07 to
  about 1.00 GHz.
- **The guide's wavefront tree.** It would call 49-53% issue-wait "C2", but at 1.75-1.86 of a 2-wave cap,
  with the MFMA pipe saturated and the power pinned, more waves cannot raise throughput.
- **Conclusion: time per forward now tracks the energy per forward almost directly.**

## BEFORE -> AFTER
```
BEFORE (R2): compute-bound (MFMA, power-capped) - gemm1 683.4/1267.3 us (62.7/64.2% of fwd, MFMA busy 61/66%,
             0.98 non-MFMA VALU/MFMA incl. 753 v_pk_fma_f32), s2 363.2/650.7 us; 4 dispatches; 750 W @ ~1.07 GHz;
             1.0942/1.9910 ms = 1.5717x
AFTER  (R3): compute-bound (MFMA, power-capped, clock lower) - gemm1 645.5/1199.9 us (60.9/62.1%, MFMA busy 65/74%,
             0 pk_fma, 256 VGPR + 35 spills, pad-skip -3.5/-1.9% MFMA), s2 369.3/674.9 us (34.8/34.9%, slower:
             clock carry-over +4.6/4.8% minus pad-skip -3.4/-1.8%); 4 dispatches; 750 W @ ~1.00 GHz;
             1.0603/1.9207 ms = 1.6256x (+3.4%)
SHIFT:  The class did not change (compute -> compute), but the binding constraint became explicit.
        - R3 cut 9.5-9.9% of cycles per forward (same-session A/B), and the 750 W cap took back 6.3% of clock.
        - Removing stall cycles (pk->scalar FMA, setprio) raises power density, so only about 45% of it
          reaches time, and the next kernel (s2) inherits the lower clock.
        - Removing work (s2 pad-skip) converts 1:1.
        - Adding bytes (gemm1 pad-skip: +16%/+8% L2 misses, +215/155 MB EA reads, spills) cancels about 80%
          of a work cut.
NEXT:   Rank levers by energy removed (MFMA ops, EA/HBM bytes, VALU ops), not by cycles or latency hidden:
        1. repair the gemm1 pad-skip's L2/spill side effects (up to ~1.9%/0.9% of the forward);
        2. scalar-FMA swap in s2 (178 pk per 1024 MFMA);
        3. single-local-expert plain stores to shrink the zero-fill and the atomic RMW bytes;
        4. trim gemm1 epilogue VALU.
```

## Ranked opportunities for the TechLead
1. **Repair the gemm1 pad-skip's side effects. Estimated value: up to about -1.9% / -0.9% of the forward.**
   - **Evidence:** the table above. The skip removes 3.49% / 1.90% of gemm1 MFMAs but delivers only
     0.2-0.4% of the forward, while L2 misses rise +16% / +8% and VGPR goes from 230 to 256 with 35 spills.
     With `FBS_PADSKIP=0`, misses return exactly to the R2 level.
   - **Step 1:** make the reduced paths spill-free. One option is to narrow the common-prologue live-outs
     (recompute addresses inside each variant), or to branch before the prologue. Note that gemm1 has no
     VGPR headroom left, so any future addition spills.
   - **Step 2:** restore L2 sharing for tail CTAs. The fast 1-subtile tails lose about 100% of their L2 hits
     (about 1.2 MB each). Cheap experiments, each judged by the same `l2` pass
     (`profile_output_r3_pmc/tools/l2_attrib.sh`) and a paired A/B:
     - **Pace the tail CTA so it stays in lockstep with its siblings:** put `s_sleep` in place of the
       skipped MFMAs. The point is to save MFMA energy, not tail time.
     - **Change tail-block placement in `k_place`:** for example, emit each expert's partial block first.
     - **Keep only `4,1`:** it has the smallest miss penalty, +12% / +6%.
2. **Port the scalar-FMA swap (`_fma_sc`, `FBS_RS=1`) to s2. Estimated value: about -0.7 to -0.9% of the forward.**
   - **Evidence:** the s2 full body still has **178 `v_pk_fma_f32` per 1024 MFMA**. The tail bodies have
     123 per 256 and 156 per 512.
   - On gfx942 a packed-FP32 op blocks the SIMD's MFMA pipe. r3_d0's micro-bench: MFMA+pk takes 30.1 cycles
     against 17.1 for MFMA plus 2 scalar FMAs.
   - In gemm1 the same swap cut gemm1 cycles -10.7% / -11.8% at 0.37 pk per MFMA. s2 has 0.17 pk per MFMA,
     so expect about -5% of s2 cycles. That converts to time at about 45%.
   - Results should stay bit-identical (the same per-lane FMA).
3. **Plain stores for single-local-expert tokens. Estimated value: about -1 to -1.5% of the forward; it removes bytes, so it should convert about 1:1.**
   - **60.0-60.6%** of received tokens have exactly 1 local expert: 6505 / 10841 and 13037 / 21575.
   - `k_place` should zero only the rows with 2 or more local experts:
     - it writes 89 / 177 MB at 3.2-4.2 TB/s (37.8 / 48.8 us, HBM-write-bound);
     - this cuts about 53 / 106 MB of writes.
   - s2 should plain-store those rows instead of using bf16 atomics:
     - it removes about 53 / 106 MB of RMW line fetches from s2's 598 / 850 MB of EA fetch;
     - it also removes part of the atomic cost, measured at 3.6-4.2% of s2 in R2.
   - `k_count` already knows each token's local count.
4. **Trim gemm1 epilogue and other non-rescale VALU. Estimated value: at most about -1% of the forward.**
   - The rescale is irreducible on gfx942: exactly 1 scalar FMA per MFMA, and `DBG_NOSCALE` is worth only
     +6%.
   - The epilogue is worth +5.3% / +3.8% of gemm1 (`DBG_NOEPI`). It runs 0.033 TRANS, 0.069 MUL and
     0.008 CVT per MFMA, plus row-max and address VALU, about 0.27 non-FMA VALU per MFMA in total.
   - Under the cap, these cost energy even when they co-issue.
5. **Deprioritize these:**
   - **More pad-skip variants.** A 3-subtile variant would remove only 0.76% / 0.39% more MFMA rows (residual
     padding is 2.29% / 1.29%), and gemm1 has no VGPR or L2 margin until item 1 is fixed.
   - **A 3rd wave per SIMD.** gemm1 is at 256 VGPR with spills and s2 at 246. Issue-wait is MFMA-pipe sharing,
     and the clock is power-pinned.
   - **VALU-slotted scheduling or nop removal.** Stall cycles do not cost energy; r3_d0 measured `FBS_SCHED=5`
     as noise.
   - **Plain XCD remaps** (R2 dead end), **HBM or LDS tuning** (EA at 35-40% of peak, LDS conflicts at or
     below 12.9%), and **I-cache work** (99.9% hit).
   - **Stage1 -> stage2 fusion.** The a2 round-trip is under 3% of EA bytes.
   - **The 128-row stage1.** It is still closed (R1). The energy angle (half the W L2->CU bytes per FLOP) did
     not show up as speed in R1's s1t.

## Artifacts
- COMMANDMENT profile: `profile_output_r3/profile_report.txt` and
  `profile_output_r3/rocprofv3/*/819279_kernel_{stats,trace}.csv`
- PMC, all under `profile_output_r3_pmc/`:
  - passes: `{mfma,stall,l2,bytes,inst,lds,icache,valu}/`
  - merged summary: `pmc_summary.json`
  - parse log: `pmc_parse.log`
- L2 attribution: `profile_output_r3_pmc/l2_attrib/summary.txt` (arms log: `arms.log`)
- Steady-state trace: `profile_output_r3_pmc/long_trace/`
- Power and clock:
  - sustained, GPU1: `profile_output_r3_pmc/sustained_power.log`
  - same-session R2 vs R3 A/B: `profile_output_r3_pmc/power_ab.log`
- Benchmarks: `profile_output_r3_pmc/bench.log`
- Routing and padding under 4/2/1: `profile_output_r3_pmc/routing_stats.log`
- Live ISA: `profile_output_r3_pmc/isa/{moe_fp8bs_gemm1_0,moe_s2_fp8bs_0}/21_final_isa.s`. Assembled
  sizes are in `isa/obj/`.
- Tools, all in `profile_output_r3_pmc/tools/`:
  - `run_pmc_passes.sh`, `parse_pmc.py`, `parse_trace.py`
  - `l2_attrib.sh`, `l2_attrib2.sh`, `parse_l2_attrib.py`
  - `power_ab.sh`, `sustained_power.py`, `routing_stats.py`, `region_census.py`
- Integrate-round evidence used above: `round_3/integrate/logs/{ab,ab2,ab3}_g*.log` and
  `round_3/integrate/logs/trace*.out`.
