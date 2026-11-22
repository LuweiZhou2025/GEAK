# r2_d0 (memory): XCD-aware CTA remap of the live gemm1 (stage 1). Result: dead end, no patch

## Verdict
No remap beat the same-session canonical in time. Nothing is shipped (there is no best_patch.diff).
COMMANDMENT harness, 3 interleaved rounds, medians in ms (tpr2048 / tpr4096):

| config | tpr2048 | tpr4096 | geomean vs frozen baseline | vs canonical |
|---|---|---|---|---|
| canonical (ws_base) | 1.1197 | 2.0412 | 1.5345x | 1.0000 |
| gn=4,nx=8,c=1,ord=0 (XCD tiling) | 1.1212 | 2.0435 | 1.5326x | 0.9988 (noise) |
| gn=16,nx=8,ord=3 (expert-run order) | 1.1316 | 2.0763 | 1.5135x | 0.9863 (loss) |

Both candidates pass COMMANDMENT correctness, with rel_l2 identical to canonical (outputs are bit-identical).

## What was built (kept in the workspace, default OFF; reference diff: xcd_remap_explore_NOT_A_CANDIDATE.diff)
- `flydsl_moe_fp8bs.py`: `compile_stage1` now wraps the lru-cached `_compile_stage1(..., xcd_gn, xcd_nx, xcd_c, xcd_ord, dbg)`.
  The knobs are `FBS_XCD_GN/NX/C/ORD`. `gn=0` is the legacy 2-D grid. The `s1(...)` signature is unchanged.
- With `gn>0` the kernel launches a 1-D grid of `nx*ceil(blocks/m_parts)*gn` CTAs (69,632 at blocks=4352) and maps each CTA as follows:
  - logical XCD = raw % nx, and j = raw / nx.
  - The XCD owns an N-tile group (xcd % (16/gn)) and one contiguous M partition. The partition is computed on device from `n_mblk = ceil(num_valid_ids[0]/64)`.
  - Work is chunk-interleaved: ord 0 walks N fastest, ord 1 walks M fastest.
  - ord 3 is the expert-run order: inside a 64-block window it finds the run of equal `sorted_expert_ids` with a vector load, a ballot and ctlz/cttz, then walks the run N-major.
  - Each CTA does one item. Out-of-range CTAs exit through the existing if-guard. There is no device loop.
- Bijection proof (host mirror `xcd_remap_ref`):
  - All n_mblk in [1, 4352] for 11 ord-0/1 configs (`scratch/bijection.py`).
  - Random expert layouts for ord 3 (`scratch/bijection3.py`).
  - On device, a2/a2s are bit-identical to the legacy grid for num_valid in {1, 63, 64, 65, 129, 1000, 4097, half, full-1, full} in both cases (`scratch/dev.py check`).
- `dbg` what-ifs, timing only: X always comes from M block 0 (L2-hot); W always reads K tile 0.

## gemm1 PMC (rocprofv3, median of 4 dispatches; misses in M lines, tpr2048 / tpr4096)
| variant gn,nx,c,ord | hit % | L2 misses | stage1 time vs legacy (in-process) |
|---|---|---|---|
| legacy 2-D | 77.0 / 81.1 | 10.38 / 16.50 | - |
| 2,8,1,0 (1-D control) | 76.8 / 81.0 | 10.44 / 16.59 | same as legacy |
| 4,8,1,0 | 77.1 / 82.0 | 10.30 / 15.74 | +-1% (noise) |
| 4,8,2,1 / 4,8,4,1 | ~77 / ~81 | 10.25-10.68 / 16.0-17.0 | +-1% |
| 8,8,{1,2,4},* | ~72 / ~76 | 12.4-13.0 / 20.8-21.8 | -0.5 .. -5.8% |
| 16,8,1,0 (pure M partition) | 59.2 / 62.0 | 18.36 / 33.19 | -4.9 / -10.9% |
| 16,8,8,1 / 16,8,16,1 | 77.4 / 80.1 ; 75.7 / 80.7 | 10.17 / 17.40 ; 10.96 / 16.87 | forward -2.4 / -1.3% |
| 2,8,{2..16},1 | 72.5-76.5 / 76.8-80.5 | 10.6-12.4 / 17.0-20.3 | worse |
| 4,8,1,3 expert-run | 80.0 / 81.4 | 9.03 / 16.29 (-13% / -1%) | -2.7 / -3.5% |
| 8,8,1,3 expert-run | 81.4 / 82.3 | 8.36 / 15.42 (-19.5% / -6.6%) | -1.9 / +1.3% |
| 16,8,1,3 expert-run | 82.8 / 82.9 | 7.74 / 14.95 (-25.5% / -9.5%) | -1.8 / -2.1% |
| what-if: X always L2-hot | 86.8 / 92.3 | 5.99 / 6.79 | +3.2 / +5.5% (ceiling) |
| what-if: W always L2-hot | 88.5 / 88.2 | 4.84 / 9.46 | +12.1 / +6.9% (~88 us in both cases) |

Diagnostic ord 4 keeps the ord-3 prologue chain but uses the ord-0 mapping, so its output is bit-identical to ord 0.
- Its stage-1 geomean is 0.966 vs 1.001 for 4,8,1,0. The dependent seid load, ballot and division on the CTA's critical path therefore costs about 2-3.5% of stage 1.
- That is enough to wipe out the gain from the expert-run order's fewer misses, which was worth about +1% when compared at equal prologue cost (4,8,1,3 vs 4,8,1,4).

## Why the lever is dead on this kernel
1. **The ceiling is small.** Even perfect X (activation) L2 reuse, including compulsory X, buys only 3.2% / 5.5% of stage 1.
   A remap can at best halve the X replication.
2. **Plain XCD tiling (gn>=4) only trades X misses for W misses.**
   - It spreads the CTAs that reuse an (expert, N tile) weight panel further apart in each XCD's dispatch order.
   - The measured W refetch probability grows linearly, at about 2.6% per position of reuse distance. W extra fetches per panel = 0.026*gn*(Mb-1), which fits gn=2/4/8/16 at both cases.
   - The extra W refetches cancel the X savings: misses net -1% / -4.6% at best, and the time is unchanged.
3. **Cutting misses does not cut time here.**
   - The expert-run order reaches the -25% miss target at tpr2048, but stage 1 gets slower.
   - Its prologue needs a vector load of the seid window before bx/by are known. That load queues behind the heavy W/X streams.
   - Large-gn partitioned orders (16,8,8,1) are slower even without the extra load.
4. **Per-miss cost is asymmetric.**
   - The W-hot what-if saves the same ~88 us in both cases, i.e. the same 512 MB of compulsory HBM reads of W13.
   - X-replication misses are mostly cheap MALL hits. Remaps only move MALL traffic.

## Not tried / left for later
A remap is only worth retrying if the per-CTA (m, n) assignment is free on the critical path. One option is a per-CTA table, or per-expert cumulative block counts, emitted by the sort (r2_d1's moe_sort_fast) and read via s_load. Combined with the expert-run order, that might recover about +1% of stage 1. It is unmeasured.
