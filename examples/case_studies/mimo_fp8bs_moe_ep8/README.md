# MiMo-V2.6-Flash EP8 FP8 block-scale MoE on MI300X: GEAK run archive + pyhip comparison

Rank-local routed-expert MoE of MiMo-V2.6-Flash-MOPD under EP8 (SGLang MoriEPMoE + MORI FP8 dispatch): hidden 4096,
moe_intermediate 2048, 256 experts (32 local), top-8, SiLU, W8A8 FP8 128x128 block scale, BF16 out, gfx942.
Baseline = aiter `fused_moe` → 1-stage asm `fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256`.
GEAK result = FlyDSL 2-stage (gemm1 + s2) with a 2-kernel HIP sort, **1.62x geomean** (Director-verified).

## Layout

| path | content |
|---|---|
| `0_install_geak.sh` … `4_run_geak.sh` | setup and GEAK launch (`1_env.sh` prompts for the gateway key, nothing stored) |
| `mimo_fp8bs_moe_ep8/` | the task exactly as GEAK received it (baseline `moe_ep.py`, verbatim `aiter_fused_moe.py`, frozen `scripts/task_runner.py`) |
| `geak_run/` | GEAK output of the accepted run `team_mimo_fp8bs_moe_ep8_20261119_100217_515084_14709` |
| `geak_run/optimized/`, `final_patch.diff` | optimized code (`flydsl_moe_fp8bs_s1v2.py` does not parse; it is imported only under `MOE_S1_IMPL=v2`) |
| `geak_run/*.md`, `*.json`, `report/`, `reports/`, `rounds/` | TechLead report, Director validation (+ logs), insight log, per-round metrics / shift analyses, run report, token stats |
| `geak_run/learned_inbox/` | the experience cards these runs wrote to `kernel_workflow/knowledge/learned/_inbox` (copies, not drained) |
| `geak_run/history/` | reports of the earlier GEAK runs on this task |
| `pyhip_vs_geak/` | op-level A/B (`bench_moe_op.py`) and ATT phase analysis (`att_moe.py`) vs pyhip `luwei/942-fp8-blockwise` |
| `pyhip_vs_geak/results/` | raw bench JSONs, `bench/report.txt`, ATT tables |
| `slides/` | 2-page GEAK summary slides and their chart scripts |

## GEAK runs (Director-verified, harness ms per call)

| run | result | tpr2048 | tpr4096 | geomean |
|---|---|---|---|---|
| 20261118_141910 | accepted, empty patch | 1.6737 → 1.6742 | 3.2115 → 3.2200 | 0.999x |
| 20261119_033008 | accepted | 1.6807 → 1.1207 | 3.2239 → 2.0419 | 1.539x |
| 20261119_084500 | report only (same patch as 033008) | | | |
| **20261119_100217** | **accepted (final)** | **1.6747 → 1.0599** | **3.2135 → 1.9242** | **1.624x** |

## pyhip vs GEAK, op level (MI300X, idle GPU, 2026-11-20)

Same harness inputs, reference and timing loop; each side keeps its own M block (GEAK 64, pyhip 256). pyhip runs the
steps of its aiter hook (luocheng25/aiter `ac2b48fe`): a1_scale transpose, CK sort into an empty moe_buf,
route counts, zero rows, GateUp, Down. ms per call, median of the clean samples (idle GPU before the timed window,
round spread ≤ 3%); speedup = pyhip / GEAK.

| tokens/rank (rows / valid) | same CK sort | same GEAK fast sort | as shipped |
|---|---|---|---|
| 2048 (32768 / 10841) | 1.073 vs 1.146 → 1.07x | 1.054 vs 1.136 → 1.08x | 1.054 vs 1.146 → **1.09x** |
| 4096 (32768 / 21575) | 1.962 vs 1.959 → 1.00x | 1.930 vs 1.948 → 1.01x | 1.930 vs 1.959 → **1.02x** |
| 8192 (65536 / 43237) | 3.756 vs 3.645 → 0.97x | 3.716 vs 3.613 → 0.97x | 3.716 vs 3.645 → **0.98x** |

Sorted rows (padding over the local routes) explain most of the trend:

| tokens/rank | local routes | block 64 (GEAK) | block 256 (pyhip) |
|---|---|---|---|
| 2048 | 16415 | 17408 (+6.0%) | 20480 (+24.8%) |
| 4096 | 32536 | 33600 (+3.3%) | 36096 (+10.9%) |
| 8192 | 65284 | 66432 (+1.8%) | 69120 (+5.9%) |

Kernel breakdown (µs, as shipped, tpr4096): GEAK sort 55 + gemm1 1194 + s2 674 (4 launches); pyhip sort 77 + other 48
+ GateUp 1177 + Down 686 (10 launches). Full table: `pyhip_vs_geak/results/bench/report.txt`.

Correctness: every config passes the harness on both seeds and with NaN scales past `num_local_tokens`. GEAK and pyhip
outputs are 99.95–99.97% bit-identical BF16 on single-route tokens (rest 1 ULP); multi-route tokens differ only by
BF16 atomic order.

## ATT, tpr4096 (rocprofv3 1.1.0 + trace decoder 0.1.6, SIMD 0 of CU 1 on 16 SEs)

Mainloop = first MFMA to end of the last MFMA per wave; MFMA = SIMD MFMA busy within the span.

| kernel | prolog | mainloop | epilog | whole wave |
|---|---|---|---|---|
| pyhip GateUp | 4.6–5.0%, MFMA 3% | 87.8–88.2%, 140.6k cyc, **92.7–92.9%** | 7.2%, MFMA 2% | 81.8–82.3% |
| GEAK gemm1 | 7.0–7.4%, MFMA 47–48% | 85.9–86.2%, 67.2k cyc, **83.5–83.7%** | 6.8%, MFMA 55–56% | 79.0–79.3% |
| pyhip Down | 7.1–7.6%, MFMA 4% | 76.9–77.9%, 70.0k cyc, **92.9–93.0%** | 14.5–16.0%, MFMA 2% | 72.1–73.0% |
| GEAK s2 | 9.1–9.2%, MFMA 35–51% | 74.3–75.0%, 30.8k cyc, **77.6–78.1%** | 15.9–16.6%, MFMA 46–59% | 68.6–72.5% |

asm baseline (persistent, 1 wave per SIMD, 27–28 tiles of 32x256 per wave): GateUp K loop **64.7–68.9%** MFMA
(35–39% of the time), Down K loop **22.4–26.4%** (50–53%), outside the loops 0% (11%), whole wave 36–39%.
The decoder loses the PC at the asm's `s_setvskip` epilogue; `att_moe.py replay` rebuilds it from the code object
(0 mismatches on all 48 traced waves). 8192 tables: `pyhip_vs_geak/results/att/`. The ATT traces were taken while
another tenant was using the GPU; pyhip's numbers match its own idle-GPU ATT.

## Functional differences (interface, not math)

1. Sort block 64 (GEAK) vs 256 (pyhip, hard-coded): the sort kernel can be shared, its output cannot.
2. pyhip needs `a1_scale` as `[H/128, M]` (extra transpose); GEAK reads MORI's token-major scales.
3. GEAK zeroes rows `[0, num_local_tokens)` in the sort and adds every route atomically; pyhip zeroes only rows
   without exactly one local route and stores single-route rows directly.
4. GEAK returns its own `moe_buf` (rows past `num_local_tokens` uninitialized and may get `+0` dummy atomics);
   pyhip writes in place into a caller buffer and never touches those rows.
5. GEAK gemm1 checks only `token < M`, so padding rows are computed from a stale row and masked in stage 2;
   its padding-subtile skip needs the fast sort's per-block counts (off with CK sort).
6. GEAK's fast sort needs topk 8, ≤ 256 experts, rows % 256 == 0; pyhip checks hidden ≤ 4096 and a 2 GiB
   intermediate (about 130K rows); GEAK has no explicit guard.
7. GEAK builds with aiter's FlyDSL 0.1.5.dev504, pyhip needs FlyDSL ≥ 0.4.0: they cannot share one process.

## Reproduce

Stack: MI300X, ROCm 7.2 (torch 2.9.1+rocm7.2.0), aiter `a6bb49937`, GEAK `cec4603`, pyhip `e609ace`.

```bash
cd pyhip_vs_geak
pip install --no-deps --target .flydsl040 flydsl==0.4.0          # pyhip's FlyDSL, kept out of the system env
git clone -b luwei/942-fp8-blockwise https://github.com/tingqli/pyhip.git /sgl-workspace/pyhip
python3 bench_moe_op.py compare --gpu 0 --passes 2 --rounds 5 --profile-calls 30 --tpr 2048 4096 8192 --wait-idle 20
# ATT: decoder from https://github.com/ROCm/rocprof-trace-decoder/releases (0.1.6) unpacked under .att_decoder/
python3 att_moe.py att --gpu 0 --tpr 4096 --impls asm geak pyhip
python3 att_moe.py replay /tmp/pyhip_vs_geak_att/asm_ck_tpr4096/ui_output_agent_*
```
