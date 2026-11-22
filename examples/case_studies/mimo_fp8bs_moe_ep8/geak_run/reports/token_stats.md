# LLM token + time ledger

Every API call this run made, however small, with what it cost and how long it took.

- run: `/sgl-workspace/GEAK/exp/team_mimo_fp8bs_moe_ep8_20261119_100217_515084_14709/mimo_fp8bs_moe_ep8`
- observed window: 2026-11-19T10:01:25.400Z → 2026-11-19T20:42:54.736Z (10h41m) — from the first request (start inferred by stepping back that call's gap) to the last flush seen in the transcripts
- phase attribution: **timeline**
- transcripts read: 62
- **incomplete**: workflow invocation(s) wf_810900de-294 have no workflow record (still running, or ended before returning) and were tied to this eval-dir by their journal naming it as their eval_dir -- ownership INFERRED, not proven

## Run totals

| API calls | agents (>=) | wall | in (total) | out | in:out | billed | no-reuse | saved | re-sent cheaply |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 2,673 | 62 | 10h41m | 470,856,749 | 4,499,562 | 105:1 | $251.71 | $1977.11 | 87.3% | 97.1% |

- agents: 62 conversation(s) in the scoped transcripts, 60 dispatch attempt(s) recorded by the workflow (6 of them failed). The two are counted separately and cannot be joined — no shared identity is recorded — so how far they overlap is unknown and the figure above is a lower bound, not a count.

*billed* is what the reuse discount actually cost; *no-reuse* is the same traffic with every input token at full price. *re-sent cheaply* is the share of input that was already stored — high is good, but it is not the same as efficient: it can also mean cheaply re-sending something that should not be sent at all.

## Tokens by phase

| phase | calls | fresh in | re-sent | stored | out | $ | % of $ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| kernel-lane | 2,526 | 5,084 | 444,527,350 | 12,917,797 | 4,291,246 | $240.18 | 95.4% |
| kernel/Validate | 73 | 146 | 8,945,051 | 362,809 | 111,351 | $5.83 | 2.3% |
| kernel/Report | 21 | 42 | 2,029,460 | 159,230 | 46,201 | $2.13 | 0.8% |
| kernel/Profile | 14 | 28 | 556,234 | 82,924 | 16,137 | $1.20 | 0.5% |
| kernel/Benchmark | 17 | 34 | 496,765 | 45,867 | 11,850 | $0.83 | 0.3% |
| kernel/Analyze | 12 | 24 | 420,580 | 51,376 | 11,571 | $0.82 | 0.3% |
| kernel/Setup | 5 | 10 | 128,499 | 46,496 | 6,302 | $0.51 | 0.2% |
| Bakeoff | 5 | 10 | 61,974 | 18,959 | 4,904 | $0.21 | 0.1% |

## Time by phase

| phase | wall | % wall | in LLM | % of phase in LLM | median call | p95 call | slowest |
|:--|--:|--:|--:|--:|--:|--:|--:|
| kernel-lane | 10h29m | 98.1% | 8h23m | 80.1% | 4.1s | 54.7s | 6m31s |
| kernel/Validate | 0h19m | 3.1% | 0h13m | 67.7% | 3.7s | 53.1s | 2m06s |
| kernel/Report | 0h06m | 1.0% | 0h03m | 60.0% | 4.3s | 1m04s | 1m08s |
| kernel/Profile | 0h04m | 0.7% | 0h02m | 65.7% | 2.8s | 37.4s | 37.4s |
| kernel/Benchmark | 0h03m | 0.6% | 0h01m | 32.8% | 3.8s | 15.2s | 15.2s |
| kernel/Analyze | 0h02m | 0.4% | 0h01m | 61.6% | 3.3s | 30.6s | 30.6s |
| kernel/Setup | 0h01m | 0.2% | 0h00m | 45.0% | 6.9s | 12.9s | 12.9s |
| Bakeoff | 0h00m | 0.1% | 0h00m | 89.9% | 2.5s | 24.2s | 24.2s |

*wall* is elapsed time including the compiling and benchmarking between calls; *in LLM* is the part spent waiting on the model. A phase with a low *% of phase in LLM* is not token-bound, so token savings will barely move it.

## By role

| role | calls | in | out | $ | median call |
|:--|--:|--:|--:|--:|--:|
| engineer | 1,384 | 331,897,194 | 2,489,003 | $153.41 | 4.5s |
| tech_lead | 433 | 53,043,952 | 980,095 | $42.79 | 3.4s |
| verify_engineer | 341 | 23,941,754 | 371,077 | $18.01 | 3.7s |
| profile_engineer | 165 | 22,721,982 | 296,998 | $14.55 | 4.4s |
| integrator | 194 | 27,789,125 | 170,594 | $12.92 | 3.7s |
| update_experience | 41 | 6,777,689 | 75,741 | $4.05 | 4.4s |
| director | 37 | 2,705,322 | 41,912 | $2.29 | 3.6s |
| storage | 40 | 1,078,995 | 43,182 | $1.79 | 4.1s |
| benchmark_engineer | 17 | 542,666 | 11,850 | $0.83 | 3.8s |
| kb | 10 | 205,185 | 9,947 | $0.56 | 6.2s |
| warm_start | 6 | 71,942 | 4,259 | $0.30 | 2.7s |
| file_writer | 5 | 80,943 | 4,904 | $0.21 | 2.5s |

## Ten most expensive agents

| agent | phase | calls | in | out | $ |
|:--|:--|--:|--:|--:|--:|
| eng r4_d2:memory | kernel-lane | 157 | 48,927,804 | 237,722 | $19.27 |
| eng r1_d1:host_runtime | kernel-lane | 176 | 49,690,305 | 286,693 | $19.09 |
| eng r3_d0:compute | kernel-lane | 142 | 39,461,499 | 235,895 | $14.92 |
| eng r4_d0:compute | kernel-lane | 138 | 29,409,064 | 173,786 | $14.36 |
| eng r1_d2:compute | kernel-lane | 117 | 26,565,010 | 268,580 | $13.85 |
| eng r1_d0:memory | kernel-lane | 113 | 26,731,464 | 253,856 | $13.81 |
| eng r4_d1:algorithm | kernel-lane | 71 | 17,398,440 | 199,295 | $13.35 |
| eng r2_d2:compute | kernel-lane | 126 | 27,931,777 | 189,593 | $12.70 |
| eng r3_d1:algorithm | kernel-lane | 107 | 24,822,089 | 160,569 | $9.93 |
| eng r2_d1:host_runtime | kernel-lane | 87 | 15,086,871 | 163,592 | $7.75 |

---

Prices used, per million tokens: cache_read $0.50, cache_write_1h $10.00, cache_write_5m $6.25, input $5.00, output $25.00. Raw token counts are measured; the dollar columns are those counts times these rates.
