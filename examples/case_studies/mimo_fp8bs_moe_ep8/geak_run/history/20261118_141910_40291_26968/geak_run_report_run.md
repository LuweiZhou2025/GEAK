# GEAK run report — run

## Completeness

- **workflow telemetry**: incomplete — workflow invocation(s) wf_3b52f81a-25d have no workflow record (still running, or ended before returning) and were tied to this eval-dir by their journal naming it as their eval_dir -- ownership INFERRED, not proven
- **transcript scope**: run-scoped-inferred (INFERRED — an invocation with no workflow record was tied to this eval dir by its journal; ownership not proven) [anchor: live-journal]
- **capture**: 232/232 API responses have a terminal stop
- **cost coverage**: child-scope, estimated from a fixed rate card (not an SDK total, not an invoice); excludes parent driver/resume/monitor scope.

## Run totals

- **API calls**: 232
- **Observed span (first request → last observed flush)**: 53m59s (2026-11-18 14:18:30Z → 2026-11-18 15:12:30Z) — the left edge is inferred by stepping back the first call's gap, and the right edge is the last flush written to the transcripts, not a completed response
- **Billed span (Σ per-call, not wall-time)**: 34m11s
- **Cost**: $18.37
  - cache-write $6.73, cache-read $5.25, uncached-context $0.0023, router $0.0000, output $6.39
- **Tokens**: uncached-input 464 · cache-read 10,496,270 · cache-write 1,076,623 · output 255,705

### By model

| model | calls | cost |
|---|---|---|
| claude-opus-4-8 | 232 | $18.37 |

## Time, cost and tokens by phase

| phase | started | elapsed (inferred) | billed span | calls | agents | input tok | output tok | cost |
|---|---|---|---|---|---|---|---|---|
| kernel/Setup | 2026-11-18 14:18:30Z | 1m05s | 26.1s | 8 | 1 | 243,566 | 4,946 | $0.4662 |
| kernel/Analyze | 2026-11-18 14:19:35Z | 5m04s | 3m30s | 19 | 1 | 1,094,920 | 22,224 | $1.64 |
| kernel/Benchmark | 2026-11-18 14:24:40Z | 3m23s | 1m06s | 16 | 1 | 510,265 | 10,018 | $0.7531 |
| kernel/Profile | 2026-11-18 14:28:04Z | 3m50s | 2m11s | 14 | 1 | 617,126 | 16,497 | $1.18 |
| kernel-lane | 2026-11-18 14:31:54Z | 40m17s | 25m22s | 156 | 14 | 8,467,487 | 184,122 | $12.93 |
| kernel/Report | 2026-11-18 15:08:17Z | 1m58s | 37.5s | 7 | 1 | 265,459 | 9,371 | $0.6768 |
| kernel/Validate | 2026-11-18 15:10:15Z | 1m48s | 53.5s | 10 | 1 | 348,531 | 6,875 | $0.5937 |
| Bakeoff | 2026-11-18 15:12:12Z | 17.5s | 4.6s | 2 | 1 | 26,003 | 1,652 | $0.1336 |

*elapsed\** is an OBSERVED span: from a start inferred by stepping back the gap since the previous transcript record, to the last time the final response was seen. The transcript records no request time, so the left edge is inferred, and the span includes the compiling and benchmarking between calls. *billed span* is the sum of those same inferred per-call gaps, i.e. time plausibly spent waiting on the model. Phases overlap whenever work ran in parallel, so elapsed times do not add up to the run's.

## Workflow invocations counted

| invocation | started | elapsed (inferred) | billed span | calls | agents | input tok | output tok | cost |
|---|---|---|---|---|---|---|---|---|
| wf_3b52f81a-25d | 2026-11-18 14:18:30Z | 53m59s | 34m11s | 232 | 21 | 11,573,357 | 255,705 | $18.37 |

Every workflow invocation that worked in this run's eval dir is billed to it — a resumed or re-entered run is several invocations.

## Role view (organizational)

> Nesting is by role rank + execution order, not a literal spawn tree (transcripts carry no cross-agent spawn edge). Each node is one agent attempt; counts/time/cost are that agent's **own** API calls, exclusive of children. "Billed span" is Σ per-call observed durations, not true request wall-time.

- **director:setup** — 8 API calls · 26.1s billed · $0.4662 (own)
  - **tech_lead:analyze** — 19 API calls · 3m30s billed · $1.64 (own)
    - **benchmark_engineer:setup** `benchmark_engineer` — 16 API calls · 1m06s billed · $0.7531 (own)
    - **profile_engineer:baseline** — 14 API calls · 2m11s billed · $1.18 (own)
      - **warm_start:resolve** — 2 API calls · 2.9s billed · $0.1252 (own)
  - **tech_lead:plan_round** `tech_lead:plan r1` — 8 API calls · 1m59s billed · $1.20 (own)
    - **engineer:algorithm** `eng r1_d0:algorithm` — 26 API calls · 5m56s billed · $2.77 (own)
    - **engineer:compute** `eng r1_d1:compute` — 32 API calls · 6m03s billed · $2.90 (own)
      - **verify_engineer:verify** `verify r1_d0 (recovered)` — 5 API calls · 28.5s billed · $0.3011 (own)
      - **verify_engineer:verify** `verify r1_d1` — 12 API calls · 1m01s billed · $0.4941 (own)
  - **tech_lead:update_memory** `tech_lead:memory r1` — 6 API calls · 37.3s billed · $0.5418 (own)
    - **storage:reclaim** `storage:reclaim r1` — 5 API calls · 1m11s billed · $0.3782 (own)
  - **tech_lead:plan_round** `tech_lead:plan r2` — 6 API calls · 1m56s billed · $0.8341 (own)
    - **engineer:deep_explore** `deep r2_d0:deep_explore` — 26 API calls · 3m25s billed · $1.88 (own)
      - **verify_engineer:verify** `verify r2_d0` — 16 API calls · 1m10s billed · $0.6601 (own)
  - **tech_lead:update_memory** `tech_lead:memory r2` — 6 API calls · 57.3s billed · $0.5283 (own)
    - **storage:reclaim** `storage:reclaim r2` — 4 API calls · 29.3s billed · $0.2325 (own)
  - **tech_lead:report** — 7 API calls · 37.5s billed · $0.6768 (own)
- **director:validate** — 10 API calls · 53.5s billed · $0.5937 (own)
  - **kb:cite** — 2 API calls · 2.9s billed · $0.0922 (own)
  - **file_writer:persist_llm_stats** — 2 API calls · 4.6s billed · $0.1336 (own)

## Agent details

### director:setup
- label: `director:setup`
- models: claude-opus-4-8
- phase: kernel/Setup · workflow run: wf_3b52f81a-25d
- calls: 8 · billed: 26.1s · elapsed: 1m05s
- cost: $0.4662 (cache-write $0.2399, cache-read $0.1026, uncached-context $0.0001, router $0.0000, output $0.1237)
- tokens: uncached-input 16 · cache-read 205,160 · cache-write 38,390 · output 4,946

### tech_lead:analyze
- label: `tech_lead:analyze`
- models: claude-opus-4-8
- phase: kernel/Analyze · workflow run: wf_3b52f81a-25d
- calls: 19 · billed: 3m30s · elapsed: 5m04s
- cost: $1.64 (cache-write $0.5823, cache-read $0.5009, uncached-context $0.0002, router $0.0000, output $0.5556)
- tokens: uncached-input 38 · cache-read 1,001,713 · cache-write 93,169 · output 22,224

### benchmark_engineer:setup
- label: `benchmark_engineer`
- models: claude-opus-4-8
- phase: kernel/Benchmark · workflow run: wf_3b52f81a-25d
- calls: 16 · billed: 1m06s · elapsed: 3m23s
- cost: $0.7531 (cache-write $0.2689, cache-read $0.2336, uncached-context $0.0002, router $0.0000, output $0.2505)
- tokens: uncached-input 32 · cache-read 467,207 · cache-write 43,026 · output 10,018

### profile_engineer:baseline
- label: `profile_engineer:baseline`
- models: claude-opus-4-8
- phase: kernel/Profile · workflow run: wf_3b52f81a-25d
- calls: 14 · billed: 2m11s · elapsed: 3m50s
- cost: $1.18 (cache-write $0.5035, cache-read $0.2683, uncached-context $0.0001, router $0.0000, output $0.4124)
- tokens: uncached-input 28 · cache-read 536,539 · cache-write 80,559 · output 16,497

### warm_start:resolve
- label: `warm_start:resolve`
- models: claude-opus-4-8
- phase: kernel-lane · workflow run: wf_3b52f81a-25d
- calls: 2 · billed: 2.9s · elapsed: 15.5s
- cost: $0.1252 (cache-write $0.0850, cache-read $0.0061, uncached-context $0.0000, router $0.0000, output $0.0340)
- tokens: uncached-input 4 · cache-read 12,267 · cache-write 13,605 · output 1,360

### tech_lead:plan_round
- label: `tech_lead:plan r1`
- models: claude-opus-4-8
- phase: kernel-lane · workflow run: wf_3b52f81a-25d
- calls: 8 · billed: 1m59s · elapsed: 3m14s
- cost: $1.20 (cache-write $0.6087, cache-read $0.2098, uncached-context $0.0001, router $0.0000, output $0.3788)
- tokens: uncached-input 16 · cache-read 419,639 · cache-write 97,399 · output 15,153

### engineer:algorithm
- label: `eng r1_d0:algorithm`
- models: claude-opus-4-8
- phase: kernel-lane · workflow run: wf_3b52f81a-25d
- calls: 26 · billed: 5m56s · elapsed: 9m15s
- cost: $2.77 (cache-write $0.8180, cache-read $1.04, uncached-context $0.0003, router $0.0000, output $0.9067)
- tokens: uncached-input 52 · cache-read 2,082,948 · cache-write 130,881 · output 36,270

### engineer:compute
- label: `eng r1_d1:compute`
- models: claude-opus-4-8
- phase: kernel-lane · workflow run: wf_3b52f81a-25d
- calls: 32 · billed: 6m03s · elapsed: 12m24s
- cost: $2.90 (cache-write $0.6637, cache-read $1.22, uncached-context $0.0003, router $0.0000, output $1.02)
- tokens: uncached-input 64 · cache-read 2,431,034 · cache-write 106,197 · output 40,708

### verify_engineer:verify
- label: `verify r1_d0 (recovered)`
- models: claude-opus-4-8
- phase: kernel-lane · workflow run: wf_3b52f81a-25d
- calls: 5 · billed: 28.5s · elapsed: 57.3s
- cost: $0.3011 (cache-write $0.1464, cache-read $0.0354, uncached-context $0.0001, router $0.0000, output $0.1193)
- tokens: uncached-input 10 · cache-read 70,879 · cache-write 23,419 · output 4,771

### verify_engineer:verify
- label: `verify r1_d1`
- models: claude-opus-4-8
- phase: kernel-lane · workflow run: wf_3b52f81a-25d
- calls: 12 · billed: 1m01s · elapsed: 2m16s
- cost: $0.4941 (cache-write $0.1495, cache-read $0.1289, uncached-context $0.0001, router $0.0000, output $0.2156)
- tokens: uncached-input 24 · cache-read 257,736 · cache-write 23,922 · output 8,623

### tech_lead:update_memory
- label: `tech_lead:memory r1`
- models: claude-opus-4-8
- phase: kernel-lane · workflow run: wf_3b52f81a-25d
- calls: 6 · billed: 37.3s · elapsed: 1m42s
- cost: $0.5418 (cache-write $0.2691, cache-read $0.0745, uncached-context $0.0001, router $0.0000, output $0.1981)
- tokens: uncached-input 12 · cache-read 149,097 · cache-write 43,059 · output 7,922

### storage:reclaim
- label: `storage:reclaim r1`
- models: claude-opus-4-8
- phase: kernel-lane · workflow run: wf_3b52f81a-25d
- calls: 5 · billed: 1m11s · elapsed: 1m34s
- cost: $0.3782 (cache-write $0.1667, cache-read $0.0360, uncached-context $0.0001, router $0.0000, output $0.1755)
- tokens: uncached-input 10 · cache-read 71,971 · cache-write 26,667 · output 7,021

### tech_lead:plan_round
- label: `tech_lead:plan r2`
- models: claude-opus-4-8
- phase: kernel-lane · workflow run: wf_3b52f81a-25d
- calls: 6 · billed: 1m56s · elapsed: 3m12s
- cost: $0.8341 (cache-write $0.3738, cache-read $0.0924, uncached-context $0.0001, router $0.0000, output $0.3679)
- tokens: uncached-input 12 · cache-read 184,749 · cache-write 59,816 · output 14,714

### engineer:deep_explore
- label: `deep r2_d0:deep_explore`
- models: claude-opus-4-8
- phase: kernel-lane · workflow run: wf_3b52f81a-25d
- calls: 26 · billed: 3m25s · elapsed: 6m32s
- cost: $1.88 (cache-write $0.5021, cache-read $0.7421, uncached-context $0.0003, router $0.0000, output $0.6319)
- tokens: uncached-input 52 · cache-read 1,484,102 · cache-write 80,337 · output 25,277

### verify_engineer:verify
- label: `verify r2_d0`
- models: claude-opus-4-8
- phase: kernel-lane · workflow run: wf_3b52f81a-25d
- calls: 16 · billed: 1m10s · elapsed: 2m39s
- cost: $0.6601 (cache-write $0.2138, cache-read $0.1928, uncached-context $0.0002, router $0.0000, output $0.2533)
- tokens: uncached-input 32 · cache-read 385,548 · cache-write 34,213 · output 10,132

### tech_lead:update_memory
- label: `tech_lead:memory r2`
- models: claude-opus-4-8
- phase: kernel-lane · workflow run: wf_3b52f81a-25d
- calls: 6 · billed: 57.3s · elapsed: 1m39s
- cost: $0.5283 (cache-write $0.2521, cache-read $0.0768, uncached-context $0.0001, router $0.0000, output $0.1992)
- tokens: uncached-input 12 · cache-read 153,643 · cache-write 40,342 · output 7,970

### storage:reclaim
- label: `storage:reclaim r2`
- models: claude-opus-4-8
- phase: kernel-lane · workflow run: wf_3b52f81a-25d
- calls: 4 · billed: 29.3s · elapsed: 47.0s
- cost: $0.2325 (cache-write $0.1212, cache-read $0.0209, uncached-context $0.0000, router $0.0000, output $0.0903)
- tokens: uncached-input 8 · cache-read 41,896 · cache-write 19,390 · output 3,614

### tech_lead:report
- label: `tech_lead:report`
- models: claude-opus-4-8
- phase: kernel/Report · workflow run: wf_3b52f81a-25d
- calls: 7 · billed: 37.5s · elapsed: 1m58s
- cost: $0.6768 (cache-write $0.3367, cache-read $0.1058, uncached-context $0.0001, router $0.0000, output $0.2343)
- tokens: uncached-input 14 · cache-read 211,578 · cache-write 53,867 · output 9,371

### director:validate
- label: `director:validate`
- models: claude-opus-4-8
- phase: kernel/Validate · workflow run: wf_3b52f81a-25d
- calls: 10 · billed: 53.5s · elapsed: 1m48s
- cost: $0.5937 (cache-write $0.2690, cache-read $0.1527, uncached-context $0.0001, router $0.0000, output $0.1719)
- tokens: uncached-input 20 · cache-read 305,466 · cache-write 43,045 · output 6,875

### kb:cite
- label: `kb:cite`
- models: claude-opus-4-8
- phase: kernel-lane · workflow run: wf_3b52f81a-25d
- calls: 2 · billed: 2.9s · elapsed: 8.6s
- cost: $0.0922 (cache-write $0.0720, cache-read $0.0054, uncached-context $0.0000, router $0.0000, output $0.0147)
- tokens: uncached-input 4 · cache-read 10,893 · cache-write 11,526 · output 587

### file_writer:persist_llm_stats
- label: `file_writer:persist_llm_stats`
- models: claude-opus-4-8
- phase: Bakeoff · workflow run: wf_3b52f81a-25d
- calls: 2 · billed: 4.6s · elapsed: 17.5s
- cost: $0.1336 (cache-write $0.0862, cache-read $0.0061, uncached-context $0.0000, router $0.0000, output $0.0413)
- tokens: uncached-input 4 · cache-read 12,205 · cache-write 13,794 · output 1,652

