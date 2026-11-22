#!/usr/bin/env python3
"""ATT (rocprofv3 thread trace) of the GEAK and pyhip GateUp / Down kernels and the aiter 1-stage asm baseline on the
bench_moe_op.py workload: prolog / mainloop / epilog share and MFMA efficiency per kernel dispatch.

  python att_moe.py att [--gpu 0] [--tpr 4096] [--impls asm geak pyhip] [--wait-idle 30]   # trace, then report
  python att_moe.py phases DIR...                                 # report ui_output_agent_*_dispatch_* folders
  python att_moe.py trace --impl geak --sort fast --tpr 4096      # the traced workload

Phases are per wave and split at its MFMAs, so no runtime K loop is needed (GEAK's gemm1 K loop is unrolled):
prolog = wave start .. first MFMA issue, mainloop = first MFMA .. end of the last MFMA, epilog = the rest.
MFMA efficiency of a span = union of the MFMA windows of every traced wave on that SIMD (co-resident waves share
the MFMA pipe) inside the span / span cycles, summed over the working waves. Waves without an MFMA are the
workgroups past num_valid_ids that exit at once; they are counted separately.
"""

import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path

import bench_moe_op as bench

DECODER = os.environ.get("ATT_DECODER_DIR", os.path.join(bench.HERE, ".att_decoder", "opt", "rocm", "lib"))
KERNELS = {"geak": "moe_fp8bs_gemm1|moe_s2_fp8bs", "pyhip": "gateup_kernel|down_kernel", "asm": "fmoe_bf16_blockscaleFp8"}
# Sort each implementation ships with; asm (the GEAK task baseline: aiter fused_moe 1-stage) ignores it.
SHIPPED_SORT = {"geak": "fast", "pyhip": "ck", "asm": "ck"}
ASM_CO = "/sgl-workspace/aiter/hsa/gfx942/fmoe/silu/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co"
OBJDUMP = "/opt/rocm/llvm/bin/llvm-objdump"
INST_RE = re.compile(r"^\s+(\S+)(.*?)//\s*([0-9A-Fa-f]+):((?:\s+[0-9A-Fa-f]{8})+)\s*$")
LABEL_RE = re.compile(r"^([0-9a-f]+) <([^>]+)>:")
# rocprof-trace-decoder instruction categories
SMEM, SALU, VMEM, FLAT, LDS, VALU, JUMP, NEXT, IMMED, MESSAGE = 1, 2, 3, 4, 5, 6, 7, 8, 9, 11
WAVE_RE = re.compile(r"se(\d+)_sm(\d+)_sl(\d+)_wv(\d+)\.json")
PHASES = ("prolog", "mainloop", "epilog")
REALTIME_HZ = 100e6


def trace(args):
    import torch

    harness = bench.load_harness()
    if args.impl == "asm":
        sys.path.insert(0, bench.GEAK_TASK)
        import moe_ep

        prepare, forward = moe_ep.prepare_weights, moe_ep.forward
    else:
        prepare, forward = (bench.geak_impl if args.impl == "geak" else bench.pyhip_impl)(args.sort)
    weights = prepare(*harness.make_weights())
    sets = [harness.make_inputs(args.tpr, max(args.tpr, 4096), seed=s)[0] for s in (args.seed, args.seed + 100)]
    for k in range(args.calls):
        forward(weights, **sets[k % 2])
    torch.cuda.synchronize()


def att(args):
    os.makedirs(args.out_dir, exist_ok=True)
    pkg = os.path.join(bench.HERE, ".pyhip_pkg")
    for impl in args.impls:
        sort = SHIPPED_SORT[impl]
        env = dict(os.environ, HIP_VISIBLE_DEVICES=args.gpu, PYTHONDONTWRITEBYTECODE="1")
        if impl == "pyhip":
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [bench.PYHIP_FLYDSL, pkg, env.get("PYTHONPATH")]))
        busy = bench.wait_idle(args.gpu, args.wait_idle)
        cmd = ["rocprofv3", "--att", "--att-library-path", DECODER, "--att-target-cu", "1",
               "--att-shader-engine-mask", "0xffff", "--att-simd-select", "0x1", "--att-buffer-size", "0x40000000",
               "--kernel-include-regex", KERNELS[impl], "--kernel-iteration-range", args.iterations,
               "-d", os.path.join(args.out_dir, f"{impl}_{sort}_tpr{args.tpr}"), "-o", "att", "--",
               sys.executable, os.path.abspath(__file__), "trace", "--impl", impl, "--sort", sort,
               "--tpr", str(args.tpr), "--seed", str(args.seed), "--calls", str(args.calls)]
        subprocess.run(cmd, env=env, check=True)
        print(f"{impl}/{sort}: GPU {args.gpu} activity before/after {busy}/{bench.gpu_activity(args.gpu)} %", flush=True)
    report(sorted(glob.glob(os.path.join(args.out_dir, "**", "ui_output_agent_*_dispatch_*"), recursive=True)))


def load_dispatch(path):
    rows = json.loads((path / "code.json").read_text())["code"]
    asm = {int(r[2]): str(r[0]).strip() for r in rows}
    waves, partial = [], 0
    for wave_file in sorted(path.glob("se*_sm*_sl*_wv*.json")):
        payload = json.loads(wave_file.read_text())
        if payload["num_insts"] != payload["num_stitched"]:
            partial += 1
            continue
        wave = payload["wave"]
        # An instruction issues at first attempt + stall; 16x16x32 FP8 MFMA holds the SIMD 16 cycles, 32x32 32.
        mfma = [(int(r[0]) + int(r[2]), 32 if "32x32" in op else 16)
                for r in wave["instructions"] if (op := asm.get(int(r[4]), "")).startswith("v_mfma")]
        waves.append(dict(simd=(int(WAVE_RE.fullmatch(wave_file.name).group(1)), int(wave["cu"]), int(wave["simd"])),
                          begin=int(wave["begin"]), end=int(wave["end"]), mfma=mfma))
    ghz = []
    realtime = path / "realtime.json"
    if realtime.exists():
        for key, samples in json.loads(realtime.read_text()).items():
            if key.startswith("SE") and len(samples) > 1:
                (c0, r0), (c1, r1) = samples[0], samples[-1]
                ghz.append((c1 - c0) / ((r1 - r0) / REALTIME_HZ) / 1e9)
    return asm.get(0, "?").lstrip("; ").strip(), waves, partial, statistics.median(ghz) if ghz else None


def merged_windows(waves):
    """Per SIMD: starts, ends and prefix lengths of the merged MFMA busy windows of all its waves."""
    per_simd = defaultdict(list)
    for w in waves:
        per_simd[w["simd"]] += [(t, t + c) for t, c in w["mfma"]]
    merged = {}
    for simd, windows in per_simd.items():
        starts, ends = [], []
        for a, b in sorted(windows):
            if ends and a <= ends[-1]:
                ends[-1] = max(ends[-1], b)
            else:
                starts.append(a)
                ends.append(b)
        prefix = [0]
        for a, b in zip(starts, ends):
            prefix.append(prefix[-1] + b - a)
        merged[simd] = (starts, ends, prefix)
    return merged


def busy_cycles(merged, lo, hi):
    starts, ends, prefix = merged
    i, j = bisect_right(ends, lo), bisect_left(starts, hi)
    if hi <= lo or i >= j:
        return 0
    return prefix[j] - prefix[i] - max(0, lo - starts[i]) - max(0, ends[j - 1] - hi)


def analyze(path):
    path = Path(path)
    kernel, waves, partial, ghz = load_dispatch(path)
    merged = merged_windows(waves)
    working = [w for w in waves if w["mfma"]]
    empty = [w for w in waves if not w["mfma"]]
    cycles, mfma = defaultdict(int), defaultdict(int)
    for w in working:
        first = min(t for t, _ in w["mfma"])
        last = min(max(t + c for t, c in w["mfma"]), w["end"])
        for phase, lo, hi in (("prolog", w["begin"], first), ("mainloop", first, last), ("epilog", last, w["end"])):
            cycles[phase] += hi - lo
            mfma[phase] += busy_cycles(merged[w["simd"]], lo, hi)
    life = sum(cycles.values())
    empty_life = sum(w["end"] - w["begin"] for w in empty)
    span = defaultdict(lambda: [float("inf"), 0])
    for w in waves:
        span[w["simd"]][0] = min(span[w["simd"]][0], w["begin"])
        span[w["simd"]][1] = max(span[w["simd"]][1], w["end"])
    simd_busy = sum(m[2][-1] for m in merged.values())
    dispatch = int(re.search(r"dispatch_(\d+)", path.name).group(1))
    return dict(kernel=kernel, dispatch=dispatch, working=len(working), empty=len(empty), partial=partial, ghz=ghz,
                mfma_per_wave=statistics.mean(len(w["mfma"]) for w in working) if working else 0,
                wave_cycles=life / max(len(working), 1), empty_cycles=empty_life / max(len(empty), 1),
                empty_share=empty_life / max(life + empty_life, 1), wave_mfma=sum(mfma.values()) / max(life, 1),
                span_mfma=simd_busy / max(sum(b - a for a, b in span.values()), 1),
                phases={p: dict(share=cycles[p] / max(life, 1), cycles=cycles[p] / max(len(working), 1),
                                mfma=mfma[p] / max(cycles[p], 1)) for p in PHASES})


def report(dirs):
    results = [analyze(d) for d in dirs]
    print(f"\n{'kernel':22s} {'disp':>4s} {'waves':>11s} {'MFMA/w':>6s} | {'prolog':^20s} | {'mainloop':^20s} |"
          f" {'epilog':^20s} | {'wave':>5s} {'cycles':>7s} {'empty':>6s} {'SIMD':>5s} {'GHz':>5s}")
    print(f"{'':22s} {'':4s} {'work/empty':>11s} {'':6s} | {'share cyc MFMA':^20s} | {'share cyc MFMA':^20s} |"
          f" {'share cyc MFMA':^20s} | {'MFMA':>5s} {'':7s} {'':6s} {'MFMA':>5s}")
    for r in sorted(results, key=lambda r: (r["kernel"], r["dispatch"])):
        cols = " | ".join(f"{q['share']:5.1%} {q['cycles'] / 1e3:6.1f}k {q['mfma']:5.1%}"
                          for q in (r["phases"][p] for p in PHASES))
        print(f"{r['kernel'][:22]:22s} {r['dispatch']:4d} {r['working']:5d}/{r['empty']:<5d} {r['mfma_per_wave']:6.0f} |"
              f" {cols} | {r['wave_mfma']:5.1%} {r['wave_cycles'] / 1e3:6.1f}k {r['empty_share']:6.1%}"
              f" {r['span_mfma']:5.1%} {r['ghz'] or 0:5.2f}" + (f"  ({r['partial']} waves not stitched)" if r["partial"] else ""))
    print("share = phase cycles / working-wave cycles; cyc = mean cycles per working wave; MFMA = SIMD MFMA busy in the"
          " phase;\nwave MFMA = over the whole working wave; empty = cycles of MFMA-free (exiting) waves / all traced"
          " wave cycles;\nSIMD MFMA = MFMA busy over each traced SIMD's span from its first wave start to its last wave end")


def disassemble(co):
    out = subprocess.run([OBJDUMP, "-d", "--mcpu=gfx942", co], capture_output=True, text=True, check=True).stdout
    code, labels = {}, {}
    for line in out.splitlines():
        if m := LABEL_RE.match(line):
            labels[m.group(2)] = int(m.group(1), 16)
        elif m := INST_RE.match(line):
            code[int(m.group(3), 16)] = (m.group(1), m.group(2).strip(), 4 * len(m.group(4).split()))
    return code, labels


def category(mnemonic):
    if mnemonic.startswith(("s_cbranch", "s_branch")):
        return None  # JUMP when taken, NEXT when not
    if mnemonic.startswith(("s_load", "s_buffer_load", "s_store", "s_dcache", "s_memtime", "s_memrealtime")):
        return SMEM
    if mnemonic.startswith(("s_waitcnt", "s_barrier", "s_nop", "s_setprio", "s_sleep", "s_icache", "s_ttracedata")):
        return IMMED
    if mnemonic.startswith(("s_endpgm", "s_sendmsg")):
        return MESSAGE
    if mnemonic.startswith("s_"):
        return SALU
    for prefix, cat in (("buffer_", VMEM), ("global_", FLAT), ("flat_", FLAT), ("scratch_", FLAT), ("ds_", LDS)):
        if mnemonic.startswith(prefix):
            return cat
    return VALU


def replay_pcs(records, code, labels, entry):
    """PC of every trace record, from walking the code with the trace's taken (JUMP) / not-taken (NEXT) branches.

    The decoder loses the PC in the first s_setvskip region: a vector instruction skipped there still issues, as an
    IMMED record."""
    pc, vskip, pcs = entry, False, []
    for r in records:
        mnemonic, operands, size = code[pc]
        cat, expected = int(r[1]), category(mnemonic)
        if expected is None:
            ok = cat in (JUMP, NEXT)
        else:
            ok = cat == expected or (vskip and cat == IMMED and expected in (VMEM, FLAT, LDS, VALU))
        if not ok:
            raise ValueError(f"category {cat} at {pc:#x} {mnemonic}: replay lost sync")
        pcs.append(pc)
        if mnemonic == "s_setvskip":
            vskip = operands.replace(" ", "") != "0,0"
        taken = expected is None and (cat == JUMP or mnemonic == "s_branch")
        pc = labels[operands.split()[0]] if taken else pc + size
    return pcs


def analyze_replayed(path, code, labels):
    """Per innermost MFMA loop (K loops) and the rest of the wave: cycle share and SIMD MFMA busy."""
    path = Path(path)
    kernel = json.loads((path / "code.json").read_text())["code"][0][0].lstrip("; ").strip()
    waves = []
    for wave_file in sorted(path.glob("se*_sm*_sl*_wv*.json")):
        wave = json.loads(wave_file.read_text())["wave"]
        records = wave["instructions"]
        pcs = replay_pcs(records, code, labels, labels[kernel])
        times = [int(r[0]) + int(r[2]) for r in records]
        waves.append(dict(simd=(int(WAVE_RE.fullmatch(wave_file.name).group(1)), int(wave["cu"]), int(wave["simd"])),
                          begin=int(wave["begin"]), end=int(wave["end"]), pcs=pcs, times=times, records=records,
                          mfma=[(t, 32 if "32x32" in code[p][0] else 16)
                                for t, p in zip(times, pcs) if code[p][0].startswith("v_mfma")]))
    loops = set()
    for w in waves:
        for r, p in zip(w["records"], w["pcs"]):
            mnemonic, operands, _ = code[p]
            if category(mnemonic) is None and (int(r[1]) == JUMP or mnemonic == "s_branch"):
                target = labels[operands.split()[0]]
                if target <= p:
                    loops.add((target, p))
    innermost = {}
    for pc in code:
        inside = [loop for loop in loops if loop[0] <= pc <= loop[1]]
        innermost[pc] = min(inside, key=lambda loop: loop[1] - loop[0]) if inside else None
    mfma_loops = sorted({innermost[pc] for pc in code if innermost[pc] and code[pc][0].startswith("v_mfma")})
    merged = merged_windows(waves)
    cycles, busy, entries = defaultdict(int), defaultdict(int), defaultdict(int)
    end_marker = object()
    for w in waves:
        regions = [innermost[p] if innermost[p] in mfma_loops else None for p in w["pcs"]]
        start, current = w["begin"], regions[0]
        for t, region in zip(w["times"][1:] + [w["end"]], regions[1:] + [end_marker]):
            if region != current:
                cycles[current] += t - start
                busy[current] += busy_cycles(merged[w["simd"]], start, t)
                entries[current] += 1
                start, current = t, region
    life = sum(w["end"] - w["begin"] for w in waves)
    static = {loop: sum(code[a][0].startswith("v_mfma") for a in code if loop[0] <= a <= loop[1]) for loop in mfma_loops}
    dispatch = int(re.search(r"dispatch_(\d+)", path.name).group(1))
    return dict(kernel=kernel, dispatch=dispatch, waves=len(waves), life=life / len(waves),
                mfma_per_wave=statistics.mean(len(w["mfma"]) for w in waves),
                wave_mfma=sum(m[2][-1] for m in merged.values()) / life,
                regions=[(f"loop {lo:#x}-{hi:#x} ({static[(lo, hi)]} MFMA/iter)", (lo, hi)) for lo, hi in mfma_loops]
                + [("outside the MFMA loops", None)],
                cycles=cycles, busy=busy, entries=entries)


def replay_report(dirs, co):
    code, labels = disassemble(co)
    for r in sorted((analyze_replayed(d, code, labels) for d in dirs), key=lambda r: r["dispatch"]):
        print(f"\n{r['kernel'][:60]} dispatch {r['dispatch']}: {r['waves']} traced waves, {r['life'] / 1e3:.0f}k cycles"
              f" and {r['mfma_per_wave']:.0f} MFMAs per wave, wave MFMA {r['wave_mfma']:.1%}")
        total = sum(r["cycles"].values())
        for name, key in r["regions"]:
            n = max(r["entries"][key], 1)
            print(f"  {name:40s} share {r['cycles'][key] / total:6.1%}  {r['cycles'][key] / n / 1e3:7.1f}k cycles x{n / r['waves']:5.1f}"
                  f" per wave  MFMA {r['busy'][key] / max(r['cycles'][key], 1):6.1%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    trace_p, att_p, phases_p, replay_p = (sub.add_parser(name) for name in ("trace", "att", "phases", "replay"))
    for p in (trace_p, att_p):
        p.add_argument("--tpr", type=int, default=4096)
        p.add_argument("--seed", type=int, default=101, help="seeds seed and seed + 100 (the harness tpr4096 pair)")
        p.add_argument("--calls", type=int, default=10)
    trace_p.add_argument("--impl", choices=tuple(KERNELS), required=True)
    trace_p.add_argument("--sort", choices=("fast", "ck"), required=True)
    att_p.add_argument("--gpu", default=os.environ.get("HIP_VISIBLE_DEVICES", "0"))
    att_p.add_argument("--impls", nargs="+", choices=tuple(KERNELS), default=["geak", "pyhip"])
    att_p.add_argument("--iterations", default="[7-9]", help="rocprofv3 --kernel-iteration-range")
    att_p.add_argument("--wait-idle", type=float, default=0, help="minutes to wait for an idle GPU before each trace")
    att_p.add_argument("--out-dir", default="/tmp/pyhip_vs_geak_att")
    phases_p.add_argument("dirs", nargs="+")
    replay_p.add_argument("dirs", nargs="+")
    replay_p.add_argument("--co", default=ASM_CO, help="code object of the traced kernel")
    args = parser.parse_args()
    if args.command == "trace":
        trace(args)
    elif args.command == "att":
        att(args)
    elif args.command == "replay":
        replay_report(args.dirs, args.co)
    else:
        report(args.dirs)


if __name__ == "__main__":
    main()
