#!/usr/bin/env python3
"""Op-level A/B of the MiMo-V2.6-Flash EP8 FP8 block-scale routed MoE on one gfx942 rank: pyhip FlyDSL GateUp + Down
(tingqli/pyhip luwei/942-fp8-blockwise) vs GEAK's optimized moe_ep.py, both driven by the GEAK task harness
(scripts/task_runner.py): same weights, MORI receive buffers, reference, tolerances and timing loop.

GEAK's kernels build with aiter's pinned FlyDSL (0.1.5.dev504), pyhip's with FlyDSL 0.4.0, so each (impl, sort) runs in
its own process; `compare` runs them alternately on one GPU and prints the tables.

  python bench_moe_op.py compare [--gpu 0] [--passes 2] [--rounds 3] [--tpr 2048 4096 8192] [--wait-idle 30]
  python bench_moe_op.py run --impl {geak,pyhip} --sort {fast,ck} --json out.json
  python bench_moe_op.py report [--out-dir DIR]

--sort is the moe_sorting both sides are fed; each keeps its own M block (GEAK 64, pyhip 256):
  ck   aiter CK moe_sorting_fwd, as aiter.fused_moe and the pyhip aiter hook call it (GEAK: its MOE_FAST_SORT=0 path)
  fast GEAK's moe_sort_fast (GEAK also fuses its moe_buf zero-fill and per-block valid counts into it)
"""

import argparse
import glob
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from importlib.metadata import version

HERE = os.path.dirname(os.path.abspath(__file__))
GEAK_TASK = os.environ.get("GEAK_TASK", os.path.join(os.path.dirname(HERE), "mimo_fp8bs_moe_ep8"))
GEAK_OPT = os.environ.get("GEAK_OPT", os.path.join(os.path.dirname(HERE), "geak_run", "optimized"))
PYHIP_SRC = os.environ.get("PYHIP_SRC", "/sgl-workspace/pyhip/src")
PYHIP_FLYDSL = os.environ.get("PYHIP_FLYDSL", os.path.join(HERE, ".flydsl040"))
CONFIGS = [("geak", "fast"), ("pyhip", "fast"), ("geak", "ck"), ("pyhip", "ck")]
PAIRS = [("same sort: aiter CK moe_sorting", "ck", "ck"), ("same sort: GEAK moe_sort_fast", "fast", "fast"),
         ("as shipped (own sort)", "fast", "ck")]
DIFFS = [(("geak", "ck"), ("pyhip", "ck")), (("geak", "fast"), ("pyhip", "fast")), (("geak", "fast"), ("pyhip", "ck")),
         (("geak", "fast"), ("geak", "ck")), (("pyhip", "fast"), ("pyhip", "ck"))]
CATEGORIES = [("sort", re.compile(r"(?i)sort|k_count|k_place")), ("gemm1", re.compile(r"gateup_kernel|gemm1")),
              ("gemm2", re.compile(r"down_kernel|s2_fp8bs"))]
IDLE_PCT = 5
MAX_SPREAD = 0.03


def load_harness():
    cwd, path = os.getcwd(), list(sys.path)
    spec = importlib.util.spec_from_file_location("task_runner", os.path.join(GEAK_TASK, "scripts", "task_runner.py"))
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)  # it chdirs into the task dir and prepends that to sys.path: undo both
    os.chdir(cwd)
    sys.path[:] = path
    return harness


def geak_impl(sort):
    os.environ["MOE_FAST_SORT"] = "1" if sort == "fast" else "0"
    sys.path.insert(0, GEAK_OPT)
    import moe_ep

    return moe_ep.prepare_weights, moe_ep.forward


def pyhip_impl(sort):
    import pyhip.contrib.flydsl.moe_a8w8_blockscale_gfx942 as pm
    import torch
    from aiter.ops.moe_sorting import moe_sorting_fwd
    from aiter.ops.shuffle import shuffle_weight

    msf = None
    if sort == "fast":
        sys.path.insert(0, GEAK_OPT)
        import moe_sort_fast as msf

        assert msf.available(), "moe_sort_fast failed to build"
    block = pm.SORT_BLOCK_M

    def prepare(w13, w2, w13_scale, w2_scale):
        return {"w1": shuffle_weight(w13, (16, 16)), "w2": shuffle_weight(w2, (16, 16)),
                "w1_scale": w13_scale.contiguous(), "w2_scale": w2_scale.contiguous()}

    def forward(weights, x, x_scale, topk_weights, topk_ids, expert_mask, num_local_tokens):
        # What aiter's AITER_FMOE_FP8_BLOCKSCALE_FLYDSL=1 hook (luocheng25/aiter ac2b48fe) runs for MORI FP8 input.
        M, topk = topk_ids.shape
        E, H = expert_mask.numel(), x.shape[1]
        local_experts, _, inter_dim = weights["w2"].shape
        reason = pm.unsupported_reason(M, H, inter_dim, topk, local_experts)
        assert not reason, reason
        a1_scale_t = x_scale.t().contiguous()  # GateUp reads [H/128, M]; MORI hands over token-major scales
        if msf is not None:
            sorted_ids, sorted_w, sorted_eids, num_valid, _ = msf.moe_sorting_fast(
                topk_ids, topk_weights, E, block, expert_mask, num_local_tokens)
        else:
            pad = topk_ids.numel() + E * block - topk
            sorted_ids = torch.empty(pad, dtype=torch.int32, device=x.device)
            sorted_w = torch.empty(pad, dtype=torch.float32, device=x.device)
            sorted_eids = torch.empty(-(-pad // block), dtype=torch.int32, device=x.device)
            num_valid = torch.empty(2, dtype=torch.int32, device=x.device)
            # An empty moe_buf keeps the sort from clearing anything: the pipeline zeroes the rows it does not store.
            moe_sorting_fwd(topk_ids, topk_weights, sorted_ids, sorted_w, sorted_eids, num_valid,
                            torch.empty((0, 0), dtype=torch.bfloat16, device=x.device), E, block, expert_mask,
                            num_local_tokens, 0)
        out = torch.empty((M, H), dtype=torch.bfloat16, device=x.device)
        return pm.fused_moe_a8w8_blockscale_gfx942(
            x, a1_scale_t, weights["w1"], weights["w1_scale"], weights["w2"], weights["w2_scale"], sorted_ids,
            sorted_w, sorted_eids, num_valid, out, topk, num_rows=num_local_tokens)

    return prepare, forward


def time_rounds(forward, weights, sets, rounds, warmup, iters):
    """The harness's performance loop (two alternating input sets, one event pair around iters calls), per round."""
    import torch

    ms = []
    for _ in range(rounds):
        for k in range(warmup):
            forward(weights, **sets[k % 2])
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for k in range(iters):
            forward(weights, **sets[k % 2])
        end.record()
        torch.cuda.synchronize()
        ms.append(start.elapsed_time(end) / iters)
    return ms


def kernel_times(forward, weights, sets, calls):
    """{kernel name: [us per call, launches per call, fastest launch us]} from a torch profiler trace."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    for k in range(4):
        forward(weights, **sets[k % 2])
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for k in range(calls):
            forward(weights, **sets[k % 2])
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json") as trace:
        prof.export_chrome_trace(trace.name)
        with open(trace.name) as f:
            events = [e for e in json.load(f)["traceEvents"] if e.get("cat") == "kernel"]
    durations = defaultdict(list)
    for e in events:
        durations[e["name"]].append(e["dur"])
    return {name: [sum(d) / calls, len(d) / calls, min(d)] for name, d in durations.items()}


def run(args):
    import torch

    harness = load_harness()
    prepare, forward = (geak_impl if args.impl == "geak" else pyhip_impl)(args.sort)
    raw = harness.make_weights()
    weights = prepare(*(t.clone() for t in raw))
    lo = harness.SIM_EP_RANK * harness.LOCAL_EXPERTS
    gpu = os.environ.get("HIP_VISIBLE_DEVICES", "0")
    result = dict(impl=args.impl, sort=args.sort, flydsl=version("flydsl"), device=torch.cuda.get_device_name(), cases={})
    for i, tpr in enumerate(args.tpr):
        # Receive buffer = 8 * max(tpr, MORI max dispatch) rows; with 4096 this reproduces the harness CASES.
        cid, max_disp = f"tpr{tpr}", max(tpr, 4096)
        sets, valid = zip(*(harness.make_inputs(tpr, max_disp, seed=s) for s in (100 + i, 200 + i)))
        refs = [harness.reference(*raw, s, n) for s, n in zip(sets, valid)]
        # A MORI receive buffer holds arbitrary data past num_local_tokens: NaN scales there must not reach valid rows.
        stale = dict(sets[0], x_scale=sets[0]["x_scale"].clone())
        stale["x_scale"][valid[0]:] = float("nan")
        checks = {}
        for name, inputs, j in (("seed%d" % (100 + i), sets[0], 0), ("seed%d" % (200 + i), sets[1], 1),
                                ("seed%d stale NaN" % (100 + i), stale, 0)):
            out = forward(weights, **inputs)
            torch.cuda.synchronize()
            checks[name] = harness.compare(out, refs[j], valid[j])
            if inputs is sets[0] and args.save_dir:
                ids = inputs["topk_ids"][: valid[0]]
                routes = ((ids >= lo) & (ids < lo + harness.LOCAL_EXPERTS)).sum(dim=1)
                torch.save({"out": out[: valid[0]].cpu(), "routes": routes.cpu()},
                           os.path.join(args.save_dir, f"{args.impl}_{args.sort}_{cid}.pt"))
            del out
        del refs, stale
        busy = [wait_idle(gpu, args.wait_idle)]
        ms = time_rounds(forward, weights, sets, args.rounds, args.warmup, args.iters)
        kernels = kernel_times(forward, weights, sets, args.profile_calls)
        busy.append(gpu_activity(gpu))
        result["cases"][cid] = dict(rows=sets[0]["topk_ids"].shape[0], valid=valid[0], checks=checks, ms=ms,
                                    kernels=kernels, gpu_busy=busy)
        print(f"{args.impl}/{args.sort} {cid}: " + " ".join("PASS" if ok else "FAIL" for ok, _ in checks.values())
              + "  ms " + " ".join(f"{t:.4f}" for t in ms) + f"  GPU busy before/after {busy[0]}/{busy[1]} %",
              flush=True)
        del sets
        torch.cuda.empty_cache()
    with open(args.json, "w") as f:
        json.dump(result, f, indent=1)


def gpu_activity(gpu):
    try:
        out = subprocess.run(["amd-smi", "metric", "-g", gpu, "-u"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"GFX_ACTIVITY:\s*(\d+)", out)
    return int(match.group(1)) if match else None


def wait_idle(gpu, minutes):
    """GFX activity once two readings in a row are idle, or the last reading when `minutes` run out."""
    deadline, quiet = time.time() + minutes * 60, 0
    while True:
        busy = gpu_activity(gpu)
        quiet = quiet + 1 if busy is not None and busy <= IDLE_PCT else 0
        if quiet >= 2 or time.time() >= deadline:
            return busy
        time.sleep(10)


def compare(args):
    os.makedirs(args.out_dir, exist_ok=True)
    # pyhip's pyproject maps the package root to src/, so expose src as "pyhip" without installing it.
    pkg = os.path.join(HERE, ".pyhip_pkg")
    os.makedirs(pkg, exist_ok=True)
    if not os.path.exists(os.path.join(pkg, "pyhip")):
        os.symlink(PYHIP_SRC, os.path.join(pkg, "pyhip"))
    # Passes accumulate across invocations; report() drops the samples another tenant disturbed.
    first = len(glob.glob(os.path.join(args.out_dir, "%s_%s_p*.json" % CONFIGS[0])))
    for p in range(first, first + args.passes):
        for impl, sort in CONFIGS if p % 2 == 0 else CONFIGS[::-1]:
            env = dict(os.environ, HIP_VISIBLE_DEVICES=args.gpu, PYTHONDONTWRITEBYTECODE="1")
            if impl == "pyhip":
                env["PYTHONPATH"] = os.pathsep.join(filter(None, [PYHIP_FLYDSL, pkg, env.get("PYTHONPATH")]))
            path = os.path.join(args.out_dir, f"{impl}_{sort}_p{p}.json")
            cmd = [sys.executable, os.path.abspath(__file__), "run", "--impl", impl, "--sort", sort, "--json", path,
                   "--rounds", str(args.rounds), "--warmup", str(args.warmup), "--iters", str(args.iters),
                   "--profile-calls", str(args.profile_calls), "--wait-idle", str(args.wait_idle),
                   "--tpr", *map(str, args.tpr)]
            if p == 0:
                cmd += ["--save-dir", args.out_dir]
            subprocess.run(cmd, env=env, check=True)
    report(args)


def is_clean(run, cid):
    """Idle GPU before the timed window and a steady harness loop (another tenant shows up as a spread)."""
    case = run["cases"][cid]
    ms = case["ms"]
    # The reading right after the window still averages in our own last kernels, so only the one before counts.
    busy = case.get("gpu_busy", run.get("gpu_busy", [None]))[0]
    return (busy is None or busy <= IDLE_PCT) and (max(ms) - min(ms)) / min(ms) <= MAX_SPREAD


def report(args):
    import torch

    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.out_dir, "*_p*.json"))):
        with open(path) as f:
            r = json.load(f)
        runs[(r["impl"], r["sort"])].append(r)
    cases = list(next(iter(runs.values()))[0]["cases"])
    used = {}
    for key in CONFIGS:
        for cid in cases:
            clean = [r for r in runs[key] if is_clean(r, cid)]
            used[(key, cid)] = (clean or runs[key], not clean)
    for (impl, sort), rs in sorted(runs.items()):
        kept = ", ".join(f"{cid} {len(used[((impl, sort), cid)][0])}" for cid in cases)
        print(f"{impl:5s} sort={sort:4s}: FlyDSL {rs[0]['flydsl']}, {rs[0]['device']}, {len(rs)} pass(es); clean: {kept}")

    print("\n== correctness vs the GEAK harness reference (rel_l2 <= 2e-2, max_token_rel <= 5e-2) ==")
    for key in CONFIGS:
        for cid in cases:
            for name, (ok, msg) in runs[key][0]["cases"][cid]["checks"].items():
                print(f"{key[0]:5s} sort={key[1]:4s} {cid} {name:20s} {'PASS' if ok else 'FAIL'}  {msg}")

    samples = {k: [t for r in rs for t in r["cases"][k[1]]["ms"]] for k, (rs, _) in used.items()}
    flag = {k: "*" if contended else " " for k, (_, contended) in used.items()}
    print("\n== whole forward, ms/call (harness loop; median and min over the clean rounds x passes) ==")
    print(f"{'case':8s} {'comparison':34s} {'GEAK med':>9s} {'pyhip med':>9s} {'speedup':>8s}"
          f" {'GEAK min':>9s} {'pyhip min':>9s} {'speedup':>8s}")
    for cid in cases:
        for label, gs, ps in PAIRS:
            gk, pk = (("geak", gs), cid), (("pyhip", ps), cid)
            g, p = samples[gk], samples[pk]
            gm, pm_ = statistics.median(g), statistics.median(p)
            print(f"{cid:8s} {label:34s} {gm:8.4f}{flag[gk]} {pm_:8.4f}{flag[pk]} {pm_ / gm:7.3f}x"
                  f" {min(g):9.4f} {min(p):9.4f} {min(p) / min(g):7.3f}x")
    print("speedup = pyhip / GEAK time (> 1: GEAK faster); * = no clean sample, all (contended) samples used")

    print("\n== GPU kernel time per call, us (torch profiler; mean over calls and clean passes / fastest launch) ==")
    print(f"{'case':8s} {'impl':5s} {'sort':4s} {'sort':>13s} {'other':>13s} {'gemm1':>13s} {'gemm2':>13s}"
          f" {'sum':>15s} {'launch':>6s}")
    for cid in cases:
        for key in CONFIGS:
            rs = used[(key, cid)][0]
            kernels = defaultdict(lambda: [0.0, 0.0, float("inf")])
            for r in rs:
                for name, (us, n, fastest) in r["cases"][cid]["kernels"].items():
                    kernels[name][0] += us / len(rs)
                    kernels[name][1] += n / len(rs)
                    kernels[name][2] = min(kernels[name][2], fastest)
            mean, best = defaultdict(float), defaultdict(float)
            for name, (us, n, fastest) in kernels.items():
                cat = next((c for c, rx in CATEGORIES if rx.search(name)), "other")
                mean[cat] += us
                best[cat] += fastest * n
            cols = " ".join(f"{mean[c]:6.1f}/{best[c]:6.1f}" for c in ("sort", "other", "gemm1", "gemm2"))
            print(f"{cid:8s} {key[0]:5s} {key[1]:4s} {cols} {sum(mean.values()):7.1f}/{sum(best.values()):7.1f}"
                  f" {sum(n for _, n, _ in kernels.values()):6.1f}")
            if args.kernels:
                for name, (us, n, fastest) in sorted(kernels.items(), key=lambda kv: -kv[1][0]):
                    print(f"{'':20s} {us:8.1f} / {fastest:8.1f} us x{n:.0f}  {name[:90]}")

    print("\n== output A vs B on the same inputs (valid rows; bitwise = share of identical BF16 elements) ==")
    print(f"{'case':8s} {'A':10s} {'B':10s} {'rel_l2':>9s} {'max_abs':>9s} {'bitwise 1-route':>16s}"
          f" {'bitwise multi':>14s}")
    for cid in cases:
        for a_key, b_key in DIFFS:
            paths = [os.path.join(args.out_dir, f"{k[0]}_{k[1]}_{cid}.pt") for k in (a_key, b_key)]
            if not all(os.path.exists(p) for p in paths):
                continue
            a, b = (torch.load(p) for p in paths)
            single = a["routes"] == 1
            same = a["out"].view(torch.int16) == b["out"].view(torch.int16)
            diff = a["out"].float() - b["out"].float()
            print(f"{cid:8s} {'/'.join(a_key):10s} {'/'.join(b_key):10s} {(diff.norm() / b['out'].float().norm()).item():9.2e}"
                  f" {diff.abs().max().item():9.2e} {same[single].float().mean().item():15.3%}"
                  f" {same[~single].float().mean().item():13.3%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p, cmp_p, rep_p = (sub.add_parser(name) for name in ("run", "compare", "report"))
    for p in (run_p, cmp_p):
        p.add_argument("--rounds", type=int, default=3)
        p.add_argument("--warmup", type=int, default=10)
        p.add_argument("--iters", type=int, default=100)
        p.add_argument("--profile-calls", type=int, default=20)
        p.add_argument("--tpr", type=int, nargs="+", default=[2048, 4096], help="tokens per rank of each case")
        p.add_argument("--wait-idle", type=float, default=0,
                       help="minutes to wait for an idle GPU before each case's timed window")
    run_p.add_argument("--impl", choices=("geak", "pyhip"), required=True)
    run_p.add_argument("--sort", choices=("fast", "ck"), required=True)
    run_p.add_argument("--json", required=True)
    run_p.add_argument("--save-dir", help="save the valid rows of the first input set for the A/B output diff")
    cmp_p.add_argument("--gpu", default=os.environ.get("HIP_VISIBLE_DEVICES", "0"))
    cmp_p.add_argument("--passes", type=int, default=2, help="alternating passes over the 4 configs")
    for p in (cmp_p, rep_p):
        p.add_argument("--out-dir", default="/tmp/pyhip_vs_geak")
        p.add_argument("--kernels", action="store_true", help="also list every kernel")
    args = parser.parse_args()
    {"run": run, "compare": compare, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
