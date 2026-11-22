#!/usr/bin/env python3
"""GEAK harness: MiMo-V2.6-Flash-MOPD routed-expert MoE, EP8 rank-local fp8 block-scale fused MoE.

Runs on ONE GPU and reproduces what EP rank 0 executes in SGLang MoriEPMoE: aiter fused_moe on the
MORI fp8-dispatched receive buffer. Do not edit; the code under test is moe_ep.py.
Usage: python3 scripts/task_runner.py {compile|correctness|performance|profile}
"""
import argparse
import os
import sys

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_DIR)
os.chdir(TASK_DIR)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# XiaomiMiMo/MiMo-V2.6-Flash-MOPD config.json; experts run as fp8 block-scale [128, 128].
HIDDEN = 4096
INTER = 2048
NUM_EXPERTS = 256
TOPK = 8
BLK = 128
# Simulated expert parallelism: this process is rank 0 of 8 and owns experts [0, 32).
SIM_EP_SIZE = 8
SIM_EP_RANK = 0
LOCAL_EXPERTS = NUM_EXPERTS // SIM_EP_SIZE
# (case_id, tokens_per_rank, SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK); receive buffer = 8 * the latter.
CASES = [("tpr2048", 2048, 4096), ("tpr4096", 4096, 4096)]
REL_TOL = 2e-2  # global relative L2 error vs the reference
ROW_TOL = 5e-2  # worst per-token relative L2 error

torch.backends.cuda.matmul.allow_tf32 = False
ARCH = torch.cuda.get_device_properties(0).gcnArchName
FP8 = torch.float8_e4m3fnuz if ARCH.startswith("gfx94") else torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max


def quant_1x128(x):
    g = x.float().view(x.shape[0], -1, BLK)
    s = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10) / FP8_MAX
    return (g / s).to(FP8).view(x.shape), s.squeeze(-1).contiguous()


def dequant_1x128(q, s):
    return (q.float().view(q.shape[0], -1, BLK) * s.unsqueeze(-1)).view(q.shape)


def dequant_128x128(w, s):
    n, k = w.shape
    w = w.float().view(n // BLK, BLK, k // BLK, BLK) * s.view(n // BLK, 1, k // BLK, 1)
    return w.view(n, k)


def make_weights(seed=0):
    """Unshuffled fp8 local-expert weights + fp32 dequant scales, as loaded from the checkpoint."""
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def block_quantized(n, k):
        q = torch.empty(LOCAL_EXPERTS, n, k, dtype=FP8, device="cuda")
        s = torch.empty(LOCAL_EXPERTS, n // BLK, k // BLK, device="cuda")
        for e in range(LOCAL_EXPERTS):
            b = torch.randn(n, k, device="cuda", generator=gen).mul_(0.02)
            b = b.view(n // BLK, BLK, k // BLK, BLK)
            se = b.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-10) / FP8_MAX
            q[e] = (b / se).to(FP8).view(n, k)
            s[e] = se.view(n // BLK, k // BLK)
        return q, s

    w13, w13_scale = block_quantized(2 * INTER, HIDDEN)  # rows [0, INTER) gate, [INTER, 2*INTER) up
    w2, w2_scale = block_quantized(HIDDEN, INTER)
    return w13, w2, w13_scale, w2_scale


def make_inputs(tokens_per_rank, max_dispatch_per_rank, seed):
    """Rank-0 MORI dispatch output: rows [0, n_recv) are valid, the rest are stale and must be ignored."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    # MiMo router over every rank's tokens: sigmoid, noaux_tc with n_group=1 (zero bias), renormalized top-8.
    n_all = SIM_EP_SIZE * tokens_per_rank
    scores = torch.randn(n_all, NUM_EXPERTS, device="cuda", generator=gen).sigmoid()
    w_all, ids_all = scores.topk(TOPK, dim=-1)
    w_all = w_all / w_all.sum(dim=-1, keepdim=True)
    lo = SIM_EP_RANK * LOCAL_EXPERTS
    recv = ((ids_all >= lo) & (ids_all < lo + LOCAL_EXPERTS)).any(dim=-1)
    n_recv = int(recv.sum())

    rows = SIM_EP_SIZE * max(tokens_per_rank, max_dispatch_per_rank)
    topk_ids = torch.randint(
        0, NUM_EXPERTS, (rows, TOPK), device="cuda", generator=gen, dtype=torch.int32
    )
    topk_weights = torch.rand(rows, TOPK, device="cuda", generator=gen)
    topk_ids[:n_recv] = ids_all[recv].to(torch.int32)
    topk_weights[:n_recv] = w_all[recv]
    x = torch.randn(rows, HIDDEN, device="cuda", generator=gen, dtype=torch.bfloat16)
    x, x_scale = quant_1x128(x)
    expert_mask = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device="cuda")
    expert_mask[lo : lo + LOCAL_EXPERTS] = 1
    inputs = dict(
        x=x,
        x_scale=x_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        expert_mask=expert_mask,
        num_local_tokens=torch.tensor([n_recv], dtype=torch.int32, device="cuda"),
    )
    return inputs, n_recv


def reference(w13, w2, w13_scale, w2_scale, inputs, n_recv):
    x = dequant_1x128(inputs["x"][:n_recv], inputs["x_scale"][:n_recv])
    ids = inputs["topk_ids"][:n_recv]
    weights = inputs["topk_weights"][:n_recv]
    out = torch.zeros(n_recv, HIDDEN, device="cuda")
    for e in range(LOCAL_EXPERTS):
        tok, slot = (ids == SIM_EP_RANK * LOCAL_EXPERTS + e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        gate, up = (x[tok] @ dequant_128x128(w13[e], w13_scale[e]).t()).chunk(2, dim=-1)
        # The fused kernel feeds the down projection fp8 activations with 1x128 scales.
        act = dequant_1x128(*quant_1x128(F.silu(gate) * up))
        y = act @ dequant_128x128(w2[e], w2_scale[e]).t()
        out.index_add_(0, tok, y * weights[tok, slot].unsqueeze(-1))
    return out


def compare(out, ref, n_recv):
    if (
        not torch.is_tensor(out)
        or out.dtype != torch.bfloat16
        or out.dim() != 2
        or out.shape[0] < n_recv
        or out.shape[1] != HIDDEN
    ):
        return False, f"bad output: {getattr(out, 'shape', None)} {getattr(out, 'dtype', None)}"
    got = out[:n_recv].float()
    if not torch.isfinite(got).all():
        return False, "non-finite values in valid rows"
    diff = got - ref
    rel = (diff.norm() / ref.norm()).item()
    row = (diff.norm(dim=1) / ref.norm(dim=1).clamp(min=1e-6)).max().item()
    ok = rel <= REL_TOL and row <= ROW_TOL
    return ok, f"rel_l2={rel:.3e} (tol {REL_TOL:g}) max_token_rel={row:.3e} (tol {ROW_TOL:g})"


def load_impl():
    import moe_ep

    # Both register torch op aiter::fused_moe_ and the first import wins silently.
    if "aiter.fused_moe" in sys.modules and "aiter_fused_moe" in sys.modules:
        sys.exit("aiter.fused_moe and aiter_fused_moe were both imported: edits to the copy may be bypassed")
    return moe_ep


def run_correctness():
    impl = load_impl()
    raw = make_weights()
    prepared = impl.prepare_weights(*(t.clone() for t in raw))
    ok_all = True
    for i, (cid, tpr, max_disp) in enumerate(CASES):
        inputs, n_recv = make_inputs(tpr, max_disp, seed=100 + i)
        ref = reference(*raw, inputs, n_recv)
        out = impl.forward(prepared, **inputs)
        torch.cuda.synchronize()
        ok, msg = compare(out, ref, n_recv)
        rows = inputs["topk_ids"].shape[0]
        print(f"{'PASS' if ok else 'FAIL'} {cid}: rows={rows} valid={n_recv} {msg}", flush=True)
        ok_all = ok_all and ok
    return ok_all


def run_performance(warmup=10, iters=100):
    impl = load_impl()
    prepared = impl.prepare_weights(*make_weights())
    for i, (cid, tpr, max_disp) in enumerate(CASES):
        # Two input sets alternate so per-input caching cannot fake a speedup.
        sets = [make_inputs(tpr, max_disp, seed=s)[0] for s in (100 + i, 200 + i)]
        for k in range(warmup):
            impl.forward(prepared, **sets[k % 2])
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for k in range(iters):
            impl.forward(prepared, **sets[k % 2])
        end.record()
        torch.cuda.synchronize()
        print(f"Perf: {start.elapsed_time(end) / iters:.4f} ms ({cid})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["compile", "correctness", "performance", "profile"])
    mode = parser.parse_args().mode
    if mode == "compile":
        load_impl()
        print("Compilation: PASS")
    elif mode == "correctness":
        ok = run_correctness()
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
    elif mode == "performance":
        run_performance()
    else:
        run_performance(warmup=3, iters=3)


if __name__ == "__main__":
    main()
