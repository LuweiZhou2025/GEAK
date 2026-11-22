"""Code under test (editable): rank-local fp8 block-scale fused MoE for MiMo-V2.6-Flash-MOPD at EP8.

Baseline = what SGLang MoriEPMoE runs: aiter fused_moe(quant_type=per_128x128) on the MORI
fp8-dispatched receive buffer. aiter_fused_moe.py is a verbatim, editable copy of aiter/fused_moe.py.
"""
import os

# FlyDSL disk cache is keyed by source, not by the closure tile constants -> keep it off.
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch
from aiter import ActivationType, QuantType
from aiter.ops.shuffle import shuffle_weight

# Never also import aiter.fused_moe: both register torch op aiter::fused_moe_ and the first one wins.
from aiter_fused_moe import fused_moe, moe_sorting

try:  # FlyDSL native-fp8 block-scale 2-stage path (new kernels in this task dir)
    import flydsl_moe_fp8bs as _fbs
except Exception:  # pragma: no cover - fall back to aiter asm
    _fbs = None

_FBS_TILE_M = int(os.environ.get("MOE_FBS_TILE_M", "64"))
_FBS_TILE_N1 = int(os.environ.get("MOE_FBS_TILE_N1", "128"))
_FBS_TILE_N2 = int(os.environ.get("MOE_FBS_TILE_N2", "256"))
_USE_FBS = os.environ.get("MOE_USE_FBS", "1") == "1"
# stage1 selector: "v2" = flydsl_moe_fp8bs_s1v2.compile_stage1_t (128-row sorting units), else d0 gemm1.
# Default = d0 gemm1: the 128-row s1v2 path measured 0.976x/0.994x vs gemm1 (R1), so repairing s1v2's import
# must not silently switch the forward onto it.
_S1_IMPL = os.environ.get("MOE_S1_IMPL", "gemm1")
_s1v2 = None
if _S1_IMPL == "v2":
    try:
        import flydsl_moe_fp8bs_s1v2 as _s1v2
    except Exception:  # pragma: no cover - fall back to d0 gemm1
        _s1v2 = None
_S1V2_TILE_M = 128
_S1V2_CFG = dict(tile_m=_S1V2_TILE_M, a_stream=2, sched=10, valu_per=3, setprio=1, rsplit=0, rescale=2,
                 split_num=0, roll=1, halfskip=2)
# Stage-2 selector: "s2" = flydsl_moe_fp8bs_s2 (new down+atomic kernel), "d0" = flydsl_moe_fp8bs gemm2.
_S2_IMPL = os.environ.get("MOE_S2_IMPL", "s2")
_s2 = None
if _S2_IMPL == "s2":
    try:
        import flydsl_moe_fp8bs_s2 as _s2
    except Exception:  # pragma: no cover - fall back to d0 gemm2
        _s2 = None
# Fast align&sort (moe_sort_fast.py): 2-kernel HIP drop-in (k_count + k_place) for CK moe_sorting's
# ClearWorkspace/P0/P1/P23 chain with bit-identical sorted outputs. Its placement kernel also owns the moe_buf
# zero-fill (rows [0, T)) through MOE_ZF_SORT_BLOCKS extra blocks that overlap the latency-bound sort work.
# gemm1 stage-1 path only (s1v2 keeps CK); falls back to CK when unsupported or the JIT build fails.
# MOE_FAST_SORT=0 restores CK moe_sorting.
_FAST_SORT = os.environ.get("MOE_FAST_SORT", "1") == "1"
_ZF_SORT_BLOCKS = int(os.environ.get("MOE_ZF_SORT_BLOCKS", "304"))
_msf = None
if _FAST_SORT:
    try:
        import moe_sort_fast as _msf
    except Exception:  # pragma: no cover - fall back to CK moe_sorting
        _msf = None


def prepare_weights(w13, w2, w13_scale, w2_scale):
    """Untimed, once per weight set (= SGLang process_weights_after_loading).

    w13 [32, 4096, 4096] fp8 (gate rows then up rows), w2 [32, 4096, 2048] fp8,
    w13_scale [32, 32, 32] / w2_scale [32, 32, 16] fp32 dequant scales per 128x128 block.
    """
    E, N1, K1 = w13.shape
    fbs_ok = (
        _USE_FBS and _fbs is not None and K1 % 128 == 0 and N1 % 256 == 0
        and tuple(w13_scale.shape) == (E, N1 // 128, K1 // 128)
    )
    if fbs_ok and _msf is not None:
        _msf.available()  # JIT-build the fast sort here (untimed), not in the first forward
    return {
        "fbs": fbs_ok,
        "w13_scale_flat": w13_scale.contiguous().view(-1),
        "w2_scale_flat": w2_scale.contiguous().view(-1),
        "w13": shuffle_weight(w13, (16, 16)),
        "w2": shuffle_weight(w2, (16, 16)),
        "w13_scale": w13_scale,
        "w2_scale": w2_scale,
    }


def forward(weights, x, x_scale, topk_weights, topk_ids, expert_mask, num_local_tokens):
    """Timed. x [rows, 4096] fp8 + x_scale [rows, 32] fp32 (1x128 groups); topk_ids int32 global expert ids
    and topk_weights fp32, both [rows, 8]; expert_mask [256] int32 (1 = local); num_local_tokens [1] int32 =
    valid rows (rows after it are stale). Returns bf16 [>= valid rows, 4096]; only valid rows are checked.
    """
    if weights.get("fbs", False):
        return _forward_fbs(weights, x, x_scale, topk_weights, topk_ids, expert_mask, num_local_tokens)
    return fused_moe(
        hidden_states=x,
        w1=weights["w13"],
        w2=weights["w2"],
        topk_weight=topk_weights,
        topk_ids=topk_ids,
        expert_mask=expert_mask,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_128x128,
        w1_scale=weights["w13_scale"],
        w2_scale=weights["w2_scale"],
        a1_scale=x_scale,
        num_local_tokens=num_local_tokens,
        dtype=torch.bfloat16,
    )


def _forward_fbs(weights, x, x_scale, topk_weights, topk_ids, expert_mask, num_local_tokens):
    rows, model_dim = x.shape
    w13 = weights["w13"]
    E, n1, _ = w13.shape
    inter = n1 // 2
    topk = topk_ids.shape[1]
    tile_m = _FBS_TILE_M
    use_v2 = _s1v2 is not None and _S1V2_TILE_M % tile_m == 0
    unit = _S1V2_TILE_M if use_v2 else tile_m
    blk_cnt = None  # per-M-block valid-row counts (fast sort only) -> gemm1 + stage-2 all-padding subtile skips
    if (not use_v2 and _msf is not None and _msf.supported(topk_ids, topk_weights, expert_mask, num_local_tokens)
            and _msf.available()):
        sid, sw, seid, nvalid, moe_buf, blk_cnt = _msf.moe_sorting_fast(
            topk_ids, topk_weights, expert_mask.numel(), unit, expert_mask, num_local_tokens,
            model_dim=model_dim, zero_fill=True, zfill_blocks=_ZF_SORT_BLOCKS, blk_counts=True,
        )
    else:
        sid, sw, seid, nvalid, moe_buf = moe_sorting(
            topk_ids, topk_weights, expert_mask.numel(), model_dim, torch.bfloat16, unit,
            expert_mask, num_local_tokens,
        )
    S = sid.shape[0]
    blocks = seid.shape[0]
    a2 = torch.empty((S, inter), dtype=torch.uint8, device=x.device)
    a2s = torch.empty((inter // 128, S), dtype=torch.float32, device=x.device)
    stream = torch.cuda.current_stream()
    if use_v2:
        s1 = _s1v2.compile_stage1_t(model_dim=model_dim, inter_dim=inter, experts=E, topk=topk, **_S1V2_CFG)
        s1(a2, a2s, x.view(torch.uint8), w13.view(torch.uint8), x_scale, weights["w13_scale_flat"],
           sid, seid, nvalid, rows, S, blocks, stream)
        rep = unit // tile_m
        if rep > 1:  # stage2 walks tile_m-row blocks of the same sorted layout
            seid = seid.repeat_interleave(rep)
            blocks = blocks * rep
    else:
        # gemm1 all-padding subtile skip (FBS_PADSKIP) needs the fast sort's per-block counts; CK fallback -> off
        g1_ps = _fbs._parse_padskip(_fbs._G1_PADSKIP, tile_m // 16) if blk_cnt is not None else (tile_m // 16,)
        g1_ps = g1_ps if len(g1_ps) > 1 else ()
        s1 = _fbs.compile_stage1(model_dim=model_dim, inter_dim=inter, experts=E, topk=topk,
                                 tile_m=tile_m, tile_n=_FBS_TILE_N1, padskip=g1_ps)
        s1(a2, a2s, x.view(torch.uint8), w13.view(torch.uint8), x_scale, weights["w13_scale_flat"],
           sid, seid, nvalid, blk_cnt if g1_ps else nvalid, rows, S, blocks, stream)
    if _s2 is not None:
        _s2.stage2_fp8bs(moe_buf, a2, weights["w2"].view(torch.uint8), a2s, weights["w2_scale_flat"], sid, seid,
                         sw, nvalid, rows, S, blocks, stream, topk=topk, tile_m=tile_m, blk_cnt=blk_cnt)
        return moe_buf
    s2 = _fbs.compile_stage2(model_dim=model_dim, inter_dim=inter, experts=E, topk=topk,
                             tile_m=tile_m, tile_n=_FBS_TILE_N2)
    s2(moe_buf, a2, weights["w2"].view(torch.uint8), a2s, weights["w2_scale_flat"], sid, seid, sw,
       nvalid, rows, S, blocks, stream)
    return moe_buf
