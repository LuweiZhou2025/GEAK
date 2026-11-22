"""Code under test (editable): rank-local fp8 block-scale fused MoE for MiMo-V2.6-Flash-MOPD at EP8.

Baseline = what SGLang MoriEPMoE runs: aiter fused_moe(quant_type=per_128x128) on the MORI
fp8-dispatched receive buffer. aiter_fused_moe.py is a verbatim, editable copy of aiter/fused_moe.py.
"""
import torch
from aiter import ActivationType, QuantType
from aiter.ops.shuffle import shuffle_weight

# Never also import aiter.fused_moe: both register torch op aiter::fused_moe_ and the first one wins.
from aiter_fused_moe import fused_moe


def prepare_weights(w13, w2, w13_scale, w2_scale):
    """Untimed, once per weight set (= SGLang process_weights_after_loading).

    w13 [32, 4096, 4096] fp8 (gate rows then up rows), w2 [32, 4096, 2048] fp8,
    w13_scale [32, 32, 32] / w2_scale [32, 32, 16] fp32 dequant scales per 128x128 block.
    """
    return {
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
