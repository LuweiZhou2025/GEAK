"""Two-kernel MoE align&sort for the EP rank-local FlyDSL path: a drop-in for aiter moe_sorting's
CK multiphase ClearWorkspace + P0 + P1 + P23 chain (same buffers, same allocation order).

Output contract == CK moe_sorting (local-expert masking, local token count, skip empty experts):
  sorted_ids[i]   = (slot << 24) | token, experts in local-index order, tokens ascending,
                    each expert padded to a multiple of unit with (topk << 24) | T
  sorted_w[i]     = topk_weights[token, slot] (0.0 for padding)
  sorted_eids[m]  = local expert index of M block m
  num_valid[0:2]  = [padded total, T]          (T = num_local_tokens[0])
  moe_buf         = [M, model_dim]; rows [0, T) zeroed when zero_fill=True
Entries past num_valid[0] are left unwritten (consumers never read them). Bit-identical to CK for
real routing (distinct experts per token; CK keeps a single copy of a repeated expert, this keeps
one entry per slot like the fused_moe reference).

k_count : one block per 256 tokens -> per-block local-expert histogram (LDS atomics).
k_place : per block, expert totals + this block's prefix from the histograms, padded expert bases,
          wave-ballot ranks (deterministic token order), scatter; blocks also own the padding /
          expert-id fill of experts le = b, b + grid, ...  Extra blocks past the sort grid zero the
          moe_buf rows, overlapping the latency-bound placement work (CK P23 fuses its clear the
          same way).
"""
import hashlib
import os

import torch

_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>

#define TB 256
#define MAXE 256
#define NPART 8

__device__ __forceinline__ int build_lid(int mval, int E, int* s_lid, int* s_tmp) {
    const int tid = threadIdx.x, wave = tid >> 6, lane = tid & 63;
    const int m = (tid < E) ? (mval != 0) : 0;
    const unsigned long long b = __ballot(m);
    const int pre = __popcll(b & ((1ull << lane) - 1ull));
    if (lane == 0) s_tmp[wave] = __popcll(b);
    __syncthreads();
    int wpre = 0;
    for (int w = 0; w < wave; ++w) wpre += s_tmp[w];
    const int nl = s_tmp[0] + s_tmp[1] + s_tmp[2] + s_tmp[3];
    if (tid < E) s_lid[tid] = m ? (wpre + pre) : -1;
    __syncthreads();
    return nl;
}

template <int TOPK>
__global__ __launch_bounds__(256) void k_count(const int* __restrict__ ids, const int* __restrict__ mask,
                                               const int* __restrict__ nlt, int E, int* __restrict__ bcnt) {
    __shared__ int s_lid[MAXE];
    __shared__ int s_tmp[8];
    __shared__ int s_cnt[MAXE];
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int t = b * TB + tid;            // M % TB == 0 (supported()): row t < M exists (stale or not)
    // every load is issued up front (independent of T) so their latencies overlap
    const int mval = (tid < E) ? mask[tid] : 0;
    int e8[TOPK];
    {
        const int4* p = reinterpret_cast<const int4*>(ids + (size_t)t * TOPK);
#pragma unroll
        for (int v = 0; v < TOPK / 4; ++v) {
            const int4 x = p[v];
            e8[4 * v] = x.x; e8[4 * v + 1] = x.y; e8[4 * v + 2] = x.z; e8[4 * v + 3] = x.w;
        }
    }
    const int T = nlt[0];
    if (b * TB >= T) return;
    const int nl = build_lid(mval, E, s_lid, s_tmp);
    s_cnt[tid] = 0;
    __syncthreads();
    if (t < T) {
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            const int e = e8[k];
            const int le = ((unsigned)e < (unsigned)E) ? s_lid[e] : -1;
            if (le >= 0) atomicAdd(&s_cnt[le], 1);
        }
    }
    __syncthreads();
    if (tid < nl) bcnt[b * MAXE + tid] = s_cnt[tid];
}

template <int TOPK>
__global__ __launch_bounds__(256) void k_place(const int* __restrict__ ids, const float* __restrict__ wts,
                                               const int* __restrict__ mask, const int* __restrict__ nlt,
                                               int E, int unit, const int* __restrict__ bcnt,
                                               int* __restrict__ sorted_ids, float* __restrict__ sorted_w,
                                               int* __restrict__ sorted_eid, int* __restrict__ num_valid,
                                               int4* __restrict__ zbuf, int zrow16, int sort_grid,
                                               int* __restrict__ blk_cnt) {
    __shared__ int s_lid[MAXE];
    __shared__ int s_tmp[8];
    __shared__ int s_tot[MAXE];
    __shared__ int s_pre[MAXE];
    __shared__ int s_base[MAXE];
    __shared__ int s_ptot[MAXE];   // [le * g + r], g * nl <= 256
    __shared__ int s_ppre[MAXE];
    __shared__ int s_wcnt[4][MAXE];
    if ((int)blockIdx.x >= sort_grid) {
        // zero-fill blocks: rows [0, T) of the stage-2 accumulation buffer, overlapping the
        // latency-bound placement blocks (same idea as CK P23's fused moe_buf clear)
        const long total = (long)nlt[0] * zrow16;
        const long stride = (long)(gridDim.x - sort_grid) * TB;
        const int4 z = make_int4(0, 0, 0, 0);
        for (long c = (long)(blockIdx.x - sort_grid) * TB + threadIdx.x; c < total; c += stride) zbuf[c] = z;
        return;
    }
    const int b = blockIdx.x;
    const int tid = threadIdx.x, wave = tid >> 6, lane = tid & 63;
    const int t = b * TB + tid;
    const int mval = (tid < E) ? mask[tid] : 0;
    int e8[TOPK];
    float wk[TOPK];
    {
        const int4* p = reinterpret_cast<const int4*>(ids + (size_t)t * TOPK);
        const float4* q = reinterpret_cast<const float4*>(wts + (size_t)t * TOPK);
#pragma unroll
        for (int v = 0; v < TOPK / 4; ++v) {
            const int4 x = p[v];
            const float4 y = q[v];
            e8[4 * v] = x.x; e8[4 * v + 1] = x.y; e8[4 * v + 2] = x.z; e8[4 * v + 3] = x.w;
            wk[4 * v] = y.x; wk[4 * v + 1] = y.y; wk[4 * v + 2] = y.z; wk[4 * v + 3] = y.w;
        }
    }
    const int T = nlt[0];
    const int nb = (T + TB - 1) / TB;
    const int nl = build_lid(mval, E, s_lid, s_tmp);
    if (b >= nb && b >= nl) return;  // uniform per block

    // ---- expert totals and this block's prefix: G threads per expert, unrolled independent loads ----
    {
        int g = nl > 0 ? (TB / nl) : 1;
        g = g < 1 ? 1 : (g > NPART ? NPART : g);
        const int le = tid / g, r = tid - (tid / g) * g;
        if (le < nl) {
            int tot = 0, pre = 0;
            int bb = r;
            for (; bb + 3 * g < nb; bb += 4 * g) {
                const int c0 = bcnt[(bb) * MAXE + le];
                const int c1 = bcnt[(bb + g) * MAXE + le];
                const int c2 = bcnt[(bb + 2 * g) * MAXE + le];
                const int c3 = bcnt[(bb + 3 * g) * MAXE + le];
                tot += c0 + c1 + c2 + c3;
                pre += (bb < b ? c0 : 0) + (bb + g < b ? c1 : 0) + (bb + 2 * g < b ? c2 : 0) + (bb + 3 * g < b ? c3 : 0);
            }
            for (; bb < nb; bb += g) {
                const int c = bcnt[bb * MAXE + le];
                tot += c;
                pre += (bb < b) ? c : 0;
            }
            s_ptot[le * g + r] = tot;
            s_ppre[le * g + r] = pre;
        }
        __syncthreads();
        if (tid < nl) {
            int tt = 0, pp = 0;
            for (int q = 0; q < g; ++q) { tt += s_ptot[tid * g + q]; pp += s_ppre[tid * g + q]; }
            s_tot[tid] = tt;
            s_pre[tid] = pp;
        }
        __syncthreads();
    }
    // ---- padded expert sizes -> exclusive prefix (bases) ----
    const int my_pad = (tid < nl) ? ((s_tot[tid] + unit - 1) / unit) * unit : 0;
    int incl = my_pad;
#pragma unroll
    for (int off = 1; off < 64; off <<= 1) {
        const int v = __shfl_up(incl, off, 64);
        if (lane >= off) incl += v;
    }
    if (lane == 63) s_tmp[4 + wave] = incl;
    __syncthreads();
    int wpre = 0;
    for (int w = 0; w < wave; ++w) wpre += s_tmp[4 + w];
    const int total = s_tmp[4] + s_tmp[5] + s_tmp[6] + s_tmp[7];
    if (tid < nl) s_base[tid] = wpre + incl - my_pad;
    __syncthreads();
    if (b == 0 && tid == 0) {
        num_valid[0] = total;
        num_valid[1] = T;
    }
    // ---- padding entries + M-block expert ids for experts owned by this block ----
    // Stride by sort_grid, not gridDim.x: the zero-fill blocks (b >= sort_grid) returned above, so with
    // sort_grid < nl (M < 256 * nl rows) a gridDim.x stride would leave experts le >= sort_grid unpadded.
    const int pad_id = (TOPK << 24) | (T & 0xFFFFFF);
    for (int le = b; le < nl; le += sort_grid) {
        const int base = s_base[le], cnt = s_tot[le];
        const int pad_end = base + ((cnt + unit - 1) / unit) * unit;
        for (int i = base + cnt + tid; i < pad_end; i += TB) {
            sorted_ids[i] = pad_id;
            sorted_w[i] = 0.0f;
        }
        for (int m = base / unit + tid; m < pad_end / unit; m += TB) {
            sorted_eid[m] = le;
            // valid (non-padding) rows of M block m: padding is a suffix of the expert's last block
            if (blk_cnt != nullptr) blk_cnt[m] = min(unit, base + cnt - m * unit);
        }
    }
    if (b >= nb) return;
    // ---- token placement: deterministic ranks (token order) via wave ballots ----
    const bool valid = t < T;
    int lek[TOPK];
    int rk[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        const int e = e8[k];
        lek[k] = (valid && (unsigned)e < (unsigned)E) ? s_lid[e] : -1;
        rk[k] = 0;
    }
    const unsigned long long lt_mask = (1ull << lane) - 1ull;
    for (int le = 0; le < nl; ++le) {
        bool f = false;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) f |= (lek[k] == le);
        const unsigned long long B = __ballot(f);
        if (lane == 0) s_wcnt[wave][le] = __popcll(B);
        const int r = __popcll(B & lt_mask);
#pragma unroll
        for (int k = 0; k < TOPK; ++k) rk[k] = (lek[k] == le) ? r : rk[k];
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        const int le = lek[k];
        if (le >= 0) {
            int pos = s_base[le] + s_pre[le] + rk[k];
            for (int w = 0; w < wave; ++w) pos += s_wcnt[w][le];
            sorted_ids[pos] = (k << 24) | t;
            sorted_w[pos] = wk[k];
        }
    }
}

void moe_sort_fast(torch::Tensor topk_ids, torch::Tensor topk_w, torch::Tensor mask, torch::Tensor nlt,
                   int64_t unit, torch::Tensor bcnt, torch::Tensor sorted_ids, torch::Tensor sorted_w,
                   torch::Tensor sorted_eid, torch::Tensor num_valid, torch::Tensor zbuf, int64_t zfill_blocks,
                   torch::Tensor blk_cnt, int64_t want_cnt) {
    const int M = topk_ids.size(0), topk = topk_ids.size(1), E = mask.numel();
    TORCH_CHECK(topk == 8 && E <= MAXE, "moe_sort_fast: topk 8, <=256 experts");
    const int grid = (M + TB - 1) / TB;
    hipStream_t st = at::hip::getCurrentHIPStream();
    hipLaunchKernelGGL(k_count<8>, dim3(grid), dim3(TB), 0, st, topk_ids.data_ptr<int>(), mask.data_ptr<int>(),
                       nlt.data_ptr<int>(), E, bcnt.data_ptr<int>());
    const int zrow16 = zfill_blocks > 0 ? (int)(zbuf.size(-1) * zbuf.element_size() / 16) : 0;
    TORCH_CHECK(zfill_blocks == 0 || (zbuf.is_contiguous() && (zbuf.size(-1) * zbuf.element_size()) % 16 == 0),
                "moe_sort_fast: zero-fill buffer rows must be 16-B multiples");
    hipLaunchKernelGGL(k_place<8>, dim3(grid + (int)zfill_blocks), dim3(TB), 0, st, topk_ids.data_ptr<int>(),
                       topk_w.data_ptr<float>(), mask.data_ptr<int>(), nlt.data_ptr<int>(), E, (int)unit,
                       bcnt.data_ptr<int>(), sorted_ids.data_ptr<int>(), sorted_w.data_ptr<float>(),
                       sorted_eid.data_ptr<int>(), num_valid.data_ptr<int>(),
                       zfill_blocks > 0 ? reinterpret_cast<int4*>(zbuf.data_ptr()) : nullptr, zrow16, grid,
                       want_cnt ? blk_cnt.data_ptr<int>() : nullptr);
}
"""
_mod = None
_err = None


def _load():
    global _mod, _err
    if _mod is not None or _err is not None:
        return _mod
    try:
        from torch.utils.cpp_extension import load_inline
        tag = hashlib.sha1(_SRC.encode()).hexdigest()[:12]
        _mod = load_inline(
            name=f"moe_sort_fast_{tag}",
            cpp_sources="void moe_sort_fast(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, "
                        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, "
                        "torch::Tensor, int64_t);",
            cuda_sources=_SRC,
            functions=["moe_sort_fast"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as e:  # pragma: no cover - caller falls back to CK sorting
        _err = e
        _mod = None
    return _mod


def available():
    return _load() is not None


def moe_sorting_fast(topk_ids, topk_weights, num_experts, block_size, expert_mask, num_local_tokens,
                     model_dim=0, moebuf_dtype=torch.bfloat16, zero_fill=False, zfill_blocks=304,
                     blk_counts=False):
    """Drop-in for aiter_fused_moe.moe_sorting (same buffers, same allocation order):
    returns (sorted_ids, sorted_w, sorted_eids, num_valid, moe_buf [M, model_dim]).
    zero_fill=True: rows [0, num_local_tokens) of moe_buf are zeroed by extra blocks of the placement
    kernel (overlapping its latency-bound sort work, like CK P23's fused clear).
    blk_counts=True: also returns a 6th tensor, int32 [len(sorted_eids)], parallel to sorted_eids:
    blk_cnt[m] = number of valid (non-padding) rows of M block m (1..block_size; a block's padding is a
    suffix), written for m < num_valid[0] / block_size (later entries are unwritten)."""
    mod = _load()
    dev = topk_ids.device
    M, topk = topk_ids.shape
    max_pad = int(topk_ids.numel() + num_experts * block_size - topk)
    max_blk = int((max_pad + block_size - 1) // block_size)
    sorted_ids = torch.empty(max_pad, dtype=torch.int32, device=dev)
    sorted_w = torch.empty(max_pad, dtype=torch.float32, device=dev)
    sorted_eid = torch.empty(max_blk, dtype=torch.int32, device=dev)
    num_valid = torch.empty(2, dtype=torch.int32, device=dev)
    moe_buf = torch.empty((M, model_dim), dtype=moebuf_dtype, device=dev)
    nblk = (M + 255) // 256
    bcnt = torch.empty(nblk * 256, dtype=torch.int32, device=dev)
    zf = bool(zero_fill) and model_dim > 0
    blk_cnt = torch.empty(max_blk, dtype=torch.int32, device=dev) if blk_counts else num_valid
    mod.moe_sort_fast(topk_ids, topk_weights, expert_mask, num_local_tokens, int(block_size), bcnt,
                      sorted_ids, sorted_w, sorted_eid, num_valid, moe_buf if zf else num_valid,
                      int(zfill_blocks) if zf else 0, blk_cnt, 1 if blk_counts else 0)
    if blk_counts:
        return sorted_ids, sorted_w, sorted_eid, num_valid, moe_buf, blk_cnt
    return sorted_ids, sorted_w, sorted_eid, num_valid, moe_buf


def supported(topk_ids, topk_weights, expert_mask, num_local_tokens):
    return (
        topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.float32
        and topk_ids.dim() == 2 and topk_ids.shape[1] == 8 and topk_ids.is_contiguous()
        and topk_weights.is_contiguous() and tuple(topk_weights.shape) == tuple(topk_ids.shape)
        and expert_mask is not None and expert_mask.dtype == torch.int32 and expert_mask.numel() <= 256
        and expert_mask.is_contiguous()
        and num_local_tokens is not None and num_local_tokens.dtype == torch.int32
        and topk_ids.shape[0] < (1 << 24) and topk_ids.shape[0] % 256 == 0  # kernels read whole 256-row blocks
        and topk_ids.data_ptr() % 16 == 0 and topk_weights.data_ptr() % 16 == 0  # 16-B vector loads
    )
