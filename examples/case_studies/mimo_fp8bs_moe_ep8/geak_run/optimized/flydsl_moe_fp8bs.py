"""FlyDSL fp8 (e4m3fnuz) W8A8 block-scale 2-stage MoE for gfx942.

Stage1: A2[sorted_row, n] = fp8_quant_1x128( silu(X[tok] @ Wg^T) * (X[tok] @ Wu^T) )
        - native fp8 MFMA (16x16x32), tile_k = 128 = one block-scale K group per K tile:
          acc += mfma_partial(Ktile) * x_scale[tok, kb] * w_scale[nblk, kb]
        - per-row 1x128 requant fused in the epilogue; A2 is indexed by SORTED row (no
          [tokens*topk, inter] scatter) and its scale is stored as [inter/128, S] (kb-major).
Stage2: out[tok, :] += topk_w * (A2[sorted_row] @ W2[e]^T) with the same block-scale fold,
        bf16 global atomics into the [tokens, model_dim] output (no [M*topk, hidden] buffer).

Skeleton (preshuffled-B direct-to-register loads, X via XOR16-swizzled LDS ping-pong, CShuffle
atomic epilogue) is forked from aiter/ops/flydsl/kernels/moe_gemm_2stage.py.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith
from flydsl.expr import gpu, buffer_ops, vector, rocdl
from flydsl.expr import range_constexpr, const_expr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl._mlir.dialects import math as mmath
from flydsl.expr.typing import T
from flydsl.expr.arith import ArithValue

from aiter.ops.flydsl.kernels.mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    lds_store_16b_xor16,
    make_preshuffle_b_layout,
    load_b_pack_k32,
    tile_chunk_coord_i32,
    swizzle_xor16,
    crd2idx,
)
from aiter.ops.flydsl.kernels.mfma_epilogues import c_shuffle_epilog, default_epilog

BLK = 128
FP8_MAX = 240.0  # e4m3fnuz
import os as _os
_DBG_NOSCALE = _os.environ.get("FBS_DBG_NOSCALE", "0") == "1"
_DBG_ATOM = int(_os.environ.get("FBS_DBG_ATOM", "0"))
_TR = _os.environ.get("FBS_TR", "1") == "1"  # stage1: compute C^T (W as MFMA A operand)
_MF32 = _os.environ.get("FBS_MF32", "0") == "1"  # stage1: 32x32x16 fp8 MFMA (requires _TR)
_MF32_NODEFER = _os.environ.get("FBS_MF32_NODEFER", "1") == "1"
_SX_FIRST = _os.environ.get("FBS_SX_FIRST", "1") == "1"
_DEFER = int(_os.environ.get("FBS_DEFER", "1"))
_DBG_NOBAR = _os.environ.get("FBS_DBG_NOBAR", "0") == "1"
_DSRD_FIX = _os.environ.get("FBS_DSRD_FIX", "1") == "1"
_BPF = int(_os.environ.get("FBS_BPF", "2"))
_DBG_BK0 = _os.environ.get("FBS_DBG_BK0", "0") == "1"
_DBG_NOEPI = _os.environ.get("FBS_DBG_NOEPI", "0") == "1"
_SX_TAB = _os.environ.get("FBS_SX_TAB", "1") == "1"
_PK_SPLIT = _os.environ.get("FBS_PK_SPLIT", "1") == "1"
_VALID_TOK = _os.environ.get("FBS_VALID_TOK", "0") == "1"
_SLOAD = _os.environ.get("FBS_SLOAD", "1") == "1"
_SETPRIO = _os.environ.get("FBS_SETPRIO", "0") == "1"
_SCHED_MODE2 = int(_os.environ.get("FBS_SCHED2", "3"))
_SCHED_G1 = int(_os.environ.get("FBS_SCHED_G1", "8"))
_SCHED_G2 = int(_os.environ.get("FBS_SCHED_G2", "8"))
# ---- r3_d0 gemm1 knobs (closure-captured per compile, see compile_stage1) ----
# FBS_G1_PRIO: 1 (default) = whole gemm1 K loop at s_setprio(1) (prologue/epilogue at 0); 0 = off; 2 = per-tile
# toggle (the pre-r3 FBS_SETPRIO placement: s_setprio is a scheduling boundary, so it cut each tile's
# sched_group_barrier pattern off its MFMAs -> measured -5.4%/-4.8% gemm1). FBS_SETPRIO only drives the old stage-2.
_G1_PRIO = int(_os.environ.get("FBS_G1_PRIO", "1"))
# FBS_WPE>0: rocdl.waves_per_eu attribute on the gemm1 gpu.func (2 = cap at 256 VGPR, 2 waves/SIMD).
_WPE = int(_os.environ.get("FBS_WPE", "2"))
# FBS_RS: K-loop rescale form. 0 = v2f32 llvm.fma (v_pk_fma_f32, partly unpacked by LLVM in MFMA shadows);
# 1 (default) = scalar f32 fma per lane element (v_fmac_f32). gfx942: V_PK_*_F32 cannot execute while ANY wave's
# MFMA is in flight on the SIMD (r3_d0 micro-bench: mfma+1pk 30.1 cyc/slot vs mfma+2 v_fma 17.1 vs mfma 16.6; a
# pk-only wave co-resident with an MFMA wave makes ~0 progress). Measured: gemm1 -3.5%/-3.8%, a2/a2s bit-identical.
_RS = int(_os.environ.get("FBS_RS", "1"))
# FBS_EPI_MAXNUM=1 (default): epilogue row-max with arith.maxnumf (v_max_f32/v_max3_f32) instead of the NaN-
# propagating arith.maximumf (v_max + v_cmp_o + v_cndmask): -127 VALU/wave; inputs are finite -> bit-identical.
_EPI_MAXNUM = int(_os.environ.get("FBS_EPI_MAXNUM", "1"))
# FBS_PADSKIP (r3_d1 gemm1 pricing, integrated on the r3_d0 body): skip all-padding 16-row (token) MFMA subtiles
# in gemm1, reading the per-M-block valid-row counts emitted by moe_sort_fast (blk_counts=True). Descending
# variant list starting at m_repeat ("4,2,1"); "0" = off. Skipped subtiles keep zero accumulators, so the
# (unchanged) epilogue still writes every a2/a2s row of the block.
_G1_PADSKIP = _os.environ.get("FBS_PADSKIP", "4,2,1")


def _parse_padskip(spec, m_repeat):
    if not spec or spec.strip() in ("0", "off", ""):
        return (m_repeat,)
    vs = tuple(int(v) for v in spec.split(",") if v.strip())
    if not vs or vs[0] != m_repeat or any(b >= a for a, b in zip(vs, vs[1:])) or vs[-1] < 1:
        raise ValueError(f"FBS_PADSKIP={spec!r}: need a descending list starting at m_repeat={m_repeat}")
    return vs


def _variant_chain(cnt_i32, vs, body):
    """Nested uniform scf.if/else: valid rows > 16*vs[1] -> body(vs[0]); else recurse on vs[1:]."""
    if len(vs) == 1:
        body(vs[0])
        return
    cond = arith.cmpi(arith.CmpIPredicate.sgt, cnt_i32, fx.Int32(16 * vs[1]))
    op = scf.IfOp(arith.unwrap(cond), [], has_else=True)
    with ir.InsertionPoint(op.then_block):
        body(vs[0])
        scf.YieldOp([])
    with ir.InsertionPoint(op.else_block):
        _variant_chain(cnt_i32, vs[1:], body)
        scf.YieldOp([])


def _if_then(if_op):
    class _Ctx:
        def __enter__(self):
            self.ip = ir.InsertionPoint(if_op.then_block)
            self.ip.__enter__()
            return if_op.then_block

        def __exit__(self, *a):
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])
            self.ip.__exit__(*a)
            return False

    return _Ctx()


def _w(v):
    if isinstance(v, ArithValue):
        return v
    if isinstance(v, ir.Value):
        return ArithValue(v)
    return v


def _fma(a, b, c, pk_split=None):
    if pk_split is None:
        pk_split = _PK_SPLIT
    if not pk_split:
        return mmath.fma(arith.unwrap(a), arith.unwrap(b), arith.unwrap(c))
    from flydsl._mlir.dialects import vector as _mv
    a, b, c = arith.unwrap(a), arith.unwrap(b), arith.unwrap(c)
    v2 = T.vec(2, T.f32)
    nel = ir.VectorType(a.type).shape[0]
    outs = []
    for h in range(nel // 2):
        ah = _mv.extract_strided_slice(v2, a, [2 * h], [2], [1])
        bh = _mv.extract_strided_slice(v2, b, [2 * h], [2], [1])
        ch = _mv.extract_strided_slice(v2, c, [2 * h], [2], [1])
        outs.append(mmath.fma(ah, bh, ch))
    while len(outs) > 1:
        nxt = []
        for i in range(0, len(outs), 2):
            w = ir.VectorType(outs[i].type).shape[0]
            nxt.append(_mv.shuffle(outs[i], outs[i + 1], list(range(2 * w))))
        outs = nxt
    return outs[0]


def _fma_sc(a, s, c):
    """a * s + c with a, c f32xN vectors and s a scalar f32, as N scalar llvm.fma (v_fma_f32, VOP3)."""
    a, s, c = arith.unwrap(a), arith.unwrap(s), arith.unwrap(c)
    nel = ir.VectorType(a.type).shape[0]
    outs = []
    for i in range(nel):
        ai = vector.extract(a, static_position=[i], dynamic_position=[])
        ci = vector.extract(c, static_position=[i], dynamic_position=[])
        outs.append(mmath.fma(arith.unwrap(ai), s, arith.unwrap(ci)))
    return vector.from_elements(T.vec(nel, T.f32), outs)


_SCHED_MODE = int(_os.environ.get("FBS_SCHED", "3"))


def hot_loop_sched(*, n_dsrd, n_vmem, n_mfma, n_dswr, mode=None, groups=8):
    _m = _SCHED_MODE if mode is None else mode
    """sched_group_barrier pattern for one K tile: LDS reads first, VMEM spread over MFMAs."""
    if _m == 1:
        rocdl.sched_dsrd(n_dsrd)
        per = max(1, (n_mfma * 3 // 4) // max(1, n_vmem))
        used = 0
        for _ in range(n_vmem):
            rocdl.sched_vmem(1)
            rocdl.sched_mfma(per)
            used += per
        if n_mfma - used > 0:
            rocdl.sched_mfma(n_mfma - used)
        rocdl.sched_dswr(n_dswr)
    elif _m == 3:
        # interleave: groups of (dsrd, vmem, mfma)
        dpg = [n_dsrd // groups + (1 if i < n_dsrd % groups else 0) for i in range(groups)]
        vpg = [n_vmem // groups + (1 if i < n_vmem % groups else 0) for i in range(groups)]
        mpg = n_mfma // groups
        rocdl.sched_dsrd(dpg[0] + dpg[1])
        for g in range(groups):
            if g + 2 < groups and dpg[g + 2] > 0:
                rocdl.sched_dsrd(dpg[g + 2])
            if vpg[g] > 0:
                rocdl.sched_vmem(vpg[g])
            rocdl.sched_mfma(mpg)
        rocdl.sched_dswr(n_dswr)
    elif _m == 4:
        rocdl.sched_dsrd(n_dsrd)
        per = max(1, (n_mfma // 2) // max(1, n_vmem))
        used = 0
        for _ in range(n_vmem):
            rocdl.sched_vmem(1)
            rocdl.sched_mfma(per)
            used += per
        if n_mfma - used > 0:
            rocdl.sched_mfma(n_mfma - used)
        rocdl.sched_dswr(n_dswr)
    elif _m == 2:
        rocdl.sched_vmem(n_vmem)
        rocdl.sched_dsrd(n_dsrd)
        rocdl.sched_mfma(n_mfma)
        rocdl.sched_dswr(n_dswr)
    rocdl.sched_barrier(0)


# ----------------------------------------------------------------------------------------------
# Stage 1
# ----------------------------------------------------------------------------------------------
_G1_KNOBS = ("_DBG_NOSCALE", "_TR", "_MF32", "_MF32_NODEFER", "_SX_FIRST", "_DEFER", "_DBG_NOBAR", "_DSRD_FIX",
             "_BPF", "_DBG_BK0", "_DBG_NOEPI", "_SX_TAB", "_PK_SPLIT", "_VALID_TOK", "_SLOAD", "_G1_PRIO",
             "_SCHED_MODE", "_SCHED_G1", "_WPE", "_RS", "_EPI_MAXNUM")


def compile_stage1(*, model_dim: int, inter_dim: int, experts: int, topk: int,
                   tile_m: int, tile_n: int = 128, padskip: tuple = ()):
    """Stage-1 launcher. The module-level knobs are snapshotted into the compile key (and captured by the kernel
    closure), so an in-process A/B that flips module globals gets a distinct compiled kernel per knob set.
    padskip: descending active-subtile variants (e.g. (4, 2, 1)) -> the launcher takes an extra arg_blk_cnt
    (moe_sort_fast per-M-block valid-row counts) after num_valid_ids; () = full body only (arg_blk_cnt unused)."""
    g = globals()
    return _compile_stage1(model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk, tile_m=tile_m,
                           tile_n=tile_n, knobs=tuple((k, g[k]) for k in _G1_KNOBS), padskip=tuple(padskip))


@functools.lru_cache(maxsize=64)
def _compile_stage1(*, model_dim: int, inter_dim: int, experts: int, topk: int,
                    tile_m: int, tile_n: int = 128, knobs=(), padskip: tuple = ()):
    kn = dict(knobs)
    _DBG_NOSCALE = kn["_DBG_NOSCALE"]
    _TR = kn["_TR"]
    _MF32 = kn["_MF32"]
    _MF32_NODEFER = kn["_MF32_NODEFER"]
    _SX_FIRST = kn["_SX_FIRST"]
    _DEFER = kn["_DEFER"]
    _DBG_NOBAR = kn["_DBG_NOBAR"]
    _DSRD_FIX = kn["_DSRD_FIX"]
    _BPF = kn["_BPF"]
    _DBG_BK0 = kn["_DBG_BK0"]
    _DBG_NOEPI = kn["_DBG_NOEPI"]
    _SX_TAB = kn["_SX_TAB"]
    _PK_SPLIT = kn["_PK_SPLIT"]
    _VALID_TOK = kn["_VALID_TOK"]
    _SLOAD = kn["_SLOAD"]
    _G1_PRIO = kn["_G1_PRIO"]
    _SCHED_MODE = kn["_SCHED_MODE"]
    _SCHED_G1 = kn["_SCHED_G1"]
    _WPE = kn["_WPE"]
    _RS = kn["_RS"]
    _EPI_MAXNUM = kn["_EPI_MAXNUM"]
    tile_k = BLK
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    total_threads = 256
    elem_bytes = 1
    tile_k_bytes = tile_k
    assert tile_n % BLK == 0 and inter_dim % tile_n == 0 and model_dim % tile_k == 0
    num_kb = model_dim // BLK
    n_blk_total = (2 * inter_dim) // BLK  # weight-scale N blocks per expert
    bytes_per_thread_x = tile_m * tile_k // total_threads
    assert bytes_per_thread_x % 16 == 0
    num_x_loads = bytes_per_thread_x // 16
    lds_stride = tile_k
    num_waves = 4
    n_per_wave = tile_n // num_waves
    num_acc_n = n_per_wave // 16
    waves_per_group = BLK // n_per_wave
    assert BLK % n_per_wave == 0
    m_repeat = tile_m // 16
    k_unroll = tile_k_bytes // 64
    pad_variants = tuple(padskip) if padskip else (m_repeat,)
    if len(pad_variants) > 1 and (_MF32 or not _TR or _DBG_NOSCALE or _DBG_NOEPI or pad_variants[0] != m_repeat):
        raise ValueError("gemm1 padskip is built for the default _TR 16x16 path")

    # LDS: X ping-pong | x_scale [num_kb, tile_m] f32 | row-max [4, tile_m] f32
    lds_x_bytes = 2 * tile_m * lds_stride
    lds_x_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_x_off + lds_x_bytes
    _use_tab = (tile_n == BLK) and _SX_TAB
    lds_sx_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_sx_off + num_kb * tile_m * 4 * (2 if _use_tab else 1)
    if _use_tab:
        # row-max scratch aliases the X region behind the quantized staging tile (epilogue only)
        assert tile_m * tile_n + num_waves * tile_m * 4 <= lds_x_bytes
        lds_red_off = lds_x_off + tile_m * tile_n
    else:
        lds_red_off = allocator._align(allocator.ptr, 16)
        allocator.ptr = lds_red_off + num_waves * tile_m * 4
    # quantized output staging tile [tile_m, tile_n] bytes, aliases X ping-pong (after loop)
    assert tile_m * tile_n <= lds_x_bytes

    @flyc.kernel
    def moe_fp8bs_gemm1(
        arg_out: fx.Tensor,
        arg_out_scale: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        arg_blk_cnt: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_s_stride: fx.Int32,
    ):
        tokens_in = arith.index_cast(T.index, i32_tokens_in)
        s_stride = arith.index_cast(T.index, i32_s_stride)
        k_in = fx.Index(model_dim)
        x_elem = T.f8
        w_elem = T.f8
        vec16_x = T.vec(16, x_elem)

        def silu(x):
            t = x * (-1.4426950408889634)
            emu = _w(rocdl.exp2(T.f32, arith.unwrap(t)))
            den = emu + 1.0
            sig = _w(rocdl.rcp(T.f32, arith.unwrap(den)))
            return x * sig

        def _rmax(a, b):
            if const_expr(_EPI_MAXNUM):
                return _w(arith.maxnumf(arith.unwrap(a), arith.unwrap(b)))
            return _w(arith.maximumf(a, b))

        acc_init = arith.constant_vector(0.0, T.f32x4)

        c_n_total = arith.index(experts * (2 * inter_dim))
        b_layout = make_preshuffle_b_layout(arith, c_n=c_n_total, c_k=k_in, kpack_bytes=16, elem_bytes=1)
        layout_b = b_layout.layout_b
        layout_lds = fx.make_layout(fx.make_shape(tile_m, tile_k), fx.make_stride(lds_stride, 1))

        tx = gpu.thread_id("x")
        by = gpu.block_id("x")  # N tile
        bx = gpu.block_id("y")  # sorted M tile
        bx_m = bx * fx.Index(tile_m)

        nvalid_rsrc = buffer_ops.create_buffer_resource(arg_num_valid_ids, max_size=False, num_records_bytes=fx.Index(8))
        if const_expr(_SLOAD):
            _nv_ptr = buffer_ops.create_llvm_ptr(buffer_ops.extract_base_index(arg_num_valid_ids), address_space=4)
            num_valid_i32 = _w(llvm.LoadOp(T.i32, _nv_ptr, alignment=4, invariant=True).result)
        else:
            num_valid_i32 = buffer_ops.buffer_load(nvalid_rsrc, fx.Index(0), vec_width=1, dtype=T.i32)
        if const_expr(_VALID_TOK):
            # num_valid_ids[1] = number of valid tokens: padding rows (sentinel id == that count)
            # then read zeros (buffer OOB) and are skipped by the atomics.
            tokens_in = arith.index_cast(T.index, buffer_ops.buffer_load(nvalid_rsrc, fx.Index(1), vec_width=1, dtype=T.i32))
        bx_m_i32 = arith.index_cast(T.i32, bx_m)
        blk_valid = arith.cmpi(arith.CmpIPredicate.ult, bx_m_i32, num_valid_i32)
        k_blocks16 = arith.index(tile_k_bytes // 16)
        layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

        _if_blk = scf.IfOp(blk_valid)
        with _if_then(_if_blk):
            base_ptr = allocator.get_base()
            lds_x = SmemPtr(base_ptr, lds_x_off, T.f8, shape=(lds_x_bytes,)).get()
            lds_q = SmemPtr(base_ptr, lds_x_off, T.i8, shape=(lds_x_bytes,)).get()
            lds_sx = SmemPtr(base_ptr, lds_sx_off, T.f32, shape=(num_kb * tile_m * (2 if _use_tab else 1),)).get()
            lds_red = SmemPtr(base_ptr, lds_red_off, T.f32, shape=(num_waves * tile_m,)).get()

            x_nbytes_idx = tokens_in * k_in
            x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=False, num_records_bytes=x_nbytes_idx)
            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)
            out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=False)
            outs_rsrc = buffer_ops.create_buffer_resource(arg_out_scale, max_size=False)
            sx_rsrc = buffer_ops.create_buffer_resource(
                arg_scale_x, max_size=False, num_records_bytes=tokens_in * fx.Index(num_kb * 4))
            sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False)
            sorted_rsrc = buffer_ops.create_buffer_resource(arg_sorted_token_ids, max_size=False)
            expert_rsrc = buffer_ops.create_buffer_resource(arg_expert_ids, max_size=False)

            expert_i32 = buffer_ops.buffer_load(expert_rsrc, bx, vec_width=1, dtype=T.i32)
            expert_idx = arith.index_cast(T.index, expert_i32)
            expert_off_idx = expert_idx * arith.index(2 * inter_dim)
            cnt_i32 = None
            if const_expr(len(pad_variants) > 1):
                # valid rows of this M block (uniform: bx is the block id) -> s_load_dword
                _cnt_ptr = buffer_ops.create_llvm_ptr(
                    buffer_ops.extract_base_index(arg_blk_cnt) + bx * fx.Index(4), address_space=4)
                cnt_i32 = _w(llvm.LoadOp(T.i32, _cnt_ptr, alignment=4, invariant=True).result)

            c_k_div4 = k_in // arith.index(4)
            tile_k_dwords = tile_k // 4
            layout_x_tile_div4 = fx.make_layout((tile_m, tile_k_dwords), stride=(tile_k_dwords, 1))
            tx_i32_base = tx * fx.Index(4)
            mask24 = fx.Int32(0xFFFFFF)
            tokens_i32 = arith.index_cast(T.i32, tokens_in)

            x_row_base_div4 = []
            x_col_local_i32 = []
            x_row_local = []
            for i in range_constexpr(num_x_loads):
                row_local, col_local_i32 = tile_chunk_coord_i32(
                    arith, tx_i32_base=tx_i32_base, i=i, total_threads=total_threads,
                    layout_tile_div4=layout_x_tile_div4, chunk_i32=4)
                x_row_local.append(row_local)
                x_col_local_i32.append(col_local_i32)
                fused_i = buffer_ops.buffer_load(sorted_rsrc, bx_m + row_local, vec_width=1, dtype=T.i32)
                t_raw = fused_i & mask24
                t_valid_i32 = arith.cmpi(arith.CmpIPredicate.ult, t_raw, tokens_i32)
                t_idx = arith.index_cast(T.index, t_raw)
                t_safe = t_valid_i32.select(t_idx, fx.Index(0))
                x_row_base_div4.append(t_safe * c_k_div4)

            def load_x_tile(base_k):
                base_k_div4 = base_k // fx.Index(4)
                parts = []
                for i in range_constexpr(num_x_loads):
                    idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                    x_vec = buffer_copy_gmem16_dwordx4(
                        buffer_ops, vector, elem_type=x_elem, idx_i32=idx_i32, rsrc=x_rsrc,
                        vec_elems=16, elem_bytes=1)
                    parts.append(vector.bitcast(T.i32x4, x_vec))
                return parts

            # ---- x_scale rows of this tile -> LDS [kb, row] ----
            n_sx_chunks = tile_m * (num_kb // 4)
            assert n_sx_chunks % total_threads == 0 or n_sx_chunks < total_threads
            sx_iters = max(1, n_sx_chunks // total_threads)
            for it in range_constexpr(sx_iters):
                c_lin = tx + fx.Index(it * total_threads)
                r_loc = c_lin // fx.Index(num_kb // 4)
                kq = c_lin % fx.Index(num_kb // 4)
                fused_s = buffer_ops.buffer_load(sorted_rsrc, bx_m + r_loc, vec_width=1, dtype=T.i32)
                t_s = fused_s & mask24
                t_s_ok = arith.cmpi(arith.CmpIPredicate.ult, t_s, tokens_i32)
                t_s_idx = t_s_ok.select(arith.index_cast(T.index, t_s), fx.Index(0))
                sv = buffer_ops.buffer_load(sx_rsrc, t_s_idx * fx.Index(num_kb) + kq * fx.Index(4),
                                            vec_width=4, dtype=T.f32)
                if const_expr(_use_tab):
                    swg_base = (expert_idx * fx.Index(n_blk_total) + by * fx.Index(tile_n // BLK)) * fx.Index(num_kb)
                    swg4 = buffer_ops.buffer_load(sw_rsrc, swg_base + kq * fx.Index(4), vec_width=4, dtype=T.f32)
                    swu4 = buffer_ops.buffer_load(sw_rsrc, swg_base + fx.Index((inter_dim // BLK) * num_kb) + kq * fx.Index(4),
                                                  vec_width=4, dtype=T.f32)
                    tg = _w(sv) * _w(swg4)
                    tu = _w(sv) * _w(swu4)
                    for j in range_constexpr(4):
                        if const_expr(_MF32):
                            lidx = ((kq * fx.Index(4) + fx.Index(j)) * fx.Index(tile_m)
                                    + (r_loc % fx.Index(32)) * fx.Index(tile_m // 32) + r_loc // fx.Index(32))
                        elif const_expr(_TR):
                            # [kb][row % 16][row // 16]: one b128 per lane gives all m_repeat scales
                            lidx = ((kq * fx.Index(4) + fx.Index(j)) * fx.Index(tile_m)
                                    + (r_loc % fx.Index(16)) * fx.Index(m_repeat) + r_loc // fx.Index(16))
                        else:
                            lidx = (kq * fx.Index(4) + fx.Index(j)) * fx.Index(tile_m) + r_loc
                        eg = vector.extract(arith.unwrap(tg), static_position=[j], dynamic_position=[])
                        eu = vector.extract(arith.unwrap(tu), static_position=[j], dynamic_position=[])
                        vector.store(vector.from_elements(T.vec(1, T.f32), [eg]), lds_sx, [lidx])
                        vector.store(vector.from_elements(T.vec(1, T.f32), [eu]), lds_sx,
                                     [lidx + fx.Index(num_kb * tile_m)])
                else:
                    for j in range_constexpr(4):
                        e = vector.extract(sv, static_position=[j], dynamic_position=[])
                        lidx = (kq * fx.Index(4) + fx.Index(j)) * fx.Index(tile_m) + r_loc
                        vector.store(vector.from_elements(T.vec(1, T.f32), [e]), lds_sx, [lidx])

            coord_wl = fx.idx2crd(tx, layout_tx_wave_lane)
            wave_id = fx.get(coord_wl, 0)
            lane_id = fx.get(coord_wl, 1)
            coord_l16 = fx.idx2crd(lane_id, layout_lane16)
            lane_div_16 = fx.get(coord_l16, 0)
            lane_mod_16 = fx.get(coord_l16, 1)
            row_a_lds = lane_mod_16
            col_offset_base_bytes = lane_div_16 * arith.index(16)

            by_n = by * fx.Index(tile_n)
            n_tile_base = wave_id * fx.Index(n_per_wave)

            n_intra_gate, n_blk_gate, n_intra_up, n_blk_up = [], [], [], []
            inter_idx = arith.index(inter_dim)
            layout_n_blk_intra = fx.make_layout((experts * (2 * inter_dim) // 16, 16), stride=(16, 1))
            for ni in range_constexpr(num_acc_n):
                col_g = by_n + n_tile_base + arith.index(ni * 16) + lane_mod_16
                row_gate = expert_off_idx + col_g
                row_up = row_gate + inter_idx
                cg = fx.idx2crd(row_gate, layout_n_blk_intra)
                n_blk_gate.append(fx.get(cg, 0))
                n_intra_gate.append(fx.get(cg, 1))
                cu = fx.idx2crd(row_up, layout_n_blk_intra)
                n_blk_up.append(fx.get(cu, 0))
                n_intra_up.append(fx.get(cu, 1))

            # weight-scale base indices (uniform per wave): w_scale[e, nblk, kb]
            nblk_g = (by_n + n_tile_base) // fx.Index(BLK)
            sw_g_base = (expert_idx * fx.Index(n_blk_total) + nblk_g) * fx.Index(num_kb)
            sw_u_base = sw_g_base + fx.Index((inter_dim // BLK) * num_kb)

            def load_b_tile(base_k, blk_list, intra_list):
                b_tile = []
                for ku in range_constexpr(k_unroll):
                    packs0, packs1 = [], []
                    for ni in range_constexpr(num_acc_n):
                        packs0.append(load_b_pack_k32(
                            buffer_ops, arith, vector, arg_b=arg_w, b_rsrc=w_rsrc, layout_b=layout_b,
                            base_k=base_k, ki_step=ku * 2, n_blk=blk_list[ni],
                            n_intra=intra_list[ni], lane_div_16=lane_div_16, elem_type=w_elem,
                            kpack_bytes=16, elem_bytes=1))
                        packs1.append(load_b_pack_k32(
                            buffer_ops, arith, vector, arg_b=arg_w, b_rsrc=w_rsrc, layout_b=layout_b,
                            base_k=base_k, ki_step=ku * 2 + 1, n_blk=blk_list[ni],
                            n_intra=intra_list[ni], lane_div_16=lane_div_16, elem_type=w_elem,
                            kpack_bytes=16, elem_bytes=1))
                    b_tile.append((packs0, packs1))
                return b_tile

            def load_sw(kb):
                g = buffer_ops.buffer_load(sw_rsrc, sw_g_base + fx.Index(kb), vec_width=1, dtype=T.f32)
                u = buffer_ops.buffer_load(sw_rsrc, sw_u_base + fx.Index(kb), vec_width=1, dtype=T.f32)
                return g, u

            acc_gate = [acc_init] * (num_acc_n * m_repeat)
            acc_up = [acc_init] * (num_acc_n * m_repeat)

            def store_x_tile_to_lds(parts, lds_base):
                for i in range_constexpr(num_x_loads):
                    lds_store_16b_xor16(
                        arith, vector, lds_memref=lds_x, vec16_ty=vec16_x, layout_lds=layout_lds,
                        row_local=x_row_local[i], col_local_i32=x_col_local_i32[i], tx_c4=fx.Index(4),
                        k_blocks16=k_blocks16, lds_base=lds_base, vec_part_i32x4=parts[i], elem_bytes=1)

            def lds_load_packs_k64(curr_row_a_lds, col_base_bytes, lds_base):
                col_swz = swizzle_xor16(curr_row_a_lds, col_base_bytes, k_blocks16)
                idx_a16 = crd2idx((curr_row_a_lds, col_swz), layout_lds) + lds_base
                loaded = vector.load_op(vec16_x, lds_x, [idx_a16])
                a_i64x2 = vector.bitcast(T.i64x2, loaded)
                a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                return a0, a1

            mfma_fn = rocdl.mfma_f32_16x16x32_fp8_fp8
            lane_row4 = lane_div_16 * fx.Index(4)

            def compute_tile(acc_g_in, acc_u_in, b_g, b_u, lds_base, kb, swg, swu, m_eff=m_repeat):
                g_list = list(acc_g_in)
                u_list = list(acc_u_in)
                # issue all LDS reads of this K tile up front (A packs + x-scales)
                a_all = []
                sx_all = []
                if const_expr(_TR and _SX_FIRST):
                    tr_idx = fx.Index(kb * tile_m) + lane_mod_16 * fx.Index(m_repeat)
                    tr_g = vector.load_op(T.vec(m_repeat, T.f32), lds_sx, [tr_idx])
                    tr_u = vector.load_op(T.vec(m_repeat, T.f32), lds_sx, [tr_idx + fx.Index(num_kb * tile_m)])
                for mi in range_constexpr(m_eff):
                    curr_row = row_a_lds + arith.index(mi * 16)
                    a_packs = []
                    for ku in range_constexpr(k_unroll):
                        a_packs.append(lds_load_packs_k64(curr_row, col_offset_base_bytes + arith.index(ku * 64), lds_base))
                    a_all.append(a_packs)
                    sx_idx = fx.Index(kb * tile_m + mi * 16) + lane_row4
                    if const_expr(_TR):
                        pass
                    elif const_expr(_use_tab):
                        sx_all.append((vector.load_op(T.f32x4, lds_sx, [sx_idx]),
                                       vector.load_op(T.f32x4, lds_sx, [sx_idx + fx.Index(num_kb * tile_m)])))
                    else:
                        sx_all.append(vector.load_op(T.f32x4, lds_sx, [sx_idx]))
                if const_expr(_TR and not _SX_FIRST):
                    tr_idx = fx.Index(kb * tile_m) + lane_mod_16 * fx.Index(m_repeat)
                    tr_g = vector.load_op(T.vec(m_repeat, T.f32), lds_sx, [tr_idx])
                    tr_u = vector.load_op(T.vec(m_repeat, T.f32), lds_sx, [tr_idx + fx.Index(num_kb * tile_m)])
                sx_sc = []
                if const_expr(_TR):
                    for mi in range_constexpr(m_eff):
                        eg = vector.extract(tr_g, static_position=[mi], dynamic_position=[])
                        eu = vector.extract(tr_u, static_position=[mi], dynamic_position=[])
                        sx_sc.append((eg, eu))
                        sx_all.append((vector.broadcast(T.f32x4, eg), vector.broadcast(T.f32x4, eu)))
                pending = []

                def flush(pend):
                    for (idx_p, pg_p, pu_p, mi_p) in pend:
                        if const_expr(_DBG_NOSCALE):
                            g_list[idx_p] = pg_p
                            u_list[idx_p] = pu_p
                        elif const_expr(_use_tab and _TR and _RS == 1):
                            sgs, sus = sx_sc[mi_p]
                            g_list[idx_p] = _fma_sc(pg_p, sgs, g_list[idx_p])
                            u_list[idx_p] = _fma_sc(pu_p, sus, u_list[idx_p])
                        elif const_expr(_use_tab):
                            sg, su = sx_all[mi_p]
                            g_list[idx_p] = _fma(pg_p, sg, g_list[idx_p], pk_split=_PK_SPLIT)
                            u_list[idx_p] = _fma(pu_p, su, u_list[idx_p], pk_split=_PK_SPLIT)
                        else:
                            sg = _w(sx_all[mi_p]) * _w(swg)
                            su = _w(sx_all[mi_p]) * _w(swu)
                            g_list[idx_p] = _fma(pg_p, sg, g_list[idx_p], pk_split=_PK_SPLIT)
                            u_list[idx_p] = _fma(pu_p, su, u_list[idx_p], pk_split=_PK_SPLIT)

                for mi in range_constexpr(m_eff):
                    for ni in range_constexpr(num_acc_n):
                        idx = mi * num_acc_n + ni
                        pg = acc_init
                        pu = acc_init
                        if const_expr(_DBG_NOSCALE):
                            pg = g_list[idx]
                            pu = u_list[idx]
                        for ku in range_constexpr(k_unroll):
                            a0, a1 = a_all[mi][ku]
                            if const_expr(_TR):
                                pg = mfma_fn(T.f32x4, [b_g[ku][0][ni], a0, pg, 0, 0, 0])
                                pu = mfma_fn(T.f32x4, [b_u[ku][0][ni], a0, pu, 0, 0, 0])
                                pg = mfma_fn(T.f32x4, [b_g[ku][1][ni], a1, pg, 0, 0, 0])
                                pu = mfma_fn(T.f32x4, [b_u[ku][1][ni], a1, pu, 0, 0, 0])
                            else:
                                pg = mfma_fn(T.f32x4, [a0, b_g[ku][0][ni], pg, 0, 0, 0])
                                pu = mfma_fn(T.f32x4, [a0, b_u[ku][0][ni], pu, 0, 0, 0])
                                pg = mfma_fn(T.f32x4, [a1, b_g[ku][1][ni], pg, 0, 0, 0])
                                pu = mfma_fn(T.f32x4, [a1, b_u[ku][1][ni], pu, 0, 0, 0])
                        pending.append((idx, pg, pu, mi))
                        if len(pending) > _DEFER:
                            flush(pending[:1])
                            pending = pending[1:]
                flush(pending)
                return g_list, u_list

            if const_expr(_MF32):
                assert n_per_wave == 32 and tile_m % 32 == 0
                m_rep32 = tile_m // 32
                lane_mod_32 = lane_id % fx.Index(32)
                lane_div_32 = lane_id // fx.Index(32)
                acc_init16 = arith.constant_vector(0.0, T.vec(16, T.f32))
                col_g32 = by_n + n_tile_base + lane_mod_32
                rg = fx.idx2crd(expert_off_idx + col_g32, layout_n_blk_intra)
                ru = fx.idx2crd(expert_off_idx + col_g32 + inter_idx, layout_n_blk_intra)
                n_blk_gate = [fx.get(rg, 0)]
                n_intra_gate = [fx.get(rg, 1)]
                n_blk_up = [fx.get(ru, 0)]
                n_intra_up = [fx.get(ru, 1)]
                k1_lane = [lane_div_32, lane_div_32 + fx.Index(2)]

                def load_b_tile(base_k, blk_list, intra_list):
                    # [slab][j] -> (h0, h1) i64 halves of one 16B pack
                    out = []
                    for sl in range_constexpr(tile_k // 64):
                        row = []
                        for j in range_constexpr(2):
                            hs = []
                            for h in range_constexpr(2):
                                hs.append(load_b_pack_k32(
                                    buffer_ops, arith, vector, arg_b=arg_w, b_rsrc=w_rsrc, layout_b=layout_b,
                                    base_k=base_k, ki_step=sl * 2 + h, n_blk=blk_list[0],
                                    n_intra=intra_list[0], lane_div_16=k1_lane[j], elem_type=w_elem,
                                    kpack_bytes=16, elem_bytes=1))
                            row.append(hs)
                        out.append(row)
                    return out

                acc_gate = [acc_init16] * m_rep32
                acc_up = [acc_init16] * m_rep32
                from flydsl.expr.rocdl import _split_mfma_operands

                def mfma32(rt, ops):
                    a_, b_, c_, cbsz, abid, blgp = _split_mfma_operands(ops)
                    return rocdl.mfma_f32_32x32x16_fp8_fp8(rt, a_, b_, c_, cbsz, abid, blgp).result

                def compute_tile(acc_g_in, acc_u_in, b_g, b_u, lds_base, kb, swg, swu):
                    g_list = list(acc_g_in)
                    u_list = list(acc_u_in)
                    a_all = []
                    for mi in range_constexpr(m_rep32):
                        curr_row = lane_mod_32 + fx.Index(mi * 32)
                        a_mi = []
                        for sl in range_constexpr(tile_k // 64):
                            rw = []
                            for j in range_constexpr(2):
                                cb = (lane_div_32 + fx.Index(2 * j)) * fx.Index(16) + fx.Index(sl * 64)
                                rw.append(lds_load_packs_k64(curr_row, cb, lds_base))
                            a_mi.append(rw)
                        a_all.append(a_mi)
                    tr_idx = fx.Index(kb * tile_m) + lane_mod_32 * fx.Index(m_rep32)
                    tr_g = vector.load_op(T.vec(m_rep32, T.f32), lds_sx, [tr_idx])
                    tr_u = vector.load_op(T.vec(m_rep32, T.f32), lds_sx, [tr_idx + fx.Index(num_kb * tile_m)])
                    pending = []

                    def flush(pend):
                        for (mi_p, pg_p, pu_p) in pend:
                            eg = vector.extract(tr_g, static_position=[mi_p], dynamic_position=[])
                            eu = vector.extract(tr_u, static_position=[mi_p], dynamic_position=[])
                            if const_expr(_RS == 1):
                                g_list[mi_p] = _fma_sc(pg_p, eg, g_list[mi_p])
                                u_list[mi_p] = _fma_sc(pu_p, eu, u_list[mi_p])
                            else:
                                g_list[mi_p] = _fma(pg_p, vector.broadcast(T.vec(16, T.f32), eg), g_list[mi_p],
                                                    pk_split=_PK_SPLIT)
                                u_list[mi_p] = _fma(pu_p, vector.broadcast(T.vec(16, T.f32), eu), u_list[mi_p],
                                                    pk_split=_PK_SPLIT)

                    for mi in range_constexpr(m_rep32):
                        pg = acc_init16
                        pu = acc_init16
                        for sl in range_constexpr(tile_k // 64):
                            for j in range_constexpr(2):
                                for h in range_constexpr(2):
                                    xa = a_all[mi][sl][j][h]
                                    pg = mfma32(T.vec(16, T.f32), [b_g[sl][j][h], xa, pg, 0, 0, 0])
                                    pu = mfma32(T.vec(16, T.f32), [b_u[sl][j][h], xa, pu, 0, 0, 0])
                        if const_expr(_MF32_NODEFER):
                            flush([(mi, pg, pu)])
                        else:
                            flush(pending)
                            pending = [(mi, pg, pu)]
                    flush(pending)
                    return g_list, u_list

            lds_tile_elems = arith.index(tile_m * lds_stride)
            lds_pong = fx.Index(0)
            lds_ping = lds_tile_elems
            rocdl.sched_barrier(0)

            total_tiles = num_kb
            # prologue
            x_regs = load_x_tile(fx.Index(0))
            bq0 = []  # B prefetch queue (depth _BPF tiles); shared prologue of every padskip variant body
            for pf in range_constexpr(min(_BPF, total_tiles)):
                bq0.append((load_b_tile(fx.Index(pf * tile_k), n_blk_gate, n_intra_gate),
                            load_b_tile(fx.Index(pf * tile_k), n_blk_up, n_intra_up)))
            store_x_tile_to_lds(x_regs, lds_pong)
            gpu.barrier()
            acc_gate0, acc_up0 = acc_gate, acc_up

            def kbody(m_eff):
                # K loop + epilogue over the first m_eff 16-token subtiles of the M block (m_eff == m_repeat: the
                # unchanged full body). Skipped (all-padding) subtiles keep zero accumulators -> the epilogue still
                # writes zeros / 1e-10 scales for them, so every a2/a2s row of a valid block stays defined.
                ckw = {} if m_eff == m_repeat else {"m_eff": m_eff}
                acc_gate = list(acc_gate0)
                acc_up = list(acc_up0)
                bq = list(bq0)
                bufs = [lds_pong, lds_ping]
                if const_expr(_G1_PRIO == 1):
                    # whole K loop at priority 1 (s_setprio is a scheduling boundary: per-tile toggles (mode 2, the
                    # pre-r3 FBS_SETPRIO placement) cut each tile's sched_group_barrier region off its MFMAs)
                    rocdl.s_setprio(1)
                for kt in range_constexpr(total_tiles - 1):
                    cur = bufs[kt % 2]
                    nxt = bufs[(kt + 1) % 2]
                    k_next = fx.Index((kt + 1) * tile_k)
                    x_regs_n = load_x_tile(k_next)
                    kpf = kt + _BPF
                    if kpf < total_tiles:
                        k_b = fx.Index(0) if _DBG_BK0 else fx.Index(kpf * tile_k)
                        bq.append((load_b_tile(k_b, n_blk_gate, n_intra_gate),
                                   load_b_tile(k_b, n_blk_up, n_intra_up)))
                    b_g_cur, b_u_cur = bq.pop(0)
                    if const_expr(_G1_PRIO == 2):
                        rocdl.s_setprio(1)
                    acc_gate, acc_up = compute_tile(acc_gate, acc_up, b_g_cur, b_u_cur, cur, kt, None, None, **ckw)
                    if const_expr(_G1_PRIO == 2):
                        rocdl.s_setprio(0)
                    store_x_tile_to_lds(x_regs_n, nxt)
                    if const_expr(_SCHED_MODE > 0):
                        nv_b = 2 * k_unroll * num_acc_n if kpf < total_tiles else 0
                        if const_expr(_MF32):
                            n_dsrd_ = (tile_m // 32) * 4 + 2
                            n_mfma_ = (tile_m // 32) * 2 * 8
                        else:
                            n_dsrd_ = m_eff * k_unroll + 2 if (_TR and _DSRD_FIX) else m_eff * (k_unroll + (2 if _use_tab else 1))
                            n_mfma_ = m_eff * num_acc_n * 2 * k_unroll * 2
                        hot_loop_sched(n_dsrd=n_dsrd_, n_vmem=num_x_loads + nv_b,
                                       n_mfma=n_mfma_, n_dswr=num_x_loads,
                                       mode=_SCHED_MODE, groups=_SCHED_G1)
                    if const_expr(not _DBG_NOBAR):
                        gpu.barrier()
                b_g_cur, b_u_cur = bq.pop(0)
                acc_gate, acc_up = compute_tile(acc_gate, acc_up, b_g_cur, b_u_cur, bufs[(total_tiles - 1) % 2],
                                                total_tiles - 1, None, None, **ckw)
                if const_expr(_G1_PRIO == 1):
                    rocdl.s_setprio(0)

                # ---------------- epilogue: silu*up, 1x128 row quant, fp8 store ----------------
                if const_expr(_DBG_NOEPI):
                    tot = None
                    for a_ in list(acc_gate) + list(acc_up):
                        for ii in range_constexpr(4):
                            v_ = _w(vector.extract(arith.unwrap(a_), static_position=[ii], dynamic_position=[]))
                            tot = v_ if tot is None else tot + v_
                    bad = arith.cmpf(arith.CmpFPredicate.OEQ, arith.unwrap(tot), arith.unwrap(_w(arith.constant(12345.0, type=T.f32))))
                    _if_d = scf.IfOp(bad)
                    with _if_then(_if_d):
                        buffer_ops.buffer_store(tot, outs_rsrc, tx)
                    return
                c64_i32 = arith.constant(64, type=T.i32)
                red_base = wave_id * fx.Index(tile_m)
                if const_expr(_MF32):
                    grp = wave_id // fx.Index(waves_per_group)
                    ys32 = []
                    for mi in range_constexpr(tile_m // 32):
                        yv = []
                        for v in range_constexpr(16):
                            vg = vector.extract(arith.unwrap(acc_gate[mi]), static_position=[v], dynamic_position=[])
                            vu = vector.extract(arith.unwrap(acc_up[mi]), static_position=[v], dynamic_position=[])
                            yv.append(silu(_w(vg)) * _w(vu))
                        ys32.append(yv)
                        m = None
                        for v in range_constexpr(16):
                            av = _w(mmath.absf(arith.unwrap(yv[v])))
                            m = av if m is None else _rmax(m, av)
                        peer = m.shuffle_xor(arith.constant(32, type=T.i32), c64_i32)
                        m = _rmax(m, _w(peer))
                        ridx = red_base + fx.Index(mi * 32) + lane_mod_32
                        vector.store(vector.from_elements(T.vec(1, T.f32), [m]), lds_red, [ridx])
                    gpu.barrier()
                    for mi in range_constexpr(tile_m // 32):
                        rloc = fx.Index(mi * 32) + lane_mod_32
                        m = None
                        for wv in range_constexpr(waves_per_group):
                            widx = (grp * fx.Index(waves_per_group) + fx.Index(wv)) * fx.Index(tile_m) + rloc
                            v_ = _w(vector.extract(vector.load_op(T.vec(1, T.f32), lds_red, [widx]),
                                                   static_position=[0], dynamic_position=[]))
                            m = v_ if m is None else _rmax(m, v_)
                        m = _rmax(m, _w(arith.constant(1e-10, type=T.f32)))
                        inv = _w(rocdl.rcp(T.f32, m * (1.0 / FP8_MAX)))
                        yv = ys32[mi]
                        for jj in range_constexpr(4):
                            p0 = rocdl.cvt_pk_fp8_f32(T.i32, _w(yv[4 * jj]) * inv, _w(yv[4 * jj + 1]) * inv,
                                                      arith.constant(0, type=T.i32), False)
                            p1 = rocdl.cvt_pk_fp8_f32(T.i32, _w(yv[4 * jj + 2]) * inv, _w(yv[4 * jj + 3]) * inv, p0, True)
                            col_l = n_tile_base + fx.Index(8 * jj) + lane_div_32 * fx.Index(4)
                            qidx = rloc * fx.Index(tile_n) + col_l
                            vector.store(vector.bitcast(T.vec(4, T.i8), vector.from_elements(T.vec(1, T.i32), [p1])),
                                         lds_q, [qidx])
                ys = []
                for mi in range_constexpr(0 if _MF32 else m_repeat):
                    for ni in range_constexpr(num_acc_n):
                        idx = mi * num_acc_n + ni
                        yv = []
                        for ii in range_constexpr(4):
                            vg = vector.extract(arith.unwrap(acc_gate[idx]), static_position=[ii], dynamic_position=[])
                            vu = vector.extract(arith.unwrap(acc_up[idx]), static_position=[ii], dynamic_position=[])
                            yv.append(silu(_w(vg)) * _w(vu))
                        ys.append(yv)

                c64_i32 = arith.constant(64, type=T.i32)
                red_base = wave_id * fx.Index(tile_m)
                if const_expr(_TR and not _MF32):
                    # lane holds row mi*16 + lane_mod_16, cols ni*16 + lane_div_16*4 + ii
                    for mi in range_constexpr(m_repeat):
                        m = None
                        for ni in range_constexpr(num_acc_n):
                            for ii in range_constexpr(4):
                                av = _w(mmath.absf(arith.unwrap(ys[mi * num_acc_n + ni][ii])))
                                m = av if m is None else _rmax(m, av)
                        for si in (16, 32):
                            peer = m.shuffle_xor(arith.constant(si, type=T.i32), c64_i32)
                            m = _rmax(m, _w(peer))
                        ridx = red_base + fx.Index(mi * 16) + lane_mod_16
                        vector.store(vector.from_elements(T.vec(1, T.f32), [m]), lds_red, [ridx])
                    gpu.barrier()
                    grp = wave_id // fx.Index(waves_per_group)
                    for mi in range_constexpr(m_repeat):
                        rloc = fx.Index(mi * 16) + lane_mod_16
                        m = None
                        for wv in range_constexpr(waves_per_group):
                            widx = (grp * fx.Index(waves_per_group) + fx.Index(wv)) * fx.Index(tile_m) + rloc
                            v = _w(vector.extract(vector.load_op(T.vec(1, T.f32), lds_red, [widx]),
                                                  static_position=[0], dynamic_position=[]))
                            m = v if m is None else _rmax(m, v)
                        m = _rmax(m, _w(arith.constant(1e-10, type=T.f32)))
                        inv = _w(rocdl.rcp(T.f32, m * (1.0 / FP8_MAX)))
                        for ni in range_constexpr(num_acc_n):
                            yv = ys[mi * num_acc_n + ni]
                            p0 = rocdl.cvt_pk_fp8_f32(T.i32, _w(yv[0]) * inv, _w(yv[1]) * inv,
                                                      arith.constant(0, type=T.i32), False)
                            p1 = rocdl.cvt_pk_fp8_f32(T.i32, _w(yv[2]) * inv, _w(yv[3]) * inv, p0, True)
                            col_l = n_tile_base + fx.Index(ni * 16) + lane_div_16 * fx.Index(4)
                            qidx = rloc * fx.Index(tile_n) + col_l
                            vector.store(vector.bitcast(T.vec(4, T.i8), vector.from_elements(T.vec(1, T.i32), [p1])),
                                         lds_q, [qidx])
                for mi in range_constexpr(0 if _TR else m_repeat):
                    for ii in range_constexpr(4):
                        m = None
                        for ni in range_constexpr(num_acc_n):
                            av = _w(mmath.absf(arith.unwrap(ys[mi * num_acc_n + ni][ii])))
                            m = av if m is None else _w(arith.maximumf(m, av))
                        for si in range_constexpr(4):
                            peer = m.shuffle_xor(arith.constant(1 << si, type=T.i32), c64_i32)
                            m = _w(arith.maximumf(m, _w(peer)))
                        ridx = red_base + fx.Index(mi * 16 + ii) + lane_row4
                        vector.store(vector.from_elements(T.vec(1, T.f32), [m]), lds_red, [ridx])
                if const_expr(not _TR):
                    gpu.barrier()  # row maxima visible; all waves done reading X LDS

                grp = wave_id // fx.Index(waves_per_group)
                for mi in range_constexpr(0 if _TR else m_repeat):
                    for ii in range_constexpr(4):
                        rloc = fx.Index(mi * 16 + ii) + lane_row4
                        m = None
                        for wv in range_constexpr(waves_per_group):
                            widx = (grp * fx.Index(waves_per_group) + fx.Index(wv)) * fx.Index(tile_m) + rloc
                            v = _w(vector.extract(vector.load_op(T.vec(1, T.f32), lds_red, [widx]),
                                                  static_position=[0], dynamic_position=[]))
                            m = v if m is None else _w(arith.maximumf(m, v))
                        m = _w(arith.maximumf(m, _w(arith.constant(1e-10, type=T.f32))))
                        inv = _w(rocdl.rcp(T.f32, m * (1.0 / FP8_MAX)))
                        for ni in range_constexpr(num_acc_n):
                            q = _w(ys[mi * num_acc_n + ni][ii]) * inv
                            packed = rocdl.cvt_pk_fp8_f32(T.i32, q, q, arith.constant(0, type=T.i32), 0)
                            b8 = arith.TruncIOp(T.i8, packed).result
                            col_l = n_tile_base + fx.Index(ni * 16) + lane_mod_16
                            qidx = rloc * fx.Index(tile_n) + col_l
                            vector.store(vector.from_elements(T.vec(1, T.i8), [b8]), lds_q, [qidx])
                        # scale: one writer per (row, group): wave == group-leader, lane_mod_16 == 0
                # scale write by threads tx < tile_m * groups
                n_groups = tile_n // BLK
                gpu.barrier()
                for it in range_constexpr((tile_m * n_groups + total_threads - 1) // total_threads):
                    lin = tx + fx.Index(it * total_threads)
                    ok = arith.cmpi(arith.CmpIPredicate.ult, arith.index_cast(T.i32, lin), fx.Int32(tile_m * n_groups))
                    _if_s = scf.IfOp(ok)
                    with _if_then(_if_s):
                        g = lin // fx.Index(tile_m)
                        rl = lin % fx.Index(tile_m)
                        m = None
                        for wv in range_constexpr(waves_per_group):
                            widx = (g * fx.Index(waves_per_group) + fx.Index(wv)) * fx.Index(tile_m) + rl
                            v = _w(vector.extract(vector.load_op(T.vec(1, T.f32), lds_red, [widx]),
                                                  static_position=[0], dynamic_position=[]))
                            m = v if m is None else _rmax(m, v)
                        m = _rmax(m, _w(arith.constant(1e-10, type=T.f32)))
                        s = m * (1.0 / FP8_MAX)
                        kb_out = by * fx.Index(n_groups) + g
                        buffer_ops.buffer_store(s, outs_rsrc, kb_out * s_stride + bx_m + rl)
                # copy quantized tile [tile_m, tile_n] bytes to A2[sorted_row, by_n:]
                chunks = tile_m * tile_n // 16
                for it in range_constexpr((chunks + total_threads - 1) // total_threads):
                    lin = tx + fx.Index(it * total_threads)
                    rl = lin // fx.Index(tile_n // 16)
                    cc = lin % fx.Index(tile_n // 16)
                    v = vector.load_op(T.vec(16, T.i8), lds_q, [rl * fx.Index(tile_n) + cc * fx.Index(16)])
                    vi = vector.bitcast(T.i32x4, v)
                    dw = ((bx_m + rl) * fx.Index(inter_dim) + by_n + cc * fx.Index(16)) // fx.Index(4)
                    buffer_ops.buffer_store(vi, out_rsrc, dw)

            if const_expr(len(pad_variants) > 1):
                _variant_chain(cnt_i32, pad_variants, kbody)
            else:
                kbody(m_repeat)

    @flyc.jit
    def launch_gemm1(
        arg_out: fx.Tensor,
        arg_out_scale: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        arg_blk_cnt: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_s_stride: fx.Int32,
        i32_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        gy = arith.index_cast(T.index, i32_blocks)
        moe_fp8bs_gemm1(
            arg_out, arg_out_scale, arg_x, arg_w, arg_scale_x, arg_scale_w,
            arg_sorted_token_ids, arg_expert_ids, arg_num_valid_ids, arg_blk_cnt, i32_tokens_in, i32_s_stride,
        ).launch(grid=(inter_dim // tile_n, gy, 1), block=(256, 1, 1), stream=stream)
        if const_expr(_WPE > 0):
            for op in ctx.gpu_module_body.operations:
                if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, int(_WPE))

    return launch_gemm1


# ----------------------------------------------------------------------------------------------
# Stage 2
# ----------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=64)
def compile_stage2(*, model_dim: int, inter_dim: int, experts: int, topk: int,
                   tile_m: int, tile_n: int = 128):
    tile_k = BLK
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    total_threads = 256
    tile_k_bytes = tile_k
    num_kb = inter_dim // BLK
    n_blk_total = model_dim // BLK
    bytes_per_thread_x = tile_m * tile_k // total_threads
    assert bytes_per_thread_x % 16 == 0
    num_x_loads = bytes_per_thread_x // 16
    lds_stride = tile_k
    num_waves = 4
    n_per_wave = tile_n // num_waves
    num_acc_n = n_per_wave // 16
    assert BLK % n_per_wave == 0
    m_repeat = tile_m // 16
    k_unroll = tile_k_bytes // 64
    e_vec = 2

    lds_x_bytes = 2 * tile_m * lds_stride
    lds_out_bytes = 2 * tile_m * tile_n
    lds_bytes = max(lds_x_bytes, lds_out_bytes)
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + lds_bytes

    @flyc.kernel
    def moe_fp8bs_gemm2(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_s_stride: fx.Int32,
    ):
        tokens_in = arith.index_cast(T.index, i32_tokens_in)
        s_stride = arith.index_cast(T.index, i32_s_stride)
        k_in = fx.Index(inter_dim)
        x_elem = T.f8
        w_elem = T.f8
        vec16_x = T.vec(16, x_elem)
        acc_init = arith.constant_vector(0.0, T.f32x4)

        c_n_total = arith.index(experts * model_dim)
        b_layout = make_preshuffle_b_layout(arith, c_n=c_n_total, c_k=k_in, kpack_bytes=16, elem_bytes=1)
        layout_b = b_layout.layout_b
        layout_lds = fx.make_layout(fx.make_shape(tile_m, tile_k), fx.make_stride(lds_stride, 1))

        tx = gpu.thread_id("x")
        by = gpu.block_id("x")
        bx = gpu.block_id("y")
        bx_m = bx * fx.Index(tile_m)
        k_blocks16 = arith.index(tile_k_bytes // 16)
        layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

        nvalid_rsrc = buffer_ops.create_buffer_resource(arg_num_valid_ids, max_size=False, num_records_bytes=fx.Index(8))
        if const_expr(_SLOAD):
            _nv_ptr = buffer_ops.create_llvm_ptr(buffer_ops.extract_base_index(arg_num_valid_ids), address_space=4)
            num_valid_i32 = _w(llvm.LoadOp(T.i32, _nv_ptr, alignment=4, invariant=True).result)
        else:
            num_valid_i32 = buffer_ops.buffer_load(nvalid_rsrc, fx.Index(0), vec_width=1, dtype=T.i32)
        if const_expr(_VALID_TOK):
            # num_valid_ids[1] = number of valid tokens: padding rows (sentinel id == that count)
            # then read zeros (buffer OOB) and are skipped by the atomics.
            tokens_in = arith.index_cast(T.index, buffer_ops.buffer_load(nvalid_rsrc, fx.Index(1), vec_width=1, dtype=T.i32))
        bx_m_i32 = arith.index_cast(T.i32, bx_m)
        blk_valid = arith.cmpi(arith.CmpIPredicate.ult, bx_m_i32, num_valid_i32)

        _if_blk = scf.IfOp(blk_valid)
        with _if_then(_if_blk):
            base_ptr = allocator.get_base()
            lds_x = SmemPtr(base_ptr, lds_off, T.f8, shape=(lds_bytes,)).get()
            lds_out = SmemPtr(base_ptr, lds_off, T.bf16, shape=(tile_m * tile_n,)).get()

            x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=False)
            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)
            sx_rsrc = buffer_ops.create_buffer_resource(arg_scale_x, max_size=False)
            sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False)
            sorted_rsrc = buffer_ops.create_buffer_resource(arg_sorted_token_ids, max_size=False)
            sorted_w_rsrc = buffer_ops.create_buffer_resource(arg_sorted_weights, max_size=False)
            expert_rsrc = buffer_ops.create_buffer_resource(arg_expert_ids, max_size=False)

            expert_i32 = buffer_ops.buffer_load(expert_rsrc, bx, vec_width=1, dtype=T.i32)
            expert_idx = arith.index_cast(T.index, expert_i32)
            expert_off_idx = expert_idx * fx.Index(model_dim)

            tile_k_dwords = tile_k // 4
            layout_x_tile_div4 = fx.make_layout((tile_m, tile_k_dwords), stride=(tile_k_dwords, 1))
            tx_i32_base = tx * fx.Index(4)
            mask24_i32 = fx.Int32(0xFFFFFF)
            tokens_i32 = arith.index_cast(T.i32, tokens_in)
            topk_i32 = fx.Int32(topk)
            c_k_div4 = fx.Index(inter_dim // 4)

            x_row_base_div4 = []
            x_col_local_i32 = []
            x_row_local = []
            for i in range_constexpr(num_x_loads):
                row_local, col_local_i32 = tile_chunk_coord_i32(
                    arith, tx_i32_base=tx_i32_base, i=i, total_threads=total_threads,
                    layout_tile_div4=layout_x_tile_div4, chunk_i32=4)
                x_row_local.append(row_local)
                x_col_local_i32.append(col_local_i32)
                x_row_base_div4.append((bx_m + row_local) * c_k_div4)

            def load_x_tile(base_k):
                base_k_div4 = base_k // fx.Index(4)
                parts = []
                for i in range_constexpr(num_x_loads):
                    idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                    x_vec = buffer_copy_gmem16_dwordx4(
                        buffer_ops, vector, elem_type=x_elem, idx_i32=idx_i32, rsrc=x_rsrc,
                        vec_elems=16, elem_bytes=1)
                    parts.append(vector.bitcast(T.i32x4, x_vec))
                return parts

            coord_wl = fx.idx2crd(tx, layout_tx_wave_lane)
            wave_id = fx.get(coord_wl, 0)
            lane_id = fx.get(coord_wl, 1)
            coord_l16 = fx.idx2crd(lane_id, layout_lane16)
            lane_div_16 = fx.get(coord_l16, 0)
            lane_mod_16 = fx.get(coord_l16, 1)
            row_a_lds = lane_mod_16
            col_offset_base_bytes = lane_div_16 * arith.index(16)
            lane_row4 = lane_div_16 * fx.Index(4)

            by_n = by * fx.Index(tile_n)
            n_tile_base = wave_id * fx.Index(n_per_wave)

            n_intra_list, n_blk_list = [], []
            layout_n_blk_intra = fx.make_layout((experts * model_dim // 16, 16), stride=(16, 1))
            for ni in range_constexpr(num_acc_n):
                col_g = by_n + n_tile_base + arith.index(ni * 16) + lane_mod_16
                cw = fx.idx2crd(expert_off_idx + col_g, layout_n_blk_intra)
                n_blk_list.append(fx.get(cw, 0))
                n_intra_list.append(fx.get(cw, 1))

            nblk = (by_n + n_tile_base) // fx.Index(BLK)
            sw_base = (expert_idx * fx.Index(n_blk_total) + nblk) * fx.Index(num_kb)

            def load_b_tile(base_k):
                b_tile = []
                for ku in range_constexpr(k_unroll):
                    packs0, packs1 = [], []
                    for ni in range_constexpr(num_acc_n):
                        packs0.append(load_b_pack_k32(
                            buffer_ops, arith, vector, arg_b=arg_w, b_rsrc=w_rsrc, layout_b=layout_b,
                            base_k=base_k, ki_step=ku * 2, n_blk=n_blk_list[ni],
                            n_intra=n_intra_list[ni], lane_div_16=lane_div_16, elem_type=w_elem,
                            kpack_bytes=16, elem_bytes=1))
                        packs1.append(load_b_pack_k32(
                            buffer_ops, arith, vector, arg_b=arg_w, b_rsrc=w_rsrc, layout_b=layout_b,
                            base_k=base_k, ki_step=ku * 2 + 1, n_blk=n_blk_list[ni],
                            n_intra=n_intra_list[ni], lane_div_16=lane_div_16, elem_type=w_elem,
                            kpack_bytes=16, elem_bytes=1))
                    b_tile.append((packs0, packs1))
                return b_tile

            def load_scales(kb):
                sw = buffer_ops.buffer_load(sw_rsrc, sw_base + fx.Index(kb), vec_width=1, dtype=T.f32)
                sxs = []
                for mi in range_constexpr(m_repeat):
                    sidx = fx.Index(kb) * s_stride + bx_m + fx.Index(mi * 16) + lane_row4
                    sxs.append(buffer_ops.buffer_load(sx_rsrc, sidx, vec_width=4, dtype=T.f32))
                return sw, sxs

            def store_x_tile_to_lds(parts, lds_base):
                for i in range_constexpr(num_x_loads):
                    lds_store_16b_xor16(
                        arith, vector, lds_memref=lds_x, vec16_ty=vec16_x, layout_lds=layout_lds,
                        row_local=x_row_local[i], col_local_i32=x_col_local_i32[i], tx_c4=fx.Index(4),
                        k_blocks16=k_blocks16, lds_base=lds_base, vec_part_i32x4=parts[i], elem_bytes=1)

            def lds_load_packs_k64(curr_row_a_lds, col_base_bytes, lds_base):
                col_swz = swizzle_xor16(curr_row_a_lds, col_base_bytes, k_blocks16)
                idx_a16 = crd2idx((curr_row_a_lds, col_swz), layout_lds) + lds_base
                loaded = vector.load_op(vec16_x, lds_x, [idx_a16])
                a_i64x2 = vector.bitcast(T.i64x2, loaded)
                a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                return a0, a1

            mfma_fn = rocdl.mfma_f32_16x16x32_fp8_fp8

            def compute_tile(acc_in, b_t, lds_base, scales):
                sw, sxs = scales
                acc_list = list(acc_in)
                for mi in range_constexpr(m_repeat):
                    curr_row = row_a_lds + arith.index(mi * 16)
                    a_packs = []
                    for ku in range_constexpr(k_unroll):
                        a_packs.append(lds_load_packs_k64(curr_row, col_offset_base_bytes + arith.index(ku * 64), lds_base))
                    s4 = _w(sxs[mi]) * _w(sw)
                    for ni in range_constexpr(num_acc_n):
                        p = acc_init
                        for ku in range_constexpr(k_unroll):
                            a0, a1 = a_packs[ku]
                            p = mfma_fn(T.f32x4, [a0, b_t[ku][0][ni], p, 0, 0, 0])
                            p = mfma_fn(T.f32x4, [a1, b_t[ku][1][ni], p, 0, 0, 0])
                        idx = mi * num_acc_n + ni
                        acc_list[idx] = _fma(p, s4, acc_list[idx])
                return acc_list

            lds_pong = fx.Index(0)
            lds_ping = arith.index(tile_m * lds_stride)
            rocdl.sched_barrier(0)
            total_tiles = num_kb
            acc = [acc_init] * (num_acc_n * m_repeat)
            x_regs = load_x_tile(fx.Index(0))
            b_cur = load_b_tile(fx.Index(0))
            sc_cur = load_scales(0)
            store_x_tile_to_lds(x_regs, lds_pong)
            gpu.barrier()
            bufs = [lds_pong, lds_ping]
            for kt in range_constexpr(total_tiles - 1):
                cur = bufs[kt % 2]
                nxt = bufs[(kt + 1) % 2]
                k_next = fx.Index((kt + 1) * tile_k)
                x_regs_n = load_x_tile(k_next)
                b_n = load_b_tile(k_next)
                sc_n = load_scales(kt + 1)
                if const_expr(_SETPRIO):
                    rocdl.s_setprio(1)
                acc = compute_tile(acc, b_cur, cur, sc_cur)
                if const_expr(_SETPRIO):
                    rocdl.s_setprio(0)
                store_x_tile_to_lds(x_regs_n, nxt)
                if const_expr(_SCHED_MODE > 0):
                    hot_loop_sched(n_dsrd=m_repeat * k_unroll, n_vmem=num_x_loads + k_unroll * num_acc_n + 1 + m_repeat,
                                   n_mfma=m_repeat * num_acc_n * k_unroll * 2, n_dswr=num_x_loads,
                                   mode=_SCHED_MODE2, groups=_SCHED_G2)
                gpu.barrier()
                b_cur = b_n
                sc_cur = sc_n
            acc = compute_tile(acc, b_cur, bufs[(total_tiles - 1) % 2], sc_cur)

            # ---------------- epilogue: CShuffle + bf16 global atomics ----------------
            model_i32 = fx.Int32(model_dim)
            c2_i32 = fx.Int32(2)
            mask_even_i32 = fx.Int32(0xFFFFFFFE)
            out_base_idx = buffer_ops.extract_base_index(arg_out)

            def write_row_to_lds(*, mi, ii, row_in_tile, row, row_base_lds, col_base_local, num_acc_n, lds_out):
                tw = _w(buffer_ops.buffer_load(sorted_w_rsrc, row, vec_width=1, dtype=T.f32))
                for ni in range_constexpr(num_acc_n):
                    col_local = col_base_local + (ni * 16)
                    idx = mi * num_acc_n + ni
                    v = _w(vector.extract(arith.unwrap(acc[idx]), static_position=[ii], dynamic_position=[]))
                    v = v * tw
                    v_out = arith.trunc_f(T.bf16, v)
                    v1 = vector.from_elements(T.vec(1, T.bf16), [v_out])
                    vector.store(v1, lds_out, [row_base_lds + col_local], alignment=2)

            def precompute_row(*, row_local, row):
                fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=T.i32)
                row_i32 = arith.index_cast(T.i32, row)
                row_valid0 = arith.cmpi(arith.CmpIPredicate.ult, row_i32, num_valid_i32)
                t = fused2 & mask24_i32
                s = fused2 >> 24
                t_ok = arith.cmpi(arith.CmpIPredicate.ult, t, tokens_i32)
                s_ok = arith.cmpi(arith.CmpIPredicate.ult, s, topk_i32)
                return (fused2, row_valid0 & t_ok & s_ok)

            def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                t = row_ctx & mask24_i32
                idx_elem = t * model_i32 + arith.index_cast(T.i32, col_g0)
                idx_elem_even = idx_elem & mask_even_i32
                byte_off = idx_elem_even * c2_i32
                ptr_addr_idx = out_base_idx + arith.index_cast(T.index, byte_off)
                out_ptr = buffer_ops.create_llvm_ptr(ptr_addr_idx, address_space=1)
                out_ptr_v = out_ptr._value if hasattr(out_ptr, "_value") else out_ptr
                frag_v = frag._value if hasattr(frag, "_value") else frag
                if const_expr(_DBG_ATOM == 0):
                    llvm.AtomicRMWOp(llvm.AtomicBinOp.fadd, out_ptr_v, frag_v, llvm.AtomicOrdering.monotonic,
                                     syncscope="agent", alignment=4)
                elif const_expr(_DBG_ATOM == 1):
                    llvm.StoreOp(frag_v, out_ptr_v, alignment=4)

            c_shuffle_epilog(
                arith=arith, vector=vector, gpu=gpu, scf=scf, range_constexpr=range_constexpr,
                tile_m=tile_m, tile_n=tile_n, e_vec=e_vec, m_repeat=m_repeat, num_acc_n=num_acc_n,
                tx=tx, lane_div_16=lane_div_16, lane_mod_16=lane_mod_16, bx_m=bx_m, by_n=by_n,
                n_tile_base=n_tile_base, lds_out=lds_out, frag_elem_type=T.bf16,
                write_row_to_lds=write_row_to_lds, precompute_row=precompute_row, store_pair=store_pair)

    @flyc.jit
    def launch_gemm2(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_s_stride: fx.Int32,
        i32_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        gy = arith.index_cast(T.index, i32_blocks)
        moe_fp8bs_gemm2(
            arg_out, arg_x, arg_w, arg_scale_x, arg_scale_w, arg_sorted_token_ids, arg_expert_ids,
            arg_sorted_weights, arg_num_valid_ids, i32_tokens_in, i32_s_stride,
        ).launch(grid=(model_dim // tile_n, gy, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm2
