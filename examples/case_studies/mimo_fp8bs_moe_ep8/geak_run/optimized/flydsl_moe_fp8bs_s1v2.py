"""Alternative FlyDSL stage1 (gate/up + SiLU*up + 1x128 fp8 requant) for the fp8 block-scale MoE, gfx942.

compile_stage1_t: same inputs/outputs as flydsl_moe_fp8bs.compile_stage1 (a2 [S, inter] fp8 by sorted row,
a2 scale [inter/128, S] kb-major) but a structurally different main loop:
  * transposed MFMA: weights are the src-A operand, tokens the src-B operand, so each lane's accumulator
    column is one token and the x block-scale is a per-lane scalar;
  * every wave owns all 128 tokens of the tile x (32 gate + 32 up) columns -> half the weight stream per
    FLOP of the 64-row gemm1;
  * the block-scale fold is done by RESCALING the running accumulator once per 128-K block
    (U <- U * s_{kb-1}/s_kb, then MFMA accumulates straight into U; out = U * s_last). No partial
    accumulators, one v_pk_mul per 2 accumulator elements per K block, 244 VGPR -> 2 waves/SIMD;
  * X staged through LDS in K=64 ping-pong stages, weights prefetched one stage ahead in registers;
  * rolled scf.for K loop (code ~2.5k ISA lines) and a per-tile branch that skips trailing all-padding
    16-row groups of the 128-row sorting block.
Requires moe_sorting with unit 128; stage2 (d0 gemm2, 64-row tiles) consumes the same sorted layout with
sorted_expert_ids repeated x2.
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
_SX_TAB = _os.environ.get("FBS_SX_TAB", "1") == "1"
_PK_SPLIT = _os.environ.get("FBS_PK_SPLIT", "1") == "1"
_VALID_TOK = _os.environ.get("FBS_VALID_TOK", "0") == "1"
_SLOAD = _os.environ.get("FBS_SLOAD", "1") == "1"
_SETPRIO = _os.environ.get("FBS_SETPRIO", "0") == "1"
_SCHED_MODE2 = int(_os.environ.get("FBS_SCHED2", "3"))
_SCHED_G1 = int(_os.environ.get("FBS_SCHED_G1", "8"))
_SCHED_G2 = int(_os.environ.get("FBS_SCHED_G2", "8"))


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


def _fma(a, b, c):
    if not _PK_SPLIT:
        return mmath.fma(arith.unwrap(a), arith.unwrap(b), arith.unwrap(c))
    from flydsl._mlir.dialects import vector as _mv
    a, b, c = arith.unwrap(a), arith.unwrap(b), arith.unwrap(c)
    v2 = T.vec(2, T.f32)
    outs = []
    for h in range(2):
        ah = _mv.extract_strided_slice(v2, a, [2 * h], [2], [1])
        bh = _mv.extract_strided_slice(v2, b, [2 * h], [2], [1])
        ch = _mv.extract_strided_slice(v2, c, [2 * h], [2], [1])
        outs.append(mmath.fma(ah, bh, ch))
    r = _mv.shuffle(outs[0], outs[1], [0, 1, 2, 3])
    return r


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
# Stage 1 "T" (transposed, 128 tokens per wave): weights are the MFMA src-A operand and tokens
# the src-B operand, so every lane's accumulator column is one token -> the x block-scale is a
# per-lane scalar. Each wave owns ALL tile_m tokens x (32 gate + 32 up) columns, which halves
# the weight (B) stream per FLOP vs the 64-row tile of gemm1. X is staged in K=64 LDS stages
# (2-deep ping-pong) and partials are folded once per stage.
# ----------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=64)
def compile_stage1_t(*, model_dim: int, inter_dim: int, experts: int, topk: int,
                     tile_m: int = 128, a_stream: int = 2, sched: int = 0, groups: int = 8,
                     setprio: int = 0, sw_sgpr: int = 0, wpe: int = 2, valu_per: int = 2, dbg_noscale: int = 0, dbg_nob: int = 0, rsplit: int = 1, rescale: int = 0, roll: int = 0, halfskip: int = 0, dbg_norw: int = 0, split_num: int = 0, post_defer: int = 2, xcd_swz: int = int(_os.environ.get("FBS_XCD", "1")), xcd_n: int = int(_os.environ.get("FBS_XCD_N", "8")):
    SK = 128 if rsplit else 64  # K bytes per LDS stage
    SROWS = tile_m // 2 if rsplit else tile_m  # token rows per LDS stage
    tile_n = BLK
    nwaves = 4
    total_threads = 64 * nwaves
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    assert model_dim % BLK == 0 and inter_dim % tile_n == 0 and tile_m % 16 == 0
    n_stages = 2 * (model_dim // BLK)
    num_kb = model_dim // BLK
    n_blk_total = (2 * inter_dim) // BLK
    m_repeat = tile_m // 16
    num_acc_n = (tile_n // nwaves) // 16  # 2
    stage_bytes = SROWS * SK
    x_chunks = SROWS * SK // 16
    mi_st = SROWS // 16  # 16-row groups per stage
    assert x_chunks % total_threads == 0
    num_x_loads = x_chunks // total_threads

    lds_x_bytes = 2 * stage_bytes
    lds_x_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_x_off + lds_x_bytes
    lds_sx_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_sx_off + num_kb * tile_m * 4
    # epilogue aliases: quant staging tile -> X region; row-max scratch -> sx[kb 0..] (long dead)
    assert tile_m * tile_n <= lds_x_bytes
    assert nwaves * tile_m <= 4 * tile_m and num_kb >= 4

    @flyc.kernel(known_block_size=[total_threads, 1, 1])
    def moe_fp8bs_s1t(
        arg_out: fx.Tensor,
        arg_out_scale: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_s_stride: fx.Int32,
        i32_blocks: fx.Int32,
    ):
        tokens_in = arith.index_cast(T.index, i32_tokens_in)
        s_stride = arith.index_cast(T.index, i32_s_stride)

        def silu(x):
            t = x * (-1.4426950408889634)
            emu = _w(rocdl.exp2(T.f32, arith.unwrap(t)))
            den = emu + 1.0
            sig = _w(rocdl.rcp(T.f32, arith.unwrap(den)))
            return x * sig

        acc_init = arith.constant_vector(0.0, T.f32x4)
        tx = gpu.thread_id("x")
        gx_c = inter_dim // tile_n  # number of N tiles (grid x)
        gy = arith.index_cast(T.index, i32_blocks)  # number of sorted M tiles
        _L_ok = None
        if const_expr(xcd_swz):
            # Device-side XCD-aware workgroup rasterization. The hardware dispatches a
            # 1-D grid round-robin across the xcd_n XCDs (physical XCD = raw % xcd_n), so
            # un-round-robin it: contiguous logical ids L map to one XCD. Walk order is
            # N-major (by outer, bx inner) so that, per XCD, every M-block that consumes the
            # same (expert, N-tile) of B co-resides and runs back-to-back -> B stays L2-hot.
            raw = gpu.block_id("x")
            nx = fx.Index(xcd_n)
            G = fx.Index(gx_c) * gy
            bpx = (G + nx - fx.Index(1)) // nx  # blocks per XCD (ceil)
            L = (raw % nx) * bpx + raw // nx
            _L_ok = arith.cmpi(arith.CmpIPredicate.ult,
                               arith.index_cast(T.i32, L), arith.index_cast(T.i32, G))
            by = L // gy  # N tile (128 gate + 128 up cols)
            bx = L % gy   # sorted M tile
        else:
            by = gpu.block_id("x")  # N tile (128 gate + 128 up cols)
            bx = gpu.block_id("y")  # sorted M tile
        bx_m = bx * fx.Index(tile_m)

        _nv_ptr = buffer_ops.create_llvm_ptr(buffer_ops.extract_base_index(arg_num_valid_ids), address_space=4)
        num_valid_i32 = _w(llvm.LoadOp(T.i32, _nv_ptr, alignment=4, invariant=True).result)
        bx_m_i32 = arith.index_cast(T.i32, bx_m)
        blk_valid = arith.cmpi(arith.CmpIPredicate.ult, bx_m_i32, num_valid_i32)
        if const_expr(xcd_swz):
            blk_valid = arith.andi(blk_valid, _L_ok)

        _if_blk = scf.IfOp(blk_valid)
        with _if_then(_if_blk):
            base_ptr = allocator.get_base()
            lds_x = SmemPtr(base_ptr, lds_x_off, T.f8, shape=(lds_x_bytes,)).get()
            lds_q = SmemPtr(base_ptr, lds_x_off, T.i8, shape=(lds_x_bytes,)).get()
            lds_sx = SmemPtr(base_ptr, lds_sx_off, T.f32, shape=(num_kb * tile_m,)).get()
            lds_red = SmemPtr(base_ptr, lds_sx_off, T.f32, shape=(nwaves * tile_m,)).get()

            x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=False,
                                                       num_records_bytes=tokens_in * fx.Index(model_dim))
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
            mask24 = fx.Int32(0xFFFFFF)
            tokens_i32 = arith.index_cast(T.i32, tokens_in)

            def ix(v):
                return fx.Index(v) if isinstance(v, int) else v

            # ---- per-thread X load coordinates: chunk id -> (row, 16B col chunk) ----
            x_row_base_div4 = [[], []]
            x_lds_wr = []
            for i in range_constexpr(num_x_loads):
                cid = tx + fx.Index(i * total_threads)
                row = cid // fx.Index(SK // 16)
                cc = cid % fx.Index(SK // 16)
                for hh in range_constexpr(2 if rsplit else 1):
                    fused_i = buffer_ops.buffer_load(sorted_rsrc, bx_m + fx.Index(hh * SROWS) + row, vec_width=1, dtype=T.i32)
                    t_raw = fused_i & mask24
                    t_ok = arith.cmpi(arith.CmpIPredicate.ult, t_raw, tokens_i32)
                    t_safe = t_ok.select(arith.index_cast(T.index, t_raw), fx.Index(0))
                    x_row_base_div4[hh].append(t_safe * fx.Index(model_dim // 4) + cc * fx.Index(4))
                if const_expr(rsplit):
                    swz = cc ^ arith.andi(row, fx.Index(7))
                else:
                    swz = cc ^ arith.andi(row // fx.Index(2), fx.Index(3))
                x_lds_wr.append(row * fx.Index(SK) + swz * fx.Index(16))

            def load_x(s):
                parts = []
                if const_expr(rsplit):
                    hh = s % 2
                    koff = fx.Index((s // 2) * (BLK // 4))
                else:
                    hh = 0
                    koff = ix(s) * fx.Index(SK // 4)
                for i in range_constexpr(num_x_loads):
                    v = buffer_copy_gmem16_dwordx4(
                        buffer_ops, vector, elem_type=T.f8,
                        idx_i32=x_row_base_div4[hh][i] + koff, rsrc=x_rsrc,
                        vec_elems=16, elem_bytes=1)
                    parts.append(vector.bitcast(T.i32x4, v))
                return parts

            def store_x(parts, buf):
                for i in range_constexpr(num_x_loads):
                    v16 = vector.bitcast(T.vec(16, T.f8), parts[i])
                    vector.store(v16, lds_x, [x_lds_wr[i] + fx.Index(buf * stage_bytes)])

            # ---- x_scale rows of this tile -> LDS [kb, row] ----
            n_sx_chunks = tile_m * (num_kb // 4)
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
                if const_expr(rescale == 2):
                    # ratio table rx[kb] = sx[kb-1]/sx[kb] (kb>=1); rx[0] = sx[last]
                    kprev = (kq * fx.Index(4) + fx.Index(num_kb - 1)) % fx.Index(num_kb)
                    pv = buffer_ops.buffer_load(sx_rsrc, t_s_idx * fx.Index(num_kb) + kprev, vec_width=1, dtype=T.f32)
                    epsv = _w(arith.constant(1e-20, type=T.f32))
                    cl = [_w(arith.maxnumf(_w(vector.extract(sv, static_position=[j], dynamic_position=[])), epsv))
                          for j in range_constexpr(4)]
                    cp = _w(arith.maxnumf(_w(pv), epsv))
                    first = arith.cmpi(arith.CmpIPredicate.eq, arith.index_cast(T.i32, kq), fx.Int32(0))
                    r0 = first.select(cp, cp * _w(rocdl.rcp(T.f32, arith.unwrap(cl[0]))))
                    vals = [r0] + [cl[j - 1] * _w(rocdl.rcp(T.f32, arith.unwrap(cl[j]))) for j in range_constexpr(1, 4)]
                else:
                    vals = [vector.extract(sv, static_position=[j], dynamic_position=[]) for j in range_constexpr(4)]
                for j in range_constexpr(4):
                    e = vals[j]
                    lidx = (kq * fx.Index(4) + fx.Index(j)) * fx.Index(tile_m) + r_loc
                    vector.store(vector.from_elements(T.vec(1, T.f32), [arith.unwrap(e)]), lds_sx, [lidx])

            wave_id = tx // fx.Index(64)
            lane_id = tx % fx.Index(64)
            lane_div_16 = lane_id // fx.Index(16)
            lane_mod_16 = lane_id % fx.Index(16)

            # LDS read base (bytes) for token row lane_mod_16, K chunk lane_div_16 of a stage
            if const_expr(rsplit):
                x_lds_rd = [lane_mod_16 * fx.Index(SK)
                            + ((lane_div_16 + fx.Index(ku * 4)) ^ arith.andi(lane_mod_16, fx.Index(7))) * fx.Index(16)
                            for ku in range(2)]
            else:
                rd_swz = lane_div_16 ^ arith.andi(lane_mod_16 // fx.Index(2), fx.Index(3))
                x_lds_rd = [lane_mod_16 * fx.Index(SK) + rd_swz * fx.Index(16)]

            # ---- preshuffled weight offsets (bytes) for (ni, gate/up); stage adds s*1024 ----
            # layout: [n/16][k/64][klane 4][nlane 16][16B]; n-block stride = (model_dim/64)*1024
            n0_stride = (model_dim // 64) * 1024
            col_w = by * fx.Index(tile_n) + wave_id * fx.Index(tile_n // nwaves)
            lane_part = lane_div_16 * fx.Index(256) + lane_mod_16 * fx.Index(16)
            boff = []
            for gu in range_constexpr(2):
                lst = []
                for ni in range_constexpr(num_acc_n):
                    r16 = (expert_idx * fx.Index(2 * inter_dim) + fx.Index(gu * inter_dim) + col_w
                           + fx.Index(ni * 16)) // fx.Index(16)
                    lst.append((r16 * fx.Index(n0_stride) + lane_part) // fx.Index(4))
                boff.append(lst)

            def load_b(s):
                res = []
                for gu in range_constexpr(2):
                    lst = []
                    for ni in range_constexpr(num_acc_n):
                        v = buffer_ops.buffer_load(w_rsrc, boff[gu][ni], vec_width=4, dtype=T.i32,
                                                   soffset_bytes=(s * 1024 if isinstance(s, int) else s * fx.Index(1024)))
                        v2 = vector.bitcast(T.i64x2, v)
                        lst.append((vector.extract(v2, static_position=[0], dynamic_position=[]),
                                    vector.extract(v2, static_position=[1], dynamic_position=[])))
                    res.append(lst)
                return res

            # weight scales (uniform per WG: tile_n == 128 -> one N block)
            sw_g_base = (expert_idx * fx.Index(n_blk_total) + by) * fx.Index(num_kb)
            sw_u_base = sw_g_base + fx.Index((inter_dim // BLK) * num_kb)

            def load_sw(kb):
                g = buffer_ops.buffer_load(sw_rsrc, sw_g_base + ix(kb), vec_width=1, dtype=T.f32)
                u = buffer_ops.buffer_load(sw_rsrc, sw_u_base + ix(kb), vec_width=1, dtype=T.f32)
                if const_expr(sw_sgpr):
                    g = rocdl.readfirstlane(T.f32, g) if hasattr(rocdl, "readfirstlane") else g
                    u = rocdl.readfirstlane(T.f32, u) if hasattr(rocdl, "readfirstlane") else u
                return g, u

            mfma_fn = rocdl.mfma_f32_16x16x32_fp8_fp8
            acc_g = [acc_init] * (m_repeat * num_acc_n)
            acc_u = [acc_init] * (m_repeat * num_acc_n)

            EPS = 1e-20

            def ld_sx(kb_, mi_abs):
                v = vector.extract(
                    vector.load_op(T.vec(1, T.f32), lds_sx, [ix(kb_) * fx.Index(tile_m) + fx.Index(mi_abs * 16) + lane_mod_16]),
                    static_position=[0], dynamic_position=[])
                if const_expr(rescale == 2):
                    return _w(v)
                return _w(arith.maxnumf(_w(v), _w(arith.constant(EPS, type=T.f32))))

            def compute_stage(s, b, swg, swu, acc_g_in, acc_u_in, rw=None, kb_dyn=None, rw_post=None):
                g_list = list(acc_g_in)
                u_list = list(acc_u_in)
                buf = s % 2
                kb = s // 2 if kb_dyn is None else kb_dyn

                mbase = (s % 2) * mi_st if rsplit else 0

                def read_a(mi):
                    xs = []
                    for ku in range_constexpr(len(x_lds_rd)):
                        idx = x_lds_rd[ku] + fx.Index(buf * stage_bytes + mi * 16 * SK)
                        loaded = vector.load_op(T.vec(16, T.f8), lds_x, [idx])
                        a2 = vector.bitcast(T.i64x2, loaded)
                        xs.append(vector.extract(a2, static_position=[0], dynamic_position=[]))
                        xs.append(vector.extract(a2, static_position=[1], dynamic_position=[]))
                    hm = (cur_m[0] * split_num) // 8
                    if const_expr(rescale == 2):
                        if const_expr(rw is not None and mi >= hm):
                            return xs, ld_sx(kb, mbase + mi)
                        if const_expr(rw_post is not None and mi < hm):
                            return xs, ld_sx((kb + 1) if isinstance(kb, int) else kb + fx.Index(1), mbase + mi)
                        return xs, None
                    if const_expr(rescale):
                        if const_expr(rw is not None):
                            sp = ld_sx((kb - 1) if isinstance(kb, int) else kb - fx.Index(1), mbase + mi)
                            sc_ = ld_sx(kb, mbase + mi)
                            return xs, sp * _w(rocdl.rcp(T.f32, arith.unwrap(sc_)))
                        return xs, None
                    sxv = vector.extract(
                        vector.load_op(T.vec(1, T.f32), lds_sx,
                                       [ix(kb) * fx.Index(tile_m) + fx.Index((mbase + mi) * 16) + lane_mod_16]),
                        static_position=[0], dynamic_position=[])
                    return xs, sxv

                a_all = [None] * cur_m[0]
                for mi in range_constexpr(min(a_stream, cur_m[0])):
                    a_all[mi] = read_a(mi)
                pending = []

                def flush(pend):
                    for (mi_p, sxv_p, pgs, pus) in pend:
                        sg = _w(sxv_p) * _w(swg)
                        su = _w(sxv_p) * _w(swu)
                        sgv = vector.broadcast(T.f32x4, arith.unwrap(sg))
                        suv = vector.broadcast(T.f32x4, arith.unwrap(su))
                        for ni in range_constexpr(num_acc_n):
                            idx = (mbase + mi_p) * num_acc_n + ni
                            g_list[idx] = _fma(pgs[ni], sgv, g_list[idx])
                            u_list[idx] = _fma(pus[ni], suv, u_list[idx])

                post_q = []

                def apply_post(item):
                    gi, fx_ = item
                    rgp = vector.broadcast(T.f32x4, arith.unwrap(_w(fx_) * _w(rw_post[0])))
                    rup = vector.broadcast(T.f32x4, arith.unwrap(_w(fx_) * _w(rw_post[1])))
                    for ni in range_constexpr(num_acc_n):
                        g_list[gi + ni] = _w(g_list[gi + ni]) * _w(rgp)
                        u_list[gi + ni] = _w(u_list[gi + ni]) * _w(rup)

                for mi in range_constexpr(cur_m[0]):
                    if const_expr(mi + a_stream < cur_m[0]):
                        a_all[mi + a_stream] = read_a(mi + a_stream)
                    xs, sxv = a_all[mi]
                    gidx = (mbase + mi) * num_acc_n
                    hm2 = (cur_m[0] * split_num) // 8
                    if const_expr(rescale and rw is not None and (rescale == 1 or mi >= hm2)):
                        rgv = vector.broadcast(T.f32x4, arith.unwrap(_w(sxv) * _w(rw[0])))
                        ruv = vector.broadcast(T.f32x4, arith.unwrap(_w(sxv) * _w(rw[1])))
                        for ni in range_constexpr(num_acc_n):
                            g_list[gidx + ni] = _w(g_list[gidx + ni]) * _w(rgv)
                            u_list[gidx + ni] = _w(u_list[gidx + ni]) * _w(ruv)
                    acc_mode = dbg_noscale or rescale
                    pgs = [g_list[gidx + ni] if acc_mode else acc_init for ni in range(num_acc_n)]
                    pus = [u_list[gidx + ni] if acc_mode else acc_init for ni in range(num_acc_n)]
                    for kq in range_constexpr(len(xs)):
                        for ni in range_constexpr(num_acc_n):
                            pgs[ni] = mfma_fn(T.f32x4, [b[kq // 2][0][ni][kq % 2], xs[kq], pgs[ni], 0, 0, 0])
                            pus[ni] = mfma_fn(T.f32x4, [b[kq // 2][1][ni][kq % 2], xs[kq], pus[ni], 0, 0, 0])
                    if const_expr(dbg_noscale or rescale):
                        for ni in range_constexpr(num_acc_n):
                            g_list[gidx + ni] = pgs[ni]
                            u_list[gidx + ni] = pus[ni]
                        if const_expr(rescale == 2 and rw_post is not None and mi < hm2):
                            post_q.append((gidx, sxv))
                        for _q in range_constexpr(max(0, len(post_q) - post_defer)):
                            apply_post(post_q.pop(0))
                    else:
                        flush(pending)
                        pending = [(mi, sxv, pgs, pus)]
                for it_ in range_constexpr(len(post_q)):
                    apply_post(post_q[it_])
                flush(pending)
                return g_list, u_list

            cur_m = [m_repeat]

            def tail(m_act):
                cur_m[0] = m_act
                acc_g = [acc_init] * (m_repeat * num_acc_n)
                acc_u = [acc_init] * (m_repeat * num_acc_n)
                # ---------------- main loop over K=64 stages ----------------
                rocdl.sched_barrier(0)
                xr = load_x(0)
                if const_expr(rsplit):
                    bq = [[load_b(0), load_b(1)]]
                else:
                    bq = [[load_b(0)]]
                sw_cur = load_sw(0)
                sw_nxt = sw_cur
                sw_prev = sw_cur
                store_x(xr, 0)
                gpu.barrier()
                def _cl(v):
                    return _w(arith.maximumf(_w(v), _w(arith.constant(EPS, type=T.f32))))

                def ratio(swp, swc):
                    return (_cl(swp[0]) * _w(rocdl.rcp(T.f32, arith.unwrap(_cl(swc[0])))),
                            _cl(swp[1]) * _w(rocdl.rcp(T.f32, arith.unwrap(_cl(swc[1])))))

                def t_sched(par):
                    nvm = num_x_loads + 2 * num_acc_n + (2 if par == 0 else 0)
                    rocdl.sched_dsrd(2 * min(a_stream, cur_m[0]))
                    per_g = (nvm + cur_m[0] - 1) // cur_m[0]
                    left = nvm
                    for g in range_constexpr(cur_m[0]):
                        if const_expr(g + a_stream < cur_m[0]):
                            rocdl.sched_dsrd(2)
                        if const_expr(left > 0):
                            rocdl.sched_vmem(min(per_g, left))
                        left -= per_g
                        for j in range_constexpr(num_acc_n * 4):
                            rocdl.sched_mfma(1)
                            rocdl.sched_group_barrier(0x2, valu_per, 0)
                    rocdl.sched_dswr(num_x_loads)
                    rocdl.sched_barrier(0)

                def stage_pair(kb, bcur, swc, swp, accg, accu, first, last):
                    dyn = not isinstance(kb, int)
                    s0 = kb * fx.Index(2) if dyn else 2 * kb
                    s1 = s0 + fx.Index(1) if dyn else s0 + 1
                    xr1 = load_x(s1)
                    b1 = bcur[0] if dbg_nob else load_b(s1)
                    if const_expr(last):
                        swn = swc
                    else:
                        swn = load_sw(kb + fx.Index(1) if dyn else kb + 1)
                    rw_ = None if (first or dbg_norw) else ratio(swp, swc)
                    if const_expr(setprio):
                        rocdl.s_setprio(1)
                    accg, accu = compute_stage(0, bcur, swc[0], swc[1], accg, accu, rw_, kb_dyn=kb)
                    store_x(xr1, 1)
                    if const_expr(sched == 10):
                        t_sched(0)
                    gpu.barrier()
                    b2 = None
                    xr2 = None
                    if const_expr(not last):
                        s2 = s0 + fx.Index(2) if dyn else s0 + 2
                        xr2 = load_x(s2)
                        b2 = bcur[0] if dbg_nob else load_b(s2)
                    rwp = None if (last or rescale != 2) else ratio(swc, swn)
                    accg, accu = compute_stage(1, [b1], swc[0], swc[1], accg, accu, None, kb_dyn=kb, rw_post=rwp)
                    if const_expr(not last):
                        store_x(xr2, 0)
                        if const_expr(sched == 10):
                            t_sched(1)
                        gpu.barrier()
                    if const_expr(setprio):
                        rocdl.s_setprio(0)
                    return accg, accu, b2, swn, swc

                nacc = m_repeat * num_acc_n

                def pack(accg, accu, b, swc, swp):
                    st = list(accg) + list(accu)
                    for gu in range_constexpr(2):
                        for ni in range_constexpr(num_acc_n):
                            st += [b[0][gu][ni][0], b[0][gu][ni][1]]
                    return st + [swc[0], swc[1], swp[0], swp[1]]

                def unpack(st):
                    st = list(st)
                    accg = st[:nacc]
                    accu = st[nacc:2 * nacc]
                    o = 2 * nacc
                    bb = []
                    for gu in range_constexpr(2):
                        lst = []
                        for ni in range_constexpr(num_acc_n):
                            lst.append((st[o], st[o + 1]))
                            o += 2
                        bb.append(lst)
                    return accg, accu, [bb], (st[o], st[o + 1]), (st[o + 2], st[o + 3])

                if const_expr(roll):
                    acc_g, acc_u, b_n, sw_c, sw_p = stage_pair(0, bq[0], sw_cur, sw_cur, acc_g, acc_u, True, False)
                    init_state = pack(acc_g, acc_u, [b_n], sw_c, sw_p)
                    results = init_state
                    for iv, state in range(1, num_kb - 1, 1, init=init_state):
                        ag, au, bb, swc_, swp_ = unpack(state)
                        ag, au, b_n2, swc2, swp2 = stage_pair(iv, bb, swc_, swp_, ag, au, False, False)
                        results = yield pack(ag, au, [b_n2], swc2, swp2)
                    ag, au, bb, swc_, swp_ = unpack(results)
                    acc_g, acc_u, _, sw_cur, _ = stage_pair(num_kb - 1, bb, swc_, swp_, ag, au, False, True)
                for s in range_constexpr(0 if roll else n_stages):
                    if const_expr(s + 1 < n_stages):
                        xr_n = load_x(s + 1)
                    if const_expr(rsplit):
                        if const_expr(s % 2 == 0 and s + 2 < n_stages):
                            bq.append(bq[0] if dbg_nob else [load_b(s + 2), load_b(s + 3)])
                    elif const_expr(s + 1 < n_stages):
                        bq.append(bq[0] if dbg_nob else [load_b(s + 1)])
                    if const_expr(s % 2 == 0 and s // 2 + 1 < num_kb):
                        sw_nxt = load_sw(s // 2 + 1)
                    b_cur = bq[0]
                    if const_expr((not rsplit) or s % 2 == 1):
                        bq.pop(0)
                    if const_expr(setprio):
                        rocdl.s_setprio(1)
                    rw = None
                    if const_expr(rescale and s // 2 > 0 and (rsplit or s % 2 == 0)):
                        def _cl(v):
                            return _w(arith.maximumf(_w(v), _w(arith.constant(EPS, type=T.f32))))
                        rw = (_cl(sw_prev[0]) * _w(rocdl.rcp(T.f32, arith.unwrap(_cl(sw_cur[0])))),
                              _cl(sw_prev[1]) * _w(rocdl.rcp(T.f32, arith.unwrap(_cl(sw_cur[1])))))
                    acc_g, acc_u = compute_stage(s, b_cur, sw_cur[0], sw_cur[1], acc_g, acc_u, rw)
                    if const_expr(setprio):
                        rocdl.s_setprio(0)
                    if const_expr(s + 1 < n_stages):
                        store_x(xr_n, (s + 1) % 2)
                        if const_expr(sched == 10):
                            nbl = (4 * num_acc_n if s % 2 == 0 else 0) if rsplit else 2 * num_acc_n
                            nvm = num_x_loads + nbl + (2 if s % 2 == 0 and s // 2 + 1 < num_kb else 0)
                            nrd = len(x_lds_rd) + 1
                            rocdl.sched_dsrd(nrd * min(a_stream, mi_st))
                            per_g = (nvm + cur_m[0] - 1) // cur_m[0]
                            left = nvm
                            for g in range_constexpr(mi_st):
                                if const_expr(g + a_stream < mi_st):
                                    rocdl.sched_dsrd(nrd)
                                if const_expr(left > 0):
                                    rocdl.sched_vmem(min(per_g, left))
                                left -= per_g
                                for j in range_constexpr(num_acc_n * 4 * len(x_lds_rd)):
                                    rocdl.sched_mfma(1)
                                    rocdl.sched_group_barrier(0x2, valu_per, 0)
                            rocdl.sched_dswr(num_x_loads)
                            rocdl.sched_barrier(0)
                        elif const_expr(sched > 0):
                            hot_loop_sched(mode=sched, n_dsrd=m_repeat * 2, n_vmem=num_x_loads + 2 * num_acc_n,
                                           n_mfma=m_repeat * num_acc_n * 4, n_dswr=num_x_loads, groups=groups)
                        gpu.barrier()
                    if const_expr(s % 2 == 1):
                        sw_prev = sw_cur
                        sw_cur = sw_nxt

                # ---------------- epilogue: silu*up, 1x128 per-token quant, fp8 store ----------------
                if const_expr(rescale):
                    swl_g = _w(arith.maximumf(_w(sw_cur[0]), _w(arith.constant(EPS, type=T.f32))))
                    swl_u = _w(arith.maximumf(_w(sw_cur[1]), _w(arith.constant(EPS, type=T.f32))))
                    for mi in range_constexpr(m_repeat):
                        sl = ld_sx(0 if rescale == 2 else num_kb - 1, mi)
                        sgv = vector.broadcast(T.f32x4, arith.unwrap(sl * swl_g))
                        suv = vector.broadcast(T.f32x4, arith.unwrap(sl * swl_u))
                        for ni in range_constexpr(num_acc_n):
                            acc_g[mi * num_acc_n + ni] = _w(acc_g[mi * num_acc_n + ni]) * _w(sgv)
                            acc_u[mi * num_acc_n + ni] = _w(acc_u[mi * num_acc_n + ni]) * _w(suv)
                ys = []
                for mi in range_constexpr(m_repeat):
                    for ni in range_constexpr(num_acc_n):
                        idx = mi * num_acc_n + ni
                        yv = []
                        for r in range_constexpr(4):
                            vg = vector.extract(arith.unwrap(acc_g[idx]), static_position=[r], dynamic_position=[])
                            vu = vector.extract(arith.unwrap(acc_u[idx]), static_position=[r], dynamic_position=[])
                            yv.append(silu(_w(vg)) * _w(vu))
                        ys.append(yv)

                c64_i32 = arith.constant(64, type=T.i32)
                for mi in range_constexpr(m_repeat):
                    m = None
                    for ni in range_constexpr(num_acc_n):
                        for r in range_constexpr(4):
                            av = _w(mmath.absf(arith.unwrap(ys[mi * num_acc_n + ni][r])))
                            m = av if m is None else _w(arith.maximumf(m, av))
                    for sh in (16, 32):
                        peer = m.shuffle_xor(arith.constant(sh, type=T.i32), c64_i32)
                        m = _w(arith.maximumf(m, _w(peer)))
                    ridx = wave_id * fx.Index(tile_m) + fx.Index(mi * 16) + lane_mod_16
                    vector.store(vector.from_elements(T.vec(1, T.f32), [m]), lds_red, [ridx])
                gpu.barrier()  # row maxima visible; all waves done with X / sx LDS

                for mi in range_constexpr(m_repeat):
                    tok_l = fx.Index(mi * 16) + lane_mod_16
                    m = None
                    for wv in range_constexpr(nwaves):
                        v = _w(vector.extract(vector.load_op(T.vec(1, T.f32), lds_red, [fx.Index(wv * tile_m) + tok_l]),
                                              static_position=[0], dynamic_position=[]))
                        m = v if m is None else _w(arith.maximumf(m, v))
                    m = _w(arith.maximumf(m, _w(arith.constant(1e-10, type=T.f32))))
                    inv = _w(rocdl.rcp(T.f32, m * (1.0 / FP8_MAX)))
                    for ni in range_constexpr(num_acc_n):
                        yv = ys[mi * num_acc_n + ni]
                        q = [_w(yv[r]) * inv for r in range(4)]
                        p = rocdl.cvt_pk_fp8_f32(T.i32, q[0], q[1], arith.constant(0, type=T.i32), 0)
                        p = rocdl.cvt_pk_fp8_f32(T.i32, q[2], q[3], p, 1)
                        qidx = tok_l * fx.Index(tile_n) + wave_id * fx.Index(tile_n // nwaves) \
                            + fx.Index(ni * 16) + lane_div_16 * fx.Index(4)
                        v4 = vector.bitcast(T.vec(4, T.i8), vector.from_elements(T.vec(1, T.i32), [p]))
                        vector.store(v4, lds_q, [qidx])
                # per-token scale: threads tx < tile_m
                for it in range_constexpr((tile_m + total_threads - 1) // total_threads):
                    lin = tx + fx.Index(it * total_threads)
                    ok = arith.cmpi(arith.CmpIPredicate.ult, arith.index_cast(T.i32, lin), fx.Int32(tile_m))
                    _if_s = scf.IfOp(ok)
                    with _if_then(_if_s):
                        m = None
                        for wv in range_constexpr(nwaves):
                            v = _w(vector.extract(vector.load_op(T.vec(1, T.f32), lds_red, [fx.Index(wv * tile_m) + lin]),
                                                  static_position=[0], dynamic_position=[]))
                            m = v if m is None else _w(arith.maximumf(m, v))
                        m = _w(arith.maximumf(m, _w(arith.constant(1e-10, type=T.f32))))
                        sc = m * (1.0 / FP8_MAX)
                        buffer_ops.buffer_store(sc, outs_rsrc, by * s_stride + bx_m + lin)
                gpu.barrier()
                chunks = tile_m * tile_n // 16
                for it in range_constexpr((chunks + total_threads - 1) // total_threads):
                    lin = tx + fx.Index(it * total_threads)
                    rl = lin // fx.Index(tile_n // 16)
                    cc = lin % fx.Index(tile_n // 16)
                    v = vector.load_op(T.vec(16, T.i8), lds_q, [rl * fx.Index(tile_n) + cc * fx.Index(16)])
                    vi = vector.bitcast(T.i32x4, v)
                    dw = ((bx_m + rl) * fx.Index(inter_dim) + by * fx.Index(tile_n) + cc * fx.Index(16)) // fx.Index(4)
                    buffer_ops.buffer_store(vi, out_rsrc, dw)

            if const_expr(halfskip):
                ntv_ptr = buffer_ops.create_llvm_ptr(buffer_ops.extract_base_index(arg_num_valid_ids) + fx.Index(4),
                                                     address_space=4)
                ntok_valid = _w(llvm.LoadOp(T.i32, ntv_ptr, alignment=4, invariant=True).result)
                def is_pad(row_off):
                    fh = buffer_ops.buffer_load(sorted_rsrc, bx_m + fx.Index(row_off), vec_width=1, dtype=T.i32)
                    return arith.cmpi(arith.CmpIPredicate.uge, fh & mask24, ntok_valid)

                def branch(lo, hi):
                    # valid 16-row groups are a prefix; choose m_act in (lo, hi] by bisection
                    if const_expr(hi - lo <= {1: 4, 2: 2}.get(halfskip, 1)):
                        tail(hi)
                    else:
                        mid = (lo + hi) // 2
                        _if_h = scf.IfOp(is_pad(mid * 16), has_else=True)
                        with ir.InsertionPoint(_if_h.then_block):
                            branch(lo, mid)
                            scf.YieldOp([])
                        with ir.InsertionPoint(_if_h.else_block):
                            branch(mid, hi)
                            scf.YieldOp([])
                branch(0, m_repeat)
            else:
                tail(m_repeat)

    @flyc.jit
    def launch_s1t(
        arg_out: fx.Tensor,
        arg_out_scale: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
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
        if const_expr(xcd_swz):
            grid = (arith.index_cast(T.index, i32_blocks) * fx.Index(inter_dim // tile_n), 1, 1)
        else:
            grid = (inter_dim // tile_n, gy, 1)
        moe_fp8bs_s1t(
            arg_out, arg_out_scale, arg_x, arg_w, arg_scale_x, arg_scale_w,
            arg_sorted_token_ids, arg_expert_ids, arg_num_valid_ids, i32_tokens_in, i32_s_stride,
            i32_blocks,
        ).launch(grid=grid, block=(total_threads, 1, 1), stream=stream)
        if const_expr(wpe > 0):
            for op in ctx.gpu_module_body.operations:
                if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, int(wpe))

    return launch_s1t
