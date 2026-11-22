"""FlyDSL stage-2 kernel for the fp8 block-scale MoE (down GEMM, routing-weighted bf16 atomic scatter-add).

out[t, n] += w_route[row] * sum_kb ( sum_{k in kb} A2[p,k] * W2[e, n, k] ) * sa2[p, kb] * sw2[e, n/128, kb]
  A2 : a2 fp8 [tokens*topk, inter] (p = token*topk + slot) + a2_scale fp32 [tokens*topk, inter/128]
       (exactly what flydsl_stage1_fp8bs writes)
  W2 : w2 fp8 [E, hidden, inter] preshuffled with shuffle_weight(w, (16, 16)), w2_scale [E, hidden/128, inter/128]
  out: bf16 [tokens, hidden], pre-zeroed (rows < num_local_tokens) by the stage-1 kernel.

Same main-loop template as flydsl_stage1_fp8bs (3-deep LDS ring for A with a mid-tile barrier, B streamed
gmem->VGPR one K-tile ahead, per-128-K block-scale promotion of short-lived MFMA partials), with a single
B stream of tile_n columns.  Epilogue: acc * routing weight -> bf16 LDS tile (128-col chunks) -> each lane
adds 8 contiguous bf16 with 4 global_atomic_pk_add_bf16.
"""
import functools
import os
from pathlib import Path

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.compiler.kernel_function import CompilationContext  # noqa: E402
from flydsl.expr import arith, gpu, buffer_ops, vector, rocdl  # noqa: E402
from flydsl.expr import range_constexpr, const_expr  # noqa: E402
from flydsl.expr import math as fmath  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr  # noqa: E402
from flydsl._mlir import ir  # noqa: E402
from flydsl._mlir.dialects import scf, llvm  # noqa: E402
from flydsl.expr.typing import T  # noqa: E402

from aiter.ops.flydsl.kernels.mfma_preshuffle_pipeline import (  # noqa: E402
    crd2idx,
    lds_store_16b_xor16,
    swizzle_xor16,
    tile_chunk_coord_i32,
)


def _w(v):
    from flydsl.expr.arith import ArithValue
    if isinstance(v, ir.Value) and not isinstance(v, ArithValue):
        return ArithValue(v)
    return v


def _perm(s0, s1, sel):
    """v_perm_b32: result byte k = byte sel[k] of {s0 (bytes 7..4), s1 (bytes 3..0)}."""
    return _w(llvm.call_intrinsic(T.i32, "llvm.amdgcn.perm", [arith.unwrap(s0), arith.unwrap(s1),
                                                              arith.unwrap(fx.Int32(sel))], [], []))


def _bf16_bits_rne(f):
    """fp32 -> i32 whose high 16 bits are the RNE bf16 (finite inputs)."""
    b = _w(arith.bitcast(T.i32, f))
    lsb = (b >> fx.Int32(16)) & fx.Int32(1)
    return b + fx.Int32(0x7FFF) + lsb


_E2_RFL = os.environ.get("S2_E2_RFL", "1") == "1"
from flydsl._mlir.dialects import rocdl as _rocdl_d  # noqa: E402
_E2_BRANCHLESS = os.environ.get("S2_E2_BL", "1") == "1"
_VM_MODE = int(os.environ.get("S2_VM_MODE", "0"))
_XCM = int(os.environ.get("S2_XCM", "0"))
_BCM = int(os.environ.get("S2_BCM", "0"))
_B2_KU = int(os.environ.get("S2_B2_KU", "0"))
_E2_DBG = int(os.environ.get("S2_E2_DBG", "0"))
_E2_PRE = os.environ.get("S2_E2_PRE", "1") == "1"
# S2_PRIO=1 (default; r1_d2 + r2_d2 measured: 3 interleaved COMMANDMENT rounds 1.5329x -> 1.5425x, +0.63%):
# K main loop runs at s_setprio(1), prologue/epilogue at 0, so a
# co-resident CTA's prologue/epilogue cannot steal issue slots from the MFMA loop. S2_PRIO=0 = previous ISA.
_PRIO = int(os.environ.get("S2_PRIO", "1"))
# S2_DELOOP=1: one M tile per CTA behind a uniform if-guard (num_valid via SMEM), grid sized to the guaranteed
# bound (blocks = len(sorted_expert_ids)); replaces the dynamic grid-stride scf.for. 0 = previous loop.
_DELOOP = int(os.environ.get("S2_DELOOP", "0"))
# S2_XCD_GN>0 (requires deloop): remap the linear CTA id so each XCD (= linear id % 8) owns gn N tiles x one
# contiguous M partition (8*gn/16 partitions, computed on device from n_mblk). 0 = plain (16 N, M) grid.
_XCD_GN = int(os.environ.get("S2_XCD_GN", "0"))
# within-XCD order: 0 = M-major (gn N tiles of one M block back-to-back), 1 = chunked N-major (xcd_c M blocks
# per chunk, the chunk's N tiles back-to-back)
_XCD_ORD = int(os.environ.get("S2_XCD_ORD", "0"))
_XCD_C = int(os.environ.get("S2_XCD_C", "8"))
_DBG_GRID_CAP = 0  # diagnostic only (A/B tool): cap the deloop grid M (NOT correctness-safe)
_DBG = 0  # diagnostic only (A/B tool): overrides stage2_fp8bs(dbg=); 6 = no block-scale rescale (wrong numerics)
# S2_BSOFF=1: B (w2) / A (a2) K-tile, ni and ku deltas go to the buffer soffset (SGPR constant) instead of
# per-load VGPR address adds.
_BSOFF = int(os.environ.get("S2_BSOFF", "0"))
# S2_FPRE=1: the prologue stages f = a2_scale[row, kb] * w2_scale[e, nb, kb] (one table per 128-col block nb
# of the CTA's tile) in LDS, so the K loop rescale is a bare FMA (no per-kb sw load / sa*sw multiply).
_FPRE = int(os.environ.get("S2_FPRE", "1"))
# S2_EPI>=0 overrides stage2_fp8bs(epi_mode=). Default 3 = epi_mode 2 (row-half bf16 LDS transpose, preloaded ids,
# branchless pk_add_bf16 atomics) + the routing weight folded into the fpre scale table (no epilogue multiply / tw
# loads) + v_perm_b32 bf16 packing. Measured (isolated s2, 9 rounds): epi2 vs epi0 -1.4%; epi3 vs epi2 -1.3%/-1.2%.
_EPI = int(os.environ.get("S2_EPI", "3"))
# epi_mode 3 sub-knobs: S2_E3_BL=1 branchless dummy-row zero atomics (else uniform s_cbranch per row),
# S2_E3_PERM=1 v_perm_b32 bf16 packing, S2_E3_SADDR=1 SGPR row base + 32-bit lane offset atomic address
_E3_BL = int(os.environ.get("S2_E3_BL", "1"))  # 0 (uniform branch) spills 521 VGPR: do not use
_E3_PERM = int(os.environ.get("S2_E3_PERM", "1"))
_E3_SADDR = int(os.environ.get("S2_E3_SADDR", "1"))
_VA_NV = int(os.environ.get("S2_VA_NV", "1"))  # VALU per MFMA slot in the K-loop sched_group_barrier pattern (1: s2 -1.3%/-1.6% vs 2)
_VA_ST = int(os.environ.get("S2_VA_ST", "2"))  # first MFMA slot that takes VALU
_LLVM_OPTS = {}  # A/B tool only: LLVM cl::opt overrides applied while the stage-2 kernel JIT-compiles
_WPE = int(os.environ.get("S2_WPE", "2"))  # >0: rocdl.waves_per_eu attribute (2 = cap at 256 VGPR; fpre needs it)
# S2_PADSKIP: skip all-padding 16-row MFMA subtiles. moe_sort_fast emits the valid-row count of every M block
# (padding is a suffix of an expert's last block); the CTA s_loads it next to the expert id and selects, with
# uniform scalar branches, between the unchanged full body (m_repeat subtiles) and reduced-subtile copies of the
# K loop + epilogue. Value = descending comma list of active-subtile variants starting at m_repeat; "0" = off.
# Default "4,2,1" (blocks with 1 / 2 active subtiles run 1- / 2-subtile copies, 3-active blocks run the full body):
# isolated s2 -2.2% / -1.2% (9/9 rounds each); full-body ISA unchanged (1024 MFMA, 668 FMA, 178 pk_fma), 246 VGPR,
# no spills. "4,3,2,1" / "4,2" perturb the full body or spill (s2 -0.3%/0.0%, -0.5%/0.0%).
_PADSKIP = os.environ.get("S2_PADSKIP", "4,2,1")


def _parse_padskip(spec, m_repeat):
    if not spec or spec.strip() in ("0", "off", ""):
        return (m_repeat,)
    vs = tuple(int(v) for v in spec.split(",") if v.strip())
    if not vs or vs[0] != m_repeat or any(b >= a for a, b in zip(vs, vs[1:])) or vs[-1] < 1:
        raise ValueError(f"S2_PADSKIP={spec!r}: need a descending list starting at m_repeat={m_repeat}")
    return vs


def _variant_chain(cnt_i32, vs, body):
    """Nested uniform scf.if/else: valid rows > 16*vs[1] -> body(vs[0]); else recurse on vs[1:].
    Module level (not AST-rewritten): plain Python recursion emitting explicit scf ops."""
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


class _IfThen:
    def __init__(self, cond):
        self.op = scf.IfOp(cond)

    def __enter__(self):
        self.ip = ir.InsertionPoint(self.op.then_block)
        self.ip.__enter__()
        return self.op

    def __exit__(self, *exc):
        blk = self.op.then_block
        if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
            scf.YieldOp([])
        self.ip.__exit__(*exc)
        return False


@functools.lru_cache(maxsize=64)
def compile_stage2_fp8bs(
    *,
    hidden_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int = 64,
    tile_n: int = 128,
    sched: int = 1,
    epi_mode: int = 0,
    dbg: int = 0,
    a_sorted: int = 0,
    prio: int = 1,
    deloop: int = 0,
    xcd_gn: int = 0,
    xcd_ord: int = 0,
    xcd_c: int = 8,
    bsoff: int = 0,
    fpre: int = 0,
    wpe: int = 0,
    vm_mode: int = -1,
    e3_bl: int = 0,
    e3_perm: int = 1,
    e3_saddr: int = 1,
    va_nv: int = 2,
    va_st: int = 2,
    llvm_opts: tuple = (),
    bcm: int = -1,
    xcm: int = -1,
    b2ku: int = -1,
    padskip: tuple = (),
):
    vm_mode = _VM_MODE if vm_mode < 0 else vm_mode
    bcm = _BCM if bcm < 0 else bcm
    b2ku = _B2_KU if b2ku < 0 else b2ku
    xcm = _XCM if xcm < 0 else xcm
    tile_k = 128
    total_threads = 256
    num_waves = 4
    n_per_wave = tile_n // 4
    num_acc_n = n_per_wave // 16
    m_repeat = tile_m // 16
    k_unroll = tile_k // 64
    K = inter_dim
    nkt = K // tile_k
    nkb = K // 128
    n_sblk = hidden_dim // 128
    if hidden_dim % tile_n or K % tile_k or m_repeat < 2 or m_repeat % 2 or tile_n % 128 or n_per_wave > 128:
        raise ValueError("shape/tile mismatch")
    half = m_repeat // 2
    pad_variants = tuple(padskip) if padskip else (m_repeat,)
    if len(pad_variants) > 1 and (vm_mode != 0 or dbg != 0 or epi_mode != 3 or deloop or not fpre):
        raise ValueError("padskip variants are built for the default path (fpre, epi_mode 3, vm_mode 0, grid loop)")
    bytes_per_thread_x = tile_m * tile_k // total_threads
    if bytes_per_thread_x % 16 != 0:
        raise ValueError("tile_m too small")
    num_x_loads = bytes_per_thread_x // 16

    arch = get_rocm_arch()
    allocator = SmemAllocator(None, arch=arch)
    n_stages = 3
    lds_tile = tile_m * tile_k
    epi_cols = 128
    epi_stride = epi_cols + 8
    lds_a_bytes = n_stages * lds_tile
    lds_epi_bytes = tile_m * epi_stride * 2
    # epi_mode 2: row-half chunks, LDS [rowgroup(4 rows)][col][4 rows] bf16 (+pad per rowgroup)
    e2_rows = tile_m // 2
    e2_groups = e2_rows // 4
    e2_gstride = tile_n * 4 + 8
    e2_nhc = tile_n // 128
    e2_upw = e2_groups * e2_nhc // num_waves
    e2_gpw = e2_upw // e2_nhc
    if epi_mode == 3 and not fpre:
        raise ValueError("epi_mode 3 folds the routing weight into the fpre table")
    if epi_mode in (2, 3):
        assert m_repeat % 2 == 0 and e2_upw * num_waves == e2_groups * e2_nhc and e2_upw % e2_nhc == 0
        lds_epi_bytes = e2_groups * e2_gstride * 2
    region0 = max(lds_a_bytes, lds_epi_bytes)
    off0 = allocator._align(allocator.ptr, 16)
    allocator.ptr = off0 + region0
    n_nb = tile_n // 128 if fpre else 1
    off_sa = allocator._align(allocator.ptr, 16)
    allocator.ptr = off_sa + n_nb * nkt * tile_m * 4
    if allocator.ptr > 65536:
        raise ValueError(f"LDS overflow: {allocator.ptr}")
    n_tiles = hidden_dim // tile_n
    waves_per_chunk = epi_cols // n_per_wave  # waves whose columns fall into one 128-col epilogue chunk
    n_chunks = tile_n // epi_cols

    @flyc.kernel(known_block_size=[total_threads, 1, 1])
    def moe_s2_fp8bs(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_sx: fx.Tensor,
        arg_w: fx.Tensor,
        arg_sw: fx.Tensor,
        arg_sorted_ids: fx.Tensor,
        arg_sorted_w: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_num_valid: fx.Tensor,
        arg_blk_cnt: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_size_eids: fx.Int32,
        i32_s_stride: fx.Int32,
        i32_grid_m: fx.Int32,
    ):
        tokens_in = arith.index_cast(T.index, i32_tokens)
        size_eids = arith.index_cast(T.index, i32_size_eids)
        tx = gpu.thread_id("x")
        if const_expr(deloop):
            # uniform (SGPR) num_valid -> scalar branch for the one-tile-per-CTA guard
            _nv_ptr = buffer_ops.create_llvm_ptr(buffer_ops.extract_base_index(arg_num_valid), address_space=4)
            nv_i32 = _w(llvm.LoadOp(T.i32, _nv_ptr, alignment=4, invariant=True).result)
        else:
            nv_rsrc = buffer_ops.create_buffer_resource(
                arg_num_valid, max_size=False, num_records_bytes=fx.Index(4)
            )
            nv_i32 = buffer_ops.buffer_load(nv_rsrc, fx.Index(0), vec_width=1, dtype=T.i32)

        base_ptr = allocator.get_base()
        lds_x = SmemPtr(base_ptr, off0, T.f8, shape=(lds_a_bytes,)).get()
        lds_epi = SmemPtr(base_ptr, off0, T.bf16, shape=(max(tile_m * epi_stride, e2_groups * e2_gstride),)).get()
        lds_epi_i32 = SmemPtr(base_ptr, off0, T.i32, shape=(e2_groups * e2_gstride // 2,)).get()
        lds_sa = SmemPtr(base_ptr, off_sa, T.f32, shape=(n_nb * nkt * tile_m,)).get()

        s_stride = arith.index_cast(T.index, i32_s_stride)
        n_pairs = s_stride
        x_rsrc = buffer_ops.create_buffer_resource(
            arg_x, max_size=False, num_records_bytes=n_pairs * fx.Index(K)
        )
        sx_rsrc = buffer_ops.create_buffer_resource(
            arg_sx, max_size=False, num_records_bytes=n_pairs * fx.Index(nkb * 4)
        )
        w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)
        sw_rsrc = buffer_ops.create_buffer_resource(arg_sw, max_size=False)
        sorted_rsrc = buffer_ops.create_buffer_resource(arg_sorted_ids, max_size=False)
        sortw_rsrc = buffer_ops.create_buffer_resource(arg_sorted_w, max_size=False)
        eid_rsrc = buffer_ops.create_buffer_resource(
            arg_expert_ids, max_size=False, num_records_bytes=size_eids * fx.Index(4)
        )
        out_base = buffer_ops.extract_base_index(arg_out)
        out_rsrc = buffer_ops.create_buffer_resource(
            arg_out, max_size=False, num_records_bytes=tokens_in * fx.Index(hidden_dim * 2))
        mask24 = fx.Int32(0xFFFFFF)
        tokens_i32 = i32_tokens
        topk_i32 = fx.Int32(topk)

        layout_tx_wave_lane = fx.make_layout((num_waves, 64), stride=(64, 1))
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))
        coord_wl = fx.idx2crd(tx, layout_tx_wave_lane)
        wave_id = fx.get(coord_wl, 0)
        lane_id = fx.get(coord_wl, 1)
        coord_l16 = fx.idx2crd(lane_id, layout_lane16)
        lane_div_16 = fx.get(coord_l16, 0)
        lane_mod_16 = fx.get(coord_l16, 1)
        n_tile_base = wave_id * fx.Index(n_per_wave)

        shape_lds = fx.make_shape(tile_m, tile_k)
        stride_lds = fx.make_stride(tile_k, 1)
        layout_lds = fx.make_layout(shape_lds, stride_lds)
        k_blocks16 = fx.Index(tile_k // 16)
        vec16_x = T.vec(16, T.f8)
        tile_k_dwords = tile_k // 4
        layout_x_tile_div4 = fx.make_layout((tile_m, tile_k_dwords), stride=(tile_k_dwords, 1))
        tx_i32_base = tx * fx.Index(4)
        c_k_div4 = fx.Index(K // 4)
        col_offset_base = lane_div_16 * fx.Index(16)
        lane_b_off = lane_div_16 * fx.Index(256) + lane_mod_16 * fx.Index(16)
        sa_lane_off = lane_div_16 * fx.Index(4)
        if const_expr(fpre):
            sa_lane_off = sa_lane_off + (n_tile_base // fx.Index(128)) * fx.Index(nkt * tile_m)
        zero4 = arith.constant_vector(0.0, T.f32x4)

        def mfma(a, b, c):
            return rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [a, b, c, 0, 0, 0])

        def halves(v_i32x4):
            v64 = vector.bitcast(T.i64x2, v_i32x4)
            return (
                vector.extract(v64, static_position=[0], dynamic_position=[]),
                vector.extract(v64, static_position=[1], dynamic_position=[]),
            )

        def pair_of(fused):
            t = fused & mask24
            s = fused >> fx.Int32(24)
            ok = arith.cmpi(arith.CmpIPredicate.ult, t, tokens_i32) & arith.cmpi(
                arith.CmpIPredicate.ult, s, topk_i32)
            return t, ok, None

        def tile_body(bx, by):
            bx_m = bx * fx.Index(tile_m)
            by_n = by * fx.Index(tile_n)
            expert_i32 = buffer_ops.buffer_load(eid_rsrc, bx, vec_width=1, dtype=T.i32)
            expert_idx = arith.index_cast(T.index, expert_i32)
            cnt_i32 = None
            if const_expr(len(pad_variants) > 1):
                # valid rows of this M block (uniform: bx is the SGPR loop index) -> s_load_dword
                _cnt_ptr = buffer_ops.create_llvm_ptr(
                    buffer_ops.extract_base_index(arg_blk_cnt) + bx * fx.Index(4), address_space=4)
                cnt_i32 = _w(llvm.LoadOp(T.i32, _cnt_ptr, alignment=4, invariant=True).result)

            # ---- A2 scales [nkb, S] (kb-major, d0 stage-1 layout) -> LDS [kb][row], 4 rows/thread ----
            q_per_kb = tile_m // 4
            n_sa = nkb * q_per_kb
            for p_ in range_constexpr((n_sa + total_threads - 1) // total_threads):
                flat = tx + fx.Index(p_ * total_threads)
                kb_ = flat // fx.Index(q_per_kb)
                r4 = (flat % fx.Index(q_per_kb)) * fx.Index(4)
                v4 = buffer_ops.buffer_load(
                    sx_rsrc, kb_ * s_stride + bx_m + r4, vec_width=4, dtype=T.f32)
                if const_expr(fpre):
                    assert n_sa % total_threads == 0
                    if const_expr(epi_mode == 3):
                        tw4 = buffer_ops.buffer_load(sortw_rsrc, bx_m + r4, vec_width=4, dtype=T.f32)
                        v4 = v4 * tw4
                    nb0 = by_n // fx.Index(128)
                    swq = [buffer_ops.buffer_load(
                        sw_rsrc, (expert_idx * fx.Index(n_sblk) + nb0 + fx.Index(q)) * fx.Index(nkb) + kb_,
                        vec_width=1, dtype=T.f32) for q in range(n_nb)]
                    for q in range_constexpr(n_nb):
                        swv4 = vector.broadcast(T.f32x4, swq[q])
                        vector.store(v4 * swv4, lds_sa,
                                     [fx.Index(q * nkt * tile_m) + kb_ * fx.Index(tile_m) + r4])
                elif const_expr(n_sa % total_threads == 0):
                    vector.store(v4, lds_sa, [kb_ * fx.Index(tile_m) + r4])
                else:
                    in_r = arith.cmpi(arith.CmpIPredicate.ult, arith.index_cast(T.i32, flat), fx.Int32(n_sa))
                    with _IfThen(in_r):
                        vector.store(v4, lds_sa, [kb_ * fx.Index(tile_m) + r4])

            x_row_base_div4 = []
            x_row_local = []
            x_col_local_i32 = []
            for i in range_constexpr(num_x_loads):
                row_local, col_local_i32 = tile_chunk_coord_i32(
                    arith,
                    tx_i32_base=tx_i32_base,
                    i=i,
                    total_threads=total_threads,
                    layout_tile_div4=layout_x_tile_div4,
                    chunk_i32=4,
                )
                x_row_local.append(row_local)
                x_col_local_i32.append(col_local_i32)
                x_row_base_div4.append((bx_m + row_local) * c_k_div4)

            x_off_div4 = [x_row_base_div4[i] + x_col_local_i32[i] for i in range(num_x_loads)]

            def load_x_one(kt, i):
                if const_expr(bsoff):
                    return buffer_ops.buffer_load(x_rsrc, x_off_div4[i], vec_width=4, dtype=T.i32,
                                                  cache_modifier=xcm,
                                                  soffset_bytes=(kt if dbg != 5 else 0) * tile_k)
                idx = x_row_base_div4[i] + fx.Index((kt if dbg != 5 else 0) * tile_k_dwords) + x_col_local_i32[i]
                return buffer_ops.buffer_load(x_rsrc, idx, vec_width=4, dtype=T.i32, cache_modifier=xcm)

            def store_x_tile(parts, stage):
                for i in range_constexpr(num_x_loads):
                    lds_store_16b_xor16(
                        arith,
                        vector,
                        lds_memref=lds_x,
                        vec16_ty=vec16_x,
                        layout_lds=layout_lds,
                        row_local=x_row_local[i],
                        col_local_i32=x_col_local_i32[i],
                        tx_c4=fx.Index(4),
                        k_blocks16=k_blocks16,
                        lds_base=fx.Index(stage * lds_tile),
                        vec_part_i32x4=parts[i],
                        elem_bytes=1,
                    )

            def lds_load_a(mi, ku, stage):
                row = lane_mod_16 + fx.Index(mi * 16)
                col = col_offset_base + fx.Index(ku * 64)
                col_swz = swizzle_xor16(row, col, k_blocks16)
                idx = crd2idx((row, col_swz), layout_lds) + fx.Index(stage * lds_tile)
                v = vector.load_op(vec16_x, lds_x, [idx])
                return halves(vector.bitcast(T.i32x4, v))

            def load_a_mi(mi, stage):
                return [lds_load_a(mi, ku, stage) for ku in range(k_unroll)]

            def load_sa(kt, mi):
                return vector.load_op(
                    T.f32x4, lds_sa, [fx.Index(kt * tile_m + mi * 16) + sa_lane_off]
                )

            row_e = expert_idx * fx.Index(hidden_dim)
            b_base = []
            for ni in range_constexpr(num_acc_n):
                rg = row_e + by_n + n_tile_base + fx.Index(ni * 16)
                b_base.append((rg // fx.Index(16)) * fx.Index(K * 16) + lane_b_off)
            nb_w = (by_n + n_tile_base) // fx.Index(128)
            sw_base = (expert_idx * fx.Index(n_sblk) + nb_w) * fx.Index(nkb)

            b_base0_div4 = b_base[0] // fx.Index(4)

            def load_b_one(kt, ku, ni):
                if const_expr(bsoff):
                    so = ni * 16 * K + ((kt if dbg != 4 else 0) * k_unroll + ku) * 1024
                    return buffer_ops.buffer_load(w_rsrc, b_base0_div4, vec_width=4, dtype=T.i32,
                                                  cache_modifier=bcm, soffset_bytes=so)
                koff = fx.Index(((kt if dbg != 4 else 0) * k_unroll + ku) * 1024)
                return buffer_ops.buffer_load(
                    w_rsrc, (b_base[ni] + koff) // fx.Index(4), vec_width=4, dtype=T.i32, cache_modifier=bcm
                )

            def load_sw(kt):
                return buffer_ops.buffer_load(sw_rsrc, sw_base + fx.Index(kt), vec_width=1, dtype=T.f32)

            x0 = [load_x_one(0, i) for i in range(num_x_loads)]
            b_la = [2 if ku < b2ku else 1 for ku in range(k_unroll)]
            bbuf0 = {}
            for ku in range_constexpr(k_unroll):
                for t_ in range_constexpr(min(b_la[ku], nkt)):
                    bbuf0[(t_, ku)] = [load_b_one(t_, ku, ni) for ni in range(num_acc_n)]
            sw_cur0 = load_sw(0) if not fpre else None
            x1 = None
            if const_expr(nkt > 1):
                x1 = [load_x_one(1, i) for i in range(num_x_loads)]
            store_x_tile(x0, 0)
            gpu.barrier()
            if const_expr(prio == 1):
                rocdl.s_setprio(1)
            a_first0 = load_a_mi(0, 0)
            s_first0 = load_sa(0, 0)
            rocdl.sched_barrier(0)

            def kbody(m_eff):
                # K loop + epilogue over the first m_eff 16-row subtiles of the M block (m_eff == m_repeat: the
                # unchanged full-block body). Rows >= 16*m_eff must be routing padding (sorted_w 0, dummy rows).
                acc = [zero4] * (m_repeat * num_acc_n)
                bbuf = dict(bbuf0)
                sw_cur = sw_cur0
                a_first = a_first0
                s_first = s_first0
                x_pend = x1
                half_e = half if m_eff == m_repeat else max(1, m_eff // 2)
                ep_tw = [None] * (m_repeat * 4)
                ep_ids = [None] * (tile_m // num_waves)
                ep_ids2 = [None] * (2 * e2_gpw * 4)

                def preload_epilogue():
                    for mi in range_constexpr(m_repeat if epi_mode != 3 else 0):
                        for ii in range_constexpr(4):
                            r = bx_m + fx.Index(mi * 16 + ii) + lane_div_16 * fx.Index(4)
                            ep_tw[mi * 4 + ii] = buffer_ops.buffer_load(sortw_rsrc, r, vec_width=1, dtype=T.f32)
                    if const_expr((epi_mode == 2 and _E2_PRE) or epi_mode == 3):
                        for h in range_constexpr(2):
                            for gi in range_constexpr(e2_gpw):
                                for ii in range_constexpr(4):
                                    r = bx_m + fx.Index(h * e2_rows + ii) + (wave_id * fx.Index(e2_gpw) + fx.Index(gi)) * fx.Index(4)
                                    ep_ids2[(h * e2_gpw + gi) * 4 + ii] = buffer_ops.buffer_load(
                                        sorted_rsrc, r, vec_width=1, dtype=T.i32)
                    for p_ in range_constexpr(tile_m // num_waves if epi_mode not in (2, 3) else 0):
                        ep_ids[p_] = buffer_ops.buffer_load(
                            sorted_rsrc, bx_m + wave_id + fx.Index(p_ * num_waves), vec_width=1, dtype=T.i32
                        )

                for kt in range_constexpr(nkt):
                    if const_expr(kt == nkt - 1):
                        preload_epilogue()
                    stage = kt % n_stages
                    has_next = kt + 1 < nkt
                    has_next2 = kt + 2 < nkt
                    swv = sw_cur
                    bh = [[halves(bbuf[(kt, ku)][ni]) for ni in range(num_acc_n)] for ku in range(k_unroll)]
                    vm = []
                    vm += [("b", ku, ni) for ku in range(k_unroll) for ni in range(num_acc_n) if kt + b_la[ku] < nkt]
                    if const_expr(has_next and not fpre):
                        vm += [("s", 0, 0)]
                    if const_expr(has_next2):
                        vm += [("x", i, 0) for i in range(num_x_loads)]
                    per_group = (len(vm) + m_eff - 1) // m_eff
                    vm_groups = [vm[g * per_group:(g + 1) * per_group] for g in range(m_eff)]
                    if const_expr(vm_mode == 1 and m_eff == m_repeat):
                        vb = [t for t in vm if t[0] != "x"]
                        vx = [t for t in vm if t[0] == "x"]
                        hb = (len(vb) + 1) // 2
                        vm_groups = [vb[:hb], vb[hb:], vx[: (len(vx) + 1) // 2], vx[(len(vx) + 1) // 2:]] + [[]] * (m_repeat - 4)
                    elif const_expr(vm_mode == 2 and m_eff == m_repeat):
                        vb = [t for t in vm if t[0] != "x"]
                        vx = [t for t in vm if t[0] == "x"]
                        vm_groups = [vb, vx] + [[]] * (m_repeat - 2)
                    elif const_expr(vm_mode == 3 and m_eff == m_repeat):
                        vb = [t for t in vm if t[0] != "x"]
                        vx = [t for t in vm if t[0] == "x"]
                        vm_groups = [vx + vb[:3], vb[3:6], vb[6:]] + [[]] * (m_repeat - 3)
                    x_new = [None] * num_x_loads
                    for ku in range_constexpr(k_unroll):
                        if const_expr(kt + b_la[ku] < nkt):
                            bbuf[(kt + b_la[ku], ku)] = [None] * num_acc_n
                    sw_new = [None]

                    def issue(th):
                        kind, ku, ni = th
                        if const_expr(kind == "x"):
                            x_new[ku] = load_x_one(kt + 2, ku)
                        elif const_expr(kind == "s"):
                            sw_new[0] = load_sw(kt + 1)
                        else:
                            bbuf[(kt + b_la[ku], ku)][ni] = load_b_one(kt + b_la[ku], ku, ni)

                    a_cur = a_first
                    s_cur = s_first
                    prev = None
                    a_nf = None
                    s_nf = None
                    for mi in range_constexpr(m_eff):
                        n_ds = 0
                        if const_expr(mi + 1 < m_eff):
                            a_nxt = load_a_mi(mi + 1, stage)
                            s_nxt = load_sa(kt, mi + 1)
                            n_ds = k_unroll + 1
                        elif const_expr(has_next and m_eff > 1):
                            a_nf = load_a_mi(0, (kt + 1) % n_stages)
                            s_nf = load_sa(kt + 1, 0)
                            n_ds = k_unroll + 1
                        for th in vm_groups[mi]:
                            issue(th)
                        n_vm = len(vm_groups[mi])
                        tp = [zero4] * num_acc_n
                        if const_expr(dbg == 6):
                            tp = [acc[mi * num_acc_n + ni] for ni in range(num_acc_n)]
                        for ku in range_constexpr(k_unroll):
                            a0, a1 = a_cur[ku]
                            for ni in range_constexpr(num_acc_n):
                                tp[ni] = mfma(a0, bh[ku][ni][0], tp[ni])
                            for ni in range_constexpr(num_acc_n):
                                tp[ni] = mfma(a1, bh[ku][ni][1], tp[ni])
                        n_valu = 0
                        if const_expr(dbg == 6):
                            for ni in range_constexpr(num_acc_n):
                                acc[mi * num_acc_n + ni] = tp[ni]
                        elif const_expr(prev is not None):
                            ptp, ps, pmi = prev
                            f = ps * swv if not fpre else ps
                            for ni in range_constexpr(num_acc_n):
                                idx = pmi * num_acc_n + ni
                                acc[idx] = fmath.fma(ptp[ni], f, acc[idx])
                            n_valu = (1 if not fpre else 0) + num_acc_n
                        if const_expr(dbg != 6):
                            prev = (tp, s_cur, mi)
                        if const_expr(sched >= 1):
                            n_mfma = k_unroll * 2 * num_acc_n
                            rem_ds, rem_vm, rem_va = n_ds, n_vm, n_valu * 2
                            for si in range_constexpr(n_mfma):
                                if const_expr(rem_ds > 0):
                                    rocdl.sched_dsrd(1)
                                    rem_ds -= 1
                                elif const_expr(rem_vm > 0):
                                    rocdl.sched_vmem(1)
                                    rem_vm -= 1
                                rocdl.sched_mfma(1)
                                if const_expr(rem_va > 0 and si >= va_st):
                                    nv = min(va_nv, rem_va)
                                    rocdl.sched_group_barrier(0x002, nv, 0)
                                    rem_va -= nv
                            rocdl.sched_barrier(0)
                        if const_expr(mi == half_e - 1 and has_next):
                            store_x_tile(x_pend, (kt + 1) % n_stages)
                            gpu.barrier()
                            rocdl.sched_barrier(0)
                            if const_expr(m_eff == 1):
                                # single active subtile: the next K tile's first A/scale reads must follow
                                # the barrier that publishes its LDS stage
                                a_nf = load_a_mi(0, (kt + 1) % n_stages)
                                s_nf = load_sa(kt + 1, 0)
                                rocdl.sched_barrier(0)
                        if const_expr(mi + 1 < m_eff):
                            a_cur = a_nxt
                            s_cur = s_nxt
                    if const_expr(dbg != 6):
                        ptp, ps, pmi = prev
                        f = ps * swv if not fpre else ps
                        for ni in range_constexpr(num_acc_n):
                            idx = pmi * num_acc_n + ni
                            acc[idx] = fmath.fma(ptp[ni], f, acc[idx])
                    if const_expr(has_next):
                        sw_cur = sw_new[0]
                        a_first = a_nf
                        s_first = s_nf
                        x_pend = x_new if has_next2 else None
                if const_expr(prio == 1):
                    rocdl.s_setprio(0)

                # ---- epilogue: (acc * w_route) -> bf16 LDS (128-col chunks) -> pk_add_bf16 atomics ----
                tw = ep_tw
                lanes_per_row = epi_cols // 8
                rows_per_pass = total_threads // lanes_per_row
                e_r = tx // fx.Index(lanes_per_row)
                e_c = (tx % fx.Index(lanes_per_row)) * fx.Index(8)
                chunk_of_wave = wave_id // fx.Index(waves_per_chunk)
                col_in_chunk = (wave_id % fx.Index(waves_per_chunk)) * fx.Index(n_per_wave)
                if const_expr(dbg == 2):
                    ssum = acc[0]
                    for q in range_constexpr(1, m_repeat * num_acc_n):
                        ssum = ssum + acc[q]
                    sv = vector.extract(ssum, static_position=[0], dynamic_position=[])
                    svb = arith.trunc_f(T.bf16, sv)
                    v2d = vector.from_elements(T.vec(2, T.bf16), [svb, svb])
                    ptr = buffer_ops.create_llvm_ptr(out_base + tx * fx.Index(4), address_space=1)
                    llvm.StoreOp(ptr._value if hasattr(ptr, "_value") else ptr, v2d._value if hasattr(v2d, "_value") else v2d, alignment=4) if False else llvm.StoreOp(v2d._value if hasattr(v2d, "_value") else v2d, ptr._value if hasattr(ptr, "_value") else ptr, alignment=4)
                if const_expr(epi_mode == 3 and dbg != 2):
                    packed = []
                    for mi in range_constexpr(m_repeat):
                        for ni in range_constexpr(num_acc_n):
                            a4 = acc[mi * num_acc_n + ni]
                            ws = [_bf16_bits_rne(vector.extract(a4, static_position=[ii], dynamic_position=[]))
                                  for ii in range(4)]
                            if const_expr(e3_perm):
                                w01 = _perm(ws[1], ws[0], 0x07060302)
                                w23 = _perm(ws[3], ws[2], 0x07060302)
                            else:
                                w01 = (ws[0] >> fx.Int32(16)) | (ws[1] & fx.Int32(-65536))
                                w23 = (ws[2] >> fx.Int32(16)) | (ws[3] & fx.Int32(-65536))
                            packed.append(vector.from_elements(T.vec(2, T.i32), [w01, w23]))
                    lane_off = [arith.index_cast(T.i32, (fx.Index(hc * 128) + lane_id * fx.Index(2)) * fx.Index(2))
                                for hc in range(e2_nhc)]
                    for h in range_constexpr(2 if 2 * m_eff > m_repeat else 1):
                        gpu.barrier()
                        for mh in range_constexpr(m_repeat // 2):
                            mi = h * (m_repeat // 2) + mh
                            grp = fx.Index(mh * 4) + lane_div_16
                            for ni in range_constexpr(num_acc_n):
                                col = n_tile_base + fx.Index(ni * 16) + lane_mod_16
                                vector.store(packed[mi * num_acc_n + ni], lds_epi_i32,
                                             [grp * fx.Index(e2_gstride // 2) + col * fx.Index(2)], alignment=8)
                        gpu.barrier()
                        v8s = []
                        for u in range_constexpr(e2_upw):
                            gi = u // e2_nhc
                            hc = u % e2_nhc
                            g = wave_id * fx.Index(e2_gpw) + fx.Index(gi)
                            c = fx.Index(hc * 128) + lane_id * fx.Index(2)
                            v8s.append(vector.load_op(T.vec(4, T.i32), lds_epi_i32,
                                                      [g * fx.Index(e2_gstride // 2) + c * fx.Index(2)]))
                        for u in range_constexpr(e2_upw):
                            gi = u // e2_nhc
                            hc = u % e2_nhc
                            for ii in range_constexpr(4):
                                idv = _w(_rocdl_d.readfirstlane(T.i32, arith.unwrap(ep_ids2[(h * e2_gpw + gi) * 4 + ii])))
                                t2, t_ok, _ = pair_of(idv)
                                wa = _w(vector.extract(v8s[u], static_position=[ii // 2], dynamic_position=[]))
                                wb = _w(vector.extract(v8s[u], static_position=[2 + ii // 2], dynamic_position=[]))
                                if const_expr(e3_perm):
                                    fw = _perm(wb, wa, 0x05040100 if ii % 2 == 0 else 0x07060302)
                                elif const_expr(ii % 2 == 0):
                                    fw = (wa & fx.Int32(65535)) | (wb << fx.Int32(16))
                                else:
                                    fw = (wa >> fx.Int32(16)) & fx.Int32(65535) | (wb & fx.Int32(-65536))
                                if const_expr(e3_bl):
                                    pos = bx_m + fx.Index(h * e2_rows + ii) + (wave_id * fx.Index(e2_gpw) + fx.Index(gi)) * fx.Index(4)
                                    pos_i32 = arith.index_cast(T.i32, pos)
                                    pos_ok = arith.cmpi(arith.CmpIPredicate.ult, pos_i32, tokens_i32)
                                    dummy = arith.select(pos_ok, pos_i32, fx.Int32(0))
                                    trow = arith.select(t_ok, t2, dummy)
                                    fw = arith.select(t_ok, fw, fx.Int32(0))
                                else:
                                    trow = t2
                                frag = vector.bitcast(T.vec(2, T.bf16), vector.from_elements(T.vec(1, T.i32), [fw]))
                                frag_v = frag._value if hasattr(frag, "_value") else frag

                                def _emit(trow=trow, frag_v=frag_v, hc=hc):
                                    if const_expr(e3_saddr):
                                        row_b = out_base + (arith.index_cast(T.index, trow) * fx.Index(hidden_dim) + by_n) * fx.Index(2)
                                        row_i64 = arith.index_cast(T.i64, row_b)
                                        addr = _w(row_i64) + _w(arith.extui(T.i64, arith.unwrap(lane_off[hc])))
                                        ptr = buffer_ops.create_llvm_ptr(arith.index_cast(T.index, addr), address_space=1)
                                    else:
                                        c = fx.Index(hc * 128) + lane_id * fx.Index(2)
                                        elem = arith.index_cast(T.index, trow) * fx.Index(hidden_dim) + by_n + c
                                        ptr = buffer_ops.create_llvm_ptr(out_base + elem * fx.Index(2), address_space=1)
                                    ptr_v = ptr._value if hasattr(ptr, "_value") else ptr
                                    llvm.AtomicRMWOp(llvm.AtomicBinOp.fadd, ptr_v, frag_v,
                                                     llvm.AtomicOrdering.monotonic, syncscope="agent", alignment=4)
                                if const_expr(e3_bl):
                                    _emit()
                                else:
                                    with _IfThen(t_ok):
                                        _emit()
                if const_expr(epi_mode == 2 and dbg != 2):
                    packed = []
                    for mi in range_constexpr(m_repeat):
                        for ni in range_constexpr(num_acc_n):
                            a4 = acc[mi * num_acc_n + ni]
                            ws = []
                            for ii in range_constexpr(4):
                                v = vector.extract(a4, static_position=[ii], dynamic_position=[])
                                ws.append(_bf16_bits_rne(v * tw[mi * 4 + ii]))
                            w01 = (ws[0] >> fx.Int32(16)) | (ws[1] & fx.Int32(-65536))
                            w23 = (ws[2] >> fx.Int32(16)) | (ws[3] & fx.Int32(-65536))
                            packed.append(vector.from_elements(T.vec(2, T.i32), [w01, w23]))
                    for h in range_constexpr(1 if _E2_DBG == 2 else 2):
                        gpu.barrier()
                        if const_expr(not _E2_PRE):
                            for gi in range_constexpr(e2_gpw):
                                for ii in range_constexpr(4):
                                    r = bx_m + fx.Index(h * e2_rows + ii) + (wave_id * fx.Index(e2_gpw) + fx.Index(gi)) * fx.Index(4)
                                    ep_ids2[(h * e2_gpw + gi) * 4 + ii] = buffer_ops.buffer_load(
                                        sorted_rsrc, r, vec_width=1, dtype=T.i32)
                        for mh in range_constexpr(m_repeat // 2):
                            mi = h * (m_repeat // 2) + mh
                            grp = fx.Index(mh * 4) + lane_div_16
                            for ni in range_constexpr(num_acc_n):
                                col = n_tile_base + fx.Index(ni * 16) + lane_mod_16
                                vector.store(packed[mi * num_acc_n + ni], lds_epi_i32,
                                             [grp * fx.Index(e2_gstride // 2) + col * fx.Index(2)], alignment=8)
                        gpu.barrier()
                        v8s = []
                        for u in range_constexpr(e2_upw if _E2_DBG != 1 else 0):
                            gi = u // e2_nhc
                            hc = u % e2_nhc
                            g = wave_id * fx.Index(e2_gpw) + fx.Index(gi)
                            c = fx.Index(hc * 128) + lane_id * fx.Index(2)
                            v8s.append(vector.load_op(T.vec(4, T.i32), lds_epi_i32,
                                                      [g * fx.Index(e2_gstride // 2) + c * fx.Index(2)]))
                        for u in range_constexpr(e2_upw if _E2_DBG != 1 else 0):
                            gi = u // e2_nhc
                            hc = u % e2_nhc
                            c = fx.Index(hc * 128) + lane_id * fx.Index(2)
                            for ii in range_constexpr(4):
                                idv = ep_ids2[(h * e2_gpw + gi) * 4 + ii]
                                if const_expr(_E2_RFL):
                                    idv = _w(_rocdl_d.readfirstlane(T.i32, arith.unwrap(idv)))
                                t2, t_ok, _ = pair_of(idv)
                                wa = _w(vector.extract(v8s[u], static_position=[ii // 2], dynamic_position=[]))
                                wb = _w(vector.extract(v8s[u], static_position=[2 + ii // 2], dynamic_position=[]))
                                if const_expr(ii % 2 == 0):
                                    fw = (wa & fx.Int32(65535)) | (wb << fx.Int32(16))
                                else:
                                    fw = (wa >> fx.Int32(16)) & fx.Int32(65535) | (wb & fx.Int32(-65536))
                                frag = vector.bitcast(T.vec(2, T.bf16), vector.from_elements(T.vec(1, T.i32), [fw]))
                                if const_expr(_E2_BRANCHLESS):
                                    pos = bx_m + fx.Index(h * e2_rows + ii) + (wave_id * fx.Index(e2_gpw) + fx.Index(gi)) * fx.Index(4)
                                    pos_i32 = arith.index_cast(T.i32, pos)
                                    pos_ok = arith.cmpi(arith.CmpIPredicate.ult, pos_i32, tokens_i32)
                                    dummy = arith.select(pos_ok, pos_i32, fx.Int32(0))
                                    trow = arith.select(t_ok, t2, dummy)
                                    fw = arith.select(t_ok, fw, fx.Int32(0))
                                    frag = vector.bitcast(T.vec(2, T.bf16), vector.from_elements(T.vec(1, T.i32), [fw]))
                                    elem = arith.index_cast(T.index, trow) * fx.Index(hidden_dim) + by_n + c
                                    ptr = buffer_ops.create_llvm_ptr(out_base + elem * fx.Index(2), address_space=1)
                                    ptr_v = ptr._value if hasattr(ptr, "_value") else ptr
                                    frag_v = frag._value if hasattr(frag, "_value") else frag
                                    if const_expr(dbg == 1):
                                        llvm.StoreOp(frag_v, ptr_v, alignment=4)
                                    elif const_expr(dbg == 3):
                                        pass
                                    else:
                                        llvm.AtomicRMWOp(llvm.AtomicBinOp.fadd, ptr_v, frag_v,
                                                         llvm.AtomicOrdering.monotonic, syncscope="agent", alignment=4)
                                else:
                                  with _IfThen(t_ok):
                                    elem = arith.index_cast(T.index, t2) * fx.Index(hidden_dim) + by_n + c
                                    ptr = buffer_ops.create_llvm_ptr(out_base + elem * fx.Index(2), address_space=1)
                                    ptr_v = ptr._value if hasattr(ptr, "_value") else ptr
                                    frag_v = frag._value if hasattr(frag, "_value") else frag
                                    llvm.AtomicRMWOp(llvm.AtomicBinOp.fadd, ptr_v, frag_v,
                                                     llvm.AtomicOrdering.monotonic, syncscope="agent", alignment=4)
                for ch in range_constexpr(n_chunks if (dbg != 2 and epi_mode not in (2, 3)) else 0):
                    gpu.barrier()
                    mine = arith.cmpi(
                        arith.CmpIPredicate.eq,
                        arith.index_cast(T.i32, chunk_of_wave),
                        fx.Int32(ch),
                    )
                    with _IfThen(mine):
                        for mi in range_constexpr(m_repeat):
                            for ii in range_constexpr(4):
                                row = fx.Index(mi * 16 + ii) + lane_div_16 * fx.Index(4)
                                for ni in range_constexpr(num_acc_n):
                                    v = vector.extract(
                                        acc[mi * num_acc_n + ni], static_position=[ii], dynamic_position=[]
                                    )
                                    y = arith.trunc_f(T.bf16, v * tw[mi * 4 + ii])
                                    col = col_in_chunk + fx.Index(ni * 16) + lane_mod_16
                                    vector.store(
                                        vector.from_elements(T.vec(1, T.bf16), [y]),
                                        lds_epi,
                                        [row * fx.Index(epi_stride) + col],
                                    )
                    gpu.barrier()
                    if const_expr(epi_mode == 0):
                        # lane = 2 bf16, one wave instruction = one row's 128 cols (256 B contiguous).
                        # All LDS reads first, then the guarded atomics (branches would serialize them).
                        n_pass2 = tile_m // num_waves
                        c2 = lane_id * fx.Index(2)
                        v2s = [
                            vector.load_op(
                                T.vec(2, T.bf16),
                                lds_epi,
                                [(wave_id + fx.Index(p_ * num_waves)) * fx.Index(epi_stride) + c2],
                            )
                            for p_ in range_constexpr(n_pass2)
                        ]
                        for p_ in range_constexpr(n_pass2):
                            t2, t_ok, _ = pair_of(ep_ids[p_])
                            with _IfThen(t_ok):
                                col_g = by_n + fx.Index(ch * epi_cols) + c2
                                elem = arith.index_cast(T.index, t2) * fx.Index(hidden_dim) + col_g
                                addr = out_base + elem * fx.Index(2)
                                ptr = buffer_ops.create_llvm_ptr(addr, address_space=1)
                                ptr_v = ptr._value if hasattr(ptr, "_value") else ptr
                                v2 = v2s[p_]
                                frag_v = v2._value if hasattr(v2, "_value") else v2
                                if const_expr(dbg == 1):
                                    llvm.StoreOp(frag_v, ptr_v, alignment=4)
                                else:
                                    llvm.AtomicRMWOp(
                                        llvm.AtomicBinOp.fadd,
                                        ptr_v,
                                        frag_v,
                                        llvm.AtomicOrdering.monotonic,
                                        syncscope="agent",
                                        alignment=4,
                                    )
                    else:
                        for p_ in range_constexpr(tile_m // rows_per_pass):
                            r_loc = e_r + fx.Index(p_ * rows_per_pass)
                            v8 = vector.load_op(
                                T.vec(8, T.bf16), lds_epi, [r_loc * fx.Index(epi_stride) + e_c]
                            )
                            fused = buffer_ops.buffer_load(sorted_rsrc, bx_m + r_loc, vec_width=1, dtype=T.i32)
                            t2, t_ok, _ = pair_of(fused)
                            with _IfThen(t_ok):
                                col_g = by_n + fx.Index(ch * epi_cols) + e_c
                                elem = arith.index_cast(T.index, t2) * fx.Index(hidden_dim) + col_g
                                for j in range_constexpr(4):
                                    e0 = vector.extract(v8, static_position=[2 * j], dynamic_position=[])
                                    e1 = vector.extract(v8, static_position=[2 * j + 1], dynamic_position=[])
                                    frag = vector.from_elements(T.vec(2, T.bf16), [e0, e1])
                                    addr = out_base + (elem + fx.Index(2 * j)) * fx.Index(2)
                                    ptr = buffer_ops.create_llvm_ptr(addr, address_space=1)
                                    ptr_v = ptr._value if hasattr(ptr, "_value") else ptr
                                    frag_v = frag._value if hasattr(frag, "_value") else frag
                                    llvm.AtomicRMWOp(
                                        llvm.AtomicBinOp.fadd,
                                        ptr_v,
                                        frag_v,
                                        llvm.AtomicOrdering.monotonic,
                                        syncscope="agent",
                                        alignment=4,
                                    )
            if const_expr(len(pad_variants) > 1):
                _variant_chain(cnt_i32, pad_variants, kbody)
            else:
                kbody(m_repeat)

        by0 = gpu.block_id("x")
        bx0 = gpu.block_id("y")
        if const_expr(deloop == 0):
            nv_idx = arith.index_cast(T.index, nv_i32)
            n_mblk = (nv_idx + fx.Index(tile_m - 1)) // fx.Index(tile_m)
            gm = arith.index_cast(T.index, i32_grid_m)
            for mb in range(bx0, n_mblk, gm):
                gpu.barrier()
                tile_body(mb, by0)
        elif const_expr(xcd_gn == 0):
            # plain (N, M) grid, one tile per CTA; grid M = guaranteed bound, empty CTAs exit
            ok = arith.cmpi(arith.CmpIPredicate.ult, arith.index_cast(T.i32, bx0 * fx.Index(tile_m)), nv_i32)
            with _IfThen(ok):
                tile_body(bx0, by0)
        else:
            # XCD tiling: XCD x = lin % 8 owns N group g (gn tiles) x M partition p (contiguous, from n_mblk)
            n_grp = n_tiles // xcd_gn
            n_part = 8 // n_grp
            n_mblk_i = (_w(nv_i32) + fx.Int32(tile_m - 1)) // fx.Int32(tile_m)
            lin = _w(arith.index_cast(T.i32, bx0)) * fx.Int32(n_tiles) + _w(arith.index_cast(T.i32, by0))
            xcd = lin % fx.Int32(8)
            j = lin // fx.Int32(8)
            g = xcd % fx.Int32(n_grp)
            p = xcd // fx.Int32(n_grp)
            m0 = (p * n_mblk_i) // fx.Int32(n_part)
            m1 = ((p + fx.Int32(1)) * n_mblk_i) // fx.Int32(n_part)
            if const_expr(xcd_ord == 0):
                m = m0 + j // fx.Int32(xcd_gn)
                n = g * fx.Int32(xcd_gn) + j % fx.Int32(xcd_gn)
            else:
                cs = xcd_c * xcd_gn
                c = j // fx.Int32(cs)
                r = j % fx.Int32(cs)
                m = m0 + c * fx.Int32(xcd_c) + r % fx.Int32(xcd_c)
                n = g * fx.Int32(xcd_gn) + r // fx.Int32(xcd_c)
            ok = arith.cmpi(arith.CmpIPredicate.slt, m, m1)
            with _IfThen(ok):
                tile_body(arith.index_cast(T.index, m), arith.index_cast(T.index, n))

    @flyc.jit
    def launch(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_sx: fx.Tensor,
        arg_w: fx.Tensor,
        arg_sw: fx.Tensor,
        arg_sorted_ids: fx.Tensor,
        arg_sorted_w: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_num_valid: fx.Tensor,
        arg_blk_cnt: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_size_eids: fx.Int32,
        i32_s_stride: fx.Int32,
        i32_grid_m: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        gy = arith.index_cast(T.index, i32_grid_m)
        moe_s2_fp8bs(
            arg_out,
            arg_x,
            arg_sx,
            arg_w,
            arg_sw,
            arg_sorted_ids,
            arg_sorted_w,
            arg_expert_ids,
            arg_num_valid,
            arg_blk_cnt,
            i32_tokens,
            i32_size_eids,
            i32_s_stride,
            i32_grid_m,
        ).launch(grid=(n_tiles, gy, 1), block=(total_threads, 1, 1), stream=stream)
        if const_expr(wpe > 0):
            for op in ctx.gpu_module_body.operations:
                if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, int(wpe))

    return launch


def _xcd_map(lin, n_mblk, n_tiles, gn, ordr=0, cc=8):
    """Python mirror of the device CTA remap (for the bijection check). Returns (m, n) or None."""
    if gn == 0:
        m, n = lin // n_tiles, lin % n_tiles
        return (m, n) if m < n_mblk else None
    n_grp = n_tiles // gn
    n_part = 8 // n_grp
    xcd, j = lin % 8, lin // 8
    g, p = xcd % n_grp, xcd // n_grp
    m0, m1 = (p * n_mblk) // n_part, ((p + 1) * n_mblk) // n_part
    if ordr == 0:
        m, n = m0 + j // gn, g * gn + j % gn
    else:
        c, r = j // (cc * gn), j % (cc * gn)
        m, n = m0 + c * cc + r % cc, g * gn + r // cc
    return (m, n) if m < m1 else None


def _xcd_grid_m(blocks, n_tiles, gn, ordr=0, cc=8):
    if gn == 0:
        return max(1, blocks)
    n_part = 8 // (n_tiles // gn)
    # largest partition holds ceil(blocks / n_part) M blocks -> needs ceil(.)*gn slots on its XCD
    per_part = -(-blocks // n_part)
    if ordr == 1:
        per_part = -(-per_part // cc) * cc
    per_xcd = per_part * gn
    return max(1, -(-(per_xcd * 8) // n_tiles))


def stage2_fp8bs(out, a2, w2, a2_scale, w2_scale_flat, sorted_ids, sorted_expert_ids, sorted_weights,
                 num_valid_ids, tokens, S, blocks, stream, *, topk, tile_m=64, tile_n=256, grid_m=None,
                 sched=1, epi_mode=0, dbg=0, blk_cnt=None):
    """Same argument order as d0's compile_stage2 launcher. a2 [S, inter] fp8 (sorted rows),
    a2_scale [inter/128, S] fp32 (kb-major), w2 preshuffled [E, hidden, inter].
    blk_cnt: optional int32 [blocks] valid-row count per M block (moe_sort_fast blk_counts=True); enables the
    S2_PADSKIP all-padding-subtile skip. None (e.g. the CK sorting fallback) -> full-block body only."""
    E, hidden, inter = w2.shape
    deloop = _DELOOP or (_XCD_GN > 0)
    # fpre needs one a2-scale vec4 per thread (inter/128 * tile_m/4 == 256 lanes) and 128-col weight-scale blocks;
    # epi_mode 3 needs fpre (routing weight lives in its table). Other shapes fall back to the previous path.
    fpre = _FPRE if ((inter // 128) * (tile_m // 4) == 256 and tile_n % 128 == 0) else 0
    epi = epi_mode if _EPI < 0 else _EPI
    if epi == 3 and not fpre:
        epi = 2
    m_rep = tile_m // 16
    pad_vs = _parse_padskip(_PADSKIP, m_rep) if blk_cnt is not None else (m_rep,)
    if len(pad_vs) > 1 and (_VM_MODE != 0 or (_DBG or dbg) or epi != 3 or deloop or not fpre):
        pad_vs = (m_rep,)
    exe = compile_stage2_fp8bs(
        hidden_dim=hidden, inter_dim=inter, experts=E, topk=topk, tile_m=tile_m, tile_n=tile_n, sched=sched,
        epi_mode=epi, a_sorted=1, dbg=(_DBG or dbg), prio=_PRIO, deloop=int(deloop), xcd_gn=_XCD_GN,
        xcd_ord=_XCD_ORD, xcd_c=_XCD_C, bsoff=_BSOFF, fpre=fpre, wpe=(_WPE if fpre else 0), vm_mode=_VM_MODE,
        e3_bl=_E3_BL, e3_perm=_E3_PERM, e3_saddr=_E3_SADDR, va_nv=_VA_NV, va_st=_VA_ST,
        llvm_opts=tuple(sorted(_LLVM_OPTS.items())), bcm=_BCM, xcm=_XCM, b2ku=_B2_KU,
        padskip=pad_vs if len(pad_vs) > 1 else (),
    )
    cnt_arg = blk_cnt if len(pad_vs) > 1 else num_valid_ids  # unused dummy when the skip is off
    if deloop:
        # one tile per CTA: grid M must cover the guaranteed bound n_mblk <= blocks (+ remap slack)
        grid_m = _xcd_grid_m(blocks, hidden // tile_n, _XCD_GN, _XCD_ORD, _XCD_C)
        if _DBG_GRID_CAP:
            grid_m = min(grid_m, _DBG_GRID_CAP)
    else:
        if grid_m is None:
            exp_pairs = tokens * topk // 8  # balanced EP8 routing hint; kernel loops over extra M blocks
            grid_m = (exp_pairs + tile_m - 1) // tile_m + E
        grid_m = max(1, min(grid_m, blocks))
    if _LLVM_OPTS:
        with CompilationContext.compile_hints({"llvm_options": dict(_LLVM_OPTS)}):
            exe(out.view(-1), a2.view(-1), a2_scale.view(-1), w2.view(-1), w2_scale_flat.view(-1),
                sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cnt_arg, tokens, blocks, S, grid_m,
                stream)
        return out
    exe(out.view(-1), a2.view(-1), a2_scale.view(-1), w2.view(-1), w2_scale_flat.view(-1),
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cnt_arg, tokens, blocks, S, grid_m, stream)
    return out
