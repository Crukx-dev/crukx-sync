"""Triton kernel for the block decoder: one lane per 256-symbol block, all blocks in parallel.

Same algorithm and the same validation as codec._decode_blocks_fast (which stays the reference and
the CPU path): 64-bit window reads, unary run by count-leading-zeros (continued across windows for
long runs), Rice remainder, canonical Huffman by code length. Corrupt input only raises flags;
every load index is clamped.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .wire import BLOCK, MAX_CODE_LEN

LANES = 128


@triton.jit
def _read64(W, n_words, pos):
    """64 bits at bit `pos`, MSB first, as uint64 (indices clamped)."""
    w = pos >> 6
    w = tl.minimum(tl.maximum(w, 0), n_words - 2)
    s = (pos & 63).to(tl.uint64)
    hi = tl.load(W + w).to(tl.uint64)
    lo = tl.load(W + w + 1).to(tl.uint64)
    lo_part = tl.where(s == 0, tl.zeros_like(lo), lo >> (64 - s))
    return (hi << s) | lo_part


@triton.jit
def _decode_kernel(W, n_words, cur_p, end_p, count_p, base_p, rice_p, qmax_p, tbl_p,
                   first_p, ncode_p, symbase_p, syms_p, n_syms,
                   gaps_p, zz_p, flags_p, n_blocks,
                   NSTEP: tl.constexpr, MAXLEN: tl.constexpr, LANES: tl.constexpr):
    lane = tl.program_id(0) * LANES + tl.arange(0, LANES)
    live = lane < n_blocks
    cur = tl.load(cur_p + lane, mask=live, other=0)
    end = tl.load(end_p + lane, mask=live, other=0)
    count = tl.load(count_p + lane, mask=live, other=0)
    ebase = tl.load(base_p + lane, mask=live, other=0)
    rice = tl.load(rice_p + lane, mask=live, other=0)
    qmax = tl.load(qmax_p + lane, mask=live, other=0)
    tbl = tl.load(tbl_p + lane, mask=live, other=0)
    bad = tl.zeros([LANES], dtype=tl.int32)
    rice_u = rice.to(tl.uint64)
    for step in range(NSTEP):
        act = live & (step < count)
        bad = tl.where(act & (cur >= end), 1, bad)
        y = ~_read64(W, n_words, cur)
        q = libdevice.clz(y.to(tl.int64)).to(tl.int64)
        # unary runs longer than one window: keep reading windows (bounded by qmax: a run longer
        # than the group allows is corrupt and stops here)
        need = act & (y == 0) & (q <= qmax)
        while tl.max(need.to(tl.int32), axis=0) > 0:
            y = ~_read64(W, n_words, cur + q)
            q = tl.where(need, q + libdevice.clz(y.to(tl.int64)).to(tl.int64), q)
            need = need & (y == 0) & (q <= qmax)
        bad = tl.where(act & (q > qmax), 1, bad)
        c = cur + q + 1
        x = _read64(W, n_words, c)
        rem = tl.where(rice_u == 0, tl.zeros_like(x), x >> (64 - rice_u))
        peek = ((x << rice_u) >> (64 - MAXLEN)).to(tl.int64)
        found = tl.zeros([LANES], dtype=tl.int32)
        L = tl.zeros([LANES], dtype=tl.int64)
        sym_i = tl.zeros([LANES], dtype=tl.int64)
        for ln in tl.static_range(1, MAXLEN + 1):
            code = peek >> (MAXLEN - ln)
            fc = tl.load(first_p + tbl * (MAXLEN + 1) + ln, mask=act, other=0)
            nc = tl.load(ncode_p + tbl * (MAXLEN + 1) + ln, mask=act, other=0)
            sb = tl.load(symbase_p + tbl * (MAXLEN + 1) + ln, mask=act, other=0)
            hit = (found == 0) & (code >= fc) & (code < fc + nc)
            L = tl.where(hit, ln, L)
            sym_i = tl.where(hit, sb + code - fc, sym_i)
            found = tl.where(hit, 1, found)
        bad = tl.where(act & (found == 0), 1, bad)
        sym_i = tl.minimum(tl.maximum(sym_i, 0), n_syms - 1)
        sym = tl.load(syms_p + sym_i, mask=act, other=0)
        out = ebase + step
        gap = (tl.minimum(q, 1 << 32) << rice) | rem.to(tl.int64)
        tl.store(gaps_p + out, gap, mask=act)
        tl.store(zz_p + out, sym, mask=act)
        cur = tl.where(act, c + rice + L, cur)
    bad = tl.where(live & (cur != end), 1, bad)
    tl.store(flags_p + lane, bad, mask=live)


def decode_blocks(W: torch.Tensor, blocks, tabs, total: int):
    """(gaps, zz, bad, long_run) like codec._decode_blocks_fast, for CUDA tensors. Long unary runs
    are decoded in the kernel, so long_run is always False here."""
    cur, blk_end, blk_count, elem_base, rice, q_max, tbl = blocks
    first, ncode, base, syms, _ = tabs
    n = cur.numel()
    gaps = torch.zeros(total, dtype=torch.long, device=W.device)
    zz = torch.zeros(total, dtype=torch.long, device=W.device)
    flags = torch.zeros(n, dtype=torch.int32, device=W.device)
    grid = (triton.cdiv(n, LANES),)
    _decode_kernel[grid](W, W.numel(), cur.contiguous(), blk_end.contiguous(), blk_count.contiguous(),
                         elem_base.contiguous(), rice.contiguous(), q_max.contiguous(), tbl.contiguous(),
                         first.contiguous(), ncode.contiguous(), base.contiguous(), syms.contiguous(),
                         syms.numel(), gaps, zz, flags, n,
                         NSTEP=BLOCK, MAXLEN=MAX_CODE_LEN, LANES=LANES)
    return gaps, zz, flags.any(), torch.zeros((), dtype=torch.bool, device=W.device)
