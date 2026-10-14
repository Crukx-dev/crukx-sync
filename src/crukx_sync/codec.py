"""Encode and decode one tensor's update (docs/FORMAT.md). Inputs are uint16 bf16 bit patterns.

The encoder and decoder are vectorised with numpy across 256-element blocks; every block decodes
independently, which is the shape a GPU kernel will take.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from .wire import CorruptUpdate

BLOCK = 256
MAX_CODE_LEN = 17  # residuals span +-65535: <= 131071 zigzag symbols always fit
MAX_SYMBOL = 131070
MAX_RICE = 30
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


@dataclass
class Group:
    exponent: int
    count: int
    rice_r: int
    symbols: np.ndarray  # zigzag residual symbols (uint32)
    lengths: np.ndarray  # canonical Huffman code length per symbol


@dataclass
class TensorUpdate:
    numel: int
    checksum: int
    groups: list[Group]
    block_bits: np.ndarray  # bit length of every block, groups in order
    payload: bytes


# ---------------------------------------------------------------- shared helpers

def exponent_order(old: np.ndarray):
    """Element order by old bf16 exponent (stable) and each exponent's start in that order."""
    exps = ((old >> 7) & 0xFF).astype(np.uint8)
    order = np.argsort(exps, kind="stable")
    starts = np.zeros(257, dtype=np.int64)
    starts[1:] = np.cumsum(np.bincount(exps, minlength=256))
    return order, starts


def checksum(bits: np.ndarray, chunk: int = 1 << 22) -> int:
    """Position-weighted 64-bit sum of uint16 bits (see FORMAT.md, Integrity)."""
    total = np.uint64(0)
    with np.errstate(over="ignore"):
        for lo in range(0, bits.size, chunk):
            idx = np.arange(lo, min(lo + chunk, bits.size), dtype=np.uint64)
            w = (idx * _GOLDEN) | np.uint64(1)
            total += np.sum(bits[lo:lo + chunk].astype(np.uint64) * w, dtype=np.uint64)
    return int(total)


def _zigzag(v: np.ndarray) -> np.ndarray:
    return ((v << 1) ^ (v >> 31)).astype(np.uint32)


def _unzigzag(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.int64)
    return (z >> 1) ^ -(z & 1)


def _huffman_lengths(counts: np.ndarray) -> np.ndarray:
    """Huffman code lengths limited to MAX_CODE_LEN (flatten skewed counts, else equal lengths)."""
    counts = np.asarray(counts, dtype=np.int64)
    for _ in range(8):
        lengths = _huffman_lengths_unlimited(counts)
        if lengths.max() <= MAX_CODE_LEN:
            return lengths
        counts = np.maximum(1, counts >> 2)
    return np.full(len(counts), max(1, int(np.ceil(np.log2(len(counts))))), dtype=np.uint8)


def _huffman_lengths_unlimited(counts: np.ndarray) -> np.ndarray:
    n = len(counts)
    if n == 1:
        return np.ones(1, dtype=np.uint8)
    heap = [(int(c), i) for i, c in enumerate(counts)]
    heapq.heapify(heap)
    parent = [0] * (2 * n - 1)
    nxt = n
    while len(heap) > 1:
        c1, a = heapq.heappop(heap)
        c2, b = heapq.heappop(heap)
        parent[a] = parent[b] = nxt
        heapq.heappush(heap, (c1 + c2, nxt))
        nxt += 1
    depth = [0] * (2 * n - 1)  # root is the last node; parents always have larger ids
    for node in range(2 * n - 3, -1, -1):
        depth[node] = depth[parent[node]] + 1
    return np.array(depth[:n], dtype=np.int64)


def _canonical_codes(lengths: np.ndarray) -> np.ndarray:
    """Canonical Huffman codes for lengths, in the given symbol order (symbols sorted ascending)."""
    order = np.lexsort((np.arange(len(lengths)), lengths))
    codes = np.zeros(len(lengths), dtype=np.int64)
    code, prev = 0, int(lengths[order[0]])
    for k, i in enumerate(order):
        L = int(lengths[i])
        if k:
            code = (code + 1) << (L - prev)
        codes[i] = code
        prev = L
    return codes


def _check_kraft(lengths: np.ndarray) -> None:
    if len(lengths) == 0 or lengths.min() < 1 or lengths.max() > MAX_CODE_LEN:
        raise CorruptUpdate("bad Huffman code length")
    if np.sum(2.0 ** -lengths.astype(np.float64)) > 1.0 + 1e-12:
        raise CorruptUpdate("Huffman code lengths overflow the code space")


# ---------------------------------------------------------------- encoder

def encode_tensor(old: np.ndarray, new: np.ndarray) -> TensorUpdate | None:
    """Update turning `old` into `new` (flat uint16 arrays), or None if they are identical."""
    changed = old != new
    if not changed.any():
        return None
    order, starts = exponent_order(old)
    pos = np.flatnonzero(changed[order])
    exps = ((old[order[pos]] >> 7) & 0xFF).astype(np.int64)
    local = pos - starts[exps]
    first = np.r_[True, exps[1:] != exps[:-1]]
    prev_local = np.r_[-1, local[:-1]]
    prev_local[first] = -1
    gaps = (local - prev_local - 1).astype(np.int64)
    res = new[order[pos]].view(np.int16).astype(np.int64) - old[order[pos]].view(np.int16).astype(np.int64)
    zz = _zigzag(res)

    groups: list[Group] = []
    rice = np.empty(len(pos), dtype=np.int64)
    hcode = np.empty(len(pos), dtype=np.int64)
    hlen = np.empty(len(pos), dtype=np.int64)
    block_bits: list[np.ndarray] = []
    cuts = np.flatnonzero(np.r_[True, exps[1:] != exps[:-1], True])
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        g = gaps[lo:hi]
        costs = [int((g >> r).sum()) + len(g) * (1 + r) for r in range(MAX_RICE + 1)]
        r = int(np.argmin(costs))
        syms, inv, cnt = np.unique(zz[lo:hi], return_inverse=True, return_counts=True)
        lengths = _huffman_lengths(cnt)
        codes = _canonical_codes(lengths)
        rice[lo:hi] = r
        hcode[lo:hi] = codes[inv]
        hlen[lo:hi] = lengths[inv]
        groups.append(Group(int(exps[lo]), int(hi - lo), r, syms, lengths))

    q = gaps >> rice
    rice_len = q + 1 + rice
    elem_bits = rice_len + hlen
    starts_bit = np.concatenate([[0], np.cumsum(elem_bits)[:-1]])
    total_bits = int(elem_bits.sum())
    bits = np.zeros(total_bits + 8, dtype=np.uint8)
    # unary ones
    ones_at = np.repeat(starts_bit, q) + (np.arange(int(q.sum())) - np.repeat(np.cumsum(q) - q, q))
    bits[ones_at] = 1
    # Rice remainder, MSB first
    rem_start = starts_bit + q + 1
    for j in range(int(rice.max()) + 1):
        m = j < rice
        bits[rem_start[m] + j] = (gaps[m] >> (rice[m] - 1 - j)) & 1
    # Huffman code, MSB first
    h_start = starts_bit + rice_len
    for j in range(int(hlen.max())):
        m = j < hlen
        bits[h_start[m] + j] = (hcode[m] >> (hlen[m] - 1 - j)) & 1

    for lo, hi in zip(cuts[:-1], cuts[1:]):
        b = elem_bits[lo:hi]
        block_bits.append(np.add.reduceat(b, np.arange(0, len(b), BLOCK)))
    payload = np.packbits(bits[:total_bits], bitorder="big").tobytes()
    return TensorUpdate(old.size, checksum(new), groups, np.concatenate(block_bits), payload)


# ---------------------------------------------------------------- decoder

def _read_bits(buf: np.ndarray, pos: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Read n[i] (<= 32) bits starting at bit pos[i] from a byte buffer padded with 8 zero bytes."""
    byte = pos >> 3
    win = np.zeros(len(pos), dtype=np.uint64)
    for k in range(8):
        win |= buf[byte + k].astype(np.uint64) << np.uint64(56 - 8 * k)
    win <<= (pos & 7).astype(np.uint64)
    out = np.zeros(len(pos), dtype=np.int64)
    nz = n > 0
    out[nz] = (win[nz] >> (np.uint64(64) - n[nz].astype(np.uint64))).astype(np.int64)
    return out


def decode_tensor(old: np.ndarray, upd: TensorUpdate) -> np.ndarray:
    """New uint16 bits for one tensor; raises CorruptUpdate if the update does not decode exactly."""
    order, starts = exponent_order(old)
    sizes = np.diff(starts)
    total = sum(g.count for g in upd.groups)
    n_blocks = sum(-(-g.count // BLOCK) for g in upd.groups)
    if len(upd.block_bits) != n_blocks:
        raise CorruptUpdate("block table does not match group counts")
    if sum(upd.block_bits.tolist()) > 8 * len(upd.payload):  # Python ints: no wraparound
        raise CorruptUpdate("payload shorter than its block table")

    # per-block tables
    blk_group, blk_count = [], []
    seen = set()
    lmax = 1
    for gi, g in enumerate(upd.groups):
        if g.exponent in seen or not 0 < g.count <= sizes[g.exponent] or g.rice_r > MAX_RICE:
            raise CorruptUpdate("bad group header")
        if len(np.unique(g.symbols)) != len(g.symbols) or g.symbols.max() > MAX_SYMBOL:
            raise CorruptUpdate("bad symbol table")
        seen.add(g.exponent)
        _check_kraft(g.lengths)
        lmax = max(lmax, int(g.lengths.max()))
        nb = -(-g.count // BLOCK)
        blk_group += [gi] * nb
        blk_count += [min(BLOCK, g.count - b * BLOCK) for b in range(nb)]
    blk_group = np.array(blk_group, dtype=np.int64)
    blk_count = np.array(blk_count, dtype=np.int64)
    elem_base = np.cumsum(np.r_[0, blk_count[:-1]])
    cur = np.concatenate([[0], np.cumsum(upd.block_bits)[:-1]]).astype(np.int64)
    blk_end = cur + upd.block_bits.astype(np.int64)

    # lookup table per group: peek lmax bits -> (symbol, length); length 0 = invalid code
    lut_sym = np.zeros((len(upd.groups), 1 << lmax), dtype=np.int64)
    lut_len = np.zeros((len(upd.groups), 1 << lmax), dtype=np.int64)
    for gi, g in enumerate(upd.groups):
        codes = _canonical_codes(g.lengths)
        for s, c, L in zip(g.symbols.tolist(), codes.tolist(), g.lengths.tolist()):
            lo, hi = c << (lmax - L), (c + 1) << (lmax - L)
            lut_sym[gi, lo:hi] = s
            lut_len[gi, lo:hi] = L
    rice = np.array([g.rice_r for g in upd.groups], dtype=np.int64)[blk_group]
    # a gap can never reach its group's size; bounding q keeps (q << r) and cumsum in int64
    q_max = np.array([int(sizes[g.exponent]) >> g.rice_r for g in upd.groups], dtype=np.int64)[blk_group]

    buf = np.frombuffer(upd.payload + bytes(16), dtype=np.uint8)
    bitarr = np.unpackbits(buf, bitorder="big")
    zeros = np.flatnonzero(bitarr == 0)
    gaps = np.zeros(total, dtype=np.int64)
    zz = np.zeros(total, dtype=np.int64)
    for step in range(BLOCK):
        act = np.flatnonzero(step < blk_count)
        if act.size == 0:
            break
        c = cur[act]
        if (c >= blk_end[act]).any():
            raise CorruptUpdate("block ends before all its symbols are decoded")
        k = np.searchsorted(zeros, c)
        if (k >= len(zeros)).any():
            raise CorruptUpdate("unterminated Rice code")
        nxt = zeros[k]  # end of the unary run
        q = nxt - c
        if (q > q_max[act]).any():
            raise CorruptUpdate("Rice code longer than its group")
        c = nxt + 1
        r = rice[act]
        rem = _read_bits(buf, c, r)
        c = c + r
        peek = _read_bits(buf, c, np.full(act.size, lmax))
        L = lut_len[blk_group[act], peek]
        if (L == 0).any():
            raise CorruptUpdate("invalid Huffman code")
        out = elem_base[act] + step
        gaps[out] = (q << r) | rem
        zz[out] = lut_sym[blk_group[act], peek]
        cur[act] = c + L
    if (cur != blk_end).any():
        raise CorruptUpdate("block lengths do not match decoded bits")

    new = old.copy()
    lo = 0
    for g in upd.groups:
        hi = lo + g.count
        if (gaps[lo:hi] >= sizes[g.exponent]).any():
            raise CorruptUpdate("gap larger than its exponent group")
        local = np.cumsum(gaps[lo:hi] + 1) - 1
        if local[-1] >= sizes[g.exponent]:
            raise CorruptUpdate("position outside its exponent group")
        idx = order[starts[g.exponent] + local]
        val = old[idx].view(np.int16).astype(np.int64) + _unzigzag(zz[lo:hi])
        new[idx] = (val & 0xFFFF).astype(np.uint16)
        lo = hi
    if checksum(new) != upd.checksum:
        raise CorruptUpdate("checksum mismatch after applying the update")
    return new
