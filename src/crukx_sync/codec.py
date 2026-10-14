"""Encode and apply updates with torch, on CPU or GPU (docs/FORMAT.md, version 2).

Everything is vectorised over the whole update. The decoder runs all 256-symbol blocks of all
tensors in one loop of 256 steps; corrupt input only raises flags (indices are clamped), checked
once at the end, so bad bytes can never trigger a device-side assert.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping

import torch

from .wire import (BLOCK, MAX_CODE_LEN, MAX_SYMBOL, MAX_RICE, CorruptUpdate, Group, Table,
                   TensorHeader, Update, n_blocks)

_SYM_SPACE = MAX_SYMBOL + 1
_SMALL = 64  # residual symbols below this are counted with a dense bincount
_MASK64 = (1 << 64) - 1


def _i64(v: int) -> int:
    """A uint64 constant as signed int64 (torch int64 arithmetic wraps mod 2^64)."""
    return v - (1 << 64) if v >= 1 << 63 else v


_GOLDEN, _MIX1, _MIX2 = _i64(0x9E3779B97F4A7C15), _i64(0xBF58476D1CE4E5B9), _i64(0x94D049BB133111EB)
_CHUNK = 1 << 24   # elements per checksum chunk
_SLAB = 1 << 24    # elements sorted at once: bounds temporary memory for any tensor size


# ---------------------------------------------------------------- shared helpers

def bits16(name: str, t: torch.Tensor) -> torch.Tensor:
    """Flat int16 view of a bf16 (or int16 bit-pattern) tensor."""
    if not isinstance(t, torch.Tensor) or t.dtype not in (torch.bfloat16, torch.int16):
        raise ValueError(f"{name}: expected a bfloat16 or int16 torch tensor, got {getattr(t, 'dtype', type(t))}")
    if not t.is_contiguous():
        raise ValueError(f"{name}: tensor must be contiguous")
    return t.view(torch.int16).view(-1)


def exponents(a: torch.Tensor) -> torch.Tensor:
    return ((a >> 7) & 0xFF).long()


def _shr(z: torch.Tensor, s: int) -> torch.Tensor:
    """Logical right shift of int64 bit patterns."""
    return (z >> s) & ((1 << (64 - s)) - 1)


def _weights(idx: torch.Tensor) -> torch.Tensor:
    """splitmix64(i) | 1: a non-linear per-position weight, so structured errors cannot cancel."""
    z = idx * _GOLDEN + _GOLDEN
    z = (z ^ _shr(z, 30)) * _MIX1
    z = (z ^ _shr(z, 27)) * _MIX2
    return (z ^ _shr(z, 31)) | 1


def checksum(a: torch.Tensor) -> int:
    """sum(u_i * w_i) mod 2^64 over the uint16 bits u_i (FORMAT.md, Integrity)."""
    total = torch.zeros((), dtype=torch.long, device=a.device)
    for lo in range(0, a.numel(), _CHUNK):
        hi = min(lo + _CHUNK, a.numel())
        idx = torch.arange(lo, hi, dtype=torch.long, device=a.device)
        total += ((a[lo:hi].long() & 0xFFFF) * _weights(idx)).sum()
    return int(total) & _MASK64


def huffman_lengths(counts: list[int]) -> list[int]:
    """Huffman code lengths limited to MAX_CODE_LEN (flatten skewed counts, else equal lengths)."""
    for _ in range(8):
        lengths = _huffman_unlimited(counts)
        if max(lengths) <= MAX_CODE_LEN:
            return lengths
        counts = [max(1, c >> 2) for c in counts]
    return [max(1, (len(counts) - 1).bit_length())] * len(counts)


def _huffman_unlimited(counts: list[int]) -> list[int]:
    n = len(counts)
    if n == 1:
        return [1]
    heap = [(c, i) for i, c in enumerate(counts)]
    heapq.heapify(heap)
    parent = [0] * (2 * n - 1)
    nxt = n
    while len(heap) > 1:
        c1, a = heapq.heappop(heap)
        c2, b = heapq.heappop(heap)
        parent[a] = parent[b] = nxt
        heapq.heappush(heap, (c1 + c2, nxt))
        nxt += 1
    depth = [0] * (2 * n - 1)  # the root is the last node; parents have larger ids
    for node in range(2 * n - 3, -1, -1):
        depth[node] = depth[parent[node]] + 1
    return depth[:n]


def canonical_codes(lengths: list[int]) -> list[int]:
    """Canonical codes, assigned in order of (length, position in the table)."""
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    codes = [0] * len(lengths)
    code, prev = 0, lengths[order[0]]
    for k, i in enumerate(order):
        if k:
            code = (code + 1) << (lengths[i] - prev)
        codes[i] = code
        prev = lengths[i]
    return codes


def _exclusive_cumsum(x: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(x, 0) - x


def _slab_hists(a: torch.Tensor) -> torch.Tensor:
    """Exponent histogram of every slab: [n_slabs, 256]."""
    return torch.stack([torch.bincount(exponents(a[lo:lo + _SLAB]), minlength=256)
                        for lo in range(0, a.numel(), _SLAB)])


def _sorted_slab(a: torch.Tensor, lo: int):
    """One slab's exponents (int16), their stable sort order, and where each exponent starts in
    that order ([257]). A stable sort on 16-bit keys is the fastest rank/select we measured."""
    e = (a[lo:lo + _SLAB] >> 7) & 0xFF
    order = torch.argsort(e, stable=True)
    starts = torch.searchsorted(e[order], torch.arange(257, dtype=e.dtype, device=e.device))
    return e, order, starts


def _changed_ranks(a: torch.Tensor, m: torch.Tensor):
    """For changed elements: (exponent, rank among same-exponent elements, position), sorted by
    (exponent, rank): the order of a stable sort by exponent, done slab by slab."""
    before = torch.zeros(256, dtype=torch.long, device=a.device)
    out = []
    for lo in range(0, a.numel(), _SLAB):
        e, order, starts = _sorted_slab(a, lo)
        r = torch.nonzero(m[lo:lo + _SLAB][order]).flatten()
        if r.numel():
            x = e[order[r]].long()
            out.append((x, before[x] + r - starts[x], lo + order[r]))
        before += starts[1:] - starts[:-1]
    ex, rank, src = (torch.cat(t) for t in zip(*out))
    key = torch.argsort((ex << 40) + rank)
    return ex[key], rank[key], src[key]


def _positions(a: torch.Tensor, ex: torch.Tensor, rank: torch.Tensor, hists: torch.Tensor) -> torch.Tensor:
    """Inverse of _changed_ranks: element positions for (exponent, rank) pairs."""
    slab_before = _exclusive_cumsum(hists)                     # [n_slabs, 256]
    slab = torch.searchsorted(slab_before.T.contiguous()[ex], rank.unsqueeze(1), right=True).squeeze(1) - 1
    pos = torch.empty_like(rank)
    for s, lo in enumerate(range(0, a.numel(), _SLAB)):
        sel = torch.nonzero(slab == s).flatten()
        if sel.numel() == 0:
            continue
        _, order, starts = _sorted_slab(a, lo)
        x = ex[sel]
        pos[sel] = lo + order[starts[x] + rank[sel] - slab_before[s, x]]
    return pos


# ---------------------------------------------------------------- encoder

def _new_checksum(name, a, b, idx, checksums) -> int:
    """Checksum of `b`: incremental from the cached checksum of `a` when available."""
    if checksums is not None and name in checksums:
        d = (b[idx].long() & 0xFFFF) - (a[idx].long() & 0xFFFF)
        return (checksums[name] + int((d * _weights(idx)).sum())) & _MASK64
    return checksum(b)


def encode(old: Mapping[str, torch.Tensor], new: Mapping[str, torch.Tensor],
           checksums: dict[str, int] | None = None) -> Update:
    headers, E, G, Z, group_counts = [], [], [], [], []
    new_checksums = {}  # committed to `checksums` only if the whole encode succeeds
    pairs = {}
    for name in old:
        a, b = bits16(name, old[name]), bits16(name, new[name])
        if a.shape != b.shape or a.device != b.device:
            raise ValueError(f"{name}: old and new differ in shape or device")
        pairs[name] = (a, b)
    if not pairs:
        return Update([], [])
    changed_any = torch.stack([(a != b).any() for a, b in pairs.values()]).tolist()  # one sync
    for (name, (a, b)), any_change in zip(pairs.items(), changed_any):
        if not any_change:
            continue
        m = a != b
        ex, local, src = _changed_ranks(a, m)
        first = torch.ones_like(ex, dtype=torch.bool)
        first[1:] = ex[1:] != ex[:-1]
        prev = torch.roll(local, 1)
        prev[first] = -1
        res = b[src].long() - a[src].long()
        exps, counts = torch.unique_consecutive(ex, return_counts=True)
        chk = new_checksums[name] = _new_checksum(name, a, b, src, checksums)
        headers.append(TensorHeader(name, a.numel(), chk,
                                    [Group(int(x), int(c), 0) for x, c in zip(exps.tolist(), counts.tolist())]))
        E.append(ex)
        G.append(local - prev - 1)
        Z.append((res << 1) ^ (res >> 63))
        group_counts += counts.tolist()
    if not headers:
        return Update([], [])
    if checksums is not None:
        checksums.update(new_checksums)
    E, G, Z = torch.cat(E), torch.cat(G), torch.cat(Z)
    dev, n = E.device, E.numel()
    counts = torch.tensor(group_counts, dtype=torch.long, device=dev)
    gid = torch.repeat_interleave(torch.arange(len(group_counts), device=dev), counts)

    # Rice parameter per group: cheapest total length
    costs = torch.stack([torch.zeros(len(group_counts), dtype=torch.long, device=dev)
                         .scatter_add_(0, gid, (G >> r) + 1 + r) for r in range(MAX_RICE + 1)])
    rice_g = costs.argmin(0)
    R = rice_g[gid]

    # Huffman table per exponent, shared by the whole update
    # count (exponent, symbol) pairs: dense bincount for small symbols, unique() for the rare rest
    key = E * _SYM_SPACE + Z
    small = Z < _SMALL
    dense = torch.bincount(E[small] * _SMALL + Z[small], minlength=256 * _SMALL)
    nz = torch.nonzero(dense).flatten()
    rare_keys, rare_counts = torch.unique(key[~small], return_counts=True)
    keys = torch.cat([(nz // _SMALL) * _SYM_SPACE + nz % _SMALL, rare_keys])
    kcount = torch.cat([dense[nz], rare_counts])
    srt = torch.argsort(keys)
    keys, kcount = keys[srt], kcount[srt]
    inv = torch.searchsorted(keys, key)
    keys, kcount = keys.tolist(), kcount.tolist()
    code_of, len_of, tables, lo = [0] * len(keys), [0] * len(keys), [], 0
    while lo < len(keys):
        exp = keys[lo] // _SYM_SPACE
        hi = lo
        while hi < len(keys) and keys[hi] // _SYM_SPACE == exp:
            hi += 1
        lengths = huffman_lengths(kcount[lo:hi])
        code_of[lo:hi], len_of[lo:hi] = canonical_codes(lengths), lengths
        tables.append(Table(exp, [k % _SYM_SPACE for k in keys[lo:hi]], lengths))
        lo = hi
    HC = torch.tensor(code_of, dtype=torch.long, device=dev)[inv]
    HL = torch.tensor(len_of, dtype=torch.long, device=dev)[inv]

    # bit lengths, blocks
    q = G >> R
    Lr = q + 1 + R
    L = Lr + HL
    start = _exclusive_cumsum(L)
    total_bits = int(L.sum())
    k_in_group = torch.arange(n, device=dev) - _exclusive_cumsum(counts)[gid]
    blocks_per_group = (counts + BLOCK - 1) // BLOCK
    bid = _exclusive_cumsum(blocks_per_group)[gid] + k_in_group // BLOCK
    block_bits = torch.zeros(int(blocks_per_group.sum()), dtype=torch.long, device=dev).scatter_add_(0, bid, L)

    # set bits: unary ones, Rice remainder (MSB first), Huffman code (MSB first)
    ones = [torch.repeat_interleave(start, q)
            + torch.arange(int(q.sum()), device=dev) - torch.repeat_interleave(_exclusive_cumsum(q), q)]
    for j in range(int(R.max()) + 1):
        bit = (j < R) & (((G >> (R - 1 - j).clamp(min=0)) & 1) == 1)
        ones.append((start + q + 1 + j)[bit])
    for j in range(int(HL.max())):
        bit = (j < HL) & (((HC >> (HL - 1 - j).clamp(min=0)) & 1) == 1)
        ones.append((start + Lr + j)[bit])
    pos = torch.cat(ones)
    words = torch.zeros(total_bits // 32 + 1, dtype=torch.long, device=dev)
    words.scatter_add_(0, pos >> 5, torch.ones_like(pos) << (31 - (pos & 31)))
    payload = torch.stack([(words >> s) & 0xFF for s in (24, 16, 8, 0)], 1).flatten()
    payload = payload[: (total_bits + 7) // 8].to(torch.uint8).cpu().numpy().tobytes()

    rice_list = rice_g.tolist()
    gi = 0
    for h in headers:
        for g in h.groups:
            g.rice_r = rice_list[gi]
            gi += 1
    return Update(tables, headers, block_bits.tolist(), payload)


# ---------------------------------------------------------------- decoder

def _read_bits(buf: torch.Tensor, pos: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """n (<= 32) bits at bit position pos, MSB first; pos is clamped into the buffer."""
    byte = (pos >> 3).clamp(0, buf.numel() - 5)
    win = torch.zeros_like(pos)
    for k in range(5):
        win |= buf[byte + k].long() << (32 - 8 * k)
    return (win >> (40 - (pos & 7) - n)) & ((torch.ones_like(n) << n) - 1)


def apply(weights: Mapping[str, torch.Tensor], u: Update, checksums: dict[str, int] | None = None) -> int:
    """Verify `u` against `weights`, then write the new values in place. Returns elements changed."""
    if not u.tensors:
        return 0
    targets = []
    for h in u.tensors:
        if h.name not in weights:
            raise ValueError(f"update has tensor {h.name!r} that the receiver does not have")
        a = bits16(h.name, weights[h.name])
        if a.numel() != h.numel:
            raise ValueError(f"{h.name}: update is for {h.numel} elements, receiver has {a.numel()}")
        targets.append(a)
    dev = targets[0].device
    if any(a.device != dev for a in targets):
        raise ValueError("all updated tensors must be on one device")

    # per group / per block tables (small, built on CPU)
    table_of = {t.exponent: i for i, t in enumerate(u.tables)}
    g_count, g_rice, g_table, g_size = [], [], [], []
    # exact group sizes need a pass over the tensor; single-slab tensors get them later from the
    # same sort that places the changes, so until then their bound is numel
    hists = [_slab_hists(a) if a.numel() > _SLAB else None for a in targets]
    for h, a, hist in zip(u.tensors, targets, hists):
        sizes = hist.sum(0).tolist() if hist is not None else [a.numel()] * 256
        for g in h.groups:
            if g.count > sizes[g.exponent]:
                raise CorruptUpdate("group has more changes than elements")
            g_count.append(g.count)
            g_rice.append(g.rice_r)
            g_table.append(table_of[g.exponent])
            g_size.append(sizes[g.exponent])
    counts = torch.tensor(g_count, dtype=torch.long, device=dev)
    rice_g = torch.tensor(g_rice, dtype=torch.long, device=dev)
    size_g = torch.tensor(g_size, dtype=torch.long, device=dev)
    nb_g = (counts + BLOCK - 1) // BLOCK
    if int(nb_g.sum()) != len(u.block_bits):
        raise CorruptUpdate("block table does not match group counts")
    blk_group = torch.repeat_interleave(torch.arange(len(g_count), device=dev), nb_g)
    blk_k = torch.arange(len(u.block_bits), device=dev) - _exclusive_cumsum(nb_g)[blk_group]
    blk_count = torch.clamp(counts[blk_group] - blk_k * BLOCK, max=BLOCK)
    elem_base = _exclusive_cumsum(counts)[blk_group] + blk_k * BLOCK
    rice = rice_g[blk_group]
    q_max = size_g[blk_group] >> rice
    tbl = torch.tensor(g_table, dtype=torch.long, device=dev)[blk_group]
    bb = torch.tensor(u.block_bits, dtype=torch.long, device=dev)
    cur = _exclusive_cumsum(bb)
    blk_end = cur + bb

    lmax = max(max(t.lengths) for t in u.tables)
    lut_sym = torch.zeros(len(u.tables), 1 << lmax, dtype=torch.long)
    lut_len = torch.zeros(len(u.tables), 1 << lmax, dtype=torch.long)
    for ti, t in enumerate(u.tables):
        for s, c, L in zip(t.symbols, canonical_codes(t.lengths), t.lengths):
            lut_sym[ti, c << (lmax - L):(c + 1) << (lmax - L)] = s
            lut_len[ti, c << (lmax - L):(c + 1) << (lmax - L)] = L
    lut_sym, lut_len = lut_sym.to(dev), lut_len.to(dev)

    buf = torch.frombuffer(bytearray(u.payload + bytes(16)), dtype=torch.uint8).to(dev)
    bits = ((buf.unsqueeze(1) >> torch.arange(7, -1, -1, device=dev)) & 1).flatten()
    zeros = torch.nonzero(bits == 0).flatten()
    total = int(counts.sum())
    gaps = torch.zeros(total + 1, dtype=torch.long, device=dev)  # last slot absorbs inactive writes
    zz = torch.zeros(total + 1, dtype=torch.long, device=dev)
    lmax_t = torch.full_like(cur, lmax)
    bad = torch.zeros((), dtype=torch.bool, device=dev)
    for step in range(BLOCK):
        active = step < blk_count
        bad |= (active & (cur >= blk_end)).any()
        k = torch.searchsorted(zeros, cur)
        bad |= (active & (k >= zeros.numel())).any()
        nxt = zeros[k.clamp(max=zeros.numel() - 1)]
        q = nxt - cur
        bad |= (active & (q > q_max)).any()
        c = nxt + 1
        rem = _read_bits(buf, c, rice)
        c = c + rice
        peek = _read_bits(buf, c, lmax_t)
        L = lut_len[tbl, peek]
        bad |= (active & (L == 0)).any()
        out = torch.where(active, elem_base + step, total)
        gaps[out] = (q.clamp(0, 1 << 32) << rice) | rem
        zz[out] = lut_sym[tbl, peek]
        cur = torch.where(active, c + L, cur)
    bad |= (cur != blk_end).any()
    gaps, zz = gaps[:total], zz[:total]

    elem_group = torch.repeat_interleave(torch.arange(len(g_count), device=dev), counts)
    bad |= (gaps >= size_g[elem_group]).any()
    cs = torch.cumsum(gaps.clamp(max=1 << 40) + 1, 0)
    local = cs - 1 - (cs - (gaps + 1))[_exclusive_cumsum(counts)][elem_group]
    bad |= (local >= size_g[elem_group]).any()
    if bool(bad):
        raise CorruptUpdate("payload does not decode to the declared changes")

    # positions and new values per tensor; verify every checksum before writing anything
    staged, lo, gi = [], 0, 0
    for h, a, hist in zip(u.tensors, targets, hists):
        n_t = sum(g.count for g in h.groups)
        sl = slice(lo, lo + n_t)
        ex = torch.repeat_interleave(torch.tensor([g.exponent for g in h.groups], device=dev),
                                     counts[gi:gi + len(h.groups)])
        if hist is None:
            _, order, starts = _sorted_slab(a, 0)
            if bool((local[sl] >= (starts[1:] - starts[:-1])[ex]).any()):
                raise CorruptUpdate(f"{h.name}: position outside its exponent group")
            idx = order[starts[ex] + local[sl]]
        else:
            idx = _positions(a, ex, local[sl], hist)
        old_u = a[idx].long() & 0xFFFF
        z = zz[sl]
        new_u = (old_u + ((z >> 1) ^ -(z & 1))) & 0xFFFF
        old_chk = checksums[h.name] if checksums is not None and h.name in checksums else checksum(a)
        chk = (old_chk + int(((new_u - old_u) * _weights(idx)).sum())) & _MASK64
        if chk != h.checksum:
            raise CorruptUpdate(f"{h.name}: checksum mismatch after applying the update")
        staged.append((a, idx, (new_u - ((new_u >> 15) << 16)).to(torch.int16)))
        lo += n_t
        gi += len(h.groups)
    for a, idx, vals in staged:
        a[idx] = vals
    if checksums is not None:
        checksums.update({h.name: h.checksum for h in u.tensors})
    return total
