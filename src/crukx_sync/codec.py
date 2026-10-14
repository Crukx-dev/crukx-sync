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


def _pack_bits(G, R, HC, HL, carry: int):
    """Bitstream for one tensor's changes (Rice gap then Huffman code per element), starting at
    bit offset `carry` (0-7) inside its first byte. Returns (bytes, per-element bit lengths)."""
    dev = G.device
    q = G >> R
    Lr = q + 1 + R
    L = Lr + HL
    start = _exclusive_cumsum(L) + carry
    total_bits = carry + int(L.sum())
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
    data = torch.stack([(words >> s) & 0xFF for s in (24, 16, 8, 0)], 1).flatten()
    return data[: (total_bits + 7) // 8].to(torch.uint8).cpu().numpy().tobytes(), L


def encode(old: Mapping[str, torch.Tensor], new: Mapping[str, torch.Tensor],
           checksums: dict[str, int] | None = None) -> tuple[Update, dict[str, int]]:
    """The update, and the new checksums for the caller to commit once the update is serialized.

    Works one tensor at a time (find changes, then write bits), so GPU memory is bounded by the
    largest tensor's changes, not the whole update's."""
    pairs = {}
    for name in old:
        a, b = bits16(name, old[name]), bits16(name, new[name])
        if a.shape != b.shape:
            raise ValueError(f"{name}: old and new differ in shape")
        pairs[name] = (a, b)
    if not pairs:
        return Update([], []), {}
    if len({t.device for ab in pairs.values() for t in ab}) != 1:
        raise ValueError("all tensors must be on one device")
    dev = next(iter(pairs.values()))[0].device
    changed_any = torch.stack([(a != b).any() for a, b in pairs.values()]).tolist()  # one sync

    # pass 1, per tensor: changes, groups, Rice parameters, symbol counts; keep compact arrays
    headers, kept, new_checksums = [], [], {}
    dense = torch.zeros(256 * _SMALL, dtype=torch.long, device=dev)
    rare = []
    for (name, (a, b)), any_change in zip(pairs.items(), changed_any):
        if not any_change:
            continue
        ex, local, src = _changed_ranks(a, a != b)
        first = torch.ones_like(ex, dtype=torch.bool)
        first[1:] = ex[1:] != ex[:-1]
        prev = torch.roll(local, 1)
        prev[first] = -1
        gaps = local - prev - 1
        res = b[src].long() - a[src].long()
        zz = (res << 1) ^ (res >> 63)
        exps, counts = torch.unique_consecutive(ex, return_counts=True)
        gid = torch.repeat_interleave(torch.arange(len(counts), device=dev), counts)
        costs = torch.stack([torch.zeros(len(counts), dtype=torch.long, device=dev)
                             .scatter_add_(0, gid, (gaps >> r) + 1 + r) for r in range(MAX_RICE + 1)])
        rice = costs.argmin(0)
        small = zz < _SMALL
        dense += torch.bincount(ex[small] * _SMALL + zz[small], minlength=256 * _SMALL)
        if not bool(small.all()):
            rare.append(ex[~small] * _SYM_SPACE + zz[~small])
        new_checksums[name] = _new_checksum(name, a, b, src, checksums)
        headers.append(TensorHeader(name, a.numel(), new_checksums[name],
                                    [Group(int(x), int(c), int(r)) for x, c, r in
                                     zip(exps.tolist(), counts.tolist(), rice.tolist())]))
        # compact copies for pass 3 (exponent fits int16, zigzag symbol fits int32)
        kept.append((ex.to(torch.int16), gaps, zz.to(torch.int32), rice[gid].to(torch.int8), counts))
    if not headers:
        return Update([], []), {}

    # pass 2: one Huffman table per exponent for the whole update
    nz = torch.nonzero(dense).flatten()
    keys = (nz // _SMALL) * _SYM_SPACE + nz % _SMALL
    kcount = dense[nz]
    if rare:
        rk, rc = torch.unique(torch.cat(rare), return_counts=True)
        keys, kcount = torch.cat([keys, rk]), torch.cat([kcount, rc])
    srt = torch.argsort(keys)
    keys_t = keys[srt]
    keys, kcount = keys_t.tolist(), kcount[srt].tolist()
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
    code_t = torch.tensor(code_of, dtype=torch.long, device=dev)
    len_t = torch.tensor(len_of, dtype=torch.long, device=dev)

    # pass 3, per tensor: write bits; tensors join mid-byte, so carry the bit offset across
    payload, carry, block_bits = bytearray(), 0, []
    for ex, gaps, zz, rice_e, counts in kept:
        inv = torch.searchsorted(keys_t, ex.long() * _SYM_SPACE + zz.long())
        data, L = _pack_bits(gaps, rice_e.long(), code_t[inv], len_t[inv], carry)
        if carry:
            payload[-1] |= data[0]
            payload += data[1:]
        else:
            payload += data
        carry = (carry + int(L.sum())) % 8
        gid = torch.repeat_interleave(torch.arange(len(counts), device=dev), counts)
        k_in_group = torch.arange(L.numel(), device=dev) - _exclusive_cumsum(counts)[gid]
        nb = (counts + BLOCK - 1) // BLOCK
        bid = _exclusive_cumsum(nb)[gid] + k_in_group // BLOCK
        bb = torch.zeros(int(nb.sum()), dtype=torch.long, device=dev).scatter_add_(0, bid, L)
        if int(bb.max()) >= 1 << 32:
            raise ValueError("a block exceeds 2^32 bits; the update cannot be represented")
        block_bits += bb.tolist()
    return Update(tables, headers, block_bits, bytes(payload)), new_checksums


# ---------------------------------------------------------------- decoder

def _read_bits(buf: torch.Tensor, pos: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """n (<= 32) bits at bit position pos, MSB first; pos is clamped into the buffer."""
    byte = (pos >> 3).clamp(0, buf.numel() - 5)
    win = torch.zeros_like(pos)
    for k in range(5):
        win |= buf[byte + k].long() << (32 - 8 * k)
    return (win >> (40 - (pos & 7) - n)) & ((torch.ones_like(n) << n) - 1)


def _canonical_tables(tables, dev):
    """Per table and code length: first canonical code, number of codes, index of the first symbol
    of that length; plus all symbols in (length, table position) order. A few hundred bytes per
    table regardless of what the update claims, unlike a 2^max_length lookup table."""
    n = len(tables)
    first = torch.zeros(n, MAX_CODE_LEN + 1, dtype=torch.long)
    count = torch.zeros(n, MAX_CODE_LEN + 1, dtype=torch.long)
    base = torch.zeros(n, MAX_CODE_LEN + 1, dtype=torch.long)
    syms = []
    for ti, t in enumerate(tables):
        order = sorted(range(len(t.lengths)), key=lambda i: (t.lengths[i], i))
        per_len = [0] * (MAX_CODE_LEN + 1)
        for L in t.lengths:
            per_len[L] += 1
        code, idx = 0, len(syms)
        for L in range(1, MAX_CODE_LEN + 1):
            first[ti, L], count[ti, L], base[ti, L] = code, per_len[L], idx
            code = (code + per_len[L]) << 1
            idx += per_len[L]
        syms += [t.symbols[i] for i in order]
    return first.to(dev), count.to(dev), base.to(dev), torch.tensor(syms, dtype=torch.long, device=dev)


def _cache_get(checksums, name, t):
    """A cached checksum, trusted only if the tensor was not modified since we recorded it."""
    hit = checksums.get(name) if checksums is not None else None
    return hit[0] if isinstance(hit, tuple) and hit[1] == t._version else None


def apply(weights: Mapping[str, torch.Tensor], u: Update, checksums: dict | None = None) -> int:
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

    # per group / per block tables (sizes bounded by the update's length, checked in parse)
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
    nb_g = (counts + BLOCK - 1) // BLOCK
    blk_group = torch.repeat_interleave(torch.arange(len(g_count), device=dev), nb_g)
    blk_k = torch.arange(len(u.block_bits), device=dev) - _exclusive_cumsum(nb_g)[blk_group]
    blk_count = torch.clamp(counts[blk_group] - blk_k * BLOCK, max=BLOCK)
    elem_base = _exclusive_cumsum(counts)[blk_group] + blk_k * BLOCK
    rice = torch.tensor(g_rice, dtype=torch.long, device=dev)[blk_group]
    q_max = torch.tensor(g_size, dtype=torch.long, device=dev)[blk_group] >> rice
    tbl = torch.tensor(g_table, dtype=torch.long, device=dev)[blk_group]
    bb = torch.tensor(u.block_bits, dtype=torch.long, device=dev)
    cur = _exclusive_cumsum(bb)
    blk_end = cur + bb
    first, ncode, base, syms = _canonical_tables(u.tables, dev)
    lens = torch.arange(1, MAX_CODE_LEN + 1, device=dev)

    buf = torch.frombuffer(bytearray(u.payload + bytes(16)), dtype=torch.uint8).to(dev)
    bits = (buf.unsqueeze(1) >> torch.arange(7, -1, -1, dtype=torch.uint8, device=dev)) & 1
    idx_t = torch.int32 if bits.numel() < 1 << 31 else torch.long
    zeros = torch.nonzero(bits.flatten() == 0).flatten().to(idx_t)
    del bits
    total = int(counts.sum())
    gaps = torch.zeros(total + 1, dtype=torch.long, device=dev)  # last slot absorbs inactive writes
    zz = torch.zeros(total + 1, dtype=torch.long, device=dev)
    peek_n = torch.full_like(cur, MAX_CODE_LEN)
    bad = torch.zeros((), dtype=torch.bool, device=dev)
    for step in range(BLOCK):
        active = step < blk_count
        bad |= (active & (cur >= blk_end)).any()
        k = torch.searchsorted(zeros, cur.to(idx_t))
        bad |= (active & (k >= zeros.numel())).any()
        nxt = zeros[k.clamp(max=zeros.numel() - 1)].long()
        q = nxt - cur
        bad |= (active & (q > q_max)).any()
        c = nxt + 1
        rem = _read_bits(buf, c, rice)
        c = c + rice
        # canonical decode: the shortest length whose code range contains the prefix
        codes = _read_bits(buf, c, peek_n).unsqueeze(1) >> (MAX_CODE_LEN - lens)
        fc, nc = first[tbl, 1:], ncode[tbl, 1:]
        ok = (codes >= fc) & (codes < fc + nc)
        bad |= (active & ~ok.any(1)).any()
        L = ok.int().argmax(1) + 1
        code = codes.gather(1, (L - 1).unsqueeze(1)).squeeze(1)
        sym_i = base[tbl, L] + code - first[tbl, L]
        out = torch.where(active, elem_base + step, total)
        gaps[out] = (q.clamp(0, 1 << 32) << rice) | rem
        zz[out] = syms[sym_i.clamp(0, syms.numel() - 1)]
        cur = torch.where(active, c + L, cur)
    bad |= (cur != blk_end).any()

    # per tensor: positions, new values, checksum delta. Validation flags and checksums are
    # gathered as tensors and checked once at the end; nothing is written until all pass.
    staged, lo, gi, deltas, expected = [], 0, 0, [], []
    for h, a, hist in zip(u.tensors, targets, hists):
        n_t = sum(x.count for x in h.groups)
        g_cnt = counts[gi:gi + len(h.groups)]
        g = gaps[lo:lo + n_t]
        ex = torch.repeat_interleave(torch.tensor([x.exponent for x in h.groups], device=dev), g_cnt)
        grp = torch.repeat_interleave(torch.arange(len(h.groups), device=dev), g_cnt)
        if hist is None:
            _, order, starts = _sorted_slab(a, 0)
            size = (starts[1:] - starts[:-1])[ex]
        else:
            size = hist.sum(0)[ex]
        bad |= (g >= size).any()
        cs = torch.cumsum(g.clamp(max=1 << 40) + 1, 0)
        local = cs - 1 - (cs - (g.clamp(max=1 << 40) + 1))[_exclusive_cumsum(g_cnt)][grp]
        bad |= (local >= size).any()
        local = torch.minimum(local, size - 1)  # keep gathers in range; `bad` already flags it
        idx = order[starts[ex] + local] if hist is None else _positions(a, ex, local, hist)
        old_u = a[idx].long() & 0xFFFF
        z = zz[lo:lo + n_t]
        new_u = (old_u + ((z >> 1) ^ -(z & 1))) & 0xFFFF
        old_chk = _cache_get(checksums, h.name, weights[h.name])
        if old_chk is None:
            old_chk = checksum(a)
        deltas.append(((new_u - old_u) * _weights(idx)).sum() + _i64(old_chk))
        expected.append(_i64(h.checksum))
        staged.append((a, idx, (new_u - ((new_u >> 15) << 16)).to(torch.int16)))
        lo += n_t
        gi += len(h.groups)
    ok = torch.stack(deltas) == torch.tensor(expected, dtype=torch.long, device=dev)
    if bool(bad) or not bool(ok.all()):  # one sync for the whole update
        names = [h.name for h, good in zip(u.tensors, ok.tolist()) if not good]
        raise CorruptUpdate(f"update does not verify (checksum mismatch: {names[:3]})" if names
                            else "payload positions outside their exponent groups")
    for a, idx, vals in staged:
        a[idx] = vals
    if checksums is not None:  # record the version after our own write
        checksums.update({h.name: (h.checksum, weights[h.name]._version) for h in u.tensors})
    return total
