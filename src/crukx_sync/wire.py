"""The update container (docs/FORMAT.md, version 2): plain data plus bounds-checked parsing."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

MAGIC = b"CRXS"
VERSION = 2
MAX_CODE_LEN = 17   # residuals span +-65535, so <= 131071 zigzag symbols always fit in 17 bits
MAX_SYMBOL = 131070
MAX_RICE = 30
BLOCK = 256


class CorruptUpdate(ValueError):
    """The update bytes are malformed, truncated, or do not reproduce the expected weights."""


@dataclass
class Table:
    """Canonical Huffman table for residual symbols of one old-weight exponent."""
    exponent: int
    symbols: list[int]
    lengths: list[int]


@dataclass
class Group:
    exponent: int
    count: int
    rice_r: int


@dataclass
class TensorHeader:
    name: str
    numel: int
    checksum: int
    groups: list[Group]


@dataclass
class Update:
    tables: list[Table]
    tensors: list[TensorHeader]
    block_bits: list[int] = field(default_factory=list)
    payload: bytes = b""


def n_blocks(count: int) -> int:
    return -(-count // BLOCK)


def write_varint(out: bytearray, v: int) -> None:
    if not 0 <= v < 1 << 63:
        raise ValueError(f"varint out of range: {v}")
    while v >= 0x80:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v)


def serialize(u: Update) -> bytes:
    out = bytearray(MAGIC)
    out.append(VERSION)
    write_varint(out, len(u.tables))
    for t in u.tables:
        out.append(t.exponent)
        write_varint(out, len(t.symbols))
        for s, L in zip(t.symbols, t.lengths):
            write_varint(out, s)
            out.append(L)
    write_varint(out, len(u.tensors))
    for t in u.tensors:
        raw = t.name.encode()
        write_varint(out, len(raw))
        out += raw
        write_varint(out, t.numel)
        out += struct.pack("<Q", t.checksum)
        write_varint(out, len(t.groups))
        for g in t.groups:
            out.append(g.exponent)
            write_varint(out, g.count)
            out.append(g.rice_r)
    out += struct.pack(f"<{len(u.block_bits)}I", *u.block_bits)
    write_varint(out, len(u.payload))
    out += u.payload
    return bytes(out)


class _Reader:
    """Sequential reader; every overrun or out-of-range value is a CorruptUpdate."""

    def __init__(self, data: bytes):
        self.data = memoryview(data)
        self.pos = 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            raise CorruptUpdate("update is truncated")
        b = bytes(self.data[self.pos:self.pos + n])
        self.pos += n
        return b

    def u8(self) -> int:
        return self.take(1)[0]

    def varint(self) -> int:
        v = shift = 0
        while True:
            b = self.u8()
            v |= (b & 0x7F) << shift
            if v >= 1 << 63:
                raise CorruptUpdate("varint out of range")
            if not b & 0x80:
                return v
            shift += 7
            if shift > 63:
                raise CorruptUpdate("varint too long")

    def remaining(self) -> int:
        return len(self.data) - self.pos


def parse(data: bytes) -> Update:
    """Parse and structurally validate an update (limits in docs/FORMAT.md)."""
    r = _Reader(data)
    if r.take(4) != MAGIC:
        raise CorruptUpdate("not a crukx-sync update")
    if (v := r.u8()) != VERSION:
        raise CorruptUpdate(f"unsupported format version {v}")

    n_tables = r.varint()
    if n_tables > 256:
        raise CorruptUpdate("too many tables")
    tables, table_exps = [], set()
    for _ in range(n_tables):
        exp, n_sym = r.u8(), r.varint()
        if exp in table_exps or not 0 < n_sym <= min(MAX_SYMBOL + 1, r.remaining() // 2):
            raise CorruptUpdate("bad table header")
        table_exps.add(exp)
        syms, lens = [], []
        for _ in range(n_sym):
            syms.append(r.varint())
            lens.append(r.u8())
        if len(set(syms)) != n_sym or max(syms) > MAX_SYMBOL:
            raise CorruptUpdate("bad table symbols")
        if min(lens) < 1 or max(lens) > MAX_CODE_LEN or sum(2.0 ** -L for L in lens) > 1.0 + 1e-12:
            raise CorruptUpdate("bad Huffman code lengths")
        tables.append(Table(exp, syms, lens))

    n_tensors = r.varint()
    if n_tensors > r.remaining():
        raise CorruptUpdate("too many tensors")
    tensors, names, total_blocks = [], set(), 0
    for _ in range(n_tensors):
        try:
            name = r.take(r.varint()).decode("utf-8")
        except UnicodeDecodeError:
            raise CorruptUpdate("tensor name is not valid UTF-8") from None
        if name in names:
            raise CorruptUpdate(f"tensor {name!r} appears twice")
        names.add(name)
        numel = r.varint()
        chk = struct.unpack("<Q", r.take(8))[0]
        n_groups = r.varint()
        if not 0 < n_groups <= 256:
            raise CorruptUpdate("bad group count")
        groups, exps = [], set()
        for _ in range(n_groups):
            g = Group(r.u8(), r.varint(), r.u8())
            if g.exponent in exps or g.exponent not in table_exps:
                raise CorruptUpdate("bad group exponent")
            if not 0 < g.count <= numel or g.rice_r > MAX_RICE:
                raise CorruptUpdate("bad group header")
            exps.add(g.exponent)
            groups.append(g)
            total_blocks += n_blocks(g.count)
        tensors.append(TensorHeader(name, numel, chk, groups))

    if 4 * total_blocks > r.remaining():
        raise CorruptUpdate("block table longer than the update")
    block_bits = list(struct.unpack(f"<{total_blocks}I", r.take(4 * total_blocks)))
    payload = r.take(r.varint())
    if r.remaining():
        raise CorruptUpdate("trailing bytes after the payload")
    if sum(block_bits) > 8 * len(payload):
        raise CorruptUpdate("payload shorter than its block table")
    return Update(tables, tensors, block_bits, payload)
