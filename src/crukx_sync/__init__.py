"""crukx-sync: exact, compact bf16 weight updates for RL weight sync.

    update = encode_update(old, new)   # trainer side
    apply_update(old, update)          # inference side: old's arrays are overwritten with new

Weights are dicts of name -> bf16 bit patterns as numpy uint16 arrays (from torch:
``t.view(torch.int16).cpu().numpy().view(np.uint16)``). Format: docs/FORMAT.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from . import codec
from .wire import MAGIC, VERSION, CorruptUpdate, Reader, write_varint

__all__ = ["encode_update", "apply_update", "UpdateStats", "CorruptUpdate"]


@dataclass
class UpdateStats:
    tensors: int
    changed: int
    bytes: int


def _flat(name: str, a: np.ndarray) -> np.ndarray:
    if not isinstance(a, np.ndarray) or a.dtype != np.uint16:
        raise ValueError(f"{name}: expected a numpy uint16 array of bf16 bits, got {getattr(a, 'dtype', type(a))}")
    return a.reshape(-1)


def encode_update(old: Mapping[str, np.ndarray], new: Mapping[str, np.ndarray]) -> bytes:
    """Bytes that turn `old` into `new` exactly. Both must have the same names and shapes."""
    if old.keys() != new.keys():
        raise ValueError(f"old and new differ in tensor names: {sorted(old.keys() ^ new.keys())[:5]}")
    parts = []
    for name in old:
        a, b = _flat(name, old[name]), _flat(name, new[name])
        if a.shape != b.shape:
            raise ValueError(f"{name}: shape changed {old[name].shape} -> {new[name].shape}")
        upd = codec.encode_tensor(a, b)
        if upd is not None:
            parts.append((name, upd))
    out = bytearray(MAGIC)
    out.append(VERSION)
    write_varint(out, len(parts))
    for name, upd in parts:
        raw = name.encode()
        write_varint(out, len(raw))
        out += raw
        write_varint(out, upd.numel)
        out += upd.checksum.to_bytes(8, "little")
        write_varint(out, len(upd.groups))
        for g in upd.groups:
            out.append(g.exponent)
            write_varint(out, g.count)
            out.append(g.rice_r)
            write_varint(out, len(g.symbols))
            for s, L in zip(g.symbols.tolist(), g.lengths.tolist()):
                write_varint(out, s)
                out.append(L)
        for n in upd.block_bits.tolist():
            write_varint(out, n)
        write_varint(out, len(upd.payload))
        out += upd.payload
    return bytes(out)


def _parse(data: bytes, old: Mapping[str, np.ndarray]):
    r = Reader(data)
    if r.take(4) != MAGIC:
        raise CorruptUpdate("not a crukx-sync update")
    if (v := r.u8()) != VERSION:
        raise CorruptUpdate(f"unsupported format version {v}")
    seen = set()
    for _ in range(r.varint()):
        try:
            name = r.take(r.varint()).decode("utf-8")
        except UnicodeDecodeError:
            raise CorruptUpdate("tensor name is not valid UTF-8") from None
        if name not in old:
            raise ValueError(f"update has tensor {name!r} that the receiver does not have")
        if name in seen:
            raise CorruptUpdate(f"tensor {name!r} appears twice")
        seen.add(name)
        numel, chk = r.varint(), r.u64()
        if numel != old[name].size:
            raise ValueError(f"{name}: update is for {numel} elements, receiver has {old[name].size}")
        groups = []
        n_groups = r.varint()
        if not 0 < n_groups <= 256:
            raise CorruptUpdate("bad group count")
        for _ in range(n_groups):
            exp, count, rice_r = r.u8(), r.varint(), r.u8()
            n_sym = r.varint()
            if not 0 < count <= numel:
                raise CorruptUpdate("group count exceeds tensor size")
            if not 0 < n_sym <= min(count, codec.MAX_SYMBOL + 1, r.remaining() // 2):
                raise CorruptUpdate("bad symbol table size")
            syms = np.empty(n_sym, dtype=np.int64)
            lens = np.empty(n_sym, dtype=np.uint8)
            for i in range(n_sym):
                syms[i], lens[i] = r.varint(), r.u8()
            groups.append(codec.Group(exp, count, rice_r, syms, lens))
        n_blocks = sum(-(-g.count // codec.BLOCK) for g in groups)
        if n_blocks > r.remaining():
            raise CorruptUpdate("block table longer than the update")
        block_bits = [r.varint() for _ in range(n_blocks)]
        if max(block_bits) > 8 * r.remaining():
            raise CorruptUpdate("block longer than the update")
        block_bits = np.array(block_bits, dtype=np.int64)
        payload = r.take(r.varint())
        yield name, codec.TensorUpdate(numel, chk, groups, block_bits, payload)
    if not r.done():
        raise CorruptUpdate("trailing bytes after the last tensor")


def apply_update(old: Mapping[str, np.ndarray], update: bytes) -> UpdateStats:
    """Overwrite `old`'s arrays with the updated weights. Nothing is written unless every tensor
    decodes and verifies, so on any error `old` is unchanged."""
    staged = {}
    for name, upd in _parse(update, old):
        staged[name] = codec.decode_tensor(_flat(name, old[name]), upd)
    changed = 0
    for name, flat in staged.items():
        target = old[name]
        changed += int(np.count_nonzero(flat != target.reshape(-1)))
        target[...] = flat.reshape(target.shape)  # writes through to any view the caller holds
    return UpdateStats(len(staged), changed, len(update))
