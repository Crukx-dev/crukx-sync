"""crukx-sync: exact, compact bf16 weight updates for RL weight sync.

    update = encode_update(prev, cur)   # trainer: dicts of name -> bf16 torch tensors (CPU or GPU)
    apply_update(weights, update)       # worker: writes the new values into `weights` in place

Nothing is written unless the whole update decodes and every tensor's checksum verifies.

For per-step sync, pass a `checksums` dict on each side and keep it between steps: checksums are
then updated from the changed elements only instead of re-reading every tensor. A stale or wrong
cache makes the next update fail verification (CorruptUpdate); it never yields wrong weights.
Format: docs/FORMAT.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from . import codec
from .wire import CorruptUpdate, parse, serialize

__all__ = ["encode_update", "apply_update", "UpdateStats", "CorruptUpdate"]


@dataclass
class UpdateStats:
    tensors: int
    changed: int
    bytes: int


def encode_update(old: Mapping[str, torch.Tensor], new: Mapping[str, torch.Tensor],
                  checksums: dict[str, int] | None = None) -> bytes:
    """Bytes that turn `old` into `new` exactly. Same names and shapes; tensors are bf16 or int16."""
    if old.keys() != new.keys():
        raise ValueError(f"old and new differ in tensor names: {sorted(old.keys() ^ new.keys())[:5]}")
    with torch.no_grad():
        return serialize(codec.encode(old, new, checksums))


def apply_update(weights: Mapping[str, torch.Tensor], update: bytes,
                 checksums: dict[str, int] | None = None) -> UpdateStats:
    """Write the update into `weights` in place; on any error nothing is written."""
    u = parse(update)
    with torch.no_grad():
        changed = codec.apply(weights, u, checksums)
    return UpdateStats(len(u.tensors), changed, len(update))
