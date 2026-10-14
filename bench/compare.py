"""Other formats' size for the same update, implemented from their public descriptions.

Every encoder sees the identical (old, new) pair crukx-sync encodes. Where a description is
ambiguous we pick the choice most favourable to that format, as noted per function.

- raw:       int32 position + bf16 value per change (verl `delta_sharded`, vLLM `SparseWeightPatch`,
             TRL sparse safetensors). Uncompressed; per-tensor headers ignored (favourable).
- nemo:      NeMo RL sparse refit: sorted positions delta-coded at the smallest lossless width of
             16/32/64 bits per tensor, values as XOR with the old bits, zstd level 1 over the payload.
             Range encoding of contiguous runs is not modelled (runs are rare at ~1% density).
- pulse:     PULSE / PULSESync: delta-coded positions (same widths) + raw bf16 values, zstd level 1.
- sparserl:  SparseRL-Sync: 16-bit position deltas (32-bit where a gap does not fit) + bf16 values
             entropy-coded; values counted at their ideal order-0 entropy (a lower bound for them).
"""

from __future__ import annotations

import numpy as np
import torch
import zstandard


def _width(deltas: np.ndarray) -> np.ndarray:
    m = int(deltas.max()) if deltas.size else 0
    for dt in (np.uint16, np.uint32, np.uint64):
        if m <= np.iinfo(dt).max:
            return deltas.astype(dt)
    raise AssertionError


def sizes(old: dict[str, torch.Tensor], new: dict[str, torch.Tensor]) -> dict[str, int]:
    pos_parts, xor_parts, val_parts = [], [], []
    raw = 0
    sparserl_pos = 0
    all_vals = []
    for name in old:
        a = old[name].view(torch.int16).view(-1)
        b = new[name].view(torch.int16).view(-1)
        idx = torch.nonzero(a != b).flatten()
        if idx.numel() == 0:
            continue
        i = idx.cpu().numpy()
        va = a[idx].cpu().numpy()
        vb = b[idx].cpu().numpy()
        raw += i.size * (4 + 2)
        d = np.diff(i, prepend=0)                     # first entry absolute, then gaps
        pos_parts.append(_width(d).tobytes())
        xor_parts.append((va ^ vb).tobytes())
        val_parts.append(vb.tobytes())
        sparserl_pos += 2 * i.size + 4 * int((d > 0xFFFF).sum())   # 16-bit, escapes to 32-bit
        all_vals.append(vb.view(np.uint16))
    if not pos_parts:
        return {"raw": 0, "nemo": 0, "pulse": 0, "sparserl": 0}
    z1 = zstandard.ZstdCompressor(level=1)
    pos = b"".join(pos_parts)
    nemo = len(z1.compress(pos + b"".join(xor_parts)))
    pulse = len(z1.compress(pos + b"".join(val_parts)))
    v = np.concatenate(all_vals)
    counts = np.bincount(v, minlength=65536)
    p = counts[counts > 0] / v.size
    sparserl = sparserl_pos + int(np.ceil(-(p * np.log2(p)).sum() * v.size / 8))
    return {"raw": raw, "nemo": nemo, "pulse": pulse, "sparserl": sparserl}
