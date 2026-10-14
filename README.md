# crukx-sync

Exact, compact bf16 weight updates for RL weight sync. The trainer sends only what changed since
the last broadcast; the inference worker rebuilds the new weights bit-for-bit and verifies them.

```python
from crukx_sync import encode_update, apply_update

send_chk, recv_chk = {}, {}                          # keep across steps
update = encode_update(prev, cur, send_chk)          # trainer: dicts of name -> bf16 torch tensors (CPU or GPU)
apply_update(weights, update, recv_chk)              # worker: overwrites `weights` in place, or raises and writes nothing
```

## Why it is small

Between RL steps only ~0.5–3% of bf16 weights change, and ~87% of those move by one unit in the last
place. The receiver already knows every old weight, and the old weight's exponent predicts both
*whether* it changes and *by how much*. So positions are coded per old-exponent group (Golomb-Rice
gaps) and values as the difference from the old value (Huffman), in independent 256-symbol blocks.
Spec: [docs/FORMAT.md](docs/FORMAT.md). Research: `crukx-research/studies/2026-10-14-r1-rl-step-deltas`.

## Measured (real GRPO, Qwen2.5-1.5B-Instruct, lr 1e-6, RTX 3090, steps 2–8)

| per update | full bf16 broadcast | verl sparse format | crukx-sync |
|---|---|---|---|
| bytes | 3.09 GB | 56–80 MB | **6.3–9.9 MB** (0.31–0.65 B per changed weight) |
| exact | — | yes | yes, checksum-verified |

GPU time (torch, no custom kernels, checksum caches on): **encode 0.61–0.77 s, apply 0.48–0.59 s**
per update at 1.5B (the training step is 6.7 s). First update ~1.5 s / 1.1 s (full checksums).
Where bandwidth binds this is already decisive: at 1 Gbit/s a 3.09 GB broadcast takes ~25 s,
crukx-sync ~0.1 s on the wire + ~1.2 s of codec. Next: fuse the ~330 small tensors and write CUDA
kernels for ranking and bit packing.

## Develop

```bash
pip install -e '.[test]'
pytest -q                       # round trips, edge cases, every single-byte corruption, hostile headers
python bench/grpo.py 1e-6 8 out # real GRPO benchmark (torch, transformers, CUDA GPU)
```

Tests run on CPU and CUDA; `_SLAB` can be set small to exercise the multi-slab paths.

## Layout

- `src/crukx_sync/__init__.py`: public API (`encode_update`, `apply_update`, `CorruptUpdate`)
- `src/crukx_sync/codec.py`: encode/apply in torch (exponent ranking, Golomb-Rice, canonical Huffman, blocks, checksum)
- `src/crukx_sync/wire.py`: container types, serialization and bounds-checked parsing
- `docs/FORMAT.md`: wire format v1 and the limits a receiver enforces
