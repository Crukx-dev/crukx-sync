# crukx-sync

Exact, compact bf16 weight updates for RL weight sync. The trainer sends only what changed since
the last broadcast; the inference worker rebuilds the new weights bit-for-bit and verifies them.

```python
from crukx_sync import encode_update, apply_update

update = encode_update(prev, cur)   # trainer: dicts of name -> uint16 bf16 bits
apply_update(weights, update)       # worker: overwrites `weights` in place, or raises and leaves it untouched
```

From torch: `t.view(torch.int16).cpu().numpy().view(np.uint16)`.

## Why it is small

Between RL steps only ~0.5–3% of bf16 weights change, and ~87% of those move by one unit in the last
place. The receiver already knows every old weight, and the old weight's exponent predicts both
*whether* it changes and *by how much*. So positions are coded per old-exponent group (Golomb-Rice
gaps) and values as the difference from the old value (Huffman), in independent 256-symbol blocks.
Spec: [docs/FORMAT.md](docs/FORMAT.md). Research: `crukx-research/studies/2026-10-14-r1-rl-step-deltas`.

## Measured (real GRPO, Qwen2.5-1.5B-Instruct, lr 1e-6, 8 steps, RTX 3090 host)

| per update | full bf16 broadcast | verl sparse format | crukx-sync |
|---|---|---|---|
| bytes | 3.09 GB | 56–80 MB | **6.3–9.9 MB** (0.31–0.66 B per changed weight) |
| vs crukx-sync | 310–490x larger | 9–19x larger | exact, checksum-verified |

Speed: this is the **reference implementation** (numpy, CPU): ~13 s encode and ~15 s apply at 1.5B.
Too slow for per-step sync; a GPU kernel is next. Use it to check correctness and sizes.

## Develop

```bash
pip install -e '.[test]'
pytest -q                       # round trips, edge cases, every single-byte corruption, hostile headers
python bench/grpo.py 1e-6 8 out # real GRPO benchmark (torch, transformers, CUDA GPU)
```

Tested on numpy 1.26, 2.2 and 2.5.

## Layout

- `src/crukx_sync/__init__.py`: public API (`encode_update`, `apply_update`, `CorruptUpdate`) and the container
- `src/crukx_sync/codec.py`: per-tensor encode/decode (grouping, Golomb-Rice, canonical Huffman, blocks, checksum)
- `src/crukx_sync/wire.py`: varints and a bounds-checked reader
- `docs/FORMAT.md`: wire format v1 and the limits a receiver enforces
