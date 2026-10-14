# crukx-sync update format, version 2

An update turns one set of bf16 weights (`old`) into the next (`new`) exactly. It is built for
RL weight sync: between consecutive broadcasts only ~0.5–3% of bf16 weights change, and most
changes are one unit in the last place. Evidence: `crukx-research/studies/2026-10-14-r1-rl-step-deltas`.

All integers are little-endian. `varint` is unsigned LEB128. `zigzag(v) = 2v` for `v ≥ 0`, `-2v - 1` for `v < 0`.

```
update   := "CRXS" version:u8(=2) n_tables:varint table* n_tensors:varint tensor*
            block_bits:u32*n_blocks payload_len:varint payload:bytes
table    := exponent:u8 n_sym:varint (sym:varint len:u8)*n_sym
tensor   := name_len:varint name:utf8 numel:varint checksum:u64 n_groups:varint group*
group    := exponent:u8 count:varint rice_r:u8
```

Only tensors with at least one changed element are present.

## Groups (context the receiver already has)

Elements of a tensor are taken in the order `argsort(exponent(old), stable)`, where
`exponent(x) = (bits(x) >> 7) & 0xFF`. The receiver computes the same order from its own copy of
`old`, so no position map is sent. Elements with the same old exponent form a group; small-exponent
weights have tiny bf16 steps and change far more often, which is what makes this context pay.

Each present group lists its exponent, the number of changed elements `count` and a Golomb-Rice
parameter `rice_r`. Residuals are coded with the update-level Huffman table of the group's exponent
(one table per exponent, shared by all tensors), over `zigzag(new_bits - old_bits)` given as
(symbol, code length) pairs. The difference is taken on the bits as signed int16 without wrapping,
so it lies in ±65535 and symbols in 0–131070. Codes are assigned canonically in order of
(length, position in the table); the encoder lists symbols in ascending order.

## Symbols and blocks

For every changed element, in group order then position order:

1. `gap` = positions skipped in the group since the previous change (first: the group-local index),
   Golomb-Rice coded with `rice_r`: `gap >> r` ones, a zero, then the low `r` bits, MSB first.
2. `residual` = `zigzag(new_bits - old_bits)` with bits as signed int16, canonical Huffman coded.

Each group's elements are cut into blocks of 256 (the last may be shorter). `block_bits` gives the
bit length of every block of the update (tensors in order, then groups in order), so every block of
every tensor can be decoded in parallel. The payload is all blocks' bits concatenated, MSB first,
zero-padded to a byte.

## Limits (a receiver rejects anything outside them)

- tensor names unique within an update; `numel` equals the receiver's tensor size
- `n_tables ≤ 256`, table exponents unique; `1 ≤ n_groups ≤ 256`, group exponents unique within a
  tensor and each with a table
- `1 ≤ count ≤` the number of elements with that old exponent
- `0 ≤ rice_r ≤ 30`; every gap smaller than its group
- every table used by some group; `1 ≤ n_sym ≤` total changes with that exponent; symbols unique
  and ≤ 131070, code lengths 1–17, Kraft sum ≤ 1
- per tensor, total changes ≤ `numel`; every block at least `changes × (2 + rice_r)` bits
- varints < 2^63; `payload_len = ceil(sum(block_bits) / 8)`, padding bits zero; every block decodes
  exactly to its length

## Integrity

`checksum` covers the reconstructed tensor: `sum(u_i * w_i) mod 2^64` with `u_i` the uint16 bits
and `w_i = splitmix64(i) | 1`, where `splitmix64(i)`: `z = (i+1) * 0x9E3779B97F4A7C15`;
`z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9`; `z = (z ^ (z >> 27)) * 0x94D049BB133111EB`;
`z ^ (z >> 31)` (all mod 2^64, logical shifts). The weights are non-linear in `i` so that no
structured error (e.g. two opposite changes shifted by the same offset) can cancel. A receiver
verifies incrementally: old checksum plus the changed elements' differences. A receiver must reject the update (and keep
`old`) if any field is out of range, the payload does not decode exactly, or the checksum differs.
It is an error-detection code, not a signature; authenticate the transport separately.
