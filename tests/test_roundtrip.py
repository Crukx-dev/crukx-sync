import numpy as np
import pytest

from crukx_sync import CorruptUpdate, apply_update, encode_update


def bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bit patterns (round to nearest even)."""
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def rl_like_pair(n=200_000, lr=1e-6, seed=0):
    """Weights and the next broadcast after one Adam-like step (most bf16 values unchanged)."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.02, n).astype(np.float32)
    step = lr * np.sign(rng.normal(size=n)) * rng.uniform(0.5, 1.5, n)
    return bf16_bits(w), bf16_bits(w + step.astype(np.float32))


def roundtrip(old, new):
    upd = encode_update(old, new)
    got = {k: v.copy() for k, v in old.items()}
    stats = apply_update(got, upd)
    for k in new:
        assert np.array_equal(got[k], new[k]), k
    return upd, stats


def test_rl_update_is_exact_and_small():
    a, b = rl_like_pair()
    old, new = {"w": a.reshape(400, 500)}, {"w": b.reshape(400, 500)}
    upd, stats = roundtrip(old, new)
    changed = int((a != b).sum())
    assert 0 < changed < a.size
    assert stats.changed == changed
    assert len(upd) / changed < 1.5, "should be well under verl's 6 bytes per change"


def test_many_tensors_and_unchanged_ones_are_skipped():
    a, b = rl_like_pair(50_000, seed=1)
    old = {"a": a, "same": a.copy(), "c": a[:777].copy()}
    new = {"a": b, "same": a.copy(), "c": b[:777].copy()}
    _, stats = roundtrip(old, new)
    assert stats.tensors == 2


def test_no_changes_is_a_tiny_update():
    a, _ = rl_like_pair(1000)
    upd, stats = roundtrip({"w": a}, {"w": a.copy()})
    assert stats.tensors == 0 and len(upd) == 6


def test_every_element_changed_and_large_jumps():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 65536, 5000, dtype=np.uint16)
    b = a ^ rng.integers(1, 65536, 5000, dtype=np.uint16)  # every element, arbitrary bits
    roundtrip({"w": a}, {"w": b})


def test_single_change_and_single_element_tensor():
    a = np.array([0x3F80], dtype=np.uint16)
    roundtrip({"w": a}, {"w": np.array([0x3F81], dtype=np.uint16)})


def test_corrupt_bytes_are_rejected_and_old_is_untouched():
    a, b = rl_like_pair(20_000, seed=3)
    upd = bytearray(encode_update({"w": a}, {"w": b}))
    for i in (len(upd) // 2, len(upd) - 1):
        bad = bytearray(upd)
        bad[i] ^= 0xFF
        old = {"w": a.copy()}
        with pytest.raises(CorruptUpdate):
            apply_update(old, bytes(bad))
        assert np.array_equal(old["w"], a)
    with pytest.raises(CorruptUpdate):
        apply_update({"w": a.copy()}, bytes(upd[:-3]))
    with pytest.raises(CorruptUpdate):
        apply_update({"w": a.copy()}, b"nope" + bytes(upd[4:]))


def test_mismatched_receiver_is_an_error():
    a, b = rl_like_pair(1000)
    upd = encode_update({"w": a}, {"w": b})
    with pytest.raises(ValueError):
        apply_update({"other": a.copy()}, upd)
    with pytest.raises(ValueError):
        apply_update({"w": a[:999].copy()}, upd)
    with pytest.raises(ValueError):
        encode_update({"w": a}, {"w": b.astype(np.float32)})
    with pytest.raises(ValueError):
        encode_update({"w": a}, {"x": b})


# ------------------------------------------------------------ edge cases and hostile input

def test_group_with_more_distinct_residuals_than_15_bit_codes():
    rng = np.random.default_rng(4)
    a = np.full(200_000, 0x3C00, dtype=np.uint16)  # one exponent group
    b = rng.integers(0, 65536, a.size, dtype=np.uint16)
    roundtrip({"w": a}, {"w": b})


def test_rice_edge_cases():
    a = np.full(1 << 20, 0x3F80, dtype=np.uint16)
    dense = a.copy(); dense[:4096] += 1          # all gaps 0
    tail = a.copy(); tail[-1] += 1               # one huge gap
    roundtrip({"w": a}, {"w": dense})
    roundtrip({"w": a}, {"w": tail})


def test_apply_writes_through_to_caller_views():
    a, b = rl_like_pair(10_000, seed=5)
    buf = a.copy()
    view = buf[:]  # e.g. a numpy view of a torch tensor
    apply_update({"w": buf}, encode_update({"w": a}, {"w": b}))
    assert np.array_equal(view, b)


def test_every_single_byte_flip_is_rejected_cleanly():
    a, b = rl_like_pair(3000, seed=6)
    upd = encode_update({"w": a}, {"w": b})
    for i in range(len(upd)):
        for x in (0x01, 0x80, 0xFF):
            bad = bytearray(upd); bad[i] ^= x
            old = {"w": a.copy()}
            try:
                apply_update(old, bytes(bad))
            except (CorruptUpdate, ValueError):
                assert np.array_equal(old["w"], a)
            else:  # a flip that still decodes must still verify
                assert np.array_equal(old["w"], b)


def _header(numel, groups_bytes):
    from crukx_sync.wire import write_varint
    out = bytearray(b"CRXS\x01"); write_varint(out, 1)
    write_varint(out, 1); out += b"w"; write_varint(out, numel); out += bytes(8)
    out += groups_bytes
    return bytes(out)


def test_hostile_headers_fail_fast():
    from crukx_sync.wire import write_varint
    a = np.zeros(100, dtype=np.uint16)
    cases = []
    g = bytearray(b"\x01\x00"); write_varint(g, 1 << 62); g += b"\x00"; write_varint(g, 1 << 45)
    cases.append(g)                                  # huge count / n_sym
    cases.append(bytearray(b"\x01\x00") + b"\xff" * 9 + b"\x7f")  # 10-byte varint >= 2^63
    g = bytearray(b"\x01\x00\x05\x1f\x01\x00\x01")   # rice_r = 31
    cases.append(g)
    g = bytearray(b"\x00")                           # zero groups
    cases.append(g)
    for c in cases:
        with pytest.raises(CorruptUpdate):
            apply_update({"w": a.copy()}, _header(100, bytes(c)))
