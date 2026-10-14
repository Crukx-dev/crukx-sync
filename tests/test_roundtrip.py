import pytest
import torch

from crukx_sync import CorruptUpdate, apply_update, encode_update
from crukx_sync.wire import write_varint

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def rl_like_pair(n=200_000, lr=1e-6, seed=0, device="cpu"):
    """bf16 weights and the next broadcast after one Adam-like step (most values unchanged)."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, generator=g) * 0.02
    step = lr * torch.sign(torch.randn(n, generator=g)) * (0.5 + torch.rand(n, generator=g))
    return w.bfloat16().to(device), (w + step).bfloat16().to(device)


def roundtrip(old, new, apply_device=None):
    upd = encode_update(old, new)
    got = {k: v.clone().to(apply_device or v.device) for k, v in old.items()}
    stats = apply_update(got, upd)
    for k in new:
        assert torch.equal(got[k].cpu().view(torch.int16), new[k].cpu().view(torch.int16)), k
    return upd, stats


@pytest.mark.parametrize("dev", DEVICES)
def test_rl_update_is_exact_and_small(dev):
    a, b = rl_like_pair(device=dev)
    upd, stats = roundtrip({"w": a.view(400, 500)}, {"w": b.view(400, 500)})
    changed = int((a.view(torch.int16) != b.view(torch.int16)).sum())
    assert 0 < changed < a.numel() and stats.changed == changed
    assert len(upd) / changed < 1.5, "should be well under verl's 6 bytes per change"


def test_encode_on_one_device_apply_on_another():
    if "cuda" not in DEVICES:
        pytest.skip("no GPU")
    a, b = rl_like_pair(device="cuda")
    roundtrip({"w": a}, {"w": b}, apply_device="cpu")


@pytest.mark.parametrize("dev", DEVICES)
def test_many_tensors_and_unchanged_ones_are_skipped(dev):
    a, b = rl_like_pair(50_000, seed=1, device=dev)
    old = {"a": a, "same": a.clone(), "c": a[:777].clone()}
    new = {"a": b, "same": a.clone(), "c": b[:777].clone()}
    _, stats = roundtrip(old, new)
    assert stats.tensors == 2


def test_no_changes():
    a, _ = rl_like_pair(1000)
    _, stats = roundtrip({"w": a}, {"w": a.clone()})
    assert stats.tensors == 0 and stats.changed == 0


def test_every_element_changed_with_arbitrary_bits():
    g = torch.Generator().manual_seed(2)
    a = torch.randint(-32768, 32768, (5000,), generator=g, dtype=torch.int16)
    b = a ^ torch.randint(1, 32768, (5000,), generator=g, dtype=torch.int16)
    roundtrip({"w": a}, {"w": b})


def test_group_with_more_distinct_residuals_than_15_bit_codes():
    a = torch.full((200_000,), 0x3C00, dtype=torch.int16)  # one exponent group
    b = torch.randint(-32768, 32768, (200_000,), generator=torch.Generator().manual_seed(4), dtype=torch.int16)
    roundtrip({"w": a}, {"w": b})


def test_rice_edge_cases():
    a = torch.full((1 << 20,), 0x3F80, dtype=torch.int16)
    dense, tail = a.clone(), a.clone()
    dense[:4096] += 1   # all gaps 0
    tail[-1] += 1       # one huge gap
    roundtrip({"w": a}, {"w": dense})
    roundtrip({"w": a}, {"w": tail})


def test_single_element_tensor():
    roundtrip({"w": torch.tensor([0x3F80], dtype=torch.int16)}, {"w": torch.tensor([0x3F81], dtype=torch.int16)})


def test_apply_writes_through_to_views():
    a, b = rl_like_pair(10_000, seed=5)
    buf = a.clone()
    view = buf[:]
    apply_update({"w": buf}, encode_update({"w": a}, {"w": b}))
    assert torch.equal(view.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("dev", DEVICES)
def test_every_single_byte_flip_is_rejected_cleanly(dev):
    a, b = rl_like_pair(3000, seed=6, device=dev)
    upd = encode_update({"w": a}, {"w": b})
    for i in range(len(upd)):
        for x in (0x01, 0x80, 0xFF):
            bad = bytearray(upd)
            bad[i] ^= x
            old = {"w": a.clone()}
            try:
                apply_update(old, bytes(bad))
            except (CorruptUpdate, ValueError):
                assert torch.equal(old["w"].view(torch.int16), a.view(torch.int16))
            else:  # a flip that still decodes must still verify
                assert torch.equal(old["w"].view(torch.int16), b.view(torch.int16))


def _update(tables: bytes, tensor_tail: bytes) -> bytes:
    out = bytearray(b"CRXS\x02") + tables
    write_varint(out, 1)
    write_varint(out, 1)
    out += b"w"
    write_varint(out, 100)
    out += bytes(8)
    return bytes(out + tensor_tail)


def test_hostile_headers_fail_fast():
    a = torch.zeros(100, dtype=torch.int16)
    one_table = b"\x01\x00\x01\x02\x01"           # 1 table: exponent 0, 1 symbol (2), length 1
    huge = bytearray(b"\x01\x00")
    write_varint(huge, 1 << 62)
    cases = [
        _update(b"\x01\x00" + b"\xff" * 9 + b"\x7f", b""),        # varint >= 2^63
        _update(one_table, b"\x01\x00" + bytes(huge) + b"\x00"),   # huge count
        _update(one_table, b"\x01\x00\x05\x1f"),                   # rice_r = 31
        _update(one_table, b"\x00"),                               # zero groups
        _update(b"\x01\x00\x02\x01\x01\x02\x01", b""),             # Kraft overflow
        _update(one_table, b"\x02\x00\x05\x00\x00\x05\x00"),       # duplicate exponent
        _update(b"\x01\x00\x01\x02\x01", b"\x01\x07\x05\x00"),     # group without a table
    ]
    for c in cases:
        with pytest.raises(CorruptUpdate):
            apply_update({"w": a.clone()}, c)


def test_mismatched_receiver_is_an_error():
    a, b = rl_like_pair(1000)
    upd = encode_update({"w": a}, {"w": b})
    for bad in ({"other": a.clone()}, {"w": a[:999].clone()}):
        with pytest.raises(ValueError):
            apply_update(bad, upd)
    with pytest.raises(ValueError):
        encode_update({"w": a}, {"w": b.float()})
    with pytest.raises(ValueError):
        encode_update({"w": a}, {"x": b})


@pytest.mark.parametrize("dev", DEVICES)
def test_checksum_cache_over_many_steps(dev):
    """Sender and receiver keep checksum caches; a run of updates stays exact."""
    g = torch.Generator().manual_seed(7)
    w = torch.randn(300_000, generator=g) * 0.02
    sender, receiver = {}, {}
    prev = {"w": w.bfloat16().to(dev)}
    recv = {"w": prev["w"].clone()}
    for step in range(5):
        w = w + 1e-6 * torch.sign(torch.randn(w.shape, generator=g))
        cur = {"w": w.bfloat16().to(dev)}
        apply_update(recv, encode_update(prev, cur, sender), receiver)
        assert torch.equal(recv["w"].view(torch.int16), cur["w"].view(torch.int16))
        prev = cur
    assert sender == {k: v[0] for k, v in receiver.items()}


def test_wrong_receiver_cache_is_rejected_not_silently_wrong():
    a, b = rl_like_pair(20_000, seed=8)
    upd = encode_update({"w": a}, {"w": b})
    old = {"w": a.clone()}
    with pytest.raises(CorruptUpdate):
        apply_update(old, upd, {"w": (12345, old["w"]._version)})
    assert torch.equal(old["w"].view(torch.int16), a.view(torch.int16))


@pytest.mark.parametrize("dev", DEVICES)
def test_receiver_cache_does_not_hide_weights_that_drifted(dev):
    """Weights modified outside the library: the cache must not vouch for them."""
    g = torch.Generator().manual_seed(9)
    w = torch.randn(50_000, generator=g) * 0.02
    s1, s2 = w.bfloat16().to(dev), (w + 1e-6).bfloat16().to(dev)
    s3 = (w + 2e-6).bfloat16().to(dev)
    sender, receiver = {}, {}
    recv = {"w": s1.clone()}
    apply_update(recv, encode_update({"w": s1}, {"w": s2}, sender), receiver)
    recv["w"].view(torch.int16)[7] ^= 1  # drift: one mantissa bit flipped outside the library
    with pytest.raises(CorruptUpdate):
        apply_update(recv, encode_update({"w": s2}, {"w": s3}, sender), receiver)


def test_sender_cache_is_not_committed_when_encoding_fails(monkeypatch):
    import crukx_sync
    a, b = rl_like_pair(20_000, seed=10)
    cache = {}
    def boom(_):
        raise RuntimeError("serialize failed")
    monkeypatch.setattr(crukx_sync, "serialize", boom)
    with pytest.raises(RuntimeError):
        encode_update({"w": a}, {"w": b}, cache)
    assert cache == {}


@pytest.mark.parametrize("n", [4096, 4097, 3 * 4096 + 5])
def test_multi_slab_paths_and_slab_boundaries(monkeypatch, n):
    import crukx_sync.codec as codec
    monkeypatch.setattr(codec, "_SLAB", 4096)
    for dev in DEVICES:
        a, b = rl_like_pair(n, seed=n, device=dev)
        roundtrip({"w": a}, {"w": b})
        c = a.clone()
        c[-1] = b[-1] if not torch.equal(a[-1:], b[-1:]) else c[-1]
        roundtrip({"w": a}, {"w": c})


def test_extreme_residuals():
    a = torch.tensor([0x7FFF, -0x8000, 0, 0x7FFF], dtype=torch.int16)
    b = torch.tensor([-0x8000, 0x7FFF, 0x7FFF, 0], dtype=torch.int16)
    roundtrip({"w": a}, {"w": b})


def test_hostile_counts_are_rejected_before_allocating():
    """A short update claiming millions of changes must fail in parse, not allocate."""
    big = torch.zeros(1 << 22, dtype=torch.int16)
    out = bytearray(b"CRXS\x02")
    write_varint(out, 1)
    out += b"\x00\x01\x02\x01"            # table: exponent 0, 1 symbol (2), length 1
    write_varint(out, 1)
    write_varint(out, 1)
    out += b"w"
    write_varint(out, big.numel())
    out += bytes(8)
    write_varint(out, 1)
    out += b"\x00"
    write_varint(out, big.numel())         # every element "changed"
    out += b"\x00"
    out += bytes(4 * (big.numel() // 256))  # all blocks claim 0 bits
    write_varint(out, 0)
    with pytest.raises(CorruptUpdate):
        apply_update({"w": big}, bytes(out))


def test_encode_rejects_mixed_devices():
    if "cuda" not in DEVICES:
        pytest.skip("no GPU")
    a, b = rl_like_pair(1000)
    with pytest.raises(ValueError):
        encode_update({"w": a, "x": a.cuda()}, {"w": b, "x": b.cuda()})


def test_encode_on_cpu_apply_on_gpu():
    if "cuda" not in DEVICES:
        pytest.skip("no GPU")
    a, b = rl_like_pair()
    roundtrip({"w": a}, {"w": b}, apply_device="cuda")


@pytest.mark.parametrize("dev", DEVICES)
def test_long_unary_run_takes_the_exact_slow_path(dev):
    """Many small gaps and one huge one in the same group: the Rice parameter stays small, so one
    unary run is far longer than a 64-bit window."""
    a = torch.full((1 << 20,), 0x3F80, dtype=torch.int16, device=dev)
    b = a.clone()
    b[:2000] += 1          # dense changes: gaps of 0
    b[-1] += 1             # then one gap of ~1M
    upd, _ = roundtrip({"w": a}, {"w": b})
    from crukx_sync.wire import parse
    assert parse(upd).tensors[0].groups[0].rice_r < 10  # so the long gap's unary run is >> 64 bits
