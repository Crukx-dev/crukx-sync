"""Byte-level reading and writing for the update container (docs/FORMAT.md)."""

MAGIC = b"CRXS"
VERSION = 1


class CorruptUpdate(ValueError):
    """The update bytes are malformed, truncated, or do not reproduce the expected weights."""


def write_varint(out: bytearray, v: int) -> None:
    if v < 0:
        raise ValueError(f"varint must be non-negative, got {v}")
    while v >= 0x80:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v)


class Reader:
    """Sequential reader over update bytes; every overrun is a CorruptUpdate."""

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

    def u64(self) -> int:
        return int.from_bytes(self.take(8), "little")

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

    def done(self) -> bool:
        return self.pos == len(self.data)
