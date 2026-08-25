# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""RFC 6979 — deterministic nonce generation for ECDSA.

Host-only on purpose. Nonce generation happens on the signing path, which runs
concretely (`docs/reference/security.md`), and it is bytes-and-integers work
with no batch axis — so everything here is Python integers and `bytes`, the
representation the RFC itself speaks.

HMAC is the standard library's, over a `hashlib`-style constructor the caller
injects. hash-frx's `Hmac` is the array construction — it takes a `ByteHash` and
a `[B, L]` batch — and there is no batch here to give it; its host rows wrap
`hashlib` besides, so the stdlib construction *is* what this path would reach
through one more layer. The injection matters beyond taste: RFC 6979
§3.2 requires the HMAC hash to be the same `H` that hashed the message, and a
chain variant that swaps the message hash swaps this one with it. The pairing
cannot drift silently — the known-answer `k` values bind the two together.

The transforms carry the RFC's own names (`bits2int`, `int2octets`,
`bits2octets`) because the document defines them as a trio whose asymmetry is
load-bearing (§2.3.5 notes int2octets is *not* the inverse of bits2int).
"""

from __future__ import annotations

import hmac
from collections.abc import Callable, Iterator
from typing import Any

# A `hashlib`-style hash constructor — what `hmac.new` takes as digestmod.
# Variadic because hashlib constructors take an optional initial message.
HashConstructor = Callable[..., Any]


def bits2int(data: bytes, qlen: int) -> int:
    """RFC 6979 §2.3.2: the leftmost `qlen` bits of `data`, as an integer."""
    value = int.from_bytes(data, "big")
    blen = 8 * len(data)
    if blen > qlen:
        value >>= blen - qlen
    return value


def int2octets(value: int, qlen: int) -> bytes:
    """RFC 6979 §2.3.3: `value` big-endian over `rlen = 8·ceil(qlen/8)` bits."""
    return value.to_bytes((qlen + 7) // 8, "big")


def bits2octets(data: bytes, q: int, qlen: int) -> bytes:
    """RFC 6979 §2.3.4: `bits2int(data) mod q`, re-encoded to `rlen` bits."""
    return int2octets(bits2int(data, qlen) % q, qlen)


def nonces(
    q: int, x: int, h1: bytes, hash_constructor: HashConstructor
) -> Iterator[int]:
    """Candidate nonces `k ∈ [1, q-1]` for signing `h1` under key `x` — §3.2.

    An iterator rather than a value because the RFC's step h.3 loops: a
    candidate that yields `r = 0` or `s = 0` is discarded by the *signing*
    algorithm, and the generator state (`K`, `V`) carries into the next draw.
    The first yielded value is the `k` the RFC's test vectors publish.

    The body follows RFC 6979 §3.2's lettered steps b through h in order; the
    `0x00` / `0x01` separators and the two-phase key derivation are the
    standard's, not choices made here.
    """
    qlen = q.bit_length()
    hlen = hash_constructor().digest_size

    def mac(key: bytes, message: bytes) -> bytes:
        return hmac.new(key, message, hash_constructor).digest()

    suffix = int2octets(x, qlen) + bits2octets(h1, q, qlen)
    v = b"\x01" * hlen
    k = b"\x00" * hlen
    k = mac(k, v + b"\x00" + suffix)
    v = mac(k, v)
    k = mac(k, v + b"\x01" + suffix)
    v = mac(k, v)
    while True:
        t = b""
        while 8 * len(t) < qlen:
            v = mac(k, v)
            t += v
        candidate = bits2int(t, qlen)
        # Compared to q, never reduced modulo q — a reduction would bias k.
        if 1 <= candidate <= q - 1:
            yield candidate
        k = mac(k, v + b"\x00")
        v = mac(k, v)
