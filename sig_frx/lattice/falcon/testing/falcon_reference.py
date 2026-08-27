# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Falcon's algorithms, transcribed the way the specification writes them.

The modules under test reshape every one of these. Algorithm 18's cursor walk
becomes an associative scan over a nine-state machine, Algorithm 3's `while`
becomes a fixed candidate budget with a compaction, the ring product becomes an
NTT round trip, and Algorithm 16 becomes one `vmap`ped computation over a batch.
Each is a change made for the compiler, and it is the only thing about
`encoding.py` and `falcon.py` a reader has to take on trust — so this file takes
it back, looping one coefficient and one bit at a time over Python integers, and
the tests require the two to agree
([`testing.md`](../../../../docs/reference/testing.md)).

`Compress` is here as well as `Decompress`, even though the implementation has no
encoder yet ([#27](https://github.com/fractalyze/sig-frx/issues/27) brings one).
A decoder cannot be exercised over the space of *valid* inputs without something
that produces them, and the published vectors carry two signatures per degree —
which reaches none of the coefficient magnitudes that make `k` large. So the
encoder is the test's, and what it buys is `Decompress(Compress(s)) = s` over
coefficients drawn to stress the unary run rather than over the two that
upstream happened to publish.

Transcribed from the published document (§3.9, §3.10, §3.11), not from memory and
not from another implementation. Nothing here is written for speed: Python
integers have no width, which is exactly why this side is trustworthy and the
other side delegates its arithmetic to a field dtype.

The hashing half takes the standard library's SHAKE — the point being that
squeezing here asks for more bytes whenever it needs them, so the reference never
has a block bound to get wrong, and the implementation's budget is compared
against a stream that has none.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

Q = 12289

# Table 3.3, in full and in one place, so no test module transcribes a column of
# its own. `sbytelen` and the key lengths are the standard's stated values rather
# than the implementation's formulas — checking a formula against itself proves
# nothing.
PARAMETER_SETS: dict[str, dict[str, int]] = {
    "Falcon-512": {
        "n": 512,
        "squared_norm_bound": 34034726,
        "public_key_size": 897,
        "secret_key_size": 1281,
        "signature_size": 666,
    },
    "Falcon-1024": {
        "n": 1024,
        "squared_norm_bound": 70265242,
        "public_key_size": 1793,
        "secret_key_size": 2305,
        "signature_size": 1280,
    },
}


def parameter_cases() -> tuple[dict[str, Any], ...]:
    """Table 3.3 as `parameterized.parameters` records, name included."""
    return tuple({"name": name, **values} for name, values in PARAMETER_SETS.items())


def slen(signature_size: int) -> int:
    """Algorithm 16 line 2's `8 · sbytelen − 328`."""
    return 8 * signature_size - 328


def bits_of(data: bytes) -> list[int]:
    """§3.11.1 — a byte's leftmost bit has weight 128."""
    return [(byte >> shift) & 1 for byte in data for shift in range(7, -1, -1)]


def bytes_of(bits: list[int]) -> bytes:
    """The inverse of `bits_of`; `bits` must be a whole number of bytes."""
    return bytes(
        sum(bit << shift for bit, shift in zip(bits[i : i + 8], range(7, -1, -1)))
        for i in range(0, len(bits), 8)
    )


def compress(s: list[int], length: int) -> bytes | None:
    """Algorithm 17 — `None` where the specification returns `⊥`."""
    bits: list[int] = []
    for value in s:
        bits.append(1 if value < 0 else 0)
        magnitude = abs(value)
        bits.extend((magnitude >> j) & 1 for j in range(6, -1, -1))
        bits.extend([0] * (magnitude >> 7))
        bits.append(1)
    if len(bits) > length:
        return None
    return bytes_of(bits + [0] * (length - len(bits)))


def decompress(bits: list[int], length: int, n: int) -> list[int] | None:
    """Algorithm 18 — `None` where the specification returns `⊥`."""
    if len(bits) != length:
        return None
    out: list[int] = []
    cursor = 0
    for _ in range(n):
        if cursor + 8 > length:
            return None
        sign = bits[cursor]
        low = 0
        for j in range(7):
            low = (low << 1) | bits[cursor + 1 + j]
        k = 0
        while True:
            if cursor + 8 + k >= length:
                return None
            if bits[cursor + 8 + k] == 1:
                break
            k += 1
        value = (-1 if sign else 1) * (low + (k << 7))
        if value == 0 and sign == 1:
            return None
        out.append(value)
        cursor += 9 + k
    if any(bits[cursor:]):
        return None
    return out


def hash_to_point(message: bytes, n: int) -> list[int]:
    """Algorithm 3 — 16-bit big-endian draws, rejecting `t ≥ kq`."""
    k = (1 << 16) // Q
    shake = hashlib.shake_256(message)
    out: list[int] = []
    taken = 0
    while len(out) < n:
        taken += 64
        stream = shake.digest(taken)
        for i in range((taken - 64) // 2, taken // 2):
            if len(out) == n:
                break
            t = (stream[2 * i] << 8) | stream[2 * i + 1]
            if t < k * Q:
                out.append(t % Q)
    return out


def pk_decode(pk: bytes, n: int) -> list[int] | None:
    """§3.11.4 — `None` for a bad header or a coefficient at or above `q`."""
    if len(pk) != 1 + (14 * n + 7) // 8 or pk[0] != n.bit_length() - 1:
        return None
    bits = bits_of(pk[1:])
    out = []
    for i in range(n):
        value = 0
        for j in range(14):
            value = (value << 1) | bits[14 * i + j]
        if value >= Q:
            return None
        out.append(value)
    return out


def negacyclic_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`(a · b) mod (x^n + 1) mod q`, in exact integers.

    The wrap is a subtraction rather than an addition — that sign is the whole
    difference between this ring and the cyclic one.

    `int64` holds it exactly rather than nearly: a coefficient is under
    `q = 12289`, so a product is under `1.6e8` and a length-1024 sum under
    `1.6e11`, against `int64`'s `9.2e18`. That exactness is what makes this an
    oracle; the convolution is numpy's because nothing about *how* the exact
    integers are added is the property under test, and a `n²` Python loop would
    put a minute onto `arith_test`.

    Shared with [`arith_test`](arith_test.py), which is where this form comes
    from: two oracles for one ring is one edit away from gating the transform
    and the scheme against different rings.
    """
    n = len(a)
    full = np.convolve(a.astype(np.int64), b.astype(np.int64))
    wrapped = full[:n].copy()
    wrapped[: n - 1] -= full[n:]
    return wrapped % Q


def centered(w: list[int]) -> list[int]:
    """Residues as the `⌈−q/2⌉ … ⌊q/2⌋` representatives a norm is measured over."""
    return [x - Q if x > Q // 2 else x for x in w]


def verify(pk: bytes, message: bytes, signature: bytes, name: str) -> bool:
    """Algorithm 16 over §3.11.3's padded signature, verdict only."""
    params = PARAMETER_SETS[name]
    n = params["n"]
    if len(signature) != params["signature_size"]:
        return False
    if signature[0] != 0x30 | (n.bit_length() - 1):
        return False
    h = pk_decode(pk, n)
    if h is None:
        return False
    s2 = decompress(bits_of(signature[41:]), slen(params["signature_size"]), n)
    if s2 is None:
        return False
    c = hash_to_point(signature[1:41] + message, n)
    product = negacyclic_mul(np.array(s2) % Q, np.array(h))
    s1 = centered([(ci - pi) % Q for ci, pi in zip(c, product)])
    norm = sum(x * x for x in s1) + sum(x * x for x in s2)
    return norm <= params["squared_norm_bound"]


def exact_negacyclic_mul(a: list[int], b: list[int]) -> list[int]:
    """`(a · b) mod (x^m + 1)` over Python integers, at any coefficient width.

    Distinct from [`negacyclic_mul`](#negacyclic_mul) above, which reduces mod
    `q` and holds its accumulator in `int64`. The NTRU descent's coefficients
    pass 9,000 bits and are not reduced by anything, so numpy cannot hold them
    and the loop is the price of an oracle that is exact rather than nearly.
    """
    degree = len(a)
    out = [0] * degree
    for i, left in enumerate(a):
        if left == 0:
            continue
        for j, right in enumerate(b):
            position = i + j
            if position < degree:
                out[position] += left * right
            else:
                out[position - degree] -= left * right
    return out


def field_norm(a: list[int]) -> list[int]:
    """`N(f) = f_e² - x·f_o²`, Algorithm 6's descent, one level.

    `f(x) = f_e(x²) + x·f_o(x²)` splits by parity, and the norm lands in
    `Z[x]/(x^(m/2) + 1)`. The wrap that makes the last term of `x·f_o²` come
    back with a flipped sign is the same one `exact_negacyclic_mul` applies, and
    it is the only place the ring shows through.
    """
    half = len(a) // 2
    even = exact_negacyclic_mul(a[0::2], a[0::2])
    odd = exact_negacyclic_mul(a[1::2], a[1::2])
    out = list(even)
    out[0] += odd[half - 1]
    for i in range(1, half):
        out[i] -= odd[i - 1]
    return out


def gram_schmidt_squared_norm(f: list[int], g: list[int], q: int) -> float:
    """Algorithm 5 line 5, written the way the specification states it.

    The larger of the NTRU basis's two orthogonalised rows: `‖(g, −f)‖²`, and
    `q²` times the norm of `(ḡ, f̄)` divided by `f f̄ + g ḡ`. The division is
    what forces the rational transform — there is no integer form of this — so
    the transcription uses numpy's own complex arithmetic rather than the
    module's, which is the independence that makes it an oracle.
    """
    n = len(f)
    roots = np.exp(1j * np.pi * (2 * np.arange(n) + 1) / n)
    f_hat = np.polyval(np.asarray(f, float)[::-1], roots)
    g_hat = np.polyval(np.asarray(g, float)[::-1], roots)
    energy = (f_hat * np.conj(f_hat) + g_hat * np.conj(g_hat)).real
    # Back to coefficients by solving the Vandermonde system the roots define,
    # which is the inverse transform written as what it is.
    powers = np.vander(roots, n, increasing=True)
    from_g = np.linalg.solve(powers, np.conj(g_hat) / energy).real
    from_f = np.linalg.solve(powers, np.conj(f_hat) / energy).real
    first = float(np.sum(np.asarray(f, float) ** 2) + np.sum(np.asarray(g, float) ** 2))
    second = q**2 * float(np.sum(from_g**2) + np.sum(from_f**2))
    return max(first, second)


def is_invertible(f: list[int], q: int) -> bool:
    """Algorithm 5 line 6 — `f` a unit in `Z_q[x]/(x^n + 1)`.

    Evaluated at the `2n`-th roots of unity by direct exponentiation rather than
    by a transform, so the check and the module's NTT share nothing.
    """
    n = len(f)
    root = pow(11, (q - 1) // (2 * n), q)
    assert pow(root, n, q) == q - 1, "11 does not generate a 2n-th root here"
    return all(
        sum(int(f[j]) * pow(root, (2 * i + 1) * j, q) for j in range(n)) % q
        for i in range(n)
    )
