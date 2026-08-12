# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""FIPS 204's algorithms, transcribed the way the standard writes them.

The modules under test reshape every one of these: the transform becomes one
opcode call plus an ordering conversion, the rounding functions run over whole
arrays, and a rejection loop that squeezes until enough candidates survive
becomes a fixed block schedule with a mask. Each of those is a change made for
the compiler, and it is the only thing about `arith.py` and `sampling.py` a
reader has to take on trust — so this file takes it back, looping one
coefficient at a time over Python integers, and the tests require the two to
agree ([`conventions.md`](../../../../docs/reference/conventions.md)).

Transcribed from the published document (§7.1–§7.5, Appendix A), not from memory
or from another implementation. Nothing here is written for speed: Python integers
have no width, which is exactly why this side is trustworthy and the other side
delegates its arithmetic to a field dtype.

The sampling half needs a SHAKE, and it takes the standard library's — which is
the point. Squeezing here is a `while` loop that asks for one more byte whenever
it needs one, so the reference never has a block bound to get wrong, and the
implementation's bound is compared against a stream that has none.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

Q = 8380417
ZETA = 1753
D = 13
N = 256


def bit_rev8(m: int) -> int:
    """Algorithm 43."""
    bits = [(m >> i) & 1 for i in range(8)]
    return sum(bit << (7 - i) for i, bit in enumerate(bits))


def zetas() -> list[int]:
    """`zeta^BitRev8(k) mod q`, the array Appendix B tabulates."""
    return [pow(ZETA, bit_rev8(k), Q) for k in range(N)]


def mod_plus_minus(r: int, alpha: int) -> int:
    """§2.3's `mod±`: the representative in `(-alpha/2, alpha/2]`."""
    low = r % alpha
    return low - alpha if low > alpha // 2 else low


def ntt_by_definition(w: list[int]) -> list[int]:
    """Eq. (2.1): `NTT(w)[i] = w(zeta^(2·BitRev8(i)+1))`, by direct evaluation.

    §2.5 defines the transform as evaluation at 256 points; §7.5's Algorithm 41 is
    an implementation of it. Evaluating the polynomial with Horner at those points
    shares no structure with the butterfly walk, which is what makes this the check
    that pins the output *ordering*.

    A round trip and the convolution property cannot: a consistently permuted
    evaluation order round-trips exactly and multiplies correctly, because a
    pointwise product does not care what order its slots are in.
    """
    points = [pow(ZETA, 2 * bit_rev8(i) + 1, Q) for i in range(N)]
    values = []
    for point in points:
        acc = 0
        for coefficient in reversed(w):
            acc = (acc * point + coefficient) % Q
        values.append(acc)
    return values


def ntt(w: list[int]) -> list[int]:
    """Algorithm 41, line for line."""
    zeta_table = zetas()
    w_hat = list(w)
    m = 0
    length = 128
    while length >= 1:
        start = 0
        while start < 256:
            m += 1
            z = zeta_table[m]
            for j in range(start, start + length):
                t = (z * w_hat[j + length]) % Q
                w_hat[j + length] = (w_hat[j] - t) % Q
                w_hat[j] = (w_hat[j] + t) % Q
            start += 2 * length
        length //= 2
    return w_hat


def intt(w_hat: list[int]) -> list[int]:
    """Algorithm 42, line for line."""
    zeta_table = zetas()
    w = list(w_hat)
    m = 256
    length = 1
    while length < 256:
        start = 0
        while start < 256:
            m -= 1
            z = (-zeta_table[m]) % Q
            for j in range(start, start + length):
                t = w[j]
                w[j] = (t + w[j + length]) % Q
                w[j + length] = (t - w[j + length]) % Q
                w[j + length] = (z * w[j + length]) % Q
            start += 2 * length
        length *= 2
    f = 8347681
    return [(f * value) % Q for value in w]


def power2round(r: int) -> tuple[int, int]:
    """Algorithm 35."""
    r_plus = r % Q
    r0 = mod_plus_minus(r_plus, 1 << D)
    return (r_plus - r0) // (1 << D), r0


def decompose(r: int, gamma2: int) -> tuple[int, int]:
    """Algorithm 36."""
    r_plus = r % Q
    r0 = mod_plus_minus(r_plus, 2 * gamma2)
    if r_plus - r0 == Q - 1:
        return 0, r0 - 1
    return (r_plus - r0) // (2 * gamma2), r0


def high_bits(r: int, gamma2: int) -> int:
    """Algorithm 37."""
    return decompose(r, gamma2)[0]


def low_bits(r: int, gamma2: int) -> int:
    """Algorithm 38."""
    return decompose(r, gamma2)[1]


def make_hint(z: int, r: int, gamma2: int) -> bool:
    """Algorithm 39."""
    return high_bits(r, gamma2) != high_bits((r + z) % Q, gamma2)


def use_hint(h: bool, r: int, gamma2: int) -> int:
    """Algorithm 40."""
    m = (Q - 1) // (2 * gamma2)
    r1, r0 = decompose(r, gamma2)
    if h and r0 > 0:
        return (r1 + 1) % m
    if h and r0 <= 0:
        return (r1 - 1) % m
    return r1


def infinity_norm(w: list[int]) -> int:
    """§2.3's `||w||∞`, over one polynomial."""
    return max(abs(mod_plus_minus(value, Q)) for value in w)


def integer_to_bytes(x: int, alpha: int) -> bytes:
    """Algorithm 11 — base-256, little-endian."""
    out = bytearray(alpha)
    for i in range(alpha):
        out[i] = (x >> (8 * i)) & 0xFF
    return bytes(out)


def bytes_to_bits(z: bytes) -> list[int]:
    """Algorithm 13 — each byte low bit first."""
    return [(byte >> i) & 1 for byte in z for i in range(8)]


def bits_to_integer(y: list[int]) -> int:
    """Algorithm 10 — little-endian, over the whole bit string."""
    return sum(bit << i for i, bit in enumerate(y))


def bit_unpack(v: bytes, a: int, b: int) -> list[int]:
    """Algorithm 19."""
    c = (a + b).bit_length()
    z = bytes_to_bits(v)
    return [b - bits_to_integer(z[i * c : (i + 1) * c]) for i in range(N)]


class Xof:
    """§3.7's incremental SHAKE: absorb once, then squeeze without a bound.

    Rejection sampling asks for the next few bytes until it has enough
    candidates, and how many that is depends on the bytes. `hashlib` has no
    incremental squeeze, so this re-derives a longer digest whenever the stream
    runs past what it has — SHAKE's output is a prefix chain, so a longer digest
    extends the shorter one rather than replacing it.
    """

    def __init__(self, shake: Callable[[bytes], Any], seed: bytes) -> None:
        self._hasher = shake(seed)
        self._buffer = b""
        self._position = 0

    def squeeze(self, nbytes: int) -> bytes:
        """The next `nbytes` of output."""
        end = self._position + nbytes
        if end > len(self._buffer):
            self._buffer = self._hasher.digest(max(end, 2 * len(self._buffer), 256))
        out = self._buffer[self._position : end]
        self._position = end
        return out


def coeff_from_three_bytes(b0: int, b1: int, b2: int) -> int | None:
    """Algorithm 14 — `None` is the standard's `⊥`."""
    b2_prime = b2 - 128 if b2 > 127 else b2
    z = (1 << 16) * b2_prime + (1 << 8) * b1 + b0
    return z if z < Q else None


def coeff_from_half_byte(b: int, eta: int) -> int | None:
    """Algorithm 15 — `None` is the standard's `⊥`."""
    if eta == 2 and b < 15:
        return 2 - (b % 5)
    if eta == 4 and b < 9:
        return 4 - b
    return None


def rej_ntt_poly(rho: bytes) -> list[int]:
    """Algorithm 30, over `G` = SHAKE128 (§3.7)."""
    ctx = Xof(hashlib.shake_128, rho)
    a_hat: list[int] = []
    while len(a_hat) < N:
        s = ctx.squeeze(3)
        coefficient = coeff_from_three_bytes(s[0], s[1], s[2])
        if coefficient is not None:
            a_hat.append(coefficient)
    return a_hat


def rej_bounded_poly(rho: bytes, eta: int) -> list[int]:
    """Algorithm 31, over `H` = SHAKE256 (§3.7)."""
    ctx = Xof(hashlib.shake_256, rho)
    a: list[int] = []
    while len(a) < N:
        z = ctx.squeeze(1)[0]
        z0 = coeff_from_half_byte(z % 16, eta)
        z1 = coeff_from_half_byte(z // 16, eta)
        if z0 is not None:
            a.append(z0)
        if z1 is not None and len(a) < N:
            a.append(z1)
    return a


def expand_a(rho: bytes, k: int, ell: int) -> list[list[list[int]]]:
    """Algorithm 32."""
    return [
        [
            rej_ntt_poly(rho + integer_to_bytes(s, 1) + integer_to_bytes(r, 1))
            for s in range(ell)
        ]
        for r in range(k)
    ]


def expand_s(
    rho: bytes, k: int, ell: int, eta: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Algorithm 33."""
    s1 = [rej_bounded_poly(rho + integer_to_bytes(r, 2), eta) for r in range(ell)]
    s2 = [rej_bounded_poly(rho + integer_to_bytes(r + ell, 2), eta) for r in range(k)]
    return s1, s2


def expand_mask(rho: bytes, mu: int, ell: int, gamma1: int) -> list[list[int]]:
    """Algorithm 34."""
    c = 1 + (gamma1 - 1).bit_length()
    y = []
    for r in range(ell):
        rho_prime = rho + integer_to_bytes(mu + r, 2)
        v = hashlib.shake_256(rho_prime).digest(32 * c)
        y.append(bit_unpack(v, gamma1 - 1, gamma1))
    return y


def sample_in_ball(rho: bytes, tau: int) -> list[int]:
    """Algorithm 29."""
    c = [0] * N
    ctx = Xof(hashlib.shake_256, rho)
    h = bytes_to_bits(ctx.squeeze(8))
    for i in range(N - tau, N):
        j = ctx.squeeze(1)[0]
        while j > i:
            j = ctx.squeeze(1)[0]
        c[i] = c[j]
        c[j] = (-1) ** h[i + tau - N]
    return c
