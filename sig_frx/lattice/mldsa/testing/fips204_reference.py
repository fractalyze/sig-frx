# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""FIPS 204's arithmetic algorithms, transcribed the way the standard writes them.

The module under test reshapes every one of these: the transform becomes one
opcode call plus an ordering conversion, and the rounding functions run over
whole arrays. Each of those is a change made for the compiler, and it is the only
thing about `arith.py` a reader has to take on trust — so this file takes it
back, looping one coefficient at a time over Python integers, and the test
requires the two to agree
([`conventions.md`](../../../../docs/reference/conventions.md)).

Transcribed from the published document (§7.4, §7.5, Appendix A), not from memory
or from another implementation. Nothing here is written for speed: Python integers
have no width, which is exactly why this side is trustworthy and the other side
delegates its arithmetic to a field dtype.
"""

from __future__ import annotations

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
