# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""`secp256k1_XMD:SHA-256_SSWU_RO_` (RFC 9380 §8.7): bytes to a curve point.

A random oracle whose *codomain is the group* — the thing a protocol needs
when it must hold a group element nobody knows the discrete logarithm of.
Hashing to a scalar and multiplying the generator does not answer it: the
party that hashed knows the exponent, which is exactly what the constructions
that ask for this oracle cannot allow.

The companion to [`xmd.py`](xmd.py), which is the same RFC one layer down —
`expand_message_xmd` and §5.2 are where this starts. The two belong together,
which is what places this module: splitting one specification across packages
would cost more than the curve dependency this side adds. That neither of them
is protocol-specific, and that a non-threshold consumer of hash-to-curve would
reach across for it, is a fair argument for moving the pair rather than for
separating them.

Host-only for the reason the rest of this package is: protocol-side work over
bytes and integers, at `B = 1`. The base-field arithmetic here is plain
`pow`/`%` on Python integers rather than `secp`'s field dtype, because a
single map has no batch to place and an exact host integer is what
[`security.md`](../../docs/reference/security.md) asks of protocol-side
values. What is *not* re-derived here is the group: the two points are summed
through `secp.sum_points` and read back through `secp.affine_ints`, so no
curve arithmetic exists in this module.

## The suite is fixed, not injected

Every constant below — `Z`, `E'`, the isogeny — is secp256k1's, and a second
curve shares none of them. So the curve is named rather than passed, the same
way `xmd.py` fixes SHA-256; the parameter arrives with the first suite that
needs a different answer.

## Why the isogeny is applied twice rather than once

§6.6.3 offers an optimization: `iso_map` is a group homomorphism, so mapping
both `u` values onto `E'`, adding *there*, and applying the isogeny once to
the sum gives the same output for one evaluation instead of two.

This module does not take it. RFC 9380's published vectors give `Q0` and `Q1`
as `map_to_curve` returns them — on `E`, after the isogeny — and the optimized
form never materializes those points, so it would trade the only intermediates
the standard publishes for one field inversion. `testing.md` asks the opposite
trade in as many words: pin the intermediates beneath a value, because a
mismatch in the final point alone says only that something is wrong. At `B = 1`
the saving is unmeasurable and the localization is not.
"""

from __future__ import annotations

import numpy as np

from sig_frx.classical import secp
from sig_frx.threshold import xmd

# RFC 9380 §8.7. `L` is the same 48 RFC 9591's secp256k1 ciphersuite uses, for
# the same reason — both derive it from this curve's field size and `k = 128`.
_L = 48

_CURVE = secp.SECP256K1
_P = _CURVE.p

# The isogenous curve `E'`, and `Z`. `E` itself has `A == 0`, which the
# Simplified SWU map cannot take — hence `E'`, where both coefficients are
# nonzero, and the isogeny that carries the result back.
_A_PRIME = 0x3F8731ABDD661ADCA08A5558F0F5D272E953D363CB6F0E5D405447C01A444533
_B_PRIME = 1771
_Z = -11 % _P

# RFC 9380 App. E.1, the 3-isogeny `E' -> E`, as the four coefficient lists its
# rational functions read from lowest power up:
#     x = (k1 . x'^i) / (x'^2 + k2 . x'^i)
#     y = y' * (k3 . x'^i) / (x'^3 + k4 . x'^i)
_K1 = (
    0x8E38E38E38E38E38E38E38E38E38E38E38E38E38E38E38E38E38E38DAAAAA8C7,
    0x07D3D4C80BC321D5B9F315CEA7FD44C5D595D2FC0BF63B92DFFF1044F17C6581,
    0x534C328D23F234E6E2A413DECA25CAECE4506144037C40314ECBD0B53D9DD262,
    0x8E38E38E38E38E38E38E38E38E38E38E38E38E38E38E38E38E38E38DAAAAA88C,
)
_K2 = (
    0xD35771193D94918A9CA34CCBB7B640DD86CD409542F8487D9FE6B745781EB49B,
    0xEDADC6F64383DC1DF7C4B2D51B54225406D36B641F5E41BBC52A56612A8C6D14,
)
_K3 = (
    0x4BDA12F684BDA12F684BDA12F684BDA12F684BDA12F684BDA12F684B8E38E23C,
    0xC75E0C32D5CB7C0FA9D0A54B12A0A6D5647AB046D686DA6FDFFC90FC201D71A3,
    0x29A6194691F91A73715209EF6512E576722830A201BE2018A765E85A9ECEE931,
    0x2F684BDA12F684BDA12F684BDA12F684BDA12F684BDA12F684BDA12F38E38D84,
)
_K4 = (
    0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFF93B,
    0x7A06534BB8BDB49FD5E9E6632722C2989467C1BFC8E8D978DFB425D2685C2573,
    0x6484AA716545CA2CF3A70C3FA8FE337E0A3D21162F0D6299A7BF8192BFD2A76F,
)


def _inv0(value: int) -> int:
    """RFC 9380 §4: the field inverse, with `inv0(0) = 0` rather than an error.

    `pow(0, p-2, p)` is already `0`, so the exceptional case needs no branch —
    stated because the absence of one reads like an oversight otherwise.
    """
    return pow(value, _P - 2, _P)


def _is_square(value: int) -> bool:
    """Whether `value` has a square root in the field — `0` counts."""
    return pow(value, (_P - 1) // 2, _P) != _P - 1


def _sqrt(value: int) -> int:
    """A square root of a residue, for `p ≡ 3 (mod 4)`.

    The same Tonelli shortcut `secp.sqrt` runs over the field dtype. It is
    spelled out here rather than called because that one takes and returns
    field arrays for a batch this module never has, and the exponent is the
    whole algorithm.
    """
    return pow(value, (_P + 1) // 4, _P)


def _sgn0(value: int) -> int:
    """RFC 9380 §4.1 at `m = 1`: the low bit of the representative."""
    return value % 2


def _poly(coefficients: tuple[int, ...], x: int) -> int:
    """A polynomial at `x`, coefficients lowest power first, by Horner's rule.

    The isogeny is four of these and nothing else, so writing them out by
    subscript would be ten hand-placed indices across four expressions that
    differ only in which list they read.
    """
    total = 0
    for coefficient in reversed(coefficients):
        total = (total * x + coefficient) % _P
    return total


def _g_prime(x: int) -> int:
    """`g'(x) = x³ + A'x + B'`, the curve equation's right side on `E'`.

    The same shape `secp._weierstrass_rhs` factors out on `E`, and named for
    the same reason: §6.6.2 evaluates it at two candidates, and calling it `g`
    is how the specification refers to it.
    """
    return (pow(x, 3, _P) + _A_PRIME * x + _B_PRIME) % _P


# The two values `x1` is chosen between, `-B'/A'` and `B'/(Z·A')`. Both are
# fixed by the suite, so neither inversion is per-call work.
_NEG_B_OVER_A = (-_B_PRIME % _P) * _inv0(_A_PRIME) % _P
_B_OVER_ZA = _B_PRIME * _inv0(_Z * _A_PRIME % _P) % _P


def hash_to_field(message: bytes, dst: bytes) -> list[int]:
    """RFC 9380 §5.2 at this suite's `m = 1`, `L = 48`: the two `u` values.

    Fixed at two because §3's random-oracle encoding is the only caller and
    asks for exactly that; the general form is `xmd.hash_to_field`.
    """
    return xmd.hash_to_field(message, dst, 2, _P, _L)


def _map_to_curve_simple_swu(u: int) -> tuple[int, int]:
    """RFC 9380 §6.6.2 on `E'`, transcribed as its ten numbered operations.

    Straight-line rather than App. F.2's optimized form: this runs once per
    map at `B = 1`, and the numbered version is the one a reader can check
    against the document line by line.
    """
    # 1-3. `x1`, with the exceptional case the condition on `Z` exists to make
    # safe: where the denominator vanishes, `B / (Z * A)` has a square `g(x1)`.
    tv1 = _inv0((_Z * _Z * pow(u, 4, _P) + _Z * pow(u, 2, _P)) % _P)
    x1 = _B_OVER_ZA if tv1 == 0 else _NEG_B_OVER_A * (1 + tv1) % _P

    # 4-6. The two candidates and their curve-equation values.
    gx1 = _g_prime(x1)
    x2 = _Z * pow(u, 2, _P) % _P * x1 % _P
    gx2 = _g_prime(x2)

    # 7-8. Exactly one of the two is a square; that is the map's whole point.
    if _is_square(gx1):
        x, y = x1, _sqrt(gx1)
    else:
        x, y = x2, _sqrt(gx2)

    # 9. `u` and `-u` reach the same `x`, so the sign is taken from `u` — which
    # is what makes the map injective on the sign bit rather than two-to-one.
    if _sgn0(u) != _sgn0(y):
        y = -y % _P
    return x, y


def _iso_map(x: int, y: int) -> tuple[int, int]:
    """RFC 9380 App. E.1: a point on `E'` to its image on `E`.

    A zero denominator is not an error — the RFC requires the identity, and
    those inputs are the isogeny's kernel rather than bad data.
    """
    # The `+ (1,)` is each denominator's monic leading term, which App. E.1
    # writes into the formula rather than into the coefficient list — so the
    # tuples above stay transcribable against it line for line.
    x_num, x_den = _poly(_K1, x), _poly(_K2 + (1,), x)
    y_num, y_den = _poly(_K3, x), _poly(_K4 + (1,), x)
    if x_den == 0 or y_den == 0:
        return secp.AFFINE_IDENTITY
    return x_num * _inv0(x_den) % _P, y * y_num % _P * _inv0(y_den) % _P


def map_to_curve(u: int) -> tuple[int, int]:
    """RFC 9380 §6.6.3: one field element to an affine point on secp256k1.

    Public because the standard publishes its output — `Q0` and `Q1` in the
    vectors are exactly two calls to this — and a gate that could only see the
    final point would not say which half of the map drifted.
    """
    return _iso_map(*_map_to_curve_simple_swu(u))


def hash_to_curve(message: bytes, dst: bytes) -> tuple[int, int]:
    """RFC 9380 §3's random-oracle encoding: `bytes -> (x, y)` on secp256k1.

    `clear_cofactor` is the identity map here — secp256k1 has `h = 1`, so
    §8.7 sets `h_eff = 1` and the sum is already in the prime-order group.

    The sum runs through `secp`, not through affine formulas written here:
    every special case a hand-rolled addition would have to get right — equal
    `x`, opposite `y`, a doubling — is already decided in the dtype's group
    law, and the two points genuinely can coincide.
    """
    u = hash_to_field(message, dst)
    points = np.array(
        [_CURVE.point(map_to_curve(value)) for value in u], dtype=_CURVE.point
    )
    total = secp.sum_points(_CURVE, points)
    return secp.affine_ints(_CURVE, total)[0]
