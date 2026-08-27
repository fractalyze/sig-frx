# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The short Weierstrass curve equation, written the way the standard defines it.

The implementation evaluates `y² = x³ + ax + b` over the curated Montgomery
field dtypes — `secp.on_curve` builds a 0-d field array per coordinate and
`secp._weierstrass_rhs` runs Horner over them — and reaches a point's `x` only
after a Jacobian ladder and an inversion. So the independent form to hold that
to is the curve's *defining* affine rule over Python integers, which is what
this module is.

The two must not share code, and the reason is sharper than ordinary
duplication. These expressions are the reference a lift is checked against: a
square root held against its own `sqrt` agrees with itself on a wrong root, and
a parity check between two paths that share a substrate cannot see a substrate
bug. `pow` and Python integers share nothing with the field dtype, the
Montgomery storage or the ladder, which is exactly why they can catch one.

Every entry point takes the curve rather than a modulus, so the `a` term cannot
be dropped by a caller that happens to be working on secp256k1 — where `a = 0`
makes the specialised form look right and leaves it silently wrong for P-256.
"""

from __future__ import annotations

from sig_frx.classical import secp


def rhs(curve: secp.Curve, x: int) -> int:
    """`x³ + ax + b mod p` — the curve equation's right-hand side."""
    return (pow(x, 3, curve.p) + curve.a * x + curve.b) % curve.p


def has_point_at(curve: secp.Curve, x: int) -> bool:
    """Whether some `y` satisfies the curve equation at `x`, by Euler's criterion.

    `rhs == 0` is a point too — the root is `y = 0` — and it is a separate case
    because Euler's criterion answers `0` there rather than `1`.
    """
    value = rhs(curve, x)
    return value == 0 or pow(value, (curve.p - 1) // 2, curve.p) == 1
