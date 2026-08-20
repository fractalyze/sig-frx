# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""SEC 1 §2.2.1's addition rule, looping the way the standard writes it.

The implementation under test computes the group law as one complete projective
formula; the standard writes it as five affine cases. This module is those five
cases over Python integers, case by case, so the reshaped form has the
standard's own form to answer to (`conventions.md`). It depends on nothing
outside the standard library, so a disagreement is never the reference's fault.

A point is an `(x, y)` tuple of ints, and the point at infinity is `None` —
SEC 1's `O`, which has no coordinates.
"""

from __future__ import annotations

AffinePoint = tuple[int, int] | None


def add(p: int, a: int, p1: AffinePoint, p2: AffinePoint) -> AffinePoint:
    """`P1 + P2` on `y² = x³ + ax + b` over `F_p` — SEC 1 §2.2.1, rules 1-5."""
    # Rule 1: O + O = O.  Rule 2: (x, y) + O = O + (x, y) = (x, y).
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2 and (y1 != y2 or y1 == 0):
        # Rule 3: (x, y) + (x, -y) = O, covering y = 0 doubling with it.
        return None
    if p1 == p2:
        # Rule 5: doubling, λ = (3x₁² + a) / 2y₁.
        lam = (3 * x1 * x1 + a) * pow(2 * y1, -1, p) % p
    else:
        # Rule 4: distinct x, λ = (y₂ - y₁) / (x₂ - x₁).
        lam = (y2 - y1) * pow(x2 - x1, -1, p) % p
    x3 = (lam * lam - x1 - x2) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return (x3, y3)


def scalar_mul(p: int, a: int, k: int, point: AffinePoint) -> AffinePoint:
    """`kP` by double-and-add over `k`'s bits — SEC 1 §2.2's named algorithm."""
    acc: AffinePoint = None
    for bit in bin(k)[2:] if k else "":
        acc = add(p, a, acc, acc)
        if bit == "1":
            acc = add(p, a, acc, point)
    return acc


def on_curve(p: int, a: int, b: int, point: AffinePoint) -> bool:
    """Whether `point` satisfies the defining equation — SEC 1 §2.2.1."""
    if point is None:
        return True
    x, y = point
    return y * y % p == (x * x * x + a * x + b) % p
