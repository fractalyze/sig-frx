# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The affine twisted-Edwards addition, looping the way the textbook writes it.

RFC 8032 §5.1.4 publishes the extended-homogeneous formulas the implementation
uses, so the independent form to hold them to is the curve's defining affine
rule: `x3 = (x1y2 + x2y1)/(1 + d·x1x2y1y2)`, `y3 = (y1y2 + x1x2)/(1 - d·…)`,
over Python integers. Complete for `a = -1` and non-square `d`, so there are
no cases to write out — which is the property the extended formulas inherit.
"""

from __future__ import annotations

AffinePoint = tuple[int, int]

IDENTITY: AffinePoint = (0, 1)


def add(p: int, d: int, p1: AffinePoint, p2: AffinePoint) -> AffinePoint:
    """`P1 + P2` on `-x² + y² = 1 + d·x²y²` over `F_p`."""
    x1, y1 = p1
    x2, y2 = p2
    product = d * x1 * x2 * y1 * y2 % p
    x3 = (x1 * y2 + x2 * y1) * pow(1 + product, -1, p) % p
    y3 = (y1 * y2 + x1 * x2) * pow(1 - product, -1, p) % p
    return (x3, y3)


def scalar_mul(p: int, d: int, k: int, point: AffinePoint) -> AffinePoint:
    """`kP` by plain double-and-add over `k`'s bits."""
    acc = IDENTITY
    for bit in bin(k)[2:] if k else "":
        acc = add(p, d, acc, acc)
        if bit == "1":
            acc = add(p, d, acc, point)
    return acc


def on_curve(p: int, d: int, point: AffinePoint) -> bool:
    """Whether `point` satisfies `-x² + y² = 1 + d·x²y²` over `F_p`."""
    x, y = point
    return (-x * x + y * y) % p == (1 + d * x * x * y * y) % p
