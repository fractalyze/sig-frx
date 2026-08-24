# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The shared field plumbing, held against the definition it optimizes away.

`pow_const` is a ladder, so what makes it right is not a round trip but
agreement with the bit-at-a-time exponentiation everyone can read. That
reference is written out here rather than imported, because the point is to
compare against the definition, not against another copy of the optimization.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from absl.testing import absltest, parameterized

from sig_frx.classical import edwards, group, secp


def _binary_pow(curve: Any, base: Any, exponent: int) -> Any:
    """`base^exponent`, square-and-multiply one bit at a time."""
    acc = base * np.array(0, dtype=curve.field) + curve.one
    for bit in bin(exponent)[2:]:
        acc = acc * acc
        if bit == "1":
            acc = acc * base
    return acc


# The two exponents the substrates actually run, each with its curve. Both are
# square roots and both are near the binary method's worst case: 247 of 254
# bits set on secp256k1, 251 of 252 on ed25519.
_CURVES = (
    ("secp256k1", secp.SECP256K1, (secp.SECP256K1.p + 1) // 4),
    ("secp256r1", secp.SECP256R1, (secp.SECP256R1.p + 1) // 4),
    ("ed25519", edwards.ED25519, (edwards.ED25519.p - 5) // 8),
)


def _sample_batch(curve: Any) -> np.ndarray:
    """Field values worth exponentiating, boundaries included.

    0 and 1 are the ladder's degenerate inputs; `p - 1` exercises the top of
    the range; the rest are arbitrary. A non-residue is in here by
    construction — half of them are — and the reference comparison catches it
    without needing to know which.
    """
    values = [0, 1, 2, 3, 7, curve.p - 1, curve.p - 2, 0x2A, 0xDEADBEEF]
    return np.array(values, dtype=curve.field)


class PowConstTest(parameterized.TestCase):
    @parameterized.named_parameters(
        (name, curve, exponent) for name, curve, exponent in _CURVES
    )
    def test_matches_binary_reference_on_the_real_exponent(
        self, curve: Any, exponent: int
    ) -> None:
        base = _sample_batch(curve)
        want = _binary_pow(curve, base, exponent)
        self.assertTrue(bool(np.all(group.pow_const(curve, base, exponent) == want)))

    @parameterized.named_parameters(
        (f"{name}_w{window}", curve, exponent, window)
        for name, curve, exponent in _CURVES
        for window in (1, 2, 3, 4, 5, 6)
    )
    def test_every_window_agrees(self, curve: Any, exponent: int, window: int) -> None:
        # The window is a cost knob, never a value one — `w = 1` is the binary
        # method itself, so this also pins the default against it.
        base = _sample_batch(curve)
        want = _binary_pow(curve, base, exponent)
        got = group.pow_const(curve, base, exponent, window=window)
        self.assertTrue(bool(np.all(got == want)))

    @parameterized.named_parameters((name, curve) for name, curve, _ in _CURVES)
    def test_small_exponents_including_the_empty_product(self, curve: Any) -> None:
        # Exponent 0 is the one case that never touches the table, and the one
        # whose result has to be broadcast to the batch rather than taken from
        # it. 1 is the case where the ladder makes no multiplication at all.
        base = _sample_batch(curve)
        for exponent in range(0, 40):
            want = _binary_pow(curve, base, exponent)
            got = group.pow_const(curve, base, exponent)
            self.assertTrue(bool(np.all(got == want)), f"exponent {exponent} disagrees")
            self.assertEqual(np.shape(got), np.shape(base))

    def test_square_root_still_squares_back(self) -> None:
        # The property the callers rely on, stated where it is easy to read:
        # for p = 3 mod 4 the shortcut returns a root exactly on the residues.
        curve = secp.SECP256K1
        squares = np.array([1, 4, 9, 16, 25, 36], dtype=curve.field)
        root = secp.sqrt(curve, squares)
        self.assertTrue(bool(np.all(root * root == squares)))

    def test_negative_exponent_is_refused(self) -> None:
        # The helper has no inverse in it; a negative exponent would otherwise
        # decompose into no digits and silently return the wrong thing.
        with self.assertRaises(ValueError):
            group.pow_const(secp.SECP256K1, secp.SECP256K1.one, -1)

    def test_zero_window_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            group.pow_const(secp.SECP256K1, secp.SECP256K1.one, 5, window=0)


if __name__ == "__main__":
    absltest.main()
