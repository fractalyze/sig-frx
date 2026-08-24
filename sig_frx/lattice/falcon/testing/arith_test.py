# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The transform computes negacyclic convolution, checked against exact integers.

Falcon publishes no NTT table to gate against and its NTT domain never leaves the
implementation — the public key and the signature are both coefficient-domain, and
nothing is sampled in the transform domain — so there is no reference ordering or
root for this to reproduce. What there is instead is the property that defines the
transform, and it is checkable without any reference at all: multiplying through
the transform must equal multiplying in `Z_q[x]/(x^n + 1)`.

So the oracle here is `int`. Python's arbitrary-precision integers compute the
convolution directly, with no modular arithmetic of ours in the path, which is the
one comparison a wrong root or a wrong order cannot pass. A round trip alone would
pass under either mistake, which is why it is the weaker of the two cases below
and not the only one.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest, parameterized
from frx import numpy as fnp

from sig_frx.lattice.falcon import arith


def _negacyclic_reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`(a · b) mod (x^n + 1) mod q`, in exact integers.

    The wrap is a subtraction rather than an addition — that sign is the whole
    difference between this ring and the cyclic one, and getting it wrong is the
    mistake a round trip cannot see.
    """
    n = len(a)
    out = [0] * n
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            k = i + j
            if k < n:
                out[k] += int(ai) * int(bj)
            else:
                out[k - n] -= int(ai) * int(bj)
    return np.array([c % arith.Q for c in out], dtype=np.int64)


def _random(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, arith.Q, size=n, dtype=np.int64)


class TransformTest(parameterized.TestCase):
    @parameterized.parameters(*arith.DEGREES)
    def test_multiplication_matches_exact_integer_convolution(self, n: int) -> None:
        """The gate: the transform implements the ring's multiplication."""
        a, b = _random(n, seed=1), _random(n, seed=2)
        got = arith.centered(arith.mul(arith.to_field(a), arith.to_field(b)))
        want = _negacyclic_reference(a, b)
        np.testing.assert_array_equal(np.asarray(got) % arith.Q, want)

    @parameterized.parameters(*arith.DEGREES)
    def test_round_trip(self, n: int) -> None:
        """Weaker than the convolution case, and it catches a wrong scale."""
        a = _random(n, seed=3)
        field = arith.to_field(a)
        back = arith.intt(arith.ntt(field))
        np.testing.assert_array_equal(
            np.asarray(back.astype(np.uint32)), a.astype(np.uint32)
        )

    @parameterized.parameters(*arith.DEGREES)
    def test_multiplication_is_batched_over_leading_axes(self, n: int) -> None:
        """Verification takes a batch, so the transform has to carry one."""
        a = np.stack([_random(n, seed=s) for s in (4, 5, 6)])
        b = np.stack([_random(n, seed=s) for s in (7, 8, 9)])
        got = arith.centered(arith.mul(arith.to_field(a), arith.to_field(b)))
        for row, (a_row, b_row) in enumerate(zip(a, b)):
            np.testing.assert_array_equal(
                np.asarray(got[row]) % arith.Q,
                _negacyclic_reference(a_row, b_row),
                err_msg=f"batch row {row}",
            )

    def test_the_wrap_is_negacyclic_and_not_cyclic(self) -> None:
        """`x^(n-1) · x = −1`, which is the sign the reference above encodes.

        Stated as its own case because it is the one value that separates this
        ring from `x^n − 1`, and a transform in the wrong ring still round-trips.
        """
        n = arith.DEGREES[0]
        top = np.zeros(n, dtype=np.int64)
        top[n - 1] = 1
        x = np.zeros(n, dtype=np.int64)
        x[1] = 1
        got = arith.centered(arith.mul(arith.to_field(top), arith.to_field(x)))
        want = np.zeros(n, dtype=np.int64)
        want[0] = -1
        np.testing.assert_array_equal(np.asarray(got), want)


class RepresentationTest(absltest.TestCase):
    def test_centered_is_the_inverse_of_to_field(self) -> None:
        values = np.array([-(arith.Q // 2), -1, 0, 1, arith.Q // 2], dtype=np.int64)
        got = arith.centered(arith.to_field(values))
        np.testing.assert_array_equal(np.asarray(got), values)

    def test_centered_reads_q_minus_one_as_minus_one(self) -> None:
        """The reason a norm cannot be measured on canonical residues.

        `(q − 1)²` is ~1.5e8 against a bound in the millions, so a signature
        carrying a single `−1` coefficient would be rejected.
        """
        canonical = np.array([arith.Q - 1], dtype=np.int64)
        got = arith.centered(
            fnp.asarray(canonical.astype(np.uint32)).astype(arith.FIELD)
        )
        self.assertEqual(int(np.asarray(got)[0]), -1)


class DegreeTest(absltest.TestCase):
    def test_a_degree_falcon_does_not_define_is_refused(self) -> None:
        """A mis-shaped array is otherwise a wrong verdict, not an error."""
        with self.assertRaisesRegex(ValueError, "not a Falcon parameter set"):
            arith.ntt(arith.to_field(np.zeros(256, dtype=np.int64)))


if __name__ == "__main__":
    absltest.main()
