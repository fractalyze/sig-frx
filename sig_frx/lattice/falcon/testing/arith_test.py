# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The transform computes negacyclic convolution, checked against exact integers.

[`arith.py`](../arith.py) argues why there is no table, root or ordering here to
reproduce. What replaces them is the property that defines the transform, and it
is checkable with no reference at all: multiplying through the transform must
equal multiplying in `Z_q[x]/(x^n + 1)`.

So the oracle is integer arithmetic with none of ours in the path — `np.convolve`
over `int64`, which is plain multiply-accumulate on integers rather than anything
that routes through a float FFT. A round trip alone would pass against a wrong
root *and* against the wrong ring, which is why it is the weakest case here and
not the only one.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest, parameterized

from sig_frx.lattice.falcon import arith
from sig_frx.lattice.falcon.testing import falcon_reference as ref


def _random(shape: int | tuple[int, ...], seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, arith.Q, size=shape, dtype=np.int64)


def _ring_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`a · b` through the transform, back on the host as centered integers.

    Composed here rather than in `arith`, which stops at `base_mul` on purpose:
    a caller with an operand to reuse hoists its transform instead.
    """
    a_hat = arith.ntt(arith.to_field(a))
    b_hat = arith.ntt(arith.to_field(b))
    return np.asarray(arith.centered(arith.intt(arith.base_mul(a_hat, b_hat))))


class TransformTest(parameterized.TestCase):
    @parameterized.parameters(*arith.DEGREES)
    def test_multiplication_matches_exact_integer_convolution(self, n: int) -> None:
        """The gate: the transform implements the ring's multiplication."""
        a, b = _random(n, seed=1), _random(n, seed=2)
        np.testing.assert_array_equal(
            _ring_mul(a, b) % arith.Q, ref.negacyclic_mul(a, b)
        )

    @parameterized.parameters(*arith.DEGREES)
    def test_round_trip(self, n: int) -> None:
        """Weaker than the convolution case, and it catches a wrong scale."""
        a = _random(n, seed=3)
        back = arith.intt(arith.ntt(arith.to_field(a)))
        np.testing.assert_array_equal(
            np.asarray(back.astype(np.uint32)), a.astype(np.uint32)
        )

    @parameterized.parameters(*arith.DEGREES)
    def test_multiplication_is_batched_over_leading_axes(self, n: int) -> None:
        """Verification takes a batch, so the transform has to carry one."""
        a, b = _random((3, n), seed=4), _random((3, n), seed=5)
        got = _ring_mul(a, b)
        for row, (a_row, b_row) in enumerate(zip(a, b)):
            np.testing.assert_array_equal(
                got[row] % arith.Q,
                ref.negacyclic_mul(a_row, b_row),
                err_msg=f"batch row {row}",
            )

    def test_the_wrap_is_negacyclic_and_not_cyclic(self) -> None:
        """`x^(n-1) · x = −1`, which is the sign the reference above encodes.

        Its own case because it is the one value separating this ring from
        `x^n − 1`, and a transform in that ring round-trips just as well.
        """
        n = arith.DEGREES[0]
        top, x = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
        top[n - 1], x[1] = 1, 1
        want = np.zeros(n, dtype=np.int64)
        want[0] = -1
        np.testing.assert_array_equal(_ring_mul(top, x), want)


class RepresentationTest(absltest.TestCase):
    def test_centered_is_the_inverse_of_to_field(self) -> None:
        values = np.array([-(arith.Q // 2), -1, 0, 1, arith.Q // 2], dtype=np.int64)
        np.testing.assert_array_equal(
            np.asarray(arith.centered(arith.to_field(values))), values
        )

    def test_centered_reads_q_minus_one_as_minus_one(self) -> None:
        """The reason a norm cannot be measured on canonical residues.

        `(q − 1)²` is ~1.5e8 against a bound in the millions, so a signature
        carrying a single `−1` coefficient would be rejected.
        """
        got = arith.centered(arith.to_field(np.array([arith.Q - 1], dtype=np.int64)))
        self.assertEqual(int(np.asarray(got)[0]), -1)

    def test_centered_leaves_an_already_centered_argument_alone(self) -> None:
        """A norm is measured over values that have already been through it."""
        signed = np.array([-(arith.Q // 2), -5, 0, 5, arith.Q // 2], dtype=np.int32)
        np.testing.assert_array_equal(np.asarray(arith.centered(signed)), signed)


class DegreeTest(absltest.TestCase):
    def test_a_degree_falcon_does_not_define_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a Falcon parameter set"):
            arith.ntt(arith.to_field(np.zeros(256, dtype=np.int64)))


if __name__ == "__main__":
    absltest.main()
