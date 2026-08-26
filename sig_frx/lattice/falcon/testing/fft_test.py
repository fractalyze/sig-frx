# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The transform is checked against its definition, not against its own inverse.

A round trip passes for any convention the two halves agree on. It cannot tell
`fft` from `ifft`, and it cannot tell a `split` that pairs a root with its
negative from one that pairs adjacent indices — both round-trip, and only one
of them is the split `f(x) = f0(x²) + x·f1(x²)` defines. Both mistakes were
made while writing this module and both survived a round trip.

So every case below evaluates the polynomial directly and compares. Direct
evaluation is `O(n²)` and its own error grows with `n`, which is why the
tolerances differ by degree: at `n = 1024` the *reference* is the noisy side.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest, parameterized

from sig_frx.lattice.falcon import fft

# Falcon's two degrees, plus a small one whose values are checkable by hand.
DEGREES = (8, 512, 1024)


def _roots(n: int) -> np.ndarray:
    """The `n` roots of `x^n = -1`, on the host and independent of the module."""
    return np.exp(1j * np.pi * (2 * np.arange(n) + 1) / n)


def _evaluate(coefficients: np.ndarray, points: np.ndarray) -> np.ndarray:
    """`Σ_j c_j · p^j` at each `p`, written the way the definition reads."""
    return np.array([np.polyval(coefficients[::-1], p) for p in points])


class ScopeTest(absltest.TestCase):
    def test_every_entry_point_refuses_outside_the_scope(self) -> None:
        """Narrowing to `complex64` is a warning in frx and an error here.

        24 bits of mantissa against 53 is not a tolerance this transform can
        absorb — it is the difference the security analysis rests on — so it
        must not be reachable by forgetting a context manager.
        """
        one = np.ones(8)
        for name, call in (
            ("fft", lambda: fft.fft(one)),
            ("ifft", lambda: fft.ifft(one)),
            ("split", lambda: fft.split(one)),
            ("merge", lambda: fft.merge(one, one)),
            ("roots", lambda: fft.roots(8)),
        ):
            with self.subTest(name):
                with self.assertRaisesRegex(RuntimeError, "double precision"):
                    call()

    def test_the_scope_actually_reaches_double(self) -> None:
        with fft.double_precision():
            self.assertEqual(fft.fft(np.ones(8)).dtype, np.dtype("complex128"))


class TransformTest(parameterized.TestCase):
    @parameterized.parameters(*DEGREES)
    def test_forward_is_evaluation_at_the_roots(self, n: int) -> None:
        """The definition, and the check that catches an `fft`/`ifft` swap.

        The library's forward transform carries the `-i` convention and this
        ring wants `+i`; a round trip is blind to that because the inverse is
        wrong in the same direction.
        """
        f = np.random.default_rng(1).standard_normal(n)
        want = _evaluate(f, _roots(n))
        with fft.double_precision():
            got = np.asarray(fft.fft(f))
        # Direct evaluation is the noisy side at these degrees, not the FFT.
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-9 * n)

    @parameterized.parameters(*DEGREES)
    def test_inverse_returns_the_coefficients(self, n: int) -> None:
        f = np.random.default_rng(2).standard_normal(n)
        with fft.double_precision():
            got = np.asarray(fft.ifft(fft.fft(f)))
        np.testing.assert_allclose(got.real, f, rtol=0, atol=1e-12)
        self.assertLess(np.max(np.abs(got.imag)), 1e-12)


class SplitMergeTest(parameterized.TestCase):
    @parameterized.parameters(*DEGREES)
    def test_split_gives_the_transforms_of_the_two_halves(self, n: int) -> None:
        """`f = f0(x²) + x·f1(x²)`, so the halves are the even and odd terms.

        This is the assertion a round trip cannot make. A `split` that pairs
        adjacent indices — which is what the reference implementation does over
        its own bit-reversed representation — round-trips just as well and is a
        different function.
        """
        f = np.random.default_rng(3).standard_normal(n)
        even, odd = f[0::2], f[1::2]
        want0 = _evaluate(even, _roots(n // 2))
        want1 = _evaluate(odd, _roots(n // 2))
        with fft.double_precision():
            got0, got1 = fft.split(fft.fft(f))
        np.testing.assert_allclose(np.asarray(got0), want0, rtol=0, atol=1e-9 * n)
        np.testing.assert_allclose(np.asarray(got1), want1, rtol=0, atol=1e-9 * n)

    @parameterized.parameters(*DEGREES)
    def test_merge_inverts_split(self, n: int) -> None:
        """Kept for what it does cover: that the two agree on one convention."""
        f = np.random.default_rng(4).standard_normal(n)
        with fft.double_precision():
            f_fft = fft.fft(f)
            got = np.asarray(fft.merge(*fft.split(f_fft)))
        np.testing.assert_allclose(got, np.asarray(f_fft), rtol=0, atol=1e-12)


if __name__ == "__main__":
    absltest.main()
