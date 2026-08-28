# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Algorithm 11's tree walk, against the archive's own traced signature."""

from __future__ import annotations

from typing import Any
from unittest import mock

import numpy as np
from absl.testing import absltest, parameterized

from sig_frx.lattice.falcon import sampler, sign
from sig_frx.lattice.falcon.testing import falcon_reference, sampler_vectors

_DEGREES = (512, 1024)

# `σ'` comes out a few parts in `10^12` from the trace's, and the reason is
# Table 3.3 rather than this walk: the table publishes `σ` to twelve significant
# figures and the reference carries a binary constant the decimal does not round
# to. Measured against `fpr_inv_sigma[]`, the gap is `1.4e-13` at `n = 512` —
# which rounds away, so that degree agrees exactly — and `2.7e-12` at
# `n = 1024`. `_SigmaConstantTest` pins both, so a real drift is not absorbed
# here.
#
# It changes no output. Every one of the 3 072 sampled integers matches with the
# published value in use, because the width only sets a Gaussian's scale and a
# relative `3e-12` is orders below anything the acceptance can resolve. Using
# the specification's own number is the choice; matching the reference's
# generator is not what standards-exact means here.
_SIGMA_TOLERANCE = 1e-11


def _basis(signed: sampler_vectors.Signature, degree: int) -> sign.SigningBasis:
    """The traced signature's own key, as signing loads it.

    Through the production `signing_basis` rather than rebuilt here: the four
    transforms and the `gram → ffldl → normalize` composition are what the walk
    below depends on, so a second spelling of them would be the thing under test
    vouching for itself.
    """
    sigma = falcon_reference.PARAMETER_SETS[f"Falcon-{degree}"]["sigma"]
    return sign.signing_basis(signed.f, signed.g, signed.big_f, sigma)


def _target(
    signed: sampler_vectors.Signature, basis: sign.SigningBasis
) -> tuple[Any, Any]:
    """Algorithm 10 line 3, through the production function it gates."""
    return sign.target(signed.hashed_message, basis.f_hat, basis.big_f_hat)


class FfSamplingTest(parameterized.TestCase):
    @parameterized.parameters(*_DEGREES)
    def test_the_walk_presents_every_published_centre(self, degree: int) -> None:
        """Every `SamplerZ` the walk makes, against the one the trace records.

        The centres are what `ffSampling` computes and the widths are what the
        tree carries, so this holds the whole walk: the split convention, the
        order the two children are descended in, and line 10's correction, which
        is the only thing making the leaves depend on each other.
        """
        signed = sampler_vectors.signature(degree)
        published = sampler_vectors.calls(degree)
        stream = sampler_vectors.cursor(b"".join(c.randomness for c in published))

        seen: list[tuple[float, float, int]] = []
        real = sampler.sampler_z

        def recording(center, inverse_sigma, deg, randomness):  # type: ignore[no-untyped-def]
            z = real(center, inverse_sigma, deg, randomness)
            seen.append((center, inverse_sigma, z))
            return z

        with mock.patch.object(sign.sampler, "sampler_z", recording):
            basis = _basis(signed, degree)
            sign.ff_sampling(*_target(signed, basis), basis.tree, stream)

        self.assertLen(seen, len(published))
        for index, (got, want) in enumerate(zip(seen, published)):
            with self.subTest(call=index):
                # The integer is the one thing that must be exact, and it is —
                # at both degrees, for every call, in spite of the width below.
                self.assertEqual(got[2], want.result)
                self.assertAlmostEqual(got[0], want.center, places=9)
                self.assertAlmostEqual(
                    got[1] / want.inverse_sigma, 1.0, delta=_SIGMA_TOLERANCE
                )

    @parameterized.parameters(*_DEGREES)
    def test_the_walk_reproduces_the_published_lattice_point(self, degree: int) -> None:
        """`s2` of `s = (t − z)B̂`, against the signature the trace ends with.

        The case above holds every centre the walk presents; this holds what it
        hands back. They are not the same claim: the per-call centres would
        still line up if `merge` reassembled the halves wrongly at the very last
        level, since nothing after that feeds another centre.

        Algorithm 10 line 7 is computed here rather than called, because the
        signing loop is a later slice — what is being gated is `ffSampling`'s
        output, and this is the published value it has to produce.
        """
        signed = sampler_vectors.signature(degree)
        published = sampler_vectors.calls(degree)
        stream = sampler_vectors.cursor(b"".join(c.randomness for c in published))
        basis = _basis(signed, degree)

        # Algorithm 10 lines 3-9 through the production path, which is what
        # publishes a value to compare: the archive's `signature_vector` is the
        # `s2` one draw produces from this key and this recorded randomness.
        s2, _ = sign.lattice_point(signed.hashed_message, basis, stream)
        np.testing.assert_array_equal(s2, np.asarray(signed.signature_vector))


class SigmaConstantTest(parameterized.TestCase):
    """What Table 3.3's rounding costs, pinned rather than left as a tolerance."""

    # `fpr_inv_sigma[]` at `logn` 9 and 10 — the reference's own `1/σ`, which is
    # what its tree normalizes with and therefore what the traces were made
    # with.
    _REFERENCE_INVERSE_SIGMA = {
        512: 0.006033669668157724,
        1024: 0.005938645309533116,
    }

    @parameterized.parameters(*_DEGREES)
    def test_the_published_sigma_does_not_round_trip(self, degree: int) -> None:
        published = falcon_reference.PARAMETER_SETS[f"Falcon-{degree}"]["sigma"]
        reference = 1.0 / self._REFERENCE_INVERSE_SIGMA[degree]
        self.assertNotEqual(published, reference)
        self.assertLess(abs(published - reference) / reference, _SIGMA_TOLERANCE)


if __name__ == "__main__":
    absltest.main()
