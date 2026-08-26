# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The prime and the base-p decomposition every leanSig hash input rests on.

Pinned directly rather than only through the digests they feed: a digest says
that something is wrong, not which limb — and the modulus is *derived* from the
dtype rather than written down, so the one thing that could move it is a wheel
bump that no vector here would attribute.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from sig_frx.hash.leansig import field


class PrimeTest(absltest.TestCase):
    """The modulus the pinned wheel hands over is the one leanSpec states."""

    def test_it_is_koalabears(self) -> None:
        # leanSpec's `spec/crypto/koalabear.py`: P = 2^31 - 2^24 + 1.
        self.assertEqual(field.PRIME, 2**31 - 2**24 + 1)


class ToFieldTest(absltest.TestCase):
    """Canonical residues in, the same residues back out."""

    def test_it_round_trips_through_the_canonical_read(self) -> None:
        canonical = [0, 1, 7, field.PRIME - 1]

        got = field.to_field(canonical)

        self.assertEqual(got.dtype, F)
        self.assertEqual(np.asarray(got).astype(np.uint32).tolist(), canonical)

    def test_it_is_astype_and_not_a_bitcast(self) -> None:
        # The storage is a Montgomery representative, so the bytes under a
        # nonzero residue are a different number. A bitcast would round-trip
        # forever while feeding the permutation something else.
        stored = np.asarray(field.to_field([1])).view(np.uint32)

        self.assertNotEqual(int(stored[0]), 1)


class LaneReversedLimbsTest(absltest.TestCase):
    """The composite every caller reaches for: decompose, place, convert."""

    def test_it_places_the_limbs_lane_reversed(self) -> None:
        value = 7 + 11 * field.PRIME + 13 * field.PRIME**2

        got = field.lane_reversed_limbs(value, 3)

        # Least significant first upstream, so last in a reversed vector.
        self.assertEqual(np.asarray(got).astype(np.uint32).tolist(), [13, 11, 7])
        self.assertEqual(got.dtype, F)


class DecompositionTest(absltest.TestCase):
    """The base-p decomposition the placement above rests on.

    Gated through the private name rather than only through
    `lane_reversed_limbs`: a placed vector says that something is wrong, not
    which limb.
    """

    def test_it_is_least_significant_first(self) -> None:
        value = 7 + 11 * field.PRIME + 13 * field.PRIME**2

        self.assertEqual(field._int_to_base_p(value, 3), [7, 11, 13])

    def test_it_pads_with_zeros(self) -> None:
        self.assertEqual(field._int_to_base_p(5, 4), [5, 0, 0, 0])

    def test_a_short_decomposition_is_rejected_rather_than_truncated(self) -> None:
        # Dropping the high part would silently change the hash.
        with self.assertRaisesRegex(ValueError, "base-p limbs"):
            field._int_to_base_p(field.PRIME**3, 2)


if __name__ == "__main__":
    absltest.main()
