# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The FXMSS walk climbs from the reference's leaf to the reference's root.

`sf_root` is the public key's third part, so reaching it is the whole of what a
stateful signature claims. The cases vary the two things the walk's shape depends
on — how deep the leaf sits, and which side of each parent it falls on — because
one depth and one parity would pass with the mask or the bit test wrong.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest
from hash_frx import Sha256

from sig_frx.hash.shrincs import fxmss
from sig_frx.hash.shrincs.testing import stateful_vectors as vectors
from sig_frx.hash.tweakable import Sha2TweakableHash

_TWEAK = Sha2TweakableHash(Sha256(), n=16, m=24)


def _rows(*values: bytes) -> np.ndarray:
    return np.stack([np.frombuffer(v, dtype=np.uint8) for v in values])


def _padded(case: vectors.StatefulVectors) -> bytes:
    """The FXMSS signature, zero-padded to the widest the format allows."""
    body = case.signature[17 + case.leaf_index_size :]
    return body + bytes(fxmss.SIGNATURE_SIZE_MAX - len(body))


def _index(case: vectors.StatefulVectors) -> bytes:
    return case.leaf_index.to_bytes(8, "big")


class ConstantTest(absltest.TestCase):
    def test_the_sizes_are_the_specifications(self) -> None:
        self.assertEqual(fxmss.HEIGHT, 255)
        self.assertEqual(fxmss.SIGNATURE_SIZE_MIN, 530)
        self.assertEqual(fxmss.SIGNATURE_SIZE_MAX, 4594)

    def test_the_index_field_widens_with_the_depth(self) -> None:
        # The widths the cases below actually exercise, plus the two boundaries:
        # eight bytes is reached at 57 bits and never exceeded.
        self.assertEqual(fxmss.index_field_bytes(1), 1)
        self.assertEqual(fxmss.index_field_bytes(8), 1)
        self.assertEqual(fxmss.index_field_bytes(9), 2)
        self.assertEqual(fxmss.index_field_bytes(56), 7)
        self.assertEqual(fxmss.index_field_bytes(57), 8)
        self.assertEqual(fxmss.index_field_bytes(64), 8)
        self.assertEqual(fxmss.index_field_bytes(255), 8)

    def test_every_case_agrees_with_it(self) -> None:
        for case in vectors.REFERENCE:
            with self.subTest(case.label):
                self.assertEqual(
                    fxmss.index_field_bytes(case.leaf_depth), case.leaf_index_size
                )


class RootTest(absltest.TestCase):
    def test_every_reference_signature_climbs_to_sf_root(self) -> None:
        for case in vectors.REFERENCE:
            with self.subTest(case.label, depth=case.leaf_depth):
                roots, accepted = fxmss.root_from_sig(
                    _TWEAK,
                    _rows(case.pk_seed),
                    _rows(_padded(case)),
                    _rows(case.message_digest),
                    np.array([case.leaf_height], dtype=np.uint32),
                    _rows(_index(case)),
                )
                self.assertEqual(list(np.asarray(accepted)), [True])
                self.assertEqual(bytes(np.asarray(roots)[0]), case.sf_root)

    def test_a_batch_of_different_depths_climbs_independently(self) -> None:
        """The masking is what this is for: depths 1, 4, 16 and 64 in one call.

        Each entry must stop at its own root and hold there while the others keep
        climbing — a mask that leaked would give every entry the deepest walk.
        """
        cases = list(vectors.REFERENCE)
        roots, accepted = fxmss.root_from_sig(
            _TWEAK,
            _rows(*(c.pk_seed for c in cases)),
            _rows(*(_padded(c) for c in cases)),
            _rows(*(c.message_digest for c in cases)),
            np.array([c.leaf_height for c in cases], dtype=np.uint32),
            _rows(*(_index(c) for c in cases)),
        )
        self.assertEqual(list(np.asarray(accepted)), [True] * len(cases))
        self.assertEqual(
            [bytes(row) for row in np.asarray(roots)], [c.sf_root for c in cases]
        )

    def test_a_wrong_sibling_reaches_another_root(self) -> None:
        case = vectors.REFERENCE[3]  # depth 16, so there are siblings to break
        body = bytearray(_padded(case))
        body[fxmss.SIGNATURE_SIZE_MIN - 1] ^= 0x01  # the first auth-path node
        roots, _ = fxmss.root_from_sig(
            _TWEAK,
            _rows(case.pk_seed),
            _rows(bytes(body)),
            _rows(case.message_digest),
            np.array([case.leaf_height], dtype=np.uint32),
            _rows(_index(case)),
        )
        self.assertNotEqual(bytes(np.asarray(roots)[0]), case.sf_root)

    def test_a_wrong_leaf_index_reaches_another_root(self) -> None:
        """The index picks a side at every level, so changing it changes the walk.

        Case 1 sits at index 11 — `0b1011` — so its four levels go right, right,
        left, right, and flipping the low bit changes the very first pairing.
        """
        case = vectors.REFERENCE[1]
        roots, _ = fxmss.root_from_sig(
            _TWEAK,
            _rows(case.pk_seed),
            _rows(_padded(case)),
            _rows(case.message_digest),
            np.array([case.leaf_height], dtype=np.uint32),
            _rows((case.leaf_index ^ 1).to_bytes(8, "big")),
        )
        self.assertNotEqual(bytes(np.asarray(roots)[0]), case.sf_root)

    def test_a_wrong_depth_reaches_another_root(self) -> None:
        """One step too many or too few, and the walk stops somewhere else."""
        case = vectors.REFERENCE[0]
        for height in (case.leaf_height - 1, case.leaf_height + 1):
            with self.subTest(height=height):
                roots, _ = fxmss.root_from_sig(
                    _TWEAK,
                    _rows(case.pk_seed),
                    _rows(_padded(case)),
                    _rows(case.message_digest),
                    np.array([height], dtype=np.uint32),
                    _rows(_index(case)),
                )
                self.assertNotEqual(bytes(np.asarray(roots)[0]), case.sf_root)


if __name__ == "__main__":
    absltest.main()
