# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""WOTS+C recovers the FXMSS leaf the reference computed, and rejects a counter.

The leaf is the pinned intermediate this localizes against: if a chain walk, an
address or the digest-to-index map is wrong, it fails here rather than at a root
comparison five hundred hashes later.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest
from hash_frx import Sha256

from sig_frx.hash.shrincs import wots_c
from sig_frx.hash.shrincs.testing import stateful_vectors as vectors
from sig_frx.hash.tweakable import Sha2TweakableHash

_N = 16
_TWEAK = Sha2TweakableHash(Sha256(), n=_N, m=24)


def _rows(*values: bytes) -> np.ndarray:
    return np.stack([np.frombuffer(v, dtype=np.uint8) for v in values])


def _fxmss_signature(case: vectors.StatefulVectors) -> bytes:
    """The FXMSS signature, past the indicator, randomizer and leaf-index field."""
    return case.signature[17 + case.leaf_index_size :]


def _position(case: vectors.StatefulVectors) -> tuple[np.ndarray, np.ndarray]:
    """The leaf's height as a column and its index as an eight-byte string."""
    return (
        np.array([case.leaf_height], dtype=np.uint32),
        _rows(case.leaf_index.to_bytes(8, "big")),
    )


class ConstantSumTest(absltest.TestCase):
    def test_the_constant_is_the_specifications(self) -> None:
        self.assertEqual(wots_c.CONSTANT_SUM, 240)
        self.assertEqual(wots_c.CHAINS_SIZE, 512)
        self.assertEqual(wots_c.SIGNATURE_SIZE, 514)

    def test_the_reference_counter_maps_into_the_subset(self) -> None:
        for case in vectors.REFERENCE:
            with self.subTest(case.label):
                heights, indices = _position(case)
                indexes, accepted = wots_c.map_digest(
                    _TWEAK,
                    _rows(case.pk_seed),
                    _rows(case.message_digest),
                    _rows(_fxmss_signature(case)[:2]),
                    heights,
                    indices,
                )
                self.assertEqual(list(np.asarray(accepted)), [True])
                self.assertEqual(int(np.asarray(indexes).sum()), wots_c.CONSTANT_SUM)
                self.assertEqual(np.asarray(indexes).shape, (1, wots_c.CHAIN_COUNT))

    def test_a_wrong_counter_falls_outside_it(self) -> None:
        """The grind is the signer's; the verifier gets one counter and checks it.

        Only about one counter in sixty maps into the subset at these parameters,
        so a handful of wrong ones is a real check rather than a coincidence.
        """
        case = vectors.REFERENCE[0]
        heights, indices = _position(case)
        counter = int.from_bytes(_fxmss_signature(case)[:2], "big")
        wrong = [
            (counter ^ 1).to_bytes(2, "big"),
            (counter ^ 0x100).to_bytes(2, "big"),
            (0).to_bytes(2, "big"),
        ]
        _, accepted = wots_c.map_digest(
            _TWEAK,
            _rows(*([case.pk_seed] * len(wrong))),
            _rows(*([case.message_digest] * len(wrong))),
            _rows(*wrong),
            np.full(len(wrong), case.leaf_height, dtype=np.uint32),
            _rows(*([case.leaf_index.to_bytes(8, "big")] * len(wrong))),
        )
        self.assertNotIn(True, list(np.asarray(accepted)))


class PublicKeyTest(absltest.TestCase):
    def test_every_reference_leaf_is_recovered(self) -> None:
        for case in vectors.REFERENCE:
            with self.subTest(case.label):
                heights, indices = _position(case)
                keys, accepted = wots_c.pk_from_sig(
                    _TWEAK,
                    _rows(case.pk_seed),
                    _rows(_fxmss_signature(case)[: wots_c.SIGNATURE_SIZE]),
                    _rows(case.message_digest),
                    heights,
                    indices,
                )
                self.assertEqual(list(np.asarray(accepted)), [True])
                self.assertEqual(bytes(np.asarray(keys)[0]), case.wots_c_public_key)

    def test_the_position_is_part_of_the_leaf(self) -> None:
        """A tip walked at the wrong height or index compresses to another leaf.

        This is what the address tweak is for: the same chains at another position
        must not produce the same key, or a leaf could be replayed elsewhere in
        the tree.
        """
        case = vectors.REFERENCE[1]
        signature = _fxmss_signature(case)[: wots_c.SIGNATURE_SIZE]
        for height, index in (
            (case.leaf_height + 1, case.leaf_index),
            (case.leaf_height, case.leaf_index + 1),
        ):
            with self.subTest(height=height, index=index):
                keys, _ = wots_c.pk_from_sig(
                    _TWEAK,
                    _rows(case.pk_seed),
                    _rows(signature),
                    _rows(case.message_digest),
                    np.array([height], dtype=np.uint32),
                    _rows(index.to_bytes(8, "big")),
                )
                self.assertNotEqual(bytes(np.asarray(keys)[0]), case.wots_c_public_key)

    def test_a_batch_recovers_each_entry_independently(self) -> None:
        cases = [c for c in vectors.REFERENCE if c.leaf_index_size == 1]
        keys, accepted = wots_c.pk_from_sig(
            _TWEAK,
            _rows(*(c.pk_seed for c in cases)),
            _rows(*(_fxmss_signature(c)[: wots_c.SIGNATURE_SIZE] for c in cases)),
            _rows(*(c.message_digest for c in cases)),
            np.array([c.leaf_height for c in cases], dtype=np.uint32),
            _rows(*(c.leaf_index.to_bytes(8, "big") for c in cases)),
        )
        self.assertEqual(list(np.asarray(accepted)), [True] * len(cases))
        self.assertEqual(
            [bytes(row) for row in np.asarray(keys)],
            [c.wots_c_public_key for c in cases],
        )


if __name__ == "__main__":
    absltest.main()
