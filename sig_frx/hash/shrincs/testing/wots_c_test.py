# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""WOTS+C recovers the FXMSS leaf the reference computed, and makes it again.

The leaf is the pinned intermediate this localizes against: if a chain walk, an
address or the digest-to-index map is wrong, it fails here rather than at a root
comparison five hundred hashes later.

Both directions are gated, and the signer's is the one that cannot be gated on
itself. A chain walked from a secret start to a public end and then back down to
the same end proves only that two walks agree; what has to be shown is that the
end is the reference's, which is why `wots_c_public_key` is pinned per case and
compared against both `public_key` and `pk_from_sig`.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest
from hash_frx import Sha256

from sig_frx.hash.shrincs import wots_c
from sig_frx.hash.shrincs.testing import harness
from sig_frx.hash.shrincs.testing import stateful_vectors as vectors
from sig_frx.hash.tweakable import Sha2TweakableHash

_N = 16
_TWEAK = Sha2TweakableHash(Sha256(), n=_N, m=24)


def _rows(*values: bytes) -> np.ndarray:
    return np.stack([np.frombuffer(v, dtype=np.uint8) for v in values])


def _fxmss_signature(case: vectors.StatefulVectors) -> bytes:
    """The FXMSS signature, past the indicator, randomizer and leaf-index field."""
    return harness.fxmss_body(case)


def _structure(case: vectors.StatefulVectors) -> np.ndarray:
    """The two bytes the signer's PRF address carries and the verifier never sees."""
    return np.array([case.shape, case.depth], dtype=np.uint8)


def _secrets(case: vectors.StatefulVectors) -> tuple[np.ndarray, np.ndarray]:
    """`SK.seed` and `PK.seed`, which key generation takes as one 48-byte seed."""
    return (
        np.frombuffer(case.seed[:_N], dtype=np.uint8),
        np.frombuffer(case.seed[2 * _N :], dtype=np.uint8),
    )


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


class SignerTest(absltest.TestCase):
    """The half that makes a leaf, rather than the half that recovers one."""

    def test_every_reference_leaf_is_generated(self) -> None:
        """The same leaf `pk_from_sig` recovers, from the secret it was made of."""
        for case in vectors.REFERENCE:
            with self.subTest(case.label):
                heights, indices = _position(case)
                sk_seed, pk_seed = _secrets(case)
                keys = wots_c.public_key(
                    _TWEAK, pk_seed, sk_seed, _structure(case), heights, indices
                )
                self.assertEqual(bytes(np.asarray(keys)[0]), case.wots_c_public_key)

    def test_the_tree_structure_reaches_the_leaf(self) -> None:
        """The one address that carries the shape, so two trees do not share leaves.

        A key generated at one position under a balanced tree must not be the key
        at that position under an unbalanced one, or a signature made in either
        would carry to the other.
        """
        case = vectors.REFERENCE[0]
        heights, indices = _position(case)
        sk_seed, pk_seed = _secrets(case)
        for structure in (
            np.array([case.shape ^ 1, case.depth], dtype=np.uint8),
            np.array([case.shape, case.depth + 1], dtype=np.uint8),
        ):
            with self.subTest(structure=list(structure)):
                keys = wots_c.public_key(
                    _TWEAK, pk_seed, sk_seed, structure, heights, indices
                )
                self.assertNotEqual(bytes(np.asarray(keys)[0]), case.wots_c_public_key)

    def test_every_reference_counter_is_the_one_grinding_finds(self) -> None:
        """The lowest counter that lands in the subset, which is what the signer ships.

        A signature verifies under any counter that lands, so agreeing with the
        reference on *which* one is a stronger statement than verifying: it says
        the search starts where the reference's does and steps the way it does.
        """
        for case in vectors.REFERENCE:
            with self.subTest(case.label):
                heights, indices = _position(case)
                counter, indexes = wots_c.grind(
                    _TWEAK,
                    np.frombuffer(case.pk_seed, dtype=np.uint8),
                    np.frombuffer(case.message_digest, dtype=np.uint8),
                    heights,
                    indices,
                )
                self.assertEqual(counter, case.grinding_counter)
                self.assertEqual(int(np.asarray(indexes).sum()), wots_c.CONSTANT_SUM)

    def test_every_reference_signature_is_reproduced(self) -> None:
        for case in vectors.REFERENCE:
            with self.subTest(case.label):
                heights, indices = _position(case)
                sk_seed, pk_seed = _secrets(case)
                signature = wots_c.sign(
                    _TWEAK,
                    pk_seed,
                    sk_seed,
                    _structure(case),
                    np.frombuffer(case.message_digest, dtype=np.uint8),
                    heights,
                    indices,
                )
                self.assertEqual(
                    bytes(np.asarray(signature)),
                    _fxmss_signature(case)[: wots_c.SIGNATURE_SIZE],
                )


if __name__ == "__main__":
    absltest.main()
