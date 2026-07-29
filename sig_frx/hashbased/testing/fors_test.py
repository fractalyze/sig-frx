# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""FORS agrees with FIPS 205 §8, including where it computes a forest at once.

The claim this module makes beyond the standard is that `k` trees reduced as one
contiguous forest are the same `k` trees the standard's recursion builds
separately. So a naive transcription of Algorithm 15's recursion runs beside it,
per tree, and the roots must match — and the signature is checked through its use:
what `sign` produces must reconstruct to what `pk_gen` computed, and must not when
anything about the claim moves.

Small parameters (`a = 3`, `k = 4`) rather than a real set's `a = 12`, `k = 14`:
the recursion is exponential in `a` and the arithmetic under test does not care.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest
from hash_frx.sha256 import Sha256

from sig_frx.hashbased import adrs, fors, tree
from sig_frx.hashbased.tweakable import Sha2TweakableHash

_N = 16
_PARAMS = fors.ForsParams(n=_N, a=3, k=4)
_POSITION = fors.ForsPosition(tree=6, key_pair=2)
_PK_SEED = np.frombuffer(bytes(range(_N)), dtype=np.uint8)
_SK_SEED = np.frombuffer(bytes(range(100, 100 + _N)), dtype=np.uint8)
_DIGEST = np.frombuffer(bytes([0b10110100, 0b01101110]), dtype=np.uint8)


def _family() -> Sha2TweakableHash:
    return Sha2TweakableHash(Sha256(), n=_N, m=30)


def _address(height: int, index: int) -> np.ndarray:
    return adrs.encode_batch(
        [
            adrs.fors_tree(
                layer=0,
                tree=_POSITION.tree,
                key_pair=_POSITION.key_pair,
                height=height,
                index=index,
            )
        ],
        compressed=True,
    )


def _spec_sk(tweak: Sha2TweakableHash, index: int) -> np.ndarray:
    """Algorithm 14, transcribed."""
    address = adrs.encode_batch(
        [
            adrs.fors_prf(
                layer=0,
                tree=_POSITION.tree,
                key_pair=_POSITION.key_pair,
                index=index,
            )
        ],
        compressed=True,
    )
    return np.asarray(tweak.prf(_PK_SEED, _SK_SEED, address))[0]


def _spec_node(tweak: Sha2TweakableHash, index: int, height: int) -> np.ndarray:
    """Algorithm 15, transcribed: the recursion, one node at a time."""
    if height == 0:
        secret = _spec_sk(tweak, index)
        return np.asarray(tweak.f(_PK_SEED, _address(0, index), secret[None, :]))[0]
    left = _spec_node(tweak, 2 * index, height - 1)
    right = _spec_node(tweak, 2 * index + 1, height - 1)
    pair = np.concatenate([left, right])[None, :]
    return np.asarray(tweak.h(_PK_SEED, _address(height, index), pair))[0]


class ForestTest(absltest.TestCase):
    def test_the_leaves_are_the_recursions_base_case(self) -> None:
        tweak = _family()
        got = np.asarray(fors.leaves(tweak, _PARAMS, _PK_SEED, _SK_SEED, _POSITION))
        self.assertEqual(got.shape, (_PARAMS.leaves, _N))
        for index in (0, 1, _PARAMS.t, _PARAMS.leaves - 1):
            self.assertEqual(
                bytes(got[index]), bytes(_spec_node(tweak, index, 0)), f"leaf {index}"
            )

    def test_one_forest_reduction_gives_the_k_separate_roots(self) -> None:
        # The claim: the trees are contiguous in FIPS 205's numbering, so a
        # level's pairs are exactly the per-tree pairs and one pass over the
        # forest is `k` independent trees.
        tweak = _family()
        roots = np.asarray(
            tree.reduce_levels(
                tweak,
                _PK_SEED,
                fors.leaves(tweak, _PARAMS, _PK_SEED, _SK_SEED, _POSITION),
                _PARAMS.a,
                fors._node_addresses(_POSITION),
            )
        )
        self.assertEqual(roots.shape, (_PARAMS.k, _N))
        for index in range(_PARAMS.k):
            self.assertEqual(
                bytes(roots[index]),
                bytes(_spec_node(tweak, index, _PARAMS.a)),
                f"tree {index}",
            )


class MessageIndexTest(absltest.TestCase):
    def test_each_tree_gets_one_leaf_inside_its_own_range(self) -> None:
        indices = fors.message_indices(_PARAMS, _DIGEST)
        self.assertLen(indices, _PARAMS.k)
        for tree_index, leaf in enumerate(indices):
            self.assertGreaterEqual(leaf, tree_index * _PARAMS.t)
            self.assertLess(leaf, (tree_index + 1) * _PARAMS.t)

    def test_the_forest_index_and_the_within_tree_index_agree_on_side(self) -> None:
        # Why one index can address a node *and* decide its sibling's side: the
        # tree offset contributes an even amount at every level a path reaches.
        for tree_index in range(_PARAMS.k):
            for within in range(_PARAMS.t):
                forest = tree_index * _PARAMS.t + within
                for level in range(_PARAMS.a):
                    self.assertEqual(
                        (forest >> level) & 1,
                        (within >> level) & 1,
                        f"tree {tree_index}, leaf {within}, level {level}",
                    )


class SignVerifyTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tweak = _family()
        self.public_key = np.asarray(
            fors.pk_gen(self.tweak, _PARAMS, _PK_SEED, _SK_SEED, _POSITION)
        )
        self.signature = np.asarray(
            fors.sign(self.tweak, _PARAMS, _DIGEST, _PK_SEED, _SK_SEED, _POSITION)
        )

    def _pk_from_sig(self, signature: np.ndarray, digest: np.ndarray) -> bytes:
        return bytes(
            np.asarray(
                fors.pk_from_sig(
                    self.tweak, _PARAMS, signature, digest, _PK_SEED, _POSITION
                )
            )
        )

    def test_a_signature_reconstructs_the_public_key(self) -> None:
        self.assertEqual(self.signature.shape, (_PARAMS.k, _PARAMS.a + 1, _N))
        self.assertEqual(
            self._pk_from_sig(self.signature, _DIGEST), bytes(self.public_key)
        )

    def test_the_revealed_secret_is_the_one_the_digest_picks(self) -> None:
        # Algorithm 16 line 4: the signature's first value per tree is that
        # tree's chosen leaf's secret, not any other leaf's.
        indices = fors.message_indices(_PARAMS, _DIGEST)
        for tree_index, leaf in enumerate(indices):
            self.assertEqual(
                bytes(self.signature[tree_index, 0]),
                bytes(_spec_sk(self.tweak, int(leaf))),
                f"tree {tree_index}",
            )

    def test_another_digest_does_not_reconstruct_the_public_key(self) -> None:
        # Flip a bit the digest is actually read at. FORS consumes exactly
        # `k · a` bits — 12 of these 16 — so the low nibble of the second byte
        # changes nothing, and a test that flipped one of those would pass while
        # asserting nothing.
        other = np.frombuffer(bytes([0b10110101, 0b01101110]), dtype=np.uint8)
        self.assertNotEqual(
            list(fors.message_indices(_PARAMS, other)),
            list(fors.message_indices(_PARAMS, _DIGEST)),
        )
        self.assertNotEqual(
            self._pk_from_sig(self.signature, other), bytes(self.public_key)
        )

    def test_a_tampered_secret_does_not_reconstruct(self) -> None:
        tampered = self.signature.copy()
        tampered[2, 0, 0] ^= 1
        self.assertNotEqual(
            self._pk_from_sig(tampered, _DIGEST), bytes(self.public_key)
        )

    def test_a_tampered_authentication_path_does_not_reconstruct(self) -> None:
        tampered = self.signature.copy()
        tampered[1, 2, 0] ^= 1
        self.assertNotEqual(
            self._pk_from_sig(tampered, _DIGEST), bytes(self.public_key)
        )

    def test_a_key_at_another_position_is_a_different_key(self) -> None:
        elsewhere = fors.ForsPosition(tree=6, key_pair=3)
        other = np.asarray(
            fors.pk_gen(self.tweak, _PARAMS, _PK_SEED, _SK_SEED, elsewhere)
        )
        self.assertNotEqual(bytes(other), bytes(self.public_key))

    def test_a_misshapen_signature_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "FORS signature is"):
            self._pk_from_sig(self.signature[:, :-1, :], _DIGEST)


if __name__ == "__main__":
    absltest.main()
