# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The XMSS layer and the hypertree above it, FIPS 205 §6 and §7.

The claim beyond the standard is the batching: a tree's `2^h'` WOTS+ key pairs
are walked in one pass rather than one at a time, and a hypertree verifies `B`
signatures layer by layer rather than signature by signature. Both are checked by
running the standard's own per-key-pair and per-signature form beside them.

The round trips are the properties these layers exist for — an XMSS signature
implies its tree's root, and a hypertree signature implies the published one — and
each is asserted alongside a case that must miss, since a construction ignoring
its inputs would satisfy a round trip on its own.

Small parameters (`h' = 2`, `d = 2`) rather than a real set's `h' = 9`, `d = 7`:
the arithmetic under test does not care, and a real tree is 512 WOTS+ key pairs
whose per-key-pair reference form would dominate the suite.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest
from hash_frx.sha256 import Sha256

from sig_frx.hashbased import hypertree, tree, wots, xmss
from sig_frx.hashbased.tweakable import Sha2TweakableHash

_N = 16
_WOTS = wots.WotsParams(n=_N)
_HEIGHT = 2
_POSITION = tree.TreePosition(layer=0, tree=3)
_PK_SEED = np.frombuffer(bytes(range(_N)), dtype=np.uint8)
_SK_SEED = np.frombuffer(bytes(range(100, 100 + _N)), dtype=np.uint8)
_MESSAGE = np.frombuffer(bytes((i * 37 + 11) % 256 for i in range(_N)), dtype=np.uint8)


def _family() -> Sha2TweakableHash:
    return Sha2TweakableHash(Sha256(), n=_N, m=30)


class XmssLeafTest(absltest.TestCase):
    def test_one_batched_walk_gives_the_same_leaves_as_one_walk_each(self) -> None:
        # The claim: no key pair depends on another, so all of them advance in
        # the same `w − 1` hashes. The reference form is the standard's — one
        # `wots_pkGen` per key pair.
        tweak = _family()
        got = np.asarray(
            xmss.leaves(tweak, _WOTS, _PK_SEED, _SK_SEED, _POSITION, _HEIGHT)
        )
        self.assertEqual(got.shape, (1 << _HEIGHT, _N))
        for index in range(1 << _HEIGHT):
            position = wots.WotsPosition(_POSITION.layer, _POSITION.tree, index)
            expected = np.asarray(
                wots.pk_gen(
                    tweak,
                    _WOTS,
                    _PK_SEED,
                    _SK_SEED,
                    position,
                    wots.fips205_compression(tweak, _PK_SEED, position),
                )
            )[0]
            self.assertEqual(bytes(got[index]), bytes(expected), f"key pair {index}")


class XmssSignatureTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tweak = _family()
        self.root = np.asarray(
            xmss.root(self.tweak, _WOTS, _PK_SEED, _SK_SEED, _POSITION, _HEIGHT)
        )

    def _pk_from_sig(
        self, signature: np.ndarray, message: np.ndarray, leaf: int
    ) -> bytes:
        return bytes(
            np.asarray(
                xmss.pk_from_sig(
                    self.tweak,
                    _WOTS,
                    signature[None, ...],
                    message[None, :],
                    _PK_SEED,
                    _POSITION,
                    [leaf],
                )
            )[0]
        )

    def test_every_leaf_signs_and_reconstructs_the_root(self) -> None:
        for leaf in range(1 << _HEIGHT):
            signature = np.asarray(
                xmss.sign(
                    self.tweak,
                    _WOTS,
                    _MESSAGE,
                    _PK_SEED,
                    _SK_SEED,
                    _POSITION,
                    _HEIGHT,
                    leaf,
                )
            )
            self.assertEqual(signature.shape, (_WOTS.len + _HEIGHT, _N))
            self.assertEqual(
                self._pk_from_sig(signature, _MESSAGE, leaf),
                bytes(self.root),
                f"leaf {leaf}",
            )

    def test_the_wrong_leaf_index_does_not_reconstruct(self) -> None:
        signature = np.asarray(
            xmss.sign(
                self.tweak, _WOTS, _MESSAGE, _PK_SEED, _SK_SEED, _POSITION, _HEIGHT, 1
            )
        )
        self.assertNotEqual(self._pk_from_sig(signature, _MESSAGE, 2), bytes(self.root))

    def test_another_message_does_not_reconstruct(self) -> None:
        signature = np.asarray(
            xmss.sign(
                self.tweak, _WOTS, _MESSAGE, _PK_SEED, _SK_SEED, _POSITION, _HEIGHT, 1
            )
        )
        other = _MESSAGE.copy()
        other[0] ^= 1
        self.assertNotEqual(self._pk_from_sig(signature, other, 1), bytes(self.root))

    def test_a_batch_reconstructs_each_entry_under_its_own_leaf(self) -> None:
        # What a hypertree layer hands it: independent claims, one call.
        leaves = [0, 3, 1]
        signatures = np.stack(
            [
                np.asarray(
                    xmss.sign(
                        self.tweak,
                        _WOTS,
                        _MESSAGE,
                        _PK_SEED,
                        _SK_SEED,
                        _POSITION,
                        _HEIGHT,
                        leaf,
                    )
                )
                for leaf in leaves
            ]
        )
        messages = np.stack([_MESSAGE] * len(leaves))
        got = np.asarray(
            xmss.pk_from_sig(
                self.tweak,
                _WOTS,
                signatures,
                messages,
                _PK_SEED,
                tree.TreePosition(
                    layer=_POSITION.layer, tree=np.full(len(leaves), _POSITION.tree)
                ),
                leaves,
            )
        )
        for index, leaf in enumerate(leaves):
            self.assertEqual(bytes(got[index]), bytes(self.root), f"leaf {leaf}")


class HypertreeTest(absltest.TestCase):
    PARAMS = hypertree.HypertreeParams(layers=2, tree_height=_HEIGHT, wots=_WOTS)

    def setUp(self) -> None:
        super().setUp()
        self.tweak = _family()
        # The top layer's root is the published key: one tree, at the top layer.
        self.pk_root = np.asarray(
            xmss.root(
                self.tweak,
                _WOTS,
                _PK_SEED,
                _SK_SEED,
                tree.TreePosition(layer=self.PARAMS.layers - 1, tree=0),
                _HEIGHT,
            )
        )

    def _sign(self, tree_index: int, leaf_index: int) -> np.ndarray:
        return np.asarray(
            hypertree.sign(
                self.tweak,
                self.PARAMS,
                _MESSAGE,
                _PK_SEED,
                _SK_SEED,
                tree_index,
                leaf_index,
            )
        )

    def _verify(
        self,
        signatures: np.ndarray,
        messages: np.ndarray,
        trees: list[int],
        leaves: list[int],
    ) -> list[bool]:
        return [
            bool(v)
            for v in np.asarray(
                hypertree.verify(
                    self.tweak,
                    self.PARAMS,
                    signatures,
                    messages,
                    _PK_SEED,
                    trees,
                    leaves,
                    self.pk_root,
                )
            )
        ]

    def test_a_signature_climbs_to_the_published_root(self) -> None:
        signature = self._sign(tree_index=2, leaf_index=1)
        self.assertEqual(
            signature.shape,
            (self.PARAMS.layers, self.PARAMS.signature_values, _N),
        )
        self.assertEqual(
            self._verify(signature[None, ...], _MESSAGE[None, :], [2], [1]), [True]
        )

    def test_every_position_in_the_hypertree_verifies(self) -> None:
        # `h = d · h'` index bits, so every one of them addresses a real leaf.
        positions = [
            (t, leaf) for t in range(1 << _HEIGHT) for leaf in range(1 << _HEIGHT)
        ]
        signatures = np.stack([self._sign(t, leaf) for t, leaf in positions])
        messages = np.stack([_MESSAGE] * len(positions))
        verdicts = self._verify(
            signatures,
            messages,
            [t for t, _ in positions],
            [leaf for _, leaf in positions],
        )
        self.assertEqual(verdicts, [True] * len(positions))

    def test_one_bad_signature_in_a_batch_fails_alone(self) -> None:
        # The property a batched verifier quietly loses: a verdict is per entry,
        # and a claim that misses at any layer misses at the top.
        positions = [(0, 0), (1, 2), (3, 1)]
        signatures = np.stack([self._sign(t, leaf) for t, leaf in positions])
        signatures[1, 0, 0, 0] ^= 1
        verdicts = self._verify(
            signatures,
            np.stack([_MESSAGE] * len(positions)),
            [t for t, _ in positions],
            [leaf for _, leaf in positions],
        )
        self.assertEqual(verdicts, [True, False, True])

    def test_a_signature_claimed_at_another_position_fails(self) -> None:
        signature = self._sign(tree_index=2, leaf_index=1)
        self.assertEqual(
            self._verify(signature[None, ...], _MESSAGE[None, :], [2], [0]), [False]
        )
        self.assertEqual(
            self._verify(signature[None, ...], _MESSAGE[None, :], [1], [1]), [False]
        )

    def test_another_message_fails(self) -> None:
        signature = self._sign(tree_index=2, leaf_index=1)
        other = _MESSAGE.copy()
        other[0] ^= 1
        self.assertEqual(
            self._verify(signature[None, ...], other[None, :], [2], [1]), [False]
        )

    def test_a_misshapen_signature_batch_is_an_error(self) -> None:
        signature = self._sign(tree_index=0, leaf_index=0)
        with self.assertRaisesRegex(ValueError, "hypertree signature batch"):
            self._verify(signature[None, :-1, ...], _MESSAGE[None, :], [0], [0])


class TwoPublicSeedsTest(absltest.TestCase):
    """A verification batch spans public keys, so `PK.seed` varies per entry.

    Every hash under the seam is tweaked with `PK.seed`, and the batch a verifier
    holds carries one per signature. The entries widen unevenly on the way down —
    a claim becomes `len` WOTS+ chains — so the seed has to widen with them, and a
    version that lined the wrong seed up with a row would be self-consistent and
    wrong. Two seeds is the smallest batch that can tell.
    """

    PARAMS = hypertree.HypertreeParams(layers=2, tree_height=_HEIGHT, wots=_WOTS)
    SEEDS = (_PK_SEED, _PK_SEED ^ 0xFF)
    POSITIONS = ((2, 1), (0, 3))

    def setUp(self) -> None:
        super().setUp()
        self.tweak = _family()
        self.roots = np.stack(
            [
                np.asarray(
                    xmss.root(
                        self.tweak,
                        _WOTS,
                        seed,
                        _SK_SEED,
                        tree.TreePosition(layer=self.PARAMS.layers - 1, tree=0),
                        _HEIGHT,
                    )
                )
                for seed in self.SEEDS
            ]
        )
        self.signatures = np.stack(
            [
                np.asarray(
                    hypertree.sign(
                        self.tweak,
                        self.PARAMS,
                        _MESSAGE,
                        seed,
                        _SK_SEED,
                        tree_index,
                        leaf_index,
                    )
                )
                for seed, (tree_index, leaf_index) in zip(
                    self.SEEDS, self.POSITIONS, strict=True
                )
            ]
        )

    def _verify(self, seeds: np.ndarray, roots: np.ndarray) -> list[bool]:
        return [
            bool(v)
            for v in np.asarray(
                hypertree.verify(
                    self.tweak,
                    self.PARAMS,
                    self.signatures,
                    np.stack([_MESSAGE] * len(self.SEEDS)),
                    seeds,
                    [tree_index for tree_index, _ in self.POSITIONS],
                    [leaf for _, leaf in self.POSITIONS],
                    roots,
                )
            )
        ]

    def test_each_entry_verifies_under_its_own_seed(self) -> None:
        self.assertNotEqual(bytes(self.roots[0]), bytes(self.roots[1]))
        self.assertEqual(self._verify(np.stack(self.SEEDS), self.roots), [True, True])

    def test_swapping_the_seeds_fails_both(self) -> None:
        swapped = np.stack(self.SEEDS[::-1])
        self.assertEqual(self._verify(swapped, self.roots), [False, False])


if __name__ == "__main__":
    absltest.main()
