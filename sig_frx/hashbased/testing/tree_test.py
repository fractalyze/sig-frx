# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The tree agrees with FIPS 205's recursion, and the two directions agree.

`root` is an iteration where Algorithm 9 is a recursion, so a naive transcription
of that recursion runs beside it and the two must produce the same root. The
authentication path is then checked the way it is actually used: reconstructing
from a leaf must land on the root the tree computed, for every leaf, and must not
land there when anything about the claim is altered.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest
from hash_frx import Sha256

from sig_frx.hashbased import adrs, tree
from sig_frx.hashbased.tweakable import Sha2TweakableHash

_N = 16
_HEIGHT = 3
_LEAVES = 1 << _HEIGHT
_POSITION = tree.TreePosition(layer=2, tree=5)
_ADDRESSES = tree.xmss_node_addresses(_POSITION, compressed=True)
_PK_SEED = np.frombuffer(bytes(range(_N)), dtype=np.uint8)


def _family() -> Sha2TweakableHash:
    return Sha2TweakableHash(Sha256(), n=_N, m=30)


def _leaves() -> np.ndarray:
    return np.arange(_LEAVES * _N, dtype=np.uint8).reshape(_LEAVES, _N)


def _spec_node(
    tweak: Sha2TweakableHash, leaves: np.ndarray, index: int, height: int
) -> np.ndarray:
    """Algorithm 9 lines 5 to 12, transcribed: the recursion, one node at a time."""
    if height == 0:
        return leaves[index]
    left = _spec_node(tweak, leaves, 2 * index, height - 1)
    right = _spec_node(tweak, leaves, 2 * index + 1, height - 1)
    address = adrs.encode_batch(
        adrs.hash_tree(
            layer=_POSITION.layer, tree=_POSITION.tree, height=height, index=index
        ),
        compressed=True,
    )
    pair = np.concatenate([left, right])[None, :]
    return np.asarray(tweak.h(_PK_SEED, address, pair))[0]


class RootTest(absltest.TestCase):
    def test_it_agrees_with_the_standards_recursion(self) -> None:
        tweak = _family()
        leaves = _leaves()
        self.assertEqual(
            bytes(np.asarray(tree.root(tweak, _PK_SEED, leaves, _ADDRESSES))),
            bytes(_spec_node(tweak, leaves, 0, _HEIGHT)),
        )

    def test_a_single_leaf_is_its_own_root(self) -> None:
        leaf = np.arange(_N, dtype=np.uint8)
        self.assertEqual(
            bytes(
                np.asarray(tree.root(_family(), _PK_SEED, leaf[None, :], _ADDRESSES))
            ),
            bytes(leaf),
        )

    def test_the_leaf_count_must_be_a_power_of_two(self) -> None:
        with self.assertRaisesRegex(ValueError, "power-of-two"):
            tree.root(
                _family(), _PK_SEED, np.zeros((3, _N), dtype=np.uint8), _ADDRESSES
            )

    def test_moving_the_tree_changes_the_root(self) -> None:
        # The address tweak is what stops a subtree computed at one position in
        # the hypertree from being replayed at another.
        tweak = _family()
        leaves = _leaves()
        elsewhere = tree.TreePosition(layer=2, tree=6)
        self.assertNotEqual(
            bytes(np.asarray(tree.root(tweak, _PK_SEED, leaves, _ADDRESSES))),
            bytes(
                np.asarray(
                    tree.root(
                        tweak,
                        _PK_SEED,
                        leaves,
                        tree.xmss_node_addresses(elsewhere, compressed=True),
                    )
                )
            ),
        )


class AuthPathTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tweak = _family()
        self.leaves = _leaves()
        self.root = np.asarray(tree.root(self.tweak, _PK_SEED, self.leaves, _ADDRESSES))

    def _reconstruct(
        self, leaves: np.ndarray, indices: np.ndarray, paths: np.ndarray
    ) -> np.ndarray:
        return np.asarray(
            tree.root_from_path(
                self.tweak, _PK_SEED, leaves, indices, paths, _ADDRESSES
            )
        )

    def test_every_leaf_reaches_the_root_from_its_path(self) -> None:
        paths = np.asarray(
            tree.auth_path(
                self.tweak,
                _PK_SEED,
                self.leaves,
                np.arange(_LEAVES),
                _HEIGHT,
                _ADDRESSES,
            )
        )
        self.assertEqual(paths.shape, (_LEAVES, _HEIGHT, _N))

        # The whole batch in one call, each entry with its own index and path —
        # the shape verification runs.
        got = self._reconstruct(self.leaves, np.arange(_LEAVES), paths)

        for index in range(_LEAVES):
            self.assertEqual(bytes(got[index]), bytes(self.root), f"leaf {index}")

    def test_a_leaf_reconstructed_under_the_wrong_index_misses(self) -> None:
        # The index decides the left/right order at every level, so claiming the
        # wrong one is a different computation — which is what stops a sibling
        # pair from being swapped.
        path = np.asarray(
            tree.auth_path(self.tweak, _PK_SEED, self.leaves, [3], _HEIGHT, _ADDRESSES)[
                0
            ]
        )
        got = self._reconstruct(self.leaves[3][None, :], np.array([2]), path[None, ...])
        self.assertNotEqual(bytes(got[0]), bytes(self.root))

    def test_a_tampered_leaf_misses(self) -> None:
        path = np.asarray(
            tree.auth_path(self.tweak, _PK_SEED, self.leaves, [5], _HEIGHT, _ADDRESSES)[
                0
            ]
        )
        leaf = self.leaves[5].copy()
        leaf[0] ^= 1
        got = self._reconstruct(leaf[None, :], np.array([5]), path[None, ...])
        self.assertNotEqual(bytes(got[0]), bytes(self.root))

    def test_a_tampered_path_misses(self) -> None:
        path = np.array(
            tree.auth_path(self.tweak, _PK_SEED, self.leaves, [5], _HEIGHT, _ADDRESSES)[
                0
            ]
        )
        path[1][0] ^= 1
        got = self._reconstruct(self.leaves[5][None, :], np.array([5]), path[None, ...])
        self.assertNotEqual(bytes(got[0]), bytes(self.root))

    def test_one_wrong_entry_does_not_disturb_the_others(self) -> None:
        # A batch decides per entry: a bad signature in the batch must fail alone.
        paths = np.asarray(
            tree.auth_path(
                self.tweak, _PK_SEED, self.leaves, np.arange(4), _HEIGHT, _ADDRESSES
            )
        )
        leaves = self.leaves[:4].copy()
        leaves[2][0] ^= 1
        got = self._reconstruct(leaves, np.arange(4), paths)
        for index in range(4):
            if index == 2:
                self.assertNotEqual(bytes(got[index]), bytes(self.root))
            else:
                self.assertEqual(bytes(got[index]), bytes(self.root), f"leaf {index}")

    def test_a_leaf_outside_the_tree_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside a forest"):
            tree.auth_path(
                self.tweak, _PK_SEED, self.leaves, [_LEAVES], _HEIGHT, _ADDRESSES
            )

    def test_mismatched_batch_lengths_are_an_error(self) -> None:
        path = np.asarray(
            tree.auth_path(self.tweak, _PK_SEED, self.leaves, [0], _HEIGHT, _ADDRESSES)[
                0
            ]
        )
        with self.assertRaisesRegex(ValueError, "one index and one path per leaf"):
            self._reconstruct(self.leaves[:2], np.array([0]), path[None, ...])


if __name__ == "__main__":
    absltest.main()
