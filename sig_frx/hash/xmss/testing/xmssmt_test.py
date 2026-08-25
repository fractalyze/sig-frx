# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""XMSS-MT reproduces the reference implementation's key and signature digests.

Single-tree XMSS is the `d = 1` case of the same class, and `xmss_test`
gates that; what is left for here is everything the layer count changes — the
per-layer index split, keygen building only the top tree, the wider signature, and
the narrower index field.

Both sampled sets are `20/4`: four layers of height 5, reaching `2^20` signatures
for a keygen that builds 32 leaves. The fixture signs at `2^19`, which resolves to
a leaf of 0 on every layer but the top and a non-zero tree address on every layer
but the top — so an implementation that mixed up which of the two a layer takes
fails rather than passing on a degenerate split.
"""

from __future__ import annotations

import hashlib

import numpy as np
from absl.testing import absltest

from sig_frx.hash.xmss import params as p
from sig_frx.hash.xmss import xmss
from sig_frx.hash.xmss.testing import vectors as v

_KEYS: dict[int, tuple[xmss.Xmss, np.ndarray, np.ndarray]] = {}


def _key(oid: int) -> tuple[xmss.Xmss, np.ndarray, np.ndarray]:
    """The scheme, its public key, and a secret key wound to the fixture's index."""
    if oid not in _KEYS:
        scheme = xmss.mt_sha2(oid)
        public_key, secret_key = scheme.keygen(v.fixture_bytes(3 * scheme.params.n))
        _KEYS[oid] = (
            scheme,
            np.asarray(public_key),
            _at_index(
                scheme,
                np.asarray(secret_key),
                v.xmss_signing_index(scheme.params.height),
            ),
        )
    return _KEYS[oid]


def _at_index(scheme: xmss.Xmss, secret_key: np.ndarray, index: int) -> np.ndarray:
    """The same key with its index overwritten, the way `vectors_xmss` does it."""
    wound = secret_key.copy()
    wound[: scheme.params.index_bytes] = np.frombuffer(
        index.to_bytes(scheme.params.index_bytes, "big"), dtype=np.uint8
    )
    return wound


def _message() -> np.ndarray:
    return np.frombuffer(v.XMSS_MESSAGE, dtype=np.uint8)


def _digest10(artifact: np.ndarray) -> bytes:
    return hashlib.shake_128(bytes(np.asarray(artifact).reshape(-1))).digest(10)


class ReferenceDigestTest(absltest.TestCase):
    def test_the_public_key_matches(self) -> None:
        for oid, vectors in v.XMSSMT_REFERENCE.items():
            with self.subTest(oid=oid):
                scheme, public_key, _ = _key(oid)
                self.assertEqual(bytes(public_key[: scheme.params.n]), vectors.root)
                self.assertEqual(_digest10(public_key), vectors.digest_public_key)

    def test_the_signature_matches(self) -> None:
        for oid, vectors in v.XMSSMT_REFERENCE.items():
            with self.subTest(oid=oid):
                scheme, _, secret_key = _key(oid)
                signature, _ = scheme.sign(secret_key, _message())
                self.assertEqual(
                    _digest10(np.asarray(signature)), vectors.digest_signature
                )

    def test_the_randomizer_and_digest_match(self) -> None:
        # Both are functions of the whole index, before it is split across layers.
        for oid, vectors in v.XMSSMT_REFERENCE.items():
            with self.subTest(oid=oid):
                scheme, _, secret_key = _key(oid)
                signature = np.asarray(scheme.sign(secret_key, _message())[0])
                head = scheme.params.index_bytes
                self.assertEqual(
                    bytes(signature[head : head + scheme.params.n]), vectors.randomizer
                )

    def test_the_index_travels_in_the_signature(self) -> None:
        for oid in v.XMSSMT_REFERENCE:
            with self.subTest(oid=oid):
                scheme, _, secret_key = _key(oid)
                signature = np.asarray(scheme.sign(secret_key, _message())[0])
                self.assertEqual(
                    int.from_bytes(
                        bytes(signature[: scheme.params.index_bytes]), "big"
                    ),
                    v.xmss_signing_index(scheme.params.height),
                )


class LayerSplitTest(absltest.TestCase):
    """The derivation everything multi-tree follows from."""

    def test_the_split_matches_the_reference(self) -> None:
        for oid, vectors in v.XMSSMT_REFERENCE.items():
            with self.subTest(oid=oid):
                params = p.XMSSMT_PARAMETER_SETS[oid]
                index = v.xmss_signing_index(params.height)
                walk = xmss.locate(params, index)
                self.assertEqual(
                    tuple((position.tree, leaf) for position, leaf in walk),
                    vectors.layer_split,
                )

    def test_each_layer_is_numbered_by_its_own_depth(self) -> None:
        params = p.XMSSMT_PARAMETER_SETS[0x02]
        walk = xmss.locate(params, 0)
        self.assertEqual([position.layer for position, _ in walk], [0, 1, 2, 3])

    def test_the_index_is_consumed_h_prime_bits_at_a_time(self) -> None:
        # Lowest layer first, `h'` bits each. `h' = 5` here, so an index that fits
        # in five bits is entirely layer 0's leaf and leaves every tree at zero.
        params = p.XMSSMT_PARAMETER_SETS[0x02]
        walk = xmss.locate(params, 0b11111)
        self.assertEqual([leaf for _, leaf in walk], [31, 0, 0, 0])
        self.assertEqual([position.tree for position, _ in walk], [0, 0, 0, 0])

        # One bit higher spills into layer 1's leaf, and layer 0 moves to tree 1.
        walk = xmss.locate(params, 1 << params.tree_height)
        self.assertEqual([leaf for _, leaf in walk], [0, 1, 0, 0])
        self.assertEqual([position.tree for position, _ in walk], [1, 0, 0, 0])

    def test_single_tree_is_the_one_layer_case(self) -> None:
        # The claim the whole class rests on: XMSS is XMSS-MT at `d = 1`.
        params = p.XMSS_PARAMETER_SETS[0x01]
        walk = xmss.locate(params, 517)
        self.assertLen(walk, 1)
        self.assertEqual((walk[0][0].layer, walk[0][0].tree, walk[0][1]), (0, 0, 517))

    def test_an_index_past_the_structure_addresses_a_tree_that_is_not_there(
        self,
    ) -> None:
        params = p.XMSSMT_PARAMETER_SETS[0x02]
        walk = xmss.locate(params, 1 << params.height)
        self.assertEqual(walk[-1][0].tree, 1, "the top layer holds only tree 0")


class RoundTripTest(absltest.TestCase):
    def test_a_signature_verifies_under_its_own_key(self) -> None:
        for oid in v.XMSSMT_REFERENCE:
            with self.subTest(oid=oid):
                scheme, public_key, secret_key = _key(oid)
                signature, _ = scheme.sign(secret_key, _message())
                np.testing.assert_array_equal(
                    np.asarray(
                        scheme.verify(
                            public_key[None, :],
                            _message()[None, :],
                            np.asarray(signature)[None, :],
                        )
                    ),
                    [True],
                )

    def test_indices_in_different_subtrees_verify_in_one_batch(self) -> None:
        # The point of batching across signatures: entry `k` walks its own tree at
        # every layer, so a batch whose entries sit in different subtrees is still
        # `d` batched passes rather than `B · d` sequential ones.
        scheme, public_key, secret_key = _key(0x22)
        indices = [
            0,
            1 << scheme.params.tree_height,
            v.xmss_signing_index(scheme.params.height),
        ]
        messages = np.array([[1], [2], [3]], dtype=np.uint8)
        signatures = np.stack(
            [
                np.asarray(
                    scheme.sign(
                        _at_index(scheme, np.asarray(secret_key), index), messages[row]
                    )[0]
                )
                for row, index in enumerate(indices)
            ]
        )
        # The three sit in three different layer-0 trees, which is what makes this
        # more than a repeat of the single-tree batch case.
        self.assertLen(
            {xmss.locate(scheme.params, index)[0][0].tree for index in indices},
            3,
        )
        np.testing.assert_array_equal(
            np.asarray(
                scheme.verify(
                    np.broadcast_to(public_key, (3, scheme.public_key_size)),
                    messages,
                    signatures,
                )
            ),
            [True, True, True],
        )


class RejectionTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.scheme, self.public_key, self.secret_key = _key(0x22)
        self.signature = np.asarray(self.scheme.sign(self.secret_key, _message())[0])

    def _verdict(self, signature: np.ndarray) -> bool:
        return bool(
            np.asarray(
                self.scheme.verify(
                    self.public_key[None, :],
                    _message()[None, :],
                    signature[None, :],
                )
            )[0]
        )

    def test_the_unmodified_signature_is_accepted(self) -> None:
        self.assertTrue(self._verdict(self.signature))

    def test_a_tampered_layer_is_rejected(self) -> None:
        # One offset inside each layer's block, so no layer can be skipped.
        head = self.scheme.params.index_bytes + self.scheme.params.n
        stride = (
            self.scheme.params.wots.len + self.scheme.params.tree_height
        ) * self.scheme.params.n
        for layer in range(self.scheme.params.layers):
            with self.subTest(layer=layer):
                corrupt = self.signature.copy()
                corrupt[head + layer * stride] ^= 1
                self.assertFalse(self._verdict(corrupt))

    def test_an_index_in_another_subtree_is_rejected(self) -> None:
        # Moving the index by a whole subtree leaves layer 0's *leaf* where it was
        # and moves the trees above it — the case a single-tree scheme cannot
        # express, and the one a verifier that ignored the upper layers would pass.
        params = self.scheme.params
        index = v.xmss_signing_index(params.height)
        moved = index + (1 << (2 * params.tree_height))
        original = xmss.locate(params, index)
        walked = xmss.locate(params, moved)
        self.assertEqual(original[0][1], walked[0][1], "layer 0's leaf is unchanged")
        self.assertNotEqual(
            original[1][0].tree, walked[1][0].tree, "layer 1 sits in another tree"
        )

        relabelled = self.signature.copy()
        relabelled[: params.index_bytes] = np.frombuffer(
            moved.to_bytes(params.index_bytes, "big"), dtype=np.uint8
        )
        self.assertFalse(self._verdict(relabelled))

    def test_an_index_past_the_structure_is_rejected(self) -> None:
        relabelled = self.signature.copy()
        relabelled[: self.scheme.params.index_bytes] = np.frombuffer(
            (self.scheme.signatures_per_key + 1).to_bytes(
                self.scheme.params.index_bytes, "big"
            ),
            dtype=np.uint8,
        )
        self.assertFalse(self._verdict(relabelled))


class ParameterSetTest(absltest.TestCase):
    def test_the_table_carries_every_registered_oid(self) -> None:
        self.assertEqual(sorted(p.XMSSMT_PARAMETER_SETS), list(range(0x01, 0x39)))

    def test_the_two_oid_spaces_are_separate(self) -> None:
        # OID 2 names different sets in the two tables, which is why `mt_sha2` is a
        # separate lookup rather than a flag on `sha2`.
        self.assertEqual(p.XMSS_PARAMETER_SETS[0x02].name, "XMSS-SHA2_16_256")
        self.assertEqual(p.XMSSMT_PARAMETER_SETS[0x02].name, "XMSSMT-SHA2_20/4_256")
        self.assertNotEqual(xmss.sha2(0x02), xmss.mt_sha2(0x02))

    def test_the_sizes_are_the_reference_s(self) -> None:
        # `params.c`'s own derivations: `sig = index_bytes + n + d·wots_sig + h·n`,
        # `sk = index_bytes + 4n`, `pk = 2n`.
        for oid, sizes in ((0x02, (64, 131, 9251)), (0x22, (48, 99, 5403))):
            with self.subTest(oid=oid):
                scheme = xmss.mt_sha2(oid)
                self.assertEqual(
                    (
                        scheme.public_key_size,
                        scheme.secret_key_size,
                        scheme.signature_max_size,
                    ),
                    sizes,
                )

    def test_the_height_splits_evenly_across_the_layers(self) -> None:
        for params in p.XMSSMT_PARAMETER_SETS.values():
            self.assertEqual(
                params.tree_height * params.layers, params.height, params.name
            )

    def test_a_height_the_layers_do_not_divide_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "must divide"):
            p.XmssParams(0x99, "made-up", p.CoreHash.SHA2, 32, 32, 20, 3)

    def test_one_key_signs_two_to_the_h_messages(self) -> None:
        # Which is the argument for the multi-tree variant: this key reaches 2^20
        # signatures, and the keygen that produced it built 32 leaves.
        self.assertEqual(xmss.mt_sha2(0x02).signatures_per_key, 1 << 20)

    def test_an_unregistered_oid_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "no XMSS-MT parameter set"):
            xmss.mt_sha2(0x99)

    def test_a_set_needing_a_hash_hash_frx_lacks_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "SHAKE128"):
            xmss.mt_sha2(0x11)


class StateTest(absltest.TestCase):
    def test_sign_advances_the_index_across_layers(self) -> None:
        # Advancing is on the whole index, not per layer: the next signature may
        # sit in a different subtree at every level.
        scheme, _, secret_key = _key(0x22)
        index = v.xmss_signing_index(scheme.params.height)
        _, advanced = scheme.sign(secret_key, _message())
        self.assertEqual(
            int.from_bytes(
                bytes(np.asarray(advanced)[: scheme.params.index_bytes]), "big"
            ),
            index + 1,
        )

    def test_a_spent_key_refuses_to_sign(self) -> None:
        scheme, _, secret_key = _key(0x22)
        spent = _at_index(scheme, np.asarray(secret_key), scheme.signatures_per_key)
        with self.assertRaisesRegex(ValueError, "spent"):
            scheme.sign(spent, _message())


if __name__ == "__main__":
    absltest.main()
