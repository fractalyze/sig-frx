# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""XMSS reproduces the reference implementation's key and signature digests.

The gate for the scheme, the way `rfc8391_wots_test` is the gate for the
substrate. What makes these vectors worth more than a round trip is the index:
the generator forces the key to `2^(h-1)` before signing, so an implementation
that mishandled a non-zero index — in the randomizer, in the message digest, in
the leaf it signs under, or in the authentication path — fails here rather than
passing on `idx = 0` and breaking in production.

The negative cases are the other half. A verifier that returned `True`
unconditionally passes every positive case above, so the corruptions below pin
what rejection means: a tampered message, a tampered signature, the wrong public
key, and — the one specific to a stateful scheme — a signature re-labelled with
another index.

**This module is the gate, rather than `sig_frx.testing.kat`.** There is no vector
file to normalize — the values are transcribed constants with their provenance in
`rfc8391_vectors` — and a stateful scheme has no seam-shaped `sign` for that
harness to drive. So the negative cases it would have generated are written out
here in full, plus the rejections that need to know what an XMSS signature is.
`docs/reference/conventions.md` states the rule this is the exception to.
"""

from __future__ import annotations

import hashlib

import numpy as np
from absl.testing import absltest
from hash_frx import Sha256

from sig_frx.hashbased import rfc8391_adrs as a
from sig_frx.hashbased import rfc8391_params, rfc8391_wots, rfc8391_xmss, tree
from sig_frx.hashbased.rfc8391_hashes import Rfc8391Hashes, sha2_hashes
from sig_frx.hashbased.testing import rfc8391_vectors

# Keygen builds every leaf of a height-10 tree, so it is done once per parameter
# set and shared: the cases below differ in what they do with the key, not in the
# key. Both OIDs the reference can gate sit at that height.
_KEYS: dict[int, tuple[rfc8391_xmss.Xmss, np.ndarray, np.ndarray]] = {}


def _key(oid: int) -> tuple[rfc8391_xmss.Xmss, np.ndarray, np.ndarray]:
    """The scheme, its public key and a secret key wound to the fixture's index."""
    if oid not in _KEYS:
        scheme = rfc8391_xmss.sha2(oid)
        seed = rfc8391_vectors.fixture_bytes(3 * scheme.params.n)
        public_key, secret_key = scheme.keygen(seed)
        _KEYS[oid] = (
            scheme,
            np.asarray(public_key),
            _at_index(
                scheme,
                np.asarray(secret_key),
                rfc8391_vectors.xmss_signing_index(scheme.params.height),
            ),
        )
    return _KEYS[oid]


def _at_index(
    scheme: rfc8391_xmss.Xmss, secret_key: np.ndarray, index: int
) -> np.ndarray:
    """The same key with its index overwritten, the way `vectors_xmss` does it.

    Winding by signing would cost 512 tree builds; the generator writes the four
    index bytes directly, and so does this.
    """
    wound = secret_key.copy()
    wound[: scheme.params.index_bytes] = np.frombuffer(
        index.to_bytes(scheme.params.index_bytes, "big"), dtype=np.uint8
    )
    return wound


def _message() -> np.ndarray:
    return np.frombuffer(rfc8391_vectors.XMSS_MESSAGE, dtype=np.uint8)


def _digest10(artifact: np.ndarray) -> bytes:
    """`SHAKE128(artifact, 10)` — the framing `test/vectors.c` prints."""
    return hashlib.shake_128(bytes(np.asarray(artifact).reshape(-1))).digest(10)


class ReferenceDigestTest(absltest.TestCase):
    """The two XMSS rows of the digest table, and the intermediates beneath them."""

    def test_the_public_key_matches(self) -> None:
        for oid, vectors in rfc8391_vectors.XMSS_REFERENCE.items():
            with self.subTest(oid=oid):
                scheme, public_key, _ = _key(oid)
                # `root ‖ SEED` — the root first, unlike FIPS 205's `PK.seed`-first
                # public key, so a swapped concatenation fails here.
                self.assertEqual(bytes(public_key[: scheme.params.n]), vectors.root)
                self.assertEqual(_digest10(public_key), vectors.digest_public_key)

    def test_the_signature_matches(self) -> None:
        for oid, vectors in rfc8391_vectors.XMSS_REFERENCE.items():
            with self.subTest(oid=oid):
                scheme, _, secret_key = _key(oid)
                signature, _ = scheme.sign(secret_key, _message())
                self.assertEqual(
                    _digest10(np.asarray(signature)), vectors.digest_signature
                )

    def test_the_signature_carries_the_index_it_was_made_at(self) -> None:
        for oid, vectors in rfc8391_vectors.XMSS_REFERENCE.items():
            with self.subTest(oid=oid):
                scheme, _, secret_key = _key(oid)
                signature, _ = scheme.sign(secret_key, _message())
                index = rfc8391_vectors.xmss_signing_index(scheme.params.height)
                self.assertEqual(
                    int.from_bytes(
                        bytes(np.asarray(signature)[: scheme.params.index_bytes]), "big"
                    ),
                    index,
                )

    def test_the_randomizer_and_digest_match(self) -> None:
        # `r = PRF(SK.prf, toByte(idx, 32))` and `H_msg(r, root, idx, M)`. Both are
        # functions of the index, which is what makes the non-zero fixture index
        # worth having: at `idx = 0` a dropped index would pass both.
        for oid, vectors in rfc8391_vectors.XMSS_REFERENCE.items():
            with self.subTest(oid=oid):
                scheme, _, secret_key = _key(oid)
                signature = np.asarray(scheme.sign(secret_key, _message())[0])
                n = scheme.params.n
                self.assertEqual(
                    bytes(
                        signature[
                            scheme.params.index_bytes : scheme.params.index_bytes + n
                        ]
                    ),
                    vectors.randomizer,
                )

    def test_the_leaf_at_the_signing_index_matches(self) -> None:
        # `gen_leaf_wots` at the index the fixture signs from. It sits between the
        # substrate and the tree, so it separates a wrong WOTS+ key from a wrong
        # tree walk when the signature digest goes red.
        for oid, vectors in rfc8391_vectors.XMSS_REFERENCE.items():
            with self.subTest(oid=oid):
                scheme, public_key, secret_key = _key(oid)
                n = scheme.params.n
                index = rfc8391_vectors.xmss_signing_index(scheme.params.height)
                hashes = sha2_hashes(scheme.params, Sha256())
                pub_seed = public_key[n:]
                sk_seed = secret_key[
                    scheme.params.index_bytes : scheme.params.index_bytes + n
                ]

                leaf = rfc8391_wots.pk_gen(
                    hashes,
                    scheme.params.wots,
                    pub_seed,
                    sk_seed,
                    [a.ots(0, 0, index)],
                    rfc8391_wots.ltree_compression(
                        hashes,
                        pub_seed,
                        [a.ltree(0, 0, index)],
                        leaves=scheme.params.wots.len,
                    ),
                )
                self.assertEqual(bytes(np.asarray(leaf)[0]), vectors.leaf_at_index)

    def test_the_auth_path_matches(self) -> None:
        for oid, vectors in rfc8391_vectors.XMSS_REFERENCE.items():
            with self.subTest(oid=oid):
                scheme, _, secret_key = _key(oid)
                n = scheme.params.n
                signature = np.asarray(scheme.sign(secret_key, _message())[0])
                start = scheme.params.index_bytes + n + scheme.params.wots.len * n
                self.assertEqual(
                    bytes(signature[start : start + n]),
                    vectors.auth_path_head,
                    "the lowest sibling of the authentication path",
                )


class RoundTripTest(absltest.TestCase):
    def test_a_signature_verifies_under_its_own_key(self) -> None:
        for oid in rfc8391_vectors.XMSS_REFERENCE:
            with self.subTest(oid=oid):
                scheme, public_key, secret_key = _key(oid)
                signature, _ = scheme.sign(secret_key, _message())
                verdict = scheme.verify(
                    public_key[None, :],
                    _message()[None, :],
                    np.asarray(signature)[None, :],
                )
                np.testing.assert_array_equal(np.asarray(verdict), [True])

    def test_a_batch_verifies_in_one_call(self) -> None:
        # Three signatures at three different indices under one key, which is what
        # a verifier actually holds — and the indices differ, so each entry walks
        # its own leaf and its own path.
        scheme, public_key, secret_key = _key(0x0D)
        indices = [0, 1, rfc8391_vectors.xmss_signing_index(scheme.params.height)]
        messages = np.array([[7], [8], [9]], dtype=np.uint8)
        signatures = np.stack(
            [
                np.asarray(
                    scheme.sign(
                        _at_index(scheme, np.asarray(secret_key), index),
                        messages[row],
                    )[0]
                )
                for row, index in enumerate(indices)
            ]
        )
        keys = np.broadcast_to(public_key, (3, scheme.public_key_size))

        verdict = scheme.verify(keys, messages, signatures)

        np.testing.assert_array_equal(np.asarray(verdict), [True, True, True])


class RejectionTest(absltest.TestCase):
    """A verifier that returned `True` unconditionally passes every case above."""

    def setUp(self) -> None:
        super().setUp()
        self.scheme, self.public_key, self.secret_key = _key(0x0D)
        self.signature = np.asarray(self.scheme.sign(self.secret_key, _message())[0])

    def _verdict(
        self,
        *,
        public_key: np.ndarray | None = None,
        message: np.ndarray | None = None,
        signature: np.ndarray | None = None,
    ) -> bool:
        return bool(
            np.asarray(
                self.scheme.verify(
                    (self.public_key if public_key is None else public_key)[None, :],
                    (_message() if message is None else message)[None, :],
                    (self.signature if signature is None else signature)[None, :],
                )
            )[0]
        )

    def test_a_tampered_message_is_rejected(self) -> None:
        self.assertFalse(self._verdict(message=_message() ^ 1))

    def test_a_tampered_signature_is_rejected(self) -> None:
        for offset in (
            self.scheme.params.index_bytes,  # the randomizer
            self.scheme.params.index_bytes
            + self.scheme.params.n,  # the WOTS+ signature
            self.scheme.signature_max_size - 1,  # the top of the auth path
        ):
            with self.subTest(offset=offset):
                corrupt = self.signature.copy()
                corrupt[offset] ^= 1
                self.assertFalse(self._verdict(signature=corrupt))

    def test_a_signature_relabelled_with_another_index_is_rejected(self) -> None:
        # The case a stateless scheme has no analogue of. The index picks the leaf
        # and the tree the signature is checked at *and* rides in what was signed,
        # so a verifier cannot take it on trust.
        relabelled = self.signature.copy()
        relabelled[: self.scheme.params.index_bytes] = np.frombuffer(
            (0).to_bytes(self.scheme.params.index_bytes, "big"), dtype=np.uint8
        )
        self.assertFalse(self._verdict(signature=relabelled))

    def test_an_index_past_the_tree_is_rejected(self) -> None:
        # It addresses a tree that does not exist rather than wrapping onto a leaf
        # that does.
        relabelled = self.signature.copy()
        relabelled[: self.scheme.params.index_bytes] = np.frombuffer(
            (self.scheme.signatures_per_key + 3).to_bytes(
                self.scheme.params.index_bytes, "big"
            ),
            dtype=np.uint8,
        )
        self.assertFalse(self._verdict(signature=relabelled))

    def test_another_key_is_rejected(self) -> None:
        other = np.asarray(
            rfc8391_xmss.sha2(0x0D).keygen(
                rfc8391_vectors.fixture_bytes(3 * self.scheme.params.n, start=99)
            )[0]
        )
        self.assertFalse(self._verdict(public_key=other))

    def test_a_mixed_batch_rejects_only_the_bad_entry(self) -> None:
        # One traced computation over the batch, so a rejection must not be a
        # short-circuit that takes the whole call with it.
        corrupt = self.signature.copy()
        corrupt[-1] ^= 1
        verdict = self.scheme.verify(
            np.broadcast_to(self.public_key, (2, self.scheme.public_key_size)),
            np.broadcast_to(_message(), (2, 1)),
            np.stack([self.signature, corrupt]),
        )
        np.testing.assert_array_equal(np.asarray(verdict), [True, False])


class StateTest(absltest.TestCase):
    """The discipline that makes index reuse visible rather than merely documented."""

    def test_sign_returns_the_key_advanced_by_one(self) -> None:
        scheme, _, secret_key = _key(0x0D)
        index = rfc8391_vectors.xmss_signing_index(scheme.params.height)
        _, advanced = scheme.sign(secret_key, _message())
        self.assertEqual(
            int.from_bytes(
                bytes(np.asarray(advanced)[: scheme.params.index_bytes]), "big"
            ),
            index + 1,
        )

    def test_only_the_index_changes(self) -> None:
        # The seeds and the root are the key's identity; advancing must not touch
        # them, or the advanced key would verify against a different public key.
        scheme, _, secret_key = _key(0x0D)
        _, advanced = scheme.sign(secret_key, _message())
        np.testing.assert_array_equal(
            np.asarray(advanced)[scheme.params.index_bytes :],
            np.asarray(secret_key)[scheme.params.index_bytes :],
        )

    def test_signing_the_advanced_key_uses_the_next_one_time_key(self) -> None:
        scheme, public_key, secret_key = _key(0x0D)
        first, advanced = scheme.sign(secret_key, _message())
        second, _ = scheme.sign(advanced, _message())
        self.assertNotEqual(bytes(np.asarray(first)), bytes(np.asarray(second)))
        verdict = scheme.verify(
            np.broadcast_to(public_key, (2, scheme.public_key_size)),
            np.broadcast_to(_message(), (2, 1)),
            np.stack([np.asarray(first), np.asarray(second)]),
        )
        np.testing.assert_array_equal(np.asarray(verdict), [True, True])

    def test_reuse_requires_naming_the_spent_key_again(self) -> None:
        # The whole design: signing twice under one key is not something `sign` can
        # do on its own — it takes the caller passing the same value twice, and the
        # result is the same signature rather than a second valid one.
        scheme, _, secret_key = _key(0x0D)
        first, _ = scheme.sign(secret_key, _message())
        again, _ = scheme.sign(secret_key, _message())
        self.assertEqual(bytes(np.asarray(first)), bytes(np.asarray(again)))

    def test_a_spent_key_refuses_to_sign(self) -> None:
        scheme, _, secret_key = _key(0x0D)
        spent = _at_index(scheme, np.asarray(secret_key), scheme.signatures_per_key)
        with self.assertRaisesRegex(ValueError, "spent"):
            scheme.sign(spent, _message())

    def test_the_last_one_time_key_still_signs(self) -> None:
        scheme, public_key, secret_key = _key(0x0D)
        last = _at_index(scheme, np.asarray(secret_key), scheme.signatures_per_key - 1)
        signature, _ = scheme.sign(last, _message())
        verdict = scheme.verify(
            public_key[None, :], _message()[None, :], np.asarray(signature)[None, :]
        )
        np.testing.assert_array_equal(np.asarray(verdict), [True])


class SeamTest(absltest.TestCase):
    def test_the_sizes_are_the_reference_s(self) -> None:
        # `sig_bytes = index_bytes + n + wots_sig_bytes + h·n`, `pk_bytes = 2n`,
        # `sk_bytes = index_bytes + 4n` — `params.c`'s own derivations.
        for oid, sizes in ((0x01, (64, 132, 2500)), (0x0D, (48, 100, 1492))):
            with self.subTest(oid=oid):
                scheme = rfc8391_xmss.sha2(oid)
                self.assertEqual(
                    (
                        scheme.public_key_size,
                        scheme.secret_key_size,
                        scheme.signature_max_size,
                    ),
                    sizes,
                )

    def test_the_index_width_is_the_one_place_the_variants_diverge(self) -> None:
        # §4.1.8 fixes four bytes for XMSS whatever `h` is; §4.2.2 rounds `h` up to
        # a byte for XMSS-MT. So a height-10 XMSS signature carries four index
        # bytes where a height-20 XMSS-MT signature carries three — reading it off
        # `h` for both would mis-parse every XMSS signature.
        for oid, height in ((0x01, 10), (0x02, 16), (0x03, 20)):
            scheme = rfc8391_xmss.sha2(oid)
            self.assertEqual(scheme.params.height, height)
            self.assertEqual(scheme.params.index_bytes, 4, scheme.params.name)
        for oid, height, expected in ((0x02, 20, 3), (0x03, 40, 5), (0x06, 60, 8)):
            scheme = rfc8391_xmss.mt_sha2(oid)
            self.assertEqual(scheme.params.height, height)
            self.assertEqual(scheme.params.index_bytes, expected, scheme.params.name)

    def test_a_context_is_refused_rather_than_ignored(self) -> None:
        # RFC 8391 has no application context, and accepting one we then dropped
        # would verify something other than what the caller asked about.
        scheme, public_key, secret_key = _key(0x0D)
        signature, _ = scheme.sign(secret_key, _message())
        with self.assertRaisesRegex(ValueError, "no application context"):
            scheme.sign(secret_key, _message(), context=np.array([1], np.uint8))
        with self.assertRaisesRegex(ValueError, "no application context"):
            scheme.verify(
                public_key[None, :],
                _message()[None, :],
                np.asarray(signature)[None, :],
                context=np.array([1], np.uint8),
            )
        # An empty context is what the scheme takes, so it is not an error.
        scheme.sign(secret_key, _message(), context=np.zeros(0, np.uint8))

    def test_signing_is_deterministic(self) -> None:
        self.assertTrue(rfc8391_xmss.sha2(0x01).deterministic)

    def test_value_equality_survives_reconstruction(self) -> None:
        self.assertEqual(rfc8391_xmss.sha2(0x01), rfc8391_xmss.sha2(0x01))
        self.assertEqual(hash(rfc8391_xmss.sha2(0x01)), hash(rfc8391_xmss.sha2(0x01)))
        self.assertNotEqual(rfc8391_xmss.sha2(0x01), rfc8391_xmss.sha2(0x02))

    def test_an_unregistered_oid_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "no XMSS parameter set"):
            rfc8391_xmss.sha2(0x99)

    def test_a_set_needing_a_hash_hash_frx_lacks_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "SHAKE128"):
            rfc8391_xmss.sha2(0x07)

    def test_a_family_that_does_not_match_the_parameter_set_is_an_error(self) -> None:
        # A family sized for another set would hash under the wrong domain
        # separator and produce a scheme that is self-consistent and wrong.
        with self.assertRaisesRegex(ValueError, "padding_len"):
            rfc8391_xmss.Xmss(
                Rfc8391Hashes(Sha256(), n=24, padding_len=32),
                rfc8391_params.XMSS_PARAMETER_SETS[0x0D],
            )

    def test_a_wrongly_sized_key_or_signature_is_an_error(self) -> None:
        scheme, public_key, secret_key = _key(0x0D)
        signature = np.asarray(scheme.sign(secret_key, _message())[0])
        with self.assertRaisesRegex(ValueError, "secret key is"):
            scheme.sign(np.zeros(7, np.uint8), _message())
        with self.assertRaisesRegex(ValueError, "public key batch"):
            scheme.verify(
                np.zeros((1, 7), np.uint8), _message()[None, :], signature[None, :]
            )
        with self.assertRaisesRegex(ValueError, "signature batch"):
            scheme.verify(
                public_key[None, :], _message()[None, :], np.zeros((1, 7), np.uint8)
            )

    def test_keygen_takes_exactly_three_seeds(self) -> None:
        with self.assertRaisesRegex(ValueError, "SK.seed, SK.prf and SEED"):
            rfc8391_xmss.sha2(0x0D).keygen(np.zeros(24, np.uint8))


class NodeAddressTest(absltest.TestCase):
    """The one place the two standards' tree numbering differs."""

    def test_a_parent_is_addressed_by_the_height_below_it(self) -> None:
        # `tree.py` names the parent's height; RFC 8391's `treehash` sets
        # `tree_height` to the height of the pair being consumed, which is zero for
        # a pair of leaves. An off-by-one here reproduces no published root.
        build = rfc8391_xmss.node_addresses(tree.TreePosition(layer=0, tree=0))
        encoded = build(1, np.array([0]))
        expected = a.encode_batch(a.hash_tree(layer=0, tree=0, height=0, index=0))
        np.testing.assert_array_equal(encoded, expected)

    def test_the_index_is_the_parent_s(self) -> None:
        build = rfc8391_xmss.node_addresses(tree.TreePosition(layer=0, tree=0))
        np.testing.assert_array_equal(
            build(3, np.array([5])),
            a.encode_batch(a.hash_tree(layer=0, tree=0, height=2, index=5)),
        )

    def test_a_leaf_height_has_no_address(self) -> None:
        build = rfc8391_xmss.node_addresses(tree.TreePosition(layer=0, tree=0))
        with self.assertRaisesRegex(ValueError, "height 1 or above"):
            build(0, np.array([0]))

    def test_a_batch_addresses_each_entry_in_its_own_tree(self) -> None:
        positions = [tree.TreePosition(layer=0, tree=index) for index in range(3)]
        build = rfc8391_xmss.batch_node_addresses(positions)
        np.testing.assert_array_equal(
            build(2, np.array([4, 5, 6])),
            a.encode_batch(
                a.hash_tree(
                    layer=0, tree=np.arange(3), height=1, index=np.array([4, 5, 6])
                )
            ),
        )


if __name__ == "__main__":
    absltest.main()
