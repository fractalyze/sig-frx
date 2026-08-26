# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The seam verifies both SHRINCS paths, tells them apart, and makes them.

The component tests below this one gate the pieces — WOTS+C against the leaf it
recovers, FXMSS against the root it climbs to, the stateless half against the
SLH-DSA it wraps. What is left for this one is the assembly: the indicator byte
that chooses a path, the variable-width field that says where the FXMSS signature
starts, and the select that keeps one entry's verdict out of its neighbour's.

**A signature made here is checked against the reference's bytes, not against
this repo's own verifier.** A round trip proves that signing and verifying agree,
which a self-consistently wrong implementation does forever; what the vectors say
is that both halves are the specification's. So `SignerTest` compares whole
signatures byte for byte and treats the round trip as the second check rather
than the first.

The counter is the other thing this file has to hold: a leaf that signs twice
reveals its secret, so what is gated is that the value handed back is the one the
caller must store, and that a counter past the tree's last leaf raises instead of
quietly becoming a stateless signature five times the length.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from absl.testing import absltest

from sig_frx.hash.shrincs import fxmss, shrincs
from sig_frx.hash.shrincs.testing import fixtures
from sig_frx.hash.shrincs.testing import stateful_vectors as vectors
from sig_frx.hash.shrincs.testing import vectors as stateless_vectors


def _padded(signature: bytes) -> bytes:
    """A signature at the seam's width — the stateless length, zero-padded to."""
    return signature + bytes(shrincs.stateless.SIGNATURE_SIZE - len(signature))


def _row_of(public_key: bytes, message: bytes, signature: bytes) -> fixtures.Row:
    """A row at the seam's width.

    The padding is applied here rather than in `fixtures.verdicts`, which the
    stateless file shares: the width is *this* scheme's, and a stateful signature
    is shorter than it. One place, because a batch mixing the two paths will not
    stack at all if a caller forgets.
    """
    return (public_key, message, _padded(signature))


def _row(
    case: vectors.StatefulVectors | stateless_vectors.StatelessVectors,
) -> fixtures.Row:
    """The three fields a verification takes, from either path's vectors."""
    return _row_of(case.public_key, case.message, case.signature)


class SizeTest(absltest.TestCase):
    def test_the_seam_sizes_are_the_specifications(self) -> None:
        scheme = shrincs.Shrincs()
        self.assertEqual(scheme.public_key_size, 48)
        self.assertEqual(scheme.secret_key_size, 82)
        self.assertEqual(scheme.signature_max_size, 5777)

    def test_a_stateful_signature_stays_below_a_stateless_one(self) -> None:
        """Which is what makes the two distinguishable by length at all."""
        widest = shrincs.INDEX_FIELD_START + 8 + fxmss.SIGNATURE_SIZE_MAX
        self.assertEqual(widest, 4619)
        self.assertLess(widest, shrincs.stateless.SIGNATURE_SIZE)

    def test_every_reference_length_is_the_indicator_s(self) -> None:
        """The seam derives a length from the indicator; the vectors agree."""
        for case in vectors.REFERENCE:
            with self.subTest(case.label):
                derived = (
                    shrincs.INDEX_FIELD_START
                    + fxmss.index_field_bytes(case.leaf_depth)
                    + 514
                    + 16 * case.leaf_depth
                )
                self.assertEqual(derived, len(case.signature))


class StatefulTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.scheme = shrincs.Shrincs()

    def test_every_reference_signature_verifies(self) -> None:
        for case in vectors.REFERENCE:
            with self.subTest(case.label, depth=case.leaf_depth):
                self.assertEqual(
                    fixtures.verdicts(self.scheme.verify, [_row(case)], case.context),
                    [True],
                )

    def test_a_batch_of_different_depths_verifies(self) -> None:
        """Depths 1 to 64, so the index field is one, two and eight bytes wide.

        The gather that finds each FXMSS signature starts at a different offset
        per entry, which is the one place this path's shape depends on its data.
        """
        # One context and one message length, which is what a batch shares; the
        # depths are what it must not.
        cases = [
            c
            for c in vectors.REFERENCE
            if not c.context and len(c.message) == len(vectors.REFERENCE[0].message)
        ]
        self.assertGreater(len(cases), 1)
        self.assertEqual(
            fixtures.verdicts(self.scheme.verify, [_row(c) for c in cases], b""),
            [True] * len(cases),
        )
        self.assertEqual(
            sorted(fxmss.index_field_bytes(c.leaf_depth) for c in cases), [1, 1, 8]
        )
        self.assertEqual(sorted(c.leaf_depth for c in cases), [1, 4, 64])


class BothPathsTest(absltest.TestCase):
    """One key, one message, signed each way — and a batch holding both."""

    def setUp(self) -> None:
        super().setUp()
        self.scheme = shrincs.Shrincs()
        self.pair = vectors.BOTH_PATHS

    def _verify(self, *signatures: bytes) -> list[bool]:
        pair = self.pair
        return fixtures.verdicts(
            self.scheme.verify,
            [_row_of(pair.public_key, pair.message, s) for s in signatures],
            pair.context,
        )

    def test_each_path_verifies_alone(self) -> None:
        self.assertEqual(self._verify(self.pair.stateful_signature), [True])
        self.assertEqual(self._verify(self.pair.stateless_signature), [True])

    def test_a_mixed_batch_keeps_its_verdicts_apart(self) -> None:
        """The select this scheme is shaped around, with a rejection between them.

        The two paths recompute different halves of the same public key, so an
        entry taking the other one's verdict would be a signature verifying
        against a root it never touched.
        """
        broken = bytearray(self.pair.stateful_signature)
        broken[200] ^= 0x01
        self.assertEqual(
            self._verify(
                self.pair.stateful_signature,
                self.pair.stateless_signature,
                bytes(broken),
                self.pair.stateless_signature,
            ),
            [True, True, False, True],
        )

    def test_neither_signature_verifies_as_the_other_path(self) -> None:
        """Retagging a signature must not move it onto the path that accepts it."""
        stateful, stateless_sig = (
            self.pair.stateful_signature,
            self.pair.stateless_signature,
        )
        retagged_stateful = bytes([255]) + stateful[1:]
        retagged_stateless = bytes([self.pair.leaf_height]) + stateless_sig[1:]
        self.assertEqual(
            self._verify(retagged_stateful, retagged_stateless), [False, False]
        )


class RejectionTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.scheme = shrincs.Shrincs()
        self.case = vectors.REFERENCE[3]  # depth 16, a two-byte index field

    def _verdict(self, case: vectors.StatefulVectors) -> bool:
        return fixtures.verdicts(self.scheme.verify, [_row(case)], case.context)[0]

    def test_the_control_case_accepts(self) -> None:
        self.assertTrue(self._verdict(self.case))

    def test_a_flipped_bit_in_the_signature_is_rejected(self) -> None:
        for offset in (1, 17, 20, 300, len(self.case.signature) - 1):
            with self.subTest(offset=offset):
                broken = bytearray(self.case.signature)
                broken[offset] ^= 0x80
                self.assertFalse(
                    self._verdict(replace(self.case, signature=bytes(broken)))
                )

    def test_a_wrong_indicator_is_rejected(self) -> None:
        """It names the leaf's height, so changing it changes the walk's length."""
        for height in (self.case.leaf_height - 1, self.case.leaf_height + 1, 0):
            with self.subTest(height=height):
                broken = bytes([height]) + self.case.signature[1:]
                self.assertFalse(self._verdict(replace(self.case, signature=broken)))

    def test_a_leaf_index_outside_the_tree_is_rejected(self) -> None:
        """The field is whole bytes and a tree is not, so it can name too much.

        A depth-4 tree holds sixteen leaves and its index field holds 256 values,
        and nothing in the walk would notice the difference: the extra bits just
        pick sides at levels above the root. So this is a check the verifier makes
        rather than one the arithmetic makes for it — and it only bites where the
        depth is not a whole number of bytes, which is why the cases are chosen
        that way rather than taken from `self.case`.
        """
        # Also `< 64`: at or past that the field is exactly 64 bits, so every
        # value it can hold names a real position and there is nothing to reject.
        cases = [c for c in vectors.REFERENCE if c.leaf_depth % 8 and c.leaf_depth < 64]
        self.assertTrue(cases, "a depth that is neither a byte multiple nor 64+")
        for case in cases:
            size = fxmss.index_field_bytes(case.leaf_depth)
            for index in (1 << case.leaf_depth, (1 << (8 * size)) - 1):
                with self.subTest(case.label, index=index):
                    broken = (
                        case.signature[: shrincs.INDEX_FIELD_START]
                        + index.to_bytes(size, "big")
                        + case.signature[shrincs.INDEX_FIELD_START + size :]
                    )
                    self.assertFalse(self._verdict(replace(case, signature=broken)))

    def test_a_wrong_public_key_third_is_rejected(self) -> None:
        for offset, name in ((0, "pk_seed"), (16, "sl_root"), (32, "sf_root")):
            with self.subTest(name):
                broken = bytearray(self.case.public_key)
                broken[offset] ^= 0x01
                self.assertFalse(
                    self._verdict(replace(self.case, public_key=bytes(broken)))
                )

    def test_a_wrong_message_or_context_is_rejected(self) -> None:
        broken = bytearray(self.case.message)
        broken[0] ^= 0x01
        self.assertFalse(self._verdict(replace(self.case, message=bytes(broken))))
        self.assertFalse(self._verdict(replace(self.case, context=b"")))
        self.assertFalse(self._verdict(replace(self.case, context=b"ctX")))

    def test_a_signature_of_the_wrong_length_is_a_verdict(self) -> None:
        for signature in (self.case.signature, b""):
            with self.subTest(length=len(signature)):
                got = self.scheme.verify(
                    fixtures.rows(self.case.public_key),
                    fixtures.rows(self.case.message),
                    fixtures.rows(signature),
                    context=fixtures.context(self.case.context),
                )
                self.assertEqual(list(np.asarray(got)), [False])


class MessageDigestTest(absltest.TestCase):
    """`H_msg_sf` against the digest the reference computed.

    The one construction SHRINCS does not share with FIPS 205, and the only place
    the pinned `message_digest` is checked rather than fed in. Without it the
    digest is gated solely through a final verdict, which reports that something
    is wrong and not that it was this.
    """

    @staticmethod
    def _digest(case: vectors.StatefulVectors, height: int, index: int) -> bytes:
        got = shrincs.message_digest(
            fixtures.rows(case.randomizer),
            fixtures.rows(case.pk_seed),
            fixtures.rows(case.sl_root),
            fixtures.rows(case.sf_root),
            # The address's first nine bytes: the leaf's height and its index.
            fixtures.rows(bytes([height]) + index.to_bytes(8, "big")),
            fixtures.rows(case.message),
            context=fixtures.context(case.context),
        )
        return bytes(np.asarray(got)[0])

    def test_every_reference_digest_is_reproduced(self) -> None:
        for case in vectors.REFERENCE:
            with self.subTest(case.label):
                self.assertEqual(
                    self._digest(case, case.leaf_height, case.leaf_index),
                    case.message_digest,
                )

    def test_the_leaf_position_separates_two_digests(self) -> None:
        """The position goes into both hashes, so one leaf's digest is not another's."""
        case = vectors.REFERENCE[1]
        digests = {
            self._digest(case, height, index)
            for height, index in (
                (case.leaf_height, case.leaf_index),
                (case.leaf_height + 1, case.leaf_index),
                (case.leaf_height, case.leaf_index + 1),
            )
        }
        self.assertLen(digests, 3)
        self.assertIn(case.message_digest, digests)


class StatelessAtTheSeamTest(absltest.TestCase):
    """A stateless signature routes through the assembled scheme.

    One case rather than the whole stateless set: those are gated in
    `stateless_test`, and every entry of a seam call pays a stateful verification
    beside its stateless one. What is left to show is the routing, and
    `BothPathsTest` shows it under one key — this adds a second, so the select is
    not reading something that happened to be constant.
    """

    def test_a_stateless_reference_signature_verifies(self) -> None:
        case = stateless_vectors.REFERENCE[1]
        self.assertEqual(
            fixtures.verdicts(shrincs.Shrincs().verify, [_row(case)], case.context),
            [True],
        )


def _signer(case: vectors.StatefulVectors) -> shrincs.Shrincs:
    """A scheme built over the tree this case's key was generated with."""
    return shrincs.Shrincs(np.array([case.shape, case.depth], dtype=np.uint8))


def _seed(case: vectors.StatefulVectors) -> np.ndarray:
    return np.frombuffer(case.seed, dtype=np.uint8)


def _secret_key(seed: bytes, public_key: bytes, shape: int, depth: int) -> np.ndarray:
    """The 82 bytes assembled from what a case records, rather than regenerated.

    `test_every_reference_key_pair_is_regenerated` is what pins this layout
    against `keygen`; every other test here is about signing, and building the key
    rather than deriving it saves each of them a whole FXMSS tree.
    """
    return np.frombuffer(
        seed + public_key[16:32] + bytes([shape, depth]) + public_key[32:],
        dtype=np.uint8,
    )


class SignerTest(absltest.TestCase):
    def test_every_reference_key_pair_is_regenerated(self) -> None:
        """`pk_seed ‖ sl_root ‖ sf_root`, and the 82 bytes the reference serializes."""
        for case in vectors.REFERENCE:
            with self.subTest(case.label):
                public, secret = _signer(case).keygen(_seed(case))
                self.assertEqual(bytes(np.asarray(public)), case.public_key)
                self.assertEqual(
                    bytes(np.asarray(secret)),
                    case.seed
                    + case.sl_root
                    + bytes([case.shape, case.depth])
                    + case.sf_root,
                )

    def test_every_reference_signature_is_reproduced(self) -> None:
        """From the recorded seed and counter — the stateful path draws nothing."""
        for case in vectors.REFERENCE:
            with self.subTest(case.label):
                signature, counter = _signer(case).sign(
                    _secret_key(case.seed, case.public_key, case.shape, case.depth),
                    np.frombuffer(case.message, dtype=np.uint8),
                    case.state_counter,
                    context=fixtures.context(case.context),
                )
                made = bytes(np.asarray(signature))
                self.assertEqual(made, _padded(case.signature))
                self.assertEqual(counter, case.state_counter + 1)

    def test_a_signature_this_made_verifies(self) -> None:
        """The round trip, which is the second check and not the first.

        Both paths under one key, at the batch width the rest of the file uses so
        that it costs no further compile.
        """
        case = vectors.BOTH_PATHS
        scheme = shrincs.Shrincs(np.array([case.shape, case.depth], dtype=np.uint8))
        secret = _secret_key(case.seed, case.public_key, case.shape, case.depth)
        message = np.frombuffer(case.message, dtype=np.uint8)
        stateful, next_counter = scheme.sign(
            secret, message, case.state_counter, context=fixtures.context(case.context)
        )
        stateless_signature, no_counter = scheme.sign(
            secret,
            message,
            None,
            randomness=np.frombuffer(case.stateless_opt_rand, dtype=np.uint8),
            context=fixtures.context(case.context),
        )
        self.assertEqual(bytes(np.asarray(stateful)), _padded(case.stateful_signature))
        self.assertEqual(
            bytes(np.asarray(stateless_signature)), case.stateless_signature
        )
        self.assertEqual(next_counter, case.state_counter + 1)
        # No counter in, no counter back: the stateless path spends no leaf, so
        # there is nothing for the caller to store.
        self.assertIsNone(no_counter)
        self.assertEqual(
            int(np.asarray(stateless_signature)[0]),
            shrincs.stateless.STATELESS_INDICATOR,
        )
        self.assertEqual(
            fixtures.verdicts(
                scheme.verify,
                [
                    (case.public_key, case.message, bytes(np.asarray(stateful))),
                    (
                        case.public_key,
                        case.message,
                        bytes(np.asarray(stateless_signature)),
                    ),
                ],
                case.context,
            ),
            [True, True],
        )

    def test_a_spent_counter_raises_rather_than_falling_back(self) -> None:
        """The reference signs statelessly here; a caller that lost count is told.

        A silent fallback is a signature five times the length under a path the
        caller did not choose, which is the sort of thing noticed in production
        rather than in a test.
        """
        case = vectors.REFERENCE[0]
        with self.assertRaisesRegex(ValueError, "no leaf left"):
            _signer(case).sign(
                _secret_key(case.seed, case.public_key, case.shape, case.depth),
                np.frombuffer(case.message, dtype=np.uint8),
                2**case.depth,
                context=fixtures.context(case.context),
            )

    def test_a_salt_the_stateful_path_cannot_use_is_refused(self) -> None:
        """Ignoring it would leave a caller believing it salted something."""
        case = vectors.REFERENCE[0]
        with self.assertRaisesRegex(ValueError, "nowhere to put"):
            _signer(case).sign(
                _secret_key(case.seed, case.public_key, case.shape, case.depth),
                np.frombuffer(case.message, dtype=np.uint8),
                case.state_counter,
                randomness=np.zeros(16, dtype=np.uint8),
                context=fixtures.context(case.context),
            )

    def test_a_verifier_cannot_generate_a_key(self) -> None:
        """The shape is not something a verifier has, and not something to guess."""
        with self.assertRaisesRegex(ValueError, "no tree structure"):
            shrincs.Shrincs().keygen(np.zeros(48, dtype=np.uint8))


class ValueTest(absltest.TestCase):
    def test_equality_and_hash_are_value_based(self) -> None:
        self.assertEqual(shrincs.Shrincs(), shrincs.Shrincs())
        self.assertEqual(hash(shrincs.Shrincs()), hash(shrincs.Shrincs()))

    def test_two_trees_are_two_schemes(self) -> None:
        """The structure rides pytree aux, so an instance that forgot it re-traces."""
        balanced = np.array([fxmss.SHAPE_BALANCED, 4], dtype=np.uint8)
        self.assertEqual(shrincs.Shrincs(balanced), shrincs.Shrincs(balanced))
        self.assertEqual(
            hash(shrincs.Shrincs(balanced)), hash(shrincs.Shrincs(balanced))
        )
        self.assertNotEqual(shrincs.Shrincs(balanced), shrincs.Shrincs())
        self.assertNotEqual(
            shrincs.Shrincs(balanced),
            shrincs.Shrincs(np.array([fxmss.SHAPE_UNBALANCED, 4], dtype=np.uint8)),
        )


if __name__ == "__main__":
    absltest.main()
