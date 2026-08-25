# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The seam verifies both SHRINCS paths, and tells them apart per entry.

The component tests below this one gate the pieces — WOTS+C against the leaf it
recovers, FXMSS against the root it climbs to, the stateless half against the
SLH-DSA it wraps. What is left for this one is the assembly: the indicator byte
that chooses a path, the variable-width field that says where the FXMSS signature
starts, and the select that keeps one entry's verdict out of its neighbour's.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from absl.testing import absltest

from sig_frx.hash.shrincs import fxmss, shrincs
from sig_frx.hash.shrincs.testing import stateful_vectors as vectors
from sig_frx.hash.shrincs.testing import vectors as stateless_vectors


def _rows(*values: bytes) -> np.ndarray:
    return np.stack([np.frombuffer(v, dtype=np.uint8) for v in values])


def _ctx(context: bytes) -> np.ndarray | None:
    return np.frombuffer(context, dtype=np.uint8) if context else None


def _padded(signature: bytes) -> bytes:
    """A signature at the seam's width — the stateless length, zero-padded to."""
    return signature + bytes(shrincs.SIGNATURE_MAX_SIZE - len(signature))


# The one batch width this file verifies at. Four, because the mixed-path case
# below is four entries wide. Verification compiles per input shape and runs both
# paths for every entry, so a second batch width costs another whole compile —
# the same reason `stateless_test` settles on one.
_BATCH = 4


def _verdicts(
    scheme: shrincs.Shrincs,
    rows: list[tuple[bytes, bytes, bytes]],
    context: bytes,
) -> list[bool]:
    """Verify `(public key, message, signature)` rows in one call at `_BATCH`.

    Padded by repeating the last row, so the shape holds however many rows a test
    has something to say about. Rows in one call share a context, which is one per
    call, and a message length, which is one per batch.
    """
    if not 1 <= len(rows) <= _BATCH:
        raise ValueError(f"1 to {_BATCH} rows per call, got {len(rows)}")
    padded = list(rows) + [rows[-1]] * (_BATCH - len(rows))
    got = scheme.verify(
        _rows(*(r[0] for r in padded)),
        _rows(*(r[1] for r in padded)),
        _rows(*(_padded(r[2]) for r in padded)),
        context=_ctx(context),
    )
    return [bool(v) for v in np.asarray(got)][: len(rows)]


def _row(
    case: vectors.StatefulVectors | stateless_vectors.StatelessVectors,
) -> tuple[bytes, bytes, bytes]:
    """The three fields a verification takes, from either path's vectors."""
    return (case.public_key, case.message, case.signature)


class SizeTest(absltest.TestCase):
    def test_the_seam_sizes_are_the_specifications(self) -> None:
        self.assertEqual(shrincs.PUBLIC_KEY_SIZE, 48)
        self.assertEqual(shrincs.SECRET_KEY_SIZE, 82)
        self.assertEqual(shrincs.SIGNATURE_MAX_SIZE, 5777)

    def test_a_stateful_signature_stays_below_a_stateless_one(self) -> None:
        """Which is what makes the two distinguishable by length at all."""
        widest = 17 + 8 + fxmss.SIGNATURE_SIZE_MAX
        self.assertEqual(widest, 4619)
        self.assertLess(widest, shrincs.SIGNATURE_MAX_SIZE)

    def test_every_reference_length_is_the_indicator_s(self) -> None:
        """The seam derives a length from the indicator; the vectors agree."""
        for case in vectors.REFERENCE:
            with self.subTest(case.label):
                derived = (
                    17
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
                    _verdicts(self.scheme, [_row(case)], case.context), [True]
                )

    def test_a_batch_of_different_depths_verifies(self) -> None:
        """Depths 1 to 64, so the index field is one, two and eight bytes wide.

        The gather that finds each FXMSS signature starts at a different offset
        per entry, which is the one place this path's shape depends on its data.
        """
        cases = [c for c in vectors.REFERENCE if not c.context]
        self.assertGreater(len(cases), 1)
        lengths = {len(c.message) for c in cases}
        self.assertEqual(len(lengths), 1, "a batch shares one message length")
        self.assertEqual(
            _verdicts(self.scheme, [_row(c) for c in cases], b""),
            [True] * len(cases),
        )
        self.assertEqual(
            sorted(fxmss.index_field_bytes(c.leaf_depth) for c in cases), [1, 1, 8]
        )


class BothPathsTest(absltest.TestCase):
    """One key, one message, signed each way — and a batch holding both."""

    def setUp(self) -> None:
        super().setUp()
        self.scheme = shrincs.Shrincs()
        self.pair = vectors.BOTH_PATHS

    def _verify(self, *signatures: bytes) -> list[bool]:
        pair = self.pair
        return _verdicts(
            self.scheme,
            [(pair.public_key, pair.message, s) for s in signatures],
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
        return _verdicts(self.scheme, [_row(case)], case.context)[0]

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
        cases = [c for c in vectors.REFERENCE if c.leaf_depth % 8]
        self.assertTrue(cases, "a depth that is not a byte multiple")
        for case in cases:
            size = fxmss.index_field_bytes(case.leaf_depth)
            for index in (1 << case.leaf_depth, (1 << (8 * size)) - 1):
                with self.subTest(case.label, index=index):
                    broken = (
                        case.signature[:17]
                        + index.to_bytes(size, "big")
                        + case.signature[17 + size :]
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
                    _rows(self.case.public_key),
                    _rows(self.case.message),
                    _rows(signature),
                    context=_ctx(self.case.context),
                )
                self.assertEqual(list(np.asarray(got)), [False])


class StatelessAtTheSeamTest(absltest.TestCase):
    """The stateless vectors verify through the assembled scheme too."""

    def test_every_stateless_reference_signature_verifies(self) -> None:
        scheme = shrincs.Shrincs()
        for case in stateless_vectors.REFERENCE:
            with self.subTest(case.label):
                self.assertEqual(_verdicts(scheme, [_row(case)], case.context), [True])


class SignerTest(absltest.TestCase):
    def test_key_generation_says_what_it_needs(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "FXMSS tree"):
            shrincs.Shrincs().keygen(np.zeros(48, dtype=np.uint8))

    def test_there_is_no_seam_shaped_sign(self) -> None:
        """A stateful signer returns the advanced key too — see `signature.py`."""
        self.assertFalse(hasattr(shrincs.Shrincs(), "sign"))


class ValueTest(absltest.TestCase):
    def test_equality_and_hash_are_value_based(self) -> None:
        self.assertEqual(shrincs.Shrincs(), shrincs.Shrincs())
        self.assertEqual(hash(shrincs.Shrincs()), hash(shrincs.Shrincs()))


if __name__ == "__main__":
    absltest.main()
