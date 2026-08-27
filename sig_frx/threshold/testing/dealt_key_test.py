# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""A dealt group key signs, and the ECDSA core verifies it unchanged.

A threshold ECDSA signature is an ordinary ECDSA signature, so the threshold
track adds signing and not verification. That is the premise the milestone
rests on, and it is worth a test rather than an argument — but the part that
could actually be wrong is narrower than "ECDSA verifies ECDSA", and it is
checkable now, before any threshold signing protocol exists:

    the key a trusted dealer shares out is the key the group signs under.

`vss_commit`'s `φ_0` is what the dealer publishes as the group public key, and
the verifier has to accept exactly that point. If the two agree here, then any
protocol that reconstructs the same group secret emits signatures the existing
chain-agnostic core verifies, and "no new verification path" is a property of
the dealer agreeing with the verifier rather than of the protocol.

The failure this guards against is silent by construction. A dealer whose
sharing disagreed with its own commitment still produces a signature that
verifies — under the *wrong* key — so every sign-then-verify round trip passes
and only the comparison against `φ_0` notices. `testing.md` says
self-consistency is not evidence; this is what that looks like concretely.

The dealer is driven through `PrimeOrderGroup` (`group.py`), so what runs here
is the group half of `Secp256k1Sha256` and none of RFC 9591's transcript — the
reuse the seam exists for, with a non-Schnorr consumer on the other end. The
secret and the polynomial coefficients are fixed constants rather than draws,
for the reason `signature.py` gives: an implicit draw is how a test stops being
reproducible.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

from sig_frx.classical import secp
from sig_frx.classical.ecdsa import core
from sig_frx.threshold import frost
from sig_frx.threshold import group as group_seam
from sig_frx.threshold.secp256k1_sha256 import Secp256k1Sha256

# 2-of-3: the threshold is `len(_COEFFICIENTS) + 1`, since the dealer's
# polynomial is `f(x) = secret + a1·x`. One share alone therefore reconstructs
# `f(1)` rather than `f(0)` — the sub-threshold case below, and the reason it
# produces a wrong key rather than an error.
_PARTICIPANTS = 3

_SECRET = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
_COEFFICIENTS = [0x1FA1E8F0A0B1E9A7F3B4C5D6E7F8091A2B3C4D5E6F708192A3B4C5D6E7F80910]

_MESSAGE = b"a dealt group key signs an ordinary ECDSA message"


def _reconstruct(group: group_seam.PrimeOrderGroup, shares: dict[int, int]) -> int:
    """Lagrange interpolation at zero over a quorum: `Σ λ_i · share_i`."""
    participants = sorted(shares)
    field = group.scalar_field
    total = field(0)
    for identifier in participants:
        lambda_i = frost.derive_interpolating_value(group, participants, identifier)
        total = total + field(lambda_i) * shares[identifier]
    return int(total)


def _compressed(public_key: bytes) -> bytes:
    """An ECDSA `04 ‖ X ‖ Y` key as the SEC 1 compressed form the dealer publishes.

    The two encodings name the same point; the dealer's `φ_0` is compressed
    (RFC 9591 §6.5) and the ECDSA seam's key is uncompressed (SEC 1 §2.3.3),
    so one of them has to be re-encoded before the comparison means anything.
    """
    if len(public_key) != 65 or public_key[0] != 4:
        raise ValueError("an ECDSA public key is 65 bytes, header 04")
    x = int.from_bytes(public_key[1:33], "big")
    y = int.from_bytes(public_key[33:], "big")
    return secp.compressed_bytes(secp.SECP256K1, x, y)


def _batch(
    public_key: np.ndarray, message: bytes, *signatures: bytes
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One key and message across a batch of signatures: `[B,65]`, `[B,L]`, `[B,64]`."""
    batch = len(signatures)
    return (
        np.broadcast_to(public_key, (batch, 65)).copy(),
        np.broadcast_to(
            np.frombuffer(message, dtype=np.uint8), (batch, len(message))
        ).copy(),
        np.stack([np.frombuffer(s, dtype=np.uint8) for s in signatures]),
    )


class DealtKeyTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.group: group_seam.PrimeOrderGroup = Secp256k1Sha256()
        self.scheme = core.Ecdsa(secp.SECP256K1, core.SHA256)
        self.shares = dict(
            frost.secret_share_split(self.group, _SECRET, _COEFFICIENTS, _PARTICIPANTS)
        )
        self.commitment = frost.vss_commit(self.group, _SECRET, _COEFFICIENTS)

    def _sign(self, secret: int) -> tuple[np.ndarray, bytes]:
        """The ECDSA key pair for a scalar, and its signature over `_MESSAGE`."""
        public, secret_key = self.scheme.keygen(
            np.frombuffer(secret.to_bytes(32, "big"), dtype=np.uint8)
        )
        signature = self.scheme.sign(
            secret_key,
            np.frombuffer(_MESSAGE, dtype=np.uint8),
            randomness=None,
            context=None,
        )
        return public, np.asarray(signature, dtype=np.uint8).tobytes()

    def test_every_dealt_share_opens_against_the_commitment(self) -> None:
        for identifier, share in self.shares.items():
            self.assertTrue(
                frost.vss_verify(self.group, identifier, share, self.commitment)
            )
        self.assertFalse(
            frost.vss_verify(self.group, 1, self.shares[1] + 1, self.commitment)
        )

    def test_any_quorum_reconstructs_the_dealt_secret(self) -> None:
        for quorum in ((1, 2), (1, 3), (2, 3), (1, 2, 3)):
            with self.subTest(quorum=quorum):
                self.assertEqual(
                    _reconstruct(self.group, {i: self.shares[i] for i in quorum}),
                    _SECRET,
                )

    def test_the_reconstructed_key_is_the_key_the_dealer_published(self) -> None:
        """`φ_0` and the ECDSA public key are the same point, byte for byte."""
        secret = _reconstruct(self.group, {i: self.shares[i] for i in (1, 2)})
        public, _ = self._sign(secret)
        self.assertEqual(_compressed(public.tobytes()), self.commitment[0])

    def test_the_signature_verifies_through_the_existing_ecdsa_core(self) -> None:
        """Batch-first, `bool[B]` out — the seam's shape, no new verify path."""
        secret = _reconstruct(self.group, {i: self.shares[i] for i in (2, 3)})
        public, signature = self._sign(secret)

        corrupt_signature = signature[:32] + bytes([signature[32] ^ 1]) + signature[33:]
        keys, messages, signatures = _batch(
            public, _MESSAGE, signature, corrupt_signature, signature
        )
        got = np.asarray(self.scheme.verify(keys, messages, signatures, context=None))
        self.assertEqual(list(got), [True, False, True])

    def test_a_moved_bit_in_the_message_or_the_key_is_refused(self) -> None:
        secret = _reconstruct(self.group, {i: self.shares[i] for i in (1, 3)})
        public, signature = self._sign(secret)

        keys, messages, signatures = _batch(public, _MESSAGE, signature, signature)
        messages[1, 0] ^= 1
        self.assertEqual(
            list(
                np.asarray(self.scheme.verify(keys, messages, signatures, context=None))
            ),
            [True, False],
        )

        keys, messages, signatures = _batch(public, _MESSAGE, signature, signature)
        keys[1, 33] ^= 1  # a coordinate bit — the point leaves the curve
        self.assertEqual(
            list(
                np.asarray(self.scheme.verify(keys, messages, signatures, context=None))
            ),
            [True, False],
        )

    def test_a_sub_threshold_quorum_signs_under_a_different_key(self) -> None:
        """One share of a 2-of-3 reconstructs `f(1)`, and it is not the group key.

        The interesting half is that nothing raises: the short quorum produces
        a perfectly valid ECDSA key pair whose signatures verify under *it*.
        What separates it from the group key is the comparison against `φ_0`,
        which is why that comparison is the test above and not a formality.
        """
        short = _reconstruct(self.group, {1: self.shares[1]})
        self.assertNotEqual(short, _SECRET)

        public, signature = self._sign(short)
        self.assertNotEqual(_compressed(public.tobytes()), self.commitment[0])

        # It verifies under its own key ...
        keys, messages, signatures = _batch(public, _MESSAGE, signature)
        self.assertEqual(
            list(
                np.asarray(self.scheme.verify(keys, messages, signatures, context=None))
            ),
            [True],
        )

        # ... and not under the group key the dealer published.
        group_public, _ = self._sign(_SECRET)
        keys, messages, signatures = _batch(group_public, _MESSAGE, signature)
        self.assertEqual(
            list(
                np.asarray(self.scheme.verify(keys, messages, signatures, context=None))
            ),
            [False],
        )


if __name__ == "__main__":
    absltest.main()
