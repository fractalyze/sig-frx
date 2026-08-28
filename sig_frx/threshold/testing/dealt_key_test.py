# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""A dealt group key signs, and the ECDSA core verifies it unchanged.

A threshold ECDSA signature is an ordinary ECDSA signature, so the threshold
track adds signing and not verification. That is the premise the milestone
rests on, and it is worth a test rather than an argument — but the part that
could actually be wrong is narrower than "ECDSA verifies ECDSA", and it is
checkable now, before any threshold signing protocol exists:

    the key a trusted dealer shares out is the key the group signs under.

**The published key is what makes that checkable.** Both sides of the
comparison would otherwise be one function composition — `vss_commit` and
`Ecdsa.keygen` reach the same base-point ladder through `secp.multiple`, and
the same encoder writes both — so asserting one against the other would gate
the encodings round-tripping and nothing else, and a wrong ladder would
cancel. So the dealing runs on RFC 9591's own secp256k1 inputs and the point
in the middle is the **published** `group_public_key`: the dealer has to
reproduce that byte string, and the ECDSA core has to name that same point
for the reconstructed scalar. Neither side defines what the answer is.

The vector file's `inputs` are already this shape — a 2-of-3 with one
polynomial coefficient and shares for participants 1, 2 and 3 — so nothing
here is invented except the message.

What that buys is a failure that is otherwise silent. A dealer whose sharing
disagreed with its own commitment still produces a signature that verifies,
under the *wrong* key, so every sign-then-verify round trip passes and only
the published-key comparison notices.

The dealer takes a `PrimeOrderGroup` (`group.py`), so the group half of
`Secp256k1Sha256` is what carries this; RFC 9591's transcript is not involved
and the consumer on the other end is not a Schnorr scheme.
"""

from __future__ import annotations

import json

import numpy as np
from absl.testing import absltest
from python.runfiles import Runfiles

from sig_frx.classical import secp
from sig_frx.classical.ecdsa import core
from sig_frx.threshold import frost
from sig_frx.threshold import group as group_seam
from sig_frx.threshold.secp256k1_sha256 import Secp256k1Sha256

_RUNFILES = Runfiles.Create()
_VECTORS = "frost_secp256k1_sha256_vectors/file/frost-secp256k1-sha256.json"

# The vector file's `config` declares 2-of-3. The threshold is
# `len(coefficients) + 1`, so one share alone reconstructs `f(1)` rather than
# `f(0)` — the sub-threshold case below, and why it yields a wrong key rather
# than an error.
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


def _batch(
    rows: list[tuple[np.ndarray, bytes]], message: bytes
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(key, signature)` pairs over one message: `[B,65]`, `[B,L]`, `[B,64]`.

    Keys ride per row rather than broadcast, so one batch can mix them — which
    is what the sub-threshold case needs to reject one signature and accept
    another in the same call.
    """
    return (
        np.stack([key for key, _ in rows]),
        np.broadcast_to(
            np.frombuffer(message, dtype=np.uint8), (len(rows), len(message))
        ).copy(),
        np.stack([np.frombuffer(sig, dtype=np.uint8) for _, sig in rows]),
    )


class DealtKeyTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        path = _RUNFILES.Rlocation(_VECTORS)
        assert path is not None
        inputs = json.loads(open(path).read())["inputs"]

        self.group: group_seam.PrimeOrderGroup = Secp256k1Sha256()
        self.scheme = core.Ecdsa(secp.SECP256K1, core.SHA256)

        scalar = self.group.deserialize_scalar
        self.secret = scalar(bytes.fromhex(inputs["group_secret_key"]))
        self.coefficients = [
            scalar(bytes.fromhex(c)) for c in inputs["share_polynomial_coefficients"]
        ]
        # The dealer's own published output — the point neither side of the
        # comparison below is allowed to define.
        self.group_public_key = bytes.fromhex(inputs["group_public_key"])
        self.participants = [e["identifier"] for e in inputs["participant_shares"]]

        self.shares = dict(
            frost.secret_share_split(
                self.group, self.secret, self.coefficients, len(self.participants)
            )
        )
        self.commitment = frost.vss_commit(self.group, self.secret, self.coefficients)

    def _public(self, secret: int) -> np.ndarray:
        """The ECDSA public key for a scalar: `04 ‖ X ‖ Y`, 65 bytes."""
        public, _ = self.scheme.keygen(
            np.frombuffer(secret.to_bytes(32, "big"), dtype=np.uint8)
        )
        return np.asarray(public, dtype=np.uint8)

    def _sign(self, secret: int) -> bytes:
        """One RFC 6979 signature over `_MESSAGE` under `secret`."""
        _, secret_key = self.scheme.keygen(
            np.frombuffer(secret.to_bytes(32, "big"), dtype=np.uint8)
        )
        signature = self.scheme.sign(
            secret_key,
            np.frombuffer(_MESSAGE, dtype=np.uint8),
            randomness=None,
            context=None,
        )
        return np.asarray(signature, dtype=np.uint8).tobytes()

    def _verdicts(
        self, keys: np.ndarray, messages: np.ndarray, signatures: np.ndarray
    ) -> list[bool]:
        return np.asarray(
            self.scheme.verify(keys, messages, signatures, context=None)
        ).tolist()

    def _uncompressed(self, element: bytes) -> bytes:
        """A serialized group element in the ECDSA seam's own key encoding.

        In through `deserialize_elements`, so the commitment's encoding is
        validated rather than re-parsed here, and out through
        `secp.uncompressed_rows` — the same writer `Ecdsa.keygen` uses, so the
        comparison lands on that encoder instead of on a second transcription
        of SEC 1 §2.3.3.
        """
        point = self.group.deserialize_elements([element])
        row = secp.uncompressed_rows(secp.SECP256K1, point, np.array([True]))
        return row[0].tobytes()

    def test_the_dealer_reproduces_the_published_group_key(self) -> None:
        """`φ_0` is RFC 9591's own `group_public_key`, byte for byte."""
        self.assertEqual(self.commitment[0], self.group_public_key)

    def test_any_quorum_reconstructs_the_dealt_secret(self) -> None:
        for quorum in ((1, 2), (1, 3), (2, 3), (1, 2, 3)):
            with self.subTest(quorum=quorum):
                self.assertEqual(
                    _reconstruct(self.group, {i: self.shares[i] for i in quorum}),
                    self.secret,
                )

    def test_the_reconstructed_key_is_the_published_group_key(self) -> None:
        """The ECDSA core names the published point for the reconstructed scalar.

        With the two tests above, this is the milestone's claim: what a quorum
        reconstructs is the key the standard says the dealer shared out, and
        the verifier's own encoding of that key agrees.
        """
        secret = _reconstruct(self.group, {i: self.shares[i] for i in (1, 2)})
        self.assertEqual(
            self._public(secret).tobytes(),
            self._uncompressed(self.group_public_key),
        )

    def test_the_signature_verifies_through_the_existing_ecdsa_core(self) -> None:
        """Batch-first, `bool[B]` out — the seam's shape, no new verify path.

        The rejecting row sits between accepting ones, so a `verify` that
        reduced over the batch fails, and so does one that ignored its input.
        """
        public = self._public(self.secret)
        signature = self._sign(self.secret)
        corrupt = signature[:32] + bytes([signature[32] ^ 1]) + signature[33:]

        keys, messages, signatures = _batch(
            [(public, signature), (public, corrupt), (public, signature)], _MESSAGE
        )
        self.assertEqual(
            self._verdicts(keys, messages, signatures), [True, False, True]
        )

    def test_a_moved_bit_in_the_message_or_the_key_is_refused(self) -> None:
        public = self._public(self.secret)
        signature = self._sign(self.secret)

        keys, messages, signatures = _batch([(public, signature)] * 3, _MESSAGE)
        messages[0, 0] ^= 1
        keys[2, 33] ^= 1  # a coordinate bit — the point leaves the curve
        self.assertEqual(
            self._verdicts(keys, messages, signatures), [False, True, False]
        )

    def test_a_share_that_fails_its_commitment_is_refused(self) -> None:
        for identifier, share in self.shares.items():
            self.assertTrue(
                frost.vss_verify(self.group, identifier, share, self.commitment)
            )
        self.assertFalse(
            frost.vss_verify(self.group, 1, self.shares[1] + 1, self.commitment)
        )

    def test_a_sub_threshold_quorum_signs_under_a_different_key(self) -> None:
        """One share of a 2-of-3 reconstructs `f(1)`, and it is not the group key.

        Nothing raises: the short quorum yields a perfectly valid key pair
        whose signatures verify under *it*. The batch holds both readings at
        once — the short signature accepted under its own key and rejected
        under the published one — which is what makes the comparison against
        `group_public_key` load-bearing rather than decorative.
        """
        short = _reconstruct(self.group, {1: self.shares[1]})
        self.assertNotEqual(short, self.secret)

        short_public = self._public(short)
        short_signature = self._sign(short)
        group_public = self._public(self.secret)
        self.assertNotEqual(short_public.tobytes(), group_public.tobytes())

        keys, messages, signatures = _batch(
            [
                (group_public, self._sign(self.secret)),
                (group_public, short_signature),
                (short_public, short_signature),
            ],
            _MESSAGE,
        )
        self.assertEqual(
            self._verdicts(keys, messages, signatures), [True, False, True]
        )


if __name__ == "__main__":
    absltest.main()
