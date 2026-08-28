# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""This repo and the reference implementation, each judging the other's output.

The published vectors gate one direction — upstream's key and signature checked
by this repo's `verify`, in `falcon_kat_test`. Both directions here are the
other one, and they are the half a transcription cannot supply:
`falcon_reference.py` is a reading of the specification by the same author as
the code it checks, so agreement with it cannot catch a misreading they share.
The C here was written by the people who designed the scheme.

**A key generated here, signed by the reference.** The reference signs a message
with a key this repo generated; this repo's `verify` accepts that signature
under the matching public key. Nothing but a key the reference loaded and used
could produce a signature that verifies, so the acceptance is carried by a check
the published vectors already gate. `Falcon-512` only, which is a budget
decision rather than a coverage one — the same one `falcon_test` records: a key
costs about 45 s at `Falcon-512` against 140 s at `Falcon-1024`, and the degree
does not change what any of those cases assert.

**A signature produced here, judged by the reference.** The direction with no
published value to compare against at all: §3.9 draws a salt per signature, so
two correct implementations disagree by construction and there is nothing to
reproduce. It runs at both degrees, because it needs no key generation — the
published secret key is the input, which also keeps the signer the only thing
under test.

The rejections are asserted on both sides throughout. Without them a pass is
what an oracle stuck at "yes" would also produce.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

from sig_frx.lattice.falcon import encoding, falcon
from sig_frx.lattice.falcon.testing import falcon_oracle
from sig_frx.lattice.falcon.testing.falcon_vectors import SECRET_KEYS, VECTORS

_DEGREE = 512
_MESSAGE = b"a key generated here, signed by the reference implementation"
# Any fixed value: the scheme is `keygen(seed)`, so this names the key pair the
# whole case runs on, and a failure reproduces from it.
_SEED = bytes(range(32))

# The public key of the same published record each `SECRET_KEYS` entry is the
# secret half of, so an acceptance is evidence about that pair.
PUBLIC_KEYS = {name: cases[0].public_key for name, cases in VECTORS.items()}


class ReferenceAcceptsAGeneratedKeyTest(absltest.TestCase):
    """The last acceptance criterion of sig-frx#26."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.scheme = falcon.named(f"Falcon-{_DEGREE}")
        public_key, secret_key = cls.scheme.keygen(np.frombuffer(_SEED, dtype=np.uint8))
        cls.public_key = bytes(np.asarray(public_key, dtype=np.uint8))
        cls.secret_key = bytes(np.asarray(secret_key, dtype=np.uint8))

    def _verify(self, signature: bytes) -> bool:
        """This repo's verdict on one signature, as a Python bool."""
        verdict = self.scheme.verify(
            np.frombuffer(self.public_key, dtype=np.uint8)[None],
            np.frombuffer(_MESSAGE, dtype=np.uint8)[None],
            np.frombuffer(signature, dtype=np.uint8)[None],
        )
        return bool(np.asarray(verdict)[0])

    def test_the_reference_signs_with_it_and_verify_accepts(self) -> None:
        signature = falcon_oracle.sign(self.secret_key, _MESSAGE, _DEGREE)
        self.assertIsNotNone(signature, "the reference refused a key generated here")
        assert signature is not None  # narrowing, for the type checker
        self.assertLen(signature, self.scheme.params.signature_size)
        self.assertTrue(self._verify(signature))

    def test_a_signature_under_another_key_is_refused(self) -> None:
        """The control, without which the accepted case gates nothing.

        `verify` returning true for a real signature is also what a verifier
        stuck at true would return. So the reference signs the same message
        with the key it published — a valid Falcon-512 key, correctly loaded,
        producing a signature its own verifier accepts — and this repo refuses
        it, because the public key it is checked against is not that key's. The
        difference between this case and the accepted one is the trapdoor and
        nothing else.
        """
        published = bytes.fromhex(SECRET_KEYS[f"Falcon-{_DEGREE}"])
        signature = falcon_oracle.sign(published, _MESSAGE, _DEGREE)
        self.assertIsNotNone(signature, "the published key must load")
        assert signature is not None
        self.assertFalse(self._verify(signature))

    def test_a_corrupted_signature_is_refused_here(self) -> None:
        """The negative half on this side of the round trip.

        Byte 41 is the first compressed coefficient — past the header and the
        salt, so the salt still matches and `hash_to_point` still produces the
        same target. That makes this a wrong signature for the right challenge
        rather than a signature over a different message.
        """
        signature = falcon_oracle.sign(self.secret_key, _MESSAGE, _DEGREE)
        assert signature is not None
        self.assertTrue(self._verify(signature), "the accepted case must accept first")
        corrupted = bytearray(signature)
        corrupted[41] ^= 0x01
        self.assertFalse(self._verify(bytes(corrupted)))

    def test_a_corrupted_key_is_refused_there(self) -> None:
        """The negative half on the reference's side.

        Both corruptions land in `f`'s coefficients rather than in the header
        byte, which is the point: rejecting a wrong header would only show that
        a constant was compared. What refuses these is `complete_private`
        recovering a `G` that does not fit `[-127, +127]` — see
        [`falcon_oracle`](falcon_oracle.py) for the measurement behind that.
        """
        self.assertIsNotNone(
            falcon_oracle.sign(self.secret_key, _MESSAGE, _DEGREE),
            "the accepted case must accept first",
        )
        for offset in (1, len(self.secret_key) // 2):
            with self.subTest(offset=offset):
                corrupted = bytearray(self.secret_key)
                corrupted[offset] ^= 0x01
                self.assertIsNone(
                    falcon_oracle.sign(bytes(corrupted), _MESSAGE, _DEGREE)
                )

    def test_the_oracle_links_at_both_degrees(self) -> None:
        """A shared object that is never loaded is one nobody knows links.

        `Falcon-1024` has no key here — generating one is the 140 s this suite
        declines to spend — so what this reaches is the loader, with a key of
        the right length that cannot decode. `None` rather than an exception is
        the whole assertion: the library resolved, bound, and returned a
        verdict.
        """
        params = falcon.PARAMETER_SETS["Falcon-1024"]
        self.assertIsNone(
            falcon_oracle.sign(bytes(params.secret_key_size), _MESSAGE, 1024)
        )


class ReferenceAcceptsASignatureProducedHereTest(absltest.TestCase):
    """sig-frx#27's acceptance criterion, and the direction with no published value.

    Everything above is the reference producing something this repo checks. This
    is the reverse: a signature **produced here** put to the reference
    implementation. Signing is randomized, so §3.9 publishes no signature to
    reproduce — a second implementation's verdict is what stands in for one, and
    it is a stronger claim than this repo's own `verify` could make about its own
    output.

    Run under the *published* secret key rather than a generated one, so a
    failure separates cleanly: the key is upstream's and correctly loaded — the
    class above proves the reference itself signs with it — leaving the signer as
    the only thing under test.
    """

    def _signature(self, degree: int, seed: int) -> bytes:
        scheme = falcon.named(f"Falcon-{degree}")
        secret = np.frombuffer(
            bytes.fromhex(SECRET_KEYS[f"Falcon-{degree}"]), dtype=np.uint8
        )
        return bytes(
            np.asarray(
                scheme.sign(
                    secret,
                    np.frombuffer(_MESSAGE, dtype=np.uint8),
                    randomness=np.full(scheme.seed_size, seed, dtype=np.uint8),
                ),
                dtype=np.uint8,
            )
        )

    def test_the_reference_accepts_it(self) -> None:
        """At both degrees, and over several salts: the loop is what varies.

        Three seeds rather than one because signing draws a salt per signature
        and rejects on a norm bound — so consecutive seeds take different trip
        counts through Algorithm 10, and a signer that only worked when the
        first draw was accepted would pass a single case.
        """
        for degree in (512, 1024):
            published = bytes.fromhex(PUBLIC_KEYS[f"Falcon-{degree}"])
            for seed in (0, 1, 2):
                with self.subTest(degree=degree, seed=seed):
                    signature = self._signature(degree, seed)
                    self.assertLen(
                        signature,
                        falcon.PARAMETER_SETS[f"Falcon-{degree}"].signature_size,
                    )
                    self.assertTrue(
                        falcon_oracle.accepts(published, _MESSAGE, signature, degree),
                        "the reference rejects a signature produced here",
                    )

    def test_the_reference_refuses_it_once_a_byte_moves(self) -> None:
        """The negative half, without which the case above gates nothing.

        An oracle stuck at accept would pass it while the signer emitted
        anything at all. Byte 41 is the first compressed coefficient, so the
        salt still matches and the challenge is unchanged — a wrong signature
        for the right target rather than a malformed one.
        """
        for degree in (512, 1024):
            with self.subTest(degree=degree):
                published = bytes.fromhex(PUBLIC_KEYS[f"Falcon-{degree}"])
                moved = bytearray(self._signature(degree, 0))
                moved[1 + encoding.SALT_SIZE] ^= 0x01
                self.assertFalse(
                    falcon_oracle.accepts(published, _MESSAGE, bytes(moved), degree)
                )


if __name__ == "__main__":
    absltest.main()
