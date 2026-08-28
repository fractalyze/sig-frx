# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""A key generated here, put to the reference implementation.

The published vectors gate one direction — upstream's key and signature checked
by this repo's `verify`, in `falcon_kat_test`. This is the other one, and it is
the half a transcription cannot supply: `falcon_reference.py` is a reading of
the specification by the same author as the code it checks, so agreement with
it cannot catch a misreading they share. The C here was written by the people
who designed the scheme.

The claim is one round trip. The reference signs a message with a key this repo
generated; this repo's `verify` accepts that signature under the matching public
key. Nothing but a key the reference loaded and used could produce a signature
that verifies, so the acceptance is carried by a check that the published
vectors already gate — and the rejections are asserted on both sides, which is
what stops the pass from being what an oracle stuck at "yes" would also produce.

**`Falcon-512` only, and that is a budget decision rather than a coverage one**
— the same one `falcon_test` records: a key costs about 45 s at `Falcon-512`
against 140 s at `Falcon-1024`, and the degree does not change what any case
here asserts. The oracle is built at both degrees, since a shared object that
is never loaded is one nobody knows links.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

from sig_frx.lattice.falcon import falcon
from sig_frx.lattice.falcon.testing import falcon_oracle, falcon_vectors

_DEGREE = 512
_MESSAGE = b"a key generated here, signed by the reference implementation"
# Any fixed value: the scheme is `keygen(seed)`, so this names the key pair the
# whole case runs on, and a failure reproduces from it.
_SEED = bytes(range(32))


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
        published = falcon_vectors.secret_key(f"Falcon-{_DEGREE}")
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


if __name__ == "__main__":
    absltest.main()
