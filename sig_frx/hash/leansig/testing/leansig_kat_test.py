# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""leanSig against the keys leanSpec publishes — the gate on the shipped scheme.

`leansig_test` gates the wiring, and it does so at `TEST`: 4 chains, 8 tree
levels, a target sum of 6. Every reading this package makes is reachable there,
which is what makes it the suite that localizes a failure. What it cannot reach
is the scheme as deployed — 46 chains, 32 levels, a target sum of 200 — nor a key
that anyone but this repo produced.

This suite is both. The cases are signed by upstream under two of the key pairs
its fixtures archive publishes, at `PROD_CONFIG`, and they carry upstream's
verdict for every one — the four refusals included, because a
verifier that returns `True` unconditionally reproduces every accepted signature
ever published ([`testing.md`](../../../../docs/reference/testing.md)).

leanSig is not driven by [`kat.py`](../../../testing/kat.py), and the reasons are
the two that page names. There is no published *format* to normalize: the archive
carries no `(key, slot, root, signature)` family, so these are transcribed
constants with their provenance in
[`prod_verify_vectors.py`](prod_verify_vectors.py) rather than a file a loader
parses. And the scheme is stateful, so it has no seam-shaped `sign` for the
harness to drive — its caller names the slot, and what advances is the prepared
window. What the exception costs is the rest of that section in full, which is
the refusals below and the batch the shared harness would otherwise have built.

The `TEST` round trip that `keygen` and `sign` owe is `signing_test`'s, and it
stays there: the published keys cover a partial activation window, which upstream
pads with fresh randomness, so no `PROD` key can be regenerated from its seed and
no `PROD` round trip exists to run.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from sig_frx.hash.leansig import leansig
from sig_frx.hash.leansig.testing.harness import bytes_of
from sig_frx.hash.leansig.testing.prod_verify_vectors import (
    PROD_VERIFY_VECTORS,
    ProdVerifyVector,
)

_SCHEME = leansig.named("prod")
_ACCEPTED = tuple(v for v in PROD_VERIFY_VECTORS if v.verdict)


def _batch(
    vectors: tuple[ProdVerifyVector, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The vectors as the four arrays the seam takes, keys included.

    Unlike the `TEST` set's, the key is per entry here: two published key pairs
    are in play, so a batch that replicated one would not be the batch the
    vectors describe.
    """
    return (
        np.stack([bytes_of(v.public_key) for v in vectors]),
        np.stack([bytes_of(v.message) for v in vectors]),
        np.stack([bytes_of(v.signature) for v in vectors]),
        np.asarray([v.slot for v in vectors]),
    )


def _verdicts(values: object) -> list[bool]:
    return [bool(value) for value in np.asarray(values)]


def _verify(vectors: tuple[ProdVerifyVector, ...]) -> list[bool]:
    keys, messages, signatures, slots = _batch(vectors)
    return _verdicts(_SCHEME.verify(keys, messages, signatures, position=slots))


class PublishedKeyTest(parameterized.TestCase):
    """Upstream's verdict on every case, one at a time and then all at once."""

    @parameterized.named_parameters(
        *[(vector.name, vector) for vector in PROD_VERIFY_VECTORS]
    )
    def test_one_vector(self, vector: ProdVerifyVector) -> None:
        self.assertEqual(_verify((vector,)), [vector.verdict])

    def test_the_whole_set_in_one_call(self) -> None:
        """One call, and entry `i` answers entry `i`.

        The set carries both verdicts and both published keys, so this fails for
        a verifier that decided once for the batch — an `all` where a per-entry
        select belongs — and for one that read a single key across it, each of
        which passes every single-entry case above.
        """
        self.assertEqual(
            _verify(PROD_VERIFY_VECTORS),
            [vector.verdict for vector in PROD_VERIFY_VECTORS],
        )

    def test_both_legs_agree(self) -> None:
        """The traced leg returns the eager verdicts, refusals included.

        `verify` is an eager entrance — both encoders decompose a wide integer
        base-p, which no lane holds — so what is gated is that the hashing runs
        as one computation rather than `B` dispatches, and that it answers the
        same at the parameters the scheme ships at.
        """
        keys, messages, signatures, slots = _batch(PROD_VERIFY_VECTORS)
        traced = _verdicts(
            _SCHEME.verify(
                fnp.asarray(keys),
                messages,
                fnp.asarray(signatures),
                position=slots,
            )
        )
        self.assertEqual(traced, [vector.verdict for vector in PROD_VERIFY_VECTORS])


class ShippedParameterTest(absltest.TestCase):
    """The numbers that separate `PROD` from `TEST`, read off the published bytes.

    A wrong constant here is not caught by a size assertion alone — it is caught
    by the vectors — but a size that disagrees with what leanSpec published says
    the preset was mis-transcribed before any of them runs, which is a much
    cheaper failure to read.
    """

    def test_the_sizes_are_the_published_bytes(self) -> None:
        vector = _ACCEPTED[0]
        self.assertEqual(_SCHEME.public_key_size, len(vector.public_key) // 2)
        self.assertEqual(_SCHEME.signature_max_size, len(vector.signature) // 2)

    def test_the_lifetime_is_the_deployed_one(self) -> None:
        # 2^32 slots, which is what makes the top/bottom split necessary and
        # what every case's 32-level opening climbs.
        self.assertEqual(_SCHEME.signatures_per_key, 1 << 32)

    def test_the_published_keys_are_distinct(self) -> None:
        # Both roles of one validator. If these ever collapsed to one value the
        # `wrong_key` case would be vacuous rather than failing.
        keys = {vector.public_key for vector in PROD_VERIFY_VECTORS}
        self.assertLen(keys, 2)


class TamperedSignatureTest(absltest.TestCase):
    """Bytes that decode but do not attest, at the shipped shape.

    The codec's own rejections are `ssz_test`'s. What these pin is that they
    arrive at the caller as a verdict rather than as an exception, on a signature
    whose 2536 bytes and 32-level path are the deployed ones.
    """

    def test_one_flipped_byte_is_refused(self) -> None:
        keys, messages, signatures, slots = _batch(_ACCEPTED[:1])
        for offset, what in ((-1, "a released chain hash"), (40, "a path sibling")):
            with self.subTest(what):
                tampered = signatures.copy()
                tampered[0, offset] ^= 0x01
                self.assertEqual(
                    _verdicts(_SCHEME.verify(keys, messages, tampered, position=slots)),
                    [False],
                )

    def test_an_off_length_path_is_a_verdict(self) -> None:
        """An opening declaring 31 siblings where the scheme climbs 32.

        Upstream's `verify` refuses a wrong `hashes` or `siblings` count in its
        phase 3, before the chain walk. Here the buffer is a fixed 2536 bytes, so
        the count is not a matter of sending fewer bytes — it is the offset at
        byte 32 that says where the siblings end and the chain ends begin. Moving
        it down by one digest's 32 bytes declares a 31-level path, which is a
        tree that is not this key's.

        `ssz_test` gates the codec's flag on the same mutation. What this adds is
        that the flag reaches the caller folded into a verdict rather than as an
        exception, which is what the seam's static batch shape requires.
        """
        keys, messages, signatures, slots = _batch(_ACCEPTED[:1])
        shortened = signatures.copy()
        # 1064 -> 1032, low byte only: the offset is little-endian and the change
        # is smaller than a byte's worth of place value.
        self.assertEqual(shortened[0, 32], 0x28)
        shortened[0, 32] = 0x08
        self.assertEqual(
            _verdicts(_SCHEME.verify(keys, messages, shortened, position=slots)),
            [False],
        )


if __name__ == "__main__":
    absltest.main()
