# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The Signature seam holds its shape against an implementation.

Two obligations, and the second is the one a scheme gets wrong: the Protocol must
accept a conforming implementation *and* the batch axis must be honored per
entry. A `verify` that reduced over the batch — one verdict for the whole set —
passes every all-valid test ever written, so the tamper cases here corrupt
exactly one entry and pin the other verdicts alongside it.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import Array

from sig_frx.signature import Signature
from sig_frx.testing.checksum_scheme import KEY_SIZE, ChecksumScheme

_MESSAGE_LEN = 16
_BATCH = 4


def _valid_batch() -> tuple[ChecksumScheme, Array, Array, Array]:
    """A scheme plus a `(public_keys, messages, signatures)` batch it accepts."""
    scheme = ChecksumScheme(domain=7)
    rng = np.random.default_rng(0)
    seeds = fnp.asarray(rng.integers(0, 256, (_BATCH, KEY_SIZE)), dtype=fnp.uint8)
    messages = fnp.asarray(
        rng.integers(0, 256, (_BATCH, _MESSAGE_LEN)), dtype=fnp.uint8
    )
    # keygen and sign are single-message; vmap is how a caller batches them, which
    # is exactly what the seam's docstring says.
    public_keys, secret_keys = frx.vmap(scheme.keygen)(seeds)
    signatures = frx.vmap(
        lambda k, m: scheme.sign(k, m, randomness=None, context=None)
    )(secret_keys, messages)
    return scheme, public_keys, messages, signatures


def _flip_bit(batch: Array, entry: int) -> Array:
    """Corrupt one entry of a batch, leaving every other entry intact."""
    return batch.at[entry, 0].set(batch[entry, 0] ^ 1)


class SignatureSeamTest(absltest.TestCase):
    def test_protocol_accepts_a_conforming_implementation(self) -> None:
        # `Signature` is runtime_checkable, so this is the structural check a
        # consumer's constructor gets for free when it takes the seam.
        self.assertIsInstance(ChecksumScheme(domain=7), Signature)

    def test_verify_returns_one_verdict_per_batch_entry(self) -> None:
        scheme, public_keys, messages, signatures = _valid_batch()
        verdicts = scheme.verify(public_keys, messages, signatures, context=None)
        self.assertEqual(verdicts.shape, (_BATCH,))
        self.assertEqual(verdicts.dtype, fnp.bool_)
        self.assertTrue(bool(fnp.all(verdicts)))

    def test_verify_rejects_only_the_tampered_signature(self) -> None:
        scheme, public_keys, messages, signatures = _valid_batch()
        verdicts = scheme.verify(
            public_keys, messages, _flip_bit(signatures, 2), context=None
        )
        self.assertEqual([bool(v) for v in verdicts], [True, True, False, True])

    def test_verify_rejects_only_the_tampered_message(self) -> None:
        scheme, public_keys, messages, signatures = _valid_batch()
        verdicts = scheme.verify(
            public_keys, _flip_bit(messages, 1), signatures, context=None
        )
        self.assertEqual([bool(v) for v in verdicts], [True, False, True, True])

    def test_verify_rejects_only_the_tampered_public_key(self) -> None:
        scheme, public_keys, messages, signatures = _valid_batch()
        verdicts = scheme.verify(
            _flip_bit(public_keys, 3), messages, signatures, context=None
        )
        self.assertEqual([bool(v) for v in verdicts], [True, True, True, False])

    def test_the_whole_batch_survives_jit_as_one_call(self) -> None:
        # The batch is what maps onto a GPU's width, so it has to trace as one
        # computation — not B of them.
        scheme, public_keys, messages, signatures = _valid_batch()
        verdicts = frx.jit(scheme.verify)(
            public_keys, messages, signatures, context=None
        )
        self.assertTrue(bool(fnp.all(verdicts)))

    def test_value_equality_survives_reconstruction(self) -> None:
        # A freshly built instance with the same parameters must compare equal, or
        # riding as pytree aux silently re-traces the enclosing jit zone.
        self.assertEqual(ChecksumScheme(domain=7), ChecksumScheme(domain=7))
        self.assertEqual(hash(ChecksumScheme(domain=7)), hash(ChecksumScheme(7)))
        self.assertNotEqual(ChecksumScheme(domain=7), ChecksumScheme(domain=8))


if __name__ == "__main__":
    absltest.main()
