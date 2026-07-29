# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The Signature seam holds its shape against an implementation.

Two obligations, and the second is the one a scheme gets wrong: the Protocol must
accept a conforming implementation *and* the batch axis must be honored per
entry. A `verify` that reduced over the batch — one verdict for the whole set —
passes every all-valid test ever written, so the tamper cases here corrupt
exactly one entry and pin the other verdicts alongside it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import Array
from frx.typing import ArrayLike

from sig_frx.signature import Signature

_KEY_SIZE = 8
_MESSAGE_LEN = 16
_BATCH = 4


class _ChecksumScheme:
    """NOT a signature scheme — the smallest thing that has the seam's shape.

    Signing masks a message checksum with the secret key; the public key is that
    key's complement, so verifying recovers the mask. Trivially forgeable by
    anyone who wants to forge it. It exists to exercise the Protocol: the shapes,
    the batch axis, and the value-equality rule.
    """

    def __init__(self, domain: int) -> None:
        self._domain = domain
        self.public_key_size = _KEY_SIZE
        self.secret_key_size = _KEY_SIZE
        self.signature_max_size = _KEY_SIZE
        self.deterministic = True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _ChecksumScheme):
            return NotImplemented
        return self._domain == other._domain

    def __hash__(self) -> int:
        return hash(self._domain)

    def keygen(self, seed: ArrayLike) -> tuple[Array, Array]:
        secret = fnp.asarray(seed, dtype=fnp.uint8)[:_KEY_SIZE]
        return 255 - secret, secret

    def sign(
        self,
        secret_key: Array,
        message: ArrayLike,
        *,
        randomness: ArrayLike | None = None,
    ) -> Array:
        return self._mask(secret_key, message)

    def verify(self, public_key: Array, message: ArrayLike, signature: Array) -> Array:
        return fnp.all(self._mask(255 - public_key, message) == signature, axis=-1)

    def _mask(self, secret_key: Array, message: ArrayLike) -> Array:
        checksum = fnp.sum(fnp.asarray(message, dtype=fnp.uint32), axis=-1)
        return ((checksum[..., None] + self._domain + secret_key) % 256).astype(
            fnp.uint8
        )


def _valid_batch() -> tuple[_ChecksumScheme, Array, Array, Array]:
    """A scheme plus a `(public_keys, messages, signatures)` batch it accepts."""
    scheme = _ChecksumScheme(domain=7)
    rng = np.random.default_rng(0)
    seeds = fnp.asarray(rng.integers(0, 256, (_BATCH, _KEY_SIZE)), dtype=fnp.uint8)
    messages = fnp.asarray(
        rng.integers(0, 256, (_BATCH, _MESSAGE_LEN)), dtype=fnp.uint8
    )
    # keygen and sign are single-message; vmap is how a caller batches them, which
    # is exactly what the seam's docstring says.
    public_keys, secret_keys = frx.vmap(scheme.keygen)(seeds)
    signatures = frx.vmap(lambda k, m: scheme.sign(k, m, randomness=None))(
        secret_keys, messages
    )
    return scheme, public_keys, messages, signatures


def _flip_bit(batch: Array, entry: int) -> Array:
    """Corrupt one entry of a batch, leaving every other entry intact."""
    return batch.at[entry, 0].set(batch[entry, 0] ^ 1)


class SignatureSeamTest(absltest.TestCase):
    def test_protocol_accepts_a_conforming_implementation(self) -> None:
        # `Signature` is runtime_checkable, so this is the structural check a
        # consumer's constructor gets for free when it takes the seam.
        self.assertIsInstance(_ChecksumScheme(domain=7), Signature)

    def test_verify_returns_one_verdict_per_batch_entry(self) -> None:
        scheme, public_keys, messages, signatures = _valid_batch()
        verdicts = scheme.verify(public_keys, messages, signatures)
        self.assertEqual(verdicts.shape, (_BATCH,))
        self.assertEqual(verdicts.dtype, fnp.bool_)
        self.assertTrue(bool(fnp.all(verdicts)))

    def test_verify_rejects_only_the_tampered_signature(self) -> None:
        scheme, public_keys, messages, signatures = _valid_batch()
        verdicts = scheme.verify(public_keys, messages, _flip_bit(signatures, 2))
        self.assertEqual([bool(v) for v in verdicts], [True, True, False, True])

    def test_verify_rejects_only_the_tampered_message(self) -> None:
        scheme, public_keys, messages, signatures = _valid_batch()
        verdicts = scheme.verify(public_keys, _flip_bit(messages, 1), signatures)
        self.assertEqual([bool(v) for v in verdicts], [True, False, True, True])

    def test_verify_rejects_only_the_tampered_public_key(self) -> None:
        scheme, public_keys, messages, signatures = _valid_batch()
        verdicts = scheme.verify(_flip_bit(public_keys, 3), messages, signatures)
        self.assertEqual([bool(v) for v in verdicts], [True, True, True, False])

    def test_the_whole_batch_survives_jit_as_one_call(self) -> None:
        # The batch is what maps onto a GPU's width, so it has to trace as one
        # computation — not B of them.
        scheme, public_keys, messages, signatures = _valid_batch()
        verdicts = frx.jit(scheme.verify)(public_keys, messages, signatures)
        self.assertTrue(bool(fnp.all(verdicts)))

    def test_value_equality_survives_reconstruction(self) -> None:
        # A freshly built instance with the same parameters must compare equal, or
        # riding as pytree aux silently re-traces the enclosing jit zone.
        self.assertEqual(_ChecksumScheme(domain=7), _ChecksumScheme(domain=7))
        self.assertEqual(hash(_ChecksumScheme(domain=7)), hash(_ChecksumScheme(7)))
        self.assertNotEqual(_ChecksumScheme(domain=7), _ChecksumScheme(domain=8))


if TYPE_CHECKING:
    # mypy-enforced seam conformance — the pin every implementation module ends
    # with, exercised here against the one implementation this repo has so far.
    _: type[Signature[Array, Array, Array]] = _ChecksumScheme


if __name__ == "__main__":
    absltest.main()
