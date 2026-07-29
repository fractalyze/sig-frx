# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""A stand-in with the seam's shape and none of its security. TEST ONLY.

Signing masks a message checksum with the secret key; the public key is that
key's complement, so verifying recovers the mask. Trivially forgeable by anyone
who wants to forge it, and never re-exported from the package.

It exists so the seam and the known-answer harness can be exercised before any
real scheme lands — the shapes, the batch axis, the context, and the
value-equality rule are all the same, and those are what both are testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import frx.numpy as fnp
from frx import Array
from frx.typing import ArrayLike

from sig_frx.signature import Signature

KEY_SIZE = 8


class ChecksumScheme:
    """Not a signature scheme. See the module docstring."""

    def __init__(self, domain: int) -> None:
        self._domain = domain
        self.public_key_size = KEY_SIZE
        self.secret_key_size = KEY_SIZE
        self.signature_max_size = KEY_SIZE
        self.deterministic = True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChecksumScheme):
            return NotImplemented
        return self._domain == other._domain

    def __hash__(self) -> int:
        return hash(self._domain)

    def keygen(self, seed: ArrayLike) -> tuple[Array, Array]:
        secret = fnp.asarray(seed, dtype=fnp.uint8)[:KEY_SIZE]
        return 255 - secret, secret

    def sign(
        self,
        secret_key: ArrayLike,
        message: ArrayLike,
        *,
        randomness: ArrayLike | None = None,
        context: ArrayLike | None = None,
    ) -> Array:
        return self._mask(fnp.asarray(secret_key), message, context)

    def verify(
        self,
        public_key: ArrayLike,
        message: ArrayLike,
        signature: ArrayLike,
        *,
        context: ArrayLike | None = None,
    ) -> Array:
        secret_key = 255 - fnp.asarray(public_key)
        return fnp.all(self._mask(secret_key, message, context) == signature, axis=-1)

    def _mask(
        self, secret_key: Array, message: ArrayLike, context: ArrayLike | None
    ) -> Array:
        # The context goes into what is signed, not beside it — the property the
        # seam requires and the reason a dropped context is a wrong answer rather
        # than a missing feature.
        checksum = fnp.sum(fnp.asarray(message, dtype=fnp.uint32), axis=-1)
        if context is not None:
            checksum = checksum + fnp.sum(fnp.asarray(context, dtype=fnp.uint32))
        return ((checksum[..., None] + self._domain + secret_key) % 256).astype(
            fnp.uint8
        )


if TYPE_CHECKING:
    # The seam conformance pin every implementation module ends with.
    _: type[Signature] = ChecksumScheme
