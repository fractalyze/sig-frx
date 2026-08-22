# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""FROST(Ed25519, SHA-512) — RFC 9591 §6.1, over the Edwards substrate.

The ciphersuite is constants and hash instantiations over the group the
Ed25519 scheme already uses: same curve module, same encode/decode, same
curated point dtypes, at `B = 1` on the host. Its aggregate output is a
plain RFC 8032 signature — `H2` deliberately omits the domain separator for
exactly that compatibility — so this suite's `verify` delegates to the
existing batched Ed25519 verifier, and the tests gate on that crossing.

SHA-512 comes from `hashlib` for the same recorded reason as the Ed25519
scheme's concrete paths: hash-frx ships none, and everything here is host
work (fractalyze/hash-frx#66 tracks the device row, which this module does
not need).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from frx.typing import ArrayLike

from sig_frx.classical import edwards
from sig_frx.classical.eddsa import ed25519
from sig_frx.threshold import frost

_CONTEXT = b"FROST-ED25519-SHA512-v1"

# The delegate `verify` forwards to, built once — the scheme is a frozen
# constant, so per-call construction would buy nothing.
_VERIFIER = ed25519.Ed25519()


def _reduced(digest: bytes, order: int) -> int:
    """A 64-byte digest as a little-endian integer modulo the group order."""
    return int.from_bytes(digest, "little") % order


@dataclass(frozen=True)
class Ed25519Sha512:
    """RFC 9591 §6.1's ciphersuite, elements riding as `[1]`-batch points."""

    order = edwards.ED25519.order
    element_size = 32

    curve = edwards.ED25519
    scalar_field = edwards.ED25519.scalar

    def h1(self, message: bytes) -> int:
        return _reduced(
            hashlib.sha512(_CONTEXT + b"rho" + message).digest(), self.order
        )

    def h2(self, message: bytes) -> int:
        # No context prefix: RFC 8032's challenge, for signature compatibility.
        return _reduced(hashlib.sha512(message).digest(), self.order)

    def h3(self, message: bytes) -> int:
        return _reduced(
            hashlib.sha512(_CONTEXT + b"nonce" + message).digest(), self.order
        )

    def h4(self, message: bytes) -> bytes:
        return hashlib.sha512(_CONTEXT + b"msg" + message).digest()

    def h5(self, message: bytes) -> bytes:
        return hashlib.sha512(_CONTEXT + b"com" + message).digest()

    def serialize_scalar(self, scalar: int) -> bytes:
        return (scalar % self.order).to_bytes(32, "little")

    def deserialize_scalar(self, data: bytes) -> int:
        if len(data) != 32:
            raise ValueError("a serialized scalar is 32 bytes")
        value = int.from_bytes(data, "little")
        if value >= self.order:
            raise ValueError("a scalar is in [0, L-1]")
        return value

    def scalar_base_mult(self, scalar: int) -> bytes:
        return self.serialize_element(
            self.element_scalar_mult(self.curve.generator, scalar)
        )

    def deserialize_element(self, data: bytes) -> np.ndarray:
        # RFC 9591 §6.1 deserializes group elements per RFC 8032 §5.1.3,
        # canonicity refusals included — the ciphersuite names the strict
        # reading, not one of the consensus relaxations.
        point, ok = edwards.decode(
            self.curve,
            np.frombuffer(data, dtype=np.uint8)[None, :],
            canonical_only=True,
        )
        if not bool(np.asarray(ok)[0]):
            raise ValueError("not a canonical encoding of a curve point")
        if bool(np.asarray(edwards.is_identity(self.curve, point))[0]):
            raise ValueError("the identity element has no place on the wire")
        return point.astype(self.curve.accumulator)

    def element_add(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return left + right

    def element_scalar_mult(self, element: np.ndarray, scalar: int) -> np.ndarray:
        # Every scalar crossing this seam is already in [0, L-1] (the seam's
        # scalar contract), so multiple's % L is a no-op, not a reduction.
        return edwards.multiple(self.curve, [scalar], element)

    def identity_element(self) -> np.ndarray:
        return self.curve.identity.astype(self.curve.accumulator)

    def serialize_element(self, element: np.ndarray) -> bytes:
        if bool(np.asarray(edwards.is_identity(self.curve, element))[0]):
            raise ValueError("the identity element has no encoding here")
        ((x, y),) = edwards.affine_ints(self.curve, element)
        return edwards.encode_affine(x, y)

    def verify(
        self, public_key: ArrayLike, message: ArrayLike, signature: ArrayLike
    ) -> Any:
        """The aggregate's verdicts, `bool[B]`: plain RFC 8032 verification.

        A delegation, not a second Edwards path — the aggregate is a plain
        RFC 8032 signature (`H2` above), so the scheme's own batched
        verifier is this suite's: strict decoding, cofactorless, a
        malformed row answering `False` rather than raising. RFC 8032's
        Ed25519 takes no context, so none is exposed here.
        """
        return _VERIFIER.verify(public_key, message, signature, context=None)


if TYPE_CHECKING:
    _: type[frost.Ciphersuite] = Ed25519Sha512
