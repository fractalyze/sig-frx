# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""FROST(Ed25519, SHA-512) — RFC 9591 §6.1, over the Edwards substrate.

The ciphersuite is constants and hash instantiations over the group the
Ed25519 scheme already uses: same curve module, same encode/decode, same
ladder, at `B = 1` on the host. Its aggregate output is a plain RFC 8032
signature — `H2` deliberately omits the domain separator for exactly that
compatibility — so the existing batched Ed25519 verifier is this suite's
verifier, and the tests gate on that crossing.

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

from sig_frx.classical import edwards, group
from sig_frx.threshold import frost

_CONTEXT = b"FROST-ED25519-SHA512-v1"


def _reduced(digest: bytes, order: int) -> int:
    """A 64-byte digest as a little-endian integer modulo the group order."""
    return int.from_bytes(digest, "little") % order


@dataclass(frozen=True)
class Ed25519Sha512:
    """RFC 9591 §6.1's ciphersuite, elements riding as `[1]`-batch points."""

    order = edwards.ED25519.order
    element_size = 32

    curve = edwards.ED25519

    @property
    def scalar_field(self) -> Any:
        # A property so the curve's lazy mint stays lazy — a class attribute
        # would resolve it at import time.
        return edwards.ED25519.scalar_field

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

    def deserialize_element(self, data: bytes) -> edwards.ExtPoint:
        point, ok = edwards.decode(
            self.curve, np.frombuffer(data, dtype=np.uint8)[None, :]
        )
        if not bool(np.asarray(ok)[0]):
            raise ValueError("not a canonical encoding of a curve point")
        # The identity in extended coordinates is (0 : Z : Z : 0) — read off
        # the projective components, sparing the affine division a readback
        # would pay.
        zero = np.array(0, dtype=self.curve.field)
        if bool(np.asarray((point.x == zero) & (point.y == point.z))[0]):
            raise ValueError("the identity element has no place on the wire")
        return point

    def element_add(
        self, left: edwards.ExtPoint, right: edwards.ExtPoint
    ) -> edwards.ExtPoint:
        return edwards.add(self.curve, left, right)

    def element_scalar_mult(
        self, element: edwards.ExtPoint, scalar: int
    ) -> edwards.ExtPoint:
        return edwards.scalar_mul(self.curve, group.int_bits(scalar), element)

    def identity_element(self) -> edwards.ExtPoint:
        return edwards.identity(self.curve, self.curve.generator.x)

    def serialize_element(self, element: edwards.ExtPoint) -> bytes:
        ((x, y),) = group.to_affine_ints(element)
        if (x, y) == (0, 1):
            raise ValueError("the identity element has no encoding here")
        return edwards.encode_affine(x, y)


if TYPE_CHECKING:
    _: type[frost.Ciphersuite] = Ed25519Sha512
