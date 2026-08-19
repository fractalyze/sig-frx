# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""FROST(secp256k1, SHA-256) — RFC 9591 §6.5, over the Weierstrass substrate.

The second ciphersuite, and the one that keeps the skeleton honest: no round
logic lives here, only §6.5's constants — SEC 1 compressed points, big-endian
scalars, and the hash-to-field scalar derivations (RFC 9380's
`expand_message_xmd`, where the Edwards suite reduces a raw digest).

One caveat stated where it can be read rather than discovered: this suite's
output is **not** a BIP-340 signature — x-only keys and tagged hashes differ —
so it has no Taproot verifier. It is RFC 9591's own Schnorr encoding, verified
per the RFC's Appendix B, which the tests transcribe.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sig_frx.classical import secp
from sig_frx.threshold import frost, xmd

_CONTEXT = b"FROST-secp256k1-SHA256-v1"

# hash_to_field's L for this suite: RFC 9591 §6.5 sets L = 48.
_FIELD_BYTES = 48


@dataclass(frozen=True)
class Secp256k1Sha256:
    """RFC 9591 §6.5's ciphersuite, elements riding as `[1]`-batch points."""

    order = secp.SECP256K1.n
    element_size = 33

    curve = secp.SECP256K1

    def _hash_to_scalar(self, label: bytes, message: bytes) -> int:
        return xmd.hash_to_scalar(message, _CONTEXT + label, self.order, _FIELD_BYTES)

    def h1(self, message: bytes) -> int:
        return self._hash_to_scalar(b"rho", message)

    def h2(self, message: bytes) -> int:
        return self._hash_to_scalar(b"chal", message)

    def h3(self, message: bytes) -> int:
        return self._hash_to_scalar(b"nonce", message)

    def h4(self, message: bytes) -> bytes:
        return hashlib.sha256(_CONTEXT + b"msg" + message).digest()

    def h5(self, message: bytes) -> bytes:
        return hashlib.sha256(_CONTEXT + b"com" + message).digest()

    def serialize_scalar(self, scalar: int) -> bytes:
        return (scalar % self.order).to_bytes(32, "big")

    def deserialize_scalar(self, data: bytes) -> int:
        if len(data) != 32:
            raise ValueError("a serialized scalar is 32 bytes")
        value = int.from_bytes(data, "big")
        if value >= self.order:
            raise ValueError("a scalar is in [0, n-1]")
        return value

    def scalar_base_mult(self, scalar: int) -> bytes:
        return self.serialize_element(
            self.element_scalar_mult(self.curve.generator, scalar)
        )

    def deserialize_element(self, data: bytes) -> np.ndarray:
        """SEC 1 §2.3.4 compressed decoding plus §3.2.2.1 validation."""
        if len(data) != 33 or data[0] not in (2, 3):
            raise ValueError("a serialized element is 33 bytes, prefix 02 or 03")
        x = int.from_bytes(data[1:], "big")
        if x >= self.curve.p:
            raise ValueError("the x-coordinate is out of range")
        points, on_curve = secp.lift_x_to_parity(self.curve, [x], [data[0] & 1])
        if not bool(on_curve[0]):
            raise ValueError("the x-coordinate is not on the curve")
        return points.astype(self.curve.accumulator)

    def element_add(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return left + right

    def element_scalar_mult(self, element: np.ndarray, scalar: int) -> np.ndarray:
        return secp.multiple(self.curve, [scalar], element)

    def identity_element(self) -> np.ndarray:
        return np.zeros([1], dtype=self.curve.accumulator)

    def serialize_element(self, element: np.ndarray) -> bytes:
        if bool(secp.is_identity(self.curve, element)[0]):
            raise ValueError("the identity element has no encoding here")
        ((x, y),) = secp.affine_ints(self.curve, element)
        return (2 + (y & 1)).to_bytes(1, "big") + x.to_bytes(32, "big")


if TYPE_CHECKING:
    _: type[frost.Ciphersuite] = Secp256k1Sha256
