# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""FROST(secp256k1, SHA-256) — RFC 9591 §6.5, over the Weierstrass substrate.

The second ciphersuite, and the one that keeps the skeleton honest: no round
logic lives here, only §6.5's constants — SEC 1 compressed points, big-endian
scalars, and the hash-to-field scalar derivations (RFC 9380's
`expand_message_xmd`, where the Edwards suite reduces a raw digest).

One caveat stated where it can be read rather than discovered: this suite's
output is **not** a BIP-340 signature — x-only keys and tagged hashes differ —
so it has no Taproot verifier. It is RFC 9591's own Schnorr encoding, verified
per the RFC's Appendix B by `verify` below; the tests keep the RFC's naive
transcription beside it as the reference pair.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from frx.typing import ArrayLike

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
    scalar_field = secp.SECP256K1.scalar

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

    def verify(
        self, public_key: ArrayLike, message: ArrayLike, signature: ArrayLike
    ) -> np.ndarray:
        """RFC 9591 Appendix B's prime-order verification: `bool[B]` verdicts.

        `c = H2(R ‖ PK ‖ msg)`; accept iff `[z]B = R + [c]PK`, computed as
        `[z]B - [c]PK` and compared against `R`'s x and parity — the same
        two-term combination and coordinate readback the BIP-340 verifier
        rides on this substrate. A point equal to `R` is on the curve by
        construction, so the wire checks that remain per row are SEC 1
        §2.3.4's — a `02`/`03` prefix and a coordinate below `p` — plus the
        RFC's scalar bound `z < n`; a failing row carries masked junk to a
        cleared verdict instead of raising out of the batch, and no
        unvalidated integer meets a field op on the way (the scalar-field
        dtype aborts on an out-of-range operand, fractalyze/zk_dtypes#179 —
        the substrate reduces every scalar first).
        """
        curve = self.curve
        keys = np.asarray(public_key, dtype=np.uint8)
        messages = np.asarray(message, dtype=np.uint8)
        signatures = np.asarray(signature, dtype=np.uint8)
        if keys.shape[-1] != self.element_size:
            raise ValueError("a group public key is a 33-byte compressed point")
        if signatures.shape[-1] != self.element_size + 32:
            raise ValueError("a signature is a 33-byte element and a 32-byte scalar")

        def wire(rows: np.ndarray) -> list[tuple[int, int]]:
            return [
                (int(row[0]), int.from_bytes(row[1:].tobytes(), "big")) for row in rows
            ]

        pk_wire = wire(keys)
        r_wire = wire(signatures[..., : self.element_size])
        z_scalars = [
            int.from_bytes(row.tobytes(), "big")
            for row in signatures[..., self.element_size :]
        ]
        p = curve.p
        checks = [
            pk_prefix in (2, 3)
            and pk_x < p
            and r_prefix in (2, 3)
            and r_x < p
            and z < self.order
            for (pk_prefix, pk_x), (r_prefix, r_x), z in zip(pk_wire, r_wire, z_scalars)
        ]
        key_points, on_curve = secp.lift_x_to_parity(
            curve,
            [x % p for _, x in pk_wire],
            [prefix & 1 for prefix, _ in pk_wire],
        )
        ok = np.array(checks, dtype=bool) & on_curve

        challenges = [
            self.h2(sig[: self.element_size].tobytes() + key.tobytes() + msg.tobytes())
            for key, msg, sig in zip(keys, messages, signatures)
        ]
        big_r = secp.double_multiple(
            curve, z_scalars, [-c % self.order for c in challenges], key_points
        )
        # Reject the identity before its `(0, 0)` readback can match an
        # `R` claiming `x = 0` — the same guard the BIP-340 verifier holds.
        gone = secp.is_identity(curve, big_r)
        verdicts = [
            bool(valid) and not bool(dead) and x == r_x and y % 2 == (r_prefix & 1)
            for (x, y), valid, dead, (r_prefix, r_x) in zip(
                secp.affine_ints(curve, big_r), ok, gone, r_wire
            )
        ]
        return np.array(verdicts, dtype=bool)


if TYPE_CHECKING:
    _: type[frost.Ciphersuite] = Secp256k1Sha256
