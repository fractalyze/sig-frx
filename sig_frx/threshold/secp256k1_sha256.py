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
from collections.abc import Sequence
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
        (encoded,) = self.serialize_elements(
            self.elements_scalar_mult(self.curve.generator, [scalar])
        )
        return encoded

    def deserialize_elements(self, data: Sequence[bytes]) -> np.ndarray:
        """SEC 1 §2.3.4 compressed decoding plus §3.2.2.1 validation, batched.

        The prefix and range checks are per entry and stay on the host — they
        read bytes, and a rejected entry has to name itself. The lift is the
        one piece with an axis to be large along, so it runs once over the
        whole batch, which is also where `secp` decides the namespace.
        """
        xs, parities = [], []
        for index, entry in enumerate(data):
            if len(entry) != 33 or entry[0] not in (2, 3):
                raise ValueError(
                    f"a serialized element is 33 bytes, prefix 02 or 03 (entry {index})"
                )
            x = int.from_bytes(entry[1:], "big")
            if x >= self.curve.p:
                raise ValueError(f"the x-coordinate is out of range (entry {index})")
            xs.append(x)
            parities.append(entry[0] & 1)
        points, on_curve = secp.lift_x_to_parity(self.curve, xs, parities)
        off = [i for i, ok in enumerate(np.asarray(on_curve)) if not bool(ok)]
        if off:
            raise ValueError(f"the x-coordinate is not on the curve (entries {off})")
        return points.astype(self.curve.accumulator)

    def elements_add(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return left + right

    def elements_scalar_mult(
        self, elements: np.ndarray, scalars: Sequence[int]
    ) -> np.ndarray:
        return secp.multiple(self.curve, list(scalars), elements)

    def sum_elements(self, elements: np.ndarray) -> np.ndarray:
        return secp.sum_points(self.curve, elements)

    def serialize_elements(self, elements: np.ndarray) -> list[bytes]:
        identity = np.asarray(secp.is_identity(self.curve, elements))
        bad = [i for i, flag in enumerate(identity) if bool(flag)]
        if bad:
            raise ValueError(
                f"the identity element has no encoding here (entries {bad})"
            )
        return [
            secp.compressed_bytes(self.curve, x, y)
            for x, y in secp.affine_ints(self.curve, elements)
        ]

    def verify(
        self, public_key: ArrayLike, message: ArrayLike, signature: ArrayLike
    ) -> np.ndarray:
        """RFC 9591 Appendix B's prime-order verification: `bool[B]` verdicts.

        The challenge is the round skeleton's own §4.6 derivation
        (`frost.compute_challenge`), and accept-iff-`[z]B = R + [c]PK` is
        the substrate's shared Schnorr readback (`secp.schnorr_verdicts`)
        against `R`'s x and parity — a point equal to `R` is on the curve
        by construction, so the wire checks that remain per row are SEC 1
        §2.3.4's — a `02`/`03` prefix and a coordinate below `p` — plus the
        RFC's scalar bound `z < n`. A failing row carries masked junk to a
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

        key_bytes = [row.tobytes() for row in keys]
        r_bytes = [row.tobytes() for row in signatures[:, : self.element_size]]
        z_scalars = [
            int.from_bytes(row.tobytes(), "big")
            for row in signatures[:, self.element_size :]
        ]

        def wire(encodings: list[bytes]) -> list[tuple[int, int]]:
            return [
                (encoding[0], int.from_bytes(encoding[1:], "big"))
                for encoding in encodings
            ]

        pk_wire = wire(key_bytes)
        r_wire = wire(r_bytes)
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
            frost.compute_challenge(self, commitment, key, msg.tobytes())
            for commitment, key, msg in zip(r_bytes, key_bytes, messages)
        ]
        return secp.schnorr_verdicts(
            curve,
            z_scalars,
            challenges,
            key_points,
            [x for _, x in r_wire],
            [prefix & 1 for prefix, _ in r_wire],
            ok,
        )


if TYPE_CHECKING:
    _: type[frost.Ciphersuite] = Secp256k1Sha256
