# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Ed25519 per RFC 8032, on the `Signature` seam.

Plain Ed25519 — the variants that prepare a different message (Ed25519ph,
Ed25519ctx) are different operations and will live under their own names, per
the seam's rule. `context` is therefore required empty here: plain Ed25519
defines none.

## Where SHA-512 comes from, and what that blocks

hash-frx ships no SHA-512 — not a host row, not a device row — so this module
reaches `hashlib` directly on the concrete paths, the way RFC 6979's HMAC
does: signing and key generation are host-only (`docs/reference/security.md`)
and `hashlib` is what a host row would wrap anyway. What that cannot cover is
batched verification under a tracer, which needs a device SHA-512 the way
ECDSA's verification uses hash-frx's device SHA-256; until hash-frx grows one,
the traced path raises, and the traced cases carry the classical blocker
marker either way. The decision the issue asked to record: the dependency
lands in hash-frx (where every symmetric primitive lives), not here.

## The verification equation, and which profile this is

`[S]B = R + [k]A`, cofactorless. RFC 8032 §5.1.7 names the cofactored
`[8][S]B = [8]R + [8][k]A` check and calls this one sufficient; which of the
two a consensus system demands — and what it accepts as a canonical encoding
— is exactly the variant surface tracked separately (strict RFC 8032 versus
ZIP-215). The core takes the strict readings the document states outright:
`y ≥ p` fails decoding for both `A` and `R`, and `S ≥ L` is rejected. The
digest scalar `k` stays unreduced in meaning: it reaches the group through
`edwards.wide_multiple`, whose modulo-`8L` reduction is the group's own fact
for every decoded point, torsion components included. `S` rides
`edwards.multiple`'s `% L` against the prime-order base point, with `S ≥ L`
rows already refused by the range check.

Leaving `k` unreduced is a third axis of the same variant surface, and one
worth naming because the reference implementations take the other side: ref10,
libsodium and Go all reduce the digest modulo `L` before the multiply. The two
readings agree on every honest signature and diverge only when `A` carries a
torsion component — so this is a consensus-divergence class, not an interop
bug, and §7.1's vectors cannot gate it in either direction. What decides it
here is that §5.1.7 states the equation over the unreduced scalar, the same
literal-reading rule the rest of this module follows.

Nothing re-encodes a point in verification: the `k` hash absorbs `R`'s and
`A`'s *wire* bytes, which is what the standard hashes too.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from frx.typing import ArrayLike

from sig_frx import context as context_rules
from sig_frx.arrays import namespace
from sig_frx.classical import edwards, group
from sig_frx.signature import Signature


def _clamp(data: bytes) -> int:
    """RFC 8032 §5.1.5's scalar clamping, on the digest's first half."""
    scalar = bytearray(data)
    scalar[0] &= 248
    scalar[31] &= 127
    scalar[31] |= 64
    return int.from_bytes(scalar, "little")


def _sha512_rows(data: Any) -> Any:
    """SHA-512 over the last axis, per batch row — no device row exists yet.

    A concrete batch — host or device — reads its bytes back and hashes
    through `hashlib`: eager is the same code without `jit`, and reading
    concrete bytes is exactly what a host row would do. What cannot work is a
    *tracer*, whose bytes do not exist yet; that call needs the device
    SHA-512 row hash-frx does not ship (fractalyze/hash-frx#66), and refuses
    loudly rather than approximating.
    """
    try:
        rows = np.asarray(data, dtype=np.uint8)
    except Exception as error:
        raise NotImplementedError(
            "batched Ed25519 verification under a tracer needs a device "
            "SHA-512, which hash-frx does not ship yet"
        ) from error
    digests = [hashlib.sha512(row.tobytes()).digest() for row in rows]
    return np.frombuffer(b"".join(digests), dtype=np.uint8).reshape(
        rows.shape[:-1] + (64,)
    )


@dataclass(frozen=True)
class Ed25519:
    """RFC 8032 Ed25519: 32-byte keys, 64-byte `R ‖ S` signatures."""

    public_key_size = 32
    secret_key_size = 32
    signature_max_size = 64
    deterministic = True

    curve = edwards.ED25519

    def keygen(self, seed: ArrayLike) -> tuple[Any, Any]:
        """RFC 8032 §5.1.5: the seed is the secret key; `A = s·B` encoded."""
        seed_bytes = np.asarray(seed, dtype=np.uint8).reshape(-1)
        if seed_bytes.shape[0] != self.secret_key_size:
            raise ValueError(f"a seed is {self.secret_key_size} bytes")
        digest = hashlib.sha512(seed_bytes.tobytes()).digest()
        public = self._encode_multiple_of_b(_clamp(digest[:32]))
        return np.frombuffer(public, dtype=np.uint8).copy(), seed_bytes.copy()

    def sign(
        self,
        secret_key: ArrayLike,
        message: ArrayLike,
        *,
        randomness: ArrayLike | None,
        context: ArrayLike | None,
    ) -> Any:
        """RFC 8032 §5.1.6 — deterministic by construction, so `randomness`
        is ignored and reproducing the published vectors is what gates it."""
        del randomness
        context_rules.require_empty(context, "Ed25519")
        secret = np.asarray(secret_key, dtype=np.uint8).reshape(-1)
        if secret.shape[0] != self.secret_key_size:
            raise ValueError(f"a secret key is {self.secret_key_size} bytes")
        message_bytes = np.asarray(message, dtype=np.uint8).tobytes()
        order = self.curve.order

        digest = hashlib.sha512(secret.tobytes()).digest()
        scalar = _clamp(digest[:32])
        prefix = digest[32:]
        public = self._encode_multiple_of_b(scalar)
        r = int.from_bytes(hashlib.sha512(prefix + message_bytes).digest(), "little")
        r %= order
        commitment = self._encode_multiple_of_b(r)
        k = int.from_bytes(
            hashlib.sha512(commitment + public + message_bytes).digest(), "little"
        )
        s = (r + k % order * scalar) % order
        signature = commitment + s.to_bytes(32, "little")
        return np.frombuffer(signature, dtype=np.uint8).copy()

    def verify(
        self,
        public_key: ArrayLike,
        message: ArrayLike,
        signature: ArrayLike,
        *,
        context: ArrayLike | None,
    ) -> Any:
        """The batched verdict, `bool[B]` — strict decoding, cofactorless."""
        context_rules.require_empty(context, "Ed25519")
        curve = self.curve
        xnp = namespace(public_key, message, signature)
        public_key = xnp.asarray(public_key)
        message = xnp.asarray(message)
        signature = xnp.asarray(signature)

        # The digest comes first: it is where a tracer is refused (no device
        # SHA-512 yet), and everything after it is host codec and kernels
        # that need concrete bytes.
        digest = _sha512_rows(
            xnp.concatenate([signature[..., :32], public_key, message], axis=-1)
        )
        public_key = np.asarray(public_key, dtype=np.uint8)
        signature = np.asarray(signature, dtype=np.uint8)

        point_a, a_ok = edwards.decode(curve, public_key)
        point_r, r_ok = edwards.decode(curve, signature[..., :32])
        s_bytes = signature[..., 32:64]
        s_ok = group.bytes_below(s_bytes, curve.order, byteorder="little")

        # Both scalars are little-endian integers off the wire and digest
        # bytes. An S at or above L is gated by s_ok, so multiple's % L only
        # ever rewrites a row the verdict already refuses.
        s_ints = [int.from_bytes(row.tobytes(), "little") for row in s_bytes]
        k_ints = [int.from_bytes(row.tobytes(), "little") for row in digest]

        lhs = edwards.multiple(curve, s_ints, curve.generator)
        rhs = point_r + edwards.wide_multiple(curve, k_ints, point_a)
        return a_ok & r_ok & s_ok & (lhs == rhs)

    def _encode_multiple_of_b(self, scalar: int) -> bytes:
        """`scalar·B` encoded per §5.1.2, on the host path.

        `B` generates the prime-order subgroup, so `multiple`'s `% L` is the
        group's own reduction here (the clamped scalar does exceed `L`).
        """
        curve = self.curve
        ((x, y),) = edwards.affine_ints(
            curve, edwards.multiple(curve, [scalar], curve.generator)
        )
        return edwards.encode_affine(x, y)


if TYPE_CHECKING:
    _: type[Signature] = Ed25519
