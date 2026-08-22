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

`[S]B = R + [k]A`, cofactorless, over strictly decoded points: `y ≥ p` fails
decoding for both `A` and `R`, an `x = 0` carrying the sign bit fails with
it, and `S ≥ L` is rejected. That is RFC 8032 §5.1.3 and §5.1.7 read
literally — §5.1.7 states the cofactored `[8][S]B = [8]R + [8][k]A` first
and calls this one sufficient in its place.

Which of the two a consensus system demands, and what it accepts as an
encoding, is not a robustness knob — so it is not a `strict=` flag but a
`ValidationRule` fixed per construction, with the other two named in
[`consensus.py`](consensus.py). A rule differs from its neighbours on
several axes at once, which is why a caller names one rather than composing
it.

The digest scalar `k` reaches the group reduced modulo `L`, through
`edwards.multiple`. RFC 8032 words its equation over the unreduced integer,
but words it over the *cofactored* form — where the two readings provably
agree, since multiplying by 8 clears the torsion component that is the only
place they can differ. Reducing is therefore not a departure from the
document, and it is what ref10, libsodium, Go and ed25519-dalek all do. The
interoperability vectors settle it: a cofactorless verifier that kept `k`
wide disagrees with all of them on 178 of the 914 cases, matching no
published rule
([`../testing/ed25519_cctv_vectors.py`](../testing/ed25519_cctv_vectors.py)).

Nothing re-encodes a point in verification: the `k` hash absorbs `R`'s and
`A`'s *wire* bytes, which is what the standard hashes too — and the vectors'
`reencoded_k` cases are there to separate that reading from the other one.
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
class ValidationRule:
    """Which Ed25519 signatures a verifier accepts.

    Three readings the standards leave open, pinned together because a
    consensus system has to answer all three at once and because the
    published rules each move more than one of them
    ([`consensus.py`](consensus.py) tabulates who takes which).

    - `canonical_encodings` — whether `A` and `R` must satisfy RFC 8032
      §5.1.3's refusals, or whether any bytes that encode a curve point are
      taken (`edwards.decode`).
    - `reject_small_order` — whether an `A` or `R` whose order divides the
      cofactor is refused before the equation runs. RFC 8032 asks for no
      such check.
    - `cofactored` — `[8][S]B = [8]R + [8][k]A` rather than §5.1.7's
      sufficient `[S]B = R + [k]A`. The cofactored form is the one batch
      verification can aggregate, which is why ZIP-215 mandates it.
    """

    canonical_encodings: bool
    reject_small_order: bool
    cofactored: bool


# RFC 8032 read literally: §5.1.3's refusals, no small-order check the
# document does not ask for, and §5.1.7's sufficient cofactorless equation.
RFC_8032 = ValidationRule(
    canonical_encodings=True, reject_small_order=False, cofactored=False
)


@dataclass(frozen=True)
class ParsedBatch:
    """A wire batch decoded under one rule, before any equation runs.

    Plain data that never crosses a `jit` or `vmap` boundary — the whole
    parse is host codec — so it is not a registered pytree, per the
    conventions' rule for a record that stays inside one call.
    """

    point_a: np.ndarray
    point_r: np.ndarray
    s_ints: list[int]
    k_ints: list[int]
    ok: Any


@dataclass(frozen=True)
class Ed25519:
    """RFC 8032 Ed25519: 32-byte keys, 64-byte `R ‖ S` signatures."""

    public_key_size = 32
    secret_key_size = 32
    signature_max_size = 64
    deterministic = True

    curve = edwards.ED25519
    # A class attribute, not a field: the rule a construction verifies under
    # is what the construction *is*, so there is no constructor to pass a
    # different one through.
    rule = RFC_8032

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
        """The batched verdict, `bool[B]`, under this construction's rule."""
        context_rules.require_empty(context, "Ed25519")
        curve = self.curve
        parsed = self._parsed(public_key, message, signature)

        lhs = edwards.multiple(curve, parsed.s_ints, curve.generator)
        rhs = parsed.point_r + edwards.multiple(curve, parsed.k_ints, parsed.point_a)
        if self.rule.cofactored:
            lhs = edwards.mul_by_cofactor(curve, lhs)
            rhs = edwards.mul_by_cofactor(curve, rhs)
        return parsed.ok & (lhs == rhs)

    def _parsed(
        self,
        public_key: ArrayLike,
        message: ArrayLike,
        signature: ArrayLike,
    ) -> ParsedBatch:
        """The wire batch under this rule: points, scalars, and the verdicts
        that do not depend on which equation runs.

        Everything both verification forms share, so the aggregate check
        cannot drift from the per-signature one on a decoding or a refusal
        (`consensus.py`).
        """
        curve = self.curve
        rule = self.rule
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

        canonical_only = rule.canonical_encodings
        point_a, a_ok = edwards.decode(curve, public_key, canonical_only=canonical_only)
        point_r, r_ok = edwards.decode(
            curve, signature[..., :32], canonical_only=canonical_only
        )
        s_bytes = signature[..., 32:64]
        ok = a_ok & r_ok & group.bytes_below(s_bytes, curve.order, byteorder="little")
        if rule.reject_small_order:
            ok = ok & ~(
                edwards.is_small_order(curve, point_a)
                | edwards.is_small_order(curve, point_r)
            )

        # Both scalars are little-endian integers off the wire and digest
        # bytes, and `multiple` reduces each modulo L. That is the reading
        # of k the module docstring argues for; for S it rewrites nothing,
        # since an S at or above L is already refused above.
        return ParsedBatch(
            point_a=point_a,
            point_r=point_r,
            s_ints=[int.from_bytes(row.tobytes(), "little") for row in s_bytes],
            k_ints=[int.from_bytes(row.tobytes(), "little") for row in digest],
            ok=ok,
        )

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
