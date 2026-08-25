# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The chain-agnostic ECDSA core: SEC 1 over an injected curve and hash.

No blockchain is named here and no message convention is chosen: the curve is
a `secp.Curve`, the hash is a `MessageHash`, and everything Ethereum and
Bitcoin disagree about — message preparation, recovery, low-`S` policy,
encodings — rides above this module as variants.

## One host path, on the curated point types

The mod-n formula cores ride the curve's scalar-field dtype — `s⁻¹` is field
division, `u₁ = e·s⁻¹` and `u₂ = r·s⁻¹` are field products — while the
representative facts the standards define on integers stay Python integers:
RFC 6979's HMAC chain, the `x(R) mod n` readback, parities and low-`S`, the
`[1, n-1]` bounds (rejected, never reduced), and the wire encodings. The
bounds run first for a second reason: a field constructor or int operand
outside `[0, n)` aborts instead of reducing (fractalyze/zk_dtypes#179), so
every integer reaching a field expression is already reduced. Point
arithmetic is the curve's zk_dtypes kernels through `secp.py`, batched
across each call. Verification
is the standard's own §4.1.4: `u₁ = e·s⁻¹`, `u₂ = r·s⁻¹ (mod n)`, accept iff
`x(u₁G + u₂Q) mod n = r` — which takes the `x(R) = r + n` wrap along for
free. An earlier reshaped, inversion-free form existed to keep mod-n
arithmetic off a traced device path; it retired with that path, and the GPU
story for these curves is EC kernels over the same dtypes (the decision is
recorded on fractalyze/sig-frx#139). Batch-first stays the seam's contract
regardless: one `bool[B]` per call, the point kernels batched across it.

Public-key recovery (SEC 1 §4.1.6) lives here rather than in a chain variant
because the curve defines it; Ethereum is merely its loudest consumer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from frx.typing import ArrayLike

from sig_frx import context as context_rules
from sig_frx import hashes
from sig_frx.classical import secp
from sig_frx.classical.ecdsa import rfc6979
from sig_frx.signature import Signature


@dataclass(frozen=True)
class MessageHash:
    """The hash a variant pairs with the curve — both of its faces.

    `byte_hash` is a namespace dispatcher in the `sig_frx.hashes` shape,
    serving batched hashing over message rows. `host_constructor` is the
    `hashlib`-style constructor the signing path uses for the message and
    for RFC 6979's HMAC — the RFC requires those two to be the same `H`,
    and the known-answer `k` values are what hold this record's two faces
    to one function.
    """

    byte_hash: Callable[..., Any]
    host_constructor: rfc6979.HashConstructor


# The FIPS 186-5 pairing both curves are deployed with.
SHA256 = MessageHash(byte_hash=hashes.sha256, host_constructor=hashlib.sha256)


def _signature_bytes(r: int, s: int) -> np.ndarray:
    """`r ‖ s` as the 64-byte wire row both signing surfaces emit."""
    packed = r.to_bytes(32, "big") + s.to_bytes(32, "big")
    return np.frombuffer(packed, dtype=np.uint8).copy()


def _masked_quotient_pair(
    scalar: Any, valid: bool, u1_numerator: int, u2_numerator: int, denominator: int
) -> tuple[int, int]:
    """The two field quotients `(a/d, b/d)` a batch row rides, as ints.

    A rejected row's wire values may exceed `n`, and a field op on such an
    operand aborts rather than reducing (fractalyze/zk_dtypes#179) — so a
    masked row rides zero scalars on its placeholder point instead; its
    verdict is already sealed.
    """
    if not valid:
        return 0, 0
    w = scalar(denominator) ** -1
    return int(scalar(u1_numerator) * w), int(scalar(u2_numerator) * w)


def is_low_s(curve: secp.Curve, signature: ArrayLike) -> np.ndarray:
    """Whether each `r ‖ s` signature carries the low half of `s`: `bool[B]`.

    A curve-level fact with chain-policy consumers: SEC 1 accepts both
    halves, so the core's own verification never consults this — the
    variants that reject the high half (their malleability rules) share the
    predicate instead of each re-deriving `n/2`.
    """
    rows = np.asarray(signature, dtype=np.uint8)
    half = curve.n // 2
    return np.array(
        [int.from_bytes(entry[32:64].tobytes(), "big") <= half for entry in rows],
        dtype=bool,
    )


@dataclass(frozen=True)
class Ecdsa:
    """ECDSA per SEC 1 over an injected curve and hash, on the `Signature` seam.

    Keys and signatures cross in their standard encodings: an uncompressed
    SEC 1 §2.3.3 public key (`04 ‖ X ‖ Y`, 65 bytes), a 32-byte big-endian
    secret scalar, and the fixed 64-byte `r ‖ s` concatenation — the padded
    form; DER is a chain convention and lives with the variant that wants it.

    `low_s` normalizes `s` to the smaller of `{s, n-s}` at signing. Off by
    default: it is a chain policy (a malleability rule), not an ECDSA rule,
    and verification accepts both halves either way, as SEC 1 does.
    """

    curve: secp.Curve
    hash: MessageHash
    low_s: bool = False

    public_key_size = 65
    secret_key_size = 32
    signature_max_size = 64
    deterministic = True

    def keygen(self, seed: ArrayLike) -> tuple[Any, Any]:
        """The key pair whose secret scalar is `seed` itself.

        SEC 1 §3.2.1 defines a valid key pair as `d ∈ [1, n-1]` with `Q = dG`;
        no standard derivation from a shorter seed exists for ECDSA, so the
        seed *is* the 32-byte scalar encoding, and a value outside the valid
        range is refused rather than reduced — reduction would silently map
        two seeds to one key.
        """
        seed_bytes, scalar = secp.secret_scalar(self.curve, seed, "seed")
        x, y = secp.host_multiple_of_g(self.curve, scalar)
        public = b"\x04" + x.to_bytes(32, "big") + y.to_bytes(32, "big")
        return (
            np.frombuffer(public, dtype=np.uint8).copy(),
            seed_bytes.copy(),
        )

    def sign(
        self,
        secret_key: ArrayLike,
        message: ArrayLike,
        *,
        randomness: ArrayLike | None,
        context: ArrayLike | None,
    ) -> Any:
        """One RFC 6979 deterministic signature: `r ‖ s`, 64 bytes.

        `randomness` is ignored — the instance is deterministic, which is what
        makes signing reproducible against the published vectors. `context` is
        required empty: ECDSA's standards define none, and accepting one only
        to ignore it would verify something other than what the caller asked.
        """
        signature, _ = self.sign_recoverable(
            secret_key, message, randomness=randomness, context=context
        )
        return signature

    def sign_recoverable(
        self,
        secret_key: ArrayLike,
        message: ArrayLike,
        *,
        randomness: ArrayLike | None,
        context: ArrayLike | None,
    ) -> tuple[Any, int]:
        """`sign`, plus the recovery id naming which point `recover` rebuilds.

        The id is curve-level data — `y(R)`'s parity and whether `x(R)`
        wrapped past `n` — so a signer emits it without knowing what a chain
        is; how it rides the wire (`v`, a header byte) is the variant's
        encoding. Same signature bytes as `sign`, by construction.
        """
        del randomness  # deterministic (RFC 6979); the seam documents this.
        context_rules.require_empty(context, "ECDSA")
        _, x = secp.secret_scalar(self.curve, secret_key, "secret key")
        r, s, recovery_id = self._signature_scalars(x, message)
        return _signature_bytes(r, s), recovery_id

    def sign_digest_recoverable(
        self,
        secret_key: ArrayLike,
        digest: ArrayLike,
        *,
        nonce_hash: rfc6979.HashConstructor,
    ) -> tuple[Any, int]:
        """`sign_recoverable` for a caller that arrives with the digest.

        Off the seam under its own name, per the seam's rule for pre-hashed
        variants. The message-level record binds `H(message)` and RFC 6979's
        HMAC to one function; a digest-level caller has severed that by
        construction, so the HMAC face is named explicitly instead of read
        off the record. (Ethereum's ecosystem convention — libsecp256k1's
        default nonce function — is HMAC-SHA256 beneath a Keccak-256 digest,
        and the EIP-155 example vector pins that pairing in the variant's
        tests.)
        """
        _, x = secp.secret_scalar(self.curve, secret_key, "secret key")
        h1 = np.asarray(digest, dtype=np.uint8).tobytes()
        r, s, recovery_id = self._digest_signature_scalars(x, h1, nonce_hash)
        return _signature_bytes(r, s), recovery_id

    def _signature_scalars(self, x: int, message: ArrayLike) -> tuple[int, int, int]:
        """One RFC 6979 signature as integers, from the record's own pairing."""
        h1 = self.hash.host_constructor(
            np.asarray(message, dtype=np.uint8).tobytes()
        ).digest()
        return self._digest_signature_scalars(x, h1, self.hash.host_constructor)

    def _digest_signature_scalars(
        self, x: int, h1: bytes, nonce_hash: rfc6979.HashConstructor
    ) -> tuple[int, int, int]:
        """One RFC 6979 signature as integers: `(r, s, recovery_id)`.

        The recovery id is two bits off the nonce's point, read before it is
        thrown away — bit 0 is `y(R)`'s parity, bit 1 is `x(R) >= n` (SEC 1
        §4.1.6's second candidate pair). The low-S flip replaces `s` with
        `n - s`, which verifies against `-R` instead of `R`, so the parity
        bit flips with it — the round-trip tests hold the two together.
        """
        n = self.curve.n
        scalar = self.curve.scalar
        e = rfc6979.bits2int(h1, n.bit_length()) % n
        for k in rfc6979.nonces(n, x, h1, nonce_hash):
            rx, ry = secp.host_multiple_of_g(self.curve, k)
            r = rx % n
            if r == 0:  # SEC 1 §4.1.3 discards the draw and takes the next
                continue
            s = int((scalar(e) + scalar(r) * x) / scalar(k))
            if s == 0:
                continue
            recovery_id = 2 * (rx >= n) + (ry & 1)
            if self.low_s and s > n // 2:
                s = n - s
                recovery_id ^= 1
            return r, s, recovery_id
        raise AssertionError("unreachable: the nonce stream is infinite")

    def verify(
        self,
        public_key: ArrayLike,
        message: ArrayLike,
        signature: ArrayLike,
        *,
        context: ArrayLike | None,
    ) -> Any:
        """The batched verdict, `bool[B]`, over the record's message hash."""
        context_rules.require_empty(context, "ECDSA")
        digest = self.hash.byte_hash(message).digest(message)
        return self.verify_digest(public_key, digest, signature)

    def verify_digest(
        self,
        public_key: ArrayLike,
        digest: ArrayLike,
        signature: ArrayLike,
    ) -> Any:
        """SEC 1 §4.1.4 over a digest batch: `bool[B]`.

        The per-entry rejections are the standard's: a key must be the
        `04 ‖ X ‖ Y` encoding of a point on the curve with in-field
        coordinates (§2.3.4 — range and equation both live in
        `secp.on_curve_rows`, since the dtypes construct off-curve
        coordinates without complaint), and
        `r, s` must sit in `[1, n-1]`. Rejected rows carry the generator as
        a placeholder so the batch stays rectangular; their verdict is
        already sealed.

        The key check runs over the whole batch before the loop while the
        rest stays per-row, and the split is where the measurement put it
        rather than where the shape suggests: at B=1024 the curve equation
        was 19% of this call as `on_curve` per row and is 2% as one
        expression over `[B]`, while the `int.from_bytes` decode it sits
        behind is 4%. The remaining per-row work — the quotient pair's
        modular inverse and the point construction — is tracked separately.
        """
        curve = self.curve
        n = curve.n
        scalar = curve.scalar
        keys = np.asarray(public_key, dtype=np.uint8)
        digest = np.asarray(digest, dtype=np.uint8)
        signature = np.asarray(signature, dtype=np.uint8)

        qlen = n.bit_length()
        placeholder = curve.point((curve.gx, curve.gy))

        # The coordinates are parsed once for the whole batch so the curve
        # equation can be checked as one expression over `[B]` rather than a
        # 0-d field array per coordinate — see `secp.on_curve_rows`, which is
        # 9x the row-at-a-time form at B=1024. The parse itself stays here:
        # `int.from_bytes` beats a field-arithmetic weighted sum on bytes by
        # an order of magnitude, so it is the batching that pays, not moving
        # the decode into the field.
        qxs = [int.from_bytes(key[1:33].tobytes(), "big") for key in keys]
        qys = [int.from_bytes(key[33:65].tobytes(), "big") for key in keys]
        key_ok = secp.on_curve_rows(curve, qxs, qys)

        checks, points, u1_scalars, u2_scalars, r_scalars = [], [], [], [], []
        for i, (key, entry, sig) in enumerate(zip(keys, digest, signature)):
            qx, qy = qxs[i], qys[i]
            r = int.from_bytes(sig[:32].tobytes(), "big")
            s = int.from_bytes(sig[32:64].tobytes(), "big")
            e = rfc6979.bits2int(entry.tobytes(), qlen) % n
            ok = int(key[0]) == 4 and bool(key_ok[i]) and 1 <= r < n and 1 <= s < n
            u1, u2 = _masked_quotient_pair(scalar, ok, e, r, s)
            u1_scalars.append(u1)
            u2_scalars.append(u2)
            r_scalars.append(r)
            checks.append(ok)
            points.append(curve.point((qx, qy)) if ok else placeholder)

        q_points = np.array(points, dtype=curve.point)
        big_r = secp.double_multiple(curve, u1_scalars, u2_scalars, q_points)
        gone = secp.is_identity(curve, big_r)
        verdicts = [
            ok and not bool(dead) and x % n == r
            for (x, _), ok, dead, r in zip(
                secp.affine_ints(curve, big_r), checks, gone, r_scalars
            )
        ]
        return np.array(verdicts, dtype=bool)

    def recover(
        self,
        message: ArrayLike,
        signature: ArrayLike,
        recovery_id: ArrayLike,
        *,
        context: ArrayLike | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The signing keys back out of a batch: `(uint8[B, 65], bool[B])`.

        SEC 1 §4.1.6, with the candidate index made explicit instead of
        searched: `recovery_id[i]`'s bit 0 is `y(R)`'s parity and bit 1 is
        whether `x(R)` wrapped past `n`.

        A per-entry failure — `r` or `s` out of range, an id naming no curve
        point, the identity result — clears that entry's verdict and zeroes
        its key row. Rejection is a verdict rather than an exception because
        signatures and ids are wire data: one bad entry must not take down
        the batch, and a zeroed row cannot be mistaken for a key.
        """
        context_rules.require_empty(context, "ECDSA")
        message = np.asarray(message, dtype=np.uint8)
        # The record's batched hash row — the same function signing uses.
        digest = np.asarray(self.hash.byte_hash(message).digest(message))
        return self.recover_digest(digest, signature, recovery_id)

    def recover_digest(
        self,
        digest: ArrayLike,
        signature: ArrayLike,
        recovery_id: ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray]:
        """`recover` for a caller that arrives with the digests: `[B, L]`.

        Off the seam under its own name, like `sign_digest_recoverable` — and
        with no `context` parameter, because context framing belongs to the
        seam's message-level surface. Everything else, verdicts included, is
        `recover`'s contract.
        """
        digest = np.asarray(digest, dtype=np.uint8)
        signature = np.asarray(signature, dtype=np.uint8)
        recovery_id = np.asarray(recovery_id)
        if signature.shape[-1] != 64:
            raise ValueError("a signature is 64 bytes of r ‖ s")
        if not digest.shape[0] == signature.shape[0] == recovery_id.shape[0]:
            raise ValueError("the batch axes disagree")
        n = self.curve.n
        qlen = n.bit_length()
        digest_scalars = [
            rfc6979.bits2int(entry.tobytes(), qlen) % n for entry in digest
        ]
        return self._recover(digest_scalars, signature, recovery_id)

    def _recover(
        self,
        digest_scalars: list[int],
        signature: np.ndarray,
        recovery_id: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recovery below the message hash: `e` arrives as scalars.

        A seam of its own so the identity-result guard is testable — forcing
        `s·R = e·G` through `recover` would take a hash preimage, and
        unreachable-in-production is not a licence to leave it ungated.
        """
        curve = self.curve
        p, n = curve.p, curve.n
        r_scalars = [int.from_bytes(entry[:32].tobytes(), "big") for entry in signature]
        s_scalars = [int.from_bytes(entry[32:].tobytes(), "big") for entry in signature]
        ids = [int(entry) for entry in recovery_id]

        # §4.1.6's plain-integer checks: the candidate x with the wrapped
        # offset, and the range bounds. Failed entries keep an in-field dummy
        # so the batch stays rectangular; their verdict is already sealed.
        checks, xs = [], []
        for r, s, j in zip(r_scalars, s_scalars, ids):
            x = r + (j >> 1) * n
            checks.append(0 <= j <= 3 and 1 <= r < n and 1 <= s < n and x < p)
            xs.append(x % p)
        ok = np.array(checks, dtype=bool)

        # R rebuilt from x; the id's bit 0 is the parity that picks between
        # the root and its negation.
        point_r, lifted = secp.lift_x_to_parity(curve, xs, [j & 1 for j in ids])
        ok &= lifted

        # Q = r⁻¹(sR - eG), taken as u₁G + u₂R with u₁ = -e/r, u₂ = s/r
        # (mod n): the scalar field for the algebra, the curve's kernels for
        # the points.
        scalar = curve.scalar
        u1_scalars, u2_scalars = [], []
        for e, r, s, valid in zip(digest_scalars, r_scalars, s_scalars, ok):
            # -e % n: the field constructor rejects a negative int outright
            # (OverflowError), so the negation arrives already canonical.
            u1, u2 = _masked_quotient_pair(scalar, bool(valid), -e % n, s, r)
            u1_scalars.append(u1)
            u2_scalars.append(u2)

        public = secp.double_multiple(curve, u1_scalars, u2_scalars, point_r)

        # The identity has no encoding, so it is a rejection, not a key
        # (§4.1.6 rejects it by name).
        ok &= ~secp.is_identity(curve, public)
        return secp.uncompressed_rows(curve, public, ok), ok


if TYPE_CHECKING:
    _: type[Signature] = Ecdsa
