# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The chain-agnostic ECDSA core: SEC 1 over an injected curve and hash.

No blockchain is named here and no message convention is chosen: the curve is
a `weierstrass.Curve`, the hash is a `MessageHash`, and everything Ethereum
and Bitcoin disagree about — message preparation, recovery, low-`S` policy,
encodings — rides above this module as variants.

## Verification without a modular inversion

SEC 1 §4.1.4 computes `R = u₁G + u₂Q` with `u₁ = e·s⁻¹, u₂ = r·s⁻¹ (mod n)`
and accepts iff `x(R) mod n = r`. The reshaped form here multiplies through
by `s`: accept iff there is a point `R'` with x-coordinate `r` (or `r + n`,
when that still fits below `p`) such that `s·R' = e·G + r·Q` — the same
predicate, since `s` is invertible on the checked range. What the reshaping
buys is that every scalar the ladders consume (`e`, `r`, `s`) arrives as wire
bytes, so no arithmetic ever happens modulo `n` on the device — which matters
because a 256-bit scalar-field element has no integer lane to read its bits
back from. The candidate `R'` is rebuilt from `r` by a square root (both
curves have `p ≡ 3 mod 4`), and `±R'` fold into one ladder because
`s·(-R') = -(s·R')`. The tests hold this form to SEC 1's own, case by case.

A scalar wider than `n` is left unreduced on purpose: `k·P = (k mod n)·P`,
so the group performs the standard's `mod n` itself (`weierstrass.scalar_mul`).

## The two paths

Verification is batch-first and namespace-generic — one traced computation
over the whole batch, per the seam. Key generation and signing are concrete:
Python integers for the scalar arithmetic (exact, and host-only per
`docs/reference/security.md`), the shared group law for the point work, and
RFC 6979 for the nonce, so signing is deterministic and reproducible against
the published vectors.

Public-key recovery (SEC 1 §4.1.6) is batch-first like verification but
host-path like signing, for a reason the reshaping above cannot remove: its
output is an encoding, and reading coordinates back into bytes is the one
thing a traced value cannot do (`group.to_affine_ints`). The scalar side —
`r⁻¹ mod n`, and a ladder driven by its bits — shares the constraint. So the
per-entry integer work runs on the host, and the curve arithmetic stays the
substrate's, one stacked ladder over the whole batch. Recovery lives here
rather than in a chain variant because the curve defines it; Ethereum is
merely its loudest consumer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from frx.typing import ArrayLike

from sig_frx import context as context_rules
from sig_frx import hashes
from sig_frx.arrays import namespace
from sig_frx.classical import group, weierstrass
from sig_frx.classical.ecdsa import rfc6979
from sig_frx.signature import Signature


@dataclass(frozen=True)
class MessageHash:
    """The hash a variant pairs with the curve — both of its faces.

    `byte_hash` is a namespace dispatcher in the `sig_frx.hashes` shape,
    serving batched verification in whichever namespace the values arrive.
    `host_constructor` is the `hashlib`-style constructor the signing path
    uses for the message and for RFC 6979's HMAC — the RFC requires those two
    to be the same `H`, and the known-answer `k` values are what hold this
    record's two faces to one function.
    """

    byte_hash: Callable[..., Any]
    host_constructor: rfc6979.HashConstructor


# The FIPS 186-5 pairing both curves are deployed with.
SHA256 = MessageHash(byte_hash=hashes.sha256, host_constructor=hashlib.sha256)


def _nonzero(xnp: Any, data: Any) -> Any:
    """Whether big-endian bytes name a nonzero integer, elementwise."""
    return xnp.any(data != 0, axis=-1)


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

    curve: weierstrass.Curve
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
        seed_bytes, scalar = weierstrass.secret_scalar(self.curve, seed, "seed")
        x, y = weierstrass.host_multiple_of_g(self.curve, scalar)
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
        _, x = weierstrass.secret_scalar(self.curve, secret_key, "secret key")
        r, s, recovery_id = self._signature_scalars(x, message)
        signature = r.to_bytes(32, "big") + s.to_bytes(32, "big")
        return np.frombuffer(signature, dtype=np.uint8).copy(), recovery_id

    def _signature_scalars(self, x: int, message: ArrayLike) -> tuple[int, int, int]:
        """One RFC 6979 signature as integers: `(r, s, recovery_id)`.

        The recovery id is two bits off the nonce's point, read before it is
        thrown away — bit 0 is `y(R)`'s parity, bit 1 is `x(R) >= n` (SEC 1
        §4.1.6's second candidate pair). The low-S flip replaces `s` with
        `n - s`, which verifies against `-R` instead of `R`, so the parity
        bit flips with it — the round-trip tests hold the two together.
        """
        n = self.curve.n
        h1 = self.hash.host_constructor(
            np.asarray(message, dtype=np.uint8).tobytes()
        ).digest()
        e = rfc6979.bits2int(h1, n.bit_length()) % n
        for k in rfc6979.nonces(n, x, h1, self.hash.host_constructor):
            rx, ry = weierstrass.host_multiple_of_g(self.curve, k)
            r = rx % n
            if r == 0:  # SEC 1 §4.1.3 discards the draw and takes the next
                continue
            s = pow(k, -1, n) * (e + r * x) % n
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
        """The batched verdict, `bool[B]` — one traced computation, no scalar path."""
        context_rules.require_empty(context, "ECDSA")
        curve = self.curve
        xnp = namespace(public_key, message, signature)
        public_key = xnp.asarray(public_key)
        signature = xnp.asarray(signature)

        digest = self.hash.byte_hash(public_key, message, signature).digest(message)
        z_bits = group.bits_of(digest)[..., :256]

        # SEC 1 §2.3.4: an uncompressed point is `04 ‖ X ‖ Y` with both
        # coordinates in [0, p-1], and the point must satisfy the curve
        # equation. Cofactor 1 makes that the whole subgroup check.
        qx_bytes = public_key[..., 1:33]
        qy_bytes = public_key[..., 33:65]
        qx = weierstrass.field_from_bytes(curve, qx_bytes)
        qy = weierstrass.field_from_bytes(curve, qy_bytes)
        key_ok = (
            (public_key[..., 0] == np.uint8(4))
            & group.bytes_below(xnp, qx_bytes, curve.p, byteorder="big")
            & group.bytes_below(xnp, qy_bytes, curve.p, byteorder="big")
            & weierstrass.on_curve(curve, qx, qy)
        )

        # SEC 1 §4.1.4 requires r and s in [1, n-1] before anything else runs.
        r_bytes = signature[..., :32]
        s_bytes = signature[..., 32:64]
        range_ok = (
            _nonzero(xnp, r_bytes)
            & group.bytes_below(xnp, r_bytes, curve.n, byteorder="big")
            & _nonzero(xnp, s_bytes)
            & group.bytes_below(xnp, s_bytes, curve.n, byteorder="big")
        )

        # The two x-candidates: x(R) mod n = r also admits x(R) = r + n,
        # reachable only while r + n < p — a sliver of size p - n on either
        # curve. Stacked on a leading axis so one square-root chain lifts
        # both; a junk root (x not on the curve) fails lift_x's own square
        # check, so an invalid candidate drops out arithmetically.
        first_x = weierstrass.field_from_bytes(curve, r_bytes)
        candidate_x = xnp.stack(
            [first_x, first_x + np.array(curve.n % curve.p, dtype=curve.field)]
        )
        root, is_point = weierstrass.lift_x(curve, candidate_x)
        lifted = weierstrass.from_affine(curve, candidate_x, root)

        # The four scalar multiples — z·G, r·Q, s·R'₁, s·R'₂ — have no data
        # dependencies, so they ride one stacked ladder call instead of four:
        # same 256 steps, a quarter of the dispatches (and a quarter of the
        # traced graph).
        s_bits = group.bits_of(s_bytes)
        generator = weierstrass.Point(
            *(xnp.broadcast_to(c, first_x.shape) for c in curve.generator)
        )
        point_q = weierstrass.from_affine(curve, qx, qy)
        bases = weierstrass.Point(
            *(
                xnp.stack([g, q, lifted_coord[0], lifted_coord[1]])
                for g, q, lifted_coord in zip(generator, point_q, lifted)
            )
        )
        multiples = weierstrass.scalar_mul(
            curve,
            xnp.stack([z_bits, group.bits_of(r_bytes), s_bits, s_bits]),
            bases,
        )
        target = weierstrass.add(
            curve,
            weierstrass.Point(*(c[0] for c in multiples)),
            weierstrass.Point(*(c[1] for c in multiples)),
        )
        s_multiple = weierstrass.Point(*(c[2:] for c in multiples))
        matches = group.equal(s_multiple, target) | group.equal(
            s_multiple, weierstrass.negate(target)
        )
        found_per_candidate = is_point & matches
        wraps = group.bytes_below(xnp, r_bytes, curve.p - curve.n, byteorder="big")
        found = found_per_candidate[0] | (wraps & found_per_candidate[1])

        return key_ok & range_ok & found

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
        whether `x(R)` wrapped past `n` — the same sliver `verify`'s second
        candidate covers from the other side. Batch-first like verification,
        host-path like signing (the module docstring owns why there is no
        traced form).

        A per-entry failure — `r` or `s` out of range, an id naming no curve
        point, the identity result — clears that entry's verdict and zeroes
        its key row. Rejection is a verdict rather than an exception because
        signatures and ids are wire data: one bad entry must not take down
        the batch, and a zeroed row cannot be mistaken for a key.
        """
        context_rules.require_empty(context, "ECDSA")
        message = np.asarray(message, dtype=np.uint8)
        signature = np.asarray(signature, dtype=np.uint8)
        recovery_id = np.asarray(recovery_id)
        if signature.shape[-1] != 64:
            raise ValueError("a signature is 64 bytes of r ‖ s")
        if not message.shape[0] == signature.shape[0] == recovery_id.shape[0]:
            raise ValueError("the batch axes disagree")
        n = self.curve.n
        # On host values `byte_hash` dispatches to the record's host face —
        # the same function signing uses, already batched over the rows.
        digest = np.asarray(self.hash.byte_hash(message).digest(message))
        digest_scalars = [
            rfc6979.bits2int(entry.tobytes(), n.bit_length()) % n for entry in digest
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

        # R rebuilt from x through the substrate's own lift; the id's bit 0
        # is the parity that picks between the root and its negation.
        point_r, is_point = weierstrass.lift_x_to_parity(
            curve, np.array(xs, dtype=curve.field), np.array(ids) & 1
        )
        ok &= np.asarray(is_point)

        # Q = r⁻¹(sR - eG), taken as u₁G + u₂R with u₁ = -e/r, u₂ = s/r
        # (mod n): exact host arithmetic for the scalars, one stacked ladder
        # for the points — the same shape verification's four multiples ride.
        u1_scalars, u2_scalars = [], []
        for e, r, s, valid in zip(digest_scalars, r_scalars, s_scalars, ok):
            r_inverse = pow(r, -1, n) if valid else 1
            u1_scalars.append(-e * r_inverse % n)
            u2_scalars.append(s * r_inverse % n)

        batch = len(ids)
        generator = weierstrass.generator_at(curve, batch)
        bases = weierstrass.Point(
            *(np.stack([g, rc]) for g, rc in zip(generator, point_r))
        )
        multiples = weierstrass.scalar_mul(
            curve,
            np.stack([group.ints_bits(u1_scalars), group.ints_bits(u2_scalars)]),
            bases,
        )
        public = weierstrass.add(
            curve,
            weierstrass.Point(*(c[0] for c in multiples)),
            weierstrass.Point(*(c[1] for c in multiples)),
        )

        # The identity has no encoding, so it is a rejection, not a key
        # (§4.1.6 rejects it by name). The readback divides by Z, so
        # rejected rows are steered to G first and zeroed after.
        ok &= ~np.asarray(weierstrass.is_identity(curve, public))
        flag = ok.astype(np.int32).astype(curve.field)
        readable = group.select(curve, flag, public, generator)
        keys = np.zeros((batch, 65), dtype=np.uint8)
        for i, ((qx, qy), valid) in enumerate(zip(group.to_affine_ints(readable), ok)):
            if valid:
                keys[i, 0] = 4
                keys[i, 1:33] = np.frombuffer(qx.to_bytes(32, "big"), dtype=np.uint8)
                keys[i, 33:] = np.frombuffer(qy.to_bytes(32, "big"), dtype=np.uint8)
        return keys, ok


if TYPE_CHECKING:
    _: type[Signature] = Ecdsa
