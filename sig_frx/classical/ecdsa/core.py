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
        seed_bytes, scalar = self._secret_scalar(seed, "seed")
        x, y = self._host_multiple_of_g(scalar)
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
        del randomness  # deterministic (RFC 6979); the seam documents this.
        context_rules.require_empty(context, "ECDSA")
        _, x = self._secret_scalar(secret_key, "secret key")
        n = self.curve.n

        h1 = self.hash.host_constructor(
            np.asarray(message, dtype=np.uint8).tobytes()
        ).digest()
        e = rfc6979.bits2int(h1, n.bit_length()) % n
        for k in rfc6979.nonces(n, x, h1, self.hash.host_constructor):
            rx, _ = self._host_multiple_of_g(k)
            r = rx % n
            if r == 0:  # SEC 1 §4.1.3 discards the draw and takes the next
                continue
            s = pow(k, -1, n) * (e + r * x) % n
            if s == 0:
                continue
            if self.low_s and s > n // 2:
                s = n - s
            signature = r.to_bytes(32, "big") + s.to_bytes(32, "big")
            return np.frombuffer(signature, dtype=np.uint8).copy()
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

    def _secret_scalar(self, data: ArrayLike, role: str) -> tuple[Any, int]:
        """The 32-byte encoding as `(bytes, scalar)`, refused outside [1, n-1].

        Out-of-range is refused rather than reduced — reduction would silently
        map two encodings to one key (SEC 1 §3.2.1's validity range).
        """
        raw = np.asarray(data, dtype=np.uint8).reshape(-1)
        if raw.shape[0] != self.secret_key_size:
            raise ValueError(f"a {role} is {self.secret_key_size} bytes")
        scalar = int.from_bytes(raw.tobytes(), "big")
        if not 1 <= scalar <= self.curve.n - 1:
            raise ValueError(f"the {role} scalar is outside [1, n-1]")
        return raw, scalar

    def _host_multiple_of_g(self, scalar: int) -> tuple[int, int]:
        """`scalar·G` as affine integers, on the host path.

        The same ladder verification traces, run concretely at `B = 1` — one
        group-law implementation, per the substrate's contract.
        """
        point = weierstrass.scalar_mul(
            self.curve, group.int_bits(scalar), self.curve.generator
        )
        ((x, y),) = group.to_affine_ints(point)
        return x, y


if TYPE_CHECKING:
    _: type[Signature] = Ecdsa
