# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""BIP-340 Schnorr signatures over secp256k1, on the `Signature` seam.

Not an ECDSA variant: a different equation over the same curve, so it shares
the Weierstrass substrate and nothing above it. The curve is fixed rather
than injected — BIP-340 names secp256k1 and its constants (the tagged-hash
strings, the even-y convention) with it. Keys are 32-byte x-only with the
implicit even y; signing takes 32 bytes of auxiliary randomness (the seam's
`randomness`), which hedges the nonce without breaking reproducibility — the
published vectors fix it, so signing is a function of its inputs.

## Verification without a parity read of the computed point

The spec computes `R = s·G - e·P` and rejects unless `y(R)` is even and
`x(R) = r`. Testing the parity of a computed point needs its canonical bytes
— a readback. Rebuilt the other way, the same predicate is `R' = lift_x(r)`
with even y, accept iff `s·G - e·P = R'` exactly: the even-y rejection
becomes the equation missing (the odd candidate is `-R'`), and an infinite
`R` can never equal a lifted point, so both of the spec's rejections fall
out arithmetically. What stays host-side is the parity pick inside the one
stacked lift of the key and r columns (`weierstrass.lift_x_to_parity`, over
wire values rather than computed points) — which is why the scheme is
host-path, like everything else these curves' traced blocker gates — while
the curve work rides one stacked ladder per batch.

## Two batch notions, on purpose

`verify` is the seam's: independent verdicts, `bool[B]`. `aggregate_verify`
is BIP-340's random-linear-combination check — one verdict for the whole
batch, sound because forging it means hitting a random linear relation. A
`False` names no entry; a caller that needs the culprit falls back to
`verify`. The aggregate form is why BIP-340 specifies batching at all: the
combined equation is one multi-scalar multiplication instead of two ladders
per entry. This implementation states that equation over the shared ladder
and leaves the MSM speedup to the substrate, the same way `weierstrass.py`
leaves formula costs to fractalyze/sig-frx#36.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from frx.typing import ArrayLike

from sig_frx import context as context_rules
from sig_frx.classical import group, weierstrass
from sig_frx.signature import Signature

_CURVE = weierstrass.SECP256K1

# The tagged-hash prefixes: SHA256(tag) once, doubled at use (BIP-340's
# `SHA256(SHA256(tag) || SHA256(tag) || x)`).
_AUX = hashlib.sha256(b"BIP0340/aux").digest()
_NONCE = hashlib.sha256(b"BIP0340/nonce").digest()
_CHALLENGE = hashlib.sha256(b"BIP0340/challenge").digest()


def _tagged(prefix: bytes, payload: bytes) -> bytes:
    return hashlib.sha256(prefix + prefix + payload).digest()


@dataclass(frozen=True)
class Bip340:
    """BIP-340 on the seam: x-only keys, tagged hashes, hedged deterministic sign.

    `randomness` is the spec's 32-byte auxiliary input `a` and is required —
    the spec's own default signer takes it, and the all-zero value the
    vectors use is a caller's choice, not a default this scheme invents.
    `context` is required empty: BIP-340 defines none.
    """

    public_key_size = 32
    secret_key_size = 32
    signature_max_size = 64
    deterministic = False

    def keygen(self, seed: ArrayLike) -> tuple[Any, Any]:
        """The x-only public key of secret scalar `seed`, per the spec.

        The secret crosses unchanged: the even-y negation of `d` is signing's
        business (the spec applies it there), so a stored key never depends
        on which parity its point happened to draw.
        """
        _, secret = weierstrass.secret_scalar(_CURVE, seed, "seed")
        x, _ = weierstrass.host_multiple_of_g(_CURVE, secret)
        public = np.frombuffer(x.to_bytes(32, "big"), dtype=np.uint8).copy()
        return public, np.asarray(seed, dtype=np.uint8).reshape(-1).copy()

    def sign(
        self,
        secret_key: ArrayLike,
        message: ArrayLike,
        *,
        randomness: ArrayLike | None,
        context: ArrayLike | None,
    ) -> Any:
        """One signature, `bytes(R) ‖ bytes(s)` — the spec's default signer.

        Ends by verifying its own output, as the spec instructs — the check
        that catches a fault (a miscomputed ladder, a flipped bit) before a
        secret-dependent signature leaves the process.
        """
        context_rules.require_empty(context, "BIP-340")
        if randomness is None:
            raise ValueError("BIP-340 signing takes 32 bytes of auxiliary data")
        aux = np.asarray(randomness, dtype=np.uint8).reshape(-1)
        if aux.shape[0] != 32:
            raise ValueError("BIP-340 signing takes 32 bytes of auxiliary data")
        n = _CURVE.n
        message_bytes = np.asarray(message, dtype=np.uint8).tobytes()

        _, d0 = weierstrass.secret_scalar(_CURVE, secret_key, "secret key")
        px, py = weierstrass.host_multiple_of_g(_CURVE, d0)
        d = d0 if py % 2 == 0 else n - d0
        p_bytes = px.to_bytes(32, "big")

        mask = int.from_bytes(_tagged(_AUX, aux.tobytes()), "big")
        t = (d ^ mask).to_bytes(32, "big")
        k0 = int.from_bytes(_tagged(_NONCE, t + p_bytes + message_bytes), "big") % n
        if k0 == 0:  # the spec fails rather than redraws (a ~2^-256 draw)
            raise ValueError("the derived nonce is zero")
        rx, ry = weierstrass.host_multiple_of_g(_CURVE, k0)
        k = k0 if ry % 2 == 0 else n - k0
        r_bytes = rx.to_bytes(32, "big")

        e = (
            int.from_bytes(
                _tagged(_CHALLENGE, r_bytes + p_bytes + message_bytes), "big"
            )
            % n
        )
        signature = r_bytes + ((k + e * d) % n).to_bytes(32, "big")
        result = np.frombuffer(signature, dtype=np.uint8).copy()
        verified = self.verify(
            np.frombuffer(p_bytes, dtype=np.uint8)[None],
            np.asarray(message, dtype=np.uint8).reshape(1, -1),
            result[None],
            context=None,
        )
        if not bool(np.asarray(verified)[0]):
            raise AssertionError("the produced signature failed its own check")
        return result

    def verify(
        self,
        public_key: ArrayLike,
        message: ArrayLike,
        signature: ArrayLike,
        *,
        context: ArrayLike | None,
    ) -> Any:
        """The seam's independent verdicts, `bool[B]`.

        The module docstring owns the reshaped predicate; per entry it is the
        spec's, rejection for rejection.
        """
        context_rules.require_empty(context, "BIP-340")
        keys, messages, signatures, ok, key_point, r_point = self._parsed(
            public_key, message, signature
        )
        challenges = self._challenge_bits(keys, messages, signatures)

        generator = weierstrass.generator_at(_CURVE, keys.shape[0])
        bases = weierstrass.Point(
            *(np.stack([g, q]) for g, q in zip(generator, key_point))
        )
        multiples = weierstrass.scalar_mul(
            _CURVE,
            np.stack([group.bits_of(signatures[..., 32:]), challenges]),
            bases,
        )
        lhs = weierstrass.add(
            _CURVE,
            weierstrass.Point(*(c[0] for c in multiples)),
            weierstrass.negate(weierstrass.Point(*(c[1] for c in multiples))),
        )
        return ok & group.equal(lhs, r_point)

    def aggregate_verify(
        self,
        public_key: ArrayLike,
        message: ArrayLike,
        signature: ArrayLike,
    ) -> bool:
        """One verdict for the whole batch — the spec's aggregate check.

        `False` names no entry, only that the combined equation fails; a
        caller that needs the culprit falls back to `verify`. The random
        coefficients are drawn deterministically from a hash of the entire
        batch: the spec asks for a CSPRNG seeded exactly that way and leaves
        the generator implementation-defined, and a derivation with no
        hidden draw keeps the verdict reproducible and the seam's
        no-implicit-randomness rule intact. A forged batch would need the
        combination to vanish over coefficients fixed only after every byte
        of it was committed.
        """
        keys, messages, signatures, ok, key_points, r_points = self._parsed(
            public_key, message, signature
        )
        if not bool(np.all(ok)):
            return False

        n = _CURVE.n
        batch = keys.shape[0]
        seed = hashlib.sha256(
            keys.tobytes() + messages.tobytes() + signatures.tobytes()
        ).digest()
        coefficients = [1] + [
            1
            + int.from_bytes(
                hashlib.sha256(seed + index.to_bytes(8, "big")).digest(), "big"
            )
            % (n - 1)
            for index in range(1, batch)
        ]
        s_scalars = [
            int.from_bytes(entry[32:].tobytes(), "big") for entry in signatures
        ]
        e_scalars = [
            int.from_bytes(digest, "big")
            for digest in self._challenge_digests(keys, messages, signatures)
        ]

        combined = sum(a * s for a, s in zip(coefficients, s_scalars)) % n
        # The s·G side rides the same stacked ladder as the 2B point
        # multiples — a separate call would pay the full 256-step ladder
        # again for its one lane.
        weights = np.concatenate(
            [
                group.ints_bits(coefficients),
                group.ints_bits([a * e % n for a, e in zip(coefficients, e_scalars)]),
                group.int_bits(combined),
            ]
        )
        bases = weierstrass.Point(
            *(
                np.concatenate([rc, pc, g])
                for rc, pc, g in zip(r_points, key_points, _CURVE.generator)
            )
        )
        multiples = weierstrass.scalar_mul(_CURVE, weights, bases)
        lhs = weierstrass.Point(*(c[-1:] for c in multiples))
        total = _sum_points(weierstrass.Point(*(c[:-1] for c in multiples)))
        return bool(np.asarray(group.equal(lhs, total))[0])

    def _parsed(
        self, public_key: ArrayLike, message: ArrayLike, signature: ArrayLike
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        weierstrass.Point,
        weierstrass.Point,
    ]:
        """Both verifiers' shared prologue: the wire arrays, the combined
        per-entry verdict, and the even-y lifted `P` and `R'` points.

        One stacked lift serves the key and r columns (the spec's `lift_x`
        is the parity-0 case), and the verdict already folds the bytes-level
        ranges with both memberships — an out-of-field x is checked on the
        bytes, where the field wrap would hide it.
        """
        keys = np.asarray(public_key, dtype=np.uint8)
        messages = np.asarray(message, dtype=np.uint8)
        signatures = np.asarray(signature, dtype=np.uint8)
        if keys.shape[-1] != 32 or signatures.shape[-1] != 64:
            raise ValueError("x-only keys are 32 bytes; signatures are 64")
        ok = (
            group.bytes_below(np, keys, _CURVE.p, byteorder="big")
            & group.bytes_below(np, signatures[..., :32], _CURVE.p, byteorder="big")
            & group.bytes_below(np, signatures[..., 32:], _CURVE.n, byteorder="big")
        )
        stacked = np.stack([keys, signatures[..., :32]])
        points, on_curve = weierstrass.lift_x_to_parity(
            _CURVE, weierstrass.field_from_bytes(_CURVE, stacked), 0
        )
        ok = ok & np.asarray(on_curve[0]) & np.asarray(on_curve[1])
        key_point = weierstrass.Point(*(c[0] for c in points))
        r_point = weierstrass.Point(*(c[1] for c in points))
        return keys, messages, signatures, ok, key_point, r_point

    def _challenge_digests(
        self, keys: np.ndarray, messages: np.ndarray, signatures: np.ndarray
    ) -> list[bytes]:
        """`e_i` as 32-byte tagged digests, reduced by the ladder or mod n."""
        return [
            _tagged(
                _CHALLENGE,
                entry[:32].tobytes() + key.tobytes() + row.tobytes(),
            )
            for key, row, entry in zip(keys, messages, signatures)
        ]

    def _challenge_bits(
        self, keys: np.ndarray, messages: np.ndarray, signatures: np.ndarray
    ) -> np.ndarray:
        """The challenges as `[B, 256]` ladder bits, reduced mod n first.

        The reduction is the spec's step, not a nicety: the ladder would
        perform it through the group, but the KAT holds `e` to the spec's
        integer, so the bytes fed here are `int(hash) mod n` re-encoded.
        """
        n = _CURVE.n
        return group.ints_bits(
            [
                int.from_bytes(digest, "big") % n
                for digest in self._challenge_digests(keys, messages, signatures)
            ]
        )


def _sum_points(points: weierstrass.Point) -> weierstrass.Point:
    """The sum of a `[K]` batch of points, by vectorized halving."""
    while points.x.shape[0] > 1:
        size = points.x.shape[0]
        if size % 2:
            filler = weierstrass.identity(_CURVE, points.x[:1])
            points = weierstrass.Point(
                *(np.concatenate([c, f]) for c, f in zip(points, filler))
            )
            size += 1
        half = size // 2
        points = weierstrass.add(
            _CURVE,
            weierstrass.Point(*(c[:half] for c in points)),
            weierstrass.Point(*(c[half:] for c in points)),
        )
    return points


if TYPE_CHECKING:
    _: type[Signature] = Bip340
