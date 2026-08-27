# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""BIP-340 Schnorr signatures over secp256k1, on the `Signature` seam.

Not an ECDSA variant: a different equation over the same curve, so it shares
the `secp` substrate — the curated zk_dtypes point kernels — and nothing
above it. The curve is fixed rather than injected: BIP-340 names secp256k1
and its constants (the tagged-hash strings, the even-y convention) together.
Keys are 32-byte x-only with the implicit even y; signing takes 32 bytes of
auxiliary randomness (the seam's `randomness`), which hedges the nonce
without breaking reproducibility — the published vectors fix it, so signing
is a function of its inputs.

Verification is the spec's own: `R = s·G - e·P`, reject unless `R` is a
point with even y whose x is `r`. The scalar work is Python integers, the
point work the curve's kernels, the parity and coordinate reads host
readbacks — one path, like the rest of the classical stack after
fractalyze/sig-frx#139.

## Two batch notions, on purpose

`verify` is the seam's: independent verdicts, `bool[B]`. `aggregate_verify`
is BIP-340's random-linear-combination check — one verdict for the whole
batch, sound because forging it means hitting a random linear relation. A
`False` names no entry; a caller that needs the culprit falls back to
`verify`. The aggregate form is why BIP-340 specifies batching at all: the
combined equation is one multi-scalar multiplication, and the MSM kernel it
wants is the same fractalyze/sig-frx#139 GPU story as everything else here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from frx.typing import ArrayLike

from sig_frx import context as context_rules
from sig_frx.classical import group, secp
from sig_frx.signature import Signature

_CURVE = secp.SECP256K1

# The tagged-hash prefixes: SHA256(tag) once, doubled at use (BIP-340's
# `SHA256(SHA256(tag) || SHA256(tag) || x)`).
_AUX = hashlib.sha256(b"BIP0340/aux").digest()
_NONCE = hashlib.sha256(b"BIP0340/nonce").digest()
_CHALLENGE = hashlib.sha256(b"BIP0340/challenge").digest()


def tagged(prefix: bytes, payload: bytes) -> bytes:
    """BIP-340's `SHA256(SHA256(tag) || SHA256(tag) || x)`, shared because
    BIP-327 incorporates it by reference rather than defining its own."""
    return hashlib.sha256(prefix + prefix + payload).digest()


def challenge(nonce: bytes, public_key: bytes, message: bytes) -> int:
    """BIP-340's `e`, reduced.

    Reduced here rather than by the callers so that no unreduced challenge can
    reach a field op — the same reason `_challenge_scalars` gives, now owed by
    one function rather than by each caller that spells the tagged hash out.
    BIP-327 derives its own challenge this way, which is why this is shared
    rather than private.
    """
    return (
        int.from_bytes(tagged(_CHALLENGE, nonce + public_key + message), "big")
        % _CURVE.n
    )


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
        _, secret = secp.secret_scalar(_CURVE, seed, "seed")
        x, _ = secp.host_multiple_of_g(_CURVE, secret)
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
        that catches a fault (a miscomputed kernel, a flipped bit) before a
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

        _, d0 = secp.secret_scalar(_CURVE, secret_key, "secret key")
        px, py = secp.host_multiple_of_g(_CURVE, d0)
        d = d0 if py % 2 == 0 else n - d0
        p_bytes = px.to_bytes(32, "big")

        mask = int.from_bytes(tagged(_AUX, aux.tobytes()), "big")
        t = (d ^ mask).to_bytes(32, "big")
        k0 = int.from_bytes(tagged(_NONCE, t + p_bytes + message_bytes), "big") % n
        if k0 == 0:  # the spec fails rather than redraws (a ~2^-256 draw)
            raise ValueError("the derived nonce is zero")
        rx, ry = secp.host_multiple_of_g(_CURVE, k0)
        k = k0 if ry % 2 == 0 else n - k0
        r_bytes = rx.to_bytes(32, "big")

        e = challenge(r_bytes, p_bytes, message_bytes)
        scalar = _CURVE.scalar
        signature = r_bytes + int(scalar(k) + scalar(e) * d).to_bytes(32, "big")
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
        position: ArrayLike | None = None,
    ) -> Any:
        """The seam's independent verdicts, `bool[B]` — the spec's algorithm,
        rejection for rejection."""
        context_rules.require_no_position(position, "BIP-340")
        context_rules.require_empty(context, "BIP-340")
        keys, messages, signatures, ok, key_points = self._parsed(
            public_key, message, signature
        )
        s_scalars = [
            int.from_bytes(entry[32:].tobytes(), "big") for entry in signatures
        ]
        e_scalars = self._challenge_scalars(keys, messages, signatures)
        r_scalars = [
            int.from_bytes(entry[:32].tobytes(), "big") for entry in signatures
        ]

        # R = s·G - e·P with even y and x = r: the spec's three rejections
        # (identity, odd y, x mismatch) ride the shared Schnorr readback,
        # the parity claim pinned even per the x-only convention.
        return secp.schnorr_verdicts(
            _CURVE,
            s_scalars,
            e_scalars,
            key_points,
            r_scalars,
            [0] * len(r_scalars),
            ok,
        )

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
        keys, messages, signatures, ok, key_points = self._parsed(
            public_key, message, signature
        )
        # The aggregate equation consumes R as a point, so each r lifts to
        # its even-y point — the same lift `verify` folds into a coordinate
        # comparison instead.
        r_ints = [
            int.from_bytes(entry[:32].tobytes(), "big") % _CURVE.p
            for entry in signatures
        ]
        r_points, r_lifted = secp.lift_x_to_parity(_CURVE, r_ints, [0] * len(r_ints))
        if not bool(np.all(ok & r_lifted)):
            return False

        n = _CURVE.n
        coefficients = group.batch_coefficients(
            n,
            keys.shape[0],
            keys.tobytes() + messages.tobytes() + signatures.tobytes(),
            digest=hashlib.sha256,
        )
        s_scalars = [
            int.from_bytes(entry[32:].tobytes(), "big") for entry in signatures
        ]
        e_scalars = self._challenge_scalars(keys, messages, signatures)

        scalar = _CURVE.scalar
        field_coefficients = [scalar(a) for a in coefficients]
        combined = int(sum(c * s for c, s in zip(field_coefficients, s_scalars)))
        lhs = secp.multiple(_CURVE, [combined], _CURVE.generator)
        terms = np.concatenate(
            [
                secp.multiple(_CURVE, coefficients, r_points),
                secp.multiple(
                    _CURVE,
                    [int(c * e) for c, e in zip(field_coefficients, e_scalars)],
                    key_points,
                ),
            ]
        )
        return bool(np.asarray(lhs == secp.sum_points(_CURVE, terms))[0])

    def _parsed(
        self, public_key: ArrayLike, message: ArrayLike, signature: ArrayLike
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Both verifiers' shared prologue: the wire arrays, the per-entry
        verdict so far, and the even-y lifted `P` points.

        The bounds run on the wire integers — `pk` and `r` below `p`, `s`
        below `n` (the spec's own rejections; vectors 12-14 pin them) — and
        the key lift's membership folds in. A failed row carries junk
        coordinates its cleared verdict drops.
        """
        keys = np.asarray(public_key, dtype=np.uint8)
        messages = np.asarray(message, dtype=np.uint8)
        signatures = np.asarray(signature, dtype=np.uint8)
        if keys.shape[-1] != 32 or signatures.shape[-1] != 64:
            raise ValueError("x-only keys are 32 bytes; signatures are 64")
        p, n = _CURVE.p, _CURVE.n
        pk_ints = [int.from_bytes(entry.tobytes(), "big") for entry in keys]
        checks = [
            pk < p
            and int.from_bytes(sig[:32].tobytes(), "big") < p
            and int.from_bytes(sig[32:].tobytes(), "big") < n
            for pk, sig in zip(pk_ints, signatures)
        ]
        key_points, on_curve = secp.lift_x_to_parity(
            _CURVE, [pk % p for pk in pk_ints], [0] * len(pk_ints)
        )
        ok = np.array(checks, dtype=bool) & on_curve
        return keys, messages, signatures, ok, key_points

    def _challenge_scalars(
        self, keys: np.ndarray, messages: np.ndarray, signatures: np.ndarray
    ) -> list[int]:
        """`e_i`: each tagged challenge digest, already reduced mod n.

        `challenge` reduces, so no unreduced value can reach a field op (the
        module gotcha in `secp.py`).
        """
        return [
            challenge(entry[:32].tobytes(), key.tobytes(), row.tobytes())
            for key, row, entry in zip(keys, messages, signatures)
        ]


if TYPE_CHECKING:
    _: type[Signature] = Bip340
