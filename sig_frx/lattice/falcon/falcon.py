# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Falcon verification — §3.10's Algorithm 16, assembled over the modules beneath it.

Verification is the whole of Falcon that lives in `Z_q`. It hashes the salt and
the message to a challenge, decompresses `s2` out of the signature, recovers
`s1 = c − s2·h mod q`, and asks whether `‖(s1, s2)‖²` is inside the bound its
parameter set fixes. None of the floating-point machinery, none of the trapdoor
sampling, and — since the transform never leaves this file — none of the root or
ordering pins ML-DSA needs ([`arith.py`](arith.py)).

It is also, for most consumers, the only Falcon operation they will ever call,
which is why it lands ahead of the operations that produce what it checks.

**Key generation and signing are not here.** They work over `Q[x]/(x^n + 1)` in
the complex domain and are gated on a precision question that is still open
([#178](https://github.com/fractalyze/sig-frx/issues/178)); the seam methods
raise until [#26](https://github.com/fractalyze/sig-frx/issues/26) and
[#27](https://github.com/fractalyze/sig-frx/issues/27) land. The conformance pin
is carried anyway, so the day a body replaces a `raise` the seam is already the
thing it has to satisfy.

## Where the batch axis is, and what sits outside it

`verify` takes the whole batch and traces as one computation, as the seam
requires. Inside, the public keys are decoded and transformed **batch-first**,
and only the per-signature remainder is `vmap`ped.

That split is the one shape decision this file makes. `arith` deliberately
exposes no composed `a · b`, so a caller hoists `ntt(h)` and stays in the
transform domain; putting that transform inside the mapped body would undo the
hoist, because `vmap` lifts `h` onto the batch axis and nothing downstream can
common up rows that are real data. With `B` distinct keys the two cost the same
— the seam hands every entry its own — but the hoisted form is the one a public
key object can later carry `h_hat` on, and the one a deployment verifying many
signatures under a *single* key can exploit. That deployment is ML-DSA's
[#23](https://github.com/fractalyze/sig-frx/issues/23) seen from Falcon, and it
is a surface below the seam rather than a shape change here.

## The norm does not fit the lane it is measured in

`‖(s1, s2)‖²` runs to `2n·(q/2)²`, which is 2^37 at Falcon-1024 against a 32-bit
integer lane ([`CLAUDE.md`](../../../CLAUDE.md)). The bound itself is small —
`⌊β²⌋` is under 2^27 — and that asymmetry is what `_within_bound` uses: the
running sum is compared against the bound at *every* prefix, so the comparison
that matters happens long before the accumulator could wrap. What makes the
prefixes trustworthy is the clamp ahead of them, which is the only part with a
constant to check: a coefficient at or above `⌊√⌊β²⌋⌋ + 1` already squares to
more than the whole bound allows, so clamping there cannot turn a rejection into
an acceptance and it keeps a single square inside 2^27.

The alternative is the reference implementation's: accumulate in `uint32` and
fold the sign bit of every partial into a saturating flag. Same guarantee, one
more thing to argue about, and it reads as a bit trick rather than as the
comparison the standard writes.

## Where the time goes

Measured on a workstation CPU at Falcon-1024, `B = 256`, warm, each stage timed
as the program it would be on its own, `verify` divides into `HashToPoint`
(69%), decompression (23%), and the key decode plus the ring product (4%).
Falcon-512 ranks the same way at the same shape and costs 0.52x of it, so
verification is very nearly linear in the degree.

The shape worth carrying forward is that **on CPU the challenge hash is the pole,
not the transform**. Falcon's `s1` recovery is one forward NTT and one inverse
over a single polynomial, where a SHAKE has to absorb the message and squeeze
`⌊2^16/q⌋`-rejected draws until `n` survive — 2720 bytes at Falcon-1024 against
`2n` bytes of useful output. That is why the arithmetic is the 4% and not the
pole. The decoder is the other stage worth reading, and `encoding.py` records
what its walk and its ranking cost at each granularity.

**The GPU leg does not rank them the same way, and that inversion is the more
useful half of the measurement.** On an RTX 5090 at Falcon-1024, `B` = 1024,
warm, each stage timed as the program it would be on its own and taken in the
same session as the `verify` it divides, `HashToPoint` is 31% of `verify` and
the decoder 69%. The same harness on the workstation CPU at that batch puts
`HashToPoint` at 70% and the decoder at 20%, on a `verify` costing 77x the
GPU's. So the challenge hash is what a CPU
verification spends its time on and the decoder is what a GPU one does, and
work aimed at either is work aimed at one leg. The batch is 1024 in both, rather
than the 256 the table above uses, because a share and its total have to come
from one session and this pair was taken in one.

These numbers compare implementations and size no budget.

## What leaks

Nothing a verifier holds is secret — a public key, a message and a signature are
all public — so `hash_to_point`'s rejection loop and the decoder's data-dependent
shape have nothing to leak, and the fixed budget both are written against is for
the tracer rather than for an attacker
([`security.md`](../../../docs/reference/security.md)).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from typing import TYPE_CHECKING, Any

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike
from hash_frx import SHAKE256_RATE

from sig_frx import context as context_rules
from sig_frx.batch import require_batch, require_no_position
from sig_frx.hashes import shake256
from sig_frx.lattice import rejection
from sig_frx.lattice.falcon import arith, encoding
from sig_frx.lattice.falcon.arith import Q
from sig_frx.signature import Signature

# Algorithm 3 line 1: `k = ⌊2^16/q⌋`, and a 16-bit draw is kept when it lands
# below `kq`. Keeping the rest would skew the challenge, since `2^16` is not a
# multiple of `q`.
_DRAW_BITS = 16
_HASH_ACCEPT = ((1 << _DRAW_BITS) // Q * Q, 1 << _DRAW_BITS)
_HASH_PER_BLOCK = SHAKE256_RATE // (_DRAW_BITS // 8)

# How many squared coefficients `_within_bound` folds before comparing. Any
# divisor of `2n` under `2^32 / (⌊√⌊β²⌋⌋ + 1)²` works — that ceiling is 61 at
# Falcon-1024 and 126 at Falcon-512 — and 32 divides both degrees' `2n`.
_NORM_BLOCK = 32


@dataclass(frozen=True)
class FalconParams:
    """One column of Falcon Table 3.3 — the four values a parameter set fixes.

    `q` is shared by both sets and lives in [`arith.py`](arith.py). The key and
    signature sizes are derived rather than transcribed, because §3.11 states
    them as expressions over `n` and a second copy is a second chance to mistype
    what the encoding already determines — the derivations are checked against
    Table 3.3's own 897 / 1793 / 666 / 1280 in the tests.
    """

    n: int  # ring degree, `deg(ϕ)`
    squared_norm_bound: int  # `⌊β²⌋`, the largest accepted `‖(s1, s2)‖²`
    signature_size: int  # `sbytelen`, the padded signature length

    def __post_init__(self) -> None:
        if self.n not in arith.DEGREES:
            raise ValueError(f"degree {self.n} is not a Falcon parameter set")
        # §3.11.3 gives the compressed coefficients whatever the header byte and
        # the salt leave, and `n` of them need nine bits each at minimum. A set
        # whose `sbytelen` cannot hold that would decode a prefix of its own
        # signatures — which is a falsifiable claim about the two numbers, where
        # comparing Algorithm 16 line 2's `328` against `8·(1 + SALT_SIZE)` is
        # the same constant twice (see `encoding.slen`).
        if encoding.slen(self.signature_size) < 9 * self.n:
            raise ValueError(
                f"sbytelen {self.signature_size} leaves "
                f"{encoding.slen(self.signature_size)} bits, under the {9 * self.n} "
                f"a degree-{self.n} signature needs at minimum"
            )

    @property
    def public_key_size(self) -> int:
        """`1 + ⌈14n/8⌉` — §3.11.4's header byte over 14-bit coefficients."""
        return 1 + -(-encoding.PK_BITS * self.n // 8)

    @property
    def secret_key_size(self) -> int:
        """`1 + n·(2·w + 8)/8` — §3.11.5's `f`, `g` at `w` bits and `F` at 8."""
        return 1 + self.n * (2 * encoding.SK_FG_BITS[self.n] + encoding.SK_F_BITS) // 8


# Table 3.3's two columns. The names are the standard's own, `Falcon-n` after the
# ring degree.
PARAMETER_SETS: dict[str, FalconParams] = {
    "Falcon-512": FalconParams(
        n=512,
        squared_norm_bound=34034726,
        signature_size=666,
    ),
    "Falcon-1024": FalconParams(
        n=1024,
        squared_norm_bound=70265242,
        signature_size=1280,
    ),
}


def hash_to_point(message: ArrayLike, n: int) -> Any:
    """Algorithm 3 — the challenge `c ∈ Z_q[x]/(ϕ)` from a byte string.

    Returns uint32 `[n]` residues. One message at a time; the batch axis is
    `verify`'s `vmap`, and the `[None, :]` here is the `B = 1` that `ByteHash`
    and [`rejection.first_accepted`](../rejection.py) both take.

    The standard's `while` squeezes until `n` draws have survived, which is a
    trip count no tracer has, so it is the fixed budget plus compaction both
    lattice schemes share. Falcon's draw is the widest and its acceptance the
    highest of the three: `⌊2^16/q⌋·q / 2^16` is about 0.938, against ML-DSA's
    0.998 for a nibble and 0.317 for a 23-bit residue.

    **Big-endian, and it is the one place in this scheme that is.** §3.9 says so
    in as many words — "the first of the b bits has numerical weight 2^(b−1)" —
    and it agrees with §3.11.1's byte order, so the two bytes of a draw read the
    way the encoder writes a field.
    """
    body = fnp.asarray(message, dtype=np.uint8)
    blocks = rejection.budget(n, _HASH_ACCEPT, _HASH_PER_BLOCK)
    stream = shake256(body)(blocks * SHAKE256_RATE).digest(body[None, :])
    pairs = stream.reshape(1, -1, _DRAW_BITS // 8).astype(np.uint32)
    draws = (pairs[..., 0] << np.uint32(8)) | pairs[..., 1]
    accepted = draws < np.uint32(_HASH_ACCEPT[0])
    return rejection.first_accepted(draws, accepted, n, "HashToPoint")[0] % np.uint32(Q)


def _within_bound(values: ArrayLike, bound: int) -> Any:
    """`Σ vᵢ² ≤ bound`, decided without a lane wide enough to hold the sum.

    Algorithm 16 line 6, and the reason it is not one `sum` is in the module
    docstring. The clamp is what keeps a single square inside 2^27 and cannot
    change a verdict; the running comparison is what catches the total before
    the accumulator could reach 2^32.
    """
    cap = isqrt(bound) + 1
    magnitude = fnp.minimum(
        fnp.abs(fnp.asarray(values, dtype=np.int32)), np.int32(cap)
    ).astype(np.uint32)
    # Zero-padded to a whole number of blocks. A zero square adds nothing to a
    # sum of squares, so this is an identity rather than a case to reason about,
    # and it keeps the helper independent of `2n` — `verify` always arrives at a
    # multiple, a test asking about five coefficients does not.
    squares = magnitude * magnitude
    tail = -squares.shape[-1] % _NORM_BLOCK
    if tail:
        squares = fnp.pad(squares, [(0, 0)] * (squares.ndim - 1) + [(0, tail)])
    # A plain reduction inside each block, and the running comparison only
    # across blocks: `2^32 / cap²` is 61 at Falcon-1024, so any 32 squares sum
    # without wrapping, and the first *block* prefix to pass `bound` is at most
    # `bound + 32·cap²`, still far under 2^32. Same guarantee as comparing every
    # element's prefix, and 3.4x cheaper than the full-length scan.
    blocked = squares.reshape(*squares.shape[:-1], -1, _NORM_BLOCK).sum(
        axis=-1, dtype=np.uint32
    )
    running = fnp.cumsum(blocked, axis=-1, dtype=np.uint32)
    return (running <= np.uint32(bound)).all(axis=-1)


class Falcon:
    """Falcon at one parameter set — §3.10's verification, and the seam around it.

    Build one with `named` rather than by hand unless a test wants parameters no
    published set has.
    """

    def __init__(self, params: FalconParams) -> None:
        self.params = params
        self.public_key_size = params.public_key_size
        self.secret_key_size = params.secret_key_size
        # Exact rather than an upper bound at this encoding: §3.11.3 pads every
        # signature to `sbytelen`. The seam's name carries the unpadded form the
        # spec allows a verifier to support, which this one does not.
        self.signature_max_size = params.signature_size
        # §3.9: the salt is drawn per signature, so signing is randomized. It is
        # a property of the scheme here rather than of the instance, since
        # Falcon defines no deterministic variant the way FIPS 204 does.
        self.deterministic = False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Falcon):
            return NotImplemented
        return self.params == other.params

    def __hash__(self) -> int:
        return hash((type(self), self.params))

    # -- key generation and signing ----------------------------------------

    def keygen(self, seed: ArrayLike) -> tuple[Array, Array]:
        """§3.8's `Keygen` — not yet implemented.

        The NTRU trapdoor and the ffLDL tree are
        [#26](https://github.com/fractalyze/sig-frx/issues/26), gated on the
        FFT's precision ([#178](https://github.com/fractalyze/sig-frx/issues/178)).
        """
        raise NotImplementedError(
            "Falcon key generation is sig-frx#26, gated on the FFT in #178; "
            "verification takes a public key in §3.11.4's encoding from elsewhere"
        )

    def sign(
        self,
        secret_key: ArrayLike,
        message: ArrayLike,
        *,
        randomness: ArrayLike | None = None,
        context: ArrayLike | None = None,
    ) -> Array:
        """§3.9's `Sign` — not yet implemented.

        ffSampling and the discrete Gaussian sampler are
        [#27](https://github.com/fractalyze/sig-frx/issues/27), gated on the
        same FFT.
        """
        raise NotImplementedError(
            "Falcon signing is sig-frx#27, gated on the FFT in #178; "
            "verification is independent of it and is implemented"
        )

    # -- verification ------------------------------------------------------

    def verify(
        self,
        public_key: ArrayLike,
        message: ArrayLike,
        signature: ArrayLike,
        *,
        context: ArrayLike | None = None,
        position: ArrayLike | None = None,
    ) -> Array:
        """Algorithm 16 over a whole batch: `bool[B]`.

        A public key or a signature of the wrong length verifies as false rather
        than raising, which is what §3.11's decoders do with a malformed input
        and what ML-DSA's `verify_internal` does for the same reason. A wrong
        *rank*, or a batch whose parts do not line up, stays an error: that is a
        caller mistake and not a verdict the standard defines.

        Falcon's standard has no application context, so one is refused rather
        than accepted and ignored — the seam's rule for RFC 8391 and ECDSA,
        which this joins.
        """
        require_no_position(position, "Falcon (FN-DSA)")
        context_rules.require_empty(context, "Falcon (FN-DSA)")
        params = self.params
        operands = require_batch(
            public_key,
            message,
            signature,
            public_key_size=params.public_key_size,
            signature_size=params.signature_size,
        )
        if not operands.well_formed:
            return fnp.zeros(operands.size, dtype=bool)

        # The hoist: decoded and transformed once for the batch, outside the
        # mapped body, so the transform is not re-entered per row.
        h, key_ok = encoding.pk_decode(operands.public_key, params.n)
        h_hat = arith.ntt(arith.to_field(h))
        return key_ok & frx.vmap(self._verify_one)(
            operands.message, operands.signature, h_hat
        )

    def _verify_one(self, message: Array, signature: Array, h_hat: Array) -> Array:
        """One entry of Algorithm 16, as the body `verify` maps.

        Written for one signature and mapped rather than transcribed a second
        time over a batch axis, which is ML-DSA's shape and for its reason: the
        decoder's scan and `searchsorted` are one-dimensional, and a second
        batch-shaped transcription would be a second thing to keep in agreement
        with the standard.
        """
        params = self.params
        # Lines 2-4. A malformed encoding is a verdict about the input, so it
        # lands in the same `bool` rather than raising.
        salt, s2, ok = encoding.sig_decode(signature, params.n, params.signature_size)
        # Line 1: the salt is what makes one message's signature unrepeatable,
        # and it is hashed ahead of the message exactly as `r ‖ m`.
        c = hash_to_point(fnp.concatenate([salt, message]), params.n)
        # Line 5. Only the product goes through the transform: `c` is already a
        # coefficient-domain polynomial, so subtracting it from a transform-
        # domain value and inverting the difference would return a different
        # polynomial — `intt` is linear, so nothing about the result looks
        # wrong. Two transforms rather than the three that hoisting `c` too
        # would cost.
        product = arith.intt(arith.base_mul(arith.ntt(arith.to_field(s2)), h_hat))
        s1 = arith.centered(arith.to_field(c) - product)
        # Line 6, over both halves at once — the bound is on the pair.
        within = _within_bound(fnp.concatenate([s1, s2]), params.squared_norm_bound)
        return ok & within


def named(name: str) -> Falcon:
    """The Table 3.3 parameter set called `name`, e.g. `Falcon-512`."""
    if name not in PARAMETER_SETS:
        raise ValueError(f"{name!r} is not one of {sorted(PARAMETER_SETS)}")
    return Falcon(PARAMETER_SETS[name])


if TYPE_CHECKING:
    # The seam conformance pin: mypy fails this module if it drifts from the
    # Protocol, rather than failing the consumer that calls it.
    _: type[Signature] = Falcon
