# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""FIPS 204's pseudorandom sampling (§7.3): ExpandA, ExpandS, ExpandMask, SampleInBall.

Everything random in ML-DSA is squeezed out of a SHAKE and rejection-sampled. The
standard writes each sampler as a `while` loop that squeezes until enough
candidates have survived, and that trip count is the data — so it is not
available to a tracer, and the loop is reshaped into a fixed candidate budget
plus a compaction. That reshaping is [`rejection.py`](../rejection.py)'s, shared
with Falcon's `HashToPoint`, and what is left here is the part FIPS 204 fixes:
which draws each sampler reads and which of them it keeps.

Which loop shape a rejection gets is the scheme's decision to record, and
[`ml-dsa.md`](../../../docs/schemes/ml-dsa.md) records this one alongside the
signing loop that went the other way.

## One-shot SHAKE, not the incremental one

`hash_frx.keccak.streaming` exists for exactly this scheme, and this does not
use it, which is worth saying plainly. An incremental sponge buys the ability to
squeeze without knowing the total in advance; a fixed budget knows the total by
construction. Against a static length the one-shot `KeccakSponge` is better on
every axis: it takes a `[B, L]` batch natively, so ExpandA's `k·ℓ` streams are
one call rather than a `vmap`, and its squeeze emits `ceil(n/rate)` blocks where
the incremental one emits `ceil((rate−1+n)/rate)` — a block more per stream,
because its offset is traced and a fixed budget's is not. §3.7's equivalence is
what makes the substitution exact: a series of Absorb/Squeeze calls yields what
one call on the concatenation yields.

None of which is an argument against the incremental sponge. It is what a
`while_loop` on the acceptance count needs, and that is the design this one is
weighed against — the budget is what makes the incremental surface unnecessary
here, not the other way round.

## Â is already in the standard's order

`RejNTTPoly` samples directly in `T_q`, and `arith.ntt` speaks FIPS 204's
ordering in and out, so the two pair index for index and nothing here permutes.
Adding a `BitRev8` gather would break it silently: a consistently wrong ordering
still round-trips and still convolves.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from frx.typing import ArrayLike
from hash_frx import SHAKE128_RATE, SHAKE256_RATE

from sig_frx.arrays import namespace
from sig_frx.hashes import shake128, shake256
from sig_frx.lattice.mldsa.arith import FIELD, N, Q
from sig_frx.lattice.mldsa.encoding import bytes_to_bits, unpack_fields
from sig_frx.lattice.rejection import budget, first_accepted, require_enough

# CoeffFromThreeBytes reads 23 uniform bits and keeps them when they land below
# `q` (Algorithm 14); the three bytes it reads divide `G`'s rate exactly.
_NTT_ACCEPT = (Q, 1 << 23)
_NTT_PER_BLOCK = SHAKE128_RATE // 3

# CoeffFromHalfByte (Algorithm 15) reads two candidates per byte and keeps a
# nibble below 15 at `η = 2` and below 9 at `η = 4`. The threshold is written
# once: `_rej_bounded_poly` rejects by it and `budget` is sized against it, and
# a divergence between those two is the one thing that could exhaust a budget
# the tail says is safe.
_BOUNDED_THRESHOLD = {2: 15, 4: 9}
_BOUNDED_ACCEPT = {
    eta: (threshold, 16) for eta, threshold in _BOUNDED_THRESHOLD.items()
}
_BOUNDED_PER_BLOCK = 2 * SHAKE256_RATE


def _seeds(rho: ArrayLike, width: int, tails: np.ndarray) -> Any:
    """`rho` repeated once per stream, each row carrying its own index bytes.

    The seed is left in the namespace it arrives in — a host value for key
    generation, a tracer for verification — and the sponge is picked to match
    ([`hashes.py`](../../hashes.py)) rather than lifting it. Hashing is not the
    lift `arith.ntt` makes: `frx.lax.ntt` has no host implementation and a SHAKE
    has one ([`conventions.md`](../../../docs/reference/conventions.md)).
    """
    xnp = namespace(rho)
    seed = xnp.asarray(rho, dtype=np.uint8).reshape(-1)
    if seed.shape[0] != width:
        raise ValueError(f"seed must be {width} bytes, got {seed.shape[0]}")
    repeated = xnp.broadcast_to(seed, (tails.shape[0], width))
    return xnp.concatenate([repeated, xnp.asarray(tails)], axis=-1)


def _integer_to_bytes(x: int, alpha: int) -> list[int]:
    """Algorithm 11 — base-256, little-endian, on the host where the index lives."""
    return [(x >> (8 * i)) & 0xFF for i in range(alpha)]


def _rej_ntt_poly(seeds: Any, blocks: int) -> Any:
    """Algorithm 30 over a batch of streams: uint8 `[B, 34]` -> `FIELD[B, 256]`.

    `G`'s 168-byte rate is 56 whole three-byte candidates, so the budget divides
    into candidates without a remainder and no block boundary splits one.
    """
    stream = shake128(seeds)(blocks * SHAKE128_RATE).digest(seeds)
    groups = stream.reshape(stream.shape[0], -1, 3).astype(fnp.uint32)
    # Algorithm 14, over the whole stream at once: the top bit of `b2` is
    # dropped, and the rest is a little-endian 23-bit integer.
    z = groups[..., 0] | (groups[..., 1] << 8) | ((groups[..., 2] & 0x7F) << 16)
    accepted = z < np.uint32(Q)
    return first_accepted(z, accepted, N, "ExpandA").astype(FIELD)


def _rej_bounded_poly(seeds: Any, eta: int, blocks: int) -> Any:
    """Algorithm 31 over a batch of streams: uint8 `[B, 66]` -> int32 `[B, 256]`."""
    stream = shake256(seeds)(blocks * SHAKE256_RATE).digest(seeds).astype(np.int32)
    # `z mod 16` then `⌊z/16⌋`, which is the order the standard consumes them in.
    nibbles = fnp.stack([stream & 0xF, stream >> 4], axis=-1).reshape(
        stream.shape[0], -1
    )
    # Algorithm 15. Both of its rules are evaluated over the whole stream and the
    # rejected slots are dropped by the compaction rather than by a branch.
    accepted = nibbles < np.int32(_BOUNDED_THRESHOLD[eta])
    if eta == 2:
        coefficients = np.int32(2) - nibbles % np.int32(5)
    else:
        coefficients = np.int32(4) - nibbles
    return first_accepted(coefficients, accepted, N, "ExpandS")


def expand_a(rho: ArrayLike, k: int, ell: int) -> Any:
    """Algorithm 32 — the public matrix `Â ∈ (T_q)^{k×ℓ}` from a 32-byte seed.

    Returns `FIELD[k, ell, 256]`, already in the NTT domain and already in FIPS
    204's ordering, so it pairs with `arith.base_mul` and `arith.ntt` directly.

    The `k·ℓ` entries are independent streams that differ only in the two index
    bytes appended to `rho`, so they are one batched sponge call and one batched
    compaction — there is no Python loop over entries, and the matrix shape is
    recovered by a reshape at the end.
    """
    tails = np.array(
        [
            _integer_to_bytes(s, 1) + _integer_to_bytes(r, 1)
            for r in range(k)
            for s in range(ell)
        ],
        dtype=np.uint8,
    )
    blocks = budget(N, _NTT_ACCEPT, _NTT_PER_BLOCK)
    hats = _rej_ntt_poly(_seeds(rho, 32, tails), blocks)
    return hats.reshape(k, ell, N)


def expand_s(rho: ArrayLike, k: int, ell: int, eta: int) -> tuple[Any, Any]:
    """Algorithm 33 — the secret vectors `(s1, s2)` from a 64-byte seed.

    Returns int32 `[ell, 256]` and `[k, 256]` with coefficients in `[−η, η]`,
    which is the standard's own range and not a residue: the conversion into
    `FIELD` belongs to the scheme step that multiplies with them.

    `s1` and `s2` differ only by where their index bytes start, so both vectors
    are sampled as one batch of `ℓ + k` streams and split on the way out.
    """
    if eta not in _BOUNDED_ACCEPT:
        raise ValueError(f"eta ({eta}) must be 2 or 4 — FIPS 204 Table 1")
    tails = np.array([_integer_to_bytes(r, 2) for r in range(ell + k)], dtype=np.uint8)
    blocks = budget(N, _BOUNDED_ACCEPT[eta], _BOUNDED_PER_BLOCK)
    coefficients = _rej_bounded_poly(_seeds(rho, 64, tails), eta, blocks)
    return coefficients[:ell], coefficients[ell:]


def expand_mask(rho: ArrayLike, mu: int, ell: int, gamma1: int) -> Any:
    """Algorithm 34 — the masking vector `y ∈ R^ℓ` from a 64-byte seed and `μ`.

    Returns int32 `[ell, 256]` with coefficients in `[−γ1 + 1, γ1]`. No rejection
    happens here: `H` is squeezed to exactly the length the unpacking consumes,
    which is what makes this the one sampler with no budget to argue about.

    `μ` is the signing loop's counter and stays a host integer, since the loop
    that increments it is the host's
    ([`conventions.md`](../../../docs/reference/conventions.md)).

    The bit-field read is `encoding.unpack_fields` — Algorithm 19's unsigned
    half, which this was the first consumer of and the wire formats are the
    second, so it lives with them now.
    """
    if gamma1 & (gamma1 - 1):
        raise ValueError(f"gamma1 ({gamma1}) must be a power of 2 — FIPS 204 Table 1")
    width = 1 + (gamma1 - 1).bit_length()
    tails = np.array([_integer_to_bytes(mu + r, 2) for r in range(ell)], dtype=np.uint8)
    seeds = _seeds(rho, 64, tails)
    v = shake256(seeds)(32 * width).digest(seeds)
    return np.int32(gamma1) - unpack_fields(v, width).astype(np.int32)


def sample_in_ball(rho: ArrayLike, tau: int) -> Any:
    """Algorithm 29 — the challenge `c ∈ R`, `tau` coefficients of ±1 and the rest 0.

    Returns int32 `[256]`.

    This is the one sampler whose rejection is *sequential*: the `τ` steps run in
    order, each consuming as many bytes as it takes to find one below its own
    bound, and a later step cannot be sampled before an earlier one has finished
    consuming. So it is `τ` unrolled steps — a static count, `τ` being a
    parameter — over one squeezed byte string, each step selecting the first
    still-unread byte that its bound admits.

    Selection and the swap are both `where`, never an index: `c[j]` is read as a
    masked sum and written as a masked select, so a traced `j` addresses nothing.
    That also keeps the two lines of the standard's swap in their order, which
    matters only when `i = j` — where the second write must win, and does.

    Those steps are compiled — `_ball_from_stream`, which is where the reason
    is. The squeeze stays out here, so the sponge is still chosen by the seed's
    namespace as everything else is.
    """
    seed = namespace(rho).asarray(rho, dtype=np.uint8).reshape(-1)
    # A step's byte is accepted when it is at most `i`, and `i` is smallest at
    # the first step — so every step accepts at least as often as `(257−τ)/256`,
    # and one budget sized at that rate covers all of them.
    allowance = budget(tau, (N - tau + 1, N), 1)
    stream = shake256(seed)(8 + allowance).digest(seed[None, :])[0]
    c, completed = _ball_from_stream(stream, tau)
    # Outside the traced region deliberately, so that the concrete path keeps the
    # check: `_ball_from_stream` returns a count rather than consuming it, and a
    # count that has come back out of `jit` is a number wherever the caller is
    # not itself traced.
    require_enough(completed, tau, "SampleInBall")
    return c


@partial(frx.jit, static_argnums=(1,))
def _ball_from_stream(stream: Any, tau: int) -> tuple[Any, Any]:
    """The `τ` steps of Algorithm 29 over a squeezed byte string, as one program.

    **This is the one sampler that is compiled, and the reason is its shape.**
    The others are a batched squeeze and a compaction — a handful of array
    operations whose cost is the work they name. This one is `τ` *sequential*
    steps of half a dozen elementwise operations each on a few hundred bytes, so
    eagerly it is several hundred dispatches doing almost nothing, and what it
    costs is the dispatching. Compiling collapses them into one program: measured
    at ML-DSA-65, 6.80 ms eager against 0.12 ms warm, for a compile of about 0.8 s
    per `(τ, stream length)` that a caller pays once and every later signature
    reuses.

    Compiling the batched samplers instead makes them worse, which is why none of
    them is: their squeeze is the bulk of what they do, and tracing it puts the
    sponge under a tracer — where [`hashes`](../../hashes.py) must select the
    device row, giving back what selecting the host one just bought.

    `τ` is static because it is the unrolling, and the count comes back out
    rather than being checked in here: a comparison on it cannot raise under a
    tracer, so `sample_in_ball` keeps that check where it can still run.

    The allowance is read off the stream rather than recomputed from `τ`. It is
    the same number either way, and taking it from the bytes that were actually
    squeezed means the two cannot disagree — which they otherwise could, since a
    compiled body is reused for every call whose stream is the same length.
    """
    allowance = stream.shape[0] - 8
    signs = bytes_to_bits(stream[:8]).astype(np.int32)
    candidates = stream[8:].astype(np.int32)
    position = np.arange(allowance, dtype=np.int32)
    slot = np.arange(N, dtype=np.int32)
    # Position and byte in one sortable key, so that "the first admissible byte"
    # is a single minimum rather than a running count and two masked sums. The
    # byte is recovered from the low half because it is what the key ends in.
    key = position * np.int32(N) + candidates
    exhausted = np.int32(allowance * N + N)

    c = fnp.zeros(N, dtype=np.int32)
    cursor = fnp.zeros((), dtype=np.int32)
    completed = fnp.zeros((), dtype=np.int32)
    for step in range(tau):
        i = N - tau + step
        available = (position >= cursor) & (candidates <= np.int32(i))
        first = fnp.min(fnp.where(available, key, exhausted))
        found = first < exhausted
        completed = completed + found.astype(np.int32)
        j = first % np.int32(N)
        cursor = fnp.where(found, first // np.int32(N) + np.int32(1), cursor)
        # `c[i] ← c[j]` then `c[j] ← (−1)^h[i+τ−256]`, in that order.
        at_j = slot == j
        c = fnp.where(slot == i, fnp.sum(fnp.where(at_j, c, np.int32(0))), c)
        c = fnp.where(at_j, np.int32(1) - np.int32(2) * signs[step], c)
    return c, completed
