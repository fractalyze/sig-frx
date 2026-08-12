# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""FIPS 204's pseudorandom sampling (§7.3): ExpandA, ExpandS, ExpandMask, SampleInBall.

Everything random in ML-DSA is squeezed out of a SHAKE and rejection-sampled. The
standard writes each sampler as a `while` loop that squeezes until enough
candidates have survived, and that trip count is the data — so it is not
available to a tracer, and the loop is reshaped into a fixed candidate budget
plus a compaction. This module is that reshaping, and the two things it has to
get right are how large the budget is and how the survivors are collected.

## The budget is computed, not chosen

A generous static bound is only as good as the argument that it is generous
enough, and "5 blocks is what everyone uses" is not one — the implementations
that use 5 also loop when 5 is not enough, which is the part this cannot do. So
`_budget` computes the smallest budget whose *exact* binomial shortfall
probability is at most `2^-256`, the collision strength of the strongest
parameter set. It is integer arithmetic over the acceptance probability the
standard's own rejection rule defines, evaluated once per parameterisation on
the host, and it means the bound-exhausted path is unreachable rather than
merely unlikely (see `_require_enough` for what happens if it is reached anyway).

A fixed budget also settles the question `conventions.md` asks of every rejection
loop: how many blocks are squeezed and how many candidates are examined is the
same for every seed, so the trip count leaks nothing. That is a consequence of
the shape rather than a claim — this repo makes no side-channel claim
([`security.md`](../../../docs/reference/security.md)) — and it is the shape the
compiler wanted anyway.

## Collection is a gather, and the sort is the schedule

Keeping the accepted candidates and closing the gaps is a data-dependent
permutation however it is written. It is written as a stable sort on the
rejection flag: `False` sorts first and equal keys hold their order, so the
survivors land contiguously at the front in stream order, and reading them off
is one `take_along_axis`. The direction matters — the same permutation as a
scatter serialises on a GPU, where a gather vectorises.

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

## The batch axis

The samplers take one seed and return one matrix, vector, or polynomial, the way
FIPS 204 defines them; the axis inside each is over the streams that seed fans
out into, which is where ExpandA's `k·ℓ` independent entries live. A batch of
*signatures* is `frx.vmap` over these, per
[`conventions.md`](../../../docs/reference/conventions.md).

## Â is already in the standard's order

`RejNTTPoly` samples directly in `T_q`, and `arith.ntt` speaks FIPS 204's
ordering in and out, so the two pair index for index and nothing here permutes.
Adding a `BitRev8` gather would break it silently: a consistently wrong ordering
still round-trips and still convolves.
"""

from __future__ import annotations

from functools import lru_cache
from math import comb
from typing import Any

import frx.core
import frx.numpy as fnp
import numpy as np
from frx.typing import ArrayLike
from hash_frx.keccak.byte_hashes import (
    SHAKE128_RATE,
    SHAKE256_RATE,
    Shake128,
    Shake256,
)

from sig_frx.arrays import namespace
from sig_frx.lattice.mldsa.arith import FIELD, N, Q

# The shortfall probability every budget below is sized against. `2^-256` is the
# collision strength `λ` of the strongest parameter set (Table 1), so a budget
# that meets it is not the weakest part of any set — and the margin is cheap,
# since the tail falls off fast enough that buying it costs one further block.
_LOG2_SHORTFALL = 256

# CoeffFromThreeBytes reads 23 uniform bits and keeps them when they land below
# `q` (Algorithm 14); the three bytes it reads divide `G`'s rate exactly.
_NTT_ACCEPT = (Q, 1 << 23)
_NTT_PER_BLOCK = SHAKE128_RATE // 3

# CoeffFromHalfByte (Algorithm 15) reads two candidates per byte and keeps a
# nibble below 15 at `η = 2` and below 9 at `η = 4`.
_BOUNDED_ACCEPT = {2: (15, 16), 4: (9, 16)}
_BOUNDED_PER_BLOCK = 2 * SHAKE256_RATE


def _shortfall_exceeds_margin(
    trials: int, needed: int, accept: tuple[int, int]
) -> bool:
    """Whether `trials` candidates yield fewer than `needed` more often than `2^-256`.

    The exact binomial lower tail, cleared of its denominator: with acceptance
    `num/den`, `P = Σ_{a<needed} C(trials, a)·num^a·(den−num)^(trials−a) /
    den^trials`, and every factor is an integer. Exact rather than a Chernoff
    bound because it is a few hundred big-integer multiplications on the host,
    and because a safety argument that is itself approximate is one more thing a
    reader has to check.
    """
    num, den = accept
    shortfall = sum(
        comb(trials, survivors) * num**survivors * (den - num) ** (trials - survivors)
        for survivors in range(needed)
    )
    return (shortfall << _LOG2_SHORTFALL) > den**trials


@lru_cache(maxsize=None)
def _budget(needed: int, accept: tuple[int, int], per_block: int) -> int:
    """The fewest blocks of `per_block` candidates that safely yield `needed`.

    Cached because it is the same handful of parameterisations for the life of a
    process, and because the tail is big-integer work that no caller should pay
    twice.
    """
    blocks = max(1, -(-needed // per_block))
    while _shortfall_exceeds_margin(blocks * per_block, needed, accept):
        blocks += 1
    return blocks


def _require_enough(survivors: Any, needed: int, sampler: str) -> None:
    """Raise if any stream produced fewer than `needed` survivors.

    The check runs wherever it can be run. Key generation and signing are
    concrete, so the count is a number there and a shortfall raises; under a
    tracer it is not a number, no comparison on it can raise, and frx has no
    `checkify` to defer one to. What stands in for the check on that path is
    `_budget`: the count it sizes for cannot fall short more often than
    `2^-256`, which is below the collision strength of the scheme being sampled
    for. The alternative — returning a validity flag for every caller to thread
    and most to drop — trades a guarantee for a convention.
    """
    if isinstance(survivors, frx.core.Tracer):
        return
    short = np.atleast_1d(np.asarray(survivors) < needed)
    if short.any():
        raise RuntimeError(
            f"{sampler}: {int(short.sum())} of {short.size} streams ran out of "
            f"candidates before {needed} survived rejection. The budget is sized "
            f"for a shortfall probability below 2^-{_LOG2_SHORTFALL}, so this is "
            f"a wrong budget rather than an unlucky seed."
        )


def _first_accepted(values: Any, accepted: Any, count: int) -> Any:
    """The first `count` accepted values of each row, in stream order.

    A stable sort on the rejection flag is the compaction: `False` sorts first
    and equal keys keep their order, so a row's survivors arrive contiguously at
    the front in the order the stream produced them, and the prefix is the
    answer. One gather per row and no scatter — the same permutation written the
    other way round serialises on a GPU.
    """
    order = fnp.argsort(~accepted, axis=-1, stable=True)[..., :count]
    return fnp.take_along_axis(values, order, axis=-1)


def _seeds(rho: ArrayLike, width: int, tails: np.ndarray) -> Any:
    """`rho` repeated once per stream, each row carrying its own index bytes.

    The seed is left in the namespace it arrives in — a host value for key
    generation, a tracer for verification — right up to the sponge, which lifts
    it either way. Hashing is an opcode, so this is the same lift `arith.ntt`
    makes ([`conventions.md`](../../../docs/reference/conventions.md)).
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
    stream = Shake128(blocks * SHAKE128_RATE).digest(seeds)
    groups = stream.reshape(stream.shape[0], -1, 3).astype(fnp.uint32)
    # Algorithm 14, over the whole stream at once: the top bit of `b2` is
    # dropped, and the rest is a little-endian 23-bit integer.
    z = groups[..., 0] | (groups[..., 1] << 8) | ((groups[..., 2] & 0x7F) << 16)
    accepted = z < np.uint32(Q)
    _require_enough(accepted.sum(axis=-1), N, "ExpandA")
    return _first_accepted(z, accepted, N).astype(FIELD)


def _rej_bounded_poly(seeds: Any, eta: int, blocks: int) -> Any:
    """Algorithm 31 over a batch of streams: uint8 `[B, 66]` -> int32 `[B, 256]`."""
    stream = Shake256(blocks * SHAKE256_RATE).digest(seeds).astype(np.int32)
    # `z mod 16` then `⌊z/16⌋`, which is the order the standard consumes them in.
    nibbles = fnp.stack([stream & 0xF, stream >> 4], axis=-1)
    nibbles = nibbles.reshape(stream.shape[0], -1)
    # Algorithm 15. Both branches are evaluated over the whole stream and the
    # rejected slots are dropped by the compaction rather than by a branch.
    if eta == 2:
        accepted, coefficients = nibbles < 15, np.int32(2) - nibbles % np.int32(5)
    else:
        accepted, coefficients = nibbles < 9, np.int32(4) - nibbles
    _require_enough(accepted.sum(axis=-1), N, "ExpandS")
    return _first_accepted(coefficients, accepted, N)


def _bytes_to_bits(values: Any) -> Any:
    """Algorithm 13 — each byte low bit first, over the trailing axis."""
    bits = (values[..., None] >> np.arange(8, dtype=np.uint8)) & np.uint8(1)
    return bits.reshape(*values.shape[:-1], -1)


def _bit_unpack_field(v: Any, width: int) -> Any:
    """`v` read as 256 little-endian `width`-bit fields per row, as uint32.

    The unsigned half of Algorithm 19 — the subtraction its last line performs is
    the caller's, since only the caller knows `b`. Bit fields tile the byte
    string with period `lcm(width, 8)`, and that period holds a whole number of
    fields, so every field's window is a static slice and no field crosses a
    period boundary. The general BitPack/BitUnpack pair belongs with the wire
    formats that need both directions; this is the half ExpandMask needs, and it
    is the piece to hoist when that second consumer arrives rather than a seam
    guessed at now ([`conventions.md`](../../../docs/reference/conventions.md)).
    """
    period = int(np.lcm(width, 8)) // 8
    per_period = period * 8 // width
    # A field starting mid-byte reaches one byte further than its own width, and
    # the last field of a period reaches past the period, so each period carries
    # the padding that keeps its windows in range.
    span = -(-(width + 7) // 8)
    periods = v.reshape(v.shape[0], -1, period)
    padding = fnp.zeros((periods.shape[0], periods.shape[1], span - 1), dtype=fnp.uint8)
    padded = fnp.concatenate([periods, padding], axis=-1).astype(fnp.uint32)

    fields = []
    for index in range(per_period):
        start, shift = divmod(index * width, 8)
        window = padded[..., start]
        for byte in range(1, span):
            window = window | (padded[..., start + byte] << np.uint32(8 * byte))
        fields.append((window >> np.uint32(shift)) & np.uint32((1 << width) - 1))
    return fnp.stack(fields, axis=-1).reshape(v.shape[0], -1)


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
    blocks = _budget(N, _NTT_ACCEPT, _NTT_PER_BLOCK)
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
    blocks = _budget(N, _BOUNDED_ACCEPT[eta], _BOUNDED_PER_BLOCK)
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
    """
    if gamma1 & (gamma1 - 1):
        raise ValueError(f"gamma1 ({gamma1}) must be a power of 2 — FIPS 204 Table 1")
    width = 1 + (gamma1 - 1).bit_length()
    tails = np.array([_integer_to_bytes(mu + r, 2) for r in range(ell)], dtype=np.uint8)
    v = Shake256(32 * width).digest(_seeds(rho, 64, tails))
    return np.int32(gamma1) - _bit_unpack_field(v, width).astype(np.int32)


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
    """
    seed = namespace(rho).asarray(rho, dtype=np.uint8).reshape(-1)
    # A step's byte is accepted when it is at most `i`, and `i` is smallest at
    # the first step — so every step accepts at least as often as `(257−τ)/256`,
    # and one budget sized at that rate covers all of them.
    budget = _budget(tau, (N - tau + 1, N), 1)
    stream = Shake256(8 + budget).digest(seed[None, :])[0]

    signs = _bytes_to_bits(stream[:8]).astype(np.int32)
    candidates = stream[8:].astype(np.int32)
    position = fnp.arange(budget, dtype=np.int32)
    slot = fnp.arange(N, dtype=np.int32)

    c = fnp.zeros(N, dtype=np.int32)
    cursor = fnp.zeros((), dtype=np.int32)
    completed = fnp.zeros((), dtype=np.int32)
    for step in range(tau):
        i = N - tau + step
        available = (position >= cursor) & (candidates <= np.int32(i))
        # The first still-unread admissible byte, as a mask rather than an index.
        first = available & (fnp.cumsum(available) == np.int32(1))
        completed = completed + first.any().astype(np.int32)
        j = fnp.sum(fnp.where(first, candidates, np.int32(0)))
        cursor = fnp.sum(fnp.where(first, position + np.int32(1), np.int32(0)))
        # `c[i] ← c[j]` then `c[j] ← (−1)^h[i+τ−256]`, in that order.
        at_j = fnp.sum(fnp.where(slot == j, c, np.int32(0)))
        c = fnp.where(slot == np.int32(i), at_j, c)
        c = fnp.where(slot == j, np.int32(1) - np.int32(2) * signs[step], c)
    _require_enough(completed, tau, "SampleInBall")
    return c
