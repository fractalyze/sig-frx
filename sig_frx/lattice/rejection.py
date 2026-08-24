# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Rejection sampling as a fixed budget plus a compaction, for the lattice schemes.

Both lattice standards draw uniform values by squeezing a SHAKE and discarding
the candidates that land outside a range, and both write it as a `while` that
squeezes until enough have survived. That trip count is the data, so no tracer
has it, and the loop is reshaped into a fixed candidate budget plus a
compaction — the shape
[`conventions.md`](../../docs/reference/conventions.md) allows for a rejection
whose candidates are public. This module is that reshaping, and the two things
it has to get right are how large the budget is and how the survivors are
collected.

It lives here rather than in either scheme because both ask the same two
questions of different constants: FIPS 204 §7.3's samplers keep 23-bit draws
below `q` and nibbles below a threshold, and Falcon's `HashToPoint` (Algorithm 3)
keeps 16-bit draws below `⌊2^16/q⌋·q`. The cost the shared page names is not the
duplicated lines but two adaptations that look unrelated, so a mistake understood
in one is never looked for in the other.

## The budget is computed, not chosen

A generous static bound is only as good as the argument that it is generous
enough, and "5 blocks is what everyone uses" is not one — the implementations
that use 5 also loop when 5 is not enough, which is the part this cannot do. So
`budget` computes the smallest budget whose *exact* binomial shortfall
probability is at most `2^-256`, the collision strength of the strongest
parameter set either scheme defines. It is integer arithmetic over the acceptance
probability the standard's own rejection rule defines, evaluated once per
parameterisation on the host, and it means the bound-exhausted path is
unreachable rather than merely unlikely (see `require_enough` for what happens if
it is reached anyway).

Which loop shape a rejection gets is the scheme's decision to record, and each
scheme's page records its own alongside the loops that went the other way.

## Collection is a gather, and a running count is the schedule

Keeping the accepted candidates and closing the gaps is a data-dependent
permutation however it is written, and the direction it is written in matters:
a gather vectorises where the same permutation as a scatter serialises on a GPU.
So it is a gather — but of the two ways to reach one, the cheap one is a
`cumsum`. Ranking the survivors by a running count and looking up where each
rank first appears answers the question about the outputs actually wanted;
sorting the whole budget on a one-bit key answers it by computing a permutation
of everything else as well, and measured 5-11x the cost of the scan at the shapes
here.
"""

from __future__ import annotations

from functools import lru_cache
from math import comb
from typing import Any

import frx
import frx.core
import frx.numpy as fnp
import numpy as np

# The shortfall probability every budget is sized against. `2^-256` is the
# collision strength `λ` of the strongest parameter set FIPS 204 Table 1 defines,
# so a budget that meets it is not the weakest part of any scheme sampling
# through here — and the margin is cheap, since the tail falls off fast enough
# that buying it costs one further block.
LOG2_SHORTFALL = 256


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
    return (shortfall << LOG2_SHORTFALL) > den**trials


@lru_cache(maxsize=None)
def budget(needed: int, accept: tuple[int, int], per_block: int) -> int:
    """The fewest blocks of `per_block` candidates that safely yield `needed`.

    Public because the sizing is the schemes' too, not only the samplers': at
    `needed = 1` and one candidate per block it answers how many independent
    attempts make failing altogether unreachable, which is what ML-DSA bounds its
    rejection loop by. One derivation of the tail for every loop that has one.

    Cached because it is the same handful of parameterisations for the life of a
    process, and because the tail is big-integer work that no caller should pay
    twice.
    """
    blocks = -(-needed // per_block)
    while _shortfall_exceeds_margin(blocks * per_block, needed, accept):
        blocks += 1
    return blocks


def require_enough(survivors: Any, needed: int, sampler: str) -> None:
    """Raise if any stream produced fewer than `needed` survivors.

    The check runs wherever it can be run. Key generation and signing are
    concrete, so the count is a number there and a shortfall raises; under a
    tracer it is not a number and no comparison on it can raise. What stands in
    for the check on that path is `budget`: the count it sizes for cannot fall
    short more often than `2^-256`, which is below the collision strength of the
    scheme being sampled for.

    `frx.experimental.checkify` would defer a real check into the traced path,
    and it is deliberately not used. It functionalises the caller — `checkify`
    turns `f` into one returning `(error, out)` — so the error has to be threaded
    to the top of the enclosing zone and thrown there. That top is
    `Signature.verify`, which returns `bool[B]` and nothing else, so adopting it
    means a validity flag every caller threads and most drop: a guarantee traded
    for a convention.
    """
    if isinstance(survivors, frx.core.Tracer):
        return
    short = np.atleast_1d(np.asarray(survivors) < needed)
    if short.any():
        raise RuntimeError(
            f"{sampler}: {int(short.sum())} of {short.size} streams ran out of "
            f"candidates before {needed} survived rejection. The budget is sized "
            f"for a shortfall probability below 2^-{LOG2_SHORTFALL}, so this is "
            f"a wrong budget rather than an unlucky seed."
        )


def first_accepted(values: Any, accepted: Any, count: int, sampler: str) -> Any:
    """The first `count` accepted values of each row, in stream order.

    `cumsum` over the acceptance flags gives each candidate the rank it would
    have among the survivors, and that running count is non-decreasing — so the
    source of output `r` is where rank `r + 1` first appears, which is a
    `searchsorted`. The survivor count falls out of the same scan as its last
    entry, so the shortfall check costs nothing here.

    **`clip` is not redundant**, even though `require_enough` has just refused
    the case it covers. Under a tracer that refusal cannot fire, and a shortfall
    leaves `searchsorted` returning one past the end for every unfilled slot —
    where `take`'s default is to fill with `INT32_MIN` rather than to clamp. The
    mode pins the unreachable branch to the same wrong answer on every backend
    instead of to whatever each one does at the edge, which is the property
    enc-frx's `ml_kem.sampling._compact` states for the same reason.
    """
    ranks = fnp.cumsum(accepted, axis=-1, dtype=np.int32)
    require_enough(ranks[..., -1], count, sampler)
    wanted = fnp.arange(1, count + 1, dtype=np.int32)
    return frx.vmap(
        lambda row, rank: fnp.take(
            row, fnp.searchsorted(rank, wanted), axis=-1, mode="clip"
        )
    )(values, ranks)
