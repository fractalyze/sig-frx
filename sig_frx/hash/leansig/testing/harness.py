# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The vocabulary every leanSig suite needs: lane order, and the two legs.

Five suites here gate five layers of one scheme — the permutation, the two
modes over it, the message-to-codeword pipeline, the PRF and the signer — and
each of them meets the same two facts. Upstream's vectors are in leanSpec's
lane order while everything [`poseidon.py`](../poseidon.py) runs is over the
reverse of it, so a case
reverses on the way in and back on the way out; and every case runs twice, once
eagerly and once traced, because the two must agree in *dtype* as well as value.

Written once rather than per suite. The reversal is the convention most likely
to move — a dtype change, an x64 switch, a different canonical read would each
touch it — and three copies of it are three places a change has to be found,
with nothing to make a reader confident they still agree. That is the same
argument [`encoding_vectors.py`](encoding_vectors.py) makes for taking
`operand_elements` from `mode_vectors` rather than restating the rule.

The conversions here are the tests' own and deliberately not
[`field.to_field`](../field.py), which they would otherwise be gating with
itself. `field_test` gates that one directly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any, Final

import frx
import frx.numpy as fnp
import numpy as np
from zk_dtypes import koalabear_mont as F

from sig_frx.hash import tweakable as shared_tweakable
from sig_frx.hash.leansig import params
from sig_frx.hash.leansig.params import LeanSigParams

PRESETS: Final[dict[str, LeanSigParams]] = params.PRESETS
"""The preset a vector names, by the key its module spells.

Re-exported rather than restated: `params.py` owns the map now that the scheme
resolves a caller's preset string through the same one. Kept as a name here
because four suites already resolve through `harness.PRESETS`, and the
indirection is what let it move without touching them.
"""


def to_field(canonical: Sequence[int]) -> fnp.ndarray:
    """Canonical residues -> a field array. The dtype cast Montgomery-encodes.

    Separate from `lane_reversed` because a case that feeds a leanSpec-ordered
    vector *deliberately* — the ones that prove the reversal is load-bearing —
    spells the conversion the same way rather than inlining its own.
    """
    return fnp.asarray(np.asarray(canonical, dtype=np.int64).astype(F))


def lane_reversed(canonical: Sequence[int]) -> fnp.ndarray:
    """leanSpec-ordered residues -> the lane-reversed field vector the scheme takes.

    The reversal is on the host, where it is a slice of a sequence rather than a
    device `reverse`.
    """
    return to_field(canonical[::-1])


def to_canonical(values: fnp.ndarray) -> list[int]:
    """A field array -> canonical residues.

    The object cast Montgomery-decodes without needing frx x64, which is why it
    is not a bitcast.
    """
    return [int(x) for x in np.asarray(values).astype(object)]


def to_leanspec_order(values: fnp.ndarray) -> list[int]:
    """A lane-reversed field array -> canonical residues in leanSpec's order.

    The mirror of `lane_reversed`, and here for the same reason: both sides of
    the convention belong in one place, so a case compares against upstream's
    order without re-deriving which end to read from.
    """
    return to_canonical(values)[::-1]


def lane_reversed_rows(rows: Sequence[Sequence[int]]) -> fnp.ndarray:
    """leanSpec-ordered rows -> the `[count, n]` stack of lane-reversed vectors.

    The stacked `lane_reversed`, which two suites want: a slot's chain ends and
    an authentication path are both a run of digests the vectors seed one at a
    time.

    Built as one host array and moved once, rather than one array per row. That
    is not a micro-optimization at these counts — a `PROD` signature is 46 chain
    ends and 32 siblings, so the per-row form is 78 transfers per case.
    """
    return fnp.asarray(np.asarray(rows, dtype=np.int64)[:, ::-1].astype(F))


def to_leanspec_rows(values: fnp.ndarray) -> list[list[int]]:
    """A `[count, n]` lane-reversed stack -> rows of residues in upstream's order.

    The mirror of `lane_reversed_rows`, and the row-wise `to_leanspec_order`.
    One `asarray` up front, so a stack costs one transfer rather than one per
    row.
    """
    return [to_canonical(row)[::-1] for row in np.asarray(values)]


def hex_of(values: Any) -> str:
    """A `uint8` array as the hex string a vector states it in."""
    return bytes(np.asarray(values, dtype=np.uint8)).hex()


def bytes_of(hex_string: str) -> np.ndarray:
    """A vector's hex string as the `uint8` row the seam takes."""
    return np.frombuffer(bytes.fromhex(hex_string), dtype=np.uint8)


def broadcast(operand: fnp.ndarray, count: int) -> fnp.ndarray:
    """A `[k]` operand widened to `[count, k]`, as the batched entry points take.

    `sig_frx.hash.tweakable.batched` under a name a suite can reach without
    learning what a family is — the leanSig callers all hand it the same field
    dtype, so there is nothing for a case to choose.
    """
    return shared_tweakable.batched(operand, count, dtype=F)


def on_leg(function: Callable[..., Any], jit: bool) -> Callable[..., Any]:
    """`function` on the requested leg — traced through the shared jit cache.

    Every suite here runs each case twice and picks the callable the same way.
    Spelled once so `jitted`'s static argument — `params`, for all four — is not
    restated at each of them.
    """
    return jitted(function, "params") if jit else function


@lru_cache(maxsize=None)
def jitted(function: Callable[..., Any], *static_argnames: str) -> Callable[..., Any]:
    """One jit wrapper per callable, shared across the cases that trace it.

    Not a compile saving — frx keys its executable cache on the wrapped function,
    so a fresh wrapper still hits it — but it keeps the per-call dispatch off the
    slowest target. Only module-level functions are ever passed: a lambda would
    be a fresh key each time, pinning its closure alongside every executable it
    compiled.
    """
    return frx.jit(function, static_argnames=static_argnames)


def both_legs(vectors: Sequence[Any]) -> list[tuple[str, Any, bool]]:
    """Each vector twice, named for the leg it runs on."""
    return [
        (f"{vector.name}_{'traced' if jit else 'host'}", vector, jit)
        for vector in vectors
        for jit in (False, True)
    ]
