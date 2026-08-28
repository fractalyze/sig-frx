# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""KoalaBear residues, and the wide host integers that decompose into them.

Several things in leanSig are integers before they are field elements. The
sponge's capacity packs a hashing task's shape into 32-bit slots
([`poseidon.py`](poseidon.py)), the message hash packs a 32-byte root and a
`(epoch << 8) | prefix` tweak ([`encoding.py`](encoding.py)), and the chain and
tree hashes pack a position ([`tweakable.py`](tweakable.py)). Every one of them
is wider than an array lane — the root is 256 bits, the tweaks 40 and 56 — so
each stays a host integer and only the limbs, all below `PRIME`, ever cross onto
a device ([`../../../CLAUDE.md`](../../../CLAUDE.md)).

That is the whole module. It exists rather than staying private to its first
caller because the decomposition has a second one, which is the question
[`conventions.md`](../../../docs/reference/conventions.md#generalize-a-component-when-its-second-consumer-arrives)
asks — and the answer here is not a judgement call, because upstream already
made it: leanSpec's own `int_to_base_p` lives in `spec/crypto/xmss/field.py`,
shared by `poseidon.safe_domain_separator` and `encoding` for exactly these
call sites. One operation used twice, not two that resemble each other.

**What callers get is `lane_reversed_limbs`, not the limbs.** Every one of them
wants the decomposition placed for a hash, and the placement is a reversal
([`poseidon.py`](poseidon.py) says why the state runs lane-reversed). Spelling
`to_field(limbs[::-1])` per call site would leave the reversal optional at each
of them, and a forgotten one is a silently different hash — it round-trips, it
self-checks, and only an upstream vector for that particular family catches it.
The chain and tree tweaks were the fourth and fifth call sites to want it, and
they are what made the column form below necessary. Reversing a host list is not
the device-side placement `poseidon.py` reserves to itself — its own docstring
carves this case out.

The conversion is `astype` and never a bitcast: the dtype's storage is a
Montgomery representative, so reinterpreting the bytes yields a different number
and yields it consistently, which is the failure no round trip reveals.

**`to_field` is not `arith.to_field`**, though the repo gives one operation one
name and these two share it. The lattice pair
([`mldsa/arith.py`](../../lattice/mldsa/arith.py),
[`falcon/arith.py`](../../lattice/falcon/arith.py)) reduce their input mod `q`
and answer in the namespace it arrived in; this one takes residues that are
canonical already and lifts unconditionally, because everything leanSig hashes
reaches a device permutation anyway. A reader who transfers what they know from
those is wrong on both counts.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import frx.numpy as fnp
import numpy as np
import zk_dtypes
from frx import Array
from frx.typing import ArrayLike
from zk_dtypes import koalabear_mont as F

PRIME: Final = zk_dtypes.pfinfo(F).modulus
"""The KoalaBear prime, read off the dtype rather than restated beside it.

The pinned wheel is then the single source of truth for the modulus, which is
what [`secp.py`](../../classical/secp.py) does with its curve moduli. leanSpec
states the same value in `spec/crypto/koalabear.py`.
"""


def to_field(canonical: ArrayLike) -> Array:
    """Canonical residues -> a field array. The dtype cast Montgomery-encodes.

    Host-side values only: everything that reaches this is a parameter set or the
    limbs `_int_to_base_p` returned, which are host integers by construction.

    Residues *do* arrive traced, off the wire — but they arrive already in lanes
    and in leanSpec's order, so what that path needs is a reversal as well as a
    cast, and it lives with the codec that reads them ([`ssz.py`](ssz.py)).
    """
    return fnp.asarray(np.asarray(canonical, dtype=np.int64).astype(F))


def host_column(values: ArrayLike, describe: str, bound: int) -> np.ndarray:
    """`values` as a host `int64` column, refused unless all of it is in range.

    Three things in this package take a column of host integers and have to
    know every entry is bounded before packing it: a tweak field, whose
    neighbour it would otherwise carry into ([`tweakable.py`](tweakable.py)); a
    PRF input, whose fixed-width big-endian packing it would overflow
    ([`prf.py`](prf.py)); and a public parameter, which the cast into the field
    would silently *reduce* rather than refuse
    ([`leansig.py`](leansig.py)). All three failures are a wrong value that is
    still well-formed, so none of them is caught downstream.

    `describe` is the whole of the caller's own sentence — "a step takes 8 bits
    in a packed tweak, so it must be in [0, 256)" — because what makes each
    bound wrong differs and the diagnosis is the useful half. What is shared,
    and written once here, is the mechanism and the tail that counts the
    offenders and names the first.
    """
    column = np.asarray(values, dtype=np.int64).reshape(-1)
    outside = (column < 0) | (column >= bound)
    if np.any(outside):
        raise ValueError(
            f"{describe}; {int(np.count_nonzero(outside))} of {column.size} "
            f"entries are not, the first being {int(column[np.argmax(outside)])}"
        )
    return column


def lane_reversed(canonical: ArrayLike) -> Array:
    """Canonical residues in leanSpec's order -> the lane-reversed field array.

    The placement half of `lane_reversed_limbs`, for values that are residues
    already and so have nothing to decompose. A PRF squeeze is what wants it —
    SHAKE128 hands back a digest as `hash_length` residues read big-endian
    ([`prf.py`](prf.py)) — and so is a public parameter arriving as the numbers a
    key pair published rather than as bytes.

    Here rather than at either caller for the reason the module docstring gives:
    the reversal spelled per call site is a reversal that can be forgotten at
    one, and a forgotten one is a silently different hash. The last axis is the
    one that moves, so a `[N, k]` batch reverses each row and a `[k]` vector
    reverses itself.

    Host-side, like everything else here. The traced counterpart is
    [`ssz.py`](ssz.py)'s, which reverses residues that arrived in lanes off the
    wire and goes through `poseidon`'s own seam to do it.
    """
    return to_field(np.asarray(canonical, dtype=np.int64)[..., ::-1])


def lane_reversed_limbs(value: int | np.ndarray, num_limbs: int) -> Array:
    """`value` base-p, as the lane-reversed field vector a leanSig hash takes.

    The composite every caller wants: decompose, place, convert. Host-only for
    the reason the module docstring gives — `value` is wider than a lane.

    **One value, or a column of them.** A Python integer gives `[num_limbs]`, and
    a host integer array of shape `[B]` gives `[B, num_limbs]` — the batch a
    Merkle level or a chain step tweaks with, where the level is shared and the
    index is per entry ([`tweakable.py`](tweakable.py)). The two are one function
    because they are one operation: `%` and `//` are elementwise, so the column
    form is the scalar one with nothing removed, and writing it twice would
    leave two places for the reversal to be forgotten in.

    The widths do differ, and that difference is the column form's only real
    constraint. A scalar caller packs something no integer dtype holds — a
    256-bit root, a four-slot capacity shape — so it must stay a Python integer.
    A column arrives as `int64`, which caps it at `2^63`; every packed tweak is
    far below that, and anything that is not is refused by the limb-fit check
    below rather than wrapping silently.
    """
    return to_field(np.stack(_int_to_base_p(value, num_limbs), axis=-1)[..., ::-1])


def lane_reversed_limbs_stack(values: Sequence[int], num_limbs: int) -> Array:
    """A whole batch of *wide* values, decomposed on the host and lifted once.

    The scalar form above, repeated — and it is a third entry point rather than a
    third mode of that one because the input type is what differs. A column of
    `int64` is an array and decomposes elementwise; a batch of 256-bit roots
    cannot be one, since no integer dtype holds them, so the loop is Python's and
    stays here.

    What it buys is the transfer. `lane_reversed_limbs` ends in `to_field`, so
    calling it per entry is one host-to-device copy per entry; this assembles the
    limbs as a single host array and converts once. That is
    [`tweakable.py`](tweakable.py)'s argument for `chain_step_tweaks` — "one
    host-to-device transfer instead of seven" — applied to the operand a verifier
    has `B` of.
    """
    return to_field(
        np.stack([np.stack(_int_to_base_p(value, num_limbs))[::-1] for value in values])
    )


def _int_to_base_p(value: int | np.ndarray, num_limbs: int) -> list[int | np.ndarray]:
    """`value` as `num_limbs` base-p limbs, least significant first.

    Private because `lane_reversed_limbs` is what callers want; kept as its own
    function because a digest says only that something is wrong, not which limb,
    so this is gated directly.

    Host-only, and the packing is why: every caller decomposes something wider
    than a lane, so only a Python integer — or a host `int64` column — holds the
    input without truncating. Each limb that comes back is below `PRIME` and may
    then cross onto a device.

    A short decomposition is rejected rather than truncated — dropping the high
    part would silently change the hash it feeds, which is upstream's reasoning
    for the same rejection. `np.any` is what makes that check read the same for
    a scalar and for a column: one entry that does not fit rejects the batch,
    because the alternative is a wrong hash for that entry alone.
    """
    limbs = []
    remaining = value
    for _ in range(num_limbs):
        # `divmod` rather than `%` then `//=`, which on an ndarray would floor
        # divide the caller's array in place.
        remaining, limb = divmod(remaining, PRIME)
        limbs.append(limb)
    if np.any(remaining):
        raise ValueError(f"value does not fit in {num_limbs} base-p limbs")
    return limbs
