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
    limbs `_int_to_base_p` returned, and a traced value has no business being
    rebuilt from residues.
    """
    return fnp.asarray(np.asarray(canonical, dtype=np.int64).astype(F))


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
