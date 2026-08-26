# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""KoalaBear residues, and the wide host integers that decompose into them.

Two things in leanSig are integers before they are field elements. The sponge's
capacity packs a hashing task's shape into 32-bit slots
([`poseidon.py`](poseidon.py)), and the message hash packs a 32-byte root and a
`(epoch << 8) | prefix` tweak ([`encoding.py`](encoding.py)). Every one of them
is wider than an array lane — the root is 256 bits, the tweak 40 — so each stays
a Python integer on the host and only the limbs, all below `PRIME`, ever cross
onto a device ([`../../../CLAUDE.md`](../../../CLAUDE.md)).

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
Two more call sites are already scheduled: the chain and tree tweaks arrive with
the family that hashes with them ([`params.py`](params.py)). Reversing a host
list is not the device-side placement `poseidon.py` reserves to itself — its own
docstring carves this case out.

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


def lane_reversed_limbs(value: int, num_limbs: int) -> Array:
    """`value` base-p, as the lane-reversed field vector a leanSig hash takes.

    The composite every caller wants: decompose, place, convert. Host-only for
    the reason the module docstring gives — `value` is wider than a lane.
    """
    return to_field(_int_to_base_p(value, num_limbs)[::-1])


def _int_to_base_p(value: int, num_limbs: int) -> list[int]:
    """`value` as `num_limbs` base-p limbs, least significant first.

    Private because `lane_reversed_limbs` is what callers want; kept as its own
    function because a digest says only that something is wrong, not which limb,
    so this is gated directly.

    Host-only, and the packing is why: every caller decomposes something wider
    than a lane, so only a Python integer holds the input without truncating.
    Each limb that comes back is below `PRIME` and may then cross onto a device.

    A short decomposition is rejected rather than truncated — dropping the high
    part would silently change the hash it feeds, which is upstream's reasoning
    for the same rejection.
    """
    limbs = []
    remaining = value
    for _ in range(num_limbs):
        limbs.append(remaining % PRIME)
        remaining //= PRIME
    if remaining:
        raise ValueError(f"value does not fit in {num_limbs} base-p limbs")
    return limbs
