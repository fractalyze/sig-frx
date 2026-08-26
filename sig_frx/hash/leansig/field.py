# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""KoalaBear residues, and the wide host integers that decompose into them.

Two things in leanSig are integers before they are field elements. The sponge's
capacity packs a hashing task's shape into 32-bit slots
([`poseidon.py`](poseidon.py)), and the message hash packs a 32-byte root and a
`(epoch << 8) | prefix` tweak ([`encoding.py`](encoding.py)). Every one of them
is wider than an array lane — the root is 256 bits, the tweak 40 — so each stays
a Python integer on the host and only the limbs, all below `PRIME`, ever cross
onto a device ([`../../../CLAUDE.md`](../../../CLAUDE.md)).

That is the whole module: one decomposition and one conversion. It exists as a
module rather than as two helpers private to their first caller because the
decomposition has a second one, which is the question
[`conventions.md`](../../../docs/reference/conventions.md#generalize-a-component-when-its-second-consumer-arrives)
asks — and the answer here is not a judgement call, because upstream already
made it: leanSpec's own `int_to_base_p` lives in `spec/crypto/xmss/field.py`,
shared by `poseidon.safe_domain_separator` and `encoding` for exactly these two
call sites. One operation used twice, not two that resemble each other.

The conversion is `astype` and never a bitcast: the dtype's storage is a
Montgomery representative, so reinterpreting the bytes yields a different number
and yields it consistently, which is the failure no round trip reveals.
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
    limbs `int_to_base_p` returned, and a traced value has no business being
    rebuilt from residues.
    """
    return fnp.asarray(np.asarray(canonical, dtype=np.int64).astype(F))


def int_to_base_p(value: int, num_limbs: int) -> list[int]:
    """`value` as `num_limbs` base-p limbs, least significant first.

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
