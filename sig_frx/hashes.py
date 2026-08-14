# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Which implementation of a hash a value's namespace calls for.

[`arrays.namespace`](arrays.py) picks the array module off its arguments; this
picks the `ByteHash` off the same question, and for the same reason. A hash is
the one operation a scheme reaches for that has two implementations of the same
function — hash-frx ships a device sponge and a `hashlib` sibling, gated against
each other — so it is the one place where "a value is used in the namespace it
arrives in" ([`conventions.md`](../docs/reference/conventions.md)) has an answer
to give rather than a lift to accept.

**The lift is not forced here, which is the whole of it.** `frx.lax.ntt` has no
host implementation, so a host argument to `arith.ntt` is lifted because there is
nowhere else for it to go. A hash is not that: a concrete caller reading its
digest back immediately is exactly what `hash_frx`'s host rows exist for, and
naming the device sponge unconditionally spends a dispatch to batch a message
with itself. So the exemption hashing was granted on the grounds that the lift is
forced does not survive the grounds being false.

**Which way a call goes is the caller's fact, never the scheme's.** One instance
verifies under a tracer and signs concretely, so this cannot be a property fixed
when a scheme is built — it is read off the values at each call, which is what
makes `verify` keep the device sponge without anything here naming verification.

The reverse direction needs no rule and gets none: a host hash cannot be called
on a tracer at all, because it reads the message bytes. `ByteHash`'s return type
is what says so, and it is why nothing below can pick wrong in a way that runs.
"""

from __future__ import annotations

from collections.abc import Callable

from hash_frx.byte_hash import ByteHash
from hash_frx.keccak.byte_hashes import (
    HostShake128,
    HostShake256,
    Shake128,
    Shake256,
)

from sig_frx.arrays import traced

# A `ByteHash` family: the constructor an output length is handed to. It is the
# type `tweakable.ShakeTweakableHash` already takes an XOF as, so a caller that
# holds one of these can pass it there unchanged.
Xof = Callable[[int], ByteHash]


def shake128(*values: object) -> Xof:
    """SHAKE128 as the namespace of `values` calls for it.

    ML-DSA's `G` — `ExpandA` and nothing else, since `G` is the standard's name
    for the 128-bit XOF and only the matrix is sampled from it.
    """
    return Shake128 if traced(*values) else HostShake128


def shake256(*values: object) -> Xof:
    """SHAKE256 as the namespace of `values` calls for it.

    ML-DSA's `H`, which is the rest of the scheme: the commitment hash, the two
    seed derivations, and the three samplers that are not `ExpandA`.
    """
    return Shake256 if traced(*values) else HostShake256
