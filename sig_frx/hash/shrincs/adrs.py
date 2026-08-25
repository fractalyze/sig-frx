# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""SHRINCS's stateful address — the specification's Stateful ADRS Format.

The same slots FIPS 205 compresses to ([`../adrs.py`](../adrs.py)): one byte,
then eight, then the type byte, then twelve. What differs is what the two leading
fields mean and what may sit in the type byte.

**The leading fields name a node, not a tree.** FIPS 205 addresses a position in
a hypertree — which layer, which tree within it — while a stateful SHRINCS
signature climbs one FXMSS tree, so the same slots carry the node's height and
its index across that height. The widths line up exactly, which is why the
encoder is shared rather than copied: the eight-byte slot that held a tree
address holds a node index of the same 64 bits, and it arrives as bytes for the
same reason (see [`../bytestring.py`](../bytestring.py)).

**The type values start at 16 and the stateless ones stop at 6.** That gap is
what keeps one key's two signing paths apart: `pk_seed` is shared between them,
so a hash computed on the stateful path must not be reachable as a stateless one
at the same position. Numbering them disjointly is the whole of the separation,
which is why they are transcribed here rather than derived.

Verification reaches four of the five. `WOTS_C_PRF` is a signer's address — it
derives secret chain starts — and this module carries no `prf` builder for the
same reason there is no `sign` beside it.
"""

from __future__ import annotations

import enum

import numpy as np
from frx import Array

from sig_frx.hash import adrs

Field = adrs.Field


class StatefulAdrsType(enum.IntEnum):
    """The stateful address types — the specification's ADRS Types table."""

    WOTS_C_HASH = 16
    WOTS_C_PK = 17
    FXMSS_TREE = 18
    WOTS_C_PRF = 21
    WOTS_C_GRIND = 22


# The `H_grind` tweak is the address's first ten bytes — the node's height and
# index and the type — and not the twelve-byte payload behind them, which that
# function leaves to its own message. Slicing rather than encoding a shorter
# address keeps one layout in one place.
GRIND_TWEAK_BYTES = 10


def wots_c_hash(
    node_height: Field, node_index: Field, chain: Field, hash_index: Field
) -> adrs.Adrs:
    """A WOTS+C chain step.

    The payload's first word is zero padding where FIPS 205 puts a key pair
    address: an FXMSS leaf holds one WOTS+C key, so the node index already names
    it and there is nothing left for that word to say.
    """
    return adrs.Adrs(
        node_height, node_index, StatefulAdrsType.WOTS_C_HASH, (0, chain, hash_index)
    )


def wots_c_pk(node_height: Field, node_index: Field) -> adrs.Adrs:
    """Compressing a WOTS+C public key into its FXMSS leaf."""
    return adrs.Adrs(node_height, node_index, StatefulAdrsType.WOTS_C_PK, (0, 0, 0))


def fxmss_tree(node_height: Field, node_index: Field) -> adrs.Adrs:
    """A parent in the FXMSS tree — the node this address names."""
    return adrs.Adrs(node_height, node_index, StatefulAdrsType.FXMSS_TREE, (0, 0, 0))


def wots_c_grind(node_height: Field, node_index: Field) -> adrs.Adrs:
    """Mapping a message digest into the constant-sum space, under a counter."""
    return adrs.Adrs(node_height, node_index, StatefulAdrsType.WOTS_C_GRIND, (0, 0, 0))


def encode_batch(address: adrs.Adrs) -> np.ndarray | Array:
    """The 22-byte encoding, as `[rows, 22]`.

    Always the compressed form: SHRINCS instantiates every hash with SHA-256, and
    the 22-byte address is what FIPS 205 §11.2 pairs with that. There is no
    SHAKE-shaped SHRINCS to make this a parameter.
    """
    return adrs.encode_batch(address, compressed=True)
