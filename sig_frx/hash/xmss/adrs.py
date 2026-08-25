# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The XMSS address structure — RFC 8391 §2.5.

Eight uniform 32-bit words, big-endian, written end to end: the encoding is
`addr_to_bytes`, `for i in 0..8: toByte(addr[i], 4)`. That flatness is the first
thing that separates it from FIPS 205's address (`adrs.py`), which partitions the
same 32 bytes as 4/12/4/4/4/4 and gives the tree address twelve of them.

The second difference is the last word. RFC 8391 randomizes every hash with a key
*and* a bitmask, both produced by `PRF(SEED, ADRS)`, so one position needs two or
three addresses that differ only in a `keyAndMask` counter — a word FIPS 205 has
no equivalent of. `with_key_and_mask` is that derivation, and it works on the
encoded bytes: the hash family gets one address and rewrites four bytes rather
than re-encoding. `with_third_word` is the same move one word earlier, for a
WOTS+ chain that advances the hash address and holds everything before it.

**Nothing here normalizes, and that is deliberate.** §2.5 says of the address
setters:

> we furthermore assume that the setType() method sets the four words following
> the type word to zero.

The reference implementation the test vectors come from does no such thing — its
`set_type` writes `addr[3]` and leaves words 4 through 7 to the caller — and the
vectors follow the reference. `test/vectors.c` fills its fixture address with
`addr[i] = 500000000*i`, so every word starts non-zero, and `wots_pkgen` overwrites
only the words it names: the fixture's type word and its word 4 reach the hash
unchanged. An address module that zeroed on behalf of a type would compute a
different WOTS+ public key and could not reproduce a single published digest.

So `type` is a plain field here rather than an enum, and the per-type constructors
below fill in exactly the words §2.5 names for their layout and nothing else.
That is the opposite of `adrs.py`, whose per-type constructors zero the trailing
words their type does not use and whose test asserts they do — the two modules
disagree on this on purpose, because their standards do.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from sig_frx.hash import adrs_encoding

Field = adrs_encoding.Field

# §2.5: eight words of four bytes each, the layer first and `keyAndMask` last.
WORD_BYTES = 4
_WORDS = 8
ADRS_SIZE = _WORDS * WORD_BYTES

# The tree address spans words 1 and 2, so it is carried as one 64-bit field and
# written into eight bytes — which is what `set_tree_addr`'s split into a high and
# a low word amounts to once both are packed big-endian.
_TREE_BYTES = 2 * WORD_BYTES

_SLOTS: tuple[adrs_encoding.Slot, ...] = (
    ("layer address", WORD_BYTES),
    ("tree address", _TREE_BYTES),
    ("type", WORD_BYTES),
    ("first word", WORD_BYTES),
    ("second word", WORD_BYTES),
    ("third word", WORD_BYTES),
    ("keyAndMask", WORD_BYTES),
)

# Which slot each rewritable word is. The tree spans two words, so these are not
# §2.5's word numbers.
_THIRD_WORD = 5
_KEY_AND_MASK = 6


class AdrsType(enum.IntEnum):
    """The three address types — RFC 8391 §2.5."""

    OTS = 0
    LTREE = 1
    HASH_TREE = 2


@dataclass(frozen=True)
class Adrs:
    """One address, or a batch of them.

    The three words between the type and `keyAndMask` carry different meanings per
    type — an OTS or L-tree address, a chain address or tree height, a hash address
    or tree index — and they are one tuple rather than named fields precisely
    because the names are not shared: word 5 is a chain address under `OTS` and a
    tree height under `LTREE`, and a field named for one of those would be a lie in
    the other. Build through the per-type constructors below, which name their
    arguments the way §2.5 does.

    `type` is a `Field`, not an `AdrsType`, because the reference's `set_type`
    writes one word without normalizing and the published WOTS+ vectors are
    generated from an address whose type word is neither 0, 1 nor 2. A component
    that receives an address passes its type through rather than asserting it.

    Every field is an integer or an array of them, and the arrays broadcast against
    each other: one address is all scalars, and a batch varies whichever fields
    vary.
    """

    layer: Field
    tree: Field
    type: Field
    trailing: tuple[Field, Field, Field]
    key_and_mask: Field = 0


def ots(
    layer: Field,
    tree: Field,
    ots_index: Field,
    chain: Field = 0,
    hash_index: Field = 0,
) -> Adrs:
    """A WOTS+ chain step — §2.5, Figure 2."""
    return Adrs(layer, tree, AdrsType.OTS, (ots_index, chain, hash_index))


def ltree(
    layer: Field,
    tree: Field,
    ltree_index: Field,
    height: Field = 0,
    index: Field = 0,
) -> Adrs:
    """A node of the L-tree compressing one WOTS+ public key — Figure 3."""
    return Adrs(layer, tree, AdrsType.LTREE, (ltree_index, height, index))


def hash_tree(layer: Field, tree: Field, height: Field, index: Field) -> Adrs:
    """A node of the main Merkle tree — Figure 4.

    The first of the three words is the padding §2.5 defines as zero for this
    layout, not a field a caller supplies. Writing it is the type's own definition
    rather than the normalization this module refuses to do.
    """
    return Adrs(layer, tree, AdrsType.HASH_TREE, (0, height, index))


def _fields(adrs: Adrs) -> tuple[Field, ...]:
    """The seven values the slots take, in §2.5's order."""
    return (
        adrs.layer,
        adrs.tree,
        adrs.type,
        *adrs.trailing,
        adrs.key_and_mask,
    )


def encode(adrs: Adrs) -> bytes:
    """One address as its 32 bytes, word by word, the way §2.5 writes it.

    `encode_batch` produces these same bytes for a whole batch at once, and
    `rfc8391_adrs_test` requires the two to agree — this is the form that agreement
    is checked against.
    """
    return adrs_encoding.encode(_fields(adrs), _SLOTS)


def encode_batch(adrs: Adrs) -> np.ndarray:
    """A batch of addresses as uint8 `[rows, 32]`.

    The batch axis is where a component hashes many positions at once — every chain
    of every key pair, every node of an L-tree level — and the addresses vary per
    row while the seed does not.
    """
    return adrs_encoding.encode_batch(_fields(adrs), _SLOTS)


def with_key_and_mask(encoded: np.ndarray, value: int) -> np.ndarray:
    """The same encoded addresses with `keyAndMask` replaced.

    §5.1 derives a hash's key and its bitmask from the same position, separated
    only by this word: `F` needs values 0 and 1, and `H` needs 0, 1 and 2 because
    its bitmask is twice as long. So this rewrites four bytes of an address a
    caller already encoded rather than making it build two or three.
    """
    return adrs_encoding.with_field(encoded, _SLOTS, _KEY_AND_MASK, value)


def with_third_word(encoded: np.ndarray, value: int) -> np.ndarray:
    """The same encoded addresses with their third word replaced.

    A WOTS+ chain is what asks for it: §3.1.2 advances the hash address a step at a
    time and holds the layer, tree, type, OTS address and chain address across the
    whole walk, so its `w − 1` addresses are one batch encoded and `w − 1` words
    written into it rather than `w − 1` encodes of fields that never move.

    Named for the slot rather than for what it holds, the way `Adrs.trailing` is:
    the word is a hash address under `OTS` and a tree index under `LTREE`.
    """
    return adrs_encoding.with_field(encoded, _SLOTS, _THIRD_WORD, value)
