# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The SLH-DSA address structure — FIPS 205 §4.2, encoded per §11.

Every hash in SLH-DSA is tweaked by an address saying exactly where in the
hypertree it sits: which layer, which tree, which node. That tweak is what stops
a hash computed in one position from being replayed in another, and it is the
most error-prone part of the scheme — a field packed in the wrong order produces
an implementation that is perfectly self-consistent and fails every known-answer
test with no signal about why.

An address is structural, never data-dependent: it is a function of where the
caller is in the tree, so it is built on the host and encoded into bytes before
any hashing starts. `encode` and `encode_batch` are that boundary — everything
above them is integers, everything below is a `uint8` array.

**A field is one integer or one per row.** The callers address a whole batch at a
time — every chain of every key pair, every node of a Merkle level — and building
those one object at a time is what dominated verification: a dataclass plus six
`int.to_bytes` calls per address, several hundred thousand times per call. So the
per-type constructors below take array-likes as readily as integers, and
`encode_batch` writes the whole batch with array arithmetic. What does *not* change
is that a caller still names its fields: `wots_hash(layer=…, chain=…)` reads the
same whether it is given scalars or columns, and a field written into the wrong
slot stays the error this module is shaped to prevent.

Two encodings, and a parameter set uses exactly one. The full 32-byte form (§4.2)
is what the SHAKE sets tweak with; the SHA-2 sets use the compressed 22-byte
ADRS^c, which §11.2 defines as
`ADRS_c = ADRS[3] ‖ ADRS[8:16] ‖ ADRS[19] ‖ ADRS[20:32]` — the same address with
the leading bytes of the layer address, tree address and type dropped, all of
which are zero for every address a defined parameter set can produce.

The slot table below is the whole of what is FIPS 205 about the encoding; writing
fields into slots is `adrs_encoding`, shared with the RFC 8391 layout.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from sig_frx.hashbased import adrs_encoding

# Word sizes of the full encoding, in bytes (FIPS 205 §4.2, Figure 1).
_LAYER_BYTES = 4
_TREE_BYTES = 12
_TYPE_BYTES = 4
_WORD_BYTES = 4
ADRS_SIZE = _LAYER_BYTES + _TREE_BYTES + _TYPE_BYTES + 3 * _WORD_BYTES

# The compressed encoding (§11.2.1) keeps only the low byte of the layer address
# and the low 8 bytes of the tree address; the trailing three words are unchanged.
_COMPRESSED_LAYER_BYTES = 1
_COMPRESSED_TREE_BYTES = 8
_COMPRESSED_TYPE_BYTES = 1
COMPRESSED_ADRS_SIZE = (
    _COMPRESSED_LAYER_BYTES
    + _COMPRESSED_TREE_BYTES
    + _COMPRESSED_TYPE_BYTES
    + 3 * _WORD_BYTES
)


Field = adrs_encoding.Field


class AdrsType(enum.IntEnum):
    """The seven address types — FIPS 205 §4.2, Table 1."""

    WOTS_HASH = 0
    WOTS_PK = 1
    TREE = 2
    FORS_TREE = 3
    FORS_ROOTS = 4
    WOTS_PRF = 5
    FORS_PRF = 6


@dataclass(frozen=True)
class Adrs:
    """One address, or a batch of them at one type.

    The three trailing words carry different meanings per type — key pair
    address, chain address or tree height, hash address or tree index — and a
    type zeroes the ones it does not use. They are stored as one tuple rather
    than as named fields precisely because the names are not shared: `TREE`'s
    first trailing word is padding while `FORS_TREE`'s is a key pair address, and
    a field named for one of those would be a lie in the other. Build through the
    per-type constructors below, which name their arguments the way Table 1 does.

    Every field is an integer or an array of them, and the arrays broadcast against
    each other: one address is all scalars, and a batch varies whichever fields
    vary. The type is always one value, since a batch of addresses is a batch of one
    type — every caller here hashes one kind of position at a time.
    """

    layer: Field
    tree: Field
    type: AdrsType
    trailing: tuple[Field, Field, Field]


def wots_hash(
    layer: Field, tree: Field, key_pair: Field, chain: Field, hash_index: Field
) -> Adrs:
    """A WOTS+ chain step."""
    return Adrs(layer, tree, AdrsType.WOTS_HASH, (key_pair, chain, hash_index))


def wots_pk(layer: Field, tree: Field, key_pair: Field) -> Adrs:
    """The compression of one WOTS+ public key."""
    return Adrs(layer, tree, AdrsType.WOTS_PK, (key_pair, 0, 0))


def hash_tree(layer: Field, tree: Field, height: Field, index: Field) -> Adrs:
    """An XMSS tree node. The first trailing word is padding, not a key pair."""
    return Adrs(layer, tree, AdrsType.TREE, (0, height, index))


def fors_tree(
    layer: Field, tree: Field, key_pair: Field, height: Field, index: Field
) -> Adrs:
    """A FORS tree node."""
    return Adrs(layer, tree, AdrsType.FORS_TREE, (key_pair, height, index))


def fors_roots(layer: Field, tree: Field, key_pair: Field) -> Adrs:
    """The compression of all FORS roots."""
    return Adrs(layer, tree, AdrsType.FORS_ROOTS, (key_pair, 0, 0))


def wots_prf(layer: Field, tree: Field, key_pair: Field, chain: Field) -> Adrs:
    """A WOTS+ secret-key element, generated by the PRF."""
    return Adrs(layer, tree, AdrsType.WOTS_PRF, (key_pair, chain, 0))


def fors_prf(layer: Field, tree: Field, key_pair: Field, index: Field) -> Adrs:
    """A FORS secret-key element, generated by the PRF."""
    return Adrs(layer, tree, AdrsType.FORS_PRF, (key_pair, 0, index))


def _slot_widths(*, compressed: bool) -> tuple[adrs_encoding.Slot, ...]:
    """Each field's slot, in the order §4.2 lays them out."""
    return (
        ("layer address", _COMPRESSED_LAYER_BYTES if compressed else _LAYER_BYTES),
        ("tree address", _COMPRESSED_TREE_BYTES if compressed else _TREE_BYTES),
        ("type", _COMPRESSED_TYPE_BYTES if compressed else _TYPE_BYTES),
        ("first word", _WORD_BYTES),
        ("second word", _WORD_BYTES),
        ("third word", _WORD_BYTES),
    )


def _fields(adrs: Adrs) -> tuple[Field, ...]:
    """The six values the slots take, in §4.2's order."""
    return (adrs.layer, adrs.tree, int(adrs.type), *adrs.trailing)


def encode(adrs: Adrs, *, compressed: bool) -> bytes:
    """One address as bytes — 22 when `compressed`, 32 otherwise.

    §11.2.1 defines the compressed form by taking the least significant byte of
    the layer address and the least significant 8 bytes of the tree address. A
    field too wide for its compressed slot raises rather than being cut down: no
    parameter set the standard defines can reach one (the tallest hypertree has 12
    layers and 64 bits of tree address), so an overflow is a caller that computed
    the wrong address.

    One address, field by field, the way §4.2 writes it. `encode_batch` produces
    these same bytes for a whole batch at once, and `adrs_test` requires the two to
    agree — this is the form that agreement is checked against.
    """
    return adrs_encoding.encode(_fields(adrs), _slot_widths(compressed=compressed))


def encode_batch(adrs: Adrs, *, compressed: bool) -> np.ndarray:
    """A batch of addresses of one type as uint8 `[rows, size]`.

    The batch axis is where a caller hashes many positions at once — every chain of
    every key pair, every node of a Merkle level — and the addresses vary per row
    while the seed does not. So the fields arrive as columns and the bytes come out
    of one gather: an address costs no Python of its own, which matters because a
    single verification addresses hundreds of thousands of them.

    Fields are bounded at 64 bits here, where `encode` is not. Nothing a defined
    parameter set produces comes close — the tallest hypertree has 12 layers and 64
    bits of tree address, and the compressed encoding gives the tree 8 bytes anyway
    — so a wider field is a caller that computed the wrong address, and it raises
    for the same reason a field too wide for its slot does.
    """
    return adrs_encoding.encode_batch(
        _fields(adrs), _slot_widths(compressed=compressed)
    )
