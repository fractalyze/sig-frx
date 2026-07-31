# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Packing integer fields into fixed-width big-endian slots.

Both address encodings this package carries — FIPS 205 §4.2 and RFC 8391 §2.5 —
reduce to the same mechanical step over a different table: each field is written
big-endian into a slot of a fixed width, and the slots run end to end. What the two
disagree about is everything above that — which fields exist, what a type word
means, which words a per-type constructor may fill in — and that is why they are
two modules. The packing is one, so it is here rather than copied into both, where
a fix to one would silently leave the other behind.

Two entry points, and an address module is required to make them agree. `encode`
writes one address the way its standard spells it out, and is what a layout test
pins against the specification. `encode_batch` writes a whole batch with array
arithmetic, and is what every component actually calls — a single verification
addresses hundreds of thousands of positions, so an address has to cost no Python
of its own.

`rows` and `columns` are that batch rule made available to the components above,
which carry their own positions as columns and have to agree with the encoding
about what a batch is before they build one.

A field too wide for its slot raises rather than being cut down. Silently dropping
its high bytes would tweak two different positions identically, which is the one
failure an address exists to prevent.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np

# One field of an address: a single value, or one per row of a batch. The arrays
# broadcast, so a caller mixes the two freely — a fixed layer against a column of
# node indices is the common shape.
Field = int | np.ndarray

# One slot of an encoding: what the field is called, and how many bytes it gets.
# The name is only ever read back out in an error message, which is the whole
# point of carrying it — "tree address does not fit in 8 bytes" names the caller's
# mistake where a slot number would not.
Slot = tuple[str, int]

# The batched path carries each field as a uint64, so its big-endian form is eight
# bytes. Nothing a defined parameter set produces comes near that limit.
FIELD_BYTES = 8


def size(slots: Sequence[Slot]) -> int:
    """How many bytes an encoding of these slots takes."""
    return sum(width for _, width in slots)


def rows(values: Sequence[Field]) -> int:
    """How many addresses these fields describe — one, or the length they broadcast to.

    The batch a caller is addressing, asked before any address is built. A caller
    that carries its position as columns needs it to lay out the rows hanging off
    each one — a WOTS+ key pair owns `len` chains, so it has to know how many key
    pairs it has before it can tile the chain numbers under them.

    The same broadcast `columns` performs, so the two cannot disagree about what a
    batch is.
    """
    shape = np.broadcast_shapes(*(np.shape(value) for value in values))
    if len(shape) > 1:
        raise ValueError(
            f"address fields broadcast to one value per row, got shape {shape}"
        )
    return shape[0] if shape else 1


def columns(values: Sequence[Field]) -> np.ndarray:
    """The fields as one `[rows, len(values)]` unsigned table, broadcast together.

    Each field is cast to uint64 before the stack rather than after. Stacking a
    signed field beside an unsigned one promotes the pair to float64, which
    silently rounds any value above 2^53 — and a tree address reaches 2^63. That
    is the failure the width check below cannot see, because the rounding has
    already happened by the time there is a table to check.
    """
    arrays = []
    for value in values:
        array = np.asarray(value)
        if array.dtype.kind not in "iu":
            raise ValueError(
                f"address fields are integers, got dtype {array.dtype}; a field "
                f"wider than 64 bits needs `encode`, which is not width-bounded"
            )
        if array.dtype.kind == "i" and array.min() < 0:
            raise ValueError(f"address fields are unsigned, got {array.min()}")
        arrays.append(array)
    rows(values)
    return np.stack(
        [
            np.atleast_1d(value).astype(np.uint64)
            for value in np.broadcast_arrays(*arrays)
        ],
        axis=-1,
    )


def encode(values: Sequence[Field], slots: tuple[Slot, ...]) -> bytes:
    """One address as bytes, field by field, the way its standard writes it.

    This is the reference the batched form is checked against: it is what a layout
    test pins against the specification, so requiring the two to agree is what
    carries that agreement over to the form every component actually calls.

    Fields are not width-bounded here, where `encode_batch` bounds them at 64 bits.
    """
    return b"".join(
        _to_byte(int(value), width)
        for value, (_, width) in zip(values, slots, strict=True)
    )


def encode_batch(values: Sequence[Field], slots: tuple[Slot, ...]) -> np.ndarray:
    """A batch of addresses as uint8 `[rows, size(slots)]`.

    The fields arrive as columns and the bytes come out of one gather, so an
    address costs no Python of its own. Fields that are single values broadcast
    across the batch, which is the common shape: a fixed layer and type against a
    column of node indices.
    """
    table = columns(values)
    _reject_overflow(table, slots)
    positions, sources = _byte_plan(slots)
    encoded = np.zeros((table.shape[0], size(slots)), dtype=np.uint8)
    # `>u8` puts each field's most significant byte first, so the slots read off in
    # the standard's order.
    encoded[:, positions] = table.astype(">u8").view(np.uint8)[:, sources]
    return encoded


def _to_byte(value: int, length: int) -> bytes:
    """`toByte(x, n)` — the big-endian n-byte encoding of x, which both standards
    define identically (FIPS 205 §4.1, RFC 8391 §2.4)."""
    if value < 0:
        raise ValueError(f"address fields are unsigned, got {value}")
    try:
        return value.to_bytes(length, "big")
    except OverflowError as exc:
        raise ValueError(f"{value} does not fit in {length} bytes") from exc


@lru_cache(maxsize=None)
def _byte_plan(slots: tuple[Slot, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Which of the fields' big-endian bytes each output byte comes from.

    The mapping from output byte to source byte is a constant of the encoding, so
    precomputing it turns the whole encode into one gather — which is what keeps a
    small batch from costing more in array overhead than it saves in Python.

    Returns the output positions that carry a byte, and the source index for each.
    A slot wider than the eight bytes a field carries — FIPS 205 gives the tree
    address twelve — has leading bytes that are zero by construction and are absent
    from both.
    """
    positions: list[int] = []
    sources: list[int] = []
    offset = 0
    for field, (_, width) in enumerate(slots):
        for index in range(width):
            significance = width - 1 - index  # bytes below this one in the slot
            if significance < FIELD_BYTES:
                positions.append(offset + index)
                sources.append(FIELD_BYTES * field + (FIELD_BYTES - 1 - significance))
        offset += width
    return np.array(positions), np.array(sources)


def _reject_overflow(table: np.ndarray, slots: Sequence[Slot]) -> None:
    """Refuse a field too wide for the slot it is written into.

    Slots of eight bytes or more cannot overflow a uint64 field, so only the narrow
    ones are checked.
    """
    for field, (name, width) in enumerate(slots):
        if width >= FIELD_BYTES:
            continue
        column = table[:, field]
        limit = np.uint64(1) << np.uint64(8 * width)
        if column.max() >= limit:
            raise ValueError(
                f"{name} does not fit in {width} bytes: "
                f"{int(column[column >= limit][0])}"
            )
