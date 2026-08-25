# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Packing fields into fixed-width big-endian slots.

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

`with_field` is the one thing done to a batch after it is encoded, and it is there
because a component often addresses a run of positions that differ in one field —
a WOTS+ chain's `w − 1` steps differ in the hash address alone. Splicing that
field's bytes into a batch already built is what keeps the run from costing an
encode per position, and it produces the same bytes as encoding each of them,
which is what its test pins.

`rows` is that batch rule made available to the components above, which carry
their own positions as columns and have to agree with the encoding about what a
batch is before they build one.

**A field may arrive as bytes, and one has to.** An integer field rides an array
lane, and under a tracer that lane is 32 bits wide — frx runs without x64, where
`uint64` becomes `uint32` and a larger value truncates without raising. FIPS 205's
tree address reaches 2^64, so a traced caller hands those eight bytes over as
bytes rather than as a number it cannot represent. That is what the standard says
the value is anyway: §4.1 defines `toInt` and `toByte` as a pair precisely because
these live as byte strings, and §4.2's tree address is a byte slot. The integer
form is the host implementation's convenience.

A field too wide for its slot raises rather than being cut down. Silently dropping
its high bytes would tweak two different positions identically, which is the one
failure an address exists to prevent — **but that check reads the values, so it
only holds where they are concrete.** Under a tracer there is no data-dependent
control flow to raise from, and a caller has to bound its own fields. Every traced
field this repo builds is bounded by construction: they come out of a digest split
that masks each to its own bit width.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any, TypeAlias

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array

from sig_frx import arrays
from sig_frx.hash import bytestring

# One field of an address, in either of two forms. An **integer** field is a
# single value or one per row, and the arrays broadcast, so a caller mixes the two
# freely — a fixed layer against a column of node indices is the common shape. A
# **byte** field is `[rows, width]` uint8, already big-endian, for a slot whose
# value the caller holds as bytes.
Field: TypeAlias = int | bytestring.ByteString

# One slot of an encoding: what the field is called, and how many bytes it gets.
# The name is only ever read back out in an error message, which is the whole
# point of carrying it — "tree address does not fit in 8 bytes" names the caller's
# mistake where a slot number would not.
Slot = tuple[str, int]


def size(slots: Sequence[Slot]) -> int:
    """How many bytes an encoding of these slots takes."""
    return sum(width for _, width in slots)


def rows(values: Sequence[Field]) -> int:
    """How many addresses these fields describe — one, or the length they broadcast to.

    The batch a caller is addressing, asked before any address is built. A caller
    that carries its position as columns needs it to lay out the rows hanging off
    each one — a WOTS+ key pair owns `len` chains, so it has to know how many key
    pairs it has before it can tile the chain numbers under them.

    A byte field contributes its leading axis, since its trailing one is the
    slot's width rather than a second address.
    """
    shape = np.broadcast_shapes(
        *[
            np.shape(value)[:1] if bytestring.is_bytes(value) else np.shape(value)
            for value in values
        ]
    )
    if len(shape) > 1:
        raise ValueError(
            f"address fields broadcast to one value per row, got shape {shape}"
        )
    return shape[0] if shape else 1


def repeat_rows(value: Field, times: int) -> Field:
    """A field with each of its rows repeated `times` times, consecutively.

    What a caller laying rows out under each address needs — a WOTS+ key pair owns
    `len` chains, so its prefix repeats `len` times before the next key pair's.
    Either form of field, since the caller should not have to know which it holds:
    an integer column repeats its values, a byte field repeats its rows.

    A value that already covers every row is returned untouched — a single value
    by broadcasting, or any value when `times` is one. It already covers every row by
    broadcasting, and materializing it would only make the batch bigger. So the
    result does **not** report the row count: a caller that needs one has it from
    its own data, and `rows` answers the broadcast width rather than the batch.

    Not `tweakable.repeat_per_entry`, which looks like this and is not: that one
    passes rank-1 values through (an address column is rank 1) and forces uint8
    onto a device. The two sit side by side in `fors.pk_from_sig`, correctly.
    """
    if times == 1 or np.ndim(value) == 0:
        return value
    return arrays.namespace(value).repeat(value, times, axis=0)


def encode(values: Sequence[Field], slots: tuple[Slot, ...]) -> bytes:
    """One address as bytes, field by field, the way its standard writes it.

    This is the reference the batched form is checked against: it is what a layout
    test pins against the specification, so requiring the two to agree is what
    carries that agreement over to the form every component actually calls.

    Fields are not width-bounded here, where `encode_batch` bounds an integer one
    by its array lane.
    """
    return b"".join(
        _to_byte(int(value), width)
        for value, (_, width) in zip(values, slots, strict=True)
    )


def encode_batch(
    values: Sequence[Field], slots: tuple[Slot, ...]
) -> np.ndarray | Array:
    """A batch of addresses as uint8 `[rows, size(slots)]`.

    Every field becomes its own big-endian bytes and the address is one gather out
    of them, so an address costs no Python of its own. Fields that are single
    values broadcast across the batch, which is the common shape: a fixed layer and
    type against a column of node indices. A byte field broadcasts on the same
    rule, from one row rather than from a scalar, since its trailing axis is the
    slot's width rather than a second address.

    **One gather, never a scatter.** The obvious shape — write each field into its
    slot of a zeroed buffer — is an in-place assignment on the host and a scatter
    on a device, and XLA serializes a large GPU scatter. Reading the same mapping
    backwards makes it a gather: every output byte names the source byte it comes
    from, and the ones a slot leaves empty name a zero column.

    Host in, host out: a batch of concrete fields is packed with numpy and never
    touches a device. The same expression runs on traced fields, so there is one
    implementation rather than a host one and a device one that must agree.
    """
    count = rows(values)
    xnp = arrays.namespace(*values)
    blocks = _field_bytes(values, count, xnp)
    if xnp is np:
        _reject_overflow(blocks, slots)
    widths = tuple(block.shape[1] for block in blocks)
    source = xnp.concatenate([*blocks, xnp.zeros((count, 1), dtype=xnp.uint8)], axis=-1)
    return source[:, _byte_plan(slots, widths)]


def with_field(
    encoded: np.ndarray | Array, slots: tuple[Slot, ...], field: int, value: int
) -> np.ndarray | Array:
    """The same encoded addresses with one field replaced, in every row at once.

    What a caller addressing a run of positions that differ in a single field
    needs. A WOTS+ chain step advances the hash address and holds the layer, tree,
    key pair, chain and type across the whole walk, and RFC 8391's key and bitmask
    are one position under a counter — so both build one batch and vary a word of
    it, where encoding each position separately rebuilds every field it shares with
    the last one.

    The field takes **one value for the whole batch**, which is what makes this
    cheaper than an encode rather than another form of it: the bytes are `encode`'s,
    written on the host, so a field too wide for its slot raises here for the same
    reason it does there, and splicing them in costs the same handful of array
    operations whatever the batch is.

    Host in, host out, and traced in, traced out — the same rule `encode_batch`
    follows, since this is a caller's next step after it.
    """
    width = size(slots)
    shape = np.shape(encoded)
    if shape[-1] != width:
        raise ValueError(f"an encoded address is {width} bytes, got {shape[-1]}")
    xnp = arrays.namespace(encoded)
    start = size(slots[:field])
    _, slot_width = slots[field]
    replacement = xnp.broadcast_to(
        xnp.asarray(np.frombuffer(encode((value,), (slots[field],)), dtype=np.uint8)),
        (*shape[:-1], slot_width),
    )
    # A field at either end of the layout leaves no bytes on that side, and an
    # empty piece is dropped rather than concatenated: this runs once per step of
    # every chain, so the operation it would cost is one per step.
    pieces = [encoded[..., :start], replacement, encoded[..., start + slot_width :]]
    return xnp.concatenate([piece for piece in pieces if piece.shape[-1]], axis=-1)


def _field_bytes(
    values: Sequence[Field], count: int, xnp: Any
) -> list[np.ndarray | Array]:
    """Each field as `[count, width]` big-endian bytes, in the order given.

    A byte field is already those bytes, broadcast to the batch the way an integer
    field is — a caller holding one value for every row passes one row, which is
    what `rows` already counts it as. The integer fields are taken **together**
    — stacked into one table and shifted apart in one expression — rather than a
    field at a time. Six fields done separately is six times the array calls for
    the same bytes, and this is the module that exists so an address costs no
    Python of its own.

    `width` for an integer field is what its lane holds, which is where the 32-bit
    ceiling under a tracer comes from.
    """
    numbers = [
        index for index, value in enumerate(values) if not bytestring.is_bytes(value)
    ]
    blocks: dict[int, np.ndarray | Array] = {
        index: xnp.broadcast_to(
            xnp.asarray(value, dtype=xnp.uint8), (count, np.shape(value)[1])
        )
        for index, value in enumerate(values)
        if bytestring.is_bytes(value)
    }
    if numbers:
        table = _integer_table([values[index] for index in numbers], count, xnp)
        exploded = _big_endian_bytes(table, xnp)
        for slot, index in enumerate(numbers):
            blocks[index] = exploded[:, slot, :]
    return [blocks[index] for index in range(len(values))]


def _big_endian_bytes(table: np.ndarray | Array, xnp: Any) -> np.ndarray | Array:
    """`[count, fields]` integers -> `[count, fields, itemsize]` big-endian bytes.

    Both backends reinterpret the bytes rather than computing them, which is the
    difference between free and not: shifting each byte out arithmetically is
    `itemsize` multiplies over every element of the table, and this module encodes
    hundreds of thousands of addresses per verification.

    They spell it differently, and the endianness is why. numpy names the byte
    order in the dtype, so a big-endian view comes out in the standard's order.
    XLA's bitcast has no such notion and yields host order — little-endian
    everywhere this runs — so the last axis is reversed back.
    """
    if xnp is np:
        return table.astype(">u8").view(np.uint8).reshape(*table.shape, 8)
    return frx.lax.bitcast_convert_type(table, fnp.uint8)[..., ::-1]


def _integer_table(values: Sequence[Field], count: int, xnp: Any) -> np.ndarray | Array:
    """The integer fields broadcast into one `[count, len(values)]` table.

    One dtype for all of them, so the byte explosion above is a single expression.
    Casting each field before the stack rather than after is what keeps a signed
    column beside an unsigned one from promoting the pair to float64, which would
    silently round anything above 2^53 — and a tree address reaches 2^63.
    """
    arrays = []
    for value in values:
        column = xnp.asarray(value)
        if column.dtype.kind not in "iu":
            raise ValueError(
                f"address fields are integers, got dtype {column.dtype}; a field "
                f"wider than the lane needs to arrive as bytes"
            )
        # Host-only, for the reason `_reject_overflow` gives: it reads the value.
        if xnp is np and column.dtype.kind == "i" and column.min() < 0:
            raise ValueError(f"address fields are unsigned, got {column.min()}")
        arrays.append(column)
    unsigned = fnp.uint32 if xnp is fnp else np.uint64
    return xnp.stack(
        [xnp.broadcast_to(column, (count,)).astype(unsigned) for column in arrays],
        axis=-1,
    )


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
def _byte_plan(slots: tuple[Slot, ...], widths: tuple[int, ...]) -> np.ndarray:
    """Which source byte each output byte comes from — one index per output byte.

    The mapping is a constant of the encoding and the fields' widths, so
    precomputing it turns the whole encode into one gather, which is what keeps a
    small batch from costing more in array overhead than it saves in Python.

    Source bytes run field by field in the order the fields were given, and the
    index one past the last names a zero column. A slot wider than the bytes its
    field carries — FIPS 205 gives the tree address twelve — takes that zero for
    its leading bytes, which are zero by construction.
    """
    empty = sum(widths)
    plan: list[int] = []
    starts = np.cumsum((0, *widths))
    for field, (_, width) in enumerate(slots):
        for index in range(width):
            significance = width - 1 - index  # bytes below this one in the slot
            plan.append(
                starts[field] + widths[field] - 1 - significance
                if significance < widths[field]
                else empty
            )
    return np.array(plan)


def _reject_overflow(blocks: Sequence[np.ndarray], slots: Sequence[Slot]) -> None:
    """Refuse a field too wide for the slot it is written into.

    Host-only, and the caller enforces that by not calling it otherwise: this reads
    the bytes, and a traced field has none to read. So it is a host-path guarantee
    rather than an invariant of the encoding, and a traced caller bounds its own
    fields — see the module docstring. All or nothing per call, since one traced
    field makes every block traced.

    A field whose slot is at least as wide as its own bytes cannot overflow one, so
    the check is the bytes the slot has no room for being zero.
    """
    for field, (name, width) in enumerate(slots):
        block = blocks[field]
        if width >= block.shape[1]:
            continue
        dropped = block[:, : block.shape[1] - width]
        if dropped.any():
            row = int(np.argmax(dropped.any(axis=1)))
            raise ValueError(
                f"{name} does not fit in {width} bytes: "
                f"{int.from_bytes(bytes(block[row]), 'big')}"
            )
