# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Big-endian byte strings read as numbers, for values wider than an array lane.

FIPS 205's hypertree index is 64 bits at the tallest parameter sets, and an
integer array lane is 32 under a tracer — frx runs without x64, so `uint64`
becomes `uint32` and a larger value truncates without raising. An index that
truncates addresses the wrong subtree while staying perfectly self-consistent,
which is the failure this package's conventions are written against.

So the index is never made into a number. It arrives as bytes — a slice of the
message digest — and it is consumed as bytes, since §4.2's tree address is a byte
slot. §4.1 defines `toInt` and `toByte` as a pair for the same reason: the value's
type is a byte string, and the integer was the host implementation's convenience.

What the hypertree actually asks of it is small: keep the low bits, read the low
bits as a number, and shift right. All three are static schedules — the widths
come from the parameter set rather than from the data — so each is a handful of
array operations over the whole batch.

`low_bits` is where a number is allowed, and only because the values it reads are
small by construction: a leaf index is `h'` bits, at most 9 at any defined set.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, TypeAlias

import frx.numpy as fnp
import numpy as np
from frx import Array

# A big-endian byte string per row: `[rows, width]` uint8, host or traced.
ByteString: TypeAlias = np.ndarray | Array

# The lane `low_bits` reads into. Deliberately the narrow one: it is what frx
# gives without x64, so a value this refuses is one no traced caller could hold.
_COLUMN_BITS = 32


def namespace(*values: object) -> Any:
    """Which array namespace these values belong to.

    Host values stay on numpy rather than being lifted onto a device, which is
    what keeps the signing path — one signature, all concrete — from paying for
    a dispatch per operation. `adrs_encoding` asks the same question of address
    fields, so the rule lives here rather than once per module.
    """
    return fnp if any(isinstance(value, Array) for value in values) else np


def is_bytes(value: int | ByteString) -> bool:
    """Whether this value is a byte string rather than a number.

    Rank two **and** uint8, and the dtype is half the test on purpose: without it
    a two-dimensional integer array — a caller who reshaped wrongly — would read
    as bytes and be consumed silently.
    """
    return len(np.shape(value)) == 2 and getattr(value, "dtype", None) == np.uint8


def mask_to(values: ByteString, bits: int) -> ByteString:
    """Keep the low `bits` of each row, zeroing everything above them.

    The reduction FIPS 205 writes as `mod 2^bits`, done where the value lives. A
    digest slice is byte-rounded, so it carries up to seven bits more than the
    index does and those bits are not part of it.
    """
    return values & _byte_mask(values.shape[-1], bits)


def low_bits(values: ByteString, bits: int) -> ByteString:
    """The low `bits` of each row as a uint32 column.

    Only for values that fit one: a leaf index is `h'` bits, at most 9 at any
    defined parameter set. The bytes above what `bits` reaches are never read, so
    a caller does not have to mask first.

    Reads from the **end** of the row, where `wots.base_2b` reads digits from the
    front of a stream at a static bit offset. Neither can stand in for the other:
    at `h' = 3` this wants the bottom three bits and `base_2b` gives the top three.
    """
    if bits > _COLUMN_BITS:
        raise ValueError(f"{bits} bits do not fit the column this reads into")
    span = -(-bits // 8)
    tail = values[:, -span:]
    column = tail[:, 0].astype(np.uint32)
    for index in range(1, span):
        column = (column << 8) | tail[:, index].astype(np.uint32)
    return column & np.uint32((1 << bits) - 1)


def shift_right(values: ByteString, bits: int) -> ByteString:
    """Each row shifted right by `bits`, staying the same width.

    Climbing a hypertree layer consumes `h'` bits, and `h'` is 3 at SHA2-128f —
    so this is not a byte move. It is a whole-byte slide for the bytes it does
    cover and a two-slice recombination for the remainder, both static.
    """
    xnp = namespace(values)
    width = values.shape[-1]
    whole, part = divmod(bits, 8)
    if whole >= width:
        return xnp.zeros_like(values)
    # One zero byte more than the slide needs, so that each byte and the one above
    # it — its left neighbour, big-endian — are two slices of the same array.
    padded = xnp.concatenate(
        [
            xnp.zeros((values.shape[0], whole + 1), dtype=np.uint8),
            values[:, : width - whole],
        ],
        axis=-1,
    )
    if not part:
        return padded[:, 1:]
    return (padded[:, 1:] >> part) | (padded[:, :-1] << (8 - part))


@lru_cache(maxsize=None)
def _byte_mask(width: int, bits: int) -> np.ndarray:
    """Which bits of each byte survive keeping the low `bits` of a `width`-byte row."""
    reaches = np.clip(bits - 8 * np.arange(width - 1, -1, -1), 0, 8)
    return ((1 << reaches) - 1).astype(np.uint8)
