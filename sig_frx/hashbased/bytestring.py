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


def namespace(*values: ByteString) -> Any:
    """Which array namespace these values belong to.

    Host values stay on numpy rather than being lifted onto a device, which is
    what keeps the signing path — one signature, all concrete — from paying for
    a dispatch per operation.
    """
    return fnp if any(isinstance(value, Array) for value in values) else np


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
    """
    if bits > 32:
        raise ValueError(f"{bits} bits do not fit the column this reads into")
    span = -(-bits // 8)
    tail = values[:, values.shape[-1] - span :]
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
    moved = (
        xnp.concatenate([_zeros(values, whole, xnp), values[:, : width - whole]], -1)
        if whole
        else values
    )
    if not part:
        return moved
    # Each byte takes its own high bits down and the bits that fell out of the
    # byte above it, which is the byte to its left in big-endian order.
    carried = xnp.concatenate([_zeros(values, 1, xnp), moved[:, :-1]], -1)
    return ((moved >> part) | (carried << (8 - part))).astype(np.uint8)


def _zeros(like: ByteString, width: int, xnp: Any) -> ByteString:
    return xnp.zeros((like.shape[0], width), dtype=np.uint8)


@lru_cache(maxsize=None)
def _byte_mask(width: int, bits: int) -> np.ndarray:
    """Which bits of each byte survive keeping the low `bits` of a `width`-byte row."""
    mask = np.zeros(width, dtype=np.uint8)
    for index in range(width):
        reaches = bits - 8 * (width - 1 - index)  # bits of this byte the value has
        mask[index] = (
            0 if reaches <= 0 else 0xFF if reaches >= 8 else (1 << reaches) - 1
        )
    return mask
