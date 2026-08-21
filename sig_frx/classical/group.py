# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The encoding-side plumbing the classical substrates hold in common.

`secp.py` and `edwards.py` own their curves' constants, codecs, and dtype
pairings — the parts their standards fix. What is left here is arithmetic
neither standard fixes: the constant-exponent field power both square-root
recoveries run on, plus the bytewise range compare and the weighted
byte-to-field sum, which only the Edwards encodings call today (SEC 1 is
big-endian and compares its bounds as host integers) but which are written
against neither curve. The group law itself lives in the curated zk_dtypes
point types (fractalyze/sig-frx#139 for the Weierstrass family, #36 for the
Edwards one), so nothing here walks scalar bits or selects points anymore —
and with the traced path gone from both substrates, nothing here dispatches
on a namespace either.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from frx.typing import ArrayLike


def bytes_below(data: Any, bound: int, *, byteorder: Literal["little", "big"]) -> Any:
    """Whether `[..., 32]` bytes name an integer `< bound`, elementwise.

    Bytewise lexicographic compare — the first differing byte decides — so the
    order is read off the encoding, where it still exists; a field element has
    already lost it. `byteorder` names which end the most significant byte
    lives at, matching `int.to_bytes`.
    """
    bound_bytes = bound.to_bytes(32, byteorder)
    indices = range(32) if byteorder == "big" else reversed(range(32))
    verdict = np.zeros(data.shape[:-1], dtype=np.int32)
    for i in indices:
        diff = np.sign(data[..., i].astype(np.int32) - np.int32(bound_bytes[i]))
        verdict = np.where(verdict != 0, verdict, diff)
    return verdict == -1


def pow_const(curve: Any, base: ArrayLike, exponent: int) -> Any:
    """`base^exponent` in the curve's base field, for a static exponent.

    Square-and-multiply with the branches decided at trace time — the exponent
    is a Python integer, so the unrolled ~256 squarings compile once per
    exponent and the value path stays pure field arithmetic.
    """
    acc = base * np.array(0, dtype=curve.field) + curve.one
    for bit in bin(exponent)[2:]:
        acc = acc * acc
        if bit == "1":
            acc = acc * base
    return acc


def field_from_bytes(weights: np.ndarray, data: ArrayLike) -> Any:
    """`[..., 32]` bytes as field elements, summed by the curve's weights.

    `weights` carries both the field dtype and the byte order (`256^i mod p`
    in the encoding's significance order), so this is one weighted sum for
    either endianness. The reduction is the field's, which means a value at or
    above the modulus wraps silently — a caller enforcing an encoding's range
    bound checks the bytes first with `bytes_below`, where the order still
    exists.
    """
    lanes = np.asarray(data).astype(np.int32).astype(weights.dtype)
    return (lanes * weights).sum(axis=-1)
