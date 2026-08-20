# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The curve-family-agnostic plumbing both classical substrates share.

`weierstrass.py` and `edwards.py` each own their formulas, constants, and
encodings — the parts their standards fix. Everything here is what neither
standard fixes and both substrates need identically: byte and bit reshapes,
the bytewise range compare an encoding's rejection rules hang on, the
fixed-shape scalar ladder, and the host readback of a point as Python
integers. It exists because the Edwards substrate arrived as the Weierstrass
one's second consumer (`docs/reference/conventions.md`); the substrates keep
thin named wrappers where a docstring owes the reader curve-specific context.

Functions here are generic over the two point representations by reading only
what both carry: a curve with `field` and `one`, and a point NamedTuple whose
leading fields are coordinates (so an arithmetic select rebuilds it with
`type(point)(*...)`). The batch-axis rule (`weierstrass.py`) applies
throughout.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from frx import lax
from frx.typing import ArrayLike

from sig_frx.arrays import namespace


def bits_of(data: ArrayLike) -> Any:
    """Big-endian bytes as bits, most significant first: `[..., L] -> [..., 8L]`.

    The shape the scalar ladder consumes, produced from wire bytes — a scalar's
    bits come off its encoding, never off a field element, because a 256-bit
    residue has no integer lane to be read back onto.
    """
    xnp = namespace(data)
    data = xnp.asarray(data)
    shifts = np.arange(7, -1, -1, dtype=np.uint8)
    bits = (data[..., :, None] >> shifts) & np.uint8(1)
    return bits.reshape(data.shape[:-1] + (8 * data.shape[-1],))


def int_bits(scalar: int, size: int = 32) -> np.ndarray:
    """A Python integer as `[1, 8·size]` ladder bits, on the host path.

    The `[1]` batch axis is the substrate's B = 1 rule applied at the one
    place host callers keep re-deriving it.
    """
    return bits_of(np.frombuffer(scalar.to_bytes(size, "big"), dtype=np.uint8))[None, :]


def ints_bits(scalars: list[int], size: int = 32) -> np.ndarray:
    """A batch of Python integers as `[B, 8·size]` ladder bits.

    `int_bits`'s batch sibling, for the host paths that assemble per-entry
    scalars and drive one stacked ladder with them.
    """
    return bits_of(
        np.stack(
            [
                np.frombuffer(value.to_bytes(size, "big"), dtype=np.uint8)
                for value in scalars
            ]
        )
    )


def bytes_below(
    xnp: Any, data: Any, bound: int, *, byteorder: Literal["little", "big"]
) -> Any:
    """Whether `[..., 32]` bytes name an integer `< bound`, elementwise.

    Bytewise lexicographic compare — the first differing byte decides — so the
    order is read off the encoding, where it still exists; a field element has
    already lost it. `byteorder` names which end the most significant byte
    lives at, matching `int.to_bytes`.
    """
    bound_bytes = bound.to_bytes(32, byteorder)
    indices = range(32) if byteorder == "big" else reversed(range(32))
    verdict = xnp.zeros(data.shape[:-1], dtype=np.int32)
    for i in indices:
        diff = xnp.sign(data[..., i].astype(np.int32) - np.int32(bound_bytes[i]))
        verdict = xnp.where(verdict != 0, verdict, diff)
    return verdict == -1


def select(curve: Any, flag: Any, when_set: Any, when_clear: Any) -> Any:
    """`flag ? when_set : when_clear`, arithmetically, per batch entry.

    `flag` is a field element in {0, 1}. Arithmetic rather than a `where` so a
    ladder's carried state is field-dtype arithmetic end to end. Rebuilds
    whichever point NamedTuple it was handed.
    """
    keep = curve.one - flag
    return type(when_set)(*(flag * s + keep * c for s, c in zip(when_set, when_clear)))


def ladder(
    curve: Any,
    bits: ArrayLike,
    point: Any,
    *,
    double: Any,
    add: Any,
    identity: Any,
) -> Any:
    """`k·P` over `k`'s bits, most significant first — the shared ladder.

    Double-and-add with a fixed trip count and an arithmetic select, so one
    traced computation serves the whole batch. A scalar wider than the group
    order reduces through the group itself — `k·P = (k mod n)·P` — which is
    why callers hand bits straight off wire bytes with no reduction in front.

    The loop is `lax.fori_loop` under a tracer and a Python loop on the host:
    same body, and the host path never lifts (`conventions.md`). The group law
    arrives as the substrate's `double`/`add`/`identity`, each taking the
    curve first — the only three facts the two curve families do not share.
    """
    xnp = namespace(bits, *point)
    bits = xnp.asarray(bits)
    flags = bits.astype(np.int32).astype(curve.field)
    length = flags.shape[-1]

    def step(flag: Any, acc: Any) -> Any:
        doubled = double(curve, acc)
        added = add(curve, doubled, point)
        return select(curve, flag, added, doubled)

    start = identity(curve, flags[..., 0] * point.x)
    if xnp is np:
        acc = start
        for i in range(length):
            acc = step(flags[..., i], acc)
        return acc
    return lax.fori_loop(
        0, length, lambda i, acc: step(xnp.take(flags, i, axis=-1), acc), start
    )


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


def equal(p: Any, q: Any) -> Any:
    """Whether two projective points name the same element, elementwise.

    Cross-multiplied, so no division: equal iff `X₁Z₂ = X₂Z₁` and
    `Y₁Z₂ = Y₂Z₁`. Reads only `(x, y, z)`, so it serves both representations;
    an extended point's `t` is determined by the other three and adds nothing.
    Sound for everything the complete formulas produce — they never emit a
    degenerate all-zero triple.
    """
    return (p.x * q.z == q.x * p.z) & (p.y * q.z == q.y * p.z)


def field_from_bytes(weights: np.ndarray, data: ArrayLike) -> Any:
    """`[..., 32]` bytes as field elements, summed by the curve's weights.

    `weights` carries both the field dtype and the byte order (`256^i mod p`
    in the encoding's significance order), so this is one weighted sum for
    either endianness. The reduction is the field's, which means a value at or
    above the modulus wraps silently — a caller enforcing an encoding's range
    bound checks the bytes first with `bytes_below`, where the order still
    exists.
    """
    xnp = namespace(data)
    lanes = xnp.asarray(data).astype(np.int32).astype(weights.dtype)
    return (lanes * weights).sum(axis=-1)


def to_affine_ints(point: Any) -> list[tuple[int, int]]:
    """A host batch of projective points back to affine Python integers.

    Reads `(x, y, z)` only, so it serves both representations. Host-only by
    construction — the readback is what a traced value cannot do.
    """
    xs = np.asarray(point.x / point.z).astype(object)
    ys = np.asarray(point.y / point.z).astype(object)
    return [(int(x), int(y)) for x, y in zip(xs, ys)]
