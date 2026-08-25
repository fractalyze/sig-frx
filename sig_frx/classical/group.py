# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The plumbing the classical substrates hold in common.

`secp.py` and `edwards.py` own their curves' constants, codecs, and dtype
pairings — the parts their standards fix. What is left here is arithmetic
neither standard fixes: the constant-exponent field power both square-root
recoveries run on, plus the bytewise range compare and the weighted
byte-to-field sum, which only the Edwards encodings call today (SEC 1 is
big-endian and compares its bounds as host integers) but which are written
against neither curve. The group law itself lives in the curated zk_dtypes
point types (fractalyze/sig-frx#139 for the Weierstrass family, #36 for the
Edwards one), so nothing here walks scalar bits or selects points anymore.
Nothing here dispatches on a namespace either — these are operator-generic
over the values their callers have already placed, so each follows whichever
namespace it is handed. `pow_const` is the one that now sees both: `secp.py`'s
lift places its coordinate batch before calling `sqrt`, so the ladder runs
traced above that threshold and on the host below it. Keep it that way — a
helper here that reached for numpy directly, or read a value back, would
break the placed path only at the batch sizes the known-answer tests never
reach.

The last two are the aggregate verifiers' shared parts. BIP-340 and ZIP-215
specify unrelated schemes over unrelated curves, and both check a batch the
same way: fold it into one point sum against coefficients nobody could
predict before the batch existed. `sum_points` and `batch_coefficients` are
that shape with the curve taken out of it — a fold needs `+` and an element
to pad with, and a coefficient needs a modulus and a hash. Keeping them here
rather than one per scheme is what stops two soundness arguments from
drifting into two different constructions.
"""

from __future__ import annotations

from collections.abc import Callable
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


def _window_digits(exponent: int, window: int) -> list[int]:
    """`exponent` in base `2^window`, most significant digit first.

    The leading digit is non-zero, which is what lets the ladder below seed
    the accumulator from the table instead of from one.
    """
    mask = (1 << window) - 1
    digits = []
    while exponent:
        digits.append(exponent & mask)
        exponent >>= window
    return digits[::-1]


def pow_const(curve: Any, base: ArrayLike, exponent: int, *, window: int = 4) -> Any:
    """`base^exponent` in the curve's base field, for a static exponent.

    A fixed-window ladder: `base^1 … base^(2^w - 1)` is tabulated once, then
    the exponent is consumed `w` bits at a time. Both callers are square
    roots whose exponent has almost every bit set — `(p+1)/4` on the SEC
    curves, `(p-5)/8` on ed25519 — which is the bit-at-a-time method's worst
    case, so the window is worth about a third of the multiplications there
    (~325 against ~501 for a 254-bit exponent at `w = 4`).

    `w = 4` is the floor for a 256-bit exponent; 3 and 5 both cost a few
    percent more, because a wider window pays for its table faster than it
    saves on digits. It is a parameter because the cost only holds for
    exponents of that size, not because a caller is expected to tune it.

    The exponent is a Python integer, so the digits, the table size, and the
    ladder's branches are all decided at trace time exactly as the bit loop's
    were: the kernel still unrolls and compiles once per exponent, and the
    value path stays pure field arithmetic.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    if exponent < 0:
        raise ValueError("pow_const does not invert; exponent must be >= 0")
    # `curve.one` is a scalar, so it is broadcast against `base` to give the
    # empty product the batch's shape.
    if exponent == 0:
        return base * np.array(0, dtype=curve.field) + curve.one

    digits = _window_digits(exponent, window)
    # `table[i]` is `base^(i + 1)`, built only as far as the digits reach.
    table = [base]
    for _ in range(1, max(digits)):
        table.append(table[-1] * base)

    acc = table[digits[0] - 1]
    for digit in digits[1:]:
        for _ in range(window):
            acc = acc * acc
        if digit:
            acc = acc * table[digit - 1]
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


def batch_inverse(values: np.ndarray) -> np.ndarray:
    """`1/values[i]` elementwise, at the cost of **one** inversion: `[B]`.

    Montgomery's trick. The running products give every row the product of
    every *other* row's value, so inverting the total once and multiplying it
    back through leaves each row holding its own inverse. `B` inversions
    become `3B` multiplies and one inversion, which is worth doing because the
    dtype's inversion is the expensive operation and its multiply is not.

    Measured on secp256k1's scalar field at B=1024: 0.10 ms here against
    1.91 ms for a Python loop calling `** -1` per row, and against 1.19 ms for
    the elementwise `a / b` the dtype also offers — that one still inverts `B`
    times, so a caller needing two quotients over one denominator pays for it
    twice and ends up behind the loop it replaced.

    **Every value must be non-zero, and the failure is not local.** A zero
    anywhere sends the whole product to zero, and the dtype's division by zero
    answers zero rather than raising, so *every* row comes back zero — a
    caller that lets one rejected row carry a zero denominator silently
    destroys the verdicts of every valid row beside it. Masked rows therefore
    substitute a one before the chain rather than being filtered out, which
    keeps the batch rectangular and the substitution visible at the call site.
    This is not defensive commentary: it is measured behaviour, and
    `group_test` pins it.

    An empty batch answers empty — `accumulate` has nothing to fold and there
    is no total to invert.
    """
    values = np.asarray(values)
    if values.shape[0] == 0:
        return values
    one = np.array([1], dtype=values.dtype)
    prefix = np.multiply.accumulate(values)
    suffix = np.multiply.accumulate(values[::-1])[::-1]
    total = one / prefix[-1:]
    before = np.concatenate([one, prefix[:-1]])
    after = np.concatenate([suffix[1:], one])
    return before * after * total


def sum_points(points: np.ndarray, identity: np.ndarray) -> np.ndarray:
    """The sum of a `[K]` point batch, by vectorized halving to `[1]`.

    `identity` pads an odd length, and it is a parameter because the two
    substrates disagree about what the neutral element looks like in memory.
    A Jacobian buffer of zeros *is* infinity, so `secp` can pad with
    `np.zeros`; an all-zero extended Edwards point is not the identity but a
    value whose projective compare answers `True` against every point, so
    `edwards` must pad with a real `(0, 1)`. Padding wrongly does not raise
    — it makes a later comparison agree — which is why the choice is the
    caller's to state rather than this function's to guess.
    """
    total = np.asarray(points)
    while total.shape[0] > 1:
        if total.shape[0] % 2:
            total = np.concatenate([total, identity.astype(total.dtype)])
        half = total.shape[0] // 2
        total = total[:half] + total[half:]
    return total


def batch_coefficients(
    order: int, count: int, batch: bytes, *, digest: Callable[[bytes], Any]
) -> list[int]:
    """`count` scalars in `[1, order-1]`, fixed only after all of `batch`.

    What an aggregate check needs of them is that a forger cannot choose a
    batch knowing them: they are derived from a hash of every byte of it,
    with no draw from a generator, so the verdict stays reproducible and no
    implicit randomness enters a seam that forbids it.

    `digest` has no default: it is the caller's scheme that names the hash,
    and a wrong-but-plausible one here would still produce coefficients that
    look fine and agree with nothing.

    The first is 1. One coefficient may be fixed without weakening the
    combination — a single wrong entry is caught by its own residual, and
    two or more still have to satisfy a relation in the remaining scalars —
    and it saves the batch's most common case, `count == 1`, a
    multiplication.
    """
    seed = digest(batch).digest()
    return [1] + [
        1
        + int.from_bytes(digest(seed + index.to_bytes(8, "big")).digest(), "big")
        % (order - 1)
        for index in range(1, count)
    ]
