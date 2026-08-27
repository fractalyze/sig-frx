# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Falcon key generation — §3.8.2's tower-of-fields descent.

Solving `fG - gF = q` (Algorithm 6) recurses on the field norm: `f` of degree
`n` becomes `N(f) = f_e² - x·f_o²` of degree `n/2`, down to degree 1 where an
extended GCD closes it. This module is that descent. The base case, the lift
back up, and Babai's reduction are the rest of
[#26](https://github.com/fractalyze/sig-frx/issues/26).

## Coefficients outgrow every lane on the way down, which sets the shape

Squaring doubles a coefficient's width at every level, so `f`'s 4-bit
coefficients reach **9,427 bits** by the bottom at `n = 1024` — measured on #26.
[`bigint`](bigint.py) is what holds them, and this module is where its two
representations meet a polynomial:

- **residues carry the multiplication.** A negacyclic product is independent per
  channel, so the whole convolution is one batched array operation.
- **limbs carry the value between levels.** This is the part that is not
  optional: the next level needs *more* channels than the current one, and
  adding a channel to a residue representation requires the value it represents.
  So each level reconstructs, and `from_rns` is why the descent costs what it
  costs.

Everything is signed. `f` and `g` are drawn from a discrete Gaussian centered at
zero, and the difference in `N` is signed even where its operands are not — so
both bridge directions run with `signed=True` throughout, and a `bits` here is
always a bound on `|coefficient|` rather than on the coefficient.

## The width bound is proved, not measured

`|N(f)_c| <= 2·(m/2)·2^(2w)` for `|f_i| < 2^w` and degree `m`: the negacyclic
convolution sums `m/2` products and the subtraction doubles the result. So a
level's bound is `2w + log2(m/2) + 1` and nothing in the descent can exceed it.

The width is static either way — a tracer cannot size a shape from a value, so
measuring a key's actual widths and allocating to them is not on the table. What
is on the table is *which* static width, and the reference implementation
answers differently: `MAX_BL_SMALL` is a table of widths measured over thousands
of random keys, plus a margin. Tighter, and exceeded with some probability,
where exceeding it wraps a lane in silence.

So this is a choice between two constants rather than between a constant and a
measurement, and it goes to the one that cannot be wrong. The slack is real and
worth knowing: against the widths a random key actually reaches, the bound runs
**1.55x at the first level and converges to 2.10x** by the bottom, because it
assumes every coefficient is at its extreme where random signs cancel. That is
roughly twice the channels and twice the limbs, on an operation that runs once
per key — and it doubles what the base case below has to grind through, which
is where it is actually felt.

## Layouts, which differ between the two forms on purpose

- residues: `[..., channels, degree]`, the coefficient axis **trailing**, because
  that is the axis a convolution runs along.
- limbs: `[..., degree, limbs]`, the limb axis trailing, because that is the axis
  every [`bigint`](bigint.py) operation runs along.

Neither can be trailing in both, so the bridge transposes. Writing the two the
same way round would put the transpose inside the convolution instead, where it
would run once per level per channel rather than once per level.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import frx.numpy as fnp
import numpy as np
from frx.typing import ArrayLike

from sig_frx.lattice.falcon import bigint


def norm_bits(bits: int, degree: int) -> int:
    """The proved bound on `|N(f)_c|` for `|f_i| < 2^bits` at `degree`.

    `degree` is the input's, so the convolution runs at `degree/2` and sums that
    many products; the trailing `+ 1` is the subtraction in `f_e² - x·f_o²`.
    """
    if degree < 2 or degree & (degree - 1):
        raise ValueError(f"degree {degree} is not a power of two above 1")
    return 2 * bits + (degree // 2 - 1).bit_length() + 1


@lru_cache(maxsize=None)
def _convolution_indices(degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Which coefficient of `b` meets each coefficient of `a`, and with what sign.

    `out_c = Σ_i a_i · b_(c-i mod m)`, negated exactly when the index wrapped —
    that sign is the whole difference between this ring and the cyclic one.
    Host tables, because `degree` arrives as a Python integer.
    """
    positions = np.arange(degree)
    source = (positions[:, None] - positions[None, :]) % degree
    wrapped = positions[None, :] > positions[:, None]
    return source.astype(np.int32), wrapped


def negacyclic_mul(a: ArrayLike, b: ArrayLike, mods: ArrayLike) -> Any:
    """`(a · b) mod (x^m + 1)`, per residue channel, over `[..., channels, m]`.

    Schoolbook rather than an NTT, and not for want of trying: `frx.lax.ntt`
    needs a `2m`-th root of unity in the channel's field, which means a modulus
    congruent to `1 mod 2m`. Under `2^15` and at `m = 1024` there are only a
    handful of such primes and the descent needs hundreds of channels, so an
    NTT-friendly channel set does not exist at this width.

    **Each product is reduced before it is summed.** Two residues multiply to
    under `2^30`, which a lane holds — but `m` of them do not, and the sum is
    what would leave the lane rather than the product.
    """
    left, right = fnp.asarray(a), fnp.asarray(b)
    degree = left.shape[-1]
    source, wrapped = _convolution_indices(degree)
    modulus = fnp.asarray(mods)[..., None, None]
    gathered = fnp.take(right, source, axis=-1)
    terms = (left[..., None, :] * gathered) % modulus
    terms = fnp.where(wrapped, (modulus - terms) % modulus, terms)
    return fnp.sum(terms, axis=-1, dtype=np.uint32) % fnp.asarray(mods)[..., None]


def field_norm(coefficients: ArrayLike, bits: int) -> tuple[Any, int]:
    """`N(f) = f_e² - x·f_o²`, one level of Algorithm 6's descent.

    Takes and returns limbs — `[degree, limbs]` in, `[degree/2, limbs']` out —
    because the level below needs channels this level does not have, and only
    the value can supply them.
    """
    values = fnp.asarray(coefficients)
    degree = values.shape[0]
    result_bits = norm_bits(bits, degree)
    channels, limbs = bigint.signed_shape(result_bits)
    mods = bigint.moduli(channels)

    # `[degree/2, limbs]` to `[channels, degree/2]`: the bridge is where the two
    # trailing-axis conventions meet, so it is where the transpose belongs.
    def residues(part: Any) -> Any:
        return fnp.swapaxes(bigint.to_rns(part, channels, signed=True), -1, -2)

    even_part, odd_part = residues(values[0::2]), residues(values[1::2])
    even = negacyclic_mul(even_part, even_part, mods)
    odd = negacyclic_mul(odd_part, odd_part, mods)

    # `-x·odd` in `Z[x]/(x^(m/2) + 1)`: every coefficient shifts up one place and
    # the one that falls off the top comes back at the bottom with its sign
    # flipped — so subtracting it means *adding* that single entry.
    modulus = mods[:, None]
    shifted = fnp.concatenate(
        [odd[..., -1:], (modulus - odd[..., :-1]) % modulus], axis=-1
    )
    combined = (even + shifted) % modulus
    result = bigint.from_rns(
        fnp.swapaxes(combined, -1, -2), channels, limbs, signed=True
    )
    return result, result_bits


def descend(coefficients: ArrayLike, bits: int, levels: int) -> list[tuple[Any, int]]:
    """`levels` applications of [`field_norm`](#field_norm), innermost last.

    The whole left-hand side of Algorithm 6: what the base case receives is the
    last entry, and what the lift back up consumes is every entry in reverse.
    """
    current: Any = fnp.asarray(coefficients)
    out: list[tuple[Any, int]] = []
    for _ in range(levels):
        current, bits = field_norm(current, bits)
        out.append((current, bits))
    return out


def to_limbs(coefficients: ArrayLike, bits: int) -> Any:
    """Signed host coefficients as `[degree, limbs]`, in two's complement.

    The descent's entry point: `f` and `g` arrive as small signed integers from
    the Gaussian draw and have to be the wide form before anything else. The
    budget only has to hold the sign and the magnitude — a level sizes its own
    output, and [`bigint.to_rns`](bigint.py) reads whatever budget it is handed.
    """
    values = np.asarray(coefficients, dtype=np.int64)
    limbs = bigint.limb_count(bits + 2)
    span = 1 << (limbs * bigint.LIMB_BITS)
    packed = np.stack(
        [bigint.to_limbs(int(value) % span, limbs) for value in values.reshape(-1)]
    )
    return fnp.asarray(packed.reshape(*values.shape, limbs))
