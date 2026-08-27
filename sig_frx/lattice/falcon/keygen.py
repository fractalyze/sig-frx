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

from functools import lru_cache, partial
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from frx import lax
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


# -- the base case: Algorithm 6 at degree 1 -----------------------------------


def _is_even(a: Any) -> Any:
    """Whether the value is even, which is one bit of one limb."""
    return fnp.asarray(a)[..., 0] & np.uint32(1) == np.uint32(0)


def _is_zero(a: Any) -> Any:
    return fnp.all(fnp.asarray(a) == np.uint32(0), axis=-1)


def _select(condition: Any, when: Any, otherwise: Any) -> Any:
    """`where` over whole numbers: the predicate is per value, not per limb."""
    return fnp.where(fnp.asarray(condition)[..., None], when, otherwise)


def _negate(a: Any) -> Any:
    """`-a`, which two's complement makes a subtraction from zero."""
    return bigint.sub(fnp.zeros_like(a), a)


def _halve_cofactors(p: Any, q: Any, x: Any, y: Any) -> tuple[Any, Any]:
    """Halve `(p, q)` so that `p·x + q·y` halves with it.

    The cofactors are only both even about half the time, and a pair that is not
    cannot be halved as it stands. `(p + y, q - x)` is the same value of
    `p·x + q·y` written through a pair that can be — the two corrections cancel,
    `y·x - x·y` — and it is even in every case *because* one of `x`, `y` is odd,
    which is [`base_case`](#base_case)'s precondition and the whole of what that
    precondition is for. With `p` odd and `q` even the evenness of `p·x + q·y`
    forces `x` even and so `y` odd; the mirrored case forces `x` odd; and with
    both odd it forces `x` and `y` to share a parity, which the precondition
    makes odd. This is the binary extended GCD's halving step — Menezes, van
    Oorschot and Vanstone, *Handbook of Applied Cryptography*, Algorithm 14.61.
    """
    both_even = _is_even(p) & _is_even(q)
    shifted_p = _select(both_even, p, bigint.add(p, y))
    shifted_q = _select(both_even, q, bigint.sub(q, x))
    return (
        bigint.shift_right(shifted_p, 1, signed=True),
        bigint.shift_right(shifted_q, 1, signed=True),
    )


@partial(frx.jit, static_argnums=(1,))
def _solve(state: tuple[Any, ...], trips: int, x: Any, y: Any) -> tuple[Any, ...]:
    """The extended binary GCD's loop, as one program — Algorithm 14.61's steps.

    `frx.jit` because the loop is the point: eager, this is `trips` dispatches —
    25,210 of them at `n = 1024` — of a body doing real work on 421 limbs, and
    the whole reason #26 chose a bounded-precision device form over the host is
    that traced it compiles into one executable instead. `trips` is static so
    that a program is built once per width, the way
    [`mldsa.sampling`](../mldsa/sampling.py) compiles its own sequential sampler.

    **Each of the three cases is computed once, not once per branch it serves.**
    Halving `(a, b)` and halving `(c, d)` never both happen, so the operands are
    selected first and the one result is scattered back; the same holds for
    `u - v` against `v - u` and for the cofactor pairs that follow them. A
    `where` is elementwise over the limbs where an `add` or `sub` is an
    `associative_scan` over all of them, so folding the pairs takes the body from
    ten scans to five and pays eight selects for it.
    """

    def step(_: Any, state: tuple[Any, ...]) -> tuple[Any, ...]:
        u, v, a, b, c, d = state
        # `u` odd is the discriminant: zero is even, so an exhausted loop lands
        # on no branch at all and every register holds.
        u_odd = ~_is_even(u)
        both_odd = u_odd & ~_is_even(v)
        halve_u = ~u_odd & ~_is_zero(u)
        halve_v = u_odd & ~both_odd
        take_u = both_odd & bigint.at_least(u, v)
        take_v = both_odd & ~take_u

        # One halving. Whichever pair is live is the one selected in; when
        # neither is, the result is discarded by both selects below.
        halved_p, halved_q = _halve_cofactors(
            _select(halve_u, a, c), _select(halve_u, b, d), x, y
        )
        # One subtraction per register, oriented by which operand is larger.
        difference = bigint.sub(_select(take_u, u, v), _select(take_u, v, u))
        reduced_p = bigint.sub(_select(take_u, a, c), _select(take_u, c, a))
        reduced_q = bigint.sub(_select(take_u, b, d), _select(take_u, d, b))

        return (
            _select(halve_u, bigint.shift_right(u, 1), _select(take_u, difference, u)),
            _select(halve_v, bigint.shift_right(v, 1), _select(take_v, difference, v)),
            _select(halve_u, halved_p, _select(take_u, reduced_p, a)),
            _select(halve_u, halved_q, _select(take_u, reduced_q, b)),
            _select(halve_v, halved_p, _select(take_v, reduced_p, c)),
            _select(halve_v, halved_q, _select(take_v, reduced_q, d)),
        )

    return lax.fori_loop(0, trips, step, state)


def base_case(f: ArrayLike, g: ArrayLike, bits: int) -> tuple[Any, Any, Any]:
    """Algorithm 6 at degree 1: `u·f - v·g = gcd(f, g)`, over signed limbs.

    The descent bottoms out at one coefficient apiece and the recursion turns
    around here — `(F, G) = (v·q, u·q)` is what satisfies `fG - gF = q` at this
    level, so the Bezout pair is the whole deliverable. The gcd comes back with
    it because Algorithm 6 redraws the key when it is not 1, and a caller that
    only learned "not 1" would have to recompute this to find out by how much.

    **A both-even pair comes back with `gcd` zero**, which is not a gcd and is
    unmistakably not one. The halving step needs one operand odd, and a pair that
    is not has gcd at least 2 — a key Algorithm 6 redraws — so the loop declines
    it rather than carrying a common-power-of-two prefix for a case its caller
    throws away. Reported rather than left to the caller because the wrong answer
    here is *quiet*: the first halving drops a factor of two the loop never
    restores, so a `gcd` computed through it is incorrect rather than merely
    even, and a caller testing `gcd != 1` would be reading a number that no
    longer means what it says.

    Everything after the first step is safe without a check. A subtract leaves
    the other operand odd and a halving does not touch it, so once one of the two
    is odd it stays that way for the whole loop.

    The trip count is fixed at `4·bits + 2`, which is a proof rather than a
    sample. Write `phi` for `bits(u) + bits(v)`: a halving drops it by exactly
    one, so there are at most `2·bits` halvings; a subtract leaves an even value
    and is therefore always followed by one, so there are at most a halving's
    worth plus one of those. Randomly drawn operands need about `2.2·bits` and
    the reserve is the rest — a count fitted to what was sampled would return a
    `v` that is not the gcd on the keys that need more, and nothing would raise.
    A tracer cannot stop on a value, so the loop runs its bound every time and
    holds `u == 0` as a no-op once it converges.
    """
    limbs = bigint.limb_count(bits + 3)
    wide_f, wide_g = bigint.sign_extend(f, limbs), bigint.sign_extend(g, limbs)

    # The loop runs on magnitudes; the identity is over the signed operands, so
    # the signs come back at the end rather than being dropped.
    f_negative, g_negative = bigint.is_negative(wide_f), bigint.is_negative(wide_g)
    x = _select(f_negative, _negate(wide_f), wide_f)
    y = _select(g_negative, _negate(wide_g), wide_g)

    one = fnp.broadcast_to(fnp.asarray(bigint.to_limbs(1, limbs)), x.shape)
    zero = fnp.zeros_like(one)
    # `v` is where the gcd lands: the loop runs until `u` is zero.
    _, gcd, _, _, c, d = _solve((x, y, one, zero, zero, one), 4 * bits + 2, x, y)

    # A both-even pair is the one input the halving step has no answer for, and
    # the answer it produces instead is not merely coarse: the first halving
    # drops a factor of two the loop never restores, so `gcd` would come back
    # *wrong* rather than merely even. Reported as zero, which is not a gcd and
    # is unmistakably not one, because the caller's question is whether it is 1.
    both_even = _is_even(x) & _is_even(y)
    gcd = _select(both_even, zero, gcd)

    # `c·x + d·y = gcd` over magnitudes becomes `u·f - v·g = gcd` over the
    # operands: a negative operand flips its cofactor's sign, and `v` carries the
    # subtraction the identity is written with.
    u = _select(f_negative, _negate(c), c)
    v = _select(g_negative, d, _negate(d))
    return u, v, gcd
