# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Falcon key generation — Algorithm 5's draw, Algorithm 6 under it, and the tree.

Six sections, in the order key generation runs them rather than the order they
were built:

- **Algorithm 5** draws `f` and `g` from a discrete Gaussian and rejects the
  pair unless the basis is short enough and `f` is a unit.
- **the two `Z_q` steps** — Algorithm 4 line 9's public key `h = g·f^{-1}`, and
  (3.35)'s recovery of the `G` that §3.11.5 does not encode. They are the only
  part of key generation that lives where verification does
  ([`arith.py`](arith.py)), and the recovery is a key *loading* step rather than
  one Algorithm 5 runs.
- **the descent** (Algorithm 6's left-hand side) recurses on the field norm:
  `f` of degree `n` becomes `N(f) = f_e² - x·f_o²` of degree `n/2`.
- **the base case** closes it at degree 1, where the NTRU equation is Bezout's.
- **the lift and Babai's reduction** walk back up, each level substituting the
  level below's answer into its own and shortening what that produces.
- **the ffLDL tree** (Algorithms 8 and 9) decomposes the basis's Gram matrix
  down the same tower, and is what the sampler
  ([#27](https://github.com/fractalyze/sig-frx/issues/27)) walks.

The restart loop that drives Algorithm 5 lands with the caller that has a loop
to put it in, which is [`falcon.Falcon.keygen`](falcon.py).

The six share almost nothing: the descent is `bigint`'s residues, the base
case is `bigint`'s limbs under a `fori_loop`, the walk up is a host loop over
`fft`, Algorithm 5 is `fft` and a table built with `decimal`, the two `Z_q`
steps are one opcode each, and the tree is `float64` with no integer in it at
all. They are one module because they are one operation, and the reason to
split would be that the file stops being readable rather than that the sections
differ.

## The two directions are sized differently, and on purpose

Going down, a width is a **bound**: `field_norm` is traced, a tracer cannot
size a shape from a value, and [`norm_bits`](#norm_bits) is the constant that
cannot be wrong. Coming up, a width is **measured**. Two things force it. A
reduced width has no bound to lean on — it is a property of the basis rather
than of the arithmetic, which is why the reference implementation's own table
of widths is sampled — and the bound that does exist runs 2.10x wide by the
bottom, so a lift sized from it would carry four times the channels it needs
on the widest product in the recursion.

What makes measuring available is that the upward pass is driven from the
host: [`reduce`](#reduce) has to read a width back anyway to know where the
next chunk of its quotient sits. So the levels above size themselves from what
the level below actually produced, and the limbs are trimmed on the way up.
The arithmetic does not move — `k·f`, the subtraction and the lift are the wide
operations and all of them stay on device.

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

import decimal
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from frx import lax
from frx.typing import ArrayLike

from sig_frx.arrays import namespace
from sig_frx.lattice.falcon import arith, bigint, fft


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


def _negate(values: ArrayLike) -> Any:
    """`-values` in two's complement, which is what every signed step here means."""
    return bigint.sub(fnp.zeros_like(fnp.asarray(values)), values)


def _widen(values: ArrayLike, limbs: int) -> Any:
    """The same signed value at a wider limb budget.

    Widening a magnitude pads with zeros and widening a two's-complement value
    pads with its sign; they differ exactly where the value is negative, which
    is everywhere in this module.
    """
    narrow = fnp.asarray(values)
    pad = limbs - narrow.shape[-1]
    if pad <= 0:
        return narrow
    fill = fnp.where(bigint.is_negative(narrow)[..., None], bigint.MASK, np.uint32(0))
    return fnp.concatenate(
        [narrow, fnp.broadcast_to(fill, (*narrow.shape[:-1], pad))], axis=-1
    ).astype(np.uint32)


# -- the base case ---------------------------------------------------------


def gcd_budget(bits: int) -> int:
    """Steps the binary GCD needs for two magnitudes under `2^bits`.

    Proved rather than sampled, and the two differ by enough to matter. Each
    halving drops `bits(u) + bits(v)` by one and no subtraction ever raises it,
    so there are at most `2·bits` halvings; a subtraction leaves an odd minus an
    odd, which is even, so the step after one is always a halving and there are
    at most one more of those than there are halvings. Hence `4·bits + 1`.

    The measured worst case is well inside it — 13,454 steps against a bound of
    25,209 over 300 random pairs at `n = 1024` — but sampling is the wrong tool
    here, because a budget that is short does not fail loudly. It returns a
    `(u, v)` that is simply not the Bezout pair, and only the equation check
    downstream would notice. This is the same call
    [`norm_bits`](#norm_bits) makes for widths, for the same reason.

    Worth recording that the earlier probe on
    [#26](https://github.com/fractalyze/sig-frx/issues/26) sized this loop at
    `2·bits` — 12,604 at `n = 1024`. That is **below** the measured worst case,
    so it was never a budget, and the probe says so itself: it was measuring
    whether the loop compiles, not whether the number theory closes.

    The bound is tight rather than merely safe: `f0` of 6,302 bits against
    `g0 = 1` takes 12,603 steps against its own bound of 12,607.
    """
    return 4 * bits + 1


def _is_odd(value: Any) -> Any:
    """The low bit of the least significant limb."""
    return (fnp.asarray(value)[..., 0] & np.uint32(1)) != 0


def _pick(flag: Any, when_true: Any, when_false: Any) -> Any:
    """`where` over a `[...]` flag and `[..., limbs]` operands."""
    return fnp.where(flag[..., None], when_true, when_false)


def base_case(f0: ArrayLike, g0: ArrayLike, bits: int, q: int) -> tuple[Any, Any, Any]:
    """Algorithm 6 at degree 1: `(F, G, ok)` with `f0·G - g0·F = q`.

    The bottom of [`descend`](#descend), where the ring is `Z` and the NTRU
    equation is Bezout's. `f0·u + g0·v = 1` gives `F = -q·v` and `G = q·u`,
    since `f0·G - g0·F = q·(f0·u + g0·v)`.

    `ok` is false when `gcd(f0, g0) != 1`, which is not an error: Algorithm 5
    draws a fresh `f` and `g` and descends again.

    It is also false when either input is zero, and that one is a domain
    restriction rather than something the loop settles. The gcd is read out of
    `v`, so the two sides are not symmetric: `g0 = 0` leaves `v` at zero and is
    refused, while `f0 = 0` runs to a correct answer — `(0, ±1)` really does
    solve. Carrying that asymmetry would mean documenting which zero is
    recoverable for a pair Algorithm 5's norm and invertibility checks exclude
    before the descent is ever called, so the domain is closed at both non-zero
    instead. Both refusals are conservative: a solvable pair may be rejected
    here, and an unsolvable one is never accepted.

    ## Binary, because the substrate has no divide and no positional multiply

    HAC 14.61 over [`bigint`](bigint.py): shifts, adds, subtracts and one
    magnitude compare, which is exactly the operation set
    [`bigint`](bigint.py) chose to carry. Its step 2 — the common factor of two
    — is not run, because a pair that shares one has `gcd != 1` and is a reject
    rather than something to descend into; what step 2 guarantees for the rest
    of the algorithm is instead asserted by `ok` and the parity argument below.

    The four branches are exclusive and priority-ordered, so one traced step is
    one of them rather than HAC's nested `while`s. Past termination the loop is
    a no-op that matters: `u = 0` is even, so every remaining step takes the
    first branch, which touches `u`, `A` and `B` and leaves `v`, `C` and `D` —
    the three the answer is read from — where they settled.

    ## Why the halved sum is always an integer

    The branch that replaces `A` with `(A + y)/2` needs `A + y` even, and that
    follows from `u = A·x + B·y` with `u` even and `x`, `y` not both even. Take
    `x` odd and `y` even: then `u ≡ A (mod 2)`, so `A` is even, and the branch
    is only reached when `A` and `B` are not both even — so `B` is odd, `B - x`
    is even, and `A + y` is even. The other two parities go the same way. This
    is where not running step 2 is paid for, and `ok` is what pays it.
    """
    leading = fnp.asarray(f0).shape[:-1]
    # Registers hold a two's-complement value: `A` through `D` peak at `bits+2`
    # and the sum halved above them at `bits+3`, measured over 60 pairs at
    # `n = 1024`. A whole further limb of headroom costs two limbs of 421 and
    # buys 15 bits against a peak that is otherwise checked only by the
    # equation the tests assert.
    limbs = bigint.limb_count(bits + 4 + bigint.LIMB_BITS)

    def constant(value: int) -> Any:
        packed = bigint.to_limbs(value % (1 << (limbs * bigint.LIMB_BITS)), limbs)
        return fnp.broadcast_to(fnp.asarray(packed), (*leading, limbs))

    signed_f, signed_g = _widen(f0, limbs), _widen(g0, limbs)
    negative_f, negative_g = bigint.is_negative(signed_f), bigint.is_negative(signed_g)
    zero = constant(0)
    x = _pick(negative_f, bigint.sub(zero, signed_f), signed_f)
    y = _pick(negative_g, bigint.sub(zero, signed_g), signed_g)

    def body(_: Any, state: tuple[Any, ...]) -> tuple[Any, ...]:
        u, v, a, b, c, d = state
        u_odd, v_odd = _is_odd(u), _is_odd(v)
        halve_u = ~u_odd
        halve_v = u_odd & ~v_odd
        subtract = u_odd & v_odd
        take_u = subtract & bigint.at_least(u, v)
        take_v = subtract & ~take_u

        # The branches are exclusive, so the arithmetic runs once on operands
        # chosen by the flags rather than once per branch on operands chosen
        # after. A `_pick` is an elementwise select and a `bigint.add`/`sub` is
        # an `associative_scan` over every limb, so selecting first turns ten
        # carry scans per step into five — at `n = 1024` that is ~126,000 scans
        # a key that were being computed and discarded. Worth 1.54x on the warm
        # call there: 576.9 ms to 374.4 ms, median of seven on one workstation,
        # the two forms interleaved against the same key.
        halving, subtracting = _pick(halve_v, v, u), _pick(take_v, v, u)
        other = _pick(take_v, u, v)
        left, right = _pick(halve_v, c, a), _pick(halve_v, d, b)
        left_other, right_other = _pick(take_v, c, a), _pick(take_v, d, b)
        left_from, right_from = _pick(take_v, a, c), _pick(take_v, b, d)

        # `(left, right)` halved, keeping `left·x + right·y` where it was.
        plain = ~_is_odd(left) & ~_is_odd(right)
        left_halved = bigint.shift_right_signed(
            _pick(plain, left, bigint.add(left, y)), 1
        )
        right_halved = bigint.shift_right_signed(
            _pick(plain, right, bigint.sub(right, x)), 1
        )
        shifted = bigint.shift_right(halving, 1)
        value_step = bigint.sub(subtracting, other)
        left_step = bigint.sub(left_other, left_from)
        right_step = bigint.sub(right_other, right_from)

        return (
            _pick(halve_u, shifted, _pick(take_u, value_step, u)),
            _pick(halve_v, shifted, _pick(take_v, value_step, v)),
            _pick(halve_u, left_halved, _pick(take_u, left_step, a)),
            _pick(halve_u, right_halved, _pick(take_u, right_step, b)),
            _pick(halve_v, left_halved, _pick(take_v, left_step, c)),
            _pick(halve_v, right_halved, _pick(take_v, right_step, d)),
        )

    one = constant(1)
    start = (x, y, one, zero, zero, one)
    _, gcd, _, _, u_coeff, v_coeff = lax.fori_loop(0, gcd_budget(bits), body, start)

    # `C·|f0| + D·|g0| = 1`, and `|f0| = ±f0`, so the sign rides into the
    # coefficient rather than being carried alongside it.
    u_coeff = _pick(negative_f, bigint.sub(zero, u_coeff), u_coeff)
    v_coeff = _pick(negative_g, bigint.sub(zero, v_coeff), v_coeff)

    # Both even is `gcd >= 2` and so a reject either way, but it has to be
    # tested rather than left to the loop: it is exactly HAC's step-2
    # precondition, and without it the parity argument above fails and the
    # halved sums stop being integers. The loop then runs on and can land on
    # `gcd == 1` spuriously — `f0 = 44450, g0 = 624` does, and their gcd is 2.
    coprime_parity = _is_odd(x) | _is_odd(y)
    nonzero = ~fnp.all(x == 0, axis=-1) & ~fnp.all(y == 0, axis=-1)
    ok = fnp.all(gcd == one, axis=-1) & coprime_parity & nonzero
    big_f = bigint.sub(zero, bigint.mul_small(v_coeff, np.uint32(q)))
    big_g = bigint.mul_small(u_coeff, np.uint32(q))
    return big_f, big_g, ok


# -- Algorithm 5: the draw and the two checks ---------------------------------

# (3.29)'s `4096` — the draws one polynomial always costs, whatever its degree,
# and eight bytes each. Named because four things read them: the guard and the
# reshape in `draw_polynomial`, the per-attempt budget below, and the scheme
# module that has to squeeze exactly that many bytes.
DRAW_COUNT = 4096
DRAW_BYTES = 8

# §3.8.2's `σ_{f,g} = 1.17·sqrt(q/2n)` at `n = 4096`, which is the width one
# draw carries. A coefficient at degree `n` is the sum of `4096/n` of them, so
# its variance is `(4096/n)·σ²` — the standard's `σ_{f,g}` for that degree,
# reached with a single table rather than one per parameter set.
DRAW_SIGMA = 1.43300980528773

# Bits of the uniform each draw consumes. The table below is exact well past
# this, so it is the quantisation alone that separates the sampled distribution
# from `D_{Z,σ,0}` — under `2^-62` per draw, and under `2^-50` over the 4096
# draws a key costs. That is the *keygen* requirement, which is that `f` and `g`
# have the right geometry; the sampler #27 needs for signing is a different
# object with a Rényi bound of its own, and this is not it.
DRAW_BITS = 62
_HALF_BITS = DRAW_BITS // 2

# §3.8.2's `1.17·sqrt(q)`, squared, because the quantity it gates is a squared
# norm. Here rather than at the call site: the constant is what turns
# [`gram_schmidt_squared_norm`](#gram_schmidt_squared_norm) into Algorithm 5's
# line 5, and a caller that supplied it by hand would also have to know to
# square it. Degree-independent, unlike `FalconParams.squared_norm_bound`, so it
# is a module constant rather than a parameter-set field.
GRAM_SCHMIDT_BOUND = 1.17**2 * arith.Q


@lru_cache(maxsize=None)
def _draw_table() -> np.ndarray:
    """The CDT for `|X|`, as `[tail + 1, 2]` halves of a `DRAW_BITS` threshold.

    Two 31-bit columns rather than one 62-bit value, because a lane is 32 bits
    and the comparison has to be exact — a table quantised to what a lane holds
    would be the dominant error rather than a negligible one.

    Built with `decimal` rather than `math.exp`: a `float64` exponential is
    accurate to `2^-52`, which would put the table's own error two orders above
    the quantisation it is meant to be below.
    """
    with decimal.localcontext() as context:
        context.prec = 80
        variance = decimal.Decimal(DRAW_SIGMA) ** 2
        weights, tail = [decimal.Decimal(1)], 0
        while True:
            tail += 1
            weight = 2 * (-decimal.Decimal(tail) ** 2 / (2 * variance)).exp()
            weights.append(weight)
            if weight < decimal.Decimal(2) ** -(DRAW_BITS + 2):
                break
        total = sum(weights)
        scale = decimal.Decimal(2) ** DRAW_BITS
        running = decimal.Decimal(0)
        rows = []
        for weight in weights:
            running += weight
            threshold = int(running / total * scale)
            rows.append((threshold >> _HALF_BITS, threshold & ((1 << _HALF_BITS) - 1)))
    table = np.array(rows, dtype=np.uint32)
    table.flags.writeable = False  # cached and shared, so hand out no-one's to edit
    return table


def draw_polynomial(degree: int, stream: ArrayLike) -> Any:
    """`4096/n` discrete Gaussian draws summed per coefficient, as `[degree]`.

    `stream` is `[4096, 8]` bytes — eight per draw, of which `DRAW_BITS` choose
    the magnitude and one chooses the sign.

    The magnitude is a table lookup and not a rejection loop, which is what lets
    the draw be traced at all: a cumulative table turns "sample until accepted"
    into "count the thresholds this uniform passes".
    """
    if DRAW_COUNT % degree:
        raise ValueError(f"degree {degree} does not divide {DRAW_COUNT}")
    # Lifting for the caller would put a device dispatch on each of the ~17
    # attempts a key costs, on a path that is concrete throughout
    # ([`conventions.md`](../../../docs/reference/conventions.md)).
    xnp = namespace(stream)
    bytes_ = xnp.asarray(stream).astype(np.uint32)

    def word(chunk: Any) -> Any:
        """Four bytes as one 32-bit value, most significant first."""
        return (
            (chunk[:, 0] << np.uint32(24))
            | (chunk[:, 1] << np.uint32(16))
            | (chunk[:, 2] << np.uint32(8))
            | chunk[:, 3]
        )

    # Each half drops its low bit to reach 31, and the bit the second half drops
    # is the sign: 62 magnitude bits and one sign bit, so 63 of the 64 reach an
    # output and each of those reaches exactly one. The 64th — the first half's
    # dropped bit — is spare because the budget is odd, not because it is lost.
    #
    # Worth spelling out. The first version of this was three shifts against a
    # mask, and the arithmetic hid that one byte reached no output at all, that
    # seven bits of `high` were always zero, and that two bytes collided in one
    # position — none of which the distribution test could see, because its
    # tolerance is far wider than the tail probabilities a skew like that moves.
    first, second = word(bytes_[:, :4]), word(bytes_[:, 4:])
    high, low = first >> np.uint32(1), second >> np.uint32(1)
    sign = second & np.uint32(1)

    top, bottom = _draw_table()[:, 0], _draw_table()[:, 1]
    high, low = high[:, None], low[:, None]
    passed = (high > top) | ((high == top) & (low >= bottom))
    magnitude = xnp.sum(passed, axis=-1, dtype=np.int32)
    values = xnp.where(sign.astype(bool), -magnitude, magnitude)
    return xnp.sum(
        values.reshape(degree, DRAW_COUNT // degree), axis=-1, dtype=np.int32
    )


def gram_schmidt_squared_norm(f: ArrayLike, g: ArrayLike) -> Any:
    """Algorithm 5 line 5's quantity — the basis's Gram-Schmidt norm, squared.

    `max(‖(g, −f)‖², q²·‖(ḡ/(f f̄ + g ḡ), f̄/(f f̄ + g ḡ))‖²)`, which is the
    larger of the two rows of the NTRU basis after orthogonalisation, against
    [`GRAM_SCHMIDT_BOUND`](#GRAM_SCHMIDT_BOUND). The second row is a division in
    the FFT domain, so this is where key generation needs the rational transform
    rather than the integer one.

    **The second row never leaves the transform domain.** Only the sum of its
    coefficients' squares is read, and Parseval turns that into a function of
    the energy alone — `Σ from_g² + Σ from_f² = (1/n)·Σ 1/energy`, since the
    numerator `|f̂|² + |ĝ|²` *is* the energy. So the two inverse transforms the
    formula reads as are not computed. The reference transcription
    ([`falcon_reference`](testing/falcon_reference.py)) does compute them, which
    is what makes the test a check of the identity rather than of itself.

    Reads the namespace off its arguments, as [`fft`](fft.py) does: key
    generation's rejection loop is concrete, so this is called on the host,
    where the double-precision scope is unnecessary — and lifting it here would
    make the scope mandatory for a caller that never needed one.
    """
    xnp = namespace(f, g)
    left, right = xnp.asarray(f, dtype=np.float64), xnp.asarray(g, dtype=np.float64)
    degree = np.shape(left)[-1]
    left_hat, right_hat = fft.fft(left), fft.fft(right)
    energy = (left_hat * xnp.conj(left_hat) + right_hat * xnp.conj(right_hat)).real
    first_row = xnp.sum(left * left) + xnp.sum(right * right)
    second_row = (arith.Q**2 / degree) * xnp.sum(1.0 / energy)
    return xnp.maximum(first_row, second_row)


def invertible(f: ArrayLike) -> Any:
    """Algorithm 5 line 6 — whether `f` is a unit in `Z_q[x]/(x^n + 1)`.

    The public key is `g/f mod q`, so this is the condition that one exists.
    Read off the transform: a product of evaluations is zero exactly when one of
    them is, and `arith.ntt` is what evaluates. This is the one step of key
    generation that has no host form at all — the opcode does not have one — so
    it lands on the device whatever the caller is.
    """
    residues = arith.ntt(arith.to_field(fnp.asarray(f)))
    return fnp.all(residues.astype(np.uint32) != np.uint32(0))


# -- the two `Z_q` steps ------------------------------------------------------


def public_key(f: ArrayLike, g: ArrayLike) -> Any:
    """Algorithm 4 line 9 — `h = g·f^{-1} mod q`, as `[n]` residues under `q`.

    The whole of what a Falcon public key is. Read back with `astype` and never
    a bitcast, since the field's storage is a Montgomery representative
    ([`arith`](arith.py)), and left uncentered because §3.11.4 encodes "each
    value in the 0 to `q − 1` range".

    Lands on the device whatever the caller is, for the reason
    [`invertible`](#invertible) does: `arith.ntt` is an opcode with no host
    form. So key generation is host for the trapdoor and device for its two
    `Z_q` steps, which is the same shape verification's `decompress` has.
    """
    f_hat = arith.ntt(arith.to_field(f))
    g_hat = arith.ntt(arith.to_field(g))
    return arith.intt(arith.base_div(g_hat, f_hat)).astype(np.uint32)


def recover_g(f: ArrayLike, g: ArrayLike, big_f: ArrayLike) -> Any:
    """(3.35) — `G = (q + g·F)/f mod ϕ`, the polynomial §3.11.5 does not encode.

    A private key stores `f`, `g` and `F`; `G` is recomputed on load, which is
    what makes the encoding a quarter shorter than the trapdoor it carries.

    **The `q +` vanishes here and that is the point.** It is what makes the
    numerator divisible by `f` over the integers, where this computes modulo `q`
    and `q ≡ 0` — which is exactly the observation §3.11.5 makes when it says
    the recovery "can be done modulo `q` as well, using the same techniques as
    signature verification". So one division in the transform domain recovers a
    polynomial whose coefficients are not residues at all.

    What licenses reading the answer back as centered integers is that `G` is
    *small*: §3.11.5 asserts it in as many words, and the encoding gives `F`
    eight bits for the same reason. A `G` that outgrew `q/2` would come back as
    a different polynomial with no complaint, so
    [`keygen_test`](testing/keygen_test.py) checks the NTRU equation over the
    recovered `G` rather than trusting the round trip.
    """
    numerator = arith.base_mul(
        arith.ntt(arith.to_field(g)), arith.ntt(arith.to_field(big_f))
    )
    return arith.centered(
        arith.intt(arith.base_div(numerator, arith.ntt(arith.to_field(f))))
    )


# What one attempt draws: `f` and `g`, at `DRAW_COUNT` draws of `DRAW_BYTES`
# each. Exported as a byte count rather than as the shape it reshapes to,
# because a caller only has to size a squeeze — the layout is this module's.
_ATTEMPT_SHAPE = (2, DRAW_COUNT, DRAW_BYTES)
ATTEMPT_BYTES = 2 * DRAW_COUNT * DRAW_BYTES


def ntru_gen(stream: ArrayLike, degree: int) -> tuple[Any, Any, Any, Any] | None:
    """Algorithm 5 once — `(f, g, F, G)`, or `None` where the standard restarts.

    All three of the standard's rejections, and none of the retrying. **The
    restart loop is the caller's**, for two reasons that point the same way: its
    trip count is data, so it is a host loop and this module has no host loop to
    put it in; and a restart needs *fresh bytes*, which means a seed expansion,
    which is a scheme's decision and not Algorithm 5's — the standard says only
    "restart". [`falcon.Falcon.keygen`](falcon.py) is where both live.

    `stream` is [`ATTEMPT_BYTES`](#ATTEMPT_BYTES) bytes, `f`'s draw and then
    `g`'s. Taking bytes rather than a seed is what keeps this deterministic in
    its input and testable without a hash.

    ## The order of the rejections is the standard's, and it is also the cheap one

    Lines 7-11 come before line 12 in Algorithm 5, and what each costs shows
    why: a measured run at `n = 512` took **16 rejections at line 10 costing
    about 30 ms between them, then one accepted attempt costing 64 s**, nearly
    all of it the solve. Reordering would put the expensive test first for
    nothing.

    Line 13's rejection is the one that cannot be moved earlier — the descent's
    bottom is coprime for roughly half the pairs that reach it, and finding out
    costs the descent. So a key costs a small number of *solves*, where it costs
    a larger number of attempts.
    """
    draws = np.asarray(stream, dtype=np.uint8).reshape(_ATTEMPT_SHAPE)
    f = np.asarray(draw_polynomial(degree, draws[0]))
    g = np.asarray(draw_polynomial(degree, draws[1]))

    # Line 7. On the device whatever the caller is, since `arith.ntt` has no
    # host form — the one step of an attempt that leaves the host.
    if not bool(np.asarray(invertible(f))):
        return None
    # Lines 9-11.
    if float(np.asarray(gram_schmidt_squared_norm(f, g))) > GRAM_SCHMIDT_BOUND:
        return None
    # Line 12, and line 13's `⊥` is `ok`.
    bits = max(int(np.abs(f).max()), int(np.abs(g).max())).bit_length()
    limbs_f, limbs_g, _, ok = ntru_solve(
        to_limbs(f, bits), to_limbs(g, bits), bits, arith.Q
    )
    if not bool(np.asarray(ok)):
        return None
    return f, g, _from_limbs(limbs_f), _from_limbs(limbs_g)


def _from_limbs(limbs: ArrayLike) -> Any:
    """A reduced `[degree, limbs]` back to the small integers it holds.

    `F` and `G` come out of [`reduce`](#reduce) at whatever limb budget the
    answer needed, and what they hold at that point fits eight bits — §3.11.5
    gives `F` exactly that. So this narrows to `int32` rather than staying wide,
    and the encoder is what would notice if it did not.
    """
    rows = np.asarray(limbs)
    return np.array(
        [bigint.from_limbs(row, signed=True) for row in rows], dtype=np.int32
    )


# -- the lift back up ---------------------------------------------------------


def lift_bits(lower_bits: int, other_bits: int, degree: int) -> int:
    """The proved bound on `|(F'(x²)·h(-x))_c|` at `degree`.

    `F'(x²)` is empty at every odd position, so the convolution sums `degree/2`
    products rather than `degree` of them. Each is one coefficient of each
    operand, and the negacyclic wrap flips signs without touching magnitudes.

    `degree` is the **result's**, where [`norm_bits`](#norm_bits) takes its
    input's — the two steps move in opposite directions, so each names the end
    it is sized from. Both refuse a degree their step cannot reach: a lift lands
    at twice its operand's degree, so 1 is not one of them.
    """
    if degree < 2 or degree & (degree - 1):
        raise ValueError(f"degree {degree} is not a power of two above 1")
    return lower_bits + other_bits + (degree // 2 - 1).bit_length()


def lift(
    lower: ArrayLike, other: ArrayLike, lower_bits: int, other_bits: int
) -> tuple[Any, int]:
    """`lower(x²) · other(-x)` at `other`'s degree — Algorithm 6's step back up.

    The identity the whole recursion turns on: `f(x)·f(-x)` is `N(f)(x²)`, so
    setting `F = F'(x²)·g(-x)` and `G = G'(x²)·f(-x)` gives

        f·G − g·F = G'(x²)·[f(x)f(-x)] − F'(x²)·[g(x)g(-x)]
                  = [N(f)·G' − N(g)·F'](x²) = q

    — the level below's solution *is* this level's, once both sides are put back
    over `x` rather than `x²`. Nothing here is approximate; the product is exact
    and the equation holds on the nose. What it costs is width, which is why
    [`reduce`](#reduce) follows immediately.

    Residues carry the product for the same reason they carry the descent's:
    the convolution is independent per channel. The two operands may arrive at
    different limb budgets — the lower level's is wider — and
    [`bigint.to_rns`](bigint.py) reads whichever it is handed.
    """
    low, high = fnp.asarray(lower), fnp.asarray(other)
    degree = high.shape[0]
    if low.shape[0] * 2 != degree:
        raise ValueError(f"{low.shape[0]} coefficients cannot lift to degree {degree}")
    result_bits = lift_bits(lower_bits, other_bits, degree)
    channels, limbs = bigint.signed_shape(result_bits)
    mods = bigint.moduli(channels)

    # `lower(x²)`: interleaving with zeros is the whole substitution, since the
    # coefficient of `x^(2i)` is `lower_i` and every odd position is empty.
    spread = fnp.reshape(
        fnp.stack([low, fnp.zeros_like(low)], axis=1), (degree, low.shape[-1])
    )
    # `other(-x)`: odd positions change sign and even ones do not.
    odd = fnp.asarray(np.arange(degree) % 2 == 1)
    negated = fnp.where(odd[:, None], _negate(high), high)

    def residues(part: Any) -> Any:
        return fnp.swapaxes(bigint.to_rns(part, channels, signed=True), -1, -2)

    product = negacyclic_mul(residues(spread), residues(negated), mods)
    result = bigint.from_rns(
        fnp.swapaxes(product, -1, -2), channels, limbs, signed=True
    )
    return result, result_bits


# -- Babai's reduction --------------------------------------------------------

# How much of `k` one step takes, and what each operand is read back to.
# `_CHUNK` is bounded by `float64`: `k` has to survive `round` as an exact
# integer, which stops at `2^53`, and 50 leaves room for the rounding itself.
#
# The windows are the scale that puts the quotient near there, and they are an
# estimate rather than a guarantee — a level's widths bound `|F|/|f|` measured
# coefficient-wise, while `k` is a quotient of *evaluations*, and `|f(ζ)|` at
# the worst root runs well under `f`'s largest coefficient. That gap is real
# and about 15 bits at the degrees here, so `reduce` rescales by what the
# quotient turns out to be rather than trusting the estimate. Getting this
# wrong is quiet: `k` overflows the digits below, the correction is garbage,
# the step is rejected for making no progress, and the level simply does not
# reduce.
_SMALL_WINDOW = 60
_CHUNK = 50
_LARGE_WINDOW = _SMALL_WINDOW + _CHUNK

# One limb more than the larger window needs, because a window is limb-aligned
# and the value's top bit is not: the extra limb is what guarantees the window
# still holds `_LARGE_WINDOW` bits when the top limb carries only one of them.
_WINDOW_LIMBS = bigint.limb_count(_LARGE_WINDOW) + 1


@frx.jit
def _magnitudes(values: ArrayLike) -> tuple[Any, Any]:
    """A signed polynomial's magnitudes and its signs, in one device call."""
    signed = fnp.asarray(values)
    negative = bigint.is_negative(signed)
    return fnp.where(negative[..., None], _negate(signed), signed), negative


def _host_summary(magnitude: Any, negative: Any) -> tuple[np.ndarray, np.ndarray, int]:
    """Magnitudes, signs and the widest bit length among them, on the host.

    The one place this module moves a wide value off the device, and it is
    small in the way that matters: limbs times degree is roughly constant down
    the recursion — the coefficients widen exactly as fast as the degree halves
    — so every level moves about 2,000 limbs, a few kilobytes, however deep it
    is. What the host then does with them is bookkeeping and a transform over
    `m` doubles; the arithmetic stays where the limbs live.
    """
    limbs, signs = np.asarray(magnitude), np.asarray(negative)
    used = np.flatnonzero(np.any(limbs != 0, axis=tuple(range(limbs.ndim - 1))))
    if used.size == 0:
        return limbs, signs, 0
    top = int(used[-1])
    return (
        limbs,
        signs,
        bigint.LIMB_BITS * top + int(limbs[..., top].max()).bit_length(),
    )


def _view(values: ArrayLike) -> tuple[np.ndarray, np.ndarray, int]:
    """[`_host_summary`](#_host_summary) of a value the device still holds."""
    return _host_summary(*_magnitudes(values))


def _width(values: ArrayLike) -> int:
    """The bit length of the widest magnitude in `values`, read back to the host."""
    return _view(values)[2]


def _floats(view: tuple[np.ndarray, np.ndarray, int], exponent: int) -> np.ndarray:
    """`view`'s value divided by `2^exponent`, as host doubles.

    The exponent is the caller's to choose and is what makes the quotient
    computable: `F` and `f` are scaled apart by exactly the amount `k` is
    expected to span, so `k` comes out near `2^_CHUNK` however wide the two
    operands actually are.

    Only the top `_WINDOW_LIMBS` limbs take part, which is not an optimization:
    `2^(15j)` for a limb index into a 629-limb number overflows a double long
    before the sum could reach it, and the zero limb it would be multiplying is
    exactly the case that turns an infinity into a NaN. Windowing keeps every
    place value inside the range where it means something, which is also the
    only range a 53-bit mantissa can report.
    """
    limbs, signs, width = view
    top = max(0, (width - 1) // bigint.LIMB_BITS)
    start = max(0, top + 1 - _WINDOW_LIMBS)
    window = limbs[..., start : top + 1].astype(np.float64)
    places = bigint.LIMB_BITS * (start + np.arange(window.shape[-1])) - exponent
    return np.where(signs, -1.0, 1.0) * (window @ (2.0**places))


@frx.jit
def _subtract_multiple(
    big: ArrayLike,
    wide: ArrayLike,
    source: ArrayLike,
    flip: ArrayLike,
    digits: ArrayLike,
    shift: ArrayLike,
) -> tuple[Any, Any, Any]:
    """`big − ((k · wide) << shift)`, with `k` handed over as `digits`.

    The product that keeps this loop affordable. `k` is small and `wide` is
    not, so the work is [`bigint.mul_small`](bigint.py) rather than a residue
    round trip — and that matters more than it looks: a trip through `from_rns`
    costs one sequential step per channel, which at the bottom level is 630 of
    them, and this runs a few hundred times per key rather than once per level.

    The sum over the convolution's terms is a tree rather than a walk, so it
    costs `log2(m)` carry scans instead of `m` of them. Two's complement makes
    the wrap free: a term that crossed `x^m` is negated rather than subtracted,
    and a partial sum that leaves the limb budget still lands on the right
    value, because every operation here is exact modulo `2^(L·15)` and the
    answer fits.

    **The wrap is applied to the operand, not to the products.** Every step
    here is exact modulo `2^(L·15)`, so `-(a·d)` and `(-a)·d` are the same
    limbs — and one is a single carry scan over the gathered array while the
    other is one per digit of `k`. Negating first is what makes the digit loop
    below cost four multiplies rather than four multiplies and four negations.

    Compiled, and that is the whole reason `shift` arrives as a value rather
    than as a Python integer. A step is a few hundred carry scans; dispatched
    one at a time they cost more than the arithmetic by an order of magnitude,
    and a shift baked in as a constant would have meant one compilation per
    step rather than one per level.
    """
    values = fnp.asarray(wide)
    gathered = fnp.take(values, source, axis=0)
    signed = _pick(fnp.asarray(flip), _negate(gathered), gathered)
    total = fnp.zeros_like(values)
    for level in range(fnp.asarray(digits).shape[0]):
        terms = bigint.mul_small(signed, digits[level][None, :, None])
        while terms.shape[1] > 1:
            half = terms.shape[1] // 2
            terms = bigint.add(terms[:, :half], terms[:, half:])
        total = bigint.add(
            total, bigint.shift_left(terms[:, 0], bigint.LIMB_BITS * level)
        )
    result = bigint.sub(fnp.asarray(big), bigint.shift_left_dynamic(total, shift))
    magnitude, negative = _magnitudes(result)
    return result, magnitude, negative


def reduce(
    big_f: ArrayLike, big_g: ArrayLike, f: ArrayLike, g: ArrayLike
) -> tuple[Any, Any, int]:
    """Algorithm 7 — `(F, G)` shortened against `(f, g)`, and its width.

    `k = ⌊(F·f̄ + G·ḡ) / (f·f̄ + g·ḡ)⌉` and then `F -= k·f`, `G -= k·g`. The
    equation survives any `k` at all — `f(G − kg) − g(F − kf)` is `fG − gF`
    whatever `k` is — so this cannot produce a wrong answer, only a wide one.
    Width is the entire content of the step, and the next level's cost is what
    pays for missing it.

    ## `k` is thousands of bits and a double holds 53, so it arrives in chunks

    [`lift`](#lift) roughly triples the width, so `|k| ≈ |F|/|f|` spans the
    difference — 6,276 bits at the bottom level of `n = 1024`. No transform
    over doubles produces that. What one *can* produce is its top
    `_CHUNK` bits: scale `F` and `f` apart by a chosen exponent and the
    quotient lands where a double is exact. Subtract that much, and the next
    pass sees a narrower `F` and takes the next chunk down. The loop is the
    specification's own `while k ≠ 0`, run against a `k` that is deliberately
    truncated rather than one that is merely rounded.

    ## The loop runs on the host; the shift it decides is spent on the device

    A step's shift is `width(F) − width(f) − _CHUNK`, which depends on the
    values rather than on their budgets — and their budgets are no guide, since
    [`norm_bits`](#norm_bits) runs 2.10x wide by the bottom, so a shift taken
    from the bound would scale `f` clean out of existence rather than merely
    imprecisely. Reading the width back is what makes it knowable, and it is
    knowable *here*, as a Python integer.

    It is then handed to [`_subtract_multiple`](#_subtract_multiple) as a value
    rather than as a constant, which is a separate decision and a compile-time
    one: baked in, each of the few hundred steps would compile a program of its
    own. That is the caller [`bigint.shift_left_dynamic`](bigint.py) exists for
    — and it is the *left* shift, because a step scales its correction up to
    meet `F` rather than scaling `F` down to meet the correction, which is why
    [`bigint.shift_right`](bigint.py)'s deferral still stands.

    The arithmetic all stays on device — `k·f`, the subtraction and the lift
    are the wide operations and none of them moves. What the host does is
    decide the exponent and run the transform, which is `m` doubles wide; what
    crosses back is 120 bits of a 9,000-bit number.

    Returning the width rather than a bound is the other half of that choice.
    A reduced width has no proof to lean on — it is a property of the basis,
    which is why the reference implementation's own width table is measured —
    so the level above sizes its lift from what this one actually produced.
    Measured beats bounded here precisely because it cannot be short.
    """
    values_f, values_g = fnp.asarray(big_f), fnp.asarray(big_g)
    degree, limbs = values_f.shape[0], values_f.shape[-1]
    wide_f, wide_g = _widen(f, limbs), _widen(g, limbs)

    view_f, view_g = _view(f), _view(g)
    small_width = max(view_f[2], view_g[2])
    view_big_f, view_big_g = _view(values_f), _view(values_g)
    width = max(view_big_f[2], view_big_g[2])
    if small_width == 0:
        # `(f, g)` is the zero pair, which the base case already refused; there
        # is no lattice to reduce against and no quotient to divide by.
        return values_f, values_g, width

    exponent = small_width - _SMALL_WINDOW
    f_fft = fft.fft(_floats(view_f, exponent))
    g_fft = fft.fft(_floats(view_g, exponent))
    f_conjugate, g_conjugate = np.conj(f_fft), np.conj(g_fft)
    denominator = (f_fft * f_conjugate + g_fft * g_conjugate).real

    source, wrapped = _convolution_indices(degree)
    device_source = fnp.asarray(source)
    levels = bigint.limb_count(_CHUNK + 1)

    while width > small_width:
        shift = max(0, width - small_width - _CHUNK)
        scaled = exponent + shift
        numerator = (
            fft.fft(_floats(view_big_f, scaled)) * f_conjugate
            + fft.fft(_floats(view_big_g, scaled)) * g_conjugate
        )
        quotient = numerator / denominator
        if not np.isfinite(quotient).all():
            break
        # `k` is bounded by the quotient's largest *evaluation* — every
        # coefficient is an average of them — so the scale that makes it fit is
        # readable here rather than guessable from the widths. Dividing in the
        # evaluation domain is free: it commutes with the transform below, so
        # this costs a scan of `m` doubles and no second transform.
        peak = float(np.max(np.abs(quotient)))
        excess = max(0, int(np.ceil(np.log2(peak))) - _CHUNK) if peak else 0
        k = np.round(fft.ifft(quotient / 2.0**excess).real).astype(np.int64)
        if not k.any():
            break
        applied = shift + excess

        # A term's sign is the wrap's, flipped again where `k` itself is
        # negative, and its digits are `k` cut into limbs the multiply can take.
        magnitude = np.abs(k)
        flip = fnp.asarray(wrapped ^ (k < 0)[None, :])
        digits = fnp.asarray(
            np.stack(
                [
                    (magnitude >> (bigint.LIMB_BITS * level)) & int(bigint.MASK)
                    for level in range(levels)
                ]
            ).astype(np.uint32)
        )
        amount = fnp.asarray(np.int32(applied))
        candidate_f, magnitude_f, negative_f = _subtract_multiple(
            values_f, wide_f, device_source, flip, digits, amount
        )
        candidate_g, magnitude_g, negative_g = _subtract_multiple(
            values_g, wide_g, device_source, flip, digits, amount
        )
        next_f = _host_summary(magnitude_f, negative_f)
        next_g = _host_summary(magnitude_g, negative_g)
        next_width = max(next_f[2], next_g[2])
        # A step that does not shorten is where the truncated `k` has run out of
        # meaning, and stopping on it is what makes the loop terminate: the
        # width is a non-negative integer and every step it takes lowers it.
        if next_width >= width:
            break
        values_f, values_g = candidate_f, candidate_g
        view_big_f, view_big_g = next_f, next_g
        width = next_width

    # Trimmed to what the answer needs rather than to what the lift needed.
    # The level above multiplies this by its own `f`, so carrying the budget
    # forward would put the *widest* level's limb count on every level over it
    # — and the widest is the bottom, where a limb count in the hundreds meets
    # a degree of two. Leaving that to the caller is a trap worth closing here:
    # it costs nothing to get right and a great deal to get wrong.
    trimmed = min(bigint.limb_count(width + 2), values_f.shape[-1])
    return values_f[..., :trimmed], values_g[..., :trimmed], width


def ntru_solve(
    f: ArrayLike, g: ArrayLike, bits: int, q: int
) -> tuple[Any, Any, int, Any]:
    """Algorithm 6 in full: `(F, G, bits, ok)` with `f·G − g·F = q`.

    The descent, the base case and the walk back up, in the order the recursion
    runs them. `ok` is the base case's — a pair whose bottom is not coprime has
    no solution, and Algorithm 5 answers that by drawing again.

    Each level's lift is sized from the level below's *measured* width rather
    than from a bound on it (see [`reduce`](#reduce)), so the upward pass is
    tight where the descent is deliberately not.
    """
    values_f, values_g = fnp.asarray(f), fnp.asarray(g)
    degree = values_f.shape[0]
    levels = degree.bit_length() - 1

    # A level's width bound follows the entry width and the degree alone, so
    # both descents produce the same sequence of bounds and only `f`'s is kept.
    descent_f, descent_g = descend(values_f, bits, levels), descend(
        values_g, bits, levels
    )
    chain_f = [values_f, *(value for value, _ in descent_f)]
    chain_g = [values_g, *(value for value, _ in descent_g)]
    bottom_bits = descent_f[-1][1]

    big_f, big_g, ok = base_case(chain_f[-1][0], chain_g[-1][0], bottom_bits, q)
    big_f, big_g = big_f[None, :], big_g[None, :]
    big_bits = max(_width(big_f), _width(big_g))

    for depth in range(levels - 1, -1, -1):
        level_f, level_g = chain_f[depth], chain_g[depth]
        # Measured on both sides, not bounded. The descent's bound runs 2.10x
        # wide by the bottom, and sizing the lift from it would put roughly four
        # times the channels on the widest product in the whole recursion.
        level_bits = max(_width(level_f), _width(level_g))
        lifted_f, lifted_bits = lift(big_f, level_g, big_bits, level_bits)
        lifted_g, _ = lift(big_g, level_f, big_bits, level_bits)
        big_f, big_g, big_bits = reduce(lifted_f, lifted_g, level_f, level_g)

    return big_f, big_g, big_bits, ok


# -- the ffLDL tree -----------------------------------------------------------


@dataclass(frozen=True)
class FalconTree:
    """§3.8.3's tree, held one level per array rather than one node per object.

    Algorithm 9 recurses, so the obvious form is a `(value, left, right)` object
    per node — 2n−1 of them at `n = 1024`, each holding an array of a different
    length. Every node at a depth does the same arithmetic on the same shape,
    though, so a depth is one array and the tree is `log₂n` of them:

    - `values[d]` is `L10` for every node at depth `d`, as `[2^d, n >> d]`.
    - a node at index `i` has children `2i` and `2i + 1` at depth `d + 1`, and
      `leaves` — `[n]`, real — follows the same rule one step past `values[-1]`.

    So the leaf under the path `b₀b₁…b_{k−1}` is at index `Σ bⱼ·2^(k−1−j)`,
    which is the path read as a binary number with the root's bit at the top.

    What that buys is the difference between `log₂n` array operations and 2n−1
    of them: at `n = 1024`, ten against 2,047. It costs the same memory — `n`
    complex per level either way — and it is the layout a traced walk of the
    tree would want, which is what
    [#27](https://github.com/fractalyze/sig-frx/issues/27) does with it.

    A tree out of [`ffldl`](#ffldl) is Algorithm 9's; one out of
    [`normalize`](#normalize) is a *Falcon* tree, which is Algorithm 4's name
    for the same shape after its leaves have been divided into `σ`. The
    specification draws that distinction in §3.8.3 and so does this, by which
    function returned it — the two differ in what a leaf means and in nothing a
    type could carry.
    """

    values: tuple[Any, ...]
    leaves: Any


def gram(
    f_hat: ArrayLike, g_hat: ArrayLike, big_f_hat: ArrayLike, big_g_hat: ArrayLike
) -> tuple[Any, Any, Any]:
    """Algorithm 4 line 4's `B̂ × B̂*` for `B = [[g, −f], [G, −F]]`, as `(G00, G01, G11)`.

    Three entries rather than four because `G10` is `conj(G01)`: the product of
    a matrix with its own adjoint is self-adjoint, and an adjoint is elementwise
    conjugation in this domain — a root of `x^n = −1` lies on the unit circle,
    so `f*(ζ) = conj(f(ζ))`. Handing back a fourth array would be handing back
    something a caller has to keep in step with the first three.

    **The diagonal is real, and is returned real.** `G00` is `|ĝ|² + |f̂|²`, a
    sum of squared magnitudes; carrying it as a complex value with a zero
    imaginary part would leave every step below deciding what to do with a
    rounding artefact the algebra says is not there.

    Takes each polynomial's transform rather than computing it, which is
    [`arith`](arith.py)'s convention seen from the rational side. Signing needs
    `B̂` itself — Algorithm 10 lines 3 and 7 — so a `gram` that transformed
    internally would compute the same four transforms twice.
    """
    xnp = namespace(f_hat, g_hat, big_f_hat, big_g_hat)
    small_f, small_g = xnp.asarray(f_hat), xnp.asarray(g_hat)
    large_f, large_g = xnp.asarray(big_f_hat), xnp.asarray(big_g_hat)
    # `(−f)·(−f)*` is `f·f*` and `(−f)·(−F)*` is `f·F*`, so both of `B`'s
    # negations cancel out of the product and this reads as `f` and `F`. They
    # are not decorative for having cancelled: they are what makes `det B` equal
    # `f·G − g·F`, which is the quantity Algorithm 6 sets to `q`.
    return (
        (small_g * xnp.conj(small_g) + small_f * xnp.conj(small_f)).real,
        small_g * xnp.conj(large_g) + small_f * xnp.conj(large_f),
        (large_g * xnp.conj(large_g) + large_f * xnp.conj(large_f)).real,
    )


def ldl(g00: ArrayLike, g01: ArrayLike, g11: ArrayLike) -> tuple[Any, Any, Any]:
    """Algorithm 8 — `(L10, D00, D11)` with `G = L·D·L*`, over `[..., n]` nodes.

    `L` is unit lower triangular and `D` diagonal, so `L10` is the only entry of
    either that is not a constant, and the three returned here are the whole
    decomposition. `G10` is not taken as an argument for the reason
    [`gram`](#gram) does not return it.

    **`L10 ⊙ L10*` is `|L10|²`.** The standard writes the product; taking its
    real part is the same number and it keeps `D11` real, which a Gram diagonal
    is. Without it the recursion carries an imaginary part of about `2^-52` that
    the algebra says is zero, and the leaves at the bottom would need it removed
    anyway. The transcription in
    [`falcon_reference`](testing/falcon_reference.py) computes the complex
    product as written, so the two forms are held against each other rather than
    the simplification being taken on trust.
    """
    xnp = namespace(g00, g01, g11)
    diagonal = xnp.asarray(g00)
    l10 = xnp.conj(xnp.asarray(g01)) / diagonal
    d11 = xnp.asarray(g11) - (l10 * xnp.conj(l10)).real * diagonal
    return l10, diagonal, d11


def _weave(left: Any, right: Any) -> Any:
    """Two `[m, ...]` levels as one `[2m, ...]`, left at `2i` and right at `2i+1`.

    Algorithm 9's `leftchild` / `rightchild` written as an index: the left child
    of node `i` is node `2i` of the level below. Interleaving rather than
    concatenating is what makes the depth-first path of
    [`FalconTree`](#FalconTree) a binary number.
    """
    xnp = namespace(left, right)
    return xnp.reshape(
        xnp.stack([left, right], axis=1), (2 * left.shape[0], *left.shape[1:])
    )


def ffldl(g00: ArrayLike, g01: ArrayLike, g11: ArrayLike) -> FalconTree:
    """Algorithm 9 — the LDL tree of a self-adjoint `[n]` Gram matrix.

    One iteration per depth instead of one call per node, which is the whole of
    what [`FalconTree`](#FalconTree) buys. Nothing else is reshaped: each
    iteration is Algorithm 8 followed by lines 8-10, and the loop ends where
    line 3 does.

    ## Why the children's Gram matrix is built the way it is

    Lines 8-10 break each diagonal entry `D` into a matrix over the ring below.
    `D` is self-adjoint, so (3.30) says its multiplication map has the
    transformation matrix `[[d0, d1], [d1*, d0]]` for `(d0, d1) = splitfft(D)` —
    the same `d0` twice on the diagonal, which is why the child's `G11` here is
    literally the child's `G00` and not a second array computed alongside it.

    The recursion therefore descends the same tower of rings
    `Q[x]/(x^n + 1) ⊃ Q[x]/(x^(n/2) + 1) ⊃ …` that
    [`descend`](#descend) walks over the integers. It is the only thing the two
    halves of key generation have in common: this side is `float64` and holds no
    integer, where that side is thousands of bits wide and holds nothing else.

    ## The bottom is ring degree 2, and its children are numbers

    Line 3 stops at `n = 2`, where `D00` and `D11` are the leaves rather than
    subtrees — and there they are *rational constants*: `d* = d` in
    `Q[x]/(x² + 1)` forces the `x` coefficient to zero, since the adjoint sends
    `x` to `−x`. So the two evaluations of a leaf are the same number, and
    `leaves` carries that number rather than the pair. Their mean is taken over
    picking one, which would privilege an index over a fact;
    [`keygen_test`](testing/keygen_test.py) is what pins that the two agree.

    That also fixes the leaf count at `n` rather than `2n`: one rational leaf
    stands for both dimensions the ring contributes, which is what makes the
    leaves multiply to `q^n` for a basis of determinant `q` rather than to
    `q^(2n)`.
    """
    xnp = namespace(g00, g01, g11)
    if np.ndim(g00) != 1:
        raise ValueError(f"a Gram matrix entry is one polynomial, not {np.ndim(g00)}")
    degree = np.shape(g00)[-1]
    if degree < 2 or degree & (degree - 1):
        raise ValueError(f"degree {degree} is not a power of two above 1")

    # The node axis the loop below runs on: a depth's `2^d` nodes are rows of
    # one array, so a depth costs one array operation instead of `2^d` calls.
    top = tuple(xnp.asarray(entry)[None, :] for entry in (g00, g01, g11))
    values: list[Any] = []
    while True:
        l10, d00, d11 = ldl(*top)
        values.append(l10)
        if d00.shape[-1] == 2:
            pairs = _weave(d00, d11)
            return FalconTree(tuple(values), 0.5 * (pairs[..., 0] + pairs[..., 1]))
        left_even, left_odd = fft.split(d00)
        right_even, right_odd = fft.split(d11)
        child_diagonal = _weave(left_even, right_even)
        top = (child_diagonal, _weave(left_odd, right_odd), child_diagonal)


def normalize(tree: FalconTree, sigma: float) -> FalconTree:
    """Algorithm 4 lines 6-7 — `leaf ← σ/√leaf`, which is what makes it a Falcon tree.

    `σ` is the parameter set's own, Table 3.3's 165.736 617 183 at `n = 512` and
    168.388 571 447 at `n = 1024`, and it arrives as an argument rather than as
    a constant here because it is the one quantity in this module that depends
    on which set is being generated — unlike
    [`GRAM_SCHMIDT_BOUND`](#GRAM_SCHMIDT_BOUND), which does not.

    A leaf is a squared Gram-Schmidt norm of the basis, so this turns it into
    the standard deviation the sampler rounds that coordinate with: a longer
    orthogonalised vector is divided by, and rounds more tightly. Algorithm 11
    line 2 states the range that comes out — `σ' ∈ [σmin, σmax]` — and it is
    what Algorithm 5's line-5 rejection buys, since the largest leaf *is*
    `‖B‖²_GS` and the smallest is `q²` over it.

    Only the leaves move. `values` is `L10` per node and is untouched by
    normalization, which is why the two trees share one type.
    """
    xnp = namespace(tree.leaves)
    return FalconTree(tree.values, sigma / xnp.sqrt(xnp.asarray(tree.leaves)))
