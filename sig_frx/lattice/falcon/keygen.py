# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Falcon key generation — Algorithm 5's draw, and Algorithm 6 under it.

Three sections, in the order key generation runs them rather than the order
they were built:

- **Algorithm 5** draws `f` and `g` from a discrete Gaussian and rejects the
  pair unless the basis is short enough and `f` is a unit.
- **the descent** (Algorithm 6's left-hand side) recurses on the field norm:
  `f` of degree `n` becomes `N(f) = f_e² - x·f_o²` of degree `n/2`.
- **the base case** closes it at degree 1, where the NTRU equation is Bezout's.

The lift back up and Babai's reduction land here too, and the restart loop that
drives Algorithm 5 lands with the caller that has a loop to put it in — all of
them [#26](https://github.com/fractalyze/sig-frx/issues/26).

The three share almost nothing: the descent is `bigint`'s residues, the base
case is `bigint`'s limbs under a `fori_loop`, and Algorithm 5 is `fft` and
`arith` and a table built with `decimal`. They are one module because they are
one operation, and the reason to split would be that the file stops being
readable rather than that the sections differ.

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
from functools import lru_cache
from typing import Any

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

    def widen(value: ArrayLike) -> Any:
        narrow = fnp.asarray(value)
        pad = limbs - narrow.shape[-1]
        fill = fnp.where(bigint.is_negative(narrow)[..., None], bigint.MASK, 0)
        return fnp.concatenate(
            [narrow, fnp.broadcast_to(fill, (*narrow.shape[:-1], pad))], axis=-1
        ).astype(np.uint32)

    def constant(value: int) -> Any:
        packed = bigint.to_limbs(value % (1 << (limbs * bigint.LIMB_BITS)), limbs)
        return fnp.broadcast_to(fnp.asarray(packed), (*leading, limbs))

    signed_f, signed_g = widen(f0), widen(g0)
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
    if 4096 % degree:
        raise ValueError(f"degree {degree} does not divide 4096")
    bytes_ = fnp.asarray(stream).astype(np.uint32)

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
    magnitude = fnp.sum(passed, axis=-1, dtype=np.int32)
    values = fnp.where(sign.astype(bool), -magnitude, magnitude)
    return fnp.sum(values.reshape(degree, 4096 // degree), axis=-1, dtype=np.int32)


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
