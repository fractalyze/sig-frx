# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Integers wider than a lane, carried on device, for Falcon's NTRU trapdoor.

Solving `fG - gF = q` descends a tower of fields, and its intermediates reach
**9,427 bits** at `n = 1024` and 4,732 at `n = 512` — 295x a 32-bit lane.
[#26](https://github.com/fractalyze/sig-frx/issues/26) measured that and decided
to carry those magnitudes on device rather than on the host, so something has to
hold them. This module is that something, and it holds nothing else: there is no
`q` here, no degree, and no Falcon. It is integer magnitude.

## Two representations, because no one of them does both jobs

Multiplication wants residues and comparison wants place value, so both are here
and the CRT bridges them:

- **residue form** — `[..., k]` residues against `k` pairwise-coprime moduli.
  Multiply and add are elementwise and independent per channel, which is the
  whole reason the recursion above can be array-shaped at all.
- **positional form** — `[..., L]` limbs, little-endian, base `2^15`. Compare,
  shift and the carry-bearing operations live here; residues cannot express
  them, since a residue says nothing about magnitude.

**No positional multiply exists and none should be added.** Splitting the work
this way is what keeps the limb narrow: the binary GCD at the base of the
recursion is shifts and adds, and Babai's `k*f` is a product, which goes back
through residues. A positional multiply would be the one operation that forces a
wider accumulator than anything else here needs.

## Everything is 15 bits wide, and that is one decision rather than two

Limbs and moduli are the same width, so every product in the module is under
`2^30` and every sum of two limbs is under `2^16`. That is what lets one carry
rule serve `add`, `sub` and `mul_small` alike.

Sixteen would have been the obvious width for a residue — a 16-bit product still
fits a lane, and it needs 590 channels against 629. It is not used because
`mul_small` is the operation the CRT rebuild multiplies *by a residue*: split a
`2^31` product into a low and a high half and the high half no longer fits
below `2^15`, so a limb can take a carry of two and the rule below stops
holding. Paying 6.6% more channels buys a single width and a single carry rule.

## A carry is a prefix, not a walk

`carry_out(i) = generate(i) | (propagate(i) & carry_out(i-1))` composes
associatively, so the carries of a whole number are one
[`associative_scan`](#_carry) over that monoid rather than `L` sequential steps.
At the widths above `L` is 629, and a walk would put 629 sequential steps inside
every step of a loop that already runs 12,604 of its own.

**The carry dtype is `uint8` on purpose.** The scan moves its carry array at
every combine step, so the carry's width is the scan's bandwidth — in this
repo's Algorithm 18 decoder, narrowing a nine-state carry from `int32` to
`uint8` measured 3.0x for bit-identical output
([#189](https://github.com/fractalyze/sig-frx/issues/189)). Here the carry is
two bits of information, so it has no business in the `uint32` the limbs live
in.

## Traced only, deliberately

Nothing here reads `arrays.namespace`. The host already has arbitrary-precision
integers built in, so a host implementation of this module would be a slower
spelling of `int` — and #26 decided against carrying these magnitudes on the
host, which is the entire reason the module exists. `to_limbs` and `from_limbs`
are the boundary, and they are the only functions that touch a Python `int`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import frx.numpy as fnp
import numpy as np
from frx import lax
from frx.typing import ArrayLike

# The one width. See the module docstring for why limbs and moduli share it.
LIMB_BITS = 15
BASE = np.uint32(1 << LIMB_BITS)
MASK = np.uint32((1 << LIMB_BITS) - 1)


# The bridge reduces each limb against its place value and then sums, so the
# accumulator holds at most `L` values under `2^15`. Past this many limbs that
# sum leaves a `uint32` without raising — which is the failure the repo's first
# non-negotiable names, and the reason it is a bound rather than a comment. It
# sits four orders of magnitude above the 629 limbs Falcon's widest intermediate
# needs, so nothing here is near it; what it buys is that a future caller with a
# much wider value fails loudly instead of quietly.
MAX_LIMBS = (1 << 32) // (1 << LIMB_BITS)


def limb_count(bits: int) -> int:
    """How many limbs hold an unsigned value of `bits` bits."""
    return -(-bits // LIMB_BITS)


def _check_bridgeable(limbs: int) -> None:
    """Refuse a limb budget whose reduced-limb sum would leave the lane."""
    if limbs > MAX_LIMBS:
        raise ValueError(
            f"{limbs} limbs exceeds the {MAX_LIMBS} the bridge can sum without "
            f"leaving a 32-bit lane; a wider value needs a wider accumulator, "
            f"not more limbs"
        )


@lru_cache(maxsize=None)
def _primes_below_base() -> np.ndarray:
    """Every prime under `2^15`, descending — the pool channels are drawn from.

    Descending because the largest moduli need the fewest channels, and the
    channel count is what the recursion above pays for.
    """
    sieve = np.ones(1 << LIMB_BITS, dtype=bool)
    sieve[:2] = False
    for candidate in range(2, int(np.sqrt(1 << LIMB_BITS)) + 1):
        if sieve[candidate]:
            sieve[candidate * candidate :: candidate] = False
    return np.flatnonzero(sieve)[::-1].astype(np.uint32)


def _frozen(table: np.ndarray) -> np.ndarray:
    """A cached table is shared, so hand out something a caller cannot edit."""
    table.flags.writeable = False
    return table


@lru_cache(maxsize=None)
def moduli(channels: int) -> np.ndarray:
    """The `channels` largest primes under `2^15`, as a `[channels]` table.

    Distinct primes rather than merely coprime moduli: the CRT needs pairwise
    coprimality and primality is how it is guaranteed without an argument.
    """
    pool = _primes_below_base()
    if channels > pool.size:
        raise ValueError(
            f"{channels} channels requested but only {pool.size} primes exist "
            f"under 2^{LIMB_BITS}; a wider value needs a wider limb, not more "
            f"channels"
        )
    return _frozen(pool[:channels].copy())


@lru_cache(maxsize=None)
def channel_count(bits: int) -> int:
    """How many channels represent every unsigned value of `bits` bits uniquely.

    Counted against the actual product of the chosen primes rather than against
    `bits / 15`: the primes are slightly under `2^15`, and a bound that ignores
    that is short by a channel at these widths.
    """
    pool = _primes_below_base()
    product, used = 1, 0
    target = 1 << bits
    while product <= target:
        if used == pool.size:
            raise ValueError(f"{bits} bits exceeds the product of every 15-bit prime")
        product *= int(pool[used])
        used += 1
    return used


# -- the boundary: Python integers in, Python integers out --------------------


def to_limbs(value: int, limbs: int) -> np.ndarray:
    """A non-negative Python `int` as `[limbs]` limbs, least significant first."""
    if value < 0:
        raise ValueError("to_limbs takes a non-negative value; see from_limbs(signed=)")
    if value >> (limbs * LIMB_BITS):
        raise ValueError(f"value needs more than {limbs} limbs of {LIMB_BITS} bits")
    mask = (1 << LIMB_BITS) - 1  # a Python int: `value` is far wider than a lane
    return np.array(
        [(value >> (LIMB_BITS * i)) & mask for i in range(limbs)], dtype=np.uint32
    )


def from_limbs(limbs: ArrayLike, *, signed: bool = False) -> int:
    """Limbs back to a Python `int`.

    `signed` reads the same limbs as two's complement over `L*15` bits, which is
    what [`sub`](#sub) already produces when it runs past zero — the operation
    wraps rather than saturating, so the representative is there whether or not
    a caller asks for it. Reading it is opt-in because a magnitude and its
    two's-complement twin are the same limbs, and only the caller knows which
    one it meant.
    """
    values = np.asarray(limbs).astype(np.uint32)
    if values.ndim != 1:
        raise ValueError(f"from_limbs takes one number, not shape {values.shape}")
    total = 0
    for i, limb in enumerate(values.tolist()):
        total |= limb << (LIMB_BITS * i)
    if signed and total >> (values.size * LIMB_BITS - 1):
        total -= 1 << (values.size * LIMB_BITS)
    return total


# -- positional form ----------------------------------------------------------


def _carry(generate: Any, propagate: Any) -> Any:
    """The carry entering each limb, as a prefix scan over the carry monoid.

    `generate` and `propagate` are per-limb predicates; the carry leaving a
    prefix is `g | (p & carry_out_of_the_shorter_prefix)`, and composing two
    prefixes is associative — which is the whole reason this is a scan. The
    carry *into* limb `i` is the carry *out of* limb `i-1`, so the scan's output
    shifts up one place with a zero at the bottom.
    """

    def compose(lower: Any, upper: Any) -> Any:
        lower_generate, lower_propagate = lower
        upper_generate, upper_propagate = upper
        return (
            upper_generate | (upper_propagate & lower_generate),
            upper_propagate & lower_propagate,
        )

    out, _ = lax.associative_scan(
        compose,
        (generate.astype(np.uint8), propagate.astype(np.uint8)),
        axis=-1,
    )
    return fnp.concatenate(
        [fnp.zeros((*out.shape[:-1], 1), np.uint8), out[..., :-1]], axis=-1
    ).astype(np.uint32)


def _normalize(raw: Any) -> Any:
    """Limbs each under `2^16` reduced to limbs under `2^15`, carries resolved.

    Every carry-bearing operation here lands in that state first, which is why
    they share this: a limb over `2^16` would let one absorb a carry of two and
    make the monoid above wrong rather than merely imprecise.
    """
    return (raw + _carry(raw >= BASE, raw == BASE - np.uint32(1))) & MASK


def add(a: ArrayLike, b: ArrayLike) -> Any:
    """`a + b` over limbs, modulo `2^(L*15)`.

    The wrap is two's complement, so this is also signed addition and a signed
    layer above needs no operation of its own.
    """
    return _normalize(fnp.asarray(a) + fnp.asarray(b))


def sub(a: ArrayLike, b: ArrayLike) -> Any:
    """`a - b` over limbs, modulo `2^(L*15)`.

    The offset by `BASE` is what keeps every intermediate inside an unsigned
    lane: a limb difference is signed, and a lane that went negative would wrap
    to something enormous rather than to something a borrow can describe. Read
    the result with [`from_limbs(signed=True)`](#from_limbs) when `b > a`.
    """
    raw = fnp.asarray(a) + BASE - fnp.asarray(b)
    borrow = _carry(raw < BASE, raw == BASE)
    return (raw - borrow) & MASK


def mul_small(a: ArrayLike, scalar: ArrayLike) -> Any:
    """`a * scalar` over limbs, modulo `2^(L*15)`, for a `scalar` under `2^15`.

    The product of two limbs is under `2^30`, so it fits a lane whole and splits
    into two halves that are each under `2^15`. Realigning the high halves one
    place up leaves every limb under `2^16`, which is exactly the state
    [`_normalize`](#_normalize) resolves — so multiplying by a small value costs
    one carry scan, the same as adding.

    The scalar is not checked under a tracer, where it cannot be. Passing
    something wider silently overflows the split, which is the failure mode the
    single width in this module exists to keep out of reach.
    """
    product = fnp.asarray(a) * fnp.asarray(scalar).astype(np.uint32)
    high = product >> np.uint32(LIMB_BITS)
    shifted = fnp.concatenate(
        [fnp.zeros((*high.shape[:-1], 1), np.uint32), high[..., :-1]], axis=-1
    )
    return _normalize((product & MASK) + shifted)


def is_negative(a: ArrayLike) -> Any:
    """Whether limbs read as two's complement are negative — the top bit."""
    values = fnp.asarray(a)
    return (values[..., -1] >> np.uint32(LIMB_BITS - 1)) & np.uint32(1) != 0


def _sign_fill(values: Any) -> Any:
    """The one limb a two's-complement value extends with — all ones, or none.

    A negative value's limbs are all ones above its magnitude, which is the same
    rule read from both ends: [`shift_right`](#shift_right) moves it in at the
    top and [`sign_extend`](#sign_extend) appends it past the end.
    """
    return fnp.where(is_negative(values), MASK, np.uint32(0))[..., None]


def sign_extend(a: ArrayLike, limbs: int) -> Any:
    """The same value at a wider limb budget, read as two's complement.

    Widening a magnitude is a pad with zeros and widening a signed value is a
    pad with its sign; they differ exactly where the value is negative. The
    caller is a layer whose working registers need more room than the operands
    they are derived from, which is what the base case's cofactors are.
    """
    values = fnp.asarray(a)
    extra = limbs - values.shape[-1]
    if extra <= 0:
        return values
    return fnp.concatenate(
        [values, fnp.broadcast_to(_sign_fill(values), (*values.shape[:-1], extra))],
        axis=-1,
    )


def shift_right(a: ArrayLike, bits: int, *, signed: bool = False) -> Any:
    """`a >> bits` over limbs, for a `bits` known at trace time.

    `signed` shifts the sign bit in rather than zeros, which is the difference
    between halving a magnitude and halving a two's-complement value. The base
    case's Bezout cofactors are the caller: they go negative by construction and
    are halved once per step, so a logical shift there turns `-3` into something
    near `2^(L*15)` and the identity stops holding without anything raising.

    A data-dependent shift *amount* is still deliberately absent. Babai's
    reduction wants one and will need it, but it is not written here until that
    caller exists — a traced amount costs a gather per limb, which is not a
    price this module's current callers should pay for a generality none of them
    uses.
    """
    values = fnp.asarray(a)
    limbs = values.shape[-1]
    whole, part = divmod(bits, LIMB_BITS)
    fill = (
        _sign_fill(values) if signed else fnp.zeros((*values.shape[:-1], 1), np.uint32)
    )
    if whole >= limbs:
        return fnp.broadcast_to(fill, values.shape)
    moved = fnp.concatenate(
        [values[..., whole:], fnp.broadcast_to(fill, (*values.shape[:-1], whole))],
        axis=-1,
    )
    if part == 0:
        return moved
    upper = fnp.concatenate([moved[..., 1:], fill], axis=-1)
    return (moved >> np.uint32(part)) | ((upper << np.uint32(LIMB_BITS - part)) & MASK)


def shift_left(a: ArrayLike, bits: int) -> Any:
    """`a << bits` over limbs, modulo `2^(L*15)`, for a `bits` known at trace time."""
    values = fnp.asarray(a)
    limbs = values.shape[-1]
    whole, part = divmod(bits, LIMB_BITS)
    if whole >= limbs:
        return fnp.zeros_like(values)
    moved = fnp.concatenate(
        [
            fnp.zeros((*values.shape[:-1], whole), np.uint32),
            values[..., : limbs - whole],
        ],
        axis=-1,
    )
    if part == 0:
        return moved
    lower = fnp.concatenate(
        [fnp.zeros((*moved.shape[:-1], 1), np.uint32), moved[..., :-1]], axis=-1
    )
    return ((moved << np.uint32(part)) & MASK) | (lower >> np.uint32(LIMB_BITS - part))


def at_least(a: ArrayLike, b: ArrayLike) -> Any:
    """`a >= b`, comparing magnitudes rather than two's-complement values.

    Decided at the most significant limb where the two differ, found with a
    reduction rather than a scan: place value means every less significant limb
    is irrelevant once one differs, so there is nothing to propagate.

    The operands are broadcast first. Every other operation here broadcasts for
    free, being elementwise; this one gathers at an index derived from *both*,
    so a bare `[..., L]` against a `[L]` constant — comparing a whole polynomial
    against one bound, which is what the callers do — would otherwise fail on
    the gather rather than answer.
    """
    left, right = fnp.broadcast_arrays(fnp.asarray(a), fnp.asarray(b))
    positions = fnp.arange(left.shape[-1], dtype=np.int32)
    top = fnp.max(fnp.where(left != right, positions, np.int32(-1)), axis=-1)
    index = fnp.maximum(top, np.int32(0))[..., None]
    highest_left = fnp.take_along_axis(left, index, axis=-1)[..., 0]
    highest_right = fnp.take_along_axis(right, index, axis=-1)[..., 0]
    return fnp.where(top < 0, True, highest_left >= highest_right)


# -- residue form -------------------------------------------------------------


def rns_add(a: ArrayLike, b: ArrayLike, mods: ArrayLike) -> Any:
    """`a + b` per channel."""
    return (fnp.asarray(a) + fnp.asarray(b)) % fnp.asarray(mods)


def rns_sub(a: ArrayLike, b: ArrayLike, mods: ArrayLike) -> Any:
    """`a - b` per channel, offset so no lane goes negative."""
    return (fnp.asarray(a) + fnp.asarray(mods) - fnp.asarray(b)) % fnp.asarray(mods)


def rns_mul(a: ArrayLike, b: ArrayLike, mods: ArrayLike) -> Any:
    """`a * b` per channel — the operation residues exist for.

    Both operands are under `2^15`, so the product is under `2^30` and the lane
    holds it whole. This is where the width in this module's name is spent.
    """
    return (fnp.asarray(a) * fnp.asarray(b)) % fnp.asarray(mods)


# -- the bridge ---------------------------------------------------------------


@lru_cache(maxsize=None)
def _place_values(channels: int, limbs: int) -> np.ndarray:
    """`BASE^j mod m_i`, as `[channels, limbs]` — what reduces a limb sum.

    Host-built because `channels` and `limbs` arrive as Python integers, so the
    table is a host value by the rule that a value is used in the namespace it
    arrives in ([`conventions.md`](../../../docs/reference/conventions.md)).
    """
    mods = moduli(channels).astype(np.int64)
    table = np.empty((channels, limbs), dtype=np.uint32)
    place = np.ones(channels, dtype=np.int64)
    for j in range(limbs):
        table[:, j] = place
        place = (place << LIMB_BITS) % mods
    return _frozen(table)


@lru_cache(maxsize=None)
def _limb_span(channels: int, limbs: int) -> np.ndarray:
    """`BASE^L mod m_i` — what a two's-complement value is offset by.

    Limbs wrap modulo `BASE^L` and residues reduce modulo a product of primes,
    so the two forms disagree about a negative value by exactly this. It is the
    whole content of the `signed` flag on both bridge directions.
    """
    mods = moduli(channels).astype(np.int64)
    place = np.ones(channels, dtype=np.int64)
    for _ in range(limbs):
        place = (place << LIMB_BITS) % mods
    return _frozen(place.astype(np.uint32))


def to_rns(a: ArrayLike, channels: int, *, signed: bool = False) -> Any:
    """Positional limbs to `[..., channels]` residues.

    Each limb is reduced against its own place value *before* the sum, which is
    what keeps the accumulator in a lane: a limb times a place value is under
    `2^30` and would overflow after two additions, while the reduced form is
    under `2^15` and survives every limb this module can hold.

    `signed` reads the limbs as two's complement. Without it a negative value
    arrives as `x + BASE^L`, which is a perfectly good residue of the wrong
    number — no operation downstream can tell, which is why the flag exists
    rather than a convention.
    """
    values = fnp.asarray(a)
    _check_bridgeable(values.shape[-1])
    mods = moduli(channels)
    table = _place_values(channels, values.shape[-1])
    terms = (values[..., None, :] * table) % mods[:, None]
    residues = fnp.sum(terms, axis=-1, dtype=np.uint32) % mods
    if not signed:
        return residues
    span = _limb_span(channels, values.shape[-1])
    return fnp.where(
        is_negative(values)[..., None],
        (residues + mods - span) % mods,
        residues,
    )


@lru_cache(maxsize=None)
def signed_shape(bits: int) -> tuple[int, int]:
    """Channels and limbs a signed value of magnitude under `2^bits` needs.

    Two requirements, and only the first is obvious. The channel product has to
    exceed `2^(bits+1)`, because a signed value occupies twice the classes an
    unsigned one of the same magnitude does. The **limb** budget then has to hold
    that product rather than the value — `from_rns(signed=True)` centers by
    subtracting `M`, and a budget sized to the value has nowhere to put it.
    """
    channels = channel_count(bits + 1)
    product = 1
    for modulus in moduli(channels).tolist():
        product *= modulus
    return channels, limb_count(product.bit_length() + 1)


@lru_cache(maxsize=None)
def _garner_tables(channels: int, limbs: int) -> tuple[np.ndarray, np.ndarray]:
    """The running products as limbs, and the inverse each mixed-radix digit needs.

    `prefix[i]` is `m_0 * ... * m_(i-1)` in positional form and `inverse[i]` is
    that product inverted modulo `m_i`. Both are host constants: they depend on
    the channel set and the limb budget and on nothing a caller passes.
    """
    mods = [int(m) for m in moduli(channels)]
    prefix = np.zeros((channels, limbs), dtype=np.uint32)
    inverse = np.empty(channels, dtype=np.uint32)
    running = 1
    for i, modulus in enumerate(mods):
        prefix[i] = to_limbs(running % (1 << (limbs * LIMB_BITS)), limbs)
        inverse[i] = 1 if i == 0 else pow(running % modulus, -1, modulus)
        running *= modulus
    return _frozen(prefix), _frozen(inverse)


@lru_cache(maxsize=None)
def _product_limbs(channels: int, limbs: int) -> tuple[np.ndarray, np.ndarray]:
    """The modulus product and its half, as limbs — where `signed` splits.

    Garner lands in `[0, M)`, and the centered representative of that class is
    what a value that was ever negative has to come back as. `M/2` is the
    boundary and `M` is what crossing it costs.
    """
    product = 1
    for modulus in moduli(channels).tolist():
        product *= modulus
    span = 1 << (limbs * LIMB_BITS)
    return _frozen(to_limbs(product % span, limbs)), _frozen(
        to_limbs((product // 2) % span, limbs)
    )


def from_rns(
    residues: ArrayLike, channels: int, limbs: int, *, signed: bool = False
) -> Any:
    """Residues back to positional limbs, by Garner's mixed-radix reconstruction.

    Garner rather than the explicit `Σ r_i * M_i * y_i`: the explicit form lands
    at up to `channels` times the modulus product and then owes a reduction by
    it, which is a division by a number this module cannot divide by. Garner
    builds the digits so the running value is already in range.

    The loop is sequential over channels by construction — digit `i` is defined
    against the value the first `i` channels reconstruct — so this costs
    `channels` iterations of a carry scan rather than one. That is the same
    order the reference implementation's `zint_rebuild_CRT` pays, and no caller
    has yet measured it as a pole; a tree-shaped form exists and is not written
    for that reason.

    `signed` returns the **centered** representative — Garner lands in `[0, M)`,
    and a value that was negative before it became residues comes back as
    `x + M`, which is the right class and the wrong integer. The limb budget has
    to hold `M` for that subtraction to mean anything, which is a stronger
    requirement than merely holding the value.
    """
    values = fnp.asarray(residues)
    _check_bridgeable(limbs)
    prefix, inverse = _garner_tables(channels, limbs)
    # The tables are indexed by the loop counter, which is traced — so they are
    # device arrays gathered from, not host arrays subscripted.
    mods, prefix, inverse, places = (
        fnp.asarray(table)
        for table in (moduli(channels), prefix, inverse, _place_values(channels, limbs))
    )

    def step(index: Any, total: Any) -> Any:
        modulus = fnp.take(mods, index)
        residue = fnp.take(values, index, axis=-1)
        place = fnp.take(places, index, axis=0)
        reduced = fnp.sum((total * place) % modulus, axis=-1, dtype=np.uint32) % modulus
        digit = (residue + modulus - reduced) % modulus
        digit = (digit * fnp.take(inverse, index)) % modulus
        return add(total, mul_small(fnp.take(prefix, index, axis=0), digit[..., None]))

    start = fnp.zeros((*values.shape[:-1], limbs), np.uint32)
    total = lax.fori_loop(0, channels, step, start)
    if not signed:
        return total
    product, half = _product_limbs(channels, limbs)
    return fnp.where(at_least(total, half)[..., None], sub(total, product), total)
