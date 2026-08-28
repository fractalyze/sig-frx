# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""§4.4's `SamplerZ` and the three algorithms under it.

Signing rounds each coordinate of a lattice point to a nearby integer, and
*which* integer has to be drawn from a discrete Gaussian centred where the
coordinate fell rather than picked. Doing that with the wrong distribution does
not produce wrong signatures — it produces signatures that leak the trapdoor —
so this is the one place in Falcon where matching the standard is a security
property and not only an interoperability one.

Four algorithms, each a layer over the one below:

| | | |
|---|---|---|
| Algorithm 12 | [`base_sampler`](#base_sampler) | a half-Gaussian on `{0..18}` |
| Algorithm 13 | [`approx_exp`](#approx_exp) | `2^63 · ccs · exp(−x)`, in integers |
| Algorithm 14 | [`ber_exp`](#ber_exp) | a bit, true with probability `ccs·exp(−x)` |
| Algorithm 15 | [`sampler_z`](#sampler_z) | the rejection loop over the three |

## This is host code, and deliberately so

Every other operation in this package is written for a batch and traced.
This one is a scalar Python loop:

- **Its trip count is the data.** Algorithm 15 rejects until a candidate is
  accepted, and how many attempts that takes depends on the centre — which is
  secret. [`rejection.py`](../rejection.py) reshapes a rejection into a fixed
  budget plus a compaction, but that shape is for rejections whose *candidates
  are public*, which this one's are not: the reshaping would have to draw and
  retain the rejected candidates, and they are exactly what must not be kept.
- **It has no batch axis to give up.** Signing is not the supported path here
  ([`security.md`](../../../docs/reference/security.md)); it exists to reproduce
  known-answer tests and for development. A traced form would be work spent
  where the repo does not claim performance.

## No timing claim is made, and the standard's word for this is not ours

§4.4 calls `SamplerZ` *isochronous* and the reference implementation goes to
real lengths for it — the `>> 63` in `approx_exp` is there so a 126-bit product
never branches, and Algorithm 14's byte-at-a-time comparison exists to bound the
randomness consumption without a data-dependent exit.

**None of that survives being written here**, and this module claims none of it.
Python integers are variable-width and allocate; `math.floor` and the float
arithmetic are the host's. `security.md` names this operation directly as one
whose signing time is a function of the secret, and permits it because signing
carries no side-channel claim at all. The isochrony described above is a
property of the *algorithm as specified*, restated so a reader knows what the
shape is for — never a property of this code.

## Two constants that cannot be derived, only transcribed

- **`u` is read big-endian.** Algorithm 12 says `u ← UniformBits(72)` and the
  published vectors read their nine bytes most-significant-first. The reference
  implementation instead splits a little-endian `uint64` plus a byte into three
  24-bit limbs, which is a *different* map from a byte stream to a sample: on
  the first published vector the two disagree, `z0 = 3` against `z0 = 1`. Both
  are correct for their own randomness source — the reference's is a ChaCha20
  buffer, not this byte string — and only the vectors settle which one a test
  driven by `randombytes` needs.
- **`SIGMA_MIN` is transcribed, not computed.** `σmin = σ/(1.17√q)` holds to
  about `1e-13`, which is close enough to check the identity and not close
  enough to reproduce a vector: the products differ in the last bits and `ccs`
  is a float. These are the reference's own `fpr_sigma_min[]` at `logn` 9 and
  10, which reproduce every published `ccs` exactly.
"""

from __future__ import annotations

import math
from typing import Callable

# The reverse cumulative distribution table of Table 3.1, scaled by `2^72`.
# `RCDT[i] = Σ_{j>i} pdt[j]`, so Algorithm 12 counts how many entries the draw
# falls under. Eighteen entries for a support of `{0..18}`: the nineteenth
# would be `Σ_{j>18} pdt[j] = 0` and no draw is under it.
#
# Identical to the reference's `dist[]` in `sign.c`, which carries the same
# values pre-split into 24-bit limbs — checked entry by entry in the tests,
# because two tables that are meant to be one table are worth holding to it.
RCDT: tuple[int, ...] = (
    3024686241123004913666,
    1564742784480091954050,
    636254429462080897535,
    199560484645026482916,
    47667343854657281903,
    8595902006365044063,
    1163297957344668388,
    117656387352093658,
    8867391802663976,
    496969357462633,
    20680885154299,
    638331848991,
    14602316184,
    247426747,
    3104126,
    28824,
    198,
    1,
)

# Algorithm 13's `C`. `f(-x) = 2^-63 · Σ C[i]·x^(12-i)` approximates `exp(-x)`
# on `[0, ln 2]` — the interval Algorithm 14 reduces into — to well past the
# 51-odd bits the rest of the computation carries.
_APPROX_EXP_COEFFICIENTS: tuple[int, ...] = (
    0x00000004741183A3,
    0x00000036548CFC06,
    0x0000024FDCBF140A,
    0x0000171D939DE045,
    0x0000D00CF58F6F84,
    0x000680681CF796E3,
    0x002D82D8305B0FEA,
    0x011111110E066FD0,
    0x0555555555070F00,
    0x155555555581FF00,
    0x400000000002B400,
    0x7FFFFFFFFFFF4800,
    0x8000000000000000,
)

# Table 3.3's `σmin`, per parameter set. See the module docstring: transcribed
# from the reference's `fpr_sigma_min[]` rather than derived, because the
# derivation agrees only to about `1e-13` and `ccs` is a float product.
SIGMA_MIN: dict[int, float] = {
    512: 1.2778336969128337,
    1024: 1.298280334344292,
}

# `1/(2·σ0²)` at `σ0 = 1.8205`, the half-Gaussian `base_sampler` draws from.
# Algorithm 15 line 7 subtracts `z0²` scaled by this, which is what corrects the
# base distribution into the one centred at `r`.
_INV_TWICE_SIGMA0_SQUARED = 0.15086504887537272

# Algorithm 12 draws 72 bits, and Algorithm 14 compares a byte at a time.
_BASE_SAMPLER_BYTES = 9
_BER_EXP_BITS = 64

_UINT64 = (1 << 64) - 1
_SCALE = 1 << 63

#: Hands back exactly `n` uniformly random bytes. The tests pass a cursor over
#: the published `randombytes` string, which is what makes Table 3.2 runnable;
#: signing passes a squeezed stream.
RandomBytes = Callable[[int], bytes]


def base_sampler(randomness: RandomBytes) -> int:
    """Algorithm 12 — a draw from `χ`, the half-Gaussian on `{0, ..., 18}`.

    `χ` is within a Rényi divergence of `1 + 2^-78` of `D_{Z+, σmax}`, which is
    why a 72-bit table replaces sampling the half-Gaussian directly.

    The comparison is against `RCDT`, the *reverse* cumulative table, so the
    result counts the entries the draw falls under. Algorithm 12 flags this
    ("one should use RCDT, not pdt or cdt") because all three are tabulated
    beside each other in Table 3.1 and only one of them answers `< u`.
    """
    u = int.from_bytes(randomness(_BASE_SAMPLER_BYTES), "big")
    return sum(1 for threshold in RCDT if u < threshold)


def approx_exp(x: float, ccs: float) -> int:
    """Algorithm 13 — `2^63 · ccs · exp(−x)`, as an integer.

    Horner over `_APPROX_EXP_COEFFICIENTS` in fixed point: every intermediate is
    a 63-bit quantity, each step forms a 126-bit product and keeps its top half.
    Requires `x ∈ [0, ln 2]`, which is what `ber_exp` reduces into before
    calling — outside that interval the polynomial is not the approximation.

    The masking to 64 bits is the reference's `uint64` arithmetic, not the
    specification's: Algorithm 13 states the intermediates stay in
    `{0, ..., 2^63 − 1}`, and Python's unbounded integers would carry a
    borrow the C does not.
    """
    y = _APPROX_EXP_COEFFICIENTS[0]
    z = int(x * _SCALE)
    for coefficient in _APPROX_EXP_COEFFICIENTS[1:]:
        y = (coefficient - ((z * y) >> 63)) & _UINT64
    return (((int(ccs * _SCALE)) * y) >> 63) & _UINT64


def ber_exp(x: float, ccs: float, randomness: RandomBytes) -> bool:
    """Algorithm 14 — one bit, true with probability `≈ ccs · exp(−x)`.

    `x` is split as `x = s·ln 2 + r` with `r ∈ [0, ln 2)` so that `approx_exp`
    sees its own interval, and `exp(−x) = 2^-s · exp(−r)` puts `s` back as a
    shift. `s` saturates at 63: it exceeds that only when the half-Gaussian
    returned `z0 ≥ 13` at the smallest `σ'`, about `2^-32` of the time, and the
    bit is then true with probability under `2^-64` either way.

    The comparison walks `z` a byte at a time from the top and stops at the
    first byte that differs, which bounds the randomness a call consumes at one
    byte in all but about `2^-8` of cases. Algorithm 14 notes this loop is the
    one part that "does not need to be done in constant-time" — an exit that
    depends only on uniform bytes and not on `x`.
    """
    scale = int(x * (1 / math.log(2)))
    remainder = x - scale * math.log(2)
    scale = min(scale, 63)
    z = (((approx_exp(remainder, ccs) << 1) - 1) & _UINT64) >> scale
    offset = _BER_EXP_BITS
    while True:
        offset -= 8
        difference = randomness(1)[0] - ((z >> offset) & 0xFF)
        if difference != 0 or offset <= 0:
            return difference < 0


def sampler_z(
    center: float, inverse_sigma: float, degree: int, randomness: RandomBytes
) -> int:
    """Algorithm 15 — an integer from a distribution very close to `D_{Z,µ,σ'}`.

    `inverse_sigma` is `1/σ'` rather than `σ'` because that is what the caller
    has: `keygen.normalize` already divided `σ` into the root of each leaf, so
    walking the tree produces the reciprocal directly and inverting it here
    would be a round trip. The reference takes the same argument for the same
    reason.

    `σ'` must lie in `[σmin, σmax]`, which Algorithm 5's line-5 rejection is
    what guarantees — the largest leaf of an accepted basis *is* `‖B‖²_GS`. This
    does not check it: the range is a property of the key, established where the
    key is made rather than re-derived per coordinate.

    The loop draws from the fixed half-Gaussian, gives it a sign, and accepts
    with the probability that corrects the fixed width into `σ'` and the fixed
    centre into `r`. It has no iteration bound because it is not a rejection
    against a budget — the acceptance rate is bounded below by `σmin/σmax` by
    construction, so an unbounded loop terminates with probability one and a cap
    would only convert a hang into a wrong distribution.
    """
    if degree not in SIGMA_MIN:
        raise ValueError(f"degree {degree} is not a Falcon parameter set")
    floor = math.floor(center)
    fraction = center - floor
    ccs = inverse_sigma * SIGMA_MIN[degree]
    # `1/(2σ'²)`, formed once and in this order because the association is
    # observable. Writing line 7 as `(z-r)² * 0.5 * (1/σ')²` instead rounds one
    # ULP away on some inputs — a difference the acceptance bit hides, since it
    # compares a byte at a time from the top, but which the published
    # intermediates catch. Every float step below is the reference's own.
    half_inverse_sigma_squared = inverse_sigma * inverse_sigma * 0.5
    while True:
        z0 = base_sampler(randomness)
        bit = randomness(1)[0] & 1
        z = bit + (2 * bit - 1) * z0
        offset = z - fraction
        x = offset * offset * half_inverse_sigma_squared
        x -= (z0 * z0) * _INV_TWICE_SIGMA0_SQUARED
        if ber_exp(x, ccs, randomness):
            return z + floor
