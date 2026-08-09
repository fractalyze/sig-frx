# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Arithmetic in `Z_q` and the NTT over `q = 8380417`, for ML-DSA (FIPS 204).

Every polynomial multiplication in ML-DSA goes through the NTT, and the rounding
functions are where the scheme's correctness lives, so both are here and both are
testable without any scheme logic above them.

## The field is a dtype, and the transform is not

`q` is not one of `zk_dtypes`' curated fields, but it does not have to be: a field
minted from the modulus reduces internally, so `+`, `-` and `*` on `FIELD` are
already modular and nothing here implements them. That is what keeps this short,
and the alternative is worth naming — hand-written modular arithmetic means a
product of two residues (2^46) against a 32-bit lane with no widening multiply, so
every operation becomes a limb split carrying a bound that has to be argued.

The transform is the part that cannot be delegated. `frx.lax.ntt` has the
`NEGACYCLIC_NTT` mode this ring needs and resolves its kernels by curated field
family, so it refuses this modulus — see
[`conventions.md`](../../../docs/reference/conventions.md), which names the
condition that retires the layer walk below.

## Two representations, on purpose

The transform works on `FIELD`; the rounding functions work on canonical
integers, because `Power2Round` and `Decompose` are bit manipulation on a
representative rather than field arithmetic — `mod±`, a floor division, and a
carry case. They convert on entry with `astype`, which yields the canonical
residue. **Not a bitcast:** the field's storage is a Montgomery representative,
so reinterpreting the bytes gives a different number, and wrongly in a way no
round trip reveals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import zk_dtypes
from frx.typing import ArrayLike

from sig_frx.arrays import namespace

# FIPS 204 Table 1. `d` and `zeta` are fixed across the parameter sets; `gamma2`
# is not, so it is an argument rather than a constant here.
Q = 8380417
ZETA = 1753
D = 13
N = 256

FIELD = zk_dtypes.prime_field(Q)

# FIPS 204 Algorithm 42 line 21: `f = 256^-1 mod q`.
_N_INVERSE = 8347681


def bit_rev8(m: int) -> int:
    """FIPS 204 Algorithm 43: reverse the bits of a byte.

    A host function on purpose — it indexes the zeta table, and the table is
    static, so this never sees a traced value.
    """
    return int(format(m & 0xFF, "08b")[::-1], 2)


def _zeta_table() -> np.ndarray:
    """`zeta^BitRev8(k) mod q` for `k = 0..255` — FIPS 204 Appendix B.

    Generated rather than transcribed: it is 255 values that a single line
    defines, and a table nobody can regenerate is a table nobody can check. The
    published array is in the test, which is what pins this against it.

    Exponentiated with Python integers rather than in the field. Both are exact,
    and this way a reader comparing the source to Appendix B sees the values the
    standard prints rather than Montgomery representatives.

    Slot 0 is unused — Algorithms 41 and 42 index from 1 — and holds `zeta^0 = 1`
    where the standard prints a placeholder 0.
    """
    return np.array([pow(ZETA, bit_rev8(k), Q) for k in range(N)], dtype=FIELD)


ZETAS = _zeta_table()


def base_mul(a_hat: ArrayLike, b_hat: ArrayLike) -> Any:
    """FIPS 204 Algorithm 45 — multiplication in `T_q`, which is pointwise.

    Named for the shape convention the lattice pages share
    ([`conventions.md`](../../../docs/reference/conventions.md)): ML-KEM's
    equivalent multiplies degree-1 polynomials, and having the two under one name
    is what makes the difference visible instead of hidden.
    """
    xnp = namespace(a_hat, b_hat)
    return xnp.asarray(a_hat) * xnp.asarray(b_hat)


def _butterfly_layers(xnp: Any, w: Any, lengths: list[int], forward: bool) -> Any:
    """The shared layer walk of FIPS 204 Algorithms 41 and 42.

    The standard's `while` over `len` and its inner `while` over `start` are a
    static schedule — the trip counts come from `n = 256`, never from the data —
    so the outer walk is a Python loop and each layer is one batched operation
    over every block at once. The `for` over `j` inside a block becomes the
    trailing axis; the blocks become the axis above it.

    `w` arrives as `[..., 256]` and keeps that shape, so a vector of `k` or `l`
    polynomials is one call with no Python loop over the vector axis.
    """
    lead = w.shape[:-1]
    index = 0 if forward else N
    for length in lengths:
        blocks = N // (2 * length)
        # `[..., blocks, 2, length]` puts a block's two halves on their own axis.
        pairs = w.reshape(*lead, blocks, 2, length)
        low, high = pairs[..., 0, :], pairs[..., 1, :]

        if forward:
            zetas = ZETAS[index + 1 : index + 1 + blocks]
            index += blocks
        else:
            # Algorithm 42 walks the table downwards and negates: `z = -zetas[m]`.
            zetas = (-ZETAS[index - blocks : index])[::-1]
            index -= blocks
        z = xnp.asarray(zetas.reshape(*(1,) * len(lead), blocks, 1))

        if forward:
            t = z * high
            new_low, new_high = low + t, low - t
        else:
            new_low, new_high = low + high, z * (low - high)

        w = xnp.stack([new_low, new_high], axis=-2).reshape(*lead, N)
    return w


def ntt(w: ArrayLike) -> Any:
    """FIPS 204 Algorithm 41 — the forward NTT, batched over leading axes.

    Takes `[..., 256]` coefficients in `R_q` and returns `[..., 256]` in `T_q`.
    """
    xnp = namespace(w)
    lengths = [N >> (i + 1) for i in range(8)]
    return _butterfly_layers(xnp, xnp.asarray(w), lengths, forward=True)


def intt(w_hat: ArrayLike) -> Any:
    """FIPS 204 Algorithm 42 — the inverse NTT, batched over leading axes.

    The trailing scale by `f = 256^-1 mod q` is the standard's lines 21-24.
    """
    xnp = namespace(w_hat)
    lengths = [1 << i for i in range(8)]
    w = _butterfly_layers(xnp, xnp.asarray(w_hat), lengths, forward=False)
    return w * xnp.asarray(np.array(_N_INVERSE, dtype=FIELD))


def _canonical(xnp: Any, r: ArrayLike) -> Any:
    """`r` as canonical integers in `[0, q)`, on a signed lane.

    `astype` and not a bitcast, per the module docstring. Signed because every
    caller goes on to produce a `mod±` residue.
    """
    return xnp.asarray(r).astype(np.uint32).astype(np.int32)


def _centered_mod(xnp: Any, r: Any, modulus: int) -> Any:
    """`r mod± modulus` in `(-modulus/2, modulus/2]`, for non-negative `r`.

    FIPS 204 §2.3's `mod±`, which is what Power2Round and Decompose reduce by.
    """
    low = r % np.int32(modulus)
    return xnp.where(low > modulus // 2, low - np.int32(modulus), low)


def power2round(r: ArrayLike) -> tuple[Any, Any]:
    """FIPS 204 Algorithm 35: `r ≡ r1·2^d + r0 mod q`, returning `(r1, r0)`.

    `r0` is signed — it is a `mod±` residue — so both halves come back `int32`.
    """
    xnp = namespace(r)
    r_plus = _canonical(xnp, r)
    r0 = _centered_mod(xnp, r_plus, 1 << D)
    return (r_plus - r0) >> np.int32(D), r0


def decompose(r: ArrayLike, gamma2: int) -> tuple[Any, Any]:
    """FIPS 204 Algorithm 36: `r ≡ r1·2γ2 + r0 mod q`, returning `(r1, r0)`.

    The carry case — `r+ - r0 = q - 1`, where straightforward rounding would put
    `r1` one step past the top of its range — is the whole reason Decompose exists
    rather than Power2Round being reused, so it is written as the standard's
    branch rather than folded into arithmetic.
    """
    xnp = namespace(r)
    alpha = 2 * gamma2
    r_plus = _canonical(xnp, r)
    r0 = _centered_mod(xnp, r_plus, alpha)
    carries = (r_plus - r0) == np.int32(Q - 1)
    r1 = xnp.where(carries, np.int32(0), (r_plus - r0) // np.int32(alpha))
    return r1, xnp.where(carries, r0 - np.int32(1), r0)


def high_bits(r: ArrayLike, gamma2: int) -> Any:
    """FIPS 204 Algorithm 37 — `r1` from Decompose."""
    return decompose(r, gamma2)[0]


def low_bits(r: ArrayLike, gamma2: int) -> Any:
    """FIPS 204 Algorithm 38 — `r0` from Decompose."""
    return decompose(r, gamma2)[1]


def make_hint(z: ArrayLike, r: ArrayLike, gamma2: int) -> Any:
    """FIPS 204 Algorithm 39 — whether adding `z` to `r` moves its high bits."""
    xnp = namespace(z, r)
    total = xnp.asarray(r) + xnp.asarray(z)
    return high_bits(r, gamma2) != high_bits(total, gamma2)


def use_hint(h: ArrayLike, r: ArrayLike, gamma2: int) -> Any:
    """FIPS 204 Algorithm 40 — the high bits of `r`, adjusted by hint `h`."""
    xnp = namespace(h, r)
    m = (Q - 1) // (2 * gamma2)
    r1, r0 = decompose(r, gamma2)
    hinted = xnp.asarray(h).astype(bool)
    step = xnp.where(r0 > np.int32(0), np.int32(1), np.int32(-1))
    return xnp.where(hinted, (r1 + step) % np.int32(m), r1)


def infinity_norm(w: ArrayLike) -> Any:
    """`||w||∞` — FIPS 204 §2.3, the largest `|w_j mod± q|` over the trailing axis.

    The rejection bounds in signing are all stated against this, and it is a
    reduction rather than a comparison so that a caller can compare it to `beta`
    or `gamma1` itself.
    """
    xnp = namespace(w)
    canonical = _canonical(xnp, w)
    return xnp.max(xnp.minimum(canonical, np.int32(Q) - canonical), axis=-1)
