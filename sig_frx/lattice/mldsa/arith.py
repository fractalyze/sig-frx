# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Arithmetic in `Z_q` and the NTT over `q = 8380417`, for ML-DSA (FIPS 204).

Every polynomial multiplication in ML-DSA goes through the NTT, and the rounding
functions are where the scheme's correctness lives, so both are here and both are
testable without any scheme logic above them.

## The field is a dtype, and so is the transform

`q` is not one of `zk_dtypes`' curated fields, but it does not have to be: a field
minted from the modulus reduces internally, so `+`, `-` and `*` on `FIELD` are
already modular and nothing here implements them. That is what keeps this short,
and the alternative is worth naming — hand-written modular arithmetic means a
product of two residues (2^46) against a 32-bit lane with no widening multiply, so
every operation becomes a limb split carrying a bound that has to be argued.

The transform is delegated the same way. `frx.lax.ntt`'s `NEGACYCLIC_NTT` mode
is this ring's transform and takes a field minted from a modulus, because its
generated kernel derives the root from the runtime modulus rather than matching
a curated family. Two things it does not decide for us, and both are below: which
root it transforms with, and which order it returns.

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

import itertools
from typing import Any

import numpy as np
import zk_dtypes
from frx import lax
from frx.typing import ArrayLike

from sig_frx.arrays import namespace

# FIPS 204 Table 1. `d` and `zeta` are fixed across the parameter sets; `gamma2`
# is not, so it is an argument rather than a constant here.
Q = 8380417
ZETA = 1753
D = 13
N = 256

FIELD = zk_dtypes.prime_field(Q)


def bit_rev8(m: int) -> int:
    """FIPS 204 Algorithm 43: reverse the bits of a byte.

    A host function on purpose — it builds the output permutation, which is
    static, so this never sees a traced value.
    """
    return int(format(m & 0xFF, "08b")[::-1], 2)


# `BitRev8` as an index vector. The op returns natural order —
# `out[k] = w(zeta^(2k+1))` — and FIPS 204 indexes the same values by
# `BitRev8`, so the two differ by this gather and nothing else. It is an
# involution, so the same vector converts in both directions.
#
# Applied with `take`, not `arr[..., idx]`: the latter runs JAX's advanced-index
# normalisation per call, which costs more than the transform on the host path.
# `lax.bit_reverse` names this permutation exactly and is byte-equal, but its
# lowering materialises a transposed copy and is several times slower at batch
# shapes.
_BIT_REV8 = np.array([bit_rev8(k) for k in range(N)], dtype=np.int32)


def _generator() -> int:
    """The `g` the opcode needs in order to land on the standard's `zeta`.

    `lax.ntt` takes a generator of the multiplicative group, not the root: it
    derives the root itself as `g^((q-1)/2N)`. `zeta` cannot be passed in its
    place — it is a square, and no power of a square is primitive — so the pin
    is its preimage under that map, searched rather than transcribed for the
    reason the standard's own tables are generated here: a constant nobody can
    regenerate is a constant nobody can check.

    Left unpinned the opcode finds *a* primitive `2N`-th root, which is a
    correct negacyclic transform and a wrong FIPS 204 — and it round-trips and
    convolves either way, so only the published vectors would catch it.
    """
    exponent = (Q - 1) // (2 * N)
    return next(g for g in itertools.count(2) if pow(g, exponent, Q) == ZETA)


GENERATOR = _generator()


def base_mul(a_hat: ArrayLike, b_hat: ArrayLike) -> Any:
    """FIPS 204 Algorithm 45 — multiplication in `T_q`, which is pointwise.

    Named for the shape convention the lattice pages share
    ([`conventions.md`](../../../docs/reference/conventions.md)): ML-KEM's
    equivalent multiplies degree-1 polynomials, and having the two under one name
    is what makes the difference visible instead of hidden.
    """
    xnp = namespace(a_hat, b_hat)
    return xnp.asarray(a_hat) * xnp.asarray(b_hat)


def ntt(w: ArrayLike) -> Any:
    """FIPS 204 Algorithm 41 — the forward NTT, batched over leading axes.

    Takes `[..., 256]` coefficients in `R_q` and returns `[..., 256]` in `T_q`,
    both as `FIELD`. A raw integer array is refused rather than read as a
    residue: the opcode reads the field's algebra to derive its root, which is
    the same reason the rounding functions convert with `astype` and never a
    bitcast.

    The result is a device array whichever namespace the input arrived in: the
    transform is an opcode with no host implementation, so the lift is forced
    rather than chosen ([`conventions.md`](../../../docs/reference/conventions.md)).
    `sampling.py` lifts too, for the different reason that it hashes.
    """
    hat = lax.ntt(w, ntt_type=lax.NttType.NEGACYCLIC_NTT, generator=GENERATOR)
    return namespace(hat).take(hat, _BIT_REV8, axis=-1)


def intt(w_hat: ArrayLike) -> Any:
    """FIPS 204 Algorithm 42 — the inverse NTT, batched over leading axes.

    The standard's trailing scale by `256^-1` is the opcode's, not ours: its
    inverse mode applies it.
    """
    xnp = namespace(w_hat)
    natural = xnp.take(w_hat, _BIT_REV8, axis=-1)
    return lax.ntt(natural, ntt_type=lax.NttType.NEGACYCLIC_INTT, generator=GENERATOR)


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


def centered(r: ArrayLike) -> Any:
    """`r mod± q` — §2.3's representative in `(-q/2, q/2]`, as int32.

    Signing needs it twice over: the bounds it rejects on are stated against this
    representative, and `sigEncode` packs `z mod± q` rather than `z`. It takes a
    residue in either form — a `FIELD` element or a canonical integer — because
    `_canonical` reads both, and a value already centered is its own `mod±`.
    """
    xnp = namespace(r)
    return _centered_mod(xnp, _canonical(xnp, r), Q)


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
    or `gamma1` itself. Expressed through `centered` so that the norm and the
    representative it is defined over cannot drift — and so that it answers for
    an already-signed argument, which `LowBits` produces and signing measures.
    """
    xnp = namespace(w)
    return xnp.max(xnp.abs(centered(w)), axis=-1)
