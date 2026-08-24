# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Arithmetic in `Z_q` and the NTT over `q = 12289`, for Falcon (FN-DSA).

Verification is the only Falcon operation that lives entirely in this ring — it
recovers `s1 = c − s2·h mod q` and measures a norm — so this module is what
verification needs and nothing more. Key generation and signing work over
`Q[x]/(x^n + 1)` in the complex domain instead, which is a different transform
with a precision requirement of its own and is not here.

## The field is a dtype, and so is the transform

`q` is not one of `zk_dtypes`' curated fields and does not have to be: a field
minted from the modulus reduces internally, so `+`, `-` and `*` on `FIELD` are
already modular and nothing here implements them. `frx.lax.ntt`'s
`NEGACYCLIC_NTT` mode is this ring's transform, and it derives its root from the
runtime modulus rather than matching a curated family. This is the third lattice
scheme over that one opcode, and the shape it is written in is shared on purpose
([`conventions.md`](../../../docs/reference/conventions.md)).

## What ML-DSA pins here, and why this scheme must not

The shared shape says a scheme pins two things the opcode leaves open — which
root it transforms with, and which order it returns. **Falcon pins neither, and
that is a property of the standard rather than an omission.**

FIPS 204 has to pin them because its NTT domain is *observable*: `ExpandA`
samples `Â` directly from the seed, so a public key commits to one root, and
`BitRev8` fixes the index order the standard's tables are written in. Falcon
does neither. Its public key is `h` in the coefficient domain, its signature is a
compressed coefficient-domain `s`, and nothing anywhere is sampled in the NTT
domain — the transform is an internal route to `mul_zq` and never leaves.

So any primitive `2n`-th root gives the same `h`, the same signature and the
same verdict, and the opcode's default is one. Pinning a root here would import
a convention no standard states, and reproducing a particular reference's
intermediate values would gate this repo on that implementation's private
choice — the thing
[the byte-exactness rule](../../../docs/reference/conventions.md) refuses when it
declines to pin against another implementation.

What is left to gate on is the property that *is* the specification: the
transform has to compute negacyclic convolution. `arith_test.py` checks it
against exact integer arithmetic rather than against a table.

`q − 1 = 12288 = 2^12 · 3`, so roots of order up to 4096 exist and a length-`n`
negacyclic transform needs a primitive `2n`-th one — 1024 and 2048 for Falcon's
two parameter sets, both inside it.

## Two representations, on purpose

The transform works on `FIELD`; a norm does not. Falcon bounds `‖(s1, s2)‖²`
over *centered* representatives, so a residue near `q` counts as the small
negative number it stands for and not as a large positive one. `centered` and
`to_field` are that boundary, and both live here because a scheme above should
never be reducing by hand.

The conversion is `astype` and **never a bitcast**: the field's storage is a
Montgomery representative, so reinterpreting the bytes gives a different number,
and wrongly in a way no round trip reveals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import zk_dtypes
from frx import lax
from frx.typing import ArrayLike

from sig_frx.arrays import namespace

# Falcon §2.4. `q` is shared by both parameter sets; the degree is not, so it is
# read off the array rather than fixed here.
Q = 12289

# The degrees Falcon defines. A transform length outside this set would still be
# a correct negacyclic transform and not a Falcon one, so it is refused rather
# than silently accepted — the mistake it catches is a mis-shaped array, which
# otherwise reaches the norm check as a wrong verdict.
DEGREES = (512, 1024)

FIELD = zk_dtypes.prime_field(Q)


def _checked_degree(w: ArrayLike) -> int:
    """The transform length, refused unless Falcon defines it."""
    n = np.shape(w)[-1]
    if n not in DEGREES:
        raise ValueError(f"degree {n} is not a Falcon parameter set: {DEGREES}")
    return n


def to_field(w: ArrayLike) -> Any:
    """Centered coefficients as `FIELD` residues — the inverse of `centered`.

    The reduction comes first: a negative coefficient is not a residue the field
    dtype can read.
    """
    xnp = namespace(w)
    return (xnp.asarray(w, dtype=np.int32) % np.int32(Q)).astype(FIELD)


def centered(w: ArrayLike) -> Any:
    """`FIELD` residues as centered integers in `(−q/2, q/2]`.

    What a norm has to be measured over: `q − 1` is the coefficient `−1`, and
    summing its square as `(q − 1)²` would reject every honest signature.
    """
    xnp = namespace(w)
    canonical = xnp.asarray(w).astype(np.uint32).astype(np.int32)
    return canonical - np.int32(Q) * (canonical > np.int32(Q // 2))


def ntt(w: ArrayLike) -> Any:
    """The forward negacyclic NTT, batched over leading axes.

    Takes `[..., n]` coefficients as `FIELD` and returns `[..., n]` in the
    transform domain. A raw integer array is refused rather than read as a
    residue: the opcode reads the field's algebra to derive its root.

    The result is a device array whichever namespace the input arrived in — the
    transform is an opcode with no host implementation, so the lift is forced
    rather than chosen
    ([`conventions.md`](../../../docs/reference/conventions.md)).
    """
    _checked_degree(w)
    return lax.ntt(w, ntt_type=lax.NttType.NEGACYCLIC_NTT)


def intt(w_hat: ArrayLike) -> Any:
    """The inverse negacyclic NTT, batched over leading axes.

    The trailing scale by `n^-1` is the opcode's, not ours: its inverse mode
    applies it.
    """
    _checked_degree(w_hat)
    return lax.ntt(w_hat, ntt_type=lax.NttType.NEGACYCLIC_INTT)


def base_mul(a_hat: ArrayLike, b_hat: ArrayLike) -> Any:
    """Multiplication in the transform domain, which is pointwise.

    Named for the shape convention the lattice pages share
    ([`conventions.md`](../../../docs/reference/conventions.md)). Falcon
    transforms the whole ring at once, so this is elementwise where ML-KEM's
    equivalent multiplies degree-1 polynomials — having the two under one name
    is what makes that difference visible instead of hidden.
    """
    xnp = namespace(a_hat, b_hat)
    return xnp.asarray(a_hat) * xnp.asarray(b_hat)


def mul(a: ArrayLike, b: ArrayLike) -> Any:
    """`a · b` in `Z_q[x]/(x^n + 1)`, over `FIELD` coefficients.

    The round trip through the transform domain is an implementation of the ring
    multiplication and not a second operation, which is why verification calls
    this rather than assembling the three steps itself.
    """
    return intt(base_mul(ntt(a), ntt(b)))
