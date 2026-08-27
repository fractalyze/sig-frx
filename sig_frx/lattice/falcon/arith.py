# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Arithmetic in `Z_q` and the NTT over `q = 12289`, for Falcon (FN-DSA).

Verification is the only Falcon operation that lives *entirely* in this ring —
it recovers `s1 = c − s2·h mod q` and measures a norm. Key generation is mostly
elsewhere, over `Q[x]/(x^n + 1)` in the complex domain, which is a different
transform with a precision requirement of its own and is not here; but two of
its steps are `Z_q` and do reach in, so this module is the scheme's `Z_q` and
not verification's alone. Those two are Algorithm 5 line 7's invertibility test
and Algorithm 4 line 9's `h = g·f^{-1}`, and they live in
[`keygen.py`](keygen.py) — what belongs here is the arithmetic, not the step.

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
[the byte-exactness rule](../../../docs/reference/testing.md) refuses when it
declines to pin against another implementation.

What is left to gate on is the property that *is* the specification: the
transform has to compute negacyclic convolution. `arith_test.py` checks it
against exact integer arithmetic rather than against a table.

`q − 1 = 12288 = 2^12 · 3`, so roots of order up to 4096 exist and a length-`n`
negacyclic transform needs a primitive `2n`-th one — 1024 and 2048 for Falcon's
two parameter sets, both inside it.

## The surface stops at `base_mul`, and a composed `mul` is not an omission

`ntt` / `intt` / `base_mul` is the aligned set, and there is no `a · b` on top of
them. Verification computes one product, `s1 = c − s2·h`, and the composed form
would transform `h` inside every call: a third of the transform work, for an
operand that is the public key and does not change per signature. Under a batch
that assembles by vmapping a one-signature body — this repo's pattern — `h` is
lifted onto the batch axis and the same NTT is computed `B` times over identical
rows, which nothing downstream can common up because the rows are real data.

So the caller hoists `ntt(h)` and stays in the transform domain, and the shape
that would have invited otherwise is absent rather than documented against.

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


def _check_degree(w: ArrayLike) -> None:
    """Refuses a transform length Falcon does not define.

    Reads the shape and nothing else, so under a tracer it is a metadata test
    that emits no HLO.
    """
    n = np.shape(w)[-1]
    if n not in DEGREES:
        raise ValueError(f"degree {n} is not a Falcon parameter set: {DEGREES}")


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

    Written the way ML-DSA's `centered` is written, down to the `where`, so that
    diffing the two files shows one modulus differing and nothing else — the
    cost [`conventions.md`](../../../docs/reference/conventions.md) names is not
    the duplicated lines but two adaptations that look unrelated. The leading
    `%` is what carries the same property theirs has: a value already centered
    is its own representative, which a norm over `LowBits`-style output needs.
    """
    xnp = namespace(w)
    canonical = xnp.asarray(w).astype(np.uint32).astype(np.int32)
    low = canonical % np.int32(Q)
    return xnp.where(low > Q // 2, low - np.int32(Q), low)


def ntt(w: ArrayLike) -> Any:
    """The forward negacyclic NTT, batched over leading axes.

    Takes `[..., n]` coefficients as `FIELD` and returns `[..., n]` in the
    transform domain. A raw integer array is refused rather than read as a
    residue: the opcode reads the field's algebra to derive its root.

    **No `generator=`, deliberately** — where ML-DSA's call pins one. The
    opcode's default root is taken, and what makes that safe is a property of
    the scheme rather than of this function: no transform-domain value is ever
    serialized, hashed or compared, so the root cancels inside every call that
    introduces it. A step that broke that invariant would need the pin back, and
    would not fail a round trip or a convolution check when it did — see the
    module docstring and
    [`conventions.md`](../../../docs/reference/conventions.md).

    The result is a device array whichever namespace the input arrived in — the
    transform is an opcode with no host implementation, so the lift is forced
    rather than chosen.
    """
    _check_degree(w)
    return lax.ntt(w, ntt_type=lax.NttType.NEGACYCLIC_NTT)


def intt(w_hat: ArrayLike) -> Any:
    """The inverse negacyclic NTT, batched over leading axes.

    The trailing scale by `n^-1` is the opcode's, not ours: its inverse mode
    applies it. Unpinned for the reason `ntt` gives.
    """
    _check_degree(w_hat)
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


def base_div(a_hat: ArrayLike, b_hat: ArrayLike) -> Any:
    """Division in the transform domain, which is pointwise and exact.

    **The one entry in this set the other two lattice schemes have no
    counterpart for**, and it is the standard that asks for it rather than a
    convenience: Algorithm 4 line 9 builds the public key as `h = g·f^{-1}`, and
    §3.11.5 recovers the unencoded `G` as `(q + gF)/f`. Neither ML-DSA nor
    ML-KEM ever divides, so this has no shared name to match and says so instead
    of looking like an omission from theirs.

    The inversion is the field dtype's, not this module's — the same rule
    `base_mul` follows, one operation further. A zero divisor is what
    [`keygen.invertible`](keygen.py) refuses before it can get here; the
    standard's own check is Algorithm 5 line 7 and it is a rejection rather than
    an error, so the guard belongs there and not in an arithmetic primitive.
    """
    xnp = namespace(a_hat, b_hat)
    return xnp.asarray(a_hat) / xnp.asarray(b_hat)
