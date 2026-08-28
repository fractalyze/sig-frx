# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""§3.9's signing arithmetic — Algorithm 11's tree walk, and Algorithm 10 over it.

Signing needs a lattice point near a target `t`, and "near" has to mean *drawn
from the right distribution* rather than *closest*: rounding each coordinate
independently would produce signatures whose deviation from `t` reveals the
basis that produced them. Algorithm 11 is what replaces rounding. It descends
the tower of rings the tree was built over, and at each of the `n` leaves draws
two integers with [`sampler.sampler_z`](sampler.py) at a centre and a width the
path has determined.

## What the tree contributes

[`keygen.ffldl`](keygen.py) holds the tree one array per depth rather than one
object per node, and this walks it by index: at depth `d`, node `i` is
`values[d][i]`, its children are `2i` and `2i + 1`, and past the last depth the
leaf is `leaves[i]`. `_weave` put the `D00` child at `2i` and the `D11` child at
`2i + 1`, which are Algorithm 9's `leftchild` and `rightchild` — so line 8's
first recursive call, into `T1`, is the **odd** index, and line 12's into `T0`
is the even one. Descending them in the other order still returns a lattice
point; it is just not the one the tree's own `L10` corrects for.

**The leaf is already `σ'`.** `keygen.normalize` divided `σ` into the root of
each leaf, so line 2's `σ' ← T.value` is a read. `sampler_z` takes `1/σ'`, so
the reciprocal is formed here.

## Host code, following the sampler

This is a scalar recursion in Python for the reason
[`sampler.py`](sampler.py) records at length: the sampler underneath it has a
secret-dependent rejection loop with no traced form, and signing is not the
path this repo claims (see
[`security.md`](../../../docs/reference/security.md)). Every array here is a
numpy `complex128`, so `fft.split` and `fft.merge` need no precision scope —
that scope exists for the traced path, where `float64` would silently narrow.

## The loop above the walk

[`lattice_point`](#lattice_point) is Algorithm 10 lines 3-9 for **one** draw,
and [`signing_basis`](#signing_basis) is lines 3-7 of Algorithm 4, which a key
loaded from §3.11.5 has to redo. What is *not* here is the salt, the challenge,
the encoding and the loop that ties them — lines 1-2 and 4-12 — which live on
the seam in [`falcon.py`](falcon.py), because they are where a signature stops
being a lattice point and becomes bytes. The split is the same one `keygen`
draws: the arithmetic is here and the expansion of a caller's seed is there.

The loop is at the seam rather than split across both because Algorithm 10's
two rejections restart at the same line, and a budget on each would multiply.

## `split` is this repo's, and it is not the reference's

`fft.split` pairs index `i` with `i + n/2` — a root with its negative — where
the reference pairs *adjacent* indices, because its representation is
bit-reversed. Both are correct for their own layout, and `merge(split(f)) == f`
holds for both, so a round trip cannot tell them apart. The tree in
[`keygen.py`](keygen.py) was built through this one, so the walk has to use it
throughout; mixing the two would produce a point that verifies against nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sig_frx.lattice.falcon import arith, fft, keygen, sampler
from sig_frx.lattice.falcon.sampler import RandomBytes


def _leaf(
    t0: Any, t1: Any, sigma: float, degree: int, randomness: RandomBytes
) -> tuple[Any, Any]:
    """Algorithm 11 lines 1-5 — the `n = 1` case, which is two draws.

    At ring degree 1 the transform is the identity, so `t0` and `t1` are the
    rationals themselves and the samples are integers rather than polynomials.
    They arrive as one-element `complex128` arrays all the same, because that is
    what the level above split; the imaginary part is zero up to the transform's
    own error and is dropped rather than carried into a centre.
    """
    inverse_sigma = 1.0 / sigma
    return (
        np.array(
            [sampler.sampler_z(float(t0[0].real), inverse_sigma, degree, randomness)],
            dtype=np.complex128,
        ),
        np.array(
            [sampler.sampler_z(float(t1[0].real), inverse_sigma, degree, randomness)],
            dtype=np.complex128,
        ),
    )


def _walk(
    t0: Any,
    t1: Any,
    tree: keygen.FalconTree,
    depth: int,
    index: int,
    degree: int,
    randomness: RandomBytes,
) -> tuple[Any, Any]:
    """Algorithm 11 at one node, recursing into its two children."""
    if depth == len(tree.values):
        return _leaf(t0, t1, float(tree.leaves[index]), degree, randomness)

    l10 = tree.values[depth][index]

    # Lines 7-9. The right child first, and it is the odd index — see the module
    # docstring on `_weave`.
    z1 = fft.merge(
        *_walk(*fft.split(t1), tree, depth + 1, 2 * index + 1, degree, randomness)
    )
    # Line 10: the correction that makes this a sampler over the lattice rather
    # than `2n` independent draws — what the first child rounded away is carried
    # into the second child's centre.
    t0_corrected = t0 + (t1 - z1) * l10
    # Lines 11-13.
    z0 = fft.merge(
        *_walk(*fft.split(t0_corrected), tree, depth + 1, 2 * index, degree, randomness)
    )
    return z0, z1


def ff_sampling(
    t0: Any, t1: Any, tree: keygen.FalconTree, randomness: RandomBytes
) -> tuple[Any, Any]:
    """Algorithm 11 — `(z0, z1)` near `(t0, t1)`, in FFT representation.

    `t0` and `t1` are the target in the transform domain, `tree` is a *Falcon*
    tree (out of `keygen.normalize`, not out of `keygen.ffldl` — the leaves have
    to be deviations rather than squared norms), and `randomness` is the byte
    source every `sampler_z` call underneath reads from.

    The result is complex arrays whose entries are integers up to the
    transform's error, which is what the caller's `invFFT` rounds. They are not
    rounded here: line 7 of Algorithm 10 subtracts them from `t` in the
    transform domain, so rounding at this boundary would be undone immediately.
    """
    degree = np.shape(t0)[-1]
    if np.shape(t1)[-1] != degree:
        raise ValueError(
            f"the target's two halves have degrees {degree} and "
            f"{np.shape(t1)[-1]}; they are one point"
        )
    if degree != 2 ** len(tree.values):
        raise ValueError(
            f"a degree-{degree} target needs a tree of depth "
            f"{degree.bit_length() - 1}, got {len(tree.values)}"
        )
    return _walk(
        np.asarray(t0, dtype=np.complex128),
        np.asarray(t1, dtype=np.complex128),
        tree,
        0,
        0,
        degree,
        randomness,
    )


@dataclass(frozen=True)
class SigningBasis:
    """`B̂` and the Falcon tree, as a loaded §3.11.5 key becomes them.

    Algorithm 4 lines 3-7 build both at generation time and §3.11.5 encodes
    neither, so a key that arrives as bytes rebuilds them — which is what makes
    this a record rather than four loose arrays: the four transforms and the
    tree come from one pass and every consumer needs both halves.
    """

    f_hat: Any
    g_hat: Any
    big_f_hat: Any
    big_g_hat: Any
    tree: keygen.FalconTree


def signing_basis(f: Any, g: Any, big_f: Any, sigma: float) -> SigningBasis:
    """Algorithm 4 lines 3-7 over the three polynomials §3.11.5 does carry.

    `G` is recovered rather than decoded — that is the quarter of the trapdoor
    the encoding leaves out — and the four transforms are shared: `gram` takes
    them, and so does [`target`](#target), so transforming per call site would
    do it twice.
    """
    f_hat, g_hat, big_f_hat, big_g_hat = (
        fft.fft(np.asarray(p, dtype=np.float64))
        for p in (f, g, big_f, keygen.recover_g(f, g, big_f))
    )
    gram = keygen.gram(f_hat, g_hat, big_f_hat, big_g_hat)
    return SigningBasis(
        f_hat=f_hat,
        g_hat=g_hat,
        big_f_hat=big_f_hat,
        big_g_hat=big_g_hat,
        tree=keygen.normalize(keygen.ffldl(*gram), sigma),
    )


def target(challenge: Any, f_hat: Any, big_f_hat: Any) -> tuple[Any, Any]:
    """Algorithm 10 line 3 — `t = (FFT(c), FFT(0)) · B̂⁻¹`, in the transform domain.

    `B̂⁻¹` is not formed: `det B = q` by the NTRU equation, so the inverse of
    `[[g, −f], [G, −F]]` is `[[−F, f], [−G, g]]/q` and the row `(c, 0)` times it
    is the two products below. Writing it out is what keeps the division a
    single scalar `q` rather than a polynomial inversion.

    The second component carries `+f` and the first `−F`, which is the pairing
    that makes `s2` come out as the polynomial §3.11.3 compresses.
    """
    c_hat = fft.fft(np.asarray(challenge, dtype=np.float64))
    return -(c_hat * big_f_hat) / arith.Q, (c_hat * f_hat) / arith.Q


def lattice_point(
    challenge: Any, basis: SigningBasis, randomness: RandomBytes
) -> tuple[Any, int]:
    """Algorithm 10 lines 3-9, once — `(s2, ‖s‖²)` for a single draw.

    One attempt rather than a loop, because Algorithm 10's two rejections both
    restart at line 4 and only the caller can see the second of them: line 8
    weighs the norm returned here, line 11 weighs whether `Compress` fits. Two
    loops with a budget each would multiply — the product, not either number,
    would be what a wrong basis costs before it raised — so the loop and its
    bound live once, at the seam ([`falcon.py`](falcon.py)).

    **The norm is measured on the rounded integers, not on the transform.**
    (3.8) allows the check in the FFT domain and the reference does not use it
    either — line 9's `invFFT` has to happen regardless, and a float norm can
    sit on the far side of the bound from the integers that are actually
    encoded. What is compared is what a verifier will compare
    ([`falcon._within_bound`](falcon.py)).

    `s1` is returned only through that norm. It is not part of the signature —
    §3.11.3 encodes `s2` alone and a verifier recovers `s1` as `c − s2·h` — but
    the bound is on the pair, so dropping it would measure half a signature.
    """
    t0, t1 = target(challenge, basis.f_hat, basis.big_f_hat)
    z0, z1 = ff_sampling(t0, t1, basis.tree, randomness)
    # Line 7. `B = [[g, −f], [G, −F]]`, so `(t − z)B̂` is these two rows.
    offset0, offset1 = t0 - z0, t1 - z1
    first = offset0 * basis.g_hat + offset1 * basis.big_g_hat
    second = -(offset0 * basis.f_hat + offset1 * basis.big_f_hat)
    # Line 9, brought above the bound for the reason in the docstring.
    s1 = np.rint(np.real(fft.ifft(first))).astype(np.int64)
    s2 = np.rint(np.real(fft.ifft(second))).astype(np.int64)
    # Line 8's quantity. `2n·(q/2)²` is 2^37 at Falcon-1024, so this needs a
    # lane wider than the repo's 32 bits — which host `int64` is, and which is
    # the other reason this operation never became traced code.
    return s2, int((s1 * s1).sum() + (s2 * s2).sum())
