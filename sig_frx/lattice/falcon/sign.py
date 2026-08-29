# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""§3.9.2's fast Fourier sampler — Algorithm 11's walk of the Falcon tree.

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

## `split` is this repo's, and it is not the reference's

`fft.split` pairs index `i` with `i + n/2` — a root with its negative — where
the reference pairs *adjacent* indices, because its representation is
bit-reversed. Both are correct for their own layout, and `merge(split(f)) == f`
holds for both, so a round trip cannot tell them apart. The tree in
[`keygen.py`](keygen.py) was built through this one, so the walk has to use it
throughout; mixing the two would produce a point that verifies against nothing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from frx.typing import ArrayLike

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


def signing_basis(
    f: ArrayLike, g: ArrayLike, big_f: ArrayLike, sigma: float
) -> tuple[tuple[Any, Any, Any, Any], keygen.FalconTree]:
    """Algorithm 4 lines 3-7 over the three polynomials §3.11.5 carries.

    A key that arrives as bytes has neither `B̂` nor the tree — the encoding
    stores `f`, `g` and `F` and `keygen` deliberately builds no tree — so both
    are rebuilt on load. `G` is recovered rather than decoded, being the quarter
    of the trapdoor the encoding leaves out.

    Returned together because they come from one pass and every caller needs
    both: `gram` consumes the four transforms and so does [`target`](#target),
    so a caller that rebuilt either half would transform twice. Here rather than
    at the seam for the reason this module exists — it is the arithmetic, and
    the seam is the salt, the encoding and the loop.
    """
    basis_hat = tuple(
        fft.fft(np.asarray(entry, dtype=np.float64))
        for entry in (f, g, big_f, keygen.recover_g(f, g, big_f))
    )
    tree = keygen.normalize(keygen.ffldl(*keygen.gram(*basis_hat)), sigma)
    return basis_hat, tree


def target(point: ArrayLike, f_hat: Any, big_f_hat: Any) -> tuple[Any, Any]:
    """Algorithm 10 line 3 — `t = (FFT(c), FFT(0)) · B̂⁻¹`.

    Written out rather than as a matrix solve, because `B̂⁻¹` has a closed form
    for `B = [[g, −f], [G, −F]]` whose determinant is `q`: the second row of the
    product is what `(c, 0)` picks out, so both halves are `FFT(c)` times one
    basis polynomial over `q`, and the `−` is the one the inverse carries.
    """
    c = fft.fft(np.asarray(point, dtype=np.float64))
    return -(c * big_f_hat) / arith.Q, (c * f_hat) / arith.Q


def attempt(
    t0: Any,
    t1: Any,
    tree: keygen.FalconTree,
    basis_hat: tuple[Any, Any, Any, Any],
    bound: int,
    randomness: RandomBytes,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Algorithm 10 lines 6-9 — `(s1, s2)`, or `None` where line 8 rejects.

    One trip, not the loop: the trip count is data and a caller that restarts
    needs fresh bytes, which is the same split `keygen.ntru_gen` makes for
    Algorithm 5. It also makes this deterministic in `randomness` and so
    testable without a hash.

    The bound is on the **pair**. `‖(s1, s2)‖² > ⌊β²⌋` is a property of the
    whole lattice point, and checking `s2` alone — the half that gets encoded —
    would accept points a verifier rejects, since `verify` reconstructs `s1`
    from `s2` and holds the pair to the same number.
    """
    z0, z1 = ff_sampling(t0, t1, tree, randomness)
    f_hat, g_hat, big_f_hat, big_g_hat = basis_hat
    d0, d1 = t0 - z0, t1 - z1
    # Line 7's `(t − z)B̂`, with `B = [[g, −f], [G, −F]]`.
    s1 = fft.ifft(d0 * g_hat + d1 * big_g_hat)
    s2 = fft.ifft(-(d0 * f_hat + d1 * big_f_hat))
    # Line 9. The coefficients are integers up to the transform's error, which
    # is what `rint` removes — they are a lattice point by construction, so a
    # rounding here is reading the answer rather than choosing one.
    first = np.rint(np.real(s1)).astype(np.int64)
    second = np.rint(np.real(s2)).astype(np.int64)
    if int((first * first).sum() + (second * second).sum()) > bound:
        return None
    return first, second
