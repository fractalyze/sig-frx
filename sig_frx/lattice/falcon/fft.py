# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The transform over `Q[x]/(x^n + 1)`, for Falcon's key generation and signing.

Verification lives in `Z_q` and uses the integer NTT ([`arith.py`](arith.py)).
Key generation and signing work over the rationals instead, embedded in `C`, and
that is a different transform with a precision requirement of its own. This
module is that transform: the evaluation map, its inverse, and the `split` /
`merge` pair the `ffLDL` tree recurses through.

## Double precision is the requirement, and byte-exactness is not

Falcon's security analysis assumes double precision. `ffSampling` is where that
is load-bearing: too little precision moves the sampled distribution away from
the ideal one, which is what leaks the secret basis.

On the host that costs nothing — numpy is `complex128` natively. Traced, it
does: this stack runs `jax_enable_x64 = False`, so a `float64` is silently
narrowed to `float32`, 24 bits of mantissa against 53. A traced caller therefore
runs inside [`double_precision`](#double_precision), and every entry point
refuses rather than narrowing quietly.

Reproducing the reference implementation's exact output bytes is **not** a
requirement. Signing is randomized by design — §3.9 draws the salt per
signature — so two correct implementations disagree on output by construction
and there is no single signature to be exact about. What must hold is that a
signature verifies, interoperates, and comes from a correct distribution; the
first two are gated against the reference implementation as an oracle, and the
third is what the precision above buys.

That decision is what lets this module take the compiler as it is. Under `jit`,
XLA contracts a multiply-add into a fused `fma`, which rounds once where the
reference rounds twice; the result is *more* accurate and differs in the last
place. The only lever is global to the whole stack, so a scheme's test
methodology is the wrong reason to reach for it.

## The transform is a pre-twist and a DFT, not a hand-rolled recursion

`f` evaluated at the `n` roots of `x^n = −1` is

    f(z^(2k+1)) = Σ_j (f_j · z^j) · ω^(jk),   z = e^(iπ/n),  ω = z² = e^(2iπ/n)

so multiplying the coefficients by `z^j` turns the negacyclic transform into an
ordinary length-`n` DFT, which both namespaces already carry. The `+i`
convention is `ifft` scaled by `n` rather than `fft`, because the library's
forward transform is the `−i` one — a sign this module gets wrong silently, so
[`fft_test`](testing/fft_test.py) checks it against direct polynomial
evaluation rather than against a round trip.

**The twiddles are host tables, and that is not an optimization detail.** `n`
arrives as a Python `int`, so the table is a host value by the rule that a value
is used in the namespace it arrives in — and building it with the traced
namespace instead is not folded away: the optimized HLO for `split` keeps
`iota`, `exponential`, `sine` and `cosine`, where a plain `fnp.fft.ifft` keeps
none of them. It is recomputed per execution, not per trace. The `ffLDL`
recursion walks `~n` nodes across `log₂n` distinct sizes, so the same handful of
tables would be rebuilt once per node per signature.

`roots(n)[:n/2]` is bit-identically `_twist(n)[1::2]`, since `z^(2k+1)` for
`k < n/2` is `_twist`'s own entry at `2k+1`. So `split` and `merge` need no
transcendental of their own.

## `split` pairs a root with its negative, and a round trip cannot check that

`f(x) = f0(x²) + x·f1(x²)` evaluated at `r` and at `−r` gives

    f0(r²) = (f(r) + f(−r)) / 2        f1(r²) = (f(r) − f(−r)) / 2r

so the pair is `r` and `−r`, which in this ordering is index `i` and `i + n/2`.
The reference implementation pairs adjacent indices instead — its FFT
representation is in bit-reversed order, which its own comment notes "changes
indexes with regards to the Falcon specification". Either is a correct split of
its own representation and they are not interchangeable.

**`merge(split(f))` holds for both, which is exactly why it is not the test.**
It undoes whatever `split` did. `fft_test` checks the halves against `f0` and
`f1` transformed independently, which is the only thing that distinguishes them.

## What this module does not decide

It stays in the coefficient-and-evaluation domain and knows nothing about
`ffLDL`, the Gram matrix, or the sampler. Those recurse *through* `split` and
`merge` and belong with the scheme
([#26](https://github.com/fractalyze/sig-frx/issues/26),
[#27](https://github.com/fractalyze/sig-frx/issues/27)).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

import numpy as np
from frx import enable_x64
from frx.typing import ArrayLike

from sig_frx.arrays import namespace


@contextlib.contextmanager
def double_precision() -> Iterator[None]:
    """The scope a **traced** caller has to run inside. A host caller needs none.

    Wrap the whole operation, not each call. An array survives the scope with
    its dtype and value intact, and the first operation *outside* it narrows the
    result — so a function that opened a scope of its own and returned would
    hand back a `float64` that dies on the caller's next line, which reads as a
    precision bug in the caller.

    Two dtypes change meaning inside, and neither belongs to this module:

    - an integer lane widens, so `uint32.sum()` returns `uint64` where the
      repo's first non-negotiable says 32 bits. Nothing here holds integers;
      a caller that does keeps them outside, or pins the accumulator dtype.
    - a `prime_field` is immune, because it reduces internally.
    """
    with enable_x64(True):
        yield


def require_scope(*values: object) -> Any:
    """The namespace `values` belong to, refusing a traced call outside the scope.

    Traced and outside it, `asarray(..., dtype='complex128')` *warns* and
    returns `complex64`, and 24 bits of mantissa is not a tolerance this
    transform can absorb — it is the difference the security analysis rests on.
    A warning is the wrong shape for that, so it is an error.

    Public because the rational half of Falcon is not only this module:
    [`keygen`](keygen.py)'s `ffLDL` tree is `float64` arithmetic over the same
    domain with the same requirement, and a second copy of this check would be a
    second chance to word the refusal differently. The scope it names lives here,
    so the guard for it does too.
    """
    xnp = namespace(*values)
    if xnp is not np and not enable_x64.value:
        raise RuntimeError(
            "a traced Falcon rational-domain operation requires double "
            "precision, which is off by default in this stack; wrap it in "
            "`sig_frx.lattice.falcon.fft.double_precision()`"
        )
    return xnp


def _frozen(table: np.ndarray) -> np.ndarray:
    """A cached table is shared, so hand out something a caller cannot edit."""
    table.flags.writeable = False
    return table


@lru_cache(maxsize=None)
def _twist(n: int) -> np.ndarray:
    """`z^j` for `j < n`, with `z = e^(iπ/n)` — the pre-twist, on the host.

    One multiplication per entry rather than an accumulation, so the largest
    angle carries no accumulated error.
    """
    return _frozen(np.exp(1j * (np.pi / n) * np.arange(n)))


@lru_cache(maxsize=None)
def _half_roots(n: int) -> np.ndarray:
    """The first `n/2` roots of `x^n = −1`, which is `_twist(n)` at odd indices."""
    return _frozen(_twist(n)[1::2].copy())


@lru_cache(maxsize=None)
def _split_factor(n: int) -> np.ndarray:
    """`½·conj(r)` for the first `n/2` roots — `split`'s whole per-lane constant.

    Halving is exact at a power of two, so folding it into the table changes no
    bit and saves a pass over the array.
    """
    return _frozen(0.5 * np.conj(_half_roots(n)))


def fft(f: ArrayLike) -> Any:
    """Coefficients to evaluations at the `n` roots of `x^n = −1`.

    `ifft` rather than `fft`, scaled — the library's forward transform carries
    the `−i` convention and this ring wants `+i`.
    """
    xnp = require_scope(f)
    n = np.shape(f)[-1]
    return xnp.fft.ifft(xnp.asarray(f, dtype="complex128") * _twist(n)) * n


def ifft(f_fft: ArrayLike) -> Any:
    """Evaluations back to coefficients — the inverse of [`fft`](#fft)."""
    xnp = require_scope(f_fft)
    n = np.shape(f_fft)[-1]
    return xnp.fft.fft(xnp.asarray(f_fft, dtype="complex128")) / n * np.conj(_twist(n))


def split(f_fft: ArrayLike) -> tuple[Any, Any]:
    """`f_fft` to the transforms of `f0` and `f1`, where `f = f0(x²) + x·f1(x²)`.

    The pair is a root and its negative, which is `i` and `i + n/2` here.
    """
    require_scope(f_fft)
    half = np.shape(f_fft)[-1] // 2
    lo, hi = f_fft[..., :half], f_fft[..., half:]
    return 0.5 * (lo + hi), (lo - hi) * _split_factor(2 * half)


def merge(f0_fft: ArrayLike, f1_fft: ArrayLike) -> Any:
    """The inverse of [`split`](#split): two half transforms back to one."""
    xnp = require_scope(f0_fft, f1_fft)
    half = np.shape(f0_fft)[-1]
    t = f1_fft * _half_roots(2 * half)
    return xnp.concatenate([f0_fft + t, f0_fft - t], axis=-1)
