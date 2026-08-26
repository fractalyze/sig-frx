# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The transform over `Q[x]/(x^n + 1)`, for Falcon's key generation and signing.

Verification lives in `Z_q` and uses the integer NTT ([`arith.py`](arith.py)).
Key generation and signing work over the rationals instead, embedded in `C`, and
that is a different transform with a precision requirement of its own. This
module is that transform: the evaluation map, its inverse, and the `split` /
`merge` pair the `ffLDL` tree recurses through.

## Double precision is the requirement, and byte-exactness is not

Falcon's security analysis assumes double precision. This stack runs
`jax_enable_x64 = False`, so a `float64` is silently narrowed to `float32` — 24
bits of mantissa against 53 — and `ffSampling` is where that is load-bearing:
too little precision moves the sampled distribution away from the ideal one,
which is what leaks the secret basis. So every function here must run inside
[`double_precision`](#double_precision) and says so by refusing otherwise.

Reproducing the reference implementation's exact output bytes is **not** a
requirement, and pursuing it would cost more than it buys. Signing is randomized
by design — §3.9 draws the salt per signature — so two correct implementations
disagree on output by construction and there is no single signature to be exact
about. What must hold is that a signature verifies, interoperates, and comes
from a correct distribution; the first two are gated against the reference
implementation as an oracle, and the third is what the precision above buys.

That decision is what lets this module use the compiler as it is. Under `jit`,
XLA contracts a multiply-add into a fused `fma`, which rounds once where the
reference rounds twice; the result is *more* accurate and differs in the last
place. The only lever is global to the whole stack, so a scheme's test
methodology is the wrong reason to reach for it.

## The transform is a pre-twist and a DFT, not a hand-rolled recursion

`f` evaluated at the `n` roots of `x^n = −1` is

    f(z^(2k+1)) = Σ_j (f_j · z^j) · ω^(jk),   z = e^(iπ/n),  ω = z² = e^(2iπ/n)

so multiplying the coefficients by `z^j` turns the negacyclic transform into an
ordinary length-`n` DFT, which `frx.numpy.fft` already carries. The `+i`
convention is `ifft` scaled by `n` rather than `fft`, because the library's
forward transform is the `−i` one — a sign this module gets wrong silently, so
[`fft_test`](testing/fft_test.py) checks it against direct polynomial
evaluation rather than against a round trip.

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
from typing import Any

import numpy as np
from frx import numpy as fnp

# `enable_x64` is a `bool_state` carrying `include_in_jit_key`, so a scope
# around a traced call compiles its own variant rather than colliding with the
# 32-bit one. frx does not re-export it, and reaching into `_src` is the
# narrower of the two available wrongs: the alternative is a module-level
# `JAX_ENABLE_X64`, which would widen every integer lane in the process and
# invert the rule that a lane here is 32 bits.
from frx._src.config import enable_x64 as _enable_x64


@contextlib.contextmanager
def double_precision() -> Iterator[None]:
    """The scope every function in this module has to run inside.

    Wrap the **whole operation**, not each call. An array survives the scope
    with its dtype and value intact, and the first operation *outside* it
    narrows the result — so a function that opened a scope of its own and
    returned would hand back a `float64` that dies on the caller's next line,
    which reads as a precision bug in the caller.

    Two dtypes change meaning inside, and neither belongs to this module:

    - an integer lane widens, so `uint32.sum()` returns `uint64` where the
      repo's first non-negotiable says 32 bits. Nothing here holds integers;
      a caller that does keeps them outside, or pins the accumulator dtype.
    - a `prime_field` is immune, because it reduces internally.
    """
    with _enable_x64(True):
        yield


def _require_scope() -> None:
    """Refuse rather than narrow silently.

    Outside the scope `fnp.asarray(..., dtype='complex128')` warns and returns
    `complex64`, and 24 bits of mantissa is not a tolerance this transform can
    absorb — it is the difference the security analysis rests on. A warning is
    the wrong shape for that, so it is an error.
    """
    if not _enable_x64.value:
        raise RuntimeError(
            "the Falcon FFT requires double precision, which is off by default "
            "in this stack; wrap the operation in "
            "`sig_frx.lattice.falcon.fft.double_precision()`"
        )


def _twist(n: int) -> Any:
    """`z^j` for `j < n`, with `z = e^(iπ/n)` — the pre-twist and its conjugate.

    Built on the host from an integer range rather than accumulated, so the
    largest angle is one multiplication rather than `n` of them.
    """
    j = fnp.arange(n, dtype="float64")
    return fnp.exp(1j * (np.pi / n) * j)


def roots(n: int) -> Any:
    """The `n` roots of `x^n = −1`, in the order this module evaluates at."""
    _require_scope()
    k = fnp.arange(n, dtype="float64")
    return fnp.exp(1j * (np.pi / n) * (2.0 * k + 1.0))


def fft(f: Any) -> Any:
    """Coefficients to evaluations at the `n` roots of `x^n = −1`.

    `frx.numpy.fft.ifft` rather than `fft`, scaled — the library's forward
    transform carries the `−i` convention and this ring wants `+i`.
    """
    _require_scope()
    n = f.shape[-1]
    return fnp.fft.ifft(fnp.asarray(f, dtype="complex128") * _twist(n)) * n


def ifft(f_fft: Any) -> Any:
    """Evaluations back to coefficients — the inverse of [`fft`](#fft)."""
    _require_scope()
    n = f_fft.shape[-1]
    return fnp.fft.fft(fnp.asarray(f_fft, dtype="complex128")) / n * fnp.conj(_twist(n))


def split(f_fft: Any) -> tuple[Any, Any]:
    """`f_fft` to the transforms of `f0` and `f1`, where `f = f0(x²) + x·f1(x²)`.

    The pair is a root and its negative, which is `i` and `i + n/2` here.
    """
    _require_scope()
    n = f_fft.shape[-1]
    half = n // 2
    lo, hi = f_fft[..., :half], f_fft[..., half:]
    f0 = 0.5 * (lo + hi)
    f1 = 0.5 * (lo - hi) * fnp.conj(roots(n)[:half])
    return f0, f1


def merge(f0_fft: Any, f1_fft: Any) -> Any:
    """The inverse of [`split`](#split): two half transforms back to one."""
    _require_scope()
    half = f0_fft.shape[-1]
    n = 2 * half
    t = f1_fft * roots(n)[:half]
    return fnp.concatenate([f0_fft + t, f0_fft - t], axis=-1)
