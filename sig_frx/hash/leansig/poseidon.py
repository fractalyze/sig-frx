# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""LeanSig's two Poseidon permutations, over KoalaBear at widths 16 and 24.

The permutation itself is hash-frx's `Poseidon` — classic Hades, every round
`ARC -> S-box -> dense MDS`, split `rounds_f/2` full, `rounds_p` partial,
`rounds_f/2` full. What this module owns is the parameterization: leanSig's
constants ([`poseidon_constants.py`](poseidon_constants.py)) and the one
convention mismatch between them and the engine that runs them.

## The lane, and why the state arrives reversed

A partial round applies its S-box to a single lane, and *which* lane is a
convention rather than a design. hash-frx follows ark-sponge 0.3 and uses the
**last** lane; leanSpec uses **lane 0**, as HorizenLabs, circomlib and Plonky3
do. hash-frx's `PoseidonParams` states the consumer's side of this: a parameter
set written for lane 0 is run by conjugating the MDS and the round constants
rather than passing them through.

That conjugation is exact, not an approximation. Write `R` for reversing the
lane order. Reversal moves lane 0 to the last lane, so `R . sbox_0(x) =
sbox_last(R . x)`, and one round therefore satisfies

    R . (M . sbox_0(x + c)) = (R M R) . sbox_last(R . x + R . c)

Every round is that same identity, so with `M' = R M R` and `c' = R . c` per
round, hash-frx's last-lane permutation on `R . x` is `R .` leanSig's
permutation on `x`:

    leanSig_permute(x) = R . hashfrx_permute_{M', c'}(R . x)

`M` is circulant with `M[i][j] = r[(j - i) mod w]`, so `M'` is simply the other
circulant, `M'[i][j] = r[(i - j) mod w]` — the conjugation costs one index flip
on a constant, not a runtime operation.

**So the permutation this module hands out runs on a lane-reversed state**, and
its name says so. That is deliberate: reversing at every call would put a device
`reverse` either side of ~180 permutations per verification, while a caller that
*builds* its state — the compression and sponge layers, which place a public
parameter, a tweak and a message into fixed positions and read a truncated
prefix back — reverses for free by placing and slicing from the other end. The
reversal is a layout decision made once at the boundary, never data movement.

"Once" is load-bearing on the callers too. This seam states the convention in
one place, and it stays one place only if the layers that build state share a
placement helper rather than each working out which end to fill from; a second
module that re-derives it has turned a boundary into a convention spread across
the package.

Nothing else about the two widths differs, so both come from one builder: width
16 is the chain hash, width 24 the message, tree and leaf hashes.

Two pointers a later reader will want. Conjugating here rather than teaching
hash-frx the lane is the settled choice, recorded on
[#195](https://github.com/fractalyze/sig-frx/issues/195); hash-frx's own
`PoseidonParams` leaves the lane-as-a-parameter surface to its redesign epic, and
if that surface ever lands this module unwinds to passing the constants through.
And the permutation does not lower at these widths yet —
[hash-frx#293](https://github.com/fractalyze/hash-frx/issues/293), which is a
limit of the marker's MDS attribute above width 7 rather than anything about the
conjugation.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final

import frx.numpy as fnp
import numpy as np
from frx import Array
from hash_frx import Poseidon, PoseidonParams
from zk_dtypes import koalabear_mont as _F

from sig_frx.hash.leansig import poseidon_constants as _c

_WIDTHS: Final = {
    16: (_c.WIDTH_16_ROUNDS, _c.MDS_FIRST_ROW_16, _c.ROUND_CONSTANTS_16),
    24: (_c.WIDTH_24_ROUNDS, _c.MDS_FIRST_ROW_24, _c.ROUND_CONSTANTS_24),
}


def _to_field(canonical: np.ndarray) -> Array:
    """Canonical ints -> field array. The dtype cast Montgomery-encodes."""
    return fnp.asarray(canonical.astype(np.int64).astype(_F))


def _params(width: int) -> PoseidonParams:
    """LeanSig's parameters at `width`, conjugated onto the last-lane engine."""
    (rounds_f, rounds_p), first_row, constants = _WIDTHS[width]

    lanes = np.arange(width)
    # Upstream's MDS is circulant, `M[i][j] = r[(j - i) mod w]`; the conjugate
    # `R M R` is the same first row read the other way round.
    mds = np.asarray(first_row, dtype=np.int64)[(lanes[:, None] - lanes) % width]
    # One ARC vector per round, full and partial alike, each lane-reversed.
    round_constants = np.asarray(constants, dtype=np.int64).reshape(
        rounds_f + rounds_p, width
    )[:, ::-1]

    return PoseidonParams(
        width=width,
        dtype=_F,
        alpha=_c.ALPHA,
        full_rounds=rounds_f,
        partial_rounds=rounds_p,
        round_constants=_to_field(round_constants),
        mds=_to_field(mds),
    )


@lru_cache(maxsize=None)
def lane_reversed_permutation(width: int) -> Poseidon:
    """LeanSig's Poseidon at `width` — **over a lane-reversed state**.

    `permute(R . x)` returns `R .` what leanSpec's permutation returns for `x`,
    for the reason the module docstring gives. A caller that builds its own
    state places its operands reversed and reads its output from the other end;
    a caller that holds a leanSpec-oriented state reverses either side.

    Cached rather than rebuilt, and built on first use rather than at import.
    Both halves matter: a `Poseidon` rides pytree aux, so a fresh instance per
    call re-traces the enclosing jit zone instead of erroring, and constructing
    one reads `frx.default_backend()` to choose its marker, so building at
    import would freeze the routing to whichever backend happened to be default
    when this module loaded.
    """
    if width not in _WIDTHS:
        raise ValueError(f"leanSig hashes at widths {sorted(_WIDTHS)}, not {width}")
    return Poseidon(_params(width))
