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

"Once" is load-bearing on the callers too, and it is why the two modes below
live here rather than in a module of their own. The convention is two facts —
`_reversed_lanes`, where a leanSpec lane range sits in a reversed vector, and
`_join`, that `a ‖ b` reverses to `R(b) ‖ R(a)` — and every device-side
placement here goes through one of them, `_padded` included, which is `_join`
with upstream's own trailing zeros rather than a third rule. A second module
that re-derived either has turned a boundary into a convention spread across the
package. (Reversing a *host* list, as `safe_domain_separator` does to its limbs,
is not that: it is a Python slice on values that have not reached a lane yet.)

Nothing else about the two widths differs, so both come from one builder: width
16 is the chain hash, width 24 the message, tree and leaf hashes.

## The two modes over it

- **Compression** — `Truncate(Permute(padded) + padded)`. The feed-forward
  addition is part of the Hades design; the chain step, the Merkle interior node
  and the message hash all reach the permutation this way.
- **Sponge** — capacity lanes first, then rate lanes, absorbing by overwriting
  the rate. The Merkle *leaf* needs it: it hashes `DIMENSION` digests at once and
  no state is wide enough to compress them.

Neither is hash-frx's. Its `Compression` has no feed-forward term and takes a
fixed `(arity, chunk)` grid rather than a flat operand list; its `Sponge` puts
capacity *after* the rate, starts it at zero rather than at a domain separator,
and squeezes from the other side. Each differs in construction rather than in
parameterization, so this is a reimplementation and not a seam hash-frx should
grow — the same "is this one scheme's?" answer that sited the constants.

Both modes take the operands the spec names — the digest, the public parameter,
the tweak — rather than one pre-concatenated vector, so `_join` is what reverses
the piece order and no caller decides which end to fill from.

Two pointers a later reader will want. Conjugating here rather than teaching
hash-frx the lane is the settled choice, recorded on
[#195](https://github.com/fractalyze/sig-frx/issues/195); hash-frx's own
`PoseidonParams` leaves the lane-as-a-parameter surface to its redesign epic, and
if that surface ever lands this module unwinds to passing the constants through.
And the permutation lowers to one kernel at both widths, but on hash-frx's
**generic** fused-region marker rather than the dedicated classic-Poseidon
emitter: that emitter applies the MDS as a small-integer add-chain, so it takes
entries in `[0, 64)` and no matrix over a 31-bit field qualifies. Correct, and
at ~180 permutations per verification the gap is worth closing —
[xla#604](https://github.com/fractalyze/xla/issues/604). Nothing about it is the
conjugation's doing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Final

import frx
import frx.numpy as fnp
import numpy as np
import zk_dtypes
from frx import Array
from hash_frx import Poseidon, PoseidonParams
from zk_dtypes import koalabear_mont as _F

from sig_frx.hash.leansig import poseidon_constants as _c

_WIDTHS: Final = {
    16: (_c.WIDTH_16_ROUNDS, _c.MDS_FIRST_ROW_16, _c.ROUND_CONSTANTS_16),
    24: (_c.WIDTH_24_ROUNDS, _c.MDS_FIRST_ROW_24, _c.ROUND_CONSTANTS_24),
}

# Off the dtype's own metadata rather than restated, so the pinned wheel stays
# the single source of truth — the reason `classical/secp.py` derives its moduli
# the same way. leanSpec states the same value in `spec/crypto/koalabear.py`.
_PRIME: Final = zk_dtypes.pfinfo(_F).modulus

# Upstream fixes the domain separator at width 24 rather than at the sponge's
# own width — the sponge is the only construction that needs one, and it runs
# there.
_DOMAIN_SEPARATOR_WIDTH: Final = 24


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


def _reversed_lanes(start: int, length: int, size: int) -> slice:
    """Where leanSpec lanes `[start, start + length)` sit in a reversed vector.

    Reversal maps lane `i` of a `size`-long vector to `size - 1 - i`, so a range
    that starts `start` from the front starts `start` from the back. Every read
    and write in this module is expressed through this rather than through an
    index worked out at the call site.
    """
    stop = size - start
    return slice(stop - length, stop)


def _join(pieces: Sequence[Array]) -> Array:
    """Concatenate lane-reversed `pieces` listed in leanSpec's order.

    `R(a ‖ b) = R(b) ‖ R(a)`: reversing a concatenation reverses the order of
    its parts as well as each part. Callers name their operands the way the spec
    writes them and this puts them where the reversed state wants them.
    """
    return fnp.concatenate(list(pieces)[::-1])


def _padded(pieces: Sequence[Array], size: int) -> Array:
    """`pieces` joined and zero-extended to `size`, upstream's own padding.

    Every mode here zero-extends something — a compression pre-image to the
    state width, a sponge input to a whole number of chunks, a capacity to the
    full state — and upstream always pads at the *end*, so `_join` is what puts
    the zeros at the front. Routing all three through this is what keeps that a
    single decision rather than an idiom re-spelled per site.
    """
    return _join([*pieces, fnp.zeros(size - _length(pieces), dtype=pieces[0].dtype)])


def _length(pieces: Sequence[Array]) -> int:
    """Elements across `pieces`, which is what the modes bound their inputs by."""
    return sum(piece.shape[0] for piece in pieces)


def _int_to_base_p(value: int, num_limbs: int) -> list[int]:
    """`value` as `num_limbs` base-p limbs, least significant first.

    Host-only, and the packing is why: `safe_domain_separator` shifts its
    lengths into 32-bit slots, so the value it decomposes is wider than any lane
    and only a Python integer holds it without truncating (`CLAUDE.md`). Each
    limb it returns is below `p`, which is what may then cross onto the device.

    A short decomposition is rejected rather than truncated — dropping the high
    part would silently change the hash, which is upstream's reasoning too.
    """
    limbs = []
    remaining = value
    for _ in range(num_limbs):
        limbs.append(remaining % _PRIME)
        remaining //= _PRIME
    if remaining:
        raise ValueError(f"value does not fit in {num_limbs} base-p limbs")
    return limbs


@lru_cache(maxsize=None)
def _absorb_step(
    width: int, capacity_length: int
) -> Callable[[Array, Array], tuple[Array, None]]:
    """One absorb-and-permute over a lane-reversed state, as a *stable* callable.

    Absorbing overwrites the rate lanes and leaves capacity alone, and the
    permutation that follows belongs to the construction rather than to the
    caller.

    Memoized because `sponge` scans this from eager code as well as from inside
    a trace: `frx.lax.scan`'s cache is keyed on the body's identity, so a body
    built per call recompiles the same graph every time — orders of magnitude
    over the work being scanned, and silent. Keyed on everything it closes over.
    """
    permutation = lane_reversed_permutation(width)
    capacity_lanes = _reversed_lanes(0, capacity_length, width)

    def step(state: Array, chunk: Array) -> tuple[Array, None]:
        return permutation.permute(_join([state[capacity_lanes], chunk])), None

    return step


def compress(operands: Sequence[Array], *, width: int, output_length: int) -> Array:
    """Poseidon in compression mode: `Truncate(Permute(padded) + padded)`.

    `operands` are the pieces the spec names, each lane-reversed, listed in the
    spec's own order — for a chain step, `(digest, parameter, tweak)`. They are
    zero-extended to `width`, permuted, added back, and the leading
    `output_length` elements are the digest.

    Returns a lane-reversed digest, so a caller that feeds it straight back in —
    which is what a chain step and a Merkle walk do — never reverses anything.
    """
    if not operands:
        raise ValueError("compress needs at least one operand")

    length = _length(operands)
    if length > width:
        raise ValueError(f"{length} operand elements do not fit a width-{width} state")
    # Upstream's own bound, and it is on the unpadded length: truncating to more
    # than was fed in would return padding as digest.
    if output_length > length:
        raise ValueError(
            f"output_length {output_length} exceeds the {length} elements fed in"
        )

    padded = _padded(operands, width)
    state = lane_reversed_permutation(width).permute(padded) + padded
    return state[_reversed_lanes(0, output_length, width)]


def sponge(
    operands: Sequence[Array],
    capacity: Array,
    *,
    width: int,
    output_length: int,
) -> Array:
    """Poseidon in sponge mode, over a lane-reversed state.

    `capacity` is the domain separator (`safe_domain_separator`) and sits in the
    leading lanes upstream, so the reversed state carries it in the tail and the
    rate leads. `operands` follow the same convention `compress` states.

    The squeeze loops because the construction does, not because leanSig asks
    it to: every call the scheme makes wants `HASH_LENGTH_FIELD_ELEMENTS` = 8
    out of a rate of 15. A `sponge` that quietly handled only that would be a
    trap for the next caller, and the loop is gated rather than assumed.
    """
    if not operands:
        raise ValueError("sponge needs at least one operand")

    capacity_length = capacity.shape[0]
    if capacity_length >= width:
        raise ValueError(
            f"a capacity of {capacity_length} leaves no rate lane at width {width}"
        )

    rate = width - capacity_length
    length = _length(operands)

    # Padding to a whole number of chunks makes every absorb the same width, so
    # the block loop needs no tail case.
    absorbed_length = length + (-length % rate)
    absorbed = _padded(operands, absorbed_length)

    permutation = lane_reversed_permutation(width)

    # `reverse=True` is what makes the reshape legal: reversal put leanSpec's
    # block 0 in the *last* row, so walking the rows from the back is walking the
    # blocks in spec order — no device `reverse`, no per-block slice.
    state, _ = frx.lax.scan(
        _absorb_step(width, capacity_length),
        _padded([capacity], width),
        absorbed.reshape(absorbed_length // rate, rate),
        reverse=True,
    )

    squeezed: list[Array] = []
    remaining = output_length
    while remaining:
        take = min(remaining, rate)
        squeezed.append(state[_reversed_lanes(capacity_length, take, width)])
        remaining -= take
        if remaining:
            state = permutation.permute(state)
    return _join(squeezed)


def safe_domain_separator(lengths: Sequence[int], *, capacity_length: int) -> Array:
    """A sponge capacity that binds it to one hashing task's shape.

    The lengths pack into 32-bit slots, decompose base-p and compress. Two
    sponges absorbing differently shaped data therefore start from different
    capacities and cannot collide.

    Depends only on the configuration, so a caller hashing many leaves computes
    it once and hoists it. Uncached on purpose, and the reason is the seam rather
    than the cost: `sponge` takes the capacity as an argument and never calls
    this itself, so the layer above physically cannot land it in a per-block
    loop — the most it can waste is one permutation per leaf hash, against the
    ~180 a verification runs. Caching a *concrete array* is also the thing
    `lane_reversed_permutation` avoids by caching a builder instead.
    """
    packed = 0
    for length in lengths:
        packed = (packed << 32) | length

    limbs = _int_to_base_p(packed, _DOMAIN_SEPARATOR_WIDTH)
    return compress(
        [_to_field(np.asarray(limbs[::-1]))],
        width=_DOMAIN_SEPARATOR_WIDTH,
        output_length=capacity_length,
    )
