# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""LeanSig's tweakable hash family — three shapes over two Poseidon permutations.

Upstream states all three as one function, `PoseidonXmss.tweak_hash`, which picks
its mode by how many digests it is handed:

| digests | mode | operands, in upstream's order |
|---|---|---|
| 1 | width-16 compression | `digest ‖ parameter ‖ tweak` |
| 2 | width-24 compression | `parameter ‖ tweak ‖ left ‖ right` |
| `DIMENSION` | width-24 sponge | `parameter ‖ tweak ‖ ends…`, capacity = separator |

They are three methods here rather than one dispatch, because the callers are
three and each knows which it wants: a Winternitz chain step is `f`, a Merkle
interior node is `h`, and a Merkle leaf is `leaf`. Splitting them is also what
lets the first two *be* [`tweakable.ChainHash`](../tweakable.py) and `NodeHash`,
so [`wots.chain`](../wots.py) walks leanSig's chains and [`tree.py`](../tree.py)
climbs its tree without either learning what a leanSig digest is made of. That
sharing is why those protocols carry a `dtype` at all.

The operand orders above are not interchangeable and nothing here derives them —
they are transcribed from `spec/crypto/xmss/poseidon.py` at the pinned commit,
and a chain step that hashed `parameter ‖ tweak ‖ digest` would be a perfectly
self-consistent wrong scheme.

## What is not shared, and why the leaf is not `T_l`

FIPS 205's `T_l` compresses `l` blocks and SLH-DSA reaches it for a WOTS+ public
key and a FORS root; leanSig's leaf looks like that and is a different
construction — a sponge, with a capacity that binds it to this hashing task's
shape. So `leaf` is this family's own method and not the protocol's `t`, which
is also why this class implements `ChainHash` and `NodeHash` but not
`TweakableHash`: it has no `prf`, no `prf_msg` and no `h_msg`, and claiming the
wider protocol would be the lie [`tweakable.py`](../tweakable.py)'s own docstring
warns about.

## The tweak, and where the host ends

A tweak is a position packed into one integer and decomposed base-p — a level and
an index for a tree hash, an epoch, a chain and a step for a chain hash — with
the prefix byte that keeps the three families apart ([`params.py`](params.py)).
Packing reaches `2^45` for a tree tweak and `2^56` for a chain one, so it is host
integer arithmetic and cannot be otherwise: an array lane is 32 bits
([`../../../CLAUDE.md`](../../../CLAUDE.md)), and `index << 8` alone leaves one.

That makes the two builders below **host-only**, exactly as
[`encoding.py`](encoding.py)'s `encode_epoch` is, and for the same reason. It is
not a batch-parallelism compromise: they are `np.int64` column arithmetic over
the whole batch, so a verifier's `B` positions cost a handful of numpy operations
rather than `B` of anything. What they refuse is a *traced* index, which
`np.asarray` raises on rather than silently pulling to the host — and a leanSig
slot is a verifier input, so whether it can arrive traced at all is the seam
question this issue tracks separately.

## The batch axis

`poseidon.compress` and `poseidon.sponge` are one-dimensional, because
`Poseidon.permute` is: hash-frx takes a state of shape `(width,)` and nothing
wider. So the batch is `frx.vmap` over the mode, which is the one construction
that cannot disagree with the gated scalar path — it is the same function, and
[`mode_test`](testing/mode_test.py)'s 34 cases stay the gate. A Python loop over
the batch axis would be the bug `../../../CLAUDE.md` names; this is not one.

The vmapped callables are memoized for the reason `poseidon._absorb_step` is:
built per call they would re-trace the same graph on every eager call, which is
silent and orders of magnitude over the work.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import lru_cache

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike
from zk_dtypes import koalabear_mont as F

from sig_frx.hash import tree
from sig_frx.hash.leansig import poseidon
from sig_frx.hash.leansig.field import lane_reversed_limbs
from sig_frx.hash.leansig.params import (
    TWEAK_PREFIX_CHAIN,
    TWEAK_PREFIX_TREE,
    LeanSigParams,
)

_CHAIN_WIDTH = 16
"""The permutation a chain step runs on — one digest and the tweak fit a narrow
state, and upstream spends the wide one only where two digests have to."""

_TREE_WIDTH = 24
"""The permutation both tree hashes run on, interior node and leaf alike."""

# Where each field sits in a packed tweak, transcribed from upstream's
# `tweak_hash`. The prefix owns the low byte in both, which is what stops a tree
# tweak and a chain tweak at the same position from packing to the same integer.
_TREE_LEVEL_SHIFT = 40
_TREE_INDEX_SHIFT = 8
_CHAIN_EPOCH_SHIFT = 24
_CHAIN_INDEX_SHIFT = 16
_CHAIN_STEP_SHIFT = 8

# How wide each field may be, derived from the layout rather than restated. A
# field packed *under* another gets the gap to it; the top field of each tweak
# has nothing above it to collide with, so what bounds it is the signed host
# width the shift is performed in.
_HOST_BITS = 63
_TREE_LEVEL_BITS = _HOST_BITS - _TREE_LEVEL_SHIFT
_TREE_INDEX_BITS = _TREE_LEVEL_SHIFT - _TREE_INDEX_SHIFT
_CHAIN_EPOCH_BITS = _HOST_BITS - _CHAIN_EPOCH_SHIFT
_CHAIN_INDEX_BITS = _CHAIN_EPOCH_SHIFT - _CHAIN_INDEX_SHIFT
_CHAIN_STEP_BITS = _CHAIN_INDEX_SHIFT - _CHAIN_STEP_SHIFT


def _column(values: ArrayLike, name: str, bits: int) -> np.ndarray:
    """A tweak field as a host `int64` column, bounded by the room it packs into.

    The bound is the whole point, and it is two different hazards wearing one
    check. A field packed under another does not overflow when it runs past its
    range — it *carries* into the field above and packs a valid tweak for a
    different position, which is a wrong digest with nothing to notice it. The
    top field has no neighbour to corrupt, but its shift happens in `int64`, and
    a shift past that wraps silently. Neither is caught by the limb-fit check in
    [`field.py`](field.py), which only ever sees the sum.
    """
    column = np.asarray(values, dtype=np.int64).reshape(-1)
    outside = (column < 0) | (column >= 1 << bits)
    if np.any(outside):
        raise ValueError(
            f"{name} takes {bits} bits in a packed tweak, so it must be in "
            f"[0, {1 << bits}); {int(np.count_nonzero(outside))} of "
            f"{column.size} entries are not, the first being "
            f"{int(column[np.argmax(outside)])}"
        )
    return column


def tree_tweaks(level: int, indices: ArrayLike, *, params: LeanSigParams) -> Array:
    """`TreeTweak(level, index)` for a whole level: -> `[B, tweak_length]`.

    `level` is shared — a Merkle walk hashes one level at a time — and `indices`
    is per entry. Host-only, per the module docstring.
    """
    packed = (
        (_column(level, "level", _TREE_LEVEL_BITS) << _TREE_LEVEL_SHIFT)
        | (_column(indices, "index", _TREE_INDEX_BITS) << _TREE_INDEX_SHIFT)
        | TWEAK_PREFIX_TREE
    )
    return lane_reversed_limbs(packed, params.tweak_length)


def chain_tweaks(
    epochs: ArrayLike,
    chain_indices: ArrayLike,
    step: int,
    *,
    params: LeanSigParams,
) -> Array:
    """`ChainTweak(epoch, chain_index, step)` for one step: -> `[B, tweak_length]`.

    `step` is shared and the other two are per entry, which is the shape
    [`wots.chain`](../wots.py) walks in: it runs every chain the full `base - 1`
    applications and masks, so at its `j`-th iteration every active entry is
    making its `(j - start)`-th application at absolute step `j + 1` — the same
    number for all of them, whatever each chain's own starting digit was.

    Upstream counts steps from one: step zero is the chain start, which is never
    hashed. Host-only, per the module docstring.
    """
    epoch_column = _column(epochs, "epoch", _CHAIN_EPOCH_BITS)
    chain_column = _column(chain_indices, "chain_index", _CHAIN_INDEX_BITS)
    packed = (
        (epoch_column << _CHAIN_EPOCH_SHIFT)
        | (chain_column << _CHAIN_INDEX_SHIFT)
        | (_column(step, "step", _CHAIN_STEP_BITS) << _CHAIN_STEP_SHIFT)
        | TWEAK_PREFIX_CHAIN
    )
    return lane_reversed_limbs(packed, params.tweak_length)


def node_tweaks(params: LeanSigParams) -> tree.NodeAddresses:
    """The builder [`tree.py`](../tree.py) tweaks each Merkle level with.

    The same seam `tree.xmss_node_addresses` fills for FIPS 205: the walk takes a
    builder and never learns what one encodes, which is what lets one tree climb
    over byte addresses and over field elements.
    """

    def build(height: int, indices: ArrayLike) -> Array:
        return tree_tweaks(height, indices, params=params)

    return build


def chain_step_tweaks(
    epochs: ArrayLike, chain_indices: ArrayLike, *, params: LeanSigParams
) -> Iterator[Array]:
    """Every step's tweaks for a full walk, in [`wots.chain`](../wots.py)'s order.

    `base - 1` batches, the walk's own length, and the `+ 1` is upstream's
    one-indexing kept in one place: `chain`'s `j`-th iteration is absolute step
    `j + 1`, and a caller that spelled that itself would be one rename away from
    hashing every step at the wrong position.

    A generator because `chain` reads its addresses once and forward, so nothing
    materialises every step at once.
    """
    for step in range(1, params.base):
        yield chain_tweaks(epochs, chain_indices, step, params=params)


@lru_cache(maxsize=None)
def _compression(width: int, output_length: int) -> Callable[..., Array]:
    """`compress` over a leading batch axis, one memoized callable per shape."""

    def one(*operands: Array) -> Array:
        return poseidon.compress(operands, width=width, output_length=output_length)

    return frx.vmap(one)


@lru_cache(maxsize=None)
def _leaf_sponge(dimension: int, output_length: int) -> Callable[..., Array]:
    """The leaf's sponge over a leading batch axis, with a shared capacity.

    The capacity rides `in_axes=None` because it is one value for the whole
    batch — it depends only on the configuration — so the separator is computed
    once per call rather than per leaf.
    """

    def one(parameter: Array, tweak: Array, ends: Array, capacity: Array) -> Array:
        # The chain ends are `dimension` operands the spec names, not one
        # vector. The lane reversal reverses the order of the pieces as well as
        # each piece, so a flattened `[dimension, n]` is a different pre-image —
        # and one that hashes, self-checks and disagrees only with upstream.
        return poseidon.sponge(
            (parameter, tweak, *(ends[index] for index in range(dimension))),
            capacity,
            width=_TREE_WIDTH,
            output_length=output_length,
        )

    return frx.vmap(one, in_axes=(0, 0, 0, None))


def _rows(value: ArrayLike, batch: int) -> Array:
    """Broadcast a shared `[k]` operand to `[B, k]`, or pass a `[B, k]` through.

    The public parameter is the operand that wants this: one key's own tree
    shares it across every position, and a batch spanning several public keys
    carries one per entry. `tweakable._batched` is the same function at `uint8`;
    this one is at the family's dtype, and merging them would put a dtype
    argument on a helper whose whole job is that its caller already knows one.
    """
    array = fnp.asarray(value, dtype=F)
    if array.ndim == 1:
        return fnp.broadcast_to(array, (batch, array.shape[0]))
    return array


class LeanSigTweakableHash:
    """LeanSig's family at one preset — `PoseidonXmss.tweak_hash`, split by mode.

    Digests are lane-reversed field vectors throughout, the convention
    [`poseidon.py`](poseidon.py) states and justifies: a digest comes back
    reversed and feeds straight back in, so a chain walk and a Merkle climb move
    no data between the ~180 permutations a verification runs.
    """

    def __init__(self, params: LeanSigParams) -> None:
        self._params = params
        # The protocol's two fields. `n` is a digest in *elements* here rather
        # than in bytes, which is what `dtype` alongside it says.
        self.n = params.hash_length
        self.dtype = F

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LeanSigTweakableHash):
            return NotImplemented
        return self._params == other._params

    def __hash__(self) -> int:
        return hash((type(self), self._params))

    # -- the tweaked family ------------------------------------------------

    def f(self, parameter: ArrayLike, tweak: ArrayLike, digest: ArrayLike) -> Array:
        """A Winternitz chain step — `compress((digest, parameter, tweak), 16)`.

        `[B, n]` digests under `[B, tweak_length]` tweaks: -> `[B, n]`.
        """
        tweaks = fnp.asarray(tweak, dtype=F)
        batch = tweaks.shape[0]
        return _compression(_CHAIN_WIDTH, self.n)(
            _rows(digest, batch), _rows(parameter, batch), tweaks
        )

    def h(self, parameter: ArrayLike, tweak: ArrayLike, pair: ArrayLike) -> Array:
        """A Merkle interior node — `compress((parameter, tweak, left, right), 24)`.

        `pair` is `[B, 2n]`, the concatenation `tree.py` hands every family. The
        halves go in as two operands rather than as that vector, because the
        reversal turns `left ‖ right` into `R(right) ‖ R(left)` — splitting here
        is what lets the caller concatenate the way every other family wants.
        """
        tweaks = fnp.asarray(tweak, dtype=F)
        batch = tweaks.shape[0]
        pairs = fnp.asarray(pair, dtype=F)
        if pairs.shape[-1] != 2 * self.n:
            raise ValueError(
                f"a Merkle pair is {2 * self.n} field elements, got {pairs.shape[-1]}"
            )
        return _compression(_TREE_WIDTH, self.n)(
            _rows(parameter, batch),
            tweaks,
            pairs[:, : self.n],
            pairs[:, self.n :],
        )

    def leaf(
        self, parameter: ArrayLike, tweak: ArrayLike, chain_ends: ArrayLike
    ) -> Array:
        """A Merkle leaf — the sponge over one slot's `dimension` chain ends.

        `chain_ends` is `[B, dimension, n]`: -> `[B, n]`. A compression cannot do
        this — `dimension · n` elements is 368 at `PROD` against a 24-lane state —
        which is the whole reason upstream reaches for a sponge here and nowhere
        else in the scheme.
        """
        tweaks = fnp.asarray(tweak, dtype=F)
        ends = fnp.asarray(chain_ends, dtype=F)
        dimension = self._params.dimension
        if ends.shape[-2:] != (dimension, self.n):
            raise ValueError(
                f"a leaf hashes {dimension} digests of {self.n} elements, "
                f"got shape {ends.shape}"
            )
        return _leaf_sponge(dimension, self.n)(
            _rows(parameter, tweaks.shape[0]), tweaks, ends, self.capacity_value
        )

    @property
    def capacity_value(self) -> Array:
        """The leaf sponge's capacity — `safe_domain_separator` at this preset.

        Upstream's own `lengths`, in its order. A property rather than a cached
        attribute for the reason `poseidon.safe_domain_separator` gives for not
        caching itself: it would be a concrete device array carrying backend
        affinity, and the cost it saves is one permutation against the ~180 a
        verification runs.
        """
        return poseidon.safe_domain_separator(
            [
                self._params.parameter_length,
                self._params.tweak_length,
                self._params.dimension,
                self._params.hash_length,
            ],
            capacity_length=self._params.capacity,
        )
