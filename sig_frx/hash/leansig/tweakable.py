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

from collections.abc import Callable, Iterator, Sequence
from functools import lru_cache, partial
from typing import TYPE_CHECKING

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike
from zk_dtypes import koalabear_mont as F

from sig_frx.hash import tree
from sig_frx.hash.leansig import poseidon
from sig_frx.hash.leansig.field import host_column, lane_reversed_limbs
from sig_frx.hash.leansig.params import (
    TWEAK_PREFIX_CHAIN,
    TWEAK_PREFIX_TREE,
    LeanSigParams,
)
from sig_frx.hash.tweakable import ChainHash, NodeHash
from sig_frx.hash.tweakable import batched as _batched

_CHAIN_WIDTH = 16
"""The permutation a chain step runs on — one digest and the tweak fit a narrow
state, and upstream spends the wide one only where two digests have to."""

_TREE_WIDTH = 24
"""The permutation both tree hashes run on, interior node and leaf alike."""

# The two packings, transcribed from upstream's `tweak_hash`:
#
#     tree   (level << 40) | (index << 8)        | TWEAK_PREFIX_TREE
#     chain  (epoch << 24) | (chain_index << 16) | (step << 8) | TWEAK_PREFIX_CHAIN
#
# Below they are stated as field *widths*, low to high, because that is what a
# bound has to check and it is the form that cannot disagree with itself: each
# field's shift is the running sum of the ones under it, so the 40 and the 24
# above are reproduced rather than restated. The prefix owns the low byte of
# both, which is the whole of what keeps a chain hash and a tree hash at the
# same position from packing to the same integer.
_PREFIX_BITS = 8

# A packed tweak is built in `int64`, so the top field of each layout is bounded
# by what is left of that rather than by a neighbour it could carry into.
_HOST_BITS = 63

_TREE_FIELDS = (("index", 32), ("level", _HOST_BITS - 8 - 32))
_CHAIN_FIELDS = (("step", 8), ("chain_index", 8), ("epoch", _HOST_BITS - 8 - 8 - 8))


def _column(values: ArrayLike, name: str, bits: int) -> np.ndarray:
    """A tweak field as a host `int64` column, bounded by the room it packs into.

    The bound is the whole point, and it is two hazards wearing one check. A
    field packed under another does not overflow when it runs past its range —
    it *carries* into the field above and packs a valid tweak for a different
    position, which is a wrong digest with nothing to notice it. The top field
    has no neighbour to corrupt, but its shift happens in `int64`, and a shift
    past that wraps silently. Neither is caught by the limb-fit check in
    [`field.py`](field.py), which only ever sees the sum.

    The mechanism is `field.host_column`'s, which two other callers in this
    package share; what stays here is the sentence, because the two hazards
    above are what make *this* bound worth stating.
    """
    return host_column(
        values,
        f"{name} takes {bits} bits in a packed tweak, so it must be in "
        f"[0, {1 << bits})",
        1 << bits,
    )


def _packed(
    prefix: int, layout: Sequence[tuple[str, int]], *values: ArrayLike
) -> np.ndarray:
    """`values` packed above `prefix` in `layout`'s fields, low to high.

    One packer for both tweaks, so the two cannot drift in how they bound a
    field or where they put the prefix — only in the layout they hand it.
    """
    packed = np.asarray(prefix, dtype=np.int64)
    shift = _PREFIX_BITS
    for (name, bits), value in zip(layout, values, strict=True):
        packed = packed | (_column(value, name, bits) << shift)
        shift += bits
    return packed


def tree_tweaks(level: int, indices: ArrayLike, *, params: LeanSigParams) -> Array:
    """`TreeTweak(level, index)` for a whole level: -> `[B, tweak_length]`.

    `level` is shared — a Merkle walk hashes one level at a time — and `indices`
    is per entry. Host-only, per the module docstring.
    """
    return lane_reversed_limbs(
        _packed(TWEAK_PREFIX_TREE, _TREE_FIELDS, indices, level), params.tweak_length
    )


def _chain_base(epochs: ArrayLike, chain_indices: ArrayLike) -> np.ndarray:
    """A chain tweak with its step left at zero — everything a walk does not move.

    A chain's epoch and index are its own whatever step it is at, so a walk packs
    them once and each step contributes its own byte. Rebuilding the whole tweak
    per step is the mistake [`wots.py`](../wots.py)'s `_chain_addresses` records
    a measurement against: at SLH-DSA-SHA2-128f re-encoding the invariant fields
    was two fifths of an eager verification.
    """
    return _packed(TWEAK_PREFIX_CHAIN, _CHAIN_FIELDS, 0, chain_indices, epochs)


def chain_tweaks(
    epochs: ArrayLike, chain_indices: ArrayLike, step: int, *, params: LeanSigParams
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
    step_bits = _CHAIN_FIELDS[0][1]
    packed = _chain_base(epochs, chain_indices) | (
        _column(step, "step", step_bits) << _PREFIX_BITS
    )
    return lane_reversed_limbs(packed, params.tweak_length)


def node_tweaks(params: LeanSigParams) -> tree.NodeAddresses:
    """The builder [`tree.py`](../tree.py) tweaks each Merkle level with.

    The same seam `tree.xmss_node_addresses` fills for FIPS 205: the walk takes a
    builder and never learns what one encodes, which is what lets one tree climb
    over byte addresses and over field elements. Host-only, unlike that one —
    `tree.NodeAddresses` records what that costs a caller.
    """
    return partial(tree_tweaks, params=params)


def chain_columns(
    slots: ArrayLike, *, params: LeanSigParams
) -> tuple[np.ndarray, np.ndarray]:
    """Every chain of every slot, as the `(epoch, chain_index)` column pair.

    `slots` is `[B]` and the result is two `[B * dimension]` columns naming one
    chain each, slot-major: slot 0's `dimension` chains, then slot 1's.

    That layout is an invariant three callers have to agree on byte for byte —
    the verifier's chain walk, the signer's leaf build and its release — and it
    is stated once here because the two consumers of the pair are both in this
    package: `chain_step_tweaks` below, and
    [`prf.chain_starts`](prf.py), which take it as their signature. Spelling the
    `repeat`/`tile` at each call site leaves three chances to transpose them,
    which is a walk over the right chains at the wrong positions rather than an
    error. `wots._position_columns` ([`../wots.py`](../wots.py)) is the same move
    for FIPS 205's address prefix.
    """
    epochs = np.asarray(slots, dtype=np.int64).reshape(-1)
    return (
        np.repeat(epochs, params.dimension),
        np.tile(np.arange(params.dimension), epochs.size),
    )


def chain_step_tweaks(
    epochs: ArrayLike, chain_indices: ArrayLike, *, params: LeanSigParams
) -> Iterator[Array]:
    """Every step's tweaks for a full walk, in [`wots.chain`](../wots.py)'s order.

    `base - 1` batches, the walk's own length, and the `+ 1` is upstream's
    one-indexing kept in one place: `chain`'s `j`-th iteration is absolute step
    `j + 1`, and a caller that spelled that itself would be one rename away from
    hashing every step at the wrong position.

    Every step is packed in one pass over one array rather than one pass each.
    The invariant half is `_chain_base`'s reason; the rest is that the base-p
    decomposition and the field conversion under `lane_reversed_limbs` are the
    expensive part, and a `[base - 1, B, 2]` array pays them once instead of
    `base - 1` times — including one host-to-device transfer instead of seven.
    That is the shape `wots._chain_addresses` already hands `chain`, at addresses
    far larger than these.
    """
    steps = np.arange(1, params.base, dtype=np.int64)
    _column(steps, "step", _CHAIN_FIELDS[0][1])
    packed = _chain_base(epochs, chain_indices)[None, :] | (
        steps[:, None] << _PREFIX_BITS
    )
    return iter(lane_reversed_limbs(packed, params.tweak_length))


@lru_cache(maxsize=None)
def _compression(width: int, output_length: int) -> Callable[..., Array]:
    """`compress` over a leading batch axis, one memoized callable per shape."""

    def one(*operands: Array) -> Array:
        return poseidon.compress(operands, width=width, output_length=output_length)

    return frx.vmap(one)


@lru_cache(maxsize=None)
def _leaf_sponge(output_length: int) -> Callable[..., Array]:
    """The leaf's sponge over a leading batch axis, with a shared capacity.

    The capacity rides `in_axes=None` because it is one value for the whole
    batch — it depends only on the configuration — so the separator is computed
    once per call rather than per leaf.

    The chain ends arrive already joined, by `poseidon.join_digests` and outside
    this trace. Upstream names them as `dimension` separate operands and handing
    them over that way is the same pre-image, but it is `dimension` slices into
    a `dimension + 2`-way concatenate inside the batched body — 48 operands at
    `PROD`, against one.
    """

    def one(parameter: Array, tweak: Array, ends: Array, capacity: Array) -> Array:
        return poseidon.sponge(
            (parameter, tweak, ends),
            capacity,
            width=_TREE_WIDTH,
            output_length=output_length,
        )

    return frx.vmap(one, in_axes=(0, 0, 0, None))


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
            _batched(digest, batch, dtype=F),
            _batched(parameter, batch, dtype=F),
            tweaks,
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
            _batched(parameter, batch, dtype=F),
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
        return _leaf_sponge(self.n)(
            _batched(parameter, tweaks.shape[0], dtype=F),
            tweaks,
            poseidon.join_digests(ends),
            self.capacity_value,
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


if TYPE_CHECKING:
    # The two protocols the shared walks take, pinned here so mypy fails this
    # module when it drifts rather than the walk that calls it —
    # [`signature.py`](../../signature.py) states the convention for the seam
    # proper, and this is the same argument one layer down. `TweakableHash` is
    # deliberately absent: this family has no `prf`, `prf_msg` or `h_msg`.
    _chain: type[ChainHash] = LeanSigTweakableHash
    _node: type[NodeHash] = LeanSigTweakableHash
