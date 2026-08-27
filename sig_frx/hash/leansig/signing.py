# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The signer's side: the lifetime tree, what stays resident, and the window.

A leanSig public key is the root of one Merkle tree with a leaf per slot, and at
`PROD` that is `2^32` leaves. No signer holds them, so upstream cuts the tree in
half by height: one **top tree** whose leaves are the roots of `2^(h/2)` **bottom
trees**, each covering `2^(h/2)` consecutive slots. The signer keeps the whole
top tree plus the two bottom trees around the active slot, and slides that pair
forward as slots are spent — so resident state is the square root of the
lifetime rather than the lifetime.

**It is not XMSS-MT's hypertree**, which the shape invites a reader to assume.
There, each layer signs the root below it with its own one-time key and a
signature carries `d` WOTS+ signatures. Here the bottom roots are ordinary
*leaves* of one tree, the split is a storage decision alone, and a signature
carries one authentication path of `log_lifetime` siblings that happens to be
served from two resident objects — which is why `combined_path` below is a
concatenation and not a second signing step.

## What this module builds, and what it borrows

Nothing here re-implements a Merkle walk. [`tree.reduce_levels`](../tree.py)
climbs a level and [`wots.chain`](../wots.py) walks the chains, both driven by
[`tweakable.py`](tweakable.py)'s family — the same two functions the verifier
runs, which is what makes a round trip evidence of anything. What is this
module's own is the *addressing*: a subtree's nodes are numbered in whole-tree
coordinates, so a builder that handed `tree.py` its local indices would tweak
every node at the wrong position and produce a self-consistently wrong key.

## Key generation is full-lifetime, and that is a real restriction

Upstream's `key_gen` takes an activation window and snaps it out to whole bottom
trees, building only those. The unbuilt part of the tree still has to hash to
*something*, and upstream fills it with `random_domain` — fresh OS randomness,
one digest per unpaired node per level. That is sound (an unbuilt leaf should be
unusable) and it means **a partial window's public key is not a function of its
inputs**: generating twice from one seed and one parameter gives two different
keys. Confirmed against upstream directly — a window of `[0, 32)` at `TEST`
changes root when the pad source changes, and the full window does not.

So `keygen` here covers the whole lifetime, where no pad is ever load-bearing:
every layer is dense, the one pad upstream draws sits beside a subtree root and
is discarded unread. A partial window is refused rather than silently seeded,
because a key nobody can regenerate cannot be gated
([`testing.md`](../../../docs/reference/testing.md)). Supporting one means taking
the pads as an argument the way `parameter` and `prf_key` are taken — the same
move, waiting for the caller that needs it.

The cost is that this is a `TEST`-preset operation. A full `PROD` lifetime is
`2^32` leaves and `2^32 · 46` chain starts; nothing generates one, which is why
upstream publishes eight `PROD` keys rather than a recipe for making them.

## Where the batch is

One call per *level*, and one call per chain *step* — never one per node or per
slot. A `TEST` key pair is 256 slots of 4 chains, which is one `wots.chain` over
1024 rows in 7 batched hashes, one leaf sponge over 256 rows, and 8 batched
levels: about 20 dispatches for a whole key, against the ~9000 tweaked hashes
upstream's nested loops make. That the leaves of every bottom tree are built at
once is what `reduce_levels` already allows for FORS — a level's pairs are
per-subtree pairs, so a dense climb reduces all `2^(h/2)` bottom trees
simultaneously and the split reappears only when the layers are *sliced* into
what stays resident.

`advance_preparation` is the one path that builds a single bottom tree, because
that is genuinely all it needs.

## Host and device

The layers are device arrays; every index into them is the host's. That is the
same boundary the verifier has and for the same reason — a tweak packs a
position past a lane ([`tweakable.py`](tweakable.py)) — and it is why a slot
arrives here as a Python integer rather than as an operand.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from sig_frx.hash import tree, wots
from sig_frx.hash.leansig import encoding, prf, tweakable
from sig_frx.hash.leansig.params import LeanSigParams
from sig_frx.hash.leansig.tweakable import LeanSigTweakableHash


@dataclass(frozen=True)
class SubTree:
    """One resident slice of the lifetime tree, layer by layer up to its root.

    `layers[0]` is the lowest layer and `layers[-1]` holds the single node this
    subtree roots at, so a subtree of `k` levels carries `k + 1` layers. Every
    layer is dense and contiguous — the sparse `start_index` bookkeeping upstream
    carries per layer collapses to one number here, because a full-lifetime build
    leaves no holes.
    """

    lowest_layer: int
    """Which level of the whole tree `layers[0]` sits at — 0 for a bottom tree,
    `log_lifetime / 2` for the top one."""

    start_index: int
    """The whole-tree index of `layers[0]`'s first node.

    Aligned to the subtree's own width, which is what lets a sibling be found by
    flipping a bit of the *relative* index: the layer at height `h` starts at
    `start_index >> h`, and that is even at every height a path climbs through,
    so subtracting it commutes with the flip.
    """

    layers: tuple[Array, ...]
    """`[count, hash_length]` per layer, lowest first."""

    def __post_init__(self) -> None:
        levels = len(self.layers) - 1
        if levels < 0 or self.layers[-1].shape[0] != 1:
            raise ValueError(
                "a subtree runs from its lowest layer up to a single root node; "
                f"got {len(self.layers)} layers ending in "
                f"{0 if not self.layers else self.layers[-1].shape[0]} nodes"
            )
        if self.start_index % (1 << levels):
            raise ValueError(
                f"a subtree of {levels} levels starts on a multiple of "
                f"{1 << levels}; got {self.start_index}"
            )

    @property
    def root(self) -> Array:
        """The single node the top layer holds: `[hash_length]`."""
        return self.layers[-1][0]

    def path(self, position: int) -> Array:
        """The siblings connecting leaf `position` to this root: `[levels, n]`.

        `position` is in whole-tree coordinates, as every index here is. At each
        level the sibling is the node sharing a parent — the position with its
        low bit flipped — and the walk stops one level below the root, which has
        none.
        """
        lowest = self.layers[0]
        if not self.start_index <= position < self.start_index + lowest.shape[0]:
            raise ValueError(
                f"this subtree covers positions "
                f"[{self.start_index}, {self.start_index + lowest.shape[0]}); "
                f"got {position}"
            )
        siblings = []
        index = position
        for height, layer in enumerate(self.layers[:-1]):
            siblings.append(layer[(index ^ 1) - (self.start_index >> height)])
            index >>= 1
        return fnp.stack(siblings)


@dataclass(frozen=True)
class SecretKey:
    """What a leanSig signer holds — a seed, a parameter, and three subtrees.

    **A materialized state rather than a seed**, which is why this is a value and
    not a byte string: the seam's `secret_key_size` is a constant of a parameter
    set at every other scheme here, and there is no such constant for an object
    whose size is a residency decision. `prf_key` alone would regenerate all of
    it; the trees are the cache that makes signing a path lookup instead of a
    rebuild.

    No activation window is carried. Upstream's is a sub-range of the lifetime
    and its two fields say which; `keygen` builds the whole lifetime (see the
    module docstring), so the range is the preset's and restating it would be a
    second place for it to be wrong.
    """

    params: LeanSigParams
    """The preset this key was built at.

    Carried so a key and a scheme cannot be crossed. Every digest here is tweaked
    by a position whose packing depends on the preset, so signing a `TEST` key
    under `PROD` produces a signature that is wrong rather than one that fails a
    check.
    """

    prf_key: bytes
    """The 32-byte master seed ([`prf.py`](prf.py)). The whole secret."""

    parameter: Array
    """The public parameter, lane-reversed — `[parameter_length]`.

    Public, and in the secret key because every hash the signer makes is
    personalized by it. It also travels in the public key.
    """

    left_bottom_tree_index: int
    """Which bottom tree the resident pair starts at."""

    left_bottom_tree: SubTree
    right_bottom_tree: SubTree
    """The two resident bottom trees — the prepared window, in the order they
    cover slots."""

    top_tree: SubTree
    """Every level from the bottom roots up to the public root."""

    @property
    def prepared(self) -> range:
        """The slots this key can sign without rebuilding a bottom tree.

        Two bottom trees wide, and `advance_preparation` is what moves it. A slot
        outside it is refused rather than served, because serving it means
        deriving a bottom tree's worth of chain starts inside a signature.
        """
        width = leaves_per_bottom_tree(self.params)
        start = self.left_bottom_tree_index * width
        return range(start, start + 2 * width)


def leaves_per_bottom_tree(params: LeanSigParams) -> int:
    """`LEAVES_PER_BOTTOM_TREE` — slots under one bottom tree, `2^(h/2)`.

    The square root of the lifetime, which is what makes the resident state that
    size. A module function rather than a `LeanSigParams` property for the reason
    [`ssz.signature_size`](ssz.py) is one: it belongs to the split, and the split
    lives here.
    """
    return 1 << (params.log_lifetime // 2)


def leaves(
    family: LeanSigTweakableHash,
    parameter: ArrayLike,
    prf_key: bytes,
    slots: ArrayLike,
    *,
    params: LeanSigParams,
) -> Array:
    """The one-time public key committed at each of `slots`: `[N, hash_length]`.

    One leaf is `dimension` chains walked from their secret starts to their far
    ends and hashed together — the same walk and the same sponge the verifier
    finishes and rebuilds, run forward from step zero over the whole batch.

    The rows are slot-major, matching the layout `chain_step_tweaks` and
    `wots.chain` both take: slot 0's `dimension` chains, then slot 1's.
    """
    epochs = np.asarray(slots, dtype=np.int64).reshape(-1)
    dimension = params.dimension
    rows = epochs.size * dimension
    chains = np.tile(np.arange(dimension), epochs.size)
    per_chain_slots = np.repeat(epochs, dimension)
    ends = wots.chain(
        family,
        parameter,
        prf.chain_starts(prf_key, per_chain_slots, chains, params=params),
        fnp.zeros(rows, dtype=fnp.uint32),
        fnp.full(rows, params.base - 1, dtype=fnp.uint32),
        tweakable.chain_step_tweaks(per_chain_slots, chains, params=params),
    )
    return family.leaf(
        parameter,
        tweakable.tree_tweaks(0, epochs, params=params),
        ends.reshape(epochs.size, dimension, params.hash_length),
    )


def climb(
    family: LeanSigTweakableHash,
    parameter: ArrayLike,
    nodes: ArrayLike,
    *,
    lowest_layer: int,
    start_index: int,
    levels: int,
    params: LeanSigParams,
) -> tuple[Array, ...]:
    """`nodes` and every layer above them, up to a single root.

    One batched hash per level, and the levels are a Python loop because the
    height is a parameter rather than data — [`tree.py`](../tree.py)'s own split
    between what a tree builder and a path verifier each are.

    Returned rather than reduced away because the signer needs the intermediate
    layers: an authentication path is one node from each of them, and rebuilding
    the tree per signature is what the resident state exists to avoid.
    """
    layers = [fnp.asarray(nodes, dtype=family.dtype)]
    for height in range(levels):
        layers.append(
            tree.reduce_levels(
                family,
                parameter,
                layers[-1],
                1,
                _node_addresses(
                    lowest_layer + height, start_index >> height, params=params
                ),
            )
        )
    return tuple(layers)


def bottom_tree(
    family: LeanSigTweakableHash,
    parameter: ArrayLike,
    prf_key: bytes,
    index: int,
    *,
    params: LeanSigParams,
) -> SubTree:
    """Bottom tree `index`, rebuilt from the seed — upstream's `from_prf_key`.

    The one path that builds a single tree. `keygen` does not use it: building
    the lifetime's leaves at once and slicing the layers is the same nodes in a
    fraction of the dispatches, and only `advance_preparation` genuinely needs
    one tree in isolation.
    """
    width = leaves_per_bottom_tree(params)
    half = params.log_lifetime // 2
    start = index * width
    return SubTree(
        lowest_layer=0,
        start_index=start,
        layers=climb(
            family,
            parameter,
            leaves(
                family,
                parameter,
                prf_key,
                np.arange(start, start + width),
                params=params,
            ),
            lowest_layer=0,
            start_index=start,
            levels=half,
            params=params,
        ),
    )


def keygen(
    family: LeanSigTweakableHash,
    prf_key: bytes,
    parameter: ArrayLike,
    *,
    params: LeanSigParams,
) -> tuple[Array, SecretKey]:
    """A key pair over the whole lifetime: `(root, secret key)`.

    Both secrets are taken rather than drawn. Upstream's `key_gen` calls
    `os.urandom` and `secrets.randbelow` itself, so its keys cannot be
    reproduced; taking them is what makes a key a function of published bytes,
    which is the same choice [`xmss.py`](../xmss/xmss.py)'s `keygen` makes about
    RFC 8391's three seeds.

    The whole lifetime, and the module docstring says why a sub-range is refused
    rather than supported.
    """
    half = params.log_lifetime // 2
    width = leaves_per_bottom_tree(params)
    layers = climb(
        family,
        parameter,
        leaves(
            family,
            parameter,
            prf_key,
            np.arange(1 << params.log_lifetime),
            params=params,
        ),
        lowest_layer=0,
        start_index=0,
        levels=params.log_lifetime,
        params=params,
    )
    # The split is a slice of one dense climb: bottom tree `i` owns the nodes
    # `[i * width, (i + 1) * width)` of the leaf layer and their reductions, and
    # the top tree is every layer from the bottom roots up.
    resident = [
        SubTree(
            lowest_layer=0,
            start_index=index * width,
            layers=tuple(
                layer[(index * width) >> height : ((index + 1) * width) >> height]
                for height, layer in enumerate(layers[: half + 1])
            ),
        )
        for index in (0, 1)
    ]
    top = SubTree(lowest_layer=half, start_index=0, layers=layers[half:])
    return top.root, SecretKey(
        params=params,
        prf_key=bytes(prf_key),
        parameter=fnp.asarray(parameter, dtype=family.dtype),
        left_bottom_tree_index=0,
        left_bottom_tree=resident[0],
        right_bottom_tree=resident[1],
        top_tree=top,
    )


def advance_preparation(
    family: LeanSigTweakableHash, secret_key: SecretKey
) -> SecretKey:
    """The prepared window slid one bottom tree forward.

    The previous right tree becomes the left one and the next is rebuilt from the
    seed, so the cost of advancing is one bottom tree however far the key has
    already run. A key whose window already reaches the end of the lifetime comes
    back unchanged — upstream's own answer, and it keeps a caller that advances
    in a loop from having to bound the loop itself.
    """
    params = secret_key.params
    width = leaves_per_bottom_tree(params)
    left = secret_key.left_bottom_tree_index
    if (left + 3) * width > 1 << params.log_lifetime:
        return secret_key
    return replace(
        secret_key,
        left_bottom_tree_index=left + 1,
        left_bottom_tree=secret_key.right_bottom_tree,
        right_bottom_tree=bottom_tree(
            family, secret_key.parameter, secret_key.prf_key, left + 2, params=params
        ),
    )


_GRIND_BLOCK: Final = 128
"""How many randomness candidates one pass of `search` tries.

The loop is the host's, so the choice is between dispatches and wasted hashes.
Landing on the target layer takes about 49 attempts at `TEST` and about 909 at
`PROD`, so 128 finishes `TEST` in one pass better than nine times in ten and
`PROD` in about eight — against the 49 and 909 dispatches a
candidate-at-a-time loop makes. What it costs is the half block past the
candidate that lands, which is one batched compression's worth of rows either
way.

`wots_c.grind` ([`../shrincs/wots_c.py`](../shrincs/wots_c.py)) is the shape this
follows, and [`encoding.py`](encoding.py) is where it was named as the one a
signer should want. It divides its limit exactly and this does not —
`max_tries` is 100,000 — so the last pass is clamped rather than the block
chosen to fit.
"""


def search(secret_key: SecretKey, slot: int, message: bytes) -> tuple[Array, Array]:
    """The lowest attempt whose codeword lands on the target layer.

    Returns `(that attempt's randomness, its codeword)`. This is leanSig's
    rejection loop, and [`conventions.md`](../../../docs/reference/conventions.md)
    asks every scheme to say which of the two forms its loop takes: **this one is
    a host loop**, not a masked fixed-size sample. What makes that sound rather
    than merely convenient is that the acceptance test is a public function of
    public inputs — a verifier recomputes it from the randomness the signature
    carries — so a trip count that depends on the message leaks nothing a
    verifier does not already hold.

    A block of candidates per pass, for the reason `_GRIND_BLOCK` gives. The
    three operands that do not move across a block are broadcast rather than
    re-derived, which is what lets one batched `codewords` serve both this and
    the verifier.

    **The lowest**, not any that lands: upstream counts attempts from zero and
    stops at the first, so a signer that returned the last landing candidate of a
    pass would produce a perfectly valid signature that upstream's own signer
    disagrees with byte for byte.
    """
    params = secret_key.params
    message_elements = encoding.encode_message(bytes(message), params=params)
    epoch_elements = encoding.encode_epoch(slot, params=params)
    for base in range(0, params.max_tries, _GRIND_BLOCK):
        counters = np.arange(base, min(base + _GRIND_BLOCK, params.max_tries))
        randomness = prf.randomness(
            secret_key.prf_key, slot, message, counters, params=params
        )
        block = randomness.shape[0]
        digits, accepted = encoding.codewords(
            fnp.broadcast_to(message_elements, (block, params.message_length)),
            fnp.broadcast_to(secret_key.parameter, (block, params.parameter_length)),
            fnp.broadcast_to(epoch_elements, (block, params.tweak_length)),
            randomness,
            params=params,
        )
        landed = np.flatnonzero(np.asarray(accepted))
        if landed.size:
            first = int(landed[0])
            return randomness[first], digits[first]
    raise ValueError(
        f"no randomness below {params.max_tries} encodes this message onto the "
        f"target-sum layer; about one draw in "
        f"{round(1 / _acceptance_rate(params))} does, so this is a wrong "
        f"parameter set rather than bad luck"
    )


def release(
    family: LeanSigTweakableHash, secret_key: SecretKey, slot: int, digits: ArrayLike
) -> Array:
    """Each chain walked from its secret start to the digit it stops at.

    `[dimension, hash_length]` — the values a signature carries. The mirror of
    what a verifier does with them: it begins where this stops and applies the
    remaining `base - 1 - digit` steps, so the two meet at the chain end the leaf
    was built from, and a chain released one step either side of its digit misses
    it.

    One `wots.chain` over the slot's `dimension` chains, so the walk is `base - 1`
    batched hashes however the digits differ — the same masked walk `leaves` runs
    to the top and for the same reason.
    """
    params = secret_key.params
    chains = np.arange(params.dimension)
    slots = np.full(params.dimension, slot)
    return wots.chain(
        family,
        secret_key.parameter,
        prf.chain_starts(secret_key.prf_key, slots, chains, params=params),
        fnp.zeros(params.dimension, dtype=fnp.uint32),
        digits,
        tweakable.chain_step_tweaks(slots, chains, params=params),
    )


def combined_path(secret_key: SecretKey, slot: int) -> Array:
    """One leaf's whole authentication path: `[log_lifetime, hash_length]`.

    The bottom tree's siblings then the top tree's, in that order, because the
    path is read from the leaf upward and the bottom half is what the leaf sits
    in. Which resident tree holds the slot is the only choice here, and the
    prepared window's midpoint is what decides it.
    """
    width = leaves_per_bottom_tree(secret_key.params)
    prepared = secret_key.prepared
    if slot not in prepared:
        raise ValueError(
            f"this key has slots [{prepared.start}, {prepared.stop}) prepared; "
            f"got {slot}"
        )
    bottom = (
        secret_key.left_bottom_tree
        if slot < prepared.start + width
        else secret_key.right_bottom_tree
    )
    return fnp.concatenate([bottom.path(slot), secret_key.top_tree.path(slot // width)])


def _acceptance_rate(params: LeanSigParams) -> float:
    """What fraction of randomness draws lands a codeword on the target layer.

    Counted rather than quoted, so a failed search's message carries this
    parameter set's own number: the codewords of `dimension` digits in
    `[0, base)` summing to `target_sum`, over `base^dimension`. That count is the
    coefficient of `x^target_sum` in `(1 + x + ... + x^(base-1))^dimension`,
    accumulated as a polynomial product — about 1 in 909 at `PROD` and 1 in 49 at
    `TEST`.

    Only an error message reaches for this, so it is host integer arithmetic and
    exact until the final division. The abort in `aborting_decode` narrows the
    rate further, by about `4.7e-10` per element, which no message would show.
    """
    counts = [1]
    for _ in range(params.dimension):
        widened = [0] * (len(counts) + params.base - 1)
        for power, count in enumerate(counts):
            for digit in range(params.base):
                widened[power + digit] += count
        counts = widened
    landing = counts[params.target_sum] if params.target_sum < len(counts) else 0
    return landing / params.base**params.dimension


def _node_addresses(
    lowest_layer: int, start_index: int, *, params: LeanSigParams
) -> tree.NodeAddresses:
    """[`tree.py`](../tree.py)'s builder, for a subtree at whole-tree positions.

    The walk numbers a level's nodes from zero and this is what puts them where
    they belong: level `lowest_layer + height`, index offset by the subtree's own
    start at that height. Getting it wrong costs nothing visible — every node
    still hashes, the tree still has a root — and produces a key no verifier
    agrees with, which is the failure the whole-tree coordinates exist to
    prevent.
    """

    def build(height: int, indices: ArrayLike) -> Array:
        return tweakable.tree_tweaks(
            lowest_layer + height,
            np.asarray(indices, dtype=np.int64) + (start_index >> height),
            params=params,
        )

    return build
