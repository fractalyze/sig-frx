# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The Merkle hash tree SLH-DSA's XMSS and FORS layers are both built from.

FIPS 205 §6.1 and §8.2. A node is `H(PK.seed, ADRS, left ‖ right)` tweaked by its
height and index, and a signature carries the sibling at each level so a verifier
reaches the root in `h` hashes rather than `2^h`. §8.2 says outright that
`fors_node` "is the same as `xmss_node`, except that the leaf nodes are the
hashes of the FORS secret values instead of WOTS+ public keys" — so the machinery
lands once and the two layers differ only in the leaves they hand it and the
address type they tweak with.

That address type is why these take a `node_addresses` callable rather than a
position: the XMSS trees tweak with `TREE` addresses and the FORS trees with
`FORS_TREE`, and FORS numbers its nodes across all `k` of its trees at once.
`xmss_node_addresses` is the XMSS builder; FORS supplies its own.

Two directions, and they are not symmetric because their callers are not:

- `reduce_levels`, `root` and `auth_path` build whole trees — key generation and
  signing, once per key. A level is one batched hash over that level's nodes and
  the levels are a static Python loop, since the height is a parameter rather
  than data.
- `root_from_path` is what verification runs, so it takes a **batch**: many
  leaves, many indices, many paths, one hash call per level. The left/right
  choice is a select on the index bit rather than a branch — a branch there is
  per-signature control flow, which is what the batch exists to remove.

hash-frx's `Compression` does not fit and this does not use it: that one is an
n-to-1 truncated-permutation compression over a field `Permutation`, while these
nodes are a byte hash tweaked by an address. The two share the word and nothing
else.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from sig_frx.hashbased import adrs
from sig_frx.hashbased.tweakable import TweakableHash

# Builds the addresses of the given nodes at one height: `(height, indices)`.
NodeAddresses = Callable[[int, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class TreePosition:
    """Which XMSS tree this is — the address prefix every node in it tweaks with."""

    layer: int
    tree: int


def xmss_node_addresses(position: TreePosition) -> NodeAddresses:
    """The `TREE`-type addresses an XMSS tree's nodes tweak with."""

    def build(height: int, indices: np.ndarray) -> np.ndarray:
        return adrs.encode_batch(
            [
                adrs.hash_tree(
                    layer=position.layer,
                    tree=position.tree,
                    height=height,
                    index=int(index),
                )
                for index in np.asarray(indices).reshape(-1)
            ],
            compressed=True,
        )

    return build


def reduce_levels(
    tweak: TweakableHash,
    pk_seed: ArrayLike,
    nodes: ArrayLike,
    levels: int,
    node_addresses: NodeAddresses,
) -> Array:
    """Hash `levels` levels of a forest, halving the node count each time.

    One batched hash per level, whatever the forest's width — which is what lets
    FORS reduce all `k` of its trees in one pass: its node numbering is contiguous
    across them, so a level's pairs are exactly the per-tree pairs.
    """
    current = fnp.asarray(nodes, dtype=fnp.uint8)
    for level in range(levels):
        if current.shape[0] % 2:
            raise ValueError(
                f"level {level} has an odd node count ({current.shape[0]}); "
                f"a Merkle level pairs its nodes"
            )
        pairs = current.reshape(current.shape[0] // 2, 2 * current.shape[1])
        current = tweak.h(
            pk_seed,
            node_addresses(level + 1, np.arange(pairs.shape[0])),
            pairs,
        )
    return current


def root(
    tweak: TweakableHash,
    pk_seed: ArrayLike,
    leaves: ArrayLike,
    node_addresses: NodeAddresses,
) -> Array:
    """The root of one tree over `leaves` — `[2^h, n]` -> `[n]`.

    Algorithm 9 as an iteration rather than a recursion: a level's nodes are
    independent, so each is one batched hash. `h` calls, not `2^h − 1`.
    """
    count = int(np.shape(leaves)[0])
    if count == 0 or count & (count - 1):
        raise ValueError(f"a tree needs a power-of-two number of leaves, got {count}")
    return reduce_levels(
        tweak, pk_seed, leaves, count.bit_length() - 1, node_addresses
    )[0]


def auth_path(
    tweak: TweakableHash,
    pk_seed: ArrayLike,
    leaves: ArrayLike,
    indices: ArrayLike,
    levels: int,
    node_addresses: NodeAddresses,
) -> Array:
    """The siblings a verifier needs, for one leaf per tree in the forest.

    Algorithm 10 lines 1 to 4 and Algorithm 16 lines 5 to 8, which are the same
    walk: at each level, the sibling of the node the leaf sits under. Returns
    `[len(indices), levels, n]`, lowest level first — the order `root_from_path`
    consumes them in.
    """
    nodes = fnp.asarray(leaves, dtype=fnp.uint8)
    leaf_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if np.any(leaf_indices < 0) or np.any(leaf_indices >= nodes.shape[0]):
        raise ValueError(
            f"leaf indices {leaf_indices.tolist()} are outside a forest of "
            f"{nodes.shape[0]} leaves"
        )
    path = []
    for level in range(levels):
        path.append(nodes[(leaf_indices >> level) ^ 1])
        nodes = reduce_levels(tweak, pk_seed, nodes, 1, _shifted(node_addresses, level))
    return fnp.stack(path, axis=1)


def _shifted(node_addresses: NodeAddresses, by: int) -> NodeAddresses:
    """A builder whose heights are offset, for reducing one level at a time."""

    def build(height: int, indices: np.ndarray) -> np.ndarray:
        return node_addresses(height + by, indices)

    return build


def root_from_path(
    tweak: TweakableHash,
    pk_seed: ArrayLike,
    leaves: ArrayLike,
    indices: ArrayLike,
    paths: ArrayLike,
    node_addresses: NodeAddresses,
) -> Array:
    """The root each `(leaf, index, path)` implies — the verifier's operation.

    Batched: `leaves` is `[B, n]`, `indices` is `[B]`, `paths` is `[B, h, n]`, and
    the result is `[B, n]`. Algorithm 11 lines 6 to 18 (and Algorithm 17 lines 8
    to 18, which are identical) with the batch axis added.

    The index is the node's number in the whole forest, and it serves two roles:
    it addresses the node, and its bit at each level says which side the sibling
    goes. Those agree even for FORS, whose leaf `i·2^a + local` sits in tree `i` —
    `i·2^(a−j)` is even for every level `j < a`, so the forest-wide index and the
    within-tree one have the same parity at every level a path walks through.
    """
    nodes = fnp.asarray(leaves, dtype=fnp.uint8)
    sibling_paths = fnp.asarray(paths, dtype=fnp.uint8)
    leaf_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if sibling_paths.shape[0] != nodes.shape[0] or len(leaf_indices) != nodes.shape[0]:
        raise ValueError(
            f"one index and one path per leaf: got {nodes.shape[0]} leaves, "
            f"{len(leaf_indices)} indices, {sibling_paths.shape[0]} paths"
        )

    for level in range(sibling_paths.shape[1]):
        siblings = sibling_paths[:, level, :]
        on_the_right = fnp.asarray((leaf_indices >> level) & 1, dtype=fnp.uint8)[
            :, None
        ]
        left = fnp.where(on_the_right, siblings, nodes)
        right = fnp.where(on_the_right, nodes, siblings)
        nodes = tweak.h(
            pk_seed,
            node_addresses(level + 1, leaf_indices >> (level + 1)),
            fnp.concatenate([left, right], axis=-1),
        )
    return nodes
