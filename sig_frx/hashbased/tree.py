# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The Merkle hash tree SLH-DSA and XMSS both build over WOTS+ public keys.

FIPS 205 §6.1. A node is `H(PK.seed, ADRS, left ‖ right)` tweaked by its height
and index, the root is the public key, and a signature carries the sibling at
each level so a verifier reaches the root in `h` hashes rather than `2^h`.

Two directions, and they are not symmetric:

- `root` and `auth_path` build a whole tree, which is key generation and signing
  — once per key, not on the hot path. A level is one batched hash over that
  level's nodes and the levels are a static Python loop, since the height is a
  parameter rather than data.
- `root_from_path` is what verification runs, so it takes a **batch of
  signatures**: many leaves, many indices, many paths, one call per level. The
  left/right choice at each level is a select on the index bit, never a branch —
  a branch there would be per-signature control flow, which is what the batch
  exists to avoid.

hash-frx's `Compression` does not fit and this does not use it: that one is an
n-to-1 truncated-permutation compression over a field `Permutation`, while these
nodes are a byte hash tweaked by an address. The two share the word "compression"
and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from sig_frx.hashbased import adrs
from sig_frx.hashbased.tweakable import TweakableHash


@dataclass(frozen=True)
class TreePosition:
    """Which tree this is — the address prefix every node in it tweaks with."""

    layer: int
    tree: int


def _node_addresses(
    position: TreePosition, height: int, indices: ArrayLike
) -> np.ndarray:
    """The TREE addresses of the given nodes at one height."""
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


def root(
    tweak: TweakableHash,
    pk_seed: ArrayLike,
    leaves: ArrayLike,
    position: TreePosition,
) -> Array:
    """The root of a tree over `leaves` — `[2^h, n]` -> `[n]`.

    Algorithm 9 as an iteration rather than a recursion: the standard's recursive
    form computes each level's nodes one at a time, and a level's nodes are
    independent, so each level is one batched hash. `h` calls, not `2^h − 1`.
    """
    nodes = fnp.asarray(leaves, dtype=fnp.uint8)
    count = nodes.shape[0]
    if count == 0 or count & (count - 1):
        raise ValueError(f"a tree needs a power-of-two number of leaves, got {count}")
    height = 0
    while nodes.shape[0] > 1:
        height += 1
        pairs = nodes.reshape(nodes.shape[0] // 2, 2 * nodes.shape[1])
        nodes = tweak.h(
            pk_seed,
            _node_addresses(position, height, np.arange(pairs.shape[0])),
            pairs,
        )
    return nodes[0]


def auth_path(
    tweak: TweakableHash,
    pk_seed: ArrayLike,
    leaves: ArrayLike,
    index: int,
    position: TreePosition,
) -> Array:
    """The siblings a verifier needs to reach the root from leaf `index`.

    Algorithm 10 lines 1 to 4: at each level, the sibling of the node the leaf
    sits under. `[h, n]`, lowest level first — the order `root_from_path`
    consumes them in.
    """
    nodes = fnp.asarray(leaves, dtype=fnp.uint8)
    count = nodes.shape[0]
    if not 0 <= index < count:
        raise ValueError(f"leaf {index} is outside a tree of {count} leaves")
    path = []
    height = 0
    while nodes.shape[0] > 1:
        path.append(nodes[(index >> height) ^ 1])
        height += 1
        pairs = nodes.reshape(nodes.shape[0] // 2, 2 * nodes.shape[1])
        nodes = tweak.h(
            pk_seed,
            _node_addresses(position, height, np.arange(pairs.shape[0])),
            pairs,
        )
    return fnp.stack(path)


def root_from_path(
    tweak: TweakableHash,
    pk_seed: ArrayLike,
    leaves: ArrayLike,
    indices: ArrayLike,
    paths: ArrayLike,
    position: TreePosition,
) -> Array:
    """The root each `(leaf, index, path)` implies — the verifier's operation.

    Batched: `leaves` is `[B, n]`, `indices` is `[B]`, `paths` is `[B, h, n]`, and
    the result is `[B, n]`. One hash call per level for the whole batch, `h`
    levels deep, which is Algorithm 11 lines 6 to 18 with the batch axis added.

    The per-level left/right choice is a select on the index bit rather than a
    branch: entry `k` puts its sibling on the side its own index says, and every
    entry does the same work.
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
        # A node whose own index is odd is the right child, so its sibling goes
        # first. The bit is public — it is the leaf index a signature carries —
        # but expressing it as a select keeps every entry on one code path.
        on_the_right = fnp.asarray((leaf_indices >> level) & 1, dtype=fnp.uint8)[
            :, None
        ]
        left = fnp.where(on_the_right, siblings, nodes)
        right = fnp.where(on_the_right, nodes, siblings)
        nodes = tweak.h(
            pk_seed,
            _node_addresses(position, level + 1, leaf_indices >> (level + 1)),
            fnp.concatenate([left, right], axis=-1),
        )
    return nodes
