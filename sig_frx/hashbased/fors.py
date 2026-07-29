# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""FORS — the few-time signature SLH-DSA signs a message digest with.

FIPS 205 §8. `k` Merkle trees of height `a`; the message digest picks one leaf
per tree, and the signature reveals that leaf's secret plus its authentication
path. Few-time, not one-time: revealing leaves from many signatures under one key
is what the hypertree above it exists to make improbable rather than impossible.

**The `k` trees are computed as one forest, not `k` trees.** FIPS 205 numbers
FORS nodes across all of them — tree `i`'s leaves are `i·2^a` through
`(i+1)·2^a − 1` — so the trees are contiguous, a Merkle level's pairs are exactly
the per-tree pairs, and `a` batched hashes reduce the whole forest to its `k`
roots. At the smallest parameter set that is 14 trees of 4096 leaves reduced in
12 calls rather than 14 separate walks.

That numbering is also why one index does two jobs in `tree.root_from_path`: the
forest-wide index addresses the node, and its bit says which side the sibling
goes, and those agree with the within-tree index at every level a path reaches
(see that function's docstring).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from sig_frx.hashbased import adrs, tree, wots
from sig_frx.hashbased.tweakable import TweakableHash


@dataclass(frozen=True)
class ForsParams:
    """`k` trees of height `a` over `n`-byte values — FIPS 205 §8."""

    n: int
    a: int
    k: int

    @cached_property
    def t(self) -> int:
        """Leaves per tree."""
        return 1 << self.a

    @cached_property
    def leaves(self) -> int:
        """Leaves in the whole forest."""
        return self.k * self.t


@dataclass(frozen=True)
class ForsPosition:
    """Which FORS key this is.

    The layer is always zero — §8.2: the XMSS tree that signs a FORS key is
    always at layer 0 — so only the tree and the key pair within it vary.
    """

    tree: int
    key_pair: int


def _node_addresses(position: ForsPosition) -> tree.NodeAddresses:
    """The `FORS_TREE` addresses this forest's nodes tweak with."""

    def build(height: int, indices: np.ndarray) -> np.ndarray:
        return adrs.encode_batch(
            [
                adrs.fors_tree(
                    layer=0,
                    tree=position.tree,
                    key_pair=position.key_pair,
                    height=height,
                    index=int(index),
                )
                for index in np.asarray(indices).reshape(-1)
            ],
            compressed=True,
        )

    return build


def secret_values(
    tweak: TweakableHash,
    pk_seed: ArrayLike,
    sk_seed: ArrayLike,
    position: ForsPosition,
    indices: ArrayLike,
) -> Array:
    """`fors_skGen` — Algorithm 14, for a batch of forest-wide leaf indices."""
    addresses = adrs.encode_batch(
        [
            adrs.fors_prf(
                layer=0,
                tree=position.tree,
                key_pair=position.key_pair,
                index=int(index),
            )
            for index in np.asarray(indices).reshape(-1)
        ],
        compressed=True,
    )
    return tweak.prf(pk_seed, sk_seed, addresses)


def leaves(
    tweak: TweakableHash,
    params: ForsParams,
    pk_seed: ArrayLike,
    sk_seed: ArrayLike,
    position: ForsPosition,
) -> Array:
    """Every leaf of the forest — Algorithm 15 lines 1 to 5, batched.

    A leaf is `F` over its secret value, tweaked at height 0 by the leaf's own
    forest-wide index.
    """
    indices = np.arange(params.leaves)
    secrets = secret_values(tweak, pk_seed, sk_seed, position, indices)
    return tweak.f(pk_seed, _node_addresses(position)(0, indices), secrets)


def message_indices(params: ForsParams, digest: ArrayLike) -> np.ndarray:
    """Which leaf of each tree the digest picks — `base_2b(md, a, k)`, forest-wide.

    Algorithm 16 line 2 gives the within-tree index of each tree's chosen leaf;
    tree `i`'s offset `i·2^a` turns that into the forest-wide numbering
    everything else here uses.
    """
    within = np.asarray(wots.base_2b(digest, params.a, params.k))[0]
    return np.arange(params.k) * params.t + within


def sign(
    tweak: TweakableHash,
    params: ForsParams,
    digest: ArrayLike,
    pk_seed: ArrayLike,
    sk_seed: ArrayLike,
    position: ForsPosition,
) -> Array:
    """`fors_sign` — Algorithm 16.

    `[k, a + 1, n]`: per tree, the chosen leaf's secret followed by its path.
    """
    indices = message_indices(params, digest)
    forest = leaves(tweak, params, pk_seed, sk_seed, position)
    paths = tree.auth_path(
        tweak, pk_seed, forest, indices, params.a, _node_addresses(position)
    )
    secrets = secret_values(tweak, pk_seed, sk_seed, position, indices)
    return fnp.concatenate([secrets[:, None, :], paths], axis=1)


def public_key(
    tweak: TweakableHash, pk_seed: ArrayLike, position: ForsPosition, roots: ArrayLike
) -> Array:
    """`T_k` over the `k` tree roots — Algorithm 17 lines 21 to 24."""
    address = adrs.encode_batch(
        [adrs.fors_roots(layer=0, tree=position.tree, key_pair=position.key_pair)],
        compressed=True,
    )
    stacked = fnp.asarray(roots, dtype=fnp.uint8)
    return tweak.t(pk_seed, address, stacked.reshape(1, -1))[0]


def pk_gen(
    tweak: TweakableHash,
    params: ForsParams,
    pk_seed: ArrayLike,
    sk_seed: ArrayLike,
    position: ForsPosition,
) -> Array:
    """The FORS public key: the forest reduced to `k` roots, then compressed."""
    roots = tree.reduce_levels(
        tweak,
        pk_seed,
        leaves(tweak, params, pk_seed, sk_seed, position),
        params.a,
        _node_addresses(position),
    )
    return public_key(tweak, pk_seed, position, roots)


def pk_from_sig(
    tweak: TweakableHash,
    params: ForsParams,
    signature: ArrayLike,
    digest: ArrayLike,
    pk_seed: ArrayLike,
    position: ForsPosition,
) -> Array:
    """`fors_pkFromSig` — Algorithm 17: the operation verification runs.

    All `k` trees walk their paths together: one hash call per level for the whole
    forest, `a` levels deep, then one `T_k` over the roots.
    """
    parts = fnp.asarray(signature, dtype=fnp.uint8)
    if parts.shape[:2] != (params.k, params.a + 1):
        raise ValueError(
            f"a FORS signature is {params.k} trees of {params.a + 1} values, "
            f"got shape {tuple(parts.shape)}"
        )
    indices = message_indices(params, digest)
    node_addresses = _node_addresses(position)
    leaf_nodes = tweak.f(pk_seed, node_addresses(0, indices), parts[:, 0, :])
    roots = tree.root_from_path(
        tweak, pk_seed, leaf_nodes, indices, parts[:, 1:, :], node_addresses
    )
    return public_key(tweak, pk_seed, position, roots)
