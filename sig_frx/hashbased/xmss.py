# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The XMSS layer — a Merkle tree over WOTS+ public keys. FIPS 205 §6.

One XMSS key signs `2^h'` messages, one per WOTS+ key pair beneath it, and its
public key is the tree's root. A signature is the WOTS+ signature plus the
authentication path from that key pair's leaf.

**A tree's leaves are one batched WOTS+ walk, not `2^h'` of them.** No key pair
depends on another, so all `2^h' · len` chains advance together: `w − 1` batched
hashes build every leaf of the tree, where walking key pairs one at a time would
be `2^h'` times that. At the smallest parameter set that is 512 key pairs of 35
chains — 17920 chains — in 15 hash calls.

Verification batches the other way. `pk_from_sig` takes many signatures, each
with its own leaf index in its own tree, because that is what SLH-DSA's verifier
holds: `d` layers of the hypertree, each layer a batch of independent claims.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from frx import Array
from frx.typing import ArrayLike

from sig_frx.hashbased import adrs, tree, wots
from sig_frx.hashbased.tweakable import TweakableHash


def key_pairs(position: tree.TreePosition, height: int) -> list[wots.WotsPosition]:
    """Every WOTS+ key pair under one XMSS tree, leaf order."""
    return [
        wots.WotsPosition(layer=position.layer, tree=position.tree, key_pair=index)
        for index in range(1 << height)
    ]


def leaves(
    tweak: TweakableHash,
    params: wots.WotsParams,
    pk_seed: ArrayLike,
    sk_seed: ArrayLike,
    position: tree.TreePosition,
    height: int,
) -> Array:
    """Every leaf of one XMSS tree — `[2^height, n]`, one batched WOTS+ walk."""
    positions = key_pairs(position, height)
    return wots.pk_gen(
        tweak,
        params,
        pk_seed,
        sk_seed,
        positions,
        wots.fips205_compression(tweak, pk_seed, positions),
    )


def root(
    tweak: TweakableHash,
    params: wots.WotsParams,
    pk_seed: ArrayLike,
    sk_seed: ArrayLike,
    position: tree.TreePosition,
    height: int,
) -> Array:
    """The XMSS public key: the root over this tree's WOTS+ public keys."""
    return tree.root(
        tweak,
        pk_seed,
        leaves(tweak, params, pk_seed, sk_seed, position, height),
        tree.xmss_node_addresses(position),
    )


def sign(
    tweak: TweakableHash,
    params: wots.WotsParams,
    message: ArrayLike,
    pk_seed: ArrayLike,
    sk_seed: ArrayLike,
    position: tree.TreePosition,
    height: int,
    leaf_index: int,
) -> Array:
    """`xmss_sign` — Algorithm 10. `[len + height, n]`: the WOTS+ signature, then
    the authentication path."""
    forest = leaves(tweak, params, pk_seed, sk_seed, position, height)
    path = tree.auth_path(
        tweak,
        pk_seed,
        forest,
        [leaf_index],
        height,
        tree.xmss_node_addresses(position),
    )[0]
    signature = wots.sign(
        tweak,
        params,
        message,
        pk_seed,
        sk_seed,
        wots.WotsPosition(position.layer, position.tree, leaf_index),
    )
    return np.concatenate([np.asarray(signature), np.asarray(path)])


def node_addresses(positions: Sequence[tree.TreePosition]) -> tree.NodeAddresses:
    """Node addresses for a batch whose entries sit in *different* trees.

    Entry `k`'s node is addressed in `positions[k]`'s tree. Only usable where the
    batch keeps one entry per position at every level — `root_from_path`, where a
    signature walks its own path — and not in a whole-tree reduction, where the
    node count halves each level and the correspondence would not hold.

    The prefix columns are built once rather than per level, since every level of a
    path addresses the same trees.
    """
    layers = np.array([position.layer for position in positions])
    trees = np.array([position.tree for position in positions])

    def build(height: int, indices: np.ndarray) -> np.ndarray:
        flat = np.asarray(indices).reshape(-1)
        if flat.shape[0] != len(positions):
            raise ValueError(
                f"one node per position: got {len(positions)} positions and "
                f"{flat.shape[0]} indices"
            )
        return adrs.encode_batch(
            adrs.hash_tree(layer=layers, tree=trees, height=height, index=flat),
            compressed=True,
        )

    return build


def pk_from_sig(
    tweak: TweakableHash,
    params: wots.WotsParams,
    signatures: ArrayLike,
    messages: ArrayLike,
    pk_seed: ArrayLike,
    positions: Sequence[tree.TreePosition],
    leaf_indices: ArrayLike,
) -> Array:
    """`xmss_pkFromSig` — Algorithm 11, for a batch. `[B, len + height, n]` -> `[B, n]`.

    Each entry carries its own tree, its own leaf index and its own message, which
    is what a hypertree layer hands it: `B` independent claims that each imply a
    root.
    """
    parts = np.asarray(signatures)
    if parts.ndim != 3 or parts.shape[0] != len(positions):
        raise ValueError(
            f"one signature per position: got shape {tuple(parts.shape)} for "
            f"{len(positions)} positions"
        )
    indices = np.asarray(leaf_indices, dtype=np.int64).reshape(-1)
    wots_positions = [
        wots.WotsPosition(position.layer, position.tree, int(index))
        for position, index in zip(positions, indices, strict=True)
    ]
    computed = wots.pk_from_sig(
        tweak,
        params,
        parts[:, : params.len, :],
        messages,
        pk_seed,
        wots_positions,
        wots.fips205_compression(tweak, pk_seed, wots_positions),
    )
    return tree.root_from_path(
        tweak,
        pk_seed,
        computed,
        indices,
        parts[:, params.len :, :],
        node_addresses(positions),
    )
