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

import numpy as np
from frx import Array
from frx.typing import ArrayLike

from sig_frx.hashbased import adrs, bytestring, tree, wots
from sig_frx.hashbased.tweakable import TweakableHash


def key_pairs(position: tree.TreePosition, height: int) -> wots.WotsPosition:
    """Every WOTS+ key pair under one XMSS tree, leaf order.

    One position whose key pair number runs over the tree, not `2^height` of them:
    the layer and the tree are shared, so the batch is a column of leaf numbers
    against two constants.
    """
    return wots.WotsPosition(
        layer=position.layer, tree=position.tree, key_pair=np.arange(1 << height)
    )


def leaves(
    tweak: TweakableHash,
    params: wots.WotsParams,
    pk_seed: ArrayLike,
    sk_seed: ArrayLike,
    position: tree.TreePosition,
    height: int,
) -> Array:
    """Every leaf of one XMSS tree — `[2^height, n]`, one batched WOTS+ walk."""
    under = key_pairs(position, height)
    return wots.pk_gen(
        tweak,
        params,
        pk_seed,
        sk_seed,
        under,
        wots.fips205_compression(tweak, pk_seed, under),
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
        tree.xmss_node_addresses(position, compressed=tweak.compressed_address),
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
        tree.xmss_node_addresses(position, compressed=tweak.compressed_address),
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


def node_addresses(
    position: tree.TreePosition, *, compressed: bool
) -> tree.NodeAddresses:
    """Node addresses for a batch whose entries sit in *different* trees.

    Entry `k`'s node is addressed in the tree `position`'s columns name for row
    `k`. Only usable where the batch keeps one entry per tree at every level —
    `root_from_path`, where a signature walks its own path — and not in a
    whole-tree reduction, where the node count halves each level and the
    correspondence would not hold.

    `compressed` is the parameter set's address encoding, as in
    `tree.xmss_node_addresses`.
    """
    count = position.count

    def build(height: int, indices: ArrayLike) -> bytestring.ByteString:
        flat = bytestring.index_column(indices)
        if flat.shape[0] != count:
            raise ValueError(
                f"one node per tree: got {count} trees and {flat.shape[0]} indices"
            )
        return adrs.encode_batch(
            adrs.hash_tree(
                layer=position.layer, tree=position.tree, height=height, index=flat
            ),
            compressed=compressed,
        )

    return build


def pk_from_sig(
    tweak: TweakableHash,
    params: wots.WotsParams,
    signatures: ArrayLike,
    messages: ArrayLike,
    pk_seed: ArrayLike,
    position: tree.TreePosition,
    leaf_indices: ArrayLike,
) -> Array:
    """`xmss_pkFromSig` — Algorithm 11, for a batch. `[B, len + height, n]` -> `[B, n]`.

    Each entry carries its own tree, its own leaf index and its own message, which
    is what a hypertree layer hands it: `B` independent claims that each imply a
    root.

    The signature and the leaf index stay in the namespace they arrive in: a
    hypertree walk hands over a slice of what it was given, which is concrete when
    a signer climbs its own layers and traced when a verifier walks a batch.
    """
    parts = bytestring.namespace(signatures).asarray(signatures)
    count = position.count
    if parts.ndim != 3 or parts.shape[0] != count:
        raise ValueError(
            f"one signature per tree: got shape {tuple(parts.shape)} for "
            f"{count} trees"
        )
    indices = bytestring.index_column(leaf_indices)
    # The leaf index is which key pair of that tree signed, so it is the batch's
    # key pair column against the tree's own layer and tree columns.
    signing_keys = wots.WotsPosition(position.layer, position.tree, indices)
    computed = wots.pk_from_sig(
        tweak,
        params,
        parts[:, : params.len, :],
        messages,
        pk_seed,
        signing_keys,
        wots.fips205_compression(tweak, pk_seed, signing_keys),
    )
    return tree.root_from_path(
        tweak,
        pk_seed,
        computed,
        indices,
        parts[:, params.len :, :],
        node_addresses(position, compressed=tweak.compressed_address),
    )
