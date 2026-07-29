# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The SLH-DSA hypertree — `d` layers of XMSS trees. FIPS 205 §7.

A single XMSS key cannot hold the number of WOTS+ keys SLH-DSA needs, so the
trees are stacked: layer 0 signs the FORS public key, and each layer above signs
the root of the tree below it. Only the top layer's root is published, which is
what makes the whole structure one public key.

The index says where. `idx_tree` picks the layer-0 tree and `idx_leaf` the key
pair within it; climbing a layer consumes `h'` bits — the leaf index at the next
layer up is the low `h'` bits of the tree index, and the tree index shifts right
by `h'`.

**Verification batches across signatures, layer by layer.** `verify` takes `B`
claims and walks all of them up together: at each layer every entry sits in its
own tree with its own leaf index, so a layer is one batched `xmss_pkFromSig`
rather than `B` of them. `d` layers deep, that is `d` batched passes instead of
`B · d` sequential ones — and the per-entry verdict survives, because a claim
that reaches the wrong root at any layer reaches the wrong root at the top.
"""

from __future__ import annotations

from dataclasses import dataclass

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from sig_frx.hashbased import tree, wots, xmss
from sig_frx.hashbased.tweakable import TweakableHash


@dataclass(frozen=True)
class HypertreeParams:
    """`d` layers of XMSS trees of height `h'` — FIPS 205 §7."""

    layers: int
    tree_height: int
    wots: wots.WotsParams

    @property
    def total_height(self) -> int:
        """`h` — the number of index bits the whole hypertree addresses."""
        return self.layers * self.tree_height

    @property
    def signature_values(self) -> int:
        """`n`-byte values in one layer's XMSS signature."""
        return self.wots.len + self.tree_height


def _climb(tree_index: int, height: int) -> tuple[int, int]:
    """One layer up: the leaf index it lands on, and the tree index above it."""
    return tree_index % (1 << height), tree_index >> height


def sign(
    tweak: TweakableHash,
    params: HypertreeParams,
    message: ArrayLike,
    pk_seed: ArrayLike,
    sk_seed: ArrayLike,
    tree_index: int,
    leaf_index: int,
) -> Array:
    """`ht_sign` — Algorithm 12. `[d, len + h', n]`, lowest layer first.

    Each layer signs the root of the one below it, so the layers are sequential by
    construction — this is the signing path, and signing is not what this repo
    batches (see `docs/reference/security.md`).
    """
    signatures = []
    current = np.asarray(message)
    current_tree, current_leaf = tree_index, leaf_index
    for layer in range(params.layers):
        position = tree.TreePosition(layer=layer, tree=current_tree)
        layer_signature = xmss.sign(
            tweak,
            params.wots,
            current,
            pk_seed,
            sk_seed,
            position,
            params.tree_height,
            current_leaf,
        )
        signatures.append(np.asarray(layer_signature))
        if layer + 1 < params.layers:
            current = np.asarray(
                xmss.pk_from_sig(
                    tweak,
                    params.wots,
                    layer_signature[None, ...],
                    current[None, :],
                    pk_seed,
                    [position],
                    [current_leaf],
                )
            )[0]
            current_leaf, current_tree = _climb(current_tree, params.tree_height)
    return fnp.asarray(np.stack(signatures))


def roots_from_sig(
    tweak: TweakableHash,
    params: HypertreeParams,
    signatures: ArrayLike,
    messages: ArrayLike,
    pk_seed: ArrayLike,
    tree_indices: ArrayLike,
    leaf_indices: ArrayLike,
) -> Array:
    """The top-layer root each claim implies — Algorithm 13 lines 1 to 12, batched.

    `signatures` is `[B, d, len + h', n]`; the result is `[B, n]`, which the caller
    compares against the public key's root.
    """
    parts = np.asarray(signatures)
    batch = parts.shape[0]
    expected = (batch, params.layers, params.signature_values, params.wots.n)
    if parts.shape != expected:
        raise ValueError(
            f"a hypertree signature batch is {expected}, got {tuple(parts.shape)}"
        )
    trees = np.asarray(tree_indices, dtype=np.int64).reshape(-1).copy()
    leaves = np.asarray(leaf_indices, dtype=np.int64).reshape(-1).copy()
    if len(trees) != batch or len(leaves) != batch:
        raise ValueError(
            f"one tree index and one leaf index per signature: got {batch} "
            f"signatures, {len(trees)} tree indices, {len(leaves)} leaf indices"
        )

    nodes = np.asarray(messages)
    for layer in range(params.layers):
        positions = [tree.TreePosition(layer=layer, tree=int(index)) for index in trees]
        nodes = np.asarray(
            xmss.pk_from_sig(
                tweak,
                params.wots,
                parts[:, layer, :, :],
                nodes,
                pk_seed,
                positions,
                leaves,
            )
        )
        if layer + 1 < params.layers:
            climbed = [_climb(int(index), params.tree_height) for index in trees]
            leaves = np.array([leaf for leaf, _ in climbed], dtype=np.int64)
            trees = np.array([above for _, above in climbed], dtype=np.int64)
    return fnp.asarray(nodes)


def verify(
    tweak: TweakableHash,
    params: HypertreeParams,
    signatures: ArrayLike,
    messages: ArrayLike,
    pk_seed: ArrayLike,
    tree_indices: ArrayLike,
    leaf_indices: ArrayLike,
    pk_root: ArrayLike,
) -> Array:
    """`ht_verify` — Algorithm 13. `bool[B]`, one verdict per signature."""
    roots = roots_from_sig(
        tweak, params, signatures, messages, pk_seed, tree_indices, leaf_indices
    )
    expected = fnp.asarray(pk_root, dtype=fnp.uint8)
    if expected.ndim == 1:
        expected = fnp.broadcast_to(expected, roots.shape)
    return fnp.all(roots == expected, axis=-1)
