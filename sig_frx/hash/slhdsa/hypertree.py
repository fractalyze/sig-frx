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

from sig_frx import arrays
from sig_frx.hash import bytestring, tree, wots
from sig_frx.hash.slhdsa import xmss
from sig_frx.hash.tweakable import TweakableHash


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
    def tree_index_bytes(self) -> int:
        """How wide the index a walk starts from is, in bytes.

        `h − h'` bits, byte-rounded: what `bytestring` carries between layers, and
        what a caller has to encode an index to before handing it over.
        """
        return -(-(self.total_height - self.tree_height) // 8)

    @property
    def signature_values(self) -> int:
        """`n`-byte values in one layer's XMSS signature."""
        return self.wots.len + self.tree_height


def _climb(tree_index: int, height: int) -> tuple[int, int]:
    """One layer up: the leaf index it lands on, and the tree index above it.

    The signer's form. Signing is one signature on the host, where a Python
    integer holds the index at any width — the one representation that never
    truncates. A verifier climbs a batch and takes `_climb_batch` instead.
    """
    return tree_index % (1 << height), tree_index >> height


def _climb_batch(
    tree_indices: bytestring.ByteString, height: int
) -> tuple[Array, bytestring.ByteString]:
    """The same step for a whole batch, over indices carried as bytes.

    The leaf index comes off the bottom and the tree index above is what is left.
    See `bytestring` for why the one stays bytes and the other does not.
    """
    return (
        bytestring.low_bits(tree_indices, height),
        bytestring.shift_right(tree_indices, height),
    )


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
                    position,
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
    tree_indices: bytestring.ByteString,
    leaf_indices: ArrayLike,
) -> Array:
    """The top-layer root each claim implies — Algorithm 13 lines 1 to 12, batched.

    `signatures` is `[B, d, len + h', n]`; the result is `[B, n]`, which the caller
    compares against the public key's root.

    `tree_indices` is `[B, tree_index_bytes]` bytes rather than a column of
    numbers — see `bytestring` — while `leaf_indices` is a column, since `h'` bits
    fit one.

    Nothing here comes back to the host between layers: each layer's roots are the
    next layer's messages, and the climb is a static schedule of shifts over the
    index. So the whole walk is one expression whether the batch is concrete or
    traced.
    """
    parts = arrays.namespace(signatures).asarray(signatures)
    batch = parts.shape[0]
    expected = (batch, params.layers, params.signature_values, params.wots.n)
    if parts.shape != expected:
        raise ValueError(
            f"a hypertree signature batch is {expected}, got {tuple(parts.shape)}"
        )
    trees, leaves = tree_indices, leaf_indices
    if trees.shape[0] != batch or leaves.shape[0] != batch:
        raise ValueError(
            f"one tree index and one leaf index per signature: got {batch} "
            f"signatures, {trees.shape[0]} tree indices, {leaves.shape[0]} leaf "
            f"indices"
        )

    nodes = messages
    for layer in range(params.layers):
        nodes = xmss.pk_from_sig(
            tweak,
            params.wots,
            parts[:, layer, :, :],
            nodes,
            pk_seed,
            tree.TreePosition(layer=layer, tree=trees),
            leaves,
        )
        if layer + 1 < params.layers:
            leaves, trees = _climb_batch(trees, params.tree_height)
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
