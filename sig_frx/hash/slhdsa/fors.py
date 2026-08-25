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

Signing takes one FORS key and verification takes a batch of them, the same
asymmetry WOTS+ and the XMSS layer carry: a signer holds one key, while a verifier
holds `B` claims whose digests each pick their own key. So `pk_from_sig` widens
over signatures and everything on the key-generation and signing path does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from sig_frx import arrays
from sig_frx.hash import adrs, adrs_encoding, bytestring, tree, wots
from sig_frx.hash.tweakable import TweakableHash, repeat_per_entry


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

    One key, or a batch of them: both fields are an integer or one per key, and
    they broadcast against each other the way an address's fields do. A verifier
    holds the batch, since every entry's digest picks its own FORS key.
    """

    tree: adrs.Field
    key_pair: adrs.Field

    @property
    def count(self) -> int:
        """How many FORS keys this is — one, or the length the fields broadcast to."""
        return adrs_encoding.rows((self.tree, self.key_pair))


def _node_addresses(position: ForsPosition, *, compressed: bool) -> tree.NodeAddresses:
    """The `FORS_TREE` addresses this forest's nodes tweak with.

    `compressed` is the parameter set's address encoding, as in
    `tree.xmss_node_addresses`.
    """

    def build(height: int, indices: ArrayLike) -> bytestring.ByteString:
        return adrs.encode_batch(
            adrs.fors_tree(
                layer=0,
                tree=position.tree,
                key_pair=position.key_pair,
                height=height,
                index=bytestring.index_column(indices),
            ),
            compressed=compressed,
        )

    return build


def _batch_node_addresses(
    position: ForsPosition, keys: int, *, compressed: bool
) -> tree.NodeAddresses:
    """Node addresses for a row batch whose rows sit in *different* FORS keys.

    Row `r`'s node is addressed in the forest `position`'s columns name for row
    `r`. Only usable where the batch keeps one row per key at every level —
    `root_from_path`, where each row walks its own path — and not in a
    whole-forest reduction, where the node count halves each level and the
    correspondence would not hold.

    `keys` is the row count rather than something read off `position`: a position
    whose fields are all single values covers every row by broadcasting, so it
    cannot say how many there are.
    """

    def build(height: int, indices: ArrayLike) -> bytestring.ByteString:
        flat = bytestring.index_column(indices)
        if flat.shape[0] != keys:
            raise ValueError(
                f"one node per FORS key: got {keys} keys and {flat.shape[0]} indices"
            )
        return adrs.encode_batch(
            adrs.fors_tree(
                layer=0,
                tree=position.tree,
                key_pair=position.key_pair,
                height=height,
                index=flat,
            ),
            compressed=compressed,
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
        adrs.fors_prf(
            layer=0,
            tree=position.tree,
            key_pair=position.key_pair,
            index=bytestring.index_column(indices),
        ),
        compressed=tweak.compressed_address,
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
    addresses = _node_addresses(position, compressed=tweak.compressed_address)
    return tweak.f(pk_seed, addresses(0, indices), secrets)


def message_indices(params: ForsParams, digest: ArrayLike) -> bytestring.ByteString:
    """Which leaf of each tree the digest picks — `base_2b(md, a, k)`, forest-wide.

    Algorithm 16 line 2 gives the within-tree index of each tree's chosen leaf;
    tree `i`'s offset `i·2^a` turns that into the forest-wide numbering
    everything else here uses.

    Rank-preserving: one digest gives `[k]`, and a batch of them gives `[B, k]` —
    which is what verifying a batch of signatures asks for, one digest per entry.

    Namespace-preserving too, and that is the load-bearing half: these index
    *addresses* rather than being hashed, so where they land decides where every
    address batch built from them is packed. `base_2b` reads its digits on the
    device whatever it is given, so a concrete digest gets them back — which is
    what keeps the signer's host-width tree address off a 32-bit lane.

    The offsets are the parameter set's own, so they are a host constant the
    digits broadcast against either way.
    """
    xnp = arrays.namespace(digest)
    within = xnp.asarray(wots.base_2b(digest, params.a, params.k))
    indices = within + np.arange(params.k, dtype=np.uint32) * np.uint32(params.t)
    return indices[0] if np.ndim(digest) == 1 else indices


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
        tweak,
        pk_seed,
        forest,
        indices,
        params.a,
        _node_addresses(position, compressed=tweak.compressed_address),
    )
    secrets = secret_values(tweak, pk_seed, sk_seed, position, indices)
    return fnp.concatenate([secrets[:, None, :], paths], axis=1)


def public_key(
    tweak: TweakableHash,
    pk_seed: ArrayLike,
    position: ForsPosition,
    roots: ArrayLike,
) -> Array:
    """`T_k` over each entry's `k` tree roots — Algorithm 17 lines 21 to 24.

    `[B, k, n]` -> `[B, n]`: one `T_k` per entry, all of them in one call.
    """
    addresses = adrs.encode_batch(
        adrs.fors_roots(layer=0, tree=position.tree, key_pair=position.key_pair),
        compressed=tweak.compressed_address,
    )
    stacked = fnp.asarray(roots, dtype=fnp.uint8).reshape(position.count, -1)
    return tweak.t(pk_seed, addresses, stacked)


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
        _node_addresses(position, compressed=tweak.compressed_address),
    )
    return public_key(tweak, pk_seed, position, roots)[0]


def pk_from_sig(
    tweak: TweakableHash,
    params: ForsParams,
    signatures: ArrayLike,
    digests: ArrayLike,
    pk_seed: ArrayLike,
    position: ForsPosition,
) -> Array:
    """`fors_pkFromSig` — Algorithm 17, for a batch. `[B, k, a + 1, n]` -> `[B, n]`.

    The operation verification actually runs, so it takes many signatures: each
    with its own digest, its own FORS key and — when the batch spans more than one
    public key — its own `pk_seed`. That is what an SLH-DSA verifier holds, since
    the digest picks the FORS key and every entry's digest is its own.

    All `B · k` trees walk their paths together: one hash call per level for every
    tree of every entry, `a` levels deep, then one `T_k` per entry over its roots.
    """
    parts = fnp.asarray(signatures, dtype=fnp.uint8)
    batch = position.count
    expected = (batch, params.k, params.a + 1, params.n)
    if parts.shape != expected:
        raise ValueError(
            f"a FORS signature batch is {expected}, got {tuple(parts.shape)}"
        )
    md = arrays.namespace(digests).asarray(digests)
    if md.ndim == 1:
        md = md[None, :]
    if md.shape[0] != batch:
        raise ValueError(
            f"one digest per FORS key: got {batch} keys, {md.shape[0]} digests"
        )

    # Every entry's `k` trees are rows of one batch, entry-major, so the position
    # and the seed both repeat `k` times to line up with them.
    rows = batch * params.k
    per_tree = ForsPosition(
        tree=adrs_encoding.repeat_rows(position.tree, params.k),
        key_pair=adrs_encoding.repeat_rows(position.key_pair, params.k),
    )
    node_addresses = _batch_node_addresses(
        per_tree, rows, compressed=tweak.compressed_address
    )
    seeds = repeat_per_entry(pk_seed, params.k)
    indices = message_indices(params, md).reshape(-1)
    leaf_nodes = tweak.f(
        seeds,
        node_addresses(0, indices),
        parts[:, :, 0, :].reshape(rows, params.n),
    )
    roots = tree.root_from_path(
        tweak,
        seeds,
        leaf_nodes,
        indices,
        parts[:, :, 1:, :].reshape(rows, params.a, params.n),
        node_addresses,
    )
    return public_key(
        tweak, pk_seed, position, roots.reshape(batch, params.k, params.n)
    )
