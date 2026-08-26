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
nodes are a tweaked hash the family supplies. The two share the word and nothing
else.

**Nothing here is about bytes.** The walk pairs nodes, selects a side and hashes,
and none of those is a byte operation — so the arrays are built at the family's
own `dtype` rather than at `uint8`, which is what lets leanSig climb the same
tree over KoalaBear digests ([`leansig/tweakable.py`](leansig/tweakable.py)).
The one operand that is *not* the digest dtype is the left/right condition, which
is a boolean at every family.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from sig_frx.hash import adrs, adrs_encoding, bytestring
from sig_frx.hash.tweakable import NodeHash

# What a builder hands the family as its tweak. Deliberately not
# `bytestring.ByteString`: FIPS 205's and RFC 8391's builders encode an address
# into bytes, and leanSig's packs a level and an index into field elements, so
# the element type is the family's business and naming one here would be a lie
# in the third caller.
Tweaks: TypeAlias = np.ndarray | Array

# Builds the tweaks of the given nodes at one height: `(height, indices)`. The
# indices are concrete where a whole tree is built and traced where a batch walks
# its paths, so a builder takes them as they come rather than on the host.
NodeAddresses = Callable[[int, ArrayLike], Tweaks]


@dataclass(frozen=True)
class TreePosition:
    """Which XMSS tree this is — the address prefix every node in it tweaks with.

    One tree, or a batch of them: both fields are an integer or one per tree, and
    they broadcast against each other the way an address's fields do. The batch is
    what a hypertree layer holds — `B` claims that each climb their own tree — so
    the layer carries three columns rather than `B` objects.
    """

    layer: adrs.Field
    tree: adrs.Field

    @property
    def count(self) -> int:
        """How many trees this is — one, or the length the fields broadcast to."""
        return adrs_encoding.rows((self.layer, self.tree))


def xmss_node_addresses(position: TreePosition, *, compressed: bool) -> NodeAddresses:
    """The `TREE`-type addresses an XMSS tree's nodes tweak with.

    `compressed` is the parameter set's address encoding — the SHA-2 sets tweak
    with the 22-byte `ADRS^c` and the SHAKE sets with the full 32 — which a
    caller reads off its family as `TweakableHash.compressed_address`. It is the
    builder's parameter rather than the walk's: the walk below is shared with
    RFC 8391, whose addresses are neither of FIPS 205's two encodings, so it
    takes a builder and never learns what one encodes.
    """

    def build(height: int, indices: ArrayLike) -> Tweaks:
        return adrs.encode_batch(
            adrs.hash_tree(
                layer=position.layer,
                tree=position.tree,
                height=height,
                index=bytestring.index_column(indices),
            ),
            compressed=compressed,
        )

    return build


def reduce_levels(
    tweak: NodeHash,
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
    current = fnp.asarray(nodes, dtype=tweak.dtype)
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
    tweak: NodeHash,
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
    tweak: NodeHash,
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
    nodes = fnp.asarray(leaves, dtype=tweak.dtype)
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

    def build(height: int, indices: ArrayLike) -> Tweaks:
        return node_addresses(height + by, indices)

    return build


def root_from_path(
    tweak: NodeHash,
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

    The index stays where it arrives — see `bytestring.index_column`. Every use it
    is put to is a shift and a mask against a static level, so the walk is the same
    expression over a concrete index and a traced one.
    """
    nodes = fnp.asarray(leaves, dtype=tweak.dtype)
    sibling_paths = fnp.asarray(paths, dtype=tweak.dtype)
    leaf_indices = bytestring.index_column(indices)
    if sibling_paths.shape[0] != nodes.shape[0] or len(leaf_indices) != nodes.shape[0]:
        raise ValueError(
            f"one index and one path per leaf: got {nodes.shape[0]} leaves, "
            f"{len(leaf_indices)} indices, {sibling_paths.shape[0]} paths"
        )

    for level in range(sibling_paths.shape[1]):
        siblings = sibling_paths[:, level, :]
        on_the_right = fnp.asarray((leaf_indices >> level) & 1, dtype=bool)[:, None]
        left = fnp.where(on_the_right, siblings, nodes)
        right = fnp.where(on_the_right, nodes, siblings)
        nodes = tweak.h(
            pk_seed,
            node_addresses(level + 1, leaf_indices >> (level + 1)),
            fnp.concatenate([left, right], axis=-1),
        )
    return nodes
