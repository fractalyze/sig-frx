# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""FXMSS — the flexible XMSS tree SHRINCS's stateful path signs under.

An XMSS tree whose shape the signer chooses: an unbalanced tree makes the first
signatures tiny and each one after it larger, a balanced tree makes them all the
same size, and a signer picks whichever suits how often it signs.

**The verifier is agnostic to that choice**, which the specification says
outright and which is the whole reason `root_from_sig` is short. A signature
carries the height and index of the leaf that made it, so the verifier walks from
that leaf to the root and never learns what shape the rest of the tree had —
`root_from_sig` takes no shape and no depth.

**The signer is the half that knows.** `Structure` is the two bytes the secret
key carries, and everything below the verification walk reads them: which leaves
exist, which one a counter names, and how the tree above them closes. The two
prescribed shapes are different trees rather than one parameterized tree, so they
climb separately — a balanced one is a Merkle forest and reuses
[`../tree.py`](../tree.py); an unbalanced one is a left spine of nodes each
combined with a single right-hand leaf, which no forest walk describes.

**Two things about the walk are not `../tree.py`'s**, which is why it is written
out here rather than reached for:

- **The leaf index is 64 bits.** `tree.root_from_path` takes it as a uint32
  column, which is right for a hypertree layer of at most nine bits and wrong
  here by thirty-two — and wrong silently, since a lane truncates without
  raising. It stays a byte string end to end (see
  [`../bytestring.py`](../bytestring.py)), and the bit that picks a side is
  read out of the byte holding it rather than by shifting a number.
- **The depth varies per entry.** A batch holds signatures made at different
  heights, so the walk runs to `FXMSS_HEIGHT` for everyone and each entry stops
  taking its parents once it has reached its own root. That is the same masking
  `wots.chain` does for a chain that has finished, and for the same reason: the
  alternative is a trip count that depends on the signature.

The cost of the second one is that every verification pays the tallest tree the
format allows, whatever the signature in hand actually needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from sig_frx.hash import bytestring, tree
from sig_frx.hash.shrincs import adrs as sf_adrs
from sig_frx.hash.shrincs import wots_c
from sig_frx.hash.tweakable import TweakableHash

# `FXMSS_HEIGHT` — the imaginary height of the tree, and so the deepest a leaf
# can sit. It is 255 because a node height is one byte of the address and of the
# signature, which is the same reason the walk below is 255 steps.
HEIGHT = 255

_N = 16
# The index field is as many whole bytes as the depth needs, capped at eight:
# nothing beyond 64 bits is addressable, which is what the tree slot holds. This
# module owns the number because it owns the format the signature is in.
INDEX_BYTES = 8

# `FXMSS_SIGNATURE_SIZE_MIN` and `_MAX`: a WOTS+C signature and one node per step.
SIGNATURE_SIZE_MIN = wots_c.SIGNATURE_SIZE + _N
SIGNATURE_SIZE_MAX = wots_c.SIGNATURE_SIZE + HEIGHT * _N


# The two shapes the specification prescribes, as its structure byte spells them.
SHAPE_UNBALANCED = 0
SHAPE_BALANCED = 1

# A shape and a depth, one byte each. This module owns the width for the reason it
# owns `INDEX_BYTES`: it owns the format those bytes are in. `wots_c` only carries
# them into an address and never reads them.
STRUCTURE_BYTES = 2

# Key generation builds every leaf the shape names, and a balanced tree of depth
# `d` names `2^d` of them — so a structure from an untrusted source is a denial of
# service. The specification warns about exactly that and its reference
# implementation hangs; this refuses, at a bound past anything a signer would
# choose, since a million leaves is already thirty million chain walks.
MAX_LEAVES = 2**20


@dataclass(frozen=True)
class Structure:
    """The two bytes of the secret key that say what tree the signer built.

    The verifier never sees them, which is why nothing on the verification path
    above takes one. What they decide is which leaves exist and which one a
    counter names — the leaf schedule, which is the whole of what a stateful
    signer must not get wrong twice.
    """

    shape: int
    depth: int

    @classmethod
    def parse(cls, sf_structure: ArrayLike) -> Structure:
        """The two bytes, checked. A shape this does not know is refused.

        Refused rather than defaulted: the reference returns "no leaf" for an
        unknown shape, which sends a signer that meant to sign statefully down the
        stateless path with a five-times-longer signature and no complaint.
        """
        values = np.asarray(sf_structure, dtype=np.uint8).reshape(-1)
        if values.shape != (STRUCTURE_BYTES,):
            raise ValueError(
                f"a tree structure is {STRUCTURE_BYTES} bytes — a shape and a "
                f"depth — got shape {tuple(values.shape)}"
            )
        structure = cls(shape=int(values[0]), depth=int(values[1]))
        if structure.shape not in (SHAPE_UNBALANCED, SHAPE_BALANCED):
            raise ValueError(
                f"the prescribed FXMSS shapes are {SHAPE_UNBALANCED} (unbalanced) "
                f"and {SHAPE_BALANCED} (balanced), got {structure.shape}"
            )
        if structure.leaves_built > MAX_LEAVES:
            raise ValueError(
                f"a balanced tree of depth {structure.depth} has "
                f"{structure.leaves_built} leaves and key generation builds every "
                f"one; this refuses above {MAX_LEAVES}"
            )
        return structure

    @property
    def leaves_built(self) -> int:
        """How many WOTS+C leaves key generation makes: `2^d` balanced, `d + 1` not.

        One at depth zero either way — the tree is a single leaf standing where
        the root goes.
        """
        if self.shape == SHAPE_BALANCED:
            return 2**self.depth
        return self.depth + 1

    @property
    def leaf_count(self) -> int:
        """How many stateful signatures this key has — `leaves_built`, or none.

        A depth-zero tree signs nothing, though it still has the one leaf above:
        the indicator byte names a leaf by its height, and 255 is the value that
        means "stateless", so a leaf at the root's height cannot be named. Such a
        key exists to carry an `sf_root` for the stateless path to bind to.
        """
        return 0 if self.depth == 0 else self.leaves_built

    @property
    def encoded(self) -> np.ndarray:
        """The two bytes back — what the key carries and the PRF addresses with."""
        return np.array([self.shape, self.depth], dtype=np.uint8)

    @property
    def bottom(self) -> int:
        """The height the deepest leaves sit at."""
        return HEIGHT - self.depth

    def holds(self, leaf_index: int, leaf_height: int) -> bool:
        """Whether a WOTS+C leaf of this tree can sign from that position.

        Can sign, not merely exists. The two readings part at depth zero, where
        the tree is one leaf standing at the root's height that no counter names,
        and `sign` wants this one — a position `leaf` will never hand out is not a
        position a signature may be made at.

        Asked rather than looked up: a balanced tree of depth 20 has a million
        leaves, and a signer should not walk them to find out that the one it was
        handed is among them.
        """
        if self.leaf_count == 0:
            return False
        if leaf_height == self.bottom and leaf_index == 0:
            return True
        if self.shape == SHAPE_BALANCED:
            return leaf_height == self.bottom and 0 < leaf_index < self.leaves_built
        return leaf_index == 1 and self.bottom <= leaf_height < HEIGHT

    def leaf(self, counter: int) -> tuple[int, int]:
        """`shrincs_sf_leaf_select` — the (index, height) a counter names.

        A counter the tree cannot hold raises. The reference returns "no leaf" and
        its caller falls back to the stateless path; here that fallback is
        something the caller asks for by passing no counter at all, so reaching
        this with a spent counter is a signer that has lost count rather than one
        that meant to fall back.
        """
        if counter < 0 or counter >= self.leaf_count:
            raise ValueError(
                f"this tree holds {self.leaf_count} WOTS+C leaves and the counter "
                f"names number {counter}; a leaf that signs twice reveals its "
                f"secret, so there is no leaf left. Pass no counter to sign on "
                f"the stateless path instead"
            )
        if self.shape == SHAPE_BALANCED:
            return counter, self.bottom
        # The unbalanced tree spends its right-hand leaves top down and finishes
        # on the one left-hand leaf at the bottom, which is why its first
        # signature is the shortest the format allows.
        if counter == self.depth:
            return 0, self.bottom
        return 1, HEIGHT - 1 - counter


def index_field_bytes(leaf_depth: int) -> int:
    """`ceil(min(leaf_depth, 64) / 8)` — the index field's width, 1 to 8 bytes.

    A host function, because the width is what a *parser* needs and a parser has
    the indicator byte concretely: the field's width decides where the FXMSS
    signature starts, so it cannot itself be read out of the signature.
    """
    return -(-min(leaf_depth, 8 * INDEX_BYTES) // 8)


def root_from_sig(
    tweak: TweakableHash,
    pk_seed: ArrayLike,
    signatures: ArrayLike,
    message_digests: ArrayLike,
    leaf_heights: ArrayLike,
    leaf_indices: ArrayLike,
) -> tuple[Array, Array]:
    """`fxmss_pubkey_from_sig` — the root each signature implies, for a batch.

    `signatures` is `[B, SIGNATURE_SIZE_MAX]`, zero-padded past each entry's own
    length; `leaf_indices` is `[B, 8]` bytes and `leaf_heights` a `[B]` column.
    The result is the `[B, 16]` root and a `bool[B]` carrying WOTS+C's verdict on
    the grinding counter, which is the one rejection this path can make before
    the root comparison.
    """
    parts = fnp.asarray(signatures, dtype=fnp.uint8)
    if parts.ndim != 2 or parts.shape[1] != SIGNATURE_SIZE_MAX:
        raise ValueError(
            f"an FXMSS signature batch is [B, {SIGNATURE_SIZE_MAX}], got shape "
            f"{tuple(parts.shape)}"
        )
    batch = parts.shape[0]
    indices = fnp.asarray(leaf_indices, dtype=fnp.uint8)
    if indices.ndim != 2 or indices.shape[1] != INDEX_BYTES:
        raise ValueError(
            f"a leaf index batch is [B, {INDEX_BYTES}] bytes, got shape "
            f"{tuple(indices.shape)}"
        )
    heights = fnp.asarray(leaf_heights, dtype=fnp.uint32)

    nodes, accepted = wots_c.pk_from_sig(
        tweak,
        pk_seed,
        parts[:, : wots_c.SIGNATURE_SIZE],
        message_digests,
        heights,
        indices,
    )
    path = parts[:, wots_c.SIGNATURE_SIZE :].reshape(batch, HEIGHT, _N)
    # A leaf at height `p` is `HEIGHT - p` steps from the root, so the height is
    # the depth's complement and the walk needs no second argument for it.
    depths = np.uint32(HEIGHT) - heights

    # `parents` carries the index shifted right by the steps already run, so at
    # the top of a step it is `index >> step`: its low bit is the side this node
    # falls on, and one more shift makes it the parent's own index.
    parents = indices
    for step in range(HEIGHT):
        siblings = path[:, step, :]
        # The low bit of the running shift, not bit `step` of the original. Reading
        # the original means indexing the byte that holds bit `step`, which runs off
        # the front of an eight-byte index once `step` reaches 64 — and a tree at
        # the format's full height has 191 steps past that, every one of which must
        # fall left because an index has no bits up there. The shift feeds in zeros
        # and gives that for free.
        on_the_right = (parents[:, -1] & np.uint8(1))[:, None]
        left = fnp.where(on_the_right, siblings, nodes)
        right = fnp.where(on_the_right, nodes, siblings)
        parents = bytestring.shift_right(parents, 1)
        # Clamped because a masked step can address above the root: a leaf at
        # height 254 has one real step and 254 discarded ones, whose heights would
        # run past the byte the slot gives them. This walk is traced, and
        # `adrs_encoding` can only width-check a concrete field — so the overflow
        # would wrap into the slot silently rather than raise, which is why the
        # clamp is here and not left to the encoder to catch.
        parent_heights = fnp.minimum(heights + (step + 1), np.uint32(HEIGHT))
        combined = tweak.h(
            pk_seed,
            sf_adrs.encode_batch(sf_adrs.fxmss_tree(parent_heights, parents)),
            fnp.concatenate([left, right], axis=-1),
        )
        nodes = fnp.where((step < depths)[:, None], combined, nodes)
    return nodes, accepted


# -- the signer --------------------------------------------------------------


def root(
    tweak: TweakableHash,
    pk_seed: ArrayLike,
    sk_seed: ArrayLike,
    structure: Structure,
) -> Array:
    """`fxmss_node` at the top — the `[16]` value a public key's third part is.

    The one place the tree's shape reaches anything, and the reason the structure
    bytes ride in the secret key: every leaf the shape names is built here, and
    the root is what the whole of it reduces to.

    A depth-zero tree has a root like any other — its single leaf — even though it
    can sign nothing. Refusing to generate that key would refuse a stateless-only
    SHRINCS key, which is a thing the specification has and a public key still
    needs a third part for.
    """
    leaves = _leaves(tweak, pk_seed, sk_seed, structure)
    if structure.shape == SHAPE_BALANCED:
        return tree.root(tweak, pk_seed, leaves, _node_addresses(structure.bottom))
    return _spine(tweak, pk_seed, leaves, structure, structure.depth)


def sign(
    tweak: TweakableHash,
    pk_seed: ArrayLike,
    sk_seed: ArrayLike,
    structure: Structure,
    message_digest: ArrayLike,
    leaf_height: int,
    leaf_index: int,
) -> Array:
    """`fxmss_sign` — the WOTS+C signature, then the path to the root.

    `[2 + 512 + 16·depth]`, the length the leaf's own depth fixes; the seam pads
    it, and the indicator byte is what lets a verifier derive it back.

    The leaf is passed rather than derived because its position also tweaks the
    digest being signed — `H_msg_sf` binds it — so the caller has already chosen
    it by the time there is a digest to hand over.
    """
    depth = HEIGHT - leaf_height
    if not structure.holds(leaf_index, leaf_height):
        raise ValueError(
            f"no leaf of this tree sits at height {leaf_height} index "
            f"{leaf_index}; a WOTS+C signature made anywhere else recovers a node "
            f"the root was not built from"
        )
    leaves = _leaves(tweak, pk_seed, sk_seed, structure)
    index = index_bytes(np.array([leaf_index], dtype=np.uint64))
    heights = np.array([leaf_height], dtype=np.uint32)
    return fnp.concatenate(
        [
            wots_c.sign(
                tweak,
                pk_seed,
                sk_seed,
                structure.encoded,
                message_digest,
                heights,
                index,
            ),
            _auth_path(
                tweak, pk_seed, leaves, structure, leaf_height, leaf_index
            ).reshape(depth * _N),
        ]
    )


def _leaf_positions(structure: Structure) -> tuple[np.ndarray, np.ndarray]:
    """Every leaf's height and index, in the order the leaves are built.

    Balanced: `2^d` leaves side by side at one height. Unbalanced: the single
    left-hand leaf at the bottom, then one right-hand leaf at each height from
    the bottom up — which is what makes each signature one node longer than the
    last, and the first the shortest the format allows. At depth zero both come
    to the same one leaf standing at the root's height.
    """
    if structure.shape == SHAPE_BALANCED:
        return (
            np.full(structure.leaves_built, structure.bottom, dtype=np.uint32),
            np.arange(structure.leaves_built, dtype=np.uint64),
        )
    return (
        np.array([structure.bottom, *range(structure.bottom, HEIGHT)], dtype=np.uint32),
        np.array([0, *([1] * structure.depth)], dtype=np.uint64),
    )


def _leaves(
    tweak: TweakableHash,
    pk_seed: ArrayLike,
    sk_seed: ArrayLike,
    structure: Structure,
) -> Array:
    """Every WOTS+C leaf the shape names, in `_leaf_positions`' order. `[L, 16]`.

    One batched call, whatever the shape: no leaf depends on another, so a tree of
    `L` leaves costs the `w − 1` hashes one leaf does.
    """
    heights, indices = _leaf_positions(structure)
    return wots_c.public_key(
        tweak, pk_seed, sk_seed, structure.encoded, heights, index_bytes(indices)
    )


def _node_addresses(leaf_height: int) -> tree.NodeAddresses:
    """`tree.py`'s node addresses, in FXMSS's numbering.

    `tree.py` counts a level up from the leaves and FXMSS names a node by its
    height in the whole 255-high format, so the two differ by where this tree's
    leaves sit. Only the balanced shape goes through it — an unbalanced tree is
    not a forest, and `_spine` walks it instead.
    """

    def build(height: int, indices: ArrayLike) -> bytestring.ByteString:
        return sf_adrs.encode_batch(sf_adrs.fxmss_tree(leaf_height + height, indices))

    return build


def _spine(
    tweak: TweakableHash,
    pk_seed: ArrayLike,
    leaves: Array,
    structure: Structure,
    levels: int,
) -> Array:
    """An unbalanced tree's left-hand node `levels` steps up. `[16]`.

    The node at index zero and height `bottom + levels`: at zero steps it is the
    one left-hand leaf, and at `depth` steps it is the root. Each step combines the
    node below with the right-hand leaf at that lower height, which is the whole of
    the shape — every level adds one leaf rather than doubling.

    Bounded rather than built whole because the two callers want different nodes
    of it and neither wants them all: a root is the top one and an authentication
    path is the single node below the leaf that signed. A signature made at
    counter `c` sits `depth - 1 - c` steps up, so walking to the top every time
    would spend more of the climb on nodes nothing reads than on nodes something
    does.
    """
    node = leaves[:1]
    for step in range(levels):
        node = tweak.h(
            pk_seed,
            sf_adrs.encode_batch(sf_adrs.fxmss_tree(structure.bottom + step + 1, 0)),
            fnp.concatenate([node, leaves[1 + step : 2 + step]], axis=-1),
        )
    # A batch of one throughout, because that is what `tweak.h` takes; the caller
    # wants the node, not a batch it has to unwrap.
    return node[0]


def _auth_path(
    tweak: TweakableHash,
    pk_seed: ArrayLike,
    leaves: Array,
    structure: Structure,
    leaf_height: int,
    leaf_index: int,
) -> Array:
    """The siblings the verifier's walk consumes, lowest first. `[depth, 16]`."""
    if structure.shape == SHAPE_BALANCED:
        return tree.auth_path(
            tweak,
            pk_seed,
            leaves,
            np.array([leaf_index]),
            structure.depth,
            _node_addresses(structure.bottom),
        )[0]
    # Above its own height every sibling is a right-hand leaf, because an
    # unbalanced tree has nothing else up there: the index shifts to zero after
    # one step and stays there, so `(index >> j) ^ 1` is one for every `j ≥ 1`.
    rights = leaves[1:]
    row = leaf_height - structure.bottom
    below = (
        _spine(tweak, pk_seed, leaves, structure, row)
        if leaf_index == 1
        else rights[row]
    )
    return fnp.concatenate([below[None, :], rights[row + 1 :]], axis=0)


def index_bytes(values: np.ndarray) -> np.ndarray:
    """Node indices as `[rows, INDEX_BYTES]` big-endian bytes.

    Public because a signature carries the same bytes: `shrincs.py` writes the
    low end of one into the leaf-index field, and the whole of it into the address
    that binds the digest.

    A tree is built on the host, so these are concrete — but the index still
    crosses into an address as bytes rather than as a number, because the slot is
    eight bytes wide and an array lane is four (`../bytestring.py`).
    """
    return bytestring.big_endian(values, INDEX_BYTES)
