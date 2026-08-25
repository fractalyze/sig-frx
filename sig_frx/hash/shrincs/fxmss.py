# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""FXMSS — the flexible XMSS tree SHRINCS's stateful path signs under.

An XMSS tree whose shape the signer chooses: an unbalanced tree makes the first
signatures tiny and each one after it larger, a balanced tree makes them all the
same size, and a signer picks whichever suits how often it signs.

**The verifier is agnostic to that choice**, which the specification says
outright and which is the whole reason this module is short. A signature carries
the height and index of the leaf that made it, so the verifier walks from that
leaf to the root and never learns what shape the rest of the tree had. Nothing
here takes a shape or a depth parameter.

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

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from sig_frx.hash import bytestring
from sig_frx.hash.shrincs import adrs as sf_adrs
from sig_frx.hash.shrincs import wots_c
from sig_frx.hash.tweakable import TweakableHash

# `FXMSS_HEIGHT` — the imaginary height of the tree, and so the deepest a leaf
# can sit. It is 255 because a node height is one byte of the address and of the
# signature, which is the same reason the walk below is 255 steps.
HEIGHT = 255

_N = 16
# The index field is as many whole bytes as the depth needs, capped at eight:
# nothing beyond 64 bits is addressable, which is what the tree slot holds.
_MAX_INDEX_BYTES = 8

# `FXMSS_SIGNATURE_SIZE_MIN` and `_MAX`: a WOTS+C signature and one node per step.
SIGNATURE_SIZE_MIN = wots_c.SIGNATURE_SIZE + _N
SIGNATURE_SIZE_MAX = wots_c.SIGNATURE_SIZE + HEIGHT * _N


def index_field_bytes(leaf_depth: int) -> int:
    """`ceil(min(leaf_depth, 64) / 8)` — the index field's width, 1 to 8 bytes.

    A host function, because the width is what a *parser* needs and a parser has
    the indicator byte concretely: the field's width decides where the FXMSS
    signature starts, so it cannot itself be read out of the signature.
    """
    return -(-min(leaf_depth, 8 * _MAX_INDEX_BYTES) // 8)


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
    if indices.ndim != 2 or indices.shape[1] != _MAX_INDEX_BYTES:
        raise ValueError(
            f"a leaf index batch is [B, {_MAX_INDEX_BYTES}] bytes, got shape "
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

    parents = indices
    for step in range(HEIGHT):
        siblings = path[:, step, :]
        # Bit `step` of the index, read from the byte that holds it: big-endian,
        # so bit 0 lives in the last byte. Shifting the whole string once per step
        # would be the same answer for more work.
        byte = _MAX_INDEX_BYTES - 1 - step // 8
        on_the_right = ((indices[:, byte] >> (step % 8)) & 1).astype(fnp.uint8)[:, None]
        left = fnp.where(on_the_right, siblings, nodes)
        right = fnp.where(on_the_right, nodes, siblings)
        parents = bytestring.shift_right(parents, 1)
        # Clamped because a masked step can address above the root: a leaf at
        # height 254 has one real step and 254 discarded ones, whose heights would
        # run past the byte the slot gives them and raise on a value nothing reads.
        parent_heights = fnp.minimum(heights + (step + 1), np.uint32(HEIGHT))
        combined = tweak.h(
            pk_seed,
            sf_adrs.encode_batch(sf_adrs.fxmss_tree(parent_heights, parents)),
            fnp.concatenate([left, right], axis=-1),
        )
        nodes = fnp.where((step < depths)[:, None], combined, nodes)
    return nodes, accepted
