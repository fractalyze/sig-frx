# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""WOTS+ and its L-tree compression, the RFC 8391 way — §3.1 and §4.1.5.

The chains themselves are `wots.chain`, and the message's chain lengths are
`wots.message_digits`: §3.1.5's `base_w` and checksum are FIPS 205 §5's, digit for
digit, so those are shared rather than rewritten. What is not shared is everything
that touches an address or a seed, which is why this module exists:

- **Secret values come from `PRF_keygen`, keyed by the secret seed.** FIPS 205
  derives a chain's start from `PRF(PK.seed, SK.seed, ADRS)`; here it is
  `PRF_keygen(SK.seed, SEED ‖ ADRS)` — the public seed is part of the *input*, and
  the address is appended to it rather than tweaking a prefix.
- **The addresses are RFC 8391's eight words** (`rfc8391_adrs`), and a component
  passes the caller's leading words through untouched, type word included. That is
  what makes the published WOTS+ vectors reproducible: the fixture they were
  generated from carries a type word no scheme would ever produce.
- **The public key compresses through an L-tree, not `T_len`.** Which is the
  argument `wots.pk_gen` takes a `Compress` for; this module supplies the other
  implementation of it.

**The L-tree is not `tree.reduce_levels` and cannot be.** A Merkle level pairs its
nodes, and `reduce_levels` refuses an odd count on purpose. An L-tree spans
`wots_len` leaves — 67, or 51 at `n = 24` — so it meets an odd level almost
immediately, and §4.1.5's rule is to lift the unpaired node to the next level
*unhashed*. A flag that switched `reduce_levels` between refusing and lifting would
be two behaviours behind one name, so the walk is written out here.

A batch is key-pair-major throughout — key pair 0's `len` chains, then key pair
1's — the same layout `wots.py` uses, so an XMSS tree's `2^h` leaves are one
batched walk rather than `2^h` of them.
"""

from __future__ import annotations

from collections.abc import Sequence

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from sig_frx.hashbased import rfc8391_adrs, wots
from sig_frx.hashbased.rfc8391_adrs import Adrs
from sig_frx.hashbased.rfc8391_hashes import Rfc8391Hashes
from sig_frx.hashbased.tweakable import NodeHash, repeat_per_entry


def _leading_columns(
    positions: Sequence[Adrs], per_position: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Each position's four leading words, repeated once per row it owns.

    The layer, tree, type and first word are what a component inherits from its
    caller and must not touch — the type word included, since RFC 8391's `setType`
    writes one word without normalizing and the vectors depend on it. Building them
    as columns once keeps the cost of an address batch independent of its size.

    A position is one address, not a batch of them: the sequence is the batch.
    """

    def column(values: list[int]) -> np.ndarray:
        return np.repeat(values, per_position)

    return (
        column([int(position.layer) for position in positions]),
        column([int(position.tree) for position in positions]),
        column([int(position.type) for position in positions]),
        column([int(position.trailing[0]) for position in positions]),
    )


def _chain_addresses(
    params: wots.WotsParams, positions: Sequence[Adrs]
) -> list[np.ndarray]:
    """Every chain's address at every step, for every key pair.

    `w − 1` batches of `len(positions) · len` addresses. §3.1.2 advances the hash
    address with the step and holds the chain address across it, so the addresses
    differ per step and not just per chain.
    """
    layers, trees, types, ots = _leading_columns(positions, params.len)
    chains = np.tile(np.arange(params.len), len(positions))
    return [
        rfc8391_adrs.encode_batch(Adrs(layers, trees, types, (ots, chains, step)))
        for step in range(params.w - 1)
    ]


def secret_values(
    hashes: Rfc8391Hashes,
    params: wots.WotsParams,
    pub_seed: ArrayLike,
    sk_seed: ArrayLike,
    positions: Sequence[Adrs],
) -> Array:
    """Every key pair's `len` chain starting points — §3.1.3's `expand_seed`.

    `[len(positions) · len, n]`, key-pair-major. The address is the chain's, with
    the hash address and `keyAndMask` both zero, and it is appended to the public
    seed to make `PRF_keygen`'s input rather than tweaking it.
    """
    layers, trees, types, ots = _leading_columns(positions, params.len)
    addresses = rfc8391_adrs.encode_batch(
        Adrs(
            layers,
            trees,
            types,
            (ots, np.tile(np.arange(params.len), len(positions)), 0),
        )
    )
    seeds = repeat_per_entry(pub_seed, params.len)
    if seeds.ndim == 1:
        seeds = fnp.broadcast_to(seeds, (addresses.shape[0], hashes.n))
    return hashes.prf_keygen(
        sk_seed,
        fnp.concatenate([seeds, fnp.asarray(addresses, dtype=fnp.uint8)], axis=-1),
    )


def pk_gen(
    hashes: Rfc8391Hashes,
    params: wots.WotsParams,
    pub_seed: ArrayLike,
    sk_seed: ArrayLike,
    positions: Sequence[Adrs],
    compress: wots.Compress,
) -> Array:
    """`WOTS_genPK` — §3.1.4, for every key pair at once. `[P, n]`."""
    rows = len(positions) * params.len
    ends = wots.chain(
        hashes,
        pub_seed,
        secret_values(hashes, params, pub_seed, sk_seed, positions),
        fnp.zeros(rows, dtype=fnp.uint32),
        fnp.full(rows, params.w - 1, dtype=fnp.uint32),
        _chain_addresses(params, positions),
    )
    return compress(ends.reshape(len(positions), params.len * hashes.n))


def sign(
    hashes: Rfc8391Hashes,
    params: wots.WotsParams,
    message: ArrayLike,
    pub_seed: ArrayLike,
    sk_seed: ArrayLike,
    position: Adrs,
) -> Array:
    """`WOTS_sign` — §3.1.5: each chain stopped at its digit. `[len, n]`.

    One key pair, because signing is one key pair — verification is the side that
    batches, and it goes through `pk_from_sig`.
    """
    digits = wots.message_digits(params, message)[0]
    return wots.chain(
        hashes,
        pub_seed,
        secret_values(hashes, params, pub_seed, sk_seed, [position]),
        fnp.zeros(params.len, dtype=fnp.uint32),
        digits,
        _chain_addresses(params, [position]),
    )


def pk_from_sig(
    hashes: Rfc8391Hashes,
    params: wots.WotsParams,
    signatures: ArrayLike,
    messages: ArrayLike,
    pub_seed: ArrayLike,
    positions: Sequence[Adrs],
    compress: wots.Compress,
) -> Array:
    """`WOTS_pkFromSig` — §3.1.6, for a batch. `[P, len, n]` -> `[P, n]`.

    The operation verification runs, so it takes many signatures: each with its own
    message, its own key pair, and — when the batch spans more than one public key —
    its own seed. A signature that stopped its chains anywhere other than its
    message's digits arrives at a different public key.
    """
    digits = wots.message_digits(params, messages).reshape(-1)
    rows = fnp.asarray(signatures, dtype=fnp.uint8).reshape(
        len(positions) * params.len, hashes.n
    )
    ends = wots.chain(
        hashes,
        repeat_per_entry(pub_seed, params.len),
        rows,
        digits,
        (params.w - 1) - digits,
        _chain_addresses(params, positions),
    )
    return compress(ends.reshape(len(positions), params.len * hashes.n))


def ltree_compression(
    hashes: NodeHash,
    pub_seed: ArrayLike,
    positions: Sequence[Adrs],
    *,
    leaves: int,
) -> wots.Compress:
    """§4.1.5's `ltree` as a `wots.Compress` — `[P, leaves · n]` -> `[P, n]`.

    `positions` are the L-tree addresses, which is a different address from the OTS
    one the chains were walked under: §4.1.6 builds both from the same leaf and they
    differ in their type and in what their first word means. So the caller supplies
    both rather than this deriving one from the other, which is also what makes the
    reference's own WOTS+ fixture — two unrelated addresses — expressible.

    `leaves` is `wots_len` for every caller a parameter set has, and a plain count
    here because the walk does not care where it came from: it is the only thing
    about WOTS+ the compression needs.

    One batched hash per level, over every key pair's level at once: `P` L-trees
    cost `ceil(log2(leaves))` calls, not `P` times that.
    """
    layers, trees, types, ltrees = _leading_columns(positions, 1)

    def compress(ends: Array) -> Array:
        nodes = fnp.asarray(ends, dtype=fnp.uint8).reshape(
            len(positions), leaves, hashes.n
        )
        remaining = leaves
        height = 0
        while remaining > 1:
            parents = remaining // 2
            pairs = nodes[:, : 2 * parents, :].reshape(
                len(positions) * parents, 2 * hashes.n
            )
            hashed = hashes.h(
                repeat_per_entry(pub_seed, parents),
                rfc8391_adrs.encode_batch(
                    Adrs(
                        np.repeat(layers, parents),
                        np.repeat(trees, parents),
                        np.repeat(types, parents),
                        (
                            np.repeat(ltrees, parents),
                            height,
                            np.tile(np.arange(parents), len(positions)),
                        ),
                    )
                ),
                pairs,
            ).reshape(len(positions), parents, hashes.n)
            if remaining % 2:
                # §4.1.5 lifts the unpaired node to the next level unhashed. Padding
                # or duplicating it instead would compress two different public keys
                # to the same leaf.
                nodes = fnp.concatenate(
                    [hashed, nodes[:, remaining - 1 : remaining, :]], axis=1
                )
                remaining = parents + 1
            else:
                nodes = hashed
                remaining = parents
            height += 1
        return nodes[:, 0, :]

    return compress
