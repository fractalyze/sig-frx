# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""WOTS+C — the one-time signature SHRINCS's stateful path is built from.

WOTS+C replaces WOTS+'s checksum chains with a protocol requirement: a message
must map to chain indexes that sum to a fixed constant. A forger who raises one
index has to lower another, and lowering one means walking a chain backwards, so
the checksum's job is done by the constraint instead of by three extra chains.
That is what takes a signature from 35 chains to 32 — 512 bytes of chain tips
rather than 560 — and it is why the verifier's work is fixed rather than
message-dependent: every chain is walked to the same end.

Not every digest maps into the constant-sum subset, so the signer *grinds*: it
hashes the digest under a rolling 16-bit counter until the indexes sum to
`CONSTANT_SUM`, and puts the counter in the signature. **The verifier does not
grind.** It recomputes the map once at the counter it was handed and rejects a
counter that does not land in the subset — one hash, not a search, which is the
only reason this path can be a fixed-cost verifier at all.

**The grind is the one place a loop here runs a data-dependent number of times**,
and it is the signer's alone. `grind` searches counters until one lands in the
subset, so it reads a verdict back to decide whether to search further — which
makes it concrete-only, unlike everything else in this module. It searches a
block of counters per pass rather than one at a time: at these parameters about
one counter in sixty-five lands, so a block of `_GRIND_BLOCK` is a single batched
dispatch where stepping would be dozens of serial ones, and the hashes a hit
leaves unused cost less than the dispatches they save. Signing carries no
side-channel claim here — see
[`security.md`](../../../docs/reference/security.md) — which is what makes
reading that verdict back allowed at all.

SHRINCS's signer is also stateful — see [`../../signature.py`](../../signature.py)
on why a stateful scheme's `sign` is not the seam's. What this module signs is one
leaf; which leaf, and never twice, is [`fxmss.py`](fxmss.py)'s and its caller's.

The chain walk itself is [`../wots.py`](../wots.py)'s: the same masked
`w − 1` steps under a different address and a different `F` domain, exactly as
RFC 8391 shares it. What is not shared is the map from digest to indexes, which
is where the checksum used to be.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from sig_frx.hash import adrs, adrs_encoding, wots
from sig_frx.hash.shrincs import adrs as sf_adrs
from sig_frx.hash.tweakable import TweakableHash, repeat_per_entry

# The specification's stateful parameters, and the constants derived from them.
CHAIN_BITS = 4  # `WOTS_C_CHAIN_BITS`
CHAIN_COUNT = 32  # `WOTS_C_CHAIN_COUNT`
_N = 16
CHAINS_SIZE = CHAIN_COUNT * _N  # `WOTS_C_CHAINS_SIZE`, 512
_MAX_INDEX = 2**CHAIN_BITS - 1
# `ceil(count · (w − 1) / 2)` — the most likely sum, which is what makes grinding
# terminate quickly: it is the mode of the distribution the map draws from.
CONSTANT_SUM = -(-(CHAIN_COUNT * _MAX_INDEX) // 2)  # 240

# The tree's shape and depth, which only the signer's PRF address carries. They
# sit in the high half of a payload word, so the rest of it is zero padding.
STRUCTURE_BYTES = 2
_STRUCTURE_PADDING = adrs.WORD_SIZE - STRUCTURE_BYTES

# The counter is two bytes at the front of the signature, then the chain tips.
COUNTER_BYTES = 2
SIGNATURE_SIZE = COUNTER_BYTES + CHAINS_SIZE  # 514

# `H_grind` hashes the digest, four zero bytes and the counter. The padding is
# the specification's, and it is what keeps the grind input a different length
# from every other tweaked hash's.
_GRIND_PADDING = 4

# The counter is two bytes, so the search cannot run past this many.
_GRIND_LIMIT = 2 ** (8 * COUNTER_BYTES)
# How many counters `grind` tries per pass. Chosen so that one pass almost always
# suffices — the map lands in the subset about once in sixty-five, so the chance
# a block of this size holds none is under two percent — while staying one batched
# hash rather than a Python loop over counters.
_GRIND_BLOCK = 256


def map_digest(
    tweak: TweakableHash,
    pk_seed: ArrayLike,
    message_digests: ArrayLike,
    counters: ArrayLike,
    node_heights: ArrayLike,
    node_indices: ArrayLike,
) -> tuple[Array, Array]:
    """`wots_c_map_digest` — the indexes a counter yields, and whether they sum.

    `message_digests` is `[B, 32]`, `counters` is `[B, 2]` **bytes** — the two the
    signature carries, unread as a number, since they go back into a hash as
    bytes and nothing here does arithmetic on them. The result is the `[B, 32]`
    index column and a `bool[B]` saying whether each row met the constraint.

    A row that did not is not an error: a counter is attacker-supplied, so a
    counter that maps nowhere is a signature that does not verify.
    """
    digests = fnp.asarray(message_digests, dtype=fnp.uint8)
    counter_bytes = fnp.asarray(counters, dtype=fnp.uint8)
    batch = digests.shape[0]
    grind = sf_adrs.encode_batch(sf_adrs.wots_c_grind(node_heights, node_indices))
    hashed = tweak.t(
        pk_seed,
        grind[:, : sf_adrs.GRIND_TWEAK_BYTES],
        fnp.concatenate(
            [
                digests,
                fnp.zeros((batch, _GRIND_PADDING), dtype=fnp.uint8),
                counter_bytes,
            ],
            axis=-1,
        ),
    )
    indexes = wots.base_2b(hashed, CHAIN_BITS, CHAIN_COUNT)
    # `dtype` pinned: numpy promotes a reduction's accumulator and frx does not,
    # so a bare `sum` is uint64 on the host and uint32 traced from this one line.
    # The values agree at these widths, and the dtypes would not.
    return indexes, indexes.sum(axis=-1, dtype=np.uint32) == np.uint32(CONSTANT_SUM)


def pk_from_sig(
    tweak: TweakableHash,
    pk_seed: ArrayLike,
    signatures: ArrayLike,
    message_digests: ArrayLike,
    node_heights: ArrayLike,
    node_indices: ArrayLike,
) -> tuple[Array, Array]:
    """`wots_c_pubkey_from_sig` — the public key each signature implies.

    `signatures` is `[B, 514]`, `message_digests` is `[B, 32]`, and the result is
    the `[B, 16]` compressed public key with a `bool[B]` saying whether the
    counter was valid. The key is computed either way: a rejected row still walks
    its chains, because the alternative is a branch on the counter, and the
    verdict rides beside the value rather than short-circuiting it.

    Every chain runs the full `w − 1` steps whatever its index says, which is
    `wots.chain`'s doing and the property WOTS+C already guarantees at the
    protocol level: the indexes sum to a constant, so the total work was fixed
    before the mask was.
    """
    parts = fnp.asarray(signatures, dtype=fnp.uint8)
    if parts.ndim != 2 or parts.shape[1] != SIGNATURE_SIZE:
        raise ValueError(
            f"a WOTS+C signature batch is [B, {SIGNATURE_SIZE}], got shape "
            f"{tuple(parts.shape)}"
        )
    batch = parts.shape[0]
    indexes, accepted = map_digest(
        tweak,
        pk_seed,
        message_digests,
        parts[:, :COUNTER_BYTES],
        node_heights,
        node_indices,
    )

    # Chain-major within each entry, which is the layout `wots.chain` walks and
    # the order the tips are concatenated back in.
    tips = parts[:, COUNTER_BYTES:].reshape(batch * CHAIN_COUNT, _N)
    starts = indexes.reshape(-1)
    ends = wots.chain(
        tweak,
        # One leaf is `CHAIN_COUNT` chains, so a per-entry seed repeats that many
        # times to line up with the chain-major rows the addresses were built in.
        repeat_per_entry(pk_seed, CHAIN_COUNT),
        tips,
        starts,
        _MAX_INDEX - starts,
        _chain_addresses(node_heights, node_indices, batch),
    )
    return _compress(tweak, pk_seed, ends, node_heights, node_indices, batch), accepted


def _chain_addresses(
    node_heights: ArrayLike, node_indices: ArrayLike, batch: int
) -> list[np.ndarray | Array]:
    """Every chain's address at every step, chain-major within each entry.

    Encoded once at step zero and spliced per step: only the hash index moves, and
    it is one value for the whole batch. `wots.py` learned that on the SLH-DSA
    verify path, where re-encoding was two fifths of an eager verification.

    Shared by the two directions rather than written twice: a signer walks the
    chains from zero and a verifier from the message's indexes, and the addresses
    those walks tweak with are the same batch.
    """
    at_first_step = sf_adrs.encode_batch(
        sf_adrs.wots_c_hash(
            adrs_encoding.repeat_rows(node_heights, CHAIN_COUNT),
            adrs_encoding.repeat_rows(node_indices, CHAIN_COUNT),
            np.tile(np.arange(CHAIN_COUNT, dtype=np.uint32), batch),
            0,
        )
    )
    return [sf_adrs.with_hash_index(at_first_step, step) for step in range(_MAX_INDEX)]


def _compress(
    tweak: TweakableHash,
    pk_seed: ArrayLike,
    ends: Array,
    node_heights: ArrayLike,
    node_indices: ArrayLike,
    batch: int,
) -> Array:
    """`T_sf` over each entry's chain ends — the FXMSS leaf they make. `[B, 16]`.

    The signer reaches the same ends by walking every chain to `w − 1` and the
    verifier by walking each from its index, so the leaf they compress to is one
    operation with two callers.
    """
    return tweak.t(
        pk_seed,
        sf_adrs.encode_batch(sf_adrs.wots_c_pk(node_heights, node_indices)),
        ends.reshape(batch, CHAINS_SIZE),
    )


# -- the signer --------------------------------------------------------------


def secret_values(
    tweak: TweakableHash,
    pk_seed: ArrayLike,
    sk_seed: ArrayLike,
    sf_structure: ArrayLike,
    node_heights: ArrayLike,
    node_indices: ArrayLike,
) -> Array:
    """Each leaf's `CHAIN_COUNT` chain starting points — `[B · 32, 16]`.

    Chain-major within each entry, the layout `_chain_addresses` builds and
    `wots.chain` walks. `sf_structure` is the two structure bytes, which ride in
    the address for the reason `adrs.wots_c_prf` gives.

    `pk_seed` is `[n]` when one key owns the whole batch — which is what key
    generation is — or `[B, n]` when it does not, the same rule `pk_from_sig`
    follows. `sk_seed` is always one key's.
    """
    indices = fnp.asarray(node_indices, dtype=fnp.uint8)
    batch = indices.shape[0]
    structure = np.asarray(sf_structure, dtype=np.uint8).reshape(-1)
    if structure.shape != (STRUCTURE_BYTES,):
        raise ValueError(
            f"a tree structure is {STRUCTURE_BYTES} bytes — a shape and a depth — "
            f"got shape {tuple(structure.shape)}"
        )
    return tweak.prf(
        # One leaf is `CHAIN_COUNT` chains, the same widening `pk_from_sig` does.
        repeat_per_entry(pk_seed, CHAIN_COUNT),
        sk_seed,
        sf_adrs.encode_batch(
            sf_adrs.wots_c_prf(
                adrs_encoding.repeat_rows(node_heights, CHAIN_COUNT),
                adrs_encoding.repeat_rows(indices, CHAIN_COUNT),
                # Right-padded into the four-byte word the slot gives it, and
                # spread over the rows by hand: `adrs_encoding` broadcasts an
                # integer field across a batch but takes a byte field as the rows
                # it already is.
                np.broadcast_to(
                    np.concatenate([structure, np.zeros(_STRUCTURE_PADDING, np.uint8)]),
                    (batch * CHAIN_COUNT, _STRUCTURE_PADDING + STRUCTURE_BYTES),
                ),
                np.tile(np.arange(CHAIN_COUNT, dtype=np.uint32), batch),
            )
        ),
    )


def public_key(
    tweak: TweakableHash,
    pk_seed: ArrayLike,
    sk_seed: ArrayLike,
    sf_structure: ArrayLike,
    node_heights: ArrayLike,
    node_indices: ArrayLike,
) -> Array:
    """`wots_c_pubkey_gen` — the FXMSS leaf at each position. `[B, 16]`.

    Batched, because key generation builds every leaf the tree's shape names and
    none of them depend on each other: a balanced tree of depth `d` is one call of
    `2^d` leaves rather than `2^d` calls.
    """
    indices = fnp.asarray(node_indices, dtype=fnp.uint8)
    batch = indices.shape[0]
    rows = batch * CHAIN_COUNT
    ends = wots.chain(
        tweak,
        repeat_per_entry(pk_seed, CHAIN_COUNT),
        secret_values(tweak, pk_seed, sk_seed, sf_structure, node_heights, indices),
        fnp.zeros(rows, dtype=fnp.uint32),
        fnp.full(rows, _MAX_INDEX, dtype=fnp.uint32),
        _chain_addresses(node_heights, indices, batch),
    )
    return _compress(tweak, pk_seed, ends, node_heights, indices, batch)


def grind(
    tweak: TweakableHash,
    pk_seed: ArrayLike,
    message_digest: ArrayLike,
    node_height: ArrayLike,
    node_index: ArrayLike,
) -> tuple[int, Array]:
    """`wots_c_grind_to_constant_sum` — the lowest counter that lands in the subset.

    Returns that counter and the `[32]` index set it yields. The counter comes back
    as a Python integer because it is read to decide the search is over, which
    makes it concrete by construction; the signature carries it as two bytes, and
    that is an encoding rather than arithmetic on it.

    A block of counters per pass, for the reason the module docstring gives. The
    block divides `_GRIND_LIMIT`, both being powers of two, so the last pass ends
    exactly at the largest counter two bytes hold.
    """
    digest = fnp.asarray(message_digest, dtype=fnp.uint8).reshape(1, -1)
    index = fnp.asarray(node_index, dtype=fnp.uint8).reshape(1, -1)
    digests = fnp.broadcast_to(digest, (_GRIND_BLOCK, digest.shape[1]))
    indices = fnp.broadcast_to(index, (_GRIND_BLOCK, index.shape[1]))
    heights = fnp.broadcast_to(
        fnp.asarray(node_height, dtype=fnp.uint32).reshape(1), (_GRIND_BLOCK,)
    )
    for base in range(0, _GRIND_LIMIT, _GRIND_BLOCK):
        # Big-endian in `COUNTER_BYTES`, reinterpreted rather than shifted apart:
        # the counter is a number while the search enumerates it and bytes from
        # the moment it reaches a hash, and this is the one place it crosses.
        counters = np.arange(base, base + _GRIND_BLOCK, dtype=np.uint16)
        indexes, accepted = map_digest(
            tweak,
            pk_seed,
            digests,
            counters.astype(">u2").view(np.uint8).reshape(-1, COUNTER_BYTES),
            heights,
            indices,
        )
        landed = np.flatnonzero(np.asarray(accepted))
        if landed.size:
            first = int(landed[0])
            return base + first, indexes[first]
    raise ValueError(
        f"no counter below {_GRIND_LIMIT} maps this digest into the constant-sum "
        f"subset; about one in sixty-five does, so this is a wrong digest or a "
        f"wrong address rather than bad luck"
    )


def sign(
    tweak: TweakableHash,
    pk_seed: ArrayLike,
    sk_seed: ArrayLike,
    sf_structure: ArrayLike,
    message_digest: ArrayLike,
    node_height: ArrayLike,
    node_index: ArrayLike,
) -> Array:
    """`wots_c_sign` — the counter, then each chain stopped at its index. `[514]`.

    One leaf, because a WOTS+C key signs once and signing it twice reveals the
    secret: the batch axis belongs to verification, which is the side that meets
    many signatures. `fxmss.sign` is what holds a leaf to one use.
    """
    index = fnp.asarray(node_index, dtype=fnp.uint8).reshape(1, -1)
    heights = fnp.asarray(node_height, dtype=fnp.uint32).reshape(1)
    counter, indexes = grind(tweak, pk_seed, message_digest, heights, index)
    ends = wots.chain(
        tweak,
        pk_seed,
        secret_values(tweak, pk_seed, sk_seed, sf_structure, heights, index),
        fnp.zeros(CHAIN_COUNT, dtype=fnp.uint32),
        indexes,
        _chain_addresses(heights, index, 1),
    )
    return fnp.concatenate(
        [
            fnp.asarray(
                np.frombuffer(counter.to_bytes(COUNTER_BYTES, "big"), dtype=np.uint8),
                dtype=fnp.uint8,
            ),
            ends.reshape(CHAINS_SIZE),
        ]
    )
