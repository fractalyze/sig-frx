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

**Verification only.** Signing needs the grind and the secret chain starts, and
SHRINCS's signer is stateful — see [`../../signature.py`](../../signature.py) on
why a stateful scheme's `sign` is not the seam's.

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

from sig_frx.hash import adrs_encoding, wots
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

# The counter is two bytes at the front of the signature, then the chain tips.
COUNTER_BYTES = 2
SIGNATURE_SIZE = COUNTER_BYTES + CHAINS_SIZE  # 514

# `H_grind` hashes the digest, four zero bytes and the counter. The padding is
# the specification's, and it is what keeps the grind input a different length
# from every other tweaked hash's.
_GRIND_PADDING = 4


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
    heights = adrs_encoding.repeat_rows(node_heights, CHAIN_COUNT)
    nodes = adrs_encoding.repeat_rows(node_indices, CHAIN_COUNT)
    chains = np.tile(np.arange(CHAIN_COUNT, dtype=np.uint32), batch)
    step_addresses = [
        sf_adrs.encode_batch(sf_adrs.wots_c_hash(heights, nodes, chains, step))
        for step in range(_MAX_INDEX)
    ]
    ends = wots.chain(
        tweak,
        # One leaf is `CHAIN_COUNT` chains, so a per-entry seed repeats that many
        # times to line up with the chain-major rows the addresses were built in.
        repeat_per_entry(pk_seed, CHAIN_COUNT),
        tips,
        starts,
        _MAX_INDEX - starts,
        step_addresses,
    )

    public_keys = tweak.t(
        pk_seed,
        sf_adrs.encode_batch(sf_adrs.wots_c_pk(node_heights, node_indices)),
        ends.reshape(batch, CHAINS_SIZE),
    )
    return public_keys, accepted
