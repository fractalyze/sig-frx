# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""WOTS+ — the one-time signature SLH-DSA and XMSS are both built from.

FIPS 205 §5. A private key is `len` secret values; each is walked up a hash chain
of length `w`, and the chain ends compress to the public key. A signature stops
each chain at the base-`w` digit of the message it signs, so a verifier can walk
the rest and arrive at the public key — and cannot walk backwards.

**Every chain runs the full `w − 1` steps, and a mask decides which ones advance.**
A chain's stopping point is a function of the message, so walking exactly as far
as each digit asks would be a data-dependent trip count: it lowers to control
flow, splits the kernel, and makes the timing a function of the message digest.
The masked form does the same work whatever the message is, and `chain` is one
batched hash per step instead of one per chain step — 15 calls where the literal
reading of Algorithm 5 would make 525.

Public-key compression is deliberately *not* here. FIPS 205 compresses the chain
ends with one tweakable hash while RFC 8391 uses an L-tree, and that is the only
place the two disagree — so `pk_gen` and `pk_from_sig` take the compression as an
argument and XMSS supplies its own.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from sig_frx.hashbased import adrs
from sig_frx.hashbased.tweakable import TweakableHash, repeat_per_entry

# The compression a caller supplies: each key pair's chain ends, `[P, len · n]`,
# compressed to `[P, n]`.
Compress = Callable[[Array], Array]


@dataclass(frozen=True)
class WotsParams:
    """The four derived values of FIPS 205 §5, equations 5.1 to 5.4."""

    n: int
    lg_w: int = 4

    @cached_property
    def w(self) -> int:
        return 1 << self.lg_w

    @cached_property
    def len1(self) -> int:
        return -(-8 * self.n // self.lg_w)

    @cached_property
    def len2(self) -> int:
        return (self.len1 * (self.w - 1)).bit_length() // self.lg_w + 1

    @cached_property
    def len(self) -> int:
        return self.len1 + self.len2


@dataclass(frozen=True)
class WotsPosition:
    """Which WOTS+ key pair this is — the address prefix every call tweaks with."""

    layer: int
    tree: int
    key_pair: int


def base_2b(data: ArrayLike, b: int, out_len: int) -> Array:
    """`base_2b` — FIPS 205 §4.4, Algorithm 4: `X` as `out_len` base-2^b digits.

    Vectorized over a leading batch axis, and unrolled: `out_len` and `b` are
    parameter-set constants, so the standard's `while` over input bytes is a
    static schedule rather than a loop over data. The accumulator keeps 24 bits,
    which is more than the `b + 8` the algorithm can ever read from it.
    """
    values = fnp.asarray(data, dtype=fnp.uint32)
    if values.ndim == 1:
        values = values[None, :]
    needed = -(-out_len * b // 8)
    if values.shape[-1] < needed:
        raise ValueError(
            f"base_2b needs at least {needed} bytes for {out_len} digits of "
            f"{b} bits, got {values.shape[-1]}"
        )

    digits = []
    total = fnp.zeros(values.shape[0], dtype=fnp.uint32)
    consumed = 0
    bits = 0
    for _ in range(out_len):
        while bits < b:
            total = ((total << 8) | values[:, consumed]) & 0xFFFFFF
            consumed += 1
            bits += 8
        bits -= b
        digits.append((total >> bits) & (2**b - 1))
    return fnp.stack(digits, axis=-1)


def message_digits(params: WotsParams, message: ArrayLike) -> Array:
    """The `len` base-`w` digits a signature's chain lengths come from.

    Algorithm 7 lines 1 to 7: the message's own digits, then the checksum's. The
    checksum is what stops an attacker from raising a digit — walking one chain
    further — because raising any message digit lowers the checksum, and the
    checksum's own chains cannot be walked backwards.
    """
    digits = base_2b(message, params.lg_w, params.len1)
    checksum = fnp.sum(
        (params.w - 1) - digits.astype(fnp.uint32), axis=-1, dtype=fnp.uint32
    )
    # Left-aligned in its byte string, per line 6: for lg_w = 4 and len2 = 3 the
    # 12 checksum bits sit in the top of two bytes.
    checksum = checksum << ((8 - ((params.len2 * params.lg_w) % 8)) % 8)
    checksum_bytes = -(-params.len2 * params.lg_w // 8)
    packed = fnp.stack(
        [
            (checksum >> (8 * (checksum_bytes - 1 - i))) & 0xFF
            for i in range(checksum_bytes)
        ],
        axis=-1,
    ).astype(fnp.uint8)
    return fnp.concatenate([digits, base_2b(packed, params.lg_w, params.len2)], axis=-1)


def chain(
    tweak: TweakableHash,
    pk_seed: ArrayLike,
    values: ArrayLike,
    start: ArrayLike,
    steps: ArrayLike,
    step_addresses: Sequence[ArrayLike],
) -> Array:
    """`chain` — FIPS 205 §5, Algorithm 5, for a whole batch of chains at once.

    Entry `k` iterates `F` on `values[k]` for `steps[k]` applications beginning at
    index `start[k]`. `step_addresses[j]` holds every entry's address for step
    `j`, which the caller built on the host — the hash address advances with the
    step, so the addresses differ per step and not just per chain.

    The work is `len(step_addresses)` batched hashes regardless of the starts and
    steps: an entry that has already stopped is hashed anyway and its result
    discarded by the select. That is the point — the alternative branches on
    secret-adjacent data.
    """
    current = fnp.asarray(values, dtype=fnp.uint8)
    starts = fnp.asarray(start, dtype=fnp.uint32)
    counts = fnp.asarray(steps, dtype=fnp.uint32)
    for step, addresses in enumerate(step_addresses):
        stepped = tweak.f(pk_seed, addresses, current)
        active = (starts <= step) & (step < starts + counts)
        current = fnp.where(active[:, None], stepped, current)
    return current


def _position_columns(
    positions: Sequence[WotsPosition], per_position: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each key pair's address prefix, repeated once per row it owns.

    The rows are key-pair-major, which is the layout every batch here uses: key
    pair 0's `len` chains, then key pair 1's. Building the prefix as three columns
    once is what keeps the cost of an address batch independent of how many
    addresses are in it.
    """
    return (
        np.repeat([position.layer for position in positions], per_position),
        np.repeat([position.tree for position in positions], per_position),
        np.repeat([position.key_pair for position in positions], per_position),
    )


def _chain_addresses(
    params: WotsParams, positions: Sequence[WotsPosition]
) -> list[np.ndarray]:
    """Every chain's address at every step, for every key pair.

    `w − 1` batches of `len(positions) · len` addresses, laid out key pair by key
    pair. That layout is what lets an XMSS tree walk all of its WOTS+ keys in one
    call: 2^h' key pairs of `len` chains are one batch, so a tree's leaves cost
    `w − 1` hashes rather than `2^h' · (w − 1)`.
    """
    layers, trees, key_pairs = _position_columns(positions, params.len)
    chains = np.tile(np.arange(params.len), len(positions))
    return [
        adrs.encode_batch(
            adrs.wots_hash(
                layer=layers,
                tree=trees,
                key_pair=key_pairs,
                chain=chains,
                hash_index=step,
            ),
            compressed=True,
        )
        for step in range(params.w - 1)
    ]


def secret_values(
    tweak: TweakableHash,
    params: WotsParams,
    pk_seed: ArrayLike,
    sk_seed: ArrayLike,
    positions: Sequence[WotsPosition],
) -> Array:
    """Every key pair's `len` chain starting points — Algorithm 6 lines 1 to 6.

    `[len(positions) · len, n]`, in the same key-pair-major order as
    `_chain_addresses`.
    """
    layers, trees, key_pairs = _position_columns(positions, params.len)
    addresses = adrs.encode_batch(
        adrs.wots_prf(
            layer=layers,
            tree=trees,
            key_pair=key_pairs,
            chain=np.tile(np.arange(params.len), len(positions)),
        ),
        compressed=True,
    )
    return tweak.prf(pk_seed, sk_seed, addresses)


def pk_gen(
    tweak: TweakableHash,
    params: WotsParams,
    pk_seed: ArrayLike,
    sk_seed: ArrayLike,
    positions: Sequence[WotsPosition],
    compress: Compress,
) -> Array:
    """`wots_pkGen` — Algorithm 6, for every key pair at once. `[P, n]`.

    Every chain of every key pair walks to its end in the same `w − 1` batched
    hashes, because none of them depend on each other.
    """
    rows = len(positions) * params.len
    ends = chain(
        tweak,
        pk_seed,
        secret_values(tweak, params, pk_seed, sk_seed, positions),
        fnp.zeros(rows, dtype=fnp.uint32),
        fnp.full(rows, params.w - 1, dtype=fnp.uint32),
        _chain_addresses(params, positions),
    )
    return compress(ends.reshape(len(positions), params.len * params.n))


def sign(
    tweak: TweakableHash,
    params: WotsParams,
    message: ArrayLike,
    pk_seed: ArrayLike,
    sk_seed: ArrayLike,
    position: WotsPosition,
) -> Array:
    """`wots_sign` — Algorithm 7: each chain stopped at its digit. `[len, n]`.

    One key pair, because signing is one key pair — verification is the side that
    batches, and it goes through `pk_from_sig`.
    """
    digits = message_digits(params, message)[0]
    return chain(
        tweak,
        pk_seed,
        secret_values(tweak, params, pk_seed, sk_seed, [position]),
        fnp.zeros(params.len, dtype=fnp.uint32),
        digits,
        _chain_addresses(params, [position]),
    )


def pk_from_sig(
    tweak: TweakableHash,
    params: WotsParams,
    signatures: ArrayLike,
    messages: ArrayLike,
    pk_seed: ArrayLike,
    positions: Sequence[WotsPosition],
    compress: Compress,
) -> Array:
    """`wots_pkFromSig` — Algorithm 8, for a batch. `[P, len, n]` -> `[P, n]`.

    The operation verification actually runs, so it takes many signatures: each
    with its own message, its own key pair, and — when the batch spans more than
    one public key — its own `pk_seed`. A signature that stopped its chains
    anywhere other than its message's digits arrives at a different public key,
    which is what the caller compares against a root it already trusts.
    """
    digits = message_digits(params, messages).reshape(-1)
    rows = fnp.asarray(signatures, dtype=fnp.uint8).reshape(
        len(positions) * params.len, params.n
    )
    ends = chain(
        tweak,
        # One key pair is `len` chains, so a per-entry seed repeats `len` times to
        # match the key-pair-major row order `_chain_addresses` lays out.
        repeat_per_entry(pk_seed, params.len),
        rows,
        digits,
        (params.w - 1) - digits,
        _chain_addresses(params, positions),
    )
    return compress(ends.reshape(len(positions), params.len * params.n))


def fips205_compression(
    tweak: TweakableHash, pk_seed: ArrayLike, positions: Sequence[WotsPosition]
) -> Compress:
    """FIPS 205's public-key compression: one `T_len` over each key pair's ends.

    RFC 8391 compresses with an L-tree instead, which is why this is a value a
    caller passes rather than something WOTS+ decides for itself.
    """
    layers, trees, key_pairs = _position_columns(positions, 1)
    addresses = adrs.encode_batch(
        adrs.wots_pk(layers, trees, key_pairs), compressed=True
    )

    def compress(ends: Array) -> Array:
        return tweak.t(pk_seed, addresses, ends)

    return compress
