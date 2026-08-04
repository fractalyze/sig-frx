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
from functools import cached_property, lru_cache

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from sig_frx.hashbased import adrs, adrs_encoding
from sig_frx.hashbased.tweakable import ChainHash, TweakableHash, repeat_per_entry

# The compression a caller supplies: each key pair's chain ends, `[P, len · n]`,
# compressed to `[P, n]`.
Compress = Callable[[Array], Array]

Field = adrs.Field

# The widest window `base_2b` can read a digit out of, bounded by the uint32 it
# assembles the window in. A digit may start part-way into a byte, so the window
# has to hold `b` bits from any of the eight offsets within one.
_MAX_WINDOW_BYTES = 4
_MAX_DIGIT_BITS = 8 * _MAX_WINDOW_BYTES - 7


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
    """Which WOTS+ key pair this is — the address prefix every call tweaks with.

    One key pair, or a batch of them: every field is an integer or one per key
    pair, and the fields broadcast against each other the way an address's do. A
    batch is what both callers hold — an XMSS tree's `2^h'` key pairs share a
    layer and a tree while their key pair numbers run, and a verifier's `B` claims
    each sit in their own tree — so a key pair is a row of three columns rather
    than an object, and adding one costs no Python.
    """

    layer: Field
    tree: Field
    key_pair: Field

    @property
    def count(self) -> int:
        """How many key pairs this is — one, or the length the fields broadcast to."""
        return adrs_encoding.rows((self.layer, self.tree, self.key_pair))


@lru_cache(maxsize=None)
def _digit_plan(b: int, out_len: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Where the digits sit in the byte stream: each one's first byte and shift,
    and how many bytes the window they are read out of has to span.

    Digit `j` is bits `[j·b, (j+1)·b)` of the input read as one big-endian
    stream, so all three are functions of `j` and `b` alone — constants of the
    parameter set rather than of the message. Precomputing them is what turns the
    conversion into one gather.

    The span is what the digits actually reach rather than a fixed width. Every
    WOTS+ call has `b = lg_w = 4`, which divides a byte, so no digit crosses one
    and the window is the byte itself — the assembly in `base_2b` disappears.
    FORS asks for `a` bits, which does not divide a byte at any parameter set, so
    it pays for a two- or three-byte window; it makes one call where WOTS+ makes
    two per hypertree layer.
    """
    offsets = np.arange(out_len) * b
    within = offsets % 8
    span = -(-(int(within.max(initial=0)) + b) // 8)
    return offsets // 8, 8 * span - b - within, span


def base_2b(data: ArrayLike, b: int, out_len: int) -> Array:
    """`base_2b` — FIPS 205 §4.4, Algorithm 4: `X` as `out_len` base-2^b digits.

    Vectorized over the batch axis and over the digits. Algorithm 4 feeds the
    input into an accumulator a byte at a time because it is written for a scalar
    implementation; read as a whole it says something simpler — digit `j` is bits
    `[j·b, (j+1)·b)` of the big-endian stream — and `b` and `out_len` are
    parameter-set constants, so where every digit sits is known on the host and
    the conversion is one gather, one shift and one mask.

    Transcribing the loop instead costs `out_len` rounds of array work on a batch
    that is already `[B]` wide, and `out_len` is `len1 = 32` at every defined
    parameter set. Measured on the verify path it was 19% of the host time and a
    third of the whole call's array dispatches, against a handful of operations
    here whatever `out_len` is.

    Rank is preserved the way the callers expect: `[L]` gives `[1, out_len]` and
    `[B, L]` gives `[B, out_len]`.
    """
    if b > _MAX_DIGIT_BITS:
        raise ValueError(
            f"a {b}-bit digit does not fit the {8 * _MAX_WINDOW_BYTES}-bit "
            f"window it is read out of, which has to hold it from any bit "
            f"offset within a byte; the widest a defined parameter set asks "
            f"for is 14"
        )
    values = fnp.asarray(data, dtype=fnp.uint32)
    if values.ndim == 1:
        values = values[None, :]
    needed = -(-out_len * b // 8)
    if values.shape[-1] < needed:
        raise ValueError(
            f"base_2b needs at least {needed} bytes for {out_len} digits of "
            f"{b} bits, got {values.shape[-1]}"
        )

    starts, shifts, span = _digit_plan(b, out_len)
    stream = values[:, :needed]
    if span > 1:
        # A window near the end runs past the input. Those bits are never read —
        # a digit starting there would be past `out_len` — so zeros will do.
        stream = fnp.concatenate(
            [stream, fnp.zeros((values.shape[0], span - 1), dtype=fnp.uint32)],
            axis=-1,
        )
    # Every byte position's own window, assembled big-endian. The loop is over
    # the window's bytes — none at all where a digit fits inside one — and never
    # over the digits or the data.
    window = stream[:, :needed]
    for offset in range(1, span):
        window = (window << 8) | stream[:, offset : offset + needed]
    return (window[:, starts] >> fnp.asarray(shifts, dtype=fnp.uint32)) & ((1 << b) - 1)


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
    tweak: ChainHash,
    pk_seed: ArrayLike,
    values: ArrayLike,
    start: ArrayLike,
    steps: ArrayLike,
    step_addresses: Sequence[ArrayLike],
) -> Array:
    """`chain` — FIPS 205 §5, Algorithm 5, for a whole batch of chains at once.

    Also RFC 8391 §3.1.2's `chain`, which is the same walk under a different `F`:
    the standards agree here, and `ChainHash` is what lets both supply theirs.

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
    position: WotsPosition, per_position: int
) -> tuple[Field, Field, Field]:
    """The key pairs' address prefix, each repeated once per row it owns.

    The rows are key-pair-major, which is the layout every batch here uses: key
    pair 0's `len` chains, then key pair 1's. Building the prefix as three columns
    once is what keeps the cost of an address batch independent of how many
    addresses are in it.
    """
    return (
        adrs_encoding.repeat_rows(position.layer, per_position),
        adrs_encoding.repeat_rows(position.tree, per_position),
        adrs_encoding.repeat_rows(position.key_pair, per_position),
    )


def _chain_addresses(
    params: WotsParams, position: WotsPosition, *, compressed: bool
) -> list[np.ndarray]:
    """Every chain's address at every step, for every key pair.

    `w − 1` batches of `position.count · len` addresses, laid out key pair by key
    pair. That layout is what lets an XMSS tree walk all of its WOTS+ keys in one
    call: 2^h' key pairs of `len` chains are one batch, so a tree's leaves cost
    `w − 1` hashes rather than `2^h' · (w − 1)`.

    `compressed` is the parameter set's address encoding, which the callers read
    off their family as `TweakableHash.compressed_address`. `chain` itself never
    sees it: the walk takes the addresses already built, which is what lets
    RFC 8391 hand it a batch of its own.
    """
    layers, trees, key_pairs = _position_columns(position, params.len)
    chains = np.tile(np.arange(params.len), position.count)
    return [
        adrs.encode_batch(
            adrs.wots_hash(
                layer=layers,
                tree=trees,
                key_pair=key_pairs,
                chain=chains,
                hash_index=step,
            ),
            compressed=compressed,
        )
        for step in range(params.w - 1)
    ]


def secret_values(
    tweak: TweakableHash,
    params: WotsParams,
    pk_seed: ArrayLike,
    sk_seed: ArrayLike,
    position: WotsPosition,
) -> Array:
    """Every key pair's `len` chain starting points — Algorithm 6 lines 1 to 6.

    `[position.count · len, n]`, in the same key-pair-major order as
    `_chain_addresses`.
    """
    layers, trees, key_pairs = _position_columns(position, params.len)
    addresses = adrs.encode_batch(
        adrs.wots_prf(
            layer=layers,
            tree=trees,
            key_pair=key_pairs,
            chain=np.tile(np.arange(params.len), position.count),
        ),
        compressed=tweak.compressed_address,
    )
    return tweak.prf(pk_seed, sk_seed, addresses)


def pk_gen(
    tweak: TweakableHash,
    params: WotsParams,
    pk_seed: ArrayLike,
    sk_seed: ArrayLike,
    position: WotsPosition,
    compress: Compress,
) -> Array:
    """`wots_pkGen` — Algorithm 6, for every key pair at once. `[P, n]`.

    Every chain of every key pair walks to its end in the same `w − 1` batched
    hashes, because none of them depend on each other.
    """
    keys = position.count
    rows = keys * params.len
    ends = chain(
        tweak,
        pk_seed,
        secret_values(tweak, params, pk_seed, sk_seed, position),
        fnp.zeros(rows, dtype=fnp.uint32),
        fnp.full(rows, params.w - 1, dtype=fnp.uint32),
        _chain_addresses(params, position, compressed=tweak.compressed_address),
    )
    return compress(ends.reshape(keys, params.len * params.n))


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
        secret_values(tweak, params, pk_seed, sk_seed, position),
        fnp.zeros(params.len, dtype=fnp.uint32),
        digits,
        _chain_addresses(params, position, compressed=tweak.compressed_address),
    )


def pk_from_sig(
    tweak: TweakableHash,
    params: WotsParams,
    signatures: ArrayLike,
    messages: ArrayLike,
    pk_seed: ArrayLike,
    position: WotsPosition,
    compress: Compress,
) -> Array:
    """`wots_pkFromSig` — Algorithm 8, for a batch. `[P, len, n]` -> `[P, n]`.

    The operation verification actually runs, so it takes many signatures: each
    with its own message, its own key pair, and — when the batch spans more than
    one public key — its own `pk_seed`. A signature that stopped its chains
    anywhere other than its message's digits arrives at a different public key,
    which is what the caller compares against a root it already trusts.
    """
    keys = position.count
    digits = message_digits(params, messages).reshape(-1)
    rows = fnp.asarray(signatures, dtype=fnp.uint8).reshape(keys * params.len, params.n)
    ends = chain(
        tweak,
        # One key pair is `len` chains, so a per-entry seed repeats `len` times to
        # match the key-pair-major row order `_chain_addresses` lays out.
        repeat_per_entry(pk_seed, params.len),
        rows,
        digits,
        (params.w - 1) - digits,
        _chain_addresses(params, position, compressed=tweak.compressed_address),
    )
    return compress(ends.reshape(keys, params.len * params.n))


def fips205_compression(
    tweak: TweakableHash, pk_seed: ArrayLike, position: WotsPosition
) -> Compress:
    """FIPS 205's public-key compression: one `T_len` over each key pair's ends.

    RFC 8391 compresses with an L-tree instead, which is why this is a value a
    caller passes rather than something WOTS+ decides for itself.
    """
    layers, trees, key_pairs = _position_columns(position, 1)
    addresses = adrs.encode_batch(
        adrs.wots_pk(layers, trees, key_pairs), compressed=tweak.compressed_address
    )

    def compress(ends: Array) -> Array:
        return tweak.t(pk_seed, addresses, ends)

    return compress
