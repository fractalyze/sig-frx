# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The two presets leanSig instantiates — leanSpec's `PROD_CONFIG` and `TEST_CONFIG`.

There is no RFC and no OID registry here, so unlike
[`xmss/params.py`](../xmss/params.py) this is not a table a public key indexes
into: it is two rows an implementation picks between, and which one a devnet
runs is a build-time fact rather than a wire one. `PROD` is what pq-devnet pins;
`TEST` shortens the lifetime and the codeword so key generation and signing —
`2^32` leaves and a rejection loop at the production preset — are cheap enough
to gate for real.

**The rows are the spec's, at the pinned commit.** Upstream states them in
`spec/crypto/xmss/constants.py` and the names below say which column each field
is, because upstream's are shouted (`DIMENSION`, `TWEAK_LENGTH_FIELD_ELEMENTS`)
and this repo's are not. Anyone re-deriving a signature size or a chain count
from the technical note will get different numbers: the note gives `v = 64`, and
the widely-quoted 3112-byte signature is that figure. `PROD` is `v = 46`.

**Only the columns something here reads.** `MAX_TRIES` is the last one upstream
carries that nothing here does — the signer's rejection loop is what reads it,
and it arrives with that slice. `HASH_LENGTH_FIELD_ELEMENTS` and `CAPACITY`
arrived exactly that way, with the tweakable hash family
([`tweakable.py`](tweakable.py)); `LOG_LIFETIME` arrived with the wire format
([`ssz.py`](ssz.py)), which sizes an authentication path by it. That is
[`conventions.md`](../../../docs/reference/conventions.md#generalize-a-component-when-its-second-consumer-arrives)
read from the data side: a column nothing reads is a number nothing can
disagree with, so a wrong one is found by the slice that finally uses it rather
than here. It is *not* the argument `xmss/params.py` makes for carrying rows it
cannot construct — that table is indexed by an OID off the wire, where a missing
row reads as a set that does not exist, and these two presets are picked at build
time.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Final

from sig_frx.hash.leansig.field import PRIME

TWEAK_PREFIX_CHAIN: Final = 0x00
"""A Winternitz chain step — `TWEAK_PREFIX_CHAIN`.

The low byte of a packed tweak, and the whole of what keeps three hashes at the
same position from colliding: a chain step, a Merkle node and a message hash all
compress a parameter and a tweak, and only this byte tells them apart.
"""

TWEAK_PREFIX_TREE: Final = 0x01
"""A Merkle node or leaf — `TWEAK_PREFIX_TREE`."""

TWEAK_PREFIX_MESSAGE: Final = 0x02
"""The message hash — the one [`encoding.py`](encoding.py) packs."""


@dataclass(frozen=True)
class LeanSigParams:
    """One preset's parameters, as far as anything in this package reads them."""

    log_lifetime: int
    """`LOG_LIFETIME`: how many slots one key covers, as a power of two — and so
    how many levels the Merkle tree has, which is what sizes an authentication
    path. Even, because the tree splits into a top and a bottom half of
    `log_lifetime / 2` levels each."""

    dimension: int
    """`DIMENSION`, the `v` of the papers: how many chains a signature commits to,
    and so how many digits a codeword has."""

    base: int
    """`BASE`, the Winternitz `w`: the alphabet a codeword digit is drawn from."""

    digits_per_element: int
    """`Z`: how many base-`base` digits the decode extracts from one field element."""

    quotient: int
    """`Q`: what the decode divides by, fixed by `Q * BASE^Z == PRIME - 1`."""

    target_sum: int
    """`TARGET_SUM`: the hypercube layer a codeword has to land on to be accepted."""

    parameter_length: int
    """`PARAMETER_LENGTH`: the public parameter, in field elements."""

    tweak_length: int
    """`TWEAK_LENGTH_FIELD_ELEMENTS`: a domain-separating tweak, in field elements."""

    message_length: int
    """`MESSAGE_LENGTH_FIELD_ELEMENTS`: the encoded message, in field elements."""

    randomness_length: int
    """`RAND_LENGTH_FIELD_ELEMENTS`: the per-attempt randomness, in field elements."""

    hash_length: int
    """`HASH_LENGTH_FIELD_ELEMENTS`: one digest, in field elements. The scheme's
    collision resistance is set here."""

    capacity: int
    """`CAPACITY`: the leaf sponge's capacity, in field elements. The sponge's
    security level is set here, and it is what the rate is `24 - capacity`."""

    def __post_init__(self) -> None:
        # Upstream's other validator, and the tree layout is what rests on it: the
        # lifetime splits into a top and a bottom half of equal height, so an odd
        # exponent has no split.
        if self.log_lifetime % 2 != 0:
            raise ValueError(f"LOG_LIFETIME must be even, got {self.log_lifetime}")

        # Upstream's own validator. The decode's uniformity argument rests on it:
        # `0 .. PRIME - 2` is exactly `BASE^Z` groups of `Q` consecutive integers,
        # so every quotient is equally likely and `PRIME - 1` is the one value
        # left over — which is why the abort exists at all.
        if self.decode_threshold != PRIME - 1:
            raise ValueError(
                f"Q * BASE^Z must equal PRIME - 1 = {PRIME - 1}, got "
                f"{self.quotient} * {self.base}^{self.digits_per_element}"
            )

    @cached_property
    def decode_threshold(self) -> int:
        """`Q * BASE^Z` — what the decode rejects at or above.

        The invariant above makes this `PRIME - 1`, so it fires on that single
        value. It is carried as the product rather than as the prime because that
        is what makes the decode a range check rather than a coincidence, and it
        lives here so the check and the invariant that justifies it cannot come
        to be spelled differently.
        """
        return self.quotient * self.base**self.digits_per_element

    @cached_property
    def message_hash_length(self) -> int:
        """`MH_HASH_LENGTH_FIELD_ELEMENTS` — Poseidon outputs the decode consumes.

        `ceil(DIMENSION / Z)`, so the decode has at least `DIMENSION` digits to
        truncate to. At `PROD` that is 6 elements yielding 48 digits for 46
        chains; the last two are dropped.

        The float-free ceiling is the one every parameter set here uses —
        [`wots.py`](../wots.py), [`xmss/params.py`](../xmss/params.py).
        """
        return -(-self.dimension // self.digits_per_element)


PROD: Final = LeanSigParams(
    log_lifetime=32,
    dimension=46,
    base=8,
    digits_per_element=8,
    quotient=127,
    target_sum=200,
    parameter_length=5,
    tweak_length=2,
    message_length=9,
    randomness_length=7,
    hash_length=8,
    capacity=9,
)
"""Upstream's `PROD_CONFIG` — what the pq-devnet series pins."""

TEST: Final = LeanSigParams(
    log_lifetime=8,
    dimension=4,
    base=8,
    digits_per_element=8,
    quotient=127,
    target_sum=6,
    parameter_length=5,
    tweak_length=2,
    message_length=9,
    randomness_length=7,
    hash_length=8,
    capacity=9,
)
"""Upstream's `TEST_CONFIG` — the same scheme at a codeword short enough to sign."""
