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

**Only the columns something here reads.** The preset upstream carries also
fixes `LOG_LIFETIME`, `HASH_LENGTH_FIELD_ELEMENTS`, `CAPACITY` and `MAX_TRIES`,
and none of them has a call site in this package yet — the tree, the tweakable
hash family and the signer's rejection loop are what read them, and each field
arrives with the slice that does
([`conventions.md`](../../../docs/reference/conventions.md#a-seam-field-ships-with-the-call-site-that-reads-it)).
A column carried early is a value nothing can disagree with, which is the shape
of error that rule exists to catch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Final

from sig_frx.hash.leansig.field import PRIME

TWEAK_PREFIX_MESSAGE: Final = 0x02
"""Separates the message hash from the chain and tree hashes — `TWEAK_PREFIX_MESSAGE`.

The chain (`0x00`) and tree (`0x01`) prefixes arrive with the family that hashes
with them; this is the one [`encoding.py`](encoding.py) packs into its tweak.
"""


@dataclass(frozen=True)
class LeanSigParams:
    """One preset's parameters, as far as the encoding pipeline reads them."""

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

    def __post_init__(self) -> None:
        # Upstream's own validator. The decode's uniformity argument rests on it:
        # `0 .. PRIME - 2` is exactly `BASE^Z` groups of `Q` consecutive integers,
        # so every quotient is equally likely and `PRIME - 1` is the one value
        # left over — which is why the abort exists at all.
        if self.quotient * self.base**self.digits_per_element != PRIME - 1:
            raise ValueError(
                f"Q * BASE^Z must equal PRIME - 1 = {PRIME - 1}, got "
                f"{self.quotient} * {self.base}^{self.digits_per_element}"
            )

    @cached_property
    def message_hash_length(self) -> int:
        """`MH_HASH_LENGTH_FIELD_ELEMENTS` — Poseidon outputs the decode consumes.

        `ceil(DIMENSION / Z)`, so the decode has at least `DIMENSION` digits to
        truncate to. At `PROD` that is 6 elements yielding 48 digits for 46
        chains; the last two are dropped.
        """
        return math.ceil(self.dimension / self.digits_per_element)


PROD: Final = LeanSigParams(
    dimension=46,
    base=8,
    digits_per_element=8,
    quotient=127,
    target_sum=200,
    parameter_length=5,
    tweak_length=2,
    message_length=9,
    randomness_length=7,
)
"""Upstream's `PROD_CONFIG` — what the pq-devnet series pins."""

TEST: Final = LeanSigParams(
    dimension=4,
    base=8,
    digits_per_element=8,
    quotient=127,
    target_sum=6,
    parameter_length=5,
    tweak_length=2,
    message_length=9,
    randomness_length=7,
)
"""Upstream's `TEST_CONFIG` — the same scheme at a codeword short enough to sign."""
