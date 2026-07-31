# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""What the RFC 8391 reference implementation computes, for the two runnable sets.

RFC 8391 publishes no test vectors — its appendices are XDR format definitions,
and §7 says only that "a reference implementation in C is available". The
validation program has none either: it covers LMS and not XMSS. So the gate is
that reference implementation, which is what §7 points at, and the provenance is a
commit plus a program rather than a published file:

    XMSS/xmss-reference at 171ccbd26f098542a67eb5d2b128281c80bd71a6

The fixture is the reference's own `test/vectors.c`: `sk_seed[i] = i`,
`pub_seed[i] = 2i`, `m[i] = 3i`, `addr[i] = 500000000·i` and
`addr2[i] = 400000000·i`, plus `in32[i] = i + 7` and `twoblocks[i] = i + 1` for the
primitives. **Those addresses are deliberately garbage** — every word is non-zero,
including the type word, which is why they pin the no-normalization rule
`rfc8391_adrs` is built around.

`WOTS_PK`, `WOTS_SIG` and `LTREE_LEAF` are `SHAKE128(artifact, 10)`, the framing
`test/vectors.c` prints, and their values are the WOTS+ rows for OIDs 1 and 13 in
the table on fractalyze/sig-frx#16. Everything else is the artifact in full,
because those are small and a full-byte comparison localizes a failure that a
digest only detects. The program that produces all of it is on
fractalyze/sig-frx#57.

Two traps this file exists to record, both of which cost a session's worth of
debugging when they were guessed at instead:

- **`test/vectors.c` calls `gen_leaf_wots(params, leaf, sk_seed, pub_seed, addr,
  addr2)`**, and the parameter order is `(ltree_addr, ots_addr)`. So the leaf whose
  digest gates this is compressed from the WOTS+ public key at **addr2** under an
  L-tree at **addr** — the opposite assignment from the one the two addresses'
  names suggest. Swapping them yields a perfectly self-consistent leaf that matches
  no published digest.
- **`padding_len` is not `n`.** OID 1 pads to 32 bytes at `n = 32`; OID 13 pads to
  **4** at `n = 24`. Every value below differs between the two sets for that reason
  alone, which is what makes OID 13 worth running.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReferenceVectors:
    """Every value the reference computes for one parameter set, at the fixture.

    `addr` and `addr2` are the fixture's two addresses as `Adrs` field values: the
    tree address spans words 1 and 2, so it is the 64-bit value those two words
    make. Everything else is bytes.
    """

    oid: int
    n: int
    padding_len: int
    wots_len: int
    addr_bytes: bytes
    addr2_bytes: bytes
    prf: bytes
    prf_keygen: bytes
    thash_f: bytes
    thash_h: bytes
    hash_message: bytes
    ltree_leaf: bytes
    digest_wots_pk: bytes
    digest_wots_sig: bytes
    digest_ltree_leaf: bytes


def _fixture_words(step: int) -> list[int]:
    """`addr[i] = step·i` — the eight words of a fixture address."""
    return [step * i for i in range(8)]


# The two fixture addresses, as their eight raw words.
ADDR_WORDS = _fixture_words(500000000)
ADDR2_WORDS = _fixture_words(400000000)


def fixture_bytes(length: int, start: int = 0, step: int = 1) -> np.ndarray:
    """`out[i] = start + step·i` truncated to a byte — how every seed is filled."""
    return np.array([(start + step * i) & 0xFF for i in range(length)], dtype=np.uint8)


# The `hash_message` row signs a three-byte message under `R = pub_seed`,
# `root = sk_seed` and this index, which is wide enough to exercise every byte of
# the 64-bit value the reference passes.
HASH_MESSAGE_INDEX = 0x0102030405060708
HASH_MESSAGE_BODY = bytes.fromhex("aabbcc")


REFERENCE: dict[int, ReferenceVectors] = {
    0x01: ReferenceVectors(
        oid=0x01,
        n=32,
        padding_len=32,
        wots_len=67,
        addr_bytes=bytes.fromhex(
            "000000001dcd65003b9aca0059682f00773594009502f900b2d05e00d09dc300"
        ),
        addr2_bytes=bytes.fromhex(
            "0000000017d784002faf080047868c005f5e1000773594008f0d1800a6e49c00"
        ),
        prf=bytes.fromhex(
            "1de51bd88c0b79f5a89b724d41320d968bce47c1571928c509855b927c473daa"
        ),
        prf_keygen=bytes.fromhex(
            "ccf337abaa59ebc9a853729505da7b8ad74c483994b3b0b6c6b59a3533a5f11d"
        ),
        thash_f=bytes.fromhex(
            "13b6766d48b181260d65d0926ff667380e09d52d92482e16f410bc28753aac54"
        ),
        thash_h=bytes.fromhex(
            "f74926cb354f8e1776e1660f118f073eaba0f907593880b79ad6b7124123e7eb"
        ),
        hash_message=bytes.fromhex(
            "95557452ce9075bcdfbcfe7e09c3f74fca5c0ae824203e66257c5c157b355c2f"
        ),
        ltree_leaf=bytes.fromhex(
            "2f12f321f01c36805fdc61b012283f69bc0b035936ea7db49ef6385ea3029728"
        ),
        digest_wots_pk=bytes.fromhex("a5df5a7785a48961552e"),
        digest_wots_sig=bytes.fromhex("4443fb313e5b0c2e8bec"),
        digest_ltree_leaf=bytes.fromhex("fc27066a9b31c0069597"),
    ),
    0x0D: ReferenceVectors(
        oid=0x0D,
        n=24,
        padding_len=4,
        wots_len=51,
        addr_bytes=bytes.fromhex(
            "000000001dcd65003b9aca0059682f00773594009502f900b2d05e00d09dc300"
        ),
        addr2_bytes=bytes.fromhex(
            "0000000017d784002faf080047868c005f5e1000773594008f0d1800a6e49c00"
        ),
        prf=bytes.fromhex("ef2d6f2dd9830b76483a61b57af95b2aecd1711933c982ed"),
        prf_keygen=bytes.fromhex("388ca8b70eca7073715b485ad858f9962fcfced45c9afa26"),
        thash_f=bytes.fromhex("bf1e19df3ea6dbd9f25a086451007bd07055b277be6a9dcb"),
        thash_h=bytes.fromhex("d3806e965de28fbd1bcd057f7aa7324be49a77e808c54d31"),
        hash_message=bytes.fromhex("05e8bf47f266d24f270f77769c09c2aff1f01bb8d6e83ffa"),
        ltree_leaf=bytes.fromhex("c6d49557f4a72d8eeec2592bd5c3acde2e603230dd8189dd"),
        digest_wots_pk=bytes.fromhex("adbec5b9ba94bff3447d"),
        digest_wots_sig=bytes.fromhex("b32683d5888df51aa074"),
        digest_ltree_leaf=bytes.fromhex("58eb225e44f38082b356"),
    ),
}
