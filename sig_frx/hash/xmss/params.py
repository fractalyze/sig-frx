# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The XMSS and XMSS-MT parameter sets — RFC 8391 §5.3 and §5.4, keyed by OID.

A parameter set is a table row, not an implementation: every set below runs the
same constructions over a different core hash, a different output length, a
different height and a different number of layers. Sets are keyed by OID rather
than by name because the OID is what a public key carries in its first four bytes,
so it is what a parser has in hand.

**Two tables, because the OID space is per-variant.** OID 2 names
`XMSS-SHA2_16_256` in one and `XMSSMT-SHA2_20/4_256` in the other, and the
reference implementation keeps `xmss_parse_oid` and `xmssmt_parse_oid` apart for
exactly that reason. A single table keyed by OID would silently resolve half of
them to the wrong set.

**The padding length is not `n`.** §5.1's domain separators are `toByte(c, padlen)`
prefixes, and `padlen` is a property of the parameter set: 32 bytes at `n = 32`,
64 at `n = 64`, and **4** at `n = 24`. An implementation that padded to `n` would
pass every `n = 32` set and fail only the 192-bit ones, which is why those are
worth carrying rather than skipping. The 192-bit sets come from NIST SP 800-208,
which registered OIDs 13 through 21 against the same §5.1 constructions.

Every set fixes `w = 16`, so the WOTS+ lengths follow from `n` alone and are
derived here rather than restated — the formulas are §3.1.1's, which
`wots.WotsParams` already carries because FIPS 205 §5 defines them identically.

Only the SHA-2 sets at `n = 24` and `n = 32` are constructible today: `n = 64`
needs a SHA-512 `ByteHash` and every SHAKE set needs Keccak, neither of which
hash-frx has yet. The rows are all here anyway, because the table is the
standard's and a missing row would read as a set that does not exist.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from functools import cached_property

from sig_frx.hash import wots


class CoreHash(enum.Enum):
    """Which hash a parameter set's `core_hash` is — §5.1 and the OID registry.

    Carried so that a family cannot be built for a set over a hash the set does not
    name. The `ByteHash` seam deliberately does not identify its implementation, so
    this is the only thing that can tell a SHAKE128 row from a SHA-2 one.
    """

    SHA2 = "SHA-2"
    SHAKE128 = "SHAKE128"
    SHAKE256 = "SHAKE256"


@dataclass(frozen=True)
class XmssParams:
    """One row of the §5.3 or §5.4 table.

    `n` is both the security parameter and what `core_hash` truncates to: the SHA-2
    sets at `n = 24` are SHA-256 cut to 24 bytes rather than a different hash.

    One row shape covers both variants because RFC 8391 makes XMSS the `d = 1` case
    of XMSS-MT and its own reference implementation says so structurally —
    `xmss_core_keypair` and `xmss_core_sign_open` both just call the multi-tree
    routine. `layers` is the column that differs, and `index_bytes` is the one
    place the two do not quite collapse.
    """

    oid: int
    name: str
    core_hash: CoreHash
    n: int
    padding_len: int
    height: int  # `h`, the total height; one key signs `2^h` messages
    layers: int = 1  # `d`, the layers of trees the height is split across

    def __post_init__(self) -> None:
        if self.height % self.layers:
            raise ValueError(
                f"the {self.layers} layers each hold one tree of height h / d, so "
                f"d must divide h = {self.height}"
            )

    @cached_property
    def tree_height(self) -> int:
        """`h'` — the height of one tree. Equal to `h` for single-tree XMSS."""
        return self.height // self.layers

    @cached_property
    def index_bytes(self) -> int:
        """How many bytes a signature spends on its index.

        The one place the two variants do not collapse: §4.1.8 gives XMSS a fixed
        four bytes whatever `h` is, while §4.2.2 rounds `h` up to a byte — so a
        height-20 XMSS-MT signature carries three where a height-10 XMSS signature
        carries four. Reading it off `h` for both would mis-parse every XMSS
        signature by two bytes.
        """
        return 4 if self.layers == 1 else -(-self.height // 8)

    @cached_property
    def wots(self) -> wots.WotsParams:
        """The WOTS+ lengths `n` and `w = 16` imply — §3.1.1."""
        return wots.WotsParams(n=self.n)


XMSS_PARAMETER_SETS: dict[int, XmssParams] = {
    params.oid: params
    for params in (
        # OID, name, core hash, n, padding_len, h — the table's own column order.
        XmssParams(0x01, "XMSS-SHA2_10_256", CoreHash.SHA2, 32, 32, 10),
        XmssParams(0x02, "XMSS-SHA2_16_256", CoreHash.SHA2, 32, 32, 16),
        XmssParams(0x03, "XMSS-SHA2_20_256", CoreHash.SHA2, 32, 32, 20),
        XmssParams(0x04, "XMSS-SHA2_10_512", CoreHash.SHA2, 64, 64, 10),
        XmssParams(0x05, "XMSS-SHA2_16_512", CoreHash.SHA2, 64, 64, 16),
        XmssParams(0x06, "XMSS-SHA2_20_512", CoreHash.SHA2, 64, 64, 20),
        XmssParams(0x07, "XMSS-SHAKE_10_256", CoreHash.SHAKE128, 32, 32, 10),
        XmssParams(0x08, "XMSS-SHAKE_16_256", CoreHash.SHAKE128, 32, 32, 16),
        XmssParams(0x09, "XMSS-SHAKE_20_256", CoreHash.SHAKE128, 32, 32, 20),
        XmssParams(0x0A, "XMSS-SHAKE_10_512", CoreHash.SHAKE256, 64, 64, 10),
        XmssParams(0x0B, "XMSS-SHAKE_16_512", CoreHash.SHAKE256, 64, 64, 16),
        XmssParams(0x0C, "XMSS-SHAKE_20_512", CoreHash.SHAKE256, 64, 64, 20),
        XmssParams(0x0D, "XMSS-SHA2_10_192", CoreHash.SHA2, 24, 4, 10),
        XmssParams(0x0E, "XMSS-SHA2_16_192", CoreHash.SHA2, 24, 4, 16),
        XmssParams(0x0F, "XMSS-SHA2_20_192", CoreHash.SHA2, 24, 4, 20),
        XmssParams(0x10, "XMSS-SHAKE256_10_256", CoreHash.SHAKE256, 32, 32, 10),
        XmssParams(0x11, "XMSS-SHAKE256_16_256", CoreHash.SHAKE256, 32, 32, 16),
        XmssParams(0x12, "XMSS-SHAKE256_20_256", CoreHash.SHAKE256, 32, 32, 20),
        XmssParams(0x13, "XMSS-SHAKE256_10_192", CoreHash.SHAKE256, 24, 4, 10),
        XmssParams(0x14, "XMSS-SHAKE256_16_192", CoreHash.SHAKE256, 24, 4, 16),
        XmssParams(0x15, "XMSS-SHAKE256_20_192", CoreHash.SHAKE256, 24, 4, 20),
    )
}

XMSSMT_PARAMETER_SETS: dict[int, XmssParams] = {
    params.oid: params
    for params in (
        # OID, name, core hash, n, padding_len, h, d — §5.4's column order. The
        # OID space is XMSS-MT's own: OID 2 here is `XMSSMT-SHA2_20/4_256`, which
        # is a different set from XMSS's OID 2, so the two tables never merge.
        XmssParams(0x01, "XMSSMT-SHA2_20/2_256", CoreHash.SHA2, 32, 32, 20, 2),
        XmssParams(0x02, "XMSSMT-SHA2_20/4_256", CoreHash.SHA2, 32, 32, 20, 4),
        XmssParams(0x03, "XMSSMT-SHA2_40/2_256", CoreHash.SHA2, 32, 32, 40, 2),
        XmssParams(0x04, "XMSSMT-SHA2_40/4_256", CoreHash.SHA2, 32, 32, 40, 4),
        XmssParams(0x05, "XMSSMT-SHA2_40/8_256", CoreHash.SHA2, 32, 32, 40, 8),
        XmssParams(0x06, "XMSSMT-SHA2_60/3_256", CoreHash.SHA2, 32, 32, 60, 3),
        XmssParams(0x07, "XMSSMT-SHA2_60/6_256", CoreHash.SHA2, 32, 32, 60, 6),
        XmssParams(0x08, "XMSSMT-SHA2_60/12_256", CoreHash.SHA2, 32, 32, 60, 12),
        XmssParams(0x09, "XMSSMT-SHA2_20/2_512", CoreHash.SHA2, 64, 64, 20, 2),
        XmssParams(0x0A, "XMSSMT-SHA2_20/4_512", CoreHash.SHA2, 64, 64, 20, 4),
        XmssParams(0x0B, "XMSSMT-SHA2_40/2_512", CoreHash.SHA2, 64, 64, 40, 2),
        XmssParams(0x0C, "XMSSMT-SHA2_40/4_512", CoreHash.SHA2, 64, 64, 40, 4),
        XmssParams(0x0D, "XMSSMT-SHA2_40/8_512", CoreHash.SHA2, 64, 64, 40, 8),
        XmssParams(0x0E, "XMSSMT-SHA2_60/3_512", CoreHash.SHA2, 64, 64, 60, 3),
        XmssParams(0x0F, "XMSSMT-SHA2_60/6_512", CoreHash.SHA2, 64, 64, 60, 6),
        XmssParams(0x10, "XMSSMT-SHA2_60/12_512", CoreHash.SHA2, 64, 64, 60, 12),
        XmssParams(0x11, "XMSSMT-SHAKE_20/2_256", CoreHash.SHAKE128, 32, 32, 20, 2),
        XmssParams(0x12, "XMSSMT-SHAKE_20/4_256", CoreHash.SHAKE128, 32, 32, 20, 4),
        XmssParams(0x13, "XMSSMT-SHAKE_40/2_256", CoreHash.SHAKE128, 32, 32, 40, 2),
        XmssParams(0x14, "XMSSMT-SHAKE_40/4_256", CoreHash.SHAKE128, 32, 32, 40, 4),
        XmssParams(0x15, "XMSSMT-SHAKE_40/8_256", CoreHash.SHAKE128, 32, 32, 40, 8),
        XmssParams(0x16, "XMSSMT-SHAKE_60/3_256", CoreHash.SHAKE128, 32, 32, 60, 3),
        XmssParams(0x17, "XMSSMT-SHAKE_60/6_256", CoreHash.SHAKE128, 32, 32, 60, 6),
        XmssParams(0x18, "XMSSMT-SHAKE_60/12_256", CoreHash.SHAKE128, 32, 32, 60, 12),
        XmssParams(0x19, "XMSSMT-SHAKE_20/2_512", CoreHash.SHAKE256, 64, 64, 20, 2),
        XmssParams(0x1A, "XMSSMT-SHAKE_20/4_512", CoreHash.SHAKE256, 64, 64, 20, 4),
        XmssParams(0x1B, "XMSSMT-SHAKE_40/2_512", CoreHash.SHAKE256, 64, 64, 40, 2),
        XmssParams(0x1C, "XMSSMT-SHAKE_40/4_512", CoreHash.SHAKE256, 64, 64, 40, 4),
        XmssParams(0x1D, "XMSSMT-SHAKE_40/8_512", CoreHash.SHAKE256, 64, 64, 40, 8),
        XmssParams(0x1E, "XMSSMT-SHAKE_60/3_512", CoreHash.SHAKE256, 64, 64, 60, 3),
        XmssParams(0x1F, "XMSSMT-SHAKE_60/6_512", CoreHash.SHAKE256, 64, 64, 60, 6),
        XmssParams(0x20, "XMSSMT-SHAKE_60/12_512", CoreHash.SHAKE256, 64, 64, 60, 12),
        XmssParams(0x21, "XMSSMT-SHA2_20/2_192", CoreHash.SHA2, 24, 4, 20, 2),
        XmssParams(0x22, "XMSSMT-SHA2_20/4_192", CoreHash.SHA2, 24, 4, 20, 4),
        XmssParams(0x23, "XMSSMT-SHA2_40/2_192", CoreHash.SHA2, 24, 4, 40, 2),
        XmssParams(0x24, "XMSSMT-SHA2_40/4_192", CoreHash.SHA2, 24, 4, 40, 4),
        XmssParams(0x25, "XMSSMT-SHA2_40/8_192", CoreHash.SHA2, 24, 4, 40, 8),
        XmssParams(0x26, "XMSSMT-SHA2_60/3_192", CoreHash.SHA2, 24, 4, 60, 3),
        XmssParams(0x27, "XMSSMT-SHA2_60/6_192", CoreHash.SHA2, 24, 4, 60, 6),
        XmssParams(0x28, "XMSSMT-SHA2_60/12_192", CoreHash.SHA2, 24, 4, 60, 12),
        XmssParams(0x29, "XMSSMT-SHAKE256_20/2_256", CoreHash.SHAKE256, 32, 32, 20, 2),
        XmssParams(0x2A, "XMSSMT-SHAKE256_20/4_256", CoreHash.SHAKE256, 32, 32, 20, 4),
        XmssParams(0x2B, "XMSSMT-SHAKE256_40/2_256", CoreHash.SHAKE256, 32, 32, 40, 2),
        XmssParams(0x2C, "XMSSMT-SHAKE256_40/4_256", CoreHash.SHAKE256, 32, 32, 40, 4),
        XmssParams(0x2D, "XMSSMT-SHAKE256_40/8_256", CoreHash.SHAKE256, 32, 32, 40, 8),
        XmssParams(0x2E, "XMSSMT-SHAKE256_60/3_256", CoreHash.SHAKE256, 32, 32, 60, 3),
        XmssParams(0x2F, "XMSSMT-SHAKE256_60/6_256", CoreHash.SHAKE256, 32, 32, 60, 6),
        XmssParams(
            0x30, "XMSSMT-SHAKE256_60/12_256", CoreHash.SHAKE256, 32, 32, 60, 12
        ),
        XmssParams(0x31, "XMSSMT-SHAKE256_20/2_192", CoreHash.SHAKE256, 24, 4, 20, 2),
        XmssParams(0x32, "XMSSMT-SHAKE256_20/4_192", CoreHash.SHAKE256, 24, 4, 20, 4),
        XmssParams(0x33, "XMSSMT-SHAKE256_40/2_192", CoreHash.SHAKE256, 24, 4, 40, 2),
        XmssParams(0x34, "XMSSMT-SHAKE256_40/4_192", CoreHash.SHAKE256, 24, 4, 40, 4),
        XmssParams(0x35, "XMSSMT-SHAKE256_40/8_192", CoreHash.SHAKE256, 24, 4, 40, 8),
        XmssParams(0x36, "XMSSMT-SHAKE256_60/3_192", CoreHash.SHAKE256, 24, 4, 60, 3),
        XmssParams(0x37, "XMSSMT-SHAKE256_60/6_192", CoreHash.SHAKE256, 24, 4, 60, 6),
        XmssParams(0x38, "XMSSMT-SHAKE256_60/12_192", CoreHash.SHAKE256, 24, 4, 60, 12),
    )
}
