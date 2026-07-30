# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The XMSS parameter sets — RFC 8391 §5.3, keyed by their OIDs.

A parameter set is a table row, not an implementation: every set below runs the
same constructions over a different core hash, a different output length and a
different tree height. Sets are keyed by OID rather than by name because the OID is
what an XMSS public key carries in its first four bytes, so it is what a parser
has in hand.

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

from sig_frx.hashbased import wots


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
    """One row of the §5.3 table.

    `n` is both the security parameter and what `core_hash` truncates to: the SHA-2
    sets at `n = 24` are SHA-256 cut to 24 bytes rather than a different hash.
    """

    oid: int
    name: str
    core_hash: CoreHash
    n: int
    padding_len: int
    height: int  # `h`, the height of the tree; one XMSS key signs `2^h` messages

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
