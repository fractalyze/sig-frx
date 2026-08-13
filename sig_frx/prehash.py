# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The pre-hash variants, whose framing both FIPS signature standards share.

FIPS 204 §5.4.1 and FIPS 205 §10.2.1 define the same operation over two different
schemes: hash the content first, then sign
`toByte(domain, 1) ‖ toByte(|ctx|, 1) ‖ ctx ‖ OID ‖ PH(M)` in place of the content.
Only the domain separator's value is the scheme's own, and
[`context.py`](context.py) already carries the framing the pure variants share, so
what is left to share is the record — the digest and the identifier that names it.

**The OID is inside what gets signed, so a pre-hash function is a value and not a
flag.** A signature says which function produced the digest it answers for, which
is what stops one from being reinterpreted under another. That makes the OID and
the hash one indivisible pair, which is what this record is; a variant that took
the function per call would be as many operations as it has cases behind one name.

**Where the constants come from, since only three of them are in a standard.**
FIPS 204 Algorithm 4 and FIPS 205 Algorithm 23 spell out SHA-256, SHA-512 and
SHAKE128 and then write `case …` — the rest are approved but unenumerated. Their
DER encodings are the NIST CSOR arc `2.16.840.1.101.3.4.2` (`nistAlgorithms`
`hashAlgs`), whose final component is the function's index in that arc, and the
validation program's generator is what fixes the two an arc cannot state: a XOF's
output length. It reads them from the same table FIPS 186-5 sets for RSA-PSS —
256 bits for SHAKE128 and 512 for SHAKE256 — which is
`ShaAttributes.xofSignatureAttributes` in ACVP-Server at the commit
[`MODULE.bazel`](../MODULE.bazel) pins its vectors to, reached from
`PreHashProperties` by the `isXofPss` flag. FIPS 204's own 256 bits for SHAKE128
agrees with it, which is the one overlap there is to check.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike
from hash_frx.byte_hash import ByteHash
from hash_frx.keccak.byte_hashes import Sha3_256, Sha3_512, Shake128, Shake256
from hash_frx.sha256 import Sha256

from sig_frx import context as ctx

# The `hashAlgs` arc, `2.16.840.1.101.3.4.2`, DER-encoded with its tag and length
# — the form both standards paste into the message. A function's own OID is this
# followed by its index in the arc, which is the byte each constant below adds.
_HASH_ALGS_ARC = bytes.fromhex("06096086480165030402")


def _oid(index: int) -> bytes:
    """The DER encoding of `2.16.840.1.101.3.4.2.index`, tag and length included."""
    return _HASH_ALGS_ARC + bytes([index])


# The five the constructors below reach. The indices are the arc's own ordering,
# and the three a standard states — SHA-256 at 1, SHA-512 at 3, SHAKE128 at 11 —
# are what pins the reading of the other two.
SHA2_256_OID = _oid(1)
SHA3_256_OID = _oid(8)
SHA3_512_OID = _oid(10)
SHAKE128_OID = _oid(11)
SHAKE256_OID = _oid(12)

# What a XOF is squeezed for when it stands in for a hash of a signed message.
_SHAKE128_DIGEST_SIZE = 32
_SHAKE256_DIGEST_SIZE = 64


@dataclass(frozen=True)
class PreHash:
    """A pre-hash function: the digest, and the OID that names it in the message.

    `oid` is the DER encoding of the function's object identifier, tag and length
    included, as both standards' `switch PH` writes it.
    """

    oid: bytes
    byte_hash: ByteHash

    def digest(self, messages: ArrayLike) -> Array:
        """`PH_M` — the pre-hash of the content. One message, or a batch of them."""
        values = fnp.asarray(messages, dtype=fnp.uint8)
        if values.ndim == 1:
            return fnp.asarray(self.byte_hash.digest(values[None, :]), dtype=fnp.uint8)[
                0
            ]
        return fnp.asarray(self.byte_hash.digest(values), dtype=fnp.uint8)

    def prefix(self, domain: int, context: ArrayLike | None) -> np.ndarray:
        """Everything the signed message carries ahead of `PH_M`.

        `domain` is the caller's, because the two standards number their
        separators independently — the same reason
        [`context.py`](context.py) takes it rather than naming one.
        """
        return np.concatenate(
            [ctx.prefix(domain, context), np.frombuffer(self.oid, dtype=np.uint8)]
        )


def sha2_256() -> PreHash:
    """SHA-256 — FIPS 204 Algorithm 4 line 12, FIPS 205 Algorithm 23 line 10."""
    return PreHash(oid=SHA2_256_OID, byte_hash=Sha256())


def sha3_256() -> PreHash:
    """SHA3-256 — approved by both standards, enumerated by neither."""
    return PreHash(oid=SHA3_256_OID, byte_hash=Sha3_256())


def sha3_512() -> PreHash:
    """SHA3-512 — approved by both standards, enumerated by neither."""
    return PreHash(oid=SHA3_512_OID, byte_hash=Sha3_512())


def shake128() -> PreHash:
    """SHAKE128 at 256 bits — FIPS 204 Algorithm 4 line 19 fixes the length."""
    return PreHash(oid=SHAKE128_OID, byte_hash=Shake128(_SHAKE128_DIGEST_SIZE))


def shake256() -> PreHash:
    """SHAKE256 at 512 bits — the length from the generator, per the module note.

    Twice SHAKE128's, which is the footnote to FIPS 204 §5.4 read forward: a
    digest signed at `λ` bits of classical collision strength has to be `2λ` bits
    long, and SHAKE256 is the one a category-5 parameter set can use.
    """
    return PreHash(oid=SHAKE256_OID, byte_hash=Shake256(_SHAKE256_DIGEST_SIZE))


# Every constructor above, under the name a published vector selects it by. The
# validation program's spelling rather than a standard's, because that is what a
# case carries and a mapping written per caller is how two of them come to
# disagree about which functions this repo can compute.
#
# What is *not* here is the strength pairing. A pre-hash weaker than `2λ` bits is
# below the level the parameter set claims (FIPS 204 §5.4 footnote 6), and the
# published sets pair the two freely — so it is a property of a deployment's
# choice rather than something an implementation may refuse, and a scheme that
# refused could not be gated on the vectors that exercise it.
BY_NAME: dict[str, Callable[[], PreHash]] = {
    "SHA2-256": sha2_256,
    "SHA3-256": sha3_256,
    "SHA3-512": sha3_512,
    "SHAKE-128": shake128,
    "SHAKE-256": shake256,
}
