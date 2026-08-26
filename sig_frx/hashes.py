# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Which hash each scheme uses, named once.

Every hash hash-frx ships is a device row: `digest` takes a tracer and returns
an `Array`, over messages shaped `[B, L]` where a single message is `B = 1`.
There is no second implementation to choose between, so nothing here dispatches
— this module is the one place a scheme's hash is named, not a place a choice is
made.

**It used to be a dispatcher**, because hash-frx shipped a `hashlib` sibling
beside each device row and which one a call wanted was a property of the values
rather than of the scheme holding them. `keccak256` was the exception that had
only ever had one row, and it is now the shape of all of them: hash-frx retired
its host rows (hash-frx#324), so the namespace question has no second answer to
give for any hash.

**What that costs the concrete caller**, stated because it is a real cost and
not a wash: a caller that is not tracing now pays a device dispatch and a
compilation per distinct message length, where a host row cost neither. Where
that matters, reach for `hashlib` directly — ECDSA already does, which is what
`MessageHash.host_constructor` is
([`classical/ecdsa/core.py`](classical/ecdsa/core.py)): RFC 6979 requires the
message hash and the HMAC to be one `H`, and the signing path takes that face
rather than this one.
"""

from __future__ import annotations

from hash_frx import ByteHash, Keccak256, Sha256, Shake128, Shake256, Xof


def shake128(*values: object) -> Xof:
    """SHAKE128.

    ML-DSA's `G` — `ExpandA` and nothing else, since `G` is the standard's name
    for the 128-bit XOF and only the matrix is sampled from it.
    """
    del values  # one row exists; nothing to dispatch on
    return Shake128


def shake256(*values: object) -> Xof:
    """SHAKE256.

    ML-DSA's `H`, which is the rest of the scheme: the commitment hash, the two
    seed derivations, and the three samplers that are not `ExpandA`.
    """
    del values  # one row exists; nothing to dispatch on
    return Shake256


def sha256(*values: object) -> ByteHash:
    """SHA-256.

    ECDSA's message hash in the FIPS 186-5 pairing. Fixed-length, so this
    returns an instance where the XOFs above return a family awaiting an output
    length. The signing path's concrete face is `MessageHash.host_constructor`,
    not this.
    """
    del values  # one row exists; nothing to dispatch on
    return Sha256()


def keccak256(*values: object) -> ByteHash:
    """Keccak-256 — the pre-FIPS submission.

    Ethereum's address derivation and message framing are the consumers.

    """
    del values  # one row exists; nothing to dispatch on
    return Keccak256()
