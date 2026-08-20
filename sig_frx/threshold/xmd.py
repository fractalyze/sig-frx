# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""`expand_message_xmd` (RFC 9380 §5.3.1), for the hash-to-field ciphersuites.

RFC 9591's non-Edwards ciphersuites derive scalars as
`hash_to_field(m, 1)` — an `expand_message_xmd` of `L` bytes reduced modulo
the group order — rather than the Edwards suites' plain reduce-a-digest.
Host-only for the same reason the rest of the threshold package is: this is
protocol-side scalar derivation, bytes and integers.
"""

from __future__ import annotations

import hashlib


def expand_message_xmd(message: bytes, dst: bytes, length: int) -> bytes:
    """RFC 9380 §5.3.1 over SHA-256, transcribed for a Merkle-Damgård `H`.

    SHA-256 is fixed rather than injected: every hash-to-field ciphersuite
    this package ships uses it, and the parameter arrives with the first
    suite that does not.
    """
    digest_size = hashlib.sha256().digest_size
    block_size = hashlib.sha256().block_size
    ell = -(-length // digest_size)
    if ell > 255 or length > 65535 or len(dst) > 255:
        raise ValueError("expand_message_xmd parameters out of range")
    dst_prime = dst + len(dst).to_bytes(1, "big")
    msg_prime = (
        b"\x00" * block_size + message + length.to_bytes(2, "big") + b"\x00" + dst_prime
    )
    b0 = hashlib.sha256(msg_prime).digest()
    blocks = [hashlib.sha256(b0 + b"\x01" + dst_prime).digest()]
    for i in range(2, ell + 1):
        mixed = bytes(x ^ y for x, y in zip(b0, blocks[-1]))
        blocks.append(hashlib.sha256(mixed + i.to_bytes(1, "big") + dst_prime).digest())
    return b"".join(blocks)[:length]


def hash_to_scalar(message: bytes, dst: bytes, order: int, length: int) -> int:
    """RFC 9380 §5.2's `hash_to_field` at `m = 1`: expand, then reduce."""
    return int.from_bytes(expand_message_xmd(message, dst, length), "big") % order
