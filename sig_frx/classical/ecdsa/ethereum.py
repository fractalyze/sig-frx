# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Ethereum's ECDSA: the chain's conventions as a thin layer over the core.

This module names Ethereum freely — that is its whole job; the core beneath
it never does. What rides here is exactly the set of choices the chain made
above the curve operation. Every entry point takes the 32-byte Keccak-256
digest, because what Ethereum signs is the Keccak of an encoding — RLP, a
typed-transaction payload, EIP-191 framing — the caller already holds.
Signatures cross as `(r ‖ s, v)`: v carries the recovery id's parity in the
legacy `{27, 28}` form or EIP-155's `chain_id·2 + 35 + parity` form. Signing
is low-S; verification-by-recovery rejects a high `s` (EIP-2) and a v from
another chain. The address is the last 20 bytes of the public key's
Keccak-256.

Nonces are RFC 6979 under HMAC-SHA256 — the pairing libsecp256k1's default
nonce function fixed for the ecosystem, severed from the digest's own hash on
purpose (`core.sign_digest_recoverable` owns the reasoning). The EIP-155
example reproducing byte for byte is what holds this module to that
convention end to end.

## EIP-712 is out of scope, by decision

Typed-data hashing is an encoding standard — how to fold structured data into
a digest — with a signature bolted on its end. Everything after the digest is
`personal_message_digest`'s sibling and already served here; the encoder
earns its own module the day a consumer needs one. Recorded so the boundary
is a choice, not an omission (fractalyze/sig-frx#31).
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from frx.typing import ArrayLike

from sig_frx import hashes
from sig_frx.classical import weierstrass
from sig_frx.classical.ecdsa import core

# The engine: secp256k1, low-S at signing (EIP-2's signing half). The
# MessageHash record serves construction only — every call below is on the
# digest-level surface, which reads neither of its faces.
_SCHEME = core.Ecdsa(weierstrass.SECP256K1, core.SHA256, low_s=True)

# EIP-155: v = chain_id·2 + 35 + parity; {27, 28} is the pre-fork form the
# fork kept valid.
_LEGACY_BASE = 27
_CHAIN_BASE = 35


def v_encode(recovery_id: int, chain_id: int | None = None) -> int:
    """The recovery id as Ethereum's v — legacy when `chain_id` is `None`.

    Rejects the `x(R) >= n` ids rather than folding them: v carries one
    parity bit, so a signature that draws one (a ~2^-127 event on this
    curve) cannot ride Ethereum's wire at all.
    """
    if recovery_id not in (0, 1):
        raise ValueError("Ethereum's v encodes a parity bit: recovery id 0 or 1")
    if chain_id is None:
        return _LEGACY_BASE + recovery_id
    return chain_id * 2 + _CHAIN_BASE + recovery_id


def v_decode(v: int) -> tuple[int, int | None]:
    """v back to `(recovery_id, chain_id)`, `None` marking the legacy form.

    `{27, 28}` and `>= 35` are the two published shapes; everything under and
    between them is no encoding, so it raises rather than guessing.
    """
    if v in (_LEGACY_BASE, _LEGACY_BASE + 1):
        return v - _LEGACY_BASE, None
    if v >= _CHAIN_BASE:
        return (v - _CHAIN_BASE) % 2, (v - _CHAIN_BASE) // 2
    raise ValueError(f"v = {v} is neither legacy {{27, 28}} nor EIP-155")


def sign(
    secret_key: ArrayLike, digest: ArrayLike, *, chain_id: int | None = None
) -> tuple[Any, int]:
    """One low-S signature over a Keccak-256 digest: `(r ‖ s, v)`.

    `chain_id` picks v's form and nothing else — (r, s) never see it. The
    replay protection is in the digest: an EIP-155 transaction's signing data
    already commits to the chain.
    """
    signature, recovery_id = _SCHEME.sign_digest_recoverable(
        secret_key, digest, nonce_hash=hashlib.sha256
    )
    return signature, v_encode(recovery_id, chain_id)


def recover_address(
    digest: ArrayLike,
    signature: ArrayLike,
    v: ArrayLike,
    *,
    chain_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The sending addresses of a batch: `(uint8[B, 20], bool[B])`.

    Ethereum verifies by recovering — the wire carries no public key — so
    this is the variant's whole verification surface, and the chain's
    policies gate each entry before the curve runs: a high `s` is rejected
    (EIP-2; the core accepts both halves, the chain must not), and an
    EIP-155 v must name this verifier's `chain_id` — a `None` verifier is
    pre-fork and takes legacy v only. Rejected entries clear the verdict and
    zero the address row, per the core's wire-data rule.
    """
    signature = np.asarray(signature, dtype=np.uint8)
    ids, v_ok = [], []
    for value in np.asarray(v):
        try:
            recovery_id, encoded_chain = v_decode(int(value))
        except ValueError:
            ids.append(0)
            v_ok.append(False)
            continue
        ids.append(recovery_id)
        v_ok.append(encoded_chain is None or encoded_chain == chain_id)
    keys, ok = _SCHEME.recover_digest(digest, signature, np.array(ids))
    low = np.asarray(core.is_low_s(_SCHEME.curve, signature))  # EIP-2
    ok = ok & np.array(v_ok, dtype=bool) & low
    addresses = np.where(ok[..., None], address_from_key(keys), np.uint8(0))
    return addresses, ok


def address_from_key(public_key: ArrayLike) -> np.ndarray:
    """`[..., 65]` uncompressed keys to `[..., 20]` addresses.

    The last 20 bytes of Keccak-256 over `X ‖ Y` — the `04` header is not
    part of what Ethereum hashes. Bytes, not EIP-55 hex: checksumming is a
    display encoding, and this layer stays in wire values.
    """
    key = np.asarray(public_key, dtype=np.uint8)
    # The device sponge takes exactly [B, L]; leading axes fold into B and
    # come back after.
    body = key[..., 1:].reshape(-1, 64)
    digest = np.asarray(hashes.keccak256(body).digest(body))
    return digest[..., 12:].reshape(key.shape[:-1] + (20,))


def personal_message_digest(message: ArrayLike) -> np.ndarray:
    """EIP-191 version 0x45: Keccak-256 of the personal-message framing.

    `"\\x19Ethereum Signed Message:\\n" ‖ len ‖ message`, the length in
    decimal ASCII. The `0x19` byte is the point of the framing: no
    transaction RLP can start there, so a signed personal message can never
    double as a signed transaction.
    """
    data = np.asarray(message, dtype=np.uint8).reshape(-1)
    prefix = b"\x19Ethereum Signed Message:\n" + str(data.shape[0]).encode()
    framed = np.concatenate([np.frombuffer(prefix, dtype=np.uint8), data])[None]
    return np.asarray(hashes.keccak256(framed).digest(framed))[0]
