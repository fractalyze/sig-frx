# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Bitcoin's ECDSA: consensus strictness as a thin layer over the core.

This module names Bitcoin freely; the core beneath it never does. What rides
here is what the chain fixed above the curve operation: double-SHA-256
digests, strict DER signatures with the sighash-type byte validated alongside
(BIP-66), low-S signing and high-S rejection (BIP-62), and the two public-key
encodings. Transaction serialization — deciding which bytes get hashed — is
out of scope on purpose: entry points take the 32-byte digest, so the layer
is testable against signature encodings rather than a transaction parser
(fractalyze/sig-frx#32).

## BIP-66 strictness is consensus

A decoder that accepts one non-canonical encoding accepts signatures the
network rejects — the fork BIP-66 exists to prevent. `der_decode` is a
transcription of the BIP's `IsValidSignatureEncoding`, kept in its shape
(one check per line, the BIP's order) so review against the BIP is a diff,
and each check carries its own message so a test can violate exactly one
rule and name the rejection it expects.

## Host-path, because the wire format says so

DER is variable-length and self-delimiting, so parsing it is per-entry byte
work — the wire format forces the concrete path the way recovery's readback
does. The curve arithmetic underneath stays the core's batched
`verify_digest`; what loops here is only the codec.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from frx.typing import ArrayLike

from sig_frx import hashes
from sig_frx.classical import group, weierstrass
from sig_frx.classical.ecdsa import core

# The engine: secp256k1, low-S at signing (BIP-62's signing half). The
# MessageHash record serves construction only — every call below is on the
# digest-level surface. Nonces are RFC 6979 under HMAC-SHA256, as in the
# Ethereum variant: libsecp256k1's default nonce function, which here is
# also the RFC's own pairing, the digest family being SHA-256.
_SCHEME = core.Ecdsa(weierstrass.SECP256K1, core.SHA256, low_s=True)


def message_digest(data: ArrayLike) -> Any:
    """Double SHA-256 — `[B, L]` to `[B, 32]`, in `data`'s namespace.

    Bitcoin hashes everything it signs twice; which bytes arrive here is the
    caller's serialization, per the module's scope rule. Messages take the
    ByteHash rows' shape — a single message is `B = 1`.
    """
    first = hashes.sha256(data).digest(data)
    return hashes.sha256(first).digest(first)


def sign(secret_key: ArrayLike, digest: ArrayLike, *, sighash: int) -> np.ndarray:
    """One low-S signature over a digest, as strict DER ‖ sighash byte.

    Low-S is BIP-62's signing half — the network's standardness rule, so a
    signature emitted here is never the malleable twin relays drop. The
    sighash type is required, not defaulted: which parts of a transaction
    were hashed is a statement only the caller can make.
    """
    signature, _ = _SCHEME.sign_digest_recoverable(
        secret_key, digest, nonce_hash=hashlib.sha256
    )
    return der_encode(signature, sighash)


def verify(
    public_key: ArrayLike, digest: ArrayLike, signature: ArrayLike
) -> np.ndarray:
    """The batched verdict over strict-DER rows and either key form: `bool[B]`.

    `public_key` is `[B, 33]` compressed or `[B, 65]` uncompressed — one form
    per batch, since the rows are rectangular. Signature rows are the
    self-delimiting DER ‖ sighash blob zero-padded to a common width (the
    seam's padding rule); a nonzero byte past the declared end rejects, so
    padding cannot smuggle data. A high `s` rejects here (BIP-62): the core
    accepts both halves, the chain's standardness rule does not.
    """
    keys = np.asarray(public_key, dtype=np.uint8)
    if keys.shape[-1] == 33:
        keys, key_ok = decompress(keys)
    elif keys.shape[-1] == 65:
        key_ok = np.ones(keys.shape[:-1], dtype=bool)
    else:
        raise ValueError("a public key is 33 or 65 bytes")

    rows = np.asarray(signature, dtype=np.uint8)
    packed = np.zeros(rows.shape[:-1] + (64,), dtype=np.uint8)
    encoding_ok = []
    for i, row in enumerate(rows):
        try:
            packed[i], _ = der_decode_padded(row)
        except ValueError:
            encoding_ok.append(False)
            continue
        encoding_ok.append(True)
    verdict = np.asarray(_SCHEME.verify_digest(keys, digest, packed))
    low = np.asarray(core.is_low_s(_SCHEME.curve, packed))  # BIP-62
    return verdict & key_ok & np.array(encoding_ok, dtype=bool) & low


def der_encode(signature: ArrayLike, sighash: int) -> np.ndarray:
    """`r ‖ s` (64 bytes) as strict DER with the sighash byte appended.

    Emits the unique BIP-66-valid encoding: minimal big-endian integers, one
    `0x00` prefix exactly where a top bit demands it.
    """
    data = np.asarray(signature, dtype=np.uint8).reshape(-1)
    if data.shape[0] != 64:
        raise ValueError("a signature is 64 bytes of r ‖ s")
    if not 0 <= sighash <= 255:
        raise ValueError("the sighash type is one byte")
    body = _der_integer(data[:32].tobytes()) + _der_integer(data[32:].tobytes())
    blob = b"\x30" + bytes([len(body)]) + body + bytes([sighash])
    return np.frombuffer(blob, dtype=np.uint8).copy()


def _der_integer(value: bytes) -> bytes:
    stripped = value.lstrip(b"\x00") or b"\x00"
    if stripped[0] & 0x80:
        stripped = b"\x00" + stripped
    return b"\x02" + bytes([len(stripped)]) + stripped


def der_decode(signature: ArrayLike) -> tuple[np.ndarray, int]:
    """Strict DER ‖ sighash back to `(r ‖ s as 64 bytes, sighash)`.

    BIP-66's `IsValidSignatureEncoding`, check for check and in its order;
    each rejection names its rule. One bound is this module's, not the
    BIP's: an integer whose value needs more than 32 bytes cannot ride the
    `r ‖ s` wire form — every value it excludes is outside `[1, n-1]`, so no
    verifiable signature is lost, only encoded-but-hopeless ones.
    """
    sig = np.asarray(signature, dtype=np.uint8).reshape(-1).tobytes()
    if len(sig) < 9:
        raise ValueError("BIP-66: too short")
    if len(sig) > 73:
        raise ValueError("BIP-66: too long")
    if sig[0] != 0x30:
        raise ValueError("BIP-66: not a DER sequence")
    if sig[1] != len(sig) - 3:
        raise ValueError("BIP-66: wrong total length")
    len_r = sig[3]
    if 5 + len_r >= len(sig):
        raise ValueError("BIP-66: R overruns the signature")
    len_s = sig[5 + len_r]
    if len_r + len_s + 7 != len(sig):
        raise ValueError("BIP-66: wrong integer lengths")
    if sig[2] != 0x02:
        raise ValueError("BIP-66: R is not an integer")
    if len_r == 0:
        raise ValueError("BIP-66: R is empty")
    if sig[4] & 0x80:
        raise ValueError("BIP-66: R is negative")
    if len_r > 1 and sig[4] == 0x00 and not sig[5] & 0x80:
        raise ValueError("BIP-66: R is padded")
    if sig[len_r + 4] != 0x02:
        raise ValueError("BIP-66: S is not an integer")
    if len_s == 0:
        raise ValueError("BIP-66: S is empty")
    if sig[len_r + 6] & 0x80:
        raise ValueError("BIP-66: S is negative")
    if len_s > 1 and sig[len_r + 6] == 0x00 and not sig[len_r + 7] & 0x80:
        raise ValueError("BIP-66: S is padded")
    r = int.from_bytes(sig[4 : 4 + len_r], "big")
    s = int.from_bytes(sig[6 + len_r : 6 + len_r + len_s], "big")
    if max(r, s) >> 256:
        raise ValueError("an integer exceeds the r ‖ s wire form")
    packed = r.to_bytes(32, "big") + s.to_bytes(32, "big")
    return np.frombuffer(packed, dtype=np.uint8).copy(), sig[-1]


def der_decode_padded(signature: ArrayLike) -> tuple[np.ndarray, int]:
    """`der_decode` for a row zero-padded to a batch width.

    The declared length is trusted only as a slice bound — the slice then
    passes through the transcription like any other blob — and a nonzero
    byte past the declared end rejects, so padding cannot smuggle data. It
    lives beside `der_decode` so every byte-level DER decision stays in the
    codec's one home.
    """
    blob = np.asarray(signature, dtype=np.uint8).reshape(-1).tobytes()
    declared = blob[1] + 3 if len(blob) >= 2 else len(blob)
    if declared > len(blob) or any(blob[declared:]):
        raise ValueError("BIP-66: data past the declared end")
    return der_decode(np.frombuffer(blob[:declared], dtype=np.uint8))


def compress(public_key: ArrayLike) -> np.ndarray:
    """`[..., 65]` uncompressed keys to `[..., 33]`: `02 | parity(y) ‖ X`.

    Raises on a wrong header rather than a verdict: the uncompressed form is
    this stack's own key material (keygen, decompress), not wire data.
    """
    key = np.asarray(public_key, dtype=np.uint8)
    if key.shape[-1] != 65:
        raise ValueError("an uncompressed public key is 65 bytes")
    if np.any(key[..., 0] != 4):
        raise ValueError("not an uncompressed-point encoding")
    header = np.uint8(2) + (key[..., 64:] & np.uint8(1))
    return np.concatenate([header, key[..., 1:33]], axis=-1)


def decompress(public_key: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """`[..., 33]` compressed keys to `(uint8[..., 65], bool[...])`.

    Wire data, so failures — a header naming neither parity, an x outside
    the field, an x on no curve point — clear the verdict and zero the row,
    per the core's rule. The lift is the substrate's; the parity pick reads
    the root back, which is what keeps this codec on the host path.
    """
    key = np.asarray(public_key, dtype=np.uint8)
    if key.shape[-1] != 33:
        raise ValueError("a compressed public key is 33 bytes")
    curve = weierstrass.SECP256K1
    flat = key.reshape(-1, 33)
    x_bytes = flat[:, 1:]
    ok = ((flat[:, 0] == 2) | (flat[:, 0] == 3)) & group.bytes_below(
        np, x_bytes, curve.p, byteorder="big"
    )
    points, on_curve = weierstrass.lift_x_to_parity(
        curve, weierstrass.field_from_bytes(curve, x_bytes), flat[:, 0] & 1
    )
    ok = ok & np.asarray(on_curve)
    out = np.zeros((flat.shape[0], 65), dtype=np.uint8)
    for i, ((_, y), valid) in enumerate(zip(group.to_affine_ints(points), ok)):
        if valid:
            out[i, 0] = 4
            out[i, 1:33] = flat[i, 1:]
            out[i, 33:] = np.frombuffer(y.to_bytes(32, "big"), dtype=np.uint8)
    return out.reshape(key.shape[:-1] + (65,)), ok.reshape(key.shape[:-1])
