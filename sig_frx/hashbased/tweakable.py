# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The SLH-DSA tweakable hash family — FIPS 205 §11.

Six functions every hash-based component calls: two PRFs, the message hash, the
one-way function `F`, the fixed-length hash `H`, and the variable-length `T_l`.
Each is tweaked by an address (`adrs`), so the same input hashed at two positions
in the hypertree gives two unrelated outputs.

A parameter-set family is a `ByteHash` from hash-frx plus the sizes — which is
what makes the SHA-2 and SHAKE sets one implementation rather than two, and is
the reason that seam exists. `Sha2TweakableHash` is the SHA-2 instantiation
(§11.2); a SHAKE instantiation (§11.1) is the same class shape over SHAKE256,
differing in how each function derives its digest rather than in what it is for.

Inputs and outputs are `uint8` arrays with a leading batch axis, because the
callers hash a whole WOTS+ chain step or a whole Merkle level at once. `pk_seed`
is shared across the batch and the address is not: one seed, many positions.
Addresses arrive already encoded (`adrs.encode_batch`) — they are structural, so
they are built on the host before any hashing starts.

`prf_msg` is the exception, taking one message rather than a batch: it is called
once per signature, on the signing path, which this repo does not put on the hot
path (see `docs/reference/security.md`).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike
from hash_frx.byte_hash import ByteHash


@runtime_checkable
class TweakableHash(Protocol):
    n: int  # security parameter and output length, in bytes
    m: int  # message-digest length `h_msg` produces, in bytes
    # Which address encoding this family tweaks with: the SHA-2 sets compress to
    # 22 bytes, the SHAKE sets use the full 32. A caller reads it to build its
    # addresses and never names the family.
    compressed_address: bool

    def prf(self, pk_seed: ArrayLike, sk_seed: ArrayLike, adrs: ArrayLike) -> Array:
        """Secret-key element from the seeds and a position: -> uint8 `[B, n]`."""
        ...

    def prf_msg(
        self, sk_prf: ArrayLike, opt_rand: ArrayLike, message: ArrayLike
    ) -> Array:
        """The per-signature randomizer `R`: -> uint8 `[n]`. One message."""
        ...

    def h_msg(
        self,
        randomizer: ArrayLike,
        pk_seed: ArrayLike,
        pk_root: ArrayLike,
        message: ArrayLike,
    ) -> Array:
        """The message digest a signature is over: -> uint8 `[B, m]`."""
        ...

    def f(self, pk_seed: ArrayLike, adrs: ArrayLike, m1: ArrayLike) -> Array:
        """One-way function on one n-byte block: -> uint8 `[B, n]`."""
        ...

    def h(self, pk_seed: ArrayLike, adrs: ArrayLike, m2: ArrayLike) -> Array:
        """Hash of two n-byte blocks — a Merkle parent: -> uint8 `[B, n]`."""
        ...

    def t(self, pk_seed: ArrayLike, adrs: ArrayLike, messages: ArrayLike) -> Array:
        """Hash of `l` n-byte blocks — a WOTS+ or FORS root: -> uint8 `[B, n]`."""
        ...


def _batched(value: ArrayLike, batch: int) -> Array:
    """Broadcast a shared `[k]` operand to `[B, k]`, or pass a `[B, k]` through."""
    array = fnp.asarray(value, dtype=fnp.uint8)
    if array.ndim == 1:
        return fnp.broadcast_to(array, (batch, array.shape[0]))
    return array


class Sha2TweakableHash:
    """The SHA-2 instantiation for security category 1 — FIPS 205 §11.2.1.

    That is SLH-DSA-SHA2-128s and -128f, which reach every function with SHA-256
    alone. Categories 3 and 5 (§11.2.2) keep SHA-256 for `PRF` and `F` but move
    `H`, `T_l` and `PRF_msg` to SHA-512 with `toByte(0, 128 − n)`, so they are a
    family over *two* hashes rather than one — a different constructor, not a
    different constant.

    `F`, `H`, `T_l` and `PRF` are the same construction here, differing only in
    what follows the address: `Trunc_n(SHA-256(PK.seed ‖ toByte(0, 64 − n) ‖
    ADRS^c ‖ <input>))`. They stay separate names because the callers and the
    standard name them separately, and because the SHAKE instantiation does not
    collapse them.

    The zero padding after `PK.seed` is not decoration: it pads the seed to
    exactly one 64-byte compression block, so an implementation may precompute
    that block's midstate once per key and resume it per call.
    """

    def __init__(
        self,
        byte_hash: ByteHash,
        *,
        n: int,
        m: int,
        block_size: int = 64,
    ) -> None:
        if byte_hash.digest_size < n:
            raise ValueError(
                f"n={n} exceeds the {byte_hash.digest_size}-byte digest it truncates"
            )
        self._byte_hash = byte_hash
        # The hash's compression-block size, which HMAC and the seed padding both
        # need. `ByteHash` carries the digest size but not this, so it is stated
        # by the parameter set rather than read off the hash.
        self._block_size = block_size
        self.n = n
        self.m = m
        self.compressed_address = True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sha2TweakableHash):
            return NotImplemented
        return (
            self._byte_hash == other._byte_hash
            and self._block_size == other._block_size
            and self.n == other.n
            and self.m == other.m
        )

    def __hash__(self) -> int:
        return hash((type(self), self._byte_hash, self._block_size, self.n, self.m))

    # -- the tweaked family ------------------------------------------------

    def prf(self, pk_seed: ArrayLike, sk_seed: ArrayLike, adrs: ArrayLike) -> Array:
        return self._tweak(pk_seed, adrs, sk_seed)

    def f(self, pk_seed: ArrayLike, adrs: ArrayLike, m1: ArrayLike) -> Array:
        return self._tweak(pk_seed, adrs, m1)

    def h(self, pk_seed: ArrayLike, adrs: ArrayLike, m2: ArrayLike) -> Array:
        return self._tweak(pk_seed, adrs, m2)

    def t(self, pk_seed: ArrayLike, adrs: ArrayLike, messages: ArrayLike) -> Array:
        return self._tweak(pk_seed, adrs, messages)

    def prf_msg(
        self, sk_prf: ArrayLike, opt_rand: ArrayLike, message: ArrayLike
    ) -> Array:
        payload = fnp.concatenate(
            [
                fnp.asarray(opt_rand, dtype=fnp.uint8),
                fnp.asarray(message, dtype=fnp.uint8),
            ]
        )
        return self._hmac(sk_prf, payload)[: self.n]

    def h_msg(
        self,
        randomizer: ArrayLike,
        pk_seed: ArrayLike,
        pk_root: ArrayLike,
        message: ArrayLike,
    ) -> Array:
        messages = fnp.asarray(message, dtype=fnp.uint8)
        if messages.ndim == 1:
            messages = messages[None, :]
        batch = messages.shape[0]
        randomizers = _batched(randomizer, batch)
        seeds = _batched(pk_seed, batch)
        inner = self._digest(
            fnp.concatenate(
                [randomizers, seeds, _batched(pk_root, batch), messages], axis=-1
            )
        )
        return self._mgf1(fnp.concatenate([randomizers, seeds, inner], axis=-1), self.m)

    # -- the constructions the family is built from ------------------------

    def _tweak(self, pk_seed: ArrayLike, adrs: ArrayLike, payload: ArrayLike) -> Array:
        payloads = fnp.asarray(payload, dtype=fnp.uint8)
        addresses = fnp.asarray(adrs, dtype=fnp.uint8)
        if addresses.ndim == 1:
            addresses = addresses[None, :]
        if payloads.ndim == 1:
            payloads = payloads[None, :]
        batch = addresses.shape[0]
        seeds = _batched(pk_seed, batch)
        padding = fnp.zeros((batch, self._block_size - self.n), dtype=fnp.uint8)
        return self._digest(
            fnp.concatenate([seeds, padding, addresses, payloads], axis=-1)
        )[:, : self.n]

    def _mgf1(self, seed: Array, length: int) -> Array:
        """MGF1 over the injected hash — RFC 8017 §B.2.1.

        `T = H(seed ‖ toByte(0, 4)) ‖ H(seed ‖ toByte(1, 4)) ‖ ...`, truncated.
        """
        batch = seed.shape[0]
        blocks = -(-length // self._byte_hash.digest_size)
        counters = [
            fnp.broadcast_to(
                fnp.asarray(np.frombuffer(c.to_bytes(4, "big"), dtype=np.uint8)),
                (batch, 4),
            )
            for c in range(blocks)
        ]
        return fnp.concatenate(
            [self._digest(fnp.concatenate([seed, c], axis=-1)) for c in counters],
            axis=-1,
        )[:, :length]

    def _hmac(self, key: ArrayLike, message: Array) -> Array:
        """HMAC over the injected hash — RFC 2104."""
        key_bytes = fnp.asarray(key, dtype=fnp.uint8)
        if key_bytes.shape[0] > self._block_size:
            key_bytes = self._digest(key_bytes[None, :])[0]
        padded = (
            fnp.zeros(self._block_size, dtype=fnp.uint8)
            .at[: key_bytes.shape[0]]
            .set(key_bytes)
        )
        inner = self._digest(
            fnp.concatenate([padded ^ 0x36, message])[None, :],
        )[0]
        return self._digest(fnp.concatenate([padded ^ 0x5C, inner])[None, :])[0]

    def _digest(self, messages: Array) -> Array:
        return fnp.asarray(self._byte_hash.digest(messages), dtype=fnp.uint8)
