# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The XMSS hash family — RFC 8391 §5.1.

Five functions, and they differ from FIPS 205's family (`tweakable.py`) in what a
tweaked hash *is*. There, an address is prefix material: `F` hashes
`PK.seed ‖ pad ‖ ADRS ‖ M`. Here the address never reaches the hash at all — it is
run through `PRF` to produce a key and a bitmask, and the message is XORed with the
mask before hashing:

    F(SEED, ADRS, M) = core_hash( toByte(0, padlen) ‖ KEY ‖ (M XOR BM) )

with `KEY = PRF(SEED, ADRS ‖ keyAndMask=0)` and `BM = PRF(SEED, ADRS ‖ 1)`. `H`
compresses two blocks and so needs twice the mask, which it gets by concatenating
the `keyAndMask = 1` and `= 2` derivations in that order. Every call is therefore
two or three `PRF` evaluations plus one core hash, all of them batched: a WOTS+
chain step over `P · len` chains is four batched hashes, not `4 · P · len` calls.

**One class covers every parameter set**, unlike the FIPS 205 side where the SHAKE
instantiation is a different construction. RFC 8391 puts the whole difference in
`core_hash` — SHA-256, SHA-512, SHAKE128 or SHAKE256, truncated to `n` — so the
family is an injected `ByteHash` plus two integers. The 192-bit sets are exactly
that: SHA-256 truncated to 24 bytes, with a 4-byte padding rather than a 24-byte
one.

**The padding length is a parameter, not `n`.** `toByte(c, padlen)` is what
separates the five functions from each other, and `padlen` is 4 at `n = 24` where
it is 32 at `n = 32`. Hard-coding it to `n` passes every 256-bit vector.

Inputs and outputs are `uint8` arrays with a leading batch axis, the shape every
component above this one works in. Addresses arrive already encoded
(`adrs.encode_batch`) and stay on the host, because they are structural:
`f` and `h` rewrite their last word rather than asking a caller for three
addresses per position.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike
from hash_frx import ByteHash

from sig_frx.hash.xmss.adrs import with_key_and_mask
from sig_frx.hash.xmss.params import CoreHash, XmssParams

# §5.1's domain separators, the `c` of each function's `toByte(c, padlen)` prefix.
# They are what stops one function's output from being another's, so they are the
# first bytes of what gets hashed rather than metadata.
_PADDING_F = 0
_PADDING_H = 1
_PADDING_HASH = 2
_PADDING_PRF = 3
_PADDING_PRF_KEYGEN = 4

# `PRF` takes a 32-byte input, which is exactly one encoded address — the only
# thing §5.1 ever calls it on besides the signature index.
_PRF_INPUT_SIZE = 32


class Rfc8391Hashes:
    """The keyed-and-masked family over an injected core hash — §5.1.

    `n` is the output length every function truncates to, and `padding_len` the
    width of the domain separator each prefixes with. Both come off a parameter set
    (`params`); neither is derivable from the other.
    """

    def __init__(self, byte_hash: ByteHash, *, n: int, padding_len: int) -> None:
        if byte_hash.digest_size < n:
            raise ValueError(
                f"n={n} exceeds the {byte_hash.digest_size}-byte digest it truncates"
            )
        self._byte_hash = byte_hash
        self.n = n
        self.padding_len = padding_len

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rfc8391Hashes):
            return NotImplemented
        return (
            self._byte_hash == other._byte_hash
            and self.n == other.n
            and self.padding_len == other.padding_len
        )

    def __hash__(self) -> int:
        return hash((type(self), self._byte_hash, self.n, self.padding_len))

    # -- the family --------------------------------------------------------

    def prf(self, key: ArrayLike, message: ArrayLike) -> Array:
        """`PRF(KEY, M)` — a 32-byte input under an n-byte key: -> uint8 `[B, n]`.

        Called on an encoded address to derive a hash key or a bitmask, and on
        `toByte(idx, 32)` to derive a signature's randomizer. The key is `SEED` in
        the first case and `SK_PRF` in the second, so it is shared across the batch
        or given per row like any other operand.
        """
        inputs = _rows(message)
        if inputs.shape[-1] != _PRF_INPUT_SIZE:
            raise ValueError(
                f"PRF takes a {_PRF_INPUT_SIZE}-byte input, got {inputs.shape[-1]}"
            )
        return self._core(_PADDING_PRF, key, inputs)

    def prf_keygen(self, key: ArrayLike, message: ArrayLike) -> Array:
        """`PRF_keygen(KEY, M)` — an `n + 32`-byte input: -> uint8 `[B, n]`.

        What a WOTS+ secret value is derived from, the input being
        `SEED ‖ ADRS`. Its own domain separator, so a secret value can never
        collide with the bitmask derived at the same position.
        """
        inputs = _rows(message)
        expected = self.n + _PRF_INPUT_SIZE
        if inputs.shape[-1] != expected:
            raise ValueError(
                f"PRF_keygen takes an {expected}-byte input, got {inputs.shape[-1]}"
            )
        return self._core(_PADDING_PRF_KEYGEN, key, inputs)

    def f(self, pub_seed: ArrayLike, adrs: ArrayLike, m1: ArrayLike) -> Array:
        """`F` — the one-way function a WOTS+ chain step applies: -> uint8 `[B, n]`."""
        return self._masked(_PADDING_F, pub_seed, adrs, m1, mask_words=1)

    def h(self, pub_seed: ArrayLike, adrs: ArrayLike, m2: ArrayLike) -> Array:
        """`H` — two n-byte blocks to one, a Merkle parent: -> uint8 `[B, n]`."""
        return self._masked(_PADDING_H, pub_seed, adrs, m2, mask_words=2)

    def hash_message(
        self,
        randomizer: ArrayLike,
        root: ArrayLike,
        index: ArrayLike,
        message: ArrayLike,
    ) -> Array:
        """`H_msg` — the digest a signature is over: -> uint8 `[B, n]`.

        `toByte(2, padlen) ‖ R ‖ Root ‖ toByte(idx_sig, n) ‖ M`. The index is what
        binds a signature to the one-time key that made it, and it is encoded in `n`
        bytes rather than the 32 `PRF` uses. It is a host integer, since it is a
        position in the tree rather than data.
        """
        messages = _rows(message)
        batch = messages.shape[0]
        indices = np.asarray(index, dtype=np.int64).reshape(-1)
        if indices.shape[0] != batch:
            # Not broadcast: every signature in a batch carries its own index, and
            # spreading one across several would digest them all under the wrong
            # one-time key while agreeing with itself.
            raise ValueError(
                f"one index per message: got {batch} messages and "
                f"{indices.shape[0]} indices"
            )
        encoded = np.stack(
            [np.frombuffer(_to_byte(int(i), self.n), dtype=np.uint8) for i in indices]
        )
        return self._digest(
            fnp.concatenate(
                [
                    self._padding(_PADDING_HASH, batch),
                    _batched(randomizer, batch),
                    _batched(root, batch),
                    fnp.asarray(encoded, dtype=fnp.uint8),
                    messages,
                ],
                axis=-1,
            )
        )

    # -- the constructions the family is built from ------------------------

    def _masked(
        self,
        separator: int,
        pub_seed: ArrayLike,
        adrs: ArrayLike,
        payload: ArrayLike,
        *,
        mask_words: int,
    ) -> Array:
        """`core_hash(toByte(c, padlen) ‖ KEY ‖ (M XOR BM))` — §5.1's `F` and `H`.

        The addresses set the batch: there is one hash per position, and both the
        seed and the payload may be shared across them. The key and each n-byte word
        of the mask come from the same address under a different `keyAndMask`, which
        is why this takes one encoded address rather than `mask_words + 1` of them.
        """
        addresses = np.asarray(adrs, dtype=np.uint8)
        if addresses.ndim == 1:
            addresses = addresses[None, :]
        batch = addresses.shape[0]
        payloads = _batched(payload, batch)
        if payloads.shape[-1] != mask_words * self.n:
            raise ValueError(
                f"this hash takes {mask_words * self.n} bytes, got {payloads.shape[-1]}"
            )
        key = self.prf(pub_seed, with_key_and_mask(addresses, 0))
        mask = fnp.concatenate(
            [
                self.prf(pub_seed, with_key_and_mask(addresses, word + 1))
                for word in range(mask_words)
            ],
            axis=-1,
        )
        return self._digest(
            fnp.concatenate(
                [self._padding(separator, batch), key, payloads ^ mask], axis=-1
            )
        )

    def _core(self, separator: int, key: ArrayLike, message: Array) -> Array:
        """`core_hash(toByte(c, padlen) ‖ KEY ‖ M)` — the two keyed PRFs."""
        batch = message.shape[0]
        return self._digest(
            fnp.concatenate(
                [self._padding(separator, batch), _batched(key, batch), message],
                axis=-1,
            )
        )

    def _padding(self, separator: int, batch: int) -> Array:
        """`toByte(c, padding_len)`, broadcast across the batch."""
        return fnp.broadcast_to(
            fnp.asarray(
                np.frombuffer(_to_byte(separator, self.padding_len), dtype=np.uint8)
            ),
            (batch, self.padding_len),
        )

    def _digest(self, messages: Array) -> Array:
        """`core_hash` — the injected hash, truncated to `n`.

        The 192-bit sets are SHA-256 cut to 24 bytes rather than a different hash,
        so truncation is part of the family rather than something a caller does.
        """
        return fnp.asarray(self._byte_hash.digest(messages), dtype=fnp.uint8)[
            :, : self.n
        ]


def sha2_hashes(params: XmssParams, byte_hash: ByteHash) -> Rfc8391Hashes:
    """The family for a SHA-2 parameter set.

    Refuses a row whose `core_hash` is not SHA-2, because a `ByteHash` does not say
    which hash it is: without this, OID 7 would build happily over SHA-256 and
    produce a self-consistent implementation of a parameter set that does not exist.
    """
    if params.core_hash is not CoreHash.SHA2:
        raise ValueError(
            f"{params.name} is a {params.core_hash.value} set; it needs a "
            f"{params.core_hash.value} core hash, which hash-frx does not have yet"
        )
    return Rfc8391Hashes(byte_hash, n=params.n, padding_len=params.padding_len)


def _rows(value: ArrayLike) -> Array:
    """A `[k]` operand as the `[1, k]` batch of one it stands for."""
    array = fnp.asarray(value, dtype=fnp.uint8)
    return array[None, :] if array.ndim == 1 else array


def _batched(value: ArrayLike, batch: int) -> Array:
    """Broadcast a shared `[k]` operand to `[B, k]`, or pass a `[B, k]` through."""
    array = fnp.asarray(value, dtype=fnp.uint8)
    if array.ndim == 1:
        return fnp.broadcast_to(array, (batch, array.shape[0]))
    return array


def _to_byte(value: int, length: int) -> bytes:
    """`toByte(x, n)` — RFC 8391 §2.4: the big-endian n-byte encoding of x."""
    return value.to_bytes(length, "big")
