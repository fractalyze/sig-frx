# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The signer's PRF — one SHAKE128 seed, and the two secrets it expands into.

A leanSig key covers `2^log_lifetime` slots and each slot owns `dimension`
Winternitz chains, so a stored secret key would be a chain start per chain per
slot — 1.6 trillion digests at `PROD`. Upstream stores a 32-byte seed instead and
derives every one of them on demand, which is what makes the top/bottom split of
[`signing.py`](signing.py) worth having: a tree that can be rebuilt need not be
kept.

Two derivations ride the same seed, and the byte between the domain separator and
the key is the whole of what keeps them apart:

| subdomain | what it derives | varies over |
|---|---|---|
| `0x00` | a chain's secret start | slot, chain index |
| `0x01` | one signing attempt's randomness | slot, message, attempt |

Both squeeze 16 bytes per field element and reduce mod the prime, which is
upstream's own margin: a 128-bit read reduced into a 31-bit field is biased by
about `2^-97`, so the digest is uniform for every purpose a proof makes of it.

**Deterministic, and that is the security property rather than a convenience.**
The randomness is keyed by `(slot, message, attempt)`, so signing one message at
one slot twice yields the same signature rather than a second one — and a
synchronized scheme's whole discipline is that a slot is used once. Drawing the
randomness freshly instead would make a repeated slot produce two distinct
signatures over two distinct codewords, which is exactly the multi-target opening
a one-time key cannot survive.

## Host-only, and not because of a lane

Everything else in this package that stays on the host does so because it packs
a value wider than 32 bits ([`field.py`](field.py)). This one is host-only for a
plainer reason: it is SHAKE128 over a byte string, and hashing bytes is not
array work. The output *is* array work and is handed back as one lifted array
per call rather than one per element, which is
[`field.py`](field.py)'s `lane_reversed_limbs_stack` argument at a different
input type.

There is no batch axis to be had inside a squeeze — each is a different input —
so what "batched" means here is the transfer and nothing else. `chain_starts`
takes columns because a bottom tree wants `leaves_per_bottom_tree · dimension`
of them at once, and `randomness` takes a block of counters because the signer's
rejection loop tries a block per pass ([`signing.py`](signing.py)).

## Provenance

`spec/crypto/xmss/prf.py` at the pinned commit. The domain separator, the two
subdomain bytes, the four-byte big-endian epoch and the eight-byte big-endian
counter are transcribed rather than derived — a PRF has no structure to check
them against, so a wrong width here is a key that is self-consistently wrong and
only an upstream vector catches it
([`prf_vectors.py`](testing/prf_vectors.py)).
"""

from __future__ import annotations

import hashlib
from typing import Final

import numpy as np
from frx import Array
from frx.typing import ArrayLike

from sig_frx.hash.leansig.field import PRIME, lane_reversed
from sig_frx.hash.leansig.params import LeanSigParams

KEY_BYTES: Final = 32
"""`PRF_KEY_LENGTH` — the master seed, in bytes.

The whole secret. Every chain start under every slot of the key's lifetime is a
function of these 32 bytes, so this is what a signer protects and what a backup
holds; the resident trees of [`signing.py`](signing.py) are a cache of work
already derivable from it.
"""

_DOMAIN: Final = bytes.fromhex("aeae22ff0001faff21af12000111ff00")
"""`PRF_DOMAIN_SEP` — the fixed 16-byte prefix on every squeeze.

Upstream's stated reason is cross-context collision resistance if SHAKE128 is
reused elsewhere in a client. It has no structure; it is a nothing-up-my-sleeve
constant transcribed byte for byte.
"""

_CHAIN_START: Final = b"\x00"
"""`PRF_DOMAIN_SEP_DOMAIN_ELEMENT` — the chain-start subdomain."""

_RANDOMNESS: Final = b"\x01"
"""`PRF_DOMAIN_SEP_RANDOMNESS` — the signing-randomness subdomain."""

_BYTES_PER_ELEMENT: Final = 16
"""`PRF_BYTES_PER_FIELD_ELEMENT` — squeezed bytes per output element."""

_EPOCH_BYTES: Final = 4
"""How wide a slot is packed in a PRF input — big-endian, four bytes.

Not the `Uint64` a slot is elsewhere. Upstream fixes it at four on the strength
of `LOG_LIFETIME <= 32`, so the whole lifetime fits, and `_epoch_bytes` refuses a
slot that does not rather than letting `int.to_bytes` raise somewhere less
legible.
"""

_COUNTER_BYTES: Final = 8
"""How wide an attempt counter is packed — big-endian, eight bytes.

Wider than the epoch because it is a `Uint64` upstream and `MAX_TRIES` is the
only thing bounding it, where a slot is bounded by the lifetime.
"""


_MESSAGE_BYTES: Final = 32
"""What a signature's randomness is bound to — the same `Bytes32` root
[`encoding.py`](encoding.py) hashes, checked here so a wrong-width message is
refused before it reaches a squeeze that would accept any length."""


def chain_starts(
    key: bytes, epochs: ArrayLike, chain_indices: ArrayLike, *, params: LeanSigParams
) -> Array:
    """Every named chain's secret start: -> `[N, hash_length]`, lane-reversed.

    `epochs` and `chain_indices` are `[N]` host columns naming one chain each, in
    the row order the caller wants them back in — entry-major, for the callers
    here, which is what [`wots.chain`](../wots.py) and the tweak columns beside
    it are laid out to agree with.

    The two columns rather than a slot and a count because the caller decides the
    layout: a bottom tree wants every chain of a run of slots, and a signature
    wants every chain of one.
    """
    slots = _column(epochs, "epoch", 1 << (8 * _EPOCH_BYTES))
    chains = _column(chain_indices, "chain index", params.dimension)
    if slots.shape != chains.shape:
        raise ValueError(
            f"one chain index per epoch: got {slots.size} epochs and "
            f"{chains.size} chain indices"
        )
    prefix = _prefix(key, _CHAIN_START)
    return _squeeze(
        [
            prefix
            + _epoch_bytes(int(slot))
            + int(chain).to_bytes(_COUNTER_BYTES, "big")
            for slot, chain in zip(slots, chains, strict=True)
        ],
        params.hash_length,
    )


def randomness(
    key: bytes,
    epoch: int,
    message: ArrayLike,
    counters: ArrayLike,
    *,
    params: LeanSigParams,
) -> Array:
    """One block of signing attempts' randomness: -> `[N, randomness_length]`.

    One slot and one message across the block, because that is the shape the
    rejection loop has: an attempt differs from the last one in its counter
    alone, and the whole point of the loop is that the message and the slot do
    not move ([`signing.py`](signing.py)).

    Lane-reversed, like every other field vector this package holds — it goes
    straight into the message hash and out onto the wire, and the codec restores
    upstream's order at the boundary ([`ssz.py`](ssz.py)).
    """
    root = _root(message)
    attempts = _column(counters, "counter", params.max_tries)
    prefix = _prefix(key, _RANDOMNESS) + _epoch_bytes(epoch) + root
    return _squeeze(
        [prefix + int(counter).to_bytes(_COUNTER_BYTES, "big") for counter in attempts],
        params.randomness_length,
    )


def _prefix(key: bytes, subdomain: bytes) -> bytes:
    """Everything a squeeze's input opens with — `domain ‖ subdomain ‖ key`.

    The order is upstream's and is the whole of what a transcription can get
    wrong here: putting the subdomain byte first, or the key before the domain
    separator, gives a PRF that is uniform, deterministic and someone else's.

    The seed's width is checked rather than trusted, because SHAKE128 accepts
    any length and nothing downstream can tell a 16-byte seed from a 32-byte
    one — a short key derives a whole self-consistent key pair.
    """
    material = bytes(key)
    if len(material) != KEY_BYTES:
        raise ValueError(
            f"a leanSig master seed is {KEY_BYTES} bytes, got {len(material)}"
        )
    return _DOMAIN + subdomain + material


def _root(message: ArrayLike) -> bytes:
    """The message a randomness draw is bound to, as its 32 bytes.

    Taken as bytes or as a `uint8` array, because both callers are natural: a
    test holds a literal and `LeanSig.sign` holds what the caller handed the
    seam. The width is checked here rather than felt downstream — a squeeze
    would accept any length.
    """
    root = (
        np.frombuffer(message, dtype=np.uint8)
        if isinstance(message, (bytes, bytearray, memoryview))
        else np.asarray(message, dtype=np.uint8)
    )
    if root.shape != (_MESSAGE_BYTES,):
        raise ValueError(
            f"leanSig signs a {_MESSAGE_BYTES}-byte root, got shape "
            f"{tuple(root.shape)}"
        )
    return bytes(root)


def _epoch_bytes(epoch: int) -> bytes:
    """A slot in the four-byte big-endian packing upstream's PRF inputs use."""
    if not 0 <= epoch < 1 << (8 * _EPOCH_BYTES):
        raise ValueError(
            f"a PRF input packs a slot in {_EPOCH_BYTES} bytes, so it must be "
            f"in [0, {1 << (8 * _EPOCH_BYTES)}); got {epoch}"
        )
    return int(epoch).to_bytes(_EPOCH_BYTES, "big")


def _column(values: ArrayLike, name: str, bound: int) -> np.ndarray:
    """A `[N]` host column, checked against what its packing admits."""
    column = np.asarray(values, dtype=np.int64).reshape(-1)
    outside = (column < 0) | (column >= bound)
    if np.any(outside):
        raise ValueError(
            f"a {name} is in [0, {bound}); "
            f"{int(np.count_nonzero(outside))} of {column.size} are not, "
            f"the first being {int(column[np.argmax(outside)])}"
        )
    return column


def _squeeze(inputs: list[bytes], length: int) -> Array:
    """One SHAKE128 squeeze per input -> `[N, length]` lane-reversed elements.

    Each squeeze is `length` groups of 16 bytes read big-endian and reduced. The
    reduction is upstream's `Fp(value=...)`, whose constructor takes any integer
    and reduces — so it is the modulo here rather than a rejection, and the bias
    it leaves is the margin the module docstring quantifies.

    Assembled as one host array and lifted once. That is not a micro-optimization
    at the counts here: a `TEST` key pair is 1024 chain starts and a `PROD`
    bottom tree is three million, so a per-row lift would be one transfer each.
    """
    wide = _BYTES_PER_ELEMENT * length
    return lane_reversed(
        [
            [
                int.from_bytes(digest[at : at + _BYTES_PER_ELEMENT], "big") % PRIME
                for at in range(0, wide, _BYTES_PER_ELEMENT)
            ]
            for digest in (hashlib.shake_128(data).digest(wide) for data in inputs)
        ]
    )
