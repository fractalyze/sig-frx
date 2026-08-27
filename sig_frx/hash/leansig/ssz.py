# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""leanSig's wire format: the SSZ encoding of a public key and a signature.

Every other scheme in this repo carries its standard's own byte encoding, and
each of those was written for the scheme — FIPS 205 concatenates fixed-width
byte strings, RFC 8391 prefixes an index. leanSig has no encoding of its own: it
rides Ethereum consensus' serialization, so a signature is an SSZ container and a
digest is a run of `Fp`, each a four-byte little-endian canonical residue.

**The container is fixed-size, and that is a decision upstream had to defend.**
Two of the three fields are SSZ *lists* at the type level — the authentication
path's siblings and the released chain ends — so the wire carries an offset for
each and, in principle, a length. Every valid signature pins both: one sibling
per tree level and one released end per chain, which is `log_lifetime` and
`dimension`. Upstream declares the container fixed-size on the strength of that
and rejects, after decoding, any signature whose two lists came out a different
length (`Signature._check_list_lengths`, `spec/crypto/xmss/containers.py`). That
rejection is not decoration: list bounds are read from attacker-controlled
offsets, so without it two distinct byte strings of the same length decode to
signatures holding different numbers of digests, and the verifier walks a path
that is not the one the bytes claim.

**Here that rejection is three offset comparisons.** In a buffer whose length is
already pinned by the seam's static shape, the offsets are the only freedom left:
fix all three and both list lengths follow. So this module checks them rather
than decoding a length and comparing it, which is the same predicate with nothing
between the wire and the answer.

**A residue is checked too, and `astype` is why.** Upstream's `Fp.deserialize`
raises on a four-byte group at or above the prime; the cast this module uses
*reduces* instead, quietly turning `PRIME` into zero and `PRIME + 5` into five.
So a non-canonical group would otherwise verify as a different, well-formed
element — a malleability the seam cannot see, since both encodings are the same
length. The range check is what upstream's exception becomes here.

**Both rejections are a `bool` and not a raise**, for the reason
[`encoding.py`](encoding.py) gives: a tracer has no exception to take. What is
static is checked statically — a buffer of the wrong length is a shape error and
raises here — and what depends on the bytes comes back beside the values, for the
verifier to fold into its verdict. The values are meaningless where the flag is
false, and nothing narrows them for a caller.

**Leading axes are free.** The last axis is the encoding and everything before it
is the caller's: `[B, size]` is the batch `verify` takes, `[size]` the single
signature a signer produces. There is no scalar/batch split because the work is
a reshape and a weighted sum, which do not have one.

**Digests come back placed.** Everything on the wire is in leanSpec's lane order,
and every hash in this package runs over the reverse of it
([`poseidon.py`](poseidon.py)) — so a decode that handed back wire order would
leave its consumer to reverse three arrays and would be silently wrong the one
time it forgot. The placement rides with the decode for the reason
[`field.py`](field.py)'s does, and it goes through `poseidon`'s own seam rather
than a `[::-1]` spelled here.

The one thing this does *not* do is SSZ merkleization. The published container
fixtures carry a `root` beside every `serialized` — the container's hash-tree
root, which is a consensus-layer identity for the object rather than part of the
signature scheme. Nothing in verification reads it, and nothing here computes it.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final, NamedTuple

import numpy as np
from frx import Array
from frx.typing import ArrayLike
from zk_dtypes import koalabear_mont as F

from sig_frx.arrays import namespace
from sig_frx.hash.leansig import poseidon
from sig_frx.hash.leansig.field import PRIME
from sig_frx.hash.leansig.params import LeanSigParams

P_BYTES: Final = 4
"""One KoalaBear residue on the wire — upstream's `P_BYTES`.

Four rather than the three the prime would fit in: SSZ has no 24-bit basic type,
and `Fp` declares itself a `uint32`.
"""

OFFSET_BYTES: Final = 4
"""SSZ's `BYTES_PER_LENGTH_OFFSET`, and the width every offset below is read at.

The same four as `P_BYTES` by coincidence rather than by construction — one is
the serialization's, the other the field's — so they are named apart. Their being
equal is what makes the whole encoding a run of four-byte words, which is the
only reason this module can index it in words rather than in bytes.
"""

_SHIFTS: Final = np.array([0, 8, 16, 24], dtype=np.uint32)
"""Little-endian place values, as the static schedule the packing rides.

A host constant on both paths, like [`encoding.py`](encoding.py)'s `_places` and
[`bytestring.py`](../bytestring.py)'s `_PLACES`. Eight bits shifted by 24 reaches
`2^32 - 1` exactly, so a group of four arbitrary bytes lands in `uint32` without
wrapping — which is what lets the range check below see a non-canonical value
rather than its remainder.
"""


class _SignatureLayout(NamedTuple):
    """Where a signature's parts sit, in four-byte words and in bytes."""

    path_offset: int
    """The `path` field's offset, in bytes. The fixed part's whole length."""

    sibling_offset: int
    """The offset nested inside `path`, in bytes. `HashTreeOpening` has one
    variable field and nothing ahead of it, so this is `OFFSET_BYTES` flat."""

    hashes_offset: int
    """The `hashes` field's offset, in bytes: past the fixed part and the path."""

    rho_at: int
    """`rho`'s first word. It is fixed-length, so it sits inline in the fixed
    part between the two offsets rather than out in the heap."""

    siblings_at: int
    """The first sibling word."""

    hashes_at: int
    """The first chain-end word."""

    words: int
    """The whole encoding, in four-byte words."""


@lru_cache(maxsize=None)
def _signature_layout(params: LeanSigParams) -> _SignatureLayout:
    """The static layout of a signature at `params`.

    Derived from the preset rather than restated, because upstream derives its
    `SIGNATURE_LENGTH_BYTES` the same way and a transcribed constant is a second
    place for the two to disagree. Cached for the reason every static schedule
    here is: a frozen preset is hashable, and rebuilding the tuple per call would
    cost a signer one allocation per attempt.
    """
    siblings = params.log_lifetime * params.hash_length
    hashes = params.dimension * params.hash_length
    path_offset = OFFSET_BYTES + params.randomness_length * P_BYTES + OFFSET_BYTES
    return _SignatureLayout(
        path_offset=path_offset,
        sibling_offset=OFFSET_BYTES,
        hashes_offset=path_offset + OFFSET_BYTES + siblings * P_BYTES,
        rho_at=1,
        siblings_at=path_offset // P_BYTES + 1,
        hashes_at=path_offset // P_BYTES + 1 + siblings,
        words=path_offset // P_BYTES + 1 + siblings + hashes,
    )


def public_key_size(params: LeanSigParams) -> int:
    """A serialized public key, in bytes — 52 at both presets.

    Fixed-size throughout: a root and a public parameter, both `Fp` vectors, so
    the container carries no offset at all. The two presets differ in lifetime
    and codeword and in neither of these, which is why one number covers both.
    """
    return (params.hash_length + params.parameter_length) * P_BYTES


def signature_size(params: LeanSigParams) -> int:
    """A serialized signature, in bytes — upstream's `SIGNATURE_LENGTH_BYTES`.

    2536 at `PROD` and 424 at `TEST`. The widely-quoted 3112 is the technical
    note's `v = 64` figure and describes no preset that ships
    ([`params.py`](params.py)).
    """
    return _signature_layout(params).words * P_BYTES


def encode_public_key(
    root: ArrayLike, parameter: ArrayLike, *, params: LeanSigParams
) -> Array:
    """`(root, parameter)` -> uint8 `[..., public_key_size]`.

    Both operands are lane-reversed field vectors, as everything in this package
    holds them; the wire order is restored here.
    """
    xnp = namespace(root, parameter)
    named = (
        ("root", root, (params.hash_length,)),
        ("parameter", parameter, (params.parameter_length,)),
    )
    return _bytes(xnp.concatenate([_words_of(*entry) for entry in named], axis=-1))


def decode_public_key(
    encoded: ArrayLike, *, params: LeanSigParams
) -> tuple[Array, Array, Array]:
    """uint8 `[..., public_key_size]` -> `(root, parameter, well_formed)`.

    `root` and `parameter` come back lane-reversed and shaped `[..., n]` and
    `[..., parameter_length]`. `well_formed` is `[...]`: every four-byte group
    is a canonical residue. There are no offsets in this container, so that is
    the whole of what the bytes can get wrong.
    """
    words = _words(encoded, public_key_size(params), "a public key")
    root, parameter = (
        words[..., : params.hash_length],
        words[..., params.hash_length :],
    )
    return _to_field(root), _to_field(parameter), _canonical(words)


def encode_signature(
    siblings: ArrayLike,
    rho: ArrayLike,
    hashes: ArrayLike,
    *,
    params: LeanSigParams,
) -> Array:
    """`(siblings, rho, hashes)` -> uint8 `[..., signature_size]`.

    `siblings` is `[..., log_lifetime, n]` and `hashes` is `[..., dimension, n]`,
    both lane-reversed, ordered as upstream orders them: siblings from the leaf
    upward, chain ends by chain index. `rho` is the `[..., randomness_length]`
    randomness that made the codeword hit its target sum.

    The three offsets are written from the layout rather than taken from the
    operands, which is what makes this the exact inverse of the decode's
    comparison: an encoding this produces is one that decode accepts.
    """
    xnp = namespace(siblings, rho, hashes)
    layout = _signature_layout(params)
    named = (
        ("siblings", siblings, (params.log_lifetime, params.hash_length)),
        ("rho", rho, (params.randomness_length,)),
        ("hashes", hashes, (params.dimension, params.hash_length)),
    )
    sibling_words, rho_words, hash_words = (_words_of(*entry) for entry in named)

    def offset(value: int) -> Array:
        return xnp.broadcast_to(
            xnp.asarray(value, dtype=xnp.uint32), (*rho_words.shape[:-1], 1)
        )

    return _bytes(
        xnp.concatenate(
            [
                offset(layout.path_offset),
                rho_words,
                offset(layout.hashes_offset),
                offset(layout.sibling_offset),
                sibling_words,
                hash_words,
            ],
            axis=-1,
        )
    )


def decode_signature(
    encoded: ArrayLike, *, params: LeanSigParams
) -> tuple[Array, Array, Array, Array]:
    """uint8 `[..., signature_size]` -> `(siblings, rho, hashes, well_formed)`.

    The three values come back in the shapes `encode_signature` takes, and
    `well_formed` is `[...]`: the three offsets are the ones a valid signature
    carries, and every field-bearing group is a canonical residue. Those are the
    two ways bytes of the right length can fail to be a signature, and the module
    docstring says why each is a flag rather than a raise.

    The offset words are deliberately outside the range check. They are `uint32`
    lengths rather than residues, so bounding them by the prime would be checking
    the wrong type — and the equality below already admits exactly one value.
    """
    layout = _signature_layout(params)
    words = _words(encoded, signature_size(params), "a signature")
    rho = words[..., layout.rho_at : layout.rho_at + params.randomness_length]
    siblings = words[..., layout.siblings_at : layout.hashes_at]
    hashes = words[..., layout.hashes_at :]

    # The three offsets, at the only three word positions they can occupy: the
    # container's first word, the word after the inline `rho`, and the word just
    # before the siblings that one introduces.
    pinned = (
        (words[..., 0], layout.path_offset),
        (words[..., layout.rho_at + params.randomness_length], layout.hashes_offset),
        (words[..., layout.siblings_at - 1], layout.sibling_offset),
    )
    well_formed = _canonical(rho) & _canonical(siblings) & _canonical(hashes)
    for value, expected in pinned:
        well_formed = well_formed & (value == np.uint32(expected))

    return (
        _to_field(_digests(siblings, params)),
        _to_field(rho),
        _to_field(_digests(hashes, params)),
        well_formed,
    )


def _words(encoded: ArrayLike, size: int, what: str) -> Array:
    """uint8 `[..., size]` -> uint32 `[..., size / P_BYTES]`, little-endian.

    The length is a shape and so a static fact, which is why a wrong one raises
    here where the content checks return a flag.
    """
    xnp = namespace(encoded)
    values = xnp.asarray(encoded, dtype=xnp.uint8)
    if values.ndim == 0 or values.shape[-1] != size:
        raise ValueError(f"{what} is {size} bytes, got shape {tuple(values.shape)}")
    grouped = values.reshape(*values.shape[:-1], size // P_BYTES, P_BYTES)
    return (grouped.astype(xnp.uint32) << _SHIFTS).sum(axis=-1, dtype=xnp.uint32)


def _bytes(words: Array) -> Array:
    """uint32 `[..., count]` -> uint8 `[..., count * P_BYTES]`, little-endian.

    The inverse of `_words`, and the accumulator rule
    ([`../../../CLAUDE.md`](../../../CLAUDE.md)) is why the mask is spelled at a
    pinned width: `& 0xFF` against a Python integer promotes on one path and not
    on the other, from one source line.
    """
    xnp = namespace(words)
    spread = (words[..., None] >> _SHIFTS) & np.uint32(0xFF)
    return spread.astype(xnp.uint8).reshape(*words.shape[:-1], -1)


def _words_of(name: str, values: ArrayLike, shape: tuple[int, ...]) -> Array:
    """A lane-reversed field operand -> the flat wire words it serializes to.

    The shape is checked against the preset rather than trusted, because every
    one of these is a length the encoding pins: a caller that hands over a path
    of the wrong height would otherwise produce a buffer of the wrong size and
    find out at the concatenate, naming no field.
    """
    xnp = namespace(values)
    array = xnp.asarray(values)
    if (
        array.ndim < len(shape)
        or tuple(array.shape[array.ndim - len(shape) :]) != shape
    ):
        raise ValueError(f"{name} is {shape}, got shape {tuple(array.shape)}")
    words = poseidon.undo_lane_reversal(array).astype(np.uint32)
    return words.reshape(*array.shape[: array.ndim - len(shape)], -1)


def _digests(words: Array, params: LeanSigParams) -> Array:
    """A flat run of words -> `[..., count, hash_length]` digests."""
    return words.reshape(*words.shape[:-1], -1, params.hash_length)


def _to_field(words: Array) -> Array:
    """Wire-ordered `uint32` residues -> the lane-reversed field array.

    The cast Montgomery-encodes rather than reinterpreting, which is
    [`field.py`](field.py)'s rule read in the other direction. It also *reduces*,
    which is what `decode_*`'s range check is there to catch: this function has
    no way to refuse a value and does not try.
    """
    return poseidon.apply_lane_reversal(words).astype(F)


def _canonical(words: Array) -> Array:
    """Whether every word in the last axis is a residue below the prime."""
    return (words < np.uint32(PRIME)).all(axis=-1)
