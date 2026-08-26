# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Message to codeword: the hash, the aborting decode, and the target-sum filter.

A leanSig signature commits to a **codeword** — `DIMENSION` digits in base
`BASE`, one per Winternitz chain — and this module is everything between the
32-byte root being signed and those digits. It is two filters in a row, and
which of them rejects an attempt is the whole shape of the module:

1. **The aborting decode.** One Poseidon call yields `ceil(DIMENSION / Z)` field
   elements; each expands into `Z` base-`BASE` digits by dividing out `Q`. The
   prime is chosen so `Q * BASE^Z == PRIME - 1`, which makes `0 .. PRIME - 2`
   exactly `BASE^Z` equal-sized groups and every digit uniform — the property the
   random-oracle argument needs. The single value `PRIME - 1` belongs to no group,
   so it aborts. Probability about 4.7e-10 per element: real, and unreachable by
   sampling, which is why it is gated on a constructed input rather than a lucky
   one. Reference: the aborting encoding of
   [eprint 2026/016](https://eprint.iacr.org/2026/016) §6.1.
2. **The target-sum filter.** The codeword is accepted only if its digits sum to
   `TARGET_SUM` — a single layer of the hypercube. The layer sits above the mean
   digit sum on purpose, so most attempts miss and the signer resamples.

## What is reshaped, and what a caller gets instead of `None`

Upstream returns `list[int] | None` and the caller reads the `None`. A tracer has
no `None` to read, so both entry points return `(digits, accepted)` and the
digits are meaningless wherever `accepted` is false — the abort leaves a quotient
too wide for `Z` digits, so what comes back is the low part of it. Every caller
therefore reads the flag; nothing here narrows the pair for you.

## Where the host ends

`encode_message` and `encode_epoch` are host-only and cannot be otherwise. Both
decompose a value wider than a lane — a 256-bit root, and a `(epoch << 8) | prefix`
tweak — and long division by a 31-bit modulus has no 32-bit lane form to fall back
on, so they take a Python integer and hand back the limbs
([`field.py`](field.py)). Everything after them is namespace-agnostic: the hash is
one `compress`, and the decode is division and masking over values that are all
below `PRIME` and so all fit a lane.

That boundary is a real one for batch verification and it is not resolved here.
A verifier holds `[B]` roots and `[B]` slots, and the two encoders would meet them
with a Python loop; `compress` in front of the decode is one-dimensional as well,
so of this pipeline only `aborting_decode` takes a batch axis today. What the seam
does about it is the verify slice's question — it is item 6 of
[#195](https://github.com/fractalyze/sig-frx/issues/195), where the slot being a
per-entry input already needs an answer — and the constraint to carry into it is
the one above: the decomposition is bignum division, not a shift schedule.

## The rejection loop, and where it is not

`conventions.md` asks every scheme to say which form its rejection loop takes.
leanSig's is over `rho` with `MAX_TRIES`, and it is **a host loop**: signing is
host-side here, and the acceptance test is a public function of public inputs —
the verifier recomputes it from the `rho` the signature carries — so a
data-dependent trip count leaks nothing a verifier does not already hold. The
loop itself is the signer's and is not in this module, which is upstream's split
too: `target_sum_encode` reports one attempt and the caller retries.

## Lane order

Field vectors here are lane-reversed, the convention
[`poseidon.py`](poseidon.py) states and the reason it gives — the encoders place
their limbs reversed, and `compress` hands back a reversed digest. The codeword
is where that ends: digit `i` addresses chain `i`, a chain number has no lane,
and `aborting_decode` un-reverses once at that boundary rather than pushing a
reversed-codeword convention onto everything downstream.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from frx import Array
from frx.typing import ArrayLike

from sig_frx.arrays import namespace
from sig_frx.hash.leansig import poseidon
from sig_frx.hash.leansig.field import int_to_base_p, to_field
from sig_frx.hash.leansig.params import TWEAK_PREFIX_MESSAGE, LeanSigParams

MESSAGE_BYTES: Final = 32
"""What leanSig signs: a 32-byte root, upstream's `Bytes32`.

Not a parameter — no preset moves it, and `MESSAGE_LENGTH_FIELD_ELEMENTS = 9`
does not pin it, since 9 base-p limbs hold far more than 256 bits.
"""

_MESSAGE_HASH_WIDTH: Final = 24
"""The permutation the message hash runs on, fixed by upstream at the call site.

Width 16 is the chain hash's; everything that mixes a message, a parameter and a
tweak in one compression needs the wider state.
"""


def encode_message(message: bytes, *, params: LeanSigParams) -> Array:
    """A 32-byte root as `message_length` lane-reversed field elements.

    The bytes are read **little-endian** as one integer and decomposed base-p.
    Host-only, per the module docstring.
    """
    if len(message) != MESSAGE_BYTES:
        raise ValueError(
            f"leanSig signs a {MESSAGE_BYTES}-byte root, got {len(message)} bytes"
        )
    limbs = int_to_base_p(int.from_bytes(message, "little"), params.message_length)
    return to_field(limbs[::-1])


def encode_epoch(epoch: int, *, params: LeanSigParams) -> Array:
    """A slot as `tweak_length` lane-reversed field elements, message-subdomain.

    The tweak is `(epoch << 8) | TWEAK_PREFIX_MESSAGE`: the 8-bit prefix is what
    keeps a message hash from colliding with a chain or tree hash at the same
    position. Host-only, per the module docstring.

    The bound checked here is the slot's own type. The tighter one — a slot below
    the key's lifetime — belongs to the tree that indexes by it; what a slot past
    it hits instead is the limb count, since two base-p limbs stop holding the
    shifted tweak well before a `Uint64` runs out.
    """
    if not 0 <= epoch < 2**64:
        raise ValueError(f"a slot is a Uint64, got {epoch}")
    tweak = (epoch << 8) | TWEAK_PREFIX_MESSAGE
    return to_field(int_to_base_p(tweak, params.tweak_length)[::-1])


def _places(params: LeanSigParams) -> np.ndarray:
    """`base^0 .. base^(digits_per_element - 1)`, the digit extraction's weights.

    A static schedule off the parameter set, so it stays a host constant on both
    paths rather than becoming `digits_per_element` traced multiplies.
    """
    return np.asarray(
        [params.base**place for place in range(params.digits_per_element)],
        dtype=np.uint32,
    )


def aborting_decode(
    elements: ArrayLike, *, params: LeanSigParams
) -> tuple[Array, Array]:
    """Field elements -> `(digits, accepted)`, the hypercube decode.

    `elements` is a lane-reversed digest of `message_hash_length` elements, with
    a leading batch axis allowed; `digits` comes back in codeword order, so
    `[..., dimension]`, and `accepted` is `[...]`.

    Each element divides by `quotient` and expands into `digits_per_element`
    base-`base` digits, least significant first; the digits of all the elements
    run together and the first `dimension` of them are the codeword. An element
    equal to `PRIME - 1` has no quotient in range and rejects the whole vector.
    """
    xnp = namespace(elements)
    values = xnp.asarray(elements).astype(np.uint32)

    # Upstream's own comparison. `quotient * base^digits_per_element` is
    # `PRIME - 1` by the parameter invariant, so this fires on that one value —
    # written as the threshold rather than as the value because that is what
    # makes it a range check rather than a coincidence.
    threshold = np.uint32(params.quotient * params.base**params.digits_per_element)
    accepted = ~(values >= threshold).any(axis=-1)

    # The digest arrives lane-reversed and the codeword does not, so the element
    # order flips exactly here. Within an element the digits are arithmetic
    # rather than layout, and their order does not move.
    quotients = values[..., ::-1] // np.uint32(params.quotient)
    digits = (quotients[..., None] // _places(params)) % np.uint32(params.base)

    return digits.reshape(*digits.shape[:-2], -1)[..., : params.dimension], accepted


def message_hash(
    message_elements: Array,
    parameter: Array,
    epoch_elements: Array,
    randomness: Array,
    *,
    params: LeanSigParams,
) -> tuple[Array, Array]:
    """One Poseidon compression, decoded — `(digits, accepted)`.

    The four operands are lane-reversed field vectors listed in upstream's own
    order: the encoded message, the public parameter, the encoded epoch, and the
    per-attempt randomness. They are the *encoded* forms rather than a root and a
    slot, which is where this splits from upstream's signature and why: the two
    encoders are host-only and this is not.
    """
    operands = (message_elements, parameter, epoch_elements, randomness)
    lengths = (
        ("message", params.message_length),
        ("parameter", params.parameter_length),
        ("epoch", params.tweak_length),
        ("randomness", params.randomness_length),
    )
    # Per operand rather than on the total, which is what `compress` bounds: the
    # four sum to 23 into a width-24 state, so two wrong lengths that cancel
    # would pass that check and hash a different pre-image.
    for operand, (name, length) in zip(operands, lengths):
        if operand.shape != (length,):
            raise ValueError(
                f"the {name} operand is {length} field elements, "
                f"got shape {operand.shape}"
            )

    digest = poseidon.compress(
        operands,
        width=_MESSAGE_HASH_WIDTH,
        output_length=params.message_hash_length,
    )
    return aborting_decode(digest, params=params)


def target_sum_encode(
    message_elements: Array,
    parameter: Array,
    epoch_elements: Array,
    randomness: Array,
    *,
    params: LeanSigParams,
) -> tuple[Array, Array]:
    """One signing attempt: the message hash, accepted only on the target layer.

    Same operands as `message_hash`, and `accepted` is narrowed by the second
    filter — the digits must also sum to `target_sum`. A false flag is what a
    signer retries on, with fresh `randomness`.
    """
    digits, accepted = message_hash(
        message_elements, parameter, epoch_elements, randomness, params=params
    )
    # `dtype` pinned: numpy promotes a reduction's accumulator and frx does not,
    # so a bare `.sum()` is `uint64` on the host and `uint32` traced from this
    # one line (`CLAUDE.md`). The widest sum here is `dimension * (base - 1)`.
    on_layer = digits.sum(axis=-1, dtype=np.uint32) == np.uint32(params.target_sum)
    return digits, accepted & on_layer
