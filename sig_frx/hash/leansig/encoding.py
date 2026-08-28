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

This repo already has the second filter once, under another name: SHRINCS drops
WOTS+'s checksum chains for the same trick, and
[`shrincs/wots_c.py`](../shrincs/wots_c.py)'s `map_digest` ends in the same
`(digits, digits.sum() == constant)` pair. The one difference is the one that
decides the cost. `wots_c.CONSTANT_SUM` is the *mode* of its distribution, so
grinding lands about one counter in sixty-five; leanSig's `TARGET_SUM = 200` sits
2.5 standard deviations above a mean digit sum of 161, so an attempt lands about
one in nine hundred. Same construction, three orders of magnitude apart in what
signing costs — worth knowing before reading either as the other.

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

That boundary is a real one and it is settled the way it had to be. Every stage
here is one signature's, batch axis included — `compress` is one-dimensional too,
so a leading axis on the decode alone would be a seam nothing could reach — and
the batch arrives through `frx.vmap` in `codewords` below rather than as a second
transcription. The host half stays host: a verifier meets its `[B]` roots and
slots with `encode_messages` and `encode_epochs`, which decompose on the host and
lift once, and a signer meets its one of each with the singular forms. The
constraint underneath all of it is the one above — the decomposition is bignum
division, not a shift schedule.

## The rejection loop, and where it is not

`conventions.md` asks every scheme to say which form its rejection loop takes.
leanSig's is over `rho` with `MAX_TRIES`, and it is **a host loop**: signing is
host-side here, and the acceptance test is a public function of public inputs —
the verifier recomputes it from the `rho` the signature carries — so the loop is
free to take whichever form is cheapest. What that costs is named rather than
defended: the trip count depends on the message and on PRF output keyed by the
secret seed, so it is not something an observer recomputes, and it is permitted
because signing carries no timing claim
([`security.md`](../../../docs/reference/security.md)).

The loop itself is not in this module, which is upstream's split too:
`target_sum_encode` reports on one attempt and the caller retries. It lives in
[`signing.py`](signing.py)'s `search`, in the shape this section predicted it
would want — `wots_c.grind`'s, a host search trying a *block* of candidates per
pass, so the cost is one batched dispatch per block rather than a Python
iteration per candidate. `codewords` below is the batched entry point that made
that possible, and the verifier is its other caller.

## Lane order

Field vectors here are lane-reversed, the convention
[`poseidon.py`](poseidon.py) states and the reason it gives — the encoders place
their limbs reversed, and `compress` hands back a reversed digest. The codeword
is where that ends: digit `i` addresses chain `i`, a chain number has no lane,
and the decode un-reverses once at that boundary rather than pushing a
reversed-codeword convention onto everything downstream. It does so through
`poseidon.undo_lane_reversal` rather than an open-coded slice, because that
module owns the convention and a second one re-deriving it is what spreads it.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache, partial
from typing import Final

import frx
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from sig_frx.arrays import namespace
from sig_frx.hash.leansig import field, poseidon
from sig_frx.hash.leansig.field import lane_reversed_limbs
from sig_frx.hash.leansig.params import (
    MESSAGE_BYTES,
    TWEAK_PREFIX_MESSAGE,
    LeanSigParams,
)

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
    return lane_reversed_limbs(int.from_bytes(message, "little"), params.message_length)


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
    return lane_reversed_limbs(tweak, params.tweak_length)


def encode_messages(messages: ArrayLike, *, params: LeanSigParams) -> Array:
    """A batch of 32-byte roots as `[B, message_length]` lane-reversed elements.

    `encode_message` for a whole batch, and the reason it exists separately is
    the transfer rather than the arithmetic: the bignum division is Python's
    either way, but calling the singular form per row lifts each result on its
    own. Measured at `PROD` with `B = 64`, that loop is ~2.97 ms against
    ~0.12 ms here, of which the division is ~0.1 ms — so nearly all of what the
    per-row form costs is dispatch.

    Host-only, per the module docstring.
    """
    rows = np.asarray(messages, dtype=np.uint8)
    if rows.ndim != 2 or rows.shape[1] != MESSAGE_BYTES:
        raise ValueError(
            f"leanSig signs {MESSAGE_BYTES}-byte roots, so a batch is "
            f"[B, {MESSAGE_BYTES}], got shape {tuple(rows.shape)}"
        )
    return field.lane_reversed_limbs_stack(
        [int.from_bytes(bytes(row), "little") for row in rows], params.message_length
    )


def encode_epochs(epochs: ArrayLike, *, params: LeanSigParams) -> Array:
    """A batch of slots as `[B, tweak_length]`, message-subdomain.

    `encode_epoch` for a whole batch. The packing is elementwise and the column
    is `int64`, so unlike the message this is the singular form with nothing
    removed — `lane_reversed_limbs` already takes a column, which is the same
    shape `tweakable.tree_tweaks` hands it for the very same slots.
    """
    column = np.asarray(epochs, dtype=np.int64)
    if column.ndim != 1:
        raise ValueError(f"a slot batch is [B], got shape {tuple(column.shape)}")
    if np.any(column < 0):
        raise ValueError("a slot is a Uint64, so it cannot be negative")
    return lane_reversed_limbs(
        (column << 8) | TWEAK_PREFIX_MESSAGE, params.tweak_length
    )


@lru_cache(maxsize=None)
def _places(params: LeanSigParams) -> np.ndarray:
    """`base^0 .. base^(digits_per_element - 1)`, the digit extraction's weights.

    A static schedule off the parameter set, so it stays a host constant on both
    paths rather than becoming `digits_per_element` traced multiplies.

    Cached the way every other static schedule in the repo is —
    [`wots.py`](../wots.py)'s `_digit_plan`, [`bytestring.py`](../bytestring.py)'s
    `_PLACES`, `poseidon.lane_reversed_permutation`. Free under a tracer either
    way; eagerly a rebuilt host array costs frx a fresh canonicalization on every
    call, and the signer makes ~900 of them per signature. What
    `safe_domain_separator` refuses to cache is a concrete *device* array, which
    carries backend affinity; this is inert numpy.
    """
    return np.asarray(
        [params.base**place for place in range(params.digits_per_element)],
        dtype=np.uint32,
    )


def aborting_decode(
    elements: ArrayLike, *, params: LeanSigParams
) -> tuple[Array, Array]:
    """A digest -> `(digits, accepted)`, the hypercube decode.

    `elements` is one lane-reversed digest of `message_hash_length` elements;
    `digits` comes back in codeword order, `[dimension]`, and `accepted` is a
    scalar. One signature's, like every stage here — see the module docstring on
    where the batch axis is not.

    Each element divides by `quotient` and expands into `digits_per_element`
    base-`base` digits, least significant first; the digits of all the elements
    run together and the first `dimension` of them are the codeword. An element
    equal to `PRIME - 1` has no quotient in range and rejects the whole vector.

    **Not `wots.base_2b`, and not `mldsa.encoding.unpack_fields`,** which are the
    third and fourth spellings of "read digits out of something" in this repo —
    [`falcon/encoding.py`](../../lattice/falcon/encoding.py) records the same
    near-miss against both. Those two read bit-fields out of a *byte string* at a
    fixed width; this is integer arithmetic on a field residue already in a lane,
    `base` is not required to be a power of two, and the value has to be divided
    by `quotient` first. Unifying any two of them round-trips forever while being
    wrong.
    """
    xnp = namespace(elements)
    values = xnp.asarray(elements).astype(np.uint32)

    # Upstream's own comparison, against the product rather than against
    # `PRIME - 1`, for the reason `decode_threshold` records.
    accepted = ~(values >= np.uint32(params.decode_threshold)).any()

    # The digest is lane-reversed and the codeword is not, so the convention
    # exits here — through `poseidon`'s own seam, since this is device movement
    # rather than placement. Within an element the digit order is arithmetic
    # rather than layout and does not move.
    quotients = poseidon.undo_lane_reversal(values) // np.uint32(params.quotient)
    digits = (quotients[:, None] // _places(params)) % np.uint32(params.base)

    return digits.reshape(-1)[: params.dimension], accepted


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
    # One tuple in upstream's operand order, so the names, the lengths and what
    # gets hashed cannot drift apart.
    named = (
        ("message", message_elements, params.message_length),
        ("parameter", parameter, params.parameter_length),
        ("epoch", epoch_elements, params.tweak_length),
        ("randomness", randomness, params.randomness_length),
    )
    # Checked per operand rather than on the total, which is all `compress`
    # bounds: the four sum to 23 into a width-24 state, so two wrong lengths that
    # cancel would pass that check and hash a different pre-image.
    for name, operand, length in named:
        if operand.shape != (length,):
            raise ValueError(
                f"the {name} operand is {length} field elements, "
                f"got shape {operand.shape}"
            )

    digest = poseidon.compress(
        [operand for _, operand, _ in named],
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

    **This, and not `message_hash`, is what both callers want**, and the two are
    easy to confuse because they take the same operands and differ by one `&`.
    The filter is the whole of what makes a codeword unforgeable. A verifier
    walks each chain `base - 1 - digit` steps from the value the signature
    released, so a codeword whose digits are all at least the signed one's
    reaches the same endpoints and rebuilds the same root — *any* codeword does,
    given released values that match it. The target sum is what collapses that
    set to a single point: two codewords that dominate elementwise and sum alike
    are equal. Drop the filter and the scheme is forgeable while every published
    vector still passes, which is why
    [`verify_vectors.py`](testing/verify_vectors.py) carries one case that fails
    on this and nothing else.

    Upstream spells the same rejection as `target_sum_encode(...) is None` in its
    `verify` phase 2, so it is its check rather than an extra one.
    """
    digits, accepted = message_hash(
        message_elements, parameter, epoch_elements, randomness, params=params
    )
    # `dtype` pinned: numpy promotes a reduction's accumulator and frx does not,
    # so a bare `.sum()` is `uint64` on the host and `uint32` traced from this
    # one line (`CLAUDE.md`). The widest sum here is `dimension * (base - 1)`.
    on_layer = digits.sum(dtype=np.uint32) == np.uint32(params.target_sum)
    return digits, accepted & on_layer


def codewords(
    message_elements: Array,
    parameters: Array,
    epoch_elements: Array,
    randomness: Array,
    *,
    params: LeanSigParams,
) -> tuple[Array, Array]:
    """`target_sum_encode` over a leading batch axis: `[B, dimension]`, `[B]`.

    The batched form both entry points of the scheme reach — a verifier holds `B`
    signatures and a signer holds a block of randomness candidates for one — and
    it is here for the reason `encode_messages` and `encode_epochs` are: the
    singular form is what a reader checks against upstream, and the batch is what
    anything actually calls.

    Every operand carries the batch axis, including the three a signer's block
    does not vary. Broadcasting them at the call site rather than giving this an
    `in_axes` for each is `wots_c.grind`'s shape
    ([`../shrincs/wots_c.py`](../shrincs/wots_c.py)) and keeps one callable for
    two callers instead of one per axis pattern.

    `frx.vmap` around the one-signature body rather than a second transcription
    of it over a batch axis — [`tweakable.py`](tweakable.py)'s `_compression` and
    `_leaf_sponge` are the same shape, and `ml_dsa.py` gives the reason: a
    batch-shaped copy is a second thing to keep in agreement with the first.
    """
    return _vmapped(params)(message_elements, parameters, epoch_elements, randomness)


@lru_cache(maxsize=None)
def _vmapped(params: LeanSigParams) -> Callable[..., tuple[Array, Array]]:
    """One vmapped `target_sum_encode` per preset, memoized.

    Built per call it would re-trace the same graph on every eager call — silent,
    and orders of magnitude over the work, which is what
    [`tweakable.py`](tweakable.py)'s memoized modes are guarding against too.
    """
    return frx.vmap(partial(target_sum_encode, params=params))
