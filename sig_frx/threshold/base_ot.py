# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Masny-Rindal endemic OT (MR19 Figure 8) over secp256k1, a batch at a time.

The base OTs DKLs23 names. Its §8.1 realizes the OT layer "via the
OT-extension protocol of Roy [Roy22] ... with base OTs supplied by the
two-round UC-secure endemic OT protocol of Masny and Rindal [MR19]",
instantiated "from the decisional Diffie-Hellman assumption over the same
group G in which signatures are to be computed" — secp256k1, for the ECDSA
track this belongs to.

**Endemic** is the weakest of the OT security notions and the one the layer
above wants: neither party supplies the transferred messages, and a corrupt
party is allowed to determine its own outputs. Both parties' messages fall
out of the protocol rather than going into it, which is why nothing here
takes a payload — DKLs23's Functionality 5.1 consumes exactly that shape.

## Which of MR19's protocols this is, and how that was decided

MR19 gives two. Figure 8 is the generic two-round construction from any
uniform key agreement; Figure 14, in Appendix D.2, is an optimized variant
whose sender transmits a single group element and whose security rests on an
interactive DDH assumption the paper is explicit it cannot reduce to plain
DDH. DKLs23 does not name a figure, so its own cost accounting decides.
§8.1 prices the base OT at

    EOTCost(|G|, l_OT) = 2 * |G| * l_OT

per party, "and it requires each party to compute 3 l_OT elliptic curve
scalar operations, on average". At `n = 2` Figure 8 sends two elements each
way, which is `2 |G|` per party; Figure 14 sends two one way and one the
other, which is `1.5 |G|`. And Figure 8 costs the receiver two scalar
operations against the sender's four, averaging three, where Figure 14
averages two and a half. Both prices are Figure 8's, and it is the one plain
DDH proves, so it is the one here.

That op count reads `r <- G` as a free sample from the group, which is what
the figure writes. This implementation spends a scalar multiplication on it
(see `choose`), so its receiver costs three rather than two and the average
is three and a half. The extra multiplication rides the same batched call as
the rest and is the honest price of sampling the group rather than its
encoding.

## The protocol, and the names it is written in

Figure 8 at `n = 2`, with `UKA = (A, B, Key)` the Diffie-Hellman key
exchange: `A(t_A) = [t_A]G`, `B(t_B, m_A) = [t_B]G`, and `Key` the shared
`[t_A t_B]G` computed from either side. Its two random oracles `H_0` and
`H_1` map `G -> G`. The receiver hides its choice by sending a pair whose
unchosen half is a uniform group element and whose chosen half is its key
agreement message minus that half's oracle output, so the sender recovers
`m_A` in exactly one of the two slots and cannot tell which.

`H_j` is [`hash_to_curve.py`](hash_to_curve.py) under a per-slot domain
separator. That module exists for this call: hashing to a scalar and
multiplying `G` would leave the hashing party knowing the exponent, and the
whole construction rests on neither party knowing it.

## The curve is named, not injected

The same rule `hash_to_curve.py` states, for the same reason one layer up:
the oracle into the group is `secp256k1_XMD:SHA-256_SSWU_RO_` and no other
suite here has one, so a group-generic base OT could not be instantiated
anyway. The parameter arrives with the second curve that has an oracle. It
is also why this module reaches for `secp` directly rather than for
[`group.py`](group.py)'s Protocol — a seam that cannot carry the oracle
carries only half of this protocol.

## The randomness is the caller's

Both round functions take the bytes they expand rather than drawing them,
which is [`frost.py`](frost.py)'s rule and what makes a transcript
reproducible under test. What is expanded is not what RFC 9591 expands:
that standard fixes `nonce_generate` and MR19 fixes nothing, so the
expansion is SHAKE-256 through `hashlib` — the escape hatch
[`hashes.py`](../hashes.py) names, and the right face for host-side protocol
work over bytes at these sizes.

**Every domain separator below is this repo's own.** MR19 leaves `H_0`,
`H_1` and the derivation of a bit string from a group element as random
oracles and names no function for any of them, and no two independent
DKLs23 implementations agree: one publishes ASCII tags, another a numeric
label pair, neither the paper's. A transcript here therefore interoperates
with nothing, which is a property of the protocol's specification and not of
this implementation.

## One batch, and — the part that costs — one shape

DKLs23 §8.1-8.2 sizes the setup at `lambda_c = 128` instances per party pair,
so nothing here loops over instances: a Python loop over a batch axis is the
shape this repo calls a bug. Each operation is one call over the whole batch —
two in `choose`, two in `transfer`, one in `receive`.

What that buys is **not** a device scalar multiplication, and it is worth
saying so because the obvious reading is wrong. `secp.multiple` places on a
batch-size threshold, but three of the five calls here multiply
`_CURVE.generator`, which is `[1]`-shaped and therefore below any threshold at
every `B`; and whether the other two place depends on the point dtypes being
admitted, which `secp` probes per wheel and which is false at the pinned one.
So the curve arithmetic is host work, and no value in this module ever changes
namespace.

The one thing that does reach the device is the square root inside `_decode`:
`secp.lift_x_to_parity` places its *base-field* batch, which is admitted where
the point types are not, and the fused ladder compiles **once per distinct
batch shape**. That is what the batch is really for here, and it is why both
`transfer` and `receive` decode the wire as one flat `2B` rather than a `[B]`
per slot — two shapes would compile the ladder twice and pay for it twice.
Measured on the CPU leg, the suite runs 13.8 s against 23.4 s when `transfer`
split its decode by slot.

The Python loop that remains is inside `hash_to_curve_batch`, over
`map_to_curve`'s host integer arithmetic — the namespace that arithmetic is
exact in, with no array form to move to. Picking one slot out of a decoded
pair is a single gather over the batch rather than a loop over it, which is
what lets `receive` validate both slots and still read only one.

## What this is gated on, and what it is not

The composition has no vector authority at any of
[`testing.md`](../../docs/reference/testing.md)'s three levels, and none is
coming: DKLs23 fixes the OT layer functionally and leaves every byte to the
implementer, so two correct implementations disagree on the wire by
construction and there is nothing for them to agree on.

So this module may not claim to be gated, and
[`testing/base_ot_test.py`](testing/base_ot_test.py) says so in as many
words. What holds it up instead is components on published vectors — the
oracle on RFC 9380 Appendix J.8.1, the curve arithmetic on the dtypes' own
gates — and the composition on **published breaks**: three independent
implementations have had key extraction or state leakage found in their
base-OT layer, and all three round-trip correctly, so a round trip reaches
none of them. Each is a test, with its provenance, in that file.

## Not a side-channel claim

[`security.md`](../../docs/reference/security.md) governs and is unchanged
here: the choice bits index Python lists and the exponents are host
integers. What this module does owe, and does not leave to its caller, is
that a slot is validated whether or not it is used — see `receive`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sig_frx.classical import secp
from sig_frx.threshold import hash_to_curve

_CURVE = secp.SECP256K1
_N = _CURVE.n
_P = _CURVE.p

# One endemic output is one SHA-256 digest.
KEY_SIZE = hashlib.sha256().digest_size

# The scalar expansion's per-value width: RFC 9380 §5.2's `L`, which at this
# curve's size and `k = 128` is the same 48 `hash_to_curve` uses. 32 bytes
# reduced modulo `n` is biased and 48 is not.
_SCALAR_BYTES = 48

# One draw per party per batch, matching `frost.nonce_generate`'s width. The
# per-instance exponents are expanded from it rather than drawn separately.
_RANDOMNESS_SIZE = 32

# The suite id trails an oracle separator in RFC 9380 §3.1's recommended
# shape. What precedes it is this repo's, because the paper names no function.
_SUITE = b"secp256k1_XMD:SHA-256_SSWU_RO_"
_PAD_ORACLE = b"MR19-OT-v1-pad-"
_KEY_ORACLE = b"MR19-OT-v1-key-"
_EXPAND = b"MR19-OT-v1-expand-"

# Keeps every separator inside `expand_message_xmd`'s 255-byte bound with room
# to spare: a pad separator's fixed part is 57 bytes, so the longest one this
# admits is 185.
_MAX_SESSION = 128


@dataclass(frozen=True)
class ReceiverState:
    """What the receiver holds between the two rounds.

    The choice bits and the key agreement exponents `t_A`, neither of which
    goes on the wire. Carried rather than re-derived, so `receive` cannot
    silently disagree with `choose` about which slot was asked for.
    """

    session: bytes
    choices: tuple[int, ...]
    exponents: tuple[int, ...]


@dataclass(frozen=True)
class Transfer:
    """The sender's round: what it sends, and what it keeps.

    `messages` is Figure 8's `(m_B0, m_B1)` per instance and goes to the
    receiver. `keys` is the sender's own pair of endemic outputs per
    instance, of which the receiver learns exactly one.
    """

    messages: tuple[tuple[bytes, bytes], ...]
    keys: tuple[tuple[bytes, bytes], ...]


def _require_session(session: bytes) -> None:
    """A session identifier short enough to leave the separators in range."""
    if len(session) > _MAX_SESSION:
        raise ValueError(f"a session identifier is at most {_MAX_SESSION} bytes")


def _scalars(randomness: bytes, session: bytes, label: bytes, count: int) -> list[int]:
    """`count` exponents in `[1, n-1]`, expanded from one draw.

    One squeeze split `count` ways rather than `count` squeezes. RFC 9380
    §5.2 draws the same distinction for the same reason: the two are
    different functions, because the output length is bound into the state.

    The session is bound in so that a caller who reuses a draw across two of
    them does not reuse the exponents — which would put one `[t_A]G` on both
    wires and make the two transcripts linkable. It costs nothing, and the
    alternative is a footgun whose damage is silent. The session trails the
    fixed-width randomness, so no two inputs share an expansion.

    A zero would reduce to the identity everywhere downstream instead of
    raising, so it is refused. The draw is a `2^-256` event; the check is one
    line and what it replaces is a transcript whose points are all infinity.
    """
    if len(randomness) != _RANDOMNESS_SIZE:
        raise ValueError(f"randomness is {_RANDOMNESS_SIZE} bytes")
    raw = hashlib.shake_256(_EXPAND + label + randomness + session).digest(
        count * _SCALAR_BYTES
    )
    scalars = [
        int.from_bytes(raw[index * _SCALAR_BYTES : (index + 1) * _SCALAR_BYTES], "big")
        % _N
        for index in range(count)
    ]
    if not all(scalars):
        raise ValueError("an expanded exponent was zero")
    return scalars


def _pad_dst(session: bytes, index: int, slot: int) -> bytes:
    """The domain separator for `H_slot` in instance `index`.

    Every field ahead of the session is fixed width or a fixed string, so no
    two `(session, index, slot)` triples share a separator. A variable-length
    field anywhere but last would let one triple's encoding read as another's.
    """
    return (
        _PAD_ORACLE
        + index.to_bytes(4, "big")
        + bytes([slot])
        + b"-with-"
        + _SUITE
        + b"-"
        + session
    )


def _key(session: bytes, index: int, slot: int, element: bytes) -> bytes:
    """One endemic output: the shared group element through a random oracle.

    Figure 8's outputs are group elements and an OT's are bit strings;
    MR19 Remark D.4 is where the paper closes that gap with the oracle. The
    slot and the instance index are bound in so a key cannot be moved between
    positions of a transcript that is otherwise replayed intact.
    """
    return hashlib.sha256(
        _KEY_ORACLE + index.to_bytes(4, "big") + bytes([slot]) + element + session
    ).digest()


def _encode(points: np.ndarray) -> list[bytes]:
    """A point batch as SEC 1 compressed encodings, identity refused.

    Refused rather than encoded because it has no encoding here, and because
    reaching it means an exponent or an oracle output cancelled — which the
    party that produced it aborts on rather than transmits. The refusal is
    this protocol's and not the substrate's, which is why `secp` supplies the
    identity scan and the per-point encoder but not this policy.
    """
    infinite = secp.identity_entries(_CURVE, points)
    if infinite:
        raise ValueError(f"the group identity has no encoding (entries {infinite})")
    return [
        secp.compressed_bytes(_CURVE, x, y) for x, y in secp.affine_ints(_CURVE, points)
    ]


def _oracle(
    session: bytes, encodings: Sequence[bytes], slots: Sequence[int]
) -> np.ndarray:
    """`H_slot` over a batch, one slot per row: `[B]` points.

    One batched map even where the separator differs per row, because the
    separator is an argument rather than a mode — which is what
    `hash_to_curve_batch` was widened to carry.
    """
    return hash_to_curve.hash_to_curve_batch(
        encodings,
        [_pad_dst(session, index, slot) for index, slot in enumerate(slots)],
    )


def choose(
    choices: Sequence[int], session: bytes, randomness: bytes
) -> tuple[ReceiverState, list[tuple[bytes, bytes]]]:
    """Round one — MR19 Figure 8's receiver, over a batch of instances.

    Per instance the unchosen slot carries a uniform group element `r` and
    the chosen slot carries `[t_A]G - H_c(r)`, so the sender's recovery
    `r_j + H_j(r_(1-j))` returns `[t_A]G` at `j = c` and an element nobody
    knows the exponent of at `j = 1-c`.

    **`r` is `[k]G` for a uniform `k`, which samples the group rather than
    its encoding's byte space.** Sampling bytes there instead is a published
    break, distinguishable from a group element with probability 7/8. Knowing
    `k` buys the receiver nothing: the element it would need an exponent for
    is `r + H_(1-c)(r_c)`, and an oracle output has none — which is the
    property `hash_to_curve` exists to provide.

    Returns the state to carry into `receive`, and the `(r_0, r_1)` pair per
    instance in wire order.
    """
    _require_session(session)
    count = len(choices)
    if count == 0:
        raise ValueError("a batch has at least one instance")
    if any(choice not in (0, 1) for choice in choices):
        raise ValueError("a choice is 0 or 1")

    exponents = _scalars(randomness, session, b"receiver-agreement", count)
    pad_exponents = _scalars(randomness, session, b"receiver-pad", count)

    agreement = secp.multiple(_CURVE, exponents, _CURVE.generator)
    pad_encodings = _encode(secp.multiple(_CURVE, pad_exponents, _CURVE.generator))
    chosen = agreement - _oracle(session, pad_encodings, choices)
    chosen_encodings = _encode(chosen)

    wire = [
        (chosen, pad) if choice == 0 else (pad, chosen)
        for choice, chosen, pad in zip(choices, chosen_encodings, pad_encodings)
    ]
    return ReceiverState(session, tuple(choices), tuple(exponents)), wire


def transfer(
    pads: Sequence[tuple[bytes, bytes]], session: bytes, randomness: bytes
) -> Transfer:
    """Round two — MR19 Figure 8's sender, over the same batch.

    Recovers `m_Aj = r_j + H_j(r_(1-j))` in both slots, answers each with its
    own key agreement message `[t_Bj]G`, and keeps the shared `[t_Bj]m_Aj`.
    Exactly one of the two agrees with what the receiver can compute, and the
    sender does not learn which.

    Both received elements are validated before either is used. Omitting that
    check is a published break: a sender that skips it on a point handed to it
    by the network lets an active adversary extract its secret. `m_Aj` is
    checked for the identity as well — a receiver can force one slot to it by
    choosing `r_j = -H_j(r_(1-j))`, and the output would then be a constant
    that party already knows.
    """
    _require_session(session)
    count = len(pads)
    if count == 0:
        raise ValueError("a batch has at least one instance")

    columns = [[pair[slot] for pair in pads] for slot in (0, 1)]
    both = secp.decompressed(
        _CURVE, [entry for pair in pads for entry in pair], "receiver pad"
    )
    received = [both[0::2], both[1::2]]
    recovered = [
        received[slot] + _oracle(session, columns[1 - slot], [slot] * count)
        for slot in (0, 1)
    ]
    for slot, points in enumerate(recovered):
        infinite = secp.identity_entries(_CURVE, points)
        if infinite:
            raise ValueError(
                "the recovered agreement message is the identity "
                f"(slot {slot}, entries {infinite})"
            )

    # One flat `2B` batch per operation rather than a call per slot: the
    # exponents already arrive in that layout, and `_encode`/`secp.multiple`
    # have no reason to see the slot split the wire format imposes.
    exponents = _scalars(randomness, session, b"sender-agreement", 2 * count)
    messages = _encode(secp.multiple(_CURVE, exponents, _CURVE.generator))
    shared = _encode(secp.multiple(_CURVE, exponents, np.concatenate(recovered)))
    return Transfer(
        messages=tuple(zip(messages[:count], messages[count:])),
        keys=tuple(
            (
                _key(session, index, 0, shared[index]),
                _key(session, index, 1, shared[count + index]),
            )
            for index in range(count)
        ),
    )


def receive(
    state: ReceiverState, messages: Sequence[tuple[bytes, bytes]]
) -> list[bytes]:
    """The receiver's output — one key per instance, at its own choice bit.

    **Both slots are validated, including the one that is not used.** A
    receiver that checked only the slot it reads would abort exactly when the
    sender corrupted that slot, so a malicious sender could recover the choice
    bits one instance at a time by corrupting one side and watching for the
    abort. The unused slot costs one batched square root and closes that
    selective-failure channel; it is not defensive tidiness.
    """
    count = len(state.choices)
    if len(messages) != count:
        raise ValueError("one agreement message pair per instance")

    both = secp.decompressed(
        _CURVE, [entry for pair in messages for entry in pair], "sender message"
    )
    chosen = both[np.arange(count) * 2 + np.array(state.choices)]
    shared = _encode(secp.multiple(_CURVE, list(state.exponents), chosen))
    return [
        _key(state.session, index, state.choices[index], shared[index])
        for index in range(count)
    ]
