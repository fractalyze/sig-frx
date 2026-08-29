# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""MR19 Figure 8 over secp256k1: components on vectors, composition on breaks.

**This suite is not a known-answer gate, and there is no known-answer gate to
be had.** No standard defines oblivious transfer, endemic OT or OT extension,
so `testing.md`'s level 1 and level 2 are empty; and there is no standard, so
level 3 — the reference implementation a standard points at — has nothing to
point at it. What exists instead is a field of independent implementations,
none of which publishes values: the DKLs authors' own `mpecdsa` carries no
test at all in `src/ote.rs` or `src/rot.rs`, libOTe's `OtExt_Kos_Test` asserts
only that a receiver's output matches the sender's chosen row, and the one
library that does commit deterministic output — emp-ot's
`test/trace_hash.baseline` — commits a digest of its own whole wire transcript
under its own serialization, which localizes nothing and would gate this repo
on that library's private choices.

Nor is one coming. DKLs23 specifies the OT layer *functionally* — an `F_EOTE`
and an `H` that is a random oracle — and fixes no bytes, so two correct
implementations of the same paper disagree on the wire by construction. There
is no interop surface here the way an ECDSA signature is one.

So the exception this file takes is named in both halves, as `testing.md`
requires:

- **What it is gated on.** Its components, on published vectors: the oracle
  into the group is `secp256k1_XMD:SHA-256_SSWU_RO_`, gated in
  `hash_to_curve_test.py` on RFC 9380 Appendix J.8.1 at four depths, which is
  authority level 1; the curve arithmetic is the curated dtypes', gated in
  `classical/testing/secp_test.py`. And the composition, on **published
  breaks** — see below.
- **What it is not gated on.** The composition has no published values.
  `test_round_trip` is a round trip, which `testing.md` says is not evidence,
  and it is here to localize a failure rather than to find one.

## The published breaks, which is what replaces a vector set

Fordefi's *Devious Transfer* found key extraction or state leakage in the
base-OT layer of three independent implementations, and **all three round-trip
correctly** — a round-trip test reaches none of them, which is exactly why
they are the useful negative cases. Each is a test below, named for the
implementation it was found in:

- `test_break_point_validation` — mpecdsa: "Bob does not verify the validity
  of the point `A` he receives from Alice", giving key extraction to an active
  adversary.
- `test_break_pad_sampled_from_the_group` — sl-crypto: `r̄` "was sampled as
  random 32-bytes" where the protocol requires sampling from `G`,
  "distinguishable from `r ∈ G`" with probability 7/8, giving key extraction
  to a passive one.
- `test_break_output_independent_of_the_secret` — docknetwork/crypto: the
  implementation "deviated from the original specification" so that the
  "sender's output messages were a direct function of the messages sent to the
  receiver, without involving a secret value".

  Provenance: https://blog.fordefi.com/devious-transfer-breaking-oblivious-transfer-based-threshold-ecdsa

`test_selective_failure_is_not_observable` is not from that set. It covers a
channel the protocol's shape opens rather than one an implementation opened,
and it is here because the abort it pins is the module's to own.

## The batch the protocol actually runs at

Everything above is sized to make a failure readable, which is far below the
`lambda_c = 128` a real setup runs. `DeviceBatchTest` runs that size — see its
own note for what it catches that the small batches cannot. It costs the
target most of its budget and is the reason for the bucket the BUILD file
argues for.

## The reference transcription

`testing.md` asks that a reshaped form be tested against the standard's own
form. The reshaping here is the batch axis — one `secp.multiple` per round
where Figure 8 describes one instance — so `_reference` transcribes the figure
the way it is written: one instance at a time, in affine Python integers, over
a double-and-add written out below rather than over the seams the module uses.
The domain separators are re-derived from their documented byte layout rather
than imported, so a mixed-up instance index or slot fails here instead of
agreeing with itself.
"""

from __future__ import annotations

import hashlib
from typing import NamedTuple

import numpy as np
from absl.testing import absltest, parameterized

from sig_frx.classical import secp
from sig_frx.classical.testing import weierstrass_reference
from sig_frx.threshold import base_ot, hash_to_curve

_CURVE = secp.SECP256K1
_P = _CURVE.p
_N = _CURVE.n
_GENERATOR = (_CURVE.gx, _CURVE.gy)

# The two draws every test below runs from. Fixed rather than sampled: the
# module takes its caller's randomness precisely so a transcript is
# reproducible, and a suite that drew its own would fail intermittently.
_RECEIVER_RANDOMNESS = bytes(range(32))
_SENDER_RANDOMNESS = bytes(range(32, 64))
_SESSION = b"dkls23-setup"

# On no point of this curve — asserted against `weierstrass_reference` below
# rather than stated, so a curve constant that drifted would fail here.
_OFF_CURVE_X = 5

# The same `x` as a wire encoding, which is how both the sender and the
# receiver see a point that is not one.
_OFF_CURVE_ENCODING = bytes([2]) + _OFF_CURVE_X.to_bytes(32, "big")

# DKLs23 §8.1: "Roy's protocol requires a one-time setup that comprises exactly
# lambda_c instances of OT", and §8.2 fixes `lambda_c = 128`.
_LAMBDA_C = 128


# ---------------------------------------------------------------------------
# Figure 8 as written: one instance, affine integers, no batched seam.
# ---------------------------------------------------------------------------


def _affine_add(
    left: tuple[int, int] | None, right: tuple[int, int] | None
) -> tuple[int, int] | None:
    """The short-Weierstrass chord-and-tangent law, `None` the identity."""
    if left is None:
        return right
    if right is None:
        return left
    (x1, y1), (x2, y2) = left, right
    if x1 == x2 and (y1 + y2) % _P == 0:
        return None
    if left == right:
        slope = 3 * x1 * x1 % _P * pow(2 * y1, _P - 2, _P) % _P
    else:
        slope = (y2 - y1) * pow(x2 - x1, _P - 2, _P) % _P
    x3 = (slope * slope - x1 - x2) % _P
    return x3, (slope * (x1 - x3) - y1) % _P


def _affine_mul(scalar: int, point: tuple[int, int] | None) -> tuple[int, int] | None:
    """Double-and-add, most significant bit first."""
    total = None
    for bit in bin(scalar % _N)[2:]:
        total = _affine_add(total, total)
        if bit == "1":
            total = _affine_add(total, point)
    return total


def _affine_negate(point: tuple[int, int]) -> tuple[int, int]:
    return point[0], (-point[1]) % _P


def _encode(point: tuple[int, int] | None) -> bytes:
    """Encode a point the figure's flow reached, identity refused.

    `None` is the identity, which the reference never produces here: reaching
    it would mean the same cancellation `base_ot._encode` refuses, so the two
    sides fail on it together rather than one of them encoding `(0, 0)`.
    """
    if point is None:
        raise ValueError("the reference reached the group identity")
    return secp.compressed_bytes(_CURVE, point[0], point[1])


def _reference_scalars(
    randomness: bytes, session: bytes, label: bytes, count: int
) -> list[int]:
    """The module's documented expansion, re-derived byte for byte."""
    raw = hashlib.shake_256(
        b"MR19-OT-v1-expand-" + label + randomness + session
    ).digest(count * 48)
    return [
        int.from_bytes(raw[index * 48 : (index + 1) * 48], "big") % _N
        for index in range(count)
    ]


def _reference_pad_dst(session: bytes, index: int, slot: int) -> bytes:
    return (
        b"MR19-OT-v1-pad-"
        + index.to_bytes(4, "big")
        + bytes([slot])
        + b"-with-secp256k1_XMD:SHA-256_SSWU_RO_-"
        + session
    )


def _reference_key(session: bytes, index: int, slot: int, element: bytes) -> bytes:
    return hashlib.sha256(
        b"MR19-OT-v1-key-"
        + index.to_bytes(4, "big")
        + bytes([slot])
        + element
        + session
    ).digest()


def _reference_oracle(
    session: bytes, index: int, slot: int, element: bytes
) -> tuple[int, int]:
    return hash_to_curve.hash_to_curve(
        element, _reference_pad_dst(session, index, slot)
    )


class _Expected(NamedTuple):
    """Everything the batched module produces, from the figure's own form."""

    wire: list[tuple[bytes, bytes]]
    messages: list[tuple[bytes, bytes]]
    sender_keys: list[tuple[bytes, bytes]]
    receiver_keys: list[bytes]


def _reference(
    choices: list[int], session: bytes, receiver_bytes: bytes, sender_bytes: bytes
) -> _Expected:
    """MR19 Figure 8 at `n = 2`, one instance at a time.

    Returns every value the batched module also produces, so a disagreement
    names which of the two rounds drifted rather than only that one did.
    """
    count = len(choices)
    agreement_exponents = _reference_scalars(
        receiver_bytes, session, b"receiver-agreement", count
    )
    pad_exponents = _reference_scalars(receiver_bytes, session, b"receiver-pad", count)
    sender_exponents = _reference_scalars(
        sender_bytes, session, b"sender-agreement", 2 * count
    )

    wire, receiver_keys, sender_messages, sender_keys = [], [], [], []
    for index, choice in enumerate(choices):
        # Receiver: `r` for the unchosen slot, `[t_A]G - H_c(r)` for its own.
        pad = _affine_mul(pad_exponents[index], _GENERATOR)
        agreement = _affine_mul(agreement_exponents[index], _GENERATOR)
        pad_encoding = _encode(pad)
        chosen = _affine_add(
            agreement,
            _affine_negate(_reference_oracle(session, index, choice, pad_encoding)),
        )
        pair = (
            (_encode(chosen), pad_encoding)
            if choice == 0
            else (pad_encoding, _encode(chosen))
        )
        wire.append(pair)

        # Sender: recover both slots, answer both, keep both shared elements.
        messages, keys = [], []
        for slot in (0, 1):
            recovered = _affine_add(
                _decode_affine(pair[slot]),
                _reference_oracle(session, index, slot, pair[1 - slot]),
            )
            exponent = sender_exponents[slot * count + index]
            messages.append(_encode(_affine_mul(exponent, _GENERATOR)))
            keys.append(
                _reference_key(
                    session, index, slot, _encode(_affine_mul(exponent, recovered))
                )
            )
        sender_messages.append((messages[0], messages[1]))
        sender_keys.append((keys[0], keys[1]))

        # Receiver: open the slot it asked for.
        shared = _affine_mul(
            agreement_exponents[index], _decode_affine(messages[choice])
        )
        receiver_keys.append(_reference_key(session, index, choice, _encode(shared)))

    return _Expected(wire, sender_messages, sender_keys, receiver_keys)


def _decode_affine(encoding: bytes) -> tuple[int, int]:
    """SEC 1 §2.3.4 decompression, in integers, for the reference side."""
    x = int.from_bytes(encoding[1:], "big")
    y = pow(weierstrass_reference.rhs(_CURVE, x), (_P + 1) // 4, _P)
    if y % 2 != encoding[0] & 1:
        y = (-y) % _P
    return x, y


# ---------------------------------------------------------------------------


class ReferenceTest(parameterized.TestCase):
    """The batched module against Figure 8 written out one instance at a time."""

    @parameterized.named_parameters(
        ("all_zero", [0, 0, 0]),
        ("all_one", [1, 1, 1]),
        ("mixed", [0, 1, 1, 0, 1]),
        ("single", [1]),
    )
    def test_agrees_with_the_figure(self, choices: list[int]) -> None:
        expected = _reference(
            choices, _SESSION, _RECEIVER_RANDOMNESS, _SENDER_RANDOMNESS
        )
        state, wire = base_ot.choose(choices, _SESSION, _RECEIVER_RANDOMNESS)
        sent = base_ot.transfer(wire, _SESSION, _SENDER_RANDOMNESS)
        opened = base_ot.receive(state, sent.messages)

        self.assertEqual(wire, expected.wire)
        self.assertEqual(list(sent.messages), expected.messages)
        self.assertEqual(list(sent.keys), expected.sender_keys)
        self.assertEqual(opened, expected.receiver_keys)


def _assert_opens_at_choices(
    case: absltest.TestCase,
    choices: list[int],
    sent: base_ot.Transfer,
    opened: list[bytes],
) -> None:
    """Each instance opens its own slot and no other — the OT's whole claim."""
    for index, choice in enumerate(choices):
        case.assertEqual(opened[index], sent.keys[index][choice])
        case.assertNotEqual(opened[index], sent.keys[index][1 - choice])


class RoundTripTest(absltest.TestCase):
    """Localization, not evidence — see the module docstring."""

    def test_round_trip(self) -> None:
        choices = [0, 1, 1, 0, 1, 0]
        state, wire = base_ot.choose(choices, _SESSION, _RECEIVER_RANDOMNESS)
        sent = base_ot.transfer(wire, _SESSION, _SENDER_RANDOMNESS)
        opened = base_ot.receive(state, sent.messages)
        _assert_opens_at_choices(self, choices, sent, opened)

    def test_instances_do_not_share_a_key(self) -> None:
        """Two instances with the same choice bit still get different keys.

        The instance index is bound into both the oracle separator and the key
        derivation, so an implementation that derived per batch rather than per
        instance — or that reduced over the batch axis — collides here.
        """
        state, wire = base_ot.choose([1, 1, 1, 1], _SESSION, _RECEIVER_RANDOMNESS)
        sent = base_ot.transfer(wire, _SESSION, _SENDER_RANDOMNESS)
        opened = base_ot.receive(state, sent.messages)
        self.assertLen(set(opened), 4)
        self.assertLen({pair[0] for pair in sent.keys}, 4)

    def test_session_separates(self) -> None:
        """One draw reused across two sessions shares nothing.

        The session reaches the exponents, the oracle separators and the key
        derivation, so the two runs differ from the wire outward rather than
        only at the output. A caller that reuses randomness this way has made
        a mistake either way; what this pins is that the mistake does not put
        one `[t_A]G` on two transcripts.
        """
        choices = [0, 1]
        first, first_wire = base_ot.choose(choices, b"one", _RECEIVER_RANDOMNESS)
        second, second_wire = base_ot.choose(choices, b"two", _RECEIVER_RANDOMNESS)
        self.assertNotEqual(first.exponents, second.exponents)
        self.assertNotEqual(first_wire, second_wire)
        sent = base_ot.transfer(first_wire, b"one", _SENDER_RANDOMNESS)
        self.assertNotEqual(
            base_ot.receive(first, sent.messages),
            base_ot.receive(second, sent.messages),
        )


class PublishedBreakTest(absltest.TestCase):
    """The three findings the module docstring names, one test each."""

    def test_break_point_validation(self) -> None:
        """mpecdsa: the point that arrived from the network is validated.

        Five ways a pad can be malformed, and the identity as a sixth. All six
        reach the sender through the same wire slot an honest pad would.
        """
        _, wire = base_ot.choose([0, 1], _SESSION, _RECEIVER_RANDOMNESS)
        honest = wire[0][0]

        off_curve = _OFF_CURVE_ENCODING
        self.assertFalse(weierstrass_reference.has_point_at(_CURVE, _OFF_CURVE_X))
        cases = {
            "off the curve": off_curve,
            "x at the modulus": bytes([2]) + _P.to_bytes(32, "big"),
            "x above the modulus": bytes([3]) + (_P + 1).to_bytes(32, "big"),
            "uncompressed prefix": bytes([4]) + honest[1:],
            "truncated": honest[:-1],
            "the identity": bytes([2]) + bytes(32),
        }
        for name, corrupt in cases.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    base_ot.transfer(
                        [(corrupt, wire[0][1]), wire[1]], _SESSION, _SENDER_RANDOMNESS
                    )

        # The same check on the receiver's side of the wire.
        state, wire = base_ot.choose([0], _SESSION, _RECEIVER_RANDOMNESS)
        sent = base_ot.transfer(wire, _SESSION, _SENDER_RANDOMNESS)
        with self.assertRaises(ValueError):
            base_ot.receive(state, [(off_curve, sent.messages[0][1])])

    def test_break_point_validation_reaches_the_recovered_element(self) -> None:
        """A pad chosen so the *recovered* element is the identity is refused.

        A receiver can drive one slot to infinity by sending
        `r_j = -H_j(r_(1-j))`, which no per-point check catches because the
        pad itself is a perfectly good point. The sender's output there would
        be a constant that the receiver already knows.
        """
        other = _affine_mul(5, _GENERATOR)
        other_encoding = _encode(other)
        forced = _encode(
            _affine_negate(_reference_oracle(_SESSION, 0, 0, other_encoding))
        )
        with self.assertRaisesRegex(ValueError, "identity"):
            base_ot.transfer([(forced, other_encoding)], _SESSION, _SENDER_RANDOMNESS)

    def test_break_pad_sampled_from_the_group(self) -> None:
        """sl-crypto: the pad is a sample from `G`, not from a byte space.

        Every pad the receiver emits decompresses to a point on the curve. An
        implementation that emitted 32 random bytes instead passes one
        instance about half the time, so a batch this size is decisive; the
        published distinguisher is stronger still, at 7/8 per instance.
        """
        count = 32
        choices = [index % 2 for index in range(count)]
        _, wire = base_ot.choose(choices, _SESSION, _RECEIVER_RANDOMNESS)
        pads = [wire[index][1 - choices[index]] for index in range(count)]
        for index, pad in enumerate(pads):
            with self.subTest(index):
                self.assertIn(pad[0], (2, 3))
                x, y = _decode_affine(pad)
                self.assertTrue(secp.on_curve(_CURVE, x, y))
                self.assertNotEqual((x, y), secp.AFFINE_IDENTITY)
        self.assertLen(set(pads), count)

    def test_break_output_independent_of_the_secret(self) -> None:
        """docknetwork: the sender's output involves a secret and the oracle.

        The sharp form of that finding. A receiver knows the exponent `k` of
        the pad it sent, so if the sender's output were a function of the
        transcript it would be `[k]m_B` in the unchosen slot. It is not,
        because the sender multiplies `r + H(...)` rather than `r` — dropping
        the oracle term is what would make the two agree.
        """
        choice = 0
        state, wire = base_ot.choose([choice], _SESSION, _RECEIVER_RANDOMNESS)
        sent = base_ot.transfer(wire, _SESSION, _SENDER_RANDOMNESS)

        pad_exponent = _reference_scalars(
            _RECEIVER_RANDOMNESS, _SESSION, b"receiver-pad", 1
        )[0]
        forged = _reference_key(
            _SESSION,
            0,
            1 - choice,
            _encode(
                _affine_mul(pad_exponent, _decode_affine(sent.messages[0][1 - choice]))
            ),
        )
        self.assertNotEqual(forged, sent.keys[0][1 - choice])

    def test_sender_output_moves_with_its_own_randomness(self) -> None:
        """The same wire under a second draw gives a different transfer."""
        _, wire = base_ot.choose([0, 1], _SESSION, _RECEIVER_RANDOMNESS)
        first = base_ot.transfer(wire, _SESSION, _SENDER_RANDOMNESS)
        second = base_ot.transfer(wire, _SESSION, bytes(32))
        self.assertNotEqual(first.messages, second.messages)
        self.assertNotEqual(first.keys, second.keys)

    def test_the_wire_hides_the_agreement_message(self) -> None:
        """Neither slot is `[t_A]G` itself.

        An implementation that forgot the `- H_c(r)` term would put the key
        agreement message on the wire in the clear, which hands the choice bit
        to anyone who can recognize it.
        """
        for choice in (0, 1):
            with self.subTest(choice=choice):
                state, wire = base_ot.choose([choice], _SESSION, _RECEIVER_RANDOMNESS)
                agreement = _encode(_affine_mul(state.exponents[0], _GENERATOR))
                self.assertNotIn(agreement, wire[0])


class SelectiveFailureTest(absltest.TestCase):
    """The abort must not depend on which slot the receiver asked for."""

    def test_selective_failure_is_not_observable(self) -> None:
        """A corrupt *unchosen* slot aborts exactly as a chosen one does.

        A receiver that validated only the slot it reads would abort iff the
        sender corrupted that slot, which recovers the choice bit one instance
        at a time.
        """
        bad = _OFF_CURVE_ENCODING
        for choice in (0, 1):
            for corrupted in (0, 1):
                with self.subTest(choice=choice, corrupted=corrupted):
                    state, wire = base_ot.choose(
                        [choice], _SESSION, _RECEIVER_RANDOMNESS
                    )
                    sent = base_ot.transfer(wire, _SESSION, _SENDER_RANDOMNESS)
                    messages = list(sent.messages[0])
                    messages[corrupted] = bad
                    with self.assertRaises(ValueError):
                        base_ot.receive(state, [(messages[0], messages[1])])


class InterfaceTest(absltest.TestCase):
    """The bounds each entry point states, and that it states them."""

    def test_choice_is_a_bit(self) -> None:
        with self.assertRaises(ValueError):
            base_ot.choose([0, 2], _SESSION, _RECEIVER_RANDOMNESS)

    def test_batch_is_not_empty(self) -> None:
        with self.assertRaises(ValueError):
            base_ot.choose([], _SESSION, _RECEIVER_RANDOMNESS)
        with self.assertRaises(ValueError):
            base_ot.transfer([], _SESSION, _SENDER_RANDOMNESS)

    def test_randomness_width(self) -> None:
        for width in (0, 31, 33):
            with self.subTest(width=width):
                with self.assertRaises(ValueError):
                    base_ot.choose([0], _SESSION, bytes(width))

    def test_session_bound(self) -> None:
        with self.assertRaises(ValueError):
            base_ot.choose([0], bytes(129), _RECEIVER_RANDOMNESS)

    def test_one_message_pair_per_instance(self) -> None:
        state, wire = base_ot.choose([0, 1], _SESSION, _RECEIVER_RANDOMNESS)
        sent = base_ot.transfer(wire, _SESSION, _SENDER_RANDOMNESS)
        with self.assertRaises(ValueError):
            base_ot.receive(state, sent.messages[:1])

    def test_key_width(self) -> None:
        state, wire = base_ot.choose([0], _SESSION, _RECEIVER_RANDOMNESS)
        sent = base_ot.transfer(wire, _SESSION, _SENDER_RANDOMNESS)
        self.assertLen(base_ot.receive(state, sent.messages)[0], base_ot.KEY_SIZE)


class DeviceBatchTest(absltest.TestCase):
    """The protocol at the batch size it actually runs at.

    DKLs23 §8.1-8.2 sizes the one-time setup at `lambda_c = 128` instances per
    party pair. That crosses `secp.DEVICE_MIN_BATCH`, so the square root
    inside the wire decode compiles a traced ladder here that none of the
    small batches above reach — the one place this module leaves the host, and
    the target's whole runtime.

    One test, because one is what the axis costs to cover: a batch that
    reduced over its axis, that broadcast one derivation across it, or that
    let one instance's oracle output reach another's key produces a `receive`
    that no longer matches `transfer` at that entry's choice bit. The
    reference transcription above is what says the values are right; this says
    the axis survived being large.
    """

    def test_lambda_c_instances(self) -> None:
        self.assertGreater(_LAMBDA_C, secp.DEVICE_MIN_BATCH)
        # Seeded rather than hand-written: the choice bits are the one input
        # whose pattern could hide an off-by-one in the batched gather, and a
        # hand-written list would be a pattern.
        choices = [
            int(bit) for bit in np.random.default_rng(0).integers(0, 2, size=_LAMBDA_C)
        ]

        state, wire = base_ot.choose(choices, _SESSION, _RECEIVER_RANDOMNESS)
        sent = base_ot.transfer(wire, _SESSION, _SENDER_RANDOMNESS)
        opened = base_ot.receive(state, sent.messages)

        self.assertLen(wire, _LAMBDA_C)
        self.assertLen(sent.messages, _LAMBDA_C)
        self.assertLen(opened, _LAMBDA_C)
        _assert_opens_at_choices(self, choices, sent, opened)


if __name__ == "__main__":
    absltest.main()
