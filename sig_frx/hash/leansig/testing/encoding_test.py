# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The message-to-codeword pipeline against what leanSpec computes.

Every case that can trace runs both eagerly and traced, and the two must agree in
*dtype* as well as value. The decode is where that matters most: it is a division
and a masking over a reduction's neighbourhood, and `CLAUDE.md`'s promotion trap
lands on exactly the sum the target-sum filter takes.

The two encoders have no traced leg because they cannot have one — they
decompose values wider than a lane and only a Python integer holds those, which
is the module's own boundary rather than a gap in this file.

Vectors arrive in leanSpec's lane order and the pipeline works over the reverse
of it, so each case reverses on the way in. That belongs to the test rather than
to the scheme, for the reason [`poseidon.py`](../poseidon.py) gives.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from zk_dtypes import koalabear_mont as F

from sig_frx.hash.leansig import encoding, params
from sig_frx.hash.leansig.params import LeanSigParams
from sig_frx.hash.leansig.testing import harness
from sig_frx.hash.leansig.testing.encoding_vectors import (
    DECODE_VECTORS,
    EPOCH_VECTORS,
    MESSAGE_HASH_VECTORS,
    MESSAGE_VECTORS,
    DecodeVector,
    EpochVector,
    MessageHashVector,
    MessageVector,
)
from sig_frx.hash.leansig.testing.mode_vectors import operand_elements


def _digits(codeword: str) -> list[int]:
    """A vector's codeword string as the digits it stands for."""
    return [int(digit) for digit in codeword]


def _on(function: Callable[..., Any], jit: bool) -> Callable[..., Any]:
    """`function` on the requested leg — traced through the shared jit cache."""
    return harness.jitted(function, "params") if jit else function


def _reference_decode(
    elements: Sequence[int], preset: LeanSigParams
) -> list[int] | None:
    """leanSpec's `aborting_decode`, transcribed — the loop, one element at a time.

    Upstream is `src/lean_spec/spec/crypto/xmss/encoding.py` at the pin
    [`encoding_vectors.py`](encoding_vectors.py) records; this is that function's
    body written the way it writes it.

    The implementation expands every element and every digit position at once and
    masks; this is the form the spec writes, and the two have to agree. Cases
    beyond the transcribed vectors are what it buys, since it costs nothing to
    run over an input nobody asked upstream about.
    """
    threshold = preset.quotient * preset.base**preset.digits_per_element
    digits: list[int] = []
    for element in elements:
        if element >= threshold:
            return None
        quotient = element // preset.quotient
        for _ in range(preset.digits_per_element):
            quotient, digit = divmod(quotient, preset.base)
            digits.append(digit)
    return digits[: preset.dimension]


class EncodeMessageTest(parameterized.TestCase):
    """A 32-byte root becomes the limbs upstream decomposes it into."""

    @parameterized.named_parameters(
        *((vector.name, vector) for vector in MESSAGE_VECTORS)
    )
    def test_it_matches_upstream(self, vector: MessageVector) -> None:
        message = vector.message.to_bytes(encoding.MESSAGE_BYTES, "little")

        got = encoding.encode_message(message, params=params.PROD)

        self.assertEqual(harness.to_leanspec_order(got), list(vector.elements))
        self.assertEqual(got.dtype, F)
        self.assertEqual(got.shape, (params.PROD.message_length,))

    def test_it_rejects_a_root_of_the_wrong_length(self) -> None:
        # The message space is a 32-byte root, and nine base-p limbs hold far
        # more than that — so nothing downstream would notice a short one.
        with self.assertRaisesRegex(ValueError, "32-byte root"):
            encoding.encode_message(bytes(31), params=params.PROD)


class EncodeEpochTest(parameterized.TestCase):
    """A slot becomes the message-subdomain tweak upstream packs it into."""

    @parameterized.named_parameters(
        *((vector.name, vector) for vector in EPOCH_VECTORS)
    )
    def test_it_matches_upstream(self, vector: EpochVector) -> None:
        got = encoding.encode_epoch(vector.epoch, params=params.PROD)

        self.assertEqual(harness.to_leanspec_order(got), list(vector.elements))
        self.assertEqual(got.dtype, F)
        self.assertEqual(got.shape, (params.PROD.tweak_length,))

    def test_the_prefix_separates_it_from_the_other_subdomains(self) -> None:
        # Slot 0 carries nothing but the prefix, so this is the one case where
        # the tweak is the domain separator alone.
        got = encoding.encode_epoch(0, params=params.PROD)

        self.assertEqual(harness.to_leanspec_order(got)[0], params.TWEAK_PREFIX_MESSAGE)

    def test_it_rejects_a_slot_outside_its_own_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "Uint64"):
            encoding.encode_epoch(-1, params=params.PROD)

    def test_a_slot_too_wide_for_the_tweak_is_rejected_not_truncated(self) -> None:
        # Two base-p limbs stop holding the shifted tweak well before a `Uint64`
        # runs out, and truncating would silently hash a different slot.
        with self.assertRaisesRegex(ValueError, "base-p limbs"):
            encoding.encode_epoch(2**63, params=params.PROD)


class AbortingDecodeTest(parameterized.TestCase):
    """The hypercube decode matches upstream, digits and verdict alike."""

    @parameterized.named_parameters(*harness.both_legs(DECODE_VECTORS))
    def test_it_matches_upstream(self, vector: DecodeVector, jit: bool) -> None:
        preset = harness.PRESETS[vector.preset]
        elements = harness.lane_reversed(vector.elements)

        digits, accepted = _on(encoding.aborting_decode, jit)(elements, params=preset)

        self.assertEqual(bool(accepted), vector.digits is not None)
        if vector.digits is not None:
            self.assertEqual(np.asarray(digits).tolist(), _digits(vector.digits))
        self.assertEqual(digits.dtype, np.uint32)
        self.assertEqual(digits.shape, (preset.dimension,))
        self.assertEqual(accepted.dtype, bool)
        self.assertEqual(accepted.shape, ())

    @parameterized.named_parameters(
        *(
            (f"{name}_{seed}", name, seed)
            for name in harness.PRESETS
            for seed in (0, 1, 12345, 99991)
        )
    )
    def test_it_agrees_with_the_transcribed_loop(self, name: str, seed: int) -> None:
        preset = harness.PRESETS[name]
        elements = operand_elements(preset.message_hash_length, seed)

        digits, accepted = encoding.aborting_decode(
            harness.lane_reversed(elements), params=preset
        )

        self.assertTrue(bool(accepted))
        self.assertEqual(
            np.asarray(digits).tolist(), _reference_decode(elements, preset)
        )

    @parameterized.named_parameters(
        *((vector.name, vector) for vector in DECODE_VECTORS)
    )
    def test_the_transcribed_loop_agrees_on_the_vector_inputs(
        self, vector: DecodeVector
    ) -> None:
        # The seeded cases above never abort and never hit a boundary; these do,
        # and running the loop over them costs nothing beyond what upstream was
        # already asked.
        preset = harness.PRESETS[vector.preset]

        digits, accepted = encoding.aborting_decode(
            harness.lane_reversed(vector.elements), params=preset
        )

        expected = _reference_decode(vector.elements, preset)
        self.assertEqual(bool(accepted), expected is not None)
        if expected is not None:
            self.assertEqual(np.asarray(digits).tolist(), expected)


class MessageHashTest(parameterized.TestCase):
    """One compression and its decode, against the reference pipeline."""

    @parameterized.named_parameters(*harness.both_legs(MESSAGE_HASH_VECTORS))
    def test_both_entry_points_match_upstream(
        self, vector: MessageHashVector, jit: bool
    ) -> None:
        # Both in one case rather than two: `target_sum_encode` returns the same
        # digits and only narrows the flag, so a second case would re-run the
        # width-24 compression for one boolean.
        preset = harness.PRESETS[vector.preset]
        operands = _operands(vector, preset)
        expected = _digits(vector.digits)

        digits, accepted = _on(encoding.message_hash, jit)(*operands, params=preset)
        on_layer_digits, on_layer = _on(encoding.target_sum_encode, jit)(
            *operands, params=preset
        )

        self.assertEqual(np.asarray(digits).tolist(), expected)
        # `message_hash`'s flag is the abort's alone, and no vector here reaches
        # it; `target_sum_encode`'s is that one narrowed by the layer.
        self.assertTrue(bool(accepted))
        self.assertEqual(bool(on_layer), vector.on_layer)
        self.assertEqual(np.asarray(on_layer_digits).tolist(), expected)
        self.assertEqual(digits.dtype, np.uint32)
        self.assertEqual(digits.shape, (preset.dimension,))

    def test_an_accepted_codeword_sums_to_the_target(self) -> None:
        # The filter's whole content, stated once so that a vector transcribed
        # with the wrong flag cannot pass quietly.
        for vector in MESSAGE_HASH_VECTORS:
            preset = harness.PRESETS[vector.preset]
            on_layer = sum(_digits(vector.digits)) == preset.target_sum
            self.assertEqual(vector.on_layer, on_layer, vector.name)


class OperandLengthTest(parameterized.TestCase):
    """Each operand is checked on its own, not through the total."""

    @parameterized.named_parameters(
        ("message", 0, "message"),
        ("parameter", 1, "parameter"),
        ("epoch", 2, "epoch"),
        ("randomness", 3, "randomness"),
    )
    def test_it_names_the_operand_that_is_wrong(self, index: int, name: str) -> None:
        operands = list(_operands(MESSAGE_HASH_VECTORS[0], params.PROD))
        operands[index] = operands[index][:-1]

        with self.assertRaisesRegex(ValueError, f"the {name} operand"):
            encoding.message_hash(*operands, params=params.PROD)

    def test_two_wrong_lengths_that_cancel_are_still_caught(self) -> None:
        # `compress` bounds the total against the state width, and 23 elements
        # split 8/6/2/7 fit a width-24 state exactly as well as 9/5/2/7 do — so
        # the total is the one thing that cannot find this.
        operands = list(_operands(MESSAGE_HASH_VECTORS[0], params.PROD))
        operands[0] = operands[0][:-1]
        operands[1] = fnp.concatenate([operands[1], operands[1][:1]])

        with self.assertRaisesRegex(ValueError, "operand"):
            encoding.message_hash(*operands, params=params.PROD)


def _operands(
    vector: MessageHashVector, preset: LeanSigParams
) -> tuple[fnp.ndarray, ...]:
    """The four lane-reversed operands one pipeline case feeds in."""
    message = vector.message.to_bytes(encoding.MESSAGE_BYTES, "little")
    return (
        encoding.encode_message(message, params=preset),
        harness.lane_reversed(
            operand_elements(preset.parameter_length, vector.parameter_seed)
        ),
        encoding.encode_epoch(vector.epoch, params=preset),
        harness.lane_reversed(
            operand_elements(preset.randomness_length, vector.randomness_seed)
        ),
    )


if __name__ == "__main__":
    absltest.main()
