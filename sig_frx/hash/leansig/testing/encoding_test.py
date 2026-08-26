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
from functools import lru_cache
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from zk_dtypes import koalabear_mont as F

from sig_frx.hash.leansig import encoding, params
from sig_frx.hash.leansig.params import LeanSigParams
from sig_frx.hash.leansig.testing.encoding_vectors import (
    DECODE_VECTORS,
    EPOCH_VECTORS,
    MESSAGE_HASH_VECTORS,
    MESSAGE_VECTORS,
    DecodeVector,
    MessageHashVector,
)
from sig_frx.hash.leansig.testing.mode_vectors import operand_elements

PRESETS: dict[str, LeanSigParams] = {"prod": params.PROD, "test": params.TEST}


def _to_field(canonical: Sequence[int]) -> fnp.ndarray:
    """Canonical residues -> a field array, without the reversal."""
    return fnp.asarray(np.asarray(canonical, dtype=np.int64).astype(F))


def _lane_reversed(canonical: Sequence[int]) -> fnp.ndarray:
    """leanSpec-ordered residues -> the lane-reversed vector the pipeline takes."""
    return _to_field(list(canonical)[::-1])


def _to_leanspec_order(elements: fnp.ndarray) -> list[int]:
    """A lane-reversed field vector -> canonical residues in leanSpec's order.

    The object cast is `poseidon_test._to_canonical`'s, for the reason recorded
    there.
    """
    return [int(x) for x in np.asarray(elements).astype(object)][::-1]


def _digits(codeword: str) -> list[int]:
    """A vector's codeword string as the digits it stands for."""
    return [int(digit) for digit in codeword]


@lru_cache(maxsize=None)
def _jitted(function: Callable[..., Any]) -> Callable[..., Any]:
    """One jit wrapper per callable, shared across the cases that trace it.

    `params` is static: it is a frozen dataclass the shapes are derived from, and
    every branch here reads it at trace time. The caching argument is
    `mode_test._jitted`'s.
    """
    return frx.jit(function, static_argnames=("params",))


def _both_legs(vectors: Sequence[Any]) -> list[tuple[str, Any, bool]]:
    """Each vector twice, once eagerly and once traced."""
    return [
        (f"{vector.name}_{'traced' if jit else 'host'}", vector, jit)
        for vector in vectors
        for jit in (False, True)
    ]


def _reference_decode(
    elements: Sequence[int], preset: LeanSigParams
) -> list[int] | None:
    """leanSpec's `aborting_decode`, transcribed — the loop, one element at a time.

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
    def test_it_matches_upstream(self, vector: Any) -> None:
        message = vector.message.to_bytes(encoding.MESSAGE_BYTES, "little")

        got = encoding.encode_message(message, params=params.PROD)

        self.assertEqual(_to_leanspec_order(got), list(vector.elements))
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
    def test_it_matches_upstream(self, vector: Any) -> None:
        got = encoding.encode_epoch(vector.epoch, params=params.PROD)

        self.assertEqual(_to_leanspec_order(got), list(vector.elements))
        self.assertEqual(got.dtype, F)
        self.assertEqual(got.shape, (params.PROD.tweak_length,))

    def test_the_prefix_separates_it_from_the_other_subdomains(self) -> None:
        # Slot 0 carries nothing but the prefix, so this is the one case where
        # the tweak is the domain separator alone.
        got = encoding.encode_epoch(0, params=params.PROD)

        self.assertEqual(_to_leanspec_order(got)[0], params.TWEAK_PREFIX_MESSAGE)

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

    @parameterized.named_parameters(*_both_legs(DECODE_VECTORS))
    def test_it_matches_upstream(self, vector: DecodeVector, jit: bool) -> None:
        preset = PRESETS[vector.preset]
        elements = _lane_reversed(vector.elements)

        decode = _jitted(encoding.aborting_decode) if jit else encoding.aborting_decode
        digits, accepted = decode(elements, params=preset)

        self.assertEqual(bool(accepted), vector.digits is not None)
        if vector.digits is not None:
            self.assertEqual(np.asarray(digits).tolist(), _digits(vector.digits))
        self.assertEqual(digits.dtype, np.uint32)
        self.assertEqual(digits.shape, (preset.dimension,))
        self.assertEqual(accepted.dtype, bool)
        self.assertEqual(accepted.shape, ())

    @parameterized.named_parameters(("host", False), ("traced", True))
    def test_it_decodes_a_batch_entry_by_entry(self, jit: bool) -> None:
        # The digests a verifier decodes arrive stacked, and a decode that
        # reduced across the batch instead of within a vector would pass every
        # single-entry case above.
        rows = [v for v in DECODE_VECTORS if v.preset == "prod"]
        stacked = fnp.stack([_lane_reversed(v.elements) for v in rows])

        decode = _jitted(encoding.aborting_decode) if jit else encoding.aborting_decode
        digits, accepted = decode(stacked, params=params.PROD)

        self.assertEqual(
            np.asarray(accepted).tolist(), [v.digits is not None for v in rows]
        )
        self.assertEqual(
            [np.asarray(row).tolist() for row, v in zip(digits, rows) if v.digits],
            [_digits(v.digits) for v in rows if v.digits],
        )

    @parameterized.named_parameters(
        *(
            (f"{name}_{seed}", name, seed)
            for name in PRESETS
            for seed in (0, 1, 12345, 99991)
        )
    )
    def test_it_agrees_with_the_transcribed_loop(self, name: str, seed: int) -> None:
        preset = PRESETS[name]
        elements = operand_elements(preset.message_hash_length, seed)

        digits, accepted = encoding.aborting_decode(
            _lane_reversed(elements), params=preset
        )

        expected = _reference_decode(elements, preset)
        self.assertTrue(bool(accepted))
        self.assertEqual(np.asarray(digits).tolist(), expected)

    def test_the_transcribed_loop_aborts_where_the_implementation_does(self) -> None:
        preset = params.PROD
        threshold = preset.quotient * preset.base**preset.digits_per_element
        elements = (threshold,) + operand_elements(preset.message_hash_length - 1, 3)

        _, accepted = encoding.aborting_decode(_lane_reversed(elements), params=preset)

        self.assertIsNone(_reference_decode(elements, preset))
        self.assertFalse(bool(accepted))


class MessageHashTest(parameterized.TestCase):
    """One compression and its decode, against the reference pipeline."""

    @parameterized.named_parameters(*_both_legs(MESSAGE_HASH_VECTORS))
    def test_it_matches_upstream(self, vector: MessageHashVector, jit: bool) -> None:
        preset = PRESETS[vector.preset]
        operands = _operands(vector, preset)

        hash_ = _jitted(encoding.message_hash) if jit else encoding.message_hash
        digits, accepted = hash_(*operands, params=preset)

        self.assertEqual(np.asarray(digits).tolist(), _digits(vector.digits))
        # The flag at this stage is the abort's alone — none of these vectors
        # reaches it, and the target-sum narrowing is the next test.
        self.assertTrue(bool(accepted))
        self.assertEqual(digits.dtype, np.uint32)
        self.assertEqual(digits.shape, (preset.dimension,))

    @parameterized.named_parameters(*_both_legs(MESSAGE_HASH_VECTORS))
    def test_the_target_sum_filter_matches_upstream(
        self, vector: MessageHashVector, jit: bool
    ) -> None:
        preset = PRESETS[vector.preset]
        operands = _operands(vector, preset)

        encode = (
            _jitted(encoding.target_sum_encode) if jit else encoding.target_sum_encode
        )
        digits, accepted = encode(*operands, params=preset)

        self.assertEqual(bool(accepted), vector.on_layer)
        self.assertEqual(np.asarray(digits).tolist(), _digits(vector.digits))

    def test_an_accepted_codeword_sums_to_the_target(self) -> None:
        # The filter's whole content, stated once so that a vector transcribed
        # with the wrong flag cannot pass quietly.
        for vector in MESSAGE_HASH_VECTORS:
            preset = PRESETS[vector.preset]
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
        _lane_reversed(
            operand_elements(preset.parameter_length, vector.parameter_seed)
        ),
        encoding.encode_epoch(vector.epoch, params=preset),
        _lane_reversed(
            operand_elements(preset.randomness_length, vector.randomness_seed)
        ),
    )


if __name__ == "__main__":
    absltest.main()
