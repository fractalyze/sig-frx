# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""LeanSig's compression and sponge modes against what leanSpec computes.

Every case runs both eagerly and traced, and the two must agree in *dtype* as
well as value — the placement here is all concatenate and slice, which is where
a promotion would go unnoticed by a comparison that only reads residues back.

The vectors are in leanSpec's lane order and the modes work over the reverse of
it, so each case reverses on the way in and back on the way out. That belongs to
this test rather than to the scheme, for the reason
[`poseidon.py`](../poseidon.py) gives: a real caller holds lane-reversed digests
already and reverses nothing.
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

from sig_frx.hash.leansig import poseidon
from sig_frx.hash.leansig.testing.mode_vectors import (
    COMPRESSION_VECTORS,
    DOMAIN_SEPARATOR_VECTORS,
    PRIME,
    SPONGE_VECTORS,
    CompressionVector,
    DomainSeparatorVector,
    SpongeVector,
    operand_elements,
)


def _lane_reversed(canonical: Sequence[int]) -> fnp.ndarray:
    """leanSpec-ordered residues -> the lane-reversed field array the modes take.

    The reversal is on the host, where it is a slice of a tuple rather than a
    device `reverse`.
    """
    return fnp.asarray(np.asarray(canonical[::-1], dtype=np.int64).astype(F))


def _to_leanspec_order(digest: fnp.ndarray) -> list[int]:
    """A lane-reversed digest -> canonical residues in leanSpec's lane order.

    The object cast Montgomery-decodes without needing frx x64, which is why it
    is not a bitcast.
    """
    return [int(x) for x in np.asarray(digest).astype(object)][::-1]


@lru_cache(maxsize=None)
def _jitted(function: Callable[..., fnp.ndarray]) -> Callable[..., fnp.ndarray]:
    """One jit wrapper per callable. A fresh `frx.jit` around the same function
    re-traces every call, so wrapping per case would pay one compile per
    vector."""
    return frx.jit(function, static_argnames=("width", "output_length"))


def _run(
    function: Callable[..., fnp.ndarray], *args: object, jit: bool, **kwargs: object
) -> fnp.ndarray:
    return (_jitted(function) if jit else function)(*args, **kwargs)


def _cases(vectors: Sequence[Any]) -> list[tuple[str, Any, bool]]:
    """Each vector twice, once eagerly and once traced."""
    return [
        (f"{vector.name}_{'traced' if jit else 'host'}", vector, jit)
        for vector in vectors
        for jit in (False, True)
    ]


class CompressionTest(parameterized.TestCase):
    """Compression matches upstream at every shape the scheme hashes."""

    @parameterized.named_parameters(*_cases(COMPRESSION_VECTORS))
    def test_it_matches_upstream(self, vector: CompressionVector, jit: bool) -> None:
        operands = _lane_reversed(
            operand_elements(vector.input_length, vector.input_seed)
        )

        got = _run(
            poseidon.compress,
            [operands],
            jit=jit,
            width=vector.width,
            output_length=vector.output_length,
        )

        self.assertEqual(_to_leanspec_order(got), list(vector.output))
        self.assertEqual(got.dtype, F)
        self.assertEqual(got.shape, (vector.output_length,))


class OperandSplitTest(absltest.TestCase):
    """Splitting the operands changes nothing, which is what lets a caller name
    them the way the spec does."""

    def test_it_ignores_where_the_operands_are_cut(self) -> None:
        vector = next(v for v in COMPRESSION_VECTORS if v.name == "width24_tree_node")
        elements = operand_elements(vector.input_length, vector.input_seed)
        # The Merkle node's own operands: parameter, tweak, left, right.
        cuts = (5, 2, 8, 8)

        pieces, start = [], 0
        for length in cuts:
            pieces.append(_lane_reversed(elements[start : start + length]))
            start += length

        got = poseidon.compress(
            pieces, width=vector.width, output_length=vector.output_length
        )

        self.assertEqual(_to_leanspec_order(got), list(vector.output))


class SpongeTest(parameterized.TestCase):
    """The sponge matches upstream across chunk counts, capacities and widths."""

    @parameterized.named_parameters(*_cases(SPONGE_VECTORS))
    def test_it_matches_upstream(self, vector: SpongeVector, jit: bool) -> None:
        operands = _lane_reversed(
            operand_elements(vector.input_length, vector.input_seed)
        )
        capacity = _lane_reversed(vector.capacity)

        got = _run(
            poseidon.sponge,
            [operands],
            capacity,
            jit=jit,
            width=vector.width,
            output_length=vector.output_length,
        )

        self.assertEqual(_to_leanspec_order(got), list(vector.output))
        self.assertEqual(got.dtype, F)
        self.assertEqual(got.shape, (vector.output_length,))


class DomainSeparatorTest(parameterized.TestCase):
    """The separator matches upstream, so two sponge shapes cannot collide.

    One leg only, unlike the modes: the separator takes a configuration's
    lengths and nothing else, so there is no array to trace over.
    """

    @parameterized.named_parameters(
        *((vector.name, vector) for vector in DOMAIN_SEPARATOR_VECTORS)
    )
    def test_it_matches_upstream(self, vector: DomainSeparatorVector) -> None:
        got = poseidon.safe_domain_separator(vector.lengths, vector.capacity_length)

        self.assertEqual(_to_leanspec_order(got), list(vector.output))
        self.assertEqual(got.dtype, F)
        self.assertEqual(got.shape, (vector.capacity_length,))

    def test_it_separates_two_shapes(self) -> None:
        prod = poseidon.safe_domain_separator((5, 2, 46, 8), 9)
        test = poseidon.safe_domain_separator((5, 2, 4, 8), 9)

        self.assertNotEqual(_to_leanspec_order(prod), _to_leanspec_order(test))


class LaneConventionTest(absltest.TestCase):
    """The reversal is load-bearing, and a caller that forgets it gets a
    different digest rather than a subtly wrong one."""

    def test_feeding_leanspec_order_directly_does_not_match(self) -> None:
        vector = next(v for v in COMPRESSION_VECTORS if v.name == "width16_chain_step")
        elements = operand_elements(vector.input_length, vector.input_seed)

        # The whole mistake: place a leanSpec-ordered operand and read the digest
        # back as if it were leanSpec-ordered too.
        mistaken = poseidon.compress(
            [fnp.asarray(np.asarray(elements, dtype=np.int64).astype(F))],
            width=vector.width,
            output_length=vector.output_length,
        )

        self.assertNotEqual(_to_leanspec_order(mistaken), list(vector.output))

    def test_operands_join_in_spec_order(self) -> None:
        """`_join` reverses the piece order, so operands listed the spec's way
        land contiguously rather than back to front."""
        first, second = (1, 2, 3), (4, 5, 6)

        joined = poseidon._join([_lane_reversed(first), _lane_reversed(second)])

        self.assertEqual(_to_leanspec_order(joined), list(first + second))


class RejectionTest(absltest.TestCase):
    """What the modes refuse, so a caller learns rather than gets a wrong hash."""

    def test_compress_rejects_operands_wider_than_the_state(self) -> None:
        too_wide = _lane_reversed(operand_elements(17, 1))

        with self.assertRaisesRegex(ValueError, "do not fit"):
            poseidon.compress([too_wide], width=16, output_length=8)

    def test_compress_rejects_an_output_longer_than_its_input(self) -> None:
        # Upstream's own bound, and it is on the unpadded length: the padding
        # would otherwise be returned as digest.
        short = _lane_reversed(operand_elements(4, 1))

        with self.assertRaisesRegex(ValueError, "exceeds"):
            poseidon.compress([short], width=16, output_length=8)

    def test_compress_rejects_no_operands(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one operand"):
            poseidon.compress([], width=16, output_length=8)

    def test_sponge_rejects_a_capacity_that_leaves_no_rate(self) -> None:
        operands = _lane_reversed(operand_elements(4, 1))
        capacity = _lane_reversed(operand_elements(16, 2))

        with self.assertRaisesRegex(ValueError, "no rate lane"):
            poseidon.sponge([operands], capacity, width=16, output_length=8)

    def test_a_short_decomposition_is_rejected_rather_than_truncated(self) -> None:
        # Dropping the high part would silently change the hash.
        with self.assertRaisesRegex(ValueError, "base-p limbs"):
            poseidon._int_to_base_p(PRIME**3, 2)


class DecompositionTest(absltest.TestCase):
    """The base-p decomposition the separator's packing rests on.

    Pinned directly rather than only through a separator digest: a digest says
    that something is wrong, not which limb.
    """

    def test_it_is_least_significant_first(self) -> None:
        value = 7 + 11 * PRIME + 13 * PRIME**2

        self.assertEqual(poseidon._int_to_base_p(value, 3), [7, 11, 13])

    def test_it_pads_with_zeros(self) -> None:
        self.assertEqual(poseidon._int_to_base_p(5, 4), [5, 0, 0, 0])


if __name__ == "__main__":
    absltest.main()
