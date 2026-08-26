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

from absl.testing import absltest, parameterized
from zk_dtypes import koalabear_mont as F

from sig_frx.hash.leansig import poseidon
from sig_frx.hash.leansig.testing import harness
from sig_frx.hash.leansig.testing.mode_vectors import (
    COMPRESSION_VECTORS,
    DOMAIN_SEPARATOR_VECTORS,
    SPONGE_VECTORS,
    CompressionVector,
    DomainSeparatorVector,
    SpongeVector,
    operand_elements,
)


class CompressionTest(parameterized.TestCase):
    """Compression matches upstream at every shape the scheme hashes."""

    @parameterized.named_parameters(*harness.both_legs(COMPRESSION_VECTORS))
    def test_it_matches_upstream(self, vector: CompressionVector, jit: bool) -> None:
        operands = harness.lane_reversed(
            operand_elements(vector.input_length, vector.input_seed)
        )

        compress = (
            harness.jitted(poseidon.compress, "width", "output_length")
            if jit
            else poseidon.compress
        )
        got = compress(
            [operands], width=vector.width, output_length=vector.output_length
        )

        self.assertEqual(harness.to_leanspec_order(got), list(vector.output))
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
            pieces.append(harness.lane_reversed(elements[start : start + length]))
            start += length

        got = poseidon.compress(
            pieces, width=vector.width, output_length=vector.output_length
        )

        self.assertEqual(harness.to_leanspec_order(got), list(vector.output))


class SpongeTest(parameterized.TestCase):
    """The sponge matches upstream across chunk counts, capacities and widths."""

    @parameterized.named_parameters(*harness.both_legs(SPONGE_VECTORS))
    def test_it_matches_upstream(self, vector: SpongeVector, jit: bool) -> None:
        operands = harness.lane_reversed(
            operand_elements(vector.input_length, vector.input_seed)
        )
        capacity = harness.lane_reversed(vector.capacity)

        sponge = (
            harness.jitted(poseidon.sponge, "width", "output_length")
            if jit
            else poseidon.sponge
        )
        got = sponge(
            [operands],
            capacity,
            width=vector.width,
            output_length=vector.output_length,
        )

        self.assertEqual(harness.to_leanspec_order(got), list(vector.output))
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
        got = poseidon.safe_domain_separator(
            vector.lengths, capacity_length=vector.capacity_length
        )

        self.assertEqual(harness.to_leanspec_order(got), list(vector.output))
        self.assertEqual(got.dtype, F)
        self.assertEqual(got.shape, (vector.capacity_length,))


class LaneConventionTest(absltest.TestCase):
    """The reversal is load-bearing, and a caller that forgets it gets a
    different digest rather than a subtly wrong one."""

    def test_feeding_leanspec_order_directly_does_not_match(self) -> None:
        vector = next(v for v in COMPRESSION_VECTORS if v.name == "width16_chain_step")
        elements = operand_elements(vector.input_length, vector.input_seed)

        # The whole mistake: place a leanSpec-ordered operand and read the digest
        # back as if it were leanSpec-ordered too.
        mistaken = poseidon.compress(
            [harness.to_field(elements)],
            width=vector.width,
            output_length=vector.output_length,
        )

        self.assertNotEqual(harness.to_leanspec_order(mistaken), list(vector.output))


class RejectionTest(absltest.TestCase):
    """What the modes refuse, so a caller learns rather than gets a wrong hash."""

    def test_compress_rejects_operands_wider_than_the_state(self) -> None:
        too_wide = harness.lane_reversed(operand_elements(17, 1))

        with self.assertRaisesRegex(ValueError, "do not fit"):
            poseidon.compress([too_wide], width=16, output_length=8)

    def test_compress_rejects_an_output_longer_than_its_input(self) -> None:
        # Upstream's own bound, and it is on the unpadded length: the padding
        # would otherwise be returned as digest.
        short = harness.lane_reversed(operand_elements(4, 1))

        with self.assertRaisesRegex(ValueError, "exceeds"):
            poseidon.compress([short], width=16, output_length=8)

    def test_compress_rejects_no_operands(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one operand"):
            poseidon.compress([], width=16, output_length=8)

    def test_sponge_rejects_a_capacity_that_leaves_no_rate(self) -> None:
        operands = harness.lane_reversed(operand_elements(4, 1))
        capacity = harness.lane_reversed(operand_elements(16, 2))

        with self.assertRaisesRegex(ValueError, "no rate lane"):
            poseidon.sponge([operands], capacity, width=16, output_length=8)


if __name__ == "__main__":
    absltest.main()
