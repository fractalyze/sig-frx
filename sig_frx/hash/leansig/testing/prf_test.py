# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The two derivations a leanSig seed expands into, against leanSpec's.

A PRF has no internal structure to check itself against: every byte of its input
layout is a convention, and a wrong one — a little-endian epoch, a swapped
subdomain byte, a 4-byte counter where upstream writes 8 — produces output that
is uniform, reproducible, and a different scheme. Nothing downstream notices,
because a key generated under a wrong PRF verifies its own signatures perfectly.
So this suite is a transcription check and is meant to read like one.

The lane order is the other half. Upstream returns a digest in its own order and
everything here hashes over the reverse ([`poseidon.py`](../poseidon.py)), so the
cases assert the reversal is applied rather than only that the residues are
right — a set that came back in upstream's order would satisfy any test written
against `sorted()` or against a sum.
"""

from __future__ import annotations

from absl.testing import absltest, parameterized

from sig_frx.hash.leansig import prf
from sig_frx.hash.leansig.testing import harness
from sig_frx.hash.leansig.testing.prf_vectors import (
    CHAIN_STARTS,
    PRF_KEY,
    RANDOMNESS,
    ChainStartVector,
    RandomnessVector,
)

_KEY = bytes.fromhex(PRF_KEY)
_MESSAGE = bytes(range(32))


class ChainStartTest(parameterized.TestCase):
    """`derive_chain_start`, at both presets and across its two input fields."""

    @parameterized.named_parameters(*[(v.name, v) for v in CHAIN_STARTS])
    def test_it_matches_upstream(self, vector: ChainStartVector) -> None:
        params = harness.PRESETS[vector.preset]

        got = prf.chain_starts(
            _KEY, [vector.epoch], [vector.chain_index], params=params
        )

        self.assertEqual(got.shape, (1, params.hash_length))
        self.assertEqual(harness.to_leanspec_rows(got), [list(vector.digest)])

    def test_the_lane_order_is_load_bearing(self) -> None:
        """Read without the reversal, a digest is a different vector.

        The one case that would pass under a forgotten reversal is a palindrome,
        which none of these is — asserted, so a regenerated set that happened to
        produce one fails here rather than silently weakening every case above.
        """
        vector = CHAIN_STARTS[0]
        params = harness.PRESETS[vector.preset]

        got = prf.chain_starts(
            _KEY, [vector.epoch], [vector.chain_index], params=params
        )

        self.assertNotEqual(list(vector.digest), list(reversed(vector.digest)))
        self.assertEqual(harness.to_canonical(got[0]), list(reversed(vector.digest)))

    def test_a_batch_is_the_singular_form_repeated(self) -> None:
        """Every row of a batch is what that chain gets on its own.

        The batch exists for the transfer rather than for the hashing — each
        squeeze is its own input — so a row that differed would mean the columns
        were paired up wrong, which is exactly what a `repeat`/`tile` transpose
        does.
        """
        params = harness.PRESETS["test"]
        epochs = [3, 3, 4, 4]
        chains = [0, 1, 0, 1]

        batched = prf.chain_starts(_KEY, epochs, chains, params=params)

        for row, epoch, chain in zip(batched, epochs, chains, strict=True):
            one = prf.chain_starts(_KEY, [epoch], [chain], params=params)
            self.assertEqual(harness.to_canonical(row), harness.to_canonical(one[0]))


class RandomnessTest(parameterized.TestCase):
    """`derive_randomness`, across the epoch, the message and the counter."""

    @parameterized.named_parameters(*[(v.name, v) for v in RANDOMNESS])
    def test_it_matches_upstream(self, vector: RandomnessVector) -> None:
        params = harness.PRESETS[vector.preset]

        got = prf.randomness(
            _KEY,
            vector.epoch,
            bytes.fromhex(vector.message),
            [vector.counter],
            params=params,
        )

        self.assertEqual(got.shape, (1, params.randomness_length))
        self.assertEqual(harness.to_leanspec_rows(got), [list(vector.randomness)])

    def test_a_block_of_counters_is_the_singular_form_repeated(self) -> None:
        """The block the signer's search tries per pass, row by row."""
        params = harness.PRESETS["test"]
        counters = list(range(5))

        block = prf.randomness(_KEY, 7, _MESSAGE, counters, params=params)

        for row, counter in zip(block, counters, strict=True):
            one = prf.randomness(_KEY, 7, _MESSAGE, [counter], params=params)
            self.assertEqual(harness.to_canonical(row), harness.to_canonical(one[0]))

    def test_every_input_field_changes_the_answer(self) -> None:
        """The epoch, the message and the counter each separate two draws.

        A layout that dropped one of the three would still be a PRF and would
        still be deterministic; what it would lose is the binding that makes a
        signature at one slot useless at another.
        """
        params = harness.PRESETS["test"]
        base = harness.to_canonical(
            prf.randomness(_KEY, 7, _MESSAGE, [0], params=params)[0]
        )
        other_message = bytes([_MESSAGE[0] ^ 1]) + _MESSAGE[1:]

        for label, drawn in [
            ("epoch", prf.randomness(_KEY, 8, _MESSAGE, [0], params=params)),
            ("message", prf.randomness(_KEY, 7, other_message, [0], params=params)),
            ("counter", prf.randomness(_KEY, 7, _MESSAGE, [1], params=params)),
        ]:
            with self.subTest(field=label):
                self.assertNotEqual(base, harness.to_canonical(drawn[0]))


class SubdomainTest(absltest.TestCase):
    """The byte that keeps the two derivations apart."""

    def test_the_two_families_do_not_meet(self) -> None:
        """A chain start and a randomness draw at the same position differ.

        They are the same seed under the same slot, and at `TEST` the randomness
        is 7 elements against a digest's 8 — so this compares the overlap. Only
        the subdomain byte separates them, and dropping it would make a chain
        secret derivable from a published `rho`.
        """
        params = harness.PRESETS["test"]

        start = prf.chain_starts(_KEY, [0], [0], params=params)
        drawn = prf.randomness(_KEY, 0, _MESSAGE, [0], params=params)

        shared = min(params.hash_length, params.randomness_length)
        self.assertNotEqual(
            harness.to_canonical(start[0])[:shared],
            harness.to_canonical(drawn[0])[:shared],
        )


class RefusalTest(absltest.TestCase):
    """What a PRF input will not silently accept."""

    def setUp(self) -> None:
        super().setUp()
        self.params = harness.PRESETS["test"]

    def test_a_seed_of_the_wrong_width(self) -> None:
        """SHAKE128 takes any length, so the width is checked rather than felt."""
        with self.assertRaisesRegex(ValueError, "32 bytes"):
            prf.chain_starts(_KEY[:16], [0], [0], params=self.params)

    def test_a_slot_past_the_four_byte_packing(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[0, 4294967296\)"):
            prf.chain_starts(_KEY, [1 << 32], [0], params=self.params)

    def test_a_chain_index_past_the_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "chain index"):
            prf.chain_starts(_KEY, [0], [self.params.dimension], params=self.params)

    def test_columns_that_do_not_line_up(self) -> None:
        with self.assertRaisesRegex(ValueError, "one chain index per epoch"):
            prf.chain_starts(_KEY, [0, 1], [0], params=self.params)

    def test_a_counter_past_max_tries(self) -> None:
        with self.assertRaisesRegex(ValueError, "counter"):
            prf.randomness(
                _KEY, 0, _MESSAGE, [self.params.max_tries], params=self.params
            )

    def test_a_message_of_the_wrong_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "32-byte root"):
            prf.randomness(_KEY, 0, _MESSAGE[:31], [0], params=self.params)


if __name__ == "__main__":
    absltest.main()
