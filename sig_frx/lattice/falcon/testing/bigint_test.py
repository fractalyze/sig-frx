# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Every operation is checked against Python `int`, which is the whole point.

This module exists because a lane cannot hold Falcon's NTRU intermediates, so
the thing it must be right about is arithmetic the host does natively. That
makes the oracle free and unambiguous: `int` is the specification here, and a
round trip proves nothing on its own — `to_rns` and `from_rns` agreeing with
each other is exactly what a consistently wrong pair of them looks like.

The widths are not arbitrary. 4,732 and 9,427 bits are what
[#26](https://github.com/fractalyze/sig-frx/issues/26) measured the solver's
intermediates reach at `n = 512` and `n = 1024`, so a module that is correct at
smaller widths and wrong at these is wrong where it is used.

## The extremes are separate cases, because the failure is silent

A random 9,427-bit value does not exercise a carry that cascades the length of
the number, and it never puts a limb at the top of its range. Both are where
this module's single width is load-bearing: a limb pair summing past `2^16`, or
a product past `2^30`, wraps in a `uint32` lane without raising — the failure
the repo's first non-negotiable names. So the saturated and cascading cases are
constructed rather than sampled.
"""

from __future__ import annotations

import random

import numpy as np
from absl.testing import absltest, parameterized

from sig_frx.lattice.falcon import bigint

# The solver's measured peaks, plus a small width whose values are checkable by
# hand and whose channel count keeps the exhaustive cases cheap.
WIDTHS = (120, 4732, 9427)


def _modulus(limbs: int) -> int:
    """What every positional operation here is exact modulo."""
    return 1 << (limbs * bigint.LIMB_BITS)


def _product(channels: int) -> int:
    """The modulus the residue form represents uniquely."""
    total = 1
    for modulus in bigint.moduli(channels).tolist():
        total *= modulus
    return total


class PositionalTest(parameterized.TestCase):
    """Limb arithmetic, against `int` at the widths the solver reaches."""

    @parameterized.parameters(*WIDTHS)
    def test_round_trip_through_limbs(self, bits: int) -> None:
        limbs = bigint.limb_count(bits)
        rng = random.Random(bits)
        for value in (
            0,
            1,
            _modulus(limbs) - 1,
            *(rng.getrandbits(bits) for _ in range(8)),
        ):
            value %= _modulus(limbs)
            self.assertEqual(bigint.from_limbs(bigint.to_limbs(value, limbs)), value)

    @parameterized.parameters(*WIDTHS)
    def test_add_and_sub_agree_with_int(self, bits: int) -> None:
        limbs = bigint.limb_count(bits)
        modulus = _modulus(limbs)
        rng = random.Random(bits + 1)
        for _ in range(8):
            left, right = rng.getrandbits(bits), rng.getrandbits(bits)
            packed = bigint.to_limbs(left, limbs), bigint.to_limbs(right, limbs)
            self.assertEqual(
                bigint.from_limbs(bigint.add(*packed)), (left + right) % modulus
            )
            self.assertEqual(
                bigint.from_limbs(bigint.sub(*packed)), (left - right) % modulus
            )

    @parameterized.parameters(*WIDTHS)
    def test_a_carry_cascades_the_whole_length(self, bits: int) -> None:
        """`(2^(15L) - 1) + 1` — a carry that travels from the bottom to the top."""
        limbs = bigint.limb_count(bits)
        saturated = bigint.to_limbs(_modulus(limbs) - 1, limbs)
        one = bigint.to_limbs(1, limbs)
        self.assertEqual(bigint.from_limbs(bigint.add(saturated, one)), 0)
        self.assertEqual(
            bigint.from_limbs(bigint.add(saturated, saturated)),
            _modulus(limbs) - 2,
        )

    @parameterized.parameters(*WIDTHS)
    def test_a_carry_stops_where_it_should(self, bits: int) -> None:
        """`(2^15 - 1) + 1` — one limb generates and no limb above it propagates.

        This is the case that fixes the *direction* of the carry monoid, and the
        cascading case above does not. When every limb propagates, composing the
        monoid the wrong way round gives the same answer — the mutation survives
        a number that is all ones. It only shows up when a carry has somewhere
        to stop: reversed, the carry generated at limb 0 keeps travelling and
        lands in limbs that should have stayed zero.
        """
        limbs = bigint.limb_count(bits)
        self.assertGreaterEqual(limbs, 3, "the case needs a limb above the carry")
        below = bigint.to_limbs(int(bigint.MASK), limbs)
        one = bigint.to_limbs(1, limbs)
        self.assertEqual(bigint.from_limbs(bigint.add(below, one)), int(bigint.BASE))

    @parameterized.parameters(*WIDTHS)
    def test_a_borrow_cascades_the_whole_length(self, bits: int) -> None:
        """`0 - 1` borrows through every limb and wraps to the saturated value."""
        limbs = bigint.limb_count(bits)
        zero, one = bigint.to_limbs(0, limbs), bigint.to_limbs(1, limbs)
        self.assertEqual(bigint.from_limbs(bigint.sub(zero, one)), _modulus(limbs) - 1)
        self.assertEqual(bigint.from_limbs(bigint.sub(zero, one), signed=True), -1)

    @parameterized.parameters(*WIDTHS)
    def test_a_borrow_stops_where_it_should(self, bits: int) -> None:
        """`5*2^15 - 1` — the borrow's direction, for the reason carries have one."""
        limbs = bigint.limb_count(bits)
        minuend = bigint.to_limbs(5 * int(bigint.BASE), limbs)
        one = bigint.to_limbs(1, limbs)
        self.assertEqual(
            bigint.from_limbs(bigint.sub(minuend, one)), 5 * int(bigint.BASE) - 1
        )

    @parameterized.parameters(*WIDTHS)
    def test_mul_small_agrees_with_int_including_both_extremes(self, bits: int) -> None:
        """The saturated limb times the largest scalar is where the lane is tightest."""
        limbs = bigint.limb_count(bits)
        modulus = _modulus(limbs)
        biggest = int(bigint.MASK)
        rng = random.Random(bits + 2)
        cases = [(modulus - 1, biggest), (modulus - 1, 1), (0, biggest)]
        cases += [
            (rng.getrandbits(bits), rng.getrandbits(bigint.LIMB_BITS)) for _ in range(6)
        ]
        for value, scalar in cases:
            got = bigint.mul_small(
                bigint.to_limbs(value % modulus, limbs), np.uint32(scalar)
            )
            self.assertEqual(bigint.from_limbs(got), (value * scalar) % modulus)

    @parameterized.parameters(*WIDTHS)
    def test_shifts_agree_with_int_across_limb_boundaries(self, bits: int) -> None:
        """Amounts that are and are not multiples of the limb width, plus the ends."""
        limbs = bigint.limb_count(bits)
        modulus = _modulus(limbs)
        value = random.Random(bits + 3).getrandbits(bits) % modulus
        packed = bigint.to_limbs(value, limbs)
        amounts = (0, 1, 14, 15, 16, 30, 37, bits // 2, limbs * bigint.LIMB_BITS)
        for amount in amounts:
            with self.subTest(amount=amount):
                self.assertEqual(
                    bigint.from_limbs(bigint.shift_right(packed, amount)),
                    value >> amount,
                )
                self.assertEqual(
                    bigint.from_limbs(bigint.shift_left(packed, amount)),
                    (value << amount) % modulus,
                )

    @parameterized.parameters(*WIDTHS)
    def test_the_dynamic_shift_matches_the_one_that_knows_its_amount(
        self, bits: int
    ) -> None:
        """Same answers as [`shift_left`](../bigint.py), off a traced amount.

        The two exist for different reasons — one is cheaper and the other
        compiles once for a whole loop — so the property that matters is that
        they cannot disagree. Both ends of the range are in the sweep: an
        amount of zero takes the intra-limb path with nothing to carry across,
        and one past the budget has to shift the value away rather than wrap it.
        """
        limbs = bigint.limb_count(bits)
        modulus = _modulus(limbs)
        value = random.Random(bits + 7).getrandbits(bits) % modulus
        packed = bigint.to_limbs(value, limbs)
        amounts = (0, 1, 14, 15, 16, 29, 30, 37, bits // 2, limbs * bigint.LIMB_BITS)
        for amount in amounts:
            with self.subTest(amount=amount):
                got = bigint.shift_left_dynamic(packed, np.int32(amount))
                self.assertEqual(bigint.from_limbs(got), (value << amount) % modulus)
                np.testing.assert_array_equal(
                    np.asarray(got), np.asarray(bigint.shift_left(packed, amount))
                )

    def test_the_dynamic_shift_broadcasts_over_a_whole_polynomial(self) -> None:
        """One amount against `[m, L]`, and one amount *per* value.

        The first is how the reduction asks — a single exponent scales the
        whole correction. The second is not asked for yet and is tested anyway,
        because the amount is indexed against the limb axis and a shape that
        broadcast against the wrong axis would answer rather than raise. The
        row count here is deliberately *not* the limb count, which is the case
        where such a mistake would go unnoticed.
        """
        limbs = bigint.limb_count(120)
        modulus = _modulus(limbs)
        values = [0, 1, (1 << 100) - 1, 1 << 100]
        stacked = np.stack([bigint.to_limbs(v, limbs) for v in values])
        self.assertNotEqual(len(values), limbs)

        shared = bigint.shift_left_dynamic(stacked, np.int32(17))
        for row, value in zip(np.asarray(shared), values):
            self.assertEqual(bigint.from_limbs(row), (value << 17) % modulus)

        amounts = np.array([0, 15, 29, 44], dtype=np.int32)
        per_value = bigint.shift_left_dynamic(stacked, amounts)
        for row, value, amount in zip(np.asarray(per_value), values, amounts.tolist()):
            self.assertEqual(bigint.from_limbs(row), (value << amount) % modulus)

    @parameterized.parameters(*WIDTHS)
    def test_the_signed_shift_fills_from_the_sign_not_from_zero(
        self, bits: int
    ) -> None:
        """A negative operand is the whole point, so both signs run the sweep.

        `shift_right` fills with zeros, which is right for a magnitude and turns
        a negative two's-complement value into something enormous. Python's `>>`
        is arithmetic, so it is the oracle for both halves here.
        """
        limbs = bigint.limb_count(bits)
        modulus = _modulus(limbs)
        magnitude = random.Random(bits + 5).getrandbits(bits - 1) % modulus
        amounts = (0, 1, 14, 15, 16, 30, 37, bits // 2, limbs * bigint.LIMB_BITS)
        for value in (magnitude, -magnitude):
            packed = bigint.to_limbs(value % modulus, limbs)
            for amount in amounts:
                with self.subTest(value=value, amount=amount):
                    self.assertEqual(
                        bigint.from_limbs(
                            bigint.shift_right_signed(packed, amount), signed=True
                        ),
                        value >> amount,
                    )

    def test_at_least_compares_many_values_against_one_bound(self) -> None:
        """`[m, L]` against `[L]`, which is how every caller actually asks.

        The elementwise operations broadcast for free and this one does not: it
        gathers at an index derived from both operands, so a shape mismatch
        surfaced as a gather error rather than as an answer. Found by the
        descent above this, whose centering step compares a whole polynomial
        against a single modulus.
        """
        limbs = bigint.limb_count(120)
        bound = bigint.to_limbs(1 << 100, limbs)
        values = [0, (1 << 100) - 1, 1 << 100, (1 << 100) + 1, (1 << 119)]
        stacked = np.stack([bigint.to_limbs(v, limbs) for v in values])
        got = np.asarray(bigint.at_least(stacked, bound))
        np.testing.assert_array_equal(got, np.array([v >= 1 << 100 for v in values]))

    @parameterized.parameters(*WIDTHS)
    def test_at_least_orders_by_magnitude(self, bits: int) -> None:
        """Including the equal case, which has no most-significant differing limb."""
        limbs = bigint.limb_count(bits)
        rng = random.Random(bits + 4)
        pairs = [(0, 0), (0, 1), (1, 0), (_modulus(limbs) - 1, _modulus(limbs) - 1)]
        pairs += [(rng.getrandbits(bits), rng.getrandbits(bits)) for _ in range(6)]
        # A pair differing only in its least significant limb: the reduction has
        # to find that limb rather than stop at the first one that differs.
        near = rng.getrandbits(bits) | 1
        pairs.append((near, near - 1))
        for left, right in pairs:
            left, right = left % _modulus(limbs), right % _modulus(limbs)
            with self.subTest(left=left % 97, right=right % 97):
                got = bigint.at_least(
                    bigint.to_limbs(left, limbs), bigint.to_limbs(right, limbs)
                )
                self.assertEqual(bool(np.asarray(got)), left >= right)


class ResidueTest(parameterized.TestCase):
    """The residue form and the bridge, against `int` rather than against itself."""

    @parameterized.parameters(*WIDTHS)
    def test_channel_count_covers_the_width(self, bits: int) -> None:
        """The chosen primes are under `2^15`, so `bits / 15` is short of enough."""
        channels = bigint.channel_count(bits)
        self.assertGreater(_product(channels), 1 << bits)
        self.assertLessEqual(_product(channels - 1), 1 << bits)

    def test_the_moduli_are_distinct_primes(self) -> None:
        """Pairwise coprimality is what the CRT needs; primality is how it is had."""
        chosen = bigint.moduli(bigint.channel_count(max(WIDTHS))).tolist()
        self.assertLen(set(chosen), len(chosen))
        for modulus in (chosen[0], chosen[len(chosen) // 2], chosen[-1]):
            self.assertTrue(all(modulus % f for f in range(2, int(modulus**0.5) + 1)))
            self.assertLess(modulus, 1 << bigint.LIMB_BITS)

    @parameterized.parameters(*WIDTHS)
    def test_to_rns_agrees_with_int_remainders(self, bits: int) -> None:
        """Checked against `%` per channel, not against `from_rns`."""
        limbs, channels = bigint.limb_count(bits), bigint.channel_count(bits)
        mods = bigint.moduli(channels)
        rng = random.Random(bits + 5)
        for value in (0, 1, _product(channels) - 1, rng.getrandbits(bits)):
            value %= min(_product(channels), _modulus(limbs))
            got = np.asarray(bigint.to_rns(bigint.to_limbs(value, limbs), channels))
            want = np.array([value % int(m) for m in mods], dtype=np.uint32)
            np.testing.assert_array_equal(got, want)

    @parameterized.parameters(*WIDTHS)
    def test_rns_arithmetic_agrees_with_int_at_the_top_of_the_range(
        self, bits: int
    ) -> None:
        """`(m-1)*(m-1)` is where a 15-bit product is closest to leaving the lane."""
        channels = bigint.channel_count(bits)
        mods = bigint.moduli(channels)
        saturated = (mods - np.uint32(1)).astype(np.uint32)
        expected_mul = np.array(
            [(int(m) - 1) ** 2 % int(m) for m in mods], dtype=np.uint32
        )
        np.testing.assert_array_equal(
            np.asarray(bigint.rns_mul(saturated, saturated, mods)), expected_mul
        )
        np.testing.assert_array_equal(
            np.asarray(bigint.rns_add(saturated, saturated, mods)),
            np.array([(int(m) - 1) * 2 % int(m) for m in mods], dtype=np.uint32),
        )
        np.testing.assert_array_equal(
            np.asarray(bigint.rns_sub(np.zeros_like(saturated), saturated, mods)),
            np.array([(-(int(m) - 1)) % int(m) for m in mods], dtype=np.uint32),
        )

    @parameterized.parameters(*WIDTHS)
    def test_from_rns_reconstructs_the_integer(self, bits: int) -> None:
        """The bridge closes: `int` in, residues, `int` out, same value."""
        limbs, channels = bigint.limb_count(bits), bigint.channel_count(bits)
        ceiling = min(_product(channels), _modulus(limbs))
        rng = random.Random(bits + 6)
        for value in (0, 1, ceiling - 1, rng.getrandbits(bits) % ceiling):
            residues = bigint.to_rns(bigint.to_limbs(value, limbs), channels)
            self.assertEqual(
                bigint.from_limbs(bigint.from_rns(residues, channels, limbs)), value
            )

    def test_the_bridge_refuses_a_limb_budget_it_could_not_sum(self) -> None:
        """The accumulator's bound is a check, because passing it is silent.

        Both bridge directions reduce each limb and then sum, so the accumulator
        holds `L` values under `2^15`. Past `MAX_LIMBS` that sum leaves a
        `uint32` without raising — and a wrong answer that raises nothing is
        what the repo's first non-negotiable is about. Nothing in Falcon comes
        within four orders of magnitude of the bound; the check is here for the
        caller that one day does.
        """
        # The bound is checked against the lane rather than against itself: a
        # case built from `MAX_LIMBS + 1` follows the constant wherever it goes
        # and passes for a bound that is wrong by orders of magnitude, which is
        # what this assertion first did.
        self.assertLess(bigint.MAX_LIMBS * int(bigint.MASK), 1 << 32)
        self.assertGreater(bigint.MAX_LIMBS, 100 * bigint.limb_count(max(WIDTHS)))

        channels = bigint.channel_count(120)
        too_many = bigint.MAX_LIMBS + 1
        with self.assertRaisesRegex(ValueError, "exceeds"):
            bigint.to_rns(np.zeros(too_many, dtype=np.uint32), channels)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            bigint.from_rns(np.zeros(channels, dtype=np.uint32), channels, too_many)

    def test_the_bridge_carries_a_product_no_lane_could_hold(self) -> None:
        """The operation the two forms exist for: multiply as residues, read as `int`.

        A positional multiply is deliberately absent, so this is the only way the
        module can multiply two wide values at all — and it is what the descent
        above will do at every level.
        """
        bits = 4700
        limbs, channels = bigint.limb_count(2 * bits + 1), bigint.channel_count(
            2 * bits + 1
        )
        rng = random.Random(11)
        left, right = rng.getrandbits(bits), rng.getrandbits(bits)
        mods = bigint.moduli(channels)
        product = bigint.rns_mul(
            bigint.to_rns(bigint.to_limbs(left, limbs), channels),
            bigint.to_rns(bigint.to_limbs(right, limbs), channels),
            mods,
        )
        got = bigint.from_limbs(bigint.from_rns(product, channels, limbs))
        self.assertEqual(got, left * right)
        self.assertGreater((left * right).bit_length(), 9000)


if __name__ == "__main__":
    absltest.main()
