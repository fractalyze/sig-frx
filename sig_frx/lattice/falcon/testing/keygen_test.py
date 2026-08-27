# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Key generation against Python integers, and against the distribution itself.

Two halves with two different oracles. Algorithm 6's recursion and its base case
are exact integer arithmetic, so the oracle is exact integer arithmetic.
Algorithm 5's draw is a *distribution*, which no single output can be right or
wrong about — so it is checked against `exp(-x²/2σ²)` normalised, over enough
samples that the sampling noise is smaller than the thing being asserted.

Every level of Algorithm 6's recursion is compared against
[`falcon_reference.field_norm`](falcon_reference.py), which is the definition
looped one product at a time over unbounded integers. That is the only oracle
available here: the coefficients pass 9,000 bits, so no array type can hold the
intermediate and no published vector exists for a step inside key generation.

## The last level is the one that matters, and the cheap tests never reach it

The descent's whole difficulty is that widths double on the way down, so a test
that stops at a small degree exercises the easy end and nothing else. The cases
below run the **full** descent at both parameter sets — nine levels to 3,141
bits at `n = 512` and ten to 6,327 at `n = 1024` — and check every level, not
only the last.

That the actual widths land where
[#26](https://github.com/fractalyze/sig-frx/issues/26) measured them is asserted
too. It is what says the descent is the *right* recursion rather than merely a
self-consistent one: a wrong split or a wrong wrap sign still produces a
polynomial, still descends, and still ends up at degree 1.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

import numpy as np
from absl.testing import absltest, parameterized

from sig_frx.lattice.falcon import arith, bigint, fft, keygen
from sig_frx.lattice.falcon.testing import falcon_reference

# The widths #26 measured a random key's descent to reach, at `n = 1024`. The
# entry is `f` itself; the rest are one level of `N` each. Sampling moves these
# by a bit or two, so they are a corridor rather than an equality.
MEASURED_WIDTHS = (4, 11, 24, 51, 102, 203, 406, 807, 1593, 3151, 6302)


def _draw(degree: int, seed: int) -> list[int]:
    """A cheap stand-in for the draw, for the tests that are not about the draw.

    A rounded continuous Gaussian, where [`keygen.draw_polynomial`](../keygen.py)
    is the real discrete one. Deliberately not that: the descent's tests want a
    coefficient distribution that is fixed, seeded and independent of the
    sampler, so that a change to the sampler cannot move the width corridor they
    assert. The two differ in tail shape and not in variance, and it is variance
    that sets the widths.
    """
    rng = np.random.default_rng(seed)
    draws = rng.normal(0.0, keygen.DRAW_SIGMA, size=(degree, 4096 // degree))
    return [int(v) for v in np.round(draws).astype(np.int64).sum(axis=1)]


def _drawn(degree: int, rng: np.random.Generator) -> np.ndarray:
    """`draw_polynomial` at `degree`, over the `[4096, 8]` bytes it consumes."""
    stream = rng.integers(0, 256, size=(4096, 8), dtype=np.uint8)
    return np.asarray(keygen.draw_polynomial(degree, stream))


def _squared_norm(
    f: Sequence[int] | np.ndarray, g: Sequence[int] | np.ndarray
) -> float:
    """`gram_schmidt_squared_norm` as a host float, scope included.

    Opening a scope and returning is what [`fft.double_precision`](../fft.py)
    warns against — but what crosses the boundary here is a Python float,
    materialised inside the scope, so there is no array left to narrow.
    """
    with fft.double_precision():
        return float(np.asarray(keygen.gram_schmidt_squared_norm(f, g)))


def _unpack(limbs: object) -> list[int]:
    """Limbs back to the signed integers the oracle speaks in."""
    return [
        bigint.from_limbs(np.asarray(row), signed=True) for row in np.asarray(limbs)
    ]


class WidthBoundTest(absltest.TestCase):
    """`norm_bits` has to be an upper bound, and be one for the right reason."""

    def test_the_bound_is_attained_by_the_input_that_saturates_it(self) -> None:
        """The saturating input is not the obvious one, and the obvious one is slack.

        Setting every coefficient to `+max` looks like the extreme case and is
        not: the negacyclic wrap subtracts half the products, so they cancel and
        the result lands 2-5 bits under the bound. A bound one bit too small
        survives that case.

        What saturates it is the sign flipping every *two* positions — after the
        split by parity each half is constant, so no product cancels and the
        wrap adds. Asserting the slack from both sides is what pins the bound:
        below catches a bound that is too small to be safe, above catches one
        loose enough to be paying for channels it does not need. One bit of
        slack remains because a coefficient is `2^w - 1` rather than `2^w`.
        """
        for degree, bits in ((4, 3), (8, 5), (16, 5), (32, 4), (64, 7)):
            with self.subTest(degree=degree, bits=bits):
                extreme = (1 << bits) - 1
                saturating = [
                    extreme if (i // 2) % 2 == 0 else -extreme for i in range(degree)
                ]
                result = falcon_reference.field_norm(saturating)
                widest = max(abs(v) for v in result).bit_length()
                bound = keygen.norm_bits(bits, degree)
                self.assertLessEqual(widest, bound)
                self.assertLessEqual(bound - widest, 1)

    def test_the_bound_refuses_a_degree_the_descent_has_no_step_for(self) -> None:
        for degree in (0, 1, 3, 12):
            with self.subTest(degree=degree):
                with self.assertRaisesRegex(ValueError, "power of two"):
                    keygen.norm_bits(4, degree)


class FieldNormTest(parameterized.TestCase):
    """One level, against the definition."""

    @parameterized.parameters(8, 16, 32, 64)
    def test_agrees_with_the_reference_at_small_degrees(self, degree: int) -> None:
        source = _draw(degree, degree)
        bits = max(abs(v) for v in source).bit_length()
        result, _ = keygen.field_norm(keygen.to_limbs(source, bits), bits)
        self.assertEqual(_unpack(result), falcon_reference.field_norm(source))

    def test_a_negative_coefficient_survives_the_round_trip(self) -> None:
        """The descent is signed end to end, and the bridge is where that is lost.

        Residues carry a negative value as `x + M`, which is a correct residue
        of the wrong integer — nothing downstream can tell. So a case whose
        coefficients are deliberately all negative, where an unsigned bridge
        would still produce a polynomial of the right degree.
        """
        source = [-((i % 7) + 1) for i in range(16)]
        result, _ = keygen.field_norm(keygen.to_limbs(source, 3), 3)
        self.assertEqual(_unpack(result), falcon_reference.field_norm(source))

    def test_the_wrap_sign_is_the_negacyclic_one(self) -> None:
        """`y^(m/2) = -1`, not `+1`, and only one case tells them apart.

        `f = x` has `f_o = [1, 0]`, so `f_o² = [1, 0]` under either wrap and
        `N(f) = -y` either way — the case proves nothing. `f = x³` has
        `f_o = [0, 1]`, and there the wrap decides: negacyclically `y² = -1`
        gives `f_o² = -1` and `N(f) = +y`, while cyclically `y² = +1` gives
        `f_o² = +1` and `N(f) = -y`. So the second case is the test and the
        first is only the control.
        """
        for source, expected in (([0, 1, 0, 0], [0, -1]), ([0, 0, 0, 1], [0, 1])):
            with self.subTest(source=source):
                result, _ = keygen.field_norm(keygen.to_limbs(source, 2), 2)
                self.assertEqual(_unpack(result), expected)
                self.assertEqual(falcon_reference.field_norm(source), expected)


class DescentTest(parameterized.TestCase):
    """The whole recursion, at the degrees Falcon actually defines."""

    @parameterized.parameters(*arith.DEGREES)
    def test_every_level_agrees_and_lands_where_the_issue_measured(
        self, degree: int
    ) -> None:
        """Three assertions per level, off one descent.

        They are together because the oracle is `O(n²)` over integers thousands
        of bits wide, so it is the expensive half of this file and running it
        twice to ask two questions of the same numbers buys nothing.

        The width corridor is what says the descent is the *right* recursion
        rather than a self-consistent one: a wrong split or a wrong wrap sign
        still produces a polynomial, still halves its degree, and still arrives
        at degree 1. It is a corridor and not an equality because the draw is
        random — a level's width moves by a bit or two between keys, which is
        the spread the reference implementation's table carries as a standard
        deviation of under 30 bits at the widest level.

        **Both degrees index the same row.** A width follows how many squarings
        have happened, not what degree they happened at — `n = 512` and
        `n = 1024` both enter at 4 bits, so level `d` lands on the same width
        for both and `n = 512` simply stops a level earlier. The reference
        implementation's table is indexed the same way, by depth alone.
        """
        source = _draw(degree, degree + 1)
        bits = max(abs(v) for v in source).bit_length()
        levels = degree.bit_length() - 1
        chain = keygen.descend(keygen.to_limbs(source, bits), bits, levels)

        self.assertLen(chain, levels)
        expected = source
        for level, (limbs, bound) in enumerate(chain, start=1):
            expected = falcon_reference.field_norm(expected)
            widest = max(abs(v) for v in expected).bit_length()
            with self.subTest(level=level):
                self.assertEqual(_unpack(limbs), expected)
                self.assertLen(expected, degree >> level)
                self.assertLessEqual(widest, bound)
                self.assertAlmostEqual(
                    widest,
                    MEASURED_WIDTHS[level],
                    delta=max(8, MEASURED_WIDTHS[level] // 20),
                )


def _pack(value: int, bits: int) -> np.ndarray:
    """One signed integer as the `[limbs]` the base case takes."""
    limbs = bigint.limb_count(bits + 2)
    return np.asarray(bigint.to_limbs(value % (1 << (limbs * bigint.LIMB_BITS)), limbs))


def _solve_batch(
    pairs: Sequence[tuple[int, int]], bits: int
) -> tuple[list[int], list[int], Any]:
    """Every pair through one `base_case` call, as `[B, limbs]`.

    One call rather than one per pair, and not only for the runtime: the seam
    takes a trailing limb axis and says nothing about what is in front of it,
    so a batch is what checks that claim. Tracing the loop once per pair also
    costs about a second each, which is most of this target's budget.
    """
    stacked_f = np.stack([_pack(f0, bits) for f0, _ in pairs])
    stacked_g = np.stack([_pack(g0, bits) for _, g0 in pairs])
    big_f, big_g, ok = keygen.base_case(stacked_f, stacked_g, bits, arith.Q)
    return _unpack(big_f), _unpack(big_g), ok


def _coprime_pair(bits: int, seed: int) -> tuple[int, int]:
    """A pair the base case can solve: coprime, and not both even."""
    rng = random.Random(seed)
    while True:
        f0 = rng.randrange(1 << (bits - 1), 1 << bits)
        g0 = rng.randrange(1 << (bits - 1), 1 << bits)
        if math.gcd(f0, g0) == 1 and not (f0 % 2 == 0 and g0 % 2 == 0):
            return f0, g0


def _binary_gcd_steps(x: int, y: int) -> int:
    """Steps HAC 14.61 takes on two positive magnitudes, counted not bounded.

    The traced form runs a fixed budget, so what it needs is the number this
    would have stopped at. One step is one of the four exclusive branches — the
    flattening `base_case` applies to HAC's nested `while`s.

    Here rather than in [`falcon_reference`](falcon_reference.py) precisely
    because it mirrors the implementation: that module transcribes the
    specification so the tests can hold the code to something independent of it,
    and a deliberate copy of the code's own loop shape would dilute the claim
    four other targets lean on. What this counts is a property of the loop, so
    it belongs beside the test that pins the loop's budget.
    """
    if x % 2 == 0 and y % 2 == 0:
        raise ValueError("both even is HAC's step 2, which the base case rejects")
    u, v = x, y
    a, b, c, d = 1, 0, 0, 1
    steps = 0
    while u:
        steps += 1
        if u % 2 == 0:
            u //= 2
            if a % 2 == 0 and b % 2 == 0:
                a, b = a // 2, b // 2
            else:
                a, b = (a + y) // 2, (b - x) // 2
        elif v % 2 == 0:
            v //= 2
            if c % 2 == 0 and d % 2 == 0:
                c, d = c // 2, d // 2
            else:
                c, d = (c + y) // 2, (d - x) // 2
        elif u >= v:
            u, a, b = u - v, a - c, b - d
        else:
            v, c, d = v - u, c - a, d - b
    return steps


class GcdBudgetTest(absltest.TestCase):
    """The trip count is a correctness constant, so it is bounded, not sampled."""

    def test_the_bound_covers_the_worst_case_the_loop_can_reach(self) -> None:
        # Euclid's worst case is consecutive Fibonacci numbers, and the binary
        # variant inherits it. A budget that is short does not raise — it
        # returns a pair that is simply not Bezout's — so this is the check
        # that the constant is a bound rather than an average.
        previous, current = 1, 1
        for _ in range(2000):
            previous, current = current, previous + current
        steps = _binary_gcd_steps(current, previous)
        bits = max(current.bit_length(), previous.bit_length())
        self.assertLessEqual(steps, keygen.gcd_budget(bits))

    def test_the_earlier_probes_trip_count_is_below_the_worst_case(self) -> None:
        # #26's probe sized this loop at `2 * bits` while measuring whether it
        # compiles. That is under the worst case, so it was never a budget —
        # pinned here so the smaller constant cannot come back as one.
        worst = 0
        for seed in range(40):
            f0, g0 = _coprime_pair(256, seed)
            worst = max(worst, _binary_gcd_steps(f0, g0))
        self.assertGreater(worst, 2 * 256)
        self.assertLessEqual(worst, keygen.gcd_budget(256))


class BaseCaseTest(parameterized.TestCase):
    """Algorithm 6 at degree 1, against the equation it exists to satisfy."""

    def test_a_sign_rides_into_the_coefficient(self) -> None:
        # A sign on either side and on both: the descent's coefficients are
        # signed and the magnitudes the loop runs on are not. One batched call
        # rather than four, which is also what the seam is shaped for.
        bits = 24
        pairs = ((17, 5), (-17, 5), (17, -5), (-17, -5))
        big_f, big_g, ok = _solve_batch(pairs, bits)

        self.assertEqual(list(np.asarray(ok)), [True] * len(pairs))
        for (f0, g0), value_f, value_g in zip(pairs, big_f, big_g):
            with self.subTest(f0=f0, g0=g0):
                self.assertEqual(f0 * value_g - g0 * value_f, arith.Q)

    def test_refuses_what_it_cannot_solve(self) -> None:
        bits = 24
        cases = (
            (6, 4, "both even, so the gcd is at least two"),
            (44450, 624, "both even, and the loop lands on one spuriously"),
            (15, 9, "an odd common factor"),
            (0, 7, "a zero the halving branch never leaves"),
            (7, 0, "a zero on the other side"),
        )
        _, _, ok = _solve_batch([(f0, g0) for f0, g0, _ in cases], bits)

        for (f0, g0, why), verdict in zip(cases, np.asarray(ok)):
            with self.subTest(f0=f0, g0=g0):
                self.assertFalse(bool(verdict), why)

    @parameterized.parameters(64, 256)
    def test_the_verdict_is_coprimality_and_nothing_else(self, bits: int) -> None:
        # The flag has to track `gcd == 1` over inputs drawn without regard to
        # it, rather than only over pairs chosen to be solvable.
        rng = random.Random(bits)
        pairs: list[tuple[int, int]] = []
        while len(pairs) < 32:
            f0 = rng.randrange(-(1 << bits), 1 << bits)
            g0 = rng.randrange(-(1 << bits), 1 << bits)
            if f0 and g0:
                pairs.append((f0, g0))
        _, _, ok = _solve_batch(pairs, bits)

        self.assertEqual(
            [bool(v) for v in np.asarray(ok)],
            [math.gcd(f0, g0) == 1 for f0, g0 in pairs],
        )

    @parameterized.named_parameters(
        ("toy_64", 64),
        ("toy_256", 256),
        # The widths #26 measured at the bottom of the descent. Everything else
        # here runs at toy sizes; these are the ones that say the budget and the
        # limb count hold where they have to.
        ("falcon_512", 3161),
        ("falcon_1024", 6302),
    )
    def test_solves_the_ntru_equation_exactly(self, bits: int) -> None:
        f0, g0 = _coprime_pair(bits, seed=bits)
        # A rank-1 call rather than `_solve_batch`: the seam takes a bare
        # `[limbs]` as readily as a batch, and that is worth exercising once.
        big_f, big_g, ok = keygen.base_case(
            _pack(f0, bits), _pack(g0, bits), bits, arith.Q
        )

        self.assertTrue(bool(np.asarray(ok)))
        value_f = bigint.from_limbs(np.asarray(big_f), signed=True)
        value_g = bigint.from_limbs(np.asarray(big_g), signed=True)
        self.assertEqual(f0 * value_g - g0 * value_f, arith.Q)
        # A register that wrapped could still satisfy nothing above, but it
        # would also blow the size the levels above budget for, so both are
        # checked rather than only the equation.
        self.assertLessEqual(abs(value_f), 2 * arith.Q * abs(f0))
        self.assertLessEqual(abs(value_g), 2 * arith.Q * abs(g0))


class DrawTest(absltest.TestCase):
    """The Gaussian draw, against the distribution it is supposed to be."""

    def test_the_table_tops_out_at_the_tail(self) -> None:
        """The last row is a sentinel no uniform reaches, which caps the count.

        A magnitude is how many thresholds a uniform passes, so the table needs
        one more row than the largest magnitude — set at exactly `2^62`, which
        a `62`-bit uniform cannot reach. Without it the count could run one past
        the tail the table was built for.
        """
        table = keygen._draw_table()

        def threshold(row: np.ndarray) -> int:
            return (int(row[0]) << keygen._HALF_BITS) | int(row[1])

        self.assertEqual(threshold(table[-1]), 1 << keygen.DRAW_BITS)
        self.assertLess(threshold(table[-2]), 1 << keygen.DRAW_BITS)

    def test_the_tail_is_past_the_quantisation(self) -> None:
        """Truncating the table must cost less than rounding it does."""
        sigma = keygen.DRAW_SIGMA
        tail = keygen._draw_table().shape[0] - 1
        weight = math.exp(-(tail**2) / (2 * sigma**2))
        self.assertLess(weight, 2.0**-keygen.DRAW_BITS)

    def test_the_draw_matches_the_discrete_gaussian(self) -> None:
        """Compared against `exp(-x²/2σ²)` normalised, not against itself.

        `degree = 4096` is one draw per coefficient, which is the only shape
        that exposes the sampler rather than a sum of samplers.
        """
        rng = np.random.default_rng(0)
        samples = np.concatenate([_drawn(4096, rng) for _ in range(24)])
        sigma = keygen.DRAW_SIGMA
        partition = sum(math.exp(-x * x / (2 * sigma**2)) for x in range(-60, 61))
        noise = 4.0 / math.sqrt(samples.size)
        for x in range(-5, 6):
            ideal = math.exp(-x * x / (2 * sigma**2)) / partition
            with self.subTest(x=x):
                got = float(np.count_nonzero(samples == x)) / samples.size
                self.assertAlmostEqual(got, ideal, delta=noise)

    def test_summing_scales_the_variance_by_the_degree(self) -> None:
        """`4096/n` draws per coefficient is what makes one table serve every set.

        The standard's `σ_{f,g}` depends on `n`; this sampler does not, and the
        sum is where the dependence comes from. A draw that ignored the sum
        would still look Gaussian and would have the wrong width.
        """
        rng = np.random.default_rng(1)
        # Eight rounds rather than twenty-four: the margin is 6.8 sigma at the
        # worst degree, which cannot flake, and a draw costs the same whatever
        # `degree` is — it always consumes `[4096, 8]` and only the yield differs.
        for degree in (512, 1024, 4096):
            samples = np.concatenate([_drawn(degree, rng) for _ in range(8)])
            ideal = (4096 // degree) * keygen.DRAW_SIGMA**2
            with self.subTest(degree=degree):
                self.assertAlmostEqual(float(samples.var()), ideal, delta=0.15 * ideal)


class QualityCheckTest(parameterized.TestCase):
    """Algorithm 5's two rejections, against the specification's own form."""

    @parameterized.parameters(8, 64)
    def test_the_gram_schmidt_norm_agrees_with_the_reference(self, degree: int) -> None:
        """Small integers rather than the real draw, which this is not about.

        The assertion is a numeric identity against the transcription, and any
        small integer polynomial exercises it the same way — where a real draw
        costs a `[4096, 8]` stream and a transform per case to yield `degree`
        coefficients, and couples this test to the sampler.
        """
        rng = np.random.default_rng(degree)
        pair = [rng.integers(-6, 7, size=degree).tolist() for _ in range(2)]
        got = _squared_norm(*pair)
        want = falcon_reference.gram_schmidt_squared_norm(*pair)
        self.assertAlmostEqual(got / want, 1.0, delta=1e-9)

    def test_the_first_row_wins_when_the_basis_is_badly_skewed(self) -> None:
        """Both rows are read, and a test drawing only random pairs reads one.

        A near-zero `f` and `g` make the *second* row enormous and the first
        tiny; scaling them up inverts which one the maximum comes from. Without
        a case on each side, dropping either from the maximum still passes.
        """
        tiny = _squared_norm([1] + [0] * 7, [0] * 8)
        large = _squared_norm([4000] * 8, [0, 4000, 0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(tiny, arith.Q**2, delta=1.0)  # second row
        self.assertGreater(large, 8 * 4000**2)  # first row

    @parameterized.parameters(*arith.DEGREES)
    def test_invertibility_agrees_with_direct_evaluation(self, degree: int) -> None:
        rng = np.random.default_rng(degree + 7)
        drawn = _drawn(degree, rng)
        self.assertEqual(
            bool(np.asarray(keygen.invertible(drawn))),
            falcon_reference.is_invertible(drawn.tolist()),
        )

    @parameterized.parameters(*arith.DEGREES)
    def test_one_zero_evaluation_is_enough_to_refuse(self, degree: int) -> None:
        """The case that separates "every evaluation" from "some evaluation".

        A random draw is invertible and the zero polynomial is not, and both
        answer the same under `all` and under `any` — so a check reading `any`
        passes a test built from those two. What tells them apart is a
        polynomial that is not zero and whose transform has a single zero in it,
        which is what a non-unit actually looks like: built by inverting a
        transform that has one.
        """
        transform = np.ones(degree, dtype=np.int32)
        transform[degree // 3] = 0
        singular = np.asarray(
            arith.centered(arith.intt(arith.to_field(transform)))
        ).astype(np.int64)

        self.assertTrue(np.any(singular != 0), "the polynomial itself is not zero")
        self.assertFalse(bool(np.asarray(keygen.invertible(singular))))
        self.assertFalse(falcon_reference.is_invertible(singular.tolist()))
        self.assertFalse(
            bool(np.asarray(keygen.invertible(np.zeros(degree, np.int64))))
        )


if __name__ == "__main__":
    absltest.main()
