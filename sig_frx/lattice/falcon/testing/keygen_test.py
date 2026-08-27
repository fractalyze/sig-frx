# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Key generation against Python integers, and against the distribution itself.

Two halves with two different oracles. Algorithm 6's recursion and its base case
are exact integer arithmetic, so the oracle is exact integer arithmetic.
Algorithm 5's draw is a *distribution*, which no single output can be right or
wrong about — so it is checked against `exp(-x²/2σ²)` normalised, over enough
samples that the sampling noise is smaller than the thing being asserted.

Every level of the recursion is compared against
[`falcon_reference`](falcon_reference.py), which is each definition looped one
product at a time over unbounded integers. That is the only oracle available
here: the coefficients pass 9,000 bits, so no array type can hold the
intermediate and no published vector exists for a step inside key generation.

## The last level is the one that matters, and the cheap tests never reach it

The descent's whole difficulty is that widths double on the way down, so a test
that stops at a small degree exercises the easy end and nothing else. The cases
below run the **full** recursion at both parameter sets — nine levels to 3,141
bits at `n = 512` and ten to 6,327 at `n = 1024` — and check every level, not
only the last.

That the actual widths land where
[#26](https://github.com/fractalyze/sig-frx/issues/26) measured them is asserted
too. It is what says the recursion is the *right* one rather than merely a
self-consistent one: a wrong split or a wrong wrap sign still produces a
polynomial, still descends, and still ends up at degree 1.

## Coming back up, the equation is the gate and the width is the other half

`f·G - g·F = q` is preserved by every step, so it catches an error anywhere in
the recursion — and it catches it as an exact integer comparison, because none
of the arithmetic that produced it was approximate. Babai's rounding moves the
*width* of the answer and cannot move the equation it satisfies.

Which is exactly why the width is asserted beside it. `q` is satisfied by pairs
of every size, the un-reduced lift included, so a reduction that quietly did
nothing would pass the equation and has to be caught somewhere else.

## The tree is a third kind of thing, and it cannot be gated exactly

Algorithms 8 and 9 hold no integers at all — the leaves are irrational before
any implementation touches them — so the exact oracle above does not reach them
and a tolerance is unavoidable. Agreement with the node-by-node recursion is
therefore only half of it; the other half is three identities the leaves satisfy
that a wrong tree would not, and that are checked against quantities computed
some other way:

- they multiply to `q^n` whenever the basis has determinant `q`, since they are
  its squared Gram-Schmidt norms and one rational leaf stands for the two
  dimensions its ring contributes;
- the largest is `‖B‖²_GS`, which is Algorithm 5 line 5's own quantity arrived
  at from the opposite direction — a closed form there, a decomposition here;
- the smallest is `q²` over the largest.

Those hold for any basis of determinant `q`, so they are checked on one built by
hand rather than solved for. `f = 1` makes `f·G − g·F = q` solvable by
substitution, which buys several degrees for the cost of none and isolates the
tree from the whole integer half of this file.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

import numpy as np
from absl.testing import absltest, parameterized
from frx import numpy as fnp

from sig_frx.lattice.falcon import arith, bigint, encoding, fft, keygen
from sig_frx.lattice.falcon.testing import falcon_reference, falcon_vectors

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


def _bits(*polynomials: Sequence[int]) -> int:
    """The widest coefficient's bit length across every polynomial given.

    Variadic because a width here is almost always a property of a *pair* —
    `f` and `g` descend together and are budgeted together, and the level above
    sizes itself from whichever of the two is wider.
    """
    return max(abs(v) for polynomial in polynomials for v in polynomial).bit_length()


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
                widest = _bits(result)
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
        bits = _bits(source)
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
        bits = _bits(source)
        levels = degree.bit_length() - 1
        chain = keygen.descend(keygen.to_limbs(source, bits), bits, levels)

        self.assertLen(chain, levels)
        expected = source
        for level, (limbs, bound) in enumerate(chain, start=1):
            expected = falcon_reference.field_norm(expected)
            widest = _bits(expected)
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


def _pack_poly(values: Sequence[int], bits: int) -> np.ndarray:
    """A whole polynomial as `[degree, limbs]`, at any coefficient width.

    [`keygen.to_limbs`](../keygen.py) is the descent's entry point and takes
    what the Gaussian draw produces, which fits an `int64`. The lift's output
    does not, so a test that starts from one packs it here instead.
    """
    return np.stack([_pack(value, bits) for value in values])


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


class LiftBoundTest(absltest.TestCase):
    """`lift_bits` has to be an upper bound, and be one for the right reason."""

    def test_the_bound_is_attained_by_the_input_that_saturates_it(self) -> None:
        """The lift sums `degree/2` products, and this is the input where none cancel.

        `lower(x²)` is empty at every odd position, so only half the operand's
        coefficients ever meet a non-zero one — which is the whole content of
        the bound, and the thing an input has to reach for it to be tight.

        The constant coefficient is where they can all be made to agree in sign:
        the term from `i = 0` does not wrap and every other one does, so an
        operand that is positive at `0` and negative at the remaining even
        positions puts every product on the same side. One bit of slack remains
        because a coefficient is `2^w - 1` rather than `2^w`.
        """
        for degree, low_bits, high_bits in ((4, 5, 6), (8, 7, 4), (16, 3, 9)):
            with self.subTest(degree=degree):
                low = (1 << low_bits) - 1
                high = (1 << high_bits) - 1
                lower = [low] * (degree // 2)
                other = [
                    0 if i % 2 else (high if i == 0 else -high) for i in range(degree)
                ]
                result = falcon_reference.lift(lower, other)
                widest = _bits(result)
                bound = keygen.lift_bits(low_bits, high_bits, degree)
                self.assertLessEqual(widest, bound)
                self.assertLessEqual(bound - widest, 1)

    def test_the_bound_refuses_a_degree_the_lift_has_no_step_for(self) -> None:
        # 1 is in the sweep and is the one that is not obvious: a lift lands at
        # *twice* its operand's degree, so degree 1 is not a result it can
        # produce — where the descent, which halves, bottoms out there.
        for degree in (0, 1, 3, 12):
            with self.subTest(degree=degree):
                with self.assertRaisesRegex(ValueError, "power of two"):
                    keygen.lift_bits(4, 4, degree)


class LiftTest(parameterized.TestCase):
    """One step back up, against the definition."""

    @parameterized.parameters(2, 4, 8, 16, 32)
    def test_agrees_with_the_reference_at_small_degrees(self, degree: int) -> None:
        lower = _draw(degree // 2, degree)
        other = _draw(degree, degree + 1)
        lower_bits = _bits(lower)
        other_bits = _bits(other)

        result, bits = keygen.lift(
            keygen.to_limbs(lower, lower_bits),
            keygen.to_limbs(other, other_bits),
            lower_bits,
            other_bits,
        )

        expected = falcon_reference.lift(lower, other)
        self.assertEqual(_unpack(result), expected)
        self.assertLessEqual(_bits(expected), bits)

    def test_the_substitution_and_the_wrap_sign_are_both_the_right_ones(self) -> None:
        """Two ways to write this step wrong, and one case that separates each.

        `lower(x²)` at `x` instead of `x²` and `other(-x)` at `+x` are both
        substitutions that still produce a polynomial of the right degree, so
        the equation downstream is the only thing that would ever notice. These
        do it here instead.

        `F' = x` lifts to `x²`, and against `other = x` — whose `other(-x)` is
        `-x` — the product is `-x³`. Placing `F'` at `x` rather than `x²` would
        put it at `-x²`, and dropping the sign flip would put it at `+x³`.
        """
        cases = (
            ([0, 1], [0, 1, 0, 0], [0, 0, 0, -1]),
            ([1, 0], [0, 1, 0, 0], [0, -1, 0, 0]),
            ([0, 1], [0, 0, 1, 0], [-1, 0, 0, 0]),
        )
        for lower, other, expected in cases:
            with self.subTest(lower=lower, other=other):
                result, _ = keygen.lift(
                    keygen.to_limbs(lower, 1), keygen.to_limbs(other, 1), 1, 1
                )
                self.assertEqual(_unpack(result), expected)
                self.assertEqual(falcon_reference.lift(lower, other), expected)

    def test_it_refuses_a_pair_that_does_not_halve(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot lift"):
            keygen.lift(keygen.to_limbs([1, 2], 2), keygen.to_limbs([1, 2], 2), 2, 2)


class ReduceTest(absltest.TestCase):
    """Algorithm 7 moves `(F, G)` by a multiple of `(f, g)` and nothing else."""

    def test_it_removes_a_planted_multiple_and_leaves_the_pairing_alone(self) -> None:
        """Both halves of the step, against a case whose answer is known in advance.

        A multiple of `(f, g)` is exactly what the reduction is licensed to
        take away — and the only thing it can. That is worth saying explicitly,
        because it rules out the test one would write first: an arbitrary wide
        `(F, G)` does *not* shorten, since `k` ranges over the ring rather than
        over the whole space and only the component along `(f, g)` is reachable.
        The lift's output is close to that line by construction, which is why
        the reduction works there; here the line is planted so the width has
        something to be checked against.

        The invariance is the other half and holds unconditionally:
        `f(G - kg) - g(F - kf)` is `fG - gF` for every `k` there is, so the
        pairing has to come back bit for bit whether or not `k` was any good.
        """
        degree = 16
        f, g = _draw(degree, 11), _draw(degree, 12)
        short_f, short_g = _draw(degree, 13), _draw(degree, 14)
        multiple = [v << 250 for v in _draw(degree, 15)]
        big_f = [
            a + b
            for a, b in zip(short_f, falcon_reference.exact_negacyclic_mul(multiple, f))
        ]
        big_g = [
            a + b
            for a, b in zip(short_g, falcon_reference.exact_negacyclic_mul(multiple, g))
        ]
        before = falcon_reference.ntru_equation(f, g, big_f, big_g)

        small_bits = _bits(f, g)
        big_bits = _bits(big_f, big_g)
        reduced_f, reduced_g, width = keygen.reduce(
            _pack_poly(big_f, big_bits),
            _pack_poly(big_g, big_bits),
            keygen.to_limbs(f, small_bits),
            keygen.to_limbs(g, small_bits),
        )

        values_f, values_g = _unpack(reduced_f), _unpack(reduced_g)
        self.assertEqual(
            falcon_reference.ntru_equation(f, g, values_f, values_g), before
        )
        widest = _bits(values_f, values_g)
        self.assertEqual(widest, width)
        # The planted multiple is gone in full: what is left is the short
        # representative, give or take one more copy of `(f, g)` where `k`
        # rounded the other way.
        short_bits = _bits(short_f, short_g)
        self.assertLessEqual(
            widest, max(short_bits, small_bits) + degree.bit_length() + 1
        )
        self.assertLess(widest, big_bits - 200)

    def test_a_pair_already_at_the_basis_width_is_left_alone(self) -> None:
        # There is nothing above `(f, g)`'s scale to remove, so the loop has to
        # decline to run rather than subtract a rounding artefact.
        degree = 8
        f, g = _draw(degree, 21), _draw(degree, 22)
        bits = _bits(f, g)
        big_f, big_g = _draw(degree, 23), _draw(degree, 24)
        reduced_f, reduced_g, _ = keygen.reduce(
            keygen.to_limbs(big_f, bits),
            keygen.to_limbs(big_g, bits),
            keygen.to_limbs(f, bits),
            keygen.to_limbs(g, bits),
        )
        self.assertEqual(_unpack(reduced_f), big_f)
        self.assertEqual(_unpack(reduced_g), big_g)


class NtruSolveTest(parameterized.TestCase):
    """Algorithm 6 end to end — the first point where the equation can be checked."""

    def _coprime_chain(self, degree: int) -> list[tuple[list[int], list[int]]]:
        """A draw whose descent bottoms out at a coprime pair, and that descent.

        Roughly half of them do not, which is not a defect: Algorithm 5 answers
        a non-coprime bottom by drawing again, and `ok` is how the solver says
        so. Searching here keeps that retry out of the assertions below.

        The chain comes back rather than the seed because finding the seed
        *is* the chain — an `O(n²)` unbounded-integer descent per candidate —
        and a caller handed the seed alone would descend a second time to get
        what this already computed.
        """
        for seed in range(degree, degree + 40):
            chain = [(_draw(degree, seed), _draw(degree, seed + 1))]
            for _ in range(degree.bit_length() - 1):
                current_f, current_g = chain[-1]
                chain.append(
                    (
                        falcon_reference.field_norm(current_f),
                        falcon_reference.field_norm(current_g),
                    )
                )
            bottom_f, bottom_g = chain[-1]
            if math.gcd(bottom_f[0], bottom_g[0]) == 1 and (
                bottom_f[0] % 2 or bottom_g[0] % 2
            ):
                return chain
        raise AssertionError(f"no coprime bottom for degree {degree}")

    @parameterized.parameters(*arith.DEGREES)
    def test_the_equation_closes_exactly_at_full_degree(self, degree: int) -> None:
        """`f·G - g·F = q`, in integers, at `n = 512` and `n = 1024`.

        The gate this whole issue exists for. Every step of the recursion
        preserves the equation, so an error anywhere in the descent, the base
        case or the walk back up lands here and nowhere else — and it lands as
        an exact integer comparison rather than as a tolerance, because none of
        the arithmetic that produced it was approximate. The rounding in
        Babai's `k` moves the *width* of the answer and cannot move the
        equation it satisfies.

        The width corridor is the second half. `q` is satisfied by pairs of
        every size, including the un-reduced lift that is thousands of bits
        wide — so a reduction that quietly did nothing would pass the equation
        and fail here.
        """
        f, g = self._coprime_chain(degree)[0]
        bits = _bits(f, g)

        big_f, big_g, width, ok = keygen.ntru_solve(
            keygen.to_limbs(f, bits), keygen.to_limbs(g, bits), bits, arith.Q
        )

        self.assertTrue(bool(np.asarray(ok)))
        values_f, values_g = _unpack(big_f), _unpack(big_g)
        equation = falcon_reference.ntru_equation(f, g, values_f, values_g)
        self.assertEqual(equation[0], arith.Q)
        self.assertEqual(equation[1:], [0] * (degree - 1))
        self.assertLessEqual(width, MEASURED_WIDTHS[0] + 8)

    def test_every_level_lands_where_the_issue_measured(self) -> None:
        """The lift roughly triples the width and the reduction takes it back.

        Run at `n = 512` alone: the widths follow how many squarings have
        happened rather than what degree they happened at, so both parameter
        sets walk the same rows and the second one would only re-measure them a
        level further down.

        Three assertions per level, off one solve. The bound is what says the
        lift's own budget cannot be short; the corridor is what says the
        reduction actually ran, since a lift that was never reduced still
        satisfies the equation; and the equation is what says the pair being
        measured is the right one.
        """
        degree = 512
        levels = degree.bit_length() - 1
        chain = self._coprime_chain(degree)
        bottom_f, bottom_g = chain[-1]
        bottom_bits = _bits(bottom_f, bottom_g)
        solved_f, solved_g, ok = keygen.base_case(
            _pack(bottom_f[0], bottom_bits),
            _pack(bottom_g[0], bottom_bits),
            bottom_bits,
            arith.Q,
        )
        self.assertTrue(bool(np.asarray(ok)))
        big_f, big_g = np.asarray(solved_f)[None, :], np.asarray(solved_g)[None, :]
        big_bits = max(
            abs(bigint.from_limbs(np.asarray(solved_f), signed=True)),
            abs(bigint.from_limbs(np.asarray(solved_g), signed=True)),
        ).bit_length()

        for depth in range(levels - 1, -1, -1):
            level_f, level_g = chain[depth]
            level_bits = _bits(level_f, level_g)
            limbs_f = _pack_poly(level_f, level_bits)
            limbs_g = _pack_poly(level_g, level_bits)

            lifted_f, lifted_bits = keygen.lift(big_f, limbs_g, big_bits, level_bits)
            lifted_g, _ = keygen.lift(big_g, limbs_f, big_bits, level_bits)
            widest_lift = _bits(_unpack(lifted_f), _unpack(lifted_g))
            big_f, big_g, big_bits = keygen.reduce(lifted_f, lifted_g, limbs_f, limbs_g)
            values_f, values_g = _unpack(big_f), _unpack(big_g)

            with self.subTest(depth=depth):
                self.assertLessEqual(widest_lift, lifted_bits)
                equation = falcon_reference.ntru_equation(
                    level_f, level_g, values_f, values_g
                )
                self.assertEqual(equation[0], arith.Q)
                self.assertEqual(equation[1:], [0] * (len(level_f) - 1))
                self.assertAlmostEqual(
                    big_bits,
                    MEASURED_WIDTHS[depth],
                    delta=max(8, MEASURED_WIDTHS[depth] // 20),
                )


_Basis = tuple[list[int], list[int], list[int], list[int]]


def _basis(degree: int, seed: int) -> _Basis:
    """`(f, g, F, G)` — four small polynomials, a full-rank matrix and nothing more.

    Algorithms 8 and 9 ask only that their input be a full-rank self-adjoint
    Gram matrix, which `B × B*` is for any `B` with a non-zero first row. The
    cases that need a *trapdoor* say so and build one.
    """
    rng = np.random.default_rng(seed)
    f, g, big_f, big_g = (rng.integers(-6, 7, size=degree).tolist() for _ in range(4))
    return f, g, big_f, big_g


def _determinant_q_basis(degree: int, seed: int) -> _Basis:
    """A basis with `f·G − g·F = q`, built by substitution rather than solved.

    `f = 1` reduces the NTRU equation to `G = q + g·F`, so any `g` and `F` give
    a basis of determinant `q` for the cost of one negacyclic product. Its
    geometry is nothing like a real key's — Algorithm 5 would reject it out of
    hand — and that is the point: the identities it is used for follow from the
    determinant alone, so a basis that isolates them from `ntru_solve` is a
    better witness than one that ties the two together.
    """
    rng = np.random.default_rng(seed)
    f = [1] + [0] * (degree - 1)
    g = rng.integers(-6, 7, size=degree).tolist()
    big_f = rng.integers(-20, 21, size=degree).tolist()
    big_g = falcon_reference.exact_negacyclic_mul(g, big_f)
    big_g[0] += arith.Q
    return f, g, big_f, big_g


def _transformed(basis: _Basis) -> tuple[Any, ...]:
    """A coefficient-domain basis in the transform domain, host side.

    No [`fft.double_precision`](../fft.py) anywhere, and that is the tree's
    whole precision story rather than an omission: numpy is `complex128`
    natively, so a host caller needs no scope. The traced case below opens one.
    """
    return tuple(fft.fft(polynomial) for polynomial in basis)


def _tree(basis: _Basis) -> keygen.FalconTree:
    """Algorithm 4 lines 3-5 over a coefficient-domain `(f, g, F, G)`."""
    return keygen.ffldl(*keygen.gram(*_transformed(basis)))


def _reference_tree(basis: _Basis) -> tuple[Any, Any, Any]:
    """The same, recursed one node at a time."""
    return falcon_reference.ffldl(falcon_reference.gram(*basis))


def _flatten(tree: tuple[Any, Any, Any]) -> tuple[list[np.ndarray], np.ndarray]:
    """The recursion's tree as one array per depth, plus its leaves whole.

    Depth-first, left before right — which at a fixed depth is left-to-right, so
    appending in visit order *is* the claim that node `i`'s children are `2i`
    and `2i + 1`. Nothing here reorders, so a level that agrees with
    [`keygen.FalconTree`](../keygen.py)'s has agreed about the layout too.

    The leaves come back as the length-2 evaluations the specification's tree
    holds, not as the single number the module reduces them to, so that the
    reduction is something a test can check rather than share.
    """
    levels: dict[int, list[np.ndarray]] = {}
    leaves: list[np.ndarray] = []

    def walk(node: tuple[Any, Any, Any], depth: int) -> None:
        value, left, right = node
        levels.setdefault(depth, []).append(value)
        if isinstance(left, tuple):
            walk(left, depth + 1)
            walk(right, depth + 1)
        else:
            leaves.extend((left, right))

    walk(tree, 0)
    return [np.stack(levels[depth]) for depth in sorted(levels)], np.stack(leaves)


class GramTest(parameterized.TestCase):
    """Algorithm 4 lines 2-4, against the matrix product written out."""

    @parameterized.parameters(8, 32, 128)
    def test_agrees_with_the_product_of_the_basis_and_its_adjoint(
        self, degree: int
    ) -> None:
        basis = _basis(degree, degree)
        got = keygen.gram(*_transformed(basis))
        want = falcon_reference.gram(*basis)
        for entry, (row, column) in zip(got, ((0, 0), (0, 1), (1, 1))):
            with self.subTest(entry=f"G{row}{column}"):
                np.testing.assert_allclose(entry, want[row][column], rtol=1e-9)

    @parameterized.parameters(8, 32)
    def test_the_entry_it_drops_is_the_conjugate_of_one_it_keeps(
        self, degree: int
    ) -> None:
        """`G10` is dropped on a claim, and this is the claim.

        Both halves are needed. That the product is self-adjoint is a property
        of `B × B*` and is checked on the transcription, which computes all
        four entries; that the module's `G01` is the one to conjugate is a
        property of the module, and is checked against it.
        """
        basis = _basis(degree, degree + 1)
        want = falcon_reference.gram(*basis)
        np.testing.assert_allclose(want[1][0], np.conj(want[0][1]), rtol=1e-9)
        _, g01, _ = keygen.gram(*_transformed(basis))
        np.testing.assert_allclose(np.conj(g01), want[1][0], rtol=1e-9)

    def test_the_diagonal_comes_back_real(self) -> None:
        """A sum of squared magnitudes, typed as one.

        Not cosmetic: `ldl` subtracts from `G11` and `ffldl` splits `D00`, so a
        complex diagonal with a zero imaginary part would carry a rounding
        artefact all the way to the leaves, where it would have to be removed
        anyway and by then would look like something the algorithm produced.
        """
        g00, g01, g11 = keygen.gram(*_transformed(_basis(16, 3)))
        self.assertFalse(np.iscomplexobj(g00))
        self.assertFalse(np.iscomplexobj(g11))
        self.assertTrue(np.iscomplexobj(g01), "the off-diagonal is not real")


class LdlTest(absltest.TestCase):
    """Algorithm 8, against what it decomposed and against the way it is written."""

    def test_it_reconstructs_the_matrix_it_decomposed(self) -> None:
        """`G = L·D·L*`, checked at all four entries including the dropped one.

        The one claim Algorithm 8 makes that does not depend on how it is
        written, so it catches a sign or a conjugate the transcription below
        might share.
        """
        g00, g01, g11 = keygen.gram(*_transformed(_basis(32, 8)))
        l10, d00, d11 = keygen.ldl(g00, g01, g11)

        # `L·D·L*` for `L = [[1, 0], [L10, 1]]` and `D = diag(D00, D11)`.
        np.testing.assert_allclose(d00, g00, rtol=1e-9)
        np.testing.assert_allclose(d00 * np.conj(l10), g01, rtol=1e-9)
        np.testing.assert_allclose(l10 * d00, np.conj(g01), rtol=1e-9)
        np.testing.assert_allclose(
            (l10 * np.conj(l10)).real * d00 + d11, g11, rtol=1e-9
        )

    def test_agrees_with_the_standards_own_expression(self) -> None:
        """The module folds `L10 ⊙ L10*` to a real; the transcription does not."""
        basis = _basis(32, 9)
        g00, g01, g11 = keygen.gram(*_transformed(basis))
        got = keygen.ldl(g00, g01, g11)
        want = falcon_reference.ldl(falcon_reference.gram(*basis))
        for entry, name in zip(zip(got, want), ("L10", "D00", "D11")):
            with self.subTest(entry=name):
                np.testing.assert_allclose(entry[0], entry[1], rtol=1e-9)

    def test_the_decomposed_diagonal_stays_real(self) -> None:
        g00, g01, g11 = keygen.gram(*_transformed(_basis(16, 10)))
        _, d00, d11 = keygen.ldl(g00, g01, g11)
        self.assertFalse(np.iscomplexobj(d00))
        self.assertFalse(np.iscomplexobj(d11))


class FfldlTest(parameterized.TestCase):
    """Algorithm 9's recursion, held to the level-major form that replaces it."""

    @parameterized.parameters(8, 32, 128)
    def test_every_level_agrees_with_the_node_by_node_recursion(
        self, degree: int
    ) -> None:
        """One array per depth against `2^d` calls, at every depth and not the last.

        The shape is asserted beside the values because it is the reshaping's
        whole content: a level that came out `[1, n]` at every depth would still
        hold the right numbers somewhere.
        """
        basis = _basis(degree, degree + 2)
        got = _tree(basis)
        want_levels, want_leaves = _flatten(_reference_tree(basis))

        self.assertLen(got.values, degree.bit_length() - 1)
        self.assertLen(want_levels, len(got.values))
        for depth, (level, want) in enumerate(zip(got.values, want_levels)):
            with self.subTest(depth=depth):
                self.assertEqual(np.shape(level), (2**depth, degree >> depth))
                np.testing.assert_allclose(level, want, rtol=1e-9)
        np.testing.assert_allclose(
            got.leaves, want_leaves.real.mean(axis=-1), rtol=1e-9
        )

    def test_a_leaf_is_one_rational_and_not_two(self) -> None:
        """`d* = d` in `Q[x]/(x² + 1)` forces the `x` term to zero.

        So a leaf is a constant, its two evaluations are the same real number,
        and the module keeps one where the specification's tree holds the pair.
        That is a theorem rather than a rounding convenience, and this is where
        it is checked instead of assumed — a split or an adjoint that was wrong
        would leave the pair disagreeing while every level above still matched
        its own transcription.
        """
        basis = _basis(64, 11)
        _, pairs = _flatten(_reference_tree(basis))
        scale = np.max(np.abs(pairs.real))
        self.assertLess(np.max(np.abs(pairs.imag)) / scale, 1e-12)
        self.assertLess(np.max(np.abs(pairs[:, 0] - pairs[:, 1])) / scale, 1e-12)
        # The transcription carries the imaginary part it is entitled to and
        # the module does not, so the module's side of this is a dtype: a leaf
        # that arrived complex would mean the diagonal stopped being real
        # somewhere above, which the mean would then average away rather than
        # report.
        self.assertFalse(np.iscomplexobj(np.asarray(_tree(basis).leaves)))

    def test_the_traced_tree_is_the_host_one(self) -> None:
        """The same source line on both namespaces, which is what `namespace` buys.

        Traced, `complex64` would be 24 bits of mantissa against the 53 Falcon's
        analysis assumes, so the precision is checked three ways rather than
        trusted: [`fft`](../fft.py) refuses outside the scope at all, the dtype
        is asserted per level, and the values are held against the host's answer
        instead of against a tolerance of the device's own.
        """
        basis = _basis(16, 12)
        host = _tree(basis)
        with fft.double_precision():
            device = keygen.ffldl(
                *keygen.gram(
                    *(fft.fft(fnp.asarray(p, dtype=np.float64)) for p in basis)
                )
            )
            values = [np.asarray(level) for level in device.values]
            leaves = np.asarray(device.leaves)

        self.assertLen(values, len(host.values))
        for depth, (level, want) in enumerate(zip(values, host.values)):
            with self.subTest(depth=depth):
                self.assertEqual(level.dtype, np.dtype("complex128"))
                np.testing.assert_allclose(level, want, rtol=1e-9)
        np.testing.assert_allclose(leaves, host.leaves, rtol=1e-9)

    def test_it_refuses_what_it_has_no_tree_for(self) -> None:
        for degree in (1, 3, 12):
            with self.subTest(degree=degree):
                entry = np.ones(degree)
                with self.assertRaisesRegex(ValueError, "power of two"):
                    keygen.ffldl(entry, entry.astype(complex), entry)

    def test_it_refuses_a_batch_of_gram_matrices(self) -> None:
        """One key at a time. A leading axis would be silently read as the tree's."""
        entry = np.ones((2, 8))
        with self.assertRaisesRegex(ValueError, "one polynomial"):
            keygen.ffldl(entry, entry.astype(complex), entry)


class FalconTreeTest(parameterized.TestCase):
    """What the leaves are, which is the only thing the sampler reads them for."""

    @parameterized.parameters(8, 16, 64, 256)
    def test_the_leaves_are_the_bases_squared_gram_schmidt_norms(
        self, degree: int
    ) -> None:
        """Three identities off one tree, none of them computed the tree's way.

        `q^n` and not `q^(2n)`: the leaves are the `2n` squared Gram-Schmidt
        norms of `B` over `Q`, but the bottom of the recursion is ring degree 2
        and a self-adjoint element there is a constant — so one leaf stands for
        both dimensions and there are `n` of them.

        Summed in logs rather than multiplied, because the product itself is
        `q^n` and overflows a double at any interesting degree.
        """
        basis = _determinant_q_basis(degree, degree)
        f, g, big_f, big_g = basis
        equation = falcon_reference.ntru_equation(f, g, big_f, big_g)
        self.assertEqual(equation, [arith.Q] + [0] * (degree - 1), "not determinant q")

        leaves = np.asarray(_tree(basis).leaves)
        self.assertEqual(leaves.shape, (degree,))
        self.assertTrue(np.all(leaves > 0.0), "a squared norm is positive")

        self.assertAlmostEqual(
            float(np.sum(np.log2(leaves))) / (degree * math.log2(arith.Q)),
            1.0,
            delta=1e-12,
        )
        # Algorithm 5 line 5 reaches this by a closed form over the transform;
        # the tree reaches it by decomposing. They are the same number.
        norm = float(np.asarray(keygen.gram_schmidt_squared_norm(f, g)))
        self.assertAlmostEqual(float(leaves.max()) / norm, 1.0, delta=1e-9)
        self.assertAlmostEqual(
            float(leaves.max() * leaves.min()) / arith.Q**2, 1.0, delta=1e-9
        )

    def test_normalization_divides_the_standard_deviation_by_the_root(self) -> None:
        """Algorithm 4 line 7, and that it touches nothing else.

        `values` is what the sampler's line 10 multiplies by and normalization
        has no business in it — which is why one type carries both trees, and
        so is worth a case rather than a sentence.
        """
        sigma = falcon_reference.PARAMETER_SETS["Falcon-512"]["sigma"]
        tree = _tree(_determinant_q_basis(32, 13))
        normalized = keygen.normalize(tree, sigma)

        np.testing.assert_allclose(
            normalized.leaves, sigma / np.sqrt(np.asarray(tree.leaves)), rtol=1e-12
        )
        self.assertLen(normalized.values, len(tree.values))
        for depth, (after, before) in enumerate(zip(normalized.values, tree.values)):
            with self.subTest(depth=depth):
                np.testing.assert_array_equal(after, before)

    @parameterized.parameters(*falcon_reference.parameter_cases())
    def test_the_samplers_range_is_what_the_rejection_buys(
        self, name: str, **params: Any
    ) -> None:
        """Algorithm 11 line 2 asserts `σ' ∈ [σmin, σmax]`; here is why it holds.

        It is not a property of the tree — it is Algorithm 5 line 5's rejection
        seen through it. The largest leaf is `‖B‖²_GS`, which line 5 refuses
        above [`GRAM_SCHMIDT_BOUND`](../keygen.py), and the smallest is `q²`
        over the largest; `σ'` is `σ` over the root of a leaf, so both ends
        follow from that one rejection and Table 3.3. No key is needed, and none
        would make it more true — a key exhibits one pair of values inside the
        interval where this is the interval. Measured for one anyway, at
        `n = 512` on a trapdoor out of `ntru_solve` whose basis line 5 accepted:
        `σ'` ran `[1.2786, 1.7481]`.

        **The two ends are not the same kind of number, and the table hides it.**
        `σmin` is not an independent parameter at all — it *is* `σ/(1.17√q)`,
        the smallest `σ'` the rejection allows, published rounded to ten
        significant figures. So it is checked as an equality rather than as a
        bound: asserting `≥` fails at `n = 512` by 7e-11, which is the table's
        own last digit and not a fact about Falcon. `σmax` is a genuine bound
        with slack in it — the largest reachable `σ'` is 1.7492 and 1.7772
        against a published 1.8205 — so that end is asserted as one.
        """
        del name
        bound = keygen.GRAM_SCHMIDT_BOUND
        # The extreme leaves a basis Algorithm 5 accepts can reach.
        largest, smallest = bound, arith.Q**2 / bound
        self.assertAlmostEqual(
            params["sigma"] / math.sqrt(largest) / params["sigma_min"], 1.0, delta=1e-9
        )
        self.assertLess(params["sigma"] / math.sqrt(smallest), params["sigma_max"])


class PublishedKeyTest(parameterized.TestCase):
    """The reference implementation's own key pair, agreed with four ways.

    Everything else in this file is gated against a transcription of the
    specification, which cannot catch a misreading two implementations share.
    This is the case that can: the round-3 KAT's `sk` and `pk` are the reference
    implementation's output, and nothing here invented either of them.

    It is also as much of #26's third acceptance criterion as can be had without
    compiling that implementation — a key it generated is accepted here and its
    public key reproduced. What is left for the criterion is the direction that
    needs the C: a key generated *here* accepted by it.
    """

    @parameterized.parameters(*falcon_reference.parameter_cases())
    def test_the_published_private_key_implies_the_published_public_one(
        self, name: str, **params: Any
    ) -> None:
        """Four agreements off one decode, and they are together because it is one.

        Splitting them would decode the same 2,305 bytes four times to ask four
        questions of the same numbers — the reason `DescentTest` groups its
        assertions, applied to a key rather than to a recursion.
        """
        n = params["n"]
        sk = np.frombuffer(
            bytes.fromhex(falcon_vectors.SECRET_KEYS[name]), dtype=np.uint8
        )
        pk = np.frombuffer(
            bytes.fromhex(falcon_vectors.VECTORS[name][0].public_key), dtype=np.uint8
        )
        self.assertLen(sk, params["secret_key_size"])

        f, g, big_f, ok = encoding.sk_decode(sk, n)
        self.assertTrue(bool(np.asarray(ok)))
        f, g, big_f = (np.asarray(value) for value in (f, g, big_f))

        # §3.11.5, against the transcription: the widths and the sign convention
        # are shared between the two decoders and nothing else is.
        self.assertEqual(
            falcon_reference.sk_decode(sk.tobytes(), n),
            (f.tolist(), g.tolist(), big_f.tolist()),
        )

        # (3.35), and then the equation the recovery exists to satisfy. The
        # recovery runs modulo `q` and the answer is read back centered, which
        # is only sound because `G` is small — so its magnitude is asserted
        # rather than left to the equation, which a wrapped `G` would fail in a
        # way that says nothing about why.
        big_g = np.asarray(keygen.recover_g(f, g, big_f))
        self.assertLess(int(np.abs(big_g).max()), arith.Q // 2)
        self.assertEqual(
            falcon_reference.ntru_equation(
                f.tolist(), g.tolist(), big_f.tolist(), big_g.tolist()
            ),
            [arith.Q] + [0] * (n - 1),
        )

        # Algorithm 4 line 9, against bytes this repo did not produce.
        published, key_ok = encoding.pk_decode(pk, n)
        self.assertTrue(bool(np.asarray(key_ok)))
        np.testing.assert_array_equal(
            np.asarray(keygen.public_key(f, g)), np.asarray(published)
        )

        # And back out again, which is what says the encoders agree with the
        # decoders on more than their own output.
        np.testing.assert_array_equal(
            np.asarray(encoding.sk_encode(f, g, big_f, n)), sk
        )
        np.testing.assert_array_equal(np.asarray(encoding.pk_encode(published, n)), pk)

    @parameterized.parameters(*arith.DEGREES)
    def test_the_public_key_inverts_what_it_divides_by(self, degree: int) -> None:
        """`h = g/f` means `f·h = g mod q`, which is checkable without a key pair.

        A separate case from the published one because it fails differently: the
        published test says the division agrees with the reference, and this
        says it is a division at all. A `base_div` that multiplied instead would
        still produce a polynomial, and only this notices.
        """
        f, g = _draw(degree, degree + 40), _draw(degree, degree + 41)
        if not bool(np.asarray(keygen.invertible(f))):
            self.skipTest(f"the draw at degree {degree} is not a unit")
        h = keygen.public_key(f, g)
        product = arith.intt(
            arith.base_mul(arith.ntt(arith.to_field(h)), arith.ntt(arith.to_field(f)))
        )
        np.testing.assert_array_equal(
            np.asarray(arith.centered(product)),
            np.asarray(arith.centered(arith.to_field(g))),
        )


if __name__ == "__main__":
    absltest.main()
