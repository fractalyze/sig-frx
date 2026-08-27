# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The descent is checked coefficient by coefficient against Python integers.

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

import numpy as np
from absl.testing import absltest, parameterized

from sig_frx.lattice.falcon import arith, bigint, keygen
from sig_frx.lattice.falcon.testing import falcon_reference

# The widths #26 measured a random key's descent to reach, at `n = 1024`. The
# entry is `f` itself; the rest are one level of `N` each. Sampling moves these
# by a bit or two, so they are a corridor rather than an equality.
MEASURED_WIDTHS = (4, 11, 24, 51, 102, 203, 406, 807, 1593, 3151, 6302)


def _draw(degree: int, seed: int) -> list[int]:
    """§3.8.2's `f`: `4096/n` draws summed, so the variance is degree-independent.

    A rounded continuous Gaussian rather than the reference's table sampler.
    They differ in tail shape and not in variance, and it is variance that sets
    the coefficient magnitude the descent's widths follow from.
    """
    rng = np.random.default_rng(seed)
    draws = rng.normal(0.0, 1.43300980528773, size=(degree, 4096 // degree))
    return [int(v) for v in np.round(draws).astype(np.int64).sum(axis=1)]


def _pack(value: int, bits: int) -> np.ndarray:
    """One wide signed integer as limbs, in two's complement.

    Not [`keygen.to_limbs`](../keygen.py): that one is the descent's entry point
    and goes through `int64`, because what it packs is the four-bit Gaussian
    draw. The base case's operands are thousands of bits, which is the whole
    reason they are limbs by the time they reach it.
    """
    limbs = bigint.limb_count(bits + 2)
    return bigint.to_limbs(value % (1 << (limbs * bigint.LIMB_BITS)), limbs)


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


class BaseCaseTest(parameterized.TestCase):
    """Algorithm 6 at degree 1 — the Bezout pair, against Python `int`.

    The descent ends at one coefficient apiece and the recursion turns around on
    `u·f - v·g = gcd(f, g)`. That identity is the whole gate: a `u` and `v` that
    satisfy it are correct whatever route produced them, and no published vector
    reaches inside key generation to say otherwise.

    **The widths are the ones the base case actually sees.** #26 measured 3,161
    bits at `n = 512` and 6,302 at `n = 1024`, and a loop that is right at 120
    bits says nothing about a loop whose trip count and register width are both
    two orders larger. The small width is here to be checkable by hand when the
    wide ones fail, not to stand in for them.
    """

    # The base case's measured widths, plus a small one for a legible failure.
    WIDTHS = (120, 3161, 6302)

    @staticmethod
    def _coprime_pair(bits: int, seed: int) -> tuple[int, int]:
        """Two coprime integers of `bits` bits, one of them free to be even.

        `gcd = 1` is what Algorithm 6 requires and what a key that fails it is
        redrawn for, so a coprime pair is the case the solver runs on. Both even
        is the one combination the identity cannot hold for, and it is excluded
        by coprimality rather than by construction.
        """
        rng = random.Random(seed)
        while True:
            f = rng.getrandbits(bits) | (1 << (bits - 1))
            g = rng.getrandbits(bits) | (1 << (bits - 1))
            if math.gcd(f, g) == 1:
                return f, g

    @parameterized.parameters(*WIDTHS)
    def test_the_bezout_identity_holds(self, bits: int) -> None:
        f, g = self._coprime_pair(bits, bits)
        u, v, gcd = keygen.base_case(_pack(f, bits), _pack(g, bits), bits)

        self.assertEqual(bigint.from_limbs(np.asarray(gcd), signed=True), 1)
        self.assertEqual(
            bigint.from_limbs(np.asarray(u), signed=True) * f
            - bigint.from_limbs(np.asarray(v), signed=True) * g,
            1,
        )

    @parameterized.parameters(*WIDTHS)
    def test_a_shared_factor_is_reported_rather_than_asserted(self, bits: int) -> None:
        # Algorithm 6 restarts the draw when the base case is not coprime, so the
        # solver has to be able to ask. Returning the gcd is what lets it: a
        # boolean would say a key was rejected without saying by how much.
        f, g = self._coprime_pair(bits - 1, bits)
        wide = bits + 2
        _, _, gcd = keygen.base_case(_pack(3 * f, wide), _pack(3 * g, wide), wide)
        self.assertEqual(bigint.from_limbs(np.asarray(gcd), signed=True), 3)

    def test_a_both_even_pair_is_declined_rather_than_answered(self) -> None:
        # The halving step needs one operand odd. A pair that is not has gcd at
        # least 2 and is a key Algorithm 6 redraws — but the loop's answer for it
        # is not merely coarse, it is wrong, so the zero is what keeps a caller
        # testing `gcd != 1` from reading a number that no longer means anything.
        f, g = self._coprime_pair(120, 21)
        _, _, gcd = keygen.base_case(_pack(2 * f, 122), _pack(2 * g, 122), 122)
        self.assertEqual(bigint.from_limbs(np.asarray(gcd), signed=True), 0)

    def test_a_negative_operand_keeps_the_identity(self) -> None:
        # The descent's coefficients are signed and the base case inherits that.
        # The identity is over the signed values, so a sign that is dropped on
        # the way in produces a `u` and `v` that satisfy it for the wrong pair.
        f, g = self._coprime_pair(120, 7)
        for signs in ((1, -1), (-1, 1), (-1, -1)):
            signed_f, signed_g = signs[0] * f, signs[1] * g
            with self.subTest(signs=signs):
                u, v, _ = keygen.base_case(
                    _pack(signed_f, 120), _pack(signed_g, 120), 120
                )
                self.assertEqual(
                    bigint.from_limbs(np.asarray(u), signed=True) * signed_f
                    - bigint.from_limbs(np.asarray(v), signed=True) * signed_g,
                    1,
                )


if __name__ == "__main__":
    absltest.main()
