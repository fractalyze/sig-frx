# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The shared rejection budget, and the tail that sizes it.

The budget is the whole safety argument of the reshaping every lattice sampler
gets: the standard's loop cannot run short, a fixed budget can, so it is sized by
an exact binomial tail rather than by convention. These pin the sizing from both
sides — it is safe, and it is the smallest safe one — so a change to either the
margin or an acceptance probability fails here rather than silently widening the
window in which a wrong polynomial comes back.

It lives beside [`rejection.py`](../rejection.py) rather than in either
consumer's tests. The module was extracted because both lattice schemes ask the
same two questions of different constants, and leaving its only tests inside
ML-DSA's file would mean deleting or re-parameterizing that scheme deletes the
shared module's entire test surface — including the two reaches into the private
`_shortfall_exceeds_margin`, whose only caller is no longer in that package.

The acceptance rates below are named by the sampler they come from, but nothing
here imports a scheme: they are the four shapes the tail has to be right about,
and a scheme's own budget wiring is tested where that scheme is.
"""

from __future__ import annotations

from absl.testing import absltest, parameterized

from sig_frx.lattice import rejection

# The four parameterisations the two lattice schemes reach for, as
# `(needed, (accept_numerator, accept_denominator), candidates_per_block)`.
# FIPS 204 Algorithm 14's 23-bit draw kept below `q`; Algorithm 15's nibble at
# both `η`; Algorithm 29's `τ` sequential steps; and Falcon Algorithm 3's 16-bit
# draw kept below `⌊2^16/q⌋·q`.
_ML_DSA_Q = 8380417
_FALCON_Q = 12289
_CASES = (
    ("expand_a", 256, (_ML_DSA_Q, 1 << 23), 56),
    ("expand_s_eta_2", 256, (15, 16), 272),
    ("expand_s_eta_4", 256, (9, 16), 272),
    ("sample_in_ball_tau_60", 60, (256 - 60 + 1, 256), 1),
    ("hash_to_point_512", 512, ((1 << 16) // _FALCON_Q * _FALCON_Q, 1 << 16), 68),
    ("hash_to_point_1024", 1024, ((1 << 16) // _FALCON_Q * _FALCON_Q, 1 << 16), 68),
)


class BudgetTest(parameterized.TestCase):
    @parameterized.named_parameters(*_CASES)
    def test_is_the_smallest_budget_that_meets_the_margin(
        self, needed: int, accept: tuple[int, int], per_block: int
    ) -> None:
        blocks = rejection.budget(needed, accept, per_block)
        self.assertFalse(
            rejection._shortfall_exceeds_margin(blocks * per_block, needed, accept),
            "the chosen budget does not meet the margin",
        )
        self.assertTrue(
            rejection._shortfall_exceeds_margin(
                (blocks - 1) * per_block, needed, accept
            ),
            "a smaller budget would also have met it, so this one is not minimal",
        )

    def test_the_margin_is_the_strongest_parameter_set_s_strength(self) -> None:
        """`2^-256` is `λ` at ML-DSA-87 (FIPS 204 Table 1), not a round number.

        Written as the number rather than read off ML-DSA's table: the constant
        is shared now, and sourcing it from one consumer's parameters is what
        would make re-parameterizing that consumer silently move the margin for
        the other.
        """
        self.assertEqual(rejection.LOG2_SHORTFALL, 256)

    def test_a_certain_acceptance_needs_no_slack(self) -> None:
        """The tail is exact, so `p = 1` sizes to exactly what is asked for."""
        self.assertEqual(rejection.budget(256, (1, 1), 8), 32)


if __name__ == "__main__":
    absltest.main()
