# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The shared rejection budget and the tail that sizes it, and the compaction.

The budget is the whole safety argument of the reshaping every lattice sampler
gets: the standard's loop cannot run short, a fixed budget can, so it is sized by
an exact binomial tail rather than by convention. These pin the sizing from both
sides — it is safe, and it is the smallest safe one — so a change to either the
margin or an acceptance probability fails here rather than silently widening the
window in which a wrong polynomial comes back.

The compaction is pinned here for a different reason. Which *form* it takes is a
performance question with more than one right answer — a gather and a scatter
compute the same permutation, and
[`compaction_bench`](compaction_bench.py) is what ranks them — so the form is
expected to change. What must not change is the answer, and the samplers' own
tests reach it only through a SHAKE, where a compaction bug and a hashing bug
look alike. `FirstAcceptedTest` states the property directly and against a host
reference: the first `count` accepted entries of each row, in stream order, at
the shapes both schemes actually reach and at the survivor patterns a random
stream would need a great many draws to produce.

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

from collections.abc import Callable
from typing import Any, TypeAlias

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from sig_frx.lattice import rejection

_Lift: TypeAlias = Callable[[np.ndarray], Any]

# Every compaction case runs both ways, as the samplers' own tests do: a host
# array for the key generation and signing paths, a traced one for verification.
# The two are different code — `require_enough` can raise on one and provably
# cannot on the other — so a form that is right in one namespace is not thereby
# right in the other.
_LIFT: dict[str, _Lift] = {"host": np.asarray, "traced": fnp.asarray}
_NAMESPACES = tuple(_LIFT.items())

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


# The three `(rows, candidates, keep)` the two schemes reach the compaction with,
# read off a real call rather than derived here: ML-DSA-65's `ExpandA` and its
# `ExpandS` at `η = 4`, and Falcon-1024's `HashToPoint`. Named after the samplers
# for the reader's sake; nothing below imports a scheme, which is the rule the
# budget cases above already follow.
_SHAPES = (
    ("expand_a", 30, 336, 256),
    ("expand_s", 11, 1088, 256),
    ("hash_to_point", 1, 1360, 1024),
)


def _accepting(rows: int, candidates: int, keep: int, pattern: str) -> np.ndarray:
    """A `[rows, candidates]` mask with at least `keep` survivors per row.

    The patterns are the ones a random stream will not hand over: survivors
    packed against either end, spread to the widest stride that still fits, and
    exactly enough with no slack. Where a survivor sits is what the compaction
    has to get right — a form that reads a rank off the wrong end agrees with the
    reference on a uniform mask and disagrees on these.
    """
    mask = np.zeros((rows, candidates), dtype=bool)
    for row in range(rows):
        # Rotating per row keeps neighbouring rows from sharing a pattern, so a
        # form that silently broadcast one row's indices across the batch fails.
        shift = row % max(1, candidates - keep + 1)
        if pattern == "all":
            chosen = np.arange(candidates)
        elif pattern == "front":
            chosen = np.arange(keep) + shift
        elif pattern == "back":
            chosen = np.arange(candidates - keep, candidates)
        elif pattern == "spread":
            chosen = np.linspace(0, candidates - 1, keep).astype(np.int64)
        elif pattern == "one_to_spare":
            chosen = np.linspace(0, candidates - 1, keep + 1).astype(np.int64)
        else:
            raise ValueError(f"unknown pattern {pattern!r}")
        mask[row, chosen] = True
    return mask


def _reference(values: np.ndarray, accepted: np.ndarray, keep: int) -> np.ndarray:
    """The first `keep` accepted entries of each row, in stream order.

    A host loop over rows, which is the definition the standards write and the
    thing the traced form is a reshaping of. Deliberately not vectorised: a
    reference that shared the implementation's trick would agree with it for the
    same wrong reason.
    """
    return np.stack([row[mask][:keep] for row, mask in zip(values, accepted)])


class FirstAcceptedTest(parameterized.TestCase):
    """The compaction: the first `count` survivors of each row, in stream order."""

    @parameterized.named_parameters(
        (f"{name}_{pattern}_{namespace}", rows, candidates, keep, pattern, lift)
        for name, rows, candidates, keep in _SHAPES
        for pattern in ("all", "front", "back", "spread", "one_to_spare")
        for namespace, lift in _NAMESPACES
    )
    def test_matches_a_host_reference(
        self,
        rows: int,
        candidates: int,
        keep: int,
        pattern: str,
        lift: _Lift,
    ) -> None:
        values = np.arange(rows * candidates, dtype=np.uint32).reshape(rows, candidates)
        accepted = _accepting(rows, candidates, keep, pattern)
        got = rejection.first_accepted(
            lift(values), lift(accepted), keep, "first_accepted_test"
        )
        np.testing.assert_array_equal(
            np.asarray(got), _reference(values, accepted, keep)
        )

    @parameterized.named_parameters(*_NAMESPACES)
    def test_a_signed_candidate_survives_its_sign(self, lift: _Lift) -> None:
        """`ExpandS` compacts int32 centered coefficients, not only unsigned draws.

        A form that reached for an unsigned slot type, or that filled an unwritten
        slot with a value it could not represent, passes every uint32 case above
        and fails here.
        """
        rows, candidates, keep = 11, 1088, 256
        values = (
            np.arange(-(rows * candidates) // 2, (rows * candidates) // 2)
            .astype(np.int32)[: rows * candidates]
            .reshape(rows, candidates)
        )
        accepted = _accepting(rows, candidates, keep, "spread")
        got = rejection.first_accepted(
            lift(values), lift(accepted), keep, "first_accepted_test"
        )
        np.testing.assert_array_equal(
            np.asarray(got), _reference(values, accepted, keep)
        )

    def test_a_shortfall_is_refused_where_it_can_be_seen(self) -> None:
        """On the host the count is a number, so a shortfall raises rather than pads."""
        values = np.arange(2 * 16, dtype=np.uint32).reshape(2, 16)
        accepted = np.zeros((2, 16), dtype=bool)
        accepted[:, :4] = True
        with self.assertRaisesRegex(RuntimeError, "ran out of candidates"):
            rejection.first_accepted(values, accepted, 8, "short")

    def test_a_shortfall_is_pinned_under_a_tracer(self) -> None:
        """Traced, no comparison on the count can raise — so it must be deterministic.

        The budget makes this branch a `2^-256` event, which is why it is reached
        here by hand. What is asserted is not *which* wrong answer comes back —
        that is the compaction form's to choose, and the forms differ — but that
        the same one comes back every time, on whatever backend is running. An
        unpinned edge is how two legs of CI disagree about a value neither is
        supposed to produce.
        """
        values = fnp.asarray(np.arange(2 * 16, dtype=np.uint32).reshape(2, 16))
        accepted = fnp.asarray(
            np.tile(np.arange(16) < 4, (2, 1)),
        )
        compact = frx.jit(
            lambda draws, mask: rejection.first_accepted(draws, mask, 8, "short")
        )
        first = np.asarray(compact(values, accepted))
        self.assertEqual(first.shape, (2, 8))
        np.testing.assert_array_equal(first, np.asarray(compact(values, accepted)))


if __name__ == "__main__":
    absltest.main()
