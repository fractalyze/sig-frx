# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Every ML-DSA parameter set against every published vector.

The exhaustive sweep, tagged `slow_kat`: the merge gate next door signs and
verifies at ML-DSA-44, and this adds ML-DSA-65 and ML-DSA-87. Table 1 is not a
scale — `A` grows from 4x4 to 8x7, `η` and `τ` move in opposite directions, `γ2`
changes the rounding and with it `w1Encode`'s packing width, and `λ` changes how
much of the commitment hash a signature carries. So ML-DSA-44 passing says
nothing about the other two, and the run that says something about all three
costs enough to belong in the scheduled job rather than in every review.

**One test per published operation, so a failure names the operation.** That is
the unit `ml_dsa_vectors.check` already takes, and the alternative — one method
per ACVP mode — reports `sigGen` failing without saying which of its fourteen
operations did. It is not for sharding: the case work here is seconds against a
fixed startup cost that every shard would pay in full, which
[`BUILD.bazel`](BUILD.bazel) measures and is why this target runs undivided.

Which operations exist is `ml_dsa_vectors.operations`, a product of constants
rather than whatever the vector files happen to contain, and `CoverageTest` is
what keeps the two from drifting apart: an operation the vectors publish and that
product does not name would run nowhere at all, and a sweep that quietly got
smaller reads exactly like one that passed.
"""

from __future__ import annotations

from absl.testing import absltest, parameterized

from sig_frx.lattice.mldsa.testing import ml_dsa_vectors

# One case per published operation, named mode first and parameter set last —
# which keeps one set's operations adjacent in sorted order, the order a sharded
# run would be cut out of if this target ever grows into wanting shards.
_CASES = [
    (f"{mode}_{operation.name}", mode, operation)
    for mode, published in ml_dsa_vectors.operations().items()
    for operation in published
]


class SweepTest(parameterized.TestCase):
    @parameterized.named_parameters(*_CASES)
    def test_the_published_cases_are_reproduced(
        self, mode: str, operation: ml_dsa_vectors.Operation
    ) -> None:
        ml_dsa_vectors.check(operation, ml_dsa_vectors.runnable_groups(mode)[operation])


class CoverageTest(absltest.TestCase):
    """What makes the sweep a sweep, asserted apart from running it.

    These are the claims the cases above cannot make one at a time — that the
    operations the vectors publish are the ones parameterized, and that nothing
    was dropped between the files and the run. They are set and count arithmetic
    over an already-parsed set of vectors, so they cost nothing next to a single
    signature.
    """

    def test_every_published_operation_has_a_case(self) -> None:
        # Both directions, which is what also makes this the statement that every
        # parameter set is swept. An operation the vectors publish and the product
        # does not name would run nowhere; one the product names and the vectors do
        # not publish is a coverage boundary that moved.
        for mode, published in ml_dsa_vectors.operations().items():
            with self.subTest(mode):
                self.assertEqual(
                    set(ml_dsa_vectors.runnable_groups(mode)), set(published)
                )

    def test_every_published_case_is_reached(self) -> None:
        # 25 keyGen cases for each of the three sets. For sigGen, 15 pure and 15
        # internal per signing mode — 30 each — plus the pre-hash cases whose
        # function hash-frx provides, which is 42 of the 90 published; ACVP draws
        # each case's function at random from twelve, so that share is not a round
        # number and neither is sigVer's 19 of 45.
        for mode, expected in (("keyGen", 75), ("sigGen", 222), ("sigVer", 109)):
            with self.subTest(mode):
                groups = ml_dsa_vectors.runnable_groups(mode)
                self.assertEqual(sum(len(group) for group in groups.values()), expected)


if __name__ == "__main__":
    absltest.main()
