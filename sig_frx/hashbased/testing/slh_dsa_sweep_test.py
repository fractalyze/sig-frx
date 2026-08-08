# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Every constructible SLH-DSA parameter set against every published vector.

The exhaustive sweep, tagged `slow_kat`: the merge gate next door runs the same
operations at the `f` parameter set, whose XMSS trees are 8 WOTS+ key pairs, and
this adds `s`, whose are 512. Signing walks a whole tree per hypertree layer, so
the `s` set costs roughly what its 64-fold wider trees suggest — which is why an
exhaustive run belongs in the scheduled job rather than in every review.

What it adds beyond size is the guarantee the standard actually asks for: no
parameter set is a representative sample of another. The sets differ in the tree
geometry, in how many digest bytes each index consumes, and in how many FORS trees
sign the digest, so `f` passing says nothing about `s`.

**One test per published operation, because the cost is not spread evenly.**
Signing at the four `s` sets is two thirds of the sweep, and the largest single
operation is a fourteenth of it — so a sweep written as one method per ACVP mode
cannot be divided below `sigGen`, and its budget has to cover every set's
hypertree walk at once. Cut at the operation instead — the unit
`slh_dsa_vectors.check` already takes — and the longest piece is one signing
operation, which is what lets `shard_count` divide the run and gives each shard a
budget that means something. A failure also names the operation rather than the
sweep.

Which operations exist is `slh_dsa_vectors.operations`, a product of constants
rather than whatever the vector files happen to contain, and `CoverageTest` is
what keeps the two from drifting apart: an operation the vectors publish and that
product does not name would run nowhere at all, and a sweep that quietly got
smaller reads exactly like one that passed.
"""

from __future__ import annotations

from absl.testing import absltest, parameterized

from sig_frx.hashbased.testing import slh_dsa_vectors

# One case per published operation, named mode first and parameter set last. The
# order is what the shards are cut out of — bazel assigns cases round-robin over
# their sorted names — and trailing the set is what lands a shard on operations
# of one parameter set: each then compiles that set's kernels once, where a set
# scattered across eight shards compiles in all eight. Naming it the other way
# round costs a third of the sweep's total processor time in repeated compiles.
_CASES = [
    (f"{mode}_{operation.name}", mode, operation)
    for mode, published in slh_dsa_vectors.operations().items()
    for operation in published
]


class SweepTest(parameterized.TestCase):
    @parameterized.named_parameters(*_CASES)
    def test_the_published_cases_are_reproduced(
        self, mode: str, operation: slh_dsa_vectors.Operation
    ) -> None:
        slh_dsa_vectors.check(
            operation, slh_dsa_vectors.runnable_groups(mode)[operation]
        )


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
        # constructible set is swept. An operation the vectors publish and the
        # product does not name would run nowhere; one the product names and the
        # vectors do not publish is a coverage boundary that moved.
        for mode, published in slh_dsa_vectors.operations().items():
            with self.subTest(mode):
                self.assertEqual(
                    set(slh_dsa_vectors.runnable_groups(mode)), set(published)
                )

    def test_every_published_case_is_reached(self) -> None:
        # 10 keyGen cases for each of the eight constructible sets. For sigGen, 7
        # pure and 7 internal per mode, plus the one pre-hash case per mode whose
        # function hash-frx provides: 30 per set. For sigVer, 14 pure and 14
        # internal plus the runnable pre-hash cases — one per set, except
        # SLH-DSA-SHAKE-256f, for which ACVP publishes two, so 233 rather than a
        # multiple of eight.
        for mode, expected in (("keyGen", 80), ("sigGen", 240), ("sigVer", 233)):
            with self.subTest(mode):
                groups = slh_dsa_vectors.runnable_groups(mode)
                self.assertEqual(sum(len(group) for group in groups.values()), expected)


if __name__ == "__main__":
    absltest.main()
