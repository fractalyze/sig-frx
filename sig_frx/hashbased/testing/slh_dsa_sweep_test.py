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

Which operations exist is a product of constants here rather than whatever the
vector files happen to contain, and `CoverageTest` is what keeps the two from
drifting apart: an operation the vectors publish and this does not name would run
nowhere at all, and a sweep that quietly got smaller reads exactly like one that
passed.
"""

from __future__ import annotations

import functools
import re

from absl.testing import absltest, parameterized

from sig_frx.hashbased.testing import slh_dsa_vectors
from sig_frx.testing import kat

# How a case reaches the scheme: the pure external interface, the external one
# under each pre-hash this repo can compute, and §9's internal one. Signing and
# verifying publish all three; key generation has only the one.
_INTERFACES: tuple[tuple[str, str | None], ...] = (
    ("external", None),
    *(("external", pre_hash) for pre_hash in slh_dsa_vectors.PRE_HASHES),
    ("internal", None),
)

# What each ACVP mode publishes, as `slh_dsa_kat_test` states it for the merge
# gate: which interfaces it reaches, and which signing modes it distinguishes.
# `sigGen` publishes deterministic and hedged cases separately; the other two
# modes do not have the distinction — a key pair has no randomness in it, and a
# signature verifies the same way whichever mode made it.
_MODES: dict[
    str, tuple[tuple[tuple[str, str | None], ...], tuple[bool | None, ...]]
] = {
    "keyGen": ((("external", None),), (None,)),
    "sigGen": (_INTERFACES, (True, False)),
    "sigVer": (_INTERFACES, (None,)),
}


@functools.lru_cache(maxsize=None)
def _groups(mode: str) -> dict[slh_dsa_vectors.Operation, list[kat.KatVector]]:
    """One mode's runnable cases, grouped by the operation each belongs to.

    Cached because a sharded run loads the same mode for several of its cases, and
    parsing the published sets again per case would cost more than the shortest of
    them takes to run.
    """
    return slh_dsa_vectors.group(slh_dsa_vectors.runnable(slh_dsa_vectors.load(mode)))


def _operations(mode: str) -> list[slh_dsa_vectors.Operation]:
    """Every operation `mode` publishes, for every constructible set."""
    interfaces, signing_modes = _MODES[mode]
    return [
        slh_dsa_vectors.Operation(
            parameter_set=parameter_set,
            interface=interface,
            pre_hash=pre_hash,
            deterministic=deterministic,
        )
        for parameter_set in slh_dsa_vectors.CONSTRUCTIBLE_SETS
        for interface, pre_hash in interfaces
        for deterministic in signing_modes
    ]


def _cases() -> list[tuple[str, str, slh_dsa_vectors.Operation]]:
    """The sweep as one named case per operation, the mode first.

    The name leads with the mode because that is what orders the cases, and the
    order is what a shard is cut out of: the operations of one mode sit together,
    so consecutive shards take consecutive parameter sets rather than one shard
    taking every expensive one.
    """
    return [
        (re.sub(r"[^0-9A-Za-z]+", "_", f"{mode} {operation}"), mode, operation)
        for mode in _MODES
        for operation in _operations(mode)
    ]


class SweepTest(parameterized.TestCase):
    @parameterized.named_parameters(*_cases())
    def test_the_published_cases_are_reproduced(
        self, mode: str, operation: slh_dsa_vectors.Operation
    ) -> None:
        groups = _groups(mode)
        self.assertIn(operation, groups)
        slh_dsa_vectors.check(operation, groups[operation])


class CoverageTest(absltest.TestCase):
    """What makes the sweep a sweep, asserted apart from running it.

    These are the claims the cases above cannot make one at a time — that every
    set is present, that every published case is claimed by some case, and that
    nothing was dropped between the vectors and the parameterization. They are
    counting and set arithmetic, so they cost nothing next to a single signature.
    """

    def test_every_constructible_set_is_swept(self) -> None:
        for mode in _MODES:
            with self.subTest(mode):
                self.assertEqual(
                    {operation.parameter_set for operation in _groups(mode)},
                    set(slh_dsa_vectors.CONSTRUCTIBLE_SETS),
                )

    def test_every_published_operation_has_a_case(self) -> None:
        # Both directions. An operation the vectors publish and the
        # parameterization does not name would run nowhere; one the
        # parameterization names and the vectors do not publish fails as a missing
        # group, which is a coverage boundary that moved rather than a case.
        for mode in _MODES:
            with self.subTest(mode):
                self.assertEqual(set(_groups(mode)), set(_operations(mode)))

    def test_every_published_case_is_reached(self) -> None:
        # 10 keyGen cases for each of the eight constructible sets. For sigGen, 7
        # pure and 7 internal per mode, plus the one pre-hash case per mode whose
        # function hash-frx provides: 30 per set. For sigVer, 14 pure and 14
        # internal plus the runnable pre-hash cases — one per set, except
        # SLH-DSA-SHAKE-256f, for which ACVP publishes two, so 233 rather than a
        # multiple of eight.
        for mode, expected in (("keyGen", 80), ("sigGen", 240), ("sigVer", 233)):
            with self.subTest(mode):
                cases = sum(len(group) for group in _groups(mode).values())
                self.assertEqual(cases, expected)


if __name__ == "__main__":
    absltest.main()
