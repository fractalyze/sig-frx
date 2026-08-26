# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""ML-DSA against the bytes NIST published — the check nothing else here makes.

Every other case in this package compares an implementation against a
transcription of the standard, and one author wrote both sides, so a misreading
they share survives all of it. These cases compare against ACVP's `ML-DSA-keyGen`,
`-sigGen` and `-sigVer` sets for FIPS 204 instead, fetched and sha256-pinned in
`//MODULE.bazel` rather than committed.

ML-DSA has more places to diverge than anything else here — the transform's
ordering, four samplers, the rounding functions, half a dozen packing widths — and
a round trip exercises all of them consistently and catches none of them.
`sigGen` reaches further still: a signature is the rejection loop's output at
whichever iteration accepted, so reproducing one pins every candidate rejected
before it, and the loop has no other observable.

Key generation runs at all three parameter sets because it is the widest gate for
its cost: `pk = ρ ‖ t1` with `t = As1 + s2` rounded, so one published public key
confirms `ExpandA`'s ordering, `ExpandS`'s coefficient range, the transform's root
and `Power2Round` at once, without a rejection loop in the way. Signing and
verifying are the merge gate's expensive half and run at ML-DSA-44, whose `A` is
4x4 against ML-DSA-87's 8x7, and at a bounded number of vectors per operation,
because what this target costs is the number of distinct message shapes it
compiles rather than the number of vectors it checks; the exhaustive run across
all three sets and every published length is `ml_dsa_sweep_test`, tagged
`slow_kat`.

**The pre-hash operations are gated on the same evidence as the pure one**, which
takes sourcing an accepted case from `sigGen` for the ones ACVP's random draw
left with nothing accepted. What that is worth is asserted here rather than
argued: a `hash_verify` that rejects everything fails this target at every
pre-hash function hash-frx provides, and not only at the ones the draw happened
to accept a case for.

**Two coverage boundaries and one refusal**, each asserted rather than described.
ACVP exercises twelve pre-hash functions and hash-frx provides five. Its
`externalMu` cases hand over a pre-computed message representative in place of a
message, which is an operation nothing here names, so the harness refuses them.
And unlike FIPS 205's set, this one publishes no wrong-length signature, so §3.6.2's
length verdict has no published exercise — `ml_dsa_test` is what covers it.
"""

from __future__ import annotations

from collections.abc import Sequence

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import Array
from frx.typing import ArrayLike

from sig_frx import prehash
from sig_frx.lattice.mldsa import ml_dsa
from sig_frx.lattice.mldsa.testing import ml_dsa_vectors
from sig_frx.testing import kat

# The cheapest set to sign and verify at: Table 1's smallest `A`, and the one
# whose 78-byte `β` bound makes the rejection loop run fewest times on average.
_MERGE_GATE_SET = "ML-DSA-44"

# How many published vectors of each operation the merge gate runs.
#
# What this target costs is set by how many distinct *shapes* it reaches, not by
# how many vectors: ACVP draws a fresh message length per case, and a traced
# sponge compiles once per distinct one. The published set is nearly all
# singletons, so the gate's 109 vectors are 99 compiles.
#
# That is free on a backend that declines the whole-hash Keccak marker and
# ruinous on one that routes it, because the routed composite's decomposition is
# the entire absorb and squeeze — traced per shape, and in proportion to the
# message's block count. Measured: a 61-block message costs 22x a 6-block one,
# while a second message landing in a bucket already compiled costs ~1% of a
# fresh one. Bounding the *count* per operation is therefore what bounds the cost,
# and padding lengths to a rate multiple does nothing at all — that is already
# the bucket.
#
# The shortest few rather than a length cap: a flat cap empties ten of the
# fourteen `sigGen` operations, because ACVP's pre-hash cases all carry long
# messages, and an operation that runs nowhere is not a cheaper gate. Every
# operation, both signing modes and all five pre-hash functions survive this,
# which the cases below assert rather than assume. The exhaustive run over every
# published length is `ml_dsa_sweep_test`.
_MERGE_GATE_VECTORS_PER_OPERATION = 4


def _bounded(group: Sequence[kat.KatVector]) -> list[kat.KatVector]:
    """An operation's shortest published messages, which is its cheapest shapes.

    Shortest because trace cost rises with the message's block count, so the
    same bound over the long end of the set would buy a fraction as much.
    """
    return sorted(group, key=lambda vector: len(vector.message or b""))[
        :_MERGE_GATE_VECTORS_PER_OPERATION
    ]


# The whole published file's gap, not the gate set's — a boundary is a property of
# what this repo can compute, so it is stated once against everything NIST ships.
_EXCLUDED = {
    "keyGen": {},
    "sigGen": {
        "operation nothing here names": 90,
        "pre-hash function hash-frx does not provide": 48,
    },
    "sigVer": {
        "operation nothing here names": 45,
        "pre-hash function hash-frx does not provide": 26,
    },
}


class _RejectsEveryPreHashedSignature(ml_dsa.MlDsa):
    """A pre-hash verifier that answers before it looks at anything.

    The one failure mode an operation gated on published failures alone cannot
    tell from a correct implementation: every verdict such a group publishes is a
    rejection, and so is every verdict this returns.
    """

    def hash_verify(
        self,
        public_key: ArrayLike,
        message: ArrayLike,
        signature: ArrayLike,
        pre_hash: prehash.PreHash,
        *,
        context: ArrayLike | None = None,
    ) -> Array:
        return fnp.zeros(np.shape(public_key)[0], dtype=fnp.bool_)


class KeyGenKatTest(absltest.TestCase):
    """All three sets: 25 published seeds each, against both key encodings."""

    def setUp(self) -> None:
        super().setUp()
        self.vectors = ml_dsa_vectors.load("keyGen")

    def test_the_published_keys_are_reproduced(self) -> None:
        for operation, group in ml_dsa_vectors.group(
            ml_dsa_vectors.runnable(self.vectors)
        ).items():
            with self.subTest(str(operation)):
                self.assertNotEmpty(group)
                ml_dsa_vectors.check(operation, group)

    def test_every_published_set_is_one_the_family_builds(self) -> None:
        # Which cases ran, stated rather than implied. ML-DSA instantiates every
        # function of every set with SHAKE, so unlike FIPS 205 there is no second
        # hash family to be missing and nothing is left out here at all.
        published = {vector.parameter_set for vector in self.vectors}
        self.assertEqual(published, set(ml_dsa_vectors.CONSTRUCTIBLE_SETS))
        self.assertEqual(published, set(ml_dsa.PARAMETER_SETS))
        self.assertEqual(ml_dsa_vectors.excluded_by_reason(self.vectors), {})

    def test_a_seed_the_standard_did_not_publish_gives_another_key(self) -> None:
        # The published cases all pass under a keygen that ignored its seed and
        # returned a memorized answer per case; this is the case that does not.
        scheme = ml_dsa.named(_MERGE_GATE_SET)
        vector = next(v for v in self.vectors if v.parameter_set == _MERGE_GATE_SET)
        assert vector.seed is not None and vector.public_key is not None
        tampered = np.frombuffer(vector.seed, dtype=np.uint8).copy()
        tampered[0] ^= 1
        public_key, _ = scheme.keygen(tampered)
        self.assertNotEqual(kat.to_bytes(public_key), vector.public_key)


class SignatureKatTest(absltest.TestCase):
    """Signing and verifying, for every operation at the merge-gate set."""

    def _published(
        self, mode: str
    ) -> dict[ml_dsa_vectors.Operation, list[kat.KatVector]]:
        """Every vector the merge-gate set publishes, before the count bound.

        For the claims that are set and length arithmetic over already-parsed
        vectors: those cost nothing next to a single signature, so they are made
        about what NIST published rather than about the subset that runs.
        """
        groups = {
            operation: group
            for operation, group in ml_dsa_vectors.runnable_groups(mode).items()
            if operation.parameter_set == _MERGE_GATE_SET
        }
        self.assertNotEmpty(groups)
        return groups

    def _groups(self, mode: str) -> dict[ml_dsa_vectors.Operation, list[kat.KatVector]]:
        """What runs: every operation, bounded to its cheapest shapes."""
        return {
            operation: _bounded(group)
            for operation, group in self._published(mode).items()
        }

    def test_the_published_signatures_are_reproduced(self) -> None:
        groups = self._groups("sigGen")
        # Every operation and both modes: the pure and pre-hash external
        # interfaces and the internal one, deterministic and hedged. Against
        # `operations` rather than a list written here, so that the gate and the
        # sweep cannot come to disagree about what a mode publishes.
        self.assertEqual(
            set(groups),
            {
                operation
                for operation in ml_dsa_vectors.operations()["sigGen"]
                if operation.parameter_set == _MERGE_GATE_SET
            },
        )
        for operation, group in groups.items():
            with self.subTest(str(operation)):
                ml_dsa_vectors.check(operation, group)

    def test_the_published_verdicts_are_reproduced(self) -> None:
        groups = self._groups("sigVer")
        accepted = 0
        rejected = 0
        for operation, group in groups.items():
            with self.subTest(str(operation)):
                ml_dsa_vectors.check(operation, group)
            accepted += sum(1 for v in group if v.valid)
            rejected += sum(1 for v in group if not v.valid)
        # Mostly failures by design, and a suite of positives alone would say
        # nothing about rejection.
        self.assertGreater(rejected, accepted)
        self.assertGreater(accepted, 0)

    def test_every_operation_with_no_accepted_case_is_handed_one(self) -> None:
        # `kat.check` refuses a set whose derived negative checks have no accepted
        # case to start from, so what keeps the sweep green is that `sigGen`
        # reaches every operation whose verification cases are all failures — and
        # the sweep is the only thing that runs the sets outside the merge gate.
        # This is set arithmetic and a public key per operation over already-parsed
        # vectors, which is what lets the whole census be a merge-gate claim rather
        # than leaving most of it to the scheduled job.
        observed = {
            operation: group
            for operation, group in ml_dsa_vectors.runnable_groups("sigVer").items()
            if not any(vector.valid for vector in group)
        }
        self.assertNotEmpty(observed)
        for operation, group in observed.items():
            with self.subTest(str(operation)):
                # Every one of them is a pre-hash operation, which is the shape of
                # the gap: ACVP draws the function per case, so the pure and
                # internal groups are large enough to always hold an accepted one
                # and these are not.
                self.assertIsNotNone(operation.pre_hash, str(operation))
                self.assertIsNotNone(ml_dsa_vectors.accepted_case(operation, group))

    def test_a_pre_hash_verifier_that_rejects_everything_fails_the_gate(self) -> None:
        # The failure a set of published failures cannot see, at every pre-hash
        # function this repo can compute rather than at the ones the draw happened
        # to accept a case for: three of the five reach `kat.check` with nothing
        # published to accept, and what fails them is the case sourced from
        # `sigGen`. Sabotaging the whole of `hash_verify` at once is what an OID
        # read wrongly, a wrong domain separator or a dropped context does to a
        # pre-hash operation, and each of those verifies nothing while agreeing
        # with every verdict such a group publishes.
        groups = self._groups("sigVer")
        pre_hashed = {
            operation: group
            for operation, group in groups.items()
            if operation.pre_hash is not None
        }
        self.assertEqual(
            {operation.pre_hash for operation in pre_hashed},
            set(ml_dsa_vectors.PRE_HASHES),
        )
        for operation, group in pre_hashed.items():
            with self.subTest(str(operation)):
                with self.assertRaisesRegex(
                    kat.KatError, "published verdict is accept"
                ):
                    ml_dsa_vectors.check(
                        operation,
                        group,
                        scheme=_RejectsEveryPreHashedSignature(
                            ml_dsa.PARAMETER_SETS[operation.parameter_set]
                        ),
                    )

    def test_every_operation_survives_the_count_bound(self) -> None:
        # What this target costs is the bound, so the bound is asserted rather
        # than left to a constant nothing reads. A change that quietly widened it
        # would otherwise surface only as a slow leg — which is how it got here —
        # and one that narrowed it past an operation would report a smaller gate
        # as a passing one.
        for mode in ("sigGen", "sigVer"):
            with self.subTest(mode):
                published = self._published(mode)
                running = self._groups(mode)
                self.assertEqual(set(running), set(published))
                self.assertLess(
                    sum(len(group) for group in running.values()),
                    sum(len(group) for group in published.values()),
                )
                for operation, group in running.items():
                    with self.subTest(str(operation)):
                        self.assertNotEmpty(group)
                        self.assertLessEqual(
                            len(group), _MERGE_GATE_VECTORS_PER_OPERATION
                        )
                        # The cheapest shapes and not any subset of that size:
                        # trace cost rises with the message's block count, so
                        # which ones are kept is the whole point.
                        kept = sorted(len(v.message or b"") for v in group)
                        shortest = sorted(
                            len(v.message or b"") for v in published[operation]
                        )[: len(group)]
                        self.assertEqual(kept, shortest)

    def test_the_boundaries_of_what_ran_are_the_ones_stated(self) -> None:
        for mode, expected in _EXCLUDED.items():
            with self.subTest(mode):
                self.assertEqual(
                    ml_dsa_vectors.excluded_by_reason(ml_dsa_vectors.load(mode)),
                    expected,
                )

    def test_the_external_mu_cases_are_refused_rather_than_run(self) -> None:
        # The refusal is the harness's, not a filter here: those cases carry a
        # message representative instead of a message, so running the operation
        # this repo does implement against them would report a pass for a case
        # nobody ran.
        vectors = ml_dsa_vectors.load("sigVer")
        external_mu = [v for v in vectors if v.unsupported]
        self.assertNotEmpty(external_mu)
        for vector in external_mu:
            self.assertContainsSubset({"externalMu", "mu"}, vector.unsupported)
            self.assertIsNone(vector.message, vector.case_id)
        with self.assertRaisesRegex(kat.KatError, "unsupported fields"):
            ml_dsa_vectors.check(
                ml_dsa_vectors.Operation(_MERGE_GATE_SET, "internal", None, None),
                external_mu,
            )

    def test_the_pre_hash_functions_the_sets_reach_are_stated(self) -> None:
        # ACVP exercises twelve pre-hash functions and hash-frx provides five. The
        # rest are excluded because a pre-hash case signs its function's OID, so no
        # stand-in computes the right message.
        published = {
            v.pre_hash for v in ml_dsa_vectors.load("sigGen") if v.pre_hash is not None
        }
        self.assertLen(published, 12)
        self.assertContainsSubset(ml_dsa_vectors.PRE_HASHES, published)
        self.assertLen(ml_dsa_vectors.PRE_HASHES, 5)

    def test_no_published_case_exercises_the_wrong_length_verdict(self) -> None:
        # FIPS 205's sigVer set carries signatures one byte short or long and this
        # one does not, so §3.6.2's "verify as false rather than raise" has no
        # published exercise. Asserted rather than assumed: a regenerated set that
        # started publishing them would be coverage this gate should pick up.
        params = ml_dsa.PARAMETER_SETS[_MERGE_GATE_SET]
        wrong_length = [
            v
            for group in self._published("sigVer").values()
            for v in group
            if v.signature is not None and len(v.signature) != params.signature_size
        ]
        self.assertEmpty(wrong_length)

    def test_no_published_group_holds_two_cases_of_one_shape(self) -> None:
        # Why `kat.check` builds a batch instead of finding one: a batch needs a
        # static shape and every case here is published with a message length of
        # its own, so the harness's shape grouping yields nothing but `B = 1`
        # groups and its per-entry check has no second entry to pin. Asserted
        # rather than assumed — a regenerated set that published two cases of one
        # shape would be a published batch, and this is what says so.
        for operation, group in self._published("sigVer").items():
            with self.subTest(str(operation)):
                shapes = {
                    (
                        len(v.public_key or b""),
                        len(v.message or b""),
                        len(v.signature or b""),
                        v.context,
                    )
                    for v in group
                }
                self.assertLen(shapes, len(group))


if __name__ == "__main__":
    absltest.main()
