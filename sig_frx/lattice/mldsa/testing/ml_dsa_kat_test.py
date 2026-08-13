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
4x4 against ML-DSA-87's 8x7; the exhaustive run across all three is
`ml_dsa_sweep_test`, tagged `slow_kat`.

**Two coverage boundaries and one refusal**, each asserted rather than described.
ACVP exercises twelve pre-hash functions and hash-frx provides five. Its
`externalMu` cases hand over a pre-computed message representative in place of a
message, which is an operation nothing here names, so the harness refuses them.
And unlike FIPS 205's set, this one publishes no wrong-length signature, so §3.6.2's
length verdict has no published exercise — `ml_dsa_test` is what covers it.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

from sig_frx.lattice.mldsa import ml_dsa
from sig_frx.lattice.mldsa.testing import ml_dsa_vectors
from sig_frx.testing import kat

# The cheapest set to sign and verify at: Table 1's smallest `A`, and the one
# whose 78-byte `β` bound makes the rejection loop run fewest times on average.
_MERGE_GATE_SET = "ML-DSA-44"

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

    def _groups(self, mode: str) -> dict[ml_dsa_vectors.Operation, list[kat.KatVector]]:
        groups = {
            operation: group
            for operation, group in ml_dsa_vectors.runnable_groups(mode).items()
            if operation.parameter_set == _MERGE_GATE_SET
        }
        self.assertNotEmpty(groups)
        return groups

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
            for group in self._groups("sigVer").values()
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
        for operation, group in self._groups("sigVer").items():
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
