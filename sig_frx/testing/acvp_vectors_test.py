# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The loader reads NIST's real files, not our idea of their shape.

The hand-built fixtures next door pin the join and the field mapping; they cannot
tell us the field names are right, because we wrote both sides. These cases run
the loader over the published ACVP sets themselves — fetched and sha256-pinned in
`//MODULE.bazel`, never committed — so a wrong field name or a missed group-level
field fails here rather than the first time a scheme is gated on them.

They also prove the storage decision end to end: the fetch, the pin, and the
loader are exercised together, so the mechanism the schemes will use is not
theoretical when they arrive.
"""

from __future__ import annotations

from absl.testing import absltest
from python.runfiles import Runfiles

from sig_frx.testing import kat
from sig_frx.testing.checksum_scheme import ChecksumScheme

_RUNFILES = Runfiles.Create()


def _vector_set(prompt_repo: str, expected_repo: str) -> list[kat.KatVector]:
    prompt = _RUNFILES.Rlocation(f"{prompt_repo}/file/prompt.json")
    expected = _RUNFILES.Rlocation(f"{expected_repo}/file/expectedResults.json")
    assert prompt is not None and expected is not None
    return kat.load_acvp(prompt, expected)


class SlhDsaKeyGenTest(absltest.TestCase):
    """FIPS 205 key generation — the seed-assembly case."""

    def setUp(self) -> None:
        super().setUp()
        self.vectors = _vector_set(
            "acvp_slh_dsa_keygen_prompt", "acvp_slh_dsa_keygen_expected"
        )

    def test_every_case_is_loaded(self) -> None:
        self.assertLen(self.vectors, 120)
        self.assertLen({v.parameter_set for v in self.vectors}, 12)

    def test_every_case_carries_a_seed_and_both_keys(self) -> None:
        for vector in self.vectors:
            self.assertIsNotNone(vector.seed, vector.case_id)
            self.assertIsNotNone(vector.public_key, vector.case_id)
            self.assertIsNotNone(vector.secret_key, vector.case_id)

    def test_the_seed_is_the_three_pieces_the_standard_takes(self) -> None:
        # FIPS 205 key generation takes (SK.seed, SK.prf, PK.seed), each n bytes,
        # and publishes the secret key as their concatenation followed by the
        # public root. If the loader assembled them in another order the seed
        # would not prefix the secret key.
        for vector in self.vectors:
            assert vector.seed is not None and vector.secret_key is not None
            self.assertEqual(len(vector.seed) % 3, 0, vector.case_id)
            self.assertTrue(
                vector.secret_key.startswith(vector.seed[: len(vector.seed) // 3 * 2]),
                f"{vector.case_id}: seed pieces are not in the standard's order",
            )


class MlDsaSigVerTest(absltest.TestCase):
    """FIPS 204 verification — the negative-vector case."""

    def setUp(self) -> None:
        super().setUp()
        self.vectors = _vector_set(
            "acvp_ml_dsa_sigver_prompt", "acvp_ml_dsa_sigver_expected"
        )

    def test_the_published_failures_survive_loading(self) -> None:
        # 144 of the 180 are deliberate failures. A loader that dropped the
        # verdict would turn every one of them into a false positive, and the
        # suite would still look green.
        self.assertLen(self.vectors, 180)
        self.assertLen([v for v in self.vectors if not v.valid], 144)

    def test_every_case_carries_a_key_and_a_signature(self) -> None:
        for vector in self.vectors:
            self.assertIsNotNone(vector.public_key, vector.case_id)
            self.assertIsNotNone(vector.signature, vector.case_id)

    def test_the_modes_the_seam_does_not_name_are_recorded_not_dropped(self) -> None:
        # The set covers every mode of FIPS 204's interface. The seam names the
        # external, pure one; the pre-hashed variant, the internal interface, and
        # the external-mu variant whose input is a 64-byte message representative
        # are each a different operation.
        recorded = {name for v in self.vectors for name in v.unsupported}
        self.assertContainsSubset(
            {"signatureInterface", "preHash", "hashAlg", "externalMu", "mu"}, recorded
        )
        # externalMu cases genuinely have no message — the loader must not have
        # invented one.
        external_mu = [v for v in self.vectors if "mu" in v.unsupported]
        self.assertNotEmpty(external_mu)
        for vector in external_mu:
            self.assertIsNone(vector.message, vector.case_id)

    def test_a_context_is_carried_rather_than_refused(self) -> None:
        # The seam takes a context, so a case carrying one is runnable. Refusing
        # it would exclude most of what FIPS 204 publishes for verification.
        self.assertNotIn(
            "context", {name for v in self.vectors for name in v.unsupported}
        )
        with_context = [v for v in self.vectors if v.context]
        self.assertNotEmpty(with_context)

    def test_the_runnable_subset_is_the_external_pure_interface(self) -> None:
        # Three parameter sets x fifteen cases: what a scheme implementing the
        # seam's operation can be gated on today, out of the twelve groups
        # published.
        runnable = [v for v in self.vectors if not v.unsupported]
        self.assertLen(runnable, 45)
        self.assertLen({v.parameter_set for v in runnable}, 3)
        for vector in runnable:
            self.assertIsNotNone(vector.message, vector.case_id)

    def test_the_harness_refuses_them_rather_than_running_the_wrong_operation(
        self,
    ) -> None:
        with self.assertRaisesRegex(kat.KatError, "unsupported fields"):
            kat.check(ChecksumScheme(domain=7), self.vectors)


if __name__ == "__main__":
    absltest.main()
