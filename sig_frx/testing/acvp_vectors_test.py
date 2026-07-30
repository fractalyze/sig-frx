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

import collections

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

    def test_the_operations_the_set_covers_are_expressed_not_refused(self) -> None:
        # The set covers every mode of FIPS 204's interface, and they are separate
        # operations rather than options: the external one wraps the message, the
        # internal one signs it as given, and the pre-hash one signs a digest under
        # a named function. Recording which is which is what lets each be routed to
        # whatever implements it, instead of the whole set being refused because
        # the seam names only one of them.
        self.assertEqual({v.interface for v in self.vectors}, {"external", "internal"})
        pre_hashes = {v.pre_hash for v in self.vectors} - {None}
        self.assertContainsSubset({"SHA2-256", "SHAKE-256"}, pre_hashes)
        self.assertLen(pre_hashes, 12)

    def test_the_external_mu_variant_stays_refused(self) -> None:
        # Its input is a pre-computed 64-byte message representative rather than a
        # message, and nothing here names that operation — so unlike the interface
        # and the pre-hash variant it is recorded as unrunnable. The cases really
        # have no message, so the loader must not have invented one.
        external_mu = [v for v in self.vectors if "mu" in v.unsupported]
        self.assertNotEmpty(external_mu)
        for vector in external_mu:
            self.assertIsNone(vector.message, vector.case_id)
            self.assertContainsSubset({"externalMu", "mu"}, vector.unsupported)

    def test_a_context_is_carried_rather_than_refused(self) -> None:
        # The seam takes a context, so a case carrying one is runnable. Refusing
        # it would exclude most of what FIPS 204 publishes for verification.
        self.assertNotIn(
            "context", {name for v in self.vectors for name in v.unsupported}
        )
        with_context = [v for v in self.vectors if v.context]
        self.assertNotEmpty(with_context)

    def test_each_operation_is_gateable_on_its_own_group(self) -> None:
        # Three parameter sets x fifteen cases per operation. The external pure
        # subset is what the seam itself names; the other two are gateable by
        # whatever implements them, which is the point of expressing the mode.
        runnable = [v for v in self.vectors if not v.unsupported]
        self.assertLen(runnable, 135)
        by_operation = collections.Counter(
            (v.interface, v.pre_hash is not None) for v in runnable
        )
        self.assertEqual(
            by_operation,
            collections.Counter(
                {
                    ("external", False): 45,
                    ("external", True): 45,
                    ("internal", False): 45,
                }
            ),
        )
        for vector in runnable:
            self.assertIsNotNone(vector.message, vector.case_id)

    def test_the_harness_refuses_a_set_carrying_an_operation_nothing_names(
        self,
    ) -> None:
        with self.assertRaisesRegex(kat.KatError, "unsupported fields"):
            kat.check(ChecksumScheme(domain=7), self.vectors)

    def test_the_harness_refuses_vectors_published_for_another_operation(self) -> None:
        # The declaration at the call site is the only thing standing between the
        # plain operation and a set published for the internal one, since a
        # `Signature` cannot say which operation it performs.
        external = [
            v
            for v in self.vectors
            if not v.unsupported
            and v.interface == "external"
            and v.pre_hash is None
            and v.parameter_set == "ML-DSA-44"
        ]
        with self.assertRaisesRegex(kat.KatError, "were published for"):
            kat.check(ChecksumScheme(domain=7), external, interface="internal")
        with self.assertRaisesRegex(kat.KatError, "were published for"):
            kat.check(ChecksumScheme(domain=7), external, pre_hash="SHA2-256")


if __name__ == "__main__":
    absltest.main()
