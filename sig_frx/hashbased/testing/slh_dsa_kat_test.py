# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""SLH-DSA against the bytes NIST published — the check nothing else here makes.

Every other case in this package compares an implementation against a
transcription of the standard, and one author wrote both sides, so a misreading
they share survives all of it. These cases compare against ACVP's
`SLH-DSA-keyGen`, `-sigGen` and `-sigVer` sets for FIPS 205 instead, fetched and
sha256-pinned in `//MODULE.bazel` rather than committed.

Between them the three sets cover every operation the standard defines, which is
what makes them a gate on the whole scheme rather than on one entry point: key
generation, then signing and verifying under the pure external interface, the
pre-hash external interface and §9's internal one, each in both the deterministic
and the hedged mode. `slh_dsa_vectors` groups the cases by operation and hands
each to the shared harness, whose tampering pass is what stops a verifier that
accepts everything from passing a set of positives.

Key generation is the widest gate for its size. `pk = PK.seed ‖ PK.root`, and
`PK.root` is the root of the top layer's XMSS tree, so reproducing one published
public key confirms the address layouts, the compressed-address slice, the six
§11.2.1 formulas, the WOTS+ chains and their compression, and the tree tweaks —
every reading made in the components beneath this one — from one comparison.

This target gates a merge, so its signature coverage is the `f` parameter set,
whose trees are 8 WOTS+ key pairs against the `s` set's 512. The exhaustive sweep
across every constructible set is `slh_dsa_sweep_test`, tagged `slow_kat`.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

from sig_frx.hashbased import slh_dsa
from sig_frx.hashbased.testing import slh_dsa_vectors
from sig_frx.testing import kat

# The cheap constructible set: 22 layers of 8 key pairs, where `128s` is 7 layers
# of 512 and belongs in the sweep.
_MERGE_GATE_SET = "SLH-DSA-SHA2-128f"


class KeyGenKatTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.vectors = slh_dsa_vectors.load("keyGen")

    def test_the_published_keys_are_reproduced(self) -> None:
        for operation, group in slh_dsa_vectors.group(
            slh_dsa_vectors.runnable(self.vectors)
        ).items():
            with self.subTest(str(operation)):
                self.assertNotEmpty(group)
                slh_dsa_vectors.check(operation, group)

    def test_the_runnable_groups_are_the_ones_the_family_reaches(self) -> None:
        # Which cases ran, stated rather than implied: a set this cannot build is a
        # coverage boundary, and one that silently vanished would read as a pass.
        published = {vector.parameter_set for vector in self.vectors}
        self.assertLen(published, 12)
        self.assertContainsSubset(slh_dsa_vectors.CONSTRUCTIBLE_SETS, published)
        self.assertEqual(
            slh_dsa_vectors.excluded_by_reason(self.vectors),
            {"parameter set needs SHA-512": 40},
        )
        for name in published - set(slh_dsa_vectors.CONSTRUCTIBLE_SETS):
            with self.subTest(name):
                # All that is left out is SHA-2 at categories 3 and 5, which
                # §11.2.2 hashes with SHA-512. Each says so rather than quietly
                # building a family over the wrong hash.
                self.assertIn(name, slh_dsa.SHA2_PARAMETER_SETS)
                with self.assertRaises(NotImplementedError):
                    slh_dsa.sha2(name)

    def test_a_seed_the_standard_did_not_publish_gives_another_key(self) -> None:
        # The published cases all pass under a keygen that ignored its seed and
        # returned a memorized answer per case; this is the case that does not.
        scheme = slh_dsa.sha2(_MERGE_GATE_SET)
        vector = next(v for v in self.vectors if v.parameter_set == _MERGE_GATE_SET)
        assert vector.seed is not None and vector.public_key is not None
        tampered = np.frombuffer(vector.seed, dtype=np.uint8).copy()
        tampered[0] ^= 1
        public_key, _ = scheme.keygen(tampered)
        self.assertNotEqual(kat.to_bytes(public_key), vector.public_key)


class SignatureKatTest(absltest.TestCase):
    """Signing and verifying, for every operation at the merge-gate set."""

    def _groups(
        self, mode: str
    ) -> dict[slh_dsa_vectors.Operation, list[kat.KatVector]]:
        vectors = [
            v
            for v in slh_dsa_vectors.runnable(slh_dsa_vectors.load(mode))
            if v.parameter_set == _MERGE_GATE_SET
        ]
        self.assertNotEmpty(vectors)
        return slh_dsa_vectors.group(vectors)

    def test_the_published_signatures_are_reproduced(self) -> None:
        groups = self._groups("sigGen")
        # Every operation and both modes: the pure and pre-hash external
        # interfaces and the internal one, deterministic and hedged.
        self.assertEqual(
            {
                (op.interface, op.pre_hash is not None, op.deterministic)
                for op in groups
            },
            {
                (interface, pre_hash, deterministic)
                for interface, pre_hash in (
                    ("external", False),
                    ("external", True),
                    ("internal", False),
                )
                for deterministic in (True, False)
            },
        )
        for operation, group in groups.items():
            with self.subTest(str(operation)):
                slh_dsa_vectors.check(operation, group)

    def test_the_published_verdicts_are_reproduced(self) -> None:
        groups = self._groups("sigVer")
        accepted = 0
        rejected = 0
        for operation, group in groups.items():
            with self.subTest(str(operation)):
                slh_dsa_vectors.check(operation, group)
            accepted += sum(1 for v in group if v.valid)
            rejected += sum(1 for v in group if not v.valid)
        # Mostly failures by design, and a suite of positives alone would say
        # nothing about rejection.
        self.assertGreater(rejected, accepted)
        self.assertGreater(accepted, 0)

    def test_the_wrong_length_signatures_are_among_the_published_failures(self) -> None:
        # Algorithm 20's length check has no other exercise: these cases carry a
        # signature one byte short or long, so a verifier that raised instead of
        # rejecting would fail the set rather than pass it.
        size = slh_dsa.SHA2_PARAMETER_SETS[_MERGE_GATE_SET].signature_size
        wrong_length = [
            v
            for group in self._groups("sigVer").values()
            for v in group
            if v.signature is not None and len(v.signature) != size
        ]
        self.assertNotEmpty(wrong_length)
        for vector in wrong_length:
            self.assertFalse(vector.valid, vector.case_id)

    def test_the_pre_hash_functions_the_sets_reach_are_stated(self) -> None:
        # ACVP exercises twelve pre-hash functions and hash-frx provides one. The
        # rest are excluded because a pre-hash case signs its function's OID, so no
        # stand-in computes the right message.
        published = {
            v.pre_hash for v in slh_dsa_vectors.load("sigGen") if v.pre_hash is not None
        }
        self.assertLen(published, 12)
        self.assertEqual(set(slh_dsa_vectors.PRE_HASHES), {"SHA2-256"})
        self.assertContainsSubset(slh_dsa_vectors.PRE_HASHES, published)


if __name__ == "__main__":
    absltest.main()
