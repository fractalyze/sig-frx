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
"""

from __future__ import annotations

from absl.testing import absltest

from sig_frx.hashbased.testing import slh_dsa_vectors


class SweepTest(absltest.TestCase):
    def _sweep(self, mode: str) -> int:
        vectors = slh_dsa_vectors.runnable(slh_dsa_vectors.load(mode))
        self.assertNotEmpty(vectors)
        groups = slh_dsa_vectors.group(vectors)
        # Every constructible set is present, so the sweep is a sweep.
        self.assertEqual(
            {operation.parameter_set for operation in groups},
            set(slh_dsa_vectors.CONSTRUCTIBLE_SETS),
        )
        for operation, group in groups.items():
            with self.subTest(str(operation)):
                slh_dsa_vectors.check(operation, group)
        return len(vectors)

    def test_every_published_key_is_reproduced(self) -> None:
        # 10 cases for each of the eight constructible sets.
        self.assertEqual(self._sweep("keyGen"), 80)

    def test_every_published_signature_is_reproduced(self) -> None:
        # 7 pure and 7 internal cases per mode, plus the one pre-hash case per
        # mode whose function hash-frx provides: 30 for each of the eight sets.
        self.assertEqual(self._sweep("sigGen"), 240)

    def test_every_published_verdict_is_reproduced(self) -> None:
        # 14 pure and 14 internal per set, plus the runnable pre-hash cases —
        # one per set, except SLH-DSA-SHAKE-256f, for which ACVP publishes two.
        # So 29 for seven of the eight sets and 30 for that one, not a multiple.
        self.assertEqual(self._sweep("sigVer"), 233)


if __name__ == "__main__":
    absltest.main()
