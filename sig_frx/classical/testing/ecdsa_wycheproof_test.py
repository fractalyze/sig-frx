# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""ECDSA against Wycheproof, the per-PR slice.

Every (message length, verdict) bucket is represented, so each batch shape the
set produces and both verdicts run on every pull request; the exhaustive sweep
is `ecdsa_wycheproof_sweep_test`, in the scheduled run. The encoding boundary
is asserted here in full — the dropped-case counts are cheap and the whole
point is that they cannot drift silently.
"""

from __future__ import annotations

from absl.testing import absltest, parameterized

from sig_frx.classical.testing import ecdsa_wycheproof_vectors as vectors
from sig_frx.testing import kat


class WycheproofSliceTest(parameterized.TestCase):
    @parameterized.named_parameters(*(("_" + name, name) for name in vectors.SCHEMES))
    def test_encoding_boundary_is_all_published_failures(self, curve: str) -> None:
        _, dropped = vectors.load(curve)
        self.assertLen(dropped, vectors.DROPPED[curve])
        accepted = [v.case_id for v in dropped if v.valid]
        self.assertEmpty(
            accepted,
            "a published accepted case no longer fits the fixed encoding: "
            f"{accepted}",
        )

    @parameterized.named_parameters(*(("_" + name, name) for name in vectors.SCHEMES))
    def test_every_bucket_of_the_set(self, curve: str) -> None:
        runnable, _ = vectors.load(curve)
        kat.check(vectors.SCHEMES[curve], vectors.subset(runnable, per_bucket=1))


if __name__ == "__main__":
    absltest.main()
