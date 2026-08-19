# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""ECDSA against every runnable Wycheproof case — the scheduled sweep."""

from __future__ import annotations

from absl.testing import absltest, parameterized

from sig_frx.classical.testing import ecdsa_wycheproof_vectors as vectors
from sig_frx.testing import kat


class WycheproofSweepTest(parameterized.TestCase):
    @parameterized.named_parameters(*(("_" + name, name) for name in vectors.SCHEMES))
    def test_the_whole_set(self, curve: str) -> None:
        runnable, _ = vectors.load(curve)
        kat.check(vectors.SCHEMES[curve], runnable)


if __name__ == "__main__":
    absltest.main()
