# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Falcon against the round-3 vectors, through the shared harness.

[`kat.check`](../../../testing/kat.py) owns three passes and Falcon owns none of
them: the published verdicts, the tampering that moves a bit in each of the three
inputs, and the batch axis built by replicating an accepted case and corrupting
some entries. `testing.md` puts them there rather than in a scheme's own
tests "because the gap is a property of how the vectors are published, not of any
one scheme — a per-scheme fix would be written once per scheme for one cause",
and a Falcon-shaped copy would be the third.

**A verification-only scheme is drivable here, which is the part worth naming.**
`_check_keygen` skips a vector carrying no seed and `_check_sign` one carrying no
secret key, and Falcon's records carry neither — so `keygen` and `sign` raising
until [#26](https://github.com/fractalyze/sig-frx/issues/26) and
[#27](https://github.com/fractalyze/sig-frx/issues/27) is not a reason to opt
out. That also means this file gains those two operations' coverage for free the
day their bodies land and the vectors grow a seed.

The rejections a generic bit flip cannot reach — the uncompressed header byte,
the padding byte, a public key coefficient at or above `q` — are Falcon's own
structure and stay in [`falcon_test.py`](falcon_test.py), which is what
`testing.md` asks a scheme to add on top of the harness rather than instead
of it.
"""

from __future__ import annotations

from typing import Any

from absl.testing import absltest, parameterized

from sig_frx.lattice.falcon import falcon
from sig_frx.lattice.falcon.testing import falcon_reference as ref
from sig_frx.lattice.falcon.testing import falcon_vectors
from sig_frx.testing import kat


class FalconKatTest(parameterized.TestCase):
    @parameterized.parameters(*ref.parameter_cases())
    def test_the_published_cases_and_what_the_harness_derives(
        self, name: str, **params: Any
    ) -> None:
        del params
        kat.check(falcon.named(name), falcon_vectors.vectors(name))

    @parameterized.parameters(*ref.parameter_cases())
    def test_the_reference_accepts_every_transcribed_case(
        self, name: str, **params: Any
    ) -> None:
        """The transcription is evidence only if it reproduces upstream's verdict.

        These bytes were regrouped out of §3.11.6's aggregate into §3.11.3's
        padded signature, so this is what says the regrouping put every field
        where the standard says it goes — independently of the implementation
        the case is meant to gate.
        """
        del params
        for vector in falcon_vectors.VECTORS[name]:
            self.assertTrue(
                ref.verify(
                    bytes.fromhex(vector.public_key),
                    bytes.fromhex(vector.message),
                    bytes.fromhex(vector.signature),
                    name,
                ),
                f"count={vector.case} is not accepted by the reference",
            )


if __name__ == "__main__":
    absltest.main()
