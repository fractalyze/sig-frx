# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The exhaustive sweep, tagged `slow_kat`: every published record, both degrees.

The merge gate next door ([`falcon_kat_test`](falcon_kat_test.py)) takes the
shortest few records per degree, because the generator gives record `i` a
message of `33·(i+1)` bytes and a traced `hash_to_point` compiles once per
distinct length. This is the other half of that trade — all 100 per degree, run
by the scheduled job rather than by a pull request, so bounding the gate costs
coverage over the day rather than coverage outright.

Two passes, and the split is the point:

- **`kat.check`** drives the whole harness — the published verdicts plus the
  tampering and batch passes it derives — over every record. This is the
  expensive one and the reason the file is tagged.
- **the reference's own verdict** over every record, which is host arithmetic
  and cheap. It gates the `sm` → §3.11.3 regrouping across the full set rather
  than across the first few, which matters because a later record could carry a
  field width the early ones do not exercise.

What this sweep does **not** cover is the same thing the gate does not: key
generation and signing. Every record carries `sk` and a `seed`, and neither is
usable here — the seed expands through NIST's AES-256-CTR-DRBG rather than this
repo's `SHAKE256(seed ‖ attempt)`, and a `secret_key` on a record would drive
the harness into a `sign` that [#27](https://github.com/fractalyze/sig-frx/issues/27)
has not finished. Running more records changes neither, which is worth saying
here so that "the exhaustive run passed" is not read as "everything published is
checked".
"""

from __future__ import annotations

from typing import Any

from absl.testing import absltest, parameterized

from sig_frx.lattice.falcon import falcon
from sig_frx.lattice.falcon.testing import falcon_reference as ref
from sig_frx.lattice.falcon.testing import falcon_vectors
from sig_frx.testing import kat


class FalconSweepTest(parameterized.TestCase):
    @parameterized.parameters(*ref.parameter_cases())
    def test_every_published_case_and_what_the_harness_derives(
        self, name: str, **params: Any
    ) -> None:
        del params
        vectors = falcon_vectors.vectors(name, limit=None)
        self.assertLen(vectors, len(falcon_vectors.records(name)))
        kat.check(falcon.named(name), vectors)

    @parameterized.parameters(*ref.parameter_cases())
    def test_the_reference_accepts_every_published_case(
        self, name: str, **params: Any
    ) -> None:
        """The regrouping, across the whole file rather than its first records."""
        del params
        for record in falcon_vectors.records(name):
            self.assertTrue(
                ref.verify(record.public_key, record.message, record.signature, name),
                f"count={record.case} is not accepted by the reference",
            )


if __name__ == "__main__":
    absltest.main()
