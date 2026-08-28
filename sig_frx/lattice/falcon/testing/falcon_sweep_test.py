# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The exhaustive sweep, tagged `slow_kat`: every published record, both degrees.

The merge gate next door ([`falcon_kat_test`](falcon_kat_test.py)) takes the
shortest few records per degree, for the reason its own docstring gives. This is
the other half of that trade — all of them, run by the scheduled job rather than
by a pull request, so bounding the gate costs coverage over the day rather than
coverage outright.

Only `kat.check` is here. The gate's other claims — that the bound does not empty
it, that the reference accepts every published record, which records have no
§3.11.3 form — are all host arithmetic that compiles nothing, so they run
unbounded there and repeating them here would buy nothing.

Running every record does not widen what is *covered*: key generation and signing
stay out for the reasons [`falcon_vectors`](falcon_vectors.py) records, and no
number of records changes either.

## The wall clock has a floor, and splitting the set does not lower it

`kat.check` over one parameter set is a single test method and bazel shards by
method, so two methods put a floor under this target that no `shard_count` can
go below. Chunking the records into more cases to break that floor was tried
and measured, and it is worse in both directions — the numbers are in
//sig_frx/lattice/falcon/testing/BUILD.bazel. Each shard is a process that
re-pays the import and warm-up, and at 100 records per degree that constant is
large next to the per-record work it would be dividing.
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
        kat.check(falcon.named(name), falcon_vectors.vectors(name, limit=None))


if __name__ == "__main__":
    absltest.main()
