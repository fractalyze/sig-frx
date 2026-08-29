# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The exhaustive sweep, tagged `slow_kat`: every published record, both degrees.

The merge gate next door ([`falcon_kat_test`](falcon_kat_test.py)) takes the
shortest few records per degree, for the reason its own docstring gives. This is
the other half of that trade — all of them, run by the scheduled job rather than
by a pull request, so bounding the gate costs coverage over the day rather than
coverage outright.

`kat.check` and the one assertion that guards its own bound are all that is here.
The gate's other claims — that the record bound does not empty it, that the
reference accepts every published record, which records have no §3.11.3 form —
are host arithmetic that compiles nothing, so they run unbounded there and
repeating them here would buy nothing.

Running every record does not widen what is *covered* on the verification side:
key generation stays out for the reason [`falcon_vectors`](falcon_vectors.py)
records, and no number of records changes it.

**Signing is the one thing this covers that the gate does not**, and it is
bounded here rather than swept. Signing a record costs tens of seconds against
roughly a second to verify one, and it grows with the degree, so the two do not
belong under one bound: every record reaches verification and `_SIGNED_VECTORS`
of them reach `sign`.

The bound was measured rather than assumed. At two the slowest shard runs
1042.2 s against the 1016.5 s recorded for signing none — inside the spread. At
four it ran 1578.1 s, and `eternal` is the last bucket
(../../../../docs/reference/measurement.md), so a target that outgrows it has
nowhere to be promoted to. All 100 records would be hours.

**Raising it means measuring it again, because the cost is not a rate.**
Algorithm 10 restarts on the norm bound and on a compression that does not fit,
so what a record costs to sign is its trip count, and that is data — the
Falcon-1024 records past the second cost multiples of the first two. What the
extra records would buy is message-length variety in the one pass whose cost
does not come from message length: `Falcon.sign` is concrete, so a longer
message is a longer hash and not another compilation.

What the signed cases are worth, and the reason they are checked with Falcon's
own verifier rather than against the published bytes, is on
`kat.check`'s `not_the_published_signature` and in `kat._check_sign`.

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

# The records per degree this signs, out of the 100 it verifies. A budget rather
# than a property of Falcon — the measurement that chose it is in
# //sig_frx/lattice/falcon/testing/BUILD.bazel beside the deadline it has to fit.
# Two rather than one for the reason `falcon_vectors` gives for its own pair: a
# fluke cannot carry a degree on its own.
_SIGNED_VECTORS = 2


class FalconSweepTest(parameterized.TestCase):
    @parameterized.parameters(*ref.parameter_cases())
    def test_every_published_case_and_what_the_harness_derives(
        self, name: str, **params: Any
    ) -> None:
        del params
        swept = falcon_vectors.vectors(name, limit=None, sign_limit=_SIGNED_VECTORS)

        # Asserted in this case rather than its own, because bazel shards by
        # method and the two `kat.check` calls are the floor this target was
        # sized against. `kat.check` refuses a declaration with nothing to sign,
        # so a bound of zero already fails loudly; what needs saying is that a
        # bound between that and the budget did not quietly shrink.
        signed = [vector for vector in swept if vector.secret_key is not None]
        self.assertLen(signed, _SIGNED_VECTORS)
        self.assertLess(_SIGNED_VECTORS, len(swept), "the bound is not a bound")

        kat.check(
            falcon.named(name),
            swept,
            not_the_published_signature=falcon_vectors.NOT_THE_PUBLISHED_SIGNATURE,
        )


if __name__ == "__main__":
    absltest.main()
