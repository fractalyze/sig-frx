# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Falcon against the round-3 vectors, through the shared harness.

[`kat.check`](../../../testing/kat.py) owns three passes and Falcon owns none of
them: the published verdicts, the tampering that moves a bit in each of the three
inputs, and the batch axis built by replicating an accepted case and corrupting
some entries. `testing.md` puts them there rather than in a scheme's own
tests "because the gap is a property of how the vectors are published, not of any
one scheme — a per-scheme fix would be written once per scheme for one cause",
and a Falcon-shaped copy would be the third.

**Key generation stays uncovered by this set, and will.** The reason is not that
the vectors are missing it — every record carries a `seed` — but that it is not
reproducible here: the seed expands through NIST's AES-256-CTR-DRBG rather than
this repo's `SHAKE256(seed ‖ attempt)`, a decision recorded on
[#26](https://github.com/fractalyze/sig-frx/issues/26).
[`falcon_vectors`](falcon_vectors.py) withholds the field for that reason and
says so; this is where the consequence is recorded, because "the harness skipped
it" and "the harness cannot check it" read identically from a green run.

**Signing is covered by this set, and not from here — `sign_limit` is 0 below.**
Every record carries `sk`, so the harness can sign from published inputs, and
[`falcon_sweep_test`](falcon_sweep_test.py) is where it does. What keeps it out
of the merge gate is arithmetic rather than doubt: signing a record costs tens
of seconds against roughly a second for the verification passes, and this target
already sits at 44% of `large` — so signing even the four records it verifies
took it 2.3x, past the half-a-budget rule in
../../../../docs/reference/measurement.md and into the next bucket. The bucket
is not what the coverage is worth. Every pull request already holds signing to
the reference implementation itself in
[`falcon_interop_test`](falcon_interop_test.py), which `testing.md` ranks above
the round trip the harness pass would add.

## The bound, and why it is stated at the call rather than defaulted

The generator gives record `i` a message of `33·(i+1)` bytes, so the published
set is 100 distinct message lengths per degree and a traced `hash_to_point`
compiles once per length. That is ML-DSA's cost driver applied to Falcon, and
the answer `testing.md` gives is the same: bound the per-PR gate by the number
of vectors, keep the exhaustive run behind `slow_kat`
([`falcon_sweep_test`](falcon_sweep_test.py)), and assert the bound cannot
silently eat coverage.

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

# The vectors per degree this gate takes, out of the 100 published. A budget
# rather than a property of Falcon, so the measurement that chose it lives in
# //sig_frx/lattice/falcon/testing/BUILD.bazel beside the bucket it has to fit.
_GATE_VECTORS = 4

# The published records §3.11.3 cannot express, pinned here rather than in the
# loader: a boundary is a claim about the source, and `ml_dsa_kat_test`'s
# `_EXCLUDED` states the same kind of claim in the same place. The loader
# reports it and this asserts it, so a regenerated source fails one test rather
# than every consumer of the fixture.
_UNENCODABLE = {"Falcon-512": (), "Falcon-1024": (82,)}


class FalconKatTest(parameterized.TestCase):
    @parameterized.parameters(*ref.parameter_cases())
    def test_the_published_cases_and_what_the_harness_derives(
        self, name: str, **params: Any
    ) -> None:
        del params
        kat.check(
            falcon.named(name),
            falcon_vectors.vectors(name, limit=_GATE_VECTORS, sign_limit=0),
        )

    @parameterized.parameters(*ref.parameter_cases())
    def test_the_bound_does_not_empty_the_gate(self, name: str, **params: Any) -> None:
        """A cap that ate the set would look exactly like a cheaper gate.

        So the count is asserted rather than assumed, and against the published
        total rather than against itself: `_GATE_VECTORS` records reach the
        harness, they are the shortest ones, and there are more where they came
        from — which is what says the sweep next door has something left to do.
        """
        del params
        published = falcon_vectors.records(name)
        gated = falcon_vectors.vectors(name, limit=_GATE_VECTORS, sign_limit=0)

        self.assertLen(gated, _GATE_VECTORS)
        self.assertLess(_GATE_VECTORS, len(published), "the bound is not a bound")
        self.assertEqual(
            [vector.case_id for vector in gated],
            [
                vector.case_id
                for vector in falcon_vectors.vectors(name, limit=None, sign_limit=0)[
                    :_GATE_VECTORS
                ]
            ],
        )

    @parameterized.parameters(*ref.parameter_cases())
    def test_the_reference_accepts_every_published_case(
        self, name: str, **params: Any
    ) -> None:
        """The loaded set is evidence only if it reproduces upstream's verdict.

        These bytes were regrouped out of §3.11.6's aggregate into §3.11.3's
        padded signature, so this is what says the regrouping put every field
        where the standard says it goes — independently of the implementation
        the case is meant to gate.

        Unbounded, where the harness pass above is not. `_GATE_VECTORS` bounds
        *traced shapes* and this compiles nothing, so slicing it would apply a
        budget to work the budget does not cover — and the sweep would then have
        to carry a copy of this method to reach the rest.
        """
        del params
        for record in falcon_vectors.records(name):
            self.assertTrue(
                ref.verify(record.public_key, record.message, record.signature, name),
                f"count={record.case} is not accepted by the reference",
            )

    @parameterized.parameters(*ref.parameter_cases())
    def test_the_records_without_an_encoding_are_the_ones_stated(
        self, name: str, **params: Any
    ) -> None:
        """§3.11.3 is fixed-width and the NIST API's signature is not.

        The generator drives the raw signing call, which carries no equivalent
        of Algorithm 10's restart when `enc_s` compresses past `sbytelen - 41`,
        so a published record can have no §3.11.3 form at all. Pinning which
        ones is what stops a regenerated source from quietly dropping more.
        """
        del params
        self.assertEqual(falcon_vectors.unencodable(name), _UNENCODABLE[name])


if __name__ == "__main__":
    absltest.main()
