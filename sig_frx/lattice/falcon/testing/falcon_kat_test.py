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
out.

**Key generation and signing stay uncovered by this set, and will.** The reason
is not that the vectors are missing them — every record carries `sk`, and a
`seed` besides — but that neither is reproducible here: the seed expands through
NIST's AES-256-CTR-DRBG rather than this repo's `SHAKE256(seed ‖ attempt)`, and
a `secret_key` on a record drives the harness into a `sign` that #27 has not
finished. [`falcon_vectors`](falcon_vectors.py) withholds both fields for those
two reasons and says so; this is where the consequence is recorded, because "the
harness skipped it" and "the harness cannot check it" read identically from a
green run.

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

# The vectors per degree this gate takes, out of the 100 published. Chosen to
# hold the target inside its bucket on the leg that decides it — the value is
# argued in //sig_frx/lattice/falcon/testing/BUILD.bazel next to the measurement
# it comes from, since it is a budget rather than a property of Falcon.
GATE_VECTORS = 4


class FalconKatTest(parameterized.TestCase):
    @parameterized.parameters(*ref.parameter_cases())
    def test_the_published_cases_and_what_the_harness_derives(
        self, name: str, **params: Any
    ) -> None:
        del params
        kat.check(falcon.named(name), falcon_vectors.vectors(name, limit=GATE_VECTORS))

    @parameterized.parameters(*ref.parameter_cases())
    def test_the_bound_does_not_empty_the_gate(self, name: str, **params: Any) -> None:
        """A cap that ate the set would look exactly like a cheaper gate.

        So the count is asserted rather than assumed, and against the published
        total rather than against itself: `GATE_VECTORS` records reach the
        harness, they are the shortest ones, and there are more where they came
        from — which is what says the sweep next door has something left to do.
        """
        del params
        published = falcon_vectors.records(name)
        gated = falcon_vectors.vectors(name, limit=GATE_VECTORS)

        self.assertLen(gated, GATE_VECTORS)
        self.assertLess(GATE_VECTORS, len(published), "the bound is not a bound")
        self.assertEqual(
            [vector.case_id for vector in gated],
            [
                f"falcon-round3 KAT {name} count={record.case}"
                for record in published[:GATE_VECTORS]
            ],
        )

    @parameterized.parameters(*ref.parameter_cases())
    def test_the_reference_accepts_every_gated_case(
        self, name: str, **params: Any
    ) -> None:
        """The loaded set is evidence only if it reproduces upstream's verdict.

        These bytes were regrouped out of §3.11.6's aggregate into §3.11.3's
        padded signature, so this is what says the regrouping put every field
        where the standard says it goes — independently of the implementation
        the case is meant to gate. It runs on the gated subset here and on all
        100 in the sweep, because it is host arithmetic and cheap either way;
        what makes it worth repeating there is that a later record could differ
        in a way the first few do not.
        """
        del params
        for record in falcon_vectors.records(name)[:GATE_VECTORS]:
            self.assertTrue(
                ref.verify(record.public_key, record.message, record.signature, name),
                f"count={record.case} is not accepted by the reference",
            )


if __name__ == "__main__":
    absltest.main()
