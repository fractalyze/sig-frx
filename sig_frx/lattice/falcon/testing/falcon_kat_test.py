# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Falcon against the round-3 vectors, through the shared harness.

[`kat.check`](../../../testing/kat.py) owns three passes and Falcon owns none of
them: the published verdicts, the tampering that moves a bit in each of the three
inputs, and the batch axis built by replicating an accepted case and corrupting
some entries. `testing.md` puts them there rather than in a scheme's own
tests "because the gap is a property of how the vectors are published, not of any
one scheme — a per-scheme fix would be written once per scheme for one cause",
and a Falcon-shaped copy would be the third.

**Key generation and signing stay uncovered by this set, and will** — now that
both are implemented, which is when the reason has to stop being "they raise".
It is not that the vectors are missing them: every record carries `sk`, and a
`seed` besides. It is that **neither output is reproducible here, by decision**.

- `_check_keygen` would require `keygen(seed)` to return the published pair, and
  the seed expands through NIST's AES-256-CTR-DRBG rather than this repo's
  `SHAKE256(seed ‖ attempt)` ([#26](https://github.com/fractalyze/sig-frx/issues/26)).
- `_check_sign` compares byte for byte, and §3.9 draws a salt per signature — so
  a correct randomized signer disagrees with the published bytes by
  construction ([#27](https://github.com/fractalyze/sig-frx/issues/27)).

So a `seed` or a `secret_key` on a record would not extend this gate; it would
fail a correct implementation. [`falcon_vectors`](falcon_vectors.py) withholds
both and says why, and `test_the_two_secrets_stay_off_the_record` below is what
keeps the withholding deliberate — "the harness skipped it" and "the harness
cannot check it" read identically from a green run, so the skip is asserted
rather than left to the absence of a field.

Where signing *is* gated, since it is not gated here: a round trip under this
published key pair in [`falcon_test`](falcon_test.py), and the reference
implementation's verdict on signatures produced here in
[`falcon_interop_test`](falcon_interop_test.py).

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
        kat.check(falcon.named(name), falcon_vectors.vectors(name, limit=_GATE_VECTORS))

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
        gated = falcon_vectors.vectors(name, limit=_GATE_VECTORS)

        self.assertLen(gated, _GATE_VECTORS)
        self.assertLess(_GATE_VECTORS, len(published), "the bound is not a bound")
        self.assertEqual(
            [vector.case_id for vector in gated],
            [
                vector.case_id
                for vector in falcon_vectors.vectors(name, limit=None)[:_GATE_VECTORS]
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
    def test_the_two_secrets_stay_off_the_record(
        self, name: str, **params: Any
    ) -> None:
        """The skip above, asserted rather than inherited from a missing field.

        `_check_keygen` and `_check_sign` both continue past a vector that
        carries no `seed` / no `secret_key`, so Falcon's key generation and
        signing are skipped by *omission* — and an omission looks exactly like
        coverage from a green run. The published set carries both secrets, so
        putting either on the record is a one-word change that would silently
        convert this gate into one that fails a correct implementation.

        Asserted against the loader's own `Record`, which does carry `sk`, so
        this cannot pass by the vectors having lost the data instead.
        """
        del params
        published = falcon_vectors.records(name)
        self.assertNotEmpty(published)
        self.assertTrue(
            all(record.secret_key for record in published),
            "the loader stopped carrying `sk`, so the assertion below is vacuous",
        )
        for vector in falcon_vectors.vectors(name, _GATE_VECTORS):
            with self.subTest(case=vector.case_id):
                self.assertIsNone(
                    vector.secret_key,
                    "a `secret_key` here drives `_check_sign`, which compares "
                    "byte for byte against a signature Falcon's randomized "
                    "signing cannot reproduce",
                )
                self.assertIsNone(
                    vector.seed,
                    "a `seed` here drives `_check_keygen`, which requires the "
                    "published pair back out of an expansion this repo does "
                    "not implement",
                )

    @parameterized.parameters(*ref.parameter_cases())
    def test_the_published_secret_key_is_reachable_and_gated_elsewhere(
        self, name: str, **params: Any
    ) -> None:
        """Withheld from the record is not withheld from the suite.

        The bytes the record does not carry are the ones `keygen_test` gates
        §3.11.5's decoder on and `falcon_test` signs under, so this pins that
        the accessor keeps working — a withholding that quietly became a
        deletion would take those gates with it.
        """
        secret = falcon_vectors.secret_key(name)
        self.assertLen(secret, falcon.named(name).secret_key_size)
        self.assertEqual(secret, falcon_vectors.records(name)[0].secret_key)

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
