# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Each Ed25519 construction against the vectors built to make rules differ.

The assertion is per-case and per-rule: for all 914 published cases, the
verdict equals what that construction's authority says it should be
([`ed25519_cctv_vectors.py`](ed25519_cctv_vectors.py) states the accept sets
and where each comes from). A rule that quietly became a different rule
fails here as a list of case numbers rather than as a count.

Three properties ride along, and each is one the per-case comparison cannot
see on its own:

- **The published RFC 8032 signatures are accepted by all three.** The rules
  only ever move signatures no honest signer emits; a construction that
  started rejecting §7.1 has stopped being Ed25519 rather than become
  stricter.
- **The accept sets nest.** `verify_strict` ⊆ RFC 8032 ⊆ ZIP-215, asserted
  over the verdicts rather than over the predicates, so a rule wired to the
  wrong axis shows up as a set that stopped nesting.
- **A verdict is the case's own.** Verifying one case alone reproduces its
  verdict from the full batch, which is what a `verify` that reduced over
  the batch axis would fail.

Then the property ZIP-215 exists for: its *aggregate* check — one verdict
for a whole batch, from a random linear combination — agrees with the
per-signature one on every published case. Every case ZIP-215 accepts
aggregates to `True` in one call, and adding any case it rejects turns that
call `False`. This is the claim the cofactored equation is mandated for,
and the vectors are where a cofactorless aggregate would part company with
its own `verify`.
"""

from __future__ import annotations

import functools

import numpy as np
from absl.testing import absltest, parameterized

from sig_frx.classical.eddsa import consensus, ed25519
from sig_frx.classical.testing import ed25519_cctv_vectors as cctv
from sig_frx.signature import Signature

# RFC 8032 §7.1 TEST 3 — an honestly produced signature, which every rule
# here accepts. Kept as its own copy rather than imported from `eddsa_test`:
# a test that shares its fixture with the suite it is checking against can
# be made green by editing the fixture.
_RFC_8032_KEY = "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025"
_RFC_8032_MESSAGE = "af82"
_RFC_8032_SIGNATURE = (
    "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac"
    "18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"
)


@functools.cache
def _verdicts(name: str) -> dict[int, bool]:
    """Every published case's verdict under `name`, keyed by case number.

    One `verify` call per message length — the whole set, in the batches its
    own varying message lengths cut it into.
    """
    scheme = cctv.RULESETS[name].scheme
    verdicts = {}
    for batch in cctv.by_message_length():
        public_key, message, signature = cctv.as_arrays(batch)
        got = np.asarray(scheme.verify(public_key, message, signature, context=None))
        for vector, verdict in zip(batch, got):
            verdicts[vector.number] = bool(verdict)
    return verdicts


def _hex(value: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(value), dtype=np.uint8)


class AcceptSetTest(parameterized.TestCase):
    def test_the_pinned_set_is_the_one_that_was_measured(self) -> None:
        self.assertLen(cctv.load(), cctv.TOTAL)

    @parameterized.named_parameters(*(("_" + name, name) for name in cctv.RULESETS))
    def test_matches_its_specification_case_by_case(self, name: str) -> None:
        ruleset = cctv.RULESETS[name]
        verdicts = _verdicts(name)
        wrong = [
            f"#{v.number} got={verdicts[v.number]} flags={sorted(v.flags)}"
            for v in cctv.load()
            if verdicts[v.number] != ruleset.accepts(v.flags)
        ]
        self.assertEmpty(wrong, f"{name} does not implement its rule: {wrong[:5]}")

    @parameterized.named_parameters(*(("_" + name, name) for name in cctv.RULESETS))
    def test_accepts_the_pinned_number_of_cases(self, name: str) -> None:
        # The predicate above is a statement about flags; this is a statement
        # about the population the flags were drawn over. A regenerated set
        # that reshuffles the draw satisfies the first and fails this one.
        accepted = sum(_verdicts(name).values())
        self.assertEqual(accepted, cctv.RULESETS[name].expected_accepts)

    @parameterized.named_parameters(*(("_" + name, name) for name in cctv.RULESETS))
    def test_accepts_the_published_rfc_8032_signature(self, name: str) -> None:
        scheme: Signature = cctv.RULESETS[name].scheme
        verdict = scheme.verify(
            _hex(_RFC_8032_KEY)[None, :],
            _hex(_RFC_8032_MESSAGE)[None, :],
            _hex(_RFC_8032_SIGNATURE)[None, :],
            context=None,
        )
        self.assertTrue(bool(np.asarray(verdict)[0]))

    def test_the_accept_sets_nest(self) -> None:
        strict = _verdicts("Ed25519Strict")
        rfc = _verdicts("Ed25519")
        zip215 = _verdicts("Ed25519Zip215")
        for vector in cctv.load():
            number, flags = vector.number, sorted(vector.flags)
            if strict[number]:
                self.assertTrue(rfc[number], f"#{number} {flags}: strict ⊄ RFC 8032")
            if rfc[number]:
                self.assertTrue(
                    zip215[number], f"#{number} {flags}: RFC 8032 ⊄ ZIP-215"
                )


class BatchAxisTest(absltest.TestCase):
    def test_one_at_a_time_agrees_with_the_batch_under_zip_215(self) -> None:
        scheme = cctv.RULESETS["Ed25519Zip215"].scheme
        batched = _verdicts("Ed25519Zip215")
        for vector in cctv.load():
            alone = scheme.verify(
                np.frombuffer(vector.public_key, dtype=np.uint8)[None, :],
                np.frombuffer(vector.message, dtype=np.uint8)[None, :],
                np.frombuffer(vector.signature, dtype=np.uint8)[None, :],
                context=None,
            )
            self.assertEqual(
                bool(np.asarray(alone)[0]),
                batched[vector.number],
                f"#{vector.number}: alone and in a batch disagree",
            )


class AggregateAgreesWithVerifyTest(absltest.TestCase):
    """ZIP-215's own claim: the aggregate accepts exactly what `verify` does."""

    def setUp(self) -> None:
        super().setUp()
        self.scheme = consensus.Ed25519Zip215()
        self.verdicts = _verdicts("Ed25519Zip215")

    def _aggregate(self, batch: tuple[cctv.CctvVector, ...]) -> bool:
        return self.scheme.aggregate_verify(*cctv.as_arrays(batch))

    def test_every_accepted_case_aggregates_to_one_true(self) -> None:
        checked = 0
        for batch in cctv.by_message_length():
            accepted = tuple(v for v in batch if self.verdicts[v.number])
            self.assertNotEmpty(accepted)
            self.assertTrue(
                self._aggregate(accepted),
                f"{len(accepted)} individually accepted cases failed together",
            )
            checked += len(accepted)
        self.assertEqual(checked, cctv.RULESETS["Ed25519Zip215"].expected_accepts)

    def test_any_rejected_case_fails_the_aggregate(self) -> None:
        # Each rejected case rides with accepted ones of its own message
        # length: a residual that survives the combination is what the
        # aggregate has to catch, and burying it among valid rows is the
        # case that a sloppy combination would let through.
        accepted_by_length = {
            len(batch[0].message): tuple(v for v in batch if self.verdicts[v.number])[
                :4
            ]
            for batch in cctv.by_message_length()
        }
        rejected = [v for v in cctv.load() if not self.verdicts[v.number]]
        self.assertNotEmpty(rejected)
        for vector in rejected:
            company = tuple(
                v
                for v in accepted_by_length[len(vector.message)]
                if v.number != vector.number
            )[:3]
            self.assertFalse(
                self._aggregate((vector,) + company),
                f"#{vector.number}: verify rejects it, the aggregate does not",
            )

    def test_the_aggregate_is_not_offered_where_it_would_disagree(self) -> None:
        # The method's absence on the cofactorless constructions is the
        # design, not an oversight — see `consensus.py`.
        self.assertFalse(hasattr(ed25519.Ed25519(), "aggregate_verify"))
        self.assertFalse(hasattr(consensus.Ed25519Strict(), "aggregate_verify"))


if __name__ == "__main__":
    absltest.main()
