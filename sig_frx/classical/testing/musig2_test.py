# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""MuSig2 against BIP-327's published vectors, one stage at a time.

BIP-327 publishes a file per protocol stage, and this test gates them in the
order the protocol runs them. Key aggregation comes first because every later
stage consumes its output: a wrong aggregate key makes the challenge wrong,
which makes every partial signature wrong, and a gate that only checked the
final signature would report all of that as one failure.

`key_agg_vectors.json` carries the tweak refusals alongside the aggregations,
so the whole file is run here rather than split: a vector the harness cannot
run is an error, not a skip.

The error cases are half the gate and they say more than "reject". An
`invalid_contribution` case names the *signer* who sent the bad key, because a
coordinator that only learns the ceremony failed cannot exclude anyone and has
to start over — so the exception carries the index, and these tests assert it
rather than just asserting a raise.
"""

from __future__ import annotations

import json
from typing import Any
from unittest import mock

from absl.testing import absltest
from python.runfiles import Runfiles

from sig_frx.classical import secp
from sig_frx.classical.schnorr import musig2

_RUNFILES = Runfiles.Create()


def _load(repo: str, file_name: str) -> dict[str, Any]:
    path = _RUNFILES.Rlocation(f"{repo}/file/{file_name}")
    assert path is not None
    with open(path, "rb") as handle:
        return json.load(handle)


def _aggregate(data: dict[str, Any], case: dict[str, Any]) -> Any:
    """A case's `key_indices` and `tweak_indices` applied in published order."""
    pubkeys = [bytes.fromhex(data["pubkeys"][i]) for i in case["key_indices"]]
    context = musig2.key_agg(pubkeys)
    tweaks = case.get("tweak_indices", [])
    xonly_flags = case.get("is_xonly", [])
    for index, is_xonly in zip(tweaks, xonly_flags, strict=True):
        context = context.apply_tweak(bytes.fromhex(data["tweaks"][index]), is_xonly)
    return context


class KeyAggTest(absltest.TestCase):
    """BIP-327 `key_agg_vectors.json`, every case."""

    def setUp(self) -> None:
        super().setUp()
        self.data = _load("bip327_key_agg_vectors", "key_agg_vectors.json")

    def test_the_published_aggregations(self) -> None:
        cases = self.data["valid_test_cases"]
        self.assertNotEmpty(cases)
        for index, case in enumerate(cases):
            with self.subTest(case=index, keys=case["key_indices"]):
                context = _aggregate(self.data, case)
                self.assertEqual(
                    context.xonly_bytes().hex().upper(), case["expected"].upper()
                )

    def test_key_order_changes_the_aggregate(self) -> None:
        """Not a permutation-invariant sum: the coefficients bind the order."""
        forward, reversed_ = self.data["valid_test_cases"][0:2]
        self.assertEqual(
            sorted(forward["key_indices"]), sorted(reversed_["key_indices"])
        )
        self.assertNotEqual(forward["expected"], reversed_["expected"])

    def test_an_invalid_contribution_names_its_signer(self) -> None:
        cases = [
            case
            for case in self.data["error_test_cases"]
            if case["error"]["type"] == "invalid_contribution"
        ]
        self.assertNotEmpty(cases)
        for index, case in enumerate(cases):
            with self.subTest(case=index, comment=case["comment"]):
                with self.assertRaises(musig2.InvalidContributionError) as caught:
                    _aggregate(self.data, case)
                self.assertEqual(caught.exception.signer, case["error"]["signer"])
                self.assertEqual(caught.exception.contrib, case["error"]["contrib"])

    def test_the_named_signer_follows_the_bad_key_s_position(self) -> None:
        """The published cases pin one position each; the index has to move.

        Each `invalid_contribution` vector fixes the bad key at one index, and
        for the malformed-encoding case that index is 0 — so an implementation
        that reported signer 0 unconditionally would pass every published case.
        The keys here are the published bad ones, moved: no case is invented,
        only its position, which is the part the vectors cannot vary.
        """
        good = self.data["pubkeys"][0]
        for bad_index in (3, 4, 5):
            bad = self.data["pubkeys"][bad_index]
            for position in (0, 1, 2):
                keys = [good, good, good]
                keys[position] = bad
                with self.subTest(bad_key=bad_index, position=position):
                    with self.assertRaises(musig2.InvalidContributionError) as caught:
                        musig2.key_agg([bytes.fromhex(k) for k in keys])
                    self.assertEqual(caught.exception.signer, position)
                    self.assertEqual(caught.exception.contrib, "pubkey")

    def test_a_refused_tweak_is_a_value_error(self) -> None:
        cases = [
            case
            for case in self.data["error_test_cases"]
            if case["error"]["type"] == "value"
        ]
        self.assertNotEmpty(cases)
        for index, case in enumerate(cases):
            with self.subTest(case=index, comment=case["comment"]):
                with self.assertRaises(ValueError):
                    _aggregate(self.data, case)

    def test_a_cosigner_count_past_the_device_threshold_agrees_with_the_host(
        self,
    ) -> None:
        """The published cases top out at four keys; the batch seam moves at 64.

        `secp` places a batch on the device once it reaches `DEVICE_MIN_BATCH`,
        so every published vector aggregates below that line and none of them
        can say whether the placed path agrees. Raising the threshold past the
        batch runs the identical call on the host, which makes the comparison a
        differential on placement alone rather than a second implementation of
        the aggregation.

        This is the shape `secp._place` warns about — a threshold the gate's
        own batches never cross — and a ceremony is where it gets crossed.
        """
        pubkeys = [
            bytes.fromhex(self.data["pubkeys"][i % 3])
            for i in range(secp.DEVICE_MIN_BATCH + 1)
        ]
        placed = musig2.key_agg(pubkeys).xonly_bytes()

        # A threshold no batch can reach keeps the identical call on the host,
        # so the two differ in where the point sum ran and in nothing else.
        with mock.patch.object(secp, "DEVICE_MIN_BATCH", len(pubkeys) + 1):
            on_host = musig2.key_agg(pubkeys).xonly_bytes()

        self.assertEqual(placed, on_host)

    def test_every_published_case_runs(self) -> None:
        """The counts the file ships with, so a regenerated set fails loudly."""
        self.assertLen(self.data["valid_test_cases"], 4)
        self.assertLen(self.data["error_test_cases"], 5)


if __name__ == "__main__":
    absltest.main()
