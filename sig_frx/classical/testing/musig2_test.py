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

import functools
import json
from typing import Any
from unittest import mock

from absl.testing import absltest
from python.runfiles import Runfiles

from sig_frx.classical import secp
from sig_frx.classical.schnorr import musig2

_RUNFILES = Runfiles.Create()


@functools.cache
def _load(repo: str, file_name: str) -> dict[str, Any]:
    path = _RUNFILES.Rlocation(f"{repo}/file/{file_name}")
    assert path is not None
    with open(path, "rb") as handle:
        return json.load(handle)


def _aggregate(data: dict[str, Any], case: dict[str, Any]) -> musig2.KeyAggContext:
    """A case's `key_indices` and `tweak_indices` applied in published order."""
    pubkeys = [bytes.fromhex(data["pubkeys"][i]) for i in case["key_indices"]]
    context = musig2.key_agg(pubkeys)
    for index, is_xonly in zip(
        case.get("tweak_indices", []), case.get("is_xonly", []), strict=True
    ):
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
        keys = [bytes.fromhex(k) for k in self.data["pubkeys"][:3]]
        self.assertNotEqual(
            musig2.key_agg(keys).xonly_bytes(),
            musig2.key_agg(keys[::-1]).xonly_bytes(),
        )

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

    def test_the_aggregate_does_not_depend_on_where_the_batch_ran(self) -> None:
        """Forced, not reached — and each side held against the published value.

        `secp` places a batch on the device from `DEVICE_MIN_BATCH`, and every
        published aggregation is far below it, so the placed path would ship
        untested. Moving the threshold out from under the call is how
        `secp_device_test` exercises the same seam.

        Both legs are compared to the vector's own `expected` rather than to
        each other, which is the stronger statement: two paths that agree while
        both being wrong pass a parity check and fail this one.
        """
        for threshold, where in ((1 << 30, "host"), (0, "device")):
            for index, case in enumerate(self.data["valid_test_cases"]):
                with self.subTest(ran_on=where, case=index):
                    with mock.patch.object(secp, "DEVICE_MIN_BATCH", threshold):
                        context = _aggregate(self.data, case)
                    self.assertEqual(
                        context.xonly_bytes().hex().upper(), case["expected"].upper()
                    )

    def test_every_published_case_runs(self) -> None:
        """The counts the file ships with, so a regenerated set fails loudly."""
        self.assertLen(self.data["valid_test_cases"], 4)
        self.assertLen(self.data["error_test_cases"], 5)


if __name__ == "__main__":
    absltest.main()
