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


class KeySortTest(absltest.TestCase):
    """BIP-327 `key_sort_vectors.json`.

    Sorting is defined on the serializations, not on the points: BIP-327 orders
    the 33-byte encodings lexicographically and never parses them. So an
    unparseable key sorts as happily as any other and is refused later, by
    `key_agg` — which is why this stage raises nothing of its own.
    """

    def setUp(self) -> None:
        super().setUp()
        self.data = _load("bip327_key_sort_vectors", "key_sort_vectors.json")

    def test_the_published_ordering(self) -> None:
        pubkeys = [bytes.fromhex(k) for k in self.data["pubkeys"]]
        expected = [bytes.fromhex(k) for k in self.data["sorted_pubkeys"]]
        self.assertEqual(musig2.key_sort(pubkeys), expected)

    def test_a_repeated_key_is_kept_rather_than_collapsed(self) -> None:
        """A duplicate is a distinct cosigner slot, so the count cannot move."""
        pubkeys = [bytes.fromhex(k) for k in self.data["pubkeys"]]
        self.assertLen(set(pubkeys), len(pubkeys) - 1)
        self.assertLen(musig2.key_sort(pubkeys), len(pubkeys))

    def test_the_order_is_decided_by_the_whole_encoding(self) -> None:
        """The set pins two keys that differ only in their final byte, so a
        comparison that stopped early would still pass the published case."""
        near = sorted(
            k for k in self.data["pubkeys"] if k.startswith("02DD308AFEC5777E")
        )
        self.assertLen(near, 3)
        self.assertNotEqual(near[0][-2:], near[-1][-2:])
        sorted_keys = musig2.key_sort([bytes.fromhex(k) for k in near[::-1]])
        self.assertEqual([k.hex().upper() for k in sorted_keys], near)


def _optional(value: str | None) -> bytes | None:
    """A vector's optional field. `None` and `""` are different inputs here:
    an absent message and a present empty one take different prefixes, and the
    published set pins both."""
    return None if value is None else bytes.fromhex(value)


class NonceGenTest(absltest.TestCase):
    """BIP-327 `nonce_gen_vectors.json`.

    Nonce generation is deterministic in `rand_`, which is what makes it
    gateable at all — the deployed call draws `rand_` fresh, and the published
    cases fix it so the derivation can be compared byte for byte.
    """

    def setUp(self) -> None:
        super().setUp()
        self.data = _load("bip327_nonce_gen_vectors", "nonce_gen_vectors.json")

    def _generate(self, case: dict[str, Any]) -> tuple[Any, bytes]:
        return musig2.nonce_gen(
            bytes.fromhex(case["rand_"]),
            bytes.fromhex(case["pk"]),
            secret_key=_optional(case["sk"]),
            aggregate_key=_optional(case["aggpk"]),
            message=_optional(case["msg"]),
            extra_input=_optional(case["extra_in"]),
        )

    def test_the_published_nonces(self) -> None:
        cases = self.data["test_cases"]
        self.assertNotEmpty(cases)
        for index, case in enumerate(cases):
            with self.subTest(case=index):
                secnonce, pubnonce = self._generate(case)
                self.assertEqual(
                    secnonce.to_bytes().hex().upper(),
                    case["expected_secnonce"].upper(),
                )
                self.assertEqual(
                    pubnonce.hex().upper(), case["expected_pubnonce"].upper()
                )

    def test_an_absent_message_is_not_an_empty_one(self) -> None:
        """The two take different prefixes, so a signer that conflated them
        would derive one signer's nonce for two different sessions."""
        case = self.data["test_cases"][0]
        rand = bytes.fromhex(case["rand_"])
        public_key = bytes.fromhex(case["pk"])
        absent = musig2.nonce_gen(rand, public_key, message=None)
        empty = musig2.nonce_gen(rand, public_key, message=b"")
        self.assertNotEqual(absent[1], empty[1])

    def test_the_secnonce_carries_the_key_it_was_drawn_for(self) -> None:
        """Its tail is the signer's own public key, which is what lets signing
        refuse a secnonce drawn for a different key rather than sign with it."""
        case = self.data["test_cases"][0]
        secnonce, _ = self._generate(case)
        self.assertEqual(secnonce.public_key.hex().upper(), case["pk"].upper())


class NonceAggTest(absltest.TestCase):
    """BIP-327 `nonce_agg_vectors.json`.

    The two halves of a pubnonce aggregate independently, and either can sum to
    the identity without the other doing so — which is a value the wire format
    has to carry rather than an error, so the published set pins it.
    """

    def setUp(self) -> None:
        super().setUp()
        self.data = _load("bip327_nonce_agg_vectors", "nonce_agg_vectors.json")

    def _nonces(self, case: dict[str, Any]) -> list[bytes]:
        return [bytes.fromhex(self.data["pnonces"][i]) for i in case["pnonce_indices"]]

    def test_the_published_aggregations(self) -> None:
        cases = self.data["valid_test_cases"]
        self.assertNotEmpty(cases)
        for index, case in enumerate(cases):
            with self.subTest(case=index, comment=case.get("comment", "")):
                self.assertEqual(
                    musig2.nonce_agg(self._nonces(case)).hex().upper(),
                    case["expected"].upper(),
                )

    def test_a_half_summing_to_the_identity_is_a_value_not_a_failure(self) -> None:
        """Serialized as 33 zero bytes, which no real point can occupy. A signer
        that raised here would abort a session the specification continues."""
        case = next(
            c
            for c in self.data["valid_test_cases"]
            if "infinity" in c.get("comment", "")
        )
        aggregate = musig2.nonce_agg(self._nonces(case))
        self.assertEqual(aggregate[33:], bytes(33))
        self.assertNotEqual(aggregate[:33], bytes(33))

    def test_an_invalid_contribution_names_its_signer(self) -> None:
        cases = self.data["error_test_cases"]
        self.assertNotEmpty(cases)
        for index, case in enumerate(cases):
            with self.subTest(case=index, comment=case["comment"]):
                with self.assertRaises(musig2.InvalidContributionError) as caught:
                    musig2.nonce_agg(self._nonces(case))
                self.assertEqual(caught.exception.signer, case["error"]["signer"])
                self.assertEqual(caught.exception.contrib, "pubnonce")

    def test_a_nonce_of_the_wrong_length_is_refused(self) -> None:
        """No published case is mis-sized, and a short one would be caught by
        the point parse anyway — so an over-long nonce is the only thing the
        length check alone rejects, and it would otherwise go untested."""
        good = bytes.fromhex(self.data["pnonces"][0])
        for position in (0, 1):
            for bad in (good + b"\x00", good[:-1]):
                nonces = [good, good]
                nonces[position] = bad
                with self.subTest(position=position, length=len(bad)):
                    with self.assertRaises(musig2.InvalidContributionError) as caught:
                        musig2.nonce_agg(nonces)
                    self.assertEqual(caught.exception.signer, position)
                    self.assertEqual(caught.exception.contrib, "pubnonce")

    def test_a_bad_nonce_is_named_at_whatever_position_it_sits(self) -> None:
        """As with key aggregation, each published case fixes one position, so
        the index has to be shown to follow the nonce rather than be a constant."""
        good = bytes.fromhex(self.data["pnonces"][0])
        for bad_index in (4, 5, 6):
            bad = bytes.fromhex(self.data["pnonces"][bad_index])
            for position in (0, 1, 2):
                nonces = [good, good, good]
                nonces[position] = bad
                with self.subTest(bad_nonce=bad_index, position=position):
                    with self.assertRaises(musig2.InvalidContributionError) as caught:
                        musig2.nonce_agg(nonces)
                    self.assertEqual(caught.exception.signer, position)


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
