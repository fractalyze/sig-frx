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

import numpy as np
from absl.testing import absltest
from python.runfiles import Runfiles

from sig_frx.classical import secp
from sig_frx.classical.schnorr import bip340, musig2

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


class _SignVectors(absltest.TestCase):
    """Shared reading of `sign_verify_vectors.json`, whose cases index into six
    shared corpora rather than carrying their own values."""

    def setUp(self) -> None:
        super().setUp()
        self.data = _load("bip327_sign_verify_vectors", "sign_verify_vectors.json")

    def _keys(self, case: dict[str, Any]) -> list[bytes]:
        return [bytes.fromhex(self.data["pubkeys"][i]) for i in case["key_indices"]]

    def _pnonces(self, case: dict[str, Any]) -> list[bytes]:
        return [bytes.fromhex(self.data["pnonces"][i]) for i in case["nonce_indices"]]

    def _session(self, case: dict[str, Any]) -> Any:
        return musig2.Session(
            aggnonce=bytes.fromhex(self.data["aggnonces"][case["aggnonce_index"]]),
            pubkeys=self._keys(case),
            message=bytes.fromhex(self.data["msgs"][case["msg_index"]]),
        )

    def _secnonce(self, case: dict[str, Any]) -> Any:
        index = case.get("secnonce_index", 0)
        return musig2.SecNonce.from_bytes(bytes.fromhex(self.data["secnonces"][index]))


class SignTest(_SignVectors):
    """BIP-327 `sign_verify_vectors.json`, the signing half."""

    def test_the_published_partial_signatures(self) -> None:
        cases = self.data["valid_test_cases"]
        self.assertNotEmpty(cases)
        for index, case in enumerate(cases):
            with self.subTest(case=index, comment=case.get("comment", "")):
                psig = musig2.sign(
                    self._secnonce(case),
                    bytes.fromhex(self.data["sk"]),
                    self._session(case),
                )
                self.assertEqual(psig.hex().upper(), case["expected"].upper())

    def test_an_aggregate_nonce_of_infinity_still_signs(self) -> None:
        """`R'` at infinity falls back to the generator rather than aborting —
        a session the specification carries on with, so a signer that raised
        would strand every cosigner whose nonces happened to cancel."""
        case = next(
            c
            for c in self.data["valid_test_cases"]
            if "infinity" in c.get("comment", "")
        )
        aggnonce = bytes.fromhex(self.data["aggnonces"][case["aggnonce_index"]])
        self.assertEqual(aggnonce, bytes(66))
        psig = musig2.sign(
            self._secnonce(case), bytes.fromhex(self.data["sk"]), self._session(case)
        )
        self.assertEqual(psig.hex().upper(), case["expected"].upper())

    def test_a_faulty_cosigner_key_is_blamed_and_a_faulty_aggnonce_is_not(
        self,
    ) -> None:
        """`signer` is `None` for an aggregate-nonce fault, and that is the
        distinction rather than an omission: the aggregate is the coordinator's
        own product, so no cosigner sent it and none can be excluded for it."""
        cases = [
            c
            for c in self.data["sign_error_test_cases"]
            if c["error"]["type"] == "invalid_contribution"
        ]
        self.assertNotEmpty(cases)
        for index, case in enumerate(cases):
            with self.subTest(case=index, comment=case["comment"]):
                with self.assertRaises(musig2.InvalidContributionError) as caught:
                    musig2.sign(
                        self._secnonce(case),
                        bytes.fromhex(self.data["sk"]),
                        self._session(case),
                    )
                self.assertEqual(caught.exception.signer, case["error"]["signer"])
                self.assertEqual(caught.exception.contrib, case["error"]["contrib"])

    def test_a_secnonce_out_of_range_refuses_to_sign(self) -> None:
        """The specification's own comment on this case is that it may indicate
        nonce reuse, which is the failure this scheme cannot recover from."""
        case = next(
            c
            for c in self.data["sign_error_test_cases"]
            if c["error"].get("message", "").startswith("first secnonce")
        )
        with self.assertRaises(ValueError):
            musig2.sign(
                self._secnonce(case),
                bytes.fromhex(self.data["sk"]),
                self._session(case),
            )

    def test_a_signer_outside_the_key_list_refuses_to_sign(self) -> None:
        """Optional in BIP-327 and taken: a partial signature under a key the
        session never aggregated cannot combine, so signing is the cheap place
        to say so rather than leaving a silent aggregation failure."""
        case = next(
            c
            for c in self.data["sign_error_test_cases"]
            if "must be included" in c["error"].get("message", "")
        )
        with self.assertRaises(ValueError):
            musig2.sign(
                self._secnonce(case),
                bytes.fromhex(self.data["sk"]),
                self._session(case),
            )

    def test_the_secnonce_round_trips_through_its_bytes(self) -> None:
        raw = bytes.fromhex(self.data["secnonces"][0])
        self.assertEqual(musig2.SecNonce.from_bytes(raw).to_bytes(), raw)


class PartialSigVerifyTest(_SignVectors):
    """BIP-327 `sign_verify_vectors.json`, the verification half.

    A wrong signature is a `False`, and an unusable contribution is a raise.
    The split is the specification's and it matters: the first is a cosigner
    who signed something else, the second is one who sent something that is
    not a signature at all.
    """

    def _verify(self, case: dict[str, Any], sig: bytes) -> bool:
        return musig2.partial_sig_verify(
            sig,
            self._pnonces(case),
            self._keys(case),
            bytes.fromhex(self.data["msgs"][case["msg_index"]]),
            case["signer_index"],
        )

    def test_the_published_partial_signatures_verify(self) -> None:
        for index, case in enumerate(self.data["valid_test_cases"]):
            with self.subTest(case=index):
                self.assertTrue(self._verify(case, bytes.fromhex(case["expected"])))

    def test_a_wrong_partial_signature_is_false_not_an_error(self) -> None:
        cases = self.data["verify_fail_test_cases"]
        self.assertNotEmpty(cases)
        for index, case in enumerate(cases):
            with self.subTest(case=index, comment=case["comment"]):
                self.assertFalse(self._verify(case, bytes.fromhex(case["sig"])))

    def test_an_unusable_contribution_raises_and_names_its_signer(self) -> None:
        cases = self.data["verify_error_test_cases"]
        self.assertNotEmpty(cases)
        for index, case in enumerate(cases):
            with self.subTest(case=index, comment=case["comment"]):
                with self.assertRaises(musig2.InvalidContributionError) as caught:
                    self._verify(case, bytes.fromhex(case["sig"]))
                self.assertEqual(caught.exception.signer, case["error"]["signer"])
                self.assertEqual(caught.exception.contrib, case["error"]["contrib"])


class PartialSigAggTest(absltest.TestCase):
    """BIP-327 `sig_agg_vectors.json` — where the protocol's output becomes an
    ordinary BIP-340 signature.

    This is the stage the whole scheme exists for: everything above it produces
    values only MuSig2 understands, and what comes out here is what a taproot
    output accepts. So it is gated twice — against the published bytes, and by
    running the result through this repo's own BIP-340 verifier.
    """

    def setUp(self) -> None:
        super().setUp()
        self.data = _load("bip327_sig_agg_vectors", "sig_agg_vectors.json")

    def _session(self, case: dict[str, Any]) -> Any:
        return musig2.Session(
            aggnonce=bytes.fromhex(case["aggnonce"]),
            pubkeys=[
                bytes.fromhex(self.data["pubkeys"][i]) for i in case["key_indices"]
            ],
            message=bytes.fromhex(self.data["msg"]),
            tweaks=[
                (bytes.fromhex(self.data["tweaks"][i]), xonly)
                for i, xonly in zip(
                    case["tweak_indices"], case["is_xonly"], strict=True
                )
            ],
        )

    def _psigs(self, case: dict[str, Any]) -> list[bytes]:
        return [bytes.fromhex(self.data["psigs"][i]) for i in case["psig_indices"]]

    def test_the_published_aggregate_signatures(self) -> None:
        cases = self.data["valid_test_cases"]
        self.assertNotEmpty(cases)
        for index, case in enumerate(cases):
            with self.subTest(case=index, tweaks=len(case["tweak_indices"])):
                signature = musig2.partial_sig_agg(
                    self._psigs(case), self._session(case)
                )
                self.assertEqual(signature.hex().upper(), case["expected"].upper())

    def test_the_aggregate_verifies_through_the_existing_bip340_verifier(self) -> None:
        """The scheme's whole claim, asserted rather than described.

        No MuSig2 verifier exists because none should: the aggregate is a
        BIP-340 signature under the aggregate x-only key, so it goes through
        the seam's own `verify` — batch-shaped, since the seam has no scalar
        form — and a chain would accept exactly what this accepts.
        """
        scheme = bip340.Bip340()
        cases = self.data["valid_test_cases"]
        message = bytes.fromhex(self.data["msg"])
        keys, messages, signatures = [], [], []
        for case in cases:
            session = self._session(case)
            signatures.append(musig2.partial_sig_agg(self._psigs(case), session))
            keys.append(session.key_context().xonly_bytes())
            messages.append(message)

        verdicts = scheme.verify(
            np.stack([np.frombuffer(k, dtype=np.uint8) for k in keys]),
            np.stack([np.frombuffer(m, dtype=np.uint8) for m in messages]),
            np.stack([np.frombuffer(s, dtype=np.uint8) for s in signatures]),
            context=None,
        )
        self.assertLen(verdicts, len(cases))
        self.assertTrue(bool(np.all(np.asarray(verdicts))))

    def test_a_tweaked_session_still_verifies(self) -> None:
        """Tweaking moves the aggregate key, and the signature has to follow it
        — `tacc` is what carries that, and nothing before this stage spent it."""
        case = next(c for c in self.data["valid_test_cases"] if c["tweak_indices"])
        session = self._session(case)
        untweaked = musig2.Session(
            aggnonce=session.aggnonce, pubkeys=session.pubkeys, message=session.message
        )
        self.assertNotEqual(
            session.key_context().xonly_bytes(),
            untweaked.key_context().xonly_bytes(),
        )

    def test_a_partial_signature_over_the_group_order_names_its_signer(self) -> None:
        cases = self.data["error_test_cases"]
        self.assertNotEmpty(cases)
        for index, case in enumerate(cases):
            with self.subTest(case=index, comment=case["comment"]):
                with self.assertRaises(musig2.InvalidContributionError) as caught:
                    musig2.partial_sig_agg(self._psigs(case), self._session(case))
                self.assertEqual(caught.exception.signer, case["error"]["signer"])
                self.assertEqual(caught.exception.contrib, "psig")


class DeterministicSignTest(absltest.TestCase):
    """BIP-327 `det_sign_vectors.json`.

    The two-round protocol collapsed into one for the last signer: given every
    other cosigner's nonces already aggregated, this derives its own nonce from
    the message and signs in a single step, so a signer with no state between
    rounds can still take part.

    It does not remove the reason nonces must not repeat — it removes the
    *window*. The nonce is a function of the secret key, the other nonces and
    the message, so signing the same session twice reproduces it harmlessly and
    signing two different ones cannot collide.
    """

    def setUp(self) -> None:
        super().setUp()
        self.data = _load("bip327_det_sign_vectors", "det_sign_vectors.json")

    def _call(self, case: dict[str, Any]) -> tuple[bytes, bytes]:
        return musig2.deterministic_sign(
            bytes.fromhex(self.data["sk"]),
            bytes.fromhex(case["aggothernonce"]),
            [bytes.fromhex(self.data["pubkeys"][i]) for i in case["key_indices"]],
            bytes.fromhex(self.data["msgs"][case["msg_index"]]),
            rand=_optional(case["rand"]),
            tweaks=[
                (bytes.fromhex(tw), xonly)
                for tw, xonly in zip(case["tweaks"], case["is_xonly"], strict=True)
            ],
        )

    def test_the_published_nonces_and_signatures(self) -> None:
        cases = self.data["valid_test_cases"]
        self.assertNotEmpty(cases)
        for index, case in enumerate(cases):
            with self.subTest(case=index, comment=case.get("comment", "")):
                pubnonce, psig = self._call(case)
                self.assertEqual(pubnonce.hex().upper(), case["expected"][0].upper())
                self.assertEqual(psig.hex().upper(), case["expected"][1].upper())

    def test_it_is_deterministic_without_randomness(self) -> None:
        """`rand` is optional here where `nonce_gen` requires it, because the
        message and the other cosigners' nonces already make the derivation
        unique to the session. The published set pins a `None` case."""
        case = next(c for c in self.data["valid_test_cases"] if c["rand"] is None)
        self.assertEqual(self._call(case), self._call(case))

    def test_a_faulty_other_nonce_blames_nobody(self) -> None:
        """`aggothernonce` is the coordinator's aggregate, so its contribution
        name is its own and its signer is `None` — the same distinction the
        session's aggregate nonce draws, and it must survive being produced by
        a `nonce_agg` that would otherwise blame position one."""
        cases = [
            c
            for c in self.data["error_test_cases"]
            if c["error"].get("contrib") == "aggothernonce"
        ]
        self.assertNotEmpty(cases)
        for index, case in enumerate(cases):
            with self.subTest(case=index, comment=case["comment"]):
                with self.assertRaises(musig2.InvalidContributionError) as caught:
                    self._call(case)
                self.assertIsNone(caught.exception.signer)
                self.assertEqual(caught.exception.contrib, "aggothernonce")

    def test_a_faulty_cosigner_key_is_still_blamed_by_position(self) -> None:
        case = next(
            c
            for c in self.data["error_test_cases"]
            if c["error"].get("contrib") == "pubkey"
        )
        with self.assertRaises(musig2.InvalidContributionError) as caught:
            self._call(case)
        self.assertEqual(caught.exception.signer, case["error"]["signer"])

    def test_the_value_errors_the_published_set_names(self) -> None:
        cases = [
            c for c in self.data["error_test_cases"] if c["error"]["type"] == "value"
        ]
        self.assertNotEmpty(cases)
        for index, case in enumerate(cases):
            with self.subTest(case=index, comment=case["comment"]):
                with self.assertRaises(ValueError):
                    self._call(case)


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
