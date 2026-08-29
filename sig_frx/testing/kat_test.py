# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The harness fails the schemes it is supposed to fail.

A test suite for a test harness has one job, and it is not to watch a correct
scheme pass. Every case below breaks the scheme in one specific way — the way a
real implementation breaks — and requires the harness to catch it. A harness that
only ever ran against something correct would be indistinguishable from one that
returns `None`.

The vectors are produced by the reference scheme itself. That is circular as
evidence about the *scheme* and irrelevant here: what is under test is whether
the harness notices when the bytes stop matching, and where the bytes came from
does not change that.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import Array
from frx.typing import ArrayLike

from sig_frx.testing import kat
from sig_frx.testing.checksum_scheme import KEY_SIZE, ChecksumScheme

_BATCH = 4
_MESSAGE_LEN = 16
# The length of the case supplied from outside the set under test. Its own, so
# that it lands in a shape group of its own the way a case sourced from another
# mode of a published set does.
_SOURCED_MESSAGE_LEN = 20

# What both published sets look like to the harness's shape grouping: a message
# length per case, so no two cases share a batch and every group is a singleton.
_SINGLETON_MESSAGE_LENS = (13, 14, 15, 16)


def _reference_vectors(
    message_lens: Sequence[int] = (_MESSAGE_LEN,) * _BATCH,
) -> list[kat.KatVector]:
    """Cases the reference scheme satisfies, in the harness's normalized form."""
    scheme = ChecksumScheme(domain=7)
    rng = np.random.default_rng(0)
    vectors = []
    for index, message_len in enumerate(message_lens):
        seed = bytes(rng.integers(0, 256, KEY_SIZE, dtype=np.uint8))
        message = bytes(rng.integers(0, 256, message_len, dtype=np.uint8))
        public_key, secret_key = scheme.keygen(fnp.asarray(bytearray(seed)))
        signature = scheme.sign(
            secret_key,
            fnp.asarray(bytearray(message)),
            randomness=None,
            context=None,
        )
        vectors.append(
            kat.KatVector(
                case_id=f"reference/tc{index}",
                parameter_set="reference",
                seed=seed,
                public_key=kat.to_bytes(public_key),
                secret_key=kat.to_bytes(secret_key),
                message=message,
                signature=kat.to_bytes(signature),
            )
        )
    return vectors


def _reference_vectors_with_context() -> list[kat.KatVector]:
    """The same cases, signed under a non-empty context."""
    scheme = ChecksumScheme(domain=7)
    context = b"\x07\x07"
    vectors = []
    for vector in _reference_vectors():
        assert vector.secret_key is not None and vector.message is not None
        signature = scheme.sign(
            fnp.asarray(bytearray(vector.secret_key)),
            fnp.asarray(bytearray(vector.message)),
            randomness=None,
            context=fnp.asarray(bytearray(context)),
        )
        vectors.append(
            kat.KatVector(
                **{
                    **vars(vector),
                    "context": context,
                    "signature": kat.to_bytes(signature),
                }
            )
        )
    return vectors


def _rejected_vectors() -> list[kat.KatVector]:
    """An all-failure group, as a published set with no accepted case is one.

    The signature is moved along with the verdict, so the reference scheme
    reproduces the rejection rather than failing the comparison — these are cases
    a correct implementation passes, which is what makes them the shape under
    test rather than a broken set.
    """
    vectors = []
    for vector in _reference_vectors():
        assert vector.signature is not None
        vectors.append(
            kat.KatVector(
                **{
                    **vars(vector),
                    "valid": False,
                    "signature": bytes([vector.signature[0] ^ 1])
                    + vector.signature[1:],
                }
            )
        )
    return vectors


def _sourced_accepted_case() -> kat.KatVector:
    """An accepted case from outside the set under test, as a sourced one is.

    Its own key pair and its own message length, like the case a caller lifts out
    of the signing set: nothing lines it up with the group it is handed to beyond
    the operation both belong to. The secret key goes, because what is missing
    from an all-failure group is a case to verify and not a case to sign.
    """
    (vector,) = _reference_vectors((_SOURCED_MESSAGE_LEN,))
    return kat.KatVector(
        **{**vars(vector), "case_id": "sourced/tc1", "seed": None, "secret_key": None}
    )


class _AlwaysAccepts(ChecksumScheme):
    """The failure mode positive-only KAT suites cannot see."""

    def verify(
        self,
        public_key: ArrayLike,
        message: ArrayLike,
        signature: ArrayLike,
        *,
        context: ArrayLike | None = None,
        position: ArrayLike | None = None,
    ) -> Array:
        return fnp.ones(np.shape(public_key)[0], dtype=fnp.bool_)


class _AlwaysRejects(ChecksumScheme):
    """The mirror of `_AlwaysAccepts`, and the one an all-failure set cannot see."""

    def verify(
        self,
        public_key: ArrayLike,
        message: ArrayLike,
        signature: ArrayLike,
        *,
        context: ArrayLike | None = None,
        position: ArrayLike | None = None,
    ) -> Array:
        return fnp.zeros(np.shape(public_key)[0], dtype=fnp.bool_)


class _VerdictForTheWholeBatch(ChecksumScheme):
    """Decides once for the batch instead of per entry — the `all` is the bug."""

    def verify(
        self,
        public_key: ArrayLike,
        message: ArrayLike,
        signature: ArrayLike,
        *,
        context: ArrayLike | None = None,
        position: ArrayLike | None = None,
    ) -> Array:
        per_entry = super().verify(
            public_key, message, signature, context=context, position=position
        )
        return fnp.full(np.shape(public_key)[0], fnp.all(per_entry))


class _WrongSignature(ChecksumScheme):
    """An off-by-one in the encoding: verifies against itself, matches nothing."""

    def sign(
        self,
        secret_key: ArrayLike,
        message: ArrayLike,
        *,
        randomness: ArrayLike | None = None,
        context: ArrayLike | None = None,
    ) -> Array:
        return (
            super().sign(secret_key, message, randomness=randomness, context=context)
            + 1
        )


class _AnotherValidSignature(ChecksumScheme):
    """A verifier that accepts more than one signature per case.

    The trailing byte is not checked, so a case has 256 valid signatures and the
    published one is not privileged — Falcon's situation in miniature, where the
    salt and the sampler's stream leave a correct signer free to produce bytes
    the published set does not contain. Signing moves that byte, so every case
    disagrees with its published signature and every one still verifies.

    It is the *trailing* byte rather than the leading one because the harness
    tampers with the first: a scheme that ignored byte zero would accept
    `_flip_first_bit`'s output and fail the pass that exists to catch exactly
    that.
    """

    def sign(
        self,
        secret_key: ArrayLike,
        message: ArrayLike,
        *,
        randomness: ArrayLike | None = None,
        context: ArrayLike | None = None,
    ) -> Array:
        signature = super().sign(
            secret_key, message, randomness=randomness, context=context
        )
        return fnp.concatenate([signature[:-1], 255 - signature[-1:]])

    def verify(
        self,
        public_key: ArrayLike,
        message: ArrayLike,
        signature: ArrayLike,
        *,
        context: ArrayLike | None = None,
        position: ArrayLike | None = None,
    ) -> Array:
        secret_key = 255 - fnp.asarray(public_key)
        expected = self._mask(secret_key, message, context)
        return fnp.all(expected[..., :-1] == fnp.asarray(signature)[..., :-1], axis=-1)


class _WrongKeygen(ChecksumScheme):
    """A key derivation that is self-consistent and not the standard's."""

    def keygen(self, seed: ArrayLike) -> tuple[Array, Array]:
        public_key, secret_key = super().keygen(seed)
        return public_key, secret_key + 1


class _DropsTheContext(ChecksumScheme):
    """Accepts a context and signs without it — a wrong answer, not a gap."""

    def _mask(
        self, secret_key: Array, message: ArrayLike, context: ArrayLike | None
    ) -> Array:
        return super()._mask(secret_key, message, None)


class KatHarnessTest(absltest.TestCase):
    def test_the_reference_scheme_passes(self) -> None:
        kat.check(ChecksumScheme(domain=7), _reference_vectors())

    def test_catches_a_verifier_that_accepts_everything(self) -> None:
        with self.assertRaisesRegex(kat.KatError, "accepted after a bit flip"):
            kat.check(_AlwaysAccepts(domain=7), _reference_vectors())

    def test_catches_a_verdict_decided_for_the_whole_batch(self) -> None:
        with self.assertRaisesRegex(kat.KatError, "not deciding per entry"):
            kat.check(_VerdictForTheWholeBatch(domain=7), _reference_vectors())

    def test_catches_a_batch_verdict_where_no_published_group_batches(self) -> None:
        # The case above is caught by the tampering pass, which needs a group of
        # more than one case to compare entries across — and no published set has
        # one, because they vary the message length per case. These vectors are
        # that set: at `B = 1` a verifier that decides once for the batch is
        # indistinguishable from a correct one, so what catches it is the batch
        # the harness builds by replicating a published case.
        vectors = _reference_vectors(_SINGLETON_MESSAGE_LENS)
        self.assertLen({len(v.message or b"") for v in vectors}, len(vectors))
        with self.assertRaisesRegex(kat.KatError, "replicated across a batch"):
            kat.check(_VerdictForTheWholeBatch(domain=7), vectors)

    def test_the_reference_scheme_passes_where_nothing_batches(self) -> None:
        # The premise of the case above: the constructed batch is a check the
        # correct scheme still passes, not one that fails on singleton sets.
        kat.check(ChecksumScheme(domain=7), _reference_vectors(_SINGLETON_MESSAGE_LENS))

    def test_catches_a_wrong_signature(self) -> None:
        with self.assertRaisesRegex(kat.KatError, "wrong signature"):
            kat.check(_WrongSignature(domain=7), _reference_vectors())

    def test_a_signer_the_published_bytes_do_not_fix_passes_when_declared(
        self,
    ) -> None:
        """The case the declaration exists for: correct, and not byte-identical.

        Undeclared, the same scheme against the same set is the test above — the
        byte comparison fails it. So this pins that the declaration is what moves
        it, rather than the set or the scheme being different.
        """
        vectors = _reference_vectors()
        with self.assertRaisesRegex(kat.KatError, "wrong signature"):
            kat.check(_AnotherValidSignature(domain=7), vectors)

        kat.check(
            _AnotherValidSignature(domain=7),
            vectors,
            not_the_published_signature="under test",
        )

    def test_catches_a_signature_the_declared_schemes_own_verifier_refuses(
        self,
    ) -> None:
        """What the weaker comparison still catches, and the reason to run it."""
        with self.assertRaisesRegex(kat.KatError, "own verifier refuses"):
            kat.check(
                _WrongSignature(domain=7),
                _reference_vectors(),
                not_the_published_signature="under test",
            )

    def test_refuses_a_declaration_the_set_turns_out_to_fix(self) -> None:
        """A scheme that reproduces its published bytes is told to compare them.

        The declaration buys a weaker check, so a call that does not need it is
        an error rather than a harmless extra argument.
        """
        with self.assertRaisesRegex(kat.KatError, "reproduced theirs exactly"):
            kat.check(
                ChecksumScheme(domain=7),
                _reference_vectors(),
                not_the_published_signature="under test",
            )

    def test_refuses_a_declaration_about_cases_this_call_does_not_sign(self) -> None:
        vectors = [
            kat.KatVector(**{**vars(vector), "secret_key": None})
            for vector in _reference_vectors()
        ]
        with self.assertRaisesRegex(kat.KatError, "not one case in this call signs"):
            kat.check(
                _AnotherValidSignature(domain=7),
                vectors,
                not_the_published_signature="under test",
            )

    def test_refuses_a_declared_case_with_nothing_to_check_it_under(self) -> None:
        """Verifying what was produced needs a public key the record may not carry."""
        vectors = _reference_vectors()
        vectors[0] = kat.KatVector(**{**vars(vectors[0]), "public_key": None})
        with self.assertRaisesRegex(kat.KatError, "no public key to check it under"):
            kat.check(
                _AnotherValidSignature(domain=7),
                vectors,
                not_the_published_signature="under test",
            )

    def test_catches_a_wrong_key(self) -> None:
        with self.assertRaisesRegex(kat.KatError, "wrong secret key"):
            kat.check(_WrongKeygen(domain=7), _reference_vectors())

    def test_refuses_an_empty_set(self) -> None:
        # An empty set is the quietest way to claim a scheme is gated on vectors
        # it never ran.
        with self.assertRaisesRegex(kat.KatError, "empty set"):
            kat.check(ChecksumScheme(domain=7), [])

    def test_catches_a_scheme_that_ignores_the_context(self) -> None:
        # A dropped context verifies a different message than the one asked
        # about, and the vectors that would catch it are the ones carrying a
        # non-empty context.
        vectors = [
            kat.KatVector(**{**vars(v), "context": b"\x07\x07"})
            for v in _reference_vectors_with_context()
        ]
        with self.assertRaisesRegex(kat.KatError, "wrong signature"):
            kat.check(_DropsTheContext(domain=7), vectors)

    def test_refuses_a_vector_the_loader_could_not_fully_express(self) -> None:
        # The loader records what it could not feed to the seam. Running the
        # plain operation against a vector published for another one is a pass
        # for a case nobody ran, so the harness stops instead.
        vectors = list(_reference_vectors())
        vectors[1] = kat.KatVector(
            **{**vars(vectors[1]), "unsupported": ("context", "preHash")}
        )
        with self.assertRaisesRegex(kat.KatError, r"unsupported fields.*context"):
            kat.check(ChecksumScheme(domain=7), vectors)

    def test_refuses_two_parameter_sets_in_one_run(self) -> None:
        # One scheme instance is one parameter set; mixing them would run half
        # the cases against the wrong instance and report a pass.
        vectors = list(_reference_vectors())
        vectors[2] = kat.KatVector(
            **{**vars(vectors[2]), "parameter_set": "something-else"}
        )
        with self.assertRaisesRegex(kat.KatError, "one parameter set"):
            kat.check(ChecksumScheme(domain=7), vectors)

    def test_a_published_failure_verdict_must_be_rejected(self) -> None:
        # ACVP's sigVer sets are mostly deliberate failures. A scheme that
        # accepts one has to fail the harness, or those vectors are decoration.
        vectors = list(_reference_vectors())
        vectors[0] = kat.KatVector(**{**vars(vectors[0]), "valid": False})
        with self.assertRaisesRegex(kat.KatError, "published verdict is reject"):
            kat.check(ChecksumScheme(domain=7), vectors)

    def test_refuses_a_set_with_no_accepted_case(self) -> None:
        # Everything the harness derives moves a bit in a case the standard
        # accepts, so a set without one runs the verdict comparison and nothing
        # else. The reference scheme passes that comparison, which is exactly why
        # the shrinkage is invisible without this.
        with self.assertRaisesRegex(kat.KatError, "not one the standard accepts"):
            kat.check(ChecksumScheme(domain=7), _rejected_vectors())

    def test_a_verifier_that_rejects_everything_is_what_the_declaration_costs(
        self,
    ) -> None:
        # The mirror image of the always-accepting verifier, and the reason the
        # refusal is about the set rather than the scheme: rejecting
        # unconditionally agrees with every verdict an all-failure group
        # publishes, and the pass that would separate the two needs an accepted
        # case to move a bit in. So the declaration is not free — it is the
        # statement that this operation cannot tell those apart, and this is that
        # statement under test rather than described.
        vectors = _rejected_vectors()
        with self.assertRaisesRegex(kat.KatError, "not one the standard accepts"):
            kat.check(_AlwaysRejects(domain=7), vectors)
        kat.check(_AlwaysRejects(domain=7), vectors, no_accepted_case="under test")

    def test_a_declared_set_still_compares_its_published_verdicts(self) -> None:
        # What the declaration switches off is the derived half, not the
        # published one: an all-failure group still fails a verifier that accepts
        # everything, which is the check that survives where tampering cannot run.
        with self.assertRaisesRegex(kat.KatError, "published verdict is reject"):
            kat.check(
                _AlwaysAccepts(domain=7),
                _rejected_vectors(),
                no_accepted_case="under test",
            )

    def test_refuses_a_declaration_the_set_outgrew(self) -> None:
        # A regenerated set that starts publishing an accepted case is the good
        # outcome, and the declaration is what turns it into a prompt to delete
        # the declaration rather than a check that silently stays off.
        with self.assertRaisesRegex(kat.KatError, "declared as publishing no accepted"):
            kat.check(
                ChecksumScheme(domain=7),
                _reference_vectors(),
                no_accepted_case="under test",
            )

    def test_an_accepted_case_supplied_for_an_all_failure_set_is_verified(self) -> None:
        # The declaration's alternative, and what it buys back: the group that an
        # always-rejecting verifier satisfies is one it fails as soon as a case it
        # has to accept is handed to it. The correct scheme passes the same call,
        # which is what makes the failure the verifier's rather than the case's.
        vectors = _rejected_vectors()
        kat.check(
            ChecksumScheme(domain=7), vectors, accepted_case=_sourced_accepted_case()
        )
        with self.assertRaisesRegex(kat.KatError, "published verdict is accept"):
            kat.check(
                _AlwaysRejects(domain=7),
                vectors,
                accepted_case=_sourced_accepted_case(),
            )

    def test_a_supplied_accepted_case_is_what_makes_the_derived_passes_run(
        self,
    ) -> None:
        # Not only the extra verdict: everything the harness derives starts from
        # the supplied case too. A verdict decided for the whole batch survives an
        # all-failure group under the declaration — a reduction over rejections is
        # a rejection — and the batch built from the supplied case is what sees it.
        vectors = _rejected_vectors()
        kat.check(
            _VerdictForTheWholeBatch(domain=7), vectors, no_accepted_case="under test"
        )
        with self.assertRaisesRegex(kat.KatError, "replicated across a batch"):
            kat.check(
                _VerdictForTheWholeBatch(domain=7),
                vectors,
                accepted_case=_sourced_accepted_case(),
            )

    def test_a_supplied_case_that_does_not_verify_fails_the_call(self) -> None:
        # A stand-in is bytes the caller derived part of — a public key, for the
        # scheme that has to recompute one — and a derivation that drifted would
        # hand over a case nothing accepts. That gates nothing while looking like
        # coverage, so it fails against the correct scheme rather than passing.
        case = _sourced_accepted_case()
        assert case.public_key is not None
        wrong = kat.KatVector(
            **{
                **vars(case),
                "public_key": bytes([case.public_key[0] ^ 1]) + case.public_key[1:],
            }
        )
        with self.assertRaisesRegex(kat.KatError, "published verdict is accept"):
            kat.check(
                ChecksumScheme(domain=7), _rejected_vectors(), accepted_case=wrong
            )

    def test_a_supplied_case_is_held_to_the_operation_the_call_runs(self) -> None:
        # It is checked like a published vector, which is what keeps a case
        # belonging to another operation from standing in here: one that verifies
        # under a different instance would report coverage nobody has.
        case = kat.KatVector(
            **{**vars(_sourced_accepted_case()), "parameter_set": "something-else"}
        )
        with self.assertRaisesRegex(kat.KatError, "one parameter set"):
            kat.check(ChecksumScheme(domain=7), _rejected_vectors(), accepted_case=case)

    def test_refuses_a_supplied_case_the_set_did_not_need(self) -> None:
        # The mirror of the stale declaration, and wrong in the same direction:
        # once the set publishes an accepted case, the derived passes start from
        # the standard's own and the stand-in is coverage nobody has to invent.
        with self.assertRaisesRegex(kat.KatError, "standing in for one that is there"):
            kat.check(
                ChecksumScheme(domain=7),
                _reference_vectors(),
                accepted_case=_sourced_accepted_case(),
            )

    def test_refuses_a_supplied_case_and_a_declaration_at_once(self) -> None:
        # They answer the same question in opposite directions: one says the
        # derived checks cannot run and the other is what they run from.
        with self.assertRaisesRegex(kat.KatError, "exactly one of them describes"):
            kat.check(
                ChecksumScheme(domain=7),
                _rejected_vectors(),
                accepted_case=_sourced_accepted_case(),
                no_accepted_case="under test",
            )

    def test_refuses_a_declaration_about_cases_this_call_does_not_verify(self) -> None:
        # The other direction the declaration can be wrong in. A key generation
        # set has no verify case for the claim to be about, so a declaration
        # there describes nothing and would sit unread — which is the shape of
        # the problem it exists to prevent.
        vectors = [
            kat.KatVector(**{**vars(v), "message": None, "signature": None})
            for v in _reference_vectors()
        ]
        with self.assertRaisesRegex(kat.KatError, "no verify case at all"):
            kat.check(ChecksumScheme(domain=7), vectors, no_accepted_case="under test")

    def test_to_bytes_refuses_a_non_byte_wire_form(self) -> None:
        # The message is the seam question it defers, not a type error.
        with self.assertRaisesRegex(kat.KatError, "wire codec at the seam"):
            kat.to_bytes(fnp.zeros(4, dtype=fnp.uint32))


class AcvpLoaderTest(absltest.TestCase):
    """The join and the field mapping, against hand-built ACVP-shaped files."""

    def _write(self, name: str, payload: dict[str, Any]) -> str:
        return self.create_tempfile(name, content=json.dumps(payload)).full_path

    def test_joins_prompt_and_expected_on_group_and_case(self) -> None:
        prompt = self._write(
            "prompt.json",
            {
                "testGroups": [
                    {
                        "tgId": 1,
                        "parameterSet": "TEST-1",
                        "tests": [
                            {
                                "tcId": 1,
                                "skSeed": "00ff",
                                "skPrf": "1122",
                                "pkSeed": "3344",
                            }
                        ],
                    }
                ]
            },
        )
        expected = self._write(
            "expectedResults.json",
            {
                "testGroups": [
                    {"tgId": 1, "tests": [{"tcId": 1, "sk": "aabb", "pk": "ccdd"}]}
                ]
            },
        )
        (vector,) = kat.load_acvp(prompt, expected)
        self.assertEqual(vector.case_id, "TEST-1/tg1/tc1")
        self.assertEqual(vector.parameter_set, "TEST-1")
        # The three seed pieces concatenate in the order the standard's keygen
        # takes them, which is what lets the seam keep one `keygen(seed)`.
        self.assertEqual(vector.seed, bytes.fromhex("00ff11223344"))
        self.assertEqual(vector.secret_key, bytes.fromhex("aabb"))
        self.assertEqual(vector.public_key, bytes.fromhex("ccdd"))
        self.assertTrue(vector.valid)

    def test_carries_the_published_failure_verdict(self) -> None:
        prompt = self._write(
            "prompt.json",
            {
                "testGroups": [
                    {
                        "tgId": 1,
                        "parameterSet": "TEST-1",
                        "tests": [
                            {"tcId": 1, "pk": "00", "message": "01", "signature": "02"}
                        ],
                    }
                ]
            },
        )
        expected = self._write(
            "expectedResults.json",
            {"testGroups": [{"tgId": 1, "tests": [{"tcId": 1, "testPassed": False}]}]},
        )
        (vector,) = kat.load_acvp(prompt, expected)
        self.assertFalse(vector.valid)

    def test_rejects_a_mismatched_file_pair(self) -> None:
        prompt = self._write(
            "prompt.json",
            {
                "testGroups": [
                    {"tgId": 1, "parameterSet": "TEST-1", "tests": [{"tcId": 9}]}
                ]
            },
        )
        expected = self._write("expectedResults.json", {"testGroups": []})
        with self.assertRaisesRegex(kat.KatError, "not a matching pair"):
            kat.load_acvp(prompt, expected)


if __name__ == "__main__":
    absltest.main()
