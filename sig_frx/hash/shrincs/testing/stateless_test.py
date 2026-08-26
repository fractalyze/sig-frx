# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The stateless component reproduces the SHRINCS reference, and rejects.

The positive cases are the gate: the reference implementation the specification
names as normative is the authority here, because SHRINCS publishes no vectors
and no validation program covers it ([`vectors.py`](vectors.py) records the pin).

The negative cases are the other half of it. A verifier that returned `True`
unconditionally passes every positive case, so what has to be shown is that each
thing this component checks is a thing it actually checks — the indicator byte,
the context, the message, the key, and above all `sf_root`, which is the binding
the wrapper exists for and the one a correct-looking implementation can drop
without failing any round trip of its own.

**Every verification in this file runs at one batch size**, which is what
`_verdicts` is for. Verification is compiled per input shape, so an unseen batch
size costs a full compile of the program — seconds — where a repeat of one
already seen costs milliseconds. Sixteen rejections at `B = 1` and one batch case
at `B = 3` is two compiles and sixteen round trips; the same coverage grouped
into a single shape is one compile and six.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from absl.testing import absltest

from sig_frx.hash.shrincs import shrincs, stateless
from sig_frx.hash.shrincs.testing import fixtures, vectors

# Past the indicator byte and the randomizer, so inside the signature proper.
_BODY = shrincs.INDEX_FIELD_START


def _verdicts(
    scheme: stateless.Stateless, cases: list[vectors.StatelessVectors]
) -> list[bool]:
    """`fixtures.verdicts` over recorded cases, which carry their own context.

    The adapter is here and the batching is not: what a `StatelessVectors` looks
    like is this file's business, and one batch width per file is every file's.
    """
    return fixtures.verdicts(
        scheme.verify,
        [(case.public_key, case.message, case.signature) for case in cases],
        cases[0].context,
    )


def _one_bit_flipped(signature: bytes) -> bytes:
    """The same signature, one bit of its body different."""
    broken = bytearray(signature)
    broken[_BODY] ^= 0x80
    return bytes(broken)


class ParameterTest(absltest.TestCase):
    """The set derives the specification's stateless table, rather than restating it."""

    def test_the_derived_constants_are_the_published_ones(self) -> None:
        params = stateless.PARAMS
        self.assertEqual(params.n, 16)
        self.assertEqual(params.tree_height, 9)  # `SPHX_XMSS_HEIGHT`
        self.assertEqual(params.d, 5)  # `SPHX_LAYER_COUNT`
        self.assertEqual(params.md_bytes, 17)  # `FORS_DIGEST_SIZE`
        self.assertEqual(params.m, 24)
        self.assertEqual(params.wots_params.len, 35)  # `WOTS_TW_CHAIN_COUNT`
        self.assertEqual(params.signature_size, 5776)  # `SPHX_SIGNATURE_SIZE`
        self.assertEqual(params.security_category, 1)

    def test_the_sizes_a_consumer_allocates_from(self) -> None:
        self.assertEqual(stateless.PUBLIC_KEY_SIZE, 48)
        self.assertEqual(stateless.SIGNATURE_SIZE, 5777)


class ReferenceTest(absltest.TestCase):
    def test_the_public_key_splits_into_the_reference_thirds(self) -> None:
        for case in vectors.REFERENCE:
            with self.subTest(case.label):
                self.assertEqual(
                    case.public_key, case.pk_seed + case.sl_root + case.sf_root
                )

    def test_the_randomizer_follows_a_one_byte_indicator(self) -> None:
        """Pins the stateless signature's layout ahead of the SLH-DSA it wraps."""
        for case in vectors.REFERENCE:
            with self.subTest(case.label):
                self.assertEqual(case.randomizer, case.signature[1:17])

    def test_every_reference_signature_verifies(self) -> None:
        scheme = stateless.Stateless()
        for case in vectors.REFERENCE:
            with self.subTest(case.label):
                self.assertEqual(_verdicts(scheme, [case]), [True])

    def test_the_wrapper_is_the_two_bindings_and_nothing_else(self) -> None:
        """Driving SLH-DSA directly, with the bindings applied by hand, agrees.

        This is what pins the wrapper as a wrapper: the same verdict has to come
        out of `slh_dsa.verify` given `pk_seed ‖ sl_root`, `sf_root ‖ M` and the
        signature past its indicator byte. If the component did anything else on
        the way, the two would part here rather than at a published byte.
        """
        scheme = stateless.Stateless()
        for case in vectors.REFERENCE:
            with self.subTest(case.label):
                got = scheme.slh_dsa.verify(
                    fixtures.rows(*([case.pk_seed + case.sl_root] * fixtures.BATCH)),
                    fixtures.rows(*([case.sf_root + case.message] * fixtures.BATCH)),
                    fixtures.rows(*([case.signature[1:]] * fixtures.BATCH)),
                    context=fixtures.context(case.context),
                )
                self.assertEqual(list(np.asarray(got)), [True] * fixtures.BATCH)


class SigningTest(absltest.TestCase):
    """The other half of the wrapper: the two bindings applied rather than undone.

    One case rather than every one. A stateless signature is 5776 bytes of FORS
    and hypertree, so signing is the dearest operation in this package by a wide
    margin, and what this has to show is the wrapper — the indicator byte, the
    `sf_root` binding and the salt reaching `PRF_msg_sl`. None of that varies case
    to case; what does is the context, and `RejectionTest` already shows the
    context reaching the digest.
    """

    def test_a_reference_signature_is_reproduced(self) -> None:
        """Byte for byte, at the recorded salt — a round trip would show less.

        The salt is why `opt_rand` is a recorded field: without the value a case
        was made under there is nothing here a signer could be held to, only a
        signature it could verify for itself.
        """
        case = vectors.REFERENCE[1]
        made = stateless.Stateless().sign(
            np.frombuffer(case.seed + case.sl_root, dtype=np.uint8),
            np.frombuffer(case.sf_root, dtype=np.uint8),
            np.frombuffer(case.message, dtype=np.uint8),
            randomness=np.frombuffer(case.opt_rand, dtype=np.uint8),
            context=fixtures.context(case.context),
        )
        self.assertEqual(bytes(np.asarray(made)), case.signature)

    def test_a_hedged_signer_will_not_quietly_go_deterministic(self) -> None:
        """`opt_rand` omitted would substitute `PK.seed` and still verify.

        Which is exactly why it raises: the signature would be a valid one over
        the right message, so nothing downstream could notice that the salt the
        caller meant to supply never arrived.
        """
        case = vectors.REFERENCE[0]
        with self.assertRaisesRegex(ValueError, "addrnd"):
            stateless.Stateless().sign(
                np.frombuffer(case.seed + case.sl_root, dtype=np.uint8),
                np.frombuffer(case.sf_root, dtype=np.uint8),
                np.frombuffer(case.message, dtype=np.uint8),
                context=fixtures.context(case.context),
            )


class RejectionTest(absltest.TestCase):
    """Each of these is a thing the component claims to check."""

    def setUp(self) -> None:
        super().setUp()
        self.scheme = stateless.Stateless()
        self.case = vectors.REFERENCE[1]  # the one with a non-empty context

    def test_the_control_case_accepts(self) -> None:
        # Without this the rejections below prove nothing: a helper that rejected
        # everything would pass all of them.
        self.assertEqual(_verdicts(self.scheme, [self.case]), [True])

    def test_a_batch_verdict_is_per_entry(self) -> None:
        """A good and a tampered signature in one call, and neither moves.

        The seam exists so a batch is one traced computation; what that must not
        cost is an entry's verdict bleeding into its neighbour's.
        """
        broken = bytearray(self.case.signature)
        broken[100] ^= 0x01
        bad = replace(self.case, signature=bytes(broken))
        self.assertEqual(
            _verdicts(self.scheme, [self.case, bad, self.case, bad]),
            [True, False, True, False],
        )

    def test_a_stateful_indicator_is_rejected(self) -> None:
        cases = [
            replace(self.case, signature=bytes([i]) + self.case.signature[1:])
            for i in (0, 1, 128, 254)
        ]
        self.assertEqual(_verdicts(self.scheme, cases), [False] * len(cases))

    def test_a_flipped_bit_in_the_signature_is_rejected(self) -> None:
        cases = []
        for offset in (1, 17, 2000, len(self.case.signature) - 1):
            broken = bytearray(self.case.signature)
            broken[offset] ^= 0x80
            cases.append(replace(self.case, signature=bytes(broken)))
        self.assertEqual(_verdicts(self.scheme, cases), [False] * len(cases))

    def test_a_wrong_public_key_third_is_rejected(self) -> None:
        """`sf_root` is the binding the wrapper exists for.

        It reaches no hash unless the wrapper prepends it, so an implementation
        that dropped it would verify its own signatures forever and accept one
        issued under a different stateful half of the same key. The other two
        thirds are SLH-DSA's own and ride along here.
        """
        cases = []
        for offset in (0, 16, 32):  # pk_seed, sl_root, sf_root
            broken = bytearray(self.case.public_key)
            broken[offset] ^= 0x01
            cases.append(replace(self.case, public_key=bytes(broken)))
        self.assertEqual(_verdicts(self.scheme, cases), [False] * len(cases))

    def test_a_wrong_message_is_rejected(self) -> None:
        broken = bytearray(self.case.message)
        broken[0] ^= 0x01
        self.assertEqual(
            _verdicts(self.scheme, [replace(self.case, message=bytes(broken))]), [False]
        )

    def test_a_wrong_context_is_rejected(self) -> None:
        # One call each: the context is one value for the whole batch, so it is
        # the one thing a batch cannot vary.
        for context in (b"", b"sig-frY", b"sig-frx "):
            with self.subTest(context=context):
                self.assertEqual(
                    _verdicts(self.scheme, [replace(self.case, context=context)]),
                    [False],
                )

    def test_a_signature_of_the_wrong_length_is_a_verdict(self) -> None:
        # These cost nothing: a wrong length is answered before any hashing.
        for signature in (
            self.case.signature[:-1],
            self.case.signature + b"\x00",
            b"",
        ):
            with self.subTest(length=len(signature)):
                self.assertEqual(
                    _verdicts(self.scheme, [replace(self.case, signature=signature)]),
                    [False],
                )


class ComponentTest(absltest.TestCase):
    """`accepts` — the wrapper without the door in front of it."""

    def setUp(self) -> None:
        super().setUp()
        self.scheme = stateless.Stateless()
        self.case = vectors.REFERENCE[1]  # the one with a non-empty context

    def _parsed(
        self, case: vectors.StatelessVectors
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        return (
            fixtures.rows(case.public_key),
            fixtures.rows(case.message),
            fixtures.rows(case.signature),
            fixtures.context(case.context),
        )

    def test_it_agrees_with_the_public_verifier(self) -> None:
        """The class is this function plus a preamble, so the two cannot differ.

        Both a signature that verifies and one that does not: a component that
        returned `True` unconditionally would agree with the verifier on the
        first and not the second.
        """
        for label, signature in (
            ("as issued", self.case.signature),
            ("tampered", _one_bit_flipped(self.case.signature)),
        ):
            case = replace(self.case, signature=signature)
            keys, messages, signatures, context = self._parsed(case)
            with self.subTest(label):
                np.testing.assert_array_equal(
                    np.asarray(
                        stateless.accepts(
                            self.scheme.slh_dsa, keys, messages, signatures, context
                        )
                    ),
                    np.asarray(
                        self.scheme.verify(keys, messages, signatures, context=context)
                    ),
                )

    def test_a_wrong_width_raises_here_and_is_a_verdict_at_the_door(self) -> None:
        """The one place the two deliberately part.

        A verifier handed a signature of the wrong length answers no — that is a
        thing the world can send it. A component handed one has a caller inside
        the package that got the format wrong, and saying `False` would let that
        caller keep going.
        """
        case = replace(self.case, signature=self.case.signature + b"\x00")
        keys, messages, signatures, context = self._parsed(case)
        self.assertEqual(
            [
                bool(v)
                for v in np.asarray(
                    self.scheme.verify(keys, messages, signatures, context=context)
                )
            ],
            [False],
        )
        with self.assertRaisesRegex(ValueError, r"stateless signature batch is"):
            stateless.accepts(self.scheme.slh_dsa, keys, messages, signatures, context)


class ShapeTest(absltest.TestCase):
    """A caller mistake is an error, not a verdict."""

    def setUp(self) -> None:
        super().setUp()
        self.scheme = stateless.Stateless()
        self.case = vectors.REFERENCE[0]

    def test_an_unbatched_argument_raises(self) -> None:
        case = self.case
        with self.assertRaisesRegex(ValueError, r"public key batch is \[B, 48\]"):
            self.scheme.verify(
                np.frombuffer(case.public_key, dtype=np.uint8),
                fixtures.rows(case.message),
                fixtures.rows(case.signature),
            )

    def test_a_batch_that_does_not_line_up_raises(self) -> None:
        case = self.case
        with self.assertRaisesRegex(ValueError, "one signature per public key"):
            self.scheme.verify(
                fixtures.rows(case.public_key, case.public_key),
                fixtures.rows(case.message, case.message),
                fixtures.rows(case.signature),
            )
        with self.assertRaisesRegex(ValueError, "one message per public key"):
            self.scheme.verify(
                fixtures.rows(case.public_key, case.public_key),
                fixtures.rows(case.message),
                fixtures.rows(case.signature, case.signature),
            )


class ValueTest(absltest.TestCase):
    """Two instances are equal by value, so a fresh one does not re-trace."""

    def test_equality_and_hash_are_value_based(self) -> None:
        self.assertEqual(stateless.Stateless(), stateless.Stateless())
        self.assertEqual(hash(stateless.Stateless()), hash(stateless.Stateless()))


if __name__ == "__main__":
    absltest.main()
