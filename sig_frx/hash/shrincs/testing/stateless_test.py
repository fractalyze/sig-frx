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
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

from sig_frx.hash.shrincs import stateless
from sig_frx.hash.shrincs.testing import vectors


def _batch(*values: bytes) -> np.ndarray:
    return np.frombuffer(b"".join(values), dtype=np.uint8).reshape(len(values), -1)


def _ctx(context: bytes) -> np.ndarray | None:
    """The context as the seam takes it: a `uint8` array, `None` meaning empty."""
    return np.frombuffer(context, dtype=np.uint8) if context else None


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

    def test_the_sizes_the_seam_would_publish(self) -> None:
        self.assertEqual(stateless.PUBLIC_KEY_SIZE, 48)
        self.assertEqual(stateless.SIGNATURE_SIZE, 5777)


class ReferenceTest(absltest.TestCase):
    def test_the_public_key_splits_into_the_reference_thirds(self) -> None:
        for case in vectors.REFERENCE:
            with self.subTest(case.label):
                self.assertEqual(
                    case.public_key, case.pk_seed + case.sl_root + case.sf_root
                )

    def test_every_reference_signature_verifies(self) -> None:
        scheme = stateless.Stateless()
        for case in vectors.REFERENCE:
            with self.subTest(case.label):
                got = scheme.verify(
                    _batch(case.public_key),
                    _batch(case.message),
                    _batch(case.signature),
                    context=_ctx(case.context),
                )
                self.assertEqual(list(np.asarray(got)), [True])

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
                    _batch(case.pk_seed + case.sl_root),
                    _batch(case.sf_root + case.message),
                    _batch(case.signature[1:]),
                    context=_ctx(case.context),
                )
                self.assertEqual(list(np.asarray(got)), [True])


class BatchTest(absltest.TestCase):
    def test_a_batch_verdict_is_per_entry(self) -> None:
        """A good and a tampered signature under one call, and neither moves.

        The seam exists so a batch is one traced computation; what that must not
        cost is an entry's verdict bleeding into its neighbour's.
        """
        case = vectors.REFERENCE[0]
        broken = bytearray(case.signature)
        broken[100] ^= 0x01
        scheme = stateless.Stateless()
        got = scheme.verify(
            _batch(case.public_key, case.public_key, case.public_key),
            _batch(case.message, case.message, case.message),
            _batch(case.signature, bytes(broken), case.signature),
            context=_ctx(case.context),
        )
        self.assertEqual(list(np.asarray(got)), [True, False, True])


class RejectionTest(absltest.TestCase):
    """Each of these is a thing the component claims to check."""

    def setUp(self) -> None:
        super().setUp()
        self.scheme = stateless.Stateless()
        self.case = vectors.REFERENCE[1]  # the one with a non-empty context

    def _verdict(
        self,
        *,
        public_key: bytes | None = None,
        message: bytes | None = None,
        signature: bytes | None = None,
        context: bytes | None = None,
    ) -> bool:
        case = self.case
        ctx = case.context if context is None else context
        got = self.scheme.verify(
            _batch(public_key if public_key is not None else case.public_key),
            _batch(message if message is not None else case.message),
            _batch(signature if signature is not None else case.signature),
            context=_ctx(ctx),
        )
        return bool(np.asarray(got)[0])

    def test_the_control_case_accepts(self) -> None:
        # Without this the rejections below prove nothing: a helper that rejected
        # everything would pass all of them.
        self.assertTrue(self._verdict())

    def test_a_stateful_indicator_is_rejected(self) -> None:
        for indicator in (0, 1, 128, 254):
            with self.subTest(indicator=indicator):
                tampered = bytes([indicator]) + self.case.signature[1:]
                self.assertFalse(self._verdict(signature=tampered))

    def test_a_flipped_bit_in_the_signature_is_rejected(self) -> None:
        for offset in (1, 17, 2000, len(self.case.signature) - 1):
            with self.subTest(offset=offset):
                broken = bytearray(self.case.signature)
                broken[offset] ^= 0x80
                self.assertFalse(self._verdict(signature=bytes(broken)))

    def test_a_wrong_sf_root_is_rejected(self) -> None:
        """The binding the wrapper exists for.

        `sf_root` reaches no hash unless the wrapper prepends it, so an
        implementation that dropped it would verify its own signatures forever and
        accept one issued under a different stateful half of the same key.
        """
        broken = bytearray(self.case.public_key)
        broken[32] ^= 0x01  # first byte of `sf_root`
        self.assertFalse(self._verdict(public_key=bytes(broken)))

    def test_a_wrong_sl_root_or_pk_seed_is_rejected(self) -> None:
        for offset, name in ((0, "pk_seed"), (16, "sl_root")):
            with self.subTest(name):
                broken = bytearray(self.case.public_key)
                broken[offset] ^= 0x01
                self.assertFalse(self._verdict(public_key=bytes(broken)))

    def test_a_wrong_context_is_rejected(self) -> None:
        self.assertFalse(self._verdict(context=b""))
        self.assertFalse(self._verdict(context=b"sig-frY"))
        self.assertFalse(self._verdict(context=b"sig-frx "))

    def test_a_wrong_message_is_rejected(self) -> None:
        broken = bytearray(self.case.message)
        broken[0] ^= 0x01
        self.assertFalse(self._verdict(message=bytes(broken)))

    def test_a_signature_of_the_wrong_length_is_a_verdict(self) -> None:
        for signature in (
            self.case.signature[:-1],
            self.case.signature + b"\x00",
            b"",
        ):
            with self.subTest(length=len(signature)):
                self.assertFalse(self._verdict(signature=signature))


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
                _batch(case.message),
                _batch(case.signature),
            )

    def test_a_batch_that_does_not_line_up_raises(self) -> None:
        case = self.case
        with self.assertRaisesRegex(ValueError, "one signature per public key"):
            self.scheme.verify(
                _batch(case.public_key, case.public_key),
                _batch(case.message, case.message),
                _batch(case.signature),
            )
        with self.assertRaisesRegex(ValueError, "one message per public key"):
            self.scheme.verify(
                _batch(case.public_key, case.public_key),
                _batch(case.message),
                _batch(case.signature, case.signature),
            )


class ValueTest(absltest.TestCase):
    """Two instances are equal by value, so a fresh one does not re-trace."""

    def test_equality_and_hash_are_value_based(self) -> None:
        self.assertEqual(stateless.Stateless(), stateless.Stateless())
        self.assertEqual(hash(stateless.Stateless()), hash(stateless.Stateless()))


if __name__ == "__main__":
    absltest.main()
