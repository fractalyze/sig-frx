# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The batch preamble's two verdicts, and the line between them.

`require_batch` decides one thing six schemes used to decide separately: whether
a batch that does not fit the parameter set is a caller mistake or an answer
about a signature. The line is a rank against a width — a wrong rank cannot be
answered, a wrong width can, because a batch carries one static width and so a
wrong one is every entry's answer at once.

What is pinned here is that line and the per-standard reading of which side a
width falls on (`batch.py`). The schemes' own suites pin that each passes the
reading its standard gave it; this one pins that both readings exist and that
neither leaks into the other — a `VERDICT` that raised, or an `ERROR` that
quietly answered `False`, would be a scheme silently changing what its verifier
returns, which is the drift this module was extracted to end.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest

from sig_frx.batch import WrongWidth, require_batch

_PUBLIC_KEY_SIZE = 4
_SIGNATURE_SIZE = 6
_MESSAGE_LENGTH = 3
_BATCH = 2


def _operands(
    *,
    public_key_size: int = _PUBLIC_KEY_SIZE,
    signature_size: int = _SIGNATURE_SIZE,
    batch: int = _BATCH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A well-formed batch, or one dimension of it bent by the caller."""
    return (
        np.arange(batch * public_key_size, dtype=np.uint8).reshape(
            batch, public_key_size
        ),
        np.zeros((batch, _MESSAGE_LENGTH), dtype=np.uint8),
        np.zeros((batch, signature_size), dtype=np.uint8),
    )


@frx.jit
def _first_key_bytes(public_key: object, message: object, signature: object) -> object:
    """The preamble under a tracer, returning something only it can produce.

    A jitted `verify` is where this runs in production, and every shape it reads
    is static there — so the same expressions decide, and the point of tracing it
    is that nothing in the preamble forces a value to the host on the way.
    """
    operands = require_batch(
        public_key,
        message,
        signature,
        public_key_size=_PUBLIC_KEY_SIZE,
        signature_size=_SIGNATURE_SIZE,
    )
    return operands.public_key[:, 0]


class WellFormedTest(absltest.TestCase):
    """What comes back when the batch is the shape every scheme asks for."""

    def test_the_operands_come_back_as_uint8(self) -> None:
        keys, messages, signatures = _operands()
        operands = require_batch(
            keys,
            messages,
            signatures,
            public_key_size=_PUBLIC_KEY_SIZE,
            signature_size=_SIGNATURE_SIZE,
        )
        self.assertTrue(operands.well_formed)
        self.assertEqual(operands.size, _BATCH)
        for value, source in (
            (operands.public_key, keys),
            (operands.message, messages),
            (operands.signature, signatures),
        ):
            self.assertEqual(value.dtype, fnp.uint8)
            self.assertEqual(tuple(value.shape), source.shape)
            np.testing.assert_array_equal(np.asarray(value), source)

    def test_a_batch_of_one_is_a_batch(self) -> None:
        # `B = 1` is the single verification the seam refuses to give its own
        # entry point, so it has to pass the same preamble as any other width.
        operands = require_batch(
            *_operands(batch=1),
            public_key_size=_PUBLIC_KEY_SIZE,
            signature_size=_SIGNATURE_SIZE,
        )
        self.assertTrue(operands.well_formed)
        self.assertEqual(operands.size, 1)

    def test_an_empty_batch_is_a_batch(self) -> None:
        # Nothing to verify is a well-formed question with an empty answer, not a
        # malformed batch: a caller that filtered its input down to nothing gets
        # `bool[0]` rather than an exception.
        operands = require_batch(
            *_operands(batch=0),
            public_key_size=_PUBLIC_KEY_SIZE,
            signature_size=_SIGNATURE_SIZE,
        )
        self.assertTrue(operands.well_formed)
        self.assertEqual(operands.size, 0)

    def test_the_message_width_is_the_callers(self) -> None:
        # `L` is static but not the scheme's, so no width is prescribed for it —
        # only that there is one message per key.
        keys, _, signatures = _operands()
        for length in (0, 1, 64):
            operands = require_batch(
                keys,
                np.zeros((_BATCH, length), dtype=np.uint8),
                signatures,
                public_key_size=_PUBLIC_KEY_SIZE,
                signature_size=_SIGNATURE_SIZE,
            )
            self.assertTrue(operands.well_formed)

    def test_it_traces(self) -> None:
        keys, messages, signatures = _operands()
        np.testing.assert_array_equal(
            np.asarray(_first_key_bytes(keys, messages, signatures)), keys[:, 0]
        )


class WrongWidthTest(absltest.TestCase):
    """A width the parameter set does not have, under each standard's reading."""

    def test_a_wrong_public_key_width_is_a_verdict_by_default(self) -> None:
        keys, messages, signatures = _operands(public_key_size=_PUBLIC_KEY_SIZE - 1)
        operands = require_batch(
            keys,
            messages,
            signatures,
            public_key_size=_PUBLIC_KEY_SIZE,
            signature_size=_SIGNATURE_SIZE,
        )
        # The verdict is the caller's to spell — the preamble reports the batch
        # size so it can, and does not build the answer itself.
        self.assertFalse(operands.well_formed)
        self.assertEqual(operands.size, _BATCH)

    def test_a_wrong_signature_width_is_a_verdict_by_default(self) -> None:
        operands = require_batch(
            *_operands(signature_size=_SIGNATURE_SIZE + 1),
            public_key_size=_PUBLIC_KEY_SIZE,
            signature_size=_SIGNATURE_SIZE,
        )
        self.assertFalse(operands.well_formed)

    def test_either_wrong_width_alone_decides(self) -> None:
        # `well_formed` is an `and` over both operands, so a right key beside a
        # wrong signature must not report the batch usable — the failure mode of
        # a preamble that checked only the operand it was written for.
        for kwargs in (
            {"public_key_size": _PUBLIC_KEY_SIZE + 1},
            {"signature_size": _SIGNATURE_SIZE - 1},
            {"public_key_size": _PUBLIC_KEY_SIZE + 1, "signature_size": 1},
        ):
            with self.subTest(**kwargs):
                operands = require_batch(
                    *_operands(**kwargs),
                    public_key_size=_PUBLIC_KEY_SIZE,
                    signature_size=_SIGNATURE_SIZE,
                )
                self.assertFalse(operands.well_formed)

    def test_a_wrong_public_key_width_raises_when_asked(self) -> None:
        with self.assertRaisesRegex(ValueError, r"a public key batch is \[B, 4\]"):
            require_batch(
                *_operands(public_key_size=_PUBLIC_KEY_SIZE - 1),
                public_key_size=_PUBLIC_KEY_SIZE,
                signature_size=_SIGNATURE_SIZE,
                public_key_width=WrongWidth.ERROR,
            )

    def test_a_wrong_signature_width_raises_when_asked(self) -> None:
        with self.assertRaisesRegex(ValueError, r"a signature batch is \[B, 6\]"):
            require_batch(
                *_operands(signature_size=_SIGNATURE_SIZE + 1),
                public_key_size=_PUBLIC_KEY_SIZE,
                signature_size=_SIGNATURE_SIZE,
                signature_width=WrongWidth.ERROR,
            )

    def test_asking_for_an_error_on_one_operand_leaves_the_other_a_verdict(
        self,
    ) -> None:
        # FIPS 205's reading, which is the one three schemes take: the signature
        # is a verdict the standard defines and the key is not.
        operands = require_batch(
            *_operands(signature_size=_SIGNATURE_SIZE - 1),
            public_key_size=_PUBLIC_KEY_SIZE,
            signature_size=_SIGNATURE_SIZE,
            public_key_width=WrongWidth.ERROR,
        )
        self.assertFalse(operands.well_formed)


class MisshapenBatchTest(absltest.TestCase):
    """A wrong rank, or parts that do not line up — a caller mistake either way."""

    def test_a_public_key_of_the_wrong_rank_raises(self) -> None:
        _, messages, signatures = _operands()
        for keys in (
            np.zeros(_PUBLIC_KEY_SIZE, dtype=np.uint8),
            np.zeros((_BATCH, 1, _PUBLIC_KEY_SIZE), dtype=np.uint8),
        ):
            with self.subTest(rank=keys.ndim):
                with self.assertRaisesRegex(ValueError, "a public key batch is"):
                    require_batch(
                        keys,
                        messages,
                        signatures,
                        public_key_size=_PUBLIC_KEY_SIZE,
                        signature_size=_SIGNATURE_SIZE,
                    )

    def test_a_signature_that_does_not_line_up_raises(self) -> None:
        keys, messages, signatures = _operands()
        for wrong in (signatures[:1], np.zeros(_SIGNATURE_SIZE, dtype=np.uint8)):
            with self.subTest(shape=wrong.shape):
                with self.assertRaisesRegex(ValueError, "one signature per public key"):
                    require_batch(
                        keys,
                        messages,
                        wrong,
                        public_key_size=_PUBLIC_KEY_SIZE,
                        signature_size=_SIGNATURE_SIZE,
                    )

    def test_a_message_that_does_not_line_up_raises(self) -> None:
        keys, messages, signatures = _operands()
        with self.assertRaisesRegex(ValueError, "one message per public key"):
            require_batch(
                keys,
                messages[:1],
                signatures,
                public_key_size=_PUBLIC_KEY_SIZE,
                signature_size=_SIGNATURE_SIZE,
            )

    def test_a_bare_message_is_not_a_batch_of_one(self) -> None:
        # The mistake the rank check exists for: a `[L]` message passed to a
        # `B = 1` call would otherwise be read as a batch of its own bytes, and
        # verify against nothing in particular.
        keys, _, signatures = _operands(batch=1)
        with self.assertRaisesRegex(ValueError, "one message per public key"):
            require_batch(
                keys,
                np.zeros(_MESSAGE_LENGTH, dtype=np.uint8),
                signatures,
                public_key_size=_PUBLIC_KEY_SIZE,
                signature_size=_SIGNATURE_SIZE,
            )

    def test_a_rank_error_outranks_a_wrong_width(self) -> None:
        # A rank this wrong has no width to read, so the width policy cannot
        # apply — `VERDICT` must not turn an unanswerable batch into `False`.
        _, messages, signatures = _operands()
        with self.assertRaisesRegex(ValueError, "a public key batch is"):
            require_batch(
                np.zeros(_PUBLIC_KEY_SIZE - 1, dtype=np.uint8),
                messages,
                signatures,
                public_key_size=_PUBLIC_KEY_SIZE,
                signature_size=_SIGNATURE_SIZE,
            )


if __name__ == "__main__":
    absltest.main()
