# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""leanSig's verifier against leanSpec's, and the seam field that reaches it.

The layers below this one are each gated on their own vectors; what is left to
pin is the assembly — that the chain walk, the leaf sponge and the Merkle climb
are wired to each other and to the codec in the order upstream wires them, and
that the verdict folds in everything the bytes can get wrong.

So the positive claim is narrow and total: for every `(key, slot, root,
signature)` upstream accepts, this accepts, and for the two it refuses, this
refuses. A verifier that returned `True` unconditionally reproduces every
accepting case, which is why the refusals are half the gate
([`testing.md`](../../../../docs/reference/testing.md)).

The mutations below are the other half. Each is a single wrong decision a
plausible implementation makes — the index dropped from the climb, the chain
walked the wrong distance, the operands transposed — and each has to cost a
case, or this suite is not measuring the wiring it claims to.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from sig_frx.hash.leansig import leansig
from sig_frx.hash.leansig.field import PRIME
from sig_frx.hash.leansig.testing.verify_vectors import (
    PUBLIC_KEY,
    VERIFY_VECTORS,
    VerifyVector,
)

_SCHEME = leansig.named("test")


def _batch(
    vectors: tuple[VerifyVector, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The whole vector set as one batch — keys, messages, signatures, slots."""
    return (
        np.stack(
            [np.frombuffer(bytes.fromhex(PUBLIC_KEY), dtype=np.uint8)] * len(vectors)
        ),
        np.stack(
            [np.frombuffer(bytes.fromhex(v.message), dtype=np.uint8) for v in vectors]
        ),
        np.stack(
            [np.frombuffer(bytes.fromhex(v.signature), dtype=np.uint8) for v in vectors]
        ),
        np.asarray([v.slot for v in vectors]),
    )


def _verdicts(values: object) -> list[bool]:
    return [bool(value) for value in np.asarray(values)]


class AgainstUpstreamTest(parameterized.TestCase):
    """Every vector's verdict, one at a time and then all at once."""

    @parameterized.named_parameters(
        *[(vector.name, vector) for vector in VERIFY_VECTORS]
    )
    def test_one_vector(self, vector: VerifyVector) -> None:
        keys, messages, signatures, slots = _batch((vector,))
        self.assertEqual(
            _verdicts(_SCHEME.verify(keys, messages, signatures, position=slots)),
            [vector.verdict],
        )

    def test_the_whole_set_in_one_call(self) -> None:
        """The batch is the point: one call, and entry `i` answers entry `i`.

        Run together rather than only one at a time because a verifier that
        decided once for the batch — an `all` where a per-entry select belongs —
        passes every single-entry case above and fails here, the set having both
        verdicts in it.
        """
        keys, messages, signatures, slots = _batch(VERIFY_VECTORS)
        self.assertEqual(
            _verdicts(_SCHEME.verify(keys, messages, signatures, position=slots)),
            [vector.verdict for vector in VERIFY_VECTORS],
        )

    def test_a_batch_of_one_is_a_batch(self) -> None:
        accepted = VERIFY_VECTORS[0]
        keys, messages, signatures, slots = _batch((accepted,))
        self.assertEqual(
            _verdicts(_SCHEME.verify(keys, messages, signatures, position=slots)),
            [True],
        )


class SlotTest(absltest.TestCase):
    """`position` is per entry, required, and bounded by the key's lifetime."""

    def setUp(self) -> None:
        super().setUp()
        self.accepted = tuple(v for v in VERIFY_VECTORS if v.verdict)

    def test_a_slot_belongs_to_its_own_entry(self) -> None:
        """Two entries, slots swapped: both must fail.

        The failure this catches is a verifier that reads one slot for the whole
        batch — with the right slot first, a shared read reproduces entry zero's
        verdict and gets entry one's wrong.
        """
        pair = self.accepted[:2]
        keys, messages, signatures, slots = _batch(pair)
        self.assertEqual(
            _verdicts(_SCHEME.verify(keys, messages, signatures, position=slots[::-1])),
            [False, False],
        )

    def test_it_is_required(self) -> None:
        keys, messages, signatures, _ = _batch(self.accepted[:1])
        with self.assertRaisesRegex(ValueError, "position.* is required"):
            _SCHEME.verify(keys, messages, signatures)

    def test_one_slot_per_entry(self) -> None:
        keys, messages, signatures, slots = _batch(self.accepted[:2])
        with self.assertRaisesRegex(ValueError, "one slot per public key"):
            _SCHEME.verify(keys, messages, signatures, position=slots[:1])

    def test_a_slot_past_the_lifetime_is_an_error(self) -> None:
        # The tree has no leaf to index, so there is nothing to compute a
        # `False` from — unlike a wrong-but-valid slot, which is a verdict.
        keys, messages, signatures, _ = _batch(self.accepted[:1])
        for slot in (_SCHEME.signatures_per_key, -1):
            with self.subTest(slot=slot):
                with self.assertRaisesRegex(ValueError, "covers slots"):
                    _SCHEME.verify(
                        keys, messages, signatures, position=np.asarray([slot])
                    )


class SeamTest(absltest.TestCase):
    """What the seam prescribes and what leanSig adds to it."""

    def test_the_message_is_a_32_byte_root(self) -> None:
        # `L` is the caller's everywhere else on the seam; this scheme signs a
        # root, so the width is its own to require.
        keys, _, signatures, slots = _batch(VERIFY_VECTORS[:1])
        with self.assertRaisesRegex(ValueError, "32-byte root"):
            _SCHEME.verify(
                keys,
                np.zeros((1, 31), dtype=np.uint8),
                signatures,
                position=slots,
            )

    def test_a_context_is_refused(self) -> None:
        keys, messages, signatures, slots = _batch(VERIFY_VECTORS[:1])
        with self.assertRaisesRegex(ValueError, "no application context"):
            _SCHEME.verify(
                keys,
                messages,
                signatures,
                context=np.asarray([1], dtype=np.uint8),
                position=slots,
            )

    def test_a_mis_sized_signature_is_a_verdict(self) -> None:
        # The FIPS reading, which `batch.py` makes the default: a batch carries
        # one static width, so a wrong one is every entry's answer.
        keys, messages, signatures, slots = _batch(VERIFY_VECTORS[:1])
        self.assertEqual(
            _verdicts(
                _SCHEME.verify(keys, messages, signatures[:, :-1], position=slots)
            ),
            [False],
        )

    def test_instances_are_value_equal(self) -> None:
        # A scheme rides pytree aux, where identity equality re-traces the
        # enclosing jit zone on every freshly built instance (`signature.py`).
        self.assertEqual(leansig.named("test"), leansig.named("test"))
        self.assertEqual(hash(leansig.named("test")), hash(leansig.named("test")))
        self.assertNotEqual(leansig.named("test"), leansig.named("prod"))

    def test_named_refuses_an_unknown_preset(self) -> None:
        with self.assertRaisesRegex(ValueError, "not one of"):
            leansig.named("production")

    def test_the_sizes_are_the_codec_s(self) -> None:
        self.assertEqual(_SCHEME.public_key_size, len(PUBLIC_KEY) // 2)
        self.assertEqual(
            _SCHEME.signature_max_size, len(VERIFY_VECTORS[0].signature) // 2
        )


class TamperedSignatureTest(parameterized.TestCase):
    """Bytes that decode but do not attest — each a verdict, none an error."""

    @parameterized.named_parameters(
        ("a_released_chain_hash", -1),
        ("a_path_sibling", 40),
        ("the_randomness", 8),
    )
    def test_one_flipped_byte_is_refused(self, offset: int) -> None:
        accepted = next(v for v in VERIFY_VECTORS if v.verdict)
        keys, messages, signatures, slots = _batch((accepted,))
        tampered = signatures.copy()
        tampered[0, offset] ^= 0x01
        self.assertEqual(
            _verdicts(_SCHEME.verify(keys, messages, tampered, position=slots)),
            [False],
        )

    def test_a_non_canonical_public_key_is_refused(self) -> None:
        """The same key spelled twice — malleability, not a wrong root.

        A four-byte group holding `v` and one holding `v + PRIME` reduce to the
        same residue under `astype`, where upstream's `Fp.deserialize` raises. So
        the second spelling decodes to *this key* and climbs to *this root*: the
        comparison cannot see it, and without the codec's range check one public
        key would have many valid encodings. Every residue is below `PRIME` and
        `2 * PRIME < 2^32`, so the shifted spelling always fits the group.

        This is why the flag is folded in beside the comparison rather than
        trusted to it — a mis-typed key byte would fail either way, and that is
        the case this deliberately is not.
        """
        accepted = next(v for v in VERIFY_VECTORS if v.verdict)
        keys, messages, signatures, slots = _batch((accepted,))
        restated = keys.copy()
        group = restated[0, :4].view(np.uint32)[0] + np.uint32(PRIME)
        restated[0, :4] = np.frombuffer(np.uint32(group).tobytes(), dtype=np.uint8)
        # It really is the same key: only the spelling moved.
        self.assertNotEqual(bytes(restated[0, :4]), bytes(keys[0, :4]))
        self.assertEqual(
            _verdicts(_SCHEME.verify(restated, messages, signatures, position=slots)),
            [False],
        )

    def test_a_non_canonical_residue_is_refused(self) -> None:
        """A four-byte group at or above the prime.

        `astype` reduces where upstream's `Fp.deserialize` raises, so without the
        codec's range check this would verify as a different, well-formed element
        of the same length — invisible to the seam. The check is `ssz.py`'s; this
        is the verdict reaching the caller through the scheme.
        """
        accepted = next(v for v in VERIFY_VECTORS if v.verdict)
        keys, messages, signatures, slots = _batch((accepted,))
        tampered = signatures.copy()
        tampered[0, -4:] = np.asarray([0xFF, 0xFF, 0xFF, 0xFF], dtype=np.uint8)
        self.assertEqual(
            _verdicts(_SCHEME.verify(keys, messages, tampered, position=slots)),
            [False],
        )


class TracedCoreTest(absltest.TestCase):
    """The hashing traces even though the encode cannot.

    `verify` is an eager entrance — both encoders decompose a wide integer
    base-p, which no lane holds — so what is claimed here is what the seam's rule
    actually asks for: the work traces as one computation rather than as `B`
    dispatches. The gate is that a traced run returns the eager verdicts entry
    for entry, including the entries meant to fail.
    """

    def test_the_batch_agrees_with_itself_on_both_legs(self) -> None:
        keys, messages, signatures, slots = _batch(VERIFY_VECTORS)
        eager = _verdicts(_SCHEME.verify(keys, messages, signatures, position=slots))
        traced = _verdicts(
            _SCHEME.verify(
                fnp.asarray(keys),
                messages,
                fnp.asarray(signatures),
                position=slots,
            )
        )
        self.assertEqual(traced, eager)
        self.assertEqual(eager, [vector.verdict for vector in VERIFY_VECTORS])


if __name__ == "__main__":
    absltest.main()
