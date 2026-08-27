# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""LeanSig's wire format against leanSpec's own containers.

Two provenances, and the suite keeps them apart because they gate different
things. The published `PROD` vectors are bytes with no rule behind them, so they
gate the layout — the corners of each run, and a round trip through both
directions. The generated `TEST` vector's elements *are* a rule, so it gates the
encode direction from values, which a round trip cannot: an error symmetric
across encode and decode survives one.

Every case runs both eagerly and traced. That matters more here than in the
suites above it, because this layer is where bytes first become field elements —
the accumulator rule ([`../../../../CLAUDE.md`](../../../../CLAUDE.md)) is about
exactly the `sum` this packing rides, and a promotion there agrees in value on
both legs while disagreeing in dtype.
"""

from __future__ import annotations

from collections.abc import Sequence

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from zk_dtypes import koalabear_mont as F

from sig_frx.hash.leansig import ssz
from sig_frx.hash.leansig.field import PRIME
from sig_frx.hash.leansig.params import PROD, TEST, LeanSigParams
from sig_frx.hash.leansig.testing import harness
from sig_frx.hash.leansig.testing.mode_vectors import operand_elements
from sig_frx.hash.leansig.testing.ssz_vectors import (
    GENERATED_SIGNATURES,
    PUBLIC_KEYS,
    PUBLISHED_SIGNATURES,
    GeneratedSignature,
    PublicKeyVector,
    PublishedSignature,
)


def _buffer(encoded: str, jit: bool) -> np.ndarray | fnp.ndarray:
    """A transcribed vector as the uint8 array the codec takes, on either leg."""
    raw = np.frombuffer(bytes.fromhex(encoded), dtype=np.uint8)
    return fnp.asarray(raw) if jit else raw


def _hex(values: np.ndarray | fnp.ndarray) -> str:
    """An encoded uint8 array back as the hex a vector transcribes."""
    return bytes(np.asarray(values, dtype=np.uint8)).hex()


def _rows(values: fnp.ndarray) -> list[list[int]]:
    """A stack of lane-reversed digests -> canonical residues, leanSpec's order.

    The per-row form of `harness.to_leanspec_order`, kept here because this is
    the only suite that holds digests stacked — every other one gates a single
    digest at a time.
    """
    return [harness.to_leanspec_order(row) for row in values]


def _digests(seed: int, count: int, params: LeanSigParams) -> fnp.ndarray:
    """`count` lane-reversed digests, seeded the way the generated vector says."""
    return fnp.stack(
        [
            harness.lane_reversed(operand_elements(params.hash_length, seed + index))
            for index in range(count)
        ]
    )


def _decode_signature(encoded: str, params: LeanSigParams, jit: bool) -> tuple:
    decode = (
        harness.jitted(ssz.decode_signature, "params") if jit else ssz.decode_signature
    )
    return decode(_buffer(encoded, jit), params=params)


def _both_presets(vectors: Sequence[PublicKeyVector]) -> list[tuple]:
    """Each public-key vector at both presets and on both legs.

    The public-key encoding reads only `hash_length` and `parameter_length`,
    which the two presets share — so this runs the pair to prove that rather
    than assuming it.
    """
    return [
        (f"{vector.name}_{preset}_{'traced' if jit else 'host'}", vector, params, jit)
        for vector in vectors
        for preset, params in (("prod", PROD), ("test", TEST))
        for jit in (False, True)
    ]


class PublicKeyTest(parameterized.TestCase):
    """A public key decodes to upstream's elements and re-encodes to its bytes."""

    @parameterized.named_parameters(*_both_presets(PUBLIC_KEYS))
    def test_it_decodes_to_upstreams_elements(
        self, vector: PublicKeyVector, params: LeanSigParams, jit: bool
    ) -> None:
        decode = (
            harness.jitted(ssz.decode_public_key, "params")
            if jit
            else ssz.decode_public_key
        )

        root, parameter, well_formed = decode(
            _buffer(vector.encoded, jit), params=params
        )

        self.assertEqual(harness.to_leanspec_order(root), list(vector.root))
        self.assertEqual(harness.to_leanspec_order(parameter), list(vector.parameter))
        self.assertTrue(bool(np.asarray(well_formed)))
        self.assertEqual(root.dtype, F)
        self.assertEqual(parameter.dtype, F)
        self.assertEqual(root.shape, (params.hash_length,))
        self.assertEqual(parameter.shape, (params.parameter_length,))

    @parameterized.named_parameters(*_both_presets(PUBLIC_KEYS))
    def test_it_encodes_upstreams_elements_to_its_bytes(
        self, vector: PublicKeyVector, params: LeanSigParams, jit: bool
    ) -> None:
        encode = (
            harness.jitted(ssz.encode_public_key, "params")
            if jit
            else ssz.encode_public_key
        )

        got = encode(
            harness.lane_reversed(vector.root),
            harness.lane_reversed(vector.parameter),
            params=params,
        )

        self.assertEqual(_hex(got), vector.encoded)
        self.assertEqual(got.dtype, fnp.uint8 if jit else np.uint8)


class PublishedSignatureTest(parameterized.TestCase):
    """A published signature decodes at its corners and re-encodes to its bytes."""

    @parameterized.named_parameters(*harness.both_legs(PUBLISHED_SIGNATURES))
    def test_it_decodes_to_upstreams_layout(
        self, vector: PublishedSignature, jit: bool
    ) -> None:
        siblings, rho, hashes, well_formed = _decode_signature(
            vector.encoded, PROD, jit
        )

        self.assertTrue(bool(np.asarray(well_formed)))
        self.assertEqual(harness.to_leanspec_order(rho), list(vector.rho))
        self.assertEqual(_rows(siblings)[0], list(vector.first_sibling))
        self.assertEqual(_rows(siblings)[-1], list(vector.last_sibling))
        self.assertEqual(_rows(hashes)[0], list(vector.first_hash))
        self.assertEqual(_rows(hashes)[-1], list(vector.last_hash))
        self.assertEqual(siblings.shape, (PROD.log_lifetime, PROD.hash_length))
        self.assertEqual(hashes.shape, (PROD.dimension, PROD.hash_length))
        self.assertEqual(rho.shape, (PROD.randomness_length,))
        for values in (siblings, rho, hashes):
            self.assertEqual(values.dtype, F)

    @parameterized.named_parameters(*harness.both_legs(PUBLISHED_SIGNATURES))
    def test_it_re_encodes_to_the_same_bytes(
        self, vector: PublishedSignature, jit: bool
    ) -> None:
        siblings, rho, hashes, _ = _decode_signature(vector.encoded, PROD, jit)

        encode = (
            harness.jitted(ssz.encode_signature, "params")
            if jit
            else ssz.encode_signature
        )
        got = encode(siblings, rho, hashes, params=PROD)

        self.assertEqual(_hex(got), vector.encoded)


class GeneratedSignatureTest(parameterized.TestCase):
    """The `TEST` preset, gated from elements rather than through a round trip."""

    @parameterized.named_parameters(*harness.both_legs(GENERATED_SIGNATURES))
    def test_it_encodes_the_elements_to_upstreams_bytes(
        self, vector: GeneratedSignature, jit: bool
    ) -> None:
        params = harness.PRESETS[vector.preset]
        encode = (
            harness.jitted(ssz.encode_signature, "params")
            if jit
            else ssz.encode_signature
        )

        got = encode(
            _digests(vector.sibling_seed, params.log_lifetime, params),
            harness.lane_reversed(
                operand_elements(params.randomness_length, vector.rho_seed)
            ),
            _digests(vector.hash_seed, params.dimension, params),
            params=params,
        )

        self.assertEqual(_hex(got), vector.encoded)

    @parameterized.named_parameters(*harness.both_legs(GENERATED_SIGNATURES))
    def test_it_decodes_upstreams_bytes_to_the_elements(
        self, vector: GeneratedSignature, jit: bool
    ) -> None:
        params = harness.PRESETS[vector.preset]

        siblings, rho, hashes, well_formed = _decode_signature(
            vector.encoded, params, jit
        )

        self.assertTrue(bool(np.asarray(well_formed)))
        self.assertEqual(
            harness.to_leanspec_order(rho),
            list(operand_elements(params.randomness_length, vector.rho_seed)),
        )
        self.assertEqual(
            _rows(siblings),
            [
                list(operand_elements(params.hash_length, vector.sibling_seed + level))
                for level in range(params.log_lifetime)
            ],
        )
        self.assertEqual(
            _rows(hashes),
            [
                list(operand_elements(params.hash_length, vector.hash_seed + chain))
                for chain in range(params.dimension)
            ],
        )


class BatchTest(absltest.TestCase):
    """A batch decodes entrywise, which is what the seam takes."""

    def test_it_decodes_a_leading_axis_without_a_loop(self) -> None:
        stacked = fnp.stack(
            [_buffer(vector.encoded, jit=True) for vector in PUBLISHED_SIGNATURES]
        )

        siblings, rho, hashes, well_formed = ssz.decode_signature(stacked, params=PROD)

        self.assertEqual(well_formed.shape, (len(PUBLISHED_SIGNATURES),))
        self.assertEqual(
            siblings.shape,
            (len(PUBLISHED_SIGNATURES), PROD.log_lifetime, PROD.hash_length),
        )
        self.assertEqual(
            hashes.shape, (len(PUBLISHED_SIGNATURES), PROD.dimension, PROD.hash_length)
        )
        for index, vector in enumerate(PUBLISHED_SIGNATURES):
            self.assertEqual(harness.to_leanspec_order(rho[index]), list(vector.rho))
            self.assertEqual(_rows(siblings[index])[0], list(vector.first_sibling))
            self.assertEqual(_rows(hashes[index])[-1], list(vector.last_hash))

    def test_a_batch_re_encodes_to_the_bytes_it_came_from(self) -> None:
        stacked = fnp.stack(
            [_buffer(vector.encoded, jit=True) for vector in PUBLISHED_SIGNATURES]
        )

        siblings, rho, hashes, _ = ssz.decode_signature(stacked, params=PROD)
        got = ssz.encode_signature(siblings, rho, hashes, params=PROD)

        for index, vector in enumerate(PUBLISHED_SIGNATURES):
            self.assertEqual(_hex(got[index]), vector.encoded)


# `(name, byte position, replacement word)` for each way bytes of the right
# length can still fail to be a signature. Every offset is moved by one word
# rather than to something absurd, because a plausible offset is the one an
# equality check has to earn: at `PROD` the three are 36, 1064 and 4, and the
# replacements below are each a word off. The residues are what upstream's
# `Fp.deserialize` raises on and this repo's cast would silently reduce — the
# prime itself, which becomes zero, and a value just past it.
_MUTATIONS = (
    ("path_offset", 0, 40),
    ("hashes_offset", 32, 1060),
    ("sibling_offset", 36, 8),
    ("rho_residue", 4, PRIME),
    ("first_sibling_residue", 40, PRIME),
    ("last_hash_residue", 2532, PRIME + 5),
)


class RejectionTest(parameterized.TestCase):
    """What bytes of the right length can still get wrong."""

    @parameterized.named_parameters(*_MUTATIONS)
    def test_it_reports_a_mutated_signature_as_malformed(
        self, position: int, replacement: int
    ) -> None:
        raw = bytearray(bytes.fromhex(PUBLISHED_SIGNATURES[0].encoded))
        raw[position : position + ssz.P_BYTES] = replacement.to_bytes(
            ssz.P_BYTES, "little"
        )

        _, _, _, well_formed = ssz.decode_signature(
            np.frombuffer(bytes(raw), dtype=np.uint8), params=PROD
        )

        self.assertFalse(bool(np.asarray(well_formed)))

    def test_a_malformed_signature_still_comes_back_with_its_values(self) -> None:
        # Nothing narrows the values for a caller: a tracer has no `None`, and a
        # verifier folds the flag into its verdict rather than branching here.
        raw = bytearray(bytes.fromhex(PUBLISHED_SIGNATURES[0].encoded))
        raw[0:4] = (0).to_bytes(4, "little")

        siblings, rho, hashes, well_formed = ssz.decode_signature(
            np.frombuffer(bytes(raw), dtype=np.uint8), params=PROD
        )

        self.assertFalse(bool(np.asarray(well_formed)))
        self.assertEqual(siblings.shape, (PROD.log_lifetime, PROD.hash_length))
        self.assertEqual(hashes.shape, (PROD.dimension, PROD.hash_length))
        self.assertEqual(rho.shape, (PROD.randomness_length,))

    def test_it_rejects_a_buffer_of_the_wrong_length(self) -> None:
        raw = np.zeros(ssz.signature_size(PROD) - ssz.P_BYTES, dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "a signature is 2536 bytes"):
            ssz.decode_signature(raw, params=PROD)

    def test_it_rejects_a_public_key_of_the_wrong_length(self) -> None:
        raw = np.zeros(ssz.public_key_size(PROD) + 1, dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "a public key is 52 bytes"):
            ssz.decode_public_key(raw, params=PROD)

    def test_it_rejects_a_path_of_the_wrong_height(self) -> None:
        # The `TEST` preset's path under `PROD`, which is the mistake a preset
        # mix-up makes: it would otherwise produce a short buffer and fail at the
        # concatenate, naming no field.
        siblings = _digests(10, TEST.log_lifetime, PROD)
        rho = harness.lane_reversed(operand_elements(PROD.randomness_length, 3))
        hashes = _digests(100, PROD.dimension, PROD)

        with self.assertRaisesRegex(ValueError, "siblings is"):
            ssz.encode_signature(siblings, rho, hashes, params=PROD)


class SizeTest(absltest.TestCase):
    """The sizes this derives are the ones upstream states and publishes."""

    def test_the_sizes_are_upstreams(self) -> None:
        self.assertEqual(ssz.signature_size(PROD), 2536)
        self.assertEqual(ssz.signature_size(TEST), 424)
        self.assertEqual(ssz.public_key_size(PROD), 52)
        self.assertEqual(ssz.public_key_size(TEST), 52)

    def test_the_sizes_are_what_the_published_bytes_measure(self) -> None:
        for signature in PUBLISHED_SIGNATURES:
            self.assertLen(bytes.fromhex(signature.encoded), ssz.signature_size(PROD))
        for public_key in PUBLIC_KEYS:
            self.assertLen(bytes.fromhex(public_key.encoded), ssz.public_key_size(PROD))


if __name__ == "__main__":
    absltest.main()
