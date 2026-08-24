# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Falcon verification against the round-3 vectors, and against tampering.

A scheme that accepts every published signature has proven nothing about
rejection — a verifier returning `True` unconditionally passes all of it — so the
negative cases are half of this file
([`conventions.md`](../../../../docs/reference/conventions.md)). Every one of
them is a mutation of a case the reference implementation accepts, and the
accepted case is asserted alongside, because a rejection that would also reject
the genuine signature proves nothing either.

The mutations are chosen to reach each stage of Algorithm 16 separately: the
message and the salt reach `HashToPoint`, a byte inside `enc_s` reaches the norm
through a decoded-but-different `s2`, the header byte and the padding reach the
decoder's own rejections, and a byte of the public key reaches the product. A
single generic bit flip would land in whichever stage happened to be first.

**The batch axis is gated on a batch this file builds.** The generator varies the
message length per case, so no two published cases share a static `L` and
grouping them yields only `B = 1` — the gap `conventions.md` records for the FIPS
validation programs, arriving here for the same reason. So an accepted case is
replicated and some entries are corrupted, which fails a `verify` that reduced
over the batch and equally one that ignored its input.

Falcon has no `sign` here ([#27](https://github.com/fractalyze/sig-frx/issues/27)),
so there is no round trip to lean on — which is the right way round: a scheme
verifying its own signatures is the self-consistency the same page says is not
evidence.
"""

from __future__ import annotations

from typing import Any

import frx
import numpy as np
from absl.testing import absltest, parameterized

from sig_frx.lattice.falcon import verify as falcon
from sig_frx.lattice.falcon.testing import falcon_reference as ref
from sig_frx.lattice.falcon.testing.falcon_vectors import VECTORS, Vector
from sig_frx.signature import Signature

_PARAMETER_SETS = ref.parameter_cases()
_CASES = tuple(
    {"name": name, "case": vector.case, "vector": vector}
    for name, vectors in VECTORS.items()
    for vector in vectors
)


def _bytes(blob: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(blob), dtype=np.uint8)


def _flip(blob: np.ndarray, index: int) -> np.ndarray:
    mutated = blob.copy()
    mutated[index] ^= 1
    return mutated


class Sizes(parameterized.TestCase):
    """Table 3.3's four lengths, against the formulas §3.11 states them as."""

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_derived_sizes_match_the_table(self, name: str, **params: Any) -> None:
        scheme = falcon.named(name)
        self.assertEqual(scheme.public_key_size, params["public_key_size"])
        self.assertEqual(scheme.secret_key_size, params["secret_key_size"])
        self.assertEqual(scheme.signature_max_size, params["signature_size"])

    def test_an_unknown_parameter_set_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            falcon.named("Falcon-256")

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_instances_are_value_equal(self, name: str, **params: Any) -> None:
        """The seam requires it: identity equality silently re-traces per call."""
        del params
        self.assertEqual(falcon.named(name), falcon.named(name))
        self.assertEqual(hash(falcon.named(name)), hash(falcon.named(name)))

    def test_the_seam_is_satisfied(self) -> None:
        self.assertIsInstance(falcon.named("Falcon-512"), Signature)


class HashToPoint(parameterized.TestCase):
    """Algorithm 3's fixed budget against a stream that has no bound."""

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_agrees_with_the_reference(self, name: str, **params: Any) -> None:
        del name
        n = params["n"]
        for message in (b"", b"\x00", bytes(range(64)) * 3):
            got = falcon.hash_to_point(np.frombuffer(message, dtype=np.uint8), n)
            np.testing.assert_array_equal(
                np.asarray(got), ref.hash_to_point(message, n)
            )

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_every_coefficient_is_a_residue(self, name: str, **params: Any) -> None:
        del name
        got = np.asarray(
            falcon.hash_to_point(
                np.frombuffer(b"residues", dtype=np.uint8), params["n"]
            )
        )
        self.assertLen(got, params["n"])
        self.assertTrue((got < ref.Q).all())


class Bound(parameterized.TestCase):
    """The norm comparison, at the edges the 32-bit lane makes interesting."""

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_accepts_at_the_bound_and_refuses_one_past_it(
        self, name: str, **params: Any
    ) -> None:
        del name
        bound = params["squared_norm_bound"]
        # A single coefficient carrying the whole budget: `⌊√b⌋² ≤ b < (⌊√b⌋+1)²`.
        root = int(np.floor(np.sqrt(bound)))
        self.assertTrue(bool(falcon._within_bound(np.array([root]), bound)))
        self.assertFalse(bool(falcon._within_bound(np.array([root + 1]), bound)))

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_a_total_that_overflows_the_lane_is_refused(
        self, name: str, **params: Any
    ) -> None:
        """`2n·(q/2)²` is 2^37; a sum that wrapped would accept a huge signature."""
        del name
        n, bound = params["n"], params["squared_norm_bound"]
        worst = np.full(2 * n, ref.Q // 2, dtype=np.int32)
        self.assertGreater(int((worst.astype(np.int64) ** 2).sum()), 1 << 32)
        self.assertFalse(bool(falcon._within_bound(worst, bound)))

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_the_sign_of_a_coefficient_does_not_matter(
        self, name: str, **params: Any
    ) -> None:
        del name
        bound = params["squared_norm_bound"]
        values = np.array([-30, 17, -4, 0, 9], dtype=np.int32)
        self.assertTrue(bool(falcon._within_bound(values, bound)))
        self.assertTrue(bool(falcon._within_bound(-values, bound)))


class Vectors(parameterized.TestCase):
    """Every transcribed case, accepted; every mutation of it, refused."""

    def _verify(
        self, name: str, pk: np.ndarray, message: np.ndarray, signature: np.ndarray
    ) -> bool:
        verdict = falcon.named(name).verify(
            pk[None, :], message[None, :], signature[None, :]
        )
        return bool(np.asarray(verdict)[0])

    @parameterized.parameters(*_CASES)
    def test_the_published_signature_is_accepted(
        self, name: str, case: int, vector: Vector
    ) -> None:
        del case
        self.assertTrue(
            self._verify(
                name,
                _bytes(vector.public_key),
                _bytes(vector.message),
                _bytes(vector.signature),
            )
        )

    @parameterized.parameters(*_CASES)
    def test_the_reference_agrees_that_it_is_accepted(
        self, name: str, case: int, vector: Vector
    ) -> None:
        """The transcription is only evidence if it reproduces upstream's verdict."""
        del case
        self.assertTrue(
            ref.verify(
                bytes.fromhex(vector.public_key),
                bytes.fromhex(vector.message),
                bytes.fromhex(vector.signature),
                name,
            )
        )

    @parameterized.parameters(*_CASES)
    def test_a_moved_bit_in_the_message_is_refused(
        self, name: str, case: int, vector: Vector
    ) -> None:
        del case
        self.assertFalse(
            self._verify(
                name,
                _bytes(vector.public_key),
                _flip(_bytes(vector.message), 0),
                _bytes(vector.signature),
            )
        )

    @parameterized.parameters(*_CASES)
    def test_a_moved_bit_in_the_salt_is_refused(
        self, name: str, case: int, vector: Vector
    ) -> None:
        """Reaches `HashToPoint` without touching what the decoder reads."""
        del case
        self.assertFalse(
            self._verify(
                name,
                _bytes(vector.public_key),
                _bytes(vector.message),
                _flip(_bytes(vector.signature), 1),
            )
        )

    @parameterized.parameters(*_CASES)
    def test_a_moved_bit_in_the_coefficients_is_refused(
        self, name: str, case: int, vector: Vector
    ) -> None:
        """Decodes to a different `s2`, so the norm is what has to catch it."""
        del case
        self.assertFalse(
            self._verify(
                name,
                _bytes(vector.public_key),
                _bytes(vector.message),
                _flip(_bytes(vector.signature), 60),
            )
        )

    @parameterized.parameters(*_CASES)
    def test_a_moved_bit_in_the_public_key_is_refused(
        self, name: str, case: int, vector: Vector
    ) -> None:
        del case
        self.assertFalse(
            self._verify(
                name,
                _flip(_bytes(vector.public_key), 3),
                _bytes(vector.message),
                _bytes(vector.signature),
            )
        )

    @parameterized.parameters(*_CASES)
    def test_the_uncompressed_header_is_refused(
        self, name: str, case: int, vector: Vector
    ) -> None:
        """§3.11.3's `cc = 10` is a different length; this decoder takes `01`."""
        del case
        signature = _bytes(vector.signature).copy()
        signature[0] = (signature[0] & 0x0F) | 0x50
        self.assertFalse(
            self._verify(
                name, _bytes(vector.public_key), _bytes(vector.message), signature
            )
        )

    @parameterized.parameters(*_CASES)
    def test_a_nonzero_padding_byte_is_refused(
        self, name: str, case: int, vector: Vector
    ) -> None:
        """Malleability: the same `s` with a different tail would verify."""
        del case
        signature = _bytes(vector.signature).copy()
        self.assertEqual(int(signature[-1]), 0)
        signature[-1] = 1
        self.assertFalse(
            self._verify(
                name, _bytes(vector.public_key), _bytes(vector.message), signature
            )
        )


class Batch(parameterized.TestCase):
    """The property the seam exists for, on a batch built here."""

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_each_entry_gets_its_own_verdict(self, name: str, **params: Any) -> None:
        del params
        vector = VECTORS[name][0]
        pk, message = _bytes(vector.public_key), _bytes(vector.message)
        good = _bytes(vector.signature)
        bad = _flip(good, 60)
        signatures = np.stack([good, bad, good, bad, bad])
        verdict = np.asarray(
            falcon.named(name).verify(
                np.stack([pk] * 5), np.stack([message] * 5), signatures
            )
        )
        np.testing.assert_array_equal(verdict, [True, False, True, False, False])

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_one_entry_is_a_batch_of_one(self, name: str, **params: Any) -> None:
        del params
        vector = VECTORS[name][0]
        verdict = falcon.named(name).verify(
            _bytes(vector.public_key)[None, :],
            _bytes(vector.message)[None, :],
            _bytes(vector.signature)[None, :],
        )
        self.assertEqual(np.asarray(verdict).shape, (1,))

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_it_traces_as_one_computation(self, name: str, **params: Any) -> None:
        """`jit` around the batch is the compilation unit `conventions.md` names."""
        del params
        vector = VECTORS[name][0]
        scheme = falcon.named(name)
        verdict = frx.jit(scheme.verify)(
            _bytes(vector.public_key)[None, :],
            _bytes(vector.message)[None, :],
            _bytes(vector.signature)[None, :],
        )
        self.assertTrue(bool(np.asarray(verdict)[0]))


class Malformed(parameterized.TestCase):
    """§3.6.2's shape: a wrong length is a verdict, a wrong rank is an error."""

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_a_short_public_key_verifies_as_false(
        self, name: str, **params: Any
    ) -> None:
        vector = VECTORS[name][0]
        verdict = falcon.named(name).verify(
            np.zeros((1, params["public_key_size"] - 1), dtype=np.uint8),
            _bytes(vector.message)[None, :],
            _bytes(vector.signature)[None, :],
        )
        np.testing.assert_array_equal(np.asarray(verdict), [False])

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_a_short_signature_verifies_as_false(
        self, name: str, **params: Any
    ) -> None:
        vector = VECTORS[name][0]
        verdict = falcon.named(name).verify(
            _bytes(vector.public_key)[None, :],
            _bytes(vector.message)[None, :],
            np.zeros((1, params["signature_size"] - 1), dtype=np.uint8),
        )
        np.testing.assert_array_equal(np.asarray(verdict), [False])

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_an_unbatched_argument_is_an_error(self, name: str, **params: Any) -> None:
        del params
        vector = VECTORS[name][0]
        with self.assertRaises(ValueError):
            falcon.named(name).verify(
                _bytes(vector.public_key),
                _bytes(vector.message)[None, :],
                _bytes(vector.signature)[None, :],
            )

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_mismatched_batch_lengths_are_an_error(
        self, name: str, **params: Any
    ) -> None:
        del params
        vector = VECTORS[name][0]
        with self.assertRaises(ValueError):
            falcon.named(name).verify(
                np.stack([_bytes(vector.public_key)] * 2),
                _bytes(vector.message)[None, :],
                _bytes(vector.signature)[None, :],
            )

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_a_context_is_refused(self, name: str, **params: Any) -> None:
        """Falcon defines none, so accepting one would verify a different claim."""
        del params
        vector = VECTORS[name][0]
        with self.assertRaises(ValueError):
            falcon.named(name).verify(
                _bytes(vector.public_key)[None, :],
                _bytes(vector.message)[None, :],
                _bytes(vector.signature)[None, :],
                context=np.frombuffer(b"domain", dtype=np.uint8),
            )


class NotYetImplemented(parameterized.TestCase):
    """The two seam methods #26 and #27 fill in, refusing loudly until then."""

    @parameterized.parameters("keygen", "sign")
    def test_the_producing_operations_raise(self, operation: str) -> None:
        scheme = falcon.named("Falcon-512")
        with self.assertRaises(NotImplementedError) as raised:
            if operation == "keygen":
                scheme.keygen(np.zeros(48, dtype=np.uint8))
            else:
                scheme.sign(np.zeros(1281, dtype=np.uint8), np.zeros(4, np.uint8))
        self.assertIn("#178", str(raised.exception))


if __name__ == "__main__":
    absltest.main()
