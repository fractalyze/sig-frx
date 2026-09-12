# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""FROST against RFC 9591's vectors, stage by stage, per ciphersuite.

The vector files carry the RFC appendix's intermediates — dealer shares,
nonces, binding factors, signature shares — so each stage is pinned where it
can fail as itself: a wrong dealer fails before a wrong nonce, a wrong nonce
before a wrong share, a wrong share before a wrong signature. Every case runs
for both ciphersuites, which is what holds the skeleton to its
parameterization claim: the suites differ in scalar endianness, point
encoding, and scalar derivation (raw-digest reduction versus hash-to-field),
and none of that may leak into the round logic.

The last gate is per-suite, because it is what each suite's output *is*:
FROST(Ed25519, SHA-512)'s aggregate is a plain RFC 8032 signature, its suite
`verify` delegating to the existing batched Ed25519 verifier; FROST(secp256k1,
SHA-256)'s is RFC 9591's own Schnorr encoding, verified through the suite's
own batched surface and held row for row against Appendix B's naive
transcription kept below as the reference pair (it is not a BIP-340
signature, and the test would be lying if it pretended a chain verifier
existed for it).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from absl.testing import absltest, parameterized
from python.runfiles import Runfiles

from sig_frx.threshold import frost, group
from sig_frx.threshold.ed25519_sha512 import Ed25519Sha512
from sig_frx.threshold.secp256k1_sha256 import Secp256k1Sha256

_RUNFILES = Runfiles.Create()

_SUITES: dict[str, tuple[frost.Ciphersuite, str]] = {
    "ed25519": (
        Ed25519Sha512(),
        "frost_ed25519_sha512_vectors/file/frost-ed25519-sha512.json",
    ),
    "secp256k1": (
        Secp256k1Sha256(),
        "frost_secp256k1_sha256_vectors/file/frost-secp256k1-sha256.json",
    ),
}

_PARAMS = tuple(("_" + name, name) for name in _SUITES)


def _stack(*rows: bytes) -> np.ndarray:
    """Byte strings as one `uint8[B, L]` batch, a row each."""
    return np.stack([np.frombuffer(row, dtype=np.uint8) for row in rows])


class _Vectors:
    """One suite's vector file, in the shapes the round functions consume."""

    def __init__(self, name: str) -> None:
        self.cs: frost.Ciphersuite
        self.cs, resource = _SUITES[name]
        path = _RUNFILES.Rlocation(resource)
        assert path is not None
        data = json.loads(open(path).read())
        scalar = lambda h: self.cs.deserialize_scalar(bytes.fromhex(h))  # noqa: E731
        inputs = data["inputs"]
        self.secret = scalar(inputs["group_secret_key"])
        self.group_public_key = bytes.fromhex(inputs["group_public_key"])
        self.message = bytes.fromhex(inputs["message"])
        self.coefficients = [scalar(c) for c in inputs["share_polynomial_coefficients"]]
        self.shares = {
            entry["identifier"]: scalar(entry["participant_share"])
            for entry in inputs["participant_shares"]
        }
        self.round_one = {
            entry["identifier"]: entry for entry in data["round_one_outputs"]["outputs"]
        }
        self.round_two = data["round_two_outputs"]["outputs"]
        self.final_signature = data["final_output"]["sig"]
        self.commitment_list = [
            frost.Commitment(
                identifier,
                bytes.fromhex(entry["hiding_nonce_commitment"]),
                bytes.fromhex(entry["binding_nonce_commitment"]),
            )
            for identifier, entry in sorted(self.round_one.items())
        ]
        self.scalar = scalar

    def nonces(self, identifier: int) -> frost.Nonces:
        """The published round-one state for `identifier`, as the caller holds it."""
        entry = self.round_one[identifier]
        return frost.Nonces(
            self.scalar(entry["hiding_nonce"]),
            self.scalar(entry["binding_nonce"]),
            bytes.fromhex(entry["hiding_nonce_commitment"]),
            bytes.fromhex(entry["binding_nonce_commitment"]),
        )

    def signature_shares(self) -> list[bytes]:
        return [bytes.fromhex(entry["sig_share"]) for entry in self.round_two]

    def aggregate(self) -> bytes:
        return frost.aggregate(
            self.cs,
            self.commitment_list,
            self.message,
            self.group_public_key,
            self.signature_shares(),
        )


class _GroupOnly:
    """A suite's group half with the five hashes withheld.

    The point of the split is that Appendix C's dealer and the interpolation
    need no transcript, and the way to hold that claim honest is to run them
    against something that *cannot* provide one: every `PrimeOrderGroup`
    member delegates, and `h1`-`h5` are absent rather than raising, so a
    function that reached for a hash fails as `AttributeError` here instead
    of quietly working because a real ciphersuite was passed.

    This is the shape a second threshold protocol over these curves brings —
    the reuse the seam exists for, exercised rather than asserted.
    """

    def __init__(self, suite: frost.Ciphersuite) -> None:
        self._suite = suite
        self.order = suite.order
        self.scalar_field = suite.scalar_field

    def serialize_scalar(self, scalar: int) -> bytes:
        return self._suite.serialize_scalar(scalar)

    def deserialize_scalar(self, data: bytes) -> int:
        return self._suite.deserialize_scalar(data)

    def scalar_base_mult(self, scalar: int) -> bytes:
        return self._suite.scalar_base_mult(scalar)

    def deserialize_elements(self, data: Sequence[bytes]) -> Any:
        return self._suite.deserialize_elements(data)

    def elements_add(self, left: Any, right: Any) -> Any:
        return self._suite.elements_add(left, right)

    def elements_scalar_mult(self, elements: Any, scalars: Sequence[int]) -> Any:
        return self._suite.elements_scalar_mult(elements, scalars)

    def select_elements(self, elements: Any, indices: Sequence[int]) -> Any:
        return self._suite.select_elements(elements, indices)

    def sum_elements(self, elements: Any) -> Any:
        return self._suite.sum_elements(elements)

    def serialize_elements(self, elements: Any) -> list[bytes]:
        return self._suite.serialize_elements(elements)


if TYPE_CHECKING:
    _: type[group.PrimeOrderGroup] = _GroupOnly


class GroupSeamTest(parameterized.TestCase):
    """The functions that take a group take *only* a group."""

    @parameterized.named_parameters(*_PARAMS)
    def test_the_stand_in_is_a_group_and_not_a_ciphersuite(self, name: str) -> None:
        bare = _GroupOnly(_SUITES[name][0])
        self.assertIsInstance(bare, group.PrimeOrderGroup)
        self.assertNotIsInstance(bare, frost.Ciphersuite)
        for absent in ("h1", "h2", "h3", "h4", "h5"):
            self.assertFalse(hasattr(bare, absent), absent)

    @parameterized.named_parameters(*_PARAMS)
    def test_every_group_taking_function_agrees_through_the_narrow_seam(
        self, name: str
    ) -> None:
        """Same answers with the transcript withheld as with the whole suite.

        Equality rather than the published values, which `FrostTest` already
        gates: what this asks is whether the narrowing holds, and a function
        that reached for a hash raises `AttributeError` here instead of
        answering.
        """
        v = _Vectors(name)
        bare = _GroupOnly(v.cs)
        participants = sorted(v.shares)

        self.assertEqual(
            frost.secret_share_split(bare, v.secret, v.coefficients, 3),
            frost.secret_share_split(v.cs, v.secret, v.coefficients, 3),
        )
        commitment = frost.vss_commit(bare, v.secret, v.coefficients)
        self.assertEqual(commitment, frost.vss_commit(v.cs, v.secret, v.coefficients))
        for identifier, share in v.shares.items():
            self.assertTrue(frost.vss_verify(bare, identifier, share, commitment))
        for identifier in participants:
            self.assertEqual(
                frost.derive_interpolating_value(bare, participants, identifier),
                frost.derive_interpolating_value(v.cs, participants, identifier),
            )
        self.assertEqual(
            frost.encode_group_commitment_list(bare, v.commitment_list),
            frost.encode_group_commitment_list(v.cs, v.commitment_list),
        )
        identifiers = [entry.identifier for entry in v.commitment_list]
        hidings = bare.deserialize_elements([e.hiding for e in v.commitment_list])
        bindings = bare.deserialize_elements([e.binding for e in v.commitment_list])
        factors = dict.fromkeys(identifiers, 1)
        self.assertEqual(
            v.cs.serialize_elements(
                frost.compute_group_commitment(
                    bare, identifiers, hidings, bindings, factors
                )
            ),
            v.cs.serialize_elements(
                frost.compute_group_commitment(
                    v.cs, identifiers, hidings, bindings, factors
                )
            ),
        )


class FrostTest(parameterized.TestCase):
    @parameterized.named_parameters(*_PARAMS)
    def test_dealer_reproduces_the_published_shares(self, name: str) -> None:
        v = _Vectors(name)
        got = frost.secret_share_split(v.cs, v.secret, v.coefficients, 3)
        self.assertEqual(dict(got), v.shares)

    @parameterized.named_parameters(*_PARAMS)
    def test_vss_commitment_opens_with_the_group_key(self, name: str) -> None:
        v = _Vectors(name)
        commitment = frost.vss_commit(v.cs, v.secret, v.coefficients)
        self.assertEqual(commitment[0], v.group_public_key)
        for identifier, share in v.shares.items():
            self.assertTrue(frost.vss_verify(v.cs, identifier, share, commitment))
        self.assertFalse(frost.vss_verify(v.cs, 1, v.shares[1] + 1, commitment))

    @parameterized.named_parameters(*_PARAMS)
    def test_interpolation_refuses_an_out_of_range_identifier(self, name: str) -> None:
        # The Lagrange core runs on the scalar field, whose ops abort on an
        # operand outside [0, order) (fractalyze/zk_dtypes#179) — the
        # NonZeroScalar guard surfaces the module's ValueError instead.
        v = _Vectors(name)
        with self.assertRaises(ValueError):
            frost.derive_interpolating_value(v.cs, [1, v.cs.order], v.cs.order)
        with self.assertRaises(ValueError):
            frost.derive_interpolating_value(v.cs, [0, 2], 0)

    @parameterized.named_parameters(*_PARAMS)
    def test_vss_verify_refuses_an_out_of_range_identifier(self, name: str) -> None:
        v = _Vectors(name)
        commitment = frost.vss_commit(v.cs, v.secret, v.coefficients)
        for identifier in (0, v.cs.order):
            with self.assertRaises(ValueError):
                frost.vss_verify(v.cs, identifier, v.shares[1], commitment)

    @parameterized.named_parameters(*_PARAMS)
    def test_round_one_reproduces_nonces_and_commitments(self, name: str) -> None:
        v = _Vectors(name)
        for identifier, entry in v.round_one.items():
            nonces = frost.commit(
                v.cs,
                v.shares[identifier],
                bytes.fromhex(entry["hiding_nonce_randomness"]),
                bytes.fromhex(entry["binding_nonce_randomness"]),
            )
            self.assertEqual(nonces.hiding, v.scalar(entry["hiding_nonce"]))
            self.assertEqual(nonces.binding, v.scalar(entry["binding_nonce"]))
            self.assertEqual(
                nonces.hiding_commitment.hex(), entry["hiding_nonce_commitment"]
            )
            self.assertEqual(
                nonces.binding_commitment.hex(), entry["binding_nonce_commitment"]
            )

    @parameterized.named_parameters(*_PARAMS)
    def test_binding_factors_match_the_published_intermediates(self, name: str) -> None:
        v = _Vectors(name)
        factors = frost.compute_binding_factors(
            v.cs, v.group_public_key, v.commitment_list, v.message
        )
        for identifier, entry in v.round_one.items():
            self.assertEqual(factors[identifier], v.scalar(entry["binding_factor"]))

    @parameterized.named_parameters(*_PARAMS)
    def test_round_two_reproduces_the_signature_shares(self, name: str) -> None:
        v = _Vectors(name)
        for entry in v.round_two:
            identifier = entry["identifier"]
            got = frost.sign_share(
                v.cs,
                identifier,
                v.shares[identifier],
                v.group_public_key,
                v.nonces(identifier),
                v.message,
                v.commitment_list,
            )
            self.assertEqual(got.hex(), entry["sig_share"])

    @parameterized.named_parameters(*_PARAMS)
    def test_aggregate_reproduces_the_published_signature(self, name: str) -> None:
        v = _Vectors(name)
        self.assertEqual(v.aggregate().hex(), v.final_signature)

    @parameterized.named_parameters(*_PARAMS)
    def test_verify_share_names_the_bad_share(self, name: str) -> None:
        v = _Vectors(name)
        for entry, share in zip(v.round_two, v.signature_shares()):
            identifier = entry["identifier"]
            public_share = v.cs.scalar_base_mult(v.shares[identifier])

            def verdict(candidate: bytes) -> bool:
                return frost.verify_share(
                    v.cs,
                    identifier,
                    public_share,
                    candidate,
                    v.commitment_list,
                    v.group_public_key,
                    v.message,
                )

            corrupted = v.cs.serialize_scalar(
                (v.cs.deserialize_scalar(share) + 1) % v.cs.order
            )
            self.assertTrue(verdict(share))
            self.assertFalse(verdict(corrupted))

    @parameterized.named_parameters(*_PARAMS)
    def test_sign_share_refuses_a_swapped_commitment(self, name: str) -> None:
        v = _Vectors(name)
        first = min(v.round_one)
        swapped = [
            (
                frost.Commitment(c.identifier, c.binding, c.hiding)
                if c.identifier == first
                else c
            )
            for c in v.commitment_list
        ]
        with self.assertRaises(ValueError):
            frost.sign_share(
                v.cs,
                first,
                v.shares[first],
                v.group_public_key,
                v.nonces(first),
                v.message,
                swapped,
            )


class Ed25519CrossingTest(absltest.TestCase):
    """The suite surface as the RFC 8032 crossing it delegates to.

    The full malformed-wire gate lives with the delegate
    (`classical/testing/eddsa_test.py`); one malformed row here pins the
    surface's reject-without-raising contract against a future
    non-delegating rewrite without re-owning that coverage.
    """

    def test_the_aggregate_verifies_as_plain_ed25519(self) -> None:
        v = _Vectors("ed25519")
        cs = Ed25519Sha512()
        signature = v.aggregate()
        corrupted = bytearray(signature)
        corrupted[0] ^= 1
        # s = L: RFC 8032 §5.1.7's scalar bound, one past [0, L-1].
        out_of_range = signature[:32] + cs.order.to_bytes(32, "little")
        verdicts = cs.verify(
            _stack(*[v.group_public_key] * 3),
            _stack(*[v.message] * 3),
            _stack(signature, bytes(corrupted), out_of_range),
        )
        self.assertEqual(list(np.asarray(verdicts)), [True, False, False])


class Secp256k1SchnorrTest(absltest.TestCase):
    """The production surface against Appendix B's own form.

    The batched `verify` reshapes the RFC's per-signature check — masked
    wire validation, one two-term combination, a coordinate compare — so
    the RFC's algorithm is kept transcribed beside it and the two are held
    to agree row for row (`docs/reference/testing.md`: test the
    reshaped form against the standard's own form).
    """

    def _accepts(
        self, cs: frost.Ciphersuite, public_key: bytes, message: bytes, signature: bytes
    ) -> bool:
        """RFC 9591 Appendix B's prime-order verification, transcribed.

        `c = H2(R ‖ PK ‖ msg)`; accept iff `[z]B = R + [c]PK` — naively,
        one signature at a time, raising where deserialization MUST fail.
        """
        commitment_bytes = signature[: cs.element_size]
        z = cs.deserialize_scalar(signature[cs.element_size :])
        challenge = cs.h2(commitment_bytes + public_key + message)
        expected = cs.elements_add(
            cs.deserialize_elements([commitment_bytes]),
            cs.elements_scalar_mult(cs.deserialize_elements([public_key]), [challenge]),
        )
        (encoded,) = cs.serialize_elements(expected)
        return cs.scalar_base_mult(z) == encoded

    def _naive(
        self, cs: frost.Ciphersuite, public_key: bytes, message: bytes, signature: bytes
    ) -> bool:
        """The transcription as a verdict: a MUST-abort is a rejection.

        The batch surface answers `False` where the RFC's deserialization
        aborts, so agreement is checked on the verdict both express.
        """
        try:
            return self._accepts(cs, public_key, message, signature)
        except ValueError:
            return False

    def test_the_production_surface_agrees_with_appendix_b(self) -> None:
        v = _Vectors("secp256k1")
        cs = Secp256k1Sha256()
        signature = v.aggregate()

        def flipped(data: bytes, index: int) -> bytes:
            corrupted = bytearray(data)
            corrupted[index] ^= 1
            return bytes(corrupted)

        cases = [
            (v.group_public_key, v.message, signature),
            # One flip per wire field: R's prefix, R's x, z, the key, and
            # the message — rejections the transcription reaches as a
            # failed equation or a MUST-abort, and the batch as `False`.
            (v.group_public_key, v.message, flipped(signature, 0)),
            (v.group_public_key, v.message, flipped(signature, 1)),
            (v.group_public_key, v.message, flipped(signature, 64)),
            (flipped(v.group_public_key, 32), v.message, signature),
            (v.group_public_key, flipped(v.message, 0), signature),
        ]
        verdicts = cs.verify(
            _stack(*(pk for pk, _, _ in cases)),
            _stack(*(msg for _, msg, _ in cases)),
            _stack(*(sig for _, _, sig in cases)),
        )
        expected = [self._naive(cs, *case) for case in cases]
        self.assertEqual(list(map(bool, np.asarray(verdicts))), expected)
        # The agreement must not be vacuous: the published case accepts,
        # every corruption rejects.
        self.assertEqual(expected, [True] + [False] * (len(cases) - 1))

    def test_malformed_wire_rows_ride_to_false_without_raising(self) -> None:
        v = _Vectors("secp256k1")
        cs = Secp256k1Sha256()
        signature = v.aggregate()
        p = cs.curve.p
        cases = [
            # The accepted row the malformed ones sit beside — a surface
            # that rejects everything does not pass this gate.
            (v.group_public_key, signature),
            # z = n: the scalar bound, one past the RFC's [0, n-1].
            (v.group_public_key, signature[:33] + cs.order.to_bytes(32, "big")),
            # An element prefix SEC 1 §2.3.4 does not define.
            (v.group_public_key, b"\x05" + signature[1:]),
            # R's x-coordinate at p: out of the field's range.
            (
                v.group_public_key,
                signature[:1] + p.to_bytes(32, "big") + signature[33:],
            ),
            # R's x on no point: x = 0 lies on neither SEC curve here.
            (v.group_public_key, signature[:1] + bytes(32) + signature[33:]),
            # A key that is wrong twice over: bad prefix, x out of range.
            (b"\xff" * 33, signature),
        ]
        verdicts = cs.verify(
            _stack(*(pk for pk, _ in cases)),
            _stack(*([v.message] * len(cases))),
            _stack(*(sig for _, sig in cases)),
        )
        self.assertEqual(
            list(np.asarray(verdicts)), [True] + [False] * (len(cases) - 1)
        )


if __name__ == "__main__":
    absltest.main()
