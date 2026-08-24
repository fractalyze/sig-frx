# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""ML-DSA's assembly against FIPS 204's own algorithms, and the seam's obligations.

`ml_dsa.py` reshapes the standard in three places: the rejection loop's bounds are
array reductions rather than scalar comparisons, verification runs as one `vmap`ped
computation over a batch, and every polynomial operation is delegated to the field
dtype and the transform opcode. `fips204_reference` is Algorithms 6, 7 and 8 written
the way the standard writes them — Python integers, one coefficient at a time, a
`while` that squeezes `hashlib` — and key generation and signing are required to
reproduce its bytes exactly at every parameter set
([`conventions.md`](../../../../docs/reference/conventions.md)).

Byte equality with the reference is what makes the loop checkable at all. A
signature agrees only if the two implementations rejected the same candidates in
the same order and left the counter `κ` in the same place, so the whole of
Algorithm 7's control flow is pinned by comparing its output.

What none of it can catch is a misreading of the standard shared by both sides,
since one author wrote both. The published vectors close that gap and are
[#22](https://github.com/fractalyze/sig-frx/issues/22); the derived sizes are
checked against Table 2's published columns here, which is the one part of that
gap a transcription cannot cover for itself.

The behavioural cases — the tamper verdicts, the malformed hint, the context, the
traced batch — run at ML-DSA-44 alone. None of them depends on a parameter set,
each costs a key generation and a signing on the concrete path, and the agreement
tests above already drive all three sets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from frx import Array

from sig_frx import prehash
from sig_frx.context import MAX_SIZE as MAX_CONTEXT_SIZE
from sig_frx.lattice import rejection
from sig_frx.lattice.mldsa import ml_dsa
from sig_frx.lattice.mldsa.testing import fips204_reference as ref
from sig_frx.signature import Signature

_DEFAULT = "ML-DSA-44"
_MESSAGE = b"the reshaped form and the standard's own form"
# A non-empty context, so the pre-hash cases pin the length byte and the ordering
# of `ctx` against the OID rather than only the digest.
_CONTEXT = b"sig-frx"
_BATCH = 3
_SEED = bytes(range(32))
# A `rnd` that is not the deterministic variant's zeros, so a signature that
# ignored it would still match the deterministic one and pass.
_RANDOMIZER = bytes(range(100, 132))

# Table 2's published sizes, which the parameter record derives rather than
# stores: a derivation nobody checks against the standard is a formula, not a size.
_PUBLISHED_SIZES = {
    "ML-DSA-44": (1312, 2560, 2420),
    "ML-DSA-65": (1952, 4032, 3309),
    "ML-DSA-87": (2592, 4896, 4627),
}

_PARAMETER_SETS = tuple(
    {"testcase_name": name.replace("-", "_"), "name": name}
    for name in ml_dsa.PARAMETER_SETS
)

_PRE_HASHES = tuple(
    {"testcase_name": name.replace("-", "_"), "name": name} for name in prehash.BY_NAME
)


def _reference_params(name: str) -> dict[str, int]:
    """Table 1 for `name`, from the reference's copy — the tests use one table."""
    return ref.PARAMETER_SETS[name.replace("-", "_")]


def _pure_message(message: bytes, context: bytes = b"") -> bytes:
    """`M′` — Algorithm 2 line 10, which the internal interface takes as given."""
    return bytes([0, len(context)]) + context + message


@dataclass(frozen=True)
class _Signed:
    """One parameter set's key pair and its deterministic signature over `_MESSAGE`."""

    scheme: ml_dsa.MlDsa
    public_key: Array
    secret_key: Array
    signature: Array


@lru_cache(maxsize=None)
def _signed(name: str) -> _Signed:
    """Cached because keygen and signing are the concrete path — seconds, not ms."""
    scheme = ml_dsa.named(name, deterministic=True)
    public_key, secret_key = scheme.keygen(np.frombuffer(_SEED, dtype=np.uint8))
    return _Signed(
        scheme=scheme,
        public_key=public_key,
        secret_key=secret_key,
        signature=scheme.sign(secret_key, np.frombuffer(_MESSAGE, dtype=np.uint8)),
    )


def _batch(signed: _Signed) -> tuple[Array, Array, Array]:
    """`_BATCH` copies of one valid triple — the shape `verify` takes."""
    message = np.frombuffer(_MESSAGE, dtype=np.uint8)
    return (
        fnp.asarray(np.tile(np.asarray(signed.public_key), (_BATCH, 1))),
        fnp.asarray(np.tile(message, (_BATCH, 1))),
        fnp.asarray(np.tile(np.asarray(signed.signature), (_BATCH, 1))),
    )


def _corrupt(batch: Array, entry: int, index: int = 0) -> Array:
    """Flip one bit of one entry, leaving every other entry intact."""
    values = np.array(np.asarray(batch))
    values[entry, index] ^= 1
    return fnp.asarray(values)


def _verdicts(values: Array) -> list[bool]:
    """One verdict per entry as Python bools, which read better in a failure."""
    return [bool(verdict) for verdict in values]


class ParameterSetTest(parameterized.TestCase):
    """Table 1 as this module holds it, and the sizes it derives from it."""

    def test_table_1_matches_the_reference_s_copy(self) -> None:
        """One table per side, compared — not one table transcribed twice."""
        got = {
            name.replace("-", "_"): asdict(params)
            for name, params in ml_dsa.PARAMETER_SETS.items()
        }
        self.assertEqual(got, ref.PARAMETER_SETS)

    @parameterized.named_parameters(*_PARAMETER_SETS)
    def test_derived_sizes_match_the_published_ones(self, name: str) -> None:
        """FIPS 204 Table 2 — the columns the record computes rather than stores."""
        params = ml_dsa.PARAMETER_SETS[name]
        self.assertEqual(
            (params.public_key_size, params.secret_key_size, params.signature_size),
            _PUBLISHED_SIZES[name],
        )

    @parameterized.named_parameters(
        ("ML_DSA_44", "ML-DSA-44", 78),
        ("ML_DSA_65", "ML-DSA-65", 196),
        ("ML_DSA_87", "ML-DSA-87", 120),
    )
    def test_beta_matches_table_1_s_own_column(self, name: str, published: int) -> None:
        self.assertEqual(ml_dsa.PARAMETER_SETS[name].beta, published)

    def test_a_gamma2_that_does_not_divide_is_an_error(self) -> None:
        """`w1Encode` packs at `bitlen((q−1)/(2γ2) − 1)`, so a γ2 that does not
        divide changes the encoding without changing anything else visible."""
        with self.assertRaisesRegex(ValueError, "must divide"):
            ml_dsa.MlDsaParams(
                k=4, ell=4, eta=2, tau=39, lam=128, gamma1=1 << 17, gamma2=95, omega=80
            )

    def test_an_unknown_parameter_set_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "ML-DSA-99"):
            ml_dsa.named("ML-DSA-99")


class ReferenceAgreementTest(parameterized.TestCase):
    """Key generation and signing, byte for byte against the standard's own loops."""

    @parameterized.named_parameters(*_PARAMETER_SETS)
    def test_keygen_reproduces_the_reference_key_pair(self, name: str) -> None:
        want_pk, want_sk = ref.keygen_internal(_SEED, _reference_params(name))
        signed = _signed(name)
        self.assertEqual(bytes(np.asarray(signed.public_key)), want_pk)
        self.assertEqual(bytes(np.asarray(signed.secret_key)), want_sk)

    @parameterized.named_parameters(*_PARAMETER_SETS)
    def test_deterministic_signing_reproduces_the_reference_signature(
        self, name: str
    ) -> None:
        signed = _signed(name)
        want = ref.sign_internal(
            bytes(np.asarray(signed.secret_key)),
            _pure_message(_MESSAGE),
            bytes(32),
            _reference_params(name),
        )
        self.assertEqual(bytes(np.asarray(signed.signature)), want)

    @parameterized.named_parameters(*_PARAMETER_SETS)
    def test_hedged_signing_reproduces_the_reference_signature(self, name: str) -> None:
        """`rnd` is the only difference between the two variants (§3.4), so a
        signature that dropped it would still match the deterministic one."""
        signed = _signed(name)
        want = ref.sign_internal(
            bytes(np.asarray(signed.secret_key)),
            _pure_message(_MESSAGE),
            _RANDOMIZER,
            _reference_params(name),
        )
        got = ml_dsa.named(name).sign(
            signed.secret_key,
            np.frombuffer(_MESSAGE, dtype=np.uint8),
            randomness=np.frombuffer(_RANDOMIZER, dtype=np.uint8),
        )
        self.assertEqual(bytes(np.asarray(got)), want)
        self.assertNotEqual(bytes(np.asarray(got)), bytes(np.asarray(signed.signature)))

    @parameterized.named_parameters(*_PARAMETER_SETS)
    def test_the_reference_verifier_accepts_what_the_module_signed(
        self, name: str
    ) -> None:
        """The other direction: the reference verifier shares no code with
        `verify`, so an agreement here is the assembly checked from both ends."""
        signed = _signed(name)
        self.assertTrue(
            ref.verify_internal(
                bytes(np.asarray(signed.public_key)),
                _pure_message(_MESSAGE),
                bytes(np.asarray(signed.signature)),
                _reference_params(name),
            )
        )


class SigningTest(absltest.TestCase):
    """The two variants, and what each refuses."""

    def test_deterministic_signing_is_reproducible(self) -> None:
        signed = _signed(_DEFAULT)
        again = signed.scheme.sign(
            signed.secret_key, np.frombuffer(_MESSAGE, dtype=np.uint8)
        )
        self.assertEqual(bytes(np.asarray(again)), bytes(np.asarray(signed.signature)))

    def test_a_hedged_instance_refuses_to_draw_its_own_randomness(self) -> None:
        """An implicit draw is how a scheme stops being reproducible against its
        KATs, and falling back to zeros would produce the deterministic signature
        — which verifies, so nothing downstream would notice."""
        scheme = ml_dsa.named(_DEFAULT, deterministic=False)
        with self.assertRaisesRegex(ValueError, "rnd"):
            scheme.sign(
                _signed(_DEFAULT).secret_key, np.frombuffer(_MESSAGE, dtype=np.uint8)
            )

    def test_a_wrong_sized_randomizer_is_an_error(self) -> None:
        scheme = ml_dsa.named(_DEFAULT, deterministic=False)
        with self.assertRaisesRegex(ValueError, "rnd"):
            scheme.sign(
                _signed(_DEFAULT).secret_key,
                np.frombuffer(_MESSAGE, dtype=np.uint8),
                randomness=np.zeros(31, dtype=np.uint8),
            )

    def test_a_wrong_sized_secret_key_is_an_error(self) -> None:
        signed = _signed(_DEFAULT)
        with self.assertRaisesRegex(ValueError, "secret key is"):
            signed.scheme.sign(
                signed.secret_key[:-1], np.frombuffer(_MESSAGE, dtype=np.uint8)
            )

    def test_a_wrong_sized_seed_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "keygen takes"):
            ml_dsa.named(_DEFAULT).keygen(np.zeros(31, dtype=np.uint8))


class VerificationTest(absltest.TestCase):
    """The batch axis, and the three corruptions the standard is defined against."""

    def test_the_seam_accepts_the_implementation(self) -> None:
        self.assertIsInstance(ml_dsa.named(_DEFAULT), Signature)

    def test_verify_returns_one_verdict_per_batch_entry(self) -> None:
        signed = _signed(_DEFAULT)
        verdicts = signed.scheme.verify(*_batch(signed))
        self.assertEqual(verdicts.shape, (_BATCH,))
        self.assertEqual(verdicts.dtype, fnp.bool_)
        self.assertTrue(bool(fnp.all(verdicts)))

    def test_verify_rejects_only_the_tampered_signature(self) -> None:
        signed = _signed(_DEFAULT)
        keys, messages, signatures = _batch(signed)
        self.assertEqual(
            _verdicts(signed.scheme.verify(keys, messages, _corrupt(signatures, 1))),
            [True, False, True],
        )

    def test_verify_rejects_only_the_tampered_message(self) -> None:
        signed = _signed(_DEFAULT)
        keys, messages, signatures = _batch(signed)
        self.assertEqual(
            _verdicts(signed.scheme.verify(keys, _corrupt(messages, 2), signatures)),
            [True, True, False],
        )

    def test_verify_rejects_only_the_tampered_public_key(self) -> None:
        signed = _signed(_DEFAULT)
        keys, messages, signatures = _batch(signed)
        # Past `rho`, so the corruption lands in `t1` rather than in the seed the
        # matrix is expanded from — the half of the key the hint corrects around.
        self.assertEqual(
            _verdicts(
                signed.scheme.verify(_corrupt(keys, 0, index=64), messages, signatures)
            ),
            [False, True, True],
        )

    def test_the_whole_batch_survives_jit_as_one_call(self) -> None:
        """`conventions.md` puts the `jit` boundary around the batch: one traced
        computation for `B` signatures, not a driver loop around a traced body."""
        signed = _signed(_DEFAULT)
        verdicts = frx.jit(signed.scheme.verify)(*_batch(signed))
        self.assertEqual(_verdicts(verdicts), [True] * _BATCH)

    def test_a_wrong_length_input_verifies_as_false(self) -> None:
        """§3.6.2: a signature or a public key of any other length is a verdict and
        not an error — failing to check it interferes with strong unforgeability."""
        signed = _signed(_DEFAULT)
        keys, messages, signatures = _batch(signed)
        self.assertEqual(
            _verdicts(signed.scheme.verify(keys, messages, signatures[:, :-1])),
            [False] * _BATCH,
        )
        self.assertEqual(
            _verdicts(signed.scheme.verify(keys[:, :-1], messages, signatures)),
            [False] * _BATCH,
        )

    def test_a_batch_that_does_not_line_up_is_an_error(self) -> None:
        signed = _signed(_DEFAULT)
        keys, messages, signatures = _batch(signed)
        with self.assertRaisesRegex(ValueError, "one signature per public key"):
            signed.scheme.verify(keys, messages, signatures[:-1])
        with self.assertRaisesRegex(ValueError, "one message per public key"):
            signed.scheme.verify(keys, messages[:-1], signatures)


class MalformedHintTest(absltest.TestCase):
    """Algorithm 21's rejections, which reach `bool[B]` rather than raising.

    FIPS 204 §D.2 records that the draft standard omitted this check and that
    omitting it makes the scheme not strongly existentially unforgeable, so each
    of these is a forgery a verifier that dropped `ok` would accept. They are
    edits to an otherwise valid signature, so nothing but the hint's encoding
    distinguishes them from the accepted case above.
    """

    def _rejects(self, edits: dict[int, int]) -> None:
        """Overwrite entry 1's hint bytes and require exactly that entry to fail.

        Offsets are from the start of the hint — the last `ω + k` bytes, which is
        where Algorithm 20 writes `positions ‖ cuts`.
        """
        signed = _signed(_DEFAULT)
        keys, messages, signatures = _batch(signed)
        params = signed.scheme.params
        values = np.array(np.asarray(signatures))
        start = values.shape[1] - (params.omega + params.k)
        for offset, byte in edits.items():
            values[1, start + offset] = byte
        self.assertEqual(
            _verdicts(signed.scheme.verify(keys, messages, fnp.asarray(values))),
            [True, False, True],
        )

    def test_a_cut_past_omega_is_rejected(self) -> None:
        """Line 4's range check: a cut past `ω` claims more positions than the
        encoding holds."""
        omega = _signed(_DEFAULT).scheme.params.omega
        self._rejects({omega: omega + 1})

    def test_a_cut_that_goes_backwards_is_rejected(self) -> None:
        """The other half of line 4: the per-polynomial ends are non-decreasing,
        so the second cut may not fall below the first."""
        omega = _signed(_DEFAULT).scheme.params.omega
        self._rejects({omega: omega, omega + 1: 0})

    def test_a_nonzero_byte_in_the_unused_tail_is_rejected(self) -> None:
        """Lines 16 to 18: the tail past the last cut is zero, which is what makes
        the encoding of a given hint unique."""
        signed = _signed(_DEFAULT)
        params = signed.scheme.params
        signature = np.asarray(signed.signature)
        last_cut = int(signature[-1])
        self.assertLess(
            last_cut,
            params.omega,
            "the fixture's hint fills every slot, so it has no tail to corrupt",
        )
        self._rejects({last_cut: 1})


class ContextTest(absltest.TestCase):
    """The application context, which goes into the message that gets signed."""

    def test_a_signature_does_not_verify_under_another_context(self) -> None:
        signed = _signed(_DEFAULT)
        keys, messages, signatures = _batch(signed)
        verdicts = signed.scheme.verify(
            keys, messages, signatures, context=np.array([7], dtype=np.uint8)
        )
        self.assertEqual(_verdicts(verdicts), [False] * _BATCH)

    def test_a_context_the_signer_used_verifies(self) -> None:
        signed = _signed(_DEFAULT)
        keys, messages, _ = _batch(signed)
        context = np.array([7, 8], dtype=np.uint8)
        signature = signed.scheme.sign(
            signed.secret_key,
            np.frombuffer(_MESSAGE, dtype=np.uint8),
            context=context,
        )
        batch = fnp.asarray(np.tile(np.asarray(signature), (_BATCH, 1)))
        self.assertEqual(
            _verdicts(signed.scheme.verify(keys, messages, batch)), [False] * _BATCH
        )
        verdicts = signed.scheme.verify(keys, messages, batch, context=context)
        self.assertEqual(_verdicts(verdicts), [True] * _BATCH)

    def test_a_context_longer_than_its_length_byte_is_an_error(self) -> None:
        signed = _signed(_DEFAULT)
        keys, messages, signatures = _batch(signed)
        signed.scheme.verify(
            keys,
            messages,
            signatures,
            context=np.zeros(MAX_CONTEXT_SIZE, dtype=np.uint8),
        )
        with self.assertRaisesRegex(ValueError, "context string is at most"):
            signed.scheme.verify(
                keys, messages, signatures, context=np.zeros(256, dtype=np.uint8)
            )


class InternalInterfaceTest(absltest.TestCase):
    """§6's pair, which signs `M′` as given — a different operation from §5's."""

    def _one(self, signature: Array) -> list[bool]:
        """The verdict on a batch of one, which is what `B = 1` means here."""
        signed = _signed(_DEFAULT)
        return _verdicts(
            signed.scheme.verify_internal(
                signed.public_key[None],
                fnp.asarray(np.frombuffer(_MESSAGE, dtype=np.uint8))[None],
                fnp.asarray(signature)[None],
            )
        )

    def test_the_internal_pair_round_trips_without_the_context_framing(self) -> None:
        signed = _signed(_DEFAULT)
        signature = signed.scheme.sign_internal(
            signed.secret_key, np.frombuffer(_MESSAGE, dtype=np.uint8)
        )
        self.assertEqual(self._one(signature), [True])

    def test_an_external_signature_is_not_an_internal_one(self) -> None:
        """The framing is the separation: `sign` signs `M′` and `sign_internal`
        signs `M`, so one verifying as the other would mean the domain separator
        was not part of what got signed."""
        self.assertEqual(self._one(_signed(_DEFAULT).signature), [False])


class PreHashTest(parameterized.TestCase):
    """§5.4's pair, which signs a digest under an OID that names the function.

    Every function this repo can compute is driven, because what differs between
    them is exactly the two constants no round trip reads back — the OID and, for a
    XOF, how many bytes it is squeezed for. The reference's `switch PH` is a second
    transcription of that table, so an agreement here is the pair checked against
    the standard rather than against itself.
    """

    @parameterized.named_parameters(*_PRE_HASHES)
    def test_signing_reproduces_the_reference_signature(self, name: str) -> None:
        signed = _signed(_DEFAULT)
        got = signed.scheme.hash_sign(
            signed.secret_key,
            np.frombuffer(_MESSAGE, dtype=np.uint8),
            prehash.BY_NAME[name](),
            context=np.frombuffer(_CONTEXT, dtype=np.uint8),
        )
        want = ref.hash_sign(
            bytes(np.asarray(signed.secret_key)),
            _MESSAGE,
            bytes(32),
            _CONTEXT,
            name,
            _reference_params(_DEFAULT),
        )
        self.assertEqual(bytes(np.asarray(got)), want)

    @parameterized.named_parameters(*_PRE_HASHES)
    def test_the_pair_round_trips_over_a_batch(self, name: str) -> None:
        signed = _signed(_DEFAULT)
        pre_hash = prehash.BY_NAME[name]()
        signature = signed.scheme.hash_sign(
            signed.secret_key, np.frombuffer(_MESSAGE, dtype=np.uint8), pre_hash
        )
        public_keys, messages, _ = _batch(signed)
        signatures = fnp.asarray(np.tile(np.asarray(signature), (_BATCH, 1)))
        self.assertEqual(
            _verdicts(
                signed.scheme.hash_verify(public_keys, messages, signatures, pre_hash)
            ),
            [True] * _BATCH,
        )

    def test_a_pre_hash_signature_is_not_a_pure_one(self) -> None:
        """The separator is the separation, and §5.4 introduces it for this: the
        pure variant's domain byte is zero and the pre-hash variant's is one, so
        one verifying as the other would mean neither was signed."""
        signed = _signed(_DEFAULT)
        message = np.frombuffer(_MESSAGE, dtype=np.uint8)
        signature = signed.scheme.hash_sign(
            signed.secret_key, message, prehash.sha2_256()
        )
        self.assertEqual(
            _verdicts(
                signed.scheme.verify(
                    signed.public_key[None], fnp.asarray(message)[None], signature[None]
                )
            ),
            [False],
        )

    def test_a_signature_does_not_verify_under_another_pre_hash_function(self) -> None:
        """The OID is inside `M′`, so the function a signature answers for is part
        of what was signed. Two functions of the same digest length is the case
        that only the OID separates."""
        signed = _signed(_DEFAULT)
        message = np.frombuffer(_MESSAGE, dtype=np.uint8)
        signature = signed.scheme.hash_sign(
            signed.secret_key, message, prehash.sha2_256()
        )
        self.assertEqual(
            _verdicts(
                signed.scheme.hash_verify(
                    signed.public_key[None],
                    fnp.asarray(message)[None],
                    signature[None],
                    prehash.sha3_256(),
                )
            ),
            [False],
        )

    def test_the_reference_verifier_accepts_what_the_module_signed(self) -> None:
        signed = _signed(_DEFAULT)
        signature = signed.scheme.hash_sign(
            signed.secret_key,
            np.frombuffer(_MESSAGE, dtype=np.uint8),
            prehash.shake256(),
            context=np.frombuffer(_CONTEXT, dtype=np.uint8),
        )
        self.assertTrue(
            ref.hash_verify(
                bytes(np.asarray(signed.public_key)),
                _MESSAGE,
                bytes(np.asarray(signature)),
                _CONTEXT,
                "SHAKE-256",
                _reference_params(_DEFAULT),
            )
        )

    def test_the_functions_the_repo_computes_are_the_ones_it_states(self) -> None:
        """`BY_NAME` and the reference's table are written apart and have to hold
        the same names, since a function present in one alone would be either an
        untested constructor or an unreachable transcription."""
        self.assertEqual(set(prehash.BY_NAME), set(ref.PRE_HASHES))
        for name, build in prehash.BY_NAME.items():
            with self.subTest(name):
                _, oid, digest_size = ref.PRE_HASHES[name]
                pre_hash = build()
                self.assertEqual(pre_hash.oid.hex(), oid)
                self.assertEqual(pre_hash.byte_hash.digest_size, digest_size)


class InstanceTest(absltest.TestCase):
    """Value equality, which the seam requires of every implementation."""

    def test_equal_instances_hash_alike(self) -> None:
        """Identity equality does not error; it re-traces the enclosing `jit` zone
        for every freshly built instance, which surfaces as a slow call."""
        self.assertEqual(ml_dsa.named(_DEFAULT), ml_dsa.named(_DEFAULT))
        self.assertEqual(hash(ml_dsa.named(_DEFAULT)), hash(ml_dsa.named(_DEFAULT)))

    def test_the_variant_and_the_parameter_set_are_part_of_the_value(self) -> None:
        self.assertNotEqual(
            ml_dsa.named(_DEFAULT, deterministic=True),
            ml_dsa.named(_DEFAULT, deterministic=False),
        )
        self.assertNotEqual(ml_dsa.named("ML-DSA-44"), ml_dsa.named("ML-DSA-65"))


class IterationBoundTest(absltest.TestCase):
    """The signing loop's cap, which Appendix C leaves optional and this derives.

    It is `rejection.budget` asked for one success from one attempt per trial, so
    what is worth pinning here is the number that comes back rather than a second
    copy of the tail: the same two-sided check the samplers' budget gets, written
    against the acceptance rate this scheme claims.
    """

    def test_is_the_smallest_bound_that_meets_the_margin(self) -> None:
        bound = ml_dsa._MAX_ITERATIONS
        margin = rejection.LOG2_SHORTFALL
        num, den = ml_dsa._WORST_ACCEPTANCE
        self.assertLessEqual((den - num) ** bound << margin, den**bound)
        self.assertGreater((den - num) ** (bound - 1) << margin, den ** (bound - 1))

    def test_the_acceptance_rate_is_table_1_s_worst_repetition_count(self) -> None:
        """5.1 repetitions at ML-DSA-65 — the fraction, so no float rounds it."""
        num, den = ml_dsa._WORST_ACCEPTANCE
        self.assertEqual(den / num, 5.1)


if __name__ == "__main__":
    absltest.main()
