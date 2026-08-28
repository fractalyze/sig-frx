# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""What Falcon's own structure makes rejectable, and the seam's shape.

The published verdicts, the generic tampering pass and the batch axis are
[`falcon_kat_test`](falcon_kat_test.py)'s, through the shared harness. What is
here is the half `testing.md` asks a scheme to add on top: "every rejection
its own structure makes possible that a generic bit flip would not reach."

For Falcon that is the encoding. The harness moves a bit in each of the three
inputs; it does not know that byte 0 of a signature is a format nibble over a
degree, that the tail past the last terminator is padding whose non-zero
spelling is a second encoding of the same `s`, or that fourteen bits can hold a
public key coefficient the modulus cannot. Each of those is malleability — same
message, same key, different bytes, still valid — and each gets a case here,
with the unmutated control asserted alongside, because a rejection that would
also reject the genuine signature proves nothing.

The rest is the seam rather than the standard: the derived sizes against Table
3.3, a wrong rank raising where a wrong length is a verdict, the refusal of an
application context Falcon does not define, and Algorithm 4 producing a key pair
that Algorithm 6's equation and Algorithm 5's bounds both hold for.

Falcon still has no `sign` here, so there is no round trip to lean on — which is
the right way round: a scheme verifying its own signatures is the
self-consistency `testing.md` says is not evidence. `keygen` is checked against
the standard's own conditions on its output instead, and against nothing it
produced itself.
"""

from __future__ import annotations

import functools
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from sig_frx.lattice.falcon import encoding, falcon, keygen
from sig_frx.lattice.falcon.testing import falcon_reference as ref
from sig_frx.lattice.falcon.testing.falcon_vectors import VECTORS
from sig_frx.signature import Signature


@functools.lru_cache(maxsize=None)
def _key_pair() -> tuple[bytes, bytes]:
    """The `Falcon-512` pair every case here that needs a real key runs on.

    Cached because it is the target's single largest cost — about 45 s, nearly
    all of it Algorithm 6's solve — and two cases want the same pair. A key is a
    pure function of its seed, so a second call would buy nothing and would put
    this target past half its budget on the leg that decides it.
    """
    scheme = falcon.named("Falcon-512")
    public, secret = scheme.keygen(np.arange(scheme.seed_size, dtype=np.uint8))
    return (
        bytes(np.asarray(public, dtype=np.uint8)),
        bytes(np.asarray(secret, dtype=np.uint8)),
    )


_PARAMETER_SETS = ref.parameter_cases()


def _bytes(blob: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(blob), dtype=np.uint8)


def _verdict(
    name: str,
    *,
    public_key: np.ndarray | None = None,
    signature: np.ndarray | None = None,
) -> bool:
    """The first published case of `name`, with the named input replaced.

    Every case below varies exactly one of the three inputs, so spelling the
    other two at each site is what the overrides remove.
    """
    vector = VECTORS[name][0]
    verdict = falcon.named(name).verify(
        (_bytes(vector.public_key) if public_key is None else public_key)[None, :],
        _bytes(vector.message)[None, :],
        (_bytes(vector.signature) if signature is None else signature)[None, :],
    )
    return bool(np.asarray(verdict)[0])


def _signature(name: str) -> np.ndarray:
    """A writable copy — `frombuffer` hands back a read-only view."""
    return _bytes(VECTORS[name][0].signature).copy()


class Sizes(parameterized.TestCase):
    """Table 3.3's four lengths, against the formulas §3.11 states them as."""

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_derived_sizes_match_the_table(self, name: str, **params: Any) -> None:
        scheme = falcon.named(name)
        self.assertEqual(scheme.public_key_size, params["public_key_size"])
        self.assertEqual(scheme.secret_key_size, params["secret_key_size"])
        self.assertEqual(scheme.signature_max_size, params["signature_size"])
        # Transcribed rather than derived, so it is the one value a typo can
        # reach — and a bound mistyped upward widens what Falcon accepts without
        # failing a single published vector.
        self.assertEqual(scheme.params.squared_norm_bound, params["squared_norm_bound"])

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


class Encoding(parameterized.TestCase):
    """The rejections Falcon's own encoding makes possible.

    A generic bit flip — which the shared harness already does across all three
    inputs — lands wherever it lands. These reach the format nibble, the padding
    tail, and a coefficient the modulus cannot hold, none of which it knows to
    aim at.
    """

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_the_unmutated_case_is_accepted(self, name: str, **params: Any) -> None:
        """The control every rejection below is measured against."""
        del params
        self.assertTrue(_verdict(name))

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_the_uncompressed_header_is_refused(self, name: str, **params: Any) -> None:
        """§3.11.3's `cc = 10` is a different length; this decoder takes `01`."""
        del params
        signature = _signature(name)
        signature[0] = (signature[0] & 0x0F) | 0x50
        self.assertFalse(_verdict(name, signature=signature))

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_a_wrong_degree_in_the_header_is_refused(
        self, name: str, **params: Any
    ) -> None:
        """The nibble names `log2(n)`; the other set's signature is not this one's."""
        del params
        signature = _signature(name)
        signature[0] ^= 0x01
        self.assertFalse(_verdict(name, signature=signature))

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_a_nonzero_padding_byte_is_refused(self, name: str, **params: Any) -> None:
        """Malleability: the same `s` with a different tail would verify."""
        del params
        signature = _signature(name)
        self.assertEqual(int(signature[-1]), 0)
        signature[-1] = 1
        self.assertFalse(_verdict(name, signature=signature))

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_a_public_key_coefficient_at_or_above_q_is_refused(
        self, name: str, **params: Any
    ) -> None:
        """Fourteen bits hold 16383 against `q = 12289`, so this is representable."""
        n = params["n"]
        bits: list[int] = [0] * 8
        bits[4:8] = [(n.bit_length() - 1) >> shift & 1 for shift in range(3, -1, -1)]
        for index in range(n):
            value = ref.Q if index == 0 else 0
            bits.extend((value >> shift) & 1 for shift in range(13, -1, -1))
        forged = np.frombuffer(ref.bytes_of(bits), dtype=np.uint8)
        self.assertLen(forged, params["public_key_size"])
        self.assertFalse(_verdict(name, public_key=forged))


class CompilationUnit(parameterized.TestCase):
    """`jit` belongs around the batch, which is a claim only this file can make.

    The shared harness calls `verify` eagerly, so nothing it derives would notice
    a body that only traces one entry at a time.
    """

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_the_whole_batch_traces_as_one_computation(
        self, name: str, **params: Any
    ) -> None:
        del params
        vector = VECTORS[name][0]
        good = _bytes(vector.signature)
        broken = good.copy()
        broken[60] ^= 1
        verdict = frx.jit(falcon.named(name).verify)(
            np.stack([_bytes(vector.public_key)] * 3),
            np.stack([_bytes(vector.message)] * 3),
            np.stack([good, broken, good]),
        )
        np.testing.assert_array_equal(np.asarray(verdict), [True, False, True])


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


class Keygen(parameterized.TestCase):
    """Algorithm 4 end to end — the first Falcon key this repo produces itself.

    One `keygen` call and every assertion taken off it, because the call is 46
    seconds at `n = 512` and nearly all of that is one NTRU solve. Asking four
    questions of one key is the shape `DescentTest` uses for the same reason.
    """

    def test_a_generated_key_satisfies_the_equation_and_the_bounds(self) -> None:
        """#26's acceptance criteria, on a key pair nothing published.

        The published-key case in [`keygen_test`](keygen_test.py) runs the
        reference implementation's key through this repo's decoders; this runs
        the repo's own key through its own, which is a weaker claim on its own
        and a necessary one — the encoders agreeing with the decoders says
        nothing about whether the *trapdoor* is a trapdoor, and the equation
        below is what does.

        Both of Algorithm 5's checks are re-asserted on the output rather than
        trusted from inside the loop. That is not redundant: the loop tests them
        on the pair it drew, and this tests them on the pair that came back
        through two encodings.
        """
        scheme = falcon.named("Falcon-512")
        n = scheme.params.n
        public_bytes, secret_bytes = _key_pair()
        public = np.frombuffer(public_bytes, dtype=np.uint8)
        secret = np.frombuffer(secret_bytes, dtype=np.uint8)
        self.assertLen(public, scheme.public_key_size)
        self.assertLen(secret, scheme.secret_key_size)

        f, g, big_f, ok = encoding.sk_decode(np.asarray(secret), n)
        self.assertTrue(bool(np.asarray(ok)))
        f, g, big_f = (np.asarray(value) for value in (f, g, big_f))

        big_g = np.asarray(keygen.recover_g(f, g, big_f))
        self.assertEqual(
            ref.ntru_equation(f.tolist(), g.tolist(), big_f.tolist(), big_g.tolist()),
            [ref.Q] + [0] * (n - 1),
        )
        self.assertTrue(bool(np.asarray(keygen.invertible(f))), "Algorithm 5 line 7")
        self.assertLessEqual(
            float(np.asarray(keygen.gram_schmidt_squared_norm(f, g))),
            keygen.GRAM_SCHMIDT_BOUND,
            "Algorithm 5 line 10",
        )

        h, key_ok = encoding.pk_decode(np.asarray(public), n)
        self.assertTrue(bool(np.asarray(key_ok)))
        np.testing.assert_array_equal(
            np.asarray(keygen.public_key(f, g)), np.asarray(h)
        )

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_a_seed_of_the_wrong_size_is_refused(
        self, name: str, **params: Any
    ) -> None:
        del params
        scheme = falcon.named(name)
        for size in (0, scheme.seed_size - 1, scheme.seed_size + 1):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, "seed"):
                    scheme.keygen(np.zeros(size, dtype=np.uint8))

    def test_a_traced_seed_is_refused(self) -> None:
        """There is nothing to trace: the restart loop's trip count is data.

        The refusal is `np.asarray` on a tracer rather than a check written
        here, which is the namespace rule's own note that the conversion *onto*
        the host announces itself
        ([`conventions.md`](../../../../docs/reference/conventions.md)). Pinned
        so that a later change putting the seed on the device would fail here
        rather than by looping over a tracer.
        """
        scheme = falcon.named("Falcon-512")
        with self.assertRaises(Exception) as raised:
            frx.jit(scheme.keygen)(fnp.zeros(scheme.seed_size, dtype=np.uint8))
        self.assertNotIsInstance(raised.exception, NotImplementedError)


class Signing(absltest.TestCase):
    """Algorithm 10, held to the verifier the published vectors already gate.

    `Falcon-512` only, and for the budget reason the rest of this file records:
    a key is what costs, and the degree changes nothing any case here asserts.
    """

    _SALT = bytes(range(encoding.SALT_SIZE))
    _MESSAGE = b"a message signed by this implementation"

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.scheme = falcon.named("Falcon-512")
        public, secret = _key_pair()
        cls.public_key = np.frombuffer(public, dtype=np.uint8)
        cls.secret_key = np.frombuffer(secret, dtype=np.uint8)

    def _sign(self, message: bytes, salt: bytes | None = None) -> np.ndarray:
        return np.asarray(
            self.scheme.sign(
                self.secret_key,
                np.frombuffer(message, dtype=np.uint8),
                randomness=np.frombuffer(salt or self._SALT, dtype=np.uint8),
            ),
            dtype=np.uint8,
        )

    def _verify(self, signature: np.ndarray, message: bytes) -> bool:
        return bool(
            np.asarray(
                self.scheme.verify(
                    self.public_key[None],
                    np.frombuffer(message, dtype=np.uint8)[None],
                    signature[None],
                )
            )[0]
        )

    def test_a_signature_verifies_and_has_the_standard_shape(self) -> None:
        signature = self._sign(self._MESSAGE)
        self.assertLen(signature, self.scheme.params.signature_size)
        # §3.11.3's header, which `sig_decode` checks and a wrong nibble fails.
        self.assertEqual(signature[0], encoding.degree_header(512, 0b0011))
        np.testing.assert_array_equal(
            signature[1 : 1 + encoding.SALT_SIZE],
            np.frombuffer(self._SALT, dtype=np.uint8),
        )
        self.assertTrue(self._verify(signature, self._MESSAGE))

    def test_the_salt_is_what_makes_two_signatures_differ(self) -> None:
        """§3.9's whole reason for drawing one, and both still verify."""
        other = bytes(range(1, encoding.SALT_SIZE + 1))
        first, second = self._sign(self._MESSAGE), self._sign(self._MESSAGE, other)
        self.assertFalse(np.array_equal(first, second))
        self.assertTrue(self._verify(first, self._MESSAGE))
        self.assertTrue(self._verify(second, self._MESSAGE))

    def test_the_same_inputs_give_the_same_signature(self) -> None:
        """Not a property Falcon has — a property of this expansion.

        The sampler's stream is derived from the salt and the key, so a caller
        that repeats both gets the same bytes. That is what makes a failing case
        reproducible; it is not determinism in the scheme, which draws a fresh
        salt per signature.
        """
        np.testing.assert_array_equal(
            self._sign(self._MESSAGE), self._sign(self._MESSAGE)
        )

    def test_a_signature_does_not_verify_for_another_message(self) -> None:
        signature = self._sign(self._MESSAGE)
        self.assertFalse(self._verify(signature, self._MESSAGE + b"!"))

    def test_a_corrupted_signature_is_refused(self) -> None:
        """Byte 41 is the first compressed coefficient — past header and salt.

        So the salt still matches and `hash_to_point` still produces the same
        target: this is a wrong point for the right challenge rather than a
        signature over a different message.
        """
        signature = self._sign(self._MESSAGE)
        self.assertTrue(self._verify(signature, self._MESSAGE))
        corrupted = signature.copy()
        corrupted[41] ^= 0x01
        self.assertFalse(self._verify(corrupted, self._MESSAGE))

    def test_the_salt_is_required_and_checked(self) -> None:
        """The seam does not draw randomness, so a caller that omits it fails."""
        message = np.frombuffer(self._MESSAGE, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "does not draw it"):
            self.scheme.sign(self.secret_key, message)
        with self.assertRaisesRegex(ValueError, "40 bytes"):
            self.scheme.sign(
                self.secret_key, message, randomness=np.zeros(8, dtype=np.uint8)
            )

    def test_a_malformed_secret_key_is_refused(self) -> None:
        """A wrong header byte is not a §3.11.5 encoding, and does not sign."""
        broken = self.secret_key.copy()
        broken[0] ^= 0xFF
        with self.assertRaisesRegex(ValueError, "§3.11.5"):
            self.scheme.sign(
                broken,
                np.frombuffer(self._MESSAGE, dtype=np.uint8),
                randomness=np.frombuffer(self._SALT, dtype=np.uint8),
            )


if __name__ == "__main__":
    absltest.main()
