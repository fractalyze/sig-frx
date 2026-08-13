# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""ML-DSA's wire formats against FIPS 204's own, host and traced.

Encoding has no cryptography in it, so almost all of it is covered by two
cheap things: it agrees with the standard's algorithm transcribed into
`fips204_reference`, and it round-trips. What neither of those covers is the
part that matters most, so it gets its own class:

**`HintBitUnpack`'s rejections are a security property, not a nicety.** A
verifier that accepts a malformed hint encoding is exploitable, and every
rejection Algorithm 21 specifies exists to make the encoding *canonical* —
without them one hint has many encodings, which is a malleability bug in a
signature. Each of the three is driven by a mutation of a genuine encoding
here, together with the control that the unmutated one is accepted, because a
rejection test that would also reject the valid case proves nothing.

The sizes are checked against FIPS 204's own formulas rather than against
constants copied from the table, so a wrong parameter set fails as a size
mismatch here rather than as an unreadable key somewhere downstream.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from sig_frx.lattice.mldsa import arith, encoding
from sig_frx.lattice.mldsa.testing import fips204_reference as ref

_Lift: TypeAlias = Callable[[np.ndarray], Any]
_LIFT = {"host": np.asarray, "traced": fnp.asarray}
_NAMESPACES = tuple(_LIFT.items())

# FIPS 204 Table 1, from the reference so the two test modules cannot drift.
_PARAMETER_SETS = ref.parameter_cases()

# Every `(a, b)` pair the scheme packs at: `t1`, `s1`/`s2` at both etas, `t0`,
# and `z` at both gamma1s.
_WIDTHS = (
    (0, (1 << encoding.T1_BITS) - 1),
    (2, 2),
    (4, 4),
    ((1 << (arith.D - 1)) - 1, 1 << (arith.D - 1)),
    ((1 << 17) - 1, 1 << 17),
    ((1 << 19) - 1, 1 << 19),
)


def _coefficients(low: int, high: int, salt: int, rows: int = 1) -> np.ndarray:
    rng = np.random.default_rng(salt)
    return rng.integers(low, high + 1, size=(rows, arith.N)).astype(np.int32)


def _bytes(width: int, salt: int) -> np.ndarray:
    return np.asarray(
        np.random.default_rng(salt).integers(0, 256, size=width), dtype=np.uint8
    )


def _hint(k: int, omega: int, salt: int) -> np.ndarray:
    """Half of `omega`'s budget of nonzero coefficients, spread over `k` rows.

    Deliberately under budget: the encoding then has an unused tail, which is
    what the tail rejection mutates. Leaving the weight to chance would make
    that test's premise an accident of the seed.
    """
    flat = np.zeros(k * arith.N, dtype=np.int32)
    rng = np.random.default_rng(salt)
    flat[rng.choice(k * arith.N, size=omega // 2, replace=False)] = 1
    return flat.reshape(k, arith.N)


class BitPackingTest(parameterized.TestCase):
    """Algorithms 16-19, at every width the scheme actually uses."""

    @parameterized.named_parameters(*((f"a_{a}_b_{b}", a, b) for a, b in _WIDTHS))
    def test_bit_pack_matches_algorithm_17(self, a: int, b: int) -> None:
        w = _coefficients(-a, b, a + b)
        got = np.asarray(encoding.bit_pack(w, a, b))[0].tolist()
        self.assertEqual(got, list(ref.bit_pack(w[0].tolist(), a, b)))

    @parameterized.named_parameters(*((f"a_{a}_b_{b}", a, b) for a, b in _WIDTHS))
    def test_bit_unpack_inverts_bit_pack(self, a: int, b: int) -> None:
        w = _coefficients(-a, b, a + b + 1, rows=3)
        packed = encoding.bit_pack(w, a, b)
        self.assertTrue(
            np.array_equal(np.asarray(encoding.bit_unpack(packed, a, b)), w)
        )

    @parameterized.named_parameters(*((f"a_{a}_b_{b}", a, b) for a, b in _WIDTHS))
    def test_the_packed_size_is_the_standard_s(self, a: int, b: int) -> None:
        """`32·bitlen(a+b)` bytes — Algorithm 17's stated output length."""
        packed = np.asarray(encoding.bit_pack(_coefficients(-a, b, 7), a, b))
        self.assertEqual(packed.shape[-1], 32 * (a + b).bit_length())

    @parameterized.named_parameters(("b_1023", 1023), ("b_15", 15), ("b_43", 43))
    def test_simple_bit_pack_matches_algorithm_16(self, b: int) -> None:
        w = _coefficients(0, b, b)
        got = np.asarray(encoding.simple_bit_pack(w, b))[0].tolist()
        self.assertEqual(got, list(ref.simple_bit_pack(w[0].tolist(), b)))
        unpacked = np.asarray(encoding.simple_bit_unpack(np.asarray(got, np.uint8), b))
        self.assertEqual(unpacked.tolist(), ref.simple_bit_unpack(bytes(got), b))

    @parameterized.named_parameters(*_NAMESPACES)
    def test_host_and_traced_agree(self, lift: _Lift) -> None:
        a, b = (1 << 19) - 1, 1 << 19
        w = _coefficients(-a, b, 11, rows=2)
        packed = encoding.bit_pack(lift(w), a, b)
        self.assertTrue(
            np.array_equal(np.asarray(encoding.bit_unpack(packed, a, b)), w)
        )


class HintTest(parameterized.TestCase):
    """Algorithms 20 and 21, including every rejection the standard requires."""

    @parameterized.named_parameters(*_PARAMETER_SETS)
    def test_hint_bit_pack_matches_algorithm_20(
        self, k: int, omega: int, **_: Any
    ) -> None:
        h = _hint(k, omega, k * 100)
        got = np.asarray(encoding.hint_bit_pack(h, omega)).tolist()
        self.assertEqual(got, list(ref.hint_bit_pack(h.tolist(), omega)))

    @parameterized.named_parameters(*_PARAMETER_SETS)
    def test_the_encoded_size_is_omega_plus_k(
        self, k: int, omega: int, **_: Any
    ) -> None:
        packed = np.asarray(encoding.hint_bit_pack(_hint(k, omega, 1), omega))
        self.assertEqual(packed.shape[-1], omega + k)

    @parameterized.named_parameters(*_PARAMETER_SETS)
    def test_hint_round_trips(self, k: int, omega: int, **_: Any) -> None:
        h = _hint(k, omega, k + 7)
        packed = encoding.hint_bit_pack(h, omega)
        got, ok = encoding.hint_bit_unpack(packed, k, omega)
        self.assertTrue(bool(ok))
        self.assertTrue(np.array_equal(np.asarray(got), h))
        self.assertEqual(
            np.asarray(got).tolist(),
            ref.hint_bit_unpack(bytes(np.asarray(packed)), k, omega),
        )

    @parameterized.named_parameters(*_PARAMETER_SETS)
    def test_an_empty_hint_round_trips(self, k: int, omega: int, **_: Any) -> None:
        """All-zero is a legal hint and its encoding is all-zero — the case a
        tail check written slightly wrong rejects."""
        h = np.zeros((k, arith.N), dtype=np.int32)
        got, ok = encoding.hint_bit_unpack(encoding.hint_bit_pack(h, omega), k, omega)
        self.assertTrue(bool(ok))
        self.assertTrue(np.array_equal(np.asarray(got), h))

    def _valid(self) -> tuple[np.ndarray, int, int]:
        k, omega = 4, 80
        packed = np.asarray(encoding.hint_bit_pack(_hint(k, omega, 3), omega))
        return packed, k, omega

    def test_the_control_case_is_accepted(self) -> None:
        """Without this the rejection cases below prove nothing."""
        packed, k, omega = self._valid()
        self.assertTrue(bool(encoding.hint_bit_unpack(packed, k, omega)[1]))
        self.assertIsNotNone(ref.hint_bit_unpack(bytes(packed), k, omega))

    def test_rejects_a_cut_that_goes_backwards(self) -> None:
        """Algorithm 21 line 4: `y[omega+i] < Index`."""
        packed, k, omega = self._valid()
        bad = packed.copy()
        bad[omega + k - 1] = 0
        self.assertFalse(bool(encoding.hint_bit_unpack(bad, k, omega)[1]))
        self.assertIsNone(ref.hint_bit_unpack(bytes(bad), k, omega))

    def test_rejects_a_cut_past_omega(self) -> None:
        """Algorithm 21 line 4: `y[omega+i] > omega`."""
        packed, k, omega = self._valid()
        bad = packed.copy()
        bad[omega + k - 1] = omega + 1
        self.assertFalse(bool(encoding.hint_bit_unpack(bad, k, omega)[1]))
        self.assertIsNone(ref.hint_bit_unpack(bytes(bad), k, omega))

    @parameterized.named_parameters(("decreasing", (9, 5)), ("repeated", (5, 5)))
    def test_rejects_positions_that_do_not_strictly_increase(
        self, replacement: tuple[int, int]
    ) -> None:
        """Algorithm 21 line 9 — what makes the encoding canonical.

        Without it the same hint has many encodings, which is signature
        malleability rather than a decoding nicety. Line 9 says `>=` and not
        `>`, so a repeated position is malformed for the same reason a
        decreasing one is.
        """
        k, omega = 4, 80
        h = np.zeros((k, arith.N), dtype=np.int32)
        h[0, 5] = h[0, 9] = 1
        bad = np.asarray(encoding.hint_bit_pack(h, omega)).copy()
        bad[0], bad[1] = replacement
        self.assertFalse(bool(encoding.hint_bit_unpack(bad, k, omega)[1]))
        self.assertIsNone(ref.hint_bit_unpack(bytes(bad), k, omega))

    def test_rejects_a_nonzero_byte_in_the_unused_tail(self) -> None:
        """Algorithm 21 line 17 — the same canonicality at the buffer's end."""
        packed, k, omega = self._valid()
        bad = packed.copy()
        bad[omega - 1] = 7
        self.assertFalse(bool(encoding.hint_bit_unpack(bad, k, omega)[1]))
        self.assertIsNone(ref.hint_bit_unpack(bytes(bad), k, omega))

    def test_rejection_survives_a_tracer(self) -> None:
        """The rejection is the verifier's, and the verifier is traced."""
        packed, k, omega = self._valid()
        bad = packed.copy()
        bad[omega + k - 1] = omega + 1
        compiled = frx.jit(lambda y: encoding.hint_bit_unpack(y, k, omega)[1])
        self.assertFalse(bool(compiled(fnp.asarray(bad))))
        self.assertTrue(bool(compiled(fnp.asarray(packed))))

    def test_hint_bit_pack_survives_a_tracer(self) -> None:
        """The packer is built around a masked sum rather than a scatter so it
        can be traced; nothing pinned that it actually can."""
        k, omega = 4, 80
        h = _hint(k, omega, 42)
        compiled = frx.jit(lambda flags: encoding.hint_bit_pack(flags, omega))
        self.assertEqual(
            np.asarray(compiled(fnp.asarray(h))).tolist(),
            np.asarray(encoding.hint_bit_pack(h, omega)).tolist(),
        )

    def test_rejection_batches_over_signatures(self) -> None:
        """`B` signatures decode in one call, and each gets its own verdict."""
        packed, k, omega = self._valid()
        bad = packed.copy()
        bad[omega + k - 1] = omega + 1
        batch = fnp.asarray(np.stack([packed, bad, packed]))
        verdicts = frx.jit(
            frx.vmap(lambda y: encoding.hint_bit_unpack(y, k, omega)[1])
        )(batch)
        self.assertEqual(np.asarray(verdicts).tolist(), [True, False, True])


class WireFormatTest(parameterized.TestCase):
    """Algorithms 22-28 — the key, signature and `w1` encodings."""

    @parameterized.named_parameters(*_PARAMETER_SETS)
    def test_pk_matches_algorithm_22_and_round_trips(self, k: int, **_: Any) -> None:
        rho = _bytes(32, k)
        t1 = _coefficients(0, (1 << encoding.T1_BITS) - 1, k + 1, rows=k)
        got = np.asarray(encoding.pk_encode(rho, t1))
        self.assertEqual(got.tolist(), list(ref.pk_encode(bytes(rho), t1.tolist())))
        self.assertEqual(got.shape[-1], 32 + 32 * k * encoding.T1_BITS)
        back_rho, back_t1 = encoding.pk_decode(got, k)
        want_rho, want_t1 = ref.pk_decode(bytes(got), k)
        self.assertEqual(np.asarray(back_rho).tolist(), list(want_rho))
        self.assertEqual(np.asarray(back_t1).tolist(), want_t1)
        self.assertTrue(np.array_equal(np.asarray(back_t1), t1))

    @parameterized.named_parameters(*_PARAMETER_SETS)
    def test_sk_matches_algorithm_24_and_round_trips(
        self, k: int, ell: int, eta: int, **_: Any
    ) -> None:
        rho, key, tr = _bytes(32, 1), _bytes(32, 2), _bytes(64, 3)
        s1 = _coefficients(-eta, eta, 10, rows=ell)
        s2 = _coefficients(-eta, eta, 11, rows=k)
        low = 1 << (arith.D - 1)
        t0 = _coefficients(-low + 1, low, 12, rows=k)

        got = np.asarray(encoding.sk_encode(rho, key, tr, s1, s2, t0, eta))
        self.assertEqual(
            got.tolist(),
            list(
                ref.sk_encode(
                    bytes(rho),
                    bytes(key),
                    bytes(tr),
                    s1.tolist(),
                    s2.tolist(),
                    t0.tolist(),
                    eta,
                )
            ),
        )
        width = (2 * eta).bit_length()
        self.assertEqual(got.shape[-1], 128 + 32 * ((k + ell) * width + arith.D * k))
        back = encoding.sk_decode(got, k, ell, eta)
        self.assertEqual(np.asarray(back[0]).tolist(), rho.tolist())
        self.assertEqual(np.asarray(back[2]).tolist(), tr.tolist())
        for got_part, want in zip(back[3:], (s1, s2, t0)):
            self.assertTrue(np.array_equal(np.asarray(got_part), want))

    @parameterized.named_parameters(*_PARAMETER_SETS)
    def test_sig_matches_algorithm_26_and_round_trips(
        self, k: int, ell: int, lam: int, gamma1: int, omega: int, **_: Any
    ) -> None:
        c_tilde = _bytes(lam // 4, 4)
        z = _coefficients(-gamma1 + 1, gamma1, 5, rows=ell)
        h = _hint(k, omega, 6)

        got = np.asarray(encoding.sig_encode(c_tilde, z, h, gamma1, omega))
        self.assertEqual(
            got.tolist(),
            list(ref.sig_encode(bytes(c_tilde), z.tolist(), h.tolist(), gamma1, omega)),
        )
        self.assertEqual(
            got.shape[-1],
            lam // 4 + ell * 32 * (1 + (gamma1 - 1).bit_length()) + omega + k,
        )
        back_c, back_z, back_h, ok = encoding.sig_decode(
            got, lam, ell, k, gamma1, omega
        )
        self.assertTrue(bool(ok))
        self.assertEqual(np.asarray(back_c).tolist(), c_tilde.tolist())
        self.assertTrue(np.array_equal(np.asarray(back_z), z))
        self.assertTrue(np.array_equal(np.asarray(back_h), h))

    @parameterized.named_parameters(*_PARAMETER_SETS)
    def test_sig_decode_reports_a_malformed_hint(
        self, k: int, ell: int, lam: int, gamma1: int, omega: int, **_: Any
    ) -> None:
        """The `⊥` of Algorithm 27, which is the hint's and only the hint's:
        `z` cannot be out of range because `gamma1` is a power of two."""
        c_tilde = _bytes(lam // 4, 4)
        z = _coefficients(-gamma1 + 1, gamma1, 5, rows=ell)
        sigma = np.asarray(
            encoding.sig_encode(c_tilde, z, _hint(k, omega, 6), gamma1, omega)
        ).copy()
        sigma[-1] = omega + 1
        self.assertFalse(
            bool(encoding.sig_decode(sigma, lam, ell, k, gamma1, omega)[3])
        )

    @parameterized.named_parameters(*_PARAMETER_SETS)
    def test_w1_matches_algorithm_28(self, k: int, gamma2: int, **_: Any) -> None:
        bound = (arith.Q - 1) // (2 * gamma2) - 1
        w1 = _coefficients(0, bound, k + 20, rows=k)
        got = np.asarray(encoding.w1_encode(w1, gamma2))
        self.assertEqual(got.tolist(), list(ref.w1_encode(w1.tolist(), gamma2)))
        self.assertEqual(got.shape[-1], 32 * k * bound.bit_length())


if __name__ == "__main__":
    absltest.main()
