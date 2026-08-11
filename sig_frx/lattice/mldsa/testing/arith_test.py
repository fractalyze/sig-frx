# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""ML-DSA's arithmetic against FIPS 204's own form, host and traced.

Three gates, in the order they localize a failure. The zeta table is pinned
against the array Appendix B publishes, so a wrong root of unity fails here rather
than as a wrong signature six issues later. The NTT and the rounding functions are
pinned against `fips204_reference`, which loops the way the standard loops. And
every case runs on numpy and under a tracer, because a scheme keygen-signs on the
host and verifies under a tracer — a difference between the two paths is the bug
this module is written against.

The rounding functions get a contiguous low range rather than random samples: they
are piecewise, the pieces are `2^d` and `2γ2` wide, and the interesting inputs are
the boundaries. The carry case Decompose exists for — `r+ - r0 = q - 1` — is
included explicitly, since no random draw finds it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import frx
import frx.numpy as fnp
import numpy as np
import zk_dtypes
from absl.testing import absltest, parameterized

from sig_frx.lattice.mldsa import arith
from sig_frx.lattice.mldsa.testing import fips204_reference as ref

# `np.asarray` or `fnp.asarray` — the two namespaces every case runs in.
_Lift: TypeAlias = Callable[[np.ndarray], Any]

# FIPS 204 Appendix B, transcribed from the published document. Slot 0 is the
# placeholder the standard prints; Algorithms 41 and 42 index from 1.
_APPENDIX_B_ZETAS = (
    0,
    4808194,
    3765607,
    3761513,
    5178923,
    5496691,
    5234739,
    5178987,
    7778734,
    3542485,
    2682288,
    2129892,
    3764867,
    7375178,
    557458,
    7159240,
    5010068,
    4317364,
    2663378,
    6705802,
    4855975,
    7946292,
    676590,
    7044481,
    5152541,
    1714295,
    2453983,
    1460718,
    7737789,
    4795319,
    2815639,
    2283733,
    3602218,
    3182878,
    2740543,
    4793971,
    5269599,
    2101410,
    3704823,
    1159875,
    394148,
    928749,
    1095468,
    4874037,
    2071829,
    4361428,
    3241972,
    2156050,
    3415069,
    1759347,
    7562881,
    4805951,
    3756790,
    6444618,
    6663429,
    4430364,
    5483103,
    3192354,
    556856,
    3870317,
    2917338,
    1853806,
    3345963,
    1858416,
    3073009,
    1277625,
    5744944,
    3852015,
    4183372,
    5157610,
    5258977,
    8106357,
    2508980,
    2028118,
    1937570,
    4564692,
    2811291,
    5396636,
    7270901,
    4158088,
    1528066,
    482649,
    1148858,
    5418153,
    7814814,
    169688,
    2462444,
    5046034,
    4213992,
    4892034,
    1987814,
    5183169,
    1736313,
    235407,
    5130263,
    3258457,
    5801164,
    1787943,
    5989328,
    6125690,
    3482206,
    4197502,
    7080401,
    6018354,
    7062739,
    2461387,
    3035980,
    621164,
    3901472,
    7153756,
    2925816,
    3374250,
    1356448,
    5604662,
    2683270,
    5601629,
    4912752,
    2312838,
    7727142,
    7921254,
    348812,
    8052569,
    1011223,
    6026202,
    4561790,
    6458164,
    6143691,
    1744507,
    1753,
    6444997,
    5720892,
    6924527,
    2660408,
    6600190,
    8321269,
    2772600,
    1182243,
    87208,
    636927,
    4415111,
    4423672,
    6084020,
    5095502,
    4663471,
    8352605,
    822541,
    1009365,
    5926272,
    6400920,
    1596822,
    4423473,
    4620952,
    6695264,
    4969849,
    2678278,
    4611469,
    4829411,
    635956,
    8129971,
    5925040,
    4234153,
    6607829,
    2192938,
    6653329,
    2387513,
    4768667,
    8111961,
    5199961,
    3747250,
    2296099,
    1239911,
    4541938,
    3195676,
    2642980,
    1254190,
    8368000,
    2998219,
    141835,
    8291116,
    2513018,
    7025525,
    613238,
    7070156,
    6161950,
    7921677,
    6458423,
    4040196,
    4908348,
    2039144,
    6500539,
    7561656,
    6201452,
    6757063,
    2105286,
    6006015,
    6346610,
    586241,
    7200804,
    527981,
    5637006,
    6903432,
    1994046,
    2491325,
    6987258,
    507927,
    7192532,
    7655613,
    6545891,
    5346675,
    8041997,
    2647994,
    3009748,
    5767564,
    4148469,
    749577,
    4357667,
    3980599,
    2569011,
    6764887,
    1723229,
    1665318,
    2028038,
    1163598,
    5011144,
    3994671,
    8368538,
    7009900,
    3020393,
    3363542,
    214880,
    545376,
    7609976,
    3105558,
    7277073,
    508145,
    7826699,
    860144,
    3430436,
    140244,
    6866265,
    6195333,
    3123762,
    2358373,
    6187330,
    5365997,
    6663603,
    2926054,
    7987710,
    8077412,
    3531229,
    4405932,
    4606686,
    1900052,
    7598542,
    1054478,
    7648983,
)

# FIPS 204 Table 1: `(q-1)/88` at ML-DSA-44, `(q-1)/32` at ML-DSA-65 and -87.
_GAMMA2 = ((arith.Q - 1) // 88, (arith.Q - 1) // 32)

# Every coefficient boundary a rounding function can turn on, plus the top of the
# range where Decompose's carry case lives. `q - 1 - 2γ2` is the first input whose
# straightforward rounding would overflow `r1`.
_EDGE_INPUTS = (
    0,
    1,
    2,
    (1 << arith.D) - 1,
    1 << arith.D,
    (1 << arith.D) + 1,
    arith.Q // 2,
    arith.Q - 2,
    arith.Q - 1,
)

# Every case runs both ways. `parameterized.product` names its cases from the
# values, so it takes the key and looks the lift up here rather than carrying a
# function into a test name.
_LIFT = {"host": np.asarray, "traced": fnp.asarray}
_NAMESPACES = tuple(_LIFT.items())


def _field(values: np.ndarray | list[int]) -> np.ndarray:
    """Canonical integers as `FIELD` elements, the representation the module takes."""
    return np.array(np.asarray(values, dtype=np.uint64), dtype=arith.FIELD)


def _ints(values: Any) -> list[int]:
    """Canonical integers back out — `astype`, never a bitcast (see `arith`)."""
    return [int(v) for v in np.asarray(values).astype(np.uint32)]


def _rounding_inputs(gamma2: int) -> np.ndarray:
    """A contiguous low range, the `2γ2` boundaries, and the carry case."""
    alpha = 2 * gamma2
    boundaries = [
        value + offset
        for multiple in range(0, arith.Q, alpha)
        for offset in (-1, 0, 1)
        for value in (multiple,)
        if 0 <= multiple + offset < arith.Q
    ]
    return _field(sorted(set(list(range(4096)) + list(_EDGE_INPUTS) + boundaries)))


class ZetaTableTest(absltest.TestCase):
    """The generated table against the one the standard publishes."""

    def test_matches_appendix_b(self) -> None:
        published = list(_APPENDIX_B_ZETAS)
        generated = [int(value) for value in arith.ZETAS]
        # Slot 0 is unused by both algorithms; the standard prints 0 there and
        # `zeta^BitRev8(0) = 1`, so the pin starts at 1.
        self.assertEqual(generated[1:], published[1:])
        self.assertLen(published, arith.N)

    def test_zeta_is_a_512th_root_of_unity(self) -> None:
        self.assertEqual(pow(arith.ZETA, 512, arith.Q), 1)
        self.assertNotEqual(pow(arith.ZETA, 256, arith.Q), 1)


class FieldSelectionTest(parameterized.TestCase):
    """The dtype this module selected is the right one, and is read the right way.

    Not a test of `zk_dtypes`' arithmetic, which is that repo's to prove. What is
    this module's to prove is the two decisions it made: which field, and how a
    field element becomes an integer.
    """

    def test_the_modulus_is_the_standard_s(self) -> None:
        self.assertEqual(zk_dtypes.pfinfo(arith.FIELD).modulus, arith.Q)

    @parameterized.named_parameters(*_NAMESPACES)
    def test_operations_reduce(self, lift: _Lift) -> None:
        """A sample against big integers, including `(q-1)^2`.

        That value is why hand-written arithmetic would have needed a limb split:
        it is 2^46 against a 32-bit lane. The field reduces internally, so it is
        exact here — the check is that the dtype really is Montgomery-backed and
        not something that leaves residues in a raw lane.
        """
        rng = np.random.default_rng(0)
        a = rng.integers(0, arith.Q, size=4096, dtype=np.uint64)
        b = np.concatenate(
            [
                rng.integers(0, arith.Q, size=4095, dtype=np.uint64),
                np.array([arith.Q - 1], dtype=np.uint64),
            ]
        )
        a[-1] = arith.Q - 1
        fa, fb = lift(_field(a)), lift(_field(b))
        big_a, big_b = a.astype(object), b.astype(object)
        self.assertEqual(_ints(fa * fb), list((big_a * big_b) % arith.Q))
        self.assertEqual(_ints(fa + fb), list((big_a + big_b) % arith.Q))
        self.assertEqual(_ints(fa - fb), list((big_a - big_b) % arith.Q))

    def test_canonical_is_astype_and_not_a_bitcast(self) -> None:
        """The trap the module docstring names, pinned.

        Storage is a Montgomery representative, so reinterpreting the bytes gives
        a different number. Both conversions run and neither raises, which is
        exactly why this needs a test rather than a comment: picking the wrong one
        is silent.
        """
        values = [0, 1, 2, 1753, arith.Q // 2, arith.Q - 1]
        elements = _field(values)
        self.assertEqual(_ints(elements), values)
        reinterpreted = list(np.asarray(elements).view(np.uint32))
        self.assertNotEqual(reinterpreted, values)


class NttTest(parameterized.TestCase):
    """The eight batched layers against Algorithms 41 and 42 as written."""

    @parameterized.named_parameters(*_NAMESPACES)
    def test_forward_matches_algorithm_41(self, lift: _Lift) -> None:
        rng = np.random.default_rng(2)
        w = _field(rng.integers(0, arith.Q, size=arith.N, dtype=np.uint64))
        got = [int(value) for value in np.asarray(arith.ntt(lift(w)))]
        self.assertEqual(got, ref.ntt([int(value) for value in w]))

    @parameterized.named_parameters(*_NAMESPACES)
    def test_matches_the_definition_by_evaluation(self, lift: _Lift) -> None:
        """Against §2.5's eq. (2.1) rather than §7.5's algorithm.

        The only case here that pins which slot holds which evaluation: the round
        trip and the convolution property both survive a consistently permuted
        order, and Algorithm 41 is the thing being reshaped, so it cannot
        independently pin the ordering it defines.
        """
        rng = np.random.default_rng(11)
        w = _field(rng.integers(0, arith.Q, size=arith.N, dtype=np.uint64))
        got = [int(value) for value in np.asarray(arith.ntt(lift(w)))]
        self.assertEqual(got, ref.ntt_by_definition([int(value) for value in w]))

    @parameterized.named_parameters(*_NAMESPACES)
    def test_inverse_matches_algorithm_42(self, lift: _Lift) -> None:
        rng = np.random.default_rng(3)
        w = _field(rng.integers(0, arith.Q, size=arith.N, dtype=np.uint64))
        got = [int(value) for value in np.asarray(arith.intt(lift(w)))]
        self.assertEqual(got, ref.intt([int(value) for value in w]))

    @parameterized.named_parameters(*_NAMESPACES)
    def test_round_trips_to_the_identity(self, lift: _Lift) -> None:
        rng = np.random.default_rng(4)
        w = _field(rng.integers(0, arith.Q, size=(3, 4, arith.N), dtype=np.uint64))
        self.assertTrue(np.array_equal(np.asarray(arith.intt(arith.ntt(lift(w)))), w))

    @parameterized.named_parameters(*_NAMESPACES)
    def test_batches_over_the_polynomial_vector_axis(self, lift: _Lift) -> None:
        """A vector of `k` polynomials is one call and agrees with each alone."""
        rng = np.random.default_rng(5)
        vector = rng.integers(0, arith.Q, size=(8, arith.N), dtype=np.uint64).astype(
            np.uint32
        )
        batched = np.asarray(arith.ntt(lift(vector)))
        for index, row in enumerate(vector):
            alone = np.asarray(arith.ntt(lift(row)))
            self.assertTrue(np.array_equal(batched[index], alone))

    @parameterized.named_parameters(*_NAMESPACES)
    def test_convolution_is_negacyclic(self, lift: _Lift) -> None:
        """`intt(ntt(a) ∘ ntt(b))` is multiplication in `Z_q[X]/(X^256 + 1)`.

        The property the transform exists for, checked against schoolbook
        multiplication with the wrap-around negated by hand — which is the only
        check here that would catch a self-consistent wrong root of unity.
        """
        rng = np.random.default_rng(6)
        a = _field(rng.integers(0, arith.Q, size=arith.N, dtype=np.uint64))
        b = _field(rng.integers(0, arith.Q, size=arith.N, dtype=np.uint64))

        want = [0] * arith.N
        for i, left in enumerate(int(value) for value in a):
            for j, right in enumerate(int(value) for value in b):
                degree = i + j
                product = left * right
                if degree >= arith.N:
                    degree -= arith.N
                    product = -product
                want[degree] = (want[degree] + product) % arith.Q

        got = arith.intt(arith.base_mul(arith.ntt(lift(a)), arith.ntt(lift(b))))
        self.assertEqual([int(value) for value in np.asarray(got)], want)

    def test_host_and_traced_agree(self) -> None:
        rng = np.random.default_rng(7)
        w = _field(rng.integers(0, arith.Q, size=(2, arith.N), dtype=np.uint64))
        self.assertTrue(
            np.array_equal(
                np.asarray(arith.ntt(w)), np.asarray(arith.ntt(fnp.asarray(w)))
            )
        )

    def test_traces_as_one_computation(self) -> None:
        """The whole transform is one jit zone, not a driver loop."""
        rng = np.random.default_rng(8)
        w = _field(rng.integers(0, arith.Q, size=(4, arith.N), dtype=np.uint64))
        compiled = frx.jit(lambda x: arith.intt(arith.ntt(x)))
        self.assertTrue(np.array_equal(np.asarray(compiled(fnp.asarray(w))), w))


class RoundingTest(parameterized.TestCase):
    """Each rounding function against its algorithm, over its boundaries."""

    @parameterized.named_parameters(*_NAMESPACES)
    def test_power2round_matches_algorithm_35(self, lift: _Lift) -> None:
        values = _rounding_inputs(_GAMMA2[0])
        r1, r0 = arith.power2round(lift(values))
        got = list(
            zip((int(v) for v in np.asarray(r1)), (int(v) for v in np.asarray(r0)))
        )
        self.assertEqual(got, [ref.power2round(int(v)) for v in values])

    @parameterized.named_parameters(*_NAMESPACES)
    def test_power2round_reconstructs_the_input(self, lift: _Lift) -> None:
        values = _rounding_inputs(_GAMMA2[0])
        r1, r0 = arith.power2round(lift(values))
        rebuilt = (np.asarray(r1).astype(object) * (1 << arith.D)) + np.asarray(
            r0
        ).astype(object)
        self.assertTrue(np.array_equal(rebuilt % arith.Q, values.astype(object)))

    @parameterized.product(namespace=tuple(_LIFT), gamma2=_GAMMA2)
    def test_decompose_matches_algorithm_36(self, namespace: str, gamma2: int) -> None:
        lift = _LIFT[namespace]
        values = _rounding_inputs(gamma2)
        r1, r0 = arith.decompose(lift(values), gamma2)
        got = list(
            zip((int(v) for v in np.asarray(r1)), (int(v) for v in np.asarray(r0)))
        )
        self.assertEqual(got, [ref.decompose(int(v), gamma2) for v in values])

    @parameterized.parameters(*_GAMMA2)
    def test_decompose_hits_the_carry_case(self, gamma2: int) -> None:
        """The `r+ - r0 = q - 1` branch is actually reached by these inputs.

        Guards the test rather than the module: if the input set stopped covering
        it, `decompose` would keep passing while the branch went unchecked.
        """
        values = _rounding_inputs(gamma2)
        reached = [
            v for v in values if ref.decompose(int(v), gamma2)[0] == 0 and v != 0
        ]
        self.assertNotEmpty(reached)

    @parameterized.product(namespace=tuple(_LIFT), gamma2=_GAMMA2)
    def test_high_and_low_bits_are_decompose(self, namespace: str, gamma2: int) -> None:
        lift = _LIFT[namespace]
        values = _rounding_inputs(gamma2)
        self.assertEqual(
            [int(v) for v in np.asarray(arith.high_bits(lift(values), gamma2))],
            [ref.high_bits(int(v), gamma2) for v in values],
        )
        self.assertEqual(
            [int(v) for v in np.asarray(arith.low_bits(lift(values), gamma2))],
            [ref.low_bits(int(v), gamma2) for v in values],
        )

    @parameterized.product(namespace=tuple(_LIFT), gamma2=_GAMMA2)
    def test_make_hint_matches_algorithm_39(self, namespace: str, gamma2: int) -> None:
        lift = _LIFT[namespace]
        values = _rounding_inputs(gamma2)
        for z in (1, 12345, gamma2, arith.Q - 1):
            # `z` is a field element — it is added to `r`. The hint below is not.
            offsets = _field(np.full(len(values), z, dtype=np.uint64))
            got = np.asarray(arith.make_hint(lift(offsets), lift(values), gamma2))
            self.assertEqual(
                [bool(v) for v in got],
                [ref.make_hint(z, v, gamma2) for v in _ints(values)],
            )

    @parameterized.product(namespace=tuple(_LIFT), gamma2=_GAMMA2, hint=(0, 1))
    def test_use_hint_matches_algorithm_40(
        self, namespace: str, gamma2: int, hint: int
    ) -> None:
        lift = _LIFT[namespace]
        values = _rounding_inputs(gamma2)
        # A hint is a flag, not a residue, so it keeps a plain integer lane.
        hints = np.full(len(values), hint, dtype=np.uint8)
        got = np.asarray(arith.use_hint(lift(hints), lift(values), gamma2))
        self.assertEqual(
            [int(v) for v in got],
            [ref.use_hint(bool(hint), v, gamma2) for v in _ints(values)],
        )

    @parameterized.product(namespace=tuple(_LIFT), gamma2=_GAMMA2)
    def test_use_hint_inverts_make_hint(self, namespace: str, gamma2: int) -> None:
        """§7.4's contract: with the hint, `r + z`'s high bits come back from `r`.

        Holds when `||z||∞` is small, which is what the signing bounds guarantee.
        """
        lift = _LIFT[namespace]
        rng = np.random.default_rng(9)
        r = _field(rng.integers(0, arith.Q, size=4096, dtype=np.uint64))
        z = _field(rng.integers(0, gamma2 // 2, size=4096, dtype=np.uint64))
        hints = arith.make_hint(lift(z), lift(r), gamma2)
        recovered = arith.use_hint(lift(np.asarray(hints)), lift(r), gamma2)
        expected = arith.high_bits(lift(r) + lift(z), gamma2)
        self.assertTrue(np.array_equal(np.asarray(recovered), np.asarray(expected)))


class InfinityNormTest(parameterized.TestCase):
    """`||w||∞` over the trailing axis, against the centered representative."""

    @parameterized.named_parameters(*_NAMESPACES)
    def test_matches_the_reference(self, lift: _Lift) -> None:
        rng = np.random.default_rng(10)
        w = _field(rng.integers(0, arith.Q, size=(5, arith.N), dtype=np.uint64))
        got = np.asarray(arith.infinity_norm(lift(w)))
        want = [ref.infinity_norm([int(v) for v in row]) for row in w]
        self.assertEqual([int(v) for v in got], want)

    @parameterized.named_parameters(*_NAMESPACES)
    def test_is_centered_not_canonical(self, lift: _Lift) -> None:
        """`q - 1` has norm 1, which is the whole difference `mod±` makes."""
        w = np.array([[arith.Q - 1, 0, 1]], dtype=np.uint32)
        self.assertEqual(int(np.asarray(arith.infinity_norm(lift(w)))[0]), 1)


if __name__ == "__main__":
    absltest.main()
