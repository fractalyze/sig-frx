# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The curve substrate against SEC 1's own form, host and traced.

Three gates, ordered to localize a failure. The curve constants are checked
against what SEC 2 asserts about them — the base point satisfies the defining
equation and has order `n` — so a transcription slip fails here rather than as
a wrong signature later. The complete formulas are held to `sec1_reference`,
which cases the way the standard cases, across every input class the
completeness claim covers: distinct points, a doubling, an inverse pair, and
the identity on either side. And every case runs on numpy and under a tracer,
because keygen and signing are concrete while verification is traced — a
difference between those paths is the bug this file is written against.

Random points are deterministic (a seeded `random.Random`), and small batches
are enough: the batch axis's own semantics are the harness's business, this
file only needs traced-equals-host.
"""

from __future__ import annotations

import random
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from sig_frx.classical import group, weierstrass
from sig_frx.classical.testing import sec1_reference as ref
from sig_frx.classical.testing.traced_blocker import TRACED_BLOCKED as _TRACED_BLOCKED

_CURVES = (
    ("secp256k1", weierstrass.SECP256K1),
    ("secp256r1", weierstrass.SECP256R1),
)


def _points(curve: weierstrass.Curve, count: int) -> list[tuple[int, int]]:
    """`count` deterministic points, as multiples of the base point."""
    rng = random.Random(f"weierstrass:{curve.p}")
    generator = (curve.gx, curve.gy)
    result = []
    for _ in range(count):
        point = ref.scalar_mul(curve.p, curve.a, rng.randrange(1, curve.n), generator)
        assert point is not None
        result.append(point)
    return result


def _lift(curve: weierstrass.Curve, points: list, xnp: Any) -> weierstrass.Point:
    """Affine int pairs as a batched projective `Point` in namespace `xnp`.

    `None` entries — SEC 1's `O` — become `(0 : 1 : 0)`.
    """
    field = curve.field
    xs = np.array([0 if p is None else p[0] for p in points], dtype=field)
    ys = np.array([1 if p is None else p[1] for p in points], dtype=field)
    zs = np.array([0 if p is None else 1 for p in points], dtype=field)
    return weierstrass.Point(xnp.asarray(xs), xnp.asarray(ys), xnp.asarray(zs))


def _affine(curve: weierstrass.Curve, point: weierstrass.Point) -> list:
    """Batched projective results back to SEC 1's terms: `(x, y)` ints or `None`."""
    flags = np.asarray(weierstrass.is_identity(curve, point))
    # Divide only where defined: patch identity entries' Z to 1 first.
    z = np.where(flags, np.array(1, dtype=curve.field), np.asarray(point.z))
    xs = np.asarray(point.x) / z
    ys = np.asarray(point.y) / z
    out = []
    for flag, x, y in zip(flags, xs.astype(object), ys.astype(object)):
        out.append(None if flag else (int(x), int(y)))
    return out


def _bits(k: int, width: int = 256) -> np.ndarray:
    """`k` as big-endian bytes, through the shared bit expansion."""
    return group.bits_of(np.frombuffer(k.to_bytes(width // 8, "big"), dtype=np.uint8))


class CurveConstantsTest(parameterized.TestCase):
    @parameterized.named_parameters(*_CURVES)
    def test_base_point_satisfies_defining_equation(
        self, curve: weierstrass.Curve
    ) -> None:
        self.assertTrue(ref.on_curve(curve.p, curve.a, curve.b, (curve.gx, curve.gy)))

    @parameterized.named_parameters(*_CURVES)
    def test_base_point_has_order_n(self, curve: weierstrass.Curve) -> None:
        generator = (curve.gx, curve.gy)
        self.assertIsNone(ref.scalar_mul(curve.p, curve.a, curve.n, generator))
        self.assertEqual(
            ref.scalar_mul(curve.p, curve.a, curve.n + 1, generator), generator
        )


class GroupLawTest(parameterized.TestCase):
    """The complete formulas against SEC 1's five cases, per input class."""

    def _case_pairs(self, curve: weierstrass.Curve) -> list:
        """Input pairs covering every class the completeness claim spans."""
        p1, p2, p3 = _points(curve, 3)
        neg = (p3[0], (-p3[1]) % curve.p)
        return [
            (p1, p2),  # rule 4: distinct x
            (p2, p2),  # rule 5: a doubling
            (p3, neg),  # rule 3: an inverse pair
            (None, p1),  # rule 2: identity on the left
            (p2, None),  # rule 2: identity on the right
            (None, None),  # rule 1
        ]

    @parameterized.named_parameters(*_CURVES)
    def test_add_matches_sec1_host(self, curve: weierstrass.Curve) -> None:
        pairs = self._case_pairs(curve)
        got = weierstrass.add(
            curve,
            _lift(curve, [a for a, _ in pairs], np),
            _lift(curve, [b for _, b in pairs], np),
        )
        want = [ref.add(curve.p, curve.a, a, b) for a, b in pairs]
        self.assertEqual(_affine(curve, got), want)

    @parameterized.named_parameters(*_CURVES)
    @_TRACED_BLOCKED
    def test_add_matches_sec1_traced(self, curve: weierstrass.Curve) -> None:
        pairs = self._case_pairs(curve)
        compiled = frx.jit(lambda p, q: weierstrass.add(curve, p, q))
        got = compiled(
            _lift(curve, [a for a, _ in pairs], fnp),
            _lift(curve, [b for _, b in pairs], fnp),
        )
        want = [ref.add(curve.p, curve.a, a, b) for a, b in pairs]
        self.assertEqual(_affine(curve, weierstrass.Point(*map(np.asarray, got))), want)

    @parameterized.named_parameters(*_CURVES)
    def test_double_matches_sec1_host(self, curve: weierstrass.Curve) -> None:
        points = _points(curve, 4) + [None]
        want = [ref.add(curve.p, curve.a, p, p) for p in points]
        host = weierstrass.double(curve, _lift(curve, points, np))
        self.assertEqual(_affine(curve, host), want)

    @parameterized.named_parameters(*_CURVES)
    @_TRACED_BLOCKED
    def test_double_matches_sec1_traced(self, curve: weierstrass.Curve) -> None:
        points = _points(curve, 4) + [None]
        want = [ref.add(curve.p, curve.a, p, p) for p in points]
        compiled = frx.jit(lambda p: weierstrass.double(curve, p))
        traced = compiled(_lift(curve, points, fnp))
        self.assertEqual(
            _affine(curve, weierstrass.Point(*map(np.asarray, traced))), want
        )


class ScalarMulTest(parameterized.TestCase):
    def _scalars(self, curve: weierstrass.Curve) -> list[int]:
        rng = random.Random(f"ladder:{curve.p}")
        # The boundary scalars a random draw never finds: 1, n-1 (the inverse),
        # n (the identity — a full-width scalar the group itself reduces), n+1.
        return [1, curve.n - 1, curve.n, curve.n + 1, rng.randrange(1, curve.n)]

    @parameterized.named_parameters(*_CURVES)
    def test_ladder_matches_reference_host(self, curve: weierstrass.Curve) -> None:
        scalars = self._scalars(curve)
        generator = (curve.gx, curve.gy)
        bits = np.stack([_bits(k) for k in scalars])
        base = _lift(curve, [generator] * len(scalars), np)
        got = weierstrass.scalar_mul(curve, bits, base)
        want = [ref.scalar_mul(curve.p, curve.a, k, generator) for k in scalars]
        self.assertEqual(_affine(curve, got), want)

    @parameterized.named_parameters(*_CURVES)
    @_TRACED_BLOCKED
    def test_ladder_matches_reference_traced(self, curve: weierstrass.Curve) -> None:
        scalars = self._scalars(curve)
        generator = (curve.gx, curve.gy)
        bits = fnp.asarray(np.stack([_bits(k) for k in scalars]))
        base = _lift(curve, [generator] * len(scalars), fnp)
        compiled = frx.jit(lambda b, pt: weierstrass.scalar_mul(curve, b, pt))
        got = compiled(bits, base)
        want = [ref.scalar_mul(curve.p, curve.a, k, generator) for k in scalars]
        self.assertEqual(_affine(curve, weierstrass.Point(*map(np.asarray, got))), want)


class BitsTest(absltest.TestCase):
    def test_bits_of_reads_big_endian_msb_first(self) -> None:
        data = np.frombuffer(bytes([0x80, 0x01]), dtype=np.uint8)
        got = group.bits_of(data)
        want = [1] + [0] * 7 + [0] * 7 + [1]
        self.assertEqual(list(got), want)
        self.assertEqual(got.shape, (16,))

    def test_bits_of_keeps_leading_axes(self) -> None:
        data = np.zeros((3, 2, 4), dtype=np.uint8)
        self.assertEqual(group.bits_of(data).shape, (3, 2, 32))


if __name__ == "__main__":
    absltest.main()
