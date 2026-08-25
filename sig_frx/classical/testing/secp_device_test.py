# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The substrate answers the same thing on either side of the namespace seam.

`secp.py` reads the namespace off its arguments rather than naming one, so the
same call is a host computation or a device one depending only on where the
caller's batch already lives. That is worth a test rather than an assertion in
a docstring, because the two paths are different kernels over different
representations of the same group, and nothing else in the suite would notice
if they diverged: every KAT runs one of them.

The comparison is on the readback, `affine_ints`, because that is where a
coordinate stops being a dtype and becomes the integers the standards define
encodings on. Comparing the point arrays directly would compare storage, which
is the distinction `secp_test.py` exists to keep honest.
"""

from __future__ import annotations

from unittest import mock

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from sig_frx import arrays
from sig_frx.classical import secp
from sig_frx.classical.testing import ecdsa_wycheproof_vectors as wycheproof
from sig_frx.testing import kat

# secp256k1 only, and not by preference. A traced array can hold a point type
# only if frx's admission table has a row for it, and at the pinned wheel
# `secp256r1_g1_*` has none — it fails with "not a valid JAX array type" before
# any arithmetic happens. P-256's `a = -3` also needs the generated group law
# confirmed for a non-zero `a` before its parity could be claimed. Both are the
# same upstream gap; when the rows land, adding the curve here is the test that
# should gate them.
_CURVES = (("secp256k1", secp.SECP256K1),)

# The lift is not restricted the same way. Its arithmetic is over the *base
# field*, and `secp256r1_bf_mont` is admitted even though `secp256r1_g1_*` is
# not — so P-256 exercises the device lift here while it cannot exercise the
# point seams above. `_place` reads the probe off the dtype it is handed, which
# is what keeps that true; `test_p256_lifts_on_the_device_though_its_points_cannot`
# pins it.
_FIELD_CURVES = (("secp256k1", secp.SECP256K1), ("secp256r1", secp.SECP256R1))

# Forcing the threshold, rather than reaching it: every batch the merge gate
# builds is smaller than `DEVICE_MIN_BATCH`, so a device path only gets
# exercised by moving the threshold out from under it.
_FORCE_HOST = 1 << 30
_FORCE_DEVICE = 0


def _generators(curve: secp.Curve, count: int) -> np.ndarray:
    """`G` repeated `count` times, which is what `generator` is shaped for."""
    return np.broadcast_to(curve.generator, (count,)).astype(curve.point)


def _scalars(curve: secp.Curve, count: int) -> list[int]:
    """Full-width scalars, spread over the range rather than sampled small.

    A scalar's bit length is part of what a ladder costs and part of what it
    exercises, so the small values a hand-written fixture reaches for would
    leave the top of every window untouched.
    """
    step = curve.n // (count + 1)
    return [1 + (i + 1) * step % (curve.n - 1) for i in range(count)]


def _identity_and_generator(curve: secp.Curve) -> np.ndarray:
    """`[0·G, 1·G]` — the identity and a real point, in one jacobian batch."""
    return secp.multiple(curve, [curve.n, 1], _generators(curve, 2))


def _has_a_point(curve: secp.Curve, x: int) -> bool:
    """Whether some `y` satisfies the curve equation at `x`, by Euler.

    Python integers and `pow`, sharing nothing with the substrate under test —
    the square root, the field dtype, the Montgomery storage. A lift held
    against its own `sqrt` would agree with itself on a wrong root, which is
    the failure a parity check between two paths also cannot see.
    """
    rhs = (pow(x, 3, curve.p) + curve.a * x + curve.b) % curve.p
    return rhs == 0 or pow(rhs, (curve.p - 1) // 2, curve.p) == 1


def _lift_fixture(count: int) -> tuple[list[int], list[int]]:
    """`(xs, parities)` spanning both answers the lift can give.

    `x` walks from 1 so that roughly half the rows land on no point at all —
    the case whose coordinates are junk the caller's mask drops, and the one a
    fixture built only from real points would never reach. Both curves take
    the same `xs`: the values are far below either `p`, and which of them land
    on a point is the curve's own fact. The parities alternate so a lift that
    ignored the bit, or applied it to every row, would disagree with the
    reference on some row rather than none.
    """
    xs = list(range(1, count + 1))
    return xs, [x % 2 for x in xs]


class SecpDeviceParityTest(parameterized.TestCase):
    @parameterized.named_parameters(*_CURVES)
    def test_multiple_agrees_across_the_namespace(self, curve: secp.Curve) -> None:
        batch = 8
        points = _generators(curve, batch)
        scalars = _scalars(curve, batch)

        host = secp.affine_ints(curve, secp.multiple(curve, scalars, points))
        device = secp.affine_ints(
            curve, secp.multiple(curve, scalars, fnp.asarray(points))
        )
        self.assertEqual(host, device)

    @parameterized.named_parameters(*_CURVES)
    def test_double_multiple_agrees_across_the_namespace(
        self, curve: secp.Curve
    ) -> None:
        # The two-term form is the one every verification equation reduces to,
        # and the one where `G` is a host constant that has to reach whichever
        # side the batch is on. The key points are distinct multiples rather
        # than `G` again, so swapping the two terms would not still pass.
        batch = 8
        keys = secp.multiple(curve, _scalars(curve, batch), _generators(curve, batch))
        g_scalars = _scalars(curve, batch)
        p_scalars = list(reversed(g_scalars))

        host = secp.affine_ints(
            curve, secp.double_multiple(curve, g_scalars, p_scalars, keys)
        )
        device = secp.affine_ints(
            curve,
            secp.double_multiple(curve, g_scalars, p_scalars, fnp.asarray(keys)),
        )
        self.assertEqual(host, device)

    @parameterized.named_parameters(*_CURVES)
    def test_is_identity_agrees_across_the_namespace(self, curve: secp.Curve) -> None:
        # The batch carries both answers rather than a single one a broadcast
        # could fake.
        folded = _identity_and_generator(curve)

        host = secp.is_identity(curve, folded)
        device = secp.is_identity(curve, fnp.asarray(folded))
        np.testing.assert_array_equal(host, device)
        self.assertEqual(list(host), [True, False])

    @parameterized.named_parameters(*_CURVES)
    def test_jacobian_to_affine_is_wrong_on_the_cpu_backend(
        self, curve: secp.Curve
    ) -> None:
        """Pins fractalyze/xla#594, which `is_identity` is written around.

        Converting a jacobian batch to affine inside the traced namespace
        returns the identity for every point on the CPU backend, while the host
        and CUDA agree on the real coordinates. `is_identity` therefore pulls
        back before converting, and this states the reason.

        The expectation is per-backend because the bug is: asserting one answer
        everywhere would make the correct leg the one that fails. On CUDA this
        asserts the conversion is right; on CPU it asserts it is still wrong,
        and starts passing — announcing the fix — only when it is not.
        """
        folded = _identity_and_generator(curve)
        want = secp.affine_ints(curve, folded)
        self.assertEqual(want, [(0, 0), (curve.gx, curve.gy)])

        got = secp.affine_ints(curve, fnp.asarray(folded).astype(curve.point))
        if any(device.platform == "cpu" for device in fnp.zeros(1).devices()):
            if got == want:
                self.skipTest(
                    "fractalyze/xla#594 is fixed on the CPU backend: drop the"
                    " host round trip in secp.is_identity and delete this test"
                )
            self.assertEqual(got, [(0, 0), (0, 0)])
        else:
            self.assertEqual(got, want)

    @parameterized.named_parameters(*_CURVES)
    def test_a_device_batch_is_not_pulled_back_by_the_callee(
        self, curve: secp.Curve
    ) -> None:
        # The rule this module follows is that the callee does not decide the
        # namespace. `double_multiple` is where that is easy to break, because
        # `G` is a host constant: lifting the batch down to it would still
        # produce the right numbers and silently cost the caller the device.
        points = fnp.asarray(_generators(curve, 4))
        result = secp.double_multiple(curve, [1] * 4, [1] * 4, points)
        self.assertTrue(arrays.traced(result))


class LiftParityTest(parameterized.TestCase):
    """The square-root lift answers the same on either side of the seam.

    `lift_x_to_parity` hands back host points and a host mask whichever side
    its arithmetic ran on, so unlike the point seams there is nothing in the
    return value that says where it went. The threshold is therefore forced
    rather than reached, exactly as `WycheproofVerdictParityTest` forces it —
    every batch the merge gate builds is smaller than `DEVICE_MIN_BATCH`, so
    without this the device lift would ship untested.
    """

    @parameterized.named_parameters(*_FIELD_CURVES)
    def test_lift_agrees_across_the_namespace(self, curve: secp.Curve) -> None:
        xs, parities = _lift_fixture(24)

        with mock.patch.object(secp, "DEVICE_MIN_BATCH", _FORCE_HOST):
            host_points, host_ok = secp.lift_x_to_parity(curve, xs, parities)
        with mock.patch.object(secp, "DEVICE_MIN_BATCH", _FORCE_DEVICE):
            device_points, device_ok = secp.lift_x_to_parity(curve, xs, parities)

        np.testing.assert_array_equal(host_ok, device_ok)
        self.assertEqual(
            secp.affine_ints(curve, host_points),
            secp.affine_ints(curve, device_points),
        )

    @parameterized.named_parameters(*_FIELD_CURVES)
    def test_the_lift_matches_the_curve_equation_on_both_sides(
        self, curve: secp.Curve
    ) -> None:
        """Each leg against the definition, not only against the other leg.

        Two paths that agree while both being wrong pass a parity check and
        fail this one — the same argument `WycheproofVerdictParityTest` makes
        for holding each leg against Wycheproof's own verdicts.
        """
        xs, parities = _lift_fixture(24)
        want_ok = [_has_a_point(curve, x) for x in xs]
        # The fixture is only worth running if it reaches both answers.
        self.assertIn(True, want_ok)
        self.assertIn(False, want_ok)

        for label, threshold in (("host", _FORCE_HOST), ("device", _FORCE_DEVICE)):
            with self.subTest(label):
                with mock.patch.object(secp, "DEVICE_MIN_BATCH", threshold):
                    points, ok = secp.lift_x_to_parity(curve, xs, parities)

                self.assertEqual(list(np.asarray(ok)), want_ok)
                for (x, y), want_x, want_parity, on_curve in zip(
                    secp.affine_ints(curve, points), xs, parities, want_ok
                ):
                    if not on_curve:
                        continue  # junk coordinates the caller's mask drops
                    self.assertEqual(x, want_x)
                    self.assertEqual(y % 2, want_parity)
                    self.assertTrue(secp.on_curve(curve, x, y))


class CurvePlacementTest(absltest.TestCase):
    def test_which_curves_a_traced_array_can_hold(self) -> None:
        """States the gap `place` routes around, so it is not silent.

        Without this, secp256r1's fallback to the host makes the parity test
        above compare the host path against itself and pass for the wrong
        reason. When frx admits the P-256 point types this fails, which is the
        prompt to widen `_CURVES` and drop the special case.
        """
        self.assertTrue(secp.SECP256K1.traceable)
        self.assertFalse(secp.SECP256R1.traceable)

    def test_the_threshold_decides_where_a_multiple_runs(self) -> None:
        # Asserted through the seam rather than through the placement helper,
        # because where the arithmetic ends up is the claim; the helper is an
        # implementation detail of these two functions.
        curve = secp.SECP256K1
        for count, want_traced in (
            (secp.DEVICE_MIN_BATCH - 1, False),
            (secp.DEVICE_MIN_BATCH, True),
        ):
            result = secp.multiple(curve, [1] * count, _generators(curve, count))
            self.assertEqual(arrays.traced(result), want_traced, f"B={count}")

    def test_a_single_signature_is_never_moved(self) -> None:
        # The seam's own definition of one verification, and the case the
        # device loses by 25x. It is also the shape the signing path reaches
        # these seams in, which is what keeps the namespace rule's hazard out.
        curve = secp.SECP256K1
        self.assertFalse(
            arrays.traced(secp.multiple(curve, [1], _generators(curve, 1)))
        )
        # And the signing readback still answers in host integers, which is
        # what it would stop doing if `B = 1` were ever placed.
        self.assertEqual(secp.host_multiple_of_g(curve, 1), (curve.gx, curve.gy))

    def test_an_untraceable_curve_stays_on_the_host_at_any_size(self) -> None:
        # The case that would raise rather than run slowly.
        curve = secp.SECP256R1
        count = secp.DEVICE_MIN_BATCH * 2
        result = secp.multiple(curve, [1] * count, _generators(curve, count))
        self.assertFalse(arrays.traced(result))

    def test_the_two_probes_answer_per_dtype_not_per_curve(self) -> None:
        """P-256's field is admitted while its points are not.

        The asymmetry `_place` exists to respect: it reads the probe off the
        dtype it is handed, so the square root is not gated on a point type it
        never constructs. `test_p256_lifts_on_the_device_though_its_points_cannot`
        is the behavioural half. The point half of the gap is stated once, in
        `test_which_curves_a_traced_array_can_hold` above — restating it here
        would make the upstream fix break a test whose docstring promises it
        will not.
        """
        self.assertTrue(secp.SECP256R1.field_traceable)
        self.assertTrue(secp.SECP256K1.field_traceable)


class LiftPlacementTest(absltest.TestCase):
    """Where the lift's arithmetic runs, asserted at the seam.

    The lift returns host values either way, so the placement is read off the
    argument `sqrt` receives — the ~325-multiplication ladder is the whole
    cost being placed, so the side it runs on is the claim. `sqrt` is reached
    by module-global lookup, which is what lets it be observed without
    exposing the placement helper.
    """

    def _where_the_ladder_ran(self, curve: secp.Curve, count: int) -> bool:
        xs, parities = _lift_fixture(count)
        with mock.patch.object(secp, "sqrt", wraps=secp.sqrt) as spy:
            secp.lift_x_to_parity(curve, xs, parities)
        spy.assert_called_once()
        return arrays.traced(spy.call_args.args[1])

    def test_the_threshold_decides_where_the_lift_runs(self) -> None:
        curve = secp.SECP256K1
        for count, want_traced in (
            (1, False),
            (secp.DEVICE_MIN_BATCH - 1, False),
            (secp.DEVICE_MIN_BATCH, True),
        ):
            self.assertEqual(
                self._where_the_ladder_ran(curve, count), want_traced, f"B={count}"
            )

    def test_a_batch_below_the_threshold_does_not_probe_the_device(self) -> None:
        """`B = 1` must not pay a device round trip to be told it is staying.

        The probe is itself a lift — it asks frx to hold one element — so
        consulting it before the threshold check would put a transfer, and on
        a cold GPU process the whole backend initialization, on the signing
        path this seam exists to keep on the host. That is invisible to every
        other test here: the answer is identical either way, only the cost
        differs, and `_admits` caches after the first call so even a repeat
        run would look clean.
        """
        with mock.patch.object(secp, "_admits", side_effect=AssertionError) as probe:
            secp.lift_x_to_parity(secp.SECP256K1, [secp.SECP256K1.gx], [0])
            secp.host_multiple_of_g(secp.SECP256K1, 1)
        probe.assert_not_called()

    def test_p256_lifts_on_the_device_though_its_points_cannot(self) -> None:
        """The behavioural half of the field-versus-point probe.

        `test_the_two_probes_answer_per_dtype_not_per_curve` states that the
        two answers differ for P-256; this states that the lift follows the
        one that applies. Without it, gating on the point probe would strand
        P-256 on the host and `LiftParityTest`'s secp256r1 cases would go on
        passing by comparing the host path against itself.
        """
        self.assertTrue(
            self._where_the_ladder_ran(secp.SECP256R1, secp.DEVICE_MIN_BATCH)
        )


class WycheproofVerdictParityTest(parameterized.TestCase):
    """The verdicts a consumer sees must not depend on where the batch ran.

    The substrate tests above compare coordinates. This runs the published
    Wycheproof corpus through the whole scheme and compares the thing that
    actually leaves the library, because that is what a chain treats as
    consensus — and because every batch the merge gate builds is smaller than
    `secp.DEVICE_MIN_BATCH`, so without forcing the threshold nothing here
    would exercise the device path at the scheme level at all.

    Each leg is held against Wycheproof's published verdicts rather than
    against the other leg, which is the stronger statement: two paths that
    agree while both being wrong pass a parity check and fail this one.
    """

    @parameterized.named_parameters(
        *(
            (f"_{name}{label}", name, threshold)
            for name in wycheproof.SCHEMES
            for label, threshold in (("_host", _FORCE_HOST), ("_device", _FORCE_DEVICE))
        )
    )
    def test_verdicts_do_not_depend_on_where_the_batch_ran(
        self, curve: str, threshold: int
    ) -> None:
        runnable, _ = wycheproof.load(curve)
        with mock.patch.object(secp, "DEVICE_MIN_BATCH", threshold):
            kat.check(
                wycheproof.SCHEMES[curve], wycheproof.subset(runnable, per_bucket=2)
            )


if __name__ == "__main__":
    absltest.main()
