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

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from sig_frx import arrays
from sig_frx.classical import secp

# secp256k1 only, and not by preference. A traced array can hold a point type
# only if frx's admission table has a row for it, and at the pinned wheel
# `secp256r1_g1_*` has none — it fails with "not a valid JAX array type" before
# any arithmetic happens. P-256's `a = -3` also needs the generated group law
# confirmed for a non-zero `a` before its parity could be claimed. Both are the
# same upstream gap; when the rows land, adding the curve here is the test that
# should gate them.
_CURVES = (("secp256k1", secp.SECP256K1),)


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


if __name__ == "__main__":
    absltest.main()
