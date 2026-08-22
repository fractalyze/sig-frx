# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Ed25519 against RFC 8032's own vectors, and the substrate beneath it.

The curve constants are held to what the standard asserts of them — `d` is
the §5.1 table's decimal, the base point satisfies the equation and has order
`L` — and the curated dtype kernels to the affine rule transcribed into
`edwards_reference`. §7.1's TEST 1-3 then run as KatVectors through the
shared harness, which reproduces the key pairs, the signature bytes, and
acceptance, and derives the tampering and batch-axis gates — TEST 1 signs
the empty message, which is exactly the empty-input case the tampering pass
skips per input rather than per case. What stays local is what only this
scheme defines: the flipped sign bit, `S = L`, the non-canonical `y = p`
encoding, and the context refusal.
"""

from __future__ import annotations

import random

import numpy as np
from absl.testing import absltest

from sig_frx.classical import edwards
from sig_frx.classical.eddsa import ed25519
from sig_frx.classical.testing import edwards_reference as ref
from sig_frx.testing import kat

# RFC 8032 §5.1's table prints d as this decimal; the curve computes it from
# the -121665/121666 the same table defines it by.
_D_DECIMAL = (
    37095705934669439343138083508754565189542113879843219016388785533085940283555
)

# RFC 8032 §7.1, TEST 1-3: (secret key, public key, message, signature).
_VECTORS = (
    (
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
        "",
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
    ),
    (
        "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
        "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
        "72",
        "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da"
        "085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
    ),
    (
        "c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
        "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
        "af82",
        "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac"
        "18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a",
    ),
)


def _kat_vectors() -> list[kat.KatVector]:
    """§7.1's TEST 1-3 in the harness's record — the seed is the secret key."""
    return [
        kat.KatVector(
            case_id=f"RFC 8032 §7.1 message {message_hex!r}",
            parameter_set="Ed25519",
            seed=bytes.fromhex(secret_hex),
            public_key=bytes.fromhex(public_hex),
            secret_key=bytes.fromhex(secret_hex),
            message=bytes.fromhex(message_hex),
            signature=bytes.fromhex(signature_hex),
            deterministic=True,
        )
        for secret_hex, public_hex, message_hex, signature_hex in _VECTORS
    ]


def _bytes(hex_string: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(hex_string), dtype=np.uint8)


def _scheme() -> ed25519.Ed25519:
    return ed25519.Ed25519()


def _torsion_point() -> tuple[int, int]:
    """`(√-1, 0)`: on the curve, order 4 — the smallest torsion witness."""
    curve = edwards.ED25519
    return (pow(2, (curve.p - 1) // 4, curve.p), 0)


def _mixed_order_point() -> tuple[int, int]:
    """The base point plus that one: a prime-order *and* a torsion part.

    The shape every cofactor question turns on — reducing a scalar modulo
    `L` and multiplying by 8 do different things to it, and agree on
    everything else.
    """
    curve = edwards.ED25519
    return ref.add(curve.p, curve.d, (curve.gx, curve.gy), _torsion_point())


class CurveConstantsTest(absltest.TestCase):
    def test_d_matches_the_published_decimal(self) -> None:
        self.assertEqual(edwards.ED25519.d, _D_DECIMAL)

    def test_base_point_satisfies_the_equation(self) -> None:
        curve = edwards.ED25519
        self.assertTrue(ref.on_curve(curve.p, curve.d, (curve.gx, curve.gy)))

    def test_base_point_has_order_l(self) -> None:
        curve = edwards.ED25519
        base = (curve.gx, curve.gy)
        self.assertEqual(
            ref.scalar_mul(curve.p, curve.d, curve.order, base), ref.IDENTITY
        )


class SubstrateTest(absltest.TestCase):
    """The curated dtype kernels against the affine rule's transcription.

    The group law lives in zk_dtypes now, so what this holds is the pairing:
    a wrong curve config in the wheel would compute a consistent-but-wrong
    group, and only an independent transcription of the defining rule
    catches that before a signature does.
    """

    def test_multiple_matches_the_affine_rule(self) -> None:
        curve = edwards.ED25519
        rng = random.Random("edwards")
        base = (curve.gx, curve.gy)
        scalars = [rng.randrange(1, curve.order) for _ in range(3)]
        got = edwards.affine_ints(
            curve,
            edwards.multiple(curve, scalars, curve.generator),
        )
        want = [ref.scalar_mul(curve.p, curve.d, k, base) for k in scalars]
        self.assertEqual(got, want)

    def test_the_torsion_fixture_is_a_curve_point_of_order_four(self) -> None:
        curve = edwards.ED25519
        torsion = _torsion_point()
        self.assertTrue(ref.on_curve(curve.p, curve.d, torsion))
        self.assertEqual(ref.scalar_mul(curve.p, curve.d, 4, torsion), ref.IDENTITY)
        self.assertNotEqual(ref.scalar_mul(curve.p, curve.d, 2, torsion), ref.IDENTITY)

    def test_multiple_reduces_modulo_l_on_a_mixed_order_point(self) -> None:
        # The reduction is a reading of the standard, not an artifact, so it
        # is pinned where a failure localizes: on a point with a torsion
        # component, [k]P and [k mod L]P are different points, and `multiple`
        # is the second one. Verification wants exactly that
        # (`eddsa/ed25519.py`), and the interoperability vectors are what
        # decide it (`ed25519_cctv_test`).
        curve = edwards.ED25519
        mixed = _mixed_order_point()
        # Any scalar above L exercises the reduction; the digest scalar this
        # stands in for is 512 bits, so use one that wide. The assertion
        # below is what confirms the two readings actually differ here.
        wide = 2**511 + curve.order + 5
        self.assertNotEqual(
            ref.scalar_mul(curve.p, curve.d, wide, mixed),
            ref.scalar_mul(curve.p, curve.d, wide % curve.order, mixed),
        )
        points = np.array([curve.point(mixed)], dtype=curve.point)
        got = edwards.affine_ints(curve, edwards.multiple(curve, [wide], points))
        self.assertEqual(
            got, [ref.scalar_mul(curve.p, curve.d, wide % curve.order, mixed)]
        )

    def test_mul_by_cofactor_clears_the_torsion_component(self) -> None:
        curve = edwards.ED25519
        mixed = _mixed_order_point()
        points = np.array([curve.point(mixed)], dtype=curve.point)
        got = edwards.affine_ints(curve, edwards.mul_by_cofactor(curve, points))
        self.assertEqual(got, [ref.scalar_mul(curve.p, curve.d, 8, mixed)])
        # Clearing it is what leaves a prime-order point behind: [8]P for a
        # mixed-order P is 8 times its prime-order part and nothing else.
        base = (curve.gx, curve.gy)
        self.assertEqual(got, [ref.scalar_mul(curve.p, curve.d, 8, base)])

    def test_is_small_order_names_the_torsion_points_and_nothing_else(self) -> None:
        curve = edwards.ED25519
        base = (curve.gx, curve.gy)
        torsion = _torsion_point()
        cases = [
            curve.identity[0],
            curve.point(torsion),
            curve.point(base),
            curve.point(_mixed_order_point()),
        ]
        got = edwards.is_small_order(curve, np.array(cases, dtype=curve.point))
        self.assertEqual(list(np.asarray(got)), [True, True, False, False])


class Ed25519Test(absltest.TestCase):
    def test_the_published_cases_through_the_shared_harness(self) -> None:
        # Key pairs, signature bytes, acceptance, and the derived tampering
        # and batch-axis gates, all off §7.1's own cases.
        kat.check(_scheme(), _kat_vectors())

    def test_rejections_in_one_batch(self) -> None:
        scheme = _scheme()
        _, public_hex, message_hex, signature_hex = _VECTORS[2]
        good = bytes.fromhex(signature_hex)
        flipped_r = bytes([good[0] ^ 1]) + good[1:]
        flipped_s = good[:32] + bytes([good[32] ^ 1]) + good[33:]
        sign_bit = good[:31] + bytes([good[31] ^ 0x80]) + good[32:]
        s_is_order = good[:32] + edwards.ED25519.order.to_bytes(32, "little")
        rows = [good, flipped_r, flipped_s, sign_bit, s_is_order]
        message = _bytes(message_hex)
        verdicts = scheme.verify(
            np.broadcast_to(_bytes(public_hex), (len(rows), 32)).copy(),
            np.broadcast_to(message, (len(rows), len(message))).copy(),
            np.stack([np.frombuffer(r, dtype=np.uint8) for r in rows]),
            context=None,
        )
        self.assertEqual(list(np.asarray(verdicts)), [True] + [False] * 4)

    def test_rejects_a_non_canonical_y(self) -> None:
        # y = p encodes as p's little-endian bytes: a value the decoding must
        # refuse before it wraps to y = 0 (RFC 8032 §5.1.3).
        scheme = _scheme()
        _, _, message_hex, signature_hex = _VECTORS[2]
        bad_key = np.frombuffer(
            edwards.ED25519.p.to_bytes(32, "little"), dtype=np.uint8
        )[None, :]
        verdict = scheme.verify(
            bad_key,
            _bytes(message_hex)[None, :],
            _bytes(signature_hex)[None, :],
            context=None,
        )
        self.assertFalse(bool(np.asarray(verdict)[0]))

    def test_rejects_a_context(self) -> None:
        scheme = _scheme()
        with self.assertRaises(ValueError):
            scheme.sign(
                _bytes(_VECTORS[0][0]),
                np.zeros(1, dtype=np.uint8),
                randomness=None,
                context=np.array([1], dtype=np.uint8),
            )


if __name__ == "__main__":
    absltest.main()
