# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Ed25519 against RFC 8032's own vectors, and the substrate beneath it.

The curve constants are held to what the standard asserts of them — `d` is
the §5.1 table's decimal, the base point satisfies the equation and has order
`L` — and the extended-coordinate formulas to the affine rule transcribed
into `edwards_reference`. The scheme is then pinned to §7.1's TEST 1-3: key
pair, signature bytes, and acceptance, with the rejections derived from them.
TEST 1 signs the empty message, which is what exercises the zero-length
message axis end to end.
"""

from __future__ import annotations

import random

import numpy as np
from absl.testing import absltest

from sig_frx.classical import edwards
from sig_frx.classical.eddsa import ed25519
from sig_frx.classical.testing import edwards_reference as ref

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


def _bytes(hex_string: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(hex_string), dtype=np.uint8)


def _scheme() -> ed25519.Ed25519:
    return ed25519.Ed25519()


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


class GroupLawTest(absltest.TestCase):
    def _affine(self, point: edwards.ExtPoint) -> list[tuple[int, int]]:
        xs = np.asarray(point.x / point.z).astype(object)
        ys = np.asarray(point.y / point.z).astype(object)
        return [(int(x), int(y)) for x, y in zip(xs, ys)]

    def test_extended_formulas_match_the_affine_rule(self) -> None:
        curve = edwards.ED25519
        rng = random.Random("edwards")
        base = (curve.gx, curve.gy)
        points = [
            ref.scalar_mul(curve.p, curve.d, rng.randrange(1, curve.order), base)
            for _ in range(3)
        ]
        pairs = [
            (points[0], points[1]),
            (points[2], points[2]),
            (ref.IDENTITY, points[0]),
            (points[1], ref.IDENTITY),
        ]
        field = curve.field
        lift = lambda pts: edwards.ExtPoint(  # noqa: E731
            np.array([q[0] for q in pts], dtype=field),
            np.array([q[1] for q in pts], dtype=field),
            np.array([1] * len(pts), dtype=field),
            np.array([q[0] * q[1] % curve.p for q in pts], dtype=field),
        )
        got = edwards.add(
            curve, lift([a for a, _ in pairs]), lift([b for _, b in pairs])
        )
        want = [ref.add(curve.p, curve.d, a, b) for a, b in pairs]
        self.assertEqual(self._affine(got), want)
        doubled = edwards.double(curve, lift([p for p, _ in pairs]))
        self.assertEqual(
            self._affine(doubled),
            [ref.add(curve.p, curve.d, p, p) for p, _ in pairs],
        )


class Ed25519Test(absltest.TestCase):
    def test_reproduces_the_published_key_pairs(self) -> None:
        scheme = _scheme()
        for secret_hex, public_hex, _, _ in _VECTORS:
            public, secret = scheme.keygen(_bytes(secret_hex))
            self.assertEqual(public.tobytes().hex(), public_hex)
            self.assertEqual(secret.tobytes().hex(), secret_hex)

    def test_reproduces_the_published_signatures(self) -> None:
        scheme = _scheme()
        for secret_hex, _, message_hex, signature_hex in _VECTORS:
            got = scheme.sign(
                _bytes(secret_hex),
                _bytes(message_hex),
                randomness=None,
                context=None,
            )
            self.assertEqual(got.tobytes().hex(), signature_hex)

    def test_accepts_the_published_signatures(self) -> None:
        scheme = _scheme()
        for _, public_hex, message_hex, signature_hex in _VECTORS:
            verdict = scheme.verify(
                _bytes(public_hex)[None, :],
                _bytes(message_hex)[None, :],
                _bytes(signature_hex)[None, :],
                context=None,
            )
            self.assertTrue(bool(np.asarray(verdict)[0]), msg=public_hex)

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

    def test_rejects_a_tampered_message_and_key(self) -> None:
        scheme = _scheme()
        _, public_hex, message_hex, signature_hex = _VECTORS[1]
        public = _bytes(public_hex)[None, :].copy()
        message = _bytes(message_hex)[None, :].copy()
        signature = _bytes(signature_hex)[None, :]
        message[0, 0] ^= 1
        self.assertFalse(
            bool(np.asarray(scheme.verify(public, message, signature, context=None))[0])
        )
        message[0, 0] ^= 1
        public[0, 0] ^= 1
        self.assertFalse(
            bool(np.asarray(scheme.verify(public, message, signature, context=None))[0])
        )

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
