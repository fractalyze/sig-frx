# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The derived Curve integers against SEC 2's published parameters.

`secp.py` reads every curve integer off the zk_dtypes handles (`ecinfo` /
`pfinfo`) instead of transcribing it, so a wrong value in the wheel's
metadata would otherwise surface only as a failing signature somewhere
downstream. SEC 2 publishes no vectors for the parameters — the parameters
are the vectors, so the standard's literals live here, transcribed in this
one place, and the derivation is held against them directly.
"""

from __future__ import annotations

from absl.testing import absltest

from sig_frx.classical import secp

# SEC 2 §2.4.1, "Recommended Parameters secp256k1".
_K1_P = 0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFE_FFFFFC2F
_K1_A = 0
_K1_B = 7
_K1_N = 0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFE_BAAEDCE6_AF48A03B_BFD25E8C_D0364141
_K1_GX = 0x79BE667E_F9DCBBAC_55A06295_CE870B07_029BFCDB_2DCE28D9_59F2815B_16F81798
_K1_GY = 0x483ADA77_26A3C465_5DA4FBFC_0E1108A8_FD17B448_A6855419_9C47D08F_FB10D4B8

# SEC 2 §2.4.2, "Recommended Parameters secp256r1" — NIST's P-256
# (FIPS 186-5 §6.1.1 points at SP 800-186 §3.2.1.3 for the same values).
_R1_P = 0xFFFFFFFF_00000001_00000000_00000000_00000000_FFFFFFFF_FFFFFFFF_FFFFFFFF
_R1_A = 0xFFFFFFFF_00000001_00000000_00000000_00000000_FFFFFFFF_FFFFFFFF_FFFFFFFC
_R1_B = 0x5AC635D8_AA3A93E7_B3EBBD55_769886BC_651D06B0_CC53B0F6_3BCE3C3E_27D2604B
_R1_N = 0xFFFFFFFF_00000000_FFFFFFFF_FFFFFFFF_BCE6FAAD_A7179E84_F3B9CAC2_FC632551
_R1_GX = 0x6B17D1F2_E12C4247_F8BCE6E5_63A440F2_77037D81_2DEB33A0_F4A13945_D898C296
_R1_GY = 0x4FE342E2_FE1A7F9B_8EE7EB4A_7C0F9E16_2BCE3357_6B315ECE_CBB64068_37BF51F5


class SecpTest(absltest.TestCase):
    def _check(self, curve: secp.Curve, expected: dict[str, int]) -> None:
        for name, value in expected.items():
            self.assertEqual(getattr(curve, name), value, name)

    def test_secp256k1_matches_sec2(self) -> None:
        self._check(
            secp.SECP256K1,
            {
                "p": _K1_P,
                "a": _K1_A,
                "b": _K1_B,
                "n": _K1_N,
                "gx": _K1_GX,
                "gy": _K1_GY,
            },
        )

    def test_secp256r1_matches_sec2(self) -> None:
        self._check(
            secp.SECP256R1,
            {
                "p": _R1_P,
                "a": _R1_A,
                "b": _R1_B,
                "n": _R1_N,
                "gx": _R1_GX,
                "gy": _R1_GY,
            },
        )

    def test_generator_satisfies_the_curve_equation(self) -> None:
        # One cross-check tying the derived integers to each other rather
        # than to the standard: G must lie on y² = x³ + ax + b (mod p).
        for curve in (secp.SECP256K1, secp.SECP256R1):
            self.assertTrue(secp.on_curve(curve, curve.gx, curve.gy))


if __name__ == "__main__":
    absltest.main()
