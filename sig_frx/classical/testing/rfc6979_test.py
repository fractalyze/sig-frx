# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""RFC 6979's nonce generation against the RFC's own vectors.

Two vector sources, chosen to localize differently. Appendix A.1 is the RFC's
detailed worked example: a 163-bit order, so `qlen` is not a multiple of eight
and `bits2int`'s truncation and `int2octets`' 21-byte width are both load
bearing — a transform bug fails here by construction. Appendix A.2.5 is the
P-256 set the ECDSA tests sign with, pinned here at the `k` level so a wrong
signature there is attributable: if these pass and a signature differs, the
nonce is not the suspect.
"""

from __future__ import annotations

import hashlib

from absl.testing import absltest

from sig_frx.classical.ecdsa import rfc6979

# RFC 6979 Appendix A.1: the detailed example's group order and private key.
_A1_Q = 0x4000000000000000000020108A2E0CC0D99F8A5EF
_A1_X = 0x09A4D6792295A7F730FC3F2B49CBC0F62E862272F

# RFC 6979 Appendix A.2.5: NIST P-256's order and the example private key.
_P256_Q = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
_P256_X = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721


def _first_nonce(q: int, x: int, message: bytes) -> int:
    h1 = hashlib.sha256(message).digest()
    return next(rfc6979.nonces(q, x, h1, hashlib.sha256))


class Rfc6979Test(absltest.TestCase):
    def test_appendix_a1_detailed_example(self) -> None:
        self.assertEqual(
            _first_nonce(_A1_Q, _A1_X, b"sample"),
            0x23AF4074C90A02B3FE61D286D5C87F425E6BDD81B,
        )

    def test_appendix_a25_p256_sha256(self) -> None:
        self.assertEqual(
            _first_nonce(_P256_Q, _P256_X, b"sample"),
            0xA6E3C57DD01ABE90086538398355DD4C3B17AA873382B0F24D6129493D8AAD60,
        )
        self.assertEqual(
            _first_nonce(_P256_Q, _P256_X, b"test"),
            0xD16B6AE827F17175E040871A1C7EC3500192C4C92677336EC2537ACAEE0008E0,
        )

    def test_candidates_stay_in_range(self) -> None:
        # The stream never yields outside [1, q-1] — §3.2 compares to q and
        # redraws rather than reducing, so the property is worth pinning.
        stream = rfc6979.nonces(
            _A1_Q, _A1_X, hashlib.sha256(b"range").digest(), hashlib.sha256
        )
        for _ in range(8):
            candidate = next(stream)
            self.assertTrue(1 <= candidate <= _A1_Q - 1)


if __name__ == "__main__":
    absltest.main()
