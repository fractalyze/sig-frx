# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The ECDSA core against RFC 6979's P-256 vectors, and its own rejections.

The accepted cases are the standard's: Appendix A.2.5 publishes the key pair
and the SHA-256 signatures over "sample" and "test", so key generation,
signing, and verification are each pinned to published bytes. The rejections
are derived from those accepted cases — a moved bit in each argument, the
range bounds on `r` and `s`, and the public-key encoding checks — because a
verifier that accepts everything passes every positive vector
(`docs/reference/conventions.md`). secp256k1 has no vectors of standard
weight in this file yet; the Wycheproof sets are the planned gate, and until
they land its coverage here is a round trip labeled as the smoke test it is.

Verification runs in the host namespace throughout: the traced path is the
same code, and its cases carry the shared upstream-blocker marker.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest

from sig_frx.classical import weierstrass
from sig_frx.classical.ecdsa import core
from sig_frx.classical.testing.traced_blocker import TRACED_BLOCKED

# RFC 6979 Appendix A.2.5: the P-256 example key pair.
_X = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
_UX = 0x60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6
_UY = 0x7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299

# The SHA-256 signatures the same appendix publishes over the two messages.
_VECTORS = {
    b"sample": (
        0xEFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716,
        0xF7CB1C942D657C41D436C7A1B6E29F65F3E900DBB9AFF4064DC4AB2F843ACDA8,
    ),
    b"test": (
        0xF1ABB023518351CD71D881567B1EA663ED3EFCF6C5132B354F28D3B0B7D38367,
        0x019F4113742A2B14BD25926B49C649155F267E60D3814B4C0CC84250E46F0083,
    ),
}


def _p256() -> core.Ecdsa:
    return core.Ecdsa(weierstrass.SECP256R1, core.SHA256)


def _seed() -> np.ndarray:
    return np.frombuffer(_X.to_bytes(32, "big"), dtype=np.uint8)


def _signature(message: bytes) -> bytes:
    r, s = _VECTORS[message]
    return r.to_bytes(32, "big") + s.to_bytes(32, "big")


def _rows(message: bytes, *signatures: bytes) -> tuple:
    """A batch around one message: `[B, 65]`, `[B, L]`, `[B, 64]`."""
    public, _ = _p256().keygen(_seed())
    batch = len(signatures)
    return (
        np.broadcast_to(public, (batch, 65)).copy(),
        np.broadcast_to(
            np.frombuffer(message, dtype=np.uint8), (batch, len(message))
        ).copy(),
        np.stack([np.frombuffer(s, dtype=np.uint8) for s in signatures]),
    )


class KeygenTest(absltest.TestCase):
    def test_reproduces_the_published_key_pair(self) -> None:
        public, secret = _p256().keygen(_seed())
        want = b"\x04" + _UX.to_bytes(32, "big") + _UY.to_bytes(32, "big")
        self.assertEqual(public.tobytes(), want)
        self.assertEqual(secret.tobytes(), _X.to_bytes(32, "big"))

    def test_rejects_out_of_range_seeds(self) -> None:
        scheme = _p256()
        for bad in (0, scheme.curve.n):
            with self.assertRaises(ValueError):
                scheme.keygen(np.frombuffer(bad.to_bytes(32, "big"), dtype=np.uint8))


class SignTest(absltest.TestCase):
    def test_reproduces_the_published_signatures(self) -> None:
        scheme = _p256()
        for message in _VECTORS:
            got = scheme.sign(
                _seed(),
                np.frombuffer(message, dtype=np.uint8),
                randomness=None,
                context=None,
            )
            self.assertEqual(got.tobytes(), _signature(message), msg=message)

    def test_rejects_a_context(self) -> None:
        with self.assertRaises(ValueError):
            _p256().sign(
                _seed(),
                np.zeros(4, dtype=np.uint8),
                randomness=None,
                context=np.array([1], dtype=np.uint8),
            )

    def test_low_s_takes_the_smaller_half(self) -> None:
        scheme = core.Ecdsa(weierstrass.SECP256R1, core.SHA256, low_s=True)
        n = scheme.curve.n
        r, s = _VECTORS[b"sample"]
        self.assertGreater(s, n // 2)  # the vector is the high half
        got = scheme.sign(
            _seed(),
            np.frombuffer(b"sample", dtype=np.uint8),
            randomness=None,
            context=None,
        )
        want = r.to_bytes(32, "big") + (n - s).to_bytes(32, "big")
        self.assertEqual(got.tobytes(), want)


class VerifyTest(absltest.TestCase):
    def test_accepts_the_published_signatures(self) -> None:
        scheme = _p256()
        for message in _VECTORS:
            keys, messages, signatures = _rows(message, _signature(message))
            got = scheme.verify(keys, messages, signatures, context=None)
            self.assertTrue(bool(np.asarray(got)[0]), msg=message)

    def test_rejections_and_malleation_in_one_batch(self) -> None:
        scheme = _p256()
        n = scheme.curve.n
        r, s = _VECTORS[b"sample"]
        good = _signature(b"sample")
        flipped_r = bytes([good[0] ^ 1]) + good[1:]
        flipped_s = good[:33] + bytes([good[33] ^ 1]) + good[34:]
        zero_r = b"\x00" * 32 + good[32:]
        zero_s = good[:32] + b"\x00" * 32
        r_is_n = n.to_bytes(32, "big") + good[32:]
        # SEC 1 accepts both halves of s; rejecting the high one is a chain
        # policy that belongs to a variant, so the core must take n - s.
        malleated = good[:32] + (n - s).to_bytes(32, "big")
        keys, messages, signatures = _rows(
            b"sample", good, flipped_r, flipped_s, zero_r, zero_s, r_is_n, malleated
        )
        got = np.asarray(scheme.verify(keys, messages, signatures, context=None))
        self.assertEqual(list(got), [True, False, False, False, False, False, True])

    def test_rejects_a_tampered_message_and_key(self) -> None:
        scheme = _p256()
        keys, messages, signatures = _rows(
            b"sample",
            _signature(b"sample"),
            _signature(b"sample"),
            _signature(b"sample"),
        )
        messages[0, 0] ^= 1
        keys[1, 0] = 5  # not the uncompressed-point header
        keys[2, 64] ^= 1  # off the curve
        got = np.asarray(scheme.verify(keys, messages, signatures, context=None))
        self.assertEqual(list(got), [False, False, False])

    def test_verify_rejects_a_context(self) -> None:
        scheme = _p256()
        keys, messages, signatures = _rows(b"sample", _signature(b"sample"))
        with self.assertRaises(ValueError):
            scheme.verify(
                keys, messages, signatures, context=np.array([1], dtype=np.uint8)
            )

    @TRACED_BLOCKED
    def test_traced_matches_host(self) -> None:
        scheme = _p256()
        keys, messages, signatures = _rows(
            b"sample", _signature(b"sample"), _signature(b"test")
        )
        host = np.asarray(scheme.verify(keys, messages, signatures, context=None))
        compiled = frx.jit(lambda k, m, s: scheme.verify(k, m, s, context=None))
        traced = compiled(
            fnp.asarray(keys), fnp.asarray(messages), fnp.asarray(signatures)
        )
        self.assertEqual(list(np.asarray(traced)), list(host))


class Secp256k1SmokeTest(absltest.TestCase):
    """A round trip, which is smoke and not a gate.

    secp256k1 has no RFC 6979 vectors and no ACVP coverage; the Wycheproof
    sets are the planned authority, tracked on the issue. Until they land,
    this pins only that signing and verification agree with each other.
    """

    def test_round_trip_and_a_moved_bit(self) -> None:
        scheme = core.Ecdsa(weierstrass.SECP256K1, core.SHA256)
        seed = np.frombuffer((0x1234567890ABCDEF).to_bytes(32, "big"), np.uint8)
        public, secret = scheme.keygen(seed)
        message = np.frombuffer(b"smoke", dtype=np.uint8)
        signature = scheme.sign(secret, message, randomness=None, context=None)
        corrupted = signature.copy()
        corrupted[0] ^= 1
        got = np.asarray(
            scheme.verify(
                np.stack([public, public]),
                np.stack([message, message]),
                np.stack([signature, corrupted]),
                context=None,
            )
        )
        self.assertEqual(list(got), [True, False])


if __name__ == "__main__":
    absltest.main()
