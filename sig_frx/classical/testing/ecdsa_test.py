# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The ECDSA core against RFC 6979's P-256 vectors, and its own rejections.

The accepted cases are the standard's: Appendix A.2.5 publishes the key pair
and the SHA-256 signatures over "sample" and "test". Expressed as KatVectors,
the shared harness reproduces keygen, signing, and verification byte for byte
and derives the tampering and batch-axis gates from them — the same driver
the Wycheproof sets run through. What stays local is what only this scheme
defines: the seed range refusals, the context refusal, the low-S signing
option, the r/s range bounds, n - s accepted (malleability policy stays out
of the core), and traced-equals-host. secp256k1's authority is the
Wycheproof gate; its round trip here is smoke for the signing path those
verify-only sets cannot reach.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest

from sig_frx.classical import weierstrass
from sig_frx.classical.ecdsa import core
from sig_frx.classical.testing.traced_blocker import TRACED_BLOCKED
from sig_frx.testing import kat

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


# One fixed key pair serves every verification row; a full keygen ladder per
# helper call would re-derive it identically.
_PUBLIC = b"\x04" + _UX.to_bytes(32, "big") + _UY.to_bytes(32, "big")


def _kat_vectors() -> list[kat.KatVector]:
    """A.2.5's cases in the harness's record, one per message."""
    return [
        kat.KatVector(
            case_id=f"RFC 6979 A.2.5 P-256/SHA-256 {message!r}",
            parameter_set="ECDSA-P256-SHA256",
            seed=_X.to_bytes(32, "big"),
            public_key=_PUBLIC,
            secret_key=_X.to_bytes(32, "big"),
            message=message,
            signature=_signature(message),
            deterministic=True,
        )
        for message in _VECTORS
    ]


def _signature(message: bytes) -> bytes:
    r, s = _VECTORS[message]
    return r.to_bytes(32, "big") + s.to_bytes(32, "big")


def _rows(message: bytes, *signatures: bytes) -> tuple:
    """A batch around one message: `[B, 65]`, `[B, L]`, `[B, 64]`."""
    batch = len(signatures)
    return (
        np.broadcast_to(np.frombuffer(_PUBLIC, dtype=np.uint8), (batch, 65)).copy(),
        np.broadcast_to(
            np.frombuffer(message, dtype=np.uint8), (batch, len(message))
        ).copy(),
        np.stack([np.frombuffer(s, dtype=np.uint8) for s in signatures]),
    )


class HarnessTest(absltest.TestCase):
    def test_the_published_cases_through_the_shared_harness(self) -> None:
        # keygen, signing, and verification against the published bytes, plus
        # the derived tampering and batch-axis gates — one driver, the same
        # one the Wycheproof sets run through.
        kat.check(_p256(), _kat_vectors())


class KeygenTest(absltest.TestCase):
    def test_rejects_out_of_range_seeds(self) -> None:
        scheme = _p256()
        for bad in (0, scheme.curve.n):
            with self.assertRaises(ValueError):
                scheme.keygen(np.frombuffer(bad.to_bytes(32, "big"), dtype=np.uint8))


class SignTest(absltest.TestCase):
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

    def test_rejects_a_malformed_key_encoding(self) -> None:
        # The harness's tampering pass moves a bit; the header byte and an
        # off-curve coordinate are this scheme's own encoding refusals.
        scheme = _p256()
        keys, messages, signatures = _rows(
            b"sample", _signature(b"sample"), _signature(b"sample")
        )
        keys[0, 0] = 5  # not the uncompressed-point header
        keys[1, 64] ^= 1  # off the curve
        got = np.asarray(scheme.verify(keys, messages, signatures, context=None))
        self.assertEqual(list(got), [False, False])

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

    secp256k1's verification authority is the Wycheproof gate
    (ecdsa_wycheproof_test); those sets are verify-only, so what this round
    trip uniquely reaches is the secp256k1 signing path agreeing with the
    verifier it is gated against.
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
