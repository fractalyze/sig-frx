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

Recovery (SEC 1 §4.1.6) publishes no vectors, so its authority is
cross-consistency (`conventions.md`): round trips against keygen's own key,
the wrapping-x case held to verification's independent second-candidate
branch, and the rejection set. The identity-result guard is exercised at the
scalar seam below the message hash, because reaching it through H would take
a preimage.
"""

from __future__ import annotations

import hashlib

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest

from sig_frx.classical import group, weierstrass
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


class DigestSurfaceTest(absltest.TestCase):
    """The digest-level names held to the message-level seam.

    When the digest is H(message) for the record's own H and the injected
    HMAC face matches the record's, the two surfaces must be one function —
    the only cross-check the pre-hashed variant admits without new vectors.
    """

    def test_sign_digest_recoverable_matches_the_message_path(self) -> None:
        scheme = _p256()
        message = np.frombuffer(b"sample", dtype=np.uint8)
        digest = np.frombuffer(hashlib.sha256(b"sample").digest(), dtype=np.uint8)
        want_sig, want_id = scheme.sign_recoverable(
            _seed(), message, randomness=None, context=None
        )
        got_sig, got_id = scheme.sign_digest_recoverable(
            _seed(), digest, nonce_hash=hashlib.sha256
        )
        self.assertEqual(got_sig.tobytes(), want_sig.tobytes())
        self.assertEqual(got_id, want_id)

    def test_recover_digest_matches_the_message_path(self) -> None:
        scheme = _p256()
        message = np.frombuffer(b"sample", dtype=np.uint8)
        digest = np.frombuffer(hashlib.sha256(b"sample").digest(), dtype=np.uint8)
        signature, recovery_id = scheme.sign_recoverable(
            _seed(), message, randomness=None, context=None
        )
        keys, ok = scheme.recover_digest(
            digest[None], signature[None], np.array([recovery_id])
        )
        self.assertEqual(list(ok), [True])
        self.assertEqual(keys[0].tobytes(), _PUBLIC)


def _curve_point_from(curve: weierstrass.Curve, start: int) -> tuple[int, int]:
    """The first `x >= start` on the curve, with a square root of its rhs.

    Host integers, because this is test-vector construction: the x >= n
    recovery ids sit behind a ~2^-64 draw at best, so the only way to gate
    them is to build a wrapping x directly (the issue's point — mishandling
    them is untestable through honest signatures).
    """
    x = start
    while True:
        rhs = (pow(x, 3, curve.p) + curve.a * x + curve.b) % curve.p
        root = pow(rhs, (curve.p + 1) // 4, curve.p)
        if root * root % curve.p == rhs:
            return x, root
        x += 1


def _non_residue_r(curve: weierstrass.Curve) -> int:
    """The first `r >= 1` whose curve equation rhs has no square root."""
    r = 1
    while True:
        rhs = (pow(r, 3, curve.p) + curve.a * r + curve.b) % curve.p
        if rhs != 0 and pow(rhs, (curve.p - 1) // 2, curve.p) != 1:
            return r
        r += 1


def _recover_rows(*signatures: bytes) -> tuple[np.ndarray, np.ndarray]:
    """A recovery batch around b"sample": `[B, L]` messages, `[B, 64]` sigs."""
    batch = len(signatures)
    messages = np.broadcast_to(
        np.frombuffer(b"sample", dtype=np.uint8), (batch, 6)
    ).copy()
    return messages, np.stack([np.frombuffer(s, dtype=np.uint8) for s in signatures])


class SignRecoverableTest(absltest.TestCase):
    def test_the_signature_matches_sign_and_the_id_is_a_parity(self) -> None:
        scheme = _p256()
        message = np.frombuffer(b"sample", dtype=np.uint8)
        plain = scheme.sign(_seed(), message, randomness=None, context=None)
        signature, recovery_id = scheme.sign_recoverable(
            _seed(), message, randomness=None, context=None
        )
        self.assertEqual(signature.tobytes(), plain.tobytes())
        # The x >= n ids sit behind a ~2^-64 draw, so an honest signature's
        # id is a parity bit.
        self.assertIn(recovery_id, (0, 1))

    def test_rejects_a_context(self) -> None:
        with self.assertRaises(ValueError):
            _p256().sign_recoverable(
                _seed(),
                np.zeros(4, dtype=np.uint8),
                randomness=None,
                context=np.array([1], dtype=np.uint8),
            )


class RecoverTest(absltest.TestCase):
    def test_recovers_the_published_key(self) -> None:
        scheme = _p256()
        for message in _VECTORS:
            data = np.frombuffer(message, dtype=np.uint8)
            signature, recovery_id = scheme.sign_recoverable(
                _seed(), data, randomness=None, context=None
            )
            keys, ok = scheme.recover(
                data[None], signature[None], np.array([recovery_id]), context=None
            )
            self.assertEqual(list(ok), [True])
            self.assertEqual(keys[0].tobytes(), _PUBLIC)

    def test_round_trips_random_keys_in_one_batch(self) -> None:
        for curve in (weierstrass.SECP256K1, weierstrass.SECP256R1):
            scheme = core.Ecdsa(curve, core.SHA256)
            seeds = [
                np.frombuffer(d.to_bytes(32, "big"), dtype=np.uint8)
                for d in (1, 0xDEADBEEF, curve.n - 1)
            ]
            pairs = [scheme.keygen(seed) for seed in seeds]
            messages = np.stack(
                [np.frombuffer(b"message%d" % i, dtype=np.uint8) for i in range(3)]
            )
            signed = [
                scheme.sign_recoverable(secret, message, randomness=None, context=None)
                for (_, secret), message in zip(pairs, messages)
            ]
            keys, ok = scheme.recover(
                messages,
                np.stack([signature for signature, _ in signed]),
                np.array([recovery_id for _, recovery_id in signed]),
                context=None,
            )
            self.assertEqual(list(ok), [True, True, True])
            for (public, _), got in zip(pairs, keys):
                self.assertEqual(got.tobytes(), public.tobytes())

    def test_low_s_normalization_flips_the_recovery_id(self) -> None:
        # A.2.5's "sample" s is the high half, so the low-S instance flips it —
        # which negates R, so the id's parity bit must flip with it, and both
        # forms must recover the same key.
        message = np.frombuffer(b"sample", dtype=np.uint8)
        plain_sig, plain_id = _p256().sign_recoverable(
            _seed(), message, randomness=None, context=None
        )
        low_scheme = core.Ecdsa(weierstrass.SECP256R1, core.SHA256, low_s=True)
        low_sig, low_id = low_scheme.sign_recoverable(
            _seed(), message, randomness=None, context=None
        )
        self.assertNotEqual(low_sig.tobytes(), plain_sig.tobytes())
        self.assertEqual(low_id, plain_id ^ 1)
        keys, ok = _p256().recover(
            np.stack([message, message]),
            np.stack([plain_sig, low_sig]),
            np.array([plain_id, low_id]),
            context=None,
        )
        self.assertEqual(list(ok), [True, True])
        for got in keys:
            self.assertEqual(got.tobytes(), _PUBLIC)

    def test_a_wrapping_x_agrees_with_verification(self) -> None:
        # A crafted r whose point sits at x = r + n: recovery must rebuild it
        # through the second pair of ids, and verification's own r + n
        # candidate branch — an independent implementation of the same
        # sliver — must accept the recovered key. Gating either branch
        # against the other is the only cross-check the ~2^-64 draw allows.
        scheme = core.Ecdsa(weierstrass.SECP256K1, core.SHA256)
        curve = scheme.curve
        x, y = _curve_point_from(curve, curve.n + 1)
        r, s = x - curve.n, 12345
        signature = r.to_bytes(32, "big") + s.to_bytes(32, "big")
        messages, signatures = _recover_rows(signature)
        keys, ok = scheme.recover(
            messages, signatures, np.array([2 | (y & 1)]), context=None
        )
        self.assertEqual(list(ok), [True])
        verdict = scheme.verify(keys, messages, signatures, context=None)
        self.assertEqual(list(np.asarray(verdict)), [True])
        # The same bytes under the non-wrapping id name a different point, so
        # they must not recover the same key.
        other_keys, other_ok = scheme.recover(
            messages, signatures, np.array([y & 1]), context=None
        )
        if other_ok[0]:
            self.assertNotEqual(other_keys[0].tobytes(), keys[0].tobytes())

    def test_rejections_in_one_batch(self) -> None:
        scheme = _p256()
        curve = scheme.curve
        n = curve.n
        good = _signature(b"sample")
        _, good_id = scheme.sign_recoverable(
            _seed(),
            np.frombuffer(b"sample", dtype=np.uint8),
            randomness=None,
            context=None,
        )
        zero_r = b"\x00" * 32 + good[32:]
        zero_s = good[:32] + b"\x00" * 32
        r_is_n = n.to_bytes(32, "big") + good[32:]
        s_is_n = good[:32] + n.to_bytes(32, "big")
        # r = p - n puts the wrapped candidate at exactly x = p — out of the
        # field, so the second-id sliver's own bound rejects it.
        beyond_p = (curve.p - n).to_bytes(32, "big") + good[32:]
        off_curve = _non_residue_r(curve).to_bytes(32, "big") + good[32:]
        messages, signatures = _recover_rows(
            good, zero_r, zero_s, r_is_n, s_is_n, good, beyond_p, off_curve
        )
        ids = np.array([good_id, 0, 0, 0, 0, 4, 2, 0])
        keys, ok = scheme.recover(messages, signatures, ids, context=None)
        self.assertEqual(
            list(ok), [True, False, False, False, False, False, False, False]
        )
        # A rejected entry's key is zeroed, not a garbage point.
        self.assertFalse(keys[1:].any())
        self.assertEqual(keys[0].tobytes(), _PUBLIC)

    def test_the_identity_result_is_rejected(self) -> None:
        # sR = eG makes the recovered point the identity, which has no
        # encoding. e is pinned to the digest by construction, so reaching
        # this through `recover` would take a hash preimage — the guard is
        # exercised at the scalar seam instead, per the docstring's rule that
        # unreachable-in-production is not untested.
        scheme = core.Ecdsa(weierstrass.SECP256K1, core.SHA256)
        curve = scheme.curve
        k, s = 7, 1234
        point = weierstrass.scalar_mul(curve, group.int_bits(k), curve.generator)
        ((rx, ry),) = group.to_affine_ints(point)
        r = rx % curve.n
        e = s * k % curve.n
        signature = r.to_bytes(32, "big") + s.to_bytes(32, "big")
        _, signatures = _recover_rows(signature)
        keys, ok = scheme._recover(
            [e], signatures, np.array([2 * (rx >= curve.n) + (ry & 1)])
        )
        self.assertEqual(list(ok), [False])
        self.assertFalse(keys.any())

    def test_rejects_a_context(self) -> None:
        messages, signatures = _recover_rows(_signature(b"sample"))
        with self.assertRaises(ValueError):
            _p256().recover(
                messages,
                signatures,
                np.array([0]),
                context=np.array([1], dtype=np.uint8),
            )


if __name__ == "__main__":
    absltest.main()
