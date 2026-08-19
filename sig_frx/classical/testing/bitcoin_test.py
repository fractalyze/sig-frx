# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The Bitcoin variant: BIP-66 strictness, BIP-62 low-S, and the key codecs.

BIP-66 publishes rules, not vectors, so the gate is one constructed violation
per rule of `IsValidSignatureEncoding`, each required to be rejected with
exactly its rule's name — plus the valid edge shapes (a one-byte integer, a
`0x00` prefix exactly where a top bit demands one) accepted and round-tripped.
The consensus stake is stated in the BIP: a decoder that accepts one
non-canonical encoding accepts signatures the network rejects. BIP-62's gate
is the malleable twin: the core verifies it, the variant must not. Key
codecs are gated by round trip over both parities plus the rejection set;
the double-SHA digest by the stdlib composed with itself.
"""

from __future__ import annotations

import hashlib

import numpy as np
from absl.testing import absltest

from sig_frx.classical import weierstrass
from sig_frx.classical.ecdsa import bitcoin, core

_CURVE = weierstrass.SECP256K1


def _keypair(seed: int) -> tuple[np.ndarray, np.ndarray]:
    scheme = core.Ecdsa(_CURVE, core.SHA256)
    return scheme.keygen(np.frombuffer(seed.to_bytes(32, "big"), dtype=np.uint8))


def _digest(data: bytes) -> np.ndarray:
    batched = np.frombuffer(data, dtype=np.uint8)[None]  # the rows take [B, L]
    return np.asarray(bitcoin.message_digest(batched), dtype=np.uint8)[0]


def _padded_rows(*blobs: bytes) -> np.ndarray:
    # Wider than the longest blob on purpose: every row carries real padding,
    # so the smuggled-byte case below lands past a declared end, not on one.
    width = max(len(blob) for blob in blobs) + 4
    return np.stack(
        [
            np.frombuffer(blob + b"\x00" * (width - len(blob)), dtype=np.uint8)
            for blob in blobs
        ]
    )


class MessageDigestTest(absltest.TestCase):
    def test_double_sha256_matches_the_stdlib(self) -> None:
        want = hashlib.sha256(hashlib.sha256(b"bitcoin").digest()).digest()
        self.assertEqual(_digest(b"bitcoin").tobytes(), want)


class DerCodecTest(absltest.TestCase):
    def test_round_trips_the_edge_integers(self) -> None:
        # One-byte integers, a top-bit value that demands the 0x00 prefix,
        # and a full-width pair — the shapes where minimal encoding differs.
        cases = [(1, 1), ((1 << 255) | 5, 7), (_CURVE.n - 1, _CURVE.n // 2)]
        for r, s in cases:
            packed = np.frombuffer(
                r.to_bytes(32, "big") + s.to_bytes(32, "big"), dtype=np.uint8
            )
            blob = bitcoin.der_encode(packed, 0x81)
            decoded, sighash = bitcoin.der_decode(blob)
            self.assertEqual(decoded.tobytes(), packed.tobytes())
            self.assertEqual(sighash, 0x81)

    def test_rejects_each_bip66_rule_by_name(self) -> None:
        # One blob per check of IsValidSignatureEncoding, each violating only
        # its own rule, asserted against that rule's message.
        minimal = bytes.fromhex("300602010102010101")  # r = 1, s = 1, sighash 1
        cases = [
            (minimal[:-1], "too short"),
            (
                # 34-byte R and 33-byte S: every field consistent, one byte
                # past the cap.
                bytes.fromhex("3047")
                + b"\x02\x22\x00\x81"
                + b"\x11" * 32
                + b"\x02\x21\x00\x81"
                + b"\x22" * 31
                + b"\x01",
                "too long",
            ),
            (b"\x31" + minimal[1:], "not a DER sequence"),
            (minimal[:1] + b"\x07" + minimal[2:], "wrong total length"),
            (bytes.fromhex("300602050102010101"), "R overruns"),
            (bytes.fromhex("30070201010201010001"), "wrong integer lengths"),
            (bytes.fromhex("300603010102010101"), "R is not an integer"),
            (bytes.fromhex("300602020080020001"), "S is empty"),
            (bytes.fromhex("300602018002010101"), "R is negative"),
            (bytes.fromhex("30070202000102010101"), "R is padded"),
            (bytes.fromhex("300602010103010101"), "S is not an integer"),
            (bytes.fromhex("300602000202008001"), "R is empty"),
            (bytes.fromhex("300602010102018001"), "S is negative"),
            (bytes.fromhex("30070201010202000101"), "S is padded"),
            (
                # A 33-byte R opening 0x01: minimally encoded and positive,
                # so BIP-66 accepts it — but its value tops 2^256, which the
                # r ‖ s wire form (and [1, n-1]) can never hold.
                bytes.fromhex("3026")
                + b"\x02\x21\x01"
                + b"\x00" * 32
                + b"\x02\x01\x01"
                + b"\x01",
                "wire form",
            ),
        ]
        for blob, rule in cases:
            with self.assertRaisesRegex(ValueError, rule, msg=f"case {rule!r}"):
                bitcoin.der_decode(np.frombuffer(blob, dtype=np.uint8))


class KeyCodecTest(absltest.TestCase):
    def test_round_trips_both_parities(self) -> None:
        # Seeds picked so both parities appear: 1·G through 4·G all land on
        # even y for this curve; 6·G is the first odd one.
        publics = np.stack([_keypair(seed)[0] for seed in (1, 2, 5, 6)])
        parities = {int(p) & 1 for p in publics[:, 64]}
        self.assertEqual(parities, {0, 1})  # both header cases exercised
        compressed = bitcoin.compress(publics)
        self.assertEqual(compressed.shape, (4, 33))
        restored, ok = bitcoin.decompress(compressed)
        self.assertTrue(ok.all())
        self.assertEqual(restored.tobytes(), publics.tobytes())

    def test_rejections_zero_the_row(self) -> None:
        good = bitcoin.compress(_keypair(1)[0][None])[0]
        bad_header = good.copy()
        bad_header[0] = 5
        beyond_p = np.concatenate(
            [good[:1], np.frombuffer(b"\xff" * 32, dtype=np.uint8)]
        )
        # x = 5 has no point on secp256k1 (its rhs is a non-residue).
        rhs = (pow(5, 3, _CURVE.p) + _CURVE.b) % _CURVE.p
        self.assertNotEqual(pow(rhs, (_CURVE.p - 1) // 2, _CURVE.p), 1)
        off_curve = np.concatenate(
            [good[:1], np.frombuffer((5).to_bytes(32, "big"), dtype=np.uint8)]
        )
        keys, ok = bitcoin.decompress(np.stack([good, bad_header, beyond_p, off_curve]))
        self.assertEqual(list(ok), [True, False, False, False])
        self.assertFalse(keys[1:].any())


class SignVerifyTest(absltest.TestCase):
    def test_round_trip_both_key_forms_and_the_bip62_gate(self) -> None:
        public, secret = _keypair(0xB0B)
        digest = _digest(b"pay to script")
        blob = bitcoin.sign(secret, digest, sighash=0x01).tobytes()
        packed, sighash = bitcoin.der_decode(np.frombuffer(blob, dtype=np.uint8))
        self.assertEqual(sighash, 0x01)
        # The malleable twin: the core accepts n - s and recovers the same
        # statement; BIP-62 is the rule that the network does not.
        s = int.from_bytes(packed[32:].tobytes(), "big")
        twin = bitcoin.der_encode(
            np.frombuffer(
                packed[:32].tobytes() + (_CURVE.n - s).to_bytes(32, "big"),
                dtype=np.uint8,
            ),
            0x01,
        ).tobytes()
        tampered = blob[:6] + bytes([blob[6] ^ 1]) + blob[7:]
        rows = _padded_rows(blob, twin, tampered, blob)
        smuggled = rows[3].copy()
        smuggled[-1] = 0xAA  # a nonzero byte past the declared end
        rows[3] = smuggled
        digests = np.broadcast_to(digest, (4, 32)).copy()
        keys = np.broadcast_to(public, (4, 65)).copy()
        got = bitcoin.verify(keys, digests, rows)
        self.assertEqual(list(got), [True, False, False, False])

    def test_the_core_accepts_what_bip62_rejects(self) -> None:
        # The contrast that makes the variant's gate meaningful: same bytes,
        # core says valid, chain policy says no.
        public, secret = _keypair(0xB0B)
        digest = _digest(b"pay to script")
        blob = bitcoin.sign(secret, digest, sighash=0x01)
        packed, _ = bitcoin.der_decode(blob)
        s = int.from_bytes(packed[32:].tobytes(), "big")
        twin_packed = np.frombuffer(
            packed[:32].tobytes() + (_CURVE.n - s).to_bytes(32, "big"),
            dtype=np.uint8,
        )
        scheme = core.Ecdsa(_CURVE, core.SHA256)
        core_verdict = np.asarray(
            scheme.verify_digest(public[None], digest[None], twin_packed[None])
        )
        self.assertEqual(list(core_verdict), [True])
        variant_verdict = bitcoin.verify(
            public[None],
            digest[None],
            bitcoin.der_encode(twin_packed, 0x01)[None],
        )
        self.assertEqual(list(variant_verdict), [False])

    def test_compressed_keys_verify(self) -> None:
        public, secret = _keypair(7)
        digest = _digest(b"compressed")
        blob = bitcoin.sign(secret, digest, sighash=0x01)
        got = bitcoin.verify(bitcoin.compress(public[None]), digest[None], blob[None])
        self.assertEqual(list(got), [True])


if __name__ == "__main__":
    absltest.main()
