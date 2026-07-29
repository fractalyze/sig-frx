# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The SHA-2 tweakable hashes compute what FIPS 205 §11.2 says they compute.

Each expectation is written out from the standard against `hashlib` and `hmac`,
not taken from the code under test: the point is to catch an operand in the wrong
order or a truncation at the wrong length, and a round trip through our own
construction would agree with itself whatever order it chose.

What this cannot catch is a misreading of the standard shared by both sides. The
anchor for that is SLH-DSA reproducing published key and signature bytes, which
is where the known-answer tests come in — these keep a component honest in the
meantime, and localize a failure when that anchor eventually goes red.
"""

from __future__ import annotations

import hashlib
import hmac

import numpy as np
from absl.testing import absltest
from hash_frx.sha256 import HostSha256, Sha256

from sig_frx.hashbased import adrs as a
from sig_frx.hashbased.tweakable import Sha2TweakableHash, TweakableHash

# SLH-DSA-SHA2-128s / 128f: n = 16, and the message digest is 30 bytes.
_N = 16
_M = 30
_BLOCK = 64

_PK_SEED = bytes(range(_N))
_SK_SEED = bytes(range(100, 100 + _N))
_PK_ROOT = bytes(range(200, 200 + _N))


def _u8(data: bytes) -> np.ndarray:
    """Bytes as the uint8 array the family takes — its inputs are arrays, since
    they are as likely to be a previous hash's output as a host constant."""
    return np.frombuffer(data, dtype=np.uint8)


def _family() -> Sha2TweakableHash:
    return Sha2TweakableHash(Sha256(), n=_N, m=_M)


def _spec_tweak(adrs_c: bytes, payload: bytes) -> bytes:
    """`Trunc_n(SHA-256(PK.seed ‖ toByte(0, 64 − n) ‖ ADRS^c ‖ payload))` — §11.2."""
    return hashlib.sha256(_PK_SEED + bytes(_BLOCK - _N) + adrs_c + payload).digest()[
        :_N
    ]


def _spec_mgf1(seed: bytes, length: int) -> bytes:
    """MGF1-SHA-256 — RFC 8017 §B.2.1."""
    out = b"".join(
        hashlib.sha256(seed + c.to_bytes(4, "big")).digest()
        for c in range(-(-length // 32))
    )
    return out[:length]


class TweakedHashTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.family = _family()
        self.address = a.wots_hash(layer=2, tree=7, key_pair=3, chain=5, hash_index=1)
        self.adrs_c = a.encode(self.address, compressed=True)
        self.batch = a.encode_batch([self.address], compressed=True)

    def test_f_is_the_standards_construction(self) -> None:
        m1 = bytes(range(30, 30 + _N))
        got = self.family.f(_u8(_PK_SEED), self.batch, _u8(m1))
        self.assertEqual(bytes(np.asarray(got)[0]), _spec_tweak(self.adrs_c, m1))

    def test_h_hashes_two_blocks(self) -> None:
        m2 = bytes(range(2 * _N))
        got = self.family.h(_u8(_PK_SEED), self.batch, _u8(m2))
        self.assertEqual(bytes(np.asarray(got)[0]), _spec_tweak(self.adrs_c, m2))

    def test_t_hashes_any_number_of_blocks(self) -> None:
        for blocks in (1, 3, 35):
            payload = bytes((i * 7) % 256 for i in range(blocks * _N))
            got = self.family.t(_u8(_PK_SEED), self.batch, _u8(payload))
            self.assertEqual(
                bytes(np.asarray(got)[0]), _spec_tweak(self.adrs_c, payload), blocks
            )

    def test_prf_takes_the_secret_seed_as_its_payload(self) -> None:
        got = self.family.prf(_u8(_PK_SEED), _u8(_SK_SEED), self.batch)
        self.assertEqual(bytes(np.asarray(got)[0]), _spec_tweak(self.adrs_c, _SK_SEED))

    def test_the_address_is_what_separates_two_identical_inputs(self) -> None:
        # The tweak is the security property: the same payload hashed at two
        # positions must not collide, or a hash computed in one place could be
        # replayed in another.
        payload = _u8(bytes(range(_N)))
        elsewhere = a.encode_batch(
            [a.wots_hash(layer=2, tree=7, key_pair=3, chain=5, hash_index=2)],
            compressed=True,
        )
        self.assertNotEqual(
            bytes(np.asarray(self.family.f(_u8(_PK_SEED), self.batch, payload))[0]),
            bytes(np.asarray(self.family.f(_u8(_PK_SEED), elsewhere, payload))[0]),
        )


class BatchTest(absltest.TestCase):
    """One call, many positions — the shape every component above this uses."""

    def test_each_entry_is_hashed_under_its_own_address(self) -> None:
        family = _family()
        addresses = [
            a.wots_hash(layer=0, tree=0, key_pair=0, chain=chain, hash_index=0)
            for chain in range(5)
        ]
        batch = a.encode_batch(addresses, compressed=True)
        payloads = np.arange(5 * _N, dtype=np.uint8).reshape(5, _N)

        got = np.asarray(family.f(_u8(_PK_SEED), batch, payloads))

        self.assertEqual(got.shape, (5, _N))
        for index, address in enumerate(addresses):
            self.assertEqual(
                bytes(got[index]),
                _spec_tweak(a.encode(address, compressed=True), bytes(payloads[index])),
                f"entry {index}",
            )


class MessageHashTest(absltest.TestCase):
    def test_prf_msg_is_truncated_hmac(self) -> None:
        family = _family()
        sk_prf = bytes(range(50, 50 + _N))
        opt_rand = bytes(range(70, 70 + _N))
        message = b"the message to be signed"
        got = family.prf_msg(
            _u8(sk_prf),
            _u8(opt_rand),
            _u8(message),
        )
        self.assertEqual(
            bytes(np.asarray(got)),
            hmac.new(sk_prf, opt_rand + message, hashlib.sha256).digest()[:_N],
        )

    def test_h_msg_is_mgf1_over_the_inner_digest(self) -> None:
        family = _family()
        randomizer = bytes(range(80, 80 + _N))
        message = b"the message to be signed"
        got = family.h_msg(
            _u8(randomizer),
            _u8(_PK_SEED),
            _u8(_PK_ROOT),
            _u8(message),
        )
        inner = hashlib.sha256(randomizer + _PK_SEED + _PK_ROOT + message).digest()
        self.assertEqual(
            bytes(np.asarray(got)[0]),
            _spec_mgf1(randomizer + _PK_SEED + inner, _M),
        )

    def test_h_msg_digests_a_batch_under_one_call(self) -> None:
        # Verification is batch-first, and `h_msg` is on that path: B signatures
        # means B message digests, each under its own randomizer.
        family = _family()
        randomizers = np.arange(3 * _N, dtype=np.uint8).reshape(3, _N)
        messages = np.arange(3 * 8, dtype=np.uint8).reshape(3, 8)
        got = np.asarray(
            family.h_msg(randomizers, _u8(_PK_SEED), _u8(_PK_ROOT), messages)
        )
        self.assertEqual(got.shape, (3, _M))
        for index in range(3):
            randomizer = bytes(randomizers[index])
            inner = hashlib.sha256(
                randomizer + _PK_SEED + _PK_ROOT + bytes(messages[index])
            ).digest()
            self.assertEqual(
                bytes(got[index]),
                _spec_mgf1(randomizer + _PK_SEED + inner, _M),
                f"entry {index}",
            )


class SeamTest(absltest.TestCase):
    def test_the_family_satisfies_the_protocol(self) -> None:
        self.assertIsInstance(_family(), TweakableHash)

    def test_swapping_the_injected_hash_changes_nothing_but_the_hash(self) -> None:
        # The whole reason the family takes a `ByteHash`: the parameter-set
        # families are one implementation. Device and host SHA-256 are the pair
        # available to check it with; a SHAKE family drops into the same seam.
        payload = _u8(bytes(range(_N)))
        batch = a.encode_batch([a.wots_pk(1, 2, 3)], compressed=True)
        device = Sha2TweakableHash(Sha256(), n=_N, m=_M)
        host = Sha2TweakableHash(HostSha256(), n=_N, m=_M)
        self.assertEqual(
            bytes(np.asarray(device.f(_u8(_PK_SEED), batch, payload))[0]),
            bytes(np.asarray(host.f(_u8(_PK_SEED), batch, payload))[0]),
        )

    def test_the_family_names_the_address_encoding_it_tweaks_with(self) -> None:
        # A component reads this to build its addresses, so it never names the
        # parameter-set family.
        self.assertTrue(_family().compressed_address)

    def test_value_equality_survives_reconstruction(self) -> None:
        self.assertEqual(_family(), _family())
        self.assertEqual(hash(_family()), hash(_family()))
        self.assertNotEqual(_family(), Sha2TweakableHash(Sha256(), n=_N, m=_M + 1))

    def test_a_truncation_longer_than_the_digest_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds"):
            Sha2TweakableHash(Sha256(), n=64, m=_M)


if __name__ == "__main__":
    absltest.main()
