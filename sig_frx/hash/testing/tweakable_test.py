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
from collections.abc import Callable

import numpy as np
from absl.testing import absltest
from hash_frx import Sha256, Sha512, Sha512_256

from sig_frx.hash import adrs as a
from sig_frx.hash.tweakable import Sha2TweakableHash, TweakableHash

# SLH-DSA-SHA2-128s / 128f: n = 16, and the message digest is 30 bytes.
_N = 16
_M = 30

# SLH-DSA-SHA2-192s: n = 24 with a 39-byte digest, the smallest §11.2.2 row.
_WIDE_N = 24
_WIDE_M = 39
_WIDE_PK_SEED = bytes(range(_WIDE_N))

_PK_SEED = bytes(range(_N))
_SK_SEED = bytes(range(100, 100 + _N))
_PK_ROOT = bytes(range(200, 200 + _N))


def _u8(data: bytes) -> np.ndarray:
    """Bytes as the uint8 array the family takes — its inputs are arrays, since
    they are as likely to be a previous hash's output as a host constant."""
    return np.frombuffer(data, dtype=np.uint8)


def _family() -> Sha2TweakableHash:
    return Sha2TweakableHash(Sha256(), n=_N, m=_M)


def _tweak_address() -> tuple[bytes, np.ndarray]:
    """One WOTS+ hash address, in the two encodings the assertions need.

    The oracle takes the host encoding and the family takes the batched one, so
    every tweaked-function test wants both of the same address.
    """
    address = a.wots_hash(layer=2, tree=7, key_pair=3, chain=5, hash_index=1)
    return a.encode(address, compressed=True), a.encode_batch(address, compressed=True)


def _spec_tweak(
    adrs_c: bytes,
    payload: bytes,
    *,
    hash_fn: Callable[[bytes], "hashlib._Hash"] = hashlib.sha256,
    n: int = _N,
    seed: bytes = _PK_SEED,
) -> bytes:
    """`Trunc_n(H(PK.seed ‖ toByte(0, blocksize − n) ‖ ADRS^c ‖ payload))` — §11.2.

    One formula for both subsections: §11.2.1 runs every function at the
    defaults, and §11.2.2 keeps them for `PRF` and `F` while `H` and `T_l` pass
    SHA-512. Written once so a reader checks the construction against the
    standard once.

    The block comes off `hash_fn` rather than from the caller, because §11.2 pads
    to the block of the hash it is padding for — a pair that disagreed would be a
    formula the standard does not have.
    """
    padding = hash_fn(b"").block_size - n
    return hash_fn(seed + bytes(padding) + adrs_c + payload).digest()[:n]


def _wide_family() -> Sha2TweakableHash:
    """§11.2.2's two-hash family: SHA-256 for `PRF` and `F`, SHA-512 for the rest."""
    return Sha2TweakableHash(Sha256(), n=_WIDE_N, m=_WIDE_M, wide=Sha512())


def _spec_mgf1(
    seed: bytes,
    length: int,
    *,
    hash_fn: Callable[[bytes], "hashlib._Hash"] = hashlib.sha256,
) -> bytes:
    """MGF1 over `hash_fn` — RFC 8017 §B.2.1."""
    size = hash_fn(b"").digest_size
    out = b"".join(
        hash_fn(seed + c.to_bytes(4, "big")).digest() for c in range(-(-length // size))
    )
    return out[:length]


def _spec_h_msg(
    randomizer: bytes,
    pk_root: bytes,
    message: bytes,
    *,
    hash_fn: Callable[[bytes], "hashlib._Hash"] = hashlib.sha256,
    seed: bytes = _PK_SEED,
    m: int = _M,
) -> bytes:
    """`MGF1(R ‖ PK.seed ‖ H(R ‖ PK.seed ‖ PK.root ‖ M), m)` — §11.2's `H_msg`.

    Both subsections again, and both hashes are the same one: §11.2.2 moves the
    inner digest and the MGF1 to SHA-512 together.
    """
    inner = hash_fn(randomizer + seed + pk_root + message).digest()
    return _spec_mgf1(randomizer + seed + inner, m, hash_fn=hash_fn)


class TweakedHashTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.family = _family()
        self.adrs_c, self.batch = _tweak_address()

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
            a.wots_hash(layer=2, tree=7, key_pair=3, chain=5, hash_index=2),
            compressed=True,
        )
        self.assertNotEqual(
            bytes(np.asarray(self.family.f(_u8(_PK_SEED), self.batch, payload))[0]),
            bytes(np.asarray(self.family.f(_u8(_PK_SEED), elsewhere, payload))[0]),
        )


class BatchTest(absltest.TestCase):
    """One call, many positions — the shape every component above this uses."""

    def test_a_shared_payload_is_hashed_once_per_position(self) -> None:
        # `PRF` is called with one secret seed and one address per chain, so the
        # payload is shared while the address is not — the case a batch of
        # matching shapes never exercises.
        family = _family()
        addresses = [a.wots_prf(0, 0, 0, chain) for chain in range(4)]
        batch = a.encode_batch(a.wots_prf(0, 0, 0, np.arange(4)), compressed=True)

        got = np.asarray(family.prf(_u8(_PK_SEED), _u8(_SK_SEED), batch))

        self.assertEqual(got.shape, (4, _N))
        for index, address in enumerate(addresses):
            self.assertEqual(
                bytes(got[index]),
                _spec_tweak(a.encode(address, compressed=True), _SK_SEED),
                f"chain {index}",
            )

    def test_each_entry_is_hashed_under_its_own_address(self) -> None:
        family = _family()
        addresses = [
            a.wots_hash(layer=0, tree=0, key_pair=0, chain=chain, hash_index=0)
            for chain in range(5)
        ]
        batch = a.encode_batch(
            a.wots_hash(layer=0, tree=0, key_pair=0, chain=np.arange(5), hash_index=0),
            compressed=True,
        )
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
        self.assertEqual(
            bytes(np.asarray(got)[0]),
            _spec_h_msg(randomizer, _PK_ROOT, message),
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
            self.assertEqual(
                bytes(got[index]),
                _spec_h_msg(randomizer, _PK_ROOT, bytes(messages[index])),
                f"entry {index}",
            )


class SeamTest(absltest.TestCase):
    def test_the_family_satisfies_the_protocol(self) -> None:
        self.assertIsInstance(_family(), TweakableHash)

    def test_swapping_the_injected_hash_changes_nothing_but_the_hash(self) -> None:
        # The whole reason the family takes a `ByteHash`: the parameter-set
        # families are one implementation, and the way to show that is a
        # DIFFERENT hash of the same width changing the answer. SHA-512/256 is
        # the pair available: 32 bytes out, so it drops into the same `n`, and a
        # distinct function underneath.
        payload = _u8(bytes(range(_N)))
        batch = a.encode_batch(a.wots_pk(1, 2, 3), compressed=True)
        sha256 = Sha2TweakableHash(Sha256(), n=_N, m=_M)
        sha512_256 = Sha2TweakableHash(Sha512_256(), n=_N, m=_M)
        self.assertNotEqual(
            bytes(np.asarray(sha256.f(_u8(_PK_SEED), batch, payload))[0]),
            bytes(np.asarray(sha512_256.f(_u8(_PK_SEED), batch, payload))[0]),
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

    def test_a_wide_hash_too_short_to_truncate_is_an_error(self) -> None:
        # The second hash truncates to the same n, so it is checked too — and the
        # message says which of the two is short, since either can be.
        with self.assertRaisesRegex(ValueError, "wide truncates"):
            Sha2TweakableHash(Sha512(), n=48, m=_M, wide=Sha256())


class Sha512FamilyTest(absltest.TestCase):
    """FIPS 205 §11.2.2 — the categories 3 and 5 family, over two hashes.

    Checked against `hashlib` rather than against §11.2.1's family, because "the
    bytes differ" would pass for a routing that moved the wrong four functions.
    Each function is pinned to the hash the standard gives it, which is what makes
    a mis-routed `F` or `H` fail here rather than only in the published vectors.
    """

    def setUp(self) -> None:
        super().setUp()
        self.family = _wide_family()
        self.seed = _u8(_WIDE_PK_SEED)
        self.adrs_c, self.batch = _tweak_address()

    def _spec(
        self,
        payload: bytes,
        *,
        hash_fn: Callable[[bytes], "hashlib._Hash"] = hashlib.sha256,
    ) -> bytes:
        """§11.2.2's tweaked construction at this row, over whichever hash runs it.

        SHA-256 for `PRF` and `F` and SHA-512 for `H` and `T_l` — one argument
        apart, because the padding block follows the hash. Which function gets
        which is the caller's claim, and the test names below are where it is
        made.
        """
        return _spec_tweak(
            self.adrs_c, payload, hash_fn=hash_fn, n=_WIDE_N, seed=_WIDE_PK_SEED
        )

    def test_prf_stays_on_sha256(self) -> None:
        # `prf` takes the secret seed where `f` takes `M1`; the construction under
        # both is the same, which is why they share an oracle and not a method.
        payload = bytes(range(30, 30 + _WIDE_N))
        got = self.family.prf(self.seed, _u8(payload), self.batch)
        self.assertEqual(bytes(np.asarray(got)[0]), self._spec(payload))

    def test_f_stays_on_sha256(self) -> None:
        payload = bytes(range(30, 30 + _WIDE_N))
        got = self.family.f(self.seed, self.batch, _u8(payload))
        self.assertEqual(bytes(np.asarray(got)[0]), self._spec(payload))

    def test_h_moves_to_sha512(self) -> None:
        payload = bytes(range(2 * _WIDE_N))
        got = self.family.h(self.seed, self.batch, _u8(payload))
        self.assertEqual(
            bytes(np.asarray(got)[0]), self._spec(payload, hash_fn=hashlib.sha512)
        )

    def test_t_moves_to_sha512(self) -> None:
        payload = bytes(range(2 * _WIDE_N))
        got = self.family.t(self.seed, self.batch, _u8(payload))
        self.assertEqual(
            bytes(np.asarray(got)[0]), self._spec(payload, hash_fn=hashlib.sha512)
        )

    def test_prf_msg_is_truncated_hmac_sha512(self) -> None:
        sk_prf = bytes(range(50, 50 + _WIDE_N))
        opt_rand = bytes(range(80, 80 + _WIDE_N))
        message = b"the content that gets signed"
        got = self.family.prf_msg(_u8(sk_prf), _u8(opt_rand), _u8(message))
        expected = hmac.new(sk_prf, opt_rand + message, hashlib.sha512).digest()
        self.assertEqual(bytes(np.asarray(got)), expected[:_WIDE_N])

    def test_h_msg_is_mgf1_sha512_over_the_inner_sha512_digest(self) -> None:
        randomizer = bytes(range(10, 10 + _WIDE_N))
        pk_root = bytes(range(200, 200 + _WIDE_N))
        message = b"the content that gets signed"
        got = self.family.h_msg(_u8(randomizer), self.seed, _u8(pk_root), _u8(message))
        expected = _spec_h_msg(
            randomizer,
            pk_root,
            message,
            hash_fn=hashlib.sha512,
            seed=_WIDE_PK_SEED,
            m=_WIDE_M,
        )
        self.assertEqual(bytes(np.asarray(got)[0]), expected)

    def test_the_family_is_not_equal_to_one_over_a_single_hash(self) -> None:
        over_one_hash = Sha2TweakableHash(Sha256(), n=_WIDE_N, m=_WIDE_M)
        self.assertNotEqual(self.family, over_one_hash)
        self.assertEqual(self.family, _wide_family())
        self.assertEqual(hash(self.family), hash(_wide_family()))


if __name__ == "__main__":
    absltest.main()
