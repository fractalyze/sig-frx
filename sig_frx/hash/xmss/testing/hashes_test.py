# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The keyed-and-masked family computes what RFC 8391 §5.1 says it computes.

Two independent checks, and they catch different things. The first writes each
construction out from the standard against `hashlib` — an operand in the wrong
order or a truncation at the wrong length fails it, where a round trip through our
own construction would agree with itself whatever order it chose. The second is
the reference implementation's own intermediates, which catches a misreading of
the standard that both of our sides would share.

Both parameter sets run, and that is the point of OID 13 rather than thoroughness:
its padding length is 4 where OID 1's is 32, so a padding hard-coded to `n` passes
every value on the left and none on the right.
"""

from __future__ import annotations

import hashlib

import numpy as np
from absl.testing import absltest
from hash_frx import Sha256, Sha512_256

from sig_frx.hash.tweakable import ChainHash, NodeHash
from sig_frx.hash.xmss import adrs as a
from sig_frx.hash.xmss import hashes
from sig_frx.hash.xmss import params as p
from sig_frx.hash.xmss.testing import vectors as v

_SEPARATOR_F = 0
_SEPARATOR_H = 1
_SEPARATOR_HASH = 2
_SEPARATOR_PRF = 3
_SEPARATOR_PRF_KEYGEN = 4


def _family(oid: int) -> hashes.Rfc8391Hashes:
    params = p.XMSS_PARAMETER_SETS[oid]
    return hashes.sha2_hashes(params, Sha256())


def _fixture(oid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`sk_seed[i] = i`, `pub_seed[i] = 2i`, `m[i] = 3i` — the reference's own."""
    n = v.REFERENCE[oid].n
    return (
        v.fixture_bytes(n, step=1),
        v.fixture_bytes(n, step=2),
        v.fixture_bytes(n, step=3),
    )


def _address(words: list[int]) -> a.Adrs:
    return a.Adrs(
        layer=words[0],
        tree=(words[1] << 32) | words[2],
        type=words[3],
        trailing=(words[4], words[5], words[6]),
        key_and_mask=words[7],
    )


def _core(oid: int, payload: bytes) -> bytes:
    """`core_hash` — SHA-256, truncated to `n`."""
    return hashlib.sha256(payload).digest()[: v.REFERENCE[oid].n]


def _to_byte(value: int, length: int) -> bytes:
    return value.to_bytes(length, "big")


def _spec_prf(oid: int, key: bytes, message: bytes) -> bytes:
    """`PRF(KEY, M) = core_hash(toByte(3, padlen) ‖ KEY ‖ M)` — §5.1."""
    padlen = v.REFERENCE[oid].padding_len
    return _core(oid, _to_byte(_SEPARATOR_PRF, padlen) + key + message)


class ConstructionTest(absltest.TestCase):
    """Each function against the standard, written out here rather than reused."""

    def test_prf_is_the_keyed_construction(self) -> None:
        for oid in v.REFERENCE:
            with self.subTest(oid=oid):
                _, pub_seed, _ = _fixture(oid)
                message = v.fixture_bytes(32, start=7)
                got = _family(oid).prf(pub_seed, message)
                self.assertEqual(
                    bytes(np.asarray(got)[0]),
                    _spec_prf(oid, bytes(pub_seed), bytes(message)),
                )

    def test_prf_keygen_has_its_own_separator(self) -> None:
        for oid in v.REFERENCE:
            with self.subTest(oid=oid):
                n = v.REFERENCE[oid].n
                padlen = v.REFERENCE[oid].padding_len
                sk_seed, pub_seed, _ = _fixture(oid)
                address = a.encode(_address(v.ADDR_WORDS))
                message = bytes(pub_seed) + address
                got = _family(oid).prf_keygen(sk_seed, np.frombuffer(message, np.uint8))
                self.assertEqual(
                    bytes(np.asarray(got)[0]),
                    _core(
                        oid,
                        _to_byte(_SEPARATOR_PRF_KEYGEN, padlen)
                        + bytes(sk_seed)
                        + message,
                    ),
                )
                self.assertLen(message, n + 32)

    def test_f_keys_and_masks_with_the_address(self) -> None:
        # `F(SEED, ADRS, M) = core_hash(toByte(0, padlen) ‖ KEY ‖ (M XOR BM))`,
        # `KEY` at `keyAndMask = 0` and `BM` at 1.
        for oid in v.REFERENCE:
            with self.subTest(oid=oid):
                n = v.REFERENCE[oid].n
                padlen = v.REFERENCE[oid].padding_len
                _, pub_seed, _ = _fixture(oid)
                address = _address(v.ADDR_WORDS)
                payload = v.fixture_bytes(n, start=1)

                got = _family(oid).f(pub_seed, a.encode_batch(address), payload)

                key = _spec_prf(oid, bytes(pub_seed), _keyed(address, 0))
                mask = _spec_prf(oid, bytes(pub_seed), _keyed(address, 1))
                self.assertEqual(
                    bytes(np.asarray(got)[0]),
                    _core(
                        oid,
                        _to_byte(_SEPARATOR_F, padlen)
                        + key
                        + bytes(x ^ y for x, y in zip(bytes(payload), mask)),
                    ),
                )

    def test_h_takes_a_mask_twice_as_long(self) -> None:
        # The second half of `BM` comes from `keyAndMask = 2`, in that order — the
        # one place a two-block hash differs from two one-block ones.
        for oid in v.REFERENCE:
            with self.subTest(oid=oid):
                n = v.REFERENCE[oid].n
                padlen = v.REFERENCE[oid].padding_len
                _, pub_seed, _ = _fixture(oid)
                address = _address(v.ADDR_WORDS)
                payload = v.fixture_bytes(2 * n, start=1)

                got = _family(oid).h(pub_seed, a.encode_batch(address), payload)

                key = _spec_prf(oid, bytes(pub_seed), _keyed(address, 0))
                mask = _spec_prf(oid, bytes(pub_seed), _keyed(address, 1)) + _spec_prf(
                    oid, bytes(pub_seed), _keyed(address, 2)
                )
                self.assertEqual(
                    bytes(np.asarray(got)[0]),
                    _core(
                        oid,
                        _to_byte(_SEPARATOR_H, padlen)
                        + key
                        + bytes(x ^ y for x, y in zip(bytes(payload), mask)),
                    ),
                )

    def test_hash_message_binds_the_index(self) -> None:
        for oid in v.REFERENCE:
            with self.subTest(oid=oid):
                n = v.REFERENCE[oid].n
                padlen = v.REFERENCE[oid].padding_len
                sk_seed, pub_seed, _ = _fixture(oid)
                body = v.HASH_MESSAGE_BODY
                index = v.HASH_MESSAGE_INDEX

                got = _family(oid).hash_message(
                    pub_seed, sk_seed, index, np.frombuffer(body, np.uint8)
                )

                self.assertEqual(
                    bytes(np.asarray(got)[0]),
                    _core(
                        oid,
                        _to_byte(_SEPARATOR_HASH, padlen)
                        + bytes(pub_seed)
                        + bytes(sk_seed)
                        + _to_byte(index, n)
                        + body,
                    ),
                )

    def test_the_address_is_what_separates_two_identical_inputs(self) -> None:
        # The tweak is the security property: the same payload hashed at two
        # positions must not collide, or a hash computed in one place could be
        # replayed in another.
        family = _family(0x01)
        _, pub_seed, _ = _fixture(0x01)
        payload = v.fixture_bytes(32)
        here = a.encode_batch(a.ots(1, 2, 3, 4, hash_index=5))
        there = a.encode_batch(a.ots(1, 2, 3, 4, hash_index=6))
        self.assertNotEqual(
            bytes(np.asarray(family.f(pub_seed, here, payload))[0]),
            bytes(np.asarray(family.f(pub_seed, there, payload))[0]),
        )


def _keyed(address: a.Adrs, value: int) -> bytes:
    """The address encoded with `keyAndMask` set, the way §5.1 derives each."""
    return bytes(a.with_key_and_mask(a.encode_batch(address), value)[0])


class ReferenceTest(absltest.TestCase):
    """The reference implementation's own intermediates, in dependency order.

    Checked bottom-up so the first failure names the bug: an address that packs
    wrongly fails `prf` too, and there is no point reading past the first red.
    """

    def test_prf_matches(self) -> None:
        for oid, vectors in v.REFERENCE.items():
            with self.subTest(oid=oid):
                _, pub_seed, _ = _fixture(oid)
                got = _family(oid).prf(pub_seed, v.fixture_bytes(32, start=7))
                self.assertEqual(bytes(np.asarray(got)[0]), vectors.prf)

    def test_prf_keygen_matches(self) -> None:
        for oid, vectors in v.REFERENCE.items():
            with self.subTest(oid=oid):
                sk_seed, pub_seed, _ = _fixture(oid)
                message = np.concatenate(
                    [
                        pub_seed,
                        np.frombuffer(a.encode(_address(v.ADDR_WORDS)), np.uint8),
                    ]
                )
                got = _family(oid).prf_keygen(sk_seed, message)
                self.assertEqual(bytes(np.asarray(got)[0]), vectors.prf_keygen)

    def test_f_matches(self) -> None:
        for oid, vectors in v.REFERENCE.items():
            with self.subTest(oid=oid):
                _, pub_seed, _ = _fixture(oid)
                # `thash_f` reads the first `n` bytes of the two-block fixture.
                got = _family(oid).f(
                    pub_seed,
                    a.encode_batch(_address(v.ADDR_WORDS)),
                    v.fixture_bytes(vectors.n, start=1),
                )
                self.assertEqual(bytes(np.asarray(got)[0]), vectors.thash_f)

    def test_h_matches(self) -> None:
        for oid, vectors in v.REFERENCE.items():
            with self.subTest(oid=oid):
                _, pub_seed, _ = _fixture(oid)
                got = _family(oid).h(
                    pub_seed,
                    a.encode_batch(_address(v.ADDR_WORDS)),
                    v.fixture_bytes(2 * vectors.n, start=1),
                )
                self.assertEqual(bytes(np.asarray(got)[0]), vectors.thash_h)

    def test_hash_message_matches(self) -> None:
        for oid, vectors in v.REFERENCE.items():
            with self.subTest(oid=oid):
                sk_seed, pub_seed, _ = _fixture(oid)
                got = _family(oid).hash_message(
                    pub_seed,
                    sk_seed,
                    v.HASH_MESSAGE_INDEX,
                    np.frombuffer(v.HASH_MESSAGE_BODY, np.uint8),
                )
                self.assertEqual(bytes(np.asarray(got)[0]), vectors.hash_message)


class BatchTest(absltest.TestCase):
    def test_each_entry_is_hashed_under_its_own_address(self) -> None:
        family = _family(0x01)
        _, pub_seed, _ = _fixture(0x01)
        addresses = [a.ots(0, 0, 0, chain, 0) for chain in range(5)]
        batch = a.encode_batch(a.ots(0, 0, 0, np.arange(5), 0))
        payloads = np.arange(5 * 32, dtype=np.uint8).reshape(5, 32)

        got = np.asarray(family.f(pub_seed, batch, payloads))

        self.assertEqual(got.shape, (5, 32))
        for index, address in enumerate(addresses):
            self.assertEqual(
                bytes(got[index]),
                bytes(
                    np.asarray(
                        family.f(pub_seed, a.encode_batch(address), payloads[index])
                    )[0]
                ),
                f"entry {index}",
            )

    def test_a_shared_payload_is_hashed_once_per_position(self) -> None:
        # `PRF` is called with one seed and one address per chain, so the key is
        # shared while the address is not — the case a batch of matching shapes
        # never exercises.
        family = _family(0x01)
        _, pub_seed, _ = _fixture(0x01)
        batch = a.encode_batch(a.ots(0, 0, 0, np.arange(4), 0))
        got = np.asarray(family.prf(pub_seed, batch))
        self.assertEqual(got.shape, (4, 32))
        self.assertLen({bytes(row) for row in got}, 4)

    def test_a_seed_per_entry_is_carried_through(self) -> None:
        # What verifying a batch under different public keys needs: one seed per
        # row rather than one for the batch.
        family = _family(0x01)
        seeds = np.arange(3 * 32, dtype=np.uint8).reshape(3, 32)
        batch = a.encode_batch(a.ots(0, 0, 0, np.arange(3), 0))
        payloads = np.zeros((3, 32), dtype=np.uint8)

        got = np.asarray(family.f(seeds, batch, payloads))

        for index in range(3):
            self.assertEqual(
                bytes(got[index]),
                bytes(
                    np.asarray(
                        family.f(
                            seeds[index],
                            a.encode_batch(a.ots(0, 0, 0, index, 0)),
                            payloads[index],
                        )
                    )[0]
                ),
                f"entry {index}",
            )


class SeamTest(absltest.TestCase):
    def test_the_family_satisfies_the_protocols_it_is_used_through(self) -> None:
        # `wots.chain` asks for `ChainHash` and `tree.py` for `NodeHash`; that is
        # what lets both be shared with FIPS 205 rather than reimplemented.
        family = _family(0x01)
        self.assertIsInstance(family, ChainHash)
        self.assertIsInstance(family, NodeHash)

    def test_the_injected_hash_is_the_one_that_runs(self) -> None:
        # The family is parameterized by the hash rather than hard-coding one,
        # and the way to show that is a DIFFERENT hash of the same width
        # changing the answer. SHA-512/256 is the pair available: 32 bytes out,
        # so it drops into the same `n`, and a distinct function underneath.
        _, pub_seed, _ = _fixture(0x01)
        batch = a.encode_batch(a.ots(1, 2, 3, 4, 5))
        payload = v.fixture_bytes(32)
        sha256 = hashes.Rfc8391Hashes(Sha256(), n=32, padding_len=32)
        sha512_256 = hashes.Rfc8391Hashes(Sha512_256(), n=32, padding_len=32)
        self.assertNotEqual(
            bytes(np.asarray(sha256.f(pub_seed, batch, payload))[0]),
            bytes(np.asarray(sha512_256.f(pub_seed, batch, payload))[0]),
        )

    def test_value_equality_survives_reconstruction(self) -> None:
        self.assertEqual(_family(0x01), _family(0x01))
        self.assertEqual(hash(_family(0x01)), hash(_family(0x01)))
        self.assertNotEqual(_family(0x01), _family(0x0D))

    def test_the_padding_length_is_part_of_the_family(self) -> None:
        # Two families that differ only in padding length are different families;
        # if they compared equal, one carried as pytree aux would be traced for the
        # other.
        self.assertNotEqual(
            hashes.Rfc8391Hashes(Sha256(), n=24, padding_len=4),
            hashes.Rfc8391Hashes(Sha256(), n=24, padding_len=32),
        )

    def test_a_truncation_longer_than_the_digest_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds"):
            hashes.Rfc8391Hashes(Sha256(), n=64, padding_len=64)

    def test_a_wrongly_sized_prf_input_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "32-byte input"):
            _family(0x01).prf(np.zeros(32, np.uint8), np.zeros(31, np.uint8))
        with self.assertRaisesRegex(ValueError, "64-byte input"):
            _family(0x01).prf_keygen(np.zeros(32, np.uint8), np.zeros(63, np.uint8))

    def test_one_index_per_message_is_required(self) -> None:
        # A batch of signatures is a batch of one-time keys, so an index shared
        # across it would digest every entry under the wrong one.
        family = _family(0x01)
        seed = np.zeros(32, np.uint8)
        with self.assertRaisesRegex(ValueError, "one index per message"):
            family.hash_message(seed, seed, 0, np.zeros((3, 8), np.uint8))

    def test_a_wrongly_sized_payload_is_an_error(self) -> None:
        family = _family(0x01)
        batch = a.encode_batch(a.ots(0, 0, 0, 0, 0))
        with self.assertRaisesRegex(ValueError, "32 bytes"):
            family.f(np.zeros(32, np.uint8), batch, np.zeros(64, np.uint8))
        with self.assertRaisesRegex(ValueError, "64 bytes"):
            family.h(np.zeros(32, np.uint8), batch, np.zeros(32, np.uint8))


class ParameterSetTest(absltest.TestCase):
    def test_the_table_carries_every_registered_oid(self) -> None:
        self.assertEqual(sorted(p.XMSS_PARAMETER_SETS), list(range(0x01, 0x16)))

    def test_the_padding_length_is_not_n(self) -> None:
        # The whole reason the 192-bit sets are carried: at `n = 24` the padding is
        # 4 bytes, so an implementation that padded to `n` fails only here.
        by_n = {
            params.n: params.padding_len for params in p.XMSS_PARAMETER_SETS.values()
        }
        self.assertEqual(by_n, {24: 4, 32: 32, 64: 64})

    def test_the_reference_agrees_about_the_two_runnable_sets(self) -> None:
        for oid, vectors in v.REFERENCE.items():
            with self.subTest(oid=oid):
                params = p.XMSS_PARAMETER_SETS[oid]
                self.assertEqual(params.n, vectors.n)
                self.assertEqual(params.padding_len, vectors.padding_len)
                self.assertEqual(params.wots.len, vectors.wots_len)
                self.assertEqual(params.wots.w, 16)

    def test_names_and_heights_agree(self) -> None:
        self.assertEqual(p.XMSS_PARAMETER_SETS[0x01].name, "XMSS-SHA2_10_256")
        self.assertEqual(p.XMSS_PARAMETER_SETS[0x0D].name, "XMSS-SHA2_10_192")
        for params in p.XMSS_PARAMETER_SETS.values():
            self.assertIn(params.height, (10, 16, 20), params.name)
            self.assertEqual(params.name.split("_")[1], str(params.height))

    def test_a_set_over_the_wrong_core_hash_is_refused(self) -> None:
        # OID 7 is SHAKE128. Building it over SHA-256 would be a self-consistent
        # implementation of a parameter set that does not exist.
        with self.assertRaisesRegex(ValueError, "SHAKE128"):
            hashes.sha2_hashes(p.XMSS_PARAMETER_SETS[0x07], Sha256())


if __name__ == "__main__":
    absltest.main()
