# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Addresses pack to the byte layout FIPS 205 defines.

Field order is the whole content of this module, because getting it wrong is
invisible: a transposed pair of words yields an implementation that is perfectly
self-consistent and fails every known-answer test with nothing to point at. The
positions below are read off §4.2 for the full form and §11.2.1 for the
compressed one, and asserted as byte offsets rather than through a round trip —
a round trip through our own encoder would agree with itself no matter what
order it chose.
"""

from __future__ import annotations

from absl.testing import absltest

from sig_frx.hashbased import adrs as a


class FullLayoutTest(absltest.TestCase):
    def test_the_encoding_is_thirty_two_bytes(self) -> None:
        encoded = a.encode(a.wots_hash(1, 2, 3, 4, 5), compressed=False)
        self.assertLen(encoded, 32)
        self.assertEqual(a.ADRS_SIZE, 32)

    def test_each_field_lands_where_the_standard_puts_it(self) -> None:
        encoded = a.encode(
            a.wots_hash(
                layer=0x11,
                tree=0x2233445566778899AABBCCDD,
                key_pair=0xEE,
                chain=0xFF,
                hash_index=0x0102,
            ),
            compressed=False,
        )
        self.assertEqual(encoded[0:4], bytes.fromhex("00000011"))
        self.assertEqual(encoded[4:16], bytes.fromhex("2233445566778899aabbccdd"))
        self.assertEqual(encoded[16:20], bytes.fromhex("00000000"))  # WOTS_HASH
        self.assertEqual(encoded[20:24], bytes.fromhex("000000ee"))
        self.assertEqual(encoded[24:28], bytes.fromhex("000000ff"))
        self.assertEqual(encoded[28:32], bytes.fromhex("00000102"))

    def test_every_type_encodes_its_own_number(self) -> None:
        by_type = {
            a.AdrsType.WOTS_HASH: a.wots_hash(0, 0, 0, 0, 0),
            a.AdrsType.WOTS_PK: a.wots_pk(0, 0, 0),
            a.AdrsType.TREE: a.hash_tree(0, 0, 0, 0),
            a.AdrsType.FORS_TREE: a.fors_tree(0, 0, 0, 0, 0),
            a.AdrsType.FORS_ROOTS: a.fors_roots(0, 0, 0),
            a.AdrsType.WOTS_PRF: a.wots_prf(0, 0, 0, 0),
            a.AdrsType.FORS_PRF: a.fors_prf(0, 0, 0, 0),
        }
        self.assertLen(by_type, 7)
        for expected, address in by_type.items():
            encoded = a.encode(address, compressed=False)
            self.assertEqual(
                int.from_bytes(encoded[16:20], "big"), int(expected), str(expected)
            )

    def test_a_type_zeroes_the_words_it_does_not_use(self) -> None:
        # The trailing words carry different meanings per type, and a type that
        # does not use one must not leak a value into it — that is the whole
        # reason for the per-type constructors.
        self.assertEqual(a.hash_tree(0, 0, 7, 9).trailing, (0, 7, 9))  # padding first
        self.assertEqual(a.fors_tree(0, 0, 3, 7, 9).trailing, (3, 7, 9))
        self.assertEqual(a.wots_pk(0, 0, 3).trailing, (3, 0, 0))
        self.assertEqual(a.fors_roots(0, 0, 3).trailing, (3, 0, 0))
        self.assertEqual(a.wots_prf(0, 0, 3, 5).trailing, (3, 5, 0))
        self.assertEqual(a.fors_prf(0, 0, 3, 9).trailing, (3, 0, 9))

    def test_a_field_that_does_not_fit_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not fit"):
            a.encode(a.wots_hash(1 << 32, 0, 0, 0, 0), compressed=False)
        with self.assertRaisesRegex(ValueError, "unsigned"):
            a.encode(a.wots_hash(-1, 0, 0, 0, 0), compressed=False)


class CompressedLayoutTest(absltest.TestCase):
    def test_the_encoding_is_twenty_two_bytes(self) -> None:
        encoded = a.encode(a.wots_hash(1, 2, 3, 4, 5), compressed=True)
        self.assertLen(encoded, 22)
        self.assertEqual(a.COMPRESSED_ADRS_SIZE, 22)

    def test_it_is_the_full_form_with_the_provably_zero_bytes_dropped(self) -> None:
        # §11.2 defines the compressed form as exactly this slice of the full
        # one: `ADRS_c = ADRS[3] ‖ ADRS[8:16] ‖ ADRS[19] ‖ ADRS[20:32]`. Asserting
        # the standard's own equation pins both layouts against each other, so
        # neither can drift without the other.
        address = a.fors_tree(
            layer=0x0A, tree=0x0102030405060708, key_pair=1, height=2, index=3
        )
        full = a.encode(address, compressed=False)
        # What the compression drops is zero for any address a defined parameter
        # set can produce — which is why dropping it loses nothing.
        self.assertEqual(full[0:3], bytes(3))
        self.assertEqual(full[4:8], bytes(4))
        self.assertEqual(full[16:19], bytes(3))
        self.assertEqual(
            a.encode(address, compressed=True),
            full[3:4] + full[8:16] + full[19:20] + full[20:32],
        )

    def test_an_address_too_wide_to_compress_is_an_error(self) -> None:
        # No defined parameter set reaches this, so it is a caller that computed
        # the wrong address — and cutting it down would tweak two different
        # positions identically.
        wide = a.hash_tree(layer=0, tree=1 << 64, height=0, index=0)
        a.encode(wide, compressed=False)
        with self.assertRaisesRegex(ValueError, "does not fit"):
            a.encode(wide, compressed=True)


class BatchTest(absltest.TestCase):
    def test_a_batch_is_one_row_per_address(self) -> None:
        addresses = [a.wots_hash(0, 0, 0, chain, 0) for chain in range(4)]
        batch = a.encode_batch(addresses, compressed=True)
        self.assertEqual(batch.shape, (4, 22))
        for index, address in enumerate(addresses):
            self.assertEqual(
                bytes(batch[index]), a.encode(address, compressed=True), f"row {index}"
            )

    def test_an_empty_batch_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "no addresses"):
            a.encode_batch([], compressed=True)


if __name__ == "__main__":
    absltest.main()
