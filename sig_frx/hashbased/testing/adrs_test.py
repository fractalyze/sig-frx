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

import numpy as np
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
    """The batched encoding against the per-address one, field by field.

    `encode_batch` writes a whole batch with array arithmetic while `encode` writes
    one address the way §4.2 spells it out. The second is the reference: it is what
    the cases above pin against the standard, so requiring the two to agree is what
    carries that agreement over to the form every component actually calls.
    """

    def _reference(self, addresses: list[a.Adrs], *, compressed: bool) -> np.ndarray:
        """One row per address, each encoded on its own."""
        rows = [a.encode(address, compressed=compressed) for address in addresses]
        return np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(
            len(rows), len(rows[0])
        )

    def test_a_varying_field_becomes_a_column(self) -> None:
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                batch = a.encode_batch(
                    a.wots_hash(0, 0, 0, np.arange(4), 0), compressed=compressed
                )
                self.assertEqual(batch.shape, (4, 22 if compressed else 32))
                np.testing.assert_array_equal(
                    batch,
                    self._reference(
                        [a.wots_hash(0, 0, 0, chain, 0) for chain in range(4)],
                        compressed=compressed,
                    ),
                )

    def test_every_field_can_vary_at_once(self) -> None:
        # Six columns, all different, so a field written into the wrong slot cannot
        # coincide with the right answer.
        rows = 6
        columns = [np.arange(rows) * (offset + 1) + offset for offset in range(5)]
        batch = a.encode_batch(
            a.fors_tree(
                layer=columns[0],
                tree=columns[1],
                key_pair=columns[2],
                height=columns[3],
                index=columns[4],
            ),
            compressed=True,
        )
        np.testing.assert_array_equal(
            batch,
            self._reference(
                [
                    a.fors_tree(*(int(column[row]) for column in columns))
                    for row in range(rows)
                ],
                compressed=True,
            ),
        )

    def test_a_scalar_field_covers_the_whole_batch(self) -> None:
        batch = a.encode_batch(
            a.hash_tree(layer=2, tree=9, height=1, index=np.arange(3)), compressed=True
        )
        self.assertEqual(batch.shape, (3, 22))
        # The layer, tree and type bytes are the same in every row; only the index
        # word moves.
        for row in range(1, 3):
            np.testing.assert_array_equal(batch[row, :18], batch[0, :18])
            self.assertNotEqual(bytes(batch[row, 18:]), bytes(batch[0, 18:]))

    def test_all_scalars_is_a_batch_of_one(self) -> None:
        address = a.wots_pk(1, 2, 3)
        batch = a.encode_batch(address, compressed=True)
        self.assertEqual(batch.shape, (1, 22))
        self.assertEqual(bytes(batch[0]), a.encode(address, compressed=True))

    def test_a_field_too_wide_for_its_slot_is_an_error(self) -> None:
        # The same refusal `encode` makes, and for the same reason: cutting a field
        # down would tweak two different positions identically.
        with self.assertRaisesRegex(ValueError, "does not fit"):
            a.encode_batch(
                a.hash_tree(0, 0, 0, np.array([0, 1 << 32])), compressed=True
            )

    def test_a_negative_field_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsigned"):
            a.encode_batch(a.hash_tree(0, 0, 0, np.array([0, -1])), compressed=True)

    def test_a_field_wider_than_the_columns_carry_is_an_error(self) -> None:
        # `encode` is not width-bounded and takes this; the batched path carries
        # 64-bit columns, and no defined parameter set comes near either limit.
        a.encode(
            a.hash_tree(layer=0, tree=1 << 64, height=0, index=0), compressed=False
        )
        with self.assertRaisesRegex(ValueError, "wider than 64 bits"):
            a.encode_batch(
                a.hash_tree(layer=0, tree=1 << 64, height=0, index=np.arange(2)),
                compressed=False,
            )


if __name__ == "__main__":
    absltest.main()
