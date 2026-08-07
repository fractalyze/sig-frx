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

import frx
import numpy as np
from absl.testing import absltest

from sig_frx.hashbased import adrs as a
from sig_frx.hashbased import adrs_encoding


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

    def test_a_field_wider_than_its_lane_is_an_error(self) -> None:
        # `encode` is not width-bounded and takes this; the batched path carries an
        # integer field in an array lane, and no defined parameter set comes near
        # either limit.
        a.encode(
            a.hash_tree(layer=0, tree=1 << 64, height=0, index=0), compressed=False
        )
        with self.assertRaisesRegex(ValueError, "arrive as bytes"):
            a.encode_batch(
                a.hash_tree(layer=0, tree=1 << 64, height=0, index=np.arange(2)),
                compressed=False,
            )

    def test_a_field_past_its_lane_can_arrive_as_bytes(self) -> None:
        # What the error above points at, and what a traced caller does with the
        # tree address: hand over the bytes rather than a number the lane cannot
        # hold. The full form gives the tree twelve bytes, so a 2^64 value fits the
        # slot even though it does not fit the lane.
        wide = (1 << 64) + 7
        as_bytes = np.frombuffer(wide.to_bytes(12, "big"), dtype=np.uint8)
        batch = a.encode_batch(
            a.hash_tree(
                layer=0,
                tree=np.broadcast_to(as_bytes, (2, 12)),
                height=0,
                index=np.arange(2),
            ),
            compressed=False,
        )
        for row in range(2):
            self.assertEqual(
                bytes(np.asarray(batch)[row]),
                a.encode(
                    a.hash_tree(layer=0, tree=wide, height=0, index=row),
                    compressed=False,
                ),
            )

    def test_a_high_tree_address_survives_an_unsigned_field_beside_it(self) -> None:
        # Stacking a signed column beside an unsigned one promotes the pair to
        # float64, which rounds anything above 2^53 — and the tallest hypertree
        # addresses 2^63. The columns a caller carries are uint64 while the node
        # indices it tiles under them come out of `arange` signed, so the mix is
        # the normal case rather than an exotic one.
        high = np.array([(1 << 63) + 1, (1 << 63) + 2], dtype=np.uint64)
        batch = a.encode_batch(
            a.hash_tree(layer=0, tree=high, height=0, index=np.arange(2)),
            compressed=True,
        )
        np.testing.assert_array_equal(
            batch,
            self._reference(
                [
                    a.hash_tree(layer=0, tree=int(high[row]), height=0, index=row)
                    for row in range(2)
                ],
                compressed=True,
            ),
        )


class TracedTest(absltest.TestCase):
    """The same addresses, built from traced fields.

    What a traced verifier needs: the fields it holds are tracers, so the encoder
    has to run on them and produce exactly what the host produces. The host form
    is the reference, since `BatchTest` above already ties that to the standard.
    """

    def _jit_encode(self, *, compressed: bool):  # type: ignore[no-untyped-def]
        def build(tree, key_pair, index):  # type: ignore[no-untyped-def]
            return a.encode_batch(
                a.fors_tree(
                    layer=0, tree=tree, key_pair=key_pair, height=2, index=index
                ),
                compressed=compressed,
            )

        return frx.jit(build)

    def test_traced_fields_give_the_host_bytes(self) -> None:
        rows = 5
        tree = np.arange(rows, dtype=np.uint32) * 7 + 1
        key_pair = np.arange(rows, dtype=np.uint32) * 3
        index = np.arange(rows, dtype=np.uint32) * 11 + 2
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                traced = np.asarray(
                    self._jit_encode(compressed=compressed)(tree, key_pair, index)
                )
                host = np.asarray(
                    a.encode_batch(
                        a.fors_tree(
                            layer=0, tree=tree, key_pair=key_pair, height=2, index=index
                        ),
                        compressed=compressed,
                    )
                )
                np.testing.assert_array_equal(traced, host)

    def test_a_tree_address_above_the_lane_rides_as_bytes(self) -> None:
        # The reason a byte field exists. frx runs without x64, so a traced integer
        # field is 32 bits and a tree address — up to 2^64 at the defined parameter
        # sets — would truncate silently. As bytes it does not.
        rows = 3
        wide = [(1 << 63) + 5 * row + 1 for row in range(rows)]
        as_bytes = np.stack(
            [np.frombuffer(value.to_bytes(8, "big"), dtype=np.uint8) for value in wide]
        )
        index = np.arange(rows, dtype=np.uint32)

        def build(tree_bytes, node):  # type: ignore[no-untyped-def]
            return a.encode_batch(
                a.hash_tree(layer=0, tree=tree_bytes, height=1, index=node),
                compressed=True,
            )

        traced = np.asarray(frx.jit(build)(as_bytes, index))
        for row in range(rows):
            self.assertEqual(
                bytes(traced[row]),
                a.encode(
                    a.hash_tree(layer=0, tree=wide[row], height=1, index=row),
                    compressed=True,
                ),
                f"row {row}",
            )

    def test_a_traced_uint32_field_would_have_truncated(self) -> None:
        # The case the test above defends against, made explicit: the same value
        # through an integer field does *not* survive, so the byte field is
        # load-bearing rather than stylistic.
        wide = (1 << 63) + 1
        lane = np.full(2, wide, dtype=np.uint64)
        truncated = np.asarray(
            frx.jit(
                lambda tree: a.encode_batch(
                    a.hash_tree(layer=0, tree=tree, height=1, index=np.arange(2)),
                    compressed=True,
                )
            )(lane)
        )
        self.assertNotEqual(
            bytes(truncated[0]),
            a.encode(
                a.hash_tree(layer=0, tree=wide, height=1, index=0), compressed=True
            ),
        )


class ThirdWordTest(absltest.TestCase):
    """Replacing the word against encoding the address that holds it.

    `encode_batch` is the reference here, since `BatchTest` above is what ties that
    to §4.2: a WOTS+ chain walks `w − 1` steps off one encoded batch, and every one
    of them has to tweak with the bytes the standard names for its step.
    """

    def _chain(self, hash_index: int, *, compressed: bool) -> np.ndarray:
        return np.asarray(
            a.encode_batch(
                a.wots_hash(
                    layer=1,
                    tree=9,
                    key_pair=np.arange(4),
                    chain=np.arange(4) * 3 + 2,
                    hash_index=hash_index,
                ),
                compressed=compressed,
            )
        )

    def test_it_agrees_with_encoding_the_step(self) -> None:
        for compressed in (False, True):
            for step in (0, 1, 14):
                with self.subTest(compressed=compressed, step=step):
                    np.testing.assert_array_equal(
                        a.with_third_word(
                            self._chain(0, compressed=compressed),
                            step,
                            compressed=compressed,
                        ),
                        self._chain(step, compressed=compressed),
                    )

    def test_it_rewrites_only_the_word(self) -> None:
        encoded = self._chain(0, compressed=True)
        rewritten = a.with_third_word(encoded, 5, compressed=True)
        np.testing.assert_array_equal(rewritten[:, :-4], encoded[:, :-4])
        self.assertEqual(int.from_bytes(bytes(rewritten[0, -4:]), "big"), 5)

    def test_it_leaves_the_callers_addresses_alone(self) -> None:
        # The caller keeps its batch across the whole walk, so a step that wrote
        # into it would give every later step its own hash address.
        encoded = self._chain(0, compressed=True)
        before = bytes(encoded[0])
        a.with_third_word(encoded, 3, compressed=True)
        self.assertEqual(bytes(encoded[0]), before)

    def test_a_traced_batch_gives_the_host_bytes(self) -> None:
        # Where the walk actually runs: the addresses are traced values on the
        # verification path, so the replacement has to happen there too.
        key_pair = np.arange(4, dtype=np.uint32)

        def build(pairs):  # type: ignore[no-untyped-def]
            return a.with_third_word(
                a.encode_batch(
                    a.wots_hash(
                        layer=1,
                        tree=9,
                        key_pair=pairs,
                        chain=np.arange(4) * 3 + 2,
                        hash_index=0,
                    ),
                    compressed=True,
                ),
                7,
                compressed=True,
            )

        np.testing.assert_array_equal(
            np.asarray(frx.jit(build)(key_pair)), self._chain(7, compressed=True)
        )

    def test_a_wrongly_sized_address_is_an_error(self) -> None:
        # The two encodings are the same fields in different slots, so a batch in
        # the other one would take the word at the wrong offset.
        with self.assertRaisesRegex(ValueError, "22 bytes"):
            a.with_third_word(self._chain(0, compressed=False), 0, compressed=True)

    def test_a_word_too_wide_for_its_slot_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not fit"):
            a.with_third_word(self._chain(0, compressed=True), 1 << 32, compressed=True)


class BroadcastRuleTest(absltest.TestCase):
    """`rows` is what a component asks before it lays out a batch."""

    def test_all_scalars_is_one_row(self) -> None:
        self.assertEqual(adrs_encoding.rows((1, 2, 3)), 1)

    def test_a_column_sets_the_row_count(self) -> None:
        self.assertEqual(adrs_encoding.rows((1, np.arange(5), 3)), 5)

    def test_it_agrees_with_what_the_encoding_builds(self) -> None:
        # The two must not disagree: a caller asks `rows` how many addresses it is
        # about to build, then builds them.
        self.assertEqual(
            adrs_encoding.rows((np.arange(4), 7, 0)),
            a.encode_batch(
                a.hash_tree(layer=np.arange(4), tree=7, height=0, index=0),
                compressed=True,
            ).shape[0],
        )

    def test_a_field_that_is_not_one_value_per_row_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "one value per row"):
            adrs_encoding.rows((1, np.zeros((2, 3), dtype=np.int64)))


if __name__ == "__main__":
    absltest.main()
