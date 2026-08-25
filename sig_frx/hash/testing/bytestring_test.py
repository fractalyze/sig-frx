# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Byte-string arithmetic against Python's arbitrary-precision integers.

The reference is `int.from_bytes`, which is exact at any width — which is the
whole point, since the reason this module exists is that an array lane is not.
So every case runs the same value both ways and the widths deliberately include
the ones no lane holds: the hypertree index is 54 to 64 bits at the defined
parameter sets, and 2^32 is where a traced integer field stops being able to
carry it.

Traced beside host at every case, because a difference between the two is exactly
the bug this is here to prevent.
"""

from __future__ import annotations

import frx
import numpy as np
from absl.testing import absltest, parameterized

from sig_frx.hash import bytestring

# `(width, bits)` a defined parameter set asks for, plus the boundaries around
# them: the tree index is 7 or 8 bytes reduced to 54, 56, 63 or 64 bits, and a
# leaf index is 1 or 2 bytes reduced to 3, 4, 8 or 9.
_INDEX_WIDTHS = ((7, 54), (7, 56), (8, 63), (8, 64), (2, 9), (1, 3), (1, 4), (1, 8))

# The shifts `_climb` performs: `h'` at the defined sets, plus a byte-aligned one
# and one that crosses two bytes.
_SHIFTS = (3, 4, 8, 9, 16, 17)


def _rows(width: int, count: int = 5) -> np.ndarray:
    rng = np.random.default_rng(width * 31 + count)
    # High bytes deliberately set: a value that fits 32 bits proves nothing here.
    return rng.integers(0, 256, size=(count, width), dtype=np.uint8)


def _ints(values: np.ndarray) -> list[int]:
    return [int.from_bytes(bytes(row), "big") for row in np.asarray(values)]


class MaskToTest(parameterized.TestCase):
    @parameterized.parameters(*_INDEX_WIDTHS)
    def test_it_reduces_the_way_the_standard_says(self, width: int, bits: int) -> None:
        rows = _rows(width)
        got = _ints(bytestring.mask_to(rows, bits))
        self.assertEqual(got, [value % (1 << bits) for value in _ints(rows)])

    @parameterized.parameters(*_INDEX_WIDTHS)
    def test_traced_matches_host(self, width: int, bits: int) -> None:
        rows = _rows(width)
        np.testing.assert_array_equal(
            np.asarray(frx.jit(lambda v: bytestring.mask_to(v, bits))(rows)),
            np.asarray(bytestring.mask_to(rows, bits)),
        )


class LowBitsTest(parameterized.TestCase):
    @parameterized.parameters(*_INDEX_WIDTHS)
    def test_it_reads_the_low_bits_as_a_number(self, width: int, bits: int) -> None:
        if bits > 32:
            self.skipTest("a column cannot hold it, which is what the widths test")
        rows = _rows(width)
        got = [int(value) for value in np.asarray(bytestring.low_bits(rows, bits))]
        self.assertEqual(got, [value % (1 << bits) for value in _ints(rows)])

    def test_a_width_no_column_holds_is_an_error(self) -> None:
        # Refused rather than truncated: silent truncation is the failure this
        # module exists to remove, so it must not reappear at its own boundary.
        with self.assertRaisesRegex(ValueError, "do not fit"):
            bytestring.low_bits(_rows(8), 33)

    @parameterized.parameters(3, 4, 8, 9)
    def test_traced_matches_host(self, bits: int) -> None:
        rows = _rows(8)
        np.testing.assert_array_equal(
            np.asarray(frx.jit(lambda v: bytestring.low_bits(v, bits))(rows)),
            np.asarray(bytestring.low_bits(rows, bits)),
        )


class ShiftRightTest(parameterized.TestCase):
    @parameterized.parameters(*_SHIFTS)
    def test_it_divides_by_a_power_of_two(self, bits: int) -> None:
        for width in (7, 8):
            with self.subTest(width=width):
                rows = _rows(width)
                got = _ints(bytestring.shift_right(rows, bits))
                self.assertEqual(got, [value >> bits for value in _ints(rows)])

    def test_shifting_past_the_width_leaves_nothing(self) -> None:
        rows = _rows(4)
        np.testing.assert_array_equal(
            np.asarray(bytestring.shift_right(rows, 32)), np.zeros_like(rows)
        )

    @parameterized.parameters(*_SHIFTS)
    def test_traced_matches_host(self, bits: int) -> None:
        rows = _rows(8)
        np.testing.assert_array_equal(
            np.asarray(frx.jit(lambda v: bytestring.shift_right(v, bits))(rows)),
            np.asarray(bytestring.shift_right(rows, bits)),
        )

    def test_it_stays_bytes_rather_than_widening(self) -> None:
        # The property that makes it usable at all: the result feeds an address
        # slot, which takes bytes, so nothing here may become a number.
        shifted = bytestring.shift_right(_rows(8), 9)
        self.assertEqual(shifted.dtype, np.uint8)
        self.assertEqual(shifted.shape, (5, 8))


class LaneWidthTest(absltest.TestCase):
    """Why this module exists, stated as a test rather than as a comment.

    Everything above is byte arithmetic agreeing with Python integers, which would
    also pass if the values happened to be small. These two say what goes wrong
    when they are not, so a later simplification back onto an integer lane fails
    here rather than in a signature nobody checks.
    """

    def test_an_integer_lane_truncates_where_the_bytes_do_not(self) -> None:
        wide = (1 << 63) + 12345
        row = np.frombuffer(wide.to_bytes(8, "big"), dtype=np.uint8)[None, :]

        # What a traced integer lane does with it: frx runs without x64, so this
        # is uint32 and silently keeps the low half.
        lane = np.asarray(frx.jit(lambda v: v.astype(np.uint64))(np.array([wide])))
        self.assertNotEqual(int(lane[0]), wide)
        self.assertEqual(int(lane[0]), wide % (1 << 32))

        # The bytes carry it whole, and the shift agrees with the exact arithmetic
        # at a width the lane could not have reached.
        self.assertEqual(_ints(row), [wide])
        self.assertEqual(_ints(bytestring.shift_right(row, 9)), [wide >> 9])

    def test_a_climb_from_a_high_index_stays_exact(self) -> None:
        # The shape SLH-DSA-SHA2-128f actually has: a 63-bit index climbed in
        # 3-bit steps. Every layer is above 2^32 for the first ten of them, which
        # is where an integer lane would have started returning wrong subtrees.
        index = (1 << 62) + (1 << 40) + 7
        trees = np.frombuffer(index.to_bytes(8, "big"), dtype=np.uint8)[None, :]
        expected = index
        for layer in range(21):
            with self.subTest(layer=layer):
                self.assertEqual(
                    int(np.asarray(bytestring.low_bits(trees, 3))[0]), expected % 8
                )
            trees = bytestring.shift_right(trees, 3)
            expected >>= 3
            self.assertEqual(_ints(trees), [expected])


class ClimbTest(parameterized.TestCase):
    """The two together, which is what a hypertree layer does.

    Climbing consumes `h'` bits: the leaf index is what comes off the bottom and
    the tree index is what is left. Run against the integer arithmetic the host
    walk used, at every `h'` the defined sets have.
    """

    @parameterized.parameters(3, 4, 8, 9)
    def test_a_walk_up_matches_the_integer_one(self, height: int) -> None:
        width, layers = 8, 6
        rows = _rows(width)
        trees, expected = rows, _ints(rows)
        for layer in range(layers):
            leaves = [int(v) for v in np.asarray(bytestring.low_bits(trees, height))]
            with self.subTest(layer=layer):
                self.assertEqual(leaves, [v % (1 << height) for v in expected])
            trees = bytestring.shift_right(trees, height)
            expected = [v >> height for v in expected]
            self.assertEqual(_ints(trees), expected)


if __name__ == "__main__":
    absltest.main()
