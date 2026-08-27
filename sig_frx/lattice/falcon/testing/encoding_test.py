# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Falcon's encodings against the specification's own algorithms.

`decompress` is a variable-length decoder reshaped into an associative scan over
a nine-state machine, and that reshaping is the whole risk in this module: the
implementation never walks a cursor, so nothing about it resembles Algorithm 18
line for line. So it is held to Algorithm 18 transcribed naively into
[`falcon_reference`](falcon_reference.py), over inputs the transcription
generates rather than over the two signatures per degree upstream publishes —
which between them never make `k` large.

**The rejections get their own class, because they are a security property.**
Algorithm 18 enforces that a polynomial has at most one valid encoding, and a
decoder that takes a second one admits signature malleability: same message,
same key, different bytes, still valid. Each rejection is driven by a mutation
of a genuine encoding, together with the control that the *unmutated* one is
accepted — a rejection test that would also reject the valid case proves
nothing.

The bit order gets its own case for the same reason ML-DSA's `unpack_fields`
carries a paragraph about `base_2b`: Falcon reads a field most significant bit
first and FIPS 204 least, the two agree at whole-byte widths, and a decoder
built on the wrong one round-trips forever while being wrong.
"""

from __future__ import annotations

import random
from typing import Any

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from sig_frx.lattice.falcon import encoding
from sig_frx.lattice.falcon.testing import falcon_reference as ref

_PARAMETER_SETS = ref.parameter_cases()


def _room(case: dict[str, Any]) -> int:
    """How many unary bits `slen` leaves once every coefficient has its nine.

    Small — 392 bits at Falcon-512 — which is the fact a generator that draws
    magnitudes independently gets wrong: `σ ≈ 165.7` puts `E[k]` near 0.6, so
    anything much above that overflows `slen` and encodes nothing.
    """
    return ref.slen(case["signature_size"]) - 9 * case["n"]


def _sample(case: dict[str, Any], seed: int, excess: int) -> list[int]:
    """Coefficients whose unary runs sum to exactly `excess` bits.

    Concentrated in a few coefficients rather than spread evenly, because a long
    run is what the scan has to get right and an average one is not: at
    `excess = 0` every coefficient is nine bits and the state machine never
    dwells, and at the maximum a single run can span hundreds of bits.
    """
    n = case["n"]
    rng = random.Random(seed)
    runs = [0] * n
    remaining = excess
    while remaining > 0:
        take = min(remaining, rng.randrange(1, 9))
        runs[rng.randrange(n)] += take
        remaining -= take
    return [rng.choice([1, -1]) * (rng.randrange(128) + (run << 7)) for run in runs]


class Bits(parameterized.TestCase):
    """§3.11.1's order, which is the opposite of the sibling scheme's."""

    def test_bytes_to_bits_puts_the_most_significant_bit_first(self) -> None:
        got = encoding.bytes_to_bits_high_first(fnp.asarray([0x80, 0x01], np.uint8))
        np.testing.assert_array_equal(
            np.asarray(got), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        )

    def test_bytes_to_bits_agrees_with_the_reference(self) -> None:
        data = bytes(random.Random(11).randrange(256) for _ in range(64))
        got = encoding.bytes_to_bits_high_first(fnp.asarray(bytearray(data), np.uint8))
        np.testing.assert_array_equal(np.asarray(got), ref.bits_of(data))

    def test_bits_to_bytes_undoes_bytes_to_bits(self) -> None:
        data = bytes(random.Random(12).randrange(256) for _ in range(64))
        bits = encoding.bytes_to_bits_high_first(fnp.asarray(bytearray(data), np.uint8))
        np.testing.assert_array_equal(
            np.asarray(encoding.bits_to_bytes_high_first(bits)), bytearray(data)
        )

    @parameterized.parameters(1, 3, 5, 6, 7, 8, 14)
    def test_packing_fields_undoes_reading_them(self, width: int) -> None:
        """The two are inverses at every width §3.11 uses, not only at whole bytes.

        The widths that are not divisors of eight are the ones that matter: a
        field then straddles a byte, which is the case a per-byte implementation
        of either direction would get wrong while agreeing at 8 and 14.
        """
        rng = random.Random(width + 100)
        values = [rng.randrange(1 << width) for _ in range(8 * 8)]
        bits = encoding.pack_fields_high_first(fnp.asarray(values, np.uint32), width)
        packed = encoding.bits_to_bytes_high_first(bits)
        got = encoding.unpack_fields_high_first(packed, width)
        np.testing.assert_array_equal(np.asarray(got), values)

    @parameterized.parameters(1, 3, 7, 8, 14, 16)
    def test_fields_are_big_endian_across_the_stream(self, width: int) -> None:
        rng = random.Random(width)
        values = [rng.randrange(1 << width) for _ in range(24)]
        bits: list[int] = []
        for value in values:
            bits.extend((value >> shift) & 1 for shift in range(width - 1, -1, -1))
        bits.extend([0] * (-len(bits) % 8))
        got = encoding.unpack_fields_high_first(
            fnp.asarray(bytearray(ref.bytes_of(bits)), np.uint8), width
        )
        np.testing.assert_array_equal(np.asarray(got)[: len(values)], values)


class PublicKey(parameterized.TestCase):
    """§3.11.4, including the range check fourteen bits make possible."""

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_round_trips_through_the_reference(self, **params: Any) -> None:
        n = params["n"]
        rng = random.Random(n)
        h = [rng.randrange(ref.Q) for _ in range(n)]
        bits: list[int] = [0] * 8
        bits[4:8] = [(n.bit_length() - 1) >> s & 1 for s in range(3, -1, -1)]
        for value in h:
            bits.extend((value >> shift) & 1 for shift in range(13, -1, -1))
        blob = ref.bytes_of(bits)
        self.assertLen(blob, params["public_key_size"])
        got, ok = encoding.pk_decode(fnp.asarray(bytearray(blob), np.uint8), n)
        self.assertTrue(bool(ok))
        np.testing.assert_array_equal(np.asarray(got), h)
        self.assertEqual(ref.pk_decode(blob, n), h)

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_a_coefficient_at_or_above_q_is_refused(self, **params: Any) -> None:
        """Fourteen bits hold 16383, so `q ≤ h_i` is representable and forgeable."""
        n = params["n"]
        bits: list[int] = [0] * 8
        bits[4:8] = [(n.bit_length() - 1) >> s & 1 for s in range(3, -1, -1)]
        for index in range(n):
            value = ref.Q if index == 3 else 0
            bits.extend((value >> shift) & 1 for shift in range(13, -1, -1))
        blob = ref.bytes_of(bits)
        _, ok = encoding.pk_decode(fnp.asarray(bytearray(blob), np.uint8), n)
        self.assertFalse(bool(ok))
        self.assertIsNone(ref.pk_decode(blob, n))

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_a_wrong_degree_header_is_refused(self, **params: Any) -> None:
        n = params["n"]
        blob = bytearray(params["public_key_size"])
        blob[0] = (n.bit_length() - 1) ^ 1
        _, ok = encoding.pk_decode(fnp.asarray(blob, np.uint8), n)
        self.assertFalse(bool(ok))

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_the_encoder_writes_what_the_reference_writes(self, **params: Any) -> None:
        n = params["n"]
        rng = random.Random(n + 1)
        h = [rng.randrange(ref.Q) for _ in range(n)]
        got = encoding.pk_encode(fnp.asarray(h, np.uint32), n)
        self.assertEqual(bytes(np.asarray(got)), ref.pk_encode(h, n))
        self.assertLen(np.asarray(got), params["public_key_size"])


class PrivateKey(parameterized.TestCase):
    """§3.11.5 — three runs at two widths, and the value it forbids."""

    def _draw(self, n: int, seed: int) -> tuple[list[int], list[int], list[int]]:
        """Coefficients inside each width's *valid* range, minimum excluded.

        The forbidden value is the subject of its own case below, so it is kept
        out of the ones that are about the encoding.
        """
        rng = random.Random(seed)
        width = ref.SK_WIDTHS[n]
        limit = 1 << (width - 1)
        small = [rng.randrange(-limit + 1, limit) for _ in range(2 * n)]
        return small[:n], small[n:], [rng.randrange(-127, 128) for _ in range(n)]

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_round_trips_through_the_reference(self, **params: Any) -> None:
        n = params["n"]
        f, g, big_f = self._draw(n, n)
        blob = encoding.sk_encode(
            fnp.asarray(f, np.int32),
            fnp.asarray(g, np.int32),
            fnp.asarray(big_f, np.int32),
            n,
        )
        self.assertLen(np.asarray(blob), params["secret_key_size"])
        self.assertEqual(bytes(np.asarray(blob)), ref.sk_encode(f, g, big_f, n))

        back_f, back_g, back_big_f, ok = encoding.sk_decode(blob, n)
        self.assertTrue(bool(ok))
        for got, want, name in (
            (back_f, f, "f"),
            (back_g, g, "g"),
            (back_big_f, big_f, "F"),
        ):
            with self.subTest(polynomial=name):
                np.testing.assert_array_equal(np.asarray(got), want)
        self.assertEqual(ref.sk_decode(bytes(np.asarray(blob)), n), (f, g, big_f))

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_the_forbidden_minimum_is_refused(self, **params: Any) -> None:
        """§3.11.5 excludes `−2^(w−1)`, which two's complement would otherwise carry.

        One case per run, because the check has to reach all three: `f` and `g`
        at the degree's width and `F` at eight, where a single case would leave
        two of the three unexercised and passing.
        """
        n = params["n"]
        width = ref.SK_WIDTHS[n]
        for index, (position, size) in enumerate(((0, width), (n, width), (2 * n, 8))):
            with self.subTest(run="fgF"[index]):
                f, g, big_f = self._draw(n, n + 2)
                joined = f + g + big_f
                joined[position] = -(1 << (size - 1))
                blob = ref.sk_encode(joined[:n], joined[n : 2 * n], joined[2 * n :], n)
                _, _, _, ok = encoding.sk_decode(
                    fnp.asarray(bytearray(blob), np.uint8), n
                )
                self.assertFalse(bool(ok))
                self.assertIsNone(ref.sk_decode(blob, n))

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_a_wrong_header_is_refused(self, **params: Any) -> None:
        """`0101nnnn`, so both the nibble and the degree are part of the verdict."""
        n = params["n"]
        for label, header in (
            ("public key's nibble", n.bit_length() - 1),
            ("the neighbouring degree", 0x50 | ((n.bit_length() - 1) ^ 1)),
        ):
            with self.subTest(header=label):
                blob = bytearray(params["secret_key_size"])
                blob[0] = header
                _, _, _, ok = encoding.sk_decode(fnp.asarray(blob, np.uint8), n)
                self.assertFalse(bool(ok))

    def test_the_width_table_is_the_sections_own(self) -> None:
        """The implementation carries two rows of §3.11.5's table; this is all eight.

        A table with only the used rows cannot say whether the rule was read
        correctly, and the boundaries are where a misreading lands — so the two
        the implementation defines are checked against a transcription that
        covers the degrees either side of them.
        """
        self.assertEqual(
            ref.SK_WIDTHS,
            {2: 8, 4: 8, 8: 8, 16: 8, 32: 8, 64: 7, 128: 7, 256: 6, 512: 6, 1024: 5},
        )
        for degree, width in encoding.SK_FG_BITS.items():
            with self.subTest(degree=degree):
                self.assertEqual(width, ref.SK_WIDTHS[degree])


class Decompress(parameterized.TestCase):
    """Algorithm 18's reshaping, against Algorithm 18."""

    @parameterized.product(
        case=_PARAMETER_SETS,
        fill=(0.0, 0.01, 0.5, 1.0),
    )
    def test_agrees_with_the_reference_over_generated_polynomials(
        self, case: dict[str, Any], fill: float
    ) -> None:
        """`fill` walks the unary runs from all-empty to every bit `slen` allows."""
        n, length = case["n"], ref.slen(case["signature_size"])
        s = _sample(case, seed=n + int(fill * 100), excess=int(_room(case) * fill))
        blob = ref.compress(s, length)
        self.assertIsNotNone(blob)
        assert blob is not None
        got, ok = encoding.decompress(fnp.asarray(bytearray(blob), np.uint8), n)
        self.assertTrue(bool(ok))
        np.testing.assert_array_equal(np.asarray(got), s)
        self.assertEqual(ref.decompress(ref.bits_of(blob), length, n), s)

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_the_zero_polynomial_is_a_valid_encoding(self, **params: Any) -> None:
        """`000000001` per coefficient — the canonical spelling of zero."""
        n, length = params["n"], ref.slen(params["signature_size"])
        blob = ref.compress([0] * n, length)
        assert blob is not None
        got, ok = encoding.decompress(fnp.asarray(bytearray(blob), np.uint8), n)
        self.assertTrue(bool(ok))
        np.testing.assert_array_equal(np.asarray(got), [0] * n)


class Rejections(parameterized.TestCase):
    """The three uniqueness rules, each against the genuine encoding it mutates."""

    def _genuine(self, params: dict[str, Any]) -> tuple[list[int], list[int], int]:
        n, length = params["n"], ref.slen(params["signature_size"])
        s = _sample(params, seed=n, excess=_room(params) // 2)
        blob = ref.compress(s, length)
        assert blob is not None
        return s, ref.bits_of(blob), length

    def _verdict(self, bits: list[int], n: int) -> bool:
        blob = bytearray(ref.bytes_of(bits))
        _, ok = encoding.decompress(fnp.asarray(blob, np.uint8), n)
        return bool(ok)

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_the_unmutated_encoding_is_accepted(self, **params: Any) -> None:
        """The control: without it every rejection below proves nothing."""
        _, bits, length = self._genuine(params)
        self.assertTrue(self._verdict(bits, params["n"]))
        self.assertIsNotNone(ref.decompress(bits, length, params["n"]))

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_a_negative_zero_is_refused(self, **params: Any) -> None:
        """Line 9: `100000001` and `000000001` would both spell the coefficient 0."""
        n, length = params["n"], ref.slen(params["signature_size"])
        bits = ref.bits_of(bytes(ref.compress([0] * n, length) or b""))
        self.assertTrue(self._verdict(bits, n))
        bits[0] = 1
        self.assertFalse(self._verdict(bits, n))
        self.assertIsNone(ref.decompress(bits, length, n))

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_a_nonzero_padding_bit_is_refused(self, **params: Any) -> None:
        """Lines 12-13, which is also §3.11.3's 'partial padding is not valid'."""
        _, bits, length = self._genuine(params)
        mutated = list(bits)
        mutated[-1] = 1
        self.assertFalse(self._verdict(mutated, params["n"]))
        self.assertIsNone(ref.decompress(mutated, length, params["n"]))

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_an_unterminated_unary_run_is_refused(self, **params: Any) -> None:
        """Line 6 walks `str[8 + k]` with no bound; the string has to supply one."""
        n, length = params["n"], ref.slen(params["signature_size"])
        s = _sample(params, seed=n, excess=_room(params) // 2)
        blob = ref.compress(s, length)
        assert blob is not None
        bits = ref.bits_of(blob)
        # Clear every terminator from the last coefficient onward, so the final
        # run reaches the end of the string without ever closing.
        consumed = sum(9 + (abs(value) >> 7) for value in s[:-1])
        mutated = bits[:consumed] + [0] * (length - consumed)
        self.assertFalse(self._verdict(mutated, n))
        self.assertIsNone(ref.decompress(mutated, length, n))

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_an_all_zero_string_is_refused(self, **params: Any) -> None:
        """No coefficient ever closes, so `n` terminators are never reached."""
        n, length = params["n"], ref.slen(params["signature_size"])
        bits = [0] * length
        self.assertFalse(self._verdict(bits, n))
        self.assertIsNone(ref.decompress(bits, length, n))

    @parameterized.parameters(*_PARAMETER_SETS)
    def test_an_over_long_encoding_has_no_room_to_be_written(
        self, **params: Any
    ) -> None:
        """Line 7's `⊥`: `Compress` refuses what `Decompress` could not have read."""
        n, length = params["n"], ref.slen(params["signature_size"])
        self.assertIsNone(ref.compress([1 << 14] * n, length))


if __name__ == "__main__":
    absltest.main()
