# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""WOTS+ does what FIPS 205 §5 says, including when the shape differs from it.

Two kinds of check, and the second is the one that matters here. The helpers are
compared against straight Python transcriptions of Algorithms 4, 5 and 7 —
written from the standard, kept deliberately naive, and looping the way the
standard does. The batched masked chain then has to agree with that loop, which
is the whole claim of this module: the shape changed for the compiler, not the
arithmetic.

The round trip is the property WOTS+ exists for — a signature's chains, walked
the rest of the way, land on the public key — and it is asserted alongside the
forgery shape it rules out, since a construction that always returns the same
public key would satisfy the round trip alone.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest
from hash_frx.sha256 import Sha256

from sig_frx.hashbased import adrs, wots
from sig_frx.hashbased.tweakable import Sha2TweakableHash

_N = 16
_PARAMS = wots.WotsParams(n=_N)
_POSITION = wots.WotsPosition(layer=1, tree=9, key_pair=4)
_PK_SEED = np.frombuffer(bytes(range(_N)), dtype=np.uint8)
_SK_SEED = np.frombuffer(bytes(range(100, 100 + _N)), dtype=np.uint8)
_MESSAGE = np.frombuffer(bytes((i * 37 + 11) % 256 for i in range(_N)), dtype=np.uint8)


def _family() -> Sha2TweakableHash:
    return Sha2TweakableHash(Sha256(), n=_N, m=30)


def _spec_base_2b(data: bytes, b: int, out_len: int) -> list[int]:
    """Algorithm 4, transcribed."""
    consumed = 0
    bits = 0
    total = 0
    out = []
    for _ in range(out_len):
        while bits < b:
            total = (total << 8) + data[consumed]
            consumed += 1
            bits += 8
        bits -= b
        out.append((total >> bits) % (2**b))
    return out


def _spec_digits(params: wots.WotsParams, message: bytes) -> list[int]:
    """Algorithm 7 lines 1 to 7, transcribed."""
    msg = _spec_base_2b(message, params.lg_w, params.len1)
    csum = sum(params.w - 1 - digit for digit in msg)
    csum <<= (8 - ((params.len2 * params.lg_w) % 8)) % 8
    csum_bytes = csum.to_bytes(-(-params.len2 * params.lg_w // 8), "big")
    return msg + _spec_base_2b(csum_bytes, params.lg_w, params.len2)


def _spec_chain(
    tweak: Sha2TweakableHash,
    value: np.ndarray,
    start: int,
    steps: int,
    position: wots.WotsPosition,
    chain_index: int,
) -> np.ndarray:
    """Algorithm 5, transcribed: one chain, one hash per step, no masking."""
    current = value
    for step in range(start, start + steps):
        address = adrs.encode_batch(
            adrs.wots_hash(
                layer=position.layer,
                tree=position.tree,
                key_pair=position.key_pair,
                chain=chain_index,
                hash_index=step,
            ),
            compressed=True,
        )
        current = np.asarray(tweak.f(_PK_SEED, address, current[None, :]))[0]
    return current


class ParameterTest(absltest.TestCase):
    def test_the_derived_values_match_the_standard(self) -> None:
        # §5: "When lg_w = 4, w = 16, len1 = 2n, len2 = 3, and len = 2n + 3."
        for n in (16, 24, 32):
            params = wots.WotsParams(n=n)
            self.assertEqual(params.w, 16, n)
            self.assertEqual(params.len1, 2 * n, n)
            self.assertEqual(params.len2, 3, n)
            self.assertEqual(params.len, 2 * n + 3, n)


class Base2bTest(absltest.TestCase):
    def test_it_matches_the_standards_loop(self) -> None:
        data = bytes((i * 53 + 7) % 256 for i in range(32))
        for b, out_len in ((4, 32), (6, 14), (8, 10), (9, 7), (12, 5), (14, 4)):
            got = np.asarray(
                wots.base_2b(np.frombuffer(data, dtype=np.uint8), b, out_len)
            )
            self.assertEqual(list(got[0]), _spec_base_2b(data, b, out_len), f"b={b}")

    def test_too_little_input_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least"):
            wots.base_2b(np.zeros(2, dtype=np.uint8), 4, 32)


class DigitsTest(absltest.TestCase):
    def test_the_digits_and_checksum_match_the_standards_loop(self) -> None:
        got = np.asarray(wots.message_digits(_PARAMS, _MESSAGE))[0]
        self.assertEqual(list(got), _spec_digits(_PARAMS, bytes(_MESSAGE)))
        self.assertLen(got, _PARAMS.len)

    def test_lowering_a_message_digit_raises_the_checksum(self) -> None:
        # The property the checksum chains exist for: an attacker who walks a
        # message chain further has to walk a checksum chain backwards.
        low = np.zeros(_N, dtype=np.uint8)
        high = np.full(_N, 0xFF, dtype=np.uint8)
        low_digits = np.asarray(wots.message_digits(_PARAMS, low))[0]
        high_digits = np.asarray(wots.message_digits(_PARAMS, high))[0]
        self.assertGreater(
            int(low_digits[_PARAMS.len1 :].sum()),
            int(high_digits[_PARAMS.len1 :].sum()),
        )


class ChainTest(absltest.TestCase):
    def _step_addresses(self, count: int) -> list[np.ndarray]:
        return [
            adrs.encode_batch(
                adrs.wots_hash(
                    layer=_POSITION.layer,
                    tree=_POSITION.tree,
                    key_pair=_POSITION.key_pair,
                    chain=np.arange(count),
                    hash_index=step,
                ),
                compressed=True,
            )
            for step in range(_PARAMS.w - 1)
        ]

    def test_the_masked_batch_agrees_with_the_standards_per_chain_loop(self) -> None:
        # The claim this module makes: running every chain the full length and
        # selecting is the same arithmetic as walking each chain exactly as far
        # as its digit asks. The cases cover a full walk, a partial one, two
        # no-ops, and a start in the middle.
        tweak = _family()
        starts = np.array([0, 3, 0, 15, 7], dtype=np.uint32)
        steps = np.array([15, 5, 0, 0, 8], dtype=np.uint32)
        values = np.arange(5 * _N, dtype=np.uint8).reshape(5, _N)

        got = np.asarray(
            wots.chain(tweak, _PK_SEED, values, starts, steps, self._step_addresses(5))
        )

        for index in range(5):
            self.assertEqual(
                bytes(got[index]),
                bytes(
                    _spec_chain(
                        tweak,
                        values[index],
                        int(starts[index]),
                        int(steps[index]),
                        _POSITION,
                        index,
                    )
                ),
                f"chain {index}",
            )

    def test_zero_steps_leaves_the_value_alone(self) -> None:
        values = np.arange(2 * _N, dtype=np.uint8).reshape(2, _N)
        got = np.asarray(
            wots.chain(
                _family(),
                _PK_SEED,
                values,
                np.zeros(2, dtype=np.uint32),
                np.zeros(2, dtype=np.uint32),
                self._step_addresses(2),
            )
        )
        np.testing.assert_array_equal(got, values)


class SignVerifyTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tweak = _family()
        self.compress = wots.fips205_compression(self.tweak, _PK_SEED, [_POSITION])

    def _pk_gen(self) -> np.ndarray:
        return np.asarray(
            wots.pk_gen(
                self.tweak, _PARAMS, _PK_SEED, _SK_SEED, [_POSITION], self.compress
            )[0]
        )

    def _pk_from_sig(self, signature: np.ndarray, message: np.ndarray) -> np.ndarray:
        return np.asarray(
            wots.pk_from_sig(
                self.tweak,
                _PARAMS,
                signature[None, ...],
                message[None, :],
                _PK_SEED,
                [_POSITION],
                self.compress,
            )[0]
        )

    def test_a_signature_walks_the_rest_of_the_way_to_the_public_key(self) -> None:
        signature = np.asarray(
            wots.sign(self.tweak, _PARAMS, _MESSAGE, _PK_SEED, _SK_SEED, _POSITION)
        )
        self.assertEqual(signature.shape, (_PARAMS.len, _N))
        np.testing.assert_array_equal(
            self._pk_from_sig(signature, _MESSAGE), self._pk_gen()
        )

    def test_a_different_message_does_not_arrive_at_the_public_key(self) -> None:
        # Without this the round trip would also pass for a construction that
        # ignored the message entirely.
        signature = np.asarray(
            wots.sign(self.tweak, _PARAMS, _MESSAGE, _PK_SEED, _SK_SEED, _POSITION)
        )
        other = _MESSAGE.copy()
        other[0] ^= 1
        self.assertNotEqual(
            bytes(self._pk_from_sig(signature, other)), bytes(self._pk_gen())
        )

    def test_a_key_at_another_position_is_a_different_key(self) -> None:
        # The address is what separates one key pair from the next: the same
        # seeds at another position must not produce the same public key.
        elsewhere = wots.WotsPosition(layer=1, tree=9, key_pair=5)
        other = np.asarray(
            wots.pk_gen(
                self.tweak,
                _PARAMS,
                _PK_SEED,
                _SK_SEED,
                [elsewhere],
                wots.fips205_compression(self.tweak, _PK_SEED, [elsewhere]),
            )[0]
        )
        self.assertNotEqual(bytes(other), bytes(self._pk_gen()))

    def test_the_public_key_is_the_chain_ends_compressed(self) -> None:
        # pk_gen is sign with every digit at w − 1, then compressed — which is
        # also the statement that a signature on the all-maximal digits would be
        # the private key's chain ends.
        ends = wots.chain(
            self.tweak,
            _PK_SEED,
            wots.secret_values(self.tweak, _PARAMS, _PK_SEED, _SK_SEED, [_POSITION]),
            np.zeros(_PARAMS.len, dtype=np.uint32),
            np.full(_PARAMS.len, _PARAMS.w - 1, dtype=np.uint32),
            wots._chain_addresses(_PARAMS, [_POSITION]),
        )
        np.testing.assert_array_equal(
            np.asarray(self.compress(ends.reshape(1, _PARAMS.len * _N)))[0],
            self._pk_gen(),
        )


if __name__ == "__main__":
    absltest.main()
