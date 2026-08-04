# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Every component tweaks with its family's address encoding, not with one it picked.

FIPS 205 gives the address two encodings — §11.2's 22-byte `ADRS^c` for the SHA-2
parameter sets and §4.2's full 32 bytes for the SHAKE ones — and which one applies
is a property of the tweakable hash family, published as
`TweakableHash.compressed_address`. Every component that builds an address has to
ask, because a component that decides for itself is right for exactly one family.

That is a claim no other test in this package makes. The known-answer tests catch a
wrong encoding, but only where they run: SLH-DSA key generation reaches WOTS+ and
the XMSS tree and never FORS, and the merge gate's signing cases are one SHA-2 set,
so the FORS builders and the verifier's batch builder are covered for SHAKE only by
the exhaustive sweep — which is tagged `slow_kat` and does not run on a pull
request. The cases here are the fast guard on the same property.

They assert against a family that records the addresses it is handed rather than
hashing them, which is what lets one small case cover a component at both
encodings. What is checked is the width of every address a component built: a
component that hardcoded one encoding reports 22 where 32 was asked for, and the
set equality names every address it built rather than the first.
"""

from __future__ import annotations

from collections.abc import Callable

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import Array
from frx.typing import ArrayLike

from sig_frx.hashbased import adrs, fors, tree, wots, xmss

_N = 16
_WOTS = wots.WotsParams(n=_N)
_FORS = fors.ForsParams(n=_N, a=3, k=4)
_HEIGHT = 2
_PK_SEED = np.frombuffer(bytes(range(_N)), dtype=np.uint8)
_SK_SEED = np.frombuffer(bytes(range(100, 100 + _N)), dtype=np.uint8)


class _Recorder:
    """A `TweakableHash` that records the addresses it is tweaked with.

    Every function returns zeros of the length the callers expect. Nothing here
    depends on the digest — the subject is which address a component built, and a
    real hash would only make the cases slower and their failures harder to read.
    """

    def __init__(self, *, compressed: bool) -> None:
        self.n = _N
        self.m = 30
        self.compressed_address = compressed
        self.widths: set[int] = set()

    def _record(self, address: ArrayLike) -> Array:
        rows = np.asarray(address)
        self.widths.add(int(rows.shape[-1]))
        return fnp.asarray(np.zeros((rows.shape[0], self.n), dtype=np.uint8))

    def prf(self, pk_seed: ArrayLike, sk_seed: ArrayLike, adrs: ArrayLike) -> Array:
        return self._record(adrs)

    def f(self, pk_seed: ArrayLike, adrs: ArrayLike, m1: ArrayLike) -> Array:
        return self._record(adrs)

    def h(self, pk_seed: ArrayLike, adrs: ArrayLike, m2: ArrayLike) -> Array:
        return self._record(adrs)

    def t(self, pk_seed: ArrayLike, adrs: ArrayLike, messages: ArrayLike) -> Array:
        return self._record(adrs)

    def prf_msg(
        self, sk_prf: ArrayLike, opt_rand: ArrayLike, message: ArrayLike
    ) -> Array:
        return fnp.asarray(np.zeros(self.n, dtype=np.uint8))

    def h_msg(
        self,
        randomizer: ArrayLike,
        pk_seed: ArrayLike,
        pk_root: ArrayLike,
        message: ArrayLike,
    ) -> Array:
        return fnp.asarray(np.zeros((1, self.m), dtype=np.uint8))


class AddressEncodingTest(absltest.TestCase):
    """Each operation, at both encodings, over every address it builds."""

    def _assert_encoding(self, run: Callable[[_Recorder], object]) -> None:
        for compressed in (True, False):
            with self.subTest(compressed=compressed):
                tweak = _Recorder(compressed=compressed)
                run(tweak)
                expected = adrs.COMPRESSED_ADRS_SIZE if compressed else adrs.ADRS_SIZE
                self.assertEqual(tweak.widths, {expected})

    # -- WOTS+ -------------------------------------------------------------

    def test_the_wots_secret_values_ask_their_family(self) -> None:
        position = wots.WotsPosition(layer=1, tree=2, key_pair=np.arange(2))
        self._assert_encoding(
            lambda tweak: wots.secret_values(tweak, _WOTS, _PK_SEED, _SK_SEED, position)
        )

    def test_the_wots_chains_and_their_compression_ask_their_family(self) -> None:
        position = wots.WotsPosition(layer=1, tree=2, key_pair=np.arange(2))
        self._assert_encoding(
            lambda tweak: wots.pk_gen(
                tweak,
                _WOTS,
                _PK_SEED,
                _SK_SEED,
                position,
                wots.fips205_compression(tweak, _PK_SEED, position),
            )
        )

    def test_the_wots_verifier_asks_its_family(self) -> None:
        position = wots.WotsPosition(layer=1, tree=2, key_pair=np.arange(2))
        signatures = np.zeros((2, _WOTS.len, _N), dtype=np.uint8)
        messages = np.zeros((2, _N), dtype=np.uint8)
        self._assert_encoding(
            lambda tweak: wots.pk_from_sig(
                tweak,
                _WOTS,
                signatures,
                messages,
                _PK_SEED,
                position,
                wots.fips205_compression(tweak, _PK_SEED, position),
            )
        )

    # -- the XMSS layer ----------------------------------------------------

    def test_the_xmss_root_asks_its_family(self) -> None:
        position = tree.TreePosition(layer=1, tree=2)
        self._assert_encoding(
            lambda tweak: xmss.root(tweak, _WOTS, _PK_SEED, _SK_SEED, position, _HEIGHT)
        )

    def test_the_xmss_verifier_asks_its_family(self) -> None:
        # `xmss.node_addresses`, the builder for a batch whose entries each sit in
        # their own tree — the one the verify path uses and key generation never
        # reaches.
        position = tree.TreePosition(layer=1, tree=np.arange(2))
        signatures = np.zeros((2, _WOTS.len + _HEIGHT, _N), dtype=np.uint8)
        messages = np.zeros((2, _N), dtype=np.uint8)
        self._assert_encoding(
            lambda tweak: xmss.pk_from_sig(
                tweak,
                _WOTS,
                signatures,
                messages,
                _PK_SEED,
                position,
                np.zeros(2, dtype=np.uint32),
            )
        )

    # -- FORS --------------------------------------------------------------

    def test_the_fors_key_generation_asks_its_family(self) -> None:
        position = fors.ForsPosition(tree=6, key_pair=2)
        self._assert_encoding(
            lambda tweak: fors.pk_gen(tweak, _FORS, _PK_SEED, _SK_SEED, position)
        )

    def test_the_fors_signer_asks_its_family(self) -> None:
        position = fors.ForsPosition(tree=6, key_pair=2)
        digest = np.frombuffer(bytes([0b10110100, 0b01101110]), dtype=np.uint8)
        self._assert_encoding(
            lambda tweak: fors.sign(tweak, _FORS, digest, _PK_SEED, _SK_SEED, position)
        )

    def test_the_fors_verifier_asks_its_family(self) -> None:
        position = fors.ForsPosition(tree=np.arange(2), key_pair=2)
        signatures = np.zeros((2, _FORS.k, _FORS.a + 1, _N), dtype=np.uint8)
        digests = np.zeros((2, 2), dtype=np.uint8)
        self._assert_encoding(
            lambda tweak: fors.pk_from_sig(
                tweak, _FORS, signatures, digests, _PK_SEED, position
            )
        )


class RecorderTest(absltest.TestCase):
    def test_the_recorder_would_notice_a_hardcoded_encoding(self) -> None:
        # The cases above are only as good as the recorder's reach: if it observed
        # nothing, every one of them would pass on an empty set. This is the case
        # that says it observes, and that the two encodings are distinguishable.
        position = wots.WotsPosition(layer=1, tree=2, key_pair=0)
        widths = set()
        for compressed in (True, False):
            tweak = _Recorder(compressed=compressed)
            wots.secret_values(tweak, _WOTS, _PK_SEED, _SK_SEED, position)
            self.assertNotEmpty(tweak.widths)
            widths |= tweak.widths
        self.assertEqual(widths, {adrs.COMPRESSED_ADRS_SIZE, adrs.ADRS_SIZE})


if __name__ == "__main__":
    absltest.main()
