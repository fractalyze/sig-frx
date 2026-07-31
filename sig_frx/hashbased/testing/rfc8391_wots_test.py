# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""WOTS+ and its L-tree reproduce the reference implementation's digests.

This is the gate the substrate is finished against: RFC 8391 publishes no vectors,
so what stands in for them is the reference implementation §7 points at, digested
as `SHAKE128(artifact, 10)` the way its own `test/vectors.c` prints. A wrong
implementation does not collide with a 10-byte digest, and the digest rows cover a
public key, a signature and an L-tree leaf — so every part of the walk is pinned by
something published rather than by agreement with ourselves.

Two parameter sets, and running both is what proves the padding length is a
parameter: OID 13 pads to 4 bytes where OID 1 pads to 32, and nothing else about
them differs in a way an implementation can get right by accident.

The round trip is here too, but it is not the gate — `pk_from_sig` agreeing with
`pk_gen` is self-consistency, and a self-consistent wrong implementation round-trips
forever. It is worth pinning because it is what verification actually runs.
"""

from __future__ import annotations

import hashlib

import numpy as np
from absl.testing import absltest
from hash_frx.sha256 import Sha256

from sig_frx.hashbased import rfc8391_adrs as a
from sig_frx.hashbased import rfc8391_hashes, rfc8391_params, rfc8391_wots
from sig_frx.hashbased.testing import rfc8391_vectors


def _digest10(artifact: np.ndarray) -> bytes:
    """`SHAKE128(artifact, 10)` — the framing `test/vectors.c` prints."""
    return hashlib.shake_128(bytes(np.asarray(artifact).reshape(-1))).digest(10)


def _address(words: list[int]) -> a.Adrs:
    return a.Adrs(
        layer=words[0],
        tree=(words[1] << 32) | words[2],
        type=words[3],
        trailing=(words[4], words[5], words[6]),
        key_and_mask=words[7],
    )


class _Setup:
    """One parameter set with the reference's fixture already assembled."""

    def __init__(self, oid: int) -> None:
        self.vectors = rfc8391_vectors.REFERENCE[oid]
        self.params = rfc8391_params.XMSS_PARAMETER_SETS[oid]
        self.hashes = rfc8391_hashes.sha2_hashes(self.params, Sha256())
        self.wots = self.params.wots
        n = self.params.n
        self.sk_seed = rfc8391_vectors.fixture_bytes(n, step=1)
        self.pub_seed = rfc8391_vectors.fixture_bytes(n, step=2)
        self.message = rfc8391_vectors.fixture_bytes(n, step=3)
        self.addr = _address(rfc8391_vectors.ADDR_WORDS)
        self.addr2 = _address(rfc8391_vectors.ADDR2_WORDS)

    def public_key(self, position: a.Adrs) -> np.ndarray:
        """The `len` chain ends of one key pair, uncompressed — `wots_pkgen`'s `pk`."""
        ends = rfc8391_wots.pk_gen(
            self.hashes,
            self.wots,
            self.pub_seed,
            self.sk_seed,
            [position],
            lambda values: values,
        )
        return np.asarray(ends).reshape(self.wots.len, self.params.n)


def _setups() -> dict[int, _Setup]:
    return {oid: _Setup(oid) for oid in rfc8391_vectors.REFERENCE}


class ReferenceDigestTest(absltest.TestCase):
    """The three WOTS+ rows of the digest table, for both runnable OIDs."""

    def test_the_public_key_matches(self) -> None:
        for oid, setup in _setups().items():
            with self.subTest(oid=oid):
                self.assertEqual(
                    _digest10(setup.public_key(setup.addr)),
                    setup.vectors.digest_wots_pk,
                )

    def test_the_signature_matches(self) -> None:
        for oid, setup in _setups().items():
            with self.subTest(oid=oid):
                signature = rfc8391_wots.sign(
                    setup.hashes,
                    setup.wots,
                    setup.message,
                    setup.pub_seed,
                    setup.sk_seed,
                    setup.addr,
                )
                self.assertEqual(
                    _digest10(np.asarray(signature)),
                    setup.vectors.digest_wots_sig,
                )

    def test_the_ltree_leaf_matches(self) -> None:
        # `gen_leaf_wots(params, leaf, sk_seed, pub_seed, addr, addr2)` — the
        # parameters are `(ltree_addr, ots_addr)`, so the leaf compresses the public
        # key at `addr2` under an L-tree at `addr`. The other assignment produces a
        # self-consistent leaf that matches no published digest.
        for oid, setup in _setups().items():
            with self.subTest(oid=oid):
                leaf = _leaf(setup, ltree_position=setup.addr, ots_position=setup.addr2)
                self.assertEqual(bytes(leaf), setup.vectors.ltree_leaf)
                self.assertEqual(_digest10(leaf), setup.vectors.digest_ltree_leaf)


def _leaf(setup: _Setup, *, ltree_position: a.Adrs, ots_position: a.Adrs) -> np.ndarray:
    """`gen_leaf_wots` — a WOTS+ public key compressed through its L-tree."""
    leaves = rfc8391_wots.pk_gen(
        setup.hashes,
        setup.wots,
        setup.pub_seed,
        setup.sk_seed,
        [ots_position],
        rfc8391_wots.ltree_compression(
            setup.hashes, setup.pub_seed, [ltree_position], leaves=setup.wots.len
        ),
    )
    return np.asarray(leaves)[0]


class LtreeTest(absltest.TestCase):
    def test_the_two_addresses_are_not_interchangeable(self) -> None:
        # They tweak different hashes — the chains and the compression — so swapping
        # them is a different leaf, which is why the reference takes both.
        setup = _Setup(0x01)
        swapped = _leaf(setup, ltree_position=setup.addr2, ots_position=setup.addr)
        self.assertNotEqual(bytes(swapped), setup.vectors.ltree_leaf)

    def test_an_odd_level_lifts_its_last_node_unhashed(self) -> None:
        # §4.1.5's rule, and the reason `tree.reduce_levels` cannot be reused: it
        # refuses an odd count, while an L-tree over 67 leaves meets one at once.
        # Three leaves reduce to `H(H(l0, l1), l2)`, not to `H(H(l0, l1), H(l2, l2))`
        # and not to `H(H(l0, l1), H(l2, pad))`. No parameter set is this narrow —
        # the smallest `wots_len` is 51 — so the rule gets checked at a width small
        # enough to write the expectation out by hand.
        setup = _Setup(0x01)
        n = setup.params.n
        position = a.ltree(0, 0, 0)
        compress = rfc8391_wots.ltree_compression(
            setup.hashes, setup.pub_seed, [position], leaves=3
        )
        leaves = np.arange(3 * n, dtype=np.uint8).reshape(1, 3 * n)

        got = np.asarray(compress(leaves))[0]

        first = np.asarray(
            setup.hashes.h(
                setup.pub_seed,
                a.encode_batch(a.ltree(0, 0, 0, height=0, index=0)),
                leaves[0, : 2 * n],
            )
        )[0]
        lifted = leaves[0, 2 * n : 3 * n]
        expected = np.asarray(
            setup.hashes.h(
                setup.pub_seed,
                a.encode_batch(a.ltree(0, 0, 0, height=1, index=0)),
                np.concatenate([first, lifted]),
            )
        )[0]
        np.testing.assert_array_equal(got, expected)

    def test_many_key_pairs_compress_in_one_call(self) -> None:
        # An XMSS tree compresses all of its leaves at once, so the batched result
        # has to equal the one-at-a-time one.
        setup = _Setup(0x0D)
        width = setup.wots.len * setup.params.n
        positions = [a.ltree(0, 0, index) for index in range(4)]
        ends = np.arange(4 * width, dtype=np.uint8).reshape(4, width)

        together = np.asarray(
            rfc8391_wots.ltree_compression(
                setup.hashes, setup.pub_seed, positions, leaves=setup.wots.len
            )(ends)
        )

        for index, position in enumerate(positions):
            alone = np.asarray(
                rfc8391_wots.ltree_compression(
                    setup.hashes, setup.pub_seed, [position], leaves=setup.wots.len
                )(ends[index : index + 1])
            )
            np.testing.assert_array_equal(together[index], alone[0], f"leaf {index}")


class RoundTripTest(absltest.TestCase):
    def test_a_signature_recovers_its_public_key(self) -> None:
        for oid, setup in _setups().items():
            with self.subTest(oid=oid):
                signature = rfc8391_wots.sign(
                    setup.hashes,
                    setup.wots,
                    setup.message,
                    setup.pub_seed,
                    setup.sk_seed,
                    setup.addr,
                )
                recovered = rfc8391_wots.pk_from_sig(
                    setup.hashes,
                    setup.wots,
                    np.asarray(signature)[None, :, :],
                    setup.message,
                    setup.pub_seed,
                    [setup.addr],
                    lambda values: values,
                )
                np.testing.assert_array_equal(
                    np.asarray(recovered).reshape(setup.wots.len, setup.params.n),
                    setup.public_key(setup.addr),
                )

    def test_a_wrong_message_recovers_a_different_public_key(self) -> None:
        # The negative half: a verifier that returned the right key regardless would
        # pass the case above.
        setup = _Setup(0x01)
        signature = rfc8391_wots.sign(
            setup.hashes,
            setup.wots,
            setup.message,
            setup.pub_seed,
            setup.sk_seed,
            setup.addr,
        )
        tampered = np.asarray(setup.message).copy()
        tampered[0] ^= 1
        recovered = rfc8391_wots.pk_from_sig(
            setup.hashes,
            setup.wots,
            np.asarray(signature)[None, :, :],
            tampered,
            setup.pub_seed,
            [setup.addr],
            lambda values: values,
        )
        self.assertNotEqual(
            bytes(np.asarray(recovered).reshape(-1)),
            bytes(setup.public_key(setup.addr).reshape(-1)),
        )

    def test_verification_batches_over_key_pairs(self) -> None:
        # What an XMSS tree hands `pk_from_sig`: several signatures, each under its
        # own key pair, recovered in one call.
        setup = _Setup(0x0D)
        positions = [a.ltree(0, 0, index) for index in range(3)]
        signatures = np.stack(
            [
                np.asarray(
                    rfc8391_wots.sign(
                        setup.hashes,
                        setup.wots,
                        setup.message,
                        setup.pub_seed,
                        setup.sk_seed,
                        position,
                    )
                )
                for position in positions
            ]
        )
        messages = np.broadcast_to(
            setup.message, (len(positions), setup.params.n)
        ).copy()

        recovered = np.asarray(
            rfc8391_wots.pk_from_sig(
                setup.hashes,
                setup.wots,
                signatures,
                messages,
                setup.pub_seed,
                positions,
                lambda values: values,
            )
        )

        for index, position in enumerate(positions):
            np.testing.assert_array_equal(
                recovered[index].reshape(setup.wots.len, setup.params.n),
                setup.public_key(position),
                f"key pair {index}",
            )


if __name__ == "__main__":
    absltest.main()
