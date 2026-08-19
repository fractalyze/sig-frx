# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The Ethereum variant against the EIP-155 example and go-ethereum's vectors.

EIP-155's worked example is the one published end-to-end vector: its signing
hash pins the Keccak wiring, its (v, r, s) pin RFC 6979 under the HMAC-SHA256
face (libsecp256k1's default nonce function) together with low-S signing and
the v encoding, and recovering its sender pins the address path. go-ethereum's
crypto test constants gate address derivation against an independent
implementation. What stays local is what only this variant defines: the v
codec bounds, EIP-2 high-S rejection (the core accepts both halves; the chain
must not), chain-id mismatch rejection, and the EIP-191 personal-message
framing.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

from sig_frx import hashes
from sig_frx.classical import weierstrass
from sig_frx.classical.ecdsa import core, ethereum

# EIP-155's worked example: chain id 1, nonce 9, 20 gwei, 21000 gas, one ether
# to 0x3535...35, no data. The signing data is its RLP; the hash, the key, and
# (v, r, s) are as the EIP publishes them.
_SIGNING_DATA = bytes.fromhex(
    "ec098504a817c800825208943535353535353535353535353535353535353535"
    "880de0b6b3a764000080018080"
)
_SIGNING_HASH = bytes.fromhex(
    "daf5a779ae972f972197303d7b574746c7ef83eadac0f2791ad23db92e4c8e53"
)
_SECRET = bytes.fromhex(
    "4646464646464646464646464646464646464646464646464646464646464646"
)
_V = 37
_R = 18515461264373351373200002665853028612451056578545711640558177340181847433846
_S = 46948507304638947509940763649030358759909902576025900602547168820602576006531

# go-ethereum crypto/crypto_test.go: the key its sign/recover tests run with
# and the address they require it to derive to.
_GETH_SECRET = bytes.fromhex(
    "289c2857d4598e37fb9647507e47a309d6133539bf21a8b9cb6df88fd5232032"
)
_GETH_ADDRESS = bytes.fromhex("970e8128ab834e8eac17ab8e3812f010678cf791")


def _keccak(data: bytes) -> bytes:
    array = np.frombuffer(data, dtype=np.uint8)[None]  # the sponge takes [B, L]
    return np.asarray(hashes.keccak256(array).digest(array))[0].tobytes()


def _digest() -> np.ndarray:
    return np.frombuffer(_SIGNING_HASH, dtype=np.uint8)


def _address_of(secret: bytes) -> bytes:
    scheme = core.Ecdsa(weierstrass.SECP256K1, core.SHA256)
    public, _ = scheme.keygen(np.frombuffer(secret, dtype=np.uint8))
    return np.asarray(ethereum.address_from_key(public)).tobytes()


class KeccakWiringTest(absltest.TestCase):
    def test_matches_the_published_digests(self) -> None:
        # go-ethereum's Keccak256Hash vector, then EIP-155's signing hash —
        # the sponge itself is gated in hash-frx; this pins the dispatcher
        # naming the 0x01-domain row rather than SHA3-256.
        self.assertEqual(
            _keccak(b"abc").hex(),
            "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45",
        )
        self.assertEqual(_keccak(_SIGNING_DATA), _SIGNING_HASH)


class VCodecTest(absltest.TestCase):
    def test_round_trips_legacy_and_chain_ids(self) -> None:
        for chain_id in (None, 1, 137, 11155111):
            for parity in (0, 1):
                v = ethereum.v_encode(parity, chain_id)
                self.assertEqual(ethereum.v_decode(v), (parity, chain_id))

    def test_rejects_values_between_the_forms(self) -> None:
        # {27, 28} and >= 35 are the two published shapes; everything under
        # and between them is no encoding at all.
        for bad in (0, 1, 26, 29, 34):
            with self.assertRaises(ValueError):
                ethereum.v_decode(bad)

    def test_rejects_the_unencodable_recovery_ids(self) -> None:
        # The x >= n ids exist at the curve level, but v has one parity bit;
        # a signature that draws one cannot ride Ethereum's wire.
        for recovery_id in (2, 3):
            with self.assertRaises(ValueError):
                ethereum.v_encode(recovery_id, 1)


class SignTest(absltest.TestCase):
    def test_reproduces_the_eip155_example(self) -> None:
        signature, v = ethereum.sign(
            np.frombuffer(_SECRET, dtype=np.uint8), _digest(), chain_id=1
        )
        self.assertEqual(v, _V)
        self.assertEqual(
            signature.tobytes(), _R.to_bytes(32, "big") + _S.to_bytes(32, "big")
        )

    def test_legacy_v_names_the_same_signature(self) -> None:
        # The chain id changes only how the parity rides v, never (r, s).
        signature, v = ethereum.sign(np.frombuffer(_SECRET, dtype=np.uint8), _digest())
        self.assertEqual(v, 27 + (_V - 35) % 2)
        self.assertEqual(
            signature.tobytes(), _R.to_bytes(32, "big") + _S.to_bytes(32, "big")
        )


class RecoverAddressTest(absltest.TestCase):
    def test_recovers_the_example_sender(self) -> None:
        signature = _R.to_bytes(32, "big") + _S.to_bytes(32, "big")
        addresses, ok = ethereum.recover_address(
            _digest()[None],
            np.frombuffer(signature, dtype=np.uint8)[None],
            np.array([_V]),
            chain_id=1,
        )
        self.assertEqual(list(ok), [True])
        self.assertEqual(addresses[0].tobytes(), _address_of(_SECRET))

    def test_policy_rejections_in_one_batch(self) -> None:
        n = weierstrass.SECP256K1.n
        good = _R.to_bytes(32, "big") + _S.to_bytes(32, "big")
        # The core would accept the malleated half and recover the same key
        # (the parity bit flips with s); EIP-2 is exactly the rule that the
        # chain does not.
        high_s = _R.to_bytes(32, "big") + (n - _S).to_bytes(32, "big")
        signatures = np.stack(
            [np.frombuffer(s, dtype=np.uint8) for s in (good, good, high_s, good, good)]
        )
        digests = np.broadcast_to(_digest(), (5, 32)).copy()
        vs = np.array(
            [
                _V,  # chain 1, as the verifier expects
                27,  # pre-EIP-155 legacy, still consensus-valid
                _V ^ 1,  # the malleated half's parity
                2 * 137 + 35,  # another chain's replay
                29,  # no encoding at all
            ]
        )
        addresses, ok = ethereum.recover_address(digests, signatures, vs, chain_id=1)
        self.assertEqual(list(ok), [True, True, False, False, False])
        sender = _address_of(_SECRET)
        self.assertEqual(addresses[0].tobytes(), sender)
        self.assertEqual(addresses[1].tobytes(), sender)
        self.assertFalse(addresses[2:].any())

    def test_addresses_match_go_ethereum(self) -> None:
        self.assertEqual(_address_of(_GETH_SECRET), _GETH_ADDRESS)
        # go-ethereum's TestSign, replayed: sign Keccak256("foo"), recover,
        # and land on the same address constant.
        digest = np.frombuffer(_keccak(b"foo"), dtype=np.uint8)
        signature, v = ethereum.sign(
            np.frombuffer(_GETH_SECRET, dtype=np.uint8), digest
        )
        addresses, ok = ethereum.recover_address(
            digest[None], signature[None], np.array([v])
        )
        self.assertEqual(list(ok), [True])
        self.assertEqual(addresses[0].tobytes(), _GETH_ADDRESS)


class PersonalMessageTest(absltest.TestCase):
    def test_frames_per_eip191(self) -> None:
        # The Keccak row is gated above; what this pins is the version-0x45
        # framing — the literal prefix and the decimal length.
        message = b"hello world"
        want = _keccak(b"\x19Ethereum Signed Message:\n11" + message)
        got = ethereum.personal_message_digest(np.frombuffer(message, dtype=np.uint8))
        self.assertEqual(np.asarray(got).tobytes(), want)

    def test_round_trips_via_recovery(self) -> None:
        digest = ethereum.personal_message_digest(
            np.frombuffer(b"attested", dtype=np.uint8)
        )
        signature, v = ethereum.sign(
            np.frombuffer(_GETH_SECRET, dtype=np.uint8), digest
        )
        addresses, ok = ethereum.recover_address(
            np.asarray(digest)[None], signature[None], np.array([v])
        )
        self.assertEqual(list(ok), [True])
        self.assertEqual(addresses[0].tobytes(), _GETH_ADDRESS)


if __name__ == "__main__":
    absltest.main()
