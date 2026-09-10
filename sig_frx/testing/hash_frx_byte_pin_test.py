# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The bytes each hash-frx row this repo consumes produces, recorded row by row.

hash-frx is retiring its per-hash composite markers so every hash lowers through
the generic `static_while` path instead of a dedicated emitter. That changes how
a hash lowers, not what it computes — and every scheme here signs with the bytes
a hash returns, so a row that moved by one bit is a signature every verifier in
the world rejects. This file is what that claim is checked against: one pin per
consumed row, recorded ahead of the first retirement, asserted after each pin
bump.

**What a retirement moves is right here.** At the pinned hash-frx every row
below reports `FusionPath.DEDICATED` and `digest` emits a
`composite[name="hash_frx.digest.*"]` — the marker the retirements delete, after
which the same body has to lower through the generic `static_while` path
instead. The pins are what says the bytes did not come with it.

**Each row twice, eagerly and under `frx.jit`.** Both are how this repo calls:
a scheme hashing inside its own traced region reaches the second, and ECDSA's
host constructor and Ethereum's dispatcher reach the first. The marked region is
a whole program by itself in one and one region among many in the other, so a
retirement that landed correctly on a marker standing alone and not on a fused
one would be invisible to whichever gate exercised only the other.

**The pins are `hashlib`'s, not the row's own.** Capturing them by running the
row under test would pin a break as readily as the truth — the self-consistency
[`testing.md`](../../docs/reference/testing.md#known-answer-tests-are-the-gate)
rules out. Each was produced on the host by the call named beside it, so the
assertion below is one implementation against another.

Keccak-256 is the exception with no `hashlib` sibling, so its cases are the two
published vectors — go-ethereum's `Keccak256Hash` constant and EIP-155's signing
hash — that [`ethereum_test`](../classical/testing/ethereum_test.py) gates the
dispatcher on. It carries no multi-block case of its own: SHA3-256 and SHAKE256
absorb at the same rate through the same permutation, and both have one here.

**Routing is deliberately not asserted.** `fusion_path` says whether this
backend routes the row's marker to one kernel, which is a property of the pin and
the backend and is exactly what the retirements move. A test that pinned it would
fail on the switch this one exists to let through.

What is absent is the field permutation: leanSig consumes `Poseidon`, whose
output is field elements rather than bytes, and its own known-answer test pins
those on the same pull request.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from frx import Array
from frx.typing import ArrayLike
from hash_frx import (
    Hmac,
    Keccak256,
    Mgf1,
    Sha3_256,
    Sha3_512,
    Sha256,
    Sha512,
    Shake128,
    Shake256,
)

# Three messages of one length, since `digest` takes a batch of equal-length
# messages. Eight bytes fits inside every block size and rate a row here uses, so
# these exercise the degenerate trip count: one absorb, all of it padding but the
# message itself.
_ONE_BLOCK = (b"sig-frx\n", b"\x00" * 8, bytes(range(8)))

# Two hundred bytes clears the widest rate any row here absorbs at — SHAKE128's
# 168 — so every row's absorb loop runs more than once. The content is arbitrary
# and only has to differ per batch entry: a row that ignored its input would
# return one digest three times.
_SEVERAL_BLOCKS = tuple(
    bytes((seed * 37 + i) % 256 for i in range(200)) for seed in (1, 2, 3)
)

# MGF1 is a XOF over a seed rather than a hash of a message, and SLH-DSA seeds it
# with `R ‖ PK.seed ‖ PK.root`. Thirty-two bytes is one digest wide, which is the
# unit its counter blocks are cut in.
_MGF1_SEEDS = (bytes(range(32)), bytes(range(32, 64)), b"\xff" * 32)

# HMAC's key, shaped `[1, K]`: one key broadcast across the batch, which is the
# shape SLH-DSA's `PRF_msg` hands it — the key is the secret, not the message.
_HMAC_KEY = np.arange(32, dtype=np.uint8)[None, :]

# go-ethereum crypto/crypto_test.go's `TestKeccak256Hash` input, and EIP-155's
# worked-example signing data. Both are single-entry batches because each
# published vector fixes its own message length.
_KECCAK_ABC = (b"abc",)
_KECCAK_EIP155 = (
    bytes.fromhex(
        "ec098504a817c800825208943535353535353535353535353535353535353535"
        "880de0b6b3a764000080018080"
    ),
)
_KECCAK256_ABC = ("4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45",)
_KECCAK256_EIP155 = (
    "daf5a779ae972f972197303d7b574746c7ef83eadac0f2791ad23db92e4c8e53",
)

# `hashlib.sha256(m).digest()`. SHA-256 is ECDSA's message hash, SLH-DSA's
# SHA-2 tweakable hash, XMSS's core hash, and `prehash.sha2_256` — the most
# consumed row in the repo, and the one HMAC and MGF1 below are built over.
_SHA2_256_ONE_BLOCK = (
    "cf9d2ddc0b7660eaadd18a61711acddd217ce978e3f5c0e094141aa9759bb69f",
    "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc",
    "8a851ff82ee7048ad09ec3847f1ddf44944104d2cbd17ef4e3db22c6785a0d45",
)
_SHA2_256_SEVERAL_BLOCKS = (
    "ef54a586f98e527985a71a14c2dfbcb322fc420fa85dc60267b9dde7267e88cd",
    "f803308372fb2c79f3962741faa846720be609c6825570ab81f00f8fca4c8777",
    "23d0bf7df250aaaba7b38b21d22c3005b07ad0dfa80f4342cd1736fb2e52a76f",
)

# `hashlib.sha512(m).digest()`. SLH-DSA's wide hash above security category 1.
_SHA2_512_ONE_BLOCK = (
    "ff49feb1def90d2250fd33d160f596444290f8db004c18b1470e25cd39432983"
    "8734f74f6192fadc4881fb3e590456b4576538cce6b42b525b0814a82d9fc57e",
    "1b7409ccf0d5a34d3a77eaabfa9fe27427655be9297127ee9522aa1bf4046d4f"
    "945983678169cb1a7348edcac47ef0d9e2c924130e5bcc5f0d94937852c42f1b",
    "8a414c5860cf1be7bc8531442f69a65ef2ecf0b7cad9994bcb407097eb74ccb9"
    "2e93aabd24bde60331123b4d900684ca7be6027099d4946bf537f4d6c6df3d82",
)
_SHA2_512_SEVERAL_BLOCKS = (
    "3b7dc603ec76d6e0649ba743aab66e3a5d94f50937528e965e5e00384afaee74"
    "0e285ac3605ff11c59f22c041f7be7f76150be1b0fd0a287a06e4bb61e8bc2d1",
    "8898afc6110f86398d6b6cc7b9214fee9b933c4f191bdc136350a577a6bbcf53"
    "ff4c42be88a789766dc29af4b1028315f2499e5818bb6cb09a20f9bf4042c05c",
    "660c85f37c5cfb86f7dfb72e9f7522dbae511e95959bdbd0cf2ddea6f025a2ec"
    "00dfa864b4f47e29fc5f602b61558df746b2678557e13f5a538cc0a340447781",
)

# `hashlib.sha3_256(m).digest()` — `prehash.sha3_256`.
_SHA3_256_ONE_BLOCK = (
    "e8a65d4319aa7b167eec8e726f22bf9718cbddcd93b726a475cf676c45cf13eb",
    "48dda5bbe9171a6656206ec56c595c5834b6cf38c5fe71bcb44fe43833aee9df",
    "eb4d0f2add0f6d0b26f0c65dbe71fe617cc6b43fb403649e82cc8bab41195f4e",
)
_SHA3_256_SEVERAL_BLOCKS = (
    "de10f666853336491c329caa378fd2b75375ff6b55bc46ff6db3a7bfbfe76f0a",
    "000dc0e56ad0bce7170563b23630f86e015e59e1abb623fd51836eb00052b9a1",
    "c2ab338d348e73fc7b8892750dfdf53ce11d55ef19f739364a4be97c734a544d",
)

# `hashlib.sha3_512(m).digest()` — `prehash.sha3_512`, and the narrowest rate
# here at 72 bytes.
_SHA3_512_ONE_BLOCK = (
    "10660f3c1a89042bae6574c4eac6d1ca2cda263614539edce5aa5dd1833c1526"
    "581103eb21b39cf8ba199c0c56b9f286d29b719dc48d300617b619dadd6a32f8",
    "0ade1db9cc8552ed5997a5642d835ebd191367d08c24564a735a16f777ec7a0f"
    "02e7575e5c778e39d6cdfa79006cd96bc4b40967abbc23b9109eed2f296af8f6",
    "44d125f785e8edf22739fe0ecaf0902969131b0a66b93091119b8f3ba16bad11"
    "8ff2df4cd47d2639ffd180d5e6491cf957e6d346d6c7d914b810e4560c7e662c",
)
_SHA3_512_SEVERAL_BLOCKS = (
    "e3a0d26b4881091c0bfa9a82137839eefe3b08924fca83123e2b79a65d7abbd2"
    "19d85edcb24006985b73ca672fa84d5dcea7fa26f86011cecf541e00521f19fa",
    "34f3f5f8570b8cf064ab4f5d52f2242b2a20e4c99b9b9b189bb1ef79857a8109"
    "a3467b335f04cb52353d561926ba149973e4d5c33df06d9900802c8b9dab1fe3",
    "a26e5924e0cf01ad5fd4215308e3ea2e00a8c4683a3817bcae0dec06d6155d9c"
    "ffdd310ab391d064d04265a6a23fdba4239407eb606895ae59060a686dcc3582",
)

# `hashlib.shake_128(m).digest(32)` — `prehash.shake128`'s length, which FIPS
# 204 Algorithm 4 line 19 fixes. The family is ML-DSA's `G`.
_SHAKE128_32_ONE_BLOCK = (
    "deb6b76eb786bca81d16c029f0cb6abbb1c7f35aaf2430edf6de09e5b2c1f95b",
    "7a24b666da345c98c3a400aafd14a51a6c07d748c6fc04fbd13088ed8b33948d",
    "2ea7a0601b9c5cb41e469f854077c1e33aac94d39ce46163429abcacfa992bbc",
)
_SHAKE128_32_SEVERAL_BLOCKS = (
    "aabf93fd8cdfcd9ca01cd9575d67ac90b8eba0736364ad25a15746ab005e04c5",
    "413330def9cf6a06a3fda9eb2d00c8cd3c17c4d8785e0a12d31707ee72c54d7b",
    "5336e55a8d1ced50a4490efe0acab30f3edcb199c5a72eca7ac8b207d62bff00",
)

# `hashlib.shake_128(m).digest(200)`. Two hundred bytes clears SHAKE128's
# 168-byte rate, so this is the squeeze loop permuting between blocks rather
# than reading one state out — which is how `ExpandA` draws a matrix entry.
_SHAKE128_200_ONE_BLOCK = (
    "deb6b76eb786bca81d16c029f0cb6abbb1c7f35aaf2430edf6de09e5b2c1f95b"
    "b0d6becec918f341a1e2f1fdba1930ffa6bb6e4ed413617082daaf2240b2519c"
    "98910a65294c0d3f68da6c7ed8ddac70fdbb3c2748078c5f5231d6c457462ab2"
    "e743413c92f79e7519267c0028db2c15cbabc23cc4155b3ed4d17b583a546827"
    "993e0a9caf416fea0338606c33bdfdd2dbf9295cd2657acf81d32c7412920f50"
    "71d276f37a47681aa45bc9a99b981f6385a9a85ade91291f09c45f1950af5251"
    "294a1c2192feca36",
    "7a24b666da345c98c3a400aafd14a51a6c07d748c6fc04fbd13088ed8b33948d"
    "c4301af2a06c213a0f9e5d232f107c1c89b37e0d59ba5926744c7a6f044d33e9"
    "7553980c53950e8b6cd0309dc799a67e84dd08be1ef665e97fa42559d72d0409"
    "60ddf11c540283e43450606b2bc743957e96255c9792344b7b3f9f4ad41465ed"
    "44b4bccd0e4d694ce85a23b33f9fb5e66cb920a4ae4ecaee3327637df42c969a"
    "fce2d787cc16730e3d361f99e20ae768b3005be4db9987e3a3089b738ff07730"
    "3d527611548b4027",
    "2ea7a0601b9c5cb41e469f854077c1e33aac94d39ce46163429abcacfa992bbc"
    "424ab689bb0123e0dc3e0d84376a56616ca0d2f7832f6a7facddfbc0d9d6175a"
    "e7ac86172d43f576e6af5baa80db29fc079cb737aae6cf82ea69ac6d005e072d"
    "41ff84ca47772dff4a08c2e29a5d69a564728bf5669ed8146a9fdc4edb5e58c9"
    "62140d591fa6bf520bf088926b919f7971616fc4aa4b746521accec8e07b9a36"
    "ba71817618a68ce138579b8c5038e40fbd1ea6e1ee4ead68c3026069cbd8e2ef"
    "3e0421ddc6316986",
)

# `hashlib.shake_256(m).digest(64)` — `prehash.shake256`'s length. The family
# is ML-DSA's `H`, SLH-DSA's SHAKE tweakable hash, and Falcon's.
_SHAKE256_64_ONE_BLOCK = (
    "de552cd3021c888830317edc3384d1b05ee2aab3eacbaa9fe0c8fbfd309da056"
    "8dbdf6aee95759b82b4a3f992da48a675c74f9a1fbe4b242df03191c6812e4b8",
    "119141dce89807096095d9729b0da80481a492498e235346efc58aa73335a351"
    "aa65e1dee4fb53f8c1b22eaf528f75c5fe7e87cbf3e1e407f62888f515ce2f75",
    "b44dae93f2360c4914fa999f3aed53901d47a1b614109e4bfbe8a595d5dd5058"
    "142c63b0b548e296c37f8972677b7faf0c22bf85bf9c891c1b91f69184d8b6de",
)
_SHAKE256_64_SEVERAL_BLOCKS = (
    "8002232be2492623b32969fb4b3857ca48d380fd098d3780263f65cc42b8072e"
    "cbba6be05635dd796d73e6119b2a68f65fe526b468fa58477d941769b0c7b9c0",
    "fed69dfbf3238f07b16bd8ef111b2ef045712d03848f2f68d3e65d8dd2a6102c"
    "a50ee873995dba14b150f3264ef40ea262292d9e7b921ef339e9eeca957ce6f4",
    "eb87f9ac3caa47490e3fa1be459464e04ac2a1ae7d31e05423c02020bfd2afd5"
    "d891defde02122d9c8e6d978875af549065d1f47ffa8882d2bd237da786738a8",
)

# `hashlib.shake_256(m).digest(200)` — the squeeze loop again, over the
# 136-byte rate Keccak-256 and SHA3-256 share.
_SHAKE256_200_ONE_BLOCK = (
    "de552cd3021c888830317edc3384d1b05ee2aab3eacbaa9fe0c8fbfd309da056"
    "8dbdf6aee95759b82b4a3f992da48a675c74f9a1fbe4b242df03191c6812e4b8"
    "c9fd7275bad9e44e7ccf0b586d2230750adba616111ba56fc1a2d1067c810502"
    "7ad11b013c3e590d00941c856150d61c9f5746aa5502c474f5494900d8667881"
    "0ed9f59c00ebe6b0c90114c15615281f788a32435e8b16108cd0d398ac0f4aaf"
    "2c3460b89ce53f64bd4b213781ccf86f15cc66b7055c0c48fcd8eb80ab6580fa"
    "f9d25ecb0e8895a3",
    "119141dce89807096095d9729b0da80481a492498e235346efc58aa73335a351"
    "aa65e1dee4fb53f8c1b22eaf528f75c5fe7e87cbf3e1e407f62888f515ce2f75"
    "5cfc40d4d6bc545bbabb343f13bd81f4bbb3a9be13bc5bd387d6e672a6e74a41"
    "9d209e261c7424b5e9ed8a6b2a38199a36513ad97673ba898db0cb8e05208a8f"
    "b0dfc3e899d5fe05017afd3f0d66ca85cf1a9da663914a349bcd356d85d75a0d"
    "07b588b4f6baf410b0d959a8e5bd2f7b4091d36a28be2f48f64ad3b804c106a6"
    "f3986079f06b5fed",
    "b44dae93f2360c4914fa999f3aed53901d47a1b614109e4bfbe8a595d5dd5058"
    "142c63b0b548e296c37f8972677b7faf0c22bf85bf9c891c1b91f69184d8b6de"
    "9a32b15e289912ca731a125c15fa69403cb31158f7d6159ae3e20778b59a6f43"
    "ee3e27347d5ec238b7c3a29812406487962c239a40bcf9c41eb285caa4cd270e"
    "f7ccab334605216d900d5a30e86e2f30741beea7f354c60582eae55f9fe53585"
    "67f1e236e15853dca6b95b2a7c31245876b6353faa212dba5ed1338327a95aa3"
    "8297bfc278374971",
)

# `hmac.new(key, m, hashlib.sha256).digest()` — SLH-DSA's `PRF_msg`, which
# `hash/tweakable.py` reaches through `Hmac`.
_HMAC_SHA2_256_ONE_BLOCK = (
    "43bfb4ed2ef80abe69449f938163b7b4fa2ff189655dcde2b6d4211ca58c45f3",
    "9f0cd9b94097fe4929918d2b8942b34439574261a35dc50163f06c67d4e48899",
    "97c6323b0c0ee26c6a637ad3f16be38dd3cff2dbcef3fdf0a61d01110cdc3b17",
)
_HMAC_SHA2_256_SEVERAL_BLOCKS = (
    "40f222ee1a445b5af6dac8c6abde11690050785ef5a5baaf8b0e1932ad022841",
    "fe72fa719a8766838fc5f3ab0834821df9569d819f10bf2054b7215aed851a6d",
    "787876212f3d793a3f7f680d0023d7c80f1d58604698b5dcf050d645054f326c",
)

# RFC 8017 B.2.1 over SHA-256: `T ‖= H(seed ‖ I2OSP(counter, 4))` until 64
# bytes, truncated there. SLH-DSA's `H_msg`, which `hash/tweakable.py` reaches
# through `Mgf1`. Two counter blocks, so the concatenation is exercised.
_MGF1_SHA2_256_64 = (
    "70f4003d52b6eb03da852e93256b5986b5d4883098bb7973bc5318cc66637a84"
    "04a6950a06d3e3308ad7d3606ef810eb124e3943404ca746a12c51c7bf776839",
    "efaef20e6b8940753e828d4bf7d093ce4017867f3f95ed88eaa25d73f8dcc811"
    "9600a8046a9830b2ae84c55dc49b53bfb07e4416ff17d132a085492306843cb3",
    "18e9e8dc48ff23fe7c40ee7a8a37980379f0c32a21f2c20d6f8694a1d49e2d89"
    "30553e7813efd2b87333944e1eede0cbca9e8f6f1fcb2660169b1a03c7ba798e",
)


@dataclass(frozen=True)
class _Row:
    """One consumed hash-frx face, the batch it is pinned over, and the pins.

    `call` is the face rather than the row, because the three constructions here
    do not share one: a `ByteHash` hashes a message, `Hmac` takes a key beside
    it, and `Mgf1` takes a seed. What they do share is the shape — a batch in, a
    batch of digests out — which is all the assertion needs.
    """

    name: str
    call: Callable[[ArrayLike], Array]
    messages: tuple[bytes, ...]
    pins: tuple[str, ...]


_ROWS = (
    _Row("sha2_256", Sha256().digest, _ONE_BLOCK, _SHA2_256_ONE_BLOCK),
    _Row(
        "sha2_256_absorbing", Sha256().digest, _SEVERAL_BLOCKS, _SHA2_256_SEVERAL_BLOCKS
    ),
    _Row("sha2_512", Sha512().digest, _ONE_BLOCK, _SHA2_512_ONE_BLOCK),
    _Row(
        "sha2_512_absorbing", Sha512().digest, _SEVERAL_BLOCKS, _SHA2_512_SEVERAL_BLOCKS
    ),
    _Row("sha3_256", Sha3_256().digest, _ONE_BLOCK, _SHA3_256_ONE_BLOCK),
    _Row(
        "sha3_256_absorbing",
        Sha3_256().digest,
        _SEVERAL_BLOCKS,
        _SHA3_256_SEVERAL_BLOCKS,
    ),
    _Row("sha3_512", Sha3_512().digest, _ONE_BLOCK, _SHA3_512_ONE_BLOCK),
    _Row(
        "sha3_512_absorbing",
        Sha3_512().digest,
        _SEVERAL_BLOCKS,
        _SHA3_512_SEVERAL_BLOCKS,
    ),
    _Row("shake128_32", Shake128(32).digest, _ONE_BLOCK, _SHAKE128_32_ONE_BLOCK),
    _Row(
        "shake128_32_absorbing",
        Shake128(32).digest,
        _SEVERAL_BLOCKS,
        _SHAKE128_32_SEVERAL_BLOCKS,
    ),
    _Row(
        "shake128_200_squeezing",
        Shake128(200).digest,
        _ONE_BLOCK,
        _SHAKE128_200_ONE_BLOCK,
    ),
    _Row("shake256_64", Shake256(64).digest, _ONE_BLOCK, _SHAKE256_64_ONE_BLOCK),
    _Row(
        "shake256_64_absorbing",
        Shake256(64).digest,
        _SEVERAL_BLOCKS,
        _SHAKE256_64_SEVERAL_BLOCKS,
    ),
    _Row(
        "shake256_200_squeezing",
        Shake256(200).digest,
        _ONE_BLOCK,
        _SHAKE256_200_ONE_BLOCK,
    ),
    _Row("keccak256_abc", Keccak256().digest, _KECCAK_ABC, _KECCAK256_ABC),
    _Row("keccak256_eip155", Keccak256().digest, _KECCAK_EIP155, _KECCAK256_EIP155),
    _Row(
        "hmac_sha2_256",
        lambda msg: Hmac(Sha256()).mac(_HMAC_KEY, msg),
        _ONE_BLOCK,
        _HMAC_SHA2_256_ONE_BLOCK,
    ),
    _Row(
        "hmac_sha2_256_absorbing",
        lambda msg: Hmac(Sha256()).mac(_HMAC_KEY, msg),
        _SEVERAL_BLOCKS,
        _HMAC_SHA2_256_SEVERAL_BLOCKS,
    ),
    _Row("mgf1_sha2_256_64", Mgf1(Sha256(), 64).digest, _MGF1_SEEDS, _MGF1_SHA2_256_64),
)


def _batch(messages: tuple[bytes, ...]) -> Array:
    """The batch a face takes: uint8 `[B, L]` over messages of one length."""
    joined = np.frombuffer(b"".join(messages), dtype=np.uint8)
    return fnp.asarray(joined.reshape(len(messages), len(messages[0])))


def _hex(digests: Array) -> tuple[str, ...]:
    return tuple(bytes(digest).hex() for digest in np.asarray(digests))


class BytePinTest(parameterized.TestCase):
    @parameterized.named_parameters((row.name, row) for row in _ROWS)
    def test_the_row_returns_the_pinned_bytes(self, row: _Row) -> None:
        self.assertEqual(_hex(row.call(_batch(row.messages))), row.pins)

    @parameterized.named_parameters((row.name, row) for row in _ROWS)
    def test_the_traced_row_returns_the_same_bytes(self, row: _Row) -> None:
        """The same pins under a trace, which is how a scheme reaches a hash.

        A scheme hashes inside its own traced region, so the marked call is one
        region among many there and a whole program by itself in the case above.
        A retirement that landed on one shape and not the other reaches a
        signature unless both are pinned.
        """
        self.assertEqual(_hex(frx.jit(row.call)(_batch(row.messages))), row.pins)


if __name__ == "__main__":
    absltest.main()
