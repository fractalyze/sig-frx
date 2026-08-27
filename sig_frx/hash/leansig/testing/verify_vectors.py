# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""What leanSpec's own verifier answers — one key, five slots, three refusals.

The layer this gates is the whole scheme, so the vectors are whole objects: a
public key, a slot, a 32-byte root and a signature, with the verdict upstream
gave that tuple. Every other module here gates a primitive against a call; this
one gates the assembly against `GeneralizedXmssScheme.verify`.

`TEST` rather than `PROD`, and that is what the preset is for. Signing needs a
secret key, and at `PROD` one covers 2^32 slots — the archive publishes eight of
them precisely because nobody generates one to run a test. At `LOG_LIFETIME = 8`
a key pair is built in a moment, so these are real signatures over a real tree
rather than a transcription of what one would look like.

## Provenance

- **Upstream:** `leanEthereum/leanSpec` at commit
  `0588c2d215a955a516378677a92db2a5666802f3`, the same pin every other vector
  module here carries.
- **Fixture:** none. The archive has no `(public key, slot, message, signature)`
  family — its signature-shaped fixtures are leanMultisig aggregate proofs — so
  this is the reference implementation, third in the order
  [`testing.md`](../../../../docs/reference/testing.md) fixes.
- **The exact calls each value came from:**
  `TEST_SIGNATURE_SCHEME.key_gen(Slot(0), Uint64(1 << 8))` for the key pair,
  then `sign(secret_key, Slot(slot), message)` and
  `verify(public_key, Slot(slot), message, signature)` per case. The bytes are
  `PublicKey.encode_bytes()` and `Signature.encode_bytes()`.
- **Reproducing it:** the environment recipe is
  [`mode_vectors.py`](mode_vectors.py)'s — a Python 3.12 venv with
  `pydantic numpy numba`, and a stub `lean_multisig_py` installed before the
  first `lean_spec` import.

## Why these five slots

The Merkle path's left/right select is what a wrong index survives, so the slots
differ in parity at the levels a path walks through: 0 is left the whole way up,
31 is right the whole way, and 1, 7 and 16 alternate. A verifier that ignored the
index would still reproduce slot 0's root and no other's.

Signing past slot 31 is not a gap. Upstream keeps two bottom trees resident and
refuses a slot outside the prepared window — `[0, 32)` at this preset — so
reaching slot 255 means sliding the window, which is the signer's business.

## The three refusals, and why the last one is the important one

The first two are a valid signature offered for something it does not attest to:
re-labelled with another slot, and against another root. Each is what the seam's
per-entry `position` and `message` respectively decide.

`off_target_layer` is different in kind, and it is the only vector here that a
published set could never contain — upstream would not emit it, because it is
what upstream's signer loops until it avoids. **Its Merkle root rebuilds
correctly.** The randomness is one the signer rejected, its codeword's digits sum
to 16 rather than the target 6, and each chain is released at the digit that
codeword names — so the verifier's walk reaches the true chain ends, the leaf is
the true leaf, and the climb reaches the public key's own root. Confirmed against
upstream's `merkle.verify_path` directly, which returns `True` on it.

So every check in the verifier passes except the target sum, which is the point:
that filter is the whole of what makes a codeword unforgeable — two codewords
that dominate elementwise and sum alike are equal — and a verifier that computed
the message hash without it would accept this signature while reproducing every
other vector in this file. It is the one case that separates
`target_sum_encode` from `message_hash`.
"""

from __future__ import annotations

from typing import Final, NamedTuple


class VerifyVector(NamedTuple):
    """One `(public key, slot, message, signature)` and the verdict upstream gave.

    The key is shared across the set — `PUBLIC_KEY` below — because every case
    comes from the one key pair, which is what makes each negative case negative
    for the reason claimed rather than because the key does not match.
    """

    name: str
    slot: int
    message: str
    signature: str
    verdict: bool
    note: str = ""


PUBLIC_KEY: Final = (
    "694c1a134908284f15cea93451194045e078ea291c0cc4293285d9503d86195ddccf5b4ad23c7611d21a7e2ec3a8c043cd31ea6f"
)
"""The one `TEST` public key, SSZ-encoded: `root ‖ parameter`, 52 bytes."""


VERIFY_VECTORS: Final[tuple[VerifyVector, ...]] = (
    VerifyVector(
        name="slot_0",
        slot=0,
        message="000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
        signature=(
            "240000000f0279461232a5564af5887a8fe78f698c3f134b07bd583ce597532d28"
            "01000004000000d26cbd008f8e2b0971f98d7bfa56cc670dd9c3561f7919180792"
            "6756a7afb3422b23490094fc777b7719727e8369f2382e677d0baa9e932307d517"
            "6d29cf33791b530a1e8e83f94b5d66b65e1347146e26dd0e587a204f04f1a60565"
            "dd3617175151d163f7ea553e80629c461707a01bf8ee067b2e8c986e3667220aed"
            "5ac631b4102863e014ad1567b72075fdc16b6139b0f11b73bc2646cc693e62f483"
            "772de61eec5fd0951045f7628800b98cdf58c1e3181837ec20140cf76b52921e58"
            "2ed5a0301e68a655407edec13428f6eb3835ef2b7010ebc3733c4f2865fe785f5b"
            "9257d91141f9c024b84d1453bc6a8127d871814bee6b1761ffddf73f6182556534"
            "4cd03a1612395f9c0213312212492c71d25959f8634952bec26b6e66beee0f9e34"
            "ac0e1ef2da22a2fbc401e0424c215431304ae44fe322d237c06f17046614c27c57"
            "3cbda9b97156596c1685c2c90a0dbdff76b206f2399a70875fb81a2f68c4ab635c"
            "ae99893327e2e124315de23e5e1e414cc6cc763044f66c125e322d54"
        ),
        verdict=True,
    ),
    VerifyVector(
        name="slot_1",
        slot=1,
        message="0708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20212223242526",
        signature=(
            "24000000cb8e967d375b6d56352383264cf2d271735204745451bf585e58ef0128"
            "0100000400000009d9232ce49b5f54b71a5c446f49427a473700492447fa628d37"
            "af48f423d17d2b23490094fc777b7719727e8369f2382e677d0baa9e932307d517"
            "6d29cf33791b530a1e8e83f94b5d66b65e1347146e26dd0e587a204f04f1a60565"
            "dd3617175151d163f7ea553e80629c461707a01bf8ee067b2e8c986e3667220aed"
            "5ac631b4102863e014ad1567b72075fdc16b6139b0f11b73bc2646cc693e62f483"
            "772de61eec5fd0951045f7628800b98cdf58c1e3181837ec20140cf76b52921e58"
            "2ed5a0301e68a655407edec13428f6eb3835ef2b7010ebc3733c4f2865fe785f5b"
            "9257d91141f9c024b84d1453bc6a8127d871814bee6b1761ffddf73f61825565af"
            "f76e1b8abea00e7626d14528a7a92b902c1a1a6c692e41deab0141798e722019bd"
            "49192a7c3f47f349202191c8532cb3d27d7e982a8b4104a1c146a4902515dd9f83"
            "758f72e7793a59990f2cdcfa75982e7338bdb1f82e3300331d8d12f860f62a4707"
            "7b0c5f34117a7059f304ee048695114b9852cd2f5201cb7974510f01"
        ),
        verdict=True,
    ),
    VerifyVector(
        name="slot_7",
        slot=7,
        message="3132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f50",
        signature=(
            "24000000f59d4b52fbf5993282fdac655a66b1571b9a6c58b33c46574b8e965828"
            "010000040000005837bf115bad1a2cecfbe0660404fa43b6268d36b0da66535acc"
            "943cbe89a37e1b045a12520c3f501b9fd16c03827c40f265ea26c98be233b7d727"
            "25127fb6340f5841293d550752c0bdb756fc309307bec9e57e33874c5a45114a0f"
            "5808593d5151d163f7ea553e80629c461707a01bf8ee067b2e8c986e3667220aed"
            "5ac631b4102863e014ad1567b72075fdc16b6139b0f11b73bc2646cc693e62f483"
            "772de61eec5fd0951045f7628800b98cdf58c1e3181837ec20140cf76b52921e58"
            "2ed5a0301e68a655407edec13428f6eb3835ef2b7010ebc3733c4f2865fe785f5b"
            "9257d91141f9c024b84d1453bc6a8127d871814bee6b1761ffddf73f61825565ab"
            "b7c42bdd20146cef56555bb36f74474ea87c696a7f132f3c60a440ea74c97e0769"
            "154a5826b37a2641b11f53cd862e1de1853773578f2e9556c612129e3d40a3b4dc"
            "005a95bd6924443d300a1ba21120b489566ea64c20bac7e5326f01fe21fe5b1379"
            "721ac30ec1244e058e1caa0cad7fc91953e876085c2c5441e8c80f3e"
        ),
        verdict=True,
    ),
    VerifyVector(
        name="slot_16",
        slot=16,
        message="707172737475767778797a7b7c7d7e7f808182838485868788898a8b8c8d8e8f",
        signature=(
            "24000000a7861f27d16c1e5fb3374c5ae0e1b0733f544a765b9f111e28d3334c28"
            "0100000400000095ba5d0869af454e9531660e6e0ee30168549e44c4e1bc1023ba"
            "7c4c71fda23f085a23778310ff6b8cae7f1d0fc6155c9808dd5c2210ff28bf64e1"
            "3eb7f87244cd3a820da790321a3b06c206bd7efe3bf35128348b169d2965315642"
            "68c0b0721285de2ec7efb602223caf20fdb8d864ce4112630ad7f9603fbb2b0efd"
            "0a2b38578790194bd06e24a7a691073d20790013d15367d8d58151f4e46f32b91f"
            "1e7ee61eec5fd0951045f7628800b98cdf58c1e3181837ec20140cf76b52921e58"
            "2ed5a0301e68a655407edec13428f6eb3835ef2b7010ebc3733c4f2865fe785f5b"
            "9257d91141f9c024b84d1453bc6a8127d871814bee6b1761ffddf73f6182556518"
            "af497da392b05b09714151714fa34413aa4c3c5d58d27494b80e238f418b0b59b5"
            "0c0433b91d1244ca7546cec9df13881880071088bd53fa053f123fa67f1defc9d6"
            "4e65d48100577f5b5c3ae4750734d16e7a9bd5ea7ec954b924c0cb0f274674031c"
            "de2830389c409768eb9a4d78e8aaef0a2174e9242cf1ef4d9cbf6e76"
        ),
        verdict=True,
    ),
    VerifyVector(
        name="slot_31",
        slot=31,
        message="d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6f7f8",
        signature=(
            "24000000ad80dd41bc8bb9495c0a722c3f938368b6b6a60c733ab44414e2635128"
            "010000040000003b5b777d02664f034fabc457c9e3a73b18ce9d6b4b16be53b94d"
            "73656b10db046361352a141205503683e1407dd5b41648f65901265c8c2e5acaae"
            "1adcf48246c2e25d6979402a47ad7ae66be891d71165af37242f7d6c67fd3e4e0f"
            "e21d7f741698c752bcd70e50e70b732ea9703f451239e3682f99033d0c503f41da"
            "554f65578790194bd06e24a7a691073d20790013d15367d8d58151f4e46f32b91f"
            "1e7ee61eec5fd0951045f7628800b98cdf58c1e3181837ec20140cf76b52921e58"
            "2ed5a0301e68a655407edec13428f6eb3835ef2b7010ebc3733c4f2865fe785f5b"
            "9257d91141f9c024b84d1453bc6a8127d871814bee6b1761ffddf73f61825565db"
            "c036463da11226bc22a84ec107a70af9a7cb74588aff1593a9c764e4740075782f"
            "cb6a024296126e88583d15cfb778d1b91412da731928df7435160ba2c326825bb2"
            "639859b1028d3e6979ff967d06a8fa8a33857b6539a0b087453142d46aeb66ce29"
            "5c65d25d0fa17415ad6c832186e66b4905af1471e43e5465fd8a493d"
        ),
        verdict=True,
    ),
    VerifyVector(
        name="wrong_slot",
        slot=1,
        message="000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
        signature=(
            "240000000f0279461232a5564af5887a8fe78f698c3f134b07bd583ce597532d28"
            "01000004000000d26cbd008f8e2b0971f98d7bfa56cc670dd9c3561f7919180792"
            "6756a7afb3422b23490094fc777b7719727e8369f2382e677d0baa9e932307d517"
            "6d29cf33791b530a1e8e83f94b5d66b65e1347146e26dd0e587a204f04f1a60565"
            "dd3617175151d163f7ea553e80629c461707a01bf8ee067b2e8c986e3667220aed"
            "5ac631b4102863e014ad1567b72075fdc16b6139b0f11b73bc2646cc693e62f483"
            "772de61eec5fd0951045f7628800b98cdf58c1e3181837ec20140cf76b52921e58"
            "2ed5a0301e68a655407edec13428f6eb3835ef2b7010ebc3733c4f2865fe785f5b"
            "9257d91141f9c024b84d1453bc6a8127d871814bee6b1761ffddf73f6182556534"
            "4cd03a1612395f9c0213312212492c71d25959f8634952bec26b6e66beee0f9e34"
            "ac0e1ef2da22a2fbc401e0424c215431304ae44fe322d237c06f17046614c27c57"
            "3cbda9b97156596c1685c2c90a0dbdff76b206f2399a70875fb81a2f68c4ab635c"
            "ae99893327e2e124315de23e5e1e414cc6cc763044f66c125e322d54"
        ),
        verdict=False,
    ),
    VerifyVector(
        name="wrong_message",
        slot=0,
        message="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        signature=(
            "240000000f0279461232a5564af5887a8fe78f698c3f134b07bd583ce597532d28"
            "01000004000000d26cbd008f8e2b0971f98d7bfa56cc670dd9c3561f7919180792"
            "6756a7afb3422b23490094fc777b7719727e8369f2382e677d0baa9e932307d517"
            "6d29cf33791b530a1e8e83f94b5d66b65e1347146e26dd0e587a204f04f1a60565"
            "dd3617175151d163f7ea553e80629c461707a01bf8ee067b2e8c986e3667220aed"
            "5ac631b4102863e014ad1567b72075fdc16b6139b0f11b73bc2646cc693e62f483"
            "772de61eec5fd0951045f7628800b98cdf58c1e3181837ec20140cf76b52921e58"
            "2ed5a0301e68a655407edec13428f6eb3835ef2b7010ebc3733c4f2865fe785f5b"
            "9257d91141f9c024b84d1453bc6a8127d871814bee6b1761ffddf73f6182556534"
            "4cd03a1612395f9c0213312212492c71d25959f8634952bec26b6e66beee0f9e34"
            "ac0e1ef2da22a2fbc401e0424c215431304ae44fe322d237c06f17046614c27c57"
            "3cbda9b97156596c1685c2c90a0dbdff76b206f2399a70875fb81a2f68c4ab635c"
            "ae99893327e2e124315de23e5e1e414cc6cc763044f66c125e322d54"
        ),
        verdict=False,
    ),
    VerifyVector(
        name="off_target_layer",
        slot=0,
        message="000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
        signature=(
            "24000000e1c98352ea8e475ced16312d8780107899b4c05dd65a075d43c8974a28"
            "01000004000000d26cbd008f8e2b0971f98d7bfa56cc670dd9c3561f7919180792"
            "6756a7afb3422b23490094fc777b7719727e8369f2382e677d0baa9e932307d517"
            "6d29cf33791b530a1e8e83f94b5d66b65e1347146e26dd0e587a204f04f1a60565"
            "dd3617175151d163f7ea553e80629c461707a01bf8ee067b2e8c986e3667220aed"
            "5ac631b4102863e014ad1567b72075fdc16b6139b0f11b73bc2646cc693e62f483"
            "772de61eec5fd0951045f7628800b98cdf58c1e3181837ec20140cf76b52921e58"
            "2ed5a0301e68a655407edec13428f6eb3835ef2b7010ebc3733c4f2865fe785f5b"
            "9257d91141f9c024b84d1453bc6a8127d871814bee6b1761ffddf73f61825565ac"
            "022e42d7dbcd7c59e98a1171170528cdad98742ce34d6b33ef6c3e5a9e8b1adbff"
            "0a227ca3980275ae553d9a8ec87150656425deab0e35478512173a66f44b48e558"
            "0a0208db2804d10e48b38de91cb692e159d78fd21da423724228c4d066746f004d"
            "e0582d6c848bdd4c51aa66493c8e563db69c5e18b3ebb20c3f43ad7d"
        ),
        verdict=False,
        note="digit sum 16 against a target of 6",
    ),
)
"""Five signatures upstream accepts and three it refuses, all under `PUBLIC_KEY`."""
