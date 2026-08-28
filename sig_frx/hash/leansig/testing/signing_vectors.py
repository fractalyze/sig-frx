# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""What leanSpec generates and signs — one seed, one key pair, six signatures.

The verifier was gated against upstream on a key upstream drew for itself
([`verify_vectors.py`](verify_vectors.py)). A signer cannot be: `key_gen` reaches
`os.urandom` for the master seed and `secrets.randbelow` for the public
parameter, so nothing about the key it returns is a function of anything. The
cases here substitute both — the same substitution
[`signing.keygen`](../signing.py) makes permanent by taking them as arguments —
and everything below follows from those two values alone. The seed is
[`prf_vectors.py`](prf_vectors.py)'s, so a chain start these signatures release
is one that file already pins on its own.

## Provenance

- **Upstream:** `leanEthereum/leanSpec` at commit
  `0588c2d215a955a516378677a92db2a5666802f3`, the same pin every other vector
  module here carries.
- **Fixture:** none. The archive publishes `PROD` keys and no `TEST` ones, and no
  `(seed, key pair)` family at either preset, so this is the reference
  implementation — third in the order
  [`testing.md`](../../../../docs/reference/testing.md) fixes.
- **The exact calls each value came from:** with `interface.random_parameter`
  and `PRFKey.generate` substituted to return `PARAMETER` and `PRFKey(PRF_KEY)`,
  `TEST_SIGNATURE_SCHEME.key_gen(Slot(0), Uint64(1 << 8))`, then
  `sign(secret_key, Slot(slot), message)`,
  `advance_preparation(secret_key)` and `get_prepared_interval(secret_key)`.
  The bytes are `PublicKey.encode_bytes()` and `Signature.encode_bytes()`, and
  the attempt is recovered by replaying `target_sum_encode` over
  `derive_randomness` from counter zero.
- **Reproducing it:** the environment recipe is
  [`mode_vectors.py`](mode_vectors.py)'s — a Python 3.12 venv with
  `pydantic numpy numba`, and a stub `lean_multisig_py` installed before the
  first `lean_spec` import.

## Why the window is the whole lifetime

`key_gen` snaps a requested slot range out to whole bottom trees and builds only
those, filling the rest of the tree with `merkle.random_domain` — fresh OS
randomness, one digest per unpaired node per level. A partial window's public key
therefore moves when nothing but the pad source does, which was confirmed against
upstream directly: requesting `[0, 32)` at `TEST` gives two different roots under
two different pad constants, and requesting the full `[0, 256)` gives one.
Seventeen pads are still drawn at the full window — one beside each bottom-tree
root and one beside the global root — and every one of them is discarded unread.

So the full window is the only one these vectors could describe, and it is what
[`signing.py`](../signing.py) implements.

## What the cases vary

Slots 0 and 15 are the ends of the first bottom tree and 16 and 31 the ends of
the second, so the pair crosses the boundary that decides which resident tree
serves the path; slot 1 differs from slot 0 only in the low bit every Merkle
level selects on. Slot 40 is past the prepared window entirely and is reached by
advancing it, which is the one case that gates the slide.

The attempt counters — 24 to 56 — are recorded because they are what a rejection
loop has to reproduce: a signer that drew its randomness differently, or that
took the last landing candidate of a block rather than the first, would land on
some other attempt and produce a valid signature that is not this one.
"""

from __future__ import annotations

from typing import Final, NamedTuple

from sig_frx.hash.leansig.testing.prf_vectors import PRF_KEY

__all__ = [
    "MAX_ADVANCES",
    "PARAMETER",
    "PREPARED",
    "PREPARED_AFTER_ONE_ADVANCE",
    "PREPARED_AT_END",
    "PRF_KEY",
    "PUBLIC_KEY",
    "SIGN_VECTORS",
    "SignVector",
]


class SignVector(NamedTuple):
    """One signature, and what the search that produced it settled on."""

    name: str
    slot: int
    message: str
    signature: str
    attempt: int
    """Which randomness draw landed on the target layer, counting from zero."""
    codeword: tuple[int, ...]
    """That draw's digits — one per chain, summing to `TARGET_SUM`."""


PARAMETER: Final = (1000, 1001, 1002, 1003, 1004)
"""The public parameter, in leanSpec's order — substituted for
`random_parameter`. Consecutive rather than random on purpose: a transposed
parameter vector is what these five values are chosen to expose."""

PUBLIC_KEY: Final = (
    "b00f566dc5acdc558bd2a03a868aed2bc61a1a14c4892c131e7ab90be479e931e8"
    "030000e9030000ea030000eb030000ec030000"
)
"""The `TEST` key pair `PRF_KEY` and `PARAMETER` generate, SSZ-encoded."""

PREPARED: Final = (0, 32)
"""The slots a fresh key can sign — two bottom trees of sixteen."""

PREPARED_AFTER_ONE_ADVANCE: Final = (16, 48)
"""Where one `advance_preparation` puts that window."""

MAX_ADVANCES: Final = 14
"""How many times the window slides before it stops moving — `2^(h/2) - 2`."""

PREPARED_AT_END: Final = (224, 256)
"""Where it stops: hard against the end of the lifetime."""

SIGN_VECTORS: Final[tuple[SignVector, ...]] = (
    SignVector(
        name="slot0",
        slot=0,
        message="000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
        signature=(
            "24000000660fe337b7d2231d45f8e90c9a0ac552570ea676e5628e2b714cc51a28"
            "01000004000000c8692e2d257c05425fb993469732d94ca17b41393ece6d556da4"
            "0969422858287bc3135ce90f9d4af38a6637a6f000687f61d73e56b6367bc989d1"
            "4e3150a63e9e3f3d354bbe2809f28b6d42d35bf134850a5f1799f3951f1757385e"
            "65e96328b4a8cc568499d81e369442188f1785485d351d4a3a48c325ef732e6258"
            "62316517da1328dafc4343cccd596f49d6c665c1db845e708c7c28922784201d9f"
            "a472727ad932bb6eb64cb28521704d72f82034deae06a9e7332a1bf559604acf3b"
            "77c198687c00b5b61f4ca18f2373a70946d5187b48d3eac901e44eb9521a0fa46c"
            "2be52c75d54354068f41226bff3b7934d951ee10fb15c4320d7f235ffe99725bed"
            "afdf6a1fe9726eedead57dfcac902d79e998612650727d6c033a4cc3f3f87e2bd2"
            "e836c5d3236a23854912f2149c2d6ef98859d2c36a7c40b61a08bd49a30dc43240"
            "421c9cdb7896a5e615873b27588a26ed0044e7c2179cc4a7212e00e97a68e77046"
            "df131e126702c86a11581f53d669f13c1fbeba1ee1a0f10994606d29"
        ),
        attempt=24,
        codeword=(1, 1, 4, 0),
    ),
    SignVector(
        name="slot1",
        slot=1,
        message="000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
        signature=(
            "240000006e729758e67a62455f9d5d5b54c3da3fc6c2e644c4241a4326d8014c28"
            "010000040000001a299615a9200559a3424138aa120b1913d1767be3ce2608add6"
            "f31c3dfec9447bc3135ce90f9d4af38a6637a6f000687f61d73e56b6367bc989d1"
            "4e3150a63e9e3f3d354bbe2809f28b6d42d35bf134850a5f1799f3951f1757385e"
            "65e96328b4a8cc568499d81e369442188f1785485d351d4a3a48c325ef732e6258"
            "62316517da1328dafc4343cccd596f49d6c665c1db845e708c7c28922784201d9f"
            "a472727ad932bb6eb64cb28521704d72f82034deae06a9e7332a1bf559604acf3b"
            "77c198687c00b5b61f4ca18f2373a70946d5187b48d3eac901e44eb9521a0fa46c"
            "2be52c75d54354068f41226bff3b7934d951ee10fb15c4320d7f235ffe99725b84"
            "53c604d361f4543c3cf8330d986b517ce8060088f3107484812219ee4f6446b1ea"
            "f9475c7bd412c42694708099f54cc7c9504571740269a91c914cafd98f276c245f"
            "4badec3a267faadd51d2872f0eaf01e933de7eff41889ece23f87aaa6a0be8c919"
            "881ab644e1b3e76839414c6ee797a811020ae67acb63786bbe6dd176"
        ),
        attempt=35,
        codeword=(2, 2, 2, 0),
    ),
    SignVector(
        name="slot15",
        slot=15,
        message="1111111111111111111111111111111111111111111111111111111111111111",
        signature=(
            "24000000f564f04d16c5eb11dad1731b4db6e16a49df025626071472f219652e28"
            "01000004000000bc13354b03b7737e1b2076760a9e8c2be427bd5bfbfc300c4a82"
            "6f54f4da753eb44ce21b048163007ffe9c3f0994ca7d780e9e0561a25e2ee6f274"
            "59d5e81160f63caf52ef4cbe6eddc28d63ffbc335254cc3b6958e3b33d83a30c47"
            "48e2623b6bbc2a458d3dd72a3f3729517d0f84754f6191438e46ea5a8a5309437f"
            "f67f5517da1328dafc4343cccd596f49d6c665c1db845e708c7c28922784201d9f"
            "a472727ad932bb6eb64cb28521704d72f82034deae06a9e7332a1bf559604acf3b"
            "77c198687c00b5b61f4ca18f2373a70946d5187b48d3eac901e44eb9521a0fa46c"
            "2be52c75d54354068f41226bff3b7934d951ee10fb15c4320d7f235ffe99725b5b"
            "7350451b33a55e14589669e096544a6442033d378da45ae2b6b40fed034e4c18e9"
            "7261e8ba5a72195b2307e8e8886972030e26bf96c55ee90e537908fbca3c5dad2b"
            "68900a14423aa2576f3d1f2f7de3b895129510d50452cb2d0918a2d32ce3cefb06"
            "4e02b44b52490329ef57d903ee2b160ede97e12752e21e2ab2f53311"
        ),
        attempt=26,
        codeword=(0, 3, 1, 2),
    ),
    SignVector(
        name="slot16",
        slot=16,
        message="2222222222222222222222222222222222222222222222222222222222222222",
        signature=(
            "24000000ffb3546f304ab0331b232d6578155969296ce46968f8f03660b56c2828"
            "01000004000000a0df40767c9b3a53fdc8f43b66a1c1687a670c55f21041363b58"
            "851ef0f80f3f2e678c46c36c2156a13ef11746833855e53e804ad632eb56b1faa4"
            "7306968b588b717c4bc61416247ceeaa0c29071c5713cb0b28fa72b005aa57ef0d"
            "d9c5dd57fffb6f6f22944b4134e6426511a90533ffef12417ab9543792db7300fc"
            "790946da3e3837cda13228062d6e7e195a42656fdb9207ca67ab2a9851f70b0255"
            "765e727ad932bb6eb64cb28521704d72f82034deae06a9e7332a1bf559604acf3b"
            "77c198687c00b5b61f4ca18f2373a70946d5187b48d3eac901e44eb9521a0fa46c"
            "2be52c75d54354068f41226bff3b7934d951ee10fb15c4320d7f235ffe99725b98"
            "f762344fc22a75a5ac73627d0a1a266aaf750cf0237f621ca3ae7644ad1f0cc098"
            "f339651a80050cb4e057a29a043dd5387354e4dd6350d73232646445e922e8580f"
            "334053c34397da1b2c2d345d75b9935233ff65e56a01f30d0067880615a3c5dc43"
            "2f6cc85bed8dc2001913bb1599eb2f62c20685663aa03734cdf4e808"
        ),
        attempt=24,
        codeword=(2, 2, 1, 1),
    ),
    SignVector(
        name="slot31",
        slot=31,
        message="202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f",
        signature=(
            "240000006b67d2208be6814797bf18413fc4fe07d4df672b5415243e3cca0d5628"
            "010000040000003ffbf22287375e0d29fe92625b55910faaefbd107375f96fab7e"
            "665568767c18de68527eb6b7dd1c87a75e64d753942e1a38b041d5e715471cb569"
            "4916fd622cda5bd411c4aa7b1477b947348d2729374814b705b28b7e310668547d"
            "f8eb4048bf5b646985151617dc4d264fff17e8536722f268d8108878594e024e23"
            "84115cda3e3837cda13228062d6e7e195a42656fdb9207ca67ab2a9851f70b0255"
            "765e727ad932bb6eb64cb28521704d72f82034deae06a9e7332a1bf559604acf3b"
            "77c198687c00b5b61f4ca18f2373a70946d5187b48d3eac901e44eb9521a0fa46c"
            "2be52c75d54354068f41226bff3b7934d951ee10fb15c4320d7f235ffe99725b56"
            "ac2e4627e0a94d05f77a424d3f1f0ef273b2598e694e05f2e3b612c0cb526f3aba"
            "7529b7faa23cab7b6a034d2e18269186c364c65c526fe955da6a1aa7a504f52942"
            "647ce8ef5eae34da13ea9a70190241e76bbb74c3196e8b85211e86ab506fd37b52"
            "6b018e46ce5e6f4392d3136fbda8bf6ce0f82928185247383f0d975b"
        ),
        attempt=48,
        codeword=(2, 3, 0, 1),
    ),
    SignVector(
        name="slot40_after_one_advance",
        slot=40,
        message="5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a",
        signature=(
            "24000000b998286ab4f32d2ed1ae525f44a7cb40a1faa454106e5173d968260928"
            "01000004000000554b717649209866adfd3267fd153255e788cb605bce2d7bc2fa"
            "9e5ed9aed76555f1313235866757bcb01149240b0d753fa09c706f63bc444b82d7"
            "549d94087bb3c2c10d435bee5a8bba967559033d0d7f92ff3c352e4d063831a74a"
            "6bf6075810b37f07642ed077dbd4ca2b460efd3c49693c0cd983b27e9511e96014"
            "154a07674604497c478b1b2a9a9a7c2644d93cefd6b823c90e513076006c7afc18"
            "ef5332ba0325641f801cf3dccd421e922355746aeb552831864786556246130bda"
            "4cc198687c00b5b61f4ca18f2373a70946d5187b48d3eac901e44eb9521a0fa46c"
            "2be52c75d54354068f41226bff3b7934d951ee10fb15c4320d7f235ffe99725bb5"
            "fdcf61dc668731421605273a467c4919c9d16e40241f611b1993419c30716ac55e"
            "3741e05da2203fc4244f0ef7d90c87981f381c225a1566ca824767b4eb6564b211"
            "1645db406890eadf124e6bc70845d9d67878bba349eaf5c85a2b9591248f84d157"
            "2a97021da0700b30f306aa0fa12bcf5ff7b08640a1de44752dadea2b"
        ),
        attempt=56,
        codeword=(0, 4, 2, 0),
    ),
)
