# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""What leanSpec's XMSS containers encode to — the one layer upstream publishes.

Every other leanSig suite here gates against the reference implementation,
because the fixtures archive pins the permutation and nothing above it
([`mode_vectors.py`](mode_vectors.py) says so at length). The wire format is the
exception: the archive carries a `test_xmss_containers` family, so this slice has
a *published* gate — first in the order
[`testing.md`](../../../../docs/reference/testing.md) fixes, rather than third.

What it publishes is `PROD_CONFIG` only, and leanSig's other preset changes both
variable-length runs. So the `TEST` case below comes from the reference
implementation after all, through the same recipe the other suites use; the two
families are kept apart here because their provenance differs and a reader
should not have to guess which is which.

## Provenance

- **Upstream:** `leanEthereum/leanSpec` at commit
  `0588c2d215a955a516378677a92db2a5666802f3`, the same pin the constants carry.
- **Published fixtures:** release tag `latest`, asset
  `fixtures-prod-scheme.tar.gz` (published 2026-08-16), under
  `fixtures/consensus/ssz/lstar/ssz/test_xmss_containers/`. Each vector below
  names the test id its file records. The `serialized` field is what is
  transcribed; the `root` beside it is the container's SSZ hash-tree root, which
  is a consensus-layer identity rather than part of the signature scheme and is
  not read here.
- **The generated case:** `Signature(...).encode_bytes()` from
  `src/lean_spec/spec/crypto/xmss/containers.py`, with `LEAN_ENV=test` so that
  `TARGET_CONFIG` resolves to `TEST_CONFIG`, over the elements
  [`mode_vectors.operand_elements`](mode_vectors.py) returns for the seeds each
  case names.
- **Reproducing either:** the archive is a plain `gh release download`, and the
  reference run needs the `lean_multisig_py` stub
  [`mode_vectors.py`](mode_vectors.py) records — the same Python 3.12 venv with
  `pydantic numpy numba`.

**The archive is published under a moving tag**, so nothing can pin a URL to it:
`latest` is regenerated per spec commit. That is why the bytes are transcribed
here rather than fetched, and why the commit above is the pin that matters. A
re-spin means re-running both halves and diffing this file, which is the whole
of the procedure [issue #195](https://github.com/fractalyze/sig-frx/issues/195)
asks for before a KAT lands under a versionless spec.

## Why only four elements of a published signature are transcribed

A `PROD` signature holds 631 field elements and none of them is derivable — it
came out of a real signing run, not a rule. Transcribing all of them would put
2.5 KB of digits beside the 2.5 KB of bytes they were read from, gating one
transcription against another. The four corners plus `rho` are what fixes the
*layout*: they pin where each run starts and ends, and the lane order within a
digest. Everything between them is then gated by the round trip, which cannot
pass if a run is misplaced.

The generated case is the other half of that argument. Its elements are a rule,
so it gates the encode direction from values rather than from bytes — which is
what a round trip alone cannot do, since an error symmetric across encode and
decode survives one.
"""

from __future__ import annotations

from typing import Final, NamedTuple


class PublicKeyVector(NamedTuple):
    """One published `PublicKey`, and the two vectors it decodes to.

    Both are in leanSpec's lane order. The encoding reads no preset column that
    the two presets differ in, so a case here runs at either.
    """

    name: str
    test_id: str
    """The fixture's own test id, so a re-spin diffs against the same case."""

    encoded: str
    root: tuple[int, ...]
    parameter: tuple[int, ...]


class PublishedSignature(NamedTuple):
    """One published `Signature`, and the corners of what it decodes to.

    In leanSpec's lane order, and `PROD` throughout — the archive publishes no
    other preset.
    """

    name: str
    test_id: str
    encoded: str
    rho: tuple[int, ...]
    first_sibling: tuple[int, ...]
    last_sibling: tuple[int, ...]
    first_hash: tuple[int, ...]
    last_hash: tuple[int, ...]


class GeneratedSignature(NamedTuple):
    """One `Signature` this repo built and upstream's own encoder serialized.

    Every element is a seed away, so a case here gates the encode direction
    against values rather than against a round trip.
    """

    name: str
    preset: str
    encoded: str
    rho_seed: int
    sibling_seed: int
    """Level `i` of the authentication path seeds at `sibling_seed + i`."""

    hash_seed: int
    """Chain `i`'s released end seeds at `hash_seed + i`."""


PUBLIC_KEYS: Final = (
    PublicKeyVector(
        name="public_key_typical",
        test_id="tests/consensus/lstar/ssz/test_xmss_containers.py::test_public_key_typical[fork_Lstar]",
        encoded=(
            "010000000200000003000000040000000500000006000000070000000800000064000000"
            "65000000660000006700000068000000"
        ),
        root=(
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ),
        parameter=(
            100,
            101,
            102,
            103,
            104,
        ),
    ),
    PublicKeyVector(
        name="public_key_zero",
        test_id="tests/consensus/lstar/ssz/test_xmss_containers.py::test_public_key_zero[fork_Lstar]",
        encoded=(
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "00000000000000000000000000000000"
        ),
        root=(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ),
        parameter=(
            0,
            0,
            0,
            0,
            0,
        ),
    ),
)
"""Both published public keys: ascending non-zero elements, and all zeros.

The zero case is not filler. Zero is the one residue whose Montgomery
representative is also zero, so it is the single value a bitcast would carry
correctly — a decode that reinterpreted bytes instead of converting them passes
this case and fails the one above it.
"""


PUBLISHED_SIGNATURES: Final = (
    PublishedSignature(
        name="signature_actual",
        test_id="tests/consensus/lstar/ssz/test_xmss_containers.py::test_signature_actual[fork_Lstar]",
        encoded=(
            "240000007b4ebf6feb9f3e7e13e8b74ac3d9a10a39b1372f367a252af517986928040000"
            "04000000e62e6f6c6d0f142b76481c5c5117f0384623c77298028b6d640576721224bf7b"
            "2322c937dce4db3931cbed5faf9783264929f415d11b401c0f929663bc39284a5c7fef4d"
            "341e57301dddd77ada94921c1d8af57146affe6ccbf80a50cd7e7a007d346f06b35a9431"
            "b39a12282d0b9c6161722a75909ca34e25b6b34f88ff1847ea2b2c445c86ef4b099d4875"
            "1915826723aa534b78eda76bb3fa153a36377957d92b8f7770a508365751c95b968d4627"
            "4cdadf46b061234f3b62640689aad9169cc8645048b4954e3301222a1d20a47e9e78e56a"
            "508e9626c36de26999ab272aa2794b70a475af64f96af52eb25aae784c87ce5998d4d519"
            "369b280f20c874604526b51a7e35672d1ffaae434c197c1abbce780441157d6e185ac451"
            "060634780c1cfe112ea7fe10721a9e125342f8414b6edd473bd5df7cd6553e0da56aed5e"
            "1268133979919a0da2a1960e17a9aa53a2456145befdf761885abd1bbcbaf13f8ef3cc54"
            "27251a792eb89351037da713b41eff4b8839b924a1185d20a0799a695edab959af3e395d"
            "78053102aca9cf543dcacb064076fb60a0ee684cc3b1a30679484f722989a576f4c20d18"
            "abb9c310c37eee198a40dc17874bef5b3dd4c63da8a88558b0218c154598cf2ddac6af06"
            "0cf0f351dac9e83f4f440a5b6404df269a8b6311ed7e54110eaf3414e587e64645885b46"
            "7d2bf42fc343d075107ba80d4c68d404872d5b058547420a2ac9492d3909465470f3ed78"
            "74b4240edfb9401cff9e354a8b9cc24364abc557d08b521c91bcaa29edeb15157e280924"
            "20f61e15bc2f08037753ac1d1ea1902d7d39d17cb6e15670fdacc04ab9a0042b13ab364c"
            "12a876541b10be70f6bacd2c99b583239d24361f221cf2559ca5501076ae20334e7be91c"
            "10eb256e06d93662b36ecc01bedfd40552189a5535157f1bd67c3c71edec82211904d23c"
            "c5a931164b7f456aefde270eba123d6b61c44f4c106d9869c52401648d350e2ff9053c26"
            "62be7c57e5a27609246e8d27fa9915592ce23b16ecee9602bcd035531215634a21d18a2a"
            "b1d14a726d101575ec129f23a456716cfbff7a4f68771d63e23c6514996d890285d72972"
            "593396672662e54060bed760dfdb4f68bd24f31f3e0414653c969b5c6596f361cfcc6e79"
            "b99d653011904d7024a1344a9ff54370df18327365d9ff776aa2ae47117b8a4643eb524d"
            "da68016b7e8db40a83d42753a1e36d652d83c50f6e5e9e4e94747b430d7911719f9e7c74"
            "d56a6c1ccee5d640c2985e65ca9940442b892733ce5352004001b31ffbd63d603797aa1d"
            "d036ce189976b041ecc45f0dda074313a872bd0f8eeb694845d0421607cd19585a9f3c1d"
            "00a2504000b9b3432e77b227393ecf166d2486534abbd4263edea60b17462368a974824f"
            "ad9f6e3229b2d262a66ace497bc98b729c850c4ee4e0390e1db47e767b98ce66ffe67f38"
            "948f5f6474c84829bdba3c4c1791ef1a054eff74c5cced2fdb7aec41e0c5b04b553d4d49"
            "4c5d3675876f23339eb01b3eefae4020b14e05046c56e2089503f81435b2833a2b18fc45"
            "1c96d47b0019322d57e5d17c56bc0d5bbcdbe7547f6d857a2e7b971281f03e03ae9c6d6c"
            "3449f8726dbeab10457d4956a5a4d922ab8d086b7ae0b01fb3189d42339f332c84fec260"
            "240d2747bc9f2a7c00d3a40d157f744b2f4f4e6913321d0eb672db29fc5dc2458e9dae14"
            "8c7699015765b34ee2f44d63a1131646199bd63f5510ad723c27bd3a927feb27b6555076"
            "04a58b08e4ea2f1839d8a200fa435b084012557d5be85046f16de3661b955b71a917c948"
            "09f4af236084d17cbcdf9f733630902115c5286de6683c23f45db5088d26821c025f3172"
            "53eb6947560dc423b247704d3669db485de34171e250ee4a278d9c77af8a7c10cb21784a"
            "82c08832b3d148608249af18372803342734bb0b871a907b9a0986487d2c6b0bbd061e4f"
            "3f1abb7354f0f9027c04ea5aaccbbd33e0ac60093398601c7d16714207f43c42c1138e3c"
            "99605f63329f2c75de8eec3064c96231b76cb77ba46ba54ec9e3402f5cda9d202cee1f37"
            "710c2278231acb5c6d0b72112d29ca551cf33e52bb79c8284ee3673d513df53f66736700"
            "b65bbb4f997d036563840a041ccde6181843fc0bc4d18e5e5c993609c9223e5f2e154a23"
            "c6db55386ff49a05d297cf3886742a71253967707553a531b9b21d01114b9350f83bcf20"
            "438d44207e147504b2b6e4018ea1047d0fd1db7310c9e55bb4955638afa80f5b524ac725"
            "ee57433785d30f5c0a7f936c764dd5186af9a26dab22be63e1e88e3b55280b7d45ed0730"
            "5a1fb7375f0315210c4442312eb4fd1583fd17318874442f762bd207f40d1577a0d14264"
            "8a7a5b22f3b8c728c51790415fecbf707f8d817595cc865a6f3d6241a5c8cf7b05ca2825"
            "44150357ecc70115733f385b086f96799b845354f3e49f0312a825472d3cf7278e4add0b"
            "144bd436580afb7711c8ed51802a00212a569626de119653dba4b300343d5b1463d0695a"
            "cd2ee22e8a08bf6b1e38525879d6b76f236b6e219667640e1c159076be4b4d4c1b27d022"
            "63e4a203b453e877c9cd07277b307328647ef51888685902c793826f5771b63a974e9773"
            "8cb8263c96567609bf14f429a75eda143ec0604f28325f2ab0da68517c25d7746939e93d"
            "8b3c7c5471629c535f49ab2426eb83604c78d57c58a72224074b526116b83753ec28052d"
            "b16e8b0f0d56b14f211e005a5321336e28fbf711a56774057681401375fc335a0eec0773"
            "65114b13d6662933fde8f35720f8b572e3decd70a0960577bc054053a3424013c381e82d"
            "a3cdb61c2ad2125552891967d25aec21e7b9eb1235a07951024277115594604612d15d24"
            "8678a05fa9d55a1790cda44a837cd5240b0987325b930c451f50a2717ad31c29e231e637"
            "f9f3cb5743d7ed0f473187441b6dfe31884a451126693d01f7e90e02f3ee7b6f0dd8572a"
            "ca034b64dec6d611bcb4f300242e26570911714c2e6b84581a93ea064e50bd3495ae3012"
            "13e8140bde40b1501326c575531c067df8749708cf61816272daf65342145e7e44347b16"
            "b51afb613ce4ef47af5f7a3ce620d35202ae0977db19e54cc0e13c40c1f7d93f0f205333"
            "addb16745f32bb3c5740551e08990670e78c4800ad30ad5207ec1c67528aed1141e1ce22"
            "5f5cd5192bc08c14e1c0f16397114b49dae73b78a5fb3744660f707aa0a3b03c667a8951"
            "9a2db76f6a6db90d43827753d2dfe729a73c56693fcc771e360fb2142143b5054641890f"
            "38bd08579c3aee54e7b1b605c6289d27d4e2ce7b65e8ee1429475b04b40a30278b1e2a37"
            "a928d10b73dee664cfa71f65251e5e654b33dc69f246ac30dc78e8720661787ee021f067"
            "820be81ab6ace32a2db3be707f3d6f6856cc1442a55e393307483a5703a85c7cd00e1100"
            "3f856631969b074416f7ff248aae0205cca46059b4a6d02a2dd8f9255d5ef80c6a440f3f"
            "5b79a93977ae2d18117d4b0e2d39810e"
        ),
        rho=(
            1874808443,
            2118033387,
            1253566483,
            178379203,
            792179001,
            707099190,
            1771575285,
        ),
        first_sibling=(
            1819225830,
            722734957,
            1545357430,
            955258705,
            1925653318,
            1837826712,
            1920337252,
            2076124178,
        ),
        last_sibling=(
            195485246,
            1747142167,
            1333949609,
            846110637,
            1657975337,
            1238264486,
            1921763707,
            1309443484,
        ),
        first_hash=(
            238674148,
            1988015133,
            1724815483,
            947906303,
            1683984276,
            692635764,
            1279048381,
            451907863,
        ),
        last_hash=(
            718317236,
            637130797,
            217603677,
            1057965162,
            967407963,
            405646967,
            239828241,
            243349805,
        ),
    ),
    PublishedSignature(
        name="signature_zero",
        test_id="tests/consensus/lstar/ssz/test_xmss_containers.py::test_signature_zero[fork_Lstar]",
        encoded=(
            "240000000000000000000000000000000000000000000000000000000000000028040000"
            "040000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "000000000000000000000000000000000000000000000000000000000000000000000000"
            "00000000000000000000000000000000"
        ),
        rho=(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ),
        first_sibling=(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ),
        last_sibling=(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ),
        first_hash=(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ),
        last_hash=(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ),
    ),
)
"""Both published signatures: one from a real signing run, and one all zeros.

`signature_actual` is the only leanSig object in the archive that a signer
produced — the families that look like signatures elsewhere in it are
leanMultisig aggregate proofs. It carries no key, message or slot, so it gates
the encoding and not a verification.
"""


GENERATED_SIGNATURES: Final = (
    GeneratedSignature(
        name="signature_test_preset",
        preset="test",
        encoded=(
            "2400000003000000b379371f63f36e3e136da65dc3e6dd7c7260151d22da4c3c28010000"
            "040000000a000000ba79371f6af36e3e1a6da65dcae6dd7c7960151d29da4c3cd953845b"
            "0b000000bb79371f6bf36e3e1b6da65dcbe6dd7c7a60151d2ada4c3cda53845b0c000000"
            "bc79371f6cf36e3e1c6da65dcce6dd7c7b60151d2bda4c3cdb53845b0d000000bd79371f"
            "6df36e3e1d6da65dcde6dd7c7c60151d2cda4c3cdc53845b0e000000be79371f6ef36e3e"
            "1e6da65dcee6dd7c7d60151d2dda4c3cdd53845b0f000000bf79371f6ff36e3e1f6da65d"
            "cfe6dd7c7e60151d2eda4c3cde53845b10000000c079371f70f36e3e206da65dd0e6dd7c"
            "7f60151d2fda4c3cdf53845b11000000c179371f71f36e3e216da65dd1e6dd7c8060151d"
            "30da4c3ce053845b64000000147a371fc4f36e3e746da65d24e7dd7cd360151d83da4c3c"
            "3354845b65000000157a371fc5f36e3e756da65d25e7dd7cd460151d84da4c3c3454845b"
            "66000000167a371fc6f36e3e766da65d26e7dd7cd560151d85da4c3c3554845b67000000"
            "177a371fc7f36e3e776da65d27e7dd7cd660151d86da4c3c3654845b"
        ),
        rho_seed=3,
        sibling_seed=10,
        hash_seed=100,
    ),
)
"""The `TEST` preset, which the archive does not cover.

One case rather than several: what the preset changes is two lengths, and a
length is either right for every element or wrong for all of them. The variety
that matters — a zero element, a real one, the corners of each run — is already
in the published pair above, and it is preset-independent.
"""
