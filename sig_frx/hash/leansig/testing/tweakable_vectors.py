# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""What leanSpec computes for the tweakable hash family and the Merkle walk.

The same gap [`mode_vectors.py`](mode_vectors.py) records: the fixtures archive
pins the permutation, the SSZ containers and the `PROD_CONFIG` keys, and nothing
in it pins a `tweak_hash` or a tree. So the reference implementation is the
authority, third in the order
[`testing.md`](../../../../docs/reference/testing.md) fixes, and gating on one
costs the provenance below.

## Provenance

- **Upstream:** `leanEthereum/leanSpec` at commit
  `0588c2d215a955a516378677a92db2a5666802f3`, the same pin every other vector
  module here carries.
- **Fixture:** `src/lean_spec/spec/crypto/xmss/poseidon.py`, against the
  `PoseidonXmss` singleton `POSEIDON`, and
  `src/lean_spec/spec/crypto/xmss/merkle.py` for the tree.
- **The exact call each value came from:** `POSEIDON.tweak_hash(config,
  parameter, tweak, digests)` at one, two and `DIMENSION` digests;
  `POSEIDON.hash_chain(config, parameter, epoch, chain_index, start_step,
  num_steps, start_digest)`; and `HashSubTree.new(...)` with its `root()` and
  `path(position)`.
- **Reproducing it:** the environment recipe is
  [`mode_vectors.py`](mode_vectors.py)'s — a Python 3.12 venv with
  `pydantic numpy numba`, and a stub `lean_multisig_py` installed before the
  first `lean_spec` import.

## The two references that are transcriptions rather than calls

Upstream's verifier, `merkle.verify_path`, returns a **bool**: it recomputes a
root internally and compares, so there is no root to read off it. `TREE_WALK_
VECTORS` therefore come from transcribing that function's own loop over
`POSEIDON.tweak_hash` — and every root so produced was handed back to the real
`verify_path`, which had to accept it. That round trip is what gates the
transcription instead of assuming it, and it is the same move
[`encoding_vectors.py`](encoding_vectors.py) makes against upstream's decode.

`TREE_VECTORS` need no transcription: `HashSubTree.new` is called directly. It
pads a layer with *fresh randomness* when the layer starts on an odd index or
ends on an even one, which would make a reference irreproducible — so these
start at index 0 with a power-of-two leaf count, where every layer below the root
starts even and ends odd and no padding is reached. The root layer does get a
random pad appended beside the root, at index 1; `root()` reads index 0 and
`path()` never looks at that layer.

## Why the inputs are rules and only the outputs are transcribed

As in the two sibling modules: the inputs are this repo's to choose, so they come
from [`mode_vectors.operand_elements`](mode_vectors.py) rather than from a second
rule written to resemble it. A digest is `operand_elements(8, seed)` and a public
parameter `operand_elements(5, seed)`, so a case carries the seeds and not 46
transcribed vectors — one of these leaves would otherwise be 368 integers.

## What the cases vary

- **The chain step** moves every field of a packed chain tweak independently: the
  all-zero position, a middling one, and the largest each field holds — epoch
  `2^32 - 1` (the last slot a `LOG_LIFETIME = 32` key has), chain 45 (the last at
  `PROD`) and step 7 (`BASE - 1`). A tweak that packed one field at the wrong
  shift passes the zero case and fails these.
- **The interior node** moves the level and the index together, including the
  root level and an odd index — the two are packed adjacently, so an index that
  carried into the level would still be a valid tweak for a different node.
- **The leaf** runs at both presets, because it is the one hash whose shape
  depends on `DIMENSION`: 4 digests pad to three sponge chunks at `test` and 46
  fill exactly 25 at `prod`, and the domain separator differs with it.
- **The chain walk** covers a full `BASE - 1` walk from the start, a partial walk
  from the middle, and a walk of no steps at all — the last being the identity
  the masked walk has to reproduce for a chain whose digit is already `BASE - 1`.
- **The tree walk** moves the leaf position's parity at every level: 0 is left
  the whole way up, 255 is right the whole way up, and 173 alternates.
"""

from __future__ import annotations

from typing import Final, NamedTuple


class ChainStepVector(NamedTuple):
    """One `tweak_hash` at a single digest — a Winternitz chain step."""

    name: str
    parameter_seed: int
    digest_seed: int
    epoch: int
    chain_index: int
    step: int
    """Upstream counts from one: step zero is the chain start, never hashed."""

    output: tuple[int, ...]
    """Upstream's digest, in leanSpec's lane order."""


class TreeNodeVector(NamedTuple):
    """One `tweak_hash` at two digests — a Merkle interior node."""

    name: str
    parameter_seed: int
    left_seed: int
    right_seed: int
    level: int
    node_index: int
    output: tuple[int, ...]


class LeafVector(NamedTuple):
    """One `tweak_hash` at `dimension` digests — a Merkle leaf, through the sponge."""

    name: str
    preset: str
    """Which preset this case runs at, keyed into `harness.PRESETS`. Named rather
    than implied by a digest count: `DIMENSION` is what the sponge's shape and its
    domain separator depend on, so a case that resolved its preset *from* the
    dimension would answer with a default for one it did not recognise."""

    parameter_seed: int
    chain_end_seed: int
    """Seeds the first chain end; end `i` is `operand_elements(8, seed + i)`."""

    position: int
    output: tuple[int, ...]


class ChainWalkVector(NamedTuple):
    """One `hash_chain` — the chain step iterated, at its own tweak per step."""

    name: str
    parameter_seed: int
    digest_seed: int
    epoch: int
    chain_index: int
    start_step: int
    num_steps: int
    output: tuple[int, ...]


class TreeWalkVector(NamedTuple):
    """One verifier walk: the leaf hash, then one sibling per level to the root."""

    name: str
    preset: str
    log_lifetime: int
    """Not a `LeanSigParams` column yet — the tree slice is what will read it."""

    parameter_seed: int
    chain_end_seed: int
    sibling_seed: int
    """Seeds the lowest sibling; level `i`'s is `operand_elements(8, seed + i)`."""

    position: int
    root: tuple[int, ...]


class TreeVector(NamedTuple):
    """One whole tree upstream built, with its root and some of its openings."""

    name: str
    depth: int
    parameter_seed: int
    leaf_seed: int
    """Seeds leaf 0; leaf `i` is `operand_elements(8, seed + i)`. These are
    digests standing in for leaf hashes — the subject is the walk above them,
    and upstream's own builder takes its lowest layer as given too."""

    root: tuple[int, ...]
    positions: tuple[int, ...]
    paths: tuple[tuple[tuple[int, ...], ...], ...]
    """One opening per position, lowest sibling first."""


CHAIN_STEP_VECTORS: Final = (
    ChainStepVector(
        name="chain_step_zero_position",
        parameter_seed=1101,
        digest_seed=1102,
        epoch=0,
        chain_index=0,
        step=1,
        output=(
            1429163510,
            585607859,
            22730839,
            227853052,
            1405291921,
            1332593937,
            2067881469,
            1096000769,
        ),
    ),
    ChainStepVector(
        name="chain_step_mid_position",
        parameter_seed=1102,
        digest_seed=1103,
        epoch=12345,
        chain_index=17,
        step=4,
        output=(
            1394234632,
            2053590086,
            2121668344,
            138911912,
            1320383089,
            1351271722,
            757218314,
            1339174204,
        ),
    ),
    ChainStepVector(
        name="chain_step_last_step",
        parameter_seed=1103,
        digest_seed=1104,
        epoch=4294967295,
        chain_index=45,
        step=7,
        output=(
            169049178,
            1435566426,
            453508793,
            954130397,
            1802675631,
            674844767,
            457138263,
            1503805931,
        ),
    ),
)

TREE_NODE_VECTORS: Final = (
    TreeNodeVector(
        name="tree_node_first_level",
        parameter_seed=2101,
        left_seed=2102,
        right_seed=2103,
        level=1,
        node_index=0,
        output=(
            1492266172,
            232163570,
            442372293,
            1607921474,
            358598458,
            959899215,
            1335866632,
            522643746,
        ),
    ),
    TreeNodeVector(
        name="tree_node_mid_level",
        parameter_seed=2102,
        left_seed=2103,
        right_seed=2104,
        level=9,
        node_index=173,
        output=(
            1745740673,
            612027623,
            174741385,
            649987531,
            82888542,
            1105186374,
            700547383,
            864385240,
        ),
    ),
    TreeNodeVector(
        name="tree_node_root_level",
        parameter_seed=2103,
        left_seed=2104,
        right_seed=2105,
        level=32,
        node_index=1,
        output=(
            1130405121,
            54675843,
            43890333,
            1703056605,
            1101468827,
            249143002,
            105575973,
            316410330,
        ),
    ),
)

LEAF_VECTORS: Final = (
    LeafVector(
        name="leaf_test_config",
        preset="test",
        parameter_seed=3101,
        chain_end_seed=3102,
        position=7,
        output=(
            1410550020,
            1894628146,
            1905404315,
            1160693768,
            741465090,
            1723886492,
            147725495,
            579236266,
        ),
    ),
    LeafVector(
        name="leaf_prod_config",
        preset="prod",
        parameter_seed=3102,
        chain_end_seed=3103,
        position=4000000000,
        output=(
            852885937,
            1539972656,
            1555107416,
            786834952,
            1658392128,
            1040986097,
            1334978784,
            932747859,
        ),
    ),
)

CHAIN_WALK_VECTORS: Final = (
    ChainWalkVector(
        name="walk_full_chain",
        parameter_seed=4101,
        digest_seed=4102,
        epoch=99,
        chain_index=3,
        start_step=0,
        num_steps=7,
        output=(
            1340194434,
            836638914,
            814934800,
            781301232,
            629516552,
            66142518,
            517318328,
            2010706626,
        ),
    ),
    ChainWalkVector(
        name="walk_partial_from_mid",
        parameter_seed=4102,
        digest_seed=4103,
        epoch=2147483648,
        chain_index=12,
        start_step=3,
        num_steps=4,
        output=(
            1306556671,
            1041090626,
            1907503722,
            1421061253,
            1980820908,
            706084314,
            1753602551,
            1637631107,
        ),
    ),
    ChainWalkVector(
        name="walk_no_steps",
        parameter_seed=4103,
        digest_seed=4104,
        epoch=5,
        chain_index=0,
        start_step=6,
        num_steps=0,
        output=(
            4104,
            523733432,
            1047462760,
            1571192088,
            2094921416,
            487944311,
            1011673639,
            1535402967,
        ),
    ),
)

TREE_WALK_VECTORS: Final = (
    TreeWalkVector(
        name="path_test_config_even_leaf",
        preset="test",
        log_lifetime=8,
        parameter_seed=5101,
        chain_end_seed=5102,
        sibling_seed=5201,
        position=0,
        root=(
            1061002397,
            1090931040,
            688413189,
            1987811168,
            1714691568,
            7447364,
            1619167053,
            632282051,
        ),
    ),
    TreeWalkVector(
        name="path_test_config_odd_leaf",
        preset="test",
        log_lifetime=8,
        parameter_seed=5102,
        chain_end_seed=5103,
        sibling_seed=5202,
        position=173,
        root=(
            862223147,
            403718221,
            1210715994,
            94638271,
            1804185090,
            1894689711,
            990171618,
            902384452,
        ),
    ),
    TreeWalkVector(
        name="path_test_config_last_leaf",
        preset="test",
        log_lifetime=8,
        parameter_seed=5103,
        chain_end_seed=5104,
        sibling_seed=5203,
        position=255,
        root=(
            758782537,
            1902092325,
            1920976904,
            42523875,
            1663414641,
            36668442,
            1734883837,
            1805014815,
        ),
    ),
)

TREE_VECTORS: Final = (
    TreeVector(
        name="tree_depth_four",
        depth=4,
        parameter_seed=6101,
        leaf_seed=6102,
        root=(
            1858441605,
            972934494,
            735401110,
            1603936825,
            1988465008,
            213113253,
            106718472,
            1172568493,
        ),
        positions=(0, 5, 15),
        paths=(
            (
                (
                    6103,
                    523735431,
                    1047464759,
                    1571194087,
                    2094923415,
                    487946310,
                    1011675638,
                    1535404966,
                ),
                (
                    400876442,
                    1109488077,
                    113648240,
                    964756780,
                    306126102,
                    1999107343,
                    1312289847,
                    878353162,
                ),
                (
                    262854108,
                    801322802,
                    340056242,
                    1703361599,
                    1826996066,
                    1314221267,
                    1411081797,
                    27683238,
                ),
                (
                    305682277,
                    541209637,
                    2075160960,
                    54363133,
                    348716794,
                    511457151,
                    1728530295,
                    1078439804,
                ),
            ),
            (
                (
                    6106,
                    523735434,
                    1047464762,
                    1571194090,
                    2094923418,
                    487946313,
                    1011675641,
                    1535404969,
                ),
                (
                    1804821328,
                    989591323,
                    1354396362,
                    243215894,
                    1610846285,
                    1423721252,
                    556366999,
                    113218356,
                ),
                (
                    1449081359,
                    889406884,
                    723417703,
                    325140921,
                    269120018,
                    1720803329,
                    34527116,
                    1477107569,
                ),
                (
                    305682277,
                    541209637,
                    2075160960,
                    54363133,
                    348716794,
                    511457151,
                    1728530295,
                    1078439804,
                ),
            ),
            (
                (
                    6116,
                    523735444,
                    1047464772,
                    1571194100,
                    2094923428,
                    487946323,
                    1011675651,
                    1535404979,
                ),
                (
                    1293543824,
                    21853148,
                    2010084846,
                    958696522,
                    1534162656,
                    379562114,
                    1103491982,
                    58924756,
                ),
                (
                    593654953,
                    15631000,
                    2036648065,
                    234060563,
                    68693356,
                    1195909273,
                    694304653,
                    209580588,
                ),
                (
                    231315247,
                    768692160,
                    1688301936,
                    1128049544,
                    1423281988,
                    1201063157,
                    1091066401,
                    557509967,
                ),
            ),
        ),
    ),
)
