# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""What leanSpec computes in compression and sponge mode, and for a separator.

The permutation these ride on is gated separately, against vectors upstream
publishes ([`poseidon_vectors.py`](poseidon_vectors.py)). The modes are not: the
fixtures archive carries permutation vectors, SSZ container vectors and
`PROD_CONFIG` keys, but its signature-shaped families are leanMultisig aggregate
proofs rather than XMSS signatures, so nothing upstream pins a `compress` or a
`sponge` call. That makes the reference implementation the authority — third in
the order [`conventions.md`](../../../../docs/reference/conventions.md) fixes —
and gating on one costs the provenance below.

## Provenance

- **Upstream:** `leanEthereum/leanSpec` at commit
  `0588c2d215a955a516378677a92db2a5666802f3`, the same pin the constants carry.
- **Fixture:** `src/lean_spec/spec/crypto/xmss/poseidon.py`, the `PoseidonXmss`
  singleton `POSEIDON`.
- **The exact call each value came from:** `POSEIDON.compress(elements, width,
  output_length)`, `POSEIDON.sponge(elements, capacity, output_length, width)`
  and `POSEIDON.safe_domain_separator(lengths, capacity_length)`, with
  `elements` the field elements `operand_elements` below returns.
- **Reproducing it:** leanSpec's README asks for a Rust nightly toolchain, which
  the signature layer does not need. `lean_multisig_py` is reached from one
  place, `spec/crypto/xmss/__init__.py`, which imports `setup_prover` and calls
  it at import time; nothing under `poseidon.py` touches it. So, with Python
  3.12 and `pydantic numpy numba` (numba is required — `spec/crypto/poseidon.py`
  imports `njit` at module scope), install a stub answering any attribute with a
  no-op that *returns* rather than raises, before the first `lean_spec` import:

  ```python
  class _Stub(types.ModuleType):
      def __getattr__(self, name):
          return lambda *a, **k: None

  sys.modules["lean_multisig_py"] = _Stub("lean_multisig_py")
  ```

## Why the inputs are a rule and only the outputs are transcribed

Upstream pins no call here, so the inputs are this repo's to choose rather than
values to diff against a file — and one of these cases is 375 field elements
long. Generating them from a stated arithmetic rule keeps the file readable and
costs nothing: a wrong rule cannot pass, because the outputs beside it were
produced by feeding upstream exactly what `operand_elements` returns.

## What the cases vary

Both modes are placement, so the cases move what placement depends on. For
compression: both widths, an input that fills the state and one that is
zero-extended, an output truncated to one element and one not truncated at all,
and the four shapes the scheme actually hashes — the chain step, the Merkle
interior node, the message hash and the separator itself. For the sponge: the
chunk count (one, three, twenty-five), whether padding is needed at all, a
capacity that is the real separator rather than filler, and an output longer
than the rate, which is the only thing that makes the squeeze permute twice.
That last case is deliberately outside what leanSig asks for — every call the
scheme makes squeezes 8 elements from a rate of 15 — so the loop is gated rather
than assumed.
"""

from __future__ import annotations

from typing import Final, NamedTuple

PRIME: Final = 2**31 - 2**24 + 1
"""The KoalaBear prime, the modulus `operand_elements` reduces into."""

_STEP: Final = 2654435761
"""Knuth's multiplicative constant, used only to make consecutive elements
differ in every lane rather than by one — a ramp of `0, 1, 2, ...` would leave a
misplaced lane looking almost right."""


def operand_elements(length: int, seed: int) -> tuple[int, ...]:
    """The `length` canonical residues a case feeds in, in leanSpec's order."""
    return tuple((seed + index * _STEP) % PRIME for index in range(length))


class CompressionVector(NamedTuple):
    """One `compress` call and what upstream returns from it."""

    name: str
    width: int
    """Permutation width — 16 for the chain hash, 24 for the rest."""

    input_seed: int
    """Seeds `operand_elements`; the inputs are not transcribed."""

    input_length: int
    output_length: int
    output: tuple[int, ...]
    """Upstream's digest, in leanSpec's lane order."""


class SpongeVector(NamedTuple):
    """One `sponge` call and what upstream returns from it."""

    name: str
    width: int
    input_seed: int
    input_length: int
    output_length: int

    capacity: tuple[int, ...]
    """The capacity value, transcribed: for one case it is a real
    `safe_domain_separator` output rather than filler, and the record says so
    without the reader having to recompute it."""

    output: tuple[int, ...]


class DomainSeparatorVector(NamedTuple):
    """One `safe_domain_separator` call and what upstream returns from it."""

    name: str
    lengths: tuple[int, ...]
    """The hashing task's shape, packed into 32-bit slots upstream."""

    capacity_length: int
    output: tuple[int, ...]


COMPRESSION_VECTORS: Final = (
    CompressionVector(
        # the chain step: `(digest, parameter, tweak)` at width 16.
        name="width16_chain_step",
        width=16,
        input_seed=101,
        input_length=15,
        output_length=8,
        output=(
            2010388165,
            1292799716,
            590497074,
            334948910,
            836360066,
            862317580,
            1283844646,
            2027828095,
        ),
    ),
    CompressionVector(
        # operands fill the state, so nothing is zero-extended.
        name="width16_unpadded",
        width=16,
        input_seed=202,
        input_length=16,
        output_length=8,
        output=(
            1998948661,
            109494619,
            7112756,
            2000685465,
            1755141956,
            1569286983,
            652932013,
            1126274608,
        ),
    ),
    CompressionVector(
        # the whole fed-forward state, with no truncation.
        name="width16_full_output",
        width=16,
        input_seed=303,
        input_length=16,
        output_length=16,
        output=(
            305008733,
            1227063818,
            324728228,
            1297953486,
            613472259,
            10776938,
            283279325,
            1727183638,
            1169998717,
            1788544160,
            306350051,
            177493662,
            597423825,
            1482116340,
            1748008547,
            695315301,
        ),
    ),
    CompressionVector(
        # the Merkle interior node: `(parameter, tweak, left, right)`.
        name="width24_tree_node",
        width=24,
        input_seed=404,
        input_length=23,
        output_length=8,
        output=(
            1140225411,
            180158888,
            631087090,
            695736047,
            1255872440,
            1304903432,
            913926628,
            18422388,
        ),
    ),
    CompressionVector(
        # the message hash, whose output length is `ceil(DIMENSION / Z)`.
        name="width24_message_hash",
        width=24,
        input_seed=505,
        input_length=23,
        output_length=6,
        output=(
            1234846392,
            2066645032,
            1704342393,
            723022538,
            1205921389,
            914016083,
        ),
    ),
    CompressionVector(
        # the separator's own shape — a full state in, `CAPACITY` out.
        name="width24_domain_separator_shape",
        width=24,
        input_seed=606,
        input_length=24,
        output_length=9,
        output=(
            97502824,
            323672615,
            1885062767,
            2009007721,
            945625266,
            1271267359,
            199530193,
            1501450861,
            2030642353,
        ),
    ),
    CompressionVector(
        # the shortest output there is, over a heavily padded input.
        name="width24_single_output",
        width=24,
        input_seed=707,
        input_length=10,
        output_length=1,
        output=(2089720721,),
    ),
)

SPONGE_VECTORS: Final = (
    SpongeVector(
        # a `TEST_CONFIG` leaf: 39 elements pad up to three chunks.
        name="width24_leaf_test_config",
        width=24,
        input_seed=811,
        input_length=39,
        output_length=8,
        capacity=(
            812,
            523730140,
            1047459468,
            1571188796,
            2094918124,
            487941019,
            1011670347,
            1535399675,
            2059129003,
        ),
        output=(
            110513888,
            2040620640,
            873477884,
            1720078832,
            1522889650,
            1203450028,
            777232356,
            1838653562,
        ),
    ),
    SpongeVector(
        # a `PROD_CONFIG` leaf: 375 elements are exactly 25 chunks, and the
        # capacity is a real separator rather than filler.
        name="width24_leaf_prod_config",
        width=24,
        input_seed=821,
        input_length=375,
        output_length=8,
        capacity=(
            2060061975,
            916902315,
            229801915,
            83751504,
            2093549181,
            1743125625,
            721042244,
            1252069948,
            1192880636,
        ),
        output=(
            1758874521,
            953792851,
            261502580,
            840960768,
            1346418362,
            345649828,
            1611028070,
            500464788,
        ),
    ),
    SpongeVector(
        # one chunk exactly, so absorption never loops.
        name="width24_single_chunk",
        width=24,
        input_seed=831,
        input_length=15,
        output_length=8,
        capacity=(
            832,
            523730160,
            1047459488,
            1571188816,
            2094918144,
            487941039,
            1011670367,
            1535399695,
            2059129023,
        ),
        output=(
            825009561,
            2090494872,
            1193543919,
            1868565388,
            187377862,
            2014736457,
            1441756112,
            1948061776,
        ),
    ),
    SpongeVector(
        # one element in a chunk of 15 — maximal padding.
        name="width24_short_input",
        width=24,
        input_seed=841,
        input_length=1,
        output_length=8,
        capacity=(
            842,
            523730170,
            1047459498,
            1571188826,
            2094918154,
            487941049,
            1011670377,
            1535399705,
            2059129033,
        ),
        output=(
            726476620,
            232579162,
            1042514888,
            1498239792,
            1031128691,
            1632178479,
            26788488,
            1536541088,
        ),
    ),
    SpongeVector(
        # output longer than the rate, so the squeeze permutes again.
        name="width24_multi_squeeze",
        width=24,
        input_seed=851,
        input_length=20,
        output_length=20,
        capacity=(
            852,
            523730180,
            1047459508,
            1571188836,
            2094918164,
            487941059,
            1011670387,
            1535399715,
            2059129043,
        ),
        output=(
            2096842900,
            1532423443,
            149750614,
            1640337215,
            1974831146,
            1924324319,
            1674245483,
            1215903751,
            1003274797,
            811462756,
            1733734323,
            238576064,
            54985095,
            949896169,
            911456698,
            233602649,
            1926214680,
            864083686,
            1604445906,
            1643024995,
        ),
    ),
    SpongeVector(
        # the other width, and a rate the scheme itself never asks for.
        name="width16_narrow_capacity",
        width=16,
        input_seed=861,
        input_length=20,
        output_length=8,
        capacity=(
            862,
            523730190,
            1047459518,
            1571188846,
        ),
        output=(
            407165056,
            683927401,
            345038385,
            592121916,
            692990108,
            927679052,
            79063276,
            1275268030,
        ),
    ),
)

DOMAIN_SEPARATOR_VECTORS: Final = (
    DomainSeparatorVector(
        # `PROD_CONFIG`'s shape — `DIMENSION` 46.
        name="prod_config",
        lengths=(5, 2, 46, 8),
        capacity_length=9,
        output=(
            2060061975,
            916902315,
            229801915,
            83751504,
            2093549181,
            1743125625,
            721042244,
            1252069948,
            1192880636,
        ),
    ),
    DomainSeparatorVector(
        # `TEST_CONFIG`'s shape — `DIMENSION` 4.
        name="test_config",
        lengths=(5, 2, 4, 8),
        capacity_length=9,
        output=(
            627826400,
            1244476188,
            370678638,
            978729783,
            1996000804,
            1380088873,
            1753334201,
            433326939,
            1294775677,
        ),
    ),
)
