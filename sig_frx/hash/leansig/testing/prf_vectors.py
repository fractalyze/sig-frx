# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""What leanSpec's PRF derives — chain starts and signing randomness.

A PRF has no internal structure to check itself against. Every byte of its two
input layouts is a convention — the domain separator, the subdomain byte, a
four-byte big-endian epoch, an eight-byte big-endian counter — and getting one
wrong yields output that is uniform, deterministic and a different scheme, which
nothing downstream can notice. So these are a transcription gate and nothing
else.

## Provenance

- **Upstream:** `leanEthereum/leanSpec` at commit
  `0588c2d215a955a516378677a92db2a5666802f3`, the same pin every other vector
  module here carries.
- **Fixture:** none. The fixtures archive publishes permutation vectors, SSZ
  container vectors and `PROD` keys, and nothing that pins a PRF call — so this
  is the reference implementation, third in the order
  [`testing.md`](../../../../docs/reference/testing.md) fixes.
- **The exact call each value came from:** `PRFKey(PRF_KEY)`, then
  `derive_chain_start(config, Uint64(epoch), Uint64(chain_index))` and
  `derive_randomness(config, Uint64(epoch), Bytes32(message), Uint64(counter))`.
- **Reproducing it:** the environment recipe is
  [`mode_vectors.py`](mode_vectors.py)'s — a Python 3.12 venv with
  `pydantic numpy numba`, and a stub `lean_multisig_py` installed before the
  first `lean_spec` import.

## What the cases vary

Each field of each layout moves, and both presets appear. The preset changes
only how many elements come back — `hash_length` against `randomness_length` —
so a transposed pair of lengths would pass a single-preset set and fails here.
"""

from __future__ import annotations

from typing import Final, NamedTuple


class ChainStartVector(NamedTuple):
    """One `derive_chain_start` call and the digest it returned."""

    name: str
    preset: str
    epoch: int
    chain_index: int
    digest: tuple[int, ...]  # canonical residues, in leanSpec's order


class RandomnessVector(NamedTuple):
    """One `derive_randomness` call and the randomness it returned."""

    name: str
    preset: str
    epoch: int
    message: str
    counter: int
    randomness: tuple[int, ...]  # canonical residues, in leanSpec's order


PRF_KEY: Final = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
"""The 32-byte master seed these vectors substitute for `PRFKey.generate()`."""

CHAIN_STARTS: Final[tuple[ChainStartVector, ...]] = (
    ChainStartVector(
        name="test_slot0_chain0",
        preset="test",
        epoch=0,
        chain_index=0,
        digest=(
            1617529907,
            484813519,
            1391994054,
            2116114443,
            1310920627,
            1623946583,
            874772112,
            780793951,
        ),
    ),
    ChainStartVector(
        name="test_slot0_chain3",
        preset="test",
        epoch=0,
        chain_index=3,
        digest=(
            1181804392,
            303961055,
            1791492711,
            1394563089,
            1022454230,
            515554847,
            166830305,
            695034004,
        ),
    ),
    ChainStartVector(
        name="test_slot255_chain1",
        preset="test",
        epoch=255,
        chain_index=1,
        digest=(
            542509143,
            1014074692,
            165244559,
            1396042858,
            645161650,
            2048945041,
            1800675655,
            1299525103,
        ),
    ),
    ChainStartVector(
        name="prod_slot70000_chain45",
        preset="prod",
        epoch=70000,
        chain_index=45,
        digest=(
            62737184,
            449233438,
            1233146420,
            1490377847,
            55396672,
            1494982071,
            1676254900,
            1346742523,
        ),
    ),
)

RANDOMNESS: Final[tuple[RandomnessVector, ...]] = (
    RandomnessVector(
        name="test_slot0_counter0",
        preset="test",
        epoch=0,
        message="000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
        counter=0,
        randomness=(
            384553737,
            1042862801,
            1846185079,
            2069464078,
            839893860,
            1766408653,
            958233232,
        ),
    ),
    RandomnessVector(
        name="test_slot7_counter5",
        preset="test",
        epoch=7,
        message="202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f",
        counter=5,
        randomness=(
            103451931,
            868164020,
            313384547,
            1002052579,
            714215164,
            64972086,
            1199441478,
        ),
    ),
    RandomnessVector(
        name="prod_slot123_counter900",
        preset="prod",
        epoch=123,
        message="abababababababababababababababababababababababababababababababab",
        counter=900,
        randomness=(
            1756573879,
            168980694,
            1338840390,
            1151435937,
            550846055,
            944472583,
            159262829,
        ),
    ),
)
