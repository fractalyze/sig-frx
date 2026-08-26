# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""What leanSpec computes for the message-to-codeword pipeline.

Nothing upstream publishes pins a call in this layer — the fixtures archive
carries permutation vectors, SSZ container vectors and `PROD_CONFIG` keys, and
its signature-shaped families are leanMultisig aggregate proofs rather than XMSS
signatures. So the reference implementation is the authority, third in the order
[`testing.md`](../../../../docs/reference/testing.md) fixes, and gating on one
costs the provenance below.

## Provenance

- **Upstream:** `leanEthereum/leanSpec` at commit
  `0588c2d215a955a516378677a92db2a5666802f3`, the same pin the constants carry.
- **Fixture:** `src/lean_spec/spec/crypto/xmss/encoding.py`, against the
  `PoseidonXmss` singleton `POSEIDON` and the `PROD_CONFIG` / `TEST_CONFIG`
  presets from `constants.py`.
- **The exact call each value came from:** `encode_message(config, message)`,
  `encode_epoch(config, epoch)`, `aborting_decode(config, elements)`,
  `message_hash(POSEIDON, config, parameter, epoch, rho, message)` and
  `target_sum_encode(...)` on the same arguments.
- **Reproducing it:** the environment recipe is
  [`mode_vectors.py`](mode_vectors.py)'s — a Python 3.12 venv with
  `pydantic numpy numba`, and a stub `lean_multisig_py` installed before the
  first `lean_spec` import. Nothing further is needed here: `encoding.py` reaches
  no deeper into the tree than `poseidon.py` already did.

## Why the inputs are rules and only the outputs are transcribed

Upstream pins no call here, so the inputs are this repo's to choose. The public
parameter and the randomness come from
[`mode_vectors.operand_elements`](mode_vectors.py) rather than from a second rule
written to look like it — one rule, so two vector sets cannot drift while both
claim to be the rule. Messages are written as the integer their 32 bytes read to
little-endian, because the interesting ones are boundaries (`0`, `1`,
`2**256 - 1`, a lone high byte) and a boundary is clearer as arithmetic than as a
hex blob.

**A codeword is a string, one character per digit.** `BASE` is 8 at both presets
and `DIMENSION` is 46 at the larger one, so a whole codeword is 46 octal
characters on one line — where the tuple of small integers the sibling modules
use would be 46 lines, and forty-six lines each reading `4,` are not more
readable than one line you can count. The digits are compared as integers, so
this encoding holds for any preset with `BASE <= 10` and a wider alphabet would
need another.

## What the cases vary

- **`encode_message`** moves the limb count that has to carry: nothing, one limb,
  a lone high byte, a full-width root, and the widest a 32-byte value reaches —
  the last being what fails if the little-endian read or the nine-limb bound is
  wrong.
- **`encode_epoch`** moves the slot across the byte boundary the 8-bit prefix
  shifts it past, up to `2^32 - 1`, the last slot a `LOG_LIFETIME = 32` key has.
- **`aborting_decode`** moves what the decode is sensitive to: the all-zero
  vector, the largest value that still has a quotient (`PRIME - 2`, every digit
  `BASE - 1`), a spread whose digits differ per element and per position, and the
  abort in the first element and in the last — a rejection that checked one end
  only would pass the other. `PROD` also drops two digits off the end of 48,
  where `TEST` decodes a single element rather than six.
- **`message_hash` and `target_sum_encode`** move the acceptance: each preset
  gets one attempt on the target layer and one off it, since the flag is all the
  second filter contributes and a filter that never rejects passes a
  positive-only suite. The remaining cases move the slot to the last one a key
  has and the message to zero.

**No case aborts through `message_hash`.** The abort is one value out of `PRIME`,
about 4.7e-10 per element, so it is reachable by construction and not by search —
which is why `DECODE_VECTORS` carries it and the pipeline cases cannot.
"""

from __future__ import annotations

from typing import Final, NamedTuple


class MessageVector(NamedTuple):
    """One `encode_message` call and the limbs upstream returns."""

    name: str
    message: int
    """The 32-byte root, as the integer its bytes read to little-endian."""

    elements: tuple[int, ...]
    """Upstream's limbs, least significant first — leanSpec's own order."""


class EpochVector(NamedTuple):
    """One `encode_epoch` call and the limbs upstream returns."""

    name: str
    epoch: int
    elements: tuple[int, ...]


class DecodeVector(NamedTuple):
    """One `aborting_decode` call and what upstream returns from it."""

    name: str
    preset: str
    """Which parameter row the case runs at — `prod` or `test`."""

    elements: tuple[int, ...]
    """The field elements fed in, in leanSpec's lane order."""

    digits: str | None
    """The codeword, one character per digit, or `None` where upstream aborted."""


class MessageHashVector(NamedTuple):
    """One `message_hash` and `target_sum_encode` pair, at one preset."""

    name: str
    preset: str
    message: int
    parameter_seed: int
    """Seeds `operand_elements` for the public parameter."""

    epoch: int
    randomness_seed: int
    """Seeds `operand_elements` for `rho`."""

    on_layer: bool
    """Whether `target_sum_encode` accepted — the digits sum to `TARGET_SUM`."""

    digits: str
    """What `message_hash` decoded to, accepted or not."""


MESSAGE_VECTORS: Final = (
    MessageVector(name="zero", message=0, elements=(0,) * 9),
    MessageVector(name="one", message=1, elements=(1,) + (0,) * 8),
    MessageVector(
        # A lone high byte. Only the top of the value is fed, so every limb below
        # the ninth is a remainder of the divisions above it rather than a slice
        # of the input — which a decomposition that shifted instead would miss.
        name="high_byte_only",
        message=1 << 248,
        elements=(
            1395966798,
            296233900,
            1247831268,
            1013124842,
            2088077358,
            486397775,
            1873995542,
            137975460,
            1,
        ),
    ),
    MessageVector(
        name="spread",
        message=0x9A8D807366594C3F3225180BFEF1E4D7CABDB0A396897C6F6255483B2E211407,
        elements=(
            832268087,
            1561209498,
            202095011,
            909367739,
            2068598144,
            127925761,
            2031031605,
            1195150718,
            164,
        ),
    ),
    MessageVector(
        # The widest a 32-byte root reaches. Its ninth limb is 272, so nine is the
        # count the preset needs and eight would reject rather than truncate.
        name="all_ones",
        message=2**256 - 1,
        elements=(
            1539525976,
            1261153412,
            1969546126,
            1544481308,
            1871195519,
            936857536,
            333911385,
            1230415057,
            272,
        ),
    ),
)

EPOCH_VECTORS: Final = (
    # Slot zero is the prefix alone, which is the whole point of carrying one.
    EpochVector(name="zero", epoch=0, elements=(2, 0)),
    EpochVector(name="one", epoch=1, elements=(258, 0)),
    EpochVector(name="mid", epoch=0x12345678, elements=(1482061790, 36)),
    # The last slot a `LOG_LIFETIME = 32` key covers.
    EpochVector(name="max_slot", epoch=2**32 - 1, elements=(67108094, 516)),
)

DECODE_VECTORS: Final = (
    DecodeVector(
        name="prod_zero",
        preset="prod",
        elements=(0,) * 6,
        digits="0" * 46,
    ),
    DecodeVector(
        # `PRIME - 2`, the largest value that still has a quotient: dividing out
        # `Q` leaves `BASE^Z - 1`, so every digit is `BASE - 1`.
        name="prod_max_quotient",
        preset="prod",
        elements=(2130706431,) * 6,
        digits="7" * 46,
    ),
    DecodeVector(
        name="prod_spread",
        preset="prod",
        elements=(4242, 523733570, 1047462898, 1571192226, 2094921554, 487944449),
        digits="1400000065366571376453730123417552513767240025",
    ),
    DecodeVector(
        name="prod_abort_first",
        preset="prod",
        elements=(2130706432, 7, 523729335, 1047458663, 1571187991, 2094917319),
        digits=None,
    ),
    DecodeVector(
        name="prod_abort_last",
        preset="prod",
        elements=(9, 523729337, 1047458665, 1571187993, 2094917321, 2130706432),
        digits=None,
    ),
    DecodeVector(
        name="test_spread",
        preset="test",
        elements=(31337,),
        digits="6630",
    ),
    DecodeVector(
        name="test_abort",
        preset="test",
        elements=(2130706432,),
        digits=None,
    ),
)

MESSAGE_HASH_VECTORS: Final = (
    MessageHashVector(
        name="prod_on_layer",
        preset="prod",
        message=11,
        parameter_seed=101,
        epoch=7,
        randomness_seed=3040,
        on_layer=True,
        digits="5456717445735620677010446644646521426567433655",
    ),
    MessageHashVector(
        name="prod_off_layer",
        preset="prod",
        message=11,
        parameter_seed=101,
        epoch=7,
        randomness_seed=1,
        on_layer=False,
        digits="3210564226025532334736740112733575552042031341",
    ),
    MessageHashVector(
        # The last slot and a root of all ones: the two widest host integers the
        # pipeline decomposes, in one case.
        name="prod_max_slot",
        preset="prod",
        message=2**256 - 1,
        parameter_seed=555,
        epoch=2**32 - 1,
        randomness_seed=13,
        on_layer=False,
        digits="1036242444216610656334214003304117550737313100",
    ),
    MessageHashVector(
        name="test_on_layer",
        preset="test",
        message=11,
        parameter_seed=101,
        epoch=7,
        randomness_seed=1,
        on_layer=True,
        digits="3210",
    ),
    MessageHashVector(
        name="test_off_layer",
        preset="test",
        message=11,
        parameter_seed=101,
        epoch=7,
        randomness_seed=2,
        on_layer=False,
        digits="4701",
    ),
    MessageHashVector(
        name="test_zero_epoch",
        preset="test",
        message=0,
        parameter_seed=777,
        epoch=0,
        randomness_seed=21,
        on_layer=False,
        digits="1607",
    ),
)
