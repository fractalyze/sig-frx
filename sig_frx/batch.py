# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The shape a `verify` demands of its batch, and what a wrong width means.

Every scheme's `verify` opens the same way: three operands read as `uint8`, a
rank check on each, a check that they line up on the batch axis, and then a
decision about a width that is not the parameter set's. Six schemes wrote that
out — ML-DSA, Falcon, SLH-DSA, SHRINCS, its stateless component and XMSS — and
the copies had drifted in the one place it matters, which is what a verifier
*returns*.

What every copy agrees on: a wrong **rank**, or a batch whose parts do not line
up, is a caller that built the batch wrongly, so it raises. A batch carries one
static width, so a wrong **width** is every entry's answer at once rather than
one entry's.

What they disagree on is whether that answer is `False` or an exception, and the
disagreement follows the standards rather than being an accident:

- FIPS 204 §3.6.2 requires a public key *or* a signature of the wrong length to
  verify as false rather than to raise. ML-DSA and Falcon read it that way for
  both operands.
- FIPS 205 Algorithm 20 lines 1 to 3 say it of the *signature* and say nothing
  about the public key, so SLH-DSA, SHRINCS and `stateless` take a mis-sized
  signature as a verdict and a mis-sized key as a caller mistake.
- RFC 8391 defines no such verdict at all, which is why XMSS takes both as
  caller mistakes.

This module also holds the refusal for the seam's other per-entry operand.
`position` is the slot a stateful scheme verifies at, and exactly one scheme
reads it — so the rule for every other one is the same sentence, and
`require_no_position` is it. Accepting a position only to ignore it would verify
at a slot other than the one the caller named, which is `context`'s failure
under a different name ([`context.py`](context.py)).

So the reading is per operand and per scheme, and `WrongWidth` is how a call site
states the one its standard gave it. `VERDICT` is the default because it is what
both FIPS standards ask for wherever they speak; a scheme whose standard is
silent passes `ERROR` and is on the record for it. That is the part six
undeclared copies could not do — the divergence existed, but nothing said so, so
there was no way to tell a decision from a drift.
"""

from __future__ import annotations

import enum
from typing import NamedTuple

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike


class WrongWidth(enum.Enum):
    """What an operand whose width is not the parameter set's means."""

    # Every entry is `False`: the standard defines the rejection, so a verifier
    # answers it rather than refusing to answer.
    VERDICT = enum.auto()
    # A `ValueError`: the standard defines no such verdict, so a batch that
    # carries one is a caller mistake and not a question about a signature.
    ERROR = enum.auto()


class Batch(NamedTuple):
    """A `verify`'s three operands, read as `uint8`, and the shape they agree on."""

    public_key: Array
    message: Array
    signature: Array
    # The batch axis every operand shares.
    size: int
    # Whether both widths are the parameter set's. Always true where a scheme
    # asked for `ERROR` on both, which is why XMSS does not read it.
    well_formed: bool


def require_batch(
    public_key: ArrayLike,
    message: ArrayLike,
    signature: ArrayLike,
    *,
    public_key_size: int,
    signature_size: int,
    public_key_width: WrongWidth = WrongWidth.VERDICT,
    signature_width: WrongWidth = WrongWidth.VERDICT,
) -> Batch:
    """The batch preamble every `verify` shares — see the module docstring.

    `signature_size` is the width a signature is required to have, which for a
    compressed scheme is `signature_max_size` rather than any one signature's
    own length: the seam pads to that (`signature.py`), so it is static here
    like the others.

    The message carries no width of its own. `L` is the caller's and the seam
    only requires it static, so the only thing to check is that there is one
    message per key — which is also what catches a bare `[L]` message passed to
    a `B = 1` call, the mistake that would otherwise be read as a batch of its
    own bytes.
    """
    keys = fnp.asarray(public_key, dtype=fnp.uint8)
    if keys.ndim != 2 or (
        public_key_width is WrongWidth.ERROR and keys.shape[1] != public_key_size
    ):
        raise ValueError(
            f"a public key batch is [B, {public_key_size}], got shape "
            f"{tuple(keys.shape)}"
        )
    size = keys.shape[0]

    signatures = fnp.asarray(signature, dtype=fnp.uint8)
    if signatures.ndim != 2 or signatures.shape[0] != size:
        raise ValueError(
            f"one signature per public key, as a [B, {signature_size}] batch: "
            f"got {size} keys and signatures of shape {tuple(signatures.shape)}"
        )
    if signature_width is WrongWidth.ERROR and signatures.shape[1] != signature_size:
        raise ValueError(
            f"a signature batch is [B, {signature_size}], got shape "
            f"{tuple(signatures.shape)}"
        )

    messages = fnp.asarray(message, dtype=fnp.uint8)
    if messages.ndim != 2 or messages.shape[0] != size:
        raise ValueError(
            f"one message per public key, as a [B, L] batch: got {size} keys "
            f"and messages of shape {tuple(messages.shape)}"
        )

    return Batch(
        keys,
        messages,
        signatures,
        size,
        keys.shape[1] == public_key_size and signatures.shape[1] == signature_size,
    )


def require_no_position(position: ArrayLike | None, scheme: str) -> None:
    """Refuse a per-entry position for a scheme that verifies without one.

    Most schemes take everything they need from the three operands: the position
    is either inside the signature (RFC 8391 XMSS encodes the index) or absent
    from the construction entirely. leanSig is the exception the seam field
    exists for, and it is the one that reads it (`signature.py`).

    Empty is the only honest value for the rest, by the seam's own rule for a
    field a scheme cannot use — the same sentence `context.require_empty`
    enforces, and shared for the same reason: every scheme that refuses one
    refuses it identically.
    """
    if position is None:
        return
    if np.asarray(position).size != 0:
        raise ValueError(
            f"{scheme} verifies without a per-entry position; pass None or empty"
        )
