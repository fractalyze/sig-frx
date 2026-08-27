# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The application context, as both FIPS signature standards wrap a message in it.

FIPS 204 §5.2 and FIPS 205 §10.2 prepare the message the same way: a
domain-separator byte, the context's length in one byte, then the context itself,
all ahead of the message. The two standards choose the domain byte independently
— that value stays with the scheme that reads its standard — but the framing and
its one-byte length field are common, and so is what they are for. The separator
exists so that a signature over a pre-hashed message cannot verify as one over the
message, and the length byte is what makes the concatenation unambiguous.

Shared rather than copied because a second scheme arrived asking for the same
shape ([`docs/reference/conventions.md`](../docs/reference/conventions.md)); the
cost of two copies is not the lines but that a fix to one is never looked for in
the other.

The module also owns the **refusals**, which is what widened it past the framing
its name is for. A seam field a scheme cannot use has exactly one honest value,
and saying so is one predicate however many fields there are: `context` for a
standard that defines none, `position` for a scheme whose signature carries its
own index or has no notion of one ([`signature.py`](signature.py)). They arrived
one apiece and the second is what made the shared `_refuse` below, per
[`conventions.md`](../docs/reference/conventions.md#generalize-a-component-when-its-second-consumer-arrives)
— a refinement to either (a traced value, an empty of the wrong rank) now lands
once rather than in whichever copy the reader happened to open.

A third arrived since, from a different decade and a different body: RFC 8032's
`dom2(F, C)` is this exact framing — `octet(F) ‖ octet(OLEN(C)) ‖ C`, one length
byte and the same 255-octet ceiling — behind a 32-octet ASCII constant that the
EdDSA variants prepend and the FIPS schemes have no analogue of
([`classical/eddsa/ed25519.py`](classical/eddsa/ed25519.py)). So `prefix` is
what all three build, and only the constant ahead of it is any scheme's own.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

# Both standards give the context one length byte — FIPS 204 Algorithm 2 line 1,
# FIPS 205 Algorithm 22 line 1.
MAX_SIZE = 255


def _refuse(value: ArrayLike | None, scheme: str, cannot: str) -> None:
    """Reject a non-empty `value` for a scheme that has no use for it.

    `np.size` rather than `np.asarray(...).size`: only the shape is wanted, and
    a value that arrived on a device would otherwise be copied back to read it.
    """
    if value is None:
        return
    if np.size(value) != 0:
        raise ValueError(f"{scheme} {cannot}; pass None or empty")


def require_empty(context: ArrayLike | None, scheme: str) -> None:
    """Refuse a context for a scheme whose standard defines none.

    The seam's rule: accepting a context only to ignore it would verify
    something other than what the caller asked about, so empty is the only
    honest value (`signature.py`). Shared because every classical scheme
    enforces the same sentence.
    """
    _refuse(context, scheme, "defines no application context")


def require_no_position(position: ArrayLike | None, scheme: str) -> None:
    """Refuse a per-entry position for a scheme that verifies without one.

    Most schemes take everything they need from the three operands: the position
    is either inside the signature (RFC 8391 XMSS encodes the index) or absent
    from the construction entirely. leanSig is the exception the seam field
    exists for, and it is the one that reads it (`signature.py`).

    The same sentence `require_empty` enforces, about the seam's other per-call
    field — accepting one to ignore it would verify at a position other than the
    one the caller named.
    """
    _refuse(position, scheme, "verifies without a per-entry position")


def prefix(domain: int, context: ArrayLike | None) -> np.ndarray:
    """`toByte(domain, 1) ‖ toByte(|ctx|, 1) ‖ ctx`, `None` meaning empty.

    A context longer than one byte can encode is refused rather than truncated:
    truncating would sign a different context from the one the caller named, and
    the signature would verify against the truncation.

    A host value, because it is built from a scheme's constant and the caller's
    own bytes and is consumed by a concatenation that broadcasts it.
    """
    ctx = (
        np.zeros(0, dtype=np.uint8)
        if context is None
        else np.asarray(context, dtype=np.uint8).reshape(-1)
    )
    if ctx.shape[0] > MAX_SIZE:
        raise ValueError(
            f"a context string is at most {MAX_SIZE} bytes, got {ctx.shape[0]}"
        )
    return np.concatenate([np.array([domain, ctx.shape[0]], dtype=np.uint8), ctx])


def prepend(head: np.ndarray, messages: ArrayLike) -> Array:
    """`head ‖ M`, for one message or for a whole batch of them.

    The prefix is one value per call — the domain separator and the context, which
    a verifier serves one of at a time — so a batch broadcasts it rather than
    carrying a copy per entry.
    """
    values = fnp.asarray(head, dtype=fnp.uint8)
    body = fnp.asarray(messages, dtype=fnp.uint8)
    if body.ndim == 1:
        return fnp.concatenate([values, body])
    return fnp.concatenate(
        [fnp.broadcast_to(values, (body.shape[0], values.shape[0])), body], axis=-1
    )
