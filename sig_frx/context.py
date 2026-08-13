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
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

# Both standards give the context one length byte — FIPS 204 Algorithm 2 line 1,
# FIPS 205 Algorithm 22 line 1.
MAX_SIZE = 255


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
