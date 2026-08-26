# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""What a recorded case has to become before the seam will take it.

`harness` after `leansig/testing/harness.py`, which is the same thing for another
scheme: the vocabulary a package's suites all need, written once rather than per
suite. Not to be confused with `sig_frx/testing/kat.py`, the one driver the
known-answer tests run every scheme through — `stateless.py` says outright that
the shared harness does not drive it, and this is not that harness.

Two jobs, both of which every SHRINCS test file was doing for itself.

**A batch is one shape per file.** Verification compiles per input shape, so a
second batch width costs another whole compile — and SHRINCS compiles both paths
for every entry, which makes that expensive rather than merely wasteful. So a
file settles on one width and pads up to it, and `verdicts` is that: it takes as
many rows as a test has something to say about, repeats the last one to fill the
shape, and hands back only the verdicts that were asked for.

**A recorded signature is not a component's input.** It is
`indicator ‖ R ‖ leaf index ‖ FXMSS signature`, and a test driving anything below
the seam has to skip that header. `fxmss_body` is that slice, taken at
`shrincs.INDEX_FIELD_START` rather than at 17 — a header written out in four test
modules and derived in one implementation is four places for it to stop agreeing.

This is deliberately not in `stateful_vectors`: the vectors are transcribed
reference data and depend on nothing, so that a case cannot quietly follow the
implementation it exists to hold in place. Here the dependency is fine, because
what lives here is how a *test* uses a case rather than what the case says.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from frx import Array

from sig_frx.hash.shrincs import shrincs
from sig_frx.hash.shrincs.testing import stateful_vectors

# The one batch shape the SHRINCS files compile. Four, because that is the widest
# case any of them needs — a batch whose verdicts alternate, which is what shows
# that a verdict has not smeared along the axis in either direction.
BATCH = 4

# A `(public key, message, signature)` row, as the bytes a case records them.
Row = tuple[bytes, bytes, bytes]


def fxmss_body(case: stateful_vectors.StatefulVectors) -> bytes:
    """The FXMSS signature inside a recorded one — what the walk is handed."""
    return case.signature[shrincs.INDEX_FIELD_START + case.leaf_index_size :]


def rows(*values: bytes) -> np.ndarray:
    """A batch axis over byte strings of one length."""
    # Stacked rather than joined and reshaped, so that a batch of empty
    # signatures is `[B, 0]` instead of a reshape of nothing.
    return np.stack([np.frombuffer(value, dtype=np.uint8) for value in values])


def context(value: bytes) -> np.ndarray | None:
    """The context as the seam takes it: a `uint8` array, `None` meaning empty."""
    return np.frombuffer(value, dtype=np.uint8) if value else None


def verdicts(
    verify: Callable[..., Array],
    batch: Sequence[Row],
    ctx: bytes,
) -> list[bool]:
    """Verify `batch` in one call at `BATCH`, and return its verdicts.

    Padded by repeating the last row, so the shape holds however many rows a test
    has something to say about. Rows in one call share a context, which is one per
    call, and a message length, which is one per batch — both because that is what
    the seam takes, not a convenience here.
    """
    if not 1 <= len(batch) <= BATCH:
        raise ValueError(f"1 to {BATCH} rows per call, got {len(batch)}")
    padded = list(batch) + [batch[-1]] * (BATCH - len(batch))
    got = verify(
        rows(*(row[0] for row in padded)),
        rows(*(row[1] for row in padded)),
        rows(*(row[2] for row in padded)),
        context=context(ctx),
    )
    return [bool(verdict) for verdict in np.asarray(got)][: len(batch)]
