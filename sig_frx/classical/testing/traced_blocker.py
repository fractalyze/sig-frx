# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The one skip marker every traced classical case shares, and why it exists.

The traced path is blocked upstream, in two layers: the pinned zk_dtypes has
no curated dtypes for these curves' moduli, which frxlib requires by name
(exposure: fractalyze/zk_dtypes#174), and frxlib's wide-field Montgomery
multiply is wrong for full-width moduli (fractalyze/xla#542, fix in
fractalyze/prime-ir#434).

The condition probes the capability the gated code actually needs — can a
traced multiply over one of these moduli run at all — rather than any one
upstream's symbol, so whichever pin moves first un-skips the gate. Running is
the bar on purpose: a traced multiply that runs but is *wrong* must not skip,
because at that point the un-skipped cases are gating exactly the upstream
value fix they waited for.
"""

from __future__ import annotations

import unittest
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
import zk_dtypes

from sig_frx.classical import weierstrass


def _traced_field_mul_runs() -> bool:
    field = zk_dtypes.prime_field(weierstrass.SECP256K1.p)
    operand = np.array([3], dtype=field)
    try:
        result = frx.jit(lambda value: value * value)(fnp.asarray(operand))
        # Materialize: the refusal surfaces lazily, at the device-to-host
        # transfer of a field array, not at compile time.
        np.asarray(result)
    except Exception:  # noqa: BLE001 — any refusal means the path is blocked
        return False
    return True


TRACED_BLOCKED: Any = unittest.skipIf(
    not _traced_field_mul_runs(),
    "traced curve arithmetic blocked on fractalyze/zk_dtypes#174 and"
    " fractalyze/xla#542",
)
