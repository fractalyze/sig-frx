# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The one skip marker every traced classical case shares, and why it exists.

The traced path is blocked upstream, in two layers: the pinned zk_dtypes has
no curated dtypes for these curves' moduli, which frxlib requires by name
(exposure: fractalyze/zk_dtypes#174), and frxlib's wide-field Montgomery
multiply is wrong for full-width moduli (fractalyze/xla#542). The condition
reads the exposure off zk_dtypes, so the pin bump that carries the fixes
un-skips every gated case by itself — and any case that then fails is gating
exactly the upstream fix it waited for.
"""

from __future__ import annotations

import unittest
from typing import Any

import zk_dtypes

TRACED_BLOCKED: Any = unittest.skipIf(
    not hasattr(zk_dtypes, "secp256k1_bf"),
    "traced curve arithmetic blocked on fractalyze/zk_dtypes#174 and"
    " fractalyze/xla#542",
)
