# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Which array namespace a value belongs to.

Every scheme here has a concrete side and a traced one — key generation and
signing run on the host, verification runs under a tracer — and the rule that
keeps them apart is that a value is used in the namespace it arrives in
([`conventions.md`](../docs/reference/conventions.md)). This module is that rule
as code, in the one place both the hash-based and the lattice packages can reach
it.
"""

from __future__ import annotations

from typing import Any

import frx.numpy as fnp
import numpy as np
from frx import Array


def namespace(*values: object) -> Any:
    """The array namespace `values` belong to: `frx.numpy` if any is traced.

    Host values stay on numpy rather than being lifted onto a device, which is
    what keeps a concrete path — one signature, one key pair — from paying a
    dispatch per operation. The lift is the conversion that needs a rule, because
    it succeeds everywhere except on the path that cannot afford it.
    """
    return fnp if any(isinstance(value, Array) for value in values) else np
