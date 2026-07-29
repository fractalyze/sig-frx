# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""hash-frx is reachable from this repo, and computes.

Every scheme here hashes through hash-frx, and the wiring is the part most likely
to be subtly wrong: the module pin, the second pip hub it brings, and the frx
copy each hub resolves. A scheme test would catch a break too — months later and
behind its own failure. This guard fails loudly instead: it runs one real
hash-frx computation under `jit`, and takes the seam a scheme takes.

The linear helpers are the computation because they need no parameter fixture,
which keeps this a wiring test rather than a hash test.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import hash_frx
import numpy as np
from absl.testing import absltest
from frx import Array
from frx.typing import ArrayLike
from hash_frx.byte_hash import ByteHash
from hash_frx.linear import apply_matrix
from zk_dtypes import koalabear_mont as F


class _ZeroHash:
    """The smallest thing that satisfies `ByteHash` — the shape a scheme takes."""

    digest_size = 32
    has_dedicated_fusion = False

    def digest(self, msg: ArrayLike) -> Array | np.ndarray:
        return np.zeros((np.shape(msg)[0], self.digest_size), dtype=np.uint8)


class HashFrxDependencyTest(absltest.TestCase):
    def test_package_carries_a_version(self) -> None:
        self.assertTrue(getattr(hash_frx, "__version__", ""))

    def test_computes_through_hash_frx_under_jit(self) -> None:
        w = 4
        eye = fnp.array([[int(i == j) for j in range(w)] for i in range(w)], dtype=F)
        state = fnp.array([1, 2, 3, 4], dtype=F)
        self.assertTrue(bool(fnp.array_equal(frx.jit(apply_matrix)(eye, state), state)))

    def test_byte_hash_seam_accepts_a_conforming_implementation(self) -> None:
        # `ByteHash` is runtime_checkable, so this is the structural check a
        # scheme's constructor gets for free when it takes the seam.
        self.assertIsInstance(_ZeroHash(), ByteHash)


if __name__ == "__main__":
    absltest.main()
