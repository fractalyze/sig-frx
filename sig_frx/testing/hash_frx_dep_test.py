# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""hash-frx is reachable from this repo, and computes the right bytes.

Every scheme here hashes through hash-frx, and the wiring is the part most likely
to be subtly wrong: the module pin, the second pip hub it brings, and the frx
copy each hub resolves. A scheme test would catch a break too — months later and
behind its own failure. This guard fails loudly instead.

SHA-256 is the computation because it is the primitive the hash-based schemes
build on, and because its answer is checkable against `hashlib` rather than
against itself. A wiring test that only proved something ran would not have
caught a hash that runs and is wrong.

`digest` is called eagerly, not under `jit`: it pads on the host, which is sound
because the message length is static, and means it takes a concrete array. The
traceable entry point is `sha256_chain` over pre-built blocks, which is what a
scheme hashing inside a traced region uses — that boundary belongs to the scheme
that first crosses it, not to a wiring guard.
"""

from __future__ import annotations

import hashlib

import frx.numpy as fnp
import hash_frx
import numpy as np
from absl.testing import absltest
from hash_frx.byte_hash import ByteHash
from hash_frx.sha256 import HostSha256, Sha256

# Equal length, because `digest` takes a batch of equal-length messages.
_MESSAGES = (b"abcdefgh", b"sig-frx\n", b"\x00" * 8)


def _batch() -> fnp.ndarray:
    return fnp.asarray(
        np.frombuffer(b"".join(_MESSAGES), dtype=np.uint8).reshape(len(_MESSAGES), 8)
    )


class HashFrxDependencyTest(absltest.TestCase):
    def test_package_carries_a_version(self) -> None:
        self.assertTrue(getattr(hash_frx, "__version__", ""))

    def test_the_device_hash_agrees_with_the_standard(self) -> None:
        digests = np.asarray(Sha256().digest(_batch()))
        self.assertEqual(
            [bytes(row) for row in digests],
            [hashlib.sha256(m).digest() for m in _MESSAGES],
        )

    def test_both_implementations_agree(self) -> None:
        # hash-frx ships a device and a host SHA-256 of the identical FIPS 180-4
        # bytes. A scheme picks by deployment, so the two must never diverge.
        batch = _batch()
        self.assertEqual(
            [bytes(row) for row in np.asarray(Sha256().digest(batch))],
            [bytes(row) for row in np.asarray(HostSha256().digest(batch))],
        )

    def test_both_implementations_satisfy_the_seam(self) -> None:
        # `ByteHash` is runtime_checkable, so this is the structural check a
        # scheme's constructor gets for free when it takes the seam.
        self.assertIsInstance(Sha256(), ByteHash)
        self.assertIsInstance(HostSha256(), ByteHash)


if __name__ == "__main__":
    absltest.main()
