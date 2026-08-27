# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""SHRINCS verification under `frx.jit` — the whole batch as one traced program.

The seam's contract is that one call is one traced computation over the whole
batch, and for SHRINCS that is not a nicety: eagerly the path is 292 XLA
executables and hundreds of milliseconds a call, and traced it is one executable
and single-digit milliseconds. Nothing in the suite jitted it until this, so the
path a deployment is meant to run was the one path with no coverage.

The claim here is not that a traced call runs. It is that it returns the eager
verdicts entry for entry — including the entries meant to fail, because a
verifier that accepts everything reproduces every passing one. `shrincs_test` is
what says the arithmetic is right against the reference; this says tracing did
not change it.

**Both paths in one call is the SHRINCS-specific part.** The indicator byte
selects a verdict, but a traced program has no branch to take: it runs the
stateless leg and the stateful leg for every entry and selects between them. So
a batch mixing the two paths exercises something no single-path batch reaches,
and mixing them is the default here rather than a case.

The context is closed over rather than passed. `context.prefix` documents itself
as a host value and the seam scopes a context to a whole batch, so a jitted
verifier is specialised per context — passing one as an argument traces it and
raises. That is the design, and this is what a consumer does with it.

Its own target rather than a case in `shrincs_test`, for the reason
`slh_dsa_traced_test` is: the cost is compilation rather than hashing. The walk
unrolls 255 levels for every entry whatever the tree's depth, which is most of
this module's runtime and none of what the assembly cases beside it check.
"""

from __future__ import annotations

import frx
import numpy as np
from absl.testing import absltest

from sig_frx.hash.shrincs import fxmss, shrincs


def _rows(*values: bytes, width: int | None = None) -> np.ndarray:
    """The batch axis the seam takes, zero-filled to `width` where it is given."""
    rows = []
    for value in values:
        row = np.frombuffer(value, dtype=np.uint8)
        if width is not None:
            row = np.concatenate([row, np.zeros(width - row.shape[0], dtype=np.uint8)])
        rows.append(row)
    return np.stack(rows)


# Past the indicator byte and the 16-byte randomizer, which both paths carry — so
# this offset is inside the signature proper whichever path an entry took.
_BODY = shrincs.INDEX_FIELD_START


def _tampered(signature: bytes) -> bytes:
    """One bit of the first body byte.

    **Not the last byte.** `sign` returns its result zero-padded to the seam's
    `signature_max_size`, and a stateful signature is far shorter than a
    stateless one, so the tail of a padded stateful signature is padding — which
    the verifier derives its way past and never reads. Flipping it leaves the
    verdict `True`, which is a passing-looking test that checks nothing.

    Not the first byte either: that is the indicator, and flipping it moves the
    entry to the other path rather than corrupting it there. `shrincs_test`
    covers that case.
    """
    body = bytearray(signature)
    body[_BODY] ^= 1
    return bytes(body)


class TracedVerifyTest(absltest.TestCase):
    """The traced path against the eager one, on a freshly signed pair."""

    def setUp(self) -> None:
        super().setUp()
        # A shallow balanced tree: the walk unrolls all 255 levels whatever the
        # depth, so a deeper one costs signing time and buys this nothing.
        self.scheme = shrincs.Shrincs(
            sf_structure=np.array([fxmss.SHAPE_BALANCED, 2], dtype=np.uint8)
        )
        seed = np.frombuffer(
            bytes((i * 13 + 5) % 256 for i in range(48)), dtype=np.uint8
        )
        public_key, secret_key = (np.asarray(part) for part in self.scheme.keygen(seed))
        self.public_key = bytes(public_key)
        # Signed here rather than taken from the vectors: the seam's `L` is
        # static, so one batch holds one message length, and no recorded
        # stateful and stateless case shares one. `shrincs_test` is what holds
        # these paths to the reference bytes; this holds tracing to eager.
        self.message = np.frombuffer(b"one message, signed both ways", dtype=np.uint8)
        stateful, _ = self.scheme.sign(secret_key, self.message, 0)
        stateless, _ = self.scheme.sign(
            secret_key,
            self.message,
            None,
            randomness=np.zeros(16, dtype=np.uint8),
        )
        self.stateful = bytes(np.asarray(stateful))
        self.stateless = bytes(np.asarray(stateless))
        self.assertNotEqual(self.stateful[0], self.stateless[0])

    def _verdicts(self, signatures: list[bytes]) -> np.ndarray:
        """Traced verdicts, asserted equal to the eager ones entry for entry."""
        count = len(signatures)
        keys = _rows(*([self.public_key] * count))
        messages = np.stack([self.message] * count)
        rows = _rows(*signatures, width=self.scheme.signature_max_size)
        eager = np.asarray(self.scheme.verify(keys, messages, rows))
        traced = np.asarray(
            frx.jit(lambda k, m, s: self.scheme.verify(k, m, s, context=None))(
                keys, messages, rows
            )
        )
        np.testing.assert_array_equal(traced, eager)
        return traced

    def test_both_paths_in_one_batch_trace_to_the_eager_verdicts(self) -> None:
        """The alternation is the point: a verdict that smears is visible either way.

        Each failing entry sits between two passing ones and each passing entry
        between two failing ones, so a verifier that accepts a bad entry and one
        that loses a good one both fail here — whichever direction a verdict
        would smear along the batch. A batch that agrees with itself cannot check
        that: a verifier returning one answer unconditionally is right about
        every entry of one.
        """
        verdicts = self._verdicts(
            [
                self.stateful,
                _tampered(self.stateless),
                self.stateless,
                _tampered(self.stateful),
            ]
        )
        self.assertEqual(
            [bool(verdict) for verdict in verdicts], [True, False, True, False]
        )


if __name__ == "__main__":
    absltest.main()
