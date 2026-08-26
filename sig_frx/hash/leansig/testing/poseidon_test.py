# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""LeanSig's Poseidon against leanSpec's published permutation vectors.

The gate is the permutation itself rather than a whole signature: a scheme that
only checks its own end-to-end output cannot tell a wrong round constant from a
wrong tree walk, and upstream publishes these separately for that reason.

Every case runs both eagerly and traced. The two must agree in *dtype* as well as
value — a residue read back through numpy promotes where a traced one does not,
and a permutation whose output silently changed width would still compare equal
value-by-value.

The vectors are in leanSpec's lane order and
[`poseidon.py`](../poseidon.py)'s permutation runs on the reverse of it, so each
case reverses on the way in and back on the way out. That reversal belongs to
this test, not to the scheme: the compression and sponge layers place their
operands reversed instead of moving data.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import frx
import frx.numpy as fnp
from absl.testing import absltest, parameterized
from zk_dtypes import koalabear_mont as F

from sig_frx.hash.leansig import poseidon
from sig_frx.hash.leansig.testing import harness
from sig_frx.hash.leansig.testing.poseidon_vectors import (
    VECTORS,
    PermutationVector,
)


@lru_cache(maxsize=None)
def _traced_permute(width: int) -> Callable[[fnp.ndarray], fnp.ndarray]:
    """One jit wrapper per width. A fresh `frx.jit` around the same callable
    re-traces every call, so wrapping per case would pay one compile per vector
    instead of one per shape."""
    return frx.jit(poseidon.lane_reversed_permutation(width).permute)


def _permute(vector: PermutationVector, *, jit: bool) -> fnp.ndarray:
    """The permutation's own output for `vector`, so still in reversed order.

    Reversing is this test's business, not the scheme's: the vectors are in
    leanSpec's lane order, so the input is reversed here and each caller reverses
    the residues it reads back. Both run on host values — a tuple going in, a
    list coming out — so no device `reverse` is dispatched. A real caller
    reverses neither: it places and slices from the other end (see
    [`poseidon.py`](../poseidon.py)).
    """
    width = vector.width
    permute = (
        _traced_permute(width)
        if jit
        else poseidon.lane_reversed_permutation(width).permute
    )
    return permute(harness.lane_reversed(vector.input_state))


class PublishedVectorTest(parameterized.TestCase):
    """The permutation byte-matches every vector leanSpec publishes."""

    @parameterized.named_parameters(
        *(
            (f"{vector.name}_{'traced' if jit else 'host'}", vector, jit)
            for vector in VECTORS
            for jit in (False, True)
        )
    )
    def test_it_matches_upstream(self, vector: PermutationVector, jit: bool) -> None:
        got = _permute(vector, jit=jit)

        self.assertEqual(harness.to_leanspec_order(got), list(vector.output_state))
        self.assertEqual(got.dtype, F)
        self.assertEqual(got.shape, (vector.width,))


class LaneConventionTest(absltest.TestCase):
    """The reversal is load-bearing, and a caller that forgets it gets a
    different permutation rather than a subtly wrong one."""

    def test_feeding_leanspec_order_directly_does_not_match(self) -> None:
        vector = next(
            v for v in VECTORS if v.name == "test_permutation_width16_incremental_index"
        )
        permutation = poseidon.lane_reversed_permutation(vector.width)

        # The whole mistake: feed a leanSpec-ordered state and read the result
        # back as if it were leanSpec-ordered too.
        mistaken = permutation.permute(harness.to_field(vector.input_state))

        self.assertNotEqual(
            harness.to_leanspec_order(mistaken), list(vector.output_state)
        )


class WidthTest(absltest.TestCase):
    """LeanSig hashes at two widths and asks for no others."""

    def test_it_serves_both_widths(self) -> None:
        for width in (16, 24):
            self.assertEqual(poseidon.lane_reversed_permutation(width).width, width)

    def test_it_caches_one_permutation_per_width(self) -> None:
        # A freshly built permutation rides pytree aux, where a new instance per
        # call re-traces the enclosing jit zone rather than erroring.
        self.assertIs(
            poseidon.lane_reversed_permutation(16),
            poseidon.lane_reversed_permutation(16),
        )

    def test_it_rejects_a_width_the_scheme_does_not_use(self) -> None:
        with self.assertRaisesRegex(ValueError, "widths"):
            poseidon.lane_reversed_permutation(8)


if __name__ == "__main__":
    absltest.main()
