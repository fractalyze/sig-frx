# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""LeanSig's tweakable hash family, and the shared walks it plugs into.

Two things are on trial and they fail differently. The family has to reproduce
`PoseidonXmss.tweak_hash` at all three of its modes, which is a byte-exactness
question the vectors answer. The *reuse* has to be real — [`wots.chain`](../../wots.py)
walking leanSig's chains and [`tree.py`](../../tree.py) climbing its tree — and
that is a question no single digest answers: a reimplementation here would agree
with upstream just as well while leaving two Merkle walks in the repo to drift
apart. So the walk cases drive the shared functions and never a local copy, and
the tree cases compare against a tree upstream itself built rather than against
one this file rebuilds.

Every case runs eagerly and traced, and the two must agree in *dtype* as well as
value — the batch here is `frx.vmap` over a one-dimensional mode, which is
exactly where a promotion would go unnoticed by a comparison that only reads
residues back.

Vectors are in leanSpec's lane order and everything here runs over the reverse of
it, so cases convert at the boundary through [`harness.py`](harness.py). A real
caller holds lane-reversed digests already and converts nothing.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from zk_dtypes import koalabear_mont as F

from sig_frx.hash import tree, wots
from sig_frx.hash.leansig import params as leansig_params
from sig_frx.hash.leansig import tweakable
from sig_frx.hash.leansig.testing import harness
from sig_frx.hash.leansig.testing.mode_vectors import (
    DOMAIN_SEPARATOR_VECTORS,
    DomainSeparatorVector,
    operand_elements,
)
from sig_frx.hash.leansig.testing.tweakable_vectors import (
    CHAIN_STEP_VECTORS,
    CHAIN_WALK_VECTORS,
    LEAF_VECTORS,
    TREE_NODE_VECTORS,
    TREE_VECTORS,
    TREE_WALK_VECTORS,
    ChainStepVector,
    ChainWalkVector,
    LeafVector,
    TreeNodeVector,
    TreeVector,
)
from sig_frx.hash.tweakable import ChainHash, NodeHash

_PROD = leansig_params.PROD
_HASH_LENGTH = _PROD.hash_length


def _digest(seed: int) -> fnp.ndarray:
    """One lane-reversed digest, from the rule the vectors state."""
    return harness.lane_reversed(operand_elements(_HASH_LENGTH, seed))


def _parameter(seed: int) -> fnp.ndarray:
    """One lane-reversed public parameter, from the same rule."""
    return harness.lane_reversed(operand_elements(_PROD.parameter_length, seed))


def _chain_ends(seed: int, dimension: int) -> fnp.ndarray:
    """One slot's `dimension` chain ends — `[dimension, n]`, the leaf's operand."""
    return harness.lane_reversed_rows(
        [operand_elements(_HASH_LENGTH, seed + i) for i in range(dimension)]
    )


def _assert_one_digest(
    case: absltest.TestCase, got: fnp.ndarray, expected: tuple[int, ...]
) -> None:
    """The three things every single-digest case asserts: value, dtype, shape.

    The dtype is not decoration — the batch here is `frx.vmap` over a
    one-dimensional mode, which is exactly where a promotion would pass a
    comparison that reads residues back and nothing else.
    """
    case.assertEqual(harness.to_leanspec_rows(got), [list(expected)])
    case.assertEqual(got.dtype, F)
    case.assertEqual(got.shape, (1, _HASH_LENGTH))


class ChainStepTest(parameterized.TestCase):
    """`f` reproduces upstream's single-digest `tweak_hash`."""

    @parameterized.named_parameters(*harness.both_legs(CHAIN_STEP_VECTORS))
    def test_it_matches_upstream(self, vector: ChainStepVector, jit: bool) -> None:
        family = tweakable.LeanSigTweakableHash(_PROD)
        tweaks = tweakable.chain_tweaks(
            [vector.epoch], [vector.chain_index], vector.step, params=_PROD
        )
        step = harness.jitted(family.f) if jit else family.f

        got = step(
            _parameter(vector.parameter_seed), tweaks, _digest(vector.digest_seed)
        )

        _assert_one_digest(self, got, vector.output)

    @parameterized.named_parameters(("host", False), ("traced", True))
    def test_a_batch_is_one_call_over_every_position(self, jit: bool) -> None:
        """The non-negotiable: `B` chain steps at `B` positions, in one call.

        Different parameters as well as different tweaks, since a batch of
        signatures spans public keys — which is the case a shared-parameter
        broadcast would silently pass.
        """
        family = tweakable.LeanSigTweakableHash(_PROD)
        tweaks = tweakable.chain_tweaks(
            [v.epoch for v in CHAIN_STEP_VECTORS],
            [v.chain_index for v in CHAIN_STEP_VECTORS],
            # One step for the batch, which is the shape a masked walk produces.
            CHAIN_STEP_VECTORS[0].step,
            params=_PROD,
        )
        parameters = fnp.stack(
            [_parameter(v.parameter_seed) for v in CHAIN_STEP_VECTORS]
        )
        digests = fnp.stack([_digest(v.digest_seed) for v in CHAIN_STEP_VECTORS])
        step = harness.jitted(family.f) if jit else family.f

        got = step(parameters, tweaks, digests)

        # Row `k` must equal what row `k`'s own operands give on their own. The
        # rows are sliced out of the batch rather than rebuilt from the seeds:
        # rebuilding would let a change to the batch's inputs go on passing
        # against a differently-built expectation, since every row stays
        # self-consistent. Upstream is what gates the single-row path, above.
        expected = [
            harness.to_leanspec_rows(
                family.f(parameters[k : k + 1], tweaks[k : k + 1], digests[k : k + 1])
            )[0]
            for k in range(len(CHAIN_STEP_VECTORS))
        ]
        self.assertEqual(harness.to_leanspec_rows(got), expected)
        self.assertEqual(got.dtype, F)


class TreeNodeTest(parameterized.TestCase):
    """`h` reproduces upstream's two-digest `tweak_hash`."""

    @parameterized.named_parameters(*harness.both_legs(TREE_NODE_VECTORS))
    def test_it_matches_upstream(self, vector: TreeNodeVector, jit: bool) -> None:
        family = tweakable.LeanSigTweakableHash(_PROD)
        tweaks = tweakable.tree_tweaks(vector.level, [vector.node_index], params=_PROD)
        # The concatenated pair `tree.py` hands every family, at `B = 1`.
        pair = fnp.concatenate([_digest(vector.left_seed), _digest(vector.right_seed)])[
            None, :
        ]
        node = harness.jitted(family.h) if jit else family.h

        got = node(_parameter(vector.parameter_seed), tweaks, pair)

        _assert_one_digest(self, got, vector.output)

    def test_it_is_not_symmetric_in_its_children(self) -> None:
        """Swapping the pair changes the node, so the halves cannot be conflated.

        The reversal makes `left ‖ right` land as `R(right) ‖ R(left)`, so a
        family that split at the wrong end would still be self-consistent — and
        would climb every tree to a wrong root that verified against itself.
        """
        vector = TREE_NODE_VECTORS[0]
        family = tweakable.LeanSigTweakableHash(_PROD)
        tweaks = tweakable.tree_tweaks(vector.level, [vector.node_index], params=_PROD)
        swapped = fnp.concatenate(
            [_digest(vector.right_seed), _digest(vector.left_seed)]
        )[None, :]

        got = family.h(_parameter(vector.parameter_seed), tweaks, swapped)

        self.assertNotEqual(harness.to_leanspec_rows(got), [list(vector.output)])

    def test_it_refuses_a_pair_that_is_not_two_digests(self) -> None:
        family = tweakable.LeanSigTweakableHash(_PROD)
        tweaks = tweakable.tree_tweaks(1, [0], params=_PROD)

        with self.assertRaisesRegex(ValueError, "Merkle pair"):
            family.h(_parameter(1), tweaks, _digest(2)[None, :])


class LeafTest(parameterized.TestCase):
    """`leaf` reproduces upstream's many-digest `tweak_hash`, at both presets."""

    @parameterized.named_parameters(*harness.both_legs(LEAF_VECTORS))
    def test_it_matches_upstream(self, vector: LeafVector, jit: bool) -> None:
        preset = harness.PRESETS[vector.preset]
        family = tweakable.LeanSigTweakableHash(preset)
        tweaks = tweakable.tree_tweaks(0, [vector.position], params=preset)
        ends = _chain_ends(vector.chain_end_seed, preset.dimension)[None, :, :]
        leaf = harness.jitted(family.leaf) if jit else family.leaf

        got = leaf(_parameter(vector.parameter_seed), tweaks, ends)

        _assert_one_digest(self, got, vector.output)

    @parameterized.named_parameters(
        *[
            (vector.name, vector, harness.PRESETS[vector.name.removesuffix("_config")])
            for vector in DOMAIN_SEPARATOR_VECTORS
        ]
    )
    def test_the_capacity_is_the_separator_for_this_presets_shape(
        self, vector: DomainSeparatorVector, preset: leansig_params.LeanSigParams
    ) -> None:
        """The `lengths` the family packs, against a separator already gated.

        `mode_test` pins `safe_domain_separator` itself, so what is left to get
        wrong here is which four numbers go in and in what order — and two of
        them are 5 and 2, which swap without changing the shape of anything.
        A leaf digest would catch it too, in a way that says only that some
        operand somewhere is wrong.
        """
        got = tweakable.LeanSigTweakableHash(preset).capacity_value

        self.assertEqual(harness.to_leanspec_order(got), list(vector.output))

    def test_it_refuses_a_chain_end_count_the_preset_does_not_name(self) -> None:
        """The separator is built from `DIMENSION`, so a short leaf is a wrong
        hash rather than a shorter one — it must not be hashed at all."""
        family = tweakable.LeanSigTweakableHash(leansig_params.TEST)
        tweaks = tweakable.tree_tweaks(0, [0], params=leansig_params.TEST)
        ends = fnp.stack([_digest(10 + i) for i in range(3)])[None, :, :]

        with self.assertRaisesRegex(ValueError, "a leaf hashes 4 digests"):
            family.leaf(_parameter(1), tweaks, ends)


class ChainWalkTest(parameterized.TestCase):
    """`wots.chain` walks leanSig's chains — the shared walk, not a copy here."""

    @parameterized.named_parameters(*harness.both_legs(CHAIN_WALK_VECTORS))
    def test_it_matches_upstream_hash_chain(
        self, vector: ChainWalkVector, jit: bool
    ) -> None:
        family = tweakable.LeanSigTweakableHash(_PROD)

        # `frx.jit` rather than `harness.jitted`, which memoizes on the callable
        # and so takes module-level functions only — a per-case closure has no
        # second caller to share a wrapper with.
        def walk(parameter: fnp.ndarray, digest: fnp.ndarray) -> fnp.ndarray:
            return wots.chain(
                family,
                parameter,
                digest,
                [vector.start_step],
                [vector.num_steps],
                tweakable.chain_step_tweaks(
                    [vector.epoch], [vector.chain_index], params=_PROD
                ),
            )

        got = (frx.jit(walk) if jit else walk)(
            _parameter(vector.parameter_seed), _digest(vector.digest_seed)[None, :]
        )

        self.assertEqual(harness.to_leanspec_rows(got), [list(vector.output)])
        self.assertEqual(got.dtype, F)

    def test_every_chain_runs_the_same_number_of_hashes(self) -> None:
        """The masked walk is the point: two chains stopping at different digits
        cost the same, so the step count is not a function of the message."""
        steps = list(
            tweakable.chain_step_tweaks([0, 0], [0, 1], params=leansig_params.TEST)
        )

        self.assertLen(steps, leansig_params.TEST.base - 1)
        self.assertEqual([np.asarray(s).shape for s in steps], [(2, 2)] * len(steps))


class TreeWalkTest(parameterized.TestCase):
    """The verifier's climb: `leaf`, then `tree.root_from_path` to the root."""

    @parameterized.named_parameters(("host", False), ("traced", True))
    def test_a_batch_of_walks_matches_upstream(self, jit: bool) -> None:
        """Every position in one call, which is what a verifier holds.

        The three cases differ in parity at every level — one always left, one
        always right, one alternating — so the left/right select is exercised
        both ways within a single batched climb.
        """
        vectors = TREE_WALK_VECTORS
        preset = harness.PRESETS[vectors[0].preset]
        family = tweakable.LeanSigTweakableHash(preset)
        positions = [v.position for v in vectors]
        parameters = fnp.stack([_parameter(v.parameter_seed) for v in vectors])

        leaves = family.leaf(
            parameters,
            tweakable.tree_tweaks(0, positions, params=preset),
            fnp.stack(
                [_chain_ends(v.chain_end_seed, preset.dimension) for v in vectors]
            ),
        )
        paths = fnp.stack(
            [
                fnp.stack([_digest(v.sibling_seed + i) for i in range(v.log_lifetime)])
                for v in vectors
            ]
        )

        def climb(leaf_hashes: fnp.ndarray, siblings: fnp.ndarray) -> fnp.ndarray:
            return tree.root_from_path(
                family,
                parameters,
                leaf_hashes,
                np.asarray(positions),
                siblings,
                tweakable.node_tweaks(preset),
            )

        got = (frx.jit(climb) if jit else climb)(leaves, paths)

        self.assertEqual(harness.to_leanspec_rows(got), [list(v.root) for v in vectors])
        self.assertEqual(got.dtype, F)


_TREE_CASES = [(vector.name, vector) for vector in TREE_VECTORS]


def _leaves(vector: TreeVector) -> fnp.ndarray:
    """The tree's lowest layer — digests standing in for leaf hashes, as upstream
    took them."""
    return fnp.stack([_digest(vector.leaf_seed + i) for i in range(1 << vector.depth)])


class TreeTest(parameterized.TestCase):
    """`tree.py` builds and opens leanSig's tree, against one upstream built."""

    @parameterized.named_parameters(*_TREE_CASES)
    def test_the_root_matches_upstream(self, vector: TreeVector) -> None:
        family = tweakable.LeanSigTweakableHash(_PROD)
        leaves = _leaves(vector)

        got = tree.root(
            family,
            _parameter(vector.parameter_seed),
            leaves,
            tweakable.node_tweaks(_PROD),
        )

        self.assertEqual(harness.to_leanspec_order(got), list(vector.root))
        self.assertEqual(got.dtype, F)

    @parameterized.named_parameters(*_TREE_CASES)
    def test_the_openings_match_upstream(self, vector: TreeVector) -> None:
        family = tweakable.LeanSigTweakableHash(_PROD)
        leaves = _leaves(vector)

        got = tree.auth_path(
            family,
            _parameter(vector.parameter_seed),
            leaves,
            np.asarray(vector.positions),
            vector.depth,
            tweakable.node_tweaks(_PROD),
        )

        self.assertEqual(
            [harness.to_leanspec_rows(path) for path in np.asarray(got)],
            [[list(sibling) for sibling in path] for path in vector.paths],
        )

    @parameterized.named_parameters(*_TREE_CASES)
    def test_upstreams_openings_climb_to_upstreams_root(
        self, vector: TreeVector
    ) -> None:
        """Fed upstream's own paths rather than the ones just built, so the climb
        is gated against upstream and not against this file's other direction."""
        family = tweakable.LeanSigTweakableHash(_PROD)
        leaves = _leaves(vector)
        paths = fnp.stack(
            [
                fnp.stack([harness.lane_reversed(sibling) for sibling in path])
                for path in vector.paths
            ]
        )

        got = tree.root_from_path(
            family,
            _parameter(vector.parameter_seed),
            leaves[np.asarray(vector.positions)],
            np.asarray(vector.positions),
            paths,
            tweakable.node_tweaks(_PROD),
        )

        self.assertEqual(
            harness.to_leanspec_rows(got), [list(vector.root)] * len(vector.positions)
        )


class TweakTest(absltest.TestCase):
    """The packed tweak refuses what would silently address another position."""

    def test_the_three_prefixes_separate_the_families(self) -> None:
        """A chain tweak and a tree tweak at the same numbers must differ."""
        chain = tweakable.chain_tweaks([0], [0], 1, params=_PROD)
        node = tweakable.tree_tweaks(0, [0], params=_PROD)

        self.assertNotEqual(
            harness.to_leanspec_rows(chain), harness.to_leanspec_rows(node)
        )

    def test_it_refuses_an_index_that_would_carry_into_the_level(self) -> None:
        with self.assertRaisesRegex(ValueError, "index takes 32 bits"):
            tweakable.tree_tweaks(0, [1 << 32], params=_PROD)

    def test_it_refuses_a_chain_index_that_would_carry_into_the_epoch(self) -> None:
        with self.assertRaisesRegex(ValueError, "chain_index takes 8 bits"):
            tweakable.chain_tweaks([0], [256], 1, params=_PROD)

    def test_it_refuses_a_step_that_would_carry_into_the_chain_index(self) -> None:
        with self.assertRaisesRegex(ValueError, "step takes 8 bits"):
            tweakable.chain_tweaks([0], [0], 256, params=_PROD)

    def test_it_names_the_first_offender_in_a_batch(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "1 of 3 entries.*the first being 4294967296"
        ):
            tweakable.tree_tweaks(0, [0, 1 << 32, 5], params=_PROD)

    def test_it_refuses_a_traced_index(self) -> None:
        """Host-only, and loudly. Packing reaches `2^45`, which no lane holds, so
        a traced index has to fail rather than be pulled quietly to the host —
        which is what `np.asarray` does under a tracer and nowhere else."""

        @frx.jit
        def build(indices: fnp.ndarray) -> fnp.ndarray:
            return tweakable.tree_tweaks(0, indices, params=_PROD)

        with self.assertRaises(Exception):
            build(fnp.asarray(np.asarray([1])))


class SeamTest(absltest.TestCase):
    """What the shared components ask of a family, this one answers."""

    def test_it_is_the_two_protocols_the_shared_walks_take(self) -> None:
        family = tweakable.LeanSigTweakableHash(_PROD)

        self.assertIsInstance(family, ChainHash)
        self.assertIsInstance(family, NodeHash)

    def test_the_digest_dtype_is_the_field_and_not_bytes(self) -> None:
        """The field the shared walks build their arrays at — the whole of what
        made them generic. A family that left this at `uint8` would truncate
        every digest it climbed with."""
        self.assertEqual(tweakable.LeanSigTweakableHash(_PROD).dtype, F)
        self.assertEqual(tweakable.LeanSigTweakableHash(_PROD).n, _HASH_LENGTH)

    def test_equal_presets_compare_and_hash_equal(self) -> None:
        """It rides pytree aux, where identity equality re-traces silently rather
        than failing — see the `Signature` seam's docstring."""
        self.assertEqual(
            tweakable.LeanSigTweakableHash(_PROD), tweakable.LeanSigTweakableHash(_PROD)
        )
        self.assertEqual(
            hash(tweakable.LeanSigTweakableHash(_PROD)),
            hash(tweakable.LeanSigTweakableHash(_PROD)),
        )
        self.assertNotEqual(
            tweakable.LeanSigTweakableHash(_PROD),
            tweakable.LeanSigTweakableHash(leansig_params.TEST),
        )


if __name__ == "__main__":
    absltest.main()
