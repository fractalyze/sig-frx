# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The FXMSS walk climbs from the reference's leaf to the reference's root.

`sf_root` is the public key's third part, so reaching it is the whole of what a
stateful signature claims. The cases vary the two things the walk's shape depends
on — how deep the leaf sits, and which side of each parent it falls on — because
one depth and one parity would pass with the mask or the bit test wrong.

**The signer's side is gated against the same root, from the other direction.**
`root` builds the tree the verifier only ever climbs a path through, so the two
meeting at `sf_root` is what says the build and the walk agree about the shape —
and the reference's own root is what says they are both right rather than both
wrong. The leaf schedule is checked separately and without hashing, because it is
the one thing here that a wrong answer to is a broken key rather than a failed
verification: two signatures at one leaf reveal its secret.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest
from hash_frx import Sha256

from sig_frx.hash.shrincs import fxmss, shrincs
from sig_frx.hash.shrincs.testing import stateful_vectors as vectors
from sig_frx.hash.tweakable import Sha2TweakableHash

_TWEAK = Sha2TweakableHash(Sha256(), n=16, m=24)


def _rows(*values: bytes) -> np.ndarray:
    return np.stack([np.frombuffer(v, dtype=np.uint8) for v in values])


def _padded(case: vectors.StatefulVectors) -> bytes:
    """The FXMSS signature, zero-padded to the widest the format allows."""
    body = case.signature[17 + case.leaf_index_size :]
    return body + bytes(fxmss.SIGNATURE_SIZE_MAX - len(body))


def _index(case: vectors.StatefulVectors) -> bytes:
    return case.leaf_index.to_bytes(8, "big")


def _structure(case: vectors.StatefulVectors) -> fxmss.Structure:
    return fxmss.Structure.parse(np.array([case.shape, case.depth], dtype=np.uint8))


def _secrets(
    case: vectors.StatefulVectors | vectors.DepthZero,
) -> tuple[np.ndarray, np.ndarray]:
    """`SK.seed` and `PK.seed` out of the 48 bytes key generation takes.

    Every case records that seed whatever else it records, so this takes the
    field rather than the case class it came on.
    """
    return (
        np.frombuffer(case.seed[:16], dtype=np.uint8),
        np.frombuffer(case.seed[32:], dtype=np.uint8),
    )


class ConstantTest(absltest.TestCase):
    def test_the_sizes_are_the_specifications(self) -> None:
        self.assertEqual(fxmss.HEIGHT, 255)
        self.assertEqual(fxmss.SIGNATURE_SIZE_MIN, 530)
        self.assertEqual(fxmss.SIGNATURE_SIZE_MAX, 4594)

    def test_the_batched_width_rule_agrees_with_the_host_one(self) -> None:
        """`shrincs.index_field_widths` says it is pinned here. Now it is.

        The same rule stated twice — once for a parser holding one concrete depth,
        once for a batch of them — so the two have to agree at every depth a leaf
        can sit at, not only at the ones the vectors happen to use.
        """
        heights = np.arange(fxmss.HEIGHT + 1, dtype=np.uint32)
        widths, bits = shrincs.index_field_widths(heights)
        self.assertEqual(
            [int(w) for w in np.asarray(widths)],
            [fxmss.index_field_bytes(fxmss.HEIGHT - int(h)) for h in heights],
        )
        # The companion column: the bits an index may carry at that depth, capped
        # where the eight-byte field stops being able to hold more.
        self.assertEqual(
            [int(b) for b in np.asarray(bits)],
            [min(fxmss.HEIGHT - int(h), 8 * fxmss.INDEX_BYTES) for h in heights],
        )

    def test_the_index_field_widens_with_the_depth(self) -> None:
        # The widths the cases below actually exercise, plus the two boundaries:
        # eight bytes is reached at 57 bits and never exceeded.
        self.assertEqual(fxmss.index_field_bytes(1), 1)
        self.assertEqual(fxmss.index_field_bytes(8), 1)
        self.assertEqual(fxmss.index_field_bytes(9), 2)
        self.assertEqual(fxmss.index_field_bytes(56), 7)
        self.assertEqual(fxmss.index_field_bytes(57), 8)
        self.assertEqual(fxmss.index_field_bytes(64), 8)
        self.assertEqual(fxmss.index_field_bytes(255), 8)

    def test_every_case_agrees_with_it(self) -> None:
        for case in vectors.REFERENCE:
            with self.subTest(case.label):
                self.assertEqual(
                    fxmss.index_field_bytes(case.leaf_depth), case.leaf_index_size
                )


class RootTest(absltest.TestCase):
    def test_every_reference_signature_climbs_to_sf_root(self) -> None:
        for case in vectors.REFERENCE:
            with self.subTest(case.label, depth=case.leaf_depth):
                roots, accepted = fxmss.root_from_sig(
                    _TWEAK,
                    _rows(case.pk_seed),
                    _rows(_padded(case)),
                    _rows(case.message_digest),
                    np.array([case.leaf_height], dtype=np.uint32),
                    _rows(_index(case)),
                )
                self.assertEqual(list(np.asarray(accepted)), [True])
                self.assertEqual(bytes(np.asarray(roots)[0]), case.sf_root)

    def test_a_batch_of_different_depths_climbs_independently(self) -> None:
        """The masking is what this is for: depths 1, 4, 16 and 64 in one call.

        Each entry must stop at its own root and hold there while the others keep
        climbing — a mask that leaked would give every entry the deepest walk.
        """
        cases = list(vectors.REFERENCE)
        roots, accepted = fxmss.root_from_sig(
            _TWEAK,
            _rows(*(c.pk_seed for c in cases)),
            _rows(*(_padded(c) for c in cases)),
            _rows(*(c.message_digest for c in cases)),
            np.array([c.leaf_height for c in cases], dtype=np.uint32),
            _rows(*(_index(c) for c in cases)),
        )
        self.assertEqual(list(np.asarray(accepted)), [True] * len(cases))
        self.assertEqual(
            [bytes(row) for row in np.asarray(roots)], [c.sf_root for c in cases]
        )

    def test_a_wrong_sibling_reaches_another_root(self) -> None:
        case = vectors.REFERENCE[3]  # depth 16, so there are siblings to break
        body = bytearray(_padded(case))
        body[fxmss.SIGNATURE_SIZE_MIN - 1] ^= 0x01  # the first auth-path node
        roots, _ = fxmss.root_from_sig(
            _TWEAK,
            _rows(case.pk_seed),
            _rows(bytes(body)),
            _rows(case.message_digest),
            np.array([case.leaf_height], dtype=np.uint32),
            _rows(_index(case)),
        )
        self.assertNotEqual(bytes(np.asarray(roots)[0]), case.sf_root)

    def test_a_wrong_leaf_index_reaches_another_root(self) -> None:
        """The index picks a side at every level, so changing it changes the walk.

        Case 1 sits at index 11 — `0b1011` — so its four levels go right, right,
        left, right, and flipping the low bit changes the very first pairing.
        """
        case = vectors.REFERENCE[1]
        roots, _ = fxmss.root_from_sig(
            _TWEAK,
            _rows(case.pk_seed),
            _rows(_padded(case)),
            _rows(case.message_digest),
            np.array([case.leaf_height], dtype=np.uint32),
            _rows((case.leaf_index ^ 1).to_bytes(8, "big")),
        )
        self.assertNotEqual(bytes(np.asarray(roots)[0]), case.sf_root)

    def test_a_wrong_depth_reaches_another_root(self) -> None:
        """One step too many or too few, and the walk stops somewhere else."""
        case = vectors.REFERENCE[0]
        for height in (case.leaf_height - 1, case.leaf_height + 1):
            with self.subTest(height=height):
                roots, _ = fxmss.root_from_sig(
                    _TWEAK,
                    _rows(case.pk_seed),
                    _rows(_padded(case)),
                    _rows(case.message_digest),
                    np.array([height], dtype=np.uint32),
                    _rows(_index(case)),
                )
                self.assertNotEqual(bytes(np.asarray(roots)[0]), case.sf_root)


class StructureTest(absltest.TestCase):
    """The leaf schedule, which decides what a counter signs with — and only once."""

    def test_every_reference_counter_names_its_leaf(self) -> None:
        for case in vectors.REFERENCE:
            with self.subTest(case.label):
                self.assertEqual(
                    _structure(case).leaf(case.state_counter),
                    (case.leaf_index, case.leaf_height),
                )

    def test_a_balanced_tree_spends_its_leaves_left_to_right(self) -> None:
        """`2^d` leaves at one height, so every signature is the same length."""
        structure = fxmss.Structure(shape=fxmss.SHAPE_BALANCED, depth=3)
        self.assertEqual(structure.leaf_count, 8)
        self.assertEqual(
            [structure.leaf(counter) for counter in range(8)],
            [(index, fxmss.HEIGHT - 3) for index in range(8)],
        )

    def test_an_unbalanced_tree_spends_its_leaves_top_down(self) -> None:
        """One leaf per height and one at the bottom, so each is a node longer."""
        structure = fxmss.Structure(shape=fxmss.SHAPE_UNBALANCED, depth=3)
        self.assertEqual(structure.leaf_count, 4)
        self.assertEqual(
            [structure.leaf(counter) for counter in range(4)],
            [(1, 254), (1, 253), (1, 252), (0, 252)],
        )

    def test_a_spent_counter_has_no_leaf(self) -> None:
        """The boundary is the whole point: one past the last is not the last."""
        for shape, depth, last in (
            (fxmss.SHAPE_BALANCED, 3, 7),
            (fxmss.SHAPE_UNBALANCED, 3, 3),
        ):
            structure = fxmss.Structure(shape=shape, depth=depth)
            with self.subTest(shape=shape):
                self.assertIsNotNone(structure.leaf(last))
                with self.assertRaisesRegex(ValueError, "no leaf left"):
                    structure.leaf(last + 1)

    def test_a_depth_zero_tree_signs_nothing_but_is_still_a_tree(self) -> None:
        """One leaf built, none signable — the indicator cannot name height 255."""
        for shape in (fxmss.SHAPE_UNBALANCED, fxmss.SHAPE_BALANCED):
            structure = fxmss.Structure(shape=shape, depth=0)
            with self.subTest(shape=shape):
                self.assertEqual(structure.leaves_built, 1)
                self.assertEqual(structure.leaf_count, 0)
                with self.assertRaisesRegex(ValueError, "no leaf left"):
                    structure.leaf(0)
                # The leaf is built and still may not sign, so the two questions
                # part here and `holds` answers the one `sign` asks.
                self.assertFalse(structure.holds(0, fxmss.HEIGHT))

    def test_an_unknown_shape_is_refused(self) -> None:
        """Refused rather than defaulted — see `Structure.parse`."""
        with self.assertRaisesRegex(ValueError, "prescribed FXMSS shapes"):
            fxmss.Structure.parse(np.array([2, 4], dtype=np.uint8))

    def test_a_tree_too_large_to_build_is_refused(self) -> None:
        """A balanced depth is an exponent, so an untrusted one is a hang."""
        with self.assertRaisesRegex(ValueError, "refuses above"):
            fxmss.Structure.parse(np.array([fxmss.SHAPE_BALANCED, 40], dtype=np.uint8))


class SignerTest(absltest.TestCase):
    def test_every_reference_root_is_built(self) -> None:
        """The value `root_from_sig` climbs to, reached by building the whole tree."""
        for case in vectors.REFERENCE:
            with self.subTest(case.label):
                sk_seed, pk_seed = _secrets(case)
                root = fxmss.root(_TWEAK, pk_seed, sk_seed, _structure(case))
                self.assertEqual(bytes(np.asarray(root)), case.sf_root)

    def test_each_shape_reproduces_its_reference_signature(self) -> None:
        """One case per shape: what differs between them is how the tree closes.

        The per-depth variation — the index field's width, the number of Merkle
        steps — is the assembled signature's, and `shrincs_test` covers every case
        of it. What is left here is that a balanced tree's forest walk and an
        unbalanced tree's spine each produce the reference's path.
        """
        for case in (vectors.REFERENCE[0], vectors.REFERENCE[3]):
            with self.subTest(case.label):
                sk_seed, pk_seed = _secrets(case)
                signature = fxmss.sign(
                    _TWEAK,
                    pk_seed,
                    sk_seed,
                    _structure(case),
                    np.frombuffer(case.message_digest, dtype=np.uint8),
                    case.leaf_height,
                    case.leaf_index,
                )
                body = case.signature[17 + case.leaf_index_size :]
                self.assertEqual(bytes(np.asarray(signature)), body)

    def test_a_depth_zero_tree_still_has_a_root(self) -> None:
        """A stateless-only key: it signs nothing statefully and still has `sf_root`.

        Both shapes name the same single leaf and reach different roots, which is
        the structure bytes in the WOTS+C PRF address doing their work — without
        them one seed would give the two keys the same stateful third.
        """
        case = vectors.DEPTH_ZERO
        sk_seed, pk_seed = _secrets(case)
        roots = {}
        for shape, want in (
            (fxmss.SHAPE_UNBALANCED, case.unbalanced_sf_root),
            (fxmss.SHAPE_BALANCED, case.balanced_sf_root),
        ):
            with self.subTest(shape=shape):
                root = fxmss.root(
                    _TWEAK, pk_seed, sk_seed, fxmss.Structure(shape=shape, depth=0)
                )
                roots[shape] = bytes(np.asarray(root))
                self.assertEqual(roots[shape], want)
        self.assertNotEqual(roots[fxmss.SHAPE_UNBALANCED], roots[fxmss.SHAPE_BALANCED])

    def test_a_leaf_the_shape_does_not_name_is_refused(self) -> None:
        """A key made off the tree recovers a node the root was not built from."""
        case = vectors.REFERENCE[0]
        sk_seed, pk_seed = _secrets(case)
        with self.assertRaisesRegex(ValueError, "no leaf of this tree"):
            fxmss.sign(
                _TWEAK,
                pk_seed,
                sk_seed,
                _structure(case),
                np.frombuffer(case.message_digest, dtype=np.uint8),
                case.leaf_height - 1,
                case.leaf_index,
            )

    def test_the_one_leaf_of_a_depth_zero_tree_cannot_be_signed_from(self) -> None:
        """Built is not signable: `sign` refuses where `leaf` hands out nothing.

        The position is the tree's only leaf, so a guard that asked whether a leaf
        exists there would pass it through to an authentication path of no nodes.
        """
        case = vectors.DEPTH_ZERO
        sk_seed, pk_seed = _secrets(case)
        for shape in (fxmss.SHAPE_UNBALANCED, fxmss.SHAPE_BALANCED):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "no leaf of this tree"):
                    fxmss.sign(
                        _TWEAK,
                        pk_seed,
                        sk_seed,
                        fxmss.Structure(shape=shape, depth=0),
                        # Any digest: the guard fires before one is read.
                        np.frombuffer(
                            vectors.REFERENCE[0].message_digest, dtype=np.uint8
                        ),
                        fxmss.HEIGHT,
                        0,
                    )


if __name__ == "__main__":
    absltest.main()
