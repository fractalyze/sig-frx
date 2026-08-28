# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""leanSig's signer against leanSpec's — the key, the window, and six signatures.

The verifier was gated first and against upstream's own bytes, which is what
makes this suite worth anything: a signer checked only by the verifier beside it
proves that two halves of one wrong scheme agree. So the claim here is byte
equality with `GeneralizedXmssScheme` — the same public key from the same seed,
and the same signature from the same `(key, slot, message)` — and the round trip
through `verify` is the weaker check that runs afterwards.

Byte equality is a strong claim for a signer specifically. leanSig's signature
carries three things a wrong implementation gets wrong differently: the
randomness the rejection loop settled on, which pins the search order and the
PRF; the released chain values, which pin every chain's start and its walk
distance; and the authentication path, which pins the whole tree and the
top/bottom stitch. A round trip would accept a wrong answer to any of them, as
long as the verifier made the same mistake.

## What the mutations are for

Every wiring decision in `signing.py` has a plausible alternative that still
produces a self-consistent scheme — local rather than whole-tree node indices,
the top and bottom halves of a path concatenated the other way round, a chain
released one step past its digit. Each of those costs a case below, or this
suite is not measuring the wiring it claims to.
"""

from __future__ import annotations

from functools import lru_cache

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from sig_frx.hash.leansig import (
    encoding,
    field,
    leansig,
    prf,
    signing,
    ssz,
    tweakable,
)
from sig_frx.hash.leansig import (
    params as leansig_params,
)
from sig_frx.hash.leansig.testing import harness
from sig_frx.hash.leansig.testing.signing_vectors import (
    MAX_ADVANCES,
    PARAMETER,
    PREPARED,
    PREPARED_AFTER_ONE_ADVANCE,
    PREPARED_AT_END,
    PRF_KEY,
    PUBLIC_KEY,
    SIGN_VECTORS,
    SignVector,
)

_SCHEME = leansig.named("test")
_FAMILY = tweakable.LeanSigTweakableHash(_SCHEME.params)
_KEY = bytes.fromhex(PRF_KEY)


@lru_cache(maxsize=None)
def _key_pair() -> tuple[np.ndarray, signing.SecretKey]:
    """The one key pair every case here is built from — generated once.

    Shared rather than rebuilt per case, which is safe for the reason that
    looks at first like a hazard: `SecretKey` and `SubTree` are frozen, and
    `advance_preparation` returns a *new* key rather than moving this one, so
    no case can slide another's window. Building it per case instead cost 42
    whole-lifetime keygens — 1024 chain starts and ~20 dispatches each — which
    was the whole reason this suite could not fit a `small` budget.

    `test_it_is_deterministic` deliberately does not use this, since two
    independent builds are the thing it is checking.
    """
    public, secret = _SCHEME.keygen(_KEY, PARAMETER)
    return np.asarray(public), secret


def _for_slot(secret: signing.SecretKey, slot: int) -> signing.SecretKey:
    """The key with its window slid far enough forward to serve `slot`.

    The only advance helper the suite needs: a fixed count is the same loop
    with a worse stop condition, and `prepared` is what every case actually
    cares about.
    """
    while slot not in secret.prepared:
        secret = _SCHEME.advance_preparation(secret)
    return secret


class KeygenTest(absltest.TestCase):
    """The public key one seed and one parameter produce."""

    def test_it_matches_upstream(self) -> None:
        public, _ = _key_pair()

        self.assertEqual(harness.hex_of(public), PUBLIC_KEY)

    def test_it_is_deterministic(self) -> None:
        """Twice from the same inputs is the same key.

        Not a tautology here: upstream's own `key_gen` fails this, drawing both
        the seed and the parameter itself, and the pads it fills an unbuilt
        window with fail it even when those are fixed. Taking the two secrets
        and building the whole lifetime is what buys this
        ([`signing.py`](../signing.py)).
        """
        first, _ = _key_pair()
        second, _ = _key_pair()

        self.assertEqual(harness.hex_of(first), harness.hex_of(second))

    def test_the_parameter_travels_in_the_public_key(self) -> None:
        """The container is `root ‖ parameter`, so what went in comes back out.

        Read through `ssz.decode_public_key` rather than by unpacking the words
        here: that module owns the wire format and is gated by its own suite, so
        this is not gating it with itself — and a sixth open-coded reading of
        four-byte residues is exactly what its docstring counts.
        """
        public, _ = _key_pair()

        _, parameter, ok = ssz.decode_public_key(public, params=_SCHEME.params)

        self.assertTrue(bool(ok))
        self.assertEqual(harness.to_leanspec_order(parameter), list(PARAMETER))

    def test_a_parameter_of_the_wrong_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "5 field elements"):
            _SCHEME.keygen(_KEY, PARAMETER[:4])

    def test_a_parameter_that_is_not_canonical(self) -> None:
        """The cast into the field reduces, so this is refused rather than felt.

        `PRIME` would become zero and generate a whole valid key pair for a
        parameter nobody named — the same malleability `ssz.decode_public_key`
        checks for on the way back in.
        """
        with self.assertRaisesRegex(ValueError, "canonical residues"):
            _SCHEME.keygen(_KEY, (field.PRIME, *PARAMETER[1:]))

    def test_an_odd_lifetime_has_no_split(self) -> None:
        """The preset itself refuses one, which is where the split's rule lives.

        Not reachable through `named`, since neither shipped preset is odd — but
        an odd exponent would give a bottom tree one level short of where the top
        tree starts reading, and that is a wrong key rather than an error.
        """
        with self.assertRaisesRegex(ValueError, "log_lifetime must be even"):
            leansig_params.LeanSigParams(
                log_lifetime=7,
                dimension=4,
                base=8,
                digits_per_element=8,
                quotient=127,
                target_sum=6,
                max_tries=100,
                parameter_length=5,
                tweak_length=2,
                message_length=9,
                randomness_length=7,
                hash_length=8,
                capacity=9,
            )


class PreparedWindowTest(absltest.TestCase):
    """The two resident bottom trees, and how far they slide."""

    def test_a_fresh_key_prepares_the_first_two_bottom_trees(self) -> None:
        _, secret = _key_pair()

        self.assertEqual((secret.prepared.start, secret.prepared.stop), tuple(PREPARED))

    def test_one_advance_moves_it_by_one_bottom_tree(self) -> None:
        _, secret = _key_pair()

        moved = _SCHEME.advance_preparation(secret)

        self.assertEqual(
            (moved.prepared.start, moved.prepared.stop),
            tuple(PREPARED_AFTER_ONE_ADVANCE),
        )

    def test_it_stops_at_the_end_of_the_lifetime(self) -> None:
        """It moves `MAX_ADVANCES` times, then the next one is a fixed point.

        Returning the key unchanged rather than raising is upstream's answer,
        and it is what lets a caller advance in a loop without bounding the loop
        itself — so the count is asserted by walking until it stops rather than
        by advancing a fixed number of times and checking separately.
        """
        _, secret = _key_pair()

        moves = 0
        while True:
            moved = _SCHEME.advance_preparation(secret)
            if moved.prepared == secret.prepared:
                break
            secret, moves = moved, moves + 1

        self.assertEqual(moves, MAX_ADVANCES)
        self.assertEqual(
            (secret.prepared.start, secret.prepared.stop), tuple(PREPARED_AT_END)
        )

    def test_the_slid_window_rebuilds_a_tree_that_matches_the_original(self) -> None:
        """A bottom tree regenerated from the seed is the one keygen built.

        The two paths differ — keygen slices one dense climb of the whole
        lifetime, `advance_preparation` builds a single tree from its own leaves
        — so this is what holds them to the same nodes. A whole-tree index
        applied as a local one would pass every keygen case above and fail here.
        """
        _, secret = _key_pair()

        # Tree 1 is resident at the start and is the *left* tree after one
        # advance, having been carried across rather than rebuilt. Tree 2 is
        # the first one the slide regenerates.
        moved = _SCHEME.advance_preparation(secret)
        rebuilt = signing.bottom_tree(
            _FAMILY, secret.parameter, _KEY, 2, params=_SCHEME.params
        )

        self.assertEqual(
            harness.to_canonical(moved.right_bottom_tree.root),
            harness.to_canonical(rebuilt.root),
        )
        self.assertEqual(
            harness.to_canonical(moved.left_bottom_tree.root),
            harness.to_canonical(secret.right_bottom_tree.root),
        )


class SignTest(parameterized.TestCase):
    """Every signature, byte for byte, and then through the verifier."""

    @parameterized.named_parameters(*[(v.name, v) for v in SIGN_VECTORS])
    def test_it_matches_upstream(self, vector: SignVector) -> None:
        _, secret = _key_pair()
        message = harness.bytes_of(vector.message)

        signature = _SCHEME.sign(
            _for_slot(secret, vector.slot), message, position=vector.slot
        )

        self.assertEqual(harness.hex_of(signature), vector.signature)

    @parameterized.named_parameters(*[(v.name, v) for v in SIGN_VECTORS])
    def test_the_search_settles_where_upstream_does(self, vector: SignVector) -> None:
        """The codeword the accepted draw encodes to, and its attempt number.

        The signature bytes already pin the randomness, so this is not a second
        gate on the same value — it is what says *which* draw that was, and it
        reads the search directly rather than through a signature.

        **It is also what pins "the first landing draw, not any".** The block is
        128 candidates and about one in forty-nine lands at `TEST`, so a pass
        holds two or three of them; a search that returned the last of a block
        rather than the first would produce a perfectly valid signature at some
        other attempt, and the randomness compared here would not be
        `attempt`'s. That is asserted rather than assumed — `test_a_pass_holds
        _more_than_one_landing_draw` is what keeps the case honest if a
        regenerated vector set ever lands alone in its block.
        """
        _, secret = _key_pair()
        secret = _for_slot(secret, vector.slot)
        message = bytes.fromhex(vector.message)

        randomness, digits = signing.search(secret, vector.slot, message)

        self.assertEqual([int(d) for d in np.asarray(digits)], list(vector.codeword))
        self.assertEqual(sum(vector.codeword), _SCHEME.params.target_sum)
        self.assertEqual(
            harness.to_canonical(randomness),
            harness.to_canonical(
                prf.randomness(
                    _KEY,
                    vector.slot,
                    message,
                    [vector.attempt],
                    params=_SCHEME.params,
                )[0]
            ),
        )

    def test_the_search_takes_the_first_landing_draw(self) -> None:
        """Across the set, `attempt` is the *first* draw of the block to land.

        `test_the_search_settles_where_upstream_does` pins which draw the search
        returned; this pins that it is the earliest one that could have been
        returned, which is the half a signature's bytes cannot show. Both
        matter because a signer that took the last landing candidate of a pass
        instead would produce a perfectly valid signature upstream disagrees
        with byte for byte.

        Measured over the block the search actually tries, and the count is
        asserted alongside: at `TEST` about one draw in forty-nine lands, so a
        128-candidate block usually holds two or three — but not always. Five of
        the six vectors here hold more than one and `slot31` holds exactly one,
        which is why the discriminating claim is made over the set rather than
        per vector. A regenerated set in which *every* block held one landing
        draw would stop separating "first" from "last", and fails here.
        """
        _, secret = _key_pair()
        params = _SCHEME.params
        counters = range(signing._GRIND_BLOCK)
        blocks = []
        for vector in SIGN_VECTORS:
            message = bytes.fromhex(vector.message)
            operands = [
                harness.broadcast(operand, len(counters))
                for operand in (
                    encoding.encode_message(message, params=params),
                    secret.parameter,
                    encoding.encode_epoch(vector.slot, params=params),
                )
            ]
            _, accepted = encoding.codewords(
                operands[0],
                operands[1],
                operands[2],
                prf.randomness(_KEY, vector.slot, message, counters, params=params),
                params=params,
            )
            landed = np.flatnonzero(np.asarray(accepted)).tolist()
            self.assertEqual(landed[0], vector.attempt, msg=vector.name)
            blocks.append(len(landed))

        self.assertGreater(max(blocks), 1)

    @parameterized.named_parameters(*[(v.name, v) for v in SIGN_VECTORS])
    def test_the_verifier_accepts_it(self, vector: SignVector) -> None:
        public, secret = _key_pair()
        message = harness.bytes_of(vector.message)

        signature = _SCHEME.sign(
            _for_slot(secret, vector.slot), message, position=vector.slot
        )

        self.assertTrue(
            bool(
                _SCHEME.verify(
                    public[None, :],
                    message[None, :],
                    np.asarray(signature)[None, :],
                    position=[vector.slot],
                )[0]
            )
        )

    def test_the_whole_set_verifies_in_one_batch(self) -> None:
        """One call over every slot, which is what the seam is shaped for."""
        public, secret = _key_pair()
        messages = np.stack([harness.bytes_of(v.message) for v in SIGN_VECTORS])
        signatures = np.stack(
            [
                np.asarray(
                    _SCHEME.sign(
                        _for_slot(secret, v.slot),
                        messages[index],
                        position=v.slot,
                    )
                )
                for index, v in enumerate(SIGN_VECTORS)
            ]
        )

        verdicts = _SCHEME.verify(
            np.broadcast_to(public, (len(SIGN_VECTORS), public.size)),
            messages,
            signatures,
            position=[v.slot for v in SIGN_VECTORS],
        )

        self.assertEqual(
            [bool(v) for v in np.asarray(verdicts)], [True] * len(SIGN_VECTORS)
        )

    def test_signing_twice_gives_one_signature(self) -> None:
        """Determinism in `(key, slot, message)`, which the PRF is what buys."""
        _, secret = _key_pair()
        message = harness.bytes_of(SIGN_VECTORS[0].message)

        first = _SCHEME.sign(secret, message, position=0)
        second = _SCHEME.sign(secret, message, position=0)

        self.assertEqual(harness.hex_of(first), harness.hex_of(second))

    def test_a_signature_does_not_carry_its_slot(self) -> None:
        """Two slots over one message differ, and neither says which it is.

        The slot is off the wire by design, which is why `verify` takes it as a
        seam field — so what separates these two signatures is every hash they
        made, not a label a verifier could read back.
        """
        _, secret = _key_pair()
        message = harness.bytes_of(SIGN_VECTORS[0].message)

        at_zero = _SCHEME.sign(secret, message, position=0)
        at_one = _SCHEME.sign(secret, message, position=1)

        self.assertNotEqual(harness.hex_of(at_zero), harness.hex_of(at_one))
        self.assertEqual(len(harness.hex_of(at_zero)), len(harness.hex_of(at_one)))


class RefusalTest(absltest.TestCase):
    """What a signer will not do quietly."""

    def setUp(self) -> None:
        super().setUp()
        self.public, self.secret = _key_pair()
        self.message = harness.bytes_of(bytes(range(32)).hex())

    def test_a_slot_outside_the_prepared_window(self) -> None:
        """Served would mean deriving a bottom tree inside a signature."""
        with self.assertRaisesRegex(ValueError, "advance_preparation"):
            _SCHEME.sign(self.secret, self.message, position=PREPARED[1])

    def test_a_slot_past_the_lifetime(self) -> None:
        with self.assertRaisesRegex(ValueError, r"slots \[0, 256\)"):
            _SCHEME.sign(self.secret, self.message, position=256)

    def test_a_message_of_the_wrong_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "32-byte root"):
            _SCHEME.sign(self.secret, self.message[:31], position=0)

    def test_a_key_from_another_preset(self) -> None:
        """A `PROD` scheme cannot sign with a `TEST` key.

        Every digest is tweaked by a position whose packing depends on the
        preset, so the result would be wrong rather than rejected — and the
        shapes line up, because the two presets share every length a signature
        is made of except the codeword.
        """
        with self.assertRaisesRegex(ValueError, "different preset"):
            leansig.named("prod").sign(self.secret, self.message, position=0)


class WiringTest(absltest.TestCase):
    """The decisions a self-consistent wrong signer would make differently."""

    def setUp(self) -> None:
        super().setUp()
        self.public, self.secret = _key_pair()
        self.vector = SIGN_VECTORS[0]
        self.message = harness.bytes_of(self.vector.message)

    def _verify(self, signature: object, slot: int) -> bool:
        return bool(
            _SCHEME.verify(
                self.public[None, :],
                self.message[None, :],
                np.asarray(signature, dtype=np.uint8)[None, :],
                position=[slot],
            )[0]
        )

    def test_the_path_halves_do_not_commute(self) -> None:
        """Bottom siblings first, then top — swapped, the climb misses the root.

        The two halves are the same width at every preset, so a swap is a
        signature of the right length that rebuilds a different root. Nothing
        but a vector or this catches it.
        """
        half = _SCHEME.params.log_lifetime // 2
        path = signing.combined_path(self.secret, self.vector.slot)

        swapped = fnp.concatenate([path[half:], path[:half]])

        self.assertEqual(np.asarray(path).shape, np.asarray(swapped).shape)
        self.assertNotEqual(
            harness.to_leanspec_rows(path), harness.to_leanspec_rows(swapped)
        )

    def test_a_chain_released_one_step_late_is_rejected(self) -> None:
        """The digit is where the walk stops, and the verifier finishes it.

        Walking one step further releases a value the verifier then walks
        `base - 1 - digit` steps from, landing past the chain end — so the leaf
        is wrong. This is the mutation that separates a released value from a
        chain *end*.
        """
        signature = np.asarray(
            _SCHEME.sign(self.secret, self.message, position=self.vector.slot)
        )
        siblings, randomness, hashes, ok = ssz.decode_signature(
            signature, params=_SCHEME.params
        )
        self.assertTrue(bool(ok))

        walked = signing.release(
            _FAMILY,
            self.secret,
            self.vector.slot,
            np.asarray(self.vector.codeword, dtype=np.uint32) + 1,
        )
        mutated = ssz.encode_signature(
            siblings, randomness, walked, params=_SCHEME.params
        )

        self.assertTrue(self._verify(signature, self.vector.slot))
        self.assertFalse(self._verify(mutated, self.vector.slot))

    def test_a_signature_made_at_another_slot_is_rejected(self) -> None:
        """Offered as slot 0, a slot-1 signature fails — the tweaks differ."""
        elsewhere = _SCHEME.sign(self.secret, self.message, position=1)

        self.assertFalse(self._verify(elsewhere, 0))


if __name__ == "__main__":
    absltest.main()
