# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Which sponge a namespace calls for, and that the schemes take the one it names.

The byte-equality half of this is already gated elsewhere and deliberately not
repeated: `sampling_test` runs every sampler host and traced against
`fips204_reference`, and `ml_dsa_kat_test` runs the whole scheme against the
published vectors. Both would pass if the selection were inverted, or constant —
a correct hash is a correct hash whichever implementation computes it. So what is
left to pin, and what is here, is the selection itself and the two properties
that make it more than a preference:

- **It reads the values, not the scheme.** One instance signs concretely and
  verifies under a tracer, so a hash fixed per instance would be fixing the
  caller's fact ([`conventions.md`](../../docs/reference/conventions.md)).
- **The concrete path does not lift.** Selecting correctly and then hashing on
  the device anyway is the failure this exists to prevent, and it is invisible to
  every test that only compares bytes. `ExpandA` from a host seed is the case:
  its `k·ℓ` streams are the widest batch the scheme hashes, which is where a
  device dispatch has the best argument it will get.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import TypeAlias

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from hash_frx import Shake128, Shake256

from sig_frx import hashes
from sig_frx.lattice.mldsa import ml_dsa, sampling

# What `hashes.shake128` / `shake256` are: the values in, the family out.
_Chooser: TypeAlias = Callable[..., hashes.Xof]


class SelectionTest(absltest.TestCase):
    """`shake128` / `shake256` name one row each, whatever they are handed.

    The cases sweep the argument shapes the old dispatcher branched on — a host
    array, a tracer, a mixed call, no values at all, a Python scalar — because
    those are exactly the inputs a re-introduced branch would differ on. They
    now all have to give the same answer.
    """

    def test_a_host_value_names_the_device_sponge(self) -> None:
        seed = np.zeros(32, dtype=np.uint8)
        self.assertIs(hashes.shake128(seed), Shake128)
        self.assertIs(hashes.shake256(seed), Shake256)

    def test_a_traced_value_names_the_device_sponge(self) -> None:
        seed = fnp.zeros(32, dtype=fnp.uint8)
        self.assertIs(hashes.shake128(seed), Shake128)
        self.assertIs(hashes.shake256(seed), Shake256)

    def test_a_mixed_call_names_the_device_sponge(self) -> None:
        self.assertIs(
            hashes.shake256(np.zeros(4, dtype=np.uint8), fnp.zeros(4, dtype=fnp.uint8)),
            Shake256,
        )

    def test_no_values_at_all_names_the_device_sponge(self) -> None:
        self.assertIs(hashes.shake256(), Shake256)

    def test_a_python_scalar_names_the_device_sponge(self) -> None:
        """The counter `expand_mask` takes is an `int`, and it decides nothing."""
        self.assertIs(hashes.shake256(0, b"bytes", None), Shake256)


class ConcretePathTest(absltest.TestCase):
    """WHICH sponge each step of a scheme reaches for, in order.

    Driven through the schemes rather than through `hashes` directly: the module
    naming the right row and a call site not using it is exactly the gap a unit
    test of the names alone leaves open. The substrate is no longer a variable,
    but which of the two sponges runs at each step still is, and getting that
    wrong is a wrong signature rather than a slow one.
    """

    def _sponges(self, call: Callable[[], object]) -> list[hashes.Xof]:
        """Every sponge family `call` reaches for, in order.

        The modules import the two helpers by name, so the attribute on each is
        what the call site resolves and what has to be replaced. The choice is
        recorded and passed through unchanged — a spy that decided anything would
        be pinning itself.
        """
        taken: list[hashes.Xof] = []

        def spy(chooser: _Chooser) -> _Chooser:
            def choose(*values: object) -> hashes.Xof:
                family = chooser(*values)
                taken.append(family)
                return family

            return choose

        originals: list[tuple[ModuleType, str, _Chooser]] = [
            (ml_dsa, "shake256", ml_dsa.shake256),
            (sampling, "shake256", sampling.shake256),
            (sampling, "shake128", sampling.shake128),
        ]
        for module, name, original in originals:
            setattr(module, name, spy(original))
        try:
            call()
        finally:
            for module, name, original in originals:
                setattr(module, name, original)
        return taken

    def test_keygen_reaches_the_four_sponges_of_algorithm_6(self) -> None:
        """Algorithm 6's four hashes, in order, and which sponge each is.

        `G` is the 128-bit XOF and serves `ExpandA` alone; everything else is
        `H`. Pinned as the exact sequence because a step reaching for the wrong
        one of the two produces a well-formed key nobody else computes, and an
        assertion that merely counted calls would not notice.
        """
        scheme = ml_dsa.named("ML-DSA-44")
        seed = np.arange(32, dtype=np.uint8)
        self.assertEqual(
            self._sponges(lambda: scheme.keygen(seed)),
            [
                Shake256,  # line 1: ρ ‖ ρ′ ‖ K from ξ
                Shake128,  # ExpandA, from ρ
                Shake256,  # ExpandS, from ρ′
                Shake256,  # tr = H(pk)
            ],
        )

    def test_keygen_from_a_traced_seed_reaches_the_same_sponges(self) -> None:
        """The seam takes `ξ` as it arrives, and the sequence does not move."""
        scheme = ml_dsa.named("ML-DSA-44")
        seed = fnp.asarray(np.arange(32, dtype=np.uint8))
        self.assertEqual(
            self._sponges(lambda: scheme.keygen(seed)),
            [Shake256, Shake128, Shake256, Shake256],
        )

    def test_expand_a_reaches_g_alone(self) -> None:
        """The widest batch the scheme hashes, and the only `G` call in it."""
        taken = self._sponges(
            lambda: sampling.expand_a(np.arange(32, dtype=np.uint8), 8, 7)
        )
        self.assertEqual(taken, [Shake128])

    def test_verification_reaches_the_device_sponge_throughout(self) -> None:
        """Under `vmap` every value is a tracer, and every row takes one."""
        scheme = ml_dsa.named("ML-DSA-44", deterministic=True)
        seed = np.arange(32, dtype=np.uint8)
        public_key, secret_key = scheme.keygen(seed)
        message = np.frombuffer(b"a message", dtype=np.uint8)
        signature = np.asarray(scheme.sign(secret_key, message))
        batch = (
            fnp.asarray(np.asarray(public_key)[None, :]),
            fnp.asarray(message[None, :]),
            fnp.asarray(signature[None, :]),
        )
        taken = self._sponges(lambda: frx.jit(scheme.verify)(*batch))
        self.assertNotEmpty(taken)
        self.assertContainsSubset(set(taken), {Shake128, Shake256})


if __name__ == "__main__":
    absltest.main()
