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
from hash_frx.keccak.byte_hashes import (
    HostShake128,
    HostShake256,
    Shake128,
    Shake256,
)

from sig_frx import hashes
from sig_frx.lattice.mldsa import ml_dsa, sampling

# What `hashes.shake128` / `shake256` are: the values in, the family out.
_Chooser: TypeAlias = Callable[..., hashes.Xof]


class SelectionTest(absltest.TestCase):
    """`shake128` / `shake256` against the namespace of what they are handed."""

    def test_a_host_value_picks_the_host_sponge(self) -> None:
        seed = np.zeros(32, dtype=np.uint8)
        self.assertIs(hashes.shake128(seed), HostShake128)
        self.assertIs(hashes.shake256(seed), HostShake256)

    def test_a_traced_value_picks_the_device_sponge(self) -> None:
        seed = fnp.zeros(32, dtype=fnp.uint8)
        self.assertIs(hashes.shake128(seed), Shake128)
        self.assertIs(hashes.shake256(seed), Shake256)

    def test_one_traced_value_decides_for_the_call(self) -> None:
        """The same rule `namespace` applies: a mixed call is a traced call.

        Hashing the concatenation of a host part and a traced one has to happen
        where the traced part is, since it cannot be read.
        """
        self.assertIs(
            hashes.shake256(np.zeros(4, dtype=np.uint8), fnp.zeros(4, dtype=fnp.uint8)),
            Shake256,
        )

    def test_nothing_at_all_is_a_host_call(self) -> None:
        """A hash over no values is concrete — there is nothing to be traced."""
        self.assertIs(hashes.shake256(), HostShake256)

    def test_a_python_scalar_is_not_a_tracer(self) -> None:
        """The counter `expand_mask` takes is an `int`, and it decides nothing."""
        self.assertIs(hashes.shake256(0, b"bytes", None), HostShake256)


class ConcretePathTest(absltest.TestCase):
    """That signing and key generation reach the host sponge, not merely select it.

    Driven through the schemes rather than through `hashes` directly: the
    selection being right and a call site not using it is exactly the gap a unit
    test of the selector alone leaves open.
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

    def test_keygen_hashes_on_the_host_until_the_transform_has_run(self) -> None:
        """Algorithm 6's four hashes, and the one the rule leaves on the device.

        The first three are over `ξ` and what was expanded from it, which never
        left the host. The fourth is `tr = H(pk)`, and `pk` carries `t1` — which
        came out of `arith.intt`, so it is a device array and hashing it is a
        device hash. That is the rule reaching its limit rather than failing at
        it: `frx.lax.ntt` has no host form, so nothing about the seed's namespace
        can put `t1` anywhere else.

        Pinned as the exact sequence because the interesting content is where the
        boundary falls, and an assertion that merely counted host calls would not
        notice it moving.
        """
        scheme = ml_dsa.named("ML-DSA-44")
        seed = np.arange(32, dtype=np.uint8)
        self.assertEqual(
            self._sponges(lambda: scheme.keygen(seed)),
            [
                HostShake256,  # line 1: ρ ‖ ρ′ ‖ K from ξ
                HostShake128,  # ExpandA, from ρ
                HostShake256,  # ExpandS, from ρ′
                Shake256,  # tr = H(pk), and pk carries t1 from the transform
            ],
        )

    def test_keygen_from_a_traced_seed_stays_on_the_device(self) -> None:
        """The seam takes `ξ` as it arrives, so a lifted seed keeps the tracer."""
        scheme = ml_dsa.named("ML-DSA-44")
        seed = fnp.asarray(np.arange(32, dtype=np.uint8))
        taken = self._sponges(lambda: scheme.keygen(seed))
        self.assertNotEmpty(taken)
        self.assertNotIn(HostShake128, taken)
        self.assertNotIn(HostShake256, taken)

    def test_expand_a_from_a_host_seed_hashes_on_the_host(self) -> None:
        """The widest batch the scheme hashes, and it is still a host call."""
        taken = self._sponges(
            lambda: sampling.expand_a(np.arange(32, dtype=np.uint8), 8, 7)
        )
        self.assertEqual(taken, [HostShake128])

    def test_verification_keeps_the_device_sponge_throughout(self) -> None:
        """Under `vmap` every value is a tracer, so no host row is reachable.

        The property that makes the whole change safe by construction rather than
        by review: a host hash reads the message bytes, so selecting one here
        would raise rather than return a wrong answer.
        """
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
        self.assertNotIn(HostShake128, taken)
        self.assertNotIn(HostShake256, taken)


if __name__ == "__main__":
    absltest.main()
