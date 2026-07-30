# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""SLH-DSA against the bytes NIST published — the check nothing else here makes.

Every other case in this package compares an implementation against a
transcription of the standard, and one author wrote both sides, so a misreading
they share survives all of it. These cases compare against `SLH-DSA-keyGen-FIPS205`
instead, fetched and sha256-pinned in `//MODULE.bazel` rather than committed.

Key generation is a small vector set and a wide gate. `pk = PK.seed ‖ PK.root`, and
`PK.root` is the root of the top layer's XMSS tree, so a keygen that reproduces a
published public key has confirmed the address layouts, the compressed-address
slice, the six §11.2.1 formulas, the WOTS+ chains and their compression, and the
tree tweaks — every reading made in the components beneath this one — from one
comparison.

Ten of the twelve published groups are not runnable here and are named rather than
skipped: the SHAKE sets need SHAKE256, and the SHA-2 categories 3 and 5 need
SHA-512 for `H`, `T_l` and `PRF_msg` (§11.2.2). The signature vector sets are a
separate gate on signing and verification.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest
from python.runfiles import Runfiles

from sig_frx.hashbased import slh_dsa
from sig_frx.testing import kat

_RUNFILES = Runfiles.Create()

# The two of the twelve published groups whose family is §11.2.1's SHA-256-only
# one, which is every parameter set `slh_dsa.sha2` can build.
_RUNNABLE = ("SLH-DSA-SHA2-128s", "SLH-DSA-SHA2-128f")


def _vectors() -> list[kat.KatVector]:
    prompt = _RUNFILES.Rlocation("acvp_slh_dsa_keygen_prompt/file/prompt.json")
    expected = _RUNFILES.Rlocation(
        "acvp_slh_dsa_keygen_expected/file/expectedResults.json"
    )
    assert prompt is not None and expected is not None
    return kat.load_acvp(prompt, expected)


class KeyGenKatTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.vectors = _vectors()

    def test_the_published_keys_are_reproduced(self) -> None:
        for name in _RUNNABLE:
            group = [v for v in self.vectors if v.parameter_set == name]
            with self.subTest(name):
                self.assertNotEmpty(group)
                # `check` compares both keys, and it is the harness every scheme
                # here is gated through rather than a comparison written per set.
                kat.check(slh_dsa.sha2(name), group)

    def test_the_runnable_groups_are_the_ones_the_family_reaches(self) -> None:
        # Which cases ran, stated rather than implied: a set this cannot build is
        # a coverage boundary, and one that silently vanished would read as a pass.
        published = {vector.parameter_set for vector in self.vectors}
        self.assertLen(published, 12)
        self.assertContainsSubset(_RUNNABLE, published)
        for name in published - set(_RUNNABLE):
            with self.subTest(name):
                if name in slh_dsa.SHA2_PARAMETER_SETS:
                    with self.assertRaises(NotImplementedError):
                        slh_dsa.sha2(name)
                else:
                    self.assertNotIn(name, slh_dsa.SHA2_PARAMETER_SETS)

    def test_a_seed_the_standard_did_not_publish_gives_another_key(self) -> None:
        # The published cases all pass under a keygen that ignored its seed and
        # returned a memorized answer per case; this is the case that does not.
        scheme = slh_dsa.sha2(_RUNNABLE[0])
        vector = next(v for v in self.vectors if v.parameter_set == _RUNNABLE[0])
        assert vector.seed is not None and vector.public_key is not None
        tampered = np.frombuffer(vector.seed, dtype=np.uint8).copy()
        tampered[0] ^= 1
        public_key, _ = scheme.keygen(tampered)
        self.assertNotEqual(kat.to_bytes(public_key), vector.public_key)


if __name__ == "__main__":
    absltest.main()
