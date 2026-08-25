# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""SLH-DSA verification under `frx.jit` — the whole batch as one traced program.

The batch-first seam exists so that `B` signatures verify in one call, and tracing
is what makes that one program rather than one program's worth of dispatches per
entry. Reaching it took every value the path carries off the host: the hypertree
index as bytes, addresses built from traced fields, and the message digest left
where `H_msg` produced it.

So the claim here is not that a traced call runs. It is that it returns the eager
verdicts entry for entry — nothing about the arithmetic moved — including the entry
meant to fail, because a verifier that accepts everything reproduces every passing
one. `slh_dsa_kat_test` is what says the arithmetic is right in the first place;
this says tracing did not change it.

Both category 1 parameter sets, because the dimensions are the point: at a real set
`idx_tree` is seven bytes rather than one and the hypertree is 7 or 22 layers rather
than two, which is where a value too wide for a 32-bit lane would show up. The small
set runs first as the fast signal — it traces the same path at a shape that compiles
in a moment.

Its own target rather than a case in `slh_dsa_test`, because the cost is compilation
rather than hashing: the real sets unroll the whole `d`-layer walk, which is most of
this module's runtime and none of what the assembly cases beside it are checking.
"""

from __future__ import annotations

import frx
import numpy as np
from absl.testing import absltest
from hash_frx import Sha256

from sig_frx.hash.slhdsa import slh_dsa
from sig_frx.hash.tweakable import Sha2TweakableHash

# The same small set `slh_dsa_test` uses: two layers of height two, over the `n`
# security category 1 covers.
_SMALL = slh_dsa.SlhDsaParams(n=16, h=4, d=2, a=3, k=4)
_MESSAGE = np.frombuffer(b"the content that gets signed", dtype=np.uint8)

_CATEGORY_1 = ("SLH-DSA-SHA2-128f", "SLH-DSA-SHA2-128s")


def _seed(size: int) -> np.ndarray:
    return np.frombuffer(bytes((i * 13 + 5) % 256 for i in range(size)), dtype=np.uint8)


def _mixed_batch(
    public_key: np.ndarray, signature: np.ndarray, count: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A batch of alternating verdicts — keys, messages, signatures.

    A verdict per entry is what the batch axis claims, and a batch that agrees
    with itself cannot check it: a verifier returning one answer unconditionally
    is right about every entry of one. Alternating puts each failing entry between
    two passing ones and each passing entry between two failing ones, so accepting
    a bad entry and losing a good one are both visible, whichever way a verdict
    would smear along the batch.

    Four is the smallest count that holds both of those, which is why it is the
    default; shorter batches are for the sizes themselves rather than the smear.
    """
    tampered = signature.copy()
    tampered[-1] ^= 1
    return (
        np.stack([public_key] * count),
        np.stack([_MESSAGE] * count),
        np.stack([signature if i % 2 == 0 else tampered for i in range(count)]),
    )


class TracedVerifyTest(absltest.TestCase):
    def _assert_traces_to_the_eager_verdicts(
        self, scheme: slh_dsa.SlhDsa, count: int = 4
    ) -> None:
        public_key, secret_key = (
            np.asarray(part) for part in scheme.keygen(_seed(scheme.params.seed_size))
        )
        batch = _mixed_batch(
            public_key, np.asarray(scheme.sign(secret_key, _MESSAGE)), count
        )

        traced = np.asarray(frx.jit(scheme.verify)(*batch))

        np.testing.assert_array_equal(traced, np.asarray(scheme.verify(*batch)))
        self.assertEqual(
            [bool(verdict) for verdict in traced], [i % 2 == 0 for i in range(count)]
        )

    def test_a_small_set_traces_to_the_eager_verdicts_at_every_batch_size(self) -> None:
        """Every size up to the default, because a compiled size covers only itself.

        The seam takes the batch size from its caller, and nothing in FIPS 205 makes
        one size different from another — but the backend compiler is free to fail on
        a single leading dimension, and it can do so by aborting the process rather
        than raising, which no assertion here would survive to report. The small set
        is what makes holding several sizes cheap; the real sets below trace one.
        """
        scheme = slh_dsa.SlhDsa(
            Sha2TweakableHash(Sha256(), n=_SMALL.n, m=_SMALL.m),
            _SMALL,
            deterministic=True,
        )
        for count in (1, 2, 3, 4):
            with self.subTest(batch=count):
                self._assert_traces_to_the_eager_verdicts(scheme, count)

    def test_each_category_1_set_traces_to_the_eager_verdicts(self) -> None:
        for name in _CATEGORY_1:
            with self.subTest(name):
                self._assert_traces_to_the_eager_verdicts(
                    slh_dsa.sha2(name, deterministic=True)
                )


if __name__ == "__main__":
    absltest.main()
