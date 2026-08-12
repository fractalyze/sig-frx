# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""ML-DSA's samplers against FIPS 204's own loops, host and traced.

`sampling.py` reshapes a `while` loop that squeezes until enough candidates
survive into a fixed budget plus a compaction, so what has to be gated is that
the reshaping produced the same stream the standard's loop would have: the same
candidates, in the same order, with the same ones dropped. `fips204_reference`
is that loop, squeezing one byte at a time from `hashlib` with no budget at all,
and every sampler is required to agree with it on every parameter set.

Three things beyond that agreement, because agreement alone would not catch
them:

- **The budget.** It is computed rather than chosen, so it is pinned twice — the
  probability margin holds at the budget returned, and it does not hold one
  block lower. That makes a silent change to the margin a failing test rather
  than a quieter guarantee.
- **The exhausted path.** Driven by handing the samplers a budget too small to
  cover 256 coefficients, which is otherwise a `2^-256` event nobody can wait
  for. It must raise, not return a polynomial padded with rejects.
- **Both namespaces.** A scheme keygen-signs on the host and verifies under a
  tracer, so every case runs as numpy and under `jit`.

What none of it can catch is a misreading of the standard shared by both sides,
since one author wrote both. The published vectors are what close that gap
([`conventions.md`](../../../../docs/reference/conventions.md)), and the
transcription here was additionally compared against an independent ML-DSA
implementation before it was relied on — the record of that comparison is on
[fractalyze/sig-frx#19](https://github.com/fractalyze/sig-frx/issues/19).

The ordering of `Â` is in the same position: `RejNTTPoly` samples an element of
`T_q` directly, so no arrangement of it is internally detectable, and what makes
the absence of a `BitRev8` gather here correct is `arith.ntt`'s by-definition
gate in `arith_test`.
"""

from __future__ import annotations

from typing import Any, TypeAlias
from unittest import mock

import frx
import frx.numpy as fnp
import numpy as np
import zk_dtypes
from absl.testing import absltest, parameterized

from sig_frx.lattice.mldsa import arith, sampling
from sig_frx.lattice.mldsa.testing import fips204_reference as ref

_Lift: TypeAlias = Any

# Every case runs both ways, as `arith_test` does: a host seed for the key
# generation and signing paths, a traced one for verification.
_LIFT = {"host": np.asarray, "traced": fnp.asarray}
_NAMESPACES = tuple(_LIFT.items())

# FIPS 204 Table 1, the fields the samplers read.
_PARAMETER_SETS = (
    {
        "testcase_name": "ML_DSA_44",
        "k": 4,
        "ell": 4,
        "eta": 2,
        "tau": 39,
        "lam": 128,
        "gamma1": 1 << 17,
    },
    {
        "testcase_name": "ML_DSA_65",
        "k": 6,
        "ell": 5,
        "eta": 4,
        "tau": 49,
        "lam": 192,
        "gamma1": 1 << 19,
    },
    {
        "testcase_name": "ML_DSA_87",
        "k": 8,
        "ell": 7,
        "eta": 2,
        "tau": 60,
        "lam": 256,
        "gamma1": 1 << 19,
    },
)


def _seed(width: int, salt: int) -> np.ndarray:
    """A seed of `width` bytes — the samplers take bytes, not a distribution."""
    return np.asarray(
        np.random.default_rng(salt).integers(0, 256, size=width), dtype=np.uint8
    )


def _ints(values: Any) -> list[Any]:
    """A device or host array as Python integers, nested to its rank."""
    return np.asarray(values).tolist()


class BudgetTest(parameterized.TestCase):
    """The candidate budget, which is the whole safety argument of the reshaping.

    The standard's loop cannot run short; a fixed budget can, so the budget is
    sized by an exact binomial tail rather than by convention. These pin the
    sizing from both sides — it is safe, and it is the smallest safe one — so a
    change to either the margin or the acceptance probability fails here rather
    than silently widening the window in which a wrong polynomial comes back.
    """

    @parameterized.named_parameters(
        ("expand_a", 256, sampling._NTT_ACCEPT, sampling._NTT_PER_BLOCK),
        (
            "expand_s_eta_2",
            256,
            sampling._BOUNDED_ACCEPT[2],
            sampling._BOUNDED_PER_BLOCK,
        ),
        (
            "expand_s_eta_4",
            256,
            sampling._BOUNDED_ACCEPT[4],
            sampling._BOUNDED_PER_BLOCK,
        ),
        ("sample_in_ball_tau_60", 60, (256 - 60 + 1, 256), 1),
    )
    def test_is_the_smallest_budget_that_meets_the_margin(
        self, needed: int, accept: tuple[int, int], per_block: int
    ) -> None:
        blocks = sampling._budget(needed, accept, per_block)
        self.assertFalse(
            sampling._shortfall_exceeds_margin(blocks * per_block, needed, accept),
            "the chosen budget does not meet the margin",
        )
        self.assertTrue(
            sampling._shortfall_exceeds_margin(
                (blocks - 1) * per_block, needed, accept
            ),
            "a smaller budget would also have met it, so this one is not minimal",
        )

    def test_the_margin_is_the_strongest_parameter_set_s_strength(self) -> None:
        """`2^-256` is `λ` at ML-DSA-87 (Table 1), not a round number."""
        self.assertEqual(sampling._LOG2_SHORTFALL, 256)

    def test_a_certain_acceptance_needs_no_slack(self) -> None:
        """The tail is exact, so `p = 1` sizes to exactly what is asked for."""
        self.assertEqual(sampling._budget(256, (1, 1), 8), 32)


class ExpandATest(parameterized.TestCase):
    """Algorithm 32 — the public matrix, sampled straight into `T_q`."""

    @parameterized.named_parameters(*_NAMESPACES)
    def test_matches_algorithm_32(self, lift: _Lift) -> None:
        rho = _seed(32, 20)
        got = _ints(sampling.expand_a(lift(rho), 8, 7).astype(np.uint32))
        self.assertEqual(got, ref.expand_a(bytes(rho), 8, 7))

    @parameterized.named_parameters(*_PARAMETER_SETS)
    def test_matches_algorithm_32_at_every_parameter_set(
        self, k: int, ell: int, **_: Any
    ) -> None:
        rho = _seed(32, k)
        got = _ints(sampling.expand_a(rho, k, ell).astype(np.uint32))
        self.assertEqual(got, ref.expand_a(bytes(rho), k, ell))

    def test_the_entries_are_one_batched_call(self) -> None:
        """`k·ℓ` streams trace as one computation, not `k·ℓ` of them.

        The shape is what the seed fans out into, so the check is that the whole
        matrix comes back from one traced call — a driver loop over entries
        would not survive `jit` as a single zone.
        """
        rho = _seed(32, 21)
        compiled = frx.jit(lambda seed: sampling.expand_a(seed, 8, 7))
        self.assertEqual(compiled(fnp.asarray(rho)).shape, (8, 7, arith.N))

    def test_the_entries_are_field_elements_below_q(self) -> None:
        hats = sampling.expand_a(_seed(32, 22), 6, 5)
        self.assertEqual(zk_dtypes.pfinfo(hats.dtype).modulus, arith.Q)
        self.assertTrue((np.asarray(hats.astype(np.uint32)) < arith.Q).all())

    def test_pairs_with_the_transform_without_a_conversion(self) -> None:
        """`Â` is already what `arith` speaks, in dtype and in domain.

        Not a check of the *ordering* — a consistently permuted `Â` round-trips
        and convolves like the right one, which is why `arith_test` pins the
        ordering by direct evaluation and the published vectors close the
        rest. What
        this pins is the seam: no `astype`, no permutation, no reshape between
        the sampler's output and the arithmetic that consumes it.
        """
        hats = sampling.expand_a(_seed(32, 23), 4, 4)
        polynomial = arith.intt(hats)
        self.assertTrue(
            np.array_equal(np.asarray(arith.ntt(polynomial)), np.asarray(hats))
        )


class ExpandSTest(parameterized.TestCase):
    """Algorithm 33 — the secret vectors, in the standard's `[−η, η]` range."""

    @parameterized.named_parameters(*_PARAMETER_SETS)
    def test_matches_algorithm_33(self, k: int, ell: int, eta: int, **_: Any) -> None:
        rho = _seed(64, k)
        s1, s2 = sampling.expand_s(rho, k, ell, eta)
        want1, want2 = ref.expand_s(bytes(rho), k, ell, eta)
        self.assertEqual(_ints(s1), want1)
        self.assertEqual(_ints(s2), want2)

    @parameterized.named_parameters(*_NAMESPACES)
    def test_host_and_traced_agree(self, lift: _Lift) -> None:
        rho = _seed(64, 30)
        s1, s2 = sampling.expand_s(lift(rho), 6, 5, 4)
        want1, want2 = ref.expand_s(bytes(rho), 6, 5, 4)
        self.assertEqual(_ints(s1), want1)
        self.assertEqual(_ints(s2), want2)

    @parameterized.named_parameters(*_PARAMETER_SETS)
    def test_coefficients_are_within_eta(
        self, k: int, ell: int, eta: int, **_: Any
    ) -> None:
        s1, s2 = sampling.expand_s(_seed(64, eta), k, ell, eta)
        for vector in (np.asarray(s1), np.asarray(s2)):
            self.assertTrue((np.abs(vector) <= eta).all())

    def test_the_two_vectors_are_split_not_resampled(self) -> None:
        """`s2`'s streams continue `s1`'s index, which is what Algorithm 33 says.

        The one thing sampling both vectors in a single batch could get wrong,
        and the failure it would produce is a valid-looking key that no other
        implementation agrees with.
        """
        rho = _seed(64, 31)
        _, s2 = sampling.expand_s(rho, 4, 4, 2)
        first_of_s2 = ref.rej_bounded_poly(bytes(rho) + ref.integer_to_bytes(4, 2), 2)
        self.assertEqual(_ints(s2)[0], first_of_s2)

    def test_an_eta_the_standard_does_not_define_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be 2 or 4"):
            sampling.expand_s(_seed(64, 32), 4, 4, 3)


class ExpandMaskTest(parameterized.TestCase):
    """Algorithm 34 — the masking vector, which unpacks rather than rejects."""

    @parameterized.named_parameters(*_PARAMETER_SETS)
    def test_matches_algorithm_34(self, ell: int, gamma1: int, **_: Any) -> None:
        rho = _seed(64, ell)
        for mu in (0, 7, 1000):
            got = _ints(sampling.expand_mask(rho, mu, ell, gamma1))
            self.assertEqual(
                got, ref.expand_mask(bytes(rho), mu, ell, gamma1), f"mu={mu}"
            )

    @parameterized.named_parameters(*_NAMESPACES)
    def test_host_and_traced_agree(self, lift: _Lift) -> None:
        rho = _seed(64, 40)
        got = _ints(sampling.expand_mask(lift(rho), 5, 5, 1 << 19))
        self.assertEqual(got, ref.expand_mask(bytes(rho), 5, 5, 1 << 19))

    def test_mu_wraps_at_two_bytes(self) -> None:
        """`IntegerToBytes(μ, 2)` is `μ mod 2^16` (Algorithm 11), and `μ` is a
        signing counter that a caller could hand over already past it."""
        rho = _seed(64, 41)
        wrapped = _ints(sampling.expand_mask(rho, 1 << 16, 4, 1 << 17))
        self.assertEqual(wrapped, _ints(sampling.expand_mask(rho, 0, 4, 1 << 17)))

    @parameterized.named_parameters(*_PARAMETER_SETS)
    def test_coefficients_are_within_the_gamma1_window(
        self, ell: int, gamma1: int, **_: Any
    ) -> None:
        y = np.asarray(sampling.expand_mask(_seed(64, gamma1 % 97), 3, ell, gamma1))
        self.assertTrue((y >= -gamma1 + 1).all())
        self.assertTrue((y <= gamma1).all())

    def test_a_gamma1_that_is_not_a_power_of_two_is_refused(self) -> None:
        """Algorithm 34 line 1 assumes it, and the bit width is wrong without it."""
        with self.assertRaisesRegex(ValueError, "must be a power of 2"):
            sampling.expand_mask(_seed(64, 42), 0, 4, 3 << 16)


class SampleInBallTest(parameterized.TestCase):
    """Algorithm 29 — the challenge, and the one sampler that rejects in sequence."""

    @parameterized.named_parameters(*_PARAMETER_SETS)
    def test_matches_algorithm_29(self, tau: int, lam: int, **_: Any) -> None:
        rho = _seed(lam // 4, tau)
        self.assertEqual(
            _ints(sampling.sample_in_ball(rho, tau)),
            ref.sample_in_ball(bytes(rho), tau),
        )

    @parameterized.named_parameters(*_NAMESPACES)
    def test_host_and_traced_agree(self, lift: _Lift) -> None:
        rho = _seed(64, 50)
        self.assertEqual(
            _ints(sampling.sample_in_ball(lift(rho), 60)),
            ref.sample_in_ball(bytes(rho), 60),
        )

    @parameterized.named_parameters(*_PARAMETER_SETS)
    def test_has_exactly_tau_nonzero_coefficients_all_pm_one(
        self, tau: int, lam: int, **_: Any
    ) -> None:
        """The property the algorithm's name states, over enough seeds to be a
        property rather than a sample: the swap is what could break it, and it
        breaks by writing twice to one slot."""
        for salt in range(8):
            c = np.asarray(sampling.sample_in_ball(_seed(lam // 4, 100 + salt), tau))
            self.assertEqual(int((c != 0).sum()), tau)
            self.assertEqual(set(np.unique(np.abs(c)).tolist()) - {0}, {1})

    def test_batches_over_signatures_with_vmap(self) -> None:
        """The seam batch verification goes through: one seed per signature.

        The sampler's own axis is over the streams a single seed fans out into,
        so a batch of signatures is `vmap` over it rather than a shape change —
        which is what `conventions.md` asks of every operation that is not
        verification itself.
        """
        seeds = np.stack([_seed(64, 200 + index) for index in range(4)])
        batched = frx.vmap(lambda seed: sampling.sample_in_ball(seed, 60))
        got = np.asarray(frx.jit(batched)(fnp.asarray(seeds)))
        for index, seed in enumerate(seeds):
            self.assertEqual(got[index].tolist(), ref.sample_in_ball(bytes(seed), 60))


class ExhaustionTest(parameterized.TestCase):
    """The path the budget exists to make unreachable, driven by shrinking it.

    A short budget is otherwise a `2^-256` event, so the only way to test what
    happens on it is to build one. What must not happen is a returned
    polynomial: the compaction pads with rejected candidates, which are values
    at or above `q` for `ExpandA` and `⊥`'s slot for `ExpandS` — self-consistent
    garbage that a round trip would never reveal.
    """

    def test_rej_ntt_poly_raises_rather_than_padding(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ran out of candidates"):
            sampling._rej_ntt_poly(np.zeros((2, 34), dtype=np.uint8), blocks=1)

    def test_rej_bounded_poly_raises_rather_than_padding(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ran out of candidates"):
            sampling._rej_bounded_poly(np.zeros((2, 66), dtype=np.uint8), 4, blocks=1)

    @parameterized.named_parameters(
        ("expand_a", lambda: sampling.expand_a(_seed(32, 60), 4, 4)),
        ("expand_s", lambda: sampling.expand_s(_seed(64, 61), 6, 5, 4)),
        ("sample_in_ball", lambda: sampling.sample_in_ball(_seed(32, 62), 39)),
    )
    def test_a_shrunken_budget_is_loud(self, sample: Any) -> None:
        with mock.patch.object(sampling, "_budget", return_value=1):
            with self.assertRaisesRegex(RuntimeError, "ran out of candidates"):
                sample()

    def test_the_message_names_the_budget_and_not_the_seed(self) -> None:
        """A shortfall is a wrong bound, and the message has to say so — the
        seed is the one thing that cannot be at fault."""
        with self.assertRaises(RuntimeError) as raised:
            sampling._rej_ntt_poly(np.zeros((1, 34), dtype=np.uint8), blocks=1)
        self.assertIn("wrong budget rather than an unlucky seed", str(raised.exception))


if __name__ == "__main__":
    absltest.main()
