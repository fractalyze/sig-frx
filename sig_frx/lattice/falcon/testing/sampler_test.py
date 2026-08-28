# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""§4.4's sampler, held to the archive's own per-call traces.

Four layers, each gated on its own output rather than on the integer that comes
out of the last one — which is what the traces are for, and why the sampler is
not gated on `SamplerZ` alone.
"""

from __future__ import annotations

import math
from unittest import mock

from absl.testing import absltest, parameterized

from sig_frx.lattice.falcon import sampler
from sig_frx.lattice.falcon.testing import sampler_vectors

_DEGREES = (512, 1024)

# Algorithm 15 line 7's `1/(2σ0²)`, restated here rather than imported: the
# module's copy is what this is checking.
_INV_TWICE_SIGMA0_SQUARED = 0.15086504887537272


def _shifted_exponential(x: float, ccs: float) -> int:
    """`BerExp`'s `z` — Algorithm 14 lines 1-4, transcribed around `approx_exp`.

    The specification's own steps rather than a call into `ber_exp`, which
    consumes randomness and returns only the bit; this is the intermediate the
    trace prints.
    """
    scale = int(x * (1 / math.log(2)))
    remainder = x - scale * math.log(2)
    return (((sampler.approx_exp(remainder, ccs) << 1) - 1) & (2**64 - 1)) >> min(
        scale, 63
    )


# Table 3.1's `pdt`, scaled by `2^72`. Transcribed from the specification beside
# the `RCDT` the module carries, so the two can be held to the identity that
# defines one from the other.
_PDT = (
    1697680241746640300030,
    1459943456642912959616,
    928488355018011056515,
    436693944817054414619,
    151893140790369201013,
    39071441848292237840,
    7432604049020375675,
    1045641569992574730,
    108788995549429682,
    8370422445201343,
    476288472308334,
    20042553305308,
    623729532807,
    14354889437,
    244322621,
    3075302,
    28626,
    197,
    1,
)


class TableTest(absltest.TestCase):
    """`RCDT` against the definition Table 3.1 states for it."""

    def test_rcdt_is_the_reverse_cumulative_of_pdt(self) -> None:
        """`RCDT[i] = Σ_{j>i} pdt[j]`, computed the way (3.33) writes it.

        The module's table is transcribed, so this is the check that it was
        transcribed correctly — and it is a real one rather than a restatement,
        because `pdt` and `RCDT` are printed as separate columns of Table 3.1
        and a digit lost from either does not agree with the other.
        """
        self.assertEqual(
            sampler.RCDT,
            tuple(sum(_PDT[j] for j in range(i + 1, len(_PDT))) for i in range(18)),
        )

    def test_the_distribution_is_normalized(self) -> None:
        """`Σ pdt = 2^72`, so `χ` is a distribution and not merely a table."""
        self.assertEqual(sum(_PDT), 2**72)

    def test_the_support_ends_where_the_table_does(self) -> None:
        """A draw under every entry gives 18, the largest value `χ` supports."""
        self.assertEqual(sampler.base_sampler(lambda n: bytes(n)), 18)


class ByteOrderTest(absltest.TestCase):
    """The one place a correct implementation can still read the vectors wrong.

    Algorithm 12's `u ← UniformBits(72)` is nine bytes read
    most-significant-first. The reference implementation reads a little-endian
    `uint64` plus a byte and splits them into three 24-bit limbs, which is a
    different map from this byte string to a sample — its randomness is a
    ChaCha20 buffer rather than a published trace, so both are right for their
    own source. Pinned because the two agree often enough that a spot check
    would miss it, and because nothing else in this file would fail *first* if
    the module were changed to the other reading.
    """

    def test_the_published_draw_reads_big_endian(self) -> None:
        published = bytes.fromhex("0fc5442ff043d66e91")
        self.assertEqual(sampler.base_sampler(sampler_vectors.cursor(published)), 3)

    def test_the_reference_limb_reading_would_disagree_here(self) -> None:
        published = bytes.fromhex("0fc5442ff043d66e91")
        low = int.from_bytes(published[:8], "little")
        limbed = (
            ((low >> 48) | (published[8] << 16)) << 48
            | ((low >> 24) & 0xFFFFFF) << 24
            | (low & 0xFFFFFF)
        )
        as_limbs = sum(1 for threshold in sampler.RCDT if limbed < threshold)
        self.assertNotEqual(as_limbs, 3)


class VectorTest(parameterized.TestCase):
    """Every `SamplerZ` call of the archive's signature, at both degrees."""

    @parameterized.parameters(*_DEGREES)
    def test_the_trace_is_a_whole_signature(self, degree: int) -> None:
        """`2n` calls — one per coordinate of the two-polynomial lattice point.

        Cheap, and it is what turns a parse that silently matched nothing into
        a failure: every assertion below is a loop over these.
        """
        self.assertLen(sampler_vectors.calls(degree), 2 * degree)

    @parameterized.parameters(*_DEGREES)
    def test_base_sampler_matches_every_published_draw(self, degree: int) -> None:
        """Algorithm 12 alone, on the 72 bits the trace says it read."""
        for index, call in enumerate(sampler_vectors.calls(degree)):
            for number, iteration in enumerate(call.iterations):
                with self.subTest(call=index, iteration=number):
                    source = sampler_vectors.cursor(iteration.u.to_bytes(9, "big"))
                    self.assertEqual(sampler.base_sampler(source), iteration.z0)

    @parameterized.parameters(*_DEGREES)
    def test_ccs_matches_the_published_value(self, degree: int) -> None:
        """`ccs = σmin · (1/σ')`, which is what pins `SIGMA_MIN`.

        Bit-exact rather than approximate, deliberately: `σmin = σ/(1.17√q)`
        agrees to about `1e-13`, and a tolerance here would accept the derived
        value, whose `ApproxExp` output then differs in the low bits.
        """
        for index, call in enumerate(sampler_vectors.calls(degree)):
            with self.subTest(call=index):
                self.assertEqual(
                    call.inverse_sigma * sampler.SIGMA_MIN[degree], call.ccs
                )

    def test_the_rounded_sigma_min_is_invisible_in_the_results(self) -> None:
        """Why the check above it is not enough on its own.

        Table 3.3 prints σmin to ten significant digits, and that rounded value
        is what a reader transcribing from the specification reaches for. The
        test above rejects it — but only because it compares `ccs` directly.

        Substituting it changes **every** intermediate this file checks and
        **no** result at all: all 1024 published integers still come back, while
        all 1786 recorded `BerExp` values move. So a sampler gated on
        `SamplerZ`'s output alone — the vector the specification actually
        tabulates — is green on the wrong constant, and stays green until the
        distribution is examined statistically or another implementation
        disagrees.

        That is `testing.md`'s "pin the intermediates beneath them too" as a
        measurement rather than as advice, and it is the argument for this
        file's whole shape. Falcon-512 alone: the point is about the constant,
        not about the degree, and one corpus states it.
        """
        rounded = 1.277833697
        self.assertNotEqual(rounded, sampler.SIGMA_MIN[512])
        calls = sampler_vectors.calls(512)

        with mock.patch.dict(sampler.SIGMA_MIN, {512: rounded}):
            for index, call in enumerate(calls):
                with self.subTest(call=index):
                    self.assertEqual(
                        sampler.sampler_z(
                            call.center,
                            call.inverse_sigma,
                            512,
                            sampler_vectors.cursor(call.randomness),
                        ),
                        call.result,
                    )

        moved = total = 0
        for call in calls:
            fraction = call.center - math.floor(call.center)
            half_inverse_sigma_squared = call.inverse_sigma * call.inverse_sigma * 0.5
            for iteration in call.iterations:
                z = iteration.bit + (2 * iteration.bit - 1) * iteration.z0
                offset = z - fraction
                x = offset * offset * half_inverse_sigma_squared
                x -= (iteration.z0 * iteration.z0) * _INV_TWICE_SIGMA0_SQUARED
                shifted = _shifted_exponential(x, rounded * call.inverse_sigma)
                moved += shifted != iteration.shifted_exponential
                total += 1
        # Every one of them, not merely some: a partial move would mean the
        # constant is only sometimes reached, which is a different claim.
        self.assertEqual(moved, total)
        self.assertGreater(total, len(calls))

    @parameterized.parameters(*_DEGREES)
    def test_approx_exp_matches_every_published_intermediate(self, degree: int) -> None:
        """Algorithms 13 and 14 up to the comparison — `BerExp`'s `z`.

        The trace prints `z` after `BerExp` has doubled `ApproxExp`'s output,
        subtracted one and shifted by `s`, so reproducing it holds the whole
        fixed-point chain: the Horner loop, the `2^63` scalings, the `ln 2`
        decomposition and the saturation.

        Driven from each iteration's own recorded `z0` and sign bit rather than
        by replaying the byte stream. `BerExp` reads a second comparison byte
        about one time in 256, so a replay that assumed one would desynchronise
        there and compare against the wrong iteration.
        """
        for index, call in enumerate(sampler_vectors.calls(degree)):
            fraction = call.center - math.floor(call.center)
            half_inverse_sigma_squared = call.inverse_sigma * call.inverse_sigma * 0.5
            for number, iteration in enumerate(call.iterations):
                with self.subTest(call=index, iteration=number):
                    z = iteration.bit + (2 * iteration.bit - 1) * iteration.z0
                    offset = z - fraction
                    x = offset * offset * half_inverse_sigma_squared
                    x -= (iteration.z0 * iteration.z0) * _INV_TWICE_SIGMA0_SQUARED
                    self.assertEqual(
                        _shifted_exponential(x, call.ccs),
                        iteration.shifted_exponential,
                    )

    @parameterized.parameters(*_DEGREES)
    def test_sampler_z_reproduces_every_published_result(self, degree: int) -> None:
        """The whole of Algorithm 15, driven by the randomness the trace records.

        This is the case the specification's own Table 3.2 states, at the scale
        the archive publishes it: the rejection loop has to accept and reject in
        the same places, or the stream desynchronises and `cursor` raises rather
        than returning a wrong answer.
        """
        for index, call in enumerate(sampler_vectors.calls(degree)):
            with self.subTest(call=index):
                self.assertEqual(
                    sampler.sampler_z(
                        call.center,
                        call.inverse_sigma,
                        degree,
                        sampler_vectors.cursor(call.randomness),
                    ),
                    call.result,
                )

    @parameterized.parameters(*_DEGREES)
    def test_every_byte_of_the_trace_is_consumed(self, degree: int) -> None:
        """The sampler reads exactly the stream the trace recorded, not a prefix.

        Reproducing the results while reading fewer bytes would mean the
        acceptance loop exits early somewhere and the agreement is luck; the
        published stream is exactly what upstream's own call consumed.
        """
        for index, call in enumerate(sampler_vectors.calls(degree)):
            with self.subTest(call=index):
                consumed = 0

                def counting(count: int, call: object = call) -> bytes:
                    nonlocal consumed
                    chunk = call.randomness[consumed : consumed + count]  # type: ignore[attr-defined]
                    consumed += count
                    return chunk

                sampler.sampler_z(call.center, call.inverse_sigma, degree, counting)
                self.assertEqual(consumed, len(call.randomness))


class DegreeTest(absltest.TestCase):
    def test_an_unknown_degree_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a Falcon parameter set"):
            sampler.sampler_z(0.0, 0.6, 256, lambda n: bytes(n))


if __name__ == "__main__":
    absltest.main()
