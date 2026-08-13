# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The published ML-DSA vectors, grouped by the operation each case belongs to.

FIPS 204 defines the same five operations FIPS 205 does, and ACVP publishes
vectors for all of them, so the splitting and the driving are
[`vector_sets.py`](../../../testing/vector_sets.py)'s. What is here is the part
only FIPS 204 answers — and, unusually, it is mostly the *absence* of boundaries:

- **Every published parameter set is constructible.** ML-DSA instantiates every
  function of every set with SHAKE, so there is no second hash family to be
  missing and `CONSTRUCTIBLE_SETS` is Table 1 in full. The reason string
  `Coverage` would use for a set outside it is therefore never reached, which is
  what `excluded_by_reason` returning no such bucket says.
- **Every pre-hash function this repo can compute is approved.** FIPS 204 §5.4
  enumerates three and writes `case …`; the strength footnote pairs a digest with
  a parameter set, but the published sets pair them freely, so the boundary is
  what hash-frx provides and nothing narrower. ACVP exercises twelve and
  `prehash.BY_NAME` holds five, so the other seven are the one real gap.

**What is refused rather than excluded is the external-mu variant.** Its input is
a pre-computed 64-byte message representative in place of a message, which no
operation here names, so `kat.load_acvp` records it as unsupported and the harness
will not run it. That is a different thing from a coverage boundary: a boundary
drops cases this scheme could run if a hash existed, and a refusal is the harness
declining to report a pass for a case nobody ran
([`conventions.md`](../../../../docs/reference/conventions.md)). It is also the
single largest slice of what is published — a quarter of `sigGen` and of `sigVer`
— which is why it is named here rather than left to be inferred from a count.
"""

from __future__ import annotations

from functools import lru_cache

from python.runfiles import Runfiles

from sig_frx import prehash
from sig_frx.lattice.mldsa import ml_dsa
from sig_frx.signature import Signature
from sig_frx.testing import kat, vector_sets
from sig_frx.testing.vector_sets import Operation

_RUNFILES = Runfiles.Create()

# Table 1 in full: `named` builds every set ACVP publishes.
CONSTRUCTIBLE_SETS = tuple(ml_dsa.PARAMETER_SETS)

# Every function `prehash` provides, since FIPS 204 approves all of them for this
# scheme. A pre-hash case signs its function's OID, so the seven ACVP exercises
# that hash-frx does not provide cannot be approximated by a stand-in.
PRE_HASHES = prehash.BY_NAME

COVERAGE = vector_sets.Coverage(
    parameter_sets=CONSTRUCTIBLE_SETS,
    pre_hashes=PRE_HASHES,
)


def load(mode: str) -> list[kat.KatVector]:
    """One published set, by ACVP mode name: `keyGen`, `sigGen` or `sigVer`."""
    repo = f"acvp_ml_dsa_{mode.lower()}"
    prompt = _RUNFILES.Rlocation(f"{repo}_prompt/file/prompt.json")
    expected = _RUNFILES.Rlocation(f"{repo}_expected/file/expectedResults.json")
    assert prompt is not None and expected is not None, mode
    return kat.load_acvp(prompt, expected)


def runnable(vectors: list[kat.KatVector]) -> list[kat.KatVector]:
    """The cases a scheme built here can be gated on, faithfully."""
    return COVERAGE.runnable(vectors)


def excluded_by_reason(vectors: list[kat.KatVector]) -> dict[str, int]:
    """How many cases each coverage boundary drops, by reason."""
    return COVERAGE.excluded_by_reason(vectors)


def group(vectors: list[kat.KatVector]) -> dict[Operation, list[kat.KatVector]]:
    """Split into the units one scheme instance answers for, in published order."""
    return vector_sets.group(vectors)


def operations() -> dict[str, list[Operation]]:
    """Every operation each mode publishes, for every constructible set."""
    return COVERAGE.operations()


@lru_cache(maxsize=None)
def runnable_groups(mode: str) -> dict[Operation, list[kat.KatVector]]:
    """`load`, `runnable` and `group` as the one thing every caller wants.

    Cached because the callers ask per test method, and the answer is a function
    of the mode alone. Callers read the result and do not hold it, which is what
    lets one parse serve all of them.
    """
    return group(runnable(load(mode)))


def implementation(operation: Operation) -> Signature:
    """The thing that performs `operation` — the scheme, or an adapter over it.

    One family, unlike SLH-DSA's two: every set is SHAKE-instantiated, so the
    parameter set and the signing mode are the whole of what an instance fixes.
    """
    scheme = ml_dsa.named(
        operation.parameter_set, deterministic=bool(operation.deterministic)
    )
    return vector_sets.implementation(operation, scheme, COVERAGE)


def check(operation: Operation, vectors: list[kat.KatVector]) -> None:
    """Gate one operation's cases through the shared harness."""
    kat.check(
        implementation(operation),
        vectors,
        interface=operation.interface,
        pre_hash=operation.pre_hash,
    )
