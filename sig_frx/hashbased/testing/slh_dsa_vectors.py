# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The published SLH-DSA vectors, grouped by the operation each case belongs to.

FIPS 205 defines five operations and ACVP publishes vectors for all of them: key
generation, and then signing and verifying under the pure external interface, the
pre-hash external interface, and §9's internal interface — the last three in both
the deterministic and the hedged mode. Splitting a set into those, driving each
through the shared harness, and stating what the files should hold is
[`vector_sets.py`](../../testing/vector_sets.py)'s, because FIPS 204 publishes
the same shape. What is here is the part only FIPS 205 answers.

Two things bound what runs, and both are stated as data rather than left to a
filter somewhere:

- `CONSTRUCTIBLE_SETS` — the parameter sets `slh_dsa.sha2` can build. The other
  ten published sets need SHAKE256 or SHA-512.
- `PRE_HASHES` — the pre-hash functions this scheme may be driven with. ACVP
  exercises twelve; hash-frx provides five, and §10.2.2 approves one of those for
  the sets constructible here. A pre-hash case names its function inside the
  message it signs, so the rest cannot be approximated by a stand-in.

`excluded_by_reason` reports what those two leave out, so a test asserts the size
of its own coverage gap instead of silently shrinking.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from python.runfiles import Runfiles

from sig_frx import prehash
from sig_frx.hashbased import slh_dsa
from sig_frx.signature import Signature
from sig_frx.testing import kat, vector_sets
from sig_frx.testing.vector_sets import Operation

_RUNFILES = Runfiles.Create()

# Every SHAKE set, and the SHA-2 sets §11.2.1's SHA-256-only family reaches.
# §11.1 reaches all six SHAKE sets with SHAKE256 alone, at every security
# category; the SHA-2 categories 3 and 5 hash `H`, `T_l` and `PRF_msg` with
# SHA-512 (§11.2.2) and stay out until a SHA-512 `ByteHash` exists.
CONSTRUCTIBLE_SETS = (
    "SLH-DSA-SHA2-128s",
    "SLH-DSA-SHA2-128f",
    *slh_dsa.SHAKE_PARAMETER_SETS,
)

# §10.2.2 restricts SHA-256 to parameter sets claimed in security category 1,
# which is every set `sha2` builds — so of the functions `prehash` provides, this
# is the one these sets are approved to use.
PRE_HASHES: dict[str, Callable[[], prehash.PreHash]] = {
    "SHA2-256": slh_dsa.sha256_pre_hash,
}

COVERAGE = vector_sets.Coverage(
    parameter_sets=CONSTRUCTIBLE_SETS,
    pre_hashes=PRE_HASHES,
    parameter_set_reason="parameter set needs SHA-512",
)

# The sigVer operations that publish no case the standard accepts, one entry
# each. ACVP draws every sigVer case's pre-hash function at random from twelve
# and publishes mostly deliberate failures, so the single function these sets are
# approved to use draws one case per set and it is usually a rejection. The pure
# and internal groups are unaffected — fourteen cases and two accepted ones
# apiece — which is why every entry below is a pre-hash one.
#
# Where it happened, everything `kat.check` derives has nothing to start from and
# the operation is gated on its published verdict alone. That is a real hole: it
# cannot separate a `hash_verify` that rejects for the right reason from one that
# rejects everything. What separates them is `slh_dsa_test`'s round trip, which
# is self-consistency and therefore no evidence about the standard — so this is
# the boundary of what the published bytes gate, stated where a regenerated set
# will trip over it. An accepted case arriving is what deletes the entry, and
# `kat.check` is what makes that a failure rather than a decision nobody
# revisits.
NO_ACCEPTED_VERIFY_CASE: dict[Operation, str] = {
    Operation(parameter_set, "external", pre_hash, None): reason
    for parameter_set, pre_hash, reason in (
        ("SLH-DSA-SHA2-128s", "SHA2-256", "the one case drawn for it is a failure"),
        ("SLH-DSA-SHAKE-128f", "SHA2-256", "the one case drawn for it is a failure"),
        ("SLH-DSA-SHAKE-128s", "SHA2-256", "the one case drawn for it is a failure"),
        ("SLH-DSA-SHAKE-192f", "SHA2-256", "the one case drawn for it is a failure"),
        ("SLH-DSA-SHAKE-192s", "SHA2-256", "the one case drawn for it is a failure"),
        ("SLH-DSA-SHAKE-256s", "SHA2-256", "the one case drawn for it is a failure"),
    )
}


def load(mode: str) -> list[kat.KatVector]:
    """One published set, by ACVP mode name: `keyGen`, `sigGen` or `sigVer`."""
    repo = f"acvp_slh_dsa_{mode.lower()}"
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
    """The thing that performs `operation` — the scheme, or an adapter over it."""
    build = (
        slh_dsa.shake
        if operation.parameter_set in slh_dsa.SHAKE_PARAMETER_SETS
        else slh_dsa.sha2
    )
    scheme = build(operation.parameter_set, deterministic=bool(operation.deterministic))
    return vector_sets.implementation(operation, scheme, COVERAGE)


def check(operation: Operation, vectors: list[kat.KatVector]) -> None:
    """Gate one operation's cases through the shared harness."""
    kat.check(
        implementation(operation),
        vectors,
        interface=operation.interface,
        pre_hash=operation.pre_hash,
        no_accepted_case=NO_ACCEPTED_VERIFY_CASE.get(operation),
    )
