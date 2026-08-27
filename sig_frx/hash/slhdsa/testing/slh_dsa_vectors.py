# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The published SLH-DSA vectors, grouped by the operation each case belongs to.

FIPS 205 defines five operations and ACVP publishes vectors for all of them: key
generation, and then signing and verifying under the pure external interface, the
pre-hash external interface, and §9's internal interface — the last three in both
the deterministic and the hedged mode. Splitting a set into those, driving each
through the shared harness, and stating what the files should hold is
[`vector_sets.py`](../../../testing/vector_sets.py)'s, because FIPS 204 publishes
the same shape. What is here is the part only FIPS 205 answers.

Two things bound what runs, and both are stated as data rather than left to a
filter somewhere:

- `CONSTRUCTIBLE_SETS` — the parameter sets the two builders cover, which is all
  twelve Table 2 publishes.
- `PRE_HASHES` — the pre-hash functions this scheme may be driven with. ACVP
  exercises twelve and `prehash` provides five, so this is the boundary that
  still drops cases. A pre-hash case names its function inside the message it
  signs, so the rest cannot be approximated by a stand-in.

`excluded_by_reason` reports what those two leave out, so a test asserts the size
of its own coverage gap instead of silently shrinking.

The third thing only FIPS 205 answers is how a public key comes off a secret one,
which is what lets an operation whose verification cases are all failures be
handed an accepted one from `sigGen`. Here it is a slice — Figure 15's secret key
ends in Figure 16's public key — where the lattice scheme has to recompute one.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from python.runfiles import Runfiles

from sig_frx import prehash
from sig_frx.hash.slhdsa import slh_dsa
from sig_frx.signature import Signature
from sig_frx.testing import kat, vector_sets
from sig_frx.testing.vector_sets import Operation

_RUNFILES = Runfiles.Create()

# All twelve of Table 2. Both families reach every security category: §11.1 with
# SHAKE256 alone, and §11.2 with SHA-256 at category 1 (§11.2.1) or SHA-256 and
# SHA-512 together at categories 3 and 5 (§11.2.2).
CONSTRUCTIBLE_SETS = (
    *slh_dsa.SHA2_PARAMETER_SETS,
    *slh_dsa.SHAKE_PARAMETER_SETS,
)

# Of the functions `prehash` provides, this is the one the published cases select
# for the sets here. §10.2.2 restricts SHA-256 to security category 1, which is a
# subset of the sets `sha2` builds — but that restriction is a pairing rule for a
# deployment, and a case that pairs them is still one this repo computes
# faithfully, so the boundary here is which *functions* exist rather than which
# pairings are approved.
PRE_HASHES: dict[str, Callable[[], prehash.PreHash]] = {
    "SHA2-256": slh_dsa.sha256_pre_hash,
}

# No `parameter_set_reason`: its own default says a scheme that builds every set
# it has vectors for never reaches that bucket, and this one does.
COVERAGE = vector_sets.Coverage(
    parameter_sets=CONSTRUCTIBLE_SETS,
    pre_hashes=PRE_HASHES,
)


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


def _params(parameter_set: str) -> slh_dsa.SlhDsaParams:
    """Table 2's row, out of whichever half of it names the set."""
    if parameter_set in slh_dsa.SHAKE_PARAMETER_SETS:
        return slh_dsa.SHAKE_PARAMETER_SETS[parameter_set]
    return slh_dsa.SHA2_PARAMETER_SETS[parameter_set]


def public_key_from_secret(parameter_set: str, secret_key: bytes) -> bytes:
    """`pk = PK.seed ‖ PK.root` — the second half of `sk`, Figures 15 and 16.

    A hash-based secret key ends in the public key it was generated with, so
    there is no arithmetic here to get wrong and nothing to certify a derivation
    against. What holds the slice up is published bytes all the same:
    `slh_dsa_kat_test` requires it of every key pair `keyGen` publishes, which is
    where a layout read wrongly would fail.
    """
    return secret_key[_params(parameter_set).public_key_size :]


def accepted_case(
    operation: Operation, vectors: list[kat.KatVector]
) -> kat.KatVector | None:
    """The accepted case `operation` needs and its verification cases lack.

    ACVP draws every sigVer case's pre-hash function at random from twelve and
    publishes mostly deliberate failures, so the single function these sets are
    approved to use draws one case per set and it is usually a rejection. The pure
    and internal groups are unaffected — fourteen cases and two accepted ones
    apiece — so this reaches the pre-hash operations and, in the pinned set, only
    them. Which ones is a property of the draw rather than of the scheme, so an
    operation is asked whether it needs a case instead of being listed as needing
    one.
    """
    if not vector_sets.needs_an_accepted_case(vectors):
        return None
    return vector_sets.accepted_case(
        operation, runnable_groups("sigGen"), public_key_from_secret
    )


def implementation(
    operation: Operation, scheme: vector_sets.VariantScheme | None = None
) -> Signature:
    """The thing that performs `operation` — the scheme, or an adapter over it.

    `scheme` stands in for the instance that would be built, which is how a test
    asks what the gate does with one broken in a particular way. The adapter
    around it is still this module's, so a substitute is driven exactly as the
    real instance is rather than through a second path written beside it.
    """
    if scheme is None:
        build = (
            slh_dsa.shake
            if operation.parameter_set in slh_dsa.SHAKE_PARAMETER_SETS
            else slh_dsa.sha2
        )
        scheme = build(
            operation.parameter_set, deterministic=bool(operation.deterministic)
        )
    return vector_sets.implementation(operation, scheme, COVERAGE)


def check(
    operation: Operation,
    vectors: list[kat.KatVector],
    *,
    scheme: vector_sets.VariantScheme | None = None,
) -> None:
    """Gate one operation's cases through the shared harness."""
    kat.check(
        implementation(operation, scheme),
        vectors,
        interface=operation.interface,
        pre_hash=operation.pre_hash,
        accepted_case=accepted_case(operation, vectors),
    )
