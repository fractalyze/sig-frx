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

**The other part only FIPS 204 answers is how a public key comes off a secret
one.** An operation whose verification cases are all failures is handed an
accepted one from `sigGen`, and the only input a signing case lacks is the public
key. Here that is Algorithm 6's arithmetic rather than a slice — `t = A·s1 + s2`,
recomputed and rounded — which the secret key's own published `tr` is what
certifies.

**What is refused rather than excluded is the external-mu variant.** Its input is
a pre-computed 64-byte message representative in place of a message, which no
operation here names, so `kat.load_acvp` records it as unsupported and the harness
will not run it. That is a different thing from a coverage boundary: a boundary
drops cases this scheme could run if a hash existed, and a refusal is the harness
declining to report a pass for a case nobody ran
([`testing.md`](../../../../docs/reference/testing.md)). It is also the
single largest slice of what is published — a quarter of `sigGen` and of `sigVer`
— which is why it is named here rather than left to be inferred from a count.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np
from python.runfiles import Runfiles

from sig_frx import prehash
from sig_frx.lattice.mldsa import arith, encoding, ml_dsa, sampling
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


def public_key_from_secret(parameter_set: str, secret_key: bytes) -> bytes:
    """`pk = ρ ‖ t1`, recomputed from `sk` and confirmed against the `tr` in it.

    Algorithm 6 from its fourth line on: `sk` carries `ρ`, `s1` and `s2`, so `A`
    expands and `t = A·s1 + s2` rounds exactly as key generation does it. What
    makes the result evidence rather than this repo vouching for its own input is
    the line above them — `sk` also carries `tr = H(pk, 64)`, published by NIST,
    so a public key that reproduces it is the published one, and a recomputation
    that drifted anywhere fails here instead of quietly gating nothing.

    The comparison hashes with the standard library rather than with hash-frx: the
    sponge under test cannot be the one that certifies the answer.
    """
    params = ml_dsa.PARAMETER_SETS[parameter_set]
    rho, _, tr, s1, s2, _ = encoding.sk_decode(
        np.frombuffer(secret_key, dtype=np.uint8), params.k, params.ell, params.eta
    )
    a_hat = sampling.expand_a(rho, params.k, params.ell)
    products = arith.matrix_vector(a_hat, arith.ntt(arith.to_field(s1)))
    t1, _ = arith.power2round(arith.intt(products) + arith.to_field(s2))
    public_key = kat.to_bytes(encoding.pk_encode(rho, t1))

    published = kat.to_bytes(tr)
    if hashlib.shake_256(public_key).digest(len(published)) != published:
        raise kat.KatError(
            f"{parameter_set}: a public key recomputed from a published secret key "
            f"does not hash to the `tr` that key carries, so it is not the key the "
            f"standard published — an accepted case built on it would gate nothing"
        )
    return public_key


def accepted_case(
    operation: Operation, vectors: list[kat.KatVector]
) -> kat.KatVector | None:
    """The accepted case `operation` needs and its verification cases lack.

    ACVP draws every sigVer case's pre-hash function at random from twelve and
    publishes mostly deliberate failures, so a function that came up once or twice
    can easily have come up rejected every time — which happens here and only
    here, since the pure and internal groups hold fifteen cases and three accepted
    ones apiece. Which functions it happens to is a property of the draw in
    whichever set is pinned rather than of the scheme, so an operation is asked
    whether it needs a case instead of being listed as needing one.
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

    One family, unlike SLH-DSA's two: every set is SHAKE-instantiated, so the
    parameter set and the signing mode are the whole of what an instance fixes.

    `scheme` stands in for the instance that would be built, which is how a test
    asks what the gate does with one broken in a particular way. The adapter
    around it is still this module's, so a substitute is driven exactly as the
    real instance is rather than through a second path written beside it.
    """
    if scheme is None:
        scheme = ml_dsa.named(
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
