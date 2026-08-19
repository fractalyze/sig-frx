# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The Wycheproof ECDSA sets, split into what the seam's encoding can carry.

The fixed 64-byte `r ‖ s` encoding is part of the scheme's interface, so a
published case whose signature is another length cannot reach `verify` at all
— the shape refuses it before any code runs. Dropping those silently would
shrink the gate, so the split is data: the caller asserts every dropped case
is a published failure (a wrong length can never be `valid`) and pins the
count, and a regenerated set that moves either fails the expectation instead
of quietly thinning the suite.

The subset selection exists for the merge gate: the full sweep is the
scheduled run's, and the per-PR run takes the first few cases of every
(shape, verdict) combination, so each batch shape and both verdicts stay
represented at a fraction of the tampering cost.
"""

from __future__ import annotations

import functools

from python.runfiles import Runfiles

from sig_frx.classical import secp
from sig_frx.classical.ecdsa import core
from sig_frx.testing import kat

_RUNFILES = Runfiles.Create()

# (runfiles repo, file name, instance label)
SETS = {
    "secp256k1": (
        "wycheproof_ecdsa_secp256k1_p1363",
        "ecdsa_secp256k1_sha256_p1363_test.json",
        "ECDSA-secp256k1-SHA256",
    ),
    "secp256r1": (
        "wycheproof_ecdsa_secp256r1_p1363",
        "ecdsa_secp256r1_sha256_p1363_test.json",
        "ECDSA-secp256r1-SHA256",
    ),
}

# The scheme under test per curve — one home, so the slice and the sweep
# cannot disagree about the instance the vectors gate.
SCHEMES = {
    "secp256k1": core.Ecdsa(secp.SECP256K1, core.SHA256),
    "secp256r1": core.Ecdsa(secp.SECP256R1, core.SHA256),
}

# The split predicate is the scheme's own declared wire size — the seam field
# consumers allocate by — not a constant that could drift from it.
SIGNATURE_SIZE = core.Ecdsa.signature_max_size

# Cases whose signature is not 64 bytes, per curve — all published failures.
# Pinned so a regenerated set that changes the boundary trips an expectation.
DROPPED = {"secp256k1": 18, "secp256r1": 21}


@functools.cache
def load(curve: str) -> tuple[list[kat.KatVector], list[kat.KatVector]]:
    """The set for `curve`, as `(runnable, dropped-by-encoding)`."""
    repo, file_name, label = SETS[curve]
    path = _RUNFILES.Rlocation(f"{repo}/file/{file_name}")
    assert path is not None
    vectors = kat.load_wycheproof_p1363(path, label)
    runnable = [v for v in vectors if len(v.signature or b"") == SIGNATURE_SIZE]
    dropped = [v for v in vectors if len(v.signature or b"") != SIGNATURE_SIZE]
    return runnable, dropped


def subset(vectors: list[kat.KatVector], per_bucket: int) -> list[kat.KatVector]:
    """The first `per_bucket` cases of every (message length, verdict) bucket."""
    buckets: dict[tuple[int, bool], int] = {}
    kept = []
    for vector in vectors:
        key = (len(vector.message or b""), vector.valid)
        buckets[key] = buckets.get(key, 0) + 1
        if buckets[key] <= per_bucket:
            kept.append(vector)
    return kept
