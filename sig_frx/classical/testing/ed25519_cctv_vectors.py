# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""C2SP's Ed25519 interoperability vectors, and each rule's accept set.

The set is built to make implementations disagree: every case puts an edge
case into `A` or `R` — a low-order point, one of its non-canonical
encodings, a low-order residue that only a cofactored equation absorbs — and
carries the flags saying which. So there is no `valid` field and there
cannot be one, because the verdict is a function of the verification rule,
which is the entire thing under test.

That is also why these do not route through [`kat.py`](../../testing/kat.py).
The harness normalizes a published format into one record with one expected
verdict and drives a scheme through it; here one record has three verdicts,
one per construction. RFC 8032 §7.1 remains the scheme's known-answer gate
and goes through the harness ([`eddsa_test.py`](eddsa_test.py)) — it just
cannot gate a rule, every case in it being an honest signature that all
three rules accept.

## The accept sets are stated, not measured

A rule's `accepts` below is a predicate over the published flags, so the
test asserts a *specification* rather than a snapshot of current behaviour.
Two of the three are upstream's own, from the `ed25519vectors_test.go`
pinned at the same commit as the data; the third is its README's sentence
about `verify_strict`, plus the one consequence a cofactorless equation
always has. `expected_accepts` is pinned alongside so that a regenerated set
which changes the population — new flag classes, a different draw — fails
loudly instead of asserting a predicate over a set that moved underneath it.
"""

from __future__ import annotations

import collections
import functools
import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from python.runfiles import Runfiles

from sig_frx.classical.eddsa import consensus, ed25519
from sig_frx.signature import Signature

_RUNFILES = Runfiles.Create()

# Every flag the pinned set defines. A regenerated set that adds a class is
# an error rather than a case that quietly falls through the predicates —
# the predicates are written over exactly these names.
FLAGS = frozenset(
    {
        "low_order_A",
        "low_order_R",
        "low_order_component_A",
        "low_order_component_R",
        "low_order_residue",
        "non_canonical_A",
        "non_canonical_R",
        "reencoded_k",
    }
)

# The pinned file's size. Every count below is against this population.
TOTAL = 914


@dataclass(frozen=True)
class CctvVector:
    """One published case: the inputs, and what edge cases it exercises."""

    number: int
    public_key: bytes
    message: bytes
    signature: bytes
    flags: frozenset[str]


@dataclass(frozen=True)
class Ruleset:
    """A construction, the accept set it claims, and that set's size."""

    scheme: Signature
    accepts: Callable[[frozenset[str]], bool]
    expected_accepts: int


def _rfc_8032_accepts(flags: frozenset[str]) -> bool:
    """RFC 8032 as `Ed25519` reads it.

    The README states the required refusals — "RFC8032 and FIPS 186-5
    require rejecting non_canonical_A and non_canonical_R, allow both
    rejecting or accepting low_order_residue depending on what formula is
    used, and are silent on the rest" — and the cofactorless equation
    settles the one it leaves open: a residue the equation does not multiply
    away is a residue it fails on.
    """
    return (
        "non_canonical_A" not in flags
        and "non_canonical_R" not in flags
        and "low_order_residue" not in flags
    )


def _verify_strict_accepts(flags: frozenset[str]) -> bool:
    """ed25519-dalek's `verify_strict`.

    Per the README, it "rejects any low_order_A and low_order_R vectors (and
    by extension all other flags except low_order_component_A and
    low_order_component_R)". The equation stays cofactorless, so a low-order
    residue still fails — and that one a point blocklist could not have
    reached, the components in question carrying a prime-order part as well.
    """
    return (
        "low_order_A" not in flags
        and "low_order_R" not in flags
        and "low_order_residue" not in flags
    )


def _zip_215_accepts(flags: frozenset[str]) -> bool:
    """ZIP-215, quoting upstream's own test verbatim.

    "ZIP 215 rules accept all vectors where k is computed from the provided
    R encoding. They reject reencoded_k vectors unless the public key has
    low order, in which case k does not matter."
    """
    return "reencoded_k" not in flags or "low_order_A" in flags


RULESETS = {
    "Ed25519": Ruleset(ed25519.Ed25519(), _rfc_8032_accepts, 172),
    "Ed25519Strict": Ruleset(consensus.Ed25519Strict(), _verify_strict_accepts, 43),
    "Ed25519Zip215": Ruleset(consensus.Ed25519Zip215(), _zip_215_accepts, 826),
}


@functools.cache
def load() -> tuple[CctvVector, ...]:
    """The published set, in file order."""
    path = _RUNFILES.Rlocation("cctv_ed25519_vectors/file/ed25519vectors.json")
    assert path is not None
    with open(path, encoding="utf-8") as handle:
        published = json.load(handle)
    vectors = []
    for case in published:
        # Go writes an empty flag list as `null`; a name outside FLAGS means
        # the set grew a class these predicates were never written against.
        flags = frozenset(case["flags"] or ())
        unknown = flags - FLAGS
        if unknown:
            raise ValueError(f"case {case['number']}: unknown flags {sorted(unknown)}")
        vectors.append(
            CctvVector(
                number=case["number"],
                public_key=bytes.fromhex(case["key"]),
                # The message is published as text, not as hex.
                message=case["msg"].encode(),
                signature=bytes.fromhex(case["sig"]),
                flags=flags,
            )
        )
    return tuple(vectors)


def by_message_length() -> list[tuple[CctvVector, ...]]:
    """The set split into the batches `verify` can take, shortest first.

    A batch axis needs one static shape and the set varies its message
    length to drive `k` — so the split is by message length, and it is the
    whole set rather than a sample of it.
    """
    grouped: dict[int, list[CctvVector]] = collections.defaultdict(list)
    for vector in load():
        grouped[len(vector.message)].append(vector)
    return [tuple(grouped[length]) for length in sorted(grouped)]


def as_arrays(
    batch: tuple[CctvVector, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One batch as `(public_key, message, signature)`, `[B, ...]` uint8.

    Every field is fixed-width within a batch — 32 and 64 bytes by the
    encoding, the message by the split `by_message_length` made — so the
    rows concatenate and reshape rather than needing a pad.
    """

    def stack(values: list[bytes]) -> np.ndarray:
        return np.frombuffer(b"".join(values), dtype=np.uint8).reshape(len(batch), -1)

    return (
        stack([v.public_key for v in batch]),
        stack([v.message for v in batch]),
        stack([v.signature for v in batch]),
    )
