# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The published SLH-DSA vectors, grouped by the operation each case belongs to.

FIPS 205 defines five operations and ACVP publishes vectors for all of them: key
generation, and then signing and verifying under the pure external interface, the
pre-hash external interface, and §9's internal interface — the last three in both
the deterministic and the hedged mode. Only the first is on the `Signature` seam,
so the other two reach `kat.check` through the adapters here, which is what lets
one harness — and its mandatory tampering — gate every operation instead of each
growing its own comparison.

Two things bound what runs, and both are stated as data rather than left to a
filter somewhere:

- `CONSTRUCTIBLE_SETS` — the parameter sets `slh_dsa.sha2` can build. The other
  ten published sets need SHAKE256 or SHA-512.
- `PRE_HASHES` — the pre-hash functions this repo can compute. ACVP exercises
  twelve; hash-frx provides SHA-256, and a pre-hash case names its function inside
  the message it signs, so the rest cannot be approximated by a stand-in.

`excluded_by_reason` reports what those two leave out, so a test asserts the size
of its own coverage gap instead of silently shrinking.
"""

from __future__ import annotations

import collections
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from frx import Array
from frx.typing import ArrayLike
from python.runfiles import Runfiles

from sig_frx.hashbased import slh_dsa
from sig_frx.signature import Signature
from sig_frx.testing import kat

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

# ACVP's name for each pre-hash function it exercises. A case signs the function's
# OID along with its digest, so an entry maps to a `PreHash` constructor carrying
# both rather than to a hash.
PRE_HASHES: dict[str, Callable[[], slh_dsa.PreHash]] = {
    "SHA2-256": slh_dsa.sha256_pre_hash,
}


@dataclass(frozen=True)
class Operation:
    """One (parameter set, interface, pre-hash, signing mode) the vectors publish.

    The unit `kat.check` runs: it fixes the parameter set, the operation and the
    mode, which are exactly the three things one scheme instance answers for.
    """

    parameter_set: str
    interface: str
    pre_hash: str | None
    deterministic: bool | None

    def __str__(self) -> str:
        mode = {True: "deterministic", False: "hedged", None: "either mode"}[
            self.deterministic
        ]
        return (
            f"{self.parameter_set} {self.interface}"
            f"{'' if self.pre_hash is None else f'/{self.pre_hash}'} {mode}"
        )


class InternalInterface:
    """SLH-DSA's §9 interface behind the seam, so one harness drives it too.

    Not a widening of the scheme's own surface: `sign_internal` stays off the seam
    because it signs an unwrapped message, and this exists only so a vector set
    published for it runs through the harness that owns the negative cases. A
    context is refused rather than ignored — the internal interface has no place to
    put one, and ACVP's internal groups carry none.
    """

    def __init__(self, scheme: slh_dsa.SlhDsa) -> None:
        self._scheme = scheme
        self.public_key_size = scheme.public_key_size
        self.secret_key_size = scheme.secret_key_size
        self.signature_max_size = scheme.signature_max_size
        self.deterministic = scheme.deterministic

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InternalInterface):
            return NotImplemented
        return self._scheme == other._scheme

    def __hash__(self) -> int:
        return hash((type(self), self._scheme))

    def keygen(self, seed: ArrayLike) -> tuple[Array, Array]:
        return self._scheme.keygen(seed)

    def sign(
        self,
        secret_key: ArrayLike,
        message: ArrayLike,
        *,
        randomness: ArrayLike | None = None,
        context: ArrayLike | None = None,
    ) -> Array:
        _reject_context(context)
        return self._scheme.sign_internal(secret_key, message, randomness=randomness)

    def verify(
        self,
        public_key: ArrayLike,
        message: ArrayLike,
        signature: ArrayLike,
        *,
        context: ArrayLike | None = None,
    ) -> Array:
        _reject_context(context)
        return self._scheme.verify_internal(public_key, message, signature)


class PreHashVariant:
    """HashSLH-DSA behind the seam, for one pre-hash function.

    The function is fixed at construction because it is part of what gets signed:
    a variant that took it per call would be two operations behind one name, which
    is what keeping `hash_sign` off the seam avoids in the first place.
    """

    def __init__(self, scheme: slh_dsa.SlhDsa, pre_hash: slh_dsa.PreHash) -> None:
        self._scheme = scheme
        self._pre_hash = pre_hash
        self.public_key_size = scheme.public_key_size
        self.secret_key_size = scheme.secret_key_size
        self.signature_max_size = scheme.signature_max_size
        self.deterministic = scheme.deterministic

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PreHashVariant):
            return NotImplemented
        return (self._scheme, self._pre_hash) == (other._scheme, other._pre_hash)

    def __hash__(self) -> int:
        return hash((type(self), self._scheme, self._pre_hash))

    def keygen(self, seed: ArrayLike) -> tuple[Array, Array]:
        return self._scheme.keygen(seed)

    def sign(
        self,
        secret_key: ArrayLike,
        message: ArrayLike,
        *,
        randomness: ArrayLike | None = None,
        context: ArrayLike | None = None,
    ) -> Array:
        return self._scheme.hash_sign(
            secret_key,
            message,
            self._pre_hash,
            randomness=randomness,
            context=context,
        )

    def verify(
        self,
        public_key: ArrayLike,
        message: ArrayLike,
        signature: ArrayLike,
        *,
        context: ArrayLike | None = None,
    ) -> Array:
        return self._scheme.hash_verify(
            public_key, message, signature, self._pre_hash, context=context
        )


def _reject_context(context: ArrayLike | None) -> None:
    if context is not None:
        raise ValueError(
            "the internal interface signs the message as given, so it has nowhere "
            "to put a context; the external one is what takes it"
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
    return [
        vector
        for vector in vectors
        if vector.parameter_set in CONSTRUCTIBLE_SETS
        and (vector.pre_hash is None or vector.pre_hash in PRE_HASHES)
        and not vector.unsupported
    ]


def excluded_by_reason(vectors: list[kat.KatVector]) -> dict[str, int]:
    """How many cases each coverage boundary drops, by reason.

    A test asserts these counts, so a boundary that moves — a new hash arriving, a
    parameter set becoming constructible — shows up as a failing expectation rather
    than as a suite that quietly got smaller.
    """
    counts: collections.Counter[str] = collections.Counter()
    for vector in vectors:
        if vector.unsupported:
            counts["operation nothing here names"] += 1
        elif vector.parameter_set not in CONSTRUCTIBLE_SETS:
            counts["parameter set needs SHA-512"] += 1
        elif vector.pre_hash is not None and vector.pre_hash not in PRE_HASHES:
            counts["pre-hash function hash-frx does not provide"] += 1
    return dict(counts)


def group(vectors: list[kat.KatVector]) -> dict[Operation, list[kat.KatVector]]:
    """Split into the units one scheme instance answers for, in published order."""
    grouped: dict[Operation, list[kat.KatVector]] = {}
    for vector in vectors:
        operation = Operation(
            parameter_set=vector.parameter_set,
            interface=vector.interface,
            pre_hash=vector.pre_hash,
            deterministic=vector.deterministic,
        )
        grouped.setdefault(operation, []).append(vector)
    return grouped


def implementation(operation: Operation) -> Signature:
    """The thing that performs `operation` — the scheme, or an adapter over it."""
    build = (
        slh_dsa.shake
        if operation.parameter_set in slh_dsa.SHAKE_PARAMETER_SETS
        else slh_dsa.sha2
    )
    scheme = build(operation.parameter_set, deterministic=bool(operation.deterministic))
    if operation.interface == "internal":
        if operation.pre_hash is not None:
            raise ValueError(f"{operation}: the internal interface has no pre-hash")
        return InternalInterface(scheme)
    if operation.pre_hash is not None:
        return PreHashVariant(scheme, PRE_HASHES[operation.pre_hash]())
    return scheme


def check(operation: Operation, vectors: list[kat.KatVector]) -> None:
    """Gate one operation's cases through the shared harness."""
    kat.check(
        implementation(operation),
        vectors,
        interface=operation.interface,
        pre_hash=operation.pre_hash,
    )


if TYPE_CHECKING:
    # Both adapters are what the harness takes, so they carry the same pin the
    # scheme does.
    _internal: type[Signature] = InternalInterface
    _prehash: type[Signature] = PreHashVariant
