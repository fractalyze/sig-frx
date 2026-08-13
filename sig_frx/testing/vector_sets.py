# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Published vectors, split into the operations one scheme instance answers for.

[`kat.py`](kat.py) normalizes a published *format* into `KatVector` and drives a
`Signature` through it. What sits between the two is this: a set covers several
operations at once — a standard publishes vectors for every interface it defines
— and `check` runs one. So the cases have to be split by operation, each handed
the thing that performs it, and what a scheme leaves out has to be counted rather
than filtered away silently.

That shape is the same for both FIPS signature standards, and identically so: each
defines a pure external interface, a pre-hash external one and an internal one,
publishes key generation for the first alone, distinguishes the deterministic and
hedged modes in signing but not in verification, and names its cases the same way.
So it lives here rather than once per scheme, and a scheme's own module supplies
only what is genuinely its: which parameter sets it can build, which pre-hash
functions it may use, where its files are, and how to construct an instance.

**A coverage boundary is data, and it is counted.** `Coverage` is the statement of
what a scheme can be gated on, and `excluded_by_reason` reports what each of its
boundaries drops. A test asserts those counts, so a boundary that moves — a hash
arriving, a parameter set becoming constructible — fails an expectation rather
than quietly shrinking the suite.

**`operations` is the constructive inverse of `group`.** One says what the vectors
should hold, built as the product of the constants above; the other reports what
they do. A caller compares them, which is what turns a set that stopped publishing
an operation into a failure instead of a smaller green run.

**The two adapters are why one harness can gate three interfaces.** Only the pure
external operation is on the `Signature` seam; the internal interface and the
pre-hash variant live under each scheme's own name, because a variant that
prepares a different message is a different operation
([`../../docs/reference/conventions.md`](../../docs/reference/conventions.md)).
Wrapping them here is not a widening of any scheme's surface — it exists so that
vectors published for them run through the harness that owns the negative cases,
rather than growing a second comparison each.
"""

from __future__ import annotations

import collections
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from frx import Array
from frx.typing import ArrayLike

from sig_frx import prehash
from sig_frx.signature import Signature
from sig_frx.testing import kat

# The one interface every mode has: the pure external one, no pre-hash.
PURE_EXTERNAL: tuple[str, str | None] = ("external", None)


class VariantScheme(Signature, Protocol):
    """A scheme that implements more of its standard than the seam names.

    The seam, plus the two pairs that are off it. Stated as a Protocol because the
    adapters below need exactly these and nothing about which scheme supplies them
    — and because a scheme that grew one of the pairs under a different name would
    fail here rather than at a vector set that will not run.
    """

    def sign_internal(
        self,
        secret_key: ArrayLike,
        message: ArrayLike,
        *,
        randomness: ArrayLike | None = None,
    ) -> Array: ...

    def verify_internal(
        self, public_key: ArrayLike, messages: ArrayLike, signature: ArrayLike
    ) -> Array: ...

    def hash_sign(
        self,
        secret_key: ArrayLike,
        message: ArrayLike,
        pre_hash: prehash.PreHash,
        *,
        randomness: ArrayLike | None = None,
        context: ArrayLike | None = None,
    ) -> Array: ...

    def hash_verify(
        self,
        public_key: ArrayLike,
        message: ArrayLike,
        signature: ArrayLike,
        pre_hash: prehash.PreHash,
        *,
        context: ArrayLike | None = None,
    ) -> Array: ...


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
        return f"{self.parameter_set} {self._description}"

    @property
    def name(self) -> str:
        """The same operation as an identifier, with the parameter set last.

        What a caller that runs one operation per test names its cases after. The
        field order differs from `__str__`'s deliberately: a sharded run assigns
        cases round-robin over their sorted names, so trailing the parameter set
        keeps one set's operations together in that order — and a set's cases in
        few shards is a set's kernels compiled in few shards.
        """
        return re.sub(
            r"[^0-9A-Za-z]+", "_", f"{self._description} {self.parameter_set}"
        )

    @property
    def _description(self) -> str:
        """Everything but the parameter set: the interface, pre-hash and mode."""
        mode = {True: "deterministic", False: "hedged", None: "either mode"}[
            self.deterministic
        ]
        pre_hash = "" if self.pre_hash is None else f"/{self.pre_hash}"
        return f"{self.interface}{pre_hash} {mode}"


@dataclass(frozen=True)
class _Published:
    """The two axes whose product is what one ACVP mode publishes."""

    signed_interfaces: bool
    # `sigGen` publishes the deterministic and the hedged case separately; the
    # other two modes do not have the distinction — a key pair has no randomness
    # in it, and a signature verifies the same way whichever mode made it.
    signing_modes: tuple[bool | None, ...]


# Which operations each ACVP mode publishes, as both FIPS 204 and FIPS 205
# publish them. Key generation takes a seed rather than a message, so it has the
# pure interface alone and no signing mode.
_MODES: dict[str, _Published] = {
    "keyGen": _Published(signed_interfaces=False, signing_modes=(None,)),
    "sigGen": _Published(signed_interfaces=True, signing_modes=(True, False)),
    "sigVer": _Published(signed_interfaces=True, signing_modes=(None,)),
}


@dataclass(frozen=True)
class Coverage:
    """What a scheme can be gated on, and what each boundary it has leaves out."""

    # The published parameter sets this scheme can build. The rest are excluded
    # under `parameter_set_reason`.
    parameter_sets: tuple[str, ...]
    # The pre-hash functions this scheme may be driven with, by the name a case
    # selects. A pre-hash case signs its function's OID, so one outside this
    # cannot be approximated by a stand-in — it is excluded, not substituted.
    pre_hashes: Mapping[str, Callable[[], prehash.PreHash]] = field(
        default_factory=dict
    )
    # Why the published sets outside `parameter_sets` are out. A scheme that can
    # build every set it has vectors for never reaches this.
    parameter_set_reason: str = "parameter set this repo cannot build"

    @property
    def interfaces(self) -> tuple[tuple[str, str | None], ...]:
        """How a signed case reaches the scheme, as `(interface, pre-hash)`."""
        return (
            PURE_EXTERNAL,
            *(("external", name) for name in self.pre_hashes),
            ("internal", None),
        )

    def runnable(self, vectors: list[kat.KatVector]) -> list[kat.KatVector]:
        """The cases this scheme can be gated on, faithfully."""
        return [
            vector
            for vector in vectors
            if vector.parameter_set in self.parameter_sets
            and (vector.pre_hash is None or vector.pre_hash in self.pre_hashes)
            and not vector.unsupported
        ]

    def excluded_by_reason(self, vectors: list[kat.KatVector]) -> dict[str, int]:
        """How many cases each boundary drops, by reason.

        The order below is the order `runnable` rejects in, so a case that trips
        two boundaries is counted under the first — which keeps the buckets a
        partition of what was left out rather than an overlapping tally.
        """
        counts: collections.Counter[str] = collections.Counter()
        for vector in vectors:
            if vector.unsupported:
                counts["operation nothing here names"] += 1
            elif vector.parameter_set not in self.parameter_sets:
                counts[self.parameter_set_reason] += 1
            elif vector.pre_hash is not None and vector.pre_hash not in self.pre_hashes:
                counts["pre-hash function hash-frx does not provide"] += 1
        return dict(counts)

    def operations(self) -> dict[str, list[Operation]]:
        """Every operation each mode publishes, for every constructible set."""
        return {
            mode: [
                Operation(
                    parameter_set=parameter_set,
                    interface=interface,
                    pre_hash=pre_hash,
                    deterministic=deterministic,
                )
                for parameter_set in self.parameter_sets
                for interface, pre_hash in (
                    self.interfaces if published.signed_interfaces else (PURE_EXTERNAL,)
                )
                for deterministic in published.signing_modes
            ]
            for mode, published in _MODES.items()
        }


def group(vectors: Sequence[kat.KatVector]) -> dict[Operation, list[kat.KatVector]]:
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


class _Adapter:
    """What both adapters share: the seam's sizes and mode, read off the scheme.

    They are a scheme's own values rather than the adapter's — the interfaces
    differ in what message they sign, not in what a key or a signature is.
    """

    def __init__(self, scheme: VariantScheme) -> None:
        self._scheme = scheme
        self.public_key_size = scheme.public_key_size
        self.secret_key_size = scheme.secret_key_size
        self.signature_max_size = scheme.signature_max_size
        self.deterministic = scheme.deterministic

    def keygen(self, seed: ArrayLike) -> tuple[Array, Array]:
        return self._scheme.keygen(seed)


class InternalInterface(_Adapter):
    """A standard's internal interface behind the seam, so one harness drives it.

    Not a widening of the scheme's own surface: `sign_internal` stays off the seam
    because it signs an unwrapped message, and this exists only so a vector set
    published for it runs through the harness that owns the negative cases. A
    context is refused rather than ignored — the internal interface has no place
    to put one, and the published internal groups carry none.
    """

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InternalInterface):
            return NotImplemented
        return self._scheme == other._scheme

    def __hash__(self) -> int:
        return hash((type(self), self._scheme))

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


class PreHashVariant(_Adapter):
    """A standard's pre-hash variant behind the seam, for one pre-hash function.

    The function is fixed at construction because it is part of what gets signed:
    a variant that took it per call would be two operations behind one name, which
    is what keeping `hash_sign` off the seam avoids in the first place.
    """

    def __init__(self, scheme: VariantScheme, pre_hash: prehash.PreHash) -> None:
        super().__init__(scheme)
        self._pre_hash = pre_hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PreHashVariant):
            return NotImplemented
        return (self._scheme, self._pre_hash) == (other._scheme, other._pre_hash)

    def __hash__(self) -> int:
        return hash((type(self), self._scheme, self._pre_hash))

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


def implementation(
    operation: Operation, scheme: VariantScheme, coverage: Coverage
) -> Signature:
    """The thing that performs `operation` — the scheme, or an adapter over it.

    `scheme` is already built for `operation`'s parameter set and signing mode,
    which is the half only the scheme's own module can do.
    """
    if operation.interface == "internal":
        if operation.pre_hash is not None:
            raise ValueError(f"{operation}: the internal interface has no pre-hash")
        return InternalInterface(scheme)
    if operation.pre_hash is not None:
        return PreHashVariant(scheme, coverage.pre_hashes[operation.pre_hash]())
    # The pure external operation is the one the seam itself names.
    return scheme


def _reject_context(context: ArrayLike | None) -> None:
    if context is not None:
        raise ValueError(
            "the internal interface signs the message as given, so it has nowhere "
            "to put a context; the external one is what takes it"
        )


if TYPE_CHECKING:
    # Both adapters are what the harness takes, so they carry the same pin a
    # scheme does.
    _internal: type[Signature] = InternalInterface
    _prehash: type[Signature] = PreHashVariant
