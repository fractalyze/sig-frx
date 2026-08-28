# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The prime-order group a threshold protocol runs over, as one seam.

A threshold protocol over an elliptic curve needs an order, a scalar field,
the wire codecs in both directions, and the group law — and none of that
belongs to any one protocol. `PrimeOrderGroup` is that surface with the
protocol taken out of it, so a second scheme over the same curves reuses the
group rather than copying it or borrowing a neighbour's ciphersuite and
leaving half of it unimplemented. A protocol's own vocabulary layers on top:
RFC 9591's `Ciphersuite` ([`frost.py`](frost.py)) is this Protocol plus the
five hashes §6 instantiates per suite.

What travels, and how:

- **Elements travel serialized** (`bytes`), in whatever encoding the group's
  standard fixes; the decoded form is the implementation's own and crosses
  this seam only between an operation and its own arithmetic.
- **Elements travel in batches.** Every element-typed member carries a leading
  axis, and a single element is `B = 1` — the same rule
  [`signature.py`](../signature.py) states for verification, for the same
  reason. It is not a performance hint: `secp`'s placement threshold is a batch
  size, so a seam that handed one element at a time could never reach the
  device no matter how much work its caller had. A protocol whose round really
  is one element pays nothing for the axis, because the threshold keeps it on
  the host; one with hundreds gets the device without asking.
- **Scalars travel as Python integers** in `[0, order)`. `scalar_field` is
  the matching zk_dtypes field, which is what a formula core runs on;
  integers are what the wire and the identifiers are written in.
- **`order` and `scalar_field` name the same modulus.** They are two members
  because the bounds checks want an `int` and the formula cores want the
  dtype, not because an implementation may choose separately. One that lets
  them disagree type-checks and then produces a wrong interpolation
  coefficient rather than an error — the curated substrates close this off by
  deriving the integer from the dtype (`classical/secp.py`,
  `classical/edwards.py`), and a hand-written group here has to hold it
  itself.
- **`deserialize_elements` validates and raises.** On-curve, canonical, not
  the identity — whatever the group's encoding rules require. A MUST-abort
  condition in the protocol above surfaces as a `ValueError` here rather than
  as a silently wrong point. One bad entry rejects the call: a batch is one
  protocol message, and half of one is not a thing a caller can act on.
- **Selecting from a batch is the group's operation, not the caller's.**
  A protocol that wants one participant's element out of a batch asks for it by
  index; it does not slice. The decoded form is the implementation's own, so a
  caller that indexed it would be assuming a shape the seam never promised —
  and `select_elements` returns a batch like everything else, so the result is
  still `B = 1` rather than a loose element.
- **There is no identity member.** It existed to seed a Python loop that
  accumulated elements one at a time, and `sum_elements` is that loop as a
  reduction, so the seed has no remaining caller. A group that needs the
  identity for its own arithmetic still has it — internally, where it belongs.

The scalar-field dtype's constructor and its int operands **abort at `order`
and above instead of reducing** (fractalyze/zk_dtypes#179), while negative
ints reduce. That is a safety property rather than an inconvenience — an
identifier or a wire scalar that escaped its range cannot quietly become a
different element — and it is why callers bound their operands before the
first field op rather than after.

Implementations are ordinary objects: no base class, no registration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, TypeVar, runtime_checkable

# The decoded element batch. Unbound on purpose: nothing generic here ever
# looks inside one — a batch is produced by the group, handed back to the
# group, and never indexed or unpacked by a caller — and a bound would have to
# name a substrate, in the module whose whole contract is to name none.
E = TypeVar("E")


@runtime_checkable
class PrimeOrderGroup(Protocol[E]):
    """The group operations and codecs a threshold protocol needs.

    Names nothing from any one protocol — see the module docstring for what
    crosses this seam in which form.
    """

    order: int
    scalar_field: Any

    def serialize_scalar(self, scalar: int) -> bytes: ...

    def deserialize_scalar(self, data: bytes) -> int: ...

    def scalar_base_mult(self, scalar: int) -> bytes: ...

    def deserialize_elements(self, data: Sequence[bytes]) -> E: ...

    def elements_add(self, left: E, right: E) -> E: ...

    def elements_scalar_mult(self, elements: E, scalars: Sequence[int]) -> E: ...

    def select_elements(self, elements: E, indices: Sequence[int]) -> E: ...

    def sum_elements(self, elements: E) -> E: ...

    def serialize_elements(self, elements: E) -> list[bytes]: ...
