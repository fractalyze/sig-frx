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
  this seam only between an operation and its own arithmetic. That decoded
  form is the Protocol's type parameter, so passing a `bytes` where an element
  belongs is a type error rather than a convention a caller is trusted to keep.
- **Scalars travel as Python integers** in `[0, order)`. `scalar_field` is
  the matching zk_dtypes field, which is what a formula core runs on;
  integers are what the wire and the identifiers are written in.
- **`order` and `scalar_field` name the same modulus.** They are two members
  because the bounds checks want an `int` and the formula cores want the
  dtype, not because an implementation may choose separately. The dtype
  stays `Any` where the element does not: it is a class at runtime, and the
  tightest annotation available models none of the operators the field
  arithmetic uses, so tightening it would reject that arithmetic rather than
  check it. One that lets
  them disagree type-checks and then produces a wrong interpolation
  coefficient rather than an error — the curated substrates close this off by
  deriving the integer from the dtype (`classical/secp.py`,
  `classical/edwards.py`), and a hand-written group here has to hold it
  itself.
- **`deserialize_element` validates and raises.** On-curve, canonical, not
  the identity — whatever the group's encoding rules require. A MUST-abort
  condition in the protocol above surfaces as a `ValueError` here rather than
  as a silently wrong point.

The scalar-field dtype's constructor and its int operands **abort at `order`
and above instead of reducing** (fractalyze/zk_dtypes#179), while negative
ints reduce. That is a safety property rather than an inconvenience — an
identifier or a wire scalar that escaped its range cannot quietly become a
different element — and it is why callers bound their operands before the
first field op rather than after.

Implementations are ordinary objects: no base class, no registration.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

# The decoded element. Unbound on purpose: nothing generic here touches an
# element — it is passed back into group methods and never inspected — and a
# bound would have to name a substrate, in the module whose whole contract is
# to name none.
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

    def deserialize_element(self, data: bytes) -> E: ...

    def element_add(self, left: E, right: E) -> E: ...

    def element_scalar_mult(self, element: E, scalar: int) -> E: ...

    def identity_element(self) -> E: ...

    def serialize_element(self, element: E) -> bytes: ...
