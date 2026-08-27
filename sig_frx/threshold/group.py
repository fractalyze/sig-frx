# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The prime-order group a threshold protocol runs over, as one seam.

Every threshold protocol over an elliptic curve needs the same eleven things
— an order, a scalar field, the two wire codecs in each direction, and the
group law — and none of them belong to any one protocol. `PrimeOrderGroup`
is that surface with the protocol taken out of it, so a second scheme over
the same curves reuses the group rather than copying it or borrowing a
neighbour's ciphersuite and leaving half of it unimplemented.

A protocol's own vocabulary layers on top: RFC 9591's `Ciphersuite`
(`frost.py`) is this Protocol plus the five hashes §6 instantiates per
suite. The split is where it is because that is where the reuse boundary
falls — the round functions that derive a binding factor or a challenge are
Schnorr's, while the dealer's Shamir sharing, the Feldman commitments, the
Lagrange interpolation, and the commitment-list validation name nothing a
signature scheme fixes and take this Protocol instead.

What travels, and how:

- **Elements travel serialized** (`bytes`), in whatever encoding the group's
  standard fixes; the decoded form is the implementation's own and crosses
  this seam only between an operation and its own arithmetic.
- **Scalars travel as Python integers** in `[0, order)`. `scalar_field` is
  the matching zk_dtypes field, which is what a formula core runs on;
  integers are what the wire and the identifiers are written in.
- **`deserialize_element` validates and raises.** On-curve, canonical, not
  the identity — whatever the group's encoding rules require. A
  MUST-abort condition in the protocol above surfaces as a `ValueError`
  here rather than as a silently wrong point.

The scalar-field dtype's constructor and its int operands **abort at
`order` and above instead of reducing** (fractalyze/zk_dtypes#179), while
negative ints reduce. That is a safety property rather than an
inconvenience — an identifier or a wire scalar that escaped its range
cannot quietly become a different element — and it is why callers bound
their operands before the first field op rather than after.

Implementations are ordinary objects: `secp256k1_sha256.py` and
`ed25519_sha512.py` satisfy this Protocol as part of satisfying
`Ciphersuite`, over substrates the classical schemes already audit.
"""

from __future__ import annotations

from typing import Any, Protocol


class PrimeOrderGroup(Protocol):
    """The group operations and codecs a threshold protocol needs.

    Names nothing from any one protocol — see the module docstring for what
    crosses this seam in which form.
    """

    order: int
    element_size: int
    scalar_field: Any

    def serialize_scalar(self, scalar: int) -> bytes: ...

    def deserialize_scalar(self, data: bytes) -> int: ...

    def scalar_base_mult(self, scalar: int) -> bytes: ...

    def deserialize_element(self, data: bytes) -> Any: ...

    def element_add(self, left: Any, right: Any) -> Any: ...

    def element_scalar_mult(self, element: Any, scalar: int) -> Any: ...

    def identity_element(self) -> Any: ...

    def serialize_element(self, element: Any) -> bytes: ...
