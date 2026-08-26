# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""MuSig2 key aggregation and tweaking per BIP-327, over BIP-340's substrate.

MuSig2 aggregates `u` cosigner keys into one x-only key, and what it produces
under that key is an ordinary BIP-340 signature — so this module borrows
`bip340`'s curve, its tagged-hash construction and its even-y convention, and
adds no verifier of its own. It is not on the `Signature` seam: a two-round
interactive protocol has no seam-shaped `sign`, the shape the threshold and
stateful schemes already settled, so the stage functions are their own named
surface.

## The aggregate is order-dependent, on purpose

`KeyAgg` is not a plain sum of the cosigner keys. Each key is weighted by a
coefficient derived from the hash of the whole key list, which is what stops a
participant from choosing a key that cancels the others (the rogue-key attack).
Two consequences the caller owns: the key list is ordered, so aggregating the
same keys in a different order yields a different aggregate key, and a
duplicate key is a distinct entry rather than a no-op.

The one key exempted from a hashed coefficient is the *second distinct* key in
the list, which gets coefficient 1. That is the specification's optimisation
for the common two-party case and not an accident to be normalised away.

## A bad contribution names its sender

Every failure that a cosigner could have caused raises
`InvalidContributionError` carrying that cosigner's index. A coordinator that
only learns "the ceremony failed" cannot exclude anyone and has to restart with
everybody; one that learns which index sent an unparseable key can drop that
participant and continue. The index is the difference between the two, so it is
part of the exception rather than a log line.

## Where the batch axis is

There is none here, deliberately, and for a different reason than the rounds of
a threshold protocol have none. Key aggregation is a ceremony run once for a
key, not the hot path — but it *is* internally wide, a `u`-term multi-scalar
multiplication, so the coefficients and points cross into `secp` as one batch
and the sum is one reduction rather than a Python fold.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sig_frx.classical import secp
from sig_frx.classical.schnorr import bip340

_CURVE = secp.SECP256K1

# The tag prefixes are BIP-327's own; the construction they feed is
# `bip340.tagged`, which BIP-327 incorporates by reference.
_KEYAGG_LIST = hashlib.sha256(b"KeyAgg list").digest()
_KEYAGG_COEFF = hashlib.sha256(b"KeyAgg coefficient").digest()

_PUBKEY_SIZE = 33
_TWEAK_SIZE = 32


class InvalidContributionError(Exception):
    """A named cosigner sent something unusable.

    `signer` is that cosigner's index in the key list as the caller passed it,
    and `contrib` names what was wrong with what they sent — the two fields
    BIP-327's error cases carry, so a coordinator can exclude one participant
    instead of restarting the ceremony.
    """

    def __init__(self, signer: int, contrib: str) -> None:
        super().__init__(f"signer {signer} sent an invalid {contrib}")
        self.signer = signer
        self.contrib = contrib


def _parse_pubkey(data: bytes, signer: int) -> tuple[int, int]:
    """One compressed key to `(x, parity)`, or the sender's index as an error.

    The bound on `x` is checked here, where the value is still an integer: the
    base field's dtype aborts on an out-of-range operand rather than reducing
    it, so a key above `p` that reached the lift would raise from inside the
    substrate with no cosigner index left to name.
    """
    if len(data) != _PUBKEY_SIZE or data[0] not in (2, 3):
        raise InvalidContributionError(signer, "pubkey")
    x = int.from_bytes(data[1:], "big")
    if x >= _CURVE.p:
        raise InvalidContributionError(signer, "pubkey")
    return x, data[0] - 2


def _second_key(pubkeys: Sequence[bytes]) -> bytes | None:
    """The first key that differs from the first one, if the list has one."""
    for pubkey in pubkeys[1:]:
        if pubkey != pubkeys[0]:
            return pubkey
    return None


def _coefficient(digest: bytes, second: bytes | None, pubkey: bytes) -> int:
    """One key's weight. `second` compares unequal when it is `None`, which is
    the all-keys-identical case the specification's zero sentinel also covers."""
    if pubkey == second:
        return 1
    return (
        int.from_bytes(bip340.tagged(_KEYAGG_COEFF, digest + pubkey), "big") % _CURVE.n
    )


def _coefficients(pubkeys: Sequence[bytes]) -> list[int]:
    digest = bip340.tagged(_KEYAGG_LIST, b"".join(pubkeys))
    second = _second_key(pubkeys)
    return [_coefficient(digest, second, pubkey) for pubkey in pubkeys]


@dataclass(frozen=True)
class KeyAggContext:
    """An aggregate key plus the accumulated effect of the tweaks applied to it.

    `gacc` and `tacc` are not bookkeeping: signing has to undo, on the secret
    side, exactly what tweaking did on the public side. `gacc` accumulates the
    negations that x-only tweaking forced and `tacc` the added scalars, so a
    partial signature can be formed against the tweaked key without any signer
    holding the tweaked secret.
    """

    point: np.ndarray
    gacc: int
    tacc: int

    def _affine(self) -> tuple[int, int]:
        return secp.affine_ints(_CURVE, self.point)[0]

    def has_even_y(self) -> bool:
        return self._affine()[1] % 2 == 0

    def xonly_bytes(self) -> bytes:
        """The 32-byte x-only aggregate key, which is a BIP-340 public key."""
        return self._affine()[0].to_bytes(32, "big")

    def apply_tweak(self, tweak: bytes, is_xonly: bool) -> KeyAggContext:
        """One tweak applied, returning the context it produces.

        `is_xonly` picks the convention: an x-only tweak is defined against the
        even-y representative, so an aggregate that landed on odd y is negated
        first and the negation is recorded in `gacc`. A plain tweak adds to the
        point as it stands. Both forms are specified, and which one a consumer
        wants is its own convention — a taproot output commits with the x-only
        form — so neither is a default here.
        """
        if len(tweak) != _TWEAK_SIZE:
            raise ValueError(f"a tweak is {_TWEAK_SIZE} bytes, not {len(tweak)}")
        value = int.from_bytes(tweak, "big")
        if value >= _CURVE.n:
            raise ValueError("the tweak must be less than the group order")

        parity = -1 if is_xonly and not self.has_even_y() else 1
        tweaked = secp.double_multiple(_CURVE, [value], [parity % _CURVE.n], self.point)
        if bool(secp.is_identity(_CURVE, tweaked)[0]):
            raise ValueError("the result of tweaking cannot be the identity")

        return KeyAggContext(
            point=tweaked,
            gacc=(parity * self.gacc) % _CURVE.n,
            tacc=(value + parity * self.tacc) % _CURVE.n,
        )


def key_agg(pubkeys: Sequence[bytes]) -> KeyAggContext:
    """The cosigners' compressed keys aggregated into one, in the order given.

    Raises `InvalidContributionError` naming the cosigner whose key cannot be
    parsed or does not lie on the curve, and `ValueError` when the aggregate
    itself is the identity — which no cosigner can be blamed for, since it
    takes a set of keys to reach it.
    """
    if not pubkeys:
        raise ValueError("key aggregation needs at least one key")

    parsed = [_parse_pubkey(pubkey, signer) for signer, pubkey in enumerate(pubkeys)]
    points, lifted = secp.lift_x_to_parity(
        _CURVE, [x for x, _ in parsed], [parity for _, parity in parsed]
    )
    if not lifted.all():
        raise InvalidContributionError(int(np.argmin(lifted)), "pubkey")

    aggregate = secp.sum_points(
        _CURVE, secp.multiple(_CURVE, _coefficients(pubkeys), points)
    )
    if bool(secp.is_identity(_CURVE, aggregate)[0]):
        raise ValueError("the aggregate key cannot be the identity")

    return KeyAggContext(point=aggregate, gacc=1, tacc=0)
