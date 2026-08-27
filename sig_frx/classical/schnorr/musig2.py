# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""MuSig2 per BIP-327, over BIP-340's substrate.

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

import functools
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

# A compressed point, which is both a cosigner key and half a pubnonce.
_POINT_SIZE = 33
_PUBNONCE_SIZE = 2 * _POINT_SIZE
_SCALAR_SIZE = 32
_SECNONCE_SIZE = 2 * _SCALAR_SIZE + _POINT_SIZE

# BIP-327's own tag prefixes for the nonce derivation.
_AUX_TAG = hashlib.sha256(b"MuSig/aux").digest()
_NONCE_TAG = hashlib.sha256(b"MuSig/nonce").digest()
_NONCECOEF_TAG = hashlib.sha256(b"MuSig/noncecoef").digest()
_DETERMINISTIC_TAG = hashlib.sha256(b"MuSig/deterministic/nonce").digest()


class InvalidContributionError(Exception):
    """A named cosigner sent something unusable.

    `signer` is that cosigner's index in the key list as the caller passed it,
    and `contrib` names what was wrong with what they sent — the two fields
    BIP-327's error cases carry, so a coordinator can exclude one participant
    instead of restarting the ceremony.

    `signer` is `None` when the fault is in the *aggregate* nonce, and that is
    a distinction rather than a missing value: the aggregate is the
    coordinator's own product, so no cosigner sent it and excluding one would
    not fix it.
    """

    def __init__(self, signer: int | None, contrib: str) -> None:
        who = "the coordinator" if signer is None else f"signer {signer}"
        super().__init__(f"{who} sent an invalid {contrib}")
        self.signer = signer
        self.contrib = contrib


def _parse_point(data: bytes, signer: int | None, contrib: str) -> tuple[int, int]:
    """One compressed point to `(x, parity)`, or the sender's index as an error.

    `contrib` names what the sender got wrong, because a key and a nonce fail
    the same three ways and a coordinator needs to know which one it was, and
    `signer` is `None` where the bytes are the coordinator's own aggregate.

    The bound on `x` is checked here, where the value is still an integer: the
    base field's dtype aborts on an out-of-range operand rather than reducing
    it, so a key above `p` that reached the lift would raise from inside the
    substrate with no cosigner index left to name.
    """
    if len(data) != _POINT_SIZE or data[0] not in (2, 3):
        raise InvalidContributionError(signer, contrib)
    x = int.from_bytes(data[1:], "big")
    if x >= _CURVE.p:
        raise InvalidContributionError(signer, contrib)
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
        if len(tweak) != _SCALAR_SIZE:
            raise ValueError(f"a tweak is {_SCALAR_SIZE} bytes, not {len(tweak)}")
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


@dataclass(frozen=True)
class SecNonce:
    """One signer's secret nonce pair, drawn for exactly one signing session.

    **Use once.** Signing twice from one `SecNonce` against different messages
    or different cosigner sets reveals the signer's secret key outright — the
    concurrent-session attack that MuSig2's two-nonce construction exists to
    survive. It is not a liveness abort the way a FROST round failure is; the
    key is gone. Signing therefore consumes this and hands back nothing that
    can be spent again — the shape `Xmss` uses for its leaf counter, though
    only the shape: an advanced index makes a spent key visible, and a nonce
    has no such tell. Nothing here or in `sign` can detect reuse; the type
    exists so that a caller has to write the reuse down to commit it.

    `public_key` is the signer's own, carried so that signing can refuse a
    nonce drawn for a different key rather than produce a partial signature
    that silently fails to aggregate.
    """

    first: int
    second: int
    public_key: bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> SecNonce:
        """The 97-byte layout back, without judging the values.

        Range is checked where it is acted on rather than here, so that a
        secnonce read back from a caller's own storage reports the same way as
        one that was never stored — `sign` refuses an out-of-range scalar and
        says so, which the specification flags as a possible sign of reuse.
        """
        if len(data) != _SECNONCE_SIZE:
            raise ValueError(f"a secnonce is {_SECNONCE_SIZE} bytes, not {len(data)}")
        return cls(
            int.from_bytes(data[:_SCALAR_SIZE], "big"),
            int.from_bytes(data[_SCALAR_SIZE : 2 * _SCALAR_SIZE], "big"),
            data[2 * _SCALAR_SIZE :],
        )

    def to_bytes(self) -> bytes:
        """The specification's 97-byte layout, `k1 || k2 || pk`."""
        return (
            self.first.to_bytes(_SCALAR_SIZE, "big")
            + self.second.to_bytes(_SCALAR_SIZE, "big")
            + self.public_key
        )


def _message_prefix(message: bytes | None) -> bytes:
    """An absent message and an empty one are different inputs.

    Length-prefixing alone would make `None` and `b""` hash identically, so the
    specification tags presence first. A signer that conflated them would draw
    one nonce for two sessions, which is the loss described on `SecNonce`.
    """
    if message is None:
        return b"\x00"
    return b"\x01" + len(message).to_bytes(8, "big") + message


def _nonce_hash(
    rand: bytes,
    public_key: bytes,
    aggregate_key: bytes,
    message_prefixed: bytes,
    extra_input: bytes,
    index: int,
) -> int:
    """One of the two nonce scalars.

    Every variable-length field carries its own length, so no two distinct
    inputs can concatenate to the same buffer.
    """
    payload = b"".join(
        (
            rand,
            len(public_key).to_bytes(1, "big"),
            public_key,
            len(aggregate_key).to_bytes(1, "big"),
            aggregate_key,
            message_prefixed,
            len(extra_input).to_bytes(4, "big"),
            extra_input,
            index.to_bytes(1, "big"),
        )
    )
    return int.from_bytes(bip340.tagged(_NONCE_TAG, payload), "big") % _CURVE.n


def nonce_gen(
    rand: bytes,
    public_key: bytes,
    *,
    secret_key: bytes | None = None,
    aggregate_key: bytes | None = None,
    message: bytes | None = None,
    extra_input: bytes | None = None,
) -> tuple[SecNonce, bytes]:
    """A signer's nonce pair for one session: the secret and what it publishes.

    `rand` is the caller's randomness and must be fresh per session — the
    derivation is a pure function of it, which is what makes the published
    vectors reproducible and what makes reuse fatal (see `SecNonce`).

    Every other input is optional and each one that is supplied narrows the
    sessions this nonce could belong to. `secret_key` is the strongest: it is
    folded in by masking `rand`, so the nonce stays unpredictable even if the
    randomness was not. Passing what is known is always at least as safe as
    passing nothing.
    """
    if len(public_key) != _POINT_SIZE:
        raise ValueError(f"a public key is {_POINT_SIZE} bytes, not {len(public_key)}")
    if secret_key is not None:
        rand = _mask_secret(secret_key, rand)

    prefixed = _message_prefix(message)
    scalars = [
        _nonce_hash(
            rand,
            public_key,
            aggregate_key or b"",
            prefixed,
            extra_input or b"",
            index,
        )
        for index in (0, 1)
    ]
    if 0 in scalars:
        raise ValueError("the nonce derivation produced a zero scalar")

    return SecNonce(scalars[0], scalars[1], public_key), _publish(scalars)


def _lift_all(parsed: list[tuple[int, int]], contrib: str) -> np.ndarray:
    """Every `(x, parity)` to its point, blaming the first sender that has none."""
    points, lifted = secp.lift_x_to_parity(
        _CURVE, [x for x, _ in parsed], [parity for _, parity in parsed]
    )
    if not lifted.all():
        raise InvalidContributionError(int(np.argmin(lifted)), contrib)
    return points


def _serialize_ext(point: np.ndarray) -> bytes:
    """A point that may be the identity, in the wire form BIP-327 gives it.

    The identity serializes as 33 zero bytes — a value no real point occupies,
    since `x = 0` would need `b` to be a residue. It is a legitimate aggregate
    nonce rather than a failure: one half of the nonces can cancel while the
    other does not, and signing continues from it.
    """
    if bool(secp.is_identity(_CURVE, point)[0]):
        return bytes(_POINT_SIZE)
    return secp.compressed_bytes(_CURVE, *secp.affine_ints(_CURVE, point)[0])


def nonce_agg(pubnonces: Sequence[bytes]) -> bytes:
    """The cosigners' public nonces summed into one, half by half.

    A pubnonce is two points, and they aggregate independently — so the halves
    are summed separately and either may come out as the identity.

    Raises `InvalidContributionError` naming the cosigner whose nonce cannot be
    parsed or does not lie on the curve. The halves are checked in order, so a
    session with faults in both reports the one the specification reports.
    """
    if not pubnonces:
        raise ValueError("nonce aggregation needs at least one nonce")

    halves = []
    for half in (0, 1):
        parsed = []
        for signer, pubnonce in enumerate(pubnonces):
            if half == 0 and len(pubnonce) != _PUBNONCE_SIZE:
                raise InvalidContributionError(signer, "pubnonce")
            window = pubnonce[half * _POINT_SIZE : (half + 1) * _POINT_SIZE]
            parsed.append(_parse_point(window, signer, "pubnonce"))
        points = _lift_all(parsed, "pubnonce")
        halves.append(secp.sum_points(_CURVE, points))
    return b"".join(_serialize_ext(half) for half in halves)


def _point_ext(data: bytes) -> np.ndarray | None:
    """A compressed point that may be the all-zero identity encoding.

    Returns `None` for the identity, which callers branch on — an aggregate
    nonce is allowed to be it and a cosigner key is not.
    """
    if data == bytes(_POINT_SIZE):
        return None
    x, parity = _parse_point(data, None, "aggnonce")
    # Not `_lift_all`: that blames by position, and an aggregate nonce has no
    # position to blame — the whole point of `signer=None` here.
    points, lifted = secp.lift_x_to_parity(_CURVE, [x], [parity])
    if not lifted.all():
        raise InvalidContributionError(None, "aggnonce")
    return points


def _tweaked_context(
    pubkeys: Sequence[bytes], tweaks: Sequence[tuple[bytes, bool]]
) -> KeyAggContext:
    """The aggregate key with its tweaks applied, in the order given.

    Order is part of what the key commits to, so it is a rule rather than a
    loop — and one both `Session` and `deterministic_sign` reach for, which is
    why it is not written inside either.
    """
    context = key_agg(pubkeys)
    for tweak, is_xonly in tweaks:
        context = context.apply_tweak(tweak, is_xonly)
    return context


def _mask_secret(secret_key: bytes, rand: bytes) -> bytes:
    """The secret folded into the caller's randomness, so a nonce stays
    unpredictable even where the randomness was not."""
    mask = bip340.tagged(_AUX_TAG, rand)
    return bytes(a ^ b for a, b in zip(secret_key, mask, strict=True))


def _public_key(secret_key: bytes) -> tuple[int, bytes]:
    """A secret's scalar and its compressed public key.

    The range and width checks are `secp.secret_scalar`'s rather than spelled
    again here — a 31-byte secret whose value happens to be in range is refused
    there and would not be by a bare `int.from_bytes`.
    """
    _, secret = secp.secret_scalar(
        _CURVE, np.frombuffer(secret_key, dtype=np.uint8), "secret key"
    )
    return secret, secp.compressed_bytes(
        _CURVE, *secp.host_multiple_of_g(_CURVE, secret)
    )


def _publish(scalars: Sequence[int]) -> bytes:
    """The public half of a nonce pair: each scalar's point, compressed."""
    return b"".join(
        secp.compressed_bytes(_CURVE, *secp.host_multiple_of_g(_CURVE, scalar))
        for scalar in scalars
    )


@dataclass(frozen=True)
class Session:
    """Everything the cosigners must already agree on before anyone signs.

    Signing and verifying a partial signature both derive their challenge from
    exactly this, so the type exists to make disagreement impossible to express
    halfway: two signers who differ on the message or the key list produce
    partial signatures that cannot aggregate, and nothing before aggregation
    would have said so.

    `tweaks` are applied in order as `(tweak, is_xonly)` pairs, because the
    order is part of what the aggregate key commits to.
    """

    aggnonce: bytes
    pubkeys: Sequence[bytes]
    message: bytes
    tweaks: Sequence[tuple[bytes, bool]] = ()

    def __post_init__(self) -> None:
        # Frozen against rebinding, but a caller's list stays theirs to mutate
        # — which a memoized derivation would silently outlive. Copying to
        # tuples is what makes `key_context` safe to cache.
        object.__setattr__(self, "pubkeys", tuple(self.pubkeys))
        object.__setattr__(self, "tweaks", tuple(self.tweaks))

    @functools.cached_property
    def key_context(self) -> KeyAggContext:
        """The tweaked aggregate key. Derived once: a session is signed against
        and verified against repeatedly, and every one of those re-ran a
        `u`-term multi-scalar multiplication."""
        return _tweaked_context(self.pubkeys, self.tweaks)


@dataclass(frozen=True)
class _SessionValues:
    """The derived quantities both signing and verification need."""

    keys: KeyAggContext
    key_parity: int
    coefficient: int
    nonce_x: int
    nonce_even: bool
    challenge: int


def _session_values(session: Session) -> _SessionValues:
    keys = session.key_context
    if len(session.aggnonce) != _PUBNONCE_SIZE:
        raise InvalidContributionError(None, "aggnonce")

    aggregate_key = keys.xonly_bytes()
    coefficient = (
        int.from_bytes(
            bip340.tagged(
                _NONCECOEF_TAG, session.aggnonce + aggregate_key + session.message
            ),
            "big",
        )
        % _CURVE.n
    )

    first = _point_ext(session.aggnonce[:_POINT_SIZE])
    second = _point_ext(session.aggnonce[_POINT_SIZE:])
    terms = []
    if first is not None:
        terms.append(first)
    if second is not None:
        terms.append(secp.multiple(_CURVE, [coefficient], second))
    nonce = secp.sum_points(_CURVE, np.concatenate(terms)) if terms else None

    # An aggregate nonce that cancels to the identity is a session the
    # specification continues, substituting the generator: aborting would let
    # any cosigner strand the ceremony by choosing a cancelling nonce.
    if nonce is None or bool(secp.is_identity(_CURVE, nonce)[0]):
        nonce_x, nonce_y = _CURVE.gx, _CURVE.gy
    else:
        nonce_x, nonce_y = secp.affine_ints(_CURVE, nonce)[0]

    # BIP-327 signs under BIP-340's challenge, reduction included — spelling
    # the tagged hash out here would make this the third owner of that rule.
    challenge = bip340.challenge(
        nonce_x.to_bytes(_SCALAR_SIZE, "big"), aggregate_key, session.message
    )
    return _SessionValues(
        keys=keys,
        key_parity=1 if keys.has_even_y() else _CURVE.n - 1,
        coefficient=coefficient,
        nonce_x=nonce_x,
        nonce_even=nonce_y % 2 == 0,
        challenge=challenge,
    )


def _signer_coefficient(session: Session, public_key: bytes) -> int:
    """This signer's weight in the aggregate, refusing a signer it excludes.

    BIP-327 marks the membership check optional. It is taken because the
    alternative is silent: a partial signature under a key the session never
    aggregated is well-formed and simply fails to combine, so the coordinator
    learns only that aggregation failed and not who to ask.
    """
    if public_key not in session.pubkeys:
        raise ValueError("the signer's public key is not in the session's key list")
    return _coefficient(
        bip340.tagged(_KEYAGG_LIST, b"".join(session.pubkeys)),
        _second_key(session.pubkeys),
        public_key,
    )


def sign(secnonce: SecNonce, secret_key: bytes, session: Session) -> bytes:
    """This signer's partial signature, spending `secnonce`.

    The nonce is spent here and there is nothing to hand back: unlike a leaf
    counter there is no advanced value that would make a second call visibly
    wrong (see `SecNonce`). What this can check, it does — that the secnonce
    was drawn for the key doing the signing, and that its scalars are in range,
    which the specification notes is where nonce reuse tends to show up.
    """
    values = _session_values(session)
    first = secnonce.first if values.nonce_even else _CURVE.n - secnonce.first
    second = secnonce.second if values.nonce_even else _CURVE.n - secnonce.second
    for scalar, role in ((secnonce.first, "first"), (secnonce.second, "second")):
        if not 0 < scalar < _CURVE.n:
            raise ValueError(f"the {role} secnonce value is out of range")

    secret, public_key = _public_key(secret_key)
    if public_key != secnonce.public_key:
        raise ValueError("the secnonce was drawn for a different public key")

    weight = _signer_coefficient(session, public_key)
    effective = values.key_parity * values.keys.gacc * secret % _CURVE.n
    total = (
        first + values.coefficient * second + values.challenge * weight * effective
    ) % _CURVE.n
    return total.to_bytes(_SCALAR_SIZE, "big")


def partial_sig_agg(psigs: Sequence[bytes], session: Session) -> bytes:
    """The cosigners' partial signatures combined into one BIP-340 signature.

    This is where the protocol stops being MuSig2. Everything above produces
    values only these functions understand; what comes out here is 64 bytes a
    taproot output accepts, indistinguishable on chain from a single signer's.
    That is why this module ships no verifier — `Bip340.verify` is already the
    right one, and adding a second would be a second accept set to keep in
    agreement with it.

    The tweaks are spent here. `tacc` accumulated what tweaking added to the
    public key, and folding it in is what makes the signature check against the
    tweaked key rather than the one the cosigners aggregated.

    Raises `InvalidContributionError` naming the cosigner whose partial
    signature is not a scalar. That is the last point where a name is available
    — after this there is one signature, and a bad one says only that some
    cosigner was wrong.
    """
    values = _session_values(session)
    total = 0
    for signer, psig in enumerate(psigs):
        if len(psig) != _SCALAR_SIZE:
            raise InvalidContributionError(signer, "psig")
        share = int.from_bytes(psig, "big")
        if share >= _CURVE.n:
            raise InvalidContributionError(signer, "psig")
        total = (total + share) % _CURVE.n

    total = (total + values.challenge * values.key_parity * values.keys.tacc) % _CURVE.n
    return values.nonce_x.to_bytes(_SCALAR_SIZE, "big") + total.to_bytes(
        _SCALAR_SIZE, "big"
    )


def partial_sig_verify(
    psig: bytes,
    pubnonces: Sequence[bytes],
    pubkeys: Sequence[bytes],
    message: bytes,
    signer: int,
    *,
    tweaks: Sequence[tuple[bytes, bool]] = (),
) -> bool:
    """Whether one cosigner's partial signature is the one this session wanted.

    A wrong signature is `False` and an unusable contribution raises, which is
    the specification's split and worth keeping: the first is a cosigner who
    signed something else, the second is one who sent something that is not a
    signature at all. Only the second identifies somebody to exclude.

    Checking partials before aggregating is what turns a failed aggregate — one
    bad signature in `u`, with nothing to say which — into a named participant.
    """
    session = Session(
        aggnonce=nonce_agg(pubnonces),
        pubkeys=pubkeys,
        message=message,
        tweaks=tweaks,
    )
    values = _session_values(session)
    total = int.from_bytes(psig, "big")
    if total >= _CURVE.n:
        return False

    nonce = pubnonces[signer]
    points = _lift_all(
        [
            _parse_point(nonce[at : at + _POINT_SIZE], signer, "pubnonce")
            for at in (0, _POINT_SIZE)
        ],
        "pubnonce",
    )
    commitment = secp.sum_points(
        _CURVE,
        np.concatenate(
            [points[:1], secp.multiple(_CURVE, [values.coefficient], points[1:])]
        ),
    )
    if not values.nonce_even:
        commitment = secp.multiple(_CURVE, [_CURVE.n - 1], commitment)

    key_point = _lift_all([_parse_point(pubkeys[signer], signer, "pubkey")], "pubkey")
    weight = _signer_coefficient(session, pubkeys[signer])
    scaled = values.challenge * weight * values.key_parity * values.keys.gacc % _CURVE.n

    left = secp.multiple(_CURVE, [total], _CURVE.generator)
    right = secp.sum_points(
        _CURVE,
        np.concatenate([commitment, secp.multiple(_CURVE, [scaled], key_point)]),
    )
    return bool(np.asarray(left == right)[0])


def deterministic_sign(
    secret_key: bytes,
    other_nonces: bytes,
    pubkeys: Sequence[bytes],
    message: bytes,
    *,
    rand: bytes | None = None,
    tweaks: Sequence[tuple[bytes, bool]] = (),
) -> tuple[bytes, bytes]:
    """The last signer's two rounds collapsed into one.

    Given every other cosigner's nonces already aggregated, this derives its own
    nonce from the session and signs in a single step — so a signer that cannot
    hold state between rounds can still take part, which is the shape a hardware
    device or a stateless service has.

    It does not lift the rule that a nonce must not repeat; it removes the
    window in which repeating one is possible. The nonce is a function of the
    secret key, the other nonces and the message, so signing one session twice
    reproduces the same nonce harmlessly, and two different sessions cannot
    collide. That is why `rand` is optional here and required by `nonce_gen`:
    there, the caller's randomness is the only thing making the draw unique.

    Returns this signer's public nonce alongside the partial signature, because
    the coordinator needs both and there is no round in which to send them
    separately.
    """
    signing_key = secret_key if rand is None else _mask_secret(secret_key, rand)
    context = _tweaked_context(pubkeys, tweaks)

    preimage = (
        signing_key
        + other_nonces
        + context.xonly_bytes()
        + len(message).to_bytes(8, "big")
        + message
    )
    scalars = [
        int.from_bytes(
            bip340.tagged(_DETERMINISTIC_TAG, preimage + index.to_bytes(1, "big")),
            "big",
        )
        % _CURVE.n
        for index in (0, 1)
    ]
    if 0 in scalars:
        raise ValueError("the deterministic nonce derivation produced a zero scalar")

    pubnonce = _publish(scalars)
    _, own_key = _public_key(secret_key)

    # `nonce_agg` blames by position, and only position one can fail: position
    # zero is the pubnonce built just above, from scalars already checked
    # non-zero. So every verdict it can reach here names the aggregate the
    # coordinator supplied, which is nobody's contribution.
    try:
        aggnonce = nonce_agg([pubnonce, other_nonces])
    except InvalidContributionError as error:
        raise InvalidContributionError(None, "aggothernonce") from error

    session = Session(
        aggnonce=aggnonce, pubkeys=pubkeys, message=message, tweaks=tweaks
    )
    secnonce = SecNonce(scalars[0], scalars[1], own_key)
    return pubnonce, sign(secnonce, secret_key, session)


def key_sort(pubkeys: Sequence[bytes]) -> list[bytes]:
    """The cosigner keys in BIP-327's canonical order.

    Defined on the serializations rather than on the points — the 33-byte
    encodings, ordered lexicographically — so this parses nothing and refuses
    nothing. An unusable key sorts like any other and is caught by `key_agg`,
    which is the one place that has a cosigner index to blame.

    Duplicates survive. A repeated key is a distinct cosigner slot that
    `key_agg` weights separately, so collapsing them here would silently change
    the aggregate.

    Sorting is the caller's to apply, not `key_agg`'s to assume: aggregation
    binds the order it is given, so a group that sorts and a group that keeps
    its own order derive different keys, and neither is wrong.
    """
    return sorted(pubkeys)


def key_agg(pubkeys: Sequence[bytes]) -> KeyAggContext:
    """The cosigners' compressed keys aggregated into one, in the order given.

    Raises `InvalidContributionError` naming the cosigner whose key cannot be
    parsed or does not lie on the curve, and `ValueError` when the aggregate
    itself is the identity — which no cosigner can be blamed for, since it
    takes a set of keys to reach it.
    """
    if not pubkeys:
        raise ValueError("key aggregation needs at least one key")

    parsed = [
        _parse_point(pubkey, signer, "pubkey") for signer, pubkey in enumerate(pubkeys)
    ]
    points = _lift_all(parsed, "pubkey")
    aggregate = secp.sum_points(
        _CURVE, secp.multiple(_CURVE, _coefficients(pubkeys), points)
    )
    if bool(secp.is_identity(_CURVE, aggregate)[0]):
        raise ValueError("the aggregate key cannot be the identity")

    return KeyAggContext(point=aggregate, gacc=1, tacc=0)
