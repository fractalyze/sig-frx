# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""FROST (RFC 9591): the two-round threshold Schnorr protocol, one skeleton.

The RFC defines the protocol once and five ciphersuites over it, so the round
functions here take a `Ciphersuite` and never name a curve; a ciphersuite is
constants and hash instantiations, nothing more. The seam is not
`Signature`'s: a threshold scheme signs in two communication rounds with
per-participant state, which is the shape problem a stateful scheme's `sign`
already answered — its own named surface, with the seam left whole. What
*is* unchanged is verification: the aggregate output is a standard Schnorr
signature, verified by the ciphersuite's ordinary verifier (for
FROST(Ed25519, SHA-512), RFC 8032 verification — the existing batched one).

Everything here is a pure function over bytes and Python integers: values in,
values out. Moving them between parties is the consumer's problem, and
holding nonces across the two rounds is the caller's state — the RFC's own
framing, and what makes every function reproducible against the published
vectors. Randomness is a parameter for the same reason (`signature.py`'s
rule: an implicit draw is how a scheme stops being reproducible).

The protocol runs on the host on purpose. Rounds are per-participant work
over a handful of scalars — not the hot path — and the hot path this repo
cares about, verifying the aggregate, needs nothing from this module. DKG is
out of scope, as it is for the RFC (a trusted dealer, Appendix C, is what the
vectors start from).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class Ciphersuite(Protocol):
    """What RFC 9591 §6 instantiates per suite: the group and five hashes.

    Elements travel serialized (`bytes`), scalars as Python integers in
    `[0, order)`. `deserialize_element` validates per the suite (on-curve,
    canonical, not the identity) and raises `ValueError` on anything else —
    the MUST-abort conditions of §5.2 and §5.3 surface as exceptions here.

    `scalar_field` is the suite's zk_dtypes field for `order`: the round
    functions run their mod-order formula cores on it, while scalars still
    cross this seam as integers. Its constructor and int operands abort at
    `order` and above instead of reducing, while negative ints reduce
    (fractalyze/zk_dtypes#179) — the seam's scalar contract and the
    identifier checks keep every operand in range.
    """

    order: int
    element_size: int
    scalar_field: Any

    def h1(self, message: bytes) -> int: ...

    def h2(self, message: bytes) -> int: ...

    def h3(self, message: bytes) -> int: ...

    def h4(self, message: bytes) -> bytes: ...

    def h5(self, message: bytes) -> bytes: ...

    def serialize_scalar(self, scalar: int) -> bytes: ...

    def deserialize_scalar(self, data: bytes) -> int: ...

    def scalar_base_mult(self, scalar: int) -> bytes: ...

    def deserialize_element(self, data: bytes) -> Any: ...

    def element_add(self, left: Any, right: Any) -> Any: ...

    def element_scalar_mult(self, element: Any, scalar: int) -> Any: ...

    def identity_element(self) -> Any: ...

    def serialize_element(self, element: Any) -> bytes: ...


@dataclass(frozen=True)
class Commitment:
    """One participant's round-one output as it travels: `(i, D_i, E_i)`."""

    identifier: int
    hiding: bytes
    binding: bytes


@dataclass(frozen=True)
class Nonces:
    """One participant's round-one state, held locally between rounds.

    The two secret scalars plus their own public commitments as they went to
    the Coordinator — carried rather than re-derived, so §5.2's
    my-commitments-are-in-the-list check is a byte comparison instead of two
    base-point ladders per signing call.
    """

    hiding: int
    binding: int
    hiding_commitment: bytes
    binding_commitment: bytes


def nonce_generate(cs: Ciphersuite, randomness: bytes, secret: int) -> int:
    """RFC 9591 §4.1: `H3(random_bytes ‖ SerializeScalar(secret))`.

    The randomness is the caller's 32 bytes, not a draw made here — which is
    both the seam rule and what makes the vectors' nonces reproducible.
    """
    if len(randomness) != 32:
        raise ValueError("nonce randomness is 32 bytes")
    return cs.h3(randomness + cs.serialize_scalar(secret))


def commit(
    cs: Ciphersuite,
    secret_share: int,
    hiding_randomness: bytes,
    binding_randomness: bytes,
) -> Nonces:
    """RFC 9591 §5.1: round one — nonces and their public commitments.

    The wire half of the result is `(nonces.hiding_commitment,
    nonces.binding_commitment)`; the scalars stay local.
    """
    hiding = nonce_generate(cs, hiding_randomness, secret_share)
    binding = nonce_generate(cs, binding_randomness, secret_share)
    return Nonces(
        hiding,
        binding,
        cs.scalar_base_mult(hiding),
        cs.scalar_base_mult(binding),
    )


def _require_nonzero_scalar(cs: Ciphersuite, identifier: int) -> None:
    """RFC 9591's identifier domain check: a NonZeroScalar, `[1, order-1]`.

    Runs before an identifier meets a field op — an out-of-range int
    operand aborts instead of reducing (see the Protocol docstring).
    """
    if not 1 <= identifier <= cs.order - 1:
        raise ValueError("a participant identifier is a NonZeroScalar")


def _validated_commitment_list(
    cs: Ciphersuite, commitment_list: list[Commitment]
) -> dict[int, tuple[Any, Any]]:
    """§5.2's MUST-checks on the list: sorted, distinct, deserializable.

    Returns the decoded `(hiding, binding)` elements by identifier — the
    validation already paid for the decompression, and the group commitment
    consumes the same points, so decoding them a second time would be pure
    waste.
    """
    identifiers = [c.identifier for c in commitment_list]
    if identifiers != sorted(identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError(
            "a commitment list is sorted by identifier, without duplicates"
        )
    decoded = {}
    for entry in commitment_list:
        _require_nonzero_scalar(cs, entry.identifier)
        decoded[entry.identifier] = (
            cs.deserialize_element(entry.hiding),
            cs.deserialize_element(entry.binding),
        )
    return decoded


def encode_group_commitment_list(
    cs: Ciphersuite, commitment_list: list[Commitment]
) -> bytes:
    """RFC 9591 §4.3: the byte string the binding factors hash over."""
    return b"".join(
        cs.serialize_scalar(entry.identifier) + entry.hiding + entry.binding
        for entry in commitment_list
    )


def compute_binding_factors(
    cs: Ciphersuite,
    group_public_key: bytes,
    commitment_list: list[Commitment],
    message: bytes,
) -> dict[int, int]:
    """RFC 9591 §4.4, keyed by identifier rather than positional."""
    prefix = (
        group_public_key
        + cs.h4(message)
        + cs.h5(encode_group_commitment_list(cs, commitment_list))
    )
    return {
        entry.identifier: cs.h1(prefix + cs.serialize_scalar(entry.identifier))
        for entry in commitment_list
    }


def compute_group_commitment(
    cs: Ciphersuite,
    decoded_commitments: dict[int, tuple[Any, Any]],
    binding_factors: dict[int, int],
) -> Any:
    """RFC 9591 §4.5: `Σ (D_i + [ρ_i]E_i)`, over already-decoded elements."""
    group_commitment = cs.identity_element()
    for identifier, (hiding, binding_commitment) in decoded_commitments.items():
        binding = cs.element_scalar_mult(
            binding_commitment, binding_factors[identifier]
        )
        group_commitment = cs.element_add(
            cs.element_add(group_commitment, hiding), binding
        )
    return group_commitment


def compute_challenge(
    cs: Ciphersuite,
    group_commitment: bytes,
    group_public_key: bytes,
    message: bytes,
) -> int:
    """RFC 9591 §4.6: `H2(R ‖ PK ‖ msg)` — RFC 8032's challenge for Ed25519."""
    return cs.h2(group_commitment + group_public_key + message)


def derive_interpolating_value(
    cs: Ciphersuite, participants: list[int], identifier: int
) -> int:
    """RFC 9591 §4.2: the Lagrange coefficient `λ_i` at zero."""
    if identifier not in participants:
        raise ValueError("the identifier is not among the participants")
    if len(set(participants)) != len(participants):
        raise ValueError("a participant appears more than once")
    for entry in participants:
        _require_nonzero_scalar(cs, entry)
    field = cs.scalar_field
    numerator, denominator = field(1), field(1)
    for other in participants:
        if other == identifier:
            continue
        numerator = numerator * other
        denominator = denominator * (other - identifier)
    return int(numerator / denominator)


def _signing_context(
    cs: Ciphersuite,
    commitment_list: list[Commitment],
    group_public_key: bytes,
    message: bytes,
) -> tuple[dict[int, tuple[Any, Any]], dict[int, int], Any, int]:
    """The round-two prologue all three §5 surfaces share.

    Validate the list, bind, commit, challenge — one home so the surfaces
    cannot drift on the MUST-checks; `aggregate` ignores the challenge.
    """
    decoded = _validated_commitment_list(cs, commitment_list)
    binding_factors = compute_binding_factors(
        cs, group_public_key, commitment_list, message
    )
    group_commitment = compute_group_commitment(cs, decoded, binding_factors)
    challenge = compute_challenge(
        cs, cs.serialize_element(group_commitment), group_public_key, message
    )
    return decoded, binding_factors, group_commitment, challenge


def sign_share(
    cs: Ciphersuite,
    identifier: int,
    secret_share: int,
    group_public_key: bytes,
    nonces: Nonces,
    message: bytes,
    commitment_list: list[Commitment],
) -> bytes:
    """RFC 9591 §5.2: round two — one participant's signature share.

    The §5.2 MUST-checks run first: the list validates, and this participant's
    own round-one commitments appear in it under its identifier — a
    coordinator that swapped them would otherwise make this share sign a
    different group commitment.
    """
    _, binding_factors, _, challenge = _signing_context(
        cs, commitment_list, group_public_key, message
    )
    own = [c for c in commitment_list if c.identifier == identifier]
    if not own or own[0] != Commitment(
        identifier, nonces.hiding_commitment, nonces.binding_commitment
    ):
        raise ValueError("this participant's round-one commitments are not in the list")
    participants = [c.identifier for c in commitment_list]
    lambda_i = derive_interpolating_value(cs, participants, identifier)
    field = cs.scalar_field
    share = int(
        field(nonces.hiding)
        + field(nonces.binding) * binding_factors[identifier]
        + field(lambda_i) * secret_share * challenge
    )
    return cs.serialize_scalar(share)


def aggregate(
    cs: Ciphersuite,
    commitment_list: list[Commitment],
    message: bytes,
    group_public_key: bytes,
    signature_shares: list[bytes],
) -> bytes:
    """RFC 9591 §5.3: the final signature `R ‖ z`, in the suite's encoding.

    Every share deserializes first — §5.3's MUST — so a corrupt share aborts
    here; *which* share is corrupt is `verify_share`'s question.
    """
    _, _, group_commitment, _ = _signing_context(
        cs, commitment_list, group_public_key, message
    )
    scalars = [cs.deserialize_scalar(share) for share in signature_shares]
    field = cs.scalar_field
    z = int(sum(field(scalar) for scalar in scalars))
    return cs.serialize_element(group_commitment) + cs.serialize_scalar(z)


def verify_share(
    cs: Ciphersuite,
    identifier: int,
    participant_public_key: bytes,
    signature_share: bytes,
    commitment_list: list[Commitment],
    group_public_key: bytes,
    message: bytes,
) -> bool:
    """RFC 9591 §5.3's `verify_signature_share`: identify a bad share.

    `[z_i]B = (D_i + [ρ_i]E_i) + [c·λ_i]PK_i` — false names the misbehaving
    participant, which the aggregate signature alone cannot. The RFC's
    `comm_i` input is the list entry under `identifier` — taking it as a
    separate argument would let the two silently disagree.
    """
    decoded, binding_factors, _, challenge = _signing_context(
        cs, commitment_list, group_public_key, message
    )
    share = cs.deserialize_scalar(signature_share)
    participants = [c.identifier for c in commitment_list]
    lambda_i = derive_interpolating_value(cs, participants, identifier)
    hiding, binding_commitment = decoded[identifier]
    commitment_share = cs.element_add(
        hiding,
        cs.element_scalar_mult(binding_commitment, binding_factors[identifier]),
    )
    expected = cs.element_add(
        commitment_share,
        cs.element_scalar_mult(
            cs.deserialize_element(participant_public_key),
            int(cs.scalar_field(challenge) * lambda_i),
        ),
    )
    return cs.scalar_base_mult(share) == cs.serialize_element(expected)


def polynomial_evaluate(cs: Ciphersuite, x: int, coefficients: list[int]) -> int:
    """RFC 9591 Appendix C.1.1: Horner evaluation over the scalar field."""
    field = cs.scalar_field
    x_field = field(x)
    value = field(0)
    for coefficient in reversed(coefficients):
        value = value * x_field + coefficient
    return int(value)


def secret_share_split(
    cs: Ciphersuite,
    secret: int,
    coefficients: list[int],
    max_participants: int,
) -> list[tuple[int, int]]:
    """RFC 9591 Appendix C.1's shard: shares `(i, f(i))` for `i = 1..n`.

    `coefficients` are the dealer's random non-constant terms — a parameter,
    never drawn here, which is what lets the vectors' shares reproduce.
    """
    if not 2 <= max_participants < cs.order:
        raise ValueError("MAX_PARTICIPANTS is at least 2 and below the order")
    polynomial = [secret % cs.order] + [c % cs.order for c in coefficients]
    return [
        (i, polynomial_evaluate(cs, i, polynomial))
        for i in range(1, max_participants + 1)
    ]


def vss_commit(cs: Ciphersuite, secret: int, coefficients: list[int]) -> list[bytes]:
    """RFC 9591 Appendix C.2: the coefficient commitments `[φ_0, …, φ_t]`.

    `φ_0 = [s]B` is the group public key, which is why the dealer publishes
    this vector rather than the key alone.
    """
    return [cs.scalar_base_mult(c) for c in [secret, *coefficients]]


def vss_verify(
    cs: Ciphersuite, identifier: int, share: int, commitment: list[bytes]
) -> bool:
    """RFC 9591 Appendix C.2: `[f(i)]B = Σ i^j·φ_j` — a participant's check."""
    _require_nonzero_scalar(cs, identifier)
    field = cs.scalar_field
    power = field(1)
    expected = cs.identity_element()
    for coefficient_commitment in commitment:
        expected = cs.element_add(
            expected,
            cs.element_scalar_mult(
                cs.deserialize_element(coefficient_commitment), int(power)
            ),
        )
        power = power * identifier
    return cs.scalar_base_mult(share) == cs.serialize_element(expected)
