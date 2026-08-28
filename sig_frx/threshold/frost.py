# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""FROST (RFC 9591): the two-round threshold Schnorr protocol, one skeleton.

The RFC defines the protocol once and five ciphersuites over it, so nothing
here names a curve; a ciphersuite is constants and hash instantiations,
nothing more. It arrives through one of two seams, and which one a function
takes is the point: a `Ciphersuite` where the RFC's own transcript is being
derived, and the protocol-agnostic
[`PrimeOrderGroup`](group.py) everywhere else — the dealer, the
interpolation, the commitment-list validation. A second threshold protocol
over these curves reuses the latter and brings its own transcript. The seam
is not `Signature`'s: a threshold scheme signs in two communication rounds with
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
from typing import Generic, Protocol, runtime_checkable

from sig_frx.threshold.group import E, PrimeOrderGroup


@runtime_checkable
class Ciphersuite(PrimeOrderGroup[E], Protocol[E]):
    """What RFC 9591 §6 instantiates per suite: the group and five hashes.

    The group half is [`PrimeOrderGroup`](group.py), and it is not this
    RFC's — it is what any threshold protocol over the same curves needs,
    and what a second one reuses. What §6 adds is `h1`–`h5`, and those are
    Schnorr's: the binding factor, the challenge and the nonce derivation
    are FROST's own transcript, so a protocol with a different transcript
    brings its own hashes to the same group rather than these. Which seam a
    function takes says which half it needs; the signatures are where that
    is written down.

    `element_size` is here rather than on the group because its readers are
    each suite's own `verify`, unpacking that suite's `R ‖ z` — a §6.x
    constant read off `self`, not a value crossing the group seam.
    """

    element_size: int

    def h1(self, message: bytes) -> int: ...

    def h2(self, message: bytes) -> int: ...

    def h3(self, message: bytes) -> int: ...

    def h4(self, message: bytes) -> bytes: ...

    def h5(self, message: bytes) -> bytes: ...


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


def _require_nonzero_scalar(group: PrimeOrderGroup, identifier: int) -> None:
    """RFC 9591's identifier domain check: a NonZeroScalar, `[1, order-1]`.

    Runs before an identifier meets a field op — an out-of-range int
    operand aborts instead of reducing (see the Protocol docstring).
    """
    if not 1 <= identifier <= group.order - 1:
        raise ValueError("a participant identifier is a NonZeroScalar")


@dataclass(frozen=True)
class _Decoded(Generic[E]):
    """A validated commitment list: two element batches and their order.

    `identifiers` is what indexes the batches — the list is sorted and
    duplicate-free by the time one of these exists, so a participant's
    position is `identifiers.index(...)` and nothing else needs storing. The
    batches stay batches rather than being split into per-participant entries,
    because splitting them is what the seam's shape exists to avoid.
    """

    identifiers: list[int]
    hidings: E
    bindings: E

    def index_of(self, identifier: int) -> int:
        """Where a participant sits in the batches.

        Returning the index rather than the elements is what keeps `E` opaque:
        taking them out is `select_elements`, which is the group's operation
        because only the group knows what a batch is made of.
        """
        return self.identifiers.index(identifier)


def _validated_commitment_list(
    group: PrimeOrderGroup[E], commitment_list: list[Commitment]
) -> tuple[list[int], E, E]:
    """§5.2's MUST-checks on the list: sorted, distinct, deserializable.

    Returns `(identifiers, hidings, bindings)` — the two element batches
    positionally aligned with the identifiers, which the list being sorted
    already makes a stable order. The validation paid for the decompression
    and the group commitment consumes the same points, so decoding them a
    second time would be pure waste.

    Both batches decode in one call each rather than one per participant. The
    list is the batch axis this protocol has, and it is the only one — which
    is why `t` participants cost two seam calls here and not `2t`.
    """
    identifiers = [c.identifier for c in commitment_list]
    if identifiers != sorted(identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError(
            "a commitment list is sorted by identifier, without duplicates"
        )
    for identifier in identifiers:
        _require_nonzero_scalar(group, identifier)
    hidings = group.deserialize_elements([c.hiding for c in commitment_list])
    bindings = group.deserialize_elements([c.binding for c in commitment_list])
    return identifiers, hidings, bindings


def encode_group_commitment_list(
    group: PrimeOrderGroup, commitment_list: list[Commitment]
) -> bytes:
    """RFC 9591 §4.3: the byte string the binding factors hash over."""
    return b"".join(
        group.serialize_scalar(entry.identifier) + entry.hiding + entry.binding
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
    group: PrimeOrderGroup[E],
    identifiers: list[int],
    hidings: E,
    bindings: E,
    binding_factors: dict[int, int],
) -> E:
    """RFC 9591 §4.5: `Σ (D_i + [ρ_i]E_i)`, over already-decoded elements.

    The sum is a reduction over the participant axis, not an accumulator
    threaded through a loop. That is the whole difference between this and the
    formula as written: `Σ` is what the standard says, and a fold was only
    ever how a per-element seam could spell it.
    """
    rho = [binding_factors[identifier] for identifier in identifiers]
    return group.sum_elements(
        group.elements_add(hidings, group.elements_scalar_mult(bindings, rho))
    )


def compute_challenge(
    cs: Ciphersuite,
    group_commitment: bytes,
    group_public_key: bytes,
    message: bytes,
) -> int:
    """RFC 9591 §4.6: `H2(R ‖ PK ‖ msg)` — RFC 8032's challenge for Ed25519."""
    return cs.h2(group_commitment + group_public_key + message)


def derive_interpolating_value(
    group: PrimeOrderGroup, participants: list[int], identifier: int
) -> int:
    """RFC 9591 §4.2: the Lagrange coefficient `λ_i` at zero."""
    if identifier not in participants:
        raise ValueError("the identifier is not among the participants")
    if len(set(participants)) != len(participants):
        raise ValueError("a participant appears more than once")
    for entry in participants:
        _require_nonzero_scalar(group, entry)
    field = group.scalar_field
    numerator, denominator = field(1), field(1)
    for other in participants:
        if other == identifier:
            continue
        numerator = numerator * other
        denominator = denominator * (other - identifier)
    return int(numerator / denominator)


def _signing_context(
    cs: Ciphersuite[E],
    commitment_list: list[Commitment],
    group_public_key: bytes,
    message: bytes,
) -> tuple[_Decoded[E], dict[int, int], E, int]:
    """The round-two prologue all three §5 surfaces share.

    Validate the list, bind, commit, challenge — one home so the surfaces
    cannot drift on the MUST-checks; `aggregate` ignores the challenge.
    """
    identifiers, hidings, bindings = _validated_commitment_list(cs, commitment_list)
    binding_factors = compute_binding_factors(
        cs, group_public_key, commitment_list, message
    )
    group_commitment = compute_group_commitment(
        cs, identifiers, hidings, bindings, binding_factors
    )
    (encoded,) = cs.serialize_elements(group_commitment)
    challenge = compute_challenge(cs, encoded, group_public_key, message)
    return (
        _Decoded(identifiers, hidings, bindings),
        binding_factors,
        group_commitment,
        challenge,
    )


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
    cs: Ciphersuite[E],
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
    (encoded,) = cs.serialize_elements(group_commitment)
    return encoded + cs.serialize_scalar(z)


def verify_share(
    cs: Ciphersuite[E],
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
    index = decoded.index_of(identifier)
    hiding = cs.select_elements(decoded.hidings, [index])
    binding_commitment = cs.select_elements(decoded.bindings, [index])
    commitment_share = cs.elements_add(
        hiding,
        cs.elements_scalar_mult(binding_commitment, [binding_factors[identifier]]),
    )
    expected = cs.elements_add(
        commitment_share,
        cs.elements_scalar_mult(
            cs.deserialize_elements([participant_public_key]),
            [int(cs.scalar_field(challenge) * lambda_i)],
        ),
    )
    (encoded,) = cs.serialize_elements(expected)
    return cs.scalar_base_mult(share) == encoded


def polynomial_evaluate(group: PrimeOrderGroup, x: int, coefficients: list[int]) -> int:
    """RFC 9591 Appendix C.1.1: Horner evaluation over the scalar field."""
    field = group.scalar_field
    x_field = field(x)
    value = field(0)
    for coefficient in reversed(coefficients):
        value = value * x_field + coefficient
    return int(value)


def secret_share_split(
    group: PrimeOrderGroup,
    secret: int,
    coefficients: list[int],
    max_participants: int,
) -> list[tuple[int, int]]:
    """RFC 9591 Appendix C.1's shard: shares `(i, f(i))` for `i = 1..n`.

    `coefficients` are the dealer's random non-constant terms — a parameter,
    never drawn here, which is what lets the vectors' shares reproduce.
    """
    if not 2 <= max_participants < group.order:
        raise ValueError("MAX_PARTICIPANTS is at least 2 and below the order")
    polynomial = [secret % group.order] + [c % group.order for c in coefficients]
    return [
        (i, polynomial_evaluate(group, i, polynomial))
        for i in range(1, max_participants + 1)
    ]


def vss_commit(
    group: PrimeOrderGroup, secret: int, coefficients: list[int]
) -> list[bytes]:
    """RFC 9591 Appendix C.2: the coefficient commitments `[φ_0, …, φ_t]`.

    `φ_0 = [s]B` is the group public key, which is why the dealer publishes
    this vector rather than the key alone.
    """
    return [group.scalar_base_mult(c) for c in [secret, *coefficients]]


def vss_verify(
    group: PrimeOrderGroup[E], identifier: int, share: int, commitment: list[bytes]
) -> bool:
    """RFC 9591 Appendix C.2: `[f(i)]B = Σ i^j·φ_j` — a participant's check."""
    _require_nonzero_scalar(group, identifier)
    field = group.scalar_field
    powers, power = [], field(1)
    for _ in commitment:
        powers.append(int(power))
        power = power * identifier
    expected = group.sum_elements(
        group.elements_scalar_mult(group.deserialize_elements(commitment), powers)
    )
    (encoded,) = group.serialize_elements(expected)
    return group.scalar_base_mult(share) == encoded
