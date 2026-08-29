# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The known-answer-test harness: one driver, any `Signature` implementation.

Every scheme here is gated on published vectors. The formats differ per standard
— ACVP JSON for FIPS 204 and 205, inline vectors in RFC 8391, the Falcon
submission's own files — while the shape of the test does not: load a vector, run
keygen / sign / verify, compare bytes. So a loader normalizes its format into
`KatVector` and `check` drives any implementation of the seam.

**The negative cases are not optional, and there is no flag to skip them.** A
verifier that returns `True` unconditionally passes every positive vector ever
published, which makes a suite of positives alone evidence of nothing. `check`
derives a tampered batch from whatever positives it is handed and requires the
rejection, so a scheme never gets the choice.

**The batch axis is gated on a batch `check` builds, because no validation
program publishes one.** A batch needs one static shape and the published sets
vary the message length per case, so grouping them yields singletons and a
per-entry check has no second entry to pin — which leaves the one property the
seam exists for ungated by exactly the vectors that are supposed to gate a
scheme. So `check` replicates an accepted case and moves a bit in some entries:
the multiplicity is the only invented part, since the accepting entries carry the
standard's own signature and the rejecting ones carry it corrupted.

**Both derived passes need a case the standard accepts, and an operation whose
verification set publishes none is handed one.** A moved bit is evidence only
against something that verified before it moved, which is what makes the accepted
case the starting point of the tampering pass and of the batch axis alike. ACVP's
sigVer sets are mostly deliberate failures and draw each case's pre-hash function
at random, so whole operations arrive with nothing accepted in them — and there
both passes no-op, leaving a green run that cannot be told apart from one where
they ran. So `check` takes `accepted_case`, which the caller sources from
elsewhere in the same published set rather than producing itself. It is held to
everything the published vectors are held to, the operation it belongs to first
of all, and it is refused where the set publishes an accepted case of its own,
since what it stands in for is then already there.

**Where not even that reaches, the call site declares it.** A stand-in is
published bytes from somewhere else, so an operation the standard covers nowhere
has none to be handed. `check` refuses that instead of shrinking quietly; a
caller that means it declares it, and the declaration is itself an error once the
set stops matching it.

**One scheme's signature is not fixed by its standard, so signing compares
something else there.** The shape above — load, run, compare bytes — assumes the
standard determines a signature from a case's inputs, and Falcon does not: a salt
is drawn per signature and the sampler's stream is expanded from it by a route
§3.9 never fixes, so two correct implementations disagree by construction. Byte
comparison would fail a correct signer, so a call site says that is its situation
and `check` verifies the produced signature instead. It is the weaker check and
what makes it not circular is stated where it happens: the verifier it leans on
is gated on the same published signatures, and the keypair is upstream's.

**A vector the harness cannot run faithfully is an error, not a skip.** Silently
dropping a field — a published failure verdict, a mode marker selecting a
different operation — reports a pass for a case that was never run, which is the
one outcome worse than a failure.

**Not every scheme comes through here, and the exception is narrow.** This
normalizes formats and drives `Signature`, so a scheme with neither — no published
vector file to parse, and no seam-shaped `sign` because it is stateful — gates in
its own test instead. Satisfying the Protocol for such a scheme would mean an
adapter whose `sign` discards the advanced key, which is not a different operation
the way the internal and pre-hash interfaces are: it is the operation with the
property that makes it safe removed. `docs/reference/testing.md` states what
that scheme owes in exchange.

**A standard publishes vectors per operation, so the caller says which one it is
running.** FIPS 205 publishes its internal interface and its pre-hash variant
beside the plain one, and a scheme implements those under its own names rather
than on the seam. Both reach the harness as an implementation of `Signature` —
usually a thin adapter — which cannot say for itself which operation it performs,
so `check` takes that as an argument and requires every vector to agree. The
alternative is a harness that guesses, and a guess here is a green suite that
verified the wrong thing.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array

from sig_frx.signature import Signature


@dataclass(frozen=True)
class KatVector:
    """One case, normalized out of whatever format published it."""

    # Identifies the case in a failure message. A bare index is useless when a
    # sweep of thousands fails on one.
    case_id: str
    parameter_set: str
    seed: bytes | None = None
    public_key: bytes | None = None
    secret_key: bytes | None = None
    message: bytes | None = None
    signature: bytes | None = None
    # Signing randomness for a randomized (hedged) case; `None` selects the
    # deterministic mode.
    randomness: bytes | None = None
    # The published verdict. ACVP's sigVer sets are mostly deliberate failures,
    # so this is load-bearing rather than a formality.
    valid: bool = True
    # The standard's application context string, which goes into the message the
    # scheme signs.
    context: bytes | None = None
    # Which of the standard's interfaces the case selects. A standard publishes
    # vectors for every operation its interface defines, and they are separate
    # operations rather than options on one: `external` signs the message wrapped
    # with a domain separator and the context, `internal` signs it as given. The
    # seam names the external one; a scheme that implements the internal one
    # exposes it under its own name, so this says which of them a case belongs to
    # rather than whether it can run.
    interface: str = "external"
    # The pre-hash function's published name for a pre-hash case, `None` for a
    # pure one. The scheme signs a digest under this function and the function's
    # identifier is part of what gets signed, so a case naming one the caller
    # cannot compute is unrunnable rather than approximable.
    pre_hash: str | None = None
    # Whether the case is the deterministic signing variant, where the standard
    # defines both, and `None` where it defines one or where the operation does
    # not depend on it. `check` requires it to agree with the scheme instance,
    # since the two modes produce different signatures from the same key.
    deterministic: bool | None = None
    # Published fields the record cannot express, by name. FIPS 204's external-mu
    # variant takes a pre-computed message representative in place of a message,
    # which no operation here names. A loader records what it could not feed
    # rather than dropping it, and `check` refuses the vector, because running one
    # operation against a vector published for another reports a pass for a case
    # nobody ran.
    unsupported: tuple[str, ...] = ()


class KatError(Exception):
    """A vector set the harness will not run, or a scheme that failed it."""


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

# ACVP splits a vector set in two: prompt.json holds the inputs, and
# expectedResults.json the outputs, joined on (tgId, tcId). The field names are
# per-mode, the join is not.
#
# Key generation takes its seed as several named pieces, and the order below is
# the order the standard's keygen takes them (FIPS 205 §9.1 is
# `(SK.seed, SK.prf, PK.seed)`; FIPS 204 has the single `seed`). Concatenating
# them here is what lets the seam keep one `keygen(seed)` for both.
_ACVP_SEED_FIELDS = ("seed", "skSeed", "skPrf", "pkSeed")

_ACVP_BYTE_FIELDS = {
    "sk": "secret_key",
    "pk": "public_key",
    "message": "message",
    "signature": "signature",
    # The signing randomness of a hedged case. The two standards name it
    # differently — FIPS 204's `rnd` and FIPS 205's `addrnd` — for the same role.
    "rnd": "randomness",
    "additionalRandomness": "randomness",
    "context": "context",
}

# Mode markers whose *value* decides whether the case is an operation anything
# here names. `externalMu` selects one whose input is a pre-computed message
# representative rather than a message, which nothing implements, so a case
# setting it is refused rather than run.
_ACVP_BENIGN_MODES: dict[str, frozenset[Any]] = {
    "externalMu": frozenset({False}),
}

# Mode markers the record expresses, so a case selecting one is routed rather
# than refused: the interface, the pure/pre-hash variant and its hash function,
# and the signing mode.
_ACVP_MODE_FIELDS = frozenset(
    {"signatureInterface", "preHash", "hashAlg", "deterministic"}
)

_ACVP_INTERFACES = frozenset({"external", "internal"})
_ACVP_PURE = "pure"
_ACVP_PREHASH = "preHash"

# Fields that identify or annotate a case rather than feed it. Anything outside
# these, the mapped fields and the mode fields above is a mode nothing here
# expresses, and is recorded on the vector as unsupported rather than ignored — a
# new ACVP field then surfaces as a refusal instead of a silent behavior change.
_ACVP_IGNORED_FIELDS = frozenset(
    {"tgId", "tcId", "testType", "parameterSet", "tests", "testPassed", "reason"}
)


def _acvp_modes(
    merged: dict[str, Any]
) -> tuple[str, str | None, bool | None, list[str]]:
    """The interface, pre-hash function and signing mode a case selects.

    Returns the names of any mode field whose published value it could not
    express alongside them, so an unrecognized value is refused rather than
    silently read as the benign one.
    """
    unexpressed: list[str] = []

    interface = merged.get("signatureInterface", "external")
    if interface not in _ACVP_INTERFACES:
        unexpressed.append("signatureInterface")
        interface = "external"

    pre_hash: str | None = None
    variant = merged.get("preHash", _ACVP_PURE)
    if variant == _ACVP_PREHASH:
        pre_hash = merged.get("hashAlg")
        if pre_hash is None:
            # A pre-hash case has to name its function: the identifier goes into
            # the message, so guessing one signs something else.
            unexpressed.append("preHash")
    elif variant != _ACVP_PURE:
        unexpressed.append("preHash")

    deterministic = merged.get("deterministic")
    if deterministic is not None and not isinstance(deterministic, bool):
        unexpressed.append("deterministic")
        deterministic = None

    return interface, pre_hash, deterministic, unexpressed


def load_acvp(prompt_path: Path | str, expected_path: Path | str) -> list[KatVector]:
    """Normalize one ACVP vector set — a `(prompt, expectedResults)` pair."""
    prompt = json.loads(Path(prompt_path).read_text())
    expected = json.loads(Path(expected_path).read_text())

    results = {
        (group["tgId"], test["tcId"]): test
        for group in expected["testGroups"]
        for test in group["tests"]
    }

    vectors: list[KatVector] = []
    for group in prompt["testGroups"]:
        parameter_set = group["parameterSet"]
        for test in group["tests"]:
            key = (group["tgId"], test["tcId"])
            if key not in results:
                raise KatError(
                    f"{parameter_set} tg{key[0]}/tc{key[1]} has no expected result; "
                    f"the prompt and expectedResults files are not a matching pair"
                )
            # Group-level fields (a shared key, a mode flag) are merged under the
            # test's own, which win — ACVP puts a field at whichever level it is
            # constant, and that level differs per set.
            merged = {**group, **results[key], **test}
            fields: dict[str, Any] = {
                dest: bytes.fromhex(merged[src])
                for src, dest in _ACVP_BYTE_FIELDS.items()
                if src in merged
            }
            seed = b"".join(
                bytes.fromhex(merged[name])
                for name in _ACVP_SEED_FIELDS
                if name in merged
            )
            interface, pre_hash, deterministic, unexpressed = _acvp_modes(merged)
            unsupported = tuple(
                sorted(
                    set(unexpressed)
                    | {
                        name
                        for name, value in merged.items()
                        if name not in _ACVP_IGNORED_FIELDS
                        and name not in _ACVP_BYTE_FIELDS
                        and name not in _ACVP_SEED_FIELDS
                        and name not in _ACVP_MODE_FIELDS
                        and value not in _ACVP_BENIGN_MODES.get(name, frozenset())
                    }
                )
            )
            vectors.append(
                KatVector(
                    case_id=f"{parameter_set}/tg{key[0]}/tc{key[1]}",
                    parameter_set=parameter_set,
                    seed=seed or None,
                    valid=bool(merged.get("testPassed", True)),
                    interface=interface,
                    pre_hash=pre_hash,
                    deterministic=deterministic,
                    unsupported=unsupported,
                    **fields,
                )
            )
    return vectors


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


def load_wycheproof_p1363(path: Path | str, parameter_set: str) -> list[KatVector]:
    """Normalize one Wycheproof fixed-width ECDSA verification set.

    Verify-only by publication: each group carries one uncompressed public key
    over many cases, and no case carries a seed or a secret key, so `check`'s
    keygen and signing passes have nothing to run and the verdicts are the
    gate. `parameter_set` is the caller's label — the file names its curve and
    hash, but the string a failure message needs is the caller's instance.

    Only the `valid` and `invalid` verdicts are accepted. Other Wycheproof
    families publish `acceptable` for cases whose verdict is a policy choice;
    picking a policy silently here would turn those cases into whatever the
    implementation already does, so a third verdict is refused until someone
    states one.
    """
    data = json.loads(Path(path).read_text())
    schema = data.get("schema")
    if schema != "ecdsa_p1363_verify_schema_v1.json":
        raise KatError(
            f"expected the fixed-width ECDSA verification schema, got {schema!r}; "
            f"the DER sets gate a parser, not the seam's encoding"
        )
    vectors: list[KatVector] = []
    for group in data["testGroups"]:
        public_key = bytes.fromhex(group["publicKey"]["uncompressed"])
        for test in group["tests"]:
            result = test["result"]
            if result not in ("valid", "invalid"):
                raise KatError(
                    f"{parameter_set} tcId {test['tcId']} publishes verdict "
                    f"{result!r}, which is a policy choice this loader refuses "
                    f"to make"
                )
            vectors.append(
                KatVector(
                    case_id=(
                        f"{parameter_set} tcId {test['tcId']} ({test['comment']})"
                    ),
                    parameter_set=parameter_set,
                    public_key=public_key,
                    message=bytes.fromhex(test["msg"]),
                    signature=bytes.fromhex(test["sig"]),
                    valid=result == "valid",
                )
            )
    if len(vectors) != data["numberOfTests"]:
        raise KatError(
            f"{parameter_set}: the file declares {data['numberOfTests']} cases "
            f"and holds {len(vectors)}; the fetch or the schema drifted"
        )
    return vectors


def to_bytes(value: Any) -> bytes:
    """The wire form of a key or signature, for comparison against a vector.

    A uint8 array is the only form the harness can compare, because bytes are
    what a standard publishes. A scheme whose key or signature pytree is
    something else needs a wire codec at the seam before its vectors can be run
    — and that is a seam decision, not one a test harness should make silently.
    """
    array = np.asarray(value)
    if array.dtype != np.uint8:
        raise KatError(
            f"expected a uint8 array to compare against published bytes, got "
            f"dtype {array.dtype}; the scheme needs a wire codec at the seam"
        )
    return bytes(array.reshape(-1))


def _stack(items: Sequence[Any]) -> Any:
    """Stack like-shaped pytrees along a new leading batch axis."""
    return frx.tree_util.tree_map(lambda *leaves: fnp.stack(leaves), *items)


def _as_array(data: bytes) -> Array:
    return fnp.asarray(np.frombuffer(data, dtype=np.uint8))


def _flip_first_bit(data: bytes) -> bytes:
    return bytes([data[0] ^ 1]) + data[1:]


def _corrupt_signature(vector: KatVector) -> KatVector:
    assert vector.signature is not None
    return replace(vector, signature=_flip_first_bit(vector.signature))


def _corrupt_message(vector: KatVector) -> KatVector:
    assert vector.message is not None
    return replace(vector, message=_flip_first_bit(vector.message))


def _corrupt_public_key(vector: KatVector) -> KatVector:
    assert vector.public_key is not None
    return replace(vector, public_key=_flip_first_bit(vector.public_key))


# The three inputs a verifier is given, each of which must be able to break
# it. Keyed by the KatVector attribute; a failure message spells the label as
# the attribute with its underscore read as a space.
_TAMPERINGS: tuple[tuple[str, Callable[[KatVector], KatVector]], ...] = (
    ("signature", _corrupt_signature),
    ("message", _corrupt_message),
    ("public_key", _corrupt_public_key),
)

# The batch `_check_batch_axis` builds, and which of its entries carry a moved
# bit. Two of each, so a `verify` that reduced over the batch fails whichever way
# it reduced: `all` rejects the entries carrying the published signature, `any`
# accepts the ones carrying it corrupted.
_BATCH_AXIS_ENTRIES = 4
_BATCH_AXIS_TAMPERED = frozenset({1, 2})


def verify_cases(vectors: Sequence[KatVector]) -> list[KatVector]:
    """The cases carrying everything one verification needs.

    A published set covers several operations at once and a mode publishes what
    that mode is about, so a key generation case has no signature and a signing
    case no public key. Public here because whether a call has a verify case at
    all is what decides whether it needs an accepted one — a question its caller
    answers before it can source one.
    """
    return [
        v
        for v in vectors
        if v.public_key is not None
        and v.message is not None
        and v.signature is not None
    ]


def check(
    scheme: Signature,
    vectors: Sequence[KatVector],
    *,
    interface: str = "external",
    pre_hash: str | None = None,
    accepted_case: KatVector | None = None,
    no_accepted_case: str | None = None,
    not_the_published_signature: str | None = None,
) -> None:
    """Run every check the standard requires, plus the tampering it does not.

    `vectors` are one scheme instance's — one parameter set, one signing mode, one
    operation. Raises `KatError` on the first failure, naming the case.

    `interface` and `pre_hash` are the caller's statement of *which* operation
    `scheme` performs, and every vector must agree with it. A `Signature` cannot
    say for itself: an implementation of the internal interface or of the pre-hash
    variant satisfies the same Protocol, usually as a thin adapter over the
    scheme's own entry point. Declaring it at the call site is what keeps the
    harness from running one operation against vectors published for another,
    which would report a pass for a case nobody ran.

    `accepted_case` is a case this operation's verification set does not publish,
    sourced by the caller from where the standard did publish one. Everything the
    harness derives starts from an accepted case, and a set of failures has none
    to start from; this is that starting point, and it runs as a vector like any
    other — its verdict compared, its inputs tampered with, its batch built — so
    a stand-in that does not verify fails here rather than passing quietly.

    `no_accepted_case` is the caller's statement that nothing in the published set
    reaches this operation, and why. It is the last resort behind `accepted_case`
    and costs what that one buys: the call then runs the published verdicts and
    nothing else, which the caller knows to expect and a green run cannot say. It
    is declared for the same reason `interface` is, and like a wrong interface it
    fails once it stops describing the set.

    `not_the_published_signature` is the caller's statement that this scheme does
    not produce the published signature bytes, and why. Signing then verifies what
    it produced instead of comparing it, which is the only comparison available
    where the standard fixes no map from a case's inputs to a signature. It is
    weaker than the byte comparison it replaces and says so: what it catches is a
    signer that produces something its own verifier refuses, not a signer that
    disagrees with the standard. Like the declarations above it fails once it stops
    describing the set — a call where every case *does* reproduce its published
    bytes is one that should be comparing them.
    """
    if not vectors:
        raise KatError("no vectors: an empty set passes trivially and proves nothing")
    if accepted_case is not None and no_accepted_case is not None:
        raise KatError(
            f"an accepted case was supplied ({accepted_case.case_id}) and the same "
            f"call declares there is none to supply ({no_accepted_case}); the "
            f"declaration stands in for the derived checks and the case makes them "
            f"run, so exactly one of them describes this call"
        )

    supplied = [] if accepted_case is None else [accepted_case]
    # The stand-in is held to the same identity as the published cases: it is
    # meant to be this operation's, and one belonging to another would gate the
    # wrong thing while looking like coverage.
    _reject_unrunnable(scheme, [*vectors, *supplied], interface, pre_hash)
    operation = _operation(vectors, interface, pre_hash)
    _check_keygen(scheme, vectors)
    _check_sign(scheme, vectors, operation, not_the_published_signature)
    _check_verify(
        scheme,
        vectors,
        operation,
        accepted_case,
        no_accepted_case,
    )


def _operation(
    vectors: Sequence[KatVector], interface: str, pre_hash: str | None
) -> str:
    """The operation this call runs, for a message about the call as a whole.

    Read off the first vector because `_reject_unrunnable` has already required
    every one of them to agree: one parameter set, and the declared interface and
    pre-hash function.
    """
    variant = "" if pre_hash is None else f"/{pre_hash}"
    return f"{vectors[0].parameter_set} {interface}{variant}"


def _reject_unrunnable(
    scheme: Signature,
    vectors: Sequence[KatVector],
    interface: str,
    pre_hash: str | None,
) -> None:
    """Refuse a set carrying anything this call cannot run faithfully."""
    unrunnable = [v for v in vectors if v.unsupported]
    if unrunnable:
        names = sorted({name for v in unrunnable for name in v.unsupported})
        raise KatError(
            f"{len(unrunnable)} of {len(vectors)} vectors carry unsupported "
            f"fields {names} (first: {unrunnable[0].case_id}). Each selects an "
            f"operation nothing here names, so running another against them would "
            f"report a pass for a case nobody ran. Wire the subset this scheme "
            f"implements explicitly."
        )

    parameter_sets = {v.parameter_set for v in vectors}
    if len(parameter_sets) != 1:
        raise KatError(
            f"one scheme instance is one parameter set, got {sorted(parameter_sets)}; "
            f"group the vectors and build an instance per set"
        )

    mismatched = [
        v for v in vectors if (v.interface, v.pre_hash) != (interface, pre_hash)
    ]
    if mismatched:
        published = sorted({(v.interface, v.pre_hash) for v in mismatched})
        raise KatError(
            f"this call runs the {interface} interface with pre-hash {pre_hash!r}, "
            f"and {len(mismatched)} of {len(vectors)} vectors were published for "
            f"{published} (first: {mismatched[0].case_id}); group by operation and "
            f"pass the implementation of each"
        )

    # The two signing modes give different signatures from one key, so a set run
    # against the wrong instance would fail case by case with nothing pointing at
    # the cause.
    modes = {v.deterministic for v in vectors if v.deterministic is not None}
    if len(modes) > 1:
        raise KatError(
            f"one instance is one signing mode, got both in {sorted(parameter_sets)}; "
            f"group the vectors by their `deterministic` flag"
        )
    if modes and modes != {scheme.deterministic}:
        raise KatError(
            f"the vectors are the "
            f"{'deterministic' if modes == {True} else 'hedged'} mode and the "
            f"scheme is the {'deterministic' if scheme.deterministic else 'hedged'} "
            f"one; build the instance the vector set was published for"
        )
    for vector in vectors:
        # A hedged case carries its randomness and a deterministic one does not;
        # if those disagree the loader mapped the wrong field.
        if vector.deterministic is not None and vector.deterministic != (
            vector.randomness is None
        ):
            raise KatError(
                f"{vector.case_id}: published as the "
                f"{'deterministic' if vector.deterministic else 'hedged'} mode but "
                f"{'carries' if vector.randomness is not None else 'carries no'} "
                f"signing randomness"
            )


def _check_keygen(scheme: Signature, vectors: Sequence[KatVector]) -> None:
    """Keygen is deterministic in the seed, and reproduces the published bytes."""
    for vector in vectors:
        if vector.seed is None or (
            vector.public_key is None and vector.secret_key is None
        ):
            continue
        public_key, secret_key = scheme.keygen(_as_array(vector.seed))
        if vector.public_key is not None and to_bytes(public_key) != vector.public_key:
            raise KatError(f"{vector.case_id}: keygen produced the wrong public key")
        if vector.secret_key is not None and to_bytes(secret_key) != vector.secret_key:
            raise KatError(f"{vector.case_id}: keygen produced the wrong secret key")


def _check_sign(
    scheme: Signature,
    vectors: Sequence[KatVector],
    operation: str,
    not_the_published_signature: str | None,
) -> None:
    """Signing reproduces the published signature, byte for byte.

    That comparison is the whole check wherever a standard fixes one signature per
    `(key, message, randomness)`, which is most of them: the deterministic modes by
    construction, and the hedged ones once the published `rnd` is fed back in.

    **Where it does not, the call site says so and this verifies instead.** Falcon
    draws a salt per signature and expands the sampler's stream from it by a route
    the specification never fixes, so two correct implementations disagree on the
    output bytes and the published signature is one valid answer among many.
    Comparing bytes there fails a correct signer, so the declaration switches the
    comparison rather than dropping the case.

    **What that costs is stated rather than papered over.** A produced signature
    checked by this repo's own verifier is a round trip, and
    `docs/reference/testing.md` ranks a round trip below the published bytes for
    the reason a self-consistent wrong implementation round-trips forever. Two
    things keep it from being circular. The verifier on the other side of it is
    gated independently, by `_check_verify` against those same published
    signatures. And the key is upstream's: signing runs under the published secret
    key and verifies under the published public key of the same record, so a pass
    binds this signer to a keypair it did not choose.

    The declaration is held to the set like every other one here. A call that
    reproduces its published bytes everywhere has no need of it and is told to
    compare them, and a call that declares it while signing nothing has made a
    claim about cases it never ran.
    """
    signed: list[KatVector] = []
    reproduced = 0
    for vector in vectors:
        if vector.secret_key is None or vector.message is None:
            continue
        if vector.signature is None or not vector.valid:
            continue
        if not_the_published_signature is not None and vector.public_key is None:
            raise KatError(
                f"{vector.case_id}: declared as not producing the published "
                f"signature ({not_the_published_signature}), so the produced one is "
                f"checked by this scheme's verifier instead — and this case carries "
                f"no public key to check it under. Expose one on the record or "
                f"leave the secret key off it."
            )
        randomness = None if vector.randomness is None else _as_array(vector.randomness)
        signature = scheme.sign(
            _as_array(vector.secret_key),
            _as_array(vector.message),
            randomness=randomness,
            context=None if vector.context is None else _as_array(vector.context),
        )
        produced = to_bytes(signature)
        if not_the_published_signature is None:
            if produced != vector.signature:
                raise KatError(
                    f"{vector.case_id}: signing produced the wrong signature"
                )
            continue
        reproduced += produced == vector.signature
        signed.append(replace(vector, signature=produced))

    if not_the_published_signature is None:
        return
    if not signed:
        raise KatError(
            f"{operation}: declared as not producing the published signature "
            f"({not_the_published_signature}), and not one case in this call signs "
            f"— every one is missing a secret key, a message, or an accepted "
            f"verdict. The declaration is about cases that ran, so drop it or give "
            f"the records what signing takes."
        )
    if reproduced == len(signed):
        plural = "" if len(signed) == 1 else "s"
        raise KatError(
            f"{operation}: declared as not producing the published signature "
            f"({not_the_published_signature}), and all {len(signed)} signed "
            f"case{plural} reproduced theirs exactly. The declaration buys a weaker "
            f"check than the byte comparison it replaces, so drop it and compare "
            f"the bytes this set turns out to fix."
        )

    # One call per equal-shape group, never one per case: the seam's unit is the
    # batch here for the same reason it is in `_check_verify`.
    for group in _group_by_shape(signed):
        for vector, verdict in zip(group, _verify_batch(scheme, group), strict=True):
            if not verdict:
                raise KatError(
                    f"{vector.case_id}: signing produced a signature this scheme's "
                    f"own verifier refuses"
                )


def _check_verify(
    scheme: Signature,
    vectors: Sequence[KatVector],
    operation: str,
    accepted_case: KatVector | None,
    no_accepted_case: str | None,
) -> None:
    """Verification agrees with every published verdict, and rejects tampering.

    The batch is the unit: one call per equal-length group, never a loop over
    cases, so the harness exercises the path a consumer actually runs. What those
    groups cannot reach is the batch axis itself, which is why the pass they feed
    is followed by one over a batch built here.

    Both of the passes below the published verdicts start from a case the
    standard accepts, so an operation whose verification set publishes none
    reduces to comparing verdicts — and the two things a comparison alone cannot
    separate are a verifier that rejects for the right reason and one that rejects
    everything. That is a property of the published set rather than of the scheme,
    which is why what fills the gap comes from the call site: a case sourced from
    where the standard did publish one, or, where the set holds none anywhere, the
    declaration that says so.

    Those two answer the same question, so they are wrong in the same two
    directions — an accepted case arriving in the set, and a call that has no
    verify case for the claim to be about.
    """
    runnable = verify_cases(vectors)
    accepted = [v for v in runnable if v.valid]
    already = _already_accepted(runnable, accepted)

    if accepted_case is not None:
        if already is not None:
            raise KatError(
                f"{operation}: an accepted case was supplied for the derived checks "
                f"to start from ({accepted_case.case_id}), and {already}. It stands "
                f"in for a case this operation does not publish, so drop it rather "
                f"than leave it standing in for one that is there."
            )
        runnable = [*runnable, accepted_case]
    elif no_accepted_case is not None:
        if already is not None:
            raise KatError(
                f"{operation}: declared as publishing no accepted case "
                f"({no_accepted_case}), and {already}. The declaration is what "
                f"stands in for the derived checks, so drop it rather than leave it "
                f"describing a set it no longer matches."
            )
    elif runnable and not accepted:
        plural = "" if len(runnable) == 1 else "s"
        raise KatError(
            f"{operation}: {len(runnable)} verify case{plural} and not one the "
            f"standard accepts, so the tampering pass and the batch axis have "
            f"nothing to move a bit in and neither runs — this call compares the "
            f"published verdicts and derives nothing from them. Supply an accepted "
            f"case for this operation from where the standard published one, or "
            f"declare it at the call site if the set publishes none anywhere; a "
            f"green run cannot say so on its own."
        )

    if not runnable:
        return

    for group in _group_by_shape(runnable):
        verdicts = _verify_batch(scheme, group)
        for vector, verdict in zip(group, verdicts, strict=True):
            if verdict != vector.valid:
                published = "accept" if vector.valid else "reject"
                raise KatError(
                    f"{vector.case_id}: published verdict is {published}, "
                    f"the scheme returned {'accept' if verdict else 'reject'}"
                )
        _check_tampering(scheme, [v for v in group if v.valid])

    _check_batch_axis(scheme, runnable)


def _already_accepted(
    runnable: Sequence[KatVector], accepted: Sequence[KatVector]
) -> str | None:
    """Why a call's claim to publish no accepted case does not describe it.

    `None` where the claim holds: this call verifies something, and none of it is
    a case the standard accepts.
    """
    if accepted:
        return (
            f"the set has an accepted case ({accepted[0].case_id}; "
            f"{len(accepted)} of {len(runnable)})"
        )
    if not runnable:
        return "this call has no verify case at all"
    return None


def _check_batch_axis(scheme: Signature, vectors: Sequence[KatVector]) -> None:
    """One accepted case replicated, with a bit moved in some of the entries.

    The pass above already requires a verdict per entry, and on published data it
    never gets to ask for one: every shape group is a singleton, so there is no
    second entry whose verdict a reduction could take. That leaves the seam's
    whole reason to exist — `verify` decides the batch rather than a loop over it
    — gated by nothing NIST published.

    The batch is therefore built rather than found, and its multiplicity is the
    only invented part: the accepting entries carry the standard's own signature
    over its own key and message, and the rejecting ones carry that signature
    with a bit moved. A `verify` that reduced over the batch fails, and so does
    one that ignored its input and accepted everything.

    Only the signature is moved, because which input broke the case is what the
    tampering pass covers, across all three of them; what is under test here is
    the axis the verdicts come back on.
    """
    accepted = next((v for v in vectors if v.valid), None)
    if accepted is None:
        return
    corrupted = _corrupt_signature(accepted)
    batch = [
        corrupted if index in _BATCH_AXIS_TAMPERED else accepted
        for index in range(_BATCH_AXIS_ENTRIES)
    ]
    for index, verdict in enumerate(_verify_batch(scheme, batch)):
        if verdict == (index not in _BATCH_AXIS_TAMPERED):
            continue
        replicated = (
            f"{accepted.case_id}: replicated across a batch of "
            f"{_BATCH_AXIS_ENTRIES}, entry {index} "
        )
        if index in _BATCH_AXIS_TAMPERED:
            raise KatError(
                f"{replicated}was accepted after a bit flip in the signature"
            )
        raise KatError(
            f"{replicated}carries the published signature and was rejected "
            f"because a different entry of the batch was tampered with — verify "
            f"is not deciding per entry"
        )


def _check_tampering(scheme: Signature, group: Sequence[KatVector]) -> None:
    """Every accepted case must be rejected once one of its three inputs moves.

    One call per input kind, with a bit moved in *every* entry that has one:
    each tampered entry must come back rejected. Whether a verdict belongs to
    its own entry — the reduction failure this pass used to also probe, one
    entry at a time and at a quadratic number of calls — is `_check_batch_axis`'s
    job, which mixes pinned and tampered entries in one batch; splitting the
    two keeps this pass linear in the group.

    An entry whose input is empty has no bit to move — Wycheproof publishes an
    accepted case over the empty message — so it rides along untampered, and
    its verdict is pinned to stay accepted.
    """
    if not group:
        return
    for field, corrupt in _TAMPERINGS:
        tampered_at = [bool(getattr(v, field)) for v in group]
        if not any(tampered_at):
            continue
        batch = [corrupt(v) if moved else v for v, moved in zip(group, tampered_at)]
        for vector, moved, verdict in zip(
            group, tampered_at, _verify_batch(scheme, batch)
        ):
            if moved and verdict:
                raise KatError(
                    f"{vector.case_id}: accepted after a bit flip in the "
                    f"{field.replace('_', ' ')}"
                )
            if not moved and not verdict:
                raise KatError(
                    f"{vector.case_id}: rejected while carrying the published "
                    f"inputs, because other entries of the batch were tampered "
                    f"with — verify is not deciding per entry"
                )


def _verify_batch(scheme: Signature, group: Sequence[KatVector]) -> list[bool]:
    public_keys = _stack([_as_array(v.public_key) for v in group])  # type: ignore[arg-type]
    messages = _stack([_as_array(v.message) for v in group])  # type: ignore[arg-type]
    signatures = _stack([_as_array(v.signature) for v in group])  # type: ignore[arg-type]
    context = group[0].context
    verdicts = scheme.verify(
        public_keys,
        messages,
        signatures,
        context=None if context is None else _as_array(context),
        # The harness drives the seam, so it names every per-call field. No
        # published set here carries a position: the one scheme that reads it
        # is gated on its own vectors, which are whole `(key, slot, root,
        # signature)` objects rather than this shape.
        position=None,
    )
    return [bool(v) for v in np.asarray(verdicts)]


def _group_by_shape(vectors: Sequence[KatVector]) -> list[list[KatVector]]:
    """Split into batches of equal byte lengths and one shared context.

    A batch axis needs one static shape, and published sets deliberately vary
    message length. The context is one value per call rather than per entry — the
    seam's shape, because a verifier serves one protocol domain — so cases with
    different contexts cannot share a batch either. Grouping keeps verification
    batched within each combination rather than falling back to a case-by-case
    loop.
    """
    groups: dict[tuple[int, int, int, bytes | None], list[KatVector]] = {}
    for vector in vectors:
        key = (
            len(vector.public_key or b""),
            len(vector.message or b""),
            len(vector.signature or b""),
            vector.context,
        )
        groups.setdefault(key, []).append(vector)
    return list(groups.values())
