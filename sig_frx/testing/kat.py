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

**A vector the harness cannot run faithfully is an error, not a skip.** Silently
dropping a field — a published failure verdict, a mode marker selecting a
different operation — reports a pass for a case that was never run, which is the
one outcome worse than a failure.
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
    # Published fields the record cannot express, by name. A standard's vector
    # set covers every mode of its interface — FIPS 204 publishes a pre-hashed
    # variant, an internal interface, and an external-mu variant, each a
    # different operation from the one the seam names. A loader records what it
    # could not feed rather than dropping it, and `check` refuses the vector,
    # because running the seam's operation against a vector published for another
    # one reports a pass for a case nobody ran.
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
    "rnd": "randomness",
    "context": "context",
}

# Mode markers whose *value* decides whether the case is the operation the seam
# names. Anything outside the benign value — a pre-hashed message, the internal
# interface, a pre-computed message representative — is a different operation.
_ACVP_BENIGN_MODES: dict[str, frozenset[Any]] = {
    "signatureInterface": frozenset({"external"}),
    "preHash": frozenset({"pure"}),
    "externalMu": frozenset({False}),
}

# Fields that identify or annotate a case rather than feed it. Anything outside
# these and the mapped fields above is a mode the seam does not express, and is
# recorded on the vector as unsupported rather than ignored — a new ACVP field
# then surfaces as a refusal instead of a silent behavior change.
_ACVP_IGNORED_FIELDS = frozenset(
    {"tgId", "tcId", "testType", "parameterSet", "tests", "testPassed", "reason"}
)


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
            unsupported = tuple(
                sorted(
                    name
                    for name, value in merged.items()
                    if name not in _ACVP_IGNORED_FIELDS
                    and name not in _ACVP_BYTE_FIELDS
                    and name not in _ACVP_SEED_FIELDS
                    and value not in _ACVP_BENIGN_MODES.get(name, frozenset())
                )
            )
            vectors.append(
                KatVector(
                    case_id=f"{parameter_set}/tg{key[0]}/tc{key[1]}",
                    parameter_set=parameter_set,
                    seed=seed or None,
                    valid=bool(merged.get("testPassed", True)),
                    unsupported=unsupported,
                    **fields,
                )
            )
    return vectors


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


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


# The three inputs a verifier is given, each of which must be able to break it.
_TAMPERINGS: tuple[tuple[str, Callable[[KatVector], KatVector]], ...] = (
    ("signature", _corrupt_signature),
    ("message", _corrupt_message),
    ("public key", _corrupt_public_key),
)


def check(scheme: Signature, vectors: Sequence[KatVector]) -> None:
    """Run every check the standard requires, plus the tampering it does not.

    `vectors` are one scheme instance's — one parameter set. Raises `KatError`
    on the first failure, naming the case.
    """
    if not vectors:
        raise KatError("no vectors: an empty set passes trivially and proves nothing")

    _reject_unrunnable(vectors)
    _check_keygen(scheme, vectors)
    _check_sign(scheme, vectors)
    _check_verify(scheme, vectors)


def _reject_unrunnable(vectors: Sequence[KatVector]) -> None:
    """Refuse a set carrying anything the seam cannot express."""
    unrunnable = [v for v in vectors if v.unsupported]
    if unrunnable:
        names = sorted({name for v in unrunnable for name in v.unsupported})
        raise KatError(
            f"{len(unrunnable)} of {len(vectors)} vectors carry unsupported "
            f"fields {names} (first: {unrunnable[0].case_id}). Each selects an "
            f"operation the seam does not name, so running the plain one against "
            f"them would report a pass for a case nobody ran. Wire the subset "
            f"this scheme implements explicitly."
        )

    parameter_sets = {v.parameter_set for v in vectors}
    if len(parameter_sets) != 1:
        raise KatError(
            f"one scheme instance is one parameter set, got {sorted(parameter_sets)}; "
            f"group the vectors and build an instance per set"
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


def _check_sign(scheme: Signature, vectors: Sequence[KatVector]) -> None:
    """Signing reproduces the published signature, byte for byte."""
    for vector in vectors:
        if vector.secret_key is None or vector.message is None:
            continue
        if vector.signature is None or not vector.valid:
            continue
        randomness = None if vector.randomness is None else _as_array(vector.randomness)
        signature = scheme.sign(
            _as_array(vector.secret_key),
            _as_array(vector.message),
            randomness=randomness,
            context=None if vector.context is None else _as_array(vector.context),
        )
        if to_bytes(signature) != vector.signature:
            raise KatError(f"{vector.case_id}: signing produced the wrong signature")


def _check_verify(scheme: Signature, vectors: Sequence[KatVector]) -> None:
    """Verification agrees with every published verdict, and rejects tampering.

    The batch is the unit: one call per equal-length group, never a loop over
    cases, so the harness exercises the path a consumer actually runs.
    """
    runnable = [
        v
        for v in vectors
        if v.public_key is not None
        and v.message is not None
        and v.signature is not None
    ]
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


def _check_tampering(scheme: Signature, group: Sequence[KatVector]) -> None:
    """Every accepted case must be rejected once one of its three inputs moves.

    One bit, in one entry of the batch, with the other entries' verdicts pinned:
    a `verify` that reduced over the batch instead of deciding per entry fails
    here rather than passing everything forever.
    """
    if not group:
        return
    for field, corrupt in _TAMPERINGS:
        for index in range(len(group)):
            tampered = list(group)
            tampered[index] = corrupt(group[index])
            verdicts = _verify_batch(scheme, tampered)
            if verdicts[index]:
                raise KatError(
                    f"{group[index].case_id}: accepted after a bit flip in the "
                    f"{field}"
                )
            for other, verdict in enumerate(verdicts):
                if other != index and not verdict:
                    raise KatError(
                        f"{group[other].case_id}: rejected because a different "
                        f"entry of the batch was tampered with — verify is not "
                        f"deciding per entry"
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
