# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The round-3 archive's per-call `SamplerZ` traces, parsed into records.

## Why these rather than Table 3.2's ten lines

§4.4 prints ten `SamplerZ` test vectors and then points at these, which the
submission package ships as `Supporting_Documentation/additional/`. They are the
same thing at a different scale — every call of one signature, 1 024 at
`Falcon-512` and 2 048 at `Falcon-1024` — and they carry the intermediates the
table has no room for.

That is the difference that matters. `testing.md` asks a gate on an
implementation to "pin the intermediates beneath them too, because a digest of a
final artifact says only *that* something is wrong": a wrong `ApproxExp` and a
wrong table lookup both surface as a wrong integer, and only one of the two is
where the bug is. These files carry, per iteration, the 72-bit `BaseSampler`
draw and the `z0` it produced, the sign bit and the byte it came from,
`ApproxExp`'s output, and each byte the acceptance comparison consumed. So each
layer of [`sampler.py`](../sampler.py) is gated on its own outputs.

They arrive through the `falcon_round3` archive already pinned in
[`MODULE.bazel`](../../../../MODULE.bazel) for the interop oracle, so the
sampler is gated on the same sha256 as the oracle and the transcribed vectors —
one artifact, three uses, and nothing new to keep in step.

## The randomness is reconstructed, not published separately

These files do not print a `randombytes` string the way Table 3.2 does. They
print each byte where it was consumed — `u = 0x...` for the nine
`BaseSampler` bytes, `(from random byte: 0x..)` for the sign bit and for every
byte the acceptance loop read. Concatenating them in file order rebuilds exactly
the stream the call consumed, which is what `randomness` below hands back.

That reconstruction is the one thing here that is this repo's rather than
upstream's, and it is self-checking: a stream assembled in the wrong order, or
missing a byte, does not produce a *different* answer — it desynchronises the
first `BaseSampler` draw and the case fails immediately.
"""

from __future__ import annotations

import re
from typing import Callable, NamedTuple

from python.runfiles import Runfiles

_RUNFILES = Runfiles.Create()

_PATH = (
    "falcon_round3/Supporting_Documentation/additional/test-vector-sampler-falcon{}.txt"
)

# One `BaseSampler` draw, the byte behind the sign bit, or a byte the acceptance
# comparison consumed — in the order the trace prints them, which is the order
# the call consumed them.
_CONSUMED = re.compile(
    r"BaseSampler: u = 0x([0-9A-Fa-f]{18}) -> z0 = (\d+)"
    r"|b\s+= (\d)  \(from random byte: 0x([0-9A-Fa-f]{2})\)"
    r"|i = \d+ -> w = -?\d+  \(from random byte: 0x([0-9A-Fa-f]{2})\)"
)
_HEX_FLOAT = r"([-+]0x[0-9A-Fa-f.]+p[-+]\d+)"
_CENTER = re.compile(rf"mu\s+= {_HEX_FLOAT}")
_INVERSE_SIGMA = re.compile(rf"1/sigma\s+= {_HEX_FLOAT}")
_CCS = re.compile(rf"^   ccs\s+= {_HEX_FLOAT}", re.M)
# `ApproxExp`'s output, after `BerExp` has doubled and shifted it.
_BER_EXP_Z = re.compile(r"^       z   = 0x([0-9A-Fa-f]{16})", re.M)
_ACCEPTED = re.compile(r"Accepted, ret = (-?\d+)")


class Iteration(NamedTuple):
    """One trip through Algorithm 15's loop, as the trace records it.

    Everything here is per-iteration data rather than a position in the byte
    stream, which is what lets a test check one algorithm without replaying the
    others: `BerExp` consumes a second comparison byte about one time in 256, so
    a test that re-drove the stream assuming one would silently desynchronise
    from that iteration on.
    """

    u: int  # the 72 bits `BaseSampler` read
    z0: int  # and the sample it returned
    bit: int  # Algorithm 15 line 5's sign bit
    shifted_exponential: int  # `BerExp`'s `z`, Algorithm 14 line 4


class Call(NamedTuple):
    """One `SamplerZ` call, with everything the trace records about it."""

    center: float
    inverse_sigma: float
    ccs: float
    randomness: bytes  # every byte the call consumed, in order
    iterations: tuple[Iteration, ...]
    result: int


def _field(pattern: re.Pattern[str], block: str, name: str) -> str:
    """The one capture of `pattern` in `block`, or an error naming what is missing.

    A trace whose format drifted would otherwise reach the assertions as a
    `None`, and the failure would name the attribute rather than the field.
    """
    match = pattern.search(block)
    if match is None:  # pragma: no cover - upstream format drift
        raise AssertionError(f"the trace records no {name} for a SamplerZ call")
    return match.group(1)


def _parse(text: str) -> tuple[Call, ...]:
    calls = []
    for block in text.split("SamplerZ:")[1:]:
        accepted = _ACCEPTED.search(block)
        if accepted is None:  # a trailing block the signature cut short
            continue
        consumed, draws, bits = bytearray(), [], []
        for match in _CONSUMED.finditer(block):
            u, z0, bit, bit_byte, comparison_byte = match.groups()
            if u is not None:
                consumed += bytes.fromhex(u)
                draws.append((int(u, 16), int(z0)))
            elif bit is not None:
                consumed += bytes.fromhex(bit_byte)
                bits.append(int(bit))
            else:
                consumed += bytes.fromhex(comparison_byte)
        shifted = [int(z, 16) for z in _BER_EXP_Z.findall(block)]
        if not len(draws) == len(bits) == len(shifted):
            raise AssertionError(
                f"the trace records {len(draws)} draws, {len(bits)} sign bits and "
                f"{len(shifted)} exponentials for one call; they pair per iteration"
            )
        calls.append(
            Call(
                center=float.fromhex(_field(_CENTER, block, "centre")),
                inverse_sigma=float.fromhex(_field(_INVERSE_SIGMA, block, "1/sigma")),
                ccs=float.fromhex(_field(_CCS, block, "ccs")),
                randomness=bytes(consumed),
                iterations=tuple(
                    Iteration(u=u, z0=z0, bit=bit, shifted_exponential=z)
                    for (u, z0), bit, z in zip(draws, bits, shifted)
                ),
                result=int(accepted.group(1)),
            )
        )
    return tuple(calls)


def calls(degree: int) -> tuple[Call, ...]:
    """Every `SamplerZ` call of the archive's signature at `degree`."""
    path = _RUNFILES.Rlocation(_PATH.format(degree))
    if path is None:  # pragma: no cover - a packaging error, not a case
        raise FileNotFoundError(
            f"the Falcon-{degree} sampler vectors are missing from the runfiles; "
            "the test target needs `@falcon_round3//:sampler_vectors` in `data`"
        )
    with open(path, encoding="utf-8") as handle:
        return _parse(handle.read())


def cursor(randomness: bytes) -> Callable[[int], bytes]:
    """`randomness` as the byte source `sampler.sampler_z` reads from.

    Refuses to run off the end rather than padding: a call that asked for more
    than the trace recorded has desynchronised, and zeros would turn that into a
    plausible wrong answer instead of a failure.
    """
    position = 0

    def take(count: int) -> bytes:
        nonlocal position
        chunk = randomness[position : position + count]
        if len(chunk) != count:
            raise AssertionError(
                f"the sampler asked for {count} bytes at offset {position}, past "
                f"the {len(randomness)} the published trace recorded"
            )
        position += count
        return chunk

    return take
