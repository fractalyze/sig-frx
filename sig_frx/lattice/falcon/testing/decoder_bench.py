# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Which part of Falcon's decoder a GPU verification is actually waiting for.

[`encoding.decompress`](../encoding.py) is 20% of a verification on a workstation
CPU and **69%** of one on an RTX 5090, so on the GPU leg it is the stage a caller
waits for and the one nobody has shaped for that backend
([`falcon.py`](../falcon.py) carries both halves of the pair). A share that size
is a reason to look inside it, and "inside it" is what this prints: the decoder
broken into the six steps it is, measured on whichever leg is running.

## The decomposition is a ladder of prefixes, not a bag of isolated parts

Timing each step as its own program is the measurement that has already been
wrong here once. The shared compaction's scatter form won its isolated stage by
up to 6.3x and moved `verify` by nothing at all, because in situ it fuses with
the SHAKE above it and the arithmetic below it and never pays the memory round
trip a standalone program is forced to pay
([`compaction_bench`](../../testing/compaction_bench.py)). An isolated stage
bounds what changing it can buy; it does not estimate it.

So each rung here is `decompress` **stopped** after one more step — the whole
decoder up to that cut, fused as XLA would fuse it — and the cost attributed to a
step is the difference between two neighbouring rungs. A step is therefore priced
inside everything that precedes it rather than beside it, and the rungs are
anchored at both ends: rung `input` reads the argument and does nothing else, and
rung `tail` is the whole function, which the `sig_decode` row beside it has to
agree with. Two things that follow:

- **A marginal can come out at or below zero.** Adding a step can let XLA fuse a
  chain it was materializing before, and a negative marginal is that, not a
  measurement error. It is reported as measured.
- **A cut is a fusion barrier the real function does not have.** Each rung ends
  in a reduction to one scalar per row — `_digest`, which is what keeps XLA from
  deleting the prefix as dead — and a reduction fuses into its producer. The
  ladder therefore prices the steps of a decoder that is cut where the real one
  is whole, which is the residue this method cannot remove. What it can do is
  show the residue's size, and that is what the `tail` rung against `sig_decode`
  is for.

## The transcription is pinned, not trusted

A ladder needs cut points inside a function body, so `_upto` is `decompress`
written out again with an exit after each step — a copy, and a copy drifts. It is
held to the original on every run: `--pinned` decodes the same batch through both
and refuses the whole table unless the coefficients and the verdict are
byte-identical. That is the same standard
[`rejection_test`](../../testing/rejection_test.py) holds the compaction's two
forms to, for the same reason — a transcription is right because it reproduces
the function, not because it looks like it.

## The method the rest of it inherits

Taken from `compaction_bench`, which established these on the measurement this
one follows up:

- **Interleaved, always.** Every rung and every anchor alternates inside a round
  and the reported figure is the median of round medians. Run-to-run spread on
  these stages is around 20%, and a block of one program followed by a block of
  another measures whatever the machine was doing during each block.
- **One session, or it is not a share.** `verify`, `HashToPoint` and `sig_decode`
  print beside the ladder from the same run, because this stage re-measured on
  the same commit in a later session has drifted 25% while its total held to 4%.
  A share spliced across two runs is two machines compared.
- **Real signatures.** The batch is the published round-3 vectors tiled up from
  several distinct instances, which is what `verify` is gated on anyway, so where
  the searches land is a real signature's terminator pattern.

Cold cost prints beside warm for the reason the sibling does: a step that builds
a large index expression pays for it once per distinct shape, and a column-less
compile is a decision taken without half its price.

    bazel run //sig_frx/lattice/falcon/testing:decoder_bench -- --batches=1024
    FRX_PLATFORMS=cuda bazel run //sig_frx/lattice/falcon/testing:decoder_bench
"""

from __future__ import annotations

import contextlib
import statistics
import time
from collections.abc import Callable, Iterator, Sequence
from functools import lru_cache
from typing import Any, NamedTuple

import frx
import frx.numpy as fnp
import numpy as np
from absl import app, flags
from frx import lax

from sig_frx.lattice.falcon import encoding, falcon
from sig_frx.lattice.falcon.testing import falcon_vectors

_SETS = flags.DEFINE_list(
    "parameter_sets",
    ["Falcon-1024"],
    "Falcon parameter sets to measure. Both decode by the same machine and only"
    " the string length changes; 1024 is where the decoder's GPU share was taken.",
)
_BATCHES = flags.DEFINE_list(
    "batches",
    ["1024"],
    "Batch sizes to measure. 1024 is the batch the 69% share comes from; a"
    " smaller one says whether the ranking's cost is per-row or per-batch.",
)
_ROUNDS = flags.DEFINE_integer(
    "rounds", 3, "Interleaved rounds; the reported figure is the median of them."
)
_SAMPLES = flags.DEFINE_integer(
    "samples", 40, "Timed calls per rung per round; the median of them is a round."
)
_INSTANCES = flags.DEFINE_integer(
    "instances",
    8,
    "Distinct published signatures tiled up to each batch, so neighbouring rows"
    " do not share a terminator pattern.",
)
_PINNED = flags.DEFINE_bool(
    "pinned",
    True,
    "Decode the batch through both `_upto` and `encoding.decompress` and refuse"
    " the table unless they agree. The ladder is a transcription; this is what"
    " stops it drifting from the function it claims to divide.",
)
_ANCHORS = flags.DEFINE_bool(
    "anchors",
    True,
    "Also time `verify`, `HashToPoint` and `sig_decode` in the same interleaved"
    " run, so the ladder is comparable to the stage table it comes from.",
)
_AB = flags.DEFINE_bool(
    "ab",
    True,
    "Also A/B the two forms of the `mask` step against `verify` and `sig_decode`,"
    " interleaved. The ladder says which step to aim at; only this says whether"
    " hitting it moves the operation.",
)


# -- the two forms of the `mask` step ---------------------------------------


def _terminator_masks() -> np.ndarray:
    """The terminator byte a value produces, indexed by that value and its entry state.

    The step below reads this instead of walking. It exists because the eight
    positions a byte closes at are a function of the byte and the state it is
    entered in and of nothing else — the same closure property `_BYTE_STEP`
    already uses for the transition — so the whole within-byte chain is one
    `[256, 9]` host constant. Built by running the per-bit machine, for the
    reason its two siblings in [`encoding.py`](../encoding.py) are: a table and
    the machine it stands for cannot be allowed to disagree.

    The bit packed for position `offset` is `1 << (7 − offset)`, which is
    §3.11.1's order and the order `_TERMINATOR_OFFSET` reads back.
    """
    table = np.zeros((256, encoding._ON_ZERO.shape[0]), dtype=np.uint8)
    for value in range(256):
        for start in range(encoding._ON_ZERO.shape[0]):
            state, mask = start, 0
            for shift in range(7, -1, -1):
                bit = (value >> shift) & 1
                if state == encoding._UNARY_STATE and bit:
                    mask |= 1 << shift
                state = int((encoding._ON_ONE if bit else encoding._ON_ZERO)[state])
            table[value, start] = mask
    return table


_TERMINATOR_MASK = _terminator_masks()


def _mask_by_chain(body: Any, entered: Any, within: Any) -> Any:
    """The shipped form: seven dependent steps, then the terminators packed.

    Statement for statement `decompress`'s own, which is what makes it the
    control in the A/B rather than a re-derivation of it.
    """
    walked = [entered]
    for offset in range(7):
        walked.append(
            fnp.where(
                within[:, offset] != 0,
                fnp.take(encoding._ON_ONE, walked[-1].astype(np.int32)),
                fnp.take(encoding._ON_ZERO, walked[-1].astype(np.int32)),
            )
        )
    closes = [
        (walked[offset] == np.uint8(encoding._UNARY_STATE)) & (within[:, offset] != 0)
        for offset in range(8)
    ]
    return sum(
        closes[offset].astype(np.uint8) << np.uint8(7 - offset) for offset in range(8)
    )


def _mask_by_table(body: Any, entered: Any, within: Any) -> Any:
    """The candidate: the same byte, read off `[256, 9]` in one gather.

    Seven dependent steps become one lookup, which is the whole of the change.
    `within` is unused here — that is the point, since the bits it reopens are
    exactly what the table already knows.
    """
    rows = fnp.take(_TERMINATOR_MASK, body.astype(np.int32), axis=0)
    return fnp.take_along_axis(rows, entered.astype(np.int32)[:, None], axis=-1)[:, 0]


_FORMS: dict[str, Callable[[Any, Any, Any], Any]] = {
    "chain": _mask_by_chain,
    "table": _mask_by_table,
}


# -- the ladder ------------------------------------------------------------

# `decompress`'s steps, in the order it runs them. Each name is a rung: the
# decoder stopped after that step, with everything before it fused as usual.
# `input` is the floor — the argument read and reduced, no decoding at all — so
# that the first real step is priced against something rather than against zero.
#
# `mask` is one rung and not two because the reformulation replaces exactly that
# span: a rung boundary that does not coincide with the change boundary prices
# something nobody can act on.
_RUNGS = (
    "input",
    "bits",
    "scan",
    "mask",
    "rank",
    "search",
    "locate",
    "tail",
)

# What each rung adds, for the table's second column.
_ADDS = {
    "input": "the argument, read and reduced",
    "bits": "§3.11.1's bit expansion, [slen] from [slen/8]",
    "scan": "the associative_scan over [256, 9] byte transitions",
    "mask": "the terminator byte — the form under test",
    "rank": "the cumsum that ranks the terminators",
    "search": "the n searchsorted, and `terminated`",
    "locate": "the [n] gather and the [256] offset table",
    "tail": "the seven bit reads, the sign, and the two rejections",
}


def _digest(live: Sequence[Any]) -> Any:
    """One int32 per row from every value a rung leaves live.

    A rung has to return something that depends on all of its work or XLA
    deletes it, and it has to return something *small* or the timing is a
    transfer. A sum per live array is both. It is not free — a reduction over
    `[slen]` is real work on either leg — but `stream` is live in every rung
    from `bits` down, so that cost is common to all of them and cancels in the
    differences the table is actually about.
    """
    total = None
    for value in live:
        part = fnp.sum(value.astype(np.int32))
        total = part if total is None else total + part
    return total


def _upto(
    data: Any,
    n: int,
    rung: str = "tail",
    form: Callable[[Any, Any, Any], Any] = _mask_by_chain,
) -> tuple[Any, ...]:
    """`decompress`'s body, stopped after `rung`, as the values still live there.

    Statement for statement the function in [`encoding.py`](../encoding.py),
    with an exit after each step and the `mask` step swappable. Read that one
    for what any of it means; the only things added here are where the exits are
    and which form computes the terminator byte. `--pinned` is what holds every
    combination of the two to the original.
    """
    body = fnp.asarray(data, dtype=np.uint8)
    if rung == "input":
        return (body,)

    stream = encoding.bytes_to_bits_high_first(body)
    slen = stream.shape[-1]
    if rung == "bits":
        return (stream,)

    reached = lax.associative_scan(
        lambda first, second: fnp.take_along_axis(
            second, first.astype(np.int32), axis=-1
        ),
        fnp.take(encoding._BYTE_STEP, body.astype(np.int32), axis=0),
        axis=0,
    )[:, 0]
    entered = fnp.concatenate([fnp.zeros(1, dtype=np.uint8), reached[:-1]])
    if rung == "scan":
        return (stream, entered)

    within = stream.reshape(-1, 8)
    mask = form(body, entered, within)
    if rung == "mask":
        return (stream, mask)

    ranks = fnp.cumsum(mask != np.uint8(0), axis=-1, dtype=np.int32)
    wanted = fnp.arange(1, n + 1, dtype=np.int32)
    if rung == "rank":
        return (stream, mask, ranks)

    holders = fnp.searchsorted(ranks, wanted)
    terminated = ranks[..., -1] >= np.int32(n)
    if rung == "search":
        return (stream, mask, holders, terminated)

    holding = fnp.take(mask, holders, mode="clip").astype(np.int32)
    offsets = fnp.take(encoding._TERMINATOR_OFFSET, holding).astype(np.int32)
    ends = holders * np.int32(8) + offsets
    starts = fnp.concatenate([fnp.zeros(1, dtype=np.int32), ends[:-1] + np.int32(1)])
    if rung == "locate":
        return (stream, starts, ends, terminated)

    def read(offset: int) -> Any:
        return fnp.take(stream, starts + np.int32(offset), mode="clip").astype(np.int32)

    low = sum(read(1 + j) << np.int32(6 - j) for j in range(7))
    magnitude = low + ((ends - starts - np.int32(encoding._HEADER_BITS)) << np.int32(7))
    negative = read(0) == np.int32(1)

    canonical = ~((magnitude == np.int32(0)) & negative).any(axis=-1)
    position = fnp.arange(slen, dtype=np.int32)
    padded = ~((stream != 0) & (position > ends[..., -1])).any(axis=-1)

    coefficients = fnp.where(negative, -magnitude, magnitude)
    return coefficients, terminated & canonical & padded


# -- timing ----------------------------------------------------------------


def _blocked(value: Any) -> Any:
    """Wait for a dispatch to land, so a timing measures work and not queueing."""
    frx.block_until_ready(value)
    return value


class _Prepared(NamedTuple):
    """One compiled program, and what compiling it cost."""

    call: Callable[[], Any]
    cold: float


_Build = Callable[[], Callable[[], Any]]


def _prepare(build: _Build, routing: Callable[[], Any] | None = None) -> _Prepared:
    """`build`'s program, traced under `routing` and warm, with its cold cost.

    `routing` is a factory rather than a context manager, because a generator
    one is spent after a single `with` and every program that shares a form
    needs its own.

    The cold column is the first call, which is the compile — timed here rather
    than inferred, because a rung that builds a large index expression pays for
    it once per distinct shape and that price belongs beside the warm one.
    """
    frx.clear_caches()
    with routing() if routing is not None else contextlib.nullcontext():
        call = build()
        start = time.perf_counter()
        _blocked(call())
        cold = time.perf_counter() - start
    return _Prepared(call, cold)


def _round(call: Callable[[], Any], samples: int) -> float:
    """The median of `samples` timed calls, in seconds. Assumes a warm caller."""
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        _blocked(call())
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def _interleaved(prepared: dict[str, _Prepared]) -> dict[str, float]:
    """Median-of-round-medians per entry, with the entries alternating in a round.

    The alternation is the point, and it is what makes a *difference* between
    two rungs readable at all: consecutive rungs here differ by a few percent of
    a verification, which is inside the spread a blocked run would hand back.
    """
    seen: dict[str, list[float]] = {label: [] for label in prepared}
    for _ in range(_ROUNDS.value):
        for label, entry in prepared.items():
            seen[label].append(_round(entry.call, _SAMPLES.value))
    return {label: statistics.median(times) for label, times in seen.items()}


# -- the inputs ------------------------------------------------------------


@lru_cache(maxsize=None)
def _block(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The published vectors to tile a batch out of, parsed once per set.

    Filtered to one message length first, the way `compaction_bench` filters
    them: a batch carries one static length and the KAT messages do not all
    share one.
    """
    whole = [
        vector
        for vector in falcon_vectors.vectors(name)
        if vector.public_key is not None
        and vector.message is not None
        and vector.signature is not None
    ]
    length = len(whole[0].message or b"")
    chosen = [vector for vector in whole if len(vector.message or b"") == length]
    parts = [
        np.stack(
            [
                np.frombuffer(getattr(vector, field), dtype=np.uint8)
                for vector in chosen[: _INSTANCES.value]
            ]
        )
        for field in ("public_key", "message", "signature")
    ]
    return parts[0], parts[1], parts[2]


def _tiled(block: np.ndarray, batch: int) -> Any:
    """`block`'s instances repeated up to `batch` rows, on device."""
    repeats = -(-batch // block.shape[0])
    return fnp.asarray(np.concatenate([block] * repeats, axis=0)[:batch])


def _batch(name: str, batch: int) -> tuple[Any, Any, Any]:
    """A batch of published Falcon signatures, tiled to `batch`."""
    keys, messages, signatures = (_tiled(part, batch) for part in _block(name))
    return keys, messages, signatures


def _bodies(name: str, batch: int) -> Any:
    """What `decompress` is handed: the bytes past the header and the salt."""
    _, _, signatures = _batch(name, batch)
    return signatures[:, 1 + encoding.SALT_SIZE :]


# -- the programs ----------------------------------------------------------


def _rung_program(
    name: str,
    batch: int,
    rung: str,
    form: Callable[[Any, Any, Any], Any] = _mask_by_chain,
) -> _Build:
    """The decoder stopped after `rung`, over a batch, as `verify` maps it.

    A fresh `frx.jit` per rung: a shared wrapper would answer the second
    lowering out of the first one's trace cache and report one rung twice.
    """
    params = falcon.PARAMETER_SETS[name]

    def build() -> Callable[[], Any]:
        data = _bodies(name, batch)
        program = frx.jit(
            frx.vmap(lambda one: _digest(_upto(one, params.n, rung, form)))
        )
        return lambda: program(data)

    return build


@contextlib.contextmanager
def _decompressing_by(form: Callable[[Any, Any, Any], Any]) -> Iterator[None]:
    """Route `encoding.decompress` through `form` for the duration.

    The swap is on the module attribute rather than on a parameter, because what
    has to be measured is `verify` — and nothing between `verify` and the
    decoder takes a form to pass down. `sig_decode` resolves `decompress` off
    the module at call time, so this reaches it.

    Traced inside the routing and timed outside it, the way
    [`compaction_bench`](../../testing/compaction_bench.py) routes the
    compaction: once an executable exists the routing cannot change under it,
    so two warm forms alternate freely.
    """
    original = encoding.decompress

    def decompress(data: Any, n: int) -> tuple[Any, ...]:
        return _upto(data, n, "tail", form)

    encoding.decompress = decompress
    try:
        yield
    finally:
        encoding.decompress = original


def _routing(form: Callable[[Any, Any, Any], Any]) -> Callable[[], Any]:
    """`_decompressing_by(form)`, as the factory `_prepare` wants.

    A factory and not a bound instance: a generator context manager is spent
    after one `with`, and each program traced under a form needs its own.
    """
    return lambda: _decompressing_by(form)


def _verify_program(name: str, batch: int) -> _Build:
    """`verify` over a batch of published signatures — what a caller waits for."""
    scheme = falcon.named(name)

    def build() -> Callable[[], Any]:
        arguments = _batch(name, batch)
        program = frx.jit(scheme.verify)
        return lambda: program(*arguments)

    return build


def _challenge_program(name: str, batch: int) -> _Build:
    """`HashToPoint`, the stage the CPU leg waits for instead."""
    params = falcon.PARAMETER_SETS[name]

    def build() -> Callable[[], Any]:
        _, messages, signatures = _batch(name, batch)
        salted = fnp.concatenate(
            [signatures[:, 1 : 1 + encoding.SALT_SIZE], messages], axis=-1
        )
        program = frx.jit(frx.vmap(lambda body: falcon.hash_to_point(body, params.n)))
        return lambda: program(salted)

    return build


def _decoder_program(name: str, batch: int) -> _Build:
    """`sig_decode`, the whole stage — the number the ladder has to add up to."""
    params = falcon.PARAMETER_SETS[name]

    def build() -> Callable[[], Any]:
        _, _, signatures = _batch(name, batch)
        program = frx.jit(
            frx.vmap(
                lambda one: encoding.sig_decode(one, params.n, params.signature_size)
            )
        )
        return lambda: program(signatures)

    return build


def _pin(name: str, batch: int) -> None:
    """Refuse the table unless every form's last rung *is* `decompress`.

    Both sides run here rather than one being a stored expectation, because the
    thing that drifts is the transcription against the function as it stands
    now, not against the function as it was when a constant was written down.

    Every form, not just the control: a candidate that decodes differently is
    not a faster decoder, and a ratio taken over one is the most expensive kind
    of wrong number to publish. This is the same standard `rejection_test` holds
    the compaction's two forms to — each against the specification's answer,
    rather than against each other.
    """
    params = falcon.PARAMETER_SETS[name]
    data = _bodies(name, batch)
    real = frx.jit(frx.vmap(lambda one: encoding.decompress(one, params.n)))(data)
    if not np.asarray(real[1]).all():
        raise RuntimeError(
            f"{name} B={batch}: a published signature did not decode, so this "
            f"table would time a rejected batch rather than a decode"
        )
    for label, form in _FORMS.items():
        walked = frx.jit(frx.vmap(lambda one: _upto(one, params.n, "tail", form)))(data)
        for part, (mine, theirs) in enumerate(zip(walked, real)):
            if not np.array_equal(np.asarray(mine), np.asarray(theirs)):
                raise RuntimeError(
                    f"{name} B={batch}: form `{label}` disagrees with "
                    f"`encoding.decompress` on output {part}, so it is not a form "
                    f"of that function and no row below is about it"
                )


def _routes(name: str, batch: int) -> dict[str, int]:
    """The compiled size of `sig_decode` under each form, and a refusal if they tie.

    The A/B routes by swapping a module attribute, and the failure mode of that
    is silent: if the swap does not reach the trace, both columns time the same
    executable and the ratio is 1.00x — which reads exactly like a real result
    that says the change is worth nothing. The two claims are indistinguishable
    from the numbers alone, so the program itself is asked instead.

    The `mask` rung cannot stand in for this. It takes its form as an argument
    and never routes, so it would differ even if the routing were dead.
    """
    params = falcon.PARAMETER_SETS[name]
    _, _, data = _batch(name, batch)
    sizes: dict[str, int] = {}
    for label, form in _FORMS.items():
        with _decompressing_by(form):
            # A fresh `frx.jit` per form: a shared one answers the second
            # lowering out of the first one's trace cache and reports one form
            # twice, which is the same silence this function exists to break.
            program = frx.jit(
                frx.vmap(
                    lambda one: encoding.sig_decode(
                        one, params.n, params.signature_size
                    )
                )
            )
            sizes[label] = len(program.lower(data).as_text().splitlines())
    if len(set(sizes.values())) == 1:
        raise RuntimeError(
            f"{name} B={batch}: every form lowers `sig_decode` to {sizes} lines, so "
            f"the routing did not reach the trace and the A/B below would time one "
            f"executable twice"
        )
    return sizes


# -- reporting -------------------------------------------------------------


def _ab(name: str, batch: int) -> None:
    """The two forms against the operation, and against the step they differ in.

    Three rows and not one. The step says what the change is worth at its
    ceiling, `sig_decode` says how much of that survives the decoder around it,
    and `verify` says how much reaches a caller — which is the only one of the
    three that decides anything, because the shared compaction won its stage by
    up to 6.3x and moved `verify` by nothing
    ([`compaction_bench`](../../testing/compaction_bench.py)).
    """
    builds: dict[str, _Build] = {
        "verify": _verify_program(name, batch),
        "sig_decode": _decoder_program(name, batch),
    }
    routed = _routes(name, batch) if _PINNED.value else {}

    print()
    print(f"Falcon {name} mask form at B = {batch}, one run", flush=True)
    if routed:
        print(f"  routed: `sig_decode` lowers to {routed} lines", flush=True)
    header = (
        f"{'measurement':<14}{'chain':>12}{'table':>12}{'ratio':>9}"
        f"{'cold chain':>12}{'cold table':>12}"
    )
    print(header)
    print("-" * len(header), flush=True)

    prepared: dict[str, _Prepared] = {}
    for label, form in _FORMS.items():
        for measurement, build in builds.items():
            prepared[f"{measurement}/{label}"] = _prepare(build, _routing(form))
        # The rung takes its form as an argument, so it needs no routing: it is
        # here to give the operation rows a ceiling measured in the same run.
        prepared[f"mask/{label}"] = _prepare(_rung_program(name, batch, "mask", form))
    timed = _interleaved(prepared)

    for measurement in ("verify", "sig_decode", "mask"):
        chain = timed[f"{measurement}/chain"]
        table = timed[f"{measurement}/table"]
        print(
            f"{measurement:<14}{chain * 1e3:>10.3f}ms{table * 1e3:>10.3f}ms"
            f"{chain / table:>8.2f}x{prepared[f'{measurement}/chain'].cold:>11.1f}s"
            f"{prepared[f'{measurement}/table'].cold:>11.1f}s",
            flush=True,
        )
    print(
        "  ratio above 1.00 favours the table. `mask` is the rung, so it is the"
        " ceiling; `verify` is what a caller waits for, so it is the answer.",
        flush=True,
    )


def _report(name: str, batch: int) -> None:
    """One run: the anchors and the ladder, interleaved with each other."""
    if _PINNED.value:
        _pin(name, batch)

    builds: dict[str, _Build] = {}
    if _ANCHORS.value:
        builds["verify"] = _verify_program(name, batch)
        builds["HashToPoint"] = _challenge_program(name, batch)
        builds["sig_decode"] = _decoder_program(name, batch)
    for rung in _RUNGS:
        builds[rung] = _rung_program(name, batch, rung)

    print()
    print(f"Falcon {name} decoder at B = {batch}, one run", flush=True)
    header = (
        f"{'step':<14}{'cumulative':>12}{'marginal':>11}{'share':>8}{'cold':>10}"
        "  what it adds"
    )
    print(header)
    print("-" * len(header), flush=True)

    prepared = {label: _prepare(build) for label, build in builds.items()}
    timed = _interleaved(prepared)

    if _ANCHORS.value:
        whole = timed["verify"]
        for label in ("verify", "HashToPoint", "sig_decode"):
            share = f"{timed[label] / whole:.0%}" if whole else "-"
            print(
                f"{label:<14}{timed[label] * 1e3:>10.3f}ms{'':>11}{share:>8}"
                f"{prepared[label].cold:>9.1f}s  of `verify`",
                flush=True,
            )
        print("-" * len(header), flush=True)

    stage = timed.get("sig_decode")
    previous = None
    for rung in _RUNGS:
        cumulative = timed[rung]
        marginal = ""
        if previous is not None:
            marginal = f"{(cumulative - previous) * 1e3:>9.3f}ms"
        share = ""
        if previous is not None and stage:
            share = f"{(cumulative - previous) / stage:.0%}"
        print(
            f"{rung:<14}{cumulative * 1e3:>10.3f}ms{marginal:>11}{share:>8}"
            f"{prepared[rung].cold:>9.1f}s  {_ADDS[rung]}",
            flush=True,
        )
        previous = cumulative
    print(
        "  cumulative is the decoder stopped after that step; marginal is the"
        " difference from the row above; share is that difference against"
        " `sig_decode`.",
        flush=True,
    )
    if _AB.value:
        _ab(name, batch)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError(f"unexpected arguments: {argv[1:]}")
    print(f"backend: {frx.default_backend()}   devices: {frx.devices()}", flush=True)
    print(
        f"rounds: {_ROUNDS.value}, samples: {_SAMPLES.value}, "
        f"instances: {_INSTANCES.value}",
        flush=True,
    )
    for name in _SETS.value:
        for batch in (int(entry) for entry in _BATCHES.value):
            _report(name, batch)


if __name__ == "__main__":
    app.run(main)
