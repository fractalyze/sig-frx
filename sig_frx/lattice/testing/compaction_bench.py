# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Whether the shared compaction gathers or scatters, and whether that is one answer.

[`rejection.first_accepted`](../rejection.py) collects a sampler's survivors by
ranking them with a `cumsum` and looking each rank up with a `searchsorted` — a
gather. The other direction writes each survivor to the slot its rank names — a
scatter — and the two compute the same permutation, so which one the module
carries is a measurement rather than a preference, and this is the harness that
takes it. Whichever ships is held to the same answer by
[`rejection_test`](rejection_test.py), which states the property against a host
reference rather than against the other form: a form is right because it
compacts correctly, not because it agrees with its predecessor.

It is committed for the reason [`sign_bench`](../mldsa/testing/sign_bench.py) and
[`verify_bench`](../../hashbased/testing/verify_bench.py) are: a re-measurement
compares against the same harness rather than against a fresh one. The numbers
belong on the issue that acts on them and in the docstring of whatever the
decision changes, not here.

## Why a bench and not an argument

The direction was chosen from a claim about GPUs — that a scatter serialises
where a gather vectorises — on a box that had no GPU to check it on. Both halves
of that have since moved: the claim is refuted on CPU, where a scatter measured
several times *faster*, and it is refuted on GPU too in a sibling repo, where
enc-frx's `ml_kem.sampling._compact` found the winner **inverts by backend** and
now gates on `frx.default_backend()`. A shared helper carrying an unmeasured
performance rationale is the thing this exists to end.

## Three things it measures, and why none of them is optional

- **The compaction alone, per site, per backend.** The isolated stage is what a
  rationale in `rejection.py` is a claim about, so it is the one that has to be
  quotable there.
- **The whole operation.** A form can win the stage and lose the program: a
  sibling measurement had one variant 13% faster as a stage and 49% slower as the
  program it sat in. `verify` is what a caller waits for, so it decides.
- **The cold cost beside the warm one.** A compaction that builds a large index
  expression pays for it once per distinct shape, and a table with no compile
  column is not a decision. enc-frx's scatter bought about two seconds of compile
  by never building the index arithmetic at all.

## How the timings are taken, which is not a detail here

**The A/B is interleaved.** Timing every sample of one form and then every sample
of the other reported 1.07x for a change that interleaved medians put at 1.38x —
identical code, and run-to-run spread on these stages is around 20%. The whole
output of this bench is a ratio between two forms, so a non-interleaved block is
the difference between a decision and a coin flip. Each form is warmed and
compiled under its own routing, then timed outside it: once an executable exists
the routing cannot change under it, so the two alternate freely inside a round.

**A stage share and its total come from one session.** A stage re-measured on the
same commit in a later session drifted 25% while the total it belonged to matched
within 4%, so a share spliced across two runs is not a share. `--stages` prints
the Falcon breakdown and the `verify` it divides beside each other, in one run
and interleaved with each other, for that reason.

**The inputs are replayed from real calls.** The candidate stream and its
acceptance mask are captured from a concrete `keygen` and a concrete
`hash_to_point` rather than drawn from an acceptance rate, because where the
searches land *is* the survivor pattern. Several distinct instances per site are
captured and tiled up to the batch, so no row is the same draw as its neighbour.

    bazel run //sig_frx/lattice/testing:compaction_bench -- --batches=1,64,256
    FRX_PLATFORMS=cuda bazel run //sig_frx/lattice/testing:compaction_bench
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

from sig_frx.lattice import rejection
from sig_frx.lattice.falcon import encoding as falcon_encoding
from sig_frx.lattice.falcon import falcon
from sig_frx.lattice.falcon.testing import falcon_vectors
from sig_frx.lattice.mldsa import ml_dsa, sampling

_ML_DSA_SETS = flags.DEFINE_list(
    "ml_dsa_sets",
    ["ML-DSA-65"],
    "ML-DSA parameter sets to measure. The compaction has the same shape at all"
    " three and only its width changes, so one set settles the form and the"
    " others confirm it.",
)
_FALCON_SETS = flags.DEFINE_list(
    "falcon_sets",
    ["Falcon-1024"],
    "Falcon parameter sets to measure. Both rank the same way; 1024 is where"
    " `HashToPoint` is the pole by the widest margin.",
)
_BATCHES = flags.DEFINE_list(
    "batches",
    ["1", "64", "256"],
    "Batch sizes to measure. A crossover between two forms is a function of the"
    " batch — the sibling repo's sits between 256 and 1024 — so one batch size"
    " cannot answer this.",
)
_ROUNDS = flags.DEFINE_integer(
    "rounds", 3, "Interleaved rounds; the reported figure is the median of them."
)
_SAMPLES = flags.DEFINE_integer(
    "samples", 40, "Timed calls per form per round; the median of them is a round."
)
_INSTANCES = flags.DEFINE_integer(
    "instances",
    8,
    "Distinct captured draws tiled up to each batch, so neighbouring rows are not"
    " the same acceptance pattern.",
)
_STAGES = flags.DEFINE_bool(
    "stages",
    True,
    "Also print Falcon's stage breakdown beside the `verify` it divides, taken in"
    " this same run — the decoder's ranking is the second `searchsorted` on this"
    " path, and its share is what says whether it is worth a form of its own.",
)


# -- the two forms ---------------------------------------------------------


def _by_gather(values: Any, accepted: Any, ranks: Any, count: int) -> Any:
    """`rejection.first_accepted`'s form: rank the survivors, look each rank up.

    `cumsum` is non-decreasing, so the source of output `r` is where rank `r + 1`
    first appears. `clip` pins the shortfall — unreachable at the sized budget —
    to the last candidate on every backend.
    """
    del accepted
    wanted = fnp.arange(1, count + 1, dtype=np.int32)
    return frx.vmap(
        lambda row, rank: fnp.take(
            row, fnp.searchsorted(rank, wanted), axis=-1, mode="clip"
        )
    )(values, ranks)


def _by_scatter(values: Any, accepted: Any, ranks: Any, count: int) -> Any:
    """The other direction: write each survivor to the slot its rank names.

    The rejected candidates and any overflow past `count` are sent to a sink slot
    that is dropped, so the surviving slots are unique and the write is
    deterministic without needing an ordering guarantee. The shortfall lands
    differently from the gather's — an unwritten slot keeps the zero it was
    initialised with rather than taking the last candidate — which is a
    difference inside the branch the budget makes unreachable, and it is pinned
    on whichever form lands rather than left to a backend.
    """
    slots = fnp.where(
        accepted & (ranks <= np.int32(count)), ranks - np.int32(1), np.int32(count)
    )
    return frx.vmap(
        lambda row, slot: fnp.zeros(count + 1, dtype=row.dtype)
        .at[slot]
        .set(row)[:count]
    )(values, slots)


_FORMS: dict[str, Callable[..., Any]] = {"gather": _by_gather, "scatter": _by_scatter}


def _routed(form: Callable[..., Any]) -> Callable[..., Any]:
    """`first_accepted`'s signature over `form`. The scan and the check are shared."""

    def first_accepted(values: Any, accepted: Any, count: int, sampler: str) -> Any:
        ranks = fnp.cumsum(accepted, axis=-1, dtype=np.int32)
        rejection.require_enough(ranks[..., -1], count, sampler)
        return form(values, accepted, ranks, count)

    return first_accepted


@contextlib.contextmanager
def _compacting_by(form: Callable[..., Any]) -> Iterator[None]:
    """Every binding of `first_accepted` routed through `form` for the block.

    Two bindings and not one: `sampling` imported the name, so rebinding only the
    module attribute would move Falcon and leave ML-DSA where it was — a
    half-applied patch that still prints a table.
    """
    replacement = _routed(form)
    saved = [
        (module, getattr(module, "first_accepted")) for module in (rejection, sampling)
    ]
    for module, _ in saved:
        setattr(module, "first_accepted", replacement)
    try:
        yield
    finally:
        for module, original in saved:
            setattr(module, "first_accepted", original)


# -- timing ----------------------------------------------------------------


class _Prepared(NamedTuple):
    """One form's compiled program, and what compiling it cost."""

    call: Callable[[], Any]
    cold: float


_Build = Callable[[], Callable[[], Any]]
_Validate = Callable[[Any], None]


def _prepare(
    build: _Build, form: Callable[..., Any], validate: _Validate | None = None
) -> _Prepared:
    """`build`'s program, traced under `form` and warm, with the cold cost returned.

    Traced inside the routing and timed outside it. After the first call the
    executable is fixed, so a warm timing cannot be routed anywhere and the forms
    can alternate without a context manager between them.

    `validate` sees the value the cold call already produced, rather than being
    handed a call of its own: an extra call here would be served by the warm
    executable and quietly turn the cold column into a second warm one.
    """
    frx.clear_caches()
    with _compacting_by(form):
        call = build()
        start = time.perf_counter()
        # Wait for the dispatch to land before stopping the clock: a placed
        # program returns as soon as it is enqueued, so a timing without this
        # measures the enqueue and bills the arithmetic to whoever reads the
        # array next.
        outcome = frx.block_until_ready(call())
        cold = time.perf_counter() - start
    if validate is not None:
        validate(outcome)
    return _Prepared(call, cold)


def _round(call: Callable[[], Any], samples: int) -> float:
    """The median of `samples` timed calls, in seconds. Assumes a warm caller."""
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        frx.block_until_ready(call())
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def _interleaved(prepared: dict[str, _Prepared]) -> dict[str, float]:
    """Median-of-round-medians per entry, with the entries alternating in a round.

    The alternation is the point: a block of one form followed by a block of the
    other measures whatever the machine was doing during each block.
    """
    seen: dict[str, list[float]] = {label: [] for label in prepared}
    for _ in range(_ROUNDS.value):
        for label, entry in prepared.items():
            seen[label].append(_round(entry.call, _SAMPLES.value))
    return {label: statistics.median(times) for label, times in seen.items()}


# -- the inputs, captured from real calls -----------------------------------


class _Site(NamedTuple):
    """One place the helper is reached, and the draws it was reached with.

    `values` and `accepted` are `[instances, rows, candidates]`: the leading axis
    is distinct captured draws, and a batch tiles over it.
    """

    sampler: str
    values: np.ndarray
    accepted: np.ndarray
    keep: int


_Capture = tuple[str, np.ndarray, np.ndarray, int]


@contextlib.contextmanager
def _capturing(into: list[_Capture]) -> Iterator[None]:
    """Record every `(values, accepted, count)` the helper is called with.

    Only usable around a *concrete* call: a tracer has no values to record, which
    is why the capture runs `keygen` and `hash_to_point` on host inputs rather
    than the traced `verify` the numbers are eventually about. The shapes are the
    same ones either way — `verify` maps this body over its batch axis.
    """
    original = rejection.first_accepted

    def first_accepted(values: Any, accepted: Any, count: int, sampler: str) -> Any:
        into.append((sampler, np.asarray(values), np.asarray(accepted), count))
        return original(values, accepted, count, sampler)

    for module in (rejection, sampling):
        setattr(module, "first_accepted", first_accepted)
    try:
        yield
    finally:
        for module in (rejection, sampling):
            setattr(module, "first_accepted", original)


def _stack(captured: Sequence[_Capture]) -> list[_Site]:
    """The captures grouped by sampler, each an `[instances, ...]` stack."""
    order: list[str] = []
    grouped: dict[str, list[_Capture]] = {}
    for capture in captured:
        sampler = capture[0]
        if sampler not in grouped:
            order.append(sampler)
            grouped[sampler] = []
        grouped[sampler].append(capture)
    return [
        _Site(
            sampler,
            np.stack([values for _, values, _, _ in grouped[sampler]]),
            np.stack([accepted for _, _, accepted, _ in grouped[sampler]]),
            grouped[sampler][0][3],
        )
        for sampler in order
    ]


def _seed(instance: int, size: int) -> np.ndarray:
    """A distinct byte string per instance. Not random: a bench has to repeat."""
    return np.frombuffer(
        bytes((instance * 37 + position * 11) % 256 for position in range(size)),
        dtype=np.uint8,
    )


def _ml_dsa_sites(name: str) -> list[_Site]:
    """`ExpandA` and `ExpandS` as key generation reaches them.

    Key generation rather than `verify` because it is the concrete path — the
    rejection loop is a host `while` — so the draws are values here and tracers
    there. `verify` reaches only `ExpandA`; `ExpandS` is on the key path, shares
    the helper, and is measured where it runs.
    """
    scheme = ml_dsa.named(name, deterministic=True)
    captured: list[_Capture] = []
    with _capturing(captured), _generating_inputs():
        for instance in range(_INSTANCES.value):
            scheme.keygen(_seed(instance, 32))
    return _stack(captured)


def _falcon_sites(name: str) -> list[_Site]:
    """`HashToPoint` as verification reaches it, over distinct salted messages."""
    n = falcon.PARAMETER_SETS[name].n
    captured: list[_Capture] = []
    with _capturing(captured), _generating_inputs():
        for instance in range(_INSTANCES.value):
            falcon.hash_to_point(_seed(instance, 72), n)
    return _stack(captured)


def _tiled(block: np.ndarray, batch: int) -> Any:
    """`block`'s instances repeated up to `batch` rows, on device."""
    repeats = -(-batch // block.shape[0])
    return fnp.asarray(np.concatenate([block] * repeats, axis=0)[:batch])


# -- the programs ----------------------------------------------------------


def _stage_program(site: _Site, batch: int, form: Callable[..., Any]) -> _Build:
    """The compaction alone, over a batch, as `verify` maps it.

    A fresh `frx.jit` per form, deliberately: a shared wrapper would answer the
    second lowering out of the first one's trace cache and report one form twice.
    """

    def build() -> Callable[[], Any]:
        values = _tiled(site.values, batch)
        accepted = _tiled(site.accepted, batch)
        keep = site.keep

        def compact(row_values: Any, row_accepted: Any) -> Any:
            ranks = fnp.cumsum(row_accepted, axis=-1, dtype=np.int32)
            return form(row_values, row_accepted, ranks, keep)

        program = frx.jit(frx.vmap(compact))
        return lambda: program(values, accepted)

    return build


@contextlib.contextmanager
def _generating_inputs() -> Iterator[None]:
    """Build the bench's inputs on the CPU device, whatever the leg measures.

    Key generation and signing are the *concrete* path — a host `while` around
    eager ops — so on the GPU leg each of those ops is its own dispatch and the
    NTTs among them are their own first-time compiles. That is minutes of setup
    for data that is only ever an argument to the thing being timed, and it would
    be the same data either way. Pinning it to CPU also means both legs measure
    the identical batch, which is what makes their rows comparable at all.
    """
    hosts = frx.devices("cpu")
    if not hosts:
        yield
        return
    with frx.default_device(hosts[0]):
        yield


@lru_cache(maxsize=None)
def _ml_dsa_block(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distinct signed messages to tile a batch out of, generated once per set.

    Cached because paying key generation again per batch size would put minutes
    of setup into a bench that measures neither it nor signing.
    """
    scheme = ml_dsa.named(name, deterministic=True)
    keys, messages, signatures = [], [], []
    with _generating_inputs():
        for instance in range(_INSTANCES.value):
            public, secret = scheme.keygen(_seed(instance, 32))
            message = _seed(instance + 101, 32)
            keys.append(np.asarray(public))
            messages.append(message)
            signatures.append(np.asarray(scheme.sign(secret, message)))
    return np.stack(keys), np.stack(messages), np.stack(signatures)


def _every_entry_verified(case: str) -> _Validate:
    """Refuse a verification row whose batch did not actually verify.

    Both schemes return `false` for the whole batch — without evaluating anything
    — when a key or a signature is the wrong length, because §3.6.2 makes a
    malformed input a verdict rather than an error. A batch assembled with one
    part mis-shaped therefore still times, and times a `zeros`: fast, stable, and
    measuring nothing. In the table that is indistinguishable from a real
    speedup, so it is refused where it happens rather than read there.
    """

    def validate(verdicts: Any) -> None:
        decided = np.asarray(verdicts)
        if not decided.all():
            raise RuntimeError(
                f"{case}: {int((~decided).sum())} of {decided.size} entries did "
                f"not verify, so this row would time a rejected batch rather "
                f"than the work a caller waits for"
            )

    return validate


def _ml_dsa_verify_program(name: str, batch: int) -> _Build:
    """`verify` over a batch of real signatures, which is what a caller waits for."""
    scheme = ml_dsa.named(name, deterministic=True)
    block = _ml_dsa_block(name)

    def build() -> Callable[[], Any]:
        arguments = tuple(_tiled(part, batch) for part in block)
        program = frx.jit(scheme.verify)
        return lambda: program(*arguments)

    return build


@lru_cache(maxsize=None)
def _falcon_block(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The published vectors to tile a batch out of, parsed once per set.

    The round-3 vectors rather than freshly signed ones: this scheme cannot sign
    here yet, and the vectors are what `verify` is gated on anyway. Filtered to
    one message length first — a batch carries one static length, and the KAT
    messages do not all share one. A vector missing any of the three parts is
    dropped rather than defaulted: the harness type makes them optional so that a
    keygen-only case can exist, and a verification bench has nothing to do with
    one.
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


def _falcon_batch(name: str, batch: int) -> tuple[Any, Any, Any]:
    """A batch of published Falcon signatures, tiled to `batch`."""
    keys, messages, signatures = (_tiled(part, batch) for part in _falcon_block(name))
    return keys, messages, signatures


def _falcon_verify_program(name: str, batch: int) -> _Build:
    """`verify` over a batch of published signatures."""
    scheme = falcon.named(name)

    def build() -> Callable[[], Any]:
        arguments = _falcon_batch(name, batch)
        program = frx.jit(scheme.verify)
        return lambda: program(*arguments)

    return build


# -- reporting -------------------------------------------------------------


class _Table:
    """A table that prints each row as it is measured, rather than at the end.

    Streaming rather than collected because a GPU leg spends most of its time in
    compiles, and a bench that prints nothing until the last row is
    indistinguishable from a hung one — which is the state that gets a long run
    killed halfway through.
    """

    def __init__(self, title: str) -> None:
        self._header = (
            f"{'case':<36}{'gather':>12}{'scatter':>12}{'ratio':>9}"
            f"{'cold g':>10}{'cold s':>10}"
        )
        print()
        print(title, flush=True)
        print(self._header)
        print("-" * len(self._header), flush=True)

    def row(self, case: str, warm: dict[str, float], cold: dict[str, float]) -> None:
        ratio = warm["gather"] / warm["scatter"]
        print(
            f"{case:<36}{warm['gather'] * 1e3:>10.3f}ms"
            f"{warm['scatter'] * 1e3:>10.3f}ms{ratio:>8.2f}x"
            f"{cold['gather']:>9.2f}s{cold['scatter']:>9.2f}s",
            flush=True,
        )

    def close(self) -> None:
        print("  ratio > 1 means the scatter is faster.", flush=True)


def _measure(
    build: _Build, validate: _Validate | None = None
) -> tuple[dict[str, float], dict[str, float]]:
    """Both forms of one program: warm medians, interleaved, and cold costs."""
    prepared = {
        label: _prepare(build, form, validate) for label, form in _FORMS.items()
    }
    return _interleaved(prepared), {
        label: entry.cold for label, entry in prepared.items()
    }


def _measure_stage(
    site: _Site, batch: int
) -> tuple[dict[str, float], dict[str, float]]:
    """The stage's two forms. Each form gets its own program, built around it."""
    prepared = {
        label: _prepare(_stage_program(site, batch, form), form)
        for label, form in _FORMS.items()
    }
    return _interleaved(prepared), {
        label: entry.cold for label, entry in prepared.items()
    }


def _falcon_stages(name: str, batch: int) -> None:
    """`verify` and the two stages it divides into, in this run, for both forms.

    Printed together because a share taken in one session and a total taken in
    another is not a share — the same stage re-measured on the same commit has
    drifted 25% while its total held to 4%. The three are interleaved with each
    other for the same reason the A/B is. Each stage is timed as the program it
    would be on its own, which is what makes the parts comparable to the whole
    without being a decomposition of it.
    """
    params = falcon.PARAMETER_SETS[name]
    scheme = falcon.named(name)

    def whole() -> Callable[[], Any]:
        arguments = _falcon_batch(name, batch)
        program = frx.jit(scheme.verify)
        return lambda: program(*arguments)

    def challenge() -> Callable[[], Any]:
        _, messages, signatures = _falcon_batch(name, batch)
        salted = fnp.concatenate([signatures[:, 1:41], messages], axis=-1)
        program = frx.jit(frx.vmap(lambda body: falcon.hash_to_point(body, params.n)))
        return lambda: program(salted)

    def decoder() -> Callable[[], Any]:
        _, _, signatures = _falcon_batch(name, batch)
        program = frx.jit(
            frx.vmap(
                lambda one: falcon_encoding.sig_decode(
                    one, params.n, params.signature_size
                )
            )
        )
        return lambda: program(signatures)

    print()
    print(f"Falcon {name} stages at B = {batch}, one run")
    header = (
        f"{'form':<10}{'verify':>12}{'HashToPoint':>14}{'sig_decode':>13}{'shares':>18}"
    )
    print(header)
    print("-" * len(header), flush=True)
    builds = {"verify": whole, "HashToPoint": challenge, "sig_decode": decoder}
    checks: dict[str, _Validate | None] = {
        "verify": _every_entry_verified(f"{name} stages B={batch}"),
        "HashToPoint": None,
        "sig_decode": None,
    }
    for label, form in _FORMS.items():
        prepared = {
            stage: _prepare(build, form, checks[stage])
            for stage, build in builds.items()
        }
        timed = _interleaved(prepared)
        shares = (
            f"{timed['HashToPoint'] / timed['verify']:.0%} /"
            f" {timed['sig_decode'] / timed['verify']:.0%}"
        )
        print(
            f"{label:<10}{timed['verify'] * 1e3:>10.3f}ms"
            f"{timed['HashToPoint'] * 1e3:>12.3f}ms"
            f"{timed['sig_decode'] * 1e3:>11.3f}ms{shares:>18}",
            flush=True,
        )
    print("  shares are HashToPoint / sig_decode against `verify`.", flush=True)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError(f"unexpected arguments: {argv[1:]}")
    batches = [int(batch) for batch in _BATCHES.value]
    print(f"backend: {frx.default_backend()}   devices: {frx.devices()}", flush=True)
    print(
        f"rounds: {_ROUNDS.value}, samples: {_SAMPLES.value}, "
        f"instances: {_INSTANCES.value}",
        flush=True,
    )

    sites: list[_Site] = []
    for name in _ML_DSA_SETS.value:
        sites += _ml_dsa_sites(name)
    for name in _FALCON_SETS.value:
        sites += _falcon_sites(name)
    print(
        f"captured {len(sites)} sites: {[site.sampler for site in sites]}", flush=True
    )

    stage = _Table("The compaction alone")
    for site in sites:
        shape = f"[{site.values.shape[1]}, {site.values.shape[2]}]->{site.keep}"
        for batch in batches:
            warm, cold = _measure_stage(site, batch)
            stage.row(f"{site.sampler} {shape} B={batch}", warm, cold)
    stage.close()

    operation = _Table("The whole operation")
    for name in _ML_DSA_SETS.value:
        for batch in batches:
            warm, cold = _measure(
                _ml_dsa_verify_program(name, batch),
                _every_entry_verified(f"{name} verify B={batch}"),
            )
            operation.row(f"{name} verify B={batch}", warm, cold)
    for name in _FALCON_SETS.value:
        for batch in batches:
            warm, cold = _measure(
                _falcon_verify_program(name, batch),
                _every_entry_verified(f"{name} verify B={batch}"),
            )
            operation.row(f"{name} verify B={batch}", warm, cold)
    operation.close()

    if _STAGES.value:
        for name in _FALCON_SETS.value:
            _falcon_stages(name, batches[-1])


if __name__ == "__main__":
    app.run(main)
