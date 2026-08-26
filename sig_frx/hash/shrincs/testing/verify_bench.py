# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Where SHRINCS verification spends its time, warm and cold.

Run it before optimizing anything here, and again afterwards against the same
harness. The sibling `//sig_frx/hash/slhdsa/testing:verify_bench` is the model;
what SHRINCS adds is a second path and a walk whose step count is fixed by the
format rather than by the key, which is where both of the costs below come from.

Four things it measures:

- **Per-signature latency against batch size.** The dispatch count is a function
  of the format alone, so a batch amortizes every fixed cost in the path. A flat
  per-call time across the batch axis is the signature of a dispatch-bound path,
  and it is what decides whether an optimization here is worth anything at the
  sizes a deployment would use.
- **The FXMSS walk on its own, split into host and hash.** `root_from_sig` takes
  its tweakable hash as an argument, so injecting a counting one measures the
  hashes exactly and attributes the remainder to host work — building addresses
  and the array plumbing around them. `Shrincs.verify` cannot be measured this
  way: `Stateless` builds its family internally, so there is no seam to inject
  through, and the end-to-end figures below are wall clock only.
- **What the walk's host time is made of.** The walk runs `FXMSS_HEIGHT = 255`
  steps for every entry and masks off the ones past each one's own depth. Each
  step encodes an address batch and shifts the running index, and this replays
  those two in isolation so the split between them is measured rather than
  inferred.
- **Cold: compile seconds and executables.** `frx.jit(verify)` traces both paths
  for every entry plus 255 more levels of the walk. Nothing in the suite jits it
  today, so the first consumer to do so pays a compile nothing has measured —
  and an optimization that trades warm dispatch for cold compile needs both
  numbers to be judged. Executables are counted off `frx.log_compiles`, which
  emits one line per XLA compilation.

The host / hash split is an eager-only measurement. Under a tracer the injected
hash is called once, to build the program rather than to run it, so what the
wrapper times is tracing — the split is meaningless there and is not printed.

    bazel run //sig_frx/hash/shrincs/testing:verify_bench -- --batches=1,4,64
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Callable, Sequence

import frx
import frx.numpy as fnp
import numpy as np
from absl import app, flags
from frx import Array
from frx.typing import ArrayLike
from hash_frx import Sha256

from sig_frx.hash import bytestring
from sig_frx.hash.shrincs import adrs as sf_adrs
from sig_frx.hash.shrincs import fxmss, shrincs
from sig_frx.hash.shrincs.testing import harness
from sig_frx.hash.shrincs.testing import stateful_vectors as vectors
from sig_frx.hash.tweakable import Sha2TweakableHash

_BATCHES = flags.DEFINE_list(
    "batches", ["1", "4", "8", "16", "64"], "Batch sizes to measure."
)
_REPS = flags.DEFINE_integer(
    "reps", 3, "Timed repetitions per batch size; the fastest is reported."
)
_COLD = flags.DEFINE_bool(
    "cold", True, "Measure the traced path. Off skips a compile of several seconds."
)

# The parameters `stateless.PARAMS` fixes, read here rather than imported so the
# injected hash stands in at exactly the shape the real one is built at.
_N = 16
_M = 24


class _MeasuredHash:
    """A `ByteHash` that counts its dispatches and times them.

    Dependency injection is what makes the host / hash split exact rather than
    inferred: the walk reaches every hash through this seam, so nothing it does
    is missed and nothing else is attributed to it.
    """

    def __init__(self) -> None:
        self._inner = Sha256()
        self.digest_size = self._inner.digest_size
        self.fusion_path = self._inner.fusion_path
        self.reset()

    def reset(self) -> None:
        self.seconds = 0.0
        self.calls = 0
        self.rows = 0

    def digest(self, msg: ArrayLike) -> Array | np.ndarray:
        start = time.perf_counter()
        out = _blocked(self._inner.digest(msg))
        self.seconds += time.perf_counter() - start
        self.calls += 1
        self.rows += int(np.shape(msg)[0])
        return out

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _MeasuredHash):
            return NotImplemented
        return self._inner == other._inner

    def __hash__(self) -> int:
        return hash((type(self), self._inner))


def _blocked(value: Array | np.ndarray) -> Array | np.ndarray:
    """Wait for a dispatch to land, so a timing measures work and not queueing."""
    ready = getattr(value, "block_until_ready", None)
    if ready is not None:
        ready()
    return value


def _timed(call: Callable[[], object]) -> float:
    start = time.perf_counter()
    _blocked(call())
    return time.perf_counter() - start


def _fastest(call: Callable[[], object], reps: int) -> float:
    """The fastest of `reps` timed calls, in seconds. Assumes a warm caller."""
    return min(_timed(call) for _ in range(reps))


def _case() -> vectors.StatefulVectors:
    """The deepest reference case — the walk's cost is flat in the depth, but its
    authentication path is not, so the widest one is the honest fixture."""
    return max(vectors.REFERENCE, key=lambda c: c.depth)


def _batched(value: bytes, batch: int, width: int | None = None) -> np.ndarray:
    row = np.frombuffer(value, dtype=np.uint8)
    if width is not None:
        row = np.concatenate([row, np.zeros(width - row.shape[0], dtype=np.uint8)])
    return np.stack([row] * batch)


def _walk_inputs(
    case: vectors.StatefulVectors, batch: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """What `root_from_sig` takes, at `batch` copies of one reference signature.

    The FXMSS signature is the tail of the recorded one: the indicator byte, the
    randomizer and the leaf index come off the front, which is the parse
    `Shrincs.verify` performs before it reaches the walk.
    """
    body = harness.fxmss_body(case)
    pk_seed = np.frombuffer(case.public_key[:_N], dtype=np.uint8)
    signatures = _batched(body, batch, width=fxmss.SIGNATURE_SIZE_MAX)
    digests = _batched(case.message_digest, batch)
    heights = np.full(batch, case.leaf_height, dtype=np.uint32)
    indices = np.stack(
        [np.frombuffer(case.leaf_index.to_bytes(fxmss.INDEX_BYTES), dtype=np.uint8)]
        * batch
    )
    return pk_seed, signatures, digests, heights, indices


def _latency(batches: Sequence[int], reps: int) -> None:
    """End-to-end `Shrincs.verify`, warm and eager. Wall clock only — see above."""
    scheme = shrincs.Shrincs()
    case = _case()
    print("\n== Shrincs.verify, warm, eager ==")
    print(f"{'B':>5} {'total ms':>10} {'per sig ms':>12}")
    for batch in batches:
        public_keys = _batched(case.public_key, batch)
        messages = _batched(case.message, batch)
        signatures = _batched(case.signature, batch, width=scheme.signature_max_size)
        contexts = _batched(case.context, batch)
        call = lambda: scheme.verify(  # noqa: E731
            public_keys, messages, signatures, context=contexts
        )
        _blocked(call())  # warm
        seconds = _fastest(call, reps)
        print(f"{batch:>5} {seconds * 1e3:>10.1f} {seconds / batch * 1e3:>12.2f}")


def _walk(batches: Sequence[int], reps: int) -> None:
    """`fxmss.root_from_sig` alone, with the hashes measured out of it."""
    measured = _MeasuredHash()
    tweak = Sha2TweakableHash(measured, n=_N, m=_M, block_size=64)
    case = _case()
    print("\n== the FXMSS walk (root_from_sig), warm, eager ==")
    print(
        f"{'B':>5} {'total ms':>10} {'hash ms':>9} {'host ms':>9} "
        f"{'host %':>7} {'hashes':>7}"
    )
    for batch in batches:
        pk_seed, signatures, digests, heights, indices = _walk_inputs(case, batch)
        call = lambda: fxmss.root_from_sig(  # noqa: E731
            tweak, pk_seed, signatures, digests, heights, indices
        )[0]
        _blocked(call())  # warm
        best = None
        for _ in range(reps):
            measured.reset()
            seconds = _timed(call)
            if best is None or seconds < best[0]:
                best = (seconds, measured.seconds, measured.calls)
        assert best is not None
        seconds, hashed, calls = best
        host = seconds - hashed
        print(
            f"{batch:>5} {seconds * 1e3:>10.1f} {hashed * 1e3:>9.1f} "
            f"{host * 1e3:>9.1f} {host / seconds * 100:>6.0f}% {calls:>7}"
        )


def _addresses(batches: Sequence[int], reps: int) -> None:
    """The three things every step of the walk does, replayed on their own.

    **On device arrays, which is the whole point.** `root_from_sig` reaches the
    loop through `fnp.asarray`, so `parents` and `heights` are traced values and
    every step's encode and shift dispatch. Replaying the same calls on numpy
    arrays measures `adrs_encoding`'s host path instead — a different function
    reached through the same name, and roughly thirty times cheaper. The split
    below is only meaningful because these are the namespaces the walk uses.

    The selects are here too, so the three columns can be checked against the
    walk's own host time rather than assumed to account for it.
    """
    case = _case()
    print("\n== what a step costs, over 255 steps, on device arrays ==")
    print(
        f"{'B':>5} {'encode ms':>11} {'shift ms':>10} {'select ms':>11} "
        f"{'sum ms':>9}"
    )
    for batch in batches:
        _, _, _, host_heights, host_indices = _walk_inputs(case, batch)
        heights = fnp.asarray(host_heights, dtype=fnp.uint32)
        indices = fnp.asarray(host_indices, dtype=fnp.uint8)
        nodes = fnp.zeros((batch, _N), dtype=fnp.uint8)

        def encodes() -> object:
            out = None
            for step in range(fxmss.HEIGHT):
                parent = fnp.minimum(heights + (step + 1), np.uint32(fxmss.HEIGHT))
                out = sf_adrs.encode_batch(sf_adrs.fxmss_tree(parent, indices))
            return out

        def shifts() -> object:
            running = indices
            for _ in range(fxmss.HEIGHT):
                running = bytestring.shift_right(running, 1)
            return running

        def selects() -> object:
            out = nodes
            for _ in range(fxmss.HEIGHT):
                side = (indices[:, -1] & np.uint8(1))[:, None]
                out = fnp.where(side, nodes, out)
            return out

        encode_s = _fastest(encodes, reps)
        shift_s = _fastest(shifts, reps)
        select_s = _fastest(selects, reps)
        total = encode_s + shift_s + select_s
        print(
            f"{batch:>5} {encode_s * 1e3:>11.1f} {shift_s * 1e3:>10.1f} "
            f"{select_s * 1e3:>11.1f} {total * 1e3:>9.1f}"
        )


def _compiles(call: Callable[[], object]) -> tuple[float, int]:
    """Seconds and XLA executables for one cold call.

    `log_compiles` emits a line per compilation, which is the executable count
    the acceptance criteria for a hoist here are stated against. Counting the
    lines is the only handle frx exposes; there is no counter to read.
    """
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    logger = logging.getLogger("frx._src.dispatch")
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        with frx.log_compiles():
            seconds = _timed(call)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
    executables = sum(
        1
        for line in buffer.getvalue().splitlines()
        if "Finished XLA compilation" in line
    )
    return seconds, executables


def _traced(batches: Sequence[int], reps: int) -> None:
    """The same verification under `frx.jit`, with the compile separated.

    A fresh `jit` per batch size, so the compile reported is that shape's own: a
    compile is paid once per shape and amortizes over every call at it, which is
    why it is printed beside the warm latency rather than folded into it.
    """
    case = _case()

    # The eager path's own cold cost, for the comparison the traced numbers are
    # only half of. Eager dispatch compiles a small executable per distinct op
    # shape and caches it per process, so this is measured once, first, before
    # anything below has warmed those caches.
    scheme = shrincs.Shrincs()
    batch = batches[0]
    frx.clear_caches()
    eager_cold, eager_execs = _compiles(
        lambda: scheme.verify(
            _batched(case.public_key, batch),
            _batched(case.message, batch),
            _batched(case.signature, batch, width=scheme.signature_max_size),
            context=_batched(case.context, batch),
        )
    )
    print(
        f"\n== cold, eager, B={batch}: {eager_execs} executables in "
        f"{eager_cold:.1f} s (first call, caches cleared) =="
    )

    print("\n== Shrincs.verify under frx.jit ==")
    print(f"{'B':>5} {'compile s':>10} {'execs':>7} {'warm ms':>9} {'per sig ms':>11}")
    for batch in batches:
        scheme = shrincs.Shrincs()
        public_keys = _batched(case.public_key, batch)
        messages = _batched(case.message, batch)
        signatures = _batched(case.signature, batch, width=scheme.signature_max_size)
        contexts = _batched(case.context, batch)
        # The context is closed over rather than passed as an argument: it
        # reaches `context.prefix`, which measures it on the host with
        # `np.asarray`, so a traced one raises `TracerArrayConversionError`
        # before any of this compiles. Closing over it makes it a compile-time
        # constant, which is what that host read requires — and it means a
        # jitted verifier is specialised per context, not only per shape.
        verify = frx.jit(
            lambda keys, msgs, sigs: scheme.verify(keys, msgs, sigs, context=contexts)
        )
        call = lambda: verify(public_keys, messages, signatures)  # noqa: E731
        cold, executables = _compiles(call)
        warm = _fastest(call, reps)
        print(
            f"{batch:>5} {cold - warm:>10.1f} {executables:>7} {warm * 1e3:>9.1f} "
            f"{warm / batch * 1e3:>11.2f}"
        )


def main(argv: Sequence[str]) -> None:
    del argv
    batches = [int(value) for value in _BATCHES.value]
    reps = _REPS.value
    case = _case()
    print(
        f"fixture: {case.label} — shape {case.shape}, depth {case.depth}, "
        f"leaf (index {case.leaf_index}, height {case.leaf_height})"
    )
    _latency(batches, reps)
    _walk(batches, reps)
    _addresses(batches, reps)
    if _COLD.value:
        _traced(batches, reps)


if __name__ == "__main__":
    app.run(main)
