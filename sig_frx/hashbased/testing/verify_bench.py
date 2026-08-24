# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Where SLH-DSA verification spends its time, and what the batch axis buys.

Run it before optimizing anything here. Verification is a fixed sequence of hashes
with no secret-dependent control flow, which reads like a workload that should be
hash-bound — and it is not, by a wide margin. The numbers this prints are what
decides which of the several available optimizations is worth doing.

Three things it measures:

- **Per-signature latency against batch size.** The dispatch count is a function of
  the parameter set alone, not of `B`, so a batch amortizes every fixed cost in the
  path. That is the whole argument for the seam being batch-first, and it is
  measurable rather than assumed.
- **The host / hash split.** The `ByteHash` seam is dependency-injected, so wrapping
  it counts every dispatch and times every hash exactly. Whatever is left is host
  work: building addresses, and the array plumbing around them.
- **Per stage.** `H_msg` and the digest split, FORS reconstruction, and the
  hypertree walk, in the order `verify_internal` runs them.
- **Eager against traced.** The whole path traces, so `frx.jit(verify)` is one
  program where the eager form is a few hundred thousand dispatches. What that
  buys is a wall-clock question with a compile time attached, and both are here:
  a compile is paid once per `(parameter set, batch size)` shape and amortizes
  over every call at that shape, so it is reported beside the warm latency rather
  than folded into it.

That compile is charged per process until a caller sets
`FRX_COMPILATION_CACHE_DIR`, which makes it reusable across them. Running this
twice under one cache directory is how to measure what that saves.

The host / hash split is an eager-only measurement. Under a tracer the injected
hash is called once, to build the program rather than to run it, so what the
wrapper times is tracing — the split is meaningless there and is not printed.

    bazel run //sig_frx/hashbased/testing:verify_bench -- --batches=1,8,64
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import frx
import numpy as np
from absl import app, flags
from frx import Array
from frx.typing import ArrayLike
from hash_frx import Sha256

from sig_frx.hashbased import fors, hypertree, slh_dsa
from sig_frx.hashbased.tweakable import Sha2TweakableHash

_PARAMETER_SETS = flags.DEFINE_list(
    "parameter_sets",
    ["SLH-DSA-SHA2-128f", "SLH-DSA-SHA2-128s"],
    "Parameter sets to measure. Both category 1 sets by default, since `s` and `f`"
    " sit at opposite ends of the trade and verification does not rank them the way"
    " signing does.",
)
_BATCHES = flags.DEFINE_list(
    "batches", ["1", "2", "4", "8", "16", "32", "64"], "Batch sizes to measure."
)
_REPS = flags.DEFINE_integer(
    "reps", 3, "Timed repetitions per batch size; the fastest is reported."
)


class _MeasuredHash:
    """A `ByteHash` that counts its dispatches and times them.

    Dependency injection is what makes the host / hash split exact rather than
    inferred: the scheme reaches every hash through this seam, so nothing it does
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


def _fixture(
    name: str,
) -> tuple[slh_dsa.SlhDsa, _MeasuredHash, np.ndarray, np.ndarray, np.ndarray]:
    params = slh_dsa.SHA2_PARAMETER_SETS[name]
    measured = _MeasuredHash()
    scheme = slh_dsa.SlhDsa(
        # `block_size` is explicit because hash-frx's table is keyed on the row's
        # type name, and this stands in for `Sha256` under its own.
        Sha2TweakableHash(measured, n=params.n, m=params.m, block_size=64),
        params,
        deterministic=True,
    )
    seed = np.frombuffer(
        bytes((i * 13 + 5) % 256 for i in range(params.seed_size)), dtype=np.uint8
    )
    public_key, secret_key = (np.asarray(part) for part in scheme.keygen(seed))
    message = np.frombuffer(b"a message of some length", dtype=np.uint8)
    signature = np.asarray(scheme.sign(secret_key, message))
    return scheme, measured, public_key, message, signature


def _batch_of(
    public_key: np.ndarray, message: np.ndarray, signature: np.ndarray, batch: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.stack([public_key] * batch),
        np.stack([message] * batch),
        np.stack([signature] * batch),
    )


def _fastest(call: Callable[[], object], reps: int) -> float:
    """The fastest of `reps` timed calls, in seconds. Assumes a warm caller."""
    return min(_timed(call) for _ in range(reps))


def _timed(call: Callable[[], object]) -> float:
    start = time.perf_counter()
    _blocked(call())
    return time.perf_counter() - start


def _latency(
    name: str, batches: Sequence[int], reps: int
) -> tuple[slh_dsa.SlhDsa, np.ndarray, np.ndarray, np.ndarray, dict[int, float]]:
    """The eager table. Returns the fixture and the per-signature seconds."""
    scheme, measured, public_key, message, signature = _fixture(name)
    params = scheme.params
    print(
        f"\n=== {name}: d={params.d} h'={params.tree_height} k={params.k} "
        f"a={params.a} len={params.wots_params.len}"
    )
    print(
        f"{'B':>5} {'total ms':>10} {'per sig ms':>11} {'hash %':>7} "
        f"{'host %':>7} {'dispatches':>11} {'rows/sig':>10}"
    )
    eager: dict[int, float] = {}
    for batch in batches:
        arrays = _batch_of(public_key, message, signature, batch)
        _blocked(scheme.verify(*arrays))  # warm the caches
        best = None
        for _ in range(reps):
            measured.reset()
            elapsed = _timed(lambda: scheme.verify(*arrays))
            if best is None or elapsed < best:
                best, hashing, calls, rows = (
                    elapsed,
                    measured.seconds,
                    measured.calls,
                    measured.rows,
                )
        assert best is not None
        eager[batch] = best / batch
        print(
            f"{batch:>5} {best * 1e3:>10.1f} {eager[batch] * 1e3:>11.2f} "
            f"{100 * hashing / best:>7.1f} {100 * (best - hashing) / best:>7.1f} "
            f"{calls:>11} {rows // batch:>10}"
        )
    first, last = eager[batches[0]], eager[batches[-1]]
    print(
        f"  per-signature: {first / last:.1f}x faster at B={batches[-1]} than at "
        f"B={batches[0]}, on a dispatch count that never changed"
    )
    return scheme, public_key, message, signature, eager


def _traced(
    scheme: slh_dsa.SlhDsa,
    public_key: np.ndarray,
    message: np.ndarray,
    signature: np.ndarray,
    batches: Sequence[int],
    reps: int,
    eager: dict[int, float],
) -> None:
    """The same latencies under `frx.jit`, with the compile they cost separated.

    A fresh `jit` per batch size, so the compile reported is that shape's own: the
    batch axis is part of the traced shape, and a cache hit from a previous size
    would report zero for work that was really done.

    The compile is the cold call minus the warm one, which is the same reading the
    sibling repos take — a cold call is a compile plus one execution, and the
    execution is what the warm calls measure.
    """
    print(
        f"{'B':>5} {'compile s':>10} {'total ms':>10} {'per sig ms':>11} "
        f"{'vs eager':>9}"
    )
    for batch in batches:
        arrays = _batch_of(public_key, message, signature, batch)
        verify = frx.jit(scheme.verify)
        cold = _timed(lambda: verify(*arrays))
        best = _fastest(lambda: verify(*arrays), reps)
        per_signature = best / batch
        print(
            f"{batch:>5} {cold - best:>10.1f} {best * 1e3:>10.1f} "
            f"{per_signature * 1e3:>11.2f} {eager[batch] / per_signature:>8.2f}x"
        )


def _stages(name: str, batch: int) -> None:
    """The three stages `verify_internal` runs, in its order."""
    scheme, measured, public_key, message, signature = _fixture(name)
    params = scheme.params
    keys = np.stack([public_key] * batch)
    signatures = np.asarray(np.stack([signature] * batch), dtype=np.uint8)
    # Algorithm 24 line 4 with an empty context, which is what `verify` prepends.
    messages = np.stack(
        [np.concatenate([np.zeros(2, dtype=np.uint8), message])] * batch
    )
    pk_seeds, pk_roots = keys[:, : params.n], keys[:, params.n :]
    randomizers, fors_signatures, hypertree_signatures = scheme._parse_signature(
        signatures
    )

    timings: dict[str, tuple[float, int, int]] = {}

    def record(label: str, start: float) -> None:
        timings[label] = (time.perf_counter() - start, measured.calls, measured.rows)

    measured.reset()
    start = time.perf_counter()
    md, tree_indices, leaf_indices = scheme._split_digest(
        randomizers, pk_seeds, pk_roots, messages
    )
    _blocked(md)
    record("H_msg and the split", start)

    measured.reset()
    start = time.perf_counter()
    fors_keys = fors.pk_from_sig(
        scheme._tweak,
        params.fors_params,
        fors_signatures,
        md,
        pk_seeds,
        fors.ForsPosition(tree=tree_indices, key_pair=leaf_indices),
    )
    _blocked(fors_keys)
    record("FORS reconstruction", start)

    measured.reset()
    start = time.perf_counter()
    _blocked(
        hypertree.verify(
            scheme._tweak,
            params.hypertree_params,
            hypertree_signatures,
            fors_keys,
            pk_seeds,
            tree_indices,
            leaf_indices,
            pk_roots,
        )
    )
    record("hypertree walk", start)

    total = sum(seconds for seconds, _, _ in timings.values())
    print(f"  stages at B={batch}:")
    for label, (seconds, calls, rows) in timings.items():
        print(
            f"    {label:<22} {seconds * 1e3:>8.1f} ms {100 * seconds / total:>6.1f}%"
            f" {calls:>5} dispatches {rows:>9} rows"
        )


def main(argv: Sequence[str]) -> None:
    del argv
    batches = [int(value) for value in _BATCHES.value]
    for name in _PARAMETER_SETS.value:
        scheme, public_key, message, signature, eager = _latency(
            name, batches, _REPS.value
        )
        _stages(name, batches[-1])
        _traced(scheme, public_key, message, signature, batches, _REPS.value, eager)


if __name__ == "__main__":
    app.run(main)
