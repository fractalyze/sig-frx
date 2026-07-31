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

There is no compile time to separate out. Nothing in the path is traced: the hash
takes concrete arrays and the tree and leaf indices are host integers that address
nodes, so `verify` is a sequence of eager dispatches. A jit-ed comparison is what
this becomes when that changes.

    bazel run //sig_frx/hashbased/testing:verify_bench -- --batches=1,8,64
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
from absl import app, flags
from frx import Array
from frx.typing import ArrayLike
from hash_frx.sha256 import Sha256

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
        self.has_dedicated_fusion = self._inner.has_dedicated_fusion
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
        Sha2TweakableHash(measured, n=params.n, m=params.m),
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


def _latency(name: str, batches: Sequence[int], reps: int) -> None:
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
    first = None
    last = None
    for batch in batches:
        keys = np.stack([public_key] * batch)
        messages = np.stack([message] * batch)
        signatures = np.stack([signature] * batch)
        _blocked(scheme.verify(keys, messages, signatures))  # warm the caches
        best = None
        for _ in range(reps):
            measured.reset()
            start = time.perf_counter()
            _blocked(scheme.verify(keys, messages, signatures))
            elapsed = time.perf_counter() - start
            if best is None or elapsed < best:
                best, hashing, calls, rows = (
                    elapsed,
                    measured.seconds,
                    measured.calls,
                    measured.rows,
                )
        assert best is not None
        per_signature = best / batch * 1e3
        first = per_signature if first is None else first
        last = per_signature
        print(
            f"{batch:>5} {best * 1e3:>10.1f} {per_signature:>11.2f} "
            f"{100 * hashing / best:>7.1f} {100 * (best - hashing) / best:>7.1f} "
            f"{calls:>11} {rows // batch:>10}"
        )
    assert first is not None and last is not None
    print(
        f"  per-signature: {first / last:.1f}x faster at B={batches[-1]} than at "
        f"B={batches[0]}, on a dispatch count that never changed"
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
        _latency(name, batches, _REPS.value)
        _stages(name, batches[-1])


if __name__ == "__main__":
    app.run(main)
