# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Where classical verification spends its time, and what a batch buys.

Every classical verifier here is host-path by construction — Python codecs
around the curated dtype kernels (fractalyze/sig-frx#139) — so the routing
question this answers is not "does it trace" but which side of that seam
dominates: the batched C++ kernels doing the curve arithmetic, or the per-row
Python codec around them. The substrate decision on fractalyze/sig-frx#36
turns on exactly that split, so this prints it rather than leaving it assumed.

Per lane it reports per-signature latency against batch size, with the time
split across the substrate seams the lane actually crosses:

- `mult` — `secp.multiple`, the batched scalar-multiplication ufunc: the cost
  GPU EC kernels or a fused MSM kernel would attack.
- `readback` — `secp.affine_ints` and `secp.is_identity`: the jacobian-to-
  affine conversion and the per-row `.raw` reads behind every coordinate
  compare.
- `lift` — `secp.lift_x_to_parity`: the x-plus-parity square-root lift the
  x-only and compressed encodings pay on the way in.
- `sum` — the aggregate checks' halving-tree point sum, adds an MSM kernel
  would also absorb. One bucket for both aggregate lanes: they fold through
  the same `group.sum_points`.
- `decode` / `sha512` — the Edwards substrate's own seams: point
  decompression (its square root and per-row construction included) and the
  per-row host SHA-512 (no device row exists — fractalyze/hash-frx#66). Its
  scalar multiplications ride the shared `mult` bucket via
  `edwards.multiple`.
- `hash` — ECDSA's message digest through the `MessageHash` seam.

Whatever remains is `host`: wire parsing, Python-integer scalar arithmetic,
challenge hashing, and verdict assembly. (Ed25519's remainder also carries
the final extended-coordinates add and compare.)

There is no compile column, and one seam now needs that said rather than
assumed: the square-root ladder is `jit`-ed (see `secp.sqrt`), so a placed
`lift` pays a compile the first time a process meets a batch shape. It does
not appear below — every batch size here is warmed before it is timed, which
puts that cost outside every number in the table. The point operations are
not `jit`-ed: each is a single fused op, for which compiling measured no
better than running it eagerly. There is no traced *section* either, but that
is no longer the same as saying nothing traces — a batch at or above
`secp.DEVICE_MIN_BATCH` is placed on the device by the schemes, so both the
`mult` and the `lift` buckets of every secp lane are a device cost at the
larger batch sizes and a host cost below it. The split is therefore visible
in the numbers rather than in the columns: watch each share collapse as `B`
crosses the threshold.

## Two totals, because a placed seam does not finish when it returns

`wall` is the lane's latency: run it, stop the clock, and let the device queue
overlap whatever it can. `work` is the same lane with every metered seam
waiting for what it queued, so each bucket is charged the arithmetic it
started. The bucket shares are against `work`.

Both columns are needed because a single one of them lies. Without the wait,
a placed seam is timed for its *enqueue* and its arithmetic is billed to
whichever later seam first reads the array — on every secp lane that is
`affine_ints`, inside `readback`. Measured at B=1024, `secp.double_multiple`
returns in about a fifth of the time the multiplication it queued then takes:
timed without the wait, that reads as a 2% `mult` and a 45% `readback`, and
the ranking it produces is the dispatch order rather than the profile. With
the wait, `mult` is a third of these lanes and `readback` is the smallest
bucket in all three. Below `DEVICE_MIN_BATCH` nothing is placed and the two
columns agree, which is the check that the wait is not itself the cost.

`work` is the larger of the two by 10-20% at B=1024, and that gap is real
rather than overhead: it is the overlap the queue buys back, which `wall`
keeps and `work` gives up in exchange for honest attribution.

Every batch size is run once before it is timed, so what this reports is
steady state. That makes it blind by construction to any change that moves
cost to the *first* call at a shape — a compile, a lazily built table, a
device probe. Such a change reads here as a pure win, and the larger the
first-call cost the cleaner the win looks. A change that could have one is
measured by timing the first call separately, not by reading this table: the
square-root ladder's `jit` cuts its steady state by ~19x while adding a
compile per batch shape, none of which appears below.

Each scheme's independent and aggregate forms sit in separate lanes so their
per-signature costs read against each other directly — and for both schemes
the aggregate is not yet the cheaper call, because the scalar
multiplications an MSM would collapse are most of both.

    bazel run //sig_frx/classical/testing:verify_bench -- --batches=1,16,256
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import frx
import numpy as np
from absl import app, flags

from sig_frx import hashes
from sig_frx.classical import edwards, group, secp
from sig_frx.classical.ecdsa import core
from sig_frx.classical.eddsa import consensus, ed25519
from sig_frx.classical.schnorr import bip340
from sig_frx.threshold import frost
from sig_frx.threshold.secp256k1_sha256 import Secp256k1Sha256

_LANES = flags.DEFINE_list(
    "lanes",
    ["ecdsa", "eddsa", "eddsa_aggregate", "bip340", "bip340_aggregate", "frost"],
    "Lanes to measure. `ecdsa_p256` swaps ECDSA onto secp256r1 and is off by"
    " default — same code path, different kernel instantiation.",
)
_BATCHES = flags.DEFINE_list(
    "batches", ["1", "4", "16", "64", "256"], "Batch sizes to measure."
)
_REPS = flags.DEFINE_integer(
    "reps", 3, "Timed repetitions per batch size; the fastest is reported."
)

_MESSAGE = np.frombuffer(bytes(range(32)), dtype=np.uint8)


class _Meter:
    """Wall-clock accumulator, one bucket per substrate seam.

    The wrapped seams never call one another (`double_multiple` and
    `schnorr_verdicts` are unwrapped pass-throughs), so the buckets are
    disjoint and `host` is exactly the total minus their sum.
    """

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        # Whether a seam waits for the work it queued before stopping its
        # clock. Off measures the lane's real latency, because the queue is
        # what the overlap is made of; on measures where the work is. The two
        # answer different questions and `_measure` runs a pass of each.
        self.settle = False

    def reset(self) -> None:
        self.seconds.clear()

    def wrap(
        self, bucket: str, fn: Callable[..., Any], *, settles: bool = True
    ) -> Callable[..., Any]:
        """`fn`, with its time accumulated into `bucket`.

        `settles=False` marks a seam that cannot have anything outstanding when
        it returns, so the wait is skipped rather than paid — see `install`.
        """

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            out = fn(*args, **kwargs)
            if self.settle and settles:
                # Inside the window on purpose: a placed seam returns as soon as
                # it is enqueued, so timing the call alone measures the enqueue
                # and bills the arithmetic to whichever later seam first reads
                # the array — the `readback` distortion the `wall` / `work`
                # split above exists to make visible. Host paths are unaffected;
                # a numpy array has no wait to do.
                frx.block_until_ready(out)
            elapsed = time.perf_counter() - start
            self.seconds[bucket] = self.seconds.get(bucket, 0.0) + elapsed
            return out

        return wrapped

    def install(self) -> None:
        """Route the substrate seams through the meter, process-wide.

        The schemes reach every seam by module-global lookup, so swapping the
        module attribute times every call exactly — the same argument the
        hash-based bench makes for its injected hash, without a constructor
        to inject through. `setattr` rather than assignment keeps mypy off
        the cross-module rebind.
        """
        setattr(secp, "multiple", self.wrap("mult", secp.multiple))
        # The readback seams are the synchronization point, not a queue: neither
        # can return before the values have landed on the host, so there is
        # nothing left outstanding to wait for. Skipping the wait is therefore
        # free of meaning and not free of cost — `affine_ints` hands back a
        # `list[tuple[int, int]]`, so settling it would walk 2B host leaves and
        # charge this bucket the walk (0.18 ms at B=256, against a bucket that
        # is 0.23 ms). That is the same self-inflicted `readback` inflation
        # fractalyze/sig-frx#202 was about, arriving through the meter instead
        # of through a missing wait.
        setattr(
            secp,
            "affine_ints",
            self.wrap("readback", secp.affine_ints, settles=False),
        )
        setattr(
            secp,
            "is_identity",
            self.wrap("readback", secp.is_identity, settles=False),
        )
        setattr(secp, "lift_x_to_parity", self.wrap("lift", secp.lift_x_to_parity))
        # Both aggregates fold through `group.sum_points`, so wrapping it
        # once meters the secp and Edwards lanes alike — wrapping each
        # scheme's thin caller instead would leave whichever one was added
        # later silently unmetered, in `host`.
        setattr(group, "sum_points", self.wrap("sum", group.sum_points))
        setattr(edwards, "multiple", self.wrap("mult", edwards.multiple))
        # Multiplication by a constant 8 is still point multiplication, and
        # it does not reach `multiple`, so it needs its own wrap to land in
        # `mult` rather than the remainder.
        setattr(edwards, "mul_by_cofactor", self.wrap("mult", edwards.mul_by_cofactor))
        setattr(edwards, "decode", self.wrap("decode", edwards.decode))
        setattr(ed25519, "_sha512_rows", self.wrap("sha512", ed25519._sha512_rows))

    def message_hash(self) -> core.MessageHash:
        """ECDSA's `MessageHash` with the digest timed into `hash`."""

        def byte_hash(*values: object) -> _TimedDigest:
            return _TimedDigest(self.wrap("hash", hashes.sha256(*values).digest))

        return core.MessageHash(byte_hash=byte_hash, host_constructor=hashlib.sha256)


@dataclass(frozen=True)
class _TimedDigest:
    """The one face of `ByteHash` verification touches, pre-wrapped."""

    digest: Callable[..., Any]


@dataclass(frozen=True)
class _Lane:
    """One verifier to measure: its seam buckets and its fixture."""

    title: str
    buckets: tuple[str, ...]
    build: Callable[[_Meter], tuple[Callable[..., Any], tuple[np.ndarray, ...]]]


def _seed(step: int, offset: int) -> np.ndarray:
    return np.frombuffer(
        bytes((i * step + offset) % 256 for i in range(32)), dtype=np.uint8
    )


def _ecdsa(curve: secp.Curve, meter: _Meter) -> tuple[Any, tuple[np.ndarray, ...]]:
    scheme = core.Ecdsa(curve, meter.message_hash())
    public, secret = scheme.keygen(_seed(13, 5))
    signature = scheme.sign(secret, _MESSAGE, randomness=None, context=None)

    def verify(keys: Any, messages: Any, signatures: Any) -> Any:
        return scheme.verify(keys, messages, signatures, context=None)

    return verify, (np.asarray(public), _MESSAGE, np.asarray(signature))


def _eddsa(meter: _Meter) -> tuple[Any, tuple[np.ndarray, ...]]:
    del meter
    scheme = ed25519.Ed25519()
    public, secret = scheme.keygen(_seed(13, 5))
    signature = scheme.sign(secret, _MESSAGE, randomness=None, context=None)

    def verify(keys: Any, messages: Any, signatures: Any) -> Any:
        return scheme.verify(keys, messages, signatures, context=None)

    return verify, (np.asarray(public), _MESSAGE, np.asarray(signature))


def _eddsa_aggregate(meter: _Meter) -> tuple[Any, tuple[np.ndarray, ...]]:
    """ZIP-215's aggregate: one verdict for the batch, same fixture as above.

    The signature is plain RFC 8032's — signing does not change with the
    rule — so this lane and `eddsa` verify identical bytes and the two
    columns read against each other directly.
    """
    del meter
    scheme = consensus.Ed25519Zip215()
    public, secret = scheme.keygen(_seed(13, 5))
    signature = scheme.sign(secret, _MESSAGE, randomness=None, context=None)
    return scheme.aggregate_verify, (
        np.asarray(public),
        _MESSAGE,
        np.asarray(signature),
    )


def _bip340_rows() -> tuple[bip340.Bip340, tuple[np.ndarray, ...]]:
    scheme = bip340.Bip340()
    public, secret = scheme.keygen(_seed(13, 5))
    signature = scheme.sign(
        secret, _MESSAGE, randomness=np.zeros(32, dtype=np.uint8), context=None
    )
    return scheme, (np.asarray(public), _MESSAGE, np.asarray(signature))


def _bip340(meter: _Meter) -> tuple[Any, tuple[np.ndarray, ...]]:
    del meter
    scheme, rows = _bip340_rows()

    def verify(keys: Any, messages: Any, signatures: Any) -> Any:
        return scheme.verify(keys, messages, signatures, context=None)

    return verify, rows


def _bip340_aggregate(meter: _Meter) -> tuple[Any, tuple[np.ndarray, ...]]:
    del meter
    scheme, rows = _bip340_rows()
    return scheme.aggregate_verify, rows


def _frost(meter: _Meter) -> tuple[Any, tuple[np.ndarray, ...]]:
    """A FROST-verifiable signature, minted directly.

    The aggregate output is a standard Schnorr signature over the group key
    (RFC 9591 Appendix B), so the bench signs it single-party — secret and
    nonce in hand — rather than running the dealer and both rounds; the
    verifier cannot tell the difference, which is the ciphersuite's point.
    """
    del meter
    cs = Secp256k1Sha256()
    secret = int.from_bytes(_seed(13, 5).tobytes(), "big")
    nonce = int.from_bytes(_seed(7, 3).tobytes(), "big")
    public = cs.scalar_base_mult(secret)
    commitment = cs.scalar_base_mult(nonce)
    challenge = frost.compute_challenge(cs, commitment, public, _MESSAGE.tobytes())
    z = (nonce + challenge * secret) % cs.order
    signature = commitment + cs.serialize_scalar(z)
    return cs.verify, (
        np.frombuffer(public, dtype=np.uint8),
        _MESSAGE,
        np.frombuffer(signature, dtype=np.uint8),
    )


_ALL_LANES: dict[str, _Lane] = {
    "ecdsa": _Lane(
        "ECDSA/secp256k1 — independent verdicts",
        ("hash", "mult", "readback"),
        lambda meter: _ecdsa(secp.SECP256K1, meter),
    ),
    "ecdsa_p256": _Lane(
        "ECDSA/P-256 — independent verdicts",
        ("hash", "mult", "readback"),
        lambda meter: _ecdsa(secp.SECP256R1, meter),
    ),
    "eddsa": _Lane(
        "Ed25519 — cofactorless, independent verdicts",
        ("sha512", "decode", "mult"),
        _eddsa,
    ),
    "eddsa_aggregate": _Lane(
        "Ed25519/ZIP-215 — aggregate (random linear combination), one verdict",
        ("sha512", "decode", "mult", "sum"),
        _eddsa_aggregate,
    ),
    "bip340": _Lane(
        "BIP-340 — independent verdicts",
        ("mult", "lift", "readback"),
        _bip340,
    ),
    "bip340_aggregate": _Lane(
        "BIP-340 — aggregate (random linear combination), one verdict",
        ("mult", "lift", "sum"),
        _bip340_aggregate,
    ),
    "frost": _Lane(
        "FROST(secp256k1, SHA-256) — aggregate-output verification",
        ("mult", "lift", "readback"),
        _frost,
    ),
}


def _batch_of(rows: tuple[np.ndarray, ...], batch: int) -> tuple[np.ndarray, ...]:
    return tuple(np.stack([row] * batch) for row in rows)


def _measure(lane: _Lane, batches: Sequence[int], reps: int, meter: _Meter) -> None:
    verify, rows = lane.build(meter)
    verdicts = verify(*_batch_of(rows, 2))
    if not bool(np.all(verdicts)):
        raise AssertionError(f"the {lane.title} fixture fails its own verification")

    print(f"\n=== {lane.title}")
    print(
        f"{'B':>5} {'wall ms':>9} {'work ms':>9} {'per sig ms':>11}"
        + "".join(f"{name + ' %':>11}" for name in lane.buckets)
        + f"{'host %':>9}"
    )
    per_signature: dict[int, float] = {}
    for batch in batches:
        arrays = _batch_of(rows, batch)
        verify(*arrays)  # warm the lazy dtype and curve-constant mints

        def best_of(settle: bool) -> tuple[float, dict[str, float]]:
            meter.settle = settle
            best: float | None = None
            split: dict[str, float] = {}
            for _ in range(reps):
                meter.reset()
                start = time.perf_counter()
                verify(*arrays)
                elapsed = time.perf_counter() - start
                if best is None or elapsed < best:
                    best, split = elapsed, dict(meter.seconds)
            assert best is not None
            return best, split

        wall, _ = best_of(settle=False)
        work, split = best_of(settle=True)
        meter.settle = False

        per_signature[batch] = wall / batch
        metered = sum(split.get(name, 0.0) for name in lane.buckets)
        print(
            f"{batch:>5} {wall * 1e3:>9.1f} {work * 1e3:>9.1f}"
            f" {per_signature[batch] * 1e3:>11.2f}"
            + "".join(
                f"{100 * split.get(name, 0.0) / work:>11.1f}" for name in lane.buckets
            )
            + f"{100 * (work - metered) / work:>9.1f}"
        )
    first, last = per_signature[batches[0]], per_signature[batches[-1]]
    print(
        f"  per-signature: {first / last:.1f}x faster at B={batches[-1]} than at "
        f"B={batches[0]}"
    )


def main(argv: Sequence[str]) -> None:
    del argv
    batches = [int(value) for value in _BATCHES.value]
    meter = _Meter()
    meter.install()
    for name in _LANES.value:
        _measure(_ALL_LANES[name], batches, _REPS.value, meter)


if __name__ == "__main__":
    app.run(main)
