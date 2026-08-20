# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""What ML-DSA's concrete path spends on hashing, and what the sponge choice is worth.

Key generation and signing are concrete — the rejection loop is a host `while`
([`ml_dsa.py`](../ml_dsa.py)) — so every hash on that path is one dispatch whose
result is read back immediately. That is the shape `hash_frx`'s host siblings
exist for, and whether they are worth reaching for is a measurement rather than
an argument, which is what this prints.

Three things it measures:

- **The sponge, host against device, at the shapes ML-DSA actually hashes.** Not
  a sweep over sizes: the concrete path hashes eight distinct `(rows, in, out)`
  shapes per parameter set and those are the ones that decide this. A shape with
  many rows is the interesting one, because that is where one device dispatch is
  amortized over work the host sibling pays per row.
- **Per operation.** `keygen` and `sign_internal` end to end, which is what a
  caller waits for and what the known-answer sweeps pay per case.
- **Per stage of signing.** `H` and the four samplers, at their real shapes and
  their real per-signature call counts, in the order `sign_internal` runs them.
  The counts are what turn a per-call difference into a per-signature one.

Verification is deliberately absent. It is traced and batch-first, so its sponge
is not a choice — a host hash cannot be called on a tracer at all, which is what
`ByteHash`'s return type says
([`conventions.md`](../../../../docs/reference/conventions.md)).

The numbers belong on the issue that acts on them, not here. This is committed so
that a re-measurement compares against the same harness rather than a fresh one,
which is the reason [`verify_bench`](../../../hashbased/testing/verify_bench.py)
is committed too.

    bazel run //sig_frx/lattice/mldsa/testing:sign_bench -- --parameter_sets=ML-DSA-65
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any, NamedTuple

import frx.numpy as fnp
import numpy as np
from absl import app, flags
from frx import Array
from hash_frx.keccak.byte_hashes import (
    SHAKE128_RATE,
    SHAKE256_RATE,
    HostShake128,
    HostShake256,
    Shake128,
    Shake256,
)

from sig_frx.lattice.mldsa import arith, encoding, ml_dsa, sampling
from sig_frx.lattice.mldsa.arith import N, Q

_PARAMETER_SETS = flags.DEFINE_list(
    "parameter_sets",
    sorted(ml_dsa.PARAMETER_SETS),
    "Parameter sets to measure. All three by default: the sampler shapes scale"
    " with (k, ℓ) and the per-signature call counts do not, so the sets do not"
    " rank the same way for every stage.",
)
_REPS = flags.DEFINE_integer(
    "reps", 5, "Timed repetitions per row; the fastest is reported."
)
_MESSAGE_SIZE = flags.DEFINE_integer(
    "message_size",
    32,
    "Bytes of message to sign. `μ` is the one hash whose input grows with it, and"
    " it is one call per signature.",
)
_SIGNATURES = flags.DEFINE_integer(
    "signatures",
    8,
    "Distinct messages the per-operation signing row averages over. The loop's"
    " trip count is per message, so one message measures one trip count.",
)


class _Call(NamedTuple):
    """One sponge call's shape and which implementation took it.

    Keyed on the whole triple rather than on the output length alone: at
    ML-DSA-87 the commitment hash is 64 bytes and so are `μ` and `ρ′`, so a key
    that dropped the message length would merge the loop body's hash with two
    that run once and report a trip count two too high.
    """

    rows: int
    message: int
    output: int
    family: str


class _Timed:
    """A `ByteHash` that times its digests, and the family that hands them out.

    `hashes.shake128` / `shake256` are the one place ML-DSA reaches a sponge
    through, so replacing them counts every hash the scheme takes and attributes
    nothing else to hashing — the exactness a dependency-injected seam buys, which
    is what [`verify_bench`](../../../hashbased/testing/verify_bench.py) wraps for
    the same purpose. Before those helpers existed there was no such point: the
    modules named `Shake256` directly.
    """

    def __init__(self, family: Any, record: Callable[[_Call, float], None]) -> None:
        self._family = family
        self._record = record

    def __call__(self, size: int) -> Any:
        hash_ = self._family(size)
        # The family the scheme chose, recorded rather than inferred from what
        # the call went on to cost: a wide enough host batch — `ExpandA`'s `k·ℓ`
        # streams at ML-DSA-87 — costs what a small dispatch does, so a threshold
        # on the timing labels it wrong exactly where the comparison is closest.
        family = self._family.__name__
        record = self._record

        class _Hash:
            digest_size = hash_.digest_size
            fusion_path = hash_.fusion_path

            def digest(self, msg: Any) -> Any:
                start = time.perf_counter()
                out = _blocked(hash_.digest(msg))
                rows, message = np.shape(msg)
                record(
                    _Call(int(rows), int(message), size, family),
                    time.perf_counter() - start,
                )
                return out

        return _Hash()


# FIPS 204 Table 1's expected number of repetitions of the rejection loop. A
# prediction, not a measurement: an individual signature runs its own count, and
# `_operations` reports that one beside this.
_EXPECTED_ITERATIONS = {"ML-DSA-44": 4.25, "ML-DSA-65": 5.1, "ML-DSA-87": 3.85}


def _blocked(value: Any) -> Any:
    """Wait for a dispatch to land, so a timing measures work and not queueing."""
    ready = getattr(value, "block_until_ready", None)
    if ready is not None:
        ready()
    return value


@contextlib.contextmanager
def _counting(record: Callable[[_Call, float], None]) -> Iterator[None]:
    """Every sponge the scheme takes, timed, for the duration of the block."""
    targets = ((ml_dsa, "shake256"), (sampling, "shake256"), (sampling, "shake128"))
    saved = [(module, name, getattr(module, name)) for module, name in targets]
    for module, name, original in saved:
        setattr(module, name, _wrapped(original, record))
    try:
        yield
    finally:
        for module, name, original in saved:
            setattr(module, name, original)


def _wrapped(
    chooser: Callable[..., Any], record: Callable[[_Call, float], None]
) -> Callable[..., Any]:
    """`shake128` / `shake256` with the family it returns timed.

    The choice itself is left alone — a wrapper that picked for the scheme would
    be measuring something the scheme does not do.
    """
    return lambda *values: _Timed(chooser(*values), record)


def _fastest(call: Callable[[], object], reps: int) -> float:
    """The fastest of `reps` timed calls, in seconds. Assumes a warm caller."""
    best = None
    for _ in range(reps):
        start = time.perf_counter()
        _blocked(call())
        elapsed = time.perf_counter() - start
        if best is None or elapsed < best:
            best = elapsed
    assert best is not None
    return best


class _Shape:
    """One `(rows, input, output)` the concrete path hashes, and how often.

    `calls` is per signature, which is what makes a per-call difference readable
    as a per-signature one: the commitment hash and every sampler but `ExpandA`
    and `ExpandS` run once per rejection-loop iteration, and the loop runs
    `iterations` times.
    """

    def __init__(
        self,
        label: str,
        *,
        rows: int,
        message: int,
        output: int,
        shake: int,
        calls: float,
    ) -> None:
        self.label = label
        self.rows = rows
        self.message = message
        self.output = output
        self.shake = shake
        self.calls = calls


def _shapes(params: ml_dsa.MlDsaParams, message_size: int) -> list[_Shape]:
    """Every shape `keygen` and `sign_internal` hash, derived rather than listed.

    Derived from the parameter set for the same reason the sizes on
    `MlDsaParams` are: a second transcription of what Table 1 already determines
    is a second chance to mistype it, and a bench that hashes a shape the scheme
    does not is measuring nothing.
    """
    # §7.3's budgets, as the samplers compute them.
    ntt_blocks = sampling.budget(N, (Q, 1 << 23), SHAKE128_RATE // 3)
    bounded_blocks = sampling.budget(
        N, (sampling._BOUNDED_THRESHOLD[params.eta], 16), 2 * SHAKE256_RATE
    )
    ball_allowance = sampling.budget(params.tau, (N - params.tau + 1, N), 1)
    # The mask is squeezed to exactly what the unpacking consumes, and `w1` is
    # packed at `bitlen((q−1)/(2γ2) − 1)` — Algorithms 34 and 28.
    mask_width = 1 + (params.gamma1 - 1).bit_length()
    w1_bytes = 32 * params.k * ((Q - 1) // (2 * params.gamma2) - 1).bit_length()
    # Table 1's expected repetition count is per parameter set; the loop body's
    # hashes are charged at it so the per-signature column is what a average
    # caller pays. The per-operation section reports the count the measured
    # signatures actually ran, which is what reconciles the two.
    iterations = _EXPECTED_ITERATIONS[_name_of(params)]
    return [
        _Shape(
            "H: keygen expand",
            rows=1,
            message=params.seed_size + 2,
            output=128,
            shake=256,
            calls=0.0,
        ),
        _Shape(
            "H: tr = H(pk, 64)",
            rows=1,
            message=params.public_key_size,
            output=64,
            shake=256,
            calls=0.0,
        ),
        _Shape(
            "H: μ",
            rows=1,
            message=64 + message_size,
            output=64,
            shake=256,
            calls=1.0,
        ),
        _Shape("H: ρ′", rows=1, message=128, output=64, shake=256, calls=1.0),
        _Shape(
            "H: c̃ (per iteration)",
            rows=1,
            message=64 + w1_bytes,
            output=params.commitment_hash_size,
            shake=256,
            calls=iterations,
        ),
        _Shape(
            "ExpandA (keygen/sign)",
            rows=params.k * params.ell,
            message=34,
            output=ntt_blocks * SHAKE128_RATE,
            shake=128,
            calls=1.0,
        ),
        _Shape(
            "ExpandS (keygen only)",
            rows=params.ell + params.k,
            message=66,
            output=bounded_blocks * SHAKE256_RATE,
            shake=256,
            calls=0.0,
        ),
        _Shape(
            "ExpandMask (per iteration)",
            rows=params.ell,
            message=66,
            output=32 * mask_width,
            shake=256,
            calls=iterations,
        ),
        _Shape(
            "SampleInBall (per iteration)",
            rows=1,
            message=params.commitment_hash_size,
            output=8 + ball_allowance,
            shake=256,
            calls=iterations,
        ),
    ]


def _name_of(params: ml_dsa.MlDsaParams) -> str:
    for name, candidate in ml_dsa.PARAMETER_SETS.items():
        if candidate == params:
            return name
    raise ValueError("parameters are not one of Table 1's sets")


def _sponge_table(params: ml_dsa.MlDsaParams, message_size: int, reps: int) -> None:
    """The device sponge against its host sibling, at each shape and in aggregate.

    The last column is the one that decides anything: a per-call difference
    weighted by how many of that call a signature makes. `ExpandA` and `ExpandS`
    are charged zero there because key generation is not what the column is
    about, and they are still timed because the shape is where the batch axis
    argues back — a host row is a `hashlib` call and `k·ℓ` of them are not one
    dispatch.
    """
    print(
        f"{'shape':>28} {'rows':>5} {'in B':>7} {'out B':>7} {'device ms':>10} "
        f"{'host ms':>9} {'ratio':>7} {'calls/sig':>10} {'saved ms/sig':>13}"
    )
    total_device = total_host = 0.0
    for shape in _shapes(params, message_size):
        msg = np.frombuffer(
            bytes((i * 37 + 11) % 256 for i in range(shape.rows * shape.message)),
            dtype=np.uint8,
        ).reshape(shape.rows, shape.message)
        device_hash = (Shake256 if shape.shake == 256 else Shake128)(shape.output)
        host_hash = (HostShake256 if shape.shake == 256 else HostShake128)(shape.output)
        # Byte-equality first: a timing comparison between two hashes that do not
        # agree is a comparison of two different functions.
        if not np.array_equal(
            np.asarray(device_hash.digest(msg)), np.asarray(host_hash.digest(msg))
        ):
            raise AssertionError(f"{shape.label}: host and device digests differ")
        device = _fastest(lambda: device_hash.digest(msg), reps)
        host = _fastest(lambda: host_hash.digest(msg), reps)
        total_device += device * shape.calls
        total_host += host * shape.calls
        print(
            f"{shape.label:>28} {shape.rows:>5} {shape.message:>7} "
            f"{shape.output:>7} {device * 1e3:>10.3f} {host * 1e3:>9.3f} "
            f"{device / host:>6.0f}x {shape.calls:>10.2f} "
            f"{(device - host) * shape.calls * 1e3:>13.1f}"
        )
    print(
        f"  per signature: {total_device * 1e3:.1f} ms of device sponge against "
        f"{total_host * 1e3:.1f} ms of host sponge"
    )


def _operations(name: str, message_size: int, reps: int, signatures: int) -> None:
    """`keygen` and `sign_internal` end to end — what a caller actually waits for.

    Signing is reported as a **mean over distinct messages**, not as the fastest
    of repeats on one. Its cost is a rejection loop, so a single message has a
    trip count of its own and repeating it measures that one seed however many
    times; the mean over messages is the statistic a caller pays, and the one
    commensurable with the expected repetition count the sponge table charges
    per signature. `keygen` has no loop and is reported as the fastest repeat.
    """
    scheme = ml_dsa.named(name, deterministic=True)
    params = scheme.params
    seed = np.frombuffer(
        bytes((i * 13 + 5) % 256 for i in range(params.seed_size)), dtype=np.uint8
    )
    messages = [
        np.frombuffer(
            bytes((i * 7 + 3 * which + 1) % 256 for i in range(message_size)),
            dtype=np.uint8,
        )
        for which in range(signatures)
    ]
    public_key, secret_key = (np.asarray(part) for part in scheme.keygen(seed))
    _blocked(scheme.sign_internal(secret_key, messages[0]))  # warm the caches
    keygen = _fastest(lambda: scheme.keygen(seed), reps)
    start = time.perf_counter()
    for message in messages:
        _blocked(scheme.sign_internal(secret_key, message))
    sign = (time.perf_counter() - start) / signatures
    print(f"  keygen        {keygen * 1e3:>9.1f} ms  (fastest of {reps})")
    print(f"  sign_internal {sign * 1e3:>9.1f} ms  (mean of {signatures} messages)")

    # The same signatures again with every sponge timed from inside, which is
    # what separates "hashing" from the rest without attributing either by
    # subtraction of two separately-measured things.
    profile: dict[_Call, tuple[int, float]] = {}

    def record(call: _Call, elapsed: float) -> None:
        was_calls, was_seconds = profile.get(call, (0, 0.0))
        profile[call] = (was_calls + 1, was_seconds + elapsed)

    with _counting(record):
        for message in messages:
            _blocked(scheme.sign_internal(secret_key, message))
    seconds = sum(spent for _, spent in profile.values())
    hashing = seconds / signatures
    # `c̃` is the one hash of the loop body, so its count is the trip count —
    # which is otherwise unobservable from outside, `sign_internal` returning
    # only the signature. Matched on its whole shape, since its output length
    # alone does not identify it at every parameter set.
    commitment = next(
        shape for shape in _shapes(params, message_size) if shape.calls > 1
    )
    iterations = (
        sum(
            count
            for call, (count, _) in profile.items()
            if (call.rows, call.message, call.output)
            == (commitment.rows, commitment.message, commitment.output)
        )
        / signatures
    )
    print(
        f"  of which hashing {hashing * 1e3:>6.1f} ms "
        f"({100 * hashing / sign:.0f}%), over "
        f"{sum(count for count, _ in profile.values()) / signatures:.1f} "
        f"sponge calls and {iterations:.2f} loop iterations per signature "
        f"(Table 1 expects {_EXPECTED_ITERATIONS[name]})"
    )
    print(
        f"  {'rows':>6} {'in B':>7} {'out B':>7} {'calls/sig':>10} {'ms/sig':>8} "
        f"{'sponge':>14}"
    )
    for call, (count, spent) in sorted(profile.items(), key=lambda item: -item[1][1]):
        print(
            f"  {call.rows:>6} {call.message:>7} {call.output:>7} "
            f"{count / signatures:>10.2f} {spent / signatures * 1e3:>8.2f} "
            f"{call.family:>14}"
        )


def _stages(name: str, message_size: int, reps: int) -> None:
    """`H` and the four samplers at their real shapes, in `sign_internal`'s order.

    Called directly rather than through the scheme: the hash is not injected
    here — `ml_dsa` and `sampling` name `Shake256` — so there is no seam to wrap,
    and timing the functions the scheme calls at the arguments it calls them with
    is what stands in for one.

    Each stage is timed twice, once on a host argument and once on the same bytes
    lifted onto the device, and the namespace of the result is printed beside
    them. That is the column the sponge question turns on: a stage whose cost does
    not move between the two is not paying for its argument's namespace, and one
    whose result is a device array hands the next stage a lifted argument whether
    or not that stage asked for one.

    **Both columns are what-ifs, and the profile above is what signing runs.**
    `c̃` is hashed over the commitment, which comes off `arith.intt` and is
    therefore a device array however the seed that produced it arrived — so
    signing takes the device column for it, and the host column says what it
    would cost if the commitment came home first rather than what it costs.
    """
    scheme = ml_dsa.named(name, deterministic=True)
    params = scheme.params
    seed = np.frombuffer(
        bytes((i * 13 + 5) % 256 for i in range(params.seed_size)), dtype=np.uint8
    )
    message = np.frombuffer(
        bytes((i * 7 + 3) % 256 for i in range(message_size)), dtype=np.uint8
    )
    public_key, secret_key = (np.asarray(part) for part in scheme.keygen(seed))
    key = scheme._parse_secret_key(secret_key)
    mu = ml_dsa._h(key.tr, message, size=64)
    w1 = np.zeros((params.k, N), dtype=np.int32)
    c_tilde = ml_dsa._h(
        mu,
        encoding.w1_encode(w1, params.gamma2),
        size=params.commitment_hash_size,
    )
    rho_prime = ml_dsa._h(key.key, np.zeros(32, dtype=np.uint8), mu, size=64)

    # The same bytes on both sides, so the pair differs by namespace and nothing
    # else. `np.asarray` is the conversion that cannot silently succeed on a
    # tracer, which is what makes the host column the honest one to build from.
    def host(value: Any) -> np.ndarray:
        return np.asarray(value, dtype=np.uint8)

    def lifted(value: Any) -> Any:
        return fnp.asarray(value, dtype=fnp.uint8)

    stages: list[tuple[str, Callable[[Callable[[Any], Any]], object]]] = [
        ("H: μ", lambda cast: ml_dsa._h(cast(key.tr), message, size=64)),
        (
            "H: ρ′",
            lambda cast: ml_dsa._h(
                cast(key.key), np.zeros(32, dtype=np.uint8), mu, size=64
            ),
        ),
        (
            "H: c̃",
            lambda cast: ml_dsa._h(
                cast(mu),
                encoding.w1_encode(w1, params.gamma2),
                size=params.commitment_hash_size,
            ),
        ),
        (
            "ExpandA",
            lambda cast: sampling.expand_a(cast(key.rho), params.k, params.ell),
        ),
        (
            "ExpandS",
            lambda cast: sampling.expand_s(
                cast(rho_prime), params.k, params.ell, params.eta
            ),
        ),
        (
            "ExpandMask",
            lambda cast: sampling.expand_mask(
                cast(rho_prime), 0, params.ell, params.gamma1
            ),
        ),
        (
            "SampleInBall",
            lambda cast: sampling.sample_in_ball(cast(c_tilde), params.tau),
        ),
    ]
    # What the loop body does with a sampler's output. `frx.lax.ntt` has no host
    # implementation, so this half is lifted whatever namespace it is handed —
    # which is what makes it the place a saving upstream can reappear as a cost:
    # a sampler that stops lifting hands the lift to its consumer rather than
    # removing it.
    a_hat = sampling.expand_a(key.rho, params.k, params.ell)
    y = sampling.expand_mask(rho_prime, 0, params.ell, params.gamma1)
    stages += [
        ("ntt(y)", lambda cast: arith.ntt(arith.to_field(_int32(cast, y)))),
        (
            "A·y round trip",
            lambda cast: arith.intt(
                arith.matrix_vector(a_hat, arith.ntt(arith.to_field(_int32(cast, y))))
            ),
        ),
    ]

    print(f"  {'stage':<14} {'host arg ms':>12} {'device arg ms':>14} {'result':>8}")
    for label, call in stages:
        result = _blocked(call(host))  # warm the caches, and read the namespace
        on_host = _fastest(lambda: call(host), reps)
        _blocked(call(lifted))
        on_device = _fastest(lambda: call(lifted), reps)
        landed = "device" if isinstance(_first(result), Array) else "host"
        print(
            f"  {label:<14} {on_host * 1e3:>12.2f} {on_device * 1e3:>14.2f} "
            f"{landed:>8}"
        )


def _first(value: Any) -> Any:
    """The first array of a stage's result — `expand_s` returns a pair."""
    return value[0] if isinstance(value, tuple) else value


def _int32(cast: Callable[[Any], Any], value: Any) -> Any:
    """`cast` on a coefficient vector, which is `int32` and not bytes."""
    lifted = cast(np.zeros(0, dtype=np.uint8))
    module = fnp if isinstance(lifted, Array) else np
    return module.asarray(value, dtype=np.int32)


def main(argv: Sequence[str]) -> None:
    del argv
    reps, message_size = _REPS.value, _MESSAGE_SIZE.value
    for name in _PARAMETER_SETS.value:
        params = ml_dsa.PARAMETER_SETS[name]
        print(
            f"\n=== {name}: k={params.k} ℓ={params.ell} η={params.eta} τ={params.tau}"
        )
        print("\n-- the sponge, at the shapes this scheme hashes")
        _sponge_table(params, message_size, reps)
        print("\n-- per operation, eager")
        _operations(name, message_size, reps, _SIGNATURES.value)
        print("\n-- per stage of signing, one call each")
        _stages(name, message_size, reps)


if __name__ == "__main__":
    app.run(main)
