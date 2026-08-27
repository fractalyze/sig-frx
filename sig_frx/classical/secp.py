# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The SEC curves as curated zk_dtypes point types, plus what those omit.

The group law, the scalar multiplication, and the point representations live
in zk_dtypes — the `secp256k1_g1_*` / `secp256r1_g1_*` dtypes carry `+`, `-`,
and point × scalar as ufuncs over fused C++ kernels — so nothing in this repo
implements curve arithmetic for these curves anymore. What this module keeps
is exactly what the dtypes do not expose:

- the pairing that names which dtypes form one curve, with the integer
  view of its constants derived from the dtypes' own metadata
  (`ecinfo`/`pfinfo`) — the bounds checks, wire encodings, and readbacks
  the standards define on integers still need them as integers (exact,
  host-only per `docs/reference/security.md`);
- the curve-equation membership check — the dtypes construct off-curve
  coordinates without complaint, and SEC 1's encoding rules reject them;
- the square-root lift that names a point by x plus a parity bit, which
  every recovery id, compressed key, and x-only key performs;
- the byte/int ↔ point codecs the wire encodings need.

## The three seams that place, and why they are the exception

`multiple` and `double_multiple` put their point batch on the device when it
is large enough to pay for the trip, and `lift_x_to_parity` does the same
with the square root's coordinate batch. Everything else here follows the
namespace its arguments arrive in, and a batch that arrives already traced is
left where the caller put it.

That is a documented exception to "a value is used in the namespace it
arrives in" (`docs/reference/conventions.md`), so it needs its reason stated
rather than assumed. The rule exists to stop a callee dragging a *signing*
path onto the device, where an integer array lane is 32 bits and a host
Python integer has no width. Neither half of that hazard reaches any of the
three: what they place is a point dtype or a base-field dtype, neither of
which carries an integer lane, and the signing callers arrive at `B = 1` —
`host_multiple_of_g` at the point seams, FROST's `deserialize_element` at the
lift — which is below any threshold, so they never move.

What the exception buys is that the decision exists once. Every verification
batch in this repo is born in one of five places and consumed by one of these
seams; asking each birthplace to remember would be one decision written five
times, and a sixth that forgets would be silently slow rather than wrong.
That is why `_place` is one function all three call: one round trip, one
number, and — because it reads the admission probe off the dtype it is handed
rather than taking it as an argument — no way to pair a batch with the wrong
question. That last part matters here: frx's rows are per dtype, so at the
pinned wheel P-256's field is admitted while its points are not, and a seam
that asked the point question on the square root's behalf would strand it on
the host over a gap in a type it never touches. It is worth deciding:
measured on an RTX 5090 with full-size scalars, the device form of `multiple`
runs 7.3x the host at B=256 and 105x at B=4096, where it costs no more than it
does at B=64. Those are stage ratios; the share a verification spends there is
`ecdsa/core.py`'s.

Everything that turns a point into integers is host by nature and stays there —
`affine_ints`, `uncompressed_rows`, `on_curve`, `secret_scalar` — because the
standards define those on integers and Python has no width, which is why
`lift_x_to_parity` places its ladder and not its readback. Host is not the
same as row-at-a-time, though, and the two get conflated: `on_curve_rows` is
the same host arithmetic as `on_curve` over a `[B]` array instead of a 0-d one
per row, and that stage alone is 9x at B=1024. A function stays on the host because
its *result* is integers, not because its work has to be scalar. `is_identity` is
the one exception on the wrong side of that line, and its docstring carries
the reason (fractalyze/xla#594).

There is one `jit` here and it is the square root's, which is the only seam
that can use one. The point seams are single fused ops, so compiling one
measured identical to running it eagerly while adding a compile per batch
shape. The square root is ~325 *chained* multiplications instead, so eager it
is a chain of that many sequential launches — launch-bound, not
compute-bound, which is visible in a steady state that does not move with the
batch across B = 24, 64, 256 and 1024 alike. Compiled it is one kernel, ~19x
cheaper and also flat. See `sqrt` for what that compile costs, and for why
shrinking the ladder is the wrong response to it.

The GPU story for these curves remains EC kernels over these same dtypes
(fractalyze/sig-frx#139), not a traced re-derivation of the group law.

## Three dtype gotchas the codecs absorb

A point dtype's scalar branch turns a bare integer into `k·G`, so a
coordinate pair must arrive as one tuple argument (`point((x, y))`), never
as a row of ints. And a field value refuses an integer in `[n, 2²⁵⁶)` —
construction and int *operands* in a scalar expression alike abort instead
of reducing (fractalyze/zk_dtypes#179) — so every scalar is reduced `% n`
in Python before it meets a dtype value; sound, since `k·P = (k mod n)·P`.

The third is why the substrate is the Montgomery variants: a non-Montgomery
prime field has no REDC to lower to, so a traced reduction becomes a
bit-serial shift-and-subtract loop — measured at 14× the Montgomery cost for
batched scalar multiplication on an RTX 5090. The price is that storage stops
being the residue. Constructors and `astype` convert, and `ecinfo` does not:
it reports constants as stored, so `Curve` reads them back through
`from_raw`. `raw` on a point likewise hands back `x · R`, so `affine_ints`
views the coordinates as field values and converts those instead. Every
readback in this module goes through one of those two, which is what keeps
the wire encodings defined on residues.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
import zk_dtypes
from frx.typing import ArrayLike

from sig_frx.arrays import namespace, traced
from sig_frx.classical import group


@dataclass(frozen=True)
class Curve:
    """A short-Weierstrass curve: four dtype handles, everything else derived.

    The integers come off the dtypes' own metadata — `ecinfo` carries the
    equation and the base point, `pfinfo` the two moduli — so the pinned
    wheel is the single source of truth and a Curve whose integers disagree
    with its dtypes cannot be built (`testing/secp_test.py` holds the
    derivation against SEC 2's published parameters). The base field is
    stated rather than derived: `ecinfo(...).base_field_dtype` returns the
    scalar field (fractalyze/zk_dtypes#182), so deriving it would pair the
    wrong modulus.

    Frozen with value equality so a scheme holding one rides pytree aux
    without re-tracing per instance (`signature.py`'s rule) — the dtype
    handles are classes, identical across instances of the same curve.
    """

    point: Any  # the affine G1 dtype, standard domain
    accumulator: Any  # the jacobian G1 dtype — what sums and multiples ride
    scalar: Any  # the scalar-field dtype
    field: Any  # the base-field dtype — what the lift's arithmetic runs over

    @functools.cached_property
    def p(self) -> int:
        """Base field modulus."""
        return zk_dtypes.pfinfo(self.field).modulus

    @functools.cached_property
    def n(self) -> int:
        """Order of the base point, prime."""
        return zk_dtypes.pfinfo(self.scalar).modulus

    def _residue(self, stored: Any) -> int:
        """A curve constant as its residue, whatever the storage is.

        `ecinfo` reports the constants the way the point type stores them,
        which for a Montgomery type is `value · R`, not the value. `from_raw`
        adopts the storage without re-encoding it, and the field's own
        conversion reads the residue back out — the same `astype`-not-bitcast
        rule the module docstring states, applied to metadata.
        """
        return int(self.field.from_raw(int(stored)))

    @functools.cached_property
    def a(self) -> int:
        """Curve coefficient a."""
        return self._residue(zk_dtypes.ecinfo(self.point).a)

    @functools.cached_property
    def b(self) -> int:
        """Curve coefficient b."""
        return self._residue(zk_dtypes.ecinfo(self.point).b)

    @functools.cached_property
    def gx(self) -> int:
        """Base point, affine x."""
        return self._residue(zk_dtypes.ecinfo(self.point).gx)

    @functools.cached_property
    def gy(self) -> int:
        """Base point, affine y."""
        return self._residue(zk_dtypes.ecinfo(self.point).gy)

    @functools.cached_property
    def one(self) -> np.ndarray:
        return np.array(1, dtype=self.field)

    @functools.cached_property
    def coeff_a(self) -> np.ndarray:
        return np.array(self.a % self.p, dtype=self.field)

    @functools.cached_property
    def coeff_b(self) -> np.ndarray:
        return np.array(self.b % self.p, dtype=self.field)

    @functools.cached_property
    def generator(self) -> np.ndarray:
        """`G` as a `[1]`-shaped affine array, broadcastable over any batch."""
        return np.array([self.point((self.gx, self.gy))], dtype=self.point)

    @functools.cached_property
    def traceable(self) -> bool:
        """Whether a traced array can hold this curve's points at all."""
        return _admits(self.point)

    @functools.cached_property
    def field_traceable(self) -> bool:
        """Whether a traced array can hold this curve's base-field elements.

        A different question from `traceable`, because the admission table is
        keyed on the dtype: at the pinned wheel `secp256r1_bf_mont` has a row
        while `secp256r1_g1_affine_mont` has none. The lift asks this one, and
        `_place` asks it for itself off whatever dtype it is handed.
        """
        return _admits(self.field)


# SEC 2 §2.4.1, "Recommended Parameters secp256k1". The Koblitz curve.
SECP256K1 = Curve(
    point=zk_dtypes.secp256k1_g1_affine_mont,
    accumulator=zk_dtypes.secp256k1_g1_jacobian_mont,
    scalar=zk_dtypes.secp256k1_sf_mont,
    field=zk_dtypes.secp256k1_bf_mont,
)

# SEC 2 §2.4.2, "Recommended Parameters secp256r1" — NIST's P-256
# (FIPS 186-5 §6.1.1 points at SP 800-186 §3.2.1.3 for the same values).
SECP256R1 = Curve(
    point=zk_dtypes.secp256r1_g1_affine_mont,
    accumulator=zk_dtypes.secp256r1_g1_jacobian_mont,
    scalar=zk_dtypes.secp256r1_sf_mont,
    field=zk_dtypes.secp256r1_bf_mont,
)


# Where a verification batch stops being cheaper on the host — see `_place`
# for the measurement and why one number covers both backends.
DEVICE_MIN_BATCH = 64


@functools.cache
def _admits(dtype: Any) -> bool:
    """Whether frx's admission table has a row for `dtype`.

    Keyed on the dtype because that is what the table is keyed on: a curve's
    points and its base field are admitted independently, and at the pinned
    wheel P-256 has a row for the second and none for the first. Probed rather
    than listed — a list would be a second place to update and would go stale
    silently the moment a row lands, where this starts answering `True` on its
    own.

    Any failure answers `False`, not only the `TypeError` seen today. The
    fallback is the host path, which is always correct, so a probe that
    guesses wrong costs speed — while one that let a new exception type
    through would restore the crash this exists to stop, at the batch sizes no
    gate covers.
    """
    try:
        fnp.asarray(np.zeros(1, dtype=dtype))
    except Exception:  # noqa: BLE001 — see above; the fallback is correct.
        return False
    return True


def _place(values: ArrayLike) -> Any:
    """A batch moved off the host once it is large enough to pay for the trip.

    The repo's only exception to the rule in
    `docs/reference/conventions.md` that a callee does not lift its caller's
    value, and the module docstring's "the three seams that place" section is
    where that is argued. The short form: the rule exists to stop a signing
    path being dragged onto a 32-bit integer lane, neither a point dtype nor a
    field dtype has one, and every signing caller arrives at `B = 1`, below any
    threshold.

    The decision is a batch-size threshold because the cost it is trading
    against is a fixed one. Measured on an RTX 5090 with full-size scalars,
    lifting a batch, multiplying and reading it back costs the same on CUDA
    regardless of size, so it loses badly to the host on a single signature —
    27x slower — and wins from roughly a batch of 64 up, 1.7x there and 26x at
    1 024. The CPU backend has no such floor and is ahead from a
    batch of 2, so one threshold picked for CUDA is safe for both: below it
    nothing moves and nothing regresses, above it both backends gain. One
    number covers all three seams because two that drifted apart would be two
    answers to one question, not because the crossovers are identical.

    A dtype a traced array cannot hold stays on the host at every size. That is
    not a tuning decision: lifting secp256r1's *points* raises rather than
    running slowly, so a batch of P-256 signatures large enough to cross the
    threshold would fail outright — which the KAT gate cannot see, because its
    batches are smaller than that. The probe is read off the values rather than
    passed in, so a seam cannot pair a batch with the wrong question, and it is
    consulted only after the threshold check — a signing path at `B = 1` must
    not pay a device round trip to be told it is staying home.

    A batch that is already traced is left alone, which is what lets the seams
    nest: `double_multiple` places once and the `multiple` calls under it then
    see a decision already made.
    """
    if traced(values):
        return values
    host = np.asarray(values)
    if host.shape[0] < DEVICE_MIN_BATCH or not _admits(host.dtype):
        return host
    return fnp.asarray(host)


def multiple(curve: Curve, scalars: list[int], points: ArrayLike) -> np.ndarray:
    """`scalars[i] · points[i]`, one batched kernel call — `[B]` jacobian.

    Scalars reduce `% n` in Python first (the dtype gotcha above); the
    reduction is the group's own fact, `k·P = (k mod n)·P`.

    The batch is placed first (`_place`) and the scalars are then built in the
    namespace it ended up in, so a batch large enough to pay for the device
    runs there and a small one — a single signature especially — does not. A
    caller that has already placed its batch keeps that choice.

    Measured on an RTX 5090 with full-size scalars, the device form runs 7.3x
    the host at B=256 and 105x at B=4096, where it costs no more than it does
    at B=64. **Those are stage ratios** — this call against itself in the two
    namespaces, not a share of any operation: the readback that follows it is
    host work either way, and moving that is what `affine_ints` is waiting on.
    What a verification spends here is `ecdsa/core.py`'s to state, and it does.
    """
    points = _place(points)
    xnp = namespace(points)
    reduced = xnp.asarray(np.array([k % curve.n for k in scalars], dtype=curve.scalar))
    return points * reduced


def double_multiple(
    curve: Curve, g_scalars: list[int], point_scalars: list[int], points: ArrayLike
) -> np.ndarray:
    """`g_scalars[i]·G + point_scalars[i]·points[i]`, two batched kernels.

    The two-term combination every verification equation reduces to, and
    the one seam a fused MSM kernel would replace (fractalyze/sig-frx#139)
    — a subtraction folds into the scalar as `n - e`.

    `G` is a host constant, so it is lifted to wherever the batch already is
    rather than pulling the batch back to it — `np.asarray` here would
    materialize a device batch silently and cost the caller its whole lift.
    It is passed at its own `[1]` shape and left to broadcast against the
    scalars, which is what `generator` is documented for: expanding it to `[B]`
    first would allocate and transfer `B` copies of one point on the device,
    where the host got the same thing as a zero-stride view for nothing.

    The placement happens here rather than in the two `multiple` calls below,
    because `G` arrives `[1]`-shaped and would never reach the threshold on
    its own: deciding once for the batch and lifting `G` to wherever it landed
    is what keeps the two terms in the same namespace.
    """
    points = _place(points)
    xnp = namespace(points)
    return multiple(curve, g_scalars, xnp.asarray(curve.generator)) + multiple(
        curve, point_scalars, points
    )


def schnorr_verdicts(
    curve: Curve,
    base_scalars: list[int],
    challenge_scalars: list[int],
    key_points: ArrayLike,
    claimed_xs: list[int],
    claimed_parities: list[int],
    ok: ArrayLike,
) -> np.ndarray:
    """Per-row Schnorr acceptance: does `base·G - challenge·P` equal the
    claimed `R`, named by its x and y-parity bit? `bool[B]`.

    The readback every Schnorr verifier on this substrate shares — BIP-340
    pins the claimed parity even, RFC 9591 reads it off `R`'s compressed
    prefix — with the subtraction folded into the challenge as `n - c`.
    The identity is rejected before the coordinate compare: it reads back
    as `(0, 0)` (see `affine_ints`), which a claim of `x = 0` must not
    match. ECDSA does not ride this — its verdict compares `x mod n` and
    carries no parity. Rows already failed in `ok` carry junk their
    cleared verdict drops.
    """
    big_r = double_multiple(
        curve,
        base_scalars,
        [-challenge % curve.n for challenge in challenge_scalars],
        key_points,
    )
    gone = is_identity(curve, big_r)
    verdicts = [
        bool(valid) and not bool(dead) and x == want_x and y % 2 == want_parity
        for (x, y), valid, dead, want_x, want_parity in zip(
            affine_ints(curve, big_r),
            np.asarray(ok),
            gone,
            claimed_xs,
            claimed_parities,
        )
    ]
    return np.array(verdicts, dtype=bool)


def compressed_bytes(curve: Curve, x: int, y: int) -> bytes:
    """One point as SEC 1 `02|03 ‖ X` — the parity byte and the x coordinate.

    The width comes off the curve's base field, which is the coordinate's own,
    rather than off a scalar: the two agree on secp256k1 and need not.

    What a caller does about the identity is the caller's, because the schemes
    disagree: BIP-327 encodes it as an all-zero point and RFC 9591 says it has
    no encoding at all. This encodes whatever coordinates it is handed.
    """
    width = (curve.p.bit_length() + 7) // 8
    return (2 + (y & 1)).to_bytes(1, "big") + x.to_bytes(width, "big")


def uncompressed_rows(curve: Curve, points: ArrayLike, ok: ArrayLike) -> np.ndarray:
    """Each point as SEC 1 `04 ‖ X ‖ Y`, masked rows zeroed: `uint8[B, 65]`.

    The zeroing is the codec rule shared by every rejecting consumer: a
    zeroed row cannot be mistaken for a key.
    """
    ok = np.asarray(ok)
    rows = np.zeros((ok.shape[0], 65), dtype=np.uint8)
    for i, ((x, y), valid) in enumerate(zip(affine_ints(curve, points), ok)):
        if valid:
            rows[i, 0] = 4
            rows[i, 1:33] = np.frombuffer(x.to_bytes(32, "big"), dtype=np.uint8)
            rows[i, 33:] = np.frombuffer(y.to_bytes(32, "big"), dtype=np.uint8)
    return rows


def host_multiple_of_g(curve: Curve, scalar: int) -> tuple[int, int]:
    """`scalar·G` as affine Python integers — the signing path's readback."""
    ((x, y),) = affine_ints(curve, multiple(curve, [scalar], curve.generator))
    return x, y


# What `affine_ints` reads the group identity back as, named because a caller
# working in affine integers has to compare against it. No real point on these
# curves occupies it — see `affine_ints` for why.
AFFINE_IDENTITY = (0, 0)


def affine_ints(curve: Curve, points: ArrayLike) -> list[tuple[int, int]]:
    """Any point batch back to affine `(x, y)` Python integers.

    The coordinates come out as base-field values rather than off the point's
    `raw`, because `raw` hands back the storage: for a Montgomery point type
    that is `x · R`, not `x`, and the difference is invisible to anything that
    only round-trips. Viewing the affine pair as its two field elements and
    converting those is the module's `astype`-not-bitcast rule — the view
    reinterprets the coordinate layout, the conversion reads the residue.

    The identity reads back as `(0, 0)`, which no real point on these curves
    occupies (`x = 0` would need `b` to be a residue *and* `y = 0` needs
    2-torsion a prime-order group cannot have) — callers reject it before
    encoding.
    """
    affine = np.asarray(points).astype(curve.point)
    coords = affine.reshape(-1).view(curve.field).astype(object)
    return [(int(coords[2 * i]), int(coords[2 * i + 1])) for i in range(affine.size)]


def sum_points(curve: Curve, points: ArrayLike) -> np.ndarray:
    """The sum of a `[K]` point batch — `group.sum_points` with this curve's
    identity as the pad, which on a short Weierstrass curve is a zero-filled
    Jacobian buffer (see the shared function, which makes the choice the
    caller's because padding wrongly agrees rather than raises).
    """
    points = np.asarray(points)
    return group.sum_points(points, np.zeros([1], dtype=points.dtype))


def is_identity(curve: Curve, points: ArrayLike) -> np.ndarray:
    """Whether each entry is the group identity, elementwise.

    Answered on the host even when the batch is on a device, and the reason is
    a live upstream bug rather than a preference.

    The comparison has to be on the affine form — a projective one has no
    unique representation, and the traced path refuses equality on it outright
    — but at the pinned wheel the jacobian-to-affine conversion is wrong on the
    **CPU** backend, where it returns the identity for every input
    (fractalyze/xla#594). It is correct on CUDA and on the host, so converting
    here would answer "every point is the identity" on exactly the leg the
    merge gate runs, and would do it silently.

    The pull-back costs the caller nothing it was not already paying: every
    call site reads the same batch back through `affine_ints` on the next line,
    and frx caches the materialized buffer, so this is one transfer rather than
    two.
    """
    points = np.asarray(points)
    return points == np.zeros(points.shape, dtype=points.dtype)


def _weierstrass_rhs(curve: Curve, x_field: np.ndarray) -> np.ndarray:
    """The curve equation's right side, `x³ + ax + b`, over the base field."""
    return (x_field * x_field + curve.coeff_a) * x_field + curve.coeff_b


def on_curve(curve: Curve, x: int, y: int) -> bool:
    """Whether integer coordinates satisfy `y² = x³ + ax + b`, in the field.

    Coordinates outside `[0, p)` are not a point encoding at all (SEC 1
    §2.3.4 checks the range before the equation), so they answer `False`
    here — which also keeps them away from the field constructor (the
    second dtype gotcha above).
    """
    if not (0 <= x < curve.p and 0 <= y < curve.p):
        return False
    x_field = np.array(x, dtype=curve.field)
    y_field = np.array(y, dtype=curve.field)
    return bool(y_field * y_field == _weierstrass_rhs(curve, x_field))


def on_curve_rows(curve: Curve, xs: Sequence[int], ys: Sequence[int]) -> np.ndarray:
    """`on_curve` over a whole batch of integer coordinates: `bool[B]`.

    Same answer as calling `on_curve` per row, and the reason it exists is
    that the per-row form's cost is not the curve equation. `on_curve` builds
    a *0-d field array per coordinate*, so a batch pays `2B` dtype
    constructions and `4B` scalar field ops to evaluate what is one
    expression over `[B]`. Measured at B=1024 on an RTX 5090, the row-at-a-time
    form costs 9.3x this one — a **stage ratio**, the arithmetic being the same
    and the overhead all that leaves. What that stage is worth to a verification
    is `Ecdsa.verify_digest`'s to say, and it says it: 19% of the call before,
    2% after.

    The range check runs first and is load-bearing beyond the standard's
    reason for it. SEC 1 §2.3.4 checks `[0, p)` before the equation because
    an out-of-range pair is not a point encoding; here it also keeps such a
    coordinate away from the field constructor, which **aborts** rather than
    reducing (fractalyze/zk_dtypes#179). A rejected row therefore rides a
    zero coordinate through the equation, exactly as a masked row rides zero
    scalars in `Ecdsa._masked_quotient_pair`, and its verdict is already
    sealed by the same mask.

    Coordinates arrive as Python integers rather than bytes because that is
    what the callers have already parsed, and because `group.field_from_bytes`
    is the wrong tool for this: per coordinate it costs 12x what
    `int.from_bytes` plus one `[B]` construction does, since it evaluates a
    32-wide weighted sum in field arithmetic where the host has a C path over
    32 bytes.
    """
    in_range = np.array(
        [0 <= x < curve.p and 0 <= y < curve.p for x, y in zip(xs, ys)],
        dtype=bool,
    )
    if not in_range.any():
        return in_range
    x_field = np.array(
        [x if ok else 0 for x, ok in zip(xs, in_range)], dtype=curve.field
    )
    y_field = np.array(
        [y if ok else 0 for y, ok in zip(ys, in_range)], dtype=curve.field
    )
    satisfied = np.asarray(y_field * y_field == _weierstrass_rhs(curve, x_field))
    return in_range & satisfied


@functools.cache
def _fused_sqrt(curve: Curve) -> Any:
    """`sqrt`'s ladder compiled into one kernel, built once per curve.

    Cached on the curve so a process traces each exponent once; `frx.jit`
    caches per argument shape underneath, which is the axis the compile cost
    actually varies over.
    """
    exponent = (curve.p + 1) // 4
    return frx.jit(lambda value: group.pow_const(curve, value, exponent))


def sqrt(curve: Curve, value: ArrayLike) -> Any:
    """A square-root candidate in the base field, for `p ≡ 3 (mod 4)`.

    `value^((p+1)/4)` — Tonelli's shortcut; both curves qualify. For a
    non-residue the result is a root of `-value`, so membership is decided
    by squaring, which is what `lift_x_to_parity` does.

    A placed batch runs the ladder compiled, and that is the one seam in this
    module where `jit` earns its place. The point seams are single fused ops,
    for which compiling measured identical to running them eagerly. This is
    ~325 *chained* multiplications, so eager it is a chain of that many
    sequential kernel launches — launch-bound rather than compute-bound, which
    shows up as a steady state that does not move with the batch: it holds to
    1.3% across B = 24, 64, 256 and 1024 on secp256k1, a 43-fold range of batch
    sizes — that is the spread across the four, not a mean over them. Compiled
    it is one kernel, also flat, so the win is ~19x wherever the batch is
    placed. Both curves land there (secp256k1 18.7-19.0x, secp256r1 18.9-19.6x
    for B >= 24); an earlier reading that made them differ was five reps against
    this one's twenty-one, and the difference did not survive.

    **The cost is a compile per batch shape, and it is not small.** Measured on
    an RTX 5090 it is ~440-750x one eager call the first time a process sees a
    shape, and a small fraction of that once a persistent compile cache holds
    it — so break-even moves from a few hundred calls at a shape to single
    figures. A deployment that verifies more than a handful of batches wants
    `FRX_COMPILATION_CACHE_DIR` set; one that verifies a single signature and
    exits pays the whole compile to save one call's work and would rather not.
    That trade is stated here because it is invisible to every benchmark in this
    repo: they all warm each shape before timing it, so none of them can see a
    first-call cost at all.

    Shrinking the ladder does not help the compile, which is the
    counter-intuitive part and worth recording so it is not retried. Compile
    time tracks *live values*, not operation count — holding the multiplies at
    328 and varying only how many stay live moved the compile 12x between one
    live value and 64 — and the window's `2^w - 1` table is what is live here.
    So `window=1` emits ~499 multiplies with no table, `window=4` emits ~328
    with a 15-entry table — and the fewer-multiplication form is the ~1.4x
    slower compile. `window=4` stays because it wins the steady state by 1.6x,
    which is what is paid on every call rather than once. The structural fix is
    outlining, which is the same finding as fractalyze/prime-ir#405 and belongs
    there.
    """
    if curve.p % 4 != 3:
        raise ValueError("sqrt shortcut requires p ≡ 3 (mod 4)")
    if traced(value):
        return _fused_sqrt(curve)(value)
    return group.pow_const(curve, value, (curve.p + 1) // 4)


def lift_x_to_parity(
    curve: Curve, xs: list[int], parities: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """The points at each `x` whose y-representative has the asked parity.

    `(points, ok)`: affine `[B]` and membership `[B]`. Every encoding that
    names a point by x plus one bit — a recovery id, a compressed key, an
    implicit-even convention — is this one operation. Where `x` is on no
    point the row carries junk coordinates the caller's mask drops; `xs`
    arrive as integers already reduced below `p` (the callers' encodings
    check that bound where the byte order still exists).

    The square root is placed on the same threshold the point seams use — it
    is the third seam that lifts, and the module docstring argues why. Only
    the ladder moves: the parity choice needs a residue's low bit and the
    point construction is per entry, both host by nature, so the coordinates
    come back either way. `_place` sees a field batch here and asks the field
    question, which is what keeps P-256 — whose points a traced array cannot
    hold, but whose field it can — on the device path anyway.
    """
    x_field = _place(np.array(xs, dtype=curve.field))
    rhs = _weierstrass_rhs(curve, x_field)
    root = sqrt(curve, rhs)
    ok = np.asarray(root * root == rhs)
    roots = [int(v) for v in np.asarray(root).astype(object)]
    points = np.array(
        [
            curve.point((x, y if y % 2 == int(want) % 2 else curve.p - y))
            for x, y, want in zip(xs, roots, parities)
        ],
        dtype=curve.point,
    )
    return points, ok


def secret_scalar(curve: Curve, data: ArrayLike, role: str) -> tuple[np.ndarray, int]:
    """A 32-byte big-endian secret encoding as `(bytes, scalar)`.

    Refused outside `[1, n-1]` rather than reduced — reduction would
    silently map two encodings to one key (SEC 1 §3.2.1's validity range).
    """
    raw = np.asarray(data, dtype=np.uint8).reshape(-1)
    if raw.shape[0] != 32:
        raise ValueError(f"a {role} is 32 bytes")
    scalar = int.from_bytes(raw.tobytes(), "big")
    if not 1 <= scalar <= curve.n - 1:
        raise ValueError(f"the {role} scalar is outside [1, n-1]")
    return raw, scalar
