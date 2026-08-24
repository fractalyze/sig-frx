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

## Which side of the namespace each function is on

The dtype ufuncs — `multiple`, `double_multiple`, and the compare behind
`is_identity` — follow the namespace their arguments arrive in, so a caller
holding a device batch gets the arithmetic there and one holding a host batch
does not. This module never chooses for its caller
(`docs/reference/conventions.md`). It is worth choosing: measured on an RTX
5090 with full-size scalars, the device form of `multiple` runs 7.3x the host
at B=256 and 105x at B=4096, where it still costs the same ~5 ms it does at
B=64.

Everything that turns a point into integers is host by nature and stays there —
`affine_ints`, `uncompressed_rows`, `lift_x_to_parity`, `on_curve`,
`secret_scalar` — because the standards define those on integers and Python
has no width. `is_identity` is the one exception on the wrong side of that
line, and its docstring carries the reason (fractalyze/xla#594).

There is no `jit` here and none would help: each of these is a single fused
op, so compiling one measured identical to running it eagerly while adding a
compile per batch shape. The GPU story for these curves remains EC kernels
over these same dtypes (fractalyze/sig-frx#139), not a traced re-derivation of
the group law.

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
from dataclasses import dataclass
from typing import Any

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
        """Whether a traced array can hold this curve's points at all.

        A point type needs a row in frx's admission table, and secp256r1 has
        none at the pinned wheel while secp256k1 does. Probed rather than
        listed: a list would be a second place to update and would go stale
        silently the moment the rows land, where this starts answering `True`
        on its own.

        Any failure answers `False`, not only the `TypeError` seen today. The
        fallback is the host path, which is always correct, so a probe that
        guesses wrong costs speed — while one that let a new exception type
        through would restore the crash this exists to stop, at the batch
        sizes no gate covers.
        """
        try:
            fnp.asarray(self.generator)
        except Exception:  # noqa: BLE001 — see above; the fallback is correct.
            return False
        return True


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


# Where a verification batch stops being cheaper on the host — see `place`
# for the measurement and why one number covers both backends.
DEVICE_MIN_BATCH = 64


def place(curve: Curve, points: ArrayLike) -> Any:
    """A verification batch moved off the host once the batch pays for it.

    The substrate itself never chooses a namespace — `multiple` and the rest
    read it off their arguments, and this repo's rule is that the lift belongs
    to the caller (`docs/reference/conventions.md`). This is that caller's
    decision written once instead of five times, and a scheme opts into it by
    calling it; nothing here applies it on anyone's behalf.

    The decision is a batch-size threshold because the cost it is trading
    against is a fixed one. Measured on an RTX 5090 with full-size scalars,
    lifting a batch, multiplying and reading it back costs about 5 ms on CUDA
    regardless of size, so it loses badly to the host on a single signature
    (0.12 ms against 3.2 ms) and wins from roughly a batch of 64 up — 1.7x
    there, 26x at 1 024. The CPU backend has no such floor and is ahead from a
    batch of 2, so one threshold picked for CUDA is safe for both: below it
    nothing moves and nothing regresses, above it both backends gain.

    A curve whose points a traced array cannot hold stays on the host at every
    size. That is not a tuning decision: lifting secp256r1 raises rather than
    running slowly, so a batch of P-256 signatures large enough to cross the
    threshold would fail outright — which the KAT gate cannot see, because its
    batches are smaller than that.

    A batch that is already traced is left alone — the caller has placed it
    and this is not the function that second-guesses that.
    """
    if traced(points):
        return points
    host = np.asarray(points)
    if host.shape[0] < DEVICE_MIN_BATCH or not curve.traceable:
        return host
    return fnp.asarray(host)


def multiple(curve: Curve, scalars: list[int], points: ArrayLike) -> np.ndarray:
    """`scalars[i] · points[i]`, one batched kernel call — `[B]` jacobian.

    Scalars reduce `% n` in Python first (the dtype gotcha above); the
    reduction is the group's own fact, `k·P = (k mod n)·P`.

    The scalars are built in the namespace `points` arrived in, so a caller
    that put its batch on the device gets the multiplication there and one
    that did not keeps it on the host. This function does not choose — the
    lift is the caller's (`docs/reference/conventions.md`).

    Measured on an RTX 5090 with full-size scalars, the device form runs 7.3x
    the host at B=256 and 105x at B=4096, where it still costs the same ~5 ms
    it does at B=64. Those are this call's numbers, not a lane's: the readback
    that follows it at every call site is host work either way, and moving
    that is what `affine_ints` is waiting on.
    """
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
    """
    xnp = namespace(points)
    return multiple(curve, g_scalars, xnp.asarray(curve.generator)) + multiple(
        curve, point_scalars, xnp.asarray(points)
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


def sqrt(curve: Curve, value: ArrayLike) -> Any:
    """A square-root candidate in the base field, for `p ≡ 3 (mod 4)`.

    `value^((p+1)/4)` — Tonelli's shortcut; both curves qualify. For a
    non-residue the result is a root of `-value`, so membership is decided
    by squaring, which is what `lift_x_to_parity` does.
    """
    if curve.p % 4 != 3:
        raise ValueError("sqrt shortcut requires p ≡ 3 (mod 4)")
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
    """
    x_field = np.array(xs, dtype=curve.field)
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
