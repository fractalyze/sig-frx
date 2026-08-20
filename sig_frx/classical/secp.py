# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The SEC curves as curated zk_dtypes point types, plus what those omit.

The group law, the scalar multiplication, and the point representations live
in zk_dtypes — the `secp256k1_g1_*` / `secp256r1_g1_*` dtypes carry `+`, `-`,
and point × scalar as ufuncs over fused C++ kernels — so nothing in this repo
implements curve arithmetic for these curves anymore. What this module keeps
is exactly what the dtypes do not expose:

- the integer view of the curves' constants — derived from the dtypes'
  own metadata (`ecinfo`/`pfinfo`), not transcribed — because the schemes'
  scalar work runs on Python integers (exact, host-only per
  `docs/reference/security.md`);
- the curve-equation membership check — the dtypes construct off-curve
  coordinates without complaint, and SEC 1's encoding rules reject them;
- the square-root lift that names a point by x plus a parity bit, which
  every recovery id, compressed key, and x-only key performs;
- the byte/int ↔ point codecs the wire encodings need.

Everything here is host-path: `.raw` readback and per-entry construction are
host operations. The GPU story for these curves is EC kernels over these
same dtypes (the decision is recorded on fractalyze/sig-frx#139), not a
traced re-derivation of the group law — which is why no namespace dispatch
appears anywhere in this module.

## Two dtype gotchas the codecs absorb

A point dtype's scalar branch turns a bare integer into `k·G`, so a
coordinate pair must arrive as one tuple argument (`point((x, y))`), never
as a row of ints. And constructing a scalar-field value from an integer in
`[n, 2²⁵⁶)` aborts instead of reducing, so every scalar is reduced `% n` in
Python before it becomes a dtype value — sound, since `k·P = (k mod n)·P`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import numpy as np
import zk_dtypes
from frx.typing import ArrayLike

from sig_frx.classical import group


@dataclass(frozen=True)
class Curve:
    """A short-Weierstrass curve: its integers, and its curated dtypes.

    Frozen with value equality so a scheme holding one rides pytree aux
    without re-tracing per instance (`signature.py`'s rule) — the dtype
    handles are classes, identical across instances of the same curve.
    """

    p: int  # base field modulus
    a: int  # curve coefficient a
    b: int  # curve coefficient b
    n: int  # order of the base point, prime
    gx: int  # base point, affine x
    gy: int  # base point, affine y
    point: Any  # the affine G1 dtype, standard domain
    accumulator: Any  # the jacobian G1 dtype — what sums and multiples ride
    scalar: Any  # the scalar-field dtype
    field: Any  # the base-field dtype — what the lift's arithmetic runs over

    @classmethod
    def from_dtypes(
        cls, point: Any, accumulator: Any, scalar: Any, field: Any
    ) -> Curve:
        """A Curve whose integers come from the dtypes' own metadata.

        The pinned wheel is the single source of truth for the constants:
        `ecinfo` carries the equation and the base point, `pfinfo` the two
        moduli (`testing/secp_test.py` holds them against SEC 2's published
        parameters). The base field arrives as an argument rather than being
        derived — `ecinfo(...).base_field_dtype` returns the scalar field
        (fractalyze/zk_dtypes#182), so deriving it would pair the wrong
        modulus.
        """
        ec = zk_dtypes.ecinfo(point)
        return cls(
            p=zk_dtypes.pfinfo(field).modulus,
            a=int(ec.a),
            b=int(ec.b),
            n=zk_dtypes.pfinfo(scalar).modulus,
            gx=int(ec.gx),
            gy=int(ec.gy),
            point=point,
            accumulator=accumulator,
            scalar=scalar,
            field=field,
        )

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


# SEC 2 §2.4.1, "Recommended Parameters secp256k1". The Koblitz curve.
SECP256K1 = Curve.from_dtypes(
    point=zk_dtypes.secp256k1_g1_affine,
    accumulator=zk_dtypes.secp256k1_g1_jacobian,
    scalar=zk_dtypes.secp256k1_sf,
    field=zk_dtypes.secp256k1_bf,
)

# SEC 2 §2.4.2, "Recommended Parameters secp256r1" — NIST's P-256
# (FIPS 186-5 §6.1.1 points at SP 800-186 §3.2.1.3 for the same values).
SECP256R1 = Curve.from_dtypes(
    point=zk_dtypes.secp256r1_g1_affine,
    accumulator=zk_dtypes.secp256r1_g1_jacobian,
    scalar=zk_dtypes.secp256r1_sf,
    field=zk_dtypes.secp256r1_bf,
)


def multiple(curve: Curve, scalars: list[int], points: ArrayLike) -> np.ndarray:
    """`scalars[i] · points[i]`, one batched kernel call — `[B]` jacobian.

    Scalars reduce `% n` in Python first (the dtype gotcha above); the
    reduction is the group's own fact, `k·P = (k mod n)·P`.
    """
    reduced = np.array([k % curve.n for k in scalars], dtype=curve.scalar)
    return points * reduced


def host_multiple_of_g(curve: Curve, scalar: int) -> tuple[int, int]:
    """`scalar·G` as affine Python integers — the signing path's readback."""
    ((x, y),) = affine_ints(curve, multiple(curve, [scalar], curve.generator))
    return x, y


def affine_ints(curve: Curve, points: ArrayLike) -> list[tuple[int, int]]:
    """Any point batch back to affine `(x, y)` Python integers.

    The identity reads back as `(0, 0)`, which no real point on these curves
    occupies (`x = 0` would need `b` to be a residue *and* `y = 0` needs
    2-torsion a prime-order group cannot have) — callers reject it before
    encoding.
    """
    converted = np.asarray(points).astype(curve.point).astype(object)
    return [entry.raw for entry in converted]


def is_identity(curve: Curve, points: ArrayLike) -> np.ndarray:
    """Whether each entry is the group identity, elementwise."""
    points = np.asarray(points)
    return points == np.zeros(points.shape, dtype=points.dtype)


def on_curve(curve: Curve, x: int, y: int) -> bool:
    """Whether integer coordinates satisfy `y² = x³ + ax + b (mod p)`."""
    return (y * y - (x * x * x + curve.a * x + curve.b)) % curve.p == 0


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
    rhs = (x_field * x_field + curve.coeff_a) * x_field + curve.coeff_b
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
