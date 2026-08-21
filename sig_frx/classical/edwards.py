# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Ed25519's curve as curated zk_dtypes point types, plus what those omit.

The group law, the scalar multiplication, and the point representations live
in zk_dtypes — the `ed25519_g1_*` dtypes carry `+`, `-`, and point × scalar
as ufuncs over fused C++ kernels, in the extended homogeneous
`(X : Y : Z : T)` representation RFC 8032 §5.1.4 defines its formulas over —
so nothing in this repo implements Edwards curve arithmetic anymore. What
this module keeps is exactly what the dtypes do not expose:

- the pairing that names which dtypes form one curve, with the integer view
  of its constants derived from the dtypes' own metadata (`ecinfo`/`pfinfo`);
- RFC 8032 §5.1.3's decoding — the square-root x-recovery stays field
  algebra over `prime_field` here, per the recorded contract that zk_dtypes
  exposes no lift/decompression — and §5.1.2's encoding;
- the canonical-parity read those encodings hang on;
- the host-integer scalar handling, including the widening to the full
  group order that keeps unreduced verification scalars exact
  (`wide_multiple` below).

Everything here is host-path, like `secp.py`: `.raw` readback and per-entry
construction are host operations, and the GPU story for this curve is EC
kernels over these same dtypes (fractalyze/sig-frx#36), not a traced
re-derivation of the group law.

## Canonical storage is load-bearing

The one thing an Edwards *encoding* needs that field algebra cannot give is
the sign of `x` — the parity of the canonical residue (RFC 8032 §5.1.2). With
canonical (`std`) storage the parity is a bitcast of the lowest lane; with
Montgomery storage that read is exactly the mistake `lattice/mldsa/arith.py`
warns about. So the decoding field is minted canonical *on purpose*, and
`parity` is the only place in the classical substrate allowed to read
storage as bits.

## Reduction is modulo 8L, not L

The curve's cofactor is 8, so the full group has order `8·L` and
`k·P = (k mod 8L)·P` for *every* point, torsion components included —
while reducing modulo `L` alone is a fact about the prime-order subgroup
only. `multiple` reduces `% L` and therefore serves scalars already below
`L` and multiples of the base point; `wide_multiple` is exact for
everything else, which is what lets the verifier keep driving the group
with its unreduced digest scalar (`eddsa/ed25519.py`).
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
class EdwardsCurve:
    """A twisted Edwards curve `-x² + y² = 1 + d·x²y²`: four dtype handles,
    everything else derived.

    The integers come off the dtypes' own metadata — `ecinfo` carries the
    equation and the base point, `pfinfo` the two moduli — so the pinned
    wheel is the single source of truth (`testing/eddsa_test.py` holds the
    derivation against RFC 8032 §5.1's published values). `a = -1` is a
    property of the family the dtypes implement, not a field here.

    Frozen with value equality for the seam's pytree-aux rule — the dtype
    handles are classes, identical across instances of the same curve.
    """

    point: Any  # the affine G1 dtype, standard domain
    accumulator: Any  # the extended G1 dtype — what sums and multiples ride
    scalar: Any  # the scalar-field dtype, modulo L
    field: Any  # the base-field dtype — what decoding's arithmetic runs over

    # RFC 8032 §5.1: the group has order 8·L. The cofactor is what
    # `wide_multiple`'s exactness stands on.
    cofactor = 8

    @functools.cached_property
    def p(self) -> int:
        """Base field modulus."""
        return zk_dtypes.pfinfo(self.field).modulus

    @functools.cached_property
    def order(self) -> int:
        """Order of the base point (RFC 8032's L), prime."""
        return zk_dtypes.pfinfo(self.scalar).modulus

    @functools.cached_property
    def d(self) -> int:
        """The curve's d coefficient."""
        return int(zk_dtypes.ecinfo(self.point).d)

    @functools.cached_property
    def gx(self) -> int:
        """Base point, affine x."""
        return int(zk_dtypes.ecinfo(self.point).gx)

    @functools.cached_property
    def gy(self) -> int:
        """Base point, affine y."""
        return int(zk_dtypes.ecinfo(self.point).gy)

    @functools.cached_property
    def d_field(self) -> np.ndarray:
        """`d` as a field scalar — what §5.1.3's decoding evaluates with."""
        return np.array(self.d % self.p, dtype=self.field)

    @functools.cached_property
    def one(self) -> np.ndarray:
        return np.array(1, dtype=self.field)

    @functools.cached_property
    def sqrt_minus_one(self) -> np.ndarray:
        """`2^((p-1)/4)` — the square root of -1 §5.1.3's decoding corrects by."""
        return np.array(pow(2, (self.p - 1) // 4, self.p), dtype=self.field)

    @functools.cached_property
    def generator(self) -> np.ndarray:
        """The base point `B` as a `[1]`-shaped affine array."""
        return np.array([self.point((self.gx, self.gy))], dtype=self.point)

    @functools.cached_property
    def identity(self) -> np.ndarray:
        """The neutral element as a `[1]`-shaped affine array.

        `(0, 1)` is a point on this curve like any other — the Weierstrass
        sentinel's trick of reading the all-zero buffer as infinity has no
        Edwards analogue, and an all-zero extended point is worse than
        wrong: its projective compare answers `True` against *every* point.
        """
        return np.array([self.point((0, 1))], dtype=self.point)

    @functools.cached_property
    def le_byte_weights(self) -> np.ndarray:
        """`256^i mod p` — RFC 8032 encodings are little-endian."""
        return np.array([pow(256, i, self.p) for i in range(32)], dtype=self.field)


# RFC 8032 §5.1: edwards25519. The integers derive from the dtype metadata;
# the test pins d to the decimal the §5.1 table prints and the base point to
# RFC 7748's, so a slip in the wheel fails against the standard.
ED25519 = EdwardsCurve(
    point=zk_dtypes.ed25519_g1_affine,
    accumulator=zk_dtypes.ed25519_g1_extended,
    scalar=zk_dtypes.curve25519_sf,
    field=zk_dtypes.curve25519_bf,
)


def multiple(curve: EdwardsCurve, scalars: list[int], points: ArrayLike) -> np.ndarray:
    """`scalars[i] · points[i]`, one batched kernel call — `[B]` extended.

    Scalars reduce `% L` in Python first (the dtype refuses larger ints,
    fractalyze/zk_dtypes#179) — sound when the scalar is already below `L`
    or the point lies in the base point's prime-order subgroup. A wide
    scalar on an arbitrary decoded point needs `wide_multiple`: with
    cofactor 8, `% L` is not the group's own reduction.
    """
    reduced = np.array([k % curve.order for k in scalars], dtype=curve.scalar)
    return points * reduced


def wide_multiple(
    curve: EdwardsCurve, scalars: list[int], points: ArrayLike
) -> np.ndarray:
    """`scalars[i] · points[i]` exact for any width and any curve point.

    Reduces modulo the full group order `8L` — the group's own fact for
    every point, torsion included — then splits on the cofactor rather than
    on `L`: `k = 8q + s` gives `q ≤ (8L-1)/8 < L` and `s < 8` by
    construction, so both scalars are ones `multiple` takes unchanged, and
    `k·P = q·(8P) + s·P`. Doubling three times to reach `8P` also clears
    the torsion component, which is what makes the `q` term immune to the
    reduction `multiple` applies. Two batched multiplications, against the
    three that splitting on `L` costs (that form has to synthesize `L·P`,
    a full-width multiply whose only job is to be discarded).
    """
    points = np.asarray(points)
    reduced = [k % (curve.cofactor * curve.order) for k in scalars]
    torsion_free = points
    for _ in range(curve.cofactor.bit_length() - 1):  # the cofactor is 2³
        torsion_free = torsion_free + torsion_free
    return multiple(
        curve, [k // curve.cofactor for k in reduced], torsion_free
    ) + multiple(curve, [k % curve.cofactor for k in reduced], points)


def affine_ints(curve: EdwardsCurve, points: ArrayLike) -> list[tuple[int, int]]:
    """Any point batch back to affine `(x, y)` Python integers."""
    converted = np.asarray(points).astype(curve.point).astype(object)
    return [entry.raw for entry in converted]


def is_identity(curve: EdwardsCurve, points: ArrayLike) -> np.ndarray:
    """Whether each entry is the group identity, elementwise.

    A dtype compare, so it costs no readback and reads either point
    representation — the affine identity and its extended form compare
    equal, the projective compare being what decides.
    """
    return np.asarray(points) == curve.identity


def field_from_le_bytes(curve: EdwardsCurve, data: ArrayLike) -> Any:
    """Little-endian `[..., 32]` bytes as field elements, reduced mod `p`.

    `group.field_from_bytes` over this curve's little-endian weights — RFC
    8032 encodings are little-endian — with the range caveat there (§5.1.3
    fails a `y ≥ p`, and that check reads the bytes).
    """
    return group.field_from_bytes(curve.le_byte_weights, data)


def parity(curve: EdwardsCurve, value: ArrayLike) -> Any:
    """The low bit of the canonical residue — RFC 8032's sign of `x`.

    The one deliberate storage read in the substrate: canonical storage means
    the buffer's lowest little-endian lane carries the residue's low bits, so
    the parity is a bitcast and a mask. Montgomery storage would make this
    read a different number, which is why the curve mints canonical fields.
    """
    lanes = np.asarray(value).view(np.uint32).reshape(np.shape(value) + (8,))
    return lanes[..., 0] & np.uint32(1)


def decompress_x(curve: EdwardsCurve, y: Any, sign: Any) -> tuple[Any, Any]:
    """RFC 8032 §5.1.3's x-recovery: `(x, decoded_ok)`, elementwise.

    `x² = (y² - 1)/(d·y² + 1)`; the candidate root is
    `u·v³·(u·v⁷)^((p-5)/8)`, corrected by `√-1` when it squares to `-u/v`,
    and decoding fails when neither squares back — or when `x = 0` arrives
    with the sign bit set. A failed entry carries a junk `x` and a false
    flag rather than raising, which is what lets one decoding serve a whole
    batch: the caller's mask drops the row.
    """
    y2 = y * y
    u = y2 - curve.one
    v = curve.d_field * y2 + curve.one
    v3 = v * v * v
    v7 = v3 * v3 * v
    candidate = u * v3 * group.pow_const(curve, u * v7, (curve.p - 5) // 8)
    squared = v * candidate * candidate
    is_root = squared == u
    corrected = candidate * curve.sqrt_minus_one
    needs_correction = squared == -u
    ok = is_root | needs_correction
    x = np.where(is_root, candidate, corrected)
    x_is_zero = x == np.array(0, dtype=curve.field)
    sign_arr = np.asarray(sign)
    ok = ok & ~(x_is_zero & (sign_arr == np.uint8(1)))
    wrong_sign = parity(curve, x) != sign_arr.astype(np.uint32)
    return np.where(wrong_sign, -x, x), ok


def decode(curve: EdwardsCurve, encoded: ArrayLike) -> tuple[np.ndarray, Any]:
    """RFC 8032 §5.1.3: `[B, 32]` bytes to affine points, a verdict each.

    The sign bit is the top bit of the last byte; the remaining 255 bits are
    `y`, refused when not canonical (`y ≥ p`). The recovered coordinates read
    back to integers and construct the affine dtype per row — host codec, per
    the module's host-path rule — so a failed entry carries junk coordinates
    (the dtype constructs off-curve pairs without complaint) and a false
    verdict its caller's mask drops.
    """
    encoded = np.asarray(encoded)
    sign = encoded[..., 31] >> np.uint8(7)
    y_bytes = np.concatenate(
        [encoded[..., :31], encoded[..., 31:32] & np.uint8(0x7F)], axis=-1
    )
    canonical = group.bytes_below(y_bytes, curve.p, byteorder="little")
    y = field_from_le_bytes(curve, y_bytes)
    x, ok = decompress_x(curve, y, sign)
    xs = np.asarray(x).astype(object)
    ys = np.asarray(y).astype(object)
    points = np.array(
        [curve.point((int(a), int(b))) for a, b in zip(xs, ys)], dtype=curve.point
    )
    return points, canonical & ok


def encode_affine(x: int, y: int) -> bytes:
    """RFC 8032 §5.1.2: `y` little-endian with `x`'s parity as the top bit."""
    return (y | ((x & 1) << 255)).to_bytes(32, "little")
