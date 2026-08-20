# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Twisted Edwards curve arithmetic for Ed25519 (RFC 8032).

The same substrate contract as `weierstrass.py`, for the curve family EdDSA
lives on: fields minted from the modulus, complete formulas because
verification is traced, one implementation for both namespaces, and
coordinates that keep a batch axis even at `B = 1`. What is different is the
group law's source — RFC 8032 §5.1.4 publishes the extended-homogeneous
formulas itself, complete for `a = -1` and non-square `d`, so the
transcription and the standard are one document here.

## Canonical storage is load-bearing

The one thing an Edwards *encoding* needs that field algebra cannot give is
the sign of `x` — the parity of the canonical residue (RFC 8032 §5.1.2). With
canonical (`std`) storage the parity is a bitcast of the lowest lane; with
Montgomery storage that read is exactly the mistake `lattice/mldsa/arith.py`
warns about. So the field is minted canonical *on purpose*, and `parity` is
the only place in the classical substrate allowed to read storage as bits.

Points ride as extended homogeneous `(X : Y : Z : T)` with `x = X/Z`,
`y = Y/Z`, `xy = T/Z` — the representation §5.1.4 defines its formulas over.
The identity is `(0 : Z : Z : 0)`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import zk_dtypes
from frx import lax
from frx.typing import ArrayLike

from sig_frx.arrays import namespace
from sig_frx.classical import group


class ExtPoint(NamedTuple):
    """An extended-homogeneous twisted-Edwards point `(X : Y : Z : T)`."""

    x: Any
    y: Any
    z: Any
    t: Any


@dataclass(frozen=True)
class EdwardsCurve:
    """A twisted Edwards curve `-x² + y² = 1 + d·x²y²` over `F_p`, order-`L`
    prime subgroup.

    `a = -1` is fixed rather than a field: RFC 8032 §5.1.4's complete formulas
    are for exactly that case, and edwards25519 is the one instance this repo
    ships. Frozen with value equality for the seam's pytree-aux rule.
    """

    p: int  # base field modulus
    d: int  # the curve's d coefficient
    order: int  # order of the base point (RFC 8032's L), prime
    gx: int  # base point, affine x
    gy: int  # base point, affine y

    @functools.cached_property
    def field(self) -> Any:
        """The base field, canonical storage — see the module docstring."""
        return zk_dtypes.prime_field(self.p, storage="std")

    @functools.cached_property
    def d_field(self) -> np.ndarray:
        """`d` as a field scalar — what §5.1.3's decoding evaluates with."""
        return np.array(self.d % self.p, dtype=self.field)

    @functools.cached_property
    def coeff_2d(self) -> np.ndarray:
        """`2d` as a field scalar — the constant §5.1.4's addition takes."""
        return np.array(2 * self.d % self.p, dtype=self.field)

    @functools.cached_property
    def one(self) -> np.ndarray:
        return np.array(1, dtype=self.field)

    @functools.cached_property
    def sqrt_minus_one(self) -> np.ndarray:
        """`2^((p-1)/4)` — the square root of -1 §5.1.3's decoding corrects by."""
        return np.array(pow(2, (self.p - 1) // 4, self.p), dtype=self.field)

    @functools.cached_property
    def generator(self) -> ExtPoint:
        """The base point `B`, extended coordinates, shaped `[1]`."""
        return ExtPoint(
            np.array([self.gx], dtype=self.field),
            np.array([self.gy], dtype=self.field),
            np.array([1], dtype=self.field),
            np.array([self.gx * self.gy % self.p], dtype=self.field),
        )

    @functools.cached_property
    def le_byte_weights(self) -> np.ndarray:
        """`256^i mod p` — RFC 8032 encodings are little-endian."""
        return np.array([pow(256, i, self.p) for i in range(32)], dtype=self.field)


# RFC 8032 §5.1: edwards25519. `d` is the table's -121665/121666, computed
# from the fraction the standard defines it by; the test pins the quotient to
# the decimal the same table prints, so a slip in either fails against the
# other. The base point is RFC 7748's, transcribed from the §5.1 table.
ED25519 = EdwardsCurve(
    p=2**255 - 19,
    d=-121665 * pow(121666, -1, 2**255 - 19) % (2**255 - 19),
    order=2**252 + 27742317777372353535851937790883648493,
    gx=15112221349535400772501151409588531511454012693041857206046113283949847762202,
    gy=46316835694926478169428394003475163141307993866256225615783033603165251855960,
)


def identity(curve: EdwardsCurve, like: ArrayLike) -> ExtPoint:
    """The neutral point `(0 : 1 : 1 : 0)`, broadcast to `like`'s shape."""
    zero = like * np.array(0, dtype=curve.field)
    return ExtPoint(zero, zero + curve.one, zero + curve.one, zero)


def from_affine(curve: EdwardsCurve, x: ArrayLike, y: ArrayLike) -> ExtPoint:
    """Affine `(x, y)` as an extended point: `Z = 1`, `T = xy`."""
    one = x * np.array(0, dtype=curve.field) + curve.one
    return ExtPoint(x, y, one, x * y)


def add(curve: EdwardsCurve, p: ExtPoint, q: ExtPoint) -> ExtPoint:
    """`P + Q` — RFC 8032 §5.1.4's complete addition, transcribed."""
    a = (p.y - p.x) * (q.y - q.x)
    b = (p.y + p.x) * (q.y + q.x)
    c = p.t * curve.coeff_2d * q.t
    d = (p.z + p.z) * q.z
    e = b - a
    f = d - c
    g = d + c
    h = b + a
    return ExtPoint(e * f, g * h, f * g, e * h)


def double(curve: EdwardsCurve, p: ExtPoint) -> ExtPoint:
    """`2P` — RFC 8032 §5.1.4's doubling, transcribed."""
    a = p.x * p.x
    b = p.y * p.y
    c = p.z * p.z
    c = c + c
    h = a + b
    e = h - (p.x + p.y) * (p.x + p.y)
    g = a - b
    f = c + g
    return ExtPoint(e * f, g * h, f * g, e * h)


def scalar_mul(curve: EdwardsCurve, bits: ArrayLike, point: ExtPoint) -> ExtPoint:
    """`k·P` over `k`'s bits, most significant first — the shared ladder.

    `group.ladder` bound to this family's complete formulas. A scalar wider
    than `L` reduces through the group itself, which is what lets a 512-bit
    SHA-512 digest drive the ladder straight off its bytes with no arithmetic
    modulo `L` on the device.
    """
    return group.ladder(curve, bits, point, double=double, add=add, identity=identity)


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
    xnp = namespace(value)
    if xnp is np:
        lanes = np.asarray(value).view(np.uint32).reshape(np.shape(value) + (8,))
        return lanes[..., 0] & np.uint32(1)
    return lax.bitcast_convert_type(value, np.uint32)[..., 0] & np.uint32(1)


def decompress_x(curve: EdwardsCurve, y: Any, sign: Any) -> tuple[Any, Any]:
    """RFC 8032 §5.1.3's x-recovery: `(x, decoded_ok)`, elementwise.

    `x² = (y² - 1)/(d·y² + 1)`; the candidate root is
    `u·v³·(u·v⁷)^((p-5)/8)`, corrected by `√-1` when it squares to `-u/v`,
    and decoding fails when neither squares back — or when `x = 0` arrives
    with the sign bit set. All of it stays arithmetic: failed entries carry a
    junk `x` and a false flag, never a branch.
    """
    u = y * y - curve.one
    v = curve.d_field * (y * y) + curve.one
    v3 = v * v * v
    v7 = v3 * v3 * v
    candidate = u * v3 * group.pow_const(curve, u * v7, (curve.p - 5) // 8)
    squared = v * candidate * candidate
    is_root = squared == u
    corrected = candidate * curve.sqrt_minus_one
    needs_correction = squared == -u
    ok = is_root | needs_correction
    flag = _as_field_flag(curve, is_root)
    x = flag * candidate + (curve.one - flag) * corrected
    x_is_zero = x == np.array(0, dtype=curve.field)
    xnp = namespace(y)
    ok = ok & ~(x_is_zero & (xnp.asarray(sign) == np.uint8(1)))
    wrong_sign = parity(curve, x) != xnp.asarray(sign).astype(np.uint32)
    sign_flag = _as_field_flag(curve, wrong_sign)
    x = sign_flag * (-x) + (curve.one - sign_flag) * x
    return x, ok


def _as_field_flag(curve: EdwardsCurve, mask: Any) -> Any:
    """A boolean mask as a {0, 1} field element, for arithmetic selection."""
    xnp = namespace(mask)
    return xnp.asarray(mask).astype(np.int32).astype(curve.field)


def decode(curve: EdwardsCurve, encoded: ArrayLike) -> tuple[ExtPoint, Any]:
    """RFC 8032 §5.1.3: `[..., 32]` bytes to points, with a verdict each.

    The sign bit is the top bit of the last byte; the remaining 255 bits are
    `y`, refused when not canonical (`y ≥ p`). A failed entry carries junk
    coordinates and a false verdict — arithmetic, never a branch — which is
    what lets one decoding serve a traced batch and a concrete host caller
    alike.
    """
    xnp = namespace(encoded)
    encoded = xnp.asarray(encoded)
    sign = encoded[..., 31] >> np.uint8(7)
    y_bytes = xnp.concatenate(
        [encoded[..., :31], encoded[..., 31:32] & np.uint8(0x7F)], axis=-1
    )
    canonical = group.bytes_below(xnp, y_bytes, curve.p, byteorder="little")
    y = field_from_le_bytes(curve, y_bytes)
    x, ok = decompress_x(curve, y, sign)
    return from_affine(curve, x, y), canonical & ok


def encode_affine(x: int, y: int) -> bytes:
    """RFC 8032 §5.1.2: `y` little-endian with `x`'s parity as the top bit."""
    return (y | ((x & 1) << 255)).to_bytes(32, "little")
