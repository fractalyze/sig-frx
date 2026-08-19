# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Short-Weierstrass curve arithmetic for the classical schemes.

The curve substrate ECDSA needs, over fields minted from the curve's own moduli:
`zk_dtypes.prime_field(p)` reduces internally, so `+`, `-`, `*` and `/` on curve
coordinates are already modular and nothing here implements them — the same
delegation `lattice/mldsa/arith.py` leans on, at 256 bits instead of 23. No
curated curve dtype ships for these curves today (zk-dtypes 0.0.14 carries
bn254, pallas and vesta only), so the group law lives here, over the field
dtype; if a curated dtype with fused kernels arrives, this module is the one
seam to swap.

## Complete formulas, because verification is traced

The group law is Renes-Costello-Batina's complete projective addition
(https://eprint.iacr.org/2015/1060, Algorithms 1 and 3), not the affine
textbook rule. The affine rule is five cases — SEC 1 §2.2.1 writes them out —
and a traced batch cannot branch per entry, so every case would become a
`where` over every other case's undefined arithmetic (a division by zero where
x1 = x2). The complete formulas are one arithmetic path that is correct for
*all* inputs on a prime-order curve — identity, doubling, inverses included —
which is exactly the shape a batched trace wants. The cost is a handful of
extra multiplications per addition, which is #36's business, not correctness's.

Points ride as `(X : Y : Z)` homogeneous projective triples of field elements,
identity `(0 : 1 : 0)`. The affine rule these formulas must agree with is
transcribed from SEC 1 §2.2.1 into `testing/sec1_reference.py`, and the tests
hold the two together over every case class the completeness claim covers.

## Coordinates keep a batch axis, even at `B = 1`

numpy collapses a product of two 0-d arrays of an extension dtype to the
dtype's scalar object, whose arithmetic against arrays is partial — so a
0-d-shaped point stops composing halfway through a ladder. Every function here
therefore expects coordinates with at least one axis, which costs nothing: the
seam's own rule is that a single verification is `B = 1`, not a scalar, and a
host signing path holds one-element arrays the same way.

## One implementation, both namespaces

Everything here is dtype arithmetic, so the same code runs concretely on numpy
(key generation, signing) and under a tracer (verification) — except the scalar
ladder, whose *loop* has to come from somewhere: a Python loop on the host, and
`lax.fori_loop` under a tracer, where unrolling 256 complete additions would
compile a graph nobody can afford. The select inside the ladder is arithmetic
(`bit·added + (1-bit)·doubled`) rather than a `where`, so the carried state
never leaves the field dtype.
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


class Point(NamedTuple):
    """A projective point `(X : Y : Z)`, coordinates in the curve's base field.

    A `NamedTuple` so it is a pytree as-is and crosses `jit` / `fori_loop`
    boundaries without registration. Batched points carry the batch on each
    coordinate's leading axes.
    """

    x: Any
    y: Any
    z: Any


@dataclass(frozen=True)
class Curve:
    """A short-Weierstrass curve `y² = x³ + ax + b` over `F_p`, order-`n` group.

    Prime order (cofactor 1) is a requirement, not a convention: the complete
    formulas below are complete exactly for prime-order short-Weierstrass
    groups. Both curves this repo ships satisfy it (SEC 2 §2.4.1, §2.4.2).

    Frozen with value equality so a scheme holding one can ride pytree aux
    without re-tracing per instance (`signature.py`'s rule).
    """

    p: int  # base field modulus
    a: int  # curve coefficient a
    b: int  # curve coefficient b
    n: int  # order of the base point, prime
    gx: int  # base point, affine x
    gy: int  # base point, affine y

    @functools.cached_property
    def field(self) -> Any:
        """The base field `F_p`, as a dtype."""
        return zk_dtypes.prime_field(self.p, storage="std")

    @functools.cached_property
    def scalar_field(self) -> Any:
        """The scalar field `F_n`, as a dtype."""
        return zk_dtypes.prime_field(self.n, storage="std")

    @functools.cached_property
    def coeff_a(self) -> np.ndarray:
        """`a` as a field scalar, the shape the formulas consume."""
        return np.array(self.a % self.p, dtype=self.field)

    @functools.cached_property
    def coeff_b3(self) -> np.ndarray:
        """`3·b` as a field scalar — the constant the complete formulas take."""
        return np.array(3 * self.b % self.p, dtype=self.field)

    @functools.cached_property
    def one(self) -> np.ndarray:
        return np.array(1, dtype=self.field)

    @functools.cached_property
    def generator(self) -> Point:
        """The base point `G`, projective, shaped `[1]`.

        One-element rather than 0-d for the module's batch-axis rule: a 0-d
        operand meeting another 0-d inside the formulas collapses to the
        dtype's scalar object, whose arithmetic against arrays is partial.
        `[1]` broadcasts against any batch instead.
        """
        return Point(
            np.array([self.gx], dtype=self.field),
            np.array([self.gy], dtype=self.field),
            np.array([1], dtype=self.field),
        )

    @functools.cached_property
    def byte_weights(self) -> np.ndarray:
        """`256^(31-i) mod p` as field scalars — what `field_from_bytes` sums by."""
        return np.array([pow(256, 31 - i, self.p) for i in range(32)], dtype=self.field)


# SEC 2 §2.4.1, "Recommended Parameters secp256k1". The Koblitz curve.
SECP256K1 = Curve(
    p=0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFE_FFFFFC2F,
    a=0,
    b=7,
    n=0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFE_BAAEDCE6_AF48A03B_BFD25E8C_D0364141,
    gx=0x79BE667E_F9DCBBAC_55A06295_CE870B07_029BFCDB_2DCE28D9_59F2815B_16F81798,
    gy=0x483ADA77_26A3C465_5DA4FBFC_0E1108A8_FD17B448_A6855419_9C47D08F_FB10D4B8,
)

# SEC 2 §2.4.2, "Recommended Parameters secp256r1" — NIST's P-256
# (FIPS 186-5 §6.1.1 points at SP 800-186 §3.2.1.3 for the same values).
SECP256R1 = Curve(
    p=0xFFFFFFFF_00000001_00000000_00000000_00000000_FFFFFFFF_FFFFFFFF_FFFFFFFF,
    a=0xFFFFFFFF_00000001_00000000_00000000_00000000_FFFFFFFF_FFFFFFFF_FFFFFFFC,
    b=0x5AC635D8_AA3A93E7_B3EBBD55_769886BC_651D06B0_CC53B0F6_3BCE3C3E_27D2604B,
    n=0xFFFFFFFF_00000000_FFFFFFFF_FFFFFFFF_BCE6FAAD_A7179E84_F3B9CAC2_FC632551,
    gx=0x6B17D1F2_E12C4247_F8BCE6E5_63A440F2_77037D81_2DEB33A0_F4A13945_D898C296,
    gy=0x4FE342E2_FE1A7F9B_8EE7EB4A_7C0F9E16_2BCE3357_6B315ECE_CBB64068_37BF51F5,
)


def identity(curve: Curve, like: ArrayLike) -> Point:
    """The point at infinity `(0 : 1 : 0)`, broadcast to `like`'s shape.

    `like` is a field-typed array (a coordinate of some point in the
    computation); multiplying by zero is what carries its shape and namespace
    into the result without naming either.
    """
    zero = like * np.array(0, dtype=curve.field)
    return Point(zero, zero + curve.one, zero)


def from_affine(curve: Curve, x: ArrayLike, y: ArrayLike) -> Point:
    """Affine `(x, y)` as a projective point, `Z = 1`."""
    return Point(x, y, x * np.array(0, dtype=curve.field) + curve.one)


def is_identity(curve: Curve, point: Point) -> Any:
    """Whether each entry is the point at infinity: `Z = 0`, elementwise."""
    return point.z == np.array(0, dtype=curve.field)


def add(curve: Curve, p: Point, q: Point) -> Point:
    """`P + Q` — RCB Algorithm 1, complete for prime-order curves.

    Transcribed from https://eprint.iacr.org/2015/1060, Algorithm 1 (general
    `a`), steps folded only where the paper's own temporaries allow. Complete:
    correct for every input pair, including `P = Q`, `P = -Q`, and either
    operand at infinity — no case analysis rides on top.
    """
    a, b3 = curve.coeff_a, curve.coeff_b3
    x1, y1, z1 = p
    x2, y2, z2 = q
    t0 = x1 * x2  # 1
    t1 = y1 * y2  # 2
    t2 = z1 * z2  # 3
    t3 = (x1 + y1) * (x2 + y2) - (t0 + t1)  # 4-8
    t4 = (x1 + z1) * (x2 + z2) - (t0 + t2)  # 9-13
    t5 = (y1 + z1) * (y2 + z2) - (t1 + t2)  # 14-18
    z3 = a * t4 + b3 * t2  # 19-21
    x3 = t1 - z3  # 22
    z3 = t1 + z3  # 23
    y3 = x3 * z3  # 24
    t1 = t0 + t0 + t0  # 25-26
    t2 = a * t2  # 27
    t4 = b3 * t4  # 28
    t1 = t1 + t2  # 29
    t2 = a * (t0 - t2)  # 30-31
    t4 = t4 + t2  # 32
    y3 = y3 + t1 * t4  # 33-34
    x3 = t3 * x3 - t5 * t4  # 35-37
    z3 = t5 * z3 + t3 * t1  # 38-40
    return Point(x3, y3, z3)


def double(curve: Curve, p: Point) -> Point:
    """`2P` — RCB Algorithm 3, exception-free for prime-order curves.

    Transcribed from https://eprint.iacr.org/2015/1060, Algorithm 3 (general
    `a`). `add(curve, p, p)` computes the same value — completeness covers
    doubling — but the ladder doubles every step, and the dedicated form is
    what the paper provides for exactly that.
    """
    a, b3 = curve.coeff_a, curve.coeff_b3
    x, y, z = p
    t0 = x * x  # 1
    t1 = y * y  # 2
    t2 = z * z  # 3
    t3 = x * y  # 4
    t3 = t3 + t3  # 5
    z3 = x * z  # 6
    z3 = z3 + z3  # 7
    x3 = a * z3  # 8
    y3 = b3 * t2 + x3  # 9-10
    x3 = t1 - y3  # 11
    y3 = t1 + y3  # 12
    y3 = x3 * y3  # 13
    x3 = t3 * x3  # 14
    z3 = b3 * z3  # 15
    t2 = a * t2  # 16
    t3 = a * (t0 - t2)  # 17-18
    t3 = t3 + z3  # 19
    z3 = t0 + t0  # 20
    t0 = (z3 + t0 + t2) * t3  # 21-23
    y3 = y3 + t0  # 24
    t2 = y * z  # 25
    t2 = t2 + t2  # 26
    x3 = x3 - t2 * t3  # 27-28
    z3 = t2 * t1  # 29
    z3 = z3 + z3  # 30
    z3 = z3 + z3  # 31
    return Point(x3, y3, z3)


def negate(point: Point) -> Point:
    """`-P = (X : -Y : Z)` — SEC 1 §2.2.1's rule 3, projectively."""
    return Point(point.x, -point.y, point.z)


def bits_of(data: ArrayLike) -> Any:
    """Big-endian bytes as bits, most significant first: `[..., L] -> [..., 8L]`.

    The shape the scalar ladder consumes, produced from wire bytes — a scalar's
    bits come off its encoding, never off a field element, because a 256-bit
    residue has no integer lane to be read back onto.
    """
    xnp = namespace(data)
    data = xnp.asarray(data)
    shifts = np.arange(7, -1, -1, dtype=np.uint8)
    bits = (data[..., :, None] >> shifts) & np.uint8(1)
    return bits.reshape(data.shape[:-1] + (8 * data.shape[-1],))


def _select(curve: Curve, flag: Any, when_set: Point, when_clear: Point) -> Point:
    """`flag ? when_set : when_clear`, arithmetically, per batch entry.

    `flag` is a field element in {0, 1}. Arithmetic rather than `where` so the
    ladder's carried state is field-dtype arithmetic end to end.
    """
    keep = curve.one - flag
    return Point(
        flag * when_set.x + keep * when_clear.x,
        flag * when_set.y + keep * when_clear.y,
        flag * when_set.z + keep * when_clear.z,
    )


def scalar_mul(curve: Curve, bits: ArrayLike, point: Point) -> Point:
    """`k·P` over `k`'s bits, most significant first: `[..., L]` bits by a point.

    Double-and-add with a fixed trip count `L` and an arithmetic select, so one
    traced computation serves the whole batch (SEC 1 §2.2 names the algorithm;
    the fixed shape is this repo's). A scalar wider than the group order is
    reduced by the group itself — `k·P = (k mod n)·P` — which is why callers
    hand bits straight off wire bytes without a mod-n in front.

    The loop is `lax.fori_loop` under a tracer and a Python loop on the host:
    same body, and the host path never lifts (`conventions.md`).
    """
    xnp = namespace(bits, *point)
    bits = xnp.asarray(bits)
    flags = bits.astype(np.int32).astype(curve.field)
    length = flags.shape[-1]

    def step(flag: Any, acc: Point) -> Point:
        doubled = double(curve, acc)
        added = add(curve, doubled, point)
        return _select(curve, flag, added, doubled)

    start = identity(curve, flags[..., 0] * point.x)
    if xnp is np:
        acc = start
        for i in range(length):
            acc = step(flags[..., i], acc)
        return acc
    return lax.fori_loop(
        0, length, lambda i, acc: step(xnp.take(flags, i, axis=-1), acc), start
    )


def to_affine(curve: Curve, point: Point) -> tuple[Any, Any]:
    """`(X/Z, Y/Z)` — affine coordinates, defined only away from infinity.

    The field dtype divides (Montgomery inverse), so this is two divisions and
    no Fermat ladder. At `Z = 0` the quotient is whatever the dtype makes of a
    zero divisor; a caller that may hold the identity checks `is_identity`
    first, which is what verification does anyway — an identity result is a
    reject, not a coordinate.
    """
    return point.x / point.z, point.y / point.z


def on_curve(curve: Curve, x: ArrayLike, y: ArrayLike) -> Any:
    """Whether affine `(x, y)` satisfies `y² = x³ + ax + b`, elementwise."""
    return y * y == (x * x + curve.coeff_a) * x + np.array(
        curve.b % curve.p, dtype=curve.field
    )


def equal(p: Point, q: Point) -> Any:
    """Whether two projective points name the same group element, elementwise.

    Cross-multiplied, so no division: `(X₁ : Y₁ : Z₁) = (X₂ : Y₂ : Z₂)` iff
    `X₁Z₂ = X₂Z₁` and `Y₁Z₂ = Y₂Z₁`. Sound for everything the complete
    formulas produce — they never emit the degenerate all-zero triple.
    """
    return (p.x * q.z == q.x * p.z) & (p.y * q.z == q.y * p.z)


def field_from_bytes(curve: Curve, data: ArrayLike) -> Any:
    """Big-endian `[..., 32]` bytes as base-field elements, reduced mod `p`.

    The reduction is the field's, which means a value at or above `p` wraps
    silently — a caller enforcing an encoding's range bound (SEC 1 §2.3.6
    rejects coordinates outside `[0, p-1]`) checks the bytes before, where the
    order still exists.
    """
    xnp = namespace(data)
    lanes = xnp.asarray(data).astype(np.int32).astype(curve.field)
    return (lanes * curve.byte_weights).sum(axis=-1)


def pow_const(curve: Curve, base: ArrayLike, exponent: int) -> Any:
    """`base^exponent` in the base field, for a static exponent.

    Square-and-multiply with the branches decided at trace time — the exponent
    is a Python integer, so the unrolled ~256 squarings compile once per
    exponent and the value path stays pure field arithmetic.
    """
    acc = base * np.array(0, dtype=curve.field) + curve.one
    for bit in bin(exponent)[2:]:
        acc = acc * acc
        if bit == "1":
            acc = acc * base
    return acc


def sqrt(curve: Curve, value: ArrayLike) -> Any:
    """A square root candidate in `F_p`, for `p ≡ 3 (mod 4)`.

    `value^((p+1)/4)` (Tonelli's shortcut for this residue class — both curves
    here qualify). For a non-residue the result is a root of `-value`, so a
    caller decides membership by squaring: `sqrt(v)² == v`.
    """
    if curve.p % 4 != 3:
        raise ValueError("sqrt shortcut requires p ≡ 3 (mod 4)")
    return pow_const(curve, value, (curve.p + 1) // 4)
