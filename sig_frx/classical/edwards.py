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
- the host-integer scalar handling, and the cofactor multiplication the
  verification rules are told apart by (`mul_by_cofactor` below).

Everything here is host-path: `.raw` readback and per-entry construction are
host operations, and the GPU story for this curve is EC kernels over these
same dtypes (fractalyze/sig-frx#36), not a traced re-derivation of the group
law.

`secp.py` no longer is, in one respect: its dtype ufuncs follow the namespace
their arguments arrive in. The same conversion applies here — the two
`multiple`s are the same three lines — and is left for whoever brings the
first Edwards caller that holds a device batch, rather than done on
speculation for a second substrate nobody has asked to lift.

## Canonical storage is load-bearing

The one thing an Edwards *encoding* needs that field algebra cannot give is
the sign of `x` — the parity of the canonical residue (RFC 8032 §5.1.2). With
canonical (`std`) storage the parity is a bitcast of the lowest lane; with
Montgomery storage that read is exactly the mistake `lattice/mldsa/arith.py`
warns about. So the decoding field is minted canonical *on purpose*, and
`parity` is the only place in the classical substrate allowed to read
storage as bits.

## `multiple` reduces modulo L, and that is the quantity asked for

The curve's cofactor is 8, so the full group has order `8·L` and
`k·P = (k mod 8L)·P` for *every* point, torsion components included, while
reducing modulo `L` alone is a fact about the prime-order subgroup. So
`multiple`'s `% L` is not a widening its callers have to work around: on a
point carrying a torsion component it computes `[k mod L]P`, a different
point than `[k]P`, and that is precisely what every published Ed25519
verification rule drives the group with (`eddsa/ed25519.py` records why the
standard's unreduced wording agrees).

`mul_by_cofactor` is the other half of the cofactor's story. Multiplying by
8 clears the torsion component instead of reducing around it, which is what
ZIP-215's equation does to both sides and what makes `is_small_order` a
compare rather than a blocklist.
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
    # `mul_by_cofactor` clears the torsion with.
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
    """`(scalars[i] mod L) · points[i]`, one batched kernel call — `[B]`
    extended.

    Scalars reduce `% L` in Python first, because the dtype refuses a wider
    integer (fractalyze/zk_dtypes#179). The reduction is exact on the base
    point's prime-order subgroup and a deliberate reading everywhere else —
    see the module docstring, and `eddsa/ed25519.py` for the standard it is
    read from.
    """
    reduced = np.array([k % curve.order for k in scalars], dtype=curve.scalar)
    return points * reduced


def mul_by_cofactor(curve: EdwardsCurve, points: ArrayLike) -> np.ndarray:
    """`[8]P` elementwise, as three doublings rather than a scalar multiply.

    Clearing the torsion component is what ZIP-215's equation does to both
    sides, and `[8]P == O` is what makes a small-order point a compare. The
    cofactor is `2³`, so the whole of it is three batched adds — cheaper
    than a scalar kernel whose scalar is a constant 8.
    """
    doubled = np.asarray(points)
    for _ in range(curve.cofactor.bit_length() - 1):  # the cofactor is 2³
        doubled = doubled + doubled
    return doubled


def sum_points(curve: EdwardsCurve, points: ArrayLike) -> np.ndarray:
    """The sum of a `[K]` point batch — `group.sum_points` with this curve's
    identity as the pad, which for an Edwards curve is a real `(0, 1)` and
    never a zero-filled buffer (see `identity`, and the shared function).
    """
    return group.sum_points(np.asarray(points), curve.identity)


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


def is_small_order(curve: EdwardsCurve, points: ArrayLike) -> np.ndarray:
    """Whether each entry's order divides the cofactor, elementwise.

    `[8]P == O` is the definition, so the identity counts as small-order —
    which is what the rules that reject these points intend. RFC 8032 asks
    for no such check; ed25519-dalek's `verify_strict` and libsodium do
    ([`eddsa/consensus.py`](eddsa/consensus.py)).

    A compare, where those implementations ship a blocklist of encodings.
    The blocklist has to enumerate every encoding of every small-order
    point, which is a list that can be — and in libsodium 1.0.15 was —
    incomplete; `[8]P` cannot miss one.
    """
    return is_identity(curve, mul_by_cofactor(curve, points))


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
    """RFC 8032 §5.1.3's x-recovery: `(x, on_curve)`, elementwise.

    `x² = (y² - 1)/(d·y² + 1)`; the candidate root is
    `u·v³·(u·v⁷)^((p-5)/8)`, corrected by `√-1` when it squares to `-u/v`,
    and the flag is false when neither squares back — that `y` names no
    point. §5.1.3's *canonicity* refusals are not applied here: which of
    them a rule takes is `decode`'s question, and it asks it in one place.

    A failed entry carries a junk `x` and a false flag rather than raising,
    which is what lets one decoding serve a whole batch: the caller's mask
    drops the row.
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
    sign_arr = np.asarray(sign)
    wrong_sign = parity(curve, x) != sign_arr.astype(np.uint32)
    return np.where(wrong_sign, -x, x), ok


def decode(
    curve: EdwardsCurve, encoded: ArrayLike, *, canonical_only: bool
) -> tuple[np.ndarray, Any]:
    """RFC 8032 §5.1.3: `[B, 32]` bytes to affine points, a verdict each.

    The sign bit is the top bit of the last byte and the remaining 255 bits
    are `y`. `canonical_only` carries §5.1.3's two canonicity refusals
    together — a `y ≥ p`, and an `x = 0` that arrives with the sign bit set,
    an encoding of `-0`. Dropping them reads `y` modulo `p` and lets `-0`
    decode to `0`, which is what ZIP-215 means by requiring only that the
    bytes encode *a point on the curve*.

    One parameter rather than two because the rules this repo implements
    take both refusals or neither. It has **no default**: which accept set a
    caller wants is the consensus-relevant choice this whole module exists
    to make explicit, so a call site states it and cites the rule it is
    reading (`eddsa/consensus.py`).

    The recovered coordinates read back to integers and construct the affine
    dtype per row — host codec, per the module's host-path rule — so a failed
    entry carries junk coordinates (the dtype constructs off-curve pairs
    without complaint) and a false verdict its caller's mask drops.
    """
    encoded = np.asarray(encoded)
    sign = encoded[..., 31] >> np.uint8(7)
    y_bytes = np.concatenate(
        [encoded[..., :31], encoded[..., 31:32] & np.uint8(0x7F)], axis=-1
    )
    y = field_from_le_bytes(curve, y_bytes)
    x, ok = decompress_x(curve, y, sign)
    if canonical_only:
        # `-0` negates to `0`, so the returned x answers this as well as the
        # pre-correction candidate would.
        x_is_zero = np.asarray(x) == np.array(0, dtype=curve.field)
        ok = (
            ok
            & group.bytes_below(y_bytes, curve.p, byteorder="little")
            & ~(x_is_zero & (np.asarray(sign) == np.uint8(1)))
        )
    xs = np.asarray(x).astype(object)
    ys = np.asarray(y).astype(object)
    points = np.array(
        [curve.point((int(a), int(b))) for a, b in zip(xs, ys)], dtype=curve.point
    )
    return points, ok


def encode_affine(x: int, y: int) -> bytes:
    """RFC 8032 §5.1.2: `y` little-endian with `x`'s parity as the top bit."""
    return (y | ((x & 1) << 255)).to_bytes(32, "little")
