# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""FIPS 204's encodings (§7.1, §7.2): bit packing, the hint, and the wire formats.

Everything a key or a signature is on the wire. No cryptography happens here —
it is bit manipulation over byte strings — which is why it is separable and why
round-trip plus malformed-input tests cover nearly all of it.

## It is traced, because a verifier decodes

`pkDecode` and `sigDecode` run on input a verifier did not produce, and
verification is the batch path — `B` signatures decode in one call. So these are
array operations over `uint8`, static-shaped, with no Python loop over a batch
axis. Encoding runs on the host side of key generation and signing and works
there unchanged, since a bit shift does not care which namespace it is in.

## The rejection has somewhere to go

`HintBitUnpack` is specified to reject malformed input, and FIPS 204 §D.2 says
what skipping it costs: the *draft* standard omitted this very check, and
"failure to perform this check results in a signature scheme that is not
strongly existentially unforgeable". A published standard got this wrong once.

`hint_bit_unpack` therefore returns `(h, ok)` and a caller ANDs `ok` into its
verdict — not a flag invented here but Algorithm 8 line 3, `if h = ⊥ then
return false`, with the branch removed.

**Why this reports where `rejection.require_enough` raises**, when both cite
the same `bool[B]`: they answer different questions. A malformed hint is a
*verdict about the input*, and `bool[B]` is exactly where the standard puts it.
A sampler shortfall is a claim about *our own budget* — the input is fine — so
folding it into `bool[B]` would report a valid signature as invalid, which is
worse than the raise. The channel is the same one; it only accepts verdicts.

The three conditions Algorithm 21 rejects on, each of which is a real encoding
a real attacker can send:

- a cut index that goes backwards or past `omega` — the per-polynomial ends must
  be non-decreasing and in range,
- positions inside one polynomial that are not strictly increasing — which is
  what makes the encoding canonical, since the same hint would otherwise have
  many valid encodings,
- a nonzero byte in the unused tail — the same canonicality requirement, at the
  end of the buffer.

## The bit-field pair is one mechanism

`SimpleBitPack`/`BitPack` differ only by the subtraction `BitPack` applies
(`b − w_i`), and their inverses only by adding it back. So the packing itself is
`pack_fields` / `unpack_fields`, and the four named algorithms are those two
plus a constant. Bit fields tile a byte string exactly — `32·c` bytes is
`256·c` bits — so both directions are a reshape and a weighted sum, with no
byte-window arithmetic and no traced index anywhere.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from frx.typing import ArrayLike

from sig_frx.arrays import namespace
from sig_frx.lattice.mldsa.arith import D, N, Q

# `bitlen(q-1) - d`, the width `t1`'s coefficients pack at (Algorithm 22).
T1_BITS = (Q - 1).bit_length() - D


def bytes_to_bits(values: ArrayLike) -> Any:
    """Algorithm 13 — each byte low bit first, over the trailing axis."""
    xnp = namespace(values)
    data = xnp.asarray(values)
    bits = (data[..., None] >> np.arange(8, dtype=np.uint8)) & np.uint8(1)
    return bits.reshape(*data.shape[:-1], -1)


def bits_to_bytes(bits: ArrayLike) -> Any:
    """Algorithm 12 — eight bits to a byte, low bit first."""
    xnp = namespace(bits)
    grouped = xnp.asarray(bits, dtype=np.uint8).reshape(*np.shape(bits)[:-1], -1, 8)
    return (grouped << np.arange(8, dtype=np.uint8)).sum(axis=-1).astype(np.uint8)


def unpack_fields(v: ArrayLike, width: int) -> Any:
    """`v` read as little-endian `width`-bit fields, as uint32.

    The unsigned core Algorithms 18 and 19 share; each adds its own constant.

    **`wots.base_2b` is this at the opposite bit order** and neither can stand
    in for the other: FIPS 205 numbers a digit's bits from the high end of the
    stream, FIPS 204 from the low end, so they agree only at whole-byte widths
    and disagree at every width either scheme uses. That is the near-miss
    [`bytestring.low_bits`](../../hash/bytestring.py) already records
    against `base_2b`, one level further out — and a "unification" of the two
    round-trips forever while being wrong.

    The sum's dtype is pinned rather than inferred: numpy promotes it to
    `uint64` and frx does not, so leaving it open makes the host and traced
    paths differ in a width — the divergence `CLAUDE.md`'s lane rule exists
    for, and one no round trip can see.
    """
    fields = bytes_to_bits(v).astype(np.uint32).reshape(*np.shape(v)[:-1], -1, width)
    return (fields << np.arange(width, dtype=np.uint32)).sum(axis=-1, dtype=np.uint32)


def pack_fields(values: ArrayLike, width: int) -> Any:
    """The inverse of `unpack_fields`: `width`-bit fields to a byte string."""
    xnp = namespace(values)
    data = xnp.asarray(values, dtype=np.uint32)
    bits = (data[..., None] >> np.arange(width, dtype=np.uint32)) & np.uint32(1)
    return bits_to_bytes(bits.reshape(*data.shape[:-1], -1))


def simple_bit_pack(w: ArrayLike, b: int) -> Any:
    """Algorithm 16 — coefficients in `[0, b]` to `32·bitlen(b)` bytes."""
    return pack_fields(w, b.bit_length())


def simple_bit_unpack(v: ArrayLike, b: int) -> Any:
    """Algorithm 18 — the inverse, as int32."""
    return unpack_fields(v, b.bit_length()).astype(np.int32)


def bit_pack(w: ArrayLike, a: int, b: int) -> Any:
    """Algorithm 17 — coefficients in `[−a, b]` to `32·bitlen(a+b)` bytes."""
    xnp = namespace(w)
    return pack_fields(
        np.int32(b) - xnp.asarray(w, dtype=np.int32), (a + b).bit_length()
    )


def bit_unpack(v: ArrayLike, a: int, b: int) -> Any:
    """Algorithm 19 — the inverse, as int32 in `[b − 2^c + 1, b]`."""
    return np.int32(b) - unpack_fields(v, (a + b).bit_length()).astype(np.int32)


def hint_bit_pack(h: ArrayLike, omega: int) -> Any:
    """Algorithm 20 — `h`'s nonzero positions as `omega + k` bytes.

    `h` is `[k, 256]`. The standard writes an index that advances as it walks;
    here the same index is a running count, so the write position of each
    nonzero coefficient is `cumsum(h) − 1` over the flattened vector and the
    per-polynomial cut is the running count at each polynomial's end.
    """
    xnp = namespace(h)
    k = np.shape(h)[-2]
    flags = (xnp.asarray(h) != 0).reshape(-1)
    ranks = xnp.cumsum(flags.astype(np.int32), axis=-1)
    cuts = ranks.reshape(k, N)[:, -1]

    # Slot `t` holds the coefficient index whose rank is `t + 1` — a masked sum
    # rather than a scatter, one column per slot. `index` stays a host constant:
    # `where` broadcasts it, and lifting it would put it on a device the signing
    # path never wanted to be on.
    slot = np.arange(omega, dtype=np.int32)
    index = np.tile(np.arange(N, dtype=np.int32), k)
    chosen = flags[:, None] & (ranks[:, None] == slot + np.int32(1))
    positions = xnp.where(chosen, index[:, None], np.int32(0)).sum(axis=0)
    return xnp.concatenate([positions, cuts]).astype(np.uint8)


def hint_bit_unpack(y: ArrayLike, k: int, omega: int) -> tuple[Any, Any]:
    """Algorithm 21 — `(h, ok)`, where `ok` is the standard's `⊥` inverted.

    `[omega + k]` bytes to `h` as `[k, 256]` int32 flags. Every one of
    Algorithm 21's three rejections is evaluated over the whole buffer rather
    than short-circuited, because a tracer has no branch to take and because a
    verifier must not have a data-dependent exit here anyway.

    The membership test broadcasts to `[k, omega, 256]`, which is the module's
    largest intermediate: all of `sig_decode`'s 10 MB of scratch at ML-DSA-87
    and `B = 64`, scaling linearly to 164 MB at `B = 1024`. Factoring the
    one-hot out of the `k` axis as an f32 matmul halves that memory and is not
    taken, because it buys a fraction of a percent of a verify in exchange for
    float arithmetic in a bit-manipulation module. **The number to watch is the
    memory, not the time** — the memory is shape-derived and checkable from the
    shapes above; the time is not.

    The three timing figures this paragraph used to carry — an 86% share of
    `sig_decode`, a 1.6x for the matmul form, and `sig_decode` at 0.8% of
    `expand_a` — are **withdrawn pending a re-measurement**, not disproved.
    They predate the rule that a share and its total come from one session
    (../../../docs/reference/measurement.md) and nothing records whether they
    did. The decision above does not rest on their exact values: it rests on
    `sig_decode` being a small part of a verify next to `expand_a`, which the
    memory figures and the shapes already carry.

    `h` is returned even when `ok` is false. It is meaningless in that case, and
    a caller that ignores `ok` has accepted a forged hint — which is why `ok` is
    a return value rather than something to consult.
    """
    xnp = namespace(y)
    data = xnp.asarray(y, dtype=np.uint8).astype(np.int32)
    positions, cuts = data[:omega], data[omega:]
    starts = xnp.concatenate([np.zeros(1, dtype=np.int32), cuts[:-1]])
    slot = np.arange(omega, dtype=np.int32)

    # Line 4: the cuts advance and stay in range.
    ok = (cuts >= starts).all() & (cuts <= np.int32(omega)).all()

    covered = (slot[None, :] >= starts[:, None]) & (slot[None, :] < cuts[:, None])
    # Lines 8-10: strictly increasing inside one polynomial, so an encoding of a
    # given hint is unique. `interior` excludes each polynomial's first slot,
    # which has no predecessor to compare against.
    interior = covered & (slot[None, :] > starts[:, None])
    rises = xnp.concatenate([np.zeros(1, dtype=bool), positions[1:] > positions[:-1]])
    ok = ok & (~interior | rises[None, :]).all()

    # Lines 16-18: the tail past the last cut is zero, the same uniqueness
    # requirement at the end of the buffer. Algorithm 21 line 16 starts at the
    # index its walk ended on, which is `cuts[-1]` — so the bound is a scalar
    # rather than a reduction over `covered`.
    ok = ok & ((slot < cuts[-1]) | (positions == np.int32(0))).all()

    h = (
        covered[:, :, None]
        & (positions[None, :, None] == np.arange(N, dtype=np.int32)[None, None, :])
    ).any(axis=1)
    return h.astype(np.int32), ok


def pk_encode(rho: ArrayLike, t1: ArrayLike) -> Any:
    """Algorithm 22 — `rho ‖ SimpleBitPack(t1[i], 2^(bitlen(q-1)-d) - 1)`."""
    xnp = namespace(rho, t1)
    packed = simple_bit_pack(t1, (1 << T1_BITS) - 1)
    return xnp.concatenate(
        [xnp.asarray(rho, dtype=np.uint8).reshape(-1), packed.reshape(-1)]
    )


def pk_decode(pk: ArrayLike, k: int) -> tuple[Any, Any]:
    """Algorithm 23 — `(rho, t1)`. Always in range, so there is no `⊥`."""
    xnp = namespace(pk)
    data = xnp.asarray(pk, dtype=np.uint8)
    rho, body = data[..., :32], data[..., 32:]
    t1 = simple_bit_unpack(
        body.reshape(*np.shape(body)[:-1], k, -1), (1 << T1_BITS) - 1
    )
    return rho, t1


def sk_encode(
    rho: ArrayLike,
    key: ArrayLike,
    tr: ArrayLike,
    s1: ArrayLike,
    s2: ArrayLike,
    t0: ArrayLike,
    eta: int,
) -> Any:
    """Algorithm 24 — the secret key, `s1`/`s2` at `eta` and `t0` at `2^(d-1)`."""
    xnp = namespace(rho, key, tr, s1, s2, t0)
    parts = [
        xnp.asarray(rho, dtype=np.uint8).reshape(-1),
        xnp.asarray(key, dtype=np.uint8).reshape(-1),
        xnp.asarray(tr, dtype=np.uint8).reshape(-1),
        bit_pack(s1, eta, eta).reshape(-1),
        bit_pack(s2, eta, eta).reshape(-1),
        bit_pack(t0, (1 << (D - 1)) - 1, 1 << (D - 1)).reshape(-1),
    ]
    return xnp.concatenate(parts)


def sk_decode(sk: ArrayLike, k: int, ell: int, eta: int) -> tuple[Any, ...]:
    """Algorithm 25 — the inverse.

    The standard warns that a malformed secret key decodes to `s1`/`s2` outside
    `[−eta, eta]` and that this is only for trusted input, so there is no `⊥`
    here and none is invented: a caller holding an untrusted secret key has a
    different problem than a range check.
    """
    xnp = namespace(sk)
    data = xnp.asarray(sk, dtype=np.uint8)
    low = 1 << (D - 1)
    packed = 32 * (2 * eta).bit_length()
    s1_end = 128 + packed * ell
    s2_end = s1_end + packed * k
    return (
        data[:32],
        data[32:64],
        data[64:128],
        bit_unpack(data[128:s1_end].reshape(ell, -1), eta, eta),
        bit_unpack(data[s1_end:s2_end].reshape(k, -1), eta, eta),
        bit_unpack(data[s2_end:].reshape(k, -1), low - 1, low),
    )


def sig_encode(
    c_tilde: ArrayLike, z: ArrayLike, h: ArrayLike, gamma1: int, omega: int
) -> Any:
    """Algorithm 26 — `c̃ ‖ BitPack(z[i], γ1-1, γ1) ‖ HintBitPack(h)`."""
    xnp = namespace(c_tilde, z, h)
    return xnp.concatenate(
        [
            xnp.asarray(c_tilde, dtype=np.uint8).reshape(-1),
            bit_pack(z, gamma1 - 1, gamma1).reshape(-1),
            hint_bit_pack(h, omega).reshape(-1),
        ]
    )


def sig_decode(
    sigma: ArrayLike, lam: int, ell: int, k: int, gamma1: int, omega: int
) -> tuple[Any, Any, Any, Any]:
    """Algorithm 27 — `(c̃, z, h, ok)`, with `ok` false where the standard says `⊥`.

    `z` is always in range because `γ1` is a power of two, so the only rejection
    is the hint's — and it is the caller's to consume, not this function's to
    raise on.
    """
    xnp = namespace(sigma)
    data = xnp.asarray(sigma, dtype=np.uint8)
    c_size = lam // 4
    z_bits = 1 + (gamma1 - 1).bit_length()
    z_size = 32 * z_bits * ell
    c_tilde = data[:c_size]
    z = bit_unpack(data[c_size : c_size + z_size].reshape(ell, -1), gamma1 - 1, gamma1)
    h, ok = hint_bit_unpack(data[c_size + z_size :], k, omega)
    return c_tilde, z, h, ok


def w1_encode(w1: ArrayLike, gamma2: int) -> Any:
    """Algorithm 28 — `w1` packed at `bitlen((q-1)/(2γ2) - 1)`, for `H`."""
    return simple_bit_pack(w1, (Q - 1) // (2 * gamma2) - 1).reshape(-1)
