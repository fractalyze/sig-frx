# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""FIPS 204's algorithms, transcribed the way the standard writes them.

The modules under test reshape every one of these: the transform becomes one
opcode call plus an ordering conversion, the rounding functions run over whole
arrays, a rejection loop that squeezes until enough candidates survive becomes a
fixed block schedule with a mask, and verification becomes one `vmap`ped
computation over a batch. Each of those is a change made for the compiler, and it
is the only thing about `arith.py`, `sampling.py` and `ml_dsa.py` a reader has to
take on trust — so this file takes it back, looping one coefficient at a time over
Python integers, and the tests require the two to agree
([`testing.md`](../../../../docs/reference/testing.md)).

The scheme's own three algorithms are here for the same reason and one more: the
signing loop's trip count is not observable from the outside, so the only way to
check it is to require the two loops to produce the same bytes.

Transcribed from the published document (§5, §6, §7.1–§7.5, Appendix A), not from
memory or from another implementation. Nothing here is written for speed: Python
integers have no width, which is exactly why this side is trustworthy and the
other side delegates its arithmetic to a field dtype.

The sampling half needs a SHAKE, and it takes the standard library's — which is
the point. Squeezing here is a `while` loop that asks for one more byte whenever
it needs one, so the reference never has a block bound to get wrong, and the
implementation's bound is compared against a stream that has none.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

Q = 8380417
ZETA = 1753
D = 13
N = 256

# Table 1, in full and in one place. Transcribing it per test module is how the
# copies start disagreeing about which columns they carry, and every value here
# is one the standard already determines.
PARAMETER_SETS = {
    "ML_DSA_44": {
        "k": 4,
        "ell": 4,
        "eta": 2,
        "tau": 39,
        "lam": 128,
        "gamma1": 1 << 17,
        "gamma2": (Q - 1) // 88,
        "omega": 80,
    },
    "ML_DSA_65": {
        "k": 6,
        "ell": 5,
        "eta": 4,
        "tau": 49,
        "lam": 192,
        "gamma1": 1 << 19,
        "gamma2": (Q - 1) // 32,
        "omega": 55,
    },
    "ML_DSA_87": {
        "k": 8,
        "ell": 7,
        "eta": 2,
        "tau": 60,
        "lam": 256,
        "gamma1": 1 << 19,
        "gamma2": (Q - 1) // 32,
        "omega": 75,
    },
}


def parameter_cases() -> tuple[dict, ...]:
    """`PARAMETER_SETS` as `absltest` named-parameter records."""
    return tuple(
        {"testcase_name": name, **values} for name, values in PARAMETER_SETS.items()
    )


def bit_rev8(m: int) -> int:
    """Algorithm 43."""
    bits = [(m >> i) & 1 for i in range(8)]
    return sum(bit << (7 - i) for i, bit in enumerate(bits))


def zetas() -> list[int]:
    """`zeta^BitRev8(k) mod q`, the array Appendix B tabulates."""
    return [pow(ZETA, bit_rev8(k), Q) for k in range(N)]


def mod_plus_minus(r: int, alpha: int) -> int:
    """§2.3's `mod±`: the representative in `(-alpha/2, alpha/2]`."""
    low = r % alpha
    return low - alpha if low > alpha // 2 else low


def ntt_by_definition(w: list[int]) -> list[int]:
    """Eq. (2.1): `NTT(w)[i] = w(zeta^(2·BitRev8(i)+1))`, by direct evaluation.

    §2.5 defines the transform as evaluation at 256 points; §7.5's Algorithm 41 is
    an implementation of it. Evaluating the polynomial with Horner at those points
    shares no structure with the butterfly walk, which is what makes this the check
    that pins the output *ordering*.

    A round trip and the convolution property cannot: a consistently permuted
    evaluation order round-trips exactly and multiplies correctly, because a
    pointwise product does not care what order its slots are in.
    """
    points = [pow(ZETA, 2 * bit_rev8(i) + 1, Q) for i in range(N)]
    values = []
    for point in points:
        acc = 0
        for coefficient in reversed(w):
            acc = (acc * point + coefficient) % Q
        values.append(acc)
    return values


def ntt(w: list[int]) -> list[int]:
    """Algorithm 41, line for line."""
    zeta_table = zetas()
    w_hat = list(w)
    m = 0
    length = 128
    while length >= 1:
        start = 0
        while start < 256:
            m += 1
            z = zeta_table[m]
            for j in range(start, start + length):
                t = (z * w_hat[j + length]) % Q
                w_hat[j + length] = (w_hat[j] - t) % Q
                w_hat[j] = (w_hat[j] + t) % Q
            start += 2 * length
        length //= 2
    return w_hat


def intt(w_hat: list[int]) -> list[int]:
    """Algorithm 42, line for line."""
    zeta_table = zetas()
    w = list(w_hat)
    m = 256
    length = 1
    while length < 256:
        start = 0
        while start < 256:
            m -= 1
            z = (-zeta_table[m]) % Q
            for j in range(start, start + length):
                t = w[j]
                w[j] = (t + w[j + length]) % Q
                w[j + length] = (t - w[j + length]) % Q
                w[j + length] = (z * w[j + length]) % Q
            start += 2 * length
        length *= 2
    f = 8347681
    return [(f * value) % Q for value in w]


def power2round(r: int) -> tuple[int, int]:
    """Algorithm 35."""
    r_plus = r % Q
    r0 = mod_plus_minus(r_plus, 1 << D)
    return (r_plus - r0) // (1 << D), r0


def decompose(r: int, gamma2: int) -> tuple[int, int]:
    """Algorithm 36."""
    r_plus = r % Q
    r0 = mod_plus_minus(r_plus, 2 * gamma2)
    if r_plus - r0 == Q - 1:
        return 0, r0 - 1
    return (r_plus - r0) // (2 * gamma2), r0


def high_bits(r: int, gamma2: int) -> int:
    """Algorithm 37."""
    return decompose(r, gamma2)[0]


def low_bits(r: int, gamma2: int) -> int:
    """Algorithm 38."""
    return decompose(r, gamma2)[1]


def make_hint(z: int, r: int, gamma2: int) -> bool:
    """Algorithm 39."""
    return high_bits(r, gamma2) != high_bits((r + z) % Q, gamma2)


def use_hint(h: bool, r: int, gamma2: int) -> int:
    """Algorithm 40."""
    m = (Q - 1) // (2 * gamma2)
    r1, r0 = decompose(r, gamma2)
    if h and r0 > 0:
        return (r1 + 1) % m
    if h and r0 <= 0:
        return (r1 - 1) % m
    return r1


def infinity_norm(w: list[int]) -> int:
    """§2.3's `||w||∞`, over one polynomial."""
    return max(abs(mod_plus_minus(value, Q)) for value in w)


def integer_to_bytes(x: int, alpha: int) -> bytes:
    """Algorithm 11 — base-256, little-endian."""
    out = bytearray(alpha)
    for i in range(alpha):
        out[i] = (x >> (8 * i)) & 0xFF
    return bytes(out)


def bytes_to_bits(z: bytes) -> list[int]:
    """Algorithm 13 — each byte low bit first."""
    return [(byte >> i) & 1 for byte in z for i in range(8)]


def bits_to_integer(y: list[int]) -> int:
    """Algorithm 10 — little-endian, over the whole bit string."""
    return sum(bit << i for i, bit in enumerate(y))


def bit_unpack(v: bytes, a: int, b: int) -> list[int]:
    """Algorithm 19."""
    c = (a + b).bit_length()
    z = bytes_to_bits(v)
    return [b - bits_to_integer(z[i * c : (i + 1) * c]) for i in range(N)]


class Xof:
    """§3.7's incremental SHAKE: absorb once, then squeeze without a bound.

    Rejection sampling asks for the next few bytes until it has enough
    candidates, and how many that is depends on the bytes. `hashlib` has no
    incremental squeeze, so this re-derives a longer digest whenever the stream
    runs past what it has — SHAKE's output is a prefix chain, so a longer digest
    extends the shorter one rather than replacing it.
    """

    def __init__(self, shake: Callable[[bytes], Any], seed: bytes) -> None:
        self._hasher = shake(seed)
        self._buffer = b""
        self._position = 0

    def squeeze(self, nbytes: int) -> bytes:
        """The next `nbytes` of output."""
        end = self._position + nbytes
        if end > len(self._buffer):
            self._buffer = self._hasher.digest(max(end, 2 * len(self._buffer), 256))
        out = self._buffer[self._position : end]
        self._position = end
        return out


def coeff_from_three_bytes(b0: int, b1: int, b2: int) -> int | None:
    """Algorithm 14 — `None` is the standard's `⊥`."""
    b2_prime = b2 - 128 if b2 > 127 else b2
    z = (1 << 16) * b2_prime + (1 << 8) * b1 + b0
    return z if z < Q else None


def coeff_from_half_byte(b: int, eta: int) -> int | None:
    """Algorithm 15 — `None` is the standard's `⊥`."""
    if eta == 2 and b < 15:
        return 2 - (b % 5)
    if eta == 4 and b < 9:
        return 4 - b
    return None


def rej_ntt_poly(rho: bytes) -> list[int]:
    """Algorithm 30, over `G` = SHAKE128 (§3.7)."""
    ctx = Xof(hashlib.shake_128, rho)
    a_hat: list[int] = []
    while len(a_hat) < N:
        s = ctx.squeeze(3)
        coefficient = coeff_from_three_bytes(s[0], s[1], s[2])
        if coefficient is not None:
            a_hat.append(coefficient)
    return a_hat


def rej_bounded_poly(rho: bytes, eta: int) -> list[int]:
    """Algorithm 31, over `H` = SHAKE256 (§3.7)."""
    ctx = Xof(hashlib.shake_256, rho)
    a: list[int] = []
    while len(a) < N:
        z = ctx.squeeze(1)[0]
        z0 = coeff_from_half_byte(z % 16, eta)
        z1 = coeff_from_half_byte(z // 16, eta)
        if z0 is not None:
            a.append(z0)
        if z1 is not None and len(a) < N:
            a.append(z1)
    return a


def expand_a(rho: bytes, k: int, ell: int) -> list[list[list[int]]]:
    """Algorithm 32."""
    return [
        [
            rej_ntt_poly(rho + integer_to_bytes(s, 1) + integer_to_bytes(r, 1))
            for s in range(ell)
        ]
        for r in range(k)
    ]


def expand_s(
    rho: bytes, k: int, ell: int, eta: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Algorithm 33."""
    s1 = [rej_bounded_poly(rho + integer_to_bytes(r, 2), eta) for r in range(ell)]
    s2 = [rej_bounded_poly(rho + integer_to_bytes(r + ell, 2), eta) for r in range(k)]
    return s1, s2


def expand_mask(rho: bytes, mu: int, ell: int, gamma1: int) -> list[list[int]]:
    """Algorithm 34."""
    c = 1 + (gamma1 - 1).bit_length()
    y = []
    for r in range(ell):
        rho_prime = rho + integer_to_bytes(mu + r, 2)
        v = hashlib.shake_256(rho_prime).digest(32 * c)
        y.append(bit_unpack(v, gamma1 - 1, gamma1))
    return y


def sample_in_ball(rho: bytes, tau: int) -> list[int]:
    """Algorithm 29."""
    c = [0] * N
    ctx = Xof(hashlib.shake_256, rho)
    h = bytes_to_bits(ctx.squeeze(8))
    for i in range(N - tau, N):
        j = ctx.squeeze(1)[0]
        while j > i:
            j = ctx.squeeze(1)[0]
        c[i] = c[j]
        c[j] = (-1) ** h[i + tau - N]
    return c


def integer_to_bits(x: int, alpha: int) -> list[int]:
    """Algorithm 9 — little-endian bits."""
    return [(x >> i) & 1 for i in range(alpha)]


def bits_to_bytes(y: list[int]) -> bytes:
    """Algorithm 12 — eight bits to a byte, low bit first."""
    out = bytearray(len(y) // 8)
    for i, bit in enumerate(y):
        out[i // 8] |= bit << (i % 8)
    return bytes(out)


def simple_bit_pack(w: list[int], b: int) -> bytes:
    """Algorithm 16."""
    c = b.bit_length()
    z: list[int] = []
    for value in w:
        z += integer_to_bits(value, c)
    return bits_to_bytes(z)


def bit_pack(w: list[int], a: int, b: int) -> bytes:
    """Algorithm 17."""
    c = (a + b).bit_length()
    z: list[int] = []
    for value in w:
        z += integer_to_bits(b - value, c)
    return bits_to_bytes(z)


def simple_bit_unpack(v: bytes, b: int) -> list[int]:
    """Algorithm 18."""
    c = b.bit_length()
    z = bytes_to_bits(v)
    return [bits_to_integer(z[i * c : (i + 1) * c]) for i in range(N)]


def hint_bit_pack(h: list[list[int]], omega: int) -> bytes:
    """Algorithm 20."""
    y = bytearray(omega + len(h))
    index = 0
    for i, poly in enumerate(h):
        for j in range(N):
            if poly[j] != 0:
                y[index] = j
                index += 1
        y[omega + i] = index
    return bytes(y)


def hint_bit_unpack(y: bytes, k: int, omega: int) -> list[list[int]] | None:
    """Algorithm 21 — `None` is the standard's `⊥`."""
    h = [[0] * N for _ in range(k)]
    index = 0
    for i in range(k):
        if y[omega + i] < index or y[omega + i] > omega:
            return None
        first = index
        while index < y[omega + i]:
            if index > first and y[index - 1] >= y[index]:
                return None
            h[i][y[index]] = 1
            index += 1
    for i in range(index, omega):
        if y[i] != 0:
            return None
    return h


def pk_encode(rho: bytes, t1: list[list[int]]) -> bytes:
    """Algorithm 22."""
    width = (Q - 1).bit_length() - D
    return rho + b"".join(simple_bit_pack(row, (1 << width) - 1) for row in t1)


def pk_decode(pk: bytes, k: int) -> tuple[bytes, list[list[int]]]:
    """Algorithm 23."""
    width = (Q - 1).bit_length() - D
    size = 32 * width
    rho, body = pk[:32], pk[32:]
    return rho, [
        simple_bit_unpack(body[i * size : (i + 1) * size], (1 << width) - 1)
        for i in range(k)
    ]


def sk_encode(
    rho: bytes,
    key: bytes,
    tr: bytes,
    s1: list[list[int]],
    s2: list[list[int]],
    t0: list[list[int]],
    eta: int,
) -> bytes:
    """Algorithm 24."""
    out = rho + key + tr
    for row in s1:
        out += bit_pack(row, eta, eta)
    for row in s2:
        out += bit_pack(row, eta, eta)
    for row in t0:
        out += bit_pack(row, (1 << (D - 1)) - 1, 1 << (D - 1))
    return out


def sig_encode(
    c_tilde: bytes, z: list[list[int]], h: list[list[int]], gamma1: int, omega: int
) -> bytes:
    """Algorithm 26."""
    out = c_tilde
    for row in z:
        out += bit_pack(row, gamma1 - 1, gamma1)
    return out + hint_bit_pack(h, omega)


def sk_decode(
    sk: bytes, k: int, ell: int, eta: int
) -> tuple[bytes, bytes, bytes, list[list[int]], list[list[int]], list[list[int]]]:
    """Algorithm 25."""
    packed = 32 * (2 * eta).bit_length()
    t0_packed = 32 * D
    rho, key, tr, body = sk[:32], sk[32:64], sk[64:128], sk[128:]
    s1 = [bit_unpack(body[i * packed : (i + 1) * packed], eta, eta) for i in range(ell)]
    body = body[ell * packed :]
    s2 = [bit_unpack(body[i * packed : (i + 1) * packed], eta, eta) for i in range(k)]
    body = body[k * packed :]
    low = 1 << (D - 1)
    t0 = [
        bit_unpack(body[i * t0_packed : (i + 1) * t0_packed], low - 1, low)
        for i in range(k)
    ]
    return rho, key, tr, s1, s2, t0


def sig_decode(
    sigma: bytes, lam: int, ell: int, k: int, gamma1: int, omega: int
) -> tuple[bytes, list[list[int]], list[list[int]] | None]:
    """Algorithm 27 — the hint is `None` where the standard returns `⊥`."""
    c_size = lam // 4
    z_size = 32 * (1 + (gamma1 - 1).bit_length())
    c_tilde, body = sigma[:c_size], sigma[c_size:]
    z = [
        bit_unpack(body[i * z_size : (i + 1) * z_size], gamma1 - 1, gamma1)
        for i in range(ell)
    ]
    return c_tilde, z, hint_bit_unpack(body[ell * z_size :], k, omega)


def w1_encode(w1: list[list[int]], gamma2: int) -> bytes:
    """Algorithm 28."""
    bound = (Q - 1) // (2 * gamma2) - 1
    return b"".join(simple_bit_pack(row, bound) for row in w1)


def base_mul(a_hat: list[int], b_hat: list[int]) -> list[int]:
    """Algorithm 45 — multiplication in `T_q`, which is pointwise."""
    return [(a_hat[n] * b_hat[n]) % Q for n in range(N)]


def matrix_vector(
    a_hat: list[list[list[int]]], v_hat: list[list[int]]
) -> list[list[int]]:
    """`Â ∘ v̂` — §2.4.1's matrix-vector product, pointwise and summed."""
    return [
        [sum(row[j][n] * v_hat[j][n] for j in range(len(v_hat))) % Q for n in range(N)]
        for row in a_hat
    ]


def keygen_internal(xi: bytes, params: dict[str, int]) -> tuple[bytes, bytes]:
    """Algorithm 6, line for line."""
    k, ell, eta = params["k"], params["ell"], params["eta"]
    expanded = hashlib.shake_256(
        xi + integer_to_bytes(k, 1) + integer_to_bytes(ell, 1)
    ).digest(128)
    rho, rho_prime, key = expanded[:32], expanded[32:96], expanded[96:]
    a_hat = expand_a(rho, k, ell)
    s1, s2 = expand_s(rho_prime, k, ell, eta)
    products = matrix_vector(a_hat, [ntt(row) for row in s1])
    t = [
        [(value + s2[i][n]) % Q for n, value in enumerate(intt(products[i]))]
        for i in range(k)
    ]
    rounded = [[power2round(value) for value in row] for row in t]
    t1 = [[value[0] for value in row] for row in rounded]
    t0 = [[value[1] for value in row] for row in rounded]
    pk = pk_encode(rho, t1)
    tr = hashlib.shake_256(pk).digest(64)
    return pk, sk_encode(rho, key, tr, s1, s2, t0, eta)


def sign_internal(
    sk: bytes, m_prime: bytes, rnd: bytes, params: dict[str, int]
) -> bytes:
    """Algorithm 7, line for line — the `while` included."""
    k, ell, eta = params["k"], params["ell"], params["eta"]
    tau, lam = params["tau"], params["lam"]
    gamma1, gamma2, omega = params["gamma1"], params["gamma2"], params["omega"]
    beta = tau * eta

    rho, key, tr, s1, s2, t0 = sk_decode(sk, k, ell, eta)
    s1_hat = [ntt(row) for row in s1]
    s2_hat = [ntt(row) for row in s2]
    t0_hat = [ntt(row) for row in t0]
    a_hat = expand_a(rho, k, ell)
    mu = hashlib.shake_256(tr + m_prime).digest(64)
    rho_pp = hashlib.shake_256(key + rnd + mu).digest(64)

    kappa = 0
    while True:
        y = expand_mask(rho_pp, kappa, ell, gamma1)
        w = [intt(row) for row in matrix_vector(a_hat, [ntt(poly) for poly in y])]
        w1 = [[high_bits(value, gamma2) for value in row] for row in w]
        c_tilde = hashlib.shake_256(mu + w1_encode(w1, gamma2)).digest(lam // 4)
        c_hat = ntt(sample_in_ball(c_tilde, tau))
        cs1 = [intt(base_mul(c_hat, row)) for row in s1_hat]
        cs2 = [intt(base_mul(c_hat, row)) for row in s2_hat]
        z = [[(y[i][n] + cs1[i][n]) % Q for n in range(N)] for i in range(ell)]
        difference = [[(w[i][n] - cs2[i][n]) % Q for n in range(N)] for i in range(k)]
        r0 = [[low_bits(value, gamma2) for value in row] for row in difference]
        kappa += ell
        if max(infinity_norm(row) for row in z) >= gamma1 - beta:
            continue
        if max(infinity_norm(row) for row in r0) >= gamma2 - beta:
            continue
        ct0 = [intt(base_mul(c_hat, row)) for row in t0_hat]
        hint = [
            [
                int(
                    make_hint(
                        (-ct0[i][n]) % Q, (difference[i][n] + ct0[i][n]) % Q, gamma2
                    )
                )
                for n in range(N)
            ]
            for i in range(k)
        ]
        if max(infinity_norm(row) for row in ct0) >= gamma2:
            continue
        if sum(sum(row) for row in hint) > omega:
            continue
        centered = [[mod_plus_minus(value, Q) for value in row] for row in z]
        return sig_encode(c_tilde, centered, hint, gamma1, omega)


def verify_internal(
    pk: bytes, m_prime: bytes, sigma: bytes, params: dict[str, int]
) -> bool:
    """Algorithm 8, line for line."""
    k, ell, eta = params["k"], params["ell"], params["eta"]
    tau, lam = params["tau"], params["lam"]
    gamma1, gamma2, omega = params["gamma1"], params["gamma2"], params["omega"]
    beta = tau * eta

    rho, t1 = pk_decode(pk, k)
    c_tilde, z, hint = sig_decode(sigma, lam, ell, k, gamma1, omega)
    if hint is None:
        return False
    a_hat = expand_a(rho, k, ell)
    tr = hashlib.shake_256(pk).digest(64)
    mu = hashlib.shake_256(tr + m_prime).digest(64)
    c_hat = ntt(sample_in_ball(c_tilde, tau))
    scaled = [ntt([(value << D) % Q for value in row]) for row in t1]
    az = matrix_vector(a_hat, [ntt(row) for row in z])
    ct1 = [base_mul(c_hat, row) for row in scaled]
    approx = [intt([(az[i][n] - ct1[i][n]) % Q for n in range(N)]) for i in range(k)]
    w1 = [
        [use_hint(bool(hint[i][n]), approx[i][n], gamma2) for n in range(N)]
        for i in range(k)
    ]
    c_tilde_prime = hashlib.shake_256(mu + w1_encode(w1, gamma2)).digest(lam // 4)
    if max(infinity_norm(row) for row in z) >= gamma1 - beta:
        return False
    return c_tilde == c_tilde_prime


# Algorithm 4's `switch PH`, transcribed a second time and independently of
# `prehash.py`'s table, because the OID has no other check: it is bytes inside the
# signed message rather than a value any round trip reads back, so a wrong one is
# self-consistent everywhere except against a published signature.
#
# The keys are the names the validation program's files carry, since those are
# what selects a case. The standard enumerates three of these (SHA-256, SHA-512
# and SHAKE128) and writes `case …` for the rest; each entry is the `hashlib`
# function, its OID DER-encoded with tag and length, and the bytes read out.
PRE_HASHES: dict[str, tuple[str, str, int]] = {
    "SHA2-256": ("sha256", "0609608648016503040201", 32),
    "SHA3-256": ("sha3_256", "0609608648016503040208", 32),
    "SHA3-512": ("sha3_512", "060960864801650304020a", 64),
    "SHAKE-128": ("shake_128", "060960864801650304020b", 32),
    "SHAKE-256": ("shake_256", "060960864801650304020c", 64),
}


def pre_hash_message(m: bytes, context: bytes, pre_hash: str) -> bytes:
    """`M′` — Algorithm 4 lines 10 to 23, which Algorithm 5 line 18 repeats.

    The one place the two differ is what they do with it, so both variants of the
    pre-hash pair build their message here.
    """
    name, oid, digest_size = PRE_HASHES[pre_hash]
    # A XOF reads out a length the caller names and a hash has one of its own;
    # `hashlib` distinguishes them by whether `digest` takes an argument.
    hashed = hashlib.new(name, m)
    ph_m = (
        hashed.digest(digest_size)  # type: ignore[call-arg]
        if name.startswith("shake_")
        else hashed.digest()
    )
    if len(ph_m) != digest_size:
        raise ValueError(f"{pre_hash} produced {len(ph_m)} bytes, not {digest_size}")
    return (
        integer_to_bytes(1, 1)
        + integer_to_bytes(len(context), 1)
        + context
        + bytes.fromhex(oid)
        + ph_m
    )


def hash_sign(
    sk: bytes,
    m: bytes,
    rnd: bytes,
    context: bytes,
    pre_hash: str,
    params: dict[str, int],
) -> bytes:
    """Algorithm 4 — pre-hash signing, over Algorithm 7."""
    return sign_internal(sk, pre_hash_message(m, context, pre_hash), rnd, params)


def hash_verify(
    pk: bytes,
    m: bytes,
    sigma: bytes,
    context: bytes,
    pre_hash: str,
    params: dict[str, int],
) -> bool:
    """Algorithm 5 — the counterpart of `hash_sign`."""
    return verify_internal(pk, pre_hash_message(m, context, pre_hash), sigma, params)
