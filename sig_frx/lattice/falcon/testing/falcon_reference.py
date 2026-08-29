# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Falcon's algorithms, transcribed the way the specification writes them.

The modules under test reshape every one of these. Algorithm 18's cursor walk
becomes an associative scan over a nine-state machine, Algorithm 3's `while`
becomes a fixed candidate budget with a compaction, the ring product becomes an
NTT round trip, and Algorithm 16 becomes one `vmap`ped computation over a batch.
Each is a change made for the compiler, and it is the only thing about
`encoding.py` and `falcon.py` a reader has to take on trust — so this file takes
it back, looping one coefficient and one bit at a time over Python integers, and
the tests require the two to agree
([`testing.md`](../../../../docs/reference/testing.md)).

`Compress` is here as well as `Decompress`, even though the implementation has no
encoder yet ([#27](https://github.com/fractalyze/sig-frx/issues/27) brings one).
A decoder cannot be exercised over the space of *valid* inputs without something
that produces them, and the published vectors carry two signatures per degree —
which reaches none of the coefficient magnitudes that make `k` large. So the
encoder is the test's, and what it buys is `Decompress(Compress(s)) = s` over
coefficients drawn to stress the unary run rather than over the two that
upstream happened to publish.

Transcribed from the published document (§3.8 through §3.11), not from memory and
not from another implementation. Nothing here is written for speed: Python
integers have no width, which is exactly why this side is trustworthy and the
other side delegates its arithmetic to a field dtype.

The rational half — §3.8's transform, its splitting operator, and Algorithms 8
and 9 — cannot be exact, since the quantities are irrational before any
implementation touches them. There the independence is structural instead: the
transform is direct evaluation rather than a recursion, and the splitting
operator is written from the identity that defines it rather than from an index.

The hashing half takes the standard library's SHAKE — the point being that
squeezing here asks for more bytes whenever it needs them, so the reference never
has a block bound to get wrong, and the implementation's budget is compared
against a stream that has none.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

Q = 12289

# Table 3.3, in full and in one place, so no test module transcribes a column of
# its own. `sbytelen` and the key lengths are the standard's stated values rather
# than the implementation's formulas — checking a formula against itself proves
# nothing.
#
# The column type is `Any` because the table's is: `n` and the lengths are
# integers and the three standard deviations are not, and a union would be read
# back with a cast at every use site for no property anyone checks.
PARAMETER_SETS: dict[str, dict[str, Any]] = {
    "Falcon-512": {
        "n": 512,
        "squared_norm_bound": 34034726,
        "public_key_size": 897,
        "secret_key_size": 1281,
        "signature_size": 666,
        "sigma": 165.736617183,
        "sigma_min": 1.277833697,
        "sigma_max": 1.8205,
    },
    "Falcon-1024": {
        "n": 1024,
        "squared_norm_bound": 70265242,
        "public_key_size": 1793,
        "secret_key_size": 2305,
        "signature_size": 1280,
        "sigma": 168.388571447,
        "sigma_min": 1.298280334,
        "sigma_max": 1.8205,
    },
}


def parameter_cases() -> tuple[dict[str, Any], ...]:
    """Table 3.3 as `parameterized.parameters` records, name included."""
    return tuple({"name": name, **values} for name, values in PARAMETER_SETS.items())


def slen(signature_size: int) -> int:
    """Algorithm 16 line 2's `8 · sbytelen − 328`."""
    return 8 * signature_size - 328


# §3.11.6's `sm` layout ahead of the message: the length prefix, then the salt.
SIGLEN_SIZE = 2
SALT_SIZE = 40


def signature_from_aggregate(sm: bytes, message: bytes, name: str) -> bytes | None:
    """§3.11.6's signed message as the §3.11.3 signature the seam takes.

    The NIST API packs `siglen(2) ‖ r(40) ‖ M ‖ header(1) ‖ enc_s` with a
    nonce-less header of `0010nnnn`, because the salt already appeared ahead of
    the message. §3.11.3 defines `header(1) ‖ r(40) ‖ enc_s` with header
    `0011nnnn`, zero-padded to `sbytelen`. This is that regrouping, and every
    check is a claim about the two sections agreeing — a wrong offset yields a
    plausible byte string that rejects everything, which is indistinguishable
    from a broken verifier.

    `None` when the record has no §3.11.3 form at all. That is possible because
    the NIST API's signature is variable-length and carries no equivalent of
    Algorithm 10's restart when `enc_s` comes out longer than `sbytelen − 41`;
    a caller that produced the aggregate itself should treat it as an error,
    while one reading a published file records it.

    Here rather than in [`encoding`](../encoding.py) because `sm` is the
    submission API's packaging rather than anything the standard's own wire
    format defines — and because a fixture that regrouped bytes with the
    module under test would be checking a formula against itself. The constants
    are this file's own for the same reason.
    [`aggregate_from_signature`](#aggregate_from_signature) is the way back.
    """
    n = PARAMETER_SETS[name]["n"]
    logn = n.bit_length() - 1
    salt_end = SIGLEN_SIZE + SALT_SIZE
    salt = sm[SIGLEN_SIZE:salt_end]
    body_start = salt_end + len(message)

    if sm[salt_end:body_start] != message:
        raise ValueError(f"{name}: `sm` does not carry `msg` where §3.11.6 puts it")
    nonceless = sm[body_start:]
    stated = int.from_bytes(sm[:SIGLEN_SIZE], "big")
    if len(nonceless) != stated:
        raise ValueError(
            f"{name}: `sm` tail is {len(nonceless)} bytes against a stated {stated}"
        )
    if nonceless[0] != 0x20 | logn:
        raise ValueError(
            f"{name}: nonce-less header is {nonceless[0]:#04x}, "
            f"not §3.11.6's {0x20 | logn:#04x}"
        )

    compressed = nonceless[1:]
    padding = PARAMETER_SETS[name]["signature_size"] - 1 - SALT_SIZE - len(compressed)
    if padding < 0:
        return None
    return bytes([0x30 | logn]) + salt + compressed + b"\x00" * padding


def aggregate_from_signature(
    signature: bytes, message: bytes, name: str
) -> bytes | None:
    """§3.11.3's signature as the §3.11.6 signed message the NIST API reads.

    [`signature_from_aggregate`](#signature_from_aggregate) run backwards, and
    beside it because the two are one fact about two sections: a change to the
    padding rule or to either header nibble has to be a change to both, which
    is only obvious when they are adjacent. `crypto_sign_open` reads
    `siglen ‖ r ‖ M ‖ header ‖ enc_s` and nothing else, so handing it §3.11.3's
    form directly rejects **every** case — indistinguishable from a broken
    signer, which is why the regrouping is the first thing to doubt when a
    caller starts refusing everything at once.

    The trailing zeros §3.11.3 pads to `sbytelen` are dropped rather than
    passed on: the aggregate states its own length and the reference checks the
    compressed run against it, so padding it never saw reads as malformed. The
    strip is exact rather than approximate — Algorithm 17 ends each coefficient
    with a terminating `1`, so the last byte of a well-formed `enc_s` is never
    zero and there is nothing but padding to take. That is the one claim here a
    round trip cannot check, since the forward direction re-pads whatever this
    one leaves behind, so [`falcon_kat_test`](falcon_kat_test.py) holds the
    result against upstream's published `sm` instead.

    `None` for a header that is not §3.11.3's, rather than a raise: those bytes
    come from the implementation under test, so refusing them is a verdict about
    it. It has to be refused *here* and cannot be left to the reference — the
    byte written on the way out is `0x20 | logn` whatever byte came in, so a
    corrupted header would be repaired in transit and the signature would
    verify. A wrong *length* is the caller's error instead, since §3.11.3's form
    is fixed-width and bytes of another width are not a signature to have an
    opinion about.
    """
    params = PARAMETER_SETS[name]
    logn = params["n"].bit_length() - 1
    if len(signature) != params["signature_size"]:
        raise ValueError(
            f"{name}: a signature is {params['signature_size']} bytes, "
            f"got {len(signature)}"
        )
    if signature[0] != 0x30 | logn:
        return None

    salt = signature[1 : 1 + SALT_SIZE]
    compressed = signature[1 + SALT_SIZE :].rstrip(b"\x00")
    nonceless = bytes([0x20 | logn]) + compressed
    return len(nonceless).to_bytes(SIGLEN_SIZE, "big") + salt + message + nonceless


def bits_of(data: bytes) -> list[int]:
    """§3.11.1 — a byte's leftmost bit has weight 128."""
    return [(byte >> shift) & 1 for byte in data for shift in range(7, -1, -1)]


def bytes_of(bits: list[int]) -> bytes:
    """The inverse of `bits_of`; `bits` must be a whole number of bytes."""
    return bytes(
        sum(bit << shift for bit, shift in zip(bits[i : i + 8], range(7, -1, -1)))
        for i in range(0, len(bits), 8)
    )


def compress(s: list[int], length: int) -> bytes | None:
    """Algorithm 17 — `None` where the specification returns `⊥`."""
    bits: list[int] = []
    for value in s:
        bits.append(1 if value < 0 else 0)
        magnitude = abs(value)
        bits.extend((magnitude >> j) & 1 for j in range(6, -1, -1))
        bits.extend([0] * (magnitude >> 7))
        bits.append(1)
    if len(bits) > length:
        return None
    return bytes_of(bits + [0] * (length - len(bits)))


def decompress(bits: list[int], length: int, n: int) -> list[int] | None:
    """Algorithm 18 — `None` where the specification returns `⊥`."""
    if len(bits) != length:
        return None
    out: list[int] = []
    cursor = 0
    for _ in range(n):
        if cursor + 8 > length:
            return None
        sign = bits[cursor]
        low = 0
        for j in range(7):
            low = (low << 1) | bits[cursor + 1 + j]
        k = 0
        while True:
            if cursor + 8 + k >= length:
                return None
            if bits[cursor + 8 + k] == 1:
                break
            k += 1
        value = (-1 if sign else 1) * (low + (k << 7))
        if value == 0 and sign == 1:
            return None
        out.append(value)
        cursor += 9 + k
    if any(bits[cursor:]):
        return None
    return out


def hash_to_point(message: bytes, n: int) -> list[int]:
    """Algorithm 3 — 16-bit big-endian draws, rejecting `t ≥ kq`."""
    k = (1 << 16) // Q
    shake = hashlib.shake_256(message)
    out: list[int] = []
    taken = 0
    while len(out) < n:
        taken += 64
        stream = shake.digest(taken)
        for i in range((taken - 64) // 2, taken // 2):
            if len(out) == n:
                break
            t = (stream[2 * i] << 8) | stream[2 * i + 1]
            if t < k * Q:
                out.append(t % Q)
    return out


def pk_decode(pk: bytes, n: int) -> list[int] | None:
    """§3.11.4 — `None` for a bad header or a coefficient at or above `q`."""
    if len(pk) != 1 + (14 * n + 7) // 8 or pk[0] != n.bit_length() - 1:
        return None
    bits = bits_of(pk[1:])
    out = []
    for i in range(n):
        value = 0
        for j in range(14):
            value = (value << 1) | bits[14 * i + j]
        if value >= Q:
            return None
        out.append(value)
    return out


def pk_encode(h: list[int], n: int) -> bytes:
    """§3.11.4 — the header byte `0000nnnn`, then `h` at 14 bits a coefficient."""
    bits = bits_of(bytes([0x00 | (n.bit_length() - 1)]))
    for value in h:
        bits.extend((value >> j) & 1 for j in range(13, -1, -1))
    return bytes_of(bits)


SK_WIDTHS: dict[int, int] = {
    2: 8,
    4: 8,
    8: 8,
    16: 8,
    32: 8,
    64: 7,
    128: 7,
    256: 6,
    512: 6,
    1024: 5,
}
"""§3.11.5's `f` and `g` widths, transcribed at every degree the section lists.

The implementation carries the two Falcon defines; this carries all eight,
because a table with only the used rows cannot say whether the rule was read
correctly — and the boundaries (32 to 64, 128 to 256, 512 to 1024) are where a
misreading would land.
"""


def sk_encode(f: list[int], g: list[int], big_f: list[int], n: int) -> bytes:
    """§3.11.5 — `0101nnnn`, then `f`, `g` and `F` in that order.

    Signed encoding, two's complement, at `SK_WIDTHS[n]` bits for `f` and `g`
    and eight for `F`. `G` is not encoded; (3.35) recovers it.
    """
    bits = bits_of(bytes([0x50 | (n.bit_length() - 1)]))
    for values, width in ((f, SK_WIDTHS[n]), (g, SK_WIDTHS[n]), (big_f, 8)):
        for value in values:
            field = value & ((1 << width) - 1)
            bits.extend((field >> j) & 1 for j in range(width - 1, -1, -1))
    return bytes_of(bits)


def sk_decode(sk: bytes, n: int) -> tuple[list[int], list[int], list[int]] | None:
    """§3.11.5 read back — `None` where the encoding is malformed.

    Two ways it can be, both of which the section states: a header that is not
    `0101nnnn` for this `n`, and a coefficient at the minimal value, which
    "is forbidden; e.g. when using degree 512, the valid range for a coefficient
    of `f` or `g` is −31 to +31; −32 is not allowed."
    """
    width = SK_WIDTHS[n]
    if len(sk) != 1 + n * (2 * width + 8) // 8:
        return None
    if sk[0] != 0x50 | (n.bit_length() - 1):
        return None
    bits = bits_of(sk[1:])
    out: list[list[int]] = []
    cursor = 0
    for count, size in ((n, width), (n, width), (n, 8)):
        values = []
        for _ in range(count):
            field = 0
            for _ in range(size):
                field = (field << 1) | bits[cursor]
                cursor += 1
            if field == 1 << (size - 1):
                return None
            values.append(field - (1 << size) if field >> (size - 1) else field)
        out.append(values)
    return out[0], out[1], out[2]


def negacyclic_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`(a · b) mod (x^n + 1) mod q`, in exact integers.

    The wrap is a subtraction rather than an addition — that sign is the whole
    difference between this ring and the cyclic one.

    `int64` holds it exactly rather than nearly: a coefficient is under
    `q = 12289`, so a product is under `1.6e8` and a length-1024 sum under
    `1.6e11`, against `int64`'s `9.2e18`. That exactness is what makes this an
    oracle; the convolution is numpy's because nothing about *how* the exact
    integers are added is the property under test, and a `n²` Python loop would
    put a minute onto `arith_test`.

    Shared with [`arith_test`](arith_test.py), which is where this form comes
    from: two oracles for one ring is one edit away from gating the transform
    and the scheme against different rings.
    """
    n = len(a)
    full = np.convolve(a.astype(np.int64), b.astype(np.int64))
    wrapped = full[:n].copy()
    wrapped[: n - 1] -= full[n:]
    return wrapped % Q


def centered(w: list[int]) -> list[int]:
    """Residues as the `⌈−q/2⌉ … ⌊q/2⌋` representatives a norm is measured over."""
    return [x - Q if x > Q // 2 else x for x in w]


def verify(pk: bytes, message: bytes, signature: bytes, name: str) -> bool:
    """Algorithm 16 over §3.11.3's padded signature, verdict only."""
    params = PARAMETER_SETS[name]
    n = params["n"]
    if len(signature) != params["signature_size"]:
        return False
    if signature[0] != 0x30 | (n.bit_length() - 1):
        return False
    h = pk_decode(pk, n)
    if h is None:
        return False
    s2 = decompress(bits_of(signature[41:]), slen(params["signature_size"]), n)
    if s2 is None:
        return False
    c = hash_to_point(signature[1:41] + message, n)
    product = negacyclic_mul(np.array(s2) % Q, np.array(h))
    s1 = centered([(ci - pi) % Q for ci, pi in zip(c, product)])
    norm = sum(x * x for x in s1) + sum(x * x for x in s2)
    return norm <= params["squared_norm_bound"]


def exact_negacyclic_mul(a: list[int], b: list[int]) -> list[int]:
    """`(a · b) mod (x^m + 1)` over Python integers, at any coefficient width.

    Distinct from [`negacyclic_mul`](#negacyclic_mul) above, which reduces mod
    `q` and holds its accumulator in `int64`. The NTRU descent's coefficients
    pass 9,000 bits and are not reduced by anything, so numpy cannot hold them
    and the loop is the price of an oracle that is exact rather than nearly.
    """
    degree = len(a)
    out = [0] * degree
    for i, left in enumerate(a):
        if left == 0:
            continue
        for j, right in enumerate(b):
            position = i + j
            if position < degree:
                out[position] += left * right
            else:
                out[position - degree] -= left * right
    return out


def field_norm(a: list[int]) -> list[int]:
    """`N(f) = f_e² - x·f_o²`, Algorithm 6's descent, one level.

    `f(x) = f_e(x²) + x·f_o(x²)` splits by parity, and the norm lands in
    `Z[x]/(x^(m/2) + 1)`. The wrap that makes the last term of `x·f_o²` come
    back with a flipped sign is the same one `exact_negacyclic_mul` applies, and
    it is the only place the ring shows through.
    """
    half = len(a) // 2
    even = exact_negacyclic_mul(a[0::2], a[0::2])
    odd = exact_negacyclic_mul(a[1::2], a[1::2])
    out = list(even)
    out[0] += odd[half - 1]
    for i in range(1, half):
        out[i] -= odd[i - 1]
    return out


def fft_roots(n: int) -> np.ndarray:
    """The `n` roots of `x^n = −1`: `ζ_k = e^(iπ(2k+1)/n)`, in natural order.

    The one place this file fixes an order for the rational transform. §3.6
    leaves the convention to the implementer and asks only that `FFT`, `invFFT`,
    `splitfft` and `mergefft` agree on it — so a second copy here would be a
    second chance for two oracles to disagree about what index means what.
    """
    return np.exp(1j * np.pi * (2 * np.arange(n) + 1) / n)


def evaluate(f: list[int] | list[float] | np.ndarray) -> np.ndarray:
    """`FFT(f)` — the polynomial at each of [`fft_roots`](#fft_roots).

    Direct evaluation rather than a transform, which is what makes this
    independent of [`fft.py`](../fft.py): it shares neither the twiddle table
    nor the recursion, only the roots themselves.
    """
    coefficients = np.asarray(f, dtype=float)
    return np.polyval(coefficients[::-1], fft_roots(len(coefficients)))


def split_fft(f_fft: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Algorithm 1, from the identity that defines it rather than by index.

    `f(x) = f0(x²) + x·f1(x²)` evaluated at `ζ` and at `−ζ` gives

        f0(ζ²) = (f(ζ) + f(−ζ))/2        f1(ζ²) = (f(ζ) − f(−ζ))/2ζ

    so the pairing follows from the ordering instead of being transcribed
    alongside it: in [`fft_roots`](#fft_roots)'s order `−ζ_k` is `ζ_(k + n/2)`,
    and `ζ_k²` is the `k`-th root of the ring below, so the two halves come out
    in that same natural order. The reference implementation pairs *adjacent*
    indices, correctly, because its representation is bit-reversed — which is
    exactly the sort of detail a transcription of the index arithmetic would
    import by accident.
    """
    n = f_fft.shape[-1]
    half = n // 2
    lo, hi = f_fft[..., :half], f_fft[..., half:]
    return 0.5 * (lo + hi), 0.5 * (lo - hi) / fft_roots(n)[:half]


def gram(
    f: list[int], g: list[int], big_f: list[int], big_g: list[int]
) -> list[list[np.ndarray]]:
    """Algorithm 4 lines 2-4 — `B̂ × B̂*` for `B = [[g, −f], [G, −F]]`.

    The matrix product written out, negations and all four entries included,
    where [`keygen.gram`](../keygen.py) folds `(−f)·(−f)* = f·f*` and returns
    three entries because `G10` is `conj(G01)`. Both of those are the things a
    test of it should have to check rather than share.

    An adjoint is elementwise conjugation in this domain, since a root lies on
    the unit circle and `f*(ζ) = conj(f(ζ))`.
    """
    rows = [[evaluate(g), -evaluate(f)], [evaluate(big_g), -evaluate(big_f)]]
    return [
        [sum(a * np.conj(b) for a, b in zip(left, right)) for right in rows]
        for left in rows
    ]


def ldl(matrix: list[list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Algorithm 8 at the 2×2 shape ffLDL always hands it — `(L10, D00, D11)`."""
    (g00, g01), (g10, g11) = matrix
    d00 = g00
    l10 = g10 / g00
    d11 = g11 - l10 * np.conj(l10) * g00
    return l10, d00, d11


def ffldl(matrix: list[list[np.ndarray]]) -> tuple[Any, Any, Any]:
    """Algorithm 9 as the `(value, leftchild, rightchild)` tree it describes.

    One call per node, where [`keygen.ffldl`](../keygen.py) runs one array
    operation per depth. A leaf is `D00` or `D11` at ring degree 2 and is kept
    whole — the length-2 evaluation of what the algebra says is a rational
    constant — so a test can check that its two entries agree rather than being
    handed the number they agree on.
    """
    l10, d00, d11 = ldl(matrix)
    if d00.shape[-1] == 2:
        return l10, d00, d11
    even_from_d00, odd_from_d00 = split_fft(d00)
    even_from_d11, odd_from_d11 = split_fft(d11)
    # Line 10: `D` self-adjoint makes its multiplication map `[[d0, d1],
    # [d1*, d0]]` over the ring below — (3.30).
    return (
        l10,
        ffldl([[even_from_d00, odd_from_d00], [np.conj(odd_from_d00), even_from_d00]]),
        ffldl([[even_from_d11, odd_from_d11], [np.conj(odd_from_d11), even_from_d11]]),
    )


def gram_schmidt_squared_norm(f: list[int], g: list[int]) -> float:
    """Algorithm 5 line 5, written the way the specification states it.

    The larger of the NTRU basis's two orthogonalised rows: `‖(g, −f)‖²`, and
    `Q²` times the norm of `(ḡ, f̄)` divided by `f f̄ + g ḡ`. The division is
    what forces the rational transform — there is no integer form of this — so
    the transcription uses numpy's own complex arithmetic rather than the
    module's, which is the independence that makes it an oracle.
    """
    n = len(f)
    roots = fft_roots(n)
    f_hat, g_hat = evaluate(f), evaluate(g)
    energy = (f_hat * np.conj(f_hat) + g_hat * np.conj(g_hat)).real
    # Back to coefficients by solving the Vandermonde system the roots define,
    # which is the inverse transform written as what it is.
    powers = np.vander(roots, n, increasing=True)
    from_g = np.linalg.solve(powers, np.conj(g_hat) / energy).real
    from_f = np.linalg.solve(powers, np.conj(f_hat) / energy).real
    first = float(np.sum(np.asarray(f, float) ** 2) + np.sum(np.asarray(g, float) ** 2))
    second = Q**2 * float(np.sum(from_g**2) + np.sum(from_f**2))
    return max(first, second)


def is_invertible(f: list[int]) -> bool:
    """Algorithm 5 line 6 — `f` a unit in `Z_q[x]/(x^n + 1)`.

    Evaluated at the `2n`-th roots of unity by direct exponentiation rather than
    by a transform, so the check and the module's NTT share nothing.
    """
    n = len(f)
    root = pow(11, (Q - 1) // (2 * n), Q)
    assert pow(root, n, Q) == Q - 1, "11 does not generate a 2n-th root here"
    # `root^(2n) = 1`, which the assertion above is half of, so the `n²`
    # exponentiations the definition reads as are `2n` multiplications and an
    # index. Still direct evaluation at the roots rather than a transform, which
    # is the independence this file exists for — and 7x faster, which at
    # `n = 1024` is most of a second off the leg.
    powers = [1] * (2 * n)
    for k in range(1, 2 * n):
        powers[k] = powers[k - 1] * root % Q
    return all(
        sum(int(f[j]) * powers[(2 * i + 1) * j % (2 * n)] for j in range(n)) % Q
        for i in range(n)
    )


def lift(lower: list[int], other: list[int]) -> list[int]:
    """`lower(x²) · other(-x)`, Algorithm 6's step back up out of the recursion.

    The counterpart to [`field_norm`](#field_norm) above: the descent halves the
    degree and this doubles it back. `f(x)·f(-x)` is `N(f)(x²)`, so setting
    `F = F'(x²)·g(-x)` turns the half-degree solution into a full-degree one
    without touching the equation it satisfies.

    Two substitutions and no arithmetic of its own. `lower(x²)` spreads the
    coefficients over the even positions and leaves the odd ones empty;
    `other(-x)` flips the sign of the odd ones. The product is the same
    negacyclic one every other step here uses.
    """
    degree = len(other)
    squared = [0] * degree
    for i, value in enumerate(lower):
        squared[2 * i] = value
    negated = [value if i % 2 == 0 else -value for i, value in enumerate(other)]
    return exact_negacyclic_mul(squared, negated)


def ntru_equation(
    f: list[int], g: list[int], big_f: list[int], big_g: list[int]
) -> list[int]:
    """`f·G − g·F`, which Algorithm 6 exists to make equal `q`.

    Exact and at full degree, which is the only form of this check worth
    running: every step of the recursion preserves the equation, so a version
    that held only approximately would be evidence of nothing.
    """
    return [
        left - right
        for left, right in zip(
            exact_negacyclic_mul(f, big_g), exact_negacyclic_mul(g, big_f)
        )
    ]
