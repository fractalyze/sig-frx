# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The SLH-DSA tweakable hash family — FIPS 205 §11.

Six functions every hash-based component calls: two PRFs, the message hash, the
one-way function `F`, the fixed-length hash `H`, and the variable-length `T_l`.
Each is tweaked by an address (`adrs`), so the same input hashed at two positions
in the hypertree gives two unrelated outputs.

`F` and `H` also carry their own protocols here, because the components that use
only one of them — `wots.chain` and all of `tree.py` — are shared with RFC 8391,
whose family (`hashes`) is not this one: it has no `T_l`, tweaks with a
key and a bitmask rather than a prefix, and derives its message digest from a
different set of operands. Asking those components for the whole family would make
the sharing a lie the type checker happens not to catch.

A parameter-set family is whatever hashes its instantiation names, plus the sizes
— which is what makes the SHA-2 and SHAKE sets one implementation rather than two,
and is the reason that seam exists. `Sha2TweakableHash` is the SHA-2 instantiation
(§11.2), over one hash at security category 1 and two at categories 3 and 5; a
SHAKE instantiation (§11.1) is the same class shape over SHAKE256, differing in
how each function derives its digest rather than in what it is for.

Inputs and outputs are `uint8` arrays with a leading batch axis, because the
callers hash a whole WOTS+ chain step or a whole Merkle level at once. `pk_seed`
is either shared across the batch — one seed, many positions, which is one key's
own tree — or given per entry, which is what verifying a batch of signatures
under different public keys needs. Addresses arrive already encoded
(`adrs.encode_batch`) — they are structural, so they are built on the host before
any hashing starts.

**`uint8` is this file's answer, not the protocols'.** `ChainHash` and `NodeHash`
carry a `dtype`, because the components written to them — `wots.chain` and all of
`tree.py` — are shared with a family whose digests are not bytes at all: leanSig
hashes with Poseidon over KoalaBear, so a digest is eight *field* elements and a
Merkle pair is sixteen ([`leansig/tweakable.py`](leansig/tweakable.py)). Those
components used to spell `fnp.uint8` at each `asarray`, which made a byte string
the walk's own assumption rather than the family's; reading it off the family is
the whole of the change, and `n` generalizes with it — an output length in
elements of `dtype`, which is bytes exactly when `dtype` is `uint8`.

`prf_msg` is the exception, taking one message rather than a batch: it is called
once per signature, on the signing path, which this repo does not put on the hot
path (see `docs/reference/security.md`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike, DTypeLike
from hash_frx import ByteHash, Hmac, Mgf1
from hash_frx import block_size as block_size_of

# What a family is tweaked by: FIPS 205 and RFC 8391 encode an address into
# bytes, leanSig packs a position into field elements. Named here rather than in
# a consumer because the protocols below are what take one, and spelled as an
# array rather than as `bytestring.ByteString` because only two of the three
# families produce bytes.
Tweak: TypeAlias = np.ndarray | Array


@runtime_checkable
class ChainHash(Protocol):
    """`F` alone — what a WOTS+ chain step needs and all it needs.

    Narrower than `TweakableHash` because `wots.chain` is shared with RFC 8391,
    whose family (`hashes`) has no `T_l` and no compressed address and
    computes its message digest from different operands. `F` is the one function
    the two standards agree on the shape of, so it is the one this asks for.
    """

    n: int  # security parameter and output length, in elements of `dtype`
    # What a digest is made of. `uint8` for every byte family here; leanSig's is
    # a KoalaBear field dtype. `wots.chain` reads it so the walk stops assuming
    # bytes — see the module docstring.
    dtype: DTypeLike

    def f(self, pk_seed: ArrayLike, adrs: ArrayLike, m1: ArrayLike, /) -> Array:
        """One-way function on one n-element block: -> `dtype` `[B, n]`.

        Positional-only, because the operands are named for the standard the
        family implements and no caller here spells them: `pk_seed` is leanSig's
        public parameter and `adrs` is its tweak. A protocol that fixed the names
        would make a family choose between matching this file and matching its
        own specification.
        """
        ...


@runtime_checkable
class NodeHash(Protocol):
    """`H` alone — what a Merkle level needs, and what `tree.py` is written to.

    The same reasoning as `ChainHash`: a hash tree is the same walk under either
    standard, so it asks for the parent hash rather than for a whole family.
    """

    n: int
    dtype: DTypeLike

    def h(self, pk_seed: ArrayLike, adrs: ArrayLike, m2: ArrayLike, /) -> Array:
        """Hash of two n-element blocks — a Merkle parent: -> `dtype` `[B, n]`."""
        ...


@runtime_checkable
class TweakableHash(ChainHash, NodeHash, Protocol):
    m: int  # message-digest length `h_msg` produces, in bytes
    # Which address encoding this family tweaks with: the SHA-2 sets compress to
    # 22 bytes, the SHAKE sets use the full 32. A caller reads it to build its
    # addresses and never names the family.
    compressed_address: bool

    def prf(self, pk_seed: ArrayLike, sk_seed: ArrayLike, adrs: ArrayLike, /) -> Array:
        """Secret-key element from the seeds and a position: -> uint8 `[B, n]`."""
        ...

    def prf_msg(
        self, sk_prf: ArrayLike, opt_rand: ArrayLike, message: ArrayLike, /
    ) -> Array:
        """The per-signature randomizer `R`: -> uint8 `[n]`. One message."""
        ...

    def h_msg(
        self,
        randomizer: ArrayLike,
        pk_seed: ArrayLike,
        pk_root: ArrayLike,
        message: ArrayLike,
        /,
    ) -> Array:
        """The message digest a signature is over: -> uint8 `[B, m]`."""
        ...

    def t(self, pk_seed: ArrayLike, adrs: ArrayLike, messages: ArrayLike, /) -> Array:
        """Hash of `l` n-byte blocks — a WOTS+ or FORS root: -> uint8 `[B, n]`."""
        ...


def batched(value: ArrayLike, batch: int, *, dtype: DTypeLike = fnp.uint8) -> Array:
    """Broadcast a shared `[k]` operand to `[B, k]`, or pass a `[B, k]` through.

    Every family here wants this, at whichever `dtype` its digests are: a public
    seed or parameter is one value across a key's own tree and one per entry as
    soon as a batch spans public keys. It takes the dtype rather than naming one
    for the same reason `tree.py` and `wots.chain` read `tweak.dtype` — the
    element type belongs to the family — and it defaults to `uint8` because the
    byte families outnumber the field one and would otherwise pass it at every
    call.
    """
    array = fnp.asarray(value, dtype=dtype)
    if array.ndim == 1:
        return fnp.broadcast_to(array, (batch, array.shape[0]))
    return array


def hmac(
    byte_hash: ByteHash,
    key: ArrayLike,
    message: ArrayLike,
    *,
    block_size: int | None = None,
) -> Array:
    """HMAC over `byte_hash` — FIPS 198-1, on one message.

    A module function because the second caller is not a family. `PRF_msg` is one
    of these keyed by `SK.prf` and `Sha2TweakableHash` owns it; SHRINCS's stateful
    path keys the same construction with `sk_prf` filled out to a whole block,
    which is that scheme's own construction and not a tweakable hash's.

    The block size comes off the hash rather than from the caller, because a block
    size that disagrees with its hash produces a self-consistent wrong MAC — one
    that verifies against itself forever. `block_size` overrides it for the case
    the table cannot name, which is the same case `Sha2TweakableHash` takes one
    for: a hash built to count calls rather than to be looked up.
    """
    payload = fnp.asarray(message, dtype=fnp.uint8)
    width = block_size if block_size is not None else block_size_of(byte_hash)
    return fnp.asarray(
        Hmac(byte_hash, width).mac(key, payload[None, :]), dtype=fnp.uint8
    )[0]


def repeat_per_entry(
    value: ArrayLike, times: int, *, dtype: DTypeLike = fnp.uint8
) -> Array:
    """Line a per-entry operand up with an entry-major batch of `times` rows each.

    A caller that widens `B` entries into `B · times` hashes — a key pair into its
    `len` chains, a FORS signature into its `k` trees — has to widen whatever
    varies per entry alongside them, and `pk_seed` varies per entry as soon as the
    batch spans more than one public key. A shared `[k]` operand passes through
    untouched, since `_batched` broadcasts it to whatever the row count turns out
    to be.
    """
    array = fnp.asarray(value, dtype=dtype)
    if array.ndim == 1:
        return array
    return fnp.repeat(array, times, axis=0)


@dataclass(frozen=True)
class _Sha2Hash:
    """A SHA-2 hash together with the compression-block size its padding needs.

    The pair travels rather than the hash alone because §11.2.2 runs `PRF` and
    `F` on one hash and the other four functions on another, and each function
    pads `PK.seed` out to *its* hash's block. Reading the width off the hash at
    the point of use would mean two lookups per call for a value fixed at
    construction.
    """

    byte_hash: ByteHash
    block_size: int

    def digest(self, messages: Array) -> Array:
        return fnp.asarray(self.byte_hash.digest(messages), dtype=fnp.uint8)


class Sha2TweakableHash:
    """The SHA-2 instantiation — FIPS 205 §11.2.1 and §11.2.2.

    §11.2.1 is security category 1 (SLH-DSA-SHA2-128s and -128f), which reaches
    every function with SHA-256 alone. §11.2.2 is categories 3 and 5, which keep
    SHA-256 for `PRF` and `F` and move `H`, `T_l`, `PRF_msg` and `H_msg` to
    SHA-512 — `toByte(0, 128 − n)` for the two tweaked ones, and SHA-512's
    128-byte block for HMAC and MGF1. So the family is over *two* hashes, and
    `wide` is the one that differs: leave it unset for §11.2.1, where every
    function shares the single hash.

    Which functions move is a strength argument rather than an arbitrary split.
    `PRF` and `F` are preimage-bound on n-byte inputs, so SHA-256 still covers
    them at n = 24 and 32; `H`, `T_l`, `PRF_msg` and `H_msg` are the
    collision-bound ones, and a category-3 or -5 set claims more collision
    strength than a 256-bit digest has.

    `F`, `H`, `T_l` and `PRF` are the same construction, differing only in which
    hash runs it and what follows the address: `Trunc_n(H(PK.seed ‖
    toByte(0, blocksize − n) ‖ ADRS^c ‖ <input>))`. They stay separate names
    because the callers and the standard name them separately, and because the
    SHAKE instantiation does not collapse them.

    The zero padding after `PK.seed` is not decoration: it pads the seed to
    exactly one compression block, so an implementation may precompute that
    block's midstate once per key and resume it per call.
    """

    def __init__(
        self,
        byte_hash: ByteHash,
        *,
        n: int,
        m: int,
        block_size: int | None = None,
        wide: ByteHash | None = None,
    ) -> None:
        """`byte_hash` runs `PRF` and `F`; `wide` runs the rest.

        `wide` unset is §11.2.1: one hash for all six functions. Setting it is
        §11.2.2, and the caller that does so is `slh_dsa.sha2_params`, which reads
        the security category off the parameter set.

        `block_size` overrides hash-frx's table for `byte_hash` alone, because it
        is the hash a caller wraps — a test double that counts calls, a bench that
        measures them. Nothing wraps `wide`, so it takes the table's answer.
        """
        # `PRF` and `F` truncate to n, and so does every function `wide` runs, so
        # every hash given has to reach n. Checked on the arguments rather than on
        # the pairs below: it names the role that is short, and it leaves an unset
        # `wide` out of it — where `wide` is `byte_hash` again, already checked.
        for role, given in (("byte_hash", byte_hash), ("wide", wide)):
            if given is not None and given.digest_size < n:
                raise ValueError(
                    f"n={n} exceeds the {given.digest_size}-byte digest "
                    f"{role} truncates"
                )
        # The hash's compression-block size, which HMAC and the seed padding both
        # need. `ByteHash` does not carry it, so it comes from hash-frx's table —
        # which answers 64 for SHA-256 and 128 for SHA-512, the pair §11.2.2's
        # categories 3 and 5 need.
        self._narrow = _Sha2Hash(
            byte_hash,
            block_size if block_size is not None else block_size_of(byte_hash),
        )
        self._wide = (
            self._narrow if wide is None else _Sha2Hash(wide, block_size_of(wide))
        )
        self.n = n
        self.m = m
        self.compressed_address = True
        self.dtype = fnp.uint8

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sha2TweakableHash):
            return NotImplemented
        return (
            self._narrow == other._narrow
            and self._wide == other._wide
            and self.n == other.n
            and self.m == other.m
        )

    def __hash__(self) -> int:
        return hash((type(self), self._narrow, self._wide, self.n, self.m))

    # -- the tweaked family ------------------------------------------------

    def prf(self, pk_seed: ArrayLike, sk_seed: ArrayLike, adrs: ArrayLike) -> Array:
        return self._tweak(self._narrow, pk_seed, adrs, sk_seed)

    def f(self, pk_seed: ArrayLike, adrs: ArrayLike, m1: ArrayLike) -> Array:
        return self._tweak(self._narrow, pk_seed, adrs, m1)

    def h(self, pk_seed: ArrayLike, adrs: ArrayLike, m2: ArrayLike) -> Array:
        return self._tweak(self._wide, pk_seed, adrs, m2)

    def t(self, pk_seed: ArrayLike, adrs: ArrayLike, messages: ArrayLike) -> Array:
        return self._tweak(self._wide, pk_seed, adrs, messages)

    def prf_msg(
        self, sk_prf: ArrayLike, opt_rand: ArrayLike, message: ArrayLike
    ) -> Array:
        payload = fnp.concatenate(
            [
                fnp.asarray(opt_rand, dtype=fnp.uint8),
                fnp.asarray(message, dtype=fnp.uint8),
            ]
        )
        return self._hmac(sk_prf, payload)[: self.n]

    def h_msg(
        self,
        randomizer: ArrayLike,
        pk_seed: ArrayLike,
        pk_root: ArrayLike,
        message: ArrayLike,
    ) -> Array:
        messages = fnp.asarray(message, dtype=fnp.uint8)
        if messages.ndim == 1:
            messages = messages[None, :]
        batch = messages.shape[0]
        randomizers = batched(randomizer, batch)
        seeds = batched(pk_seed, batch)
        inner = self._wide.digest(
            fnp.concatenate(
                [randomizers, seeds, batched(pk_root, batch), messages], axis=-1
            )
        )
        return self._mgf1(fnp.concatenate([randomizers, seeds, inner], axis=-1), self.m)

    # -- the constructions the family is built from ------------------------

    def _tweak(
        self,
        sha2: _Sha2Hash,
        pk_seed: ArrayLike,
        adrs: ArrayLike,
        payload: ArrayLike,
    ) -> Array:
        # The addresses set the batch: there is one hash per position, and both
        # the seed and the payload may be shared across them. `PRF` is the case
        # that makes that concrete — one secret seed, one hash per chain.
        addresses = fnp.asarray(adrs, dtype=fnp.uint8)
        if addresses.ndim == 1:
            addresses = addresses[None, :]
        batch = addresses.shape[0]
        payloads = batched(payload, batch)
        seeds = batched(pk_seed, batch)
        padding = fnp.zeros((batch, sha2.block_size - self.n), dtype=fnp.uint8)
        return sha2.digest(
            fnp.concatenate([seeds, padding, addresses, payloads], axis=-1)
        )[:, : self.n]

    def _mgf1(self, seed: Array, length: int) -> Array:
        """`H_msg`'s outer MGF1 — RFC 8017 §B.2.1, over the wide hash.

        Wide for the reason `_hmac` is: §11.2.2 moves `H_msg` to SHA-512 with the
        rest, and §11.2.1 leaves the two the same hash.
        """
        return fnp.asarray(
            Mgf1(self._wide.byte_hash, length).digest(seed), dtype=fnp.uint8
        )

    def _hmac(self, key: ArrayLike, message: Array) -> Array:
        """`PRF_msg`'s hash and block size, over `hmac`.

        The wide one at categories 3 and 5: §11.2.2 moves `PRF_msg` to SHA-512
        along with the tweaked functions, so HMAC's block size moves with it.
        """
        return hmac(
            self._wide.byte_hash, key, message, block_size=self._wide.block_size
        )


class ShakeTweakableHash:
    """The SHAKE instantiation — FIPS 205 §11.1.

    One construction for all six functions, at every security category:
    SHAKE256 over the concatenated operands, squeezed to the length wanted.
    Where §11.2 needs MGF1 to reach `m` bytes, HMAC for `PRF_msg`, a zero pad
    filling a compression block, and a second hash at categories 3 and 5, this
    needs none of them — an extendable output already produces any length, so
    the construction *is* the concatenation.

    That is why `slh_dsa.shake` names one hash where `slh_dsa.sha2` names two at
    categories 3 and 5: there is no second hash to reach for.

    The address is the full 32 bytes rather than §11.2's 22-byte compression,
    which is what `compressed_address = False` tells a caller.
    """

    def __init__(self, xof: Callable[[int], ByteHash], *, n: int, m: int) -> None:
        """`xof` yields a `ByteHash` of the requested output length.

        A `Shake256` class is exactly that callable, and so is its host sibling,
        which is what keeps this from naming a concrete hash. Two lengths are
        wanted and they are fixed by the parameter set, so both are built once
        here rather than per call: an XOF at two lengths is two hashes, not one
        hash asked twice.
        """
        self._chain = xof(n)  # PRF, F, H, T_l, PRF_msg
        self._message = xof(m)  # H_msg
        self.n = n
        self.m = m
        self.compressed_address = False
        self.dtype = fnp.uint8

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShakeTweakableHash):
            return NotImplemented
        return (self._chain, self._message) == (other._chain, other._message)

    def __hash__(self) -> int:
        return hash((type(self), self._chain, self._message))

    # -- the tweaked family ------------------------------------------------

    def prf(self, pk_seed: ArrayLike, sk_seed: ArrayLike, adrs: ArrayLike) -> Array:
        return self._tweak(pk_seed, adrs, sk_seed)

    def f(self, pk_seed: ArrayLike, adrs: ArrayLike, m1: ArrayLike) -> Array:
        return self._tweak(pk_seed, adrs, m1)

    def h(self, pk_seed: ArrayLike, adrs: ArrayLike, m2: ArrayLike) -> Array:
        return self._tweak(pk_seed, adrs, m2)

    def t(self, pk_seed: ArrayLike, adrs: ArrayLike, messages: ArrayLike) -> Array:
        return self._tweak(pk_seed, adrs, messages)

    def prf_msg(
        self, sk_prf: ArrayLike, opt_rand: ArrayLike, message: ArrayLike
    ) -> Array:
        payload = fnp.concatenate(
            [
                fnp.asarray(sk_prf, dtype=fnp.uint8),
                fnp.asarray(opt_rand, dtype=fnp.uint8),
                fnp.asarray(message, dtype=fnp.uint8),
            ]
        )
        return self._digest(self._chain, payload[None, :])[0]

    def h_msg(
        self,
        randomizer: ArrayLike,
        pk_seed: ArrayLike,
        pk_root: ArrayLike,
        message: ArrayLike,
    ) -> Array:
        messages = fnp.asarray(message, dtype=fnp.uint8)
        if messages.ndim == 1:
            messages = messages[None, :]
        batch = messages.shape[0]
        return self._digest(
            self._message,
            fnp.concatenate(
                [
                    batched(randomizer, batch),
                    batched(pk_seed, batch),
                    batched(pk_root, batch),
                    messages,
                ],
                axis=-1,
            ),
        )

    # -- the construction the family is built from -------------------------

    def _tweak(self, pk_seed: ArrayLike, adrs: ArrayLike, payload: ArrayLike) -> Array:
        # The addresses set the batch, as in the SHA-2 family: one hash per
        # position, with the seed and the payload possibly shared across them.
        # No truncation — the XOF was asked for exactly `n` bytes.
        addresses = fnp.asarray(adrs, dtype=fnp.uint8)
        if addresses.ndim == 1:
            addresses = addresses[None, :]
        batch = addresses.shape[0]
        return self._digest(
            self._chain,
            fnp.concatenate(
                [batched(pk_seed, batch), addresses, batched(payload, batch)],
                axis=-1,
            ),
        )

    def _digest(self, byte_hash: ByteHash, messages: Array) -> Array:
        return fnp.asarray(byte_hash.digest(messages), dtype=fnp.uint8)
