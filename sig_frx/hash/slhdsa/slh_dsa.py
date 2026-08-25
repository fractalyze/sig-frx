# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""SLH-DSA — FIPS 205 §9 and §10, assembled over the components beneath it.

The scheme itself is short, because everything it needs already exists: the
tweakable hash family, WOTS+, the Merkle tree, FORS and the hypertree. What is
left is the assembly the standard describes — derive a randomizer, hash the
message, split the digest into the FORS indices and the tree and leaf indices,
sign the digest with FORS, and certify the FORS public key with the hypertree.

**The digest split is the part worth reading twice.** `md` is signed by FORS while
the bytes after it choose *which* FORS key signs it, so a slice off by one byte
produces a signature over the right digest under the wrong key — self-consistent,
and rejected by every published vector with no hint as to why. The three widths
come out of §9.2 and are pinned against Table 2's own `m` column.

**A parameter set is a table row, not an implementation.** The six SHA-2 sets
differ only in the constants in `SHA2_PARAMETER_SETS`, and Table 2 gives each row
to two names — `SLH-DSA-SHA2-<suffix>` and `SLH-DSA-SHAKE-<suffix>` — because the
two families are the same parameters over a different tweakable hash. So the SHAKE
sets are these rows under a SHAKE instantiation rather than a second scheme.

Security category 1 is what is constructible here. §11.2.1 reaches every function
of the family with SHA-256 alone, while categories 3 and 5 keep SHA-256 for `PRF`
and `F` and move `H`, `T_l` and `PRF_msg` to SHA-512 (§11.2.2) — a family over two
hashes, so those sets need a SHA-512 `ByteHash` before `sha2` can build them.

**Verification takes the batch and signing does not**, the asymmetry every layer
below carries. Each entry brings its own public key, so `PK.seed` and `PK.root`
vary across the batch, and the digest — which picks the FORS key — varies with
them: the batch shares nothing but the context. Of the indices the digest yields,
the leaf index is a column and the tree index stays bytes — 54 to 64 bits do not
fit an array lane, and an address slot takes bytes anyway (see `bytestring`).

**Three interfaces, one of them on the seam.** The seam is §10's pure external
operation (Algorithms 22 and 24), which is what an application wants. The other
two sign a different message and so live under their own names: HashSLH-DSA
(Algorithms 23 and 25, `hash_sign` / `hash_verify`) prepends domain separator 1 and
the pre-hash function's OID, and §9's internal interface (Algorithms 19 and 20,
`sign_internal` / `verify_internal`) prepends nothing at all. Keeping them apart is
the point rather than an inconvenience — the separator exists so a pre-hash
signature cannot verify as a pure one — and the internal pair is also what the
validation program publishes vectors against.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike
from hash_frx import Sha256, Shake256

from sig_frx import context as ctx
from sig_frx import prehash
from sig_frx.hash import bytestring, tree, wots
from sig_frx.hash.slhdsa import fors, hypertree, xmss
from sig_frx.hash.tweakable import (
    Sha2TweakableHash,
    ShakeTweakableHash,
    TweakableHash,
)
from sig_frx.signature import Signature

# The domain separators of §10.2: zero for pure signing, one for pre-hash signing.
# They exist to stop a pre-hash signature from verifying as a pure one and vice
# versa, so they are the first byte of what gets signed rather than metadata.
_PURE_DOMAIN = 0
_PREHASH_DOMAIN = 1


@dataclass(frozen=True)
class SlhDsaParams:
    """One row of FIPS 205 Table 2 — the six values a parameter set fixes.

    Table 2 also lists `h'`, `m`, and the key and signature sizes, and its
    footnote says outright that those are computed from these. So they are derived
    here and the tests pin the derivations against the published columns: a table
    that restated them would be six chances to mistype a value the standard
    already determines.
    """

    n: int  # security parameter and hash output length, in bytes
    h: int  # total height of the hypertree
    d: int  # layers of XMSS trees in the hypertree
    a: int  # height of one FORS tree
    k: int  # FORS trees
    lg_w: int = 4  # bits per WOTS+ chain digit; every defined set uses 4

    def __post_init__(self) -> None:
        if self.h % self.d:
            raise ValueError(
                f"the hypertree's {self.d} layers each hold one XMSS tree of "
                f"height h / d, so d must divide h = {self.h}"
            )

    @cached_property
    def tree_height(self) -> int:
        """`h'` — the height of one XMSS tree."""
        return self.h // self.d

    @cached_property
    def security_category(self) -> int:
        """The claimed category — §11: `n` alone decides it.

        Which matters because it selects the hash instantiation, not just the
        strength: category 1 is §11.2.1's single-hash family, and categories 3
        and 5 are §11.2.2's two-hash one.
        """
        categories = {16: 1, 24: 3, 32: 5}
        if self.n not in categories:
            raise ValueError(f"no Table 2 parameter set has n = {self.n}")
        return categories[self.n]

    @cached_property
    def md_bytes(self) -> int:
        """`ceil(k·a/8)` — the digest prefix FORS signs, Algorithm 19 line 6."""
        return -(-(self.k * self.a) // 8)

    @cached_property
    def tree_index_bytes(self) -> int:
        """`ceil((h − h/d)/8)` — the slice `idx_tree` comes from, line 7.

        The same width `hypertree` carries an index in, which is not a coincidence:
        this slice is what it starts from.
        """
        return self.hypertree_params.tree_index_bytes

    @cached_property
    def leaf_index_bytes(self) -> int:
        """`ceil(h/(8d))` — the slice `idx_leaf` comes from, line 8."""
        return -(-self.h // (8 * self.d))

    @cached_property
    def m(self) -> int:
        """`m` — the message digest `H_msg` produces, §9.2.

        Exactly the three slices the split consumes. §9.2 notes that only `h + k·a`
        of its bits are used and that a slightly larger digest simplifies
        implementation, which is why this is a sum of byte-rounded widths rather
        than of bit widths.
        """
        return self.md_bytes + self.tree_index_bytes + self.leaf_index_bytes

    @cached_property
    def wots_params(self) -> wots.WotsParams:
        return wots.WotsParams(n=self.n, lg_w=self.lg_w)

    @cached_property
    def fors_params(self) -> fors.ForsParams:
        return fors.ForsParams(n=self.n, a=self.a, k=self.k)

    @cached_property
    def hypertree_params(self) -> hypertree.HypertreeParams:
        return hypertree.HypertreeParams(
            layers=self.d, tree_height=self.tree_height, wots=self.wots_params
        )

    @cached_property
    def seed_size(self) -> int:
        """The three `n`-byte values key generation takes, in the standard's order."""
        return 3 * self.n

    @cached_property
    def public_key_size(self) -> int:
        """`PK.seed ‖ PK.root` — Figure 16."""
        return 2 * self.n

    @cached_property
    def secret_key_size(self) -> int:
        """`SK.seed ‖ SK.prf ‖ PK.seed ‖ PK.root` — Figure 15."""
        return 4 * self.n

    @cached_property
    def signature_size(self) -> int:
        """`(1 + k(1 + a) + h + d·len)·n` — Figure 17, and Algorithm 20 line 1."""
        return (
            1 + self.k * (1 + self.a) + self.h + self.d * self.wots_params.len
        ) * self.n


# The SHA-2 half of Table 2. Each row also names `SLH-DSA-SHAKE-<suffix>`, whose
# values are identical — the families differ in the tweakable hash, not here.
SHA2_PARAMETER_SETS: dict[str, SlhDsaParams] = {
    "SLH-DSA-SHA2-128s": SlhDsaParams(n=16, h=63, d=7, a=12, k=14),
    "SLH-DSA-SHA2-128f": SlhDsaParams(n=16, h=66, d=22, a=6, k=33),
    "SLH-DSA-SHA2-192s": SlhDsaParams(n=24, h=63, d=7, a=14, k=17),
    "SLH-DSA-SHA2-192f": SlhDsaParams(n=24, h=66, d=22, a=8, k=33),
    "SLH-DSA-SHA2-256s": SlhDsaParams(n=32, h=64, d=8, a=14, k=22),
    "SLH-DSA-SHA2-256f": SlhDsaParams(n=32, h=68, d=17, a=9, k=35),
}

# The SHAKE half of the same rows, derived rather than transcribed a second time.
# Table 2 gives one row two names and one set of values, so writing the six out
# again would be six chances for the copies to disagree about a constant the
# standard states once.
SHAKE_PARAMETER_SETS: dict[str, SlhDsaParams] = {
    name.replace("SHA2", "SHAKE"): params
    for name, params in SHA2_PARAMETER_SETS.items()
}


@dataclass(frozen=True)
class _SecretKey:
    """`SK.seed ‖ SK.prf ‖ PK.seed ‖ PK.root` — Figure 15, parsed.

    Plain data: it is built and consumed inside one call and never crosses a `jit`
    or `vmap` boundary, so it is not a registered pytree.
    """

    seed: Array
    prf: Array
    pk_seed: Array
    pk_root: Array


class SlhDsa:
    """SLH-DSA over an injected tweakable hash family — FIPS 205 §9 and §10.

    The family carries the hash and the sizes, so this class is the assembly and
    the encodings and nothing about SHA-2 or SHAKE. Build one with `sha2` rather
    than by hand unless a test wants parameters no published set has.
    """

    def __init__(
        self,
        tweak: TweakableHash,
        params: SlhDsaParams,
        *,
        deterministic: bool = False,
    ) -> None:
        if tweak.n != params.n or tweak.m != params.m:
            raise ValueError(
                f"the family is sized (n={tweak.n}, m={tweak.m}) and the parameter "
                f"set (n={params.n}, m={params.m}); a mismatch silently signs a "
                f"digest of the wrong length"
            )
        self._tweak = tweak
        self.params = params
        self.deterministic = deterministic
        self.public_key_size = params.public_key_size
        self.secret_key_size = params.secret_key_size
        self.signature_max_size = params.signature_size

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SlhDsa):
            return NotImplemented
        return (
            self._tweak == other._tweak
            and self.params == other.params
            and self.deterministic == other.deterministic
        )

    def __hash__(self) -> int:
        return hash((type(self), self._tweak, self.params, self.deterministic))

    # -- key generation ----------------------------------------------------

    def keygen(self, seed: ArrayLike) -> tuple[Array, Array]:
        """`slh_keygen_internal` — Algorithm 18.

        `seed` is `SK.seed ‖ SK.prf ‖ PK.seed`, the order Algorithm 21 generates
        them in and the order the published vectors concatenate them in. The
        standard's Algorithm 21 draws those three itself; the seam takes them,
        because a key that cannot be regenerated from published bytes cannot be
        gated on published bytes.
        """
        values = fnp.asarray(seed, dtype=fnp.uint8)
        if values.shape != (self.params.seed_size,):
            raise ValueError(
                f"keygen takes SK.seed, SK.prf and PK.seed — {self.params.n} bytes "
                f"each, {self.params.seed_size} in all — got shape "
                f"{tuple(values.shape)}"
            )
        n = self.params.n
        sk_seed, sk_prf, pk_seed = values[:n], values[n : 2 * n], values[2 * n :]
        pk_root = self._top_root(pk_seed, sk_seed)
        return (
            fnp.concatenate([pk_seed, pk_root]),
            fnp.concatenate([sk_seed, sk_prf, pk_seed, pk_root]),
        )

    def _top_root(self, pk_seed: ArrayLike, sk_seed: ArrayLike) -> Array:
        """Algorithm 18 lines 1 to 3: the root of the top layer's only XMSS tree."""
        return xmss.root(
            self._tweak,
            self.params.wots_params,
            pk_seed,
            sk_seed,
            tree.TreePosition(layer=self.params.d - 1, tree=0),
            self.params.tree_height,
        )

    # -- signing -----------------------------------------------------------

    def sign(
        self,
        secret_key: ArrayLike,
        message: ArrayLike,
        *,
        randomness: ArrayLike | None = None,
        context: ArrayLike | None = None,
    ) -> Array:
        """`slh_sign` — Algorithm 22: pure signing, over Algorithm 19."""
        return self.sign_internal(
            secret_key,
            ctx.prepend(ctx.prefix(_PURE_DOMAIN, context), message),
            randomness=randomness,
        )

    def hash_sign(
        self,
        secret_key: ArrayLike,
        message: ArrayLike,
        pre_hash: prehash.PreHash,
        *,
        randomness: ArrayLike | None = None,
        context: ArrayLike | None = None,
    ) -> Array:
        """`hash_slh_sign` — Algorithm 23: pre-hash signing, over Algorithm 19.

        Deliberately not on the `Signature` seam. It signs a different message —
        domain separator one, and the pre-hash function's OID before the digest —
        so this and `sign` over the same content produce signatures that do not
        verify as each other, which is the separation §10.2.1 introduces it for.
        """
        return self.sign_internal(
            secret_key,
            ctx.prepend(
                pre_hash.prefix(_PREHASH_DOMAIN, context), pre_hash.digest(message)
            ),
            randomness=randomness,
        )

    def sign_internal(
        self,
        secret_key: ArrayLike,
        message: ArrayLike,
        *,
        randomness: ArrayLike | None = None,
    ) -> Array:
        """`slh_sign_internal` — Algorithm 19.

        §9's interface: the message is signed as given, without §10's domain
        separator and context, which is what makes it the one the validation
        program drives. Not on the seam, for the reason §10 gives — an application
        wants the domain separation, and a signature over an unwrapped message is a
        different object from one over `M'`.
        """
        key = self._parse_secret_key(secret_key)
        body = fnp.asarray(message, dtype=fnp.uint8)
        randomizer = self._tweak.prf_msg(key.prf, self._opt_rand(key, randomness), body)
        digests, tree_indices, leaf_indices = self._split_digest(
            randomizer, key.pk_seed, key.pk_root, body
        )
        # Lines 11 to 13: the digest picks the FORS key, by tree and by key pair.
        # The signer's seam: one signature on the host, where a Python integer holds
        # the index at any width, so this is the one place the bytes become a number.
        # The digest lands here with them, because the components below read their
        # FORS indices out of it and address their nodes with that same integer —
        # which is wider than the lane a device holds one in.
        md = np.asarray(digests)
        tree_index = int.from_bytes(bytes(np.asarray(tree_indices)[0]), "big")
        leaf_index = int(leaf_indices[0])
        position = fors.ForsPosition(tree=tree_index, key_pair=leaf_index)
        fors_signature = fors.sign(
            self._tweak,
            self.params.fors_params,
            md[0],
            key.pk_seed,
            key.seed,
            position,
        )
        # Line 16: the hypertree signs the FORS *public* key, which the verifier
        # recomputes from the FORS signature rather than being handed.
        fors_key = fors.pk_from_sig(
            self._tweak,
            self.params.fors_params,
            fnp.asarray(fors_signature)[None, ...],
            md,
            key.pk_seed,
            position,
        )[0]
        hypertree_signature = hypertree.sign(
            self._tweak,
            self.params.hypertree_params,
            fors_key,
            key.pk_seed,
            key.seed,
            tree_index,
            leaf_index,
        )
        return fnp.concatenate(
            [
                fnp.asarray(randomizer, dtype=fnp.uint8),
                fnp.asarray(fors_signature, dtype=fnp.uint8).reshape(-1),
                fnp.asarray(hypertree_signature, dtype=fnp.uint8).reshape(-1),
            ]
        )

    def _opt_rand(self, key: _SecretKey, randomness: ArrayLike | None) -> Array:
        """`opt_rand` — Algorithm 19 line 2, and the two variants §9.2 describes.

        The hedged variant takes `addrnd` from the caller and the deterministic one
        substitutes `PK.seed`. A hedged instance handed nothing to hedge with is an
        error: falling back to `PK.seed` would silently produce the deterministic
        signature, which verifies, so nothing downstream would notice.
        """
        if self.deterministic:
            return fnp.asarray(key.pk_seed, dtype=fnp.uint8)
        if randomness is None:
            raise ValueError(
                "a hedged instance signs with caller-supplied `addrnd`: pass "
                "`randomness`, or build the deterministic variant"
            )
        values = fnp.asarray(randomness, dtype=fnp.uint8)
        if values.shape != (self.params.n,):
            raise ValueError(
                f"`addrnd` is {self.params.n} bytes, got shape {tuple(values.shape)}"
            )
        return values

    # -- verification ------------------------------------------------------

    def verify(
        self,
        public_key: ArrayLike,
        message: ArrayLike,
        signature: ArrayLike,
        *,
        context: ArrayLike | None = None,
    ) -> Array:
        """`slh_verify` — Algorithm 24 over Algorithm 20, for the whole batch."""
        return self.verify_internal(
            public_key,
            ctx.prepend(ctx.prefix(_PURE_DOMAIN, context), message),
            signature,
        )

    def hash_verify(
        self,
        public_key: ArrayLike,
        message: ArrayLike,
        signature: ArrayLike,
        pre_hash: prehash.PreHash,
        *,
        context: ArrayLike | None = None,
    ) -> Array:
        """`hash_slh_verify` — Algorithm 25: the counterpart of `hash_sign`."""
        return self.verify_internal(
            public_key,
            ctx.prepend(
                pre_hash.prefix(_PREHASH_DOMAIN, context), pre_hash.digest(message)
            ),
            signature,
        )

    def verify_internal(
        self, public_key: ArrayLike, messages: ArrayLike, signature: ArrayLike
    ) -> Array:
        """`slh_verify_internal` — Algorithm 20, with the batch axis added.

        §9's interface: the message is verified as given, without §10's domain
        separator and context. It is what the validation program tests against, and
        it is not on the seam — a caller who reaches for it is naming the scheme
        anyway, and passing an unwrapped message to the external operation would
        verify something other than what the standard says it verifies.

        Lines 1 to 3 reject a signature whose length is wrong. A batch carries one
        static length, so a wrong one is every entry's verdict rather than an
        error: the published vectors include signatures a byte short and a byte
        long, and each expects a rejection. A wrong *rank*, or a batch that does
        not line up, stays an error — that is a caller mistake and not a verdict
        the standard defines.
        """
        params = self.params
        keys = fnp.asarray(public_key, dtype=fnp.uint8)
        if keys.ndim != 2 or keys.shape[1] != params.public_key_size:
            raise ValueError(
                f"a public key batch is [B, {params.public_key_size}], got shape "
                f"{tuple(keys.shape)}"
            )
        batch = keys.shape[0]
        signatures = fnp.asarray(signature, dtype=fnp.uint8)
        if signatures.ndim != 2 or signatures.shape[0] != batch:
            raise ValueError(
                f"one signature per public key, as a [B, {params.signature_size}] "
                f"batch: got {batch} keys and signatures of shape "
                f"{tuple(signatures.shape)}"
            )
        messages = fnp.asarray(messages, dtype=fnp.uint8)
        if messages.ndim != 2 or messages.shape[0] != batch:
            raise ValueError(
                f"one message per public key, as a [B, L] batch: got {batch} keys "
                f"and messages of shape {tuple(messages.shape)}"
            )
        if signatures.shape[1] != params.signature_size:
            # Algorithm 20 lines 1 to 3.
            return fnp.zeros(batch, dtype=bool)

        pk_seeds, pk_roots = keys[:, : params.n], keys[:, params.n :]
        randomizers, fors_signatures, hypertree_signatures = self._parse_signature(
            signatures
        )
        md, tree_indices, leaf_indices = self._split_digest(
            randomizers, pk_seeds, pk_roots, messages
        )
        # A candidate FORS public key per entry, then one hypertree walk that
        # accepts an entry only if its candidate climbs to that entry's own root.
        fors_keys = fors.pk_from_sig(
            self._tweak,
            params.fors_params,
            fors_signatures,
            md,
            pk_seeds,
            fors.ForsPosition(tree=tree_indices, key_pair=leaf_indices),
        )
        return hypertree.verify(
            self._tweak,
            params.hypertree_params,
            hypertree_signatures,
            fors_keys,
            pk_seeds,
            tree_indices,
            leaf_indices,
            pk_roots,
        )

    # -- the encodings -----------------------------------------------------

    def _parse_secret_key(self, secret_key: ArrayLike) -> _SecretKey:
        """Figure 15, read back."""
        values = fnp.asarray(secret_key, dtype=fnp.uint8)
        if values.shape != (self.params.secret_key_size,):
            raise ValueError(
                f"a secret key is {self.params.secret_key_size} bytes, got shape "
                f"{tuple(values.shape)}"
            )
        n = self.params.n
        return _SecretKey(
            seed=values[:n],
            prf=values[n : 2 * n],
            pk_seed=values[2 * n : 3 * n],
            pk_root=values[3 * n :],
        )

    def _parse_signature(self, signatures: Array) -> tuple[Array, Array, Array]:
        """`getR`, `getSIG_FORS`, `getSIG_HT` — Algorithm 20 lines 5 to 7.

        The two signature parts come back in the shapes their components take
        rather than as byte ranges: `[B, k, a + 1, n]` for FORS, and
        `[B, d, len + h', n]` for the hypertree, lowest layer first.
        """
        params = self.params
        n = params.n
        batch = signatures.shape[0]
        fors_end = n + params.k * (1 + params.a) * n
        return (
            signatures[:, :n],
            signatures[:, n:fors_end].reshape(batch, params.k, params.a + 1, n),
            signatures[:, fors_end:].reshape(
                batch, params.d, params.hypertree_params.signature_values, n
            ),
        )

    def _split_digest(
        self,
        randomizers: ArrayLike,
        pk_seeds: ArrayLike,
        pk_roots: ArrayLike,
        messages: ArrayLike,
    ) -> tuple[Array, bytestring.ByteString, bytestring.ByteString]:
        """Algorithm 19 lines 5 to 10 — the same lines as Algorithm 20's 8 to 13.

        One `H_msg` over the whole batch, then the split: `md` as
        `[B, ceil(k·a/8)]`, the tree index as `[B, tree_index_bytes]` **bytes**,
        and the leaf index as a `[B]` column.

        **The tree index is never made into a number** — see `bytestring`. The
        leaf index is `h'` bits, at most 9, so that one is a column.

        The reduction is what keeps `idx_tree` inside the hypertree: the slice is
        byte-rounded, so it carries up to seven bits more than `h − h'` and those
        bits are not part of the index.

        The digest is left where `H_msg` produced it. All three slices are static,
        so a traced digest splits the same way a concrete one does — and the split
        is the head of the verification path, so pulling it to the host here would
        settle for everything below it.
        """
        params = self.params
        digest = fnp.asarray(
            self._tweak.h_msg(randomizers, pk_seeds, pk_roots, messages),
            dtype=fnp.uint8,
        )
        tree_end = params.md_bytes + params.tree_index_bytes
        leaf_end = tree_end + params.leaf_index_bytes
        return (
            digest[:, : params.md_bytes],
            bytestring.mask_to(
                digest[:, params.md_bytes : tree_end], params.h - params.tree_height
            ),
            bytestring.low_bits(digest[:, tree_end:leaf_end], params.tree_height),
        )


def sha2_params(params: SlhDsaParams, *, deterministic: bool = False) -> SlhDsa:
    """§11.2.1's SHA-256-only family at `params`, whether Table 2 names it or not.

    `sha2` is this with a Table 2 lookup in front of it. The split exists because
    a scheme may define a parameter set of its own — SHRINCS's stateless
    component is the one in this repo — and the alternative is assembling
    `SlhDsa` and a tweakable hash at the call site, which copies the one fact
    this module should own: which hash family goes with which security category.
    That fact is about to have a second answer, since categories 3 and 5 want
    SHA-512, and a copy would not be found when it does.
    """
    if params.security_category != 1:
        raise NotImplementedError(
            f"security category {params.security_category} hashes H, T_l and "
            f"PRF_msg with SHA-512 (FIPS 205 §11.2.2); this builds §11.2.1's "
            f"SHA-256-only family"
        )
    return SlhDsa(
        Sha2TweakableHash(Sha256(), n=params.n, m=params.m),
        params,
        deterministic=deterministic,
    )


def sha2(name: str, *, deterministic: bool = False) -> SlhDsa:
    """The `SLH-DSA-SHA2-*` parameter set called `name`, over hash-frx's SHA-256.

    §11.2.1's family reaches `PRF`, `PRF_msg`, `F`, `H`, `T_l` and `H_msg` with
    SHA-256 alone, which is every function the security category 1 sets need.
    Categories 3 and 5 keep SHA-256 for `PRF` and `F` but hash `H`, `T_l` and
    `PRF_msg` with SHA-512 (§11.2.2), so they are a family over two hashes and
    need a SHA-512 `ByteHash` rather than a different constant here.
    """
    if name not in SHA2_PARAMETER_SETS:
        raise ValueError(f"{name!r} is not one of {sorted(SHA2_PARAMETER_SETS)}")
    return sha2_params(SHA2_PARAMETER_SETS[name], deterministic=deterministic)


def shake(name: str, *, deterministic: bool = False) -> SlhDsa:
    """The `SLH-DSA-SHAKE-*` parameter set called `name`, over hash-frx's SHAKE256.

    All six, unlike `sha2`. §11.1 reaches every function with SHAKE256 at every
    security category, because an extendable output already produces whatever
    length the function wants — where §11.2 has to change hash at categories 3
    and 5 to get one. The category is a parameter here rather than a constraint.
    """
    if name not in SHAKE_PARAMETER_SETS:
        raise ValueError(f"{name!r} is not one of {sorted(SHAKE_PARAMETER_SETS)}")
    params = SHAKE_PARAMETER_SETS[name]
    return SlhDsa(
        # `Shake256` is the callable the family wants: an XOF's constructor takes
        # the output length, so the two lengths this needs are two instances.
        ShakeTweakableHash(Shake256, n=params.n, m=params.m),
        params,
        deterministic=deterministic,
    )


def sha256_pre_hash() -> prehash.PreHash:
    """SHA-256 as HashSLH-DSA's pre-hash — Algorithm 23 lines 9 to 11.

    §10.2.2 restricts SHA-256 to parameter sets claimed in security category 1,
    which is every set `sha2` builds.
    """
    return prehash.sha2_256()


if TYPE_CHECKING:
    # The seam conformance pin: mypy fails this module if it drifts from the
    # Protocol, rather than failing the consumer that calls it.
    _: type[Signature] = SlhDsa
