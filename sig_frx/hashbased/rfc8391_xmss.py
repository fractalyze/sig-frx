# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Single-tree XMSS — RFC 8391 §4.1.

A Merkle tree of L-tree-compressed WOTS+ public keys. The public key is the root;
a signature is the index of the one-time key that made it, a randomizer, the WOTS+
signature, and the authentication path from that key's leaf.

**The index is the whole risk, and it is a value that crosses the API.** Signing
twice at one index leaks the WOTS+ secret — not a degradation, a total break — so
`sign` consumes a secret key and returns the advanced one. A caller that signs
twice under one key has to pass the spent value a second time, visibly, rather
than calling a method that silently did not move. Hiding the index in a mutable
object would put a side effect inside a traced, potentially `vmap`ped
computation, which is how one gets reused without anyone writing the reuse down.

**Where the index is written down is the caller's problem.** This repo makes no
side-channel claim about signing at all (`docs/reference/security.md`), and a
library that will not claim anything about how signing executes should not own the
durable storage that makes a stateful signer safe. What is provided is a format:
the secret key crosses the API as the reference implementation's own layout,
`idx ‖ SK.seed ‖ SK.prf ‖ root ‖ SEED`. Committing the advanced key before
releasing the signature it came from is the caller's discipline.

**`sign` is therefore not on the `Signature` seam and `verify` is.** The seam
returns one value from `sign`, and a stateful signer has to return two. Splitting
it the other way — a seam-shaped `sign` that quietly leaves the index alone — is
the exact footgun the discipline exists to remove. So this class carries no seam
conformance pin, which `signature.py` names as the expected shape for a stateful
scheme rather than an omission.

**The tree walk is `tree.py`'s, with one adjustment.** FIPS 205 addresses a Merkle
node by the height of the *parent* being computed; RFC 8391 addresses it by the
height of the *children* being consumed (`xmss_core.c`'s `treehash` sets
`tree_height` to `heights[offset - 1]`, which is zero for a pair of leaves). The
two differ by exactly one, and `NodeAddresses` is where that lands: the builders
here subtract it, and everything above them — `tree.root`, `tree.auth_path`,
`tree.root_from_path` — is shared unchanged. An off-by-one there produces a
perfectly self-consistent tree that reproduces no published root.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike
from hash_frx.sha256 import Sha256

from sig_frx.hashbased import rfc8391_adrs, rfc8391_wots, tree, wots
from sig_frx.hashbased.rfc8391_hashes import Rfc8391Hashes, sha2_hashes
from sig_frx.hashbased.rfc8391_params import XMSS_PARAMETER_SETS, XmssParams

# §4.1.11: an XMSS signature carries its index in four bytes whatever `h` is.
# XMSS-MT rounds `h` up to a byte instead, which is why this is named here rather
# than derived — the two are the same field with different widths.
INDEX_BYTES = 4

# `PRF`'s input is 32 bytes, so the index the randomizer is derived from is
# encoded to that width rather than to `INDEX_BYTES` — §4.1.9.
_PRF_INDEX_BYTES = 32


# -- the tree layer --------------------------------------------------------


def _shift_to_child_height(build: tree.NodeAddresses) -> tree.NodeAddresses:
    """Re-express `tree.py`'s parent height as RFC 8391's child height.

    `tree.py` names a node by the height it sits at, so the first merge is height
    one. RFC 8391's `treehash` names the same hash by the height of the pair it
    consumes, so the first merge is height zero. Nothing else about the walk
    differs, so the whole difference is this decrement.
    """

    def shifted(height: int, indices: np.ndarray) -> np.ndarray:
        if height < 1:
            raise ValueError(
                f"a Merkle parent sits at height 1 or above, got {height}; RFC "
                f"8391 addresses it by the height below, which would be negative"
            )
        return build(height - 1, indices)

    return shifted


def node_addresses(position: tree.TreePosition) -> tree.NodeAddresses:
    """The hash-tree addresses one tree's nodes are hashed under — §2.5 Figure 4."""

    def build(height: int, indices: np.ndarray) -> np.ndarray:
        return rfc8391_adrs.encode_batch(
            rfc8391_adrs.hash_tree(
                layer=position.layer,
                tree=position.tree,
                height=height,
                index=np.asarray(indices).reshape(-1),
            )
        )

    return _shift_to_child_height(build)


def batch_node_addresses(positions: Sequence[tree.TreePosition]) -> tree.NodeAddresses:
    """Node addresses for a batch whose entries sit in *different* trees.

    Entry `k`'s node is addressed in `positions[k]`'s tree. Usable only where the
    batch keeps one entry per position at every level — `tree.root_from_path`,
    where each signature walks its own path — and not in a whole-tree reduction,
    where the node count halves per level and the correspondence would not hold.
    """
    layers = np.array([position.layer for position in positions])
    trees = np.array([position.tree for position in positions])

    def build(height: int, indices: np.ndarray) -> np.ndarray:
        flat = np.asarray(indices).reshape(-1)
        if flat.shape[0] != len(positions):
            raise ValueError(
                f"one node per position: got {len(positions)} positions and "
                f"{flat.shape[0]} indices"
            )
        return rfc8391_adrs.encode_batch(
            rfc8391_adrs.hash_tree(layer=layers, tree=trees, height=height, index=flat)
        )

    return _shift_to_child_height(build)


def _leaf_positions(
    position: tree.TreePosition, indices: Sequence[int]
) -> tuple[list[rfc8391_adrs.Adrs], list[rfc8391_adrs.Adrs]]:
    """The OTS and L-tree addresses of the given leaves, which are not the same.

    §4.1.6 builds both from one leaf index — they share the layer and tree and
    differ in their type and in what their first word means — and `gen_leaf_wots`
    takes both, so they are carried as two lists rather than derived from each
    other.
    """
    return (
        [rfc8391_adrs.ots(position.layer, position.tree, index) for index in indices],
        [rfc8391_adrs.ltree(position.layer, position.tree, index) for index in indices],
    )


def leaves(
    hashes: Rfc8391Hashes,
    params: wots.WotsParams,
    pub_seed: ArrayLike,
    sk_seed: ArrayLike,
    position: tree.TreePosition,
    height: int,
) -> Array:
    """Every leaf of one tree — `[2^height, n]`, in one batched WOTS+ walk.

    No key pair depends on another, so all `2^height · len` chains advance
    together: `w − 1` batched hashes build every leaf, where walking key pairs one
    at a time would be `2^height` times that. The L-trees compress in one call for
    the same reason.
    """
    ots_positions, ltree_positions = _leaf_positions(position, range(1 << height))
    return rfc8391_wots.pk_gen(
        hashes,
        params,
        pub_seed,
        sk_seed,
        ots_positions,
        rfc8391_wots.ltree_compression(
            hashes, pub_seed, ltree_positions, leaves=params.len
        ),
    )


def root(
    hashes: Rfc8391Hashes,
    params: wots.WotsParams,
    pub_seed: ArrayLike,
    sk_seed: ArrayLike,
    position: tree.TreePosition,
    height: int,
) -> Array:
    """The tree's root over its WOTS+ public keys — the XMSS public key. `[n]`."""
    return tree.root(
        hashes,
        pub_seed,
        leaves(hashes, params, pub_seed, sk_seed, position, height),
        node_addresses(position),
    )


def pk_from_sig(
    hashes: Rfc8391Hashes,
    params: wots.WotsParams,
    signatures: ArrayLike,
    messages: ArrayLike,
    pub_seed: ArrayLike,
    positions: Sequence[tree.TreePosition],
    leaf_indices: Sequence[int],
) -> Array:
    """The root each `(WOTS+ signature, auth path, leaf index)` implies. `[B, n]`.

    §4.1.10's `XMSS_rootFromSig`, for a batch: each entry carries its own tree, its
    own leaf and its own message, which is what a verifier holds — `B` independent
    claims that each imply a root, compared against a root the caller already
    trusts.
    """
    parts = fnp.asarray(signatures, dtype=fnp.uint8)
    ots_positions: list[rfc8391_adrs.Adrs] = []
    ltree_positions: list[rfc8391_adrs.Adrs] = []
    for position, index in zip(positions, leaf_indices, strict=True):
        ots, ltree = _leaf_positions(position, [index])
        ots_positions += ots
        ltree_positions += ltree
    computed = rfc8391_wots.pk_from_sig(
        hashes,
        params,
        parts[:, : params.len, :],
        messages,
        pub_seed,
        ots_positions,
        rfc8391_wots.ltree_compression(
            hashes, pub_seed, ltree_positions, leaves=params.len
        ),
    )
    return tree.root_from_path(
        hashes,
        pub_seed,
        computed,
        np.asarray(leaf_indices, dtype=np.int64),
        parts[:, params.len :, :],
        batch_node_addresses(positions),
    )


# -- the scheme ------------------------------------------------------------


@dataclass(frozen=True)
class _SecretKey:
    """`idx ‖ SK.seed ‖ SK.prf ‖ root ‖ SEED` — §4.1.11, parsed.

    Plain data: built and consumed inside one call, never crossing a `jit` or
    `vmap` boundary, so it is not a registered pytree. The index is a host integer
    because it addresses a node — every component below builds its addresses on the
    host from a concrete index.
    """

    index: int
    seed: Array
    prf: Array
    root: Array
    pub_seed: Array


class Xmss:
    """XMSS over an injected hash family — RFC 8391 §4.1.

    The family carries the hash, `n` and the padding length, so this class is the
    assembly and the encodings and nothing about SHA-2 or SHAKE. Build one with
    `sha2` rather than by hand unless a test wants parameters no registered OID has.
    """

    def __init__(self, hashes: Rfc8391Hashes, params: XmssParams) -> None:
        if hashes.n != params.n or hashes.padding_len != params.padding_len:
            raise ValueError(
                f"the family is sized (n={hashes.n}, "
                f"padding_len={hashes.padding_len}) and the parameter set "
                f"(n={params.n}, padding_len={params.padding_len}); a mismatch "
                f"silently hashes under the wrong domain separator"
            )
        self._hashes = hashes
        self.params = params
        # Signing is `r = PRF(SK.prf, toByte(idx, 32))` and nothing else, so one
        # key signs one message one way — there is no hedged variant to select.
        self.deterministic = True
        self.public_key_size = 2 * params.n
        self.secret_key_size = INDEX_BYTES + 4 * params.n
        self.signature_max_size = (
            INDEX_BYTES + params.n + (params.wots.len + params.height) * params.n
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Xmss):
            return NotImplemented
        return self._hashes == other._hashes and self.params == other.params

    def __hash__(self) -> int:
        return hash((type(self), self._hashes, self.params))

    @property
    def signatures_per_key(self) -> int:
        """How many messages one key can sign — `2^h`, and not one more."""
        return 1 << self.params.height

    # -- key generation ----------------------------------------------------

    def keygen(self, seed: ArrayLike) -> tuple[Array, Array]:
        """§4.1.7's key generation, from a caller-supplied seed.

        `seed` is `SK.seed ‖ SK.prf ‖ SEED`, the order the reference's
        `xmssmt_core_seed_keypair` reads them in and the order the published
        vectors fill them in. The standard draws those three itself; taking them
        is what makes a key reproducible from published bytes.

        The public key is `root ‖ SEED` — the root *first*, which is the opposite
        of FIPS 205's `PK.seed ‖ PK.root`.
        """
        values = fnp.asarray(seed, dtype=fnp.uint8)
        n = self.params.n
        if values.shape != (3 * n,):
            raise ValueError(
                f"keygen takes SK.seed, SK.prf and SEED — {n} bytes each, {3 * n} "
                f"in all — got shape {tuple(values.shape)}"
            )
        sk_seed, sk_prf, pub_seed = values[:n], values[n : 2 * n], values[2 * n :]
        pk_root = root(
            self._hashes,
            self.params.wots,
            pub_seed,
            sk_seed,
            tree.TreePosition(layer=0, tree=0),
            self.params.height,
        )
        return (
            fnp.concatenate([pk_root, pub_seed]),
            self._encode_secret_key(_SecretKey(0, sk_seed, sk_prf, pk_root, pub_seed)),
        )

    # -- signing -----------------------------------------------------------

    def sign(
        self,
        secret_key: ArrayLike,
        message: ArrayLike,
        *,
        context: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        """§4.1.9's signing: `(signature, the secret key advanced past it)`.

        Deliberately not the seam's `sign`, which returns one value. The second
        return is the point rather than a convenience: the caller now holds a key
        that has moved, and re-signing under the one they passed in means naming
        the spent value again.

        The index is bound into what gets signed twice over — through the
        randomizer `PRF(SK.prf, toByte(idx, 32))` and through `H_msg`'s own
        `toByte(idx, n)` — so a signature cannot be replayed as one made at another
        leaf.
        """
        _reject_context(context)
        key = self._parse_secret_key(secret_key)
        if key.index >= self.signatures_per_key:
            raise ValueError(
                f"this key's {self.signatures_per_key} one-time keys are spent "
                f"(index {key.index}); signing again would reuse one, which "
                f"reveals the WOTS+ secret rather than merely repeating a signature"
            )
        body = fnp.asarray(message, dtype=fnp.uint8)
        if body.ndim != 1:
            raise ValueError(f"sign takes one message, got shape {tuple(body.shape)}")

        randomizer = self._hashes.prf(key.prf, _to_byte(key.index, _PRF_INDEX_BYTES))[0]
        digest = self._hashes.hash_message(randomizer, key.root, key.index, body)[0]

        position, leaf_index = self._locate(key.index)
        ots_position, _ = _leaf_positions(position, [leaf_index])
        wots_signature = rfc8391_wots.sign(
            self._hashes,
            self.params.wots,
            digest,
            key.pub_seed,
            key.seed,
            ots_position[0],
        )
        path = tree.auth_path(
            self._hashes,
            key.pub_seed,
            leaves(
                self._hashes,
                self.params.wots,
                key.pub_seed,
                key.seed,
                position,
                self.params.height,
            ),
            [leaf_index],
            self.params.height,
            node_addresses(position),
        )[0]
        signature = fnp.concatenate(
            [
                fnp.asarray(_to_byte(key.index, INDEX_BYTES), dtype=fnp.uint8),
                fnp.asarray(randomizer, dtype=fnp.uint8),
                fnp.asarray(wots_signature, dtype=fnp.uint8).reshape(-1),
                fnp.asarray(path, dtype=fnp.uint8).reshape(-1),
            ]
        )
        advanced = self._encode_secret_key(
            _SecretKey(key.index + 1, key.seed, key.prf, key.root, key.pub_seed)
        )
        return signature, advanced

    # -- verification ------------------------------------------------------

    def verify(
        self,
        public_key: ArrayLike,
        message: ArrayLike,
        signature: ArrayLike,
        *,
        context: ArrayLike | None = None,
    ) -> Array:
        """§4.1.10's verification, with the batch axis added: -> bool `[B]`.

        Each entry brings its own public key, its own message and its own claimed
        index, and the batch shares nothing. The index the signature carries picks
        the leaf and the tree it is checked at, so a signature re-labelled with
        another index climbs to a different root and is rejected — the index is not
        a hint the verifier may take on trust.
        """
        _reject_context(context)
        params = self.params
        keys = fnp.asarray(public_key, dtype=fnp.uint8)
        if keys.ndim != 2 or keys.shape[1] != self.public_key_size:
            raise ValueError(
                f"a public key batch is [B, {self.public_key_size}], got shape "
                f"{tuple(keys.shape)}"
            )
        batch = keys.shape[0]
        signatures = fnp.asarray(signature, dtype=fnp.uint8)
        if signatures.shape != (batch, self.signature_max_size):
            # RFC 8391 defines no verdict for a mis-sized signature the way FIPS
            # 205 §Algorithm 20 does, so this is a caller that built the batch
            # wrongly rather than a rejection the standard asks for.
            raise ValueError(
                f"a signature batch is [{batch}, {self.signature_max_size}], got "
                f"shape {tuple(signatures.shape)}"
            )
        messages = fnp.asarray(message, dtype=fnp.uint8)
        if messages.ndim != 2 or messages.shape[0] != batch:
            raise ValueError(
                f"one message per public key, as a [B, L] batch: got {batch} keys "
                f"and messages of shape {tuple(messages.shape)}"
            )

        roots, pub_seeds = keys[:, : params.n], keys[:, params.n :]
        indices, randomizers, bodies = self._parse_signature(signatures)
        digests = self._hashes.hash_message(randomizers, roots, indices, messages)
        located = [self._locate(index) for index in indices]
        computed = pk_from_sig(
            self._hashes,
            params.wots,
            bodies,
            digests,
            pub_seeds,
            [position for position, _ in located],
            [leaf for _, leaf in located],
        )
        return fnp.all(computed == roots, axis=-1)

    # -- encodings ---------------------------------------------------------

    def _locate(self, index: int) -> tuple[tree.TreePosition, int]:
        """Which tree an index sits in, and which leaf of it — §4.1.9 lines 8-9.

        Single-tree XMSS has one tree, so a well-formed index leaves the tree
        address zero. An index past the tree addresses a tree that does not exist,
        which is how an over-large index a signature claims is rejected rather than
        wrapping around onto a leaf that does exist.
        """
        height = self.params.height
        return (
            tree.TreePosition(layer=0, tree=index >> height),
            index & ((1 << height) - 1),
        )

    def _encode_secret_key(self, key: _SecretKey) -> Array:
        return fnp.concatenate(
            [
                fnp.asarray(_to_byte(key.index, INDEX_BYTES), dtype=fnp.uint8),
                fnp.asarray(key.seed, dtype=fnp.uint8),
                fnp.asarray(key.prf, dtype=fnp.uint8),
                fnp.asarray(key.root, dtype=fnp.uint8),
                fnp.asarray(key.pub_seed, dtype=fnp.uint8),
            ]
        )

    def _parse_secret_key(self, secret_key: ArrayLike) -> _SecretKey:
        values = fnp.asarray(secret_key, dtype=fnp.uint8)
        if values.shape != (self.secret_key_size,):
            raise ValueError(
                f"a secret key is {self.secret_key_size} bytes, got shape "
                f"{tuple(values.shape)}"
            )
        n = self.params.n
        fields = [
            values[INDEX_BYTES + step * n : INDEX_BYTES + (step + 1) * n]
            for step in range(4)
        ]
        return _SecretKey(
            int.from_bytes(bytes(np.asarray(values[:INDEX_BYTES])), "big"), *fields
        )

    def _parse_signature(self, signatures: Array) -> tuple[list[int], Array, Array]:
        """`idx ‖ r ‖ WOTS+ signature ‖ auth path` — §4.1.8.

        The WOTS+ signature and the path come back as one `[B, len + h, n]` block
        because that is what `pk_from_sig` consumes: it splits them itself, and
        cutting them apart twice is a chance for the two splits to disagree.
        """
        params = self.params
        indices = [
            int.from_bytes(bytes(row), "big")
            for row in np.asarray(signatures[:, :INDEX_BYTES])
        ]
        randomizers = signatures[:, INDEX_BYTES : INDEX_BYTES + params.n]
        body = signatures[:, INDEX_BYTES + params.n :]
        return (
            indices,
            randomizers,
            body.reshape(
                signatures.shape[0], params.wots.len + params.height, params.n
            ),
        )


def _reject_context(context: ArrayLike | None) -> None:
    """RFC 8391 has no application context, so one is refused rather than ignored.

    A scheme that accepted a context and dropped it would verify something other
    than what the caller asked about, with no signal.
    """
    if context is None:
        return
    values = np.asarray(context, dtype=np.uint8).reshape(-1)
    if values.shape[0]:
        raise ValueError(
            "RFC 8391 defines no application context; XMSS takes an empty one, "
            f"got {values.shape[0]} bytes"
        )


def _to_byte(value: int, length: int) -> np.ndarray:
    """`toByte(x, n)` — §2.4, as the uint8 array the concatenations take."""
    return np.frombuffer(value.to_bytes(length, "big"), dtype=np.uint8)


def sha2(oid: int) -> Xmss:
    """The `XMSS-SHA2_*` parameter set registered at `oid`, over hash-frx's SHA-256.

    The SHAKE sets and the `n = 64` SHA-2 sets are registered but not constructible
    here: they need Keccak and a SHA-512 `ByteHash` respectively, and `sha2_hashes`
    refuses them rather than building the wrong family.
    """
    if oid not in XMSS_PARAMETER_SETS:
        raise ValueError(
            f"no XMSS parameter set has OID {oid}; registered OIDs are "
            f"{min(XMSS_PARAMETER_SETS)} to {max(XMSS_PARAMETER_SETS)}"
        )
    params = XMSS_PARAMETER_SETS[oid]
    return Xmss(sha2_hashes(params, Sha256()), params)
