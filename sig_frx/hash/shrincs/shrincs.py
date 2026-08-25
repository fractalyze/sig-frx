# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""SHRINCS — one public key, two signing paths, one verifier.

A stateful path over an FXMSS tree of WOTS+C leaves, and a stateless SLH-DSA
fallback for a signer that has lost its state. A signature from either verifies
under the same 48-byte key, and **the first byte says which**: `255` selects the
stateless path, and any smaller value is the height of the WOTS+C leaf that
signed — so the indicator is a field of the stateful signature rather than a tag
in front of it.

**A batch runs both paths and selects.** A traced program cannot branch on a
byte it has not seen, and the batch is the compilation unit here, so every entry
pays a stateless verification *and* a stateful one and keeps the verdict its
indicator asks for. The alternative — partitioning on the host and issuing two
calls — trades that for a shape that depends on the data, which is a recompile
per batch composition and no longer one traced computation over the batch.

The two are not the same price, and which one is dearer depends on what is
counted. In SHA-256 compressions the stateless path dominates — roughly three
thousand against a stateful signature's under a thousand. In wall clock it is
the other way around: measured warm at `B = 4`, the stateless leg is about a
quarter of a verification and the stateful one the rest, because the stateful
path is bound by the 255 sequential steps the format's maximum depth forces on
every signature whatever depth it used, not by the hashing in them. So running
both costs a batch of stateful signatures — what a deployment mostly holds, the
fallback being a recovery path — about a third again over verifying it alone,
rather than the four times the hash count suggests.

**The length check the specification makes is the seam's padding rule here.** A
stateful signature is 548 to 4619 bytes and the seam zero-pads to
`signature_max_size`; what makes that unambiguous is that the indicator fixes the
length exactly — `17 + index field + 2 + 512 + 16 · depth` — so a padded batch
carries the length it derives, and there is nothing left to disagree with. What
the check would still catch, a signature whose length names a depth other than
its indicator's, cannot be expressed once padded.

**Not on the seam's `sign`, and no conformance pin.** The signer is stateful: a
leaf that signs twice reveals its WOTS+C secret, so signing has to hand back the
counter advanced past the leaf it used, which is two return values where the seam
has one. `sig_frx/signature.py` names that shape, and RFC 8391's `Xmss` already
has it.

**Where `Xmss` carries its index inside the secret key, this carries it beside
one.** The two make the same demand of a caller — the spent value has to be
passed back visibly, so signing twice at one position is something written down
rather than something a method quietly failed to do — and they differ because
the formats do: RFC 8391's secret key *is* `idx ‖ SK.seed ‖ …`, while SHRINCS's
82 bytes have no room for a counter and its reference passes `state_ctr`
alongside. Widening the key here would mean a serialization no reference produces
and a `SECRET_KEY_SIZE` that disagrees with the specification's, to gain nothing
the returned counter does not already give.

**A counter the tree cannot hold raises**, where the reference falls back to the
stateless path without comment. Signing statelessly is something a caller asks
for here, by passing no counter; arriving with a spent one is a signer that has
lost count, and answering it with a five-times-longer signature it did not choose
hides that.

**The tree's shape is `keygen`'s alone, and it is the instance's rather than an
argument.** A key pair cannot be made without one and the seam's `keygen` takes a
seed and nothing else, so it is what a key-generating `Shrincs` is constructed
with. Everything after key generation reads the shape out of the *secret key*,
which carries those two bytes for exactly this reason — so `sign` needs no
instance structure, and neither does `verify`, which never learns the shape at
all. `Shrincs()` is therefore the right thing to build for any consumer that
holds keys rather than makes them.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike
from hash_frx import block_size as block_size_of

from sig_frx import context as ctx
from sig_frx import hashes
from sig_frx.hash import bytestring, tweakable
from sig_frx.hash.shrincs import fxmss, stateless

_N = stateless.PARAMS.n

# What the signer serializes and this does not read: the 48-byte seed, `sl_root`,
# the two tree-structure bytes and `sf_root`. The public key's size and the
# signature's are `stateless`'s, since the stateless signature is the longer of
# the two — which the specification makes deliberate, so that a stateful one
# staying below it keeps the pair distinguishable by length.
SECRET_KEY_SIZE = 3 * _N + _N + 2 + _N

# The stateful signature's fixed header: the indicator byte and the randomizer.
_INDICATOR_BYTES = 1
_RANDOMIZER_BYTES = _N
_INDEX_FIELD_START = _INDICATOR_BYTES + _RANDOMIZER_BYTES  # 17


# FIPS 205 §10.2's pure-signature domain separator, which SHRINCS keeps for both
# paths: the stateful digest binds the same context encoding the stateless one
# does.
_PURE_DOMAIN = 0

# The secret key's layout, `sk_seed ‖ sk_prf ‖ pk_seed ‖ sl_root ‖ structure ‖
# sf_root` — the reference's serialization, which is why the counter is not in it.
_SK_SEED = slice(0, _N)
_SK_PRF = slice(_N, 2 * _N)
_SK_PK_SEED = slice(2 * _N, 3 * _N)
_SK_SL_ROOT = slice(3 * _N, 4 * _N)
_SK_STRUCTURE = slice(4 * _N, 4 * _N + fxmss.STRUCTURE_BYTES)
_SK_SF_ROOT = slice(4 * _N + fxmss.STRUCTURE_BYTES, SECRET_KEY_SIZE)
# `slh_dsa.sign` takes `SK.seed ‖ SK.prf ‖ PK.seed ‖ PK.root`, which is this
# key's first four values — the stateless half, contiguous and in order.
_SK_SLH_DSA = slice(0, 4 * _N)


class Shrincs:
    """SHRINCS over hash-frx's SHA-256.

    Batch-first: `verify` takes a leading `[B]` axis on every argument and
    returns `bool[B]`.

    `sf_structure` is the tree a key pair will be built over — two bytes, a shape
    and a depth. Only `keygen` needs it: `sign` reads the shape out of the secret
    key and `verify` never learns it, so an instance that holds keys rather than
    making them is built without one.
    """

    def __init__(self, sf_structure: ArrayLike | None = None) -> None:
        self.structure = (
            None if sf_structure is None else fxmss.Structure.parse(sf_structure)
        )
        self.public_key_size = stateless.PUBLIC_KEY_SIZE
        self.secret_key_size = SECRET_KEY_SIZE
        self.signature_max_size = stateless.SIGNATURE_SIZE
        # Both paths draw a randomizer per signature, so neither is deterministic
        # in the seam's sense. It is a signer's property, recorded for a consumer
        # reading the seam rather than used here.
        self.deterministic = False
        self.stateless = stateless.Stateless()
        # Borrowed, not rebuilt. Both paths tweak with FIPS 205 §11.2.1's family
        # at `n = 16` — the stateful one reaches `F`, `H` and `T` of it under
        # addresses of its own — and assembling a second one here would put
        # "which hash family goes with which security category" somewhere
        # `sha2_params` does not cover, which is the copy its docstring refuses.
        self._tweak = self.stateless.slh_dsa.tweak

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Shrincs):
            return NotImplemented
        return self.stateless == other.stateless and self.structure == other.structure

    def __hash__(self) -> int:
        return hash((type(self), self.stateless, self.structure))

    def keygen(self, seed: ArrayLike) -> tuple[Array, Array]:
        """`shrincs_keygen` — `pk_seed ‖ sl_root ‖ sf_root`, and the 82 bytes behind it.

        `seed` is `SK.seed ‖ SK.prf ‖ PK.seed`, which is SLH-DSA's own seed at this
        parameter set — so the stateless half of the key is `slh_dsa.keygen`
        unchanged and what is added is the FXMSS root. That root is the one value
        the tree's shape reaches, and the reason the structure bytes are in the
        secret key at all: a signer has to know afterwards what tree it built.
        """
        structure = self._signing_structure()
        values = fnp.asarray(seed, dtype=fnp.uint8)
        if values.shape != (stateless.PARAMS.seed_size,):
            raise ValueError(
                f"keygen takes SK.seed, SK.prf and PK.seed — {_N} bytes each, "
                f"{stateless.PARAMS.seed_size} in all — got shape "
                f"{tuple(values.shape)}"
            )
        slh_dsa_public, _ = self.stateless.slh_dsa.keygen(values)
        sl_root = slh_dsa_public[_N:]
        sf_root = fxmss.root(
            self._tweak, values[_SK_PK_SEED], values[_SK_SEED], structure
        )
        return (
            fnp.concatenate([values[_SK_PK_SEED], sl_root, sf_root]),
            fnp.concatenate(
                [
                    values,
                    sl_root,
                    fnp.asarray(structure.encoded, dtype=fnp.uint8),
                    sf_root,
                ]
            ),
        )

    # -- signing -----------------------------------------------------------

    def sign(
        self,
        secret_key: ArrayLike,
        message: ArrayLike,
        state_counter: int | None,
        *,
        randomness: ArrayLike | None = None,
        context: ArrayLike | None = None,
    ) -> tuple[Array, int | None]:
        """One signature, and the counter to store before releasing it.

        `state_counter` names the WOTS+C leaf to sign with and `None` asks for the
        stateless path instead. The second return value is what the caller must
        persist: the counter advanced past the leaf just spent, or `None` when no
        leaf was. Signing twice at one counter reveals that leaf's secret — not a
        degradation, a total break — so it is handed back rather than hidden in a
        mutable object, which is `Xmss`'s discipline under this scheme's format.

        `randomness` is SLH-DSA's `opt_rand`, and it belongs to the stateless path
        alone: the stateful one derives its randomizer from `sk_prf` and the
        leaf's position, so a stateful signature is a function of the key, the
        leaf and the message and needs nothing drawn. Supplying it with a counter
        is refused rather than ignored — a caller passing a salt believes it is
        salting something, and here it would reach nothing.

        The result is zero-padded to `signature_max_size`, which is the seam's
        rule and is unambiguous here because the indicator byte fixes the length.
        """
        key = fnp.asarray(secret_key, dtype=fnp.uint8)
        if key.shape != (SECRET_KEY_SIZE,):
            raise ValueError(
                f"a secret key is [{SECRET_KEY_SIZE}], got shape {tuple(key.shape)}"
            )
        messages = fnp.asarray(message, dtype=fnp.uint8)
        if messages.ndim != 1:
            raise ValueError(
                f"sign takes one message as [L]; the batch axis is verification's, "
                f"got shape {tuple(messages.shape)}"
            )
        if state_counter is None:
            signature = self.stateless.sign(
                key[_SK_SLH_DSA],
                key[_SK_SF_ROOT],
                messages,
                randomness=randomness,
                context=context,
            )
            return self._padded(signature), None
        if randomness is not None:
            raise ValueError(
                "the stateful path derives its randomizer from `sk_prf` and the "
                "leaf's position and has nowhere to put `randomness`; pass no "
                "counter to sign statelessly, where it is the salt"
            )
        return (
            self._padded(self._sign_stateful(key, messages, state_counter, context)),
            state_counter + 1,
        )

    def _sign_stateful(
        self,
        key: Array,
        messages: Array,
        state_counter: int,
        context: ArrayLike | None,
    ) -> Array:
        """`shrincs_sign`'s stateful half: the indicator, `R`, the index, the tree.

        The leaf is chosen first because its position tweaks the digest being
        signed, so there is no digest until there is a leaf.
        """
        structure = fxmss.Structure.parse(key[_SK_STRUCTURE])
        leaf_index, leaf_height = structure.leaf(state_counter)
        pk_seed, sl_root = key[_SK_PK_SEED], key[_SK_SL_ROOT]
        # The address's first nine bytes — the leaf's height and its index — which
        # bind both the randomizer and the digest to where the signature was made.
        index = fxmss.index_bytes(np.array([leaf_index], dtype=np.uint64))
        positions = np.concatenate([np.array([leaf_height], dtype=np.uint8), index[0]])
        # Built once: the randomizer is taken over these bytes and the digest over
        # the same ones, and `_digest_over` is what lets the second reuse them.
        bound = bound_message(sl_root, messages, context)
        random = randomizer(key[_SK_PRF], pk_seed, positions, bound)
        digest = _digest_over(
            random[None, :],
            pk_seed[None, :],
            key[_SK_SF_ROOT][None, :],
            positions[None, :],
            bound[None, :],
        )[0]
        return fnp.concatenate(
            [
                # The indicator byte is the leaf's height, which is already the
                # first of the nine position bytes bound into the digest.
                fnp.asarray(positions[:1], dtype=fnp.uint8),
                random,
                # The field is as many whole bytes as the leaf's depth needs, and
                # a verifier derives that width back from the indicator byte.
                fnp.asarray(
                    index[0][-fxmss.index_field_bytes(fxmss.HEIGHT - leaf_height) :],
                    dtype=fnp.uint8,
                ),
                fxmss.sign(
                    self._tweak,
                    pk_seed,
                    key[_SK_SEED],
                    structure,
                    digest,
                    leaf_height,
                    leaf_index,
                ),
            ]
        )

    def _padded(self, signature: Array) -> Array:
        """Zero-filled up to `signature_max_size` — the seam's variable-length rule."""
        return fnp.concatenate(
            [
                signature,
                fnp.zeros(
                    self.signature_max_size - signature.shape[0], dtype=fnp.uint8
                ),
            ]
        )

    def _signing_structure(self) -> fxmss.Structure:
        """The tree this instance builds keys over, or why it cannot build one."""
        if self.structure is None:
            raise ValueError(
                "this instance carries no tree structure, which a verifier does "
                "not need and a key pair cannot be made without: build "
                "`Shrincs(sf_structure)` with the shape and depth to generate one"
            )
        return self.structure

    def verify(
        self,
        public_key: ArrayLike,
        message: ArrayLike,
        signature: ArrayLike,
        *,
        context: ArrayLike | None = None,
    ) -> Array:
        """Verify a batch of SHRINCS signatures, of either path.

        `public_key` is `[B, 48]`, `signature` is `[B, 5777]`, `message` is
        `[B, L]`, and the result is `bool[B]`. `context` applies to the whole
        batch.
        """
        keys = fnp.asarray(public_key, dtype=fnp.uint8)
        if keys.ndim != 2 or keys.shape[1] != self.public_key_size:
            raise ValueError(
                f"a public key batch is [B, {self.public_key_size}], got shape "
                f"{tuple(keys.shape)}"
            )
        batch = keys.shape[0]
        signatures = fnp.asarray(signature, dtype=fnp.uint8)
        if signatures.ndim != 2 or signatures.shape[0] != batch:
            raise ValueError(
                f"one signature per public key, as a [B, {self.signature_max_size}] "
                f"batch: got {batch} keys and signatures of shape "
                f"{tuple(signatures.shape)}"
            )
        messages = fnp.asarray(message, dtype=fnp.uint8)
        if messages.ndim != 2 or messages.shape[0] != batch:
            raise ValueError(
                f"one message per public key, as a [B, L] batch: got {batch} keys "
                f"and messages of shape {tuple(messages.shape)}"
            )
        if signatures.shape[1] != self.signature_max_size:
            return fnp.zeros(batch, dtype=bool)

        indicators = signatures[:, 0]
        return fnp.where(
            indicators == np.uint8(stateless.STATELESS_INDICATOR),
            self.stateless.verify(keys, messages, signatures, context=context),
            self._verify_stateful(keys, messages, signatures, context),
        )

    # -- the stateful path -------------------------------------------------

    def _verify_stateful(
        self,
        keys: Array,
        messages: Array,
        signatures: Array,
        context: ArrayLike | None,
    ) -> Array:
        """The FXMSS half, for every entry — the caller selects who wanted it."""
        pk_seeds = keys[:, :_N]
        sl_roots = keys[:, _N : 2 * _N]
        sf_roots = keys[:, 2 * _N :]
        # The indicator byte is the leaf's height. It stays a byte where it goes
        # into a hash and becomes a column where it is arithmetic.
        indicators = signatures[:, :1]
        heights = signatures[:, 0].astype(fnp.uint32)
        widths, bits = index_field_widths(heights)

        indices, in_range = leaf_indices(signatures, widths, bits)
        digests = message_digest(
            signatures[:, _INDICATOR_BYTES:_INDEX_FIELD_START],
            pk_seeds,
            sl_roots,
            sf_roots,
            # The address's first nine bytes: the leaf's height and its index.
            fnp.concatenate([indicators, indices], axis=-1),
            messages,
            context,
        )
        roots, accepted = fxmss.root_from_sig(
            self._tweak,
            pk_seeds,
            fxmss_signatures(signatures, widths),
            digests,
            heights,
            indices,
        )
        return fnp.all(roots == sf_roots, axis=-1) & accepted & in_range


def index_field_widths(leaf_heights: ArrayLike) -> tuple[Array, Array]:
    """Each entry's index-field width in bytes, and the bits it may carry.

    `ceil(min(depth, 64) / 8)`, per entry, off the leaf height alone. The same
    rule `fxmss.index_field_bytes` states for one concrete depth — that one is a
    parser's, taking a Python int, and this one is a batch's; `fxmss_test` pins
    the two against each other.
    """
    depths = np.uint32(fxmss.HEIGHT) - fnp.asarray(leaf_heights, dtype=fnp.uint32)
    bits = fnp.minimum(depths, np.uint32(8 * fxmss.INDEX_BYTES))
    return (bits + np.uint32(7)) // np.uint32(8), bits


def leaf_indices(signatures: Array, widths: Array, bits: Array) -> tuple[Array, Array]:
    """The leaf index as `[B, 8]` bytes, and whether its tree holds it.

    The field is one to eight bytes wide and sits at a fixed offset, so it is
    gathered right-aligned into eight: the bytes below it belong to the FXMSS
    signature, and reading them would make a shallow leaf's index enormous.

    Bytes rather than a number for the reason `bytestring` exists — this is the
    value that does not fit an array lane, and truncating it would name a
    different subtree while verifying perfectly against itself.
    """
    lanes = np.arange(fxmss.INDEX_BYTES, dtype=np.uint32)
    columns = (
        _INDEX_FIELD_START + widths[:, None] - np.uint32(fxmss.INDEX_BYTES) + lanes
    )
    inside = columns >= np.uint32(_INDEX_FIELD_START)
    gathered = fnp.take_along_axis(signatures, columns.astype(fnp.int32), axis=1)
    indices = fnp.where(inside, gathered, np.uint8(0))
    # A depth of `d` addresses `2^d` leaves, so an index with a bit above that
    # names no position — and the field, being whole bytes, can carry one.
    return indices, fnp.all(indices == bytestring.mask_to(indices, bits), axis=-1)


def fxmss_signatures(signatures: Array, widths: Array) -> Array:
    """The FXMSS signature, gathered from past the variable-width index field.

    Padded to the longest the format allows: an entry whose own signature is
    shorter reads whatever the seam's padding left behind it, and the walk masks
    those steps off. The widest read is `17 + 8 + 4593`, inside the stateless
    signature's length, so nothing gathers out of bounds.
    """
    starts = _INDEX_FIELD_START + widths
    columns = starts[:, None] + np.arange(fxmss.SIGNATURE_SIZE_MAX, dtype=np.uint32)
    return fnp.take_along_axis(signatures, columns.astype(fnp.int32), axis=1)


def message_digest(
    randomizers: ArrayLike,
    pk_seeds: ArrayLike,
    sl_roots: ArrayLike,
    sf_roots: ArrayLike,
    positions: ArrayLike,
    messages: ArrayLike,
    context: ArrayLike | None = None,
) -> Array:
    """`H_msg_sf` — the 32 bytes WOTS+C signs, bound to the leaf's position.

    `positions` is the address's first nine bytes, the leaf's height and index;
    everything else is `[B, ·]` bytes. A module function rather than a method
    because it is the one construction SHRINCS does not share with FIPS 205, so
    it is the one the vectors have to reach directly — a digest checked only
    through a final verdict is checked by nothing that can say where it went
    wrong.

    Not `TweakableHash.h_msg`: that is FIPS 205's MGF1 construction, and this is
    a tweakable hash rather than a keyed one — the leaf's position goes into both
    the inner and the outer hash, which is what domain-separates one leaf's
    digest from another's. The trailing zero padding `H_msg_sl` carries is absent
    for the same reason: it existed to match MGF1, which does not apply here.

    The message is bound to the *stateless* half of the key, exactly as the
    stateless path's is bound to the stateful half. Neither signature carries to
    a key sharing only the other half.
    """
    return _digest_over(
        randomizers,
        pk_seeds,
        sf_roots,
        positions,
        bound_message(sl_roots, messages, context),
    )


def _digest_over(
    randomizers: ArrayLike,
    pk_seeds: ArrayLike,
    sf_roots: ArrayLike,
    positions: ArrayLike,
    bound: ArrayLike,
) -> Array:
    """`H_msg_sf`'s two hashes, over a message already bound by `bound_message`.

    Split out so the signer does not build that binding twice: it needs the bound
    message itself, for the randomizer, before there is a digest to take over it.
    """
    randoms = fnp.asarray(randomizers, dtype=fnp.uint8)
    seeds = fnp.asarray(pk_seeds, dtype=fnp.uint8)
    places = fnp.asarray(positions, dtype=fnp.uint8)
    bound = fnp.asarray(bound, dtype=fnp.uint8)
    inner = fnp.asarray(
        hashes.sha256(bound).digest(
            fnp.concatenate(
                [
                    randoms,
                    seeds,
                    fnp.asarray(sf_roots, dtype=fnp.uint8),
                    places,
                    bound,
                ],
                axis=-1,
            )
        ),
        dtype=fnp.uint8,
    )
    return fnp.asarray(
        hashes.sha256(inner).digest(
            fnp.concatenate([randoms, seeds, inner, places], axis=-1)
        ),
        dtype=fnp.uint8,
    )


def bound_message(
    sl_roots: ArrayLike, messages: ArrayLike, context: ArrayLike | None = None
) -> Array:
    """`toByte(0,1) ‖ toByte(|ctx|,1) ‖ ctx ‖ sl_root ‖ M` — the stateful path's.

    One function because two hashes are taken over exactly these bytes — the
    randomizer's and the digest's — and a binding that two callers each build for
    themselves is a binding that can come apart in one of them. Prepending
    `sl_root` is what keeps a stateful signature from carrying to a key that
    shares only the stateful half, the mirror of the `sf_root` the stateless path
    prepends.
    """
    return ctx.prepend(
        ctx.prefix(_PURE_DOMAIN, context),
        fnp.concatenate(
            [
                fnp.asarray(sl_roots, dtype=fnp.uint8),
                fnp.asarray(messages, dtype=fnp.uint8),
            ],
            axis=-1,
        ),
    )


def randomizer(
    sk_prf: ArrayLike, pk_seed: ArrayLike, positions: ArrayLike, bound: ArrayLike
) -> Array:
    """`PRF_msg_sf` — the stateful path's per-signature randomizer. `[16]`.

    HMAC-SHA256 under `sk_prf` filled out to a whole block with `0xFF` bytes, over
    the seed, the leaf's nine position bytes and the bound message. `bound` is
    `bound_message`'s output, which is also what the digest is taken over.

    **Derived, never drawn**, which is what separates this from `PRF_msg_sl`: the
    stateless path salts with an `opt_rand` the caller supplies, and the stateful
    path has none, so a stateful signature is a function of the key, the leaf and
    the message. That is what lets one reproduce from a seed and a counter with no
    third value recorded beside them.

    A module function for `message_digest`'s reason: it is a construction SHRINCS
    does not share with FIPS 205, so the vectors reach it directly rather than
    only through a signature that came out wrong somewhere.
    """
    message = fnp.concatenate(
        [
            fnp.asarray(pk_seed, dtype=fnp.uint8),
            fnp.asarray(positions, dtype=fnp.uint8),
            fnp.asarray(bound, dtype=fnp.uint8),
        ]
    )
    byte_hash = hashes.sha256(message)
    # The key is filled out to a whole block with `0xFF`, which is SHRINCS's own
    # construction — `hmac` derives the same width for itself.
    key = fnp.concatenate(
        [
            fnp.asarray(sk_prf, dtype=fnp.uint8),
            fnp.full(block_size_of(byte_hash) - _N, 0xFF, dtype=fnp.uint8),
        ]
    )
    return tweakable.hmac(byte_hash, key, message)[:_N]
