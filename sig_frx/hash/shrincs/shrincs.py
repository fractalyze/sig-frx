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

The two are not the same price. A stateless verification is roughly three
thousand SHA-256 compressions; a stateful one is under a thousand, and most of
that is the 255-step Merkle walk the format's maximum depth forces on every
signature whatever its own depth. So a batch of stateful signatures — which is
what a deployment mostly holds, the fallback being a recovery path — costs about
four times what verifying it alone would.

**The length check the specification makes is the seam's padding rule here.** A
stateful signature is 548 to 4619 bytes and the seam zero-pads to
`signature_max_size`; what makes that unambiguous is that the indicator fixes the
length exactly — `17 + index field + 2 + 512 + 16 · depth` — so a padded batch
carries the length it derives, and there is nothing left to disagree with. What
the check would still catch, a signature whose length names a depth other than
its indicator's, cannot be expressed once padded.

**Not on the seam's `sign`, and no conformance pin.** The signer is stateful: a
leaf that signs twice reveals its WOTS+C secret, so signing has to hand back the
key advanced past what it used, which is two return values where the seam has
one. `sig_frx/signature.py` names this shape, and RFC 8391's `Xmss` already has
it. `keygen` is a signer's too — it builds the FXMSS tree the public key's third
part is the root of — and raises until that lands.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike
from hash_frx import Sha256

from sig_frx import context as ctx
from sig_frx import hashes
from sig_frx.hash.shrincs import fxmss, stateless
from sig_frx.hash.tweakable import Sha2TweakableHash

# The public key's three 16-byte parts, and the secret key the signer serializes:
# the 48-byte seed, `sl_root`, the two tree-structure bytes and `sf_root`.
_N = stateless.PARAMS.n
PUBLIC_KEY_SIZE = stateless.PUBLIC_KEY_SIZE
SECRET_KEY_SIZE = 3 * _N + _N + 2 + _N

# The stateless signature is the longer of the two, which the specification makes
# deliberate: a stateful signature has to stay below it so the two remain
# distinguishable by length.
SIGNATURE_MAX_SIZE = stateless.SIGNATURE_SIZE

# The stateful signature's fixed header: the indicator byte and the randomizer.
_INDICATOR_BYTES = 1
_RANDOMIZER_BYTES = _N
_INDEX_FIELD_START = _INDICATOR_BYTES + _RANDOMIZER_BYTES  # 17

# The leaf index's widest field, and the eight-byte slot an address gives it.
_INDEX_BYTES = 8

# FIPS 205 §10.2's pure-signature domain separator, which SHRINCS keeps for both
# paths: the stateful digest binds the same context encoding the stateless one
# does.
_PURE_DOMAIN = 0


class Shrincs:
    """SHRINCS over hash-frx's SHA-256, verification-side.

    Batch-first: `verify` takes a leading `[B]` axis on every argument and
    returns `bool[B]`.
    """

    def __init__(self) -> None:
        self.public_key_size = PUBLIC_KEY_SIZE
        self.secret_key_size = SECRET_KEY_SIZE
        self.signature_max_size = SIGNATURE_MAX_SIZE
        # Both paths draw a randomizer per signature, so neither is deterministic
        # in the seam's sense. It is a signer's property, recorded for a consumer
        # reading the seam rather than used here.
        self.deterministic = False
        self.stateless = stateless.Stateless()
        # Both paths tweak with FIPS 205 §11.2.1's family at `n = 16`; the
        # stateful one reaches `F`, `H` and `T` of it under its own addresses.
        # `m` belongs to the stateless digest and is the parameter set's.
        self._tweak = Sha2TweakableHash(Sha256(), n=_N, m=stateless.PARAMS.m)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Shrincs):
            return NotImplemented
        return self.stateless == other.stateless

    def __hash__(self) -> int:
        return hash((type(self), self.stateless))

    def keygen(self, seed: ArrayLike) -> tuple[Array, Array]:
        """Not yet: a SHRINCS key pair is a signer's, and the signer is stateful."""
        raise NotImplementedError(
            "SHRINCS key generation builds the FXMSS tree whose root is the "
            "public key's third part, which is signer-side work this module does "
            "not carry; it verifies both paths and generates neither key"
        )

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
        if keys.ndim != 2 or keys.shape[1] != PUBLIC_KEY_SIZE:
            raise ValueError(
                f"a public key batch is [B, {PUBLIC_KEY_SIZE}], got shape "
                f"{tuple(keys.shape)}"
            )
        batch = keys.shape[0]
        signatures = fnp.asarray(signature, dtype=fnp.uint8)
        if signatures.ndim != 2 or signatures.shape[0] != batch:
            raise ValueError(
                f"one signature per public key, as a [B, {SIGNATURE_MAX_SIZE}] "
                f"batch: got {batch} keys and signatures of shape "
                f"{tuple(signatures.shape)}"
            )
        messages = fnp.asarray(message, dtype=fnp.uint8)
        if messages.ndim != 2 or messages.shape[0] != batch:
            raise ValueError(
                f"one message per public key, as a [B, L] batch: got {batch} keys "
                f"and messages of shape {tuple(messages.shape)}"
            )
        if signatures.shape[1] != SIGNATURE_MAX_SIZE:
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

        leaf_indices, in_range = self._leaf_indices(signatures)
        digests = self._message_digests(
            signatures, pk_seeds, sl_roots, sf_roots, leaf_indices, messages, context
        )
        roots, accepted = fxmss.root_from_sig(
            self._tweak,
            pk_seeds,
            self._fxmss_signatures(signatures),
            digests,
            signatures[:, 0].astype(fnp.uint32),
            leaf_indices,
        )
        return fnp.all(roots == sf_roots, axis=-1) & accepted & in_range

    def _index_field_widths(self, signatures: Array) -> tuple[Array, Array]:
        """Each entry's index-field width in bytes, and the bits it may carry.

        `ceil(min(depth, 64) / 8)`, per entry, off the indicator alone — the same
        rule `fxmss.index_field_bytes` states for one concrete depth, which the
        tests pin the two against each other on.
        """
        depths = np.uint32(fxmss.HEIGHT) - signatures[:, 0].astype(fnp.uint32)
        bits = fnp.minimum(depths, np.uint32(8 * _INDEX_BYTES))
        return (bits + np.uint32(7)) // np.uint32(8), bits

    def _leaf_indices(self, signatures: Array) -> tuple[Array, Array]:
        """The leaf index as `[B, 8]` bytes, and whether the tree holds it.

        The field is one to eight bytes wide and sits at a fixed offset, so it is
        gathered right-aligned into eight: the bytes below it belong to the FXMSS
        signature, and reading them would make a shallow leaf's index enormous.

        Bytes rather than a number for the reason `bytestring` exists — this is
        the value that does not fit an array lane, and truncating it would name a
        different subtree while verifying perfectly against itself.
        """
        widths, bits = self._index_field_widths(signatures)
        lanes = np.arange(_INDEX_BYTES, dtype=np.uint32)
        columns = _INDEX_FIELD_START + widths[:, None] - np.uint32(_INDEX_BYTES) + lanes
        inside = columns >= np.uint32(_INDEX_FIELD_START)
        gathered = fnp.take_along_axis(signatures, columns.astype(fnp.int32), axis=1)
        indices = fnp.where(inside, gathered, np.uint8(0))

        # A depth of `d` addresses `2^d` leaves, so an index with a bit above that
        # names no position. The mask is `bytestring._byte_mask`'s formula with the
        # width per entry rather than fixed, which is what makes it array
        # arithmetic instead of a table.
        reaches = fnp.clip(
            bits[:, None].astype(fnp.int32)
            - 8 * np.arange(_INDEX_BYTES - 1, -1, -1, dtype=np.int32),
            0,
            8,
        )
        masks = ((np.uint32(1) << reaches.astype(fnp.uint32)) - 1).astype(fnp.uint8)
        return indices, fnp.all(indices == (indices & masks), axis=-1)

    def _fxmss_signatures(self, signatures: Array) -> Array:
        """The FXMSS signature, gathered from past the variable-width index field.

        Padded to the longest the format allows: an entry whose own signature is
        shorter reads whatever the seam's padding left behind it, and the walk
        masks those steps off. The widest read is `17 + 8 + 4593`, inside the
        stateless signature's length, so nothing gathers out of bounds.
        """
        widths, _ = self._index_field_widths(signatures)
        starts = _INDEX_FIELD_START + widths
        columns = starts[:, None] + np.arange(fxmss.SIGNATURE_SIZE_MAX, dtype=np.uint32)
        return fnp.take_along_axis(signatures, columns.astype(fnp.int32), axis=1)

    def _message_digests(
        self,
        signatures: Array,
        pk_seeds: Array,
        sl_roots: Array,
        sf_roots: Array,
        leaf_indices: Array,
        messages: Array,
        context: ArrayLike | None,
    ) -> Array:
        """`H_msg_sf` — the 32 bytes WOTS+C signs, bound to the leaf's position.

        Not `TweakableHash.h_msg`: that one is FIPS 205's MGF1 construction, and
        this is a tweakable hash rather than a keyed one — the leaf's position
        goes into both the inner and the outer hash, which is what domain-separates
        one leaf's digest from another's. The trailing zero padding `H_msg_sl`
        carries is absent for the same reason: it existed to match MGF1, which
        does not apply here.

        The message it hashes is bound to the *stateless* half of the key, exactly
        as the stateless path's is bound to the stateful half. Neither signature
        carries to a key sharing only the other half.
        """
        randomizers = signatures[:, _INDICATOR_BYTES:_INDEX_FIELD_START]
        # The address's first nine bytes: the leaf's height — which is the
        # indicator, unread as a number — and its index.
        positions = fnp.concatenate([signatures[:, :1], leaf_indices], axis=-1)
        bound = ctx.prepend(
            ctx.prefix(_PURE_DOMAIN, context),
            fnp.concatenate([sl_roots, messages], axis=-1),
        )
        inner = fnp.asarray(
            hashes.sha256(bound).digest(
                fnp.concatenate(
                    [randomizers, pk_seeds, sf_roots, positions, bound], axis=-1
                )
            ),
            dtype=fnp.uint8,
        )
        return fnp.asarray(
            hashes.sha256(inner).digest(
                fnp.concatenate([randomizers, pk_seeds, inner, positions], axis=-1)
            ),
            dtype=fnp.uint8,
        )
