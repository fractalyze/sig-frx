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

from sig_frx import context as ctx
from sig_frx import hashes
from sig_frx.hash import bytestring
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


class Shrincs:
    """SHRINCS over hash-frx's SHA-256, verification-side.

    Batch-first: `verify` takes a leading `[B]` axis on every argument and
    returns `bool[B]`.
    """

    def __init__(self) -> None:
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
    randoms = fnp.asarray(randomizers, dtype=fnp.uint8)
    seeds = fnp.asarray(pk_seeds, dtype=fnp.uint8)
    places = fnp.asarray(positions, dtype=fnp.uint8)
    bound = ctx.prepend(
        ctx.prefix(_PURE_DOMAIN, context),
        fnp.concatenate(
            [
                fnp.asarray(sl_roots, dtype=fnp.uint8),
                fnp.asarray(messages, dtype=fnp.uint8),
            ],
            axis=-1,
        ),
    )
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
