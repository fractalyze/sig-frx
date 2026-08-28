# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""leanSig — the signature Ethereum's lean consensus makes and checks per slot.

A generalized XMSS: one one-time key per slot across a 2^32-slot lifetime, its
chains and its Merkle tree hashed by Poseidon over KoalaBear, its wire form SSZ.
The layers below this one each landed with their own upstream gate — the two
permutations, the compression and sponge modes, the encoding pipeline, the
tweakable family, the codec, the signer's tree — and this is the assembly: the
walk that turns a signature back into a root and compares it to the key's, and
the one that produces the signature in the first place.

**`verify` is the seam's and the other two are not.** `keygen` and `sign` here
take and return what leanSig actually has — a materialized signing state rather
than a byte string, and a slot the caller names — which is the shape
[`signature.py`](../../signature.py) reserves for a stateful scheme. The class
docstring says which of that rule's demands each one meets, and
[`signing.py`](signing.py) holds the machinery.

**The slot arrives as `position`, and that is a seam field rather than a
scheme's own argument.** leanSig's verifier takes the slot as an input, unlike
RFC 8391 XMSS, which carries the index inside the signature encoding. Upstream
leaves it off the wire deliberately: a consensus client verifying an attestation
already knows the slot it is verifying, so spending eight bytes per signature to
repeat it would be paying for what the caller has. What that costs *this* repo is
that the seam's three per-entry operands could not carry it — `context` is one
value per call by design, and a beacon block carries attestations from up to 32
slots back, so a batch spans slots and the value cannot be hoisted out of it.
`signature.py` grew `position` for this, and leanSig is what reads it.

**The message space is a 32-byte root, not arbitrary bytes.** The seam leaves
`L` to the caller and only requires it static, so this checks the width itself.
#42 is the precedent for stating both of these rather than working around them
quietly.

## What traces, and what cannot

The hashing traces as one computation over the whole batch — the ~180 Poseidon
calls a verification runs are the reason the batch-first seam exists, and they
are all downstream of the encode. The encode itself is host-only and stays
there:

- `encoding.encode_message` decomposes a 256-bit root base-p, and
  `encode_epoch` does the same to `(slot << 8) | prefix`. A running remainder
  reaches `p ≈ 2^31`, so even a 16-bit digit step needs `2^47` — genuine bignum
  division rather than a shift schedule ([`field.py`](field.py)).
- The chain and tree tweaks pack a position into 63 bits
  ([`tweakable.py`](tweakable.py)), which is past a lane by the same rule.

So `verify` is an eager entrance around a traced core, which is
[`xmss.py`](../xmss/xmss.py)'s shape and for its reason: that scheme reads its
index off the signature bytes on the host too. What the rule in
[`conventions.md`](../../../docs/reference/conventions.md) asks — that a batch
trace as one computation rather than as `B` dispatches — is satisfied where the
work is, and a caller who wants the whole call inside a `jit` zone is asking for
a `uint64` lane on the traced path, which this scheme's own acceptance criteria
refuse.
"""

from __future__ import annotations

from collections.abc import Sequence

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from sig_frx import context as context_rules
from sig_frx.batch import require_batch
from sig_frx.hash import tree, wots
from sig_frx.hash import tweakable as shared_tweakable
from sig_frx.hash.leansig import encoding, field, signing, ssz, tweakable
from sig_frx.hash.leansig.field import F
from sig_frx.hash.leansig.params import PRESETS, LeanSigParams


class LeanSig:
    """leanSig at one preset — the scheme, over the layers below it.

    Built with `named` unless a test wants a preset upstream does not ship. The
    parameter set is the only choice: one field, one pair of permutations, one
    encoding, so there is nothing else for a caller to select.

    **`verify` is the seam's; `keygen` and `sign` are not, and this carries no
    conformance pin.** `signature.py` reserves that shape for a stateful scheme,
    and leanSig is one twice over: its secret key is a materialized signing
    state rather than a byte string of a size the parameter set fixes, and its
    signer takes the slot as an argument. What it does *not* need is the
    advanced state the rule asks a stateful `sign` to hand back — the two
    precedents return an advanced key (`Xmss`) or an advanced counter
    (`Shrincs`) because in both the position lives inside what the caller
    passed. Here the caller names the slot, so a spent one is spent at the call
    site rather than inside an object, and what moves is the *prepared window* —
    which `advance_preparation` moves explicitly, returning the key that has
    moved. That is the same demand the rule makes, met by the surface the scheme
    actually has.
    """

    def __init__(self, params: LeanSigParams) -> None:
        self.params = params
        self._family = tweakable.LeanSigTweakableHash(params)
        self.public_key_size = ssz.public_key_size(params)
        # A module function rather than a `LeanSigParams` property, unlike ML-DSA
        # and Falcon: `params.py` is a leaf `ssz` imports, so the reverse would
        # be a cycle. The reason is recorded at the function.
        self.signature_max_size = ssz.signature_size(params)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LeanSig):
            return NotImplemented
        return self.params == other.params

    def __hash__(self) -> int:
        return hash((type(self), self.params))

    @property
    def signatures_per_key(self) -> int:
        """How many slots one key covers — `2^log_lifetime`, and not one more."""
        return 1 << self.params.log_lifetime

    def verify(
        self,
        public_key: ArrayLike,
        message: ArrayLike,
        signature: ArrayLike,
        *,
        context: ArrayLike | None = None,
        position: ArrayLike | None = None,
    ) -> Array:
        """Verify a batch at the slots `position` names: -> bool `[B]`.

        `public_key` is `[B, public_key_size]`, `message` is `[B, 32]`,
        `signature` is `[B, signature_max_size]`, and `position` is `[B]` — one
        slot per entry, because a beacon block carries attestations from many.

        The verdict folds in everything the bytes can get wrong alongside the
        root comparison: the two codecs' canonical-residue and offset checks, the
        decode's abort, and the target-sum filter — the last of which is not a
        malformedness check but the scheme's unforgeability, and is the one a
        rebuilt root cannot speak for (`_codeword`). Upstream returns `False` on
        each; here they are `bool` beside the values, because a tracer has no
        exception to take (`encoding.py`).

        A slot at or past the key's lifetime is a caller mistake rather than a
        verdict — the tree has no leaf to index, so there is nothing to compute a
        `False` from. That is the same reading `encode_epoch` gives the `Uint64`
        bound it checks.
        """
        context_rules.require_empty(context, "leanSig")
        params = self.params
        operands = require_batch(
            public_key,
            message,
            signature,
            public_key_size=self.public_key_size,
            signature_size=self.signature_max_size,
        )
        slots = self._slots(position, operands.size)
        # Read from the caller's own array rather than back off `operands`. The
        # message is the one operand with no device life here — a base-p
        # decomposition consumes it and nothing else — so pulling the lifted copy
        # back would be a round trip, and on a device leg a blocking sync in the
        # middle of a verification. `encode_messages` owns the width rule: `L` is
        # the caller's everywhere else on the seam, and here the scheme signs a
        # 32-byte root.
        message_bytes = np.asarray(message, dtype=np.uint8)
        if not operands.well_formed:
            return fnp.zeros(operands.size, dtype=bool)

        roots, parameters, key_ok = ssz.decode_public_key(
            operands.public_key, params=params
        )
        siblings, rho, hashes, signature_ok = ssz.decode_signature(
            operands.signature, params=params
        )

        # The host half, and the whole of it: both encoders decompose a wide
        # integer base-p, which no lane holds. Batched rather than looped, so
        # each is one transfer for the call — the per-entry form spent about an
        # eighth of a `PROD` verification on dispatch alone.
        message_elements = encoding.encode_messages(message_bytes, params=params)
        epoch_elements = encoding.encode_epochs(slots, params=params)

        digits, on_layer = encoding.codewords(
            message_elements, parameters, epoch_elements, rho, params=params
        )
        leaves = self._family.leaf(
            parameters,
            tweakable.tree_tweaks(0, slots, params=params),
            self._chain_ends(parameters, hashes, digits, slots),
        )
        computed = tree.root_from_path(
            self._family,
            parameters,
            leaves,
            slots,
            siblings,
            tweakable.node_tweaks(params),
        )
        return key_ok & signature_ok & on_layer & fnp.all(computed == roots, axis=-1)

    def _slots(self, position: ArrayLike | None, batch: int) -> np.ndarray:
        """`position` as a host column of `batch` slots, bounded by the lifetime.

        Host because everything it feeds is: the message tweak's base-p
        decomposition and the chain and tree tweaks' 63-bit packing both run
        there, so a slot that arrived on a device would be pulled back at the
        first of them anyway.
        """
        if position is None:
            raise ValueError(
                "leanSig verifies at a slot, so `position` is required: one "
                "slot per entry, as a [B] column. The signature does not carry "
                "it — upstream leaves it off the wire because a consensus "
                "client already knows the slot it is verifying"
            )
        slots = np.asarray(position)
        if slots.ndim != 1 or slots.shape[0] != batch:
            raise ValueError(
                f"one slot per public key, as a [B] column: got {batch} keys "
                f"and positions of shape {tuple(slots.shape)}"
            )
        outside = slots[(slots < 0) | (slots >= self.signatures_per_key)]
        if outside.size:
            raise ValueError(
                f"this key covers slots [0, {self.signatures_per_key}); "
                f"{outside.size} of {slots.size} are outside, "
                f"the first being {int(outside[0])}"
            )
        return slots

    # -- key generation and signing ----------------------------------------

    def keygen(
        self, prf_key: bytes, parameter: Sequence[int]
    ) -> tuple[Array, signing.SecretKey]:
        """A key pair over the whole lifetime: `(public key bytes, secret key)`.

        `prf_key` is the 32-byte master seed and `parameter` the public
        parameter's `parameter_length` canonical residues, in leanSpec's order —
        the order a published key states them in, which is the reverse of the
        one everything here hashes over ([`field.py`](field.py)).

        **Both are taken rather than drawn.** Upstream's `key_gen` reaches
        `os.urandom` and `secrets.randbelow` itself, so its keys are not
        reproducible from anything; taking them is what makes a key a function
        of published bytes, and it is the same choice
        [`xmss.py`](../xmss/xmss.py)'s `keygen` makes about RFC 8391's three
        seeds.

        The public key comes back as bytes because that is what a verifier takes
        — `verify` is the seam's and reads the SSZ container — while the secret
        key does not, for the reason `signing.SecretKey` records.

        The whole lifetime is built, and a sub-range is refused rather than
        supported: upstream fills an unbuilt window's tree with fresh OS
        randomness, so a partial key is not a function of its inputs at all
        ([`signing.py`](signing.py)). That makes this a `TEST`-preset operation —
        a `PROD` lifetime is `2^32` leaves.
        """
        root, secret = signing.keygen(
            self._family, prf_key, self._parameter(parameter), params=self.params
        )
        return (
            ssz.encode_public_key(root, secret.parameter, params=self.params),
            secret,
        )

    def sign(
        self,
        secret_key: signing.SecretKey,
        message: ArrayLike,
        *,
        position: int,
    ) -> Array:
        """Sign one 32-byte root at slot `position`: -> uint8 `[signature_size]`.

        One message, because signing is one message — the batch axis belongs to
        verification, which is the side that meets many signatures.

        **Deterministic in `(secret key, slot, message)`**, and that is a
        security property rather than a convenience: the randomness the search
        settles on is derived from the seed and the attempt number, so signing
        one slot twice yields the same signature rather than a second one over a
        different codeword ([`prf.py`](prf.py)). It is still the caller's job not
        to sign two *different* messages at one slot, which is what a
        synchronized one-time scheme forbids and what no signer can check.

        `position` is the same slot `verify` takes per entry, here one value
        because there is one signature. It must be inside the prepared window —
        `advance_preparation` is what moves that.
        """
        params = signing.paired(self._family, secret_key)
        slot = _slot(position, self.signatures_per_key)
        prepared = secret_key.prepared
        if slot not in prepared:
            raise ValueError(
                f"slot {slot} is outside the prepared interval "
                f"[{prepared.start}, {prepared.stop}); call "
                f"advance_preparation to slide the window forward"
            )
        # The 32-byte width is not checked here: `search` reaches
        # `encoding.encode_message` before it grinds anything, and that is the
        # module that owns what leanSig signs. Restating it would be a second
        # place for the two to disagree about a root.
        randomness, digits = signing.search(
            secret_key, slot, bytes(np.asarray(message, dtype=np.uint8))
        )
        return ssz.encode_signature(
            signing.combined_path(secret_key, slot),
            randomness,
            signing.release(self._family, secret_key, slot, digits),
            params=params,
        )

    def advance_preparation(self, secret_key: signing.SecretKey) -> signing.SecretKey:
        """The key with its prepared window slid one bottom tree forward.

        The state move this scheme has, and the reason `sign` needs none: what a
        leanSig signer advances past is not a position inside the key — the
        caller names that — but the window of slots it can serve without
        rebuilding a bottom tree. A key already reaching the end of its lifetime
        comes back unchanged.
        """
        return signing.advance_preparation(self._family, secret_key)

    def _parameter(self, parameter: Sequence[int]) -> Array:
        """The public parameter, checked and then placed for a hash.

        Both checks are for the same reason [`ssz.py`](ssz.py)'s decode carries
        its range check: the cast into the field *reduces*, so a value at or
        above the prime would become a different, well-formed element and
        generate a key for a parameter the caller did not name. Refusing is what
        upstream's `Fp` validation path does with the same input.
        """
        residues = field.host_column(
            parameter,
            f"a public parameter is canonical residues in [0, {field.PRIME})",
            field.PRIME,
        )
        if residues.size != self.params.parameter_length:
            raise ValueError(
                f"the public parameter is {self.params.parameter_length} field "
                f"elements, got {residues.size}"
            )
        return field.lane_reversed(residues)

    def _chain_ends(
        self, parameters: Array, hashes: Array, digits: Array, slots: np.ndarray
    ) -> Array:
        """Every chain walked from its digit to the top: -> `[B, dimension, n]`.

        The verifier's half of a Winternitz chain. The signer released the value
        at `digit`, so the verifier applies the remaining `base - 1 - digit`
        steps and every chain lands at the same top the leaf was built from — a
        codeword that claims a smaller digit than it was signed at walks too far
        and misses it.

        One `wots.chain` over `B · dimension` rows rather than `B` calls of
        `dimension`: the walk masks per row, so the whole batch's chains are
        `base - 1` batched hashes however their digits differ. The rows are
        entry-major, which is what `repeat_per_entry` and the tweak columns below
        are all laid out to agree with.
        """
        params = self.params
        batch, dimension = len(slots), params.dimension
        starts = digits.reshape(batch * dimension)
        walked = wots.chain(
            self._family,
            shared_tweakable.repeat_per_entry(parameters, dimension, dtype=F),
            hashes.reshape(batch * dimension, params.hash_length),
            starts,
            np.uint32(params.base - 1) - starts,
            tweakable.chain_step_tweaks(
                np.repeat(slots, dimension),
                np.tile(np.arange(dimension), batch),
                params=params,
            ),
        )
        return walked.reshape(batch, dimension, params.hash_length)


def _slot(position: int, lifetime: int) -> int:
    """One slot, as the host integer every position here is.

    A slot at or past the key's lifetime is a caller mistake rather than a
    verdict, which is the same reading `verify` gives its `[B]` column — there is
    no leaf to index, so there is nothing to compute an answer from.

    The prepared-window check that follows it would reject the same slots, since
    the window is always inside the lifetime — but it would advise sliding the
    window forward, which no number of slides can reach. A wrong remedy is worse
    than none, so the coarser bound is stated first.
    """
    slot = int(position)
    if not 0 <= slot < lifetime:
        raise ValueError(f"this key covers slots [0, {lifetime}); got {slot}")
    return slot


def named(name: str) -> LeanSig:
    """The preset called `name` — `prod` or `test`.

    Named for what upstream calls them rather than for a security level, because
    that is what they are: `PROD_CONFIG` is the deployed parameter set and
    `TEST_CONFIG` is the same scheme at a lifetime a test can build a key for.
    """
    if name not in PRESETS:
        raise ValueError(f"{name!r} is not one of {sorted(PRESETS)}")
    return LeanSig(PRESETS[name])
