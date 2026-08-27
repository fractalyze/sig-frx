# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""SHRINCS's stateless component — the SLH-DSA fallback under a SHRINCS key.

SHRINCS gives one public key two signing paths: a compact stateful one over a
flexible XMSS tree of WOTS+C keys, and a stateless fallback for a signer that has
lost its state. This module is the fallback, and only the fallback.

**It is SLH-DSA, at a parameter set FIPS 205 does not publish.** The
specification says so outright — "an SLH-DSA implementation that supports custom
parameter sets can therefore be used for the stateless component of SHRINCS, with
a thin wrapper to produce SHRINCS signatures" — and every primitive beneath it is
FIPS 205 §11.2.1's, byte for byte: a tweaked hash is
`Trunc_16(SHA-256(PK.seed ‖ toByte(0, 48) ‖ ADRS^c ‖ M))`, and `H_msg` is that
family's MGF1-SHA-256 truncated to `m = 24`, which is one block. So the algorithms
are [`slh_dsa.py`](../slhdsa/slh_dsa.py)'s, reached with a parameter record rather
than reimplemented, and what lands here is the wrapper.

`SlhDsaParams` derives the whole of the specification's stateless table from the
five values below — `h' = 9`, `m = 24`, `len = 35`, a 5776-byte signature — which
is why the table is not restated here. The set is not added to
`SHA2_PARAMETER_SETS`, because that dict is FIPS 205 Table 2 and a row SHRINCS
invented is not in it; it goes to `sha2_params` instead, which is that table's
lookup removed and the family choice kept. Which hash goes with which security
category stays a fact `slh_dsa.py` owns, so the day categories 3 and 5 need
SHA-512 there is one place to change.

**The wrapper is two bindings and a tag.** A SHRINCS public key is
`pk_seed ‖ sl_root ‖ sf_root`, so the SLH-DSA key is its first two thirds. The
message is `sf_root ‖ M`, which is what binds a stateless signature to the
stateful half of the key it was issued under — without it a signature would carry
over to any SHRINCS key sharing the stateless half. And the first byte of a
SHRINCS signature is the indicator: `255` selects this path, anything below it
selects the stateful one.

**It does not implement `Signature`, and there is no `keygen` here.** A SHRINCS
key pair cannot be generated without the stateful tree, since `sf_root` is one of
the three values a public key is made of — so key generation is
[`shrincs.py`](shrincs.py)'s and this module has only the two operations the
wrapper covers.

`sign` takes the pieces it binds rather than a whole SHRINCS secret key, where
`verify` splits a whole public key. That is not an inconsistency: the 48-byte
public key is `PUBLIC_KEY_SIZE` here, so splitting it is this module's own
format, while the 82-byte secret key carries the FXMSS structure bytes and is
`shrincs.py`'s. Each module parses the format it names.

That is also both halves of why the shared known-answer harness does not drive
this — `testing.md` asks for both to be named. There is no published format
for a loader to normalize, and this is not on the seam the harness signs through.
"""

from __future__ import annotations

import frx.numpy as fnp
from frx import Array
from frx.typing import ArrayLike

from sig_frx import context as context_rules
from sig_frx.batch import WrongWidth, require_batch
from sig_frx.hash.slhdsa.slh_dsa import SlhDsa, SlhDsaParams, sha2_params

# The specification's stateless parameters. `n` is not among them: every tweakable
# hash in SHRINCS truncates to 16 bytes, which is what makes this security
# category 1 and selects FIPS 205 §11.2.1's SHA-256-only family.
#
# `h` is not a parameter there either — the layer count and the per-layer height
# are — so it is their product, and giving `SlhDsaParams` the product is what lets
# it derive `h'` back out.
_LAYER_COUNT = 5  # `d`
_XMSS_HEIGHT = 9  # `h'`
PARAMS = SlhDsaParams(
    n=16,
    h=_LAYER_COUNT * _XMSS_HEIGHT,
    d=_LAYER_COUNT,
    a=13,  # `SPHX_FORS_HEIGHT`
    k=10,  # `SPHX_FORS_COUNT`
)

# `pk_seed ‖ sl_root ‖ sf_root`, three 16-byte values.
PUBLIC_KEY_SIZE = 3 * PARAMS.n

# The indicator value that selects this path. It is the height of a WOTS+C leaf on
# the stateful path, and `FXMSS_HEIGHT` is the one height no leaf can sit at — the
# root — so it is free to mean "not the stateful path".
STATELESS_INDICATOR = 255

# One indicator byte, then the SLH-DSA signature.
SIGNATURE_SIZE = 1 + PARAMS.signature_size


def accepts(
    slh_dsa: SlhDsa,
    keys: ArrayLike,
    messages: ArrayLike,
    signatures: ArrayLike,
    context: ArrayLike | None = None,
) -> Array:
    """The wrapper's verdict on an already-parsed batch — `bool[B]`.

    The two bindings and the tag the module docstring describes, and nothing
    else: `keys` is `[B, PUBLIC_KEY_SIZE]`, `signatures` is
    `[B, SIGNATURE_SIZE]`, `messages` is `[B, L]`.

    A component takes the family and the parsed arrays, the way
    `fxmss.root_from_sig` and `wots_c.pk_from_sig` do, so a caller that has
    already checked those shapes does not check them again — and a caller that
    composes both paths, as `Shrincs` does, reaches one of them without going
    back through a public verifier's front door. `Stateless.verify` is that door,
    and it is this function underneath.
    """
    parts = fnp.asarray(signatures, dtype=fnp.uint8)
    if parts.ndim != 2 or parts.shape[1] != SIGNATURE_SIZE:
        raise ValueError(
            f"a stateless signature batch is [B, {SIGNATURE_SIZE}], got shape "
            f"{tuple(parts.shape)}"
        )
    public_keys = fnp.asarray(keys, dtype=fnp.uint8)
    n = PARAMS.n
    # `pk_seed ‖ sl_root` is the SLH-DSA public key; `sf_root` binds the
    # message. Dropping the third would verify a signature under any SHRINCS
    # key that shares the first two.
    bound = fnp.concatenate([public_keys[:, 2 * n :], messages], axis=-1)
    accepted = slh_dsa.verify(
        public_keys[:, : 2 * n], bound, parts[:, 1:], context=context
    )
    return accepted & (parts[:, 0] == STATELESS_INDICATOR)


class Stateless:
    """SHRINCS's stateless component, over hash-frx's SHA-256.

    Batch-first, like every verifier here: `verify` takes a leading `[B]` axis and
    returns `bool[B]`, and a single verification is `B = 1`.
    """

    def __init__(self) -> None:
        self.slh_dsa = sha2_params(PARAMS)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Stateless):
            return NotImplemented
        return self.slh_dsa == other.slh_dsa

    def __hash__(self) -> int:
        return hash((type(self), self.slh_dsa))

    def sign(
        self,
        slh_dsa_secret_key: ArrayLike,
        sf_root: ArrayLike,
        message: ArrayLike,
        *,
        randomness: ArrayLike | None = None,
        context: ArrayLike | None = None,
    ) -> Array:
        """A stateless SHRINCS signature: the indicator byte, then SLH-DSA's.

        `slh_dsa_secret_key` is `SK.seed ‖ SK.prf ‖ PK.seed ‖ sl_root` and
        `sf_root` is the stateful half of the public key, which binds the message
        the way `verify` undoes. `randomness` is FIPS 205's `opt_rand`: this
        instance is the hedged variant, so it is required — the deterministic
        substitution of `PK.seed` for it would silently produce a different, still
        valid signature, which nothing downstream would notice.

        `[SIGNATURE_SIZE]` exactly, this path having no variable length; the seam's
        padding is `shrincs.py`'s to apply because a stateful signature is shorter.
        """
        secret = fnp.asarray(slh_dsa_secret_key, dtype=fnp.uint8)
        if secret.shape != (PARAMS.secret_key_size,):
            raise ValueError(
                f"an SLH-DSA secret key is [{PARAMS.secret_key_size}], got shape "
                f"{tuple(secret.shape)}"
            )
        bound = fnp.concatenate(
            [
                fnp.asarray(sf_root, dtype=fnp.uint8),
                fnp.asarray(message, dtype=fnp.uint8),
            ]
        )
        return fnp.concatenate(
            [
                fnp.full(1, STATELESS_INDICATOR, dtype=fnp.uint8),
                self.slh_dsa.sign(
                    secret, bound, randomness=randomness, context=context
                ),
            ]
        )

    def verify(
        self,
        public_key: ArrayLike,
        message: ArrayLike,
        signature: ArrayLike,
        *,
        context: ArrayLike | None = None,
        position: ArrayLike | None = None,
    ) -> Array:
        """Verify a batch of stateless SHRINCS signatures.

        `public_key` is `[B, 48]`, `signature` is `[B, 5777]`, `message` is
        `[B, L]`, and the result is `bool[B]`. `context` is the application context
        string, one value for the whole batch, and it reaches the same place FIPS
        205 puts it — the wrapper prepends `sf_root` to the message, and SLH-DSA's
        external interface prepends the domain byte and the context to that.

        A wrong indicator byte is a verdict rather than an error: a *stateful*
        SHRINCS signature is a well-formed thing this component cannot check, and
        the answer to "is this a valid stateless signature" is no.
        """
        context_rules.require_no_position(position, "stateless SHRINCS")
        # A width this component does not issue is a verdict, not an error — the
        # same reading as a wrong indicator byte, and SLH-DSA's own. `accepts`
        # raises on it instead, because reaching it is a caller inside the
        # package getting the format wrong rather than a verifier being handed
        # something.
        operands = require_batch(
            public_key,
            message,
            signature,
            public_key_size=PUBLIC_KEY_SIZE,
            signature_size=SIGNATURE_SIZE,
            public_key_width=WrongWidth.ERROR,
        )
        if not operands.well_formed:
            return fnp.zeros(operands.size, dtype=bool)
        return accepts(
            self.slh_dsa,
            operands.public_key,
            operands.message,
            operands.signature,
            context,
        )
