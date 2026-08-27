# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The Signature seam every scheme in this repo implements.

A digital signature scheme: `keygen` / `sign` / `verify` over keys and signatures
in their standard's wire encoding. A consumer reads the size attributes to
allocate statically and calls the three methods; it picks a scheme by
construction, and swapping SLH-DSA for ML-DSA changes that one construction
rather than every call site. This is the signature counterpart of hash-frx's
`Permutation` and `ByteHash`, and it carries the same rules.

**Verification is batch-first.** `verify` takes a leading `[B]` axis on every
argument and returns `bool[B]`; a single verification is `B = 1`, not a separate
entry point. Verification is the hot path and it is embarrassingly parallel, so
the batch is what maps onto a GPU's width. A seam that admitted a scalar `verify`
would get one implemented as a Python loop over signatures, and the parallelism
would be gone before anyone noticed. `keygen` and `sign` are not batched: they
are not the hot path, and `frx.vmap` covers the rare caller that needs it.

**Keys and signatures are `uint8` arrays in the standard's encoding**, not
scheme-named pytrees. What a consumer holds is bytes — off a chain, out of a
certificate, off a wire — so a seam taking a structured form would make every
consumer call a scheme-specific decode first, which means naming the scheme,
which is what the seam exists to prevent. The standards define these objects as
byte strings and the known-answer tests compare them as byte strings; making the
seam agree is what lets both work without a per-scheme codec.

The cost is that a scheme parses its own encoding on entry. For verification that
parse is vectorized over the batch and small beside the scheme's own arithmetic;
for signing it does not matter, because signing here is for reproducing test
vectors rather than production use (`docs/reference/security.md`). A scheme that
wants a parsed key across many calls exposes that as its own surface, below the
seam.

A variable-length signature is zero-padded to `signature_max_size`; the standards
that have one define a padded encoding that is exactly this, and a compressed
encoding is self-delimiting, so the padding is unambiguous either way.

**Domain separation rides as `context`.** FIPS 204 and 205 both take an
application context string in their external interface, and it changes the
message that gets signed — dropping it silently verifies something other than
what the caller asked about. It is one value per call rather than one per batch
entry, because a verifier serves one protocol domain at a time. A scheme whose
standard has no context (RFC 8391, ECDSA) requires it empty and raises otherwise,
rather than accepting and ignoring it.

The seam names the standards' plain external interface. The variants that prepare
a different message — a pre-hashed message, the internal interface, a
pre-computed message representative — are not on it: they are different
operations, and a scheme that implements one exposes it under its own name.

Implementations MUST define value-based `__eq__`/`__hash__` over their full
parameter surface: a scheme instance rides pytree aux, where identity equality
silently re-traces the enclosing jit zone on every freshly built instance — a
cost that does not error, it just makes every call slow. A Protocol cannot
enforce this; each implementation carries it.

Each implementation module ends with a conformance pin, so mypy fails the module
that drifts from the seam rather than the consumer that calls it::

    if TYPE_CHECKING:
        _: type[Signature] = SlhDsa

**A stateful scheme implements `verify` and not `sign`, and carries no pin.** A
one-time key signs once — signing twice at one index reveals the WOTS+ secret — so
the signer has to hand back the position advanced past what it just used, which is
two return values where this seam has one. Widening `sign` to return that value
would put a value on every stateless scheme that none of them can produce, and the
alternative of a seam-shaped `sign` that quietly leaves the state where it was is
the exact failure the discipline exists to prevent. So `XMSS` (RFC 8391) exposes
`sign(secret_key, message) -> (signature, next_secret_key)` under its own name and
implements the rest of the seam unchanged. That is the expected shape for a
stateful scheme rather than a scheme that has not finished.

What that second value *is* follows the scheme's own format rather than this rule.
RFC 8391 puts the index inside the secret key, so `XMSS` returns an advanced key;
SHRINCS's secret key has no room for a counter and its specification passes one
alongside, so `Shrincs` returns the advanced counter. Both make the same demand of
a caller — the spent value comes back visibly and has to be stored before the
signature is released — which is what the rule is for.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from frx import Array
from frx.typing import ArrayLike


@runtime_checkable
class Signature(Protocol):
    public_key_size: int  # serialized public key, in bytes
    secret_key_size: int  # serialized secret key, in bytes
    # Serialized signature, in bytes. An upper bound, because a compressed
    # scheme's signature is variable-length (Falcon); for the fixed-size schemes
    # it is the exact size.
    signature_max_size: int
    # Whether `sign` is a function of (secret key, message, context) alone. A
    # randomized scheme takes `randomness` and produces a different signature per
    # call; both FIPS 204 and FIPS 205 specify each mode, so this is a property
    # of the parameterized instance, not of the scheme family.
    deterministic: bool

    def keygen(self, seed: ArrayLike) -> tuple[Array, Array]:
        """Derive a key pair from `seed`: uint8 `[seed_size]` -> (public, secret).

        Both keys come back in the standard's encoding, sized `public_key_size`
        and `secret_key_size`. Deterministic in `seed` — the standards define
        keygen that way, and the known-answer tests reproduce published key bytes
        from a published seed.
        """
        ...

    def sign(
        self,
        secret_key: ArrayLike,
        message: ArrayLike,
        *,
        randomness: ArrayLike | None,
        context: ArrayLike | None,
    ) -> Array:
        """Sign one message: uint8 `[L]` -> uint8 `[signature_max_size]`.

        `randomness` is what separates the two modes `deterministic` reports. A
        randomized instance requires it and rejects `None`; a deterministic one
        ignores it. The seam never draws randomness itself — an implicit draw is
        how a scheme silently stops being reproducible against its KATs.

        `context` is the standard's application context string, `None` meaning
        empty. It is not a comment: it goes into the message the scheme actually
        signs.
        """
        ...

    def verify(
        self,
        public_key: ArrayLike,
        message: ArrayLike,
        signature: ArrayLike,
        *,
        context: ArrayLike | None,
    ) -> Array:
        """Verify a batch: uint8 `[B, public_key_size]`, uint8 `[B, L]`, uint8
        `[B, signature_max_size]` -> bool `[B]`.

        Every positional argument carries the leading batch axis, so entry `i` of
        the result is the verdict on `(public_key[i], message[i], signature[i])`.
        `L` is static, as it is for `hash_frx.ByteHash`, so any padding is
        data-independent. One call is one traced computation over the whole batch;
        a caller that loops has lost the point of the seam.

        `context` applies to the whole batch, matching how a verifier is deployed:
        one protocol domain, many signatures.

        A wrong rank, or a batch whose parts do not line up, is a caller mistake
        and raises; a wrong *width* is every entry's answer at once, and whether
        that answer is `False` or an exception is the implementing standard's to
        say. [`batch.py`](batch.py) holds that rule and each scheme's reading of
        it, so the six that open `verify` with it cannot drift again.
        """
        ...
