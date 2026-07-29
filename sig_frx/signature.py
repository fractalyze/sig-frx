# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The Signature seam every scheme in this repo implements.

A digital signature scheme: `keygen` / `sign` / `verify` over key and signature
types the seam never inspects. A consumer reads the size attributes to allocate
statically and calls the three methods; it picks a scheme by construction, and
swapping SLH-DSA for ML-DSA changes that one construction rather than every call
site. This is the signature counterpart of hash-frx's `Permutation` and
`ByteHash`, and it carries the same rules.

**Verification is batch-first.** `verify` takes a leading `[B]` axis on every
argument and returns `bool[B]`; a single verification is `B = 1`, not a separate
entry point. Verification is the hot path and it is embarrassingly parallel, so
the batch is what maps onto a GPU's width. A seam that admitted a scalar `verify`
would get one implemented as a Python loop over signatures, and the parallelism
would be gone before anyone noticed. `keygen` and `sign` are not batched: they
are not the hot path, and `frx.vmap` covers the rare caller that needs it.

**Keys and signatures are opaque.** Each scheme names its own registered pytree —
SLH-DSA's key is not ML-DSA's — so they ride as type parameters the seam only
passes through. A consumer that does not name them gets `Any`, which is what a
scheme-agnostic call site wants.

Implementations MUST define value-based `__eq__`/`__hash__` over their full
parameter surface: a scheme instance rides pytree aux, where identity equality
silently re-traces the enclosing jit zone on every freshly built instance — a
cost that does not error, it just makes every call slow. A Protocol cannot
enforce this; each implementation carries it.

Each implementation module ends with a conformance pin, so mypy fails the module
that drifts from the seam rather than the consumer that calls it::

    if TYPE_CHECKING:
        _: type[Signature[PublicKey, SecretKey, Sig]] = SlhDsa
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from frx import Array
from frx.typing import ArrayLike

PubKeyT = TypeVar("PubKeyT")
SecKeyT = TypeVar("SecKeyT")
SigT = TypeVar("SigT")


@runtime_checkable
class Signature(Protocol[PubKeyT, SecKeyT, SigT]):
    public_key_size: int  # serialized public key, in bytes
    secret_key_size: int  # serialized secret key, in bytes
    # Serialized signature, in bytes. An upper bound, because a compressed
    # scheme's signature is variable-length (Falcon); for the fixed-size schemes
    # it is the exact size.
    signature_max_size: int
    # Whether `sign` is a function of (secret key, message) alone. A randomized
    # scheme takes `randomness` and produces a different signature per call; both
    # FIPS 204 and FIPS 205 specify each mode, so this is a property of the
    # parameterized instance, not of the scheme family.
    deterministic: bool

    def keygen(self, seed: ArrayLike) -> tuple[PubKeyT, SecKeyT]:
        """Derive a key pair from `seed`: uint8 `[seed_size]` -> (public, secret).

        Deterministic in `seed` — the standards define keygen that way, and the
        known-answer tests reproduce published key bytes from a published seed.
        """
        ...

    def sign(
        self, secret_key: SecKeyT, message: ArrayLike, *, randomness: ArrayLike | None
    ) -> SigT:
        """Sign one message: uint8 `[L]` -> a signature of this scheme's type.

        `randomness` is what separates the two modes `deterministic` reports. A
        randomized instance requires it and rejects `None`; a deterministic one
        ignores it. The seam never draws randomness itself — an implicit draw is
        how a scheme silently stops being reproducible against its KATs.
        """
        ...

    def verify(self, public_key: PubKeyT, message: ArrayLike, signature: SigT) -> Array:
        """Verify a batch: `[B]` keys, uint8 `[B, L]` messages, `[B]` signatures
        -> bool `[B]`.

        Every argument carries the leading batch axis, so entry `i` of the result
        is the verdict on `(public_key[i], message[i], signature[i])`. `L` is
        static, as it is for `hash_frx.ByteHash`, so any padding is
        data-independent. One call is one traced computation over the whole batch;
        a caller that loops has lost the point of the seam.
        """
        ...
