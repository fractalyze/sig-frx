# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""ML-DSA — FIPS 204 §5 and §6, assembled over the modules beneath it.

The arithmetic, the samplers and the wire formats are already here, so what is
left is Fiat-Shamir with aborts: commit to a mask, hash the commitment into a
challenge, answer it with the secret, and **start over** whenever the answer would
leak. Key generation and verification are straight-line; the loop is the whole
design question, and it is answered twice below — once for where the loop runs,
and once for what that costs a batch.

**The loop runs on the host, and it is a `while`.** Its trip count depends on the
secret, so no tracer can have it, which leaves the two shapes
[`conventions.md`](../../../docs/reference/conventions.md) allows: a fixed budget
with a mask, or the host. The samplers took the budget because their trip count
is independent of every secret — the candidates they reject are public — and here
that argument is unavailable. A speculative fixed count would also have to run
every iteration it speculated: an iteration is a matrix-vector product, an NTT
round trip and a hash, against the samplers' extra hash block, and Table 1 puts
the *expected* count at 3.85 to 5.1 with no useful upper bound. So signing is
concrete, one iteration at a time, and what leaks is the number of iterations —
recorded on [`../../../docs/schemes/ml-dsa.md`](../../../docs/schemes/ml-dsa.md)
with the rest of what this scheme leaks.

**Signing is therefore not batched, and verification is.** A `vmap` over signing
would have to give every entry the same trip count, which is the worst case in
the batch and not the average — and there is no loop to run inside a `vmap`
anyway, because the exit test is a host branch. A caller with many messages loops
in Python, which is what the seam already says by taking one message. Verification
has no such loop and takes the whole batch, as `Signature.verify` requires.

**`Â` is per public key, not per batch.** Sampling the matrix is over half the cost
of a verification, and the seam hands every entry its own public key, so a batch
genuinely holds `B` distinct matrices. The deployment that verifies many signatures
under *one* key is the case worth exploiting, and it needs a surface that says so;
that is [#23](https://github.com/fractalyze/sig-frx/issues/23)'s, and it is a
surface below the seam rather than a shape change here.

**The hint's rejection is a verdict, and it is wired.** `sigDecode` hands back
`ok`, and `verify` ANDs it into `bool[B]`. FIPS 204 §D.2 records that the draft
standard omitted this check and that omitting it makes the scheme *not* strongly
existentially unforgeable, which makes dropping the flag the one wiring mistake
here with a published precedent.

**Three interfaces, one of them on the seam.** The seam is §5's external operation
(Algorithms 1, 2 and 3), which prepends the domain separator and the context. The
other two prepare a different message, so they live here under their own names —
as they do for SLH-DSA, and for the reason
[`conventions.md`](../../../docs/reference/conventions.md) gives: a variant that
prepares a different message is a different operation. §6's internal one
(Algorithms 6, 7 and 8) prepends nothing and is what the validation program
publishes vectors against; §5.4's HashML-DSA (Algorithms 4 and 5, `hash_sign` /
`hash_verify`) signs a digest under domain separator one, with the pre-hash
function's OID ahead of it so a signature names what it answers for.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from sig_frx import context as ctx
from sig_frx import prehash
from sig_frx.arrays import namespace
from sig_frx.hashes import shake256
from sig_frx.lattice.mldsa import arith, encoding, sampling
from sig_frx.lattice.mldsa.arith import D, Q
from sig_frx.signature import Signature

# §5.2's domain separators: zero for the pure variant and one for §5.4's pre-hash
# variant, which is what keeps a pre-hash signature from verifying as a pure one.
_PURE_DOMAIN = 0
_PREHASH_DOMAIN = 1

# The 32-byte `rnd` of Algorithm 2 line 5, zero for the deterministic variant.
_RANDOMIZER_SIZE = 32

# Table 1's largest expected repetition count, as the fraction it is: ML-DSA-65
# runs the loop 5.1 times on average, so an iteration succeeds with probability
# at least 1/5.1 at every parameter set.
_WORST_ACCEPTANCE = (10, 51)

# Appendix C leaves an iteration cap optional, and this is the argument for
# having one: `sampling.budget` asked for a single success from one attempt per
# trial is the fewest iterations whose chance of producing nothing at all falls
# below the margin the samplers are sized against. One derivation of the tail
# for both loops, rather than a second one to check. Reaching it means the loop
# cannot terminate — a wrong bound or a wrong sampler — rather than an unlucky
# seed, which is what `sampling._require_enough` says about its own budget in
# the same words.
_MAX_ITERATIONS = sampling.budget(1, _WORST_ACCEPTANCE, 1)


@dataclass(frozen=True)
class MlDsaParams:
    """One column of FIPS 204 Table 1 — the eight values a parameter set fixes.

    `q`, `ζ`, `d` and `n` are fixed across the sets and live in
    [`arith.py`](arith.py); `β` and the three sizes are derived, because Table 1
    states them as expressions over these and a second copy is a second chance to
    mistype what the standard already determines.
    """

    k: int  # rows of A, and of the commitment
    ell: int  # columns of A, and of the response
    eta: int  # coefficient range of the secret vectors
    tau: int  # nonzero coefficients of the challenge
    lam: int  # collision strength of the commitment hash, in bits
    gamma1: int  # coefficient range of the mask
    gamma2: int  # low-order rounding range
    omega: int  # most nonzero coefficients the hint may carry

    def __post_init__(self) -> None:
        if (Q - 1) % (2 * self.gamma2):
            raise ValueError(
                f"γ2 ({self.gamma2}) must divide (q−1)/2: Decompose rounds to "
                f"multiples of 2γ2 and w1Encode's width is bitlen((q−1)/(2γ2)−1), "
                f"so a γ2 that does not divide silently changes the encoding"
            )

    @cached_property
    def beta(self) -> int:
        """`β = τ·η` — Table 1, the bound `c·s` cannot exceed."""
        return self.tau * self.eta

    @cached_property
    def seed_size(self) -> int:
        """`ξ` — Algorithm 6's 32-byte input."""
        return 32

    @cached_property
    def commitment_hash_size(self) -> int:
        """`λ/4` — the bytes of `c̃` (§6.2 footnote: `2λ` bits for `λ` of security)."""
        return self.lam // 4

    @cached_property
    def public_key_size(self) -> int:
        """`32 + 32k(bitlen(q−1) − d)` — Algorithm 22's output."""
        return 32 + 32 * self.k * encoding.T1_BITS

    @cached_property
    def secret_key_size(self) -> int:
        """`32 + 32 + 64 + 32((ℓ+k)·bitlen(2η) + dk)` — Algorithm 24's output."""
        return 128 + 32 * (
            (self.ell + self.k) * (2 * self.eta).bit_length() + D * self.k
        )

    @cached_property
    def signature_size(self) -> int:
        """`λ/4 + ℓ·32·(1 + bitlen(γ1 − 1)) + ω + k` — Algorithm 26's output."""
        return (
            self.commitment_hash_size
            + self.ell * 32 * (1 + (self.gamma1 - 1).bit_length())
            + self.omega
            + self.k
        )


# Table 1's three columns. The names are the standard's own, `ML-DSA-kℓ` after the
# dimensions of `A`.
PARAMETER_SETS: dict[str, MlDsaParams] = {
    "ML-DSA-44": MlDsaParams(
        k=4,
        ell=4,
        eta=2,
        tau=39,
        lam=128,
        gamma1=1 << 17,
        gamma2=(Q - 1) // 88,
        omega=80,
    ),
    "ML-DSA-65": MlDsaParams(
        k=6,
        ell=5,
        eta=4,
        tau=49,
        lam=192,
        gamma1=1 << 19,
        gamma2=(Q - 1) // 32,
        omega=55,
    ),
    "ML-DSA-87": MlDsaParams(
        k=8,
        ell=7,
        eta=2,
        tau=60,
        lam=256,
        gamma1=1 << 19,
        gamma2=(Q - 1) // 32,
        omega=75,
    ),
}


@dataclass(frozen=True)
class _SecretKey:
    """Algorithm 24's layout, parsed: `ρ ‖ K ‖ tr ‖ s1 ‖ s2 ‖ t0`.

    Plain data: built and consumed inside one call, and signing is concrete, so it
    never crosses a `jit` or `vmap` boundary and is not a registered pytree.
    """

    rho: Any  # the public seed A is expanded from
    key: Any  # K, the private seed the per-signature randomness comes from
    tr: Any  # H(pk, 64), what binds a signature to its public key
    s1: Any
    s2: Any
    t0: Any  # the d low bits of t, which the public key drops


def _h(*parts: ArrayLike, size: int) -> Any:
    """`H(parts, size)` — §3.7's SHAKE256 over the concatenation, as bytes.

    One call on the concatenation rather than a series, which §3.7's equivalence
    says is the same output. The sponge is picked the way the array module is,
    off the namespace the parts arrive in ([`hashes.py`](../../hashes.py)): the
    concrete path reads every digest back immediately and a device dispatch per
    short message is what that path cannot afford.
    """
    xnp = namespace(*parts)
    message = xnp.concatenate(
        [xnp.asarray(part, dtype=np.uint8).reshape(-1) for part in parts]
    )
    return shake256(*parts)(size).digest(message[None, :])[0]


def _exceeds(values: Any, bound: int) -> Any:
    """`||values||∞ ≥ bound` over a whole vector, in the caller's namespace.

    The rejection bounds and the verifier's `||z||∞` check are the same
    comparison read in opposite directions, so they are one expression: signing
    reads it as a host `bool`, which is what lets its loop be a loop, and
    verification reads it as a mask.
    """
    xnp = namespace(values)
    return xnp.max(arith.infinity_norm(values)) >= bound


class MlDsa:
    """ML-DSA at one parameter set — FIPS 204 §5 and §6.

    Build one with `named` rather than by hand unless a test wants parameters no
    published set has.
    """

    def __init__(self, params: MlDsaParams, *, deterministic: bool = False) -> None:
        self.params = params
        self.deterministic = deterministic
        self.public_key_size = params.public_key_size
        self.secret_key_size = params.secret_key_size
        # Exact rather than an upper bound: every ML-DSA signature is this long.
        # The seam's name carries the compressed schemes' variable length.
        self.signature_max_size = params.signature_size

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MlDsa):
            return NotImplemented
        return self.params == other.params and self.deterministic == other.deterministic

    def __hash__(self) -> int:
        return hash((type(self), self.params, self.deterministic))

    # -- key generation ----------------------------------------------------

    def keygen(self, seed: ArrayLike) -> tuple[Array, Array]:
        """`ML-DSA.KeyGen_internal` — Algorithm 6.

        `seed` is `ξ`. Algorithm 1 draws it from an RBG and calls this; the seam
        takes it, because a key that cannot be regenerated from published bytes
        cannot be gated on published bytes.
        """
        params = self.params
        xi = namespace(seed).asarray(seed, dtype=np.uint8)
        if xi.shape != (params.seed_size,):
            raise ValueError(
                f"keygen takes ξ, {params.seed_size} bytes, got shape "
                f"{tuple(xi.shape)}"
            )
        # Line 1: the domain of the expansion carries (k, ℓ), so two parameter
        # sets never expand one seed the same way.
        expanded = _h(xi, np.array([params.k, params.ell], dtype=np.uint8), size=128)
        rho, rho_prime, key = expanded[:32], expanded[32:96], expanded[96:]
        a_hat = sampling.expand_a(rho, params.k, params.ell)
        s1, s2 = sampling.expand_s(rho_prime, params.k, params.ell, params.eta)
        # Lines 5 and 6: t = A·s1 + s2, then split off the d low bits.
        products = arith.matrix_vector(a_hat, arith.ntt(arith.to_field(s1)))
        t = arith.intt(products) + arith.to_field(s2)
        t1, t0 = arith.power2round(t)
        public_key = encoding.pk_encode(rho, t1)
        return public_key, encoding.sk_encode(
            rho,
            key,
            _h(public_key, size=64),
            s1,
            s2,
            t0,
            params.eta,
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
        """`ML-DSA.Sign` — Algorithm 2: pure signing, over Algorithm 7."""
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
        """`HashML-DSA.Sign` — Algorithm 4: pre-hash signing, over Algorithm 7.

        Deliberately not on the `Signature` seam. What it signs is `PH(M)` under
        domain separator one and the function's OID, so this and `sign` over the
        same content produce signatures that do not verify as each other — which
        is the separation §5.4 introduces it for, and the reason the pre-hash
        function is a value here rather than a flag.
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
        """`ML-DSA.Sign_internal` — Algorithm 7, the rejection loop included.

        §6's interface: `M′` is signed as given, without §5's domain separator and
        context, which is what makes it the one the validation program drives. Not
        on the seam, for the reason §5 gives — an application wants the domain
        separation, and a signature over an unwrapped message is a different
        object from one over `M′`.
        """
        params = self.params
        key = self._parse_secret_key(secret_key)
        a_hat = sampling.expand_a(key.rho, params.k, params.ell)
        s1_hat = arith.ntt(arith.to_field(key.s1))
        s2_hat = arith.ntt(arith.to_field(key.s2))
        t0_hat = arith.ntt(arith.to_field(key.t0))
        # Lines 6 and 7. `tr` binds the signature to the public key, and `rnd` is
        # the only difference between the hedged and deterministic variants.
        mu = _h(key.tr, message, size=64)
        rho_prime = _h(key.key, self._randomizer(randomness), mu, size=64)

        kappa = 0
        for _ in range(_MAX_ITERATIONS):
            y = sampling.expand_mask(rho_prime, kappa, params.ell, params.gamma1)
            y_field = arith.to_field(y)
            w = arith.intt(arith.matrix_vector(a_hat, arith.ntt(y_field)))
            c_tilde = _h(
                mu,
                encoding.w1_encode(arith.high_bits(w, params.gamma2), params.gamma2),
                size=params.commitment_hash_size,
            )
            c_hat = arith.ntt(
                arith.to_field(sampling.sample_in_ball(c_tilde, params.tau))
            )
            # Line 31 increments at the end of the loop body; here it happens
            # before the rejections, where a `continue` cannot skip it.
            kappa += params.ell

            z = y_field + arith.intt(arith.base_mul(c_hat, s1_hat))
            low = w - arith.intt(arith.base_mul(c_hat, s2_hat))
            r0 = arith.low_bits(low, params.gamma2)
            # Line 23: both bounds are on `mod±` representatives, which is what
            # `infinity_norm` reduces to whichever form its argument arrives in.
            if _exceeds(z, params.gamma1 - params.beta) or _exceeds(
                r0, params.gamma2 - params.beta
            ):
                continue

            ct0 = arith.intt(arith.base_mul(c_hat, t0_hat))
            hint = arith.make_hint(-ct0, low + ct0, params.gamma2)
            # Line 28: `⟨⟨ct0⟩⟩` has to stay inside the rounding step the hint
            # corrects, and the hint has to fit the ω slots that encode it.
            if (
                _exceeds(ct0, params.gamma2)
                or int(np.asarray(hint).sum()) > params.omega
            ):
                continue

            return encoding.sig_encode(
                c_tilde, arith.centered(z), hint, params.gamma1, params.omega
            )

        raise RuntimeError(
            f"signing rejected {_MAX_ITERATIONS} consecutive candidates, which "
            f"Table 1 puts below 2^-{sampling.LOG2_SHORTFALL} — the bounds or "
            f"the samplers are wrong rather than the seed being unlucky"
        )

    def _randomizer(self, randomness: ArrayLike | None) -> Any:
        """`rnd` — Algorithm 2 line 5, and the two variants §3.4 describes.

        A hedged instance handed nothing to hedge with is an error: falling back to
        zeros would silently produce the deterministic signature, which verifies,
        so nothing downstream would notice.
        """
        if self.deterministic:
            return np.zeros(_RANDOMIZER_SIZE, dtype=np.uint8)
        if randomness is None:
            raise ValueError(
                "a hedged instance signs with caller-supplied `rnd`: pass "
                "`randomness`, or build the deterministic variant"
            )
        values = np.asarray(randomness, dtype=np.uint8).reshape(-1)
        if values.shape != (_RANDOMIZER_SIZE,):
            raise ValueError(
                f"`rnd` is {_RANDOMIZER_SIZE} bytes, got shape {tuple(values.shape)}"
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
        """`ML-DSA.Verify` — Algorithm 3 over Algorithm 8, for the whole batch."""
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
        """`HashML-DSA.Verify` — Algorithm 5: the counterpart of `hash_sign`.

        Batch-first like `verify`, and for the same reason: the pre-hash is one
        more hash over each entry's message, not a reason to leave the batch.
        """
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
        """`ML-DSA.Verify_internal` — Algorithm 8, with the batch axis added.

        §3.6.2 requires a public key or a signature of the wrong length to verify
        as false rather than to raise. A batch carries one static length, so a
        wrong one is every entry's verdict; a wrong *rank*, or a batch whose parts
        do not line up, stays an error, because that is a caller mistake and not a
        verdict the standard defines.
        """
        params = self.params
        keys = fnp.asarray(public_key, dtype=fnp.uint8)
        signatures = fnp.asarray(signature, dtype=fnp.uint8)
        bodies = fnp.asarray(messages, dtype=fnp.uint8)
        if keys.ndim != 2:
            raise ValueError(
                f"a public key batch is [B, {params.public_key_size}], got shape "
                f"{tuple(keys.shape)}"
            )
        batch = keys.shape[0]
        if signatures.ndim != 2 or signatures.shape[0] != batch:
            raise ValueError(
                f"one signature per public key, as a [B, {params.signature_size}] "
                f"batch: got {batch} keys and signatures of shape "
                f"{tuple(signatures.shape)}"
            )
        if bodies.ndim != 2 or bodies.shape[0] != batch:
            raise ValueError(
                f"one message per public key, as a [B, L] batch: got {batch} keys "
                f"and messages of shape {tuple(bodies.shape)}"
            )
        if (
            keys.shape[1] != params.public_key_size
            or signatures.shape[1] != params.signature_size
        ):
            return fnp.zeros(batch, dtype=bool)
        return frx.vmap(self._verify_one)(keys, bodies, signatures)

    def _verify_one(self, public_key: Array, message: Array, signature: Array) -> Array:
        """One entry of Algorithm 8, as the body `verify_internal` maps.

        Written for one signature and mapped rather than written over a batch axis:
        every step below is the standard's, and `vmap` is what makes the batch one
        traced computation without a second, batch-shaped transcription of the
        algorithm to keep in agreement with this one.
        """
        params = self.params
        rho, t1 = encoding.pk_decode(public_key, params.k)
        c_tilde, z, hint, ok = encoding.sig_decode(
            signature,
            params.lam,
            params.ell,
            params.k,
            params.gamma1,
            params.omega,
        )
        a_hat = sampling.expand_a(rho, params.k, params.ell)
        # Lines 6 and 7: the verifier recomputes `tr` from the key it was handed,
        # which is what binds a signature to that key rather than to any other.
        mu = _h(_h(public_key, size=64), message, size=64)
        c_hat = arith.ntt(arith.to_field(sampling.sample_in_ball(c_tilde, params.tau)))
        # Line 9: `w′Approx = Az − c·t1·2^d`, the commitment as the verifier can
        # reconstruct it. `t1·2^d` is below q, so the scaling is the field's.
        approx = arith.intt(
            arith.matrix_vector(a_hat, arith.ntt(arith.to_field(z)))
            - arith.base_mul(c_hat, arith.ntt(arith.to_field(t1 * np.int32(1 << D))))
        )
        c_tilde_prime = _h(
            mu,
            encoding.w1_encode(
                arith.use_hint(hint, approx, params.gamma2), params.gamma2
            ),
            size=params.commitment_hash_size,
        )
        # Line 13, and the hint's own rejection from line 3 — which is a verdict
        # about the input, so it lands in the same `bool` rather than raising.
        return (
            ok
            & ~_exceeds(z, params.gamma1 - params.beta)
            & fnp.all(c_tilde == c_tilde_prime)
        )

    # -- the encodings -----------------------------------------------------

    def _parse_secret_key(self, secret_key: ArrayLike) -> _SecretKey:
        """Algorithm 25, read back into the record signing walks.

        Left in the namespace it arrived in: signing is the concrete path, and a
        host secret key that this lifted would drag the whole loop onto a device
        one dispatch at a time
        ([`conventions.md`](../../../docs/reference/conventions.md)).
        """
        params = self.params
        values = namespace(secret_key).asarray(secret_key, dtype=np.uint8)
        if values.shape != (params.secret_key_size,):
            raise ValueError(
                f"a secret key is {params.secret_key_size} bytes, got shape "
                f"{tuple(values.shape)}"
            )
        rho, key, tr, s1, s2, t0 = encoding.sk_decode(
            values, params.k, params.ell, params.eta
        )
        return _SecretKey(rho=rho, key=key, tr=tr, s1=s1, s2=s2, t0=t0)


def named(name: str, *, deterministic: bool = False) -> MlDsa:
    """The Table 1 parameter set called `name`, e.g. `ML-DSA-65`.

    One family, unlike SLH-DSA's two: FIPS 204 instantiates every function of every
    set with SHAKE, so a parameter set is the only choice a caller makes.
    """
    if name not in PARAMETER_SETS:
        raise ValueError(f"{name!r} is not one of {sorted(PARAMETER_SETS)}")
    return MlDsa(PARAMETER_SETS[name], deterministic=deterministic)


if TYPE_CHECKING:
    # The seam conformance pin: mypy fails this module if it drifts from the
    # Protocol, rather than failing the consumer that calls it.
    _: type[Signature] = MlDsa
