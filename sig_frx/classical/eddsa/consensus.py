# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Ed25519's consensus-relevant verification rules, as named constructions.

Ed25519 verification is not interoperable, and not because anyone
implemented it badly. RFC 8032 permits both the cofactored and the
cofactorless equation, says nothing about small-order points, and its
§5.1.3 decoding refusals are read differently by the implementations that
matter — so different libraries accept different sets of signatures. In a
consensus system that is a chain split, which is the situation ZIP-215
exists to end.

A caller therefore names the rule instead of setting a flag. `strict=True`
would read as a robustness knob and invite being chosen carelessly; a
construction reads as what it is, a commitment to one authority's accept
set, and it is the thing a protocol specification can point at.

## Who accepts what

The interoperability vectors are the measurement, not an illustration —
they exist to make implementations disagree, and the counts below are this
repo's own verdicts over the 914 of them
([`../testing/ed25519_cctv_vectors.py`](../testing/ed25519_cctv_vectors.py)).

| Construction    | Authority                     | Accepts |
| --------------- | ----------------------------- | ------- |
| `Ed25519`       | RFC 8032, read literally      | 172     |
| `Ed25519Strict` | ed25519-dalek `verify_strict` | 43      |
| `Ed25519Zip215` | ZIP-215                       | 826     |

`Ed25519` refuses a non-canonical `A` or `R` and a low-order residue.
`Ed25519Strict` refuses all of those, and any low-order `A` or `R` besides.
`Ed25519Zip215` refuses only a `k` computed from a re-encoded `A` or `R`.

The nesting is not a coincidence and it is worth reading off the table:
ZIP-215 accepts strictly more than RFC 8032, which accepts strictly more
than `verify_strict`. Cofactoring the equation admits *more* solutions
rather than fewer, so every signature an honest signer produces is accepted
by all three — the choice only ever moves signatures that no honest signer
emits. That is why `sign` and `keygen` are shared here and only `verify`
differs.

The one rule none of these takes is re-encoding `A` or `R` before hashing
them into `k`. No known validator does, and the vectors carry a
`reencoded_k` class specifically to catch one that started.

## Why only ZIP-215 gets an aggregate check

`Ed25519Zip215.aggregate_verify` is a random linear combination over the
whole batch — one verdict, one multi-scalar equation — and it exists there
and nowhere else on purpose. Cofactoring puts every per-signature residual
in the prime-order subgroup, which is what makes the combination vanish
exactly when every row would have been accepted individually. Under a
cofactorless rule the residuals keep their torsion components and an
aggregate over them accepts batches the per-signature check would not, so
the two notions of verification answer differently. Making single and batch
agree is the reason ZIP-215 mandates the cofactored equation rather than
permitting it, and an `aggregate_verify` on the other two would be a method
that quietly contradicts the `verify` beside it.

## The rule libsodium and Go implement is deliberately absent

The most widespread behaviour — ref10's, inherited by Go, OpenSSL and
libsodium — accepts a non-canonical `A` while refusing a non-canonical `R`,
because it never range-checks `A`'s `y` and catches `R` only by comparing a
re-encoding. It is a description of an implementation rather than a rule
anyone specified, and reproducing it means reproducing the asymmetry on
purpose. It lands when a consumer needs to match a deployed system
(fractalyze/sig-frx#35), not before: an accept set nobody asked for is a
consensus rule waiting to be picked by accident.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from frx.typing import ArrayLike

from sig_frx.classical import edwards
from sig_frx.classical.eddsa import ed25519

# ZIP-215 §Specification: `A` and `R` must encode points on the curve — not
# that they encode them canonically, the ZIP says so outright — `S < L`, and
# the cofactored equation, which it makes mandatory rather than optional
# ("the alternate validation equation ... MUST NOT be used"). Cofactoring is
# what lets a batch aggregate, which is the change's whole point.
ZIP_215 = ed25519.ValidationRule(
    canonical_encodings=False, reject_small_order=False, cofactored=True
)

# ed25519-dalek's `verify_strict`: RFC 8032's decoding and equation, plus a
# refusal of any small-order `A` or `R`. The property it buys is the one RFC
# 8032 does not offer — a signature binds to one key and one message — and
# the cost is that a key with a torsion component becomes unusable.
VERIFY_STRICT = ed25519.ValidationRule(
    canonical_encodings=True, reject_small_order=True, cofactored=False
)


@dataclass(frozen=True)
class Ed25519Zip215(ed25519.Ed25519):
    """Ed25519 under ZIP-215's validation rules.

    Signing is RFC 8032's, unchanged: this is a statement about which
    signatures are accepted, and every signature `sign` produces is accepted
    by all three constructions here.

    This is also the only construction with an `aggregate_verify`, and that
    is the point of the ZIP rather than an omission elsewhere — see the
    method.
    """

    rule = ZIP_215

    def aggregate_verify(
        self,
        public_key: ArrayLike,
        message: ArrayLike,
        signature: ArrayLike,
    ) -> bool:
        """One verdict for the whole batch, agreeing with `verify` row by row.

        The check is `Σ zᵢ·([8][Sᵢ]B − [8]Rᵢ − [8][kᵢ]Aᵢ) = O` over
        coefficients fixed after the batch: each per-signature residual is
        zero exactly when `verify` accepts that row, so a batch of accepted
        signatures passes, and one that carries a rejected row fails unless
        the combination vanishes by chance — which needs a random relation
        in a prime-order group.

        *Prime-order* is what the cofactor multiplication buys and why only
        this construction offers the method. Under a cofactorless rule the
        residuals keep their torsion components, an aggregate over them
        accepts combinations no per-signature check would, and the two
        notions of verification stop agreeing. That divergence is the reason
        ZIP-215 makes `[8]` mandatory rather than optional.

        `False` names no row; a caller that needs the culprit uses `verify`.
        The coefficients are drawn from a hash of the entire batch rather
        than a generator, which keeps the verdict reproducible and leaves
        the seam's no-implicit-randomness rule intact — the same derivation
        BIP-340's aggregate uses (`../schnorr/bip340.py`).
        """
        curve = self.curve
        parsed = self._parsed(public_key, message, signature)
        if not bool(np.all(parsed.ok)):
            return False

        # Cofactoring first is what makes the reduction below exact: [8]R
        # and [8]A carry no torsion, so `multiple`'s `% L` is the group's
        # own reduction on them. Scaling by 8 afterwards would not be —
        # `(8z) mod L` is not `8z` on a torsion component.
        eight_r = edwards.mul_by_cofactor(curve, parsed.point_r)
        eight_a = edwards.mul_by_cofactor(curve, parsed.point_a)

        coefficients = _coefficients(
            curve.order,
            len(parsed.s_ints),
            np.asarray(public_key, dtype=np.uint8).tobytes()
            + np.asarray(message, dtype=np.uint8).tobytes()
            + np.asarray(signature, dtype=np.uint8).tobytes(),
        )
        combined = sum(z * s for z, s in zip(coefficients, parsed.s_ints)) % curve.order
        lhs = edwards.mul_by_cofactor(
            curve, edwards.multiple(curve, [combined], curve.generator)
        )
        terms = np.concatenate(
            [
                edwards.multiple(curve, coefficients, eight_r),
                edwards.multiple(
                    curve,
                    [z * k for z, k in zip(coefficients, parsed.k_ints)],
                    eight_a,
                ),
            ]
        )
        return bool(np.asarray(lhs == edwards.sum_points(curve, terms))[0])


def _coefficients(order: int, count: int, batch: bytes) -> list[int]:
    """`count` scalars in `[1, L-1]`, fixed only after every byte of `batch`.

    The first is 1: one coefficient may be fixed without weakening the
    combination, since a single wrong row is caught by its own residual and
    two or more still have to satisfy a relation in the remaining random
    scalars.
    """
    seed = hashlib.sha512(batch).digest()
    return [1] + [
        1
        + int.from_bytes(
            hashlib.sha512(seed + index.to_bytes(8, "big")).digest(), "big"
        )
        % (order - 1)
        for index in range(1, count)
    ]


@dataclass(frozen=True)
class Ed25519Strict(ed25519.Ed25519):
    """Ed25519 under ed25519-dalek's `verify_strict` rules.

    Signing is RFC 8032's, unchanged — see `Ed25519Zip215`.
    """

    rule = VERIFY_STRICT
