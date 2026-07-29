# Coding conventions

> Code, symbols, and file paths are English.

This page carries only what is specific to implementing signature schemes. The
rules every FRX consumer shares — `@jit` placement, `for` vs `lax.scan` vs
`vmap`, pytree registration mechanics, seam conformance pins, the `testing/`
layout, the comment rules — are not repeated here. They are identical in every
repo built on FRX, and a copy per repo is exactly how they drift apart.

## Batch verification is the compilation unit

A scheme's `verify` takes the whole batch and must trace as one computation. The
`@jit` boundary belongs around it, never around a per-signature body that a
driver loop calls `B` times.

This is the rule the `Signature` seam exists to enforce
([`../../sig_frx/signature.py`](../../sig_frx/signature.py)) — there is no scalar
`verify` to implement, so a Python loop over the batch axis is a bug rather than
a slow path. Batch `keygen` and `sign` with `frx.vmap` when a caller needs it;
they are not the hot path and do not get their own entry points.

## A rejection loop is not a `while` on secret data

"Sample until the candidate is in range" — ML-DSA's signing loop, Falcon's
Gaussian sampler — has no data-dependent trip count available on device. It is
sampled in fixed-size blocks with a mask, or it runs on the host. Which one a
scheme picks is a decision its page records, along with the timing consequence
([`security.md`](security.md)).

## Keys and signatures are the pytrees

They are what crosses a `jit` / `vmap` boundary here, so each is a registered
frozen dataclass, and a scheme instance carried as pytree aux needs value-based
`__eq__`/`__hash__` — which is why the `Signature` seam requires them. Identity
equality does not error; it silently re-traces the enclosing zone for every
freshly built instance, so it surfaces as a slow call and never as a failure.

A parameter record that never crosses a boundary stays plain data.

## No blockchain in a core

Ethereum's and Bitcoin's conventions — message hashing, recovery, low-`S`, DER —
ride as variants over the curve-level core. `ecdsa/` holds no `eth_` symbol, and
a chain name inside a core module is the smell that a variant leaked downward.

## Cite the standard, by section

A magic constant, a domain separator, or a padding rule carries the document and
section it comes from: `# FIPS 205 §4.2`, not `# domain separator`. Signature
code is easy to write plausibly and wrongly, and the section number is what lets
a reviewer check it rather than agree with it.

## Known-answer tests are the gate

A scheme that reproduces every published signature byte and accepts every
published vector has proven nothing about rejection: a verifier that returns
`True` unconditionally passes all of it. Corrupt the signature, the message, and
the public key, and pin the verdicts — the negative cases are half the gate.

Self-consistency is not evidence either. Sign-then-verify round-trips forever
inside a self-consistent wrong implementation. Property-based tests supplement
the KATs; they never replace them.

An exhaustive sweep — every parameter set against every published vector — is
tagged `slow_kat`, which drops it from the per-PR run and keeps it in the
scheduled one.

## Scheme doc skeleton

Every page in [`../schemes/`](../schemes) answers three things, and everything
else is optional — don't pad to fill a template.

- **What the standard fixes and what this implementation chooses.** Parameter
  sets, encodings, and domain separators are the standard's; the batching, the
  loop shapes, and the pytree layout are this repo's.
- **Where the batch axis is.** Which operations are batched, what shape they take,
  and what a caller vmaps if they need more.
- **What leaks.** The scheme's data-dependent operations, named — see
  [`security.md`](security.md).
