# Security posture

What this repo claims, and — more importantly — what it does not. Read this
before implementing a scheme; it decides what each one owes.

## The posture

**Verification-grade.** Batch verification is the supported path. Key generation
and signing exist to reproduce known-answer tests and for development, and carry
no side-channel claim at all.

That split falls out of what the operations touch. Verification consumes a public
key, a message, and a signature — all public. There is no secret input, so
execution time, memory traffic, and the address stream carry nothing an adversary
could learn that they did not already have. Signing and key generation touch
secret key material, and this stack cannot make the timing guarantees a hardened
C implementation makes.

## Why signing cannot carry a timing claim here

The implementation is traced by FRX and compiled by XLA. Every layer between the
source and the machine is free to rewrite it:

- The compiler folds constants, reassociates arithmetic, and picks lowerings.
  Nothing in that pipeline promises data-independent instruction selection.
- A `where` is a select at the source level, not contractually at the machine
  level, and a gather's timing depends on its address stream.
- None of it is stable across compiler versions or backends, so a claim
  established once would have to be re-established on every dependency bump.

There is no test in this stack that could establish a constant-time property, and
a claim nobody can test is worse than no claim: it gets read as a guarantee.

## What is explicitly not claimed

- **Constant-time execution**, anywhere, including verification. Verification
  needs no such claim, since it holds no secret.
- **Resistance to fault injection, power and EM analysis, or microarchitectural
  attacks.**
- **Memory hygiene.** Secret material is not zeroized; a device buffer's lifetime
  belongs to the runtime.
- **Distributed key generation.** FROST's group secret is established by a
  trusted dealer (RFC 9591 Appendix C) who computes the whole key; the
  threshold property protects signing, not dealing. The decision and its
  consequences are recorded on [the FROST page](../schemes/frost.md).

Do not sign with a long-lived secret key on a machine an adversary can measure.

## What each scheme owes

- **Do not claim what the repo does not.** The words "constant-time",
  "side-channel resistant", and "hardened" do not belong in a docstring, a
  comment, or a scheme page here.
- **A scheme's page states its own leaky operations by name**, rather than
  staying silent. Rejection sampling until a candidate lands in range —
  ML-DSA's signing loop, Falcon's Gaussian sampler — has a signing time that is a
  function of the secret; it is permitted because signing carries no claim, and
  saying so is what keeps a reader from assuming otherwise.
- **Verification stays secret-free.** A verify path that takes secret material is
  a design bug, not a performance trade-off — it moves an operation out of the
  one category this repo does support.
- **Reject before you accept.** A verifier that returns `True` unconditionally
  passes every positive known-answer test, so the negative cases — a flipped bit
  in the signature, the message, and the public key — are part of a scheme's gate,
  not an optional extra.

## Changing the posture

A hardened signing path is a different implementation, not a flag on this one: it
needs a language where the codegen is answerable to the source, and it cannot
share this one's compiler. Widening the claim therefore starts by deciding where
that implementation lives, not by tightening the code here.
