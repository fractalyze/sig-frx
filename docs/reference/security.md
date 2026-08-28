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

## An assumption a scheme rests on is the scheme's, and it gets stated

Everything above is about what this *implementation* does and does not do. A
scheme also rests on a hardness assumption, and that is not this repo's to fix —
but where the assumption is materially younger or less scrutinized than the rest
of what is implemented here, the page says so, because a reader comparing schemes
on this shelf will otherwise assume they are alike in that respect.

Most of them are. SHA-2 and SHAKE preimage and collision resistance, the
lattice problems under NTRU and module-LWE, and the discrete log each have
decades of public analysis behind their lineage, and the standards that fix them
say so.

**leanSig is the exception, and the gap is the point of naming it.** Its security
proof is conditional on Poseidon meeting a **multi-target collision-resistance**
bound in the notion the construction's analysis uses — the scheme's authors put
the figure at 170 bits. Conditional in the strong sense: if the bound does not
hold, the analysis says nothing, and they state it that way. Poseidon is an
algebraic hash published in 2019 and designed for cheap arithmetization rather
than for a wide security margin, and the Ethereum Foundation runs a multi-year
[Poseidon Cryptanalysis Initiative](https://www.poseidon-initiative.info/) whose
purpose is to test exactly this assumption — bounties on reduced-round variants,
research grants against declared gaps in the theory, and workshops.

That an assumption is under active, funded cryptanalysis is a reason to state it,
not a reason to refuse the scheme: Ethereum's post-quantum consensus is going to
verify these signatures, and a verifier that does not exist protects nobody. What
this repo owes is that the reader knows which of its schemes carries the newest
assumption, and that "post-quantum" is not silently read as "more conservative"
— it buys resistance to a quantum adversary, and it spends some margin against a
classical one to do it. The consequences for the scheme itself are on
[the leanSig page](../schemes/leansig.md).

Where a scheme's assumption is ordinary, its page says nothing about it. This
section exists for the ones where silence would mislead.

## Changing the posture

A hardened signing path is a different implementation, not a flag on this one: it
needs a language where the codegen is answerable to the source, and it cannot
share this one's compiler. Widening the claim therefore starts by deciding where
that implementation lives, not by tightening the code here.
