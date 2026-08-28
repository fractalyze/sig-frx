# Coding conventions

> Code, symbols, and file paths are English.

This page carries only what is specific to implementing signature schemes. The
rules every FRX consumer shares — `@jit` placement, `for` vs `lax.scan` vs
`vmap`, pytree registration mechanics, seam conformance pins, the `testing/`
layout, the comment rules — are not repeated here. They follow from FRX and XLA
semantics rather than from what a repo computes, so no repo owns them and a copy per repo is how they
drift apart. The playbook injects them at session start as
[`conventions/frx.md`](https://github.com/fractalyze/claude-plugins/blob/main/plugins/playbook/conventions/frx.md).

This page is the code. What the code is gated on is
[`testing.md`](testing.md), and what a number about it may claim is
[`measurement.md`](measurement.md).

## Batch verification is the compilation unit

A scheme's `verify` takes the whole batch and must trace as one computation. The
`@jit` boundary belongs around it, never around a per-signature body that a
driver loop calls `B` times.

This is the rule the `Signature` seam exists to enforce
([`../../sig_frx/signature.py`](../../sig_frx/signature.py)) — there is no scalar
`verify` to implement, so a Python loop over the batch axis is a bug rather than
a slow path. Batch `keygen` and `sign` with `frx.vmap` when a caller needs it;
they are not the hot path and do not get their own entry points.

There is one implementation of verification and it is the traceable one. Eager is
that same code run without `jit`, not a second path: a host-only verifier beside
it buys a caller who never compiles about a third at `B = 64`, and a caller who
does compile nothing at all — the traced path already beats eager at every batch
size measured. What it costs is two verifiers that have to agree byte for byte,
which is the divergence the known-answer gate is least able to catch: it is
self-consistent on both sides and survives every round trip, so only the
published vectors find it, and only if both paths are driven through them. The
caller who cannot amortize a compile is answered by a persistent compilation
cache, not by a second verifier.

## A value is used in the namespace it arrives in

Host values stay on numpy, traced values stay on frx, and a function does not
decide for its caller. `namespace` in
[`arrays.py`](../../sig_frx/arrays.py) is that rule as code — it reads the
namespace off the arguments instead of naming one — and `index_column` in
[`bytestring.py`](../../sig_frx/hash/bytestring.py) is the same question
asked of the values that most invite a conversion, tree and leaf indices. Every
boundary between key generation, signing and verification is one of its call
sites.

It is a rule rather than a preference because of what a lift costs. Key
generation and signing are concrete on the host, where a Python integer has no
width; verification is traced, where an integer array lane is 32 bits
([`../../CLAUDE.md`](../../CLAUDE.md)). A callee that lifts its argument onto the
device for its caller's convenience therefore drags a signing path onto a lane it
was never on, and the hypertree's tree index — the value carried as bytes
precisely because it does not fit one — arrives there with it. The failure lands
in the caller, at a value the callee never saw.

The opposite conversion announces itself: `np.asarray` on a traced value raises
under `jit`, and it costs a signing path nothing because there is no tracer
there. The lift onto the device is the one that needs a rule, because it succeeds
everywhere except on the path that cannot afford it — and pays a dispatch per
operation to batch a signature with itself even where it works.

**The callees that lift anyway, and what they had to show.**
[`secp.py`](../../sig_frx/classical/secp.py)'s `multiple` and `double_multiple`
place their point batch themselves, on a batch-size threshold, and
`lift_x_to_parity` places the square root's coordinate batch on the same one.
They are the only exceptions in the repo and they are allowed because the
hazard above cannot reach them: the rule protects a *signing* path from being
dragged onto a 32-bit integer lane, neither a point dtype nor a base-field
dtype has an integer lane, and every signing caller of those seams arrives at
`B = 1`, which is below any threshold and so never moves. What it buys is that
the decision exists once rather than at each of the five places a verification
batch is born, where a sixth that forgot would be silently slow. An exception
wants that shape of argument — why the hazard is absent, and what the
duplication would have cost — not just a measurement.

Three seams, one exception: they share a single placement function, so the
threshold cannot drift between them and the argument above is made once rather
than three times. That module's docstring is where the details live — how the
threshold was measured, and why the seams must not assume the device admits
every dtype a curve is built from. Growing the exception means adding a caller
to that function, not writing a second one.

**An operation with a host implementation picks it the same way.** A lift needs a
reason, and "the callee only has a device form" is the reason `arith.ntt` lifts:
`frx.lax.ntt` has no host implementation, so there is nowhere else for a host
argument to go. Hashing looks like that case and is not one — hash-frx ships a
`hashlib` sibling of every sponge, gated against the device row — so
[`hashes.py`](../../sig_frx/hashes.py) reads the same namespace `namespace`
does and answers with an implementation instead of a module. Which of the two a
call gets is the caller's fact, exactly as the array module is: one scheme
instance verifies under a tracer and signs concretely, so a hash fixed when the
instance is built would be fixing what belongs to the values.

What the rule does not do is drag a value home to make the cheaper side apply.
A commitment computed by the transform is a device array because the transform
has no other form, and hashing it is a device hash; bringing it back to the host
first is a decision about that value with its own cost, not this rule applied
harder.

## hash-frx is reached by its names, not by its file tree

`from hash_frx import Sha256`, never `from hash_frx.sha256 import Sha256`, and
the Bazel dep is the whole `@hash_frx//hash_frx` rather than a narrow label. The
two are one decision, and hash-frx's
[consuming page](https://github.com/fractalyze/hash-frx/blob/main/docs/reference/consuming.md)
states why, along with what to do about a name its root does not export.

The `hash-frx-root-import` and `hash-frx-whole-package-dep` hooks hold both
halves, so this section is context for the rule rather than the thing enforcing
it.

## A rejection loop is not a `while` on secret data

"Sample until the candidate is in range" — ML-DSA's signing loop, Falcon's
Gaussian sampler — has no data-dependent trip count available on device. It is
sampled in fixed-size blocks with a mask, or it runs on the host. Which one a
scheme picks is a decision its page records, along with the timing consequence
([`security.md`](security.md)).

## The lattice NTT is `frx.lax.ntt`, and the adaptations are the shared part

ML-DSA's and Falcon's transforms live here; ML-KEM's lives in
[`enc-frx`](https://github.com/fractalyze/enc-frx). All three are the same op,
and what each scheme writes around it is small, identical in shape, and
different in constants — which is the part worth keeping aligned.

**Every one of them wants the same transform, and `frx.lax.ntt` is it.** They
multiply in the negacyclic ring `Z_q[X]/(X^n + 1)`, which is the op's
`NEGACYCLIC_NTT` mode. They are not even different lengths in any deep sense:
the 2-adicity of `q − 1` is 13 for ML-DSA, 8 for ML-KEM and 12 for Falcon, and a
length-`n` negacyclic transform
needs a primitive `2n`-th root — so ML-DSA gets length 256 directly, and ML-KEM
gets length 128 applied to the even and odd coefficient halves, which is exactly
what FIPS 203's "incomplete" NTT and its degree-1 base case `mod (X² − ζ)`
describe. Reframing, not a different algorithm.

**The op takes a field minted from a modulus.** Its generated kernel derives the
root from the runtime modulus rather than matching a curated family, so neither
8380417 nor 3329 needs to be curated, and neither repo hand-walks the layers.

**What the op does not decide, and a scheme therefore pins — when it has to.**
Two things, both constants rather than code:

- *Which root.* The `generator` argument is a generator of the multiplicative
  group, not the root: the op derives `g^((q−1)/2n)` itself. A standard that
  names a root — FIPS 204's `ζ = 1753`, FIPS 203's `ζ = 17` — has its preimage
  under that map pinned, searched for rather than transcribed, for the same
  reason the standards' tables are generated in these repos and not copied.
- *Which order.* The op returns natural order, `out[k] = w(ζ^(2k+1))`; a
  standard that indexes the same values by bit-reversal needs
  `lax.bit_reverse` on the transform axis and nothing else — it is an
  involution, so both directions use the same call.

**Whether either pin is required is decided by one question: does any value in
the transform domain leave the implementation?** Ask it of the scheme, not of
the transform.

| scheme | what observes the domain | pins |
|---|---|---|
| ML-DSA | `ExpandA` samples `Â` **directly from the seed**, so a public key commits to one root; `BitRev8` fixes the table order | root and order |
| ML-KEM | the same shape, at `BitRev7` | root and order |
| Falcon | nothing — the public key is a coefficient-domain `h`, the signature a compressed coefficient-domain `s`, and nothing is sampled in the domain | neither |

Where the domain is observable, leaving the root unpinned gives *a* primitive
`2n`-th root: a correct negacyclic transform against the wrong root, which
round-trips and convolves like the right one, so only the published vectors
catch it. Where it is not, every producer and consumer of a domain value sits
inside one call and the root cancels — pinning would import a convention no
standard states, and gating on a particular reference's intermediate values
would pin the repo to that implementation's private choice, which
[the byte-exactness rule](testing.md#known-answer-tests-are-the-gate) declines
to do.

The asymmetry is a property of the scheme and not of the code, so it is worth
restating where it bites: a step that begins serializing, hashing or comparing a
transform-domain value needs the pin back, and will not fail a round trip or a
convolution check on the day it does.

**The modulus widths are not a reason to write anything by hand, and they look
enough like it to be worth disarming.** 8380417 is 23-bit and 3329 is 12-bit,
but a field dtype reduces internally, so `(q − 1)² mod q` is exact for both and
no residue ever occupies a raw integer lane. `zk_dtypes.prime_field(q)` mints a
field from any modulus, curated or not, so `+`, `-` and `*` are already modular.
What that avoids is worth naming, since it is the shape the code would otherwise
have — a product of two ML-DSA residues is 2^46 against a 32-bit lane, the
frontend has no widening multiply and no 64-bit lane to hold it, so every
operation becomes a limb split carrying a bound that has to be argued.

The one thing the field dtype asks in return is that a residue is read out with
`astype` and never a bitcast: its storage is a Montgomery representative, so
reinterpreting the bytes gives a different number, and wrongly in a way no round
trip reveals. That matters wherever a scheme leaves field arithmetic for bit
manipulation — FIPS 204's rounding functions, FIPS 203's compression. It is also
why the transform takes `FIELD` and refuses a raw integer array: the op reads the
algebra, not the bytes.

The convention across all of them is that they **look alike**: same module
layout, the same names for the transform and its base multiplication (`ntt`,
`intt`, `base_mul`), and the pins above written the same way — including a
scheme that pins neither, which says so at the call site rather than differing
by an absence. The cost being avoided is not duplicated lines — there are barely
any left — but two adaptations that look unrelated, so a mistake understood in
one is never looked for in the other. It applies inside this repo as much as
across the two: `centered` is spelled identically in ML-DSA and Falcon so that
diffing them shows one modulus and nothing else.

What the aligned set deliberately stops short of is a composed `a · b`. A scheme
whose caller reuses an operand across products — ML-DSA hoists `Â`, `ĉ` and the
key polynomials — would pay a transform per call for it, and a scheme whose
batch assembles by vmapping a one-signature body would compute the same
transform once per batch row over identical data. Callers hoist and stay in the
transform domain; `intt` is applied once at the end.

There is no shared name for a reduction because neither file performs one — that
is the field dtype's job, and a wrapper named for it would be a function that
exists only to be matched across repos.

The measurement that governs this choice is the batch axis, not the backend.
This stack's NTT was built for one large transform, the shape a prover asks for;
a scheme's NTT is hundreds of length-256 ones. An implementation that
parallelizes *within* a transform and loops over transforms loses at that shape
however well it is generated, so what a measurement here is really asking is
whether the batch axis survived. The op's CPU path spreads a batch of transforms
across work groups, which is what makes it competitive with ordinary array
operations vectorized over the leading axes; a backend that regressed on that
would be a reason to revisit, and the shared shape is what keeps either outcome
a small change.

## Falcon's second transform runs in double precision

Verification lives in `Z_q` and uses the integer NTT above. Key generation and
signing work over the rationals instead, embedded in `C`, and that transform —
[`fft.py`](../../sig_frx/lattice/falcon/fft.py) — carries a requirement the
integer one does not.

The requirement belongs to the **domain and not to that one module**.
[`keygen.py`](../../sig_frx/lattice/falcon/keygen.py)'s rational arithmetic is
held to the same rule, which is why the guard — `fft.require_scope` — is public
and shared rather than copied.

**The precision is a security property, not a numerical nicety.** Falcon's
analysis assumes double precision, and `ffSampling` is where that is
load-bearing: too little of it moves the sampled distribution away from the ideal
one, which is what leaks the secret basis. A `float32` mantissa is 24 bits
against 53.

On the host that costs nothing, because numpy is `complex128` natively. Traced it
costs a scope, so a traced caller wraps the whole operation in `double_precision`
and every entry point **raises** outside it rather than returning a narrowed
result — the stack's own signal there is a warning, which is the wrong shape for
a difference a security analysis rests on. Every entry point means every one
that a caller reaches, not every one that happens to divide: a guard delegated
to whatever the callee eventually calls covers only the paths that get there,
and the paths that do not are the ones nobody thinks to check. How that scope
behaves in general
belongs to FRX rather than to this repo and is described in
[`conventions/frx.md`](https://github.com/fractalyze/claude-plugins/blob/main/plugins/playbook/conventions/frx.md).
The edge that decides the calling convention here is that it scopes an operation
and not a call, so it is the **caller's** to open: a callee that opened one and
returned would hand back a wide array that narrows on the caller's next line.

It is also the one place this repo's first non-negotiable is suspended, since an
integer lane widens inside the scope. Nothing in the transform holds integers, so
a caller that does keeps them outside it or pins the accumulator dtype.

**Byte-exactness against the reference implementation is deliberately not a
requirement**, and dropping it is what lets this module take the compiler as it
is. Signing is randomized by design — §3.9 draws the salt per signature — so two
correct implementations disagree on output by construction and there is no single
signature to reproduce. What must hold is that a signature verifies and
interoperates, which is why the gate is [the reference implementation driven as
an oracle](testing.md#a-standard-that-publishes-no-vectors-still-gets-gated)
rather than a table of expected outputs. Under `jit` a multiply-add is contracted
into a fused `fma`, which rounds once where the reference rounds twice — a result
that differs in the last place and is *more* accurate. Suppressing it is a change
global to the compiler, and one scheme's test methodology is the wrong reason to
make every other workload pay for it.

## Keys and signatures are bytes at the seam

They cross the `Signature` seam as `uint8` arrays in the standard's encoding, not
as scheme-named pytrees: a consumer holds bytes, and a seam taking a structured
form would make it call a scheme-specific decode first — which means naming the
scheme, which is what the seam exists to prevent. A scheme parses its own
encoding on entry, and any structured form it wants across calls is its own
surface, below the seam.

Inside a scheme, whatever crosses a `jit` / `vmap` boundary is a registered
frozen dataclass, and a scheme instance carried as pytree aux needs value-based
`__eq__`/`__hash__` — which is why the seam requires them. Identity equality does
not error; it silently re-traces the enclosing zone for every freshly built
instance, so it surfaces as a slow call and never as a failure.

A parameter record that never crosses a boundary stays plain data.

## The context is part of what gets signed

FIPS 204 and 205 both take an application context string, and it changes the
message the scheme signs — so `sign` and `verify` take it and a scheme never
accepts one it then ignores. A scheme whose standard has no context requires it
empty and raises otherwise.

One context per call, not per batch entry: a verifier serves one protocol domain
at a time.

The seam names the standards' plain external interface. A variant that prepares a
different message — a pre-hashed message, the internal interface, a pre-computed
message representative — is a different operation and lives under the scheme's
own name, not on the seam.

## No blockchain in a core

Ethereum's and Bitcoin's conventions — message hashing, recovery, low-`S`, DER —
ride as variants over the curve-level core. `ecdsa/` holds no `eth_` symbol, and
a chain name inside a core module is the smell that a variant leaked downward.

## Cite the standard, by section

A magic constant, a domain separator, or a padding rule carries the document and
section it comes from: `# FIPS 205 §4.2`, not `# domain separator`. Signature
code is easy to write plausibly and wrongly, and the section number is what lets
a reviewer check it rather than agree with it.

## Generalize a component when its second consumer arrives

Every seam in the hash-based layer was widened by the component built on top of
it, not when it first landed: the tree took a node-address builder once FORS
needed `FORS_TREE` addresses, and WOTS+ took a sequence of key pairs once an XMSS
tree needed all of them walked at once. Widening earlier would have meant guessing
which axis mattered, and a seam nobody asked for is harder to remove than one
absent.

The corollary is that refactoring a module days or hours old is normal here. The
question is whether a real second caller is forcing the shape, not how recent the
first version is.

## A seam field ships with the call site that reads it

Declaring a field on a Protocol and reading it nowhere does not create a seam. It
creates a comment with a type, and nothing can notice: the one implementation
returns exactly what every call site hardcodes, so each site is correct, the
tests pass, and the docstring describes an indirection that is not wired. What
finds it is a second implementation, and what it produces then is wrong output
rather than an error — a key built against an encoding the family does not use.

`TweakableHash.compressed_address`
([`tweakable.py`](../../sig_frx/hash/tweakable.py)) is where this comes
from. The SHA-2 sets compress an address to 22 bytes and the SHAKE sets keep the
full 32; the field said so from the start, and every caller passed `compressed`
by hand regardless until the SHAKE sets made the two disagree.

So a field arrives with the call site that reads it, or it stays a comment and
the value is hardcoded honestly until a second implementation forces the seam —
which is the rule above, seen from the side of the code that would have consumed
it.

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
