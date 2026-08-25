# Coding conventions

> Code, symbols, and file paths are English.

This page carries only what is specific to implementing signature schemes. The
rules every FRX consumer shares — `@jit` placement, `for` vs `lax.scan` vs
`vmap`, pytree registration mechanics, seam conformance pins, the `testing/`
layout, the comment rules — are not repeated here. They follow from FRX and XLA
semantics rather than from what a repo computes, so no repo owns them and a copy per repo is how they
drift apart. The playbook injects them at session start as
[`conventions/frx.md`](https://github.com/fractalyze/claude-plugins/blob/main/plugins/playbook/conventions/frx.md).

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
[`bytestring.py`](../../sig_frx/hashbased/bytestring.py) is the same question
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

**The one callee that lifts anyway, and what it had to show.**
[`secp.py`](../../sig_frx/classical/secp.py)'s `multiple` and `double_multiple`
place their point batch themselves, on a batch-size threshold. It is the only
exception in the repo and it is allowed because the hazard above cannot reach
it: the rule protects a *signing* path from being dragged onto a 32-bit
integer lane, a point dtype has no integer lane, and the only signing caller
of those seams arrives at `B = 1`, which is below any threshold and so never
moves. What it buys is that the decision exists once rather than at each of
the five places a verification batch is born, where a sixth that forgot would
be silently slow. An exception wants that shape of argument — why the hazard
is absent, and what the duplication would have cost — not just a measurement.

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
[the byte-exactness rule](#known-answer-tests-are-the-gate) declines to do.

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

## Test the reshaped form against the standard's own form

Almost nothing here computes the way its standard writes it: a hash chain runs
its full length and masks instead of stopping at a digit, a Merkle tree iterates
where the standard recurses, a forest of trees reduces in one pass, a batch of key
pairs advances together. Each of those is a change made for the compiler, and each
is the only thing about the module a reader has to take on trust.

So take it back: transcribe the standard's algorithm — naively, looping the way it
loops, one item at a time — into the test, and require the two to agree. Cover the
cases the reshaping papers over, which for a masked loop means a full pass, a
partial one, a no-op and a mid-sequence start.

Transcribe from the document, not from memory. The specifications are public and a
named algorithm is two commands away:

```sh
curl -sSfL -o spec.pdf <url> && pdftotext -layout spec.pdf spec.txt
awk '/Algorithm 5 chain/,/Algorithm 6/' spec.txt
```

What this method cannot catch is a misreading shared by both sides, since one
author wrote both. Only published vectors close that gap — so a component's tests
localize a failure, and the known-answer tests are what find it.

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
([`tweakable.py`](../../sig_frx/hashbased/tweakable.py)) is where this comes
from. The SHA-2 sets compress an address to 22 bytes and the SHAKE sets keep the
full 32; the field said so from the start, and every caller passed `compressed`
by hand regardless until the SHAKE sets made the two disagree.

So a field arrives with the call site that reads it, or it stays a comment and
the value is hardcoded honestly until a second implementation forces the seam —
which is the rule above, seen from the side of the code that would have consumed
it.

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

### The batch axis is gated on a batch the harness builds

The published sets cannot gate the property the seam exists for. A batch axis
needs one static shape, and both validation programs vary the message length per
case — deliberately — so grouping the published cases yields nothing but `B = 1`
groups, and a check that a verdict belongs to its own entry has no second entry
to compare against. What is left covering it is each scheme verifying signatures
it produced itself, which is the self-consistency this page already says is not
evidence.

So [`kat.py`](../../sig_frx/testing/kat.py) replicates an accepted case across a
batch and moves a bit in some of the entries. The multiplicity is the only
invented part: the accepting entries carry the standard's own signature over its
own key and message, and the rejecting ones carry that signature corrupted, so a
`verify` that reduced over the batch fails and so does one that ignored its
input. It sits in the harness rather than in a scheme's tests because the gap is
a property of how the vectors are published, not of any one scheme — a
per-scheme fix would be written once per scheme for one cause.

### An operation with no accepted case is handed one

Everything the harness derives starts from a case the standard accepts, the
tampering pass and the batch axis alike, because a moved bit is evidence only
against something that verified before it moved. The validation program draws
each verification case's pre-hash function at random and publishes mostly
deliberate failures, so whole operations arrive with nothing accepted in them —
and there both passes no-op, leaving a run that compares the published verdicts
and goes green having derived nothing. Which operations those are is a property
of the draw in whichever vector set is pinned, not of the scheme: a regenerated
set reshuffles it, so it is also not a list anyone can maintain.

The case they are missing is in the same pair of files. A signing set publishes,
for every operation, a signature the standard says is the right one over a
published message under a published key — an accepted verification case in all
but the public key, which it does not carry and which its secret key determines.
So `check` takes that case from the call site and runs it as a vector like any
other, held to the same operation as the published ones. That last part is not a
formality: a pre-hash variant prepares a different message than the pure one, so
a case borrowed from a sibling operation would be rejected by a **correct**
implementation.

How a public key comes off a secret one is each scheme's own answer, and the two
kinds here are not alike. A hash-based secret key ends in the public key it was
generated with, so it is a slice, confirmed against every key pair the standard
publishes. A lattice one carries the seed and the secret vectors, so the key is
recomputed — and what makes that evidence rather than an implementation vouching
for its own input is the `tr = H(pk, 64)` the same secret key carries: a
recomputation that drifted anywhere fails against published bytes instead of
quietly producing a case that gates nothing.

Where the published set reaches an operation nowhere at all, the call site
declares it instead, one entry with its reason. The declaration buys no coverage:
an operation gated on failures alone cannot separate a verifier that rejects for
the right reason from one that rejects everything, and what holds those paths up
is the scheme's own round trip, which this page already says is not evidence.
What it buys is that the boundary is written down where a regenerated set will
trip over it — a declaration that stops describing its set fails the same way a
wrong interface does, so an accepted case arriving deletes the entry rather than
going unnoticed.

### A standard that publishes no vectors still gets gated

Not every standard ships known-answer tests, and the validation program does not
cover every scheme it approves. That lowers nothing: it changes what the authority
is, and the authority has to be named rather than assumed.

The order to look, and to stop at the first that exists: the standard's own
vectors; the validation program's; then **the reference implementation the
standard points at**, which is what a standard means when it says testing is done
against one. Shipping self-gated is not on the list — a scheme that only verifies
its own signatures has demonstrated nothing.

Gating on an implementation costs more provenance than gating on a file, so all of
it is recorded where the values live: the upstream commit, the fixture, the exact
call each value came from, and the program that regenerates them. Prefer the
values that implementation's own generator publishes over ones invented here —
they are what other implementations compare against — and pin the intermediates
beneath them too, because a digest of a final artifact says only *that* something
is wrong.

Values obtained this way are transcribed constants rather than a fetched file:
there is nothing to fetch, so the rule below does not apply to them.

### Vectors are fetched and pinned, never committed

A vector set is declared as an `http_file` in [`MODULE.bazel`](../../MODULE.bazel)
with a sha256 and a URL pinned to a specific upstream commit, and reaches a test
through `data`. The published sets run to tens of megabytes — SLH-DSA sigGen's
expected results alone are 31 MB — and committing them taxes every clone forever
for data that never changes after publication. The sha256 means a swapped or
truncated fetch fails the build rather than silently changing what a scheme is
gated on, and the repository cache makes every build after the first offline.

Pin the URL to a commit, never a branch: NIST regenerates these files in place,
and a moving URL turns an upstream regeneration into a mystery failure here.

### Not every scheme is driven by the shared harness

[`kat.py`](../../sig_frx/testing/kat.py) normalizes published *formats* into one
record so one driver gates every scheme. A scheme belongs behind it when there is
a format to normalize and the scheme implements `Signature`. Where neither holds,
the scheme's own test is the gate and says so in its docstring — which is a
narrower exception than it sounds, and it earns its place only by naming both
halves:

- **No format.** A standard with no published vector file leaves a handful of
  values transcribed from whatever authority replaces it, with the provenance in
  the module that holds them. There is nothing for a loader to parse, so routing
  them through one would add a hop and normalize nothing.
- **Not on the seam.** A stateful scheme has no seam-shaped `sign` (see
  [`signature.py`](../../sig_frx/signature.py)), and the harness signs through
  `Signature`. An adapter could satisfy the Protocol by discarding the advanced
  key — but that is not a different operation the way the internal and pre-hash
  interfaces are, it is the operation with the property that makes it safe
  removed, and it would sit in the tree one import away from a caller who wanted
  a simpler `sign`.

What such a scheme owes instead is the rest of this section in full: the negative
cases, and every rejection its own structure makes possible that a generic bit
flip would not reach.

### A vector the harness cannot run is an error, not a skip

A standard's vector set covers every mode of its interface — FIPS 204 publishes
a signing context, a pre-hash variant, and an external-mu variant, each a
different operation from the plain one. A loader records what it could not
express instead of dropping it, and the harness refuses the case. Running the
plain operation against a vector published for another one reports a pass for a
case nobody ran, which is worse than a failure because it looks like coverage.

### The per-PR gate's cost is distinct shapes, not vectors

Both validation programs vary the message length per case, so a published set is
nearly all singletons and a traced implementation compiles about once per case.
That is free on a backend that inlines the hash and ruinous on one that routes a
whole-hash marker, whose decomposition is the entire absorb and squeeze — traced
per shape, and in proportion to the message's block count rather than its length,
so two messages in one block bucket cost one compile. Hence the cost is a
property of the corpus *and* the backend: **a target excluded from a leg has
never had its budget validated there**, and the `size` it carries is a guess
until it runs, however carefully the comment argues for it.

What bounds that cost is the number of vectors per operation, not their length.
Capping the length looks equivalent and is not: a program draws each case's
pre-hash function at random and those cases carry long messages, so a length cap
empties whole operations rather than trimming them — and a gate that quietly
stopped exercising an operation reads exactly like one that got cheaper. ML-DSA's
gate keeps the shortest few of each operation for that reason, and asserts that
every operation, both signing modes and every pre-hash function survive the
bound, so the bound cannot silently eat coverage.

The exhaustive run over every published length stays behind `slow_kat`, so
nothing is lost overall — only the per-PR gate shrinks.

## A budget is sized from a target's worst run, not its typical one

A target that times out reports a red build for a suite that works, and the edit
that fixes it is always the same one line. What earns it a rule rather than a
correction each time is that the measurement people reach for is the wrong one
twice over: it is taken on a workstation, and it is taken once.

**A target whose worst observed run reaches half its budget moves up a bucket.**
The buckets are 60 / 300 / 900 / 3600 s, so moving up is coarse and cheap — a
deadline is only spent when a test hangs.

Half, rather than something tighter, is what the executor's spread makes
necessary. Across runs of *unchanged* commits, a single target's duration on the
CPU leg varies by a median of 2.2x — 1.7x to 2.6x over the targets long enough
for the variation to mean anything — while the same measurement on the GPU leg
varies by 1.07x. The CPU leg executes on a shared remote pool, so any one number
it produces may be the fast one and the next run may be twice it; half the
budget is the smallest round headroom that survives that. The GPU leg's
steadiness is a property of a quiet box rather than of the leg — one GPU runner
serves the org — so it gets the same threshold rather than a tighter one earned
by present load.

This is the quantitative half of [a target excluded from a leg has never had its
budget validated there](#the-per-pr-gates-cost-is-distinct-shapes-not-vectors).
Together: the budget covers the worst run of the slowest leg the target runs on.

### A local measurement does not decide it

The CI executor is slower than a workstation by a factor that is per-target
rather than a constant to divide out — measured at 1.6x for one target and 3.4x
for another. A target that reads as using a fifth of its budget locally can be
using two thirds where it counts. Local numbers compare two implementations;
they do not size a budget.

Bazel will argue the other way, and it argues from a local run. A target sized
for the executor trips `Test execution time … outside of range for MODERATE
tests. Consider setting timeout="short" or size="small"` on a workstation, and
`bazel test` prints the summary line that points at it. Every budget on this
page is deliberately one bucket above what that warning asks for. Taking its
advice restores the flake, so it is the one bazel diagnostic this repo overrides
on purpose rather than silences — it is right about the local number and the
local number is not the one that decides.

### `size` moves two things and `timeout` moves one

`size` picks a default deadline *and* the resource estimate bazel schedules
against when it executes a test locally. Up to `large` that estimate barely
moves and `size` is the ergonomic edit. `enormous` is where it jumps — so a
target that needs the 3600 s deadline and not the weight keeps its `size` and
declares `timeout = "eternal"`. The distinction is only visible on the GPU leg,
which runs its tests on the runner rather than on the remote pool.

### A merge run re-reports, it does not measure

Nearly every target on a push to `main` is a cache hit: the merge carries the
same content as the pull request's head, so bazel replays what that run already
measured rather than running anything. Those durations are real and simply
older — the last real execution is the one that applies the next time the target
runs, so they are what to read — but "the worst time in the last green `main`
run" is one sample produced somewhere else, under load nobody recorded.

So gather the *distinct* duration values a target reports across several runs,
not one run's table. A value that repeats verbatim is one cache entry seen
twice, not two samples, and a sweep that misses that reads the CPU leg low.

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
