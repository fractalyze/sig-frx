# What gates a scheme

> Code, symbols, and file paths are English.

Standards-exact or it is not done, and this page is what "done" is checked
against: the component tests that localize a failure, the published vectors that
find it, and what the authority becomes when a standard publishes nothing to
gate on.

The rules for writing the code are [`conventions.md`](conventions.md). How long
a test is allowed to take is a budget rather than a gate, and lives on
[`measurement.md`](measurement.md).

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

### A signature the standard does not fix is verified, not compared

Reproducing the published signature byte for byte is the signing gate wherever a
standard determines one signature per `(key, message, randomness)` — the
deterministic modes by construction, and the hedged ones once the published
randomness is fed back in. Not every standard does. Falcon draws a salt per
signature and expands the sampler's stream from it by a route §3.9 never fixes,
so two correct implementations disagree on the output bytes and the published
signature is one valid answer among many. A byte comparison there fails a
correct signer, which is a broken gate rather than a strict one.

So the call site declares it, the way it declares an interface, and the harness
checks the produced signature with the scheme's own verifier instead. That is
the round trip this page calls no evidence, and three things are what keep it
from being only that: the verifier it leans on is gated independently, against
those same published signatures; the keypair is upstream's, so a pass binds the
signer to a key it did not choose; and the declaration is held to its set like
any other — a call whose cases all reproduce their published bytes is told to
compare them rather than allowed the weaker check.

It remains the weaker claim and does not stand alone. Where a reference
implementation exists, what carries the gate is that implementation **accepting
what this repo produces** — the authority order below, applied to the one
operation the published set cannot pin. Signing is not gated by the round trip;
it is gated there, and the round trip is what the published corpus adds on top.

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

**When the only URL upstream offers is a moving one, transcribe instead of
fetching.** Some publishers ship a real artifact with no fixed-version address —
leanSpec republishes its fixtures archive in place under the tag `latest`, and
Falcon's round-3 submission is frozen by the NIST process rather than by a commit
([`MODULE.bazel`](../../MODULE.bazel) records the second). An `http_file` against
such a URL is the mystery failure the paragraph above forbids, one upstream
re-spin away. So the values become transcribed constants, and what would have
been the `http_file`'s provenance goes where they live instead: the archive's
sha256, its size, its publication date, and the exact call each value came from.
That is the same bar, paid in a different place — not a lighter one.

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
