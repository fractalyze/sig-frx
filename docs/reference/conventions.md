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

## The lattice NTT is hand-written per-repo, and the shape is the shared part

ML-DSA's NTT lives here; ML-KEM's lives in
[`enc-frx`](https://github.com/fractalyze/enc-frx). Both are hand-written, and
both duplicate a primitive this stack already ships — so the reason has to be
stated precisely, because it is not the one the two schemes' parameters suggest.

**Both schemes want the same transform, and `frx.lax.ntt` is it.** They multiply
in the negacyclic ring `Z_q[X]/(X^n + 1)`, which is the op's `NEGACYCLIC_NTT`
mode. They are not even different lengths in any deep sense: the 2-adicity of
`q − 1` is 13 for ML-DSA and 8 for ML-KEM, and a length-`n` negacyclic transform
needs a primitive `2n`-th root — so ML-DSA gets length 256 directly, and ML-KEM
gets length 128 applied to the even and odd coefficient halves, which is exactly
what FIPS 203's "incomplete" NTT and its degree-1 base case `mod (X² − ζ)`
describe. Reframing, not a different algorithm.

**What blocks the op is kernel dispatch, not the transform.** It resolves its
kernels by *curated field family* — matching a runtime field against a
compile-time list — and neither 8380417 nor 3329 is a curated family. A field
minted from a modulus is reachable everywhere else in the frontend: it round-trips
through `device_put`, it lowers with its algebra, and elementwise arithmetic on it
is exact. The NTT is one of the few ops implemented as a hand-written templated
library rather than emitted code, and that is the whole reason it is the one that
refuses. So the duplication here is a workaround with a known expiry, not a design
preference — say so, or the next reader assumes it was chosen.

**The modulus widths are not the reason, and they look enough like it to be worth
disarming.** A shared implementation is not blocked by 8380417 being 23-bit and
3329 being 12-bit: a field dtype reduces internally, so `(q − 1)² mod q` is exact
for both and no residue ever occupies a raw integer lane. The lane ceiling binds
*hand-written* arithmetic over integer lanes and nothing else, so it cannot decide
whether two schemes may share code.

Where it does bind is one level down, inside the hand-written arithmetic it forces:

| | ML-DSA (FIPS 204) | ML-KEM (FIPS 203) |
| --- | --- | --- |
| modulus | 8380417 (23-bit) | 3329 (12-bit) |
| worst-case product of two residues | 2^46 | 2^24 |
| in a 32-bit lane | **truncates** | exact |
| representation this forces | 16-bit limbs | native |

ML-DSA's Montgomery reduction needs the high half of that product, and there is no
widening multiply in the frontend and no 64-bit lane to hold it (the repo's first
non-negotiable, and note it truncates *silently* when the value arrives already
64-bit rather than with the dtype named). So the limbs are 16-bit because a product
of two of them must fit a lane — forced, not tuned. ML-KEM needs none of this. The
two files genuinely differ there, which is worth recording on both sides so the
second implementer does not read the limb split as gratuitous.

Sharing the hand-written version across repos is separately not worth it: what is
common is the layer-walk skeleton, a few dozen lines, which does not carry a
cross-repo pin. `hash-frx` is the wrong home for it in any case, being the
*symmetric* layer — a lattice NTT is not a hash and does not belong there merely
because both repos already depend on it.

So each repo implements its own, and the convention is that the two **look
alike**: same module layout, the same names (`ntt`, `intt`, `base_mul`,
`montgomery_reduce`), the same twiddle-table generation style. The cost being
avoided is not duplicated lines — it is two implementations that look unrelated,
so a bug fixed in one is never looked for in the other. Whoever writes the second
should be able to read the first.

Two triggers to revisit, and the first is the real one: when `frx.lax.ntt`
dispatches on a field minted from a modulus, both schemes collapse to one call and
these implementations should be deleted rather than maintained — keeping the shape
convention is what makes that a small change. Failing that, a third lattice scheme
is the usual signal; two implementations are not evidence for an abstraction,
three usually are.

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
