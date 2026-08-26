# ML-DSA (FIPS 204)

Module-lattice signatures: Fiat-Shamir with aborts over the ring
`Z_q[X]/(X^256 + 1)` at `q = 8380417`. A signature is a short vector answering a
challenge derived from a commitment, and "short" is enforced by rejecting and
starting over — which is the one control-flow decision the whole scheme turns on.

Implementation: [`sig_frx/lattice/mldsa/ml_dsa.py`](../../sig_frx/lattice/mldsa/ml_dsa.py),
over the three libraries in the same package — `arith.py` for `Z_q` and the
transform, `sampling.py` for §7.3's pseudorandom sampling, and `encoding.py` for
the wire formats. The transform's own two pins, the root and the ordering, are in
[`../reference/conventions.md`](../reference/conventions.md#the-lattice-ntt-is-frxlaxntt-and-the-adaptations-are-the-shared-part),
where ML-KEM's identical adaptation is kept alongside them.

## What the standard fixes, and what this implementation chooses

The standard fixes everything observable: Table 1's three parameter sets, the
encodings of §7.2, the domain separator and context framing of §5.2, and the
sampling of §7.3 down to the byte each candidate is read from. A signature either
reproduces what NIST published or it does not — and it does, at all three
parameter sets, against ACVP's `keyGen`, `sigGen` and `sigVer` sets. The
exhaustive run over the three is tagged `slow_kat`; the merge gate keeps key
generation at all of them and takes signing and verifying at ML-DSA-44.

Three interfaces, and the seam names one. §5's pure external operation is the
seam; HashML-DSA and §6's internal interface prepare a different message, so they
live under `hash_sign` / `hash_verify` and `sign_internal` / `verify_internal`.
The validation program publishes vectors against all three, so all three are
gated.

Two things the published sets do not reach. Their `externalMu` cases take a
pre-computed message representative in place of a message, which is an operation
nothing here implements, so the harness refuses them rather than running another
against them. And this set publishes no signature of the wrong length, unlike FIPS
205's, so §3.6.2's requirement that a wrong length be a verdict rather than an
error is covered by `ml_dsa_test` alone.

What this implementation chooses:

- **The signing loop runs on the host.** Its trip count depends on the secret, so
  no tracer can have it, which leaves the two shapes
  [`../reference/conventions.md`](../reference/conventions.md#a-rejection-loop-is-not-a-while-on-secret-data)
  allows: a fixed budget with a mask, or the host. Signing takes the host. A
  speculative fixed count has to *run* every iteration it speculates — an
  iteration is a matrix-vector product, an NTT round trip and a hash — and Table 1
  gives the expected count (3.85 to 5.1) rather than a bound. The cost of the
  choice is on the ledger below: the number of iterations is observable.
- **The samplers' loops do not run on the host, and take the budget instead.**
  Their trip count is independent of every secret — the candidates they reject are
  a function of public seed bytes — so `sampling.py` squeezes a fixed number of
  blocks whose shortfall probability is below `2^-256` and compacts the survivors
  with a gather. How many blocks are squeezed and how many candidates are examined
  is then the same for every seed. Two loops in one scheme, resolved opposite ways,
  because the question is what the trip count is a function of and not what the
  loop looks like.
- **The loop's cap is derived, not chosen.** Appendix C leaves an iteration bound
  optional; this one exists so that a wrong rejection bound is an error rather than
  a hang, and it is sized the way the samplers' budget is — the smallest count
  whose consecutive-rejection probability falls below `2^-256`, in integer
  arithmetic, since §3.6.4 bars floating point from this scheme.
- **Signing is not batched.** A `vmap` over signing would give every entry the
  same trip count, which is the batch's worst case rather than its average, and
  there is no loop to run inside a `vmap` at all because the exit test is a host
  branch. A caller with many messages loops in Python. This is the seam's own
  shape — `sign` takes one message — arrived at from the scheme's side.
- **A parameter set is a table column, not a subclass.** `PARAMETER_SETS` holds
  Table 1's eight normative values per set and derives the rest: `β = τ·η`, and the
  three sizes Table 2 publishes. The derivations are pinned against Table 2's own
  columns, because a derivation nobody checks against the standard is a formula
  rather than a size.
- **The hedged and deterministic variants are one instance attribute.** §3.4 makes
  `rnd` the only difference between them, so `deterministic` selects Algorithm 2
  line 5's substitution. A hedged instance never draws its own randomness: the seam
  takes it, because an implicit draw is how a scheme stops being reproducible
  against its vectors.
- **The hint's rejection is a verdict, and it is wired into `bool[B]`.**
  `sigDecode` returns `ok` as its fourth value and `verify` ANDs it in. FIPS 204
  §D.2 records that the draft standard omitted this check and that omitting it
  makes the scheme not strongly existentially unforgeable, which makes dropping the
  flag the one wiring mistake here with a published precedent.
- **The interfaces that prepare a different message are not on the seam.** §6's
  internal pair prepends neither separator nor context and is what the validation
  program drives, so it lives under `sign_internal` / `verify_internal`; §5.4's
  `hash_sign` / `hash_verify` sign a digest under domain separator one and the
  pre-hash function's OID. The separator exists precisely so the three cannot
  verify as each other, which is why they do not share an entry point.
- **A pre-hash function is a value, and five of the twelve are available.** The
  OID is inside what gets signed, so a case names the function it answers for and
  no stand-in computes the right message for a function this repo cannot compute.
  FIPS 204 enumerates three of them and writes `case …` for the rest, so the
  remaining constants — the OIDs off the NIST CSOR arc, and the length a XOF is
  squeezed for — are pinned in
  [`sig_frx/prehash.py`](../../sig_frx/prehash.py) with where each comes from.
  What the standard's §5.4 footnote asks and this does not enforce is the pairing:
  a digest shorter than `2λ` bits is below the strength its parameter set claims,
  and the published sets pair the two freely, so it is a deployment's choice
  rather than something an implementation may refuse.

## Where the batch axis is

`verify` is batch-first, as the seam requires: `uint8[B, pk]`, `uint8[B, L]` and
`uint8[B, sig]` in, `bool[B]` out. Every argument carries the batch, the public key
included, so each entry brings its own `ρ` and its own `t1`. The context is one
value per call, because a verifier serves one protocol domain at a time.

Under the seam, one entry's verification is written the way Algorithm 8 writes it
and `frx.vmap` supplies the batch. That is a choice about where the batch lives:
a second, batch-shaped transcription of the algorithm would have to be kept in
agreement with the first, and the reshaping it would buy is one `vmap` already
performs.

The samplers have their own axis underneath it. Each takes one seed and returns one
matrix, vector or polynomial the way FIPS 204 defines it, and the axis *inside* each
is over the streams that seed fans out into — `ExpandA`'s `k·ℓ` independent entries
are one batched sponge call and one batched compaction, not a Python loop over
entries.

**`Â` is per public key, and that is the cost worth naming.** Sampling the matrix
is over half of a compiled `B = 64` verification at ML-DSA-65 on this repo's CPU
path — and because the seam hands every entry its
own public key, a batch genuinely holds `B` distinct matrices. The deployment that
verifies many signatures under *one* key can sample it once and close over it, but
that needs a surface which says the keys are equal; the seam cannot, and inferring
it would be a data-dependent branch. So it is a surface below the seam and it
belongs to [#23](https://github.com/fractalyze/sig-frx/issues/23).

The traced program costs a compile per `(parameter set, batch size, message
length)` — seconds rather than the tens of milliseconds it then runs in — which the
caller verifying one signature at a time amortizes over the fewest calls. A
persistent FRX compilation cache (`FRX_COMPILATION_CACHE_DIR`) makes it payable
once across processes rather than once per process.

`keygen` and `sign` take one seed and one message. They are the concrete path: the
values stay in the namespace they arrive in, the loop's exit test is a Python
`bool`, and the lift onto a device happens only where an opcode forces it — every
hash and every transform.

## What leaks

Read [`../reference/security.md`](../reference/security.md) first: signing carries
no side-channel claim in this repo, and verification needs none.

ML-DSA is the case that posture was written for. Signing's iteration count is a
function of the secret — that is what Fiat-Shamir with aborts *is* — so a signature
that took six iterations is distinguishable from one that took three by the time it
takes and by the counter `κ` the mask is expanded from. This implementation makes
no attempt to hide it, and could not: the loop is a host branch on a concrete value.

What does not leak, and is worth stating because the two loops look alike: the
samplers' rejection consumes a fixed number of candidates whatever the seed, so
`ExpandA`, `ExpandS`, `ExpandMask` and `SampleInBall` take the same work every
time. That is a consequence of the fixed budget rather than a claim about the
machine.

Verification touches no secret at all. Its inputs are a public key, a message and a
signature, and its one data-dependent value — the hint — arrives in the signature.

Still not claimed anywhere: constant-time execution, memory hygiene for `s1`, `s2`,
`t0` and `K`, and resistance to fault or physical attack.
