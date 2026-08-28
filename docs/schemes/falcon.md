# Falcon (FN-DSA draft)

NTRU-lattice signatures: hash-and-sign over `Z_q[X]/(X^n + 1)` at `q = 12289`,
where the secret key is a short basis of the lattice and a signature is a point
of that lattice near a hash of the message. What makes it a signature rather
than a way to leak the basis is that the point is *sampled* rather than rounded
to — the whole scheme turns on a discrete Gaussian whose width is set per
coordinate by the basis's own geometry.

Implementation:
[`sig_frx/lattice/falcon/falcon.py`](../../sig_frx/lattice/falcon/falcon.py) for
the seam, over six libraries in the same package — `arith.py` for `Z_q` and the
NTT, `fft.py` for the rational transform the trapdoor lives in, `bigint.py` and
`keygen.py` for the NTRU solver and the ffLDL tree, `sampler.py` for §4.4's
sampler, `sign.py` for the tree walk, and `encoding.py` for the wire formats.

## What the standard fixes, and what this implementation chooses

The standard fixes the parameter sets of Table 3.3, the encodings of §3.11,
`HashToPoint` down to the byte each candidate is read from, and every constant
the sampler compares against — `RCDT`, `ApproxExp`'s thirteen coefficients, and
`σmin` per degree. Those are transcribed, and two of them are transcribed
*rather than derived* on purpose: `σmin = σ/(1.17√q)` holds to about `1e-13`,
which checks the identity and does not reproduce a published `ccs`, and Table
3.3's `σ` itself does not round-trip to the reference's own binary constant
(`1.4e-13` at `n = 512`, `2.7e-12` at `n = 1024`). The specification's numbers
are what this uses; the gap changes no sampled integer.

**What the standard does not fix is where randomness enters, and that is this
repo's decision three times over.** Falcon states no map from a seed to a key
pair, so `keygen` expands `SHAKE256(seed ‖ attempt)` and **does not reproduce
the published KAT keys** — matching those would mean transcribing NIST's
AES-256-CTR-DRBG, which is the validation harness rather than the scheme. §3.9
draws a fresh salt per signature and the seam does not draw it: `sign` requires
`randomness`, because an implicit draw is how a scheme stops being reproducible
against its own vectors. And the sampler's stream is expanded from the salt and
the secret key, so **signatures here are not the published ones byte for byte**
— that would need the reference's ChaCha20 PRNG, which no part of the standard
fixes. Byte-exact signing was considered and declined.

So the gate is not "reproduce the published bytes", which is unavailable for two
of the three operations. It is:

- **verification against the published vectors**, which is exact and is what
  most consumers need;
- **the reference implementation, compiled and driven both ways** — it signs
  with a key generated here and `verify` accepts, and it accepts a signature
  produced here and refuses it corrupted. `testing.md` puts "the reference
  implementation the standard points at" last in the order of authorities, and
  Falcon reaches it because FN-DSA is still draft and publishes no ACVP set.

Two representation choices are this repo's and are not interchangeable with the
reference's. The ffLDL tree is held **one array per depth** rather than one
object per node — `log₂n` arrays instead of `2n−1` objects, over the same
memory. And `fft.split` pairs index `i` with `i + n/2`, a root with its
negative, where the reference pairs adjacent indices because its representation
is bit-reversed. `merge(split(f)) == f` holds for both, so a round trip cannot
tell them apart; anything walking the tree has to use this one throughout.

## Where the batch axis is

**`verify` is batch-first and it is the only operation that is.** It takes
`[B]` public keys, messages and signatures and returns `bool[B]`, decoding the
key and entering the transform once for the batch before a `vmap` over the
per-signature body. That hoist is the point: the decoder's scan and
`searchsorted` are one-dimensional, so the body is written once for one
signature rather than transcribed a second time over a batch axis.

**`keygen` and `sign` are concrete, one at a time, and a traced argument raises
rather than failing deeper.** Both carry data-dependent loops — Algorithm 5's
restart, Algorithm 10's two — and the sampler under signing draws from a table
by comparison and rejects until it accepts. None of that has a traced form, and
neither operation has a batch axis worth giving up: a key is generated once, and
`security.md` puts signing outside what this repo supports anyway. A caller who
wants many keys writes the loop.

That split is why `sampler.py` and `sign.py` are host `numpy` while everything
under `verify` is not, and why `encoding.compress` reads `numpy` directly where
`decompress` is device code — the encoder serves signing only, and `decompress`
is 69% of a GPU verification.

## What leaks

The posture is [`../reference/security.md`](../reference/security.md): this repo
is verification-grade, and key generation and signing carry **no side-channel
claim at all**. Nothing here is constant-time and no test in this stack could
establish that it were.

Named, because that page asks each scheme to name its own:

- **The sampler's rejection loop** (`sampler.sampler_z`). It draws until it
  accepts, and the acceptance probability depends on the centre — which is a
  coordinate of the secret basis. §4.4 calls the algorithm *isochronous* and the
  reference goes to real lengths for it; none of that survives being written in
  Python, and this implementation restates the property as the specification's
  rather than claiming it.
- **Algorithm 10's two loops**, which restart on the norm bound and on a
  compression that does not fit. Both trip counts are data.
- **Algorithm 5's restart** in key generation, which rejects on the norm and on
  a descent whose bottom is not coprime — measured at 16 rejections costing
  about 30 ms against one accepted attempt costing 45 s at `Falcon-512`.

`HashToPoint` is the one rejection here that does **not** leak: its candidates
are derived from the salt and the message, both public, which is what lets it
take the fixed-budget-plus-compaction shape
[`../../sig_frx/lattice/rejection.py`](../../sig_frx/lattice/rejection.py)
defines and stay on the traced path inside `verify`.
