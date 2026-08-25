# SHRINCS

Implementation: [`sig_frx/hash/shrincs/`](../../sig_frx/hash/shrincs/), a
package rather than a module because the two signing paths share only a public
key.

SHRINCS — *Shrunken SPHINCS*, [ePrint 2025/2203](https://eprint.iacr.org/2025/2203)
— puts two hash-based signature schemes under one 48-byte key. A **stateful**
path over a flexible XMSS tree of WOTS+C leaves produces 548 to 4619 bytes, and a
**stateless** SLH-DSA fallback produces 5777 for a signer that has lost its
state. A signature from either verifies under the same key, and the first byte
says which: `255` is stateless, and any smaller value is the height of the WOTS+C
leaf that signed.

## What the standard fixes, and what this implementation chooses

**The standard is a draft, and the authority is its reference implementation.**
SHRINCS publishes no test vectors — its specification says they are still
outstanding and required before it can leave Draft — and no validation program
covers it. So this is the third case in
[`../reference/testing.md`](../reference/testing.md#a-standard-that-publishes-no-vectors-still-gets-gated):
the reference implementation the standard points at, which that document names
as normative in as many words. The vectors modules under
[`testing/`](../../sig_frx/hash/shrincs/testing/) carry the pinned commit and the
calls that produced each value.

Two hazards that follow from a moving draft, recorded because guessing at either
costs a rewrite:

- **Two SHRINCS lineages exist and disagree.** The LaTeX specification and every
  C, C++ and Rust implementation of it use PORS and UXMSS; the current BIP uses
  FORS and FXMSS, and explicitly considers and declines PORS+FP. Those
  implementations cannot cross-check this one.
- **The published numbers move.** The scheme was announced with a 324-byte
  stateful signature and its minimum is now 548.

**The stateless half is FIPS 205, at a parameter set FIPS 205 does not publish.**
`SlhDsaParams(n=16, h=45, d=5, a=13, k=10)` derives the whole of the
specification's stateless table — `h' = 9`, `m = 24`, `len = 35`, a 5776-byte
signature — and every primitive beneath it is §11.2.1's byte for byte. So
[`stateless.py`](../../sig_frx/hash/shrincs/stateless.py) is a wrapper over
[`slh_dsa.py`](../../sig_frx/hash/slhdsa/slh_dsa.py) and not a second
implementation: what it adds is the public key's split, the `sf_root` binding and
the indicator byte.

**The stateful half reuses more than it looks like.** WOTS+C's chain walk is
[`wots.chain`](../../sig_frx/hash/wots.py)'s — the same masked `w − 1` steps RFC
8391 already shares — and the address is FIPS 205's compressed 22-byte layout
under [type values of SHRINCS's own](../../sig_frx/hash/shrincs/adrs.py), which
start at 16 where the stateless types stop at 6. That gap is the domain
separation between one key's two paths. What is genuinely new is the map from a
message digest to a constant-sum index set, which is what replaces WOTS+'s
checksum chains, and the FXMSS walk.

**The FXMSS walk is this repo's, not `tree.py`'s**, for two reasons named in
[`fxmss.py`](../../sig_frx/hash/shrincs/fxmss.py): the leaf index is 64 bits
where `tree.root_from_path` takes a lane-wide column, and the depth varies per
entry where that one is a fixed height.

**Verification only.** A SHRINCS key pair cannot be generated without building
the FXMSS tree whose root is the public key's third part, and signing is
stateful — a leaf that signs twice reveals its WOTS+C secret, so a signer returns
the advanced key alongside the signature, which is two values where the seam has
one. `Shrincs` therefore implements the seam's `verify`, raises from `keygen`,
has no `sign`, and carries no conformance pin — the shape
[`signature.py`](../../sig_frx/signature.py) describes for a stateful scheme and
that `Xmss` already has.

## Where the batch axis is

`verify` takes the whole batch and returns `bool[B]`, and a single verification
is `B = 1`, as everywhere here.

**Both paths run for every entry.** A traced program cannot branch on a byte it
has not seen, so each entry is verified statelessly *and* statefully and keeps
the verdict its indicator asks for. Partitioning on the host instead would make
the program's shape depend on the batch's composition — a recompile per mix, and
no longer one traced computation over the batch.

The two paths are not the same price, and which is dearer depends on what is
counted. In SHA-256 compressions the stateless path dominates: roughly three
thousand against a stateful signature's under a thousand. In wall clock it
inverts — measured warm at `B = 4`, the stateless leg is about a quarter of a
verification and the stateful one the rest — because the stateful path is bound
by the 255 sequential steps `FXMSS_HEIGHT` forces on every signature whatever
depth it used, rather than by the hashing in them. So running both costs a batch
of stateful signatures about a third again over verifying it alone, not the four
times the hash count suggests.

That the walk is dispatch-bound is also where the headroom is: most of its time
goes to building 255 address batches rather than to hashing them.

**One place the shape depends on the data.** The leaf-index field is one to eight
bytes wide, chosen by the indicator, so the FXMSS signature begins at a
per-entry offset and is gathered rather than sliced. The index itself is gathered
right-aligned into eight bytes and stays bytes from there on, because it does not
fit an array lane — see [`bytestring.py`](../../sig_frx/hash/bytestring.py) and
the first of this repo's four non-negotiables.

## What leaks

Verification has no data-dependent control flow. Every WOTS+C chain runs its full
length, the Merkle walk runs to the format's maximum depth and masks, and both
paths run for every entry — so the timing of a verification is a function of the
batch's shape and of nothing in the signatures.

What the *format* discloses is a different matter, and it is the signer's
concern rather than this code's. A stateful signature carries its leaf's height
and index in the clear, so its length reveals where in the tree it was made:
under an unbalanced tree that is a running count of how many signatures the key
has issued. The specification recommends the two prescribed shapes for exactly
this reason. Falling back to the stateless path is visible too — the signature is
five times longer.

Signing carries no side-channel claim in this repo
([`../reference/security.md`](../reference/security.md)), and none is implemented
here anyway. Were it, the grinding loop would be the operation to look at: it
runs a data-dependent number of times, and the counter it lands on ships in the
signature.
