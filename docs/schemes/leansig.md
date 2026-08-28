# leanSig

Implementation: [`sig_frx/hash/leansig/`](../../sig_frx/hash/leansig/), a package
rather than a module because the signer's tree, the codec and the encoding
pipeline are each large enough to gate on their own.

leanSig is the signature Ethereum's lean consensus makes and checks per slot,
and it replaces BLS entirely: a **generalized XMSS** with one one-time key per
slot across a `2^32`-slot lifetime, its Winternitz chains and its Merkle tree
hashed by Poseidon over KoalaBear, its wire form SSZ. At the shipped parameters a
signature is **2536 bytes** and a public key **52**.

**What a verification costs here is not the scheme's own figure, and the gap is a
choice this implementation makes.** In the abstract the verifier finishes each
chain from the digit the codeword names, which is `v(w − 1) − T` = 122 chain
hashes; with 32 tree hashes, one message compression and 25 sponge permutations
for the leaf, that is the ~180 the construction is usually quoted at. This
verifier does not do that. [`wots.chain`](../../sig_frx/hash/wots.py) walks every
chain its full `w − 1` steps and selects, so the chain term is 46 × 7 = **322**
and a verification runs about **380** permutations — a little over twice the
quoted figure, spent so that no walk branches on the codeword.

Either number is why the scheme is here. A consensus client verifies hundreds to
thousands of these per slot, unaggregated, which is a hostile budget and an
embarrassingly parallel one.

## What the standard fixes, and what this implementation chooses

**There is no standard. There is an executable spec, pinned by commit.** No RFC,
no FIPS, no EIP, and none is planned — the migration target is a devnet series.
The authority is [`leanEthereum/leanSpec`](https://github.com/leanEthereum/leanSpec),
whose `spec/crypto/xmss/` is what pq-devnet pins and what generates the fixtures
the other clients gate on. The Rust
[`leanSig`](https://github.com/leanEthereum/leanSig) calls itself a proposal, is
unaudited, and publishes no fixtures, so it is not the authority here.

This is the mirror image of the case
[`../reference/testing.md`](../reference/testing.md#a-standard-that-publishes-no-vectors-still-gets-gated)
was written for. SHRINCS has a document and no vectors; leanSig has vectors and no
versioned document. Two consequences, and both are visible in the tree:

- **Every transcribed value records the leanSpec commit it came from**, because a
  commit is the only version there is.
- **The values are transcribed rather than fetched**, which is the moving-tag
  case [`testing.md`](../reference/testing.md#vectors-are-fetched-and-pinned-never-committed)
  names: the fixtures archive is a release asset republished in place under the
  tag `latest`, so there is no commit-pinned URL for an `http_file` to point at,
  and the provenance it would have carried lives beside the values instead.

**The technical note is the wrong source, in two ways that both produce bytes
nobody else computes.** [ePrint 2025/1332](https://eprint.iacr.org/2025/1332)
describes the construction; it does not fix the instance.

- **It is classic Poseidon ([ePrint 2019/458](https://eprint.iacr.org/2019/458),
  Hades), not Poseidon2.** The note recommends Poseidon2 and every secondhand
  description of this scheme still says so. leanSpec ships Poseidon: width 16 at
  `r_f = 8, r_p = 20`, width 24 at `r_f = 8, r_p = 23`, `α = 3`, a circulant MDS
  from a first row.
- **The shipped parameters are not the note's.** The note gives `v = 64`, from
  which the widely-quoted 3112-byte signature follows. `PROD_CONFIG` is `v = 46`,
  `w = 8`, `T = 200`, which is where 2536 comes from.

**The partial-round lane is conjugated, not changed.** leanSpec applies the
partial-round S-box to lane 0, as HorizenLabs, circomlib and Plonky3 do; hash-frx
follows ark-sponge and uses the last lane. Conjugating the MDS and the round
constants by the lane reversal is an exact rewrite that costs one index flip on a
constant — the algebra, and the condition under which this module would unwind to
passing the constants through, are in
[`poseidon.py`](../../sig_frx/hash/leansig/poseidon.py). **The consequence for
everything above it is that the permutation runs on a lane-reversed state**: the
callers that build the state place and slice from the other end, so the
convention is a layout decision rather than data movement.

**The tweakable hash family is field-typed, and that is why the shared protocols
carry a `dtype`.** Upstream states three hashes as one `tweak_hash` that
dispatches on how many digests it is handed; they are three methods here, because
the callers are three. Splitting them is what lets the first two *be*
[`tweakable.ChainHash`](../../sig_frx/hash/tweakable.py) and `NodeHash`, so
[`wots.chain`](../../sig_frx/hash/wots.py) walks leanSig's chains and
[`tree.py`](../../sig_frx/hash/tree.py) climbs its tree without either learning
that a leanSig digest is eight field elements rather than `n` bytes. The leaf is
not shared: it is a sponge with a capacity that binds it to this hashing task's
shape, not FIPS 205's `T_l`.

**The slot is a seam field.** leanSig's verifier takes the slot as an input,
where RFC 8391 XMSS carries the index inside the signature encoding — upstream
leaves it off the wire because a client verifying an attestation already knows
the slot. What that costs here is that none of the seam's existing operands could
carry it: a beacon block carries attestations from up to 32 slots back, so a
batch spans slots and the value cannot be hoisted into `context`, which is one
value per call by design. [`signature.py`](../../sig_frx/signature.py) grew
`position` for this. The message space is likewise the scheme's own — a 32-byte
root rather than arbitrary bytes — and is checked rather than assumed.

**`keygen` covers the whole lifetime and refuses a sub-range.** Upstream pads
the unbuilt part of a partial window with fresh OS randomness, so such a key is
not a function of its inputs — generating twice from one seed and one parameter
gives two different roots. A key nobody can regenerate cannot be gated, so a
partial window is refused rather than silently seeded.
[`signing.py`](../../sig_frx/hash/leansig/signing.py) holds the experiment that
established it and what supporting one would take.

**`sign` returns the signature alone**, where `Xmss` returns an advanced key and
`Shrincs` an advanced counter. In both of those the position lives inside what the
caller passed; leanSig's caller *names* the slot, so a spent one is spent at the
call site, and what moves is the prepared window —
`advance_preparation` moves it explicitly and returns the key that moved
([`leansig.py`](../../sig_frx/hash/leansig/leansig.py)).

**The signer's tree is split for storage, and it is not XMSS-MT.** One top tree
whose leaves are the roots of `2^(h/2)` bottom trees, of which the signer keeps
two resident — so state is the square root of the lifetime. The shape invites the
hypertree reading and it is wrong: the bottom roots are ordinary *leaves* of one
tree, not independently signed subtrees, so a signature carries one authentication
path of `log_lifetime` siblings served from two objects, not `d` WOTS+ signatures.

**The rejection loop is a host loop**, which is the choice
[`../reference/conventions.md`](../reference/conventions.md#a-rejection-loop-is-not-a-while-on-secret-data)
asks every scheme to record. A codeword is accepted only when its `v` digits sum
to `T`, which about one draw in nine hundred does — so the signer resamples the
randomness until one lands. It grinds a block of candidates per pass rather than stepping, and
returns the **lowest** landing attempt — upstream counts from zero and stops at
the first, so taking the last of a block would produce a valid signature that
upstream's own signer disagrees with byte for byte.

The same construction is already here under another name: SHRINCS drops WOTS+'s
checksum chains for the same constant-sum trick. The one difference decides the
cost — SHRINCS's constant is the *mode* of its distribution, so about one counter
in 65 lands, against leanSig's one in 900.

**Two encoders and both tweak builders are host-only, and cannot be otherwise.**
Each packs or decomposes a value wider than a lane — a 256-bit root, an epoch
tweak, and tweak packings that run past `2^45` — which is the first of this
repo's [four non-negotiables](../../CLAUDE.md): an array lane is 32 bits, and a
value wider than one stays a Python integer or becomes bytes. The widths live
with the code that packs them, in
[`field.py`](../../sig_frx/hash/leansig/field.py) and
[`tweakable.py`](../../sig_frx/hash/leansig/tweakable.py).

## Where the batch axis is

`verify` takes the whole batch and returns `bool[B]`, with `position` a per-entry
slot; a single verification is `B = 1`, as everywhere here.

**`verify` is an eager entrance whose work is traced.** The two encoders above
run on the host, and everything downstream of them — all 180 Poseidon calls,
across all `B` entries — is one traced computation rather than `B` dispatches.
That split is what the batch-first seam exists for: the chain walk is 46 chains
of masked full-length steps, the leaf is one width-24 sponge over 46 chain ends,
and the climb is 32 levels, every one of which is the same shape for every entry.

**The codec's rejections are folded into the verdict rather than branched on.**
A signature's three SSZ offsets and the canonicality of every field-bearing
four-byte group are checked, and the result is a flag that ANDs into the answer.
A tracer has no exception to raise per entry, and a batch has one static width, so
a wrong *length* raises — it is a shape — while wrong *content* is a verdict.

**`keygen` and `sign` are not batched, and are `TEST`-preset operations.** A full
`PROD` lifetime is `2^32` leaves and `2^32 · 46` chain starts; nothing generates
one, which is why the production gate is verification against the key pairs
leanSpec publishes rather than a round trip — its fixtures archive carries eight
validators and two keys each, one per role, since a one-time signature exhausts a
leaf. The `TEST` preset shortens the lifetime to `2^8` **and the codeword** —
`v = 4`, `T = 6` — so key generation and signing are cheap enough to gate for
real. Both halves matter: an implementation that reproduced every `TEST` vector
could still have sized a sponge or a target sum from that preset, which is what
the `PROD` gate exists to catch.

## What leaks

Read [`../reference/security.md`](../reference/security.md) first: signing
carries no side-channel claim in this repo, and verification needs none.

**Verification has no data-dependent control flow.** Every Winternitz chain runs
its full length and masks — the 322-against-122 the top of this page prices —
the Merkle climb runs the tree's fixed height, and the codec's refusals are a
flag rather than a branch. So the timing of a verification is a function of the
batch's shape and of nothing in the signatures or keys.

**The signer's rejection loop has a data-dependent trip count**, and it is
named here rather than defended. How many draws it takes is a function of the
message and of PRF output keyed by the secret seed
([`prf.py`](../../sig_frx/hash/leansig/prf.py)), so the count is not something
an observer can recompute — only the holder of the seed can replay attempts
below the one that landed. It is permitted for the reason every such loop here
is: signing carries no timing claim.

What *is* public is the acceptance predicate, and that is a claim about the
verifier rather than about the loop. Given the `rho` a signature carries, anyone
can recompute the codeword and check its digits sum to `T` — which is why the
verifying side holds no secret at all, and why the loop's form was free to be a
host loop rather than a masked fixed-size sample.

**Signing is deterministic, and that is a security property rather than a
convenience.** A repeated slot yields the same signature rather than a second
one, where a fresh draw would produce two signatures over two distinct codewords
— the multi-target opening a one-time key cannot survive.
[`prf.py`](../../sig_frx/hash/leansig/prf.py) has the derivation and its two
subdomains.

**The format discloses nothing the caller does not have.** The slot is off the
wire, the signature is a fixed 2536 bytes at every slot and every codeword, and
the authentication path is the tree's full height regardless of which bottom tree
served it.

## What this scheme rests on

Named here because
[`../reference/security.md`](../reference/security.md#what-each-scheme-owes) asks
a scheme to, and leanSig is the one on this shelf that owes it.

Its security proof is conditional on **Poseidon meeting a multi-target
collision-resistance bound** in the notion the construction's analysis uses — the
authors put the figure at 170 bits, and they state the condition in the strong
form: if the bound does not hold, the analysis says nothing. Poseidon is an
algebraic hash published in 2019 and designed for cheap arithmetization rather
than for a wide security margin, and the Ethereum Foundation runs a multi-year
[Poseidon Cryptanalysis Initiative](https://www.poseidon-initiative.info/) whose
purpose is to test exactly this assumption.

That an assumption is under active, funded cryptanalysis is a reason to state it
and not a reason to refuse the scheme — Ethereum's post-quantum consensus is
going to verify these signatures, and a verifier that does not exist protects
nobody. What is worth a reader's attention is that "post-quantum" is not the same
as "more conservative": it buys resistance to a quantum adversary and spends some
classical margin to do it. There is nothing in this implementation to fix; the
assumption is the scheme's.

## What gates it

Nine vector modules under
[`testing/`](../../sig_frx/hash/leansig/testing/), each carrying the leanSpec
commit its values came from and the calls that produced them. What comes from the
**published** fixtures archive is the Poseidon permutation, the `PROD` SSZ
containers, and the two `PROD` key pairs; everything else was produced by running
the reference implementation at the same pin — including the `TEST` SSZ vector,
since the archive publishes `PROD_CONFIG` only, and every signature, since the
archive's signature-shaped families are leanMultisig aggregate proofs rather than
XMSS signatures.

The known-answer test is
[`leansig_kat_test`](../../sig_frx/hash/leansig/testing/leansig_kat_test.py):
verification at `PROD_CONFIG` under two published keys, six cases accepted and
four refused. Nine of the ten come from upstream, verdict included; the tenth is
built here, because it is the one upstream's signer loops until it avoids — a
codeword off the target-sum layer, whose Merkle opening still climbs to the key's
own root, so every check but the target sum passes. It is not driven by
[`kat.py`](../../sig_frx/testing/kat.py), for both reasons
[that page](../reference/testing.md#not-every-scheme-is-driven-by-the-shared-harness)
names — there is no published format to normalize, and a stateful scheme has no
seam-shaped `sign` for it to drive.
