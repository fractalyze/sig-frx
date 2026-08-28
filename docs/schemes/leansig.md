# leanSig

Implementation: [`sig_frx/hash/leansig/`](../../sig_frx/hash/leansig/), a package
rather than a module because the signer's tree, the codec and the encoding
pipeline are each large enough to gate on their own.

leanSig is the signature Ethereum's lean consensus makes and checks per slot,
and it replaces BLS entirely: a **generalized XMSS** with one one-time key per
slot across a `2^32`-slot lifetime, its Winternitz chains and its Merkle tree
hashed by Poseidon over KoalaBear, its wire form SSZ. At the shipped parameters a
signature is **2536 bytes**, a public key **52**, and a verification runs about
**180 Poseidon permutations** — 122 chain hashes (`v(w − 1) − T`, since the
verifier walks the steps the codeword did not), 32 tree hashes, one message
compression, and about 25 for the leaf's sponge.

That last number is why the scheme is here. A consensus client verifies hundreds
to thousands of these per slot, unaggregated, which is a hostile budget and an
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
- **The values are transcribed rather than fetched.** The fixtures archive is a
  release asset republished in place under the moving tag `latest`, so there is no
  commit-pinned URL for the `http_file` rule to point at, and that rule does not
  apply.

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
partial-round S-box to lane 0, as HorizenLabs, circomlib and Plonky3 do;
hash-frx follows ark-sponge and uses the last lane. Writing `R` for the reversal
of lane order, `R · (M · sbox₀(x + c)) = (R M R) · sbox_last(R · x + R · c)`, so
running leanSig's constants through hash-frx's engine is a matter of conjugating
the MDS and every round constant by `R` — and since `M` is circulant, `R M R` is
just the other circulant. The conjugation is exact and costs one index flip on a
constant. **So the permutation this package hands out runs on a lane-reversed
state**, and reversing at every call would put a device `reverse` either side of
180 permutations; the callers that build the state instead place and slice from
the other end, which makes the convention a layout decision rather than data
movement.

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

**`keygen` covers the whole lifetime and refuses a sub-range.** Upstream's
`key_gen` snaps a requested window out to whole bottom trees, builds those, and
fills the rest with `merkle.random_domain` — fresh OS randomness, one digest per
unpaired node per level. That is sound, and it means a partial window's public key
is not a function of its inputs: generating twice from one seed and one parameter
gives two different keys, which was confirmed against upstream by substituting the
pad source. A key nobody can regenerate cannot be gated, so a partial window is
refused rather than silently seeded. Supporting one means taking the pads as an
argument, the way `parameter` and `prf_key` are already taken.

**`sign` returns the signature alone**, where `Xmss` returns an advanced key and
`Shrincs` an advanced counter. In both of those the position lives inside what
the caller passed; leanSig's caller *names* the slot, so a spent one is spent at
the call site. What moves instead is the prepared window, and
`advance_preparation` moves it explicitly and returns the key that moved.

**The signer's tree is split for storage, and it is not XMSS-MT.** One top tree
whose leaves are the roots of `2^(h/2)` bottom trees, of which the signer keeps
two resident — so state is the square root of the lifetime. The shape invites the
hypertree reading and it is wrong: the bottom roots are ordinary *leaves* of one
tree, not independently signed subtrees, so a signature carries one authentication
path of `log_lifetime` siblings served from two objects, not `d` WOTS+ signatures.

**The rejection loop is a host loop**, which is the choice
[`../reference/conventions.md`](../reference/conventions.md#a-rejection-loop-is-not-a-while-on-secret-data)
asks every scheme to record. A codeword is accepted only when its `v` digits sum
to `T`; `T = 200` sits about 2.5 standard deviations above a mean digit sum of
161, so roughly one attempt in 900 lands and the signer resamples the randomness
until one does. It grinds a block of candidates per pass rather than stepping, and
returns the **lowest** landing attempt — upstream counts from zero and stops at
the first, so taking the last of a block would produce a valid signature that
upstream's own signer disagrees with byte for byte.

The same construction is already here under another name: SHRINCS drops WOTS+'s
checksum chains for the same constant-sum trick. The one difference decides the
cost — SHRINCS's constant is the *mode* of its distribution, so about one counter
in 65 lands, against leanSig's one in 900.

**Two encoders are host-only, and cannot be otherwise.** `encode_message`
decomposes a 256-bit root base-`p` and `encode_epoch` does the same to
`(slot << 8) | prefix`; a running remainder reaches `p ≈ 2^31`, so even a 16-bit
digit step needs 47 bits. The tweaks are the same story from the other end —
packing reaches `2^45` for a tree tweak and `2^56` for a chain one. All of it is
the first of this repo's [four non-negotiables](../../CLAUDE.md): an array lane is
32 bits, and a value wider than one stays a Python integer or becomes bytes.

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
leaf and attestation and proposal cannot share a leaf. The `TEST` preset shortens
the lifetime to `2^8` so key generation and signing are cheap enough to gate for
real.

## What leaks

**Verification has no data-dependent control flow.** Every Winternitz chain runs
its full length and masks, the Merkle climb runs the tree's fixed height, and the
codec's refusals are a flag rather than a branch — so the timing of a
verification is a function of the batch's shape and of nothing in the signatures
or keys. It also holds no secret, which is the reason it needs no such claim
([`../reference/security.md`](../reference/security.md)).

**The signer's rejection loop has a trip count that depends on the message**, and
it is worth saying exactly why that is a different situation from ML-DSA's
signing loop or Falcon's sampler. Those trip counts depend on *secret* data.
leanSig's acceptance test is a public function of public inputs — the verifier
recomputes it from the randomness the signature carries — so a host loop over it
leaks nothing a verifier does not already hold. The attempt it settles on is not
on the wire, but it is recoverable by anyone, by replaying the derivation from
counter zero.

**Signing is deterministic, and that is a security property rather than a
convenience.** The randomness is keyed by `(slot, message, attempt)`, so signing
one message at one slot twice yields the same signature. A fresh draw instead
would let a repeated slot produce two signatures over two distinct codewords,
which is precisely the multi-target opening a one-time key cannot survive.

**The format discloses nothing the caller does not have.** The slot is off the
wire, the signature is a fixed 2536 bytes at every slot and every codeword, and
the authentication path is the tree's full height regardless of which bottom tree
served it.

Signing carries no side-channel claim in this repo, and key generation none
either.

**The scheme's own assumption is stated on
[`../reference/security.md`](../reference/security.md#an-assumption-a-scheme-rests-on-is-the-schemes-and-it-gets-stated).**
leanSig's security proof is conditional on Poseidon meeting a multi-target
collision-resistance bound, and that assumption is younger and less scrutinized
than what this repo's other schemes rest on. It is a property of the scheme
rather than of this implementation, and it is written down so that
"post-quantum" is not read as "more conservative".

## What gates it

Seven vector modules under
[`testing/`](../../sig_frx/hash/leansig/testing/), each carrying the leanSpec
commit its values came from and the calls that produced them. Of those, the
Poseidon permutation and the SSZ containers come from the **published** fixtures
archive; the rest come from running the reference implementation at the same pin,
because the archive's signature-shaped families are leanMultisig aggregate proofs
rather than XMSS signatures.

The known-answer test is
[`leansig_kat_test`](../../sig_frx/hash/leansig/testing/leansig_kat_test.py):
verification at `PROD_CONFIG` against signatures upstream produced under two of
those published keys, six accepted and four refused. It is not driven
by [`kat.py`](../../sig_frx/testing/kat.py), for both reasons that harness names —
there is no published format to normalize, and a stateful scheme has no
seam-shaped `sign` for it to drive.
