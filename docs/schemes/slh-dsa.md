# SLH-DSA (FIPS 205)

Stateless hash-based signatures: a hypertree of XMSS trees over WOTS+ one-time
keys, certifying a FORS few-time signature over the message digest. Everything is
a call to the tweakable hash family, so the whole scheme is symmetric-hash work.

Implementation: [`sig_frx/hashbased/slh_dsa.py`](../../sig_frx/hashbased/slh_dsa.py),
over the shared components in the same package — the address structure, the
tweakable hash family, WOTS+, the Merkle hash tree, FORS and the hypertree.

## What the standard fixes, and what this implementation chooses

The standard fixes everything observable. The parameter sets are Table 2's, the
encodings are Figures 15 to 17, the domain separators and the context framing are
§10.2, and the digest split is Algorithm 19 lines 6 to 10. None of it is
negotiable: a signature is a byte string that either matches what NIST published
or does not — and it does, for every operation the standard defines and every
parameter set the two families here can build, against ACVP's `keyGen`, `sigGen`
and `sigVer` sets. That is all six SHAKE sets and SHA-2's category 1 pair. The
exhaustive run over every one of them is tagged `slow_kat`; the merge gate gets
the same operations at one `f` set.

Three interfaces, and the seam names one. §10's pure external operation is the
seam; HashSLH-DSA and §9's internal interface prepend a different message, so they
live under `hash_sign` / `hash_verify` and `sign_internal` / `verify_internal`. The
validation program publishes vectors against all three, so all three are gated.

What this implementation chooses:

- **A parameter set is a table row, not a subclass.** `SHA2_PARAMETER_SETS` holds
  the six SHA-2 rows and everything else derives from `n`, `h`, `d`, `a` and `k` —
  including `h'`, `m`, and the key and signature sizes, which Table 2 lists and its
  own footnote says are computed. Table 2 gives each row two names, one per hash
  family, so a SHAKE instantiation is these rows under a different tweakable hash.
- **Security category 1 is what `sha2` builds.** §11.2.1's family reaches every
  function with SHA-256 alone. Categories 3 and 5 keep SHA-256 for `PRF` and `F`
  but hash `H`, `T_l` and `PRF_msg` with SHA-512 (§11.2.2), which makes them a
  family over two hashes and needs a SHA-512 `ByteHash` rather than a constant
  change here.
- **`shake` builds all six sets where `sha2` builds two.** §11.1 reaches every
  function with SHAKE256 at every security category, because an extendable output
  already produces whatever length each one wants — there is no MGF1 to reach `m`
  bytes, no HMAC, no compression-block padding, and no second hash to change to.
  An XOF at two lengths is two hashes rather than one asked twice, so the family
  holds one instance per length its parameter set names.
- **The address encoding belongs to the family, and every component asks.** The
  address has two encodings — §11.2's 22-byte `ADRS^c` and §4.2's full 32 — and
  which one applies is a property of the hash family, published as
  `TweakableHash.compressed_address`. Components read it rather than deciding,
  because a component that decides is right for exactly one family. The builders
  that hold no family take it as a required keyword, so a caller that forgets is a
  type error rather than a wrong public key. `tree.py`'s Merkle walk never sees
  it: the walk is shared with RFC 8391, whose addresses are neither encoding, so
  it takes a builder and the encoding is that builder's parameter.
- **Loops are reshaped for the compiler, everywhere below this module.** A WOTS+
  chain runs its full length and masks instead of stopping at a digit; a Merkle
  tree iterates where the standard recurses; FORS reduces all `k` trees as one
  contiguous forest. Each is pinned against the standard's own form in that
  component's tests.
- **The hedged and deterministic variants are one instance attribute.** §9.2 makes
  the choice a property of the key rather than of the scheme, so `deterministic`
  selects Algorithm 19 line 2's substitution. A hedged instance never draws its own
  randomness: the seam takes it, because an implicit draw is how a scheme stops
  being reproducible against its vectors.
- **Pre-hash signing is not on the seam.** `hash_sign` and `hash_verify` implement
  Algorithms 23 and 25 under their own names, because they sign a different message
  — domain separator one, and the pre-hash function's OID before the digest. Which
  is the whole point of the separator, so the two must not share an entry point.
  SHA-256 is the pre-hash function available; ACVP exercises twelve, and the OID is
  part of what gets signed, so the others need their hash rather than a stand-in.

## Where the batch axis is

`verify` is batch-first, as the seam requires: `uint8[B, 2n]`, `uint8[B, L]` and
`uint8[B, sig]` in, `bool[B]` out. Every argument carries the batch, including the
public key — so `PK.seed` and `PK.root` vary per entry, and the digest that picks
the FORS key varies with them. A batch shares nothing but the context, which is one
value per call because a verifier serves one protocol domain at a time.

Under the seam the batch widens unevenly and the work stays batched at each width:

| Stage | Calls | Width |
| ----- | ----- | ----- |
| `H_msg` over the batch | 1 | `B` |
| FORS leaf and path walk | `a + 1` | `B · k` |
| `T_k` over each entry's roots | 1 | `B` |
| Hypertree, per layer | `d · (w − 1 + h')` | `B · len`, then `B` |

`keygen` and `sign` take one key and one message. They are not the hot path —
signing here exists to reproduce vectors — so a caller who needs many uses
`frx.vmap` rather than a second entry point.

The whole of `verify` traces, so `frx.jit(verify)` is one program rather than one
dispatch per level. Getting there meant taking every value the path carries off the
host. The hypertree index rides as bytes, because it reaches 64 bits and an integer
array lane is 32 — see
[`sig_frx/hashbased/bytestring.py`](../../sig_frx/hashbased/bytestring.py). An
address is packed wherever its fields already live, so the same expression builds
one from host integers and from traced columns. And the digest is left where
`H_msg` produced it, since the tree and leaf indices are slices of it.

`sign` stays on the host, and the asymmetry is deliberate rather than unfinished: a
signer holds one signature, where a Python integer carries the tree index at any
width and never truncates. The two paths meet at the digest split, which is the one
place those bytes become a number.

## What leaks

Read [`../reference/security.md`](../reference/security.md) first: signing carries
no side-channel claim in this repo, and verification needs none.

SLH-DSA is the easy case for that posture. It has no rejection sampling and no
secret-dependent trip count anywhere — a WOTS+ chain runs its full length whatever
the message digit is, a Merkle level hashes every node, and the FORS forest is
reduced in full. The one place control flow depends on a value is the digest split,
which chooses the tree and leaf indices, and those are not secret: a verifier
recomputes them from the randomizer the signature carries, the public key and the
message, so the host work driven by them is a function of published values.

What is still not claimed: constant-time execution of the underlying hash, memory
hygiene for `SK.seed` and `SK.prf`, and resistance to fault or physical attack.
Verification touches no secret at all.
