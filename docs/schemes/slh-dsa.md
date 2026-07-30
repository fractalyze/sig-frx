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
or does not.

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

Two things qualify what "batched" delivers. The hashes run eagerly, so a batched
call is one dispatch per level rather than one traced graph. And the tree and leaf
indices come out of the digest as host integers, because an address is built on the
host from a concrete index — which is where a traced index would have to be handled
differently.

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
