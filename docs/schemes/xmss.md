# XMSS (RFC 8391)

Stateful hash-based signatures: a Merkle tree over WOTS+ one-time keys, where the
signing key carries the index of the next unused one. The public key is the root.
Everything is a call to the keyed-and-masked hash family, so the whole scheme is
symmetric-hash work — the same as SLH-DSA, over a substrate that is not.

Implementation:
[`sig_frx/hashbased/rfc8391_xmss.py`](../../sig_frx/hashbased/rfc8391_xmss.py),
over the RFC 8391 substrate in the same package — the address encoding, the hash
family, the parameter sets, WOTS+ and its L-tree compression.

## The state is the risk, and where it is written down is yours

Signing twice at one index does not degrade the signature; it reveals the WOTS+
secret key, and anyone holding both signatures can forge a third. Every other
property of this page is secondary to that one.

So **`sign` consumes a secret key and returns the advanced one**:

```python
signature, secret_key = scheme.sign(secret_key, message)
```

A caller who signs twice under one key has to name the spent value a second time.
That does not make reuse impossible — nothing in a library can — but it makes it
something the calling code says out loud, rather than something that happens
because a method did not move an index the caller could not see.

**Persistence is the caller's problem, deliberately.** This repo makes no
side-channel claim about signing at all
([`../reference/security.md`](../reference/security.md)), and a library that will
not claim anything about how signing executes should not own the durable storage
that makes a stateful signer safe. What this provides is a format: the secret key
crosses the API as the reference implementation's own byte layout,
`idx ‖ SK.seed ‖ SK.prf ‖ root ‖ SEED`. Writing the advanced key down *before*
releasing the signature it came from is the caller's discipline, and a crash
between those two steps is the caller's failure mode to design around.

A key with all `2^h` one-time keys spent refuses to sign rather than wrapping.

## What the standard fixes, and what this implementation chooses

The standard fixes everything observable: the parameter sets and their OIDs
(§5.3), the address encoding (§2.5), the hash constructions (§5.1), the L-tree
(§4.1.5) and the signature encoding (§4.1.8). RFC 8391 publishes no test vectors
and the validation program has no XMSS, so what those are gated against is the
reference implementation §7 points at, digested as its own generator prints — the
decision, its alternatives and the provenance are on
[fractalyze/sig-frx#16](https://github.com/fractalyze/sig-frx/issues/16).

Two details of the standard are worth stating here because they are what an
implementation gets wrong:

- **The padding length is not `n`.** §5.1's domain separators are
  `toByte(c, padlen)`, and `padlen` is 32 bytes at `n = 32` but **4** at `n = 24`.
  An implementation that padded to `n` passes every 256-bit vector.
- **§2.5's prose and its reference implementation disagree about `setType()`**,
  and the vectors follow the implementation: setting the type sets one word and
  normalizes nothing. So the address encoder here does not zero what a type does
  not use, which is the opposite of the FIPS 205 encoder beside it.

What this implementation chooses:

- **`sign` is not on the `Signature` seam and `verify` is.** The seam returns one
  value from `sign`; a stateful signer has to return two. The alternative — a
  seam-shaped `sign` that leaves the index alone — is the footgun the discipline
  exists to remove, so this scheme carries no seam conformance pin. That is the
  expected shape for a stateful scheme rather than an omission, and
  [`signature.py`](../../sig_frx/signature.py) says so.
- **The tree walk is shared with FIPS 205, adjusted at one seam.** FIPS 205
  addresses a Merkle node by the height of the parent being computed; RFC 8391
  addresses it by the height of the children being consumed. They differ by
  exactly one, and that decrement lives in this module's `NodeAddresses` builders
  — so `tree.root`, `tree.auth_path` and `tree.root_from_path` are the same code
  under both standards.
- **The L-tree gets its own loop rather than a flag on the Merkle one.** §4.1.5
  lifts an unpaired node to the next level unhashed, where `tree.reduce_levels`
  refuses an odd count on purpose. A pull-up and a refusal behind one name would
  be two behaviours with one signature.
- **A parameter set is a table row keyed by OID.** All 21 registered rows are
  present; the SHAKE sets and the `n = 64` SHA-2 sets are not constructible until
  hash-frx has Keccak and SHA-512, and building one refuses rather than silently
  using SHA-256.
- **RFC 8391 has no application context**, so `sign` and `verify` take an empty
  one and raise on anything else rather than accepting and ignoring it.

## Where the batch axis is

`verify` is batch-first, as the seam requires: `uint8[B, 2n]`, `uint8[B, L]` and
`uint8[B, sig]` in, `bool[B]` out. Every argument carries the batch, including the
public key and the index each signature claims — so entry `k` walks its own leaf
and its own authentication path, and a batch shares nothing.

| Stage | Calls | Width |
| ----- | ----- | ----- |
| `H_msg` over the batch | 1 | `B` |
| WOTS+ chains from the signature | `w − 1` | `B · len` |
| L-tree compression | `⌈log2 len⌉` | `B · len/2`, halving |
| Merkle path to the root | `h` | `B` |

`keygen` and `sign` take one key and one message, and both build the whole tree:
`2^h` WOTS+ key pairs, walked as one batch of `2^h · len` chains rather than
`2^h` separate walks. At the gated sets that is 1024 leaves — the reference's own
generator samples only `h = 10`, which is why the vectors are runnable at all.
There is no state-management scheme here (BDS traversal, cached authentication
paths); signing recomputes, because signing exists to reproduce vectors.

## What leaks

Read [`../reference/security.md`](../reference/security.md) first: signing carries
no side-channel claim in this repo, and verification needs none.

Like SLH-DSA, XMSS has no rejection sampling and no secret-dependent trip count: a
WOTS+ chain runs its full length whatever the message digit is, and every level of
every tree is hashed in full. **The index is the one value that steers host-side
work — which leaf is signed, which siblings are taken — and it is not secret: it
travels in the clear in every signature.**

What is still not claimed: constant-time execution of the underlying hash, memory
hygiene for `SK.seed` and `SK.prf`, and resistance to fault or physical attack.
Verification touches no secret at all.

The failure mode this scheme has and SLH-DSA does not is index reuse, and it is not
a side channel — it is a total break available to anyone who collects two
signatures made at one index. See the top of this page.
