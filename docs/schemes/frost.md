# FROST (RFC 9591)

Two-round threshold Schnorr: `t` of `n` participants each hold a share of a
group secret nobody holds whole, and any `t` of them produce one ordinary
Schnorr signature under the group public key. The RFC defines the protocol once
and instantiates ciphersuites over it, and the code has the same shape: the
round skeleton in
[`sig_frx/threshold/frost.py`](../../sig_frx/threshold/frost.py) never names a
curve, and a ciphersuite —
[`ed25519_sha512.py`](../../sig_frx/threshold/ed25519_sha512.py),
[`secp256k1_sha256.py`](../../sig_frx/threshold/secp256k1_sha256.py) — is
constants and hash instantiations over a substrate the classical schemes
already audit.

## Key establishment is a trusted dealer, on purpose

RFC 9591 specifies signing, not key generation: its vectors start from
Appendix C's trusted dealer, and distributed key generation is explicitly out
of the RFC's scope. This implementation makes the same cut, and the decision
is recorded here because it is a deployment trust model, not an API detail:

- **What exists.** `secret_share_split`, `vss_commit` and `vss_verify` —
  Appendix C's Shamir sharing with Feldman commitments, enough for a
  participant to check the share it was dealt against the published
  commitment vector.
- **What that means.** The dealer computes the group secret and every share.
  A deployment of this module trusts that machine with the whole key at
  dealing time; the threshold property protects signing, not generation.
- **What is deferred.** A distributed keygen is a different protocol with its
  own security analysis and communication model, and no consumer of this repo
  has asked for one. It is deferred until one materializes, not omitted by
  oversight.

## What the standard fixes, and what this implementation chooses

The RFC fixes everything observable: the round functions and their MUST-checks
(§5), the binding-factor and challenge derivations (§4), each suite's
encodings and hash instantiations (§6), and the verification of the result
(Appendix B). All of it is gated on the RFC's published vectors stage by
stage — dealer shares, nonces, binding factors, signature shares, the final
signature — so a wrong stage fails as itself.

What this implementation chooses:

- **The rounds run on the host, as pure functions.** Values in, values out;
  nonces are the caller's state between rounds, randomness is a parameter,
  and moving messages between parties is the consumer's problem — the RFC's
  own framing, and what makes the vectors reproducible.
- **A two-round stateful signing has no seam-shaped `sign`**, so the round
  functions are their own named surface and the `Signature` seam is left
  whole, the same shape the stateful hash-based schemes settled.
- **Verification is per-suite, through each suite's `verify`.**
  FROST(secp256k1, SHA-256)'s aggregate is RFC 9591's own Schnorr encoding —
  not a BIP-340 signature, so no chain verifier exists for it — and its
  `verify` is Appendix B's prime-order check. FROST(Ed25519, SHA-512)'s
  aggregate is a plain RFC 8032 signature, and its `verify` delegates to the
  scheme layer's batched Ed25519 verifier rather than growing a second
  Edwards path.

## Where the batch axis is

The rounds have none, deliberately: a round is per-participant work over a
handful of scalars, `B = 1` on the host, and not the hot path.

The hot path is verifying the aggregate, and each suite's `verify` is
batch-first as the repo requires: `uint8[B, key]`, `uint8[B, L]` and
`uint8[B, sig]` in, `bool[B]` out, the point work in batched substrate
kernels. Wire rows validate before anything meets a field op — the
scalar-field dtype aborts on an out-of-range operand
(fractalyze/zk_dtypes#179) — so a malformed row is a `False` verdict, never
an exception out of the batch.

## What leaks

Read [`../reference/security.md`](../reference/security.md) first: signing
carries no side-channel claim in this repo, and verification needs none.

The rounds are host Python over secret shares and nonces — big-integer
arithmetic whose timing is not data-independent, permitted because signing
carries no claim. There is no rejection sampling and no secret-dependent trip
count. Dealing (Appendix C) handles the group secret itself, on the same
no-claim host path. Verification consumes only public data.
