# MuSig2 (BIP-327)

`n`-of-`n` Schnorr: every cosigner's key folds into one x-only key, and what
they jointly produce under it is an ordinary BIP-340 signature. Not a threshold
scheme — every holder must sign — which is what separates it from
[FROST](frost.md) despite the shared two-round shape.

## It is the one multi-party scheme here whose output a chain accepts

[`frost.md`](frost.md) states the gap this closes: FROST(secp256k1, SHA-256)'s
aggregate is RFC 9591's own Schnorr encoding, "not a BIP-340 signature, so no
chain verifier exists for it". MuSig2's aggregate *is* one, under the aggregate
key, indistinguishable on chain from a single signer's.

So this module ships no verifier. `partial_sig_agg` returns 64 bytes and
[`bip340.py`](../../sig_frx/classical/schnorr/bip340.py)'s `verify` is already
the right one — batch-parallel per the repo's non-negotiable, with the accept
set BIP-340 already defines. A MuSig2 verifier would be a second accept set to
keep in agreement with the first, and there is no question it could answer that
the existing one cannot. The test that matters asserts exactly this: the
published aggregates are run back through `Bip340.verify` rather than only
compared to their vector bytes.

## Where it lives, and why not under `threshold/`

In [`classical/schnorr/`](../../sig_frx/classical/schnorr/musig2.py), beside
BIP-340, because that is what it shares: the tagged-hash construction, the
even-y convention, and the `secp` substrate. It shares nothing with
[`frost.py`](../../sig_frx/threshold/frost.py) but a shape — FROST's
`Ciphersuite` is five hashes from RFC 9591 §6 that mean nothing here. And
`n`-of-`n` is not a threshold, so filing it under `threshold/` would make that
package's name false.

No shared multi-party seam is invented for it. Whether one should exist, and
whether it is FROST's group half, is a question the ECDSA threshold track's
design note owes for the whole repo; a second seam written ahead of that answer
would be one to reconcile later.

## What the standard fixes, and what this implementation chooses

BIP-327 fixes everything observable — the key-aggregation coefficients and the
second-key exemption, both tweak forms, the nonce derivation and its length
prefixes, the session's binding factor and challenge, and every encoding. All
of it is gated stage by stage on the eight published vector files, so a wrong
binding factor and a wrong partial signature fail as different things.

What this implementation chooses:

- **The stages are their own named surface, not the `Signature` seam.** A
  two-round interactive protocol has no seam-shaped `sign`, which is the shape
  the threshold and stateful hash-based schemes already settled.
- **A `Session` carries what the cosigners must already agree on.** Signing and
  verifying a partial signature derive from identical values, so they take one
  type rather than the same six arguments twice. Two signers who differ on the
  message or the key list produce partial signatures that cannot aggregate, and
  nothing before aggregation would otherwise say so.
- **`key_sort` is the caller's to apply.** Aggregation binds the order it is
  given, so a group that sorts and one that keeps its own order derive
  different keys and neither is wrong.
- **The optional membership check is taken.** A partial signature under a key
  the session never aggregated is well formed and simply fails to combine, so
  declining to check costs the coordinator the name of whoever to ask.

## A `SecNonce` is for exactly one session

Signing twice from one against different messages or cosigner sets reveals the
signer's secret key outright. That is not a liveness abort the way a FROST
round failure is — the key is gone — and it is the attack the two-nonce
construction exists to survive.

`sign` consumes the nonce and returns nothing that can be spent again, which is
the shape [`xmss.py`](../../sig_frx/hash/xmss/xmss.py) uses for its leaf
counter. **Only the shape.** Xmss hands back an advanced index, so a spent key
is visible and its `sign` can refuse one; a nonce has no such tell, and nothing
here can detect reuse. What `sign` can check it does: that the nonce was drawn
for the key doing the signing, and that its scalars are in range, which the
specification notes is where reuse tends to surface.

Enforcement is not moved into the type. A mutable spent flag would put a side
effect inside a traced computation, which `xmss.py` argues against in writing as
"how one gets reused without anyone writing the reuse down".

`deterministic_sign` removes the window rather than the rule: its nonce is a
function of the secret key, the other cosigners' nonces and the message, so one
session signed twice reproduces the same nonce harmlessly and two sessions
cannot collide. That is why its randomness is optional and `nonce_gen`'s is not.

## A bad contribution names its sender, and sometimes there is nobody to name

`InvalidContributionError` carries the cosigner's index and what they got
wrong. A coordinator that only learns the ceremony failed has to restart with
everybody; one that learns which index sent an unusable key can drop that
participant and continue.

Its `signer` is `None` for a fault in an aggregate nonce — the session's, or
the one `deterministic_sign` is handed. Those are the coordinator's own
product: no cosigner sent them, and excluding one would not fix them. The
distinction is the specification's, and it is why the aggregate's lift does not
share the by-position blame the cosigner paths use.

The same split runs through verification. A wrong partial signature is `False`
and an unusable one raises, because only the second identifies somebody to
exclude — the first is a cosigner who signed something else, which no one can
be removed for. Verifying partials before aggregating is what turns a failed
aggregate, one bad signature in `n` with nothing to say which, into a name.

## Where the batch axis is

The stages have none, and for the reason FROST's rounds have none: they are
per-participant work over a handful of scalars, `B = 1` on the host, run once
per session rather than per verification.

Key aggregation is the exception worth naming. It is internally wide — a
`u`-term multi-scalar multiplication — so its coefficients and points cross
into `secp` as one batch rather than a Python fold. At the cosigner counts a
large ceremony reaches, that is the only workload in this repo that wants a
genuinely large MSM rather than many tiny ones. Nothing has been profiled at
those sizes, and no claim is made about what it would buy.

The hot path is unchanged: verification is BIP-340's and already batch-first.

## What leaks

Read [`../reference/security.md`](../reference/security.md) first: signing
carries no side-channel claim in this repo, and verification needs none.

The stages are host Python over secret keys, nonces and shares — big-integer
arithmetic whose timing is not data-independent, permitted because signing
carries no claim. There is no rejection sampling and no secret-dependent trip
count. Verification of the aggregate consumes only public data and runs through
BIP-340's verifier, which holds no secret.
