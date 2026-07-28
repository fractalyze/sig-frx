# sig-frx

FRX-native digital signatures — post-quantum first, classical alongside.

`sig-frx` builds on [`hash-frx`](https://github.com/fractalyze/hash-frx) for every
symmetric primitive it needs, and on **FRX** — Fractalyze's fork of
[JAX](https://github.com/jax-ml/jax) — for tracing and codegen, lowered through
**Fractalyze XLA**.

## Design philosophy

- **One `Signature` seam.** `keygen` / `sign` / `verify` over opaque key and
  signature types. A consumer picks a scheme by construction, not by branching on
  a name.
- **Batch-parallel by construction.** Verification is the hot path and it is
  embarrassingly parallel — a batch of `B` signatures verifies in one call, so the
  work maps onto a GPU's width rather than a Python loop.
- **Standards-exact.** Every scheme reproduces its specification byte for byte
  (SLH-DSA = FIPS 205, ML-DSA = FIPS 204, Falcon = the FN-DSA submission, XMSS =
  RFC 8391), gated on the published known-answer tests.
- **Chain-agnostic core, chain variants on top.** ECDSA and EdDSA cores name no
  blockchain; Ethereum and Bitcoin conventions (message hashing, recovery,
  low-`S`, encoding) ride as thin variants over the shared core.

## Status

Bootstrapping. Work is tracked on the
[issues](https://github.com/fractalyze/sig-frx/issues).

## License

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).
