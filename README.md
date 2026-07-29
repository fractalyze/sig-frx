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

## Development

The build is Bazel — bzlmod, with a hermetic Python 3.11 toolchain:

```sh
bazel test //...
```

`hash-frx` rides the module graph, pinned by commit in
[`MODULE.bazel`](MODULE.bazel). To build against a local checkout instead:

```sh
echo 'common --override_module=hash_frx=/abs/path/to/hash-frx' >> .bazelrc.user
```

`pre-commit` covers formatting, typing, and the commit message. The message hook
runs at the `commit-msg` stage, which `pre-commit install` alone does not wire up
— install both:

```sh
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

## License

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).
