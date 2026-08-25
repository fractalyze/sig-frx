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
  (SLH-DSA = FIPS 205, ML-DSA = FIPS 204, XMSS / XMSS-MT = RFC 8391, ECDSA =
  SEC 1 + RFC 6979, Ed25519 = RFC 8032, BIP-340, FROST = RFC 9591), gated on the
  published known-answer tests.
- **Chain-agnostic core, chain variants on top.** ECDSA and EdDSA cores name no
  blockchain; Ethereum and Bitcoin conventions (message hashing, recovery,
  low-`S`, encoding) ride as thin variants over the shared core.

## Status

Constructible through `Signature` today:

| family | schemes |
|---|---|
| Hash-based | SLH-DSA (FIPS 205), XMSS and XMSS-MT (RFC 8391) |
| Lattice | ML-DSA (FIPS 204) |
| Classical | ECDSA over secp256k1 and P-256, with the Ethereum and Bitcoin variants; BIP-340 Schnorr; Ed25519 |
| Threshold | FROST (RFC 9591), Ed25519 and secp256k1 ciphersuites |

**Falcon (FN-DSA) is planned, not shipped** — `sig_frx/lattice/` carries `mldsa`
and nothing else. It is tracked by
[#24](https://github.com/fractalyze/sig-frx/issues/24)–[#28](https://github.com/fractalyze/sig-frx/issues/28).
A scheme is listed above only when a consumer can construct it, since that is
what picking a scheme by construction means.

Remaining work is tracked on the
[issues](https://github.com/fractalyze/sig-frx/issues).

## Development

Conventions, the security posture, and per-scheme design notes live in
[`docs/`](docs/README.md).

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
