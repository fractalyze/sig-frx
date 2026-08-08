# Project context for Claude Code

Everything load-bearing lives in repo docs. Treat those as the source of truth;
this file is the map plus the rules every change must respect.

- **Project overview, build, and dev setup:** [`README.md`](README.md)
- **Task-indexed docs hub:** [`docs/README.md`](docs/README.md)
- **Security posture — what this repo claims and what it does not:**
  [`docs/reference/security.md`](docs/reference/security.md)
- **Coding conventions — what implementing a scheme here requires, and only
  that:** [`docs/reference/conventions.md`](docs/reference/conventions.md)
- **Per-scheme design notes:** [`docs/schemes/README.md`](docs/schemes/README.md)
- **The one seam every scheme implements:**
  [`sig_frx/signature.py`](sig_frx/signature.py)
- **Detailed design & open decisions:** tracked on GitHub — epic issue
  [fractalyze/sig-frx#1](https://github.com/fractalyze/sig-frx/issues/1).

## Four non-negotiables

- **An integer array lane is 32 bits.** frx runs without x64, so `uint64` becomes
  `uint32` and a wider value truncates *without raising* — silently addressing the
  wrong subtree. Anything that can exceed 2^32 is carried as bytes, which is what
  the standards call it anyway (`toInt`/`toByte` pairs): see
  [`bytestring.py`](sig_frx/hashbased/bytestring.py). Host code hides this,
  because Python integers have no width — so a value that only ever lived on the
  host is exactly where this bites when it is first traced. The operational rule
  that keeps a value out of the wrong lane is
  [a value is used in the namespace it arrives in](docs/reference/conventions.md#a-value-is-used-in-the-namespace-it-arrives-in).
- **Standards-exact, or it is not done.** Every scheme reproduces its
  specification byte for byte, gated on the published known-answer tests. A
  scheme that verifies its own signatures has demonstrated nothing — a
  self-consistent wrong implementation round-trips forever — and the negative
  vectors are half the gate, because a verifier that returns `True`
  unconditionally passes every positive one. Not every standard publishes
  vectors; that does not lower the bar, it changes what the authority is
  ([`conventions.md`](docs/reference/conventions.md#a-standard-that-publishes-no-vectors-still-gets-gated)).
- **Batch-parallel verification.** Verification is the hot path and it is
  embarrassingly parallel, so a batch of `B` signatures verifies in one call.
  The seam has no scalar `verify` on purpose: a single verification is `B = 1`.
  A Python loop over the batch axis is a bug, not a slow implementation.
- **Chain-agnostic cores.** No scheme core names a blockchain. Ethereum's and
  Bitcoin's conventions — message hashing, recovery, low-`S`, encoding — ride as
  thin variants over the shared curve-level core.

Signing carries no side-channel claim in this repo, and verification needs none.
That is a decision with consequences for every scheme, so read
[`docs/reference/security.md`](docs/reference/security.md) before implementing
one.
