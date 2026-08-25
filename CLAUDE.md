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
- **Running the suite the way the merge gate does:**
  `bazel --bazelrc=.bazelrc.ci test //...`. `.bazelrc.ci` is loaded explicitly,
  never auto-imported, so the bare `bazel test //...` in the README additionally
  runs the `slow_kat` sweeps — which are the scheduled gate, not the merge one,
  and which starve a shared machine into TIMEOUTs that are not failures.
  **A green bazel run is not the whole gate.** `mypy` and `black` run only at
  commit time through pre-commit — `mypy` with `pass_filenames: false`, so it
  type-checks the tree rather than the diff — and CI runs pre-commit as its own
  job. A change can pass every test and still fail on a type error or a
  reformat, so run `pre-commit run --all-files` before claiming a change is
  clean rather than discovering it in the commit hook.
- **`bazel test` is one leg, not both.** `.bazelrc` pins
  `test --test_env=FRX_PLATFORMS=cpu`, so every command above runs the **CPU leg
  only**. The GPU leg is a second command, and `--local_test_jobs=1` is required
  rather than tuning — concurrent jobs each reserve a large share of free VRAM
  and the losers fail during device init, naming the wrong cause:

  ```sh
  bazel test --test_env=FRX_PLATFORMS=cuda --local_test_jobs=1 //...
  ```

  The two are different programs where routing is involved: a marker routes
  `DEDICATED` on one leg and `GENERIC` on the other, so a change to how a
  primitive is routed, fused or emitted has been validated for half the wire
  surface until both legs are green.
- **Merge commits must be titled `Merge branch 'X' into Y`.**
  fractal-commit-lint exempts only that form (and `Merge pull request #N`);
  git's default `Merge remote-tracking branch 'origin/X'` wording fails the
  commit-msg hook and leaves the merge stopped before committing.

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
  Bytes are the answer for a value with no arithmetic. A **field** element has a
  second one: `zk_dtypes.prime_field(q)` mints a field from any modulus, curated
  or not, and reduces internally — so a scheme's modular arithmetic is never
  hand-written in limbs. Read a residue back with `astype`, never a bitcast: the
  storage is a Montgomery representative.
  The same rule has a quieter form in any function that runs in **both**
  namespaces: numpy promotes a reduction's accumulator and frx does not, so a
  bare `.sum()` returns `uint64` on the host and `uint32` traced from one source
  line. The values agree, so a round trip, a reference comparison and a
  known-answer test all pass — pin it (`.sum(axis=-1, dtype=np.uint32)`), and
  assert the *dtype* in the host-vs-traced case, not only the values.
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
