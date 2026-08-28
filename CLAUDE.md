# Project context for Claude Code

Everything load-bearing lives in repo docs. Treat those as the source of truth;
this file is the map plus the rules every change must respect.

- **Project overview, build, and dev setup:** [`README.md`](README.md)
- **Task-indexed docs hub:** [`docs/README.md`](docs/README.md)
- **Security posture — what this repo claims and what it does not:**
  [`docs/reference/security.md`](docs/reference/security.md)
- **Coding conventions — what implementing a scheme here requires, and only
  that:** [`docs/reference/conventions.md`](docs/reference/conventions.md)
- **What gates a scheme — the reference transcription and the KAT rules:**
  [`docs/reference/testing.md`](docs/reference/testing.md)
- **Measurement — CI budgets, and what a recorded number may claim:**
  [`docs/reference/measurement.md`](docs/reference/measurement.md)
- **Per-scheme design notes:** [`docs/schemes/README.md`](docs/schemes/README.md)
- **The one seam every scheme implements:**
  [`sig_frx/signature.py`](sig_frx/signature.py)
- **Detailed design & open decisions:** tracked on GitHub — epic issue
  [fractalyze/sig-frx#1](https://github.com/fractalyze/sig-frx/issues/1).
- **Running the suite the way the merge gate does:**
  `bazel --bazelrc=.bazelrc.ci test //...`. `.bazelrc.ci` is loaded explicitly,
  never auto-imported, so the bare `bazel test //...` in the README additionally
  runs the `slow_kat` sweeps — which are the scheduled gate, not the merge one,
  and which starve a shared machine into TIMEOUTs that are not failures. That
  invocation is the CPU leg; the GPU leg needs two `--test_env` flags or it
  fails wholesale, and
  [`measurement.md`](docs/reference/measurement.md#running-the-gpu-leg-locally)
  carries it along with the control that proves a green run used the device.
- **Merge commits must be titled `Merge branch 'X' into Y`.**
  fractal-commit-lint exempts only that form (and `Merge pull request #N`);
  git's default `Merge remote-tracking branch 'origin/X'` wording fails the
  commit-msg hook and leaves the merge stopped before committing.
- **CI does not run on a stacked pull request.** `ci.yml` triggers
  `build-and-test` on `pull_request: branches: ["main"]`, so a PR based on
  another PR's branch shows only Commit Lint and reads as passing. Either land
  the bottom of the stack first, or verify both legs locally and say so on the
  PR — a green check list that is one entry long is the tell.
- **A red CPU leg is usually the runner, and the re-run has a queue in front of
  it.** Bazel exit 36 with `No space left on device` (or `Socket closed` at 37,
  or a missing `externals/node24/bin/node`) means the runner, not the change —
  the tell is a 15-second job reporting `Executed 0 out of N tests`. Re-run it
  rather than debugging it. But `gh run rerun --failed` refuses while *any* job
  in the run is still going, so a dead CPU leg cannot be re-run until the GPU leg
  clears its queue, which has taken 40 minutes.
- **`gh pr edit` does not work against this repo.** It queries
  `repository.pullRequest.projectCards`, which GitHub has deprecated with the
  Projects-classic sunset, and fails without editing anything — quietly enough
  to look like it worked. Edit a PR body with
  `gh api -X PATCH repos/fractalyze/sig-frx/pulls/<N> -F body=@<file>` instead,
  and re-read the body to confirm.
- **A device XOF squeeze is sized into the program, so a long one does not
  work.** hash-frx's `Shake256(size).digest(...)` compiles in time and memory
  super-linear in `size`: 4.6 s at 1 KB, 28 s at 4 KB, nothing inside 400 s at
  16 KB, and at 64 KB it exhausts the box. A concrete caller wanting more than a
  few KB uses `hashlib` — the escape hatch [`hashes.py`](sig_frx/hashes.py)
  names — as Falcon's key generation does for its 64 KB draw.

## Four non-negotiables

- **An integer array lane is 32 bits.** frx runs without x64, so `uint64` becomes
  `uint32` and a wider value truncates *without raising* — silently addressing the
  wrong subtree. Anything that can exceed 2^32 is carried as bytes, which is what
  the standards call it anyway (`toInt`/`toByte` pairs): see
  [`bytestring.py`](sig_frx/hash/bytestring.py). Host code hides this,
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
  The rule has exactly one suspension, and it inverts rather than relaxes: inside
  the `double_precision` scope
  [Falcon's rational transform](docs/reference/conventions.md#falcons-second-transform-runs-in-double-precision)
  opens, a lane *widens* and `uint32.sum()` returns `uint64`. It applies to
  everything the scope calls, not only the floats it was opened for, so integers
  stay outside it or pin their accumulator.
- **Standards-exact, or it is not done.** Every scheme reproduces its
  specification byte for byte, gated on the published known-answer tests. A
  scheme that verifies its own signatures has demonstrated nothing — a
  self-consistent wrong implementation round-trips forever — and the negative
  vectors are half the gate, because a verifier that returns `True`
  unconditionally passes every positive one. Not every standard publishes
  vectors; that does not lower the bar, it changes what the authority is
  ([`testing.md`](docs/reference/testing.md#a-standard-that-publishes-no-vectors-still-gets-gated)).
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
