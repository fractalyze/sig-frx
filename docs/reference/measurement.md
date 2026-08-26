# Measurement

> Code, symbols, and file paths are English.

A number about this repo is one of two things, and conflating them is the
failure this page exists to prevent. A **budget** is the deadline a target has
to finish inside on CI, sized from the worst run of the slowest leg it runs on.
A **local measurement** compares two implementations, and sizes nothing.

The rules for writing the code are [`conventions.md`](conventions.md); what a
scheme is gated on is [`testing.md`](testing.md).

## A budget is sized from a target's worst run, not its typical one

A target that times out reports a red build for a suite that works, and the edit
that fixes it is always the same one line. What earns it a rule rather than a
correction each time is that the measurement people reach for is the wrong one
twice over: it is taken on a workstation, and it is taken once.

**A target whose worst observed run reaches half its budget moves up a bucket.**
The buckets are 60 / 300 / 900 / 3600 s, so moving up is coarse and cheap — a
deadline is only spent when a test hangs.

Half, rather than something tighter, is what the executor's spread makes
necessary. Across runs of *unchanged* commits, a single target's duration on the
CPU leg varies by a median of 2.2x — 1.7x to 2.6x over the targets long enough
for the variation to mean anything — while the same measurement on the GPU leg
varies by 1.07x. The CPU leg executes on a shared remote pool, so any one number
it produces may be the fast one and the next run may be twice it; half the
budget is the smallest round headroom that survives that. The GPU leg's
steadiness is a property of a quiet box rather than of the leg — one GPU runner
serves the org — so it gets the same threshold rather than a tighter one earned
by present load.

This is the quantitative half of [a target excluded from a leg has never had its
budget validated there](testing.md#the-per-pr-gates-cost-is-distinct-shapes-not-vectors).
Together: the budget covers the worst run of the slowest leg the target runs on.

### A local measurement does not decide it

The CI executor is slower than a workstation by a factor that is per-target
rather than a constant to divide out — measured at 1.6x for one target and 3.4x
for another. A target that reads as using a fifth of its budget locally can be
using two thirds where it counts. Local numbers compare two implementations;
they do not size a budget.

Bazel will argue the other way, and it argues from a local run. A target sized
for the executor trips `Test execution time … outside of range for MODERATE
tests. Consider setting timeout="short" or size="small"` on a workstation, and
`bazel test` prints the summary line that points at it. Every budget on this
page is deliberately one bucket above what that warning asks for. Taking its
advice restores the flake, so it is the one bazel diagnostic this repo overrides
on purpose rather than silences — it is right about the local number and the
local number is not the one that decides.

### `size` moves two things and `timeout` moves one

`size` picks a default deadline *and* the resource estimate bazel schedules
against when it executes a test locally. Up to `large` that estimate barely
moves and `size` is the ergonomic edit. `enormous` is where it jumps — so a
target that needs the 3600 s deadline and not the weight keeps its `size` and
declares `timeout = "eternal"`. The distinction is only visible on the GPU leg,
which runs its tests on the runner rather than on the remote pool.

### A merge run re-reports, it does not measure

Nearly every target on a push to `main` is a cache hit: the merge carries the
same content as the pull request's head, so bazel replays what that run already
measured rather than running anything. Those durations are real and simply
older — the last real execution is the one that applies the next time the target
runs, so they are what to read — but "the worst time in the last green `main`
run" is one sample produced somewhere else, under load nobody recorded.

So gather the *distinct* duration values a target reports across several runs,
not one run's table. A value that repeats verbatim is one cache entry seen
twice, not two samples, and a sweep that misses that reads the CPU leg low.
