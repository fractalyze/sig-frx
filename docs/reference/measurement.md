# Measurement

> Code, symbols, and file paths are English.

A number about this repo is one of two things, and conflating them is the
failure this page exists to prevent. A **budget** is the deadline a target has
to finish inside on CI, sized from the worst run of the slowest leg it runs on.
A **recorded measurement** compares two implementations, is written into a
docstring, and is the argument that made the code what it is — it sizes
nothing, and a budget says nothing about it.

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
they do not size a budget — and [what such a comparison has to be to mean
anything](#a-recorded-measurement-is-an-argument-and-it-states-its-terms) is the
rest of this page.

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

## A recorded measurement is an argument, and it states its terms

This is the kind of number the repo carries most, and it is not the kind the
section above governs. It is the performance measurement written into a module
docstring: [`falcon.py`](../../sig_frx/lattice/falcon/falcon.py)'s division of a
verification into stages,
[`encoding.py`](../../sig_frx/lattice/falcon/encoding.py)'s table over three
decoder forms, `hint_bit_unpack`'s reason for declining a faster one,
[`rejection.py`](../../sig_frx/lattice/rejection.py)'s reason for leaving a
compaction alone. Those are not documentation of the code, they are the argument
that made it what it is — `hint_bit_unpack` refuses a 1.6x form *because* its
stage is 0.8% of a verify, and nothing else about that function explains its
shape.

They are also the least rechecked claim here: written once, true when written,
and read a milestone later as though still measured. So what a recorded number
has to be is worth stating, and every rule below is here because a number that
broke it was written down and believed.

### An A/B is interleaved, or the first number is the one you believe

Time the two forms alternately and take a median over enough samples that the
spread is visible — not all of A and then all of B. Run-to-run spread on these
stages is around 20% (±1.0 ms on 5 ms), so one sequential block lands anywhere
in that range and there is nothing in the number that says where.

Twice, the number a sequential block returned would have decided differently:

- 20 samples per form in sequential blocks reported **1.07x** for a change that
  3x40-sample interleaved medians put at **1.38x** — identical code, same box,
  idle machine.
- A three-sample smoke run reported **1.75x** on a Falcon `verify` for a change
  whose interleaved median is **1.01x**.

The second is the one worth remembering, because the honest answer there was "no
difference" — and that is the answer a sloppy A/B is least likely to return.
Noise manufactures differences; it does not manufacture ties.

### A share comes from one session, or it is not a share

A stage and the total it is a share of are taken together, or the share is not
stated. On unchanged code across two sessions the decoder stage moved from 5.8
to **7.3 ms** — 25% — while the `verify` containing it (21.7 against 21.0) and
`HashToPoint` beside it (14.7 against 14.5) both held to 4%. The drift is real
and it is not uniform across stages, so it cannot be divided out: a numerator
from one run over a denominator from another compares two machines and reports
the difference as a property of the code.

It is why `falcon.py` states its GPU stage division at `B` = 1024 rather than at
the 256 its CPU table uses. The batch is not the decision there; the session is.

### Every decomposition prices a program cut where the real one is whole

Time the operation, not only the stage. When the two disagree the operation is
the one that is about what a caller waits for, and they disagree by a lot:
`rejection.first_accepted`'s scatter form wins its isolated stage by 1.3-6.3x on
CPU and 1.0-4.6x on GPU, at nearly every site and batch, while `verify` is flat
on both legs at 0.96-1.04x.

The tell is stark. Timed on its own at `B` = 1024, the `ExpandA` compaction
costs **1.397 ms inside a `verify` that costs 0.760 ms in total**. A component
dearer than its whole is proof the standalone program is paying for memory
traffic that does not exist in situ — the compaction is fused with the SHAKE
that produces its candidates and the arithmetic that consumes its survivors, so
it never makes the round trip a benchmark forces on it.

**Fusing the step into the prefix that precedes it does not repair that.** It is
the obvious fix and it fails, which is what makes it worth writing down.
[`decoder_bench`](../../sig_frx/lattice/falcon/testing/decoder_bench.py) prices
each step of `decompress` as the whole decoder *stopped* after that step, so
every rung is already fused with everything above it. The within-byte chain came
out at 48% of the decoder on GPU; collapsing it to a `[256, 9]` lookup takes the
step 1.84-2.54x, moves `verify` by 0.98-1.00x, and costs CPU `sig_decode` 12%.
Two residues a prefix ladder cannot remove:

- a rung has to return something or the compiler deletes it, so it ends in a
  reduction — a fusion barrier the whole function never pays;
- dependent steps are **latency, not throughput**. Seven serial steps look
  expensive as the last thing in a program; in situ the code around them has
  enough independent work to cover them, so removing them frees a dependency
  chain nothing was waiting on.

So an isolated stage bounds what changing it can buy and does not estimate it,
a fused prefix is the tighter bound and still not an estimate, and the A/B is
budgeted against the operation from the start rather than after a stage figure
has already been believed.

None of which says don't profile. A per-part breakdown answers a different
question well: it *ranks* what to try next, and it cannot *price* the fix. The
decoder's breakdown earned its keep by moving the work off a `searchsorted` that
was 15% of the decoder and onto the chain that was 48% — and the change it
pointed at then bought nothing, which is the half only the operation could
report.

A docstring stating a stage ratio says so in those words.

### A benchmark proves its two columns are two programs

Two ways a bench reports a number for work it did not do, both of which read as
results rather than as failures.

**Routing that does not reach the trace.** Measuring the operation rather than
the stage means swapping the form underneath it — a module attribute like
`rejection.first_accepted` or `encoding.decompress`. If the swap misses, both
columns time one executable and print **1.00x**, which is exactly what a real
"this change is worth nothing" result looks like — and one of the two real
results here *was* 1.00x. The clock cannot separate them, so ask the program
instead:

```python
sizes[label] = len(program.lower(data).as_text().splitlines())
if len(set(sizes.values())) == 1:
    raise RuntimeError("the routing did not reach the trace")
```

998 lines against 814, for the two decoder forms. A rung that takes its form as
a *parameter* cannot stand in for that check — it differs even when the routing
is dead. Same family, one level down: build a fresh `jit` per form, or the
second lowering answers out of the first one's trace cache and one form is timed
twice.

**Input the operation rejects before evaluating it.** Both lattice schemes
return `false` for a whole batch *without computing anything* when a key or
signature length is wrong (FIPS 204 §3.6.2), so a mis-shaped batch still times —
fast, stable, and reading as a speedup. A benchmark over a rejectable input
asserts the input was accepted, or the fastest row in its table is the one where
the work was skipped.
