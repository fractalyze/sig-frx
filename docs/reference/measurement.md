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

**The factor is the CPU leg's.** CI does not run the two legs on one machine:
the CPU leg goes to the Buildbarn pool and the GPU leg executes on the
self-hosted runner itself, because the pool is CPU-only
([`ci.yml`](../../.github/workflows/ci.yml)). The factor above is therefore
measured on the only leg that has an executor, and stacking it on top of a GPU
measurement over-sizes. A GPU-decided budget can be sized from a local run on a
comparable device; a CPU-decided one cannot. Which leg decides is the slower
one, and for a compile-bound target that is usually the GPU.

What a local GPU number is worth is an order of magnitude, not a tenth. The same
four targets against the CI figures this page records, measured on one
workstation in two sessions and carried as a ratio to CI rather than as the
local durations themselves, which are the machine's and not the target's. The
sessions stay in separate columns because a ratio and its denominator have to
come from one of them:

| target | CI, as recorded | one session | another |
|---|---|---|---|
| `slh_dsa_traced_test` | 177.7 s | 1.03x | 0.78x |
| `falcon_kat_test` | 172.1 s | 1.12x | 0.85x |
| `falcon_test` | 272.9 s | 1.13x | 1.04x |
| `slh_dsa_kat_test` | ~285 s | 1.23x | 1.39x |

**0.8x to 1.4x, and it runs both ways.** `slh_dsa_kat_test` at 1.39x is a local
run *under*-sizing, which is the opposite of the error stacking an executor
factor would make — so the rule is not "divide by something smaller", it is that
a local GPU number sizes a budget and does not settle a bucket boundary. One
landing near half takes the next bucket.

Two things plausibly behind that spread, neither verified and both worth knowing
rather than resolving first: a target's duration under a full concurrent sweep
is a function of what else is on the device at that moment, so neither column is
a clean per-target measurement and neither is CI's; and a workstation's core
count sets a different concurrency than the runner's.

### Running the GPU leg locally

Why the leg needs its flags, and the control that proves a green one used the
device rather than quietly measuring a CPU, are the playbook's — it carries them
as
[`sections/environment.md`](https://github.com/fractalyze/claude-plugins/blob/main/plugins/playbook/sections/environment.md).
This is the invocation for this repo:

```sh
bazel --bazelrc=.bazelrc.ci test \
  --test_env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
  --test_env=FRX_PLATFORMS=cuda -- //...
```

`ci.yml` passes exactly those two, and dropping the preallocate one produced 31
failures out of 50 here. A box carrying no CUDA 12 toolkit adds the runtime and
`ptxas` by hand; the repo's pinned frx wheel is still what runs:

```sh
V=<virtualenv>/lib/python3.11/site-packages/nvidia
LD=$(find "$V" -maxdepth 2 -type d -name lib | tr '\n' ':')
bazel --bazelrc=.bazelrc.ci test \
  --test_env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
  --test_env=FRX_PLATFORMS=cuda \
  --test_env=LD_LIBRARY_PATH="$LD" \
  --test_env=XLA_FLAGS="--xla_gpu_cuda_data_dir=$V/cuda_nvcc" -- //...
```

A bench is a `bazel run` rather than a test, and takes the same environment
through `--run_under="env ..."` instead of `--test_env`.

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
and read a milestone later as though still measured. Every anchor below is here
because a number that broke a rule was written down and believed.

### The rules are the playbook's; the evidence behind them is this repo's

What a recorded number has to be — an interleaved A/B, a share and its total
from one session, an isolated stage read as a bound rather than an estimate, a
bench that proves its two columns are two programs — follows from how
measurement works rather than from what this repo computes. No repo owns it and
a copy per repo is how it drifts, so the playbook carries it as
[`sections/measurement.md`](https://github.com/fractalyze/claude-plugins/blob/main/plugins/playbook/sections/measurement.md).

What is this repo's is the evidence, and it belongs where the modules arguing
from it can cite it:

| the rule it anchors | measured here |
|---|---|
| an A/B is interleaved | sequential blocks read **1.07x** where interleaved medians read **1.38x**; a three-sample smoke run read **1.75x** on a Falcon `verify` whose median is **1.01x** — the honest answer was "no difference", which is the answer a sloppy A/B is least likely to return |
| a share comes from one session | the decoder stage moved **25%** across two sessions on unchanged code while `verify` and `HashToPoint`, measured in the same two, held to 4% |
| an isolated stage is a bound | `rejection.first_accepted`'s scatter wins its stage by 1.3-6.3x on CPU and 1.0-4.6x on GPU while `verify` stays flat at 0.96-1.04x, and the `ExpandA` compaction times **1.8x the `verify` it sits inside** |
| a fused prefix does not repair it | [`decoder_bench`](../../sig_frx/lattice/falcon/testing/decoder_bench.py) prices each step as the whole decoder *stopped* there; collapsing the 48% within-byte chain to a `[256, 9]` lookup takes the step 1.84-2.54x and moves `verify` 0.98-1.00x, at 12% on CPU `sig_decode` |

Two of those decide something a reader would otherwise have to guess at.
`falcon.py` states its GPU stage division at `B` = 1024 rather than the 256 its
CPU table uses — the batch is not the decision there, the session is. And the
decoder breakdown still earned its keep: it moved the work off a `searchsorted`
that was 15% of the decoder onto the chain that was 48%, ranking correctly and
unable to price the fix, which is the whole rule in one example.

**A docstring stating a stage ratio says so in those words.**

### Two shapes the "prove the experiment fired" rule takes here

**Routing that does not reach the trace.** Measuring the operation rather than
the stage means swapping the form underneath it — a module attribute like
`rejection.first_accepted` or `encoding.decompress`. If the swap misses, both
columns time one executable and print **1.00x**, which is what a real "worth
nothing" result looks like, and one of the two real results here *was* 1.00x. So
`decoder_bench` asks the program instead of the clock:

```python
sizes[label] = len(program.lower(data).as_text().splitlines())
if len(set(sizes.values())) == 1:
    raise RuntimeError("the routing did not reach the trace")
```

998 lines against 814, for the two decoder forms. A rung that takes its form as
a *parameter* cannot stand in — it differs even when the routing is dead. Same
family one level down: build a fresh `jit` per form, or the second lowering
answers out of the first one's trace cache and one form is timed twice.

**Input the operation rejects before evaluating it.** This one is the scheme's
rather than the harness's, which is why it stays here. Both lattice schemes
return `false` for a whole batch *without computing anything* when a key or
signature length is wrong (FIPS 204 §3.6.2), so a mis-shaped batch still times —
fast, stable, and reading as a speedup. A benchmark over a rejectable input
asserts the input was accepted, or the fastest row in its table is the one where
the work was skipped.
