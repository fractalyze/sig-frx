# sig-frx docs

Topic-organized reference, indexed by what you're trying to do. For the project
overview and the build, see [`../README.md`](../README.md).

The tree is small on purpose: **[`reference/`](reference)** is the rules every
scheme is held to, **[`schemes/`](schemes)** is one design-notes page per scheme.

## `reference/` — the rules

| Question                                                                                     | Where                                    |
| -------------------------------------------------------------------------------------------- | ---------------------------------------- |
| What does this implementation claim against an adversary who can measure it — and what not?  | [`security.md`](reference/security.md)   |
| How a scheme is written — the batch axis, the namespace rule, rejection loops, the transforms and their precision | [`conventions.md`](reference/conventions.md) |
| What it is gated on — the transcribed reference, the published vectors, what the harness refuses | [`testing.md`](reference/testing.md) |
| What a number about it may claim — CI budgets, and recorded measurements                     | [`measurement.md`](reference/measurement.md) |

## `schemes/` — one page per scheme

| Question                                                | Where                          |
| --------------------------------------------------------- | ------------------------------ |
| What a scheme page must answer, and the shared machinery | [`README.md`](schemes/README.md) |
| Ethereum's post-quantum consensus signature — the spec pin, the seam's slot, what leaks | [`leansig.md`](schemes/leansig.md) |
| Module-lattice signatures — the two rejection loops, batch axis, what leaks | [`ml-dsa.md`](schemes/ml-dsa.md) |
| Hash-based signatures over a hypertree — parameter sets, batch axis, what leaks | [`slh-dsa.md`](schemes/slh-dsa.md) |
| Stateful hash-based signatures, single- and multi-tree — the index discipline, and who owns persistence | [`xmss.md`](schemes/xmss.md) |

Detailed design, findings, and open decisions live on the issues, not in the
tree — the epic is
[fractalyze/sig-frx#1](https://github.com/fractalyze/sig-frx/issues/1).

## The one seam

Every scheme implements `Signature`:
[`sig_frx/signature.py`](../sig_frx/signature.py). Its docstring is the contract;
the rule worth knowing before reading any scheme is that **`verify` is
batch-first and there is no scalar entry point** — a single verification is
`B = 1`. Verification is the hot path, it is embarrassingly parallel, and a seam
that admitted a scalar `verify` would get one implemented as a Python loop over
signatures.

Symmetric primitives are not implemented here. They come from
[`hash-frx`](https://github.com/fractalyze/hash-frx), which owns byte-exactness
and the GPU fusion for every hash this repo needs.
