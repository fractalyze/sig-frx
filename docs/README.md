# sig-frx docs

Topic-organized reference, indexed by what you're trying to do. For the project
overview and the build, see [`../README.md`](../README.md).

The tree is small on purpose: **[`reference/`](reference)** is the rules every
scheme is held to, **[`schemes/`](schemes)** is one design-notes page per scheme.

## `reference/` — the rules

| Question                                                                                     | Where                                    |
| -------------------------------------------------------------------------------------------- | ---------------------------------------- |
| What does this implementation claim against an adversary who can measure it — and what not?  | [`security.md`](reference/security.md)   |
| The rules a scheme is held to — the batch axis, rejection loops, the KAT gate                | [`conventions.md`](reference/conventions.md) |

## `schemes/` — one page per scheme

| Question                                                | Where                          |
| --------------------------------------------------------- | ------------------------------ |
| What a scheme page must answer, and the shared machinery | [`README.md`](schemes/README.md) |
| Hash-based signatures over a hypertree — parameter sets, batch axis, what leaks | [`slh-dsa.md`](schemes/slh-dsa.md) |

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
