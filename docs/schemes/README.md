# Scheme design notes

One page per scheme, added as the scheme lands. A page is design notes, not an
API tour: the API is the [`Signature`
seam](../../sig_frx/signature.py) plus the scheme's own module, and both are
readable.

| Scheme | Where |
| ------ | ----- |

The three questions every page answers — what the standard fixes versus what this
implementation chooses, where the batch axis is, and what leaks — are specified
in [`../reference/conventions.md`](../reference/conventions.md#scheme-doc-skeleton).

Shared machinery that more than one scheme builds on (the tweakable hash family,
WOTS+, the Merkle hash tree, the lattice NTTs) is documented where it lives
rather than duplicated into each scheme's page. A scheme page names what it uses
and moves on.
