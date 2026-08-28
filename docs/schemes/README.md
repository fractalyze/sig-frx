# Scheme design notes

One page per scheme, added as the scheme lands. A page is design notes, not an
API tour: the API is the [`Signature`
seam](../../sig_frx/signature.py) plus the scheme's own module, and both are
readable.

| Scheme | Where |
| ------ | ----- |
| FROST (RFC 9591) | [`frost.md`](frost.md) |
| leanSig (generalized XMSS, leanSpec) | [`leansig.md`](leansig.md) |
| ML-DSA (FIPS 204) | [`ml-dsa.md`](ml-dsa.md) |
| MuSig2 (BIP-327) | [`musig2.md`](musig2.md) |
| SHRINCS | [`shrincs.md`](shrincs.md) |
| SLH-DSA (FIPS 205) | [`slh-dsa.md`](slh-dsa.md) |
| XMSS and XMSS-MT (RFC 8391) | [`xmss.md`](xmss.md) |

The three questions every page answers — what the standard fixes versus what this
implementation chooses, where the batch axis is, and what leaks — are specified
in [`../reference/conventions.md`](../reference/conventions.md#scheme-doc-skeleton).

Shared machinery that more than one scheme builds on (the tweakable hash family,
WOTS+, the Merkle hash tree, the lattice NTTs) is documented where it lives
rather than duplicated into each scheme's page. A scheme page names what it uses
and moves on.
