# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Falcon's encodings (§3.11): the two keys, and the compressed signature.

Everything a Falcon key or signature is on the wire. No cryptography happens
here — it is bit manipulation over byte strings — but unlike ML-DSA's
counterpart, one half of it is a *variable-length* code, and that is what makes
this module the interesting one.

The fixed-length half still has one thing worth knowing: §3.11.5 stores `f`, `g`
and `F` and **not** `G`, which is recovered by (3.35) when the key is loaded. So
a private key here is three quarters of a trapdoor, and the missing quarter is
arithmetic rather than bits — [`keygen.recover_g`](keygen.py) rather than
anything in this file.

## The bit order is the opposite of ML-DSA's, and the names say so

§3.11.1 numbers a byte's bits from the left: the leftmost bit has weight 128, and
a multi-bit field is big-endian across the stream. FIPS 204 numbers from the low
end. So `bytes_to_bits_high_first` and `unpack_fields_high_first` are not
`encoding.bytes_to_bits` and `encoding.unpack_fields` from
[`mldsa/`](../mldsa/encoding.py) under a different roof, and neither pair can
stand in for the other: they agree only at whole-byte widths and disagree at
every width either scheme uses.

The suffix is there rather than the shared name the transform gets
([`conventions.md`](../../../docs/reference/conventions.md)), because these are
two operations that look alike and `base_mul` is one operation shaped twice.
The near-miss ML-DSA's `unpack_fields` already records against
`wots.base_2b` is this one, a third time — and a "unification" of any two of
them round-trips forever while being wrong.

## Decompress is a state machine, and the parse is its scan

Algorithm 18 walks the bit string with a cursor: eight bits of header, then as
many zeros as it takes to reach a `1`. The number of bits a coefficient occupies
is therefore the data, and a tracer has no cursor — which is the same shape as
[a rejection loop](../../../docs/reference/conventions.md) and gets a different
answer, because the answer there is a bound and here there is none worth having:
`n` coefficients occupy anywhere from `9n` bits to the whole string.

What a variable-length code does have is a *finite state machine*, and the
composition of state transitions is associative. So the parse is
`lax.associative_scan` over the nine states — the eight header positions and the
unary run — and every rejection falls out of where the scan lands rather than
out of a branch:

| state before bit `p` | what `p` is |
|---|---|
| `0` | the sign bit, so a coefficient starts here |
| `1..7` | a low bit, at weight `2^(7−s)` |
| `8` | inside the unary run; a `1` here ends the coefficient |

**The scan only has to find the terminators.** Coefficient `i` starts one bit
past coefficient `i−1`'s terminator, so the whole parse is the terminator
positions plus arithmetic, and those come off a `cumsum` and a `searchsorted`
the way [`rejection.first_accepted`](../rejection.py) takes the survivors of a
sampler. Nothing here scatters, and nothing here is indexed by a traced cursor.

**The walk is over bytes rather than over bits, and so is the ranking that
follows it. Both are measurements rather than preferences.** The transition
composed across a whole byte is a `[256, 9]` host constant — 256 values times
nine start states, built by running the per-bit machine — so the scan runs over
`slen/8` elements and the states *inside* each byte are recovered by eight
further steps, which is `slen/8` parallel walks of eight rather than one walk of
`slen`. Those same eight positions are where the terminators are counted, so the
prefix sum and the `n` searches see `slen/8` elements too. What makes that an
equivalent parse rather than a coarser one is that a coefficient is at least
nine bits, so **a byte closes at most one of them** and a `[256]` host table
takes the byte back to the bit — `_terminator_offsets` establishes that by
running the machine rather than by asserting it here.

It is written that way because the bit-level form made decompression the single
most expensive stage of verification. Measured on a workstation CPU at
Falcon-1024, `B = 256`, warm, over a published signature, with `verify` timed
around each form and the transition table kept in `uint8` throughout — nine
states fit in a byte, and this table is what every combine step moves, which
`_ON_ZERO` prices on its own. `decompress` and `verify` are relative to the
shipped form, the last row:

| walk | ranking | `decompress` | share | `verify` |
|---|---|---|---|---|
| bits | bits | 3.10x | 46% | 1.57x |
| bytes | bits | 1.49x | 31% | 1.11x |
| bytes | bytes | 1.00x | 23% | 1.00x |

**3.1x on the stage and 1.6x on the operation**, and the stage is no longer the
pole: `HashToPoint` is, at 69%. The two steps are not the same size — the walk
is worth 2.1x on the stage and the ranking 1.5x — and the second is taken
because it costs one host table and no second walk, not because it is large.

Every figure there comes from one session, which is the only way a share is a
share: this stage's number has moved by a quarter between sessions on unchanged
code, so one spliced in from another would compare two machines and call the
difference a speedup.

That is the opposite of the answer ML-DSA's `hint_bit_unpack` reaches, which
declines a 1.6x faster form because the stage it would speed up is 0.8% of a
verify. Same question, different number, other conclusion — which is why the
number is here rather than a preference for one shape.

**What the second index space costs.** The ranking counts in bytes where the
rest of the function reads in bits, so there is a seam: `holders` indexes bytes,
`ends` indexes bits, and everything below it — `starts`, `read`, the padding
check — stays in bits. That is a real cost in the one function whose rejections
are a malleability defence, and what makes it payable is that the terminators
were already being found eight positions at a time; only the prefix sum over
them was not.

One thing this does **not** claim. The numbers compare implementations, which is
what a local measurement is good for, and they size no budget
([`measurement.md`](../../../docs/reference/measurement.md)).

**The GPU leg has since been measured, and it does rank the stages differently.**
At Falcon-1024 and `B` = 1024, taken in one session against the `verify` it
divides, this decoder is **69%** of a GPU verification against 20% of a CPU one —
the reverse of the CPU ordering above, where `HashToPoint` is the pole
([`falcon.py`](falcon.py) carries both halves of that pair). So the byte-granular
ranking is the right shape for the leg it was chosen on, and it is also the stage
a GPU verification actually waits for.

**Inside it, the ranking is not what that time is.**
[`decoder_bench`](testing/decoder_bench.py) divides this function into the steps
it runs, each timed as the whole decoder *stopped* after that step, so that a
step is priced inside everything that precedes it rather than beside it. At
`B` = 1024, as a share of the decoder on each leg:

| step | GPU | CPU |
|---|---|---|
| the bit expansion | 2% | 22% |
| the `associative_scan` over bytes | 21% | 34% |
| the seven within-byte steps | **48%** | 13% |
| the `cumsum` that ranks the terminators | 1% | 8% |
| the `n` searches | 15% | 51% |
| the `[n]` gather and the offset table | 0% | −16% |
| the bit reads and the rejections | 6% | −10% |

The `searchsorted` is a sixth of the decoder on the leg the decoder is the pole
for, and the within-byte chain is half of it. The CPU column is the noisier one —
its negative marginals are a step that lets the compiler fuse a chain it was
materializing before — and it is the leg where none of this is the pole anyway.

**The chain's half was reformulated, and it buys nothing.** The eight positions a
byte closes at are a function of that byte and the state it is entered in, which
is the closure `_BYTE_STEP` already relies on, so the seven dependent steps
collapse to one `[256, 9]` host lookup. Byte-identical, interleaved against the
shipped form, and measured on both:

| leg | `B` | the step | `sig_decode` | `verify` |
|---|---|---|---|---|
| RTX 5090 | 256 | 2.54x | 1.00x | 1.00x |
| RTX 5090 | 1024 | 1.84x | 1.00x | 0.99x |
| workstation CPU | 256 | 1.21x | 0.96x | 0.99x |
| workstation CPU | 1024 | 1.20x | 0.88x | 0.98x |

So the step nearly halves and the operation does not move, which is why the chain
is still what is written below. **A step measured inside the whole prefix that
precedes it is still an upper bound rather than an estimate** — the sharper form
of what [`rejection.py`](../rejection.py)'s compaction found from isolated
stages, since every rung here is already fused with everything above it and the
over-attribution survived that. Seven dependent steps are latency, not
throughput, and in situ the decoder around them has enough independent work to
cover it; a rung ends in a reduction, and that is the one thing the real function
never does.

That also settles the shape of the question. It is about *this* decoder rather
than about the shared compaction — `first_accepted`'s own direction was measured
over both legs and moves neither operation ([`rejection.py`](../rejection.py)) —
and what is left on the GPU leg after the chain is the scan, at a fifth, which is
a smaller target than the half that just failed to pay.

## The rejections are the point of this module

Algorithm 18 enforces that a polynomial has **at most one** valid encoding, and
§3.11.3 that a padded signature has exactly one length. A decoder that accepts a
second encoding of the same `s` admits signature malleability: the forged
signature verifies, over the same message, under the same key, with different
bytes. That is not a robustness nicety, and the spec spends three numbered
paragraphs on it.

All four are evaluated over the whole buffer rather than short-circuited,
because a tracer has no branch to take:

- **the unary run has to terminate inside the string** — Algorithm 18 line 6
  walks `str[8 + k]` with no bound of its own, so a string that runs out mid-run
  (or mid-header) yields fewer than `n` terminators,
- **`100000001` is not a zero** — line 9's `(si = 0) and (str[0] = 1)`, the one
  coefficient value with two spellings,
- **the bits past the last terminator are zero** — lines 12 and 13, the same
  uniqueness requirement at the end of the buffer, which is also what makes
  §3.11.3's "partial padding is not valid" hold,
- **the string is exactly `slen` bits** — lines 1 and 2, which here is the
  static shape of the array rather than a check, since the seam hands
  verification one fixed length.

`s` comes back even when `ok` is false. It is meaningless in that case, and a
caller that ignores `ok` has accepted a malleable signature — which is why `ok`
is a return value rather than something to consult.
"""

from __future__ import annotations

from typing import Any

import frx.numpy as fnp
import numpy as np
from frx import lax
from frx.typing import ArrayLike

from sig_frx.arrays import namespace
from sig_frx.lattice.falcon.arith import Q

# §3.11.3's salt, between the header byte and the compressed coefficients.
SALT_SIZE = 40

# The bits Algorithm 18 reads before the unary run: one sign and seven low.
_HEADER_BITS = 8

# `bitlen(q − 1)` — §3.11.4 packs each public key coefficient at this width.
PK_BITS = (Q - 1).bit_length()

# §3.11.5's `f` and `g` widths by degree; `F` is eight bits at every degree and
# `G` is not encoded at all. Here rather than in the scheme module because every
# other §3.11 width is: a scheme module reading a wire width would be the one
# place the encoding is described twice.
SK_FG_BITS = {512: 6, 1024: 5}
SK_F_BITS = 8

# The nine states of Algorithm 18's cursor, as the image of each state under a
# `0` bit and under a `1` bit. They differ only at state 8, which is the whole
# of the encoding: everywhere else the cursor advances regardless of the bit,
# and in the unary run a `1` is what ends the coefficient.
# uint8 rather than a lane-width integer: nine states fit in a byte, and this
# table is what the scan below moves on every combine step, so the width is the
# scan's bandwidth. Measured at Falcon-1024 B=256, int32 costs 3.0x what uint8
# does, for bit-identical output.
_ON_ZERO = np.array([1, 2, 3, 4, 5, 6, 7, 8, 8], dtype=np.uint8)
_ON_ONE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 0], dtype=np.uint8)
_UNARY_STATE = 8


def _byte_transitions() -> np.ndarray:
    """The nine-state transition composed across each of the 256 byte values.

    A host constant, built by running the per-bit transitions above rather than
    transcribed, so the table and the machine it stands for cannot disagree.
    The scan below is over this instead of over single bits, which is what makes
    it a factor of eight shorter.
    """
    table = np.empty((256, _ON_ZERO.shape[0]), dtype=np.uint8)
    for value in range(256):
        for start in range(_ON_ZERO.shape[0]):
            state = start
            for shift in range(7, -1, -1):
                step = _ON_ONE if (value >> shift) & 1 else _ON_ZERO
                state = int(step[state])
            table[value, start] = state
    return table


_BYTE_STEP = _byte_transitions()


def _terminator_offsets() -> np.ndarray:
    """The bit a byte closes a coefficient at, indexed by that byte's mask.

    The table at the other end of the walk from the one above, and it is one
    entry wide because **a byte closes at most one coefficient**: a close
    returns the cursor to state 0, state 8 is eight transitions away, so no two
    terminators are closer than the nine bits a coefficient occupies. That is a
    property of the transition tables rather than an assumption about them, so
    it is taken by running the machine over every entry state and byte value —
    the rule `_byte_transitions` follows, for the reason it follows it.

    The mask packs the byte's terminator most significant bit first to match
    §3.11.1's order, so the table is the position of its single set bit. Index
    0 is the byte that closes nothing, which is only ever selected by a string
    that has already failed `terminated`.
    """
    for value in range(256):
        for start in range(_ON_ZERO.shape[0]):
            state, closes = start, 0
            for shift in range(7, -1, -1):
                bit = (value >> shift) & 1
                closes += int(state == _UNARY_STATE and bit)
                state = int((_ON_ONE if bit else _ON_ZERO)[state])
            if closes > 1:
                raise AssertionError(
                    f"byte {value:#04x} entered in state {start} closes {closes} "
                    "coefficients, so a byte no longer ranks one terminator"
                )
    table = np.zeros(256, dtype=np.uint8)
    for offset in range(8):
        table[1 << (7 - offset)] = offset
    return table


_TERMINATOR_OFFSET = _terminator_offsets()


def bytes_to_bits_high_first(values: ArrayLike) -> Any:
    """§3.11.1 — each byte most significant bit first, over the trailing axis.

    The mirror image of [`mldsa.encoding.bytes_to_bits`](../mldsa/encoding.py),
    including its namespace rule: a host value stays on numpy, so the key
    encoders above do not drag key generation onto a device
    ([`conventions.md`](../../../docs/reference/conventions.md)).

    `decompress` is the one function here that cannot follow the rule — its
    scan is an frx primitive with no host form — which is the case the rule
    itself grants an exception to.
    """
    xnp = namespace(values)
    data = xnp.asarray(values, dtype=np.uint8)
    bits = (data[..., None] >> np.arange(7, -1, -1, dtype=np.uint8)) & np.uint8(1)
    return bits.reshape(*data.shape[:-1], -1)


def bits_from_fields_high_first(v: ArrayLike, width: int) -> Any:
    """`v` as big-endian `width`-bit fields, still as bits.

    Stops at bits rather than bytes because §3.11.5 has three runs at two
    widths, so a caller concatenates them and converts once. Doing it per run
    would need each run to land on a byte boundary, which is true at both
    parameter sets and is not a property of the encoding.

    Named against [`fields_from_bits_high_first`](#fields_from_bits_high_first)
    rather than against `unpack_fields_high_first`: the pair that reads as
    inverses has to *be* one, and the unpacking below starts from bytes.
    """
    xnp = namespace(v)
    values = xnp.asarray(v, dtype=np.uint32)
    weights = np.arange(width - 1, -1, -1, dtype=np.uint32)
    bits = (values[..., None] >> weights) & np.uint32(1)
    return xnp.reshape(bits, (*np.shape(values)[:-1], -1)).astype(np.uint8)


def fields_from_bits_high_first(bits: ArrayLike, width: int) -> Any:
    """A bit stream read as big-endian `width`-bit fields, as uint32.

    Takes bits rather than bytes because §3.11.5 packs three runs at two widths
    into one stream, so the second and third do not start on a byte boundary in
    general. `unpack_fields_high_first` is this over a byte string, and is the
    common case rather than a separate implementation.

    The sum's dtype is pinned rather than inferred, for the reason ML-DSA's
    `unpack_fields` pins its own: numpy promotes a reduction's accumulator to
    `uint64` and frx does not, so leaving it open makes the host and traced
    paths differ in a width that no round trip can see
    ([`CLAUDE.md`](../../../CLAUDE.md)).
    """
    xnp = namespace(bits)
    values = xnp.asarray(bits, dtype=np.uint32)
    fields = xnp.reshape(values, (*np.shape(values)[:-1], -1, width))
    weights = np.arange(width - 1, -1, -1, dtype=np.uint32)
    return (fields << weights).sum(axis=-1, dtype=np.uint32)


def bits_to_bytes_high_first(bits: ArrayLike) -> Any:
    """§3.11.1 read backwards — eight bits to a byte, most significant first.

    A byte *is* an eight-bit field, so this is the general reader above at that
    width, narrowed back to the dtype a byte string has. The trailing axis must
    be a whole number of bytes; the check is that reshape rather than a length
    argument nobody could pass wrongly.

    `bytes_to_bits_high_first` is deliberately **not** collapsed the same way.
    It is the one of these four on a measured hot path — `decompress` is 69% of
    a GPU verification — and routing it through the `uint32` form above would
    widen every bit of every signature in the batch, which this module has
    already measured at 3.0x for the same reason one function further up.
    """
    return fields_from_bits_high_first(bits, 8).astype(np.uint8)


def unpack_fields_high_first(v: ArrayLike, width: int) -> Any:
    """`v` read as big-endian `width`-bit fields, as uint32."""
    return fields_from_bits_high_first(bytes_to_bits_high_first(v), width)


def degree_header(n: int, kind: int) -> int:
    """§3.11.3 / §3.11.4's header byte: four format bits over `log2(n)`.

    `kind` is the high nibble the format fixes — `0b0000` for a public key,
    `0b0011` for a compressed signature. §3.11.3's `0 c c 1` reads as `0011`
    once `cc` is `01`, the compressed encoding; `10`, the uncompressed
    alternative, is a different length and is not decoded here.
    """
    return (kind << 4) | (n.bit_length() - 1)


def pk_decode(pk: ArrayLike, n: int) -> tuple[Any, Any]:
    """§3.11.4 — `(h, ok)` from `1 + ⌈14n/8⌉` bytes.

    `14n` is a whole number of bytes at both degrees, so there is no leftover
    field and no tail to check — the padding rule that governs the signature has
    nothing to govern here.

    **The range check is a rejection and not an assertion.** Fourteen bits hold
    up to 16383 against a modulus of 12289, so a public key carrying a
    coefficient at or above `q` is representable, and what the reference
    implementation does with one is refuse it. Reducing instead would accept two
    distinct encodings of one key.
    """
    data = namespace(pk).asarray(pk, dtype=np.uint8)
    header = data[..., 0] == np.uint8(degree_header(n, 0b0000))
    coefficients = unpack_fields_high_first(data[..., 1:], PK_BITS)
    return coefficients, header & (coefficients < np.uint32(Q)).all(axis=-1)


def pk_encode(h: ArrayLike, n: int) -> Any:
    """§3.11.4 — `1 + ⌈14n/8⌉` bytes from `[n]` residues under `q`.

    The inverse of [`pk_decode`](#pk_decode), and the range check is not
    mirrored here: a coefficient at or above `q` is a *verdict* about a key that
    arrived from elsewhere, where the only producer on this side is
    [`keygen.public_key`](keygen.py) and its output is a residue by
    construction. Checking anyway would put a branch on the value in a function
    whose whole job is bit manipulation, and it would not catch anything the
    round trip does not.
    """
    xnp = namespace(h)
    header = xnp.asarray([degree_header(n, 0b0000)], dtype=np.uint8)
    bits = bits_from_fields_high_first(xnp.asarray(h, dtype=np.uint32), PK_BITS)
    return xnp.concatenate([header, bits_to_bytes_high_first(bits)], axis=-1)


def _to_two_complement(values: ArrayLike, width: int) -> Any:
    """Signed coefficients as `width`-bit two's complement fields."""
    xnp = namespace(values)
    signed = xnp.asarray(values, dtype=np.int32)
    return signed.astype(np.uint32) & np.uint32((1 << width) - 1)


def _from_two_complement(fields: ArrayLike, width: int) -> tuple[Any, Any]:
    """`width`-bit two's complement fields as signed, and whether any is `−2^(w−1)`.

    §3.11.5 forbids the minimal value — "when using degree 512, the valid range
    for a coefficient of `f` or `g` is −31 to +31; −32 is not allowed" — so it
    is a rejection rather than a value, and it is here rather than at the call
    site because the forbidden pattern is a property of the width.
    """
    xnp = namespace(fields)
    values = xnp.asarray(fields, dtype=np.uint32)
    high = (values >> np.uint32(width - 1)) & np.uint32(1)
    signed = values.astype(np.int32) - (high << np.uint32(width)).astype(np.int32)
    return signed, (values == np.uint32(1 << (width - 1))).any(axis=-1)


def _sk_widths(n: int) -> tuple[int, int, int]:
    """§3.11.5's three runs, in order: `f` and `g` at the degree's width, `F` at 8.

    One statement of the run order and its widths, read by both directions —
    two spellings of the same table is how an encoder and a decoder drift while
    each looks right on its own.
    """
    return SK_FG_BITS[n], SK_FG_BITS[n], SK_F_BITS


def sk_encode(f: ArrayLike, g: ArrayLike, big_f: ArrayLike, n: int) -> Any:
    """§3.11.5 — `1 + n·(2w + 8)/8` bytes carrying `f`, `g` and `F` in that order.

    `G` is absent because the standard leaves it out: it is recovered on load by
    (3.35), which is [`keygen.recover_g`](keygen.py). That asymmetry is the
    whole of why this takes three polynomials and Algorithm 6 returns four.

    The two widths — `w` for `f` and `g`, eight for `F` — are why the three runs
    are concatenated as *bits* and turned into bytes once. Each happens to land
    on a byte boundary at both parameter sets, so a per-run conversion would
    also work there and would break at the first degree where it did not.
    """
    xnp = namespace(f, g, big_f)
    header = xnp.asarray([degree_header(n, 0b0101)], dtype=np.uint8)
    runs = [
        bits_from_fields_high_first(_to_two_complement(values, width), width)
        for values, width in zip((f, g, big_f), _sk_widths(n))
    ]
    body = bits_to_bytes_high_first(xnp.concatenate(runs, axis=-1))
    return xnp.concatenate([header, body], axis=-1)


def sk_decode(sk: ArrayLike, n: int) -> tuple[Any, Any, Any, Any]:
    """§3.11.5 — `(f, g, F, ok)` from `1 + n·(2w + 8)/8` bytes.

    `ok` covers the header nibble and §3.11.5's forbidden minimum, which are the
    two ways a well-sized private key can still be malformed. Length is the
    caller's, exactly as it is for [`pk_decode`](#pk_decode): a wrong one is a
    mistake about the parameter set rather than a property of the bytes, and it
    surfaces as the reshape failing rather than as a verdict.

    There is no range check beyond the forbidden minimum, and there is nothing
    to add: every other `width`-bit pattern is a coefficient the encoding can
    represent. What makes a decoded key *wrong* rather than malformed is the
    NTRU equation, and that is arithmetic — `keygen.recover_g` is where it is
    checked, not here.
    """
    xnp = namespace(sk)
    data = xnp.asarray(sk, dtype=np.uint8)
    header = data[..., 0] == np.uint8(degree_header(n, 0b0101))
    bits = bytes_to_bits_high_first(data[..., 1:])

    ok, cursor, decoded = header, 0, []
    for width in _sk_widths(n):
        run = fields_from_bits_high_first(bits[..., cursor : cursor + n * width], width)
        values, forbidden = _from_two_complement(run, width)
        decoded.append(values)
        ok = ok & ~forbidden
        cursor += n * width
    f, g, big_f = decoded
    return f, g, big_f, ok


def decompress(data: ArrayLike, n: int) -> tuple[Any, Any]:
    """Algorithm 18 — `(s, ok)` from `enc_s`, where `ok` is `⊥` inverted.

    Takes the compressed bytes and reads `slen = 8·len(data)` bits out of them,
    which is the identity §3.11.3's framing already guarantees. Returns int32
    `[n]` centered coefficients. One signature at a time: the batch axis is
    `verify`'s `vmap`, as it is for every step of the assembly.
    """
    body = fnp.asarray(data, dtype=np.uint8)
    stream = bytes_to_bits_high_first(body)
    slen = stream.shape[-1]

    # Line 6's walk. The transition is composed a byte at a time from the host
    # table, so the scan is over `slen/8` elements; the eight steps below then
    # reopen each byte, which is `slen/8` parallel walks of eight rather than
    # one of `slen`. The scan is inclusive, so shifting by one gives the state
    # each byte is *entered* in.
    reached = lax.associative_scan(
        lambda first, second: fnp.take_along_axis(
            second, first.astype(np.int32), axis=-1
        ),
        fnp.take(_BYTE_STEP, body.astype(np.int32), axis=0),
        axis=0,
    )[:, 0]
    entered = fnp.concatenate([fnp.zeros(1, dtype=np.uint8), reached[:-1]])

    # The seven transitions inside a byte, not eight: the state after the last
    # bit is the next byte's `entered`, which the scan already produced.
    within = stream.reshape(-1, 8)
    walked = [entered]
    for offset in range(7):
        walked.append(
            fnp.where(
                within[:, offset] != 0,
                fnp.take(_ON_ONE, walked[-1].astype(np.int32)),
                fnp.take(_ON_ZERO, walked[-1].astype(np.int32)),
            )
        )

    # A `1` read in the unary state closes a coefficient. Ranking those by a
    # running count makes "where the i-th one is" a `searchsorted`, the same
    # mechanism `rejection.first_accepted` takes a sampler's survivors with.
    # The count is over bytes rather than bits, so the ranking runs at the
    # granularity the walk above it does: the eight walked positions are packed
    # where they stand — a byte closes at most one coefficient, so the mask
    # names it — and the prefix sum and the `n` searches see `slen/8` elements
    # rather than `slen`.
    closes = [
        (walked[offset] == np.uint8(_UNARY_STATE)) & (within[:, offset] != 0)
        for offset in range(8)
    ]
    mask = sum(
        closes[offset].astype(np.uint8) << np.uint8(7 - offset) for offset in range(8)
    )
    ranks = fnp.cumsum(mask != np.uint8(0), axis=-1, dtype=np.int32)
    wanted = fnp.arange(1, n + 1, dtype=np.int32)
    holders = fnp.searchsorted(ranks, wanted)
    # Lines 1-2 and 6: `n` coefficients have to close inside `slen` bits. A
    # string that runs out mid-run or mid-header closes fewer.
    terminated = ranks[..., -1] >= np.int32(n)

    # The byte the `i`-th terminator falls in, then the bit inside it: one `[n]`
    # gather and a table lookup, where the bit-level form read the position
    # straight off a `[slen]` prefix. `clip` for the reason `read` below takes
    # it — a string that ran out has already failed `terminated`, and a pinned
    # wrong answer is one every backend agrees on.
    holding = fnp.take(mask, holders, mode="clip").astype(np.int32)
    offsets = fnp.take(_TERMINATOR_OFFSET, holding).astype(np.int32)
    ends = holders * np.int32(8) + offsets

    # Line 11: the next coefficient starts one bit past this one's terminator.
    starts = fnp.concatenate([fnp.zeros(1, dtype=np.int32), ends[:-1] + np.int32(1)])

    def read(offset: int) -> Any:
        """Bit `offset` of every coefficient at once.

        `clip` rather than a bound: a string that ran out has already failed
        `terminated`, and pinning the unreachable read to the last bit keeps
        every backend at the same wrong answer instead of at its own.
        """
        return fnp.take(stream, starts + np.int32(offset), mode="clip").astype(np.int32)

    # Line 4: `str[1 + j]` carries weight `2^(6−j)`, most significant first.
    low = sum(read(1 + j) << np.int32(6 - j) for j in range(7))
    # Line 5's `k`, read off the distance to the terminator rather than counted.
    magnitude = low + ((ends - starts - np.int32(_HEADER_BITS)) << np.int32(7))
    negative = read(0) == np.int32(1)

    # Lines 9-10, then 12-13.
    canonical = ~((magnitude == np.int32(0)) & negative).any(axis=-1)
    position = fnp.arange(slen, dtype=np.int32)
    padded = ~((stream != 0) & (position > ends[..., -1])).any(axis=-1)

    coefficients = fnp.where(negative, -magnitude, magnitude)
    return coefficients, terminated & canonical & padded


def compress(coefficients: ArrayLike, length: int) -> tuple[np.ndarray, bool]:
    """Algorithm 17 — `(enc_s, ok)`, where `ok` is `⊥` inverted.

    Each coefficient becomes a sign bit, seven low bits of its magnitude, then
    the high part in unary: `⌊|s|/128⌋` zeros closed by a one. So a coefficient
    costs `9 + ⌊|s|/128⌋` bits and the total is data-dependent, which is why
    Algorithm 10 line 11 can be handed a `⊥` and has to draw again.

    **Host, and `numpy` rather than the array namespace the rest of this module
    reads.** Placing the bits is a scatter — `n` runs at positions only their own
    prefix sum knows — and a scatter has no namespace-generic spelling here.
    That costs nothing: the caller is signing, which is host-only because the
    sampler under it is (see [`sampler.py`](sampler.py)), so there is no traced
    path to serve. [`decompress`](#decompress) is the one that had to be device
    code, being 69% of a GPU verification, and its shape is deliberately not
    copied here.

    Nothing it emits can be refused by `decompress`: the sign bit is set only
    where the magnitude is nonzero, so no `-0` is produced, and the tail past
    the last terminator is zero-filled.
    """
    values = np.asarray(coefficients, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError(f"one signature at a time, not {values.ndim} axes")
    magnitude = np.abs(values)
    high = magnitude >> 7

    # A run per coefficient, and where each starts: the prefix sum of the
    # lengths, which is the only thing about the layout that is not local.
    lengths = _HEADER_BITS + high + 1
    starts = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(lengths)[:-1]])
    if int(lengths.sum()) > length:
        # Line 8's `⊥`. The buffer is still the right length so a caller that
        # ignores `ok` writes a well-formed rejection rather than a short array.
        return np.zeros(length // 8, dtype=np.uint8), False

    bits = np.zeros(length, dtype=np.uint8)
    bits[starts] = (values < 0).astype(np.uint8)
    for offset in range(7):
        bits[starts + 1 + offset] = ((magnitude >> (6 - offset)) & 1).astype(np.uint8)
    # The unary run is already zero; only its terminator has to be written.
    bits[starts + _HEADER_BITS + high] = 1
    return np.asarray(bits_to_bytes_high_first(bits), dtype=np.uint8), True


def sig_encode(
    salt: ArrayLike, coefficients: ArrayLike, n: int, sbytelen: int
) -> tuple[np.ndarray, bool]:
    """§3.11.3 — `(sig, ok)`, the padded `header ‖ salt ‖ enc_s` a verifier takes.

    The inverse of [`sig_decode`](#sig_decode), and the padding is what makes
    the two agree: `slen` is every bit the signature has left after the header
    and the salt, so `compress` fills the whole of it and Algorithm 18's
    trailing-zero rule reads the remainder as padding rather than as a
    coefficient.
    """
    material = np.asarray(salt, dtype=np.uint8)
    if material.shape != (SALT_SIZE,):
        raise ValueError(f"a salt is {SALT_SIZE} bytes, got {material.shape}")
    body, ok = compress(coefficients, slen(sbytelen))
    header = np.array([degree_header(n, 0b0011)], dtype=np.uint8)
    return np.concatenate([header, material, body]), ok


def slen(sbytelen: int) -> int:
    """Algorithm 16 line 2's `8·sbytelen − 328`, as the bits it actually names.

    The 328 is the header byte and the 40-byte salt, so the standard's constant
    and `8·(sbytelen − 1 − SALT_SIZE)` are the same expression — which is why
    only one of them is written here and nothing checks them against each other.
    A guard on that identity would compare `SALT_SIZE` against itself.
    """
    return 8 * (sbytelen - 1 - SALT_SIZE)


def sig_decode(sigma: ArrayLike, n: int, sbytelen: int) -> tuple[Any, Any, Any]:
    """§3.11.3 — `(salt, s, ok)` from a padded signature of `sbytelen` bytes.

    The header byte and the salt are fixed-width; everything after them is the
    compressed `s`, and `slen = 8·sbytelen − 328` is Algorithm 16 line 2's own
    expression for its bit length — 328 being the header byte and the 40-byte
    salt — which is to say `slen` is exactly the bits left over, and `decompress`
    reads that off the array rather than being told.
    """
    data = fnp.asarray(sigma, dtype=np.uint8)
    if data.shape[-1] != sbytelen:
        raise ValueError(
            f"a signature is {sbytelen} bytes, got {data.shape[-1]} — a batch of "
            f"the wrong length is `verify`'s verdict to return, not this decoder's"
        )
    header = data[..., 0] == np.uint8(degree_header(n, 0b0011))
    salt = data[..., 1 : 1 + SALT_SIZE]
    coefficients, ok = decompress(data[..., 1 + SALT_SIZE :], n)
    return salt, coefficients, header & ok
