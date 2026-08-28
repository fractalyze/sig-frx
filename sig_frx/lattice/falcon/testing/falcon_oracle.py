# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The round-3 reference implementation, as a second opinion on a key.

[`testing.md`](../../../../docs/reference/testing.md) puts "the reference
implementation the standard points at" last in the order of authorities and
Falcon reaches it: FN-DSA is draft, so there is no ACVP set, and the round-3
KAT gates only the direction where upstream produces and this repo checks.
That direction is covered — [`falcon_vectors`](falcon_vectors.py) carries the
published records and `falcon_kat_test` drives them. What it cannot cover is
the reverse: a key generated *here* being accepted *there*. Only the C can say
that, so this module compiles it and drives it.

## What "accepted" is, given there is no signer yet

`Falcon.sign` is [#27](https://github.com/fractalyze/sig-frx/issues/27), so
"generate here, sign here, verify there" is not available. The reverse is, and
it makes the same claim: the reference **signs** with a key generated here, and
this repo's `verify` accepts the result under the matching public key. A
signature that verifies could not have come from a key the reference refused to
load — so one round trip carries the acceptance, and it lands on the verifier
the published vectors already gate.

## What refuses a corrupted key, and why it is not the header byte

The concern is real and it does not apply here: a bit flipped deep inside `f`
would ordinarily give a *different valid-looking* key rather than an invalid
one, and a test asserting rejection would then be asserting nothing. Falcon's
loader is stronger than that. `crypto_sign` decodes `f`, `g` and `F` and then
calls `complete_private`, which recovers `G = g·F/f mod q` and refuses if any
coefficient falls outside `[-127, +127]` — the range §3.11.5 leaves it, since
`G` is not encoded and (3.35) has to rebuild it.

For a genuine trapdoor that quotient is tiny by construction. For a corrupted
one it is nothing in particular, so each coefficient survives with probability
about `255/12289`, and all `n` of them do with probability around `0.02^n`.
Measured over 400 single-bit flips into a coefficient byte of the published
`Falcon-512` key: **396 refused in `complete_private`, 4 in `trim_i8_decode`**
(which rejects the forbidden `-2^(bits-1)` and nonzero padding bits), and none
was accepted. So the corruption goes where the criterion means it to go, into
the key material, rather than into the header byte where a rejection would
prove only that a constant was compared.

## Only `crypto_sign` is bound

`crypto_sign_open` is the other half of the API and is deliberately absent:
nothing here produces a signature for it to judge yet, and a binding no test
exercises is a binding nobody knows works. It arrives with #27, which is the
issue that gains a producer.
"""

from __future__ import annotations

import ctypes
import functools
import pathlib

from sig_frx.lattice.falcon import encoding, falcon

# `nist.c`'s aggregate: a big-endian signature length, then the salt, then the
# message, then the nonce-less signature. §3.11.6 rather than §3.11.3, which is
# what the seam speaks — `signature` below is the regrouping between them.
_SIGLEN_SIZE = 2


@functools.cache
def _library(degree: int) -> ctypes.CDLL:
    """The parameter set's shared object, loaded once per process.

    One per degree, because `api.h` fixes the degree by `#define` and both sets
    export the same `crypto_sign` — they are separate shared objects for the
    same reason upstream builds them in separate directories. `CDLL` maps with
    `RTLD_LOCAL`, so holding both at once resolves each call inside the object
    it was reached through rather than by whichever loaded first.
    """
    path = pathlib.Path(__file__).with_name(f"libfalcon{degree}_oracle.so")
    if not path.exists():  # pragma: no cover - a packaging error, not a case
        raise FileNotFoundError(
            f"the Falcon-{degree} oracle is missing from the runfiles at {path}; "
            "the test target needs it in `data`"
        )
    library = ctypes.CDLL(str(path))
    library.falcon_oracle_seed.argtypes = [ctypes.c_uint64]
    library.falcon_oracle_seed.restype = None
    library.crypto_sign.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.c_char_p,
        ctypes.c_ulonglong,
        ctypes.c_char_p,
    ]
    library.crypto_sign.restype = ctypes.c_int
    return library


def sign(
    secret_key: bytes, message: bytes, degree: int, *, seed: int = 1
) -> bytes | None:
    """Sign `message` with the reference, or `None` if it refused the key.

    `seed` keys the shim's generator (`falcon_oracle.c`) rather than anything
    upstream: Falcon draws a salt per signature, so the reference is randomized
    and a test that wants to reproduce a failure has to say where the bytes
    came from. It does not make this agree with the published signatures, which
    would need the NIST harness's DRBG — and which nothing here claims.

    The return is §3.11.3's `header ‖ salt ‖ enc_s`, zero-padded to `sbytelen`,
    which is the seam's form. What the reference hands back is §3.11.6's
    aggregate with the message in the middle and a nonce-less header, so the
    regrouping below is the same one `falcon_vectors` documents, run forwards.
    """
    params = falcon.PARAMETER_SETS[f"Falcon-{degree}"]
    if len(secret_key) != params.secret_key_size:
        raise ValueError(
            f"a Falcon-{degree} secret key is {params.secret_key_size} bytes, "
            f"got {len(secret_key)}"
        )
    library = _library(degree)
    library.falcon_oracle_seed(seed)

    # `crypto_sign` writes the message back out between the salt and the
    # signature, so the buffer holds the aggregate rather than the signature.
    signed = ctypes.create_string_buffer(
        _SIGLEN_SIZE + encoding.SALT_SIZE + len(message) + params.signature_size
    )
    signed_len = ctypes.c_ulonglong(0)
    status = library.crypto_sign(
        signed, ctypes.byref(signed_len), message, len(message), secret_key
    )
    if status != 0:
        return None

    aggregate = signed.raw[: signed_len.value]
    body = _SIGLEN_SIZE + encoding.SALT_SIZE
    salt = aggregate[_SIGLEN_SIZE:body]
    nonceless = aggregate[body + len(message) :]
    if len(nonceless) != int.from_bytes(aggregate[:_SIGLEN_SIZE], "big"):
        raise AssertionError("the reference's own length field disagrees with `sm`")
    if nonceless[0] != encoding.degree_header(degree, 0x2):
        raise AssertionError(
            f"expected §3.11.6's nonce-less header, got 0x{nonceless[0]:02x}"
        )
    compressed = nonceless[1:]
    padding = params.signature_size - 1 - encoding.SALT_SIZE - len(compressed)
    if padding < 0:  # pragma: no cover - upstream enforces `sbytelen` itself
        raise AssertionError(
            f"the reference produced {len(compressed)} compressed bytes, past "
            f"what a Falcon-{degree} signature holds"
        )
    return (
        bytes([encoding.degree_header(degree, 0x3)])
        + salt
        + compressed
        + b"\x00" * padding
    )
