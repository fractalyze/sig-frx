/* Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
 *
 * The one symbol the reference implementation needs and does not define.
 *
 * `nist.c` declares `randombytes` and leaves it to the NIST harness, which
 * draws from an AES-256-CTR-DRBG seeded per test case. That harness is not
 * vendored here and reproducing it would buy nothing: this repo's `keygen`
 * deliberately does not reproduce the published keys (Falcon fixes no map from
 * a seed to a key pair), so the oracle never has to agree with upstream's draw
 * — it only has to sign with a key handed to it.
 *
 * A verify-only oracle could abort in here, since `crypto_sign_open` never
 * draws. `crypto_sign` does, twice: the 40-byte nonce and the 48-byte seed the
 * sampler runs on. So the interop direction that matters — the reference
 * signing with a key generated here — needs bytes rather than a trap.
 *
 * xorshift64* keyed by the caller, so a failing test names a seed that
 * reproduces it. Nothing here rests on the quality of these bytes: the nonce
 * is public and the signature is checked by a verifier that does not care how
 * it was sampled. It is emphatically not for producing keys anyone keeps.
 */

#include <stdint.h>

static uint64_t state = UINT64_C(0x853c49e6748fea9b);

void falcon_oracle_seed(uint64_t seed) {
  /* xorshift64* is a fixed point at zero; the reference draws immediately
   * after seeding, so a zero seed would hand it a constant stream. */
  state = seed ? seed : UINT64_C(0x9e3779b97f4a7c15);
}

int randombytes(unsigned char *out, unsigned long long len) {
  for (unsigned long long i = 0; i < len; i++) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    out[i] = (unsigned char)((state * UINT64_C(0x2545f4914f6cdd1d)) >> 56);
  }
  return 0;
}
