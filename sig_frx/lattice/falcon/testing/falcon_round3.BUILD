# Build file for the `falcon_round3` archive declared in //MODULE.bazel — the
# round-3 submission package, whose reference implementation is the oracle
# Falcon's interop test drives. Upstream ships a Makefile per parameter set;
# these targets replace it, because the sources are the whole of it: dependency
# free ANSI C, no generated headers, no configuration step.
#
# One target per parameter set rather than one for both. A set is its degree:
# `api.h` `#define`s the key and signature sizes and `nist.c` writes `9` or `10`
# into the calls themselves. Both then export the same `crypto_sign*`, so they
# cannot be linked into one artifact — which is why upstream ships them as two
# directories rather than one configurable build.
#
# `alwayslink` because nothing in the shim that links these references
# `crypto_sign`: the oracle reaches it through `dlopen`, so without it the
# linker drops every object as unused and produces a shared object with no
# Falcon in it.
#
# `no-remote` because this is the only C in the build and the CPU leg executes
# on Buildbarn (//.bazelrc.rbe). The workers do have a compiler — these actions
# ran there — but the C++ toolchain is the one bazel auto-detects by inspecting
# the machine that *configured* it, so its declared builtin include directories
# are the runner's. The workers' are gcc 11's, and every source here reaches
# `stdint.h`, so bazel rejects the compile it just ran:
#
#   absolute path inclusion(s) found in rule '@@+http_archive+falcon_round3//:falcon512'
#     '/usr/lib/gcc/x86_64-linux-gnu/11/include/stdint.h'
#
# Running the compiles where the toolchain was detected costs a few seconds of
# an otherwise remote build and keeps the interop test gating every pull
# request, which is what matters: a target excluded from a leg has never had its
# budget validated there (../../../../docs/reference/testing.md). The standing
# fix is a hermetic C toolchain in //MODULE.bazel — worth doing if this repo
# ever builds C for a second reason, and over-built for one test today.

load("@rules_cc//cc:defs.bzl", "cc_library")

[
    cc_library(
        name = "falcon%d" % degree,
        srcs = glob(["Reference_Implementation/falcon%d/falcon%dint/*.c" % (degree, degree)]),
        hdrs = glob(["Reference_Implementation/falcon%d/falcon%dint/*.h" % (degree, degree)]),
        includes = ["Reference_Implementation/falcon%d/falcon%dint" % (degree, degree)],
        tags = ["no-remote"],
        visibility = ["//visibility:public"],
        alwayslink = True,
    )
    for degree in [
        512,
        1024,
    ]
]

# §4.4's per-call sampler traces, which the archive ships beside the
# implementation. Every `SamplerZ` call of one signature at each degree, with
# the randomness it consumed and its intermediates — the centre, `ccs`, each
# `BaseSampler` draw and its result, and `ApproxExp`'s output before the
# comparison. That is what lets a sampler failure land on the algorithm that
# caused it rather than on a wrong signature ten steps later.
filegroup(
    name = "sampler_vectors",
    srcs = [
        "Supporting_Documentation/additional/test-vector-sampler-falcon1024.txt",
        "Supporting_Documentation/additional/test-vector-sampler-falcon512.txt",
    ],
    visibility = ["//visibility:public"],
)

# The submission's own known-answer files: 100 records per degree, each a key
# pair, a message and the signature the reference produced for it. Falcon
# publishes no ACVP set — FIPS 206 was still draft when this landed, and
# `usnistgov/ACVP-Server` carries directories for FIPS 204 and 205 and none for
# 206 — so these are the authority
# ([../../../../docs/reference/testing.md](testing.md) names the reference
# implementation next when a standard publishes no vectors).
#
# The `.req` half is the generator's input and is not taken: it carries the
# seeds and message lengths that produced these records, and reproducing a
# record *from* its seed would mean transcribing NIST's AES-256-CTR-DRBG, which
# is the harness's and not Falcon's.
filegroup(
    name = "kat_vectors",
    srcs = [
        "KAT/falcon1024-KAT.rsp",
        "KAT/falcon512-KAT.rsp",
    ],
    visibility = ["//visibility:public"],
)
