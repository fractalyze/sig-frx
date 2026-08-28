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

load("@rules_cc//cc:defs.bzl", "cc_library")

[
    cc_library(
        name = "falcon%d" % degree,
        srcs = glob(["Reference_Implementation/falcon%d/falcon%dint/*.c" % (degree, degree)]),
        hdrs = glob(["Reference_Implementation/falcon%d/falcon%dint/*.h" % (degree, degree)]),
        includes = ["Reference_Implementation/falcon%d/falcon%dint" % (degree, degree)],
        visibility = ["//visibility:public"],
        alwayslink = True,
    )
    for degree in [
        512,
        1024,
    ]
]
