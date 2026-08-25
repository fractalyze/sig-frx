# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""SLH-DSA agrees with FIPS 205 §9 and §10, and with Table 2.

What this module adds over the components beneath it is assembly and encodings, so
that is what these cases pin: the parameter-set derivations against Table 2's
published columns, the digest split against the standard's own slice arithmetic,
and Algorithms 19 and 23 against a transcription that calls the components in the
order the standard writes them.

The claim beyond the standard is the batch axis in `verify`, so the standard's
per-signature form runs beside it: `B` single-entry calls must agree with one call
over the batch, entry for entry, including where one entry is meant to fail.

Small parameters (`h = 4`, `d = 2`, `a = 3`, `k = 4`) for the cases that check the
arithmetic, since the assembly does not care and a real set's top XMSS tree is 512
WOTS+ key pairs; both real category 1 sets for the one case that checks the
dimensions. `n = 16` throughout, since that is the security category §11.2.1's
SHA-256-only family covers.

Reproducing published bytes is `slh_dsa_kat_test`, which is what closes the gap
these cases cannot: one author wrote both the implementation and the transcription
here, so a misreading they share survives both.
"""

from __future__ import annotations

import hashlib

import numpy as np
from absl.testing import absltest
from frx import Array
from frx.typing import ArrayLike
from hash_frx import Sha256

from sig_frx.context import MAX_SIZE as MAX_CONTEXT_SIZE
from sig_frx.hash import tree
from sig_frx.hash.slhdsa import fors, hypertree, slh_dsa, xmss
from sig_frx.hash.tweakable import Sha2TweakableHash

_PARAMS = slh_dsa.SlhDsaParams(n=16, h=4, d=2, a=3, k=4)
_SEED = np.frombuffer(bytes((i * 7 + 3) % 256 for i in range(3 * 16)), dtype=np.uint8)
_OTHER_SEED = np.frombuffer(
    bytes((i * 11 + 90) % 256 for i in range(3 * 16)), dtype=np.uint8
)
_MESSAGE = np.frombuffer(b"the content that gets signed", dtype=np.uint8)
_ADDRND = np.frombuffer(bytes(range(200, 216)), dtype=np.uint8)
_CONTEXT = np.frombuffer(b"a protocol domain", dtype=np.uint8)

# Table 2, transcribed. Every column here is one the table computes from `n`, `h`,
# `d`, `a` and `k` rather than one it fixes, so agreeing with it is a check on the
# derivations rather than a restatement of the row.
_TABLE_2 = (
    # name, h', m, security category, pk bytes, sig bytes
    ("SLH-DSA-SHA2-128s", 9, 30, 1, 32, 7856),
    ("SLH-DSA-SHA2-128f", 3, 34, 1, 32, 17088),
    ("SLH-DSA-SHA2-192s", 9, 39, 3, 48, 16224),
    ("SLH-DSA-SHA2-192f", 3, 42, 3, 48, 35664),
    ("SLH-DSA-SHA2-256s", 8, 47, 5, 64, 29792),
    ("SLH-DSA-SHA2-256f", 4, 49, 5, 64, 49856),
)


def _family() -> Sha2TweakableHash:
    return Sha2TweakableHash(Sha256(), n=_PARAMS.n, m=_PARAMS.m)


def _scheme(*, deterministic: bool = True) -> slh_dsa.SlhDsa:
    return slh_dsa.SlhDsa(_family(), _PARAMS, deterministic=deterministic)


class _CountingHash:
    """A `ByteHash` that counts its calls.

    The seam is dependency-injected, so the scheme reaches every hash through
    this and the count is exact rather than inferred.
    """

    def __init__(self) -> None:
        self._inner = Sha256()
        self.digest_size = self._inner.digest_size
        self.fusion_path = self._inner.fusion_path
        self.calls = 0

    def digest(self, msg: ArrayLike) -> Array:
        self.calls += 1
        return self._inner.digest(msg)


def _pure_message(message: bytes, context: bytes = b"") -> bytes:
    """`M'` — Algorithm 22 line 8, transcribed."""
    return bytes([0, len(context)]) + context + message


def _prehash_message(message: bytes, context: bytes = b"") -> bytes:
    """`M'` — Algorithm 23 line 24, transcribed, pre-hashing with SHA-256.

    The OID is line 10's, and `PH_M` comes from the standard library rather than
    from hash-frx, so the pre-hash is not checked against itself.
    """
    return (
        bytes([1, len(context)])
        + context
        + bytes.fromhex("0609608648016503040201")
        + hashlib.sha256(message).digest()
    )


def _spec_split(digest: bytes) -> tuple[bytes, int, int]:
    """Algorithm 19 lines 6 to 10, transcribed from the slice expressions."""
    h, d, k, a = _PARAMS.h, _PARAMS.d, _PARAMS.k, _PARAMS.a
    ka = -(-(k * a) // 8)
    tree_width = -(-(h - h // d) // 8)
    leaf_width = -(-h // (8 * d))
    md = digest[0:ka]
    tmp_idx_tree = digest[ka : ka + tree_width]
    tmp_idx_leaf = digest[ka + tree_width : ka + tree_width + leaf_width]
    idx_tree = int.from_bytes(tmp_idx_tree, "big") % (1 << (h - h // d))
    idx_leaf = int.from_bytes(tmp_idx_leaf, "big") % (1 << (h // d))
    return md, idx_tree, idx_leaf


def _spec_sign(secret_key: np.ndarray, message: bytes, opt_rand: np.ndarray) -> bytes:
    """Algorithm 19, transcribed over the components in the order it writes them.

    `opt_rand` is line 2's choice — `PK.seed` for the deterministic variant, and
    `addrnd` for the hedged one.
    """
    tweak = _family()
    n = _PARAMS.n
    sk_seed = secret_key[:n]
    sk_prf = secret_key[n : 2 * n]
    pk_seed = secret_key[2 * n : 3 * n]
    pk_root = secret_key[3 * n :]
    body = np.frombuffer(message, dtype=np.uint8)

    randomizer = np.asarray(tweak.prf_msg(sk_prf, opt_rand, body))
    digest = np.asarray(tweak.h_msg(randomizer, pk_seed, pk_root, body))[0]
    md, idx_tree, idx_leaf = _spec_split(bytes(digest))
    md_array = np.frombuffer(md, dtype=np.uint8)

    position = fors.ForsPosition(tree=idx_tree, key_pair=idx_leaf)
    fors_signature = np.asarray(
        fors.sign(tweak, _PARAMS.fors_params, md_array, pk_seed, sk_seed, position)
    )
    fors_key = np.asarray(
        fors.pk_from_sig(
            tweak,
            _PARAMS.fors_params,
            fors_signature[None, ...],
            md_array[None, :],
            pk_seed,
            position,
        )
    )[0]
    hypertree_signature = np.asarray(
        hypertree.sign(
            tweak,
            _PARAMS.hypertree_params,
            fors_key,
            pk_seed,
            sk_seed,
            idx_tree,
            idx_leaf,
        )
    )
    return (
        bytes(randomizer)
        + bytes(fors_signature.reshape(-1))
        + bytes(hypertree_signature.reshape(-1))
    )


class ParameterSetTest(absltest.TestCase):
    def test_every_published_sha2_set_is_present(self) -> None:
        self.assertEqual(
            sorted(slh_dsa.SHA2_PARAMETER_SETS), sorted(name for name, *_ in _TABLE_2)
        )

    def test_the_derived_columns_are_the_published_ones(self) -> None:
        for name, height, m, category, public_key, signature in _TABLE_2:
            with self.subTest(name):
                params = slh_dsa.SHA2_PARAMETER_SETS[name]
                self.assertEqual(params.tree_height, height)
                self.assertEqual(params.m, m)
                self.assertEqual(params.security_category, category)
                self.assertEqual(params.public_key_size, public_key)
                self.assertEqual(params.signature_size, signature)
                # Figures 15 and 16: the private key is the public one plus the
                # two secret seeds.
                self.assertEqual(params.secret_key_size, 2 * public_key)

    def test_the_digest_is_exactly_the_slices_the_split_consumes(self) -> None:
        for name in slh_dsa.SHA2_PARAMETER_SETS:
            with self.subTest(name):
                params = slh_dsa.SHA2_PARAMETER_SETS[name]
                self.assertEqual(
                    params.m,
                    params.md_bytes + params.tree_index_bytes + params.leaf_index_bytes,
                )
                # §9.2: `h + k·a` bits are used, out of a digest rounded up to
                # whole bytes per slice. Fewer would leave an index short.
                self.assertGreaterEqual(8 * params.m, params.h + params.k * params.a)

    def test_the_layers_must_divide_the_height(self) -> None:
        with self.assertRaisesRegex(ValueError, "d must divide h"):
            slh_dsa.SlhDsaParams(n=16, h=63, d=8, a=12, k=14)

    def test_a_size_outside_the_table_has_no_category(self) -> None:
        with self.assertRaisesRegex(ValueError, "n = 20"):
            _ = slh_dsa.SlhDsaParams(n=20, h=4, d=2, a=3, k=4).security_category


class Sha2FactoryTest(absltest.TestCase):
    def test_the_category_1_sets_build(self) -> None:
        for name in ("SLH-DSA-SHA2-128s", "SLH-DSA-SHA2-128f"):
            with self.subTest(name):
                scheme = slh_dsa.sha2(name)
                params = slh_dsa.SHA2_PARAMETER_SETS[name]
                self.assertEqual(scheme.public_key_size, params.public_key_size)
                self.assertEqual(scheme.signature_max_size, params.signature_size)
                # §9.2: hedged is the standard's default variant.
                self.assertFalse(scheme.deterministic)

    def test_the_higher_categories_need_a_second_hash(self) -> None:
        for name in slh_dsa.SHA2_PARAMETER_SETS:
            if slh_dsa.SHA2_PARAMETER_SETS[name].security_category == 1:
                continue
            with (
                self.subTest(name),
                self.assertRaisesRegex(NotImplementedError, "SHA-512"),
            ):
                slh_dsa.sha2(name)

    def test_an_unnamed_set_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "not one of"):
            slh_dsa.sha2("SLH-DSA-SHAKE-128s")

    def test_a_family_sized_for_another_set_is_an_error(self) -> None:
        # `m` belongs to the parameter set and the family both, and a mismatch
        # would hash a digest the split then reads past the end of.
        with self.assertRaisesRegex(ValueError, "mismatch silently signs"):
            slh_dsa.SlhDsa(Sha2TweakableHash(Sha256(), n=16, m=30), _PARAMS)

    def test_instances_compare_by_value(self) -> None:
        # The seam's rule: a scheme rides pytree aux, where identity equality does
        # not fail, it silently re-traces for every freshly built instance.
        one = slh_dsa.sha2("SLH-DSA-SHA2-128s")
        same = slh_dsa.sha2("SLH-DSA-SHA2-128s")
        self.assertEqual(one, same)
        self.assertEqual(hash(one), hash(same))
        self.assertNotEqual(one, slh_dsa.sha2("SLH-DSA-SHA2-128f"))
        self.assertNotEqual(one, slh_dsa.sha2("SLH-DSA-SHA2-128s", deterministic=True))


class KeyGenTest(absltest.TestCase):
    def test_the_keys_are_the_figures_over_the_top_layer_root(self) -> None:
        # Algorithm 18 transcribed: lines 1 to 3 are `xmss_node` over the top
        # layer's only tree, and line 4's two tuples are Figures 15 and 16.
        n = _PARAMS.n
        sk_seed, sk_prf, pk_seed = _SEED[:n], _SEED[n : 2 * n], _SEED[2 * n :]
        root = bytes(
            np.asarray(
                xmss.root(
                    _family(),
                    _PARAMS.wots_params,
                    pk_seed,
                    sk_seed,
                    tree.TreePosition(layer=_PARAMS.d - 1, tree=0),
                    _PARAMS.tree_height,
                )
            )
        )
        public, secret = _scheme().keygen(_SEED)
        self.assertEqual(bytes(np.asarray(public)), bytes(pk_seed) + root)
        self.assertEqual(
            bytes(np.asarray(secret)),
            bytes(sk_seed) + bytes(sk_prf) + bytes(pk_seed) + root,
        )

    def test_the_root_is_the_top_layer_and_not_the_bottom(self) -> None:
        # `ADRS.setLayerAddress(d − 1)` on line 2. A key built over layer 0's tree
        # 0 would sign and verify against itself forever.
        n = _PARAMS.n
        bottom = bytes(
            np.asarray(
                xmss.root(
                    _family(),
                    _PARAMS.wots_params,
                    _SEED[2 * n :],
                    _SEED[:n],
                    tree.TreePosition(layer=0, tree=0),
                    _PARAMS.tree_height,
                )
            )
        )
        public, _ = _scheme().keygen(_SEED)
        self.assertNotEqual(bytes(np.asarray(public))[n:], bottom)

    def test_keygen_is_deterministic_in_the_seed(self) -> None:
        first, _ = _scheme().keygen(_SEED)
        again, _ = _scheme().keygen(_SEED)
        other, _ = _scheme().keygen(_OTHER_SEED)
        self.assertEqual(bytes(np.asarray(first)), bytes(np.asarray(again)))
        self.assertNotEqual(bytes(np.asarray(first)), bytes(np.asarray(other)))

    def test_a_wrong_sized_seed_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "keygen takes"):
            _scheme().keygen(_SEED[:-1])


class SignTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.scheme = _scheme()
        self.public_key, self.secret_key = (
            np.asarray(part) for part in self.scheme.keygen(_SEED)
        )

    def test_the_signature_is_algorithm_19_line_by_line(self) -> None:
        got = bytes(np.asarray(self.scheme.sign(self.secret_key, _MESSAGE)))
        self.assertLen(got, _PARAMS.signature_size)
        self.assertEqual(
            got,
            _spec_sign(
                self.secret_key,
                _pure_message(bytes(_MESSAGE)),
                # Line 2's deterministic substitution.
                self.secret_key[2 * _PARAMS.n : 3 * _PARAMS.n],
            ),
        )

    def test_the_context_is_part_of_what_gets_signed(self) -> None:
        got = bytes(
            np.asarray(self.scheme.sign(self.secret_key, _MESSAGE, context=_CONTEXT))
        )
        self.assertEqual(
            got,
            _spec_sign(
                self.secret_key,
                _pure_message(bytes(_MESSAGE), bytes(_CONTEXT)),
                self.secret_key[2 * _PARAMS.n : 3 * _PARAMS.n],
            ),
        )
        self.assertNotEqual(
            got, bytes(np.asarray(self.scheme.sign(self.secret_key, _MESSAGE)))
        )

    def test_signing_is_reproducible(self) -> None:
        first = bytes(np.asarray(self.scheme.sign(self.secret_key, _MESSAGE)))
        again = bytes(np.asarray(self.scheme.sign(self.secret_key, _MESSAGE)))
        self.assertEqual(first, again)

    def test_the_hedged_variant_signs_with_the_randomness_it_is_given(self) -> None:
        hedged = _scheme(deterministic=False)
        got = bytes(
            np.asarray(hedged.sign(self.secret_key, _MESSAGE, randomness=_ADDRND))
        )
        self.assertEqual(
            got, _spec_sign(self.secret_key, _pure_message(bytes(_MESSAGE)), _ADDRND)
        )
        # And it is a different signature from the deterministic one, which is the
        # whole of what line 2 changes.
        self.assertNotEqual(
            got, bytes(np.asarray(self.scheme.sign(self.secret_key, _MESSAGE)))
        )

    def test_a_hedged_instance_will_not_draw_its_own_randomness(self) -> None:
        # Falling back to `PK.seed` would emit the deterministic signature, which
        # verifies — so nothing downstream would report the lost hedging.
        with self.assertRaisesRegex(ValueError, "hedged instance"):
            _scheme(deterministic=False).sign(self.secret_key, _MESSAGE)

    def test_a_wrong_sized_randomizer_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "addrnd"):
            _scheme(deterministic=False).sign(
                self.secret_key, _MESSAGE, randomness=_ADDRND[:-1]
            )

    def test_a_wrong_sized_secret_key_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "secret key is"):
            self.scheme.sign(self.secret_key[:-1], _MESSAGE)

    def test_a_context_longer_than_its_length_byte_is_an_error(self) -> None:
        longest = np.zeros(MAX_CONTEXT_SIZE, dtype=np.uint8)
        self.scheme.sign(self.secret_key, _MESSAGE, context=longest)
        with self.assertRaisesRegex(ValueError, "context string is at most"):
            self.scheme.sign(
                self.secret_key, _MESSAGE, context=np.zeros(256, dtype=np.uint8)
            )


class DigestSplitTest(absltest.TestCase):
    """The split against the standard's own slice arithmetic, over many digests.

    A slice off by one byte signs the right digest under the wrong FORS key, which
    is self-consistent and rejected by every published vector — so the offsets are
    checked against the expressions rather than against a round trip.
    """

    def test_the_split_agrees_with_the_slice_expressions(self) -> None:
        tweak = _family()
        scheme = _scheme()
        _, secret_key = (np.asarray(part) for part in scheme.keygen(_SEED))
        n = _PARAMS.n
        sk_prf = secret_key[n : 2 * n]
        pk_seed = secret_key[2 * n : 3 * n]
        pk_root = secret_key[3 * n :]
        for index in range(24):
            body = np.frombuffer(
                _pure_message(bytes(_MESSAGE) + index.to_bytes(2, "big")),
                dtype=np.uint8,
            )
            randomizer = np.asarray(tweak.prf_msg(sk_prf, pk_seed, body))
            digest = bytes(
                np.asarray(tweak.h_msg(randomizer, pk_seed, pk_root, body))[0]
            )
            self.assertLen(digest, _PARAMS.m)
            md, idx_tree, idx_leaf = _spec_split(digest)
            got_md, got_trees, got_leaves = scheme._split_digest(
                randomizer, pk_seed, pk_root, body
            )
            with self.subTest(index):
                self.assertEqual(bytes(np.asarray(got_md)[0]), md)
                self.assertEqual(
                    int.from_bytes(bytes(np.asarray(got_trees)[0]), "big"), idx_tree
                )
                self.assertEqual(got_leaves.tolist(), [idx_leaf])
                # The reduction is what keeps a byte-rounded slice inside the
                # hypertree it indexes into.
                self.assertLess(idx_tree, 1 << (_PARAMS.h - _PARAMS.tree_height))
                self.assertLess(idx_leaf, 1 << _PARAMS.tree_height)

    def test_the_indices_come_back_one_per_signature(self) -> None:
        # An index per signature, read out of the batch at once rather than a row
        # at a time — a Python loop over signatures is what the batch axis removes.
        # The tree index is a row of bytes and the leaf index a column, which is
        # the split `bytestring` exists for: 54 to 64 bits do not fit a lane and
        # `h'` bits do.
        scheme = _scheme()
        public_key, secret_key = (np.asarray(part) for part in scheme.keygen(_SEED))
        batch = 3
        messages = np.stack(
            [
                np.frombuffer(
                    _pure_message(bytes(_MESSAGE) + index.to_bytes(2, "big")),
                    dtype=np.uint8,
                )
                for index in range(batch)
            ]
        )
        randomizers = np.stack(
            [
                np.asarray(scheme.tweak.prf_msg(secret_key[16:32], public_key[:16], m))
                for m in messages
            ]
        )
        _, trees, leaves = scheme._split_digest(
            randomizers,
            np.stack([public_key[:16]] * batch),
            np.stack([public_key[16:]] * batch),
            messages,
        )
        self.assertEqual(trees.shape, (batch, _PARAMS.tree_index_bytes))
        self.assertEqual(trees.dtype, np.uint8)
        self.assertEqual(leaves.shape, (batch,))


class VerifyTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.scheme = _scheme()
        self.public_key, self.secret_key = (
            np.asarray(part) for part in self.scheme.keygen(_SEED)
        )
        self.signature = np.asarray(self.scheme.sign(self.secret_key, _MESSAGE))

    def _verify(
        self,
        public_keys: np.ndarray,
        messages: np.ndarray,
        signatures: np.ndarray,
        context: np.ndarray | None = None,
    ) -> list[bool]:
        return [
            bool(verdict)
            for verdict in np.asarray(
                self.scheme.verify(public_keys, messages, signatures, context=context)
            )
        ]

    def _verify_one(
        self, message: np.ndarray, signature: np.ndarray, **kwargs: np.ndarray
    ) -> bool:
        return self._verify(
            self.public_key[None, :], message[None, :], signature[None, :], **kwargs
        )[0]

    def test_a_signature_verifies(self) -> None:
        self.assertTrue(self._verify_one(_MESSAGE, self.signature))

    def test_every_region_of_the_signature_is_load_bearing(self) -> None:
        # R, the FORS signature and the hypertree signature: a verifier that
        # ignored any of the three would still accept every honest signature.
        n = _PARAMS.n
        regions = {
            "R": 0,
            "SIG_FORS": n,
            "SIG_HT": n + _PARAMS.k * (1 + _PARAMS.a) * n,
        }
        for name, offset in regions.items():
            tampered = self.signature.copy()
            tampered[offset] ^= 1
            with self.subTest(name):
                self.assertFalse(self._verify_one(_MESSAGE, tampered))

    def test_a_tampered_message_is_rejected(self) -> None:
        other = _MESSAGE.copy()
        other[0] ^= 1
        self.assertFalse(self._verify_one(other, self.signature))

    def test_a_tampered_public_key_is_rejected(self) -> None:
        # Both halves: `PK.seed` tweaks every hash below, and `PK.root` is what
        # the hypertree walk lands on.
        for offset in (0, _PARAMS.n):
            tampered = self.public_key.copy()
            tampered[offset] ^= 1
            with self.subTest(offset):
                self.assertFalse(
                    self._verify(
                        tampered[None, :],
                        _MESSAGE[None, :],
                        self.signature[None, :],
                    )[0]
                )

    def test_the_hash_call_count_does_not_depend_on_the_batch_size(self) -> None:
        # The batch-parallel claim, stated where it can fail. A Python loop over
        # signatures anywhere in the path — building one position per entry,
        # reading one index per entry — shows up here as a call count that grows
        # with `B`. It does not: `B` only widens the rows each call carries.
        counted = _CountingHash()
        scheme = slh_dsa.SlhDsa(
            # `block_size` is explicit because hash-frx's table is keyed on the
            # row's type name, and this stands in for `Sha256` under its own.
            Sha2TweakableHash(counted, n=_PARAMS.n, m=_PARAMS.m, block_size=64),
            _PARAMS,
            deterministic=True,
        )
        public_key, secret_key = (np.asarray(part) for part in scheme.keygen(_SEED))
        signature = np.asarray(scheme.sign(secret_key, _MESSAGE))
        calls = {}
        for batch in (1, 2, 5):
            counted.calls = 0
            verdicts = np.asarray(
                scheme.verify(
                    np.stack([public_key] * batch),
                    np.stack([_MESSAGE] * batch),
                    np.stack([signature] * batch),
                )
            )
            self.assertEqual(verdicts.tolist(), [True] * batch)
            calls[batch] = counted.calls
        self.assertEqual(calls[2], calls[1], calls)
        self.assertEqual(calls[5], calls[1], calls)

    def test_another_context_is_rejected(self) -> None:
        signed = np.asarray(
            self.scheme.sign(self.secret_key, _MESSAGE, context=_CONTEXT)
        )
        self.assertTrue(self._verify_one(_MESSAGE, signed, context=_CONTEXT))
        self.assertFalse(self._verify_one(_MESSAGE, signed))
        self.assertFalse(
            self._verify_one(_MESSAGE, signed, context=_CONTEXT[:-1].copy())
        )

    def test_the_batch_agrees_with_the_one_at_a_time_form(self) -> None:
        # The claim `verify` makes beyond the standard. The batch spans public
        # keys, messages and verdicts at once, because that is what a verifier
        # holds — and the digest picks the FORS key, so no two entries share a
        # position either.
        other_scheme = _scheme()
        other_public, other_secret = (
            np.asarray(part) for part in other_scheme.keygen(_OTHER_SEED)
        )
        messages = [
            _MESSAGE,
            np.frombuffer(b"a different content, same length", dtype=np.uint8)[
                : len(_MESSAGE)
            ],
            _MESSAGE,
        ]
        keys = [self.public_key, other_public, other_public]
        signatures = [
            self.signature,
            np.asarray(other_scheme.sign(other_secret, messages[1])),
            # Signed under the other key, but claimed against a message it did
            # not sign: one entry that must fail while its neighbours pass.
            np.asarray(other_scheme.sign(other_secret, messages[1])),
        ]
        batched = self._verify(np.stack(keys), np.stack(messages), np.stack(signatures))
        one_at_a_time = [
            self._verify(key[None, :], message[None, :], signature[None, :])[0]
            for key, message, signature in zip(keys, messages, signatures, strict=True)
        ]
        self.assertEqual(batched, [True, True, False])
        self.assertEqual(batched, one_at_a_time)

    def test_a_wrong_length_signature_is_rejected_rather_than_refused(self) -> None:
        # Algorithm 20 lines 1 to 3: a signature of the wrong length is a false
        # verdict, not an error. The published vectors carry signatures a byte
        # short and a byte long and expect each to be rejected, so raising here
        # would turn twelve of them into crashes.
        for signature in (self.signature[:-1], np.append(self.signature, 0)):
            with self.subTest(len(signature)):
                self.assertEqual(
                    self._verify(
                        self.public_key[None, :],
                        _MESSAGE[None, :],
                        signature[None, :],
                    ),
                    [False],
                )
        # The whole batch, since a batch carries one length: every entry's
        # signature is the wrong length, so every entry is rejected.
        self.assertEqual(
            self._verify(
                np.stack([self.public_key] * 2),
                np.stack([_MESSAGE] * 2),
                np.stack([self.signature[:-1]] * 2),
            ),
            [False, False],
        )

    def test_a_misshapen_batch_is_an_error(self) -> None:
        # A wrong rank or a batch that does not line up is a caller mistake rather
        # than a verdict the standard defines, so those still raise.
        with self.assertRaisesRegex(ValueError, "one signature per public key"):
            self._verify(
                self.public_key[None, :],
                _MESSAGE[None, :],
                np.stack([self.signature, self.signature]),
            )
        with self.assertRaisesRegex(ValueError, "public key batch is"):
            self._verify(
                self.public_key[None, :-1],
                _MESSAGE[None, :],
                self.signature[None, :],
            )
        with self.assertRaisesRegex(ValueError, "one message per public key"):
            self._verify(
                self.public_key[None, :],
                np.stack([_MESSAGE, _MESSAGE]),
                self.signature[None, :],
            )
        # A single verification is `B = 1`, not an unbatched call: a bare message
        # would otherwise be read as a batch of its own bytes.
        with self.assertRaisesRegex(ValueError, "one message per public key"):
            self._verify(self.public_key[None, :], _MESSAGE, self.signature[None, :])


class InternalInterfaceTest(absltest.TestCase):
    """§9's interface signs the message as given, and the external one wraps it.

    Which makes the relationship between them testable rather than asserted: the
    external operation is exactly the internal one over `M'`, and the two are
    therefore different objects over the same content — the property §10.2.1's
    domain separator exists to guarantee.
    """

    def setUp(self) -> None:
        super().setUp()
        self.scheme = _scheme()
        self.public_key, self.secret_key = (
            np.asarray(part) for part in self.scheme.keygen(_SEED)
        )

    def test_the_external_operation_is_the_internal_one_over_m_prime(self) -> None:
        wrapped = np.frombuffer(
            _pure_message(bytes(_MESSAGE), bytes(_CONTEXT)), dtype=np.uint8
        )
        self.assertEqual(
            bytes(
                np.asarray(
                    self.scheme.sign(self.secret_key, _MESSAGE, context=_CONTEXT)
                )
            ),
            bytes(np.asarray(self.scheme.sign_internal(self.secret_key, wrapped))),
        )

    def test_an_internal_signature_round_trips_internally(self) -> None:
        signature = np.asarray(self.scheme.sign_internal(self.secret_key, _MESSAGE))
        self.assertEqual(
            [
                bool(v)
                for v in np.asarray(
                    self.scheme.verify_internal(
                        self.public_key[None, :], _MESSAGE[None, :], signature[None, :]
                    )
                )
            ],
            [True],
        )

    def test_the_interfaces_do_not_cross_verify(self) -> None:
        # An unwrapped message and `M'` are different messages, so neither
        # signature verifies under the other interface.
        internal = np.asarray(self.scheme.sign_internal(self.secret_key, _MESSAGE))
        external = np.asarray(self.scheme.sign(self.secret_key, _MESSAGE))
        self.assertNotEqual(bytes(internal), bytes(external))
        self.assertFalse(
            bool(
                np.asarray(
                    self.scheme.verify(
                        self.public_key[None, :], _MESSAGE[None, :], internal[None, :]
                    )
                )[0]
            )
        )
        self.assertFalse(
            bool(
                np.asarray(
                    self.scheme.verify_internal(
                        self.public_key[None, :], _MESSAGE[None, :], external[None, :]
                    )
                )[0]
            )
        )


class RealParameterSetTest(absltest.TestCase):
    """A round trip at the sets a caller actually gets, not just at small ones.

    The small parameters above exercise the arithmetic; they do not exercise the
    dimensions. A real set is where `idx_tree` needs seven bytes rather than one,
    where the hypertree is 22 layers rather than two, and where the signature has
    a size the standard publishes — so the sizes are asserted against Table 2 and
    the signature is required to verify and to stop verifying when it moves.

    Both category 1 sets, because `s` and `f` sit at opposite ends of the trade:
    seven layers of 512 key pairs against twenty-two of eight.
    """

    def test_each_category_1_set_signs_and_verifies(self) -> None:
        for name in ("SLH-DSA-SHA2-128f", "SLH-DSA-SHA2-128s"):
            with self.subTest(name):
                scheme = slh_dsa.sha2(name, deterministic=True)
                params = slh_dsa.SHA2_PARAMETER_SETS[name]
                seed = np.frombuffer(
                    bytes((i * 13 + 5) % 256 for i in range(params.seed_size)),
                    dtype=np.uint8,
                )
                public_key, secret_key = (
                    np.asarray(part) for part in scheme.keygen(seed)
                )
                self.assertLen(public_key, params.public_key_size)
                self.assertLen(secret_key, params.secret_key_size)

                signature = np.asarray(scheme.sign(secret_key, _MESSAGE))
                self.assertLen(signature, params.signature_size)

                tampered = signature.copy()
                tampered[-1] ^= 1
                verdicts = np.asarray(
                    scheme.verify(
                        np.stack([public_key, public_key]),
                        np.stack([_MESSAGE, _MESSAGE]),
                        np.stack([signature, tampered]),
                    )
                )
                self.assertEqual([bool(v) for v in verdicts], [True, False])


class PreHashTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.scheme = _scheme()
        self.pre_hash = slh_dsa.sha256_pre_hash()
        self.public_key, self.secret_key = (
            np.asarray(part) for part in self.scheme.keygen(_SEED)
        )

    def test_the_oid_is_the_published_der_encoding(self) -> None:
        # Algorithm 23 line 10: the DER encoding of 2.16.840.1.101.3.4.2.1, tag
        # and length included.
        self.assertEqual(self.pre_hash.oid.hex().upper(), "0609608648016503040201")

    def test_the_pre_hash_signature_is_algorithm_23_line_24(self) -> None:
        got = bytes(
            np.asarray(self.scheme.hash_sign(self.secret_key, _MESSAGE, self.pre_hash))
        )
        self.assertEqual(
            got,
            _spec_sign(
                self.secret_key,
                _prehash_message(bytes(_MESSAGE)),
                self.secret_key[2 * _PARAMS.n : 3 * _PARAMS.n],
            ),
        )

    def test_a_pre_hash_signature_round_trips_with_its_context(self) -> None:
        signed = np.asarray(
            self.scheme.hash_sign(
                self.secret_key, _MESSAGE, self.pre_hash, context=_CONTEXT
            )
        )
        self.assertEqual(
            [
                bool(v)
                for v in np.asarray(
                    self.scheme.hash_verify(
                        self.public_key[None, :],
                        _MESSAGE[None, :],
                        signed[None, :],
                        self.pre_hash,
                        context=_CONTEXT,
                    )
                )
            ],
            [True],
        )

    def test_pure_and_pre_hash_signatures_do_not_cross_verify(self) -> None:
        # What the domain separator of §10.2.1 exists for: the two are signatures
        # over different messages, so neither verifies as the other.
        pure = np.asarray(self.scheme.sign(self.secret_key, _MESSAGE))
        prehashed = np.asarray(
            self.scheme.hash_sign(self.secret_key, _MESSAGE, self.pre_hash)
        )
        self.assertNotEqual(bytes(pure), bytes(prehashed))
        self.assertFalse(
            bool(
                np.asarray(
                    self.scheme.verify(
                        self.public_key[None, :], _MESSAGE[None, :], prehashed[None, :]
                    )
                )[0]
            )
        )
        self.assertFalse(
            bool(
                np.asarray(
                    self.scheme.hash_verify(
                        self.public_key[None, :],
                        _MESSAGE[None, :],
                        pure[None, :],
                        self.pre_hash,
                    )
                )[0]
            )
        )


if __name__ == "__main__":
    absltest.main()
