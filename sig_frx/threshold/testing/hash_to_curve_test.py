# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""`secp256k1_XMD:SHA-256_SSWU_RO_` against RFC 9380's own vectors (App. J.8.1).

Authority level 1 — the standard publishes these itself, so nothing here is
derived from a reference implementation and none of `testing.md`'s
provenance-for-an-implementation rules apply. They are transcribed constants:
there is no file to fetch, so the `http_file` rule does not reach them either.

**Provenance.** RFC 9380, Appendix J.8.1, every case it lists:

    suite = secp256k1_XMD:SHA-256_SSWU_RO_
    dst   = QUUX-V01-CS02-with-secp256k1_XMD:SHA-256_SSWU_RO_

The DST is the appendix's own, not a value chosen here — it is an input the
vectors were generated under, so changing it invalidates every expected value
below. A suite used in anger picks its own, which `test_domain_separation`
is what pins.

**Each vector is pinned at four depths, not one.** The appendix publishes
`u[0]`, `u[1]`, `Q0`, `Q1` *and* `P`, which is exactly what `testing.md` asks
for — a mismatch in `P` alone says only that something is wrong, while a
mismatch that reaches `u` and stops names `hash_to_field`, one that starts at
`Q0` names the SSWU map or the isogeny, and one that appears first at `P`
names the addition. That decomposition is the reason this suite is worth
having: the map has no round trip to check it against, so a failure that
cannot be localized would have to be bisected by hand.

The structural cases below cover what a vector cannot see. The published
messages are ordinary inputs, so none of them reaches the two exceptional
branches the specification requires — and a branch that no case enters is a
branch nobody has run.
"""

from __future__ import annotations

from typing import NamedTuple

from absl.testing import absltest, parameterized

from sig_frx.classical import secp
from sig_frx.threshold import hash_to_curve

# RFC 9380 App. J.8.1's `dst`. Not this repo's choice — see the module
# docstring.
_DST = b"QUUX-V01-CS02-with-secp256k1_XMD:SHA-256_SSWU_RO_"

_CURVE = secp.SECP256K1


class _Vector(NamedTuple):
    name: str
    message: bytes
    u: tuple[int, int]
    q0: tuple[int, int]
    q1: tuple[int, int]
    point: tuple[int, int]


_VECTORS = (
    _Vector(
        name="empty",
        message=b"",
        u=(
            0x6B0F9910DD2BA71C78F2EE9F04D73B5F4C5F7FC773A701ABEA1E573CAB002FB3,
            0x1AE6C212E08FE1A5937F6202F929A2CC8EF4EE5B9782DB68B0D5799FD8F09E16,
        ),
        q0=(
            0x74519EF88B32B425A095E4EBCC84D81B64E9E2C2675340A720BB1A1857B99F1E,
            0xC174FA322AB7C192E11748BEED45B508E9FDB1CE046DEE9C2CD3A2A86B410936,
        ),
        q1=(
            0x44548ADB1B399263DED3510554D28B4BEAD34B8CF9A37B4BD0BD2BA4DB87AE63,
            0x96EB8E2FAF05E368EFE5957C6167001760233E6DD2487516B46AE725C4CCE0C6,
        ),
        point=(
            0xC1CAE290E291AEE617EBAEF1BE6D73861479C48B841EABA9B7B5852DDFEB1346,
            0x64FA678E07AE116126F08B022A94AF6DE15985C996C3A91B64C406A960E51067,
        ),
    ),
    _Vector(
        name="abc",
        message=b"abc",
        u=(
            0x128AAB5D3679A1F7601E3BDF94CED1F43E491F544767E18A4873F397B08A2B61,
            0x5897B65DA3B595A813D0FDCC75C895DC531BE76A03518B044DAAA0F2E4689E00,
        ),
        q0=(
            0x07DD9432D426845FB19857D1B3A91722436604CCBBBADAD8523B8FC38A5322D7,
            0x604588EF5138CFFE3277BBD590B8550BCBE0E523BBAF1BED4014A467122EB33F,
        ),
        q1=(
            0xE9EF9794D15D4E77DDE751E06C182782046B8DAC05F8491EB88764FC65321F78,
            0xCB07CE53670D5314BF236EE2C871455C562DD76314AA41F012919FE8E7F717B3,
        ),
        point=(
            0x3377E01EAB42DB296B512293120C6CEE72B6ECF9F9205760BD9FF11FB3CB2C4B,
            0x7F95890F33EFEBD1044D382A01B1BEE0900FB6116F94688D487C6C7B9C8371F6,
        ),
    ),
    _Vector(
        name="abcdef0123456789",
        message=b"abcdef0123456789",
        u=(
            0xEA67A7C02F2CD5D8B87715C169D055A22520F74DAEB080E6180958380E2F98B9,
            0x7434D0D1A500D38380D1F9615C021857AC8D546925F5F2355319D823A478DA18,
        ),
        q0=(
            0x576D43AB0260275ADF11AF990D130A5752704F79478628761720808862544B5D,
            0x643C4A7FB68AE6CFF55EDD66B809087434BBAFF0C07F3F9EC4D49BB3C16623C3,
        ),
        q1=(
            0xF89D6D261A5E00FE5CF45E827B507643E67C2A947A20FD9AD71039F8B0E29FF8,
            0xB33855E0CC34A9176EAD91C6C3ACB1AACB1CE936D563BC1CEE1DCFFC806CAF57,
        ),
        point=(
            0xBAC54083F293F1FE08E4A70137260AA90783A5CB84D3F35848B324D0674B0E3A,
            0x4436476085D4C3C4508B60FCF4389C40176ADCE756B398BDEE27BCA19758D828,
        ),
    ),
    _Vector(
        name="q128",
        message=b"q128_" + b"q" * 128,
        u=(
            0xEDA89A5024FAC0A8207A87E8CC4E85AA3BCE10745D501A30DEB87341B05BCDF5,
            0xDFE78CD116818FC2C16F3837FEDBE2639FAB012C407EAC9DFE9245BF650AC51D,
        ),
        q0=(
            0x9C91513CCFE9520C9C645588DFF5F9B4E92EAF6AD4AB6F1CD720D192EB58247A,
            0xC7371DCD0134412F221E386F8D68F49E7FA36F9037676E163D4A063FBF8A1FB8,
        ),
        q1=(
            0x10FEE3284D7BE6BD5912503B972FC52BF4761F47141A0015F1C6AE36848D869B,
            0x0B163D9B4BF21887364332BE3EFF3C870FA053CF508732900FC69A6EB0E1B672,
        ),
        point=(
            0xE2167BC785333A37AA562F021F1E881DEFB853839BABF52A7F72B102E41890E9,
            0xF2401DD95CC35867FFED4F367CD564763719FBC6A53E969FB8496A1E6685D873,
        ),
    ),
    _Vector(
        name="a512",
        message=b"a512_" + b"a" * 512,
        u=(
            0x8D862E7E7E23D7843FE16D811D46D7E6480127A6B78838C277BCA17DF6900E9F,
            0x68071D2530F040F081BA818D3C7188A94C900586761E9115EFA47AE9BD847938,
        ),
        q0=(
            0xB32B0AB55977B936F1E93FDC68CEC775E13245E161DBFE556BBB1F72799B4181,
            0x2F5317098360B722F132D7156A94822641B615C91F8663BE69169870A12AF9E8,
        ),
        q1=(
            0x148F98780F19388B9FA93E7DC567B5A673E5FCA7079CD9CDAFD71982EC4C5E12,
            0x3989645D83A433BC0C001F3DAC29AF861F33A6FD1E04F4B36873F5BFF497298A,
        ),
        point=(
            0xE3C8D35AAAF0B9B647E88A0A0A7EE5D5BED5AD38238152E4E6FD8C1F8CB7C998,
            0x8446EEB6181BF12F56A9D24E262221CC2F0C4725C7E3803024B5888EE5823AA6,
        ),
    ),
)

# The decorator argument, named once — `frost_test` hoists its suite list the
# same way rather than repeating the generator at every method.
_PARAMS = tuple((vector.name, vector) for vector in _VECTORS)


class HashToCurveTest(parameterized.TestCase):
    @parameterized.named_parameters(*_PARAMS)
    def test_hash_to_field(self, vector: _Vector) -> None:
        """`u[0]` and `u[1]`, the first depth the appendix publishes."""
        self.assertEqual(
            tuple(hash_to_curve.hash_to_field(vector.message, _DST)), vector.u
        )

    @parameterized.named_parameters(*_PARAMS)
    def test_map_to_curve(self, vector: _Vector) -> None:
        """`Q0` and `Q1` — the SSWU map and the isogeny, before the addition.

        Two calls, checked separately: a map that ignored its argument would
        return the same point twice and still sum to something.
        """
        self.assertEqual(hash_to_curve.map_to_curve(vector.u[0]), vector.q0)
        self.assertEqual(hash_to_curve.map_to_curve(vector.u[1]), vector.q1)

    @parameterized.named_parameters(*_PARAMS)
    def test_hash_to_curve(self, vector: _Vector) -> None:
        """`P`, the whole encoding end to end."""
        self.assertEqual(
            hash_to_curve.hash_to_curve(vector.message, _DST), vector.point
        )

    @parameterized.named_parameters(*_PARAMS)
    def test_published_points_are_on_the_curve(self, vector: _Vector) -> None:
        """Every published point satisfies the curve equation.

        Redundant against the coordinate comparison above and kept anyway:
        it is what proves the transcription landed on secp256k1 rather than
        merely landing consistently. A digit lost from any of these constants
        would fail here as well, and this is the assertion that says why.
        """
        for name, (x, y) in (
            ("Q0", vector.q0),
            ("Q1", vector.q1),
            ("P", vector.point),
        ):
            with self.subTest(name):
                self.assertTrue(secp.on_curve(_CURVE, x, y))


class StructuralTest(absltest.TestCase):
    def test_domain_separation(self) -> None:
        """A different DST is a different oracle.

        The DST is hashed into every expansion, so this fails only if the
        parameter is being dropped somewhere — which a vector run under one
        DST cannot detect.
        """
        under_rfc_dst = hash_to_curve.hash_to_curve(b"abc", _DST)
        under_other_dst = hash_to_curve.hash_to_curve(b"abc", _DST + b"-other")
        self.assertNotEqual(under_rfc_dst, under_other_dst)

    def test_swu_exceptional_input(self) -> None:
        """`u = 0` takes §6.6.2's exceptional branch, and still lands on `E`.

        `Z^2 * u^4 + Z * u^2` vanishes at `u = 0`, so `inv0` returns zero and
        the map must substitute `x1 = B / (Z * A)`. None of the published
        messages produces a zero `u`, so without this the branch is unreached
        — and the condition on `Z` that makes `g(x1)` square there is exactly
        the kind of precondition that is silently wrong until something runs
        it.
        """
        x, y = hash_to_curve.map_to_curve(0)
        self.assertTrue(secp.on_curve(_CURVE, x, y))
        self.assertNotEqual((x, y), secp.AFFINE_IDENTITY)

    def test_isogeny_kernel_maps_to_identity(self) -> None:
        """The isogeny's exceptional input returns the identity, not garbage.

        App. E.1's `y_den` is a cubic in `x'`, and it has exactly one root in
        this field — `x_den`'s discriminant is a non-residue, so the
        x-denominator has none and this is the only way in. RFC 9380 requires
        the identity for such inputs; without the check the expression would
        divide by zero, which `inv0` answers with `0` rather than raising, and
        the map would return a plausible-looking wrong point.
        """
        kernel_x = 0x89291C84DE3E11F1041DA6957255EED5FC964A4DF050DF221D6AD4CE6AB9C5A5
        self.assertEqual(hash_to_curve._iso_map(kernel_x, 1), secp.AFFINE_IDENTITY)


if __name__ == "__main__":
    absltest.main()
