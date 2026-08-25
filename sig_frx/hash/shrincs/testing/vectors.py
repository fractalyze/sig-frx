# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""What the SHRINCS reference implementation computes on the stateless path.

SHRINCS publishes no test vectors. Its specification says so in as many words —
"comprehensive test vectors covering every algorithm are still TODO; they are
required to advance this proposal from Draft to Complete" — and there is no
validation program for a scheme that is not standardized. So the gate is the
reference implementation the specification points at, which that document names
as normative: "we provide a naive, highly inefficient, and non-constant-time pure
Python 3 implementation of the SHRINCS algorithms in `impl/shrincs.py`, which is
the normative source for the algorithms specified in this document."

    SHRINCS/shrincs-bip at 9a7d8a4fe543ed9462c20d0a2f78ca81cef29e70

The values below are transcribed constants rather than a fetched file, because
there is no published file to fetch. Each was produced by importing that
`impl/shrincs.py` at that commit and calling, per case:

    sk, pk = shrincs_keygen(seed, bytes([FXMSS_SHAPE_BALANCED, 4]))
    sig = shrincs_sign(message, context, sk, None, opt_rand)

`None` in place of the state counter is what selects the stateless path;
`opt_rand` is passed rather than drawn, so a case reproduces. The tree shape
reaches only the stateful half of key generation, and it is recorded because
`sf_root` — which the stateless path binds into every message — depends on it.

**The intermediates are pinned, not only the signature.** A digest of a final
artifact says that something is wrong and nothing about where, and this path runs
a message digest, a FORS reconstruction and a five-layer hypertree walk before it
produces one. Each case therefore carries what the reference computes between
them, obtained from the same module:

    md, tree_index, leaf_index = slh_dsa_digest_message(R, pk_seed, sl_root, bound)
    fors_public_key = fors_pubkey_from_sig(sig[17:17 + FORS_SIGNATURE_SIZE], md, ...)

where `R` is the signature's first 16 bytes after the indicator, and `bound` is
`toByte(0, 1) ‖ toByte(|ctx|, 1) ‖ ctx ‖ sf_root ‖ message` — the stateless
component's message, with FIPS 205's context encoding around it.

**A pinned draft moves.** That commit is a Draft-stage specification with a
security proof still outstanding, and the numbers in it have already changed once
in public. Re-pin deliberately and re-run these, rather than tracking a branch.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StatelessVectors:
    """One stateless case, and everything the reference computes reaching it."""

    label: str
    seed: bytes  # the 48 bytes key generation takes
    message: bytes
    context: bytes
    public_key: bytes  # `pk_seed ‖ sl_root ‖ sf_root`
    signature: bytes  # the indicator byte, then the SLH-DSA signature
    # The public key's three parts, so a failure to split it is its own failure.
    pk_seed: bytes
    sl_root: bytes
    sf_root: bytes
    # What the verifier recomputes on its way to `sl_root`.
    randomizer: bytes  # `R`
    fors_digest: bytes  # `md`, 17 bytes
    tree_index: int
    leaf_index: int
    fors_public_key: bytes


REFERENCE: tuple[StatelessVectors, ...] = (
    StatelessVectors(
        label="empty_context",
        seed=bytes.fromhex(
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
            "202122232425262728292a2b2c2d2e2f"
        ),
        message=bytes.fromhex(
            "7369672d66727820534852494e43532073746174656c65737320766563746f72"
        ),
        context=b"",
        public_key=bytes.fromhex(
            "202122232425262728292a2b2c2d2e2f017bd5a63a0ec4689a6e7ab02a58182b"
            "30bfb08ba74e0f67ad0369ba45432317"
        ),
        signature=bytes.fromhex(
            "fff01bc578ba9b98f7fc1af654b1aeb7ea22dcf4797a355d59cd1482eb2452ca"
            "9b41d4ff538bc0f01ae745068aa75a13814eade0e49b7cbf1e7496734772e361"
            "4fc4bb679f3b872022a2aba0b5e374fee9252bcc502f11cf3872883dd7c30882"
            "20735959eb6ccf703819544f6cae91be21ad3fbe5e4087653dff68af0b24cf96"
            "0137186d513f6c7afc18c65d4a5c982d6fe363a8f77597444794102a72b76dd9"
            "c98cca520365f235cd8bed578c4d81d57f9dd6fcd3f1843edbcfbe6f2f5eca6f"
            "437688055f262b2247ac5a7c4bdc6d98986e33ee6cd6bed66a8334f1f445945d"
            "a3e02cd0243232969e3989102b605fc9010135fd7910f819413a49309d31a696"
            "059a71742aa767f61d6b076104328aad575e667527705e14ad30338ddb205d1c"
            "18c953f6ca90ef4e6e6785d26668879bf36c10a6bad59bb4c980187f2e1cbf65"
            "1b89f54fb5fd9f7401afa1da46ffe38c75eff218107d655eee076928c2d3c6d0"
            "ae27e70958a1837889c01b6165625d4a3073438f5abdf25426dd3d60c2387222"
            "f86758eaa032026f87a3063bf04f282a3738ab46a77874fc3ba3de2398a21d06"
            "ebbeaccdf2228b6baedeb1e4bf716a43a1e4978ad8ed02cb09020b52c002c785"
            "400bb2b750ce259b1dbbb86c3fa10a4007dd308582e4cb414d17306297ab1c62"
            "740b7c888d5d8c2149774504f534ceb296e9c4085338250e653a75cd68ef084f"
            "f8e4a168e8ee87cbb2612080bf144146639839a2ecc778592a67dd3f9e1f38eb"
            "93564f72eb2d8ce3475e9b2b049bf909ce6dd364461c95c041c8eed7b3e4c3c7"
            "0ffdcf5ffbfcbe9592af7e3575941c2d5bd28f9f502f51366066785c742d446b"
            "0f04968df3ac450c4509d3056397366a4b2454f829778895e1c67391e48abf0c"
            "e226cc42ef1d391eab4e94febe819394d1464f9cb697f30e072c1f5a5d44a7c7"
            "32719dca9d200104f5d2e65b92e11a388ad9982311b516544b990b1e5630e287"
            "9c4ee2ac36e48a386451641ff229228f2dca8f3de3990e346a5eb62fcd69d681"
            "7f78e513a1e2bbca70138c66b75681ae9aadd673ceba433d6e328c165dbb2cdd"
            "62722f04a0d329a4a99a58b621c65d41da6a37afb88120e8817319e0aa7d698c"
            "69f2355d6d815d36006f23e6e419ff6d965562faab89aa8c181a76c1f258ea58"
            "d081716a1563e05aa548da54d3f74c8684019f9a130430fa6f92d32ffd74bd17"
            "ad656573d865f693792015b42a1de2c968213177ce041ca20f3cf6d9dce6ec5f"
            "620dc17f3cbb297f6f0aff5e7a8bea2b8d49ef0f48c77b1497bc99ed87bfb8f9"
            "31ea185af863ebdba7717b0344b6318293f7b28743dec888fb7e54cf26db5d4a"
            "84c60c413e283d60fa1a2cfda0e6a66ed54cddeca2b2db04af4f7b786154f2a9"
            "0d8ac0368eec85ca17bbba7a38d31f1da50d15c776ba4cc6a32091a9f32dc690"
            "703760932163449b444fb855ed7da69b2829ddbcf4473dc77230cca1251816df"
            "ce065036099a89a059230b2e6fcc789e8c0982997653afcc25134667f66f9439"
            "87ddd649291106c97875412ecc7245d53ec67a66e8e33516f428f92d609b1119"
            "6740509d33b702554ec741d94c7a6d122ff93aaa0f8325e9eb9e9c2c90a539d4"
            "5c51a819262a070ca057303dd73196fff8a2a7c41db73ca325f8f642884854ab"
            "d84a0a478992e293506a343c37801c69d89019c87808f8059e7ba4d4cf46b0ff"
            "b87972f806135f213f7283755fc4078d383f4951ae6d8336af110f5e09a4ed1f"
            "e62251a100b73c34effeb8499ce5eba0e1296df7e957b755491ee0380b6b7565"
            "b0f584a3a25627270bf345b37eb645e5e6917299801cea7dad2623bcf23eae4f"
            "21d3c7bd3e8c8c6084b408258ee2cbfc58a401dca2473d2c2b300b64d47e05c1"
            "7d2ad741ea1f41396fdb4db9553173d9b71092c86960f907da44fc0fb19725c8"
            "2360ae5ef8d41f5070cfd76435284789a7c46aa9923f767a84fab779636d0838"
            "771d76ce5d184f0d956119e22935954c4129aa876f3cbdb35856be07b5f43b01"
            "c5e18edd5123031f6d85ed8302ce1d9e2a28be677f6ead10f67d24b685fff291"
            "7a9d0546050a95272814702cdcebc525b15e43d635040af9715e974f2732488f"
            "e69d26e21f860ec1ab5926402f6843876ae2bfd6df09db0595ecd56e01c4ca1e"
            "fea0fcf511f75474116bf796640a04046686362f55703ef6bcb7c61d0b47324c"
            "e32efc82c4aeadc8b20bff928f223e22c5c55109b82f71c9b0003c9ec4c4e41e"
            "ca28c9bdeff00150c54f69fa7cddd04544575714efab43e9fe5a56739aa8442c"
            "8fa245f86779eaf9daa5ea7866fdcac45d05194f72ef257a67718f4119de31b9"
            "d7aeed58cb879679b2948af58e527059cfbb46e8f30c030044475b2569108094"
            "a5ac4367aa53fa7dcf5b8d44d795dee621379d8249895370412c021b9fe1cb86"
            "d7cc86c967a6506b4f64081b07493bde73a28da81cb87767956e0657362112e8"
            "8a93be17e6ee14e69a6b7da96f4e5e0be776fbded247c451605b5883929893b9"
            "2e50916ca548057ed719706d0e9f2b192c988793a062ae95d1eb5734cc95def8"
            "84cf9d0932147d0f543b00d1081bae7075306048bc5514afe2e1b6fa5837ae11"
            "a719949a972a787d01e940e7fd3a1834d54a72f1ed91d42933d295d67e53ae9c"
            "884ff625bce310e7036e90045b5c266015459b9b6cc7da676e414acc20fea8dc"
            "ef75e93681e0dabf12c421a367e00411abfa9a63bf385d0348a4fc43e66ddc84"
            "59cb2b1cd5811102944deb20668ffc91ff7e56715a77d37aaecb261377631042"
            "55055910adfd9c1f79bf37ed33ec037d48545ff421b4cb15ff7fe4b690094333"
            "696e36c6283f4e6db22766b78372a37932edd9b1152778171ad42ca6b303d55f"
            "fa8629b8ee9ec26b63bc5be009e314155b2b4bef3873270fdce1b3c2ba4fe47f"
            "b85d1f1022bfbc52a645b011874bc7ec3ea1be40de389a449d80847f0826b2fe"
            "e6de7b804e3fe99cfd9a0abee7af47e1ef04ba63de3974150108bdf4bd3cc4c6"
            "61f839522a53853def3afff341f1245d6ff7055623a560e618f8f5201f2205eb"
            "89e4f476b09c197e2d4311231180495ddb3dbd837399132340de647794e709eb"
            "3c503938471949cd0ef412cf145b5d9fecd8fff9cad665e50bee75ee20e53554"
            "9309422ecb72666dfa59b7ed55a188fc08d5d5ca2b89b93cb2611c1733f1b838"
            "e53da5abef11c22d4213c3f3325d4524a77503ad2088a7bf2687d6cdc3f15071"
            "f08dc4b400df0dcd1016fabbe9615fc2a2908ac5f36c9bc3561aeefa493f0fe7"
            "f0a822de8bab417a3fa61ff7f44ee0c03570b62c81a8f79d3ee2bd5206dd1185"
            "ec861a7ae77d02b2b94f468d28e9496866c0f8ca0344f8128d6e1dd9db37ffa7"
            "b57e2fda7bbd00903a321aed6f942c46a32aaa5551ef5d31fcb5079045b84d2e"
            "099cdd7a48b7670d0e25890063459eeb8883a2fa7f508257fc96ac82799c1a15"
            "42e4d5f4b172eff4a26cb74678169bf356c7f98973dc6e6f2f52c04b6c0ce9d4"
            "a52d3b224ee889a33bb04adb74d356bb96dc342459d79199471c0fb7a37b5b75"
            "2bde934a117a0fae4c900a16ab6e6661b2581b8a30331dadc2515448415cfdcd"
            "3a8e86a28021d81beb5c21605a2bcb7cb026fe365530d3804602e57b385558cc"
            "f6a5e16f27f66f89d909d76da849ae9da6abd2befc0f730356b2ea67422a5081"
            "4fe47bf383dfea3fcccfe7fb46c3f5f8c4b50cdb9f48801407975e86b7a3880d"
            "f3db761bd1bdd562bfff9348771dae0f66794caff6901277ba583002aaacffa1"
            "058c92a59b503d034c0accf089e38c1191e37b15c140e6cd7f5a21f222b74aed"
            "4fb40ad43cfaa8386a0dc190e7e8c781510cef352e693e464e5fc5807d770b86"
            "c5793d49cb1af10a762fd0f45e0f9f9d9e875e109d961d972d1564e49d2bb8a2"
            "01844cd7a18e78097ba169feb675d8a4f6f797d0f5c7f0213b38edc8275fb2ad"
            "82fe6253abb6f57336e3c7068ddb4b628e1f517c8353bfae35b1e87fe17c20f8"
            "2817c37ea4dc88b950e57fefedf4ab89a23590ab5360eb2fb86f22628b01cc20"
            "7f8c6aa993d92f751353294b392379a479e598ef48667117415cc20b7e26f238"
            "a1605dc2fda165b7a8a073717389f0a0f6b68bdfb8130bad9a1f822fd918bea7"
            "1b0399420eebaa2f80b7f280a2e87f74e60e7a4b02cd91d1f8a87f8c24b19e82"
            "524c90a2796a5160dec60520f25abe03c3a24076e91e7603ef47473aecb66acf"
            "62cf6f8b6a7d78f10303a11ddba9044a3511755943c9d1c22008c81bd466959f"
            "b079e9a6b226e72ea47baf071c0246e40aecf743fc7e8f28535f1c96fd4726ed"
            "33e8697f37a3adc8a5d56f91ca0c8386945348fa325df766eb665278c9ebd14f"
            "f1ae2c381b7b01e5202ec0853e59745327ad5d4a0700971f23b428ab6ac5f3bc"
            "d80c9c694d48a2865e74478798dcf9ab3815759f4d1789b070e9cfa09074e5f3"
            "e484b029daf73eebf6a363f4fb9ad6eafc997750c416d967270b531d70b07b87"
            "2f459ddda8c63ca55e781cdd12f25896376f6d86a9bff8f833f98db011641834"
            "eede944d9e6589da56dece6d92e9e1a4b00ac887bb5de7dce6db934c417270f0"
            "5d4004bd4d57056faa956ba5c0d3e5febf14d66aaec9623f301217f144f78f30"
            "e35ae81dcd6ec4ee34fcedbe281e674907e4ced78edd266b9232987c95a2842a"
            "722778f4876d22df3939ed3bec61adc2c80fc062165431d4522845837a1ced80"
            "255ec221724315a6b277814fec61c2ed22b0402fc39f5acb2ff9e9d25a2fe5a9"
            "af22516298f0e3185a88b3cd3a8c55256a243c6cfbb6cf3464a321537abc7f49"
            "4418e8029bb3bdc7ec03d39bfcd55f1af834ea2d1071e6636436bf8d0081e9d6"
            "78d257364236508c825bde41188d98f4fc06ba10df2d33a26241ba472aeb6bd7"
            "82f02b0a4cad712d5a800d9cf3cd12bb48c417de814ed4c38b40e2b3d6f8133b"
            "7d3dd737b5071f9751169b2114730d5ac6a353524e343e08972be760eda0dea5"
            "d96554a6dc922e214094cfdc0b68f510303387564a6e452dce2a789c8aefbc4c"
            "cce8837146f8d1ad3d17a6a9bb7f13c295c70a9b7992e7257c0e0e6473e50487"
            "86c6f28a41ca71e0be23e7698706e029e8baf5a273bff6c2334473a561d7c8dd"
            "83ddb125a2dff903966ceb83ed89f0815bbde9d514aeebab33a50648d48f351e"
            "8088aff6714b859c6c58118569b3d53962aacbf1a206a13093e7f6f665d2e046"
            "304f3ad47350bf8ae79e410c99d3815903eaf9b315932bc97216fd7a29ec2ab0"
            "8690cee655d7a8eeec9e70dd561339a839c95a85b901810c6d0ad87442f66047"
            "e6e37c43bdd415e26176f68b3f89e4b767dc4cdefc4dbdea4a2f554f0d3462f8"
            "9a886e854629b6b5787593e88fda33f1c57c39c74068e562906491d359597978"
            "a0ec56b0473ecebde75b55f42326de6e7dfe88437d94dd1b12fcc98e1f714ac0"
            "589e0ff2c519d24704c5b95854e7873888025f5915552028923f6b911dbdfb9e"
            "3491555322e8a47eac30905605e3c20bd7976e8b162efd3ac412f23c9045eec3"
            "3b60f378ce2737363efa6079b5a8be5db17a249001863a16576fe78702c0f166"
            "0b02c6d4756b5d9781d32e63cd10ac9c6f1c7028d773e9f5fdcb0d5cca7fd709"
            "27ef0e46024c01fefab7c80795206cab48fdbaba5bcaeadb0cd544b7228a724c"
            "c359ce3ddb42e8da2eacdd65c73bb2a62e964f7bdb6dba571b0c28b6acf1d751"
            "7c41d2c48609990d63ebefc0609f532e354786f0dc597c219c07554813bff32c"
            "3f7ce173c601dfb8edb4dca5dd87c4625d962a3ac66230124bd6b91a12add44a"
            "d29d9809f5a38dc5b6c5d18940ec26243708ed3985d8a776a53ef03d9213354c"
            "4348cbb7f05d3f064f8f408d466eeba8bf4930a3e243fb4f7d9e36d07bdc69c0"
            "f7de8e4680386436fd1f5cc5b7ab1c72fc8e0d2b2ffff1383ce7f5aa58dd35a3"
            "4d0e9dd0c83168649bdc835d8a88cca3138239f58798c71e8a867e19d2949ad4"
            "7b43968c5b2a12f3a99c6fcdd94fcb860efdb598bf6d7911124b4887e3cc7d2f"
            "9f41d87c6ca0c26cd8a34b9b70f9270be575177a2a6fcf01c23023b83cdfee9d"
            "1767176f27cdc55831bb3a00d9ccaea0669784d39959f5f3244547367275f6b5"
            "85dd4e4324da6295061b9891bb3c8066dcc9688caf05080ea8a1c701489286ea"
            "1008621cd8f2438d2a87dd6ddcb75560b6fa04cb1e170efdffb2000c3f4c43bc"
            "915ec1c110691f61ebde4507143dc252fb1d92332227d8b839394d872d8404cd"
            "653b3545d9ff509d93cde6aaad6655f8c300144e7db3bc4f861e9e1267cc7f49"
            "1a713e729c7d6364f3457a7f558ca0c9dc0d3b7a1c6d17518708d39da2c3fc50"
            "cba7891b887d031d29fafec4cd958026b4cac43357fba486d009c77879e73a7b"
            "4606d41bd7c36f6ab4d8abbcba2e29dfeaf79dd21c72562c60967b7da31e3d0e"
            "96e15d7ea43f4dfcdcf9fbc4fe55bf31510ce5e2e52562f6d3e95ba1ddca3e43"
            "7d5c999440c9a8c1b86991477650095366c2e4b9e5b971b02d9910406420c8bb"
            "3184b14c48f202bdff69f58b8edf0174dc87f19c5c2e59f52dbba5207143a435"
            "52f1e0189dd339d0146abfb18e9574ab5b7c1086e188ee18024869b56e402b8c"
            "517336b89ceb98244ba6d38529e1420f39904153ebdaf14caaf2d02f03566329"
            "4db819970ecc01e4220feb1d320ae4097f762739513e6647185454ff23a2e85f"
            "fe4f4b762a9d2ccef2d268e048cdb13dab52ffaaa035963ece00d22044602d36"
            "c13caaf924494032a30e9e885627f45c4df91122c45d5f56edf030d3ae3c020f"
            "e523ce0ad02c9080ca4634b8ef978b9c449576e5d43432db947e71d7a7fb7666"
            "83f6b664fad9ce77cb21deb75be4b4f59cafe2e1f4737a03249860d5b13f7b1b"
            "e81569e793e36fa7c3754ffd7116b67479bb14a08d3246fd1a5a730fe7c3e356"
            "7832652e181159444011ca8a162b611f964305a194e18e93fcfd92d41521babe"
            "3bb234d715ed6715f91e3c0aa92227959cce7a54ab802bdf66834462b4cab09e"
            "fd8b37fdcd16cdc9f375d587c7156f201e1fc926704b7008b50c282639b80145"
            "1ffca966e2b88927923b7f17adecd030afa8592edff69b628ff35d2cee293769"
            "709062ea1de320c8547bdd87640fe22e0cb954527797edf7769ea43a0c05dc5c"
            "111c5ec6acc4b7f7c6ce23d741c03aa140da560b19b85f74ebf96f71be0b7bb6"
            "88738a02e61688fb6e6785a50eaf108c2c9340f10e164f9a98a3b4c47988f1d3"
            "3609c62d8fde2ba0f1b07855dcd083899aa3abf28932e5947515c51953e9e71d"
            "2123944e024da78df823c40f6ed3503eed9a2005b15f4517f71cdd14dda5c6ba"
            "b8e992d6d29b2bde3cbeafaf23a6a95d0a863e071ea542f00ca2ee263faef0f7"
            "d6fbc81ae02088d593f36ab916d2673513aac6c015a91f7d15679c0512c4b78f"
            "75bf3ec5be775cb20dcce57fcee5fac93be9c8f8dd5f964b6e3cb0367bedb3d3"
            "f0e232091b61fa210cbe60288b80e24366fa1cf3e39d096040686f90f79daa31"
            "c1ad0f6f21a3a28e3befd352a3e2b039c9cd8caecd4cf672a6a694b545b546f6"
            "4531a13729750bdeceb535a1dae1b1257287c886f698129f8f032149b5954dcf"
            "91e885294f60c76ae5a382ac0438f18d3501bae7e26df5e49cd53747ace2e8b3"
            "6287066fd13df10b8663cff4aad2e921edd53982cd5301558848e56ca62eed21"
            "fd2a578feb853707f2c080d8bdc438e04e1e4ba1847706a0c5b58416d898330b"
            "f49133b1304f1df2b7f888be1b6c4d283a37b69bc0737d7cd990dc0d230e7693"
            "f5945b13ffee300df05f7c7296e94d647873b7619cfef514f8a055432668c0e3"
            "da66874f0ccb4bf4cdb4712ac38a9d777f330d69c39ccec1d819d1f5800580ae"
            "c14cd591e826ae1228574a6b105d3908e4cb5ad8b23dca87f3edbdfccb0a1504"
            "b5ac50726a4dfc53c2de09e5c92a1d244b2b18aca935a5aa653f408dafa96179"
            "0669ca112f791003eb4922b4f4f0de00ffcb103f80cbcc9b2c4d92a7cc7ffa59"
            "fbab9564ceb9cbc3cb8e1a1792b0aa3f62ed189cc749849db3a9a707b2bfb5d9"
            "38161bef4cf42b1dd39e3035f225cd6738a21651475756e6020126834ed8559f"
            "3ba6bcdb4ca92e88d2db07368181945b41"
        ),
        pk_seed=bytes.fromhex("202122232425262728292a2b2c2d2e2f"),
        sl_root=bytes.fromhex("017bd5a63a0ec4689a6e7ab02a58182b"),
        sf_root=bytes.fromhex("30bfb08ba74e0f67ad0369ba45432317"),
        randomizer=bytes.fromhex("f01bc578ba9b98f7fc1af654b1aeb7ea"),
        fors_digest=bytes.fromhex("198f3c27cc6a0e547407a2f06232c81da4"),
        tree_index=40525347982,
        leaf_index=251,
        fors_public_key=bytes.fromhex("d27dacee680a3cd6a39ebf8824bb9781"),
    ),
    StatelessVectors(
        label="with_context",
        seed=bytes.fromhex(
            "404142434445464748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5f"
            "606162636465666768696a6b6c6d6e6f"
        ),
        message=bytes.fromhex(
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
            "202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f"
            "404142434445464748494a4b4c4d4e4f"
        ),
        context=bytes.fromhex("7369672d667278"),
        public_key=bytes.fromhex(
            "606162636465666768696a6b6c6d6e6f8ad4673576d48085b58acbabf508d37a"
            "fcad3e1b6e25a43d458ea3b57bb594a2"
        ),
        signature=bytes.fromhex(
            "ffd71206689228b40a3ab11e3b99add174599eaf6699f24890ee2cbb5df18766"
            "ba63aafcf095b016282b88613506d0e4e11df9434adb989bf0111c7c943a7b0c"
            "e0c31c01fd576b4cf9f0ddacd2cf76959e9a03dc2d070f64dea9bff4fa25d2a9"
            "bfbc8e1469cbe2146e4ef0d306e2fa3eeedf36799d351a17f8c7aeb54e3de459"
            "1eb5ac49e052f0234b5053022a357bc060fd10a357b052e25a271bf2c65ca805"
            "406d481d4d989e0adc9e7853acc53e7edd5eefa30c57e2a7540642d259cadf0a"
            "9d13fcc83a152329f5a584ac1fc185270795d4694e047159d34bad02e966101d"
            "ebd5c81acd9aa1e905122d31738b16cdd0c39a04de55fec0c5561b223416102b"
            "d0a8a888222ed75242147b6467f13845c00320704a4ab30fc126385855b88e26"
            "12375edde9d42b9aa9e5428938325b8726e6ebb032fce95293a75584b6256fca"
            "5b0f30a3e4fb1d1dc8f47f0a609d6c239a66aa43874a379e2a83492555526d45"
            "c17378493437a60235f86b5beac4f5888513b58cc58795df9caf73445743a332"
            "ea32db7861de24a545fd8f0f29c6c9e74aeee62db3f540421cca974f03b95157"
            "1a455f0375aca3ecd7cef7de12457c162b6a110a712410bcc87bdfcce26f5924"
            "9fda2df0e0d54a740426f8a6341457ad3a9a878fcd7fb33e3daf8d6c6be28f9f"
            "1872cfc8b6ab61f9fa65a776527c70d209f51d8bfc4fe94e3649c8c2bca11710"
            "196190ddb9e3d656ef620b2735a07a49291335843f706eabc466e8f4b440bceb"
            "8a34adfaeee25205175e4fa5c1d10bf64803a811632c950fde81256e242c4890"
            "1622d039b3489c613f3acf6558b139da21ce006f5331cb149db2d57e96d07f81"
            "461be40d797f67696c538a9ca9056e98424b97837cd6fb756c56210dc3a18f96"
            "729f4b663c3a03fffd250f36778649701596f972e66c96e7f3ca9808bdd6a66a"
            "8a399540309205687f6809f42b66220199fa92d2c179a8f0dc50954b6b91f2cf"
            "8d10952e4291b82bea39ac7abdddaa9db9745d88e452f9d074be83609eb81ff0"
            "56b2faba3d4ea81a4adbf0d672c3d131c0e38e1d74d752711d4b3139ce3fefef"
            "a72db198b4731c3c584ebf6474614a33ec733bd75ec20068554d0aacff08f902"
            "dc3f9c187117d5c806fa82661e9ff4ab0fd265fe6679f881dede6bf11d86f0f3"
            "bd84cd88e04f26fb630b66f16d32b6f7c0b1716aef86aed8cf582961eb1cd22c"
            "f939314846f240f2cd0f707c661c47ade801887dd421d3e3463f1768ea174936"
            "9a9fcfd11513b1eed53325f73eb13795c3039beaef75ecf8b8b4f74a8b6939e0"
            "97aa206b8ff073d0e65d0aa15939d68cff2a10da7b5f405fc91f2273558eef04"
            "e55c68efeab53cd900a6c915998b08b87f4e6d6973835f13cb304b3c0b829afb"
            "bdca894b079168527a2e6a01a1b541b08d62f279f451f4bec56fcae42636bb23"
            "1262b1e107deab7a448d9d286ebd9688753287c47eaba558f58db54a239becbe"
            "fcfd22158dc22f1e5a243ef7dc412867df5ef64eb0a6b6631a3c816936268823"
            "6d2c35a118a57f6b336ac9d355246cd7e82fa62590c3842a7c2828e8c6b657e1"
            "ff74fc005e437b299faffd69a7b64e9d0632023a5f13de757fc75eef55d0f29f"
            "f97967e9cc3fde8b4cc3d1f60a9e1a9d72ea68bb109e013734a3bd930b846ef2"
            "746b4d32d72ecc7e109eee6357c6bb42f6384c03ec2330fad3b9b084373d2780"
            "3acc8eaefc7a08930185f3873af67155b65a28273809645d221d8224912edf31"
            "3994d5d0c0a8c94859341343f56a584799345e54cf7db9a45215639da0afe068"
            "6a515aa727d8b892e46ca6c6dd7bb58db928e7a7b03429c704ed387a7afea6f8"
            "ffce4953bb144580e3ca13ad1c361da8a297c540984b9df31bad03beb35f5ae5"
            "e63b77f1c996596ac34b900625d49bcc0dc791367fe9b2062aab1701d2bd6067"
            "a7eb23742da07fb98d8dd5ccd23f8f345d35c9d76f494d9631c661e913386414"
            "c82fa9c32c139c1140d2517d374db828469ae236279d9703fae8d1933a8c4396"
            "994a9da8043bb2a62401f33847e0ec11fb2755da837987314c547e4101cdff21"
            "c0ec12951a059638d72b9efa9fdc4620c50b594169ab6057d641c593c88926d4"
            "a942488ca3b5315d8603eaca4bad4fb8772ec5c7d0714a1c25f6e1641351cd1a"
            "f34ffb2271fe1690206c701aa5d6a4e9796e634f7e49face04c664131017ca14"
            "067ed5c26d5b453c42c88c5d569d0f55818a6e6274cef97b865adc7be295d526"
            "2b2688be0b852864d55e0bd6ac77eabd39ae27942c71f03de170fa28bdde3252"
            "8ab8c086f54eabf0cf85ade7898d4e4ec9aab7bf7de492134726cdc45ac6ad00"
            "a05365bfc6cdd0b097d9e7647e9928793f63243d94ae34348d3e45c2dd928b59"
            "65f2f2ff79c2c6d7ac94b5b7af5631f8831173a067d42a670c81ba4caed1f4b2"
            "2a86794c0c2d92c5b8e2484a882177ee0354a3496e9ec936c5e4f2efd5468f14"
            "3f464b9155730cce30c00b39246a05bb4a61343c6ece4c1be24c2e8567b8d0f6"
            "fb96d2fe08f9861566fca14ab769e2f7885df9064adaf6228399d1fe1e831d8d"
            "5ba80af4271b2ed284d3608843fb43bc6f78b4012bb3a05e2c0933d4ae64f933"
            "421e2ad3dd0a1fb4918ae720dd00a36928c29ec531215f9a0e7475b259a2123c"
            "ab97991f7da339677d57e2689827257cde58b09a4aa4b4f5d3dd2003144fd286"
            "4db4d38e0f1bd13ac39b4c2e80c518cd0001cc6c93ee37049a29418f52b3083f"
            "cb55cd3cb70ec209b91e51002620df847c2613366ba82e17ea7ef9eb7d82dd3c"
            "e30355a76bc12671cb84b163d3e39be2e2515f6e2e1d3c81d86db51ed6286bf9"
            "e5a357ca08477df7dfa7d7a2ee92c76201a2413c19970856b325120284bd560f"
            "30a89c5a8bf10f815c2fc938036d7943781ececa99ce83a17ee81e2b980ceef6"
            "0c816c43b4779273e598b14063af01d06a0fb7710c092b9c8caa57b38108b146"
            "d851d913991350a508bb1e7a966b4a050e4a090af0e13b4ef0977c882dbd92ce"
            "1b4d6fddfa358bcc53077f781da42233d303b1da5af1e706d5ce29ecd133e00f"
            "a385dfa0dce00f9502f00054d00b7f1dd3ce468930466e8aa3f90daae44a8748"
            "8e2c5e375c6e7e3d94c5c91495c77fe71602713bfb1dcfd3955d1cbd8fd515df"
            "469d61e570cb72e05d226e0cae14715d8c8c1c9369ffeed7f8c29931f569ead5"
            "df354d3cb5f15f522190339c46a95dca78a3751bbe2eada967ac9d79472671bb"
            "1002c7bfd8eabe7b4586dd83b8dc5f16a020944d06518283d9d7fd8d8ca6a63d"
            "1ea701ba3cbcc52c9457b17351b8c87605205b8efc901ffe29de114fe10b58e6"
            "9179dd96fb723fcfd6bdd538cc6803bd7cfa4a780e67d2d968485f63bbe7f9b0"
            "84a1adb4c5b916093e03ff96668f2eac09588924052622b417635a833c39209c"
            "cfcc9a2639e22b38c2c6186f34d1d93d57caea0a1a5f207a83790b668a529487"
            "5fce1f66623eeffb978eb7b245df81476e4e4276c40f9bb7250c06cfe12a87cf"
            "4003b2b228e9951ae9ac24babc43dc98a2e933654d7ff81706e566e8915393c5"
            "9029406c2f6ad354fa68b470276f9b1dcc5539baeef4081cd8a2a7a4a051a3ca"
            "3f61e42cfa12196cb1b64e4bf7673ab7e792a430e0cb2570319aa1669533e250"
            "2c8fea60ea3c7e4292206dbb56f85e4239c9943b50010cb2df2619cc878aeca6"
            "e3b66154f98dfe162333b76d0fa4566cfa9ab00b7987de801961dbcf8ec273be"
            "77e5fac5458fa2a744cef02959529389aea6dab90dfb826f421f2ea29c98e579"
            "bd1ed5ae5757ebc8c8ac2ca3dc78e476537a0fd44b60fa3cac0cc49677069f76"
            "94c6dde2e0919e9f1b35cb3aae19e0f7561a1feeae6c5e4ee0ccd16cc91d92bf"
            "2a40c27fcda0f0a31168248585e9a5dd7b8e91b55bdef8ffc9871e13fe16a54e"
            "11edc56c7378551d7843bb8d7629b90265a4068d84cc72781812d7fec05d9db2"
            "a8300b3de7b630dad9cc7ccbd6656c9feb5bdfa7ab559804524ae613f315eb02"
            "9718eb45f055a68e853043283a8ab546d4767ad64a86a1887dc2bd9eb9260997"
            "b6f607a6f9d2df598f65a56d98cf690d380df3f668e2ac4b6785663a4f1f91bc"
            "d1e7c82a2b8fe0b638b00c36687fd59fb2a6be4545748fe0aa567800bcb2df16"
            "807232bc39453057d8a07cac780791249490e2d6658282c822cca1e0890e25f2"
            "d1647994db5c05e8923463c1370f6855ab3d7913aa224402959f3c00df074f3a"
            "5b82743f27c7d69874729e4738a717d468bd9e581a82d8eddf07a68644098d82"
            "a4cadea26e3cb56495a0d9113b3b994d8ba03211ef1b0607214a76adabade9cf"
            "d761bcff42ec609a4c1694466c39f633ed7e906f585833f59da10673aaf42ede"
            "1fa0d738ec8234a0a1e095d64ef31a1c547d6f1281d7a914f8aae394aad5a908"
            "4667b2b5a6430fd9616b66b839c799431dbf050909f598a4687d2c7f7cdbb7ec"
            "1e488254ba4eaa62f00770c20c1240f749949b757066ad5a60a605179e5fb5b5"
            "3ec03b6b30c8a3b7f3e0708995b43e9888ff539a378819881461c5b21b05208d"
            "73d09ac3c839e3e82c0fd9714cf1223e46e2a2f0656372ef1ac748ede3c5f5af"
            "0fc655e773dc2da9782f458900197ba393acc2ef2a60f47fe9f7b9d8fb56e64c"
            "f5589cdcb14f53b116f60cf964e750de5ff3c483b9b13107153bad83c307f343"
            "e8c67ea1bf78b664dc9210ae334a8ea490efb8012579dd4b981bf905cd1f0567"
            "e433a87aa555fc44fc669bdf7676d657c7f21fe10f290736238f57dff4dee760"
            "6dd1df009ccea6616683fde3f5ce025fd2e44b71b4d99a76fe6894c75d7278b1"
            "af08a24fccb51a07649e9c074fda05f87457edfbf9cb48e235e8d04d6f180ea7"
            "a0b9aa7d516028aa49f3ab2b662101d77b113a9a254ecb96b7b3babe1e9ee6ab"
            "a484fe21956dad636e0cc92269419ae2c2db4b1aa013b139c3250fef1ac62fb0"
            "18b65881881b57c202c39119a7ad6240c75395301e81b7c34059444a3d664a0a"
            "172cc0c8b6e6baa26039f5f58b4b7ec383e0b0c08a430308378ca506dec7e349"
            "4d116eb760a256792b19b54988f1c1f802ef6dfe00f2bde32d7fe310bd359831"
            "0e1bc563b63bff515512edecdfb2fc01e60baef4301fa2054ac286e4e2576ba3"
            "938052838fb0e15d58ff970f2a2f1265c43fc37bcccbb82985ab3a5b343ee56f"
            "51e8951c9a101fbc6c7131e12fc3436abf65fb1c8e50bc10c5c94bdf1ed7f152"
            "64353309b55de445cf40c0d591e2de54a1f05140c4033932005ccf246ffeb604"
            "b6090495f1c318122f451ce5bf29422279c6cb2db5716e148fa49525f59ee733"
            "f65ca52665e67783d52a56f1627cca9ab83be0c3a4807a2760ac6f7ee9c8ff86"
            "9f7d47ca1945d4621d71a7467bf54e85cb61d7df747a67fdd7462ca94b508cbe"
            "660e51a5b759f69183fa008c881aeab396d61a289b6be7d46988a483f9fd3f4d"
            "218a00001d131750f7392906966a140b7b1b7d45748c7fd2a4f224032a8e8857"
            "d407a73164ffa784a9a1c54f41b32153b11ee62ecef6da5de56c254bd078675c"
            "cb8f94ea65ebc6313e77184e8f5b898b5fc7ab364ee48a3e827c58a8e2c15213"
            "f9335e71bb7bbc38bc2b2a5bd6750ce0455fff1165c282713bb2fd065038ef6f"
            "f88cb95df02aec5b26bfc3920ecbc78857613d1334cf49a9b0e57b330d8293d1"
            "2dbdee4e0b38b69c107098154b12055e9f292b7ba5aa7d454b9eb949a6d3887c"
            "e17096979fde9a25272b72a5591f696cdae240a1f470d92cd860630b3790ccf5"
            "03a8c3541bb2b9f1eec05fe3683403115bc9dcaaef354682cf8bc9e7a94e618a"
            "e6bfd15047b893996da21802131ff2bf66b47c22663fa8726558a5df13df6d10"
            "19f6d12c9283e64966b20c6b339c32e14ffda0986fcf34362cae202a7aa15310"
            "2ed5c9cbd1ffa4fcf3b6aeac5751dad9cf794d61fa86077eb8edb294b58cfc39"
            "fac7914744feb5884ca5e1c7777f2334480cbbfe6410dc95e1d4c563012662f8"
            "0bbe924477691f411abe00097b6ac53901f17bddc9d13033cde2afb58e116931"
            "b6584c881e0dbfebe9f8429446e9cc62e3da35d455a22822a5ada0f79c51fc78"
            "bf8c11e29aee119d360bb866ba46043bb0616a63d731779fc641bacd427402d7"
            "aeec8fb4cb13111e4501eabca5845b592d5f5b8de0ae38140e808c01878506fe"
            "9cefbe2d57dc8874a2b0c089583b21c48dc70e45a58b353bbdc098b0efa293ce"
            "a7ddc2411faa4db1b0ebf5ebec2359cd54dfa1be70510eb13713442588ac944c"
            "83ad64e5722bb35bd835c23da8977a6d91ce328781deee4867fd469e18900def"
            "0be0de3dc20a3199b4e0a6d3191fd7a9ef0624be06777b7b6a94d86b1c8c2f8a"
            "91cfc225f06759667da56d427c88d15bb7468c141f51d6c9b7b4d5fc6c67e353"
            "3a89360ffa1c9f74a9a285226e43058bcbdd7bd8d5c60cd04b27efa6de2a0dbb"
            "ac4aecd43acc153cf942d27eaee958718a7273770f6c30bf28574d5761dec7ac"
            "c8406533f6ca3d1aeb21cd9ec8686d73e657afc1c6a573967c8031d94f1ac1cf"
            "d28384cff4f1d0335fbdfa85ee3d50098679b454a6ff6963bdcfc712f1f24070"
            "500593b16bb3269f65d74994a418075b2c8ce30baabfed8834c1708dd918e667"
            "7b30c0cac7e88552d097346ad459d5115d367c9a96adbf3d54663c245cde51b6"
            "91aae3d64f71fd6c88643f77054775992caacd020e8921c5777757beef7eab7e"
            "de1bbaf74e4e44a724f04c638636fb872cc3a72cb9d99b5ae83bd916575c60bd"
            "fe9295eeb5b0e11534069536e62688e601dd3014938d7c9e2cfb822b8452b4e3"
            "bdf38f69b68bf50ffbc20f93d0f388851277d5402d86ebeb4fcee7c37853fd6a"
            "436464e396c78b3773239cc84e789610b49c4df478a9a12fc1e2b3e11ebcdeec"
            "d5153d41e3984f37e746ec60fd1178a376ee032fca4f1bbefa2add73c34dc8c1"
            "6fa2f775e2521b7e6dd2cb3906bf6738559b79ef65b3c96f3f8210984ee2ea5f"
            "1401a3fbb717cbafa978ef5bbc39a6483832b644a5d02c51c92354c8158753db"
            "48b412d90dae27c828554ff2ed7efd6f4c9ef033e4a2843d301a41ee05a21923"
            "6f4ea888c02d3fe598ebe6adc2d662dbca4206c4d65384bab40b6f6acfb8a804"
            "b1bcc24a36ae6f3b31df3b9b47671c4045a68961574b98fd75fffac7e931a7c7"
            "f1d7db2a64f58eee51f523a732ad4bba9a1234079014a4cf1e60284e9c650da7"
            "17ac7fa22de21ecdbd339b178e0eac27d55bc417b990bbdad95cf4af1884fde2"
            "c75bb7f27d13b12c508e3ac50a3c553477a9867cee8ee0eecfdc7044475aef4f"
            "b72e58d6675d30379c8524afc0c73708d51e044a36938fba9fe72578b129cdf9"
            "3256c359a1ead261f3061836af03b511f4630388ce9b8f3bd9a3da7466fe0644"
            "56e9e1db092c77abd57a75cd882b337b18079298baaf18303b950bc2c56c36e5"
            "ebd0d691cd903a5ce9eb1420134da254314e54bdfb72e6c0e6a76181ec262dff"
            "8a4d1581d82e4e7e960d9da91a7b3187639c4a9272ff9468c9c321bd23e476be"
            "2a8491bbeb5ebe9788ab007dd341b09ae1ecab1015b7880b834f5b5f3cc8798f"
            "fb68c1a4ddd63b10259865f544532a079dffc7f94813bfef502259840c38bbc8"
            "854ad4d7888c4d35810b93c109f24b2b60331618ce4c03e6dfff067d3c3e2b9a"
            "6732322cea179d0940f0491bfbdbb9e0dbf65872b431e786ef0ed5b5911ad4ea"
            "fe9c449dbb57b7c0d5a732a6c5c145ac701250f104a408a14827e4e243c188af"
            "ad841a45628f049a0ff4a70b8fb6925b99260ea5222279d6ffb093fc94a9bdd1"
            "b6ad2abeb5b9915c07cff055347a4d9f3847773919bd5e1548fa2dd0f73b5cef"
            "4d985af4687bea0c59b11eacc10055b917f6c4f0e58412241d6370182a426968"
            "846fad84efdc5cbab2ce4e6896e76233cfe31ef0abf01f5473d15943f6d604d5"
            "cac8dc42899e1df29dda83eef6d367c82815a24bd763e2bf6b4ed19f113d77a0"
            "b503c256a75601dfb27e40811337d1ec2d548f96537c5320adba3559be2952f5"
            "22c424bdc425cad67b43282114312b0bf92ae6ac0d1bdc371c37cf9e593eaba6"
            "c838d1d8e0c72e1d10378fa306b79fa64465f7d79a3e16bc3013060f2129337f"
            "afa80c50cc73eaa2009318b315167ed11b"
        ),
        pk_seed=bytes.fromhex("606162636465666768696a6b6c6d6e6f"),
        sl_root=bytes.fromhex("8ad4673576d48085b58acbabf508d37a"),
        sf_root=bytes.fromhex("fcad3e1b6e25a43d458ea3b57bb594a2"),
        randomizer=bytes.fromhex("d71206689228b40a3ab11e3b99add174"),
        fors_digest=bytes.fromhex("58da71fa200bb3d3c4d1ad38c7bde34518"),
        tree_index=45588276414,
        leaf_index=193,
        fors_public_key=bytes.fromhex("865675837a7e323770894863e3785f58"),
    ),
)
