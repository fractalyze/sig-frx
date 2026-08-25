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

`None` in place of the state counter is what selects the stateless path. The tree
shape reaches only the stateful half of key generation, and it is recorded because
`sf_root` — which the stateless path binds into every message — depends on it.

**`opt_rand` is a recorded field, which is what lets a signer reproduce these.**
It salts the randomizer, so a signature is not a function of the key and the
message alone; a case that does not carry the value it was made with can be
verified and cannot be signed. Its values here are arbitrary by construction — a
salt is — and each case gets a different one so that a signer wiring the same one
into two calls is caught. This is where the stateless path differs from the
stateful one, whose `PRF_msg_sf` derives the randomizer from `sk_prf` and the
leaf's position and so needs no such field.

**What is pinned is what this layer can check.** A case carries the public key's
three parts and the randomizer separately from the artifacts they come out of, so
a failure to split either one is its own failure rather than a wrong final
verdict.

The intermediates below those — the FORS digest, the tree and leaf indices, the
FORS public key — are deliberately a recipe rather than constants. Reading them
back needs SLH-DSA's private digest split, which this layer has no business
reaching through, and the walk that consumes them is `slh_dsa.py`'s and gated by
its own tests. When a case does fail, regenerate them from the pinned commit:

    bound = toByte(0, 1) ‖ toByte(|ctx|, 1) ‖ ctx ‖ sf_root ‖ message
    md, tree_index, leaf_index = slh_dsa_digest_message(R, pk_seed, sl_root, bound)
    fors_public_key = fors_pubkey_from_sig(sig[17:17 + FORS_SIGNATURE_SIZE], md, ...)

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
    randomizer: bytes  # `R`, the signature's first 16 bytes past the indicator
    opt_rand: bytes  # the 16-byte salt `R` was derived under, without which
    # the signature cannot be reproduced — see the module docstring


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
            "ff44870324a4e4e5d2a843db8d4b6adcdd893491f7b2a5a2a1bad2eed9fd4417"
            "5a4ed302b22a3ad1a14b07f3e9c9a25ccaea15a3144ed535d26d3b81923e9c27"
            "cd77d039059849c85a7d64d9f4c6b2a899a2050fd9bc6a9b5ae1a928940c095d"
            "fa8d30868b060072058ac2a99a1c7db29e6d3579bcf844166628011d2b9837c1"
            "886f65cc3325898e0be5b9a5979a5c006e1fe63933179bdace0569901dfbc985"
            "976cf33021748b28779f05fba3b2b09d7ad39ef46829794741cffae2e6df55a5"
            "c72f2b29c5d71f355cfecf0302191cdcef7f54c9c71eb2a2deac816ac29125b4"
            "2ef72823f94f6e087e00f78a0ebbfa203ad3278eda33c9cf78de6146e65bb87f"
            "ac6a1f1b30edf8bf00b115436e59cebec352513ee63e1c59452e2c86ff6cfaa2"
            "590dfda4cde3b7f5a63256540af92690d96597aea703823bbccddedc3322a145"
            "13db439c8cfd6a9da3b7f9a671a11dd2188d040b4a8fc9eb9af7d2d020ab6cb7"
            "e6c628a30dc2173570ce937810463d9be9f7addad2a509dc76d2d0f1b38653f5"
            "28225b65fed2830cf17de698daf5d20f4991cc098cad92ba47876885b3dbc6c6"
            "21072d7021f48df08d75eaabb4291a0c64c95383b7f65908a068e8f87a6343d0"
            "3b02187404de0741e57660dcd35f921d0e0f450481e1af9360955180e31a5ba1"
            "eeb27f5fc5f3fe6a2da8aac4c022a684b7d4bd5f3066880a8593313d943be3e6"
            "f222d7a8bacdd0e8a875eedf5efc836793e51ff7aa03f03a6fc0753eaa31e952"
            "cf172b2a8884d2cef614f59cf91cd3746b7514f90b88e65df616cf08b1f42d2e"
            "ef047f59e92c0bfd3b0fb721c333b500b9e07d36e96ad234ec6be407cb8281a6"
            "489d650004f43366f181f89fe6fd92e7ccfac8b34ad7f80ecc81c5a32bd4c303"
            "a83f48e6af4bec8639b3257bbc50f5dac8563fe530a93edc18fe20d79492126c"
            "8761dc04a06af4e16a6be12cace0d14ef2850adcb4489cd0fa7a916ba7b18399"
            "26550518a2ea4940e6689a81ac544cd7f6f57ffa150fa82397b9449014b3cae8"
            "fae7c81481cd1820c70bb111d64e47036f822509cd5ea04264e8262a4bccd4e7"
            "c09831bb19780b2526530008c6c3968757a29421d112a5635fb85155920359b0"
            "c4093dde549d64cb9db8599809894f0a5afc5d1d85c677a684e04fe9cb5915cf"
            "3ed8c783c0c4782aa2508025524137c247fe9c7d2c021adf55c20212c0a19c2e"
            "f336f2aa4f52012609f3df824132cb04e7429985e1b91821b5d8ce086156464d"
            "555f4c2c85f44380f113772a8abb2c89608280bb82e9e7482517bc50111aa0f1"
            "deae68175893038048d816a4b2a764fd3f465e2615c764c4c4bf149836c14e21"
            "64d982ae259795e98b2ca5d8e9513ef1f4f5aa9ded29a2e947b033ff87f2fb0e"
            "81f4773ecf814d790c74696b0c1791c98dda994639d47261d98856fa106f97ab"
            "1034d20f872b7213677e8f70d0d3cdb9ce41b02c5713a93a58788331c31c17b7"
            "008ab85bed9822f3bcfa9efbe72e5dfca95e10b60432c7cf8eb2fab63c59a749"
            "5e5b4b73a020addccec55825bdddc76e2041f3891d937375e624be023302a11d"
            "531ba8ae47b2d2155c2d40637d8282b2b9a30cf791e7caa7889193a7e7bc3194"
            "7a7997f694b87ff363cace56f053df4d83677249b36e6544366bf1a0feea1578"
            "823e697b18ea66e0665ba64577be52d77122132f8debbf2ce4e9d57d33a65c8b"
            "f4cd292aa9a1b55ef5d18b09c3170ba63d6e7e61b806fd733ae3119e558d1699"
            "18e2ee3ee02d33287be74a799beb44fed497051f6042b8031a38478f968582b9"
            "3da7942ea5c48b21393db6b0f8c170614648d21e58317bfda762795a3880cd39"
            "21b5b5ec135d09bd1f9b9f8908638570bea11f31df21deda4d2cbf15681dd77f"
            "c2fa948b937f94e5b69b515d3049053bb017f190951e89114f782c26c018cd1d"
            "fb9802e59081f39df73ff7798ce4c9f4e1683ac54f7d27c62293cd84eea32001"
            "17390bdd6d46288183bcdac8223fc0fef89e53e244c24809b0556b81fa8a45e8"
            "61a9f0c03954503d9fb4ded605646919308d6e7b2bfd026eb5602d47aa0ad60e"
            "e7352e7e27575b299d15b9f112fa93f3cf1396c2c808080aa8835626d681c9cb"
            "78fb7c8492f73c315bfe988bcdceacf9763b8f7b3f6bb7660b1477c2f12bb5f5"
            "e3610faf3fa643ec0914946c6449f4ae0adc6b6c92fc4511bdf3ceaca901d289"
            "cb825cdc3bc8fd40859635704ec6e74d78a73b383eef06bf152cacd77a20479c"
            "c8bb24d86997aeb67442589139ec63c4040c7f3ef94d942ee7d4c0f84d8b698b"
            "bb8d4dd94f6b0eef8a6bf1359f236e518f5bad83248c87c4bf5030e3e9689dbe"
            "918919fa752abc41e05e57b67051e7fb223b49b608950b2a4c80ccf8f0fd69d5"
            "7be4017badb882b5aedf4c63fe8e9e613aeb9b353ec0b0a2a636acd8fd5f488f"
            "7d83c6227b40f7b48a09538ecd17e968c7f00d1c6d00414ae2cd2853094f01ad"
            "74bac219cc8576be3b258eea862d092e28b3edb7fd433a8b5a54ab6533c41a08"
            "5439a709ca6eb0283e5ce0b9f24aa79a56c66d9c16a7f898c51e6d27a76b554d"
            "335679f9fa3d96734537c5c7c379adec941f8cce10929e73235e986218f6ff0e"
            "09688cb8429d0756ef9d2076beeb536cd393ffcfeca1506f7c72c907527ac6b2"
            "93946f15e183444679993d353e4084cf22fb81f133224c54bf96784193996347"
            "3b950cae7301bd602eb9ad55a310aca2fd6b3a849fc8cd7d5a11e3fd506e0db0"
            "c7a8f55d8a001808c5a85e109617df17e28cb87e008796d0624dd35fd6166e70"
            "892b5cab0c4a91bd4849479ab1e43d9b7944ce170bae5d6643206b88aaa21808"
            "fb76e455f71e593da2730c5c6e0dda56b81f25febf2845348b5df7547baf3122"
            "9f3d054f3b8d352185106d120baceba2c08d7e6b75535938d71ada065cf55424"
            "c570928345baa920a933b2023cef1bf41852ea4ec2cd67da67f496b25a19627d"
            "d60d0ef89fb2d02dc248bac94d67d94e81cb9eebc359132f7bdf631f0e67fc8f"
            "3ea06bc21d64a939a9c2a4222de398802b125654ff55cd160d70991270c8e5c8"
            "78c3cb9a306091585a54cdd49459ee1c07e53f39b66732cc37f7e30395a4d636"
            "753d33b81a17bbf79fbda1d03a1669122bcc47abdd98218e21c9fe8e0434700e"
            "d6b8ebb60cb9585a09bdc521eb17547082e965cc39525668daaa94ab1dc4bdc1"
            "f32d4601c8ba9bd675d5510fa45484dcef0420993dd87f0732ead90e41e41f55"
            "e8bd81ca6aea7898c282cff7122112d70cdbd82a2d2d88c497311ed367ba3762"
            "ef40c62ed094ce4511d474aa5c2ae0f5e552965986a022f7be04538b8ada09a6"
            "e8849f63024fde1f4164defe5c31d1035d59ed0a7c9b7d31de206b85bf66e3a2"
            "ff4c2b0a695232b92483adc269beeb825f92c48f04dbd1fd43782254cf59d4c4"
            "c2b0f3c216f52a0e291e1d1f7caf4e38ce20d8cc76d82f8cea0bd6ed6e608793"
            "fe300ff0b28814abb35d9bb14255bb4c60ea1e6e088236be844d650085692d7f"
            "a9819c3ebfed04610d74a7788967a1628d1fa0a699858982e0bdf98f34873d4f"
            "699430577da250e47264a979e3a51e35c73ae0fc68289eb15547db88862d8f77"
            "36347c7f612b2d88ba5d5e6b39eb3c704ef474d52729c0b15a00d86c843f5ff4"
            "78fbde2a9561f359e06aaa1393ad7913a1f75755d0f596d6032277109a68d17e"
            "e4146c943a341cb914d38c6dd99619a9a3e3026185a48078d0bd632167c11549"
            "3220a55537090f45e16f2d21b6c0307d181f31b46cc1fec54f9b58a346a6e3ec"
            "40f8327c8f6d42fcd49900b2461d191ceccd3dc9200e51b5e869c3bc6b3a3288"
            "eef25f7ffc206f7c8af0f5b61f8916eb4443a535b08f4dd45f99913928d96d85"
            "6cc5dcf5343c59defe3d10395b22bbb1ba6c32b7260d2197ccf8f95a222381d7"
            "50f99b72326086a33c54c82799f64520062bb877c79939579b3f9fb2dda1db94"
            "416dbea3b105e3aeb590069096337b3defb2ab80b40a76c27d27debaf6f373f0"
            "7a518adbe110e2b86d33eb25b38e8774436f5a34ed6dc6aa0758071dae6d7c00"
            "4a445f46d33f6bb55ba7f5082fff43ab37511a9f619dbdd908c19cdea87e7e04"
            "2744b8b4b2d53f949fb8b1dd33e985d09926fdf5213128426ae367343b790738"
            "96452c39f81137a95cda067cf078eeae1f8517cd50e249cf2cba26950ca0b037"
            "22903d750feffb4bdf46f71bb3dbe8c3d8a22dcd80ca143121deddce0015ad9c"
            "0732033adaf069b8ce4701de18b4934a1cc830531eef7be4098ae875147b32ac"
            "c59423a5476aa7f92daccc2642688c5a0276fdcda7e3f03d4c403e4defacea83"
            "a88bf68a13bd28f68f8ce8346054e7fb0a53973115b6be72eb6cf072494bab1f"
            "46848d0d6bced0632ce7c52951be114304791ff4d5aefb8815f6864111181f5d"
            "88d4159db97b76b29269fbe084a23ed465a6de96cf3a6992ffddb7c151e9e67e"
            "16f54564782e7595ba3c1685d91dde10e72544123b2ba6fdb29e04d57a44326d"
            "5a2138ae57702eb19f54333c8e133ea489a2e2f501d5734c48196c2b9f057b9a"
            "cc62aea8106b77264793c16d83ecd33183e9dca8c1dd70cc2db2138557256e3f"
            "b63bcf8a226f52f55ef6998a767bc87c169532752e8409d88d5ca37aca96791b"
            "61b044b27373a7c6be529c68de69dac1072db88edb36494d718fd20cacd299c5"
            "ec79c08e2a33be83a479f4d8bdb6390f0ca7d3e3626bb4b9a8d2f29cf0df82a3"
            "32c5671191ef85fd1d012f5e29a50ec0e0f2d35c0901aff2a5c431484b9a4b8d"
            "83a25b1e13f49e8954d9dbe3fe38b08de23fd788fd08106999b92bc35c255c9e"
            "19d0635b65a4a1f5e1974273be699fa3593c542564fd9ef6859d1623846ff9ce"
            "96bbe9df8e44783eef76969b88b9aeb299a704806f9755cac839a4a3a8d1fc6e"
            "b6ebcd2d8701a2245a2bc22c1030301cfa7d3290c7299cb38d3078b9cb2d7455"
            "f7ac822a29fd8254d64c8236ea182117db859b60b27d6a7e0a20eb094e30a952"
            "a40c10f985d72f01f7fad253b9de00f9bc13a25b34a684af771297a6fdcd8a85"
            "36cba6aef430e4b434babeb2e65eeda2679fcef1d493a7f50f093341915e056e"
            "ada7c673b1bbf1d1ed8414cd0c1a923daa034963160d00fd907290e826d14fc6"
            "4408bc42c809346ff040b1c108be9b86b9ff07db34b19753759e760c953afe75"
            "e8e3e86272a10c63b2bfbf9f21c0f97fa7a412db80006708aa2b67bd7ee644dd"
            "f9f33f7f79ef84fa49cc1da6005f8a0e4a0f8bbc8ece2d316b61fe6f190526d5"
            "11150a05fb05be33d6688fd3fa663ea38f8a65f038a623450328160901ce1160"
            "8ab621bd68284bb88af0d798caa797f0f66447345babd5440d000c9bd6222261"
            "c5381e8faa5f2d47cae3826972cc8071135cf2230e939f15d9f5d249a23506ad"
            "d5cbe483fd871fa1aa8a512053ab3ec185afeae5b35f52ba6d3dc633c7a9f27f"
            "4ad2ec17795817054bbf25c36781619fe95c5482328348a60b0dfd889d837e05"
            "195e560cc9162d1d90458e3f5fed413d146c09459c85696db883056a6ec45243"
            "faeb9f282f3803a9106a716d25fd9583ce9176ac051b81cba6977fcf9dfb6ebe"
            "9b0573d24efb4e80ca96baba6a72dbf5391f5dfc0b035933c679b7910a30e1ea"
            "e4ae9b0f2c0c57beb8bbfcf6d4765c2115de3f10a4b04b9fbf2c7fd3169ba0d0"
            "3c5574cf25cef86eba503f23f701de4b53480dfc7c0876e9ac0859f04afff69a"
            "3b28e3191a051905e94ce6e968c706a2a06d27639b0203da3eaf10fb3a793d82"
            "43821916193f453d95ab2dfc40ed487d08b0c216c865e22a82c85ae97f261382"
            "72ee448efc2cc577f389ffcfece4b582e261be50d294df444018b3f95f9e43ef"
            "e6ae223f3f744a29a97db7c1d46a32734ba2a70e7c11db3f4341b3e9e0e96701"
            "923f637b2a00a72ae194c992e9127fb970bd7a223babf6c6e989c6c090ccc2ef"
            "2a2a9b12bc6c7b410eb1d06a218e62f66008ffca7c55b2e970aab9db5d39345c"
            "655b4503e92bf40d5919669269e972cd1b14e3ebd51921f318a0158a952cc2e3"
            "1ce27e0332326648f8d12a73da38beef3fd22049613a69b1ea35878f01ac3c48"
            "fcc56197b1df0ea16ea897c4027ae13e00bd3d8ea77882c654012d2d779b9d8a"
            "a39cadacba41e92b006f7f107ff5a46774b1c8f041841400ba2cd94f3fbda2c3"
            "20fd4074fe0fd0b38240cdbb39e1c94fc0f28091a7f31d072e9f176328f2487c"
            "93b03ad12f31e58bf4ce62a5e8b37c0db30549e607c65bd6d2ac0a2b5964df97"
            "25e02bed110419feebc0a5088090e8440daf52e71ba6a11f07d9635869c4ed68"
            "b09a3171e925de2bc2e7f42226802a511a680138069dcacbbab3689a034e1214"
            "f84097d00dad0fbf5b0b37bfbbcfc1ea27e7db1f28f9da09a2d015693f77d9ae"
            "024c802f16a1812b2446715250958ce41105ec7e0829098b78c7fa6df1f15414"
            "54c828947125e8a1ff5218eba0ef357dd5a7e09d16c44af5480c09c717ec9b86"
            "798f47b4b7a3d9c3214d0d4aa5412f88c2be0c67856d04c180bcba6bc796b3cd"
            "69d08796854634580910d3fdf6970823c771e0977705a2139423abcc3e3b7523"
            "e8354fd9137a072cb4fb302f4bd2689e32f360018243e963cd4eac1ab90e41b1"
            "35c1eb7cd11d087c7b3cb8b63cd415c19393ecef9e5a2221d7c0595755ed626d"
            "8795852ee73a743e3f2ce3330e6f18f8b71703f9748a3eea038e5e6679200a16"
            "9fa3cd4c7b68152156b53a7ca23fcf75031c7b3a5084131c9436ef174bf6e0ca"
            "674170f598561d6888097e450ba1507044bb04ebd8be7fd2baaeda290ab1b697"
            "c63e8a65fba1a53ec95a1377c37e9b340c95fb7aaa8588f1002f90f136211bc2"
            "93e5fba5f6c0e003ca7da7637a45f50ed7d8dee7ef1767bfc80044554ae51ee2"
            "5007a0b6a0cb26eee6e86ba82341be5a6872b6ad30a90d8f74d98a6f1bc80ffd"
            "25b72827337dc70257f1e56eb5d00180ab860543c44d00cf143def882961bffc"
            "102142a9f4e8458066a3cd23012ea4f59cae451e1e346683576152777e86e209"
            "69c7b39be9685d5fd79a4588d3622c77548fc4acb0fe4ccf4039e874aa559330"
            "8aa3dbb83776dd57027071ffe709ad63dc19e872539dee8cb28d37672f077cb0"
            "cb87debc862b22ac940d90718846a2d87dc5c6907633e1b3bc7dfb59a061131c"
            "04e1cdef507b9e24158563cc86cb939b3d0f3cd297dd7e45f2d5b447dc834025"
            "558a043ca298f3f5090ddec3c450b7a3a64920b7d65f9781b9c4b98b5ccd6312"
            "8c3ad7c64af52e74ec94a725904635396776a328f71894a9a619af79e2f0ba3b"
            "3caa6d0df80b5e65c84caa2ee0a5529df902c2484c36c712991afa5010d01c0b"
            "7a033426675bc3b76e2c3e7382692f93c2591a436322faf622bd0a62ccd2f476"
            "d72e4f72202be8d06bc222a208ace5b5e972cc59bc7d123cc3af93fc908af2f5"
            "d5fe0862f46bc2df094b270304854e1d75ed7bb1ec78d8e59636d0c8e3f9cd9f"
            "68e30d47065b1cc43414ad13cd7f8bc758ffe1951a966b121d70459d66d3e27d"
            "56418e513e6ac855f1878148d74bd6023c8709d81ae87ba6bb949b2323e5058f"
            "ec80be1eaa9db3a24f8ed9e793efead06ae299e5b542c6750986d69669973fc5"
            "b37e6de0a65f93a9dd66b7bfa070f30a4f3f16d40a49592e52a831ca81a1a471"
            "a4971af52c847732515db3b6eb5b9f03457ebb4d32843e6f9e12fb746a946836"
            "3a380eb4ad52f7fff1d1366dbc5f0aa748bc8a74b697e5c6f93df19f6f72b340"
            "1b932a68b6462e2848dc760134ad954cbc2b2dd10bad5517899fea87ecc9276d"
            "f9ad09a7a3a8e8d875b3eae96db5fb523fe40b7e58a1f36a90f921a36922c9f2"
            "19f1deaa9fac89517a322ab3b640e3c15e1c4286ac0f7cdaa2fc768549aad98c"
            "e014fadfa9cd5960668335d1476cf23f861292438bb509375da9ece9a625ac4d"
            "3f507adc517d6d18b716b055830d12641ea2149309339486ffcab5a3bf84cb48"
            "7839b3c233d2a52a6e1449c26fcf13d31667bb07dd8f23f828b57292299d381d"
            "3605075aa5bbb6ed9343f55871ed07bf78e5fa4b9516fb343448cf4d1efd1601"
            "ed0d40bcc086c56b85a0a6f5e86cd50ec79f06b2dc04face49c8ff6f59a5b8da"
            "6ef09425c0cc5f1f8b0902db878a664bb0"
        ),
        pk_seed=bytes.fromhex("202122232425262728292a2b2c2d2e2f"),
        sl_root=bytes.fromhex("017bd5a63a0ec4689a6e7ab02a58182b"),
        sf_root=bytes.fromhex("30bfb08ba74e0f67ad0369ba45432317"),
        randomizer=bytes.fromhex("44870324a4e4e5d2a843db8d4b6adcdd"),
        opt_rand=bytes.fromhex("a0a1a2a3a4a5a6a7a8a9aaabacadaeaf"),
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
            "fff2814b7e2ad7e7946c9835bbca392beb680d6bf57d5e6d027cc6712be2284c"
            "c9b58105987252b600ab976bb2ac1e6327e666690c26615501e83605aabfa6c7"
            "dd83846559a1d737f717a2a278df153087e4651ef8d24142e4dcf0373a9d1ee8"
            "1c120001b2f2028111e07914b2be9bb0fbc13d9fbc14197b9879f8bdca458fb4"
            "bf20a8710ea8693b32270141b2019dccc65d8672a6a6cffcab709590a6014b5a"
            "f719aa0b8f365d2cd85324fdf385623ef0136000609ab5792577bb53c3d1f23c"
            "c4ac3785b4b0c0d3957b1c134cab87335f4dd7df03a461dd3a7357219a3ea217"
            "830db4ee14482442043eb2ff96b2ccbe6e1aea6caf5cacfc8d759129d878981e"
            "c6083d9447ed8bf3b544b0cff4e02256dbe89142389805c0bfa801340b45395c"
            "bca80843045d36e64f4b9483319bde78fb90fa0552c45efb39e791bfea8b2eb7"
            "7ad23293a7a045de4e4f892c690345a425dea957096c7437da4c27b863019d78"
            "4e41c1a940c95abd92f12ddad9364268e204d4d0bf8a61ffe25a17b1530a2820"
            "ae38eadc2fe67d3692f862932803c64ce287801a78dfe5874336aa1e00cb6947"
            "abf80d814f1d2db9f2033b080c677458df6c7768348b097acc4e8331c8eab521"
            "5ffc8997c6105ed95b8549e742b4127f9bac58ac82640888ead7fc3ad448569a"
            "4956da6488a59c120fafc50e27c0b07fdc496722a97cc12d6414a2b90471957c"
            "69601edad5fa20c410be8b6e382933267be551dca6484219dcd9e6fe2293094c"
            "07d2134be7975009eb0640aa184dccbcee10c1a5399fa1000a54a3d3a0f70ecc"
            "13175e7bbda6fff56cfdadef3ee284fb504820c046908ee6febc02773e98cc1f"
            "e1a54ffe43708baba566259891a1fce8e3b53f98fa496757799fc1b190bc231f"
            "308eb9d5b1e5655ed3a5b6faf55e1b2fbeeae050ec6041a928e40866ea9814c1"
            "9e90fc77566d320bb2ea113c9b8ac7b2bebacae2da5f775654ae36a4fdffb63d"
            "9650c74f6e9eab1028a75f0f88b63d2f6d184f03e1e1a8b41d013151df891e76"
            "64f7d56e6dbbc025c65e6f1bc5b5d7a578b12e16b2d88e2f01592a560976a935"
            "2468665b881198fa85daec9a8f8438b01eb732be86413999509dfdb90623cf56"
            "6b8a31fdbde2bdd0d71d3b3446e204d5017a4318eaea43be5d913ebf93a4a740"
            "913035fef7297fb5dc39760b3cfb987dd7a891cb2926094739493487a645a5f3"
            "4ad5182beff648d5c37441174c4ae64c26f5f27b61ccedfa3e800ce632f4b0cb"
            "06813dbf2840ad858fa173aac445d203c6fda977bfd4286bb1eec6c54ab130c1"
            "c523c2ead8b93b2f5d71cf58041d892f4f1223734c9c0d799dd8a8dd7137caeb"
            "d63470e77a5161aaeba9c07488ca5a1e33639a7254bedbccfeb0850f78454fc0"
            "f3f49244c51d50046c8e38c94c592bd4cf7332ec3b97b1e80a538fb1a6481e56"
            "57ddbfe011195b34bd712c65108ddd7bfd8096acd45fe464545b735b2c7ed2ac"
            "c765b3d7b7eb5d50494e396b97e6c787fef641035fdf3289e23b62911d7f58ea"
            "cc36fc0bed64fd9f6bdedfe6b8f70fbe18601fc2dd799ff1aab333ed493a44a3"
            "c0b62961569652dbae7eff976236aff41d2d3f49114ab3bacfa8eba5496f6a6d"
            "ab335965d61f7de23545c8d9ba56db5b9168f70fd827897a686260cac9edc2ad"
            "99dd501d16e9baeeaa31cc63d211c6a33c1702a2d75f6a3b269a970f3a9fd35e"
            "f351abfaca4f84c97df426cbdb9322a8e56fb9031fc38d064ae47ea047aec62e"
            "5febf7cc6c410e2e137f5d8c251f84c5111dfeb85ea6e17188d345508400af0f"
            "441f31b0360217b905fe0ed87641f8302c314b370c05aaef04d8affea5c14218"
            "3fa922c5abcde5d82d6934d7cbb37f535d33db0067b679d189f408b182585007"
            "6f35fe7aa1ad46b9d436a4b9624ef04a3b4c70b89ca023daff2a12d61381c69f"
            "41b851bf140cebe86c2288da57776853b784723b46fbd2679588853749f2591b"
            "868bc28608e0afd8a8467cc1b77b645fe28ce5e80c73c34a5cee4c67da1486db"
            "f258a5b91d8cda8c4a3a12e846eb440887142078fcf325aa250633625ba38859"
            "bb6d0a82cfa981f2edaa72f6f1c30e2917b3b2b14b3b1956f014978d08f57aa7"
            "8e432fe9503b20bf1fa140d4ef935705fb257aeafc74a8489d7fa85350ed3408"
            "cc92582d2450e266b7f299b32e24e01eb2ac4ab631dfae7786e6231e2968c8a4"
            "6e148eb3eb5cc8533e1f170931d485e31f90897ff8fe35e715d7fccb97ab09c9"
            "62464431455fbaefe4e7de2561316fd849aa4aca7539c595e2d83c1d3b19ad75"
            "8aabba5882baa9e1a42ead26b7f3d8e28de3694f437694c73b48ef5ec0b7d51b"
            "c31bc207acb124992193a2d154d05aa1d0d3e984b171cb9eeaf49892575014b9"
            "4bfe24faca7eb968b5896cdd42cc5fda150f64ec20368bcbc3eaf40b7097a090"
            "882c8679991658a86d25f007e2b6f70da6083ee9b476244c46a0f0fe041606ef"
            "3420846a41e7ca3bca08d32eac76cafabd4c2f26f7b8a01883a499793377f5ee"
            "4a2e6181fd7bd06523b3de2b61eb259b710e67374c06d22f63dc8c48e55642a3"
            "99abf1bb34e2fb09d017736d22ed1e3aef13ae48244fba4e323b44f45cb26e6c"
            "a9b176cc3fe4e90040d620adae80d77cca3388a4d0a6d29c7e9e530287dfc59f"
            "4e89020b78c908b8736864c628f60cdfda44b93909f0ffbbc7ee1266b0189f9f"
            "8839f900adc62d8a9ad946d57e55c92af3353714722e7161ab228227a2b5a1e4"
            "656f6802e8958143d0d7dbd809069c5d7794f279e04d25d4fef87cc1f43185e4"
            "202efad3053770907351458922198c8c8f9c5eee56a411e12bfda9961fc4a2f8"
            "fc1fd4ed3a62dd367a86f6b3c44fbc754ac77c2661092135a2cf673e92355b2e"
            "87aea5d7e075fbb848763e2791cab897bd7a26020e3510fdcc366c0b80656b7e"
            "1b1f896e1b3d8c2ee5f0fdc09edaa4e3212ad0afa16d2a3549b351336a03fbcd"
            "51b97ef5af26b814464fda4ad28b7cc8da8aab4cd160d194f7f1d54e65691c6b"
            "3f8880cd2505903a7da491e664c4369cffea1bdd5c5a477fee2b4340099e66e1"
            "4f4bd22d05cb7286d2cb8b89a3f1c73ab8309502bcbe70c5e906dfc6d058ce94"
            "b68594ac3643f6936494044af09559ce4f8809cd132e80e9bbbd9e5df4dcfab9"
            "83e9d871757cffb41578176ee1201034ce7827e5901fab65fd44a83e26e6f453"
            "a3a1b010bc27ef6130c3965cc1987dad7042412965835cbcfaaea35e64514c1a"
            "3717c08ef6880816694c4510713d109e09febd0f2db293d1c70ff59d63619a98"
            "19063b91d3cfed72d73dd37886f1e4b739fbc335a1330874091e5dccf6ea7786"
            "2e53845fbdf973fb56b8e6256c937500a24b879cabb098f0416fbcf3d5146d1d"
            "cd0b150f63e5d25175cad2e39daf34d8e4b2ddb341bccb4aaa060f30c7929666"
            "d9310fd6ba33f78dda6880fcc22a7bf84656b0e54576f81df233a8eb7bdb402d"
            "5e5a3f9d7965cd6fc90e7e033157968d703e6aba2744dc55b13128f80dd0126a"
            "7e3f1956d8caf21db886bf3c59a073bcdb4b74bc22023b4cbea90cd4cebf6020"
            "5e7e51eaa5c8d694ef19e6986222f0927385b1fd69754b5f1773a00098fae6f5"
            "8019db610484f7267b0202db485579d822906ff73016ad1dfa4b5e0f889beaa3"
            "350f2360971760a67679e80c14c8e31a6abb668568da6359329fd1607972d792"
            "663549f4568b9d258f2ef24d90b29f14789ecea2c2edf82dd02eecc0a9cc0bc4"
            "09c8cd440fd7ba11c26377fc2825a52aa1f293ab6d1b7aeaf79dc12a2e10b087"
            "e7d032593961aefd740ef750be8d6bd696994a8e075ba740de65a3965afe6511"
            "8220ad99fa6d6b9aba68ef688c36b67f90874422bd343792bb905260ebdfd22d"
            "58482dc971cdb56e21b763ee5304166e905b58ceb6da92970e12ed11da1e34ff"
            "2a0000d09971eebe19fc4184ca53e09cbd2eb0c1606859b6ce5328090c6f288e"
            "d907d658d4afe5806626e87923213612e3b656f11617f9e7111a55d2c42102cf"
            "645ae24b3901de9b04a1ae8844529f3a1182115fa8b4bb168f4f500a7aa03104"
            "a256588bcf232c14be075b2f741df1c38539ed0f70240403c7e0ff8993487071"
            "1631cecf35e952190ad9590e95448cdf5de50ab5217bc03bce86495bbf870fe9"
            "4cb97e4bc4f632bcfe12fb9c6ce2d3aaca2090e359744d96b522aa91d38f8fff"
            "c647f0608bf2f8bf347acdb7592c205e2220388b898711ae8d779bdbb82a56eb"
            "cce6f159a99465b7675fa0981176c41ca2bf833a2f61ef8efccf0a370595adc8"
            "38bf79a917e0b41373cb08661e643c601768ff8bd1524813355667f9b3715b39"
            "5f2bc423aa60623653c1e019670e6c2cbe7f83d4262b61a0be4c0de93f0e9dc8"
            "d9fba4ce3012d498b2bbb82eee0ef29dc9c5e7651f74317a05a01125e6f808aa"
            "9202a6631d43665fbe589fe1551f33893705b617403863b1cb0adbf03fa58e93"
            "3554535d62ec70a61d7188e709c1ee943a1d99057f06b632a09161d7652d6118"
            "9c5fbea6823c09cf68cb650549ed5d7f3f1a88fee6ccb99498734cb202ec8c47"
            "adb219aa157ec7aa6dc57415ae83381c35e6def4c8ed9be8564129e78de79c4b"
            "07448982122c3fc59397139420e93a2dcb74b8def6956fdbb3faa6c9fb6b0886"
            "7e645e9b67d1b94c16b846c1a0735f080d2bb0e0cd7e434705a91d96f445e58e"
            "d8dc3988cd84df08ddf7f60cc05f9a157ebdabf876cb4c1ba0dc7e0d7b1028b6"
            "d22ee04e879dc8e23d5a150179ef72fd5f3fd9f14d21f4bd2faf364355b7a6c0"
            "d0ad80916814734ca40732ff151017b91ebd47a3794a53e762e783cdd27d1656"
            "954d55890f29cd42a8c35c64c4ee465a9251aa7f845008e9be2545a0754af5a2"
            "6171bb7358a3eabacaf80fc1d222e8429ceaf1de555e6985974e3a09e26d8261"
            "887c5951cc985e8afb56d43fe0627c8a353f69e201763350fc07a93614c7bb2a"
            "9149b1d1405fb2d439ecba77af60777dba8420af70e7cb293f68a2358d43339e"
            "4f558247123b66dcbc78020b3a87c2831fa33832088342a1105fd6fa07f61287"
            "6c078058701038e9e77902b379da63572e48e84a08d41aeca2b652d81b8d1047"
            "9a14729116c9ffaed8f91c24db9633bfcb189a23ebd935c486caa66c0fbfcd8b"
            "b2f044c459b74db47545879014e6251ced4943f1432012d172dcfc11d615e76c"
            "1ada9e4d8efab012af2e469d1c01878a7647ec7f475629fc9d082424232f82e8"
            "c5847208ce17a3a259dfcc5ddb40ca185b7f97416ea49809b7efbc20bc945f3f"
            "bc9876a3e117f4482a4f8c6acf480e885e202c626f3f4e45a7214c802e83d3ca"
            "3ee6e77a53290e0fa9a3fce5534f370135662193fb3ca1d8dfb6652bb5c5e48e"
            "400df38b58f954937372e943eb1bd7596c98c3ef9e7166f9de1aa94962a371f2"
            "30d5cc1808aba212a695733fc1dd1b0c381b1df4e2d8d85eacbd0ecb534002bc"
            "c3ea0058f6291b8708836a0a2aaf6364d694864aa82e302203e15cae17a97ecd"
            "6bbdfaa992074d518a4d91555b211fdd38c8ede1004af6195a5ea8d83dbd4a9c"
            "211f184458470b4a094eaa63306f5fefd63aa6d27a333d80c7f1bf16d39c5a57"
            "bec7bb5bd52a2d07846c0891354c388268c9ea2aff90093d9efcdbcc744ef647"
            "5d898457bec773b3954133c62c6f7e5804d2fc76c8b4f4fc5364658382789a08"
            "8c75af2e04eba3fc2660841ada58397c202d772d30498d490c7a65bb67e6c728"
            "d2bcd30f7dc9635b0dfb77fbdec52bd24b24601c8bf7104f655f0f0c59689664"
            "191bd311bc2eee05db65e46f6ad8a032ec1e8b2066c345a9e841dadebf842762"
            "c710f8a231ebe8eb0877c6e4268c87565f1b8396dcc5b21163925821be179725"
            "4d0db7b71bcd691e885ad39f26499daeee9b7cba723c79a0d185619388c2f17b"
            "bec627686e25061199a0049d178308b64ab00ca555ade63558699266701dec0f"
            "60f49ee506576f98b4b413ef3ddeac182c44e688d05ce6a36a17fe5f86f58f91"
            "cfa6378d39ab4a0b475aa9662227834bb847a0a22f1e6a3b4818b7af7f4c0ad9"
            "0cc094c940a9639e5a25f97edfcc6d4408b1fe08461562000af5f8aa096afd98"
            "5bce05d15ace8cb95dbf3bb5d11b18ff2b5716ba1e7aceeb8495d80ab7430c67"
            "bc8ca84156fd91f1ee605d3985d7d68c675fa9901c99afafc94a2dd3efe848dc"
            "221b47f5b3fa103f3fb62ef28cad3be1eac14efc3ffa9a507a9a68c44f7c3261"
            "79d61c33cd522d1e8274d521220532949714b9013e1f680284688ecaccb12175"
            "e8190cd1e59a32a2d087cd60b7dd0ac181dc002ce67a2359534791b2f63f7473"
            "26249d5261f7da3a38e3a4663a8ae525262bd90e0e934109954da4e445788134"
            "f6e49480c489670ff2b6a1ab30d4c386f315450d5c19d53c3b632df61613e85a"
            "74a0692305aef521297dcb868e1f2d6a2bd131ff5a0e0160b59d91c8efb8e943"
            "bd958187586c697de17d91edbcfce17f7bc31800b0a26c9df4f19b57cc9e7a65"
            "4385c7c3ccb58a710ecbcfc3185138c940d1ffac3ffa402b721fbe75dd199935"
            "4cca4fb5f1b7b99e97a78767335ffebd74797420a82c6fd60a36c9c7c7dcfdc1"
            "d3eed5a7dbd99a290d122cc1d55a293da572c0918d44d3f3447ba8a7164abf55"
            "d35e4ffe0120e099aa50533f38a8d67d8754090d1aace9a25ecebb572d8845b5"
            "48312785ae32896a22d58bcbd21f469a24f10c3bdd4734ff66cf1e61f1038d22"
            "2aef1e55c4a70594aa3d6c4de9d2282aae69e37063efd3bbd4463330f8fe824f"
            "0ecbedba6b6da36ee5af46be3409560fed8474e6b31fce2c2b3168e4774b5c1a"
            "8f7924c79ebc8909500efc2868b6e3a04707c84db2ae15951b85e521be84563f"
            "c681036b373bf41b78ddab5a74779cd31197bdbf22032955096707c4edb0277a"
            "5d64b977cf115bc1350f7a8e424f90996879d0dd6982afabc64478df9bcaf9b3"
            "f8aca22c50fc26e09461487bd220b7b0a0b0bc8dd4a329f88d2f8b96f91aaca1"
            "3feabd3ce17799f0efac80f680ec2bfc24837cd1e2c80f0e505447cfd306cca6"
            "a215b4eb8b292b9a30ad94813b3bb0bbbdbf748a2df0fd1977df43102dcaa128"
            "700cf2fb6f9b0cb7f27576a1e7017aeb80e33de638d2be93d32df65a94840142"
            "13dc5721627c62cb4f127239a37956c378697348753650ebab676d56dc7e600c"
            "b9b94e6e2ca5a1e7bd00dfe78ae126310a3f95708d8ebd33bec5277034e761a4"
            "5cc538a28cf3e32c546372ce069b1645c5fcfd78d1b07664e857c289e9863e69"
            "8a76b1d78abe3793d3214b7e37a6c44d159929d64bfd0696f5fa82a1a9bd94bb"
            "64e91535a1d22eaa2191fccc24d74c6d3a6b500945ed655518413c28f57de61d"
            "0cfe4854a067f6fd6097a65852a26a1697af63918bda596e88a25507602c3dbd"
            "828d0c223824bcc726bdc473a7d8cceefd94a9fd0bbe0c6f97adb55830f65b70"
            "af088e0cd28779656842faac55fae4e5aa1ac0fffdb1d021091d6027d437dc28"
            "143aa7cb2f4879bead9890a9d3cbd0bbb2afaef8594065a2792e7d6f0d8731f5"
            "d4d5f85a57e26dffaaed119184492546c0a1e6f4d8ca7a575bd4a90534e5c20f"
            "15cf12adf033b00e9d2da094f527b8bd74fd7febdd2da3d5b50b83cc90570f4a"
            "ceaad573ec70520fc555b645d733e4747e21e93f3cffbead6bde2b02e019794e"
            "e6cbdfc5442d5bde18ba87a1cd7c052e21932fa5a7520a22ee5397e276410158"
            "f25d973cce31eb5e07cd025a57efceaee7a39795900fff876eeb3652791b0482"
            "057bde9efbe83927f105c8e97b6bcf6f197d872f471ccf4ba10251c90f12e26a"
            "6b8e0d95cd64e190eeef027ddac2c0cc754bb89c6db899460a4f446fba12397f"
            "a0f7ad77dd59ebaf9860e0e4f5c007ab9999aa33209beeba998aad91068ca6b6"
            "271c73cd90ddad8b8857c99f321ae4191db2fee1ce7491570f6d69ca93363354"
            "a7ecfe31792cef65fa4e19877469a529eb86538dd6b3a8eab56ac500321adad2"
            "75bf337be5bfacb6d24c4f43b0b14166d4124621bfc229232ca164d0a9102ad5"
            "4172a116a67c20e9df14906cbaaba3fffa2ae6ac0d1bdc371c37cf9e593eaba6"
            "c838d1d8e0c72e1d10378fa306b79fa64465f7d79a3e16bc3013060f2129337f"
            "afa80c50cc73eaa2009318b315167ed11b"
        ),
        pk_seed=bytes.fromhex("606162636465666768696a6b6c6d6e6f"),
        sl_root=bytes.fromhex("8ad4673576d48085b58acbabf508d37a"),
        sf_root=bytes.fromhex("fcad3e1b6e25a43d458ea3b57bb594a2"),
        randomizer=bytes.fromhex("f2814b7e2ad7e7946c9835bbca392beb"),
        opt_rand=bytes.fromhex("b0b1b2b3b4b5b6b7b8b9babbbcbdbebf"),
    ),
)
