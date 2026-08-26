# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""What the SHRINCS reference implementation computes on the stateful path.

The provenance is [`vectors.py`](vectors.py)'s — the same normative reference
implementation at the same pin, `SHRINCS/shrincs-bip@9a7d8a4f` — and the
reason it is the authority rather than a published file is the same: SHRINCS
publishes no test vectors and no validation program covers it.

Each case was produced by importing that `impl/shrincs.py` and calling:

    sk, pk = shrincs_keygen(seed, bytes([shape, depth]))
    signature = shrincs_sign(message, context, sk, state_counter, opt_rand)

where the state counter selects the stateful path and, through
`shrincs_sf_leaf_select`, the leaf that signs.

**A stateful case needs no `opt_rand` and records none.** `PRF_msg_sf` derives
the randomizer from `sk_prf`, the public seed and the leaf's position, so a
stateful signature is a function of the key, the leaf and the message and
reproduces from the `seed`, `shape`, `depth` and `state_counter` above with
nothing else recorded. Only the stateless signature in `BOTH_PATHS` takes a salt,
and it carries the one it was made with ([`vectors.py`](vectors.py) says why).

**The cases exist to vary what the verifier's shape depends on.** A stateful
signature's length, the width of its leaf-index field and the number of Merkle
steps it walks all follow from the leaf's height, so the set below covers a
`leaf_index_size` of one, two and eight bytes; depths of 1, 4, 16 and 64; a leaf
index of zero, an odd one and an even one, so both sides of the Merkle walk run;
the specification's own minimum signature, 548 bytes, at depth 1; and a leaf
**deeper than the index field is wide**, where the side-bit read runs past the
64 bits an index has and every remaining level must fall left.

The verifier is agnostic to the tree's shape — it reads the leaf's position out
of the signature and climbs — so `shape` and `depth` are recorded for
reproduction rather than because anything here reads them.

**The intermediates are pinned, and here they are reachable.** Unlike the
stateless path, whose deeper values sit behind SLH-DSA's private digest split,
every value below is one this package's own modules compute: `message_digest` is
`H_msg_sf`'s output, and `wots_c_public_key` is the FXMSS leaf that
`wots_c.pk_from_sig` recovers. Pinning them is what turns a failed verdict into
a located one.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StatefulVectors:
    """One stateful case, and what the reference computes on the way to a verdict."""

    label: str
    # The tree the signer built, for reproduction. The verifier reads neither.
    shape: int
    depth: int
    state_counter: int

    seed: bytes  # the 48 bytes key generation takes
    message: bytes
    context: bytes
    public_key: bytes  # `pk_seed ‖ sl_root ‖ sf_root`
    signature: bytes  # the indicator byte, then the stateful signature

    pk_seed: bytes
    sl_root: bytes
    sf_root: bytes

    # The leaf's position, which the indicator byte and the index field encode.
    leaf_index: int
    leaf_height: int
    leaf_depth: int  # `FXMSS_HEIGHT - leaf_height`, the Merkle steps to walk
    leaf_index_size: int  # the index field's width in the signature, 1 to 8

    randomizer: bytes  # `R`
    grinding_counter: int  # the WOTS+C counter, the FXMSS signature's first two bytes
    message_digest: bytes  # `H_msg_sf`'s 32 bytes, what WOTS+C signs
    wots_c_public_key: bytes  # the FXMSS leaf the chains compress to


REFERENCE: tuple[StatefulVectors, ...] = (
    StatefulVectors(
        label="balanced_depth4_first",
        shape=1,  # FXMSS_SHAPE_BALANCED
        depth=4,
        state_counter=0,
        seed=bytes.fromhex(
            "1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b"
            "3c3d3e3f404142434445464748494a4b"
        ),
        message=bytes.fromhex("534852494e435320737461746566756c20766563746f72"),
        context=b"",
        public_key=bytes.fromhex(
            "3c3d3e3f404142434445464748494a4b50ea9d2a72701f642bcc5da664774df6"
            "8312b4d8dfeae3672ea7dd250076985d"
        ),
        signature=bytes.fromhex(
            "fb7440399415b3b46d3ef91f02c930bd2a000097ff8a976005f209c805818ffd"
            "10b25e053cb535f61b1de4add85ddb4f5c5a7b89943f37cd65a398d1aae051d1"
            "c49f0508db7f8496bb97d71e19db704567396b060ee7e9d14f82290017ae30f7"
            "31d75fba457a45ab034edb79ebfec39dec9e8197d6d87e2ed9f3376f01e7ccba"
            "65ec740374fa3093465ed7541b936adf140da73e84865b76037fc7c05e274064"
            "432082f38ce3182ad768a9a5c9fbd169d9d1d3fe72347264a16728fcca9daeb2"
            "223151476c98d38e65964c969460d0e0c54c603284e5f869f8aba4e6bb7ebffe"
            "1b16205427bede751757b61c9b8915f05a35176bdbb64432d0aeef8d6a7d0d5b"
            "cb7c9ce760ff539e98cd6d0c9f5f1eed89c77522509fdc30855419a3c154bd2e"
            "2f009d5ef55f018b804423f61ca95f4f4760cccb63a12434073110f5602b84b7"
            "da2fe05c7570af5feaa6d155edb5f675ed381fd4a04fb0e39437a5dcda23c8f6"
            "e9e070ddc8de168d1a2bd11b74d7c3b940c7ffbf27a92e9e460ce607999dff9a"
            "1c3168a745e5ae805047bec9d6dfbb1fd0b3a4cf3b456021aa1db6f4dcd92533"
            "e0ec47109a625eaa4b90c24e778c9f22766115d721a233bb0209b7855ae2b4a6"
            "bcc6fa5c880084e4ba12fe154df340f98a6fdd3c01c8f8ac2129c19ff19f787f"
            "080dd9f8d0ae4329e5eb8c4a4e55f79ea04aa9e298d672d4982415d2e9a4eb27"
            "4f83bf4b1f9dd9b57470c1efefa8232198fab12902ebc8437eadb383be22dfa2"
            "1467ca15df02cf0780109ad91163ed942acf56cc732a0e65aa7c73e60a2c6dec"
            "b279c4691a05801d00ff05abfe19be827367e4ba"
        ),
        pk_seed=bytes.fromhex("3c3d3e3f404142434445464748494a4b"),
        sl_root=bytes.fromhex("50ea9d2a72701f642bcc5da664774df6"),
        sf_root=bytes.fromhex("8312b4d8dfeae3672ea7dd250076985d"),
        leaf_index=0,
        leaf_height=251,
        leaf_depth=4,
        leaf_index_size=1,
        randomizer=bytes.fromhex("7440399415b3b46d3ef91f02c930bd2a"),
        grinding_counter=151,
        message_digest=bytes.fromhex(
            "18101e7f6795d92c13f799ef85d591896a66ee62394ff485f6d7d2ad46d4acae"
        ),
        wots_c_public_key=bytes.fromhex("d06a9b8629e07e64eabaab989ff044d8"),
    ),
    StatefulVectors(
        label="balanced_depth4_mixed",
        shape=1,  # FXMSS_SHAPE_BALANCED
        depth=4,
        state_counter=11,
        seed=bytes.fromhex(
            "2728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f40414243444546"
            "4748494a4b4c4d4e4f50515253545556"
        ),
        message=bytes.fromhex(
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
            "202122232425262728292a2b2c2d2e2f"
        ),
        context=bytes.fromhex("7369672d667278"),
        public_key=bytes.fromhex(
            "4748494a4b4c4d4e4f5051525354555616241faae6e7cbffe06622148b30458a"
            "68e2400dc8277dea0c6bc91b6b12241b"
        ),
        signature=bytes.fromhex(
            "fba99fd48d82398b8c567ed43f839b628d0b000c08e1b8f8c73a3d7387b5ce18"
            "173d47a5c7655c4f333cf0e6d780c4614dc23eea4421f8dcce9bec74e34aa6ad"
            "f0d89d5269370a99e16647a0e0e7a86b768296814f81aa69f74a49282df40cb3"
            "044fc5ab98d8a0f1461f7b1cad1c4cd5a5078b676e36a4df9cbe77360862669c"
            "fadfc7a68a9beb5f0729a60009b168641c2bce9e5fa62dfdc81481056601832d"
            "f3bb1ca620eb7bcd27f7f981e04805f35cdba038cd5fa8fd86c9311cc5a97ccd"
            "34ba8988b87562719a3a32e3be9bfd743b023aa81d1bdac002e2fd996ba71797"
            "8c06f479ed75c54e47244f553500dd276361dde3fbbbb8413f40de6baebf7c78"
            "7e4948a74512ae849b576039b9c2a93294e7a507cf2815c13492149df1600ced"
            "5a450a15b1e16def2bad2ab8648e469dbfbf20fefb224edf6719c42cbcb34995"
            "51653489e118be9b94fe89fb38ef0cf73dffbf27737fb1583ca4a28f9bf9bd38"
            "507c4ae07cbbb74a95937c234ba2aa5fb96ee2664a6227b574922d2b5061f4a3"
            "2358fe1d8275b437571104177ccd7c218a65bc30655b19f3ee5b1e8950c02d18"
            "bfc0b19ab29b93ca3cf156a801ed7165e86d7e400ac966db64e2b9d234d16d91"
            "abca12d99ea8c96f0026d829e04472f554e5b9a32d90fc296e52da268e08883b"
            "d8d32e15336169cb2363de0bdc4d89072130378a7d9c3b7020983ce1a5550985"
            "517eb112ea34731791fe0c60085376a5704532b79e05837f5f13a60443039746"
            "33c8fb0ed18f05a0492b1d24140acd88e7408c2a71a3cf0b31431aef12bd60b0"
            "8e486daffd6c58007cafcbf1f4933d1408429548"
        ),
        pk_seed=bytes.fromhex("4748494a4b4c4d4e4f50515253545556"),
        sl_root=bytes.fromhex("16241faae6e7cbffe06622148b30458a"),
        sf_root=bytes.fromhex("68e2400dc8277dea0c6bc91b6b12241b"),
        leaf_index=11,
        leaf_height=251,
        leaf_depth=4,
        leaf_index_size=1,
        randomizer=bytes.fromhex("a99fd48d82398b8c567ed43f839b628d"),
        grinding_counter=12,
        message_digest=bytes.fromhex(
            "f3b2ecd4b15582d19faf9faed2e8699f754363df76c09ed6248574f086cd2326"
        ),
        wots_c_public_key=bytes.fromhex("b9f985a438c6cfd2f84a190ca1f859e6"),
    ),
    StatefulVectors(
        label="unbalanced_shallowest",
        shape=0,  # FXMSS_SHAPE_UNBALANCED
        depth=16,
        state_counter=0,
        seed=bytes.fromhex(
            "707172737475767778797a7b7c7d7e7f808182838485868788898a8b8c8d8e8f"
            "909192939495969798999a9b9c9d9e9f"
        ),
        message=bytes.fromhex("534852494e435320737461746566756c20766563746f72"),
        context=b"",
        public_key=bytes.fromhex(
            "909192939495969798999a9b9c9d9e9fb4e7a5ed86c1fa7e10811fdaf3546b53"
            "5433a62da15318854f108855fed0ad98"
        ),
        signature=bytes.fromhex(
            "fe5d98141e93ca100e1f0934cc88867cf5010005343491bf81697f7d16152657"
            "d8f3f1f7936a22e3bb4bbb5f8b5c173870ea84800d3bb511b9c5c7f916057d3a"
            "73cf6f15b1d8dfdfe40ea23ab78871574ab4edd8991bc2b3405c772d13a44eec"
            "d2c7e9e5ac7c6e9edfca8bfe3f5f8eec49b2c32a3ca3d615b5f5349ae7b10365"
            "e738dac6fb67858407b51a098eafc4d64164c1d8d3dd58c9b4f88f16de1f7512"
            "fd96deee682a44b85e2e6bd51b1218a180216306fbcc20dbcceb73c5c8c44554"
            "df7971a1899a1dc9d3609948736507775bdea701bd8fc048b3d05509f5f1fd6e"
            "105565fbb2ecff3617e0a560ccba90ad137895b95c0f8f0497b16bd704a0534d"
            "394dbd174aea29a829dd94dfd0cb18d726d908692915dbbc79a777a89615c545"
            "5e91e4cc3ede82255bf9bc0b8680f1f97d81a85656bb4845287851f688306bf9"
            "cce26cb2f0c3ee61fc8dfdba51887212cd6c1187318fced64ddc946d03503613"
            "e56dde2da410eac7a2faec125691699f593bdfadd36507348d93aeff94571b27"
            "44fd12b0a170ad917caae4cae9c7b41532d696a1ec0c846e24578d90149aaa09"
            "2df1d9009b897c99fffd7c63ee191e071b5439cb2ccb10c3d6f187844c67ab2f"
            "8b6cf44758a11692674749b53cdfa2c4ddade0430a67740a78e8e0bfb99e9c09"
            "a81a081d2cd1c1e32486833f54a12518c3987d44656472af5eac3c8633218df4"
            "c793a9e06c884dd46c9641e4938ffe648b6bb59583f65455fd2f3be3700e1bb2"
            "e59588b1"
        ),
        pk_seed=bytes.fromhex("909192939495969798999a9b9c9d9e9f"),
        sl_root=bytes.fromhex("b4e7a5ed86c1fa7e10811fdaf3546b53"),
        sf_root=bytes.fromhex("5433a62da15318854f108855fed0ad98"),
        leaf_index=1,
        leaf_height=254,
        leaf_depth=1,
        leaf_index_size=1,
        randomizer=bytes.fromhex("5d98141e93ca100e1f0934cc88867cf5"),
        grinding_counter=5,
        message_digest=bytes.fromhex(
            "bb3fe897238b5b4749d9fdd7a6511b61c33a2e4a7309d7598e7b7890e764e5b5"
        ),
        wots_c_public_key=bytes.fromhex("00fa9a505569239c74fc635361b72de3"),
    ),
    StatefulVectors(
        label="unbalanced_last_leaf",
        shape=0,  # FXMSS_SHAPE_UNBALANCED
        depth=16,
        state_counter=16,
        seed=bytes.fromhex(
            "808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f"
            "a0a1a2a3a4a5a6a7a8a9aaabacadaeaf"
        ),
        message=bytes.fromhex(
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
            "202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f"
            "404142434445464748494a4b4c4d4e4f"
        ),
        context=bytes.fromhex("637478"),
        public_key=bytes.fromhex(
            "a0a1a2a3a4a5a6a7a8a9aaabacadaeaf556c7eb3d20ac5d08bd445192eeabb4f"
            "53e87684b29b8c541ad8572e9811602c"
        ),
        signature=bytes.fromhex(
            "ef3eb798e39edd234d630f8efc59c5636300000024894d670fad828d9124cfad"
            "6f2851db0c7b1cba55a10f5473d5328d94570cf89fdb7eed538a9d7643f4cd50"
            "21dd44dbe69825e8da5b7a62b108c3999e8a15535212f1b2d1a86d512076a5c1"
            "b073b6bd4b107c39d75a25e17430414b56a59f8b7eb469da00e53d0056c9419c"
            "d7c2193708e54e662c0b120d3dbb6e68ad69b56f26df94d21bd3ec756af7a88b"
            "1caa5ca55af73951303796b8b02b4bbf83983ceee563fa3b2f6f46286150e2d6"
            "4414930ff7717974ce2f9f925ef37a8c9f4464ef7394cdca2aec6a7864942581"
            "67c41c5d97f85b1e6ab716901fe917688ade6448384f1b36dfa91b89b79c32d0"
            "258816a3204cff3d288fccbbeeb219198c28dc509301676dddba920fe1af78e7"
            "c572a593937955782b00d313826e91e255a7657667c7d68d6ce3c3e4b01b6e2a"
            "e076918d3379f838186f842565d996919cbd211b02bc1318e4610e2c9b1214bf"
            "099f698ebf2b0270f5e716af5a2cdb8a52c4d45eaed5570cfb979c0127d38359"
            "ad209fc238382f1898ae1e97d7bf74c2a758a5b8f693dcc5f8c4e106f4215297"
            "6ea33529659347d5e51f84e093cc92a4ddadc0abc192bacd9efbedd5642c27a9"
            "fc8ec8e221081be4b12ee1abb813a347a31544bedc59daa46644cbfcde9123b6"
            "ac661c63f58d41287da67e96e583f24e7953c11e8db708b8fbd381a9317ec8d8"
            "c340d41db91d780d3f4167fc65867546cd7f6b2f338673ed289babd7a089a939"
            "fc226c90655da45d379dc5f9c8f646a176aedad612c9f6295221ae4eb6e146ba"
            "4f14130b46d4beef61d1a36eda2e10dece9243ef162c1c6b070f866608aae170"
            "d07822ca8ec21210eed6c26f75ff9626e21cbbecb68bd9d9c9551362a0f683ec"
            "ddbb890f38e5453320e0faf2dde57310bb1f5239e7c758821a1607fe7db94f21"
            "5ab7115717ba09eeee607caf8428ee0fc354e837ccca222b41428d1a8de6e63d"
            "3b5e77139bea2379bb0fefbfe239c85c5251005100746cc8574fa328ee67d230"
            "c3733136500479ed7ec886b11ebc6f527a8d72da4fe8840cb8736e4b42e85b63"
            "6f9166d71b0471385931afb9ff7047f4b972c5f479"
        ),
        pk_seed=bytes.fromhex("a0a1a2a3a4a5a6a7a8a9aaabacadaeaf"),
        sl_root=bytes.fromhex("556c7eb3d20ac5d08bd445192eeabb4f"),
        sf_root=bytes.fromhex("53e87684b29b8c541ad8572e9811602c"),
        leaf_index=0,
        leaf_height=239,
        leaf_depth=16,
        leaf_index_size=2,
        randomizer=bytes.fromhex("3eb798e39edd234d630f8efc59c56363"),
        grinding_counter=36,
        message_digest=bytes.fromhex(
            "71c4e6028270ad374b41e01bb00d3d78bc3e22540b956f7e36e30b3b8a0aef52"
        ),
        wots_c_public_key=bytes.fromhex("220f7352fef08676cfe452585e3e4c15"),
    ),
    StatefulVectors(
        label="unbalanced_wide_index",
        shape=0,  # FXMSS_SHAPE_UNBALANCED
        depth=64,
        state_counter=63,
        seed=bytes.fromhex(
            "ff000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e"
            "1f202122232425262728292a2b2c2d2e"
        ),
        message=bytes.fromhex("776964657374206c6561662d696e646578206669656c64"),
        context=b"",
        public_key=bytes.fromhex(
            "1f202122232425262728292a2b2c2d2ef9498443d0c8d174f42e75a66ffea6c1"
            "f0394d67aee3f95c70bcfafe92a37558"
        ),
        signature=bytes.fromhex(
            "bfd7b9b2b8ec736e66f8c04dff0159457e0000000000000001000739ec943c90"
            "5fb3ead1ea9e53723cd718efb167d0990e7fa24fbbf9ac8014c8c3cbeb7846df"
            "60ff72d011ea17d95beece272b728e83ecc9c4f608ed7da201fe35f7e2c2715e"
            "c7daa3e0f9e907cf6d70fd8b228933f415326f89b5195fcd68415c7e91b0e3bf"
            "2c54d009de0a1f78065d70152fd379428933c17a19def245aec4c241e1594dfe"
            "605fbb47922d37f00f369c13cc6f9bb7c1c888a947290ef27bbe399d0a5c35e3"
            "8aacc925e633ba501c272d648766946bf65d14e4a0c8ae6b299c955b43b8b10d"
            "e6f67e025513241d527ef369e9466627345cadf9f7c7dbceca5f1fa47e10af1d"
            "d9fb4ca983f30274b49710a82ddef9f40aaac3395ca15626432c0175077ce56c"
            "c353f5a691d9a1d2975e8ef1be47b83686004dd6ca9dfe59110389dc536e1a23"
            "4fa332c18efb9fcaf2b3f736fed302f389e4ab3deb92c024eafc65234508dbe5"
            "aa4927b7a4c1c9c8409dbc36c4fcd6554302126422e059766336f7f2fa912d18"
            "fd9a52d9aaa90f1eab4580abe4582c58f537589483d0538cbe3f78e7bcd67072"
            "b889756954ed1a0ae2542f133c02f0c13a074f4a6c1671cd37b5580097dc229a"
            "195e98c7d31d729ffd0425686f896710d456b10209d9cfec01120d02e0a3ff99"
            "9bfef1db4c45ecf194da79ae9b830c4993148e22b08088d976e5d4faeb79cdea"
            "3f1d81200566591e3719030aceaa66591866fd898ce741b0db928a897a79f41d"
            "79a65616ce0163c3d0e961e50d13c8dd3a09647aa625cdfea80bfc1cc901837d"
            "0165d39654976ffc776ec9218950f1a4890f2a0a9e3b8f9d7be71e6e6fa21f13"
            "78dc3484980b8cb76a23bb752eb030a5eaabfd9901246a0540f023c9ebee54a6"
            "0476024b2107b406703f3c060c881cb3c34fbb840da45616f15cf6b6a1a7a594"
            "2e0536970cfc07c36f92621e511d49ab25670e5348461866b16c0c6d824c9ce2"
            "64863908af75a513e9257a3b9d15b28707b00a9fac42216c730d99336d2a6a2d"
            "29db55e04ab283b2105031f39ee59bfb4437387e2689da5da8a6f8a094d9ab15"
            "1cbbab5a58578d1828538612e6622430489235d5167166630ffc8bdc9c73cd35"
            "d7081c5d9b8dd96813243d9dc7d47d876dde7f3d7c34d4742a61969ec5c33b26"
            "a90be345da9e5dd9dfe92a7bacc2692ba2df9dffecca4fc100bf56a107bc17df"
            "26120cdc1f7339ef3e4b0c68ef405e8e1fd546a57000477579c64d264b9f9b9d"
            "842ea9c2b8e8500ff7028afe0203ac697be678ec31344391cba84a30562d307e"
            "1775bcc771d595c13ce70cd710baae6807d9ff77f64b420a91f5fef0a0f07127"
            "179e87dc3a708ee0f4b3c2240bdbc8ba472e67e05ab48ce4b994d4653ef54245"
            "b40886b9ed452718839fd4e2bd4ed0fbcb0cfc89feb536922674dc99301020e0"
            "47bd536189c00d5e30e929a777aa5ce1f7d1c75346cfb8c0da6f998c3b275c15"
            "fecaae63ba74a58809add301369e7c69c3938235a18fad2be4b1d6635b29ea16"
            "acd908ba0c50c2c773f59c1ae1fd19a9f100d26df7f76899a153acef24b41159"
            "e233b5a8b34cf6abef5e47b8f7d3cd3b69bdf7d0dc67aa8205ff0e6da1bd6d6f"
            "ccc23527e7ceeaf196e536ea7850994728700213958165f29ce268aa5fd3b98d"
            "7556fb87d46df810054edeaed705ec909d56c5e7b8700048d0d301dd0dbd8c69"
            "161cf00870231b117604e26ea48a4b8c9eebe17a269f0de794e6423e7f211667"
            "6ee53d7bad31d2504f2e8f250b1ef53a7449996ad5dc784e712c9efcc849b5f3"
            "7940a63ff66646e986d278031c17043de25ecac7aaefab56f6603f177165bf37"
            "a45588483f36944ed1c43e2981f76fa828481bec04c239a5cf1cb27f7815ceff"
            "73a69f262fad27f69b7bad036f34594ea9e89a84f5c21a5974d9ce94e6ce467e"
            "68a64afdc611b2f0c023e557223be036f561fade8bb1b2ba2c0af7286bf691d4"
            "f5523945202486af6db8e3420866d66b0d0a8f52bf8fe94e74fb335216cfd693"
            "ac96b4eff10a34387331a9cb9a5fa990747811eb192636ea60b196bfebff1aa8"
            "226881092a80d6106134312b5b46b72a285bcfbcafe18d57b52e807d3af4d0f5"
            "590eae2b0bc3e8c38cb78aec4a10034a75a703c824d724f123c236b183b322bc"
            "654c24d40115c83248dee2c6ca7eb24a888223661978245312eddd"
        ),
        pk_seed=bytes.fromhex("1f202122232425262728292a2b2c2d2e"),
        sl_root=bytes.fromhex("f9498443d0c8d174f42e75a66ffea6c1"),
        sf_root=bytes.fromhex("f0394d67aee3f95c70bcfafe92a37558"),
        leaf_index=1,
        leaf_height=191,
        leaf_depth=64,
        leaf_index_size=8,
        randomizer=bytes.fromhex("d7b9b2b8ec736e66f8c04dff0159457e"),
        grinding_counter=7,
        message_digest=bytes.fromhex(
            "7fae32b92bd5951b9a190fb446ca1e7941293b59a177e92d8104a256b459c0ec"
        ),
        wots_c_public_key=bytes.fromhex("cfca81e3cff8019a2cdb936c7fa7ee3e"),
    ),
    StatefulVectors(
        label="unbalanced_past_the_index_width",
        shape=0,  # FXMSS_SHAPE_UNBALANCED
        depth=100,
        state_counter=99,
        seed=bytes.fromhex(
            "c0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadbdcdddedf"
            "e0e1e2e3e4e5e6e7e8e9eaebecedeeef"
        ),
        message=bytes.fromhex("646565706572207468616e20612036342d62697420696e646578"),
        context=b"",
        public_key=bytes.fromhex(
            "e0e1e2e3e4e5e6e7e8e9eaebecedeeef00c5372c34ed8355e5b187e27c7c954f"
            "d040fee5b2b1e9cb6c0ef7222e7010c7"
        ),
        signature=bytes.fromhex(
            "9b89452ff40639a22747621895a5cac7670000000000000001002e288ac2511d"
            "df5cb85648bb465490509fe00e928f119c9c612ddff1df628c20e391695fc759"
            "27721817594f4754b9a4503dd52723692036fdf44d2f30066410d53096fe499c"
            "b6fe14b69d3beb4c26c4f11770782dd97827965c3616cd2d72cd5c192312d461"
            "2d3b37144feda752302710e067e5e7ab332c96cef21351816bba0c813365f671"
            "e846475cca9407b0d6aefb951a7d2613a5722e266d7b289cb769f9ca16d17083"
            "efbb6f3d95ab9b743173b70f58456071c28e5fa627319e000b3832a7d6289000"
            "2fbc2a360ac86aae86c78caa5d510fa6ad62abf30ed554046d101a48bdb51b29"
            "553156b29e1c7487738394537465f50bcb86dea517412c3eadd21bd1833bd703"
            "c0d52acbad5b5d164088e4c84afcd950e864b83c7750f01d7be30fb3be84f2a5"
            "a7851b7d8df9a4f006387f513972bcc1b7843ca78031eaa968b2dc64480de921"
            "480dd4a5470b43796fad31e189e48c7d75c0e6e9998280eddb403082c3927cfc"
            "d1839b0bf9c8681f8043a011ad7a4817dde7d3eba5fa91079e3b967b8fd62a3a"
            "aacdd24e6eda1e451404d44e4dcf76a513e342de4b247c613ff1319797565275"
            "420c029dcc1f7907e268c1ee764695e258acc3da45407262f3b44816becb466b"
            "b0e877ea7590988943db3bafdbf19773c83eaae714967dac9cc476b8746c2ead"
            "907365927047cca59ff90501e7485e0b622081579c8b98f2df82cd03611bcbd3"
            "cb8d93974bd1732fc9c513af5a2d7e338a87a68a98b1f639753591133bb44add"
            "4461f632a258e53d930ae806ace9ab01fb77d4488f29413e1f5640537dc876ed"
            "ec79bab6235c490c9dd46e67c88e81d92f26b330638d8448c9565dbee44d4f35"
            "92b883f9dc98e6f77d22256ea89dbe8f0035467b14fec2dc298ee846807bf101"
            "3da19431177ba980a48bd798a43d1d6ba15df5cee09e12e0022445d39ebd9a6b"
            "794e4f8e33e5618aae005cfe74eff05a0596f2d8d77c7a31ca7ea4c412172ee6"
            "97d4fe1e267602dde9a6f7d1840d5cca6daf909a130abf721671f1b4f3fe1fca"
            "381468259507829024d67d2cd47b2e914eeb9f21f6aff48aff6afa6781572bbc"
            "0043e16244f94bbb1323be4d52d757c3b6d2aabaa2d8657cc31d947bdaa56b7d"
            "316bbfc26dfbaa0314ba636fa8844286f73a1ed0e3a931dd584c64ac440c5c4a"
            "891c01e73230118ca5ddcd2ac95b800e08881d5b69a940b37995689be01e76d8"
            "29955260ee7a8283e09e4e9bf012116409fd5c592b177702934cc729450b7d31"
            "6e6ed2979cea990ccdd0f3e5166bf1517f51c7133e29bdd21e658a566158d446"
            "6c14768eeb6c1411330e84ca8af927a894137426c9612d70974338398ee1fce7"
            "0228a9fd03ae222938386ed74986211e1aae46541b66c7e0c75b8982996c2268"
            "5036dc685eb59c300ca2005cb7ce4a2953f0e04ac45784f6f92296eea66e2b0b"
            "8af3add0926bf8b99cbdac3ae5c83e053ff767924a5acffb78e88cda429cd5b1"
            "777bdda00cd95b81c66a57b83e9d948c2834dbbf5a859af1dc7f6399df9a75ac"
            "6f3a538217771621f4b5c49a8b018c91a7630ee2785b02c1af78278678cf0aef"
            "d70161923a401b7a8f9f695659c6fd6e44301c68bd118c90d1ef8f895674bfec"
            "06c60a9c2bc0a61419066e44d5525d8ed8b3f3653f5531fbc7290d6ac9810b24"
            "e792a84558cd7e941dba4280a52db02292a9d4359971520cadf7d332c552056c"
            "46daa4be6e7944e202e66bb629dd706245e13ff84ca9a0463bf1cb1b05b6b518"
            "0b4ae90288a6ef09f6925023401aefb9ecace4622b5d873822d4a0b820df9384"
            "89bb223fd68ad840cb1d1fee84a13cc71c6095dae838bd410f6a1f65bad4d43f"
            "6eb42f4b02eda239f45863a4e495eebf75a189652ea4fadd5d71c9b391fd94bd"
            "9dfe178f4cc51e9068e3d7b741cac03733273d94e0ee40d655e8143cd130b485"
            "4e19003356d554060e5d7b61bbb6c2819e002200c88575ec95c6392fb0c02e3f"
            "2d2062112de8c6e6d1c4d84d54011cf497712a4c14a88fccc33bfe8c42d7c0f5"
            "f065c539cbf7d3d67a654be4c488bff1165685b377d0ad94ac1d5d5ffbfc8522"
            "76f2c19277247d3ff5bdbbb8f3393288f0e18405b58becbc0c9aa6cb41b0010a"
            "1b21625f8677657dbfc8727daebd107559c6c835546c7aa15640f13e0b3cb358"
            "995c2172bf8d136f3b7419db8abed921c660ef3731c38c220130dafe16992537"
            "485e806be999a80db6559b985ef6c5c3d54d2f2f07acd83cfed59d25c1685d73"
            "4e2c6a6b540b93390d1c966d6ebd1c4814061d184ce7d01259cfac40cde4f2a6"
            "cc7d3eb248797859ed43db4e92c16a068290035d80d113ce7e323ab169e6c646"
            "d1e37931e0e3ee029a4dcf07ca1a5fc2455f20870056d4e044f4b926038b6597"
            "89175e37901afa80d0364b70b6b968e48e294ba47c70c2507d207bb6940ae6bf"
            "9c5b1ff52fda8840e4376e5eda544e0901a038c819e28e1c6e77275d0622a894"
            "91d088a47ba7273fbf5e77d8f27ca094c2e86713263119f0a3df7960f83c2427"
            "c1de63f662ebff3b80f43e3ad35c517db81cf027240bad682f6b7c88e3f6a14f"
            "906dc153c4fb694366e68147015ffef35dd23b7ae730330933c37138cde7f8f1"
            "4d391c50acdb6b945540d3e85082bde675d42968df04337879831a66f9427e6d"
            "9a828431f529648cddc09eeb83d09cba46c3689835c24ff03548f1b32e61273d"
            "2f70aee7c80b01c1832ab92cefca1ec0e18e9b1995d0927206316da66e1d5367"
            "688dc97d45f1bbf5f25ebd3f3e15fd955c6dfca3874a670a36b46bc3b1fafd42"
            "4fd8cf39e206b8c7441d1ea71b3bceae4894e81a50493c1e9b6a7820e4467a75"
            "a025f216897f53bc1557da518515807aa9fcedf15b43c40391224125df6385d1"
            "45dd226f73c01c7f3ed9dde05f9c39bc316b6e75c493f16a007d50756e0d0a6a"
            "9d846c5cdf4dff6cac89366ae22037bd65434b2cda9a6db60b7a96"
        ),
        pk_seed=bytes.fromhex("e0e1e2e3e4e5e6e7e8e9eaebecedeeef"),
        sl_root=bytes.fromhex("00c5372c34ed8355e5b187e27c7c954f"),
        sf_root=bytes.fromhex("d040fee5b2b1e9cb6c0ef7222e7010c7"),
        leaf_index=1,
        leaf_height=155,
        leaf_depth=100,
        leaf_index_size=8,
        randomizer=bytes.fromhex("89452ff40639a22747621895a5cac767"),
        grinding_counter=46,
        message_digest=bytes.fromhex(
            "cfa5408c829a9af3dfcf767d227e48855cc50566c6fdb4db0f6e4f2f54cff564"
        ),
        wots_c_public_key=bytes.fromhex("b878f244d18bf76a561f3a5c345b010f"),
    ),
)


@dataclass(frozen=True)
class BothPaths:
    """One key and one message, signed on each path.

    What the seam's select is for. A batch that mixes them is the case a verifier
    actually meets — a signer that lost its state falls back mid-stream — and it
    is the one a per-entry verdict has to get right, since the two paths recompute
    different halves of the same public key.
    """

    seed: bytes
    shape: int
    depth: int
    state_counter: int
    message: bytes
    context: bytes
    public_key: bytes
    stateful_signature: bytes
    stateless_signature: bytes
    stateless_opt_rand: bytes  # the salt the stateless one was signed under
    leaf_index: int
    leaf_height: int


@dataclass(frozen=True)
class DepthZero:
    """A tree that signs nothing, and the `sf_root` key generation still gives it.

    `shrincs_sf_leaf_select` returns no leaf at depth zero, so such a key signs
    only on the stateless path — but a public key's third part is not optional,
    and `fxmss_node` at the root reaches a single WOTS+C leaf standing where the
    root goes. The two shapes disagree about its value even though they name the
    same one leaf, which is the WOTS+C PRF address carrying the structure bytes:
    without that, one seed would give a balanced and an unbalanced key the same
    stateful root.

        for shape in (FXMSS_SHAPE_UNBALANCED, FXMSS_SHAPE_BALANCED):
            _, pk = shrincs_keygen(seed, bytes([shape, 0]))
    """

    seed: bytes
    unbalanced_sf_root: bytes
    balanced_sf_root: bytes


DEPTH_ZERO = DepthZero(
    seed=bytes(range(48)),
    unbalanced_sf_root=bytes.fromhex("5e9f327b1a1f0911749bd72e9c504663"),
    balanced_sf_root=bytes.fromhex("1a5628a3c141988717bd7a07da7c149f"),
)


BOTH_PATHS = BothPaths(
    seed=bytes.fromhex(
        "1112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f30"
        "3132333435363738393a3b3c3d3e3f40"
    ),
    shape=1,
    depth=3,
    state_counter=5,
    message=bytes.fromhex("6f6e65206b65792c2074776f207061746873"),
    context=bytes.fromhex("736872696e6373"),
    public_key=bytes.fromhex(
        "3132333435363738393a3b3c3d3e3f4001736a7ee461235ed3b2000a4763748f"
        "80a3244164a5e8ca4c64f4912917fdff"
    ),
    stateful_signature=bytes.fromhex(
        "fcc5598dd9aaea1961bf0a3b23de7e7977050083488d958b346bc2dea62724f5"
        "d52713c2f617234325567b1c419947bb3819d0244e447c359102789f698a2241"
        "8c6c647b3dad805f64e721112c2e5a10c04b79128267c2252d3c938e84636ab7"
        "6ce94452d4fc2ae4779aec59ace24d616524b50de1f031701fbcc2f96d89ec52"
        "3ed83cc2ddf26ac885ceca63f41306685504ada2193852e1a5fb78fdb93f8365"
        "8b41df720e0650e1d18cf1104ba64fe1b479cf68bbf22ff073acbccc1ea1d2e0"
        "5663f27105682818fcfb08385f82e76e4165ee41220670ed907b4a162bf7d026"
        "a0e6d6ea3c894ee954bdf687da2f0de16a3c61dd899ca93d0ff79ed87bb0838e"
        "7fddc492b6f0b2a8cf980542ae561d89e4b76e6828c4585422d5d9956cfdb51d"
        "fde3f6a02ebc6dedc0eb0c41ff1968b3dc0e888d2010bd66d85d403ec4bfe5ab"
        "f1d76c524b02bbcca030b9af0a3421d24002c93e26503232abd0000624ef2542"
        "0df9c2a9b711451001c002eb881e375ac09941d49341477709cf7055617afd12"
        "357fc98e284b0c707ea4c3cf5dd529f31ed0206f28f8775dcf202ae23683e53e"
        "74769493e9ead136aae3dfd689dd70f0b1180a90d8c43f1e973a23c7bb9781dc"
        "6d1e9b8f1c4331ad1b06458c85893e76fb2cb124ecb1881956cc9061b5233f3b"
        "83d9a2692a154772860d2ac15013ccda539aa8361a8fd72cba3bcf3db487157e"
        "2ff2d73285b8233b35cf3a5d60267df44253ed93c5278a6fcd3031be3701a11b"
        "aa4489fc8ab46562c594631f5a2df3683576cb15a327e23a3564b1fcce8038f2"
        "45d5a018"
    ),
    stateless_signature=bytes.fromhex(
        "ff6309f051b88571b6da2d2011af1f5d9350cadcc2d5d94967a6041e7fe16cf5"
        "6f9ef0341809a68245d1add543dc1c5a4c9fefa13d19398f1d3f61173a89e7aa"
        "c5749c7d2b077456418cc15381839f3a2a0c145a87c7a338f52d295fc2d88bbd"
        "e286b0d470e56e6010d4c7f8b643fafe6006b4ee0690c26a91f4c2be5ae4100d"
        "3c14df603b6539ed797cc3d2c05d7b1b96794f1debcc72aa2985b19d2f120226"
        "c17cdee5fea9913ce2ec63b19b3c579bd4aa661c0b0b21761075495fefa7973f"
        "81f9661d54969ae3ed70f76f3f2b3b72d56fcf30d378d0182bdf0bbb4d735657"
        "b7344a09795a0a6187fcd68ac459169a0db0bb4d13a31218e22b099111906815"
        "821cdb05ba9bde4900f04fa31ea0e32fcfd28be89f5237f660df35bc667d34c0"
        "2be6d99f6f829ee0cf4fcd5054acb4dbce2e1a619717d50793a0aa247bc1dc76"
        "f73bacc5ef53adf64b9b811428e61583665e5e56f347417cfb288af60e74438f"
        "922d73a29c859f15975d1d478ac5d330b7ed2a1e81ae5d5081e02019b4d792d0"
        "5702795f14b4ea18e76a355602b504610ff6718ab61b2a20dcd9e0f27cb2aebc"
        "c9d472f9fe615e3c176326ce6c2bb318c92da4bf59910dfb5b192df9a73a9b31"
        "82ba0185b0d9fcc74968ff505015449e4282f8a4d37a2d1fc120c795f011ccca"
        "884b9470e9d8884643ba698cbbf482d6ab31ff226bc0268aa173e430e14a2851"
        "8282897aaab94532ce0a3f396a3bd8bc040b0aa4a88f33443e9d0f1717383ae2"
        "c0219113ead02f90753c1cb67c81198007ef2050403e918eefb5414a3ec0b051"
        "05b9ccda7426872c08fabb07ea4910c463501b6eef5a94449326ae48e748b421"
        "87d1e97a5d92dca7ea44c57a206fb69725c62b890d199621c503030f981fab69"
        "b4b5ff44d39965de0f0cb5ff7768ceeeea8baa20b86741a09b1dcd494c1bb887"
        "02f9e1f3242fac5aacf2a83a9c81e22700a7c248e4d7833ae074f637a7bb68c2"
        "72addf035b58969b12cd4b4df4f897e2662aa9ebb1bffae59308c9b933f62d9e"
        "8b20dd43c0e526db210bc1bf1c9906ad06f6bd5694ea659041bd390a056113ce"
        "da7795a78ff7c01668a148a330f26806f60fe8450b9b21e009bd03e5b0ba9fef"
        "1b5884e988ed2e3d9fa68959026e5bae856090dd0b1bc716a3cb8cf641282796"
        "df261ec43f45410885047b813381d0b89db20a34b234c2640a5e20a18d4128bf"
        "cf0b02e6b611b4223906865ea6f59aca95336cfead4d266394d7bb6c1669a517"
        "b28a6c0b167a97710ac1602c11915fee2c3790db00a6299a85019097d4cee0dd"
        "b33f624363bb222bcdc117ca2cc774665e6720a1fac3984823d5708f98699256"
        "c257e880981cdcc1d1c7115fed88b4e47fd70131d5882f1da28acdfc0528868b"
        "dd12297e1014434685ca5802846dd170e22704e7ab8b0e0556f05739a16a66c9"
        "07c8a9e6c734ea2cc79355c194205c702de062c8914a884a0f5c7348dd666c86"
        "49c9aa1af719e5e797a7b3521441bdb91ae62461e45b41b74dacb4c093a717e2"
        "00879fb2404275ebc2c2d4880b3bd8617fce1bb129e448cbb3ffaebc63014188"
        "6b6b395a71d86d677622df43feb1a00a8f6e5a9d9076ce7c94fbcdbece020692"
        "c14a2b4e09d51092cabca3366ad96fc2f0df26a715711b31d6ae6c8a079b67a9"
        "22b2fe84f1d2ce0cab25bf311aacc706408ee27415b24e660e3f2ccf344dca02"
        "2ca01209eac781f2b42c96e5de44c9f7d75b50059c233f0623e2ff88f5fc01ff"
        "83d5f53a7fb692468223ff46aecb9c54b6b0dd76de06875a453fc40f447bcf21"
        "41f64e67eb22934b86521756592d648da230eadcb5c32c4a57d570fee1451000"
        "fd35165dc684f2b393086b42bf288441cc0c90dfeef7ce744c4c88eb222cac87"
        "6aae61189a94b4425c6b569627d74e88a5c64937d7f948a67b76769a7dbc0631"
        "6a5030d07ac9db8307d306fff5216c72767039b726066fe3c7b857ccb4c3cf09"
        "0756181d1b885f587a3e725f00da9ae5b5cfb5339e1cc0505369a32a6313f5af"
        "91efac475ebd990d59836cdc7ca24f2d7fb37ea6f2e5b5d4c52ead62eea6e76c"
        "82a27ac2af2e7926a82b7cb024108de241ffaf2cc55c46a514648d94058cfa6d"
        "2f02947d494542b8804607770f0101a876092749f02b25b37872eed19621e3fc"
        "97197d78b078a9ded404384053d42cb7f2b46852d113e189a8d6d7cd4f7dddc0"
        "9019529a6169f01e57ad539b9b598357f959285f9967aeb9a3a284caf46640ba"
        "2174f9c9e0b30c19c234587f7920ebf66c8b4d9cef0b3c7357436940b73784b1"
        "4932a28d8a1175743330cbec931a11b00a55e55d478164def1aba7a6cf7bea0c"
        "8b787ee706330170b38a1a6852f2be48574cde09541d9b78b701c073d1f81c72"
        "a615595e22cf961585c2ff555c04c5fd68bb3e542b513dcb4900b9c5022d9689"
        "22b085946e8d604817487fe94a192b26ddf26a905824bd74e125d6861313afe8"
        "6f017482cc1fbfc1a055069d73f94e89111af12b63b8b054f751412cf02def3e"
        "3a6ae7a35657a757a37c1fc7f6924f3911b77b8ca663e4f61609ee0763095a62"
        "e4ade8a44f157d23b05ed566dc7a9abbf68376fa1fa41954b98c5df53dc845a0"
        "234aa7c9417430da3e0206c78eed4e11ca03e0c7347bf9c5e5599dd02472c9f1"
        "2397c653a4114d70413e3c9ad558eb8f1d924ed8532878644a379b918dff3132"
        "c35635b0f5bf3fcf0cb41acaeaa031b051f4a862757c2a6569a29b3671a1ee87"
        "2144a30d4db60528dc5b7c07c6d50634e1736e2f0ee9de1a8c803b0d74f955ff"
        "c306742f77bb51e30c8934c1613af65b8e7fa17d026672d8195788abb53890eb"
        "2fc41fdaf76dc8ec36f477f71875bacd72d7336ceaba2b50334c6043450e7924"
        "afbe3ed38748b522d5960f33406e7a8c510bb9bd28d841d87eea283e7a8a97f9"
        "a86b95bd0b66cef1a61fea70aa271a06d809dc622017b6c76e25835fc3f5dfff"
        "374b9808a8170ab0fa356a7d18167f5a14c93484e24c1da21f4aa33a0f3d8583"
        "54a1916a4526df72e9a03a5f59e41ad1350803fc7b86d6109d652d8d8340a218"
        "31d09a8a7e4357994df339042a90f78a479b254bf934177d6783b4585fd857bb"
        "9e16088acc0ffd5624fb3698d822046aaed5d4790742855d2931dd2df6bc89d8"
        "e28bede00239884839935b1ed5ddb0104f8e7bc434dc58c09b5b98dd23ce834d"
        "5a392d0a2ff2bd66d2402a24899e1d6fcb228c2022329a5561586e625a6d6cfd"
        "b2d374035a37f00d6568f9ec9a4ffdff7e48acc17160f3fe42b0bf6f6e1c879a"
        "410d421c6c9933ec9d8ec1d00c131db83fecd9ba7d1c3cbe20246e6e99619a15"
        "7b5097e2360818edf58b1a1adab1ea527f940fb5eb003da49e2835b3dac82099"
        "f5248ecb2a986f5672950f6c7df6ace290cd871b35864baad2de0e498bedddca"
        "6609bde4e32a8d4adecf18a5e15899e83487c0590eea09a1c213acec0409840a"
        "22ed34a41cb502a97dbe0c63387cfb57eee92d305bfcf7749b5e3eb14edd9104"
        "23ce0a9cb1b826a4e47ffa237d0f85d244b41d66caaa4baa2ab6e9d65a2ff9be"
        "643442af40c2130173eb628b72f30b0ed50acab6805e7256ce752587cf7da993"
        "c53f4e0bc196bad6cf90b4401520a17c90462dc444a4e39059baebd295e68229"
        "876810f1561dd2e39693280c16f2b11f883639dac36df30de5e8b46da30b44db"
        "9f2e5e54f87b67cfc82b11e5da217e3d0aee2c34b6d38272d9deaaa8d2b9ae47"
        "506aebfd165a7346073d42493dbfa444d5b8a40797911744490ad9efc97b5530"
        "60149726120e1b480c083225942cb833c2172d2869f384ecb123f43866518d3d"
        "5e4c87504c732f951d553d04da1bf355cdd7ffc386efed6abd70a4eb245c1c32"
        "6806e5f2bae70202e505db22ff3a86e74951fbb24187bd6c63e88f1959cffce9"
        "eb90ad64c6c06951109ebd93303e229b16e2d8dc43847c9c528bded0c5e93991"
        "7d053c3318364724f4c23c396ef7ab1922e0578005a04654e4e93907601d873a"
        "432f012b25d3ab59f98414b7fb27bb403ba1123c867f8c6a8ebf6655b783ff0b"
        "436472f87d0c4e4e543353c05fe12b22dd28a493098f6f5e72577b2fdf1073e5"
        "8f47dbfcb70b1f45477799e0d779af6379403f0518e1fd535a5cf4a590b811b3"
        "1a2b28b5510cf6668de539311c1142aba919978ce29e9c588a52b9b2a3366e89"
        "454de704796e7b7e7747ec991416206b908860dbed809b8d02d54e3b7865c78f"
        "2c7b3bd84834b254294577a288e0e087730a17c6e0ae15b7c7bf1c18f2fd35c5"
        "e6fb9d1cc90aa6f4d48909126a11e42b87686f89973c3dcdd2879e7bc042174b"
        "1ec59870b6222100c5ddc473fd77b6d0abb57e0a25e126c2f8b25fd6d9675f87"
        "46a033e9f5c3f785fe7cf36228f210e52e57a9be02e2db6bd8f454bd0778ee07"
        "2fbdee0157bb499dcb2cd3abaad6f2fd8a68179136c78c733bbbf848224b7c9a"
        "23517e97195ca31bb847395105ac2f20ac1373c84b50be7db75f752695f84447"
        "8b61d9c66422a1c4f9b1f02e8ccd842bb5810afdd57ac9de115993cfe4a1f17b"
        "e80a6e6ac60f1b07d199f8982df3735dc3e7f967efc22bce53d1e620e97230a0"
        "c77aaf0f9dad2d80dfe5b204ff79ceaa3b3c1c526a83139c66051695ca1bdf27"
        "641895a8e22c1e6cc4905f03b6f141c4c39359780159bd16ed6f27edf9636cbc"
        "9bd2c3e47325c03e3cd6874bc01aab2abee435890966f49f5f5e04b0b70d8aae"
        "35e16239cd33a2481e9766dc222f2ed31ae5ac1456d0d01c9e728f0b64feb2dd"
        "1335dc6690562bfc4b445458f3c778b9dd22bbe1a929fb38dd7a2632157668ae"
        "e95b4987c74cd1f453c981f206c3000671d8fcf2de8c73b1f9fe6da48493b154"
        "812f86bffe7dcdaee3368aa6a5bd2da4947de9800d13dcd5665966d09a91bee5"
        "8efa335d457e54125ab8339b49e182c7110255fd81d53408fa6ffecc1151231e"
        "b39e66b21f599c104dbcdd0bc6d5faf500a53a048aeb604ea851184604d2d748"
        "964aacc6eaa4db8b5d3063fa2a7706018e34ff8f10144e4c113ba24b3cf5dcd7"
        "9e9346805308cfc39cc9281708f64ae30989dc7ae15e80a4afbdeaf6cc968666"
        "85c31f4bf12642445e140cf70778bd55e8ed818609f444acf5335fb8263a3cf2"
        "767888a053ffc7164927867a0cf4cc02289d407dda82737d4af9c7ca7fac6ef0"
        "0945ba4036088a1a2ae907c0795c07d7a40265024248867913732f891ec912bd"
        "8a44543ea07dbd999e08638750fc5a70daf5067769b79a35aadaa2b42a9f45a2"
        "de83027a3b497dd718ab3cc37507ba94d0c056905b1455c55f948ff21a79c001"
        "8bcd663a870731ff33a04fa9133a358194c88dd0d8bdb77ae5f3a1742af2ac0b"
        "9fe97f3c0af15e0710291cad09e0ff1e800e1cd8b20c3ee707d10b8f70ddeab2"
        "97fd73135efac7e2518417a21d0f0dbcab60176f63d756c5a3dd9f3b442417d4"
        "fdcb047c574437452ae0c35a627a4a995226b921dab27a023aa8ba24cdb4ad16"
        "352871662451cf799b06e81b8333139203ebcbb43fce1eaa5957237c940dfcec"
        "65e8e8034b9b766fa0e7abae52fa80e04c64948e3b290c72ee6636f159fdfe24"
        "b0da618ae438806c4b67bb2b5853f05d0bd4ea08b8f042a17d5a262a73b816f7"
        "080680f0e41b3da0e258a886790fb64b55276772208e12cf4cd03084ae56ace4"
        "c205ca4240c4a410d6c628d66da904cc4e89576c479beeda4f53a2fb4e91ed80"
        "7a076a797d97eb1164e36b073590e0ba0167a898ab431eabcc642f1725f55387"
        "5184b03192bd6766c89a5e96234394487f519ddd563b12712a2948618c335516"
        "47bb3c4213c9391cd164dd0c11ca18a14469be7812866fb544478e2702df1880"
        "f12917861289f457d8c643637280dae6b37a932a998d4bf51709ff20d5450847"
        "43864b8d3b72394a07df9771324228e4e6f2c8597284d93b6179e59278b7725e"
        "efb3b12769eb6172f9de2f65664bc84cfbc352cabfbf404e45ce48d888279665"
        "5ca74acc6c558878945756e1ecd77429d62340db7929a812dd16f1c2552fe843"
        "4eac8c24f9cbc8b464297cb512d6132c7a462945e19f220397ba0568cee66875"
        "d238884258ad1e611ff347c96ca003ac1db50bae12f095254aec5e787a39c787"
        "9a529f8291916eac0d96f6b255a55a8e76520a0622c0517f36aa2d74e23ace39"
        "3d5e0ef948f29bff79479cb766ee3c1789ee2147bbb42344161692d39b9d95bd"
        "345459b4b9ae8f03436fa3db02faa6c0e5a815bc6f016f7ae8940398089e1b8e"
        "34bbffba924a62c2ea6496d9c6fd52d84d93e43a8c68105019e002e443393797"
        "d89a53da2fc336cfbc4e2b34220bc0d594599c70b1b30a004b9cf8821f757d8c"
        "4c15b65e8e63b609e24e6a38e9a1a88784b2ce4cfbc08802849571d1fa06f1b9"
        "80eb95c6337eeda9591ce4c533a626d53633cd7e93dea73ca8a801cbb547a58d"
        "603681d63e7ad7d69959427cda7e5cc5601a7ba0084d50b4ba8c772418d3b0b7"
        "64bc49dd3c8baad81d1686466c869935767805b18949161852ef2e14b9b5ea3c"
        "edae775c5e5e31d3b2a72e23922c8adb9680981853153f75eee1ebfaf5d3042b"
        "07b612e6c3e87a96dc3695cbc67271223a2ac875a0a4fa910d0a1d9aaceac94b"
        "41293eec7984d30cc3f6a20d36a105124b0a57d681464f196d2b169bf94067e1"
        "d6c9d211c7290aa3a89e5063cc4cd7f90b5c797ec1e27a9945354286604c941a"
        "c48943bf7b9d7207b3b58e1bffcc7f157cb79e9c301b6393ab9edb3c972de662"
        "350c898f7218bb41b67e0717c1249deaa7e4ebfcdcc848792f6e41a8eb04192c"
        "ce3b72970d0513f7e8d94b55203e7be560bbaaccc6cc027ae11851fe18407a76"
        "320cb8f5322cbf19e383e239c12a54285625630d78edab80cfb2c49da4acd21e"
        "00fa5eb9ff7401947c3a6f8ccf0be8298cc21daa21463d103fcb7cfb0d8be6a8"
        "0884ba7abed0cdbb0019713fd5ea75953594f4dccc7394c4eb80d59c253c76ae"
        "4fcf44a96cadc0c5b9c590d11d7f11921da18669c2916023575d549af62a841b"
        "0811735929d718b63b76e85c71df747590c89a5e1db12db3fa93aab35530175c"
        "44c58869371eb016801970493987370e0076732b53102a30fd595867366313cd"
        "43e837737fb3b8ecaa39f1835254de727adf59c6a5aee07221a85e8b86fdc022"
        "d82fc10cf38d6282e2b28a7570271b03df950f835b397362fe13a47b2039e38e"
        "5a34561f3f0858ed1f057e4c2f2adf78309b90fb66570762af09d1c799abfff0"
        "e9819e1ee83296bbaeb028213f295b7b3b85c8325ed939515c458c6d7390be60"
        "0b0783b4e1a965c1c9d2af97ddbcd40b361624ce4124eec9cd4ee8a1933141d0"
        "062aafc28f769544a41eec1be4c300c28498a73199213e72f5fd047780498b52"
        "c9f0cec0728488e92fa400c797a71bfd135eff05be17a90eb66ee3f2d5490988"
        "d639adeb9ae634392c4c2a5b99afed96e0078b89c6419ec487e973ec0b24cba0"
        "74d215c0d882b24bbb0c84e8e5a79b344d3772bb3bf1afd81f8faf88fc49b335"
        "e5dfd527c561bb0baf5e55496239531952820d6647a7d7dc8f7d29c1c63f3968"
        "19eec598fcb6f70ba32d051319d90179b2c718b8f66b65c2567ee64b68dadb3f"
        "f254596cd40c7568a535b4f3084d286c594639376b0494d6eff7a3f5b020437c"
        "58ab2e757cc19a7b3110f8d9787b6480411538e06041ce193c47efec9a411d2c"
        "2d975f286045826b3b0b7916adf3985dc74dfb8e7dcf6c19570a591019dc4347"
        "c4bb85a3d195296d6672b363f9477f891b3c06711eebc2f9756c9d91b6b83a28"
        "74ae3e8c28c913a5c2356517fc73a867f37835e6d72e41647d4e50c85abd64b8"
        "9602286ee7330c75d97b6c7ebf864b9a734c1c10cc6c85c2343671b82bdab591"
        "1251e73af8d7b1f7964d39ff1dfc214a5b8fc879e60700dd6f0eb5e4a76c6ea4"
        "5c14f476e33191aaa033610d1e049254057979d3778c821b480244bea259b8bf"
        "47f53da80e13335d6cf852c5bb6399b58aab9f78cb6a25d806e1ec0a07248859"
        "6f24e244aa0ef48d49dcd73dc5f40902a5d4b90e1be2b74a09efe120e4631085"
        "f8a2bc0a58ed8d9e2b8cdf4dcdad8a71692d5a5b65511f5b5a7d1f4ea4b004f5"
        "9575c66dd75442493515f3021d49ec6120"
    ),
    stateless_opt_rand=bytes.fromhex("c0c1c2c3c4c5c6c7c8c9cacbcccdcecf"),
    leaf_index=5,
    leaf_height=252,
)
