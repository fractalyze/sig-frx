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
`shrincs_sf_leaf_select`, the leaf that signs. `opt_rand` is passed rather than
drawn so a case reproduces.

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
    leaf_index: int
    leaf_height: int


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
        "ff11e138d82f7d6060f0070ac83b0755136b3d01a79ab94111063357e3ff9a1e"
        "36b0677b2615095b252c5c8faee1390374b9555954b9135fd75ca21ca6c484f0"
        "905f99e70b0829f2e0c9fa04ff57ee6791ae7b192b11ee0bc69b2af338cf5e42"
        "6b3c77246cd7bd9b1300eeffcff160865d352287d3ac9be781a5079ab824e111"
        "4d8dfe558485f3643ee31b20f73cec0e421003f581ffd31a42418fa276e54c36"
        "46493fb2fd28922bd62dac92a0944f24764b8cb919aed9224ccfe185aae48c30"
        "3722f40bcce35831c3ecf6658d492fa04f19737756f42ca38a9fd468f396a354"
        "951fbb512e09a21708b26201e0c0a58353a4dbb19049b817d795dd6f68f42deb"
        "30c3f88390ca0f7a6f0423ad26dd0527a032e8970b69937302604c192a85eaa3"
        "d26aa2a18d0f57187f13c5c35dd07e6f06a5abe0c524adaf88a856daefac3442"
        "27e58a0e76a8eadb454248089c5e5f02d590c6a0ae418fc7622934b92e590982"
        "f096982e7e7a3144219b68257728ce24d8805451766611642dd25fc65266b476"
        "2de3fad15c64c1694092bdb7e0d48a87e65515bbcf5564bbe1aea6522ca5ea9a"
        "9df0d594d99e389cb8b53b00ed38aaece4e335208e99ba89bba3d88555478771"
        "0a11d110804d1f3d36f357abb7a96c7c0715d20bde6a90b310e5dc7ffe6354d6"
        "88f7b584f5781297746aefcc279ec3e8bf4d547d21de92038ce9c5aa56ab8a61"
        "a74207054adf46c747e1e751b7d3ddc890cb25889e1fe51b4781859f35c9c4c9"
        "2bfdcbb92f1913ad9ce77099adfcd9597c9fe540f1e227c9a0004c55b9529ae7"
        "4afa8545986bc43d5290565bf4d05c316a3cf74da2ced892fc4cd1feed190289"
        "f3fa14d6974b12aef50aefce9a845d05346ac4972c0a2e6a8e0513970326725e"
        "8be0b897ae58fb2294917a2bc5ded01a22bddb681d5f51fc48fb8f82e497f9c3"
        "ac3943b52e36292e1747632fb34272121a847acd87bbcdd70c54245b9c6bf4b2"
        "2e268f6ea1ca713a816fe117dda0774be475b27dab8e1147b4ef63b151b11360"
        "3e128cf294c111ce703704188eb3223170d2a332d358537a2b90723ddca24d35"
        "dc193d5b12f59c7bfd6defe7735e56c9773dc9601f4b7023da58314801ebbfb7"
        "e7651531596b035ef35831604349d16a71d12d20ecf04d828989108622fe8b50"
        "c2de538accfd66f0e90be9378276f89c058f5cb02fe2d997624b2206e454e0dd"
        "83ad39873f86ff02e8a5b338cd9d731a7a5fa44f6d6e807377e2a5fafc6d01c7"
        "24e02d75ba36999f677bf4e55d693acc21aa01dd06da39b020e20c8880a2771f"
        "096116447c63218dc23a2c4a2aa519efba73514f287ef08b4bc718b6164d1d3b"
        "2b1f76310ff405138fd2db0b6a8e829b992b2d17d3c19685399e73bd1ce11674"
        "372ae3bb21881349164aa543564562efec72c86f15e0a20ed9b7a57b8ea9de24"
        "78c7ed019e0c1abc13c99807ff7419bd201d01736d4e39a9bad8c7d8d4797351"
        "7f320497ab90ba05094a9e071a8c24b08e82b7b935615ce6368b5c3ae44147ca"
        "b3c6352df6c4ce45f7de07ebabee588679ffa43f71173aadb126d6d8f0557cad"
        "a2028ab7b8a99b083e12ea687c8dc49293bd53b671a36b942b1d8e5857646da0"
        "202d4e264d50580b816fd1c97fd3bf8ae82b9e891090d4d969df647e545f35be"
        "8b727f5a5dc895b76331a3d45ca8f5736d578420b3b9b744e0e36e5bda732113"
        "68c1798908df86e02e1bfdae949793b64857e7e3860e6cc895d43cccf33d3299"
        "70683a904c2cbdaeea2f73cbc8ae6339aca3b67a56d45e525c1cef9c1248794c"
        "9690d8df3e1e562df80859adfbbd306a0376c4de77adaff47c0ccadc26f3186a"
        "680501c1ffe1d46d5174c2e64153bd20c9fa9b9b42586092f41d4c214f2e5b20"
        "ecaf980cbe581f209019a292d4778838d646d61b0f76c565fc827af99178bed2"
        "e58f0ad5e6a1ab592797ac9abedf078fb094c19becdf2a4c92b670f6860bf9b5"
        "e9b529a8dd387856b4d46d49da9158bf133ba16ab9e9a7b843c33d92f882b87c"
        "8e3aaa63d4c4d8dfb291b5dde6d7662117ab7c423cf17627e437a589a69c8da5"
        "cb3d556b8c63f7498c33df2b1e5b296621330233cb1ee0b537493b5c864b2e2d"
        "0de76895883667de5605dc63a0cbb5d180a538dde14641af9a2699bb02898ba3"
        "659ac51a0eddf84e931ae7a03c93e9493fe3595d40c35f6889f4296a9436acc7"
        "a04d650966f1af49dd8e379ce5fe96d7521053f027ffb23b4b1188090ecc3514"
        "b7e9f3233199b0f2530815e5d5f7c3a7355c194ca3ba04bc4a3420c206a79534"
        "c082628dcd7bab4758b9ca2c4ee393ec7137fb4effc568ac7f50c6e341c053f7"
        "0345c4cc27cfa65d6718f13715d6554cd5b78baf3fcdaef73e735730b25b2151"
        "b8f3d1a49522d2e5f65b41438e864176b2972d859ee16208b8dd7fbda33ca647"
        "431d7fb3fdcc74a4b0f6d7c0c413583f137253dc4e01c0f3ff0b2d85e4ca927a"
        "66ce92f08fa045f934ea9e5b0b88e79bb0364c45d1d422074ac99fde31856b60"
        "9e8fa9fd19e0164f710eb933a669a167f222133107c6afe56a9cde8b40cbdc3d"
        "f61ade99084a29b373e527d98c806da7e1c6cba2ebe416bcb78a22227d0af935"
        "39ccee71e25f54ec6ee9f7bdf7a364fde94ac261bf1f2599ca65f05714d1425b"
        "fbebf7705022bb859e817bc503dc707cf9bc97e2a06e4f6c7657135c756c986b"
        "d207b71eb83ab8d74b3f590fa01765c8fe0e3a7a8115b4b0941b760ac71cfd04"
        "582d2c640b51fdb1d2284f743a6e1f28b6839d74a51f4df0bb473b4115c51362"
        "6df0699af6854b4140311f4e96283bd3d7e74aa10365200e881a6b3ea0816b91"
        "b1d4839ec299d49343be44c9ebd89375c929df280ccc05acab261a5e2f2fb4af"
        "cb5ff8d433fa0abb275d3f412765aba98eee9c82fcad3b3b3cad979cff178899"
        "16c3290142e57755a3a432a48e4bebe2403caf25c79b1ce1dc3bd05f26c52422"
        "170f8c05d22dfd0811355f78aa84c62d25ff9c43d2ed1ad5cdc8618aaab9e3de"
        "33a36c9e4221910830d20b34a45cf4def7cae96476f23a8d243011a6b3e7d445"
        "42079d4e30b5fd6fbb6f71e983c3836b7abfa10371cd9a7fdf9a813502003be0"
        "2d4addde0e320862303ee8ebd73a14d7e83e440e1e62b835a22779c1e2b6c6ec"
        "b253a31ba865049140394776b91caff3e838245dfab0acfc924076d72286ffe2"
        "8ba50b61a6fbd07edf6db61ad80338638b3232abe54b957de154f4762edf48ff"
        "54d20ad55ccaa4aef2d594d87de6e4dc1e657ef9360bf949b86f6334678171f6"
        "2b99ce9189653c61b0fbdd9b28c71bf8a1f6a064fb3ec70b6258ba6bef52ecd8"
        "4c7af31aef4aad417789a493a9a2712b93aab9cce2cf7d7e0b609806427cbb8d"
        "c8d3506d57ef184f8b8ceef23f17c6f7fbb0cbddd0033db86a142e7fdecfaf2e"
        "7049573ace64ad3d3a69e43a3e2ec8e32aa34a09d8fa90ed586e50aef5f4e3e8"
        "284924ccd6c9d589b5082fcb9100e9841c5e024726ea2ab65059309ce0c4630c"
        "a89102123e831eebd971793323c6c97c08e5fbecd5eb459a6ef6e8447c74d80b"
        "7d0e8196a1a28ffafa5039a9fa13ec402830f9a9b7d2ae8cb2437e9d43ef1f69"
        "10115a94f173709adf415cca28ae50306b5a173db23712495e668c7ace67f9af"
        "2774f4670326f43c55c4bfed0e0ca95035929f254dcbd0a2c765149b9a28bdba"
        "6d34a557954b45b15997d0dff43d22bea11cf932694a5a0dfcf537e6aa65d92e"
        "351edc958a0a7903cb95d73d6cc1fc36660675286fa94c2f9f1075a43308d84c"
        "efc7d4a2262af5f66869dd7d6180c2505c11c56558feba136b1d4f5953081e83"
        "8d81779a4cc3cb952891463d14987cad61fa3aebf2d2e48b720cef65666fd79b"
        "ad74a65c99606c3a30f6f1e69a2ec52ecb6c14661f75fd04a8e8a11a5984f039"
        "0a7585c3d33f3e7dd2bc2d3af97338671311fa8d819c84b495cad65e1be2e75b"
        "5fe66695f0f8cbf7bcb9197c1a68229fa1105a564881648720c6c1dabe634854"
        "f362b42cba92c4f87eafe15cd096049f2b4a172e6e93a74d5c1fcb83bbf36234"
        "0de1a64c9c6549f85fda56248e9ddc714f3edfc9259e7e9bfcf113c94b40f0d1"
        "6d3438f6d4ecee478880bf989a7db8e9e07505a8d6b8c27e3ea65d0f9ca572dd"
        "e34cce412a725d25435b9ec69cde6f6ebfc0026379dd70f11f86d97ec0ba9f48"
        "f70a214174eac44a4ef35408511df8cdad73ebc58dc0a63463d0b598219138cf"
        "c36e3f75014754b51c045f380658ad299ee6c416afd19c47e6e165a97ff643dc"
        "ccee93dea390bda31ff3655cfe826777724180d86a19da81eadcf8f3fcaed816"
        "b4d526a088010e122ad10617aea6a30d23352cb6d396629a94efe08eac3e708b"
        "b679a9d846ba03d0618c2d60e90cad30abda2d9edf7d08f360702778bf372518"
        "6f08f73a1bacbde7411085aa0f0d8cd7a0d580e74f5db477bedbe977b6fb27ad"
        "d5bfecfab8836e68767bdf6ea53f397d7f495c6e19cc5b12ae4720d27d51a550"
        "b26dc2ffad4b65483ee9e4cc320937fb554eba64805286943aad88c0a9cf7ae0"
        "a77c69bdf4836d6d0358f8670885a7abf7ba335e36e7a430cc2513e0c06b7096"
        "541b89dbc1a0fd6b84380a35484bfddf1cc658c2e1f07aea2263ffb8fd8a6a2c"
        "410dd73f7877aab5a860d2166b8d63f36a1f83450bde865a3575a1966e55cf04"
        "1da4f6a0446de3cd7693becad3a59d141d832b993b4345ba6d656f73ef03d927"
        "22b8643d7512b2a2120000cbb352ddcb1c2c10bf2f286f09853d06cee2ebbb1f"
        "a0a653c1bca60c5b3b2140ac0cec84681e6f4b77ad83cd1e02de677289300cb3"
        "bcbaa48680d9caf276bb2e3a91e13ffd38bc63c67b29946c3cd7a2a868e0793d"
        "e591ebf778a9b39a414dc43b33ed2e80eae5c6cdd96c3aa43e5301296f9384b9"
        "64b24a9eb3d4c52f5541986270c25fd3fc3ae0009717e0bcefe5fe7e994f3d65"
        "0e5e1862ca00649193a845155cbae443c3b49801efac0b2acfcc40ba81565686"
        "fef74be2d5473f21ace2360f18e6e2b9df536e25353eae114e01ad0d992f6dc5"
        "0f60672d22f1a7ab11b9a36437ade895fcdd6cbf5ec5d932b43130d2c3885bd1"
        "760d92590ff4eef06548ab8de3182f809cb21592d4c0e8a7f4e5ba244f2c0709"
        "b6cb29229d89bd6672ddb6680d0fb88e8686afa79b21170c4f88336cade8fc87"
        "02403d9ec1131356f8219491439d84aef599c60a36dd3d89b4af1b80a3596e30"
        "b4ce46f674671a7ab5efced72cddf89391fee6eb966ea06d86afe8f970f199c3"
        "ff31dd78b8e6d41c7e3046f54fe20503b3651af5b4b003ef16d2093dcb81a7f8"
        "655bebb4ed8b79b0d97a8d5a7f493fbb81141f41fde42d4916e926836e393d59"
        "df1c5f68e16a4f23ed8a5e97f26ba794715c6c41c58e1d35574e271d0d1a6262"
        "1d285e618a16c686d642535a29d9cbad11bbaba1960efd0ac0d82dbb4ca3d83f"
        "197561bc7a450786243a1679d996ef9080eabc11a77b500fed82a119232a609f"
        "e17dacf65dd6feac78fd83c83be9a81813546b5d0f0be4be2fdac2cf703af5a5"
        "d40a68ca143fbb08b39c4b75148dc2600b35155273e50de018b28702d213f336"
        "11aefaf50a394497b5e615b372f5bc470e0b057e6fee7cc0b82d05795325405e"
        "5d2670db4e2b59ef2130630ab9fadb38be9b44f2c7103845a13c58b6950b35bd"
        "6533385d5e42d00737fe45e0200d01ea22c3a44ead1401179cfe328679a26eb4"
        "f2712bcebf0cb6c69aa2b86d5f09974603b0eab260c983eaaa2d10abc457bbe5"
        "95cc8b06d9d9d5a3d83b649e9ddcc1f4c8035b49b3c121d58e6afb18aa2f1d2b"
        "afa797bdb16afee2228007605f47d8c284b29f0f49de56997e83851be1745fca"
        "e79109d33a9f07f482f88785fe7d38b6e99a59a68d7ba5904c0f193e2c5476dc"
        "c4d16467b1560c6f34c41a86c592c7778e1268e77c2c0c1636637f7a3f645cbf"
        "40ea7cf966f7bbeda09dda6a812d850cda4d26fcd0624ef2ef4fb3a6c640a492"
        "d9fe19b75fc6dbff7ce3cbfa536c3a906c02d5b6a5e7a8728639e1509381895a"
        "c0f1ca236ef1e6047794ba611077a4b627e3587e6a4f88b9c06869febfb8c162"
        "86c801b2b97dffbef0bf58674596e07fc9b485d4a39c75b8106aedfc3bc0d86d"
        "a6fecade19dde691ef268b55d90dbddd54ec07fb3b2dff15bdb68515be580be7"
        "9b168a62b60d0a4c3221a303668fd1b7d6a6e4414417116753dac8721a0b6067"
        "5cc2cc2d781673ab3a782f3eb8d1602e39f7405fa07e5e34de4ac26b304dab51"
        "d24f0dbe6b7ec5eb3bb094af35fafb8c3912ed896e3721deb7e8acd5b64d64f6"
        "e6a900732dc4c667bb37ac0f8f0ae401881426f9be288c87db6648cccd1b5946"
        "c0244abfd9d1b0e02a9c220a133dc580b34d0db2e1ab077160258a25638ea497"
        "767b23dfdb4e9b4bfc0cda6231613cdd47636d10487a0c9379e2b493cab0c774"
        "e91b3468ef9150150d49ed4ed08ae42d9302e60b8a5239160800edc9c02f2cc7"
        "e7a34da8dfe1882bf8f8b4beda72244141b8649cb4cdd2c58d85e5e514c8e4c1"
        "30fe4d3db6005682b1cc856b298ad89ca91a26e172673452b0cf55b6ed05232c"
        "2cc497291f2b2a93a23404cf0730d431eeeebc917abbac0fabb11ad30f68a181"
        "18a09413cfead46c8ed27f83706c47c49956ec19d3408cdc596cf584f23fe9f0"
        "efab311e1bdc4fe5476e2d16c34448785359d98124e205fc176f8a03ef7c352b"
        "e05d35357753bde7ba079d4d8d256f3f396b385ddd54980434f692c3a4818edf"
        "edac370260603c73fdf1615e6213209239338ca5f5ed83e6f8a94709eeba0f41"
        "1f4cde4b98ffc43eeb3d57a77752c08c1030e440275ea6ea92ac835714d2bbc2"
        "c8616b953ac1422d99c07d249a10c60c06ebe2a65fe38d9f5ee708ed05fd438a"
        "aa04fe71e8ce4b7f0bceae0e4286ff0eebe821e2f757d952cc7f52253527a42a"
        "4189c7bc089f45d19093b6b9327cd3a3b9c765468fa5ad7da984d02afbf08132"
        "5727881e1989d5719a97f60b3707fe2196c6c1f5b1a9d3c785a93e0784375cee"
        "9377283fd8a3a01cce6a77dcb4090aac448b764d1631b3043c4daac10ac6df85"
        "2b55fb11e3a141651cfdd4ce983c99e72aa271b96474d0037cc8f48c04082b45"
        "6afdc9a059a12be390d02e67e14198bb48dfc29fd6322f0af43864dbcf990005"
        "53ce60fb5731659cd866a1fcbc40734ec65c0f489589d265a6cb34d0ba838a57"
        "5c750ef7decc96f8507b8984b5362bb0b74073324f4d765701c5b19e99e30470"
        "cd35acce28395dd13d3877a3ad7e90fce1923e787dcfc26ff00304c2f926dbad"
        "951f2a5c4fc25422482f536e6693ca191543b80fca8156ba7c1941576a13002e"
        "5c5ff30c885f7288af9da4796ab67d7cf43a8c97f755a447529e8d67f3053a61"
        "eb794fb6f6bc16f06ccdbb3ae12a90ab5fdf0e411c0d69776e0188dfdb3ac856"
        "d7005048fc5afa61f00120fd348345cbac8d2f1982ca0dd3cd46119cb626dceb"
        "a20c546ac9bcc07f9944e72309a0caa71e4c97263a6a212489e3e682a986659f"
        "f82148c1718bf8ad922971f67268ab77c2f82758e4aed6ad0f557fc7f7487420"
        "d9a854355c85063a2a1a471ade271629883d50d3a39616d39b31e77d84cb2112"
        "ae353f9cafe3d33692e12bc5939d56eb8429dbbbdce45eea710e537ded9e1fad"
        "fd554cc4ac7e87c02edefd61cd04e3a9aca53138494ddcfcc88757cb39f9bc24"
        "f510fea65935d02dad9ad97bcab1d2aacecf7fddde04e5d7dfc1d135c87894bf"
        "1e3ca7bd3ef7004ee46e3fbed3d1a883e566bd14b47c45fabc13c26ea705bef4"
        "8f883e1d1b2391df60b7d81f57ffc51ecadb339263a63c98314db582295e2be6"
        "a20f8d40016e43d4aa72694f602fed7f09fc167be11dffbe492452d6a898c8f6"
        "8e4bd3a57b108dc8abf57199883de277ce01e5a508dbad06d19924827c6ba409"
        "070d66a188692c9d03297b2dab61fb41a46fa20a7d2aa1dc55174cca5d6812ba"
        "f6e1110bf871d64753da484dd82b7e84dac926ae07ec08666d33ffdc4a02bbc2"
        "b4ae2a9adc2dca1435f45866fd992df386bf329e9f8dcc0f9b6788737ab2f412"
        "36a2bc0a58ed8d9e2b8cdf4dcdad8a71692d5a5b65511f5b5a7d1f4ea4b004f5"
        "9575c66dd75442493515f3021d49ec6120"
    ),
    leaf_index=5,
    leaf_height=252,
)
