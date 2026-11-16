// SPDX-License-Identifier: Apache-2.0
//! CuPy CArray ABI from live device-view metadata.
//! Layout source: cupy/_core/include/cupy/carray.cuh:229-240,462-467.
//! The caller must independently establish view ownership, shape and strides.
use super::{plan::Slice, submit::Allocation};
use std::collections::HashMap;

/// Only these observed source/image pairs have a same-run proof that the
/// copy loop writes each CIndexer scratch index before any read. Other images
/// are deliberately refused until their own source binding is established.
pub fn copy_indexer_proved(function: &str, rank: usize, source: &str, image: &str) -> bool {
    matches!((function, rank, source, image),
        // S14 (2026-11-07): fast-numerics pkg18m2 step_01 CIndexers, CUPY_FOR then _ind.set(i) (prove_cindexer_pairs.py round 1 on its stage2-residual.json: set_before_get)
        // (also) multirank (L20, 2026-11-07): the 2-rank-thread recording r2a (rank-local CIndexer shapes), proved by
        // prove_cindexer_pairs.py round 1 on its stage2-residual.json (every pair set_before_get).
        ("cupy_bitwise_and__bool_bool_bool", 1,
         "a3d7a67aefc10e6acef79ad4cd49ae4f73c185c7a48d25196a6bb47d63c67907",
         "4bc4bcfacbd549a6edd069b1cc8a944ec831e49682578d751fe0d319ca15b4d8") |
        ("cupy_nonzero_kernel_incomplete_scan", 1,
         "6cc8f3297442f5a9401177f650e290e3632010081b4fa283ca0d02be520495fe",
         "26cd659245f0273f2512b51f6a1da216756549e63fc9d5752a1aed7768964718") |
        ("cupy_less__float32_float32_bool", 1,
         "0ba9ec655b263e76d683d8d100b8f077da9bf704a1031ed5c56cfb1901af5f1d",
         "751be863be33f378c96e8b2666ea9c042ce38ef0919c9520c88ee770e399b428") |
        ("cupy_copy__float64_float32", 1,
         "be5570cb4fbc3ce5eac9328aca8f5002bda18cd50054b88359ef03de4286386f",
         "99dbc67603d0918cdaf0c5dd7a11f303d0338588898807fdf5a86e6c4dc93e18") |
        ("cupy_less__float32_float_bool", 1,
         "78761c66f1e886c15644900bc9354c4bba85844d7044ee01a2a02f767d2ab71d",
         "080e132023ea0be07c8037b9f2eddfd35e29cde2dba2fdaa1378969f84741686") |
        ("cupy_copysign__float_float32_float32", 1,
         "1d0621776a7e7175fcd67f9b7a0bd011156434a6a3a8d0573344f50deaec3610",
         "ee9356d5852f577809c0ba09859d02001eec649fb273accb8e8cbc0540547a69") |
        ("cupy_where__bool_float32_float32_float32", 1,
         "41a0d25adad76b973caa206462ced7e029214b091ef30915a31c6bfb59de0ccf",
         "9457b300d71770d13e0b157ec0096e1008eab0de401cdd4469091a4ba255b792") |
        // 2026-11-07: the 72-step (24 h) radiation- and land-live recording pkg72m2 (r72a/r72b, a test machine) seals
        // with three more set-before-get CIndexers (integ8: its nonzero-scan pair is the S14 entry above, identical hashes) (prove_cindexer_pairs.py round 1 on its stage2-residual.json, 30 words)
        ("cupy_subtract__int32_int_int32", 1,
         "f5ed973c78539870432ae76fa1a8ccbf6b2563193ad2f84381d33e6e92045f10",
         "88dbd1e4cede9c39f0a401f3b0e5908bc1a0fe19e03a31bb03a5bf240c9290a6") |
        ("cupy_copy__int32_int32", 1,
         "dcaec0881f0eb93433c2e1ee7f7d081121531ed204c559a533b0203fc95b1411",
         "ac768e0a8523bfa947458010298be35902a7be292f65a8726b5169e9edb6bba2") |
        // multirank r2a pairs not already listed above
        ("cupy_copy__float32_float32", 1,
         "af09c28b92b03dfb07f811ec29c158edddec615f6a434c8c226f312dbac249b7",
         "5bbcaab3e1da19cebc3d33b873dfaffe4538dcf7ff690960b17d2db187b2a6c0") |
        ("cupy_take", 2,
         "c97c3a1b00755a57aef013e492ea2fcacf862fe6cde140aa747116af352ba00b",
         "d1b50d5ed03e96a3d5cc27a72398ac4eb33a18db95e917a19f5d7743c69bc4f0") |
        ("cupy_take", 2,
         "245b06ce75560e95b5ee6261c0d020e920fb755bb4934918af3ca59d55665aa1",
         "89e4a5e986d7f6c1554c61324a3c09e8848f9ea1b5c08dc89dabc187dd89fbdc") |
        // S01 (convective-3km, 2026-11-07): the 3 km LAM pair (a test machine, s60f/s60g; proof round 1 on s60a), cupy 14.2,
        // CUPY_FOR then _ind.set(i) (prove_cindexer_pairs.py: set_before_get)
        ("cupy_bitwise_and__bool_bool_bool", 1,
         "a3d7a67aefc10e6acef79ad4cd49ae4f73c185c7a48d25196a6bb47d63c67907",
         "4bc4bcfacbd549a6edd069b1cc8a944ec831e49682578d751fe0d319ca15b4d8") |
        ("cupy_copy__bool_int32", 0,
         "d3c3bd40b9c98f79c9f3492f10822c9a6c24256d9c111cadb93b8a66ad8f02f9",
         "6968512264de2417477008d0e49c689d82b722f2f1bbe91c2986d988b52590bc") |
        ("cupy_copy__int32_int64", 0,
         "440fc781a9199d9cfe7243cbbc67a440f345b5c4b964772bc01ce1d57b52c9c3",
         "c098fdd54f0d6551b55babaed5fadf55701cce3593497e06e557620a26ffea55") |
        ("cupy_concatenate_same_size", 0,
         "1ae7615202e9b32e2d3f0b85e552eb80033e5603c5fe2ae8f0682b4abe33ce43",
         "5396101c0ef796bd00442a0a8b1e86918acde77f9349378760dc21c23000d056") |
        ("cupy_nonzero_kernel_incomplete_scan", 1,
         "6cc8f3297442f5a9401177f650e290e3632010081b4fa283ca0d02be520495fe",
         "26cd659245f0273f2512b51f6a1da216756549e63fc9d5752a1aed7768964718") |
        ("cupy_take", 0,
         "c97c3a1b00755a57aef013e492ea2fcacf862fe6cde140aa747116af352ba00b",
         "d1b50d5ed03e96a3d5cc27a72398ac4eb33a18db95e917a19f5d7743c69bc4f0") |
        ("cupy_take", 2,
         "c97c3a1b00755a57aef013e492ea2fcacf862fe6cde140aa747116af352ba00b",
         "d1b50d5ed03e96a3d5cc27a72398ac4eb33a18db95e917a19f5d7743c69bc4f0") |
        ("cupy_less__float32_float32_bool", 1,
         "0ba9ec655b263e76d683d8d100b8f077da9bf704a1031ed5c56cfb1901af5f1d",
         "751be863be33f378c96e8b2666ea9c042ce38ef0919c9520c88ee770e399b428") |
        ("cupy_take", 0,
         "26471bf2ed977d6ef0c5a23ea485ba0e27614912c574e4d0a27a92a9ab0bf314",
         "284bff81f3e8ac3333fc8cd12fd5634f157e3b073099fb93901cbad9e494ec66") |
        ("cupy_take", 0,
         "245b06ce75560e95b5ee6261c0d020e920fb755bb4934918af3ca59d55665aa1",
         "89e4a5e986d7f6c1554c61324a3c09e8848f9ea1b5c08dc89dabc187dd89fbdc") |
        ("cupy_take", 2,
         "245b06ce75560e95b5ee6261c0d020e920fb755bb4934918af3ca59d55665aa1",
         "89e4a5e986d7f6c1554c61324a3c09e8848f9ea1b5c08dc89dabc187dd89fbdc") |
        ("cupy_take", 0,
         "60c2a6b12469ea7b21c6e3ee79747e5099e3a6768bf11135c382d204b7225879",
         "84920ec34a567911725909f649ee18f99713c1ccba8c4a0dd16fb441ff7355ab") |
        ("cupy_copy__int_float32", 0,
         "7328681ca3bb25c08b717ef023d31c7a7c815e4fa2664f5e825322de55a5e498",
         "113a97102b9f1dfbff611d43891d472eaf92bbd33a419a2d12318dbf2b4311e2") |
        ("cupy_copy__int_float32", 1,
         "7328681ca3bb25c08b717ef023d31c7a7c815e4fa2664f5e825322de55a5e498",
         "113a97102b9f1dfbff611d43891d472eaf92bbd33a419a2d12318dbf2b4311e2") |
        ("cupy_copy__float_float32", 1,
         "b83706db0550b875fcb6a39ae6d55ae75bdd6f2d2ec3b342f50294df99683f44",
         "a8d1afe9576f284d7ca764120d2f8b2966a7bbfc3db8e54ba291b53faa15ac57") |
        ("cupy_scatter_update", 2,
         "ac691df5c8affce5583af432d2bbd01103b8d9a0dcbdee9c468714617d494b71",
         "8a21e65d63c81f1994bc5bffebba5a6026fe57567ba7ff5886abd57dc6925180") |
        ("cupy_copy__int64_int32", 1,
         "563b9d4fa7a3dc4bc06fe4a6b8899e853f82341b9a8c16a32620aaf73a450941",
         "c74f2c9dd767934aefde616fed5ddb9bdaffcd987f4eed2b37793a6aad1eea7f") |
        ("cupy_copy__float32_float32", 1,
         "af09c28b92b03dfb07f811ec29c158edddec615f6a434c8c226f312dbac249b7",
         "5bbcaab3e1da19cebc3d33b873dfaffe4538dcf7ff690960b17d2db187b2a6c0") |
        ("cupy_copy__int32_int32", 1,
         "dcaec0881f0eb93433c2e1ee7f7d081121531ed204c559a533b0203fc95b1411",
         "ac768e0a8523bfa947458010298be35902a7be292f65a8726b5169e9edb6bba2") |
        ("cupy_scatter_update", 0,
         "ac691df5c8affce5583af432d2bbd01103b8d9a0dcbdee9c468714617d494b71",
         "8a21e65d63c81f1994bc5bffebba5a6026fe57567ba7ff5886abd57dc6925180") |
        ("cupy_scatter_update", 2,
         "ac691df5c8affce5583af432d2bbd01103b8d9a0dcbdee9c468714617d494b71",
         "8a21e65d63c81f1994bc5bffebba5a6026fe57567ba7ff5886abd57dc6925180") |
        ("cupy_copy__bool_float64", 1,
         "9ddb1e1b2d924ff063d5cdae2d22d87f0981aff2924a0b361926b78ace2001cd",
         "2b071a7ad79b58a4942ea1b33f67d4135d03d351c69f5d35ae609832023d9b61") |
        ("cupy_copy__int_int32", 0,
         "635c74236b9aacbb82e23481b481b656ad95e5c308debef9054b91fe5c109779",
         "9360644393a87b5cf1c6a09e72f7a48bdb0b0ed7ef03f4f8087848ce9ef84f7e") |
        ("cupy_copy__int64_int32", 1,
         "563b9d4fa7a3dc4bc06fe4a6b8899e853f82341b9a8c16a32620aaf73a450941",
         "c74f2c9dd767934aefde616fed5ddb9bdaffcd987f4eed2b37793a6aad1eea7f") |
        ("cupy_take", 0,
         "170db0c2dc6670fbb9f004d5c2aa5ba824d47a8a671fbf294f8503f53d118075",
         "aedcbff4f2f83d461cd1399e408f262b3f2140a1f36bba0a6128d11dce4ecd77") |
        ("cupy_copy__float_float32", 0,
         "25701c7e9d17fa8dcdfd1c8d7e86f7a6ec8fa83fc2a38cff079adcb5ad3f2ed8",
         "35ac78ae7e14732ae6d30e3f3bb88f1476cea54afb0efca4ba010fbbb19c59a4") |
        ("cupy_scatter_update", 0,
         "a616abb013da7decd26055da4ad9db0216267078bc34c5a9bbbd00cdaaf71986",
         "8777ec05fe78c0d14fd5afb3c259881da1ee3965ddd99d8653aa1274bf03f2ff") |
        ("cupy_subtract__int32_int_int32", 1,
         "f5ed973c78539870432ae76fa1a8ccbf6b2563193ad2f84381d33e6e92045f10",
         "88dbd1e4cede9c39f0a401f3b0e5908bc1a0fe19e03a31bb03a5bf240c9290a6") |
        ("cupy_copy__int32_int32", 0,
         "4f84ca514f9ffa116f4f0bcce5679f1972e8b0ed48b1b487e9af24764798e5a1",
         "35975cc76ed542581bd4351ca5b792628e046cd69233d44cbbee2a87f9453855") |
        ("cupy_fill", 0,
         "bcd4027bac7ec4638b580573887311965847988abbf95209a7050b89f4dd22ca",
         "c3ed6c64b0991133e7815e0035cfc0ab30cf718f248da6e62b1fd0443a23bb30") |
        ("cupy_fill", 0,
         "32c198abfba5f4689e72a1a03728de985bb51fde99e4df9514f44f9901464bbc",
         "a1b265dd0e8a4d46e3b523a136f3c38faa6b13e279007285c05775fc2579badd") |
        ("cupy_not_equal__int32_int_bool", 0,
         "9380153115d9eff2ea6a38e189bd47d5e218391b6af7b4987f281a1eee488aa9",
         "708c5eaf6a6cc7a5ea8cbc5ea0048f4a7e2f57f95e75f368550f2bdfff82e788") |
        ("cupy_fill", 1,
         "b73d5a6d435c3c8fe339a00bd6c1775c540403ce3a81ec66eaad5c89c3ee31f1",
         "8974e5cdb6426a07d3891be17dd1a00eacc40db8ca482f9ac692053e4b4a0b76") |
        ("cupy_arange__int_int_int32", 1,
         "1feaaa08e21628bb8c49243dc6c963d57df7a55f99f4a937353a58085c10a722",
         "a3732af4d254dc31f23c2bfbf96476a4f411364f1a3a8d0d883936fb44d9efc7") |
        ("cupy_scatter_update", 0,
         "b934257b958ce5f4c3e560657225f63e2e0483f80965edb73229f1149f252867",
         "e551948ecb97bf083f867bd94b756f152d972b2e5c9d9396c196b7934d7fa919") |
        ("cupy_scatter_update", 1,
         "b934257b958ce5f4c3e560657225f63e2e0483f80965edb73229f1149f252867",
         "e551948ecb97bf083f867bd94b756f152d972b2e5c9d9396c196b7934d7fa919") |
        ("cupy_less__int32_int_bool", 0,
         "4129efce3190ff97655047cf7d588efb516805f8cb406386a1fcc4f8e8617f21",
         "865f2e6e505212a4afc53f2fec296e84ac93731d90ec2b7397940d7f979d4d3a") |
        ("cupy_less__int32_int_bool", 1,
         "4129efce3190ff97655047cf7d588efb516805f8cb406386a1fcc4f8e8617f21",
         "865f2e6e505212a4afc53f2fec296e84ac93731d90ec2b7397940d7f979d4d3a") |
        ("cupy_where__bool_int_int32_int32", 1,
         "54465abedc805f545709e526636349707157cadff8e16ec01b4cc587f4372a95",
         "ee44863f288aee7827646521bd5fd71448d105133e28548b8f026c903760d7fe") |
        ("cupy_copy__int32_int64", 1,
         "da7435c6d654a9bff268f8f17cebc3cdd1e05d5fc6084cb35f0cf7fa560eaa9c",
         "b7efc2a11bf81353443eb97a952db1b2191b59ba98b19b4d0782e364fa259d35") |
        ("cupy_greater_equal__int64_int_bool", 0,
         "45edd1049e3ad1784a1d9e4581d1ba54f06c91f67cb0da338249e02fa8cd2a40",
         "6c61fd817bdda7258a7b6a2544067377038da44432c68a88504c15ccbd5400bb") |
        ("cupy_greater_equal__int64_int_bool", 1,
         "45edd1049e3ad1784a1d9e4581d1ba54f06c91f67cb0da338249e02fa8cd2a40",
         "6c61fd817bdda7258a7b6a2544067377038da44432c68a88504c15ccbd5400bb") |
        ("cupy_less__int64_int_bool", 0,
         "5b6f8cc7434a02e988563efd3cd02ca2424c0374bd3fa49e98ce43f69b580c02",
         "fdb30b5f58b4046811150b2197929b401db7da5f7cf73f6548f484411e8b294c") |
        ("cupy_less__int64_int_bool", 1,
         "5b6f8cc7434a02e988563efd3cd02ca2424c0374bd3fa49e98ce43f69b580c02",
         "fdb30b5f58b4046811150b2197929b401db7da5f7cf73f6548f484411e8b294c") |
        ("cupy_where__bool_int64_int_int64", 0,
         "e57f91743ef17cee76e2cae6092df880ff7b543b3cb572ff57ab1b3ff25feeca",
         "d0e620d522c5f834992069edc32c7b1988db48282424afa03b2bb4abedf24259") |
        ("cupy_where__bool_int64_int_int64", 1,
         "e57f91743ef17cee76e2cae6092df880ff7b543b3cb572ff57ab1b3ff25feeca",
         "d0e620d522c5f834992069edc32c7b1988db48282424afa03b2bb4abedf24259") |
        ("cupy_take", 0,
         "59527e1aecbb98a1f65d5e484e55c8717196c8af4d64e303e2c2ca2534a6e0ed",
         "4360740ac1549f02cd95b70aa28197f34b1fb2b75811cfe429f11e48cc1e7106") |
        ("cupy_take", 1,
         "59527e1aecbb98a1f65d5e484e55c8717196c8af4d64e303e2c2ca2534a6e0ed",
         "4360740ac1549f02cd95b70aa28197f34b1fb2b75811cfe429f11e48cc1e7106") |
        ("cupy_where__bool_int32_int_int32", 1,
         "c41f84b6ffcabc243f1dc91fd215629abbc5880fabda77b726fd3ff6ad432fb2",
         "2411a09650ff18caa632bc278ba017d9b1c61e7d20ebf054f06d50b51238ff54") |
        ("cupy_take", 0,
         "e64c65dddbe3c50e4f2a3d0f13c15d0445b855419495bfdd0466782645cfeaf5",
         "339d7089418aec4eaad57d6870533f5ed40d5eb52590af2299da8090a4b078e9") |
        ("cupy_take", 1,
         "e64c65dddbe3c50e4f2a3d0f13c15d0445b855419495bfdd0466782645cfeaf5",
         "339d7089418aec4eaad57d6870533f5ed40d5eb52590af2299da8090a4b078e9") |
        ("cupy_equal__int64_int64_bool", 1,
         "cff9e009aaa30f4ec923b67676e59a584e073a2a21c01a72370a702dda6491fd",
         "16cf93256d29d52d2041d28079d021a9d6d324d4e6620d422b81d79f1816a139") |
        ("cupy_minimum__int64_int_int64", 0,
         "ebdab02613b1f170448ba479c827814cdbbdc10259655cd137dbe4b4a4f89f7a",
         "cefa01dc2e791632ae758dbbf56c3262d295f7fc8b156a962d9d97e290546229") |
        ("cupy_minimum__int64_int_int64", 1,
         "ebdab02613b1f170448ba479c827814cdbbdc10259655cd137dbe4b4a4f89f7a",
         "cefa01dc2e791632ae758dbbf56c3262d295f7fc8b156a962d9d97e290546229") |
        ("cupy_take", 0,
         "9c7eccbd81c1c83052693c2ce2447cedb8e4b06478118308b3a2ae5b2d0c03ab",
         "1f97c7f7a0eb0424e705546eb839ecc288bb11277ef0790c6acdecef21ec91c8") |
        ("cupy_take", 2,
         "9c7eccbd81c1c83052693c2ce2447cedb8e4b06478118308b3a2ae5b2d0c03ab",
         "1f97c7f7a0eb0424e705546eb839ecc288bb11277ef0790c6acdecef21ec91c8") |
        ("cupy_concatenate_same_size", 0,
         "4a66cfe30903320192f854458f2cb1b07f23506c864c1c5863a4db9dfb296160",
         "82af2f9c5466c4d839daebc05a7fc5497952d5c47cbe2ea8d059c4211b09e555") |
        // a test machine (2026-11-07): re-recording of the radiation- and land-live pkg18m2 (a test machine was lost with a test machine), r18a/r18b on
        // 2x RTX PRO 6000 WS, cupy 14.2; step_01 CIndexer, CUPY_FOR then _ind.set(i) at source line 14
        // (prove_cindexer_pairs.py round 5 on its stage2-residual.json: set_before_get, 1 word)
        ("cupy_greater_equal__float32_float_bool", 1,
         "84eb57813fba26b37e02d8c5281babe153c5251dd7196bdf9b09515c6f530119",
         "b6a3130ce2b2dffbec83fc6d4d3f6fd959b332f9b55fd7877991013304a6bede") |
        // G2: round 4 (r18a try 3)
        ("cupy_scan_naive", 1,
         "9f3e6aa6b34b128263a03c3b853161dc8a04da24f0171256a3770db17fac78c3",
         "57a6bdb3827c3d8cb07140df2f38544eff07dbadb0a046e39498fbfbe56b6086") |
        ("cupy_where__bool_float_float32_float32", 1,
         "983697d1df74032dcce62c313f249d200f7a21fca6dba159cda4292939b326e2",
         "1beef9c1365c4d09f7c9a9bd54fab92888e54c92cc8712c2cbdb24942eaad823") |
        // G2 (P1 milestone 2) 2026-11-06: the radiation- and land-live 18-step pair (a test machine, r18a/r18b, cupy 14.2),
        // CUPY_FOR then _ind.set(i) (prove_cindexer_pairs.py on the milestone-2 stage2-residual.json, three rounds)
        ("cupy_less_equal__float32_float_bool", 1,
         "fd2bfc498fc7e47c88852ef5f6ad3a8307b6aa23faf1518cc256fdfbe1dc5bcc",
         "67422a08d9f719ea8315b7e281a514902dd3d1968cec98fc9565d15e0e61db52") |
        ("cupy_invert__bool_bool", 1,
         "7fcb0f9beec0709824c09ee2e39437964f54eb8d43a3443e88032832ffb1efc5",
         "bfe1f2cb0fe09e719be2d0cd5bb180309a255ce16d3e955ba51d6be91f208d13") |
        ("cupy_greater__float32_float_bool", 1,
         "f1d0821b0f5244a5f676accb414fa3a96475dd019a3118c209de7a19bb76204f",
         "bf478f5fbacc74d9ec44c3b54280ad6b0be60c551d66175e3e6548a342cfef0e") |
        ("cupy_multiply__float32_float_float32", 1,
         "c1cec422bab3574a68bd3b7b318616871b2d192518a891fa0fd5c104ac0b0646",
         "2b3dcd8f2d7a2388523d1ddd0c7e387c041313e4c7331cf8823fc8c94bf1432c") |
        ("cupy_take", 3,
         "26471bf2ed977d6ef0c5a23ea485ba0e27614912c574e4d0a27a92a9ab0bf314",
         "284bff81f3e8ac3333fc8cd12fd5634f157e3b073099fb93901cbad9e494ec66") |
        ("cupy_take", 1,
         "60c2a6b12469ea7b21c6e3ee79747e5099e3a6768bf11135c382d204b7225879",
         "84920ec34a567911725909f649ee18f99713c1ccba8c4a0dd16fb441ff7355ab") |
        ("cupy_true_divide__float32_float_float32", 1,
         "42a0b41adf19085500d6cbf41e3a727e231bd1f19c27c8e4a85e88b64affda51",
         "26c64a807650a898030cb5c8f222be22c978cc854a7f9556c586752c8ec2003b") |
        ("cupy_multiply__float32_float32_float32", 1,
         "52c93e355e8912097c71ead00569a69ce57654c3af64837513efd1bbecf1d845",
         "f19708f3910d39ba30729cd18a6dbeb21dce5e7b5011464231c28bae7955fc64") |
        ("cupy_copy__float32_float32", 1,
         "5d98302ec14db274cfe0a70f16e2adbaa85368a65d9fff2ff3cca8020e755a71",
         "c3d46f5a2823e7d7acc545f7b57a6dc304815748da5cae3bbeb8173c9f0d057e") |
        ("cupy_copy__float_float32", 1,
         "25701c7e9d17fa8dcdfd1c8d7e86f7a6ec8fa83fc2a38cff079adcb5ad3f2ed8",
         "35ac78ae7e14732ae6d30e3f3bb88f1476cea54afb0efca4ba010fbbb19c59a4") |
        ("cupy_concatenate_same_size", 1,
         "1ae7615202e9b32e2d3f0b85e552eb80033e5603c5fe2ae8f0682b4abe33ce43",
         "5396101c0ef796bd00442a0a8b1e86918acde77f9349378760dc21c23000d056") |
        ("cupy_bsum_shfl", 1,
         "ce6041f43cdefcc5f1e1a4d72987be3b230c6a87e0d5dc5bb9dd29f39d9d22ad",
         "be54ea8924be3a37da5f0641ec52555078d67d9ed72945c73f3de6d7803f1bc4") |
        ("cupy_copy__int64_int64", 1,
         "90a3c8f2f6609e9810c253a25456a90a7a4ed74276018ac72fcba3affd417295",
         "28d34910279d976b76e6b1d59818b6b3147e56af1d215735deda6c7a067d6b21") |
        ("cupy_copy__float32_float32", 2,
         "d3786cde79a19e5e8f592a3c3b12b0713cc6563409c1c23ae03ebd387b89b685",
         "4155090a9c8587be5cd64c8dd0481ab110f201cc2ad42a1450c785537ac7be9b") |
        ("cupy_take", 2,
         "170db0c2dc6670fbb9f004d5c2aa5ba824d47a8a671fbf294f8503f53d118075",
         "aedcbff4f2f83d461cd1399e408f262b3f2140a1f36bba0a6128d11dce4ecd77") |
        ("cupy_scatter_update", 2,
         "a616abb013da7decd26055da4ad9db0216267078bc34c5a9bbbd00cdaaf71986",
         "8777ec05fe78c0d14fd5afb3c259881da1ee3965ddd99d8653aa1274bf03f2ff") |
        ("cupy_copy__int64_int64", 1,
         "90a3c8f2f6609e9810c253a25456a90a7a4ed74276018ac72fcba3affd417295",
         "28d34910279d976b76e6b1d59818b6b3147e56af1d215735deda6c7a067d6b21") |
        ("cupy_copy__float32_float32", 1,
         "d8b7e7aa77ab24585768c9be20c0e6a262fdebab9b36841f19a226c28c738c92",
         "1a7a9d8779c54da648a1d4be9b47f2d9a22c5e45bde6b199b4d5f1e5876ad6dc") |
        ("cupy_copy__float32_float32", 2,
         "7e3ae7d949db742c3413572c78d1e401cd7bccef20f1979380962ea6d3020dec",
         "7f0967d8578d8b7e4aa6a94fe69f5628809ef5fe82e57f917299587fa9cbaab0") |
        // continuation 17:20Z: observed hrrrv5 v25 source, CUPY_FOR then _ind.set(i) (prove_cindexer_pairs.py)
        ("cupy_copy__float32_float32", 3,
         "bd1a522dede272141b548ba138a4130f5f54555cabc340e8521176db37ca51ee",
         "c26b524d8692d9dd45969b6b2290787b0576793e8f103aba36bb9865a6ab3a9c") |
        // continuation 17:20Z: observed hrrrv5 v25 source, CUPY_FOR then _ind.set(i) (prove_cindexer_pairs.py)
        ("cupy_concatenate_same_size", 1,
         "4a66cfe30903320192f854458f2cb1b07f23506c864c1c5863a4db9dfb296160",
         "82af2f9c5466c4d839daebc05a7fc5497952d5c47cbe2ea8d059c4211b09e555") |
        // continuation 17:20Z: observed hrrrv5 v25 source, CUPY_FOR then _ind.set(i) (prove_cindexer_pairs.py)
        ("cupy_copy__float32_float32", 2,
         "7df7615e10655076a9634ac7221e8f615310cb91b8aa83340a98fdc25c49d716",
         "d8d380d16b88ac3692eb2762582f09b60c20cf5b1f8518a103848ab13a9b4844") |
        // continuation 17:20Z: observed hrrrv5 v25 source, CUPY_FOR then _ind.set(i) (prove_cindexer_pairs.py)
        ("cupy_copy__int_int32", 1,
         "b5eb2d79161d7489c0f2947cc9e13a291c5eb359f2b1ea00a21ca5f4a00a23e3",
         "2850b052b44633752e9a5e300383300ff848bdd206803e7e132c45e8390cdbd9") |
        // continuation 17:20Z: observed hrrrv5 v25 source, CUPY_FOR then _ind.set(i) (prove_cindexer_pairs.py)
        ("cupy_copy__bool_int32", 1,
         "e702b12b297dd40c46cfec751a3c1e2f8a3c1e7994ec9137c2f36944882a010e",
         "f7f9c4849c37097a693d1efd4d0151a7cf99065087d5e6f51452b59a4394b160") |
        // continuation 19:40Z: observed hrrrv5 v26 elementwise sources, CUPY_FOR then _ind.set(i)
        // (prove_cindexer_pairs.py, tmp/v26-cindexer-pairs-new.json)
        // P1 (ad/phys) 2026-11-04: observed hrrrv5 p1a (a test machine, cupy 14.2) sources, set before get
        // (prove_cindexer_pairs.py on seal-p1a/stage2-residual.json; the runner re-checks the source line)
        // P1 sealer-output-steps 2026-11-04: the 18-step hrrrv5 pair's output steps (9, 18; a test machine r18a, cupy 14.2),
        // CUPY_FOR then _ind.set(i) at source line 13 (prove_cindexer_pairs.py on the 18-step stage2-residual.json)
        ("cupy_copy__int_int32", 1,
         "635c74236b9aacbb82e23481b481b656ad95e5c308debef9054b91fe5c109779",
         "9360644393a87b5cf1c6a09e72f7a48bdb0b0ed7ef03f4f8087848ce9ef84f7e") |
        ("cupy_copy__uint32_uint32", 2,
         "bffeb42ade6b52730e0456e2a2529c5655a6bc967d8b5a8aba3f17421c563ab8",
         "8b1df6fd0fb8a6aaf8ee7e01150e34113e4411517292d49e98c9bf4b0c2e7161") |
        ("cupy_fill", 1,
         "86b1461b03642bc041a5c4bac17767dc0f314d71a623c1c13d60df528616630f",
         "cc222e806b4461415f60566d6af2e78e5e5b41efbd0414121333cc0ddbd894a9") |
        ("cupy_not_equal__int32_int_bool", 1,
         "9380153115d9eff2ea6a38e189bd47d5e218391b6af7b4987f281a1eee488aa9",
         "708c5eaf6a6cc7a5ea8cbc5ea0048f4a7e2f57f95e75f368550f2bdfff82e788") |
        ("cupy_fill", 1,
         "cda81f755f42ca4453c44824f69491681608ed3d89dc281c97711696c4c71539",
         "4ed8be5489bafff2af82b262b5b6b4cec5c4355f7174dd09ad969e4e1fc7c623") |
        ("cupy_copy__float32_float64", 1,
         "cb7ec9c7eda767dae6a2563a1cb90217dc4ee385b2362dc8aa9515c80a8ed459",
         "3884b94def2b066e700710985443815639bfc391793009d8ca5c5d56f36de934") |
        ("cupy_copy__float64_float64", 1,
         "b5d490a1f287f77233095dd6eb96688be5f9874aa1e98b20d2d9627366dfd73b",
         "aeaee243b103f81d4483040b9b946e3a17d006f933d42cf2b2a2be293dc14f8c") |
        ("cupy_copy__int32_float64", 1,
         "eca19925e99a199e4e6f5b6173b70732dcfe07402f28269f86de44fa28050da2",
         "629ef45d06f8520a2d815a43d4d94ad6c651bece9ba4ccfe33fcc5007fcbe6c0") |
        ("cupy_maximum__float_float32_float32", 1,
         "d97f90ea561a2de0e788b402efeb15218b34869b306a98a23fe8b8a7f6373163",
         "bf0eb2142f1d8db29682f25ff8c52eb69bf8c6df550667e1aeba050c9ba2c87c") |
        ("cupy_less__float32_float32_bool", 1,
         "49ad50df664115d357392c7cc4b3cf950aad444cc1a20f0425feaf5604fb0797",
         "ab7eb399b43594fda56e6038ca209c72b48dbde6d29a40790826348afad6908c") |
        ("cupy_copy_where__float32_bool_float32", 1,
         "b5c45fd049203b335773dfc8065e3369371cb718fb3785bfaac45905c12d7cc9",
         "823bb18459c01413cea3be386eedc4c30975ac71a302ef3a5c65f439b9f0345c"))
}

/// CuPy reduction kernels whose input CIndexer scratch is set (`_in_ind.set(_j);`) on the line before its
/// first read, with no earlier read (continuation 17:25Z; observed hrrrv5 v25 sources, prove_cindexer_pairs.py).
pub fn reduction_in_indexer_proved(function: &str, rank: usize, source: &str, image: &str) -> bool {
    matches!((function, rank, source, image),
        // P1 (ad/phys) 2026-11-04: observed hrrrv5 p1a (a test machine, cupy 14.2) sources, set before get
        // (prove_cindexer_pairs.py on seal-p1a/stage2-residual.json; the runner re-checks the source line)
        ("hex_transport_species_words", 1,
         "987cda0e588615d9033987545f37036f5f602b9e2ec4c88bf79d7c088d7b0249",
         "80a74f2b2b3bfe01d2d500d96c9b80458ee76a0247f18136dd34e75d2f7ce446") |
        ("hex_transport_species_levels", 1,
         "ee05d5dba309443a7b4c1795e21c390784b1cfb61dfb847cbf192c311436cb7d",
         "f3c10d5535154415ba5f03c62f7ae8b9aa561148b6b341498dd0bba01764d6fa") |
        ("hex_transport_species_words", 2,
         "987cda0e588615d9033987545f37036f5f602b9e2ec4c88bf79d7c088d7b0249",
         "80a74f2b2b3bfe01d2d500d96c9b80458ee76a0247f18136dd34e75d2f7ce446") |
        ("hex_transport_species_levels", 2,
         "ee05d5dba309443a7b4c1795e21c390784b1cfb61dfb847cbf192c311436cb7d",
         "f3c10d5535154415ba5f03c62f7ae8b9aa561148b6b341498dd0bba01764d6fa"))
}

/// CuPy CIndexer contains size, shape, then scratch index words. The selected
/// copy kernels overwrite the scratch before reading it; start it at zero.
pub fn encode_copy_indexer(shape: &[i64]) -> Result<Vec<u8>, String> {
    if shape.len() > 64 { return Err("invalid CIndexer rank".into()); }
    let mut size = 1_i64;
    for &extent in shape {
        if extent < 0 { return Err("negative CIndexer extent".into()); }
        size = size.checked_mul(extent).ok_or("CIndexer size overflow")?;
    }
    let mut out = size.to_le_bytes().to_vec();
    for &extent in shape { out.extend_from_slice(&extent.to_le_bytes()); }
    out.resize(8 + shape.len() * 16, 0);
    Ok(out)
}

/// Bytes with an empty pointer slot, to be patched from the current CUDA
/// allocation during submission. Shape and strides are current host values.
pub fn encode(target: Option<Slice>, shape: &[i64], strides: &[i64], item_bytes: usize,
              allocations: &HashMap<u64, Allocation>) -> Result<Vec<u8>, String> {
    if shape.len() != strides.len() || shape.len() > 64 || item_bytes == 0 {
        return Err("invalid CArray rank or item size".into());
    }
    let mut size = 1_i64;
    let mut lo = 0_i128;
    let mut hi = 0_i128;
    let mut empty = false;
    for (&extent, &stride) in shape.iter().zip(strides) {
        if extent < 0 { return Err("negative CArray extent".into()); }
        size = size.checked_mul(extent).ok_or("CArray size overflow")?;
        empty |= extent == 0;
        if extent > 0 {
            let delta = i128::from(extent - 1).checked_mul(i128::from(stride))
                .ok_or("CArray stride overflow")?;
            lo = lo.checked_add(delta.min(0)).ok_or("CArray stride overflow")?;
            hi = hi.checked_add(delta.max(0)).ok_or("CArray stride overflow")?;
        }
    }
    match target {
        None if empty => (),
        None => return Err("nonempty CArray has a null pointer".into()),
        Some(target) => {
            let alloc = allocations.get(&target.allocation).ok_or("unknown CArray allocation")?;
            if target.offset >= alloc.bytes { return Err("CArray pointer outside allocation".into()); }
            if !empty {
                let offset = i128::try_from(target.offset).map_err(|_| "CArray offset overflow")?;
                let end = offset.checked_add(hi).and_then(|n| n.checked_add(item_bytes as i128))
                    .ok_or("CArray address overflow")?;
                if offset.checked_add(lo).ok_or("CArray address overflow")? < 0 || end > alloc.bytes as i128 {
                    return Err("CArray view exceeds allocation".into());
                }
            }
        },
    }
    let mut out = vec![0; 8];
    out.extend_from_slice(&size.to_le_bytes());
    for &value in shape { out.extend_from_slice(&value.to_le_bytes()); }
    for &value in strides { out.extend_from_slice(&value.to_le_bytes()); }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn live_shape_strides_and_scalar_layout() {
        let a = HashMap::from([(1, Allocation { id: 1, address: 1000, bytes: 128 })]);
        let target = Some(Slice { allocation: 1, offset: 24 });
        let bytes = encode(target, &[2, 3], &[-12, 4], 4, &a).unwrap();
        let words: Vec<_> = bytes.chunks_exact(8).map(|v| i64::from_le_bytes(v.try_into().unwrap())).collect();
        assert_eq!(words, [0, 6, 2, 3, -12, 4]);
        assert_eq!(encode(target, &[], &[], 4, &a).unwrap().len(), 16);
        assert_eq!(encode(None, &[0, 2], &[8, 4], 4, &a).unwrap().len(), 48);
    }
    #[test]
    fn refuses_bad_shape_stale_pointer_and_out_of_bounds_stride() {
        let a = HashMap::from([(1, Allocation { id: 1, address: 1000, bytes: 32 })]);
        let target = Some(Slice { allocation: 1, offset: 0 });
        assert!(encode(target, &[2], &[-4], 4, &a).is_err());
        assert!(encode(target, &[9], &[4], 4, &a).is_err());
        assert!(encode(target, &[-1], &[4], 4, &a).is_err());
        assert!(encode(target, &[i64::MAX, 2], &[0, 0], 4, &a).is_err());
        assert!(encode(None, &[1], &[4], 4, &a).is_err());
        assert!(encode(Some(Slice { allocation: 2, offset: 0 }), &[1], &[4], 4, &a).is_err());
    }
    #[test]
    fn indexer_scratch_is_zero_and_only_observed_source_images_are_accepted() {
        let bytes = encode_copy_indexer(&[2, 3]).unwrap();
        let words: Vec<_> = bytes.chunks_exact(8).map(|v| i64::from_le_bytes(v.try_into().unwrap())).collect();
        assert_eq!(words, [6, 2, 3, 0, 0]);
        assert!(encode_copy_indexer(&[-1]).is_err());
        assert!(copy_indexer_proved("cupy_copy__float32_float32", 2,
            "7e3ae7d949db742c3413572c78d1e401cd7bccef20f1979380962ea6d3020dec",
            "7f0967d8578d8b7e4aa6a94fe69f5628809ef5fe82e57f917299587fa9cbaab0"));
        assert!(!copy_indexer_proved("cupy_copy__float32_float32", 1,
            "7e3ae7d949db742c3413572c78d1e401cd7bccef20f1979380962ea6d3020dec",
            "7f0967d8578d8b7e4aa6a94fe69f5628809ef5fe82e57f917299587fa9cbaab0"));
    }
}
