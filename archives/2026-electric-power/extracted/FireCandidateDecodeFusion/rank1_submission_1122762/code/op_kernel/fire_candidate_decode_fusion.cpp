#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "fire_candidate_decode_fusion_tiling.h"
#include "tiling_key_fire_candidate_decode_fusion.h"

//548: immutable deviceGM index source; explicitPOD initializer, no per-launch generation.
alignas(256) static __gm__ const uint32_t FireSmallIndexGM548[512] = {
0U, 24U, 48U, 72U, 96U, 120U, 144U, 168U, 192U, 216U, 240U, 264U, 288U, 312U, 336U, 360U,
384U, 408U, 432U, 456U, 480U, 504U, 528U, 552U, 576U, 600U, 624U, 648U, 672U, 696U, 720U, 744U,
4U, 28U, 52U, 76U, 100U, 124U, 148U, 172U, 196U, 220U, 244U, 268U, 292U, 316U, 340U, 364U,
388U, 412U, 436U, 460U, 484U, 508U, 532U, 556U, 580U, 604U, 628U, 652U, 676U, 700U, 724U, 748U,
8U, 32U, 56U, 80U, 104U, 128U, 152U, 176U, 200U, 224U, 248U, 272U, 296U, 320U, 344U, 368U,
392U, 416U, 440U, 464U, 488U, 512U, 536U, 560U, 584U, 608U, 632U, 656U, 680U, 704U, 728U, 752U,
12U, 36U, 60U, 84U, 108U, 132U, 156U, 180U, 204U, 228U, 252U, 276U, 300U, 324U, 348U, 372U,
396U, 420U, 444U, 468U, 492U, 516U, 540U, 564U, 588U, 612U, 636U, 660U, 684U, 708U, 732U, 756U,
16U, 40U, 64U, 88U, 112U, 136U, 160U, 184U, 208U, 232U, 256U, 280U, 304U, 328U, 352U, 376U,
400U, 424U, 448U, 472U, 496U, 520U, 544U, 568U, 592U, 616U, 640U, 664U, 688U, 712U, 736U, 760U,
20U, 44U, 68U, 92U, 116U, 140U, 164U, 188U, 212U, 236U, 260U, 284U, 308U, 332U, 356U, 380U,
404U, 428U, 452U, 476U, 500U, 524U, 548U, 572U, 596U, 620U, 644U, 668U, 692U, 716U, 740U, 764U,
0U, 20U, 40U, 60U, 80U, 100U, 120U, 140U, 160U, 180U, 200U, 220U, 240U, 260U, 280U, 300U,
320U, 340U, 360U, 380U, 400U, 420U, 440U, 460U, 480U, 500U, 520U, 540U, 560U, 580U, 600U, 620U,
4U, 24U, 44U, 64U, 84U, 104U, 124U, 144U, 164U, 184U, 204U, 224U, 244U, 264U, 284U, 304U,
324U, 344U, 364U, 384U, 404U, 424U, 444U, 464U, 484U, 504U, 524U, 544U, 564U, 584U, 604U, 624U,
8U, 28U, 48U, 68U, 88U, 108U, 128U, 148U, 168U, 188U, 208U, 228U, 248U, 268U, 288U, 308U,
328U, 348U, 368U, 388U, 408U, 428U, 448U, 468U, 488U, 508U, 528U, 548U, 568U, 588U, 608U, 628U,
12U, 32U, 52U, 72U, 92U, 112U, 132U, 152U, 172U, 192U, 212U, 232U, 252U, 272U, 292U, 312U,
332U, 352U, 372U, 392U, 412U, 432U, 452U, 472U, 492U, 512U, 532U, 552U, 572U, 592U, 612U, 632U,
16U, 36U, 56U, 76U, 96U, 116U, 136U, 156U, 176U, 196U, 216U, 236U, 256U, 276U, 296U, 316U,
336U, 356U, 376U, 396U, 416U, 436U, 456U, 476U, 496U, 516U, 536U, 556U, 576U, 596U, 616U, 636U,
0U, 128U, 256U, 384U, 512U, 4U, 132U, 260U, 388U, 516U, 8U, 136U, 264U, 392U, 520U, 12U,
140U, 268U, 396U, 524U, 16U, 144U, 272U, 400U, 528U, 20U, 148U, 276U, 404U, 532U, 24U, 152U,
280U, 408U, 536U, 28U, 156U, 284U, 412U, 540U, 32U, 160U, 288U, 416U, 544U, 36U, 164U, 292U,
420U, 548U, 40U, 168U, 296U, 424U, 552U, 44U, 172U, 300U, 428U, 556U, 48U, 176U, 304U, 432U,
560U, 52U, 180U, 308U, 436U, 564U, 56U, 184U, 312U, 440U, 568U, 60U, 188U, 316U, 444U, 572U,
64U, 192U, 320U, 448U, 576U, 68U, 196U, 324U, 452U, 580U, 72U, 200U, 328U, 456U, 584U, 76U,
204U, 332U, 460U, 588U, 80U, 208U, 336U, 464U, 592U, 84U, 212U, 340U, 468U, 596U, 88U, 216U,
344U, 472U, 600U, 92U, 220U, 348U, 476U, 604U, 96U, 224U, 352U, 480U, 608U, 100U, 228U, 356U,
484U, 612U, 104U, 232U, 360U, 488U, 616U, 108U, 236U, 364U, 492U, 620U, 112U, 240U, 368U, 496U,
624U, 116U, 244U, 372U, 500U, 628U, 120U, 248U, 376U, 504U, 632U, 124U, 252U, 380U, 508U, 636U,
};
static_assert(sizeof(FireSmallIndexGM548) == 2048, "exact512u32");

//556: exact C3 Host payload; all three tables remain immutable GM sources.
#define FIRE_P8O_TU_float 1
#define FIRE_P8O_TU_half 2
#define FIRE_P8O_TU_I(t) FIRE_P8O_TU_##t
#define FIRE_P8O_TU(t) FIRE_P8O_TU_I(t)
#if defined(DTYPE_RAW_PRED) && FIRE_P8O_TU(DTYPE_RAW_PRED) == 2
#define FIRE_P8O_HALF_IMAGE 1
#define FIRE_P8O_INDEX(v) ((v) / 2U)
#else
#define FIRE_P8O_HALF_IMAGE 0
#define FIRE_P8O_INDEX(v) v
#endif
alignas(256) static __gm__ const uint32_t FirePacketIndexGM556[1152] = {
0U, 24U, 48U, 72U, 96U, 120U, 144U, 168U, 192U, 216U, 240U, 264U, 288U, 312U, 336U, 360U,
384U, 408U, 432U, 456U, 480U, 504U, 528U, 552U, 576U, 600U, 624U, 648U, 672U, 696U, 720U, 744U,
4U, 28U, 52U, 76U, 100U, 124U, 148U, 172U, 196U, 220U, 244U, 268U, 292U, 316U, 340U, 364U,
388U, 412U, 436U, 460U, 484U, 508U, 532U, 556U, 580U, 604U, 628U, 652U, 676U, 700U, 724U, 748U,
8U, 32U, 56U, 80U, 104U, 128U, 152U, 176U, 200U, 224U, 248U, 272U, 296U, 320U, 344U, 368U,
392U, 416U, 440U, 464U, 488U, 512U, 536U, 560U, 584U, 608U, 632U, 656U, 680U, 704U, 728U, 752U,
12U, 36U, 60U, 84U, 108U, 132U, 156U, 180U, 204U, 228U, 252U, 276U, 300U, 324U, 348U, 372U,
396U, 420U, 444U, 468U, 492U, 516U, 540U, 564U, 588U, 612U, 636U, 660U, 684U, 708U, 732U, 756U,
16U, 40U, 64U, 88U, 112U, 136U, 160U, 184U, 208U, 232U, 256U, 280U, 304U, 328U, 352U, 376U,
400U, 424U, 448U, 472U, 496U, 520U, 544U, 568U, 592U, 616U, 640U, 664U, 688U, 712U, 736U, 760U,
20U, 44U, 68U, 92U, 116U, 140U, 164U, 188U, 212U, 236U, 260U, 284U, 308U, 332U, 356U, 380U,
404U, 428U, 452U, 476U, 500U, 524U, 548U, 572U, 596U, 620U, 644U, 668U, 692U, 716U, 740U, 764U,
0U, 20U, 40U, 60U, 80U, 100U, 120U, 140U, 160U, 180U, 200U, 220U, 240U, 260U, 280U, 300U,
320U, 340U, 360U, 380U, 400U, 420U, 440U, 460U, 480U, 500U, 520U, 540U, 560U, 580U, 600U, 620U,
4U, 24U, 44U, 64U, 84U, 104U, 124U, 144U, 164U, 184U, 204U, 224U, 244U, 264U, 284U, 304U,
324U, 344U, 364U, 384U, 404U, 424U, 444U, 464U, 484U, 504U, 524U, 544U, 564U, 584U, 604U, 624U,
8U, 28U, 48U, 68U, 88U, 108U, 128U, 148U, 168U, 188U, 208U, 228U, 248U, 268U, 288U, 308U,
328U, 348U, 368U, 388U, 408U, 428U, 448U, 468U, 488U, 508U, 528U, 548U, 568U, 588U, 608U, 628U,
12U, 32U, 52U, 72U, 92U, 112U, 132U, 152U, 172U, 192U, 212U, 232U, 252U, 272U, 292U, 312U,
332U, 352U, 372U, 392U, 412U, 432U, 452U, 472U, 492U, 512U, 532U, 552U, 572U, 592U, 612U, 632U,
16U, 36U, 56U, 76U, 96U, 116U, 136U, 156U, 176U, 196U, 216U, 236U, 256U, 276U, 296U, 316U,
336U, 356U, 376U, 396U, 416U, 436U, 456U, 476U, 496U, 516U, 536U, 556U, 576U, 596U, 616U, 636U,
FIRE_P8O_INDEX(0U), FIRE_P8O_INDEX(128U), FIRE_P8O_INDEX(256U), FIRE_P8O_INDEX(384U), FIRE_P8O_INDEX(512U), FIRE_P8O_INDEX(4U), FIRE_P8O_INDEX(132U), FIRE_P8O_INDEX(260U), FIRE_P8O_INDEX(388U), FIRE_P8O_INDEX(516U), FIRE_P8O_INDEX(8U), FIRE_P8O_INDEX(136U), FIRE_P8O_INDEX(264U), FIRE_P8O_INDEX(392U), FIRE_P8O_INDEX(520U), FIRE_P8O_INDEX(12U),
FIRE_P8O_INDEX(140U), FIRE_P8O_INDEX(268U), FIRE_P8O_INDEX(396U), FIRE_P8O_INDEX(524U), FIRE_P8O_INDEX(16U), FIRE_P8O_INDEX(144U), FIRE_P8O_INDEX(272U), FIRE_P8O_INDEX(400U), FIRE_P8O_INDEX(528U), FIRE_P8O_INDEX(20U), FIRE_P8O_INDEX(148U), FIRE_P8O_INDEX(276U), FIRE_P8O_INDEX(404U), FIRE_P8O_INDEX(532U), FIRE_P8O_INDEX(24U), FIRE_P8O_INDEX(152U),
FIRE_P8O_INDEX(280U), FIRE_P8O_INDEX(408U), FIRE_P8O_INDEX(536U), FIRE_P8O_INDEX(28U), FIRE_P8O_INDEX(156U), FIRE_P8O_INDEX(284U), FIRE_P8O_INDEX(412U), FIRE_P8O_INDEX(540U), FIRE_P8O_INDEX(32U), FIRE_P8O_INDEX(160U), FIRE_P8O_INDEX(288U), FIRE_P8O_INDEX(416U), FIRE_P8O_INDEX(544U), FIRE_P8O_INDEX(36U), FIRE_P8O_INDEX(164U), FIRE_P8O_INDEX(292U),
FIRE_P8O_INDEX(420U), FIRE_P8O_INDEX(548U), FIRE_P8O_INDEX(40U), FIRE_P8O_INDEX(168U), FIRE_P8O_INDEX(296U), FIRE_P8O_INDEX(424U), FIRE_P8O_INDEX(552U), FIRE_P8O_INDEX(44U), FIRE_P8O_INDEX(172U), FIRE_P8O_INDEX(300U), FIRE_P8O_INDEX(428U), FIRE_P8O_INDEX(556U), FIRE_P8O_INDEX(48U), FIRE_P8O_INDEX(176U), FIRE_P8O_INDEX(304U), FIRE_P8O_INDEX(432U),
FIRE_P8O_INDEX(560U), FIRE_P8O_INDEX(52U), FIRE_P8O_INDEX(180U), FIRE_P8O_INDEX(308U), FIRE_P8O_INDEX(436U), FIRE_P8O_INDEX(564U), FIRE_P8O_INDEX(56U), FIRE_P8O_INDEX(184U), FIRE_P8O_INDEX(312U), FIRE_P8O_INDEX(440U), FIRE_P8O_INDEX(568U), FIRE_P8O_INDEX(60U), FIRE_P8O_INDEX(188U), FIRE_P8O_INDEX(316U), FIRE_P8O_INDEX(444U), FIRE_P8O_INDEX(572U),
FIRE_P8O_INDEX(64U), FIRE_P8O_INDEX(192U), FIRE_P8O_INDEX(320U), FIRE_P8O_INDEX(448U), FIRE_P8O_INDEX(576U), FIRE_P8O_INDEX(68U), FIRE_P8O_INDEX(196U), FIRE_P8O_INDEX(324U), FIRE_P8O_INDEX(452U), FIRE_P8O_INDEX(580U), FIRE_P8O_INDEX(72U), FIRE_P8O_INDEX(200U), FIRE_P8O_INDEX(328U), FIRE_P8O_INDEX(456U), FIRE_P8O_INDEX(584U), FIRE_P8O_INDEX(76U),
FIRE_P8O_INDEX(204U), FIRE_P8O_INDEX(332U), FIRE_P8O_INDEX(460U), FIRE_P8O_INDEX(588U), FIRE_P8O_INDEX(80U), FIRE_P8O_INDEX(208U), FIRE_P8O_INDEX(336U), FIRE_P8O_INDEX(464U), FIRE_P8O_INDEX(592U), FIRE_P8O_INDEX(84U), FIRE_P8O_INDEX(212U), FIRE_P8O_INDEX(340U), FIRE_P8O_INDEX(468U), FIRE_P8O_INDEX(596U), FIRE_P8O_INDEX(88U), FIRE_P8O_INDEX(216U),
FIRE_P8O_INDEX(344U), FIRE_P8O_INDEX(472U), FIRE_P8O_INDEX(600U), FIRE_P8O_INDEX(92U), FIRE_P8O_INDEX(220U), FIRE_P8O_INDEX(348U), FIRE_P8O_INDEX(476U), FIRE_P8O_INDEX(604U), FIRE_P8O_INDEX(96U), FIRE_P8O_INDEX(224U), FIRE_P8O_INDEX(352U), FIRE_P8O_INDEX(480U), FIRE_P8O_INDEX(608U), FIRE_P8O_INDEX(100U), FIRE_P8O_INDEX(228U), FIRE_P8O_INDEX(356U),
FIRE_P8O_INDEX(484U), FIRE_P8O_INDEX(612U), FIRE_P8O_INDEX(104U), FIRE_P8O_INDEX(232U), FIRE_P8O_INDEX(360U), FIRE_P8O_INDEX(488U), FIRE_P8O_INDEX(616U), FIRE_P8O_INDEX(108U), FIRE_P8O_INDEX(236U), FIRE_P8O_INDEX(364U), FIRE_P8O_INDEX(492U), FIRE_P8O_INDEX(620U), FIRE_P8O_INDEX(112U), FIRE_P8O_INDEX(240U), FIRE_P8O_INDEX(368U), FIRE_P8O_INDEX(496U),
FIRE_P8O_INDEX(624U), FIRE_P8O_INDEX(116U), FIRE_P8O_INDEX(244U), FIRE_P8O_INDEX(372U), FIRE_P8O_INDEX(500U), FIRE_P8O_INDEX(628U), FIRE_P8O_INDEX(120U), FIRE_P8O_INDEX(248U), FIRE_P8O_INDEX(376U), FIRE_P8O_INDEX(504U), FIRE_P8O_INDEX(632U), FIRE_P8O_INDEX(124U), FIRE_P8O_INDEX(252U), FIRE_P8O_INDEX(380U), FIRE_P8O_INDEX(508U), FIRE_P8O_INDEX(636U),
0U, 24U, 48U, 72U, 96U, 120U, 144U, 168U, 192U, 216U, 240U, 264U, 288U, 312U, 336U, 360U,
384U, 408U, 432U, 456U, 480U, 504U, 528U, 552U, 576U, 600U, 624U, 648U, 672U, 696U, 720U, 744U,
4U, 28U, 52U, 76U, 100U, 124U, 148U, 172U, 196U, 220U, 244U, 268U, 292U, 316U, 340U, 364U,
388U, 412U, 436U, 460U, 484U, 508U, 532U, 556U, 580U, 604U, 628U, 652U, 676U, 700U, 724U, 748U,
8U, 32U, 56U, 80U, 104U, 128U, 152U, 176U, 200U, 224U, 248U, 272U, 296U, 320U, 344U, 368U,
392U, 416U, 440U, 464U, 488U, 512U, 536U, 560U, 584U, 608U, 632U, 656U, 680U, 704U, 728U, 752U,
12U, 36U, 60U, 84U, 108U, 132U, 156U, 180U, 204U, 228U, 252U, 276U, 300U, 324U, 348U, 372U,
396U, 420U, 444U, 468U, 492U, 516U, 540U, 564U, 588U, 612U, 636U, 660U, 684U, 708U, 732U, 756U,
16U, 40U, 64U, 88U, 112U, 136U, 160U, 184U, 208U, 232U, 256U, 280U, 304U, 328U, 352U, 376U,
400U, 424U, 448U, 472U, 496U, 520U, 544U, 568U, 592U, 616U, 640U, 664U, 688U, 712U, 736U, 760U,
20U, 44U, 68U, 92U, 116U, 140U, 164U, 188U, 212U, 236U, 260U, 284U, 308U, 332U, 356U, 380U,
404U, 428U, 452U, 476U, 500U, 524U, 548U, 572U, 596U, 620U, 644U, 668U, 692U, 716U, 740U, 764U,
640U, 0U, 20U, 40U, 60U, 80U, 100U, 120U, 140U, 160U, 180U, 200U, 220U, 240U, 260U, 280U,
300U, 320U, 340U, 360U, 380U, 400U, 420U, 440U, 460U, 480U, 500U, 520U, 540U, 560U, 580U, 600U,
644U, 4U, 24U, 44U, 64U, 84U, 104U, 124U, 144U, 164U, 184U, 204U, 224U, 244U, 264U, 284U,
304U, 324U, 344U, 364U, 384U, 404U, 424U, 444U, 464U, 484U, 504U, 524U, 544U, 564U, 584U, 604U,
648U, 8U, 28U, 48U, 68U, 88U, 108U, 128U, 148U, 168U, 188U, 208U, 228U, 248U, 268U, 288U,
308U, 328U, 348U, 368U, 388U, 408U, 428U, 448U, 468U, 488U, 508U, 528U, 548U, 568U, 588U, 608U,
652U, 12U, 32U, 52U, 72U, 92U, 112U, 132U, 152U, 172U, 192U, 212U, 232U, 252U, 272U, 292U,
312U, 332U, 352U, 372U, 392U, 412U, 432U, 452U, 472U, 492U, 512U, 532U, 552U, 572U, 592U, 612U,
656U, 16U, 36U, 56U, 76U, 96U, 116U, 136U, 156U, 176U, 196U, 216U, 236U, 256U, 276U, 296U,
316U, 336U, 356U, 376U, 396U, 416U, 436U, 456U, 476U, 496U, 516U, 536U, 556U, 576U, 596U, 616U,
FIRE_P8O_INDEX(0U), FIRE_P8O_INDEX(128U), FIRE_P8O_INDEX(256U), FIRE_P8O_INDEX(384U), FIRE_P8O_INDEX(512U), FIRE_P8O_INDEX(4U), FIRE_P8O_INDEX(132U), FIRE_P8O_INDEX(260U), FIRE_P8O_INDEX(388U), FIRE_P8O_INDEX(516U), FIRE_P8O_INDEX(8U), FIRE_P8O_INDEX(136U), FIRE_P8O_INDEX(264U), FIRE_P8O_INDEX(392U), FIRE_P8O_INDEX(520U), FIRE_P8O_INDEX(12U),
FIRE_P8O_INDEX(140U), FIRE_P8O_INDEX(268U), FIRE_P8O_INDEX(396U), FIRE_P8O_INDEX(524U), FIRE_P8O_INDEX(16U), FIRE_P8O_INDEX(144U), FIRE_P8O_INDEX(272U), FIRE_P8O_INDEX(400U), FIRE_P8O_INDEX(528U), FIRE_P8O_INDEX(20U), FIRE_P8O_INDEX(148U), FIRE_P8O_INDEX(276U), FIRE_P8O_INDEX(404U), FIRE_P8O_INDEX(532U), FIRE_P8O_INDEX(24U), FIRE_P8O_INDEX(152U),
FIRE_P8O_INDEX(280U), FIRE_P8O_INDEX(408U), FIRE_P8O_INDEX(536U), FIRE_P8O_INDEX(28U), FIRE_P8O_INDEX(156U), FIRE_P8O_INDEX(284U), FIRE_P8O_INDEX(412U), FIRE_P8O_INDEX(540U), FIRE_P8O_INDEX(32U), FIRE_P8O_INDEX(160U), FIRE_P8O_INDEX(288U), FIRE_P8O_INDEX(416U), FIRE_P8O_INDEX(544U), FIRE_P8O_INDEX(36U), FIRE_P8O_INDEX(164U), FIRE_P8O_INDEX(292U),
FIRE_P8O_INDEX(420U), FIRE_P8O_INDEX(548U), FIRE_P8O_INDEX(40U), FIRE_P8O_INDEX(168U), FIRE_P8O_INDEX(296U), FIRE_P8O_INDEX(424U), FIRE_P8O_INDEX(552U), FIRE_P8O_INDEX(44U), FIRE_P8O_INDEX(172U), FIRE_P8O_INDEX(300U), FIRE_P8O_INDEX(428U), FIRE_P8O_INDEX(556U), FIRE_P8O_INDEX(48U), FIRE_P8O_INDEX(176U), FIRE_P8O_INDEX(304U), FIRE_P8O_INDEX(432U),
FIRE_P8O_INDEX(560U), FIRE_P8O_INDEX(52U), FIRE_P8O_INDEX(180U), FIRE_P8O_INDEX(308U), FIRE_P8O_INDEX(436U), FIRE_P8O_INDEX(564U), FIRE_P8O_INDEX(56U), FIRE_P8O_INDEX(184U), FIRE_P8O_INDEX(312U), FIRE_P8O_INDEX(440U), FIRE_P8O_INDEX(568U), FIRE_P8O_INDEX(60U), FIRE_P8O_INDEX(188U), FIRE_P8O_INDEX(316U), FIRE_P8O_INDEX(444U), FIRE_P8O_INDEX(572U),
FIRE_P8O_INDEX(64U), FIRE_P8O_INDEX(192U), FIRE_P8O_INDEX(320U), FIRE_P8O_INDEX(448U), FIRE_P8O_INDEX(576U), FIRE_P8O_INDEX(68U), FIRE_P8O_INDEX(196U), FIRE_P8O_INDEX(324U), FIRE_P8O_INDEX(452U), FIRE_P8O_INDEX(580U), FIRE_P8O_INDEX(72U), FIRE_P8O_INDEX(200U), FIRE_P8O_INDEX(328U), FIRE_P8O_INDEX(456U), FIRE_P8O_INDEX(584U), FIRE_P8O_INDEX(76U),
FIRE_P8O_INDEX(204U), FIRE_P8O_INDEX(332U), FIRE_P8O_INDEX(460U), FIRE_P8O_INDEX(588U), FIRE_P8O_INDEX(80U), FIRE_P8O_INDEX(208U), FIRE_P8O_INDEX(336U), FIRE_P8O_INDEX(464U), FIRE_P8O_INDEX(592U), FIRE_P8O_INDEX(84U), FIRE_P8O_INDEX(212U), FIRE_P8O_INDEX(340U), FIRE_P8O_INDEX(468U), FIRE_P8O_INDEX(596U), FIRE_P8O_INDEX(88U), FIRE_P8O_INDEX(216U),
FIRE_P8O_INDEX(344U), FIRE_P8O_INDEX(472U), FIRE_P8O_INDEX(600U), FIRE_P8O_INDEX(92U), FIRE_P8O_INDEX(220U), FIRE_P8O_INDEX(348U), FIRE_P8O_INDEX(476U), FIRE_P8O_INDEX(604U), FIRE_P8O_INDEX(96U), FIRE_P8O_INDEX(224U), FIRE_P8O_INDEX(352U), FIRE_P8O_INDEX(480U), FIRE_P8O_INDEX(608U), FIRE_P8O_INDEX(100U), FIRE_P8O_INDEX(228U), FIRE_P8O_INDEX(356U),
FIRE_P8O_INDEX(484U), FIRE_P8O_INDEX(612U), FIRE_P8O_INDEX(104U), FIRE_P8O_INDEX(232U), FIRE_P8O_INDEX(360U), FIRE_P8O_INDEX(488U), FIRE_P8O_INDEX(616U), FIRE_P8O_INDEX(108U), FIRE_P8O_INDEX(236U), FIRE_P8O_INDEX(364U), FIRE_P8O_INDEX(492U), FIRE_P8O_INDEX(620U), FIRE_P8O_INDEX(112U), FIRE_P8O_INDEX(240U), FIRE_P8O_INDEX(368U), FIRE_P8O_INDEX(496U),
FIRE_P8O_INDEX(624U), FIRE_P8O_INDEX(116U), FIRE_P8O_INDEX(244U), FIRE_P8O_INDEX(372U), FIRE_P8O_INDEX(500U), FIRE_P8O_INDEX(628U), FIRE_P8O_INDEX(120U), FIRE_P8O_INDEX(248U), FIRE_P8O_INDEX(376U), FIRE_P8O_INDEX(504U), FIRE_P8O_INDEX(632U), FIRE_P8O_INDEX(124U), FIRE_P8O_INDEX(252U), FIRE_P8O_INDEX(380U), FIRE_P8O_INDEX(508U), FIRE_P8O_INDEX(636U),
0U, 24U, 0U, 24U, 0U, 24U, 0U, 24U, 4U, 28U, 4U, 28U, 4U, 28U, 4U, 28U,
8U, 32U, 8U, 32U, 8U, 32U, 8U, 32U, 12U, 36U, 12U, 36U, 12U, 36U, 12U, 36U,
16U, 40U, 16U, 40U, 16U, 40U, 16U, 40U, 20U, 44U, 20U, 44U, 20U, 44U, 20U, 44U,
0U, 20U, 0U, 20U, 0U, 20U, 0U, 20U, 4U, 24U, 4U, 24U, 4U, 24U, 4U, 24U,
8U, 28U, 8U, 28U, 8U, 28U, 8U, 28U, 12U, 32U, 12U, 32U, 12U, 32U, 12U, 32U,
16U, 36U, 16U, 36U, 16U, 36U, 16U, 36U, 0U, 32U, 64U, 96U, 128U, 4U, 36U, 68U,
100U, 132U, 8U, 40U, 72U, 104U, 136U, 12U, 44U, 76U, 108U, 140U, 16U, 48U, 80U, 112U,
144U, 20U, 52U, 84U, 116U, 148U, 24U, 56U, 88U, 120U, 152U, 28U, 60U, 92U, 124U, 156U,
};
static_assert(sizeof(FirePacketIndexGM556) == 4608, "exact1152u32");

alignas(256) static __gm__ const uint32_t FireSmallIndexGM678[544] = {
0U, 24U, 48U, 72U, 96U, 120U, 144U, 168U, 192U, 216U, 240U, 264U, 288U, 312U, 336U, 360U,
384U, 408U, 432U, 456U, 480U, 504U, 528U, 552U, 576U, 600U, 624U, 648U, 672U, 696U, 720U, 744U,
4U, 28U, 52U, 76U, 100U, 124U, 148U, 172U, 196U, 220U, 244U, 268U, 292U, 316U, 340U, 364U,
388U, 412U, 436U, 460U, 484U, 508U, 532U, 556U, 580U, 604U, 628U, 652U, 676U, 700U, 724U, 748U,
8U, 32U, 56U, 80U, 104U, 128U, 152U, 176U, 200U, 224U, 248U, 272U, 296U, 320U, 344U, 368U,
392U, 416U, 440U, 464U, 488U, 512U, 536U, 560U, 584U, 608U, 632U, 656U, 680U, 704U, 728U, 752U,
12U, 36U, 60U, 84U, 108U, 132U, 156U, 180U, 204U, 228U, 252U, 276U, 300U, 324U, 348U, 372U,
396U, 420U, 444U, 468U, 492U, 516U, 540U, 564U, 588U, 612U, 636U, 660U, 684U, 708U, 732U, 756U,
16U, 40U, 64U, 88U, 112U, 136U, 160U, 184U, 208U, 232U, 256U, 280U, 304U, 328U, 352U, 376U,
400U, 424U, 448U, 472U, 496U, 520U, 544U, 568U, 592U, 616U, 640U, 664U, 688U, 712U, 736U, 760U,
20U, 44U, 68U, 92U, 116U, 140U, 164U, 188U, 212U, 236U, 260U, 284U, 308U, 332U, 356U, 380U,
404U, 428U, 452U, 476U, 500U, 524U, 548U, 572U, 596U, 620U, 644U, 668U, 692U, 716U, 740U, 764U,
0U, 20U, 40U, 60U, 80U, 100U, 120U, 140U, 160U, 180U, 200U, 220U, 240U, 260U, 280U, 300U,
320U, 340U, 360U, 380U, 400U, 420U, 440U, 460U, 480U, 500U, 520U, 540U, 560U, 580U, 600U, 620U,
4U, 24U, 44U, 64U, 84U, 104U, 124U, 144U, 164U, 184U, 204U, 224U, 244U, 264U, 284U, 304U,
324U, 344U, 364U, 384U, 404U, 424U, 444U, 464U, 484U, 504U, 524U, 544U, 564U, 584U, 604U, 624U,
8U, 28U, 48U, 68U, 88U, 108U, 128U, 148U, 168U, 188U, 208U, 228U, 248U, 268U, 288U, 308U,
328U, 348U, 368U, 388U, 408U, 428U, 448U, 468U, 488U, 508U, 528U, 548U, 568U, 588U, 608U, 628U,
8U, 28U, 48U, 68U, 88U, 108U, 128U, 148U, 168U, 188U, 208U, 228U, 248U, 268U, 288U, 308U,
328U, 348U, 368U, 388U, 408U, 428U, 448U, 468U, 488U, 508U, 528U, 548U, 568U, 588U, 608U, 628U,
12U, 32U, 52U, 72U, 92U, 112U, 132U, 152U, 172U, 192U, 212U, 232U, 252U, 272U, 292U, 312U,
332U, 352U, 372U, 392U, 412U, 432U, 452U, 472U, 492U, 512U, 532U, 552U, 572U, 592U, 612U, 632U,
16U, 36U, 56U, 76U, 96U, 116U, 136U, 156U, 176U, 196U, 216U, 236U, 256U, 276U, 296U, 316U,
336U, 356U, 376U, 396U, 416U, 436U, 456U, 476U, 496U, 516U, 536U, 556U, 576U, 596U, 616U, 636U,
0U, 128U, 256U, 384U, 512U, 4U, 132U, 260U, 388U, 516U, 8U, 136U, 264U, 392U, 520U, 12U,
140U, 268U, 396U, 524U, 16U, 144U, 272U, 400U, 528U, 20U, 148U, 276U, 404U, 532U, 24U, 152U,
280U, 408U, 536U, 28U, 156U, 284U, 412U, 540U, 32U, 160U, 288U, 416U, 544U, 36U, 164U, 292U,
420U, 548U, 40U, 168U, 296U, 424U, 552U, 44U, 172U, 300U, 428U, 556U, 48U, 176U, 304U, 432U,
560U, 52U, 180U, 308U, 436U, 564U, 56U, 184U, 312U, 440U, 568U, 60U, 188U, 316U, 444U, 572U,
64U, 192U, 320U, 448U, 576U, 68U, 196U, 324U, 452U, 580U, 72U, 200U, 328U, 456U, 584U, 76U,
204U, 332U, 460U, 588U, 80U, 208U, 336U, 464U, 592U, 84U, 212U, 340U, 468U, 596U, 88U, 216U,
344U, 472U, 600U, 92U, 220U, 348U, 476U, 604U, 96U, 224U, 352U, 480U, 608U, 100U, 228U, 356U,
484U, 612U, 104U, 232U, 360U, 488U, 616U, 108U, 236U, 364U, 492U, 620U, 112U, 240U, 368U, 496U,
624U, 116U, 244U, 372U, 500U, 628U, 120U, 248U, 376U, 504U, 632U, 124U, 252U, 380U, 508U, 636U,
};
static_assert(sizeof(FireSmallIndexGM678) == 2176, "exact544u32");

// Complete static storage: no TPipe/TQue/TBuf or runtime event allocation.
// Each address is the original v072 InitBuffer address, including both DB slots.
template <typename T, uint32_t TILE>
class FireStaticStorage {
static constexpr uint32_t DEPTH = TILE > 256 ? 2 : 1;
static constexpr uint32_t STRIDE = TILE + 8;
static constexpr uint32_t RAW_BYTES = TILE * 6 * sizeof(T);
static constexpr uint32_t PRIOR_BYTES = TILE * 5 * sizeof(float);
static constexpr uint32_t OUT_BYTES = TILE * 5 * sizeof(T);
static constexpr uint32_t WORK_BYTES = ((STRIDE * 11 * sizeof(float) + 511) / 512) * 512 + 256;
static constexpr uint32_t ONE_BYTES = TILE * 6 * sizeof(float) + (sizeof(T) == sizeof(half) ? 256 : 0);
static constexpr uint32_t RAW_BASE = 0;
static constexpr uint32_t PRIOR_BASE = RAW_BASE + DEPTH * RAW_BYTES;
static constexpr uint32_t OUT_BASE = PRIOR_BASE + DEPTH * PRIOR_BYTES;
static constexpr uint32_t WORK_BASE = OUT_BASE + DEPTH * OUT_BYTES;
static constexpr uint32_t ONE_BASE = WORK_BASE + WORK_BYTES;
static constexpr uint32_t RF_BASE = ONE_BASE + ONE_BYTES;
static constexpr uint32_t PACK_BASE = RF_BASE + (sizeof(T) == sizeof(half) ? TILE * 6 * sizeof(float) : 0);
static constexpr uint32_t INDEX_BASE = PACK_BASE + (sizeof(T) == sizeof(half) ? TILE * 5 * sizeof(float) : 0);
static_assert(INDEX_BASE + TILE * 7 * sizeof(uint32_t) < 188416, "reserve the SDK temporary UB region");
static_assert((RAW_BYTES | PRIOR_BYTES | OUT_BYTES | WORK_BYTES | ONE_BYTES | INDEX_BASE) % 32 == 0,
"all static addresses and capacities retain 32-byte alignment");
uint32_t rawCursor = 0, priorCursor = 0, outCursor = 0;
uint32_t rawSlot = 0, priorSlot = 0, outSlot = 0;
uint32_t rawUsed = 0, priorUsed = 0, outUsed = 0;
public:
__aicore__ inline AscendC::LocalTensor<int32_t> Indices() {
return AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, INDEX_BASE, TILE * 7);
}
__aicore__ inline AscendC::LocalTensor<float> Work() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, WORK_BASE, WORK_BYTES / sizeof(float));
}
__aicore__ inline AscendC::LocalTensor<float> Ones() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE, ONE_BYTES / sizeof(float));
}
__aicore__ inline AscendC::LocalTensor<float> RawFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, RF_BASE, TILE * 6);
}
__aicore__ inline AscendC::LocalTensor<float> PackedFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, PACK_BASE, TILE * 5);
}
// A view only: allocation/cursor/event ownership remains in AllocRaw/FreeRaw.
__aicore__ inline AscendC::LocalTensor<T> RawFixedSlot(uint32_t slot) {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, RAW_BASE + slot * RAW_BYTES, TILE * 6);
}
__aicore__ inline AscendC::LocalTensor<T> AllocRaw() {
rawSlot = rawCursor;
rawCursor = (rawCursor + 1) % DEPTH;
if (rawUsed & (1U << rawSlot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(rawSlot);
return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, RAW_BASE + rawSlot * RAW_BYTES, TILE * 6);
}
__aicore__ inline AscendC::LocalTensor<float> AllocPrior() {
priorSlot = priorCursor;
priorCursor = (priorCursor + 1) % DEPTH;
if (priorUsed & (1U << priorSlot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2 + priorSlot);
return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE + priorSlot * PRIOR_BYTES, TILE * 5);
}
__aicore__ inline AscendC::LocalTensor<T> AllocOut() {
outSlot = outCursor;
outCursor = (outCursor + 1) % DEPTH;
if (outUsed & (1U << outSlot)) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outSlot);
return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, OUT_BASE + outSlot * OUT_BYTES, TILE * 5);
}
__aicore__ inline void RawReady() { AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(rawSlot); }
__aicore__ inline void RawWait() { AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(rawSlot); }
__aicore__ inline void PriorReady() { AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2 + priorSlot); }
__aicore__ inline void PriorWait() { AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2 + priorSlot); }
__aicore__ inline void OutReady() { AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outSlot); }
__aicore__ inline void OutWait() { AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outSlot); }
__aicore__ inline void FreeRaw() {
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(rawSlot);
rawUsed |= 1U << rawSlot;
}
__aicore__ inline void FreePrior() {
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2 + priorSlot);
priorUsed |= 1U << priorSlot;
}
__aicore__ inline void FreeOut() {
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outSlot);
outUsed |= 1U << outSlot;
}
__aicore__ inline void Drain() {
for (uint32_t slot = 0; slot < DEPTH; ++slot) {
if (rawUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
if (priorUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2 + slot);
if (outUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
}
}
};

// Work is dead until the first raw/prior Gather; seed and broadcast areas are disjoint.
template <uint32_t TILE>
__aicore__ inline void MakeFireOutputOffsets(AscendC::LocalTensor<int32_t> dst,
AscendC::LocalTensor<int32_t> rawOffsets, AscendC::LocalTensor<int32_t> priorOffsets,
AscendC::LocalTensor<int32_t> scratch) {
constexpr uint32_t STRIDE = TILE + 8;
auto seed = dst;
if constexpr (TILE > 256) seed = scratch;
for (int32_t j = 0; j < 8; ++j)
for (int32_t k = 0; k < 5; ++k)
seed.SetValue(j * 5 + k, (k * int32_t(STRIDE) + j) * 4);
constexpr int32_t event = 2;
AscendC::SetFlag<AscendC::HardEvent::S_V>(event);
AscendC::WaitFlag<AscendC::HardEvent::S_V>(event);
if constexpr (TILE > 256) {
auto blockRamp = scratch[64];
auto blockBroadcast = scratch[64 + TILE / 8];
AscendC::Sub<int32_t>(blockRamp, rawOffsets, priorOffsets, TILE / 8);
AscendC::Muls<int32_t>(blockRamp, blockRamp, int32_t(8), TILE / 8);
AscendC::Brcb<int32_t>(blockBroadcast, blockRamp, uint8_t(TILE / 64), {1, 8});
AscendC::Add<int32_t>(dst, seed, blockBroadcast, uint64_t(40), uint8_t(TILE / 8),
{1, 1, 0, 5, 0, 1});
} else {
for (uint32_t width = 8; width < TILE; width *= 2) {
const uint32_t count = width < TILE - width ? width : TILE - width;
AscendC::Adds(dst[width * 5], dst, int32_t(width * 4), count * 5);
AscendC::PipeBarrier<PIPE_V>();
}
}
}

template <typename T, uint32_t TILE>
class KernelFire {
static constexpr uint32_t DEPTH = TILE > 256 ? 2 : 1;
static constexpr uint32_t STRIDE = TILE + 8;
FireStaticStorage<T, TILE> mem;
AscendC::GlobalTensor<T> rawGm, outGm;
AscendC::GlobalTensor<float> priorGm;
template <typename V, uint32_t COUNT, uint32_t SEED_EVENT>
__aicore__ inline void MakeOffsets(AscendC::LocalTensor<V> dst, V step) {
for (int32_t j = 0; j < 8; ++j) dst.SetValue(j, V(j) * step);
constexpr int32_t event = SEED_EVENT;
AscendC::SetFlag<AscendC::HardEvent::S_V>(event);
AscendC::WaitFlag<AscendC::HardEvent::S_V>(event);
for (uint32_t width = 8; width < COUNT; width *= 2) {
const uint32_t count = width < COUNT - width ? width : COUNT - width;
AscendC::Adds(dst[width], dst, V(int32_t(width)) * step, count);
AscendC::PipeBarrier<PIPE_V>();
}
}
public:
__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded,
const FireCandidateDecodeFusionTilingData &t) {
rawGm.SetGlobalBuffer((__gm__ T *)raw);
priorGm.SetGlobalBuffer((__gm__ float *)priors);
outGm.SetGlobalBuffer((__gm__ T *)decoded);
uint32_t begin = AscendC::GetBlockIdx() * t.perCore;
uint32_t end = begin + t.perCore < t.total ? begin + t.perCore : t.total;
if (begin >= end) return;
uint32_t firstCount = end - begin < TILE ? end - begin : TILE;
const uint32_t firstPriorPos = begin % t.n;
if (firstCount > t.n - firstPriorPos) firstCount = t.n - firstPriorPos;
auto firstR = mem.AllocRaw();
auto firstP = mem.AllocPrior();
AscendC::DataCopyExtParams firstRp{1, firstCount * 6 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyExtParams firstPp{1, firstCount * 5 * uint32_t(sizeof(float)), 0, 0, 0};
AscendC::DataCopyPad(firstR, rawGm[begin * 6], firstRp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
AscendC::DataCopyPad(firstP, priorGm[firstPriorPos * 5], firstPp, AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
mem.RawReady(); mem.PriorReady();
auto rawOffsets = mem.Indices();
auto priorOffsets = rawOffsets[TILE];
auto outOffsets = rawOffsets[2 * TILE];
MakeOffsets<int32_t, TILE, 0>(rawOffsets, 4);
AscendC::Muls<int32_t>(priorOffsets, rawOffsets, int32_t(5), TILE);
AscendC::Muls<int32_t>(rawOffsets, rawOffsets, int32_t(6), TILE);
MakeFireOutputOffsets<TILE>(outOffsets, rawOffsets, priorOffsets,
mem.Work().template ReinterpretCast<int32_t>());
auto one = mem.Ones();
AscendC::Duplicate(one, 1.0f, TILE * 6);
for (uint32_t pos = begin; pos < end;) {
uint32_t count = end - pos < TILE ? end - pos : TILE;
uint32_t priorPos = pos % t.n;
if (count > t.n - priorPos) count = t.n - priorPos;
auto r = firstR;
auto p = firstP;
if (pos != begin) {
r = mem.AllocRaw();
p = mem.AllocPrior();
AscendC::DataCopyExtParams rp{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyExtParams pp{1, count * 5 * uint32_t(sizeof(float)), 0, 0, 0};
AscendC::DataCopyPad(r, rawGm[pos * 6], rp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
AscendC::DataCopyPad(p, priorGm[priorPos * 5], pp, AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
mem.RawReady(); mem.PriorReady();
}
mem.RawWait(); mem.PriorWait();
AscendC::LocalTensor<float> f[11];
for (uint32_t k = 0; k < 11; ++k) f[k] = mem.Work()[k * STRIDE];
AscendC::LocalTensor<float> rf;
if constexpr (sizeof(T) == sizeof(float)) rf = r.template ReinterpretCast<float>();
else {
rf = mem.RawFloat();
AscendC::Cast(rf, r, AscendC::RoundMode::CAST_NONE, count * 6);
}
AscendC::Muls(rf, rf, -1.0f, count * 6);
AscendC::Exp(rf, rf, count * 6);
AscendC::Adds(rf, rf, 1.0f, count * 6);
AscendC::Div(rf, one, rf, count * 6);
for (uint32_t k = 0; k < 6; ++k)
AscendC::Gather(f[k], rf, rawOffsets.template ReinterpretCast<uint32_t>(), k * 4, count);
for (uint32_t k = 0; k < 5; ++k)
AscendC::Gather(f[k + 6], p, priorOffsets.template ReinterpretCast<uint32_t>(), k * 4, count);
for (uint32_t k = 0; k < 2; ++k) {
AscendC::Muls(f[k], f[k], 2.0f, count);
AscendC::Adds(f[k], f[k], -0.5f, count);
AscendC::Add(f[k], f[k], f[k + 6], count);
AscendC::Mul(f[k], f[k], f[8], count);
AscendC::Muls(f[k + 2], f[k + 2], 2.0f, count);
AscendC::Mul(f[k + 2], f[k + 2], f[k + 2], count);
AscendC::Mul(f[k + 2], f[k + 2], f[k + 9], count);
AscendC::Mul(f[k + 2], f[k + 2], f[8], count);
}
AscendC::Mul(f[4], f[4], f[5], count);
auto out = mem.AllocOut();
if constexpr (sizeof(T) == sizeof(float)) {
AscendC::Gather(out.template ReinterpretCast<float>(), mem.Work(),
outOffsets.template ReinterpretCast<uint32_t>(), 0, count * 5);
} else {
auto packed = mem.PackedFloat();
AscendC::Gather(packed, mem.Work(), outOffsets.template ReinterpretCast<uint32_t>(), 0, count * 5);
AscendC::Cast(out, packed, AscendC::RoundMode::CAST_RINT, count * 5);
}
mem.OutReady(); mem.OutWait();
AscendC::DataCopyExtParams op{1, count * 5 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(outGm[pos * 5], out, op);
mem.FreeOut(); mem.FreeRaw(); mem.FreePrior();
pos += count;
}
mem.Drain();
}
};


// A total==1-only entry; fixed storage retains v044 arithmetic and byte layout.
template <typename T>
class KernelFireTinyStatic {
static constexpr uint32_t ONE_BASE = 128 + (sizeof(T) == sizeof(half) ? 64 : 0);
AscendC::GlobalTensor<T> rawGm, outGm;
AscendC::GlobalTensor<float> priorGm;
public:
__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded) {
rawGm.SetGlobalBuffer((__gm__ T *)raw);
priorGm.SetGlobalBuffer((__gm__ float *)priors);
outGm.SetGlobalBuffer((__gm__ T *)decoded);
auto anchorOffsets = AscendC::LocalTensor<uint32_t>(AscendC::TPosition::VECCALC, ONE_BASE + 320, 16);
auto outputOffsets = anchorOffsets[8];
anchorOffsets.SetValue(0, 0);
anchorOffsets.SetValue(1, 4);
anchorOffsets.SetValue(2, 12);
anchorOffsets.SetValue(3, 16);
outputOffsets.SetValue(0, 0);
outputOffsets.SetValue(1, 4);
outputOffsets.SetValue(2, 8);
outputOffsets.SetValue(3, 12);
outputOffsets.SetValue(4, 32);
AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
auto input = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::VECIN, 0, 96);
auto r = input.template ReinterpretCast<T>();
auto p = input[64].template ReinterpretCast<float>();
AscendC::DataCopyExtParams rp{1, 6 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyExtParams pp{1, 5 * uint32_t(sizeof(float)), 0, 0, 0};
AscendC::DataCopyPad(r, rawGm, rp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
AscendC::DataCopyPad(p, priorGm, pp, AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
AscendC::LocalTensor<float> rf;
if constexpr (sizeof(T) == sizeof(float)) rf = r.template ReinterpretCast<float>();
else {
rf = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, 128, 16);
AscendC::Cast(rf, r, AscendC::RoundMode::CAST_NONE, 6);
}
auto one = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE, 8);
auto coeff = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE + 32, 8);
auto expanded = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE + 64, 64);
AscendC::Duplicate(one, 1.0f, 8);
AscendC::Muls(rf, rf, -1.0f, 6);
AscendC::Exp(rf, rf, 6);
AscendC::Adds(rf, rf, 1.0f, 6);
AscendC::Div(rf, one, rf, 6);
AscendC::Gather(coeff, p, anchorOffsets, 0, 4);
AscendC::Muls(rf, rf, 2.0f, 4);
AscendC::Adds(rf, rf, -0.5f, 2);
AscendC::Add(rf, rf, p, 2);
// Bitmap bits select widths while keeping every tensor base 32B aligned.
AscendC::SetVectorMask<float>(0, uint64_t(0xC));
const AscendC::BinaryRepeatParams binary;
AscendC::Mul<float, false>(rf, rf, rf, uint64_t(0), 1, binary);
AscendC::Mul<float, false>(rf, rf, coeff, uint64_t(0), 1, binary);
AscendC::ResetMask();
const AscendC::BrcbRepeatParams brcb;
AscendC::Brcb(expanded, p, 1, brcb);
AscendC::Mul(rf, rf, expanded[16], 4);
AscendC::Brcb(expanded, rf, 1, brcb);
AscendC::Mul(rf[8], expanded[32], expanded[40], 1);
auto out = AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, 96, 32 / sizeof(T));
if constexpr (sizeof(T) == sizeof(float)) {
AscendC::Gather(out.template ReinterpretCast<float>(), rf, outputOffsets, 0, 5);
} else {
AscendC::Gather(p, rf, outputOffsets, 0, 5);
AscendC::Cast(out, p, AscendC::RoundMode::CAST_RINT, 5);
}
AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
AscendC::DataCopyExtParams op{1, 5 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(outGm, out, op);
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
}
};

template <typename T, uint32_t TILE>
class KernelFireRow {
static constexpr uint32_t DEPTH = TILE > 256 ? 2 : 1;
static constexpr uint32_t STRIDE = TILE + 8;
FireStaticStorage<T, TILE> mem;
AscendC::GlobalTensor<T> rawGm, outGm;
AscendC::GlobalTensor<float> priorGm;
template <typename V, uint32_t COUNT, uint32_t SEED_EVENT>
__aicore__ inline void MakeOffsets(AscendC::LocalTensor<V> dst, V step) {
for (int32_t j = 0; j < 8; ++j) dst.SetValue(j, V(j) * step);
constexpr int32_t event = SEED_EVENT;
AscendC::SetFlag<AscendC::HardEvent::S_V>(event);
AscendC::WaitFlag<AscendC::HardEvent::S_V>(event);
for (uint32_t width = 8; width < COUNT; width *= 2) {
const uint32_t count = width < COUNT - width ? width : COUNT - width;
AscendC::Adds(dst[width], dst, V(int32_t(width)) * step, count);
AscendC::PipeBarrier<PIPE_V>();
}
}
public:
__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded,
const FireCandidateDecodeFusionTilingData &t) {
rawGm.SetGlobalBuffer((__gm__ T *)raw);
priorGm.SetGlobalBuffer((__gm__ float *)priors);
outGm.SetGlobalBuffer((__gm__ T *)decoded);
const uint32_t batches = t.total / t.n;
const uint32_t coresPerBatch = static_cast<uint32_t>(AscendC::GetBlockNum()) / batches;
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
const uint32_t batch = block / coresPerBatch;
const uint32_t lane = block % coresPerBatch;
uint32_t begin = batch * t.n + lane * t.perCore;
const uint32_t rowEnd = (batch + 1) * t.n;
uint32_t end = begin + t.perCore < rowEnd ? begin + t.perCore : rowEnd;
if (begin >= end) return;
uint32_t firstCount = end - begin < TILE ? end - begin : TILE;
const uint32_t firstPriorPos = begin % t.n;
if (firstCount > t.n - firstPriorPos) firstCount = t.n - firstPriorPos;
auto firstR = mem.AllocRaw();
auto firstP = mem.AllocPrior();
AscendC::DataCopyExtParams firstRp{1, firstCount * 6 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyExtParams firstPp{1, firstCount * 5 * uint32_t(sizeof(float)), 0, 0, 0};
AscendC::DataCopyPad(firstR, rawGm[begin * 6], firstRp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
AscendC::DataCopyPad(firstP, priorGm[firstPriorPos * 5], firstPp, AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
mem.RawReady(); mem.PriorReady();
auto rawOffsets = mem.Indices();
auto priorOffsets = rawOffsets[TILE];
auto outOffsets = rawOffsets[2 * TILE];
MakeOffsets<int32_t, TILE, 0>(rawOffsets, 4);
AscendC::Muls<int32_t>(priorOffsets, rawOffsets, int32_t(5), TILE);
AscendC::Muls<int32_t>(rawOffsets, rawOffsets, int32_t(6), TILE);
MakeFireOutputOffsets<TILE>(outOffsets, rawOffsets, priorOffsets,
mem.Work().template ReinterpretCast<int32_t>());
auto one = mem.Ones();
AscendC::Duplicate(one, 1.0f, TILE * 6);
for (uint32_t pos = begin; pos < end;) {
uint32_t count = end - pos < TILE ? end - pos : TILE;
uint32_t priorPos = pos % t.n;
if (count > t.n - priorPos) count = t.n - priorPos;
auto r = firstR;
auto p = firstP;
if (pos != begin) {
r = mem.AllocRaw();
p = mem.AllocPrior();
AscendC::DataCopyExtParams rp{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyExtParams pp{1, count * 5 * uint32_t(sizeof(float)), 0, 0, 0};
AscendC::DataCopyPad(r, rawGm[pos * 6], rp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
AscendC::DataCopyPad(p, priorGm[priorPos * 5], pp, AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
mem.RawReady(); mem.PriorReady();
}
mem.RawWait(); mem.PriorWait();
AscendC::LocalTensor<float> f[11];
for (uint32_t k = 0; k < 11; ++k) f[k] = mem.Work()[k * STRIDE];
AscendC::LocalTensor<float> rf;
if constexpr (sizeof(T) == sizeof(float)) rf = r.template ReinterpretCast<float>();
else {
rf = mem.RawFloat();
AscendC::Cast(rf, r, AscendC::RoundMode::CAST_NONE, count * 6);
}
AscendC::Muls(rf, rf, -1.0f, count * 6);
AscendC::Exp(rf, rf, count * 6);
AscendC::Adds(rf, rf, 1.0f, count * 6);
AscendC::Div(rf, one, rf, count * 6);
for (uint32_t k = 0; k < 6; ++k)
AscendC::Gather(f[k], rf, rawOffsets.template ReinterpretCast<uint32_t>(), k * 4, count);
for (uint32_t k = 0; k < 5; ++k)
AscendC::Gather(f[k + 6], p, priorOffsets.template ReinterpretCast<uint32_t>(), k * 4, count);
for (uint32_t k = 0; k < 2; ++k) {
AscendC::Muls(f[k], f[k], 2.0f, count);
AscendC::Adds(f[k], f[k], -0.5f, count);
AscendC::Add(f[k], f[k], f[k + 6], count);
AscendC::Mul(f[k], f[k], f[8], count);
AscendC::Muls(f[k + 2], f[k + 2], 2.0f, count);
AscendC::Mul(f[k + 2], f[k + 2], f[k + 2], count);
AscendC::Mul(f[k + 2], f[k + 2], f[k + 9], count);
AscendC::Mul(f[k + 2], f[k + 2], f[8], count);
}
AscendC::Mul(f[4], f[4], f[5], count);
auto out = mem.AllocOut();
if constexpr (sizeof(T) == sizeof(float)) {
AscendC::Gather(out.template ReinterpretCast<float>(), mem.Work(),
outOffsets.template ReinterpretCast<uint32_t>(), 0, count * 5);
} else {
auto packed = mem.PackedFloat();
AscendC::Gather(packed, mem.Work(), outOffsets.template ReinterpretCast<uint32_t>(), 0, count * 5);
AscendC::Cast(out, packed, AscendC::RoundMode::CAST_RINT, count * 5);
}
mem.OutReady(); mem.OutWait();
AscendC::DataCopyExtParams op{1, count * 5 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(outGm[pos * 5], out, op);
mem.FreeOut(); mem.FreeRaw(); mem.FreePrior();
pos += count;
}
mem.Drain();
}
};


template <typename T, uint32_t TILE, uint32_t PAD = 16U>
class FireFrameStorage {
static_assert(TILE >= 128 && TILE <= 1024 && TILE % 128 == 0, "native hardware frames or large reuse frame");
static constexpr uint32_t FRAME = TILE;
static constexpr uint32_t DEPTH = TILE > 256 ? 2 : 1;
static constexpr uint32_t STRIDE = FRAME + PAD;
static_assert(PAD == 16U || (PAD == 8U && FRAME == 512U && sizeof(T) == sizeof(half)), "qualified half whole512 padding");
static constexpr uint32_t RAW_BYTES = (sizeof(T) == sizeof(float) ? FRAME : TILE) * 6 * sizeof(T);
static constexpr uint32_t PRIOR_WORDS = FRAME * 5;
static constexpr uint32_t PRIOR_BYTES = PRIOR_WORDS * sizeof(float);
static constexpr uint32_t OUT_BYTES = (sizeof(T) == sizeof(float) ? FRAME : TILE) * 5 * sizeof(T);
static constexpr uint32_t WORK_BYTES = ((STRIDE * 11 * sizeof(float) + 511) / 512) * 512 + 256;
static constexpr uint32_t ONE_WORDS = FRAME * 6;
static constexpr uint32_t ONE_BYTES = ONE_WORDS * sizeof(float) + (sizeof(T) == sizeof(half) ? 256 : 0);
static constexpr uint32_t RAW_BASE = 0;
static constexpr uint32_t PRIOR_BASE = RAW_BASE + DEPTH * RAW_BYTES;
static constexpr uint32_t OUT_BASE = PRIOR_BASE + PRIOR_BYTES;
static constexpr uint32_t WORK_BASE = OUT_BASE + DEPTH * OUT_BYTES;
static constexpr uint32_t ONE_BASE = WORK_BASE + WORK_BYTES;
static constexpr uint32_t RF_BASE = ONE_BASE + ONE_BYTES;
static constexpr uint32_t PACK_BASE = RF_BASE + (sizeof(T) == sizeof(half) ? FRAME * 6 * sizeof(float) : 0);
static constexpr uint32_t INDEX_BASE = PACK_BASE + (sizeof(T) == sizeof(half) ? FRAME * 5 * sizeof(float) + 256 : 0);
static constexpr uint32_t VA_BASE = INDEX_BASE + FRAME * sizeof(uint32_t);
static constexpr uint32_t VA_WORDS = 172;
static_assert(VA_BASE + VA_WORDS * sizeof(uint64_t) < 188416, "reserve the SDK temporary UB region");
static_assert((RAW_BYTES | PRIOR_BYTES | OUT_BYTES | WORK_BYTES | ONE_BYTES | INDEX_BASE) % 32 == 0,
"all static addresses and capacities retain 32-byte alignment");
uint32_t rawCursor = 0, priorCursor = 0, outCursor = 0;
uint32_t rawSlot = 0, priorSlot = 0, outSlot = 0;
uint32_t rawUsed = 0, priorUsed = 0, outUsed = 0;
public:
__aicore__ inline AscendC::LocalTensor<uint64_t> PackedVA() {
return AscendC::LocalTensor<uint64_t>(AscendC::TPosition::VECCALC, VA_BASE, VA_WORDS);
}
__aicore__ inline AscendC::LocalTensor<int32_t> Indices() {
return AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, INDEX_BASE, FRAME);
}
__aicore__ inline AscendC::LocalTensor<float> Work() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, WORK_BASE, WORK_BYTES / sizeof(float));
}
__aicore__ inline AscendC::LocalTensor<float> Ones() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE, ONE_BYTES / sizeof(float));
}
__aicore__ inline AscendC::LocalTensor<float> RawFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, RF_BASE, FRAME * 6);
}
__aicore__ inline AscendC::LocalTensor<float> PackedFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, PACK_BASE, FRAME * 5);
}
__aicore__ inline AscendC::LocalTensor<T> AllocRaw() {
rawSlot = rawCursor;
rawCursor = (rawCursor + 1) % DEPTH;
if (rawUsed & (1U << rawSlot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(rawSlot);
return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, RAW_BASE + rawSlot * RAW_BYTES, (sizeof(T) == sizeof(float) ? FRAME : TILE) * 6);
}
__aicore__ inline AscendC::LocalTensor<float> AllocPrior() {
priorSlot = priorCursor;
priorCursor = 0;
if (priorUsed & (1U << priorSlot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2 + priorSlot);
return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE + priorSlot * PRIOR_BYTES, PRIOR_WORDS);
}
__aicore__ inline AscendC::LocalTensor<T> AllocOut() {
outSlot = outCursor;
outCursor = (outCursor + 1) % DEPTH;
if (outUsed & (1U << outSlot)) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outSlot);
return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, OUT_BASE + outSlot * OUT_BYTES, (sizeof(T) == sizeof(float) ? FRAME : TILE) * 5);
}
__aicore__ inline AscendC::LocalTensor<T> RawFixedSlot(uint32_t slot) {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, RAW_BASE + slot * RAW_BYTES,
(sizeof(T) == sizeof(float) ? FRAME : TILE) * 6);
}
__aicore__ inline AscendC::LocalTensor<float> PriorFixedSlot(uint32_t slot) {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE + slot * PRIOR_BYTES, PRIOR_WORDS);
}
__aicore__ inline AscendC::LocalTensor<T> OutAt(uint32_t slot) {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, OUT_BASE + slot * OUT_BYTES,
(sizeof(T) == sizeof(float) ? FRAME : TILE) * 5);
}
__aicore__ inline uint32_t CurrentOutSlot() const { return outSlot; }
__aicore__ inline void RawReady() { AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(rawSlot); }
__aicore__ inline void RawWait() { AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(rawSlot); }
__aicore__ inline void PriorReady() { AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2 + priorSlot); }
__aicore__ inline void PriorWait() { AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2 + priorSlot); }
__aicore__ inline void OutReady() { AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outSlot); }
__aicore__ inline void OutWait() { AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outSlot); }
__aicore__ inline void FreeRaw() {
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(rawSlot);
rawUsed |= 1U << rawSlot;
}
__aicore__ inline void FreePrior() {
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2 + priorSlot);
priorUsed |= 1U << priorSlot;
}
__aicore__ inline void FreeOut() {
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outSlot);
outUsed |= 1U << outSlot;
}
__aicore__ inline void Drain() {
for (uint32_t slot = 0; slot < DEPTH; ++slot) {
if (rawUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
if (priorUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2 + slot);
if (outUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
}
}
};


template <typename T, uint32_t TILE, bool ROW, bool RAW3 = false, bool DIRECT_STRIDE = false, bool PAIR256 = false, bool VA_PHASE_ADDS = false, bool HALF_ONE8 = false, bool QUAD128 = false, bool F32_QUAD128 = false, uint32_t PAD = 16U>
class KernelFireFrame {
static constexpr uint32_t DEPTH = TILE > 256 ? 2 : 1;
static constexpr uint32_t FRAME = TILE;
static constexpr bool ANY_QUAD = QUAD128 || F32_QUAD128;
static constexpr uint32_t STRIDE = FRAME + PAD;
static constexpr uint32_t Q = FRAME / 8;
static constexpr bool CENTER_AXPY = FRAME == 512 && ROW && RAW3 && DIRECT_STRIDE && (sizeof(T) == sizeof(float) || (PAIR256 && HALF_ONE8 && sizeof(T) == sizeof(half)));
static constexpr bool SCORE_IN_ONES = DIRECT_STRIDE && RAW3 && ROW && FRAME == 512 && sizeof(T) == sizeof(half);
static constexpr uint32_t SwapXYOffset(uint32_t offset) {
const uint32_t field = offset / STRIDE;
const uint32_t lane = offset % STRIDE;
return CENTER_AXPY ? ((field < 2U ? field + 6U : (field >= 6U && field < 8U ? field - 6U : field)) * STRIDE + lane) : offset;
}
FireFrameStorage<T, TILE, PAD> mem;
AscendC::GlobalTensor<T> rawGm, outGm;
AscendC::GlobalTensor<float> priorGm;
// Fixed CANN9/C220 compiler-builtin experiment. Tables remain read-only after S_V0.
__aicore__ inline void StorePacked(AscendC::LocalTensor<uint64_t> dst, const uint64_t *address) {
#pragma unroll
for (uint32_t q = 0; q < 4; ++q) {
const uint64_t value = ((address[4 * q] >> 5) & 0x1fffULL) |
(((address[4 * q + 1] >> 5) & 0x1fffULL) << 16) |
(((address[4 * q + 2] >> 5) & 0x1fffULL) << 32) |
(((address[4 * q + 3] >> 5) & 0x1fffULL) << 48);
dst.SetValue(q, value);
}
}
template <uint8_t REPEAT, uint16_t DST_STRIDE, uint16_t SRC_STRIDE>
__aicore__ inline void PackedTranspose(AscendC::LocalTensor<uint64_t> dst,
AscendC::LocalTensor<uint64_t> src) {
auto *dstPacked = reinterpret_cast<__ubuf__ uint64_t *>(dst.GetPhyAddr());
auto *srcPacked = reinterpret_cast<__ubuf__ uint64_t *>(src.GetPhyAddr());
__cce_scalar::vld_va_reg(VA0, dstPacked, L128);
__cce_scalar::vld_va_reg(VA1, dstPacked, H128);
__cce_scalar::vld_va_reg(VA2, srcPacked, L128);
__cce_scalar::vld_va_reg(VA3, srcPacked, H128);
// A single repeat applies rep-strides to the first address, unlike multiple repeats.
__cce_scalar::scatter_vnchwconv_b32(VA0, VA2, REPEAT,
REPEAT == 1 ? 0 : DST_STRIDE, REPEAT == 1 ? 0 : SRC_STRIDE);
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitInputPhase() {
static_assert(PHASE < 6 && SLOT < (sizeof(T) == sizeof(float) ? 2 : 1),
"valid padded input specialization");
static_assert(!RAW3 || ((((FRAME == 896 || FRAME == 768) && !ROW) || ((FRAME == 512 || FRAME == 640) && ROW)) && PHASE < 3), "raw3 verified reuse or whole512 frames");
uint64_t sourceBase;
if constexpr (sizeof(T) == sizeof(float))
sourceBase = reinterpret_cast<uint64_t>(mem.RawFixedSlot(SLOT).GetPhyAddr());
else sourceBase = reinterpret_cast<uint64_t>(mem.RawFloat().GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
if constexpr (RAW3) {
const uint64_t oneScoreBase = reinterpret_cast<uint64_t>(mem.Ones().GetPhyAddr()) + 64U * sizeof(float);
uint64_t source[16] = {
sourceBase + uint64_t(0 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(48 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(96 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(144 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(192 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(240 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(288 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(336 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(24 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(72 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(120 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(168 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(216 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(264 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(312 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(360 + 8 * PHASE) * sizeof(float)
};
uint64_t target[16] = {
(SCORE_IN_ONES && (8 * PHASE + 0) % 6 == 5 ? oneScoreBase + uint64_t((((8 * PHASE + 0) / 6) + 0) * Q) * sizeof(float) : workBase + uint64_t(SwapXYOffset(((8 * PHASE + 0) % 6) * STRIDE + (((8 * PHASE + 0) / 6) + 0) * Q)) * sizeof(float)),
(SCORE_IN_ONES && (8 * PHASE + 0) % 6 == 5 ? oneScoreBase + uint64_t((((8 * PHASE + 0) / 6) + 4) * Q) * sizeof(float) : workBase + uint64_t(SwapXYOffset(((8 * PHASE + 0) % 6) * STRIDE + (((8 * PHASE + 0) / 6) + 4) * Q)) * sizeof(float)),
(SCORE_IN_ONES && (8 * PHASE + 1) % 6 == 5 ? oneScoreBase + uint64_t((((8 * PHASE + 1) / 6) + 0) * Q) * sizeof(float) : workBase + uint64_t(SwapXYOffset(((8 * PHASE + 1) % 6) * STRIDE + (((8 * PHASE + 1) / 6) + 0) * Q)) * sizeof(float)),
(SCORE_IN_ONES && (8 * PHASE + 1) % 6 == 5 ? oneScoreBase + uint64_t((((8 * PHASE + 1) / 6) + 4) * Q) * sizeof(float) : workBase + uint64_t(SwapXYOffset(((8 * PHASE + 1) % 6) * STRIDE + (((8 * PHASE + 1) / 6) + 4) * Q)) * sizeof(float)),
(SCORE_IN_ONES && (8 * PHASE + 2) % 6 == 5 ? oneScoreBase + uint64_t((((8 * PHASE + 2) / 6) + 0) * Q) * sizeof(float) : workBase + uint64_t(SwapXYOffset(((8 * PHASE + 2) % 6) * STRIDE + (((8 * PHASE + 2) / 6) + 0) * Q)) * sizeof(float)),
(SCORE_IN_ONES && (8 * PHASE + 2) % 6 == 5 ? oneScoreBase + uint64_t((((8 * PHASE + 2) / 6) + 4) * Q) * sizeof(float) : workBase + uint64_t(SwapXYOffset(((8 * PHASE + 2) % 6) * STRIDE + (((8 * PHASE + 2) / 6) + 4) * Q)) * sizeof(float)),
(SCORE_IN_ONES && (8 * PHASE + 3) % 6 == 5 ? oneScoreBase + uint64_t((((8 * PHASE + 3) / 6) + 0) * Q) * sizeof(float) : workBase + uint64_t(SwapXYOffset(((8 * PHASE + 3) % 6) * STRIDE + (((8 * PHASE + 3) / 6) + 0) * Q)) * sizeof(float)),
(SCORE_IN_ONES && (8 * PHASE + 3) % 6 == 5 ? oneScoreBase + uint64_t((((8 * PHASE + 3) / 6) + 4) * Q) * sizeof(float) : workBase + uint64_t(SwapXYOffset(((8 * PHASE + 3) % 6) * STRIDE + (((8 * PHASE + 3) / 6) + 4) * Q)) * sizeof(float)),
(SCORE_IN_ONES && (8 * PHASE + 4) % 6 == 5 ? oneScoreBase + uint64_t((((8 * PHASE + 4) / 6) + 0) * Q) * sizeof(float) : workBase + uint64_t(SwapXYOffset(((8 * PHASE + 4) % 6) * STRIDE + (((8 * PHASE + 4) / 6) + 0) * Q)) * sizeof(float)),
(SCORE_IN_ONES && (8 * PHASE + 4) % 6 == 5 ? oneScoreBase + uint64_t((((8 * PHASE + 4) / 6) + 4) * Q) * sizeof(float) : workBase + uint64_t(SwapXYOffset(((8 * PHASE + 4) % 6) * STRIDE + (((8 * PHASE + 4) / 6) + 4) * Q)) * sizeof(float)),
(SCORE_IN_ONES && (8 * PHASE + 5) % 6 == 5 ? oneScoreBase + uint64_t((((8 * PHASE + 5) / 6) + 0) * Q) * sizeof(float) : workBase + uint64_t(SwapXYOffset(((8 * PHASE + 5) % 6) * STRIDE + (((8 * PHASE + 5) / 6) + 0) * Q)) * sizeof(float)),
(SCORE_IN_ONES && (8 * PHASE + 5) % 6 == 5 ? oneScoreBase + uint64_t((((8 * PHASE + 5) / 6) + 4) * Q) * sizeof(float) : workBase + uint64_t(SwapXYOffset(((8 * PHASE + 5) % 6) * STRIDE + (((8 * PHASE + 5) / 6) + 4) * Q)) * sizeof(float)),
(SCORE_IN_ONES && (8 * PHASE + 6) % 6 == 5 ? oneScoreBase + uint64_t((((8 * PHASE + 6) / 6) + 0) * Q) * sizeof(float) : workBase + uint64_t(SwapXYOffset(((8 * PHASE + 6) % 6) * STRIDE + (((8 * PHASE + 6) / 6) + 0) * Q)) * sizeof(float)),
(SCORE_IN_ONES && (8 * PHASE + 6) % 6 == 5 ? oneScoreBase + uint64_t((((8 * PHASE + 6) / 6) + 4) * Q) * sizeof(float) : workBase + uint64_t(SwapXYOffset(((8 * PHASE + 6) % 6) * STRIDE + (((8 * PHASE + 6) / 6) + 4) * Q)) * sizeof(float)),
(SCORE_IN_ONES && (8 * PHASE + 7) % 6 == 5 ? oneScoreBase + uint64_t((((8 * PHASE + 7) / 6) + 0) * Q) * sizeof(float) : workBase + uint64_t(SwapXYOffset(((8 * PHASE + 7) % 6) * STRIDE + (((8 * PHASE + 7) / 6) + 0) * Q)) * sizeof(float)),
(SCORE_IN_ONES && (8 * PHASE + 7) % 6 == 5 ? oneScoreBase + uint64_t((((8 * PHASE + 7) / 6) + 4) * Q) * sizeof(float) : workBase + uint64_t(SwapXYOffset(((8 * PHASE + 7) % 6) * STRIDE + (((8 * PHASE + 7) / 6) + 4) * Q)) * sizeof(float))
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[4 * PHASE], target);
if constexpr (!VA_PHASE_ADDS || PHASE == 0) StorePacked(mem.PackedVA()[24 + SLOT * 24 + 4 * PHASE], source);
} else {
uint64_t source[16] = {
sourceBase + uint64_t(0 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(48 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(96 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(144 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(192 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(240 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(288 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(336 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(384 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(432 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(480 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(528 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(576 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(624 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(672 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(720 + 8 * PHASE) * sizeof(float)
};
uint64_t target[16] = {
workBase + uint64_t(SwapXYOffset(((8 * PHASE + 0) % 6) * STRIDE + ((8 * PHASE + 0) / 6) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((8 * PHASE + 0) % 6) * STRIDE + ((8 * PHASE + 0) / 6) * Q + 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((8 * PHASE + 1) % 6) * STRIDE + ((8 * PHASE + 1) / 6) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((8 * PHASE + 1) % 6) * STRIDE + ((8 * PHASE + 1) / 6) * Q + 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((8 * PHASE + 2) % 6) * STRIDE + ((8 * PHASE + 2) / 6) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((8 * PHASE + 2) % 6) * STRIDE + ((8 * PHASE + 2) / 6) * Q + 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((8 * PHASE + 3) % 6) * STRIDE + ((8 * PHASE + 3) / 6) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((8 * PHASE + 3) % 6) * STRIDE + ((8 * PHASE + 3) / 6) * Q + 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((8 * PHASE + 4) % 6) * STRIDE + ((8 * PHASE + 4) / 6) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((8 * PHASE + 4) % 6) * STRIDE + ((8 * PHASE + 4) / 6) * Q + 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((8 * PHASE + 5) % 6) * STRIDE + ((8 * PHASE + 5) / 6) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((8 * PHASE + 5) % 6) * STRIDE + ((8 * PHASE + 5) / 6) * Q + 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((8 * PHASE + 6) % 6) * STRIDE + ((8 * PHASE + 6) / 6) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((8 * PHASE + 6) % 6) * STRIDE + ((8 * PHASE + 6) / 6) * Q + 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((8 * PHASE + 7) % 6) * STRIDE + ((8 * PHASE + 7) / 6) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((8 * PHASE + 7) % 6) * STRIDE + ((8 * PHASE + 7) / 6) * Q + 8)) * sizeof(float)
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[4 * PHASE], target);
if constexpr (!VA_PHASE_ADDS || PHASE == 0) StorePacked(mem.PackedVA()[24 + SLOT * 24 + 4 * PHASE], source);
}
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitOutputPhase() {
static_assert(PHASE < 5 && SLOT < (sizeof(T) == sizeof(float) ? 2 : 1),
"valid padded output specialization");
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t outputBase;
if constexpr (sizeof(T) == sizeof(float))
outputBase = reinterpret_cast<uint64_t>(mem.OutAt(SLOT).GetPhyAddr());
else outputBase = reinterpret_cast<uint64_t>(mem.PackedFloat().GetPhyAddr());
uint64_t source[16] = {
workBase + uint64_t(SwapXYOffset(((CENTER_AXPY && (8 * PHASE + 0) % 5 < 2) ? (8 * PHASE + 0) % 5 + 6 : (8 * PHASE + 0) % 5) * STRIDE + ((8 * PHASE + 0) / 5) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((CENTER_AXPY && (8 * PHASE + 1) % 5 < 2) ? (8 * PHASE + 1) % 5 + 6 : (8 * PHASE + 1) % 5) * STRIDE + ((8 * PHASE + 1) / 5) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((CENTER_AXPY && (8 * PHASE + 2) % 5 < 2) ? (8 * PHASE + 2) % 5 + 6 : (8 * PHASE + 2) % 5) * STRIDE + ((8 * PHASE + 2) / 5) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((CENTER_AXPY && (8 * PHASE + 3) % 5 < 2) ? (8 * PHASE + 3) % 5 + 6 : (8 * PHASE + 3) % 5) * STRIDE + ((8 * PHASE + 3) / 5) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((CENTER_AXPY && (8 * PHASE + 4) % 5 < 2) ? (8 * PHASE + 4) % 5 + 6 : (8 * PHASE + 4) % 5) * STRIDE + ((8 * PHASE + 4) / 5) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((CENTER_AXPY && (8 * PHASE + 5) % 5 < 2) ? (8 * PHASE + 5) % 5 + 6 : (8 * PHASE + 5) % 5) * STRIDE + ((8 * PHASE + 5) / 5) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((CENTER_AXPY && (8 * PHASE + 6) % 5 < 2) ? (8 * PHASE + 6) % 5 + 6 : (8 * PHASE + 6) % 5) * STRIDE + ((8 * PHASE + 6) / 5) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((CENTER_AXPY && (8 * PHASE + 7) % 5 < 2) ? (8 * PHASE + 7) % 5 + 6 : (8 * PHASE + 7) % 5) * STRIDE + ((8 * PHASE + 7) / 5) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((CENTER_AXPY && (8 * PHASE + 0) % 5 < 2) ? (8 * PHASE + 0) % 5 + 6 : (8 * PHASE + 0) % 5) * STRIDE + ((8 * PHASE + 0) / 5) * Q + 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((CENTER_AXPY && (8 * PHASE + 1) % 5 < 2) ? (8 * PHASE + 1) % 5 + 6 : (8 * PHASE + 1) % 5) * STRIDE + ((8 * PHASE + 1) / 5) * Q + 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((CENTER_AXPY && (8 * PHASE + 2) % 5 < 2) ? (8 * PHASE + 2) % 5 + 6 : (8 * PHASE + 2) % 5) * STRIDE + ((8 * PHASE + 2) / 5) * Q + 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((CENTER_AXPY && (8 * PHASE + 3) % 5 < 2) ? (8 * PHASE + 3) % 5 + 6 : (8 * PHASE + 3) % 5) * STRIDE + ((8 * PHASE + 3) / 5) * Q + 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((CENTER_AXPY && (8 * PHASE + 4) % 5 < 2) ? (8 * PHASE + 4) % 5 + 6 : (8 * PHASE + 4) % 5) * STRIDE + ((8 * PHASE + 4) / 5) * Q + 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((CENTER_AXPY && (8 * PHASE + 5) % 5 < 2) ? (8 * PHASE + 5) % 5 + 6 : (8 * PHASE + 5) % 5) * STRIDE + ((8 * PHASE + 5) / 5) * Q + 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((CENTER_AXPY && (8 * PHASE + 6) % 5 < 2) ? (8 * PHASE + 6) % 5 + 6 : (8 * PHASE + 6) % 5) * STRIDE + ((8 * PHASE + 6) / 5) * Q + 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset(((CENTER_AXPY && (8 * PHASE + 7) % 5 < 2) ? (8 * PHASE + 7) % 5 + 6 : (8 * PHASE + 7) % 5) * STRIDE + ((8 * PHASE + 7) / 5) * Q + 8)) * sizeof(float)
};
uint64_t target[16] = {
outputBase + uint64_t((5 * 0 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 8 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 1 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 9 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 2 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 10 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 3 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 11 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 4 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 12 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 5 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 13 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 6 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 14 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 7 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 15 + PHASE) * 8) * sizeof(float)
};
constexpr uint32_t INPUT_WORDS = 72;
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[INPUT_WORDS + 4 * PHASE], source);
if constexpr (!VA_PHASE_ADDS || PHASE == 0) StorePacked(mem.PackedVA()[INPUT_WORDS + 20 + SLOT * 20 + 4 * PHASE], target);
}
template <uint32_t PHASE>
__aicore__ inline void InitPriorPhase() {
static_assert(PHASE < 5, "native prior hardware phase");
constexpr uint32_t PRIOR_TARGET = 132;
constexpr uint32_t PRIOR_SOURCE = 152 + 4 * PHASE;
const uint64_t sourceBase = reinterpret_cast<uint64_t>(mem.PriorFixedSlot(0).GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
if constexpr (PAIR256 && !ANY_QUAD) {
uint64_t source[16] = {
sourceBase + uint64_t(0 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(40 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(80 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(120 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(160 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(200 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(240 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(280 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(0 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(40 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(80 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(120 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(160 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(200 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(240 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(280 + 8 * PHASE) * sizeof(float)
};
uint64_t target[16] = {
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 0) % 5) * STRIDE + ((8 * PHASE + 0) / 5) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 0) % 5) * STRIDE + ((8 * PHASE + 0) / 5) * Q + 32)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 1) % 5) * STRIDE + ((8 * PHASE + 1) / 5) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 1) % 5) * STRIDE + ((8 * PHASE + 1) / 5) * Q + 32)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 2) % 5) * STRIDE + ((8 * PHASE + 2) / 5) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 2) % 5) * STRIDE + ((8 * PHASE + 2) / 5) * Q + 32)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 3) % 5) * STRIDE + ((8 * PHASE + 3) / 5) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 3) % 5) * STRIDE + ((8 * PHASE + 3) / 5) * Q + 32)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 4) % 5) * STRIDE + ((8 * PHASE + 4) / 5) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 4) % 5) * STRIDE + ((8 * PHASE + 4) / 5) * Q + 32)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 5) % 5) * STRIDE + ((8 * PHASE + 5) / 5) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 5) % 5) * STRIDE + ((8 * PHASE + 5) / 5) * Q + 32)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 6) % 5) * STRIDE + ((8 * PHASE + 6) / 5) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 6) % 5) * STRIDE + ((8 * PHASE + 6) / 5) * Q + 32)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 7) % 5) * STRIDE + ((8 * PHASE + 7) / 5) * Q + 0)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 7) % 5) * STRIDE + ((8 * PHASE + 7) / 5) * Q + 32)) * sizeof(float)
};
StorePacked(mem.PackedVA()[PRIOR_TARGET + 4 * PHASE], target);
if constexpr (!VA_PHASE_ADDS || PHASE == 0) StorePacked(mem.PackedVA()[PRIOR_SOURCE], source);
} else {
uint64_t source[16] = {
sourceBase + uint64_t(0 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(40 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(80 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(120 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(160 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(200 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(240 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(280 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(320 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(360 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(400 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(440 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(480 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(520 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(560 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(600 + 8 * PHASE) * sizeof(float)
};
uint64_t target[16] = {
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 0) % 5) * STRIDE + ((8 * PHASE + 0) / 5) * Q + 0 * 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 0) % 5) * STRIDE + ((8 * PHASE + 0) / 5) * Q + 1 * 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 1) % 5) * STRIDE + ((8 * PHASE + 1) / 5) * Q + 0 * 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 1) % 5) * STRIDE + ((8 * PHASE + 1) / 5) * Q + 1 * 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 2) % 5) * STRIDE + ((8 * PHASE + 2) / 5) * Q + 0 * 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 2) % 5) * STRIDE + ((8 * PHASE + 2) / 5) * Q + 1 * 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 3) % 5) * STRIDE + ((8 * PHASE + 3) / 5) * Q + 0 * 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 3) % 5) * STRIDE + ((8 * PHASE + 3) / 5) * Q + 1 * 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 4) % 5) * STRIDE + ((8 * PHASE + 4) / 5) * Q + 0 * 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 4) % 5) * STRIDE + ((8 * PHASE + 4) / 5) * Q + 1 * 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 5) % 5) * STRIDE + ((8 * PHASE + 5) / 5) * Q + 0 * 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 5) % 5) * STRIDE + ((8 * PHASE + 5) / 5) * Q + 1 * 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 6) % 5) * STRIDE + ((8 * PHASE + 6) / 5) * Q + 0 * 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 6) % 5) * STRIDE + ((8 * PHASE + 6) / 5) * Q + 1 * 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 7) % 5) * STRIDE + ((8 * PHASE + 7) / 5) * Q + 0 * 8)) * sizeof(float),
workBase + uint64_t(SwapXYOffset((6 + (8 * PHASE + 7) % 5) * STRIDE + ((8 * PHASE + 7) / 5) * Q + 1 * 8)) * sizeof(float)
};
StorePacked(mem.PackedVA()[PRIOR_TARGET + 4 * PHASE], target);
if constexpr (!VA_PHASE_ADDS || PHASE == 0) StorePacked(mem.PackedVA()[PRIOR_SOURCE], source);
}
}
template <uint32_t PHASE>
__aicore__ inline void NativePriorPhase() {
constexpr uint32_t PRIOR_TARGET = 132;
constexpr uint32_t PRIOR_SOURCE = 152 + 4 * PHASE;
if constexpr (ANY_QUAD)
PackedTranspose<4, 2, 0>(mem.PackedVA()[PRIOR_TARGET + 4 * PHASE], mem.PackedVA()[PRIOR_SOURCE]);
else if constexpr (PAIR256)
PackedTranspose<4, 1, 40>(mem.PackedVA()[PRIOR_TARGET + 4 * PHASE], mem.PackedVA()[PRIOR_SOURCE]);
else
PackedTranspose<FRAME / 128, 2, 80>(mem.PackedVA()[PRIOR_TARGET + 4 * PHASE],
mem.PackedVA()[PRIOR_SOURCE]);
}
__aicore__ inline void InitPackedVA() {
InitInputPhase<0, 0>();
InitInputPhase<0, 1>();
InitInputPhase<0, 2>();
if constexpr (!RAW3) {
InitInputPhase<0, 3>();
InitInputPhase<0, 4>();
InitInputPhase<0, 5>();
}
if constexpr (sizeof(T) == sizeof(float) && !(ROW)) {
InitInputPhase<1, 0>();
InitInputPhase<1, 1>();
InitInputPhase<1, 2>();
if constexpr (!RAW3) {
InitInputPhase<1, 3>();
InitInputPhase<1, 4>();
InitInputPhase<1, 5>();
}
}
InitOutputPhase<0, 0>();
InitOutputPhase<0, 1>();
InitOutputPhase<0, 2>();
InitOutputPhase<0, 3>();
InitOutputPhase<0, 4>();
if constexpr (sizeof(T) == sizeof(float) && !(ROW)) {
InitOutputPhase<1, 0>();
InitOutputPhase<1, 1>();
InitOutputPhase<1, 2>();
InitOutputPhase<1, 3>();
InitOutputPhase<1, 4>();
}
{
InitPriorPhase<0>();
InitPriorPhase<1>();
InitPriorPhase<2>();
InitPriorPhase<3>();
InitPriorPhase<4>();
}
AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
if constexpr (VA_PHASE_ADDS) {
auto *entries = reinterpret_cast<__ubuf__ int16_t *>(mem.PackedVA().GetPhyAddr());
AscendC::SetMaskCount();
AscendC::SetVectorMask<int16_t, AscendC::MaskMode::COUNTER>(16U);
#pragma unroll
for (uint32_t phase = 1U; phase < 3U; ++phase)
asc_add_scalar(entries + (24U + 4U * phase) * 4U, entries + 24U * 4U, int16_t(phase), uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
#pragma unroll
for (uint32_t phase = 1U; phase < 5U; ++phase)
asc_add_scalar(entries + (92U + 4U * phase) * 4U, entries + 92U * 4U, int16_t(phase), uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
#pragma unroll
for (uint32_t phase = 1U; phase < 5U; ++phase)
asc_add_scalar(entries + (152U + 4U * phase) * 4U, entries + 152U * 4U, int16_t(phase), uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetMaskNorm();
AscendC::ResetMask();
}
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void NativeInputPhase() {
static_assert(PHASE < 6 && SLOT < (sizeof(T) == sizeof(float) ? 2 : 1), "valid packed input phase");
if constexpr (RAW3) {
PackedTranspose<FRAME / 64, 1, 48>(mem.PackedVA()[4 * PHASE],
mem.PackedVA()[24 + SLOT * 24 + 4 * PHASE]);
} else {
PackedTranspose<FRAME / 128, 2, 96>(mem.PackedVA()[4 * PHASE],
mem.PackedVA()[24 + SLOT * 24 + 4 * PHASE]);
}
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void NativeOutputPhase() {
static_assert(PHASE < 5 && SLOT < (sizeof(T) == sizeof(float) ? 2 : 1), "valid packed output phase");
constexpr uint32_t INPUT_WORDS = 72;
PackedTranspose<FRAME / 128, 80, 2>(mem.PackedVA()[INPUT_WORDS + 20 + SLOT * 20 + 4 * PHASE],
mem.PackedVA()[INPUT_WORDS + 4 * PHASE]);
}
public:
template <bool WHOLE = false>
__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded,
const FireFrameTilingData &schedule) {
static_assert(!WHOLE || ROW, "whole entry requires one batch per core");
static_assert(!VA_PHASE_ADDS || (WHOLE && ROW && RAW3 && DIRECT_STRIDE && !PAIR256 && FRAME == 512 && sizeof(T) == sizeof(float)), "fixed float whole phase table");
static_assert(!F32_QUAD128 || (WHOLE && ROW && RAW3 && DIRECT_STRIDE && !PAIR256 && VA_PHASE_ADDS && !HALF_ONE8 && !QUAD128 && FRAME == 512 && sizeof(T) == sizeof(float)), "qualified float quad keeps axpy swap va");
static_assert(!QUAD128 || (PAIR256 && HALF_ONE8), "quad shares complete half pair math");
static_assert(PAD == 16U || (PAD == 8U && QUAD128 && !F32_QUAD128 && WHOLE && ROW && RAW3 && DIRECT_STRIDE && PAIR256 && HALF_ONE8 && !VA_PHASE_ADDS && FRAME == 512U && sizeof(T) == sizeof(half)), "PAD8 only complete half quad128");
static_assert(!HALF_ONE8 || (WHOLE && ROW && RAW3 && DIRECT_STRIDE && PAIR256 && FRAME == 512 && sizeof(T) == sizeof(half)), "qualified half pair one broadcast");
static_assert(!PAIR256 || (WHOLE && ROW && RAW3 && DIRECT_STRIDE && FRAME == 512 && sizeof(T) == sizeof(half)), "two fixed half batches only");
static_assert(!DIRECT_STRIDE || (FRAME == 512 && ROW && WHOLE), "direct stride only original whole512");
constexpr bool rawFirst = WHOLE && DIRECT_STRIDE && RAW3 && FRAME == 512 && sizeof(T) == sizeof(half);
const auto &t = schedule.header;
rawGm.SetGlobalBuffer((__gm__ T *)raw);
priorGm.SetGlobalBuffer((__gm__ float *)priors);
outGm.SetGlobalBuffer((__gm__ T *)decoded);
const uint32_t groupLanes = schedule.geometry & 0xffffU;
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
const uint32_t begin = ANY_QUAD ? block * 128U : PAIR256 ? (block % 16U) * 256U : (block % groupLanes) * t.perCore;
if (begin >= t.n) return;
const uint32_t batchCount = ROW ? 1U : schedule.geometry >> 16;
const uint32_t batchFirst = ANY_QUAD ? 0U : PAIR256 ? (block / 16U) * 2U : (block / groupLanes) * batchCount;
const uint32_t priorPos = begin;
const uint32_t count = WHOLE ? FRAME : (t.n - priorPos < t.perCore ? t.n - priorPos : t.perCore);
// Overlap first raw/prior DMA with address-table setup.
auto firstRaw = mem.AllocRaw();
auto p = mem.AllocPrior();
const AscendC::DataCopyExtParams firstRp = ANY_QUAD ?
AscendC::DataCopyExtParams{4, 128U * 6U * uint32_t(sizeof(T)), (t.n - 128U) * 6U * uint32_t(sizeof(T)), 0, 0} : PAIR256 ?
AscendC::DataCopyExtParams{2, 256U * 6U * uint32_t(sizeof(T)), (t.n - 256U) * 6U * uint32_t(sizeof(T)), 0, 0} :
AscendC::DataCopyExtParams{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0};
const AscendC::DataCopyExtParams firstPp{1, (ANY_QUAD ? 128U : PAIR256 ? 256U : count) * 5 * uint32_t(sizeof(float)), 0, 0, 0};
if constexpr (ANY_QUAD) {
if constexpr (F32_QUAD128)
AscendC::DataCopy(firstRaw, rawGm[priorPos * 6], AscendC::DataCopyParams{4, 96, 2976, 0});
else
AscendC::DataCopy(firstRaw, rawGm[priorPos * 6], AscendC::DataCopyParams{4, 48, 1488, 0});
} else if constexpr (PAIR256) {
AscendC::DataCopy(firstRaw, rawGm[(batchFirst * t.n + priorPos) * 6], AscendC::DataCopyParams{2, 96, 1440, 0});
} else {
AscendC::DataCopyPad(firstRaw, rawGm[(WHOLE ? block * FRAME : batchFirst * t.n + priorPos) * 6], firstRp,
AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
}
if constexpr (rawFirst) mem.RawReady();
AscendC::DataCopyPad(p, priorGm[priorPos * 5], firstPp,
AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
if constexpr (!rawFirst) mem.RawReady();
mem.PriorReady();
auto one = mem.Ones();
AscendC::Duplicate(one, 1.0f, HALF_ONE8 ? 8U : FRAME * 6);
AscendC::LocalTensor<float> f[11];
for (uint32_t k = 0; k < 11; ++k) f[k] = mem.Work()[k * STRIDE];
if constexpr (rawFirst) {
mem.RawWait();
auto rf = mem.RawFloat();
AscendC::Cast(rf, firstRaw, AscendC::RoundMode::CAST_NONE, count * 6);
AscendC::Muls(rf, rf, -1.0f, count * 6);
AscendC::Exp(rf, rf, count * 6);
AscendC::Adds(rf, rf, 1.0f, count * 6);
if constexpr (HALF_ONE8) {
const AscendC::BinaryRepeatParams divOne8{1, 0, 1, 8, 0, 8};
AscendC::Div<float>(rf, one, rf, uint64_t(64), uint8_t(48), divOne8);
} else AscendC::Div(rf, one, rf, count * 6);
AscendC::PipeBarrier<PIPE_V>();
}
InitPackedVA();
{
mem.PriorWait();
NativePriorPhase<0>();
NativePriorPhase<1>();
NativePriorPhase<2>();
NativePriorPhase<3>();
NativePriorPhase<4>();
if constexpr (!(WHOLE && DIRECT_STRIDE && RAW3 && FRAME == 512 && sizeof(T) == sizeof(half)))
AscendC::PipeBarrier<PIPE_V>();
if constexpr (!DIRECT_STRIDE) {
// The original prior AoS has been consumed; copy stride bits exactly.
AscendC::DataCopy(p, f[8], STRIDE);
AscendC::DataCopy(p[STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[2 * STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[3 * STRIDE], f[8], STRIDE);
AscendC::PipeBarrier<PIPE_V>();
}
// Cached f[6..10] are read-only throughout the batch loop.
for (uint32_t batch = batchFirst; batch < batchFirst + batchCount; ++batch) {
const uint32_t pos = F32_QUAD128 ? priorPos : PAIR256 ? batchFirst * t.n + priorPos : (WHOLE ? block * FRAME : batch * t.n + priorPos);
auto r = firstRaw;
if (batch != batchFirst) {
r = mem.AllocRaw();
AscendC::DataCopyExtParams rp{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(r, rawGm[pos * 6], rp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
mem.RawReady();
}
AscendC::LocalTensor<float> rf;
if constexpr (rawFirst) rf = mem.RawFloat();
else {
mem.RawWait();
if constexpr (sizeof(T) == sizeof(float)) rf = r.template ReinterpretCast<float>();
else {
rf = mem.RawFloat();
AscendC::Cast(rf, r, AscendC::RoundMode::CAST_NONE, count * 6);
}
AscendC::Muls(rf, rf, -1.0f, count * 6);
AscendC::Exp(rf, rf, count * 6);
AscendC::Adds(rf, rf, 1.0f, count * 6);
AscendC::Div(rf, one, rf, count * 6);
// Raw/RF has6*FRAME allocated words. Unloaded padding never reaches valid GM output.
AscendC::PipeBarrier<PIPE_V>();
}
if constexpr (sizeof(T) == sizeof(float)) {
if constexpr (ROW) {
NativeInputPhase<0, 0>();
NativeInputPhase<0, 1>();
NativeInputPhase<0, 2>();
if constexpr (!RAW3) {
NativeInputPhase<0, 3>();
NativeInputPhase<0, 4>();
NativeInputPhase<0, 5>();
}
} else if (reinterpret_cast<uint64_t>(rf.GetPhyAddr()) ==
reinterpret_cast<uint64_t>(mem.RawFixedSlot(0).GetPhyAddr())) {
NativeInputPhase<0, 0>();
NativeInputPhase<0, 1>();
NativeInputPhase<0, 2>();
if constexpr (!RAW3) {
NativeInputPhase<0, 3>();
NativeInputPhase<0, 4>();
NativeInputPhase<0, 5>();
}
} else {
NativeInputPhase<1, 0>();
NativeInputPhase<1, 1>();
NativeInputPhase<1, 2>();
if constexpr (!RAW3) {
NativeInputPhase<1, 3>();
NativeInputPhase<1, 4>();
NativeInputPhase<1, 5>();
}
}
} else {
NativeInputPhase<0, 0>();
NativeInputPhase<0, 1>();
NativeInputPhase<0, 2>();
if constexpr (!RAW3) {
NativeInputPhase<0, 3>();
NativeInputPhase<0, 4>();
NativeInputPhase<0, 5>();
}
}
AscendC::PipeBarrier<PIPE_V>();
if constexpr (WHOLE && DIRECT_STRIDE) {
auto *wPtr = reinterpret_cast<__ubuf__ float *>(mem.Work().GetPhyAddr());
AscendC::SetMaskCount();
if constexpr (CENTER_AXPY) {
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(2 * STRIDE);
asc_mul_scalar(wPtr + 2 * STRIDE, wPtr + 2 * STRIDE, 2.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
} else {
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(4 * STRIDE);
asc_mul_scalar(wPtr, wPtr, 2.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
}
AscendC::PipeBarrier<PIPE_V>();
if constexpr (!CENTER_AXPY) AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(2 * STRIDE);
asc_add_scalar(wPtr + (CENTER_AXPY ? 6U : 0U) * STRIDE, wPtr + (CENTER_AXPY ? 6U : 0U) * STRIDE, CENTER_AXPY ? -0.25f : -0.5f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
asc_mul(wPtr + 2 * STRIDE, wPtr + 2 * STRIDE, wPtr + 2 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
if constexpr (CENTER_AXPY)
asc_axpy(wPtr, wPtr + 6 * STRIDE, 2.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
else
asc_add(wPtr, wPtr, wPtr + 6 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
asc_mul(wPtr + 2 * STRIDE, wPtr + 2 * STRIDE, wPtr + 9 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(FRAME);
#pragma unroll
for (uint32_t k = 0; k < 4; ++k) {
constexpr uint32_t priorXY = 0;
const uint32_t field = k < 2 ? k + priorXY : k;
asc_mul(wPtr + field * STRIDE, wPtr + field * STRIDE, wPtr + 8 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
}
asc_mul(wPtr + 4 * STRIDE, wPtr + 4 * STRIDE, SCORE_IN_ONES ? reinterpret_cast<__ubuf__ float *>(mem.Ones().GetPhyAddr()) + 64U : wPtr + 5 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::SetMaskNorm();
AscendC::ResetMask();
} else {
AscendC::Muls(f[0], f[0], 2.0f, 4 * STRIDE);
AscendC::Adds(f[0], f[0], -0.5f, 2 * STRIDE);
AscendC::Add(f[0], f[0], f[6], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[2], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[9], 2 * STRIDE);
if constexpr (DIRECT_STRIDE) {
#pragma unroll
for (uint32_t k = 0; k < 4; ++k)
AscendC::Mul(f[k], f[k], f[8], FRAME);
} else {
AscendC::Mul(f[0], f[0], p, 4 * STRIDE);
}
AscendC::Mul(f[4], f[4], f[5], FRAME);
}
auto out = mem.AllocOut();
AscendC::PipeBarrier<PIPE_V>();
if constexpr (sizeof(T) == sizeof(float)) {
if constexpr (ROW) {
NativeOutputPhase<0, 0>();
NativeOutputPhase<0, 1>();
NativeOutputPhase<0, 2>();
NativeOutputPhase<0, 3>();
NativeOutputPhase<0, 4>();
} else if (mem.CurrentOutSlot() == 0) {
NativeOutputPhase<0, 0>();
NativeOutputPhase<0, 1>();
NativeOutputPhase<0, 2>();
NativeOutputPhase<0, 3>();
NativeOutputPhase<0, 4>();
} else {
NativeOutputPhase<1, 0>();
NativeOutputPhase<1, 1>();
NativeOutputPhase<1, 2>();
NativeOutputPhase<1, 3>();
NativeOutputPhase<1, 4>();
}
} else {
NativeOutputPhase<0, 0>();
NativeOutputPhase<0, 1>();
NativeOutputPhase<0, 2>();
NativeOutputPhase<0, 3>();
NativeOutputPhase<0, 4>();
}
AscendC::PipeBarrier<PIPE_V>();
if constexpr (sizeof(T) != sizeof(float))
AscendC::Cast(out, mem.PackedFloat(), AscendC::RoundMode::CAST_RINT, count * 5);
mem.OutReady(); mem.OutWait();
const AscendC::DataCopyExtParams op = ANY_QUAD ?
AscendC::DataCopyExtParams{4, 128U * 5U * uint32_t(sizeof(T)), 0, (t.n - 128U) * 5U * uint32_t(sizeof(T)), 0} : PAIR256 ?
AscendC::DataCopyExtParams{2, 256U * 5U * uint32_t(sizeof(T)), 0, (t.n - 256U) * 5U * uint32_t(sizeof(T)), 0} :
AscendC::DataCopyExtParams{1, count * 5 * uint32_t(sizeof(T)), 0, 0, 0};
if constexpr (ANY_QUAD) {
if constexpr (F32_QUAD128)
AscendC::DataCopy(outGm[pos * 5], out, AscendC::DataCopyParams{4, 80, 0, 2480});
else
AscendC::DataCopy(outGm[pos * 5], out, AscendC::DataCopyParams{4, 40, 0, 1240});
} else if constexpr (PAIR256) {
AscendC::DataCopy(outGm[pos * 5], out, AscendC::DataCopyParams{2, 80, 0, 1200});
} else {
AscendC::DataCopyPad(outGm[pos * 5], out, op);
}
mem.FreeOut();
if constexpr (!(WHOLE && DIRECT_STRIDE && RAW3 && FRAME == 512 && sizeof(T) == sizeof(half))) {
mem.FreeRaw();
}
}
if constexpr (!(WHOLE && DIRECT_STRIDE && RAW3 && FRAME == 512 && sizeof(T) == sizeof(half))) {
mem.FreePrior();
}
}
mem.Drain();
}
};



//576: only two fixedB8/C40 full832 rows are flattened; all legacy classes stay literal.
template <typename T>
class FirePairStorage576 {
static constexpr uint32_t RAW_BYTES = 1664U * 6U * sizeof(T);
static constexpr uint32_t PRIOR_BASE = 2U * RAW_BYTES;
static constexpr uint32_t PRIOR_BYTES = 4U * 1680U * sizeof(float);
static constexpr uint32_t WORK_BASE = PRIOR_BASE + PRIOR_BYTES + 256U;
static constexpr uint32_t WORK_BYTES = 74496U;
static constexpr uint32_t ONE_BASE = WORK_BASE + WORK_BYTES;
static constexpr uint32_t RF_BASE = ONE_BASE + 32U + (sizeof(T) == sizeof(half) ? 224U : 0U);
static constexpr uint32_t VA_BASE = RF_BASE + (sizeof(T) == sizeof(half) ? 39936U : 0U);
static_assert(VA_BASE + 172U * sizeof(uint64_t) == (sizeof(T) == sizeof(float) ? 182912U : 183136U), "fixed576 UB high water");
static_assert(VA_BASE + 172U * sizeof(uint64_t) < 188416U, "retain SDK UB reserve");
static_assert(WORK_BASE % 512U == 0U && (ONE_BASE - (sizeof(T) == sizeof(float) ? 0U : RF_BASE)) % 512U == 256U, "fixed phases");
uint32_t used = 0;
public:
__aicore__ inline AscendC::LocalTensor<T> RawFixedSlot(uint32_t slot) {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, slot * RAW_BYTES, 9984U);
}
__aicore__ inline AscendC::LocalTensor<T> OutAt(uint32_t slot) {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, slot * RAW_BYTES, 8320U);
}
__aicore__ inline AscendC::LocalTensor<float> Prior() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE, 6720U);
}
__aicore__ inline AscendC::LocalTensor<float> Work() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, WORK_BASE, WORK_BYTES / 4U);
}
__aicore__ inline AscendC::LocalTensor<float> Ones() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE, 8U);
}
__aicore__ inline AscendC::LocalTensor<float> RawFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, RF_BASE, 9984U);
}
__aicore__ inline AscendC::LocalTensor<float> PackedFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, RF_BASE, 8320U);
}
__aicore__ inline AscendC::LocalTensor<uint64_t> PackedVA() {
return AscendC::LocalTensor<uint64_t>(AscendC::TPosition::VECCALC, VA_BASE, 172U);
}
__aicore__ inline AscendC::LocalTensor<T> AllocSlot(uint32_t slot) {
if (used & (1U << slot)) {
AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
// Preserve the MTE3 source until its transfer completes.
AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
}
return RawFixedSlot(slot);
}
__aicore__ inline void PublishOutputAndRelease(uint32_t slot) {
AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);
used |= 1U << slot;
}
__aicore__ inline void Drain() {
for (uint32_t slot = 0; slot < 2; ++slot) if (used & (1U << slot)) {
AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
}
}
};

#define FIRE_TU_KIND_float 1
#define FIRE_TU_KIND_half 2
#define FIRE_TU_KIND_I(t) FIRE_TU_KIND_##t
#define FIRE_TU_KIND(t) FIRE_TU_KIND_I(t)
#if defined(DTYPE_RAW_PRED) && FIRE_TU_KIND(DTYPE_RAW_PRED) == 2
#define FIRE_PAIR_BASE_IMAGE 1
#else
#define FIRE_PAIR_BASE_IMAGE 0
#endif

template <typename T, uint32_t N, bool SCORE_FUSED = false, bool ROTATE_PAIRS = false, bool QUALIFIED_SCORE_ROTATE = false, bool QUALIFIED_HALF_SCORE = false>
class KernelFirePair576 {
static_assert(!SCORE_FUSED || sizeof(T) == sizeof(float) || QUALIFIED_HALF_SCORE, "score fusion requires dtype qualification");
static_assert(!(SCORE_FUSED && ROTATE_PAIRS) || QUALIFIED_SCORE_ROTATE, "636 mechanisms are exclusive");
static_assert(!QUALIFIED_SCORE_ROTATE || (!FIRE_PAIR_BASE_IMAGE && sizeof(T) == sizeof(float) && (N == 32768U || N == 32769U) && SCORE_FUSED && ROTATE_PAIRS), "only qualified float pair fusion rotation");
static_assert(!QUALIFIED_HALF_SCORE || (FIRE_PAIR_BASE_IMAGE && sizeof(T) == sizeof(half) && SCORE_FUSED && !ROTATE_PAIRS), "only qualified half pair score fusion");
static_assert(!(QUALIFIED_SCORE_ROTATE && QUALIFIED_HALF_SCORE), "qualified pair modes are separate");
static_assert(N == 32768U || N == 32769U, "only qualifiedC40 B8 dimensions");
static constexpr uint32_t FRAME = 1664U, STRIDE = 1680U, Q = 208U;
FirePairStorage576<T> mem;
AscendC::GlobalTensor<T> rawGm, outGm;
AscendC::GlobalTensor<float> priorGm;
__aicore__ inline void StorePacked(AscendC::LocalTensor<uint64_t> dst, const uint64_t *address) {
#pragma unroll
for (uint32_t q = 0; q < 4; ++q) {
const uint64_t value = ((address[4*q] >> 5) & 0x1fffULL) |
(((address[4*q+1] >> 5) & 0x1fffULL) << 16) |
(((address[4*q+2] >> 5) & 0x1fffULL) << 32) |
(((address[4*q+3] >> 5) & 0x1fffULL) << 48);
dst.SetValue(q, value);
}
}
template <uint8_t REPEAT, uint16_t DST_STRIDE, uint16_t SRC_STRIDE>
__aicore__ inline void PackedTranspose(AscendC::LocalTensor<uint64_t> dst, AscendC::LocalTensor<uint64_t> src) {
auto *dstPacked = reinterpret_cast<__ubuf__ uint64_t *>(dst.GetPhyAddr());
auto *srcPacked = reinterpret_cast<__ubuf__ uint64_t *>(src.GetPhyAddr());
__cce_scalar::vld_va_reg(VA0, dstPacked, L128);
__cce_scalar::vld_va_reg(VA1, dstPacked, H128);
__cce_scalar::vld_va_reg(VA2, srcPacked, L128);
__cce_scalar::vld_va_reg(VA3, srcPacked, H128);
__cce_scalar::scatter_vnchwconv_b32(VA0, VA2, REPEAT, DST_STRIDE, SRC_STRIDE);
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitInputPhase() {
static_assert(SLOT < (sizeof(T) == sizeof(float) ? 2U : 1U) && PHASE < 3U, "fixed RAW3 slots");
uint64_t sourceBase;
if constexpr (sizeof(T) == sizeof(float)) sourceBase = reinterpret_cast<uint64_t>(mem.RawFixedSlot(SLOT).GetPhyAddr());
else sourceBase = reinterpret_cast<uint64_t>(mem.RawFloat().GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t source[16] = {
sourceBase + uint64_t(0 + 8*PHASE) * 4U,
sourceBase + uint64_t(48 + 8*PHASE) * 4U,
sourceBase + uint64_t(96 + 8*PHASE) * 4U,
sourceBase + uint64_t(144 + 8*PHASE) * 4U,
sourceBase + uint64_t(192 + 8*PHASE) * 4U,
sourceBase + uint64_t(240 + 8*PHASE) * 4U,
sourceBase + uint64_t(288 + 8*PHASE) * 4U,
sourceBase + uint64_t(336 + 8*PHASE) * 4U,
sourceBase + uint64_t(24 + 8*PHASE) * 4U,
sourceBase + uint64_t(72 + 8*PHASE) * 4U,
sourceBase + uint64_t(120 + 8*PHASE) * 4U,
sourceBase + uint64_t(168 + 8*PHASE) * 4U,
sourceBase + uint64_t(216 + 8*PHASE) * 4U,
sourceBase + uint64_t(264 + 8*PHASE) * 4U,
sourceBase + uint64_t(312 + 8*PHASE) * 4U,
sourceBase + uint64_t(360 + 8*PHASE) * 4U
};
uint64_t target[16] = {
workBase + uint64_t(((8*PHASE+0)%6)*STRIDE + (((8*PHASE+0)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+0)%6)*STRIDE + (((8*PHASE+0)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+1)%6)*STRIDE + (((8*PHASE+1)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+1)%6)*STRIDE + (((8*PHASE+1)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+2)%6)*STRIDE + (((8*PHASE+2)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+2)%6)*STRIDE + (((8*PHASE+2)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+3)%6)*STRIDE + (((8*PHASE+3)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+3)%6)*STRIDE + (((8*PHASE+3)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+4)%6)*STRIDE + (((8*PHASE+4)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+4)%6)*STRIDE + (((8*PHASE+4)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+5)%6)*STRIDE + (((8*PHASE+5)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+5)%6)*STRIDE + (((8*PHASE+5)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+6)%6)*STRIDE + (((8*PHASE+6)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+6)%6)*STRIDE + (((8*PHASE+6)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+7)%6)*STRIDE + (((8*PHASE+7)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+7)%6)*STRIDE + (((8*PHASE+7)/6)+4)*Q) * 4U
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[4*PHASE], target);
StorePacked(mem.PackedVA()[24+SLOT*24+4*PHASE], source);
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitOutputPhase() {
static_assert(SLOT < (sizeof(T) == sizeof(float) ? 2U : 1U) && PHASE < 5U, "fixed output slots");
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t outputBase;
if constexpr (sizeof(T) == sizeof(float)) outputBase = reinterpret_cast<uint64_t>(mem.OutAt(SLOT).GetPhyAddr());
else outputBase = reinterpret_cast<uint64_t>(mem.PackedFloat().GetPhyAddr());
uint64_t source[16] = {
workBase + uint64_t(((8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 8) * 4U
};
uint64_t target[16] = {
outputBase + uint64_t((5*0+PHASE)*8) * 4U,
outputBase + uint64_t((5*8+PHASE)*8) * 4U,
outputBase + uint64_t((5*1+PHASE)*8) * 4U,
outputBase + uint64_t((5*9+PHASE)*8) * 4U,
outputBase + uint64_t((5*2+PHASE)*8) * 4U,
outputBase + uint64_t((5*10+PHASE)*8) * 4U,
outputBase + uint64_t((5*3+PHASE)*8) * 4U,
outputBase + uint64_t((5*11+PHASE)*8) * 4U,
outputBase + uint64_t((5*4+PHASE)*8) * 4U,
outputBase + uint64_t((5*12+PHASE)*8) * 4U,
outputBase + uint64_t((5*5+PHASE)*8) * 4U,
outputBase + uint64_t((5*13+PHASE)*8) * 4U,
outputBase + uint64_t((5*6+PHASE)*8) * 4U,
outputBase + uint64_t((5*14+PHASE)*8) * 4U,
outputBase + uint64_t((5*7+PHASE)*8) * 4U,
outputBase + uint64_t((5*15+PHASE)*8) * 4U
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[72+4*PHASE], source);
StorePacked(mem.PackedVA()[92+SLOT*20+4*PHASE], target);
}
template <uint32_t PHASE>
__aicore__ inline void InitPriorPhase() {
const uint64_t priorBase = reinterpret_cast<uint64_t>(mem.Prior().GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t source[16] = {
priorBase + uint64_t(0+8*PHASE) * 4U,
priorBase + uint64_t(40+8*PHASE) * 4U,
priorBase + uint64_t(80+8*PHASE) * 4U,
priorBase + uint64_t(120+8*PHASE) * 4U,
priorBase + uint64_t(160+8*PHASE) * 4U,
priorBase + uint64_t(200+8*PHASE) * 4U,
priorBase + uint64_t(240+8*PHASE) * 4U,
priorBase + uint64_t(280+8*PHASE) * 4U,
priorBase + uint64_t(0+8*PHASE) * 4U,
priorBase + uint64_t(40+8*PHASE) * 4U,
priorBase + uint64_t(80+8*PHASE) * 4U,
priorBase + uint64_t(120+8*PHASE) * 4U,
priorBase + uint64_t(160+8*PHASE) * 4U,
priorBase + uint64_t(200+8*PHASE) * 4U,
priorBase + uint64_t(240+8*PHASE) * 4U,
priorBase + uint64_t(280+8*PHASE) * 4U
};
uint64_t target[16] = {
workBase + uint64_t((6+(8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 104) * 4U,
workBase + uint64_t((6+(8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 104) * 4U,
workBase + uint64_t((6+(8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 104) * 4U,
workBase + uint64_t((6+(8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 104) * 4U,
workBase + uint64_t((6+(8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 104) * 4U,
workBase + uint64_t((6+(8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 104) * 4U,
workBase + uint64_t((6+(8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 104) * 4U,
workBase + uint64_t((6+(8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 104) * 4U
};
StorePacked(mem.PackedVA()[132+4*PHASE], target);
StorePacked(mem.PackedVA()[152+4*PHASE], source);
}
__aicore__ inline void InitPackedVA() {
InitInputPhase<0, 0>();
InitInputPhase<0, 1>();
InitInputPhase<0, 2>();
InitOutputPhase<0, 0>();
InitOutputPhase<0, 1>();
InitOutputPhase<0, 2>();
InitOutputPhase<0, 3>();
InitOutputPhase<0, 4>();
if constexpr (sizeof(T) == sizeof(float)) {
InitInputPhase<1, 0>();
InitInputPhase<1, 1>();
InitInputPhase<1, 2>();
InitOutputPhase<1, 0>();
InitOutputPhase<1, 1>();
InitOutputPhase<1, 2>();
InitOutputPhase<1, 3>();
InitOutputPhase<1, 4>();
}
InitPriorPhase<0>();
InitPriorPhase<1>();
InitPriorPhase<2>();
InitPriorPhase<3>();
InitPriorPhase<4>();
AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
}
template <uint32_t SLOT>
__aicore__ inline void Input() {
PackedTranspose<26, 1, 48>(mem.PackedVA()[0], mem.PackedVA()[24+SLOT*24+0]);
PackedTranspose<26, 1, 48>(mem.PackedVA()[4], mem.PackedVA()[24+SLOT*24+4]);
PackedTranspose<26, 1, 48>(mem.PackedVA()[8], mem.PackedVA()[24+SLOT*24+8]);
}
template <uint32_t SLOT>
__aicore__ inline void Output() {
PackedTranspose<13, 80, 2>(mem.PackedVA()[92+SLOT*20+0], mem.PackedVA()[72+0]);
PackedTranspose<13, 80, 2>(mem.PackedVA()[92+SLOT*20+4], mem.PackedVA()[72+4]);
PackedTranspose<13, 80, 2>(mem.PackedVA()[92+SLOT*20+8], mem.PackedVA()[72+8]);
PackedTranspose<13, 80, 2>(mem.PackedVA()[92+SLOT*20+12], mem.PackedVA()[72+12]);
PackedTranspose<13, 80, 2>(mem.PackedVA()[92+SLOT*20+16], mem.PackedVA()[72+16]);
}
public:
__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded) {
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
if (block >= 39U) return;
const uint32_t begin = block * 832U;
rawGm.SetGlobalBuffer((__gm__ T *)raw);
priorGm.SetGlobalBuffer((__gm__ float *)priors);
outGm.SetGlobalBuffer((__gm__ T *)decoded);
const AscendC::DataCopyExtParams rp{2, 832U*6U*uint32_t(sizeof(T)), (N-832U)*6U*uint32_t(sizeof(T)), 0, 0};
const AscendC::DataCopyExtParams pp{1, 4160U*4U, 0, 0, 0};
const uint32_t cohort = ROTATE_PAIRS ? (block & 3U) : 0U;
auto firstRaw = mem.AllocSlot(0);
auto p = mem.Prior();
AscendC::DataCopyPad(firstRaw, rawGm[(2U*cohort*N+begin)*6U], rp, AscendC::DataCopyPadExtParams<T>{false,0,0,0});
AscendC::DataCopyPad(p, priorGm[begin*5U], pp, AscendC::DataCopyPadExtParams<float>{false,0,0,0});
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
auto one = mem.Ones();
AscendC::Duplicate(one, 1.0f, 8U);
AscendC::LocalTensor<float> f[11];
for (uint32_t k=0;k<11;++k) f[k]=mem.Work()[k*STRIDE];
InitPackedVA();
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
PackedTranspose<13, 1, 40>(mem.PackedVA()[132+0], mem.PackedVA()[152+0]);
PackedTranspose<13, 1, 40>(mem.PackedVA()[132+4], mem.PackedVA()[152+4]);
PackedTranspose<13, 1, 40>(mem.PackedVA()[132+8], mem.PackedVA()[152+8]);
PackedTranspose<13, 1, 40>(mem.PackedVA()[132+12], mem.PackedVA()[152+12]);
PackedTranspose<13, 1, 40>(mem.PackedVA()[132+16], mem.PackedVA()[152+16]);
AscendC::PipeBarrier<PIPE_V>();
#if FIRE_PAIR_BASE_IMAGE
AscendC::DataCopy(p, f[8], STRIDE);
AscendC::DataCopy(p[STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[2*STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[3*STRIDE], f[8], STRIDE);
AscendC::PipeBarrier<PIPE_V>();
#else
if constexpr (sizeof(T) != sizeof(float)) {
AscendC::DataCopy(p, f[8], STRIDE);
AscendC::DataCopy(p[STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[2*STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[3*STRIDE], f[8], STRIDE);
AscendC::PipeBarrier<PIPE_V>();
}
#endif
for (uint32_t iteration=0;iteration<4;++iteration) {
const uint32_t pair = ROTATE_PAIRS ? ((iteration+cohort)&3U) : iteration;
const uint32_t slot=iteration&1U;
const uint32_t pos=pair*2U*N+begin;
auto r=firstRaw;
if (iteration!=0) {
r=mem.AllocSlot(slot);
AscendC::DataCopyPad(r,rawGm[pos*6U],rp,AscendC::DataCopyPadExtParams<T>{false,0,0,0});
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot);
}
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);
AscendC::LocalTensor<float> rf;
if constexpr(sizeof(T)==sizeof(float)) rf=r.template ReinterpretCast<float>();
else {rf=mem.RawFloat();AscendC::Cast(rf,r,AscendC::RoundMode::CAST_NONE,9984U);}
AscendC::Muls(rf,rf,-1.0f,9984U);
AscendC::Exp(rf,rf,9984U);
AscendC::Adds(rf,rf,1.0f,9984U);
const AscendC::BinaryRepeatParams divBinary{1,0,1,8,0,8};
if constexpr (!SCORE_FUSED) AscendC::Div<float>(rf,one,rf,uint64_t(64),uint8_t(156),divBinary);
AscendC::PipeBarrier<PIPE_V>();
if constexpr(sizeof(T)==sizeof(float)) {if(slot==0)Input<0>();else Input<1>();}
else Input<0>();
AscendC::PipeBarrier<PIPE_V>();
if constexpr (SCORE_FUSED) {
AscendC::Mul(f[4],f[4],f[5],FRAME);
AscendC::Div<float>(mem.Work(),one,mem.Work(),uint64_t(64),uint8_t(132),divBinary);
AscendC::PipeBarrier<PIPE_V>();
}
AscendC::Muls(f[0],f[0],2.0f,4*STRIDE);
AscendC::Adds(f[0],f[0],-0.5f,2*STRIDE);
AscendC::Add(f[0],f[0],f[6],2*STRIDE);
AscendC::Mul(f[2],f[2],f[2],2*STRIDE);
AscendC::Mul(f[2],f[2],f[9],2*STRIDE);
#if FIRE_PAIR_BASE_IMAGE
AscendC::Mul(f[0],f[0],p,4*STRIDE);
#else
if constexpr (sizeof(T) == sizeof(float)) {
AscendC::PipeBarrier<PIPE_V>();
auto *wPtr = reinterpret_cast<__ubuf__ float *>(mem.Work().GetPhyAddr());
AscendC::SetMaskCount();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(STRIDE);
#pragma unroll
for (uint32_t k=0;k<4;++k)
asc_mul(wPtr+k*STRIDE, wPtr+k*STRIDE, wPtr+8*STRIDE,
uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(8),uint8_t(8),uint8_t(8));
AscendC::SetMaskNorm();
AscendC::ResetMask();
} else {
AscendC::Mul(f[0],f[0],p,4*STRIDE);
}
#endif
if constexpr (!SCORE_FUSED) AscendC::Mul(f[4],f[4],f[5],FRAME);
AscendC::PipeBarrier<PIPE_V>();
if constexpr(sizeof(T)==sizeof(float)) {if(slot==0)Output<0>();else Output<1>();}
else Output<0>();
AscendC::PipeBarrier<PIPE_V>();
auto out=mem.OutAt(slot);
if constexpr(sizeof(T)!=sizeof(float)) {
AscendC::Cast(out,mem.PackedFloat(),AscendC::RoundMode::CAST_RINT,8320U);
// Explicit WAR: next input Cast waits until packedRF consumption completes.
AscendC::PipeBarrier<PIPE_V>();
}
AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot);
AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
const AscendC::DataCopyExtParams op{2,832U*5U*uint32_t(sizeof(T)),0,(N-832U)*5U*uint32_t(sizeof(T)),0};
AscendC::DataCopyPad(outGm[pos*5U],out,op);
mem.PublishOutputAndRelease(slot);
}
mem.Drain();
}
};
#if FIRE_PAIR_BASE_IMAGE
#define FIRE_PAIR_QUAL11_IMAGE 0
#define FIRE_PAIR_HALF_SCORE_IMAGE 1
#else
#define FIRE_PAIR_QUAL11_IMAGE 1
#define FIRE_PAIR_HALF_SCORE_IMAGE 0
#endif
#undef FIRE_PAIR_BASE_IMAGE
#undef FIRE_TU_KIND
#undef FIRE_TU_KIND_I
#undef FIRE_TU_KIND_half
#undef FIRE_TU_KIND_float


// Private REUSE3064: total stores rowLanes, n/perCore retain the public row metadata.
// D1 direct tensors match the original64 resource addresses; no reusable slot state.
template <typename T>
class FireRow64Storage {
static constexpr uint32_t RAW_BYTES = 384 * sizeof(T);
static constexpr uint32_t PRIOR_BASE = RAW_BYTES;
static constexpr uint32_t OUT_BASE = PRIOR_BASE + 320 * sizeof(float);
static constexpr uint32_t WORK_BASE = OUT_BASE + 320 * sizeof(T);
static constexpr uint32_t ONE_BASE = WORK_BASE + 3840;
static constexpr uint32_t RF_BASE = ONE_BASE + 384 * sizeof(float) + (sizeof(T) == sizeof(half) ? 256 : 0);
static constexpr uint32_t PACK_BASE = RF_BASE + (sizeof(T) == sizeof(half) ? 384 * sizeof(float) : 0);
static constexpr uint32_t INDEX_BASE = PACK_BASE + (sizeof(T) == sizeof(half) ? 320 * sizeof(float) : 0);
static_assert(INDEX_BASE + 448 * sizeof(uint32_t) == (sizeof(T) == sizeof(float) ? 11264 : 12928),
"bounded direct Row64 storage");
static_assert(INDEX_BASE + 448 * sizeof(uint32_t) < 188416, "preserve SDK temporary UB");
static_assert((RAW_BYTES | PRIOR_BASE | OUT_BASE | WORK_BASE | ONE_BASE | RF_BASE | PACK_BASE | INDEX_BASE) % 32 == 0,
"direct Row64 views retain32B alignment");
static_assert(sizeof(T) == sizeof(float) ? ONE_BASE % 512 == 256 : (RF_BASE - ONE_BASE) % 512 == 256,
"native Div sources retain opposite bank groups");
public:
__aicore__ inline AscendC::LocalTensor<T> Raw() {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, 0, 384);
}
__aicore__ inline AscendC::LocalTensor<float> Prior() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE, 320);
}
__aicore__ inline AscendC::LocalTensor<T> Out() {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, OUT_BASE, 320);
}
__aicore__ inline AscendC::LocalTensor<float> Work() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, WORK_BASE, 960);
}
__aicore__ inline AscendC::LocalTensor<float> Ones() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE, 384);
}
__aicore__ inline AscendC::LocalTensor<float> RawFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, RF_BASE, 384);
}
__aicore__ inline AscendC::LocalTensor<float> PackedFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, PACK_BASE, 320);
}
__aicore__ inline AscendC::LocalTensor<int32_t> Indices() {
return AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, INDEX_BASE, 448);
}
};

template <typename T>
class KernelFireRowTight64 {
static constexpr uint32_t STRIDE = 72;
FireRow64Storage<T> mem;
AscendC::GlobalTensor<T> rawGm, outGm;
AscendC::GlobalTensor<float> priorGm;
__aicore__ inline uint64_t Pair(uint32_t lo, uint32_t hi) {
return uint64_t(lo) | (uint64_t(hi) << 32);
}
__aicore__ inline void MakeIndices() {
auto rawOffsets = mem.Indices();
auto priorOffsets = rawOffsets[64];
auto output = rawOffsets[128];
auto rawSeed = rawOffsets.template ReinterpretCast<uint64_t>();
auto outputSeed = output.template ReinterpretCast<uint64_t>();
#pragma unroll
for (uint32_t q = 0; q < 4; ++q)
rawSeed.SetValue(q, Pair(8 * q, 8 * q + 4));
#pragma unroll
for (uint32_t q = 0; q < 20; ++q) {
const uint32_t i = 2 * q;
const uint32_t lo = ((i % 5) * STRIDE + i / 5) * 4;
const uint32_t hi = (((i + 1) % 5) * STRIDE + (i + 1) / 5) * 4;
outputSeed.SetValue(q, Pair(lo, hi));
}
AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
#pragma unroll
for (uint32_t width = 8; width < 64; width *= 2) {
AscendC::Adds<int32_t>(rawOffsets[width], rawOffsets, int32_t(width * 4), width);
AscendC::Adds<int32_t>(output[5 * width], output, int32_t(width * 4), 5 * width);
AscendC::PipeBarrier<PIPE_V>();
}
AscendC::Muls<int32_t>(priorOffsets, rawOffsets, int32_t(5), 64);
AscendC::Muls<int32_t>(rawOffsets, rawOffsets, int32_t(6), 64);
}
public:
__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded,
const FireCandidateDecodeFusionTilingData &t) {
rawGm.SetGlobalBuffer((__gm__ T *)raw);
priorGm.SetGlobalBuffer((__gm__ float *)priors);
outGm.SetGlobalBuffer((__gm__ T *)decoded);
const uint32_t rowLanes = t.total;
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
const uint32_t batch = block / rowLanes;
const uint32_t lane = block - batch * rowLanes;
const uint32_t priorPos = lane * t.perCore;
if (priorPos >= t.n) return;
const uint32_t count = t.n - priorPos < t.perCore ? t.n - priorPos : t.perCore;
const uint32_t pos = batch * t.n + priorPos;
auto r = mem.Raw();
auto p = mem.Prior();
AscendC::DataCopyExtParams rp{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyExtParams pp{1, count * 5 * uint32_t(sizeof(float)), 0, 0, 0};
AscendC::DataCopyPad(r, rawGm[pos * 6], rp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
AscendC::DataCopyPad(p, priorGm[priorPos * 5], pp, AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
MakeIndices();
auto one = mem.Ones();
AscendC::Duplicate(one, 1.0f, 384);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
auto indices = mem.Indices().template ReinterpretCast<uint32_t>();
auto work = mem.Work();
AscendC::LocalTensor<float> f[11];
#pragma unroll
for (uint32_t k = 0; k < 11; ++k) f[k] = work[k * STRIDE];
AscendC::LocalTensor<float> rf;
if constexpr (sizeof(T) == sizeof(float)) rf = r.template ReinterpretCast<float>();
else {
rf = mem.RawFloat();
AscendC::Cast(rf, r, AscendC::RoundMode::CAST_NONE, count * 6);
}
AscendC::Muls(rf, rf, -1.0f, count * 6);
AscendC::Exp(rf, rf, count * 6);
AscendC::Adds(rf, rf, 1.0f, count * 6);
AscendC::Div(rf, one, rf, count * 6);
#pragma unroll
for (uint32_t k = 0; k < 6; ++k)
AscendC::Gather(f[k], rf, indices, k * 4, count);
#pragma unroll
for (uint32_t k = 0; k < 5; ++k)
AscendC::Gather(f[k + 6], p, indices[64], k * 4, count);
const uint64_t mask = count;
const AscendC::UnaryRepeatParams fieldUnary{1, 1, 9, 9};
const AscendC::BinaryRepeatParams fieldBinary{1, 1, 1, 9, 9, 9};
const AscendC::BinaryRepeatParams strideBinary{1, 1, 1, 9, 9, 0};
AscendC::Muls<float>(f[0], f[0], 2.0f, mask, uint8_t(4), fieldUnary);
AscendC::Adds<float>(f[0], f[0], -0.5f, mask, uint8_t(2), fieldUnary);
AscendC::Add<float>(f[0], f[0], f[6], mask, uint8_t(2), fieldBinary);
AscendC::Mul<float>(f[2], f[2], f[2], mask, uint8_t(2), fieldBinary);
AscendC::Mul<float>(f[2], f[2], f[9], mask, uint8_t(2), fieldBinary);
AscendC::Mul<float>(f[0], f[0], f[8], mask, uint8_t(4), strideBinary);
AscendC::Mul<float>(f[4], f[4], f[5], mask, uint8_t(1), fieldBinary);
AscendC::ResetMask();
auto out = mem.Out();
if constexpr (sizeof(T) == sizeof(float)) {
AscendC::Gather(out.template ReinterpretCast<float>(), work, indices[128], 0, count * 5);
} else {
auto packed = mem.PackedFloat();
AscendC::Gather(packed, work, indices[128], 0, count * 5);
AscendC::Cast(out, packed, AscendC::RoundMode::CAST_RINT, count * 5);
}
AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
AscendC::DataCopyExtParams op{1, count * 5 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(outGm[pos * 5], out, op);
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
}
};





template <typename T>
class FireHostComposite32Storage {
static constexpr uint32_t RAW_BYTES = 192 * sizeof(T);
static constexpr uint32_t PRIOR_BASE = RAW_BYTES;
static constexpr uint32_t OUT_BASE = PRIOR_BASE + 160 * sizeof(float);
static constexpr uint32_t WORK_BASE = OUT_BASE + 160 * sizeof(T);
static constexpr uint32_t ONE_BASE = WORK_BASE + 2304;
static constexpr uint32_t RF_BASE = ONE_BASE + 192 * sizeof(float);
static constexpr uint32_t PACK_BASE = RF_BASE + (sizeof(T) == sizeof(half) ? 192 * sizeof(float) : 0);
static constexpr uint32_t INDEX_BASE = PACK_BASE + (sizeof(T) == sizeof(half) ? 160 * sizeof(float) : 0);
static_assert(INDEX_BASE + 512 * sizeof(uint32_t) == (sizeof(T) == sizeof(float) ? 7168 : 7872),
"bounded static32 storage");
static_assert(INDEX_BASE + 512 * sizeof(uint32_t) < 188416, "preserve SDK temporary UB");
static_assert((RAW_BYTES | PRIOR_BASE | OUT_BASE | WORK_BASE | ONE_BASE | RF_BASE | PACK_BASE | INDEX_BASE) % 32 == 0,
"static32 UB views retain32B alignment");
static_assert(sizeof(T) == sizeof(float) ? ONE_BASE % 512 == 256 : (RF_BASE - ONE_BASE) % 512 == 256,
"native Div sources retain opposite bank groups");
public:
__aicore__ inline AscendC::LocalTensor<T> Raw() {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, 0, 192);
}
__aicore__ inline AscendC::LocalTensor<float> Prior() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE, 160);
}
__aicore__ inline AscendC::LocalTensor<T> Out() {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, OUT_BASE, 160);
}
__aicore__ inline AscendC::LocalTensor<float> Work() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, WORK_BASE, 576);
}
__aicore__ inline AscendC::LocalTensor<float> Ones() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE, 192);
}
__aicore__ inline AscendC::LocalTensor<float> RawFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, RF_BASE, 192);
}
__aicore__ inline AscendC::LocalTensor<float> PackedFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, PACK_BASE, 160);
}
// GM DMA uses the public VECIN contract; Gather sees the same physical index bytes.
__aicore__ inline AscendC::LocalTensor<uint32_t> IndexInput() {
return AscendC::LocalTensor<uint32_t>(AscendC::TPosition::VECIN, INDEX_BASE, 512);
}
__aicore__ inline AscendC::LocalTensor<int32_t> Indices() {
return AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, INDEX_BASE, 512);
}
};





template <typename T>
class KernelFireSmallComposite {
FireHostComposite32Storage<T> mem;
AscendC::GlobalTensor<T> rawGm, outGm;
AscendC::GlobalTensor<float> priorGm;
AscendC::GlobalTensor<uint32_t> tableGm;
template <uint32_t STRIDE>
__aicore__ inline void RunBlock(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded,
GM_ADDR tiling, uint32_t pos, uint32_t priorPos, uint32_t count, bool tail) {
rawGm.SetGlobalBuffer((__gm__ T *)raw);
priorGm.SetGlobalBuffer((__gm__ float *)priors);
outGm.SetGlobalBuffer((__gm__ T *)decoded);
tableGm.SetGlobalBuffer((__gm__ uint32_t *)(tiling + (tail ? offsetof(FireSmallCompositeTilingData, tail) : offsetof(FireSmallCompositeTilingData, full))));
auto r = mem.Raw();
auto p = mem.Prior();
auto indices = mem.Indices().template ReinterpretCast<uint32_t>();
AscendC::DataCopyExtParams rp{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyExtParams pp{1, count * 5 * uint32_t(sizeof(float)), 0, 0, 0};
AscendC::DataCopyPad(r, rawGm[pos * 6], rp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
AscendC::DataCopyPad(p, priorGm[priorPos * 5], pp, AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
// Publish the later index DMA separately: activation only requires the first fence.
AscendC::DataCopyExtParams ip{1, 16 * STRIDE * uint32_t(sizeof(uint32_t)), 0, 0, 0};
AscendC::DataCopyPad(mem.IndexInput(), tableGm, ip, AscendC::DataCopyPadExtParams<uint32_t>{false, 0, 0, 0});
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
auto one = mem.Ones();
AscendC::Duplicate(one, 1.0f, count * 6);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
auto work = mem.Work();
AscendC::LocalTensor<float> f[11];
#pragma unroll
for (uint32_t k = 0; k < 11; ++k) f[k] = work[k * STRIDE];
AscendC::LocalTensor<float> rf;
if constexpr (sizeof(T) == sizeof(float)) rf = r.template ReinterpretCast<float>();
else {
rf = mem.RawFloat();
AscendC::Cast(rf, r, AscendC::RoundMode::CAST_NONE, count * 6);
}
AscendC::Muls(rf, rf, -1.0f, count * 6);
AscendC::Exp(rf, rf, count * 6);
AscendC::Adds(rf, rf, 1.0f, count * 6);
AscendC::Div(rf, one, rf, count * 6);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
AscendC::Gather(work, rf, indices, 0, 6 * STRIDE);
AscendC::Gather(f[6], p, indices[6 * STRIDE], 0, 5 * STRIDE);
// Compact contiguous planes retain every FP32 node, including four stride uses.
AscendC::Muls(f[0], f[0], 2.0f, 4 * STRIDE);
AscendC::Adds(f[0], f[0], -0.5f, 2 * STRIDE);
AscendC::Add(f[0], f[0], f[6], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[2], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[9], 2 * STRIDE);
const AscendC::BinaryRepeatParams strideBinary{1, 1, 1, STRIDE / 8, STRIDE / 8, 0};
AscendC::Mul<float>(f[0], f[0], f[8], uint64_t(STRIDE), uint8_t(4), strideBinary);
AscendC::Mul(f[4], f[4], f[5], STRIDE);
AscendC::ResetMask();
auto out = mem.Out();
if constexpr (sizeof(T) == sizeof(float)) {
AscendC::Gather(out.template ReinterpretCast<float>(), work, indices[11 * STRIDE], 0, count * 5);
} else {
auto packed = mem.PackedFloat();
AscendC::Gather(packed, work, indices[11 * STRIDE], 0, count * 5);
AscendC::Cast(out, packed, AscendC::RoundMode::CAST_RINT, count * 5);
}
AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
AscendC::DataCopyExtParams op{1, count * 5 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(outGm[pos * 5], out, op);
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
}
public:
__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded,
GM_ADDR tiling, const FireCandidateDecodeFusionTilingData &t) {
const uint32_t rowTiles = (t.n + 31) / 32;
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
if (block >= (t.total / t.n) * rowTiles) return;
const uint32_t priorPos = (block % rowTiles) * 32;
const uint32_t pos = (block / rowTiles) * t.n + priorPos;
const uint32_t count = t.n - priorPos < 32 ? t.n - priorPos : 32;
if (count <= 8) RunBlock<8>(raw, priors, decoded, tiling, pos, priorPos, count, true);
else if (count <= 16) RunBlock<16>(raw, priors, decoded, tiling, pos, priorPos, count, true);
else if (count <= 24) RunBlock<24>(raw, priors, decoded, tiling, pos, priorPos, count, true);
else RunBlock<32>(raw, priors, decoded, tiling, pos, priorPos, count, count != 32);
}
};

template <typename T>
class FireFlatCompositeStorage {
static constexpr uint32_t RAW_BYTES = 192 * sizeof(T);
static constexpr uint32_t PRIOR_BASE = RAW_BYTES;
static constexpr uint32_t OUT_BASE = PRIOR_BASE + 288 * sizeof(float);
static constexpr uint32_t WORK_BASE = OUT_BASE + 160 * sizeof(T);
static constexpr uint32_t ONE_BASE = WORK_BASE + 2304;
static constexpr uint32_t RF_BASE = ONE_BASE + 192 * sizeof(float);
static constexpr uint32_t PACK_BASE = RF_BASE + (sizeof(T) == sizeof(half) ? 192 * sizeof(float) : 0);
static constexpr uint32_t INDEX_BASE = PACK_BASE + (sizeof(T) == sizeof(half) ? 160 * sizeof(float) : 0);
static_assert(INDEX_BASE + 512 * sizeof(uint32_t) == (sizeof(T) == sizeof(float) ? 7680 : 8384),
"bounded split32 storage");
static_assert(INDEX_BASE + 512 * sizeof(uint32_t) < 188416, "preserve SDK temporary UB");
static_assert((RAW_BYTES | PRIOR_BASE | OUT_BASE | WORK_BASE | ONE_BASE | RF_BASE | PACK_BASE | INDEX_BASE) % 32 == 0,
"split32 UB views retain32B alignment");
static_assert(sizeof(T) == sizeof(float) ? ONE_BASE % 512 == 256 : (RF_BASE - ONE_BASE) % 512 == 256,
"native Div sources retain opposite bank groups");
public:
__aicore__ inline AscendC::LocalTensor<T> Raw() {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, 0, 192);
}
__aicore__ inline AscendC::LocalTensor<float> Prior() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE, 288);
}
__aicore__ inline AscendC::LocalTensor<T> Out() {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, OUT_BASE, 160);
}
__aicore__ inline AscendC::LocalTensor<float> Work() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, WORK_BASE, 576);
}
__aicore__ inline AscendC::LocalTensor<float> Ones() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE, 192);
}
__aicore__ inline AscendC::LocalTensor<float> RawFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, RF_BASE, 192);
}
__aicore__ inline AscendC::LocalTensor<float> PackedFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, PACK_BASE, 160);
}
// The table DMA uses VECIN; Gather reads the identical physical bytes.
__aicore__ inline AscendC::LocalTensor<uint32_t> IndexInput() {
return AscendC::LocalTensor<uint32_t>(AscendC::TPosition::VECIN, INDEX_BASE, 512);
}
__aicore__ inline AscendC::LocalTensor<int32_t> Indices() {
return AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, INDEX_BASE, 512);
}
};

template <typename T, bool SINGLE = false, bool FULL = false, typename Storage = FireFlatCompositeStorage<T>, bool CAPI_PIPE = false, bool PACKET_GM = false, bool EARLY_RAW_C1 = false>
class KernelFireFlatComposite {
static_assert(!PACKET_GM || (CAPI_PIPE && !SINGLE && !FULL), "556 fixed C3 only");
static_assert(!EARLY_RAW_C1 || (CAPI_PIPE && SINGLE && FULL && sizeof(T)==sizeof(half)), "591 halfC1 only");
Storage mem;
AscendC::GlobalTensor<T> rawGm, outGm;
AscendC::GlobalTensor<float> priorGm;
AscendC::GlobalTensor<uint32_t> tableGm;
template <uint32_t STRIDE>
__aicore__ inline void RunBlock(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded,
GM_ADDR tiling, uint32_t pos, uint32_t priorStart, uint32_t priorCount, uint32_t count, uint32_t tableOffset) {
rawGm.SetGlobalBuffer((__gm__ T *)raw);
priorGm.SetGlobalBuffer((__gm__ float *)priors);
outGm.SetGlobalBuffer((__gm__ T *)decoded);
if constexpr (PACKET_GM) {
static_assert(STRIDE == 32 || STRIDE == 8, "556 full32 and tail8");
tableGm.SetGlobalBuffer(const_cast<__gm__ uint32_t *>(FirePacketIndexGM556 + (tableOffset - 32) / 4));
} else if constexpr (CAPI_PIPE && SINGLE && FULL && STRIDE == 32) {
// Read-only constant GM table.
tableGm.SetGlobalBuffer(const_cast<__gm__ uint32_t *>(FireSmallIndexGM548));
} else {
tableGm.SetGlobalBuffer((__gm__ uint32_t *)(tiling + tableOffset));
}
auto r = mem.Raw();
auto p = mem.Prior();
auto indices = mem.Indices().template ReinterpretCast<uint32_t>();
if constexpr (CAPI_PIPE && SINGLE && FULL && STRIDE == 32) {
//546: this path is one complete S32 block; typed GM addresses need only element alignment.
AscendC::DataCopy(r, rawGm[pos * 6], uint32_t(192));
if constexpr (EARLY_RAW_C1) AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
AscendC::DataCopy(p, priorGm[priorStart * 5], uint32_t(160));
} else {
AscendC::DataCopyExtParams rp{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyExtParams pp{1, priorCount * 5 * uint32_t(sizeof(float)), 0, 0, 0};
AscendC::DataCopyPad(r, rawGm[pos * 6], rp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
AscendC::DataCopyPad(p, priorGm[priorStart * 5], pp, AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
}
if constexpr (!EARLY_RAW_C1) AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
// Publish the later index DMA separately: activation only requires the first fence.
if constexpr (CAPI_PIPE && SINGLE && FULL && STRIDE == 32) {
AscendC::DataCopy(mem.IndexInput(), tableGm, uint32_t(512));
} else {
AscendC::DataCopyExtParams ip{1, 16 * STRIDE * uint32_t(sizeof(uint32_t)), 0, 0, 0};
AscendC::DataCopyPad(mem.IndexInput(), tableGm, ip, AscendC::DataCopyPadExtParams<uint32_t>{false, 0, 0, 0});
}
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
if constexpr (CAPI_PIPE && PACKET_GM && STRIDE == 8) {
static_assert(STRIDE == 8, "fixed packet tail planes");
auto one = mem.Ones();
auto work = mem.Work();
auto out = mem.Out();
AscendC::LocalTensor<float> rf;
if constexpr (sizeof(T) == sizeof(float)) rf = r.template ReinterpretCast<float>();
else rf = mem.RawFloat();
auto rfPtr = (__ubuf__ float *)rf.GetPhyAddr();
auto onePtr = (__ubuf__ float *)one.GetPhyAddr();
auto wPtr = (__ubuf__ float *)work.GetPhyAddr();
auto pPtr = (__ubuf__ float *)p.GetPhyAddr();
auto indexPtr = (__ubuf__ uint32_t *)indices.GetPhyAddr();
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetMaskCount();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(12);
asc_duplicate(onePtr, 1.0f, uint8_t(1), uint16_t(1), uint16_t(8));
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
if constexpr (sizeof(T) != sizeof(float)) {
asc_half2float(rfPtr, (__ubuf__ half *)r.GetPhyAddr(), uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(4));
AscendC::PipeBarrier<PIPE_V>();
}
asc_mul_scalar(rfPtr, rfPtr, -1.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_exp(rfPtr, rfPtr, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_add_scalar(rfPtr, rfPtr, 1.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_div(rfPtr, onePtr, rfPtr, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(48);
asc_gather(wPtr, rfPtr, indexPtr, uint16_t(8), uint8_t(1));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(32);
asc_mul_scalar(wPtr, wPtr, 2.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(40);
asc_gather(wPtr + 48, pPtr, indexPtr + 48, uint16_t(8), uint8_t(1));
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(16);
asc_add_scalar(wPtr, wPtr, -0.5f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
asc_mul(wPtr + 16, wPtr + 16, wPtr + 16, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_add(wPtr, wPtr, wPtr + 48, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
asc_mul(wPtr + 16, wPtr + 16, wPtr + 72, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetMaskNorm();
AscendC::SetVectorMask<float>(8);
asc_mul(wPtr, wPtr, wPtr + 64, uint8_t(4), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(0));
AscendC::SetMaskCount();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(8);
asc_mul(wPtr + 32, wPtr + 32, wPtr + 40, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(10);
if constexpr (sizeof(T) == sizeof(float)) {
asc_gather((__ubuf__ float *)out.GetPhyAddr(), wPtr, indexPtr + 88, uint16_t(8), uint8_t(1));
} else {
auto packed = mem.PackedFloat();
auto packedPtr = (__ubuf__ float *)packed.GetPhyAddr();
asc_gather(packedPtr, wPtr, indexPtr + 88, uint16_t(8), uint8_t(1));
AscendC::PipeBarrier<PIPE_V>();
asc_float2half_rn((__ubuf__ half *)out.GetPhyAddr(), packedPtr, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(4), uint16_t(8));
}
AscendC::SetMaskNorm();
AscendC::ResetMask();
AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
AscendC::DataCopyExtParams op{1, count * 5 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(outGm[pos * 5], out, op);
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
} else if constexpr (CAPI_PIPE && STRIDE == 32) {
static_assert(STRIDE == 32, "Count pipeline complete32 region");
auto one = mem.Ones();
auto work = mem.Work();
auto out = mem.Out();
AscendC::LocalTensor<float> rf;
if constexpr (sizeof(T) == sizeof(float)) rf = r.template ReinterpretCast<float>();
else rf = mem.RawFloat();
auto rfPtr = (__ubuf__ float *)rf.GetPhyAddr();
auto onePtr = (__ubuf__ float *)one.GetPhyAddr();
auto wPtr = (__ubuf__ float *)work.GetPhyAddr();
auto pPtr = (__ubuf__ float *)p.GetPhyAddr();
auto indexPtr = (__ubuf__ uint32_t *)indices.GetPhyAddr();
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetMaskCount();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(192);
asc_duplicate(onePtr, 1.0f, uint8_t(1), uint16_t(1), uint16_t(8));
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
if constexpr (sizeof(T) != sizeof(float)) {
asc_half2float(rfPtr, (__ubuf__ half *)r.GetPhyAddr(), uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(4));
AscendC::PipeBarrier<PIPE_V>();
}
asc_mul_scalar(rfPtr, rfPtr, -1.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_exp(rfPtr, rfPtr, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_add_scalar(rfPtr, rfPtr, 1.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_div(rfPtr, onePtr, rfPtr, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
AscendC::PipeBarrier<PIPE_V>();
asc_gather(wPtr, rfPtr, indexPtr, uint16_t(8), uint8_t(1));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(128);
asc_mul_scalar(wPtr, wPtr, 2.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(160);
asc_gather(wPtr + 192, pPtr, indexPtr + 192, uint16_t(8), uint8_t(1));
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(64);
asc_add_scalar(wPtr, wPtr, -0.5f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
asc_mul(wPtr + 64, wPtr + 64, wPtr + 64, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_add(wPtr, wPtr, wPtr + 192, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
asc_mul(wPtr + 64, wPtr + 64, wPtr + 288, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetMaskNorm();
AscendC::SetVectorMask<float>(32);
asc_mul(wPtr, wPtr, wPtr + 256, uint8_t(4), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(4), uint8_t(4), uint8_t(0));
AscendC::SetMaskCount();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(32);
asc_mul(wPtr + 128, wPtr + 128, wPtr + 160, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(160);
if constexpr (sizeof(T) == sizeof(float)) {
asc_gather((__ubuf__ float *)out.GetPhyAddr(), wPtr, indexPtr + 352, uint16_t(8), uint8_t(1));
} else {
#if FIRE_P8O_HALF_IMAGE
if constexpr (PACKET_GM) {
auto halfPlanar = (__ubuf__ half *)rf.GetPhyAddr();
asc_float2half_rn(halfPlanar, wPtr, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(4), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_gather((__ubuf__ half *)out.GetPhyAddr(), halfPlanar, indexPtr + 352, uint16_t(8), uint8_t(1));
} else {
#endif
auto packed = mem.PackedFloat();
auto packedPtr = (__ubuf__ float *)packed.GetPhyAddr();
asc_gather(packedPtr, wPtr, indexPtr + 352, uint16_t(8), uint8_t(1));
AscendC::PipeBarrier<PIPE_V>();
asc_float2half_rn((__ubuf__ half *)out.GetPhyAddr(), packedPtr, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(4), uint16_t(8));
#if FIRE_P8O_HALF_IMAGE
}
#endif
}
AscendC::SetMaskNorm();
AscendC::ResetMask();
AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
if constexpr (CAPI_PIPE && SINGLE && FULL && STRIDE == 32) {
AscendC::DataCopy(outGm[pos * 5], out, uint32_t(160));
} else {
AscendC::DataCopyExtParams op{1, count * 5 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(outGm[pos * 5], out, op);
}
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
} else {
auto one = mem.Ones();
AscendC::Duplicate(one, 1.0f, count * 6);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
auto work = mem.Work();
AscendC::LocalTensor<float> f[11];
#pragma unroll
for (uint32_t k = 0; k < 11; ++k) f[k] = work[k * STRIDE];
AscendC::LocalTensor<float> rf;
if constexpr (sizeof(T) == sizeof(float)) rf = r.template ReinterpretCast<float>();
else {
rf = mem.RawFloat();
AscendC::Cast(rf, r, AscendC::RoundMode::CAST_NONE, count * 6);
}
AscendC::Muls(rf, rf, -1.0f, count * 6);
AscendC::Exp(rf, rf, count * 6);
AscendC::Adds(rf, rf, 1.0f, count * 6);
AscendC::Div(rf, one, rf, count * 6);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
AscendC::Gather(work, rf, indices, 0, 6 * STRIDE);
AscendC::Gather(f[6], p, indices[6 * STRIDE], 0, 5 * STRIDE);
// Compact contiguous planes retain every FP32 node, including four stride uses.
AscendC::Muls(f[0], f[0], 2.0f, 4 * STRIDE);
AscendC::Adds(f[0], f[0], -0.5f, 2 * STRIDE);
AscendC::Add(f[0], f[0], f[6], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[2], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[9], 2 * STRIDE);
const AscendC::BinaryRepeatParams strideBinary{1, 1, 1, STRIDE / 8, STRIDE / 8, 0};
AscendC::Mul<float>(f[0], f[0], f[8], uint64_t(STRIDE), uint8_t(4), strideBinary);
AscendC::Mul(f[4], f[4], f[5], STRIDE);
AscendC::ResetMask();
auto out = mem.Out();
if constexpr (sizeof(T) == sizeof(float)) {
AscendC::Gather(out.template ReinterpretCast<float>(), work, indices[11 * STRIDE], 0, count * 5);
} else {
auto packed = mem.PackedFloat();
AscendC::Gather(packed, work, indices[11 * STRIDE], 0, count * 5);
AscendC::Cast(out, packed, AscendC::RoundMode::CAST_RINT, count * 5);
}
AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
AscendC::DataCopyExtParams op{1, count * 5 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(outGm[pos * 5], out, op);
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
}
}
template <uint32_t BLOCK, uint32_t TOTAL, uint32_t N>
__aicore__ inline void RunPacketBlock(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded, GM_ADDR tiling) {
constexpr uint32_t POS = BLOCK * 32;
constexpr uint32_t COUNT = BLOCK < 2 ? 32 : TOTAL - 64;
constexpr uint32_t FIRST = POS % N;
constexpr bool WRAPS = FIRST + COUNT > N;
constexpr uint32_t PRIOR_START = WRAPS ? 0 : FIRST;
constexpr uint32_t PRIOR_COUNT = WRAPS ? N : COUNT;
constexpr uint32_t STRIDE = (COUNT + 7) / 8 * 8;
constexpr uint32_t TABLE = BLOCK == 0 ? 32 : BLOCK == 1 ? 2080 : 4128;
static_assert(COUNT > 0 && COUNT <= 32 && PRIOR_COUNT * 5 <= 288, "packet staging bounds");
RunBlock<STRIDE>(raw, priors, decoded, tiling, POS, PRIOR_START, PRIOR_COUNT, COUNT, TABLE);
}
public:
template <uint32_t TOTAL, uint32_t N, bool SPARSE_EVEN = false>
__aicore__ inline void RunPacket(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded, GM_ADDR tiling) {
static_assert(TOTAL > 64 && TOTAL <= 72 && TOTAL % N == 0 && TOTAL / N <= 8, "two full blocks and a short tail");
static_assert(!PACKET_GM || (TOTAL == 66 && N == 33), "556 exact packet payload");
const uint32_t logicalBlock = static_cast<uint32_t>(AscendC::GetBlockIdx());
const uint32_t block = SPARSE_EVEN ? logicalBlock / 2U : logicalBlock;
if (block < 2) {
constexpr uint32_t FIRST1 = 32 % N;
constexpr uint32_t START1 = FIRST1 + 32 > N ? 0 : FIRST1;
constexpr uint32_t COUNT0 = 32 > N ? N : 32;
constexpr uint32_t COUNT1 = FIRST1 + 32 > N ? N : 32;
const uint32_t priorStart = block == 0 ? 0 : START1;
const uint32_t priorCount = block == 0 ? COUNT0 : COUNT1;
RunBlock<32>(raw, priors, decoded, tiling, block * 32,
priorStart, priorCount, 32, 32 + block * 2048);
} else if (block == 2) RunPacketBlock<2, TOTAL, N>(raw, priors, decoded, tiling);
}
__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded,
GM_ADDR tiling, const FireCandidateDecodeFusionTilingData &t) {
if constexpr (FULL) {
static_assert(SINGLE, "one complete hardware block");
RunBlock<32>(raw, priors, decoded, tiling, 0, 0, t.n, 32, 32);
} else if constexpr (SINGLE) {
const uint32_t count = t.total;
if (count <= 8) RunBlock<8>(raw, priors, decoded, tiling, 0, 0, t.n, count, 32);
else if (count <= 16) RunBlock<16>(raw, priors, decoded, tiling, 0, 0, t.n, count, 32);
else if (count <= 24) RunBlock<24>(raw, priors, decoded, tiling, 0, 0, t.n, count, 32);
else RunBlock<32>(raw, priors, decoded, tiling, 0, 0, t.n, count, 32);
} else {
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
const uint32_t pos = block * 32;
if (pos >= t.total) return;
const uint32_t count = t.total - pos < 32 ? t.total - pos : 32;
const uint32_t firstPrior = pos % t.n;
const bool wraps = firstPrior + count > t.n;
const uint32_t priorStart = wraps ? 0 : firstPrior;
const uint32_t priorCount = wraps ? t.n : count;
const uint32_t tableOffset = block < 2 ? 32 + block * 2048 : 4128;
if (count <= 8) RunBlock<8>(raw, priors, decoded, tiling, pos, priorStart, priorCount, count, tableOffset);
else if (count <= 16) RunBlock<16>(raw, priors, decoded, tiling, pos, priorStart, priorCount, count, tableOffset);
else if (count <= 24) RunBlock<24>(raw, priors, decoded, tiling, pos, priorStart, priorCount, count, tableOffset);
else RunBlock<32>(raw, priors, decoded, tiling, pos, priorStart, priorCount, count, tableOffset);
}
}
};

template <typename T>
class FireHostComposite32Stride64Storage678 : public FireHostComposite32Storage<T> {
static constexpr uint32_t INDEX_BASE = sizeof(T) == sizeof(float) ? 5120 : 5824;
static_assert(INDEX_BASE + 544 * sizeof(uint32_t) == (sizeof(T) == sizeof(float) ? 7296 : 8000),
"bounded stride64 index storage");
static_assert(INDEX_BASE % 32 == 0 && INDEX_BASE + 544 * sizeof(uint32_t) < 188416,
"retain SDK temporary UB reservation");
public:
__aicore__ inline AscendC::LocalTensor<uint32_t> IndexInput() {
return AscendC::LocalTensor<uint32_t>(AscendC::TPosition::VECIN, INDEX_BASE, 544);
}
__aicore__ inline AscendC::LocalTensor<int32_t> Indices() {
return AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, INDEX_BASE, 544);
}
};

template <typename T, bool SINGLE = false, bool FULL = false, typename Storage = FireFlatCompositeStorage<T>, bool CAPI_PIPE = false, bool PACKET_GM = false, bool EARLY_RAW_C1 = false>
class KernelFireFlatCompositeInventory684 {
static_assert(!PACKET_GM || (CAPI_PIPE && !SINGLE && !FULL), "556 fixed C3 only");
static_assert(!EARLY_RAW_C1 || (CAPI_PIPE && SINGLE && FULL && sizeof(T)==sizeof(half)), "591 halfC1 only");
Storage mem;
AscendC::GlobalTensor<T> rawGm, outGm;
AscendC::GlobalTensor<float> priorGm;
AscendC::GlobalTensor<uint32_t> tableGm;
template <uint32_t STRIDE>
__aicore__ inline void RunBlock(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded,
GM_ADDR tiling, uint32_t pos, uint32_t priorStart, uint32_t priorCount, uint32_t count, uint32_t tableOffset) {
rawGm.SetGlobalBuffer((__gm__ T *)raw);
priorGm.SetGlobalBuffer((__gm__ float *)priors);
outGm.SetGlobalBuffer((__gm__ T *)decoded);
if constexpr (PACKET_GM) {
static_assert(STRIDE == 32 || STRIDE == 8, "556 full32 and tail8");
tableGm.SetGlobalBuffer(const_cast<__gm__ uint32_t *>(FirePacketIndexGM556 + (tableOffset - 32) / 4));
} else if constexpr (CAPI_PIPE && SINGLE && FULL && STRIDE == 32) {
// Read-only constant GM table.
tableGm.SetGlobalBuffer(const_cast<__gm__ uint32_t *>(FireSmallIndexGM678));
} else {
tableGm.SetGlobalBuffer((__gm__ uint32_t *)(tiling + tableOffset));
}
auto r = mem.Raw();
auto p = mem.Prior();
auto indices = mem.Indices().template ReinterpretCast<uint32_t>();
if constexpr (CAPI_PIPE && SINGLE && FULL && STRIDE == 32) {
//546: this path is one complete S32 block; typed GM addresses need only element alignment.
AscendC::DataCopy(r, rawGm[pos * 6], uint32_t(192));
if constexpr (EARLY_RAW_C1) AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
AscendC::DataCopy(p, priorGm[priorStart * 5], uint32_t(160));
} else {
AscendC::DataCopyExtParams rp{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyExtParams pp{1, priorCount * 5 * uint32_t(sizeof(float)), 0, 0, 0};
AscendC::DataCopyPad(r, rawGm[pos * 6], rp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
AscendC::DataCopyPad(p, priorGm[priorStart * 5], pp, AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
}
if constexpr (!EARLY_RAW_C1) AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
// Publish the later index DMA separately: activation only requires the first fence.
if constexpr (CAPI_PIPE && SINGLE && FULL && STRIDE == 32) {
AscendC::DataCopy(mem.IndexInput(), tableGm, uint32_t(544));
} else {
AscendC::DataCopyExtParams ip{1, 16 * STRIDE * uint32_t(sizeof(uint32_t)), 0, 0, 0};
AscendC::DataCopyPad(mem.IndexInput(), tableGm, ip, AscendC::DataCopyPadExtParams<uint32_t>{false, 0, 0, 0});
}
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
if constexpr (CAPI_PIPE && STRIDE == 32) {
static_assert(STRIDE == 32, "Count pipeline complete32 region");
auto one = mem.Ones();
auto work = mem.Work();
auto out = mem.Out();
AscendC::LocalTensor<float> rf;
if constexpr (sizeof(T) == sizeof(float)) rf = r.template ReinterpretCast<float>();
else rf = mem.RawFloat();
auto rfPtr = (__ubuf__ float *)rf.GetPhyAddr();
auto onePtr = (__ubuf__ float *)one.GetPhyAddr();
auto wPtr = (__ubuf__ float *)work.GetPhyAddr();
auto pPtr = (__ubuf__ float *)p.GetPhyAddr();
auto indexPtr = (__ubuf__ uint32_t *)indices.GetPhyAddr();
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetMaskCount();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(192);
asc_duplicate(onePtr, 1.0f, uint8_t(1), uint16_t(1), uint16_t(8));
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
if constexpr (sizeof(T) != sizeof(float)) {
asc_half2float(rfPtr, (__ubuf__ half *)r.GetPhyAddr(), uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(4));
AscendC::PipeBarrier<PIPE_V>();
}
asc_mul_scalar(rfPtr, rfPtr, -1.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_exp(rfPtr, rfPtr, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_add_scalar(rfPtr, rfPtr, 1.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_div(rfPtr, onePtr, rfPtr, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
AscendC::PipeBarrier<PIPE_V>();
asc_gather(wPtr, rfPtr, indexPtr, uint16_t(8), uint8_t(1));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(128);
asc_mul_scalar(wPtr, wPtr, 2.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>((SINGLE && FULL) ? 192 : 160);
asc_gather(wPtr + 192, pPtr, indexPtr + 192, uint16_t(8), uint8_t(1));
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(64);
asc_add_scalar(wPtr, wPtr, -0.5f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
asc_mul(wPtr + 64, wPtr + 64, wPtr + 64, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_add(wPtr, wPtr, wPtr + 192, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
asc_mul(wPtr + 64, wPtr + 64, wPtr + ((SINGLE && FULL) ? 320 : 288), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
if constexpr (SINGLE && FULL) {
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(128);
asc_mul(wPtr, wPtr, wPtr + 256, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(0));
} else {
AscendC::SetMaskNorm();
AscendC::SetVectorMask<float>(32);
asc_mul(wPtr, wPtr, wPtr + 256, uint8_t(4), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(4), uint8_t(4), uint8_t(0));
AscendC::SetMaskCount();
}
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(32);
asc_mul(wPtr + 128, wPtr + 128, wPtr + 160, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(160);
if constexpr (sizeof(T) == sizeof(float)) {
asc_gather((__ubuf__ float *)out.GetPhyAddr(), wPtr, indexPtr + ((SINGLE && FULL) ? 384 : 352), uint16_t(8), uint8_t(1));
} else {
auto packed = mem.PackedFloat();
auto packedPtr = (__ubuf__ float *)packed.GetPhyAddr();
asc_gather(packedPtr, wPtr, indexPtr + ((SINGLE && FULL) ? 384 : 352), uint16_t(8), uint8_t(1));
AscendC::PipeBarrier<PIPE_V>();
asc_float2half_rn((__ubuf__ half *)out.GetPhyAddr(), packedPtr, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(4), uint16_t(8));
}
AscendC::SetMaskNorm();
AscendC::ResetMask();
AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
if constexpr (CAPI_PIPE && SINGLE && FULL && STRIDE == 32) {
AscendC::DataCopy(outGm[pos * 5], out, uint32_t(160));
} else {
AscendC::DataCopyExtParams op{1, count * 5 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(outGm[pos * 5], out, op);
}
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
} else {
auto one = mem.Ones();
AscendC::Duplicate(one, 1.0f, count * 6);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
auto work = mem.Work();
AscendC::LocalTensor<float> f[11];
#pragma unroll
for (uint32_t k = 0; k < 11; ++k) f[k] = work[k * STRIDE];
AscendC::LocalTensor<float> rf;
if constexpr (sizeof(T) == sizeof(float)) rf = r.template ReinterpretCast<float>();
else {
rf = mem.RawFloat();
AscendC::Cast(rf, r, AscendC::RoundMode::CAST_NONE, count * 6);
}
AscendC::Muls(rf, rf, -1.0f, count * 6);
AscendC::Exp(rf, rf, count * 6);
AscendC::Adds(rf, rf, 1.0f, count * 6);
AscendC::Div(rf, one, rf, count * 6);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
AscendC::Gather(work, rf, indices, 0, 6 * STRIDE);
AscendC::Gather(f[6], p, indices[6 * STRIDE], 0, 5 * STRIDE);
// Compact contiguous planes retain every FP32 node, including four stride uses.
AscendC::Muls(f[0], f[0], 2.0f, 4 * STRIDE);
AscendC::Adds(f[0], f[0], -0.5f, 2 * STRIDE);
AscendC::Add(f[0], f[0], f[6], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[2], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[9], 2 * STRIDE);
const AscendC::BinaryRepeatParams strideBinary{1, 1, 1, STRIDE / 8, STRIDE / 8, 0};
AscendC::Mul<float>(f[0], f[0], f[8], uint64_t(STRIDE), uint8_t(4), strideBinary);
AscendC::Mul(f[4], f[4], f[5], STRIDE);
AscendC::ResetMask();
auto out = mem.Out();
if constexpr (sizeof(T) == sizeof(float)) {
AscendC::Gather(out.template ReinterpretCast<float>(), work, indices[11 * STRIDE], 0, count * 5);
} else {
auto packed = mem.PackedFloat();
AscendC::Gather(packed, work, indices[11 * STRIDE], 0, count * 5);
AscendC::Cast(out, packed, AscendC::RoundMode::CAST_RINT, count * 5);
}
AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
AscendC::DataCopyExtParams op{1, count * 5 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(outGm[pos * 5], out, op);
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
}
}
template <uint32_t BLOCK, uint32_t TOTAL, uint32_t N>
__aicore__ inline void RunPacketBlock(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded, GM_ADDR tiling) {
constexpr uint32_t POS = BLOCK * 32;
constexpr uint32_t COUNT = BLOCK < 2 ? 32 : TOTAL - 64;
constexpr uint32_t FIRST = POS % N;
constexpr bool WRAPS = FIRST + COUNT > N;
constexpr uint32_t PRIOR_START = WRAPS ? 0 : FIRST;
constexpr uint32_t PRIOR_COUNT = WRAPS ? N : COUNT;
constexpr uint32_t STRIDE = (COUNT + 7) / 8 * 8;
constexpr uint32_t TABLE = BLOCK == 0 ? 32 : BLOCK == 1 ? 2080 : 4128;
static_assert(COUNT > 0 && COUNT <= 32 && PRIOR_COUNT * 5 <= 288, "packet staging bounds");
RunBlock<STRIDE>(raw, priors, decoded, tiling, POS, PRIOR_START, PRIOR_COUNT, COUNT, TABLE);
}
public:
template <uint32_t TOTAL, uint32_t N>
__aicore__ inline void RunPacket(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded, GM_ADDR tiling) {
static_assert(TOTAL > 64 && TOTAL <= 72 && TOTAL % N == 0 && TOTAL / N <= 8, "two full blocks and a short tail");
static_assert(!PACKET_GM || (TOTAL == 66 && N == 33), "556 exact packet payload");
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
if (block < 2) {
constexpr uint32_t FIRST1 = 32 % N;
constexpr uint32_t START1 = FIRST1 + 32 > N ? 0 : FIRST1;
constexpr uint32_t COUNT0 = 32 > N ? N : 32;
constexpr uint32_t COUNT1 = FIRST1 + 32 > N ? N : 32;
const uint32_t priorStart = block == 0 ? 0 : START1;
const uint32_t priorCount = block == 0 ? COUNT0 : COUNT1;
RunBlock<32>(raw, priors, decoded, tiling, block * 32,
priorStart, priorCount, 32, 32 + block * 2048);
} else if (block == 2) RunPacketBlock<2, TOTAL, N>(raw, priors, decoded, tiling);
}
__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded,
GM_ADDR tiling, const FireCandidateDecodeFusionTilingData &t) {
if constexpr (FULL) {
static_assert(SINGLE, "one complete hardware block");
RunBlock<32>(raw, priors, decoded, tiling, 0, 0, t.n, 32, 32);
} else if constexpr (SINGLE) {
const uint32_t count = t.total;
if (count <= 8) RunBlock<8>(raw, priors, decoded, tiling, 0, 0, t.n, count, 32);
else if (count <= 16) RunBlock<16>(raw, priors, decoded, tiling, 0, 0, t.n, count, 32);
else if (count <= 24) RunBlock<24>(raw, priors, decoded, tiling, 0, 0, t.n, count, 32);
else RunBlock<32>(raw, priors, decoded, tiling, 0, 0, t.n, count, 32);
} else {
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
const uint32_t pos = block * 32;
if (pos >= t.total) return;
const uint32_t count = t.total - pos < 32 ? t.total - pos : 32;
const uint32_t firstPrior = pos % t.n;
const bool wraps = firstPrior + count > t.n;
const uint32_t priorStart = wraps ? 0 : firstPrior;
const uint32_t priorCount = wraps ? t.n : count;
const uint32_t tableOffset = block < 2 ? 32 + block * 2048 : 4128;
if (count <= 8) RunBlock<8>(raw, priors, decoded, tiling, pos, priorStart, priorCount, count, tableOffset);
else if (count <= 16) RunBlock<16>(raw, priors, decoded, tiling, pos, priorStart, priorCount, count, tableOffset);
else if (count <= 24) RunBlock<24>(raw, priors, decoded, tiling, pos, priorStart, priorCount, count, tableOffset);
else RunBlock<32>(raw, priors, decoded, tiling, pos, priorStart, priorCount, count, tableOffset);
}
}
};

template <typename T, bool SINGLE = false, bool FULL = false, typename Storage = FireFlatCompositeStorage<T>, bool CAPI_PIPE = false, bool PACKET_GM = false, bool EARLY_RAW_C1 = false>
class KernelFireFlatCompositeEarlyOne684 {
static_assert(!PACKET_GM || (CAPI_PIPE && !SINGLE && !FULL), "556 fixed C3 only");
static_assert(!EARLY_RAW_C1 || (CAPI_PIPE && SINGLE && FULL && sizeof(T)==sizeof(half)), "591 halfC1 only");
Storage mem;
AscendC::GlobalTensor<T> rawGm, outGm;
AscendC::GlobalTensor<float> priorGm;
AscendC::GlobalTensor<uint32_t> tableGm;
template <uint32_t STRIDE>
__aicore__ inline void RunBlock(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded,
GM_ADDR tiling, uint32_t pos, uint32_t priorStart, uint32_t priorCount, uint32_t count, uint32_t tableOffset) {
if constexpr (CAPI_PIPE && SINGLE && FULL && STRIDE == 32) {
auto earlyOne = mem.Ones();
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetMaskCount();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(192);
asc_duplicate((__ubuf__ float *)earlyOne.GetPhyAddr(), 1.0f, uint8_t(1), uint16_t(1), uint16_t(8));
}
rawGm.SetGlobalBuffer((__gm__ T *)raw);
priorGm.SetGlobalBuffer((__gm__ float *)priors);
outGm.SetGlobalBuffer((__gm__ T *)decoded);
if constexpr (PACKET_GM) {
static_assert(STRIDE == 32 || STRIDE == 8, "556 full32 and tail8");
tableGm.SetGlobalBuffer(const_cast<__gm__ uint32_t *>(FirePacketIndexGM556 + (tableOffset - 32) / 4));
} else if constexpr (CAPI_PIPE && SINGLE && FULL && STRIDE == 32) {
// Read-only constant GM table.
tableGm.SetGlobalBuffer(const_cast<__gm__ uint32_t *>(FireSmallIndexGM548));
} else {
tableGm.SetGlobalBuffer((__gm__ uint32_t *)(tiling + tableOffset));
}
auto r = mem.Raw();
auto p = mem.Prior();
auto indices = mem.Indices().template ReinterpretCast<uint32_t>();
if constexpr (CAPI_PIPE && SINGLE && FULL && STRIDE == 32) {
//546: this path is one complete S32 block; typed GM addresses need only element alignment.
AscendC::DataCopy(r, rawGm[pos * 6], uint32_t(192));
if constexpr (EARLY_RAW_C1) AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
AscendC::DataCopy(p, priorGm[priorStart * 5], uint32_t(160));
} else {
AscendC::DataCopyExtParams rp{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyExtParams pp{1, priorCount * 5 * uint32_t(sizeof(float)), 0, 0, 0};
AscendC::DataCopyPad(r, rawGm[pos * 6], rp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
AscendC::DataCopyPad(p, priorGm[priorStart * 5], pp, AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
}
if constexpr (!EARLY_RAW_C1) AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
// Publish the later index DMA separately: activation only requires the first fence.
if constexpr (CAPI_PIPE && SINGLE && FULL && STRIDE == 32) {
AscendC::DataCopy(mem.IndexInput(), tableGm, uint32_t(512));
} else {
AscendC::DataCopyExtParams ip{1, 16 * STRIDE * uint32_t(sizeof(uint32_t)), 0, 0, 0};
AscendC::DataCopyPad(mem.IndexInput(), tableGm, ip, AscendC::DataCopyPadExtParams<uint32_t>{false, 0, 0, 0});
}
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
if constexpr (CAPI_PIPE && STRIDE == 32) {
static_assert(STRIDE == 32, "Count pipeline complete32 region");
auto one = mem.Ones();
auto work = mem.Work();
auto out = mem.Out();
AscendC::LocalTensor<float> rf;
if constexpr (sizeof(T) == sizeof(float)) rf = r.template ReinterpretCast<float>();
else rf = mem.RawFloat();
auto rfPtr = (__ubuf__ float *)rf.GetPhyAddr();
auto onePtr = (__ubuf__ float *)one.GetPhyAddr();
auto wPtr = (__ubuf__ float *)work.GetPhyAddr();
auto pPtr = (__ubuf__ float *)p.GetPhyAddr();
auto indexPtr = (__ubuf__ uint32_t *)indices.GetPhyAddr();
if constexpr (!(CAPI_PIPE && SINGLE && FULL)) {
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetMaskCount();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(192);
asc_duplicate(onePtr, 1.0f, uint8_t(1), uint16_t(1), uint16_t(8));
}
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
if constexpr (sizeof(T) != sizeof(float)) {
asc_half2float(rfPtr, (__ubuf__ half *)r.GetPhyAddr(), uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(4));
AscendC::PipeBarrier<PIPE_V>();
}
asc_mul_scalar(rfPtr, rfPtr, -1.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_exp(rfPtr, rfPtr, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_add_scalar(rfPtr, rfPtr, 1.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_div(rfPtr, onePtr, rfPtr, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
AscendC::PipeBarrier<PIPE_V>();
asc_gather(wPtr, rfPtr, indexPtr, uint16_t(8), uint8_t(1));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(128);
asc_mul_scalar(wPtr, wPtr, 2.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(160);
asc_gather(wPtr + 192, pPtr, indexPtr + 192, uint16_t(8), uint8_t(1));
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(64);
asc_add_scalar(wPtr, wPtr, -0.5f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
asc_mul(wPtr + 64, wPtr + 64, wPtr + 64, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_add(wPtr, wPtr, wPtr + 192, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
asc_mul(wPtr + 64, wPtr + 64, wPtr + 288, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetMaskNorm();
AscendC::SetVectorMask<float>(32);
asc_mul(wPtr, wPtr, wPtr + 256, uint8_t(4), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(4), uint8_t(4), uint8_t(0));
AscendC::SetMaskCount();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(32);
asc_mul(wPtr + 128, wPtr + 128, wPtr + 160, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(160);
if constexpr (sizeof(T) == sizeof(float)) {
asc_gather((__ubuf__ float *)out.GetPhyAddr(), wPtr, indexPtr + 352, uint16_t(8), uint8_t(1));
} else {
auto packed = mem.PackedFloat();
auto packedPtr = (__ubuf__ float *)packed.GetPhyAddr();
asc_gather(packedPtr, wPtr, indexPtr + 352, uint16_t(8), uint8_t(1));
AscendC::PipeBarrier<PIPE_V>();
asc_float2half_rn((__ubuf__ half *)out.GetPhyAddr(), packedPtr, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(4), uint16_t(8));
}
AscendC::SetMaskNorm();
AscendC::ResetMask();
AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
if constexpr (CAPI_PIPE && SINGLE && FULL && STRIDE == 32) {
AscendC::DataCopy(outGm[pos * 5], out, uint32_t(160));
} else {
AscendC::DataCopyExtParams op{1, count * 5 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(outGm[pos * 5], out, op);
}
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
} else {
auto one = mem.Ones();
AscendC::Duplicate(one, 1.0f, count * 6);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
auto work = mem.Work();
AscendC::LocalTensor<float> f[11];
#pragma unroll
for (uint32_t k = 0; k < 11; ++k) f[k] = work[k * STRIDE];
AscendC::LocalTensor<float> rf;
if constexpr (sizeof(T) == sizeof(float)) rf = r.template ReinterpretCast<float>();
else {
rf = mem.RawFloat();
AscendC::Cast(rf, r, AscendC::RoundMode::CAST_NONE, count * 6);
}
AscendC::Muls(rf, rf, -1.0f, count * 6);
AscendC::Exp(rf, rf, count * 6);
AscendC::Adds(rf, rf, 1.0f, count * 6);
AscendC::Div(rf, one, rf, count * 6);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
AscendC::Gather(work, rf, indices, 0, 6 * STRIDE);
AscendC::Gather(f[6], p, indices[6 * STRIDE], 0, 5 * STRIDE);
// Compact contiguous planes retain every FP32 node, including four stride uses.
AscendC::Muls(f[0], f[0], 2.0f, 4 * STRIDE);
AscendC::Adds(f[0], f[0], -0.5f, 2 * STRIDE);
AscendC::Add(f[0], f[0], f[6], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[2], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[9], 2 * STRIDE);
const AscendC::BinaryRepeatParams strideBinary{1, 1, 1, STRIDE / 8, STRIDE / 8, 0};
AscendC::Mul<float>(f[0], f[0], f[8], uint64_t(STRIDE), uint8_t(4), strideBinary);
AscendC::Mul(f[4], f[4], f[5], STRIDE);
AscendC::ResetMask();
auto out = mem.Out();
if constexpr (sizeof(T) == sizeof(float)) {
AscendC::Gather(out.template ReinterpretCast<float>(), work, indices[11 * STRIDE], 0, count * 5);
} else {
auto packed = mem.PackedFloat();
AscendC::Gather(packed, work, indices[11 * STRIDE], 0, count * 5);
AscendC::Cast(out, packed, AscendC::RoundMode::CAST_RINT, count * 5);
}
AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
AscendC::DataCopyExtParams op{1, count * 5 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(outGm[pos * 5], out, op);
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
}
}
template <uint32_t BLOCK, uint32_t TOTAL, uint32_t N>
__aicore__ inline void RunPacketBlock(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded, GM_ADDR tiling) {
constexpr uint32_t POS = BLOCK * 32;
constexpr uint32_t COUNT = BLOCK < 2 ? 32 : TOTAL - 64;
constexpr uint32_t FIRST = POS % N;
constexpr bool WRAPS = FIRST + COUNT > N;
constexpr uint32_t PRIOR_START = WRAPS ? 0 : FIRST;
constexpr uint32_t PRIOR_COUNT = WRAPS ? N : COUNT;
constexpr uint32_t STRIDE = (COUNT + 7) / 8 * 8;
constexpr uint32_t TABLE = BLOCK == 0 ? 32 : BLOCK == 1 ? 2080 : 4128;
static_assert(COUNT > 0 && COUNT <= 32 && PRIOR_COUNT * 5 <= 288, "packet staging bounds");
RunBlock<STRIDE>(raw, priors, decoded, tiling, POS, PRIOR_START, PRIOR_COUNT, COUNT, TABLE);
}
public:
template <uint32_t TOTAL, uint32_t N>
__aicore__ inline void RunPacket(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded, GM_ADDR tiling) {
static_assert(TOTAL > 64 && TOTAL <= 72 && TOTAL % N == 0 && TOTAL / N <= 8, "two full blocks and a short tail");
static_assert(!PACKET_GM || (TOTAL == 66 && N == 33), "556 exact packet payload");
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
if (block < 2) {
constexpr uint32_t FIRST1 = 32 % N;
constexpr uint32_t START1 = FIRST1 + 32 > N ? 0 : FIRST1;
constexpr uint32_t COUNT0 = 32 > N ? N : 32;
constexpr uint32_t COUNT1 = FIRST1 + 32 > N ? N : 32;
const uint32_t priorStart = block == 0 ? 0 : START1;
const uint32_t priorCount = block == 0 ? COUNT0 : COUNT1;
RunBlock<32>(raw, priors, decoded, tiling, block * 32,
priorStart, priorCount, 32, 32 + block * 2048);
} else if (block == 2) RunPacketBlock<2, TOTAL, N>(raw, priors, decoded, tiling);
}
__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded,
GM_ADDR tiling, const FireCandidateDecodeFusionTilingData &t) {
if constexpr (FULL) {
static_assert(SINGLE, "one complete hardware block");
RunBlock<32>(raw, priors, decoded, tiling, 0, 0, t.n, 32, 32);
} else if constexpr (SINGLE) {
const uint32_t count = t.total;
if (count <= 8) RunBlock<8>(raw, priors, decoded, tiling, 0, 0, t.n, count, 32);
else if (count <= 16) RunBlock<16>(raw, priors, decoded, tiling, 0, 0, t.n, count, 32);
else if (count <= 24) RunBlock<24>(raw, priors, decoded, tiling, 0, 0, t.n, count, 32);
else RunBlock<32>(raw, priors, decoded, tiling, 0, 0, t.n, count, 32);
} else {
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
const uint32_t pos = block * 32;
if (pos >= t.total) return;
const uint32_t count = t.total - pos < 32 ? t.total - pos : 32;
const uint32_t firstPrior = pos % t.n;
const bool wraps = firstPrior + count > t.n;
const uint32_t priorStart = wraps ? 0 : firstPrior;
const uint32_t priorCount = wraps ? t.n : count;
const uint32_t tableOffset = block < 2 ? 32 + block * 2048 : 4128;
if (count <= 8) RunBlock<8>(raw, priors, decoded, tiling, pos, priorStart, priorCount, count, tableOffset);
else if (count <= 16) RunBlock<16>(raw, priors, decoded, tiling, pos, priorStart, priorCount, count, tableOffset);
else if (count <= 24) RunBlock<24>(raw, priors, decoded, tiling, pos, priorStart, priorCount, count, tableOffset);
else RunBlock<32>(raw, priors, decoded, tiling, pos, priorStart, priorCount, count, tableOffset);
}
}
};

alignas(256) static __gm__ const uint32_t FireFourIndex[128] = {
0U, 24U, 48U, 72U, 96U, 120U, 144U, 168U, 4U, 28U, 52U, 76U, 100U, 124U, 148U, 172U, 8U, 32U, 56U, 80U, 104U, 128U, 152U, 176U, 12U, 36U, 60U, 84U, 108U, 132U, 156U, 180U, 16U, 40U, 64U, 88U, 112U, 136U, 160U, 184U, 20U, 44U, 68U, 92U, 116U, 140U, 164U, 188U, 0U, 20U, 40U, 60U, 80U, 100U, 120U, 140U, 4U, 24U, 44U, 64U, 84U, 104U, 124U, 144U, 8U, 28U, 48U, 68U, 88U, 108U, 128U, 148U, 12U, 32U, 52U, 72U, 92U, 112U, 132U, 152U, 16U, 36U, 56U, 76U, 96U, 116U, 136U, 156U, 0U, 32U, 64U, 96U, 128U, 4U, 36U, 68U, 100U, 132U, 8U, 40U, 72U, 104U, 136U, 12U, 44U, 76U, 108U, 140U, 16U, 48U, 80U, 112U, 144U, 20U, 52U, 84U, 116U, 148U, 24U, 56U, 88U, 120U, 152U, 28U, 60U, 92U, 124U, 156U
};

template <typename T>
class KernelFireFourByEight {
FireHostComposite32Storage<T> mem;
public:
__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded) {
const uint32_t pos = (static_cast<uint32_t>(AscendC::GetBlockIdx()) / 2U) * 8;
AscendC::GlobalTensor<T> rawGm, outGm;
AscendC::GlobalTensor<float> priorGm;
AscendC::GlobalTensor<uint32_t> tableGm;
rawGm.SetGlobalBuffer((__gm__ T *)raw);
priorGm.SetGlobalBuffer((__gm__ float *)priors);
outGm.SetGlobalBuffer((__gm__ T *)decoded);
tableGm.SetGlobalBuffer(const_cast<__gm__ uint32_t *>(FireFourIndex));
auto r = mem.Raw(); auto p = mem.Prior(); auto one = mem.Ones();
auto rf = sizeof(T) == sizeof(float) ? r.template ReinterpretCast<float>() : mem.RawFloat();
auto w = mem.Work(); auto dst = mem.Out();
auto rfPtr = (__ubuf__ float *)rf.GetPhyAddr();
auto onePtr = (__ubuf__ float *)one.GetPhyAddr();
auto wPtr = (__ubuf__ float *)w.GetPhyAddr();
auto indexPtr = (__ubuf__ uint32_t *)mem.Indices().GetPhyAddr();
AscendC::SetMaskCount();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(48);
asc_duplicate(onePtr, 1.0f, uint8_t(1), uint16_t(1), uint16_t(8));
AscendC::DataCopyExtParams rp{1, 48 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyExtParams pp{1, 40 * uint32_t(sizeof(float)), 0, 0, 0};
AscendC::DataCopyPad(r, rawGm[pos * 6], rp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
AscendC::DataCopyPad(p, priorGm[pos * 5], pp, AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
AscendC::DataCopy(mem.IndexInput(), tableGm, uint32_t(128));
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
if constexpr (sizeof(T) != sizeof(float)) {
asc_half2float(rfPtr, (__ubuf__ half *)r.GetPhyAddr(), uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(4));
AscendC::PipeBarrier<PIPE_V>();
}
asc_mul_scalar(rfPtr, rfPtr, -1.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_exp(rfPtr, rfPtr, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_add_scalar(rfPtr, rfPtr, 1.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_div(rfPtr, onePtr, rfPtr, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
AscendC::PipeBarrier<PIPE_V>();
asc_gather(wPtr, rfPtr, indexPtr, uint16_t(8), uint8_t(1));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(32);
asc_mul_scalar(wPtr, wPtr, 2.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(40);
asc_gather(wPtr + 48, (__ubuf__ float *)p.GetPhyAddr(), indexPtr + 48, uint16_t(8), uint8_t(1));
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(16);
asc_add_scalar(wPtr, wPtr, -0.5f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
asc_mul(wPtr + 16, wPtr + 16, wPtr + 16, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_add(wPtr, wPtr, wPtr + 48, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
asc_mul(wPtr + 16, wPtr + 16, wPtr + 72, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetMaskNorm(); AscendC::SetVectorMask<float>(8);
asc_mul(wPtr, wPtr, wPtr + 64, uint8_t(4), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(0));
AscendC::SetMaskCount(); AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(8);
asc_mul(wPtr + 32, wPtr + 32, wPtr + 40, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(40);
if constexpr (sizeof(T) == sizeof(float)) {
asc_gather((__ubuf__ float *)dst.GetPhyAddr(), wPtr, indexPtr + 88, uint16_t(8), uint8_t(1));
} else {
auto packed = mem.PackedFloat();
asc_gather((__ubuf__ float *)packed.GetPhyAddr(), wPtr, indexPtr + 88, uint16_t(8), uint8_t(1));
AscendC::PipeBarrier<PIPE_V>();
asc_float2half_rn((__ubuf__ half *)dst.GetPhyAddr(), (__ubuf__ float *)packed.GetPhyAddr(), uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(4), uint16_t(8));
}
AscendC::SetMaskNorm(); AscendC::ResetMask();
AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
AscendC::DataCopyExtParams op{1, 40 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(outGm[pos * 5], dst, op);
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0); AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
}
};

alignas(256) static __gm__ const uint32_t FireEven8Half32Index[608] = {
4416U, 4440U, 4464U, 4488U, 4512U, 4536U, 4560U, 4584U, 4608U, 4632U, 4656U, 4680U, 4704U, 4728U, 4752U, 4776U,
4800U, 4824U, 4848U, 4872U, 4896U, 4920U, 4944U, 4968U, 4992U, 5016U, 5040U, 5064U, 5088U, 5112U, 5136U, 5160U,
4420U, 4444U, 4468U, 4492U, 4516U, 4540U, 4564U, 4588U, 4612U, 4636U, 4660U, 4684U, 4708U, 4732U, 4756U, 4780U,
4804U, 4828U, 4852U, 4876U, 4900U, 4924U, 4948U, 4972U, 4996U, 5020U, 5044U, 5068U, 5092U, 5116U, 5140U, 5164U,
4424U, 4448U, 4472U, 4496U, 4520U, 4544U, 4568U, 4592U, 4616U, 4640U, 4664U, 4688U, 4712U, 4736U, 4760U, 4784U,
4808U, 4832U, 4856U, 4880U, 4904U, 4928U, 4952U, 4976U, 5000U, 5024U, 5048U, 5072U, 5096U, 5120U, 5144U, 5168U,
4428U, 4452U, 4476U, 4500U, 4524U, 4548U, 4572U, 4596U, 4620U, 4644U, 4668U, 4692U, 4716U, 4740U, 4764U, 4788U,
4812U, 4836U, 4860U, 4884U, 4908U, 4932U, 4956U, 4980U, 5004U, 5028U, 5052U, 5076U, 5100U, 5124U, 5148U, 5172U,
4432U, 4456U, 4480U, 4504U, 4528U, 4552U, 4576U, 4600U, 4624U, 4648U, 4672U, 4696U, 4720U, 4744U, 4768U, 4792U,
4816U, 4840U, 4864U, 4888U, 4912U, 4936U, 4960U, 4984U, 5008U, 5032U, 5056U, 5080U, 5104U, 5128U, 5152U, 5176U,
384U, 404U, 424U, 444U, 464U, 484U, 504U, 524U, 544U, 564U, 584U, 604U, 624U, 644U, 664U, 684U,
704U, 724U, 744U, 764U, 784U, 804U, 824U, 844U, 864U, 884U, 904U, 924U, 944U, 964U, 984U, 1004U,
388U, 408U, 428U, 448U, 468U, 488U, 508U, 528U, 548U, 568U, 588U, 608U, 628U, 648U, 668U, 688U,
708U, 728U, 748U, 768U, 788U, 808U, 828U, 848U, 868U, 888U, 908U, 928U, 948U, 968U, 988U, 1008U,
396U, 416U, 436U, 456U, 476U, 496U, 516U, 536U, 556U, 576U, 596U, 616U, 636U, 656U, 676U, 696U,
716U, 736U, 756U, 776U, 796U, 816U, 836U, 856U, 876U, 896U, 916U, 936U, 956U, 976U, 996U, 1016U,
400U, 420U, 440U, 460U, 480U, 500U, 520U, 540U, 560U, 580U, 600U, 620U, 640U, 660U, 680U, 700U,
720U, 740U, 760U, 780U, 800U, 820U, 840U, 860U, 880U, 900U, 920U, 940U, 960U, 980U, 1000U, 1020U,
392U, 412U, 432U, 452U, 472U, 492U, 512U, 532U, 552U, 572U, 592U, 612U, 632U, 652U, 672U, 692U,
712U, 732U, 752U, 772U, 792U, 812U, 832U, 852U, 872U, 892U, 912U, 932U, 952U, 972U, 992U, 1012U,
392U, 412U, 432U, 452U, 472U, 492U, 512U, 532U, 552U, 572U, 592U, 612U, 632U, 652U, 672U, 692U,
712U, 732U, 752U, 772U, 792U, 812U, 832U, 852U, 872U, 892U, 912U, 932U, 952U, 972U, 992U, 1012U,
392U, 412U, 432U, 452U, 472U, 492U, 512U, 532U, 552U, 572U, 592U, 612U, 632U, 652U, 672U, 692U,
712U, 732U, 752U, 772U, 792U, 812U, 832U, 852U, 872U, 892U, 912U, 932U, 952U, 972U, 992U, 1012U,
392U, 412U, 432U, 452U, 472U, 492U, 512U, 532U, 552U, 572U, 592U, 612U, 632U, 652U, 672U, 692U,
712U, 732U, 752U, 772U, 792U, 812U, 832U, 852U, 872U, 892U, 912U, 932U, 952U, 972U, 992U, 1012U,
4436U, 4460U, 4484U, 4508U, 4532U, 4556U, 4580U, 4604U, 4628U, 4652U, 4676U, 4700U, 4724U, 4748U, 4772U, 4796U,
4820U, 4844U, 4868U, 4892U, 4916U, 4940U, 4964U, 4988U, 5012U, 5036U, 5060U, 5084U, 5108U, 5132U, 5156U, 5180U,
0U, 64U, 128U, 192U, 256U, 2U, 66U, 130U, 194U, 258U, 4U, 68U, 132U, 196U, 260U, 6U,
70U, 134U, 198U, 262U, 8U, 72U, 136U, 200U, 264U, 10U, 74U, 138U, 202U, 266U, 12U, 76U,
140U, 204U, 268U, 14U, 78U, 142U, 206U, 270U, 16U, 80U, 144U, 208U, 272U, 18U, 82U, 146U,
210U, 274U, 20U, 84U, 148U, 212U, 276U, 22U, 86U, 150U, 214U, 278U, 24U, 88U, 152U, 216U,
280U, 26U, 90U, 154U, 218U, 282U, 28U, 92U, 156U, 220U, 284U, 30U, 94U, 158U, 222U, 286U,
32U, 96U, 160U, 224U, 288U, 34U, 98U, 162U, 226U, 290U, 36U, 100U, 164U, 228U, 292U, 38U,
102U, 166U, 230U, 294U, 40U, 104U, 168U, 232U, 296U, 42U, 106U, 170U, 234U, 298U, 44U, 108U,
172U, 236U, 300U, 46U, 110U, 174U, 238U, 302U, 48U, 112U, 176U, 240U, 304U, 50U, 114U, 178U,
242U, 306U, 52U, 116U, 180U, 244U, 308U, 54U, 118U, 182U, 246U, 310U, 56U, 120U, 184U, 248U,
312U, 58U, 122U, 186U, 250U, 314U, 60U, 124U, 188U, 252U, 316U, 62U, 126U, 190U, 254U, 318U,
};

template <typename T>
class FireEven8Half32Storage : public FireHostComposite32Storage<T> {
static_assert(sizeof(T) == sizeof(half), "half full32 only");
static constexpr uint32_t INDEX_BASE = 192 * sizeof(T) + 160 * sizeof(float) + 160 * sizeof(T) +
2304 + 192 * sizeof(float) + 192 * sizeof(float) + 160 * sizeof(float);
static_assert(INDEX_BASE == 5824 && INDEX_BASE % 32 == 0, "same static base");
static_assert(INDEX_BASE + 608 * sizeof(uint32_t) == 8256, "bounded608 index view");
static_assert(INDEX_BASE + 608 * sizeof(uint32_t) < 188416, "preserve SDK temporary UB");
public:
__aicore__ inline AscendC::LocalTensor<uint32_t> IndexInput() {
return AscendC::LocalTensor<uint32_t>(AscendC::TPosition::VECIN, INDEX_BASE, 608);
}
__aicore__ inline AscendC::LocalTensor<int32_t> Indices() {
return AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, INDEX_BASE, 608);
}
};
template <typename T>
class KernelFireEven8Half32 {
static_assert(sizeof(T) == sizeof(half), "half geometry only");
FireEven8Half32Storage<T> mem;
public:
__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded) {
const uint32_t pos = (static_cast<uint32_t>(AscendC::GetBlockIdx()) / 8U) * 32U;
AscendC::GlobalTensor<T> rawGm, outGm;
AscendC::GlobalTensor<float> priorGm;
AscendC::GlobalTensor<uint32_t> tableGm;
rawGm.SetGlobalBuffer((__gm__ T *)raw);
priorGm.SetGlobalBuffer((__gm__ float *)priors);
outGm.SetGlobalBuffer((__gm__ T *)decoded);
tableGm.SetGlobalBuffer(const_cast<__gm__ uint32_t *>(FireEven8Half32Index));
auto r = mem.Raw(); auto p = mem.Prior(); auto one = mem.Ones();
auto rf = sizeof(T) == sizeof(float) ? r.template ReinterpretCast<float>() : mem.RawFloat();
auto w = mem.Work(); auto dst = mem.Out();
auto rfPtr = (__ubuf__ float *)rf.GetPhyAddr();
auto onePtr = (__ubuf__ float *)one.GetPhyAddr();
auto wPtr = (__ubuf__ float *)w.GetPhyAddr();
auto indexPtr = (__ubuf__ uint32_t *)mem.Indices().GetPhyAddr();
AscendC::SetMaskCount();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(192);
asc_duplicate(onePtr, 1.0f, uint8_t(1), uint16_t(1), uint16_t(8));
AscendC::DataCopyExtParams rp{1, 192 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyExtParams pp{1, 160 * uint32_t(sizeof(float)), 0, 0, 0};
AscendC::DataCopyPad(r, rawGm[pos * 6], rp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
AscendC::DataCopyPad(p, priorGm[pos * 5], pp, AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
AscendC::DataCopy(mem.IndexInput(), tableGm, uint32_t(608));
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
if constexpr (sizeof(T) != sizeof(float)) {
asc_half2float(rfPtr, (__ubuf__ half *)r.GetPhyAddr(), uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(4));
AscendC::PipeBarrier<PIPE_V>();
}
asc_mul_scalar(rfPtr, rfPtr, -1.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_exp(rfPtr, rfPtr, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_add_scalar(rfPtr, rfPtr, 1.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_div(rfPtr, onePtr, rfPtr, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(448);
asc_gather(wPtr, (__ubuf__ float *)r.GetPhyAddr(), indexPtr, uint16_t(8), uint8_t(1));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(128);
asc_mul_scalar(wPtr, wPtr, 2.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(64);
asc_add_scalar(wPtr, wPtr, -0.5f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
asc_mul(wPtr + 64, wPtr + 64, wPtr + 64, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_add(wPtr, wPtr, wPtr + 160, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
asc_mul(wPtr + 64, wPtr + 64, wPtr + 224, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(160);
asc_mul(wPtr, wPtr, wPtr + 288, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
if constexpr (sizeof(T) == sizeof(float)) {
asc_gather((__ubuf__ float *)dst.GetPhyAddr(), wPtr, indexPtr + 448, uint16_t(8), uint8_t(1));
} else {
auto packed = mem.PackedFloat().template ReinterpretCast<half>();
auto packedPtr = (__ubuf__ half *)packed.GetPhyAddr();
asc_float2half_rn(packedPtr, wPtr, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(4), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_gather((__ubuf__ half *)dst.GetPhyAddr(), packedPtr, indexPtr + 448, uint16_t(8), uint8_t(1));
}
AscendC::SetMaskNorm(); AscendC::ResetMask();
AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
AscendC::DataCopyExtParams op{1, 160 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(outGm[pos * 5], dst, op);
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0); AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
}
};

template <typename T, uint32_t TILE>
class FireFrameStorageStock1112 {
static_assert(TILE >= 128 && TILE <= 1024 && TILE % 128 == 0, "native hardware frames or large reuse frame");
static constexpr uint32_t FRAME = TILE;
static constexpr uint32_t DEPTH = TILE > 256 ? 2 : 1;
static constexpr uint32_t STRIDE = FRAME + 16;
static constexpr uint32_t RAW_BYTES = (sizeof(T) == sizeof(float) ? FRAME : TILE) * 6 * sizeof(T);
static constexpr uint32_t PRIOR_WORDS = FRAME * 5;
static constexpr uint32_t PRIOR_BYTES = PRIOR_WORDS * sizeof(float);
static constexpr uint32_t OUT_BYTES = (sizeof(T) == sizeof(float) ? FRAME : TILE) * 5 * sizeof(T);
static constexpr uint32_t WORK_BYTES = ((STRIDE * 11 * sizeof(float) + 511) / 512) * 512 + 256;
static constexpr uint32_t ONE_WORDS = FRAME * 6;
static constexpr uint32_t ONE_BYTES = ONE_WORDS * sizeof(float) + (sizeof(T) == sizeof(half) ? 256 : 0);
static constexpr uint32_t RAW_BASE = 0;
static constexpr uint32_t PRIOR_BASE = RAW_BASE + DEPTH * RAW_BYTES;
static constexpr uint32_t OUT_BASE = PRIOR_BASE + PRIOR_BYTES;
static constexpr uint32_t WORK_BASE = OUT_BASE + DEPTH * OUT_BYTES;
static constexpr uint32_t ONE_BASE = WORK_BASE + WORK_BYTES;
static constexpr uint32_t RF_BASE = ONE_BASE + ONE_BYTES;
static constexpr uint32_t PACK_BASE = RF_BASE + (sizeof(T) == sizeof(half) ? FRAME * 6 * sizeof(float) : 0);
static constexpr uint32_t INDEX_BASE = PACK_BASE + (sizeof(T) == sizeof(half) ? FRAME * 5 * sizeof(float) + 256 : 0);
static constexpr uint32_t VA_BASE = INDEX_BASE + FRAME * sizeof(uint32_t);
static constexpr uint32_t VA_WORDS = 172;
static_assert(VA_BASE + VA_WORDS * sizeof(uint64_t) < 188416, "reserve the SDK temporary UB region");
static_assert((RAW_BYTES | PRIOR_BYTES | OUT_BYTES | WORK_BYTES | ONE_BYTES | INDEX_BASE) % 32 == 0,
"all static addresses and capacities retain 32-byte alignment");
uint32_t rawCursor = 0, priorCursor = 0, outCursor = 0;
uint32_t rawSlot = 0, priorSlot = 0, outSlot = 0;
uint32_t rawUsed = 0, priorUsed = 0, outUsed = 0;
public:
__aicore__ inline AscendC::LocalTensor<uint64_t> PackedVA() {
return AscendC::LocalTensor<uint64_t>(AscendC::TPosition::VECCALC, VA_BASE, VA_WORDS);
}
__aicore__ inline AscendC::LocalTensor<int32_t> Indices() {
return AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, INDEX_BASE, FRAME);
}
__aicore__ inline AscendC::LocalTensor<float> Work() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, WORK_BASE, WORK_BYTES / sizeof(float));
}
__aicore__ inline AscendC::LocalTensor<float> Ones() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE, ONE_BYTES / sizeof(float));
}
__aicore__ inline AscendC::LocalTensor<float> RawFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, RF_BASE, FRAME * 6);
}
__aicore__ inline AscendC::LocalTensor<float> PackedFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, PACK_BASE, FRAME * 5);
}
__aicore__ inline AscendC::LocalTensor<T> AllocRaw() {
rawSlot = rawCursor;
rawCursor = (rawCursor + 1) % DEPTH;
if (rawUsed & (1U << rawSlot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(rawSlot);
return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, RAW_BASE + rawSlot * RAW_BYTES, (sizeof(T) == sizeof(float) ? FRAME : TILE) * 6);
}
__aicore__ inline AscendC::LocalTensor<float> AllocPrior() {
priorSlot = priorCursor;
priorCursor = 0;
if (priorUsed & (1U << priorSlot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2 + priorSlot);
return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE + priorSlot * PRIOR_BYTES, PRIOR_WORDS);
}
__aicore__ inline AscendC::LocalTensor<T> AllocOut() {
outSlot = outCursor;
outCursor = (outCursor + 1) % DEPTH;
if (outUsed & (1U << outSlot)) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outSlot);
return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, OUT_BASE + outSlot * OUT_BYTES, (sizeof(T) == sizeof(float) ? FRAME : TILE) * 5);
}
__aicore__ inline AscendC::LocalTensor<T> RawFixedSlot(uint32_t slot) {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, RAW_BASE + slot * RAW_BYTES,
(sizeof(T) == sizeof(float) ? FRAME : TILE) * 6);
}
__aicore__ inline AscendC::LocalTensor<float> PriorFixedSlot(uint32_t slot) {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE + slot * PRIOR_BYTES, PRIOR_WORDS);
}
__aicore__ inline AscendC::LocalTensor<T> OutAt(uint32_t slot) {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, OUT_BASE + slot * OUT_BYTES,
(sizeof(T) == sizeof(float) ? FRAME : TILE) * 5);
}
__aicore__ inline uint32_t CurrentOutSlot() const { return outSlot; }
__aicore__ inline void RawReady() { AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(rawSlot); }
__aicore__ inline void RawWait() { AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(rawSlot); }
__aicore__ inline void PriorReady() { AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2 + priorSlot); }
__aicore__ inline void PriorWait() { AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2 + priorSlot); }
__aicore__ inline void OutReady() { AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outSlot); }
__aicore__ inline void OutWait() { AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outSlot); }
__aicore__ inline void FreeRaw() {
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(rawSlot);
rawUsed |= 1U << rawSlot;
}
__aicore__ inline void FreePrior() {
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2 + priorSlot);
priorUsed |= 1U << priorSlot;
}
__aicore__ inline void FreeOut() {
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outSlot);
outUsed |= 1U << outSlot;
}
__aicore__ inline void Drain() {
for (uint32_t slot = 0; slot < DEPTH; ++slot) {
if (rawUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
if (priorUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2 + slot);
if (outUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
}
}
};

template <typename T, uint32_t TILE, bool ROW, bool RAW3 = false, bool DIRECT_STRIDE = false, uint32_t QUALIFIED_TAIL_N = 0U>
class KernelFireFrameStock1112 {
static_assert(QUALIFIED_TAIL_N == 0U || ((QUALIFIED_TAIL_N == 32768U || QUALIFIED_TAIL_N == 32769U) && sizeof(T) == sizeof(half) && TILE == 384U && !ROW && RAW3 && !DIRECT_STRIDE), "qualified N32769 half partial tail384");
static constexpr uint32_t DEPTH = TILE > 256 ? 2 : 1;
static constexpr uint32_t FRAME = TILE;
static constexpr uint32_t STRIDE = FRAME + 16;
static constexpr uint32_t Q = FRAME / 8;
FireFrameStorageStock1112<T, TILE> mem;
AscendC::GlobalTensor<T> rawGm, outGm;
AscendC::GlobalTensor<float> priorGm;
// Fixed CANN9/C220 compiler-builtin experiment. Tables remain read-only after S_V0.
__aicore__ inline void StorePacked(AscendC::LocalTensor<uint64_t> dst, const uint64_t *address) {
#pragma unroll
for (uint32_t q = 0; q < 4; ++q) {
const uint64_t value = ((address[4 * q] >> 5) & 0x1fffULL) |
(((address[4 * q + 1] >> 5) & 0x1fffULL) << 16) |
(((address[4 * q + 2] >> 5) & 0x1fffULL) << 32) |
(((address[4 * q + 3] >> 5) & 0x1fffULL) << 48);
dst.SetValue(q, value);
}
}
template <uint8_t REPEAT, uint16_t DST_STRIDE, uint16_t SRC_STRIDE>
__aicore__ inline void PackedTranspose(AscendC::LocalTensor<uint64_t> dst,
AscendC::LocalTensor<uint64_t> src) {
auto *dstPacked = reinterpret_cast<__ubuf__ uint64_t *>(dst.GetPhyAddr());
auto *srcPacked = reinterpret_cast<__ubuf__ uint64_t *>(src.GetPhyAddr());
__cce_scalar::vld_va_reg(VA0, dstPacked, L128);
__cce_scalar::vld_va_reg(VA1, dstPacked, H128);
__cce_scalar::vld_va_reg(VA2, srcPacked, L128);
__cce_scalar::vld_va_reg(VA3, srcPacked, H128);
// A single repeat applies rep-strides to the first address, unlike multiple repeats.
__cce_scalar::scatter_vnchwconv_b32(VA0, VA2, REPEAT,
REPEAT == 1 ? 0 : DST_STRIDE, REPEAT == 1 ? 0 : SRC_STRIDE);
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitInputPhase() {
static_assert(PHASE < 6 && SLOT < (sizeof(T) == sizeof(float) ? 2 : 1),
"valid padded input specialization");
static_assert(!RAW3 || ((((FRAME == 896 || FRAME == 768 || (FRAME == 384 && (QUALIFIED_TAIL_N == 32768U || QUALIFIED_TAIL_N == 32769U))) && !ROW) || ((FRAME == 512 || FRAME == 640) && ROW)) && PHASE < 3), "raw3 verified reuse or qualified tail frames");
uint64_t sourceBase;
if constexpr (sizeof(T) == sizeof(float))
sourceBase = reinterpret_cast<uint64_t>(mem.RawFixedSlot(SLOT).GetPhyAddr());
else sourceBase = reinterpret_cast<uint64_t>(mem.RawFloat().GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
if constexpr (RAW3) {
uint64_t source[16] = {
sourceBase + uint64_t(0 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(48 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(96 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(144 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(192 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(240 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(288 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(336 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(24 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(72 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(120 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(168 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(216 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(264 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(312 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(360 + 8 * PHASE) * sizeof(float)
};
uint64_t target[16] = {
workBase + uint64_t(((8 * PHASE + 0) % 6) * STRIDE + (((8 * PHASE + 0) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 0) % 6) * STRIDE + (((8 * PHASE + 0) / 6) + 4) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 1) % 6) * STRIDE + (((8 * PHASE + 1) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 1) % 6) * STRIDE + (((8 * PHASE + 1) / 6) + 4) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 2) % 6) * STRIDE + (((8 * PHASE + 2) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 2) % 6) * STRIDE + (((8 * PHASE + 2) / 6) + 4) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 3) % 6) * STRIDE + (((8 * PHASE + 3) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 3) % 6) * STRIDE + (((8 * PHASE + 3) / 6) + 4) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 4) % 6) * STRIDE + (((8 * PHASE + 4) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 4) % 6) * STRIDE + (((8 * PHASE + 4) / 6) + 4) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 5) % 6) * STRIDE + (((8 * PHASE + 5) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 5) % 6) * STRIDE + (((8 * PHASE + 5) / 6) + 4) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 6) % 6) * STRIDE + (((8 * PHASE + 6) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 6) % 6) * STRIDE + (((8 * PHASE + 6) / 6) + 4) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 7) % 6) * STRIDE + (((8 * PHASE + 7) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 7) % 6) * STRIDE + (((8 * PHASE + 7) / 6) + 4) * Q) * sizeof(float)
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[4 * PHASE], target);
StorePacked(mem.PackedVA()[24 + SLOT * 24 + 4 * PHASE], source);
} else {
uint64_t source[16] = {
sourceBase + uint64_t(0 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(48 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(96 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(144 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(192 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(240 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(288 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(336 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(384 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(432 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(480 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(528 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(576 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(624 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(672 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(720 + 8 * PHASE) * sizeof(float)
};
uint64_t target[16] = {
workBase + uint64_t(((8 * PHASE + 0) % 6) * STRIDE + ((8 * PHASE + 0) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 0) % 6) * STRIDE + ((8 * PHASE + 0) / 6) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 1) % 6) * STRIDE + ((8 * PHASE + 1) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 1) % 6) * STRIDE + ((8 * PHASE + 1) / 6) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 2) % 6) * STRIDE + ((8 * PHASE + 2) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 2) % 6) * STRIDE + ((8 * PHASE + 2) / 6) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 3) % 6) * STRIDE + ((8 * PHASE + 3) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 3) % 6) * STRIDE + ((8 * PHASE + 3) / 6) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 4) % 6) * STRIDE + ((8 * PHASE + 4) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 4) % 6) * STRIDE + ((8 * PHASE + 4) / 6) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 5) % 6) * STRIDE + ((8 * PHASE + 5) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 5) % 6) * STRIDE + ((8 * PHASE + 5) / 6) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 6) % 6) * STRIDE + ((8 * PHASE + 6) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 6) % 6) * STRIDE + ((8 * PHASE + 6) / 6) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 7) % 6) * STRIDE + ((8 * PHASE + 7) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 7) % 6) * STRIDE + ((8 * PHASE + 7) / 6) * Q + 8) * sizeof(float)
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[4 * PHASE], target);
StorePacked(mem.PackedVA()[24 + SLOT * 24 + 4 * PHASE], source);
}
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitOutputPhase() {
static_assert(PHASE < 5 && SLOT < (sizeof(T) == sizeof(float) ? 2 : 1),
"valid padded output specialization");
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t outputBase;
if constexpr (sizeof(T) == sizeof(float))
outputBase = reinterpret_cast<uint64_t>(mem.OutAt(SLOT).GetPhyAddr());
else outputBase = reinterpret_cast<uint64_t>(mem.PackedFloat().GetPhyAddr());
uint64_t source[16] = {
workBase + uint64_t(((8 * PHASE + 0) % 5) * STRIDE + ((8 * PHASE + 0) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 1) % 5) * STRIDE + ((8 * PHASE + 1) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 2) % 5) * STRIDE + ((8 * PHASE + 2) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 3) % 5) * STRIDE + ((8 * PHASE + 3) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 4) % 5) * STRIDE + ((8 * PHASE + 4) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 5) % 5) * STRIDE + ((8 * PHASE + 5) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 6) % 5) * STRIDE + ((8 * PHASE + 6) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 7) % 5) * STRIDE + ((8 * PHASE + 7) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 0) % 5) * STRIDE + ((8 * PHASE + 0) / 5) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 1) % 5) * STRIDE + ((8 * PHASE + 1) / 5) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 2) % 5) * STRIDE + ((8 * PHASE + 2) / 5) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 3) % 5) * STRIDE + ((8 * PHASE + 3) / 5) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 4) % 5) * STRIDE + ((8 * PHASE + 4) / 5) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 5) % 5) * STRIDE + ((8 * PHASE + 5) / 5) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 6) % 5) * STRIDE + ((8 * PHASE + 6) / 5) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 7) % 5) * STRIDE + ((8 * PHASE + 7) / 5) * Q + 8) * sizeof(float)
};
uint64_t target[16] = {
outputBase + uint64_t((5 * 0 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 8 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 1 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 9 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 2 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 10 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 3 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 11 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 4 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 12 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 5 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 13 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 6 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 14 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 7 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 15 + PHASE) * 8) * sizeof(float)
};
constexpr uint32_t INPUT_WORDS = 72;
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[INPUT_WORDS + 4 * PHASE], source);
StorePacked(mem.PackedVA()[INPUT_WORDS + 20 + SLOT * 20 + 4 * PHASE], target);
}
template <uint32_t PHASE>
__aicore__ inline void InitPriorPhase() {
static_assert(PHASE < 5, "native prior hardware phase");
constexpr uint32_t PRIOR_TARGET = 132;
constexpr uint32_t PRIOR_SOURCE = 152 + 4 * PHASE;
const uint64_t sourceBase = reinterpret_cast<uint64_t>(mem.PriorFixedSlot(0).GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t source[16] = {
sourceBase + uint64_t(0 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(40 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(80 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(120 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(160 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(200 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(240 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(280 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(320 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(360 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(400 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(440 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(480 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(520 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(560 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(600 + 8 * PHASE) * sizeof(float)
};
uint64_t target[16] = {
workBase + uint64_t((6 + (8 * PHASE + 0) % 5) * STRIDE + ((8 * PHASE + 0) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 0) % 5) * STRIDE + ((8 * PHASE + 0) / 5) * Q + 1 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 1) % 5) * STRIDE + ((8 * PHASE + 1) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 1) % 5) * STRIDE + ((8 * PHASE + 1) / 5) * Q + 1 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 2) % 5) * STRIDE + ((8 * PHASE + 2) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 2) % 5) * STRIDE + ((8 * PHASE + 2) / 5) * Q + 1 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 3) % 5) * STRIDE + ((8 * PHASE + 3) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 3) % 5) * STRIDE + ((8 * PHASE + 3) / 5) * Q + 1 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 4) % 5) * STRIDE + ((8 * PHASE + 4) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 4) % 5) * STRIDE + ((8 * PHASE + 4) / 5) * Q + 1 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 5) % 5) * STRIDE + ((8 * PHASE + 5) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 5) % 5) * STRIDE + ((8 * PHASE + 5) / 5) * Q + 1 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 6) % 5) * STRIDE + ((8 * PHASE + 6) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 6) % 5) * STRIDE + ((8 * PHASE + 6) / 5) * Q + 1 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 7) % 5) * STRIDE + ((8 * PHASE + 7) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 7) % 5) * STRIDE + ((8 * PHASE + 7) / 5) * Q + 1 * 8) * sizeof(float)
};
StorePacked(mem.PackedVA()[PRIOR_TARGET + 4 * PHASE], target);
StorePacked(mem.PackedVA()[PRIOR_SOURCE], source);
}
template <uint32_t PHASE>
__aicore__ inline void NativePriorPhase() {
constexpr uint32_t PRIOR_TARGET = 132;
constexpr uint32_t PRIOR_SOURCE = 152 + 4 * PHASE;
PackedTranspose<FRAME / 128, 2, 80>(mem.PackedVA()[PRIOR_TARGET + 4 * PHASE],
mem.PackedVA()[PRIOR_SOURCE]);
}
__aicore__ inline void InitPackedVA() {
InitInputPhase<0, 0>();
InitInputPhase<0, 1>();
InitInputPhase<0, 2>();
if constexpr (!RAW3) {
InitInputPhase<0, 3>();
InitInputPhase<0, 4>();
InitInputPhase<0, 5>();
}
if constexpr (sizeof(T) == sizeof(float) && !(ROW)) {
InitInputPhase<1, 0>();
InitInputPhase<1, 1>();
InitInputPhase<1, 2>();
if constexpr (!RAW3) {
InitInputPhase<1, 3>();
InitInputPhase<1, 4>();
InitInputPhase<1, 5>();
}
}
InitOutputPhase<0, 0>();
InitOutputPhase<0, 1>();
InitOutputPhase<0, 2>();
InitOutputPhase<0, 3>();
InitOutputPhase<0, 4>();
if constexpr (sizeof(T) == sizeof(float) && !(ROW)) {
InitOutputPhase<1, 0>();
InitOutputPhase<1, 1>();
InitOutputPhase<1, 2>();
InitOutputPhase<1, 3>();
InitOutputPhase<1, 4>();
}
{
InitPriorPhase<0>();
InitPriorPhase<1>();
InitPriorPhase<2>();
InitPriorPhase<3>();
InitPriorPhase<4>();
}
AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void NativeInputPhase() {
static_assert(PHASE < 6 && SLOT < (sizeof(T) == sizeof(float) ? 2 : 1), "valid packed input phase");
if constexpr (RAW3) {
PackedTranspose<FRAME / 64, 1, 48>(mem.PackedVA()[4 * PHASE],
mem.PackedVA()[24 + SLOT * 24 + 4 * PHASE]);
} else {
PackedTranspose<FRAME / 128, 2, 96>(mem.PackedVA()[4 * PHASE],
mem.PackedVA()[24 + SLOT * 24 + 4 * PHASE]);
}
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void NativeOutputPhase() {
static_assert(PHASE < 5 && SLOT < (sizeof(T) == sizeof(float) ? 2 : 1), "valid packed output phase");
constexpr uint32_t INPUT_WORDS = 72;
PackedTranspose<FRAME / 128, 80, 2>(mem.PackedVA()[INPUT_WORDS + 20 + SLOT * 20 + 4 * PHASE],
mem.PackedVA()[INPUT_WORDS + 4 * PHASE]);
}
public:
template <bool WHOLE = false>
__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded,
const FireFrameTilingData &schedule) {
static_assert(!WHOLE || ROW, "whole entry requires one batch per core");
static_assert(QUALIFIED_TAIL_N == 0U || !WHOLE, "qualified tail remains partial");
static_assert(!DIRECT_STRIDE || (FRAME == 512 && ROW && WHOLE), "direct stride only original whole512");
constexpr bool rawFirst = WHOLE && DIRECT_STRIDE && RAW3 && FRAME == 512 && sizeof(T) == sizeof(half);
const auto &t = schedule.header;
rawGm.SetGlobalBuffer((__gm__ T *)raw);
priorGm.SetGlobalBuffer((__gm__ float *)priors);
outGm.SetGlobalBuffer((__gm__ T *)decoded);
const uint32_t groupLanes = schedule.geometry & 0xffffU;
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
const uint32_t begin = (block % groupLanes) * t.perCore;
if (begin >= t.n) return;
const uint32_t batchCount = ROW ? 1U : schedule.geometry >> 16;
const uint32_t batchFirst = (block / groupLanes) * batchCount;
const uint32_t priorPos = begin;
const uint32_t count = WHOLE ? FRAME : (t.n - priorPos < t.perCore ? t.n - priorPos : t.perCore);
// Overlap first raw/prior DMA with address-table setup.
auto firstRaw = mem.AllocRaw();
auto p = mem.AllocPrior();
AscendC::DataCopyExtParams firstRp{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyExtParams firstPp{1, count * 5 * uint32_t(sizeof(float)), 0, 0, 0};
AscendC::DataCopyPad(firstRaw, rawGm[(WHOLE ? block * FRAME : batchFirst * t.n + priorPos) * 6], firstRp,
AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
if constexpr (rawFirst) mem.RawReady();
AscendC::DataCopyPad(p, priorGm[priorPos * 5], firstPp,
AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
if constexpr (!rawFirst) mem.RawReady();
mem.PriorReady();
auto one = mem.Ones();
AscendC::Duplicate(one, 1.0f, FRAME * 6);
AscendC::LocalTensor<float> f[11];
for (uint32_t k = 0; k < 11; ++k) f[k] = mem.Work()[k * STRIDE];
if constexpr (rawFirst) {
mem.RawWait();
auto rf = mem.RawFloat();
AscendC::Cast(rf, firstRaw, AscendC::RoundMode::CAST_NONE, count * 6);
AscendC::Muls(rf, rf, -1.0f, count * 6);
AscendC::Exp(rf, rf, count * 6);
AscendC::Adds(rf, rf, 1.0f, count * 6);
AscendC::Div(rf, one, rf, count * 6);
AscendC::PipeBarrier<PIPE_V>();
}
InitPackedVA();
{
mem.PriorWait();
NativePriorPhase<0>();
NativePriorPhase<1>();
NativePriorPhase<2>();
NativePriorPhase<3>();
NativePriorPhase<4>();
AscendC::PipeBarrier<PIPE_V>();
if constexpr (!DIRECT_STRIDE) {
// The original prior AoS has been consumed; copy stride bits exactly.
AscendC::DataCopy(p, f[8], STRIDE);
AscendC::DataCopy(p[STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[2 * STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[3 * STRIDE], f[8], STRIDE);
AscendC::PipeBarrier<PIPE_V>();
}
// Cached f[6..10] are read-only throughout the batch loop.
for (uint32_t batch = batchFirst; batch < batchFirst + batchCount; ++batch) {
const uint32_t pos = WHOLE ? block * FRAME : batch * t.n + priorPos;
auto r = firstRaw;
if (batch != batchFirst) {
r = mem.AllocRaw();
AscendC::DataCopyExtParams rp{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(r, rawGm[pos * 6], rp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
mem.RawReady();
}
AscendC::LocalTensor<float> rf;
if constexpr (rawFirst) rf = mem.RawFloat();
else {
mem.RawWait();
if constexpr (sizeof(T) == sizeof(float)) rf = r.template ReinterpretCast<float>();
else {
rf = mem.RawFloat();
AscendC::Cast(rf, r, AscendC::RoundMode::CAST_NONE, count * 6);
}
AscendC::Muls(rf, rf, -1.0f, count * 6);
AscendC::Exp(rf, rf, count * 6);
AscendC::Adds(rf, rf, 1.0f, count * 6);
AscendC::Div(rf, one, rf, count * 6);
// Raw/RF has6*FRAME allocated words. Unloaded padding never reaches valid GM output.
AscendC::PipeBarrier<PIPE_V>();
}
if constexpr (sizeof(T) == sizeof(float)) {
if constexpr (ROW) {
NativeInputPhase<0, 0>();
NativeInputPhase<0, 1>();
NativeInputPhase<0, 2>();
if constexpr (!RAW3) {
NativeInputPhase<0, 3>();
NativeInputPhase<0, 4>();
NativeInputPhase<0, 5>();
}
} else if (reinterpret_cast<uint64_t>(rf.GetPhyAddr()) ==
reinterpret_cast<uint64_t>(mem.RawFixedSlot(0).GetPhyAddr())) {
NativeInputPhase<0, 0>();
NativeInputPhase<0, 1>();
NativeInputPhase<0, 2>();
if constexpr (!RAW3) {
NativeInputPhase<0, 3>();
NativeInputPhase<0, 4>();
NativeInputPhase<0, 5>();
}
} else {
NativeInputPhase<1, 0>();
NativeInputPhase<1, 1>();
NativeInputPhase<1, 2>();
if constexpr (!RAW3) {
NativeInputPhase<1, 3>();
NativeInputPhase<1, 4>();
NativeInputPhase<1, 5>();
}
}
} else {
NativeInputPhase<0, 0>();
NativeInputPhase<0, 1>();
NativeInputPhase<0, 2>();
if constexpr (!RAW3) {
NativeInputPhase<0, 3>();
NativeInputPhase<0, 4>();
NativeInputPhase<0, 5>();
}
}
AscendC::PipeBarrier<PIPE_V>();
if constexpr (WHOLE && DIRECT_STRIDE) {
auto *wPtr = reinterpret_cast<__ubuf__ float *>(mem.Work().GetPhyAddr());
AscendC::SetMaskCount();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(4 * STRIDE);
asc_mul_scalar(wPtr, wPtr, 2.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(2 * STRIDE);
asc_add_scalar(wPtr, wPtr, -0.5f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
asc_mul(wPtr + 2 * STRIDE, wPtr + 2 * STRIDE, wPtr + 2 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_add(wPtr, wPtr, wPtr + 6 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
asc_mul(wPtr + 2 * STRIDE, wPtr + 2 * STRIDE, wPtr + 9 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(FRAME);
#pragma unroll
for (uint32_t k = 0; k < 4; ++k)
asc_mul(wPtr + k * STRIDE, wPtr + k * STRIDE, wPtr + 8 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
asc_mul(wPtr + 4 * STRIDE, wPtr + 4 * STRIDE, wPtr + 5 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::SetMaskNorm();
AscendC::ResetMask();
} else {
AscendC::Muls(f[0], f[0], 2.0f, 4 * STRIDE);
AscendC::Adds(f[0], f[0], -0.5f, 2 * STRIDE);
AscendC::Add(f[0], f[0], f[6], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[2], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[9], 2 * STRIDE);
if constexpr (DIRECT_STRIDE) {
#pragma unroll
for (uint32_t k = 0; k < 4; ++k)
AscendC::Mul(f[k], f[k], f[8], FRAME);
} else {
AscendC::Mul(f[0], f[0], p, 4 * STRIDE);
}
AscendC::Mul(f[4], f[4], f[5], FRAME);
}
auto out = mem.AllocOut();
AscendC::PipeBarrier<PIPE_V>();
if constexpr (sizeof(T) == sizeof(float)) {
if constexpr (ROW) {
NativeOutputPhase<0, 0>();
NativeOutputPhase<0, 1>();
NativeOutputPhase<0, 2>();
NativeOutputPhase<0, 3>();
NativeOutputPhase<0, 4>();
} else if (mem.CurrentOutSlot() == 0) {
NativeOutputPhase<0, 0>();
NativeOutputPhase<0, 1>();
NativeOutputPhase<0, 2>();
NativeOutputPhase<0, 3>();
NativeOutputPhase<0, 4>();
} else {
NativeOutputPhase<1, 0>();
NativeOutputPhase<1, 1>();
NativeOutputPhase<1, 2>();
NativeOutputPhase<1, 3>();
NativeOutputPhase<1, 4>();
}
} else {
NativeOutputPhase<0, 0>();
NativeOutputPhase<0, 1>();
NativeOutputPhase<0, 2>();
NativeOutputPhase<0, 3>();
NativeOutputPhase<0, 4>();
}
AscendC::PipeBarrier<PIPE_V>();
if constexpr (sizeof(T) != sizeof(float))
AscendC::Cast(out, mem.PackedFloat(), AscendC::RoundMode::CAST_RINT, count * 5);
mem.OutReady(); mem.OutWait();
AscendC::DataCopyExtParams op{1, count * 5 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(outGm[pos * 5], out, op);
mem.FreeOut();
mem.FreeRaw();
}
mem.FreePrior();
}
mem.Drain();
}
};

template <typename T, uint32_t N, bool QUALIFIED_HALF_INPUT_PI8 = false>
class KernelFirePairInputPi8H11 {
static_assert(QUALIFIED_HALF_INPUT_PI8 && sizeof(T) == sizeof(half) && N == 32768U, "qualified half N32768 input B16 pi8");
static constexpr uint32_t FRAME=1664U, STRIDE=1696U, Q=208U;
uint32_t used=0;
AscendC::GlobalTensor<T> rawGm,outGm;
AscendC::GlobalTensor<float> priorGm;
__aicore__ inline AscendC::LocalTensor<T> Raw(uint32_t slot) { return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN,slot*19968U,9984U); }
__aicore__ inline AscendC::LocalTensor<T> Out(uint32_t slot) { return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT,39936U+slot*16640U,8320U); }
__aicore__ inline AscendC::LocalTensor<float> Prior() { return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN,39936U,4160U); }
__aicore__ inline AscendC::LocalTensor<float> Work() { return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC,73216U,18656U); }
__aicore__ inline AscendC::LocalTensor<float> Ones() { return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC,147840U,8U); }
__aicore__ inline AscendC::LocalTensor<float> Twos() { return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC,147872U,8U); }
__aicore__ inline AscendC::LocalTensor<T> HalfRF() { return AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC,148096U,10176U); }
__aicore__ inline AscendC::LocalTensor<float> PackedFloat() { return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC,148096U,8320U); }
__aicore__ inline AscendC::LocalTensor<uint64_t> VA() { return AscendC::LocalTensor<uint64_t>(AscendC::TPosition::VECCALC,187232U,132U); }
__aicore__ inline AscendC::LocalTensor<T> AllocRaw(uint32_t slot) {
if(used&(1U<<slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
return Raw(slot);
}
__aicore__ inline void ReleaseRaw(uint32_t slot) { AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);used|=1U<<slot; }
__aicore__ inline void AcquireOutput(uint32_t slot) { if(used&(1U<<(slot+2U))) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot); }
__aicore__ inline void PublishOutput(uint32_t slot) { AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot);used|=1U<<(slot+2U); }
__aicore__ inline void Drain() {
for(uint32_t slot=0;slot<2;++slot) {
if(used&(1U<<slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
if(used&(1U<<(slot+2U))) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
}
}
__aicore__ inline void StorePacked(uint32_t offset,const uint64_t *a) {
auto dst=VA()[offset];
#pragma unroll
for(uint32_t q=0;q<4;++q) dst.SetValue(q,((a[4*q]>>5)&0x1fffULL)|(((a[4*q+1]>>5)&0x1fffULL)<<16)|(((a[4*q+2]>>5)&0x1fffULL)<<32)|(((a[4*q+3]>>5)&0x1fffULL)<<48));
}
template<uint32_t PHASE>
__aicore__ inline void InitInput() {
uint64_t target[16],source[16];
const uint64_t h=reinterpret_cast<uint64_t>(HalfRF().GetPhyAddr());
#pragma unroll
for(uint32_t k=0;k<16;++k) target[k]=h+(((16U*PHASE+k)%6U)*STRIDE+((16U*PHASE+k)/6U)*Q)*2U;
StorePacked(80U+4U*PHASE,target);
#pragma unroll
for(uint32_t slot=0;slot<2;++slot) {
const uint64_t r=reinterpret_cast<uint64_t>(Raw(slot).GetPhyAddr());
#pragma unroll
for(uint32_t i=0;i<16;++i) source[i]=r+(48U*i+16U*PHASE)*2U;
if constexpr(PHASE==0U) StorePacked(4U*slot,source);
}
}
template<uint32_t PHASE>
__aicore__ inline void InitOutput() {
uint64_t target[16],source[16];
const uint64_t w=reinterpret_cast<uint64_t>(Work().GetPhyAddr());
const uint64_t r=reinterpret_cast<uint64_t>(PackedFloat().GetPhyAddr());
#pragma unroll
for(uint32_t i=0;i<16;++i) source[i]=w+(((8U*PHASE+i%8U)%5U)*STRIDE+((8U*PHASE+i%8U)/5U)*Q+(i/8U)*8U)*4U;
#pragma unroll
for(uint32_t k=0;k<16;++k) target[k]=r+(5U*(k/2U+8U*(k%2U))+PHASE)*32U;
StorePacked(92U+4U*PHASE,source);
if constexpr(PHASE==0U) StorePacked(8U,target);
}
template<uint32_t PHASE>
__aicore__ inline void InitPrior() {
uint64_t target[16],source[16];
const uint64_t p=reinterpret_cast<uint64_t>(Prior().GetPhyAddr());
const uint64_t w=reinterpret_cast<uint64_t>(Work().GetPhyAddr());
#pragma unroll
for(uint32_t i=0;i<16;++i) source[i]=p+(40U*(i%8U)+8U*PHASE)*4U;
#pragma unroll
for(uint32_t k=0;k<8;++k) {
target[2*k]=w+((6U+(8U*PHASE+k)%5U)*STRIDE+((8U*PHASE+k)/5U)*Q)*4U;
target[2*k+1]=target[2*k]+104U*4U;
}
StorePacked(112U+4U*PHASE,target);
if constexpr(PHASE==0U) StorePacked(12U,source);
}
template<bool B16,uint8_t REPEAT,uint16_t DST,uint16_t SRC>
__aicore__ inline void Transpose(uint32_t dst,uint32_t src) {
auto *d=reinterpret_cast<__ubuf__ uint64_t *>(VA()[dst].GetPhyAddr());
auto *s=reinterpret_cast<__ubuf__ uint64_t *>(VA()[src].GetPhyAddr());
__cce_scalar::vld_va_reg(VA0,d,L128);__cce_scalar::vld_va_reg(VA1,d,H128);
__cce_scalar::vld_va_reg(VA2,s,L128);__cce_scalar::vld_va_reg(VA3,s,H128);
if constexpr(B16) __cce_scalar::scatter_vnchwconv_b16(VA0,VA2,REPEAT,DST,SRC);
else __cce_scalar::scatter_vnchwconv_b32(VA0,VA2,REPEAT,DST,SRC);
}
__aicore__ inline void InitVA() {
InitInput<0>();InitInput<1>();InitInput<2>();
InitOutput<0>();InitOutput<1>();InitOutput<2>();InitOutput<3>();InitOutput<4>();
InitPrior<0>();InitPrior<1>();InitPrior<2>();InitPrior<3>();InitPrior<4>();
AscendC::SetFlag<AscendC::HardEvent::S_V>(0);AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
auto *entries=reinterpret_cast<__ubuf__ int16_t *>(VA().GetPhyAddr());
AscendC::SetMaskCount();AscendC::SetVectorMask<int16_t,AscendC::MaskMode::COUNTER>(64U);
#pragma unroll
for(uint32_t phase=1U;phase<5U;++phase)
asc_add_scalar(entries+phase*64U,entries,int16_t(phase),uint8_t(1),uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();AscendC::SetMaskNorm();AscendC::ResetMask();
}
template<uint32_t SLOT>
__aicore__ inline void Input() {
Transpose<true,13,1,48>(80,4*SLOT);Transpose<true,13,1,48>(84,16+4*SLOT);Transpose<true,13,1,48>(88,32+4*SLOT);
}
__aicore__ inline void Output() {
Transpose<false,13,80,2>(8,92);Transpose<false,13,80,2>(24,96);Transpose<false,13,80,2>(40,100);
Transpose<false,13,80,2>(56,104);Transpose<false,13,80,2>(72,108);
}
public:
__aicore__ inline void Run(GM_ADDR raw,GM_ADDR priors,GM_ADDR decoded) {
const uint32_t block=static_cast<uint32_t>(AscendC::GetBlockIdx());
if(block>=39U)return;
const uint32_t begin=block*832U,cohort=block&3U;
rawGm.SetGlobalBuffer((__gm__ T *)raw);priorGm.SetGlobalBuffer((__gm__ float *)priors);outGm.SetGlobalBuffer((__gm__ T *)decoded);
const AscendC::DataCopyExtParams rp{2,9984U,(N-832U)*12U,0U,0};
const AscendC::DataCopyExtParams pp{1,16640U,0,0,0};
auto firstRaw=AllocRaw(0);
AscendC::DataCopyPad(firstRaw,rawGm[(2U*cohort*N+begin)*6U],rp,AscendC::DataCopyPadExtParams<T>{false,0,0,0});
AscendC::DataCopyPad(Prior(),priorGm[begin*5U],pp,AscendC::DataCopyPadExtParams<float>{false,0,0,0});
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
AscendC::Duplicate(Ones(),1.0f,8U);AscendC::Duplicate(Twos(),2.0f,8U);
AscendC::LocalTensor<float> f[11];
for(uint32_t k=0;k<11;++k)f[k]=Work()[k*STRIDE];
InitVA();AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
Transpose<false,13,1,40>(112,12);Transpose<false,13,1,40>(116,28);Transpose<false,13,1,40>(120,44);
Transpose<false,13,1,40>(124,60);Transpose<false,13,1,40>(128,76);
AscendC::PipeBarrier<PIPE_V>();
for(uint32_t iteration=0;iteration<4;++iteration) {
const uint32_t pair=(iteration+cohort)&3U,slot=iteration&1U,pos=2U*pair*N+begin;
if(iteration!=0) {
auto r=AllocRaw(slot);
AscendC::DataCopyPad(r,rawGm[pos*6U],rp,AscendC::DataCopyPadExtParams<T>{false,0,0,0});
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot);
}
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);
auto signedRaw=Raw(slot).template ReinterpretCast<int16_t>();
AscendC::Adds(signedRaw,signedRaw,int16_t(-32768),9984U);
AscendC::PipeBarrier<PIPE_V>();
if(slot==0)Input<0>();else Input<1>();
AscendC::PipeBarrier<PIPE_V>();ReleaseRaw(slot);
AscendC::Cast(Work(),HalfRF(),AscendC::RoundMode::CAST_NONE,10176U);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Exp(Work(),Work(),10176U);AscendC::Adds(Work(),Work(),1.0f,10176U);
AscendC::Mul(f[4],f[4],f[5],FRAME);
const AscendC::BinaryRepeatParams divBinary{1,0,1,8,0,8};
AscendC::Div<float>(Work(),Twos(),Work(),uint64_t(64),uint8_t(106),divBinary);
AscendC::Reciprocal<float>(f[4],f[4],uint64_t(64),uint8_t(26),AscendC::UnaryRepeatParams{1,1,8,8});
AscendC::PipeBarrier<PIPE_V>();
auto *wm=reinterpret_cast<__ubuf__ float *>(Work().GetPhyAddr());
AscendC::SetMaskCount();AscendC::SetVectorMask<float,AscendC::MaskMode::COUNTER>(2U*STRIDE);
asc_add_scalar(wm,wm,-0.5f,uint8_t(1),uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));
asc_mul(wm+2U*STRIDE,wm+2U*STRIDE,wm+2U*STRIDE,uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(8),uint8_t(8),uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_add(wm,wm,wm+6U*STRIDE,uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(8),uint8_t(8),uint8_t(8));
asc_mul(wm+2U*STRIDE,wm+2U*STRIDE,wm+9U*STRIDE,uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(8),uint8_t(8),uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
auto *w=reinterpret_cast<__ubuf__ float *>(Work().GetPhyAddr());
AscendC::SetMaskCount();AscendC::SetVectorMask<float,AscendC::MaskMode::COUNTER>(FRAME);
#pragma unroll
for(uint32_t k=0;k<4;++k)asc_mul(w+k*STRIDE,w+k*STRIDE,w+8*STRIDE,uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(8),uint8_t(8),uint8_t(8));
AscendC::SetMaskNorm();AscendC::ResetMask();AscendC::PipeBarrier<PIPE_V>();
Output();AscendC::PipeBarrier<PIPE_V>();AcquireOutput(slot);
asc_float2half_rn(reinterpret_cast<__ubuf__ half *>(Out(slot).GetPhyAddr()),reinterpret_cast<__ubuf__ float *>(PackedFloat().GetPhyAddr()),uint32_t(8320U));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot);AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
const AscendC::DataCopyExtParams op{2,8320U,0U,(N-832U)*10U,0};
AscendC::DataCopyPad(outGm[pos*5U],Out(slot),op);PublishOutput(slot);
}
Drain();
}
};

template <typename T, uint32_t N, bool QUALIFIED_HALF_INPUT_PI8 = false>
class KernelFirePairInputPi8H12 {
static_assert(QUALIFIED_HALF_INPUT_PI8 && sizeof(T) == sizeof(half) && N == 32769U, "qualified half N32769 input B16 pi8");
static constexpr uint32_t FRAME=1664U, STRIDE=1696U, Q=208U;
uint32_t used=0;
AscendC::GlobalTensor<T> rawGm,outGm;
AscendC::GlobalTensor<float> priorGm;
__aicore__ inline AscendC::LocalTensor<T> Raw(uint32_t slot) { return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN,slot*19968U,9984U); }
__aicore__ inline AscendC::LocalTensor<T> Out(uint32_t slot) { return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT,39936U+slot*16640U,8320U); }
__aicore__ inline AscendC::LocalTensor<float> Prior() { return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN,39936U,4160U); }
__aicore__ inline AscendC::LocalTensor<float> Work() { return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC,73216U,18656U); }
__aicore__ inline AscendC::LocalTensor<float> Ones() { return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC,147840U,8U); }
__aicore__ inline AscendC::LocalTensor<float> Twos() { return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC,147872U,8U); }
__aicore__ inline AscendC::LocalTensor<T> HalfRF() { return AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC,148096U,10176U); }
__aicore__ inline AscendC::LocalTensor<float> PackedFloat() { return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC,148096U,8320U); }
__aicore__ inline AscendC::LocalTensor<uint64_t> VA() { return AscendC::LocalTensor<uint64_t>(AscendC::TPosition::VECCALC,187232U,132U); }
__aicore__ inline AscendC::LocalTensor<T> AllocRaw(uint32_t slot) {
if(used&(1U<<slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
return Raw(slot);
}
__aicore__ inline void ReleaseRaw(uint32_t slot) { AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);used|=1U<<slot; }
__aicore__ inline void AcquireOutput(uint32_t slot) { if(used&(1U<<(slot+2U))) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot); }
__aicore__ inline void PublishOutput(uint32_t slot) { AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot);used|=1U<<(slot+2U); }
__aicore__ inline void Drain() {
for(uint32_t slot=0;slot<2;++slot) {
if(used&(1U<<slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
if(used&(1U<<(slot+2U))) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
}
}
__aicore__ inline void StorePacked(uint32_t offset,const uint64_t *a) {
auto dst=VA()[offset];
#pragma unroll
for(uint32_t q=0;q<4;++q) dst.SetValue(q,((a[4*q]>>5)&0x1fffULL)|(((a[4*q+1]>>5)&0x1fffULL)<<16)|(((a[4*q+2]>>5)&0x1fffULL)<<32)|(((a[4*q+3]>>5)&0x1fffULL)<<48));
}
template<uint32_t PHASE>
__aicore__ inline void InitInput() {
uint64_t target[16],source[16];
const uint64_t h=reinterpret_cast<uint64_t>(HalfRF().GetPhyAddr());
#pragma unroll
for(uint32_t k=0;k<16;++k) target[k]=h+(((16U*PHASE+k)%6U)*STRIDE+((16U*PHASE+k)/6U)*Q)*2U;
StorePacked(80U+4U*PHASE,target);
#pragma unroll
for(uint32_t slot=0;slot<2;++slot) {
const uint64_t r=reinterpret_cast<uint64_t>(Raw(slot).GetPhyAddr());
#pragma unroll
for(uint32_t i=0;i<16;++i) source[i]=r+(48U*i+16U*PHASE)*2U;
if constexpr(PHASE==0U) StorePacked(4U*slot,source);
}
}
template<uint32_t PHASE>
__aicore__ inline void InitOutput() {
uint64_t target[16],source[16];
const uint64_t w=reinterpret_cast<uint64_t>(Work().GetPhyAddr());
const uint64_t r=reinterpret_cast<uint64_t>(PackedFloat().GetPhyAddr());
#pragma unroll
for(uint32_t i=0;i<16;++i) source[i]=w+(((8U*PHASE+i%8U)%5U)*STRIDE+((8U*PHASE+i%8U)/5U)*Q+(i/8U)*8U)*4U;
#pragma unroll
for(uint32_t k=0;k<16;++k) target[k]=r+(5U*(k/2U+8U*(k%2U))+PHASE)*32U;
StorePacked(92U+4U*PHASE,source);
if constexpr(PHASE==0U) StorePacked(8U,target);
}
template<uint32_t PHASE>
__aicore__ inline void InitPrior() {
uint64_t target[16],source[16];
const uint64_t p=reinterpret_cast<uint64_t>(Prior().GetPhyAddr());
const uint64_t w=reinterpret_cast<uint64_t>(Work().GetPhyAddr());
#pragma unroll
for(uint32_t i=0;i<16;++i) source[i]=p+(40U*(i%8U)+8U*PHASE)*4U;
#pragma unroll
for(uint32_t k=0;k<8;++k) {
target[2*k]=w+((6U+(8U*PHASE+k)%5U)*STRIDE+((8U*PHASE+k)/5U)*Q)*4U;
target[2*k+1]=target[2*k]+104U*4U;
}
StorePacked(112U+4U*PHASE,target);
if constexpr(PHASE==0U) StorePacked(12U,source);
}
template<bool B16,uint8_t REPEAT,uint16_t DST,uint16_t SRC>
__aicore__ inline void Transpose(uint32_t dst,uint32_t src) {
auto *d=reinterpret_cast<__ubuf__ uint64_t *>(VA()[dst].GetPhyAddr());
auto *s=reinterpret_cast<__ubuf__ uint64_t *>(VA()[src].GetPhyAddr());
__cce_scalar::vld_va_reg(VA0,d,L128);__cce_scalar::vld_va_reg(VA1,d,H128);
__cce_scalar::vld_va_reg(VA2,s,L128);__cce_scalar::vld_va_reg(VA3,s,H128);
if constexpr(B16) __cce_scalar::scatter_vnchwconv_b16(VA0,VA2,REPEAT,DST,SRC);
else __cce_scalar::scatter_vnchwconv_b32(VA0,VA2,REPEAT,DST,SRC);
}
__aicore__ inline void InitVA() {
InitInput<0>();InitInput<1>();InitInput<2>();
InitOutput<0>();InitOutput<1>();InitOutput<2>();InitOutput<3>();InitOutput<4>();
InitPrior<0>();InitPrior<1>();InitPrior<2>();InitPrior<3>();InitPrior<4>();
AscendC::SetFlag<AscendC::HardEvent::S_V>(0);AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
auto *entries=reinterpret_cast<__ubuf__ int16_t *>(VA().GetPhyAddr());
AscendC::SetMaskCount();AscendC::SetVectorMask<int16_t,AscendC::MaskMode::COUNTER>(64U);
#pragma unroll
for(uint32_t phase=1U;phase<5U;++phase)
asc_add_scalar(entries+phase*64U,entries,int16_t(phase),uint8_t(1),uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();AscendC::SetMaskNorm();AscendC::ResetMask();
}
template<uint32_t SLOT>
__aicore__ inline void Input() {
Transpose<true,13,1,48>(80,4*SLOT);Transpose<true,13,1,48>(84,16+4*SLOT);Transpose<true,13,1,48>(88,32+4*SLOT);
}
__aicore__ inline void Output() {
Transpose<false,13,80,2>(8,92);Transpose<false,13,80,2>(24,96);Transpose<false,13,80,2>(40,100);
Transpose<false,13,80,2>(56,104);Transpose<false,13,80,2>(72,108);
}
public:
__aicore__ inline void Run(GM_ADDR raw,GM_ADDR priors,GM_ADDR decoded) {
const uint32_t block=static_cast<uint32_t>(AscendC::GetBlockIdx());
if(block>=39U)return;
const uint32_t begin=block*832U,cohort=block&3U;
rawGm.SetGlobalBuffer((__gm__ T *)raw);priorGm.SetGlobalBuffer((__gm__ float *)priors);outGm.SetGlobalBuffer((__gm__ T *)decoded);
const AscendC::DataCopyExtParams rp{2,9984U,(N-832U)*12U,0U,0};
const AscendC::DataCopyExtParams pp{1,16640U,0,0,0};
auto firstRaw=AllocRaw(0);
AscendC::DataCopyPad(firstRaw,rawGm[(2U*cohort*N+begin)*6U],rp,AscendC::DataCopyPadExtParams<T>{false,0,0,0});
AscendC::DataCopyPad(Prior(),priorGm[begin*5U],pp,AscendC::DataCopyPadExtParams<float>{false,0,0,0});
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
AscendC::Duplicate(Twos(),2.0f,8U);
AscendC::LocalTensor<float> f[11];
for(uint32_t k=0;k<11;++k)f[k]=Work()[k*STRIDE];
InitVA();AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
Transpose<false,13,1,40>(112,12);Transpose<false,13,1,40>(116,28);Transpose<false,13,1,40>(120,44);
Transpose<false,13,1,40>(124,60);Transpose<false,13,1,40>(128,76);
AscendC::PipeBarrier<PIPE_V>();
for(uint32_t iteration=0;iteration<4;++iteration) {
const uint32_t pair=(iteration+cohort)&3U,slot=iteration&1U,pos=2U*pair*N+begin;
if(iteration!=0) {
auto r=AllocRaw(slot);
AscendC::DataCopyPad(r,rawGm[pos*6U],rp,AscendC::DataCopyPadExtParams<T>{false,0,0,0});
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot);
}
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);
auto signedRaw=Raw(slot).template ReinterpretCast<int16_t>();
AscendC::Adds(signedRaw,signedRaw,int16_t(-32768),9984U);
AscendC::PipeBarrier<PIPE_V>();
if(slot==0)Input<0>();else Input<1>();
AscendC::PipeBarrier<PIPE_V>();ReleaseRaw(slot);
AscendC::Cast(Work(),HalfRF(),AscendC::RoundMode::CAST_NONE,10176U);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Exp(Work(),Work(),10176U);AscendC::Adds(Work(),Work(),1.0f,10176U);
AscendC::Mul(f[4],f[4],f[5],FRAME);
const AscendC::BinaryRepeatParams divBinary{1,0,1,8,0,8};
AscendC::Div<float>(Work(),Twos(),Work(),uint64_t(64),uint8_t(106),divBinary);
AscendC::Reciprocal<float>(f[4],f[4],uint64_t(64),uint8_t(26),AscendC::UnaryRepeatParams{1,1,8,8});
AscendC::PipeBarrier<PIPE_V>();
auto *wm=reinterpret_cast<__ubuf__ float *>(Work().GetPhyAddr());
AscendC::SetMaskCount();AscendC::SetVectorMask<float,AscendC::MaskMode::COUNTER>(2U*STRIDE);
asc_add_scalar(wm,wm,-0.5f,uint8_t(1),uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));
asc_mul(wm+2U*STRIDE,wm+2U*STRIDE,wm+2U*STRIDE,uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(8),uint8_t(8),uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_add(wm,wm,wm+6U*STRIDE,uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(8),uint8_t(8),uint8_t(8));
asc_mul(wm+2U*STRIDE,wm+2U*STRIDE,wm+9U*STRIDE,uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(8),uint8_t(8),uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
auto *w=reinterpret_cast<__ubuf__ float *>(Work().GetPhyAddr());
AscendC::SetMaskCount();AscendC::SetVectorMask<float,AscendC::MaskMode::COUNTER>(STRIDE);
#pragma unroll
for(uint32_t k=0;k<4;++k)asc_mul(w+k*STRIDE,w+k*STRIDE,w+8*STRIDE,uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(8),uint8_t(8),uint8_t(8));
AscendC::SetMaskNorm();AscendC::ResetMask();AscendC::PipeBarrier<PIPE_V>();
Output();AscendC::PipeBarrier<PIPE_V>();AcquireOutput(slot);
asc_float2half_rn(reinterpret_cast<__ubuf__ half *>(Out(slot).GetPhyAddr()),reinterpret_cast<__ubuf__ float *>(PackedFloat().GetPhyAddr()),uint32_t(8320U));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot);AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
const AscendC::DataCopyExtParams op{2,8320U,0U,(N-832U)*10U,0};
AscendC::DataCopyPad(outGm[pos*5U],Out(slot),op);PublishOutput(slot);
}
Drain();
}
};

template <typename T, uint32_t N, uint32_t SPAN, uint32_t COUNT, bool RAW_SCORE = false>
class KernelFirePairSpan1112 {
static_assert(!RAW_SCORE || (N==32769U && SPAN==768U && COUNT==768U), "score qualification scope");
static_assert(sizeof(T)==sizeof(half) && (N==32768U || N==32769U) && SPAN%64U==0U && COUNT<=SPAN && COUNT>SPAN-64U && SPAN<=768U, "qualified paired span");
static constexpr uint32_t FRAME=2U*SPAN, STRIDE=FRAME+16U, Q=SPAN/4U;
uint32_t used=0;
AscendC::GlobalTensor<T> rawGm,outGm;
AscendC::GlobalTensor<float> priorGm;
__aicore__ inline AscendC::LocalTensor<T> Raw(uint32_t slot) { return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN,slot*19968U,9984U); }
__aicore__ inline AscendC::LocalTensor<T> Out(uint32_t slot) { return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT,39936U+slot*16640U,8320U); }
__aicore__ inline AscendC::LocalTensor<float> Prior() { return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN,39936U,4160U); }
__aicore__ inline AscendC::LocalTensor<float> Work() { return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC,73216U,18496U); }
__aicore__ inline AscendC::LocalTensor<float> Ones() { return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC,147200U,8U); }
__aicore__ inline AscendC::LocalTensor<float> Twos() { return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC,147232U,8U); }
__aicore__ inline AscendC::LocalTensor<T> HalfRF() { return AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC,147456U,10080U); }
__aicore__ inline AscendC::LocalTensor<float> PackedFloat() { return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC,147456U,8320U); }
__aicore__ inline AscendC::LocalTensor<uint64_t> VA() { return AscendC::LocalTensor<uint64_t>(AscendC::TPosition::VECCALC,187232U,132U); }
__aicore__ inline AscendC::LocalTensor<T> AllocRaw(uint32_t slot) {
if(used&(1U<<slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
return Raw(slot);
}
__aicore__ inline void ReleaseRaw(uint32_t slot) { AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);used|=1U<<slot; }
__aicore__ inline void AcquireOutput(uint32_t slot) { if(used&(1U<<(slot+2U))) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot); }
__aicore__ inline void PublishOutput(uint32_t slot) { AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot);used|=1U<<(slot+2U); }
__aicore__ inline void Drain() {
for(uint32_t slot=0;slot<2;++slot) {
if(used&(1U<<slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
if(used&(1U<<(slot+2U))) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
}
}
__aicore__ inline void StorePacked(uint32_t offset,const uint64_t *a) {
auto dst=VA()[offset];
#pragma unroll
for(uint32_t q=0;q<4;++q) dst.SetValue(q,((a[4*q]>>5)&0x1fffULL)|(((a[4*q+1]>>5)&0x1fffULL)<<16)|(((a[4*q+2]>>5)&0x1fffULL)<<32)|(((a[4*q+3]>>5)&0x1fffULL)<<48));
}
template<uint32_t PHASE>
__aicore__ inline void InitInput() {
uint64_t target[16],source[16];
const uint64_t h=reinterpret_cast<uint64_t>(HalfRF().GetPhyAddr());
#pragma unroll
for(uint32_t k=0;k<16;++k) target[k]=h+(((16U*PHASE+k)%6U)*STRIDE+((16U*PHASE+k)/6U)*Q)*2U;
StorePacked(80U+4U*PHASE,target);
#pragma unroll
for(uint32_t slot=0;slot<2;++slot) {
const uint64_t r=reinterpret_cast<uint64_t>(Raw(slot).GetPhyAddr());
#pragma unroll
for(uint32_t i=0;i<16;++i) source[i]=r+(48U*i+16U*PHASE)*2U;
if constexpr(PHASE==0U) StorePacked(4U*slot,source);
}
}
template<uint32_t PHASE>
__aicore__ inline void InitOutput() {
uint64_t target[16],source[16];
const uint64_t w=reinterpret_cast<uint64_t>(Work().GetPhyAddr());
const uint64_t r=reinterpret_cast<uint64_t>(PackedFloat().GetPhyAddr());
#pragma unroll
for(uint32_t i=0;i<16;++i) source[i]=w+(((8U*PHASE+i%8U)%5U)*STRIDE+((8U*PHASE+i%8U)/5U)*Q+(i/8U)*8U)*4U;
#pragma unroll
for(uint32_t k=0;k<16;++k) target[k]=r+(5U*(k/2U+8U*(k%2U))+PHASE)*32U;
StorePacked(92U+4U*PHASE,source);
if constexpr(PHASE==0U) StorePacked(8U,target);
}
template<uint32_t PHASE>
__aicore__ inline void InitPrior() {
uint64_t target[16],source[16];
const uint64_t p=reinterpret_cast<uint64_t>(Prior().GetPhyAddr());
const uint64_t w=reinterpret_cast<uint64_t>(Work().GetPhyAddr());
#pragma unroll
for(uint32_t i=0;i<16;++i) source[i]=p+(40U*(i%8U)+8U*PHASE)*4U;
#pragma unroll
for(uint32_t k=0;k<8;++k) {
target[2*k]=w+((6U+(8U*PHASE+k)%5U)*STRIDE+((8U*PHASE+k)/5U)*Q)*4U;
target[2*k+1]=target[2*k]+(SPAN/8U)*4U;
}
StorePacked(112U+4U*PHASE,target);
if constexpr(PHASE==0U) StorePacked(12U,source);
}
template<bool B16,uint8_t REPEAT,uint16_t DST,uint16_t SRC>
__aicore__ inline void Transpose(uint32_t dst,uint32_t src) {
auto *d=reinterpret_cast<__ubuf__ uint64_t *>(VA()[dst].GetPhyAddr());
auto *s=reinterpret_cast<__ubuf__ uint64_t *>(VA()[src].GetPhyAddr());
__cce_scalar::vld_va_reg(VA0,d,L128);__cce_scalar::vld_va_reg(VA1,d,H128);
__cce_scalar::vld_va_reg(VA2,s,L128);__cce_scalar::vld_va_reg(VA3,s,H128);
if constexpr(B16) __cce_scalar::scatter_vnchwconv_b16(VA0,VA2,REPEAT,DST,SRC);
else __cce_scalar::scatter_vnchwconv_b32(VA0,VA2,REPEAT,DST,SRC);
}
__aicore__ inline void InitVA() {
InitInput<0>();InitInput<1>();InitInput<2>();
InitOutput<0>();InitOutput<1>();InitOutput<2>();InitOutput<3>();InitOutput<4>();
InitPrior<0>();InitPrior<1>();InitPrior<2>();InitPrior<3>();InitPrior<4>();
AscendC::SetFlag<AscendC::HardEvent::S_V>(0);AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
auto *entries=reinterpret_cast<__ubuf__ int16_t *>(VA().GetPhyAddr());
AscendC::SetMaskCount();AscendC::SetVectorMask<int16_t,AscendC::MaskMode::COUNTER>(64U);
#pragma unroll
for(uint32_t phase=1U;phase<5U;++phase)
asc_add_scalar(entries+phase*64U,entries,int16_t(phase),uint8_t(1),uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();AscendC::SetMaskNorm();AscendC::ResetMask();
}
template<uint32_t SLOT>
__aicore__ inline void Input() {
Transpose<true,SPAN/64U,1,48>(80,4*SLOT);Transpose<true,SPAN/64U,1,48>(84,16+4*SLOT);Transpose<true,SPAN/64U,1,48>(88,32+4*SLOT);
}
__aicore__ inline void Output() {
Transpose<false,SPAN/64U,80,2>(8,92);Transpose<false,SPAN/64U,80,2>(24,96);Transpose<false,SPAN/64U,80,2>(40,100);
Transpose<false,SPAN/64U,80,2>(56,104);Transpose<false,SPAN/64U,80,2>(72,108);
}
public:
__aicore__ inline void Run(GM_ADDR raw,GM_ADDR priors,GM_ADDR decoded) {
const uint32_t block=static_cast<uint32_t>(AscendC::GetBlockIdx());
if(block>=43U)return;
const uint32_t begin=block*768U,cohort=block&3U;
rawGm.SetGlobalBuffer((__gm__ T *)raw);priorGm.SetGlobalBuffer((__gm__ float *)priors);outGm.SetGlobalBuffer((__gm__ T *)decoded);
const AscendC::DataCopyExtParams rp{2,COUNT*12U,(N-COUNT)*12U,(SPAN*12U-((COUNT*12U+31U)/32U)*32U)/32U,0};
const AscendC::DataCopyExtParams pp{1,COUNT*20U,0,0,0};
auto firstRaw=AllocRaw(0);
AscendC::DataCopyPad(firstRaw,rawGm[(2U*cohort*N+begin)*6U],rp,AscendC::DataCopyPadExtParams<T>{false,0,0,0});
AscendC::DataCopyPad(Prior(),priorGm[begin*5U],pp,AscendC::DataCopyPadExtParams<float>{false,0,0,0});
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
AscendC::Duplicate(Ones(),1.0f,8U);AscendC::Duplicate(Twos(),2.0f,8U);
AscendC::LocalTensor<float> f[11];
for(uint32_t k=0;k<11;++k)f[k]=Work()[k*STRIDE];
InitVA();AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
Transpose<false,SPAN/64U,1,40>(112,12);Transpose<false,SPAN/64U,1,40>(116,28);Transpose<false,SPAN/64U,1,40>(120,44);
Transpose<false,SPAN/64U,1,40>(124,60);Transpose<false,SPAN/64U,1,40>(128,76);
AscendC::PipeBarrier<PIPE_V>();
for(uint32_t iteration=0;iteration<4;++iteration) {
const uint32_t pair=(iteration+cohort)&3U,slot=iteration&1U,pos=2U*pair*N+begin;
if(iteration!=0) {
auto r=AllocRaw(slot);
AscendC::DataCopyPad(r,rawGm[pos*6U],rp,AscendC::DataCopyPadExtParams<T>{false,0,0,0});
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot);
}
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);
auto signedRaw=Raw(slot).template ReinterpretCast<int16_t>();
AscendC::Adds(signedRaw,signedRaw,int16_t(-32768),FRAME*6U);
AscendC::PipeBarrier<PIPE_V>();
if(slot==0)Input<0>();else Input<1>();
AscendC::PipeBarrier<PIPE_V>();ReleaseRaw(slot);
AscendC::Cast(Work(),HalfRF(),AscendC::RoundMode::CAST_NONE,6U*STRIDE);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Exp(Work(),Work(),6U*STRIDE);AscendC::Adds(Work(),Work(),1.0f,6U*STRIDE);
AscendC::Mul(f[4],f[4],f[5],FRAME);
const AscendC::BinaryRepeatParams divBinary{1,0,1,8,0,8};
AscendC::Div<float>(Work(),Twos(),Work(),uint64_t(64),uint8_t(STRIDE/16U),divBinary);
if constexpr (RAW_SCORE) {
AscendC::Reciprocal<float>(f[4],f[4],uint64_t(64),uint8_t(FRAME/64U),AscendC::UnaryRepeatParams{1,1,8,8});
} else {
AscendC::Div<float>(f[4],Ones(),f[4],uint64_t(64),uint8_t(FRAME/64U),divBinary);
}
AscendC::PipeBarrier<PIPE_V>();
AscendC::Adds(f[0],f[0],-0.5f,2U*STRIDE);AscendC::Add(f[0],f[0],f[6],2U*STRIDE);
AscendC::Mul(f[2],f[2],f[2],2U*STRIDE);AscendC::Mul(f[2],f[2],f[9],2U*STRIDE);
AscendC::PipeBarrier<PIPE_V>();
auto *w=reinterpret_cast<__ubuf__ float *>(Work().GetPhyAddr());
AscendC::SetMaskCount();AscendC::SetVectorMask<float,AscendC::MaskMode::COUNTER>(STRIDE);
#pragma unroll
for(uint32_t k=0;k<4;++k)asc_mul(w+k*STRIDE,w+k*STRIDE,w+8*STRIDE,uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(8),uint8_t(8),uint8_t(8));
AscendC::SetMaskNorm();AscendC::ResetMask();AscendC::PipeBarrier<PIPE_V>();
Output();AscendC::PipeBarrier<PIPE_V>();AcquireOutput(slot);
asc_float2half_rn(reinterpret_cast<__ubuf__ half *>(Out(slot).GetPhyAddr()),reinterpret_cast<__ubuf__ float *>(PackedFloat().GetPhyAddr()),uint32_t(FRAME*5U));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot);AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
const AscendC::DataCopyExtParams op{2,COUNT*10U,(SPAN*10U-((COUNT*10U+31U)/32U)*32U)/32U,(N-COUNT)*10U,0};
AscendC::DataCopyPad(outGm[pos*5U],Out(slot),op);PublishOutput(slot);
}
Drain();
}
};

template <typename T, uint32_t N, uint32_t SPAN, uint32_t COUNT, bool RAW_SCORE = false>
class KernelFirePairSpan47 {
static_assert(!RAW_SCORE || (N==32769U && SPAN==704U && COUNT==704U), "score qualification scope");
static_assert(sizeof(T)==sizeof(half) && (N==32768U || N==32769U) && SPAN%64U==0U && COUNT<=SPAN && COUNT>SPAN-64U && SPAN<=704U, "qualified paired span");
static constexpr uint32_t FRAME=2U*SPAN, STRIDE=FRAME+16U, Q=SPAN/4U;
uint32_t used=0;
AscendC::GlobalTensor<T> rawGm,outGm;
AscendC::GlobalTensor<float> priorGm;
__aicore__ inline AscendC::LocalTensor<T> Raw(uint32_t slot) { return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN,slot*19968U,9984U); }
__aicore__ inline AscendC::LocalTensor<T> Out(uint32_t slot) { return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT,39936U+slot*16640U,8320U); }
__aicore__ inline AscendC::LocalTensor<float> Prior() { return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN,39936U,4160U); }
__aicore__ inline AscendC::LocalTensor<float> Work() { return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC,73216U,18496U); }
__aicore__ inline AscendC::LocalTensor<float> Ones() { return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC,147200U,8U); }
__aicore__ inline AscendC::LocalTensor<float> Twos() { return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC,147232U,8U); }
__aicore__ inline AscendC::LocalTensor<T> HalfRF() { return AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC,147456U,10080U); }
__aicore__ inline AscendC::LocalTensor<float> PackedFloat() { return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC,147456U,8320U); }
__aicore__ inline AscendC::LocalTensor<uint64_t> VA() { return AscendC::LocalTensor<uint64_t>(AscendC::TPosition::VECCALC,187232U,132U); }
__aicore__ inline AscendC::LocalTensor<T> AllocRaw(uint32_t slot) {
if(used&(1U<<slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
return Raw(slot);
}
__aicore__ inline void ReleaseRaw(uint32_t slot) { AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);used|=1U<<slot; }
__aicore__ inline void AcquireOutput(uint32_t slot) { if(used&(1U<<(slot+2U))) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot); }
__aicore__ inline void PublishOutput(uint32_t slot) { AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot);used|=1U<<(slot+2U); }
__aicore__ inline void Drain() {
for(uint32_t slot=0;slot<2;++slot) {
if(used&(1U<<slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
if(used&(1U<<(slot+2U))) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
}
}
__aicore__ inline void StorePacked(uint32_t offset,const uint64_t *a) {
auto dst=VA()[offset];
#pragma unroll
for(uint32_t q=0;q<4;++q) dst.SetValue(q,((a[4*q]>>5)&0x1fffULL)|(((a[4*q+1]>>5)&0x1fffULL)<<16)|(((a[4*q+2]>>5)&0x1fffULL)<<32)|(((a[4*q+3]>>5)&0x1fffULL)<<48));
}
template<uint32_t PHASE>
__aicore__ inline void InitInput() {
uint64_t target[16],source[16];
const uint64_t h=reinterpret_cast<uint64_t>(HalfRF().GetPhyAddr());
#pragma unroll
for(uint32_t k=0;k<16;++k) target[k]=h+(((16U*PHASE+k)%6U)*STRIDE+((16U*PHASE+k)/6U)*Q)*2U;
StorePacked(80U+4U*PHASE,target);
#pragma unroll
for(uint32_t slot=0;slot<2;++slot) {
const uint64_t r=reinterpret_cast<uint64_t>(Raw(slot).GetPhyAddr());
#pragma unroll
for(uint32_t i=0;i<16;++i) source[i]=r+(48U*i+16U*PHASE)*2U;
if constexpr(PHASE==0U) StorePacked(4U*slot,source);
}
}
template<uint32_t PHASE>
__aicore__ inline void InitOutput() {
uint64_t target[16],source[16];
const uint64_t w=reinterpret_cast<uint64_t>(Work().GetPhyAddr());
const uint64_t r=reinterpret_cast<uint64_t>(PackedFloat().GetPhyAddr());
#pragma unroll
for(uint32_t i=0;i<16;++i) source[i]=w+(((8U*PHASE+i%8U)%5U)*STRIDE+((8U*PHASE+i%8U)/5U)*Q+(i/8U)*8U)*4U;
#pragma unroll
for(uint32_t k=0;k<16;++k) target[k]=r+(5U*(k/2U+8U*(k%2U))+PHASE)*32U;
StorePacked(92U+4U*PHASE,source);
if constexpr(PHASE==0U) StorePacked(8U,target);
}
template<uint32_t PHASE>
__aicore__ inline void InitPrior() {
uint64_t target[16],source[16];
const uint64_t p=reinterpret_cast<uint64_t>(Prior().GetPhyAddr());
const uint64_t w=reinterpret_cast<uint64_t>(Work().GetPhyAddr());
#pragma unroll
for(uint32_t i=0;i<16;++i) source[i]=p+(40U*(i%8U)+8U*PHASE)*4U;
#pragma unroll
for(uint32_t k=0;k<8;++k) {
target[2*k]=w+((6U+(8U*PHASE+k)%5U)*STRIDE+((8U*PHASE+k)/5U)*Q)*4U;
target[2*k+1]=target[2*k]+(SPAN/8U)*4U;
}
StorePacked(112U+4U*PHASE,target);
if constexpr(PHASE==0U) StorePacked(12U,source);
}
template<bool B16,uint8_t REPEAT,uint16_t DST,uint16_t SRC>
__aicore__ inline void Transpose(uint32_t dst,uint32_t src) {
auto *d=reinterpret_cast<__ubuf__ uint64_t *>(VA()[dst].GetPhyAddr());
auto *s=reinterpret_cast<__ubuf__ uint64_t *>(VA()[src].GetPhyAddr());
__cce_scalar::vld_va_reg(VA0,d,L128);__cce_scalar::vld_va_reg(VA1,d,H128);
__cce_scalar::vld_va_reg(VA2,s,L128);__cce_scalar::vld_va_reg(VA3,s,H128);
if constexpr(B16) __cce_scalar::scatter_vnchwconv_b16(VA0,VA2,REPEAT,DST,SRC);
else __cce_scalar::scatter_vnchwconv_b32(VA0,VA2,REPEAT,DST,SRC);
}
__aicore__ inline void InitVA() {
InitInput<0>();InitInput<1>();InitInput<2>();
InitOutput<0>();InitOutput<1>();InitOutput<2>();InitOutput<3>();InitOutput<4>();
InitPrior<0>();InitPrior<1>();InitPrior<2>();InitPrior<3>();InitPrior<4>();
AscendC::SetFlag<AscendC::HardEvent::S_V>(0);AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
auto *entries=reinterpret_cast<__ubuf__ int16_t *>(VA().GetPhyAddr());
AscendC::SetMaskCount();AscendC::SetVectorMask<int16_t,AscendC::MaskMode::COUNTER>(64U);
#pragma unroll
for(uint32_t phase=1U;phase<5U;++phase)
asc_add_scalar(entries+phase*64U,entries,int16_t(phase),uint8_t(1),uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();AscendC::SetMaskNorm();AscendC::ResetMask();
}
template<uint32_t SLOT>
__aicore__ inline void Input() {
Transpose<true,SPAN/64U,1,48>(80,4*SLOT);Transpose<true,SPAN/64U,1,48>(84,16+4*SLOT);Transpose<true,SPAN/64U,1,48>(88,32+4*SLOT);
}
__aicore__ inline void Output() {
Transpose<false,SPAN/64U,80,2>(8,92);Transpose<false,SPAN/64U,80,2>(24,96);Transpose<false,SPAN/64U,80,2>(40,100);
Transpose<false,SPAN/64U,80,2>(56,104);Transpose<false,SPAN/64U,80,2>(72,108);
}
public:
__aicore__ inline void Run(GM_ADDR raw,GM_ADDR priors,GM_ADDR decoded) {
const uint32_t block=static_cast<uint32_t>(AscendC::GetBlockIdx());
if(block>=47U)return;
const uint32_t begin=block*704U,cohort=block&3U;
rawGm.SetGlobalBuffer((__gm__ T *)raw);priorGm.SetGlobalBuffer((__gm__ float *)priors);outGm.SetGlobalBuffer((__gm__ T *)decoded);
const AscendC::DataCopyExtParams rp{2,COUNT*12U,(N-COUNT)*12U,(SPAN*12U-((COUNT*12U+31U)/32U)*32U)/32U,0};
const AscendC::DataCopyExtParams pp{1,COUNT*20U,0,0,0};
auto firstRaw=AllocRaw(0);
AscendC::DataCopyPad(firstRaw,rawGm[(2U*cohort*N+begin)*6U],rp,AscendC::DataCopyPadExtParams<T>{false,0,0,0});
AscendC::DataCopyPad(Prior(),priorGm[begin*5U],pp,AscendC::DataCopyPadExtParams<float>{false,0,0,0});
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
AscendC::Duplicate(Ones(),1.0f,8U);AscendC::Duplicate(Twos(),2.0f,8U);
AscendC::LocalTensor<float> f[11];
for(uint32_t k=0;k<11;++k)f[k]=Work()[k*STRIDE];
InitVA();AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
Transpose<false,SPAN/64U,1,40>(112,12);Transpose<false,SPAN/64U,1,40>(116,28);Transpose<false,SPAN/64U,1,40>(120,44);
Transpose<false,SPAN/64U,1,40>(124,60);Transpose<false,SPAN/64U,1,40>(128,76);
AscendC::PipeBarrier<PIPE_V>();
for(uint32_t iteration=0;iteration<4;++iteration) {
const uint32_t pair=(iteration+cohort)&3U,slot=iteration&1U,pos=2U*pair*N+begin;
if(iteration!=0) {
auto r=AllocRaw(slot);
AscendC::DataCopyPad(r,rawGm[pos*6U],rp,AscendC::DataCopyPadExtParams<T>{false,0,0,0});
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot);
}
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);
auto signedRaw=Raw(slot).template ReinterpretCast<int16_t>();
AscendC::Adds(signedRaw,signedRaw,int16_t(-32768),FRAME*6U);
AscendC::PipeBarrier<PIPE_V>();
if(slot==0)Input<0>();else Input<1>();
AscendC::PipeBarrier<PIPE_V>();ReleaseRaw(slot);
AscendC::Cast(Work(),HalfRF(),AscendC::RoundMode::CAST_NONE,6U*STRIDE);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Exp(Work(),Work(),6U*STRIDE);AscendC::Adds(Work(),Work(),1.0f,6U*STRIDE);
AscendC::Mul(f[4],f[4],f[5],FRAME);
const AscendC::BinaryRepeatParams divBinary{1,0,1,8,0,8};
AscendC::Div<float>(Work(),Twos(),Work(),uint64_t(64),uint8_t(STRIDE/16U),divBinary);
if constexpr (RAW_SCORE) {
AscendC::Reciprocal<float>(f[4],f[4],uint64_t(64),uint8_t(FRAME/64U),AscendC::UnaryRepeatParams{1,1,8,8});
} else {
AscendC::Div<float>(f[4],Ones(),f[4],uint64_t(64),uint8_t(FRAME/64U),divBinary);
}
AscendC::PipeBarrier<PIPE_V>();
AscendC::Adds(f[0],f[0],-0.5f,2U*STRIDE);AscendC::Add(f[0],f[0],f[6],2U*STRIDE);
AscendC::Mul(f[2],f[2],f[2],2U*STRIDE);AscendC::Mul(f[2],f[2],f[9],2U*STRIDE);
AscendC::PipeBarrier<PIPE_V>();
auto *w=reinterpret_cast<__ubuf__ float *>(Work().GetPhyAddr());
AscendC::SetMaskCount();AscendC::SetVectorMask<float,AscendC::MaskMode::COUNTER>(STRIDE);
#pragma unroll
for(uint32_t k=0;k<4;++k)asc_mul(w+k*STRIDE,w+k*STRIDE,w+8*STRIDE,uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(8),uint8_t(8),uint8_t(8));
AscendC::SetMaskNorm();AscendC::ResetMask();AscendC::PipeBarrier<PIPE_V>();
Output();AscendC::PipeBarrier<PIPE_V>();AcquireOutput(slot);
asc_float2half_rn(reinterpret_cast<__ubuf__ half *>(Out(slot).GetPhyAddr()),reinterpret_cast<__ubuf__ float *>(PackedFloat().GetPhyAddr()),uint32_t(FRAME*5U));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot);AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
const AscendC::DataCopyExtParams op{2,COUNT*10U,(SPAN*10U-((COUNT*10U+31U)/32U)*32U)/32U,(N-COUNT)*10U,0};
AscendC::DataCopyPad(outGm[pos*5U],Out(slot),op);PublishOutput(slot);
}
Drain();
}
};

template <typename T, uint32_t TILE, bool VA_GROUP28 = false>
class FireA2RowF4Storage {
static_assert(TILE >= 128 && TILE <= 1024 && TILE % 128 == 0, "native hardware frames or large reuse frame");
static constexpr uint32_t FRAME = TILE;
static constexpr uint32_t DEPTH = TILE > 256 ? 2 : 1;
static constexpr uint32_t STRIDE = FRAME + 16;
static constexpr uint32_t RAW_BYTES = (sizeof(T) == sizeof(float) ? FRAME : TILE) * 6 * sizeof(T);
static constexpr uint32_t PRIOR_WORDS = FRAME * 5;
static constexpr uint32_t PRIOR_BYTES = PRIOR_WORDS * sizeof(float);
static constexpr uint32_t OUT_BYTES = (sizeof(T) == sizeof(float) ? FRAME : TILE) * 5 * sizeof(T);
static constexpr uint32_t WORK_BYTES = ((STRIDE * 11 * sizeof(float) + 511) / 512) * 512 + 256;
static constexpr uint32_t ONE_WORDS = FRAME * 6;
static constexpr uint32_t ONE_BYTES = ONE_WORDS * sizeof(float) + (sizeof(T) == sizeof(half) ? 256 : 0);
static constexpr uint32_t RAW_BASE = 0;
static constexpr uint32_t PRIOR_BASE = RAW_BASE + DEPTH * RAW_BYTES;
static constexpr uint32_t OUT_BASE = PRIOR_BASE + PRIOR_BYTES;
static constexpr uint32_t WORK_BASE = OUT_BASE + DEPTH * OUT_BYTES;
static constexpr uint32_t ONE_BASE = WORK_BASE + WORK_BYTES;
static constexpr uint32_t RF_BASE = ONE_BASE + ONE_BYTES;
static constexpr uint32_t PACK_BASE = RF_BASE + (sizeof(T) == sizeof(half) ? FRAME * 6 * sizeof(float) : 0);
static constexpr uint32_t INDEX_BASE = PACK_BASE + (sizeof(T) == sizeof(half) ? FRAME * 5 * sizeof(float) + 256 : 0);
static constexpr uint32_t VA_BASE = INDEX_BASE + FRAME * sizeof(uint32_t);
static constexpr uint32_t VA_WORDS = VA_GROUP28 ? 112U : 172U;
static_assert(!VA_GROUP28 || (sizeof(T) == sizeof(float) && TILE == 896U && VA_BASE == 162560U && VA_BASE + VA_WORDS * sizeof(uint64_t) == 163456U), "qualified FP32 M20 VA28 fixed UB");
static_assert(VA_BASE + VA_WORDS * sizeof(uint64_t) < 188416, "reserve the SDK temporary UB region");
static_assert((RAW_BYTES | PRIOR_BYTES | OUT_BYTES | WORK_BYTES | ONE_BYTES | INDEX_BASE) % 32 == 0,
"all static addresses and capacities retain 32-byte alignment");
uint32_t rawCursor = 0, priorCursor = 0, outCursor = 0;
uint32_t rawSlot = 0, priorSlot = 0, outSlot = 0;
uint32_t rawUsed = 0, priorUsed = 0, outUsed = 0;
public:
__aicore__ inline AscendC::LocalTensor<uint64_t> PackedVA() {
return AscendC::LocalTensor<uint64_t>(AscendC::TPosition::VECCALC, VA_BASE, VA_WORDS);
}
__aicore__ inline AscendC::LocalTensor<int32_t> Indices() {
return AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, INDEX_BASE, FRAME);
}
__aicore__ inline AscendC::LocalTensor<float> Work() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, WORK_BASE, WORK_BYTES / sizeof(float));
}
__aicore__ inline AscendC::LocalTensor<float> Ones() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE, ONE_BYTES / sizeof(float));
}
__aicore__ inline AscendC::LocalTensor<float> RawFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, RF_BASE, FRAME * 6);
}
__aicore__ inline AscendC::LocalTensor<float> PackedFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, PACK_BASE, FRAME * 5);
}
__aicore__ inline AscendC::LocalTensor<T> AllocRaw() {
rawSlot = rawCursor;
rawCursor = (rawCursor + 1) % DEPTH;
if (rawUsed & (1U << rawSlot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(rawSlot);
return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, RAW_BASE + rawSlot * RAW_BYTES, (sizeof(T) == sizeof(float) ? FRAME : TILE) * 6);
}
__aicore__ inline AscendC::LocalTensor<float> AllocPrior() {
priorSlot = priorCursor;
priorCursor = 0;
if (priorUsed & (1U << priorSlot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2 + priorSlot);
return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE + priorSlot * PRIOR_BYTES, PRIOR_WORDS);
}
__aicore__ inline AscendC::LocalTensor<T> AllocOut() {
outSlot = outCursor;
outCursor = (outCursor + 1) % DEPTH;
if (outUsed & (1U << outSlot)) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outSlot);
return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, OUT_BASE + outSlot * OUT_BYTES, (sizeof(T) == sizeof(float) ? FRAME : TILE) * 5);
}
__aicore__ inline AscendC::LocalTensor<T> RawFixedSlot(uint32_t slot) {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, RAW_BASE + slot * RAW_BYTES,
(sizeof(T) == sizeof(float) ? FRAME : TILE) * 6);
}
__aicore__ inline AscendC::LocalTensor<float> PriorFixedSlot(uint32_t slot) {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE + slot * PRIOR_BYTES, PRIOR_WORDS);
}
__aicore__ inline AscendC::LocalTensor<T> OutAt(uint32_t slot) {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, OUT_BASE + slot * OUT_BYTES,
(sizeof(T) == sizeof(float) ? FRAME : TILE) * 5);
}
__aicore__ inline uint32_t CurrentOutSlot() const { return outSlot; }
__aicore__ inline void RawReady() { AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(rawSlot); }
__aicore__ inline void RawWait() { AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(rawSlot); }
__aicore__ inline void PriorReady() { AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2 + priorSlot); }
__aicore__ inline void PriorWait() { AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2 + priorSlot); }
__aicore__ inline void OutReady() { AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outSlot); }
__aicore__ inline void OutWait() { AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outSlot); }
__aicore__ inline void FreeRaw() {
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(rawSlot);
rawUsed |= 1U << rawSlot;
}
__aicore__ inline void FreePrior() {
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2 + priorSlot);
priorUsed |= 1U << priorSlot;
}
__aicore__ inline void FreeOut() {
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outSlot);
outUsed |= 1U << outSlot;
}
__aicore__ inline void Drain() {
for (uint32_t slot = 0; slot < DEPTH; ++slot) {
if (rawUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
if (priorUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2 + slot);
if (outUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
}
}
};


template <typename T, uint32_t TILE, bool ROW, bool RAW3 = false, bool DIRECT_STRIDE = false, uint32_t QUALIFIED_ROW_N = 0U, bool VA_GROUP28 = false, bool ROW_CACHED_STRIDE = false>
class KernelFireA2RowF4 {
static constexpr uint32_t DEPTH = TILE > 256 ? 2 : 1;
static constexpr uint32_t FRAME = TILE;
static constexpr uint32_t STRIDE = FRAME + 16;
static constexpr uint32_t Q = FRAME / 8;
static_assert(QUALIFIED_ROW_N == 0U || (QUALIFIED_ROW_N == 4097U && sizeof(T) == sizeof(float) && FRAME == 896U && ROW && RAW3 && !DIRECT_STRIDE), "qualified FP32 P4 C20 ROW896 N4097");
static_assert(!VA_GROUP28 || (sizeof(T) == sizeof(float) && FRAME == 896U && ROW && RAW3 && !DIRECT_STRIDE && QUALIFIED_ROW_N == 4097U), "qualified FP32 P4 M20 ROW896 VA28");
static_assert(!ROW_CACHED_STRIDE || (VA_GROUP28 && sizeof(T) == sizeof(float) && FRAME == 896U && STRIDE == 912U && ROW && RAW3 && !DIRECT_STRIDE && QUALIFIED_ROW_N == 4097U), "qualified FP32 P4 cached stride ROW896");
FireA2RowF4Storage<T, TILE, VA_GROUP28> mem;
template <uint32_t OFFSET>
__aicore__ static constexpr uint32_t VAOffset() {
if constexpr (VA_GROUP28) {
static_assert(OFFSET % 4U == 0U && OFFSET < 172U, "VA28 original group bounds");
constexpr uint32_t group = OFFSET / 4U;
static_assert(group < 3U || (group >= 6U && group < 9U) || (group >= 18U && group < 28U) || (group >= 33U && group < 43U), "VA28 original active group");
if constexpr (group < 3U) return 4U * (15U + group);
else if constexpr (group < 9U) return 4U * (3U * (group - 6U));
else if constexpr (group < 23U) return 4U * group;
else if constexpr (group < 28U) return 4U * (1U + 3U * (group - 23U));
else if constexpr (group < 38U) return 4U * (23U + group - 33U);
else return 4U * (2U + 3U * (group - 38U));
} else return OFFSET;
}
AscendC::GlobalTensor<T> rawGm, outGm;
AscendC::GlobalTensor<float> priorGm;
__aicore__ inline void StorePacked(AscendC::LocalTensor<uint64_t> dst, const uint64_t *address) {
#pragma unroll
for (uint32_t q = 0; q < 4; ++q) {
const uint64_t value = ((address[4 * q] >> 5) & 0x1fffULL) |
(((address[4 * q + 1] >> 5) & 0x1fffULL) << 16) |
(((address[4 * q + 2] >> 5) & 0x1fffULL) << 32) |
(((address[4 * q + 3] >> 5) & 0x1fffULL) << 48);
dst.SetValue(q, value);
}
}
template <uint8_t REPEAT, uint16_t DST_STRIDE, uint16_t SRC_STRIDE>
__aicore__ inline void PackedTranspose(AscendC::LocalTensor<uint64_t> dst,
AscendC::LocalTensor<uint64_t> src) {
auto *dstPacked = reinterpret_cast<__ubuf__ uint64_t *>(dst.GetPhyAddr());
auto *srcPacked = reinterpret_cast<__ubuf__ uint64_t *>(src.GetPhyAddr());
__cce_scalar::vld_va_reg(VA0, dstPacked, L128);
__cce_scalar::vld_va_reg(VA1, dstPacked, H128);
__cce_scalar::vld_va_reg(VA2, srcPacked, L128);
__cce_scalar::vld_va_reg(VA3, srcPacked, H128);
__cce_scalar::scatter_vnchwconv_b32(VA0, VA2, REPEAT,
REPEAT == 1 ? 0 : DST_STRIDE, REPEAT == 1 ? 0 : SRC_STRIDE);
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitInputPhase() {
static_assert(PHASE < 6 && SLOT < (sizeof(T) == sizeof(float) ? 2 : 1),
"valid padded input specialization");
static_assert(!RAW3 || ((((FRAME == 896 || FRAME == 768) && !ROW) || ((FRAME == 512 || FRAME == 640) && ROW) || (QUALIFIED_ROW_N == 4097U && FRAME == 896U && ROW && sizeof(T) == sizeof(float))) && PHASE < 3), "raw3 verified reuse or qualified C20 ROW896");
uint64_t sourceBase;
if constexpr (sizeof(T) == sizeof(float))
sourceBase = reinterpret_cast<uint64_t>(mem.RawFixedSlot(SLOT).GetPhyAddr());
else sourceBase = reinterpret_cast<uint64_t>(mem.RawFloat().GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
if constexpr (RAW3) {
uint64_t source[16] = {
sourceBase + uint64_t(0 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(48 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(96 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(144 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(192 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(240 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(288 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(336 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(24 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(72 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(120 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(168 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(216 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(264 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(312 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(360 + 8 * PHASE) * sizeof(float)
};
uint64_t target[16] = {
workBase + uint64_t(((8 * PHASE + 0) % 6) * STRIDE + (((8 * PHASE + 0) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 0) % 6) * STRIDE + (((8 * PHASE + 0) / 6) + 4) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 1) % 6) * STRIDE + (((8 * PHASE + 1) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 1) % 6) * STRIDE + (((8 * PHASE + 1) / 6) + 4) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 2) % 6) * STRIDE + (((8 * PHASE + 2) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 2) % 6) * STRIDE + (((8 * PHASE + 2) / 6) + 4) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 3) % 6) * STRIDE + (((8 * PHASE + 3) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 3) % 6) * STRIDE + (((8 * PHASE + 3) / 6) + 4) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 4) % 6) * STRIDE + (((8 * PHASE + 4) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 4) % 6) * STRIDE + (((8 * PHASE + 4) / 6) + 4) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 5) % 6) * STRIDE + (((8 * PHASE + 5) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 5) % 6) * STRIDE + (((8 * PHASE + 5) / 6) + 4) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 6) % 6) * STRIDE + (((8 * PHASE + 6) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 6) % 6) * STRIDE + (((8 * PHASE + 6) / 6) + 4) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 7) % 6) * STRIDE + (((8 * PHASE + 7) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 7) % 6) * STRIDE + (((8 * PHASE + 7) / 6) + 4) * Q) * sizeof(float)
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[VAOffset<4 * PHASE>()], target);
if constexpr (!VA_GROUP28 || PHASE == 0U)
StorePacked(mem.PackedVA()[VAOffset<24 + SLOT * 24 + 4 * PHASE>()], source);
} else {
uint64_t source[16] = {
sourceBase + uint64_t(0 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(48 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(96 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(144 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(192 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(240 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(288 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(336 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(384 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(432 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(480 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(528 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(576 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(624 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(672 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(720 + 8 * PHASE) * sizeof(float)
};
uint64_t target[16] = {
workBase + uint64_t(((8 * PHASE + 0) % 6) * STRIDE + ((8 * PHASE + 0) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 0) % 6) * STRIDE + ((8 * PHASE + 0) / 6) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 1) % 6) * STRIDE + ((8 * PHASE + 1) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 1) % 6) * STRIDE + ((8 * PHASE + 1) / 6) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 2) % 6) * STRIDE + ((8 * PHASE + 2) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 2) % 6) * STRIDE + ((8 * PHASE + 2) / 6) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 3) % 6) * STRIDE + ((8 * PHASE + 3) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 3) % 6) * STRIDE + ((8 * PHASE + 3) / 6) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 4) % 6) * STRIDE + ((8 * PHASE + 4) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 4) % 6) * STRIDE + ((8 * PHASE + 4) / 6) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 5) % 6) * STRIDE + ((8 * PHASE + 5) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 5) % 6) * STRIDE + ((8 * PHASE + 5) / 6) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 6) % 6) * STRIDE + ((8 * PHASE + 6) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 6) % 6) * STRIDE + ((8 * PHASE + 6) / 6) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 7) % 6) * STRIDE + ((8 * PHASE + 7) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 7) % 6) * STRIDE + ((8 * PHASE + 7) / 6) * Q + 8) * sizeof(float)
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[VAOffset<4 * PHASE>()], target);
if constexpr (!VA_GROUP28 || PHASE == 0U)
StorePacked(mem.PackedVA()[VAOffset<24 + SLOT * 24 + 4 * PHASE>()], source);
}
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitOutputPhase() {
static_assert(PHASE < 5 && SLOT < (sizeof(T) == sizeof(float) ? 2 : 1),
"valid padded output specialization");
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t outputBase;
if constexpr (sizeof(T) == sizeof(float))
outputBase = reinterpret_cast<uint64_t>(mem.OutAt(SLOT).GetPhyAddr());
else outputBase = reinterpret_cast<uint64_t>(mem.PackedFloat().GetPhyAddr());
uint64_t source[16] = {
workBase + uint64_t(((8 * PHASE + 0) % 5) * STRIDE + ((8 * PHASE + 0) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 1) % 5) * STRIDE + ((8 * PHASE + 1) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 2) % 5) * STRIDE + ((8 * PHASE + 2) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 3) % 5) * STRIDE + ((8 * PHASE + 3) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 4) % 5) * STRIDE + ((8 * PHASE + 4) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 5) % 5) * STRIDE + ((8 * PHASE + 5) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 6) % 5) * STRIDE + ((8 * PHASE + 6) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 7) % 5) * STRIDE + ((8 * PHASE + 7) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 0) % 5) * STRIDE + ((8 * PHASE + 0) / 5) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 1) % 5) * STRIDE + ((8 * PHASE + 1) / 5) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 2) % 5) * STRIDE + ((8 * PHASE + 2) / 5) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 3) % 5) * STRIDE + ((8 * PHASE + 3) / 5) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 4) % 5) * STRIDE + ((8 * PHASE + 4) / 5) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 5) % 5) * STRIDE + ((8 * PHASE + 5) / 5) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 6) % 5) * STRIDE + ((8 * PHASE + 6) / 5) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 7) % 5) * STRIDE + ((8 * PHASE + 7) / 5) * Q + 8) * sizeof(float)
};
uint64_t target[16] = {
outputBase + uint64_t((5 * 0 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 8 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 1 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 9 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 2 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 10 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 3 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 11 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 4 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 12 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 5 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 13 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 6 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 14 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 7 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 15 + PHASE) * 8) * sizeof(float)
};
constexpr uint32_t INPUT_WORDS = 72;
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[VAOffset<INPUT_WORDS + 4 * PHASE>()], source);
if constexpr (!VA_GROUP28 || PHASE == 0U)
StorePacked(mem.PackedVA()[VAOffset<INPUT_WORDS + 20 + SLOT * 20 + 4 * PHASE>()], target);
}
template <uint32_t PHASE>
__aicore__ inline void InitPriorPhase() {
static_assert(PHASE < 5, "native prior hardware phase");
constexpr uint32_t PRIOR_TARGET = 132;
constexpr uint32_t PRIOR_SOURCE = 152 + 4 * PHASE;
const uint64_t sourceBase = reinterpret_cast<uint64_t>(mem.PriorFixedSlot(0).GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t source[16] = {
sourceBase + uint64_t(0 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(40 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(80 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(120 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(160 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(200 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(240 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(280 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(320 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(360 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(400 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(440 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(480 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(520 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(560 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(600 + 8 * PHASE) * sizeof(float)
};
uint64_t target[16] = {
workBase + uint64_t((6 + (8 * PHASE + 0) % 5) * STRIDE + ((8 * PHASE + 0) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 0) % 5) * STRIDE + ((8 * PHASE + 0) / 5) * Q + 1 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 1) % 5) * STRIDE + ((8 * PHASE + 1) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 1) % 5) * STRIDE + ((8 * PHASE + 1) / 5) * Q + 1 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 2) % 5) * STRIDE + ((8 * PHASE + 2) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 2) % 5) * STRIDE + ((8 * PHASE + 2) / 5) * Q + 1 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 3) % 5) * STRIDE + ((8 * PHASE + 3) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 3) % 5) * STRIDE + ((8 * PHASE + 3) / 5) * Q + 1 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 4) % 5) * STRIDE + ((8 * PHASE + 4) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 4) % 5) * STRIDE + ((8 * PHASE + 4) / 5) * Q + 1 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 5) % 5) * STRIDE + ((8 * PHASE + 5) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 5) % 5) * STRIDE + ((8 * PHASE + 5) / 5) * Q + 1 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 6) % 5) * STRIDE + ((8 * PHASE + 6) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 6) % 5) * STRIDE + ((8 * PHASE + 6) / 5) * Q + 1 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 7) % 5) * STRIDE + ((8 * PHASE + 7) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 7) % 5) * STRIDE + ((8 * PHASE + 7) / 5) * Q + 1 * 8) * sizeof(float)
};
StorePacked(mem.PackedVA()[VAOffset<PRIOR_TARGET + 4 * PHASE>()], target);
if constexpr (!VA_GROUP28 || PHASE == 0U)
StorePacked(mem.PackedVA()[VAOffset<PRIOR_SOURCE>()], source);
}
template <uint32_t PHASE>
__aicore__ inline void NativePriorPhase() {
constexpr uint32_t PRIOR_TARGET = 132;
constexpr uint32_t PRIOR_SOURCE = 152 + 4 * PHASE;
PackedTranspose<FRAME / 128, 2, 80>(mem.PackedVA()[VAOffset<PRIOR_TARGET + 4 * PHASE>()],
mem.PackedVA()[VAOffset<PRIOR_SOURCE>()]);
}
__aicore__ inline void InitPackedVA() {
InitInputPhase<0, 0>();
InitInputPhase<0, 1>();
InitInputPhase<0, 2>();
if constexpr (!RAW3) {
InitInputPhase<0, 3>();
InitInputPhase<0, 4>();
InitInputPhase<0, 5>();
}
if constexpr (sizeof(T) == sizeof(float) && !(ROW)) {
InitInputPhase<1, 0>();
InitInputPhase<1, 1>();
InitInputPhase<1, 2>();
if constexpr (!RAW3) {
InitInputPhase<1, 3>();
InitInputPhase<1, 4>();
InitInputPhase<1, 5>();
}
}
InitOutputPhase<0, 0>();
InitOutputPhase<0, 1>();
InitOutputPhase<0, 2>();
InitOutputPhase<0, 3>();
InitOutputPhase<0, 4>();
if constexpr (sizeof(T) == sizeof(float) && !(ROW)) {
InitOutputPhase<1, 0>();
InitOutputPhase<1, 1>();
InitOutputPhase<1, 2>();
InitOutputPhase<1, 3>();
InitOutputPhase<1, 4>();
}
{
InitPriorPhase<0>();
InitPriorPhase<1>();
InitPriorPhase<2>();
InitPriorPhase<3>();
InitPriorPhase<4>();
}
AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
if constexpr (VA_GROUP28) {
auto *entries = reinterpret_cast<__ubuf__ int16_t *>(mem.PackedVA().GetPhyAddr());
AscendC::SetMaskCount();
AscendC::SetVectorMask<int16_t, AscendC::MaskMode::COUNTER>(48U);
#pragma unroll
for (uint32_t phase = 1U; phase < 5U; ++phase)
asc_add_scalar(entries + phase * 48U, entries, int16_t(phase),
uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetMaskNorm();
AscendC::ResetMask();
}
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void NativeInputPhase() {
static_assert(PHASE < 6 && SLOT < (sizeof(T) == sizeof(float) ? 2 : 1), "valid packed input phase");
if constexpr (RAW3) {
PackedTranspose<FRAME / 64, 1, 48>(mem.PackedVA()[VAOffset<4 * PHASE>()],
mem.PackedVA()[VAOffset<24 + SLOT * 24 + 4 * PHASE>()]);
} else {
PackedTranspose<FRAME / 128, 2, 96>(mem.PackedVA()[VAOffset<4 * PHASE>()],
mem.PackedVA()[VAOffset<24 + SLOT * 24 + 4 * PHASE>()]);
}
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void NativeOutputPhase() {
static_assert(PHASE < 5 && SLOT < (sizeof(T) == sizeof(float) ? 2 : 1), "valid packed output phase");
constexpr uint32_t INPUT_WORDS = 72;
PackedTranspose<FRAME / 128, 80, 2>(mem.PackedVA()[VAOffset<INPUT_WORDS + 20 + SLOT * 20 + 4 * PHASE>()],
mem.PackedVA()[VAOffset<INPUT_WORDS + 4 * PHASE>()]);
}
public:
template <bool WHOLE = false>
__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded,
const FireFrameTilingData &schedule) {
static_assert(QUALIFIED_ROW_N == 0U || !WHOLE, "qualified C20 nonwhole only");
static_assert(!WHOLE || ROW, "whole entry requires one batch per core");
static_assert(!DIRECT_STRIDE || (FRAME == 512 && ROW && WHOLE), "direct stride only original whole512");
constexpr bool rawFirst = WHOLE && DIRECT_STRIDE && RAW3 && FRAME == 512 && sizeof(T) == sizeof(half);
constexpr bool rawFirstRow = ROW_CACHED_STRIDE && QUALIFIED_ROW_N == 4097U;
static_assert(!rawFirstRow || (!WHOLE && ROW && RAW3 && !DIRECT_STRIDE && VA_GROUP28 && FRAME == 896U && STRIDE == 912U && sizeof(T) == sizeof(float)), "qualified F4 C20 raw before VA only");
const auto &t = schedule.header;
rawGm.SetGlobalBuffer((__gm__ T *)raw);
priorGm.SetGlobalBuffer((__gm__ float *)priors);
outGm.SetGlobalBuffer((__gm__ T *)decoded);
const uint32_t groupLanes = schedule.geometry & 0xffffU;
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
const uint32_t begin = (block % groupLanes) * t.perCore;
if (begin >= t.n) return;
const uint32_t batchCount = ROW ? 1U : schedule.geometry >> 16;
const uint32_t batchFirst = (block / groupLanes) * batchCount;
const uint32_t priorPos = begin;
const uint32_t count = WHOLE ? FRAME : (t.n - priorPos < t.perCore ? t.n - priorPos : t.perCore);
const bool one8Full = ROW_CACHED_STRIDE && QUALIFIED_ROW_N == 4097U && count == 832U;
auto firstRaw = mem.AllocRaw();
auto p = mem.AllocPrior();
AscendC::DataCopyExtParams firstRp{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyExtParams firstPp{1, count * 5 * uint32_t(sizeof(float)), 0, 0, 0};
AscendC::DataCopyPad(firstRaw, rawGm[(WHOLE ? block * FRAME : batchFirst * t.n + priorPos) * 6], firstRp,
AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
if constexpr (rawFirst || rawFirstRow) mem.RawReady();
AscendC::DataCopyPad(p, priorGm[priorPos * 5], firstPp,
AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
if constexpr (!(rawFirst || rawFirstRow)) mem.RawReady();
mem.PriorReady();
auto one = mem.Ones();
AscendC::Duplicate(one, 1.0f, one8Full ? 8U : FRAME * 6);
AscendC::LocalTensor<float> f[11];
for (uint32_t k = 0; k < 11; ++k) f[k] = mem.Work()[k * STRIDE];
if constexpr (rawFirst) {
mem.RawWait();
auto rf = mem.RawFloat();
AscendC::Cast(rf, firstRaw, AscendC::RoundMode::CAST_NONE, count * 6);
AscendC::Muls(rf, rf, -1.0f, count * 6);
AscendC::Exp(rf, rf, count * 6);
AscendC::Adds(rf, rf, 1.0f, count * 6);
AscendC::Div(rf, one, rf, count * 6);
AscendC::PipeBarrier<PIPE_V>();
}
if constexpr (rawFirstRow) {
mem.RawWait();
auto rf = firstRaw.template ReinterpretCast<float>();
AscendC::Muls(rf, rf, -1.0f, count * 6);
AscendC::Exp(rf, rf, count * 6);
AscendC::Adds(rf, rf, 1.0f, count * 6);
if (one8Full) {
const AscendC::BinaryRepeatParams divOne8{1, 0, 1, 8, 0, 8};
AscendC::Div<float>(rf, one, rf, uint64_t(64), uint8_t(78), divOne8);
} else AscendC::Div(rf, one, rf, count * 6);
AscendC::PipeBarrier<PIPE_V>();
}
InitPackedVA();
{
mem.PriorWait();
NativePriorPhase<0>();
NativePriorPhase<1>();
NativePriorPhase<2>();
NativePriorPhase<3>();
NativePriorPhase<4>();
AscendC::PipeBarrier<PIPE_V>();
if constexpr (!DIRECT_STRIDE && !ROW_CACHED_STRIDE) {
AscendC::DataCopy(p, f[8], STRIDE);
AscendC::DataCopy(p[STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[2 * STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[3 * STRIDE], f[8], STRIDE);
AscendC::PipeBarrier<PIPE_V>();
}
for (uint32_t batch = batchFirst; batch < batchFirst + batchCount; ++batch) {
const uint32_t pos = WHOLE ? block * FRAME : batch * t.n + priorPos;
auto r = firstRaw;
if (batch != batchFirst) {
r = mem.AllocRaw();
AscendC::DataCopyExtParams rp{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(r, rawGm[pos * 6], rp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
mem.RawReady();
}
AscendC::LocalTensor<float> rf;
if constexpr (rawFirstRow) rf = r.template ReinterpretCast<float>();
else if constexpr (rawFirst) rf = mem.RawFloat();
else {
mem.RawWait();
if constexpr (sizeof(T) == sizeof(float)) rf = r.template ReinterpretCast<float>();
else {
rf = mem.RawFloat();
AscendC::Cast(rf, r, AscendC::RoundMode::CAST_NONE, count * 6);
}
AscendC::Muls(rf, rf, -1.0f, count * 6);
AscendC::Exp(rf, rf, count * 6);
AscendC::Adds(rf, rf, 1.0f, count * 6);
if (one8Full) {
const AscendC::BinaryRepeatParams divOne8{1, 0, 1, 8, 0, 8};
AscendC::Div<float>(rf, one, rf, uint64_t(64), uint8_t(78), divOne8);
} else AscendC::Div(rf, one, rf, count * 6);
AscendC::PipeBarrier<PIPE_V>();
}
if constexpr (sizeof(T) == sizeof(float)) {
if constexpr (ROW) {
NativeInputPhase<0, 0>();
NativeInputPhase<0, 1>();
NativeInputPhase<0, 2>();
if constexpr (!RAW3) {
NativeInputPhase<0, 3>();
NativeInputPhase<0, 4>();
NativeInputPhase<0, 5>();
}
} else if (reinterpret_cast<uint64_t>(rf.GetPhyAddr()) ==
reinterpret_cast<uint64_t>(mem.RawFixedSlot(0).GetPhyAddr())) {
NativeInputPhase<0, 0>();
NativeInputPhase<0, 1>();
NativeInputPhase<0, 2>();
if constexpr (!RAW3) {
NativeInputPhase<0, 3>();
NativeInputPhase<0, 4>();
NativeInputPhase<0, 5>();
}
} else {
NativeInputPhase<1, 0>();
NativeInputPhase<1, 1>();
NativeInputPhase<1, 2>();
if constexpr (!RAW3) {
NativeInputPhase<1, 3>();
NativeInputPhase<1, 4>();
NativeInputPhase<1, 5>();
}
}
} else {
NativeInputPhase<0, 0>();
NativeInputPhase<0, 1>();
NativeInputPhase<0, 2>();
if constexpr (!RAW3) {
NativeInputPhase<0, 3>();
NativeInputPhase<0, 4>();
NativeInputPhase<0, 5>();
}
}
AscendC::PipeBarrier<PIPE_V>();
if constexpr (ROW_CACHED_STRIDE) {
static_assert(!WHOLE && ROW && FRAME == 896U && STRIDE == 912U && VA_GROUP28 && QUALIFIED_ROW_N == 4097U && sizeof(T) == sizeof(float), "qualified FP32 P4 cached stride run");
auto *wPtr = reinterpret_cast<__ubuf__ float *>(mem.Work().GetPhyAddr());
AscendC::SetMaskCount();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(4U * STRIDE);
asc_mul_scalar(wPtr, wPtr, 2.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(2U * STRIDE);
asc_add_scalar(wPtr, wPtr, -0.5f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_add(wPtr, wPtr, wPtr + 6U * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_mul(wPtr + 2U * STRIDE, wPtr + 2U * STRIDE, wPtr + 2U * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_mul(wPtr + 2U * STRIDE, wPtr + 2U * STRIDE, wPtr + 9U * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(FRAME);
#pragma unroll
for (uint32_t k = 0; k < 4U; ++k)
asc_mul(wPtr + k * STRIDE, wPtr + k * STRIDE, wPtr + 8U * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_mul(wPtr + 4U * STRIDE, wPtr + 4U * STRIDE, wPtr + 5U * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::SetMaskNorm();
} else
if constexpr (WHOLE && DIRECT_STRIDE) {
auto *wPtr = reinterpret_cast<__ubuf__ float *>(mem.Work().GetPhyAddr());
AscendC::SetMaskCount();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(4 * STRIDE);
asc_mul_scalar(wPtr, wPtr, 2.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(2 * STRIDE);
asc_add_scalar(wPtr, wPtr, -0.5f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
asc_mul(wPtr + 2 * STRIDE, wPtr + 2 * STRIDE, wPtr + 2 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_add(wPtr, wPtr, wPtr + 6 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
asc_mul(wPtr + 2 * STRIDE, wPtr + 2 * STRIDE, wPtr + 9 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(FRAME);
#pragma unroll
for (uint32_t k = 0; k < 4; ++k)
asc_mul(wPtr + k * STRIDE, wPtr + k * STRIDE, wPtr + 8 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
asc_mul(wPtr + 4 * STRIDE, wPtr + 4 * STRIDE, wPtr + 5 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::SetMaskNorm();
AscendC::ResetMask();
} else {
AscendC::Muls(f[0], f[0], 2.0f, 4 * STRIDE);
AscendC::Adds(f[0], f[0], -0.5f, 2 * STRIDE);
AscendC::Add(f[0], f[0], f[6], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[2], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[9], 2 * STRIDE);
if constexpr (DIRECT_STRIDE) {
#pragma unroll
for (uint32_t k = 0; k < 4; ++k)
AscendC::Mul(f[k], f[k], f[8], FRAME);
} else {
AscendC::Mul(f[0], f[0], p, 4 * STRIDE);
}
AscendC::Mul(f[4], f[4], f[5], FRAME);
}
auto out = mem.AllocOut();
AscendC::PipeBarrier<PIPE_V>();
if constexpr (sizeof(T) == sizeof(float)) {
if constexpr (ROW) {
NativeOutputPhase<0, 0>();
NativeOutputPhase<0, 1>();
NativeOutputPhase<0, 2>();
NativeOutputPhase<0, 3>();
NativeOutputPhase<0, 4>();
} else if (mem.CurrentOutSlot() == 0) {
NativeOutputPhase<0, 0>();
NativeOutputPhase<0, 1>();
NativeOutputPhase<0, 2>();
NativeOutputPhase<0, 3>();
NativeOutputPhase<0, 4>();
} else {
NativeOutputPhase<1, 0>();
NativeOutputPhase<1, 1>();
NativeOutputPhase<1, 2>();
NativeOutputPhase<1, 3>();
NativeOutputPhase<1, 4>();
}
} else {
NativeOutputPhase<0, 0>();
NativeOutputPhase<0, 1>();
NativeOutputPhase<0, 2>();
NativeOutputPhase<0, 3>();
NativeOutputPhase<0, 4>();
}
AscendC::PipeBarrier<PIPE_V>();
if constexpr (sizeof(T) != sizeof(float))
AscendC::Cast(out, mem.PackedFloat(), AscendC::RoundMode::CAST_RINT, count * 5);
mem.OutReady(); mem.OutWait();
AscendC::DataCopyExtParams op{1, count * 5 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(outGm[pos * 5], out, op);
mem.FreeOut();
mem.FreeRaw();
}
mem.FreePrior();
}
mem.Drain();
}
};


template <typename T, uint32_t TILE, bool VA_GROUP28 = false>
class FireA2RowH10Storage {
static_assert(TILE >= 128 && TILE <= 1024 && TILE % 128 == 0, "native hardware frames or large reuse frame");
static constexpr uint32_t FRAME = TILE;
static constexpr uint32_t DEPTH = TILE > 256 ? 2 : 1;
static constexpr uint32_t STRIDE = FRAME + 16;
static constexpr uint32_t RAW_BYTES = (sizeof(T) == sizeof(float) ? FRAME : TILE) * 6 * sizeof(T);
static constexpr uint32_t PRIOR_WORDS = FRAME * 5;
static constexpr uint32_t PRIOR_BYTES = PRIOR_WORDS * sizeof(float);
static constexpr uint32_t OUT_BYTES = (sizeof(T) == sizeof(float) ? FRAME : TILE) * 5 * sizeof(T);
static constexpr uint32_t WORK_BYTES = ((STRIDE * 11 * sizeof(float) + 511) / 512) * 512 + 256;
static constexpr uint32_t ONE_WORDS = FRAME * 6;
static constexpr uint32_t ONE_BYTES = ONE_WORDS * sizeof(float) + (sizeof(T) == sizeof(half) ? 256 : 0);
static constexpr uint32_t RAW_BASE = 0;
static constexpr uint32_t PRIOR_BASE = RAW_BASE + DEPTH * RAW_BYTES;
static constexpr uint32_t OUT_BASE = PRIOR_BASE + PRIOR_BYTES;
static constexpr uint32_t WORK_BASE = OUT_BASE + DEPTH * OUT_BYTES;
static constexpr uint32_t ONE_BASE = WORK_BASE + WORK_BYTES;
static constexpr uint32_t RF_BASE = ONE_BASE + ONE_BYTES;
static constexpr uint32_t PACK_BASE = RF_BASE + (sizeof(T) == sizeof(half) ? FRAME * 6 * sizeof(float) : 0);
static constexpr uint32_t INDEX_BASE = PACK_BASE + (sizeof(T) == sizeof(half) ? FRAME * 5 * sizeof(float) + 256 : 0);
static constexpr uint32_t VA_BASE = INDEX_BASE + FRAME * sizeof(uint32_t);
static constexpr uint32_t VA_WORDS = VA_GROUP28 ? 112U : 172U;
static_assert(!VA_GROUP28 || (sizeof(T) == sizeof(half) && TILE == 896U && VA_BASE == 163072U && VA_BASE + VA_WORDS * sizeof(uint64_t) == 163968U), "qualified P10 VA28 fixed UB");
static_assert(VA_BASE + VA_WORDS * sizeof(uint64_t) < 188416, "reserve the SDK temporary UB region");
static_assert((RAW_BYTES | PRIOR_BYTES | OUT_BYTES | WORK_BYTES | ONE_BYTES | INDEX_BASE) % 32 == 0,
"all static addresses and capacities retain 32-byte alignment");
uint32_t rawCursor = 0, priorCursor = 0, outCursor = 0;
uint32_t rawSlot = 0, priorSlot = 0, outSlot = 0;
uint32_t rawUsed = 0, priorUsed = 0, outUsed = 0;
public:
__aicore__ inline AscendC::LocalTensor<uint64_t> PackedVA() {
return AscendC::LocalTensor<uint64_t>(AscendC::TPosition::VECCALC, VA_BASE, VA_WORDS);
}
__aicore__ inline AscendC::LocalTensor<int32_t> Indices() {
return AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, INDEX_BASE, FRAME);
}
__aicore__ inline AscendC::LocalTensor<float> Work() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, WORK_BASE, WORK_BYTES / sizeof(float));
}
__aicore__ inline AscendC::LocalTensor<float> Ones() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE, ONE_BYTES / sizeof(float));
}
__aicore__ inline AscendC::LocalTensor<float> RawFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, RF_BASE, FRAME * 6);
}
__aicore__ inline AscendC::LocalTensor<float> PackedFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, PACK_BASE, FRAME * 5);
}
__aicore__ inline AscendC::LocalTensor<T> AllocRaw() {
rawSlot = rawCursor;
rawCursor = (rawCursor + 1) % DEPTH;
if (rawUsed & (1U << rawSlot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(rawSlot);
return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, RAW_BASE + rawSlot * RAW_BYTES, (sizeof(T) == sizeof(float) ? FRAME : TILE) * 6);
}
__aicore__ inline AscendC::LocalTensor<float> AllocPrior() {
priorSlot = priorCursor;
priorCursor = 0;
if (priorUsed & (1U << priorSlot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2 + priorSlot);
return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE + priorSlot * PRIOR_BYTES, PRIOR_WORDS);
}
__aicore__ inline AscendC::LocalTensor<T> AllocOut() {
outSlot = outCursor;
outCursor = (outCursor + 1) % DEPTH;
if (outUsed & (1U << outSlot)) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outSlot);
return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, OUT_BASE + outSlot * OUT_BYTES, (sizeof(T) == sizeof(float) ? FRAME : TILE) * 5);
}
__aicore__ inline AscendC::LocalTensor<T> RawFixedSlot(uint32_t slot) {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, RAW_BASE + slot * RAW_BYTES,
(sizeof(T) == sizeof(float) ? FRAME : TILE) * 6);
}
__aicore__ inline AscendC::LocalTensor<float> PriorFixedSlot(uint32_t slot) {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE + slot * PRIOR_BYTES, PRIOR_WORDS);
}
__aicore__ inline AscendC::LocalTensor<T> OutAt(uint32_t slot) {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, OUT_BASE + slot * OUT_BYTES,
(sizeof(T) == sizeof(float) ? FRAME : TILE) * 5);
}
__aicore__ inline uint32_t CurrentOutSlot() const { return outSlot; }
__aicore__ inline void RawReady() { AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(rawSlot); }
__aicore__ inline void RawWait() { AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(rawSlot); }
__aicore__ inline void PriorReady() { AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2 + priorSlot); }
__aicore__ inline void PriorWait() { AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2 + priorSlot); }
__aicore__ inline void OutReady() { AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outSlot); }
__aicore__ inline void OutWait() { AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outSlot); }
__aicore__ inline void FreeRaw() {
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(rawSlot);
rawUsed |= 1U << rawSlot;
}
__aicore__ inline void FreePrior() {
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2 + priorSlot);
priorUsed |= 1U << priorSlot;
}
__aicore__ inline void FreeOut() {
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outSlot);
outUsed |= 1U << outSlot;
}
__aicore__ inline void Drain() {
for (uint32_t slot = 0; slot < DEPTH; ++slot) {
if (rawUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
if (priorUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2 + slot);
if (outUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
}
}
};


template <typename T, uint32_t TILE, bool ROW, bool RAW3 = false, bool DIRECT_STRIDE = false, bool VA_GROUP28 = false, uint32_t QUALIFIED_N = 0U, bool ROW_CACHED_STRIDE = false>
class KernelFireA2RowH10 {
static constexpr uint32_t DEPTH = TILE > 256 ? 2 : 1;
static constexpr uint32_t FRAME = TILE;
static constexpr uint32_t STRIDE = FRAME + 16;
static constexpr uint32_t Q = FRAME / 8;
static_assert(!VA_GROUP28 || (sizeof(T) == sizeof(half) && FRAME == 896U && ROW && RAW3 && !DIRECT_STRIDE && QUALIFIED_N == 4097U), "qualified P10 half C20 N4097 ROW896 raw3 VA28");
static_assert(!ROW_CACHED_STRIDE || (VA_GROUP28 && sizeof(T) == sizeof(half) && FRAME == 896U && STRIDE == 912U && ROW && RAW3 && !DIRECT_STRIDE && QUALIFIED_N == 4097U), "qualified P10 cached stride ROW896");
FireA2RowH10Storage<T, TILE, VA_GROUP28> mem;
template <uint32_t OFFSET>
__aicore__ static constexpr uint32_t VAOffset() {
if constexpr (VA_GROUP28) {
static_assert(OFFSET % 4U == 0U && OFFSET < 172U, "VA28 original group bounds");
constexpr uint32_t group = OFFSET / 4U;
static_assert(group < 3U || (group >= 6U && group < 9U) || (group >= 18U && group < 28U) || (group >= 33U && group < 43U), "VA28 original active group");
if constexpr (group < 3U) return 4U * (15U + group);
else if constexpr (group < 9U) return 4U * (3U * (group - 6U));
else if constexpr (group < 23U) return 4U * group;
else if constexpr (group < 28U) return 4U * (1U + 3U * (group - 23U));
else if constexpr (group < 38U) return 4U * (23U + group - 33U);
else return 4U * (2U + 3U * (group - 38U));
} else return OFFSET;
}
AscendC::GlobalTensor<T> rawGm, outGm;
AscendC::GlobalTensor<float> priorGm;
__aicore__ inline void StorePacked(AscendC::LocalTensor<uint64_t> dst, const uint64_t *address) {
#pragma unroll
for (uint32_t q = 0; q < 4; ++q) {
const uint64_t value = ((address[4 * q] >> 5) & 0x1fffULL) |
(((address[4 * q + 1] >> 5) & 0x1fffULL) << 16) |
(((address[4 * q + 2] >> 5) & 0x1fffULL) << 32) |
(((address[4 * q + 3] >> 5) & 0x1fffULL) << 48);
dst.SetValue(q, value);
}
}
template <uint8_t REPEAT, uint16_t DST_STRIDE, uint16_t SRC_STRIDE>
__aicore__ inline void PackedTranspose(AscendC::LocalTensor<uint64_t> dst,
AscendC::LocalTensor<uint64_t> src) {
auto *dstPacked = reinterpret_cast<__ubuf__ uint64_t *>(dst.GetPhyAddr());
auto *srcPacked = reinterpret_cast<__ubuf__ uint64_t *>(src.GetPhyAddr());
__cce_scalar::vld_va_reg(VA0, dstPacked, L128);
__cce_scalar::vld_va_reg(VA1, dstPacked, H128);
__cce_scalar::vld_va_reg(VA2, srcPacked, L128);
__cce_scalar::vld_va_reg(VA3, srcPacked, H128);
__cce_scalar::scatter_vnchwconv_b32(VA0, VA2, REPEAT,
REPEAT == 1 ? 0 : DST_STRIDE, REPEAT == 1 ? 0 : SRC_STRIDE);
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitInputPhase() {
static_assert(PHASE < 6 && SLOT < (sizeof(T) == sizeof(float) ? 2 : 1),
"valid padded input specialization");
static_assert(!RAW3 || ((((FRAME == 896 || FRAME == 768) && !ROW) || ((FRAME == 512 || FRAME == 640) && ROW) || (VA_GROUP28 && QUALIFIED_N == 4097U && FRAME == 896U && ROW && sizeof(T) == sizeof(half))) && PHASE < 3), "raw3 original or qualified halfC20 ROW896");
uint64_t sourceBase;
if constexpr (sizeof(T) == sizeof(float))
sourceBase = reinterpret_cast<uint64_t>(mem.RawFixedSlot(SLOT).GetPhyAddr());
else sourceBase = reinterpret_cast<uint64_t>(mem.RawFloat().GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
if constexpr (RAW3) {
uint64_t source[16] = {
sourceBase + uint64_t(0 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(48 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(96 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(144 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(192 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(240 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(288 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(336 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(24 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(72 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(120 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(168 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(216 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(264 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(312 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(360 + 8 * PHASE) * sizeof(float)
};
uint64_t target[16] = {
workBase + uint64_t(((8 * PHASE + 0) % 6) * STRIDE + (((8 * PHASE + 0) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 0) % 6) * STRIDE + (((8 * PHASE + 0) / 6) + 4) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 1) % 6) * STRIDE + (((8 * PHASE + 1) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 1) % 6) * STRIDE + (((8 * PHASE + 1) / 6) + 4) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 2) % 6) * STRIDE + (((8 * PHASE + 2) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 2) % 6) * STRIDE + (((8 * PHASE + 2) / 6) + 4) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 3) % 6) * STRIDE + (((8 * PHASE + 3) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 3) % 6) * STRIDE + (((8 * PHASE + 3) / 6) + 4) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 4) % 6) * STRIDE + (((8 * PHASE + 4) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 4) % 6) * STRIDE + (((8 * PHASE + 4) / 6) + 4) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 5) % 6) * STRIDE + (((8 * PHASE + 5) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 5) % 6) * STRIDE + (((8 * PHASE + 5) / 6) + 4) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 6) % 6) * STRIDE + (((8 * PHASE + 6) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 6) % 6) * STRIDE + (((8 * PHASE + 6) / 6) + 4) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 7) % 6) * STRIDE + (((8 * PHASE + 7) / 6) + 0) * Q) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 7) % 6) * STRIDE + (((8 * PHASE + 7) / 6) + 4) * Q) * sizeof(float)
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[VAOffset<4 * PHASE>()], target);
if constexpr (!VA_GROUP28 || PHASE == 0U)
StorePacked(mem.PackedVA()[VAOffset<24 + SLOT * 24 + 4 * PHASE>()], source);
} else {
uint64_t source[16] = {
sourceBase + uint64_t(0 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(48 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(96 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(144 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(192 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(240 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(288 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(336 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(384 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(432 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(480 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(528 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(576 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(624 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(672 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(720 + 8 * PHASE) * sizeof(float)
};
uint64_t target[16] = {
workBase + uint64_t(((8 * PHASE + 0) % 6) * STRIDE + ((8 * PHASE + 0) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 0) % 6) * STRIDE + ((8 * PHASE + 0) / 6) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 1) % 6) * STRIDE + ((8 * PHASE + 1) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 1) % 6) * STRIDE + ((8 * PHASE + 1) / 6) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 2) % 6) * STRIDE + ((8 * PHASE + 2) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 2) % 6) * STRIDE + ((8 * PHASE + 2) / 6) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 3) % 6) * STRIDE + ((8 * PHASE + 3) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 3) % 6) * STRIDE + ((8 * PHASE + 3) / 6) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 4) % 6) * STRIDE + ((8 * PHASE + 4) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 4) % 6) * STRIDE + ((8 * PHASE + 4) / 6) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 5) % 6) * STRIDE + ((8 * PHASE + 5) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 5) % 6) * STRIDE + ((8 * PHASE + 5) / 6) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 6) % 6) * STRIDE + ((8 * PHASE + 6) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 6) % 6) * STRIDE + ((8 * PHASE + 6) / 6) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 7) % 6) * STRIDE + ((8 * PHASE + 7) / 6) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 7) % 6) * STRIDE + ((8 * PHASE + 7) / 6) * Q + 8) * sizeof(float)
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[VAOffset<4 * PHASE>()], target);
if constexpr (!VA_GROUP28 || PHASE == 0U)
StorePacked(mem.PackedVA()[VAOffset<24 + SLOT * 24 + 4 * PHASE>()], source);
}
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitOutputPhase() {
static_assert(PHASE < 5 && SLOT < (sizeof(T) == sizeof(float) ? 2 : 1),
"valid padded output specialization");
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t outputBase;
if constexpr (sizeof(T) == sizeof(float))
outputBase = reinterpret_cast<uint64_t>(mem.OutAt(SLOT).GetPhyAddr());
else outputBase = reinterpret_cast<uint64_t>(mem.PackedFloat().GetPhyAddr());
uint64_t source[16] = {
workBase + uint64_t(((8 * PHASE + 0) % 5) * STRIDE + ((8 * PHASE + 0) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 1) % 5) * STRIDE + ((8 * PHASE + 1) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 2) % 5) * STRIDE + ((8 * PHASE + 2) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 3) % 5) * STRIDE + ((8 * PHASE + 3) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 4) % 5) * STRIDE + ((8 * PHASE + 4) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 5) % 5) * STRIDE + ((8 * PHASE + 5) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 6) % 5) * STRIDE + ((8 * PHASE + 6) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 7) % 5) * STRIDE + ((8 * PHASE + 7) / 5) * Q + 0) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 0) % 5) * STRIDE + ((8 * PHASE + 0) / 5) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 1) % 5) * STRIDE + ((8 * PHASE + 1) / 5) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 2) % 5) * STRIDE + ((8 * PHASE + 2) / 5) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 3) % 5) * STRIDE + ((8 * PHASE + 3) / 5) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 4) % 5) * STRIDE + ((8 * PHASE + 4) / 5) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 5) % 5) * STRIDE + ((8 * PHASE + 5) / 5) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 6) % 5) * STRIDE + ((8 * PHASE + 6) / 5) * Q + 8) * sizeof(float),
workBase + uint64_t(((8 * PHASE + 7) % 5) * STRIDE + ((8 * PHASE + 7) / 5) * Q + 8) * sizeof(float)
};
uint64_t target[16] = {
outputBase + uint64_t((5 * 0 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 8 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 1 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 9 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 2 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 10 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 3 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 11 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 4 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 12 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 5 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 13 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 6 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 14 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 7 + PHASE) * 8) * sizeof(float),
outputBase + uint64_t((5 * 15 + PHASE) * 8) * sizeof(float)
};
constexpr uint32_t INPUT_WORDS = 72;
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[VAOffset<INPUT_WORDS + 4 * PHASE>()], source);
if constexpr (!VA_GROUP28 || PHASE == 0U)
StorePacked(mem.PackedVA()[VAOffset<INPUT_WORDS + 20 + SLOT * 20 + 4 * PHASE>()], target);
}
template <uint32_t PHASE>
__aicore__ inline void InitPriorPhase() {
static_assert(PHASE < 5, "native prior hardware phase");
constexpr uint32_t PRIOR_TARGET = 132;
constexpr uint32_t PRIOR_SOURCE = 152 + 4 * PHASE;
const uint64_t sourceBase = reinterpret_cast<uint64_t>(mem.PriorFixedSlot(0).GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t source[16] = {
sourceBase + uint64_t(0 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(40 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(80 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(120 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(160 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(200 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(240 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(280 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(320 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(360 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(400 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(440 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(480 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(520 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(560 + 8 * PHASE) * sizeof(float),
sourceBase + uint64_t(600 + 8 * PHASE) * sizeof(float)
};
uint64_t target[16] = {
workBase + uint64_t((6 + (8 * PHASE + 0) % 5) * STRIDE + ((8 * PHASE + 0) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 0) % 5) * STRIDE + ((8 * PHASE + 0) / 5) * Q + 1 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 1) % 5) * STRIDE + ((8 * PHASE + 1) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 1) % 5) * STRIDE + ((8 * PHASE + 1) / 5) * Q + 1 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 2) % 5) * STRIDE + ((8 * PHASE + 2) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 2) % 5) * STRIDE + ((8 * PHASE + 2) / 5) * Q + 1 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 3) % 5) * STRIDE + ((8 * PHASE + 3) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 3) % 5) * STRIDE + ((8 * PHASE + 3) / 5) * Q + 1 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 4) % 5) * STRIDE + ((8 * PHASE + 4) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 4) % 5) * STRIDE + ((8 * PHASE + 4) / 5) * Q + 1 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 5) % 5) * STRIDE + ((8 * PHASE + 5) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 5) % 5) * STRIDE + ((8 * PHASE + 5) / 5) * Q + 1 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 6) % 5) * STRIDE + ((8 * PHASE + 6) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 6) % 5) * STRIDE + ((8 * PHASE + 6) / 5) * Q + 1 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 7) % 5) * STRIDE + ((8 * PHASE + 7) / 5) * Q + 0 * 8) * sizeof(float),
workBase + uint64_t((6 + (8 * PHASE + 7) % 5) * STRIDE + ((8 * PHASE + 7) / 5) * Q + 1 * 8) * sizeof(float)
};
StorePacked(mem.PackedVA()[VAOffset<PRIOR_TARGET + 4 * PHASE>()], target);
if constexpr (!VA_GROUP28 || PHASE == 0U)
StorePacked(mem.PackedVA()[VAOffset<PRIOR_SOURCE>()], source);
}
template <uint32_t PHASE>
__aicore__ inline void NativePriorPhase() {
constexpr uint32_t PRIOR_TARGET = 132;
constexpr uint32_t PRIOR_SOURCE = 152 + 4 * PHASE;
PackedTranspose<FRAME / 128, 2, 80>(mem.PackedVA()[VAOffset<PRIOR_TARGET + 4 * PHASE>()],
mem.PackedVA()[VAOffset<PRIOR_SOURCE>()]);
}
__aicore__ inline void InitPackedVA() {
InitInputPhase<0, 0>();
InitInputPhase<0, 1>();
InitInputPhase<0, 2>();
if constexpr (!RAW3) {
InitInputPhase<0, 3>();
InitInputPhase<0, 4>();
InitInputPhase<0, 5>();
}
if constexpr (sizeof(T) == sizeof(float) && !(ROW)) {
InitInputPhase<1, 0>();
InitInputPhase<1, 1>();
InitInputPhase<1, 2>();
if constexpr (!RAW3) {
InitInputPhase<1, 3>();
InitInputPhase<1, 4>();
InitInputPhase<1, 5>();
}
}
InitOutputPhase<0, 0>();
InitOutputPhase<0, 1>();
InitOutputPhase<0, 2>();
InitOutputPhase<0, 3>();
InitOutputPhase<0, 4>();
if constexpr (sizeof(T) == sizeof(float) && !(ROW)) {
InitOutputPhase<1, 0>();
InitOutputPhase<1, 1>();
InitOutputPhase<1, 2>();
InitOutputPhase<1, 3>();
InitOutputPhase<1, 4>();
}
{
InitPriorPhase<0>();
InitPriorPhase<1>();
InitPriorPhase<2>();
InitPriorPhase<3>();
InitPriorPhase<4>();
}
AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
if constexpr (VA_GROUP28) {
auto *entries = reinterpret_cast<__ubuf__ int16_t *>(mem.PackedVA().GetPhyAddr());
AscendC::SetMaskCount();
AscendC::SetVectorMask<int16_t, AscendC::MaskMode::COUNTER>(48U);
#pragma unroll
for (uint32_t phase = 1U; phase < 5U; ++phase)
asc_add_scalar(entries + phase * 48U, entries, int16_t(phase),
uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetMaskNorm();
AscendC::ResetMask();
}
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void NativeInputPhase() {
static_assert(PHASE < 6 && SLOT < (sizeof(T) == sizeof(float) ? 2 : 1), "valid packed input phase");
if constexpr (RAW3) {
PackedTranspose<FRAME / 64, 1, 48>(mem.PackedVA()[VAOffset<4 * PHASE>()],
mem.PackedVA()[VAOffset<24 + SLOT * 24 + 4 * PHASE>()]);
} else {
PackedTranspose<FRAME / 128, 2, 96>(mem.PackedVA()[VAOffset<4 * PHASE>()],
mem.PackedVA()[VAOffset<24 + SLOT * 24 + 4 * PHASE>()]);
}
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void NativeOutputPhase() {
static_assert(PHASE < 5 && SLOT < (sizeof(T) == sizeof(float) ? 2 : 1), "valid packed output phase");
constexpr uint32_t INPUT_WORDS = 72;
PackedTranspose<FRAME / 128, 80, 2>(mem.PackedVA()[VAOffset<INPUT_WORDS + 20 + SLOT * 20 + 4 * PHASE>()],
mem.PackedVA()[VAOffset<INPUT_WORDS + 4 * PHASE>()]);
}
public:
template <bool WHOLE = false>
__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded,
const FireFrameTilingData &schedule) {
static_assert(!VA_GROUP28 || !WHOLE, "qualified P10 VA28 nonwhole only");
static_assert(!WHOLE || ROW, "whole entry requires one batch per core");
static_assert(!DIRECT_STRIDE || (FRAME == 512 && ROW && WHOLE), "direct stride only original whole512");
constexpr bool rawFirst = WHOLE && DIRECT_STRIDE && RAW3 && FRAME == 512 && sizeof(T) == sizeof(half);
constexpr bool rawFirstRow = ROW_CACHED_STRIDE && QUALIFIED_N == 4097U;
static_assert(!rawFirstRow || (!WHOLE && ROW && RAW3 && !DIRECT_STRIDE && VA_GROUP28 && FRAME == 896U && STRIDE == 912U && sizeof(T) == sizeof(half)), "qualified H10 C20 Cast raw before VA only");
const auto &t = schedule.header;
rawGm.SetGlobalBuffer((__gm__ T *)raw);
priorGm.SetGlobalBuffer((__gm__ float *)priors);
outGm.SetGlobalBuffer((__gm__ T *)decoded);
const uint32_t groupLanes = schedule.geometry & 0xffffU;
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
const uint32_t begin = (block % groupLanes) * t.perCore;
if (begin >= t.n) return;
const uint32_t batchCount = ROW ? 1U : schedule.geometry >> 16;
const uint32_t batchFirst = (block / groupLanes) * batchCount;
const uint32_t priorPos = begin;
const uint32_t count = WHOLE ? FRAME : (t.n - priorPos < t.perCore ? t.n - priorPos : t.perCore);
const bool one8Full = ROW_CACHED_STRIDE && QUALIFIED_N == 4097U && count == 832U;
auto firstRaw = mem.AllocRaw();
auto p = mem.AllocPrior();
AscendC::DataCopyExtParams firstRp{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyExtParams firstPp{1, count * 5 * uint32_t(sizeof(float)), 0, 0, 0};
AscendC::DataCopyPad(firstRaw, rawGm[(WHOLE ? block * FRAME : batchFirst * t.n + priorPos) * 6], firstRp,
AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
if constexpr (rawFirst || rawFirstRow) mem.RawReady();
AscendC::DataCopyPad(p, priorGm[priorPos * 5], firstPp,
AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
if constexpr (!(rawFirst || rawFirstRow)) mem.RawReady();
mem.PriorReady();
auto one = mem.Ones();
AscendC::Duplicate(one, 1.0f, one8Full ? 8U : FRAME * 6);
AscendC::LocalTensor<float> f[11];
for (uint32_t k = 0; k < 11; ++k) f[k] = mem.Work()[k * STRIDE];
if constexpr (rawFirst) {
mem.RawWait();
auto rf = mem.RawFloat();
AscendC::Cast(rf, firstRaw, AscendC::RoundMode::CAST_NONE, count * 6);
AscendC::Muls(rf, rf, -1.0f, count * 6);
AscendC::Exp(rf, rf, count * 6);
AscendC::Adds(rf, rf, 1.0f, count * 6);
AscendC::Div(rf, one, rf, count * 6);
AscendC::PipeBarrier<PIPE_V>();
}
if constexpr (rawFirstRow) {
mem.RawWait();
auto rf = mem.RawFloat();
AscendC::Cast(rf, firstRaw, AscendC::RoundMode::CAST_NONE, count * 6);
AscendC::Muls(rf, rf, -1.0f, count * 6);
AscendC::Exp(rf, rf, count * 6);
AscendC::Adds(rf, rf, 1.0f, count * 6);
if (one8Full) {
const AscendC::BinaryRepeatParams divOne8{1, 0, 1, 8, 0, 8};
AscendC::Div<float>(rf, one, rf, uint64_t(64), uint8_t(78), divOne8);
} else AscendC::Div(rf, one, rf, count * 6);
AscendC::PipeBarrier<PIPE_V>();
}
InitPackedVA();
{
mem.PriorWait();
NativePriorPhase<0>();
NativePriorPhase<1>();
NativePriorPhase<2>();
NativePriorPhase<3>();
NativePriorPhase<4>();
AscendC::PipeBarrier<PIPE_V>();
if constexpr (!DIRECT_STRIDE && !ROW_CACHED_STRIDE) {
AscendC::DataCopy(p, f[8], STRIDE);
AscendC::DataCopy(p[STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[2 * STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[3 * STRIDE], f[8], STRIDE);
AscendC::PipeBarrier<PIPE_V>();
}
for (uint32_t batch = batchFirst; batch < batchFirst + batchCount; ++batch) {
const uint32_t pos = WHOLE ? block * FRAME : batch * t.n + priorPos;
auto r = firstRaw;
if (batch != batchFirst) {
r = mem.AllocRaw();
AscendC::DataCopyExtParams rp{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(r, rawGm[pos * 6], rp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
mem.RawReady();
}
AscendC::LocalTensor<float> rf;
if constexpr (rawFirst || rawFirstRow) rf = mem.RawFloat();
else {
mem.RawWait();
if constexpr (sizeof(T) == sizeof(float)) rf = r.template ReinterpretCast<float>();
else {
rf = mem.RawFloat();
AscendC::Cast(rf, r, AscendC::RoundMode::CAST_NONE, count * 6);
}
AscendC::Muls(rf, rf, -1.0f, count * 6);
AscendC::Exp(rf, rf, count * 6);
AscendC::Adds(rf, rf, 1.0f, count * 6);
if (one8Full) {
const AscendC::BinaryRepeatParams divOne8{1, 0, 1, 8, 0, 8};
AscendC::Div<float>(rf, one, rf, uint64_t(64), uint8_t(78), divOne8);
} else AscendC::Div(rf, one, rf, count * 6);
AscendC::PipeBarrier<PIPE_V>();
}
if constexpr (sizeof(T) == sizeof(float)) {
if constexpr (ROW) {
NativeInputPhase<0, 0>();
NativeInputPhase<0, 1>();
NativeInputPhase<0, 2>();
if constexpr (!RAW3) {
NativeInputPhase<0, 3>();
NativeInputPhase<0, 4>();
NativeInputPhase<0, 5>();
}
} else if (reinterpret_cast<uint64_t>(rf.GetPhyAddr()) ==
reinterpret_cast<uint64_t>(mem.RawFixedSlot(0).GetPhyAddr())) {
NativeInputPhase<0, 0>();
NativeInputPhase<0, 1>();
NativeInputPhase<0, 2>();
if constexpr (!RAW3) {
NativeInputPhase<0, 3>();
NativeInputPhase<0, 4>();
NativeInputPhase<0, 5>();
}
} else {
NativeInputPhase<1, 0>();
NativeInputPhase<1, 1>();
NativeInputPhase<1, 2>();
if constexpr (!RAW3) {
NativeInputPhase<1, 3>();
NativeInputPhase<1, 4>();
NativeInputPhase<1, 5>();
}
}
} else {
NativeInputPhase<0, 0>();
NativeInputPhase<0, 1>();
NativeInputPhase<0, 2>();
if constexpr (!RAW3) {
NativeInputPhase<0, 3>();
NativeInputPhase<0, 4>();
NativeInputPhase<0, 5>();
}
}
AscendC::PipeBarrier<PIPE_V>();
if constexpr (ROW_CACHED_STRIDE) {
static_assert(!WHOLE && ROW && FRAME == 896U && STRIDE == 912U && VA_GROUP28 && QUALIFIED_N == 4097U && sizeof(T) == sizeof(half), "qualified P10 cached stride run");
auto *wPtr = reinterpret_cast<__ubuf__ float *>(mem.Work().GetPhyAddr());
AscendC::SetMaskCount();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(4U * STRIDE);
asc_mul_scalar(wPtr, wPtr, 2.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(2U * STRIDE);
asc_add_scalar(wPtr, wPtr, -0.5f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
asc_mul(wPtr + 2U * STRIDE, wPtr + 2U * STRIDE, wPtr + 2U * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_add(wPtr, wPtr, wPtr + 6U * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
asc_mul(wPtr + 2U * STRIDE, wPtr + 2U * STRIDE, wPtr + 9U * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(FRAME);
#pragma unroll
for (uint32_t k = 0; k < 4U; ++k)
asc_mul(wPtr + k * STRIDE, wPtr + k * STRIDE, wPtr + 8U * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
asc_mul(wPtr + 4U * STRIDE, wPtr + 4U * STRIDE, wPtr + 5U * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::SetMaskNorm();
} else
if constexpr (WHOLE && DIRECT_STRIDE) {
auto *wPtr = reinterpret_cast<__ubuf__ float *>(mem.Work().GetPhyAddr());
AscendC::SetMaskCount();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(4 * STRIDE);
asc_mul_scalar(wPtr, wPtr, 2.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(2 * STRIDE);
asc_add_scalar(wPtr, wPtr, -0.5f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
asc_mul(wPtr + 2 * STRIDE, wPtr + 2 * STRIDE, wPtr + 2 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_add(wPtr, wPtr, wPtr + 6 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
asc_mul(wPtr + 2 * STRIDE, wPtr + 2 * STRIDE, wPtr + 9 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(FRAME);
#pragma unroll
for (uint32_t k = 0; k < 4; ++k)
asc_mul(wPtr + k * STRIDE, wPtr + k * STRIDE, wPtr + 8 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
asc_mul(wPtr + 4 * STRIDE, wPtr + 4 * STRIDE, wPtr + 5 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
AscendC::SetMaskNorm();
AscendC::ResetMask();
} else {
AscendC::Muls(f[0], f[0], 2.0f, 4 * STRIDE);
AscendC::Adds(f[0], f[0], -0.5f, 2 * STRIDE);
AscendC::Add(f[0], f[0], f[6], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[2], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[9], 2 * STRIDE);
if constexpr (DIRECT_STRIDE) {
#pragma unroll
for (uint32_t k = 0; k < 4; ++k)
AscendC::Mul(f[k], f[k], f[8], FRAME);
} else {
AscendC::Mul(f[0], f[0], p, 4 * STRIDE);
}
AscendC::Mul(f[4], f[4], f[5], FRAME);
}
auto out = mem.AllocOut();
AscendC::PipeBarrier<PIPE_V>();
if constexpr (sizeof(T) == sizeof(float)) {
if constexpr (ROW) {
NativeOutputPhase<0, 0>();
NativeOutputPhase<0, 1>();
NativeOutputPhase<0, 2>();
NativeOutputPhase<0, 3>();
NativeOutputPhase<0, 4>();
} else if (mem.CurrentOutSlot() == 0) {
NativeOutputPhase<0, 0>();
NativeOutputPhase<0, 1>();
NativeOutputPhase<0, 2>();
NativeOutputPhase<0, 3>();
NativeOutputPhase<0, 4>();
} else {
NativeOutputPhase<1, 0>();
NativeOutputPhase<1, 1>();
NativeOutputPhase<1, 2>();
NativeOutputPhase<1, 3>();
NativeOutputPhase<1, 4>();
}
} else {
NativeOutputPhase<0, 0>();
NativeOutputPhase<0, 1>();
NativeOutputPhase<0, 2>();
NativeOutputPhase<0, 3>();
NativeOutputPhase<0, 4>();
}
AscendC::PipeBarrier<PIPE_V>();
if constexpr (sizeof(T) != sizeof(float)) {
if constexpr (VA_GROUP28) {
static_assert(sizeof(T) == sizeof(half) && FRAME == 896U && ROW && RAW3 && !DIRECT_STRIDE && QUALIFIED_N == 4097U, "qualified C20H10 output RN count");
asc_float2half_rn(reinterpret_cast<__ubuf__ half *>(out.GetPhyAddr()),
reinterpret_cast<__ubuf__ float *>(mem.PackedFloat().GetPhyAddr()), uint32_t(count * 5U));
} else
AscendC::Cast(out, mem.PackedFloat(), AscendC::RoundMode::CAST_RINT, count * 5);
}
mem.OutReady(); mem.OutWait();
AscendC::DataCopyExtParams op{1, count * 5 * uint32_t(sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(outGm[pos * 5], out, op);
mem.FreeOut();
mem.FreeRaw();
}
mem.FreePrior();
}
mem.Drain();
}
};


template <typename T, uint32_t TILE>
class FireA2F5FrameStorage {
static_assert(TILE >= 128 && TILE <= 1024 && TILE % 128 == 0, "native hardware frames or large reuse frame");
static constexpr uint32_t FRAME = TILE;
static constexpr uint32_t DEPTH = TILE > 256 ? 2 : 1;
static constexpr uint32_t STRIDE = FRAME + 16;
static constexpr uint32_t RAW_BYTES = (sizeof(T) == sizeof(float) ? FRAME : TILE) * 6 * sizeof(T);
static constexpr uint32_t PRIOR_WORDS = FRAME * 5;
static constexpr uint32_t PRIOR_BYTES = PRIOR_WORDS * sizeof(float);
static constexpr uint32_t OUT_BYTES = (sizeof(T) == sizeof(float) ? FRAME : TILE) * 5 * sizeof(T);
static constexpr uint32_t WORK_BYTES = ((STRIDE * 11 * sizeof(float) + 511) / 512) * 512 + 256;
static constexpr uint32_t ONE_WORDS = FRAME * 6;
static constexpr uint32_t ONE_BYTES = ONE_WORDS * sizeof(float) + (sizeof(T) == sizeof(half) ? 256 : 0);
static constexpr uint32_t RAW_BASE = 0;
static constexpr uint32_t PRIOR_BASE = RAW_BASE + DEPTH * RAW_BYTES;
static constexpr uint32_t OUT_BASE = PRIOR_BASE + PRIOR_BYTES;
static constexpr uint32_t WORK_BASE = OUT_BASE + DEPTH * OUT_BYTES;
static constexpr uint32_t ONE_BASE = WORK_BASE + WORK_BYTES;
static constexpr uint32_t RF_BASE = ONE_BASE + ONE_BYTES;
static constexpr uint32_t PACK_BASE = RF_BASE + (sizeof(T) == sizeof(half) ? FRAME * 6 * sizeof(float) : 0);
static constexpr uint32_t INDEX_BASE = PACK_BASE + (sizeof(T) == sizeof(half) ? FRAME * 5 * sizeof(float) + 256 : 0);
static constexpr uint32_t VA_BASE = INDEX_BASE + FRAME * sizeof(uint32_t);
static constexpr uint32_t VA_WORDS = 172;
static_assert(VA_BASE + VA_WORDS * sizeof(uint64_t) < 188416, "reserve the SDK temporary UB region");
static_assert((RAW_BYTES | PRIOR_BYTES | OUT_BYTES | WORK_BYTES | ONE_BYTES | INDEX_BASE) % 32 == 0,
"all static addresses and capacities retain 32-byte alignment");
uint32_t rawCursor = 0, priorCursor = 0, outCursor = 0;
uint32_t rawSlot = 0, priorSlot = 0, outSlot = 0;
uint32_t rawUsed = 0, priorUsed = 0, outUsed = 0;
public:
__aicore__ inline AscendC::LocalTensor<uint64_t> PackedVA() {
return AscendC::LocalTensor<uint64_t>(AscendC::TPosition::VECCALC, VA_BASE, VA_WORDS);
}
__aicore__ inline AscendC::LocalTensor<int32_t> Indices() {
return AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, INDEX_BASE, FRAME);
}
__aicore__ inline AscendC::LocalTensor<float> Work() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, WORK_BASE, WORK_BYTES / sizeof(float));
}
__aicore__ inline AscendC::LocalTensor<float> Ones() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE, ONE_BYTES / sizeof(float));
}
__aicore__ inline AscendC::LocalTensor<float> RawFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, RF_BASE, FRAME * 6);
}
__aicore__ inline AscendC::LocalTensor<float> PackedFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, PACK_BASE, FRAME * 5);
}
__aicore__ inline AscendC::LocalTensor<T> AllocRaw() {
rawSlot = rawCursor;
rawCursor = (rawCursor + 1) % DEPTH;
if (rawUsed & (1U << rawSlot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(rawSlot);
return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, RAW_BASE + rawSlot * RAW_BYTES, (sizeof(T) == sizeof(float) ? FRAME : TILE) * 6);
}
__aicore__ inline AscendC::LocalTensor<float> AllocPrior() {
priorSlot = priorCursor;
priorCursor = 0;
if (priorUsed & (1U << priorSlot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2 + priorSlot);
return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE + priorSlot * PRIOR_BYTES, PRIOR_WORDS);
}
__aicore__ inline AscendC::LocalTensor<T> AllocOut() {
outSlot = outCursor;
outCursor = (outCursor + 1) % DEPTH;
if (outUsed & (1U << outSlot)) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outSlot);
return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, OUT_BASE + outSlot * OUT_BYTES, (sizeof(T) == sizeof(float) ? FRAME : TILE) * 5);
}
__aicore__ inline AscendC::LocalTensor<T> RawFixedSlot(uint32_t slot) {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, RAW_BASE + slot * RAW_BYTES,
(sizeof(T) == sizeof(float) ? FRAME : TILE) * 6);
}
__aicore__ inline AscendC::LocalTensor<float> PriorFixedSlot(uint32_t slot) {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE + slot * PRIOR_BYTES, PRIOR_WORDS);
}
__aicore__ inline AscendC::LocalTensor<T> OutAt(uint32_t slot) {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, OUT_BASE + slot * OUT_BYTES,
(sizeof(T) == sizeof(float) ? FRAME : TILE) * 5);
}
__aicore__ inline uint32_t CurrentOutSlot() const { return outSlot; }
__aicore__ inline void RawReady() { AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(rawSlot); }
__aicore__ inline void RawWait() { AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(rawSlot); }
__aicore__ inline void PriorReady() { AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2 + priorSlot); }
__aicore__ inline void PriorWait() { AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2 + priorSlot); }
__aicore__ inline void OutReady() { AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outSlot); }
__aicore__ inline void OutWait() { AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outSlot); }
__aicore__ inline void FreeRaw() {
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(rawSlot);
rawUsed |= 1U << rawSlot;
}
__aicore__ inline void FreePrior() {
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2 + priorSlot);
priorUsed |= 1U << priorSlot;
}
__aicore__ inline void FreeOut() {
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outSlot);
outUsed |= 1U << outSlot;
}
__aicore__ inline void Drain() {
for (uint32_t slot = 0; slot < DEPTH; ++slot) {
if (rawUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
if (priorUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2 + slot);
if (outUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
}
}
};


template <typename T, bool VA_GROUP38 = false>
class FireA2F5PairStorage {
static constexpr uint32_t RAW_BYTES = 1664U * 6U * sizeof(T);
static constexpr uint32_t PRIOR_BASE = 2U * RAW_BYTES;
static constexpr uint32_t PRIOR_BYTES = 4U * 1680U * sizeof(float);
static constexpr uint32_t WORK_BASE = PRIOR_BASE + PRIOR_BYTES + 256U;
static constexpr uint32_t WORK_BYTES = 74496U;
static constexpr uint32_t ONE_BASE = WORK_BASE + WORK_BYTES;
static constexpr uint32_t RF_BASE = ONE_BASE + 32U + (sizeof(T) == sizeof(half) ? 224U : 0U);
static constexpr uint32_t VA_BASE = RF_BASE + (sizeof(T) == sizeof(half) ? 39936U : 0U);
static constexpr uint32_t VA_WORDS = VA_GROUP38 ? 152U : 172U;
static_assert(!VA_GROUP38 || sizeof(T) == sizeof(float), "VA38 only FP32");
static_assert(VA_BASE + VA_WORDS * sizeof(uint64_t) == (VA_GROUP38 ? 182752U : (sizeof(T) == sizeof(float) ? 182912U : 183136U)), "fixed576 UB high water");
static_assert(VA_BASE + VA_WORDS * sizeof(uint64_t) < 188416U, "retain SDK UB reserve");
static_assert(WORK_BASE % 512U == 0U && (ONE_BASE - (sizeof(T) == sizeof(float) ? 0U : RF_BASE)) % 512U == 256U, "fixed phases");
uint32_t used = 0;
public:
__aicore__ inline AscendC::LocalTensor<T> RawFixedSlot(uint32_t slot) {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, slot * RAW_BYTES, 9984U);
}
__aicore__ inline AscendC::LocalTensor<T> OutAt(uint32_t slot) {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, slot * RAW_BYTES, 8320U);
}
__aicore__ inline AscendC::LocalTensor<float> Prior() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE, 6720U);
}
__aicore__ inline AscendC::LocalTensor<float> Work() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, WORK_BASE, WORK_BYTES / 4U);
}
__aicore__ inline AscendC::LocalTensor<float> Ones() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE, 8U);
}
__aicore__ inline AscendC::LocalTensor<float> Twos() {
static_assert(sizeof(T) == sizeof(float), "coordinate numerator2 only FP32");
constexpr uint32_t VA_END = VA_BASE + VA_WORDS * sizeof(uint64_t);
constexpr uint32_t TWO_BASE = ONE_BASE + ((VA_END - ONE_BASE + 511U) / 512U) * 512U;
static_assert(ONE_BASE == 181504U && VA_END == (VA_GROUP38 ? 182752U : 182912U) && TWO_BASE == 183040U, "qualified FP32 numerator address");
static_assert(TWO_BASE % 32U == 0U && TWO_BASE % 512U == 256U && TWO_BASE >= VA_END && TWO_BASE + 32U < 188416U, "separate numerator2 after fixed VA extent");
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, TWO_BASE, 8U);
}
__aicore__ inline AscendC::LocalTensor<float> RawFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, RF_BASE, 9984U);
}
__aicore__ inline AscendC::LocalTensor<float> PackedFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, RF_BASE, 8320U);
}
__aicore__ inline AscendC::LocalTensor<uint64_t> PackedVA() {
return AscendC::LocalTensor<uint64_t>(AscendC::TPosition::VECCALC, VA_BASE, VA_WORDS);
}
__aicore__ inline AscendC::LocalTensor<T> AllocSlot(uint32_t slot) {
if (used & (1U << slot)) {
AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
}
return RawFixedSlot(slot);
}
__aicore__ inline void PublishOutputAndRelease(uint32_t slot) {
AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);
used |= 1U << slot;
}
__aicore__ inline void Drain() {
for (uint32_t slot = 0; slot < 2; ++slot) if (used & (1U << slot)) {
AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
}
}
};


template <typename T, uint32_t N, bool SCORE_FUSED = false, bool ROTATE_PAIRS = false, bool QUALIFIED_SCORE_ROTATE = false, bool QUALIFIED_HALF_SCORE = false, bool COORD_DIV2 = false, bool VA_GROUP38 = false, uint32_t ROW = 832U>
class KernelFireA2F5Pair {
static_assert(!SCORE_FUSED || sizeof(T) == sizeof(float) || QUALIFIED_HALF_SCORE, "score fusion requires dtype qualification");
static_assert(!(SCORE_FUSED && ROTATE_PAIRS) || QUALIFIED_SCORE_ROTATE, "636 mechanisms are exclusive");
static_assert(!QUALIFIED_SCORE_ROTATE || (!0 && sizeof(T) == sizeof(float) && (N == 32768U || N == 32769U) && SCORE_FUSED && ROTATE_PAIRS), "only qualified float pair fusion rotation");
static_assert(!QUALIFIED_HALF_SCORE || (0 && sizeof(T) == sizeof(half) && SCORE_FUSED && !ROTATE_PAIRS), "only qualified half pair score fusion");
static_assert(!(QUALIFIED_SCORE_ROTATE && QUALIFIED_HALF_SCORE), "qualified pair modes are separate");
static_assert(N == 32768U || N == 32769U, "only qualifiedC40 B8 dimensions");
static constexpr uint32_t FRAME = 2U * ROW;
static constexpr uint32_t WORK_FRAME = ROW == 768U ? 1600U : FRAME;
static constexpr uint32_t STRIDE = WORK_FRAME + 16U, Q = WORK_FRAME / 8U;
static_assert(ROW != 768U || (!0 && sizeof(T) == sizeof(float) && N == 32768U && WORK_FRAME == 1600U && STRIDE == 1616U && Q == 200U && 107008U + 11U * STRIDE * 4U <= 181504U), "qualified Q200 physical Work prefix within fixed storage");
static_assert(ROW == 832U || (!0 && ROW == 768U && sizeof(T) == sizeof(float) && N == 32768U && SCORE_FUSED && ROTATE_PAIRS && QUALIFIED_SCORE_ROTATE && !QUALIFIED_HALF_SCORE && COORD_DIV2 && VA_GROUP38), "qualified mixed P5 medium row");
static_assert(!COORD_DIV2 || (!0 && sizeof(T) == sizeof(float) && N == 32768U && SCORE_FUSED && ROTATE_PAIRS && QUALIFIED_SCORE_ROTATE && !QUALIFIED_HALF_SCORE), "qualified N32768 FP32 coordinate Div2");
static_assert(FRAME % 128U == 0U && 4U * STRIDE % 64U == 0U, "exact coordinate and score Div spans");
static_assert(!VA_GROUP38 || (!0 && sizeof(T) == sizeof(float) && N == 32768U && SCORE_FUSED && ROTATE_PAIRS && QUALIFIED_SCORE_ROTATE && !QUALIFIED_HALF_SCORE && COORD_DIV2), "qualified N32768 FP32 VA38");
static constexpr bool BASIC2D = sizeof(T) == sizeof(float) && N == 32768U && COORD_DIV2 && VA_GROUP38;
static_assert(!BASIC2D || (SCORE_FUSED && ROTATE_PAIRS && QUALIFIED_SCORE_ROTATE && !QUALIFIED_HALF_SCORE && ((ROW == 832U && FRAME == 1664U && STRIDE == 1680U) || (ROW == 768U && FRAME == 1536U && WORK_FRAME == 1600U && STRIDE == 1616U))), "qualified FP32 N32768 basic pair copies");
static constexpr bool FIXED_TOKENS = sizeof(T) == sizeof(float) && N == 32768U && COORD_DIV2 && VA_GROUP38 && BASIC2D;
static_assert(!FIXED_TOKENS || (((ROW == 832U && FRAME == 1664U && STRIDE == 1680U) || (ROW == 768U && FRAME == 1536U && WORK_FRAME == 1600U && STRIDE == 1616U)) && SCORE_FUSED && ROTATE_PAIRS && QUALIFIED_SCORE_ROTATE && !QUALIFIED_HALF_SCORE), "qualified FP32 fixed four pair slots");
static constexpr bool PRIOR_OVERLAP = FIXED_TOKENS && ROW == 832U;
FireA2F5PairStorage<T, VA_GROUP38> mem;
template <uint32_t OFFSET>
__aicore__ static constexpr uint32_t VAOffset() {
if constexpr (!VA_GROUP38) return OFFSET;
else {
static_assert(OFFSET % 4U == 0U && OFFSET < 172U, "VA38 original group bounds");
constexpr uint32_t group = OFFSET / 4U;
static_assert(group < 3U || (group >= 6U && group < 9U) || (group >= 12U && group < 15U) || (group >= 18U && group < 43U), "VA38 original active group");
if constexpr (group < 3U) return 4U * (25U + group);
else if constexpr (group < 9U) return 4U * (5U * (group - 6U));
else if constexpr (group < 15U) return 4U * (1U + 5U * (group - 12U));
else if constexpr (group < 23U) return 4U * (28U + group - 18U);
else if constexpr (group < 28U) return 4U * (2U + 5U * (group - 23U));
else if constexpr (group < 33U) return 4U * (3U + 5U * (group - 28U));
else if constexpr (group < 38U) return 4U * (33U + group - 33U);
else return 4U * (4U + 5U * (group - 38U));
}
}
AscendC::GlobalTensor<T> rawGm, outGm;
AscendC::GlobalTensor<float> priorGm;
__aicore__ inline void StorePacked(AscendC::LocalTensor<uint64_t> dst, const uint64_t *address) {
#pragma unroll
for (uint32_t q = 0; q < 4; ++q) {
const uint64_t value = ((address[4*q] >> 5) & 0x1fffULL) |
(((address[4*q+1] >> 5) & 0x1fffULL) << 16) |
(((address[4*q+2] >> 5) & 0x1fffULL) << 32) |
(((address[4*q+3] >> 5) & 0x1fffULL) << 48);
dst.SetValue(q, value);
}
}
template <uint8_t REPEAT, uint16_t DST_STRIDE, uint16_t SRC_STRIDE>
__aicore__ inline void PackedTranspose(AscendC::LocalTensor<uint64_t> dst, AscendC::LocalTensor<uint64_t> src) {
auto *dstPacked = reinterpret_cast<__ubuf__ uint64_t *>(dst.GetPhyAddr());
auto *srcPacked = reinterpret_cast<__ubuf__ uint64_t *>(src.GetPhyAddr());
__cce_scalar::vld_va_reg(VA0, dstPacked, L128);
__cce_scalar::vld_va_reg(VA1, dstPacked, H128);
__cce_scalar::vld_va_reg(VA2, srcPacked, L128);
__cce_scalar::vld_va_reg(VA3, srcPacked, H128);
__cce_scalar::scatter_vnchwconv_b32(VA0, VA2, REPEAT, DST_STRIDE, SRC_STRIDE);
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitInputPhase() {
static_assert(SLOT < (sizeof(T) == sizeof(float) ? 2U : 1U) && PHASE < 3U, "fixed RAW3 slots");
uint64_t sourceBase;
if constexpr (sizeof(T) == sizeof(float)) sourceBase = reinterpret_cast<uint64_t>(mem.RawFixedSlot(SLOT).GetPhyAddr());
else sourceBase = reinterpret_cast<uint64_t>(mem.RawFloat().GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t source[16] = {
sourceBase + uint64_t(0 + 8*PHASE) * 4U,
sourceBase + uint64_t(48 + 8*PHASE) * 4U,
sourceBase + uint64_t(96 + 8*PHASE) * 4U,
sourceBase + uint64_t(144 + 8*PHASE) * 4U,
sourceBase + uint64_t(192 + 8*PHASE) * 4U,
sourceBase + uint64_t(240 + 8*PHASE) * 4U,
sourceBase + uint64_t(288 + 8*PHASE) * 4U,
sourceBase + uint64_t(336 + 8*PHASE) * 4U,
sourceBase + uint64_t(24 + 8*PHASE) * 4U,
sourceBase + uint64_t(72 + 8*PHASE) * 4U,
sourceBase + uint64_t(120 + 8*PHASE) * 4U,
sourceBase + uint64_t(168 + 8*PHASE) * 4U,
sourceBase + uint64_t(216 + 8*PHASE) * 4U,
sourceBase + uint64_t(264 + 8*PHASE) * 4U,
sourceBase + uint64_t(312 + 8*PHASE) * 4U,
sourceBase + uint64_t(360 + 8*PHASE) * 4U
};
uint64_t target[16] = {
workBase + uint64_t(((8*PHASE+0)%6)*STRIDE + (((8*PHASE+0)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+0)%6)*STRIDE + (((8*PHASE+0)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+1)%6)*STRIDE + (((8*PHASE+1)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+1)%6)*STRIDE + (((8*PHASE+1)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+2)%6)*STRIDE + (((8*PHASE+2)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+2)%6)*STRIDE + (((8*PHASE+2)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+3)%6)*STRIDE + (((8*PHASE+3)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+3)%6)*STRIDE + (((8*PHASE+3)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+4)%6)*STRIDE + (((8*PHASE+4)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+4)%6)*STRIDE + (((8*PHASE+4)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+5)%6)*STRIDE + (((8*PHASE+5)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+5)%6)*STRIDE + (((8*PHASE+5)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+6)%6)*STRIDE + (((8*PHASE+6)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+6)%6)*STRIDE + (((8*PHASE+6)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+7)%6)*STRIDE + (((8*PHASE+7)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+7)%6)*STRIDE + (((8*PHASE+7)/6)+4)*Q) * 4U
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[VAOffset<4*PHASE>()], target);
if constexpr (!VA_GROUP38 || PHASE == 0U) StorePacked(mem.PackedVA()[VAOffset<24+SLOT*24+4*PHASE>()], source);
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitOutputPhase() {
static_assert(SLOT < (sizeof(T) == sizeof(float) ? 2U : 1U) && PHASE < 5U, "fixed output slots");
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t outputBase;
if constexpr (sizeof(T) == sizeof(float)) outputBase = reinterpret_cast<uint64_t>(mem.OutAt(SLOT).GetPhyAddr());
else outputBase = reinterpret_cast<uint64_t>(mem.PackedFloat().GetPhyAddr());
uint64_t source[16] = {
workBase + uint64_t(((8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 8) * 4U
};
uint64_t target[16] = {
outputBase + uint64_t((5*0+PHASE)*8) * 4U,
outputBase + uint64_t((5*8+PHASE)*8) * 4U,
outputBase + uint64_t((5*1+PHASE)*8) * 4U,
outputBase + uint64_t((5*9+PHASE)*8) * 4U,
outputBase + uint64_t((5*2+PHASE)*8) * 4U,
outputBase + uint64_t((5*10+PHASE)*8) * 4U,
outputBase + uint64_t((5*3+PHASE)*8) * 4U,
outputBase + uint64_t((5*11+PHASE)*8) * 4U,
outputBase + uint64_t((5*4+PHASE)*8) * 4U,
outputBase + uint64_t((5*12+PHASE)*8) * 4U,
outputBase + uint64_t((5*5+PHASE)*8) * 4U,
outputBase + uint64_t((5*13+PHASE)*8) * 4U,
outputBase + uint64_t((5*6+PHASE)*8) * 4U,
outputBase + uint64_t((5*14+PHASE)*8) * 4U,
outputBase + uint64_t((5*7+PHASE)*8) * 4U,
outputBase + uint64_t((5*15+PHASE)*8) * 4U
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[VAOffset<72+4*PHASE>()], source);
if constexpr (!VA_GROUP38 || PHASE == 0U) StorePacked(mem.PackedVA()[VAOffset<92+SLOT*20+4*PHASE>()], target);
}
template <uint32_t PHASE>
__aicore__ inline void InitPriorPhase() {
const uint64_t priorBase = reinterpret_cast<uint64_t>(mem.Prior().GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t source[16] = {
priorBase + uint64_t(0+8*PHASE) * 4U,
priorBase + uint64_t(40+8*PHASE) * 4U,
priorBase + uint64_t(80+8*PHASE) * 4U,
priorBase + uint64_t(120+8*PHASE) * 4U,
priorBase + uint64_t(160+8*PHASE) * 4U,
priorBase + uint64_t(200+8*PHASE) * 4U,
priorBase + uint64_t(240+8*PHASE) * 4U,
priorBase + uint64_t(280+8*PHASE) * 4U,
priorBase + uint64_t(0+8*PHASE) * 4U,
priorBase + uint64_t(40+8*PHASE) * 4U,
priorBase + uint64_t(80+8*PHASE) * 4U,
priorBase + uint64_t(120+8*PHASE) * 4U,
priorBase + uint64_t(160+8*PHASE) * 4U,
priorBase + uint64_t(200+8*PHASE) * 4U,
priorBase + uint64_t(240+8*PHASE) * 4U,
priorBase + uint64_t(280+8*PHASE) * 4U
};
uint64_t target[16] = {
workBase + uint64_t((6+(8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + (ROW / 8U)) * 4U,
workBase + uint64_t((6+(8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + (ROW / 8U)) * 4U,
workBase + uint64_t((6+(8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + (ROW / 8U)) * 4U,
workBase + uint64_t((6+(8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + (ROW / 8U)) * 4U,
workBase + uint64_t((6+(8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + (ROW / 8U)) * 4U,
workBase + uint64_t((6+(8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + (ROW / 8U)) * 4U,
workBase + uint64_t((6+(8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + (ROW / 8U)) * 4U,
workBase + uint64_t((6+(8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + (ROW / 8U)) * 4U
};
StorePacked(mem.PackedVA()[VAOffset<132+4*PHASE>()], target);
if constexpr (!VA_GROUP38 || PHASE == 0U) StorePacked(mem.PackedVA()[VAOffset<152+4*PHASE>()], source);
}
__aicore__ inline void InitPackedVA() {
InitInputPhase<0, 0>();
InitInputPhase<0, 1>();
InitInputPhase<0, 2>();
InitOutputPhase<0, 0>();
InitOutputPhase<0, 1>();
InitOutputPhase<0, 2>();
InitOutputPhase<0, 3>();
InitOutputPhase<0, 4>();
if constexpr (sizeof(T) == sizeof(float)) {
InitInputPhase<1, 0>();
InitInputPhase<1, 1>();
InitInputPhase<1, 2>();
InitOutputPhase<1, 0>();
InitOutputPhase<1, 1>();
InitOutputPhase<1, 2>();
InitOutputPhase<1, 3>();
InitOutputPhase<1, 4>();
}
InitPriorPhase<0>();
InitPriorPhase<1>();
InitPriorPhase<2>();
InitPriorPhase<3>();
InitPriorPhase<4>();
AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
if constexpr (VA_GROUP38) {
auto *entries=reinterpret_cast<__ubuf__ int16_t *>(mem.PackedVA().GetPhyAddr());
AscendC::SetMaskCount();
AscendC::SetVectorMask<int16_t, AscendC::MaskMode::COUNTER>(80U);
#pragma unroll
for (uint32_t phase=1U;phase<5U;++phase)
asc_add_scalar(entries+phase*80U,entries,int16_t(phase),
uint8_t(1),uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetMaskNorm();
AscendC::ResetMask();
}
}
template <uint32_t SLOT>
__aicore__ inline void Input() {
PackedTranspose<FRAME / 64U, 1, 48>(mem.PackedVA()[VAOffset<0>()], mem.PackedVA()[VAOffset<24+SLOT*24+0>()]);
PackedTranspose<FRAME / 64U, 1, 48>(mem.PackedVA()[VAOffset<4>()], mem.PackedVA()[VAOffset<24+SLOT*24+4>()]);
PackedTranspose<FRAME / 64U, 1, 48>(mem.PackedVA()[VAOffset<8>()], mem.PackedVA()[VAOffset<24+SLOT*24+8>()]);
}
template <uint32_t SLOT>
__aicore__ inline void Output() {
PackedTranspose<ROW / 64U, 80, 2>(mem.PackedVA()[VAOffset<92+SLOT*20+0>()], mem.PackedVA()[VAOffset<72+0>()]);
PackedTranspose<ROW / 64U, 80, 2>(mem.PackedVA()[VAOffset<92+SLOT*20+4>()], mem.PackedVA()[VAOffset<72+4>()]);
PackedTranspose<ROW / 64U, 80, 2>(mem.PackedVA()[VAOffset<92+SLOT*20+8>()], mem.PackedVA()[VAOffset<72+8>()]);
PackedTranspose<ROW / 64U, 80, 2>(mem.PackedVA()[VAOffset<92+SLOT*20+12>()], mem.PackedVA()[VAOffset<72+12>()]);
PackedTranspose<ROW / 64U, 80, 2>(mem.PackedVA()[VAOffset<92+SLOT*20+16>()], mem.PackedVA()[VAOffset<72+16>()]);
}
public:
__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded) {
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
if constexpr (ROW == 768U) { if (block >= 42U) return; }
else { if (block >= 39U) return; }
const uint32_t begin = ROW == 768U ? block * 768U : block * 832U;
rawGm.SetGlobalBuffer((__gm__ T *)raw);
priorGm.SetGlobalBuffer((__gm__ float *)priors);
outGm.SetGlobalBuffer((__gm__ T *)decoded);
const AscendC::DataCopyExtParams rp{2, ROW*6U*uint32_t(sizeof(T)), (N-ROW)*6U*uint32_t(sizeof(T)), 0, 0};
const AscendC::DataCopyExtParams pp{1, ROW*5U*4U, 0, 0, 0};
const uint32_t cohort = ROTATE_PAIRS ? (block & 3U) : 0U;
auto firstRaw = mem.AllocSlot(0);
auto p = mem.Prior();
if constexpr (BASIC2D) { AscendC::DataCopy(firstRaw, rawGm[(2U*cohort*N+begin)*6U], AscendC::DataCopyParams{2U,uint16_t(ROW*24U/32U),uint16_t((N-ROW)*24U/32U),0U}); } else
AscendC::DataCopyPad(firstRaw, rawGm[(2U*cohort*N+begin)*6U], rp, AscendC::DataCopyPadExtParams<T>{false,0,0,0});
if constexpr (PRIOR_OVERLAP) AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
AscendC::DataCopyPad(p, priorGm[begin*5U], pp, AscendC::DataCopyPadExtParams<float>{false,0,0,0});
if constexpr (!PRIOR_OVERLAP) AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
auto one = mem.Ones();
AscendC::Duplicate(one, 1.0f, 8U);
if constexpr (COORD_DIV2) AscendC::Duplicate(mem.Twos(), 2.0f, 8U);
AscendC::LocalTensor<float> f[11];
for (uint32_t k=0;k<11;++k) f[k]=mem.Work()[k*STRIDE];
InitPackedVA();
if constexpr (!PRIOR_OVERLAP) {
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
PackedTranspose<ROW / 64U, 1, 40>(mem.PackedVA()[VAOffset<132+0>()], mem.PackedVA()[VAOffset<152+0>()]);
PackedTranspose<ROW / 64U, 1, 40>(mem.PackedVA()[VAOffset<132+4>()], mem.PackedVA()[VAOffset<152+4>()]);
PackedTranspose<ROW / 64U, 1, 40>(mem.PackedVA()[VAOffset<132+8>()], mem.PackedVA()[VAOffset<152+8>()]);
PackedTranspose<ROW / 64U, 1, 40>(mem.PackedVA()[VAOffset<132+12>()], mem.PackedVA()[VAOffset<152+12>()]);
PackedTranspose<ROW / 64U, 1, 40>(mem.PackedVA()[VAOffset<132+16>()], mem.PackedVA()[VAOffset<152+16>()]);
AscendC::PipeBarrier<PIPE_V>();
}
if constexpr (sizeof(T) != sizeof(float)) {
AscendC::DataCopy(p, f[8], STRIDE);
AscendC::DataCopy(p[STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[2*STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[3*STRIDE], f[8], STRIDE);
AscendC::PipeBarrier<PIPE_V>();
}
for (uint32_t iteration=0;iteration<4;++iteration) {
const uint32_t pair = ROTATE_PAIRS ? ((iteration+cohort)&3U) : iteration;
const uint32_t slot=iteration&1U;
const uint32_t pos=pair*2U*N+begin;
auto r=firstRaw;
if (iteration!=0) {
if constexpr (FIXED_TOKENS) {
if (iteration >= 2U) {
AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
}
r=mem.RawFixedSlot(slot);
} else
r=mem.AllocSlot(slot);
if constexpr (BASIC2D) { AscendC::DataCopy(r, rawGm[pos*6U], AscendC::DataCopyParams{2U,uint16_t(ROW*24U/32U),uint16_t((N-ROW)*24U/32U),0U}); } else
AscendC::DataCopyPad(r,rawGm[pos*6U],rp,AscendC::DataCopyPadExtParams<T>{false,0,0,0});
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot);
}
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);
AscendC::LocalTensor<float> rf;
if constexpr(sizeof(T)==sizeof(float)) rf=r.template ReinterpretCast<float>();
else {rf=mem.RawFloat();AscendC::Cast(rf,r,AscendC::RoundMode::CAST_NONE,FRAME*6U);}
AscendC::Muls(rf,rf,-1.0f,FRAME*6U);
AscendC::Exp(rf,rf,FRAME*6U);
if constexpr (PRIOR_OVERLAP) {
if (iteration == 0U) {
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
PackedTranspose<ROW / 64U, 1, 40>(mem.PackedVA()[VAOffset<132+0>()], mem.PackedVA()[VAOffset<152+0>()]);
PackedTranspose<ROW / 64U, 1, 40>(mem.PackedVA()[VAOffset<132+4>()], mem.PackedVA()[VAOffset<152+4>()]);
PackedTranspose<ROW / 64U, 1, 40>(mem.PackedVA()[VAOffset<132+8>()], mem.PackedVA()[VAOffset<152+8>()]);
PackedTranspose<ROW / 64U, 1, 40>(mem.PackedVA()[VAOffset<132+12>()], mem.PackedVA()[VAOffset<152+12>()]);
PackedTranspose<ROW / 64U, 1, 40>(mem.PackedVA()[VAOffset<132+16>()], mem.PackedVA()[VAOffset<152+16>()]);
}
}
AscendC::Adds(rf,rf,1.0f,FRAME*6U);
const AscendC::BinaryRepeatParams divBinary{1,0,1,8,0,8};
if constexpr (!SCORE_FUSED) AscendC::Div<float>(rf,one,rf,uint64_t(64),uint8_t(FRAME*6U/64U),divBinary);
AscendC::PipeBarrier<PIPE_V>();
if constexpr(sizeof(T)==sizeof(float)) {if(slot==0)Input<0>();else Input<1>();}
else Input<0>();
AscendC::PipeBarrier<PIPE_V>();
if constexpr (SCORE_FUSED) {
AscendC::Mul(f[4],f[4],f[5],WORK_FRAME);
if constexpr (COORD_DIV2) {
AscendC::Div<float>(mem.Work(),mem.Twos(),mem.Work(),uint64_t(64),uint8_t(4U*STRIDE/64U),divBinary);
if constexpr (VA_GROUP38) {
static_assert(!0 && sizeof(T) == sizeof(float) && N == 32768U && ((ROW == 832U && WORK_FRAME == 1664U && STRIDE == 1680U) || (ROW == 768U && WORK_FRAME == 1600U && STRIDE == 1616U)), "qualified P5 raw score Div Normal64");
auto *scorePtr = reinterpret_cast<__ubuf__ float *>(f[4].GetPhyAddr());
auto *onePtr = reinterpret_cast<__ubuf__ float *>(one.GetPhyAddr());
asc_div(scorePtr, onePtr, scorePtr, uint8_t(WORK_FRAME/64U),
uint8_t(1),uint8_t(0),uint8_t(1),uint8_t(8),uint8_t(0),uint8_t(8));
} else {
AscendC::Div<float>(f[4],one,f[4],uint64_t(64),uint8_t(WORK_FRAME/64U),divBinary);
}
} else {
AscendC::Div<float>(mem.Work(),one,mem.Work(),uint64_t(64),uint8_t(132),divBinary);
}
AscendC::PipeBarrier<PIPE_V>();
}
if constexpr (!COORD_DIV2) AscendC::Muls(f[0],f[0],2.0f,4*STRIDE);
if constexpr (sizeof(T) == sizeof(float) && N == 32768U && COORD_DIV2 && VA_GROUP38 && ROW == 832U) {
auto *mathPtr = reinterpret_cast<__ubuf__ float *>(mem.Work().GetPhyAddr());
AscendC::SetMaskCount();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(2U*STRIDE);
asc_add_scalar(mathPtr,mathPtr,-0.5f,uint8_t(1),uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));
asc_mul(mathPtr+2U*STRIDE,mathPtr+2U*STRIDE,mathPtr+2U*STRIDE,uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(8),uint8_t(8),uint8_t(8));
AscendC::PipeBarrier<PIPE_V>();
asc_add(mathPtr,mathPtr,mathPtr+6U*STRIDE,uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(8),uint8_t(8),uint8_t(8));
asc_mul(mathPtr+2U*STRIDE,mathPtr+2U*STRIDE,mathPtr+9U*STRIDE,uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(8),uint8_t(8),uint8_t(8));
} else {
AscendC::Adds(f[0],f[0],-0.5f,2*STRIDE);
AscendC::Add(f[0],f[0],f[6],2*STRIDE);
AscendC::Mul(f[2],f[2],f[2],2*STRIDE);
AscendC::Mul(f[2],f[2],f[9],2*STRIDE);
}
if constexpr (sizeof(T) == sizeof(float)) {
AscendC::PipeBarrier<PIPE_V>();
auto *wPtr = reinterpret_cast<__ubuf__ float *>(mem.Work().GetPhyAddr());
AscendC::SetMaskCount();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>((COORD_DIV2 && VA_GROUP38 && N == 32768U) ? WORK_FRAME : STRIDE);
#pragma unroll
for (uint32_t k=0;k<4;++k)
asc_mul(wPtr+k*STRIDE, wPtr+k*STRIDE, wPtr+8*STRIDE,
uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(8),uint8_t(8),uint8_t(8));
AscendC::SetMaskNorm();
AscendC::ResetMask();
} else {
AscendC::Mul(f[0],f[0],p,4*STRIDE);
}
if constexpr (!SCORE_FUSED) AscendC::Mul(f[4],f[4],f[5],FRAME);
AscendC::PipeBarrier<PIPE_V>();
if constexpr(sizeof(T)==sizeof(float)) {if(slot==0)Output<0>();else Output<1>();}
else Output<0>();
AscendC::PipeBarrier<PIPE_V>();
auto out=mem.OutAt(slot);
if constexpr(sizeof(T)!=sizeof(float)) {
AscendC::Cast(out,mem.PackedFloat(),AscendC::RoundMode::CAST_RINT,8320U);
AscendC::PipeBarrier<PIPE_V>();
}
AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot);
AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
const AscendC::DataCopyExtParams op{2,ROW*5U*uint32_t(sizeof(T)),0,(N-ROW)*5U*uint32_t(sizeof(T)),0};
if constexpr (BASIC2D) { AscendC::DataCopy(outGm[pos*5U], out, AscendC::DataCopyParams{2U,uint16_t(ROW*20U/32U),0U,uint16_t((N-ROW)*20U/32U)}); } else
AscendC::DataCopyPad(outGm[pos*5U],out,op);
if constexpr (FIXED_TOKENS) {
AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);
} else
mem.PublishOutputAndRelease(slot);
}
if constexpr (FIXED_TOKENS) {
AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
} else
mem.Drain();
}
};


template <typename T, uint32_t N>
class KernelFireA2F5Tail320 {
static_assert(!0 && sizeof(T) == sizeof(float) && N == 32768U, "qualified FP32 N32768 tail320 pairs");
static constexpr uint32_t COUNT = 320U, FRAME = 640U, STRIDE = 656U, Q = 80U;
static_assert(COUNT * 2U == FRAME && FRAME % 128U == 0U && Q == FRAME / 8U && 39U * 832U + COUNT == N, "exact tail pair geometry");
FireA2F5FrameStorage<T, FRAME> mem;
AscendC::GlobalTensor<T> rawGm, outGm;
AscendC::GlobalTensor<float> priorGm;
__aicore__ inline void StorePacked(AscendC::LocalTensor<uint64_t> dst, const uint64_t *address) {
#pragma unroll
for (uint32_t q = 0; q < 4; ++q) {
const uint64_t value = ((address[4*q] >> 5) & 0x1fffULL) |
(((address[4*q+1] >> 5) & 0x1fffULL) << 16) |
(((address[4*q+2] >> 5) & 0x1fffULL) << 32) |
(((address[4*q+3] >> 5) & 0x1fffULL) << 48);
dst.SetValue(q, value);
}
}
template <uint8_t REPEAT, uint16_t DST_STRIDE, uint16_t SRC_STRIDE>
__aicore__ inline void PackedTranspose(AscendC::LocalTensor<uint64_t> dst, AscendC::LocalTensor<uint64_t> src) {
auto *dstPacked = reinterpret_cast<__ubuf__ uint64_t *>(dst.GetPhyAddr());
auto *srcPacked = reinterpret_cast<__ubuf__ uint64_t *>(src.GetPhyAddr());
__cce_scalar::vld_va_reg(VA0, dstPacked, L128);
__cce_scalar::vld_va_reg(VA1, dstPacked, H128);
__cce_scalar::vld_va_reg(VA2, srcPacked, L128);
__cce_scalar::vld_va_reg(VA3, srcPacked, H128);
__cce_scalar::scatter_vnchwconv_b32(VA0, VA2, REPEAT, DST_STRIDE, SRC_STRIDE);
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitInputPhase() {
static_assert(SLOT < (sizeof(T) == sizeof(float) ? 2U : 1U) && PHASE < 3U, "fixed RAW3 slots");
uint64_t sourceBase;
if constexpr (sizeof(T) == sizeof(float)) sourceBase = reinterpret_cast<uint64_t>(mem.RawFixedSlot(SLOT).GetPhyAddr());
else sourceBase = reinterpret_cast<uint64_t>(mem.RawFloat().GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t source[16] = {
sourceBase + uint64_t(0 + 8*PHASE) * 4U,
sourceBase + uint64_t(48 + 8*PHASE) * 4U,
sourceBase + uint64_t(96 + 8*PHASE) * 4U,
sourceBase + uint64_t(144 + 8*PHASE) * 4U,
sourceBase + uint64_t(192 + 8*PHASE) * 4U,
sourceBase + uint64_t(240 + 8*PHASE) * 4U,
sourceBase + uint64_t(288 + 8*PHASE) * 4U,
sourceBase + uint64_t(336 + 8*PHASE) * 4U,
sourceBase + uint64_t(24 + 8*PHASE) * 4U,
sourceBase + uint64_t(72 + 8*PHASE) * 4U,
sourceBase + uint64_t(120 + 8*PHASE) * 4U,
sourceBase + uint64_t(168 + 8*PHASE) * 4U,
sourceBase + uint64_t(216 + 8*PHASE) * 4U,
sourceBase + uint64_t(264 + 8*PHASE) * 4U,
sourceBase + uint64_t(312 + 8*PHASE) * 4U,
sourceBase + uint64_t(360 + 8*PHASE) * 4U
};
uint64_t target[16] = {
workBase + uint64_t(((8*PHASE+0)%6)*STRIDE + (((8*PHASE+0)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+0)%6)*STRIDE + (((8*PHASE+0)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+1)%6)*STRIDE + (((8*PHASE+1)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+1)%6)*STRIDE + (((8*PHASE+1)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+2)%6)*STRIDE + (((8*PHASE+2)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+2)%6)*STRIDE + (((8*PHASE+2)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+3)%6)*STRIDE + (((8*PHASE+3)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+3)%6)*STRIDE + (((8*PHASE+3)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+4)%6)*STRIDE + (((8*PHASE+4)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+4)%6)*STRIDE + (((8*PHASE+4)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+5)%6)*STRIDE + (((8*PHASE+5)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+5)%6)*STRIDE + (((8*PHASE+5)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+6)%6)*STRIDE + (((8*PHASE+6)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+6)%6)*STRIDE + (((8*PHASE+6)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+7)%6)*STRIDE + (((8*PHASE+7)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+7)%6)*STRIDE + (((8*PHASE+7)/6)+4)*Q) * 4U
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[4*PHASE], target);
StorePacked(mem.PackedVA()[24+SLOT*24+4*PHASE], source);
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitOutputPhase() {
static_assert(SLOT < (sizeof(T) == sizeof(float) ? 2U : 1U) && PHASE < 5U, "fixed output slots");
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t outputBase;
if constexpr (sizeof(T) == sizeof(float)) outputBase = reinterpret_cast<uint64_t>(mem.OutAt(SLOT).GetPhyAddr());
else outputBase = reinterpret_cast<uint64_t>(mem.PackedFloat().GetPhyAddr());
uint64_t source[16] = {
workBase + uint64_t(((8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 8) * 4U
};
uint64_t target[16] = {
outputBase + uint64_t((5*0+PHASE)*8) * 4U,
outputBase + uint64_t((5*8+PHASE)*8) * 4U,
outputBase + uint64_t((5*1+PHASE)*8) * 4U,
outputBase + uint64_t((5*9+PHASE)*8) * 4U,
outputBase + uint64_t((5*2+PHASE)*8) * 4U,
outputBase + uint64_t((5*10+PHASE)*8) * 4U,
outputBase + uint64_t((5*3+PHASE)*8) * 4U,
outputBase + uint64_t((5*11+PHASE)*8) * 4U,
outputBase + uint64_t((5*4+PHASE)*8) * 4U,
outputBase + uint64_t((5*12+PHASE)*8) * 4U,
outputBase + uint64_t((5*5+PHASE)*8) * 4U,
outputBase + uint64_t((5*13+PHASE)*8) * 4U,
outputBase + uint64_t((5*6+PHASE)*8) * 4U,
outputBase + uint64_t((5*14+PHASE)*8) * 4U,
outputBase + uint64_t((5*7+PHASE)*8) * 4U,
outputBase + uint64_t((5*15+PHASE)*8) * 4U
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[72+4*PHASE], source);
StorePacked(mem.PackedVA()[92+SLOT*20+4*PHASE], target);
}
template <uint32_t PHASE>
__aicore__ inline void InitPriorPhase() {
const uint64_t priorBase = reinterpret_cast<uint64_t>(mem.PriorFixedSlot(0).GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t source[16] = {
priorBase + uint64_t(0+8*PHASE) * 4U,
priorBase + uint64_t(40+8*PHASE) * 4U,
priorBase + uint64_t(80+8*PHASE) * 4U,
priorBase + uint64_t(120+8*PHASE) * 4U,
priorBase + uint64_t(160+8*PHASE) * 4U,
priorBase + uint64_t(200+8*PHASE) * 4U,
priorBase + uint64_t(240+8*PHASE) * 4U,
priorBase + uint64_t(280+8*PHASE) * 4U,
priorBase + uint64_t(0+8*PHASE) * 4U,
priorBase + uint64_t(40+8*PHASE) * 4U,
priorBase + uint64_t(80+8*PHASE) * 4U,
priorBase + uint64_t(120+8*PHASE) * 4U,
priorBase + uint64_t(160+8*PHASE) * 4U,
priorBase + uint64_t(200+8*PHASE) * 4U,
priorBase + uint64_t(240+8*PHASE) * 4U,
priorBase + uint64_t(280+8*PHASE) * 4U
};
uint64_t target[16] = {
workBase + uint64_t((6+(8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 40) * 4U,
workBase + uint64_t((6+(8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 40) * 4U,
workBase + uint64_t((6+(8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 40) * 4U,
workBase + uint64_t((6+(8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 40) * 4U,
workBase + uint64_t((6+(8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 40) * 4U,
workBase + uint64_t((6+(8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 40) * 4U,
workBase + uint64_t((6+(8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 40) * 4U,
workBase + uint64_t((6+(8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 40) * 4U
};
StorePacked(mem.PackedVA()[132+4*PHASE], target);
StorePacked(mem.PackedVA()[152+4*PHASE], source);
}
__aicore__ inline void InitPackedVA() {
InitInputPhase<0, 0>();
InitInputPhase<0, 1>();
InitInputPhase<0, 2>();
InitOutputPhase<0, 0>();
InitOutputPhase<0, 1>();
InitOutputPhase<0, 2>();
InitOutputPhase<0, 3>();
InitOutputPhase<0, 4>();
if constexpr (sizeof(T) == sizeof(float)) {
InitInputPhase<1, 0>();
InitInputPhase<1, 1>();
InitInputPhase<1, 2>();
InitOutputPhase<1, 0>();
InitOutputPhase<1, 1>();
InitOutputPhase<1, 2>();
InitOutputPhase<1, 3>();
InitOutputPhase<1, 4>();
}
InitPriorPhase<0>();
InitPriorPhase<1>();
InitPriorPhase<2>();
InitPriorPhase<3>();
InitPriorPhase<4>();
AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
}
template <uint32_t SLOT>
__aicore__ inline void Input() {
PackedTranspose<10, 1, 48>(mem.PackedVA()[0], mem.PackedVA()[24+SLOT*24+0]);
PackedTranspose<10, 1, 48>(mem.PackedVA()[4], mem.PackedVA()[24+SLOT*24+4]);
PackedTranspose<10, 1, 48>(mem.PackedVA()[8], mem.PackedVA()[24+SLOT*24+8]);
}
template <uint32_t SLOT>
__aicore__ inline void Output() {
PackedTranspose<5, 80, 2>(mem.PackedVA()[92+SLOT*20+0], mem.PackedVA()[72+0]);
PackedTranspose<5, 80, 2>(mem.PackedVA()[92+SLOT*20+4], mem.PackedVA()[72+4]);
PackedTranspose<5, 80, 2>(mem.PackedVA()[92+SLOT*20+8], mem.PackedVA()[72+8]);
PackedTranspose<5, 80, 2>(mem.PackedVA()[92+SLOT*20+12], mem.PackedVA()[72+12]);
PackedTranspose<5, 80, 2>(mem.PackedVA()[92+SLOT*20+16], mem.PackedVA()[72+16]);
}
public:
__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded) {
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
if (block != 39U) return;
constexpr uint32_t begin = 32448U;
rawGm.SetGlobalBuffer((__gm__ T *)raw);
priorGm.SetGlobalBuffer((__gm__ float *)priors);
outGm.SetGlobalBuffer((__gm__ T *)decoded);
const AscendC::DataCopyExtParams rp{2, COUNT * 6U * uint32_t(sizeof(T)), (N - COUNT) * 6U * uint32_t(sizeof(T)), 0, 0};
const AscendC::DataCopyExtParams pp{1, COUNT * 5U * uint32_t(sizeof(float)), 0, 0, 0};
auto firstRaw = mem.AllocRaw();
auto p = mem.AllocPrior();
AscendC::DataCopy(firstRaw, rawGm[begin * 6U], AscendC::DataCopyParams{2U,240U,24336U,0U});
AscendC::DataCopyPad(p, priorGm[begin * 5U], pp, AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
mem.RawReady();
mem.PriorReady();
auto one = mem.Ones();
AscendC::Duplicate(one, 1.0f, FRAME * 6U);
AscendC::LocalTensor<float> f[11];
for (uint32_t k = 0; k < 11; ++k) f[k] = mem.Work()[k * STRIDE];
InitPackedVA();
mem.PriorWait();
PackedTranspose<5, 1, 40>(mem.PackedVA()[132+0], mem.PackedVA()[152+0]);
PackedTranspose<5, 1, 40>(mem.PackedVA()[132+4], mem.PackedVA()[152+4]);
PackedTranspose<5, 1, 40>(mem.PackedVA()[132+8], mem.PackedVA()[152+8]);
PackedTranspose<5, 1, 40>(mem.PackedVA()[132+12], mem.PackedVA()[152+12]);
PackedTranspose<5, 1, 40>(mem.PackedVA()[132+16], mem.PackedVA()[152+16]);
AscendC::PipeBarrier<PIPE_V>();
AscendC::DataCopy(p, f[8], STRIDE);
AscendC::DataCopy(p[STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[2 * STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[3 * STRIDE], f[8], STRIDE);
AscendC::PipeBarrier<PIPE_V>();
for (uint32_t iteration = 0; iteration < 4U; ++iteration) {
const uint32_t slot = iteration & 1U;
const uint32_t pos = iteration * 2U * N + begin;
auto r = firstRaw;
if (iteration != 0U) {
r = mem.AllocRaw();
AscendC::DataCopy(r, rawGm[pos * 6U], AscendC::DataCopyParams{2U,240U,24336U,0U});
mem.RawReady();
}
mem.RawWait();
auto rf = r.template ReinterpretCast<float>();
AscendC::Muls(rf, rf, -1.0f, FRAME * 6U);
AscendC::Exp(rf, rf, FRAME * 6U);
AscendC::Adds(rf, rf, 1.0f, FRAME * 6U);
AscendC::Div(rf, one, rf, FRAME * 6U);
AscendC::PipeBarrier<PIPE_V>();
if (slot == 0U) Input<0>(); else Input<1>();
AscendC::PipeBarrier<PIPE_V>();
AscendC::Muls(f[0], f[0], 2.0f, 4 * STRIDE);
AscendC::Adds(f[0], f[0], -0.5f, 2 * STRIDE);
AscendC::Add(f[0], f[0], f[6], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[2], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[9], 2 * STRIDE);
AscendC::Mul(f[0], f[0], p, 4 * STRIDE);
AscendC::Mul(f[4], f[4], f[5], FRAME);
auto out = mem.AllocOut();
AscendC::PipeBarrier<PIPE_V>();
if (slot == 0U) Output<0>(); else Output<1>();
AscendC::PipeBarrier<PIPE_V>();
mem.OutReady();
mem.OutWait();
const AscendC::DataCopyExtParams op{2, COUNT * 5U * uint32_t(sizeof(T)), 0, (N - COUNT) * 5U * uint32_t(sizeof(T)), 0};
AscendC::DataCopy(outGm[pos * 5U], out, AscendC::DataCopyParams{2U,200U,0U,20280U});
mem.FreeOut();
mem.FreeRaw();
}
mem.FreePrior();
mem.Drain();
}
};


template <typename T, uint32_t N>
class KernelFireA2F5Tail512 {
static_assert(!0 && sizeof(T) == sizeof(float) && N == 32768U, "qualified FP32 N32768 tail512 span43 pairs");
static constexpr uint32_t COUNT = 512U, FRAME = 1024U, STRIDE = 1040U, Q = 128U;
static_assert(COUNT * 2U == FRAME && FRAME % 128U == 0U && Q == FRAME / 8U && 42U * 768U + COUNT == N, "exact tail pair geometry");
FireA2F5FrameStorage<T, FRAME> mem;
AscendC::GlobalTensor<T> rawGm, outGm;
AscendC::GlobalTensor<float> priorGm;
__aicore__ inline void StorePacked(AscendC::LocalTensor<uint64_t> dst, const uint64_t *address) {
#pragma unroll
for (uint32_t q = 0; q < 4; ++q) {
const uint64_t value = ((address[4*q] >> 5) & 0x1fffULL) |
(((address[4*q+1] >> 5) & 0x1fffULL) << 16) |
(((address[4*q+2] >> 5) & 0x1fffULL) << 32) |
(((address[4*q+3] >> 5) & 0x1fffULL) << 48);
dst.SetValue(q, value);
}
}
template <uint8_t REPEAT, uint16_t DST_STRIDE, uint16_t SRC_STRIDE>
__aicore__ inline void PackedTranspose(AscendC::LocalTensor<uint64_t> dst, AscendC::LocalTensor<uint64_t> src) {
auto *dstPacked = reinterpret_cast<__ubuf__ uint64_t *>(dst.GetPhyAddr());
auto *srcPacked = reinterpret_cast<__ubuf__ uint64_t *>(src.GetPhyAddr());
__cce_scalar::vld_va_reg(VA0, dstPacked, L128);
__cce_scalar::vld_va_reg(VA1, dstPacked, H128);
__cce_scalar::vld_va_reg(VA2, srcPacked, L128);
__cce_scalar::vld_va_reg(VA3, srcPacked, H128);
__cce_scalar::scatter_vnchwconv_b32(VA0, VA2, REPEAT, DST_STRIDE, SRC_STRIDE);
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitInputPhase() {
static_assert(SLOT < (sizeof(T) == sizeof(float) ? 2U : 1U) && PHASE < 3U, "fixed RAW3 slots");
uint64_t sourceBase;
if constexpr (sizeof(T) == sizeof(float)) sourceBase = reinterpret_cast<uint64_t>(mem.RawFixedSlot(SLOT).GetPhyAddr());
else sourceBase = reinterpret_cast<uint64_t>(mem.RawFloat().GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t source[16] = {
sourceBase + uint64_t(0 + 8*PHASE) * 4U,
sourceBase + uint64_t(48 + 8*PHASE) * 4U,
sourceBase + uint64_t(96 + 8*PHASE) * 4U,
sourceBase + uint64_t(144 + 8*PHASE) * 4U,
sourceBase + uint64_t(192 + 8*PHASE) * 4U,
sourceBase + uint64_t(240 + 8*PHASE) * 4U,
sourceBase + uint64_t(288 + 8*PHASE) * 4U,
sourceBase + uint64_t(336 + 8*PHASE) * 4U,
sourceBase + uint64_t(24 + 8*PHASE) * 4U,
sourceBase + uint64_t(72 + 8*PHASE) * 4U,
sourceBase + uint64_t(120 + 8*PHASE) * 4U,
sourceBase + uint64_t(168 + 8*PHASE) * 4U,
sourceBase + uint64_t(216 + 8*PHASE) * 4U,
sourceBase + uint64_t(264 + 8*PHASE) * 4U,
sourceBase + uint64_t(312 + 8*PHASE) * 4U,
sourceBase + uint64_t(360 + 8*PHASE) * 4U
};
uint64_t target[16] = {
workBase + uint64_t(((8*PHASE+0)%6)*STRIDE + (((8*PHASE+0)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+0)%6)*STRIDE + (((8*PHASE+0)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+1)%6)*STRIDE + (((8*PHASE+1)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+1)%6)*STRIDE + (((8*PHASE+1)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+2)%6)*STRIDE + (((8*PHASE+2)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+2)%6)*STRIDE + (((8*PHASE+2)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+3)%6)*STRIDE + (((8*PHASE+3)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+3)%6)*STRIDE + (((8*PHASE+3)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+4)%6)*STRIDE + (((8*PHASE+4)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+4)%6)*STRIDE + (((8*PHASE+4)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+5)%6)*STRIDE + (((8*PHASE+5)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+5)%6)*STRIDE + (((8*PHASE+5)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+6)%6)*STRIDE + (((8*PHASE+6)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+6)%6)*STRIDE + (((8*PHASE+6)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+7)%6)*STRIDE + (((8*PHASE+7)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+7)%6)*STRIDE + (((8*PHASE+7)/6)+4)*Q) * 4U
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[4*PHASE], target);
StorePacked(mem.PackedVA()[24+SLOT*24+4*PHASE], source);
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitOutputPhase() {
static_assert(SLOT < (sizeof(T) == sizeof(float) ? 2U : 1U) && PHASE < 5U, "fixed output slots");
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t outputBase;
if constexpr (sizeof(T) == sizeof(float)) outputBase = reinterpret_cast<uint64_t>(mem.OutAt(SLOT).GetPhyAddr());
else outputBase = reinterpret_cast<uint64_t>(mem.PackedFloat().GetPhyAddr());
uint64_t source[16] = {
workBase + uint64_t(((8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 8) * 4U
};
uint64_t target[16] = {
outputBase + uint64_t((5*0+PHASE)*8) * 4U,
outputBase + uint64_t((5*8+PHASE)*8) * 4U,
outputBase + uint64_t((5*1+PHASE)*8) * 4U,
outputBase + uint64_t((5*9+PHASE)*8) * 4U,
outputBase + uint64_t((5*2+PHASE)*8) * 4U,
outputBase + uint64_t((5*10+PHASE)*8) * 4U,
outputBase + uint64_t((5*3+PHASE)*8) * 4U,
outputBase + uint64_t((5*11+PHASE)*8) * 4U,
outputBase + uint64_t((5*4+PHASE)*8) * 4U,
outputBase + uint64_t((5*12+PHASE)*8) * 4U,
outputBase + uint64_t((5*5+PHASE)*8) * 4U,
outputBase + uint64_t((5*13+PHASE)*8) * 4U,
outputBase + uint64_t((5*6+PHASE)*8) * 4U,
outputBase + uint64_t((5*14+PHASE)*8) * 4U,
outputBase + uint64_t((5*7+PHASE)*8) * 4U,
outputBase + uint64_t((5*15+PHASE)*8) * 4U
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[72+4*PHASE], source);
StorePacked(mem.PackedVA()[92+SLOT*20+4*PHASE], target);
}
template <uint32_t PHASE>
__aicore__ inline void InitPriorPhase() {
const uint64_t priorBase = reinterpret_cast<uint64_t>(mem.PriorFixedSlot(0).GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t source[16] = {
priorBase + uint64_t(0+8*PHASE) * 4U,
priorBase + uint64_t(40+8*PHASE) * 4U,
priorBase + uint64_t(80+8*PHASE) * 4U,
priorBase + uint64_t(120+8*PHASE) * 4U,
priorBase + uint64_t(160+8*PHASE) * 4U,
priorBase + uint64_t(200+8*PHASE) * 4U,
priorBase + uint64_t(240+8*PHASE) * 4U,
priorBase + uint64_t(280+8*PHASE) * 4U,
priorBase + uint64_t(0+8*PHASE) * 4U,
priorBase + uint64_t(40+8*PHASE) * 4U,
priorBase + uint64_t(80+8*PHASE) * 4U,
priorBase + uint64_t(120+8*PHASE) * 4U,
priorBase + uint64_t(160+8*PHASE) * 4U,
priorBase + uint64_t(200+8*PHASE) * 4U,
priorBase + uint64_t(240+8*PHASE) * 4U,
priorBase + uint64_t(280+8*PHASE) * 4U
};
uint64_t target[16] = {
workBase + uint64_t((6+(8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 64) * 4U,
workBase + uint64_t((6+(8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 64) * 4U,
workBase + uint64_t((6+(8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 64) * 4U,
workBase + uint64_t((6+(8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 64) * 4U,
workBase + uint64_t((6+(8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 64) * 4U,
workBase + uint64_t((6+(8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 64) * 4U,
workBase + uint64_t((6+(8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 64) * 4U,
workBase + uint64_t((6+(8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 64) * 4U
};
StorePacked(mem.PackedVA()[132+4*PHASE], target);
StorePacked(mem.PackedVA()[152+4*PHASE], source);
}
__aicore__ inline void InitPackedVA() {
InitInputPhase<0, 0>();
InitInputPhase<0, 1>();
InitInputPhase<0, 2>();
InitOutputPhase<0, 0>();
InitOutputPhase<0, 1>();
InitOutputPhase<0, 2>();
InitOutputPhase<0, 3>();
InitOutputPhase<0, 4>();
if constexpr (sizeof(T) == sizeof(float)) {
InitInputPhase<1, 0>();
InitInputPhase<1, 1>();
InitInputPhase<1, 2>();
InitOutputPhase<1, 0>();
InitOutputPhase<1, 1>();
InitOutputPhase<1, 2>();
InitOutputPhase<1, 3>();
InitOutputPhase<1, 4>();
}
InitPriorPhase<0>();
InitPriorPhase<1>();
InitPriorPhase<2>();
InitPriorPhase<3>();
InitPriorPhase<4>();
AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
}
template <uint32_t SLOT>
__aicore__ inline void Input() {
PackedTranspose<16, 1, 48>(mem.PackedVA()[0], mem.PackedVA()[24+SLOT*24+0]);
PackedTranspose<16, 1, 48>(mem.PackedVA()[4], mem.PackedVA()[24+SLOT*24+4]);
PackedTranspose<16, 1, 48>(mem.PackedVA()[8], mem.PackedVA()[24+SLOT*24+8]);
}
template <uint32_t SLOT>
__aicore__ inline void Output() {
PackedTranspose<8, 80, 2>(mem.PackedVA()[92+SLOT*20+0], mem.PackedVA()[72+0]);
PackedTranspose<8, 80, 2>(mem.PackedVA()[92+SLOT*20+4], mem.PackedVA()[72+4]);
PackedTranspose<8, 80, 2>(mem.PackedVA()[92+SLOT*20+8], mem.PackedVA()[72+8]);
PackedTranspose<8, 80, 2>(mem.PackedVA()[92+SLOT*20+12], mem.PackedVA()[72+12]);
PackedTranspose<8, 80, 2>(mem.PackedVA()[92+SLOT*20+16], mem.PackedVA()[72+16]);
}
public:
__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded) {
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
if (block != 42U) return;
constexpr uint32_t begin = 32256U;
rawGm.SetGlobalBuffer((__gm__ T *)raw);
priorGm.SetGlobalBuffer((__gm__ float *)priors);
outGm.SetGlobalBuffer((__gm__ T *)decoded);
const AscendC::DataCopyExtParams rp{2, COUNT * 6U * uint32_t(sizeof(T)), (N - COUNT) * 6U * uint32_t(sizeof(T)), 0, 0};
const AscendC::DataCopyExtParams pp{1, COUNT * 5U * uint32_t(sizeof(float)), 0, 0, 0};
auto firstRaw = mem.AllocRaw();
auto p = mem.AllocPrior();
AscendC::DataCopy(firstRaw, rawGm[begin * 6U], AscendC::DataCopyParams{2U,384U,24192U,0U});
AscendC::DataCopyPad(p, priorGm[begin * 5U], pp, AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
mem.RawReady();
mem.PriorReady();
auto one = mem.Ones();
AscendC::Duplicate(one, 1.0f, FRAME * 6U);
AscendC::LocalTensor<float> f[11];
for (uint32_t k = 0; k < 11; ++k) f[k] = mem.Work()[k * STRIDE];
InitPackedVA();
mem.PriorWait();
PackedTranspose<8, 1, 40>(mem.PackedVA()[132+0], mem.PackedVA()[152+0]);
PackedTranspose<8, 1, 40>(mem.PackedVA()[132+4], mem.PackedVA()[152+4]);
PackedTranspose<8, 1, 40>(mem.PackedVA()[132+8], mem.PackedVA()[152+8]);
PackedTranspose<8, 1, 40>(mem.PackedVA()[132+12], mem.PackedVA()[152+12]);
PackedTranspose<8, 1, 40>(mem.PackedVA()[132+16], mem.PackedVA()[152+16]);
AscendC::PipeBarrier<PIPE_V>();
AscendC::DataCopy(p, f[8], STRIDE);
AscendC::DataCopy(p[STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[2 * STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[3 * STRIDE], f[8], STRIDE);
AscendC::PipeBarrier<PIPE_V>();
for (uint32_t iteration = 0; iteration < 4U; ++iteration) {
const uint32_t slot = iteration & 1U;
const uint32_t pos = iteration * 2U * N + begin;
auto r = firstRaw;
if (iteration != 0U) {
r = mem.AllocRaw();
AscendC::DataCopy(r, rawGm[pos * 6U], AscendC::DataCopyParams{2U,384U,24192U,0U});
mem.RawReady();
}
mem.RawWait();
auto rf = r.template ReinterpretCast<float>();
AscendC::Muls(rf, rf, -1.0f, FRAME * 6U);
AscendC::Exp(rf, rf, FRAME * 6U);
AscendC::Adds(rf, rf, 1.0f, FRAME * 6U);
AscendC::Div(rf, one, rf, FRAME * 6U);
AscendC::PipeBarrier<PIPE_V>();
if (slot == 0U) Input<0>(); else Input<1>();
AscendC::PipeBarrier<PIPE_V>();
AscendC::Muls(f[0], f[0], 2.0f, 4 * STRIDE);
AscendC::Adds(f[0], f[0], -0.5f, 2 * STRIDE);
AscendC::Add(f[0], f[0], f[6], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[2], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[9], 2 * STRIDE);
AscendC::Mul(f[0], f[0], p, 4 * STRIDE);
AscendC::Mul(f[4], f[4], f[5], FRAME);
auto out = mem.AllocOut();
AscendC::PipeBarrier<PIPE_V>();
if (slot == 0U) Output<0>(); else Output<1>();
AscendC::PipeBarrier<PIPE_V>();
mem.OutReady();
mem.OutWait();
const AscendC::DataCopyExtParams op{2, COUNT * 5U * uint32_t(sizeof(T)), 0, (N - COUNT) * 5U * uint32_t(sizeof(T)), 0};
AscendC::DataCopy(outGm[pos * 5U], out, AscendC::DataCopyParams{2U,320U,0U,20160U});
mem.FreeOut();
mem.FreeRaw();
}
mem.FreePrior();
mem.Drain();
}
};


template <typename T, bool VA_GROUP38 = false>
class FireA2F6SpanPairStorage {
static constexpr uint32_t RAW_BYTES = 1664U * 6U * sizeof(T);
static constexpr uint32_t PRIOR_BASE = 2U * RAW_BYTES;
static constexpr uint32_t PRIOR_BYTES = 4U * 1680U * sizeof(float);
static constexpr uint32_t WORK_BASE = PRIOR_BASE + PRIOR_BYTES + 256U;
static constexpr uint32_t WORK_BYTES = 74496U;
static constexpr uint32_t ONE_BASE = WORK_BASE + WORK_BYTES;
static constexpr uint32_t RF_BASE = ONE_BASE + 32U + (sizeof(T) == sizeof(half) ? 224U : 0U);
static constexpr uint32_t VA_BASE = RF_BASE + (sizeof(T) == sizeof(half) ? 39936U : 0U);
static constexpr uint32_t VA_WORDS = VA_GROUP38 ? 152U : 172U;
static_assert(!VA_GROUP38 || sizeof(T) == sizeof(float), "VA38 only FP32");
static_assert(VA_BASE + VA_WORDS * sizeof(uint64_t) == (VA_GROUP38 ? 182752U : (sizeof(T) == sizeof(float) ? 182912U : 183136U)), "fixed576 UB high water");
static_assert(VA_BASE + VA_WORDS * sizeof(uint64_t) < 188416U, "retain SDK UB reserve");
static_assert(WORK_BASE % 512U == 0U && (ONE_BASE - (sizeof(T) == sizeof(float) ? 0U : RF_BASE)) % 512U == 256U, "fixed phases");
uint32_t used = 0;
public:
__aicore__ inline AscendC::LocalTensor<T> RawFixedSlot(uint32_t slot) {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, slot * RAW_BYTES, 9984U);
}
__aicore__ inline AscendC::LocalTensor<T> OutAt(uint32_t slot) {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, slot * RAW_BYTES, 8320U);
}
__aicore__ inline AscendC::LocalTensor<float> Prior() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE, 6720U);
}
__aicore__ inline AscendC::LocalTensor<float> Work() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, WORK_BASE, WORK_BYTES / 4U);
}
__aicore__ inline AscendC::LocalTensor<float> Ones() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE, 8U);
}
__aicore__ inline AscendC::LocalTensor<float> Twos() {
static_assert(sizeof(T) == sizeof(float), "coordinate numerator2 only FP32");
constexpr uint32_t VA_END = VA_BASE + VA_WORDS * sizeof(uint64_t);
constexpr uint32_t TWO_BASE = ONE_BASE + ((VA_END - ONE_BASE + 511U) / 512U) * 512U;
static_assert(ONE_BASE == 181504U && VA_END == (VA_GROUP38 ? 182752U : 182912U) && TWO_BASE == 183040U, "qualified FP32 numerator address");
static_assert(TWO_BASE % 32U == 0U && TWO_BASE % 512U == 256U && TWO_BASE >= VA_END && TWO_BASE + 32U < 188416U, "separate numerator2 after fixed VA extent");
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, TWO_BASE, 8U);
}
__aicore__ inline AscendC::LocalTensor<float> RawFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, RF_BASE, 9984U);
}
__aicore__ inline AscendC::LocalTensor<float> PackedFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, RF_BASE, 8320U);
}
__aicore__ inline AscendC::LocalTensor<uint64_t> PackedVA() {
return AscendC::LocalTensor<uint64_t>(AscendC::TPosition::VECCALC, VA_BASE, VA_WORDS);
}
__aicore__ inline AscendC::LocalTensor<T> AllocSlot(uint32_t slot) {
if (used & (1U << slot)) {
AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
}
return RawFixedSlot(slot);
}
__aicore__ inline void PublishOutputAndRelease(uint32_t slot) {
AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);
used |= 1U << slot;
}
__aicore__ inline void Drain() {
for (uint32_t slot = 0; slot < 2; ++slot) if (used & (1U << slot)) {
AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
}
}
};


template <typename T, uint32_t N, bool SCORE_FUSED = false, bool ROTATE_PAIRS = false, bool QUALIFIED_SCORE_ROTATE = false, bool QUALIFIED_HALF_SCORE = false, bool COORD_DIV2 = false, bool VA_GROUP38 = false, uint32_t ROW = 832U>
class KernelFireA2F6SpanPair {
static_assert(!SCORE_FUSED || sizeof(T) == sizeof(float) || QUALIFIED_HALF_SCORE, "score fusion requires dtype qualification");
static_assert(!(SCORE_FUSED && ROTATE_PAIRS) || QUALIFIED_SCORE_ROTATE, "636 mechanisms are exclusive");
static_assert(!QUALIFIED_SCORE_ROTATE || (!0 && sizeof(T) == sizeof(float) && (N == 32768U || N == 32769U) && SCORE_FUSED && ROTATE_PAIRS), "only qualified float pair fusion rotation");
static_assert(!QUALIFIED_HALF_SCORE || (0 && sizeof(T) == sizeof(half) && SCORE_FUSED && !ROTATE_PAIRS), "only qualified half pair score fusion");
static_assert(!(QUALIFIED_SCORE_ROTATE && QUALIFIED_HALF_SCORE), "qualified pair modes are separate");
static_assert(N == 32768U || N == 32769U, "only qualifiedC40 B8 dimensions");
static constexpr uint32_t FRAME = 2U * ROW;
static constexpr uint32_t WORK_FRAME = ROW == 768U ? 1600U : FRAME;
static constexpr uint32_t STRIDE = WORK_FRAME + 16U, Q = WORK_FRAME / 8U;
static_assert(ROW == 832U || (!0 && sizeof(T) == sizeof(float) && N == 32769U && ROW == 768U && COORD_DIV2 && VA_GROUP38 && SCORE_FUSED && ROTATE_PAIRS && QUALIFIED_SCORE_ROTATE && WORK_FRAME == 1600U && STRIDE == 1616U && Q == 200U && 107008U + 11U * STRIDE * 4U < 181504U), "qualified P6 span43 full768 Q200");
static_assert(!COORD_DIV2 || (!0 && sizeof(T) == sizeof(float) && N == 32769U && SCORE_FUSED && ROTATE_PAIRS && QUALIFIED_SCORE_ROTATE && !QUALIFIED_HALF_SCORE), "qualified N32769 FP32 coordinate Div2");
static_assert(4U * STRIDE % 64U == 0U && WORK_FRAME % 64U == 0U && FRAME % 128U == 0U, "exact coordinate and score Div spans");
static_assert(!VA_GROUP38 || (!0 && sizeof(T) == sizeof(float) && N == 32769U && SCORE_FUSED && ROTATE_PAIRS && QUALIFIED_SCORE_ROTATE && !QUALIFIED_HALF_SCORE && COORD_DIV2), "qualified N32769 FP32 VA38");
static constexpr bool FIXED_TOKENS = sizeof(T) == sizeof(float) && N == 32769U && COORD_DIV2 && VA_GROUP38;
static_assert(!FIXED_TOKENS || (((ROW == 832U && FRAME == 1664U && STRIDE == 1680U) || (ROW == 768U && FRAME == 1536U && WORK_FRAME == 1600U && STRIDE == 1616U)) && SCORE_FUSED && ROTATE_PAIRS && QUALIFIED_SCORE_ROTATE && !QUALIFIED_HALF_SCORE), "qualified FP32 fixed four pair slots");
FireA2F6SpanPairStorage<T, VA_GROUP38> mem;
template <uint32_t OFFSET>
__aicore__ static constexpr uint32_t VAOffset() {
if constexpr (!VA_GROUP38) return OFFSET;
else {
static_assert(OFFSET % 4U == 0U && OFFSET < 172U, "VA38 original group bounds");
constexpr uint32_t group = OFFSET / 4U;
static_assert(group < 3U || (group >= 6U && group < 9U) || (group >= 12U && group < 15U) || (group >= 18U && group < 43U), "VA38 original active group");
if constexpr (group < 3U) return 4U * (25U + group);
else if constexpr (group < 9U) return 4U * (5U * (group - 6U));
else if constexpr (group < 15U) return 4U * (1U + 5U * (group - 12U));
else if constexpr (group < 23U) return 4U * (28U + group - 18U);
else if constexpr (group < 28U) return 4U * (2U + 5U * (group - 23U));
else if constexpr (group < 33U) return 4U * (3U + 5U * (group - 28U));
else if constexpr (group < 38U) return 4U * (33U + group - 33U);
else return 4U * (4U + 5U * (group - 38U));
}
}
AscendC::GlobalTensor<T> rawGm, outGm;
AscendC::GlobalTensor<float> priorGm;
__aicore__ inline void StorePacked(AscendC::LocalTensor<uint64_t> dst, const uint64_t *address) {
#pragma unroll
for (uint32_t q = 0; q < 4; ++q) {
const uint64_t value = ((address[4*q] >> 5) & 0x1fffULL) |
(((address[4*q+1] >> 5) & 0x1fffULL) << 16) |
(((address[4*q+2] >> 5) & 0x1fffULL) << 32) |
(((address[4*q+3] >> 5) & 0x1fffULL) << 48);
dst.SetValue(q, value);
}
}
template <uint8_t REPEAT, uint16_t DST_STRIDE, uint16_t SRC_STRIDE>
__aicore__ inline void PackedTranspose(AscendC::LocalTensor<uint64_t> dst, AscendC::LocalTensor<uint64_t> src) {
auto *dstPacked = reinterpret_cast<__ubuf__ uint64_t *>(dst.GetPhyAddr());
auto *srcPacked = reinterpret_cast<__ubuf__ uint64_t *>(src.GetPhyAddr());
__cce_scalar::vld_va_reg(VA0, dstPacked, L128);
__cce_scalar::vld_va_reg(VA1, dstPacked, H128);
__cce_scalar::vld_va_reg(VA2, srcPacked, L128);
__cce_scalar::vld_va_reg(VA3, srcPacked, H128);
__cce_scalar::scatter_vnchwconv_b32(VA0, VA2, REPEAT, DST_STRIDE, SRC_STRIDE);
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitInputPhase() {
static_assert(SLOT < (sizeof(T) == sizeof(float) ? 2U : 1U) && PHASE < 3U, "fixed RAW3 slots");
uint64_t sourceBase;
if constexpr (sizeof(T) == sizeof(float)) sourceBase = reinterpret_cast<uint64_t>(mem.RawFixedSlot(SLOT).GetPhyAddr());
else sourceBase = reinterpret_cast<uint64_t>(mem.RawFloat().GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t source[16] = {
sourceBase + uint64_t(0 + 8*PHASE) * 4U,
sourceBase + uint64_t(48 + 8*PHASE) * 4U,
sourceBase + uint64_t(96 + 8*PHASE) * 4U,
sourceBase + uint64_t(144 + 8*PHASE) * 4U,
sourceBase + uint64_t(192 + 8*PHASE) * 4U,
sourceBase + uint64_t(240 + 8*PHASE) * 4U,
sourceBase + uint64_t(288 + 8*PHASE) * 4U,
sourceBase + uint64_t(336 + 8*PHASE) * 4U,
sourceBase + uint64_t(24 + 8*PHASE) * 4U,
sourceBase + uint64_t(72 + 8*PHASE) * 4U,
sourceBase + uint64_t(120 + 8*PHASE) * 4U,
sourceBase + uint64_t(168 + 8*PHASE) * 4U,
sourceBase + uint64_t(216 + 8*PHASE) * 4U,
sourceBase + uint64_t(264 + 8*PHASE) * 4U,
sourceBase + uint64_t(312 + 8*PHASE) * 4U,
sourceBase + uint64_t(360 + 8*PHASE) * 4U
};
uint64_t target[16] = {
workBase + uint64_t(((8*PHASE+0)%6)*STRIDE + (((8*PHASE+0)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+0)%6)*STRIDE + (((8*PHASE+0)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+1)%6)*STRIDE + (((8*PHASE+1)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+1)%6)*STRIDE + (((8*PHASE+1)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+2)%6)*STRIDE + (((8*PHASE+2)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+2)%6)*STRIDE + (((8*PHASE+2)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+3)%6)*STRIDE + (((8*PHASE+3)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+3)%6)*STRIDE + (((8*PHASE+3)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+4)%6)*STRIDE + (((8*PHASE+4)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+4)%6)*STRIDE + (((8*PHASE+4)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+5)%6)*STRIDE + (((8*PHASE+5)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+5)%6)*STRIDE + (((8*PHASE+5)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+6)%6)*STRIDE + (((8*PHASE+6)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+6)%6)*STRIDE + (((8*PHASE+6)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+7)%6)*STRIDE + (((8*PHASE+7)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+7)%6)*STRIDE + (((8*PHASE+7)/6)+4)*Q) * 4U
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[VAOffset<4*PHASE>()], target);
if constexpr (!VA_GROUP38 || PHASE == 0U) StorePacked(mem.PackedVA()[VAOffset<24+SLOT*24+4*PHASE>()], source);
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitOutputPhase() {
static_assert(SLOT < (sizeof(T) == sizeof(float) ? 2U : 1U) && PHASE < 5U, "fixed output slots");
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t outputBase;
if constexpr (sizeof(T) == sizeof(float)) outputBase = reinterpret_cast<uint64_t>(mem.OutAt(SLOT).GetPhyAddr());
else outputBase = reinterpret_cast<uint64_t>(mem.PackedFloat().GetPhyAddr());
uint64_t source[16] = {
workBase + uint64_t(((8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 8) * 4U
};
uint64_t target[16] = {
outputBase + uint64_t((5*0+PHASE)*8) * 4U,
outputBase + uint64_t((5*8+PHASE)*8) * 4U,
outputBase + uint64_t((5*1+PHASE)*8) * 4U,
outputBase + uint64_t((5*9+PHASE)*8) * 4U,
outputBase + uint64_t((5*2+PHASE)*8) * 4U,
outputBase + uint64_t((5*10+PHASE)*8) * 4U,
outputBase + uint64_t((5*3+PHASE)*8) * 4U,
outputBase + uint64_t((5*11+PHASE)*8) * 4U,
outputBase + uint64_t((5*4+PHASE)*8) * 4U,
outputBase + uint64_t((5*12+PHASE)*8) * 4U,
outputBase + uint64_t((5*5+PHASE)*8) * 4U,
outputBase + uint64_t((5*13+PHASE)*8) * 4U,
outputBase + uint64_t((5*6+PHASE)*8) * 4U,
outputBase + uint64_t((5*14+PHASE)*8) * 4U,
outputBase + uint64_t((5*7+PHASE)*8) * 4U,
outputBase + uint64_t((5*15+PHASE)*8) * 4U
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[VAOffset<72+4*PHASE>()], source);
if constexpr (!VA_GROUP38 || PHASE == 0U) StorePacked(mem.PackedVA()[VAOffset<92+SLOT*20+4*PHASE>()], target);
}
template <uint32_t PHASE>
__aicore__ inline void InitPriorPhase() {
const uint64_t priorBase = reinterpret_cast<uint64_t>(mem.Prior().GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t source[16] = {
priorBase + uint64_t(0+8*PHASE) * 4U,
priorBase + uint64_t(40+8*PHASE) * 4U,
priorBase + uint64_t(80+8*PHASE) * 4U,
priorBase + uint64_t(120+8*PHASE) * 4U,
priorBase + uint64_t(160+8*PHASE) * 4U,
priorBase + uint64_t(200+8*PHASE) * 4U,
priorBase + uint64_t(240+8*PHASE) * 4U,
priorBase + uint64_t(280+8*PHASE) * 4U,
priorBase + uint64_t(0+8*PHASE) * 4U,
priorBase + uint64_t(40+8*PHASE) * 4U,
priorBase + uint64_t(80+8*PHASE) * 4U,
priorBase + uint64_t(120+8*PHASE) * 4U,
priorBase + uint64_t(160+8*PHASE) * 4U,
priorBase + uint64_t(200+8*PHASE) * 4U,
priorBase + uint64_t(240+8*PHASE) * 4U,
priorBase + uint64_t(280+8*PHASE) * 4U
};
uint64_t target[16] = {
workBase + uint64_t((6+(8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + (ROW / 8U)) * 4U,
workBase + uint64_t((6+(8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + (ROW / 8U)) * 4U,
workBase + uint64_t((6+(8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + (ROW / 8U)) * 4U,
workBase + uint64_t((6+(8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + (ROW / 8U)) * 4U,
workBase + uint64_t((6+(8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + (ROW / 8U)) * 4U,
workBase + uint64_t((6+(8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + (ROW / 8U)) * 4U,
workBase + uint64_t((6+(8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + (ROW / 8U)) * 4U,
workBase + uint64_t((6+(8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + (ROW / 8U)) * 4U
};
StorePacked(mem.PackedVA()[VAOffset<132+4*PHASE>()], target);
if constexpr (!VA_GROUP38 || PHASE == 0U) StorePacked(mem.PackedVA()[VAOffset<152+4*PHASE>()], source);
}
__aicore__ inline void InitPackedVA() {
InitInputPhase<0, 0>();
InitInputPhase<0, 1>();
InitInputPhase<0, 2>();
InitOutputPhase<0, 0>();
InitOutputPhase<0, 1>();
InitOutputPhase<0, 2>();
InitOutputPhase<0, 3>();
InitOutputPhase<0, 4>();
if constexpr (sizeof(T) == sizeof(float)) {
InitInputPhase<1, 0>();
InitInputPhase<1, 1>();
InitInputPhase<1, 2>();
InitOutputPhase<1, 0>();
InitOutputPhase<1, 1>();
InitOutputPhase<1, 2>();
InitOutputPhase<1, 3>();
InitOutputPhase<1, 4>();
}
InitPriorPhase<0>();
InitPriorPhase<1>();
InitPriorPhase<2>();
InitPriorPhase<3>();
InitPriorPhase<4>();
AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
if constexpr (VA_GROUP38) {
auto *entries=reinterpret_cast<__ubuf__ int16_t *>(mem.PackedVA().GetPhyAddr());
AscendC::SetMaskCount();
AscendC::SetVectorMask<int16_t, AscendC::MaskMode::COUNTER>(80U);
#pragma unroll
for (uint32_t phase=1U;phase<5U;++phase)
asc_add_scalar(entries+phase*80U,entries,int16_t(phase),
uint8_t(1),uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetMaskNorm();
AscendC::ResetMask();
}
}
template <uint32_t SLOT>
__aicore__ inline void Input() {
PackedTranspose<FRAME / 64U, 1, 48>(mem.PackedVA()[VAOffset<0>()], mem.PackedVA()[VAOffset<24+SLOT*24+0>()]);
PackedTranspose<FRAME / 64U, 1, 48>(mem.PackedVA()[VAOffset<4>()], mem.PackedVA()[VAOffset<24+SLOT*24+4>()]);
PackedTranspose<FRAME / 64U, 1, 48>(mem.PackedVA()[VAOffset<8>()], mem.PackedVA()[VAOffset<24+SLOT*24+8>()]);
}
template <uint32_t SLOT>
__aicore__ inline void Output() {
PackedTranspose<ROW / 64U, 80, 2>(mem.PackedVA()[VAOffset<92+SLOT*20+0>()], mem.PackedVA()[VAOffset<72+0>()]);
PackedTranspose<ROW / 64U, 80, 2>(mem.PackedVA()[VAOffset<92+SLOT*20+4>()], mem.PackedVA()[VAOffset<72+4>()]);
PackedTranspose<ROW / 64U, 80, 2>(mem.PackedVA()[VAOffset<92+SLOT*20+8>()], mem.PackedVA()[VAOffset<72+8>()]);
PackedTranspose<ROW / 64U, 80, 2>(mem.PackedVA()[VAOffset<92+SLOT*20+12>()], mem.PackedVA()[VAOffset<72+12>()]);
PackedTranspose<ROW / 64U, 80, 2>(mem.PackedVA()[VAOffset<92+SLOT*20+16>()], mem.PackedVA()[VAOffset<72+16>()]);
}
public:
__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded) {
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
if constexpr (ROW == 768U) { if (block >= 42U) return; } else { if (block >= 39U) return; }
const uint32_t begin = block * ROW;
rawGm.SetGlobalBuffer((__gm__ T *)raw);
priorGm.SetGlobalBuffer((__gm__ float *)priors);
outGm.SetGlobalBuffer((__gm__ T *)decoded);
const AscendC::DataCopyExtParams rp{2, ROW*6U*uint32_t(sizeof(T)), (N-ROW)*6U*uint32_t(sizeof(T)), 0, 0};
const AscendC::DataCopyExtParams pp{1, ROW*5U*4U, 0, 0, 0};
const uint32_t cohort = ROTATE_PAIRS ? (block & 3U) : 0U;
auto firstRaw = mem.AllocSlot(0);
auto p = mem.Prior();
AscendC::DataCopyPad(firstRaw, rawGm[(2U*cohort*N+begin)*6U], rp, AscendC::DataCopyPadExtParams<T>{false,0,0,0});
if constexpr (VA_GROUP38) {
static_assert(!0 && sizeof(T) == sizeof(float) && N == 32769U && COORD_DIV2 && ((ROW == 832U && FRAME == 1664U && STRIDE == 1680U) || (ROW == 768U && WORK_FRAME == 1600U && STRIDE == 1616U)), "qualified P6 prior ordinary row blocks");
AscendC::DataCopy(p, priorGm[begin*5U], AscendC::DataCopyParams{1U,uint16_t(ROW*20U/32U),0U,0U});
} else {
AscendC::DataCopyPad(p, priorGm[begin*5U], pp, AscendC::DataCopyPadExtParams<float>{false,0,0,0});
}
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
auto one = mem.Ones();
AscendC::Duplicate(one, 1.0f, 8U);
if constexpr (COORD_DIV2) AscendC::Duplicate(mem.Twos(), 2.0f, 8U);
AscendC::LocalTensor<float> f[11];
for (uint32_t k=0;k<11;++k) f[k]=mem.Work()[k*STRIDE];
InitPackedVA();
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
PackedTranspose<ROW / 64U, 1, 40>(mem.PackedVA()[VAOffset<132+0>()], mem.PackedVA()[VAOffset<152+0>()]);
PackedTranspose<ROW / 64U, 1, 40>(mem.PackedVA()[VAOffset<132+4>()], mem.PackedVA()[VAOffset<152+4>()]);
PackedTranspose<ROW / 64U, 1, 40>(mem.PackedVA()[VAOffset<132+8>()], mem.PackedVA()[VAOffset<152+8>()]);
PackedTranspose<ROW / 64U, 1, 40>(mem.PackedVA()[VAOffset<132+12>()], mem.PackedVA()[VAOffset<152+12>()]);
PackedTranspose<ROW / 64U, 1, 40>(mem.PackedVA()[VAOffset<132+16>()], mem.PackedVA()[VAOffset<152+16>()]);
AscendC::PipeBarrier<PIPE_V>();
if constexpr (sizeof(T) != sizeof(float)) {
AscendC::DataCopy(p, f[8], STRIDE);
AscendC::DataCopy(p[STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[2*STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[3*STRIDE], f[8], STRIDE);
AscendC::PipeBarrier<PIPE_V>();
}
for (uint32_t iteration=0;iteration<4;++iteration) {
const uint32_t pair = ROTATE_PAIRS ? ((iteration+cohort)&3U) : iteration;
const uint32_t slot=iteration&1U;
const uint32_t pos=pair*2U*N+begin;
auto r=firstRaw;
if (iteration!=0) {
if constexpr (FIXED_TOKENS) {
if (iteration >= 2U) {
AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
}
r=mem.RawFixedSlot(slot);
} else
r=mem.AllocSlot(slot);
AscendC::DataCopyPad(r,rawGm[pos*6U],rp,AscendC::DataCopyPadExtParams<T>{false,0,0,0});
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot);
}
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);
AscendC::LocalTensor<float> rf;
if constexpr(sizeof(T)==sizeof(float)) rf=r.template ReinterpretCast<float>();
else {rf=mem.RawFloat();AscendC::Cast(rf,r,AscendC::RoundMode::CAST_NONE,FRAME*6U);}
AscendC::Muls(rf,rf,-1.0f,FRAME*6U);
AscendC::Exp(rf,rf,FRAME*6U);
AscendC::Adds(rf,rf,1.0f,FRAME*6U);
const AscendC::BinaryRepeatParams divBinary{1,0,1,8,0,8};
if constexpr (!SCORE_FUSED) AscendC::Div<float>(rf,one,rf,uint64_t(64),uint8_t(FRAME*6U/64U),divBinary);
AscendC::PipeBarrier<PIPE_V>();
if constexpr(sizeof(T)==sizeof(float)) {if(slot==0)Input<0>();else Input<1>();}
else Input<0>();
AscendC::PipeBarrier<PIPE_V>();
if constexpr (SCORE_FUSED) {
AscendC::Mul(f[4],f[4],f[5],WORK_FRAME);
if constexpr (COORD_DIV2) {
AscendC::Div<float>(mem.Work(),mem.Twos(),mem.Work(),uint64_t(64),uint8_t(4U*STRIDE/64U),divBinary);
if constexpr (VA_GROUP38) {
static_assert(!0 && sizeof(T) == sizeof(float) && N == 32769U && ((ROW == 832U && FRAME == 1664U && STRIDE == 1680U) || (ROW == 768U && WORK_FRAME == 1600U && STRIDE == 1616U)), "qualified P6 raw score Div Normal64");
auto *scorePtr = reinterpret_cast<__ubuf__ float *>(f[4].GetPhyAddr());
auto *onePtr = reinterpret_cast<__ubuf__ float *>(one.GetPhyAddr());
asc_div(scorePtr, onePtr, scorePtr, uint8_t(WORK_FRAME/64U),
uint8_t(1),uint8_t(0),uint8_t(1),uint8_t(8),uint8_t(0),uint8_t(8));
} else {
AscendC::Div<float>(f[4],one,f[4],uint64_t(64),uint8_t(26),divBinary);
}
} else {
AscendC::Div<float>(mem.Work(),one,mem.Work(),uint64_t(64),uint8_t(132),divBinary);
}
AscendC::PipeBarrier<PIPE_V>();
}
if constexpr (!COORD_DIV2) AscendC::Muls(f[0],f[0],2.0f,4*STRIDE);
AscendC::Adds(f[0],f[0],-0.5f,2*STRIDE);
AscendC::Add(f[0],f[0],f[6],2*STRIDE);
AscendC::Mul(f[2],f[2],f[2],2*STRIDE);
AscendC::Mul(f[2],f[2],f[9],2*STRIDE);
if constexpr (sizeof(T) == sizeof(float)) {
AscendC::PipeBarrier<PIPE_V>();
auto *wPtr = reinterpret_cast<__ubuf__ float *>(mem.Work().GetPhyAddr());
AscendC::SetMaskCount();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>((COORD_DIV2 && VA_GROUP38 && N == 32769U) ? WORK_FRAME : STRIDE);
#pragma unroll
for (uint32_t k=0;k<4;++k)
asc_mul(wPtr+k*STRIDE, wPtr+k*STRIDE, wPtr+8*STRIDE,
uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(8),uint8_t(8),uint8_t(8));
AscendC::SetMaskNorm();
AscendC::ResetMask();
} else {
AscendC::Mul(f[0],f[0],p,4*STRIDE);
}
if constexpr (!SCORE_FUSED) AscendC::Mul(f[4],f[4],f[5],FRAME);
AscendC::PipeBarrier<PIPE_V>();
if constexpr(sizeof(T)==sizeof(float)) {if(slot==0)Output<0>();else Output<1>();}
else Output<0>();
AscendC::PipeBarrier<PIPE_V>();
auto out=mem.OutAt(slot);
if constexpr(sizeof(T)!=sizeof(float)) {
AscendC::Cast(out,mem.PackedFloat(),AscendC::RoundMode::CAST_RINT,FRAME*5U);
AscendC::PipeBarrier<PIPE_V>();
}
AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot);
AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
const AscendC::DataCopyExtParams op{2,ROW*5U*uint32_t(sizeof(T)),0,(N-ROW)*5U*uint32_t(sizeof(T)),0};
AscendC::DataCopyPad(outGm[pos*5U],out,op);
if constexpr (FIXED_TOKENS) {
AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);
} else
mem.PublishOutputAndRelease(slot);
}
if constexpr (FIXED_TOKENS) {
AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
} else
mem.Drain();
}
};


template <typename T>
class FireA2F6SpanTailStorage {
static_assert(sizeof(T) == sizeof(float), "tail513 float storage only");
static constexpr uint32_t TILE = 1152U;
static constexpr uint32_t FRAME = TILE;
static constexpr uint32_t DEPTH = TILE > 256 ? 2 : 1;
static constexpr uint32_t STRIDE = FRAME + 16;
static constexpr uint32_t RAW_BYTES = (sizeof(T) == sizeof(float) ? FRAME : TILE) * 6 * sizeof(T);
static constexpr uint32_t PRIOR_WORDS = FRAME * 5;
static constexpr uint32_t PRIOR_BYTES = PRIOR_WORDS * sizeof(float);
static constexpr uint32_t OUT_BYTES = (sizeof(T) == sizeof(float) ? FRAME : TILE) * 5 * sizeof(T);
static constexpr uint32_t WORK_BYTES = ((STRIDE * 11 * sizeof(float) + 511) / 512) * 512 + 256;
static constexpr uint32_t ONE_WORDS = 8U;
static constexpr uint32_t ONE_BYTES = ONE_WORDS * sizeof(float) + (sizeof(T) == sizeof(half) ? 256 : 0);
static constexpr uint32_t RAW_BASE = 0;
static constexpr uint32_t PRIOR_BASE = RAW_BASE + DEPTH * RAW_BYTES;
static constexpr uint32_t OUT_BASE = PRIOR_BASE + PRIOR_BYTES;
static constexpr uint32_t WORK_BASE = OUT_BASE + DEPTH * OUT_BYTES;
static constexpr uint32_t ONE_BASE = WORK_BASE + WORK_BYTES;
static constexpr uint32_t VA_BASE = ONE_BASE + ONE_BYTES;
static constexpr uint32_t VA_WORDS = 172;
static_assert(VA_BASE + VA_WORDS * sizeof(uint64_t) < 188416, "reserve the SDK temporary UB region");
static_assert((RAW_BYTES | PRIOR_BYTES | OUT_BYTES | WORK_BYTES | ONE_BYTES | VA_BASE) % 32 == 0,
"all static addresses and capacities retain 32-byte alignment");
static_assert(RAW_BYTES == 27648U && PRIOR_BASE == 55296U && OUT_BASE == 78336U && WORK_BASE == 124416U && ONE_BASE == 176384U && VA_BASE == 176416U && VA_BASE + 172U * 8U == 177792U, "qualified tail513 exact fixed layout");
uint32_t rawCursor = 0, priorCursor = 0, outCursor = 0;
uint32_t rawSlot = 0, priorSlot = 0, outSlot = 0;
uint32_t rawUsed = 0, priorUsed = 0, outUsed = 0;
public:
__aicore__ inline AscendC::LocalTensor<uint64_t> PackedVA() {
return AscendC::LocalTensor<uint64_t>(AscendC::TPosition::VECCALC, VA_BASE, VA_WORDS);
}
__aicore__ inline AscendC::LocalTensor<float> Work() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, WORK_BASE, WORK_BYTES / sizeof(float));
}
__aicore__ inline AscendC::LocalTensor<float> Ones() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE, ONE_BYTES / sizeof(float));
}
__aicore__ inline AscendC::LocalTensor<T> AllocRaw() {
rawSlot = rawCursor;
rawCursor = (rawCursor + 1) % DEPTH;
if (rawUsed & (1U << rawSlot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(rawSlot);
return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, RAW_BASE + rawSlot * RAW_BYTES, (sizeof(T) == sizeof(float) ? FRAME : TILE) * 6);
}
__aicore__ inline AscendC::LocalTensor<float> AllocPrior() {
priorSlot = priorCursor;
priorCursor = 0;
if (priorUsed & (1U << priorSlot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2 + priorSlot);
return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE + priorSlot * PRIOR_BYTES, PRIOR_WORDS);
}
__aicore__ inline AscendC::LocalTensor<T> AllocOut() {
outSlot = outCursor;
outCursor = (outCursor + 1) % DEPTH;
if (outUsed & (1U << outSlot)) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outSlot);
return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, OUT_BASE + outSlot * OUT_BYTES, (sizeof(T) == sizeof(float) ? FRAME : TILE) * 5);
}
__aicore__ inline AscendC::LocalTensor<T> RawFixedSlot(uint32_t slot) {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, RAW_BASE + slot * RAW_BYTES,
(sizeof(T) == sizeof(float) ? FRAME : TILE) * 6);
}
__aicore__ inline AscendC::LocalTensor<float> PriorFixedSlot(uint32_t slot) {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE + slot * PRIOR_BYTES, PRIOR_WORDS);
}
__aicore__ inline AscendC::LocalTensor<T> OutAt(uint32_t slot) {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, OUT_BASE + slot * OUT_BYTES,
(sizeof(T) == sizeof(float) ? FRAME : TILE) * 5);
}
__aicore__ inline uint32_t CurrentOutSlot() const { return outSlot; }
__aicore__ inline void RawReady() { AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(rawSlot); }
__aicore__ inline void RawWait() { AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(rawSlot); }
__aicore__ inline void PriorReady() { AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2 + priorSlot); }
__aicore__ inline void PriorWait() { AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2 + priorSlot); }
__aicore__ inline void OutReady() { AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outSlot); }
__aicore__ inline void OutWait() { AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outSlot); }
__aicore__ inline void FreeRaw() {
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(rawSlot);
rawUsed |= 1U << rawSlot;
}
__aicore__ inline void FreePrior() {
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2 + priorSlot);
priorUsed |= 1U << priorSlot;
}
__aicore__ inline void FreeOut() {
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outSlot);
outUsed |= 1U << outSlot;
}
__aicore__ inline void Drain() {
for (uint32_t slot = 0; slot < DEPTH; ++slot) {
if (rawUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
if (priorUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2 + slot);
if (outUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
}
}
};


template <typename T, uint32_t N>
class KernelFireA2F6SpanTail513 {
static_assert(!0 && sizeof(T) == sizeof(float) && N == 32769U, "qualified FP32 N32769 tail513 span43 pairs");
static constexpr uint32_t COUNT = 513U, PITCH = 576U, FRAME = 1152U, STRIDE = 1168U, Q = 144U;
static_assert(PITCH * 2U == FRAME && COUNT < PITCH && PITCH % 64U == 0U && FRAME % 128U == 0U && Q == FRAME / 8U && 42U * 768U + COUNT == N, "exact tail pair geometry");
static_assert(PITCH * 6U * sizeof(T) == 13824U && PITCH * 5U * sizeof(T) == 11520U && ((COUNT * 6U * sizeof(T) + 31U) / 32U) + 47U == 432U && ((COUNT * 5U * sizeof(T) + 31U) / 32U) + 39U == 360U, "qualified byte and UB block gap geometry");
FireA2F6SpanTailStorage<T> mem;
AscendC::GlobalTensor<T> rawGm, outGm;
AscendC::GlobalTensor<float> priorGm;
__aicore__ inline void StorePacked(AscendC::LocalTensor<uint64_t> dst, const uint64_t *address) {
#pragma unroll
for (uint32_t q = 0; q < 4; ++q) {
const uint64_t value = ((address[4*q] >> 5) & 0x1fffULL) |
(((address[4*q+1] >> 5) & 0x1fffULL) << 16) |
(((address[4*q+2] >> 5) & 0x1fffULL) << 32) |
(((address[4*q+3] >> 5) & 0x1fffULL) << 48);
dst.SetValue(q, value);
}
}
template <uint8_t REPEAT, uint16_t DST_STRIDE, uint16_t SRC_STRIDE>
__aicore__ inline void PackedTranspose(AscendC::LocalTensor<uint64_t> dst, AscendC::LocalTensor<uint64_t> src) {
auto *dstPacked = reinterpret_cast<__ubuf__ uint64_t *>(dst.GetPhyAddr());
auto *srcPacked = reinterpret_cast<__ubuf__ uint64_t *>(src.GetPhyAddr());
__cce_scalar::vld_va_reg(VA0, dstPacked, L128);
__cce_scalar::vld_va_reg(VA1, dstPacked, H128);
__cce_scalar::vld_va_reg(VA2, srcPacked, L128);
__cce_scalar::vld_va_reg(VA3, srcPacked, H128);
__cce_scalar::scatter_vnchwconv_b32(VA0, VA2, REPEAT, DST_STRIDE, SRC_STRIDE);
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitInputPhase() {
static_assert(SLOT < (sizeof(T) == sizeof(float) ? 2U : 1U) && PHASE < 3U, "fixed RAW3 slots");
uint64_t sourceBase;
if constexpr (sizeof(T) == sizeof(float)) sourceBase = reinterpret_cast<uint64_t>(mem.RawFixedSlot(SLOT).GetPhyAddr());
else sourceBase = reinterpret_cast<uint64_t>(mem.RawFloat().GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t source[16] = {
sourceBase + uint64_t(0 + 8*PHASE) * 4U,
sourceBase + uint64_t(48 + 8*PHASE) * 4U,
sourceBase + uint64_t(96 + 8*PHASE) * 4U,
sourceBase + uint64_t(144 + 8*PHASE) * 4U,
sourceBase + uint64_t(192 + 8*PHASE) * 4U,
sourceBase + uint64_t(240 + 8*PHASE) * 4U,
sourceBase + uint64_t(288 + 8*PHASE) * 4U,
sourceBase + uint64_t(336 + 8*PHASE) * 4U,
sourceBase + uint64_t(24 + 8*PHASE) * 4U,
sourceBase + uint64_t(72 + 8*PHASE) * 4U,
sourceBase + uint64_t(120 + 8*PHASE) * 4U,
sourceBase + uint64_t(168 + 8*PHASE) * 4U,
sourceBase + uint64_t(216 + 8*PHASE) * 4U,
sourceBase + uint64_t(264 + 8*PHASE) * 4U,
sourceBase + uint64_t(312 + 8*PHASE) * 4U,
sourceBase + uint64_t(360 + 8*PHASE) * 4U
};
uint64_t target[16] = {
workBase + uint64_t(((8*PHASE+0)%6)*STRIDE + (((8*PHASE+0)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+0)%6)*STRIDE + (((8*PHASE+0)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+1)%6)*STRIDE + (((8*PHASE+1)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+1)%6)*STRIDE + (((8*PHASE+1)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+2)%6)*STRIDE + (((8*PHASE+2)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+2)%6)*STRIDE + (((8*PHASE+2)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+3)%6)*STRIDE + (((8*PHASE+3)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+3)%6)*STRIDE + (((8*PHASE+3)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+4)%6)*STRIDE + (((8*PHASE+4)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+4)%6)*STRIDE + (((8*PHASE+4)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+5)%6)*STRIDE + (((8*PHASE+5)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+5)%6)*STRIDE + (((8*PHASE+5)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+6)%6)*STRIDE + (((8*PHASE+6)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+6)%6)*STRIDE + (((8*PHASE+6)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+7)%6)*STRIDE + (((8*PHASE+7)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+7)%6)*STRIDE + (((8*PHASE+7)/6)+4)*Q) * 4U
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[4*PHASE], target);
StorePacked(mem.PackedVA()[24+SLOT*24+4*PHASE], source);
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitOutputPhase() {
static_assert(SLOT < (sizeof(T) == sizeof(float) ? 2U : 1U) && PHASE < 5U, "fixed output slots");
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t outputBase;
if constexpr (sizeof(T) == sizeof(float)) outputBase = reinterpret_cast<uint64_t>(mem.OutAt(SLOT).GetPhyAddr());
else outputBase = reinterpret_cast<uint64_t>(mem.PackedFloat().GetPhyAddr());
uint64_t source[16] = {
workBase + uint64_t(((8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 8) * 4U
};
uint64_t target[16] = {
outputBase + uint64_t((5*0+PHASE)*8) * 4U,
outputBase + uint64_t((5*8+PHASE)*8) * 4U,
outputBase + uint64_t((5*1+PHASE)*8) * 4U,
outputBase + uint64_t((5*9+PHASE)*8) * 4U,
outputBase + uint64_t((5*2+PHASE)*8) * 4U,
outputBase + uint64_t((5*10+PHASE)*8) * 4U,
outputBase + uint64_t((5*3+PHASE)*8) * 4U,
outputBase + uint64_t((5*11+PHASE)*8) * 4U,
outputBase + uint64_t((5*4+PHASE)*8) * 4U,
outputBase + uint64_t((5*12+PHASE)*8) * 4U,
outputBase + uint64_t((5*5+PHASE)*8) * 4U,
outputBase + uint64_t((5*13+PHASE)*8) * 4U,
outputBase + uint64_t((5*6+PHASE)*8) * 4U,
outputBase + uint64_t((5*14+PHASE)*8) * 4U,
outputBase + uint64_t((5*7+PHASE)*8) * 4U,
outputBase + uint64_t((5*15+PHASE)*8) * 4U
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[72+4*PHASE], source);
StorePacked(mem.PackedVA()[92+SLOT*20+4*PHASE], target);
}
template <uint32_t PHASE>
__aicore__ inline void InitPriorPhase() {
const uint64_t priorBase = reinterpret_cast<uint64_t>(mem.PriorFixedSlot(0).GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t source[16] = {
priorBase + uint64_t(0+8*PHASE) * 4U,
priorBase + uint64_t(40+8*PHASE) * 4U,
priorBase + uint64_t(80+8*PHASE) * 4U,
priorBase + uint64_t(120+8*PHASE) * 4U,
priorBase + uint64_t(160+8*PHASE) * 4U,
priorBase + uint64_t(200+8*PHASE) * 4U,
priorBase + uint64_t(240+8*PHASE) * 4U,
priorBase + uint64_t(280+8*PHASE) * 4U,
priorBase + uint64_t(0+8*PHASE) * 4U,
priorBase + uint64_t(40+8*PHASE) * 4U,
priorBase + uint64_t(80+8*PHASE) * 4U,
priorBase + uint64_t(120+8*PHASE) * 4U,
priorBase + uint64_t(160+8*PHASE) * 4U,
priorBase + uint64_t(200+8*PHASE) * 4U,
priorBase + uint64_t(240+8*PHASE) * 4U,
priorBase + uint64_t(280+8*PHASE) * 4U
};
uint64_t target[16] = {
workBase + uint64_t((6+(8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 72) * 4U,
workBase + uint64_t((6+(8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 72) * 4U,
workBase + uint64_t((6+(8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 72) * 4U,
workBase + uint64_t((6+(8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 72) * 4U,
workBase + uint64_t((6+(8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 72) * 4U,
workBase + uint64_t((6+(8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 72) * 4U,
workBase + uint64_t((6+(8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 72) * 4U,
workBase + uint64_t((6+(8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 72) * 4U
};
StorePacked(mem.PackedVA()[132+4*PHASE], target);
StorePacked(mem.PackedVA()[152+4*PHASE], source);
}
__aicore__ inline void InitPackedVA() {
InitInputPhase<0, 0>();
InitInputPhase<0, 1>();
InitInputPhase<0, 2>();
InitOutputPhase<0, 0>();
InitOutputPhase<0, 1>();
InitOutputPhase<0, 2>();
InitOutputPhase<0, 3>();
InitOutputPhase<0, 4>();
if constexpr (sizeof(T) == sizeof(float)) {
InitInputPhase<1, 0>();
InitInputPhase<1, 1>();
InitInputPhase<1, 2>();
InitOutputPhase<1, 0>();
InitOutputPhase<1, 1>();
InitOutputPhase<1, 2>();
InitOutputPhase<1, 3>();
InitOutputPhase<1, 4>();
}
InitPriorPhase<0>();
InitPriorPhase<1>();
InitPriorPhase<2>();
InitPriorPhase<3>();
InitPriorPhase<4>();
AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
}
template <uint32_t SLOT>
__aicore__ inline void Input() {
PackedTranspose<18, 1, 48>(mem.PackedVA()[0], mem.PackedVA()[24+SLOT*24+0]);
PackedTranspose<18, 1, 48>(mem.PackedVA()[4], mem.PackedVA()[24+SLOT*24+4]);
PackedTranspose<18, 1, 48>(mem.PackedVA()[8], mem.PackedVA()[24+SLOT*24+8]);
}
template <uint32_t SLOT>
__aicore__ inline void Output() {
PackedTranspose<9, 80, 2>(mem.PackedVA()[92+SLOT*20+0], mem.PackedVA()[72+0]);
PackedTranspose<9, 80, 2>(mem.PackedVA()[92+SLOT*20+4], mem.PackedVA()[72+4]);
PackedTranspose<9, 80, 2>(mem.PackedVA()[92+SLOT*20+8], mem.PackedVA()[72+8]);
PackedTranspose<9, 80, 2>(mem.PackedVA()[92+SLOT*20+12], mem.PackedVA()[72+12]);
PackedTranspose<9, 80, 2>(mem.PackedVA()[92+SLOT*20+16], mem.PackedVA()[72+16]);
}
public:
__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded) {
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
if (block != 42U) return;
constexpr uint32_t begin = 32256U;
rawGm.SetGlobalBuffer((__gm__ T *)raw);
priorGm.SetGlobalBuffer((__gm__ float *)priors);
outGm.SetGlobalBuffer((__gm__ T *)decoded);
const AscendC::DataCopyExtParams rp{2, COUNT * 6U * uint32_t(sizeof(T)), (N - COUNT) * 6U * uint32_t(sizeof(T)), 47, 0};
const AscendC::DataCopyExtParams pp{1, COUNT * 5U * uint32_t(sizeof(float)), 0, 0, 0};
auto firstRaw = mem.AllocRaw();
auto p = mem.AllocPrior();
AscendC::DataCopyPad(firstRaw, rawGm[begin * 6U], rp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
AscendC::DataCopyPad(p, priorGm[begin * 5U], pp, AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
mem.RawReady();
mem.PriorReady();
auto one = mem.Ones();
AscendC::Duplicate(one, 1.0f, 8U);
AscendC::LocalTensor<float> f[11];
for (uint32_t k = 0; k < 11; ++k) f[k] = mem.Work()[k * STRIDE];
InitPackedVA();
mem.PriorWait();
PackedTranspose<9, 1, 40>(mem.PackedVA()[132+0], mem.PackedVA()[152+0]);
PackedTranspose<9, 1, 40>(mem.PackedVA()[132+4], mem.PackedVA()[152+4]);
PackedTranspose<9, 1, 40>(mem.PackedVA()[132+8], mem.PackedVA()[152+8]);
PackedTranspose<9, 1, 40>(mem.PackedVA()[132+12], mem.PackedVA()[152+12]);
PackedTranspose<9, 1, 40>(mem.PackedVA()[132+16], mem.PackedVA()[152+16]);
AscendC::PipeBarrier<PIPE_V>();
AscendC::DataCopy(p, f[8], STRIDE);
AscendC::DataCopy(p[STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[2 * STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[3 * STRIDE], f[8], STRIDE);
AscendC::PipeBarrier<PIPE_V>();
for (uint32_t iteration = 0; iteration < 4U; ++iteration) {
const uint32_t slot = iteration & 1U;
const uint32_t pos = iteration * 2U * N + begin;
auto r = firstRaw;
if (iteration != 0U) {
r = mem.AllocRaw();
AscendC::DataCopyPad(r, rawGm[pos * 6U], rp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
mem.RawReady();
}
mem.RawWait();
auto rf = r.template ReinterpretCast<float>();
AscendC::Muls(rf, rf, -1.0f, FRAME * 6U);
AscendC::Exp(rf, rf, FRAME * 6U);
AscendC::Adds(rf, rf, 1.0f, FRAME * 6U);
AscendC::Div<float>(rf, one, rf, uint64_t(64), uint8_t(108), AscendC::BinaryRepeatParams{1,0,1,8,0,8});
AscendC::PipeBarrier<PIPE_V>();
if (slot == 0U) Input<0>(); else Input<1>();
AscendC::PipeBarrier<PIPE_V>();
AscendC::Muls(f[0], f[0], 2.0f, 4 * STRIDE);
AscendC::Adds(f[0], f[0], -0.5f, 2 * STRIDE);
AscendC::Add(f[0], f[0], f[6], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[2], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[9], 2 * STRIDE);
AscendC::Mul(f[0], f[0], p, 4 * STRIDE);
AscendC::Mul(f[4], f[4], f[5], FRAME);
auto out = mem.AllocOut();
AscendC::PipeBarrier<PIPE_V>();
if (slot == 0U) Output<0>(); else Output<1>();
AscendC::PipeBarrier<PIPE_V>();
mem.OutReady();
mem.OutWait();
const AscendC::DataCopyExtParams op{2, COUNT * 5U * uint32_t(sizeof(T)), 39, (N - COUNT) * 5U * uint32_t(sizeof(T)), 0};
AscendC::DataCopyPad(outGm[pos * 5U], out, op);
mem.FreeOut();
mem.FreeRaw();
}
mem.FreePrior();
mem.Drain();
}
};


template <typename T, uint32_t TILE>
class FireA2F6AddFrameStorage {
static_assert(TILE >= 128 && TILE <= 1024 && TILE % 128 == 0, "native hardware frames or large reuse frame");
static constexpr uint32_t FRAME = TILE;
static constexpr uint32_t DEPTH = TILE > 256 ? 2 : 1;
static constexpr uint32_t STRIDE = FRAME + 16;
static constexpr uint32_t RAW_BYTES = (sizeof(T) == sizeof(float) ? FRAME : TILE) * 6 * sizeof(T);
static constexpr uint32_t PRIOR_WORDS = FRAME * 5;
static constexpr uint32_t PRIOR_BYTES = PRIOR_WORDS * sizeof(float);
static constexpr uint32_t OUT_BYTES = (sizeof(T) == sizeof(float) ? FRAME : TILE) * 5 * sizeof(T);
static constexpr uint32_t WORK_BYTES = ((STRIDE * 11 * sizeof(float) + 511) / 512) * 512 + 256;
static constexpr uint32_t ONE_WORDS = FRAME * 6;
static constexpr uint32_t ONE_BYTES = ONE_WORDS * sizeof(float) + (sizeof(T) == sizeof(half) ? 256 : 0);
static constexpr uint32_t RAW_BASE = 0;
static constexpr uint32_t PRIOR_BASE = RAW_BASE + DEPTH * RAW_BYTES;
static constexpr uint32_t OUT_BASE = PRIOR_BASE + PRIOR_BYTES;
static constexpr uint32_t WORK_BASE = OUT_BASE + DEPTH * OUT_BYTES;
static constexpr uint32_t ONE_BASE = WORK_BASE + WORK_BYTES;
static constexpr uint32_t RF_BASE = ONE_BASE + ONE_BYTES;
static constexpr uint32_t PACK_BASE = RF_BASE + (sizeof(T) == sizeof(half) ? FRAME * 6 * sizeof(float) : 0);
static constexpr uint32_t INDEX_BASE = PACK_BASE + (sizeof(T) == sizeof(half) ? FRAME * 5 * sizeof(float) + 256 : 0);
static constexpr uint32_t VA_BASE = INDEX_BASE + FRAME * sizeof(uint32_t);
static constexpr uint32_t VA_WORDS = 172;
static_assert(VA_BASE + VA_WORDS * sizeof(uint64_t) < 188416, "reserve the SDK temporary UB region");
static_assert((RAW_BYTES | PRIOR_BYTES | OUT_BYTES | WORK_BYTES | ONE_BYTES | INDEX_BASE) % 32 == 0,
"all static addresses and capacities retain 32-byte alignment");
uint32_t rawCursor = 0, priorCursor = 0, outCursor = 0;
uint32_t rawSlot = 0, priorSlot = 0, outSlot = 0;
uint32_t rawUsed = 0, priorUsed = 0, outUsed = 0;
public:
__aicore__ inline AscendC::LocalTensor<uint64_t> PackedVA() {
return AscendC::LocalTensor<uint64_t>(AscendC::TPosition::VECCALC, VA_BASE, VA_WORDS);
}
__aicore__ inline AscendC::LocalTensor<int32_t> Indices() {
return AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, INDEX_BASE, FRAME);
}
__aicore__ inline AscendC::LocalTensor<float> Work() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, WORK_BASE, WORK_BYTES / sizeof(float));
}
__aicore__ inline AscendC::LocalTensor<float> Ones() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE, ONE_BYTES / sizeof(float));
}
__aicore__ inline AscendC::LocalTensor<float> RawFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, RF_BASE, FRAME * 6);
}
__aicore__ inline AscendC::LocalTensor<float> PackedFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, PACK_BASE, FRAME * 5);
}
__aicore__ inline AscendC::LocalTensor<T> AllocRaw() {
rawSlot = rawCursor;
rawCursor = (rawCursor + 1) % DEPTH;
if (rawUsed & (1U << rawSlot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(rawSlot);
return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, RAW_BASE + rawSlot * RAW_BYTES, (sizeof(T) == sizeof(float) ? FRAME : TILE) * 6);
}
__aicore__ inline AscendC::LocalTensor<float> AllocPrior() {
priorSlot = priorCursor;
priorCursor = 0;
if (priorUsed & (1U << priorSlot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2 + priorSlot);
return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE + priorSlot * PRIOR_BYTES, PRIOR_WORDS);
}
__aicore__ inline AscendC::LocalTensor<T> AllocOut() {
outSlot = outCursor;
outCursor = (outCursor + 1) % DEPTH;
if (outUsed & (1U << outSlot)) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outSlot);
return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, OUT_BASE + outSlot * OUT_BYTES, (sizeof(T) == sizeof(float) ? FRAME : TILE) * 5);
}
__aicore__ inline AscendC::LocalTensor<T> RawFixedSlot(uint32_t slot) {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, RAW_BASE + slot * RAW_BYTES,
(sizeof(T) == sizeof(float) ? FRAME : TILE) * 6);
}
__aicore__ inline AscendC::LocalTensor<float> PriorFixedSlot(uint32_t slot) {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE + slot * PRIOR_BYTES, PRIOR_WORDS);
}
__aicore__ inline AscendC::LocalTensor<T> OutAt(uint32_t slot) {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, OUT_BASE + slot * OUT_BYTES,
(sizeof(T) == sizeof(float) ? FRAME : TILE) * 5);
}
__aicore__ inline uint32_t CurrentOutSlot() const { return outSlot; }
__aicore__ inline void RawReady() { AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(rawSlot); }
__aicore__ inline void RawWait() { AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(rawSlot); }
__aicore__ inline void PriorReady() { AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2 + priorSlot); }
__aicore__ inline void PriorWait() { AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2 + priorSlot); }
__aicore__ inline void OutReady() { AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outSlot); }
__aicore__ inline void OutWait() { AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outSlot); }
__aicore__ inline void FreeRaw() {
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(rawSlot);
rawUsed |= 1U << rawSlot;
}
__aicore__ inline void FreePrior() {
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2 + priorSlot);
priorUsed |= 1U << priorSlot;
}
__aicore__ inline void FreeOut() {
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outSlot);
outUsed |= 1U << outSlot;
}
__aicore__ inline void Drain() {
for (uint32_t slot = 0; slot < DEPTH; ++slot) {
if (rawUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
if (priorUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2 + slot);
if (outUsed & (1U << slot)) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
}
}
};


template <typename T, bool VA_GROUP38 = false>
class FireA2F6AddPairStorage {
static constexpr uint32_t RAW_BYTES = 1664U * 6U * sizeof(T);
static constexpr uint32_t PRIOR_BASE = 2U * RAW_BYTES;
static constexpr uint32_t PRIOR_BYTES = 4U * 1680U * sizeof(float);
static constexpr uint32_t WORK_BASE = PRIOR_BASE + PRIOR_BYTES + 256U;
static constexpr uint32_t WORK_BYTES = 74496U;
static constexpr uint32_t ONE_BASE = WORK_BASE + WORK_BYTES;
static constexpr uint32_t RF_BASE = ONE_BASE + 32U + (sizeof(T) == sizeof(half) ? 224U : 0U);
static constexpr uint32_t VA_BASE = RF_BASE + (sizeof(T) == sizeof(half) ? 39936U : 0U);
static constexpr uint32_t VA_WORDS = VA_GROUP38 ? 152U : 172U;
static_assert(!VA_GROUP38 || sizeof(T) == sizeof(float), "VA38 only FP32");
static_assert(VA_BASE + VA_WORDS * sizeof(uint64_t) == (VA_GROUP38 ? 182752U : (sizeof(T) == sizeof(float) ? 182912U : 183136U)), "fixed576 UB high water");
static_assert(VA_BASE + VA_WORDS * sizeof(uint64_t) < 188416U, "retain SDK UB reserve");
static_assert(WORK_BASE % 512U == 0U && (ONE_BASE - (sizeof(T) == sizeof(float) ? 0U : RF_BASE)) % 512U == 256U, "fixed phases");
uint32_t used = 0;
public:
__aicore__ inline AscendC::LocalTensor<T> RawFixedSlot(uint32_t slot) {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, slot * RAW_BYTES, 9984U);
}
__aicore__ inline AscendC::LocalTensor<T> OutAt(uint32_t slot) {
return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, slot * RAW_BYTES, 8320U);
}
__aicore__ inline AscendC::LocalTensor<float> Prior() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE, 6720U);
}
__aicore__ inline AscendC::LocalTensor<float> Work() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, WORK_BASE, WORK_BYTES / 4U);
}
__aicore__ inline AscendC::LocalTensor<float> Ones() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE, 8U);
}
__aicore__ inline AscendC::LocalTensor<float> Twos() {
static_assert(sizeof(T) == sizeof(float), "coordinate numerator2 only FP32");
constexpr uint32_t VA_END = VA_BASE + VA_WORDS * sizeof(uint64_t);
constexpr uint32_t TWO_BASE = ONE_BASE + ((VA_END - ONE_BASE + 511U) / 512U) * 512U;
static_assert(ONE_BASE == 181504U && VA_END == (VA_GROUP38 ? 182752U : 182912U) && TWO_BASE == 183040U, "qualified FP32 numerator address");
static_assert(TWO_BASE % 32U == 0U && TWO_BASE % 512U == 256U && TWO_BASE >= VA_END && TWO_BASE + 32U < 188416U, "separate numerator2 after fixed VA extent");
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, TWO_BASE, 8U);
}
__aicore__ inline AscendC::LocalTensor<float> RawFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, RF_BASE, 9984U);
}
__aicore__ inline AscendC::LocalTensor<float> PackedFloat() {
return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, RF_BASE, 8320U);
}
__aicore__ inline AscendC::LocalTensor<uint64_t> PackedVA() {
return AscendC::LocalTensor<uint64_t>(AscendC::TPosition::VECCALC, VA_BASE, VA_WORDS);
}
__aicore__ inline AscendC::LocalTensor<T> AllocSlot(uint32_t slot) {
if (used & (1U << slot)) {
AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
}
return RawFixedSlot(slot);
}
__aicore__ inline void PublishOutputAndRelease(uint32_t slot) {
AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);
used |= 1U << slot;
}
__aicore__ inline void Drain() {
for (uint32_t slot = 0; slot < 2; ++slot) if (used & (1U << slot)) {
AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
}
}
};


template <typename T, uint32_t N, bool SCORE_FUSED = false, bool ROTATE_PAIRS = false, bool QUALIFIED_SCORE_ROTATE = false, bool QUALIFIED_HALF_SCORE = false, bool COORD_DIV2 = false, bool VA_GROUP38 = false>
class KernelFireA2F6AddPair {
static_assert(!SCORE_FUSED || sizeof(T) == sizeof(float) || QUALIFIED_HALF_SCORE, "score fusion requires dtype qualification");
static_assert(!(SCORE_FUSED && ROTATE_PAIRS) || QUALIFIED_SCORE_ROTATE, "636 mechanisms are exclusive");
static_assert(!QUALIFIED_SCORE_ROTATE || (!0 && sizeof(T) == sizeof(float) && (N == 32768U || N == 32769U) && SCORE_FUSED && ROTATE_PAIRS), "only qualified float pair fusion rotation");
static_assert(!QUALIFIED_HALF_SCORE || (0 && sizeof(T) == sizeof(half) && SCORE_FUSED && !ROTATE_PAIRS), "only qualified half pair score fusion");
static_assert(!(QUALIFIED_SCORE_ROTATE && QUALIFIED_HALF_SCORE), "qualified pair modes are separate");
static_assert(N == 32768U || N == 32769U, "only qualifiedC40 B8 dimensions");
static constexpr uint32_t FRAME = 1664U, STRIDE = 1680U, Q = 208U;
static_assert(!COORD_DIV2 || (!0 && sizeof(T) == sizeof(float) && N == 32769U && SCORE_FUSED && ROTATE_PAIRS && QUALIFIED_SCORE_ROTATE && !QUALIFIED_HALF_SCORE), "qualified N32769 FP32 coordinate Div2");
static_assert(4U * STRIDE == 105U * 64U && FRAME == 26U * 64U, "exact coordinate and score Div spans");
static_assert(!VA_GROUP38 || (!0 && sizeof(T) == sizeof(float) && N == 32769U && SCORE_FUSED && ROTATE_PAIRS && QUALIFIED_SCORE_ROTATE && !QUALIFIED_HALF_SCORE && COORD_DIV2), "qualified N32769 FP32 VA38");
static constexpr bool FIXED_TOKENS = sizeof(T) == sizeof(float) && N == 32769U && COORD_DIV2 && VA_GROUP38;
static_assert(!FIXED_TOKENS || (FRAME == 1664U && STRIDE == 1680U && SCORE_FUSED && ROTATE_PAIRS && QUALIFIED_SCORE_ROTATE && !QUALIFIED_HALF_SCORE), "qualified FP32 fixed four pair slots");
FireA2F6AddPairStorage<T, VA_GROUP38> mem;
template <uint32_t OFFSET>
__aicore__ static constexpr uint32_t VAOffset() {
if constexpr (!VA_GROUP38) return OFFSET;
else {
static_assert(OFFSET % 4U == 0U && OFFSET < 172U, "VA38 original group bounds");
constexpr uint32_t group = OFFSET / 4U;
static_assert(group < 3U || (group >= 6U && group < 9U) || (group >= 12U && group < 15U) || (group >= 18U && group < 43U), "VA38 original active group");
if constexpr (group < 3U) return 4U * (25U + group);
else if constexpr (group < 9U) return 4U * (5U * (group - 6U));
else if constexpr (group < 15U) return 4U * (1U + 5U * (group - 12U));
else if constexpr (group < 23U) return 4U * (28U + group - 18U);
else if constexpr (group < 28U) return 4U * (2U + 5U * (group - 23U));
else if constexpr (group < 33U) return 4U * (3U + 5U * (group - 28U));
else if constexpr (group < 38U) return 4U * (33U + group - 33U);
else return 4U * (4U + 5U * (group - 38U));
}
}
AscendC::GlobalTensor<T> rawGm, outGm;
AscendC::GlobalTensor<float> priorGm;
__aicore__ inline void StorePacked(AscendC::LocalTensor<uint64_t> dst, const uint64_t *address) {
#pragma unroll
for (uint32_t q = 0; q < 4; ++q) {
const uint64_t value = ((address[4*q] >> 5) & 0x1fffULL) |
(((address[4*q+1] >> 5) & 0x1fffULL) << 16) |
(((address[4*q+2] >> 5) & 0x1fffULL) << 32) |
(((address[4*q+3] >> 5) & 0x1fffULL) << 48);
dst.SetValue(q, value);
}
}
template <uint8_t REPEAT, uint16_t DST_STRIDE, uint16_t SRC_STRIDE>
__aicore__ inline void PackedTranspose(AscendC::LocalTensor<uint64_t> dst, AscendC::LocalTensor<uint64_t> src) {
auto *dstPacked = reinterpret_cast<__ubuf__ uint64_t *>(dst.GetPhyAddr());
auto *srcPacked = reinterpret_cast<__ubuf__ uint64_t *>(src.GetPhyAddr());
__cce_scalar::vld_va_reg(VA0, dstPacked, L128);
__cce_scalar::vld_va_reg(VA1, dstPacked, H128);
__cce_scalar::vld_va_reg(VA2, srcPacked, L128);
__cce_scalar::vld_va_reg(VA3, srcPacked, H128);
__cce_scalar::scatter_vnchwconv_b32(VA0, VA2, REPEAT, DST_STRIDE, SRC_STRIDE);
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitInputPhase() {
static_assert(SLOT < (sizeof(T) == sizeof(float) ? 2U : 1U) && PHASE < 3U, "fixed RAW3 slots");
uint64_t sourceBase;
if constexpr (sizeof(T) == sizeof(float)) sourceBase = reinterpret_cast<uint64_t>(mem.RawFixedSlot(SLOT).GetPhyAddr());
else sourceBase = reinterpret_cast<uint64_t>(mem.RawFloat().GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t source[16] = {
sourceBase + uint64_t(0 + 8*PHASE) * 4U,
sourceBase + uint64_t(48 + 8*PHASE) * 4U,
sourceBase + uint64_t(96 + 8*PHASE) * 4U,
sourceBase + uint64_t(144 + 8*PHASE) * 4U,
sourceBase + uint64_t(192 + 8*PHASE) * 4U,
sourceBase + uint64_t(240 + 8*PHASE) * 4U,
sourceBase + uint64_t(288 + 8*PHASE) * 4U,
sourceBase + uint64_t(336 + 8*PHASE) * 4U,
sourceBase + uint64_t(24 + 8*PHASE) * 4U,
sourceBase + uint64_t(72 + 8*PHASE) * 4U,
sourceBase + uint64_t(120 + 8*PHASE) * 4U,
sourceBase + uint64_t(168 + 8*PHASE) * 4U,
sourceBase + uint64_t(216 + 8*PHASE) * 4U,
sourceBase + uint64_t(264 + 8*PHASE) * 4U,
sourceBase + uint64_t(312 + 8*PHASE) * 4U,
sourceBase + uint64_t(360 + 8*PHASE) * 4U
};
uint64_t target[16] = {
workBase + uint64_t(((8*PHASE+0)%6)*STRIDE + (((8*PHASE+0)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+0)%6)*STRIDE + (((8*PHASE+0)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+1)%6)*STRIDE + (((8*PHASE+1)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+1)%6)*STRIDE + (((8*PHASE+1)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+2)%6)*STRIDE + (((8*PHASE+2)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+2)%6)*STRIDE + (((8*PHASE+2)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+3)%6)*STRIDE + (((8*PHASE+3)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+3)%6)*STRIDE + (((8*PHASE+3)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+4)%6)*STRIDE + (((8*PHASE+4)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+4)%6)*STRIDE + (((8*PHASE+4)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+5)%6)*STRIDE + (((8*PHASE+5)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+5)%6)*STRIDE + (((8*PHASE+5)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+6)%6)*STRIDE + (((8*PHASE+6)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+6)%6)*STRIDE + (((8*PHASE+6)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+7)%6)*STRIDE + (((8*PHASE+7)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+7)%6)*STRIDE + (((8*PHASE+7)/6)+4)*Q) * 4U
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[VAOffset<4*PHASE>()], target);
if constexpr (!VA_GROUP38 || PHASE == 0U) StorePacked(mem.PackedVA()[VAOffset<24+SLOT*24+4*PHASE>()], source);
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitOutputPhase() {
static_assert(SLOT < (sizeof(T) == sizeof(float) ? 2U : 1U) && PHASE < 5U, "fixed output slots");
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t outputBase;
if constexpr (sizeof(T) == sizeof(float)) outputBase = reinterpret_cast<uint64_t>(mem.OutAt(SLOT).GetPhyAddr());
else outputBase = reinterpret_cast<uint64_t>(mem.PackedFloat().GetPhyAddr());
uint64_t source[16] = {
workBase + uint64_t(((8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 8) * 4U
};
uint64_t target[16] = {
outputBase + uint64_t((5*0+PHASE)*8) * 4U,
outputBase + uint64_t((5*8+PHASE)*8) * 4U,
outputBase + uint64_t((5*1+PHASE)*8) * 4U,
outputBase + uint64_t((5*9+PHASE)*8) * 4U,
outputBase + uint64_t((5*2+PHASE)*8) * 4U,
outputBase + uint64_t((5*10+PHASE)*8) * 4U,
outputBase + uint64_t((5*3+PHASE)*8) * 4U,
outputBase + uint64_t((5*11+PHASE)*8) * 4U,
outputBase + uint64_t((5*4+PHASE)*8) * 4U,
outputBase + uint64_t((5*12+PHASE)*8) * 4U,
outputBase + uint64_t((5*5+PHASE)*8) * 4U,
outputBase + uint64_t((5*13+PHASE)*8) * 4U,
outputBase + uint64_t((5*6+PHASE)*8) * 4U,
outputBase + uint64_t((5*14+PHASE)*8) * 4U,
outputBase + uint64_t((5*7+PHASE)*8) * 4U,
outputBase + uint64_t((5*15+PHASE)*8) * 4U
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[VAOffset<72+4*PHASE>()], source);
if constexpr (!VA_GROUP38 || PHASE == 0U) StorePacked(mem.PackedVA()[VAOffset<92+SLOT*20+4*PHASE>()], target);
}
template <uint32_t PHASE>
__aicore__ inline void InitPriorPhase() {
const uint64_t priorBase = reinterpret_cast<uint64_t>(mem.Prior().GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t source[16] = {
priorBase + uint64_t(0+8*PHASE) * 4U,
priorBase + uint64_t(40+8*PHASE) * 4U,
priorBase + uint64_t(80+8*PHASE) * 4U,
priorBase + uint64_t(120+8*PHASE) * 4U,
priorBase + uint64_t(160+8*PHASE) * 4U,
priorBase + uint64_t(200+8*PHASE) * 4U,
priorBase + uint64_t(240+8*PHASE) * 4U,
priorBase + uint64_t(280+8*PHASE) * 4U,
priorBase + uint64_t(0+8*PHASE) * 4U,
priorBase + uint64_t(40+8*PHASE) * 4U,
priorBase + uint64_t(80+8*PHASE) * 4U,
priorBase + uint64_t(120+8*PHASE) * 4U,
priorBase + uint64_t(160+8*PHASE) * 4U,
priorBase + uint64_t(200+8*PHASE) * 4U,
priorBase + uint64_t(240+8*PHASE) * 4U,
priorBase + uint64_t(280+8*PHASE) * 4U
};
uint64_t target[16] = {
workBase + uint64_t((6+(8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 104) * 4U,
workBase + uint64_t((6+(8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 104) * 4U,
workBase + uint64_t((6+(8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 104) * 4U,
workBase + uint64_t((6+(8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 104) * 4U,
workBase + uint64_t((6+(8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 104) * 4U,
workBase + uint64_t((6+(8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 104) * 4U,
workBase + uint64_t((6+(8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 104) * 4U,
workBase + uint64_t((6+(8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 104) * 4U
};
StorePacked(mem.PackedVA()[VAOffset<132+4*PHASE>()], target);
if constexpr (!VA_GROUP38 || PHASE == 0U) StorePacked(mem.PackedVA()[VAOffset<152+4*PHASE>()], source);
}
__aicore__ inline void InitPackedVA() {
InitInputPhase<0, 0>();
InitInputPhase<0, 1>();
InitInputPhase<0, 2>();
InitOutputPhase<0, 0>();
InitOutputPhase<0, 1>();
InitOutputPhase<0, 2>();
InitOutputPhase<0, 3>();
InitOutputPhase<0, 4>();
if constexpr (sizeof(T) == sizeof(float)) {
InitInputPhase<1, 0>();
InitInputPhase<1, 1>();
InitInputPhase<1, 2>();
InitOutputPhase<1, 0>();
InitOutputPhase<1, 1>();
InitOutputPhase<1, 2>();
InitOutputPhase<1, 3>();
InitOutputPhase<1, 4>();
}
InitPriorPhase<0>();
InitPriorPhase<1>();
InitPriorPhase<2>();
InitPriorPhase<3>();
InitPriorPhase<4>();
AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
if constexpr (VA_GROUP38) {
auto *entries=reinterpret_cast<__ubuf__ int16_t *>(mem.PackedVA().GetPhyAddr());
AscendC::SetMaskCount();
AscendC::SetVectorMask<int16_t, AscendC::MaskMode::COUNTER>(80U);
#pragma unroll
for (uint32_t phase=1U;phase<5U;++phase)
asc_add_scalar(entries+phase*80U,entries,int16_t(phase),
uint8_t(1),uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetMaskNorm();
AscendC::ResetMask();
}
}
template <uint32_t SLOT>
__aicore__ inline void Input() {
PackedTranspose<26, 1, 48>(mem.PackedVA()[VAOffset<0>()], mem.PackedVA()[VAOffset<24+SLOT*24+0>()]);
PackedTranspose<26, 1, 48>(mem.PackedVA()[VAOffset<4>()], mem.PackedVA()[VAOffset<24+SLOT*24+4>()]);
PackedTranspose<26, 1, 48>(mem.PackedVA()[VAOffset<8>()], mem.PackedVA()[VAOffset<24+SLOT*24+8>()]);
}
template <uint32_t SLOT>
__aicore__ inline void Output() {
PackedTranspose<13, 80, 2>(mem.PackedVA()[VAOffset<92+SLOT*20+0>()], mem.PackedVA()[VAOffset<72+0>()]);
PackedTranspose<13, 80, 2>(mem.PackedVA()[VAOffset<92+SLOT*20+4>()], mem.PackedVA()[VAOffset<72+4>()]);
PackedTranspose<13, 80, 2>(mem.PackedVA()[VAOffset<92+SLOT*20+8>()], mem.PackedVA()[VAOffset<72+8>()]);
PackedTranspose<13, 80, 2>(mem.PackedVA()[VAOffset<92+SLOT*20+12>()], mem.PackedVA()[VAOffset<72+12>()]);
PackedTranspose<13, 80, 2>(mem.PackedVA()[VAOffset<92+SLOT*20+16>()], mem.PackedVA()[VAOffset<72+16>()]);
}
public:
__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded) {
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
if (block >= 39U) return;
const uint32_t begin = block * 832U;
rawGm.SetGlobalBuffer((__gm__ T *)raw);
priorGm.SetGlobalBuffer((__gm__ float *)priors);
outGm.SetGlobalBuffer((__gm__ T *)decoded);
const AscendC::DataCopyExtParams rp{2, 832U*6U*uint32_t(sizeof(T)), (N-832U)*6U*uint32_t(sizeof(T)), 0, 0};
const AscendC::DataCopyExtParams pp{1, 4160U*4U, 0, 0, 0};
const uint32_t cohort = ROTATE_PAIRS ? (block & 3U) : 0U;
auto firstRaw = mem.AllocSlot(0);
auto p = mem.Prior();
AscendC::DataCopyPad(firstRaw, rawGm[(2U*cohort*N+begin)*6U], rp, AscendC::DataCopyPadExtParams<T>{false,0,0,0});
if constexpr (FIXED_TOKENS) AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
if constexpr (VA_GROUP38) {
static_assert(!0 && sizeof(T) == sizeof(float) && N == 32769U && COORD_DIV2 && FRAME == 1664U && STRIDE == 1680U, "qualified P6 prior ordinary520 blocks");
AscendC::DataCopy(p, priorGm[begin*5U], AscendC::DataCopyParams{1U,520U,0U,0U});
} else {
AscendC::DataCopyPad(p, priorGm[begin*5U], pp, AscendC::DataCopyPadExtParams<float>{false,0,0,0});
}
if constexpr (!FIXED_TOKENS) AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
auto one = mem.Ones();
AscendC::Duplicate(one, 1.0f, 8U);
if constexpr (COORD_DIV2) AscendC::Duplicate(mem.Twos(), 2.0f, 8U);
AscendC::LocalTensor<float> f[11];
for (uint32_t k=0;k<11;++k) f[k]=mem.Work()[k*STRIDE];
InitPackedVA();
if constexpr (!FIXED_TOKENS) {
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
PackedTranspose<13, 1, 40>(mem.PackedVA()[VAOffset<132+0>()], mem.PackedVA()[VAOffset<152+0>()]);
PackedTranspose<13, 1, 40>(mem.PackedVA()[VAOffset<132+4>()], mem.PackedVA()[VAOffset<152+4>()]);
PackedTranspose<13, 1, 40>(mem.PackedVA()[VAOffset<132+8>()], mem.PackedVA()[VAOffset<152+8>()]);
PackedTranspose<13, 1, 40>(mem.PackedVA()[VAOffset<132+12>()], mem.PackedVA()[VAOffset<152+12>()]);
PackedTranspose<13, 1, 40>(mem.PackedVA()[VAOffset<132+16>()], mem.PackedVA()[VAOffset<152+16>()]);
AscendC::PipeBarrier<PIPE_V>();
}
if constexpr (sizeof(T) != sizeof(float)) {
AscendC::DataCopy(p, f[8], STRIDE);
AscendC::DataCopy(p[STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[2*STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[3*STRIDE], f[8], STRIDE);
AscendC::PipeBarrier<PIPE_V>();
}
for (uint32_t iteration=0;iteration<4;++iteration) {
const uint32_t pair = ROTATE_PAIRS ? ((iteration+cohort)&3U) : iteration;
const uint32_t slot=iteration&1U;
const uint32_t pos=pair*2U*N+begin;
auto r=firstRaw;
if (iteration!=0) {
if constexpr (FIXED_TOKENS) {
if (iteration >= 2U) {
AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
}
r=mem.RawFixedSlot(slot);
} else
r=mem.AllocSlot(slot);
AscendC::DataCopyPad(r,rawGm[pos*6U],rp,AscendC::DataCopyPadExtParams<T>{false,0,0,0});
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot);
}
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);
AscendC::LocalTensor<float> rf;
if constexpr(sizeof(T)==sizeof(float)) rf=r.template ReinterpretCast<float>();
else {rf=mem.RawFloat();AscendC::Cast(rf,r,AscendC::RoundMode::CAST_NONE,9984U);}
if constexpr (VA_GROUP38) {
static_assert(!0 && sizeof(T) == sizeof(float) && N == 32769U && FRAME == 1664U && COORD_DIV2, "qualified P6 full832 signed32 local qualification");
auto signedRaw = rf.template ReinterpretCast<int32_t>();
AscendC::Adds<int32_t>(signedRaw, signedRaw, int32_t(-2147483647 - 1), 9984U);
} else {
AscendC::Muls(rf,rf,-1.0f,9984U);
}
AscendC::Exp(rf,rf,9984U);
if constexpr (FIXED_TOKENS) {
if (iteration == 0U) {
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
PackedTranspose<13, 1, 40>(mem.PackedVA()[VAOffset<132+0>()], mem.PackedVA()[VAOffset<152+0>()]);
PackedTranspose<13, 1, 40>(mem.PackedVA()[VAOffset<132+4>()], mem.PackedVA()[VAOffset<152+4>()]);
PackedTranspose<13, 1, 40>(mem.PackedVA()[VAOffset<132+8>()], mem.PackedVA()[VAOffset<152+8>()]);
PackedTranspose<13, 1, 40>(mem.PackedVA()[VAOffset<132+12>()], mem.PackedVA()[VAOffset<152+12>()]);
PackedTranspose<13, 1, 40>(mem.PackedVA()[VAOffset<132+16>()], mem.PackedVA()[VAOffset<152+16>()]);
}
}
AscendC::Adds(rf,rf,1.0f,9984U);
const AscendC::BinaryRepeatParams divBinary{1,0,1,8,0,8};
if constexpr (!SCORE_FUSED) AscendC::Div<float>(rf,one,rf,uint64_t(64),uint8_t(156),divBinary);
AscendC::PipeBarrier<PIPE_V>();
if constexpr(sizeof(T)==sizeof(float)) {if(slot==0)Input<0>();else Input<1>();}
else Input<0>();
AscendC::PipeBarrier<PIPE_V>();
if constexpr (SCORE_FUSED) {
AscendC::Mul(f[4],f[4],f[5],FRAME);
if constexpr (COORD_DIV2) {
AscendC::Div<float>(mem.Work(),mem.Twos(),mem.Work(),uint64_t(64),uint8_t(105),divBinary);
if constexpr (VA_GROUP38) {
static_assert(!0 && sizeof(T) == sizeof(float) && N == 32769U && FRAME == 1664U && STRIDE == 1680U, "qualified P6 raw score Div Normal64");
auto *scorePtr = reinterpret_cast<__ubuf__ float *>(f[4].GetPhyAddr());
auto *onePtr = reinterpret_cast<__ubuf__ float *>(one.GetPhyAddr());
asc_div(scorePtr, onePtr, scorePtr, uint8_t(26),
uint8_t(1),uint8_t(0),uint8_t(1),uint8_t(8),uint8_t(0),uint8_t(8));
} else {
AscendC::Div<float>(f[4],one,f[4],uint64_t(64),uint8_t(26),divBinary);
}
} else {
AscendC::Div<float>(mem.Work(),one,mem.Work(),uint64_t(64),uint8_t(132),divBinary);
}
AscendC::PipeBarrier<PIPE_V>();
}
if constexpr (!COORD_DIV2) AscendC::Muls(f[0],f[0],2.0f,4*STRIDE);
AscendC::Adds(f[0],f[0],-0.5f,2*STRIDE);
AscendC::Add(f[0],f[0],f[6],2*STRIDE);
AscendC::Mul(f[2],f[2],f[2],2*STRIDE);
AscendC::Mul(f[2],f[2],f[9],2*STRIDE);
if constexpr (sizeof(T) == sizeof(float)) {
AscendC::PipeBarrier<PIPE_V>();
auto *wPtr = reinterpret_cast<__ubuf__ float *>(mem.Work().GetPhyAddr());
AscendC::SetMaskCount();
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>((COORD_DIV2 && VA_GROUP38 && N == 32769U) ? FRAME : STRIDE);
#pragma unroll
for (uint32_t k=0;k<4;++k)
asc_mul(wPtr+k*STRIDE, wPtr+k*STRIDE, wPtr+8*STRIDE,
uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(1),uint8_t(8),uint8_t(8),uint8_t(8));
AscendC::SetMaskNorm();
AscendC::ResetMask();
} else {
AscendC::Mul(f[0],f[0],p,4*STRIDE);
}
if constexpr (!SCORE_FUSED) AscendC::Mul(f[4],f[4],f[5],FRAME);
AscendC::PipeBarrier<PIPE_V>();
if constexpr(sizeof(T)==sizeof(float)) {if(slot==0)Output<0>();else Output<1>();}
else Output<0>();
AscendC::PipeBarrier<PIPE_V>();
auto out=mem.OutAt(slot);
if constexpr(sizeof(T)!=sizeof(float)) {
AscendC::Cast(out,mem.PackedFloat(),AscendC::RoundMode::CAST_RINT,8320U);
AscendC::PipeBarrier<PIPE_V>();
}
AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot);
AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
const AscendC::DataCopyExtParams op{2,832U*5U*uint32_t(sizeof(T)),0,(N-832U)*5U*uint32_t(sizeof(T)),0};
AscendC::DataCopyPad(outGm[pos*5U],out,op);
if constexpr (FIXED_TOKENS) {
if (iteration < 2U) {
AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);
}
} else
mem.PublishOutputAndRelease(slot);
}
if constexpr (FIXED_TOKENS) {
// No raw slot is reused after round three; drain all ordered output DMA once.
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
} else
mem.Drain();
}
};


template <typename T, uint32_t N>
class KernelFireA2F6AddTail321 {
static_assert(!0 && sizeof(T) == sizeof(float) && N == 32769U, "qualified FP32 N32769 tail321 pairs");
static constexpr uint32_t COUNT = 321U, PITCH = 384U, FRAME = 768U, STRIDE = 784U, Q = 96U;
static_assert(PITCH * 2U == FRAME && COUNT < PITCH && PITCH % 128U == 0U && FRAME % 128U == 0U && Q == FRAME / 8U && 39U * 832U + COUNT == N, "exact tail pair geometry");
static_assert(PITCH * 6U * sizeof(T) == 9216U && PITCH * 5U * sizeof(T) == 7680U && ((COUNT * 6U * sizeof(T) + 31U) / 32U) + 47U == 288U && ((COUNT * 5U * sizeof(T) + 31U) / 32U) + 39U == 240U, "qualified byte and UB block gap geometry");
FireA2F6AddFrameStorage<T, FRAME> mem;
AscendC::GlobalTensor<T> rawGm, outGm;
AscendC::GlobalTensor<float> priorGm;
__aicore__ inline void StorePacked(AscendC::LocalTensor<uint64_t> dst, const uint64_t *address) {
#pragma unroll
for (uint32_t q = 0; q < 4; ++q) {
const uint64_t value = ((address[4*q] >> 5) & 0x1fffULL) |
(((address[4*q+1] >> 5) & 0x1fffULL) << 16) |
(((address[4*q+2] >> 5) & 0x1fffULL) << 32) |
(((address[4*q+3] >> 5) & 0x1fffULL) << 48);
dst.SetValue(q, value);
}
}
template <uint8_t REPEAT, uint16_t DST_STRIDE, uint16_t SRC_STRIDE>
__aicore__ inline void PackedTranspose(AscendC::LocalTensor<uint64_t> dst, AscendC::LocalTensor<uint64_t> src) {
auto *dstPacked = reinterpret_cast<__ubuf__ uint64_t *>(dst.GetPhyAddr());
auto *srcPacked = reinterpret_cast<__ubuf__ uint64_t *>(src.GetPhyAddr());
__cce_scalar::vld_va_reg(VA0, dstPacked, L128);
__cce_scalar::vld_va_reg(VA1, dstPacked, H128);
__cce_scalar::vld_va_reg(VA2, srcPacked, L128);
__cce_scalar::vld_va_reg(VA3, srcPacked, H128);
__cce_scalar::scatter_vnchwconv_b32(VA0, VA2, REPEAT, DST_STRIDE, SRC_STRIDE);
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitInputPhase() {
static_assert(SLOT < (sizeof(T) == sizeof(float) ? 2U : 1U) && PHASE < 3U, "fixed RAW3 slots");
uint64_t sourceBase;
if constexpr (sizeof(T) == sizeof(float)) sourceBase = reinterpret_cast<uint64_t>(mem.RawFixedSlot(SLOT).GetPhyAddr());
else sourceBase = reinterpret_cast<uint64_t>(mem.RawFloat().GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t source[16] = {
sourceBase + uint64_t(0 + 8*PHASE) * 4U,
sourceBase + uint64_t(48 + 8*PHASE) * 4U,
sourceBase + uint64_t(96 + 8*PHASE) * 4U,
sourceBase + uint64_t(144 + 8*PHASE) * 4U,
sourceBase + uint64_t(192 + 8*PHASE) * 4U,
sourceBase + uint64_t(240 + 8*PHASE) * 4U,
sourceBase + uint64_t(288 + 8*PHASE) * 4U,
sourceBase + uint64_t(336 + 8*PHASE) * 4U,
sourceBase + uint64_t(24 + 8*PHASE) * 4U,
sourceBase + uint64_t(72 + 8*PHASE) * 4U,
sourceBase + uint64_t(120 + 8*PHASE) * 4U,
sourceBase + uint64_t(168 + 8*PHASE) * 4U,
sourceBase + uint64_t(216 + 8*PHASE) * 4U,
sourceBase + uint64_t(264 + 8*PHASE) * 4U,
sourceBase + uint64_t(312 + 8*PHASE) * 4U,
sourceBase + uint64_t(360 + 8*PHASE) * 4U
};
uint64_t target[16] = {
workBase + uint64_t(((8*PHASE+0)%6)*STRIDE + (((8*PHASE+0)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+0)%6)*STRIDE + (((8*PHASE+0)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+1)%6)*STRIDE + (((8*PHASE+1)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+1)%6)*STRIDE + (((8*PHASE+1)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+2)%6)*STRIDE + (((8*PHASE+2)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+2)%6)*STRIDE + (((8*PHASE+2)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+3)%6)*STRIDE + (((8*PHASE+3)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+3)%6)*STRIDE + (((8*PHASE+3)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+4)%6)*STRIDE + (((8*PHASE+4)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+4)%6)*STRIDE + (((8*PHASE+4)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+5)%6)*STRIDE + (((8*PHASE+5)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+5)%6)*STRIDE + (((8*PHASE+5)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+6)%6)*STRIDE + (((8*PHASE+6)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+6)%6)*STRIDE + (((8*PHASE+6)/6)+4)*Q) * 4U,
workBase + uint64_t(((8*PHASE+7)%6)*STRIDE + (((8*PHASE+7)/6)+0)*Q) * 4U,
workBase + uint64_t(((8*PHASE+7)%6)*STRIDE + (((8*PHASE+7)/6)+4)*Q) * 4U
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[4*PHASE], target);
StorePacked(mem.PackedVA()[24+SLOT*24+4*PHASE], source);
}
template <uint32_t SLOT, uint32_t PHASE>
__aicore__ inline void InitOutputPhase() {
static_assert(SLOT < (sizeof(T) == sizeof(float) ? 2U : 1U) && PHASE < 5U, "fixed output slots");
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t outputBase;
if constexpr (sizeof(T) == sizeof(float)) outputBase = reinterpret_cast<uint64_t>(mem.OutAt(SLOT).GetPhyAddr());
else outputBase = reinterpret_cast<uint64_t>(mem.PackedFloat().GetPhyAddr());
uint64_t source[16] = {
workBase + uint64_t(((8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 0) * 4U,
workBase + uint64_t(((8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 8) * 4U,
workBase + uint64_t(((8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 8) * 4U
};
uint64_t target[16] = {
outputBase + uint64_t((5*0+PHASE)*8) * 4U,
outputBase + uint64_t((5*8+PHASE)*8) * 4U,
outputBase + uint64_t((5*1+PHASE)*8) * 4U,
outputBase + uint64_t((5*9+PHASE)*8) * 4U,
outputBase + uint64_t((5*2+PHASE)*8) * 4U,
outputBase + uint64_t((5*10+PHASE)*8) * 4U,
outputBase + uint64_t((5*3+PHASE)*8) * 4U,
outputBase + uint64_t((5*11+PHASE)*8) * 4U,
outputBase + uint64_t((5*4+PHASE)*8) * 4U,
outputBase + uint64_t((5*12+PHASE)*8) * 4U,
outputBase + uint64_t((5*5+PHASE)*8) * 4U,
outputBase + uint64_t((5*13+PHASE)*8) * 4U,
outputBase + uint64_t((5*6+PHASE)*8) * 4U,
outputBase + uint64_t((5*14+PHASE)*8) * 4U,
outputBase + uint64_t((5*7+PHASE)*8) * 4U,
outputBase + uint64_t((5*15+PHASE)*8) * 4U
};
if constexpr (SLOT == 0) StorePacked(mem.PackedVA()[72+4*PHASE], source);
StorePacked(mem.PackedVA()[92+SLOT*20+4*PHASE], target);
}
template <uint32_t PHASE>
__aicore__ inline void InitPriorPhase() {
const uint64_t priorBase = reinterpret_cast<uint64_t>(mem.PriorFixedSlot(0).GetPhyAddr());
const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
uint64_t source[16] = {
priorBase + uint64_t(0+8*PHASE) * 4U,
priorBase + uint64_t(40+8*PHASE) * 4U,
priorBase + uint64_t(80+8*PHASE) * 4U,
priorBase + uint64_t(120+8*PHASE) * 4U,
priorBase + uint64_t(160+8*PHASE) * 4U,
priorBase + uint64_t(200+8*PHASE) * 4U,
priorBase + uint64_t(240+8*PHASE) * 4U,
priorBase + uint64_t(280+8*PHASE) * 4U,
priorBase + uint64_t(0+8*PHASE) * 4U,
priorBase + uint64_t(40+8*PHASE) * 4U,
priorBase + uint64_t(80+8*PHASE) * 4U,
priorBase + uint64_t(120+8*PHASE) * 4U,
priorBase + uint64_t(160+8*PHASE) * 4U,
priorBase + uint64_t(200+8*PHASE) * 4U,
priorBase + uint64_t(240+8*PHASE) * 4U,
priorBase + uint64_t(280+8*PHASE) * 4U
};
uint64_t target[16] = {
workBase + uint64_t((6+(8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+0)%5)*STRIDE + ((8*PHASE+0)/5)*Q + 48) * 4U,
workBase + uint64_t((6+(8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+1)%5)*STRIDE + ((8*PHASE+1)/5)*Q + 48) * 4U,
workBase + uint64_t((6+(8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+2)%5)*STRIDE + ((8*PHASE+2)/5)*Q + 48) * 4U,
workBase + uint64_t((6+(8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+3)%5)*STRIDE + ((8*PHASE+3)/5)*Q + 48) * 4U,
workBase + uint64_t((6+(8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+4)%5)*STRIDE + ((8*PHASE+4)/5)*Q + 48) * 4U,
workBase + uint64_t((6+(8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+5)%5)*STRIDE + ((8*PHASE+5)/5)*Q + 48) * 4U,
workBase + uint64_t((6+(8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+6)%5)*STRIDE + ((8*PHASE+6)/5)*Q + 48) * 4U,
workBase + uint64_t((6+(8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 0) * 4U,
workBase + uint64_t((6+(8*PHASE+7)%5)*STRIDE + ((8*PHASE+7)/5)*Q + 48) * 4U
};
StorePacked(mem.PackedVA()[132+4*PHASE], target);
StorePacked(mem.PackedVA()[152+4*PHASE], source);
}
__aicore__ inline void InitPackedVA() {
InitInputPhase<0, 0>();
InitInputPhase<0, 1>();
InitInputPhase<0, 2>();
InitOutputPhase<0, 0>();
InitOutputPhase<0, 1>();
InitOutputPhase<0, 2>();
InitOutputPhase<0, 3>();
InitOutputPhase<0, 4>();
if constexpr (sizeof(T) == sizeof(float)) {
InitInputPhase<1, 0>();
InitInputPhase<1, 1>();
InitInputPhase<1, 2>();
InitOutputPhase<1, 0>();
InitOutputPhase<1, 1>();
InitOutputPhase<1, 2>();
InitOutputPhase<1, 3>();
InitOutputPhase<1, 4>();
}
InitPriorPhase<0>();
InitPriorPhase<1>();
InitPriorPhase<2>();
InitPriorPhase<3>();
InitPriorPhase<4>();
AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
}
template <uint32_t SLOT>
__aicore__ inline void Input() {
PackedTranspose<12, 1, 48>(mem.PackedVA()[0], mem.PackedVA()[24+SLOT*24+0]);
PackedTranspose<12, 1, 48>(mem.PackedVA()[4], mem.PackedVA()[24+SLOT*24+4]);
PackedTranspose<12, 1, 48>(mem.PackedVA()[8], mem.PackedVA()[24+SLOT*24+8]);
}
template <uint32_t SLOT>
__aicore__ inline void Output() {
PackedTranspose<6, 80, 2>(mem.PackedVA()[92+SLOT*20+0], mem.PackedVA()[72+0]);
PackedTranspose<6, 80, 2>(mem.PackedVA()[92+SLOT*20+4], mem.PackedVA()[72+4]);
PackedTranspose<6, 80, 2>(mem.PackedVA()[92+SLOT*20+8], mem.PackedVA()[72+8]);
PackedTranspose<6, 80, 2>(mem.PackedVA()[92+SLOT*20+12], mem.PackedVA()[72+12]);
PackedTranspose<6, 80, 2>(mem.PackedVA()[92+SLOT*20+16], mem.PackedVA()[72+16]);
}
public:
__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded) {
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
if (block != 39U) return;
constexpr uint32_t begin = 32448U;
rawGm.SetGlobalBuffer((__gm__ T *)raw);
priorGm.SetGlobalBuffer((__gm__ float *)priors);
outGm.SetGlobalBuffer((__gm__ T *)decoded);
const AscendC::DataCopyExtParams rp{2, COUNT * 6U * uint32_t(sizeof(T)), (N - COUNT) * 6U * uint32_t(sizeof(T)), 47, 0};
const AscendC::DataCopyExtParams pp{1, COUNT * 5U * uint32_t(sizeof(float)), 0, 0, 0};
auto firstRaw = mem.AllocRaw();
auto p = mem.AllocPrior();
AscendC::DataCopyPad(firstRaw, rawGm[begin * 6U], rp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
AscendC::DataCopyPad(p, priorGm[begin * 5U], pp, AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
mem.RawReady();
mem.PriorReady();
auto one = mem.Ones();
AscendC::Duplicate(one, 1.0f, FRAME * 6U);
AscendC::LocalTensor<float> f[11];
for (uint32_t k = 0; k < 11; ++k) f[k] = mem.Work()[k * STRIDE];
InitPackedVA();
mem.PriorWait();
PackedTranspose<6, 1, 40>(mem.PackedVA()[132+0], mem.PackedVA()[152+0]);
PackedTranspose<6, 1, 40>(mem.PackedVA()[132+4], mem.PackedVA()[152+4]);
PackedTranspose<6, 1, 40>(mem.PackedVA()[132+8], mem.PackedVA()[152+8]);
PackedTranspose<6, 1, 40>(mem.PackedVA()[132+12], mem.PackedVA()[152+12]);
PackedTranspose<6, 1, 40>(mem.PackedVA()[132+16], mem.PackedVA()[152+16]);
AscendC::PipeBarrier<PIPE_V>();
AscendC::DataCopy(p, f[8], STRIDE);
AscendC::DataCopy(p[STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[2 * STRIDE], f[8], STRIDE);
AscendC::DataCopy(p[3 * STRIDE], f[8], STRIDE);
AscendC::PipeBarrier<PIPE_V>();
for (uint32_t iteration = 0; iteration < 4U; ++iteration) {
const uint32_t slot = iteration & 1U;
const uint32_t pos = iteration * 2U * N + begin;
auto r = firstRaw;
if (iteration != 0U) {
r = mem.AllocRaw();
AscendC::DataCopyPad(r, rawGm[pos * 6U], rp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
mem.RawReady();
}
mem.RawWait();
auto rf = r.template ReinterpretCast<float>();
AscendC::Muls(rf, rf, -1.0f, FRAME * 6U);
AscendC::Exp(rf, rf, FRAME * 6U);
AscendC::Adds(rf, rf, 1.0f, FRAME * 6U);
AscendC::Div(rf, one, rf, FRAME * 6U);
AscendC::PipeBarrier<PIPE_V>();
if (slot == 0U) Input<0>(); else Input<1>();
AscendC::PipeBarrier<PIPE_V>();
AscendC::Muls(f[0], f[0], 2.0f, 4 * STRIDE);
AscendC::Adds(f[0], f[0], -0.5f, 2 * STRIDE);
AscendC::Add(f[0], f[0], f[6], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[2], 2 * STRIDE);
AscendC::Mul(f[2], f[2], f[9], 2 * STRIDE);
AscendC::Mul(f[0], f[0], p, 4 * STRIDE);
AscendC::Mul(f[4], f[4], f[5], FRAME);
auto out = mem.AllocOut();
AscendC::PipeBarrier<PIPE_V>();
if (slot == 0U) Output<0>(); else Output<1>();
AscendC::PipeBarrier<PIPE_V>();
mem.OutReady();
mem.OutWait();
const AscendC::DataCopyExtParams op{2, COUNT * 5U * uint32_t(sizeof(T)), 39, (N - COUNT) * 5U * uint32_t(sizeof(T)), 0};
AscendC::DataCopyPad(outGm[pos * 5U], out, op);
mem.FreeOut();
mem.FreeRaw();
}
mem.FreePrior();
mem.Drain();
}
};


template <typename DT_RAW_PRED, uint32_t REUSE>
__global__ __aicore__ void fire_candidate_decode_fusion(GM_ADDR raw_pred, GM_ADDR priors,
GM_ADDR decoded, GM_ADDR workspace, GM_ADDR tiling) {
if constexpr (REUSE == 0x7b36U) {
constexpr uint32_t activeMask = 7U;
if ((static_cast<uint32_t>(AscendC::GetBlockIdx()) & activeMask) != 0U) {
return;
}
}
if constexpr (REUSE == 0x7b38U) {
const uint32_t logicalBlock = static_cast<uint32_t>(AscendC::GetBlockIdx());
if ((logicalBlock & 1U) != 0U || logicalBlock >= 6U) {
AscendC::PipeBarrier<PIPE_ALL>();
return;
}
}
AscendC::InitSocState();
REGISTER_TILING_DEFAULT(FireFrameTilingData);
REGISTER_TILING_FOR_TILINGKEY("(TILING_KEY_VAR == 16128) || (TILING_KEY_VAR == 16129) || (TILING_KEY_VAR == 16384) || (TILING_KEY_VAR == 16385) || (TILING_KEY_VAR == 18432) || (TILING_KEY_VAR == 18433)", FireFrameTilingData);
if constexpr (REUSE == 0xb240U) {
AscendC::ICachePreLoad(3);
constexpr uint32_t BATCHES = 4;
constexpr uint32_t FRAME = 512;
static_assert(BATCHES >= 1 && BATCHES <= 8 && FRAME == 512, "whole resource geometry");
const uint32_t cores = sizeof(DT_RAW_PRED) == sizeof(float) ? 32U : static_cast<uint32_t>(AscendC::GetBlockNum());
const uint32_t rowLanes = cores / BATCHES;
const FireFrameTilingData schedule{{cores * FRAME, rowLanes * FRAME, FRAME}, rowLanes | (1U << 16)};
KernelFireFrame<DT_RAW_PRED, FRAME, true, true, true, sizeof(DT_RAW_PRED) == sizeof(half), sizeof(DT_RAW_PRED) == sizeof(float), sizeof(DT_RAW_PRED) == sizeof(half), sizeof(DT_RAW_PRED) == sizeof(half), sizeof(DT_RAW_PRED) == sizeof(float), sizeof(DT_RAW_PRED) == sizeof(half) ? 8U : 16U> kernel;
kernel.template Run<true>(raw_pred, priors, decoded, schedule);
} else if constexpr (REUSE == 0xb500U || REUSE == 0xb501U) {
AscendC::ICachePreLoad(3);
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
constexpr uint32_t N = REUSE == 0xb500U ? 32768U : 32769U;
if constexpr (sizeof(DT_RAW_PRED) == sizeof(float)) {
const uint32_t launched = static_cast<uint32_t>(AscendC::GetBlockNum());
if (launched == 43U) {
if constexpr (N == 32768U) {
if (block < 42U) {
KernelFireA2F5Pair<DT_RAW_PRED, N, true, true, true, false, true, true, 768U> kernel;
kernel.Run(raw_pred, priors, decoded);
} else if (block == 42U) {
KernelFireA2F5Tail512<DT_RAW_PRED, N> kernel;
kernel.Run(raw_pred, priors, decoded);
}
} else {
if (block < 42U) {
KernelFireA2F6SpanPair<DT_RAW_PRED, N, true, true, true, false, true, true, 768U> kernel;
kernel.Run(raw_pred, priors, decoded);
} else if (block == 42U) {
KernelFireA2F6SpanTail513<DT_RAW_PRED, N> kernel;
kernel.Run(raw_pred, priors, decoded);
}
}
return;
}
if (block < 39U) {
if constexpr (N == 32768U) {
KernelFireA2F5Pair<DT_RAW_PRED, N, true, true, true, false, true, true> kernel;
kernel.Run(raw_pred, priors, decoded);
} else {
KernelFireA2F6AddPair<DT_RAW_PRED, N, true, true, true, false, true, true> kernel;
kernel.Run(raw_pred, priors, decoded);
}
} else if (block == 39U) {
if constexpr (N == 32768U) {
KernelFireA2F5Tail320<DT_RAW_PRED, N> kernel;
kernel.Run(raw_pred, priors, decoded);
} else {
KernelFireA2F6AddTail321<DT_RAW_PRED, N> kernel;
kernel.Run(raw_pred, priors, decoded);
}
}
return;
}
if (block >= 40) return;
const FireFrameTilingData schedule{{8U * N, N, 832}, (8U << 16) | 40U};
if (block < 39U) {
#if FIRE_PAIR_QUAL11_IMAGE
if constexpr (sizeof(DT_RAW_PRED) == sizeof(float) && (N == 32768U || N == 32769U)) {
KernelFirePair576<DT_RAW_PRED, N, true, true, true, false> kernel;
kernel.Run(raw_pred, priors, decoded);
} else
#endif
#if FIRE_PAIR_HALF_SCORE_IMAGE
if constexpr (sizeof(DT_RAW_PRED) == sizeof(half)) {
if constexpr (N == 32768U) {
KernelFirePairInputPi8H11<DT_RAW_PRED,N,true> kernel;
kernel.Run(raw_pred,priors,decoded);
} else {
KernelFirePairInputPi8H12<DT_RAW_PRED,N,true> kernel;
kernel.Run(raw_pred,priors,decoded);
}
} else
#endif
{
KernelFirePair576<DT_RAW_PRED, N, (sizeof(DT_RAW_PRED) == sizeof(float) && N == 32768U), (sizeof(DT_RAW_PRED) == sizeof(float) && N == 32769U)> kernel;
kernel.Run(raw_pred, priors, decoded);
}
} else {
#if FIRE_PAIR_HALF_SCORE_IMAGE
if constexpr (sizeof(DT_RAW_PRED) == sizeof(half)) {
KernelFireFrameStock1112<DT_RAW_PRED,384,false,true,false,N> kernel;
kernel.Run(raw_pred,priors,decoded,schedule);
} else
#endif
{
KernelFireFrame<DT_RAW_PRED,896,false,true> kernel;
kernel.Run(raw_pred,priors,decoded,schedule);
}
}
} else if constexpr (REUSE == 0xb602U || REUSE == 0xb603U) {
AscendC::ICachePreLoad(3);
const uint32_t block=static_cast<uint32_t>(AscendC::GetBlockIdx());
if(block>=47U)return;
constexpr uint32_t N=REUSE==0xb602U?32768U:32769U;
if constexpr(sizeof(DT_RAW_PRED)==sizeof(half)) {
if(block<46U) {
KernelFirePairSpan47<DT_RAW_PRED,N,704U,704U,(N==32769U)> kernel;
kernel.Run(raw_pred,priors,decoded);
} else if constexpr(N==32768U) {
KernelFirePairSpan47<DT_RAW_PRED,N,384U,384U> kernel;
kernel.Run(raw_pred,priors,decoded);
} else {
KernelFirePairSpan47<DT_RAW_PRED,N,448U,385U> kernel;
kernel.Run(raw_pred,priors,decoded);
}
} else {
const FireCandidateDecodeFusionTilingData schedule{8U*N,N,(8U*N+46U)/47U};
KernelFire<DT_RAW_PRED,832U> kernel;
kernel.Run(raw_pred,priors,decoded,schedule);
}
} else if constexpr (REUSE == 0xb600U || REUSE == 0xb601U) {
AscendC::ICachePreLoad(3);
const uint32_t block=static_cast<uint32_t>(AscendC::GetBlockIdx());
if(block>=43U)return;
constexpr uint32_t N=REUSE==0xb600U?32768U:32769U;
if constexpr(sizeof(DT_RAW_PRED)==sizeof(half)) {
if(block<42U) {
KernelFirePairSpan1112<DT_RAW_PRED,N,768U,768U,(N==32769U)> kernel;
kernel.Run(raw_pred,priors,decoded);
} else if constexpr(N==32768U) {
KernelFirePairSpan1112<DT_RAW_PRED,N,512U,512U> kernel;
kernel.Run(raw_pred,priors,decoded);
} else {
KernelFirePairSpan1112<DT_RAW_PRED,N,576U,513U> kernel;
kernel.Run(raw_pred,priors,decoded);
}
} else {
const FireCandidateDecodeFusionTilingData schedule{8U*N,N,(8U*N+42U)/43U};
KernelFire<DT_RAW_PRED,832U> kernel;
kernel.Run(raw_pred,priors,decoded,schedule);
}
} else if constexpr (REUSE == 0xb280U) {
AscendC::ICachePreLoad(3);
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
if (block >= 20U) return;
constexpr FireFrameTilingData schedule{{16388U, 4097U, 832U}, (1U << 16) | 5U};
if constexpr (sizeof(DT_RAW_PRED) == sizeof(float)) {
KernelFireA2RowF4<DT_RAW_PRED, 896, true, true, false, 4097U, true, true> kernel;
kernel.Run(raw_pred, priors, decoded, schedule);
} else {
KernelFireA2RowH10<DT_RAW_PRED, 896, true, true, false, true, 4097U, true> kernel;
kernel.Run(raw_pred, priors, decoded, schedule);
}
} else if constexpr (REUSE == 0xb300U) {
AscendC::ICachePreLoad(3);
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
if (block >= 48) return;
GET_TILING_DATA_PTR_WITH_STRUCT(FireFrameTilingData, schedulePtr, tiling);
const FireFrameTilingData schedule{{(*schedulePtr).header.total,
(*schedulePtr).header.n,
(*schedulePtr).header.perCore},
(*schedulePtr).geometry};
KernelFireFrame<DT_RAW_PRED, 768, false, true> kernel;
kernel.Run(raw_pred, priors, decoded, schedule);
} else if constexpr ((REUSE & 0xc000U) == 0xc000U) {
AscendC::ICachePreLoad(3);
const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
if (block >= 48) return;
GET_TILING_DATA_PTR_WITH_STRUCT(FireFrameTilingData, schedulePtr, tiling);
const FireFrameTilingData schedule{{(*schedulePtr).header.total,
(*schedulePtr).header.n,
(*schedulePtr).header.perCore},
(*schedulePtr).geometry};
constexpr uint32_t FRAME = REUSE & 0x1fffU;
constexpr bool ROW = (REUSE & 0x2000U) != 0;
KernelFireFrame<DT_RAW_PRED, FRAME, ROW> kernel;
kernel.Run(raw_pred, priors, decoded, schedule);
} else if constexpr (REUSE == 0x7b36U) {
AscendC::ICachePreLoad(2);
if constexpr (sizeof(DT_RAW_PRED) == sizeof(half)) {
KernelFireEven8Half32<DT_RAW_PRED> kernel;
kernel.Run(raw_pred, priors, decoded);
} else {
const FireCandidateDecodeFusionTilingData t{32, 32, 32};
KernelFireFlatCompositeInventory684<DT_RAW_PRED, true, true, FireHostComposite32Stride64Storage678<DT_RAW_PRED>, true, false, false> kernel;
kernel.Run(raw_pred, priors, decoded, tiling, t);
}
} else if constexpr (REUSE == 0x7b2eU) {
AscendC::ICachePreLoad(2);
const FireCandidateDecodeFusionTilingData t{32, 32, 32};
if constexpr (sizeof(DT_RAW_PRED) == sizeof(float)) {
KernelFireFlatCompositeInventory684<DT_RAW_PRED, true, true, FireHostComposite32Stride64Storage678<DT_RAW_PRED>, true, false, false> kernel;
kernel.Run(raw_pred, priors, decoded, tiling, t);
} else {
KernelFireFlatCompositeEarlyOne684<DT_RAW_PRED, true, true, FireHostComposite32Storage<DT_RAW_PRED>, true, false, true> kernel;
kernel.Run(raw_pred, priors, decoded, tiling, t);
}
} else if constexpr (REUSE == 0x7b34U || REUSE == 0x7b38U) {
AscendC::ICachePreLoad(2);
KernelFireFlatComposite<DT_RAW_PRED, false, false, FireFlatCompositeStorage<DT_RAW_PRED>, true, true> kernel;
kernel.template RunPacket<66, 33, REUSE == 0x7b38U>(raw_pred, priors, decoded, tiling);
} else {
constexpr int64_t PRELOAD_LEN = (REUSE == 0 || REUSE == 1) ? 2 : 3;
AscendC::ICachePreLoad(PRELOAD_LEN);
GET_TILING_DATA_WITH_STRUCT(FireCandidateDecodeFusionTilingData, t, tiling);
if constexpr (REUSE <= 2) {
constexpr uint32_t TILE = REUSE == 0 ? 64 : REUSE == 1 ? 256 : 832;
KernelFire<DT_RAW_PRED, TILE> kernel;
kernel.Run(raw_pred, priors, decoded, t);

} else {
KernelFireRow<DT_RAW_PRED, REUSE == 5256 ? 256 : 832> kernel;
kernel.Run(raw_pred, priors, decoded, t);
}
}
AscendC::PipeBarrier<PIPE_ALL>();
}

#undef FIRE_PAIR_QUAL11_IMAGE
#undef FIRE_PAIR_HALF_SCORE_IMAGE

#undef FIRE_P8O_INDEX
#undef FIRE_P8O_HALF_IMAGE
#undef FIRE_P8O_TU
#undef FIRE_P8O_TU_I
#undef FIRE_P8O_TU_half
#undef FIRE_P8O_TU_float
