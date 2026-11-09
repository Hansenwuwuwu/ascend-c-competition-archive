#pragma once
#include <cstdint>
// Tiling key bits follow the public AscendC template declaration order.
#define PINV_KEY_COMPACT 256UL
#define PINV_KEY_TAIL 512UL
#define PINV_KEY_WAVE 1024UL
#define PINV_KEY_LOCAL 2048UL
#define PINV_KEY_GROUP 4096UL
#define PINV_KEY_PIPELINE 8192UL
#define PINV_KEY_WINDOW 16384UL
#define PINV_KEY_CUBE 32768UL

#if defined(__NPU_ARCH__)
#define PINV_LAYOUT_FN __aicore__ inline
#else
#define PINV_LAYOUT_FN inline
#endif
namespace pinv {
constexpr float kSplitFactor = 4097.0f; // 2^12+1 splits an FP32 significand.
constexpr float kUnitRoundoff = 5.9604644775390625e-8f; // 2^-24, round-to-nearest.
constexpr float kTripleRoundoff = 2.117582368135751e-22f; // 2^-72 for posterior accumulation.
constexpr float kWideRoundoff = 3.552713678800501e-15f; // 2^-48 before operation factors.
constexpr float kScaleLimit = 1.7014118e38f; // 2^127: a finite power-of-two guard.
constexpr float kDirectGateSquared = 0.81f; // Conservative 0.9 inverse-norm cutoff.
constexpr float kGapGateSquared = 81.0f; // 9^2; discarded sigma is at most cutoff/10.
constexpr float kDirectConditionLimit = 3.6e9f; // Empirical FP32 route; not an error proof.
constexpr float kRefineConditionLimit = 4.0e11f; // Refinement admission still requires a certificate.
constexpr float kOutputAtol = 1.0e-4f;
constexpr float kOutputRtol = 1.0e-4f;

constexpr uint32_t kJacobiSweeps = 40;
constexpr uint32_t kRectRobustSweeps = 120;
constexpr float kFp32Orthogonality = 1.0e-6f;
constexpr float kRectWideOrthogonality = 2.0e-13f;
constexpr float kFp32OrthogonalitySquared = 1.0e-12f;
constexpr float kJacobiDiscardMargin = 1.0e-8f;
constexpr float kJacobiRoundoffFloor = 1.0e-14f;
constexpr float kJacobiNormCacheRatio = 0.35f;
constexpr float kJacobiLargeTau = 1.0e8f; // The omitted reciprocal term is below the Wide unit.
constexpr float kRectMaxGramDefect = 0.25f; // Polar correction requires a defect below one.
constexpr float kNormSlack = 1.001f; // Covers FP32 positive norm and bound aggregation.
constexpr float kResidualSlack = 1.002f; // Positive residual-bound aggregation margin.
constexpr float kRectBoundSlack = 1.01f;
constexpr float kRectOpFactor = 8.0f;
constexpr float kRectPairRoundOps = 32.0f;
constexpr float kRectDotCompressOps = 8.0f;
constexpr uint32_t kRectDotOverhead = 4;
constexpr uint32_t kRectNormOverhead = 8;

// All metadata indices below are float elements, not byte offsets.
// Pivot selection, Jacobi norms, and pipeline packets are distinct phases.
// Their regions may alias only after the preceding phase has synchronized.
constexpr uint32_t kMetadataFloats = 8192;
constexpr uint32_t kPeerStride = 8;
constexpr uint32_t kPivotIndex = 256;
constexpr uint32_t kJacobiNorm = 512;
constexpr uint32_t kGroupFlags = 1024;
constexpr uint32_t kPrepared = 1032;
constexpr uint32_t kGroupStatus = 2048;
constexpr uint32_t kPeerExchange = 3072;
constexpr uint32_t kCubeControlFloats = 512;
constexpr uint32_t kCubePlanes = 5;
constexpr float kCubeInputScale = 8.0f; // Initial balanced digit fits INT8.
constexpr float kCubeResidualUnit = 3.552713678800501e-15f; // 256^-6 after five digit planes.
constexpr float kCubeRadix = 256.0f;
constexpr float kCubeReciprocalRadix = 0.00390625f;
constexpr float kCubeLeadingWeight = 1.0f / 65536.0f;
constexpr uint32_t kCubeAlignFloats = 128;

PINV_LAYOUT_FN constexpr uint64_t MatrixPairFloats(uint32_t batch, uint32_t cols, uint32_t stride) {
    return uint64_t(2) * batch * cols * stride;
}
PINV_LAYOUT_FN constexpr uint64_t MetadataOffset(uint32_t batch, uint32_t cols, uint32_t stride, uint32_t index) {
    return MatrixPairFloats(batch, cols, stride) + uint64_t(index) * kMetadataFloats;
}
PINV_LAYOUT_FN constexpr uint64_t CubeOffsetFloats(uint32_t batch, uint32_t cols, uint32_t stride) {
    return (MetadataOffset(batch, cols, stride, batch) + kCubeAlignFloats - 1) / kCubeAlignFloats * kCubeAlignFloats;
}
PINV_LAYOUT_FN constexpr uint64_t CubePlaneElements(uint32_t cols) {
    return uint64_t(kCubePlanes)*cols*cols;
}
PINV_LAYOUT_FN constexpr uint64_t CubeProductOffset(uint32_t cols) {
    return 2*CubePlaneElements(cols)*sizeof(int8_t)/sizeof(int32_t);
}
PINV_LAYOUT_FN constexpr uint64_t CubeStorageBytes(uint32_t cols) {
    // Two 5*n*n INT8 arrays, one (5*n)*(5*n) INT32 array, and FP32 control.
    return uint64_t(kCubeControlFloats) * sizeof(float)
        + uint64_t(cols) * cols * (2 * kCubePlanes * sizeof(int8_t)
        + kCubePlanes * kCubePlanes * sizeof(int32_t));
}
PINV_LAYOUT_FN constexpr uint32_t RectAxis(uint32_t size) {
    return size<=8?8:(size<=16?16:(size<=32?32:(size<=64?64:(size<=128?128:(size<=256?256:512)))));
}
PINV_LAYOUT_FN constexpr uint32_t RectWidth(uint32_t rows,uint32_t cols) {
    return RectAxis(rows)+RectAxis(cols);
}
PINV_LAYOUT_FN constexpr uint64_t RectWorkspaceFloats(uint32_t batch,uint32_t rows,uint32_t cols) {
    return uint64_t(2)*batch*cols*RectWidth(rows,cols);
}
PINV_LAYOUT_FN constexpr uint32_t RectLocalFloats(uint32_t rows,uint32_t cols) {
    return 20*RectWidth(rows,cols)+32+8*rows+3*RectAxis(cols);
}
static_assert(sizeof(float)==sizeof(int32_t), "Cube product word size changed");
static_assert(kCubePlanes==5, "exact residual precision requires five digit planes");
static_assert(kPrepared + kPeerStride <= kGroupStatus, "prepared metadata exceeds its phase");
static_assert(kGroupStatus + 128 * kPeerStride <= kPeerExchange, "group status exceeds its phase");
static_assert(kPeerExchange < kMetadataFloats, "peer exchange must fit metadata");
}
#undef PINV_LAYOUT_FN

// Shape/ABI planning only. No input values or numerical decisions.
#if defined(__NPU_ARCH__)
#define PV_STAGE_LAYOUT_FN __aicore__ inline
#else
#define PV_STAGE_LAYOUT_FN inline
#endif
namespace pinv_dsl {
PV_STAGE_LAYOUT_FN uint64_t WorkFloats(uint32_t batch,uint32_t rows,uint32_t cols,uint32_t stride) {
    uint64_t rect=pinv::RectWorkspaceFloats(batch,rows,cols);
    uint64_t peer=batch*(2*uint64_t(cols)*stride+pinv::kMetadataFloats);
    uint64_t words=rect>peer?rect:peer;
    return (words+127)/128*128;
}
}
#undef PV_STAGE_LAYOUT_FN


#if defined(__NPU_ARCH__)
#define PV_WINDOW_LAYOUT_FN __aicore__ inline
#else
#define PV_WINDOW_LAYOUT_FN inline
#endif
namespace pinv_dsl {
PV_WINDOW_LAYOUT_FN bool WindowSupported(uint32_t m,uint32_t n,uint32_t batch,
    uint32_t stride,uint32_t window,uint32_t blocks) {
    uint64_t cores=uint64_t(blocks)*2;
    if(m!=n || !n || n>256 || n%128 || !batch || !cores || cores%batch ||
       cores*n>16384 || cores*16>1024 || cores*8+32>512)return false;
    uint64_t group=cores/batch;
    if(group<2 || group>48 || !window || window>=n || window%16)return false;
    uint64_t k=(n+7)/8*8,wide=(k+63)/64*64;
    uint64_t capacity=uint64_t(n)*window;
    uint64_t producer=capacity+4096+10*k+2*wide+64+(n+15)/16*64;
    uint64_t owned=(n+group-2)/(group-1),factors=(owned+7)/8*8;
    uint64_t output=owned*k>512?owned*k:512;
    uint64_t worker=owned*stride+16*(k+16)+output+8*(factors>16?factors:16)
                    +64+5*wide+32+factors+16+k;
    return producer<=49088 && worker<=49088 && (n+cores-1)/cores*n<=2048;
}
PV_WINDOW_LAYOUT_FN uint64_t WindowWorkFloats(uint32_t batch,uint32_t n,
    uint32_t stride,uint32_t blocks) {
    uint64_t extent=uint64_t(n)*n,cores=uint64_t(blocks)*2;
    uint64_t offset=(2*uint64_t(batch)*n*stride+uint64_t(batch)*8192+127)/128*128;
    uint64_t words=offset+61*extent/2+cores*n+cores*16+uint64_t(batch)*64;
    // The block strategy keeps thirteen batch-sized correction planes.
    uint64_t block=13*uint64_t(batch)*extent;
    return ((words>block?words:block)+127)/128*128;
}
PV_WINDOW_LAYOUT_FN uint64_t WindowStatusFloats(uint32_t batch) {
    return uint64_t(batch)*512+128;
}
}
#undef PV_WINDOW_LAYOUT_FN

#if !defined(__NPU_ARCH__)
namespace pinv_dsl {
inline bool BlockGridCompatible(uint32_t n,uint32_t stride,uint32_t batch,uint32_t blocks) {
    if(!blocks || !batch || n/64<4 || n>256 || n%128 || batch>16)return false;
    uint64_t cores=2*uint64_t(blocks),group=cores/batch;
    return !(cores%batch) && group>1 && uint64_t(batch)*n/64<=cores &&
           uint64_t(batch)*n/2<=1280 &&
           13*uint64_t(batch)*n*n<=WindowWorkFloats(batch,n,stride,blocks) &&
           (n+group-2)/(group-1)>=16;
}
inline uint32_t BlockLaunch(uint32_t n,uint32_t stride,uint32_t batch,
                           uint32_t window,uint32_t blocks) {
    if(!BlockGridCompatible(n,stride,batch,blocks))return blocks;
    uint64_t tiles=uint64_t(batch)*(n/64)*(n/128);
    uint64_t rounds=(tiles+blocks-1)/blocks,balance=batch%2?batch:batch/2;
    uint64_t target=(tiles+rounds-1)/rounds;
    target=(target+balance-1)/balance*balance;
    if(!target || target>=blocks || !BlockGridCompatible(n,stride,batch,target) ||
       !WindowSupported(n,n,batch,stride,window,target))return blocks;
    return uint32_t(target);
}
}
#endif
