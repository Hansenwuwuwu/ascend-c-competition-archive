#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 11265UL)
#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "swi_glu_tiling.h"
#include "tiling_key_swi_glu.h"
using namespace AscendC;
namespace {
__aicore__ inline uint32_t AlignUpBytes(uint32_t bytes)
{
return (bytes + 31U) / 32U * 32U;
}
__aicore__ inline uint32_t AlignUp16(uint32_t elems)
{
return (elems + 15U) / 16U * 16U;
}
constexpr uint32_t kChunk16 = 16;
__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
return (a + b - 1) / b;
}
}
template <typename T, bool FillOnes = false>
__aicore__ inline void GateCounter(const LocalTensor<T>& g,const LocalTensor<T>& u,
const LocalTensor<T>& sig,const LocalTensor<T>& out,
const LocalTensor<T>& ones,uint32_t count) {
constexpr uint64_t mask=0;
const UnaryRepeatParams ur{1,1,8,8};
const BinaryRepeatParams br{1,1,1,8,8,8};
SetMaskCount();SetVectorMask<T,MaskMode::COUNTER>(0,count);

Muls<T,false>(sig,g,static_cast<T>(-1),mask,1,ur);
PipeBarrier<PIPE_V>();Exp<T,false>(sig,sig,mask,1,ur);
PipeBarrier<PIPE_V>();Adds<T,false>(sig,sig,static_cast<T>(1),mask,1,ur);
PipeBarrier<PIPE_V>();Reciprocal<T,false>(sig,sig,mask,1,ur);
PipeBarrier<PIPE_V>();Mul<T,false>(out,g,sig,mask,1,br);
PipeBarrier<PIPE_V>();Mul<T,false>(out,out,u,mask,1,br);
SetMaskNorm();ResetMask();
}
template <typename T,uint32_t Count>
__aicore__ inline void GateNormal(const LocalTensor<T>& g,const LocalTensor<T>& u,
const LocalTensor<T>& sig,const LocalTensor<T>& out,const LocalTensor<T>& ones) {
static_assert(Count<=128U);
constexpr uint64_t lo=Count>=64U?~0ULL:((1ULL<<Count)-1ULL);
constexpr uint64_t hi=Count<=64U?0ULL:((1ULL<<(Count-64U))-1ULL);
const UnaryRepeatParams ur{1,1,8,8};const BinaryRepeatParams br{1,1,1,8,8,8};
SetVectorMask<T,MaskMode::NORMAL>(hi,lo);
Muls<T,false>(sig,g,static_cast<T>(-1),static_cast<uint64_t>(0),1,ur);
PipeBarrier<PIPE_V>();Exp<T,false>(sig,sig,static_cast<uint64_t>(0),1,ur);
PipeBarrier<PIPE_V>();Adds<T,false>(sig,sig,static_cast<T>(1),static_cast<uint64_t>(0),1,ur);
PipeBarrier<PIPE_V>();Div<T,false>(sig,ones,sig,static_cast<uint64_t>(0),1,br);
PipeBarrier<PIPE_V>();Mul<T,false>(out,g,sig,static_cast<uint64_t>(0),1,br);
PipeBarrier<PIPE_V>();Mul<T,false>(out,out,u,static_cast<uint64_t>(0),1,br);
ResetMask();
}
template <typename T, uint32_t Count, bool SetMask = true>
__aicore__ inline void FastGate(const LocalTensor<T>& g, const LocalTensor<T>& u,
const LocalTensor<T>& sig, const LocalTensor<T>& out, const LocalTensor<T>& ones) {
static_assert(Count <= 128U);
constexpr uint64_t lo = Count >= 64U ? ~0ULL : ((1ULL << Count) - 1ULL);
constexpr uint64_t hi = Count <= 64U ? 0ULL : (Count >= 128U ? ~0ULL : ((1ULL << (Count > 64U ? Count - 64U : 0U)) - 1ULL));
const UnaryRepeatParams ur{1, 1, 8, 8};
const BinaryRepeatParams br{1, 1, 1, 8, 8, 8};
if constexpr (SetMask) {
SetVectorMask<T, MaskMode::NORMAL>(hi, lo);
}
Muls<T, false>(sig, g, static_cast<T>(-1), static_cast<uint64_t>(0), 1, ur);
Exp<T, false>(sig, sig, static_cast<uint64_t>(0), 1, ur);
Adds<T, false>(sig, sig, static_cast<T>(1), static_cast<uint64_t>(0), 1, ur);
Div<T, false>(sig, ones, sig, static_cast<uint64_t>(0), 1, br);
Mul<T, false>(out, g, sig, static_cast<uint64_t>(0), 1, br);
Mul<T, false>(out, out, u, static_cast<uint64_t>(0), 1, br);
}
template <class DT_INPUT_X, int Mode>
class KernelSwiGlu {
public:
__aicore__ inline KernelSwiGlu() {}
__aicore__ inline void Init(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2, GM_ADDR bias_2,
GM_ADDR output, const SwiGluTilingData &tiling)
{
td_ = &tiling;
const uint64_t xElems = td_->m * td_->d;
const uint64_t wElems = td_->d * td_->d;
xGm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)input_x, xElems);
w1Gm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)weight_1, wElems);
b1Gm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)bias_1, td_->d);
w2Gm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)weight_2, wElems);
b2Gm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)bias_2, td_->d);
outGm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)output, xElems);
pipe_.InitBuffer(stageA_, 1, AlignUpBytes(td_->dAlign * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(stageB_, 1, AlignUpBytes(td_->dAlign * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(outQueue_, 1, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(xfBuf_, AlignUpBytes(td_->dAlign * sizeof(float)));
pipe_.InitBuffer(wfBuf_, AlignUpBytes(td_->dAlign * sizeof(float)));
const uint32_t productElems = td_->dAlign *
(Mode != 0 && td_->dAlign <= 256 ? kChunk16 : 1U);
pipe_.InitBuffer(prodGBuf_, AlignUpBytes(productElems * sizeof(float)));
pipe_.InitBuffer(prodUBuf_, AlignUpBytes(productElems * sizeof(float)));
pipe_.InitBuffer(accGBuf_, AlignUpBytes(td_->tileN * sizeof(float)));
pipe_.InitBuffer(accUBuf_, AlignUpBytes(td_->tileN * sizeof(float)));
pipe_.InitBuffer(accG16Buf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(accU16Buf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(negBuf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(eBuf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(sBuf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(onesBuf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
if constexpr (Mode != 0) {
pipe_.InitBuffer(w1tBuf_, AlignUpBytes(td_->dAlign * kChunk16 * sizeof(float)));
pipe_.InitBuffer(w2tBuf_, AlignUpBytes(td_->dAlign * kChunk16 * sizeof(float)));
pipe_.InitBuffer(wblkQueue_, 1, AlignUpBytes(kChunk16 * td_->dAlign * sizeof(DT_INPUT_X)));
}
}
__aicore__ inline void Process()
{
const uint64_t core = GetBlockIdx();
if (core >= td_->usedCoreNum) {
return;
}
if constexpr (Mode == 2) {
const uint32_t d = static_cast<uint32_t>(td_->d);
const uint32_t columns = Ceiling(d, kChunk16);
const uint64_t tasks = td_->m * columns;
for (uint64_t task = core; task < tasks; task += td_->usedCoreNum) {
const uint64_t row = task / columns;
const uint32_t c0 = static_cast<uint32_t>(task % columns) * kChunk16;
const uint32_t chunk = d - c0 < kChunk16 ? d - c0 : kChunk16;
PrepareWt(c0, chunk, d);
ProcessColTileT(row, c0, chunk, d);
}
} else if constexpr (Mode == 1) {
const uint32_t d = static_cast<uint32_t>(td_->d);
for (uint32_t c0 = 0; c0 < d; c0 += kChunk16) {
const uint32_t chunk = d - c0 < kChunk16 ? d - c0 : kChunk16;
PrepareWt(c0, chunk, d);
for (uint64_t row = core; row < td_->m; row += td_->usedCoreNum) {
ProcessColTileT(row, c0, chunk, d);
}
}
} else {
for (uint64_t row = core; row < td_->m; row += td_->usedCoreNum) {
ProcessRow(row);
}
}
}
private:
template <HardEvent event>
__aicore__ inline void SyncPipe()
{
const event_t id = static_cast<event_t>(pipe_.FetchEventID(event));
SetFlag<event>(id);
WaitFlag<event>(id);
}
__aicore__ inline void ProcessRow(uint64_t row)
{
const uint32_t d = static_cast<uint32_t>(td_->d);
const uint32_t dAlign = td_->dAlign;
const uint32_t tileN = td_->tileN;
LocalTensor<DT_INPUT_X> xTile = stageA_.AllocTensor<DT_INPUT_X>();
DataCopyExtParams xCopy{1, static_cast<uint32_t>(d * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> xPad{true, 0, static_cast<uint8_t>(dAlign - d),
static_cast<DT_INPUT_X>(0)};
DataCopyPad(xTile, xGm_[row * d], xCopy, xPad);
stageA_.EnQue(xTile);
xTile = stageA_.DeQue<DT_INPUT_X>();
LocalTensor<float> xf = xfBuf_.Get<float>();
Cast(xf, xTile, RoundMode::CAST_NONE, dAlign);
stageA_.FreeTensor(xTile);
for (uint32_t c0 = 0; c0 < d; c0 += tileN) {
const uint32_t chunk = d - c0 < tileN ? d - c0 : tileN;
ProcessColTile(row, c0, chunk, d, dAlign, xf);
}
}
__aicore__ inline void ProcessColTile(uint64_t row, uint32_t c0, uint32_t chunk, uint32_t d, uint32_t dAlign,
const LocalTensor<float> &xf)
{
LocalTensor<float> wf = wfBuf_.Get<float>();
LocalTensor<float> prodG = prodGBuf_.Get<float>();
LocalTensor<float> prodU = prodUBuf_.Get<float>();
LocalTensor<float> accG = accGBuf_.Get<float>();
LocalTensor<float> accU = accUBuf_.Get<float>();
LocalTensor<DT_INPUT_X> accG16 = accG16Buf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> accU16 = accU16Buf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> negBuf = negBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> eBuf = eBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> sBuf = sBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> onesBuf = onesBuf_.Get<DT_INPUT_X>();
DataCopyExtParams wCopy{1, static_cast<uint32_t>(d * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> wPad{true, 0, static_cast<uint8_t>(dAlign - d),
static_cast<DT_INPUT_X>(0)};
for (uint32_t j = 0; j < chunk; ++j) {
const uint64_t wBase = static_cast<uint64_t>(c0 + j) * d;
LocalTensor<DT_INPUT_X> w1 = stageA_.AllocTensor<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> w2 = stageB_.AllocTensor<DT_INPUT_X>();
DataCopyPad(w1, w1Gm_[wBase], wCopy, wPad);
DataCopyPad(w2, w2Gm_[wBase], wCopy, wPad);
stageA_.EnQue(w1);
stageB_.EnQue(w2);
w1 = stageA_.DeQue<DT_INPUT_X>();
w2 = stageB_.DeQue<DT_INPUT_X>();
Cast(wf, w1, RoundMode::CAST_NONE, dAlign);
Mul(prodG, xf, wf, dAlign);
Cast(wf, w2, RoundMode::CAST_NONE, dAlign);
Mul(prodU, xf, wf, dAlign);
stageA_.FreeTensor(w1);
stageB_.FreeTensor(w2);
ReduceSum(accG[j], prodG, wf, dAlign);
ReduceSum(accU[j], prodU, wf, dAlign);
}
DataCopyExtParams bCopy{1, static_cast<uint32_t>(chunk * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> bPad{true, 0, static_cast<uint8_t>(AlignUp16(chunk) - chunk),
static_cast<DT_INPUT_X>(0)};
LocalTensor<DT_INPUT_X> b1Tile = stageA_.AllocTensor<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> b2Tile = stageB_.AllocTensor<DT_INPUT_X>();
DataCopyPad(b1Tile, b1Gm_[c0], bCopy, bPad);
DataCopyPad(b2Tile, b2Gm_[c0], bCopy, bPad);
stageA_.EnQue(b1Tile);
stageB_.EnQue(b2Tile);
b1Tile = stageA_.DeQue<DT_INPUT_X>();
b2Tile = stageB_.DeQue<DT_INPUT_X>();
Cast(accG16, accG, RoundMode::CAST_RINT, chunk);
Cast(accU16, accU, RoundMode::CAST_RINT, chunk);
Add(accG16, accG16, b1Tile, chunk);
stageA_.FreeTensor(b1Tile);
Add(accU16, accU16, b2Tile, chunk);
stageB_.FreeTensor(b2Tile);
Duplicate(onesBuf, static_cast<DT_INPUT_X>(1.0f), chunk);
Muls(negBuf, accG16, (DT_INPUT_X)(-1.0), chunk);
Exp(eBuf, negBuf, chunk);
Adds(eBuf, eBuf, (DT_INPUT_X)(1.0), chunk);
Div(sBuf, onesBuf, eBuf, chunk);
Mul(accG16, accG16, sBuf, chunk);
LocalTensor<DT_INPUT_X> outTile = outQueue_.AllocTensor<DT_INPUT_X>();
Mul(outTile, accG16, accU16, chunk);
outQueue_.EnQue(outTile);
outTile = outQueue_.DeQue<DT_INPUT_X>();
DataCopyExtParams oCopy{1, static_cast<uint32_t>(chunk * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPad(outGm_[row * d + c0], outTile, oCopy);
outQueue_.FreeTensor(outTile);
}
__aicore__ inline void PrepareOneWeight(const GlobalTensor<DT_INPUT_X> &weight,
const LocalTensor<float> &cached,
uint32_t c0, uint32_t chunk, uint32_t d)
{
LocalTensor<DT_INPUT_X> wblk = wblkQueue_.AllocTensor<DT_INPUT_X>();
DataCopyExtParams blkCopy{static_cast<uint16_t>(chunk),
static_cast<uint32_t>(d * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> blkPad{true, 0, static_cast<uint8_t>(td_->dAlign - d),
static_cast<DT_INPUT_X>(0)};
DataCopyPad(wblk, weight[static_cast<uint64_t>(c0) * d], blkCopy, blkPad);
wblkQueue_.EnQue(wblk);
wblk = wblkQueue_.DeQue<DT_INPUT_X>();
Cast(cached, wblk, RoundMode::CAST_NONE, chunk * td_->dAlign);
wblkQueue_.FreeTensor(wblk);
}
__aicore__ inline void PrepareWt(uint32_t c0, uint32_t chunk, uint32_t d)
{
PrepareOneWeight(w1Gm_, w1tBuf_.Get<float>(), c0, chunk, d);
PrepareOneWeight(w2Gm_, w2tBuf_.Get<float>(), c0, chunk, d);
}
__aicore__ inline void ProcessColTileT(uint64_t row, uint32_t c0, uint32_t chunk, uint32_t d)
{
const uint32_t dAlign = td_->dAlign;
LocalTensor<float> xf = xfBuf_.Get<float>();
LocalTensor<float> wf = wfBuf_.Get<float>();
LocalTensor<float> accG = accGBuf_.Get<float>();
LocalTensor<float> accU = accUBuf_.Get<float>();
LocalTensor<DT_INPUT_X> accG16 = accG16Buf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> accU16 = accU16Buf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> negBuf = negBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> eBuf = eBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> sBuf = sBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> onesBuf = onesBuf_.Get<DT_INPUT_X>();
LocalTensor<float> w1t = w1tBuf_.Get<float>();
LocalTensor<float> w2t = w2tBuf_.Get<float>();
LocalTensor<float> prodG = prodGBuf_.Get<float>();
LocalTensor<float> prodU = prodUBuf_.Get<float>();
LocalTensor<DT_INPUT_X> xTile = stageA_.AllocTensor<DT_INPUT_X>();
DataCopyExtParams xCopy{1, static_cast<uint32_t>(d * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> xPad{true, 0, static_cast<uint8_t>(dAlign - d),
static_cast<DT_INPUT_X>(0)};
DataCopyPad(xTile, xGm_[row * d], xCopy, xPad);
stageA_.EnQue(xTile);
xTile = stageA_.DeQue<DT_INPUT_X>();
Cast(xf, xTile, RoundMode::CAST_NONE, dAlign);
stageA_.FreeTensor(xTile);
LocalTensor<DT_INPUT_X> b1Tile = stageA_.AllocTensor<DT_INPUT_X>();
DataCopyExtParams bCopy{1, static_cast<uint32_t>(chunk * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> bPad{true, 0, static_cast<uint8_t>(AlignUp16(chunk) - chunk),
static_cast<DT_INPUT_X>(0)};
DataCopyPad(b1Tile, b1Gm_[c0], bCopy, bPad);
stageA_.EnQue(b1Tile);
b1Tile = stageA_.DeQue<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> b2Tile = stageB_.AllocTensor<DT_INPUT_X>();
DataCopyPad(b2Tile, b2Gm_[c0], bCopy, bPad);
stageB_.EnQue(b2Tile);
b2Tile = stageB_.DeQue<DT_INPUT_X>();
if (dAlign <= 256U) {
const uint8_t rowStride = static_cast<uint8_t>(dAlign / 8U);
const BinaryRepeatParams mulParams{1, 1, 1, rowStride, 0, rowStride};
for (uint32_t k = 0; k < dAlign; k += 64U) {
const uint32_t width = dAlign - k < 64U ? dAlign - k : 64U;
Mul(prodG[k], xf[k], w1t[k], static_cast<uint64_t>(width),
static_cast<uint8_t>(chunk), mulParams);
Mul(prodU[k], xf[k], w2t[k], static_cast<uint64_t>(width),
static_cast<uint8_t>(chunk), mulParams);
if (k == 0) {
WholeReduceSum(accG, prodG[k], width, chunk, 1, 1, rowStride);
WholeReduceSum(accU, prodU[k], width, chunk, 1, 1, rowStride);
} else {
WholeReduceSum(wf, prodG[k], width, chunk, 1, 1, rowStride);
Add(accG, accG, wf, chunk);
WholeReduceSum(wf, prodU[k], width, chunk, 1, 1, rowStride);
Add(accU, accU, wf, chunk);
}
}
} else {
for (uint32_t j = 0; j < chunk; ++j) {
Mul(prodG, xf, w1t[j * dAlign], dAlign);
Mul(prodU, xf, w2t[j * dAlign], dAlign);
ReduceSum(accG[j], prodG, wf, dAlign);
ReduceSum(accU[j], prodU, wf, dAlign);
}
}
Cast(accG16, accG, RoundMode::CAST_RINT, chunk);
Cast(accU16, accU, RoundMode::CAST_RINT, chunk);
Add(accG16, accG16, b1Tile, chunk);
stageA_.FreeTensor(b1Tile);
Add(accU16, accU16, b2Tile, chunk);
stageB_.FreeTensor(b2Tile);
Duplicate(onesBuf, static_cast<DT_INPUT_X>(1.0f), chunk);
Muls(negBuf, accG16, (DT_INPUT_X)(-1.0), chunk);
Exp(eBuf, negBuf, chunk);
Adds(eBuf, eBuf, (DT_INPUT_X)(1.0), chunk);
Div(sBuf, onesBuf, eBuf, chunk);
Mul(accG16, accG16, sBuf, chunk);
LocalTensor<DT_INPUT_X> outTile = outQueue_.AllocTensor<DT_INPUT_X>();
Mul(outTile, accG16, accU16, chunk);
outQueue_.EnQue(outTile);
outTile = outQueue_.DeQue<DT_INPUT_X>();
DataCopyExtParams oCopy{1, static_cast<uint32_t>(chunk * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPad(outGm_[row * d + c0], outTile, oCopy);
outQueue_.FreeTensor(outTile);
}
private:
TPipe pipe_;
TQue<TPosition::VECIN, 1> stageA_;
TQue<TPosition::VECIN, 1> stageB_;
TQue<TPosition::VECOUT, 1> outQueue_;
TBuf<TPosition::VECCALC> xfBuf_;
TBuf<TPosition::VECCALC> wfBuf_;
TBuf<TPosition::VECCALC> prodGBuf_;
TBuf<TPosition::VECCALC> prodUBuf_;
TBuf<TPosition::VECCALC> accGBuf_;
TBuf<TPosition::VECCALC> accUBuf_;
TBuf<TPosition::VECCALC> accG16Buf_;
TBuf<TPosition::VECCALC> accU16Buf_;
TBuf<TPosition::VECCALC> negBuf_;
TBuf<TPosition::VECCALC> eBuf_;
TBuf<TPosition::VECCALC> sBuf_;
TBuf<TPosition::VECCALC> onesBuf_;
TBuf<TPosition::VECCALC> w1tBuf_;
TBuf<TPosition::VECCALC> w2tBuf_;
TQue<QuePosition::VECIN, 1> wblkQueue_;
GlobalTensor<DT_INPUT_X> xGm_;
GlobalTensor<DT_INPUT_X> w1Gm_;
GlobalTensor<DT_INPUT_X> b1Gm_;
GlobalTensor<DT_INPUT_X> w2Gm_;
GlobalTensor<DT_INPUT_X> b2Gm_;
GlobalTensor<DT_INPUT_X> outGm_;
const SwiGluTilingData *td_ = nullptr;
};
template <typename T, uint32_t FixedD>
class TinyVectorSwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
GM_ADDR y, const SwiGluTinyTilingData *td) {
const uint32_t d = FixedD != 0 ? FixedD : static_cast<uint32_t>(td->d);
GlobalTensor<T> gx, gw1, gw2, gb1, gb2, gy;
gx.SetGlobalBuffer((__gm__ T*)x,td->m*d); gy.SetGlobalBuffer((__gm__ T*)y,td->m*d);
gw1.SetGlobalBuffer((__gm__ T*)w1,d*d); gw2.SetGlobalBuffer((__gm__ T*)w2,d*d);
gb1.SetGlobalBuffer((__gm__ T*)b1,d); gb2.SetGlobalBuffer((__gm__ T*)b2,d);
LocalTensor<T> hw1(TPosition::VECIN,0,256), hw2(TPosition::VECIN,512,256);
LocalTensor<T> hx(TPosition::VECIN,1024,16), hb1(TPosition::VECIN,1056,16), hb2(TPosition::VECIN,1088,16);
LocalTensor<float> fw1(TPosition::VECCALC,1152,256), fw2(TPosition::VECCALC,2176,256);
LocalTensor<float> fx(TPosition::VECCALC,3200,16), pg(TPosition::VECCALC,3264,256), pu(TPosition::VECCALC,4288,256);
LocalTensor<float> ag(TPosition::VECCALC,5312,16), au(TPosition::VECCALC,5376,16);
LocalTensor<T> g(TPosition::VECCALC,5440,16), u(TPosition::VECCALC,5472,16);
LocalTensor<T> sig(TPosition::VECCALC,5504,16), ones(TPosition::VECCALC,5536,16), out(TPosition::VECOUT,5568,16);
SetAtomicNone(); SetMaskNorm(); SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams wc{static_cast<uint16_t>(d),d*2U,0,0,0};
DataCopyExtParams rc{1,d*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(16U-d),static_cast<T>(0)};
DataCopyPad(hw1,gw1,wc,pad); DataCopyPad(hw2,gw2,wc,pad);
DataCopyPad(hb1,gb1,rc,pad); DataCopyPad(hb2,gb2,rc,pad);
Sync<HardEvent::MTE2_V>();
Cast(fw1,hw1,RoundMode::CAST_NONE,d*16U); Cast(fw2,hw2,RoundMode::CAST_NONE,d*16U);
Duplicate(ones,static_cast<T>(1),16);
const BinaryRepeatParams rp{1,1,1,2,0,2};
for (uint64_t row=GetBlockIdx();row<td->m;row+=GetBlockNum()) {
DataCopyPad(hx,gx[row*d],rc,pad); Sync<HardEvent::MTE2_V>();
Cast(fx,hx,RoundMode::CAST_NONE,16); PipeBarrier<PIPE_V>();
Mul(pg,fx,fw1,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
Mul(pu,fx,fw2,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
PipeBarrier<PIPE_V>();
WholeReduceSum(ag,pg,16,d,1,1,2); WholeReduceSum(au,pu,16,d,1,1,2);
PipeBarrier<PIPE_V>();
Cast(g,ag,RoundMode::CAST_RINT,d); Cast(u,au,RoundMode::CAST_RINT,d);
PipeBarrier<PIPE_V>(); Add(g,g,hb1,d); Add(u,u,hb2,d);
PipeBarrier<PIPE_V>(); GateCounter<T>(g,u,sig,out,ones,d);
Sync<HardEvent::V_MTE3>(); DataCopyPad(gy[row*d],out,rc);
Sync<HardEvent::MTE3_MTE2>();
}
}
private:
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
};
template <typename T, uint32_t FixedD, uint32_t Rows = 1, bool SingleBatch = false>
class BatchedTinyVectorSwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
GM_ADDR y, const SwiGluTinyTilingData *td) {
const uint32_t d=FixedD!=0?FixedD:static_cast<uint32_t>(td->d);
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gy.SetGlobalBuffer((__gm__ T*)y);
gw1.SetGlobalBuffer((__gm__ T*)w1,d*d);gw2.SetGlobalBuffer((__gm__ T*)w2,d*d);
gb1.SetGlobalBuffer((__gm__ T*)b1,d);gb2.SetGlobalBuffer((__gm__ T*)b2,d);
LocalTensor<T> hw1(TPosition::VECIN,0,256),hw2(TPosition::VECIN,512,256);
LocalTensor<T> hx(TPosition::VECIN,1024,Rows*16),hb1(TPosition::VECIN,1024+Rows*32,16),hb2(TPosition::VECIN,1056+Rows*32,16);
constexpr uint32_t fb=1088+Rows*32;
LocalTensor<float> fw1(TPosition::VECCALC,fb,256),fw2(TPosition::VECCALC,fb+1024,256);
LocalTensor<float> fx(TPosition::VECCALC,fb+2048,Rows*16);
constexpr uint32_t pb=fb+2048+Rows*64;
LocalTensor<float> pg(TPosition::VECCALC,pb,256),pu(TPosition::VECCALC,pb+1024,256);
constexpr uint32_t ab=pb+2048;
LocalTensor<float> ag(TPosition::VECCALC,ab,Rows*16),au(TPosition::VECCALC,ab+Rows*64,Rows*16);
constexpr uint32_t hb=ab+Rows*128;
LocalTensor<T> g(TPosition::VECCALC,hb,Rows*16),u(TPosition::VECCALC,hb+Rows*32,Rows*16);
LocalTensor<T> sig(TPosition::VECCALC,hb+Rows*64,Rows*16),ones(TPosition::VECCALC,hb+Rows*96,Rows*16),out(TPosition::VECOUT,hb+Rows*128,Rows*16);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams wc{static_cast<uint16_t>(d),d*2U,0,0,0},bc{1,d*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(16U-d),static_cast<T>(0)};
DataCopyPad(hw1,gw1,wc,pad);DataCopyPad(hw2,gw2,wc,pad);
DataCopyPad(hb1,gb1,bc,pad);DataCopyPad(hb2,gb2,bc,pad);
if constexpr (SingleBatch) {
DataCopyExtParams rc{static_cast<uint16_t>(Rows),d*2U,0,0,0};
DataCopyPad(hx,gx[static_cast<uint64_t>(GetBlockIdx())*Rows*d],rc,pad);
}
Sync<HardEvent::MTE2_V>();
Cast(fw1,hw1,RoundMode::CAST_NONE,d*16U);Cast(fw2,hw2,RoundMode::CAST_NONE,d*16U);
Duplicate(ones,static_cast<T>(1),Rows*16U);
Duplicate(ag,0.0f,Rows*16U);Duplicate(au,0.0f,Rows*16U);
const BinaryRepeatParams rp{1,1,1,2,0,2};
const BinaryRepeatParams bp{1,1,1,1,1,0};
const uint64_t first=static_cast<uint64_t>(GetBlockIdx())*Rows;
const uint64_t stride=SingleBatch?0U:static_cast<uint64_t>(GetBlockNum())*Rows;
uint64_t groups=1U;
if constexpr (!SingleBatch) { groups=(td->m-first+stride-1U)/stride; }
for (uint64_t group=0;group<groups;++group) {
const uint64_t row=first+group*stride;
const uint32_t nr=SingleBatch?Rows:(td->m-row<Rows?static_cast<uint32_t>(td->m-row):Rows);
DataCopyExtParams rc{static_cast<uint16_t>(nr),d*2U,0,0,0};
if constexpr (!SingleBatch) { DataCopyPad(hx,gx[row*d],rc,pad);Sync<HardEvent::MTE2_V>(); }
Cast(fx,hx,RoundMode::CAST_NONE,nr*16U);PipeBarrier<PIPE_V>();
#pragma unroll
for (uint32_t i=0;i<nr;++i) {
Mul(pg,fx[i*16U],fw1,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
Mul(pu,fx[i*16U],fw2,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
PipeBarrier<PIPE_V>();
WholeReduceSum(ag[i*16U],pg,16,d,1,1,2);WholeReduceSum(au[i*16U],pu,16,d,1,1,2);
PipeBarrier<PIPE_V>();
}
Cast(g,ag,RoundMode::CAST_RINT,nr*16U);Cast(u,au,RoundMode::CAST_RINT,nr*16U);
PipeBarrier<PIPE_V>();
Add(g,g,hb1,static_cast<uint64_t>(d),static_cast<uint8_t>(nr),bp);
Add(u,u,hb2,static_cast<uint64_t>(d),static_cast<uint8_t>(nr),bp);
PipeBarrier<PIPE_V>();GateCounter<T>(g,u,sig,out,ones,nr*16U);
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[row*d],out,rc);
if constexpr (!SingleBatch) { Sync<HardEvent::MTE3_MTE2>(); }
}
if constexpr (SingleBatch) { PipeBarrier<PIPE_ALL>(); }
}
private:
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0); }
};
template <typename T>
class ScalarFeatureSwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
GM_ADDR y, const SwiGluTinyTilingData *td) {
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x,td->m);gy.SetGlobalBuffer((__gm__ T*)y,td->m);
gw1.SetGlobalBuffer((__gm__ T*)w1,1);gw2.SetGlobalBuffer((__gm__ T*)w2,1);
gb1.SetGlobalBuffer((__gm__ T*)b1,1);gb2.SetGlobalBuffer((__gm__ T*)b2,1);
LocalTensor<T> hx(TPosition::VECIN,0,256), params(TPosition::VECIN,512,64);
LocalTensor<float> fx(TPosition::VECCALC,640,256), fg(TPosition::VECCALC,1664,256), fu(TPosition::VECCALC,2688,256);
LocalTensor<T> g(TPosition::VECCALC,3712,256), u(TPosition::VECCALC,4224,256);
LocalTensor<T> sig(TPosition::VECCALC,4736,256), ones(TPosition::VECCALC,5248,256), out(TPosition::VECOUT,5760,256);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams pc{1,2,0,0,0};
DataCopyPadExtParams<T> pp{true,0,15,static_cast<T>(0)};
DataCopyPad(params,gw1,pc,pp);DataCopyPad(params[16],gw2,pc,pp);
DataCopyPad(params[32],gb1,pc,pp);DataCopyPad(params[48],gb2,pc,pp);
Sync<HardEvent::MTE2_S>();
const float a=static_cast<float>(params.GetValue(0)),b=static_cast<float>(params.GetValue(16));
const T c=params.GetValue(32),e=params.GetValue(48);
Duplicate(ones,static_cast<T>(1),256);
for (uint64_t start=static_cast<uint64_t>(GetBlockIdx())*256U;start<td->m;start+=static_cast<uint64_t>(GetBlockNum())*256U) {
const uint32_t n=td->m-start<256U?static_cast<uint32_t>(td->m-start):256U;
DataCopyExtParams cp{1,n*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>((16U-n%16U)%16U),static_cast<T>(0)};
DataCopyPad(hx,gx[start],cp,pad);Sync<HardEvent::MTE2_V>();
Cast(fx,hx,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();
Muls(fg,fx,a,n);Muls(fu,fx,b,n);PipeBarrier<PIPE_V>();
Cast(g,fg,RoundMode::CAST_RINT,n);Cast(u,fu,RoundMode::CAST_RINT,n);
PipeBarrier<PIPE_V>();Adds(g,g,c,n);Adds(u,u,e,n);
PipeBarrier<PIPE_V>(); GateCounter<T>(g,u,sig,out,ones,n);
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[start],out,cp);
Sync<HardEvent::MTE3_MTE2>();
}
}
private:
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
};
template <typename T, uint32_t FixedD, uint32_t DynamicMaxK = 256U>
class SmallMVectorSwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
GM_ADDR y, const SwiGluTinyTilingData *td) {
constexpr uint32_t maxK=FixedD==0?DynamicMaxK:FixedD;
constexpr uint32_t wElems=16U*maxK;
constexpr uint32_t fBase=4U*wElems+2U*maxK+64U;
const uint32_t d=FixedD!=0?FixedD:static_cast<uint32_t>(td->d);
const uint32_t ka=(d+15U)/16U*16U;
const uint32_t columns=(d+15U)/16U;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x,td->m*d);gy.SetGlobalBuffer((__gm__ T*)y,td->m*d);
gw1.SetGlobalBuffer((__gm__ T*)w1,d*d);gw2.SetGlobalBuffer((__gm__ T*)w2,d*d);
gb1.SetGlobalBuffer((__gm__ T*)b1,d);gb2.SetGlobalBuffer((__gm__ T*)b2,d);
LocalTensor<T> hw1(TPosition::VECIN,0,wElems),hw2(TPosition::VECIN,2U*wElems,wElems);
LocalTensor<T> hx(TPosition::VECIN,4U*wElems,maxK);
LocalTensor<T> hb1(TPosition::VECIN,4U*wElems+2U*maxK,16),hb2(TPosition::VECIN,4U*wElems+2U*maxK+32U,16);
LocalTensor<float> fw1(TPosition::VECCALC,fBase,wElems),fw2(TPosition::VECCALC,fBase+4U*wElems,wElems);
LocalTensor<float> pg(TPosition::VECCALC,fBase+8U*wElems,wElems),pu(TPosition::VECCALC,fBase+12U*wElems,wElems);
constexpr uint32_t xBase=fBase+16U*wElems;
LocalTensor<float> fx(TPosition::VECCALC,xBase,maxK);
constexpr uint32_t aBase=xBase+4U*maxK;
LocalTensor<float> ag(TPosition::VECCALC,aBase,16),au(TPosition::VECCALC,aBase+64U,16);
LocalTensor<float> sg(TPosition::VECCALC,aBase+128U,16),su(TPosition::VECCALC,aBase+192U,16);
constexpr uint32_t hBase=aBase+256U;
LocalTensor<T> g(TPosition::VECCALC,hBase,16),u(TPosition::VECCALC,hBase+32U,16);
LocalTensor<T> sig(TPosition::VECCALC,hBase+64U,16),ones(TPosition::VECCALC,hBase+96U,16),out(TPosition::VECOUT,hBase+128U,16);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
Duplicate(ones,static_cast<T>(1),16);
for (uint64_t task=GetBlockIdx();task<td->m*columns;task+=GetBlockNum()) {
const uint64_t row=task/columns;
const uint32_t col=task%columns*16U;
const uint32_t cols=d-col<16U?d-col:16U;
DataCopyExtParams wc{static_cast<uint16_t>(cols),d*2U,0,0,0};
DataCopyExtParams xc{1,d*2U,0,0,0},bc{1,cols*2U,0,0,0};
DataCopyPadExtParams<T> wp{true,0,static_cast<uint8_t>(ka-d),static_cast<T>(0)};
DataCopyPadExtParams<T> bp{true,0,static_cast<uint8_t>(16U-cols),static_cast<T>(0)};
DataCopyPad(hw1,gw1[static_cast<uint64_t>(col)*d],wc,wp);
DataCopyPad(hw2,gw2[static_cast<uint64_t>(col)*d],wc,wp);
DataCopyPad(hx,gx[row*d],xc,wp);
DataCopyPad(hb1,gb1[col],bc,bp);DataCopyPad(hb2,gb2[col],bc,bp);
Sync<HardEvent::MTE2_V>();
Cast(fw1,hw1,RoundMode::CAST_NONE,cols*ka);Cast(fw2,hw2,RoundMode::CAST_NONE,cols*ka);
Cast(fx,hx,RoundMode::CAST_NONE,ka);PipeBarrier<PIPE_V>();
Dot<true>(ag,au,sg,su,pg,pu,fx,fw1,fw2,ka,0,cols);
for (uint32_t k=64U;k<ka;k+=64U) { Dot<false>(ag,au,sg,su,pg,pu,fx,fw1,fw2,ka,k,cols); }
Cast(g,ag,RoundMode::CAST_RINT,cols);Cast(u,au,RoundMode::CAST_RINT,cols);
PipeBarrier<PIPE_V>();Add(g,g,hb1,cols);Add(u,u,hb2,cols);
PipeBarrier<PIPE_V>(); GateCounter<T>(g,u,sig,out,ones,cols);
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[row*d+col],out,bc);
Sync<HardEvent::MTE3_MTE2>();
}
}
private:
template <bool First> __aicore__ inline void Dot(
const LocalTensor<float>& ag,const LocalTensor<float>& au,const LocalTensor<float>& sg,const LocalTensor<float>& su,
const LocalTensor<float>& pg,const LocalTensor<float>& pu,const LocalTensor<float>& fx,
const LocalTensor<float>& fw1,const LocalTensor<float>& fw2,uint32_t ka,uint32_t k,uint32_t cols) {
const uint32_t width=ka-k<64U?ka-k:64U;
const uint8_t stride=static_cast<uint8_t>(ka/8U);
const BinaryRepeatParams rp{1,1,1,stride,0,stride};
Mul(pg[k],fx[k],fw1[k],static_cast<uint64_t>(width),static_cast<uint8_t>(cols),rp);
Mul(pu[k],fx[k],fw2[k],static_cast<uint64_t>(width),static_cast<uint8_t>(cols),rp);
PipeBarrier<PIPE_V>();
if constexpr (First) {
WholeReduceSum(ag,pg[k],width,cols,1,1,stride);WholeReduceSum(au,pu[k],width,cols,1,1,stride);
PipeBarrier<PIPE_V>();
} else {
WholeReduceSum(sg,pg[k],width,cols,1,1,stride);WholeReduceSum(su,pu[k],width,cols,1,1,stride);
PipeBarrier<PIPE_V>();Add(ag,ag,sg,cols);Add(au,au,su,cols);PipeBarrier<PIPE_V>();
}
}
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0); }
};
template <typename T, uint32_t FixedD>
class SingleTinySwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y,const SwiGluTinyTilingData *td) {
const uint32_t d=FixedD!=0?FixedD:static_cast<uint32_t>(td->d);
const uint64_t row=GetBlockIdx();
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gy.SetGlobalBuffer((__gm__ T*)y);
gw1.SetGlobalBuffer((__gm__ T*)w1,d*d);gw2.SetGlobalBuffer((__gm__ T*)w2,d*d);
gb1.SetGlobalBuffer((__gm__ T*)b1,d);gb2.SetGlobalBuffer((__gm__ T*)b2,d);
LocalTensor<T> hw1(TPosition::VECIN,0,256),hw2(TPosition::VECIN,512,256);
LocalTensor<T> hx(TPosition::VECIN,1024,16),hb1(TPosition::VECIN,1056,16),hb2(TPosition::VECIN,1088,16);
LocalTensor<float> fw1(TPosition::VECCALC,1152,256),fw2(TPosition::VECCALC,2176,256);
LocalTensor<float> fx(TPosition::VECCALC,3200,16),pg(TPosition::VECCALC,3264,256),pu(TPosition::VECCALC,4288,256);
LocalTensor<float> ag(TPosition::VECCALC,5312,16),au(TPosition::VECCALC,5376,16);
LocalTensor<T> g(TPosition::VECCALC,5440,16),u(TPosition::VECCALC,5472,16);
LocalTensor<T> sig(TPosition::VECCALC,5504,16),ones(TPosition::VECCALC,5536,16),out(TPosition::VECOUT,5568,16);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams wc{static_cast<uint16_t>(d),d*2U,0,0,0},rc{1,d*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(16U-d),static_cast<T>(0)};
DataCopyPad(hw1,gw1,wc,pad);DataCopyPad(hw2,gw2,wc,pad);
DataCopyPad(hb1,gb1,rc,pad);DataCopyPad(hb2,gb2,rc,pad);DataCopyPad(hx,gx[row*d],rc,pad);
Sync<HardEvent::MTE2_V>();
Duplicate(ones,static_cast<T>(1),16);
if constexpr (FixedD==1) {
Mul(g,hx,hw1,1);Mul(u,hx,hw2,1);
} else {
Cast(fw1,hw1,RoundMode::CAST_NONE,d*16U);Cast(fw2,hw2,RoundMode::CAST_NONE,d*16U);
Cast(fx,hx,RoundMode::CAST_NONE,16);
const BinaryRepeatParams rp{1,1,1,2,0,2};
Mul(pg,fx,fw1,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
Mul(pu,fx,fw2,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
WholeReduceSum(ag,pg,16,d,1,1,2);WholeReduceSum(au,pu,16,d,1,1,2);
Cast(g,ag,RoundMode::CAST_RINT,d);Cast(u,au,RoundMode::CAST_RINT,d);
}
Add(g,g,hb1,d);Add(u,u,hb2,d);
if constexpr (FixedD != 0) {
FastGate<T, FixedD>(g, u, sig, out, ones);
} else {
GateCounter<T>(g, u, sig, out, ones, d);
}
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[row*d],out,rc);
}
private:
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0); }
};
template <typename T,uint32_t FixedD,bool OneRow>
class TinyColumnsSwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y,const SwiGluTinyTilingData* td) {
const uint32_t d=FixedD!=0?FixedD:(OneRow?GetBlockNum():static_cast<uint32_t>(td->d));
const uint32_t task=GetBlockIdx();
const uint64_t row=OneRow?0U:task/d;
const uint32_t col=OneRow?task:task%d;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gw1.SetGlobalBuffer((__gm__ T*)w1);gw2.SetGlobalBuffer((__gm__ T*)w2);
gb1.SetGlobalBuffer((__gm__ T*)b1);gb2.SetGlobalBuffer((__gm__ T*)b2);gy.SetGlobalBuffer((__gm__ T*)y);
LocalTensor<T> hx(TPosition::VECIN,0,16),hw1(TPosition::VECIN,32,16),hw2(TPosition::VECIN,64,16);
LocalTensor<T> hb1(TPosition::VECIN,96,16),hb2(TPosition::VECIN,128,16);
LocalTensor<float> fx(TPosition::VECCALC,192,16),fw1(TPosition::VECCALC,256,16),fw2(TPosition::VECCALC,320,16);
LocalTensor<float> pg(TPosition::VECCALC,384,16),pu(TPosition::VECCALC,448,16);
LocalTensor<float> ag(TPosition::VECCALC,512,8),au(TPosition::VECCALC,544,8);
LocalTensor<T> g(TPosition::VECCALC,576,16),u(TPosition::VECCALC,608,16);
LocalTensor<T> sig(TPosition::VECCALC,640,16),ones(TPosition::VECCALC,672,16),out(TPosition::VECOUT,704,16);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams cp{1,d*2U,0,0,0},bp{1,2,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(16U-d),static_cast<T>(0)};
DataCopyPadExtParams<T> one{true,0,15,static_cast<T>(0)};
DataCopyPad(hx,gx[row*d],cp,pad);DataCopyPad(hw1,gw1[col*d],cp,pad);DataCopyPad(hw2,gw2[col*d],cp,pad);
DataCopyPad(hb1,gb1[col],bp,one);DataCopyPad(hb2,gb2[col],bp,one);
Sync<HardEvent::MTE2_V>();
Cast(fx,hx,RoundMode::CAST_NONE,16);Cast(fw1,hw1,RoundMode::CAST_NONE,16);Cast(fw2,hw2,RoundMode::CAST_NONE,16);
Duplicate(ones,static_cast<T>(1),16);PipeBarrier<PIPE_V>();
Mul(pg,fx,fw1,16);Mul(pu,fx,fw2,16);PipeBarrier<PIPE_V>();
WholeReduceSum(ag,pg,16,1,1,1,2);WholeReduceSum(au,pu,16,1,1,1,2);PipeBarrier<PIPE_V>();
Cast(g,ag,RoundMode::CAST_RINT,1);Cast(u,au,RoundMode::CAST_RINT,1);PipeBarrier<PIPE_V>();
Add(g,g,hb1,1);Add(u,u,hb2,1);PipeBarrier<PIPE_V>();
GateCounter<T>(g,u,sig,out,ones,1);Sync<HardEvent::V_MTE3>();
DataCopyPad(gy[row*d+col],out,bp);PipeBarrier<PIPE_ALL>();
}
private:
template <HardEvent E> __aicore__ inline void Sync() {SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0);}
};
template <typename T>
class SubBlockGemv4 {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y) {
if (GetBlockIdx() != 0U) { return; }
GlobalTensor<T> gx, gw1, gw2, gb1, gb2, gy;
gx.SetGlobalBuffer((__gm__ T*)x, 4);
gw1.SetGlobalBuffer((__gm__ T*)w1, 16);
gw2.SetGlobalBuffer((__gm__ T*)w2, 16);
gb1.SetGlobalBuffer((__gm__ T*)b1, 4);
gb2.SetGlobalBuffer((__gm__ T*)b2, 4);
gy.SetGlobalBuffer((__gm__ T*)y, 4);
LocalTensor<T> hx(TPosition::VECIN, 0, 16);
LocalTensor<T> w(TPosition::VECIN, 32, 32);
LocalTensor<T> bias(TPosition::VECIN, 96, 32);
LocalTensor<float> prod(TPosition::VECCALC, 352, 32);
LocalTensor<float> pairs(TPosition::VECCALC, 480, 16);
LocalTensor<float> acc(TPosition::VECCALC, 544, 40);
LocalTensor<T> g(TPosition::VECCALC, 704, 32);
LocalTensor<T> sig(TPosition::VECCALC, 768, 16);
LocalTensor<T> ones(TPosition::VECCALC, 800, 16);
LocalTensor<T> out(TPosition::VECOUT, 832, 16);
SetAtomicNone();
SetMaskNorm();
const DataCopyExtParams xc{1, 8, 0, 0, 0};
const DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
GlobalTensor<uint64_t> xword;
xword.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(x), 1);
DataCachePreload(xword, static_cast<int16_t>(0));
DataCopy(w, gw1, 16);
DataCopy(w[16], gw2, 16);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
DataCopyPad(bias, gb1, xc, pad);
DataCopyPad(bias[16], gb2, xc, pad);
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
Duplicate(prod, 0.0f, 32);
const uint64_t xv = xword.GetValue(0);
auto xx = hx.template ReinterpretCast<uint64_t>();
xx.SetValue(0, xv);
xx.SetValue(1, xv);
xx.SetValue(2, xv);
xx.SetValue(3, xv);
SetFlag<HardEvent::S_V>(EVENT_ID0);
WaitFlag<HardEvent::S_V>(EVENT_ID0);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
MulAddDst<float, T>(prod, hx, w, static_cast<uint64_t>(16), 2, BinaryRepeatParams{1, 1, 1, 2, 0, 1});
PairReduceSum(pairs, prod, 1, 32, 1, 1, 8);
PairReduceSum(acc, pairs, 2, 8, 1, 1, 1);
constexpr uint64_t mask = 0;
SetVectorMask<float, MaskMode::NORMAL>(0, 15);
Cast<T, float, false>(g, acc, RoundMode::CAST_RINT, mask, 2, UnaryRepeatParams{1, 1, 1, 4});
SetVectorMask<T, MaskMode::NORMAL>(0, 15);
Duplicate<T, false>(ones, static_cast<T>(1), mask, 1, 1, 8);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
Add<T, false>(g, g, bias, mask, 2, BinaryRepeatParams{1, 1, 1, 1, 1, 1});
FastGate<T, 4U, false>(g, g[16], sig, out, ones);
Sync<HardEvent::V_MTE3>();
DataCopyPad(gy, out, xc);
}
private:
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
};
template <typename T>
class ScalarChannelSwiGlu {
public:
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y) {
        if (GetBlockIdx() != 0U) { return; }
        GlobalTensor<T> gx, gw1, gw2, gb1, gb2, gy;
        gx.SetGlobalBuffer((__gm__ T*)x, 1); gy.SetGlobalBuffer((__gm__ T*)y, 1);
        gw1.SetGlobalBuffer((__gm__ T*)w1, 1); gw2.SetGlobalBuffer((__gm__ T*)w2, 1);
        gb1.SetGlobalBuffer((__gm__ T*)b1, 1); gb2.SetGlobalBuffer((__gm__ T*)b2, 1);

        LocalTensor<T> hx(TPosition::VECIN, 0, 16);
        LocalTensor<T> w(TPosition::VECIN, 32, 32);
        LocalTensor<T> bias(TPosition::VECIN, 96, 32);

        LocalTensor<T> g(TPosition::VECCALC, 160, 32);
        LocalTensor<T> sig(TPosition::VECCALC, 224, 16);
        LocalTensor<T> ones(TPosition::VECCALC, 256, 16);
        LocalTensor<T> out(TPosition::VECOUT, 288, 16);

        SetAtomicNone();
        SetMaskNorm();
        SetVectorMask<T, MaskMode::NORMAL>(0, 1);

        const DataCopyExtParams rc{1, 2, 0, 0, 0};
        const DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        const T xv=gx.GetValue(0);
const T wv1=gw1.GetValue(0),wv2=gw2.GetValue(0);
const T bv1=gb1.GetValue(0),bv2=gb2.GetValue(0);
hx.SetValue(0,xv);w.SetValue(0,wv1);w.SetValue(16,wv2);
bias.SetValue(0,bv1);bias.SetValue(16,bv2);
SetFlag<HardEvent::S_V>(EVENT_ID0);
WaitFlag<HardEvent::S_V>(EVENT_ID0);
constexpr uint64_t mask = 0;
        
        Mul<T, false>(g, hx, w, mask, 2, BinaryRepeatParams{1, 1, 1, 1, 0, 1});
        
        Add<T, false>(g, g, bias, mask, 2, BinaryRepeatParams{1, 1, 1, 1, 1, 1});

        FastGate<T, 1U, false>(g, g[16], sig, out, ones);

        Sync<HardEvent::V_MTE3>();
        DataCopyPad(gy, out, rc);
    }
private:
    template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
};
template <typename T>
class AlignedRowGemv32 {
public:
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y) {
        if (GetBlockIdx() != 0U) { return; }
        GlobalTensor<T> gx, gw1, gw2, gb1, gb2, gy;
        gx.SetGlobalBuffer((__gm__ T*)x, 32);
        gy.SetGlobalBuffer((__gm__ T*)y, 32);
        gw1.SetGlobalBuffer((__gm__ T*)w1, 1024);
        gw2.SetGlobalBuffer((__gm__ T*)w2, 1024);
        gb1.SetGlobalBuffer((__gm__ T*)b1, 32);
        gb2.SetGlobalBuffer((__gm__ T*)b2, 32);

        LocalTensor<T> hw1(TPosition::VECIN, 0, 1024);
        LocalTensor<T> hw2(TPosition::VECIN, 2048, 1024);
        LocalTensor<T> hx(TPosition::VECIN, 4096, 64);
        LocalTensor<T> hb(TPosition::VECIN, 4224, 64);

        LocalTensor<float> pg(TPosition::VECCALC, 4352, 2048);
        LocalTensor<float> pu(TPosition::VECCALC, 12544, 256);
        LocalTensor<float> ag(TPosition::VECCALC, 13568, 64);

        LocalTensor<T> gu(TPosition::VECCALC, 13824, 64);
        LocalTensor<T> sig(TPosition::VECCALC, 13952, 32);
        LocalTensor<T> ones(TPosition::VECCALC, 14016, 32);
        LocalTensor<T> out(TPosition::VECOUT, 14080, 32);

        SetAtomicNone();
        SetMaskNorm();
        constexpr uint64_t mask = 0;
        SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);

        DataCopy(hx, gx, 32);
        DataCopy(hx[32], gx, 32);
        DataCopy(hb, gb1, 32);
        DataCopy(hb[32], gb2, 32);
        DataCopy(hw1, gw1, 1024);
        
        DataCopy(hw2, gw2, 1024);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);

        Duplicate<float, false>(pg, 0.0f, mask, 32U, 1, 8);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
MulAddDst<float, T, false>(pg, hx, hw1, mask, 32U, BinaryRepeatParams{1, 1, 1, 8, 0, 4});
BlockReduceSum<float, false>(pu, pg, 32U, mask, 1, 1, 8);
PairReduceSum<float, false>(pg, pu, 4U, mask, 1, 1, 8);
        PairReduceSum<float, false>(ag, pg, 2U, mask, 1, 1, 8);

        Cast<T, float, false>(gu, ag, RoundMode::CAST_RINT, mask, 1, UnaryRepeatParams{1, 1, 4, 8});
        SetVectorMask<T, MaskMode::NORMAL>(0, ~0ULL);
        Add<T, false>(gu, gu, hb, mask, 1, BinaryRepeatParams{1, 1, 1, 8, 8, 8});

        FastGate<T, 32U, false>(gu, gu[32], sig, out, ones);

        Sync<HardEvent::V_MTE3>();
        DataCopy(gy, out, 32U);
    }
private:
    template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
};
template <typename T, uint32_t FixedD, uint32_t FixedM>
class AlignedColGemv {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
GM_ADDR y) {
const uint64_t task=GetBlockIdx();
if constexpr (FixedM==1U) {
if (task>=2U) { return; }
}
constexpr uint32_t maxK=FixedD;
constexpr uint32_t wElems=16U*maxK;
constexpr uint32_t fBase=4U*wElems+2U*maxK+128U;
const uint32_t d=FixedD;
const uint32_t ka=(d+15U)/16U*16U;
const uint32_t columns=(d+15U)/16U;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x,FixedM*d);gy.SetGlobalBuffer((__gm__ T*)y,FixedM*d);
gw1.SetGlobalBuffer((__gm__ T*)w1,d*d);gw2.SetGlobalBuffer((__gm__ T*)w2,d*d);
gb1.SetGlobalBuffer((__gm__ T*)b1,d);gb2.SetGlobalBuffer((__gm__ T*)b2,d);
LocalTensor<T> hw1(TPosition::VECIN,0,2U*wElems),hw2(TPosition::VECIN,2U*wElems,wElems);
LocalTensor<T> hx(TPosition::VECIN,4U*wElems,maxK);
LocalTensor<T> hb1(TPosition::VECIN,4U*wElems+2U*maxK,32),hb2(TPosition::VECIN,4U*wElems+32U+2U*maxK,16);
LocalTensor<float> pg(TPosition::VECCALC,fBase+8U*wElems,2U*wElems),pu(TPosition::VECCALC,fBase+16U*wElems,wElems);
constexpr uint32_t xBase=fBase+20U*wElems;
constexpr uint32_t aBase=xBase+4U*maxK;
LocalTensor<float> ag(TPosition::VECCALC,aBase,32);
constexpr uint32_t hBase=aBase+256U;
LocalTensor<T> g(TPosition::VECCALC,hBase,32),u(TPosition::VECCALC,hBase+32U,16);
LocalTensor<T> sig(TPosition::VECCALC,hBase+64U,16),ones(TPosition::VECCALC,hBase+96U,16),out(TPosition::VECOUT,hBase+128U,16);
SetAtomicNone();SetMaskNorm();
constexpr uint64_t mask=0;
SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
{
const uint64_t row=FixedM==1U?0U:task/columns;
const uint32_t col=(FixedM==1U?task:task%columns)*16U;
static_assert(FixedD==32U || FixedD==64U);
if constexpr (FixedD==64U) {
DataCopy(hw1,gw1[static_cast<uint64_t>(col)*d],16U*FixedD);
DataCopy(hw2,gw2[static_cast<uint64_t>(col)*d],16U*FixedD);
DataCopy(hx,gx[row*d],FixedD);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
DataCopy(hb1,gb1[col],16U);DataCopy(hb2,gb2[col],16U);
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
Duplicate<T,false>(ones,static_cast<T>(1),mask,1,1,8);
Duplicate<float,false>(pg,0.0f,mask,32U,1,8);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
MulAddDst<float,T,false>(pg,hx,hw1,mask,32U,BinaryRepeatParams{1,1,1,8,0,4});
BlockReduceSum<float,false>(pu,pg,32U,mask,1,1,8);
BlockReduceSum<float,false>(ag,pu,4U,mask,1,1,8);
} else {
DataCopy(hw1,gw1[static_cast<uint64_t>(col)*d],16U*FixedD);
DataCopy(hw2,gw2[static_cast<uint64_t>(col)*d],16U*FixedD);
DataCopy(hx,gx[row*d],FixedD);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
DataCopy(hb1,gb1[col],16U);DataCopy(hb2,gb2[col],16U);
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
Duplicate<T,false>(ones,static_cast<T>(1),mask,1,1,8);
Duplicate<float,false>(pg,0.0f,mask,16U,1,8);
SetVectorMask<float,MaskMode::NORMAL>(0,0xFFFFFFFFULL);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
MulAddDst<float,T,false>(pg,hx,hw1,mask,32U,BinaryRepeatParams{1,1,1,4,0,2});
SetVectorMask<float,MaskMode::NORMAL>(0,~0ULL);
BlockReduceSum<float,false>(pu,pg,16U,mask,1,1,8);
PairReduceSum<float,false>(pu[128],pu,2U,mask,1,1,8);
PairReduceSum<float,false>(ag,pu[128],1U,mask,1,1,8);
}
SetVectorMask<float,MaskMode::NORMAL>(0,0xFFFFFFFFULL);
Cast<T,float,false>(g,ag,RoundMode::CAST_RINT,mask,1,UnaryRepeatParams{1,1,4,8});
WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
Add<T,false>(g,g,hb1,mask,1,BinaryRepeatParams{1,1,1,2,2,2});
FastGate<T,16U>(g,g[16],sig,out,ones);
Sync<HardEvent::V_MTE3>();
if constexpr (FixedM==1U) {
DataCopy(gy[col],out,16U);
} else {
DataCopy(gy[row*d+col],out,16U);
}
}
}
private:
template <bool First> __aicore__ inline void Dot(
const LocalTensor<float>& ag,const LocalTensor<float>& au,const LocalTensor<float>& sg,const LocalTensor<float>& su,
const LocalTensor<float>& pg,const LocalTensor<float>& pu,const LocalTensor<float>& fx,
const LocalTensor<float>& fw1,const LocalTensor<float>& fw2,uint32_t ka,uint32_t k,uint32_t cols) {
const uint32_t width=ka-k<64U?ka-k:64U;
const uint8_t stride=static_cast<uint8_t>(ka/8U);
const BinaryRepeatParams rp{1,1,1,stride,0,stride};
Mul(pg[k],fx[k],fw1[k],static_cast<uint64_t>(width),static_cast<uint8_t>(cols),rp);
Mul(pu[k],fx[k],fw2[k],static_cast<uint64_t>(width),static_cast<uint8_t>(cols),rp);
PipeBarrier<PIPE_V>();
if constexpr (First) {
WholeReduceSum(ag,pg[k],width,cols,1,1,stride);WholeReduceSum(au,pu[k],width,cols,1,1,stride);
PipeBarrier<PIPE_V>();
} else {
WholeReduceSum(sg,pg[k],width,cols,1,1,stride);WholeReduceSum(su,pu[k],width,cols,1,1,stride);
PipeBarrier<PIPE_V>();Add(ag,ag,sg,cols);Add(au,au,su,cols);PipeBarrier<PIPE_V>();
}
}
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0); }
};
template <typename T,uint32_t D,uint32_t Rows>
class BatchedRowGemv {
static constexpr uint32_t K=(D+15U)/16U*16U, W=D*K;
static constexpr uint32_t HB=4U*W+2U*Rows*K;
static constexpr uint32_t FB=HB+4U*K;
static constexpr uint32_t XB=FB+8U*W, PB=XB+4U*Rows*K;
static constexpr uint32_t AB=PB+8U*W, GB=AB+8U*Rows*K;
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y) {
const uint64_t row=static_cast<uint64_t>(GetBlockIdx())*Rows;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gw1.SetGlobalBuffer((__gm__ T*)w1);gw2.SetGlobalBuffer((__gm__ T*)w2);
gb1.SetGlobalBuffer((__gm__ T*)b1);gb2.SetGlobalBuffer((__gm__ T*)b2);gy.SetGlobalBuffer((__gm__ T*)y);
LocalTensor<T> hw1(TPosition::VECIN,0,W),hw2(TPosition::VECIN,2U*W,W),hx(TPosition::VECIN,4U*W,Rows*K);
LocalTensor<T> hb1(TPosition::VECIN,HB,K),hb2(TPosition::VECIN,HB+2U*K,K);
LocalTensor<float> fw1(TPosition::VECCALC,FB,W),fw2(TPosition::VECCALC,FB+4U*W,W);
LocalTensor<float> fx(TPosition::VECCALC,XB,Rows*K),pg(TPosition::VECCALC,PB,W),pu(TPosition::VECCALC,PB+4U*W,W);
LocalTensor<float> ag(TPosition::VECCALC,AB,Rows*K),au(TPosition::VECCALC,AB+4U*Rows*K,Rows*K);
LocalTensor<T> g(TPosition::VECCALC,GB,Rows*K),u(TPosition::VECCALC,GB+2U*Rows*K,Rows*K);
LocalTensor<T> sig(TPosition::VECCALC,GB+4U*Rows*K,Rows*K),ones(TPosition::VECCALC,GB+6U*Rows*K,Rows*K);
LocalTensor<T> out(TPosition::VECOUT,GB+8U*Rows*K,Rows*K);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams wc{D,D*2U,0,0,0},bc{1,D*2U,0,0,0},xc{Rows,D*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(K-D),static_cast<T>(0)};
DataCopyPad(hw1,gw1,wc,pad);DataCopyPad(hw2,gw2,wc,pad);
DataCopyPad(hb1,gb1,bc,pad);DataCopyPad(hb2,gb2,bc,pad);DataCopyPad(hx,gx[row*D],xc,pad);
Sync<HardEvent::MTE2_V>();
Cast(fw1,hw1,RoundMode::CAST_NONE,W);Cast(fw2,hw2,RoundMode::CAST_NONE,W);
Cast(fx,hx,RoundMode::CAST_NONE,Rows*K);
Duplicate(ag,0.0f,Rows*K);Duplicate(au,0.0f,Rows*K);Duplicate(ones,static_cast<T>(1),Rows*K);
PipeBarrier<PIPE_V>();
const BinaryRepeatParams rp{1,1,1,K/8U,0,K/8U};
#pragma unroll
for (uint32_t i=0;i<Rows;++i) {
Mul(pg,fx[i*K],fw1,static_cast<uint64_t>(K),static_cast<uint8_t>(D),rp);
Mul(pu,fx[i*K],fw2,static_cast<uint64_t>(K),static_cast<uint8_t>(D),rp);
PipeBarrier<PIPE_V>();
WholeReduceSum(ag[i*K],pg,K,D,1,1,K/8U);WholeReduceSum(au[i*K],pu,K,D,1,1,K/8U);
PipeBarrier<PIPE_V>();
}
Cast(g,ag,RoundMode::CAST_RINT,Rows*K);Cast(u,au,RoundMode::CAST_RINT,Rows*K);PipeBarrier<PIPE_V>();
const BinaryRepeatParams bp{1,1,1,K/16U,K/16U,0};
Add(g,g,hb1,static_cast<uint64_t>(D),static_cast<uint8_t>(Rows),bp);
Add(u,u,hb2,static_cast<uint64_t>(D),static_cast<uint8_t>(Rows),bp);
PipeBarrier<PIPE_V>();GateCounter<T>(g,u,sig,out,ones,Rows*K);
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[row*D],out,xc);PipeBarrier<PIPE_ALL>();
}
private:
template <HardEvent E> __aicore__ inline void Sync() {SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0);}
};
template <typename T>
class PackedMultiBlockGemv {
static constexpr uint32_t D=33U,Rows=5U,K=48U,W=D*32U;
static constexpr uint32_t FB=7424U, PB=20032U;
static constexpr uint32_t BKB=PB+11264U*4U, PRB=BKB+1408U*4U;
static constexpr uint32_t AB=PRB+704U*4U, LB=AB+384U*4U, GB=LB+384U*4U;
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y,GM_ADDR tiling) {
const uint64_t row=static_cast<uint64_t>(GetBlockIdx())*Rows;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gw1.SetGlobalBuffer((__gm__ T*)w1);gw2.SetGlobalBuffer((__gm__ T*)w2);
gb1.SetGlobalBuffer((__gm__ T*)b1);gb2.SetGlobalBuffer((__gm__ T*)b2);gy.SetGlobalBuffer((__gm__ T*)y);
LocalTensor<T> hw1(TPosition::VECIN,0,1280),hw2(TPosition::VECIN,2560,1280),hxAll(TPosition::VECIN,5120,544);
LocalTensor<T> rawB1(TPosition::VECIN,6272,K),rawB2(TPosition::VECIN,6368,K);
LocalTensor<T> hb1(TPosition::VECIN,6464,384),hb2(TPosition::VECIN,6816,192);
LocalTensor<float> fw1(TPosition::VECCALC,FB,1088),fw1Last(TPosition::VECCALC,FB+4352U,192);
LocalTensor<float> fw2(TPosition::VECCALC,FB+5120U,1088),fw2Last(TPosition::VECCALC,FB+9472U,192);
LocalTensor<float> fxLast(TPosition::VECCALC,FB+10240U,224),fx(TPosition::VECCALC,FB+11136U,Rows*64U);
LocalTensor<float> bothP(TPosition::VECCALC,PB,11264),blkP(TPosition::VECCALC,BKB,1408),pairP(TPosition::VECCALC,PRB,704);
LocalTensor<float> ag(TPosition::VECCALC,AB,384);
LocalTensor<float> lastG(TPosition::VECCALC,LB,384),lastU(TPosition::VECCALC,LB+704U,192);
LocalTensor<T> g(TPosition::VECCALC,GB,384),u(TPosition::VECCALC,GB+352U,256);
LocalTensor<T> sig(TPosition::VECCALC,GB+768U,256),ones(TPosition::VECCALC,GB+1280U,256);
LocalTensor<T> out(TPosition::VECOUT,GB+1792U,256);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
LocalTensor<T> rawW1(TPosition::VECIN,79104,1104),rawW2(TPosition::VECIN,81312,1104),rawX(TPosition::VECIN,83520,176);
LocalTensor<uint32_t> offsets(TPosition::VECIN,83968,2000);
LocalTensor<T> allHalf(TPosition::VECIN,0,3136);
LocalTensor<float> allF(TPosition::VECCALC,FB,3136);
GlobalTensor<uint32_t> indexGm;
indexGm.SetGlobalBuffer((__gm__ uint32_t*)(tiling+offsetof(SwiGluTilingData,gatherOffsetsB)),2000);
DataCopyExtParams wc{1,2178,0,0,0},bc{1,66,0,0,0},xcFlat{1,330,0,0,0};
DataCopyPadExtParams<T> wp{true,0,15,static_cast<T>(0)},xp{true,0,11,static_cast<T>(0)};
DataCopyPad(rawW1,gw1,wc,wp);DataCopyPad(rawW2,gw2,wc,wp);
DataCopyPad(rawX,gx[row*D],xcFlat,xp);
DataCopy(offsets,indexGm,2000);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
DataCopyPad(rawB1,gb1,bc,wp);DataCopyPad(rawB2,gb2,bc,wp);
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
constexpr uint64_t mask=0;
Duplicate<float,false>(bothP[5280],0.0f,mask,6U,1,8);
Duplicate<float,false>(bothP[10912],0.0f,mask,6U,1,8);
Duplicate<T,false>(ones,static_cast<T>(1),mask,2U,1,8);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
Gather(hw1,rawW1,offsets,0,1280);
Gather(hw2,rawW2,offsets,0,1280);
Gather(hxAll,rawX,offsets[1280],0,544);
SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
Cast<float,T,false>(allF,allHalf,RoundMode::CAST_NONE,mask,49U,UnaryRepeatParams{1,1,8,4});
const BinaryRepeatParams rp{1,1,1,8,0,8},br8{1,1,1,8,8,8};
#pragma unroll
for (uint32_t i=0;i<Rows;++i) {
Mul<float,false>(bothP[i*W],fx[i*64U],fw1,mask,17U,rp);
Mul<float,false>(bothP[5632U+i*W],fx[i*64U],fw2,mask,17U,rp);
}
BlockReduceSum<float,false>(blkP,bothP,176U,mask,1,1,8);
PairReduceSum<float,false>(pairP,blkP,22U,mask,1,1,8);
PairReduceSum<float,false>(ag,pairP,11U,mask,1,1,8);
Mul<float,false>(lastG,fxLast,fw1Last,mask,3U,br8);
Mul<float,false>(lastU,fxLast,fw2Last,mask,3U,br8);
Add<float,false>(ag,ag,lastG,mask,6U,br8);
Cast<T,float,false>(g,ag,RoundMode::CAST_RINT,mask,6U,UnaryRepeatParams{1,1,4,8});
WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
Gather(hb1,rawB1,offsets[1824],0,176);
Gather(hb2,rawB2,offsets[1824],0,176);
SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
Add<T,false>(g,g,hb1,mask,3U,br8);
const UnaryRepeatParams ur8{1,1,8,8};
Muls<T,false>(sig,g,static_cast<T>(-1),mask,2U,ur8);
Exp<T,false>(sig,sig,mask,2U,ur8);
Adds<T,false>(sig,sig,static_cast<T>(1),mask,2U,ur8);
Div<T,false>(sig,ones,sig,mask,2U,br8);
Mul<T,false>(out,g,sig,mask,2U,br8);
Mul<T,false>(out,out,u,mask,2U,br8);
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[row*D],out,xcFlat);
}
private:
template <HardEvent E> __aicore__ inline void Sync() {SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0);}
};
template <typename T>
class PackedSubBlockGemv {
static constexpr uint32_t D=13U,Rows=3U,K=16U,W=256U;
static constexpr uint32_t HB=4U*W+2U*Rows*K;
static constexpr uint32_t FB=HB+4U*K;
static constexpr uint32_t XB=FB+8U*W, PB=XB+4U*Rows*K;
static constexpr uint32_t AB=PB+8U*W, GB=AB+8U*Rows*K;
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y) {
const uint32_t blk=GetBlockIdx();
if (blk>=7U) { return; }
const uint64_t row=static_cast<uint64_t>(blk)*Rows;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gw1.SetGlobalBuffer((__gm__ T*)w1);gw2.SetGlobalBuffer((__gm__ T*)w2);
gb1.SetGlobalBuffer((__gm__ T*)b1);gb2.SetGlobalBuffer((__gm__ T*)b2);gy.SetGlobalBuffer((__gm__ T*)y);
LocalTensor<T> hw1(TPosition::VECIN,0,W),hw2(TPosition::VECIN,2U*W,W);
LocalTensor<T> hb1(TPosition::VECIN,HB,K),hb2(TPosition::VECIN,HB+2U*K,K);
LocalTensor<T> gu(TPosition::VECCALC,GB,128),sig(TPosition::VECCALC,GB+256U,128),ones(TPosition::VECCALC,GB+512U,128);
LocalTensor<T> out(TPosition::VECOUT,GB+768U,128);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
LocalTensor<T> bothHalfW(TPosition::VECIN,0,512);
LocalTensor<T> bothBias(TPosition::VECIN,HB,32);
LocalTensor<T> halfExpandedX(TPosition::VECIN,10752,192);
LocalTensor<float> bothP(TPosition::VECCALC,19904,1536);
LocalTensor<float> bothBlock(TPosition::VECCALC,34048,192);
LocalTensor<float> acc(TPosition::VECCALC,AB,96);
DataCopyExtParams wc{D,D*2U,0,0,0},bc{1,D*2U,0,0,0},xcIn{Rows,D*2U,0,3,0},xcOut{Rows,D*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,3,static_cast<T>(0)};
DataCopyPad(hw1,gw1,wc,pad);
DataCopyPad(hw2,gw2,wc,pad);
DataCopyPad(halfExpandedX,gx[row*D],xcIn,pad);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
DataCopyPad(hb1,gb1,bc,pad);DataCopyPad(hb2,gb2,bc,pad);
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
constexpr uint64_t mask=0;
Duplicate<T,false>(ones,static_cast<T>(1),mask,1U,1,8);
Duplicate<float,false>(bothP,0.0f,mask,24U,1,8);
SetVectorMask<T,MaskMode::NORMAL>(0,0xFFFFFFFFFFFFULL);
Duplicate<T,false>(hw1[208],static_cast<T>(0),mask,1U,1,8);
Duplicate<T,false>(hw2[208],static_cast<T>(0),mask,1U,1,8);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
SetVectorMask<T,MaskMode::NORMAL>(0,0xFFFFULL);
Adds<T,false>(halfExpandedX[16],halfExpandedX,static_cast<T>(0),mask,3U,UnaryRepeatParams{1,1,4,4});
SetVectorMask<T,MaskMode::NORMAL>(0,0xFFFFFFFFULL);
Adds<T,false>(halfExpandedX[32],halfExpandedX,static_cast<T>(0),mask,3U,UnaryRepeatParams{1,1,4,4});
SetVectorMask<float,MaskMode::NORMAL>(0,~0ULL);
const BinaryRepeatParams rp{1,1,1,8,0,4};
#pragma unroll
for (uint32_t i=0;i<Rows;++i) {
MulAddDst<float,T,false>(bothP[i*512U],halfExpandedX[i*64U],bothHalfW,mask,8,rp);
}
BlockReduceSum<float,false>(bothBlock,bothP,24,mask,1,1,8);
PairReduceSum<float,false>(acc,bothBlock,3,mask,1,1,8);
Cast<T,float,false>(gu,acc,RoundMode::CAST_RINT,mask,2U,UnaryRepeatParams{1,1,4,8});
WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
SetVectorMask<T,MaskMode::NORMAL>(0,0xFFFFFFFFULL);
Add<T,false>(gu,gu,bothBias,mask,3U,BinaryRepeatParams{1,1,1,2,2,0});
SetVectorMask<T,MaskMode::NORMAL>(~0ULL,~0ULL);
const UnaryRepeatParams ur8{1,1,8,8};
const BinaryRepeatParams br8{1,1,1,8,8,8};
Muls<T,false>(sig,gu,static_cast<T>(-1),mask,1U,ur8);
Exp<T,false>(sig,sig,mask,1U,ur8);
Adds<T,false>(sig,sig,static_cast<T>(1),mask,1U,ur8);
Div<T,false>(sig,ones,sig,mask,1U,br8);
Mul<T,false>(sig,gu,sig,mask,1U,br8);
SetVectorMask<T,MaskMode::NORMAL>(0,0xFFFFFFFFFFFFULL);
Mul<T,false>(out,sig,gu[16],mask,1U,BinaryRepeatParams{1,2,2,8,8,8});
Sync<HardEvent::V_MTE3>();
DataCopyPad(gy[row*D],out,xcOut);
}
private:
template <HardEvent E> __aicore__ inline void Sync() {SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0);}
};
namespace optiling {
using namespace AscendC;
using namespace matmul;
constexpr uint32_t kSlotCount = 4;
constexpr uint32_t kBlockRows = 16;
__aicore__ constexpr MatmulConfig DirectBasicConfig(uint32_t rows)
{
auto config = GetBasicConfig(rows, 128, 128);
config.enUnitFlag = true;
return config;
}
constexpr MatmulConfig kBasic16 = DirectBasicConfig(16);
constexpr MatmulConfig kBasic32 = DirectBasicConfig(32);
constexpr MatmulConfig kBasic128 = DirectBasicConfig(128);
template <typename T, uint32_t FixedM = 0, uint32_t FixedD = 0, bool Unaligned = false, bool Resident = false, bool FullTile = false, uint32_t FixedN = 0>
class DirectPairCube {
public:
__aicore__ inline void Init(uint32_t k, uint32_t m, uint32_t n) {
realK_ = FixedD != 0 ? FixedD : k;
k_ = Unaligned ? (realK_ + 15U) / 16U * 16U : realK_;
m_ = FixedM != 0 ? FixedM : m;
n_ = FixedN != 0 ? FixedN : (FixedD != 0 ? (FixedD < 64U ? FixedD : 64U) : n);
}
__aicore__ inline void PrepareWeights(const GlobalTensor<T> &w1,const GlobalTensor<T> &w2,uint32_t cols) {
LocalTensor<T> b1(TPosition::B1,m_*k_*sizeof(T),2U*n_*k_);
LocalTensor<T> b2(TPosition::B2,0,2U*n_*k_);
Nd2NzParams bp{1,static_cast<uint16_t>(cols),static_cast<uint16_t>(realK_),0,
static_cast<uint16_t>(realK_),static_cast<uint16_t>(2U*n_),1,0};
DataCopy(b1,w1,bp);DataCopy(b1[n_*16U],w2,bp);
Sync<HardEvent::MTE2_MTE1>();
LoadData2dParams bl;
bl.repeatTimes=static_cast<uint8_t>(2U*n_*k_/256U);bl.srcStride=1;
LoadData(b2,b1,bl);
Sync<HardEvent::MTE1_M>();
}
__aicore__ inline void Compute(const GlobalTensor<T> &x, const GlobalTensor<T> &w1,
const GlobalTensor<T> &w2, const GlobalTensor<T> &dst,
uint32_t rows, uint32_t cols) {
LocalTensor<T> a1(TPosition::A1, 0, m_ * k_);
LocalTensor<T> b1(TPosition::B1, m_ * k_ * sizeof(T), 2U * n_ * k_);
LocalTensor<T> a2(TPosition::A2, 0, m_ * k_);
LocalTensor<T> b2(TPosition::B2, 0, 2U * n_ * k_);
LocalTensor<float> c(TPosition::CO1, 0, 2U * m_ * n_);
const uint32_t alignedM = FullTile ? FixedM : (rows + 15U) / 16U * 16U;
Nd2NzParams ap{1, static_cast<uint16_t>(rows), static_cast<uint16_t>(realK_), 0,
static_cast<uint16_t>(realK_), static_cast<uint16_t>(alignedM), 1, 0};
Nd2NzParams bp{1, static_cast<uint16_t>(cols), static_cast<uint16_t>(realK_), 0,
static_cast<uint16_t>(realK_), static_cast<uint16_t>(2U * n_), 1, 0};
DataCopy(a1, x, ap);
if constexpr (!Resident) {
DataCopy(b1, w1, bp);
DataCopy(b1[n_ * 16U], w2, bp);
}
Sync<HardEvent::MTE2_MTE1>();
LoadData2dParams al;
al.repeatTimes = static_cast<uint8_t>(k_ / 16U);
al.srcStride = static_cast<uint16_t>(alignedM / 16U);
if constexpr (FullTile) {
#pragma unroll
for (uint32_t mb=0;mb<FixedM/16U;++mb) {
LoadData(a2[mb*k_*16U],a1[mb*256U],al);
}
} else {
for (uint32_t mb = 0; mb < alignedM / 16U; ++mb) {
LoadData(a2[mb * k_ * 16U], a1[mb * 256U], al);
}
}
LoadData2dParams bl;
bl.repeatTimes = static_cast<uint8_t>(2U * n_ * k_ / 256U);
bl.srcStride = 1;
if constexpr (!Resident) { LoadData(b2, b1, bl); }
if constexpr (Resident) { Sync<HardEvent::MTE1_MTE2>(); }
Sync<HardEvent::MTE1_M>();
MmadParams mad;
mad.m = static_cast<uint16_t>(alignedM);
mad.n = static_cast<uint16_t>(2U * n_);
mad.k = static_cast<uint16_t>(k_);
mad.cmatrixInitVal = true;
Mmad(c, a2, b2, mad);
if constexpr (Resident) { Sync<HardEvent::M_MTE1>(); }
Sync<HardEvent::M_FIX>();
FixpipeParamsV220 fix;
fix.nSize = static_cast<uint16_t>(2U * n_);
fix.mSize = static_cast<uint16_t>(rows);
fix.srcStride = static_cast<uint16_t>(alignedM);
fix.dstStride = 2U * n_;
fix.quantPre = QuantMode_t::F322F16;
Fixpipe<T, float, CFG_ROW_MAJOR>(dst, c, fix);
if constexpr (Resident) { Sync<HardEvent::FIX_M>(); }
else { PipeBarrier<PIPE_ALL>(); }
}
private:
template <HardEvent E>
__aicore__ inline void Sync() {
constexpr event_t event = EVENT_ID0;
SetFlag<E>(event);
WaitFlag<E>(event);
}
uint32_t k_ = 0, realK_ = 0, m_ = 0, n_ = 0;
};
struct EmptyMatmulStorage {};
template <bool Direct, typename A, typename B, typename C, const auto& Config>
struct MatmulStorageSelector {
using Type = MatmulImpl<A, B, C, C, Config>;
};
template <typename A, typename B, typename C, const auto& Config>
struct MatmulStorageSelector<true, A, B, C, Config> {
using Type = EmptyMatmulStorage;
};
template <typename xType, const auto& Config = CFG_NORM, bool Direct = false,
bool Padded = false, bool Merged = false>
class SwiGluCubeImpl {
public:
using AType = MatmulType<TPosition::GM, CubeFormat::ND, xType, false>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, xType, true>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, xType>;
typename MatmulStorageSelector<Direct, AType, BType, CType, Config>::Type mm;
DirectPairCube<xType> direct;
__aicore__ inline void Init(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
GM_ADDR userWorkspace, TPipe *pipe, const SwiGluTilingData *td)
{
pipe_ = pipe;
core_ = GetBlockIdx();
sub_ = GetSubBlockIdx();
if ASCEND_IS_AIV { core_ /= 2U; }
m_ = td->m;
d_ = static_cast<uint32_t>(td->d);
computeD_ = td->dAlign;
taskM_ = td->taskM;
taskN_ = td->taskN;
taskNum_ = td->taskNum;
cores_ = td->usedCoreNum;
tasksN_ = (d_ + taskN_ - 1U) / taskN_;
plane_ = taskM_ * taskN_;
x_.SetGlobalBuffer((__gm__ xType *)x, m_ * d_);
w1_.SetGlobalBuffer((__gm__ xType *)w1, static_cast<uint64_t>(d_) * d_);
w2_.SetGlobalBuffer((__gm__ xType *)w2, static_cast<uint64_t>(d_) * d_);
b1_.SetGlobalBuffer((__gm__ xType *)b1, d_);
b2_.SetGlobalBuffer((__gm__ xType *)b2, d_);
y_.SetGlobalBuffer((__gm__ xType *)y, m_ * d_);
slots_.SetGlobalBuffer((__gm__ xType *)userWorkspace,
static_cast<uint64_t>(plane_) * 2 * kSlotCount * cores_);
if constexpr (padded_) {
const uint64_t base = static_cast<uint64_t>(plane_) * 2 * kSlotCount * cores_;
const uint64_t wBase = base + m_ * computeD_;
paddedX_.SetGlobalBuffer((__gm__ xType *)userWorkspace + base, m_ * computeD_);
paddedW1_.SetGlobalBuffer((__gm__ xType *)userWorkspace + wBase,
static_cast<uint64_t>(computeD_) * computeD_ * (merged_ ? 2U : 1U));
paddedW2_.SetGlobalBuffer((__gm__ xType *)userWorkspace + wBase +
static_cast<uint64_t>(computeD_) * computeD_,
static_cast<uint64_t>(computeD_) * computeD_);
packRows_ = 4096U / computeD_;
}
if ASCEND_IS_AIC {
if constexpr (Direct) {
direct.Init(computeD_, taskM_, taskN_);
} else {
mm.Init(&td->cubeTilingData, pipe_);
mm.SetHF32(false, 0);
const uint32_t matmulN = merged_ ? 2U * taskN_ : taskN_;
mm.SetSingleShape(taskM_, matmulN, computeD_);
mm.SetOrgShape(taskM_, matmulN, computeD_, computeD_);
}
}
if ASCEND_IS_AIV {
const uint32_t cBytes = (taskM_ / 2U) * taskN_ * sizeof(xType);
const uint32_t vBytes = kBlockRows * taskN_ * sizeof(xType);
pipe_->InitBuffer(gateQ_, 1, cBytes);
pipe_->InitBuffer(upQ_, 1, cBytes);
pipe_->InitBuffer(bias1Q_, 1, taskN_ * sizeof(xType));
pipe_->InitBuffer(bias2Q_, 1, taskN_ * sizeof(xType));
pipe_->InitBuffer(outQ_, 1, vBytes);
pipe_->InitBuffer(gateBuf_, vBytes);
pipe_->InitBuffer(upBuf_, vBytes);
pipe_->InitBuffer(negBuf_, vBytes);
pipe_->InitBuffer(expBuf_, vBytes);
pipe_->InitBuffer(onesBuf_, vBytes);
pipe_->InitBuffer(sigmoidBuf_, vBytes);
if constexpr (padded_) { pipe_->InitBuffer(packQ_, 1, packRows_ * computeD_ * sizeof(xType)); }
}
}
__aicore__ inline void Process()
{
if (core_ >= cores_) { return; }
if constexpr (padded_) {
if ASCEND_IS_AIV {
PackMatrix(paddedX_, x_, m_, m_);
PackMatrix(paddedW1_, w1_, d_, computeD_);
PackMatrix(paddedW2_, w2_, d_, computeD_);
SyncAll();
CrossCoreSetFlag<0x2, PIPE_MTE3>(0x4);
}
if ASCEND_IS_AIC { CrossCoreWaitFlag(0x4); }
}
uint32_t local = 0;
for (uint32_t task = core_; task < taskNum_; task += cores_, ++local) {
const uint32_t slot = local % kSlotCount;
const uint64_t row = static_cast<uint64_t>(task / tasksN_) * taskM_;
const uint32_t col = (task % tasksN_) * taskN_;
const uint32_t rows = Min(taskM_, static_cast<uint32_t>(m_ - row));
const uint32_t cols = Min(taskN_, d_ - col);
const uint64_t off = (static_cast<uint64_t>(core_) * kSlotCount + slot) * 2 * plane_;
if ASCEND_IS_AIC {
if (local >= kSlotCount) { WaitConsumed(slot); }
if constexpr (Direct) {
direct.Compute(x_[row * d_], w1_[static_cast<uint64_t>(col) * d_],
w2_[static_cast<uint64_t>(col) * d_], slots_[off], rows, cols);
} else {
if constexpr (padded_) {
const uint32_t computeCols = Min(taskN_, computeD_ - col);
if constexpr (merged_) {
Projection(paddedW1_, row, 0, rows, 2U * computeCols, off);
} else {
Projection(paddedW1_, row, col, rows, computeCols, off);
Projection(paddedW2_, row, col, rows, computeCols, off + plane_);
}
} else {
Projection(w1_, row, col, rows, cols, off);
Projection(w2_, row, col, rows, cols, off + plane_);
}
}
SignalReady(slot);
}
if ASCEND_IS_AIV {
WaitReady(slot);
Epilogue(off, row, col, rows, cols);
SignalConsumed(slot);
}
}
if ASCEND_IS_AIC {
for (uint32_t slot = 0; slot < Min(local, kSlotCount); ++slot) { WaitConsumed(slot); }
}
}
private:
__aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }
__aicore__ inline void PackMatrix(const GlobalTensor<xType> &dst,
const GlobalTensor<xType> &src,
uint64_t realRows, uint64_t totalRows)
{
const uint64_t chunks = (totalRows + packRows_ - 1U) / packRows_;
for (uint64_t block = GetBlockIdx(); block < chunks; block += cores_ * 2U) {
const uint64_t row = block * packRows_;
const uint32_t nr = Min(packRows_, static_cast<uint32_t>(totalRows - row));
const uint32_t valid = row < realRows ? Min(nr, static_cast<uint32_t>(realRows - row)) : 0;
LocalTensor<xType> tile = packQ_.AllocTensor<xType>();
if (valid != 0) {
DataCopyExtParams copy{static_cast<uint16_t>(valid), d_ * 2U, 0, 0, 0};
DataCopyPadExtParams<xType> pad{true, 0, static_cast<uint8_t>(computeD_ - d_), 0};
DataCopyPad(tile, src[row * d_], copy, pad);
}
if (valid < nr) {
Duplicate(tile[valid * computeD_], static_cast<xType>(0), (nr - valid) * computeD_);
const event_t event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
SetFlag<HardEvent::V_MTE3>(event);
WaitFlag<HardEvent::V_MTE3>(event);
}
packQ_.EnQue(tile);
tile = packQ_.DeQue<xType>();
DataCopy(dst[row * computeD_], tile, nr * computeD_);
packQ_.FreeTensor(tile);
}
}
__aicore__ inline void Projection(const GlobalTensor<xType> &weight, uint64_t row, uint32_t col,
uint32_t rows, uint32_t cols, uint64_t off)
{
mm.SetTail(rows, cols, computeD_);
if constexpr (padded_) { mm.SetTensorA(paddedX_[row * computeD_], false); }
else { mm.SetTensorA(x_[row * d_], false); }
mm.SetTensorB(weight[static_cast<uint64_t>(col) * computeD_], true);
mm.IterateAll(slots_[off], false);
mm.End();
PipeBarrier<PIPE_ALL>();
}
__aicore__ inline void Epilogue(uint64_t off, uint64_t row, uint32_t col,
uint32_t rows, uint32_t cols)
{
const uint32_t halfRows = (rows + 1U) / 2U;
const uint32_t begin = sub_ * halfRows;
if (begin >= rows) { return; }
const uint32_t count = Min(halfRows, rows - begin);
const uint32_t pitch = AlignUp16(cols);
const uint32_t cPitch = merged_ ? 2U * taskN_ : taskN_;
LocalTensor<xType> gate = gateQ_.AllocTensor<xType>();
LocalTensor<xType> up = upQ_.AllocTensor<xType>();
DataCopyExtParams cParams{static_cast<uint16_t>(count), cols * static_cast<uint32_t>(sizeof(xType)),
(cPitch - cols) * static_cast<uint32_t>(sizeof(xType)), 0, 0};
DataCopyPadExtParams<xType> pad{false, 0, 0, 0};
const uint64_t cOff = off + static_cast<uint64_t>(begin) * cPitch;
DataCopyPad(gate, slots_[cOff], cParams, pad);
DataCopyPad(up, slots_[cOff + (merged_ ? taskN_ : plane_)], cParams, pad);
gateQ_.EnQue(gate);
upQ_.EnQue(up);
LocalTensor<xType> b1 = bias1Q_.AllocTensor<xType>();
LocalTensor<xType> b2 = bias2Q_.AllocTensor<xType>();
DataCopyExtParams biasParams{1, cols * static_cast<uint32_t>(sizeof(xType)), 0, 0, 0};
DataCopyPad(b1, b1_[col], biasParams, pad);
DataCopyPad(b2, b2_[col], biasParams, pad);
bias1Q_.EnQue(b1);
bias2Q_.EnQue(b2);
gate = gateQ_.DeQue<xType>();
up = upQ_.DeQue<xType>();
b1 = bias1Q_.DeQue<xType>();
b2 = bias2Q_.DeQue<xType>();
LocalTensor<xType> a = gateBuf_.Get<xType>();
LocalTensor<xType> b = upBuf_.Get<xType>();
LocalTensor<xType> neg = negBuf_.Get<xType>();
LocalTensor<xType> exp = expBuf_.Get<xType>();
LocalTensor<xType> ones = onesBuf_.Get<xType>();
LocalTensor<xType> sig = sigmoidBuf_.Get<xType>();
for (uint32_t group = 0; group < count; group += kBlockRows) {
const uint32_t nr = Min(kBlockRows, count - group);
const uint32_t n = nr * pitch;
const uint8_t rowStride = static_cast<uint8_t>(pitch / 16U);
const BinaryRepeatParams biasRepeat{1, 1, 1, rowStride, rowStride, 0};
for (uint32_t k = 0; k < cols; k += 128U) {
const uint32_t width = Min(128U, cols - k);
Add(a[k], gate[group * pitch + k], b1[k], static_cast<uint64_t>(width),
static_cast<uint8_t>(nr), biasRepeat);
Add(b[k], up[group * pitch + k], b2[k], static_cast<uint64_t>(width),
static_cast<uint8_t>(nr), biasRepeat);
}
LocalTensor<xType> out = outQ_.AllocTensor<xType>();
GateCounter<xType,true>(a,b,exp,out,ones,n);
outQ_.EnQue(out);
out = outQ_.DeQue<xType>();
DataCopyExtParams outParams{static_cast<uint16_t>(nr), cols * static_cast<uint32_t>(sizeof(xType)),
0, (d_ - cols) * static_cast<uint32_t>(sizeof(xType)), 0};
DataCopyPad(y_[(row + begin + group) * d_ + col], out, outParams);
outQ_.FreeTensor(out);
}
gateQ_.FreeTensor(gate);
upQ_.FreeTensor(up);
bias1Q_.FreeTensor(b1);
bias2Q_.FreeTensor(b2);
}
__aicore__ inline void WaitReady(uint32_t s) {
if (s == 0) { CrossCoreWaitFlag(0x6); }
else if (s == 1) { CrossCoreWaitFlag(0x7); }
else if (s == 2) { CrossCoreWaitFlag(0xB); }
else { CrossCoreWaitFlag(0xE); }
}
__aicore__ inline void SignalReady(uint32_t s) {
if (s == 0) { CrossCoreSetFlag<0x2, PIPE_FIX>(0x6); }
else if (s == 1) { CrossCoreSetFlag<0x2, PIPE_FIX>(0x7); }
else if (s == 2) { CrossCoreSetFlag<0x2, PIPE_FIX>(0xB); }
else { CrossCoreSetFlag<0x2, PIPE_FIX>(0xE); }
}
__aicore__ inline void WaitConsumed(uint32_t s) {
if (s == 0) { CrossCoreWaitFlag(0x8); }
else if (s == 1) { CrossCoreWaitFlag(0x9); }
else if (s == 2) { CrossCoreWaitFlag(0xC); }
else { CrossCoreWaitFlag(0xF); }
}
__aicore__ inline void SignalConsumed(uint32_t s) {
if (s == 0) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0x8); }
else if (s == 1) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0x9); }
else if (s == 2) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0xC); }
else { CrossCoreSetFlag<0x2, PIPE_MTE3>(0xF); }
}
TPipe *pipe_ = nullptr;
TQue<QuePosition::VECIN, 1> gateQ_, upQ_, bias1Q_, bias2Q_;
TQue<QuePosition::VECOUT, 1> outQ_;
TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> packQ_;
TBuf<TPosition::VECCALC> gateBuf_, upBuf_, negBuf_, expBuf_, onesBuf_, sigmoidBuf_;
GlobalTensor<xType> x_, w1_, w2_, b1_, b2_, y_, slots_;
GlobalTensor<xType> paddedX_, paddedW1_, paddedW2_;
uint64_t m_ = 0;
uint32_t d_ = 0, taskM_ = 0, taskN_ = 0, taskNum_ = 0, tasksN_ = 0;
uint32_t core_ = 0, sub_ = 0, cores_ = 0, plane_ = 0;
uint32_t computeD_ = 0, packRows_ = 0;
static constexpr bool padded_ = Padded;
static constexpr bool merged_ = Merged || Direct;
};
template <typename T, bool Padded = false, bool Merged = false>
__aicore__ inline void RunCube(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
GM_ADDR workspace, TPipe *pipe, const SwiGluTilingData *td)
{
SwiGluCubeImpl<T, CFG_NORM, false, Padded, Merged> op;
op.Init(x, w1, b1, w2, b2, y, workspace, pipe, td);
op.Process();
}
template <typename T, uint32_t TileM, uint32_t FixedD = 0, bool Unaligned = false, bool Resident = false, bool FullTile = false>
class SlimDirectCube {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
GM_ADDR workspace, const SwiGluDirectTilingData *td) {
d_ = FixedD != 0 ? FixedD : static_cast<uint32_t>(td->d);
n_ = FixedD != 0 ? (FixedD < 64U ? FixedD : 64U) : td->taskN;
const uint32_t cores = td->usedCoreNum;
uint32_t core = GetBlockIdx();
if ASCEND_IS_AIV { core /= 2U; }
x_.SetGlobalBuffer((__gm__ T *)x, td->m * d_);
w1_.SetGlobalBuffer((__gm__ T *)w1, static_cast<uint64_t>(d_) * d_);
w2_.SetGlobalBuffer((__gm__ T *)w2, static_cast<uint64_t>(d_) * d_);
b1_.SetGlobalBuffer((__gm__ T *)b1, d_);
b2_.SetGlobalBuffer((__gm__ T *)b2, d_);
y_.SetGlobalBuffer((__gm__ T *)y, td->m * d_);
const uint32_t plane = tm_ * n_;
slots_.SetGlobalBuffer((__gm__ T *)workspace, static_cast<uint64_t>(2U * plane) * 4U * cores);
if ASCEND_IS_AIC { cube_.Init(d_, tm_, n_); }
SetAtomicNone();
if ASCEND_IS_AIV {
SetMaskNorm();
SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
}
const uint32_t tasksN = (d_ + n_ - 1U) / n_;
if constexpr (Resident) {
const uint32_t col=core%tasksN*n_;
const uint32_t cols=FullTile?n_:Min(n_,d_-col);
if ASCEND_IS_AIC {
cube_.PrepareWeights(w1_[static_cast<uint64_t>(col)*d_],w2_[static_cast<uint64_t>(col)*d_],cols);
}
if ASCEND_IS_AIV {
LocalTensor<T> bias(TPosition::VECIN,tm_*n_*sizeof(T),2U*n_);
if constexpr (Unaligned) {
DataCopyExtParams bp{1,cols*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>((cols+15U)/16U*16U-cols),static_cast<T>(0)};
DataCopyPad(bias,b1_[col],bp,pad);DataCopyPad(bias[n_],b2_[col],bp,pad);
} else {
DataCopy(bias,b1_[col],cols);DataCopy(bias[n_],b2_[col],cols);
}
}
}
uint32_t local = 0;
for (uint32_t task = core; task < td->taskNum; task += cores, ++local) {
const uint32_t slot = local % 4U;
const uint64_t row = static_cast<uint64_t>(task / tasksN) * tm_;
const uint32_t col = task % tasksN * n_;
const uint32_t rows = FullTile?tm_:Min(tm_, td->m - row);
const uint32_t cols = FullTile?n_:Min(n_, d_ - col);
const uint64_t off = (static_cast<uint64_t>(core) * 4U + slot) * 2U * plane;
if ASCEND_IS_AIC {
if (local >= 4U) { WaitConsumed(slot); }
cube_.Compute(x_[row * d_], w1_[static_cast<uint64_t>(col) * d_],
w2_[static_cast<uint64_t>(col) * d_], slots_[off], rows, cols);
Ready(slot);
}
if ASCEND_IS_AIV {
WaitReady(slot);
Epilogue(off, row, col, rows, cols);
Consumed(slot);
}
}
if ASCEND_IS_AIC {
for (uint32_t slot = 0; slot < Min(local, 4U); ++slot) { WaitConsumed(slot); }
}
PipeBarrier<PIPE_ALL>();
}
private:
__aicore__ inline uint32_t Min(uint32_t a, uint64_t b) { return a < b ? a : b; }
template <HardEvent E> __aicore__ inline void Sync() {
constexpr event_t e = EVENT_ID0;
SetFlag<E>(e); WaitFlag<E>(e);
}
__aicore__ inline void Epilogue(uint64_t off, uint64_t row, uint32_t col,
uint32_t rows, uint32_t cols) {
const uint32_t half = FullTile?tm_/2U:(rows + 1U) / 2U;
const uint32_t begin = GetSubBlockIdx() * half;
if constexpr (!FullTile) { if (begin >= rows) { return; } }
const uint32_t nr = FullTile?tm_/2U:Min(half, rows - begin);
const uint32_t capacity = tm_ / 2U * n_;
LocalTensor<T> input(TPosition::VECIN, 0, (tm_ + 2U) * n_);
auto g = input;
auto u = input[capacity];
auto b1 = input[2U * capacity];
auto b2 = input[2U * capacity + n_];
LocalTensor<T> work(TPosition::VECOUT, (tm_ + 2U) * n_ * sizeof(T), tm_ * n_);
auto s = work;
auto out = work[capacity];
const uint64_t start = off + begin * 2U * n_;
if constexpr (Unaligned) {
const uint32_t pitch = (cols + 15U) / 16U * 16U;
DataCopyExtParams cp{static_cast<uint16_t>(nr), cols * 2U,
(2U * n_ - cols) * 2U, (n_ - pitch) / 16U, 0};
DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(pitch - cols), static_cast<T>(0)};
DataCopyPad(g, slots_[start], cp, pad);
DataCopyPad(u, slots_[start + n_], cp, pad);
DataCopyExtParams bp{1, cols * 2U, 0, 0, 0};
if constexpr (!Resident) {
DataCopyPad(b1, b1_[col], bp, pad);
DataCopyPad(b2, b2_[col], bp, pad);
}
} else {
DataCopyParams cp{static_cast<uint16_t>(nr), static_cast<uint16_t>(cols / 16U),
static_cast<uint16_t>((2U * n_ - cols) / 16U),
static_cast<uint16_t>((n_ - cols) / 16U)};
DataCopy(g, slots_[start], cp);
DataCopy(u, slots_[start + n_], cp);
if constexpr (!Resident) {
DataCopy(b1, b1_[col], cols);
DataCopy(b2, b2_[col], cols);
}
}
Sync<HardEvent::MTE2_V>();
const uint8_t stride = static_cast<uint8_t>(n_ / 16U);
const BinaryRepeatParams repeat{1, 1, 1, stride, stride, 0};
Add(g, g, b1, static_cast<uint64_t>(cols), static_cast<uint8_t>(nr), repeat);
Add(u, u, b2, static_cast<uint64_t>(cols), static_cast<uint8_t>(nr), repeat);
PipeBarrier<PIPE_V>();
const uint32_t count = nr * n_;
GateCounter<T,true>(g,u,s,out,out,count);
Sync<HardEvent::V_MTE3>();
if constexpr (Unaligned) {
const uint32_t pitch = (cols + 15U) / 16U * 16U;
DataCopyExtParams output{static_cast<uint16_t>(nr), cols * 2U,
(n_ - pitch) / 16U, (d_ - cols) * 2U, 0};
DataCopyPad(y_[(row + begin) * d_ + col], out, output);
} else {
DataCopyParams output{static_cast<uint16_t>(nr), static_cast<uint16_t>(cols / 16U),
static_cast<uint16_t>((n_ - cols) / 16U),
static_cast<uint16_t>((d_ - cols) / 16U)};
DataCopy(y_[(row + begin) * d_ + col], out, output);
}
Sync<HardEvent::MTE3_MTE2>();
}
__aicore__ inline void WaitReady(uint32_t s) {
if (s == 0) { CrossCoreWaitFlag(0x6); } else if (s == 1) { CrossCoreWaitFlag(0x7); }
else if (s == 2) { CrossCoreWaitFlag(0xB); } else { CrossCoreWaitFlag(0xE); }
}
__aicore__ inline void Ready(uint32_t s) {
if (s == 0) { CrossCoreSetFlag<0x2, PIPE_FIX>(0x6); } else if (s == 1) { CrossCoreSetFlag<0x2, PIPE_FIX>(0x7); }
else if (s == 2) { CrossCoreSetFlag<0x2, PIPE_FIX>(0xB); } else { CrossCoreSetFlag<0x2, PIPE_FIX>(0xE); }
}
__aicore__ inline void WaitConsumed(uint32_t s) {
if (s == 0) { CrossCoreWaitFlag(0x8); } else if (s == 1) { CrossCoreWaitFlag(0x9); }
else if (s == 2) { CrossCoreWaitFlag(0xC); } else { CrossCoreWaitFlag(0xF); }
}
__aicore__ inline void Consumed(uint32_t s) {
if (s == 0) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0x8); } else if (s == 1) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0x9); }
else if (s == 2) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0xC); } else { CrossCoreSetFlag<0x2, PIPE_MTE3>(0xF); }
}
DirectPairCube<T, TileM, FixedD, Unaligned, Resident, FullTile> cube_;
GlobalTensor<T> x_, w1_, w2_, b1_, b2_, y_, slots_;
static constexpr uint32_t tm_ = TileM;
uint32_t d_ = 0, n_ = 0;
};
template <typename T>
class UnalignedTileCube {
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y,GM_ADDR ws) {
uint32_t core=GetBlockIdx();
if ASCEND_IS_AIV {core/=2U;}
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy,tmp;
gx.SetGlobalBuffer((__gm__ T*)x);gw1.SetGlobalBuffer((__gm__ T*)w1);gw2.SetGlobalBuffer((__gm__ T*)w2);
gb1.SetGlobalBuffer((__gm__ T*)b1);gb2.SetGlobalBuffer((__gm__ T*)b2);gy.SetGlobalBuffer((__gm__ T*)y);
tmp.SetGlobalBuffer((__gm__ T*)ws);
const uint32_t off=core*32U*96U;
SetAtomicNone();
if ASCEND_IS_AIC {
cube_.Init(33U,32U,48U);
cube_.Compute(gx[core*17U*33U],gw1,gw2,tmp[off],17U,33U);
CrossCoreSetFlag<0x2,PIPE_FIX>(0x6);
CrossCoreWaitFlag(0x8);
}
if ASCEND_IS_AIV {
const uint32_t sub=GetSubBlockIdx(), nr=9U-sub, begin=9U*sub;
LocalTensor<T> g(TPosition::VECIN,0,432),u(TPosition::VECIN,864,432);
LocalTensor<T> b1t(TPosition::VECIN,1728,48),b2t(TPosition::VECIN,1824,48);
LocalTensor<T> sig(TPosition::VECCALC,1920,432),out(TPosition::VECOUT,2784,432);
SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams bc{1,66,0,0,0};DataCopyPadExtParams<T> pad{true,0,15,static_cast<T>(0)};
DataCopyPad(b1t,gb1,bc,pad);DataCopyPad(b2t,gb2,bc,pad);
CrossCoreWaitFlag(0x6);
DataCopyParams cp{static_cast<uint16_t>(nr),3,3,0};
DataCopy(g,tmp[off+begin*96U],cp);DataCopy(u,tmp[off+begin*96U+48U],cp);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
const BinaryRepeatParams rp{1,1,1,3,3,0};
Add(g,g,b1t,static_cast<uint64_t>(33),static_cast<uint8_t>(nr),rp);
Add(u,u,b2t,static_cast<uint64_t>(33),static_cast<uint8_t>(nr),rp);
PipeBarrier<PIPE_V>();GateCounter<T,true>(g,u,sig,out,out,nr*48U);
SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
DataCopyExtParams oc{static_cast<uint16_t>(nr),66,0,0,0};
DataCopyPad(gy[(core*17U+begin)*33U],out,oc);
CrossCoreSetFlag<0x2,PIPE_MTE3>(0x8);
}
PipeBarrier<PIPE_ALL>();
}
private:
DirectPairCube<T,32U,0U,true,false,true> cube_;
};
template <typename T,uint32_t TM,uint32_t D,uint32_t N>
class DoubleBufferedPairCube {
static constexpr uint32_t A=TM*D,B=2U*N*D;
static constexpr bool DoubleA2=(2U*A*sizeof(T)<=64U*1024U);
static constexpr bool DoubleC=(4U*TM*N*sizeof(float)<=128U*1024U);
static_assert((2U*A+B)*sizeof(T)<=512U*1024U,"L1 capacity");
static_assert(A*sizeof(T)<=64U*1024U && B*sizeof(T)<=64U*1024U,"L0A/B capacity");
public:
__aicore__ inline void PrepareWeights(const GlobalTensor<T>& w1,const GlobalTensor<T>& w2,const GlobalTensor<T>& x0) {
LocalTensor<T> b1(TPosition::B1,2U*A*sizeof(T),B),b2(TPosition::B2,0,B);
LocalTensor<T> a1(TPosition::A1,0,A);
const Nd2NzParams bp{1,N,D,0,D,2U*N,1,0};
const Nd2NzParams ap{1,TM,D,0,D,TM,1,0};
DataCopy(b1,w1,bp);DataCopy(b1[N*16U],w2,bp);
SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
DataCopy(a1,x0,ap);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
LoadData2dParams bl;bl.repeatTimes=B/256U;bl.srcStride=1;
LoadData(b2,b1,bl);
SetFlag<HardEvent::MTE1_M>(EVENT_ID2);WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);
}
template<uint32_t I,uint32_t CurTM=TM> __aicore__ inline void Prefetch(const GlobalTensor<T>& x) {
constexpr event_t slot=I%2U==0U?EVENT_ID0:EVENT_ID1;
if constexpr(I>=2U) {WaitFlag<HardEvent::MTE1_MTE2>(slot);}
LocalTensor<T> a1(TPosition::A1,(I%2U)*A*sizeof(T),CurTM*D);
const Nd2NzParams ap{1,CurTM,D,0,D,CurTM,1,0};
DataCopy(a1,x,ap);SetFlag<HardEvent::MTE2_MTE1>(slot);
}
template<uint32_t I,uint32_t CurTM=TM> __aicore__ inline void Compute(const GlobalTensor<T>& dst) {
constexpr event_t slot=I%2U==0U?EVENT_ID0:EVENT_ID1;
constexpr event_t cSlot=DoubleC?slot:EVENT_ID0;
LocalTensor<T> a1(TPosition::A1,(I%2U)*A*sizeof(T),CurTM*D);
LocalTensor<T> b2(TPosition::B2,0,B);
if constexpr (DoubleA2) {
LocalTensor<T> a2(TPosition::A2,(I%2U)*A*sizeof(T),CurTM*D);
LocalTensor<float> c(TPosition::CO1,DoubleC?(I%2U)*2U*TM*N*sizeof(float):0U,2U*CurTM*N);
WaitFlag<HardEvent::MTE2_MTE1>(slot);
if constexpr(I>=2U) {WaitFlag<HardEvent::M_MTE1>(slot);}
LoadData2dParams al;al.repeatTimes=D/16U;al.srcStride=CurTM/16U;
#pragma unroll
for(uint32_t mb=0;mb<CurTM/16U;++mb) {LoadData(a2[mb*D*16U],a1[mb*256U],al);}
SetFlag<HardEvent::MTE1_MTE2>(slot);
SetFlag<HardEvent::MTE1_M>(slot);WaitFlag<HardEvent::MTE1_M>(slot);
if constexpr(DoubleC ? (I>=2U) : (I>0U)) {WaitFlag<HardEvent::FIX_M>(cSlot);}
MmadParams mad;mad.m=CurTM;mad.n=2U*N;mad.k=D;mad.cmatrixInitVal=true;
Mmad(c,a2,b2,mad);SetFlag<HardEvent::M_MTE1>(slot);
SetFlag<HardEvent::M_FIX>(cSlot);WaitFlag<HardEvent::M_FIX>(cSlot);
FixpipeParamsV220 fix;fix.nSize=2U*N;fix.mSize=CurTM;fix.srcStride=CurTM;fix.dstStride=2U*N;fix.quantPre=QuantMode_t::F322F16;
Fixpipe<T,float,CFG_ROW_MAJOR>(dst,c,fix);SetFlag<HardEvent::FIX_M>(cSlot);
} else {
constexpr uint32_t HM=CurTM/2U;
LocalTensor<T> a2_0(TPosition::A2,0U,HM*D);
LocalTensor<T> a2_1(TPosition::A2,HM*D*sizeof(T),HM*D);
LocalTensor<float> c0(TPosition::CO1,(I%2U)*2U*TM*N*sizeof(float),2U*HM*N);
LocalTensor<float> c1(TPosition::CO1,((I%2U)*2U*TM*N+2U*HM*N)*sizeof(float),2U*HM*N);
WaitFlag<HardEvent::MTE2_MTE1>(slot);
if constexpr(I>0U) {WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);}
LoadData2dParams al;al.repeatTimes=D/16U;al.srcStride=CurTM/16U;
#pragma unroll
for(uint32_t mb=0;mb<HM/16U;++mb) {LoadData(a2_0[mb*D*16U],a1[mb*256U],al);}
SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
if constexpr(I>0U) {WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);}
#pragma unroll
for(uint32_t mb=0;mb<HM/16U;++mb) {LoadData(a2_1[mb*D*16U],a1[(mb+HM/16U)*256U],al);}
SetFlag<HardEvent::MTE1_MTE2>(slot);
SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
if constexpr(I>=2U) {WaitFlag<HardEvent::FIX_M>(cSlot);}
MmadParams mad;mad.m=HM;mad.n=2U*N;mad.k=D;mad.cmatrixInitVal=true;
Mmad(c0,a2_0,b2,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
SetFlag<HardEvent::M_FIX>(EVENT_ID0);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
Mmad(c1,a2_1,b2,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
SetFlag<HardEvent::M_FIX>(EVENT_ID1);
if constexpr(I>=4U) {
constexpr uint32_t cid=(I%4U)==0?0x8:((I%4U)==1?0x9:((I%4U)==2?0xC:0xF));
CrossCoreWaitFlag(cid);
}
FixpipeParamsV220 fix;fix.nSize=2U*N;fix.mSize=HM;fix.srcStride=HM;fix.dstStride=2U*N;fix.quantPre=QuantMode_t::F322F16;
WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
Fixpipe<T,float,CFG_ROW_MAJOR>(dst,c0,fix);
WaitFlag<HardEvent::M_FIX>(EVENT_ID1);
Fixpipe<T,float,CFG_ROW_MAJOR>(dst[HM*2U*N],c1,fix);
SetFlag<HardEvent::FIX_M>(cSlot);
}
}
template<uint32_t Iterations> __aicore__ inline void Drain() {
WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
if constexpr(!DoubleA2 || Iterations>=2U) {WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);}
WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
if constexpr(DoubleC && Iterations>=2U) {WaitFlag<HardEvent::FIX_M>(EVENT_ID1);}
WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
if constexpr(Iterations>=2U) {WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);}
}
};
template <typename T,uint32_t D,uint32_t TM,uint32_t Cores,uint32_t Iterations,uint32_t TileN=64U,uint32_t TailIterations=Iterations,uint32_t FullCores=Cores>
class PipelinedCubeSwiGlu {
static constexpr uint32_t N=TileN, TN=D/N, Plane=TM*N, M=TM*(FullCores*Iterations+(Cores-FullCores)*TailIterations)/TN;
static constexpr bool FastAiv=true;
static constexpr uint32_t NR=TM/2U, Cap=NR*N;
static constexpr uint32_t InSlots=4U;
static constexpr uint32_t OutSlots=4U;
static_assert(D%N==0U && TM%32U==0U,"Exact tile shape");
static_assert(TM*D*sizeof(T)<=64U*1024U,"L0A capacity");
static_assert(2U*N*D*sizeof(T)<=64U*1024U,"L0B capacity");
static_assert(2U*TM*N*sizeof(float)<=128U*1024U,"L0C capacity");
static_assert(((2U*InSlots+3U+OutSlots)*Cap+2U*N+128U)*sizeof(T)<=192U*1024U,"UB capacity");
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y,GM_ADDR workspace) {
core_=GetBlockIdx();
if ASCEND_IS_AIV {core_/=2U;}
col_=core_%TN*N;
x_.SetGlobalBuffer((__gm__ T*)x,M*D);y_.SetGlobalBuffer((__gm__ T*)y,M*D);
w1_.SetGlobalBuffer((__gm__ T*)w1,D*D);w2_.SetGlobalBuffer((__gm__ T*)w2,D*D);
b1_.SetGlobalBuffer((__gm__ T*)b1,D);b2_.SetGlobalBuffer((__gm__ T*)b2,D);
slots_.SetGlobalBuffer((__gm__ T*)workspace,2U*Plane*4U*Cores);
SetAtomicNone();
if ASCEND_IS_AIC {
piped_.PrepareWeights(w1_[col_*D],w2_[col_*D],x_[(core_/TN)*TM*D]);
}
if ASCEND_IS_AIV {
SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
LocalTensor<T> bias(TPosition::VECIN,2U*InSlots*Cap*sizeof(T),2U*N);
DataCopy(bias,b1_[col_],N);DataCopy(bias[N],b2_[col_],N);
LocalTensor<T> ones(TPosition::VECCALC,((2U*InSlots+3U+OutSlots)*Cap+2U*N)*sizeof(T),128U);
Duplicate(ones,static_cast<T>(1),128U);
if constexpr (FastAiv) {
WaitReady<0U>();
LocalTensor<T> input0(TPosition::VECIN,0,2U*Cap);
DataCopy(input0,slots_[core_*8U*Plane+GetSubBlockIdx()*NR*2U*N],2U*Cap);
if constexpr (Iterations>4U) {Consumed<0U>();}
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
}
}
if constexpr (TailIterations==Iterations) {
Step<0,Iterations>();
if ASCEND_IS_AIC {piped_.template Drain<Iterations>();}
} else {
if (core_<FullCores) {
Step<0,Iterations>();
if ASCEND_IS_AIC {piped_.template Drain<Iterations>();}
} else {
Step<0,TailIterations>();
if ASCEND_IS_AIC {piped_.template Drain<TailIterations>();}
}
}
PipeBarrier<PIPE_ALL>();
}
private:
template <HardEvent E> __aicore__ inline void Sync() {SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0);}
template <uint32_t Slot> __aicore__ inline void Ready() {
constexpr uint32_t id=Slot==0?0x6:(Slot==1?0x7:(Slot==2?0xB:0xE));
CrossCoreSetFlag<0x2,PIPE_FIX>(id);
}
template <uint32_t Slot> __aicore__ inline void WaitReady() {
constexpr uint32_t id=Slot==0?0x6:(Slot==1?0x7:(Slot==2?0xB:0xE));
CrossCoreWaitFlag(id);
}
template <uint32_t Slot> __aicore__ inline void Consumed() {
constexpr uint32_t id=Slot==0?0x8:(Slot==1?0x9:(Slot==2?0xC:0xF));
CrossCoreSetFlag<0x2,PIPE_MTE2>(id);
}
template <uint32_t Slot> __aicore__ inline void WaitConsumed() {
constexpr uint32_t id=Slot==0?0x8:(Slot==1?0x9:(Slot==2?0xC:0xF));
CrossCoreWaitFlag(id);
}
template <uint32_t I,uint32_t CurIter> __aicore__ inline void Step() {
constexpr uint32_t Slot=I%4U;
const uint64_t row=(core_/TN+I*(Cores/TN))*TM;
const uint64_t off=(core_*4U+Slot)*2U*Plane;
if ASCEND_IS_AIC {
if constexpr (!FastAiv && I>=4U) {WaitConsumed<Slot>();}
if constexpr (I+1U<CurIter) {piped_.template Prefetch<I+1U>(x_[(row+(Cores/TN)*TM)*D]);}
piped_.template Compute<I>(slots_[off]);
Ready<Slot>();
}
if ASCEND_IS_AIV {
if constexpr (!FastAiv) {WaitReady<Slot>();}
Epilogue<I,CurIter>(off,row);
}
if constexpr (I+1U<CurIter) {Step<I+1U,CurIter>();}
}
template <uint32_t I,uint32_t CurIter> __aicore__ inline void Epilogue(uint64_t off,uint64_t row) {
constexpr uint32_t slot=I%4U;
constexpr event_t vEv=slot==0U?EVENT_ID0:(slot==1U?EVENT_ID1:(slot==2U?EVENT_ID2:EVENT_ID3));
const uint32_t begin=GetSubBlockIdx()*NR;
LocalTensor<T> input(TPosition::VECIN,0,2U*InSlots*Cap+2U*N);
auto curInput=input[slot*2U*Cap];
auto b1=input[2U*InSlots*Cap];auto b2=input[2U*InSlots*Cap+N];
LocalTensor<T> work(TPosition::VECOUT,(2U*InSlots*Cap+2U*N)*sizeof(T),(3U+OutSlots)*Cap);
auto g=work;auto u=work[Cap];auto sig=work[2U*Cap];auto out=work[(3U+slot)*Cap];
const uint64_t start=off+begin*2U*N;
if constexpr (!FastAiv) {
if constexpr (I>=4U) {
WaitFlag<HardEvent::V_MTE2>(vEv);
}
DataCopy(curInput,slots_[start],2U*Cap);
if constexpr (I+4U<CurIter) {Consumed<slot>();}
Sync<HardEvent::MTE2_V>();
} else {
WaitFlag<HardEvent::MTE2_V>(vEv);
}
const BinaryRepeatParams rp{1,1,1,N/16U,2U*N/16U,0};
constexpr uint64_t mask=0;
if constexpr (N!=128U) {SetVectorMask<T,MaskMode::NORMAL>(0,~0ULL);}
Add<T,false>(g,curInput,b1,mask,static_cast<uint8_t>(NR),rp);
Add<T,false>(u,curInput[N],b2,mask,static_cast<uint8_t>(NR),rp);
if constexpr (I+4U<CurIter) {
SetFlag<HardEvent::V_MTE2>(vEv);
}
static_assert(Cap%128U==0U && Cap/128U<=255U);
LocalTensor<T> ones(TPosition::VECCALC,((2U*InSlots+3U+OutSlots)*Cap+2U*N)*sizeof(T),128U);
constexpr uint8_t repeats=Cap/128U;
const UnaryRepeatParams ur{1,1,8,8};
const BinaryRepeatParams br{1,1,1,8,8,8}, broadcast{1,1,1,8,0,8};
if constexpr (N!=128U) {SetVectorMask<T,MaskMode::NORMAL>(~0ULL,~0ULL);}
Muls<T,false>(sig,g,static_cast<T>(-1),mask,repeats,ur);
Exp<T,false>(sig,sig,mask,repeats,ur);
Adds<T,false>(sig,sig,static_cast<T>(1),mask,repeats,ur);
Div<T,false>(sig,ones,sig,mask,repeats,broadcast);
if constexpr (I>=4U) {
WaitFlag<HardEvent::MTE3_V>(vEv);
}
Mul<T,false>(out,g,sig,mask,repeats,br);
Mul<T,false>(out,out,u,mask,repeats,br);
if constexpr (!FastAiv) {
Sync<HardEvent::V_MTE3>();
DataCopyParams op{NR,N/16U,0,(D-N)/16U};
if constexpr (D==N) {DataCopy(y_[(row+begin)*D],out,Cap);}
else {DataCopy(y_[(row+begin)*D+col_],out,op);}
if constexpr (I+4U<CurIter) {
SetFlag<HardEvent::MTE3_V>(vEv);
}
} else {
SetFlag<HardEvent::V_MTE3>(vEv);
if constexpr (I+1U<CurIter) {
constexpr uint32_t nextSlot=(I+1U)%4U;
constexpr event_t nextEv=nextSlot==0U?EVENT_ID0:(nextSlot==1U?EVENT_ID1:(nextSlot==2U?EVENT_ID2:EVENT_ID3));
WaitReady<nextSlot>();
if constexpr (I+1U>=4U) {
WaitFlag<HardEvent::V_MTE2>(nextEv);
}
const uint64_t nextOff=(core_*4U+nextSlot)*2U*Plane;
DataCopy(input[nextSlot*2U*Cap],slots_[nextOff+begin*2U*N],2U*Cap);
if constexpr (I+5U<CurIter) {Consumed<nextSlot>();}
SetFlag<HardEvent::MTE2_V>(nextEv);
}
WaitFlag<HardEvent::V_MTE3>(vEv);
DataCopyParams op{NR,N/16U,0,(D-N)/16U};
if constexpr (D==N) {DataCopy(y_[(row+begin)*D],out,Cap);}
else {DataCopy(y_[(row+begin)*D+col_],out,op);}
if constexpr (I+4U<CurIter) {
SetFlag<HardEvent::MTE3_V>(vEv);
}
}
}
DoubleBufferedPairCube<T,TM,D,N> piped_;
GlobalTensor<T> x_,w1_,w2_,b1_,b2_,y_,slots_;
uint32_t core_=0,col_=0;
};
}
template <typename DT_INPUT_X, int PATH>
__global__ __aicore__ void swi_glu(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2, GM_ADDR bias_2,
GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
if constexpr (PATH>=43) {
if constexpr (PATH==43) {
KERNEL_TASK_TYPE(11009,KERNEL_TYPE_AIV_ONLY);
SubBlockGemv4<DT_INPUT_X> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output);
} else if constexpr (PATH==44) {
KERNEL_TASK_TYPE(11265,KERNEL_TYPE_AIV_ONLY);
AlignedColGemv<DT_INPUT_X,64U,2U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output);
} else if constexpr (PATH==45) {
optiling::PipelinedCubeSwiGlu<DT_INPUT_X,256U,32U,16U,4U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,workspace);
} else if constexpr (PATH==46) {
KERNEL_TASK_TYPE(11777,KERNEL_TYPE_AIV_ONLY);
AlignedColGemv<DT_INPUT_X,32U,1U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output);
} else if constexpr (PATH==47) {
optiling::PipelinedCubeSwiGlu<DT_INPUT_X,128U,64U,16U,4U,128U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,workspace);
} else if constexpr (PATH==48) {
KERNEL_TASK_TYPE(12289,KERNEL_TYPE_AIV_ONLY);
ScalarChannelSwiGlu<DT_INPUT_X> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output);
} else if constexpr (PATH==49) {
KERNEL_TASK_TYPE(12545,KERNEL_TYPE_AIV_ONLY);
PackedSubBlockGemv<DT_INPUT_X> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output);
} else if constexpr (PATH==50) {
KERNEL_TASK_TYPE(12801,KERNEL_TYPE_AIV_ONLY);
PackedMultiBlockGemv<DT_INPUT_X> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,tiling);
} else if constexpr (PATH==51) {
optiling::PipelinedCubeSwiGlu<DT_INPUT_X,256U,128U,16U,16U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,workspace);
}
} else if constexpr (PATH==42) {
KERNEL_TASK_TYPE(10753,KERNEL_TYPE_AIV_ONLY);
REGISTER_TILING_DEFAULT(SwiGluTilingData);
BatchedTinyVectorSwiGlu<DT_INPUT_X,13U,7U,true> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,nullptr);
} else if constexpr (PATH >= 38) {
if constexpr (PATH==38) { KERNEL_TASK_TYPE(9729,KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH==39) { KERNEL_TASK_TYPE(9985,KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH==40) { KERNEL_TASK_TYPE(10241,KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH==41) { KERNEL_TASK_TYPE(10497,KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
if constexpr (PATH==41) {
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData,tiling_data,tiling);
TinyColumnsSwiGlu<DT_INPUT_X,0U,false> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else {
TinyColumnsSwiGlu<DT_INPUT_X,PATH==39?0U:13U,PATH!=40> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,nullptr);
}
} else if constexpr (PATH >= 36) {
if constexpr (PATH==36) { KERNEL_TASK_TYPE(9217,KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH==37) { KERNEL_TASK_TYPE(9473,KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData,tiling_data,tiling);
SmallMVectorSwiGlu<DT_INPUT_X,PATH==36?512U:0U,512U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else if constexpr (PATH >= 32) {
if (workspace == nullptr) { return; }
GM_ADDR userWorkspace=AscendC::GetUserWorkspace(workspace);
if (userWorkspace == nullptr) { return; }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluDirectTilingData,tiling_data,tiling);
constexpr uint32_t fixedD=PATH==32?256U:(PATH==33?128U:0U);
optiling::SlimDirectCube<DT_INPUT_X,64U,fixedD,PATH==35> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,userWorkspace,&tiling_data);
} else if constexpr (PATH >= 26) {
if (workspace == nullptr) { return; }
GM_ADDR userWorkspace=AscendC::GetUserWorkspace(workspace);
if (userWorkspace == nullptr) { return; }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluDirectTilingData,tiling_data,tiling);
constexpr uint32_t fixedD=(PATH==26||PATH==30)?256U:((PATH==27||PATH==31)?128U:0U);
optiling::SlimDirectCube<DT_INPUT_X,128U,fixedD,PATH==29,true,PATH>=30> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,userWorkspace,&tiling_data);
} else if constexpr (PATH >= 23) {
REGISTER_TILING_DEFAULT(SwiGluTilingData);
if constexpr (PATH == 23) { KERNEL_TASK_TYPE(5889, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 24) { KERNEL_TASK_TYPE(6145, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 25) { KERNEL_TASK_TYPE(6401, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 25) {
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData, tiling_data, tiling);
SingleTinySwiGlu<DT_INPUT_X,0U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else {
SingleTinySwiGlu<DT_INPUT_X,PATH==23?13U:1U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,nullptr);
}
} else if constexpr (PATH == 21 || PATH == 22) {
if constexpr (PATH == 21) { KERNEL_TASK_TYPE(5377, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 22) { KERNEL_TASK_TYPE(5633, KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData, tiling_data, tiling);
BatchedTinyVectorSwiGlu<DT_INPUT_X,PATH==21?13U:0U,4U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else if constexpr (PATH >= 17) {
if constexpr (PATH == 17) { KERNEL_TASK_TYPE(4353, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 18) { KERNEL_TASK_TYPE(4609, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 19) { KERNEL_TASK_TYPE(4865, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 20) { KERNEL_TASK_TYPE(5121, KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData, tiling_data, tiling);
constexpr uint32_t fixedD=PATH==17?64U:(PATH==18?128U:(PATH==19?256U:0U));
SmallMVectorSwiGlu<DT_INPUT_X,fixedD> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else if constexpr (PATH >= 14) {
if constexpr (PATH == 14) { KERNEL_TASK_TYPE(3585, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 15) { KERNEL_TASK_TYPE(3841, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 16) { KERNEL_TASK_TYPE(4097, KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData, tiling_data, tiling);
if constexpr (PATH == 15) {
ScalarFeatureSwiGlu<DT_INPUT_X> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else {
TinyVectorSwiGlu<DT_INPUT_X,PATH == 14 ? 13U : 0U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
}
} else if constexpr (PATH == 0 || PATH == 4 || PATH == 5) {
if constexpr (PATH == 0) { KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 4) { KERNEL_TASK_TYPE(1025, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 5) { KERNEL_TASK_TYPE(1281, KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tiling_data, tiling);
constexpr int mode = PATH == 4 ? 1 : (PATH == 5 ? 2 : 0);
KernelSwiGlu<DT_INPUT_X, mode> op;
op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, tiling_data);
op.Process();
} else {
if (workspace == nullptr) { return; }
GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
if (userWorkspace == nullptr) { return; }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
if constexpr (PATH == 2 || PATH == 3 || PATH >= 8) {
GET_TILING_DATA_WITH_STRUCT(SwiGluDirectTilingData, tiling_data, tiling);
constexpr uint32_t tileM = (PATH == 3 || PATH == 9 || PATH == 11 || PATH == 13) ? 128U : 32U;
constexpr uint32_t fixedD = (PATH == 8 || PATH == 9) ? 256U :
((PATH == 10 || PATH == 11) ? 128U : 0U);
optiling::SlimDirectCube<DT_INPUT_X, tileM, fixedD, PATH == 12 || PATH == 13> op;
op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output, userWorkspace, &tiling_data);
} else {
AscendC::TPipe tPipe;
GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tiling_data, tiling);
optiling::RunCube<DT_INPUT_X, PATH == 6 || PATH == 7, PATH == 7>(
input_x, weight_1, bias_1, weight_2, bias_2, output, userWorkspace, &tPipe, &tiling_data);
tPipe.Destroy();
}
}
}
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 11009UL || TILING_KEY_VAR == 11265UL || TILING_KEY_VAR == 11521UL || TILING_KEY_VAR == 11777UL || TILING_KEY_VAR == 12033UL || TILING_KEY_VAR == 12289UL || TILING_KEY_VAR == 12545UL || TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL) && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON)
namespace AscendC {
__aicore__ inline void NoSysWorkspaceForce(GM_ADDR) {}
__aicore__ inline GM_ADDR FastGetUserWorkspace(GM_ADDR workspace) {
return workspace + RESERVED_WORKSPACE;
}
}
#define SetSysWorkspaceForce NoSysWorkspaceForce
#define GetUserWorkspace FastGetUserWorkspace
#define cce_kernel cce_kernel, aligned(512), section(".text.hot")
#if TILING_KEY_VAR == 11009UL || TILING_KEY_VAR == 11265UL || TILING_KEY_VAR == 11777UL || TILING_KEY_VAR == 12289UL || TILING_KEY_VAR == 12545UL || TILING_KEY_VAR == 12801UL
#define set_ffts_base_addr(...) ((void)0)
#endif
#endif
#elif defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12033UL)
#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "swi_glu_tiling.h"
#include "tiling_key_swi_glu.h"
using namespace AscendC;
namespace {
__aicore__ inline uint32_t AlignUpBytes(uint32_t bytes)
{
return (bytes + 31U) / 32U * 32U;
}
__aicore__ inline uint32_t AlignUp16(uint32_t elems)
{
return (elems + 15U) / 16U * 16U;
}
constexpr uint32_t kChunk16 = 16;
__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
return (a + b - 1) / b;
}
}
template <typename T, bool FillOnes = false>
__aicore__ inline void GateCounter(const LocalTensor<T>& g,const LocalTensor<T>& u,
const LocalTensor<T>& sig,const LocalTensor<T>& out,
const LocalTensor<T>& ones,uint32_t count) {
constexpr uint64_t mask=0;
const UnaryRepeatParams ur{1,1,8,8};
const BinaryRepeatParams br{1,1,1,8,8,8};
SetMaskCount();SetVectorMask<T,MaskMode::COUNTER>(0,count);

Muls<T,false>(sig,g,static_cast<T>(-1),mask,1,ur);
PipeBarrier<PIPE_V>();Exp<T,false>(sig,sig,mask,1,ur);
PipeBarrier<PIPE_V>();Adds<T,false>(sig,sig,static_cast<T>(1),mask,1,ur);
PipeBarrier<PIPE_V>();Reciprocal<T,false>(sig,sig,mask,1,ur);
PipeBarrier<PIPE_V>();Mul<T,false>(out,g,sig,mask,1,br);
PipeBarrier<PIPE_V>();Mul<T,false>(out,out,u,mask,1,br);
SetMaskNorm();ResetMask();
}
template <typename T,uint32_t Count>
__aicore__ inline void GateNormal(const LocalTensor<T>& g,const LocalTensor<T>& u,
const LocalTensor<T>& sig,const LocalTensor<T>& out,const LocalTensor<T>& ones) {
static_assert(Count<=128U);
constexpr uint64_t lo=Count>=64U?~0ULL:((1ULL<<Count)-1ULL);
constexpr uint64_t hi=Count<=64U?0ULL:((1ULL<<(Count-64U))-1ULL);
const UnaryRepeatParams ur{1,1,8,8};const BinaryRepeatParams br{1,1,1,8,8,8};
SetVectorMask<T,MaskMode::NORMAL>(hi,lo);
Muls<T,false>(sig,g,static_cast<T>(-1),static_cast<uint64_t>(0),1,ur);
PipeBarrier<PIPE_V>();Exp<T,false>(sig,sig,static_cast<uint64_t>(0),1,ur);
PipeBarrier<PIPE_V>();Adds<T,false>(sig,sig,static_cast<T>(1),static_cast<uint64_t>(0),1,ur);
PipeBarrier<PIPE_V>();Div<T,false>(sig,ones,sig,static_cast<uint64_t>(0),1,br);
PipeBarrier<PIPE_V>();Mul<T,false>(out,g,sig,static_cast<uint64_t>(0),1,br);
PipeBarrier<PIPE_V>();Mul<T,false>(out,out,u,static_cast<uint64_t>(0),1,br);
ResetMask();
}
template <typename T, uint32_t Count, bool SetMask = true>
__aicore__ inline void FastGate(const LocalTensor<T>& g, const LocalTensor<T>& u,
const LocalTensor<T>& sig, const LocalTensor<T>& out, const LocalTensor<T>& ones) {
static_assert(Count <= 128U);
constexpr uint64_t lo = Count >= 64U ? ~0ULL : ((1ULL << Count) - 1ULL);
constexpr uint64_t hi = Count <= 64U ? 0ULL : (Count >= 128U ? ~0ULL : ((1ULL << (Count > 64U ? Count - 64U : 0U)) - 1ULL));
const UnaryRepeatParams ur{1, 1, 8, 8};
const BinaryRepeatParams br{1, 1, 1, 8, 8, 8};
if constexpr (SetMask) {
SetVectorMask<T, MaskMode::NORMAL>(hi, lo);
}
Muls<T, false>(sig, g, static_cast<T>(-1), static_cast<uint64_t>(0), 1, ur);
Exp<T, false>(sig, sig, static_cast<uint64_t>(0), 1, ur);
Adds<T, false>(sig, sig, static_cast<T>(1), static_cast<uint64_t>(0), 1, ur);
Div<T, false>(sig, ones, sig, static_cast<uint64_t>(0), 1, br);
Mul<T, false>(out, g, sig, static_cast<uint64_t>(0), 1, br);
Mul<T, false>(out, out, u, static_cast<uint64_t>(0), 1, br);
}
template <class DT_INPUT_X, int Mode>
class KernelSwiGlu {
public:
__aicore__ inline KernelSwiGlu() {}
__aicore__ inline void Init(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2, GM_ADDR bias_2,
GM_ADDR output, const SwiGluTilingData &tiling)
{
td_ = &tiling;
const uint64_t xElems = td_->m * td_->d;
const uint64_t wElems = td_->d * td_->d;
xGm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)input_x, xElems);
w1Gm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)weight_1, wElems);
b1Gm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)bias_1, td_->d);
w2Gm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)weight_2, wElems);
b2Gm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)bias_2, td_->d);
outGm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)output, xElems);
pipe_.InitBuffer(stageA_, 1, AlignUpBytes(td_->dAlign * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(stageB_, 1, AlignUpBytes(td_->dAlign * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(outQueue_, 1, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(xfBuf_, AlignUpBytes(td_->dAlign * sizeof(float)));
pipe_.InitBuffer(wfBuf_, AlignUpBytes(td_->dAlign * sizeof(float)));
const uint32_t productElems = td_->dAlign *
(Mode != 0 && td_->dAlign <= 256 ? kChunk16 : 1U);
pipe_.InitBuffer(prodGBuf_, AlignUpBytes(productElems * sizeof(float)));
pipe_.InitBuffer(prodUBuf_, AlignUpBytes(productElems * sizeof(float)));
pipe_.InitBuffer(accGBuf_, AlignUpBytes(td_->tileN * sizeof(float)));
pipe_.InitBuffer(accUBuf_, AlignUpBytes(td_->tileN * sizeof(float)));
pipe_.InitBuffer(accG16Buf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(accU16Buf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(negBuf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(eBuf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(sBuf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(onesBuf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
if constexpr (Mode != 0) {
pipe_.InitBuffer(w1tBuf_, AlignUpBytes(td_->dAlign * kChunk16 * sizeof(float)));
pipe_.InitBuffer(w2tBuf_, AlignUpBytes(td_->dAlign * kChunk16 * sizeof(float)));
pipe_.InitBuffer(wblkQueue_, 1, AlignUpBytes(kChunk16 * td_->dAlign * sizeof(DT_INPUT_X)));
}
}
__aicore__ inline void Process()
{
const uint64_t core = GetBlockIdx();
if (core >= td_->usedCoreNum) {
return;
}
if constexpr (Mode == 2) {
const uint32_t d = static_cast<uint32_t>(td_->d);
const uint32_t columns = Ceiling(d, kChunk16);
const uint64_t tasks = td_->m * columns;
for (uint64_t task = core; task < tasks; task += td_->usedCoreNum) {
const uint64_t row = task / columns;
const uint32_t c0 = static_cast<uint32_t>(task % columns) * kChunk16;
const uint32_t chunk = d - c0 < kChunk16 ? d - c0 : kChunk16;
PrepareWt(c0, chunk, d);
ProcessColTileT(row, c0, chunk, d);
}
} else if constexpr (Mode == 1) {
const uint32_t d = static_cast<uint32_t>(td_->d);
for (uint32_t c0 = 0; c0 < d; c0 += kChunk16) {
const uint32_t chunk = d - c0 < kChunk16 ? d - c0 : kChunk16;
PrepareWt(c0, chunk, d);
for (uint64_t row = core; row < td_->m; row += td_->usedCoreNum) {
ProcessColTileT(row, c0, chunk, d);
}
}
} else {
for (uint64_t row = core; row < td_->m; row += td_->usedCoreNum) {
ProcessRow(row);
}
}
}
private:
template <HardEvent event>
__aicore__ inline void SyncPipe()
{
const event_t id = static_cast<event_t>(pipe_.FetchEventID(event));
SetFlag<event>(id);
WaitFlag<event>(id);
}
__aicore__ inline void ProcessRow(uint64_t row)
{
const uint32_t d = static_cast<uint32_t>(td_->d);
const uint32_t dAlign = td_->dAlign;
const uint32_t tileN = td_->tileN;
LocalTensor<DT_INPUT_X> xTile = stageA_.AllocTensor<DT_INPUT_X>();
DataCopyExtParams xCopy{1, static_cast<uint32_t>(d * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> xPad{true, 0, static_cast<uint8_t>(dAlign - d),
static_cast<DT_INPUT_X>(0)};
DataCopyPad(xTile, xGm_[row * d], xCopy, xPad);
stageA_.EnQue(xTile);
xTile = stageA_.DeQue<DT_INPUT_X>();
LocalTensor<float> xf = xfBuf_.Get<float>();
Cast(xf, xTile, RoundMode::CAST_NONE, dAlign);
stageA_.FreeTensor(xTile);
for (uint32_t c0 = 0; c0 < d; c0 += tileN) {
const uint32_t chunk = d - c0 < tileN ? d - c0 : tileN;
ProcessColTile(row, c0, chunk, d, dAlign, xf);
}
}
__aicore__ inline void ProcessColTile(uint64_t row, uint32_t c0, uint32_t chunk, uint32_t d, uint32_t dAlign,
const LocalTensor<float> &xf)
{
LocalTensor<float> wf = wfBuf_.Get<float>();
LocalTensor<float> prodG = prodGBuf_.Get<float>();
LocalTensor<float> prodU = prodUBuf_.Get<float>();
LocalTensor<float> accG = accGBuf_.Get<float>();
LocalTensor<float> accU = accUBuf_.Get<float>();
LocalTensor<DT_INPUT_X> accG16 = accG16Buf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> accU16 = accU16Buf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> negBuf = negBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> eBuf = eBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> sBuf = sBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> onesBuf = onesBuf_.Get<DT_INPUT_X>();
DataCopyExtParams wCopy{1, static_cast<uint32_t>(d * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> wPad{true, 0, static_cast<uint8_t>(dAlign - d),
static_cast<DT_INPUT_X>(0)};
for (uint32_t j = 0; j < chunk; ++j) {
const uint64_t wBase = static_cast<uint64_t>(c0 + j) * d;
LocalTensor<DT_INPUT_X> w1 = stageA_.AllocTensor<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> w2 = stageB_.AllocTensor<DT_INPUT_X>();
DataCopyPad(w1, w1Gm_[wBase], wCopy, wPad);
DataCopyPad(w2, w2Gm_[wBase], wCopy, wPad);
stageA_.EnQue(w1);
stageB_.EnQue(w2);
w1 = stageA_.DeQue<DT_INPUT_X>();
w2 = stageB_.DeQue<DT_INPUT_X>();
Cast(wf, w1, RoundMode::CAST_NONE, dAlign);
Mul(prodG, xf, wf, dAlign);
Cast(wf, w2, RoundMode::CAST_NONE, dAlign);
Mul(prodU, xf, wf, dAlign);
stageA_.FreeTensor(w1);
stageB_.FreeTensor(w2);
ReduceSum(accG[j], prodG, wf, dAlign);
ReduceSum(accU[j], prodU, wf, dAlign);
}
DataCopyExtParams bCopy{1, static_cast<uint32_t>(chunk * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> bPad{true, 0, static_cast<uint8_t>(AlignUp16(chunk) - chunk),
static_cast<DT_INPUT_X>(0)};
LocalTensor<DT_INPUT_X> b1Tile = stageA_.AllocTensor<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> b2Tile = stageB_.AllocTensor<DT_INPUT_X>();
DataCopyPad(b1Tile, b1Gm_[c0], bCopy, bPad);
DataCopyPad(b2Tile, b2Gm_[c0], bCopy, bPad);
stageA_.EnQue(b1Tile);
stageB_.EnQue(b2Tile);
b1Tile = stageA_.DeQue<DT_INPUT_X>();
b2Tile = stageB_.DeQue<DT_INPUT_X>();
Cast(accG16, accG, RoundMode::CAST_RINT, chunk);
Cast(accU16, accU, RoundMode::CAST_RINT, chunk);
Add(accG16, accG16, b1Tile, chunk);
stageA_.FreeTensor(b1Tile);
Add(accU16, accU16, b2Tile, chunk);
stageB_.FreeTensor(b2Tile);
Duplicate(onesBuf, static_cast<DT_INPUT_X>(1.0f), chunk);
Muls(negBuf, accG16, (DT_INPUT_X)(-1.0), chunk);
Exp(eBuf, negBuf, chunk);
Adds(eBuf, eBuf, (DT_INPUT_X)(1.0), chunk);
Div(sBuf, onesBuf, eBuf, chunk);
Mul(accG16, accG16, sBuf, chunk);
LocalTensor<DT_INPUT_X> outTile = outQueue_.AllocTensor<DT_INPUT_X>();
Mul(outTile, accG16, accU16, chunk);
outQueue_.EnQue(outTile);
outTile = outQueue_.DeQue<DT_INPUT_X>();
DataCopyExtParams oCopy{1, static_cast<uint32_t>(chunk * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPad(outGm_[row * d + c0], outTile, oCopy);
outQueue_.FreeTensor(outTile);
}
__aicore__ inline void PrepareOneWeight(const GlobalTensor<DT_INPUT_X> &weight,
const LocalTensor<float> &cached,
uint32_t c0, uint32_t chunk, uint32_t d)
{
LocalTensor<DT_INPUT_X> wblk = wblkQueue_.AllocTensor<DT_INPUT_X>();
DataCopyExtParams blkCopy{static_cast<uint16_t>(chunk),
static_cast<uint32_t>(d * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> blkPad{true, 0, static_cast<uint8_t>(td_->dAlign - d),
static_cast<DT_INPUT_X>(0)};
DataCopyPad(wblk, weight[static_cast<uint64_t>(c0) * d], blkCopy, blkPad);
wblkQueue_.EnQue(wblk);
wblk = wblkQueue_.DeQue<DT_INPUT_X>();
Cast(cached, wblk, RoundMode::CAST_NONE, chunk * td_->dAlign);
wblkQueue_.FreeTensor(wblk);
}
__aicore__ inline void PrepareWt(uint32_t c0, uint32_t chunk, uint32_t d)
{
PrepareOneWeight(w1Gm_, w1tBuf_.Get<float>(), c0, chunk, d);
PrepareOneWeight(w2Gm_, w2tBuf_.Get<float>(), c0, chunk, d);
}
__aicore__ inline void ProcessColTileT(uint64_t row, uint32_t c0, uint32_t chunk, uint32_t d)
{
const uint32_t dAlign = td_->dAlign;
LocalTensor<float> xf = xfBuf_.Get<float>();
LocalTensor<float> wf = wfBuf_.Get<float>();
LocalTensor<float> accG = accGBuf_.Get<float>();
LocalTensor<float> accU = accUBuf_.Get<float>();
LocalTensor<DT_INPUT_X> accG16 = accG16Buf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> accU16 = accU16Buf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> negBuf = negBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> eBuf = eBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> sBuf = sBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> onesBuf = onesBuf_.Get<DT_INPUT_X>();
LocalTensor<float> w1t = w1tBuf_.Get<float>();
LocalTensor<float> w2t = w2tBuf_.Get<float>();
LocalTensor<float> prodG = prodGBuf_.Get<float>();
LocalTensor<float> prodU = prodUBuf_.Get<float>();
LocalTensor<DT_INPUT_X> xTile = stageA_.AllocTensor<DT_INPUT_X>();
DataCopyExtParams xCopy{1, static_cast<uint32_t>(d * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> xPad{true, 0, static_cast<uint8_t>(dAlign - d),
static_cast<DT_INPUT_X>(0)};
DataCopyPad(xTile, xGm_[row * d], xCopy, xPad);
stageA_.EnQue(xTile);
xTile = stageA_.DeQue<DT_INPUT_X>();
Cast(xf, xTile, RoundMode::CAST_NONE, dAlign);
stageA_.FreeTensor(xTile);
LocalTensor<DT_INPUT_X> b1Tile = stageA_.AllocTensor<DT_INPUT_X>();
DataCopyExtParams bCopy{1, static_cast<uint32_t>(chunk * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> bPad{true, 0, static_cast<uint8_t>(AlignUp16(chunk) - chunk),
static_cast<DT_INPUT_X>(0)};
DataCopyPad(b1Tile, b1Gm_[c0], bCopy, bPad);
stageA_.EnQue(b1Tile);
b1Tile = stageA_.DeQue<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> b2Tile = stageB_.AllocTensor<DT_INPUT_X>();
DataCopyPad(b2Tile, b2Gm_[c0], bCopy, bPad);
stageB_.EnQue(b2Tile);
b2Tile = stageB_.DeQue<DT_INPUT_X>();
if (dAlign <= 256U) {
const uint8_t rowStride = static_cast<uint8_t>(dAlign / 8U);
const BinaryRepeatParams mulParams{1, 1, 1, rowStride, 0, rowStride};
for (uint32_t k = 0; k < dAlign; k += 64U) {
const uint32_t width = dAlign - k < 64U ? dAlign - k : 64U;
Mul(prodG[k], xf[k], w1t[k], static_cast<uint64_t>(width),
static_cast<uint8_t>(chunk), mulParams);
Mul(prodU[k], xf[k], w2t[k], static_cast<uint64_t>(width),
static_cast<uint8_t>(chunk), mulParams);
if (k == 0) {
WholeReduceSum(accG, prodG[k], width, chunk, 1, 1, rowStride);
WholeReduceSum(accU, prodU[k], width, chunk, 1, 1, rowStride);
} else {
WholeReduceSum(wf, prodG[k], width, chunk, 1, 1, rowStride);
Add(accG, accG, wf, chunk);
WholeReduceSum(wf, prodU[k], width, chunk, 1, 1, rowStride);
Add(accU, accU, wf, chunk);
}
}
} else {
for (uint32_t j = 0; j < chunk; ++j) {
Mul(prodG, xf, w1t[j * dAlign], dAlign);
Mul(prodU, xf, w2t[j * dAlign], dAlign);
ReduceSum(accG[j], prodG, wf, dAlign);
ReduceSum(accU[j], prodU, wf, dAlign);
}
}
Cast(accG16, accG, RoundMode::CAST_RINT, chunk);
Cast(accU16, accU, RoundMode::CAST_RINT, chunk);
Add(accG16, accG16, b1Tile, chunk);
stageA_.FreeTensor(b1Tile);
Add(accU16, accU16, b2Tile, chunk);
stageB_.FreeTensor(b2Tile);
Duplicate(onesBuf, static_cast<DT_INPUT_X>(1.0f), chunk);
Muls(negBuf, accG16, (DT_INPUT_X)(-1.0), chunk);
Exp(eBuf, negBuf, chunk);
Adds(eBuf, eBuf, (DT_INPUT_X)(1.0), chunk);
Div(sBuf, onesBuf, eBuf, chunk);
Mul(accG16, accG16, sBuf, chunk);
LocalTensor<DT_INPUT_X> outTile = outQueue_.AllocTensor<DT_INPUT_X>();
Mul(outTile, accG16, accU16, chunk);
outQueue_.EnQue(outTile);
outTile = outQueue_.DeQue<DT_INPUT_X>();
DataCopyExtParams oCopy{1, static_cast<uint32_t>(chunk * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPad(outGm_[row * d + c0], outTile, oCopy);
outQueue_.FreeTensor(outTile);
}
private:
TPipe pipe_;
TQue<TPosition::VECIN, 1> stageA_;
TQue<TPosition::VECIN, 1> stageB_;
TQue<TPosition::VECOUT, 1> outQueue_;
TBuf<TPosition::VECCALC> xfBuf_;
TBuf<TPosition::VECCALC> wfBuf_;
TBuf<TPosition::VECCALC> prodGBuf_;
TBuf<TPosition::VECCALC> prodUBuf_;
TBuf<TPosition::VECCALC> accGBuf_;
TBuf<TPosition::VECCALC> accUBuf_;
TBuf<TPosition::VECCALC> accG16Buf_;
TBuf<TPosition::VECCALC> accU16Buf_;
TBuf<TPosition::VECCALC> negBuf_;
TBuf<TPosition::VECCALC> eBuf_;
TBuf<TPosition::VECCALC> sBuf_;
TBuf<TPosition::VECCALC> onesBuf_;
TBuf<TPosition::VECCALC> w1tBuf_;
TBuf<TPosition::VECCALC> w2tBuf_;
TQue<QuePosition::VECIN, 1> wblkQueue_;
GlobalTensor<DT_INPUT_X> xGm_;
GlobalTensor<DT_INPUT_X> w1Gm_;
GlobalTensor<DT_INPUT_X> b1Gm_;
GlobalTensor<DT_INPUT_X> w2Gm_;
GlobalTensor<DT_INPUT_X> b2Gm_;
GlobalTensor<DT_INPUT_X> outGm_;
const SwiGluTilingData *td_ = nullptr;
};
template <typename T, uint32_t FixedD>
class TinyVectorSwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
GM_ADDR y, const SwiGluTinyTilingData *td) {
const uint32_t d = FixedD != 0 ? FixedD : static_cast<uint32_t>(td->d);
GlobalTensor<T> gx, gw1, gw2, gb1, gb2, gy;
gx.SetGlobalBuffer((__gm__ T*)x,td->m*d); gy.SetGlobalBuffer((__gm__ T*)y,td->m*d);
gw1.SetGlobalBuffer((__gm__ T*)w1,d*d); gw2.SetGlobalBuffer((__gm__ T*)w2,d*d);
gb1.SetGlobalBuffer((__gm__ T*)b1,d); gb2.SetGlobalBuffer((__gm__ T*)b2,d);
LocalTensor<T> hw1(TPosition::VECIN,0,256), hw2(TPosition::VECIN,512,256);
LocalTensor<T> hx(TPosition::VECIN,1024,16), hb1(TPosition::VECIN,1056,16), hb2(TPosition::VECIN,1088,16);
LocalTensor<float> fw1(TPosition::VECCALC,1152,256), fw2(TPosition::VECCALC,2176,256);
LocalTensor<float> fx(TPosition::VECCALC,3200,16), pg(TPosition::VECCALC,3264,256), pu(TPosition::VECCALC,4288,256);
LocalTensor<float> ag(TPosition::VECCALC,5312,16), au(TPosition::VECCALC,5376,16);
LocalTensor<T> g(TPosition::VECCALC,5440,16), u(TPosition::VECCALC,5472,16);
LocalTensor<T> sig(TPosition::VECCALC,5504,16), ones(TPosition::VECCALC,5536,16), out(TPosition::VECOUT,5568,16);
SetAtomicNone(); SetMaskNorm(); SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams wc{static_cast<uint16_t>(d),d*2U,0,0,0};
DataCopyExtParams rc{1,d*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(16U-d),static_cast<T>(0)};
DataCopyPad(hw1,gw1,wc,pad); DataCopyPad(hw2,gw2,wc,pad);
DataCopyPad(hb1,gb1,rc,pad); DataCopyPad(hb2,gb2,rc,pad);
Sync<HardEvent::MTE2_V>();
Cast(fw1,hw1,RoundMode::CAST_NONE,d*16U); Cast(fw2,hw2,RoundMode::CAST_NONE,d*16U);
Duplicate(ones,static_cast<T>(1),16);
const BinaryRepeatParams rp{1,1,1,2,0,2};
for (uint64_t row=GetBlockIdx();row<td->m;row+=GetBlockNum()) {
DataCopyPad(hx,gx[row*d],rc,pad); Sync<HardEvent::MTE2_V>();
Cast(fx,hx,RoundMode::CAST_NONE,16); PipeBarrier<PIPE_V>();
Mul(pg,fx,fw1,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
Mul(pu,fx,fw2,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
PipeBarrier<PIPE_V>();
WholeReduceSum(ag,pg,16,d,1,1,2); WholeReduceSum(au,pu,16,d,1,1,2);
PipeBarrier<PIPE_V>();
Cast(g,ag,RoundMode::CAST_RINT,d); Cast(u,au,RoundMode::CAST_RINT,d);
PipeBarrier<PIPE_V>(); Add(g,g,hb1,d); Add(u,u,hb2,d);
PipeBarrier<PIPE_V>(); GateCounter<T>(g,u,sig,out,ones,d);
Sync<HardEvent::V_MTE3>(); DataCopyPad(gy[row*d],out,rc);
Sync<HardEvent::MTE3_MTE2>();
}
}
private:
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
};
template <typename T, uint32_t FixedD, uint32_t Rows = 1, bool SingleBatch = false>
class BatchedTinyVectorSwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
GM_ADDR y, const SwiGluTinyTilingData *td) {
const uint32_t d=FixedD!=0?FixedD:static_cast<uint32_t>(td->d);
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gy.SetGlobalBuffer((__gm__ T*)y);
gw1.SetGlobalBuffer((__gm__ T*)w1,d*d);gw2.SetGlobalBuffer((__gm__ T*)w2,d*d);
gb1.SetGlobalBuffer((__gm__ T*)b1,d);gb2.SetGlobalBuffer((__gm__ T*)b2,d);
LocalTensor<T> hw1(TPosition::VECIN,0,256),hw2(TPosition::VECIN,512,256);
LocalTensor<T> hx(TPosition::VECIN,1024,Rows*16),hb1(TPosition::VECIN,1024+Rows*32,16),hb2(TPosition::VECIN,1056+Rows*32,16);
constexpr uint32_t fb=1088+Rows*32;
LocalTensor<float> fw1(TPosition::VECCALC,fb,256),fw2(TPosition::VECCALC,fb+1024,256);
LocalTensor<float> fx(TPosition::VECCALC,fb+2048,Rows*16);
constexpr uint32_t pb=fb+2048+Rows*64;
LocalTensor<float> pg(TPosition::VECCALC,pb,256),pu(TPosition::VECCALC,pb+1024,256);
constexpr uint32_t ab=pb+2048;
LocalTensor<float> ag(TPosition::VECCALC,ab,Rows*16),au(TPosition::VECCALC,ab+Rows*64,Rows*16);
constexpr uint32_t hb=ab+Rows*128;
LocalTensor<T> g(TPosition::VECCALC,hb,Rows*16),u(TPosition::VECCALC,hb+Rows*32,Rows*16);
LocalTensor<T> sig(TPosition::VECCALC,hb+Rows*64,Rows*16),ones(TPosition::VECCALC,hb+Rows*96,Rows*16),out(TPosition::VECOUT,hb+Rows*128,Rows*16);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams wc{static_cast<uint16_t>(d),d*2U,0,0,0},bc{1,d*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(16U-d),static_cast<T>(0)};
DataCopyPad(hw1,gw1,wc,pad);DataCopyPad(hw2,gw2,wc,pad);
DataCopyPad(hb1,gb1,bc,pad);DataCopyPad(hb2,gb2,bc,pad);
if constexpr (SingleBatch) {
DataCopyExtParams rc{static_cast<uint16_t>(Rows),d*2U,0,0,0};
DataCopyPad(hx,gx[static_cast<uint64_t>(GetBlockIdx())*Rows*d],rc,pad);
}
Sync<HardEvent::MTE2_V>();
Cast(fw1,hw1,RoundMode::CAST_NONE,d*16U);Cast(fw2,hw2,RoundMode::CAST_NONE,d*16U);
Duplicate(ones,static_cast<T>(1),Rows*16U);
Duplicate(ag,0.0f,Rows*16U);Duplicate(au,0.0f,Rows*16U);
const BinaryRepeatParams rp{1,1,1,2,0,2};
const BinaryRepeatParams bp{1,1,1,1,1,0};
const uint64_t first=static_cast<uint64_t>(GetBlockIdx())*Rows;
const uint64_t stride=SingleBatch?0U:static_cast<uint64_t>(GetBlockNum())*Rows;
uint64_t groups=1U;
if constexpr (!SingleBatch) { groups=(td->m-first+stride-1U)/stride; }
for (uint64_t group=0;group<groups;++group) {
const uint64_t row=first+group*stride;
const uint32_t nr=SingleBatch?Rows:(td->m-row<Rows?static_cast<uint32_t>(td->m-row):Rows);
DataCopyExtParams rc{static_cast<uint16_t>(nr),d*2U,0,0,0};
if constexpr (!SingleBatch) { DataCopyPad(hx,gx[row*d],rc,pad);Sync<HardEvent::MTE2_V>(); }
Cast(fx,hx,RoundMode::CAST_NONE,nr*16U);PipeBarrier<PIPE_V>();
#pragma unroll
for (uint32_t i=0;i<nr;++i) {
Mul(pg,fx[i*16U],fw1,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
Mul(pu,fx[i*16U],fw2,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
PipeBarrier<PIPE_V>();
WholeReduceSum(ag[i*16U],pg,16,d,1,1,2);WholeReduceSum(au[i*16U],pu,16,d,1,1,2);
PipeBarrier<PIPE_V>();
}
Cast(g,ag,RoundMode::CAST_RINT,nr*16U);Cast(u,au,RoundMode::CAST_RINT,nr*16U);
PipeBarrier<PIPE_V>();
Add(g,g,hb1,static_cast<uint64_t>(d),static_cast<uint8_t>(nr),bp);
Add(u,u,hb2,static_cast<uint64_t>(d),static_cast<uint8_t>(nr),bp);
PipeBarrier<PIPE_V>();GateCounter<T>(g,u,sig,out,ones,nr*16U);
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[row*d],out,rc);
if constexpr (!SingleBatch) { Sync<HardEvent::MTE3_MTE2>(); }
}
if constexpr (SingleBatch) { PipeBarrier<PIPE_ALL>(); }
}
private:
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0); }
};
template <typename T>
class ScalarFeatureSwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
GM_ADDR y, const SwiGluTinyTilingData *td) {
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x,td->m);gy.SetGlobalBuffer((__gm__ T*)y,td->m);
gw1.SetGlobalBuffer((__gm__ T*)w1,1);gw2.SetGlobalBuffer((__gm__ T*)w2,1);
gb1.SetGlobalBuffer((__gm__ T*)b1,1);gb2.SetGlobalBuffer((__gm__ T*)b2,1);
LocalTensor<T> hx(TPosition::VECIN,0,256), params(TPosition::VECIN,512,64);
LocalTensor<float> fx(TPosition::VECCALC,640,256), fg(TPosition::VECCALC,1664,256), fu(TPosition::VECCALC,2688,256);
LocalTensor<T> g(TPosition::VECCALC,3712,256), u(TPosition::VECCALC,4224,256);
LocalTensor<T> sig(TPosition::VECCALC,4736,256), ones(TPosition::VECCALC,5248,256), out(TPosition::VECOUT,5760,256);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams pc{1,2,0,0,0};
DataCopyPadExtParams<T> pp{true,0,15,static_cast<T>(0)};
DataCopyPad(params,gw1,pc,pp);DataCopyPad(params[16],gw2,pc,pp);
DataCopyPad(params[32],gb1,pc,pp);DataCopyPad(params[48],gb2,pc,pp);
Sync<HardEvent::MTE2_S>();
const float a=static_cast<float>(params.GetValue(0)),b=static_cast<float>(params.GetValue(16));
const T c=params.GetValue(32),e=params.GetValue(48);
Duplicate(ones,static_cast<T>(1),256);
for (uint64_t start=static_cast<uint64_t>(GetBlockIdx())*256U;start<td->m;start+=static_cast<uint64_t>(GetBlockNum())*256U) {
const uint32_t n=td->m-start<256U?static_cast<uint32_t>(td->m-start):256U;
DataCopyExtParams cp{1,n*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>((16U-n%16U)%16U),static_cast<T>(0)};
DataCopyPad(hx,gx[start],cp,pad);Sync<HardEvent::MTE2_V>();
Cast(fx,hx,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();
Muls(fg,fx,a,n);Muls(fu,fx,b,n);PipeBarrier<PIPE_V>();
Cast(g,fg,RoundMode::CAST_RINT,n);Cast(u,fu,RoundMode::CAST_RINT,n);
PipeBarrier<PIPE_V>();Adds(g,g,c,n);Adds(u,u,e,n);
PipeBarrier<PIPE_V>(); GateCounter<T>(g,u,sig,out,ones,n);
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[start],out,cp);
Sync<HardEvent::MTE3_MTE2>();
}
}
private:
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
};
template <typename T, uint32_t FixedD, uint32_t DynamicMaxK = 256U>
class SmallMVectorSwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
GM_ADDR y, const SwiGluTinyTilingData *td) {
constexpr uint32_t maxK=FixedD==0?DynamicMaxK:FixedD;
constexpr uint32_t wElems=16U*maxK;
constexpr uint32_t fBase=4U*wElems+2U*maxK+64U;
const uint32_t d=FixedD!=0?FixedD:static_cast<uint32_t>(td->d);
const uint32_t ka=(d+15U)/16U*16U;
const uint32_t columns=(d+15U)/16U;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x,td->m*d);gy.SetGlobalBuffer((__gm__ T*)y,td->m*d);
gw1.SetGlobalBuffer((__gm__ T*)w1,d*d);gw2.SetGlobalBuffer((__gm__ T*)w2,d*d);
gb1.SetGlobalBuffer((__gm__ T*)b1,d);gb2.SetGlobalBuffer((__gm__ T*)b2,d);
LocalTensor<T> hw1(TPosition::VECIN,0,wElems),hw2(TPosition::VECIN,2U*wElems,wElems);
LocalTensor<T> hx(TPosition::VECIN,4U*wElems,maxK);
LocalTensor<T> hb1(TPosition::VECIN,4U*wElems+2U*maxK,16),hb2(TPosition::VECIN,4U*wElems+2U*maxK+32U,16);
LocalTensor<float> fw1(TPosition::VECCALC,fBase,wElems),fw2(TPosition::VECCALC,fBase+4U*wElems,wElems);
LocalTensor<float> pg(TPosition::VECCALC,fBase+8U*wElems,wElems),pu(TPosition::VECCALC,fBase+12U*wElems,wElems);
constexpr uint32_t xBase=fBase+16U*wElems;
LocalTensor<float> fx(TPosition::VECCALC,xBase,maxK);
constexpr uint32_t aBase=xBase+4U*maxK;
LocalTensor<float> ag(TPosition::VECCALC,aBase,16),au(TPosition::VECCALC,aBase+64U,16);
LocalTensor<float> sg(TPosition::VECCALC,aBase+128U,16),su(TPosition::VECCALC,aBase+192U,16);
constexpr uint32_t hBase=aBase+256U;
LocalTensor<T> g(TPosition::VECCALC,hBase,16),u(TPosition::VECCALC,hBase+32U,16);
LocalTensor<T> sig(TPosition::VECCALC,hBase+64U,16),ones(TPosition::VECCALC,hBase+96U,16),out(TPosition::VECOUT,hBase+128U,16);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
Duplicate(ones,static_cast<T>(1),16);
for (uint64_t task=GetBlockIdx();task<td->m*columns;task+=GetBlockNum()) {
const uint64_t row=task/columns;
const uint32_t col=task%columns*16U;
const uint32_t cols=d-col<16U?d-col:16U;
DataCopyExtParams wc{static_cast<uint16_t>(cols),d*2U,0,0,0};
DataCopyExtParams xc{1,d*2U,0,0,0},bc{1,cols*2U,0,0,0};
DataCopyPadExtParams<T> wp{true,0,static_cast<uint8_t>(ka-d),static_cast<T>(0)};
DataCopyPadExtParams<T> bp{true,0,static_cast<uint8_t>(16U-cols),static_cast<T>(0)};
DataCopyPad(hw1,gw1[static_cast<uint64_t>(col)*d],wc,wp);
DataCopyPad(hw2,gw2[static_cast<uint64_t>(col)*d],wc,wp);
DataCopyPad(hx,gx[row*d],xc,wp);
DataCopyPad(hb1,gb1[col],bc,bp);DataCopyPad(hb2,gb2[col],bc,bp);
Sync<HardEvent::MTE2_V>();
Cast(fw1,hw1,RoundMode::CAST_NONE,cols*ka);Cast(fw2,hw2,RoundMode::CAST_NONE,cols*ka);
Cast(fx,hx,RoundMode::CAST_NONE,ka);PipeBarrier<PIPE_V>();
Dot<true>(ag,au,sg,su,pg,pu,fx,fw1,fw2,ka,0,cols);
for (uint32_t k=64U;k<ka;k+=64U) { Dot<false>(ag,au,sg,su,pg,pu,fx,fw1,fw2,ka,k,cols); }
Cast(g,ag,RoundMode::CAST_RINT,cols);Cast(u,au,RoundMode::CAST_RINT,cols);
PipeBarrier<PIPE_V>();Add(g,g,hb1,cols);Add(u,u,hb2,cols);
PipeBarrier<PIPE_V>(); GateCounter<T>(g,u,sig,out,ones,cols);
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[row*d+col],out,bc);
Sync<HardEvent::MTE3_MTE2>();
}
}
private:
template <bool First> __aicore__ inline void Dot(
const LocalTensor<float>& ag,const LocalTensor<float>& au,const LocalTensor<float>& sg,const LocalTensor<float>& su,
const LocalTensor<float>& pg,const LocalTensor<float>& pu,const LocalTensor<float>& fx,
const LocalTensor<float>& fw1,const LocalTensor<float>& fw2,uint32_t ka,uint32_t k,uint32_t cols) {
const uint32_t width=ka-k<64U?ka-k:64U;
const uint8_t stride=static_cast<uint8_t>(ka/8U);
const BinaryRepeatParams rp{1,1,1,stride,0,stride};
Mul(pg[k],fx[k],fw1[k],static_cast<uint64_t>(width),static_cast<uint8_t>(cols),rp);
Mul(pu[k],fx[k],fw2[k],static_cast<uint64_t>(width),static_cast<uint8_t>(cols),rp);
PipeBarrier<PIPE_V>();
if constexpr (First) {
WholeReduceSum(ag,pg[k],width,cols,1,1,stride);WholeReduceSum(au,pu[k],width,cols,1,1,stride);
PipeBarrier<PIPE_V>();
} else {
WholeReduceSum(sg,pg[k],width,cols,1,1,stride);WholeReduceSum(su,pu[k],width,cols,1,1,stride);
PipeBarrier<PIPE_V>();Add(ag,ag,sg,cols);Add(au,au,su,cols);PipeBarrier<PIPE_V>();
}
}
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0); }
};
template <typename T, uint32_t FixedD>
class SingleTinySwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y,const SwiGluTinyTilingData *td) {
const uint32_t d=FixedD!=0?FixedD:static_cast<uint32_t>(td->d);
const uint64_t row=GetBlockIdx();
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gy.SetGlobalBuffer((__gm__ T*)y);
gw1.SetGlobalBuffer((__gm__ T*)w1,d*d);gw2.SetGlobalBuffer((__gm__ T*)w2,d*d);
gb1.SetGlobalBuffer((__gm__ T*)b1,d);gb2.SetGlobalBuffer((__gm__ T*)b2,d);
LocalTensor<T> hw1(TPosition::VECIN,0,256),hw2(TPosition::VECIN,512,256);
LocalTensor<T> hx(TPosition::VECIN,1024,16),hb1(TPosition::VECIN,1056,16),hb2(TPosition::VECIN,1088,16);
LocalTensor<float> fw1(TPosition::VECCALC,1152,256),fw2(TPosition::VECCALC,2176,256);
LocalTensor<float> fx(TPosition::VECCALC,3200,16),pg(TPosition::VECCALC,3264,256),pu(TPosition::VECCALC,4288,256);
LocalTensor<float> ag(TPosition::VECCALC,5312,16),au(TPosition::VECCALC,5376,16);
LocalTensor<T> g(TPosition::VECCALC,5440,16),u(TPosition::VECCALC,5472,16);
LocalTensor<T> sig(TPosition::VECCALC,5504,16),ones(TPosition::VECCALC,5536,16),out(TPosition::VECOUT,5568,16);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams wc{static_cast<uint16_t>(d),d*2U,0,0,0},rc{1,d*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(16U-d),static_cast<T>(0)};
DataCopyPad(hw1,gw1,wc,pad);DataCopyPad(hw2,gw2,wc,pad);
DataCopyPad(hb1,gb1,rc,pad);DataCopyPad(hb2,gb2,rc,pad);DataCopyPad(hx,gx[row*d],rc,pad);
Sync<HardEvent::MTE2_V>();
Duplicate(ones,static_cast<T>(1),16);
if constexpr (FixedD==1) {
Mul(g,hx,hw1,1);Mul(u,hx,hw2,1);
} else {
Cast(fw1,hw1,RoundMode::CAST_NONE,d*16U);Cast(fw2,hw2,RoundMode::CAST_NONE,d*16U);
Cast(fx,hx,RoundMode::CAST_NONE,16);
const BinaryRepeatParams rp{1,1,1,2,0,2};
Mul(pg,fx,fw1,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
Mul(pu,fx,fw2,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
WholeReduceSum(ag,pg,16,d,1,1,2);WholeReduceSum(au,pu,16,d,1,1,2);
Cast(g,ag,RoundMode::CAST_RINT,d);Cast(u,au,RoundMode::CAST_RINT,d);
}
Add(g,g,hb1,d);Add(u,u,hb2,d);
if constexpr (FixedD != 0) {
FastGate<T, FixedD>(g, u, sig, out, ones);
} else {
GateCounter<T>(g, u, sig, out, ones, d);
}
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[row*d],out,rc);
}
private:
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0); }
};
template <typename T,uint32_t FixedD,bool OneRow>
class TinyColumnsSwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y,const SwiGluTinyTilingData* td) {
const uint32_t d=FixedD!=0?FixedD:(OneRow?GetBlockNum():static_cast<uint32_t>(td->d));
const uint32_t task=GetBlockIdx();
const uint64_t row=OneRow?0U:task/d;
const uint32_t col=OneRow?task:task%d;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gw1.SetGlobalBuffer((__gm__ T*)w1);gw2.SetGlobalBuffer((__gm__ T*)w2);
gb1.SetGlobalBuffer((__gm__ T*)b1);gb2.SetGlobalBuffer((__gm__ T*)b2);gy.SetGlobalBuffer((__gm__ T*)y);
LocalTensor<T> hx(TPosition::VECIN,0,16),hw1(TPosition::VECIN,32,16),hw2(TPosition::VECIN,64,16);
LocalTensor<T> hb1(TPosition::VECIN,96,16),hb2(TPosition::VECIN,128,16);
LocalTensor<float> fx(TPosition::VECCALC,192,16),fw1(TPosition::VECCALC,256,16),fw2(TPosition::VECCALC,320,16);
LocalTensor<float> pg(TPosition::VECCALC,384,16),pu(TPosition::VECCALC,448,16);
LocalTensor<float> ag(TPosition::VECCALC,512,8),au(TPosition::VECCALC,544,8);
LocalTensor<T> g(TPosition::VECCALC,576,16),u(TPosition::VECCALC,608,16);
LocalTensor<T> sig(TPosition::VECCALC,640,16),ones(TPosition::VECCALC,672,16),out(TPosition::VECOUT,704,16);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams cp{1,d*2U,0,0,0},bp{1,2,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(16U-d),static_cast<T>(0)};
DataCopyPadExtParams<T> one{true,0,15,static_cast<T>(0)};
DataCopyPad(hx,gx[row*d],cp,pad);DataCopyPad(hw1,gw1[col*d],cp,pad);DataCopyPad(hw2,gw2[col*d],cp,pad);
DataCopyPad(hb1,gb1[col],bp,one);DataCopyPad(hb2,gb2[col],bp,one);
Sync<HardEvent::MTE2_V>();
Cast(fx,hx,RoundMode::CAST_NONE,16);Cast(fw1,hw1,RoundMode::CAST_NONE,16);Cast(fw2,hw2,RoundMode::CAST_NONE,16);
Duplicate(ones,static_cast<T>(1),16);PipeBarrier<PIPE_V>();
Mul(pg,fx,fw1,16);Mul(pu,fx,fw2,16);PipeBarrier<PIPE_V>();
WholeReduceSum(ag,pg,16,1,1,1,2);WholeReduceSum(au,pu,16,1,1,1,2);PipeBarrier<PIPE_V>();
Cast(g,ag,RoundMode::CAST_RINT,1);Cast(u,au,RoundMode::CAST_RINT,1);PipeBarrier<PIPE_V>();
Add(g,g,hb1,1);Add(u,u,hb2,1);PipeBarrier<PIPE_V>();
GateCounter<T>(g,u,sig,out,ones,1);Sync<HardEvent::V_MTE3>();
DataCopyPad(gy[row*d+col],out,bp);PipeBarrier<PIPE_ALL>();
}
private:
template <HardEvent E> __aicore__ inline void Sync() {SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0);}
};
template <typename T>
class SubBlockGemv4 {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y) {
if (GetBlockIdx() != 0U) { return; }
GlobalTensor<T> gx, gw1, gw2, gb1, gb2, gy;
gx.SetGlobalBuffer((__gm__ T*)x, 4);
gw1.SetGlobalBuffer((__gm__ T*)w1, 16);
gw2.SetGlobalBuffer((__gm__ T*)w2, 16);
gb1.SetGlobalBuffer((__gm__ T*)b1, 4);
gb2.SetGlobalBuffer((__gm__ T*)b2, 4);
gy.SetGlobalBuffer((__gm__ T*)y, 4);
LocalTensor<T> hx(TPosition::VECIN, 0, 16);
LocalTensor<T> w(TPosition::VECIN, 32, 32);
LocalTensor<T> bias(TPosition::VECIN, 96, 32);
LocalTensor<float> prod(TPosition::VECCALC, 352, 32);
LocalTensor<float> pairs(TPosition::VECCALC, 480, 16);
LocalTensor<float> acc(TPosition::VECCALC, 544, 40);
LocalTensor<T> g(TPosition::VECCALC, 704, 32);
LocalTensor<T> sig(TPosition::VECCALC, 768, 16);
LocalTensor<T> ones(TPosition::VECCALC, 800, 16);
LocalTensor<T> out(TPosition::VECOUT, 832, 16);
SetAtomicNone();
SetMaskNorm();
const DataCopyExtParams xc{1, 8, 0, 0, 0};
const DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
GlobalTensor<uint64_t> xword;
xword.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(x), 1);
DataCachePreload(xword, static_cast<int16_t>(0));
DataCopy(w, gw1, 16);
DataCopy(w[16], gw2, 16);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
DataCopyPad(bias, gb1, xc, pad);
DataCopyPad(bias[16], gb2, xc, pad);
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
Duplicate(prod, 0.0f, 32);
const uint64_t xv = xword.GetValue(0);
auto xx = hx.template ReinterpretCast<uint64_t>();
xx.SetValue(0, xv);
xx.SetValue(1, xv);
xx.SetValue(2, xv);
xx.SetValue(3, xv);
SetFlag<HardEvent::S_V>(EVENT_ID0);
WaitFlag<HardEvent::S_V>(EVENT_ID0);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
MulAddDst<float, T>(prod, hx, w, static_cast<uint64_t>(16), 2, BinaryRepeatParams{1, 1, 1, 2, 0, 1});
PairReduceSum(pairs, prod, 1, 32, 1, 1, 8);
PairReduceSum(acc, pairs, 2, 8, 1, 1, 1);
constexpr uint64_t mask = 0;
SetVectorMask<float, MaskMode::NORMAL>(0, 15);
Cast<T, float, false>(g, acc, RoundMode::CAST_RINT, mask, 2, UnaryRepeatParams{1, 1, 1, 4});
SetVectorMask<T, MaskMode::NORMAL>(0, 15);
Duplicate<T, false>(ones, static_cast<T>(1), mask, 1, 1, 8);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
Add<T, false>(g, g, bias, mask, 2, BinaryRepeatParams{1, 1, 1, 1, 1, 1});
FastGate<T, 4U, false>(g, g[16], sig, out, ones);
Sync<HardEvent::V_MTE3>();
DataCopyPad(gy, out, xc);
}
private:
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
};
template <typename T>
class ScalarChannelSwiGlu {
public:
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y) {
        if (GetBlockIdx() != 0U) { return; }
        GlobalTensor<T> gx, gw1, gw2, gb1, gb2, gy;
        gx.SetGlobalBuffer((__gm__ T*)x, 1); gy.SetGlobalBuffer((__gm__ T*)y, 1);
        gw1.SetGlobalBuffer((__gm__ T*)w1, 1); gw2.SetGlobalBuffer((__gm__ T*)w2, 1);
        gb1.SetGlobalBuffer((__gm__ T*)b1, 1); gb2.SetGlobalBuffer((__gm__ T*)b2, 1);

        LocalTensor<T> hx(TPosition::VECIN, 0, 16);
        LocalTensor<T> w(TPosition::VECIN, 32, 32);
        LocalTensor<T> bias(TPosition::VECIN, 96, 32);

        LocalTensor<T> g(TPosition::VECCALC, 160, 32);
        LocalTensor<T> sig(TPosition::VECCALC, 224, 16);
        LocalTensor<T> ones(TPosition::VECCALC, 256, 16);
        LocalTensor<T> out(TPosition::VECOUT, 288, 16);

        SetAtomicNone();
        SetMaskNorm();
        SetVectorMask<T, MaskMode::NORMAL>(0, 1);

        const DataCopyExtParams rc{1, 2, 0, 0, 0};
        const DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        const T xv=gx.GetValue(0);
const T wv1=gw1.GetValue(0),wv2=gw2.GetValue(0);
const T bv1=gb1.GetValue(0),bv2=gb2.GetValue(0);
hx.SetValue(0,xv);w.SetValue(0,wv1);w.SetValue(16,wv2);
bias.SetValue(0,bv1);bias.SetValue(16,bv2);
SetFlag<HardEvent::S_V>(EVENT_ID0);
WaitFlag<HardEvent::S_V>(EVENT_ID0);
constexpr uint64_t mask = 0;
        
        Mul<T, false>(g, hx, w, mask, 2, BinaryRepeatParams{1, 1, 1, 1, 0, 1});
        
        Add<T, false>(g, g, bias, mask, 2, BinaryRepeatParams{1, 1, 1, 1, 1, 1});

        FastGate<T, 1U, false>(g, g[16], sig, out, ones);

        Sync<HardEvent::V_MTE3>();
        DataCopyPad(gy, out, rc);
    }
private:
    template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
};
template <typename T>
class AlignedRowGemv32 {
public:
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y) {
        if (GetBlockIdx() != 0U) { return; }
        GlobalTensor<T> gx, gw1, gw2, gb1, gb2, gy;
        gx.SetGlobalBuffer((__gm__ T*)x, 32);
        gy.SetGlobalBuffer((__gm__ T*)y, 32);
        gw1.SetGlobalBuffer((__gm__ T*)w1, 1024);
        gw2.SetGlobalBuffer((__gm__ T*)w2, 1024);
        gb1.SetGlobalBuffer((__gm__ T*)b1, 32);
        gb2.SetGlobalBuffer((__gm__ T*)b2, 32);

        LocalTensor<T> hw1(TPosition::VECIN, 0, 1024);
        LocalTensor<T> hw2(TPosition::VECIN, 2048, 1024);
        LocalTensor<T> hx(TPosition::VECIN, 4096, 64);
        LocalTensor<T> hb(TPosition::VECIN, 4224, 64);

        LocalTensor<float> pg(TPosition::VECCALC, 4352, 2048);
        LocalTensor<float> pu(TPosition::VECCALC, 12544, 256);
        LocalTensor<float> ag(TPosition::VECCALC, 13568, 64);

        LocalTensor<T> gu(TPosition::VECCALC, 13824, 64);
        LocalTensor<T> sig(TPosition::VECCALC, 13952, 32);
        LocalTensor<T> ones(TPosition::VECCALC, 14016, 32);
        LocalTensor<T> out(TPosition::VECOUT, 14080, 32);

        SetAtomicNone();
        SetMaskNorm();
        constexpr uint64_t mask = 0;
        SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);

        DataCopy(hx, gx, 32);
        DataCopy(hx[32], gx, 32);
        DataCopy(hb, gb1, 32);
        DataCopy(hb[32], gb2, 32);
        DataCopy(hw1, gw1, 1024);
        
        DataCopy(hw2, gw2, 1024);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);

        Duplicate<float, false>(pg, 0.0f, mask, 32U, 1, 8);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
MulAddDst<float, T, false>(pg, hx, hw1, mask, 32U, BinaryRepeatParams{1, 1, 1, 8, 0, 4});
BlockReduceSum<float, false>(pu, pg, 32U, mask, 1, 1, 8);
PairReduceSum<float, false>(pg, pu, 4U, mask, 1, 1, 8);
        PairReduceSum<float, false>(ag, pg, 2U, mask, 1, 1, 8);

        Cast<T, float, false>(gu, ag, RoundMode::CAST_RINT, mask, 1, UnaryRepeatParams{1, 1, 4, 8});
        SetVectorMask<T, MaskMode::NORMAL>(0, ~0ULL);
        Add<T, false>(gu, gu, hb, mask, 1, BinaryRepeatParams{1, 1, 1, 8, 8, 8});

        FastGate<T, 32U, false>(gu, gu[32], sig, out, ones);

        Sync<HardEvent::V_MTE3>();
        DataCopy(gy, out, 32U);
    }
private:
    template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
};
template <typename T, uint32_t FixedD, uint32_t FixedM>
class AlignedColGemv {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
GM_ADDR y) {
const uint64_t task=GetBlockIdx();
if constexpr (FixedM==1U) {
if (task>=2U) { return; }
}
constexpr uint32_t maxK=FixedD;
constexpr uint32_t wElems=16U*maxK;
constexpr uint32_t fBase=4U*wElems+2U*maxK+64U;
const uint32_t d=FixedD;
const uint32_t ka=(d+15U)/16U*16U;
const uint32_t columns=(d+15U)/16U;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x,FixedM*d);gy.SetGlobalBuffer((__gm__ T*)y,FixedM*d);
gw1.SetGlobalBuffer((__gm__ T*)w1,d*d);gw2.SetGlobalBuffer((__gm__ T*)w2,d*d);
gb1.SetGlobalBuffer((__gm__ T*)b1,d);gb2.SetGlobalBuffer((__gm__ T*)b2,d);
LocalTensor<T> hw1(TPosition::VECIN,0,2U*wElems),hw2(TPosition::VECIN,2U*wElems,wElems);
LocalTensor<T> hx(TPosition::VECIN,4U*wElems,maxK);
LocalTensor<T> hb1(TPosition::VECIN,4U*wElems+2U*maxK,32),hb2(TPosition::VECIN,4U*wElems+2U*maxK+32U,16);
LocalTensor<float> fw1(TPosition::VECCALC,fBase,2U*wElems),fw2(TPosition::VECCALC,fBase+4U*wElems,wElems);
LocalTensor<float> pg(TPosition::VECCALC,fBase+8U*wElems,2U*wElems),pu(TPosition::VECCALC,fBase+12U*wElems,wElems);
constexpr uint32_t xBase=fBase+16U*wElems;
LocalTensor<float> fx(TPosition::VECCALC,xBase,maxK);
constexpr uint32_t aBase=xBase+4U*maxK;
LocalTensor<float> ag(TPosition::VECCALC,aBase,32),au(TPosition::VECCALC,aBase+64U,16);
LocalTensor<float> sg(TPosition::VECCALC,aBase+128U,16),su(TPosition::VECCALC,aBase+192U,16);
constexpr uint32_t hBase=aBase+256U;
LocalTensor<T> g(TPosition::VECCALC,hBase,32),u(TPosition::VECCALC,hBase+32U,16);
LocalTensor<T> sig(TPosition::VECCALC,hBase+64U,16),ones(TPosition::VECCALC,hBase+96U,16),out(TPosition::VECOUT,hBase+128U,16);
SetAtomicNone();SetMaskNorm();
constexpr uint64_t mask=0;
SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
{
const uint64_t row=FixedM==1U?0U:task/columns;
const uint32_t col=(FixedM==1U?task:task%columns)*16U;
static_assert(FixedD==32U || FixedD==64U);
if constexpr (FixedD==64U) {
DataCopy(hw1,gw1[static_cast<uint64_t>(col)*d],16U*FixedD);
DataCopy(hw2,gw2[static_cast<uint64_t>(col)*d],16U*FixedD);
DataCopy(hx,gx[row*d],FixedD);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
DataCopy(hb1,gb1[col],16U);DataCopy(hb2,gb2[col],16U);
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
Duplicate<T,false>(ones,static_cast<T>(1),mask,1,1,8);
Duplicate<float,false>(pg,0.0f,mask,32U,1,8);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
MulAddDst<float,T,false>(pg,hx,hw1,mask,32U,BinaryRepeatParams{1,1,1,8,0,4});
BlockReduceSum<float,false>(pu,pg,32U,mask,1,1,8);
BlockReduceSum<float,false>(ag,pu,4U,mask,1,1,8);
} else {
DataCopy(hw1,gw1[static_cast<uint64_t>(col)*d],16U*FixedD);
DataCopy(hw2,gw2[static_cast<uint64_t>(col)*d],16U*FixedD);
DataCopy(hx,gx[row*d],FixedD);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
DataCopy(hb1,gb1[col],16U);DataCopy(hb2,gb2[col],16U);
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
Duplicate<T,false>(ones,static_cast<T>(1),mask,1,1,8);
Duplicate<float,false>(pg,0.0f,mask,16U,1,8);
SetVectorMask<float,MaskMode::NORMAL>(0,0xFFFFFFFFULL);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
MulAddDst<float,T,false>(pg,hx,hw1,mask,32U,BinaryRepeatParams{1,1,1,4,0,2});
SetVectorMask<float,MaskMode::NORMAL>(0,~0ULL);
BlockReduceSum<float,false>(pu,pg,16U,mask,1,1,8);
PairReduceSum<float,false>(pu[128],pu,2U,mask,1,1,8);
PairReduceSum<float,false>(ag,pu[128],1U,mask,1,1,8);
}
SetVectorMask<float,MaskMode::NORMAL>(0,0xFFFFFFFFULL);
Cast<T,float,false>(g,ag,RoundMode::CAST_RINT,mask,1,UnaryRepeatParams{1,1,4,8});
WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
SetVectorMask<T,MaskMode::NORMAL>(0,0xFFFFFFFFULL);
Add<T,false>(g,g,hb1,mask,1,BinaryRepeatParams{1,1,1,2,2,2});
FastGate<T,16U>(g,g[16],sig,out,ones);
Sync<HardEvent::V_MTE3>();
if constexpr (FixedM==1U) {
DataCopy(gy[col],out,16U);
} else {
DataCopy(gy[row*d+col],out,16U);
}
}
}
private:
template <bool First> __aicore__ inline void Dot(
const LocalTensor<float>& ag,const LocalTensor<float>& au,const LocalTensor<float>& sg,const LocalTensor<float>& su,
const LocalTensor<float>& pg,const LocalTensor<float>& pu,const LocalTensor<float>& fx,
const LocalTensor<float>& fw1,const LocalTensor<float>& fw2,uint32_t ka,uint32_t k,uint32_t cols) {
const uint32_t width=ka-k<64U?ka-k:64U;
const uint8_t stride=static_cast<uint8_t>(ka/8U);
const BinaryRepeatParams rp{1,1,1,stride,0,stride};
Mul(pg[k],fx[k],fw1[k],static_cast<uint64_t>(width),static_cast<uint8_t>(cols),rp);
Mul(pu[k],fx[k],fw2[k],static_cast<uint64_t>(width),static_cast<uint8_t>(cols),rp);
PipeBarrier<PIPE_V>();
if constexpr (First) {
WholeReduceSum(ag,pg[k],width,cols,1,1,stride);WholeReduceSum(au,pu[k],width,cols,1,1,stride);
PipeBarrier<PIPE_V>();
} else {
WholeReduceSum(sg,pg[k],width,cols,1,1,stride);WholeReduceSum(su,pu[k],width,cols,1,1,stride);
PipeBarrier<PIPE_V>();Add(ag,ag,sg,cols);Add(au,au,su,cols);PipeBarrier<PIPE_V>();
}
}
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0); }
};
template <typename T,uint32_t D,uint32_t Rows>
class BatchedRowGemv {
static constexpr uint32_t K=(D+15U)/16U*16U, W=D*K;
static constexpr uint32_t HB=4U*W+2U*Rows*K;
static constexpr uint32_t FB=HB+4U*K;
static constexpr uint32_t XB=FB+8U*W, PB=XB+4U*Rows*K;
static constexpr uint32_t AB=PB+8U*W, GB=AB+8U*Rows*K;
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y) {
const uint64_t row=static_cast<uint64_t>(GetBlockIdx())*Rows;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gw1.SetGlobalBuffer((__gm__ T*)w1);gw2.SetGlobalBuffer((__gm__ T*)w2);
gb1.SetGlobalBuffer((__gm__ T*)b1);gb2.SetGlobalBuffer((__gm__ T*)b2);gy.SetGlobalBuffer((__gm__ T*)y);
LocalTensor<T> hw1(TPosition::VECIN,0,W),hw2(TPosition::VECIN,2U*W,W),hx(TPosition::VECIN,4U*W,Rows*K);
LocalTensor<T> hb1(TPosition::VECIN,HB,K),hb2(TPosition::VECIN,HB+2U*K,K);
LocalTensor<float> fw1(TPosition::VECCALC,FB,W),fw2(TPosition::VECCALC,FB+4U*W,W);
LocalTensor<float> fx(TPosition::VECCALC,XB,Rows*K),pg(TPosition::VECCALC,PB,W),pu(TPosition::VECCALC,PB+4U*W,W);
LocalTensor<float> ag(TPosition::VECCALC,AB,Rows*K),au(TPosition::VECCALC,AB+4U*Rows*K,Rows*K);
LocalTensor<T> g(TPosition::VECCALC,GB,Rows*K),u(TPosition::VECCALC,GB+2U*Rows*K,Rows*K);
LocalTensor<T> sig(TPosition::VECCALC,GB+4U*Rows*K,Rows*K),ones(TPosition::VECCALC,GB+6U*Rows*K,Rows*K);
LocalTensor<T> out(TPosition::VECOUT,GB+8U*Rows*K,Rows*K);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams wc{D,D*2U,0,0,0},bc{1,D*2U,0,0,0},xc{Rows,D*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(K-D),static_cast<T>(0)};
DataCopyPad(hw1,gw1,wc,pad);DataCopyPad(hw2,gw2,wc,pad);
DataCopyPad(hb1,gb1,bc,pad);DataCopyPad(hb2,gb2,bc,pad);DataCopyPad(hx,gx[row*D],xc,pad);
Sync<HardEvent::MTE2_V>();
Cast(fw1,hw1,RoundMode::CAST_NONE,W);Cast(fw2,hw2,RoundMode::CAST_NONE,W);
Cast(fx,hx,RoundMode::CAST_NONE,Rows*K);
Duplicate(ag,0.0f,Rows*K);Duplicate(au,0.0f,Rows*K);Duplicate(ones,static_cast<T>(1),Rows*K);
PipeBarrier<PIPE_V>();
const BinaryRepeatParams rp{1,1,1,K/8U,0,K/8U};
#pragma unroll
for (uint32_t i=0;i<Rows;++i) {
Mul(pg,fx[i*K],fw1,static_cast<uint64_t>(K),static_cast<uint8_t>(D),rp);
Mul(pu,fx[i*K],fw2,static_cast<uint64_t>(K),static_cast<uint8_t>(D),rp);
PipeBarrier<PIPE_V>();
WholeReduceSum(ag[i*K],pg,K,D,1,1,K/8U);WholeReduceSum(au[i*K],pu,K,D,1,1,K/8U);
PipeBarrier<PIPE_V>();
}
Cast(g,ag,RoundMode::CAST_RINT,Rows*K);Cast(u,au,RoundMode::CAST_RINT,Rows*K);PipeBarrier<PIPE_V>();
const BinaryRepeatParams bp{1,1,1,K/16U,K/16U,0};
Add(g,g,hb1,static_cast<uint64_t>(D),static_cast<uint8_t>(Rows),bp);
Add(u,u,hb2,static_cast<uint64_t>(D),static_cast<uint8_t>(Rows),bp);
PipeBarrier<PIPE_V>();GateCounter<T>(g,u,sig,out,ones,Rows*K);
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[row*D],out,xc);PipeBarrier<PIPE_ALL>();
}
private:
template <HardEvent E> __aicore__ inline void Sync() {SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0);}
};
template <typename T>
class PackedMultiBlockGemv {
static constexpr uint32_t D=33U,Rows=3U;
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y,GM_ADDR tiling) {
const uint32_t blk=GetBlockIdx();
const uint64_t row=static_cast<uint64_t>(blk)*Rows;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gw1.SetGlobalBuffer((__gm__ T*)w1);gw2.SetGlobalBuffer((__gm__ T*)w2);
gb1.SetGlobalBuffer((__gm__ T*)b1);gb2.SetGlobalBuffer((__gm__ T*)b2);gy.SetGlobalBuffer((__gm__ T*)y);
LocalTensor<T> hw1(TPosition::VECIN,0,1152),hw2(TPosition::VECIN,2304,1152),bothHalfW(TPosition::VECIN,0,2304);
LocalTensor<T> halfExpandedX(TPosition::VECIN,4608,192);
LocalTensor<T> hwCol32(TPosition::VECIN,5120,256),hxCol32(TPosition::VECIN,5632,128),bothBias(TPosition::VECIN,5888,256);
LocalTensor<float> fw1(TPosition::VECCALC,6400,1152),fw2(TPosition::VECCALC,11008,1152),bothW(TPosition::VECCALC,6400,2304);
LocalTensor<float> expandedX(TPosition::VECCALC,15680,192),bothP(TPosition::VECCALC,16512,7552);
LocalTensor<float> bothBlock(TPosition::VECCALC,46848,960),bothPair(TPosition::VECCALC,50688,512),acc(TPosition::VECCALC,52736,256);
LocalTensor<T> gu(TPosition::VECCALC,53760,256),sig(TPosition::VECCALC,54272,128),ones(TPosition::VECCALC,54528,128);
LocalTensor<T> out(TPosition::VECOUT,54784,128);
LocalTensor<T> rawW1(TPosition::VECIN,55040,1104),rawW2(TPosition::VECIN,57344,1104),rawX(TPosition::VECIN,59648,112);
LocalTensor<T> rawB1(TPosition::VECIN,59904,48),rawB2(TPosition::VECIN,60000,48);
LocalTensor<uint32_t> offsets(TPosition::VECIN,60160,1792);
LocalTensor<T> packedOut(TPosition::VECOUT,67328,112);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
GlobalTensor<uint32_t> indexGm;
indexGm.SetGlobalBuffer((__gm__ uint32_t*)(tiling+offsetof(SwiGluTilingData,gatherOffsetsB)),1792);
DataCopyExtParams wc{1,2178,0,0,0},bc{1,66,0,0,0};
DataCopyExtParams xcFlat{1,static_cast<uint32_t>((blk<28U?198U:66U)),0,0,0};
DataCopyPadExtParams<T> wp{true,0,15,static_cast<T>(0)};
DataCopyPadExtParams<T> xp{true,0,static_cast<uint8_t>(blk<28U?13U:15U),static_cast<T>(0)};
DataCopyPad(rawW1,gw1,wc,wp);
DataCopyPad(rawW2,gw2,wc,wp);
DataCopyPad(rawX,gx[row*D],xcFlat,xp);
DataCopy(offsets,indexGm,1792);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
DataCopyPad(rawB1,gb1,bc,wp);DataCopyPad(rawB2,gb2,bc,wp);
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
constexpr uint64_t mask=0;
Duplicate<float,false>(bothP[3456],0.0f,mask,10U,1,8);
Duplicate<T,false>(ones,static_cast<T>(1),mask,1U,1,8);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
Gather(hw1,rawW1,offsets,0,1152);
Gather(hw2,rawW2,offsets,0,1152);
Gather(halfExpandedX,rawX,offsets[1152],0,192);
Gather(hwCol32,rawW1,offsets[1344],0,112);
Gather(hwCol32[128],rawW2,offsets[1344],0,112);
Gather(hxCol32,rawX,offsets[1456],0,112);
SetVectorMask<float,MaskMode::NORMAL>(0,~0ULL);
Cast<float,T,false>(bothW,bothHalfW,RoundMode::CAST_NONE,mask,36U,UnaryRepeatParams{1,1,8,4});
Cast<float,T,false>(expandedX,halfExpandedX,RoundMode::CAST_NONE,mask,3U,UnaryRepeatParams{1,1,8,4});
const BinaryRepeatParams rp{1,1,1,8,0,8};
#pragma unroll
for (uint32_t i=0;i<Rows;++i) {
Mul<float,false>(bothP[i*1152U],expandedX[i*64U],fw1,mask,18U,rp);
Mul<float,false>(bothP[4096U+i*1152U],expandedX[i*64U],fw2,mask,18U,rp);
}
BlockReduceSum<float,false>(bothBlock,bothP,118U,mask,1,1,8);
PairReduceSum<float,false>(bothPair,bothBlock,15U,mask,1,1,8);
PairReduceSum<float,false>(acc,bothPair,8U,mask,1,1,8);
MulAddDst<float,T,false>(acc,hxCol32,hwCol32,mask,2U,BinaryRepeatParams{1,1,1,8,4,4});
MulAddDst<float,T,false>(acc[128],hxCol32,hwCol32[128],mask,2U,BinaryRepeatParams{1,1,1,8,4,4});
Cast<T,float,false>(gu,acc,RoundMode::CAST_RINT,mask,4U,UnaryRepeatParams{1,1,4,8});
WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
Gather(bothBias,rawB1,offsets[1568],0,112);
Gather(bothBias[128],rawB2,offsets[1568],0,112);
SetVectorMask<T,MaskMode::NORMAL>(~0ULL,~0ULL);
Add<T,false>(gu,gu,bothBias,mask,2U,BinaryRepeatParams{1,1,1,8,8,8});
FastGate<T,108U,false>(gu,gu[128],sig,out,ones);
Gather(packedOut,out,offsets[1680],0,Rows*D);
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[row*D],packedOut,xcFlat);
}
private:
template <HardEvent E> __aicore__ inline void Sync() {SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0);}
};
template <typename T>
class PackedSubBlockGemv {
static constexpr uint32_t D=13U,Rows=3U,K=16U,W=256U;
static constexpr uint32_t HB=4U*W+2U*Rows*K;
static constexpr uint32_t FB=HB+4U*K;
static constexpr uint32_t XB=FB+8U*W, PB=XB+4U*Rows*K;
static constexpr uint32_t AB=PB+8U*W, GB=AB+8U*Rows*K;
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y,GM_ADDR tiling) {
const uint32_t blk=GetBlockIdx();
if (blk>=7U) { return; }
const uint64_t row=static_cast<uint64_t>(blk)*Rows;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gw1.SetGlobalBuffer((__gm__ T*)w1);gw2.SetGlobalBuffer((__gm__ T*)w2);
gb1.SetGlobalBuffer((__gm__ T*)b1);gb2.SetGlobalBuffer((__gm__ T*)b2);gy.SetGlobalBuffer((__gm__ T*)y);
LocalTensor<T> hw1(TPosition::VECIN,0,W),hw2(TPosition::VECIN,2U*W,W);
LocalTensor<T> hb1(TPosition::VECIN,HB,K),hb2(TPosition::VECIN,HB+2U*K,K);
LocalTensor<T> sig(TPosition::VECCALC,GB+4U*Rows*K,Rows*K),ones(TPosition::VECCALC,GB+6U*Rows*K,Rows*K);
LocalTensor<T> out(TPosition::VECOUT,GB+8U*Rows*K,Rows*K);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
LocalTensor<T> rawW1(TPosition::VECIN,8448,176),rawW2(TPosition::VECIN,8960,176),rawX(TPosition::VECIN,9472,48);
LocalTensor<uint32_t> offsets(TPosition::VECIN,13056,496);
GlobalTensor<uint32_t> indexGm;
indexGm.SetGlobalBuffer((__gm__ uint32_t*)(tiling+offsetof(SwiGluTilingData,gatherOffsetsA)),496);
DataCopyExtParams wc{1,338,0,0,0},bc{1,26,0,0,0},xcFlat{1,78,0,0,0};
DataCopyPadExtParams<T> wp{true,0,7,static_cast<T>(0)},xp{true,0,9,static_cast<T>(0)},bp0{true,0,3,static_cast<T>(0)};
DataCopyPad(rawW1,gw1,wc,wp);
DataCopyPad(rawW2,gw2,wc,wp);
DataCopyPad(rawX,gx[row*D],xcFlat,xp);
DataCopy(offsets,indexGm,496);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
DataCopyPad(hb1,gb1,bc,bp0);DataCopyPad(hb2,gb2,bc,bp0);
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
constexpr uint64_t mask=0;
Duplicate<T,false>(ones,static_cast<T>(1),mask,1U,1,8);
LocalTensor<T> packedOut(TPosition::VECOUT,19392,48);
LocalTensor<float> bothP(TPosition::VECCALC,19840,1536);
LocalTensor<float> bothBlock(TPosition::VECCALC,34176,192);
LocalTensor<float> acc(TPosition::VECCALC,AB,96);
LocalTensor<T> bothBias(TPosition::VECIN,HB,32),gu(TPosition::VECCALC,GB,96);
LocalTensor<T> bothHalfW(TPosition::VECIN,0,512),halfExpandedX(TPosition::VECCALC,9984,192);
Duplicate<float,false>(bothP,0.0f,mask,24U,1,8);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
Gather(hw1,rawW1,offsets,0,W);
Gather(hw2,rawW2,offsets,0,W);
Gather(halfExpandedX,rawX,offsets[256],0,192);
SetVectorMask<float,MaskMode::NORMAL>(0,~0ULL);
const BinaryRepeatParams rp{1,1,1,8,0,4};
#pragma unroll
for (uint32_t i=0;i<Rows;++i) {
MulAddDst<float,T,false>(bothP[i*512U],halfExpandedX[i*64U],bothHalfW,mask,8,rp);
}
BlockReduceSum<float,false>(bothBlock,bothP,24,mask,1,1,8);
PairReduceSum<float,false>(acc,bothBlock,3,mask,1,1,8);
Cast<T,float,false>(gu,acc,RoundMode::CAST_RINT,mask,2U,UnaryRepeatParams{1,1,4,8});
WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
SetVectorMask<T,MaskMode::NORMAL>(0,0xFFFFFFFFULL);
Add<T,false>(gu,gu,bothBias,mask,3,BinaryRepeatParams{1,1,1,2,2,0});
SetVectorMask<T,MaskMode::NORMAL>(0,0xFFFFULL);
const UnaryRepeatParams compactG{1,1,1,2},contiguous{1,1,1,1};
const BinaryRepeatParams onesBroadcast{1,1,1,1,0,1};
const BinaryRepeatParams gateProduct{1,1,1,1,2,1},upProduct{1,1,1,1,1,2};
Muls<T,false>(sig,gu,static_cast<T>(-1),mask,3,compactG);
Exp<T,false>(sig,sig,mask,3,contiguous);
Adds<T,false>(sig,sig,static_cast<T>(1),mask,3,contiguous);
Div<T,false>(sig,ones,sig,mask,3,onesBroadcast);
Mul<T,false>(out,gu,sig,mask,3,gateProduct);
Mul<T,false>(out,out,gu[16],mask,3,upProduct);
Gather(packedOut,out,offsets[448],0,Rows*D);
Sync<HardEvent::V_MTE3>();
DataCopyPad(gy[row*D],packedOut,xcFlat);
}
private:
template <HardEvent E> __aicore__ inline void Sync() {SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0);}
};
namespace optiling {
using namespace AscendC;
using namespace matmul;
constexpr uint32_t kSlotCount = 4;
constexpr uint32_t kBlockRows = 16;
__aicore__ constexpr MatmulConfig DirectBasicConfig(uint32_t rows)
{
auto config = GetBasicConfig(rows, 128, 128);
config.enUnitFlag = true;
return config;
}
constexpr MatmulConfig kBasic16 = DirectBasicConfig(16);
constexpr MatmulConfig kBasic32 = DirectBasicConfig(32);
constexpr MatmulConfig kBasic128 = DirectBasicConfig(128);
template <typename T, uint32_t FixedM = 0, uint32_t FixedD = 0, bool Unaligned = false, bool Resident = false, bool FullTile = false, uint32_t FixedN = 0>
class DirectPairCube {
public:
__aicore__ inline void Init(uint32_t k, uint32_t m, uint32_t n) {
realK_ = FixedD != 0 ? FixedD : k;
k_ = Unaligned ? (realK_ + 15U) / 16U * 16U : realK_;
m_ = FixedM != 0 ? FixedM : m;
n_ = FixedN != 0 ? FixedN : (FixedD != 0 ? (FixedD < 64U ? FixedD : 64U) : n);
}
__aicore__ inline void PrepareWeights(const GlobalTensor<T> &w1,const GlobalTensor<T> &w2,uint32_t cols) {
LocalTensor<T> b1(TPosition::B1,m_*k_*sizeof(T),2U*n_*k_);
LocalTensor<T> b2(TPosition::B2,0,2U*n_*k_);
Nd2NzParams bp{1,static_cast<uint16_t>(cols),static_cast<uint16_t>(realK_),0,
static_cast<uint16_t>(realK_),static_cast<uint16_t>(2U*n_),1,0};
DataCopy(b1,w1,bp);DataCopy(b1[n_*16U],w2,bp);
Sync<HardEvent::MTE2_MTE1>();
LoadData2dParams bl;
bl.repeatTimes=static_cast<uint8_t>(2U*n_*k_/256U);bl.srcStride=1;
LoadData(b2,b1,bl);
Sync<HardEvent::MTE1_M>();
}
__aicore__ inline void Compute(const GlobalTensor<T> &x, const GlobalTensor<T> &w1,
const GlobalTensor<T> &w2, const GlobalTensor<T> &dst,
uint32_t rows, uint32_t cols) {
LocalTensor<T> a1(TPosition::A1, 0, m_ * k_);
LocalTensor<T> b1(TPosition::B1, m_ * k_ * sizeof(T), 2U * n_ * k_);
LocalTensor<T> a2(TPosition::A2, 0, m_ * k_);
LocalTensor<T> b2(TPosition::B2, 0, 2U * n_ * k_);
LocalTensor<float> c(TPosition::CO1, 0, 2U * m_ * n_);
const uint32_t alignedM = FullTile ? FixedM : (rows + 15U) / 16U * 16U;
Nd2NzParams ap{1, static_cast<uint16_t>(rows), static_cast<uint16_t>(realK_), 0,
static_cast<uint16_t>(realK_), static_cast<uint16_t>(alignedM), 1, 0};
Nd2NzParams bp{1, static_cast<uint16_t>(cols), static_cast<uint16_t>(realK_), 0,
static_cast<uint16_t>(realK_), static_cast<uint16_t>(2U * n_), 1, 0};
DataCopy(a1, x, ap);
if constexpr (!Resident) {
DataCopy(b1, w1, bp);
DataCopy(b1[n_ * 16U], w2, bp);
}
Sync<HardEvent::MTE2_MTE1>();
LoadData2dParams al;
al.repeatTimes = static_cast<uint8_t>(k_ / 16U);
al.srcStride = static_cast<uint16_t>(alignedM / 16U);
if constexpr (FullTile) {
#pragma unroll
for (uint32_t mb=0;mb<FixedM/16U;++mb) {
LoadData(a2[mb*k_*16U],a1[mb*256U],al);
}
} else {
for (uint32_t mb = 0; mb < alignedM / 16U; ++mb) {
LoadData(a2[mb * k_ * 16U], a1[mb * 256U], al);
}
}
LoadData2dParams bl;
bl.repeatTimes = static_cast<uint8_t>(2U * n_ * k_ / 256U);
bl.srcStride = 1;
if constexpr (!Resident) { LoadData(b2, b1, bl); }
if constexpr (Resident) { Sync<HardEvent::MTE1_MTE2>(); }
Sync<HardEvent::MTE1_M>();
MmadParams mad;
mad.m = static_cast<uint16_t>(alignedM);
mad.n = static_cast<uint16_t>(2U * n_);
mad.k = static_cast<uint16_t>(k_);
mad.cmatrixInitVal = true;
Mmad(c, a2, b2, mad);
if constexpr (Resident) { Sync<HardEvent::M_MTE1>(); }
Sync<HardEvent::M_FIX>();
FixpipeParamsV220 fix;
fix.nSize = static_cast<uint16_t>(2U * n_);
fix.mSize = static_cast<uint16_t>(rows);
fix.srcStride = static_cast<uint16_t>(alignedM);
fix.dstStride = 2U * n_;
fix.quantPre = QuantMode_t::F322F16;
Fixpipe<T, float, CFG_ROW_MAJOR>(dst, c, fix);
if constexpr (Resident) { Sync<HardEvent::FIX_M>(); }
else { PipeBarrier<PIPE_ALL>(); }
}
private:
template <HardEvent E>
__aicore__ inline void Sync() {
constexpr event_t event = EVENT_ID0;
SetFlag<E>(event);
WaitFlag<E>(event);
}
uint32_t k_ = 0, realK_ = 0, m_ = 0, n_ = 0;
};
struct EmptyMatmulStorage {};
template <bool Direct, typename A, typename B, typename C, const auto& Config>
struct MatmulStorageSelector {
using Type = MatmulImpl<A, B, C, C, Config>;
};
template <typename A, typename B, typename C, const auto& Config>
struct MatmulStorageSelector<true, A, B, C, Config> {
using Type = EmptyMatmulStorage;
};
template <typename xType, const auto& Config = CFG_NORM, bool Direct = false,
bool Padded = false, bool Merged = false>
class SwiGluCubeImpl {
public:
using AType = MatmulType<TPosition::GM, CubeFormat::ND, xType, false>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, xType, true>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, xType>;
typename MatmulStorageSelector<Direct, AType, BType, CType, Config>::Type mm;
DirectPairCube<xType> direct;
__aicore__ inline void Init(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
GM_ADDR userWorkspace, TPipe *pipe, const SwiGluTilingData *td)
{
pipe_ = pipe;
core_ = GetBlockIdx();
sub_ = GetSubBlockIdx();
if ASCEND_IS_AIV { core_ /= 2U; }
m_ = td->m;
d_ = static_cast<uint32_t>(td->d);
computeD_ = td->dAlign;
taskM_ = td->taskM;
taskN_ = td->taskN;
taskNum_ = td->taskNum;
cores_ = td->usedCoreNum;
tasksN_ = (d_ + taskN_ - 1U) / taskN_;
plane_ = taskM_ * taskN_;
x_.SetGlobalBuffer((__gm__ xType *)x, m_ * d_);
w1_.SetGlobalBuffer((__gm__ xType *)w1, static_cast<uint64_t>(d_) * d_);
w2_.SetGlobalBuffer((__gm__ xType *)w2, static_cast<uint64_t>(d_) * d_);
b1_.SetGlobalBuffer((__gm__ xType *)b1, d_);
b2_.SetGlobalBuffer((__gm__ xType *)b2, d_);
y_.SetGlobalBuffer((__gm__ xType *)y, m_ * d_);
slots_.SetGlobalBuffer((__gm__ xType *)userWorkspace,
static_cast<uint64_t>(plane_) * 2 * kSlotCount * cores_);
if constexpr (padded_) {
const uint64_t base = static_cast<uint64_t>(plane_) * 2 * kSlotCount * cores_;
const uint64_t wBase = base + m_ * computeD_;
paddedX_.SetGlobalBuffer((__gm__ xType *)userWorkspace + base, m_ * computeD_);
paddedW1_.SetGlobalBuffer((__gm__ xType *)userWorkspace + wBase,
static_cast<uint64_t>(computeD_) * computeD_ * (merged_ ? 2U : 1U));
paddedW2_.SetGlobalBuffer((__gm__ xType *)userWorkspace + wBase +
static_cast<uint64_t>(computeD_) * computeD_,
static_cast<uint64_t>(computeD_) * computeD_);
packRows_ = 4096U / computeD_;
}
if ASCEND_IS_AIC {
if constexpr (Direct) {
direct.Init(computeD_, taskM_, taskN_);
} else {
mm.Init(&td->cubeTilingData, pipe_);
mm.SetHF32(false, 0);
const uint32_t matmulN = merged_ ? 2U * taskN_ : taskN_;
mm.SetSingleShape(taskM_, matmulN, computeD_);
mm.SetOrgShape(taskM_, matmulN, computeD_, computeD_);
}
}
if ASCEND_IS_AIV {
const uint32_t cBytes = (taskM_ / 2U) * taskN_ * sizeof(xType);
const uint32_t vBytes = kBlockRows * taskN_ * sizeof(xType);
pipe_->InitBuffer(gateQ_, 1, cBytes);
pipe_->InitBuffer(upQ_, 1, cBytes);
pipe_->InitBuffer(bias1Q_, 1, taskN_ * sizeof(xType));
pipe_->InitBuffer(bias2Q_, 1, taskN_ * sizeof(xType));
pipe_->InitBuffer(outQ_, 1, vBytes);
pipe_->InitBuffer(gateBuf_, vBytes);
pipe_->InitBuffer(upBuf_, vBytes);
pipe_->InitBuffer(negBuf_, vBytes);
pipe_->InitBuffer(expBuf_, vBytes);
pipe_->InitBuffer(onesBuf_, vBytes);
pipe_->InitBuffer(sigmoidBuf_, vBytes);
if constexpr (padded_) { pipe_->InitBuffer(packQ_, 1, packRows_ * computeD_ * sizeof(xType)); }
}
}
__aicore__ inline void Process()
{
if (core_ >= cores_) { return; }
if constexpr (padded_) {
if ASCEND_IS_AIV {
PackMatrix(paddedX_, x_, m_, m_);
PackMatrix(paddedW1_, w1_, d_, computeD_);
PackMatrix(paddedW2_, w2_, d_, computeD_);
SyncAll();
CrossCoreSetFlag<0x2, PIPE_MTE3>(0x4);
}
if ASCEND_IS_AIC { CrossCoreWaitFlag(0x4); }
}
uint32_t local = 0;
for (uint32_t task = core_; task < taskNum_; task += cores_, ++local) {
const uint32_t slot = local % kSlotCount;
const uint64_t row = static_cast<uint64_t>(task / tasksN_) * taskM_;
const uint32_t col = (task % tasksN_) * taskN_;
const uint32_t rows = Min(taskM_, static_cast<uint32_t>(m_ - row));
const uint32_t cols = Min(taskN_, d_ - col);
const uint64_t off = (static_cast<uint64_t>(core_) * kSlotCount + slot) * 2 * plane_;
if ASCEND_IS_AIC {
if (local >= kSlotCount) { WaitConsumed(slot); }
if constexpr (Direct) {
direct.Compute(x_[row * d_], w1_[static_cast<uint64_t>(col) * d_],
w2_[static_cast<uint64_t>(col) * d_], slots_[off], rows, cols);
} else {
if constexpr (padded_) {
const uint32_t computeCols = Min(taskN_, computeD_ - col);
if constexpr (merged_) {
Projection(paddedW1_, row, 0, rows, 2U * computeCols, off);
} else {
Projection(paddedW1_, row, col, rows, computeCols, off);
Projection(paddedW2_, row, col, rows, computeCols, off + plane_);
}
} else {
Projection(w1_, row, col, rows, cols, off);
Projection(w2_, row, col, rows, cols, off + plane_);
}
}
SignalReady(slot);
}
if ASCEND_IS_AIV {
WaitReady(slot);
Epilogue(off, row, col, rows, cols);
SignalConsumed(slot);
}
}
if ASCEND_IS_AIC {
for (uint32_t slot = 0; slot < Min(local, kSlotCount); ++slot) { WaitConsumed(slot); }
}
}
private:
__aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }
__aicore__ inline void PackMatrix(const GlobalTensor<xType> &dst,
const GlobalTensor<xType> &src,
uint64_t realRows, uint64_t totalRows)
{
const uint64_t chunks = (totalRows + packRows_ - 1U) / packRows_;
for (uint64_t block = GetBlockIdx(); block < chunks; block += cores_ * 2U) {
const uint64_t row = block * packRows_;
const uint32_t nr = Min(packRows_, static_cast<uint32_t>(totalRows - row));
const uint32_t valid = row < realRows ? Min(nr, static_cast<uint32_t>(realRows - row)) : 0;
LocalTensor<xType> tile = packQ_.AllocTensor<xType>();
if (valid != 0) {
DataCopyExtParams copy{static_cast<uint16_t>(valid), d_ * 2U, 0, 0, 0};
DataCopyPadExtParams<xType> pad{true, 0, static_cast<uint8_t>(computeD_ - d_), 0};
DataCopyPad(tile, src[row * d_], copy, pad);
}
if (valid < nr) {
Duplicate(tile[valid * computeD_], static_cast<xType>(0), (nr - valid) * computeD_);
const event_t event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
SetFlag<HardEvent::V_MTE3>(event);
WaitFlag<HardEvent::V_MTE3>(event);
}
packQ_.EnQue(tile);
tile = packQ_.DeQue<xType>();
DataCopy(dst[row * computeD_], tile, nr * computeD_);
packQ_.FreeTensor(tile);
}
}
__aicore__ inline void Projection(const GlobalTensor<xType> &weight, uint64_t row, uint32_t col,
uint32_t rows, uint32_t cols, uint64_t off)
{
mm.SetTail(rows, cols, computeD_);
if constexpr (padded_) { mm.SetTensorA(paddedX_[row * computeD_], false); }
else { mm.SetTensorA(x_[row * d_], false); }
mm.SetTensorB(weight[static_cast<uint64_t>(col) * computeD_], true);
mm.IterateAll(slots_[off], false);
mm.End();
PipeBarrier<PIPE_ALL>();
}
__aicore__ inline void Epilogue(uint64_t off, uint64_t row, uint32_t col,
uint32_t rows, uint32_t cols)
{
const uint32_t halfRows = (rows + 1U) / 2U;
const uint32_t begin = sub_ * halfRows;
if (begin >= rows) { return; }
const uint32_t count = Min(halfRows, rows - begin);
const uint32_t pitch = AlignUp16(cols);
const uint32_t cPitch = merged_ ? 2U * taskN_ : taskN_;
LocalTensor<xType> gate = gateQ_.AllocTensor<xType>();
LocalTensor<xType> up = upQ_.AllocTensor<xType>();
DataCopyExtParams cParams{static_cast<uint16_t>(count), cols * static_cast<uint32_t>(sizeof(xType)),
(cPitch - cols) * static_cast<uint32_t>(sizeof(xType)), 0, 0};
DataCopyPadExtParams<xType> pad{false, 0, 0, 0};
const uint64_t cOff = off + static_cast<uint64_t>(begin) * cPitch;
DataCopyPad(gate, slots_[cOff], cParams, pad);
DataCopyPad(up, slots_[cOff + (merged_ ? taskN_ : plane_)], cParams, pad);
gateQ_.EnQue(gate);
upQ_.EnQue(up);
LocalTensor<xType> b1 = bias1Q_.AllocTensor<xType>();
LocalTensor<xType> b2 = bias2Q_.AllocTensor<xType>();
DataCopyExtParams biasParams{1, cols * static_cast<uint32_t>(sizeof(xType)), 0, 0, 0};
DataCopyPad(b1, b1_[col], biasParams, pad);
DataCopyPad(b2, b2_[col], biasParams, pad);
bias1Q_.EnQue(b1);
bias2Q_.EnQue(b2);
gate = gateQ_.DeQue<xType>();
up = upQ_.DeQue<xType>();
b1 = bias1Q_.DeQue<xType>();
b2 = bias2Q_.DeQue<xType>();
LocalTensor<xType> a = gateBuf_.Get<xType>();
LocalTensor<xType> b = upBuf_.Get<xType>();
LocalTensor<xType> neg = negBuf_.Get<xType>();
LocalTensor<xType> exp = expBuf_.Get<xType>();
LocalTensor<xType> ones = onesBuf_.Get<xType>();
LocalTensor<xType> sig = sigmoidBuf_.Get<xType>();
for (uint32_t group = 0; group < count; group += kBlockRows) {
const uint32_t nr = Min(kBlockRows, count - group);
const uint32_t n = nr * pitch;
const uint8_t rowStride = static_cast<uint8_t>(pitch / 16U);
const BinaryRepeatParams biasRepeat{1, 1, 1, rowStride, rowStride, 0};
for (uint32_t k = 0; k < cols; k += 128U) {
const uint32_t width = Min(128U, cols - k);
Add(a[k], gate[group * pitch + k], b1[k], static_cast<uint64_t>(width),
static_cast<uint8_t>(nr), biasRepeat);
Add(b[k], up[group * pitch + k], b2[k], static_cast<uint64_t>(width),
static_cast<uint8_t>(nr), biasRepeat);
}
LocalTensor<xType> out = outQ_.AllocTensor<xType>();
GateCounter<xType,true>(a,b,exp,out,ones,n);
outQ_.EnQue(out);
out = outQ_.DeQue<xType>();
DataCopyExtParams outParams{static_cast<uint16_t>(nr), cols * static_cast<uint32_t>(sizeof(xType)),
0, (d_ - cols) * static_cast<uint32_t>(sizeof(xType)), 0};
DataCopyPad(y_[(row + begin + group) * d_ + col], out, outParams);
outQ_.FreeTensor(out);
}
gateQ_.FreeTensor(gate);
upQ_.FreeTensor(up);
bias1Q_.FreeTensor(b1);
bias2Q_.FreeTensor(b2);
}
__aicore__ inline void WaitReady(uint32_t s) {
if (s == 0) { CrossCoreWaitFlag(0x6); }
else if (s == 1) { CrossCoreWaitFlag(0x7); }
else if (s == 2) { CrossCoreWaitFlag(0xB); }
else { CrossCoreWaitFlag(0xE); }
}
__aicore__ inline void SignalReady(uint32_t s) {
if (s == 0) { CrossCoreSetFlag<0x2, PIPE_FIX>(0x6); }
else if (s == 1) { CrossCoreSetFlag<0x2, PIPE_FIX>(0x7); }
else if (s == 2) { CrossCoreSetFlag<0x2, PIPE_FIX>(0xB); }
else { CrossCoreSetFlag<0x2, PIPE_FIX>(0xE); }
}
__aicore__ inline void WaitConsumed(uint32_t s) {
if (s == 0) { CrossCoreWaitFlag(0x8); }
else if (s == 1) { CrossCoreWaitFlag(0x9); }
else if (s == 2) { CrossCoreWaitFlag(0xC); }
else { CrossCoreWaitFlag(0xF); }
}
__aicore__ inline void SignalConsumed(uint32_t s) {
if (s == 0) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0x8); }
else if (s == 1) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0x9); }
else if (s == 2) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0xC); }
else { CrossCoreSetFlag<0x2, PIPE_MTE3>(0xF); }
}
TPipe *pipe_ = nullptr;
TQue<QuePosition::VECIN, 1> gateQ_, upQ_, bias1Q_, bias2Q_;
TQue<QuePosition::VECOUT, 1> outQ_;
TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> packQ_;
TBuf<TPosition::VECCALC> gateBuf_, upBuf_, negBuf_, expBuf_, onesBuf_, sigmoidBuf_;
GlobalTensor<xType> x_, w1_, w2_, b1_, b2_, y_, slots_;
GlobalTensor<xType> paddedX_, paddedW1_, paddedW2_;
uint64_t m_ = 0;
uint32_t d_ = 0, taskM_ = 0, taskN_ = 0, taskNum_ = 0, tasksN_ = 0;
uint32_t core_ = 0, sub_ = 0, cores_ = 0, plane_ = 0;
uint32_t computeD_ = 0, packRows_ = 0;
static constexpr bool padded_ = Padded;
static constexpr bool merged_ = Merged || Direct;
};
template <typename T, bool Padded = false, bool Merged = false>
__aicore__ inline void RunCube(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
GM_ADDR workspace, TPipe *pipe, const SwiGluTilingData *td)
{
SwiGluCubeImpl<T, CFG_NORM, false, Padded, Merged> op;
op.Init(x, w1, b1, w2, b2, y, workspace, pipe, td);
op.Process();
}
template <typename T, uint32_t TileM, uint32_t FixedD = 0, bool Unaligned = false, bool Resident = false, bool FullTile = false>
class SlimDirectCube {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
GM_ADDR workspace, const SwiGluDirectTilingData *td) {
d_ = FixedD != 0 ? FixedD : static_cast<uint32_t>(td->d);
n_ = FixedD != 0 ? (FixedD < 64U ? FixedD : 64U) : td->taskN;
const uint32_t cores = td->usedCoreNum;
uint32_t core = GetBlockIdx();
if ASCEND_IS_AIV { core /= 2U; }
x_.SetGlobalBuffer((__gm__ T *)x, td->m * d_);
w1_.SetGlobalBuffer((__gm__ T *)w1, static_cast<uint64_t>(d_) * d_);
w2_.SetGlobalBuffer((__gm__ T *)w2, static_cast<uint64_t>(d_) * d_);
b1_.SetGlobalBuffer((__gm__ T *)b1, d_);
b2_.SetGlobalBuffer((__gm__ T *)b2, d_);
y_.SetGlobalBuffer((__gm__ T *)y, td->m * d_);
const uint32_t plane = tm_ * n_;
slots_.SetGlobalBuffer((__gm__ T *)workspace, static_cast<uint64_t>(2U * plane) * 4U * cores);
if ASCEND_IS_AIC { cube_.Init(d_, tm_, n_); }
SetAtomicNone();
if ASCEND_IS_AIV {
SetMaskNorm();
SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
}
const uint32_t tasksN = (d_ + n_ - 1U) / n_;
if constexpr (Resident) {
const uint32_t col=core%tasksN*n_;
const uint32_t cols=FullTile?n_:Min(n_,d_-col);
if ASCEND_IS_AIC {
cube_.PrepareWeights(w1_[static_cast<uint64_t>(col)*d_],w2_[static_cast<uint64_t>(col)*d_],cols);
}
if ASCEND_IS_AIV {
LocalTensor<T> bias(TPosition::VECIN,tm_*n_*sizeof(T),2U*n_);
if constexpr (Unaligned) {
DataCopyExtParams bp{1,cols*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>((cols+15U)/16U*16U-cols),static_cast<T>(0)};
DataCopyPad(bias,b1_[col],bp,pad);DataCopyPad(bias[n_],b2_[col],bp,pad);
} else {
DataCopy(bias,b1_[col],cols);DataCopy(bias[n_],b2_[col],cols);
}
}
}
uint32_t local = 0;
for (uint32_t task = core; task < td->taskNum; task += cores, ++local) {
const uint32_t slot = local % 4U;
const uint64_t row = static_cast<uint64_t>(task / tasksN) * tm_;
const uint32_t col = task % tasksN * n_;
const uint32_t rows = FullTile?tm_:Min(tm_, td->m - row);
const uint32_t cols = FullTile?n_:Min(n_, d_ - col);
const uint64_t off = (static_cast<uint64_t>(core) * 4U + slot) * 2U * plane;
if ASCEND_IS_AIC {
if (local >= 4U) { WaitConsumed(slot); }
cube_.Compute(x_[row * d_], w1_[static_cast<uint64_t>(col) * d_],
w2_[static_cast<uint64_t>(col) * d_], slots_[off], rows, cols);
Ready(slot);
}
if ASCEND_IS_AIV {
WaitReady(slot);
Epilogue(off, row, col, rows, cols);
Consumed(slot);
}
}
if ASCEND_IS_AIC {
for (uint32_t slot = 0; slot < Min(local, 4U); ++slot) { WaitConsumed(slot); }
}
PipeBarrier<PIPE_ALL>();
}
private:
__aicore__ inline uint32_t Min(uint32_t a, uint64_t b) { return a < b ? a : b; }
template <HardEvent E> __aicore__ inline void Sync() {
constexpr event_t e = EVENT_ID0;
SetFlag<E>(e); WaitFlag<E>(e);
}
__aicore__ inline void Epilogue(uint64_t off, uint64_t row, uint32_t col,
uint32_t rows, uint32_t cols) {
const uint32_t half = FullTile?tm_/2U:(rows + 1U) / 2U;
const uint32_t begin = GetSubBlockIdx() * half;
if constexpr (!FullTile) { if (begin >= rows) { return; } }
const uint32_t nr = FullTile?tm_/2U:Min(half, rows - begin);
const uint32_t capacity = tm_ / 2U * n_;
LocalTensor<T> input(TPosition::VECIN, 0, (tm_ + 2U) * n_);
auto g = input;
auto u = input[capacity];
auto b1 = input[2U * capacity];
auto b2 = input[2U * capacity + n_];
LocalTensor<T> work(TPosition::VECOUT, (tm_ + 2U) * n_ * sizeof(T), tm_ * n_);
auto s = work;
auto out = work[capacity];
const uint64_t start = off + begin * 2U * n_;
if constexpr (Unaligned) {
const uint32_t pitch = (cols + 15U) / 16U * 16U;
DataCopyExtParams cp{static_cast<uint16_t>(nr), cols * 2U,
(2U * n_ - cols) * 2U, (n_ - pitch) / 16U, 0};
DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(pitch - cols), static_cast<T>(0)};
DataCopyPad(g, slots_[start], cp, pad);
DataCopyPad(u, slots_[start + n_], cp, pad);
DataCopyExtParams bp{1, cols * 2U, 0, 0, 0};
if constexpr (!Resident) {
DataCopyPad(b1, b1_[col], bp, pad);
DataCopyPad(b2, b2_[col], bp, pad);
}
} else {
DataCopyParams cp{static_cast<uint16_t>(nr), static_cast<uint16_t>(cols / 16U),
static_cast<uint16_t>((2U * n_ - cols) / 16U),
static_cast<uint16_t>((n_ - cols) / 16U)};
DataCopy(g, slots_[start], cp);
DataCopy(u, slots_[start + n_], cp);
if constexpr (!Resident) {
DataCopy(b1, b1_[col], cols);
DataCopy(b2, b2_[col], cols);
}
}
Sync<HardEvent::MTE2_V>();
const uint8_t stride = static_cast<uint8_t>(n_ / 16U);
const BinaryRepeatParams repeat{1, 1, 1, stride, stride, 0};
Add(g, g, b1, static_cast<uint64_t>(cols), static_cast<uint8_t>(nr), repeat);
Add(u, u, b2, static_cast<uint64_t>(cols), static_cast<uint8_t>(nr), repeat);
PipeBarrier<PIPE_V>();
const uint32_t count = nr * n_;
GateCounter<T,true>(g,u,s,out,out,count);
Sync<HardEvent::V_MTE3>();
if constexpr (Unaligned) {
const uint32_t pitch = (cols + 15U) / 16U * 16U;
DataCopyExtParams output{static_cast<uint16_t>(nr), cols * 2U,
(n_ - pitch) / 16U, (d_ - cols) * 2U, 0};
DataCopyPad(y_[(row + begin) * d_ + col], out, output);
} else {
DataCopyParams output{static_cast<uint16_t>(nr), static_cast<uint16_t>(cols / 16U),
static_cast<uint16_t>((n_ - cols) / 16U),
static_cast<uint16_t>((d_ - cols) / 16U)};
DataCopy(y_[(row + begin) * d_ + col], out, output);
}
Sync<HardEvent::MTE3_MTE2>();
}
__aicore__ inline void WaitReady(uint32_t s) {
if (s == 0) { CrossCoreWaitFlag(0x6); } else if (s == 1) { CrossCoreWaitFlag(0x7); }
else if (s == 2) { CrossCoreWaitFlag(0xB); } else { CrossCoreWaitFlag(0xE); }
}
__aicore__ inline void Ready(uint32_t s) {
if (s == 0) { CrossCoreSetFlag<0x2, PIPE_FIX>(0x6); } else if (s == 1) { CrossCoreSetFlag<0x2, PIPE_FIX>(0x7); }
else if (s == 2) { CrossCoreSetFlag<0x2, PIPE_FIX>(0xB); } else { CrossCoreSetFlag<0x2, PIPE_FIX>(0xE); }
}
__aicore__ inline void WaitConsumed(uint32_t s) {
if (s == 0) { CrossCoreWaitFlag(0x8); } else if (s == 1) { CrossCoreWaitFlag(0x9); }
else if (s == 2) { CrossCoreWaitFlag(0xC); } else { CrossCoreWaitFlag(0xF); }
}
__aicore__ inline void Consumed(uint32_t s) {
if (s == 0) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0x8); } else if (s == 1) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0x9); }
else if (s == 2) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0xC); } else { CrossCoreSetFlag<0x2, PIPE_MTE3>(0xF); }
}
DirectPairCube<T, TileM, FixedD, Unaligned, Resident, FullTile> cube_;
GlobalTensor<T> x_, w1_, w2_, b1_, b2_, y_, slots_;
static constexpr uint32_t tm_ = TileM;
uint32_t d_ = 0, n_ = 0;
};
template <typename T>
class UnalignedTileCube {
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y,GM_ADDR ws) {
uint32_t core=GetBlockIdx();
if ASCEND_IS_AIV {core/=2U;}
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy,tmp;
gx.SetGlobalBuffer((__gm__ T*)x);gw1.SetGlobalBuffer((__gm__ T*)w1);gw2.SetGlobalBuffer((__gm__ T*)w2);
gb1.SetGlobalBuffer((__gm__ T*)b1);gb2.SetGlobalBuffer((__gm__ T*)b2);gy.SetGlobalBuffer((__gm__ T*)y);
tmp.SetGlobalBuffer((__gm__ T*)ws);
const uint32_t off=core*32U*96U;
SetAtomicNone();
if ASCEND_IS_AIC {
cube_.Init(33U,32U,48U);
cube_.Compute(gx[core*17U*33U],gw1,gw2,tmp[off],17U,33U);
CrossCoreSetFlag<0x2,PIPE_FIX>(0x6);
CrossCoreWaitFlag(0x8);
}
if ASCEND_IS_AIV {
const uint32_t sub=GetSubBlockIdx(), nr=9U-sub, begin=9U*sub;
LocalTensor<T> g(TPosition::VECIN,0,432),u(TPosition::VECIN,864,432);
LocalTensor<T> b1t(TPosition::VECIN,1728,48),b2t(TPosition::VECIN,1824,48);
LocalTensor<T> sig(TPosition::VECCALC,1920,432),out(TPosition::VECOUT,2784,432);
SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams bc{1,66,0,0,0};DataCopyPadExtParams<T> pad{true,0,15,static_cast<T>(0)};
DataCopyPad(b1t,gb1,bc,pad);DataCopyPad(b2t,gb2,bc,pad);
CrossCoreWaitFlag(0x6);
DataCopyParams cp{static_cast<uint16_t>(nr),3,3,0};
DataCopy(g,tmp[off+begin*96U],cp);DataCopy(u,tmp[off+begin*96U+48U],cp);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
const BinaryRepeatParams rp{1,1,1,3,3,0};
Add(g,g,b1t,static_cast<uint64_t>(33),static_cast<uint8_t>(nr),rp);
Add(u,u,b2t,static_cast<uint64_t>(33),static_cast<uint8_t>(nr),rp);
PipeBarrier<PIPE_V>();GateCounter<T,true>(g,u,sig,out,out,nr*48U);
SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
DataCopyExtParams oc{static_cast<uint16_t>(nr),66,0,0,0};
DataCopyPad(gy[(core*17U+begin)*33U],out,oc);
CrossCoreSetFlag<0x2,PIPE_MTE3>(0x8);
}
PipeBarrier<PIPE_ALL>();
}
private:
DirectPairCube<T,32U,0U,true,false,true> cube_;
};
template <typename T,uint32_t TM,uint32_t D,uint32_t N>
class DoubleBufferedPairCube {
static constexpr uint32_t A=TM*D,B=2U*N*D;
static constexpr bool DoubleA2=(2U*A*sizeof(T)<=64U*1024U);
static constexpr bool DoubleC=(4U*TM*N*sizeof(float)<=128U*1024U);
static_assert((2U*A+B)*sizeof(T)<=512U*1024U,"L1 capacity");
static_assert(A*sizeof(T)<=64U*1024U && B*sizeof(T)<=64U*1024U,"L0A/B capacity");
public:
__aicore__ inline void PrepareWeights(const GlobalTensor<T>& w1,const GlobalTensor<T>& w2,const GlobalTensor<T>& x0) {
LocalTensor<T> b1(TPosition::B1,2U*A*sizeof(T),B),b2(TPosition::B2,0,B);
LocalTensor<T> a1(TPosition::A1,0,A);
const Nd2NzParams bp{1,N,D,0,D,2U*N,1,0};
const Nd2NzParams ap{1,TM,D,0,D,TM,1,0};
DataCopy(b1,w1,bp);DataCopy(b1[N*16U],w2,bp);
SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
DataCopy(a1,x0,ap);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
LoadData2dParams bl;bl.repeatTimes=B/256U;bl.srcStride=1;
LoadData(b2,b1,bl);
// The first A load and its MTE1_M fence also order the preceding resident B load.
}
template<uint32_t I,uint32_t CurTM=TM> __aicore__ inline void Prefetch(const GlobalTensor<T>& x) {
constexpr event_t slot=I%2U==0U?EVENT_ID0:EVENT_ID1;
if constexpr(I>=2U) {WaitFlag<HardEvent::MTE1_MTE2>(slot);}
LocalTensor<T> a1(TPosition::A1,(I%2U)*A*sizeof(T),CurTM*D);
const Nd2NzParams ap{1,CurTM,D,0,D,CurTM,1,0};
DataCopy(a1,x,ap);SetFlag<HardEvent::MTE2_MTE1>(slot);
}
template<uint32_t I,uint32_t CurTM=TM> __aicore__ inline void Compute(const GlobalTensor<T>& dst) {
constexpr event_t slot=I%2U==0U?EVENT_ID0:EVENT_ID1;
constexpr event_t cSlot=DoubleC?slot:EVENT_ID0;
LocalTensor<T> a1(TPosition::A1,(I%2U)*A*sizeof(T),CurTM*D);
LocalTensor<T> b2(TPosition::B2,0,B);
if constexpr (DoubleA2) {
LocalTensor<T> a2(TPosition::A2,(I%2U)*A*sizeof(T),CurTM*D);
LocalTensor<float> c(TPosition::CO1,DoubleC?(I%2U)*2U*TM*N*sizeof(float):0U,2U*CurTM*N);
WaitFlag<HardEvent::MTE2_MTE1>(slot);
if constexpr(I>=2U) {WaitFlag<HardEvent::M_MTE1>(slot);}
LoadData2dParams al;al.repeatTimes=D/16U;al.srcStride=CurTM/16U;
#pragma unroll
for(uint32_t mb=0;mb<CurTM/16U;++mb) {LoadData(a2[mb*D*16U],a1[mb*256U],al);}
SetFlag<HardEvent::MTE1_MTE2>(slot);
SetFlag<HardEvent::MTE1_M>(slot);WaitFlag<HardEvent::MTE1_M>(slot);
if constexpr(DoubleC ? (I>=2U) : (I>0U)) {WaitFlag<HardEvent::FIX_M>(cSlot);}
MmadParams mad;mad.m=CurTM;mad.n=2U*N;mad.k=D;mad.cmatrixInitVal=true;
Mmad(c,a2,b2,mad);SetFlag<HardEvent::M_MTE1>(slot);
SetFlag<HardEvent::M_FIX>(cSlot);

} else {
constexpr uint32_t HM=CurTM/2U;
LocalTensor<T> a2_0(TPosition::A2,0U,HM*D);
LocalTensor<T> a2_1(TPosition::A2,HM*D*sizeof(T),HM*D);
LocalTensor<float> c0(TPosition::CO1,(I%2U)*2U*TM*N*sizeof(float),2U*HM*N);
LocalTensor<float> c1(TPosition::CO1,((I%2U)*2U*TM*N+2U*HM*N)*sizeof(float),2U*HM*N);
WaitFlag<HardEvent::MTE2_MTE1>(slot);
if constexpr(I>0U) {WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);}
LoadData2dParams al;al.repeatTimes=D/16U;al.srcStride=CurTM/16U;
#pragma unroll
for(uint32_t mb=0;mb<HM/16U;++mb) {LoadData(a2_0[mb*D*16U],a1[mb*256U],al);}
SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
if constexpr(I>0U) {WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);}
#pragma unroll
for(uint32_t mb=0;mb<HM/16U;++mb) {LoadData(a2_1[mb*D*16U],a1[(mb+HM/16U)*256U],al);}
SetFlag<HardEvent::MTE1_MTE2>(slot);
SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
if constexpr(I>=2U) {WaitFlag<HardEvent::FIX_M>(cSlot);}
MmadParams mad;mad.m=HM;mad.n=2U*N;mad.k=D;mad.cmatrixInitVal=true;
Mmad(c0,a2_0,b2,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
SetFlag<HardEvent::M_FIX>(EVENT_ID0);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
Mmad(c1,a2_1,b2,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
SetFlag<HardEvent::M_FIX>(EVENT_ID1);
FixpipeParamsV220 fix;fix.nSize=2U*N;fix.mSize=HM;fix.srcStride=HM;fix.dstStride=2U*N;fix.quantPre=QuantMode_t::F322F16;
WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
Fixpipe<T,float,CFG_ROW_MAJOR>(dst,c0,fix);
WaitFlag<HardEvent::M_FIX>(EVENT_ID1);
Fixpipe<T,float,CFG_ROW_MAJOR>(dst[HM*2U*N],c1,fix);
SetFlag<HardEvent::FIX_M>(cSlot);
}
}
template<uint32_t I> __aicore__ inline void Store(const GlobalTensor<T>& dst) {
static_assert(DoubleA2 && DoubleC,"Two independent producer buffers");
constexpr event_t slot=(I%2U==0U)?EVENT_ID0:EVENT_ID1;
LocalTensor<float> c(TPosition::CO1,(I%2U)*2U*TM*N*sizeof(float),2U*TM*N);
FixpipeParamsV220 fix;fix.nSize=2U*N;fix.mSize=TM;fix.srcStride=TM;fix.dstStride=2U*N;fix.quantPre=QuantMode_t::F322F16;
WaitFlag<HardEvent::M_FIX>(slot);Fixpipe<T,float,CFG_ROW_MAJOR>(dst,c,fix);SetFlag<HardEvent::FIX_M>(slot);
}
template<uint32_t Iterations> __aicore__ inline void Drain() {
WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
if constexpr(!DoubleA2 || Iterations>=2U) {WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);}
WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
if constexpr(DoubleC && Iterations>=2U) {WaitFlag<HardEvent::FIX_M>(EVENT_ID1);}
WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
if constexpr(Iterations>=2U) {WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);}
}
};
template <typename T,uint32_t D,uint32_t TM,uint32_t Cores,uint32_t Iterations,uint32_t TileN=64U>
class PipelinedCubeSwiGlu {
static constexpr uint32_t N=TileN,TN=D/N,Plane=TM*N,M=TM*Cores/TN*Iterations;
static constexpr uint32_t NR=TM/2U,Cap=NR*N,MaxCap=2U*Cap;
static_assert(D==128U && N==128U && TM==64U && Cores==16U && Iterations==4U,
              "Case5 four producer tiles, three vector batches");
static_assert(180480U<=192U*1024U,"Distinct input/output batches fit UB");
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y,GM_ADDR workspace) {
core_=GetBlockIdx();
if ASCEND_IS_AIV {core_/=2U;}
col_=0U;
x_.SetGlobalBuffer((__gm__ T*)x,M*D);y_.SetGlobalBuffer((__gm__ T*)y,M*D);
w1_.SetGlobalBuffer((__gm__ T*)w1,D*D);w2_.SetGlobalBuffer((__gm__ T*)w2,D*D);
b1_.SetGlobalBuffer((__gm__ T*)b1,D);b2_.SetGlobalBuffer((__gm__ T*)b2,D);
slots_.SetGlobalBuffer((__gm__ T*)workspace,2U*Plane*4U*Cores);
SetAtomicNone();
if ASCEND_IS_AIC {piped_.PrepareWeights(w1_,w2_,x_[core_*TM*D]);}
if ASCEND_IS_AIV {
SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
LocalTensor<T> bias(TPosition::VECIN,65536U,2U*N);
DataCopy(bias,b1_,N);DataCopy(bias[N],b2_,N);
LocalTensor<T> ones(TPosition::VECCALC,123392U,128U);
Duplicate(ones,static_cast<T>(1),128U);
PrefetchBatch<0U,1U>();
}
Step<0U>();
if ASCEND_IS_AIC {piped_.template Drain<Iterations>();}
PipeBarrier<PIPE_ALL>();
}
private:
template<uint32_t Slot> __aicore__ inline void Ready() {
constexpr uint32_t id=Slot==0U?0x6:(Slot==1U?0x7:(Slot==2U?0xB:0xE));
CrossCoreSetFlag<0x2,PIPE_FIX>(id);
}
template<uint32_t Slot> __aicore__ inline void WaitReady() {
constexpr uint32_t id=Slot==0U?0x6:(Slot==1U?0x7:(Slot==2U?0xB:0xE));
CrossCoreWaitFlag(id);
}
template<uint32_t First,uint32_t Count> __aicore__ inline void PrefetchBatch() {
static_assert((First==0U && Count==1U)||(First==1U && Count==2U)||(First==3U && Count==1U));
constexpr event_t ev=First==0U?EVENT_ID0:(First==1U?EVENT_ID1:EVENT_ID3);
WaitReady<First>();
if constexpr(Count==2U) {WaitReady<First+1U>();}
LocalTensor<T> input(TPosition::VECIN,First*2U*Cap*sizeof(T),Count*2U*Cap);
const uint64_t off=(core_*4U+First)*2U*Plane+GetSubBlockIdx()*NR*2U*N;
if constexpr(Count==1U) {DataCopy(input,slots_[off],2U*Cap);}
else {
const DataCopyParams dp{2U,static_cast<uint16_t>(2U*Cap*sizeof(T)/32U),
                           static_cast<uint16_t>(2U*Cap*sizeof(T)/32U),0U};
DataCopy(input,slots_[off],dp);
}
SetFlag<HardEvent::MTE2_V>(ev);
}
template <uint32_t I> __aicore__ inline void Step() {
constexpr uint32_t Slot=I%4U;
const uint64_t row=(core_/TN+I*(Cores/TN))*TM;
const uint64_t off=(core_*4U+Slot)*2U*Plane;
if ASCEND_IS_AIC {
static_assert(Iterations==4U,"One-ahead schedule covers four tiles");
if constexpr(I==0U) {
piped_.template Prefetch<1U>(x_[(row+(Cores/TN)*TM)*D]);
piped_.template Compute<0U>(slots_[off]);
}
if constexpr(I+1U<Iterations) {
if constexpr(I+2U<Iterations) {piped_.template Prefetch<I+2U>(x_[(row+2U*(Cores/TN)*TM)*D]);}
const uint64_t nextOff=(core_*4U+(I+1U))*2U*Plane;
piped_.template Compute<I+1U>(slots_[nextOff]);
}
piped_.template Store<I>(slots_[off]);
Ready<Slot>();
}
if ASCEND_IS_AIV {
if constexpr(I==0U) {EpilogueBatch<0U,1U>();}
else if constexpr(I==1U) {EpilogueBatch<1U,2U>();}
else if constexpr(I==3U) {EpilogueBatch<3U,1U>();}
}
if constexpr (I+1U<Iterations) {Step<I+1U>();}
}

template<uint32_t First,uint32_t Count> __aicore__ inline void EpilogueBatch() {
constexpr event_t ev=First==0U?EVENT_ID0:(First==1U?EVENT_ID1:EVENT_ID3);
constexpr uint32_t BatchCap=Count*Cap,BatchRows=Count*NR;
LocalTensor<T> input(TPosition::VECIN,First*2U*Cap*sizeof(T),2U*BatchCap);
LocalTensor<T> bias(TPosition::VECIN,65536U,2U*N);
LocalTensor<T> g(TPosition::VECCALC,66048U,BatchCap);
LocalTensor<T> u(TPosition::VECCALC,82432U,BatchCap);
LocalTensor<T> sig(TPosition::VECCALC,131328U,BatchCap);
LocalTensor<T> out(TPosition::VECOUT,147712U+First*Cap*sizeof(T),BatchCap);
LocalTensor<T> ones(TPosition::VECCALC,123392U,128U);
WaitFlag<HardEvent::MTE2_V>(ev);
constexpr uint64_t mask=0;
const BinaryRepeatParams rp{1,1,1,N/16U,2U*N/16U,0};
Add<T,false>(g,input,bias,mask,static_cast<uint8_t>(BatchRows),rp);
Add<T,false>(u,input[N],bias[N],mask,static_cast<uint8_t>(BatchRows),rp);
constexpr uint8_t repeats=BatchCap/128U;
const UnaryRepeatParams ur{1,1,8,8};
const BinaryRepeatParams br{1,1,1,8,8,8},broadcast{1,1,1,8,0,8};
Muls<T,false>(sig,g,static_cast<T>(-1),mask,repeats,ur);
Exp<T,false>(sig,sig,mask,repeats,ur);
Adds<T,false>(sig,sig,static_cast<T>(1),mask,repeats,ur);
Div<T,false>(sig,ones,sig,mask,repeats,broadcast);
Mul<T,false>(out,g,sig,mask,repeats,br);
Mul<T,false>(out,out,u,mask,repeats,br);
// A2 CrossCoreWaitFlag blocks later dispatch. Queue this output before waiting for future input.
SetFlag<HardEvent::V_MTE3>(ev);WaitFlag<HardEvent::V_MTE3>(ev);
const uint64_t row=(core_+First*Cores)*TM+GetSubBlockIdx()*NR;
if constexpr(Count==1U) {DataCopy(y_[row*D],out,Cap);}
else {
const DataCopyParams dp{2U,static_cast<uint16_t>(Cap*sizeof(T)/32U),0U,
                           static_cast<uint16_t>((Cores*TM*D-Cap)*sizeof(T)/32U)};
DataCopy(y_[row*D],out,dp);
}
if constexpr(First==0U) {PrefetchBatch<1U,2U>();}
else if constexpr(First==1U) {PrefetchBatch<3U,1U>();}
}
DoubleBufferedPairCube<T,TM,D,N> piped_;
GlobalTensor<T> x_,w1_,w2_,b1_,b2_,y_,slots_;
uint32_t core_=0,col_=0;
};
}
template <typename DT_INPUT_X, int PATH>
__global__ __aicore__ void swi_glu(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2, GM_ADDR bias_2,
GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
if constexpr (PATH>=43) {
if constexpr (PATH==43) {
KERNEL_TASK_TYPE(11009,KERNEL_TYPE_AIV_ONLY);
SubBlockGemv4<DT_INPUT_X> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output);
} else if constexpr (PATH==44) {
KERNEL_TASK_TYPE(11265,KERNEL_TYPE_AIV_ONLY);
AlignedColGemv<DT_INPUT_X,64U,2U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output);
} else if constexpr (PATH==45) {
optiling::PipelinedCubeSwiGlu<DT_INPUT_X,256U,32U,16U,4U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,workspace);
} else if constexpr (PATH==46) {
KERNEL_TASK_TYPE(11777,KERNEL_TYPE_AIV_ONLY);
AlignedColGemv<DT_INPUT_X,32U,1U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output);
} else if constexpr (PATH==47) {
optiling::PipelinedCubeSwiGlu<DT_INPUT_X,128U,64U,16U,4U,128U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,workspace);
} else if constexpr (PATH==48) {
KERNEL_TASK_TYPE(12289,KERNEL_TYPE_AIV_ONLY);
ScalarChannelSwiGlu<DT_INPUT_X> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output);
} else if constexpr (PATH==49) {
KERNEL_TASK_TYPE(12545,KERNEL_TYPE_AIV_ONLY);
PackedSubBlockGemv<DT_INPUT_X> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,tiling);
} else if constexpr (PATH==50) {
KERNEL_TASK_TYPE(12801,KERNEL_TYPE_AIV_ONLY);
PackedMultiBlockGemv<DT_INPUT_X> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,tiling);
} else if constexpr (PATH==51) {
optiling::PipelinedCubeSwiGlu<DT_INPUT_X,256U,128U,16U,16U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,workspace);
}
} else if constexpr (PATH==42) {
KERNEL_TASK_TYPE(10753,KERNEL_TYPE_AIV_ONLY);
REGISTER_TILING_DEFAULT(SwiGluTilingData);
BatchedTinyVectorSwiGlu<DT_INPUT_X,13U,7U,true> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,nullptr);
} else if constexpr (PATH >= 38) {
if constexpr (PATH==38) { KERNEL_TASK_TYPE(9729,KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH==39) { KERNEL_TASK_TYPE(9985,KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH==40) { KERNEL_TASK_TYPE(10241,KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH==41) { KERNEL_TASK_TYPE(10497,KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
if constexpr (PATH==41) {
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData,tiling_data,tiling);
TinyColumnsSwiGlu<DT_INPUT_X,0U,false> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else {
TinyColumnsSwiGlu<DT_INPUT_X,PATH==39?0U:13U,PATH!=40> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,nullptr);
}
} else if constexpr (PATH >= 36) {
if constexpr (PATH==36) { KERNEL_TASK_TYPE(9217,KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH==37) { KERNEL_TASK_TYPE(9473,KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData,tiling_data,tiling);
SmallMVectorSwiGlu<DT_INPUT_X,PATH==36?512U:0U,512U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else if constexpr (PATH >= 32) {
if (workspace == nullptr) { return; }
GM_ADDR userWorkspace=AscendC::GetUserWorkspace(workspace);
if (userWorkspace == nullptr) { return; }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluDirectTilingData,tiling_data,tiling);
constexpr uint32_t fixedD=PATH==32?256U:(PATH==33?128U:0U);
optiling::SlimDirectCube<DT_INPUT_X,64U,fixedD,PATH==35> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,userWorkspace,&tiling_data);
} else if constexpr (PATH >= 26) {
if (workspace == nullptr) { return; }
GM_ADDR userWorkspace=AscendC::GetUserWorkspace(workspace);
if (userWorkspace == nullptr) { return; }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluDirectTilingData,tiling_data,tiling);
constexpr uint32_t fixedD=(PATH==26||PATH==30)?256U:((PATH==27||PATH==31)?128U:0U);
optiling::SlimDirectCube<DT_INPUT_X,128U,fixedD,PATH==29,true,PATH>=30> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,userWorkspace,&tiling_data);
} else if constexpr (PATH >= 23) {
REGISTER_TILING_DEFAULT(SwiGluTilingData);
if constexpr (PATH == 23) { KERNEL_TASK_TYPE(5889, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 24) { KERNEL_TASK_TYPE(6145, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 25) { KERNEL_TASK_TYPE(6401, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 25) {
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData, tiling_data, tiling);
SingleTinySwiGlu<DT_INPUT_X,0U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else {
SingleTinySwiGlu<DT_INPUT_X,PATH==23?13U:1U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,nullptr);
}
} else if constexpr (PATH == 21 || PATH == 22) {
if constexpr (PATH == 21) { KERNEL_TASK_TYPE(5377, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 22) { KERNEL_TASK_TYPE(5633, KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData, tiling_data, tiling);
BatchedTinyVectorSwiGlu<DT_INPUT_X,PATH==21?13U:0U,4U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else if constexpr (PATH >= 17) {
if constexpr (PATH == 17) { KERNEL_TASK_TYPE(4353, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 18) { KERNEL_TASK_TYPE(4609, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 19) { KERNEL_TASK_TYPE(4865, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 20) { KERNEL_TASK_TYPE(5121, KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData, tiling_data, tiling);
constexpr uint32_t fixedD=PATH==17?64U:(PATH==18?128U:(PATH==19?256U:0U));
SmallMVectorSwiGlu<DT_INPUT_X,fixedD> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else if constexpr (PATH >= 14) {
if constexpr (PATH == 14) { KERNEL_TASK_TYPE(3585, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 15) { KERNEL_TASK_TYPE(3841, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 16) { KERNEL_TASK_TYPE(4097, KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData, tiling_data, tiling);
if constexpr (PATH == 15) {
ScalarFeatureSwiGlu<DT_INPUT_X> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else {
TinyVectorSwiGlu<DT_INPUT_X,PATH == 14 ? 13U : 0U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
}
} else if constexpr (PATH == 0 || PATH == 4 || PATH == 5) {
if constexpr (PATH == 0) { KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 4) { KERNEL_TASK_TYPE(1025, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 5) { KERNEL_TASK_TYPE(1281, KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tiling_data, tiling);
constexpr int mode = PATH == 4 ? 1 : (PATH == 5 ? 2 : 0);
KernelSwiGlu<DT_INPUT_X, mode> op;
op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, tiling_data);
op.Process();
} else {
if (workspace == nullptr) { return; }
GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
if (userWorkspace == nullptr) { return; }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
if constexpr (PATH == 2 || PATH == 3 || PATH >= 8) {
GET_TILING_DATA_WITH_STRUCT(SwiGluDirectTilingData, tiling_data, tiling);
constexpr uint32_t tileM = (PATH == 3 || PATH == 9 || PATH == 11 || PATH == 13) ? 128U : 32U;
constexpr uint32_t fixedD = (PATH == 8 || PATH == 9) ? 256U :
((PATH == 10 || PATH == 11) ? 128U : 0U);
optiling::SlimDirectCube<DT_INPUT_X, tileM, fixedD, PATH == 12 || PATH == 13> op;
op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output, userWorkspace, &tiling_data);
} else {
AscendC::TPipe tPipe;
GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tiling_data, tiling);
optiling::RunCube<DT_INPUT_X, PATH == 6 || PATH == 7, PATH == 7>(
input_x, weight_1, bias_1, weight_2, bias_2, output, userWorkspace, &tPipe, &tiling_data);
tPipe.Destroy();
}
}
}
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 11009UL || TILING_KEY_VAR == 11265UL || TILING_KEY_VAR == 11521UL || TILING_KEY_VAR == 11777UL || TILING_KEY_VAR == 12033UL || TILING_KEY_VAR == 12289UL || TILING_KEY_VAR == 12545UL || TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL) && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON)
namespace AscendC {
__aicore__ inline void NoSysWorkspaceForce(GM_ADDR) {}
__aicore__ inline GM_ADDR FastGetUserWorkspace(GM_ADDR workspace) {
return workspace + RESERVED_WORKSPACE;
}
}
#define SetSysWorkspaceForce NoSysWorkspaceForce
#define GetUserWorkspace FastGetUserWorkspace
#define cce_kernel cce_kernel, aligned(512), section(".text.hot")
#if TILING_KEY_VAR == 11009UL || TILING_KEY_VAR == 11265UL || TILING_KEY_VAR == 11777UL || TILING_KEY_VAR == 12289UL || TILING_KEY_VAR == 12545UL || TILING_KEY_VAR == 12801UL
#define set_ffts_base_addr(...) ((void)0)
#endif
#endif
#elif defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)
#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "swi_glu_tiling.h"
#include "tiling_key_swi_glu.h"
using namespace AscendC;
namespace {
__aicore__ inline uint32_t AlignUpBytes(uint32_t bytes)
{
return (bytes + 31U) / 32U * 32U;
}
__aicore__ inline uint32_t AlignUp16(uint32_t elems)
{
return (elems + 15U) / 16U * 16U;
}
constexpr uint32_t kChunk16 = 16;
__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
return (a + b - 1) / b;
}
}
template <typename T, bool FillOnes = false>
__aicore__ inline void GateCounter(const LocalTensor<T>& g,const LocalTensor<T>& u,
const LocalTensor<T>& sig,const LocalTensor<T>& out,
const LocalTensor<T>& ones,uint32_t count) {
constexpr uint64_t mask=0;
const UnaryRepeatParams ur{1,1,8,8};
const BinaryRepeatParams br{1,1,1,8,8,8};
SetMaskCount();SetVectorMask<T,MaskMode::COUNTER>(0,count);

Muls<T,false>(sig,g,static_cast<T>(-1),mask,1,ur);
PipeBarrier<PIPE_V>();Exp<T,false>(sig,sig,mask,1,ur);
PipeBarrier<PIPE_V>();Adds<T,false>(sig,sig,static_cast<T>(1),mask,1,ur);
PipeBarrier<PIPE_V>();Reciprocal<T,false>(sig,sig,mask,1,ur);
PipeBarrier<PIPE_V>();Mul<T,false>(out,g,sig,mask,1,br);
PipeBarrier<PIPE_V>();Mul<T,false>(out,out,u,mask,1,br);
SetMaskNorm();ResetMask();
}
template <typename T,uint32_t Count>
__aicore__ inline void GateNormal(const LocalTensor<T>& g,const LocalTensor<T>& u,
const LocalTensor<T>& sig,const LocalTensor<T>& out,const LocalTensor<T>& ones) {
static_assert(Count<=128U);
constexpr uint64_t lo=Count>=64U?~0ULL:((1ULL<<Count)-1ULL);
constexpr uint64_t hi=Count<=64U?0ULL:((1ULL<<(Count-64U))-1ULL);
const UnaryRepeatParams ur{1,1,8,8};const BinaryRepeatParams br{1,1,1,8,8,8};
SetVectorMask<T,MaskMode::NORMAL>(hi,lo);
Muls<T,false>(sig,g,static_cast<T>(-1),static_cast<uint64_t>(0),1,ur);
PipeBarrier<PIPE_V>();Exp<T,false>(sig,sig,static_cast<uint64_t>(0),1,ur);
PipeBarrier<PIPE_V>();Adds<T,false>(sig,sig,static_cast<T>(1),static_cast<uint64_t>(0),1,ur);
PipeBarrier<PIPE_V>();Div<T,false>(sig,ones,sig,static_cast<uint64_t>(0),1,br);
PipeBarrier<PIPE_V>();Mul<T,false>(out,g,sig,static_cast<uint64_t>(0),1,br);
PipeBarrier<PIPE_V>();Mul<T,false>(out,out,u,static_cast<uint64_t>(0),1,br);
ResetMask();
}
template <typename T, uint32_t Count, bool SetMask = true>
__aicore__ inline void FastGate(const LocalTensor<T>& g, const LocalTensor<T>& u,
const LocalTensor<T>& sig, const LocalTensor<T>& out, const LocalTensor<T>& ones) {
static_assert(Count <= 128U);
constexpr uint64_t lo = Count >= 64U ? ~0ULL : ((1ULL << Count) - 1ULL);
constexpr uint64_t hi = Count <= 64U ? 0ULL : (Count >= 128U ? ~0ULL : ((1ULL << (Count > 64U ? Count - 64U : 0U)) - 1ULL));
const UnaryRepeatParams ur{1, 1, 8, 8};
const BinaryRepeatParams br{1, 1, 1, 8, 8, 8};
if constexpr (SetMask) {
SetVectorMask<T, MaskMode::NORMAL>(hi, lo);
}
Muls<T, false>(sig, g, static_cast<T>(-1), static_cast<uint64_t>(0), 1, ur);
Exp<T, false>(sig, sig, static_cast<uint64_t>(0), 1, ur);
Adds<T, false>(sig, sig, static_cast<T>(1), static_cast<uint64_t>(0), 1, ur);
Div<T, false>(sig, g, sig, static_cast<uint64_t>(0), 1, br);
Mul<T, false>(out, sig, u, static_cast<uint64_t>(0), 1, br);
}
template <class DT_INPUT_X, int Mode>
class KernelSwiGlu {
public:
__aicore__ inline KernelSwiGlu() {}
__aicore__ inline void Init(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2, GM_ADDR bias_2,
GM_ADDR output, const SwiGluTilingData &tiling)
{
td_ = &tiling;
const uint64_t xElems = td_->m * td_->d;
const uint64_t wElems = td_->d * td_->d;
xGm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)input_x, xElems);
w1Gm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)weight_1, wElems);
b1Gm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)bias_1, td_->d);
w2Gm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)weight_2, wElems);
b2Gm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)bias_2, td_->d);
outGm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)output, xElems);
pipe_.InitBuffer(stageA_, 1, AlignUpBytes(td_->dAlign * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(stageB_, 1, AlignUpBytes(td_->dAlign * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(outQueue_, 1, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(xfBuf_, AlignUpBytes(td_->dAlign * sizeof(float)));
pipe_.InitBuffer(wfBuf_, AlignUpBytes(td_->dAlign * sizeof(float)));
const uint32_t productElems = td_->dAlign *
(Mode != 0 && td_->dAlign <= 256 ? kChunk16 : 1U);
pipe_.InitBuffer(prodGBuf_, AlignUpBytes(productElems * sizeof(float)));
pipe_.InitBuffer(prodUBuf_, AlignUpBytes(productElems * sizeof(float)));
pipe_.InitBuffer(accGBuf_, AlignUpBytes(td_->tileN * sizeof(float)));
pipe_.InitBuffer(accUBuf_, AlignUpBytes(td_->tileN * sizeof(float)));
pipe_.InitBuffer(accG16Buf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(accU16Buf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(negBuf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(eBuf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(sBuf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(onesBuf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
if constexpr (Mode != 0) {
pipe_.InitBuffer(w1tBuf_, AlignUpBytes(td_->dAlign * kChunk16 * sizeof(float)));
pipe_.InitBuffer(w2tBuf_, AlignUpBytes(td_->dAlign * kChunk16 * sizeof(float)));
pipe_.InitBuffer(wblkQueue_, 1, AlignUpBytes(kChunk16 * td_->dAlign * sizeof(DT_INPUT_X)));
}
}
__aicore__ inline void Process()
{
const uint64_t core = GetBlockIdx();
if (core >= td_->usedCoreNum) {
return;
}
if constexpr (Mode == 2) {
const uint32_t d = static_cast<uint32_t>(td_->d);
const uint32_t columns = Ceiling(d, kChunk16);
const uint64_t tasks = td_->m * columns;
for (uint64_t task = core; task < tasks; task += td_->usedCoreNum) {
const uint64_t row = task / columns;
const uint32_t c0 = static_cast<uint32_t>(task % columns) * kChunk16;
const uint32_t chunk = d - c0 < kChunk16 ? d - c0 : kChunk16;
PrepareWt(c0, chunk, d);
ProcessColTileT(row, c0, chunk, d);
}
} else if constexpr (Mode == 1) {
const uint32_t d = static_cast<uint32_t>(td_->d);
for (uint32_t c0 = 0; c0 < d; c0 += kChunk16) {
const uint32_t chunk = d - c0 < kChunk16 ? d - c0 : kChunk16;
PrepareWt(c0, chunk, d);
for (uint64_t row = core; row < td_->m; row += td_->usedCoreNum) {
ProcessColTileT(row, c0, chunk, d);
}
}
} else {
for (uint64_t row = core; row < td_->m; row += td_->usedCoreNum) {
ProcessRow(row);
}
}
}
private:
template <HardEvent event>
__aicore__ inline void SyncPipe()
{
const event_t id = static_cast<event_t>(pipe_.FetchEventID(event));
SetFlag<event>(id);
WaitFlag<event>(id);
}
__aicore__ inline void ProcessRow(uint64_t row)
{
const uint32_t d = static_cast<uint32_t>(td_->d);
const uint32_t dAlign = td_->dAlign;
const uint32_t tileN = td_->tileN;
LocalTensor<DT_INPUT_X> xTile = stageA_.AllocTensor<DT_INPUT_X>();
DataCopyExtParams xCopy{1, static_cast<uint32_t>(d * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> xPad{true, 0, static_cast<uint8_t>(dAlign - d),
static_cast<DT_INPUT_X>(0)};
DataCopyPad(xTile, xGm_[row * d], xCopy, xPad);
stageA_.EnQue(xTile);
xTile = stageA_.DeQue<DT_INPUT_X>();
LocalTensor<float> xf = xfBuf_.Get<float>();
Cast(xf, xTile, RoundMode::CAST_NONE, dAlign);
stageA_.FreeTensor(xTile);
for (uint32_t c0 = 0; c0 < d; c0 += tileN) {
const uint32_t chunk = d - c0 < tileN ? d - c0 : tileN;
ProcessColTile(row, c0, chunk, d, dAlign, xf);
}
}
__aicore__ inline void ProcessColTile(uint64_t row, uint32_t c0, uint32_t chunk, uint32_t d, uint32_t dAlign,
const LocalTensor<float> &xf)
{
LocalTensor<float> wf = wfBuf_.Get<float>();
LocalTensor<float> prodG = prodGBuf_.Get<float>();
LocalTensor<float> prodU = prodUBuf_.Get<float>();
LocalTensor<float> accG = accGBuf_.Get<float>();
LocalTensor<float> accU = accUBuf_.Get<float>();
LocalTensor<DT_INPUT_X> accG16 = accG16Buf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> accU16 = accU16Buf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> negBuf = negBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> eBuf = eBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> sBuf = sBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> onesBuf = onesBuf_.Get<DT_INPUT_X>();
DataCopyExtParams wCopy{1, static_cast<uint32_t>(d * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> wPad{true, 0, static_cast<uint8_t>(dAlign - d),
static_cast<DT_INPUT_X>(0)};
for (uint32_t j = 0; j < chunk; ++j) {
const uint64_t wBase = static_cast<uint64_t>(c0 + j) * d;
LocalTensor<DT_INPUT_X> w1 = stageA_.AllocTensor<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> w2 = stageB_.AllocTensor<DT_INPUT_X>();
DataCopyPad(w1, w1Gm_[wBase], wCopy, wPad);
DataCopyPad(w2, w2Gm_[wBase], wCopy, wPad);
stageA_.EnQue(w1);
stageB_.EnQue(w2);
w1 = stageA_.DeQue<DT_INPUT_X>();
w2 = stageB_.DeQue<DT_INPUT_X>();
Cast(wf, w1, RoundMode::CAST_NONE, dAlign);
Mul(prodG, xf, wf, dAlign);
Cast(wf, w2, RoundMode::CAST_NONE, dAlign);
Mul(prodU, xf, wf, dAlign);
stageA_.FreeTensor(w1);
stageB_.FreeTensor(w2);
ReduceSum(accG[j], prodG, wf, dAlign);
ReduceSum(accU[j], prodU, wf, dAlign);
}
DataCopyExtParams bCopy{1, static_cast<uint32_t>(chunk * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> bPad{true, 0, static_cast<uint8_t>(AlignUp16(chunk) - chunk),
static_cast<DT_INPUT_X>(0)};
LocalTensor<DT_INPUT_X> b1Tile = stageA_.AllocTensor<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> b2Tile = stageB_.AllocTensor<DT_INPUT_X>();
DataCopyPad(b1Tile, b1Gm_[c0], bCopy, bPad);
DataCopyPad(b2Tile, b2Gm_[c0], bCopy, bPad);
stageA_.EnQue(b1Tile);
stageB_.EnQue(b2Tile);
b1Tile = stageA_.DeQue<DT_INPUT_X>();
b2Tile = stageB_.DeQue<DT_INPUT_X>();
Cast(accG16, accG, RoundMode::CAST_RINT, chunk);
Cast(accU16, accU, RoundMode::CAST_RINT, chunk);
Add(accG16, accG16, b1Tile, chunk);
stageA_.FreeTensor(b1Tile);
Add(accU16, accU16, b2Tile, chunk);
stageB_.FreeTensor(b2Tile);
Duplicate(onesBuf, static_cast<DT_INPUT_X>(1.0f), chunk);
Muls(negBuf, accG16, (DT_INPUT_X)(-1.0), chunk);
Exp(eBuf, negBuf, chunk);
Adds(eBuf, eBuf, (DT_INPUT_X)(1.0), chunk);
Div(sBuf, onesBuf, eBuf, chunk);
Mul(accG16, accG16, sBuf, chunk);
LocalTensor<DT_INPUT_X> outTile = outQueue_.AllocTensor<DT_INPUT_X>();
Mul(outTile, accG16, accU16, chunk);
outQueue_.EnQue(outTile);
outTile = outQueue_.DeQue<DT_INPUT_X>();
DataCopyExtParams oCopy{1, static_cast<uint32_t>(chunk * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPad(outGm_[row * d + c0], outTile, oCopy);
outQueue_.FreeTensor(outTile);
}
__aicore__ inline void PrepareOneWeight(const GlobalTensor<DT_INPUT_X> &weight,
const LocalTensor<float> &cached,
uint32_t c0, uint32_t chunk, uint32_t d)
{
LocalTensor<DT_INPUT_X> wblk = wblkQueue_.AllocTensor<DT_INPUT_X>();
DataCopyExtParams blkCopy{static_cast<uint16_t>(chunk),
static_cast<uint32_t>(d * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> blkPad{true, 0, static_cast<uint8_t>(td_->dAlign - d),
static_cast<DT_INPUT_X>(0)};
DataCopyPad(wblk, weight[static_cast<uint64_t>(c0) * d], blkCopy, blkPad);
wblkQueue_.EnQue(wblk);
wblk = wblkQueue_.DeQue<DT_INPUT_X>();
Cast(cached, wblk, RoundMode::CAST_NONE, chunk * td_->dAlign);
wblkQueue_.FreeTensor(wblk);
}
__aicore__ inline void PrepareWt(uint32_t c0, uint32_t chunk, uint32_t d)
{
PrepareOneWeight(w1Gm_, w1tBuf_.Get<float>(), c0, chunk, d);
PrepareOneWeight(w2Gm_, w2tBuf_.Get<float>(), c0, chunk, d);
}
__aicore__ inline void ProcessColTileT(uint64_t row, uint32_t c0, uint32_t chunk, uint32_t d)
{
const uint32_t dAlign = td_->dAlign;
LocalTensor<float> xf = xfBuf_.Get<float>();
LocalTensor<float> wf = wfBuf_.Get<float>();
LocalTensor<float> accG = accGBuf_.Get<float>();
LocalTensor<float> accU = accUBuf_.Get<float>();
LocalTensor<DT_INPUT_X> accG16 = accG16Buf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> accU16 = accU16Buf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> negBuf = negBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> eBuf = eBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> sBuf = sBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> onesBuf = onesBuf_.Get<DT_INPUT_X>();
LocalTensor<float> w1t = w1tBuf_.Get<float>();
LocalTensor<float> w2t = w2tBuf_.Get<float>();
LocalTensor<float> prodG = prodGBuf_.Get<float>();
LocalTensor<float> prodU = prodUBuf_.Get<float>();
LocalTensor<DT_INPUT_X> xTile = stageA_.AllocTensor<DT_INPUT_X>();
DataCopyExtParams xCopy{1, static_cast<uint32_t>(d * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> xPad{true, 0, static_cast<uint8_t>(dAlign - d),
static_cast<DT_INPUT_X>(0)};
DataCopyPad(xTile, xGm_[row * d], xCopy, xPad);
stageA_.EnQue(xTile);
xTile = stageA_.DeQue<DT_INPUT_X>();
Cast(xf, xTile, RoundMode::CAST_NONE, dAlign);
stageA_.FreeTensor(xTile);
LocalTensor<DT_INPUT_X> b1Tile = stageA_.AllocTensor<DT_INPUT_X>();
DataCopyExtParams bCopy{1, static_cast<uint32_t>(chunk * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> bPad{true, 0, static_cast<uint8_t>(AlignUp16(chunk) - chunk),
static_cast<DT_INPUT_X>(0)};
DataCopyPad(b1Tile, b1Gm_[c0], bCopy, bPad);
stageA_.EnQue(b1Tile);
b1Tile = stageA_.DeQue<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> b2Tile = stageB_.AllocTensor<DT_INPUT_X>();
DataCopyPad(b2Tile, b2Gm_[c0], bCopy, bPad);
stageB_.EnQue(b2Tile);
b2Tile = stageB_.DeQue<DT_INPUT_X>();
if (dAlign <= 256U) {
const uint8_t rowStride = static_cast<uint8_t>(dAlign / 8U);
const BinaryRepeatParams mulParams{1, 1, 1, rowStride, 0, rowStride};
for (uint32_t k = 0; k < dAlign; k += 64U) {
const uint32_t width = dAlign - k < 64U ? dAlign - k : 64U;
Mul(prodG[k], xf[k], w1t[k], static_cast<uint64_t>(width),
static_cast<uint8_t>(chunk), mulParams);
Mul(prodU[k], xf[k], w2t[k], static_cast<uint64_t>(width),
static_cast<uint8_t>(chunk), mulParams);
if (k == 0) {
WholeReduceSum(accG, prodG[k], width, chunk, 1, 1, rowStride);
WholeReduceSum(accU, prodU[k], width, chunk, 1, 1, rowStride);
} else {
WholeReduceSum(wf, prodG[k], width, chunk, 1, 1, rowStride);
Add(accG, accG, wf, chunk);
WholeReduceSum(wf, prodU[k], width, chunk, 1, 1, rowStride);
Add(accU, accU, wf, chunk);
}
}
} else {
for (uint32_t j = 0; j < chunk; ++j) {
Mul(prodG, xf, w1t[j * dAlign], dAlign);
Mul(prodU, xf, w2t[j * dAlign], dAlign);
ReduceSum(accG[j], prodG, wf, dAlign);
ReduceSum(accU[j], prodU, wf, dAlign);
}
}
Cast(accG16, accG, RoundMode::CAST_RINT, chunk);
Cast(accU16, accU, RoundMode::CAST_RINT, chunk);
Add(accG16, accG16, b1Tile, chunk);
stageA_.FreeTensor(b1Tile);
Add(accU16, accU16, b2Tile, chunk);
stageB_.FreeTensor(b2Tile);
Duplicate(onesBuf, static_cast<DT_INPUT_X>(1.0f), chunk);
Muls(negBuf, accG16, (DT_INPUT_X)(-1.0), chunk);
Exp(eBuf, negBuf, chunk);
Adds(eBuf, eBuf, (DT_INPUT_X)(1.0), chunk);
Div(sBuf, onesBuf, eBuf, chunk);
Mul(accG16, accG16, sBuf, chunk);
LocalTensor<DT_INPUT_X> outTile = outQueue_.AllocTensor<DT_INPUT_X>();
Mul(outTile, accG16, accU16, chunk);
outQueue_.EnQue(outTile);
outTile = outQueue_.DeQue<DT_INPUT_X>();
DataCopyExtParams oCopy{1, static_cast<uint32_t>(chunk * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPad(outGm_[row * d + c0], outTile, oCopy);
outQueue_.FreeTensor(outTile);
}
private:
TPipe pipe_;
TQue<TPosition::VECIN, 1> stageA_;
TQue<TPosition::VECIN, 1> stageB_;
TQue<TPosition::VECOUT, 1> outQueue_;
TBuf<TPosition::VECCALC> xfBuf_;
TBuf<TPosition::VECCALC> wfBuf_;
TBuf<TPosition::VECCALC> prodGBuf_;
TBuf<TPosition::VECCALC> prodUBuf_;
TBuf<TPosition::VECCALC> accGBuf_;
TBuf<TPosition::VECCALC> accUBuf_;
TBuf<TPosition::VECCALC> accG16Buf_;
TBuf<TPosition::VECCALC> accU16Buf_;
TBuf<TPosition::VECCALC> negBuf_;
TBuf<TPosition::VECCALC> eBuf_;
TBuf<TPosition::VECCALC> sBuf_;
TBuf<TPosition::VECCALC> onesBuf_;
TBuf<TPosition::VECCALC> w1tBuf_;
TBuf<TPosition::VECCALC> w2tBuf_;
TQue<QuePosition::VECIN, 1> wblkQueue_;
GlobalTensor<DT_INPUT_X> xGm_;
GlobalTensor<DT_INPUT_X> w1Gm_;
GlobalTensor<DT_INPUT_X> b1Gm_;
GlobalTensor<DT_INPUT_X> w2Gm_;
GlobalTensor<DT_INPUT_X> b2Gm_;
GlobalTensor<DT_INPUT_X> outGm_;
const SwiGluTilingData *td_ = nullptr;
};
template <typename T, uint32_t FixedD>
class TinyVectorSwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
GM_ADDR y, const SwiGluTinyTilingData *td) {
const uint32_t d = FixedD != 0 ? FixedD : static_cast<uint32_t>(td->d);
GlobalTensor<T> gx, gw1, gw2, gb1, gb2, gy;
gx.SetGlobalBuffer((__gm__ T*)x,td->m*d); gy.SetGlobalBuffer((__gm__ T*)y,td->m*d);
gw1.SetGlobalBuffer((__gm__ T*)w1,d*d); gw2.SetGlobalBuffer((__gm__ T*)w2,d*d);
gb1.SetGlobalBuffer((__gm__ T*)b1,d); gb2.SetGlobalBuffer((__gm__ T*)b2,d);
LocalTensor<T> hw1(TPosition::VECIN,0,256), hw2(TPosition::VECIN,512,256);
LocalTensor<T> hx(TPosition::VECIN,1024,16), hb1(TPosition::VECIN,1056,16), hb2(TPosition::VECIN,1088,16);
LocalTensor<float> fw1(TPosition::VECCALC,1152,256), fw2(TPosition::VECCALC,2176,256);
LocalTensor<float> fx(TPosition::VECCALC,3200,16), pg(TPosition::VECCALC,3264,256), pu(TPosition::VECCALC,4288,256);
LocalTensor<float> ag(TPosition::VECCALC,5312,16), au(TPosition::VECCALC,5376,16);
LocalTensor<T> g(TPosition::VECCALC,5440,16), u(TPosition::VECCALC,5472,16);
LocalTensor<T> sig(TPosition::VECCALC,5504,16), ones(TPosition::VECCALC,5536,16), out(TPosition::VECOUT,5568,16);
SetAtomicNone(); SetMaskNorm(); SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams wc{static_cast<uint16_t>(d),d*2U,0,0,0};
DataCopyExtParams rc{1,d*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(16U-d),static_cast<T>(0)};
DataCopyPad(hw1,gw1,wc,pad); DataCopyPad(hw2,gw2,wc,pad);
DataCopyPad(hb1,gb1,rc,pad); DataCopyPad(hb2,gb2,rc,pad);
Sync<HardEvent::MTE2_V>();
Cast(fw1,hw1,RoundMode::CAST_NONE,d*16U); Cast(fw2,hw2,RoundMode::CAST_NONE,d*16U);
Duplicate(ones,static_cast<T>(1),16);
const BinaryRepeatParams rp{1,1,1,2,0,2};
for (uint64_t row=GetBlockIdx();row<td->m;row+=GetBlockNum()) {
DataCopyPad(hx,gx[row*d],rc,pad); Sync<HardEvent::MTE2_V>();
Cast(fx,hx,RoundMode::CAST_NONE,16); PipeBarrier<PIPE_V>();
Mul(pg,fx,fw1,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
Mul(pu,fx,fw2,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
PipeBarrier<PIPE_V>();
WholeReduceSum(ag,pg,16,d,1,1,2); WholeReduceSum(au,pu,16,d,1,1,2);
PipeBarrier<PIPE_V>();
Cast(g,ag,RoundMode::CAST_RINT,d); Cast(u,au,RoundMode::CAST_RINT,d);
PipeBarrier<PIPE_V>(); Add(g,g,hb1,d); Add(u,u,hb2,d);
PipeBarrier<PIPE_V>(); GateCounter<T>(g,u,sig,out,ones,d);
Sync<HardEvent::V_MTE3>(); DataCopyPad(gy[row*d],out,rc);
Sync<HardEvent::MTE3_MTE2>();
}
}
private:
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
};
template <typename T, uint32_t FixedD, uint32_t Rows = 1, bool SingleBatch = false>
class BatchedTinyVectorSwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
GM_ADDR y, const SwiGluTinyTilingData *td) {
const uint32_t d=FixedD!=0?FixedD:static_cast<uint32_t>(td->d);
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gy.SetGlobalBuffer((__gm__ T*)y);
gw1.SetGlobalBuffer((__gm__ T*)w1,d*d);gw2.SetGlobalBuffer((__gm__ T*)w2,d*d);
gb1.SetGlobalBuffer((__gm__ T*)b1,d);gb2.SetGlobalBuffer((__gm__ T*)b2,d);
LocalTensor<T> hw1(TPosition::VECIN,0,256),hw2(TPosition::VECIN,512,256);
LocalTensor<T> hx(TPosition::VECIN,1024,Rows*16),hb1(TPosition::VECIN,1024+Rows*32,16),hb2(TPosition::VECIN,1056+Rows*32,16);
constexpr uint32_t fb=1088+Rows*32;
LocalTensor<float> fw1(TPosition::VECCALC,fb,256),fw2(TPosition::VECCALC,fb+1024,256);
LocalTensor<float> fx(TPosition::VECCALC,fb+2048,Rows*16);
constexpr uint32_t pb=fb+2048+Rows*64;
LocalTensor<float> pg(TPosition::VECCALC,pb,256),pu(TPosition::VECCALC,pb+1024,256);
constexpr uint32_t ab=pb+2048;
LocalTensor<float> ag(TPosition::VECCALC,ab,Rows*16),au(TPosition::VECCALC,ab+Rows*64,Rows*16);
constexpr uint32_t hb=ab+Rows*128;
LocalTensor<T> g(TPosition::VECCALC,hb,Rows*16),u(TPosition::VECCALC,hb+Rows*32,Rows*16);
LocalTensor<T> sig(TPosition::VECCALC,hb+Rows*64,Rows*16),ones(TPosition::VECCALC,hb+Rows*96,Rows*16),out(TPosition::VECOUT,hb+Rows*128,Rows*16);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams wc{static_cast<uint16_t>(d),d*2U,0,0,0},bc{1,d*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(16U-d),static_cast<T>(0)};
DataCopyPad(hw1,gw1,wc,pad);DataCopyPad(hw2,gw2,wc,pad);
DataCopyPad(hb1,gb1,bc,pad);DataCopyPad(hb2,gb2,bc,pad);
if constexpr (SingleBatch) {
DataCopyExtParams rc{static_cast<uint16_t>(Rows),d*2U,0,0,0};
DataCopyPad(hx,gx[static_cast<uint64_t>(GetBlockIdx())*Rows*d],rc,pad);
}
Sync<HardEvent::MTE2_V>();
Cast(fw1,hw1,RoundMode::CAST_NONE,d*16U);Cast(fw2,hw2,RoundMode::CAST_NONE,d*16U);
Duplicate(ones,static_cast<T>(1),Rows*16U);
Duplicate(ag,0.0f,Rows*16U);Duplicate(au,0.0f,Rows*16U);
const BinaryRepeatParams rp{1,1,1,2,0,2};
const BinaryRepeatParams bp{1,1,1,1,1,0};
const uint64_t first=static_cast<uint64_t>(GetBlockIdx())*Rows;
const uint64_t stride=SingleBatch?0U:static_cast<uint64_t>(GetBlockNum())*Rows;
uint64_t groups=1U;
if constexpr (!SingleBatch) { groups=(td->m-first+stride-1U)/stride; }
for (uint64_t group=0;group<groups;++group) {
const uint64_t row=first+group*stride;
const uint32_t nr=SingleBatch?Rows:(td->m-row<Rows?static_cast<uint32_t>(td->m-row):Rows);
DataCopyExtParams rc{static_cast<uint16_t>(nr),d*2U,0,0,0};
if constexpr (!SingleBatch) { DataCopyPad(hx,gx[row*d],rc,pad);Sync<HardEvent::MTE2_V>(); }
Cast(fx,hx,RoundMode::CAST_NONE,nr*16U);PipeBarrier<PIPE_V>();
#pragma unroll
for (uint32_t i=0;i<nr;++i) {
Mul(pg,fx[i*16U],fw1,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
Mul(pu,fx[i*16U],fw2,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
PipeBarrier<PIPE_V>();
WholeReduceSum(ag[i*16U],pg,16,d,1,1,2);WholeReduceSum(au[i*16U],pu,16,d,1,1,2);
PipeBarrier<PIPE_V>();
}
Cast(g,ag,RoundMode::CAST_RINT,nr*16U);Cast(u,au,RoundMode::CAST_RINT,nr*16U);
PipeBarrier<PIPE_V>();
Add(g,g,hb1,static_cast<uint64_t>(d),static_cast<uint8_t>(nr),bp);
Add(u,u,hb2,static_cast<uint64_t>(d),static_cast<uint8_t>(nr),bp);
PipeBarrier<PIPE_V>();GateCounter<T>(g,u,sig,out,ones,nr*16U);
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[row*d],out,rc);
if constexpr (!SingleBatch) { Sync<HardEvent::MTE3_MTE2>(); }
}
if constexpr (SingleBatch) { PipeBarrier<PIPE_ALL>(); }
}
private:
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0); }
};
template <typename T>
class ScalarFeatureSwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
GM_ADDR y, const SwiGluTinyTilingData *td) {
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x,td->m);gy.SetGlobalBuffer((__gm__ T*)y,td->m);
gw1.SetGlobalBuffer((__gm__ T*)w1,1);gw2.SetGlobalBuffer((__gm__ T*)w2,1);
gb1.SetGlobalBuffer((__gm__ T*)b1,1);gb2.SetGlobalBuffer((__gm__ T*)b2,1);
LocalTensor<T> hx(TPosition::VECIN,0,256), params(TPosition::VECIN,512,64);
LocalTensor<float> fx(TPosition::VECCALC,640,256), fg(TPosition::VECCALC,1664,256), fu(TPosition::VECCALC,2688,256);
LocalTensor<T> g(TPosition::VECCALC,3712,256), u(TPosition::VECCALC,4224,256);
LocalTensor<T> sig(TPosition::VECCALC,4736,256), ones(TPosition::VECCALC,5248,256), out(TPosition::VECOUT,5760,256);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams pc{1,2,0,0,0};
DataCopyPadExtParams<T> pp{true,0,15,static_cast<T>(0)};
DataCopyPad(params,gw1,pc,pp);DataCopyPad(params[16],gw2,pc,pp);
DataCopyPad(params[32],gb1,pc,pp);DataCopyPad(params[48],gb2,pc,pp);
Sync<HardEvent::MTE2_S>();
const float a=static_cast<float>(params.GetValue(0)),b=static_cast<float>(params.GetValue(16));
const T c=params.GetValue(32),e=params.GetValue(48);
Duplicate(ones,static_cast<T>(1),256);
for (uint64_t start=static_cast<uint64_t>(GetBlockIdx())*256U;start<td->m;start+=static_cast<uint64_t>(GetBlockNum())*256U) {
const uint32_t n=td->m-start<256U?static_cast<uint32_t>(td->m-start):256U;
DataCopyExtParams cp{1,n*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>((16U-n%16U)%16U),static_cast<T>(0)};
DataCopyPad(hx,gx[start],cp,pad);Sync<HardEvent::MTE2_V>();
Cast(fx,hx,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();
Muls(fg,fx,a,n);Muls(fu,fx,b,n);PipeBarrier<PIPE_V>();
Cast(g,fg,RoundMode::CAST_RINT,n);Cast(u,fu,RoundMode::CAST_RINT,n);
PipeBarrier<PIPE_V>();Adds(g,g,c,n);Adds(u,u,e,n);
PipeBarrier<PIPE_V>(); GateCounter<T>(g,u,sig,out,ones,n);
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[start],out,cp);
Sync<HardEvent::MTE3_MTE2>();
}
}
private:
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
};
template <typename T, uint32_t FixedD, uint32_t DynamicMaxK = 256U>
class SmallMVectorSwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
GM_ADDR y, const SwiGluTinyTilingData *td) {
constexpr uint32_t maxK=FixedD==0?DynamicMaxK:FixedD;
constexpr uint32_t wElems=16U*maxK;
constexpr uint32_t fBase=4U*wElems+2U*maxK+64U;
const uint32_t d=FixedD!=0?FixedD:static_cast<uint32_t>(td->d);
const uint32_t ka=(d+15U)/16U*16U;
const uint32_t columns=(d+15U)/16U;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x,td->m*d);gy.SetGlobalBuffer((__gm__ T*)y,td->m*d);
gw1.SetGlobalBuffer((__gm__ T*)w1,d*d);gw2.SetGlobalBuffer((__gm__ T*)w2,d*d);
gb1.SetGlobalBuffer((__gm__ T*)b1,d);gb2.SetGlobalBuffer((__gm__ T*)b2,d);
LocalTensor<T> hw1(TPosition::VECIN,0,wElems),hw2(TPosition::VECIN,2U*wElems,wElems);
LocalTensor<T> hx(TPosition::VECIN,4U*wElems,maxK);
LocalTensor<T> hb1(TPosition::VECIN,4U*wElems+2U*maxK,16),hb2(TPosition::VECIN,4U*wElems+2U*maxK+32U,16);
LocalTensor<float> fw1(TPosition::VECCALC,fBase,wElems),fw2(TPosition::VECCALC,fBase+4U*wElems,wElems);
LocalTensor<float> pg(TPosition::VECCALC,fBase+8U*wElems,wElems),pu(TPosition::VECCALC,fBase+12U*wElems,wElems);
constexpr uint32_t xBase=fBase+16U*wElems;
LocalTensor<float> fx(TPosition::VECCALC,xBase,maxK);
constexpr uint32_t aBase=xBase+4U*maxK;
LocalTensor<float> ag(TPosition::VECCALC,aBase,16),au(TPosition::VECCALC,aBase+64U,16);
LocalTensor<float> sg(TPosition::VECCALC,aBase+128U,16),su(TPosition::VECCALC,aBase+192U,16);
constexpr uint32_t hBase=aBase+256U;
LocalTensor<T> g(TPosition::VECCALC,hBase,16),u(TPosition::VECCALC,hBase+32U,16);
LocalTensor<T> sig(TPosition::VECCALC,hBase+64U,16),ones(TPosition::VECCALC,hBase+96U,16),out(TPosition::VECOUT,hBase+128U,16);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
Duplicate(ones,static_cast<T>(1),16);
for (uint64_t task=GetBlockIdx();task<td->m*columns;task+=GetBlockNum()) {
const uint64_t row=task/columns;
const uint32_t col=task%columns*16U;
const uint32_t cols=d-col<16U?d-col:16U;
DataCopyExtParams wc{static_cast<uint16_t>(cols),d*2U,0,0,0};
DataCopyExtParams xc{1,d*2U,0,0,0},bc{1,cols*2U,0,0,0};
DataCopyPadExtParams<T> wp{true,0,static_cast<uint8_t>(ka-d),static_cast<T>(0)};
DataCopyPadExtParams<T> bp{true,0,static_cast<uint8_t>(16U-cols),static_cast<T>(0)};
DataCopyPad(hw1,gw1[static_cast<uint64_t>(col)*d],wc,wp);
DataCopyPad(hw2,gw2[static_cast<uint64_t>(col)*d],wc,wp);
DataCopyPad(hx,gx[row*d],xc,wp);
DataCopyPad(hb1,gb1[col],bc,bp);DataCopyPad(hb2,gb2[col],bc,bp);
Sync<HardEvent::MTE2_V>();
Cast(fw1,hw1,RoundMode::CAST_NONE,cols*ka);Cast(fw2,hw2,RoundMode::CAST_NONE,cols*ka);
Cast(fx,hx,RoundMode::CAST_NONE,ka);PipeBarrier<PIPE_V>();
Dot<true>(ag,au,sg,su,pg,pu,fx,fw1,fw2,ka,0,cols);
for (uint32_t k=64U;k<ka;k+=64U) { Dot<false>(ag,au,sg,su,pg,pu,fx,fw1,fw2,ka,k,cols); }
Cast(g,ag,RoundMode::CAST_RINT,cols);Cast(u,au,RoundMode::CAST_RINT,cols);
PipeBarrier<PIPE_V>();Add(g,g,hb1,cols);Add(u,u,hb2,cols);
PipeBarrier<PIPE_V>(); GateCounter<T>(g,u,sig,out,ones,cols);
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[row*d+col],out,bc);
Sync<HardEvent::MTE3_MTE2>();
}
}
private:
template <bool First> __aicore__ inline void Dot(
const LocalTensor<float>& ag,const LocalTensor<float>& au,const LocalTensor<float>& sg,const LocalTensor<float>& su,
const LocalTensor<float>& pg,const LocalTensor<float>& pu,const LocalTensor<float>& fx,
const LocalTensor<float>& fw1,const LocalTensor<float>& fw2,uint32_t ka,uint32_t k,uint32_t cols) {
const uint32_t width=ka-k<64U?ka-k:64U;
const uint8_t stride=static_cast<uint8_t>(ka/8U);
const BinaryRepeatParams rp{1,1,1,stride,0,stride};
Mul(pg[k],fx[k],fw1[k],static_cast<uint64_t>(width),static_cast<uint8_t>(cols),rp);
Mul(pu[k],fx[k],fw2[k],static_cast<uint64_t>(width),static_cast<uint8_t>(cols),rp);
PipeBarrier<PIPE_V>();
if constexpr (First) {
WholeReduceSum(ag,pg[k],width,cols,1,1,stride);WholeReduceSum(au,pu[k],width,cols,1,1,stride);
PipeBarrier<PIPE_V>();
} else {
WholeReduceSum(sg,pg[k],width,cols,1,1,stride);WholeReduceSum(su,pu[k],width,cols,1,1,stride);
PipeBarrier<PIPE_V>();Add(ag,ag,sg,cols);Add(au,au,su,cols);PipeBarrier<PIPE_V>();
}
}
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0); }
};
template <typename T, uint32_t FixedD>
class SingleTinySwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y,const SwiGluTinyTilingData *td) {
const uint32_t d=FixedD!=0?FixedD:static_cast<uint32_t>(td->d);
const uint64_t row=GetBlockIdx();
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gy.SetGlobalBuffer((__gm__ T*)y);
gw1.SetGlobalBuffer((__gm__ T*)w1,d*d);gw2.SetGlobalBuffer((__gm__ T*)w2,d*d);
gb1.SetGlobalBuffer((__gm__ T*)b1,d);gb2.SetGlobalBuffer((__gm__ T*)b2,d);
LocalTensor<T> hw1(TPosition::VECIN,0,256),hw2(TPosition::VECIN,512,256);
LocalTensor<T> hx(TPosition::VECIN,1024,16),hb1(TPosition::VECIN,1056,16),hb2(TPosition::VECIN,1088,16);
LocalTensor<float> fw1(TPosition::VECCALC,1152,256),fw2(TPosition::VECCALC,2176,256);
LocalTensor<float> fx(TPosition::VECCALC,3200,16),pg(TPosition::VECCALC,3264,256),pu(TPosition::VECCALC,4288,256);
LocalTensor<float> ag(TPosition::VECCALC,5312,16),au(TPosition::VECCALC,5376,16);
LocalTensor<T> g(TPosition::VECCALC,5440,16),u(TPosition::VECCALC,5472,16);
LocalTensor<T> sig(TPosition::VECCALC,5504,16),ones(TPosition::VECCALC,5536,16),out(TPosition::VECOUT,5568,16);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams wc{static_cast<uint16_t>(d),d*2U,0,0,0},rc{1,d*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(16U-d),static_cast<T>(0)};
DataCopyPad(hw1,gw1,wc,pad);DataCopyPad(hw2,gw2,wc,pad);
DataCopyPad(hb1,gb1,rc,pad);DataCopyPad(hb2,gb2,rc,pad);DataCopyPad(hx,gx[row*d],rc,pad);
Sync<HardEvent::MTE2_V>();
Duplicate(ones,static_cast<T>(1),16);
if constexpr (FixedD==1) {
Mul(g,hx,hw1,1);Mul(u,hx,hw2,1);
} else {
Cast(fw1,hw1,RoundMode::CAST_NONE,d*16U);Cast(fw2,hw2,RoundMode::CAST_NONE,d*16U);
Cast(fx,hx,RoundMode::CAST_NONE,16);
const BinaryRepeatParams rp{1,1,1,2,0,2};
Mul(pg,fx,fw1,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
Mul(pu,fx,fw2,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
WholeReduceSum(ag,pg,16,d,1,1,2);WholeReduceSum(au,pu,16,d,1,1,2);
Cast(g,ag,RoundMode::CAST_RINT,d);Cast(u,au,RoundMode::CAST_RINT,d);
}
Add(g,g,hb1,d);Add(u,u,hb2,d);
if constexpr (FixedD != 0) {
FastGate<T, FixedD>(g, u, sig, out, ones);
} else {
GateCounter<T>(g, u, sig, out, ones, d);
}
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[row*d],out,rc);
}
private:
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0); }
};
template <typename T,uint32_t FixedD,bool OneRow>
class TinyColumnsSwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y,const SwiGluTinyTilingData* td) {
const uint32_t d=FixedD!=0?FixedD:(OneRow?GetBlockNum():static_cast<uint32_t>(td->d));
const uint32_t task=GetBlockIdx();
const uint64_t row=OneRow?0U:task/d;
const uint32_t col=OneRow?task:task%d;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gw1.SetGlobalBuffer((__gm__ T*)w1);gw2.SetGlobalBuffer((__gm__ T*)w2);
gb1.SetGlobalBuffer((__gm__ T*)b1);gb2.SetGlobalBuffer((__gm__ T*)b2);gy.SetGlobalBuffer((__gm__ T*)y);
LocalTensor<T> hx(TPosition::VECIN,0,16),hw1(TPosition::VECIN,32,16),hw2(TPosition::VECIN,64,16);
LocalTensor<T> hb1(TPosition::VECIN,96,16),hb2(TPosition::VECIN,128,16);
LocalTensor<float> fx(TPosition::VECCALC,192,16),fw1(TPosition::VECCALC,256,16),fw2(TPosition::VECCALC,320,16);
LocalTensor<float> pg(TPosition::VECCALC,384,16),pu(TPosition::VECCALC,448,16);
LocalTensor<float> ag(TPosition::VECCALC,512,8),au(TPosition::VECCALC,544,8);
LocalTensor<T> g(TPosition::VECCALC,576,16),u(TPosition::VECCALC,608,16);
LocalTensor<T> sig(TPosition::VECCALC,640,16),ones(TPosition::VECCALC,672,16),out(TPosition::VECOUT,704,16);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams cp{1,d*2U,0,0,0},bp{1,2,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(16U-d),static_cast<T>(0)};
DataCopyPadExtParams<T> one{true,0,15,static_cast<T>(0)};
DataCopyPad(hx,gx[row*d],cp,pad);DataCopyPad(hw1,gw1[col*d],cp,pad);DataCopyPad(hw2,gw2[col*d],cp,pad);
DataCopyPad(hb1,gb1[col],bp,one);DataCopyPad(hb2,gb2[col],bp,one);
Sync<HardEvent::MTE2_V>();
Cast(fx,hx,RoundMode::CAST_NONE,16);Cast(fw1,hw1,RoundMode::CAST_NONE,16);Cast(fw2,hw2,RoundMode::CAST_NONE,16);
Duplicate(ones,static_cast<T>(1),16);PipeBarrier<PIPE_V>();
Mul(pg,fx,fw1,16);Mul(pu,fx,fw2,16);PipeBarrier<PIPE_V>();
WholeReduceSum(ag,pg,16,1,1,1,2);WholeReduceSum(au,pu,16,1,1,1,2);PipeBarrier<PIPE_V>();
Cast(g,ag,RoundMode::CAST_RINT,1);Cast(u,au,RoundMode::CAST_RINT,1);PipeBarrier<PIPE_V>();
Add(g,g,hb1,1);Add(u,u,hb2,1);PipeBarrier<PIPE_V>();
GateCounter<T>(g,u,sig,out,ones,1);Sync<HardEvent::V_MTE3>();
DataCopyPad(gy[row*d+col],out,bp);PipeBarrier<PIPE_ALL>();
}
private:
template <HardEvent E> __aicore__ inline void Sync() {SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0);}
};
template <typename T>
class SubBlockGemv4 {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y) {
if (GetBlockIdx() != 0U) { return; }
GlobalTensor<T> gx, gw1, gw2, gb1, gb2, gy;
gx.SetGlobalBuffer((__gm__ T*)x, 4);
gw1.SetGlobalBuffer((__gm__ T*)w1, 16);
gw2.SetGlobalBuffer((__gm__ T*)w2, 16);
gb1.SetGlobalBuffer((__gm__ T*)b1, 4);
gb2.SetGlobalBuffer((__gm__ T*)b2, 4);
gy.SetGlobalBuffer((__gm__ T*)y, 4);
LocalTensor<T> hx(TPosition::VECIN, 0, 16);
LocalTensor<T> w(TPosition::VECIN, 32, 32);
LocalTensor<T> bias(TPosition::VECIN, 96, 32);
LocalTensor<float> prod(TPosition::VECCALC, 352, 32);
LocalTensor<float> pairs(TPosition::VECCALC, 480, 16);
LocalTensor<float> acc(TPosition::VECCALC, 544, 40);
LocalTensor<T> g(TPosition::VECCALC, 704, 32);
LocalTensor<T> sig(TPosition::VECCALC, 768, 16);
LocalTensor<T> ones(TPosition::VECCALC, 800, 16);
LocalTensor<T> out(TPosition::VECOUT, 832, 16);
SetAtomicNone();
SetMaskNorm();
const DataCopyExtParams xc{1, 8, 0, 0, 0};
const DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
GlobalTensor<uint64_t> xword;
xword.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(x), 1);
DataCachePreload(xword, static_cast<int16_t>(0));
DataCopy(w, gw1, 16);
DataCopy(w[16], gw2, 16);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
DataCopyPad(bias, gb1, xc, pad);
DataCopyPad(bias[16], gb2, xc, pad);
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
Duplicate(prod, 0.0f, 32);
const uint64_t xv = xword.GetValue(0);
auto xx = hx.template ReinterpretCast<uint64_t>();
xx.SetValue(0, xv);
xx.SetValue(1, xv);
xx.SetValue(2, xv);
xx.SetValue(3, xv);
SetFlag<HardEvent::S_V>(EVENT_ID0);
WaitFlag<HardEvent::S_V>(EVENT_ID0);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
MulAddDst<float, T>(prod, hx, w, static_cast<uint64_t>(16), 2, BinaryRepeatParams{1, 1, 1, 2, 0, 1});
PairReduceSum(pairs, prod, 1, 32, 1, 1, 8);
PairReduceSum(acc, pairs, 2, 8, 1, 1, 1);
constexpr uint64_t mask = 0;
SetVectorMask<float, MaskMode::NORMAL>(0, 15);
Cast<T, float, false>(g, acc, RoundMode::CAST_RINT, mask, 2, UnaryRepeatParams{1, 1, 1, 4});
WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
SetVectorMask<T, MaskMode::NORMAL>(0, 15);
Add<T, false>(g, g, bias, mask, 2, BinaryRepeatParams{1, 1, 1, 1, 1, 1});
FastGate<T, 4U, false>(g, g[16], sig, out, ones);
Sync<HardEvent::V_MTE3>();
DataCopyPad(gy, out, xc);
}
private:
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
};
template <typename T>
class ScalarChannelSwiGlu {
public:
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y) {
        if (GetBlockIdx() != 0U) { return; }
        GlobalTensor<T> gx, gw1, gw2, gb1, gb2, gy;
        gx.SetGlobalBuffer((__gm__ T*)x, 1); gy.SetGlobalBuffer((__gm__ T*)y, 1);
        gw1.SetGlobalBuffer((__gm__ T*)w1, 1); gw2.SetGlobalBuffer((__gm__ T*)w2, 1);
        gb1.SetGlobalBuffer((__gm__ T*)b1, 1); gb2.SetGlobalBuffer((__gm__ T*)b2, 1);

        LocalTensor<T> hx(TPosition::VECIN, 0, 16);
        LocalTensor<T> w(TPosition::VECIN, 32, 32);
        LocalTensor<T> bias(TPosition::VECIN, 96, 32);

        LocalTensor<T> g(TPosition::VECCALC, 160, 32);
        LocalTensor<T> sig(TPosition::VECCALC, 224, 16);
        LocalTensor<T> ones(TPosition::VECCALC, 256, 16);
        LocalTensor<T> out(TPosition::VECOUT, 288, 16);

        SetAtomicNone();
        SetMaskNorm();
        SetVectorMask<T, MaskMode::NORMAL>(0, 1);

        const DataCopyExtParams rc{1, 2, 0, 0, 0};
        const DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        const T xv=gx.GetValue(0);
const T wv1=gw1.GetValue(0),wv2=gw2.GetValue(0);
const T bv1=gb1.GetValue(0),bv2=gb2.GetValue(0);
hx.SetValue(0,xv);w.SetValue(0,wv1);w.SetValue(16,wv2);
bias.SetValue(0,bv1);bias.SetValue(16,bv2);
SetFlag<HardEvent::S_V>(EVENT_ID0);
WaitFlag<HardEvent::S_V>(EVENT_ID0);
constexpr uint64_t mask = 0;
        
        Mul<T, false>(g, hx, w, mask, 2, BinaryRepeatParams{1, 1, 1, 1, 0, 1});
        
        Add<T, false>(g, g, bias, mask, 2, BinaryRepeatParams{1, 1, 1, 1, 1, 1});

        FastGate<T, 1U, false>(g, g[16], sig, out, ones);

        Sync<HardEvent::V_MTE3>();
        DataCopyPad(gy, out, rc);
    }
private:
    template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
};
template <typename T>
class AlignedRowGemv32 {
public:
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y) {
        if (GetBlockIdx() != 0U) { return; }
        GlobalTensor<T> gx, gw1, gw2, gb1, gb2, gy;
        gx.SetGlobalBuffer((__gm__ T*)x, 32);
        gy.SetGlobalBuffer((__gm__ T*)y, 32);
        gw1.SetGlobalBuffer((__gm__ T*)w1, 1024);
        gw2.SetGlobalBuffer((__gm__ T*)w2, 1024);
        gb1.SetGlobalBuffer((__gm__ T*)b1, 32);
        gb2.SetGlobalBuffer((__gm__ T*)b2, 32);

        LocalTensor<T> hw1(TPosition::VECIN, 0, 1024);
        LocalTensor<T> hw2(TPosition::VECIN, 2048, 1024);
        LocalTensor<T> hx(TPosition::VECIN, 4096, 64);
        LocalTensor<T> hb(TPosition::VECIN, 4224, 64);

        LocalTensor<float> pg(TPosition::VECCALC, 4352, 2048);
        LocalTensor<float> pu(TPosition::VECCALC, 12544, 256);
        LocalTensor<float> ag(TPosition::VECCALC, 13568, 64);

        LocalTensor<T> gu(TPosition::VECCALC, 13824, 64);
        LocalTensor<T> sig(TPosition::VECCALC, 13952, 32);
        LocalTensor<T> ones(TPosition::VECCALC, 14016, 32);
        LocalTensor<T> out(TPosition::VECOUT, 14080, 32);

        SetAtomicNone();
        SetMaskNorm();
        constexpr uint64_t mask = 0;
        SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);

        DataCopy(hx, gx, 32);
        DataCopy(hx[32], gx, 32);
        DataCopy(hb, gb1, 32);
        DataCopy(hb[32], gb2, 32);
        DataCopy(hw1, gw1, 1024);
        
        DataCopy(hw2, gw2, 1024);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);

        Duplicate<float, false>(pg, 0.0f, mask, 32U, 1, 8);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
MulAddDst<float, T, false>(pg, hx, hw1, mask, 32U, BinaryRepeatParams{1, 1, 1, 8, 0, 4});
BlockReduceSum<float, false>(pu, pg, 32U, mask, 1, 1, 8);
PairReduceSum<float, false>(pg, pu, 4U, mask, 1, 1, 8);
        PairReduceSum<float, false>(ag, pg, 2U, mask, 1, 1, 8);

        Cast<T, float, false>(gu, ag, RoundMode::CAST_RINT, mask, 1, UnaryRepeatParams{1, 1, 4, 8});
        SetVectorMask<T, MaskMode::NORMAL>(0, ~0ULL);
        Add<T, false>(gu, gu, hb, mask, 1, BinaryRepeatParams{1, 1, 1, 8, 8, 8});

        FastGate<T, 32U, false>(gu, gu[32], sig, out, ones);

        Sync<HardEvent::V_MTE3>();
        DataCopy(gy, out, 32U);
    }
private:
    template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
};
template <typename T, uint32_t FixedD, uint32_t FixedM>
class AlignedColGemv {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
GM_ADDR y) {
const uint64_t task=GetBlockIdx();
if constexpr (FixedM==1U) {
if (task>=2U) { return; }
}
constexpr uint32_t maxK=FixedD;
constexpr uint32_t xCap=64U;
constexpr uint32_t wElems=16U*maxK;
constexpr uint32_t fBase=4U*wElems+2U*xCap+128U;
const uint32_t d=FixedD;
const uint32_t columns=(d+15U)/16U;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x,FixedM*d);gy.SetGlobalBuffer((__gm__ T*)y,FixedM*d);
gw1.SetGlobalBuffer((__gm__ T*)w1,d*d);gw2.SetGlobalBuffer((__gm__ T*)w2,d*d);
gb1.SetGlobalBuffer((__gm__ T*)b1,d);gb2.SetGlobalBuffer((__gm__ T*)b2,d);
LocalTensor<T> hw1(TPosition::VECIN,0,2U*wElems),hw2(TPosition::VECIN,2U*wElems,wElems);
LocalTensor<T> hx(TPosition::VECIN,4U*wElems,xCap);
LocalTensor<T> hb1(TPosition::VECIN,4U*wElems+2U*xCap,32),hb2(TPosition::VECIN,4U*wElems+2U*xCap+32U,16);
LocalTensor<float> pg(TPosition::VECCALC,fBase+8U*wElems,2U*wElems),pu(TPosition::VECCALC,fBase+16U*wElems,wElems);
constexpr uint32_t xBase=fBase+20U*wElems;
constexpr uint32_t aBase=xBase+4U*maxK;
LocalTensor<float> ag(TPosition::VECCALC,aBase,32);
constexpr uint32_t hBase=aBase+256U;
LocalTensor<T> g(TPosition::VECCALC,hBase,32),u(TPosition::VECCALC,hBase+32U,16);
LocalTensor<T> sig(TPosition::VECCALC,hBase+64U,16),ones(TPosition::VECCALC,hBase+96U,16),out(TPosition::VECOUT,hBase+128U,16);
SetAtomicNone();SetMaskNorm();
constexpr uint64_t mask=0;
SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
{
const uint64_t row=FixedM==1U?0U:(FixedD==64U && FixedM==2U ? (task>>2U) : task/columns);
const uint32_t col=(FixedM==1U?task:(FixedD==64U && FixedM==2U ? (task&3U) : task%columns))*16U;
static_assert(FixedD==32U || FixedD==64U);
if constexpr (FixedD==64U) {
DataCopy(hx,gx[row*d],FixedD);
DataCopy(hw1,gw1[static_cast<uint64_t>(col)*d],16U*FixedD);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
DataCopy(hw2,gw2[static_cast<uint64_t>(col)*d],16U*FixedD);
SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
DataCopy(hb1,gb1[col],16U);DataCopy(hb2,gb2[col],16U);
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
Duplicate<float,false>(pg,0.0f,mask,32U,1,8);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
MulAddDst<float,T,false>(pg,hx,hw1,mask,16U,BinaryRepeatParams{1,1,1,8,0,4});
BlockReduceSum<float,false>(pu,pg,16U,mask,1,1,8);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
MulAddDst<float,T,false>(pg[1024],hx,hw2,mask,16U,BinaryRepeatParams{1,1,1,8,0,4});
BlockReduceSum<float,false>(pu[128],pg[1024],16U,mask,1,1,8);
BlockReduceSum<float,false>(ag,pu,4U,mask,1,1,8);
SetVectorMask<float,MaskMode::NORMAL>(0,0xFFFFFFFFULL);
Cast<T,float,false>(g,ag,RoundMode::CAST_RINT,mask,1,UnaryRepeatParams{1,1,4,8});
WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
Add<T,false>(g,g,hb1,mask,1,BinaryRepeatParams{1,1,1,2,2,2});
} else {
DataCopy(hx,gx[row*d],FixedD);
DataCopy(hx[32],gx[row*d],FixedD);
DataCopy(hw1,gw1[static_cast<uint64_t>(col)*d],16U*FixedD);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
DataCopy(hw2,gw2[static_cast<uint64_t>(col)*d],16U*FixedD);
SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
DataCopy(hb1,gb1[col],16U);DataCopy(hb2,gb2[col],16U);
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
Duplicate<float,false>(pg,0.0f,mask,16U,1,8);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
MulAddDst<float,T,false>(pg,hx,hw1,mask,8U,BinaryRepeatParams{1,1,1,8,0,4});
WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
MulAddDst<float,T,false>(pg[512],hx,hw2,mask,8U,BinaryRepeatParams{1,1,1,8,0,4});
BlockReduceSum<float,false>(pu,pg,16U,mask,1,1,8);
PairReduceSum<float,false>(pg,pu,2U,mask,1,1,8);
PairReduceSum<float,false>(ag,pg,1U,mask,1,1,8);
SetVectorMask<float,MaskMode::NORMAL>(0,0xFFFFFFFFULL);
Cast<T,float,false>(g,ag,RoundMode::CAST_RINT,mask,1,UnaryRepeatParams{1,1,4,8});
WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
Add<T,false>(g,g,hb1,mask,1,BinaryRepeatParams{1,1,1,2,2,2});
}
FastGate<T,16U,false>(g,g[16],sig,out,ones);
Sync<HardEvent::V_MTE3>();
if constexpr (FixedM==1U) {
DataCopy(gy[col],out,16U);
} else {
DataCopy(gy[row*d+col],out,16U);
}
}
}
private:
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0); }
};
template <typename T,uint32_t D,uint32_t Rows>
class BatchedRowGemv {
static constexpr uint32_t K=(D+15U)/16U*16U, W=D*K;
static constexpr uint32_t HB=4U*W+2U*Rows*K;
static constexpr uint32_t FB=HB+4U*K;
static constexpr uint32_t XB=FB+8U*W, PB=XB+4U*Rows*K;
static constexpr uint32_t AB=PB+8U*W, GB=AB+8U*Rows*K;
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y) {
const uint64_t row=static_cast<uint64_t>(GetBlockIdx())*Rows;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gw1.SetGlobalBuffer((__gm__ T*)w1);gw2.SetGlobalBuffer((__gm__ T*)w2);
gb1.SetGlobalBuffer((__gm__ T*)b1);gb2.SetGlobalBuffer((__gm__ T*)b2);gy.SetGlobalBuffer((__gm__ T*)y);
LocalTensor<T> hw1(TPosition::VECIN,0,W),hw2(TPosition::VECIN,2U*W,W),hx(TPosition::VECIN,4U*W,Rows*K);
LocalTensor<T> hb1(TPosition::VECIN,HB,K),hb2(TPosition::VECIN,HB+2U*K,K);
LocalTensor<float> fw1(TPosition::VECCALC,FB,W),fw2(TPosition::VECCALC,FB+4U*W,W);
LocalTensor<float> fx(TPosition::VECCALC,XB,Rows*K),pg(TPosition::VECCALC,PB,W),pu(TPosition::VECCALC,PB+4U*W,W);
LocalTensor<float> ag(TPosition::VECCALC,AB,Rows*K),au(TPosition::VECCALC,AB+4U*Rows*K,Rows*K);
LocalTensor<T> g(TPosition::VECCALC,GB,Rows*K),u(TPosition::VECCALC,GB+2U*Rows*K,Rows*K);
LocalTensor<T> sig(TPosition::VECCALC,GB+4U*Rows*K,Rows*K),ones(TPosition::VECCALC,GB+6U*Rows*K,Rows*K);
LocalTensor<T> out(TPosition::VECOUT,GB+8U*Rows*K,Rows*K);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams wc{D,D*2U,0,0,0},bc{1,D*2U,0,0,0},xc{Rows,D*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(K-D),static_cast<T>(0)};
DataCopyPad(hw1,gw1,wc,pad);DataCopyPad(hw2,gw2,wc,pad);
DataCopyPad(hb1,gb1,bc,pad);DataCopyPad(hb2,gb2,bc,pad);DataCopyPad(hx,gx[row*D],xc,pad);
Sync<HardEvent::MTE2_V>();
Cast(fw1,hw1,RoundMode::CAST_NONE,W);Cast(fw2,hw2,RoundMode::CAST_NONE,W);
Cast(fx,hx,RoundMode::CAST_NONE,Rows*K);
Duplicate(ag,0.0f,Rows*K);Duplicate(au,0.0f,Rows*K);Duplicate(ones,static_cast<T>(1),Rows*K);
PipeBarrier<PIPE_V>();
const BinaryRepeatParams rp{1,1,1,K/8U,0,K/8U};
#pragma unroll
for (uint32_t i=0;i<Rows;++i) {
Mul(pg,fx[i*K],fw1,static_cast<uint64_t>(K),static_cast<uint8_t>(D),rp);
Mul(pu,fx[i*K],fw2,static_cast<uint64_t>(K),static_cast<uint8_t>(D),rp);
PipeBarrier<PIPE_V>();
WholeReduceSum(ag[i*K],pg,K,D,1,1,K/8U);WholeReduceSum(au[i*K],pu,K,D,1,1,K/8U);
PipeBarrier<PIPE_V>();
}
Cast(g,ag,RoundMode::CAST_RINT,Rows*K);Cast(u,au,RoundMode::CAST_RINT,Rows*K);PipeBarrier<PIPE_V>();
const BinaryRepeatParams bp{1,1,1,K/16U,K/16U,0};
Add(g,g,hb1,static_cast<uint64_t>(D),static_cast<uint8_t>(Rows),bp);
Add(u,u,hb2,static_cast<uint64_t>(D),static_cast<uint8_t>(Rows),bp);
PipeBarrier<PIPE_V>();GateCounter<T>(g,u,sig,out,ones,Rows*K);
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[row*D],out,xc);PipeBarrier<PIPE_ALL>();
}
private:
template <HardEvent E> __aicore__ inline void Sync() {SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0);}
};
template <typename T>
class PackedMultiBlockGemv {
static constexpr uint32_t D=33U,Rows=5U,K=48U,W=D*32U;
static constexpr uint32_t FB=7424U, PB=20032U;
static constexpr uint32_t BKB=PB+11264U*4U, PRB=BKB+1408U*4U;
static constexpr uint32_t AB=PRB+704U*4U, LB=AB+384U*4U, GB=LB+384U*4U;
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y) {
const uint64_t row=static_cast<uint64_t>(GetBlockIdx())*Rows;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gw1.SetGlobalBuffer((__gm__ T*)w1);gw2.SetGlobalBuffer((__gm__ T*)w2);
gb1.SetGlobalBuffer((__gm__ T*)b1);gb2.SetGlobalBuffer((__gm__ T*)b2);gy.SetGlobalBuffer((__gm__ T*)y);
LocalTensor<T> hw1(TPosition::VECIN,0,1280),hw2(TPosition::VECIN,2560,1280),hxAll(TPosition::VECIN,5120,544);
LocalTensor<T> rawB1(TPosition::VECIN,6272,K),rawB2(TPosition::VECIN,6368,K);
LocalTensor<T> hb1(TPosition::VECIN,6464,384),hb2(TPosition::VECIN,6816,192);
LocalTensor<float> fw1(TPosition::VECCALC,FB,1088),fw1Last(TPosition::VECCALC,FB+4352U,192);
LocalTensor<float> fw2(TPosition::VECCALC,FB+5120U,1088),fw2Last(TPosition::VECCALC,FB+9472U,192);
LocalTensor<float> fxLast(TPosition::VECCALC,FB+10240U,224),fx(TPosition::VECCALC,FB+11136U,Rows*64U);
LocalTensor<float> bothP(TPosition::VECCALC,PB,11264),blkP(TPosition::VECCALC,BKB,1408),pairP(TPosition::VECCALC,PRB,704);
LocalTensor<float> ag(TPosition::VECCALC,AB,384);
LocalTensor<float> lastG(TPosition::VECCALC,LB,384),lastU(TPosition::VECCALC,LB+704U,192);
LocalTensor<T> g(TPosition::VECCALC,GB,384),u(TPosition::VECCALC,GB+352U,256);
LocalTensor<T> sig(TPosition::VECCALC,GB+768U,256),ones(TPosition::VECCALC,GB+1280U,256);
LocalTensor<T> out(TPosition::VECOUT,GB+1792U,256);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
LocalTensor<T> rawW1(TPosition::VECIN,79104,1104),rawW2(TPosition::VECIN,81312,1104),rawX(TPosition::VECIN,83520,176);
LocalTensor<uint32_t> offsets(TPosition::VECIN,83968,2240);
LocalTensor<int32_t> offsetsI(TPosition::VECIN,83968,2240);
LocalTensor<T> allHalf(TPosition::VECIN,0,3136);
LocalTensor<float> allF(TPosition::VECCALC,FB,3136);
DataCopyExtParams wc{1,2178,0,0,0},bc{1,66,0,0,0},xcFlat{1,330,0,0,0};
DataCopyPadExtParams<T> wp{true,0,15,static_cast<T>(0)},xp{true,0,11,static_cast<T>(0)};
DataCopyPad(rawX,gx[row*D],xcFlat,xp);
SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
DataCopyPad(rawW1,gw1,wc,wp);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
DataCopyPad(rawW2,gw2,wc,wp);
SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
DataCopyPad(rawB1,gb1,bc,wp);DataCopyPad(rawB2,gb2,bc,wp);
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
constexpr uint64_t mask=0;
auto offsets64=offsets.template ReinterpretCast<uint64_t>();
#pragma unroll
for (uint32_t k=0;k<16U;++k) {
const uint64_t c4=k*4U;
offsets64.SetValue(k,((c4+2U)<<32)|c4);
}
SetFlag<HardEvent::S_V>(EVENT_ID0);
Duplicate<float,false>(bothP[5280],0.0f,mask,6U,1,8);
Duplicate<float,false>(bothP[10912],0.0f,mask,6U,1,8);
Duplicate<int32_t,false>(offsetsI[1024],0,mask,1U,1,8);
Duplicate<int32_t,false>(offsetsI[1408],0,mask,2U,1,8);
Duplicate<int32_t,false>(offsetsI[1992],0,mask,1U,1,8);
SetVectorMask<int32_t,MaskMode::NORMAL>(0,0x1FFFFFFFFULL);
Duplicate<int32_t,false>(offsetsI[1280],64,mask,1U,1,8);
SetVectorMask<int32_t,MaskMode::NORMAL>(0,0x3FFFFFFFEULL);
Duplicate<int32_t,false>(offsetsI[1312],130,mask,1U,1,8);
SetVectorMask<int32_t,MaskMode::NORMAL>(0,0x7FFFFFFFCULL);
Duplicate<int32_t,false>(offsetsI[1344],196,mask,1U,1,8);
SetVectorMask<int32_t,MaskMode::NORMAL>(0,0xFFFFFFFF8ULL);
Duplicate<int32_t,false>(offsetsI[1376],262,mask,1U,1,8);
SetVectorMask<int32_t,MaskMode::NORMAL>(0,0x1FFFFFFFF0ULL);
Duplicate<int32_t,false>(offsetsI[1408],328,mask,1U,1,8);
SetVectorMask<int32_t,MaskMode::NORMAL>(0,0xFFFFFFFFULL);
WaitFlag<HardEvent::S_V>(EVENT_ID0);
const UnaryRepeatParams urI{1,1,8,8};
Adds<int32_t,false>(offsetsI[32],offsetsI[0],66,mask,1U,urI);
Adds<int32_t,false>(offsetsI[1024],offsetsI[0],2112,mask,1U,urI);
Adds<int32_t,false>(offsetsI[1504],offsetsI[0],0,mask,1U,urI);
Adds<int32_t,false>(offsetsI[1536],offsetsI[0],0,mask,1U,urI);
Adds<int32_t,false>(offsetsI[1824],offsetsI[0],0,mask,1U,urI);
SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
Adds<int32_t,false>(offsetsI[64],offsetsI[0],132,mask,1U,urI);
Adds<int32_t,false>(offsetsI[128],offsetsI[0],264,mask,2U,urI);
Adds<int32_t,false>(offsetsI[256],offsetsI[0],528,mask,4U,urI);
Adds<int32_t,false>(offsetsI[512],offsetsI[0],1056,mask,8U,urI);
Adds<int32_t,false>(offsetsI[1568],offsetsI[1504],66,mask,1U,urI);
Adds<int32_t,false>(offsetsI[1632],offsetsI[1504],132,mask,2U,urI);
Adds<int32_t,false>(offsetsI[1760],offsetsI[1504],264,mask,1U,urI);
// Repeat the33-entry address seed directly in the910B compaction unit.
// Only addresses are replicated; input tensors and arithmetic are unchanged.
Duplicate<int32_t,false>(offsetsI[1824],0,mask,3U,1,8);
offsets.SetValue(2208U,64U);
offsets64.SetValue(1040U,0x1FFFFFFFFULL);
offsets64.SetValue(1041U,0ULL);
SetFlag<HardEvent::S_V>(EVENT_ID1);
SetVectorMask<int32_t,MaskMode::NORMAL>(0,0xFFFFFFFFULL);
Adds<int32_t,false>(offsetsI[2176],offsetsI[0],0,mask,1U,urI);
WaitFlag<HardEvent::S_V>(EVENT_ID1);
SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
PipeBarrier<PIPE_V>();
// Explicit per-repeat bitmap avoids hardware counter mode consuming only33 total items.
vreducev2((__ubuf__ uint32_t*)offsets[1824].GetPhyAddr(),
          (__ubuf__ uint32_t*)offsets[2176].GetPhyAddr(),
          (__ubuf__ uint32_t*)offsets[2080].GetPhyAddr(),5,1,0,0,0);
PipeBarrier<PIPE_V>();
Muls<int32_t,false>(offsetsI[1088],offsetsI[1824],33,mask,3U,urI);
Adds<int32_t,false>(offsetsI[1088],offsetsI[1088],64,mask,2U,urI);
SetVectorMask<int32_t,MaskMode::NORMAL>(0,(1ULL<<37)-1ULL);
Adds<int32_t,false>(offsetsI[1216],offsetsI[1216],64,mask,1U,urI);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
Gather(hxAll,rawX,offsets[1280],0,544);
SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
Cast<float,T,false>(allF[2560],allHalf[2560],RoundMode::CAST_NONE,mask,9U,UnaryRepeatParams{1,1,8,4});
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
Gather(hw1,rawW1,offsets,0,1280);
Cast<float,T,false>(allF,allHalf,RoundMode::CAST_NONE,mask,20U,UnaryRepeatParams{1,1,8,4});
const BinaryRepeatParams rp{1,1,1,8,0,8},br8{1,1,1,8,8,8};
#pragma unroll
for (uint32_t i=0;i<Rows;++i) {
Mul<float,false>(bothP[i*W],fx[i*64U],fw1,mask,17U,rp);
}
Mul<float,false>(lastG,fxLast,fw1Last,mask,3U,br8);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
Gather(hw2,rawW2,offsets,0,1280);
Cast<float,T,false>(allF[1280],allHalf[1280],RoundMode::CAST_NONE,mask,20U,UnaryRepeatParams{1,1,8,4});
WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
Gather(hb1,rawB1,offsets[1824],0,176);
Gather(hb2,rawB2,offsets[1824],0,176);
SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
#pragma unroll
for (uint32_t i=0;i<Rows;++i) {
Mul<float,false>(bothP[5632U+i*W],fx[i*64U],fw2,mask,17U,rp);
}
BlockReduceSum<float,false>(blkP,bothP,176U,mask,1,1,8);
PairReduceSum<float,false>(pairP,blkP,22U,mask,1,1,8);
PairReduceSum<float,false>(ag,pairP,11U,mask,1,1,8);
Mul<float,false>(lastU,fxLast,fw2Last,mask,3U,br8);
Add<float,false>(ag,ag,lastG,mask,6U,br8);
Cast<T,float,false>(g,ag,RoundMode::CAST_RINT,mask,6U,UnaryRepeatParams{1,1,4,8});
Add<T,false>(g,g,hb1,mask,3U,br8);
const UnaryRepeatParams ur8{1,1,8,8};
Muls<T,false>(sig,g,static_cast<T>(-1),mask,2U,ur8);
Exp<T,false>(sig,sig,mask,2U,ur8);
Adds<T,false>(sig,sig,static_cast<T>(1),mask,2U,ur8);
Div<T,false>(sig,g,sig,mask,2U,br8);
Mul<T,false>(out,sig,u,mask,2U,br8);
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[row*D],out,xcFlat);
}
private:
template <HardEvent E> __aicore__ inline void Sync() {SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0);}
};
template <typename T>
class PackedSubBlockGemv {
static constexpr uint32_t D=13U,Rows=3U,K=16U,W=256U;
static constexpr uint32_t HB=4U*W+2U*Rows*K;
static constexpr uint32_t FB=HB+4U*K;
static constexpr uint32_t XB=FB+8U*W, PB=XB+4U*Rows*K;
static constexpr uint32_t AB=PB+8U*W, GB=AB+8U*Rows*K;
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y) {
const uint32_t blk=GetBlockIdx();
if (blk>=7U) { return; }
const uint64_t row=static_cast<uint64_t>(blk)*Rows;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gw1.SetGlobalBuffer((__gm__ T*)w1);gw2.SetGlobalBuffer((__gm__ T*)w2);
gb1.SetGlobalBuffer((__gm__ T*)b1);gb2.SetGlobalBuffer((__gm__ T*)b2);gy.SetGlobalBuffer((__gm__ T*)y);
LocalTensor<T> hw1(TPosition::VECIN,0,W),hw2(TPosition::VECIN,2U*W,W);
LocalTensor<T> hb1(TPosition::VECIN,HB,K),hb2(TPosition::VECIN,HB+2U*K,K);
LocalTensor<T> sig(TPosition::VECCALC,GB+4U*Rows*K,Rows*K);
LocalTensor<T> out(TPosition::VECOUT,GB+8U*Rows*K,Rows*K);
LocalTensor<T> packedOut(TPosition::VECOUT,GB+10U*Rows*K,Rows*K);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
LocalTensor<T> rawW1(TPosition::VECIN,8448,176),rawW2(TPosition::VECIN,8960,176),rawX(TPosition::VECIN,9472,48);
LocalTensor<uint32_t> offsets(TPosition::VECIN,13056,512);
LocalTensor<int32_t> offsetsI(TPosition::VECIN,13056,512);
DataCopyExtParams wc{1,338,0,0,0},bc{1,26,0,0,0},xcFlat{1,78,0,0,0};
DataCopyPadExtParams<T> wp{true,0,7,static_cast<T>(0)},bp{true,0,3,static_cast<T>(0)},xp{true,0,9,static_cast<T>(0)};
DataCopyPad(rawX,gx[row*D],xcFlat,xp);
SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
DataCopyPad(rawW1,gw1,wc,wp);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
DataCopyPad(rawW2,gw2,wc,wp);
SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
DataCopyPad(hb1,gb1,bc,bp);DataCopyPad(hb2,gb2,bc,bp);
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
constexpr uint64_t mask=0;
LocalTensor<float> bothP(TPosition::VECCALC,19904,1536);
LocalTensor<float> bothBlock(TPosition::VECCALC,34048,192);
LocalTensor<float> acc(TPosition::VECCALC,AB,96);
LocalTensor<T> bothBias(TPosition::VECIN,HB,32),gu(TPosition::VECCALC,GB,96);
LocalTensor<T> bothHalfW(TPosition::VECIN,0,512),halfExpandedX(TPosition::VECCALC,10752,192);
auto offsets64=offsets.template ReinterpretCast<uint64_t>();
#pragma unroll
for (uint32_t k=0;k<6U;++k) {
const uint64_t c4=k*4U;
const uint64_t valA=((c4+2U)<<32)|c4;
offsets64.SetValue(k,valA);
offsets64.SetValue(224U+k,valA);
offsets64.SetValue(231U+k,valA+((34ULL<<32)|34ULL));
offsets64.SetValue(237U+k,valA+((64ULL<<32)|64ULL));
}
offsets64.SetValue(6U,(338ULL<<32)|24ULL);
offsets64.SetValue(7U,(338ULL<<32)|338ULL);
offsets64.SetValue(230U,(32ULL<<32)|24ULL);
offsets64.SetValue(243U,88ULL);
SetFlag<HardEvent::S_V>(EVENT_ID0);
Duplicate<float,false>(bothP,0.0f,mask,24U,1,8);
Duplicate<int32_t,false>(offsetsI[16],338,mask,3U,1,8);
Duplicate<int32_t,false>(offsetsI[256],78,mask,3U,1,8);
SetVectorMask<int32_t,MaskMode::NORMAL>(0,0xFFFFFFFFFFFFULL);
Duplicate<int32_t,false>(offsetsI[208],338,mask,1U,1,8);
WaitFlag<HardEvent::S_V>(EVENT_ID0);
const UnaryRepeatParams urI{1,1,8,8};
SetVectorMask<int32_t,MaskMode::NORMAL>(0,0x1FFFULL);
Adds<int32_t,false>(offsetsI[16],offsetsI[0],26,mask,1U,urI);
Adds<int32_t,false>(offsetsI[192],offsetsI[0],312,mask,1U,urI);
Adds<int32_t,false>(offsetsI[256],offsetsI[0],0,mask,1U,urI);
Adds<int32_t,false>(offsetsI[272],offsetsI[0],0,mask,1U,urI);
SetVectorMask<int32_t,MaskMode::NORMAL>(0,0x1FFF1FFFULL);
Adds<int32_t,false>(offsetsI[32],offsetsI[0],52,mask,1U,urI);
Adds<int32_t,false>(offsetsI[288],offsetsI[256],0,mask,1U,urI);
SetVectorMask<int32_t,MaskMode::NORMAL>(0,0x1FFF1FFF1FFF1FFFULL);
Adds<int32_t,false>(offsetsI[64],offsetsI[0],104,mask,1U,urI);
Adds<int32_t,false>(offsetsI[128],offsetsI[0],208,mask,1U,urI);
Adds<int32_t,false>(offsetsI[320],offsetsI[256],26,mask,1U,urI);
Adds<int32_t,false>(offsetsI[384],offsetsI[256],52,mask,1U,urI);
SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
Gather(halfExpandedX,rawX,offsets[256],0,192);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
Gather(hw1,rawW1,offsets,0,W);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
Gather(hw2,rawW2,offsets,0,W);
SetVectorMask<float,MaskMode::NORMAL>(0,~0ULL);
const BinaryRepeatParams rp{1,1,1,8,0,4};
#pragma unroll
for (uint32_t i=0;i<Rows;++i) {
MulAddDst<float,T,false>(bothP[i*512U],halfExpandedX[i*64U],bothHalfW,mask,8,rp);
}
BlockReduceSum<float,false>(bothBlock,bothP,24,mask,1,1,8);
PairReduceSum<float,false>(acc,bothBlock,3,mask,1,1,8);
Cast<T,float,false>(gu,acc,RoundMode::CAST_RINT,mask,2U,UnaryRepeatParams{1,1,4,8});
WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
SetVectorMask<T,MaskMode::NORMAL>(0,0xFFFFFFFFULL);
Add<T,false>(gu,gu,bothBias,mask,3,BinaryRepeatParams{1,1,1,2,2,0});
SetVectorMask<T,MaskMode::NORMAL>(0,0xFFFFULL);
const UnaryRepeatParams compactG{1,1,1,2},contiguous{1,1,1,1};
const BinaryRepeatParams gateProduct{1,1,1,1,2,1},upProduct{1,1,1,1,1,2};
Muls<T,false>(sig,gu,static_cast<T>(-1),mask,3,compactG);
Exp<T,false>(sig,sig,mask,3,contiguous);
Adds<T,false>(sig,sig,static_cast<T>(1),mask,3,contiguous);
Div<T,false>(sig,gu,sig,mask,3,gateProduct);
Mul<T,false>(out,sig,gu[16],mask,3,upProduct);
SetVectorMask<T,MaskMode::NORMAL>(0,(1ULL<<(Rows*D))-1ULL);
Gather(packedOut,out,offsets[448],0,Rows*D);
Sync<HardEvent::V_MTE3>();
DataCopyPad(gy[row*D],packedOut,xcFlat);
}
private:
template <HardEvent E> __aicore__ inline void Sync() {SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0);}
};
namespace optiling {
using namespace AscendC;
using namespace matmul;
constexpr uint32_t kSlotCount = 4;
constexpr uint32_t kBlockRows = 16;
__aicore__ constexpr MatmulConfig DirectBasicConfig(uint32_t rows)
{
auto config = GetBasicConfig(rows, 128, 128);
config.enUnitFlag = true;
return config;
}
constexpr MatmulConfig kBasic16 = DirectBasicConfig(16);
constexpr MatmulConfig kBasic32 = DirectBasicConfig(32);
constexpr MatmulConfig kBasic128 = DirectBasicConfig(128);
template <typename T, uint32_t FixedM = 0, uint32_t FixedD = 0, bool Unaligned = false, bool Resident = false, bool FullTile = false, uint32_t FixedN = 0>
class DirectPairCube {
public:
__aicore__ inline void Init(uint32_t k, uint32_t m, uint32_t n) {
realK_ = FixedD != 0 ? FixedD : k;
k_ = Unaligned ? (realK_ + 15U) / 16U * 16U : realK_;
m_ = FixedM != 0 ? FixedM : m;
n_ = FixedN != 0 ? FixedN : (FixedD != 0 ? (FixedD < 64U ? FixedD : 64U) : n);
}
__aicore__ inline void PrepareWeights(const GlobalTensor<T> &w1,const GlobalTensor<T> &w2,uint32_t cols) {
LocalTensor<T> b1(TPosition::B1,m_*k_*sizeof(T),2U*n_*k_);
LocalTensor<T> b2(TPosition::B2,0,2U*n_*k_);
Nd2NzParams bp{1,static_cast<uint16_t>(cols),static_cast<uint16_t>(realK_),0,
static_cast<uint16_t>(realK_),static_cast<uint16_t>(2U*n_),1,0};
DataCopy(b1,w1,bp);DataCopy(b1[n_*16U],w2,bp);
Sync<HardEvent::MTE2_MTE1>();
LoadData2dParams bl;
bl.repeatTimes=static_cast<uint8_t>(2U*n_*k_/256U);bl.srcStride=1;
LoadData(b2,b1,bl);
Sync<HardEvent::MTE1_M>();
}
__aicore__ inline void Compute(const GlobalTensor<T> &x, const GlobalTensor<T> &w1,
const GlobalTensor<T> &w2, const GlobalTensor<T> &dst,
uint32_t rows, uint32_t cols) {
LocalTensor<T> a1(TPosition::A1, 0, m_ * k_);
LocalTensor<T> b1(TPosition::B1, m_ * k_ * sizeof(T), 2U * n_ * k_);
LocalTensor<T> a2(TPosition::A2, 0, m_ * k_);
LocalTensor<T> b2(TPosition::B2, 0, 2U * n_ * k_);
LocalTensor<float> c(TPosition::CO1, 0, 2U * m_ * n_);
const uint32_t alignedM = FullTile ? FixedM : (rows + 15U) / 16U * 16U;
Nd2NzParams ap{1, static_cast<uint16_t>(rows), static_cast<uint16_t>(realK_), 0,
static_cast<uint16_t>(realK_), static_cast<uint16_t>(alignedM), 1, 0};
Nd2NzParams bp{1, static_cast<uint16_t>(cols), static_cast<uint16_t>(realK_), 0,
static_cast<uint16_t>(realK_), static_cast<uint16_t>(2U * n_), 1, 0};
DataCopy(a1, x, ap);
if constexpr (!Resident) {
DataCopy(b1, w1, bp);
DataCopy(b1[n_ * 16U], w2, bp);
}
Sync<HardEvent::MTE2_MTE1>();
LoadData2dParams al;
al.repeatTimes = static_cast<uint8_t>(k_ / 16U);
al.srcStride = static_cast<uint16_t>(alignedM / 16U);
if constexpr (FullTile) {
#pragma unroll
for (uint32_t mb=0;mb<FixedM/16U;++mb) {
LoadData(a2[mb*k_*16U],a1[mb*256U],al);
}
} else {
for (uint32_t mb = 0; mb < alignedM / 16U; ++mb) {
LoadData(a2[mb * k_ * 16U], a1[mb * 256U], al);
}
}
LoadData2dParams bl;
bl.repeatTimes = static_cast<uint8_t>(2U * n_ * k_ / 256U);
bl.srcStride = 1;
if constexpr (!Resident) { LoadData(b2, b1, bl); }
if constexpr (Resident) { Sync<HardEvent::MTE1_MTE2>(); }
Sync<HardEvent::MTE1_M>();
MmadParams mad;
mad.m = static_cast<uint16_t>(alignedM);
mad.n = static_cast<uint16_t>(2U * n_);
mad.k = static_cast<uint16_t>(k_);
mad.cmatrixInitVal = true;
Mmad(c, a2, b2, mad);
if constexpr (Resident) { Sync<HardEvent::M_MTE1>(); }
Sync<HardEvent::M_FIX>();
FixpipeParamsV220 fix;
fix.nSize = static_cast<uint16_t>(2U * n_);
fix.mSize = static_cast<uint16_t>(rows);
fix.srcStride = static_cast<uint16_t>(alignedM);
fix.dstStride = 2U * n_;
fix.quantPre = QuantMode_t::F322F16;
Fixpipe<T, float, CFG_ROW_MAJOR>(dst, c, fix);
if constexpr (Resident) { Sync<HardEvent::FIX_M>(); }
else { PipeBarrier<PIPE_ALL>(); }
}
private:
template <HardEvent E>
__aicore__ inline void Sync() {
constexpr event_t event = EVENT_ID0;
SetFlag<E>(event);
WaitFlag<E>(event);
}
uint32_t k_ = 0, realK_ = 0, m_ = 0, n_ = 0;
};
struct EmptyMatmulStorage {};
template <bool Direct, typename A, typename B, typename C, const auto& Config>
struct MatmulStorageSelector {
using Type = MatmulImpl<A, B, C, C, Config>;
};
template <typename A, typename B, typename C, const auto& Config>
struct MatmulStorageSelector<true, A, B, C, Config> {
using Type = EmptyMatmulStorage;
};
template <typename xType, const auto& Config = CFG_NORM, bool Direct = false,
bool Padded = false, bool Merged = false>
class SwiGluCubeImpl {
public:
using AType = MatmulType<TPosition::GM, CubeFormat::ND, xType, false>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, xType, true>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, xType>;
typename MatmulStorageSelector<Direct, AType, BType, CType, Config>::Type mm;
DirectPairCube<xType> direct;
__aicore__ inline void Init(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
GM_ADDR userWorkspace, TPipe *pipe, const SwiGluTilingData *td)
{
pipe_ = pipe;
core_ = GetBlockIdx();
sub_ = GetSubBlockIdx();
if ASCEND_IS_AIV { core_ /= 2U; }
m_ = td->m;
d_ = static_cast<uint32_t>(td->d);
computeD_ = td->dAlign;
taskM_ = td->taskM;
taskN_ = td->taskN;
taskNum_ = td->taskNum;
cores_ = td->usedCoreNum;
tasksN_ = (d_ + taskN_ - 1U) / taskN_;
plane_ = taskM_ * taskN_;
x_.SetGlobalBuffer((__gm__ xType *)x, m_ * d_);
w1_.SetGlobalBuffer((__gm__ xType *)w1, static_cast<uint64_t>(d_) * d_);
w2_.SetGlobalBuffer((__gm__ xType *)w2, static_cast<uint64_t>(d_) * d_);
b1_.SetGlobalBuffer((__gm__ xType *)b1, d_);
b2_.SetGlobalBuffer((__gm__ xType *)b2, d_);
y_.SetGlobalBuffer((__gm__ xType *)y, m_ * d_);
slots_.SetGlobalBuffer((__gm__ xType *)userWorkspace,
static_cast<uint64_t>(plane_) * 2 * kSlotCount * cores_);
if constexpr (padded_) {
const uint64_t base = static_cast<uint64_t>(plane_) * 2 * kSlotCount * cores_;
const uint64_t wBase = base + m_ * computeD_;
paddedX_.SetGlobalBuffer((__gm__ xType *)userWorkspace + base, m_ * computeD_);
paddedW1_.SetGlobalBuffer((__gm__ xType *)userWorkspace + wBase,
static_cast<uint64_t>(computeD_) * computeD_ * (merged_ ? 2U : 1U));
paddedW2_.SetGlobalBuffer((__gm__ xType *)userWorkspace + wBase +
static_cast<uint64_t>(computeD_) * computeD_,
static_cast<uint64_t>(computeD_) * computeD_);
packRows_ = 4096U / computeD_;
}
if ASCEND_IS_AIC {
if constexpr (Direct) {
direct.Init(computeD_, taskM_, taskN_);
} else {
mm.Init(&td->cubeTilingData, pipe_);
mm.SetHF32(false, 0);
const uint32_t matmulN = merged_ ? 2U * taskN_ : taskN_;
mm.SetSingleShape(taskM_, matmulN, computeD_);
mm.SetOrgShape(taskM_, matmulN, computeD_, computeD_);
}
}
if ASCEND_IS_AIV {
const uint32_t cBytes = (taskM_ / 2U) * taskN_ * sizeof(xType);
const uint32_t vBytes = kBlockRows * taskN_ * sizeof(xType);
pipe_->InitBuffer(gateQ_, 1, cBytes);
pipe_->InitBuffer(upQ_, 1, cBytes);
pipe_->InitBuffer(bias1Q_, 1, taskN_ * sizeof(xType));
pipe_->InitBuffer(bias2Q_, 1, taskN_ * sizeof(xType));
pipe_->InitBuffer(outQ_, 1, vBytes);
pipe_->InitBuffer(gateBuf_, vBytes);
pipe_->InitBuffer(upBuf_, vBytes);
pipe_->InitBuffer(negBuf_, vBytes);
pipe_->InitBuffer(expBuf_, vBytes);
pipe_->InitBuffer(onesBuf_, vBytes);
pipe_->InitBuffer(sigmoidBuf_, vBytes);
if constexpr (padded_) { pipe_->InitBuffer(packQ_, 1, packRows_ * computeD_ * sizeof(xType)); }
}
}
__aicore__ inline void Process()
{
if (core_ >= cores_) { return; }
if constexpr (padded_) {
if ASCEND_IS_AIV {
PackMatrix(paddedX_, x_, m_, m_);
PackMatrix(paddedW1_, w1_, d_, computeD_);
PackMatrix(paddedW2_, w2_, d_, computeD_);
SyncAll();
CrossCoreSetFlag<0x2, PIPE_MTE3>(0x4);
}
if ASCEND_IS_AIC { CrossCoreWaitFlag(0x4); }
}
uint32_t local = 0;
for (uint32_t task = core_; task < taskNum_; task += cores_, ++local) {
const uint32_t slot = local % kSlotCount;
const uint64_t row = static_cast<uint64_t>(task / tasksN_) * taskM_;
const uint32_t col = (task % tasksN_) * taskN_;
const uint32_t rows = Min(taskM_, static_cast<uint32_t>(m_ - row));
const uint32_t cols = Min(taskN_, d_ - col);
const uint64_t off = (static_cast<uint64_t>(core_) * kSlotCount + slot) * 2 * plane_;
if ASCEND_IS_AIC {
if (local >= kSlotCount) { WaitConsumed(slot); }
if constexpr (Direct) {
direct.Compute(x_[row * d_], w1_[static_cast<uint64_t>(col) * d_],
w2_[static_cast<uint64_t>(col) * d_], slots_[off], rows, cols);
} else {
if constexpr (padded_) {
const uint32_t computeCols = Min(taskN_, computeD_ - col);
if constexpr (merged_) {
Projection(paddedW1_, row, 0, rows, 2U * computeCols, off);
} else {
Projection(paddedW1_, row, col, rows, computeCols, off);
Projection(paddedW2_, row, col, rows, computeCols, off + plane_);
}
} else {
Projection(w1_, row, col, rows, cols, off);
Projection(w2_, row, col, rows, cols, off + plane_);
}
}
SignalReady(slot);
}
if ASCEND_IS_AIV {
WaitReady(slot);
Epilogue(off, row, col, rows, cols);
SignalConsumed(slot);
}
}
if ASCEND_IS_AIC {
for (uint32_t slot = 0; slot < Min(local, kSlotCount); ++slot) { WaitConsumed(slot); }
}
}
private:
__aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }
__aicore__ inline void PackMatrix(const GlobalTensor<xType> &dst,
const GlobalTensor<xType> &src,
uint64_t realRows, uint64_t totalRows)
{
const uint64_t chunks = (totalRows + packRows_ - 1U) / packRows_;
for (uint64_t block = GetBlockIdx(); block < chunks; block += cores_ * 2U) {
const uint64_t row = block * packRows_;
const uint32_t nr = Min(packRows_, static_cast<uint32_t>(totalRows - row));
const uint32_t valid = row < realRows ? Min(nr, static_cast<uint32_t>(realRows - row)) : 0;
LocalTensor<xType> tile = packQ_.AllocTensor<xType>();
if (valid != 0) {
DataCopyExtParams copy{static_cast<uint16_t>(valid), d_ * 2U, 0, 0, 0};
DataCopyPadExtParams<xType> pad{true, 0, static_cast<uint8_t>(computeD_ - d_), 0};
DataCopyPad(tile, src[row * d_], copy, pad);
}
if (valid < nr) {
Duplicate(tile[valid * computeD_], static_cast<xType>(0), (nr - valid) * computeD_);
const event_t event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
SetFlag<HardEvent::V_MTE3>(event);
WaitFlag<HardEvent::V_MTE3>(event);
}
packQ_.EnQue(tile);
tile = packQ_.DeQue<xType>();
DataCopy(dst[row * computeD_], tile, nr * computeD_);
packQ_.FreeTensor(tile);
}
}
__aicore__ inline void Projection(const GlobalTensor<xType> &weight, uint64_t row, uint32_t col,
uint32_t rows, uint32_t cols, uint64_t off)
{
mm.SetTail(rows, cols, computeD_);
if constexpr (padded_) { mm.SetTensorA(paddedX_[row * computeD_], false); }
else { mm.SetTensorA(x_[row * d_], false); }
mm.SetTensorB(weight[static_cast<uint64_t>(col) * computeD_], true);
mm.IterateAll(slots_[off], false);
mm.End();
PipeBarrier<PIPE_ALL>();
}
__aicore__ inline void Epilogue(uint64_t off, uint64_t row, uint32_t col,
uint32_t rows, uint32_t cols)
{
const uint32_t halfRows = (rows + 1U) / 2U;
const uint32_t begin = sub_ * halfRows;
if (begin >= rows) { return; }
const uint32_t count = Min(halfRows, rows - begin);
const uint32_t pitch = AlignUp16(cols);
const uint32_t cPitch = merged_ ? 2U * taskN_ : taskN_;
LocalTensor<xType> gate = gateQ_.AllocTensor<xType>();
LocalTensor<xType> up = upQ_.AllocTensor<xType>();
DataCopyExtParams cParams{static_cast<uint16_t>(count), cols * static_cast<uint32_t>(sizeof(xType)),
(cPitch - cols) * static_cast<uint32_t>(sizeof(xType)), 0, 0};
DataCopyPadExtParams<xType> pad{false, 0, 0, 0};
const uint64_t cOff = off + static_cast<uint64_t>(begin) * cPitch;
DataCopyPad(gate, slots_[cOff], cParams, pad);
DataCopyPad(up, slots_[cOff + (merged_ ? taskN_ : plane_)], cParams, pad);
gateQ_.EnQue(gate);
upQ_.EnQue(up);
LocalTensor<xType> b1 = bias1Q_.AllocTensor<xType>();
LocalTensor<xType> b2 = bias2Q_.AllocTensor<xType>();
DataCopyExtParams biasParams{1, cols * static_cast<uint32_t>(sizeof(xType)), 0, 0, 0};
DataCopyPad(b1, b1_[col], biasParams, pad);
DataCopyPad(b2, b2_[col], biasParams, pad);
bias1Q_.EnQue(b1);
bias2Q_.EnQue(b2);
gate = gateQ_.DeQue<xType>();
up = upQ_.DeQue<xType>();
b1 = bias1Q_.DeQue<xType>();
b2 = bias2Q_.DeQue<xType>();
LocalTensor<xType> a = gateBuf_.Get<xType>();
LocalTensor<xType> b = upBuf_.Get<xType>();
LocalTensor<xType> neg = negBuf_.Get<xType>();
LocalTensor<xType> exp = expBuf_.Get<xType>();
LocalTensor<xType> ones = onesBuf_.Get<xType>();
LocalTensor<xType> sig = sigmoidBuf_.Get<xType>();
for (uint32_t group = 0; group < count; group += kBlockRows) {
const uint32_t nr = Min(kBlockRows, count - group);
const uint32_t n = nr * pitch;
const uint8_t rowStride = static_cast<uint8_t>(pitch / 16U);
const BinaryRepeatParams biasRepeat{1, 1, 1, rowStride, rowStride, 0};
for (uint32_t k = 0; k < cols; k += 128U) {
const uint32_t width = Min(128U, cols - k);
Add(a[k], gate[group * pitch + k], b1[k], static_cast<uint64_t>(width),
static_cast<uint8_t>(nr), biasRepeat);
Add(b[k], up[group * pitch + k], b2[k], static_cast<uint64_t>(width),
static_cast<uint8_t>(nr), biasRepeat);
}
LocalTensor<xType> out = outQ_.AllocTensor<xType>();
GateCounter<xType,true>(a,b,exp,out,ones,n);
outQ_.EnQue(out);
out = outQ_.DeQue<xType>();
DataCopyExtParams outParams{static_cast<uint16_t>(nr), cols * static_cast<uint32_t>(sizeof(xType)),
0, (d_ - cols) * static_cast<uint32_t>(sizeof(xType)), 0};
DataCopyPad(y_[(row + begin + group) * d_ + col], out, outParams);
outQ_.FreeTensor(out);
}
gateQ_.FreeTensor(gate);
upQ_.FreeTensor(up);
bias1Q_.FreeTensor(b1);
bias2Q_.FreeTensor(b2);
}
__aicore__ inline void WaitReady(uint32_t s) {
if (s == 0) { CrossCoreWaitFlag(0x6); }
else if (s == 1) { CrossCoreWaitFlag(0x7); }
else if (s == 2) { CrossCoreWaitFlag(0xB); }
else { CrossCoreWaitFlag(0xE); }
}
__aicore__ inline void SignalReady(uint32_t s) {
if (s == 0) { CrossCoreSetFlag<0x2, PIPE_FIX>(0x6); }
else if (s == 1) { CrossCoreSetFlag<0x2, PIPE_FIX>(0x7); }
else if (s == 2) { CrossCoreSetFlag<0x2, PIPE_FIX>(0xB); }
else { CrossCoreSetFlag<0x2, PIPE_FIX>(0xE); }
}
__aicore__ inline void WaitConsumed(uint32_t s) {
if (s == 0) { CrossCoreWaitFlag(0x8); }
else if (s == 1) { CrossCoreWaitFlag(0x9); }
else if (s == 2) { CrossCoreWaitFlag(0xC); }
else { CrossCoreWaitFlag(0xF); }
}
__aicore__ inline void SignalConsumed(uint32_t s) {
if (s == 0) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0x8); }
else if (s == 1) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0x9); }
else if (s == 2) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0xC); }
else { CrossCoreSetFlag<0x2, PIPE_MTE3>(0xF); }
}
TPipe *pipe_ = nullptr;
TQue<QuePosition::VECIN, 1> gateQ_, upQ_, bias1Q_, bias2Q_;
TQue<QuePosition::VECOUT, 1> outQ_;
TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> packQ_;
TBuf<TPosition::VECCALC> gateBuf_, upBuf_, negBuf_, expBuf_, onesBuf_, sigmoidBuf_;
GlobalTensor<xType> x_, w1_, w2_, b1_, b2_, y_, slots_;
GlobalTensor<xType> paddedX_, paddedW1_, paddedW2_;
uint64_t m_ = 0;
uint32_t d_ = 0, taskM_ = 0, taskN_ = 0, taskNum_ = 0, tasksN_ = 0;
uint32_t core_ = 0, sub_ = 0, cores_ = 0, plane_ = 0;
uint32_t computeD_ = 0, packRows_ = 0;
static constexpr bool padded_ = Padded;
static constexpr bool merged_ = Merged || Direct;
};
template <typename T, bool Padded = false, bool Merged = false>
__aicore__ inline void RunCube(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
GM_ADDR workspace, TPipe *pipe, const SwiGluTilingData *td)
{
SwiGluCubeImpl<T, CFG_NORM, false, Padded, Merged> op;
op.Init(x, w1, b1, w2, b2, y, workspace, pipe, td);
op.Process();
}
template <typename T, uint32_t TileM, uint32_t FixedD = 0, bool Unaligned = false, bool Resident = false, bool FullTile = false>
class SlimDirectCube {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
GM_ADDR workspace, const SwiGluDirectTilingData *td) {
d_ = FixedD != 0 ? FixedD : static_cast<uint32_t>(td->d);
n_ = FixedD != 0 ? (FixedD < 64U ? FixedD : 64U) : td->taskN;
const uint32_t cores = td->usedCoreNum;
uint32_t core = GetBlockIdx();
if ASCEND_IS_AIV { core /= 2U; }
x_.SetGlobalBuffer((__gm__ T *)x, td->m * d_);
w1_.SetGlobalBuffer((__gm__ T *)w1, static_cast<uint64_t>(d_) * d_);
w2_.SetGlobalBuffer((__gm__ T *)w2, static_cast<uint64_t>(d_) * d_);
b1_.SetGlobalBuffer((__gm__ T *)b1, d_);
b2_.SetGlobalBuffer((__gm__ T *)b2, d_);
y_.SetGlobalBuffer((__gm__ T *)y, td->m * d_);
const uint32_t plane = tm_ * n_;
slots_.SetGlobalBuffer((__gm__ T *)workspace, static_cast<uint64_t>(2U * plane) * 4U * cores);
if ASCEND_IS_AIC { cube_.Init(d_, tm_, n_); }
SetAtomicNone();
if ASCEND_IS_AIV {
SetMaskNorm();
SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
}
const uint32_t tasksN = (d_ + n_ - 1U) / n_;
if constexpr (Resident) {
const uint32_t col=core%tasksN*n_;
const uint32_t cols=FullTile?n_:Min(n_,d_-col);
if ASCEND_IS_AIC {
cube_.PrepareWeights(w1_[static_cast<uint64_t>(col)*d_],w2_[static_cast<uint64_t>(col)*d_],cols);
}
if ASCEND_IS_AIV {
LocalTensor<T> bias(TPosition::VECIN,tm_*n_*sizeof(T),2U*n_);
if constexpr (Unaligned) {
DataCopyExtParams bp{1,cols*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>((cols+15U)/16U*16U-cols),static_cast<T>(0)};
DataCopyPad(bias,b1_[col],bp,pad);DataCopyPad(bias[n_],b2_[col],bp,pad);
} else {
DataCopy(bias,b1_[col],cols);DataCopy(bias[n_],b2_[col],cols);
}
}
}
uint32_t local = 0;
for (uint32_t task = core; task < td->taskNum; task += cores, ++local) {
const uint32_t slot = local % 4U;
const uint64_t row = static_cast<uint64_t>(task / tasksN) * tm_;
const uint32_t col = task % tasksN * n_;
const uint32_t rows = FullTile?tm_:Min(tm_, td->m - row);
const uint32_t cols = FullTile?n_:Min(n_, d_ - col);
const uint64_t off = (static_cast<uint64_t>(core) * 4U + slot) * 2U * plane;
if ASCEND_IS_AIC {
if (local >= 4U) { WaitConsumed(slot); }
cube_.Compute(x_[row * d_], w1_[static_cast<uint64_t>(col) * d_],
w2_[static_cast<uint64_t>(col) * d_], slots_[off], rows, cols);
Ready(slot);
}
if ASCEND_IS_AIV {
WaitReady(slot);
Epilogue(off, row, col, rows, cols);
Consumed(slot);
}
}
if ASCEND_IS_AIC {
for (uint32_t slot = 0; slot < Min(local, 4U); ++slot) { WaitConsumed(slot); }
}
PipeBarrier<PIPE_ALL>();
}
private:
__aicore__ inline uint32_t Min(uint32_t a, uint64_t b) { return a < b ? a : b; }
template <HardEvent E> __aicore__ inline void Sync() {
constexpr event_t e = EVENT_ID0;
SetFlag<E>(e); WaitFlag<E>(e);
}
__aicore__ inline void Epilogue(uint64_t off, uint64_t row, uint32_t col,
uint32_t rows, uint32_t cols) {
const uint32_t half = FullTile?tm_/2U:(rows + 1U) / 2U;
const uint32_t begin = GetSubBlockIdx() * half;
if constexpr (!FullTile) { if (begin >= rows) { return; } }
const uint32_t nr = FullTile?tm_/2U:Min(half, rows - begin);
const uint32_t capacity = tm_ / 2U * n_;
LocalTensor<T> input(TPosition::VECIN, 0, (tm_ + 2U) * n_);
auto g = input;
auto u = input[capacity];
auto b1 = input[2U * capacity];
auto b2 = input[2U * capacity + n_];
LocalTensor<T> work(TPosition::VECOUT, (tm_ + 2U) * n_ * sizeof(T), tm_ * n_);
auto s = work;
auto out = work[capacity];
const uint64_t start = off + begin * 2U * n_;
if constexpr (Unaligned) {
const uint32_t pitch = (cols + 15U) / 16U * 16U;
DataCopyExtParams cp{static_cast<uint16_t>(nr), cols * 2U,
(2U * n_ - cols) * 2U, (n_ - pitch) / 16U, 0};
DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(pitch - cols), static_cast<T>(0)};
DataCopyPad(g, slots_[start], cp, pad);
DataCopyPad(u, slots_[start + n_], cp, pad);
DataCopyExtParams bp{1, cols * 2U, 0, 0, 0};
if constexpr (!Resident) {
DataCopyPad(b1, b1_[col], bp, pad);
DataCopyPad(b2, b2_[col], bp, pad);
}
} else {
DataCopyParams cp{static_cast<uint16_t>(nr), static_cast<uint16_t>(cols / 16U),
static_cast<uint16_t>((2U * n_ - cols) / 16U),
static_cast<uint16_t>((n_ - cols) / 16U)};
DataCopy(g, slots_[start], cp);
DataCopy(u, slots_[start + n_], cp);
if constexpr (!Resident) {
DataCopy(b1, b1_[col], cols);
DataCopy(b2, b2_[col], cols);
}
}
Sync<HardEvent::MTE2_V>();
const uint8_t stride = static_cast<uint8_t>(n_ / 16U);
const BinaryRepeatParams repeat{1, 1, 1, stride, stride, 0};
Add(g, g, b1, static_cast<uint64_t>(cols), static_cast<uint8_t>(nr), repeat);
Add(u, u, b2, static_cast<uint64_t>(cols), static_cast<uint8_t>(nr), repeat);
PipeBarrier<PIPE_V>();
const uint32_t count = nr * n_;
GateCounter<T,true>(g,u,s,out,out,count);
Sync<HardEvent::V_MTE3>();
if constexpr (Unaligned) {
const uint32_t pitch = (cols + 15U) / 16U * 16U;
DataCopyExtParams output{static_cast<uint16_t>(nr), cols * 2U,
(n_ - pitch) / 16U, (d_ - cols) * 2U, 0};
DataCopyPad(y_[(row + begin) * d_ + col], out, output);
} else {
DataCopyParams output{static_cast<uint16_t>(nr), static_cast<uint16_t>(cols / 16U),
static_cast<uint16_t>((n_ - cols) / 16U),
static_cast<uint16_t>((d_ - cols) / 16U)};
DataCopy(y_[(row + begin) * d_ + col], out, output);
}
Sync<HardEvent::MTE3_MTE2>();
}
__aicore__ inline void WaitReady(uint32_t s) {
if (s == 0) { CrossCoreWaitFlag(0x6); } else if (s == 1) { CrossCoreWaitFlag(0x7); }
else if (s == 2) { CrossCoreWaitFlag(0xB); } else { CrossCoreWaitFlag(0xE); }
}
__aicore__ inline void Ready(uint32_t s) {
if (s == 0) { CrossCoreSetFlag<0x2, PIPE_FIX>(0x6); } else if (s == 1) { CrossCoreSetFlag<0x2, PIPE_FIX>(0x7); }
else if (s == 2) { CrossCoreSetFlag<0x2, PIPE_FIX>(0xB); } else { CrossCoreSetFlag<0x2, PIPE_FIX>(0xE); }
}
__aicore__ inline void WaitConsumed(uint32_t s) {
if (s == 0) { CrossCoreWaitFlag(0x8); } else if (s == 1) { CrossCoreWaitFlag(0x9); }
else if (s == 2) { CrossCoreWaitFlag(0xC); } else { CrossCoreWaitFlag(0xF); }
}
__aicore__ inline void Consumed(uint32_t s) {
if (s == 0) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0x8); } else if (s == 1) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0x9); }
else if (s == 2) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0xC); } else { CrossCoreSetFlag<0x2, PIPE_MTE3>(0xF); }
}
DirectPairCube<T, TileM, FixedD, Unaligned, Resident, FullTile> cube_;
GlobalTensor<T> x_, w1_, w2_, b1_, b2_, y_, slots_;
static constexpr uint32_t tm_ = TileM;
uint32_t d_ = 0, n_ = 0;
};
template <typename T>
class UnalignedTileCube {
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y,GM_ADDR ws) {
uint32_t core=GetBlockIdx();
if ASCEND_IS_AIV {core/=2U;}
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy,tmp;
gx.SetGlobalBuffer((__gm__ T*)x);gw1.SetGlobalBuffer((__gm__ T*)w1);gw2.SetGlobalBuffer((__gm__ T*)w2);
gb1.SetGlobalBuffer((__gm__ T*)b1);gb2.SetGlobalBuffer((__gm__ T*)b2);gy.SetGlobalBuffer((__gm__ T*)y);
tmp.SetGlobalBuffer((__gm__ T*)ws);
const uint32_t off=core*32U*96U;
SetAtomicNone();
if ASCEND_IS_AIC {
cube_.Init(33U,32U,48U);
cube_.Compute(gx[core*17U*33U],gw1,gw2,tmp[off],17U,33U);
CrossCoreSetFlag<0x2,PIPE_FIX>(0x6);
CrossCoreWaitFlag(0x8);
}
if ASCEND_IS_AIV {
const uint32_t sub=GetSubBlockIdx(), nr=9U-sub, begin=9U*sub;
LocalTensor<T> g(TPosition::VECIN,0,432),u(TPosition::VECIN,864,432);
LocalTensor<T> b1t(TPosition::VECIN,1728,48),b2t(TPosition::VECIN,1824,48);
LocalTensor<T> sig(TPosition::VECCALC,1920,432),out(TPosition::VECOUT,2784,432);
SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams bc{1,66,0,0,0};DataCopyPadExtParams<T> pad{true,0,15,static_cast<T>(0)};
DataCopyPad(b1t,gb1,bc,pad);DataCopyPad(b2t,gb2,bc,pad);
CrossCoreWaitFlag(0x6);
DataCopyParams cp{static_cast<uint16_t>(nr),3,3,0};
DataCopy(g,tmp[off+begin*96U],cp);DataCopy(u,tmp[off+begin*96U+48U],cp);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
const BinaryRepeatParams rp{1,1,1,3,3,0};
Add(g,g,b1t,static_cast<uint64_t>(33),static_cast<uint8_t>(nr),rp);
Add(u,u,b2t,static_cast<uint64_t>(33),static_cast<uint8_t>(nr),rp);
PipeBarrier<PIPE_V>();GateCounter<T,true>(g,u,sig,out,out,nr*48U);
SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
DataCopyExtParams oc{static_cast<uint16_t>(nr),66,0,0,0};
DataCopyPad(gy[(core*17U+begin)*33U],out,oc);
CrossCoreSetFlag<0x2,PIPE_MTE3>(0x8);
}
PipeBarrier<PIPE_ALL>();
}
private:
DirectPairCube<T,32U,0U,true,false,true> cube_;
};
template <typename T,uint32_t TM,uint32_t D,uint32_t N,uint32_t Iterations>
class DoubleBufferedPairCube {
static constexpr uint32_t A=TM*D,B=2U*N*D;
static constexpr bool SplitL0B=(B*sizeof(T)>64U*1024U);
static constexpr bool DoubleA2=(2U*A*sizeof(T)<=64U*1024U);
static constexpr bool DoubleC=(4U*TM*N*sizeof(float)<=128U*1024U);
static constexpr uint32_t L1Slots=SplitL0B?2U:Iterations;
static_assert((L1Slots*A+B)*sizeof(T)<=512U*1024U,"L1 capacity");
static_assert(A*sizeof(T)<=64U*1024U && (SplitL0B ? N*D*sizeof(T)<=64U*1024U : B*sizeof(T)<=64U*1024U),"L0A/B capacity");
public:
template<uint32_t FirstTM=TM>
__aicore__ inline void PrepareWeights(const GlobalTensor<T>& w1,const GlobalTensor<T>& w2,const GlobalTensor<T>& x0) {
SetMMRowMajor();
LocalTensor<uint64_t> unitState(TPosition::CO1,0U,16U);
SetFixPipeConfig(unitState,true);
SetFlag<HardEvent::FIX_M>(EVENT_ID7);
LocalTensor<T> b1(TPosition::B1,2U*A*sizeof(T),B);
LocalTensor<T> a1(TPosition::A1,0,FirstTM*D);
LocalTensor<T> b2_0(TPosition::B2,0,N*D/2U);
LocalTensor<T> b2_1(TPosition::B2,N*D/2U*sizeof(T),N*D/2U);
constexpr uint32_t HM=TM/2U;
const Nd2NzParams hap{1,HM,D,0,D,HM,1,0};
const Nd2NzParams bp{1,N/2U,D,0,D,N/2U,1,0};
DataCopy(a1,x0,hap);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
DataCopy(b1,w1,bp);DataCopy(b1[N*D/2U],w1[(N/2U)*D],bp);
SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
DataCopy(a1[HM*D],x0[HM*D],hap);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
DataCopy(b1[N*D],w2,bp);DataCopy(b1[N*D+N*D/2U],w2[(N/2U)*D],bp);
SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
LocalTensor<T> a2_0(TPosition::A2,0U,HM*D);
LocalTensor<T> a2_1(TPosition::A2,HM*D*sizeof(T),HM*D);
LoadData2dParams al;al.repeatTimes=D/16U;al.srcStride=HM/16U;
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
#pragma unroll
for(uint32_t mb=0;mb<HM/16U;++mb) {LoadData(a2_0[mb*D*16U],a1[mb*256U],al);}
SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
LoadData2dParams bl;bl.repeatTimes=(N*D/2U)/256U;bl.srcStride=1;
LoadData(b2_0,b1,bl);
SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
LoadData(b2_1,b1[N*D/2U],bl);
SetFlag<HardEvent::MTE1_M>(EVENT_ID3);
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
#pragma unroll
for(uint32_t mb=0;mb<HM/16U;++mb) {LoadData(a2_1[mb*D*16U],a1[HM*D+mb*256U],al);}
SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
}
template<uint32_t TM0,uint32_t TM1,uint32_t TM2,uint32_t TM3=TM>
__aicore__ inline void PrepareAll(const GlobalTensor<T>& w1,const GlobalTensor<T>& w2,
const GlobalTensor<T>& x0,const GlobalTensor<T>& x1,
const GlobalTensor<T>& x2,const GlobalTensor<T>& x3) {
SetMMRowMajor();
LocalTensor<uint64_t> unitState(TPosition::CO1,0U,16U);
SetFixPipeConfig(unitState,true);
SetFlag<HardEvent::FIX_M>(EVENT_ID7);
LocalTensor<T> a1_0(TPosition::A1,0U,TM0*D);
LocalTensor<T> a1_1(TPosition::A1,A*sizeof(T),TM1*D);
LocalTensor<T> a1_2(TPosition::A1,2U*A*sizeof(T),TM2*D);
LocalTensor<T> b1(TPosition::B1,Iterations*A*sizeof(T),B);
LocalTensor<T> a2_0(TPosition::A2,0U,TM0*D);
LocalTensor<T> b2_0(TPosition::B2,0U,N*D);
LocalTensor<T> b2_1(TPosition::B2,N*D*sizeof(T),N*D);
const Nd2NzParams ap0{1,TM0,D,0,D,TM0,1,0};
const Nd2NzParams ap1{1,TM1,D,0,D,TM1,1,0};
const Nd2NzParams ap2{1,TM2,D,0,D,TM2,1,0};
const Nd2NzParams bp{1,N,D,0,D,N,1,0};
DataCopy(a1_0,x0,ap0);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
DataCopy(b1,w1,bp);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
DataCopy(b1[N*D],w2,bp);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
DataCopy(a1_1,x1,ap1);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
LoadData2dParams al0;al0.repeatTimes=D/16U;al0.srcStride=TM0/16U;
#pragma unroll
for(uint32_t mb=0;mb<TM0/16U;++mb) {LoadData(a2_0[mb*D*16U],a1_0[mb*256U],al0);}
SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
DataCopy(a1_2,x2,ap2);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
if constexpr (Iterations>=4U) {SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);}
LoadData2dParams bl;bl.repeatTimes=(N*D)/256U;bl.srcStride=1;
LoadData(b2_0,b1,bl);
SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
if constexpr (Iterations>=4U) {
LocalTensor<T> a1_3(TPosition::A1,3U*A*sizeof(T),TM3*D);
const Nd2NzParams ap3{1,TM3,D,0,D,TM3,1,0};
WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
DataCopy(a1_3,x3,ap3);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
}
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
LoadData(b2_1,b1[N*D],bl);
SetFlag<HardEvent::MTE1_M>(EVENT_ID3);
}
template<uint32_t I,uint32_t CurTM=TM> __aicore__ inline void Prefetch(const GlobalTensor<T>& x) {
constexpr event_t slot=I%2U==0U?EVENT_ID0:EVENT_ID1;
if constexpr(I>=2U || (SplitL0B && I==1U)) {WaitFlag<HardEvent::MTE1_MTE2>(slot);}
LocalTensor<T> a1(TPosition::A1,(I%2U)*A*sizeof(T),CurTM*D);
const Nd2NzParams ap{1,CurTM,D,0,D,CurTM,1,0};
DataCopy(a1,x,ap);SetFlag<HardEvent::MTE2_MTE1>(slot);
}
template<uint32_t I,uint32_t CurTM=TM> __aicore__ inline void Compute(const GlobalTensor<T>& dst) {
if constexpr(I==0U) {WaitFlag<HardEvent::FIX_M>(EVENT_ID7);}
constexpr event_t slot=I%2U==0U?EVENT_ID0:EVENT_ID1;
constexpr event_t cSlot=DoubleC?slot:EVENT_ID0;
if constexpr (SplitL0B) {
LocalTensor<T> a1(TPosition::A1,(I%2U)*A*sizeof(T),CurTM*D);
LocalTensor<T> b1(TPosition::B1,2U*A*sizeof(T),B);
LocalTensor<T> b2_0(TPosition::B2,0,N*D/2U);
LocalTensor<T> b2_1(TPosition::B2,N*D/2U*sizeof(T),N*D/2U);
LoadData2dParams bl;bl.repeatTimes=(N*D/2U)/256U;bl.srcStride=1;
constexpr uint32_t HM=TM/2U;
constexpr uint32_t firstOff=(I%2U==1U)?(HM*N):0U;
constexpr uint32_t secondOff=(I%2U==1U)?0U:(HM*N);
constexpr uint32_t nextB1Off=(I%2U==1U)?0U:(N*D);
if constexpr (CurTM<=HM) {
constexpr uint32_t cFirstOff=(I%2U==1U)?(CurTM*N):0U;
constexpr uint32_t cSecondOff=(I%2U==1U)?0U:(CurTM*N);
LocalTensor<T> a2_0(TPosition::A2,0U,CurTM*D);
LocalTensor<float> c0(TPosition::CO1,0U,2U*HM*N);
if constexpr(I>0U) {
WaitFlag<HardEvent::MTE2_MTE1>(slot);
WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
LoadData2dParams al;al.repeatTimes=D/16U;al.srcStride=CurTM/16U;
#pragma unroll
for(uint32_t mb=0;mb<CurTM/16U;++mb) {LoadData(a2_0[mb*D*16U],a1[mb*256U],al);}
SetFlag<HardEvent::MTE1_MTE2>(slot);
SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
}
WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
if constexpr(I==0U) {WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);}
if constexpr(I>0U) {}
MmadParams mad;mad.m=CurTM;mad.n=N/2U;mad.k=D;mad.cmatrixInitVal=true;mad.unitFlag=3;
Mmad(c0[cFirstOff],a2_0,b2_0,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
if constexpr(I==0U) {WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);}
Mmad(c0[cFirstOff+CurTM*N/2U],a2_0,b2_1,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
if constexpr(I==0U) {WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);}
WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
LoadData(b2_0,b1[nextB1Off],bl);
SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
LoadData(b2_1,b1[nextB1Off+N*D/2U],bl);
SetFlag<HardEvent::MTE1_M>(EVENT_ID3);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);
Mmad(c0[cSecondOff],a2_0,b2_0,mad);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);
Mmad(c0[cSecondOff+CurTM*N/2U],a2_0,b2_1,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID0);

FixpipeParamsV220 fix;fix.nSize=2U*N;fix.mSize=CurTM;fix.srcStride=CurTM;fix.dstStride=2U*N;fix.quantPre=QuantMode_t::F322F16;fix.unitFlag=3;

if constexpr (I>=4U) {
constexpr uint32_t cid=(I%4U)==0U?0xA:((I%4U)==1U?0xB:((I%4U)==2U?0xC:0xE));
CrossCoreWaitFlag(cid);
}
Fixpipe<T,float,CFG_ROW_MAJOR>(dst,c0,fix);

} else {
LocalTensor<T> a2_0(TPosition::A2,0U,HM*D);
LocalTensor<T> a2_1(TPosition::A2,HM*D*sizeof(T),HM*D);
LocalTensor<float> c0(TPosition::CO1,0U,2U*HM*N);
LocalTensor<float> c1(TPosition::CO1,2U*HM*N*sizeof(float),2U*HM*N);
if constexpr (I>0U) {
WaitFlag<HardEvent::MTE2_MTE1>(slot);
WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
LoadData2dParams al;al.repeatTimes=D/16U;al.srcStride=CurTM/16U;
#pragma unroll
for(uint32_t mb=0;mb<HM/16U;++mb) {LoadData(a2_0[mb*D*16U],a1[mb*256U],al);}
SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
#pragma unroll
for(uint32_t mb=0;mb<HM/16U;++mb) {LoadData(a2_1[mb*D*16U],a1[(mb+HM/16U)*256U],al);}
SetFlag<HardEvent::MTE1_MTE2>(slot);
SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
}
WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
if constexpr(I==0U) {WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);}
if constexpr(I>0U) {}
MmadParams mad;mad.m=HM;mad.n=N/2U;mad.k=D;mad.cmatrixInitVal=true;mad.unitFlag=3;
Mmad(c0[firstOff],a2_0,b2_0,mad);
if constexpr(I==0U) {WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);}
Mmad(c0[firstOff+HM*N/2U],a2_0,b2_1,mad);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
if constexpr(I>0U) {}
Mmad(c1[firstOff],a2_1,b2_0,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
Mmad(c1[firstOff+HM*N/2U],a2_1,b2_1,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
if constexpr(I==0U) {WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);}
WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
LoadData(b2_0,b1[nextB1Off],bl);
SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
LoadData(b2_1,b1[nextB1Off+N*D/2U],bl);
SetFlag<HardEvent::MTE1_M>(EVENT_ID3);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);
Mmad(c0[secondOff],a2_0,b2_0,mad);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);
Mmad(c0[secondOff+HM*N/2U],a2_0,b2_1,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID0);

Mmad(c1[secondOff],a2_1,b2_0,mad);
Mmad(c1[secondOff+HM*N/2U],a2_1,b2_1,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID1);

FixpipeParamsV220 fix;fix.nSize=2U*N;fix.mSize=HM;fix.srcStride=HM;fix.dstStride=2U*N;fix.quantPre=QuantMode_t::F322F16;fix.unitFlag=3;

if constexpr (I>=4U) {
constexpr uint32_t cid=(I%4U)==0U?0xA:((I%4U)==1U?0xB:((I%4U)==2U?0xC:0xE));
CrossCoreWaitFlag(cid);
}
Fixpipe<T,float,CFG_ROW_MAJOR>(dst,c0,fix);


Fixpipe<T,float,CFG_ROW_MAJOR>(dst[HM*2U*N],c1,fix);

}
} else {
LocalTensor<T> a1(TPosition::A1,I*A*sizeof(T),CurTM*D);
LocalTensor<T> a2(TPosition::A2,(I%2U)*A*sizeof(T),CurTM*D);
LocalTensor<T> b2_0(TPosition::B2,0U,N*D);
LocalTensor<T> b2_1(TPosition::B2,N*D*sizeof(T),N*D);
LocalTensor<float> c(TPosition::CO1,(I%2U)*2U*TM*N*sizeof(float),2U*CurTM*N);
if constexpr (I>0U) {
constexpr event_t mte2Ev=(I==1U)?EVENT_ID1:((I==2U)?EVENT_ID0:EVENT_ID2);
WaitFlag<HardEvent::MTE2_MTE1>(mte2Ev);
if constexpr(I>=2U) {WaitFlag<HardEvent::M_MTE1>(slot);}
LoadData2dParams al;al.repeatTimes=D/16U;al.srcStride=CurTM/16U;
#pragma unroll
for(uint32_t mb=0;mb<CurTM/16U;++mb) {LoadData(a2[mb*D*16U],a1[mb*256U],al);}
SetFlag<HardEvent::MTE1_M>(slot);
}
WaitFlag<HardEvent::MTE1_M>(slot);
if constexpr (I==0U) {WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);}
if constexpr (I>=2U) {}
MmadParams mad;mad.m=CurTM;mad.n=N;mad.k=D;mad.cmatrixInitVal=true;mad.unitFlag=3;
Mmad(c,a2,b2_0,mad);
if constexpr (I==0U) {WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);}
Mmad(c[CurTM*N],a2,b2_1,mad);
SetFlag<HardEvent::M_MTE1>(slot);

FixpipeParamsV220 fix;fix.nSize=2U*N;fix.mSize=CurTM;fix.srcStride=CurTM;fix.dstStride=2U*N;fix.quantPre=QuantMode_t::F322F16;fix.unitFlag=3;
Fixpipe<T,float,CFG_ROW_MAJOR>(dst,c,fix);
}
}
__aicore__ inline void Drain() {
WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);


if constexpr (SplitL0B) {
WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
}
}
};
template <typename T,uint32_t D,uint32_t TM,uint32_t Cores,uint32_t Iterations,uint32_t TileN=64U>
class PipelinedCubeSwiGlu {
static constexpr uint32_t N=TileN, TN=D/N, Groups=Cores/TN, Plane=TM*N;
static constexpr uint32_t M=(Groups==4U)?512U:((Groups==10U)?8192U:(TM*Groups*Iterations));
static constexpr uint32_t FirstTM=(Groups==4U)?32U:TM;
static constexpr bool FastAiv=true;
static constexpr bool SplitL0B=(2U*N*D*sizeof(T)>64U*1024U);
static constexpr uint32_t NR=TM/2U, Cap=NR*N;
static constexpr uint32_t InSlots=SplitL0B?2U:4U;
static constexpr uint32_t OutSlots=SplitL0B?2U:4U;
static_assert(D%N==0U && TM%32U==0U,"Exact tile shape");
static_assert(TM*D*sizeof(T)<=64U*1024U,"L0A capacity");
static_assert(N*D*sizeof(T)<=64U*1024U,"L0B half capacity");
static_assert(2U*TM*N*sizeof(float)<=128U*1024U,"L0C capacity");
static_assert(((2U*InSlots+3U+OutSlots)*Cap+2U*N)*sizeof(T)<=192U*1024U,"UB capacity");
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y,GM_ADDR workspace) {
core_=GetBlockIdx();
if ASCEND_IS_AIV {core_/=2U;}
col_=core_%TN*N;
grp_=core_/TN;
if constexpr (Groups==4U) {
baseRow_=grp_*128U;
} else if constexpr (Groups==10U) {
baseRow_=(grp_<8U)?(grp_*832U):(6656U+(grp_-8U)*768U);
} else {
baseRow_=grp_*TM;
}
x_.SetGlobalBuffer((__gm__ T*)x,M*D);y_.SetGlobalBuffer((__gm__ T*)y,M*D);
w1_.SetGlobalBuffer((__gm__ T*)w1,D*D);w2_.SetGlobalBuffer((__gm__ T*)w2,D*D);
b1_.SetGlobalBuffer((__gm__ T*)b1,D);b2_.SetGlobalBuffer((__gm__ T*)b2,D);
slots_.SetGlobalBuffer((__gm__ T*)workspace,2U*Plane*4U*Cores);
SetAtomicNone();
if ASCEND_IS_AIC {
if constexpr (Groups==4U) {
piped_.template PrepareAll<32U,64U,32U>(w1_[col_*D],w2_[col_*D],
x_[baseRow_*D],x_[(baseRow_+32U)*D],x_[(baseRow_+96U)*D],x_[baseRow_*D]);
} else if constexpr (Groups==16U) {
piped_.template PrepareAll<64U,64U,64U,64U>(w1_[col_*D],w2_[col_*D],
x_[(grp_+0U*16U)*64U*D],x_[(grp_+1U*16U)*64U*D],
x_[(grp_+2U*16U)*64U*D],x_[(grp_+3U*16U)*64U*D]);
} else {
piped_.template PrepareWeights<FirstTM>(w1_[col_*D],w2_[col_*D],x_[baseRow_*D]);
}
}
if ASCEND_IS_AIV {
SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
LocalTensor<T> bias(TPosition::VECIN,2U*InSlots*Cap*sizeof(T),2U*N);
DataCopy(bias,b1_[col_],N);DataCopy(bias[N],b2_[col_],N);
if constexpr (FastAiv) {
constexpr uint32_t FirstNR=FirstTM/2U, FirstCap=FirstNR*N;
WaitReady<0U>();
LocalTensor<T> input0(TPosition::VECIN,0,2U*Cap);
DataCopy(input0,slots_[core_*8U*Plane+GetSubBlockIdx()*FirstNR*2U*N],2U*FirstCap);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
}
}
Step<0>();
if ASCEND_IS_AIC {piped_.Drain();}
PipeBarrier<PIPE_ALL>();
}
private:
template <HardEvent E> __aicore__ inline void Sync() {SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0);}
template <uint32_t Slot> __aicore__ inline void Ready() {
constexpr uint32_t id=Slot==0U?0x6:(Slot==1U?0x7:(Slot==2U?0x8:0x9));
CrossCoreSetFlag<0x2,PIPE_FIX>(id);
}
template <uint32_t Slot> __aicore__ inline void WaitReady() {
constexpr uint32_t id=Slot==0U?0x6:(Slot==1U?0x7:(Slot==2U?0x8:0x9));
CrossCoreWaitFlag(id);
}
template <uint32_t Slot> __aicore__ inline void Consumed() {
constexpr uint32_t id=Slot==0U?0xA:(Slot==1U?0xB:(Slot==2U?0xC:0xE));
CrossCoreSetFlag<0x2,PIPE_MTE2>(id);
}
template <uint32_t I> __aicore__ inline void Step() {
constexpr uint32_t Slot=I%4U;
const uint64_t off=(core_*4U+Slot)*2U*Plane;
if constexpr (Groups==4U) {
constexpr uint64_t localRow=(I==0U)?0U:((I==1U)?32U:96U);
constexpr uint32_t curTM=(I==1U)?64U:32U;
const uint64_t row=baseRow_+localRow;
if ASCEND_IS_AIC {
piped_.template Compute<I,curTM>(slots_[off]);
Ready<Slot>();
}
if ASCEND_IS_AIV {
Epilogue<I,curTM>(off,row);
}
} else if constexpr (Groups==10U) {
constexpr uint32_t CurTM=(I<6U)?128U:64U;
const uint64_t row=baseRow_+I*128U;
if ASCEND_IS_AIC {
if constexpr (I<5U) {
piped_.template Prefetch<I+1U,128U>(x_[(row+128U)*D]);
} else if constexpr (I==5U) {
const uint64_t pfRow=(grp_<8U)?(row+128U):baseRow_;
piped_.template Prefetch<6U,64U>(x_[pfRow*D]);
}
piped_.template Compute<I,CurTM>(slots_[off]);
Ready<Slot>();
}
if ASCEND_IS_AIV {
Epilogue<I,CurTM>(off,row);
}
} else {
const uint64_t row=(grp_+I*Groups)*TM;
if ASCEND_IS_AIC {
piped_.template Compute<I>(slots_[off]);
Ready<Slot>();
}
if ASCEND_IS_AIV {
Epilogue<I>(off,row);
}
}
if constexpr (I+1U<Iterations) {Step<I+1U>();}
}
template <uint32_t I,uint32_t CurTM=TM> __aicore__ inline void Epilogue(uint64_t off,uint64_t row) {
constexpr uint32_t ubSlot=I%InSlots;
constexpr event_t vEv=ubSlot==0U?EVENT_ID0:(ubSlot==1U?EVENT_ID1:(ubSlot==2U?EVENT_ID2:EVENT_ID3));
constexpr uint32_t CurNR=CurTM/2U, CurCap=CurNR*N;
const uint32_t begin=GetSubBlockIdx()*CurNR;
LocalTensor<T> input(TPosition::VECIN,0,2U*InSlots*Cap+2U*N);
auto curInput=input[ubSlot*2U*Cap];
auto b1=input[2U*InSlots*Cap];auto b2=input[2U*InSlots*Cap+N];
LocalTensor<T> work(TPosition::VECOUT,(2U*InSlots*Cap+2U*N)*sizeof(T),(3U+OutSlots)*Cap);
auto g=work;auto u=work[Cap];auto sig=work[2U*Cap];auto out=work[(3U+ubSlot)*Cap];
const uint64_t start=off+begin*2U*N;
if constexpr (!FastAiv) {
if constexpr (I>=InSlots) {
WaitFlag<HardEvent::V_MTE2>(vEv);
}
DataCopy(curInput,slots_[start],2U*CurCap);
if constexpr (I+4U<Iterations) {Consumed<I%4U>();}
Sync<HardEvent::MTE2_V>();
} else {
WaitFlag<HardEvent::MTE2_V>(vEv);
if constexpr (I==0U && Iterations>4U) {
Consumed<0U>();
}
}
const BinaryRepeatParams rp{1,1,1,N/16U,2U*N/16U,0};
constexpr uint64_t mask=0;
if constexpr (N!=128U) {SetVectorMask<T,MaskMode::NORMAL>(0,~0ULL);}
Add<T,false>(g,curInput,b1,mask,static_cast<uint8_t>(CurNR),rp);
Add<T,false>(u,curInput[N],b2,mask,static_cast<uint8_t>(CurNR),rp);
if constexpr (I+InSlots<Iterations) {
SetFlag<HardEvent::V_MTE2>(vEv);
}
static_assert(CurCap%128U==0U && CurCap/128U<=255U);
constexpr uint8_t repeats=CurCap/128U;
const UnaryRepeatParams ur{1,1,8,8};
const BinaryRepeatParams br{1,1,1,8,8,8};
if constexpr (N!=128U) {SetVectorMask<T,MaskMode::NORMAL>(~0ULL,~0ULL);}
Muls<T,false>(sig,g,static_cast<T>(-1),mask,repeats,ur);
Exp<T,false>(sig,sig,mask,repeats,ur);
Adds<T,false>(sig,sig,static_cast<T>(1),mask,repeats,ur);
Div<T,false>(sig,g,sig,mask,repeats,br);
if constexpr (I>=OutSlots) {
WaitFlag<HardEvent::MTE3_V>(vEv);
}
Mul<T,false>(out,sig,u,mask,repeats,br);
if constexpr (!FastAiv) {
Sync<HardEvent::V_MTE3>();
DataCopyParams op{CurNR,N/16U,0,(D-N)/16U};
if constexpr (D==N) {DataCopy(y_[(row+begin)*D],out,CurCap);}
else {DataCopy(y_[(row+begin)*D+col_],out,op);}
if constexpr (I+OutSlots<Iterations) {
SetFlag<HardEvent::MTE3_V>(vEv);
}
} else {
SetFlag<HardEvent::V_MTE3>(vEv);
WaitFlag<HardEvent::V_MTE3>(vEv);
DataCopyParams op{CurNR,N/16U,0,(D-N)/16U};
if constexpr (Groups==10U && I==6U) {
if (grp_<8U) {DataCopy(y_[(row+begin)*D+col_],out,op);}
} else if constexpr (D==N) {
DataCopy(y_[(row+begin)*D],out,CurCap);
} else {
DataCopy(y_[(row+begin)*D+col_],out,op);
}
if constexpr (I+OutSlots<Iterations) {
SetFlag<HardEvent::MTE3_V>(vEv);
}
if constexpr (I+1U<Iterations) {
constexpr uint32_t nextSlot=(I+1U)%4U;
constexpr uint32_t nextUbSlot=(I+1U)%InSlots;
constexpr event_t nextEv=nextUbSlot==0U?EVENT_ID0:(nextUbSlot==1U?EVENT_ID1:(nextUbSlot==2U?EVENT_ID2:EVENT_ID3));
WaitReady<nextSlot>();
if constexpr (I+1U>=InSlots) {
WaitFlag<HardEvent::V_MTE2>(nextEv);
}
const uint64_t nextOff=(core_*4U+nextSlot)*2U*Plane;
if constexpr (Groups==4U) {
constexpr uint32_t nextNR=(I==0U)?32U:16U;
DataCopy(input[nextUbSlot*2U*Cap],slots_[nextOff+GetSubBlockIdx()*nextNR*2U*N],2U*nextNR*N);
} else if constexpr (Groups==10U) {
constexpr uint32_t nextNR=(I<5U)?64U:32U;
DataCopy(input[nextUbSlot*2U*Cap],slots_[nextOff+GetSubBlockIdx()*nextNR*2U*N],2U*nextNR*N);
} else {
DataCopy(input[nextUbSlot*2U*Cap],slots_[nextOff+GetSubBlockIdx()*NR*2U*N],2U*Cap);
}
if constexpr (I+5U<Iterations) {
Consumed<nextSlot>();
}
SetFlag<HardEvent::MTE2_V>(nextEv);
}
}
}
DoubleBufferedPairCube<T,TM,D,N,Iterations> piped_;
GlobalTensor<T> x_,w1_,w2_,b1_,b2_,y_,slots_;
uint32_t core_=0,col_=0,grp_=0,baseRow_=0;
};
}
template <typename DT_INPUT_X, int PATH>
__global__ __aicore__ void swi_glu(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2, GM_ADDR bias_2,
GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
if constexpr (PATH>=43) {
if constexpr (PATH==43) {
KERNEL_TASK_TYPE(11009,KERNEL_TYPE_AIV_ONLY);
SubBlockGemv4<DT_INPUT_X> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output);
} else if constexpr (PATH==44) {
KERNEL_TASK_TYPE(11265,KERNEL_TYPE_AIV_ONLY);
AlignedColGemv<DT_INPUT_X,64U,2U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output);
} else if constexpr (PATH==45) {
optiling::PipelinedCubeSwiGlu<DT_INPUT_X,256U,64U,16U,3U,64U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,workspace);
} else if constexpr (PATH==46) {
KERNEL_TASK_TYPE(11777,KERNEL_TYPE_AIV_ONLY);
AlignedRowGemv32<DT_INPUT_X> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output);
} else if constexpr (PATH==47) {
optiling::PipelinedCubeSwiGlu<DT_INPUT_X,128U,64U,16U,4U,128U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,workspace);
} else if constexpr (PATH==48) {
KERNEL_TASK_TYPE(12289,KERNEL_TYPE_AIV_ONLY);
ScalarChannelSwiGlu<DT_INPUT_X> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output);
} else if constexpr (PATH==49) {
KERNEL_TASK_TYPE(12545,KERNEL_TYPE_AIV_ONLY);
PackedSubBlockGemv<DT_INPUT_X> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output);
} else if constexpr (PATH==50) {
KERNEL_TASK_TYPE(12801,KERNEL_TYPE_AIV_ONLY);
PackedMultiBlockGemv<DT_INPUT_X> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output);
} else if constexpr (PATH==51) {
optiling::PipelinedCubeSwiGlu<DT_INPUT_X,256U,128U,20U,7U,128U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,workspace);
}
} else if constexpr (PATH==42) {
KERNEL_TASK_TYPE(10753,KERNEL_TYPE_AIV_ONLY);
REGISTER_TILING_DEFAULT(SwiGluTilingData);
BatchedTinyVectorSwiGlu<DT_INPUT_X,13U,7U,true> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,nullptr);
} else if constexpr (PATH >= 38) {
if constexpr (PATH==38) { KERNEL_TASK_TYPE(9729,KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH==39) { KERNEL_TASK_TYPE(9985,KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH==40) { KERNEL_TASK_TYPE(10241,KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH==41) { KERNEL_TASK_TYPE(10497,KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
if constexpr (PATH==41) {
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData,tiling_data,tiling);
TinyColumnsSwiGlu<DT_INPUT_X,0U,false> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else {
TinyColumnsSwiGlu<DT_INPUT_X,PATH==39?0U:13U,PATH!=40> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,nullptr);
}
} else if constexpr (PATH >= 36) {
if constexpr (PATH==36) { KERNEL_TASK_TYPE(9217,KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH==37) { KERNEL_TASK_TYPE(9473,KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData,tiling_data,tiling);
SmallMVectorSwiGlu<DT_INPUT_X,PATH==36?512U:0U,512U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else if constexpr (PATH >= 32) {
if (workspace == nullptr) { return; }
GM_ADDR userWorkspace=AscendC::GetUserWorkspace(workspace);
if (userWorkspace == nullptr) { return; }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluDirectTilingData,tiling_data,tiling);
constexpr uint32_t fixedD=PATH==32?256U:(PATH==33?128U:0U);
optiling::SlimDirectCube<DT_INPUT_X,64U,fixedD,PATH==35> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,userWorkspace,&tiling_data);
} else if constexpr (PATH >= 26) {
if (workspace == nullptr) { return; }
GM_ADDR userWorkspace=AscendC::GetUserWorkspace(workspace);
if (userWorkspace == nullptr) { return; }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluDirectTilingData,tiling_data,tiling);
constexpr uint32_t fixedD=(PATH==26||PATH==30)?256U:((PATH==27||PATH==31)?128U:0U);
optiling::SlimDirectCube<DT_INPUT_X,128U,fixedD,PATH==29,true,PATH>=30> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,userWorkspace,&tiling_data);
} else if constexpr (PATH >= 23) {
REGISTER_TILING_DEFAULT(SwiGluTilingData);
if constexpr (PATH == 23) { KERNEL_TASK_TYPE(5889, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 24) { KERNEL_TASK_TYPE(6145, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 25) { KERNEL_TASK_TYPE(6401, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 25) {
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData, tiling_data, tiling);
SingleTinySwiGlu<DT_INPUT_X,0U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else {
SingleTinySwiGlu<DT_INPUT_X,PATH==23?13U:1U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,nullptr);
}
} else if constexpr (PATH == 21 || PATH == 22) {
if constexpr (PATH == 21) { KERNEL_TASK_TYPE(5377, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 22) { KERNEL_TASK_TYPE(5633, KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData, tiling_data, tiling);
BatchedTinyVectorSwiGlu<DT_INPUT_X,PATH==21?13U:0U,4U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else if constexpr (PATH >= 17) {
if constexpr (PATH == 17) { KERNEL_TASK_TYPE(4353, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 18) { KERNEL_TASK_TYPE(4609, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 19) { KERNEL_TASK_TYPE(4865, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 20) { KERNEL_TASK_TYPE(5121, KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData, tiling_data, tiling);
constexpr uint32_t fixedD=PATH==17?64U:(PATH==18?128U:(PATH==19?256U:0U));
SmallMVectorSwiGlu<DT_INPUT_X,fixedD> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else if constexpr (PATH >= 14) {
if constexpr (PATH == 14) { KERNEL_TASK_TYPE(3585, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 15) { KERNEL_TASK_TYPE(3841, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 16) { KERNEL_TASK_TYPE(4097, KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData, tiling_data, tiling);
if constexpr (PATH == 15) {
ScalarFeatureSwiGlu<DT_INPUT_X> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else {
TinyVectorSwiGlu<DT_INPUT_X,PATH == 14 ? 13U : 0U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
}
} else if constexpr (PATH == 0 || PATH == 4 || PATH == 5) {
if constexpr (PATH == 0) { KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 4) { KERNEL_TASK_TYPE(1025, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 5) { KERNEL_TASK_TYPE(1281, KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tiling_data, tiling);
constexpr int mode = PATH == 4 ? 1 : (PATH == 5 ? 2 : 0);
KernelSwiGlu<DT_INPUT_X, mode> op;
op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, tiling_data);
op.Process();
} else {
if (workspace == nullptr) { return; }
GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
if (userWorkspace == nullptr) { return; }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
if constexpr (PATH == 2 || PATH == 3 || PATH >= 8) {
GET_TILING_DATA_WITH_STRUCT(SwiGluDirectTilingData, tiling_data, tiling);
constexpr uint32_t tileM = (PATH == 3 || PATH == 9 || PATH == 11 || PATH == 13) ? 128U : 32U;
constexpr uint32_t fixedD = (PATH == 8 || PATH == 9) ? 256U :
((PATH == 10 || PATH == 11) ? 128U : 0U);
optiling::SlimDirectCube<DT_INPUT_X, tileM, fixedD, PATH == 12 || PATH == 13> op;
op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output, userWorkspace, &tiling_data);
} else {
AscendC::TPipe tPipe;
GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tiling_data, tiling);
optiling::RunCube<DT_INPUT_X, PATH == 6 || PATH == 7, PATH == 7>(
input_x, weight_1, bias_1, weight_2, bias_2, output, userWorkspace, &tPipe, &tiling_data);
tPipe.Destroy();
}
}
}
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 11009UL || TILING_KEY_VAR == 11265UL || TILING_KEY_VAR == 11521UL || TILING_KEY_VAR == 11777UL || TILING_KEY_VAR == 12033UL || TILING_KEY_VAR == 12289UL || TILING_KEY_VAR == 12545UL || TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL) && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON)
namespace AscendC {
__aicore__ inline void NoSysWorkspaceForce(GM_ADDR) {}
__aicore__ inline GM_ADDR FastGetUserWorkspace(GM_ADDR workspace) {
return workspace + RESERVED_WORKSPACE;
}
}
#define SetSysWorkspaceForce NoSysWorkspaceForce
#define GetUserWorkspace FastGetUserWorkspace
#define cce_kernel cce_kernel, aligned(512), section(".text.hot")
#if TILING_KEY_VAR == 11009UL || TILING_KEY_VAR == 11265UL || TILING_KEY_VAR == 11777UL || TILING_KEY_VAR == 12289UL || TILING_KEY_VAR == 12545UL || TILING_KEY_VAR == 12801UL
#define set_ffts_base_addr(...) ((void)0)
#endif
#endif
#else
#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "swi_glu_tiling.h"
#include "tiling_key_swi_glu.h"
using namespace AscendC;
namespace {
__aicore__ inline uint32_t AlignUpBytes(uint32_t bytes)
{
return (bytes + 31U) / 32U * 32U;
}
__aicore__ inline uint32_t AlignUp16(uint32_t elems)
{
return (elems + 15U) / 16U * 16U;
}
constexpr uint32_t kChunk16 = 16;
__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
return (a + b - 1) / b;
}
}
template <typename T, bool FillOnes = false>
__aicore__ inline void GateCounter(const LocalTensor<T>& g,const LocalTensor<T>& u,
const LocalTensor<T>& sig,const LocalTensor<T>& out,
const LocalTensor<T>& ones,uint32_t count) {
constexpr uint64_t mask=0;
const UnaryRepeatParams ur{1,1,8,8};
const BinaryRepeatParams br{1,1,1,8,8,8};
SetMaskCount();SetVectorMask<T,MaskMode::COUNTER>(0,count);

Muls<T,false>(sig,g,static_cast<T>(-1),mask,1,ur);
PipeBarrier<PIPE_V>();Exp<T,false>(sig,sig,mask,1,ur);
PipeBarrier<PIPE_V>();Adds<T,false>(sig,sig,static_cast<T>(1),mask,1,ur);
PipeBarrier<PIPE_V>();Reciprocal<T,false>(sig,sig,mask,1,ur);
PipeBarrier<PIPE_V>();Mul<T,false>(out,g,sig,mask,1,br);
PipeBarrier<PIPE_V>();Mul<T,false>(out,out,u,mask,1,br);
SetMaskNorm();ResetMask();
}
template <typename T,uint32_t Count>
__aicore__ inline void GateNormal(const LocalTensor<T>& g,const LocalTensor<T>& u,
const LocalTensor<T>& sig,const LocalTensor<T>& out,const LocalTensor<T>& ones) {
static_assert(Count<=128U);
constexpr uint64_t lo=Count>=64U?~0ULL:((1ULL<<Count)-1ULL);
constexpr uint64_t hi=Count<=64U?0ULL:((1ULL<<(Count-64U))-1ULL);
const UnaryRepeatParams ur{1,1,8,8};const BinaryRepeatParams br{1,1,1,8,8,8};
SetVectorMask<T,MaskMode::NORMAL>(hi,lo);
Muls<T,false>(sig,g,static_cast<T>(-1),static_cast<uint64_t>(0),1,ur);
PipeBarrier<PIPE_V>();Exp<T,false>(sig,sig,static_cast<uint64_t>(0),1,ur);
PipeBarrier<PIPE_V>();Adds<T,false>(sig,sig,static_cast<T>(1),static_cast<uint64_t>(0),1,ur);
PipeBarrier<PIPE_V>();Div<T,false>(sig,ones,sig,static_cast<uint64_t>(0),1,br);
PipeBarrier<PIPE_V>();Mul<T,false>(out,g,sig,static_cast<uint64_t>(0),1,br);
PipeBarrier<PIPE_V>();Mul<T,false>(out,out,u,static_cast<uint64_t>(0),1,br);
ResetMask();
}
template <typename T, uint32_t Count, bool SetMask = true>
__aicore__ inline void FastGate(const LocalTensor<T>& g, const LocalTensor<T>& u,
const LocalTensor<T>& sig, const LocalTensor<T>& out, const LocalTensor<T>& ones) {
static_assert(Count <= 128U);
constexpr uint64_t lo = Count >= 64U ? ~0ULL : ((1ULL << Count) - 1ULL);
constexpr uint64_t hi = Count <= 64U ? 0ULL : (Count >= 128U ? ~0ULL : ((1ULL << (Count > 64U ? Count - 64U : 0U)) - 1ULL));
const UnaryRepeatParams ur{1, 1, 8, 8};
const BinaryRepeatParams br{1, 1, 1, 8, 8, 8};
if constexpr (SetMask) {
SetVectorMask<T, MaskMode::NORMAL>(hi, lo);
}
Muls<T, false>(sig, g, static_cast<T>(-1), static_cast<uint64_t>(0), 1, ur);
Exp<T, false>(sig, sig, static_cast<uint64_t>(0), 1, ur);
Adds<T, false>(sig, sig, static_cast<T>(1), static_cast<uint64_t>(0), 1, ur);
Div<T, false>(sig, g, sig, static_cast<uint64_t>(0), 1, br);
Mul<T, false>(out, sig, u, static_cast<uint64_t>(0), 1, br);
}
template <class DT_INPUT_X, int Mode>
class KernelSwiGlu {
public:
__aicore__ inline KernelSwiGlu() {}
__aicore__ inline void Init(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2, GM_ADDR bias_2,
GM_ADDR output, const SwiGluTilingData &tiling)
{
td_ = &tiling;
const uint64_t xElems = td_->m * td_->d;
const uint64_t wElems = td_->d * td_->d;
xGm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)input_x, xElems);
w1Gm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)weight_1, wElems);
b1Gm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)bias_1, td_->d);
w2Gm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)weight_2, wElems);
b2Gm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)bias_2, td_->d);
outGm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)output, xElems);
pipe_.InitBuffer(stageA_, 1, AlignUpBytes(td_->dAlign * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(stageB_, 1, AlignUpBytes(td_->dAlign * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(outQueue_, 1, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(xfBuf_, AlignUpBytes(td_->dAlign * sizeof(float)));
pipe_.InitBuffer(wfBuf_, AlignUpBytes(td_->dAlign * sizeof(float)));
const uint32_t productElems = td_->dAlign *
(Mode != 0 && td_->dAlign <= 256 ? kChunk16 : 1U);
pipe_.InitBuffer(prodGBuf_, AlignUpBytes(productElems * sizeof(float)));
pipe_.InitBuffer(prodUBuf_, AlignUpBytes(productElems * sizeof(float)));
pipe_.InitBuffer(accGBuf_, AlignUpBytes(td_->tileN * sizeof(float)));
pipe_.InitBuffer(accUBuf_, AlignUpBytes(td_->tileN * sizeof(float)));
pipe_.InitBuffer(accG16Buf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(accU16Buf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(negBuf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(eBuf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(sBuf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
pipe_.InitBuffer(onesBuf_, AlignUpBytes(td_->tileN * sizeof(DT_INPUT_X)));
if constexpr (Mode != 0) {
pipe_.InitBuffer(w1tBuf_, AlignUpBytes(td_->dAlign * kChunk16 * sizeof(float)));
pipe_.InitBuffer(w2tBuf_, AlignUpBytes(td_->dAlign * kChunk16 * sizeof(float)));
pipe_.InitBuffer(wblkQueue_, 1, AlignUpBytes(kChunk16 * td_->dAlign * sizeof(DT_INPUT_X)));
}
}
__aicore__ inline void Process()
{
const uint64_t core = GetBlockIdx();
if (core >= td_->usedCoreNum) {
return;
}
if constexpr (Mode == 2) {
const uint32_t d = static_cast<uint32_t>(td_->d);
const uint32_t columns = Ceiling(d, kChunk16);
const uint64_t tasks = td_->m * columns;
for (uint64_t task = core; task < tasks; task += td_->usedCoreNum) {
const uint64_t row = task / columns;
const uint32_t c0 = static_cast<uint32_t>(task % columns) * kChunk16;
const uint32_t chunk = d - c0 < kChunk16 ? d - c0 : kChunk16;
PrepareWt(c0, chunk, d);
ProcessColTileT(row, c0, chunk, d);
}
} else if constexpr (Mode == 1) {
const uint32_t d = static_cast<uint32_t>(td_->d);
for (uint32_t c0 = 0; c0 < d; c0 += kChunk16) {
const uint32_t chunk = d - c0 < kChunk16 ? d - c0 : kChunk16;
PrepareWt(c0, chunk, d);
for (uint64_t row = core; row < td_->m; row += td_->usedCoreNum) {
ProcessColTileT(row, c0, chunk, d);
}
}
} else {
for (uint64_t row = core; row < td_->m; row += td_->usedCoreNum) {
ProcessRow(row);
}
}
}
private:
template <HardEvent event>
__aicore__ inline void SyncPipe()
{
const event_t id = static_cast<event_t>(pipe_.FetchEventID(event));
SetFlag<event>(id);
WaitFlag<event>(id);
}
__aicore__ inline void ProcessRow(uint64_t row)
{
const uint32_t d = static_cast<uint32_t>(td_->d);
const uint32_t dAlign = td_->dAlign;
const uint32_t tileN = td_->tileN;
LocalTensor<DT_INPUT_X> xTile = stageA_.AllocTensor<DT_INPUT_X>();
DataCopyExtParams xCopy{1, static_cast<uint32_t>(d * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> xPad{true, 0, static_cast<uint8_t>(dAlign - d),
static_cast<DT_INPUT_X>(0)};
DataCopyPad(xTile, xGm_[row * d], xCopy, xPad);
stageA_.EnQue(xTile);
xTile = stageA_.DeQue<DT_INPUT_X>();
LocalTensor<float> xf = xfBuf_.Get<float>();
Cast(xf, xTile, RoundMode::CAST_NONE, dAlign);
stageA_.FreeTensor(xTile);
for (uint32_t c0 = 0; c0 < d; c0 += tileN) {
const uint32_t chunk = d - c0 < tileN ? d - c0 : tileN;
ProcessColTile(row, c0, chunk, d, dAlign, xf);
}
}
__aicore__ inline void ProcessColTile(uint64_t row, uint32_t c0, uint32_t chunk, uint32_t d, uint32_t dAlign,
const LocalTensor<float> &xf)
{
LocalTensor<float> wf = wfBuf_.Get<float>();
LocalTensor<float> prodG = prodGBuf_.Get<float>();
LocalTensor<float> prodU = prodUBuf_.Get<float>();
LocalTensor<float> accG = accGBuf_.Get<float>();
LocalTensor<float> accU = accUBuf_.Get<float>();
LocalTensor<DT_INPUT_X> accG16 = accG16Buf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> accU16 = accU16Buf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> negBuf = negBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> eBuf = eBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> sBuf = sBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> onesBuf = onesBuf_.Get<DT_INPUT_X>();
DataCopyExtParams wCopy{1, static_cast<uint32_t>(d * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> wPad{true, 0, static_cast<uint8_t>(dAlign - d),
static_cast<DT_INPUT_X>(0)};
for (uint32_t j = 0; j < chunk; ++j) {
const uint64_t wBase = static_cast<uint64_t>(c0 + j) * d;
LocalTensor<DT_INPUT_X> w1 = stageA_.AllocTensor<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> w2 = stageB_.AllocTensor<DT_INPUT_X>();
DataCopyPad(w1, w1Gm_[wBase], wCopy, wPad);
DataCopyPad(w2, w2Gm_[wBase], wCopy, wPad);
stageA_.EnQue(w1);
stageB_.EnQue(w2);
w1 = stageA_.DeQue<DT_INPUT_X>();
w2 = stageB_.DeQue<DT_INPUT_X>();
Cast(wf, w1, RoundMode::CAST_NONE, dAlign);
Mul(prodG, xf, wf, dAlign);
Cast(wf, w2, RoundMode::CAST_NONE, dAlign);
Mul(prodU, xf, wf, dAlign);
stageA_.FreeTensor(w1);
stageB_.FreeTensor(w2);
ReduceSum(accG[j], prodG, wf, dAlign);
ReduceSum(accU[j], prodU, wf, dAlign);
}
DataCopyExtParams bCopy{1, static_cast<uint32_t>(chunk * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> bPad{true, 0, static_cast<uint8_t>(AlignUp16(chunk) - chunk),
static_cast<DT_INPUT_X>(0)};
LocalTensor<DT_INPUT_X> b1Tile = stageA_.AllocTensor<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> b2Tile = stageB_.AllocTensor<DT_INPUT_X>();
DataCopyPad(b1Tile, b1Gm_[c0], bCopy, bPad);
DataCopyPad(b2Tile, b2Gm_[c0], bCopy, bPad);
stageA_.EnQue(b1Tile);
stageB_.EnQue(b2Tile);
b1Tile = stageA_.DeQue<DT_INPUT_X>();
b2Tile = stageB_.DeQue<DT_INPUT_X>();
Cast(accG16, accG, RoundMode::CAST_RINT, chunk);
Cast(accU16, accU, RoundMode::CAST_RINT, chunk);
Add(accG16, accG16, b1Tile, chunk);
stageA_.FreeTensor(b1Tile);
Add(accU16, accU16, b2Tile, chunk);
stageB_.FreeTensor(b2Tile);
Duplicate(onesBuf, static_cast<DT_INPUT_X>(1.0f), chunk);
Muls(negBuf, accG16, (DT_INPUT_X)(-1.0), chunk);
Exp(eBuf, negBuf, chunk);
Adds(eBuf, eBuf, (DT_INPUT_X)(1.0), chunk);
Div(sBuf, onesBuf, eBuf, chunk);
Mul(accG16, accG16, sBuf, chunk);
LocalTensor<DT_INPUT_X> outTile = outQueue_.AllocTensor<DT_INPUT_X>();
Mul(outTile, accG16, accU16, chunk);
outQueue_.EnQue(outTile);
outTile = outQueue_.DeQue<DT_INPUT_X>();
DataCopyExtParams oCopy{1, static_cast<uint32_t>(chunk * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPad(outGm_[row * d + c0], outTile, oCopy);
outQueue_.FreeTensor(outTile);
}
__aicore__ inline void PrepareOneWeight(const GlobalTensor<DT_INPUT_X> &weight,
const LocalTensor<float> &cached,
uint32_t c0, uint32_t chunk, uint32_t d)
{
LocalTensor<DT_INPUT_X> wblk = wblkQueue_.AllocTensor<DT_INPUT_X>();
DataCopyExtParams blkCopy{static_cast<uint16_t>(chunk),
static_cast<uint32_t>(d * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> blkPad{true, 0, static_cast<uint8_t>(td_->dAlign - d),
static_cast<DT_INPUT_X>(0)};
DataCopyPad(wblk, weight[static_cast<uint64_t>(c0) * d], blkCopy, blkPad);
wblkQueue_.EnQue(wblk);
wblk = wblkQueue_.DeQue<DT_INPUT_X>();
Cast(cached, wblk, RoundMode::CAST_NONE, chunk * td_->dAlign);
wblkQueue_.FreeTensor(wblk);
}
__aicore__ inline void PrepareWt(uint32_t c0, uint32_t chunk, uint32_t d)
{
PrepareOneWeight(w1Gm_, w1tBuf_.Get<float>(), c0, chunk, d);
PrepareOneWeight(w2Gm_, w2tBuf_.Get<float>(), c0, chunk, d);
}
__aicore__ inline void ProcessColTileT(uint64_t row, uint32_t c0, uint32_t chunk, uint32_t d)
{
const uint32_t dAlign = td_->dAlign;
LocalTensor<float> xf = xfBuf_.Get<float>();
LocalTensor<float> wf = wfBuf_.Get<float>();
LocalTensor<float> accG = accGBuf_.Get<float>();
LocalTensor<float> accU = accUBuf_.Get<float>();
LocalTensor<DT_INPUT_X> accG16 = accG16Buf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> accU16 = accU16Buf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> negBuf = negBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> eBuf = eBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> sBuf = sBuf_.Get<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> onesBuf = onesBuf_.Get<DT_INPUT_X>();
LocalTensor<float> w1t = w1tBuf_.Get<float>();
LocalTensor<float> w2t = w2tBuf_.Get<float>();
LocalTensor<float> prodG = prodGBuf_.Get<float>();
LocalTensor<float> prodU = prodUBuf_.Get<float>();
LocalTensor<DT_INPUT_X> xTile = stageA_.AllocTensor<DT_INPUT_X>();
DataCopyExtParams xCopy{1, static_cast<uint32_t>(d * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> xPad{true, 0, static_cast<uint8_t>(dAlign - d),
static_cast<DT_INPUT_X>(0)};
DataCopyPad(xTile, xGm_[row * d], xCopy, xPad);
stageA_.EnQue(xTile);
xTile = stageA_.DeQue<DT_INPUT_X>();
Cast(xf, xTile, RoundMode::CAST_NONE, dAlign);
stageA_.FreeTensor(xTile);
LocalTensor<DT_INPUT_X> b1Tile = stageA_.AllocTensor<DT_INPUT_X>();
DataCopyExtParams bCopy{1, static_cast<uint32_t>(chunk * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPadExtParams<DT_INPUT_X> bPad{true, 0, static_cast<uint8_t>(AlignUp16(chunk) - chunk),
static_cast<DT_INPUT_X>(0)};
DataCopyPad(b1Tile, b1Gm_[c0], bCopy, bPad);
stageA_.EnQue(b1Tile);
b1Tile = stageA_.DeQue<DT_INPUT_X>();
LocalTensor<DT_INPUT_X> b2Tile = stageB_.AllocTensor<DT_INPUT_X>();
DataCopyPad(b2Tile, b2Gm_[c0], bCopy, bPad);
stageB_.EnQue(b2Tile);
b2Tile = stageB_.DeQue<DT_INPUT_X>();
if (dAlign <= 256U) {
const uint8_t rowStride = static_cast<uint8_t>(dAlign / 8U);
const BinaryRepeatParams mulParams{1, 1, 1, rowStride, 0, rowStride};
for (uint32_t k = 0; k < dAlign; k += 64U) {
const uint32_t width = dAlign - k < 64U ? dAlign - k : 64U;
Mul(prodG[k], xf[k], w1t[k], static_cast<uint64_t>(width),
static_cast<uint8_t>(chunk), mulParams);
Mul(prodU[k], xf[k], w2t[k], static_cast<uint64_t>(width),
static_cast<uint8_t>(chunk), mulParams);
if (k == 0) {
WholeReduceSum(accG, prodG[k], width, chunk, 1, 1, rowStride);
WholeReduceSum(accU, prodU[k], width, chunk, 1, 1, rowStride);
} else {
WholeReduceSum(wf, prodG[k], width, chunk, 1, 1, rowStride);
Add(accG, accG, wf, chunk);
WholeReduceSum(wf, prodU[k], width, chunk, 1, 1, rowStride);
Add(accU, accU, wf, chunk);
}
}
} else {
for (uint32_t j = 0; j < chunk; ++j) {
Mul(prodG, xf, w1t[j * dAlign], dAlign);
Mul(prodU, xf, w2t[j * dAlign], dAlign);
ReduceSum(accG[j], prodG, wf, dAlign);
ReduceSum(accU[j], prodU, wf, dAlign);
}
}
Cast(accG16, accG, RoundMode::CAST_RINT, chunk);
Cast(accU16, accU, RoundMode::CAST_RINT, chunk);
Add(accG16, accG16, b1Tile, chunk);
stageA_.FreeTensor(b1Tile);
Add(accU16, accU16, b2Tile, chunk);
stageB_.FreeTensor(b2Tile);
Duplicate(onesBuf, static_cast<DT_INPUT_X>(1.0f), chunk);
Muls(negBuf, accG16, (DT_INPUT_X)(-1.0), chunk);
Exp(eBuf, negBuf, chunk);
Adds(eBuf, eBuf, (DT_INPUT_X)(1.0), chunk);
Div(sBuf, onesBuf, eBuf, chunk);
Mul(accG16, accG16, sBuf, chunk);
LocalTensor<DT_INPUT_X> outTile = outQueue_.AllocTensor<DT_INPUT_X>();
Mul(outTile, accG16, accU16, chunk);
outQueue_.EnQue(outTile);
outTile = outQueue_.DeQue<DT_INPUT_X>();
DataCopyExtParams oCopy{1, static_cast<uint32_t>(chunk * sizeof(DT_INPUT_X)), 0, 0, 0};
DataCopyPad(outGm_[row * d + c0], outTile, oCopy);
outQueue_.FreeTensor(outTile);
}
private:
TPipe pipe_;
TQue<TPosition::VECIN, 1> stageA_;
TQue<TPosition::VECIN, 1> stageB_;
TQue<TPosition::VECOUT, 1> outQueue_;
TBuf<TPosition::VECCALC> xfBuf_;
TBuf<TPosition::VECCALC> wfBuf_;
TBuf<TPosition::VECCALC> prodGBuf_;
TBuf<TPosition::VECCALC> prodUBuf_;
TBuf<TPosition::VECCALC> accGBuf_;
TBuf<TPosition::VECCALC> accUBuf_;
TBuf<TPosition::VECCALC> accG16Buf_;
TBuf<TPosition::VECCALC> accU16Buf_;
TBuf<TPosition::VECCALC> negBuf_;
TBuf<TPosition::VECCALC> eBuf_;
TBuf<TPosition::VECCALC> sBuf_;
TBuf<TPosition::VECCALC> onesBuf_;
TBuf<TPosition::VECCALC> w1tBuf_;
TBuf<TPosition::VECCALC> w2tBuf_;
TQue<QuePosition::VECIN, 1> wblkQueue_;
GlobalTensor<DT_INPUT_X> xGm_;
GlobalTensor<DT_INPUT_X> w1Gm_;
GlobalTensor<DT_INPUT_X> b1Gm_;
GlobalTensor<DT_INPUT_X> w2Gm_;
GlobalTensor<DT_INPUT_X> b2Gm_;
GlobalTensor<DT_INPUT_X> outGm_;
const SwiGluTilingData *td_ = nullptr;
};
template <typename T, uint32_t FixedD>
class TinyVectorSwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
GM_ADDR y, const SwiGluTinyTilingData *td) {
const uint32_t d = FixedD != 0 ? FixedD : static_cast<uint32_t>(td->d);
GlobalTensor<T> gx, gw1, gw2, gb1, gb2, gy;
gx.SetGlobalBuffer((__gm__ T*)x,td->m*d); gy.SetGlobalBuffer((__gm__ T*)y,td->m*d);
gw1.SetGlobalBuffer((__gm__ T*)w1,d*d); gw2.SetGlobalBuffer((__gm__ T*)w2,d*d);
gb1.SetGlobalBuffer((__gm__ T*)b1,d); gb2.SetGlobalBuffer((__gm__ T*)b2,d);
LocalTensor<T> hw1(TPosition::VECIN,0,256), hw2(TPosition::VECIN,512,256);
LocalTensor<T> hx(TPosition::VECIN,1024,16), hb1(TPosition::VECIN,1056,16), hb2(TPosition::VECIN,1088,16);
LocalTensor<float> fw1(TPosition::VECCALC,1152,256), fw2(TPosition::VECCALC,2176,256);
LocalTensor<float> fx(TPosition::VECCALC,3200,16), pg(TPosition::VECCALC,3264,256), pu(TPosition::VECCALC,4288,256);
LocalTensor<float> ag(TPosition::VECCALC,5312,16), au(TPosition::VECCALC,5376,16);
LocalTensor<T> g(TPosition::VECCALC,5440,16), u(TPosition::VECCALC,5472,16);
LocalTensor<T> sig(TPosition::VECCALC,5504,16), ones(TPosition::VECCALC,5536,16), out(TPosition::VECOUT,5568,16);
SetAtomicNone(); SetMaskNorm(); SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams wc{static_cast<uint16_t>(d),d*2U,0,0,0};
DataCopyExtParams rc{1,d*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(16U-d),static_cast<T>(0)};
DataCopyPad(hw1,gw1,wc,pad); DataCopyPad(hw2,gw2,wc,pad);
DataCopyPad(hb1,gb1,rc,pad); DataCopyPad(hb2,gb2,rc,pad);
Sync<HardEvent::MTE2_V>();
Cast(fw1,hw1,RoundMode::CAST_NONE,d*16U); Cast(fw2,hw2,RoundMode::CAST_NONE,d*16U);
Duplicate(ones,static_cast<T>(1),16);
const BinaryRepeatParams rp{1,1,1,2,0,2};
for (uint64_t row=GetBlockIdx();row<td->m;row+=GetBlockNum()) {
DataCopyPad(hx,gx[row*d],rc,pad); Sync<HardEvent::MTE2_V>();
Cast(fx,hx,RoundMode::CAST_NONE,16); PipeBarrier<PIPE_V>();
Mul(pg,fx,fw1,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
Mul(pu,fx,fw2,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
PipeBarrier<PIPE_V>();
WholeReduceSum(ag,pg,16,d,1,1,2); WholeReduceSum(au,pu,16,d,1,1,2);
PipeBarrier<PIPE_V>();
Cast(g,ag,RoundMode::CAST_RINT,d); Cast(u,au,RoundMode::CAST_RINT,d);
PipeBarrier<PIPE_V>(); Add(g,g,hb1,d); Add(u,u,hb2,d);
PipeBarrier<PIPE_V>(); GateCounter<T>(g,u,sig,out,ones,d);
Sync<HardEvent::V_MTE3>(); DataCopyPad(gy[row*d],out,rc);
Sync<HardEvent::MTE3_MTE2>();
}
}
private:
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
};
template <typename T, uint32_t FixedD, uint32_t Rows = 1, bool SingleBatch = false>
class BatchedTinyVectorSwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
GM_ADDR y, const SwiGluTinyTilingData *td) {
const uint32_t d=FixedD!=0?FixedD:static_cast<uint32_t>(td->d);
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gy.SetGlobalBuffer((__gm__ T*)y);
gw1.SetGlobalBuffer((__gm__ T*)w1,d*d);gw2.SetGlobalBuffer((__gm__ T*)w2,d*d);
gb1.SetGlobalBuffer((__gm__ T*)b1,d);gb2.SetGlobalBuffer((__gm__ T*)b2,d);
LocalTensor<T> hw1(TPosition::VECIN,0,256),hw2(TPosition::VECIN,512,256);
LocalTensor<T> hx(TPosition::VECIN,1024,Rows*16),hb1(TPosition::VECIN,1024+Rows*32,16),hb2(TPosition::VECIN,1056+Rows*32,16);
constexpr uint32_t fb=1088+Rows*32;
LocalTensor<float> fw1(TPosition::VECCALC,fb,256),fw2(TPosition::VECCALC,fb+1024,256);
LocalTensor<float> fx(TPosition::VECCALC,fb+2048,Rows*16);
constexpr uint32_t pb=fb+2048+Rows*64;
LocalTensor<float> pg(TPosition::VECCALC,pb,256),pu(TPosition::VECCALC,pb+1024,256);
constexpr uint32_t ab=pb+2048;
LocalTensor<float> ag(TPosition::VECCALC,ab,Rows*16),au(TPosition::VECCALC,ab+Rows*64,Rows*16);
constexpr uint32_t hb=ab+Rows*128;
LocalTensor<T> g(TPosition::VECCALC,hb,Rows*16),u(TPosition::VECCALC,hb+Rows*32,Rows*16);
LocalTensor<T> sig(TPosition::VECCALC,hb+Rows*64,Rows*16),ones(TPosition::VECCALC,hb+Rows*96,Rows*16),out(TPosition::VECOUT,hb+Rows*128,Rows*16);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams wc{static_cast<uint16_t>(d),d*2U,0,0,0},bc{1,d*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(16U-d),static_cast<T>(0)};
DataCopyPad(hw1,gw1,wc,pad);DataCopyPad(hw2,gw2,wc,pad);
DataCopyPad(hb1,gb1,bc,pad);DataCopyPad(hb2,gb2,bc,pad);
if constexpr (SingleBatch) {
DataCopyExtParams rc{static_cast<uint16_t>(Rows),d*2U,0,0,0};
DataCopyPad(hx,gx[static_cast<uint64_t>(GetBlockIdx())*Rows*d],rc,pad);
}
Sync<HardEvent::MTE2_V>();
Cast(fw1,hw1,RoundMode::CAST_NONE,d*16U);Cast(fw2,hw2,RoundMode::CAST_NONE,d*16U);
Duplicate(ones,static_cast<T>(1),Rows*16U);
Duplicate(ag,0.0f,Rows*16U);Duplicate(au,0.0f,Rows*16U);
const BinaryRepeatParams rp{1,1,1,2,0,2};
const BinaryRepeatParams bp{1,1,1,1,1,0};
const uint64_t first=static_cast<uint64_t>(GetBlockIdx())*Rows;
const uint64_t stride=SingleBatch?0U:static_cast<uint64_t>(GetBlockNum())*Rows;
uint64_t groups=1U;
if constexpr (!SingleBatch) { groups=(td->m-first+stride-1U)/stride; }
for (uint64_t group=0;group<groups;++group) {
const uint64_t row=first+group*stride;
const uint32_t nr=SingleBatch?Rows:(td->m-row<Rows?static_cast<uint32_t>(td->m-row):Rows);
DataCopyExtParams rc{static_cast<uint16_t>(nr),d*2U,0,0,0};
if constexpr (!SingleBatch) { DataCopyPad(hx,gx[row*d],rc,pad);Sync<HardEvent::MTE2_V>(); }
Cast(fx,hx,RoundMode::CAST_NONE,nr*16U);PipeBarrier<PIPE_V>();
#pragma unroll
for (uint32_t i=0;i<nr;++i) {
Mul(pg,fx[i*16U],fw1,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
Mul(pu,fx[i*16U],fw2,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
PipeBarrier<PIPE_V>();
WholeReduceSum(ag[i*16U],pg,16,d,1,1,2);WholeReduceSum(au[i*16U],pu,16,d,1,1,2);
PipeBarrier<PIPE_V>();
}
Cast(g,ag,RoundMode::CAST_RINT,nr*16U);Cast(u,au,RoundMode::CAST_RINT,nr*16U);
PipeBarrier<PIPE_V>();
Add(g,g,hb1,static_cast<uint64_t>(d),static_cast<uint8_t>(nr),bp);
Add(u,u,hb2,static_cast<uint64_t>(d),static_cast<uint8_t>(nr),bp);
PipeBarrier<PIPE_V>();GateCounter<T>(g,u,sig,out,ones,nr*16U);
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[row*d],out,rc);
if constexpr (!SingleBatch) { Sync<HardEvent::MTE3_MTE2>(); }
}
if constexpr (SingleBatch) { PipeBarrier<PIPE_ALL>(); }
}
private:
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0); }
};
template <typename T>
class ScalarFeatureSwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
GM_ADDR y, const SwiGluTinyTilingData *td) {
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x,td->m);gy.SetGlobalBuffer((__gm__ T*)y,td->m);
gw1.SetGlobalBuffer((__gm__ T*)w1,1);gw2.SetGlobalBuffer((__gm__ T*)w2,1);
gb1.SetGlobalBuffer((__gm__ T*)b1,1);gb2.SetGlobalBuffer((__gm__ T*)b2,1);
LocalTensor<T> hx(TPosition::VECIN,0,256), params(TPosition::VECIN,512,64);
LocalTensor<float> fx(TPosition::VECCALC,640,256), fg(TPosition::VECCALC,1664,256), fu(TPosition::VECCALC,2688,256);
LocalTensor<T> g(TPosition::VECCALC,3712,256), u(TPosition::VECCALC,4224,256);
LocalTensor<T> sig(TPosition::VECCALC,4736,256), ones(TPosition::VECCALC,5248,256), out(TPosition::VECOUT,5760,256);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams pc{1,2,0,0,0};
DataCopyPadExtParams<T> pp{true,0,15,static_cast<T>(0)};
DataCopyPad(params,gw1,pc,pp);DataCopyPad(params[16],gw2,pc,pp);
DataCopyPad(params[32],gb1,pc,pp);DataCopyPad(params[48],gb2,pc,pp);
Sync<HardEvent::MTE2_S>();
const float a=static_cast<float>(params.GetValue(0)),b=static_cast<float>(params.GetValue(16));
const T c=params.GetValue(32),e=params.GetValue(48);
Duplicate(ones,static_cast<T>(1),256);
for (uint64_t start=static_cast<uint64_t>(GetBlockIdx())*256U;start<td->m;start+=static_cast<uint64_t>(GetBlockNum())*256U) {
const uint32_t n=td->m-start<256U?static_cast<uint32_t>(td->m-start):256U;
DataCopyExtParams cp{1,n*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>((16U-n%16U)%16U),static_cast<T>(0)};
DataCopyPad(hx,gx[start],cp,pad);Sync<HardEvent::MTE2_V>();
Cast(fx,hx,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();
Muls(fg,fx,a,n);Muls(fu,fx,b,n);PipeBarrier<PIPE_V>();
Cast(g,fg,RoundMode::CAST_RINT,n);Cast(u,fu,RoundMode::CAST_RINT,n);
PipeBarrier<PIPE_V>();Adds(g,g,c,n);Adds(u,u,e,n);
PipeBarrier<PIPE_V>(); GateCounter<T>(g,u,sig,out,ones,n);
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[start],out,cp);
Sync<HardEvent::MTE3_MTE2>();
}
}
private:
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
};
template <typename T, uint32_t FixedD, uint32_t DynamicMaxK = 256U>
class SmallMVectorSwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
GM_ADDR y, const SwiGluTinyTilingData *td) {
constexpr uint32_t maxK=FixedD==0?DynamicMaxK:FixedD;
constexpr uint32_t wElems=16U*maxK;
constexpr uint32_t fBase=4U*wElems+2U*maxK+64U;
const uint32_t d=FixedD!=0?FixedD:static_cast<uint32_t>(td->d);
const uint32_t ka=(d+15U)/16U*16U;
const uint32_t columns=(d+15U)/16U;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x,td->m*d);gy.SetGlobalBuffer((__gm__ T*)y,td->m*d);
gw1.SetGlobalBuffer((__gm__ T*)w1,d*d);gw2.SetGlobalBuffer((__gm__ T*)w2,d*d);
gb1.SetGlobalBuffer((__gm__ T*)b1,d);gb2.SetGlobalBuffer((__gm__ T*)b2,d);
LocalTensor<T> hw1(TPosition::VECIN,0,wElems),hw2(TPosition::VECIN,2U*wElems,wElems);
LocalTensor<T> hx(TPosition::VECIN,4U*wElems,maxK);
LocalTensor<T> hb1(TPosition::VECIN,4U*wElems+2U*maxK,16),hb2(TPosition::VECIN,4U*wElems+2U*maxK+32U,16);
LocalTensor<float> fw1(TPosition::VECCALC,fBase,wElems),fw2(TPosition::VECCALC,fBase+4U*wElems,wElems);
LocalTensor<float> pg(TPosition::VECCALC,fBase+8U*wElems,wElems),pu(TPosition::VECCALC,fBase+12U*wElems,wElems);
constexpr uint32_t xBase=fBase+16U*wElems;
LocalTensor<float> fx(TPosition::VECCALC,xBase,maxK);
constexpr uint32_t aBase=xBase+4U*maxK;
LocalTensor<float> ag(TPosition::VECCALC,aBase,16),au(TPosition::VECCALC,aBase+64U,16);
LocalTensor<float> sg(TPosition::VECCALC,aBase+128U,16),su(TPosition::VECCALC,aBase+192U,16);
constexpr uint32_t hBase=aBase+256U;
LocalTensor<T> g(TPosition::VECCALC,hBase,16),u(TPosition::VECCALC,hBase+32U,16);
LocalTensor<T> sig(TPosition::VECCALC,hBase+64U,16),ones(TPosition::VECCALC,hBase+96U,16),out(TPosition::VECOUT,hBase+128U,16);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
Duplicate(ones,static_cast<T>(1),16);
for (uint64_t task=GetBlockIdx();task<td->m*columns;task+=GetBlockNum()) {
const uint64_t row=task/columns;
const uint32_t col=task%columns*16U;
const uint32_t cols=d-col<16U?d-col:16U;
DataCopyExtParams wc{static_cast<uint16_t>(cols),d*2U,0,0,0};
DataCopyExtParams xc{1,d*2U,0,0,0},bc{1,cols*2U,0,0,0};
DataCopyPadExtParams<T> wp{true,0,static_cast<uint8_t>(ka-d),static_cast<T>(0)};
DataCopyPadExtParams<T> bp{true,0,static_cast<uint8_t>(16U-cols),static_cast<T>(0)};
DataCopyPad(hw1,gw1[static_cast<uint64_t>(col)*d],wc,wp);
DataCopyPad(hw2,gw2[static_cast<uint64_t>(col)*d],wc,wp);
DataCopyPad(hx,gx[row*d],xc,wp);
DataCopyPad(hb1,gb1[col],bc,bp);DataCopyPad(hb2,gb2[col],bc,bp);
Sync<HardEvent::MTE2_V>();
Cast(fw1,hw1,RoundMode::CAST_NONE,cols*ka);Cast(fw2,hw2,RoundMode::CAST_NONE,cols*ka);
Cast(fx,hx,RoundMode::CAST_NONE,ka);PipeBarrier<PIPE_V>();
Dot<true>(ag,au,sg,su,pg,pu,fx,fw1,fw2,ka,0,cols);
for (uint32_t k=64U;k<ka;k+=64U) { Dot<false>(ag,au,sg,su,pg,pu,fx,fw1,fw2,ka,k,cols); }
Cast(g,ag,RoundMode::CAST_RINT,cols);Cast(u,au,RoundMode::CAST_RINT,cols);
PipeBarrier<PIPE_V>();Add(g,g,hb1,cols);Add(u,u,hb2,cols);
PipeBarrier<PIPE_V>(); GateCounter<T>(g,u,sig,out,ones,cols);
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[row*d+col],out,bc);
Sync<HardEvent::MTE3_MTE2>();
}
}
private:
template <bool First> __aicore__ inline void Dot(
const LocalTensor<float>& ag,const LocalTensor<float>& au,const LocalTensor<float>& sg,const LocalTensor<float>& su,
const LocalTensor<float>& pg,const LocalTensor<float>& pu,const LocalTensor<float>& fx,
const LocalTensor<float>& fw1,const LocalTensor<float>& fw2,uint32_t ka,uint32_t k,uint32_t cols) {
const uint32_t width=ka-k<64U?ka-k:64U;
const uint8_t stride=static_cast<uint8_t>(ka/8U);
const BinaryRepeatParams rp{1,1,1,stride,0,stride};
Mul(pg[k],fx[k],fw1[k],static_cast<uint64_t>(width),static_cast<uint8_t>(cols),rp);
Mul(pu[k],fx[k],fw2[k],static_cast<uint64_t>(width),static_cast<uint8_t>(cols),rp);
PipeBarrier<PIPE_V>();
if constexpr (First) {
WholeReduceSum(ag,pg[k],width,cols,1,1,stride);WholeReduceSum(au,pu[k],width,cols,1,1,stride);
PipeBarrier<PIPE_V>();
} else {
WholeReduceSum(sg,pg[k],width,cols,1,1,stride);WholeReduceSum(su,pu[k],width,cols,1,1,stride);
PipeBarrier<PIPE_V>();Add(ag,ag,sg,cols);Add(au,au,su,cols);PipeBarrier<PIPE_V>();
}
}
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0); }
};
template <typename T, uint32_t FixedD>
class SingleTinySwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y,const SwiGluTinyTilingData *td) {
const uint32_t d=FixedD!=0?FixedD:static_cast<uint32_t>(td->d);
const uint64_t row=GetBlockIdx();
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gy.SetGlobalBuffer((__gm__ T*)y);
gw1.SetGlobalBuffer((__gm__ T*)w1,d*d);gw2.SetGlobalBuffer((__gm__ T*)w2,d*d);
gb1.SetGlobalBuffer((__gm__ T*)b1,d);gb2.SetGlobalBuffer((__gm__ T*)b2,d);
LocalTensor<T> hw1(TPosition::VECIN,0,256),hw2(TPosition::VECIN,512,256);
LocalTensor<T> hx(TPosition::VECIN,1024,16),hb1(TPosition::VECIN,1056,16),hb2(TPosition::VECIN,1088,16);
LocalTensor<float> fw1(TPosition::VECCALC,1152,256),fw2(TPosition::VECCALC,2176,256);
LocalTensor<float> fx(TPosition::VECCALC,3200,16),pg(TPosition::VECCALC,3264,256),pu(TPosition::VECCALC,4288,256);
LocalTensor<float> ag(TPosition::VECCALC,5312,16),au(TPosition::VECCALC,5376,16);
LocalTensor<T> g(TPosition::VECCALC,5440,16),u(TPosition::VECCALC,5472,16);
LocalTensor<T> sig(TPosition::VECCALC,5504,16),ones(TPosition::VECCALC,5536,16),out(TPosition::VECOUT,5568,16);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams wc{static_cast<uint16_t>(d),d*2U,0,0,0},rc{1,d*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(16U-d),static_cast<T>(0)};
DataCopyPad(hw1,gw1,wc,pad);DataCopyPad(hw2,gw2,wc,pad);
DataCopyPad(hb1,gb1,rc,pad);DataCopyPad(hb2,gb2,rc,pad);DataCopyPad(hx,gx[row*d],rc,pad);
Sync<HardEvent::MTE2_V>();
Duplicate(ones,static_cast<T>(1),16);
if constexpr (FixedD==1) {
Mul(g,hx,hw1,1);Mul(u,hx,hw2,1);
} else {
Cast(fw1,hw1,RoundMode::CAST_NONE,d*16U);Cast(fw2,hw2,RoundMode::CAST_NONE,d*16U);
Cast(fx,hx,RoundMode::CAST_NONE,16);
const BinaryRepeatParams rp{1,1,1,2,0,2};
Mul(pg,fx,fw1,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
Mul(pu,fx,fw2,static_cast<uint64_t>(16),static_cast<uint8_t>(d),rp);
WholeReduceSum(ag,pg,16,d,1,1,2);WholeReduceSum(au,pu,16,d,1,1,2);
Cast(g,ag,RoundMode::CAST_RINT,d);Cast(u,au,RoundMode::CAST_RINT,d);
}
Add(g,g,hb1,d);Add(u,u,hb2,d);
if constexpr (FixedD != 0) {
FastGate<T, FixedD>(g, u, sig, out, ones);
} else {
GateCounter<T>(g, u, sig, out, ones, d);
}
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[row*d],out,rc);
}
private:
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0); }
};
template <typename T,uint32_t FixedD,bool OneRow>
class TinyColumnsSwiGlu {
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y,const SwiGluTinyTilingData* td) {
const uint32_t d=FixedD!=0?FixedD:(OneRow?GetBlockNum():static_cast<uint32_t>(td->d));
const uint32_t task=GetBlockIdx();
const uint64_t row=OneRow?0U:task/d;
const uint32_t col=OneRow?task:task%d;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gw1.SetGlobalBuffer((__gm__ T*)w1);gw2.SetGlobalBuffer((__gm__ T*)w2);
gb1.SetGlobalBuffer((__gm__ T*)b1);gb2.SetGlobalBuffer((__gm__ T*)b2);gy.SetGlobalBuffer((__gm__ T*)y);
LocalTensor<T> hx(TPosition::VECIN,0,16),hw1(TPosition::VECIN,32,16),hw2(TPosition::VECIN,64,16);
LocalTensor<T> hb1(TPosition::VECIN,96,16),hb2(TPosition::VECIN,128,16);
LocalTensor<float> fx(TPosition::VECCALC,192,16),fw1(TPosition::VECCALC,256,16),fw2(TPosition::VECCALC,320,16);
LocalTensor<float> pg(TPosition::VECCALC,384,16),pu(TPosition::VECCALC,448,16);
LocalTensor<float> ag(TPosition::VECCALC,512,8),au(TPosition::VECCALC,544,8);
LocalTensor<T> g(TPosition::VECCALC,576,16),u(TPosition::VECCALC,608,16);
LocalTensor<T> sig(TPosition::VECCALC,640,16),ones(TPosition::VECCALC,672,16),out(TPosition::VECOUT,704,16);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams cp{1,d*2U,0,0,0},bp{1,2,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(16U-d),static_cast<T>(0)};
DataCopyPadExtParams<T> one{true,0,15,static_cast<T>(0)};
DataCopyPad(hx,gx[row*d],cp,pad);DataCopyPad(hw1,gw1[col*d],cp,pad);DataCopyPad(hw2,gw2[col*d],cp,pad);
DataCopyPad(hb1,gb1[col],bp,one);DataCopyPad(hb2,gb2[col],bp,one);
Sync<HardEvent::MTE2_V>();
Cast(fx,hx,RoundMode::CAST_NONE,16);Cast(fw1,hw1,RoundMode::CAST_NONE,16);Cast(fw2,hw2,RoundMode::CAST_NONE,16);
Duplicate(ones,static_cast<T>(1),16);PipeBarrier<PIPE_V>();
Mul(pg,fx,fw1,16);Mul(pu,fx,fw2,16);PipeBarrier<PIPE_V>();
WholeReduceSum(ag,pg,16,1,1,1,2);WholeReduceSum(au,pu,16,1,1,1,2);PipeBarrier<PIPE_V>();
Cast(g,ag,RoundMode::CAST_RINT,1);Cast(u,au,RoundMode::CAST_RINT,1);PipeBarrier<PIPE_V>();
Add(g,g,hb1,1);Add(u,u,hb2,1);PipeBarrier<PIPE_V>();
GateCounter<T>(g,u,sig,out,ones,1);Sync<HardEvent::V_MTE3>();
DataCopyPad(gy[row*d+col],out,bp);PipeBarrier<PIPE_ALL>();
}
private:
template <HardEvent E> __aicore__ inline void Sync() {SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0);}
};
template <typename T>
class SubBlockGemv4 {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y) {
if (GetBlockIdx() != 0U) { return; }
GlobalTensor<T> gx, gw1, gw2, gb1, gb2, gy;
gx.SetGlobalBuffer((__gm__ T*)x, 4);
gw1.SetGlobalBuffer((__gm__ T*)w1, 16);
gw2.SetGlobalBuffer((__gm__ T*)w2, 16);
gb1.SetGlobalBuffer((__gm__ T*)b1, 4);
gb2.SetGlobalBuffer((__gm__ T*)b2, 4);
gy.SetGlobalBuffer((__gm__ T*)y, 4);
LocalTensor<T> hx(TPosition::VECIN, 0, 16);
LocalTensor<T> w(TPosition::VECIN, 32, 32);
LocalTensor<T> bias(TPosition::VECIN, 96, 32);
LocalTensor<float> prod(TPosition::VECCALC, 352, 32);
LocalTensor<float> pairs(TPosition::VECCALC, 480, 16);
LocalTensor<float> acc(TPosition::VECCALC, 544, 40);
LocalTensor<T> g(TPosition::VECCALC, 704, 32);
LocalTensor<T> sig(TPosition::VECCALC, 768, 16);
LocalTensor<T> ones(TPosition::VECCALC, 800, 16);
LocalTensor<T> out(TPosition::VECOUT, 832, 16);
SetAtomicNone();
SetMaskNorm();
const DataCopyExtParams xc{1, 8, 0, 0, 0};
const DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
GlobalTensor<uint64_t> xword;
xword.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(x), 1);
DataCachePreload(xword, static_cast<int16_t>(0));
DataCopy(w, gw1, 16);
DataCopy(w[16], gw2, 16);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
DataCopyPad(bias, gb1, xc, pad);
DataCopyPad(bias[16], gb2, xc, pad);
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
Duplicate(prod, 0.0f, 32);
const uint64_t xv = xword.GetValue(0);
auto xx = hx.template ReinterpretCast<uint64_t>();
xx.SetValue(0, xv);
xx.SetValue(1, xv);
xx.SetValue(2, xv);
xx.SetValue(3, xv);
SetFlag<HardEvent::S_V>(EVENT_ID0);
WaitFlag<HardEvent::S_V>(EVENT_ID0);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
MulAddDst<float, T>(prod, hx, w, static_cast<uint64_t>(16), 2, BinaryRepeatParams{1, 1, 1, 2, 0, 1});
PairReduceSum(pairs, prod, 1, 32, 1, 1, 8);
PairReduceSum(acc, pairs, 2, 8, 1, 1, 1);
constexpr uint64_t mask = 0;
SetVectorMask<float, MaskMode::NORMAL>(0, 15);
Cast<T, float, false>(g, acc, RoundMode::CAST_RINT, mask, 2, UnaryRepeatParams{1, 1, 1, 4});
WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
SetVectorMask<T, MaskMode::NORMAL>(0, 15);
Add<T, false>(g, g, bias, mask, 2, BinaryRepeatParams{1, 1, 1, 1, 1, 1});
FastGate<T, 4U, false>(g, g[16], sig, out, ones);
Sync<HardEvent::V_MTE3>();
DataCopyPad(gy, out, xc);
}
private:
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
};
template <typename T>
class ScalarChannelSwiGlu {
public:
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y) {
        if (GetBlockIdx() != 0U) { return; }
        GlobalTensor<T> gx, gw1, gw2, gb1, gb2, gy;
        gx.SetGlobalBuffer((__gm__ T*)x, 1); gy.SetGlobalBuffer((__gm__ T*)y, 1);
        gw1.SetGlobalBuffer((__gm__ T*)w1, 1); gw2.SetGlobalBuffer((__gm__ T*)w2, 1);
        gb1.SetGlobalBuffer((__gm__ T*)b1, 1); gb2.SetGlobalBuffer((__gm__ T*)b2, 1);

        LocalTensor<T> hx(TPosition::VECIN, 0, 16);
        LocalTensor<T> w(TPosition::VECIN, 32, 32);
        LocalTensor<T> bias(TPosition::VECIN, 96, 32);

        LocalTensor<T> g(TPosition::VECCALC, 160, 32);
        LocalTensor<T> sig(TPosition::VECCALC, 224, 16);
        LocalTensor<T> ones(TPosition::VECCALC, 256, 16);
        LocalTensor<T> out(TPosition::VECOUT, 288, 16);

        SetAtomicNone();
        SetMaskNorm();
        SetVectorMask<T, MaskMode::NORMAL>(0, 1);

        const DataCopyExtParams rc{1, 2, 0, 0, 0};
        const DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        const T xv=gx.GetValue(0);
const T wv1=gw1.GetValue(0),wv2=gw2.GetValue(0);
const T bv1=gb1.GetValue(0),bv2=gb2.GetValue(0);
hx.SetValue(0,xv);w.SetValue(0,wv1);w.SetValue(16,wv2);
bias.SetValue(0,bv1);bias.SetValue(16,bv2);
SetFlag<HardEvent::S_V>(EVENT_ID0);
WaitFlag<HardEvent::S_V>(EVENT_ID0);
constexpr uint64_t mask = 0;
        
        Mul<T, false>(g, hx, w, mask, 2, BinaryRepeatParams{1, 1, 1, 1, 0, 1});
        
        Add<T, false>(g, g, bias, mask, 2, BinaryRepeatParams{1, 1, 1, 1, 1, 1});

        FastGate<T, 1U, false>(g, g[16], sig, out, ones);

        Sync<HardEvent::V_MTE3>();
        DataCopyPad(gy, out, rc);
    }
private:
    template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
};
template <typename T>
class AlignedRowGemv32 {
public:
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y) {
        if (GetBlockIdx() != 0U) { return; }
        GlobalTensor<T> gx, gw1, gw2, gb1, gb2, gy;
        gx.SetGlobalBuffer((__gm__ T*)x, 32);
        gy.SetGlobalBuffer((__gm__ T*)y, 32);
        gw1.SetGlobalBuffer((__gm__ T*)w1, 1024);
        gw2.SetGlobalBuffer((__gm__ T*)w2, 1024);
        gb1.SetGlobalBuffer((__gm__ T*)b1, 32);
        gb2.SetGlobalBuffer((__gm__ T*)b2, 32);

        LocalTensor<T> hw1(TPosition::VECIN, 0, 1024);
        LocalTensor<T> hw2(TPosition::VECIN, 2048, 1024);
        LocalTensor<T> hx(TPosition::VECIN, 4096, 64);
        LocalTensor<T> hb(TPosition::VECIN, 4224, 64);

        LocalTensor<float> pg(TPosition::VECCALC, 4352, 2048);
        LocalTensor<float> pu(TPosition::VECCALC, 12544, 256);
        LocalTensor<float> ag(TPosition::VECCALC, 13568, 64);

        LocalTensor<T> gu(TPosition::VECCALC, 13824, 64);
        LocalTensor<T> sig(TPosition::VECCALC, 13952, 32);
        LocalTensor<T> ones(TPosition::VECCALC, 14016, 32);
        LocalTensor<T> out(TPosition::VECOUT, 14080, 32);

        SetAtomicNone();
        SetMaskNorm();
        constexpr uint64_t mask = 0;
        SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);

        DataCopy(hx, gx, 32);
        DataCopy(hx[32], gx, 32);
        DataCopy(hb, gb1, 32);
        DataCopy(hb[32], gb2, 32);
        DataCopy(hw1, gw1, 1024);
        
        DataCopy(hw2, gw2, 1024);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);

        Duplicate<float, false>(pg, 0.0f, mask, 32U, 1, 8);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
MulAddDst<float, T, false>(pg, hx, hw1, mask, 32U, BinaryRepeatParams{1, 1, 1, 8, 0, 4});
BlockReduceSum<float, false>(pu, pg, 32U, mask, 1, 1, 8);
PairReduceSum<float, false>(pg, pu, 4U, mask, 1, 1, 8);
        PairReduceSum<float, false>(ag, pg, 2U, mask, 1, 1, 8);

        Cast<T, float, false>(gu, ag, RoundMode::CAST_RINT, mask, 1, UnaryRepeatParams{1, 1, 4, 8});
        SetVectorMask<T, MaskMode::NORMAL>(0, ~0ULL);
        Add<T, false>(gu, gu, hb, mask, 1, BinaryRepeatParams{1, 1, 1, 8, 8, 8});

        FastGate<T, 32U, false>(gu, gu[32], sig, out, ones);

        Sync<HardEvent::V_MTE3>();
        DataCopy(gy, out, 32U);
    }
private:
    template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
};
template <typename T, uint32_t FixedD, uint32_t FixedM>
class AlignedColGemv {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
GM_ADDR y) {
const uint64_t task=GetBlockIdx();
if constexpr (FixedM==1U) {
if (task>=2U) { return; }
}
constexpr uint32_t maxK=FixedD;
constexpr uint32_t xCap=64U;
constexpr uint32_t wElems=16U*maxK;
constexpr uint32_t fBase=4U*wElems+2U*xCap+128U;
const uint32_t d=FixedD;
const uint32_t columns=(d+15U)/16U;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x,FixedM*d);gy.SetGlobalBuffer((__gm__ T*)y,FixedM*d);
gw1.SetGlobalBuffer((__gm__ T*)w1,d*d);gw2.SetGlobalBuffer((__gm__ T*)w2,d*d);
gb1.SetGlobalBuffer((__gm__ T*)b1,d);gb2.SetGlobalBuffer((__gm__ T*)b2,d);
LocalTensor<T> hw1(TPosition::VECIN,0,2U*wElems),hw2(TPosition::VECIN,2U*wElems,wElems);
LocalTensor<T> hx(TPosition::VECIN,4U*wElems,xCap);
LocalTensor<T> hb1(TPosition::VECIN,4U*wElems+2U*xCap,32),hb2(TPosition::VECIN,4U*wElems+2U*xCap+32U,16);
LocalTensor<float> pg(TPosition::VECCALC,fBase+8U*wElems,2U*wElems),pu(TPosition::VECCALC,fBase+16U*wElems,wElems);
constexpr uint32_t xBase=fBase+20U*wElems;
constexpr uint32_t aBase=xBase+4U*maxK;
LocalTensor<float> ag(TPosition::VECCALC,aBase,32);
constexpr uint32_t hBase=aBase+256U;
LocalTensor<T> g(TPosition::VECCALC,hBase,32),u(TPosition::VECCALC,hBase+32U,16);
LocalTensor<T> sig(TPosition::VECCALC,hBase+64U,16),ones(TPosition::VECCALC,hBase+96U,16),out(TPosition::VECOUT,hBase+128U,16);
SetAtomicNone();SetMaskNorm();
constexpr uint64_t mask=0;
SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
{
const uint64_t row=FixedM==1U?0U:(FixedD==64U && FixedM==2U ? (task>>2U) : task/columns);
const uint32_t col=(FixedM==1U?task:(FixedD==64U && FixedM==2U ? (task&3U) : task%columns))*16U;
static_assert(FixedD==32U || FixedD==64U);
if constexpr (FixedD==64U) {
DataCopy(hx,gx[row*d],FixedD);
DataCopy(hw1,gw1[static_cast<uint64_t>(col)*d],16U*FixedD);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
DataCopy(hw2,gw2[static_cast<uint64_t>(col)*d],16U*FixedD);
SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
DataCopy(hb1,gb1[col],16U);DataCopy(hb2,gb2[col],16U);
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
Duplicate<float,false>(pg,0.0f,mask,32U,1,8);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
MulAddDst<float,T,false>(pg,hx,hw1,mask,16U,BinaryRepeatParams{1,1,1,8,0,4});
BlockReduceSum<float,false>(pu,pg,16U,mask,1,1,8);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
MulAddDst<float,T,false>(pg[1024],hx,hw2,mask,16U,BinaryRepeatParams{1,1,1,8,0,4});
BlockReduceSum<float,false>(pu[128],pg[1024],16U,mask,1,1,8);
BlockReduceSum<float,false>(ag,pu,4U,mask,1,1,8);
SetVectorMask<float,MaskMode::NORMAL>(0,0xFFFFFFFFULL);
Cast<T,float,false>(g,ag,RoundMode::CAST_RINT,mask,1,UnaryRepeatParams{1,1,4,8});
WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
Add<T,false>(g,g,hb1,mask,1,BinaryRepeatParams{1,1,1,2,2,2});
} else {
DataCopy(hx,gx[row*d],FixedD);
DataCopy(hx[32],gx[row*d],FixedD);
DataCopy(hw1,gw1[static_cast<uint64_t>(col)*d],16U*FixedD);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
DataCopy(hw2,gw2[static_cast<uint64_t>(col)*d],16U*FixedD);
SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
DataCopy(hb1,gb1[col],16U);DataCopy(hb2,gb2[col],16U);
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
Duplicate<float,false>(pg,0.0f,mask,16U,1,8);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
MulAddDst<float,T,false>(pg,hx,hw1,mask,8U,BinaryRepeatParams{1,1,1,8,0,4});
WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
MulAddDst<float,T,false>(pg[512],hx,hw2,mask,8U,BinaryRepeatParams{1,1,1,8,0,4});
BlockReduceSum<float,false>(pu,pg,16U,mask,1,1,8);
PairReduceSum<float,false>(pg,pu,2U,mask,1,1,8);
PairReduceSum<float,false>(ag,pg,1U,mask,1,1,8);
SetVectorMask<float,MaskMode::NORMAL>(0,0xFFFFFFFFULL);
Cast<T,float,false>(g,ag,RoundMode::CAST_RINT,mask,1,UnaryRepeatParams{1,1,4,8});
WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
Add<T,false>(g,g,hb1,mask,1,BinaryRepeatParams{1,1,1,2,2,2});
}
FastGate<T,16U,false>(g,g[16],sig,out,ones);
Sync<HardEvent::V_MTE3>();
if constexpr (FixedM==1U) {
DataCopy(gy[col],out,16U);
} else {
DataCopy(gy[row*d+col],out,16U);
}
}
}
private:
template <HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0); }
};
template <typename T,uint32_t D,uint32_t Rows>
class BatchedRowGemv {
static constexpr uint32_t K=(D+15U)/16U*16U, W=D*K;
static constexpr uint32_t HB=4U*W+2U*Rows*K;
static constexpr uint32_t FB=HB+4U*K;
static constexpr uint32_t XB=FB+8U*W, PB=XB+4U*Rows*K;
static constexpr uint32_t AB=PB+8U*W, GB=AB+8U*Rows*K;
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y) {
const uint64_t row=static_cast<uint64_t>(GetBlockIdx())*Rows;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gw1.SetGlobalBuffer((__gm__ T*)w1);gw2.SetGlobalBuffer((__gm__ T*)w2);
gb1.SetGlobalBuffer((__gm__ T*)b1);gb2.SetGlobalBuffer((__gm__ T*)b2);gy.SetGlobalBuffer((__gm__ T*)y);
LocalTensor<T> hw1(TPosition::VECIN,0,W),hw2(TPosition::VECIN,2U*W,W),hx(TPosition::VECIN,4U*W,Rows*K);
LocalTensor<T> hb1(TPosition::VECIN,HB,K),hb2(TPosition::VECIN,HB+2U*K,K);
LocalTensor<float> fw1(TPosition::VECCALC,FB,W),fw2(TPosition::VECCALC,FB+4U*W,W);
LocalTensor<float> fx(TPosition::VECCALC,XB,Rows*K),pg(TPosition::VECCALC,PB,W),pu(TPosition::VECCALC,PB+4U*W,W);
LocalTensor<float> ag(TPosition::VECCALC,AB,Rows*K),au(TPosition::VECCALC,AB+4U*Rows*K,Rows*K);
LocalTensor<T> g(TPosition::VECCALC,GB,Rows*K),u(TPosition::VECCALC,GB+2U*Rows*K,Rows*K);
LocalTensor<T> sig(TPosition::VECCALC,GB+4U*Rows*K,Rows*K),ones(TPosition::VECCALC,GB+6U*Rows*K,Rows*K);
LocalTensor<T> out(TPosition::VECOUT,GB+8U*Rows*K,Rows*K);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams wc{D,D*2U,0,0,0},bc{1,D*2U,0,0,0},xc{Rows,D*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(K-D),static_cast<T>(0)};
DataCopyPad(hw1,gw1,wc,pad);DataCopyPad(hw2,gw2,wc,pad);
DataCopyPad(hb1,gb1,bc,pad);DataCopyPad(hb2,gb2,bc,pad);DataCopyPad(hx,gx[row*D],xc,pad);
Sync<HardEvent::MTE2_V>();
Cast(fw1,hw1,RoundMode::CAST_NONE,W);Cast(fw2,hw2,RoundMode::CAST_NONE,W);
Cast(fx,hx,RoundMode::CAST_NONE,Rows*K);
Duplicate(ag,0.0f,Rows*K);Duplicate(au,0.0f,Rows*K);Duplicate(ones,static_cast<T>(1),Rows*K);
PipeBarrier<PIPE_V>();
const BinaryRepeatParams rp{1,1,1,K/8U,0,K/8U};
#pragma unroll
for (uint32_t i=0;i<Rows;++i) {
Mul(pg,fx[i*K],fw1,static_cast<uint64_t>(K),static_cast<uint8_t>(D),rp);
Mul(pu,fx[i*K],fw2,static_cast<uint64_t>(K),static_cast<uint8_t>(D),rp);
PipeBarrier<PIPE_V>();
WholeReduceSum(ag[i*K],pg,K,D,1,1,K/8U);WholeReduceSum(au[i*K],pu,K,D,1,1,K/8U);
PipeBarrier<PIPE_V>();
}
Cast(g,ag,RoundMode::CAST_RINT,Rows*K);Cast(u,au,RoundMode::CAST_RINT,Rows*K);PipeBarrier<PIPE_V>();
const BinaryRepeatParams bp{1,1,1,K/16U,K/16U,0};
Add(g,g,hb1,static_cast<uint64_t>(D),static_cast<uint8_t>(Rows),bp);
Add(u,u,hb2,static_cast<uint64_t>(D),static_cast<uint8_t>(Rows),bp);
PipeBarrier<PIPE_V>();GateCounter<T>(g,u,sig,out,ones,Rows*K);
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[row*D],out,xc);PipeBarrier<PIPE_ALL>();
}
private:
template <HardEvent E> __aicore__ inline void Sync() {SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0);}
};
template <typename T>
class PackedMultiBlockGemv {
static constexpr uint32_t D=33U,Rows=5U,K=48U,W=D*32U;
static constexpr uint32_t FB=7424U, PB=20032U;
static constexpr uint32_t BKB=PB+11264U*4U, PRB=BKB+1408U*4U;
static constexpr uint32_t AB=PRB+704U*4U, LB=AB+384U*4U, GB=LB+384U*4U;
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y) {
const uint64_t row=static_cast<uint64_t>(GetBlockIdx())*Rows;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gw1.SetGlobalBuffer((__gm__ T*)w1);gw2.SetGlobalBuffer((__gm__ T*)w2);
gb1.SetGlobalBuffer((__gm__ T*)b1);gb2.SetGlobalBuffer((__gm__ T*)b2);gy.SetGlobalBuffer((__gm__ T*)y);
LocalTensor<T> hw1(TPosition::VECIN,0,1280),hw2(TPosition::VECIN,2560,1280),hxAll(TPosition::VECIN,5120,544);
LocalTensor<T> rawB1(TPosition::VECIN,6272,K),rawB2(TPosition::VECIN,6368,K);
LocalTensor<T> hb1(TPosition::VECIN,6464,384),hb2(TPosition::VECIN,6816,192);
LocalTensor<float> fw1(TPosition::VECCALC,FB,1088),fw1Last(TPosition::VECCALC,FB+4352U,192);
LocalTensor<float> fw2(TPosition::VECCALC,FB+5120U,1088),fw2Last(TPosition::VECCALC,FB+9472U,192);
LocalTensor<float> fxLast(TPosition::VECCALC,FB+10240U,224),fx(TPosition::VECCALC,FB+11136U,Rows*64U);
LocalTensor<float> bothP(TPosition::VECCALC,PB,11264),blkP(TPosition::VECCALC,BKB,1408),pairP(TPosition::VECCALC,PRB,704);
LocalTensor<float> ag(TPosition::VECCALC,AB,384);
LocalTensor<float> lastG(TPosition::VECCALC,LB,384),lastU(TPosition::VECCALC,LB+704U,192);
LocalTensor<T> g(TPosition::VECCALC,GB,384),u(TPosition::VECCALC,GB+352U,256);
LocalTensor<T> sig(TPosition::VECCALC,GB+768U,256),ones(TPosition::VECCALC,GB+1280U,256);
LocalTensor<T> out(TPosition::VECOUT,GB+1792U,256);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
LocalTensor<T> rawW1(TPosition::VECIN,79104,1104),rawW2(TPosition::VECIN,81312,1104),rawX(TPosition::VECIN,83520,176);
LocalTensor<uint32_t> offsets(TPosition::VECIN,83968,2056);
LocalTensor<int32_t> offsetsI(TPosition::VECIN,83968,2056);
LocalTensor<T> allHalf(TPosition::VECIN,0,3136);
LocalTensor<float> allF(TPosition::VECCALC,FB,3136);
DataCopyExtParams wc{1,2178,0,0,0},bc{1,66,0,0,0},xcFlat{1,330,0,0,0};
DataCopyPadExtParams<T> wp{true,0,15,static_cast<T>(0)},xp{true,0,11,static_cast<T>(0)};
DataCopyPad(rawX,gx[row*D],xcFlat,xp);
SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
DataCopyPad(rawW1,gw1,wc,wp);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
DataCopyPad(rawW2,gw2,wc,wp);
SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
DataCopyPad(rawB1,gb1,bc,wp);DataCopyPad(rawB2,gb2,bc,wp);
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
constexpr uint64_t mask=0;
auto offsets64=offsets.template ReinterpretCast<uint64_t>();
#pragma unroll
for (uint32_t k=0;k<16U;++k) {
const uint64_t c4=k*4U;
offsets64.SetValue(k,((c4+2U)<<32)|c4);
}
SetFlag<HardEvent::S_V>(EVENT_ID0);
Duplicate<float,false>(bothP[5280],0.0f,mask,6U,1,8);
Duplicate<float,false>(bothP[10912],0.0f,mask,6U,1,8);
Duplicate<int32_t,false>(offsetsI[1024],0,mask,1U,1,8);
Duplicate<int32_t,false>(offsetsI[1408],0,mask,2U,1,8);
Duplicate<int32_t,false>(offsetsI[1992],0,mask,1U,1,8);
SetVectorMask<int32_t,MaskMode::NORMAL>(0,0x1FFFFFFFFULL);
Duplicate<int32_t,false>(offsetsI[1280],64,mask,1U,1,8);
SetVectorMask<int32_t,MaskMode::NORMAL>(0,0x3FFFFFFFEULL);
Duplicate<int32_t,false>(offsetsI[1312],130,mask,1U,1,8);
SetVectorMask<int32_t,MaskMode::NORMAL>(0,0x7FFFFFFFCULL);
Duplicate<int32_t,false>(offsetsI[1344],196,mask,1U,1,8);
SetVectorMask<int32_t,MaskMode::NORMAL>(0,0xFFFFFFFF8ULL);
Duplicate<int32_t,false>(offsetsI[1376],262,mask,1U,1,8);
SetVectorMask<int32_t,MaskMode::NORMAL>(0,0x1FFFFFFFF0ULL);
Duplicate<int32_t,false>(offsetsI[1408],328,mask,1U,1,8);
SetVectorMask<int32_t,MaskMode::NORMAL>(0,0xFFFFFFFFULL);
WaitFlag<HardEvent::S_V>(EVENT_ID0);
const UnaryRepeatParams urI{1,1,8,8};
Adds<int32_t,false>(offsetsI[32],offsetsI[0],66,mask,1U,urI);
Adds<int32_t,false>(offsetsI[1024],offsetsI[0],2112,mask,1U,urI);
Adds<int32_t,false>(offsetsI[1504],offsetsI[0],0,mask,1U,urI);
Adds<int32_t,false>(offsetsI[1536],offsetsI[0],0,mask,1U,urI);
Adds<int32_t,false>(offsetsI[1824],offsetsI[0],0,mask,1U,urI);
SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
Adds<int32_t,false>(offsetsI[64],offsetsI[0],132,mask,1U,urI);
Adds<int32_t,false>(offsetsI[128],offsetsI[0],264,mask,2U,urI);
Adds<int32_t,false>(offsetsI[256],offsetsI[0],528,mask,4U,urI);
Adds<int32_t,false>(offsetsI[512],offsetsI[0],1056,mask,8U,urI);
Adds<int32_t,false>(offsetsI[1568],offsetsI[1504],66,mask,1U,urI);
Adds<int32_t,false>(offsetsI[1632],offsetsI[1504],132,mask,2U,urI);
Adds<int32_t,false>(offsetsI[1760],offsetsI[1504],264,mask,1U,urI);
#pragma unroll
for (uint32_t k=0;k<16U;++k) {
const uint64_t c4=k*4U;
const uint64_t valA=((c4+2U)<<32)|c4;
const uint64_t valB=(c4<<32)|(k==0U?64ULL:c4-2U);
offsets64.SetValue(945U+k,valA);
offsets64.SetValue(978U+k,valA);
offsets64.SetValue(928U+k,valB);
offsets64.SetValue(961U+k,valB);
}
offsets64.SetValue(944U,(64ULL<<32)|62ULL);
offsets64.SetValue(977U,(64ULL<<32)|62ULL);
offsets64.SetValue(994U,64ULL);
offsets64.SetValue(995U,0ULL);
SetFlag<HardEvent::S_V>(EVENT_ID1);
WaitFlag<HardEvent::S_V>(EVENT_ID1);
Muls<int32_t,false>(offsetsI[1088],offsetsI[1824],33,mask,3U,urI);
Adds<int32_t,false>(offsetsI[1088],offsetsI[1088],64,mask,2U,urI);
SetVectorMask<int32_t,MaskMode::NORMAL>(0,(1ULL<<37)-1ULL);
Adds<int32_t,false>(offsetsI[1216],offsetsI[1216],64,mask,1U,urI);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
Gather(hxAll,rawX,offsets[1280],0,544);
SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
Cast<float,T,false>(allF[2560],allHalf[2560],RoundMode::CAST_NONE,mask,9U,UnaryRepeatParams{1,1,8,4});
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
Gather(hw1,rawW1,offsets,0,1280);
Cast<float,T,false>(allF,allHalf,RoundMode::CAST_NONE,mask,20U,UnaryRepeatParams{1,1,8,4});
const BinaryRepeatParams rp{1,1,1,8,0,8},br8{1,1,1,8,8,8};
#pragma unroll
for (uint32_t i=0;i<Rows;++i) {
Mul<float,false>(bothP[i*W],fx[i*64U],fw1,mask,17U,rp);
}
Mul<float,false>(lastG,fxLast,fw1Last,mask,3U,br8);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
Gather(hw2,rawW2,offsets,0,1280);
Cast<float,T,false>(allF[1280],allHalf[1280],RoundMode::CAST_NONE,mask,20U,UnaryRepeatParams{1,1,8,4});
WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
Gather(hb1,rawB1,offsets[1824],0,176);
Gather(hb2,rawB2,offsets[1824],0,176);
SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
#pragma unroll
for (uint32_t i=0;i<Rows;++i) {
Mul<float,false>(bothP[5632U+i*W],fx[i*64U],fw2,mask,17U,rp);
}
BlockReduceSum<float,false>(blkP,bothP,176U,mask,1,1,8);
PairReduceSum<float,false>(pairP,blkP,22U,mask,1,1,8);
PairReduceSum<float,false>(ag,pairP,11U,mask,1,1,8);
Mul<float,false>(lastU,fxLast,fw2Last,mask,3U,br8);
Add<float,false>(ag,ag,lastG,mask,6U,br8);
Cast<T,float,false>(g,ag,RoundMode::CAST_RINT,mask,6U,UnaryRepeatParams{1,1,4,8});
Add<T,false>(g,g,hb1,mask,3U,br8);
const UnaryRepeatParams ur8{1,1,8,8};
Muls<T,false>(sig,g,static_cast<T>(-1),mask,2U,ur8);
Exp<T,false>(sig,sig,mask,2U,ur8);
Adds<T,false>(sig,sig,static_cast<T>(1),mask,2U,ur8);
Div<T,false>(sig,g,sig,mask,2U,br8);
Mul<T,false>(out,sig,u,mask,2U,br8);
Sync<HardEvent::V_MTE3>();DataCopyPad(gy[row*D],out,xcFlat);
}
private:
template <HardEvent E> __aicore__ inline void Sync() {SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0);}
};
template <typename T>
class PackedSubBlockGemv {
static constexpr uint32_t D=13U,Rows=3U,K=16U,W=256U;
static constexpr uint32_t HB=4U*W+2U*Rows*K;
static constexpr uint32_t FB=HB+4U*K;
static constexpr uint32_t XB=FB+8U*W, PB=XB+4U*Rows*K;
static constexpr uint32_t AB=PB+8U*W, GB=AB+8U*Rows*K;
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y) {
const uint32_t blk=GetBlockIdx();
if (blk>=7U) { return; }
const uint64_t row=static_cast<uint64_t>(blk)*Rows;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy;
gx.SetGlobalBuffer((__gm__ T*)x);gw1.SetGlobalBuffer((__gm__ T*)w1);gw2.SetGlobalBuffer((__gm__ T*)w2);
gb1.SetGlobalBuffer((__gm__ T*)b1);gb2.SetGlobalBuffer((__gm__ T*)b2);gy.SetGlobalBuffer((__gm__ T*)y);
LocalTensor<T> hw1(TPosition::VECIN,0,W),hw2(TPosition::VECIN,2U*W,W);
LocalTensor<T> hb1(TPosition::VECIN,HB,K),hb2(TPosition::VECIN,HB+2U*K,K);
LocalTensor<T> sig(TPosition::VECCALC,GB+4U*Rows*K,Rows*K);
LocalTensor<T> out(TPosition::VECOUT,GB+8U*Rows*K,Rows*K);
LocalTensor<T> packedOut(TPosition::VECOUT,GB+10U*Rows*K,Rows*K);
SetAtomicNone();SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
LocalTensor<T> rawW1(TPosition::VECIN,8448,176),rawW2(TPosition::VECIN,8960,176),rawX(TPosition::VECIN,9472,48);
LocalTensor<uint32_t> offsets(TPosition::VECIN,13056,512);
LocalTensor<int32_t> offsetsI(TPosition::VECIN,13056,512);
DataCopyExtParams wc{1,338,0,0,0},bc{1,26,0,0,0},xcFlat{1,78,0,0,0};
DataCopyPadExtParams<T> wp{true,0,7,static_cast<T>(0)},bp{true,0,3,static_cast<T>(0)},xp{true,0,9,static_cast<T>(0)};
DataCopyPad(rawX,gx[row*D],xcFlat,xp);
SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
DataCopyPad(rawW1,gw1,wc,wp);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
DataCopyPad(rawW2,gw2,wc,wp);
SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
DataCopyPad(hb1,gb1,bc,bp);DataCopyPad(hb2,gb2,bc,bp);
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
constexpr uint64_t mask=0;
LocalTensor<float> bothP(TPosition::VECCALC,19904,1536);
LocalTensor<float> bothBlock(TPosition::VECCALC,34048,192);
LocalTensor<float> acc(TPosition::VECCALC,AB,96);
LocalTensor<T> bothBias(TPosition::VECIN,HB,32),gu(TPosition::VECCALC,GB,96);
LocalTensor<T> bothHalfW(TPosition::VECIN,0,512),halfExpandedX(TPosition::VECCALC,10752,192);
auto offsets64=offsets.template ReinterpretCast<uint64_t>();
#pragma unroll
for (uint32_t k=0;k<6U;++k) {
const uint64_t c4=k*4U;
const uint64_t valA=((c4+2U)<<32)|c4;
offsets64.SetValue(k,valA);



}
offsets64.SetValue(6U,(338ULL<<32)|24ULL);
offsets64.SetValue(7U,(338ULL<<32)|338ULL);
offsets64.SetValue(224U,0x1FFFULL);
offsets64.SetValue(225U,0ULL);


SetFlag<HardEvent::S_V>(EVENT_ID0);
Duplicate<float,false>(bothP,0.0f,mask,24U,1,8);
Duplicate<int32_t,false>(offsetsI[16],338,mask,3U,1,8);
Duplicate<int32_t,false>(offsetsI[256],78,mask,3U,1,8);
SetVectorMask<int32_t,MaskMode::NORMAL>(0,0xFFFFFFFFFFFFULL);
Duplicate<int32_t,false>(offsetsI[208],338,mask,1U,1,8);
WaitFlag<HardEvent::S_V>(EVENT_ID0);
const UnaryRepeatParams urI{1,1,8,8};
SetVectorMask<int32_t,MaskMode::NORMAL>(0,0x1FFFULL);
Adds<int32_t,false>(offsetsI[16],offsetsI[0],26,mask,1U,urI);
Adds<int32_t,false>(offsetsI[192],offsetsI[0],312,mask,1U,urI);
Adds<int32_t,false>(offsetsI[256],offsetsI[0],0,mask,1U,urI);
Adds<int32_t,false>(offsetsI[272],offsetsI[0],0,mask,1U,urI);
SetVectorMask<int32_t,MaskMode::NORMAL>(0,0x1FFF1FFFULL);
Adds<int32_t,false>(offsetsI[32],offsetsI[0],52,mask,1U,urI);
Adds<int32_t,false>(offsetsI[288],offsetsI[256],0,mask,1U,urI);
SetVectorMask<int32_t,MaskMode::NORMAL>(0,0x1FFF1FFF1FFF1FFFULL);
Adds<int32_t,false>(offsetsI[64],offsetsI[0],104,mask,1U,urI);
Adds<int32_t,false>(offsetsI[128],offsetsI[0],208,mask,1U,urI);
Adds<int32_t,false>(offsetsI[320],offsetsI[256],26,mask,1U,urI);
Adds<int32_t,false>(offsetsI[384],offsetsI[256],52,mask,1U,urI);
SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
Gather(halfExpandedX,rawX,offsets[256],0,192);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
Gather(hw1,rawW1,offsets,0,W);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
Gather(hw2,rawW2,offsets,0,W);
SetVectorMask<float,MaskMode::NORMAL>(0,~0ULL);
const BinaryRepeatParams rp{1,1,1,8,0,4};
#pragma unroll
for (uint32_t i=0;i<Rows;++i) {
MulAddDst<float,T,false>(bothP[i*512U],halfExpandedX[i*64U],bothHalfW,mask,8,rp);
}
BlockReduceSum<float,false>(bothBlock,bothP,24,mask,1,1,8);
PairReduceSum<float,false>(acc,bothBlock,3,mask,1,1,8);
Cast<T,float,false>(gu,acc,RoundMode::CAST_RINT,mask,2U,UnaryRepeatParams{1,1,4,8});
WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
SetVectorMask<T,MaskMode::NORMAL>(0,0xFFFFFFFFULL);
Add<T,false>(gu,gu,bothBias,mask,3,BinaryRepeatParams{1,1,1,2,2,0});
SetVectorMask<T,MaskMode::NORMAL>(0,0xFFFFULL);
const UnaryRepeatParams compactG{1,1,1,2},contiguous{1,1,1,1};
const BinaryRepeatParams gateProduct{1,1,1,1,2,1},upProduct{1,1,1,1,1,2};
Muls<T,false>(sig,gu,static_cast<T>(-1),mask,3,compactG);
Exp<T,false>(sig,sig,mask,3,contiguous);
Adds<T,false>(sig,sig,static_cast<T>(1),mask,3,contiguous);
Div<T,false>(sig,gu,sig,mask,3,gateProduct);
Mul<T,false>(out,sig,gu[16],mask,3,upProduct);
// Pack the13 live half elements of each16-element output row.
// Counter-mode repeat gathering removes the output index table entirely.
SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
vreducev2((__ubuf__ uint16_t*)packedOut.GetPhyAddr(),
          (__ubuf__ uint16_t*)out.GetPhyAddr(),
          (__ubuf__ uint16_t*)offsets[448].GetPhyAddr(),Rows,1,0,1,0);
Sync<HardEvent::V_MTE3>();
DataCopyPad(gy[row*D],packedOut,xcFlat);
}
private:
template <HardEvent E> __aicore__ inline void Sync() {SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0);}
};
namespace optiling {
using namespace AscendC;
using namespace matmul;
constexpr uint32_t kSlotCount = 4;
constexpr uint32_t kBlockRows = 16;
__aicore__ constexpr MatmulConfig DirectBasicConfig(uint32_t rows)
{
auto config = GetBasicConfig(rows, 128, 128);
config.enUnitFlag = true;
return config;
}
constexpr MatmulConfig kBasic16 = DirectBasicConfig(16);
constexpr MatmulConfig kBasic32 = DirectBasicConfig(32);
constexpr MatmulConfig kBasic128 = DirectBasicConfig(128);
template <typename T, uint32_t FixedM = 0, uint32_t FixedD = 0, bool Unaligned = false, bool Resident = false, bool FullTile = false, uint32_t FixedN = 0>
class DirectPairCube {
public:
__aicore__ inline void Init(uint32_t k, uint32_t m, uint32_t n) {
realK_ = FixedD != 0 ? FixedD : k;
k_ = Unaligned ? (realK_ + 15U) / 16U * 16U : realK_;
m_ = FixedM != 0 ? FixedM : m;
n_ = FixedN != 0 ? FixedN : (FixedD != 0 ? (FixedD < 64U ? FixedD : 64U) : n);
}
__aicore__ inline void PrepareWeights(const GlobalTensor<T> &w1,const GlobalTensor<T> &w2,uint32_t cols) {
LocalTensor<T> b1(TPosition::B1,m_*k_*sizeof(T),2U*n_*k_);
LocalTensor<T> b2(TPosition::B2,0,2U*n_*k_);
Nd2NzParams bp{1,static_cast<uint16_t>(cols),static_cast<uint16_t>(realK_),0,
static_cast<uint16_t>(realK_),static_cast<uint16_t>(2U*n_),1,0};
DataCopy(b1,w1,bp);DataCopy(b1[n_*16U],w2,bp);
Sync<HardEvent::MTE2_MTE1>();
LoadData2dParams bl;
bl.repeatTimes=static_cast<uint8_t>(2U*n_*k_/256U);bl.srcStride=1;
LoadData(b2,b1,bl);
Sync<HardEvent::MTE1_M>();
}
__aicore__ inline void Compute(const GlobalTensor<T> &x, const GlobalTensor<T> &w1,
const GlobalTensor<T> &w2, const GlobalTensor<T> &dst,
uint32_t rows, uint32_t cols) {
LocalTensor<T> a1(TPosition::A1, 0, m_ * k_);
LocalTensor<T> b1(TPosition::B1, m_ * k_ * sizeof(T), 2U * n_ * k_);
LocalTensor<T> a2(TPosition::A2, 0, m_ * k_);
LocalTensor<T> b2(TPosition::B2, 0, 2U * n_ * k_);
LocalTensor<float> c(TPosition::CO1, 0, 2U * m_ * n_);
const uint32_t alignedM = FullTile ? FixedM : (rows + 15U) / 16U * 16U;
Nd2NzParams ap{1, static_cast<uint16_t>(rows), static_cast<uint16_t>(realK_), 0,
static_cast<uint16_t>(realK_), static_cast<uint16_t>(alignedM), 1, 0};
Nd2NzParams bp{1, static_cast<uint16_t>(cols), static_cast<uint16_t>(realK_), 0,
static_cast<uint16_t>(realK_), static_cast<uint16_t>(2U * n_), 1, 0};
DataCopy(a1, x, ap);
if constexpr (!Resident) {
DataCopy(b1, w1, bp);
DataCopy(b1[n_ * 16U], w2, bp);
}
Sync<HardEvent::MTE2_MTE1>();
LoadData2dParams al;
al.repeatTimes = static_cast<uint8_t>(k_ / 16U);
al.srcStride = static_cast<uint16_t>(alignedM / 16U);
if constexpr (FullTile) {
#pragma unroll
for (uint32_t mb=0;mb<FixedM/16U;++mb) {
LoadData(a2[mb*k_*16U],a1[mb*256U],al);
}
} else {
for (uint32_t mb = 0; mb < alignedM / 16U; ++mb) {
LoadData(a2[mb * k_ * 16U], a1[mb * 256U], al);
}
}
LoadData2dParams bl;
bl.repeatTimes = static_cast<uint8_t>(2U * n_ * k_ / 256U);
bl.srcStride = 1;
if constexpr (!Resident) { LoadData(b2, b1, bl); }
if constexpr (Resident) { Sync<HardEvent::MTE1_MTE2>(); }
Sync<HardEvent::MTE1_M>();
MmadParams mad;
mad.m = static_cast<uint16_t>(alignedM);
mad.n = static_cast<uint16_t>(2U * n_);
mad.k = static_cast<uint16_t>(k_);
mad.cmatrixInitVal = true;
Mmad(c, a2, b2, mad);
if constexpr (Resident) { Sync<HardEvent::M_MTE1>(); }
Sync<HardEvent::M_FIX>();
FixpipeParamsV220 fix;
fix.nSize = static_cast<uint16_t>(2U * n_);
fix.mSize = static_cast<uint16_t>(rows);
fix.srcStride = static_cast<uint16_t>(alignedM);
fix.dstStride = 2U * n_;
fix.quantPre = QuantMode_t::F322F16;
Fixpipe<T, float, CFG_ROW_MAJOR>(dst, c, fix);
if constexpr (Resident) { Sync<HardEvent::FIX_M>(); }
else { PipeBarrier<PIPE_ALL>(); }
}
private:
template <HardEvent E>
__aicore__ inline void Sync() {
constexpr event_t event = EVENT_ID0;
SetFlag<E>(event);
WaitFlag<E>(event);
}
uint32_t k_ = 0, realK_ = 0, m_ = 0, n_ = 0;
};
struct EmptyMatmulStorage {};
template <bool Direct, typename A, typename B, typename C, const auto& Config>
struct MatmulStorageSelector {
using Type = MatmulImpl<A, B, C, C, Config>;
};
template <typename A, typename B, typename C, const auto& Config>
struct MatmulStorageSelector<true, A, B, C, Config> {
using Type = EmptyMatmulStorage;
};
template <typename xType, const auto& Config = CFG_NORM, bool Direct = false,
bool Padded = false, bool Merged = false>
class SwiGluCubeImpl {
public:
using AType = MatmulType<TPosition::GM, CubeFormat::ND, xType, false>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, xType, true>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, xType>;
typename MatmulStorageSelector<Direct, AType, BType, CType, Config>::Type mm;
DirectPairCube<xType> direct;
__aicore__ inline void Init(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
GM_ADDR userWorkspace, TPipe *pipe, const SwiGluTilingData *td)
{
pipe_ = pipe;
core_ = GetBlockIdx();
sub_ = GetSubBlockIdx();
if ASCEND_IS_AIV { core_ /= 2U; }
m_ = td->m;
d_ = static_cast<uint32_t>(td->d);
computeD_ = td->dAlign;
taskM_ = td->taskM;
taskN_ = td->taskN;
taskNum_ = td->taskNum;
cores_ = td->usedCoreNum;
tasksN_ = (d_ + taskN_ - 1U) / taskN_;
plane_ = taskM_ * taskN_;
x_.SetGlobalBuffer((__gm__ xType *)x, m_ * d_);
w1_.SetGlobalBuffer((__gm__ xType *)w1, static_cast<uint64_t>(d_) * d_);
w2_.SetGlobalBuffer((__gm__ xType *)w2, static_cast<uint64_t>(d_) * d_);
b1_.SetGlobalBuffer((__gm__ xType *)b1, d_);
b2_.SetGlobalBuffer((__gm__ xType *)b2, d_);
y_.SetGlobalBuffer((__gm__ xType *)y, m_ * d_);
slots_.SetGlobalBuffer((__gm__ xType *)userWorkspace,
static_cast<uint64_t>(plane_) * 2 * kSlotCount * cores_);
if constexpr (padded_) {
const uint64_t base = static_cast<uint64_t>(plane_) * 2 * kSlotCount * cores_;
const uint64_t wBase = base + m_ * computeD_;
paddedX_.SetGlobalBuffer((__gm__ xType *)userWorkspace + base, m_ * computeD_);
paddedW1_.SetGlobalBuffer((__gm__ xType *)userWorkspace + wBase,
static_cast<uint64_t>(computeD_) * computeD_ * (merged_ ? 2U : 1U));
paddedW2_.SetGlobalBuffer((__gm__ xType *)userWorkspace + wBase +
static_cast<uint64_t>(computeD_) * computeD_,
static_cast<uint64_t>(computeD_) * computeD_);
packRows_ = 4096U / computeD_;
}
if ASCEND_IS_AIC {
if constexpr (Direct) {
direct.Init(computeD_, taskM_, taskN_);
} else {
mm.Init(&td->cubeTilingData, pipe_);
mm.SetHF32(false, 0);
const uint32_t matmulN = merged_ ? 2U * taskN_ : taskN_;
mm.SetSingleShape(taskM_, matmulN, computeD_);
mm.SetOrgShape(taskM_, matmulN, computeD_, computeD_);
}
}
if ASCEND_IS_AIV {
const uint32_t cBytes = (taskM_ / 2U) * taskN_ * sizeof(xType);
const uint32_t vBytes = kBlockRows * taskN_ * sizeof(xType);
pipe_->InitBuffer(gateQ_, 1, cBytes);
pipe_->InitBuffer(upQ_, 1, cBytes);
pipe_->InitBuffer(bias1Q_, 1, taskN_ * sizeof(xType));
pipe_->InitBuffer(bias2Q_, 1, taskN_ * sizeof(xType));
pipe_->InitBuffer(outQ_, 1, vBytes);
pipe_->InitBuffer(gateBuf_, vBytes);
pipe_->InitBuffer(upBuf_, vBytes);
pipe_->InitBuffer(negBuf_, vBytes);
pipe_->InitBuffer(expBuf_, vBytes);
pipe_->InitBuffer(onesBuf_, vBytes);
pipe_->InitBuffer(sigmoidBuf_, vBytes);
if constexpr (padded_) { pipe_->InitBuffer(packQ_, 1, packRows_ * computeD_ * sizeof(xType)); }
}
}
__aicore__ inline void Process()
{
if (core_ >= cores_) { return; }
if constexpr (padded_) {
if ASCEND_IS_AIV {
PackMatrix(paddedX_, x_, m_, m_);
PackMatrix(paddedW1_, w1_, d_, computeD_);
PackMatrix(paddedW2_, w2_, d_, computeD_);
SyncAll();
CrossCoreSetFlag<0x2, PIPE_MTE3>(0x4);
}
if ASCEND_IS_AIC { CrossCoreWaitFlag(0x4); }
}
uint32_t local = 0;
for (uint32_t task = core_; task < taskNum_; task += cores_, ++local) {
const uint32_t slot = local % kSlotCount;
const uint64_t row = static_cast<uint64_t>(task / tasksN_) * taskM_;
const uint32_t col = (task % tasksN_) * taskN_;
const uint32_t rows = Min(taskM_, static_cast<uint32_t>(m_ - row));
const uint32_t cols = Min(taskN_, d_ - col);
const uint64_t off = (static_cast<uint64_t>(core_) * kSlotCount + slot) * 2 * plane_;
if ASCEND_IS_AIC {
if (local >= kSlotCount) { WaitConsumed(slot); }
if constexpr (Direct) {
direct.Compute(x_[row * d_], w1_[static_cast<uint64_t>(col) * d_],
w2_[static_cast<uint64_t>(col) * d_], slots_[off], rows, cols);
} else {
if constexpr (padded_) {
const uint32_t computeCols = Min(taskN_, computeD_ - col);
if constexpr (merged_) {
Projection(paddedW1_, row, 0, rows, 2U * computeCols, off);
} else {
Projection(paddedW1_, row, col, rows, computeCols, off);
Projection(paddedW2_, row, col, rows, computeCols, off + plane_);
}
} else {
Projection(w1_, row, col, rows, cols, off);
Projection(w2_, row, col, rows, cols, off + plane_);
}
}
SignalReady(slot);
}
if ASCEND_IS_AIV {
WaitReady(slot);
Epilogue(off, row, col, rows, cols);
SignalConsumed(slot);
}
}
if ASCEND_IS_AIC {
for (uint32_t slot = 0; slot < Min(local, kSlotCount); ++slot) { WaitConsumed(slot); }
}
}
private:
__aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }
__aicore__ inline void PackMatrix(const GlobalTensor<xType> &dst,
const GlobalTensor<xType> &src,
uint64_t realRows, uint64_t totalRows)
{
const uint64_t chunks = (totalRows + packRows_ - 1U) / packRows_;
for (uint64_t block = GetBlockIdx(); block < chunks; block += cores_ * 2U) {
const uint64_t row = block * packRows_;
const uint32_t nr = Min(packRows_, static_cast<uint32_t>(totalRows - row));
const uint32_t valid = row < realRows ? Min(nr, static_cast<uint32_t>(realRows - row)) : 0;
LocalTensor<xType> tile = packQ_.AllocTensor<xType>();
if (valid != 0) {
DataCopyExtParams copy{static_cast<uint16_t>(valid), d_ * 2U, 0, 0, 0};
DataCopyPadExtParams<xType> pad{true, 0, static_cast<uint8_t>(computeD_ - d_), 0};
DataCopyPad(tile, src[row * d_], copy, pad);
}
if (valid < nr) {
Duplicate(tile[valid * computeD_], static_cast<xType>(0), (nr - valid) * computeD_);
const event_t event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
SetFlag<HardEvent::V_MTE3>(event);
WaitFlag<HardEvent::V_MTE3>(event);
}
packQ_.EnQue(tile);
tile = packQ_.DeQue<xType>();
DataCopy(dst[row * computeD_], tile, nr * computeD_);
packQ_.FreeTensor(tile);
}
}
__aicore__ inline void Projection(const GlobalTensor<xType> &weight, uint64_t row, uint32_t col,
uint32_t rows, uint32_t cols, uint64_t off)
{
mm.SetTail(rows, cols, computeD_);
if constexpr (padded_) { mm.SetTensorA(paddedX_[row * computeD_], false); }
else { mm.SetTensorA(x_[row * d_], false); }
mm.SetTensorB(weight[static_cast<uint64_t>(col) * computeD_], true);
mm.IterateAll(slots_[off], false);
mm.End();
PipeBarrier<PIPE_ALL>();
}
__aicore__ inline void Epilogue(uint64_t off, uint64_t row, uint32_t col,
uint32_t rows, uint32_t cols)
{
const uint32_t halfRows = (rows + 1U) / 2U;
const uint32_t begin = sub_ * halfRows;
if (begin >= rows) { return; }
const uint32_t count = Min(halfRows, rows - begin);
const uint32_t pitch = AlignUp16(cols);
const uint32_t cPitch = merged_ ? 2U * taskN_ : taskN_;
LocalTensor<xType> gate = gateQ_.AllocTensor<xType>();
LocalTensor<xType> up = upQ_.AllocTensor<xType>();
DataCopyExtParams cParams{static_cast<uint16_t>(count), cols * static_cast<uint32_t>(sizeof(xType)),
(cPitch - cols) * static_cast<uint32_t>(sizeof(xType)), 0, 0};
DataCopyPadExtParams<xType> pad{false, 0, 0, 0};
const uint64_t cOff = off + static_cast<uint64_t>(begin) * cPitch;
DataCopyPad(gate, slots_[cOff], cParams, pad);
DataCopyPad(up, slots_[cOff + (merged_ ? taskN_ : plane_)], cParams, pad);
gateQ_.EnQue(gate);
upQ_.EnQue(up);
LocalTensor<xType> b1 = bias1Q_.AllocTensor<xType>();
LocalTensor<xType> b2 = bias2Q_.AllocTensor<xType>();
DataCopyExtParams biasParams{1, cols * static_cast<uint32_t>(sizeof(xType)), 0, 0, 0};
DataCopyPad(b1, b1_[col], biasParams, pad);
DataCopyPad(b2, b2_[col], biasParams, pad);
bias1Q_.EnQue(b1);
bias2Q_.EnQue(b2);
gate = gateQ_.DeQue<xType>();
up = upQ_.DeQue<xType>();
b1 = bias1Q_.DeQue<xType>();
b2 = bias2Q_.DeQue<xType>();
LocalTensor<xType> a = gateBuf_.Get<xType>();
LocalTensor<xType> b = upBuf_.Get<xType>();
LocalTensor<xType> neg = negBuf_.Get<xType>();
LocalTensor<xType> exp = expBuf_.Get<xType>();
LocalTensor<xType> ones = onesBuf_.Get<xType>();
LocalTensor<xType> sig = sigmoidBuf_.Get<xType>();
for (uint32_t group = 0; group < count; group += kBlockRows) {
const uint32_t nr = Min(kBlockRows, count - group);
const uint32_t n = nr * pitch;
const uint8_t rowStride = static_cast<uint8_t>(pitch / 16U);
const BinaryRepeatParams biasRepeat{1, 1, 1, rowStride, rowStride, 0};
for (uint32_t k = 0; k < cols; k += 128U) {
const uint32_t width = Min(128U, cols - k);
Add(a[k], gate[group * pitch + k], b1[k], static_cast<uint64_t>(width),
static_cast<uint8_t>(nr), biasRepeat);
Add(b[k], up[group * pitch + k], b2[k], static_cast<uint64_t>(width),
static_cast<uint8_t>(nr), biasRepeat);
}
LocalTensor<xType> out = outQ_.AllocTensor<xType>();
GateCounter<xType,true>(a,b,exp,out,ones,n);
outQ_.EnQue(out);
out = outQ_.DeQue<xType>();
DataCopyExtParams outParams{static_cast<uint16_t>(nr), cols * static_cast<uint32_t>(sizeof(xType)),
0, (d_ - cols) * static_cast<uint32_t>(sizeof(xType)), 0};
DataCopyPad(y_[(row + begin + group) * d_ + col], out, outParams);
outQ_.FreeTensor(out);
}
gateQ_.FreeTensor(gate);
upQ_.FreeTensor(up);
bias1Q_.FreeTensor(b1);
bias2Q_.FreeTensor(b2);
}
__aicore__ inline void WaitReady(uint32_t s) {
if (s == 0) { CrossCoreWaitFlag(0x6); }
else if (s == 1) { CrossCoreWaitFlag(0x7); }
else if (s == 2) { CrossCoreWaitFlag(0xB); }
else { CrossCoreWaitFlag(0xE); }
}
__aicore__ inline void SignalReady(uint32_t s) {
if (s == 0) { CrossCoreSetFlag<0x2, PIPE_FIX>(0x6); }
else if (s == 1) { CrossCoreSetFlag<0x2, PIPE_FIX>(0x7); }
else if (s == 2) { CrossCoreSetFlag<0x2, PIPE_FIX>(0xB); }
else { CrossCoreSetFlag<0x2, PIPE_FIX>(0xE); }
}
__aicore__ inline void WaitConsumed(uint32_t s) {
if (s == 0) { CrossCoreWaitFlag(0x8); }
else if (s == 1) { CrossCoreWaitFlag(0x9); }
else if (s == 2) { CrossCoreWaitFlag(0xC); }
else { CrossCoreWaitFlag(0xF); }
}
__aicore__ inline void SignalConsumed(uint32_t s) {
if (s == 0) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0x8); }
else if (s == 1) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0x9); }
else if (s == 2) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0xC); }
else { CrossCoreSetFlag<0x2, PIPE_MTE3>(0xF); }
}
TPipe *pipe_ = nullptr;
TQue<QuePosition::VECIN, 1> gateQ_, upQ_, bias1Q_, bias2Q_;
TQue<QuePosition::VECOUT, 1> outQ_;
TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> packQ_;
TBuf<TPosition::VECCALC> gateBuf_, upBuf_, negBuf_, expBuf_, onesBuf_, sigmoidBuf_;
GlobalTensor<xType> x_, w1_, w2_, b1_, b2_, y_, slots_;
GlobalTensor<xType> paddedX_, paddedW1_, paddedW2_;
uint64_t m_ = 0;
uint32_t d_ = 0, taskM_ = 0, taskN_ = 0, taskNum_ = 0, tasksN_ = 0;
uint32_t core_ = 0, sub_ = 0, cores_ = 0, plane_ = 0;
uint32_t computeD_ = 0, packRows_ = 0;
static constexpr bool padded_ = Padded;
static constexpr bool merged_ = Merged || Direct;
};
template <typename T, bool Padded = false, bool Merged = false>
__aicore__ inline void RunCube(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
GM_ADDR workspace, TPipe *pipe, const SwiGluTilingData *td)
{
SwiGluCubeImpl<T, CFG_NORM, false, Padded, Merged> op;
op.Init(x, w1, b1, w2, b2, y, workspace, pipe, td);
op.Process();
}
template <typename T, uint32_t TileM, uint32_t FixedD = 0, bool Unaligned = false, bool Resident = false, bool FullTile = false>
class SlimDirectCube {
public:
__aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
GM_ADDR workspace, const SwiGluDirectTilingData *td) {
d_ = FixedD != 0 ? FixedD : static_cast<uint32_t>(td->d);
n_ = FixedD != 0 ? (FixedD < 64U ? FixedD : 64U) : td->taskN;
const uint32_t cores = td->usedCoreNum;
uint32_t core = GetBlockIdx();
if ASCEND_IS_AIV { core /= 2U; }
x_.SetGlobalBuffer((__gm__ T *)x, td->m * d_);
w1_.SetGlobalBuffer((__gm__ T *)w1, static_cast<uint64_t>(d_) * d_);
w2_.SetGlobalBuffer((__gm__ T *)w2, static_cast<uint64_t>(d_) * d_);
b1_.SetGlobalBuffer((__gm__ T *)b1, d_);
b2_.SetGlobalBuffer((__gm__ T *)b2, d_);
y_.SetGlobalBuffer((__gm__ T *)y, td->m * d_);
const uint32_t plane = tm_ * n_;
slots_.SetGlobalBuffer((__gm__ T *)workspace, static_cast<uint64_t>(2U * plane) * 4U * cores);
if ASCEND_IS_AIC { cube_.Init(d_, tm_, n_); }
SetAtomicNone();
if ASCEND_IS_AIV {
SetMaskNorm();
SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
}
const uint32_t tasksN = (d_ + n_ - 1U) / n_;
if constexpr (Resident) {
const uint32_t col=core%tasksN*n_;
const uint32_t cols=FullTile?n_:Min(n_,d_-col);
if ASCEND_IS_AIC {
cube_.PrepareWeights(w1_[static_cast<uint64_t>(col)*d_],w2_[static_cast<uint64_t>(col)*d_],cols);
}
if ASCEND_IS_AIV {
LocalTensor<T> bias(TPosition::VECIN,tm_*n_*sizeof(T),2U*n_);
if constexpr (Unaligned) {
DataCopyExtParams bp{1,cols*2U,0,0,0};
DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>((cols+15U)/16U*16U-cols),static_cast<T>(0)};
DataCopyPad(bias,b1_[col],bp,pad);DataCopyPad(bias[n_],b2_[col],bp,pad);
} else {
DataCopy(bias,b1_[col],cols);DataCopy(bias[n_],b2_[col],cols);
}
}
}
uint32_t local = 0;
for (uint32_t task = core; task < td->taskNum; task += cores, ++local) {
const uint32_t slot = local % 4U;
const uint64_t row = static_cast<uint64_t>(task / tasksN) * tm_;
const uint32_t col = task % tasksN * n_;
const uint32_t rows = FullTile?tm_:Min(tm_, td->m - row);
const uint32_t cols = FullTile?n_:Min(n_, d_ - col);
const uint64_t off = (static_cast<uint64_t>(core) * 4U + slot) * 2U * plane;
if ASCEND_IS_AIC {
if (local >= 4U) { WaitConsumed(slot); }
cube_.Compute(x_[row * d_], w1_[static_cast<uint64_t>(col) * d_],
w2_[static_cast<uint64_t>(col) * d_], slots_[off], rows, cols);
Ready(slot);
}
if ASCEND_IS_AIV {
WaitReady(slot);
Epilogue(off, row, col, rows, cols);
Consumed(slot);
}
}
if ASCEND_IS_AIC {
for (uint32_t slot = 0; slot < Min(local, 4U); ++slot) { WaitConsumed(slot); }
}
PipeBarrier<PIPE_ALL>();
}
private:
__aicore__ inline uint32_t Min(uint32_t a, uint64_t b) { return a < b ? a : b; }
template <HardEvent E> __aicore__ inline void Sync() {
constexpr event_t e = EVENT_ID0;
SetFlag<E>(e); WaitFlag<E>(e);
}
__aicore__ inline void Epilogue(uint64_t off, uint64_t row, uint32_t col,
uint32_t rows, uint32_t cols) {
const uint32_t half = FullTile?tm_/2U:(rows + 1U) / 2U;
const uint32_t begin = GetSubBlockIdx() * half;
if constexpr (!FullTile) { if (begin >= rows) { return; } }
const uint32_t nr = FullTile?tm_/2U:Min(half, rows - begin);
const uint32_t capacity = tm_ / 2U * n_;
LocalTensor<T> input(TPosition::VECIN, 0, (tm_ + 2U) * n_);
auto g = input;
auto u = input[capacity];
auto b1 = input[2U * capacity];
auto b2 = input[2U * capacity + n_];
LocalTensor<T> work(TPosition::VECOUT, (tm_ + 2U) * n_ * sizeof(T), tm_ * n_);
auto s = work;
auto out = work[capacity];
const uint64_t start = off + begin * 2U * n_;
if constexpr (Unaligned) {
const uint32_t pitch = (cols + 15U) / 16U * 16U;
DataCopyExtParams cp{static_cast<uint16_t>(nr), cols * 2U,
(2U * n_ - cols) * 2U, (n_ - pitch) / 16U, 0};
DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(pitch - cols), static_cast<T>(0)};
DataCopyPad(g, slots_[start], cp, pad);
DataCopyPad(u, slots_[start + n_], cp, pad);
DataCopyExtParams bp{1, cols * 2U, 0, 0, 0};
if constexpr (!Resident) {
DataCopyPad(b1, b1_[col], bp, pad);
DataCopyPad(b2, b2_[col], bp, pad);
}
} else {
DataCopyParams cp{static_cast<uint16_t>(nr), static_cast<uint16_t>(cols / 16U),
static_cast<uint16_t>((2U * n_ - cols) / 16U),
static_cast<uint16_t>((n_ - cols) / 16U)};
DataCopy(g, slots_[start], cp);
DataCopy(u, slots_[start + n_], cp);
if constexpr (!Resident) {
DataCopy(b1, b1_[col], cols);
DataCopy(b2, b2_[col], cols);
}
}
Sync<HardEvent::MTE2_V>();
const uint8_t stride = static_cast<uint8_t>(n_ / 16U);
const BinaryRepeatParams repeat{1, 1, 1, stride, stride, 0};
Add(g, g, b1, static_cast<uint64_t>(cols), static_cast<uint8_t>(nr), repeat);
Add(u, u, b2, static_cast<uint64_t>(cols), static_cast<uint8_t>(nr), repeat);
PipeBarrier<PIPE_V>();
const uint32_t count = nr * n_;
GateCounter<T,true>(g,u,s,out,out,count);
Sync<HardEvent::V_MTE3>();
if constexpr (Unaligned) {
const uint32_t pitch = (cols + 15U) / 16U * 16U;
DataCopyExtParams output{static_cast<uint16_t>(nr), cols * 2U,
(n_ - pitch) / 16U, (d_ - cols) * 2U, 0};
DataCopyPad(y_[(row + begin) * d_ + col], out, output);
} else {
DataCopyParams output{static_cast<uint16_t>(nr), static_cast<uint16_t>(cols / 16U),
static_cast<uint16_t>((n_ - cols) / 16U),
static_cast<uint16_t>((d_ - cols) / 16U)};
DataCopy(y_[(row + begin) * d_ + col], out, output);
}
Sync<HardEvent::MTE3_MTE2>();
}
__aicore__ inline void WaitReady(uint32_t s) {
if (s == 0) { CrossCoreWaitFlag(0x6); } else if (s == 1) { CrossCoreWaitFlag(0x7); }
else if (s == 2) { CrossCoreWaitFlag(0xB); } else { CrossCoreWaitFlag(0xE); }
}
__aicore__ inline void Ready(uint32_t s) {
if (s == 0) { CrossCoreSetFlag<0x2, PIPE_FIX>(0x6); } else if (s == 1) { CrossCoreSetFlag<0x2, PIPE_FIX>(0x7); }
else if (s == 2) { CrossCoreSetFlag<0x2, PIPE_FIX>(0xB); } else { CrossCoreSetFlag<0x2, PIPE_FIX>(0xE); }
}
__aicore__ inline void WaitConsumed(uint32_t s) {
if (s == 0) { CrossCoreWaitFlag(0x8); } else if (s == 1) { CrossCoreWaitFlag(0x9); }
else if (s == 2) { CrossCoreWaitFlag(0xC); } else { CrossCoreWaitFlag(0xF); }
}
__aicore__ inline void Consumed(uint32_t s) {
if (s == 0) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0x8); } else if (s == 1) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0x9); }
else if (s == 2) { CrossCoreSetFlag<0x2, PIPE_MTE3>(0xC); } else { CrossCoreSetFlag<0x2, PIPE_MTE3>(0xF); }
}
DirectPairCube<T, TileM, FixedD, Unaligned, Resident, FullTile> cube_;
GlobalTensor<T> x_, w1_, w2_, b1_, b2_, y_, slots_;
static constexpr uint32_t tm_ = TileM;
uint32_t d_ = 0, n_ = 0;
};
template <typename T>
class UnalignedTileCube {
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y,GM_ADDR ws) {
uint32_t core=GetBlockIdx();
if ASCEND_IS_AIV {core/=2U;}
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy,tmp;
gx.SetGlobalBuffer((__gm__ T*)x);gw1.SetGlobalBuffer((__gm__ T*)w1);gw2.SetGlobalBuffer((__gm__ T*)w2);
gb1.SetGlobalBuffer((__gm__ T*)b1);gb2.SetGlobalBuffer((__gm__ T*)b2);gy.SetGlobalBuffer((__gm__ T*)y);
tmp.SetGlobalBuffer((__gm__ T*)ws);
const uint32_t off=core*32U*96U;
SetAtomicNone();
if ASCEND_IS_AIC {
cube_.Init(33U,32U,48U);
cube_.Compute(gx[core*17U*33U],gw1,gw2,tmp[off],17U,33U);
CrossCoreSetFlag<0x2,PIPE_FIX>(0x6);
CrossCoreWaitFlag(0x8);
}
if ASCEND_IS_AIV {
const uint32_t sub=GetSubBlockIdx(), nr=9U-sub, begin=9U*sub;
LocalTensor<T> g(TPosition::VECIN,0,432),u(TPosition::VECIN,864,432);
LocalTensor<T> b1t(TPosition::VECIN,1728,48),b2t(TPosition::VECIN,1824,48);
LocalTensor<T> sig(TPosition::VECCALC,1920,432),out(TPosition::VECOUT,2784,432);
SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
DataCopyExtParams bc{1,66,0,0,0};DataCopyPadExtParams<T> pad{true,0,15,static_cast<T>(0)};
DataCopyPad(b1t,gb1,bc,pad);DataCopyPad(b2t,gb2,bc,pad);
CrossCoreWaitFlag(0x6);
DataCopyParams cp{static_cast<uint16_t>(nr),3,3,0};
DataCopy(g,tmp[off+begin*96U],cp);DataCopy(u,tmp[off+begin*96U+48U],cp);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
const BinaryRepeatParams rp{1,1,1,3,3,0};
Add(g,g,b1t,static_cast<uint64_t>(33),static_cast<uint8_t>(nr),rp);
Add(u,u,b2t,static_cast<uint64_t>(33),static_cast<uint8_t>(nr),rp);
PipeBarrier<PIPE_V>();GateCounter<T,true>(g,u,sig,out,out,nr*48U);
SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
DataCopyExtParams oc{static_cast<uint16_t>(nr),66,0,0,0};
DataCopyPad(gy[(core*17U+begin)*33U],out,oc);
CrossCoreSetFlag<0x2,PIPE_MTE3>(0x8);
}
PipeBarrier<PIPE_ALL>();
}
private:
DirectPairCube<T,32U,0U,true,false,true> cube_;
};
template <typename T,uint32_t TM,uint32_t D,uint32_t N,uint32_t Iterations>
class DoubleBufferedPairCube {
static constexpr uint32_t A=TM*D,B=2U*N*D;
static constexpr bool SplitL0B=(B*sizeof(T)>64U*1024U);
static constexpr bool DoubleA2=(2U*A*sizeof(T)<=64U*1024U);
static constexpr bool DoubleC=(4U*TM*N*sizeof(float)<=128U*1024U);
static constexpr uint32_t L1Slots=SplitL0B?2U:Iterations;
static_assert((L1Slots*A+B)*sizeof(T)<=512U*1024U,"L1 capacity");
static_assert(A*sizeof(T)<=64U*1024U && (SplitL0B ? N*D*sizeof(T)<=64U*1024U : B*sizeof(T)<=64U*1024U),"L0A/B capacity");
public:
template<uint32_t FirstTM=TM>
__aicore__ inline void PrepareWeights(const GlobalTensor<T>& w1,const GlobalTensor<T>& w2,const GlobalTensor<T>& x0) {
LocalTensor<T> b1(TPosition::B1,2U*A*sizeof(T),B);
LocalTensor<T> a1(TPosition::A1,0,FirstTM*D);
LocalTensor<T> b2_0(TPosition::B2,0,N*D/2U);
LocalTensor<T> b2_1(TPosition::B2,N*D/2U*sizeof(T),N*D/2U);
constexpr uint32_t HM=TM/2U;
const Nd2NzParams hap{1,HM,D,0,D,HM,1,0};
const Nd2NzParams bp{1,N/2U,D,0,D,N/2U,1,0};
DataCopy(a1,x0,hap);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
DataCopy(b1,w1,bp);DataCopy(b1[N*D/2U],w1[(N/2U)*D],bp);
SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
DataCopy(a1[HM*D],x0[HM*D],hap);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
DataCopy(b1[N*D],w2,bp);DataCopy(b1[N*D+N*D/2U],w2[(N/2U)*D],bp);
SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
LocalTensor<T> a2_0(TPosition::A2,0U,HM*D);
LocalTensor<T> a2_1(TPosition::A2,HM*D*sizeof(T),HM*D);
LoadData2dParams al;al.repeatTimes=D/16U;al.srcStride=HM/16U;
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
#pragma unroll
for(uint32_t mb=0;mb<HM/16U;++mb) {LoadData(a2_0[mb*D*16U],a1[mb*256U],al);}
SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
LoadData2dParams bl;bl.repeatTimes=(N*D/2U)/256U;bl.srcStride=1;
LoadData(b2_0,b1,bl);
SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
LoadData(b2_1,b1[N*D/2U],bl);
SetFlag<HardEvent::MTE1_M>(EVENT_ID3);
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
#pragma unroll
for(uint32_t mb=0;mb<HM/16U;++mb) {LoadData(a2_1[mb*D*16U],a1[HM*D+mb*256U],al);}
SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
}
template<uint32_t TM0,uint32_t TM1,uint32_t TM2,uint32_t TM3=TM,uint32_t TM4=TM0>
__aicore__ inline void PrepareAll(const GlobalTensor<T>& w1,const GlobalTensor<T>& w2,
const GlobalTensor<T>& x0,const GlobalTensor<T>& x1,
const GlobalTensor<T>& x2,const GlobalTensor<T>& x3,
const GlobalTensor<T>& x4) {
LocalTensor<T> a1_0(TPosition::A1,0U,TM0*D);
LocalTensor<T> a1_1(TPosition::A1,A*sizeof(T),TM1*D);
LocalTensor<T> a1_2(TPosition::A1,2U*A*sizeof(T),TM2*D);
LocalTensor<T> b1(TPosition::B1,Iterations*A*sizeof(T),B);
LocalTensor<T> a2_0(TPosition::A2,0U,TM0*D);
LocalTensor<T> b2_0(TPosition::B2,0U,N*D);
LocalTensor<T> b2_1(TPosition::B2,N*D*sizeof(T),N*D);
const Nd2NzParams ap0{1,TM0,D,0,D,TM0,1,0};
const Nd2NzParams ap1{1,TM1,D,0,D,TM1,1,0};
const Nd2NzParams ap2{1,TM2,D,0,D,TM2,1,0};
const Nd2NzParams bp{1,N,D,0,D,N,1,0};
DataCopy(a1_0,x0,ap0);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
DataCopy(b1,w1,bp);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
DataCopy(b1[N*D],w2,bp);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
DataCopy(a1_1,x1,ap1);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
if constexpr (Iterations>=3U) {SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);}
LoadData2dParams al0;al0.repeatTimes=D/16U;al0.srcStride=TM0/16U;
#pragma unroll
for(uint32_t mb=0;mb<TM0/16U;++mb) {LoadData(a2_0[mb*D*16U],a1_0[mb*256U],al0);}
SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
if constexpr (Iterations>=3U) {
WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
DataCopy(a1_2,x2,ap2);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
}
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
if constexpr (Iterations>=4U) {SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);}
LoadData2dParams bl;bl.repeatTimes=(N*D)/256U;bl.srcStride=1;
LoadData(b2_0,b1,bl);
SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
if constexpr (Iterations>=4U) {
LocalTensor<T> a1_3(TPosition::A1,3U*A*sizeof(T),TM3*D);
const Nd2NzParams ap3{1,TM3,D,0,D,TM3,1,0};
WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
DataCopy(a1_3,x3,ap3);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
}
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
if constexpr (Iterations>=5U) {SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);}
LoadData(b2_1,b1[N*D],bl);
SetFlag<HardEvent::MTE1_M>(EVENT_ID3);
if constexpr (Iterations>=5U) {
LocalTensor<T> a1_4(TPosition::A1,4U*A*sizeof(T),TM4*D);
const Nd2NzParams ap4{1,TM4,D,0,D,TM4,1,0};
WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
DataCopy(a1_4,x4,ap4);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
}
}
template<uint32_t I,uint32_t CurTM=TM> __aicore__ inline void Prefetch(const GlobalTensor<T>& x) {
constexpr event_t slot=I%2U==0U?EVENT_ID0:EVENT_ID1;
if constexpr(I>=2U || (SplitL0B && I==1U)) {WaitFlag<HardEvent::MTE1_MTE2>(slot);}
LocalTensor<T> a1(TPosition::A1,(I%2U)*A*sizeof(T),CurTM*D);
const Nd2NzParams ap{1,CurTM,D,0,D,CurTM,1,0};
DataCopy(a1,x,ap);SetFlag<HardEvent::MTE2_MTE1>(slot);
}
template<uint32_t I,uint32_t CurTM=TM> __aicore__ inline void Compute(const GlobalTensor<T>& dst) {
constexpr event_t slot=I%2U==0U?EVENT_ID0:EVENT_ID1;
constexpr event_t cSlot=DoubleC?slot:EVENT_ID0;
if constexpr (SplitL0B) {
LocalTensor<T> a1(TPosition::A1,(I%2U)*A*sizeof(T),CurTM*D);
LocalTensor<T> b1(TPosition::B1,2U*A*sizeof(T),B);
LocalTensor<T> b2_0(TPosition::B2,0,N*D/2U);
LocalTensor<T> b2_1(TPosition::B2,N*D/2U*sizeof(T),N*D/2U);
LoadData2dParams bl;bl.repeatTimes=(N*D/2U)/256U;bl.srcStride=1;
constexpr uint32_t HM=TM/2U;
constexpr uint32_t firstOff=(I%2U==1U)?(HM*N):0U;
constexpr uint32_t secondOff=(I%2U==1U)?0U:(HM*N);
constexpr uint32_t nextB1Off=(I%2U==1U)?0U:(N*D);
if constexpr (CurTM<=HM) {
constexpr uint32_t cFirstOff=(I%2U==1U)?(CurTM*N):0U;
constexpr uint32_t cSecondOff=(I%2U==1U)?0U:(CurTM*N);
LocalTensor<T> a2_0(TPosition::A2,0U,CurTM*D);
LocalTensor<float> c0(TPosition::CO1,0U,2U*HM*N);
if constexpr(I>0U) {
WaitFlag<HardEvent::MTE2_MTE1>(slot);
WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
LoadData2dParams al;al.repeatTimes=D/16U;al.srcStride=CurTM/16U;
#pragma unroll
for(uint32_t mb=0;mb<CurTM/16U;++mb) {LoadData(a2_0[mb*D*16U],a1[mb*256U],al);}
SetFlag<HardEvent::MTE1_MTE2>(slot);
SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
}
WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
if constexpr(I==0U) {WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);}
if constexpr(I>0U) {WaitFlag<HardEvent::FIX_M>(EVENT_ID0);}
MmadParams mad;mad.m=CurTM;mad.n=N/2U;mad.k=D;mad.cmatrixInitVal=true;
Mmad(c0[cFirstOff],a2_0,b2_0,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
if constexpr(I==0U) {WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);}
Mmad(c0[cFirstOff+CurTM*N/2U],a2_0,b2_1,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
if constexpr(I==0U) {WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);}
WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
LoadData(b2_0,b1[nextB1Off],bl);
SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
LoadData(b2_1,b1[nextB1Off+N*D/2U],bl);
SetFlag<HardEvent::MTE1_M>(EVENT_ID3);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);
Mmad(c0[cSecondOff],a2_0,b2_0,mad);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);
Mmad(c0[cSecondOff+CurTM*N/2U],a2_0,b2_1,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
SetFlag<HardEvent::M_FIX>(EVENT_ID0);
FixpipeParamsV220 fix;fix.nSize=2U*N;fix.mSize=CurTM;fix.srcStride=CurTM;fix.dstStride=2U*N;fix.quantPre=QuantMode_t::F322F16;
WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
if constexpr (I>=4U) {
constexpr uint32_t cid=(I%4U)==0U?0xA:((I%4U)==1U?0xB:((I%4U)==2U?0xC:0xE));
CrossCoreWaitFlag(cid);
}
Fixpipe<T,float,CFG_ROW_MAJOR>(dst,c0,fix);
SetFlag<HardEvent::FIX_M>(EVENT_ID0);
} else {
LocalTensor<T> a2_0(TPosition::A2,0U,HM*D);
LocalTensor<T> a2_1(TPosition::A2,HM*D*sizeof(T),HM*D);
LocalTensor<float> c0(TPosition::CO1,0U,2U*HM*N);
LocalTensor<float> c1(TPosition::CO1,2U*HM*N*sizeof(float),2U*HM*N);
if constexpr (I>0U) {
WaitFlag<HardEvent::MTE2_MTE1>(slot);
WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
LoadData2dParams al;al.repeatTimes=D/16U;al.srcStride=CurTM/16U;
#pragma unroll
for(uint32_t mb=0;mb<HM/16U;++mb) {LoadData(a2_0[mb*D*16U],a1[mb*256U],al);}
SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
#pragma unroll
for(uint32_t mb=0;mb<HM/16U;++mb) {LoadData(a2_1[mb*D*16U],a1[(mb+HM/16U)*256U],al);}
SetFlag<HardEvent::MTE1_MTE2>(slot);
SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
}
WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
if constexpr(I==0U) {WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);}
if constexpr(I>0U) {WaitFlag<HardEvent::FIX_M>(EVENT_ID0);}
MmadParams mad;mad.m=HM;mad.n=N/2U;mad.k=D;mad.cmatrixInitVal=true;
Mmad(c0[firstOff],a2_0,b2_0,mad);
if constexpr(I==0U) {WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);}
Mmad(c0[firstOff+HM*N/2U],a2_0,b2_1,mad);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
if constexpr(I>0U) {WaitFlag<HardEvent::FIX_M>(EVENT_ID1);}
Mmad(c1[firstOff],a2_1,b2_0,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
Mmad(c1[firstOff+HM*N/2U],a2_1,b2_1,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
if constexpr(I==0U) {WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);}
WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
LoadData(b2_0,b1[nextB1Off],bl);
SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
LoadData(b2_1,b1[nextB1Off+N*D/2U],bl);
SetFlag<HardEvent::MTE1_M>(EVENT_ID3);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);
Mmad(c0[secondOff],a2_0,b2_0,mad);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);
Mmad(c0[secondOff+HM*N/2U],a2_0,b2_1,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
SetFlag<HardEvent::M_FIX>(EVENT_ID0);
Mmad(c1[secondOff],a2_1,b2_0,mad);
Mmad(c1[secondOff+HM*N/2U],a2_1,b2_1,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
SetFlag<HardEvent::M_FIX>(EVENT_ID1);
FixpipeParamsV220 fix;fix.nSize=2U*N;fix.mSize=HM;fix.srcStride=HM;fix.dstStride=2U*N;fix.quantPre=QuantMode_t::F322F16;
WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
if constexpr (I>=4U) {
constexpr uint32_t cid=(I%4U)==0U?0xA:((I%4U)==1U?0xB:((I%4U)==2U?0xC:0xE));
CrossCoreWaitFlag(cid);
}
Fixpipe<T,float,CFG_ROW_MAJOR>(dst,c0,fix);
SetFlag<HardEvent::FIX_M>(EVENT_ID0);
WaitFlag<HardEvent::M_FIX>(EVENT_ID1);
Fixpipe<T,float,CFG_ROW_MAJOR>(dst[HM*2U*N],c1,fix);
SetFlag<HardEvent::FIX_M>(EVENT_ID1);
}
} else if constexpr (TM==48U) {
LocalTensor<T> a1(TPosition::A1,I*A*sizeof(T),CurTM*D);
LocalTensor<T> b2_0(TPosition::B2,0U,N*D);
LocalTensor<T> b2_1(TPosition::B2,N*D*sizeof(T),N*D);
if constexpr (I==0U) {
LocalTensor<T> a2_0(TPosition::A2,0U,32U*D);
LocalTensor<float> c0(TPosition::CO1,0U,2U*32U*N);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);
MmadParams mad;mad.m=32U;mad.n=N;mad.k=D;mad.cmatrixInitVal=true;
Mmad(c0,a2_0,b2_0,mad);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);
Mmad(c0[32U*N],a2_0,b2_1,mad);
SetFlag<HardEvent::M_FIX>(EVENT_ID0);WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
FixpipeParamsV220 fix;fix.nSize=2U*N;fix.mSize=32U;fix.srcStride=32U;fix.dstStride=2U*N;fix.quantPre=QuantMode_t::F322F16;
Fixpipe<T,float,CFG_ROW_MAJOR>(dst,c0,fix);SetFlag<HardEvent::FIX_M>(EVENT_ID0);
} else if constexpr (I==1U) {
LocalTensor<T> a2_1(TPosition::A2,32U*D*sizeof(T),48U*D);
LocalTensor<float> c1(TPosition::CO1,2U*32U*N*sizeof(float),2U*48U*N);
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
LoadData2dParams al;al.repeatTimes=D/16U;al.srcStride=48U/16U;
#pragma unroll
for(uint32_t mb=0;mb<48U/16U;++mb) {LoadData(a2_1[mb*D*16U],a1[mb*256U],al);}
SetFlag<HardEvent::MTE1_M>(EVENT_ID1);WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
MmadParams mad;mad.m=48U;mad.n=N;mad.k=D;mad.cmatrixInitVal=true;
Mmad(c1,a2_1,b2_0,mad);
Mmad(c1[48U*N],a2_1,b2_1,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
SetFlag<HardEvent::M_FIX>(EVENT_ID1);WaitFlag<HardEvent::M_FIX>(EVENT_ID1);
FixpipeParamsV220 fix;fix.nSize=2U*N;fix.mSize=48U;fix.srcStride=48U;fix.dstStride=2U*N;fix.quantPre=QuantMode_t::F322F16;
Fixpipe<T,float,CFG_ROW_MAJOR>(dst,c1,fix);SetFlag<HardEvent::FIX_M>(EVENT_ID1);
} else {
LocalTensor<T> a2_2(TPosition::A2,80U*D*sizeof(T),48U*D);
LocalTensor<float> c2(TPosition::CO1,2U*80U*N*sizeof(float),2U*48U*N);
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
LoadData2dParams al;al.repeatTimes=D/16U;al.srcStride=48U/16U;
#pragma unroll
for(uint32_t mb=0;mb<48U/16U;++mb) {LoadData(a2_2[mb*D*16U],a1[mb*256U],al);}
SetFlag<HardEvent::MTE1_M>(EVENT_ID0);WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
MmadParams mad;mad.m=48U;mad.n=N;mad.k=D;mad.cmatrixInitVal=true;
Mmad(c2,a2_2,b2_0,mad);
Mmad(c2[48U*N],a2_2,b2_1,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
SetFlag<HardEvent::M_FIX>(EVENT_ID2);WaitFlag<HardEvent::M_FIX>(EVENT_ID2);
FixpipeParamsV220 fix;fix.nSize=2U*N;fix.mSize=48U;fix.srcStride=48U;fix.dstStride=2U*N;fix.quantPre=QuantMode_t::F322F16;
Fixpipe<T,float,CFG_ROW_MAJOR>(dst,c2,fix);
SetFlag<HardEvent::FIX_M>(EVENT_ID2);
}
} else {
LocalTensor<T> a1(TPosition::A1,I*A*sizeof(T),CurTM*D);
LocalTensor<T> a2(TPosition::A2,I*A*sizeof(T),CurTM*D);
LocalTensor<T> b2_0(TPosition::B2,0U,N*D);
LocalTensor<T> b2_1(TPosition::B2,N*D*sizeof(T),N*D);
LocalTensor<float> c(TPosition::CO1,(I%2U)*2U*TM*N*sizeof(float),2U*TM*N);
if constexpr (I>0U) {
constexpr event_t mte2Ev=(I==1U)?EVENT_ID1:((I==2U)?EVENT_ID0:EVENT_ID2);
WaitFlag<HardEvent::MTE2_MTE1>(mte2Ev);
if constexpr (I>=2U) {WaitFlag<HardEvent::M_MTE1>(slot);}
LoadData2dParams al;al.repeatTimes=D/16U;al.srcStride=CurTM/16U;
#pragma unroll
for(uint32_t mb=0;mb<CurTM/16U;++mb) {LoadData(a2[mb*D*16U],a1[mb*256U],al);}
SetFlag<HardEvent::MTE1_M>(slot);
}
WaitFlag<HardEvent::MTE1_M>(slot);
if constexpr (I<2U) {SetFlag<HardEvent::M_MTE1>(slot);}
if constexpr (I==0U) {WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);}
if constexpr (I>=2U) {WaitFlag<HardEvent::FIX_M>(cSlot);}
MmadParams mad;mad.m=CurTM;mad.n=N;mad.k=D;mad.cmatrixInitVal=true;
Mmad(c,a2,b2_0,mad);
if constexpr (I==0U) {WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);}
Mmad(c[CurTM*N],a2,b2_1,mad);
if constexpr (I>=2U) {SetFlag<HardEvent::M_MTE1>(slot);}
SetFlag<HardEvent::M_FIX>(cSlot);WaitFlag<HardEvent::M_FIX>(cSlot);
FixpipeParamsV220 fix;fix.nSize=2U*N;fix.mSize=CurTM;fix.srcStride=CurTM;fix.dstStride=2U*N;fix.quantPre=QuantMode_t::F322F16;
Fixpipe<T,float,CFG_ROW_MAJOR>(dst,c,fix);SetFlag<HardEvent::FIX_M>(cSlot);
}
}
__aicore__ inline void Drain() {
WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
if constexpr (TM==48U) {
WaitFlag<HardEvent::FIX_M>(EVENT_ID2);
}
if constexpr (SplitL0B) {
WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
}
}
};
template <typename T,uint32_t D,uint32_t TM,uint32_t Cores,uint32_t Iterations,uint32_t TileN=64U>
class PipelinedCubeSwiGlu {
static constexpr uint32_t N=TileN, TN=D/N, Groups=Cores/TN, Plane=TM*N;
static constexpr uint32_t M=(Groups==4U)?512U:((Groups==10U)?8192U:4096U);
static constexpr uint32_t FirstTM=(Groups==4U)?32U:TM;
static constexpr bool FastAiv=true;
static constexpr bool SplitL0B=(2U*N*D*sizeof(T)>64U*1024U);
static constexpr uint32_t NR=TM/2U, Cap=NR*N;
static constexpr uint32_t InSlots=SplitL0B?2U:4U;
static constexpr uint32_t OutSlots=SplitL0B?2U:4U;
static_assert(D%N==0U && TM%16U==0U,"Exact tile shape");
static_assert(TM*D*sizeof(T)<=64U*1024U,"L0A capacity");
static_assert(N*D*sizeof(T)<=64U*1024U,"L0B half capacity");
static_assert(2U*TM*N*sizeof(float)<=128U*1024U,"L0C capacity");
static_assert(((2U*InSlots+3U+OutSlots)*Cap+2U*N)*sizeof(T)<=192U*1024U,"UB capacity");
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y,GM_ADDR workspace) {
core_=GetBlockIdx();
if ASCEND_IS_AIV {core_/=2U;}
col_=core_%TN*N;
grp_=core_/TN;
if constexpr (Groups==4U) {
baseRow_=grp_*128U;
} else if constexpr (Groups==10U) {
baseRow_=(grp_<8U)?(grp_*832U):(6656U+(grp_-8U)*768U);
} else {
baseRow_=grp_*256U;
}
x_.SetGlobalBuffer((__gm__ T*)x,M*D);y_.SetGlobalBuffer((__gm__ T*)y,M*D);
w1_.SetGlobalBuffer((__gm__ T*)w1,D*D);w2_.SetGlobalBuffer((__gm__ T*)w2,D*D);
b1_.SetGlobalBuffer((__gm__ T*)b1,D);b2_.SetGlobalBuffer((__gm__ T*)b2,D);
slots_.SetGlobalBuffer((__gm__ T*)workspace,2U*Plane*4U*Cores);
SetAtomicNone();
if ASCEND_IS_AIC {
if constexpr (Groups==4U) {
piped_.template PrepareAll<32U,48U,48U>(w1_[col_*D],w2_[col_*D],
x_[baseRow_*D],x_[(baseRow_+32U)*D],
x_[(baseRow_+80U)*D],x_[baseRow_*D],x_[baseRow_*D]);
} else if constexpr (Groups==16U) {
piped_.template PrepareAll<64U,64U,64U,64U>(w1_[col_*D],w2_[col_*D],
x_[(grp_*64U)*D],x_[(1024U+grp_*64U)*D],
x_[(2048U+grp_*64U)*D],x_[(3072U+grp_*64U)*D],x_[(grp_*64U)*D]);
} else {
piped_.template PrepareWeights<FirstTM>(w1_[col_*D],w2_[col_*D],x_[baseRow_*D]);
}
}
if ASCEND_IS_AIV {
SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
LocalTensor<T> bias(TPosition::VECIN,2U*InSlots*Cap*sizeof(T),2U*N);
DataCopy(bias,b1_[col_],N);DataCopy(bias[N],b2_[col_],N);
if constexpr (FastAiv) {
constexpr uint32_t FirstNR=FirstTM/2U, FirstCap=FirstNR*N;
WaitReady<0U>();
LocalTensor<T> input0(TPosition::VECIN,0,2U*Cap);
DataCopy(input0,slots_[core_*8U*Plane+GetSubBlockIdx()*FirstNR*2U*N],2U*FirstCap);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
}
}
Step<0>();
if ASCEND_IS_AIC {piped_.Drain();}
PipeBarrier<PIPE_ALL>();
}
private:
template <HardEvent E> __aicore__ inline void Sync() {SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0);}
template <uint32_t Slot> __aicore__ inline void Ready() {
constexpr uint32_t id=Slot==0U?0x6:(Slot==1U?0x7:(Slot==2U?0x8:0x9));
CrossCoreSetFlag<0x2,PIPE_FIX>(id);
}
template <uint32_t Slot> __aicore__ inline void WaitReady() {
constexpr uint32_t id=Slot==0U?0x6:(Slot==1U?0x7:(Slot==2U?0x8:0x9));
CrossCoreWaitFlag(id);
}
template <uint32_t Slot> __aicore__ inline void Consumed() {
constexpr uint32_t id=Slot==0U?0xA:(Slot==1U?0xB:(Slot==2U?0xC:0xE));
CrossCoreSetFlag<0x2,PIPE_MTE2>(id);
}
template <uint32_t I> __aicore__ inline void Step() {
constexpr uint32_t Slot=I%4U;
const uint64_t off=(core_*4U+Slot)*2U*Plane;
if constexpr (Groups==4U) {
constexpr uint32_t curTM=(I==0U)?32U:48U;
constexpr uint64_t rowOff=(I==0U)?0U:((I==1U)?32U:80U);
const uint64_t row=baseRow_+rowOff;
if ASCEND_IS_AIC {
piped_.template Compute<I,curTM>(slots_[off]);
Ready<Slot>();
}
if ASCEND_IS_AIV {
Epilogue<I,curTM>(off,row);
}
} else if constexpr (Groups==10U) {
constexpr uint32_t CurTM=(I<6U)?128U:64U;
const uint64_t row=baseRow_+I*128U;
if ASCEND_IS_AIC {
if constexpr (I<5U) {
piped_.template Prefetch<I+1U,128U>(x_[(row+128U)*D]);
} else if constexpr (I==5U) {
const uint64_t pfRow=(grp_<8U)?(row+128U):baseRow_;
piped_.template Prefetch<6U,64U>(x_[pfRow*D]);
}
piped_.template Compute<I,CurTM>(slots_[off]);
Ready<Slot>();
}
if ASCEND_IS_AIV {
Epilogue<I,CurTM>(off,row);
}
} else {
const uint64_t row=(I*1024U)+grp_*64U;
if ASCEND_IS_AIC {
piped_.template Compute<I,64U>(slots_[off]);
Ready<Slot>();
}
if ASCEND_IS_AIV {
Epilogue<I,64U>(off,row);
}
}
if constexpr (I+1U<Iterations) {Step<I+1U>();}
}
template <uint32_t I,uint32_t CurTM=TM> __aicore__ inline void Epilogue(uint64_t off,uint64_t row) {
constexpr uint32_t ubSlot=I%InSlots;
constexpr event_t vEv=ubSlot==0U?EVENT_ID0:(ubSlot==1U?EVENT_ID1:(ubSlot==2U?EVENT_ID2:EVENT_ID3));
constexpr uint32_t CurNR=CurTM/2U, CurCap=CurNR*N;
const uint32_t begin=GetSubBlockIdx()*CurNR;
LocalTensor<T> input(TPosition::VECIN,0,2U*InSlots*Cap+2U*N);
auto curInput=input[ubSlot*2U*Cap];
auto b1=input[2U*InSlots*Cap];auto b2=input[2U*InSlots*Cap+N];
LocalTensor<T> work(TPosition::VECOUT,(2U*InSlots*Cap+2U*N)*sizeof(T),(3U+OutSlots)*Cap);
auto g=work;auto u=work[Cap];auto sig=work[2U*Cap];auto out=work[(3U+ubSlot)*Cap];
WaitFlag<HardEvent::MTE2_V>(vEv);
if constexpr (I==0U && Iterations>4U) {
Consumed<0U>();
}
const BinaryRepeatParams rp{1,1,1,N/16U,2U*N/16U,0};
constexpr uint64_t mask=0;
if constexpr (N!=128U) {SetVectorMask<T,MaskMode::NORMAL>(0,~0ULL);}
Add<T,false>(g,curInput,b1,mask,static_cast<uint8_t>(CurNR),rp);
Add<T,false>(u,curInput[N],b2,mask,static_cast<uint8_t>(CurNR),rp);
if constexpr (I+InSlots<Iterations) {
SetFlag<HardEvent::V_MTE2>(vEv);
}
static_assert(CurCap%128U==0U && CurCap/128U<=255U);
constexpr uint8_t repeats=CurCap/128U;
const UnaryRepeatParams ur{1,1,8,8};
const BinaryRepeatParams br{1,1,1,8,8,8};
if constexpr (N!=128U) {SetVectorMask<T,MaskMode::NORMAL>(~0ULL,~0ULL);}
Muls<T,false>(sig,g,static_cast<T>(-1),mask,repeats,ur);
Exp<T,false>(sig,sig,mask,repeats,ur);
Adds<T,false>(sig,sig,static_cast<T>(1),mask,repeats,ur);
Div<T,false>(sig,g,sig,mask,repeats,br);
if constexpr (I>=OutSlots) {
WaitFlag<HardEvent::MTE3_V>(vEv);
}
Mul<T,false>(out,sig,u,mask,repeats,br);
if constexpr (SplitL0B) {
SetFlag<HardEvent::V_MTE3>(vEv);
WaitFlag<HardEvent::V_MTE3>(vEv);
DataCopyParams op{CurNR,N/16U,0,(D-N)/16U};
if constexpr (Groups==10U && I==6U) {
if (grp_<8U) {DataCopy(y_[(row+begin)*D+col_],out,op);}
} else {
DataCopy(y_[(row+begin)*D+col_],out,op);
}
if constexpr (I+OutSlots<Iterations) {
SetFlag<HardEvent::MTE3_V>(vEv);
}
if constexpr (I+1U<Iterations) {
constexpr uint32_t nextSlot=(I+1U)%4U;
constexpr uint32_t nextUbSlot=(I+1U)%InSlots;
constexpr event_t nextEv=nextUbSlot==0U?EVENT_ID0:(nextUbSlot==1U?EVENT_ID1:(nextUbSlot==2U?EVENT_ID2:EVENT_ID3));
WaitReady<nextSlot>();
if constexpr (I+1U>=InSlots) {
WaitFlag<HardEvent::V_MTE2>(nextEv);
}
const uint64_t nextOff=(core_*4U+nextSlot)*2U*Plane;
constexpr uint32_t nextNR=(I<5U)?64U:32U;
DataCopy(input[nextUbSlot*2U*Cap],slots_[nextOff+GetSubBlockIdx()*nextNR*2U*N],2U*nextNR*N);
if constexpr (I+5U<Iterations) {
Consumed<nextSlot>();
}
SetFlag<HardEvent::MTE2_V>(nextEv);
}
} else {
SetFlag<HardEvent::V_MTE3>(vEv);
if constexpr (I+1U<Iterations) {
constexpr uint32_t nextSlot=(I+1U)%4U;
constexpr uint32_t nextUbSlot=(I+1U)%InSlots;
constexpr event_t nextEv=nextUbSlot==0U?EVENT_ID0:(nextUbSlot==1U?EVENT_ID1:(nextUbSlot==2U?EVENT_ID2:EVENT_ID3));
WaitReady<nextSlot>();
if constexpr (I+1U>=InSlots) {
WaitFlag<HardEvent::V_MTE2>(nextEv);
}
const uint64_t nextOff=(core_*4U+nextSlot)*2U*Plane;
constexpr uint32_t nextNR=(Groups==4U)?24U:32U;
DataCopy(input[nextUbSlot*2U*Cap],slots_[nextOff+GetSubBlockIdx()*nextNR*2U*N],2U*nextNR*N);
SetFlag<HardEvent::MTE2_V>(nextEv);
}
WaitFlag<HardEvent::V_MTE3>(vEv);
DataCopyParams op{CurNR,N/16U,0,(D-N)/16U};
if constexpr (D==N) {
DataCopy(y_[(row+begin)*D],out,CurCap);
} else {
DataCopy(y_[(row+begin)*D+col_],out,op);
}
if constexpr (I+OutSlots<Iterations) {
SetFlag<HardEvent::MTE3_V>(vEv);
}
}
}
DoubleBufferedPairCube<T,TM,D,N,Iterations> piped_;
GlobalTensor<T> x_,w1_,w2_,b1_,b2_,y_,slots_;
uint32_t core_=0,col_=0,grp_=0,baseRow_=0;
};
}
namespace optiling {
template <typename T,uint32_t TM,uint32_t D,uint32_t N,uint32_t Iterations>
class SingleConsumerPairCube {
static constexpr uint32_t A=TM*D,B=2U*N*D;
static constexpr bool SplitL0B=(B*sizeof(T)>64U*1024U);
static constexpr bool DoubleA2=(2U*A*sizeof(T)<=64U*1024U);
static constexpr bool DoubleC=(4U*TM*N*sizeof(float)<=128U*1024U);
static constexpr uint32_t L1Slots=SplitL0B?2U:Iterations;
static_assert((L1Slots*A+B)*sizeof(T)<=512U*1024U,"L1 capacity");
static_assert(A*sizeof(T)<=64U*1024U && (SplitL0B ? N*D*sizeof(T)<=64U*1024U : B*sizeof(T)<=64U*1024U),"L0A/B capacity");
public:
template<uint32_t FirstTM=TM>
__aicore__ inline void PrepareWeights(const GlobalTensor<T>& w1,const GlobalTensor<T>& w2,const GlobalTensor<T>& x0) {
LocalTensor<T> b1(TPosition::B1,2U*A*sizeof(T),B);
LocalTensor<T> a1(TPosition::A1,0,FirstTM*D);
LocalTensor<T> b2_0(TPosition::B2,0,N*D/2U);
LocalTensor<T> b2_1(TPosition::B2,N*D/2U*sizeof(T),N*D/2U);
constexpr uint32_t HM=TM/2U;
const Nd2NzParams hap{1,HM,D,0,D,HM,1,0};
const Nd2NzParams bp{1,N/2U,D,0,D,N/2U,1,0};
DataCopy(a1,x0,hap);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
DataCopy(b1,w1,bp);DataCopy(b1[N*D/2U],w1[(N/2U)*D],bp);
SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
DataCopy(a1[HM*D],x0[HM*D],hap);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
DataCopy(b1[N*D],w2,bp);DataCopy(b1[N*D+N*D/2U],w2[(N/2U)*D],bp);
SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
LocalTensor<T> a2_0(TPosition::A2,0U,HM*D);
LocalTensor<T> a2_1(TPosition::A2,HM*D*sizeof(T),HM*D);
LoadData2dParams al;al.repeatTimes=D/16U;al.srcStride=HM/16U;
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
#pragma unroll
for(uint32_t mb=0;mb<HM/16U;++mb) {LoadData(a2_0[mb*D*16U],a1[mb*256U],al);}
SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
LoadData2dParams bl;bl.repeatTimes=(N*D/2U)/256U;bl.srcStride=1;
LoadData(b2_0,b1,bl);
SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
LoadData(b2_1,b1[N*D/2U],bl);
SetFlag<HardEvent::MTE1_M>(EVENT_ID3);
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
#pragma unroll
for(uint32_t mb=0;mb<HM/16U;++mb) {LoadData(a2_1[mb*D*16U],a1[HM*D+mb*256U],al);}
SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
}
template<uint32_t TM0,uint32_t TM1,uint32_t TM2,uint32_t TM3=TM,uint32_t TM4=TM0>
__aicore__ inline void PrepareAll(const GlobalTensor<T>& w1,const GlobalTensor<T>& w2,
 const GlobalTensor<T>& x0,const GlobalTensor<T>& x1,const GlobalTensor<T>& x2,const GlobalTensor<T>& x3,const GlobalTensor<T>& x4) {
 LocalTensor<T> a1(TPosition::A1,0U,Iterations*A);
 LocalTensor<T> b1(TPosition::B1,Iterations*A*sizeof(T),B);
 LocalTensor<T> a2(TPosition::A2,0U,TM0*D),b2(TPosition::B2,0U,B);
 const Nd2NzParams ap0{1,TM0,D,0,D,TM0,1,0},ap1{1,TM1,D,0,D,TM1,1,0},ap2{1,TM2,D,0,D,TM2,1,0};
 const Nd2NzParams bp{1,N,D,0,D,N,1,0};
 DataCopy(a1,x0,ap0);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
 DataCopy(b1,w1,bp);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
 DataCopy(b1[N*D],w2,bp);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
 DataCopy(a1[A],x1,ap1);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
 if constexpr(Iterations>=3U) {DataCopy(a1[2U*A],x2,ap2);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);}
 if constexpr(Iterations>=4U) {
  const Nd2NzParams ap3{1,TM3,D,0,D,TM3,1,0};DataCopy(a1[3U*A],x3,ap3);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID5);
 }
 if constexpr(Iterations>=5U) {
  const Nd2NzParams ap4{1,TM4,D,0,D,TM4,1,0};DataCopy(a1[4U*A],x4,ap4);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID6);
 }
 WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
 LoadData2dParams al;al.repeatTimes=D/16U;al.srcStride=TM0/16U;
 #pragma unroll
 for(uint32_t mb=0;mb<TM0/16U;++mb) {LoadData(a2[mb*D*16U],a1[mb*256U],al);}
 SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
 LoadData2dParams bl;bl.repeatTimes=N*D/256U;bl.srcStride=1;
 WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);LoadData(b2,b1,bl);SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
 WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);LoadData(b2[N*D],b1[N*D],bl);SetFlag<HardEvent::MTE1_M>(EVENT_ID3);
}
template<uint32_t I,uint32_t CurTM=TM> __aicore__ inline void Prefetch(const GlobalTensor<T>& x) {
constexpr event_t slot=I%2U==0U?EVENT_ID0:EVENT_ID1;
if constexpr(I>=2U || (SplitL0B && I==1U)) {WaitFlag<HardEvent::MTE1_MTE2>(slot);}
LocalTensor<T> a1(TPosition::A1,(I%2U)*A*sizeof(T),CurTM*D);
const Nd2NzParams ap{1,CurTM,D,0,D,CurTM,1,0};
DataCopy(a1,x,ap);SetFlag<HardEvent::MTE2_MTE1>(slot);
}
template<uint32_t I,uint32_t CurTM=TM> __aicore__ inline void Compute(const GlobalTensor<T>& dst) {
constexpr event_t slot=I%2U==0U?EVENT_ID0:EVENT_ID1;
constexpr event_t cSlot=DoubleC?slot:EVENT_ID0;
if constexpr (SplitL0B) {
LocalTensor<T> a1(TPosition::A1,(I%2U)*A*sizeof(T),CurTM*D);
LocalTensor<T> b1(TPosition::B1,2U*A*sizeof(T),B);
LocalTensor<T> b2_0(TPosition::B2,0,N*D/2U);
LocalTensor<T> b2_1(TPosition::B2,N*D/2U*sizeof(T),N*D/2U);
LoadData2dParams bl;bl.repeatTimes=(N*D/2U)/256U;bl.srcStride=1;
constexpr uint32_t HM=TM/2U;
constexpr uint32_t firstOff=(I%2U==1U)?(HM*N):0U;
constexpr uint32_t secondOff=(I%2U==1U)?0U:(HM*N);
constexpr uint32_t nextB1Off=(I%2U==1U)?0U:(N*D);
if constexpr (CurTM<=HM) {
constexpr uint32_t cFirstOff=(I%2U==1U)?(CurTM*N):0U;
constexpr uint32_t cSecondOff=(I%2U==1U)?0U:(CurTM*N);
LocalTensor<T> a2_0(TPosition::A2,0U,CurTM*D);
LocalTensor<float> c0(TPosition::CO1,0U,2U*HM*N);
if constexpr(I>0U) {
WaitFlag<HardEvent::MTE2_MTE1>(slot);
WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
LoadData2dParams al;al.repeatTimes=D/16U;al.srcStride=CurTM/16U;
#pragma unroll
for(uint32_t mb=0;mb<CurTM/16U;++mb) {LoadData(a2_0[mb*D*16U],a1[mb*256U],al);}
SetFlag<HardEvent::MTE1_MTE2>(slot);
SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
}
WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
if constexpr(I==0U) {WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);}
if constexpr(I>0U) {WaitFlag<HardEvent::FIX_M>(EVENT_ID0);}
MmadParams mad;mad.m=CurTM;mad.n=N/2U;mad.k=D;mad.cmatrixInitVal=true;
Mmad(c0[cFirstOff],a2_0,b2_0,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
if constexpr(I==0U) {WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);}
Mmad(c0[cFirstOff+CurTM*N/2U],a2_0,b2_1,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
if constexpr(I==0U) {WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);}
WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
LoadData(b2_0,b1[nextB1Off],bl);
SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
LoadData(b2_1,b1[nextB1Off+N*D/2U],bl);
SetFlag<HardEvent::MTE1_M>(EVENT_ID3);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);
Mmad(c0[cSecondOff],a2_0,b2_0,mad);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);
Mmad(c0[cSecondOff+CurTM*N/2U],a2_0,b2_1,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
SetFlag<HardEvent::M_FIX>(EVENT_ID0);
FixpipeParamsV220 fix;fix.nSize=2U*N;fix.mSize=CurTM;fix.srcStride=CurTM;fix.dstStride=2U*N;fix.quantPre=QuantMode_t::F322F16;
WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
if constexpr (I>=4U) {
constexpr uint32_t cid=(I%4U)==0U?0xA:((I%4U)==1U?0xB:((I%4U)==2U?0xC:0xE));
CrossCoreWaitFlag(cid);
}
Fixpipe<T,float,CFG_ROW_MAJOR>(dst,c0,fix);
SetFlag<HardEvent::FIX_M>(EVENT_ID0);
} else {
LocalTensor<T> a2_0(TPosition::A2,0U,HM*D);
LocalTensor<T> a2_1(TPosition::A2,HM*D*sizeof(T),HM*D);
LocalTensor<float> c0(TPosition::CO1,0U,2U*HM*N);
LocalTensor<float> c1(TPosition::CO1,2U*HM*N*sizeof(float),2U*HM*N);
if constexpr (I>0U) {
WaitFlag<HardEvent::MTE2_MTE1>(slot);
WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
LoadData2dParams al;al.repeatTimes=D/16U;al.srcStride=CurTM/16U;
#pragma unroll
for(uint32_t mb=0;mb<HM/16U;++mb) {LoadData(a2_0[mb*D*16U],a1[mb*256U],al);}
SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
#pragma unroll
for(uint32_t mb=0;mb<HM/16U;++mb) {LoadData(a2_1[mb*D*16U],a1[(mb+HM/16U)*256U],al);}
SetFlag<HardEvent::MTE1_MTE2>(slot);
SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
}
WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
if constexpr(I==0U) {WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);}
if constexpr(I>0U) {WaitFlag<HardEvent::FIX_M>(EVENT_ID0);}
MmadParams mad;mad.m=HM;mad.n=N/2U;mad.k=D;mad.cmatrixInitVal=true;
Mmad(c0[firstOff],a2_0,b2_0,mad);
if constexpr(I==0U) {WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);}
Mmad(c0[firstOff+HM*N/2U],a2_0,b2_1,mad);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
if constexpr(I>0U) {WaitFlag<HardEvent::FIX_M>(EVENT_ID1);}
Mmad(c1[firstOff],a2_1,b2_0,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
Mmad(c1[firstOff+HM*N/2U],a2_1,b2_1,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
if constexpr(I==0U) {WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);}
WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
LoadData(b2_0,b1[nextB1Off],bl);
SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
LoadData(b2_1,b1[nextB1Off+N*D/2U],bl);
SetFlag<HardEvent::MTE1_M>(EVENT_ID3);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);
Mmad(c0[secondOff],a2_0,b2_0,mad);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);
Mmad(c0[secondOff+HM*N/2U],a2_0,b2_1,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
SetFlag<HardEvent::M_FIX>(EVENT_ID0);
Mmad(c1[secondOff],a2_1,b2_0,mad);
Mmad(c1[secondOff+HM*N/2U],a2_1,b2_1,mad);
SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
SetFlag<HardEvent::M_FIX>(EVENT_ID1);
FixpipeParamsV220 fix;fix.nSize=2U*N;fix.mSize=HM;fix.srcStride=HM;fix.dstStride=2U*N;fix.quantPre=QuantMode_t::F322F16;
WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
if constexpr (I>=4U) {
constexpr uint32_t cid=(I%4U)==0U?0xA:((I%4U)==1U?0xB:((I%4U)==2U?0xC:0xE));
CrossCoreWaitFlag(cid);
}
Fixpipe<T,float,CFG_ROW_MAJOR>(dst,c0,fix);
SetFlag<HardEvent::FIX_M>(EVENT_ID0);
WaitFlag<HardEvent::M_FIX>(EVENT_ID1);
Fixpipe<T,float,CFG_ROW_MAJOR>(dst[HM*2U*N],c1,fix);
SetFlag<HardEvent::FIX_M>(EVENT_ID1);
}
} else if constexpr (TM==48U) {
LocalTensor<T> a1(TPosition::A1,I*A*sizeof(T),CurTM*D);
LocalTensor<T> b2_0(TPosition::B2,0U,N*D);
LocalTensor<T> b2_1(TPosition::B2,N*D*sizeof(T),N*D);
if constexpr (I==0U) {
LocalTensor<T> a2_0(TPosition::A2,0U,32U*D);
LocalTensor<float> c0(TPosition::CO1,0U,2U*32U*N);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);

WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);
MmadParams mad;mad.m=32U;mad.n=N;mad.k=D;mad.cmatrixInitVal=true;
Mmad(c0,a2_0,b2_0,mad);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);
Mmad(c0[32U*N],a2_0,b2_1,mad);
SetFlag<HardEvent::M_FIX>(EVENT_ID0);WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
FixpipeParamsV220 fix;fix.nSize=2U*N;fix.mSize=32U;fix.srcStride=32U;fix.dstStride=2U*N;fix.quantPre=QuantMode_t::F322F16;
Fixpipe<T,float,CFG_ROW_MAJOR>(dst,c0,fix);SetFlag<HardEvent::FIX_M>(EVENT_ID0);
} else if constexpr (I==1U) {
LocalTensor<T> a2_1(TPosition::A2,32U*D*sizeof(T),48U*D);
LocalTensor<float> c1(TPosition::CO1,2U*32U*N*sizeof(float),2U*48U*N);
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
LoadData2dParams al;al.repeatTimes=D/16U;al.srcStride=48U/16U;
#pragma unroll
for(uint32_t mb=0;mb<48U/16U;++mb) {LoadData(a2_1[mb*D*16U],a1[mb*256U],al);}
SetFlag<HardEvent::MTE1_M>(EVENT_ID1);WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
MmadParams mad;mad.m=48U;mad.n=N;mad.k=D;mad.cmatrixInitVal=true;
Mmad(c1,a2_1,b2_0,mad);
Mmad(c1[48U*N],a2_1,b2_1,mad);

SetFlag<HardEvent::M_FIX>(EVENT_ID1);WaitFlag<HardEvent::M_FIX>(EVENT_ID1);
FixpipeParamsV220 fix;fix.nSize=2U*N;fix.mSize=48U;fix.srcStride=48U;fix.dstStride=2U*N;fix.quantPre=QuantMode_t::F322F16;
Fixpipe<T,float,CFG_ROW_MAJOR>(dst,c1,fix);SetFlag<HardEvent::FIX_M>(EVENT_ID1);
} else {
LocalTensor<T> a2_2(TPosition::A2,80U*D*sizeof(T),48U*D);
LocalTensor<float> c2(TPosition::CO1,2U*80U*N*sizeof(float),2U*48U*N);
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);

LoadData2dParams al;al.repeatTimes=D/16U;al.srcStride=48U/16U;
#pragma unroll
for(uint32_t mb=0;mb<48U/16U;++mb) {LoadData(a2_2[mb*D*16U],a1[mb*256U],al);}
SetFlag<HardEvent::MTE1_M>(EVENT_ID4);WaitFlag<HardEvent::MTE1_M>(EVENT_ID4);
MmadParams mad;mad.m=48U;mad.n=N;mad.k=D;mad.cmatrixInitVal=true;
Mmad(c2,a2_2,b2_0,mad);
Mmad(c2[48U*N],a2_2,b2_1,mad);

SetFlag<HardEvent::M_FIX>(EVENT_ID2);WaitFlag<HardEvent::M_FIX>(EVENT_ID2);
FixpipeParamsV220 fix;fix.nSize=2U*N;fix.mSize=48U;fix.srcStride=48U;fix.dstStride=2U*N;fix.quantPre=QuantMode_t::F322F16;
Fixpipe<T,float,CFG_ROW_MAJOR>(dst,c2,fix);
SetFlag<HardEvent::FIX_M>(EVENT_ID2);
}
} else {
LocalTensor<T> a1(TPosition::A1,I*A*sizeof(T),CurTM*D);
LocalTensor<T> a2(TPosition::A2,I*A*sizeof(T),CurTM*D);
LocalTensor<T> b2_0(TPosition::B2,0U,N*D);
LocalTensor<T> b2_1(TPosition::B2,N*D*sizeof(T),N*D);
LocalTensor<float> c(TPosition::CO1,(I%2U)*2U*TM*N*sizeof(float),2U*TM*N);
if constexpr (I>0U) {
constexpr event_t mte2Ev=(I==1U)?EVENT_ID1:((I==2U)?EVENT_ID4:EVENT_ID5);
WaitFlag<HardEvent::MTE2_MTE1>(mte2Ev);
if constexpr (I>=2U) {WaitFlag<HardEvent::M_MTE1>(slot);}
LoadData2dParams al;al.repeatTimes=D/16U;al.srcStride=CurTM/16U;
#pragma unroll
for(uint32_t mb=0;mb<CurTM/16U;++mb) {LoadData(a2[mb*D*16U],a1[mb*256U],al);}
SetFlag<HardEvent::MTE1_M>(slot);
}
WaitFlag<HardEvent::MTE1_M>(slot);
if constexpr (I<2U) {SetFlag<HardEvent::M_MTE1>(slot);}
if constexpr (I==0U) {WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);}
if constexpr (I>=2U) {WaitFlag<HardEvent::FIX_M>(cSlot);}
MmadParams mad;mad.m=CurTM;mad.n=N;mad.k=D;mad.cmatrixInitVal=true;
Mmad(c,a2,b2_0,mad);
if constexpr (I==0U) {WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);}
Mmad(c[CurTM*N],a2,b2_1,mad);
if constexpr (I>=2U) {SetFlag<HardEvent::M_MTE1>(slot);}
SetFlag<HardEvent::M_FIX>(cSlot);WaitFlag<HardEvent::M_FIX>(cSlot);
FixpipeParamsV220 fix;fix.nSize=2U*N;fix.mSize=CurTM;fix.srcStride=CurTM;fix.dstStride=2U*N;fix.quantPre=QuantMode_t::F322F16;
Fixpipe<T,float,CFG_ROW_MAJOR>(dst,c,fix);SetFlag<HardEvent::FIX_M>(cSlot);
}
}
__aicore__ inline void Drain() {
if constexpr(TM!=48U) {WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);}
if constexpr(TM!=48U) {WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);}
WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
if constexpr (TM==48U) {
WaitFlag<HardEvent::FIX_M>(EVENT_ID2);
}
if constexpr (SplitL0B) {
WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
}
}
};
template<typename T> class ProjectionStreamCase3 {
static constexpr uint32_t M=64U,N=128U,D=256U,Cap=M*N;
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y,GM_ADDR workspace) {
const uint32_t core=GetBlockIdx(),row=(core/2U)*M,col=(core%2U)*N;
GlobalTensor<T> gx,gw1,gw2,gb1,gb2,gy,ws;
gx.SetGlobalBuffer((__gm__ T*)x,512U*D);gy.SetGlobalBuffer((__gm__ T*)y,512U*D);
gw1.SetGlobalBuffer((__gm__ T*)w1,D*D);gw2.SetGlobalBuffer((__gm__ T*)w2,D*D);
gb1.SetGlobalBuffer((__gm__ T*)b1,D);gb2.SetGlobalBuffer((__gm__ T*)b2,D);
ws.SetGlobalBuffer((__gm__ T*)workspace,16U*2U*Cap);
const uint32_t off=core*2U*Cap;
SetAtomicNone();
if ASCEND_IS_AIC {
LocalTensor<T> ax(TPosition::A1,0U,M*D),wg(TPosition::B1,M*D*sizeof(T),N*D),wu(TPosition::B1,(M*D+N*D)*sizeof(T),N*D);
LocalTensor<T> a2(TPosition::A2,0U,M*D),b2Local(TPosition::B2,0U,N*D);
LocalTensor<float> cg(TPosition::CO1,0U,Cap),cu(TPosition::CO1,Cap*sizeof(float),Cap);
const Nd2NzParams ap{1,M,D,0,D,M,1,0},bp{1,N,D,0,D,N,1,0};
DataCopy(ax,gx[row*D],ap);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
DataCopy(wg,gw1[col*D],bp);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
DataCopy(wu,gw2[col*D],bp);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
LoadData2dParams al;al.repeatTimes=D/16U;al.srcStride=M/16U;
#pragma unroll
for(uint32_t mb=0;mb<M/16U;++mb) {LoadData(a2[mb*D*16U],ax[mb*256U],al);}
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
LoadData2dParams bl;bl.repeatTimes=N*D/256U;bl.srcStride=1;
LoadData(b2Local,wg,bl);SetFlag<HardEvent::MTE1_M>(EVENT_ID0);WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
MmadParams mad;mad.m=M;mad.n=N;mad.k=D;mad.cmatrixInitVal=true;
Mmad(cg,a2,b2Local,mad);SetFlag<HardEvent::M_FIX>(EVENT_ID0);SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
FixpipeParamsV220 fix;fix.nSize=N;fix.mSize=M;fix.srcStride=M;fix.dstStride=N;fix.quantPre=QuantMode_t::F322F16;
WaitFlag<HardEvent::M_FIX>(EVENT_ID0);Fixpipe<T,float,CFG_ROW_MAJOR>(ws[off],cg,fix);
CrossCoreSetFlag<0x2,PIPE_FIX>(0x6);
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
LoadData(b2Local,wu,bl);SetFlag<HardEvent::MTE1_M>(EVENT_ID1);WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
Mmad(cu,a2,b2Local,mad);SetFlag<HardEvent::M_FIX>(EVENT_ID1);
WaitFlag<HardEvent::M_FIX>(EVENT_ID1);Fixpipe<T,float,CFG_ROW_MAJOR>(ws[off+Cap],cu,fix);
CrossCoreSetFlag<0x2,PIPE_FIX>(0x7);
}
if ASCEND_IS_AIV {
SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
LocalTensor<T> inputG(TPosition::VECIN,0U,Cap),inputU(TPosition::VECIN,Cap*sizeof(T),Cap);
LocalTensor<T> biasG(TPosition::VECIN,33024U,N),biasU(TPosition::VECIN,33536U,N);
LocalTensor<T> g(TPosition::VECCALC,66048U,Cap),u(TPosition::VECCALC,82432U,Cap),sig(TPosition::VECCALC,131328U,Cap);
LocalTensor<T> out(TPosition::VECOUT,0U,Cap);
DataCopy(biasG,gb1[col],N);DataCopy(biasU,gb2[col],N);
CrossCoreWaitFlag(0x6);DataCopy(inputG,ws[off],Cap);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
constexpr uint64_t mask=0;
const BinaryRepeatParams biasRp{1,1,1,8,8,0},br{1,1,1,8,8,8};
const UnaryRepeatParams ur{1,1,8,8};
Add<T,false>(g,inputG,biasG,mask,M,biasRp);
Muls<T,false>(sig,g,static_cast<T>(-1),mask,Cap/128U,ur);
Exp<T,false>(sig,sig,mask,Cap/128U,ur);
Adds<T,false>(sig,sig,static_cast<T>(1),mask,Cap/128U,ur);
Div<T,false>(sig,g,sig,mask,Cap/128U,br);
CrossCoreWaitFlag(0x7);DataCopy(inputU,ws[off+Cap],Cap);
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
Add<T,false>(u,inputU,biasU,mask,M,biasRp);
Mul<T,false>(out,sig,u,mask,Cap/128U,br);
SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
const DataCopyParams outputCopy{M,N/16U,0,(D-N)/16U};
DataCopy(gy[row*D+col],out,outputCopy);
}
PipeBarrier<PIPE_ALL>();
}
};
template <typename T,uint32_t D,uint32_t TM,uint32_t Cores,uint32_t Iterations,uint32_t TileN=64U>
class SingleConsumerSwiGlu {
static constexpr uint32_t N=TileN, TN=D/N, Groups=Cores/TN, Plane=TM*N;
static constexpr uint32_t M=(Groups==4U)?512U:((Groups==10U)?8192U:4096U);
static constexpr uint32_t FirstTM=(Groups==4U)?32U:TM;
static constexpr bool FastAiv=true;
static constexpr bool SplitL0B=(2U*N*D*sizeof(T)>64U*1024U);
static constexpr uint32_t NR=TM, Cap=NR*N;
static constexpr uint32_t InSlots=3U;
static constexpr uint32_t OutSlots=3U;
static_assert(D%N==0U && TM%16U==0U,"Exact tile shape");
static_assert(TM*D*sizeof(T)<=64U*1024U,"L0A capacity");
static_assert(N*D*sizeof(T)<=64U*1024U,"L0B half capacity");
static_assert(2U*TM*N*sizeof(float)<=128U*1024U,"L0C capacity");
static_assert(((2U*InSlots+3U+OutSlots)*Cap+2U*N)*sizeof(T)<=192U*1024U,"UB capacity");
public:
__aicore__ inline void Run(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y,GM_ADDR workspace) {
core_=GetBlockIdx();
// AIC:AIV 1:1: both processors have the same logical block index.
col_=core_%TN*N;
grp_=core_/TN;
if constexpr (Groups==4U) {
baseRow_=grp_*128U;
} else if constexpr (Groups==10U) {
baseRow_=(grp_<8U)?(grp_*832U):(6656U+(grp_-8U)*768U);
} else {
baseRow_=grp_*256U;
}
x_.SetGlobalBuffer((__gm__ T*)x,M*D);y_.SetGlobalBuffer((__gm__ T*)y,M*D);
w1_.SetGlobalBuffer((__gm__ T*)w1,D*D);w2_.SetGlobalBuffer((__gm__ T*)w2,D*D);
b1_.SetGlobalBuffer((__gm__ T*)b1,D);b2_.SetGlobalBuffer((__gm__ T*)b2,D);
slots_.SetGlobalBuffer((__gm__ T*)workspace,2U*Plane*4U*Cores);
SetAtomicNone();
if ASCEND_IS_AIC {
if constexpr (Groups==4U) {
piped_.template PrepareAll<32U,48U,48U>(w1_[col_*D],w2_[col_*D],
x_[baseRow_*D],x_[(baseRow_+32U)*D],
x_[(baseRow_+80U)*D],x_[baseRow_*D],x_[baseRow_*D]);
} else if constexpr (Groups==16U) {
piped_.template PrepareAll<64U,64U,64U,64U>(w1_[col_*D],w2_[col_*D],
x_[(grp_*64U)*D],x_[(1024U+grp_*64U)*D],
x_[(2048U+grp_*64U)*D],x_[(3072U+grp_*64U)*D],x_[(grp_*64U)*D]);
} else {
piped_.template PrepareWeights<FirstTM>(w1_[col_*D],w2_[col_*D],x_[baseRow_*D]);
}
}
if ASCEND_IS_AIV {
SetMaskNorm();SetVectorMask<uint64_t,MaskMode::NORMAL>(~0ULL,~0ULL);
LocalTensor<T> bias(TPosition::VECIN,2U*InSlots*Cap*sizeof(T),2U*N);
DataCopy(bias,b1_[col_],N);DataCopy(bias[N],b2_[col_],N);
if constexpr (FastAiv) {
constexpr uint32_t FirstNR=FirstTM, FirstCap=FirstNR*N;
WaitReady<0U>();
LocalTensor<T> input0(TPosition::VECIN,0,2U*Cap);
DataCopy(input0,slots_[core_*8U*Plane+0U*FirstNR*2U*N],2U*FirstCap);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
}
}
Step<0>();
if ASCEND_IS_AIC {piped_.Drain();}
PipeBarrier<PIPE_ALL>();
}
private:
template <HardEvent E> __aicore__ inline void Sync() {SetFlag<E>(EVENT_ID0);WaitFlag<E>(EVENT_ID0);}
template <uint32_t Slot> __aicore__ inline void Ready() {
constexpr uint32_t id=Slot==0U?0x6:(Slot==1U?0x7:(Slot==2U?0x8:0x9));
CrossCoreSetFlag<0x2,PIPE_FIX>(id);
}
template <uint32_t Slot> __aicore__ inline void WaitReady() {
constexpr uint32_t id=Slot==0U?0x6:(Slot==1U?0x7:(Slot==2U?0x8:0x9));
CrossCoreWaitFlag(id);
}
template <uint32_t Slot> __aicore__ inline void Consumed() {
constexpr uint32_t id=Slot==0U?0xA:(Slot==1U?0xB:(Slot==2U?0xC:0xE));
CrossCoreSetFlag<0x2,PIPE_MTE2>(id);
}
template <uint32_t I> __aicore__ inline void Step() {
constexpr uint32_t Slot=I%4U;
const uint64_t off=(core_*4U+Slot)*2U*Plane;
if constexpr (Groups==4U) {
constexpr uint32_t curTM=(I==0U)?32U:48U;
constexpr uint64_t rowOff=(I==0U)?0U:((I==1U)?32U:80U);
const uint64_t row=baseRow_+rowOff;
if ASCEND_IS_AIC {
piped_.template Compute<I,curTM>(slots_[off]);
Ready<Slot>();
}
if ASCEND_IS_AIV {
Epilogue<I,curTM>(off,row);
}
} else if constexpr (Groups==10U) {
constexpr uint32_t CurTM=(I<6U)?128U:64U;
const uint64_t row=baseRow_+I*128U;
if ASCEND_IS_AIC {
if constexpr (I<5U) {
piped_.template Prefetch<I+1U,128U>(x_[(row+128U)*D]);
} else if constexpr (I==5U) {
const uint64_t pfRow=(grp_<8U)?(row+128U):baseRow_;
piped_.template Prefetch<6U,64U>(x_[pfRow*D]);
}
piped_.template Compute<I,CurTM>(slots_[off]);
Ready<Slot>();
}
if ASCEND_IS_AIV {
Epilogue<I,CurTM>(off,row);
}
} else {
const uint64_t row=(I*1024U)+grp_*64U;
if ASCEND_IS_AIC {
piped_.template Compute<I,64U>(slots_[off]);
Ready<Slot>();
}
if ASCEND_IS_AIV {
Epilogue<I,64U>(off,row);
}
}
if constexpr (I+1U<Iterations) {Step<I+1U>();}
}
template <uint32_t I,uint32_t CurTM=TM> __aicore__ inline void Epilogue(uint64_t off,uint64_t row) {
constexpr uint32_t ubSlot=I%InSlots;
constexpr event_t vEv=ubSlot==0U?EVENT_ID0:(ubSlot==1U?EVENT_ID1:(ubSlot==2U?EVENT_ID2:EVENT_ID3));
constexpr uint32_t CurNR=CurTM, CurCap=CurNR*N;
const uint32_t begin=0U*CurNR;
LocalTensor<T> input(TPosition::VECIN,0,2U*InSlots*Cap+2U*N);
auto curInput=input[ubSlot*2U*Cap];
auto b1=input[2U*InSlots*Cap];auto b2=input[2U*InSlots*Cap+N];
LocalTensor<T> work(TPosition::VECOUT,(2U*InSlots*Cap+2U*N)*sizeof(T),(3U+OutSlots)*Cap);
LocalTensor<T> g(TPosition::VECCALC,66048U,Cap),u(TPosition::VECCALC,66048U+Cap*sizeof(T),Cap);
LocalTensor<T> sig(TPosition::VECCALC,131328U,Cap),out(TPosition::VECOUT,131328U+(1U+ubSlot)*Cap*sizeof(T),Cap);
WaitFlag<HardEvent::MTE2_V>(vEv);
if constexpr (I==0U && Iterations>4U) {
Consumed<0U>();
}
const BinaryRepeatParams rp{1,1,1,N/16U,2U*N/16U,0};
constexpr uint64_t mask=0;
if constexpr (N!=128U) {SetVectorMask<T,MaskMode::NORMAL>(0,~0ULL);}
Add<T,false>(g,curInput,b1,mask,static_cast<uint8_t>(CurNR),rp);
Add<T,false>(u,curInput[N],b2,mask,static_cast<uint8_t>(CurNR),rp);
if constexpr (I+InSlots<Iterations) {
SetFlag<HardEvent::V_MTE2>(vEv);
}
static_assert(CurCap%128U==0U && CurCap/128U<=255U);
constexpr uint8_t repeats=CurCap/128U;
const UnaryRepeatParams ur{1,1,8,8};
const BinaryRepeatParams br{1,1,1,8,8,8};
if constexpr (N!=128U) {SetVectorMask<T,MaskMode::NORMAL>(~0ULL,~0ULL);}
Muls<T,false>(sig,g,static_cast<T>(-1),mask,repeats,ur);
Exp<T,false>(sig,sig,mask,repeats,ur);
Adds<T,false>(sig,sig,static_cast<T>(1),mask,repeats,ur);
Div<T,false>(sig,g,sig,mask,repeats,br);
if constexpr (I>=OutSlots) {
WaitFlag<HardEvent::MTE3_V>(vEv);
}
Mul<T,false>(out,sig,u,mask,repeats,br);
if constexpr (SplitL0B) {
SetFlag<HardEvent::V_MTE3>(vEv);
WaitFlag<HardEvent::V_MTE3>(vEv);
DataCopyParams op{CurNR,N/16U,0,(D-N)/16U};
if constexpr (Groups==10U && I==6U) {
if (grp_<8U) {DataCopy(y_[(row+begin)*D+col_],out,op);}
} else {
DataCopy(y_[(row+begin)*D+col_],out,op);
}
if constexpr (I+OutSlots<Iterations) {
SetFlag<HardEvent::MTE3_V>(vEv);
}
if constexpr (I+1U<Iterations) {
constexpr uint32_t nextSlot=(I+1U)%4U;
constexpr uint32_t nextUbSlot=(I+1U)%InSlots;
constexpr event_t nextEv=nextUbSlot==0U?EVENT_ID0:(nextUbSlot==1U?EVENT_ID1:(nextUbSlot==2U?EVENT_ID2:EVENT_ID3));
WaitReady<nextSlot>();
if constexpr (I+1U>=InSlots) {
WaitFlag<HardEvent::V_MTE2>(nextEv);
}
const uint64_t nextOff=(core_*4U+nextSlot)*2U*Plane;
constexpr uint32_t nextNR=(I<5U)?64U:32U;
DataCopy(input[nextUbSlot*2U*Cap],slots_[nextOff+0U*nextNR*2U*N],2U*nextNR*N);
if constexpr (I+5U<Iterations) {
Consumed<nextSlot>();
}
SetFlag<HardEvent::MTE2_V>(nextEv);
}
} else {
SetFlag<HardEvent::V_MTE3>(vEv);
if constexpr (I+1U<Iterations) {
constexpr uint32_t nextSlot=(I+1U)%4U;
constexpr uint32_t nextUbSlot=(I+1U)%InSlots;
constexpr event_t nextEv=nextUbSlot==0U?EVENT_ID0:(nextUbSlot==1U?EVENT_ID1:(nextUbSlot==2U?EVENT_ID2:EVENT_ID3));
WaitReady<nextSlot>();
if constexpr (I+1U>=InSlots) {
WaitFlag<HardEvent::V_MTE2>(nextEv);
}
const uint64_t nextOff=(core_*4U+nextSlot)*2U*Plane;
constexpr uint32_t nextNR=(Groups==4U)?48U:64U;
DataCopy(input[nextUbSlot*2U*Cap],slots_[nextOff+0U*nextNR*2U*N],2U*nextNR*N);
SetFlag<HardEvent::MTE2_V>(nextEv);
}
WaitFlag<HardEvent::V_MTE3>(vEv);
DataCopyParams op{CurNR,N/16U,0,(D-N)/16U};
if constexpr (D==N) {
DataCopy(y_[(row+begin)*D],out,CurCap);
} else {
DataCopy(y_[(row+begin)*D+col_],out,op);
}
if constexpr (I+OutSlots<Iterations) {
SetFlag<HardEvent::MTE3_V>(vEv);
}
}
}
SingleConsumerPairCube<T,TM,D,N,Iterations> piped_;
GlobalTensor<T> x_,w1_,w2_,b1_,b2_,y_,slots_;
uint32_t core_=0,col_=0,grp_=0,baseRow_=0;
};
}
template <typename DT_INPUT_X, int PATH>
__global__ __aicore__ void swi_glu(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2, GM_ADDR bias_2,
GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
if constexpr (PATH>=43) {
if constexpr (PATH==43) {
KERNEL_TASK_TYPE(11009,KERNEL_TYPE_AIV_ONLY);
SubBlockGemv4<DT_INPUT_X> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output);
} else if constexpr (PATH==44) {
KERNEL_TASK_TYPE(11265,KERNEL_TYPE_AIV_ONLY);
AlignedColGemv<DT_INPUT_X,64U,2U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output);
} else if constexpr (PATH==45) {
KERNEL_TASK_TYPE(11521,KERNEL_TYPE_MIX_AIC_1_1);
optiling::SingleConsumerSwiGlu<DT_INPUT_X,256U,48U,16U,3U,64U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,workspace);
} else if constexpr (PATH==46) {
KERNEL_TASK_TYPE(11777,KERNEL_TYPE_AIV_ONLY);
AlignedRowGemv32<DT_INPUT_X> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output);
} else if constexpr (PATH==47) {
optiling::PipelinedCubeSwiGlu<DT_INPUT_X,128U,64U,16U,4U,128U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,workspace);
} else if constexpr (PATH==48) {
KERNEL_TASK_TYPE(12289,KERNEL_TYPE_AIV_ONLY);
ScalarChannelSwiGlu<DT_INPUT_X> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output);
} else if constexpr (PATH==49) {
KERNEL_TASK_TYPE(12545,KERNEL_TYPE_AIV_ONLY);
PackedSubBlockGemv<DT_INPUT_X> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output);
} else if constexpr (PATH==50) {
KERNEL_TASK_TYPE(12801,KERNEL_TYPE_AIV_ONLY);
PackedMultiBlockGemv<DT_INPUT_X> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output);
} else if constexpr (PATH==51) {
optiling::PipelinedCubeSwiGlu<DT_INPUT_X,256U,128U,20U,7U,128U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,workspace);
}
} else if constexpr (PATH==42) {
KERNEL_TASK_TYPE(10753,KERNEL_TYPE_AIV_ONLY);
REGISTER_TILING_DEFAULT(SwiGluTilingData);
BatchedTinyVectorSwiGlu<DT_INPUT_X,13U,7U,true> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,nullptr);
} else if constexpr (PATH >= 38) {
if constexpr (PATH==38) { KERNEL_TASK_TYPE(9729,KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH==39) { KERNEL_TASK_TYPE(9985,KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH==40) { KERNEL_TASK_TYPE(10241,KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH==41) { KERNEL_TASK_TYPE(10497,KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
if constexpr (PATH==41) {
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData,tiling_data,tiling);
TinyColumnsSwiGlu<DT_INPUT_X,0U,false> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else {
TinyColumnsSwiGlu<DT_INPUT_X,PATH==39?0U:13U,PATH!=40> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,nullptr);
}
} else if constexpr (PATH >= 36) {
if constexpr (PATH==36) { KERNEL_TASK_TYPE(9217,KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH==37) { KERNEL_TASK_TYPE(9473,KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData,tiling_data,tiling);
SmallMVectorSwiGlu<DT_INPUT_X,PATH==36?512U:0U,512U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else if constexpr (PATH >= 32) {
if (workspace == nullptr) { return; }
GM_ADDR userWorkspace=AscendC::GetUserWorkspace(workspace);
if (userWorkspace == nullptr) { return; }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluDirectTilingData,tiling_data,tiling);
constexpr uint32_t fixedD=PATH==32?256U:(PATH==33?128U:0U);
optiling::SlimDirectCube<DT_INPUT_X,64U,fixedD,PATH==35> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,userWorkspace,&tiling_data);
} else if constexpr (PATH >= 26) {
if (workspace == nullptr) { return; }
GM_ADDR userWorkspace=AscendC::GetUserWorkspace(workspace);
if (userWorkspace == nullptr) { return; }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluDirectTilingData,tiling_data,tiling);
constexpr uint32_t fixedD=(PATH==26||PATH==30)?256U:((PATH==27||PATH==31)?128U:0U);
optiling::SlimDirectCube<DT_INPUT_X,128U,fixedD,PATH==29,true,PATH>=30> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,userWorkspace,&tiling_data);
} else if constexpr (PATH >= 23) {
REGISTER_TILING_DEFAULT(SwiGluTilingData);
if constexpr (PATH == 23) { KERNEL_TASK_TYPE(5889, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 24) { KERNEL_TASK_TYPE(6145, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 25) { KERNEL_TASK_TYPE(6401, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 25) {
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData, tiling_data, tiling);
SingleTinySwiGlu<DT_INPUT_X,0U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else {
SingleTinySwiGlu<DT_INPUT_X,PATH==23?13U:1U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,nullptr);
}
} else if constexpr (PATH == 21 || PATH == 22) {
if constexpr (PATH == 21) { KERNEL_TASK_TYPE(5377, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 22) { KERNEL_TASK_TYPE(5633, KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData, tiling_data, tiling);
BatchedTinyVectorSwiGlu<DT_INPUT_X,PATH==21?13U:0U,4U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else if constexpr (PATH >= 17) {
if constexpr (PATH == 17) { KERNEL_TASK_TYPE(4353, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 18) { KERNEL_TASK_TYPE(4609, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 19) { KERNEL_TASK_TYPE(4865, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 20) { KERNEL_TASK_TYPE(5121, KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData, tiling_data, tiling);
constexpr uint32_t fixedD=PATH==17?64U:(PATH==18?128U:(PATH==19?256U:0U));
SmallMVectorSwiGlu<DT_INPUT_X,fixedD> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else if constexpr (PATH >= 14) {
if constexpr (PATH == 14) { KERNEL_TASK_TYPE(3585, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 15) { KERNEL_TASK_TYPE(3841, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 16) { KERNEL_TASK_TYPE(4097, KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData, tiling_data, tiling);
if constexpr (PATH == 15) {
ScalarFeatureSwiGlu<DT_INPUT_X> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
} else {
TinyVectorSwiGlu<DT_INPUT_X,PATH == 14 ? 13U : 0U> op;
op.Run(input_x,weight_1,bias_1,weight_2,bias_2,output,&tiling_data);
}
} else if constexpr (PATH == 0 || PATH == 4 || PATH == 5) {
if constexpr (PATH == 0) { KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 4) { KERNEL_TASK_TYPE(1025, KERNEL_TYPE_AIV_ONLY); }
if constexpr (PATH == 5) { KERNEL_TASK_TYPE(1281, KERNEL_TYPE_AIV_ONLY); }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tiling_data, tiling);
constexpr int mode = PATH == 4 ? 1 : (PATH == 5 ? 2 : 0);
KernelSwiGlu<DT_INPUT_X, mode> op;
op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, tiling_data);
op.Process();
} else {
if (workspace == nullptr) { return; }
GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
if (userWorkspace == nullptr) { return; }
REGISTER_TILING_DEFAULT(SwiGluTilingData);
if constexpr (PATH == 2 || PATH == 3 || PATH >= 8) {
GET_TILING_DATA_WITH_STRUCT(SwiGluDirectTilingData, tiling_data, tiling);
constexpr uint32_t tileM = (PATH == 3 || PATH == 9 || PATH == 11 || PATH == 13) ? 128U : 32U;
constexpr uint32_t fixedD = (PATH == 8 || PATH == 9) ? 256U :
((PATH == 10 || PATH == 11) ? 128U : 0U);
optiling::SlimDirectCube<DT_INPUT_X, tileM, fixedD, PATH == 12 || PATH == 13> op;
op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output, userWorkspace, &tiling_data);
} else {
AscendC::TPipe tPipe;
GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tiling_data, tiling);
optiling::RunCube<DT_INPUT_X, PATH == 6 || PATH == 7, PATH == 7>(
input_x, weight_1, bias_1, weight_2, bias_2, output, userWorkspace, &tPipe, &tiling_data);
tPipe.Destroy();
}
}
}
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 11009UL || TILING_KEY_VAR == 11265UL || TILING_KEY_VAR == 11521UL || TILING_KEY_VAR == 11777UL || TILING_KEY_VAR == 12033UL || TILING_KEY_VAR == 12289UL || TILING_KEY_VAR == 12545UL || TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL) && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON)
namespace AscendC {
__aicore__ inline void NoSysWorkspaceForce(GM_ADDR) {}
__aicore__ inline GM_ADDR FastGetUserWorkspace(GM_ADDR workspace) {
return workspace + RESERVED_WORKSPACE;
}
}
#define SetSysWorkspaceForce NoSysWorkspaceForce
#define GetUserWorkspace FastGetUserWorkspace
#define cce_kernel cce_kernel, aligned(512), section(".text.hot")
#if TILING_KEY_VAR == 11009UL || TILING_KEY_VAR == 11265UL || TILING_KEY_VAR == 11777UL || TILING_KEY_VAR == 12289UL || TILING_KEY_VAR == 12545UL || TILING_KEY_VAR == 12801UL
#define set_ffts_base_addr(...) ((void)0)
#endif
#endif
#endif
