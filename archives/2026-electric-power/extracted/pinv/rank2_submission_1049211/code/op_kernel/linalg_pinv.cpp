#include "kernel_operator.h"
#include <type_traits>
// CANN's generated mixed-kernel wrapper calls matmul::clearWorkspace.
#include "lib/matmul_intf.h"
#include "linalg_pinv_tiling.h"
#ifndef PINV_DIRECT_BENCH
#include "tiling_key_linalg_pinv.h"
#endif
using namespace AscendC;

// Generic vector fallback: one-sided Jacobi SVD on A (tall) or A^T (wide),
// with serial or cooperative execution and explicit singular-value truncation.
template<class T,bool serialOnly=false,bool mixedCube=false> class Pinv {
    __aicore__ inline uint32_t CoreCount() {return AscendC::GetBlockNum()*(mixedCube?2:1);}
    TPipe pipe;
    TBuf<TPosition::VECCALC> matrixBuf, pBuf, qBuf, uBuf, vBuf, prodBuf, reduceBuf, scalarBuf, normBuf, rawBuf, smallTransposeBuf, smallProductBuf, smallIndexBuf;
    LocalTensor<float> matrix, p, q, u, v, prod, reduce, scalar, norm;
    GlobalTensor<T> input, output;
    GlobalTensor<uint16_t> inputBits;
    GlobalTensor<float> work;
    LinalgPinvTilingData t;
    uint32_t d,k,dp,L;
    float inputFrobenius2[4];
    static constexpr uint32_t dotWidth=16,dotTile=dotWidth*256,dotTail=dotWidth*8;
    __aicore__ inline float Root(float x) {
        scalar.SetValue(0,x); PipeBarrier<PIPE_ALL>();
        Sqrt(scalar,scalar,1); PipeBarrier<PIPE_ALL>();
        return scalar.GetValue(0);
    }
    __aicore__ inline float Dot(LocalTensor<float> a, LocalTensor<float> b) {
        Mul(prod,a,b,d); PipeBarrier<PIPE_V>();
        ReduceSum(scalar,prod,reduce,d); PipeBarrier<PIPE_ALL>();
        return scalar.GetValue(0);
    }
    __aicore__ inline void AccumulateVector(LocalTensor<float> value, uint32_t count) {
        Sub(v,value,q,count); PipeBarrier<PIPE_V>();
        Add(prod,u,v,count); PipeBarrier<PIPE_V>();
        Sub(q,prod,u,count); PipeBarrier<PIPE_V>();
        Sub(q,q,v,count); PipeBarrier<PIPE_V>();
        DataCopy(u,prod,(count+7)/8*8); PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void AccumulateOutput(LocalTensor<float> values, float weight, uint32_t count) {
        Muls(prod,values,weight,count); PipeBarrier<PIPE_V>();
        AccumulateVector(prod,count);
    }
    __aicore__ inline void AccumulateDoubleVector(LocalTensor<float> value, uint32_t count) {
        auto high=norm[dp],error=norm[2*dp],z=norm[4*dp];
        Add(high,u,value,count); PipeBarrier<PIPE_V>();
        Sub(z,high,u,count); PipeBarrier<PIPE_V>();
        Sub(v,high,z,count); PipeBarrier<PIPE_V>();
        Sub(error,u,v,count); PipeBarrier<PIPE_V>();
        Sub(v,value,z,count); PipeBarrier<PIPE_V>();
        Add(error,error,v,count); PipeBarrier<PIPE_V>();
        Add(q,q,error,count); PipeBarrier<PIPE_V>();
        Add(u,high,q,count); PipeBarrier<PIPE_V>();
        Sub(v,u,high,count); PipeBarrier<PIPE_V>();
        Sub(q,q,v,count); PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void AccumulateDoubleOutput(LocalTensor<float> value,float weight,uint32_t count) {
        Muls(prod,value,weight,count);PipeBarrier<PIPE_V>();
        AccumulateDoubleVector(prod,count);
    }
    // Dekker product split plus compensated summation for an accurate I-A*X residual.
    __aicore__ inline void AccumulateExactProduct(LocalTensor<float> values, float weight, uint32_t count) {
        auto high=norm[dp],low=norm[2*dp],error=norm[3*dp],tmp=norm[4*dp];
        union {float value;uint32_t bits;} split;
        split.value=weight;split.bits&=0xfffff000u;
        float aHigh=split.value,aLow=weight-aHigh;
        Muls(tmp,values,4097.0f,count); Muls(prod,values,weight,count); PipeBarrier<PIPE_V>();
        Sub(high,tmp,values,count); PipeBarrier<PIPE_V>();
        Sub(high,tmp,high,count); PipeBarrier<PIPE_V>();
        Sub(low,values,high,count); PipeBarrier<PIPE_V>();
        Muls(error,high,aHigh,count); PipeBarrier<PIPE_V>();
        Sub(error,error,prod,count); Muls(tmp,low,aHigh,count); PipeBarrier<PIPE_V>();
        Add(error,error,tmp,count); PipeBarrier<PIPE_V>(); Muls(tmp,high,aLow,count); PipeBarrier<PIPE_V>();
        Add(error,error,tmp,count); PipeBarrier<PIPE_V>(); Muls(tmp,low,aLow,count); PipeBarrier<PIPE_V>();
        Add(error,error,tmp,count); PipeBarrier<PIPE_V>();
        AccumulateDoubleVector(prod,count); AccumulateDoubleVector(error,count);
    }
    __aicore__ inline float ReadInput(uint64_t at) {
        if constexpr(IsSameType<T,float>::value) return input.GetValue(at);
        else { union {uint32_t bits;float value;} x; x.bits=(uint32_t)inputBits.GetValue(at)<<16;return x.value; }
    }
    __aicore__ inline void LoadRow(LocalTensor<float> row,GlobalTensor<float> tensor,uint64_t at,uint32_t count) {
        DataCopyPad(row,tensor[at],DataCopyExtParams{1,count*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void StoreResult(uint64_t at,uint32_t count) {
        if constexpr(IsSameType<T,float>::value) DataCopyPad(output[at],u,DataCopyExtParams{1,count*4,0,0,0});
        else {auto raw=rawBuf.Get<T>();Cast(raw,u,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_ALL>();DataCopyPad(output[at],raw,DataCopyExtParams{1,count*sizeof(T),0,0,0});}
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline bool FullRank(uint32_t batch,GlobalTensor<float> norms) {
        DataCopy(norm,norms[(uint64_t)batch*k*8],k*8);PipeBarrier<PIPE_ALL>();
        float largest=0;
        for(uint32_t j=0;j<k;++j) {float a=norm.GetValue(j*8);if(a>largest)largest=a;}
        float threshold=t.rtol*Root(largest);if(t.atol>threshold)threshold=t.atol;
        for(uint32_t j=0;j<k;++j) if(!(norm.GetValue(j*8)>threshold*threshold))return false;
        return true;
    }
    __aicore__ inline void RefineInverse(GM_ADDR workspace,GlobalTensor<float> norms,GlobalTensor<float> estimates,uint32_t completedMask,uint32_t knownFullRankMask=0) {
        for(uint32_t step=0;step<2;++step) {
            for(uint32_t task=GetBlockIdx();task<t.batches*k;task+=CoreCount()) {
                uint32_t batch=task/k,row=task%k;
                if(completedMask && (completedMask&(1u<<batch)))continue;
                if(knownFullRankMask ? !(knownFullRankMask&(1u<<batch)) : !FullRank(batch,norms))continue;
                GlobalTensor<float> residual;
                residual.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)batch*k*L);
                uint64_t base=(uint64_t)batch*k*k;
                Duplicate(u,0.0f,dp);Duplicate(q,0.0f,dp);PipeBarrier<PIPE_ALL>();
                u.SetValue(row,1.0f);PipeBarrier<PIPE_ALL>();
                for(uint32_t j=0;j<k;++j) {
                    LoadRow(p,estimates,base+(uint64_t)j*k,k);
                    AccumulateExactProduct(p,-ReadInput(base+(uint64_t)row*k+j),k);
                }
                DataCopyPad(residual[(uint64_t)row*k],u,DataCopyExtParams{1,k*4,0,0,0});PipeBarrier<PIPE_ALL>();
            }
            SyncAll<true>();
            for(uint32_t task=GetBlockIdx();task<t.batches*k;task+=CoreCount()) {
                uint32_t batch=task/k,row=task%k;
                if(completedMask && (completedMask&(1u<<batch)))continue;
                bool refine=knownFullRankMask ? (knownFullRankMask&(1u<<batch))!=0 : FullRank(batch,norms);
                uint64_t at=(uint64_t)batch*k*k+(uint64_t)row*k;
                LoadRow(norm,estimates,at,k);
                DataCopy(u,norm,dp);Duplicate(q,0.0f,dp);PipeBarrier<PIPE_ALL>();
                if(refine) {
                    GlobalTensor<float> residual;
                    residual.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)batch*k*L);
                    for(uint32_t j=0;j<k;++j) {
                        LoadRow(p,residual,(uint64_t)j*k,k);
                        AccumulateDoubleOutput(p,norm.GetValue(j),k);
                    }
                }
                DataCopyPad(estimates[at],u,DataCopyExtParams{1,k*4,0,0,0});PipeBarrier<PIPE_ALL>();
                if(step==1)StoreResult(at,k);
            }
            SyncAll<true>();
        }
    }
    __aicore__ inline LocalTensor<float> Column(uint32_t j, LocalTensor<float> tmp) {
        if(t.resident) return matrix[j*L];
        DataCopy(tmp,work[(uint64_t)j*L],L); PipeBarrier<PIPE_ALL>();
        return tmp;
    }
    __aicore__ inline void Store(uint32_t j, LocalTensor<float> col) {
        if(!t.resident) { DataCopy(work[(uint64_t)j*L],col,L); PipeBarrier<PIPE_ALL>(); }
    }
    // Try an inverse only when a residual/norm certificate proves no singular
    // value can be removed by the requested threshold. Otherwise use Jacobi.
    template<uint32_t K=0,bool noPivot=false,bool prepareOnly=false>
    __aicore__ inline bool TrySmallInverse(uint32_t batch) {
        const uint32_t k=K?K:t.columns,dp=(k+7)/8*8,L=2*dp+((K==64 && IsSameType<T,float>::value)?8:0);
        const uint64_t base=(uint64_t)batch*k*k;
        constexpr bool residentIO=K>0 && IsSameType<T,float>::value;
        auto originalMatrix=matrix[k*L];
        float aF2=0.0f;
        auto columnOffsets=smallIndexBuf.Get<uint32_t>();
        auto weights=smallProductBuf.Get<float>(),updates=smallTransposeBuf.Get<float>();
        for(uint32_t i=0;i<dp;++i)columnOffsets.SetValue(i,(i<k?i*L:k)*4);
        if constexpr(prepareOnly)Duplicate(columnOffsets[k*dp],uint32_t(0),dp);
        PipeBarrier<PIPE_ALL>();
        Duplicate(matrix,0.0f,k*L);PipeBarrier<PIPE_ALL>();
        if constexpr(residentIO) {
            if constexpr(K==15) {
                // Pack the contiguous 15x15 input into 16-float UB rows.
                DataCopyPad(updates,input[base],DataCopyExtParams{1,225*4,0,0,0},DataCopyPadExtParams<float>{true,0,7,0.0f});
                auto linear=smallIndexBuf.Get<int32_t>()[k*dp];
                auto packedOffsets=columnOffsets.template ReinterpretCast<int32_t>();
                CreateVecIndex(linear,int32_t(0),240);PipeBarrier<PIPE_V>();
                ShiftRight(packedOffsets,linear,int32_t(4),240);PipeBarrier<PIPE_V>();
                Sub(packedOffsets,linear,packedOffsets,240);PipeBarrier<PIPE_V>();
                ShiftLeft(packedOffsets,packedOffsets,int32_t(2),240);PipeBarrier<PIPE_ALL>();
                Gather(originalMatrix,updates,columnOffsets,0,240);PipeBarrier<PIPE_ALL>();
                for(uint32_t i=0;i<dp;++i)columnOffsets.SetValue(i,(i<k?i*L:k)*4);
                PipeBarrier<PIPE_ALL>();
            } else {
                DataCopyPad(originalMatrix,input[base],DataCopyExtParams{(uint16_t)k,k*4,0,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(dp-k),0.0f});
                PipeBarrier<PIPE_ALL>();
            }
            DataCopy(matrix,originalMatrix,DataCopyParams{(uint16_t)k,(uint16_t)(dp/8),0,(uint16_t)((L-dp)/8)});PipeBarrier<PIPE_ALL>();
        } else if constexpr(IsSameType<T,float>::value) {
            DataCopyPad(matrix,input[base],DataCopyExtParams{(uint16_t)k,k*4,0,dp/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(dp-k),0.0f});
            PipeBarrier<PIPE_ALL>();
        } else for(uint32_t i=0;i<k;++i)for(uint32_t j=0;j<k;++j)
            matrix.SetValue(i*L+j,ReadInput(base+(uint64_t)i*k+j));
        for(uint32_t i=0;i<k;++i)matrix.SetValue(i*L+dp+i,1.0f);
        PipeBarrier<PIPE_ALL>();
        BinaryRepeatParams squareRows{1,1,1,(uint8_t)(dp/8),(uint8_t)(L/8),(uint8_t)(L/8)};
        Duplicate(weights,0.0f,k*dp);PipeBarrier<PIPE_V>();
        Mul(weights,matrix,matrix,(uint64_t)k,(uint8_t)k,squareRows);PipeBarrier<PIPE_V>();
        ReduceSum(scalar,weights,reduce,k*dp);PipeBarrier<PIPE_ALL>();aF2=scalar.GetValue(0);
        if constexpr(K==64 && IsSameType<T,float>::value) {
            // Padding stays zero while the two useful halves are overwritten.
            Duplicate(updates,0.0f,k*L);PipeBarrier<PIPE_V>();
        }
        uint64_t usedRows=0;
        uint32_t pivotRows[K?K:1];
        for(uint32_t j=0;j<k;++j) {
            Gather(p,matrix[j],columnOffsets,0,k);if constexpr(K) {PipeBarrier<PIPE_V>();}else {PipeBarrier<PIPE_ALL>();}
            float best;uint32_t pivot=j;
            if constexpr(noPivot) {
                best=p.GetValue(j);if(best<0.0f)best=-best;
            } else {
            Abs(v,p,k);PipeBarrier<PIPE_V>();
            if constexpr(K) {
                uint64_t pivotMask[2]={(~uint64_t(0)>>(64-K))&~usedRows,0};
                WholeReduceMax(scalar,v,pivotMask,1,1,1,dp/8);
            } else {
                if(j>0){Duplicate(v,0.0f,j);PipeBarrier<PIPE_V>();}
                WholeReduceMax(scalar,v,k,1,1,1,dp/8);
            }
            PipeBarrier<PIPE_ALL>();
            pivot=scalar.template ReinterpretCast<uint32_t>().GetValue(1);
            if constexpr(K==50)best=p.GetValue(pivot);
            else best=scalar.GetValue(0);
            }
            if(!(best*best>aF2*(noPivot?1e-12f:1e-16f)))return false;
            auto row=matrix[(K?pivot:j)*L];
            float oldDiagonal=p.GetValue(j);
            if constexpr(K && !noPivot) {
                usedRows|=uint64_t(1)<<pivot;
                if constexpr(!prepareOnly)pivotRows[j]=pivot;
            } else if(pivot!=j) {
                DataCopy(u,row,L);DataCopy(row,matrix[pivot*L],L);PipeBarrier<PIPE_ALL>();
                DataCopy(matrix[pivot*L],u,L);PipeBarrier<PIPE_ALL>();
            }
            if constexpr(prepareOnly)columnOffsets.SetValue(k*dp+j,pivot*L*4);
            // Gather already contains the exact selected matrix entry.
            float reciprocal=1.0f/((K==50 && !noPivot)?best:row.GetValue(j));
            auto scaledRow=K?u:row;
            Muls(scaledRow,row,reciprocal,L);if constexpr(K) {PipeBarrier<PIPE_V>();}else {PipeBarrier<PIPE_ALL>();}
            if(k>=8) {
                if constexpr(!K) {if(pivot!=j)p.SetValue(pivot,oldDiagonal);p.SetValue(j,0.0f);}
                if constexpr(!K)PipeBarrier<PIPE_ALL>();
                Brcb(weights,p,(k+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                if constexpr(K) {
                    BinaryRepeatParams outer{1,0,1,(uint8_t)(L/8),1,0};
                    for(uint32_t off=0;off<2*dp;off+=64)
                        Mul(updates[off],weights,scaledRow[off],(uint64_t)(2*dp-off<64?2*dp-off:64),(uint8_t)k,outer);
                    PipeBarrier<PIPE_V>();
                    Sub(matrix,matrix,updates,k*L);PipeBarrier<PIPE_V>();
                    if constexpr(K) {
                        // Restore the normalized pivot row after the outer
                        // update, avoiding a scalar zero in the weights.
                        DataCopy(row,scaledRow,L);PipeBarrier<PIPE_V>();
                    }
                } else {
                BinaryRepeatParams outer{1,0,1,(uint8_t)(dp/8),1,0};
                BinaryRepeatParams subtract{1,1,1,(uint8_t)(L/8),(uint8_t)(L/8),(uint8_t)(dp/8)};
                Mul(updates,weights,row,(uint64_t)dp,(uint8_t)k,outer);PipeBarrier<PIPE_V>();
                Sub(matrix,matrix,updates,(uint64_t)dp,(uint8_t)k,subtract);PipeBarrier<PIPE_V>();
                Mul(updates,weights,row[dp],(uint64_t)dp,(uint8_t)k,outer);PipeBarrier<PIPE_V>();
                Sub(matrix[dp],matrix[dp],updates,(uint64_t)dp,(uint8_t)k,subtract);if constexpr(K) {PipeBarrier<PIPE_V>();}else {PipeBarrier<PIPE_ALL>();}
                }
            } else {
                for(uint32_t i=0;i<k;++i)if(i!=j) {
                    auto other=matrix[i*L];float weight=-other.GetValue(j);Axpy(other,row,weight,L);
                }
                PipeBarrier<PIPE_ALL>();
            }
        }
        if constexpr(K && !noPivot && !prepareOnly) {
            PipeBarrier<PIPE_ALL>();
            for(uint32_t i=0;i<k;++i)DataCopy(updates[i*dp],matrix[pivotRows[i]*L+dp],dp);
            PipeBarrier<PIPE_V>();
            DataCopy(matrix[dp],updates,DataCopyParams{(uint16_t)k,(uint16_t)(dp/8),0,(uint16_t)((L-dp)/8)});PipeBarrier<PIPE_ALL>();
        }
        Duplicate(weights,0.0f,k*dp);PipeBarrier<PIPE_V>();
        Mul(weights,matrix[dp],matrix[dp],(uint64_t)k,(uint8_t)k,squareRows);PipeBarrier<PIPE_V>();
        ReduceSum(scalar,weights,reduce,k*dp);PipeBarrier<PIPE_ALL>();float xF2=scalar.GetValue(0);
        float threshold=t.rtol*Root(aF2);if(t.atol>threshold)threshold=t.atol;
        // Frobenius norms bound spectral norms. With ||I-AX||F < 1/4,
        // sigma_min(A) >= (1-||I-AX||2)/||X||2 > 0.75/||X||F.
        float xNorm=Root(xF2);
        if(!(threshold*xNorm<0.5f))return false;
        if constexpr(noPivot)if(xF2>1e6f)return false;
        if constexpr(prepareOnly)return xF2<=1e6f;
        // Large inverse magnitudes amplify the rounding floor of a plain FP32
        // dot product. Use compensated products and two refinements here.
        if(xF2>1e6f) {
            for(uint32_t step=0;step<2;++step) {
                float rF2=0.0f;
                for(uint32_t i=0;i<k;++i) {
                    auto original=norm[5*dp];
                    if constexpr(IsSameType<T,float>::value) {
                        DataCopyPad(original,input[base+(uint64_t)i*k],DataCopyExtParams{1,k*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                        PipeBarrier<PIPE_ALL>();
                    } else for(uint32_t j=0;j<k;++j)original.SetValue(j,ReadInput(base+(uint64_t)i*k+j));
                    Duplicate(u,0.0f,dp);Duplicate(q,0.0f,dp);PipeBarrier<PIPE_ALL>();
                    u.SetValue(i,1.0f);PipeBarrier<PIPE_ALL>();
                    for(uint32_t j=0;j<k;++j)
                        AccumulateExactProduct(matrix[j*L+dp],-original.GetValue(j),k);
                    rF2+=Dot(u,u);DataCopy(matrix[i*L],u,dp);PipeBarrier<PIPE_ALL>();
                }
                if(!(rF2<0.0625f))return false;
                for(uint32_t i=0;i<k;++i) {
                    DataCopy(u,matrix[i*L+dp],dp);Duplicate(q,0.0f,dp);PipeBarrier<PIPE_ALL>();
                    for(uint32_t j=0;j<k;++j)
                        AccumulateDoubleOutput(matrix[j*L],matrix.GetValue(i*L+dp+j),k);
                    DataCopy(matrix[i*L+dp],u,dp);PipeBarrier<PIPE_ALL>();
                    if(step==1)StoreResult(base+(uint64_t)i*k,k);
                }
            }
            return true;
        }
        auto xt=smallTransposeBuf.Get<float>(),products=smallProductBuf.Get<float>();
        auto offsets=smallIndexBuf.Get<uint32_t>();
        if constexpr((K==15 || K==16) && residentIO) {
            // The elimination leaves the initial row offsets intact.
            auto steps=p.template ReinterpretCast<int32_t>();
            auto expanded=products.template ReinterpretCast<int32_t>();
            CreateVecIndex(steps,int32_t(0),dp);PipeBarrier<PIPE_V>();
            ShiftLeft(steps,steps,int32_t(2),dp);PipeBarrier<PIPE_V>();
            Brcb(expanded,steps,dp/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            auto indices=offsets.template ReinterpretCast<int32_t>();
            Add(indices,indices,expanded,(uint64_t)dp,(uint8_t)k,
                BinaryRepeatParams{1,1,0,(uint8_t)(dp/8),0,1});PipeBarrier<PIPE_ALL>();
        } else {
            for(uint32_t i=0;i<dp;++i)offsets.SetValue(i,(i<k?i*L:k)*4);
            PipeBarrier<PIPE_ALL>();
            auto signedOffsets=smallIndexBuf.Get<int32_t>();
            for(uint32_t j=1;j<k;++j)Adds(signedOffsets[j*dp],signedOffsets,(int32_t)(j*4),dp);
            PipeBarrier<PIPE_ALL>();
        }
        Gather(xt,matrix[dp],offsets,0,k*dp);PipeBarrier<PIPE_ALL>();
        BinaryRepeatParams broadcast{1,1,1,(uint8_t)(dp/8),(uint8_t)(dp/8),0};
        float rF2=0.0f;
        if constexpr((K==15 || K==16) && residentIO) {
            // Keep each dot product intact; batch the output-row reductions
            // after the broadcasts. Split 256 dots to fit the repeat limit.
            if constexpr(K==15){Duplicate(xt[k*dp],0.0f,dp);PipeBarrier<PIPE_V>();}
            for(uint32_t i=0;i<k;++i)
                Mul(products[i*dp*dp],xt,originalMatrix[i*dp],(uint64_t)k,(uint8_t)dp,broadcast);
            PipeBarrier<PIPE_V>();
            if constexpr(K==16) {
                WholeReduceSum(xt,products,k,128,1,1,dp/8);
                WholeReduceSum(xt[128],products[128*dp],k,128,1,1,dp/8);PipeBarrier<PIPE_V>();
            } else {WholeReduceSum(xt,products,k,k*dp,1,1,dp/8);PipeBarrier<PIPE_V>();}
            Muls(xt,xt,-1.0f,k*dp);PipeBarrier<PIPE_V>();
            DataCopy(matrix,xt,DataCopyParams{(uint16_t)k,(uint16_t)(dp/8),0,(uint16_t)((L-dp)/8)});PipeBarrier<PIPE_V>();
        } else for(uint32_t i=0;i<k;++i) {
            auto original=residentIO?originalMatrix[i*dp]:p;
            if constexpr(!residentIO) {
                if constexpr(IsSameType<T,float>::value) {
                    DataCopyPad(original,input[base+(uint64_t)i*k],DataCopyExtParams{1,k*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                    PipeBarrier<PIPE_ALL>();
                } else for(uint32_t j=0;j<k;++j)original.SetValue(j,ReadInput(base+(uint64_t)i*k+j));
            }
            Mul(products,xt,original,(uint64_t)k,(uint8_t)k,broadcast);PipeBarrier<PIPE_V>();
            if constexpr(residentIO) {
                WholeReduceSum(matrix[i*L],products,k,k,1,1,dp/8);PipeBarrier<PIPE_V>();
                Muls(matrix[i*L],matrix[i*L],-1.0f,k);PipeBarrier<PIPE_V>();
            } else {
                WholeReduceSum(u,products,k,k,1,1,dp/8);PipeBarrier<PIPE_V>();
                Muls(u,u,-1.0f,k);PipeBarrier<PIPE_ALL>();
                u.SetValue(i,u.GetValue(i)+1.0f);PipeBarrier<PIPE_ALL>();
                DataCopy(matrix[i*L],u,dp);PipeBarrier<PIPE_ALL>();
            }
        }
        if constexpr(residentIO) {
            PipeBarrier<PIPE_ALL>();
            for(uint32_t i=0;i<k;++i)matrix.SetValue(i*L+i,matrix.GetValue(i*L+i)+1.0f);
            PipeBarrier<PIPE_ALL>();
        }
        Duplicate(products,0.0f,k*dp);PipeBarrier<PIPE_V>();
        Mul(products,matrix,matrix,(uint64_t)k,(uint8_t)k,squareRows);PipeBarrier<PIPE_V>();
        ReduceSum(scalar,products,reduce,k*dp);PipeBarrier<PIPE_ALL>();rF2=scalar.GetValue(0);
        if(!(rF2<0.0625f))return false;
        if constexpr(noPivot)if(!(xNorm*rF2<(K==16?8e-5f:7.5e-6f)))return false;
        Gather(xt,matrix,offsets,0,k*dp);PipeBarrier<PIPE_ALL>();
        if constexpr((K==15 || K==16) && residentIO) {
            if constexpr(K==15){Duplicate(xt[k*dp],0.0f,dp);PipeBarrier<PIPE_V>();}
            for(uint32_t i=0;i<k;++i)
                Mul(products[i*dp*dp],xt,matrix[i*L+dp],(uint64_t)k,(uint8_t)dp,broadcast);
            PipeBarrier<PIPE_V>();
            if constexpr(K==16) {
                WholeReduceSum(originalMatrix,products,k,128,1,1,dp/8);
                WholeReduceSum(originalMatrix[128],products[128*dp],k,128,1,1,dp/8);PipeBarrier<PIPE_V>();
            } else {WholeReduceSum(originalMatrix,products,k,k*dp,1,1,dp/8);PipeBarrier<PIPE_V>();}
            Add(originalMatrix,originalMatrix,matrix[dp],(uint64_t)k,(uint8_t)k,
                BinaryRepeatParams{1,1,1,(uint8_t)(dp/8),(uint8_t)(dp/8),(uint8_t)(L/8)});PipeBarrier<PIPE_V>();
        } else for(uint32_t i=0;i<k;++i) {
            Mul(products,xt,matrix[i*L+dp],(uint64_t)k,(uint8_t)k,broadcast);PipeBarrier<PIPE_V>();
            WholeReduceSum(u,products,k,k,1,1,dp/8);PipeBarrier<PIPE_V>();
            Add(u,u,matrix[i*L+dp],k);if constexpr(residentIO) {PipeBarrier<PIPE_V>();}else {PipeBarrier<PIPE_ALL>();}
            if constexpr(residentIO) {DataCopy(originalMatrix[i*dp],u,dp);PipeBarrier<PIPE_V>();}
            else StoreResult(base+(uint64_t)i*k,k);
        }
        if constexpr(residentIO) {
            PipeBarrier<PIPE_ALL>();
            if constexpr(K==15) {
                // Compact UB row padding before the one contiguous GM store.
                auto linear=smallIndexBuf.Get<int32_t>()[k*dp];
                auto compact=offsets.template ReinterpretCast<int32_t>();
                // floor(n/15)=(n+1+floor(n/16))/16 for 0<=n<225.
                ShiftRight(compact,linear,int32_t(4),225);PipeBarrier<PIPE_V>();
                Add(compact,compact,linear,225);PipeBarrier<PIPE_V>();
                Adds(compact,compact,int32_t(1),225);PipeBarrier<PIPE_V>();
                ShiftRight(compact,compact,int32_t(4),225);PipeBarrier<PIPE_V>();
                Add(compact,compact,linear,225);PipeBarrier<PIPE_V>();
                ShiftLeft(compact,compact,int32_t(2),225);PipeBarrier<PIPE_V>();
                Gather(xt,originalMatrix,offsets,0,225);PipeBarrier<PIPE_ALL>();
                DataCopyPad(output[base],xt,DataCopyExtParams{1,225*4,0,0,0});PipeBarrier<PIPE_ALL>();
            } else {
                DataCopyPad(output[base],originalMatrix,DataCopyExtParams{(uint16_t)k,k*4,0,0,0});PipeBarrier<PIPE_ALL>();
            }
        }
        return true;
    }
    // Replicate the short elimination, then share residual/output rows across
    // the team. All cores reach the one reduction barrier, even on rejection.
    template<uint32_t K>
    __aicore__ inline void RunSmallTeam(GM_ADDR workspace) {
        constexpr uint32_t dp=(K+7)/8*8,L=2*dp+(K==64?8:0);
        uint32_t batch=GetBlockIdx()%t.batches,lane=GetBlockIdx()/t.batches;
        uint32_t lanes=CoreCount()/t.batches;
        uint64_t base=(uint64_t)batch*K*K;
        bool valid=TrySmallInverse<K,false,true>(batch);
        auto offsets=smallIndexBuf.Get<uint32_t>();
        auto rowOffsets=offsets[K*dp];
        auto xt=smallTransposeBuf.Get<float>(),products=smallProductBuf.Get<float>();
        auto original=matrix[K*L];
        GlobalTensor<float> residual,partial;
        residual.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)batch*K*dp);
        partial.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)t.batches*K*dp);
        BinaryRepeatParams broadcast{1,1,1,dp/8,dp/8,0};
        float rF2=0.0f;
        if(valid) {
            // Broadcast the physical row map and add column byte offsets.
            auto columnSteps=p.template ReinterpretCast<int32_t>();
            auto expanded=products.template ReinterpretCast<int32_t>();
            CreateVecIndex(columnSteps,int32_t(0),dp);PipeBarrier<PIPE_V>();
            ShiftLeft(columnSteps,columnSteps,int32_t(2),dp);PipeBarrier<PIPE_V>();
            Brcb(expanded,columnSteps,dp/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            Add(offsets.template ReinterpretCast<int32_t>(),rowOffsets.template ReinterpretCast<int32_t>(),expanded,
                (uint64_t)dp,(uint8_t)K,BinaryRepeatParams{1,1,0,dp/8,0,1});PipeBarrier<PIPE_ALL>();
            Gather(xt,matrix[dp],offsets,0,K*dp);PipeBarrier<PIPE_ALL>();
            uint32_t residualRows=0;
            if constexpr(K==50){Duplicate(u,0.0f,dp);PipeBarrier<PIPE_V>();}
            for(uint32_t i=lane;i<K;i+=lanes) {
                Mul(products,xt,original[i*dp],(uint64_t)K,(uint8_t)K,broadcast);PipeBarrier<PIPE_V>();
                WholeReduceSum(u,products,K,K,1,1,dp/8);PipeBarrier<PIPE_V>();
                Muls(u,u,-1.0f,K);PipeBarrier<PIPE_ALL>();
                u.SetValue(i,u.GetValue(i)+1.0f);PipeBarrier<PIPE_ALL>();
                if constexpr(K==50) {
                    // These packed rows only overwrite input rows already read.
                    DataCopy(original[residualRows*dp],u,dp);++residualRows;
                } else rF2+=Dot(u,u);
                DataCopy(residual[rowOffsets.GetValue(i)/(L*4)*dp],u,dp);PipeBarrier<PIPE_ALL>();
            }
            if constexpr(K==50) {
                Mul(products,original,original,residualRows*dp);PipeBarrier<PIPE_V>();
                ReduceSum(scalar,products,reduce,residualRows*dp);PipeBarrier<PIPE_ALL>();
                rF2=scalar.GetValue(0);
            }
        }
        Duplicate(scalar,0.0f,8);PipeBarrier<PIPE_ALL>();
        scalar.SetValue(0,rF2);scalar.SetValue(1,valid?0.0f:1.0f);PipeBarrier<PIPE_ALL>();
        DataCopy(partial[GetBlockIdx()*8],scalar,8);PipeBarrier<PIPE_ALL>();SyncAll<true>();
        DataCopy(norm,partial,CoreCount()*8);PipeBarrier<PIPE_ALL>();
        float sum=0.0f,bad=0.0f;
        for(uint32_t core=batch;core<CoreCount();core+=t.batches) {
            sum+=norm.GetValue(core*8);bad+=norm.GetValue(core*8+1);
        }
        valid=bad==0.0f && sum<0.0625f;
        if(valid) {
            DataCopy(matrix,residual,DataCopyParams{K,dp/8,0,(L-dp)/8});PipeBarrier<PIPE_ALL>();
            Gather(xt,matrix,offsets,0,K*dp);PipeBarrier<PIPE_ALL>();
            for(uint32_t i=lane;i<K;i+=lanes) {
                auto inverseRow=matrix[rowOffsets.GetValue(i)/4+dp];
                Mul(products,xt,inverseRow,(uint64_t)K,(uint8_t)K,broadcast);PipeBarrier<PIPE_V>();
                WholeReduceSum(u,products,K,K,1,1,dp/8);PipeBarrier<PIPE_V>();
                Add(u,u,inverseRow,K);PipeBarrier<PIPE_ALL>();
                StoreResult(base+(uint64_t)i*K,K);
            }
        } else if(lane==0)RunBatch(batch,workspace);
    }
    __aicore__ inline void RunBatch(uint32_t batch, GM_ADDR workspace) {
        if(t.m==t.n && k<=64 && t.resident && (k==16?((IsSameType<T,float>::value && TrySmallInverse<16,true>(batch)) || TrySmallInverse<16>(batch)):(k==15?((IsSameType<T,float>::value && TrySmallInverse<15,true>(batch)) || TrySmallInverse<15>(batch)):(k==50?TrySmallInverse<50>(batch):(k==64?TrySmallInverse<64>(batch):TrySmallInverse<0>(batch))))))return;
        const uint64_t base=(uint64_t)batch*t.m*t.n;
        if(!t.resident) work.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)batch*k*L);
        float maxNorm=0;
        for(uint32_t j=0;j<k;++j) {
            auto col=t.resident?matrix[j*L]:p;
            Duplicate(col,0.0f,L); PipeBarrier<PIPE_ALL>();
            for(uint32_t i=0;i<d;++i) {
                const uint64_t at=t.m>=t.n?(uint64_t)i*t.n+j:(uint64_t)j*t.n+i;
                if constexpr(IsSameType<T,float>::value) col.SetValue(i,input.GetValue(base+at));
                else {
                    union { uint32_t bits; float value; } x;
                    x.bits=(uint32_t)inputBits.GetValue(base+at)<<16;
                    col.SetValue(i,x.value);
                }
            }
            col.SetValue(dp+j,1.0f); PipeBarrier<PIPE_ALL>();
            float a=Dot(col,col); if(a>maxNorm) maxNorm=a;
            Store(j,col);
        }
        for(uint32_t sweep=0;sweep<16;++sweep) {
            uint32_t changed=0;
            for(uint32_t jp=0;jp<k;++jp) {
                auto cp=Column(jp,p);
                float a=Dot(cp,cp);
                for(uint32_t jq=jp+1;jq<k;++jq) {
                    auto cq=Column(jq,q);
                    float b=Dot(cq,cq), g=Dot(cp,cq);
                    if(a<=maxNorm*1e-14f || b<=maxNorm*1e-14f || g*g<=1e-16f*a*b) continue;
                    float tau=(b-a)/(2.0f*g);
                    float at=tau<0?-tau:tau;
                    float tangent=(tau<0?-1.0f:1.0f)/(at+Root(1.0f+tau*tau));
                    float cosine=1.0f/Root(1.0f+tangent*tangent), sine=cosine*tangent;
                    Muls(u,cp,cosine,L); Muls(v,cq,cosine,L); PipeBarrier<PIPE_V>();
                    Axpy(u,cq,-sine,L); Axpy(v,cp,sine,L); PipeBarrier<PIPE_V>();
                    DataCopy(cp,u,L); DataCopy(cq,v,L); PipeBarrier<PIPE_ALL>();
                    Store(jq,cq);
                    a=Dot(cp,cp); ++changed;
                }
                Store(jp,cp);
            }
            if(changed==0) break;
        }
        float largest=0;
        for(uint32_t j=0;j<k;++j) {
            auto col=Column(j,p); float a=Dot(col,col);
            norm.SetValue(j,a); if(a>largest) largest=a;
        }
        float threshold=t.rtol*Root(largest);
        if(t.atol>threshold) threshold=t.atol;
        for(uint32_t j=0;j<k;++j) {
            float a=norm.GetValue(j);
            norm.SetValue(j,a>threshold*threshold?1.0f/a:0.0f);
        }
        const uint32_t outRows=t.n, outCols=t.m;
        for(uint32_t i=0;i<outRows;++i) {
            Duplicate(u,0.0f,(outCols+7)/8*8); Duplicate(q,0.0f,(outCols+7)/8*8); PipeBarrier<PIPE_ALL>();
            for(uint32_t j=0;j<k;++j) {
                float inv=norm.GetValue(j); if(inv==0) continue;
                auto col=Column(j,p);
                float weight=col.GetValue(t.m>=t.n?dp+i:i)*inv;
                AccumulateOutput(t.m>=t.n?col:col[dp],weight,outCols);
            }
            if constexpr(IsSameType<T,float>::value) {
                DataCopyPad(output[base+(uint64_t)i*outCols],u,DataCopyExtParams{1,outCols*4,0,0,0});
            } else {
                auto raw=rawBuf.Get<T>();
                Cast(raw,u,RoundMode::CAST_RINT,outCols); PipeBarrier<PIPE_ALL>();
                DataCopyPad(output[base+(uint64_t)i*outCols],raw,DataCopyExtParams{1,outCols*sizeof(T),0,0,0});
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
    // Batched simultaneous dot products. Dekker products and a compensated
    // pairwise reduction retain the residual accuracy without scalar AXPY loops.
    template<uint32_t K>
    __aicore__ inline void PreciseRowLarge(GlobalTensor<float> right,uint64_t base,
            LocalTensor<float> left,LocalTensor<float> result,uint32_t row,bool residual) {
        constexpr uint32_t tileSize=dotTile;
        auto rhs=matrix,high=matrix[tileSize],low=matrix[2*tileSize];
        auto temp=matrix[3*tileSize],xh=matrix[4*tileSize],xl=matrix[5*tileSize];
        auto ah=norm[6*256],al=norm[7*256];
        if constexpr(K<256){Duplicate(left[K],0.0f,256-K);PipeBarrier<PIPE_V>();}
        auto tile=smallTransposeBuf.Get<float>();
        auto indices=smallIndexBuf.Get<uint32_t>();
        auto bitmask=p.template ReinterpretCast<int32_t>();
        Duplicate(bitmask,(int32_t)0xfffff000u,64);PipeBarrier<PIPE_V>();
        And(ah.template ReinterpretCast<uint16_t>(),left.template ReinterpretCast<uint16_t>(),
            bitmask.template ReinterpretCast<uint16_t>(),(uint64_t)128,4,BinaryRepeatParams{1,1,1,8,8,0});PipeBarrier<PIPE_V>();
        Sub(al,left,ah,256);PipeBarrier<PIPE_V>();
        if constexpr(K<256){Duplicate(tile,0.0f,tileSize);PipeBarrier<PIPE_ALL>();}
        for(uint32_t col=0;col<K;col+=dotWidth) {
            uint32_t columns=K==256?dotWidth:(K-col<dotWidth?K-col:dotWidth);
            DataCopyPad(tile,right[base+col],DataCopyExtParams{K,columns*4,(K-columns)*4,(dotWidth-columns)/8,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
            Gather(rhs,tile,indices,0,tileSize);PipeBarrier<PIPE_V>();

            And(xh.template ReinterpretCast<uint16_t>(),rhs.template ReinterpretCast<uint16_t>(),
                bitmask.template ReinterpretCast<uint16_t>(),(uint64_t)128,dotWidth*4,BinaryRepeatParams{1,1,1,8,8,0});PipeBarrier<PIPE_V>();
            Sub(xl,rhs,xh,tileSize);PipeBarrier<PIPE_V>();
            BinaryRepeatParams broadcast{1,1,1,32,32,0};
            for(uint32_t off=0;off<256;off+=64) {
                Mul(high[off],rhs[off],left[off],(uint64_t)64,dotWidth,broadcast);
                Mul(low[off],xh[off],ah[off],(uint64_t)64,dotWidth,broadcast);PipeBarrier<PIPE_V>();
                Sub(low[off],low[off],high[off],(uint64_t)64,dotWidth,BinaryRepeatParams{1,1,1,32,32,32});
                Mul(temp[off],xl[off],ah[off],(uint64_t)64,dotWidth,broadcast);PipeBarrier<PIPE_V>();
                Add(low[off],low[off],temp[off],(uint64_t)64,dotWidth,BinaryRepeatParams{1,1,1,32,32,32});PipeBarrier<PIPE_V>();
                Mul(temp[off],xh[off],al[off],(uint64_t)64,dotWidth,broadcast);PipeBarrier<PIPE_V>();
                Add(low[off],low[off],temp[off],(uint64_t)64,dotWidth,BinaryRepeatParams{1,1,1,32,32,32});PipeBarrier<PIPE_V>();
                Mul(temp[off],xl[off],al[off],(uint64_t)64,dotWidth,broadcast);PipeBarrier<PIPE_V>();
                Add(low[off],low[off],temp[off],(uint64_t)64,dotWidth,BinaryRepeatParams{1,1,1,32,32,32});PipeBarrier<PIPE_V>();
            }
            BinaryRepeatParams eight{1,1,1,32,32,32};
            for(uint32_t half=128;half>=8;half/=2) {
                for(uint32_t off=0;off<half;off+=64) {
                    uint64_t lanes=half<64?half:64;
                    Add(temp[off],high[off],high[half+off],lanes,dotWidth,eight);PipeBarrier<PIPE_V>();
                    Sub(xh[off],temp[off],high[off],lanes,dotWidth,eight);PipeBarrier<PIPE_V>();
                    Sub(xl[off],temp[off],xh[off],lanes,dotWidth,eight);PipeBarrier<PIPE_V>();
                    Sub(xl[off],high[off],xl[off],lanes,dotWidth,eight);
                    Sub(rhs[off],high[half+off],xh[off],lanes,dotWidth,eight);PipeBarrier<PIPE_V>();
                    Add(xl[off],xl[off],rhs[off],lanes,dotWidth,eight);
                    Add(low[off],low[off],low[half+off],lanes,dotWidth,eight);PipeBarrier<PIPE_V>();
                    Add(low[off],low[off],xl[off],lanes,dotWidth,eight);
                    Adds(high[off],temp[off],0.0f,lanes,dotWidth,UnaryRepeatParams{1,1,32,32});PipeBarrier<PIPE_V>();
                }
            }
            // Compact the final eight terms so all vector operands remain
            // 32-byte aligned, including the last three reduction levels.
            Gather(rhs,high,indices[dotTile],0,dotTail);
            Gather(temp,low,indices[dotTile],0,dotTail);PipeBarrier<PIPE_V>();
            DataCopy(high,rhs,dotTail);DataCopy(low,temp,dotTail);PipeBarrier<PIPE_V>();
            auto z=matrix[6*tileSize],error=matrix[6*tileSize+dotTail];
            for(uint32_t level=0;level<3;++level) {
                Gather(rhs,high,indices[dotTile+dotTail+level*dotTail],0,dotTail);
                Gather(temp,low,indices[dotTile+dotTail+level*dotTail],0,dotTail);PipeBarrier<PIPE_V>();
                Add(xh,high,rhs,dotTail);PipeBarrier<PIPE_V>();
                Sub(xl,xh,high,dotTail);PipeBarrier<PIPE_V>();
                Sub(z,xh,xl,dotTail);PipeBarrier<PIPE_V>();
                Sub(z,high,z,dotTail);Sub(error,rhs,xl,dotTail);PipeBarrier<PIPE_V>();
                Add(z,z,error,dotTail);Add(low,low,temp,dotTail);PipeBarrier<PIPE_V>();
                Add(low,low,z,dotTail);DataCopy(high,xh,dotTail);PipeBarrier<PIPE_V>();
            }
            {
                PipeBarrier<PIPE_ALL>();
                for(uint32_t j=0;j<columns;++j) {
                    float value;
                    if(residual)value=((row==col+j?1.0f:0.0f)-high.GetValue(j*8))-low.GetValue(j*8);
                    else value=left.GetValue(col+j)+(high.GetValue(j*8)+low.GetValue(j*8));
                    result.SetValue(col+j,value);
                }
                PipeBarrier<PIPE_ALL>();
            }
        }

    }
    __aicore__ inline uint32_t RefineFastInverse(GM_ADDR workspace,GlobalTensor<float> norms,GlobalTensor<float> estimates,const float *inverseNorm2,uint32_t acceptedMask) {
        const bool packed=(k==256) && t.batches<=4 && (t.batches*(k/2)+CoreCount()-1)/CoreCount()<=16;
        if(packed && acceptedMask) {
            auto indices=smallIndexBuf.Get<uint32_t>();
            for(uint32_t j=0;j<dotWidth;++j)for(uint32_t i=0;i<256;++i)indices.SetValue(j*256+i,(i*dotWidth+j)*4);
            for(uint32_t j=0;j<dotTail;++j) {
                indices.SetValue(dotTile+j,((j/8)*256+j%8)*4);
                for(uint32_t level=0;level<3;++level)indices.SetValue(dotTile+dotTail+level*dotTail+j,(j^(4u>>level))*4);
            }
            PipeBarrier<PIPE_ALL>();
        }
        uint32_t activeMask=acceptedMask,completedMask=0;
        float inverseBound[4]={0,0,0,0};
        for(uint32_t step=0;step<4 && activeMask;++step) {
            float residualNorms[4]={0,0,0,0};
            for(uint32_t task=GetBlockIdx();task<t.batches*k;task+=CoreCount()) {
                uint32_t batch=task/k,row=task%k;
                if(!(activeMask&(1u<<batch)))continue;
                GlobalTensor<float> residual;
                residual.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)batch*k*L);
                uint64_t base=(uint64_t)batch*k*k;
                auto original=norm[5*dp];
                if constexpr(IsSameType<T,float>::value) {
                    DataCopyPad(original,input[base+(uint64_t)row*k],DataCopyExtParams{1,k*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                    PipeBarrier<PIPE_ALL>();
                } else for(uint32_t j=0;j<k;++j)original.SetValue(j,ReadInput(base+(uint64_t)row*k+j));
                if(packed) {
                    PreciseRowLarge<256>(estimates,base,original,u,row,true);
                }
                else {
                Duplicate(u,0.0f,dp);Duplicate(q,0.0f,dp);PipeBarrier<PIPE_ALL>();
                u.SetValue(row,1.0f);PipeBarrier<PIPE_ALL>();
                for(uint32_t j=0;j<k;++j) {
                    LoadRow(p,estimates,base+(uint64_t)j*k,k);
                    AccumulateExactProduct(p,-original.GetValue(j),k);
                }
                }
                residualNorms[batch]+=Dot(u,u);
                DataCopyPad(residual[(uint64_t)row*k],u,DataCopyExtParams{1,k*4,0,0,0});PipeBarrier<PIPE_ALL>();
            }
            Duplicate(scalar,0.0f,8);PipeBarrier<PIPE_ALL>();
            for(uint32_t batch=0;batch<t.batches;++batch)scalar.SetValue(batch,residualNorms[batch]);
            PipeBarrier<PIPE_ALL>();DataCopy(norms[GetBlockIdx()*8],scalar,8);PipeBarrier<PIPE_ALL>();SyncAll<true>();
            DataCopy(norm,norms,CoreCount()*8);PipeBarrier<PIPE_ALL>();
            uint32_t finishMask=0;
            for(uint32_t batch=0;batch<t.batches;++batch) {
                if(!(activeMask&(1u<<batch)))continue;
                float sum=0;
                for(uint32_t core=0;core<CoreCount();++core)sum+=norm.GetValue(core*8+batch);
                if(!(sum<0.0625f)){activeMask&=~(1u<<batch);continue;}
                if(step==0)inverseBound[batch]=Root(inverseNorm2[batch])/(1.0f-Root(sum));
                // The large packed paths permit a 5e-5 absolute error bound before
                // output rounding, still below the 1e-4 output tolerance.
                if(inverseBound[batch]*sum<((k==200 || k==256)?5e-5f:1e-5f))finishMask|=1u<<batch;
            }
            for(uint32_t task=GetBlockIdx();task<t.batches*k;task+=CoreCount()) {
                uint32_t batch=task/k,row=task%k;
                if(!(activeMask&(1u<<batch)))continue;
                uint64_t at=(uint64_t)batch*k*k+(uint64_t)row*k;
                LoadRow(norm,estimates,at,k);
                DataCopy(u,norm,dp);Duplicate(q,0.0f,dp);PipeBarrier<PIPE_ALL>();
                GlobalTensor<float> residual;
                residual.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)batch*k*L);
                if(packed) {
                    PreciseRowLarge<256>(residual,0,norm,u,row,false);
                }
                else {
                for(uint32_t j=0;j<k;++j) {
                    LoadRow(p,residual,(uint64_t)j*k,k);
                    AccumulateDoubleOutput(p,norm.GetValue(j),k);
                }
                }
                DataCopyPad(estimates[at],u,DataCopyExtParams{1,k*4,0,0,0});PipeBarrier<PIPE_ALL>();
                if(finishMask&(1u<<batch))StoreResult(at,k);
            }
            SyncAll<true>();completedMask|=finishMask;activeMask&=~finishMask;
        }
        return completedMask;
    }
    __aicore__ inline void FinalizeSvd(GM_ADDR workspace,GlobalTensor<float> norms,
            GlobalTensor<float> estimates,uint32_t completedMask) {
        uint32_t fullMask=0;
        for(uint32_t batch=0;batch<t.batches;++batch)
            if(!(completedMask&(1u<<batch)) && FullRank(batch,norms))fullMask|=1u<<batch;
        // All original singular-value checks precede reuse of the norm storage.
        SyncAll<true>();
        float localNorm2[4]={0,0,0,0};
        for(uint32_t task=GetBlockIdx();task<t.batches*k;task+=CoreCount()) {
            uint32_t batch=task/k,row=task%k;
            if(completedMask&(1u<<batch))continue;
            uint64_t at=(uint64_t)batch*k*k+(uint64_t)row*k;
            LoadRow(u,estimates,at,k);
            if(fullMask&(1u<<batch))localNorm2[batch]+=Dot(u,u);
            else StoreResult(at,k);
        }
        if(!fullMask)return;
        Duplicate(scalar,0.0f,8);PipeBarrier<PIPE_ALL>();
        for(uint32_t batch=0;batch<t.batches;++batch)scalar.SetValue(batch,localNorm2[batch]);
        PipeBarrier<PIPE_ALL>();DataCopy(norms[GetBlockIdx()*8],scalar,8);PipeBarrier<PIPE_ALL>();SyncAll<true>();
        DataCopy(norm,norms,CoreCount()*8);PipeBarrier<PIPE_ALL>();
        float inverseNorm2[4]={0,0,0,0};
        for(uint32_t batch=0;batch<t.batches;++batch)
            for(uint32_t core=0;core<CoreCount();++core)inverseNorm2[batch]+=norm.GetValue(core*8+batch);
        uint32_t refined=RefineFastInverse(workspace,norms,estimates,inverseNorm2,fullMask);
        uint32_t remaining=fullMask&~refined;
        if(remaining)RefineInverse(workspace,norms,estimates,((1u<<t.batches)-1)&~remaining,remaining);
    }
    __aicore__ inline uint32_t TryParallelInverse(GM_ADDR workspace) {
        const uint32_t batch=GetBlockIdx()%t.batches,lane=GetBlockIdx()/t.batches;
        const uint32_t lanes=(CoreCount()+t.batches-1-batch)/t.batches;
        const uint32_t count=(k+lanes-1-lane)/lanes;
        const uint64_t base=(uint64_t)batch*k*k;
        GlobalTensor<float> norms,estimates;
        work.SetGlobalBuffer((__gm__ float*)workspace);
        norms.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)t.batches*k*L);
        estimates.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)t.batches*k*(L+8));
        uint32_t pivotForRow[32];
        float aF2=0.0f;
        for(uint32_t r=0;r<count;++r) {
            uint32_t i=lane+r*lanes;auto row=matrix[r*L];pivotForRow[r]=k;
            Duplicate(row,0.0f,L);PipeBarrier<PIPE_ALL>();
            if constexpr(IsSameType<T,float>::value) {
                DataCopyPad(row,input[base+(uint64_t)i*k],DataCopyExtParams{1,k*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                PipeBarrier<PIPE_ALL>();
            } else for(uint32_t j=0;j<k;++j)row.SetValue(j,ReadInput(base+(uint64_t)i*k+j));
            row.SetValue(dp+i,1.0f);PipeBarrier<PIPE_ALL>();aF2+=Dot(row,row);
        }
        bool valid=true;
        for(uint32_t j=0;j<k;++j) {
            float best=0.0f;uint32_t selected=k;
            for(uint32_t r=0;r<count;++r)if(pivotForRow[r]==k) {
                float a=matrix.GetValue(r*L+j);float ab=a<0?-a:a;
                if(ab>best){best=ab;selected=lane+r*lanes;}
            }
            Duplicate(scalar,0.0f,8);PipeBarrier<PIPE_ALL>();
            scalar.SetValue(0,best);scalar.template ReinterpretCast<uint32_t>().SetValue(1,selected);PipeBarrier<PIPE_ALL>();
            DataCopy(norms[GetBlockIdx()*8],scalar,8);PipeBarrier<PIPE_ALL>();
            SyncAll<true>();
            DataCopy(norm,norms,CoreCount()*8);PipeBarrier<PIPE_ALL>();
            best=0.0f;selected=k;
            for(uint32_t core=batch;core<CoreCount();core+=t.batches) {
                float ab=norm.GetValue(core*8);
                if(ab>best){best=ab;selected=norm.template ReinterpretCast<uint32_t>().GetValue(core*8+1);}
            }
            if(selected==k)valid=false;
            if(selected<k && selected%lanes==lane) {
                uint32_t r=selected/lanes;auto row=matrix[r*L];
                Muls(row,row,1.0f/row.GetValue(j),L);PipeBarrier<PIPE_ALL>();
                pivotForRow[r]=j;
                DataCopy(work[(uint64_t)batch*k*L],row,L);PipeBarrier<PIPE_ALL>();
            }
            SyncAll<true>();
            if(selected<k) {
                DataCopy(p,work[(uint64_t)batch*k*L],L);PipeBarrier<PIPE_ALL>();
                for(uint32_t r=0;r<count;++r)if(lane+r*lanes!=selected) {
                    auto row=matrix[r*L];float weight=-row.GetValue(j);
                    Axpy(row,p,weight,L);
                }
                PipeBarrier<PIPE_ALL>();
            }
        }
        float xF2=0.0f;
        for(uint32_t r=0;r<count;++r) {
            auto row=matrix[r*L+dp];xF2+=Dot(row,row);
            if(pivotForRow[r]<k) {
                DataCopyPad(estimates[base+(uint64_t)pivotForRow[r]*k],row,DataCopyExtParams{1,k*4,0,0,0});PipeBarrier<PIPE_ALL>();
            }
        }
        Duplicate(scalar,0.0f,8);PipeBarrier<PIPE_ALL>();
        scalar.SetValue(0,aF2);scalar.SetValue(1,xF2);scalar.SetValue(2,valid?0.0f:1.0f);PipeBarrier<PIPE_ALL>();
        DataCopy(norms[GetBlockIdx()*8],scalar,8);PipeBarrier<PIPE_ALL>();SyncAll<true>();
        DataCopy(norm,norms,CoreCount()*8);PipeBarrier<PIPE_ALL>();
        float allA[4]={0,0,0,0},allX[4]={0,0,0,0};
        uint32_t acceptedMask=0,gramMask=0;
        for(uint32_t b=0;b<t.batches;++b) {
            float aa=0,xx=0,bad=0;
            for(uint32_t core=b;core<CoreCount();core+=t.batches) {
                aa+=norm.GetValue(core*8);xx+=norm.GetValue(core*8+1);bad+=norm.GetValue(core*8+2);
            }
            inputFrobenius2[b]=aa;
            if(!(bad==0.0f && aa>0.0f && xx>0.0f))continue;
            allA[b]=aa;allX[b]=xx;
            // Even the minimal spectral-norm bound ||A||F/sqrt(k) cannot
            // certify this inverse. Avoid the more expensive Gram bound.
            float minimal=t.rtol*Root(aa/(float)(int32_t)k);if(t.atol>minimal)minimal=t.atol;
            if(!(minimal*Root(xx)<0.7f))continue;
            float threshold=t.rtol*Root(aa);if(t.atol>threshold)threshold=t.atol;
            if(threshold*Root(xx)<0.7f)acceptedMask|=1u<<b;else gramMask|=1u<<b;
        }
        if(gramMask) {
            // A sharper upper bound: ||A||2^2 <= ||A*A^T||inf.
            float gramMax[4]={0,0,0,0};
            for(uint32_t task=GetBlockIdx();task<t.batches*k;task+=CoreCount()) {
                uint32_t b=task/k,row=task%k;if(!(gramMask&(1u<<b)))continue;uint64_t start=(uint64_t)b*k*k;
                if constexpr(IsSameType<T,float>::value) {
                    DataCopyPad(p,input[start+(uint64_t)row*k],DataCopyExtParams{1,k*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                    PipeBarrier<PIPE_ALL>();
                } else for(uint32_t j=0;j<k;++j)p.SetValue(j,ReadInput(start+(uint64_t)row*k+j));
                float sum=0;
                for(uint32_t i=0;i<k;++i) {
                    if constexpr(IsSameType<T,float>::value) {
                        DataCopyPad(q,input[start+(uint64_t)i*k],DataCopyExtParams{1,k*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                        PipeBarrier<PIPE_ALL>();
                    } else for(uint32_t j=0;j<k;++j)q.SetValue(j,ReadInput(start+(uint64_t)i*k+j));
                    float dot=Dot(p,q);sum+=dot<0?-dot:dot;
                }
                if(sum>gramMax[b])gramMax[b]=sum;
            }
            Duplicate(scalar,0.0f,8);PipeBarrier<PIPE_ALL>();
            for(uint32_t b=0;b<t.batches;++b)scalar.SetValue(b,gramMax[b]);
            PipeBarrier<PIPE_ALL>();DataCopy(norms[GetBlockIdx()*8],scalar,8);PipeBarrier<PIPE_ALL>();SyncAll<true>();
            DataCopy(norm,norms,CoreCount()*8);PipeBarrier<PIPE_ALL>();
            for(uint32_t b=0;b<t.batches;++b) {
                if(!(gramMask&(1u<<b)))continue;
                float maximum=0;
                for(uint32_t core=0;core<CoreCount();++core) {
                    float value=norm.GetValue(core*8+b);if(value>maximum)maximum=value;
                }
                float kf=(float)(int32_t)k;
                // Include a conservative FP32 dot/sum roundoff allowance.
                float bound=maximum*1.001f+kf*4.76837158203125e-7f*Root(kf)*allA[b];
                float threshold=t.rtol*Root(bound);if(t.atol>threshold)threshold=t.atol;
                if(threshold*Root(allX[b])<0.7f)acceptedMask|=1u<<b;
            }
        }
        return RefineFastInverse(workspace,norms,estimates,allX,acceptedMask);
    }
    __aicore__ inline void PacketDot(LocalTensor<float> dst, LocalTensor<float> left,
            LocalTensor<float> right, LocalTensor<float> tmp, LocalTensor<float> products, uint32_t count) {
        constexpr uint32_t L=512,d=256;
        BinaryRepeatParams rp{1,1,1,8,(uint8_t)(2*L/8),(uint8_t)(2*L/8)};
        for(uint32_t chunk=0;chunk<d/64;++chunk) {
            Mul(products,left[chunk*64],right[chunk*64],(uint64_t)64,(uint8_t)count,rp);PipeBarrier<PIPE_V>();
            WholeReduceSum(tmp[chunk],products,64,count,8,1,8);PipeBarrier<PIPE_V>();
        }
        WholeReduceSum(dst,tmp,d/64,count,1,1,1);PipeBarrier<PIPE_V>();
    }
    // Block Jacobi: one core diagonalizes a pair of column blocks locally,
    // so global synchronization is needed only between block pairs.
    template<uint32_t width>
    __aicore__ inline void RunBlockSweeps(GM_ADDR workspace,uint32_t completedMask) {
        constexpr uint32_t k=256,L=512,dp=256,count=width,padded=(width+7)/8*8;
        constexpr uint32_t blocks=256/width,pairs=blocks/2,ring=blocks-1,localRing=2*width-1;
        uint32_t activeMap[4],activeBatches=0;
        for(uint32_t b=0;b<t.batches;++b)if(!(completedMask&(1u<<b)))activeMap[activeBatches++]=b;
        auto packet=matrix,updated=smallTransposeBuf.Get<float>(),products=smallProductBuf.Get<float>();
        auto weights=smallIndexBuf.Get<float>(),cw=weights,sw=weights[padded*8],nw=weights[padded*16];
        auto aa=norm,bb=norm[dp],gg=norm[2*dp],x=norm[3*dp],y=norm[4*dp],z=norm[5*dp],cc=norm[6*dp],ss=norm[7*dp];
        auto masks=rawBuf.Get<uint8_t>();
        auto maximumCorrelation=rawBuf.Get<float>()[32];
        auto ones=rawBuf.Get<float>()[112],zeros=rawBuf.Get<float>()[128];
        auto signMask=rawBuf.Get<int32_t>()[144];
        Duplicate(ones,1.0f,(uint64_t)padded,1,1,8);
        Duplicate(zeros,0.0f,(uint64_t)padded,1,1,8);
        Duplicate(signMask,(int32_t)0x80000000u,(uint64_t)padded,1,1,8);PipeBarrier<PIPE_V>();
        auto normPIndex=rawBuf.Get<uint32_t>()[48],normQIndex=rawBuf.Get<uint32_t>()[64];
        auto nextP=rawBuf.Get<float>()[80],nextQ=rawBuf.Get<float>()[96];
        auto crossQIndex=rawBuf.Get<uint32_t>()[160];
        for(uint32_t slot=0;slot<count;++slot) {
            normPIndex.SetValue(slot,(slot==0?0:(slot+1<count?slot+1:dp+count-1))*4);
            normQIndex.SetValue(slot,(slot==0?1:dp+slot-1)*4);
            crossQIndex.SetValue(slot,((slot+1)%count)*4);
        }
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<float> convergence;
        convergence.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)t.batches*k*L);
        work.SetGlobalBuffer((__gm__ float*)workspace);
        for(uint32_t sweep=0;sweep<24;++sweep) {
            Duplicate(maximumCorrelation,0.0f,(uint64_t)padded,1,1,8);PipeBarrier<PIPE_V>();
            for(uint32_t round=0;round<ring;++round) {
                const bool fullLocal=round==0;
                for(uint32_t task=GetBlockIdx();task<activeBatches*pairs;task+=CoreCount()) {
                    uint32_t batch=activeMap[task/pairs],pair=task%pairs;
                    uint32_t bp=pair==0?ring:(round+pair)%ring;
                    uint32_t bq=pair==0?round:(round+ring-pair)%ring;
                    for(uint32_t slot=0;slot<count;++slot) {
                        uint32_t jp=fullLocal?(slot==0?localRing:slot):slot;
                        uint32_t jq=fullLocal?(slot==0?0:localRing-slot):width+slot;
                        uint32_t cp=(jp<width?bp:bq)*width+jp%width;
                        uint32_t cq=(jq<width?bp:bq)*width+jq%width;
                        DataCopy(packet[slot*2*L],work[(uint64_t)batch*k*L+(uint64_t)cp*L],L);
                        DataCopy(packet[slot*2*L+L],work[(uint64_t)batch*k*L+(uint64_t)cq*L],L);
                    }
                    PipeBarrier<PIPE_ALL>();
                    for(uint32_t localRound=0;localRound<(fullLocal?localRing:width);++localRound) {
                if((localRound&3)==0) {
                    PacketDot(aa,packet,packet,x,products,count);
                    PacketDot(bb,packet[L],packet[L],x,products,count);
                }
                PacketDot(gg,packet,packet[L],x,products,count);
                // Compare on 910B consumes complete 64-float repeats. Only
                // the first count bits are subsequently selected.
                // Match the original pair acceptance criteria, but evaluate all
                // independent pairs on this core in SIMD lanes.
                Add(y,aa,bb,(uint64_t)count,1,BinaryRepeatParams{});PipeBarrier<PIPE_V>();Muls(y,y,1e-14f,(uint64_t)count,1,UnaryRepeatParams{});PipeBarrier<PIPE_V>();
                Compare(masks,aa,y,CMPMODE::GT,64);
                Compare(masks[32],bb,y,CMPMODE::GT,64);PipeBarrier<PIPE_V>();
                And(masks.template ReinterpretCast<uint16_t>(),masks.template ReinterpretCast<uint16_t>(),masks[32].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
                Mul(x,gg,gg,(uint64_t)count,1,BinaryRepeatParams{});Mul(y,aa,bb,(uint64_t)count,1,BinaryRepeatParams{});PipeBarrier<PIPE_V>();
                {
                    Div(z,x,y,(uint64_t)count,1,BinaryRepeatParams{});PipeBarrier<PIPE_V>();
                    float tiny=inputFrobenius2[batch]*(1e-14f/256.0f);
                    CompareScalar(masks[64],aa,tiny,CMPMODE::GT,64);
                    CompareScalar(masks[96],bb,tiny,CMPMODE::GT,64);PipeBarrier<PIPE_V>();
                    And(masks[64].template ReinterpretCast<uint16_t>(),masks[64].template ReinterpretCast<uint16_t>(),masks[96].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
                    // A pair already excluded by the rotation's norm-ratio
                    // guard cannot keep the sweep convergence test active.
                    And(masks[64].template ReinterpretCast<uint16_t>(),masks[64].template ReinterpretCast<uint16_t>(),masks.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
                    Select(z,masks[64],z,zeros,SELMODE::VSEL_CMPMASK_SPR,(uint64_t)count,1,BinaryRepeatParams{});PipeBarrier<PIPE_V>();
                    Max(maximumCorrelation,maximumCorrelation,z,(uint64_t)count,1,BinaryRepeatParams{});PipeBarrier<PIPE_V>();
                }
                Muls(y,y,1e-16f,(uint64_t)count,1,UnaryRepeatParams{});PipeBarrier<PIPE_V>();
                Compare(masks[32],x,y,CMPMODE::GT,64);PipeBarrier<PIPE_V>();
                And(masks.template ReinterpretCast<uint16_t>(),masks.template ReinterpretCast<uint16_t>(),masks[32].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
                Sub(x,bb,aa,(uint64_t)count,1,BinaryRepeatParams{});Muls(y,gg,2.0f,(uint64_t)count,1,UnaryRepeatParams{});PipeBarrier<PIPE_V>();
                Div(x,x,y,(uint64_t)count,1,BinaryRepeatParams{});PipeBarrier<PIPE_V>();
                Mul(y,x,x,(uint64_t)count,1,BinaryRepeatParams{});Abs(z,x,(uint64_t)count,1,UnaryRepeatParams{});PipeBarrier<PIPE_V>();
                Adds(y,y,1.0f,(uint64_t)count,1,UnaryRepeatParams{});PipeBarrier<PIPE_V>();Sqrt(y,y,(uint64_t)count,1,UnaryRepeatParams{});PipeBarrier<PIPE_V>();
                Add(y,y,z,(uint64_t)count,1,BinaryRepeatParams{});PipeBarrier<PIPE_V>();
                Div(cc,ones,y,(uint64_t)count,1,BinaryRepeatParams{});PipeBarrier<PIPE_V>();
                And(z.template ReinterpretCast<uint16_t>(),x.template ReinterpretCast<uint16_t>(),signMask.template ReinterpretCast<uint16_t>(),(uint64_t)(2*count),1,BinaryRepeatParams{});PipeBarrier<PIPE_V>();
                Or(x.template ReinterpretCast<uint16_t>(),cc.template ReinterpretCast<uint16_t>(),z.template ReinterpretCast<uint16_t>(),(uint64_t)(2*count),1,BinaryRepeatParams{});PipeBarrier<PIPE_V>();
                // a' = a - t*g and b' = b + t*g. Refresh from the actual
                // columns every four local rounds to bound accumulated drift.
                Mul(z,x,gg,(uint64_t)count,1,BinaryRepeatParams{});PipeBarrier<PIPE_V>();
                Select(z,masks,z,zeros,SELMODE::VSEL_CMPMASK_SPR,(uint64_t)count,1,BinaryRepeatParams{});PipeBarrier<PIPE_V>();
                Sub(aa,aa,z,(uint64_t)count,1,BinaryRepeatParams{});
                Add(bb,bb,z,(uint64_t)count,1,BinaryRepeatParams{});PipeBarrier<PIPE_V>();
                Mul(y,x,x,(uint64_t)count,1,BinaryRepeatParams{});PipeBarrier<PIPE_V>();Adds(y,y,1.0f,(uint64_t)count,1,UnaryRepeatParams{});PipeBarrier<PIPE_V>();
                Sqrt(y,y,(uint64_t)count,1,UnaryRepeatParams{});PipeBarrier<PIPE_V>();Div(cc,ones,y,(uint64_t)count,1,BinaryRepeatParams{});PipeBarrier<PIPE_V>();
                Mul(ss,cc,x,(uint64_t)count,1,BinaryRepeatParams{});PipeBarrier<PIPE_V>();
                Select(cc,masks,cc,ones,SELMODE::VSEL_CMPMASK_SPR,(uint64_t)count,1,BinaryRepeatParams{});
                Select(ss,masks,ss,zeros,SELMODE::VSEL_CMPMASK_SPR,(uint64_t)count,1,BinaryRepeatParams{});PipeBarrier<PIPE_V>();
                Muls(x,ss,-1.0f,(uint64_t)count,1,UnaryRepeatParams{});PipeBarrier<PIPE_V>();
                Brcb(cw,cc,padded/8,BrcbRepeatParams{1,8});
                Brcb(sw,ss,padded/8,BrcbRepeatParams{1,8});
                Brcb(nw,x,padded/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                BinaryRepeatParams toPacked{1,1,0,(uint8_t)(L/8),(uint8_t)(2*L/8),1};
                BinaryRepeatParams toColumn{1,1,0,(uint8_t)(2*L/8),(uint8_t)(2*L/8),1};
                if constexpr(width==4) {
                    // Eight contiguous repeats per column amortize the vector
                    // instruction setup; a broadcast coefficient remains fixed.
                    BinaryRepeatParams longRow{1,1,0,8,8,0};
                    for(uint32_t slot=0;slot<count;++slot)
                        Mul(updated[slot*L],packet[slot*2*L],cw[slot*8],(uint64_t)64,(uint8_t)(L/64),longRow);
                    PipeBarrier<PIPE_V>();
                    for(uint32_t slot=0;slot<count;++slot)
                        MulAddDst(updated[slot*L],packet[slot*2*L+L],nw[slot*8],(uint64_t)64,(uint8_t)(L/64),longRow);
                    PipeBarrier<PIPE_V>();
                    for(uint32_t slot=0;slot<count;++slot)
                        Mul(packet[slot*2*L+L],packet[slot*2*L+L],cw[slot*8],(uint64_t)64,(uint8_t)(L/64),longRow);
                    PipeBarrier<PIPE_V>();
                    for(uint32_t slot=0;slot<count;++slot)
                        MulAddDst(packet[slot*2*L+L],packet[slot*2*L],sw[slot*8],(uint64_t)64,(uint8_t)(L/64),longRow);
                    PipeBarrier<PIPE_V>();
                } else {
                for(uint32_t off=0;off<L;off+=64) {
                    Mul(updated[off],packet[off],cw,(uint64_t)64,(uint8_t)count,toPacked);PipeBarrier<PIPE_V>();
                    MulAddDst(updated[off],packet[L+off],nw,(uint64_t)64,(uint8_t)count,toPacked);PipeBarrier<PIPE_V>();
                    Mul(packet[L+off],packet[L+off],cw,(uint64_t)64,(uint8_t)count,toColumn);PipeBarrier<PIPE_V>();
                    MulAddDst(packet[L+off],packet[off],sw,(uint64_t)64,(uint8_t)count,toColumn);PipeBarrier<PIPE_V>();
                }
                }

                        if(fullLocal) {
                        // Rotate the local ring; each full local sweep restores
                        // the column layout, with the fixed column in pair 0.
                        DataCopy(p,updated[L],L);DataCopy(q,packet[(count-1)*2*L+L],L);PipeBarrier<PIPE_V>();
                        DataCopy(packet,updated,L);
                        DataCopy(packet[2*L],updated[2*L],DataCopyParams{count-2,L/8,0,L/8});
                        DataCopy(packet[(count-1)*2*L],q,L);PipeBarrier<PIPE_V>();
                        DataCopy(updated,packet[L],DataCopyParams{count,L/8,L/8,0});PipeBarrier<PIPE_V>();
                        DataCopy(packet[3*L],updated,DataCopyParams{count-1,L/8,0,L/8});
                        DataCopy(packet[L],p,L);PipeBarrier<PIPE_V>();
                        Gather(nextP,aa,normPIndex,0,count);
                        Gather(nextQ,aa,normQIndex,0,count);PipeBarrier<PIPE_V>();
                        DataCopy(aa,nextP,padded);DataCopy(bb,nextQ,padded);PipeBarrier<PIPE_V>();
                        } else {
                            // Across subsequent block pairs, rotate only Q.
                            // Every cross-block column pair is visited once;
                            // within-block pairs were covered in round zero.
                            DataCopy(q,packet[L],L);PipeBarrier<PIPE_V>();
                            DataCopy(packet,updated,DataCopyParams{count,L/8,0,L/8});PipeBarrier<PIPE_V>();
                            DataCopy(updated,packet[3*L],DataCopyParams{count-1,L/8,L/8,0});PipeBarrier<PIPE_V>();
                            DataCopy(packet[L],updated,DataCopyParams{count-1,L/8,0,L/8});
                            DataCopy(packet[(count-1)*2*L+L],q,L);PipeBarrier<PIPE_V>();
                            Gather(nextQ,bb,crossQIndex,0,count);PipeBarrier<PIPE_V>();
                            DataCopy(bb,nextQ,padded);PipeBarrier<PIPE_V>();
                        }
                    }
                    PipeBarrier<PIPE_ALL>();
                    for(uint32_t slot=0;slot<count;++slot) {
                        uint32_t jp=fullLocal?(slot==0?localRing:slot):slot;
                        uint32_t jq=fullLocal?(slot==0?0:localRing-slot):width+slot;
                        uint32_t cp=(jp<width?bp:bq)*width+jp%width;
                        uint32_t cq=(jq<width?bp:bq)*width+jq%width;
                        DataCopy(work[(uint64_t)batch*k*L+(uint64_t)cp*L],packet[slot*2*L],L);
                        DataCopy(work[(uint64_t)batch*k*L+(uint64_t)cq*L],packet[slot*2*L+L],L);
                    }
                    PipeBarrier<PIPE_ALL>();
                }
                SyncAll<true>();
            }
            {
                WholeReduceMax(scalar,maximumCorrelation,count,1,1,1,8);PipeBarrier<PIPE_ALL>();
                DataCopy(convergence[GetBlockIdx()*8],scalar,8);PipeBarrier<PIPE_ALL>();SyncAll<true>();
                DataCopy(norm,convergence,CoreCount()*8);PipeBarrier<PIPE_ALL>();
                float maximum=0.0f;
                for(uint32_t core=0;core<CoreCount();++core) {
                    float value=norm.GetValue(core*8);if(value>maximum)maximum=value;
                }
                if(maximum<1e-14f)break;
            }
        }
    }
    __aicore__ inline void RunPacketSweeps(GM_ADDR workspace,uint32_t completedMask) {
        uint32_t active=0;
        for(uint32_t batch=0;batch<t.batches;++batch)if(!(completedMask&(1u<<batch)))++active;
        if(active==1)RunBlockSweeps<4>(workspace,completedMask);
        else if(active==2)RunBlockSweeps<8>(workspace,completedMask);
        else RunBlockSweeps<16>(workspace,completedMask);
    }
    __aicore__ inline void RunParallel(GM_ADDR workspace,uint32_t completedMask=0) {
        GlobalTensor<float> norms,estimates;
        norms.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)t.batches*k*L);
        estimates.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)t.batches*k*(L+8));
        for(uint32_t task=GetBlockIdx();task<t.batches*k;task+=CoreCount()) {
            uint32_t batch=task/k,j=task%k;
            if(completedMask && (completedMask&(1u<<batch)))continue;
            work.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)batch*k*L);
            uint64_t base=(uint64_t)batch*t.m*t.n;
            Duplicate(p,0.0f,L);PipeBarrier<PIPE_ALL>();
            for(uint32_t i=0;i<d;++i) {
                uint64_t at=t.m>=t.n?(uint64_t)i*t.n+j:(uint64_t)j*t.n+i;
                if constexpr(IsSameType<T,float>::value) p.SetValue(i,input.GetValue(base+at));
                else { union {uint32_t bits;float value;} x; x.bits=(uint32_t)inputBits.GetValue(base+at)<<16; p.SetValue(i,x.value); }
            }
            p.SetValue(dp+j,1.0f);PipeBarrier<PIPE_ALL>();Store(j,p);
        }
        SyncAll<true>();
        const uint32_t even=(k+1)/2*2, pairs=even/2, ring=even-1;
        const uint32_t packetPairs=(t.batches*(k/2)+CoreCount()-1)/CoreCount();
        if(t.m==t.n && k==256 && t.batches<=4 && packetPairs<=16)RunPacketSweeps(workspace,completedMask);
        else {
        for(uint32_t sweep=0;sweep<24;++sweep) {
            for(uint32_t round=0;round<ring;++round) {
                for(uint32_t task=GetBlockIdx();task<t.batches*pairs;task+=CoreCount()) {
                    uint32_t batch=task/pairs,pair=task%pairs;
                    if(completedMask && (completedMask&(1u<<batch)))continue;
                    uint32_t jp=pair==0?ring:(round+pair)%ring;
                    uint32_t jq=pair==0?round:(round+ring-pair)%ring;
                    if(jp>=k || jq>=k) continue;
                    work.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)batch*k*L);
                    auto cp=Column(jp,p),cq=Column(jq,q);
                    float a=Dot(cp,cp),b=Dot(cq,cq),g=Dot(cp,cq);
                    if(a<=1e-14f*(a+b) || b<=1e-14f*(a+b) || g*g<=1e-16f*a*b) continue;
                    float tau=(b-a)/(2.0f*g),at=tau<0?-tau:tau;
                    float tangent=(tau<0?-1.0f:1.0f)/(at+Root(1.0f+tau*tau));
                    float cosine=1.0f/Root(1.0f+tangent*tangent),sine=cosine*tangent;
                    Muls(u,cp,cosine,L);Muls(v,cq,cosine,L);PipeBarrier<PIPE_V>();
                    Axpy(u,cq,-sine,L);Axpy(v,cp,sine,L);PipeBarrier<PIPE_ALL>();
                    Store(jp,u);Store(jq,v);
                }
                SyncAll<true>();
            }
        }
        }
        for(uint32_t task=GetBlockIdx();task<t.batches*k;task+=CoreCount()) {
            uint32_t batch=task/k,j=task%k;
            if(completedMask && (completedMask&(1u<<batch)))continue;
            work.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)batch*k*L);
            auto col=Column(j,p);float a=Dot(col,col);
            Duplicate(scalar,0.0f,8);PipeBarrier<PIPE_ALL>();scalar.SetValue(0,a);PipeBarrier<PIPE_ALL>();
            DataCopy(norms[(uint64_t)task*8],scalar,8);PipeBarrier<PIPE_ALL>();
        }
        SyncAll<true>();
        for(uint32_t task=GetBlockIdx();task<t.batches*t.n;task+=CoreCount()) {
            uint32_t batch=task/t.n,i=task%t.n;
            if(completedMask && (completedMask&(1u<<batch)))continue;
            work.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)batch*k*L);
            DataCopy(norm,norms[(uint64_t)batch*k*8],k*8);PipeBarrier<PIPE_ALL>();
            float largest=0;
            for(uint32_t j=0;j<k;++j) {float a=norm.GetValue(j*8);if(a>largest) largest=a;}
            float threshold=t.rtol*Root(largest);if(t.atol>threshold)threshold=t.atol;
            Duplicate(u,0.0f,(t.m+7)/8*8);Duplicate(q,0.0f,(t.m+7)/8*8);PipeBarrier<PIPE_ALL>();
            for(uint32_t j=0;j<k;++j) {
                float a=norm.GetValue(j*8);if(!(a>threshold*threshold)) continue;
                auto col=Column(j,p);float weight=col.GetValue(t.m>=t.n?dp+i:i)/a;
                AccumulateOutput(t.m>=t.n?col:col[dp],weight,t.m);
            }
            uint64_t base=(uint64_t)batch*t.m*t.n+(uint64_t)i*t.m;
            if(t.m==t.n) {DataCopyPad(estimates[base],u,DataCopyExtParams{1,t.m*4,0,0,0});PipeBarrier<PIPE_ALL>();}
            else StoreResult(base,t.m);
        }
        if(t.m==t.n) {
            SyncAll<true>();
            if(k==256 && t.batches<=4)FinalizeSvd(workspace,norms,estimates,completedMask);
            else RefineInverse(workspace,norms,estimates,completedMask);
        }
    }

public:
    __aicore__ inline void Run(GM_ADDR A,GM_ADDR Y,GM_ADDR workspace,LinalgPinvTilingData tiling,uint32_t preparedMask=0,const float *preparedNorm2=nullptr,uint32_t ownedBatch=0xffffffffu) {
        t=tiling;d=t.rows;k=t.columns;dp=(d+7)/8*8;L=t.stride;
        if(preparedNorm2)for(uint32_t b=0;b<t.batches;++b)inputFrobenius2[b]=preparedNorm2[b];
        if constexpr(serialOnly)t.parallel=0;
        const bool smallTeam=!serialOnly && t.parallel==2;
        if(smallTeam)t.parallel=0;
        const bool fastParallel=!serialOnly && t.parallel && t.m==t.n && k<=256 && t.batches<=4 && CoreCount()/t.batches>=8;
        uint32_t packetPairs=(t.batches*(k/2)+CoreCount()-1)/CoreCount();
        bool packetSvd=!serialOnly && t.parallel && t.m==t.n && k==256 && t.batches<=4 && packetPairs<=16;
        bool packedRefine=!serialOnly && t.parallel && t.m==t.n && (k==256) && t.batches<=4 && packetPairs<=16;
        uint32_t localRows=fastParallel?(k+CoreCount()/t.batches-1)/(CoreCount()/t.batches):0;
        if(packetSvd && packetPairs<16)packetPairs=16;
        if(packetSvd && localRows<2*packetPairs)localRows=2*packetPairs;
        if(packedRefine && localRows<(6*dotTile+2*dotTail+L-1)/L)localRows=(6*dotTile+2*dotTail+L-1)/L;
        uint32_t smallStride=L+((t.resident && t.m==64 && t.n==64 && IsSameType<T,float>::value)?8:0);
        pipe.InitBuffer(matrixBuf,t.resident?(k*smallStride+((t.m==t.n && (k==15 || k==16 || k==50 || k==64))?k*dp:0))*4:((fastParallel||packetSvd)?localRows*L*4:32));
        pipe.InitBuffer(pBuf,smallStride*4);pipe.InitBuffer(qBuf,smallStride*4);pipe.InitBuffer(uBuf,smallStride*4);pipe.InitBuffer(vBuf,smallStride*4);
        pipe.InitBuffer(prodBuf,smallStride*4);pipe.InitBuffer(reduceBuf,smallStride*4);pipe.InitBuffer(scalarBuf,256);
        pipe.InitBuffer(normBuf,t.parallel?(packedRefine?8*256*4:k*32):((k+7)/8*8>6*dp?(k+7)/8*8:6*dp)*4);pipe.InitBuffer(rawBuf,packetSvd?704:(d+15)/16*32);
        uint32_t smallBytes=(!t.parallel && t.resident && t.m==t.n && k<=64)?k*dp*4:32;
        pipe.InitBuffer(smallTransposeBuf,(packetSvd?packetPairs*L*4:(packedRefine?dotTile*4:smallBytes))*((t.resident && t.m==t.n && (k==15 || k==16 || k==50 || k==64))?2:1)+(smallStride-L)*k*4);pipe.InitBuffer(smallProductBuf,packedRefine?dotWidth*64*4:((t.resident && t.m==t.n && (k==15 || k==16) && IsSameType<T,float>::value)?k*dp*dp*4:smallBytes));pipe.InitBuffer(smallIndexBuf,packedRefine?(dotTile+4*dotTail)*4:smallBytes+(smallTeam?dp*4:0)+((t.m==15 && t.n==15 && IsSameType<T,float>::value)?15*16*4:0));
        matrix=matrixBuf.Get<float>();p=pBuf.Get<float>();q=qBuf.Get<float>();u=uBuf.Get<float>();v=vBuf.Get<float>();
        prod=prodBuf.Get<float>();reduce=reduceBuf.Get<float>();scalar=scalarBuf.Get<float>();norm=normBuf.Get<float>();
        input.SetGlobalBuffer((__gm__ T*)A);inputBits.SetGlobalBuffer((__gm__ uint16_t*)A);output.SetGlobalBuffer((__gm__ T*)Y);
        if constexpr(serialOnly) {
            if(ownedBatch!=0xffffffffu)RunBatch(ownedBatch,workspace);
            else for(uint32_t batch=GetBlockIdx();batch<t.batches;batch+=CoreCount())RunBatch(batch,workspace);
        } else if(smallTeam) {
            if constexpr(IsSameType<T,float>::value) {
                if(k==50)RunSmallTeam<50>(workspace);else RunSmallTeam<64>(workspace);
            }
        } else if(t.parallel) {
            uint32_t completedMask=preparedNorm2?preparedMask:(fastParallel?TryParallelInverse(workspace):0);
            if(!fastParallel || completedMask!=((1u<<t.batches)-1))RunParallel(workspace,completedMask);
        }
        else for(uint32_t batch=GetBlockIdx();batch<t.batches;batch+=CoreCount()) RunBatch(batch,workspace);
    }
};

// A 16-row capacity for all square inputs that fit: checked seed plus one correction.
// A rejected batch is handled by the existing serial pinv implementation.
class ResidentTinyInverse {
    static constexpr uint32_t K=16,DP=16,L=32;

    LocalTensor<float> mat,original,xt,residual,products,corrected,weights,pivot,col,scalar,reduce,partial,repeated;
    LocalTensor<uint32_t> indices;
    GlobalTensor<float> input,output;
    LinalgPinvTilingData t;
    uint32_t size;
    GM_ADDR staticTiling=nullptr;
    __aicore__ inline bool Batch(uint32_t batch) {
        uint64_t base=(uint64_t)batch*size*size;
        GlobalTensor<uint32_t> payload;
        payload.SetGlobalBuffer((__gm__ uint32_t*)(staticTiling+__builtin_offsetof(LinalgPinvTinyTilingData,packed)));
        if(size==16)DataCopy(original,input[base],K*DP);
        else DataCopyPad(products,input[base],DataCopyExtParams{1,size*size*4,0,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-size*size%8)%8),0.0f});
        DataCopy(mat.ReinterpretCast<uint32_t>(),payload,K*L);PipeBarrier<PIPE_ALL>();
        Cast(indices.ReinterpretCast<int32_t>(),mat[8].ReinterpretCast<half>(),RoundMode::CAST_RINT,(uint64_t)DP,(uint8_t)K,UnaryRepeatParams{1,1,DP/8,L/8});
        if(size<16) {
            products.SetValue(size*size,0.0f);PipeBarrier<PIPE_ALL>();
            Cast(partial.ReinterpretCast<int32_t>(),mat.ReinterpretCast<half>(),RoundMode::CAST_RINT,(uint64_t)DP,(uint8_t)K,UnaryRepeatParams{1,1,DP/8,L/8});PipeBarrier<PIPE_V>();
            Gather(original,products,partial.ReinterpretCast<uint32_t>(),0,K*DP);PipeBarrier<PIPE_V>();
            Add(original[size*DP],original[size*DP],mat[size*L+DP],(uint64_t)DP,(uint8_t)(K-size),BinaryRepeatParams{1,1,1,DP/8,DP/8,L/8});
        }
        PipeBarrier<PIPE_V>();
        float af2=0,xf2=0;bool refine=true;
        {
            constexpr uint32_t CP=16, LIVE=32;
            auto cm=products,unit=cm[LIVE*CP],cw=unit[K*DP],cd=cw[LIVE*8],pc=cd[64],pr=pc[DP];
            auto pm=partial.ReinterpretCast<int32_t>();
            // Original row-major A is the column-major representation of A^T.
            DataCopy(cm,original,DataCopyParams{K,DP/8,0,(CP-DP)/8});
            DataCopy(unit,mat[DP],DataCopyParams{K,DP/8,(L-DP)/8,0});
            // Static-arena Tiny has no TPipe; derive indices from the payload.
            ShiftRight(pm,indices.ReinterpretCast<int32_t>(),int32_t(CP==16?1:2),K);PipeBarrier<PIPE_V>();
            DataCopy(cm[K*CP],unit,DataCopyParams{K,DP/8,0,(CP-DP)/8});
            if constexpr(CP==24){Muls(pm,pm,int32_t(3),K);PipeBarrier<PIPE_V>();}
            Adds(pm[K],pm,int32_t(K*CP*4),K);PipeBarrier<PIPE_V>();
            for(uint32_t j=0;j<size;++j) {
                Brcb(cd,cm[j*CP+j/8*8],1,BrcbRepeatParams{1,8});
                Gather(pr,cm[j],pm.ReinterpretCast<uint32_t>(),0,(K+j+8)/8*8);
                Sub(pc,unit[j*DP],cm[j*CP],DP);PipeBarrier<PIPE_V>();
                Div(pc,pc,cd[j%8*8],(uint64_t)DP,1,BinaryRepeatParams{1,1,0,8,8,0});
                Brcb(cw,pr,(K+j+8)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                MulAddDst(cm[(j+1)*CP],pc,cw[(j+1)*8],(uint64_t)DP,(uint8_t)K,BinaryRepeatParams{1,1,0,CP/8,0,1});PipeBarrier<PIPE_V>();
            }
            DataCopy(mat[DP],cm[K*CP],DataCopyParams{K,DP/8,(CP-DP)/8,(L-DP)/8});PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_V>();Gather(xt,mat[DP],indices,0,K*DP);PipeBarrier<PIPE_V>();
        BinaryRepeatParams broadcast{1,1,1,8,8,0};
        // Form (I-A*X)^T directly so the correction can consume it.
        for(uint32_t r=0;r<4;++r)DataCopy(repeated[r*DP],xt,DataCopyParams{K,DP/8,0,6});
        PipeBarrier<PIPE_V>();
         {
            for(uint32_t group=0;group<4;++group)
                Mul(products[group*64],original[group*64],repeated,(uint64_t)64,(uint8_t)K,BinaryRepeatParams{1,1,1,DP*DP/8,0,8});
        }
        PipeBarrier<PIPE_V>();
        BlockReduceSum(partial,products,64,64,1,1,8);PipeBarrier<PIPE_V>();
        PairReduceSum(residual,partial,8,64,1,1,8);PipeBarrier<PIPE_V>();
        Muls(residual,residual,-1.0f,K*DP);PipeBarrier<PIPE_V>();
         {
            for(uint32_t g=0;g<4;++g) {
                uint64_t mask[2]={uint64_t(0x0008000400020001ULL)<<(g*4),0};
                Adds(residual[g*64],residual[g*64],1.0f,mask,1,UnaryRepeatParams{1,1,8,8});
            }
        }
        PipeBarrier<PIPE_V>();
         {
            // Batch three Frobenius statistics into two vector reductions.
            Mul(products,original,original,K*DP);
            Mul(products[K*DP],mat[DP],mat[DP],(uint64_t)K,(uint8_t)K,BinaryRepeatParams{1,1,1,DP/8,L/8,L/8});
            Mul(products[2*K*DP],residual,residual,K*DP);PipeBarrier<PIPE_V>();
            BlockReduceSum(partial,products,12,64,1,1,8);PipeBarrier<PIPE_V>();
            WholeReduceSum(scalar,partial,32,3,8,1,4);
            // R is transposed: the same block sums give rows of X and columns of R.
            PairReduceSum(reduce,partial[32],1,64,1,1,8);PipeBarrier<PIPE_V>();
            WholeReduceMax(scalar[24],reduce,(uint64_t)K,2,8,1,K/8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_ALL>();
            af2=scalar.GetValue(0);xf2=scalar.GetValue(8);float rf2=scalar.GetValue(16);
            float cutoff2=t.rtol*t.rtol*af2;if(t.atol*t.atol>cutoff2)cutoff2=t.atol*t.atol;
            if(!(cutoff2*xf2<0.49f && xf2<=1e6f && rf2<0.0625f && xf2*rf2*rf2<6.4e-9f))return false;
            // Entry error <= maxRowNorm(X)*maxColumnNorm(R)/(1-rho).
            // Use a 9e-5 absolute budget; larger estimates keep the correction.
            float rowX2=scalar.GetValue(24),colR2=scalar.GetValue(32);
            float limit=rf2<1e-4f?7.93881e-9f:4.55625e-9f;
            refine=!(rowX2>=0.0f && colR2>=0.0f && rowX2*colR2<limit);
        }

        if(refine) {
        // Residual rows already represent the columns used by X*R.
        for(uint32_t r=0;r<4;++r)DataCopy(repeated[r*DP],mat[DP],DataCopyParams{K,DP/8,(L-DP)/8,6});
        PipeBarrier<PIPE_V>();
         {
            for(uint32_t group=0;group<4;++group)
                Mul(products[group*64],residual[group*64],repeated,(uint64_t)64,(uint8_t)K,BinaryRepeatParams{1,1,1,DP*DP/8,0,8});
        }
        PipeBarrier<PIPE_V>();
        BlockReduceSum(partial,products,64,64,1,1,8);PipeBarrier<PIPE_V>();
        PairReduceSum(corrected,partial,8,64,1,1,8);PipeBarrier<PIPE_V>();
        Add(corrected,corrected,mat[DP],(uint64_t)K,(uint8_t)K,BinaryRepeatParams{1,1,1,DP/8,DP/8,L/8});PipeBarrier<PIPE_ALL>();

        } else {
            DataCopy(corrected,mat[DP],DataCopyParams{K,DP/8,(L-DP)/8,0});PipeBarrier<PIPE_ALL>();
        }

        if(size==16)DataCopy(output[base],corrected,K*DP);
        else {
            // Compact each capacity row to the logical width, then store n*n values.
            Duplicate(indices,uint32_t(((1u<<size)-1)*0x00010001u),8);PipeBarrier<PIPE_ALL>();
            uint64_t kept=0;
            GatherMask(products,corrected,indices,true,K*DP,GatherMaskParams{1,1,8,0},kept);PipeBarrier<PIPE_ALL>();
            DataCopyPad(output[base],products,DataCopyExtParams{1,size*size*4,0,0,0});
        }
        PipeBarrier<PIPE_ALL>();return true;
    }
    template<bool TransposeLeft=false,bool TransposeRight=false>
    __aicore__ inline void DenseProduct(LocalTensor<float> dst,LocalTensor<float> left,LocalTensor<float> right) {
        if constexpr(TransposeRight)DataCopy(xt,right,K*DP);
        else Gather(xt,right,indices,0,K*DP);
        PipeBarrier<PIPE_ALL>();
        if constexpr(TransposeLeft) {Gather(dst,left,indices,0,K*DP);PipeBarrier<PIPE_ALL>();
            for(uint32_t r=0;r<4;++r)DataCopy(repeated[r*DP],dst,DataCopyParams{K,DP/8,0,6});
        } else for(uint32_t r=0;r<4;++r)DataCopy(repeated[r*DP],left,DataCopyParams{K,DP/8,0,6});
        PipeBarrier<PIPE_V>();BinaryRepeatParams broadcast{1,1,1,8,8,0};
        for(uint32_t i=0;i<K;++i)Mul(products[i*DP*DP],xt,repeated[i*64],(uint64_t)64,(uint8_t)4,broadcast);
        PipeBarrier<PIPE_V>();BlockReduceSum(partial,products,64,64,1,1,8);PipeBarrier<PIPE_V>();
        PairReduceSum(dst,partial,8,64,1,1,8);PipeBarrier<PIPE_ALL>();
    }
    // Repair drift outside the retained row/column spaces, then correct
    // the retained block. X*=A_tau^+ is a fixed point of both operations.
    __aicore__ inline bool Polish(uint32_t batch) {
        uint64_t base=(uint64_t)batch*size*size;
        Duplicate(mat,0.0f,K*L);Duplicate(original,0.0f,K*DP);PipeBarrier<PIPE_ALL>();
        DataCopyPad(original,input[base],DataCopyExtParams{(uint16_t)size,size*4,0,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(DP-size),0.0f});
        DataCopyPad(mat,output[base],DataCopyExtParams{(uint16_t)size,size*4,0,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(DP-size),0.0f});PipeBarrier<PIPE_ALL>();
        if(size<16)for(uint32_t j=size;j<K;++j){original.SetValue(j*DP+j,1.0f);mat.SetValue(j*DP+j,1.0f);}
        for(uint32_t r=0;r<K;++r)indices.SetValue(r,r*DP*4);PipeBarrier<PIPE_ALL>();
        auto ix=indices.ReinterpretCast<int32_t>();for(uint32_t j=1;j<K;++j)Adds(ix[j*DP],ix,int32_t(j*4),DP);PipeBarrier<PIPE_ALL>();
        DenseProduct(residual,mat,original);                 // X A
        DenseProduct(corrected,original,mat);                // A X
        float retained=0;for(uint32_t r=0;r<K;++r)retained+=corrected.GetValue(r*DP+r);
        if(retained>K-0.5f)return true; // Preserve the existing full-rank fallback result.
        DenseProduct<true>(mat[K*DP],residual,mat);           // (X A)^T X
        DenseProduct<false,true>(mat,mat[K*DP],corrected);    // (X A)^T X (A X)^T
        DenseProduct(residual,original,mat);                 // A X_new
        Muls(residual,residual,-1.0f,K*DP);PipeBarrier<PIPE_ALL>();
        for(uint32_t r=0;r<K;++r)residual.SetValue(r*DP+r,residual.GetValue(r*DP+r)+1.0f);
        PipeBarrier<PIPE_ALL>();DenseProduct(corrected,mat,residual);
        Add(mat,mat,corrected,K*DP);PipeBarrier<PIPE_ALL>();
        DataCopyPad(output[base],mat,DataCopyExtParams{(uint16_t)size,size*4,(DP-(size+7)/8*8)/8,0,0});PipeBarrier<PIPE_ALL>();return true;
    }

    template<bool PolishOutput> __aicore__ inline bool RunArena(GM_ADDR a,GM_ADDR y,const LinalgPinvTilingData &tiling,LocalTensor<float> arena,GM_ADDR tables=nullptr) {
        t=tiling;size=t.n;staticTiling=tables;
        mat=arena;original=arena[512];xt=arena[768];residual=arena[1024];products=arena[1280];
        corrected=arena[5376];weights=arena[5632];pivot=arena[5760];col=arena[5792];
        indices=arena[5808].ReinterpretCast<uint32_t>();scalar=arena[6064];reduce=arena[6128];partial=arena[6384];repeated=arena[6896];
        input.SetGlobalBuffer((__gm__ float*)a);output.SetGlobalBuffer((__gm__ float*)y);
        bool ok=true;
        if constexpr(PolishOutput){for(uint32_t b=GetBlockIdx();b<t.batches;b+=GetBlockNum())Polish(b);}
        else {for(uint32_t b=GetBlockIdx();b<t.batches;b+=GetBlockNum())if(!Batch(b)){ok=false;break;}}
        PipeBarrier<PIPE_ALL>();return ok;
    }
public:
    template<bool PolishOutput=false> __aicore__ inline bool Run(GM_ADDR a,GM_ADDR y,const LinalgPinvTilingData &tiling,GM_ADDR tables=nullptr) {
        if constexpr(!PolishOutput) {
            // This fast path has one fixed UB arena and no queue/event allocation.
            LocalTensor<float> arena(TPosition::VECCALC,0,7920);
            return RunArena<PolishOutput>(a,y,tiling,arena,tables);
        } else {
            TPipe pipe;TBuf<TPosition::VECCALC> buffer;pipe.InitBuffer(buffer,7920*4);
            bool ok=RunArena<PolishOutput>(a,y,tiling,buffer.Get<float>());pipe.Destroy();return ok;
        }
    }
};

template<uint32_t Capacity> __aicore__ __attribute__((noinline)) void TinyCheckedFallback(
    GM_ADDR a,GM_ADDR y,GM_ADDR workspace,const LinalgPinvTilingData &t) {
    {Pinv<float,true> fallback;fallback.Run(a,y,workspace,t);}
    {ResidentTinyInverse refinement;refinement.Run<true>(a,y,t);}
}

template<uint32_t Capacity> __aicore__ inline void RunResidentTiny(GM_ADDR a,GM_ADDR y,GM_ADDR workspace,const LinalgPinvTilingData &t,GM_ADDR tables) {
    bool ok;{ResidentTinyInverse fast;ok=fast.Run(a,y,t,tables);}
    if(!ok)TinyCheckedFallback<Capacity>(a,y,workspace,t);
}

// Independent column shards avoid one device-wide handshake per pivot.
template<uint32_t Capacity> class ColumnSmallChecked {
    static constexpr uint32_t DP=Capacity,H=8,CP=DP+((DP/8)%2==0?8:0),L=DP+H;
    static constexpr uint32_t LIVE=DP+H,MATSIZE=LIVE*CP,DENSE=DP*DP;
    static constexpr uint32_t PRODUCTS=(5*H*DP+DP*H*8>2*DENSE+64?5*H*DP+DP*H*8:2*DENSE+64);
    uint32_t size;
    TPipe pipe;TBuf<TPosition::VECCALC> buffer;
    // Left matrix is column-major; right contains this core's output columns.
    // Four independent accumulators avoid a dependency after every outer product.
    __aicore__ inline void ProductColumns(const LocalTensor<float>& left,const LocalTensor<float>& right,
        const LocalTensor<uint32_t>& map,const LocalTensor<float>& work,uint32_t width) {
        auto accum=work,coeff=work[4*H*DP],expanded=work[5*H*DP];
        Gather(coeff,right,map,0,size*H);Duplicate(accum,0.0f,4*H*DP);PipeBarrier<PIPE_V>();
        Brcb(expanded,coeff,size*H/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t j=0;j+3<size;j+=4) {
            #pragma unroll
            for(uint32_t slot=0;slot<4;++slot)
                MulAddDst(accum[slot*H*DP],left[(j+slot)*DP],expanded[(j+slot)*H*8],(uint64_t)DP,(uint8_t)width,BinaryRepeatParams{1,1,0,DP/8,0,1});
            PipeBarrier<PIPE_V>();
        }
        for(uint32_t j=(size/4)*4;j<size;++j)
            MulAddDst(accum[(j%4)*H*DP],left[j*DP],expanded[j*H*8],(uint64_t)DP,(uint8_t)width,BinaryRepeatParams{1,1,0,DP/8,0,1});
        PipeBarrier<PIPE_V>();
        Add(accum,accum,accum[H*DP],H*DP);
        Add(accum[2*H*DP],accum[2*H*DP],accum[3*H*DP],H*DP);PipeBarrier<PIPE_V>();
        Add(accum,accum,accum[2*H*DP],H*DP);PipeBarrier<PIPE_V>();
    }
public:
    __aicore__ inline bool Run(GM_ADDR a,GM_ADDR y,GM_ADDR workspace,const LinalgPinvTilingData &t) {
        size=t.n;
        constexpr uint32_t FloatCount=MATSIZE+2*DENSE+2*H*DP+PRODUCTS+512+64+512;
        // Coefficients follow the same column stride as the update destination.
        // Offset the coefficient plane by DP elements from a 512-byte boundary.
        constexpr uint32_t WeightBase=(FloatCount+127)/128*128+DP;
        pipe.InitBuffer(buffer,(WeightBase+MATSIZE)*4);auto arena=buffer.Get<float>();
        auto mat=arena,xrows=mat[MATSIZE],xp=xrows[DENSE],rp=xp[H*DP],products=rp[H*DP];
        auto indices=products[PRODUCTS].template ReinterpretCast<uint32_t>();
        auto meta=products[PRODUCTS+DENSE],scalar=meta[512],reduce=scalar[64];
        auto pivot=xrows,col=pivot[L],active=col[DP],candidates=active[DP],weights=arena[WeightBase];
        GlobalTensor<float> input,output,candidate;
        input.SetGlobalBuffer((__gm__ float*)a);output.SetGlobalBuffer((__gm__ float*)y);
        candidate.SetGlobalBuffer((__gm__ float*)workspace);
        uint32_t shards=GetBlockNum()/t.batches,own=(size+shards-1)/shards;
        uint32_t id=GetBlockIdx(),batch=id/shards,lane=id%shards,first=lane*own,width=size-first<own?size-first:own;
        uint32_t packet=own*DP+8;
        uint64_t base=(uint64_t)batch*size*size,workBase=(uint64_t)batch*shards*packet;
        // Aligned rows transfer directly. For a partial block, one contiguous
        // GM transfer avoids a separate unaligned read for every row.
        if(size==DP) {
            DataCopyPad(mat,input[base],DataCopyExtParams{(uint16_t)size,size*4,0,(CP-DP)/8,0},
                DataCopyPadExtParams<float>{false,0,0,0.0f});
        } else {
            uint32_t zeroOffset=(size*size+7)/8*8;
            DataCopyPad(xrows,input[base],DataCopyExtParams{1,size*size*4,0,0,0},
                DataCopyPadExtParams<float>{true,0,(uint8_t)((8-size*size%8)%8),0.0f});
            Duplicate(xrows[zeroOffset],0.0f,8);
            auto map=indices.template ReinterpretCast<int32_t>(),rows=reduce.template ReinterpretCast<int32_t>(),cols=rows[DP];
            CreateVecIndex(rows,int32_t(0),DP);CreateVecIndex(cols,int32_t(0),DP);PipeBarrier<PIPE_V>();
            Muls(rows,rows,int32_t(size*4),DP);ShiftLeft(cols,cols,int32_t(2),DP);PipeBarrier<PIPE_V>();
            for(uint32_t c=0;c<DP;c+=8)Brcb(map[c],rows,DP/8,BrcbRepeatParams{DP/8,DP});
            PipeBarrier<PIPE_V>();Add(map,map,cols,(uint64_t)DP,(uint8_t)size,BinaryRepeatParams{1,1,1,DP/8,DP/8,0});PipeBarrier<PIPE_V>();
            uint64_t padMask[2]={((uint64_t(1)<<(DP-1))*2-1)&~((uint64_t(1)<<size)-1),0};
            Duplicate(map,int32_t(zeroOffset*4),padMask,(uint8_t)size,1,DP/8);PipeBarrier<PIPE_ALL>();
            if constexpr(CP==DP)Gather(mat,xrows,indices,0,size*DP);
            else {
                Gather(products,xrows,indices,0,size*DP);PipeBarrier<PIPE_V>();
                DataCopy(mat,products,DataCopyParams{(uint16_t)size,DP/8,0,(CP-DP)/8});
            }
        }
        Duplicate(mat[size*CP],0.0f,(LIVE-size)*CP);PipeBarrier<PIPE_ALL>();
        DataCopy(products,mat,DataCopyParams{(uint16_t)size,DP/8,(CP-DP)/8,0});PipeBarrier<PIPE_ALL>();
        for(uint32_t c=0;c<width;++c)mat.SetValue((size+c)*CP+first+c,1.0f);
        auto columnMap=indices.template ReinterpretCast<int32_t>();
        CreateVecIndex(columnMap,int32_t(0),LIVE);PipeBarrier<PIPE_V>();
        Muls(columnMap,columnMap,int32_t(CP*4),LIVE);PipeBarrier<PIPE_ALL>();
        auto diagonal=products[DENSE];
        auto unit=products[DENSE+64];Duplicate(unit,0.0f,DENSE);PipeBarrier<PIPE_V>();
        #pragma unroll 8
        for(uint32_t j=0;j<size;++j) {
            uint64_t m[2]={uint64_t(1)<<(j%8),0};
            Duplicate(unit[j*DP+j/8*8],1.0f,m,1,1,8);
        }
        PipeBarrier<PIPE_V>();
        bool valid=true;
        #pragma unroll 8
        for(uint32_t j=0;j<size;++j) {
            uint32_t live=j<first?size:size+width;
            uint32_t gathered=(live+7)/8*8;
            // Only the diagonal's containing block is broadcast; the full column
            // and pivot-row gather are independent of it.
            Brcb(diagonal,mat[j*CP+j/8*8],1,BrcbRepeatParams{1,8});
            uint32_t firstWeight=(j+1)/8*8;
            Gather(pivot[firstWeight],mat[j],indices[firstWeight],0,gathered-firstWeight);
            Sub(col,unit[j*DP],mat[j*CP],(uint64_t)DP,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            // Broadcast resets the mask. Divide sets the DP mask afterwards,
            // and the independent column update can reuse it.
            Brcb(weights[firstWeight*CP],pivot[firstWeight],(gathered-firstWeight)/8,BrcbRepeatParams{CP/8,CP});
            Div(col,col,diagonal[j%8*8],(uint64_t)DP,1,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();
            uint32_t firstColumn=j+1;
            if(firstColumn<live)
                MulAddDst<float,float,false>(mat[firstColumn*CP],col,weights[firstColumn*CP],(uint64_t)DP,(uint8_t)(live-firstColumn),BinaryRepeatParams{1,1,0,CP/8,0,CP/8});
            PipeBarrier<PIPE_V>();
        }
        // Keep each inverse column and its residual with the same owner.
        Duplicate(xp,0.0f,H*DP);Duplicate(rp,0.0f,H*DP);PipeBarrier<PIPE_ALL>();
        if(valid) {
            DataCopy(xp,mat[size*CP],DataCopyParams{(uint16_t)width,DP/8,(CP-DP)/8,0});
            PipeBarrier<PIPE_ALL>();
        }
        Mul(products[DENSE],xp,xp,width*DP);PipeBarrier<PIPE_V>();ReduceSum(scalar,products[DENSE],reduce,width*DP);PipeBarrier<PIPE_ALL>();float localX2=scalar.GetValue(0);
        DataCopy(mat,products,size*DP);PipeBarrier<PIPE_ALL>();
        Mul(products,mat,mat,size*DP);PipeBarrier<PIPE_V>();ReduceSum(scalar,products,reduce,size*DP);PipeBarrier<PIPE_ALL>();float af2=scalar.GetValue(0);
        BinaryRepeatParams broadcast{1,1,1,DP/8,DP/8,0};
        {
        auto rowIndices=indices.template ReinterpretCast<int32_t>(),packed=rowIndices[DP],steps=reduce.template ReinterpretCast<int32_t>();
        CreateVecIndex(rowIndices,int32_t(0),DP);CreateVecIndex(steps,int32_t(0),H);PipeBarrier<PIPE_V>();
        ShiftLeft(rowIndices,rowIndices,int32_t(2),DP);
        Muls(steps,steps,int32_t(DP*4),H);PipeBarrier<PIPE_V>();
        Brcb(packed,rowIndices,DP/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
        Add(packed,packed,steps,(uint64_t)64,(uint8_t)(DP*H/64),BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_ALL>();
        }
        if(valid) {
            ProductColumns(mat,xp,indices[DP],products,width);
            Muls(rp,products,-1.0f,width*DP);PipeBarrier<PIPE_ALL>();
            for(uint32_t c=0;c<width;++c)rp.SetValue(c*DP+first+c,rp.GetValue(c*DP+first+c)+1.0f);
            PipeBarrier<PIPE_ALL>();
        }
        Mul(products,rp,rp,width*DP);PipeBarrier<PIPE_V>();ReduceSum(scalar,products,reduce,width*DP);PipeBarrier<PIPE_ALL>();float localR2=scalar.GetValue(0);
        // One packet holds this column panel and its norm statistics.
        DataCopy(products,xp,own*DP);Duplicate(products[own*DP],0.0f,8);PipeBarrier<PIPE_ALL>();
        products.SetValue(own*DP,localX2);products.SetValue(own*DP+1,localR2);PipeBarrier<PIPE_ALL>();
        DataCopy(candidate[workBase+lane*packet],products,packet);SyncAll<true>();
        DataCopy(mat,candidate[workBase],shards*packet);PipeBarrier<PIPE_ALL>();
        constexpr uint32_t StatSlots=PINV_COLUMN_MAX_SHARDS;
        static_assert(StatSlots==8 || StatSlots==16);
        auto statIndex=reduce.template ReinterpretCast<int32_t>();
        CreateVecIndex(statIndex,int32_t(0),StatSlots);PipeBarrier<PIPE_V>();
        Muls(statIndex,statIndex,int32_t(packet*4),StatSlots);PipeBarrier<PIPE_V>();
        Adds(statIndex,statIndex,int32_t(own*DP*4),StatSlots);PipeBarrier<PIPE_V>();
        Adds(statIndex[StatSlots],statIndex,int32_t(4),StatSlots);PipeBarrier<PIPE_V>();
        // Read only the existing owners in both statistics planes.
        uint64_t statMask[2]={(uint64_t(1)<<shards)-1,0};
        statMask[0]|=statMask[0]<<StatSlots;
        Gather(scalar,mat,statIndex.template ReinterpretCast<uint32_t>(),0,statMask,1,8);PipeBarrier<PIPE_V>();
        WholeReduceSum(reduce[64],scalar,(uint64_t)shards,2,8,1,StatSlots/8);PipeBarrier<PIPE_ALL>();
        float xf2=reduce.GetValue(64),rf2=reduce.GetValue(72);
        float cut2=t.rtol*t.rtol*af2;if(t.atol*t.atol>cut2)cut2=t.atol*t.atol;
        valid=xf2>0.0f && xf2<=1e6f && cut2*xf2<0.49f && rf2<0.0625f;
        if(!valid){pipe.Destroy();return false;}
        if(xf2*rf2>=3.6e-9f) {
            if(shards*own<=DP)DataCopy(xrows,mat,DataCopyParams{(uint16_t)shards,(uint16_t)(own*DP/8),1,0});
            else {
                DataCopy(xrows,mat,DataCopyParams{(uint16_t)(shards-1),(uint16_t)(own*DP/8),1,0});
                DataCopy(xrows[(shards-1)*own*DP],mat[(shards-1)*packet],(size-(shards-1)*own)*DP);
            }
            PipeBarrier<PIPE_ALL>();
            ProductColumns(xrows,rp,indices[DP],products,width);
            Add(xp,xp,products,width*DP);PipeBarrier<PIPE_ALL>();
        }
        if(size==DP)DataCopy(output[base+first*size],xp,width*DP);
        else DataCopyPad(output[base+first*size],xp,DataCopyExtParams{(uint16_t)width,size*4,0,0,0});
        PipeBarrier<PIPE_ALL>();
        pipe.Destroy();return true;
    }
};

template<uint32_t Capacity> __aicore__ __attribute__((noinline,aligned(4096))) bool RunColumnFast(
    GM_ADDR a,GM_ADDR y,GM_ADDR workspace,const LinalgPinvTilingData &t) {
    static_assert(Capacity>=24 && Capacity<=64 && Capacity%8==0);
    ColumnSmallChecked<Capacity> fast;return fast.Run(a,y,workspace,t);
}

template<uint32_t Capacity> __aicore__ inline void RunColumnCapacity(
    GM_ADDR a,GM_ADDR y,GM_ADDR workspace,const LinalgPinvTilingData &t) {
    static_assert(Capacity==64);
    bool ok=false;
    switch((t.n+7)/8) {
        case 3:ok=RunColumnFast<24>(a,y,workspace,t);break;
        case 4:ok=RunColumnFast<32>(a,y,workspace,t);break;
        case 5:ok=RunColumnFast<40>(a,y,workspace,t);break;
        case 6:ok=RunColumnFast<48>(a,y,workspace,t);break;
        case 7:ok=RunColumnFast<56>(a,y,workspace,t);break;
        case 8:ok=RunColumnFast<64>(a,y,workspace,t);break;
    }
    uint32_t shards=GetBlockNum()/t.batches;
    if(!ok && GetBlockIdx()%shards==0){Pinv<float,true> fallback;fallback.Run(a,y,workspace,t,0,nullptr,GetBlockIdx()/shards);}
}

// A block-row inverse seed. Capacity follows the 64-row / 256-inner FP32
// Cube tile and per-core UB budget; the logical order is runtime data.
namespace RowBlock {
constexpr uint32_t CAP=PINV_ROW_BLOCK_CAPACITY,BLOCK=PINV_ROW_BLOCK_PANEL,PLANE=CAP*CAP;
static_assert(CAP==256 && BLOCK==64);
constexpr uint32_t W=0,T=PLANE,D=2*PLANE,A=3*PLANE,X=4*PLANE;
constexpr uint32_t G=5*PLANE,GI=G+4096,H=GI+4096,STATS=H+CAP*BLOCK,PACK=STATS+512,IDENTITY=PACK+12288,STRIDE=IDENTITY+8192;
static_assert(STRIDE==PINV_ROW_BLOCK_WORKSPACE_FLOATS);

class Cube {
    GlobalTensor<half> identity;
    TPipe pipe;TBuf<TPosition::A1> l1;TBuf<TPosition::A2> l0a;
    TBuf<TPosition::B2> l0b;TBuf<TPosition::CO1> l0c;
    template<bool TA,bool TB,bool Accumulate,bool AddIdentity=false>
    __aicore__ inline void Tile(GlobalTensor<float> aa,GlobalTensor<float> bb,GlobalTensor<float> cc,
        uint32_t m,uint32_t n,uint32_t k,uint32_t as,uint32_t bs,uint32_t cs,bool diagonal,GlobalTensor<half> unit) {
        auto la=l1.Get<float>(),lb=la[16384],av=l0a.Get<float>(),bv=l0b.Get<float>(),cv=l0c.Get<float>();
        const uint32_t limit=(m>64 || n>64)?128:256;
        for(uint32_t offset=0;offset<k;offset+=limit) {
            uint32_t bk=k-offset<limit?k-offset:limit;
        PipeBarrier<PIPE_ALL>();uint8_t pad[4]={};SetFmatrix(1,bk,pad,FmatrixMode::FMATRIX_LEFT);
        Nd2NzParams p{};p.ndNum=1;p.nValue=TA?bk:m;p.dValue=TA?m:bk;p.srcDValue=as;p.dstNzC0Stride=TA?bk:m;p.dstNzNStride=1;
        DataCopy(la,aa[TA?offset*as:offset],p);p.nValue=TB?n:bk;p.dValue=TB?bk:n;p.srcDValue=bs;p.dstNzC0Stride=TB?n:bk;DataCopy(lb,bb[TB?offset:offset*bs],p);
        SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
        if constexpr(TA) {LoadData3DParamsV2Pro q{};q.channelSize=m;q.extConfig=((uint64_t)bk<<16)|m;q.enTranspose=true;LoadData<float>(av,la,q);}
        else {LoadData2DParams q{};q.repeatTimes=bk/8;q.srcStride=m/16;for(uint32_t r=0;r<m/16;++r)LoadData(av[r*bk*16],la[r*128],q);}
        if constexpr(TB) {LoadData2DParams q{};q.repeatTimes=bk*n/128;q.srcStride=1;LoadData(bv,lb,q);}
        else {LoadData3DParamsV2Pro q{};q.channelSize=n;q.extConfig=((uint64_t)bk<<16)|n;LoadData<float>(bv,lb,q);}
        SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
        MmadParams mm{};mm.m=m;mm.n=n;mm.k=bk;mm.cmatrixInitVal=offset==0;Mmad(cv,av,bv,mm);
        }

        if constexpr(AddIdentity)if(diagonal) {
            // Only square diagonal output tiles receive this exact half-precision identity.
            PipeBarrier<PIPE_ALL>();auto ih=l1.Get<half>(),ia=l0a.Get<half>(),ib=l0b.Get<half>();
            Nd2NzParams p{};p.ndNum=1;p.nValue=m;p.dValue=m;p.srcDValue=128;p.dstNzC0Stride=m;p.dstNzNStride=1;
            DataCopy(ih,unit,p);SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
            uint8_t pad[4]={};SetFmatrix(1,m,pad,FmatrixMode::FMATRIX_LEFT);
            LoadData2DParams q{};q.repeatTimes=m/16;q.srcStride=m/16;
            for(uint32_t row=0;row<m/16;++row)LoadData(ia[row*m*16],ih[row*256],q);
            LoadData3DParamsV2Pro z{};z.channelSize=m;z.extConfig=((uint64_t)m<<16)|m;LoadData<half>(ib,ih,z);
            SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
            MmadParams mm{};mm.m=m;mm.n=n;mm.k=m;mm.cmatrixInitVal=false;Mmad(cv,ia,ib,mm);
        }
        SetFlag<HardEvent::M_FIX>(0);WaitFlag<HardEvent::M_FIX>(0);
        FixpipeParamsV220 f{};f.mSize=m;f.nSize=n;f.srcStride=m;f.dstStride=cs;
        if constexpr(Accumulate)SetAtomicAdd<float>();
        Fixpipe<float,float,CFG_ROW_MAJOR>(cc,cv,f);PipeBarrier<PIPE_ALL>();
        if constexpr(Accumulate)SetAtomicNone();
    }
public:
    __aicore__ inline void Init(GM_ADDR workspace){identity.SetGlobalBuffer((__gm__ half*)workspace+IDENTITY*2);pipe.InitBuffer(l1,262144);pipe.InitBuffer(l0a,65536);pipe.InitBuffer(l0b,65536);pipe.InitBuffer(l0c,131072);SetHF32Mode(false);SetHF32TransMode(false);}
    __aicore__ inline void End(){PipeBarrier<PIPE_ALL>();pipe.Destroy();}
    __aicore__ inline void Barrier(){PipeBarrier<PIPE_ALL>();ffts_cross_core_sync(PIPE_FIX,GetffstMsg(0x0,SYNC_AIC_FLAG));wait_flag_dev(SYNC_AIC_FLAG);}
    template<uint32_t K,uint32_t AS,uint32_t BS,bool Cached=false,bool Accumulate=false>
    __aicore__ inline void HalfProduct(LocalTensor<half> la,LocalTensor<half> lb,LocalTensor<float> c) {
        PipeBarrier<PIPE_ALL>();auto a=l0a.Get<half>()[Cached?0:8192],b=l0b.Get<half>();
        uint8_t pad[4]={};SetFmatrix(1,BS,pad,FmatrixMode::FMATRIX_LEFT);
        if constexpr(!Cached) {
            LoadData2DParams q{};q.repeatTimes=K/16;q.srcStride=AS/16;
            for(uint32_t r=0;r<4;++r)LoadData(a[r*K*16],la[r*256],q);
        }
        LoadData3DParamsV2Pro z{};z.channelSize=64;z.extConfig=((uint64_t)K<<16)|64;LoadData<half>(b,lb,z);
        SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
        MmadParams m{};m.m=64;m.n=64;m.k=K;m.cmatrixInitVal=!Accumulate;Mmad(c,a,b,m);
        PipeBarrier<PIPE_ALL>();
    }
    template<uint32_t DS> __aicore__ inline void HalfStore(LocalTensor<half> dst,LocalTensor<float> src) {
        FixpipeParamsV220 f{};f.mSize=64;f.nSize=64;f.srcStride=64;f.dstStride=DS;f.quantPre=QuantMode_t::F322F16;
        Fixpipe<half,float,CFG_NZ>(dst,src,f);PipeBarrier<PIPE_ALL>();
    }
    template<uint32_t Slot,uint32_t AS>
    __aicore__ inline void LoadPair64(LocalTensor<half> la,LocalTensor<half> lb) {
        auto a=l0a.Get<half>()[Slot*4096],b=l0b.Get<half>()[Slot*4096];
        uint8_t pad[4]={};SetFmatrix(1,64,pad,FmatrixMode::FMATRIX_LEFT);
        LoadData2DParams q{};q.repeatTimes=4;q.srcStride=AS/16;
        for(uint32_t row=0;row<4;++row)LoadData(a[row*1024],la[row*256],q);
        LoadData3DParamsV2Pro z{};z.channelSize=64;z.extConfig=((uint64_t)64<<16)|64;LoadData<half>(b,lb,z);
        SetFlag<HardEvent::MTE1_M>(Slot);WaitFlag<HardEvent::MTE1_M>(Slot);
    }
    __aicore__ inline void NeumannPair64(LocalTensor<half> x,LocalTensor<half> r,LocalTensor<float> y,LocalTensor<float> tmp,bool square) {
        LoadPair64<0,192>(x,r);auto a=l0a.Get<half>(),b=l0b.Get<half>();
        MmadParams m{};m.m=64;m.n=64;m.k=64;m.cmatrixInitVal=false;Mmad(y,a,b,m);SetFlag<HardEvent::M_FIX>(0);
        if(square){LoadPair64<1,64>(r,r);m.cmatrixInitVal=true;Mmad(tmp,a[4096],b[4096],m);SetFlag<HardEvent::M_FIX>(1);}
        WaitFlag<HardEvent::M_FIX>(0);
        FixpipeParamsV220 f{};f.mSize=64;f.nSize=64;f.srcStride=64;f.dstStride=192;f.quantPre=QuantMode_t::F322F16;
        Fixpipe<half,float,CFG_NZ>(x,y,f);
        if(square){WaitFlag<HardEvent::M_FIX>(1);f.dstStride=64;Fixpipe<half,float,CFG_NZ>(r,tmp,f);}
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Newton64(GlobalTensor<half> packed,GlobalTensor<float> result,GlobalTensor<float> defect,uint32_t batches,uint32_t stride,uint32_t steps) {
        auto weights=l1.Get<half>(),values=weights[12288],x=values[1024],r=weights[24576],low=weights[8192];auto y=l0c.Get<float>(),tmp=y[4096];
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=GetBlockNum()) {
            Nd2NzParams p{};p.ndNum=1;p.nValue=64;p.dValue=64;p.srcDValue=64;p.dstNzC0Stride=64;p.dstNzNStride=1;
            DataCopy(weights,packed[batch*stride*2+20480],p);DataCopy(weights[4096],packed[batch*stride*2],p);DataCopy(low,packed[batch*stride*2+16384],p);DataCopy(r,packed[batch*stride*2+12288],p);
            p.dstNzC0Stride=192;
            DataCopy(values,packed[batch*stride*2+8192],p);DataCopy(x,packed[batch*stride*2+4096],p);PipeBarrier<PIPE_ALL>();
            HalfProduct<64,192,192>(x,values,y);
            for(uint32_t iter=0;iter+1<steps;++iter) {
                NeumannPair64(x,r,y,tmp,iter+2<steps);
            }
            // Recompute the last Newton correction from G to remove residual-power drift.
            auto cachedA=l0a.Get<half>();LoadData2DParams q{};q.repeatTimes=8;q.srcStride=4;
            for(uint32_t row=0;row<4;++row)LoadData(cachedA[row*2048],weights[row*256],q);PipeBarrier<PIPE_ALL>();
            HalfProduct<128,64,192,true>(weights,values,tmp);HalfStore<64>(r,tmp);
            HalfProduct<64,192,64>(x,r,y);HalfStore<192>(x,y);
            // [I,G_hi,G_lo]*[I;Y;Y], then [Y,Y]*[I;E].
            HalfStore<192>(x[1024],y);
            q.repeatTimes=4;q.srcStride=12;
            for(uint32_t row=0;row<4;++row)LoadData(cachedA[row*3072],values[row*256],q);
            q.repeatTimes=8;q.srcStride=4;
            for(uint32_t row=0;row<4;++row)LoadData(cachedA[row*3072+1024],weights[4096+row*256],q);
            PipeBarrier<PIPE_ALL>();HalfProduct<192,64,192,true>(weights,values,tmp);
            q.repeatTimes=4;q.srcStride=12;
            for(uint32_t row=0;row<4;++row) {
                LoadData(cachedA[row*2048],x[row*256],q);
                LoadData(cachedA[row*2048+1024],x[row*256],q);
            }
            PipeBarrier<PIPE_ALL>();HalfStore<192>(x,tmp);
            HalfProduct<128,64,192,true>(weights,values,y);
            FixpipeParamsV220 f{};f.mSize=64;f.nSize=64;f.srcStride=64;f.dstStride=64;
            Fixpipe<float,float,CFG_ROW_MAJOR>(result[batch*stride],y,f);
            Fixpipe<float,float,CFG_ROW_MAJOR>(defect[batch*stride],tmp,f);PipeBarrier<PIPE_ALL>();
        }
    }


    template<bool TA=false,bool TB=false,bool Accumulate=false,bool AddIdentity=false>
    __aicore__ inline void Product(GlobalTensor<float> aa,GlobalTensor<float> bb,GlobalTensor<float> cc,
        uint32_t batches,uint32_t m,uint32_t n,uint32_t k,uint32_t as,uint32_t bs,uint32_t cs,uint32_t active=0xffffffffu) {
        uint32_t liveBatches=0;for(uint32_t i=0;i<batches;++i)if(active&(1u<<i))++liveBatches;
        uint32_t tile=(m>=128 && n>=128 && 5*liveBatches*((m+127)/128)*((n+127)/128)>=3*GetBlockNum())?128:64;
        uint32_t nr=(m+tile-1)/tile,nc=(n+tile-1)/tile;
        for(uint32_t task=GetBlockIdx();task<batches*nr*nc;task+=GetBlockNum()) {
            uint32_t batch=task/(nr*nc),row=task/nc%nr*tile,col=task%nc*tile;
            if(!(active&(1u<<batch)))continue;
            uint32_t tm=m-row<tile?m-row:tile,tn=n-col<tile?n-col:tile;
            Tile<TA,TB,Accumulate,AddIdentity>(aa[(uint64_t)batch*STRIDE+(TA?row:row*as)],bb[(uint64_t)batch*STRIDE+(TB?col*bs:col)],cc[(uint64_t)batch*STRIDE+row*cs+col],tm,tn,k,as,bs,cs,row==col,identity[(uint64_t)batch*STRIDE*2]);
        }
    }
};
class Vector {
    TPipe pipe;TBuf<TPosition::VECCALC> buffer;LocalTensor<float> w;
    GlobalTensor<float> mem,input,output;uint32_t n,N,batch,lane,teams;float scale=1.0f,normA2=0.0f,normX2=0.0f,normR2=0.0f;
    template<uint32_t C> __aicore__ __attribute__((noinline)) void Invert(uint32_t width) {
        constexpr uint32_t P=C+8;auto cm=w,unit=cm[2*C*P],diag=unit[C*C],pc=diag[64],pr=pc[C],cw=pr[2*C];
        auto ix=cw[2*C*8].template ReinterpretCast<uint32_t>();
        Duplicate(cm,0.0f,2*C*P);Duplicate(unit,0.0f,C*C);PipeBarrier<PIPE_ALL>();
        DataCopyPad(cm,mem[G],DataCopyExtParams{(uint16_t)width,width*4,(64-width)*4,(P-width)/8,0},DataCopyPadExtParams<float>{false,0,0,0});
        for(uint32_t j=0;j<C;++j){unit.SetValue(j*C+j,1.0f);ix.SetValue(j,j*P*4);ix.SetValue(C+j,(C+j)*P*4);}
        PipeBarrier<PIPE_ALL>();DataCopy(cm[C*P],unit,DataCopyParams{C,C/8,0,(P-C)/8});PipeBarrier<PIPE_V>();
        for(uint32_t j=0;j<width;++j) {
            Brcb(diag,cm[j*P+j/8*8],1,BrcbRepeatParams{1,8});
            Gather(pr,cm[j],ix,0,(C+j+8)/8*8);Sub(pc,unit[j*C],cm[j*P],C);PipeBarrier<PIPE_V>();
            Div(pc,pc,diag[j%8*8],(uint64_t)C,1,BinaryRepeatParams{1,1,0,8,8,0});
            Brcb(cw,pr,(C+j+8)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            MulAddDst(cm[(j+1)*P],pc,cw[(j+1)*8],(uint64_t)C,(uint8_t)C,BinaryRepeatParams{1,1,0,P/8,0,1});PipeBarrier<PIPE_V>();
        }
        Muls(cm[C*P],cm[C*P],-1.0f,C*P);PipeBarrier<PIPE_ALL>();
        DataCopyPad(mem[GI],cm[C*P],DataCopyExtParams{(uint16_t)width,width*4,(P-width)/8,(64-width)*4,0});PipeBarrier<PIPE_ALL>();
    }
public:
    __aicore__ inline void Init(GM_ADDR aa,GM_ADDR yy,GM_ADDR workspace,uint32_t order,uint32_t batches) {
        n=order;N=(n+15)/16*16;batch=GetBlockIdx()%batches;lane=GetBlockIdx()/batches;
        teams=(GetBlockNum()*2+batches-1-batch)/batches;
        pipe.InitBuffer(buffer,160*1024);w=buffer.Get<float>();
        if(lane<8){Duplicate(w[36864],0.0f,512);PipeBarrier<PIPE_ALL>();for(uint32_t j=0;j<8;++j)w.SetValue(36864+j*64+lane*8+j,1.0f);PipeBarrier<PIPE_ALL>();}
        mem.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)batch*STRIDE);
        input.SetGlobalBuffer((__gm__ float*)aa+(uint64_t)batch*n*n);output.SetGlobalBuffer((__gm__ float*)yy+(uint64_t)batch*n*n);
        GlobalTensor<half> unit;unit.SetGlobalBuffer((__gm__ half*)workspace+((uint64_t)batch*STRIDE+IDENTITY)*2);
        auto h=w.ReinterpretCast<half>();
        for(uint32_t row=lane*8;row<128;row+=teams*8) {
            Duplicate(h,half(0.0f),1024);PipeBarrier<PIPE_ALL>();
            for(uint32_t r=0;r<8;++r)h.SetValue(r*128+row+r,half(1.0f));
            PipeBarrier<PIPE_ALL>();DataCopy(unit[row*128],h,1024);PipeBarrier<PIPE_ALL>();
        }

    }
    __aicore__ inline void End(){PipeBarrier<PIPE_ALL>();pipe.Destroy();}
    __aicore__ inline void Load() {
        // One contiguous panel per Vector core, bounded by the existing
        // eight-core team and the 256-column UB arena.
        uint32_t panel=(N+teams*8-1)/(teams*8)*8,first=lane*panel;
        uint32_t rows=first<N?(N-first<panel?N-first:panel):0,count=rows*CAP;
        if(count) {
            Duplicate(w,0.0f,count);PipeBarrier<PIPE_ALL>();
            uint32_t logical=first<n?(n-first<rows?n-first:rows):0;
            if(logical)DataCopyPad(w,input[first*n],DataCopyExtParams{(uint16_t)logical,n*4,0,(CAP-(n+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-n%8)%8),0.0f});
        }
        PipeBarrier<PIPE_ALL>();auto scalar=w[20000],tmp=w[21000];Duplicate(scalar,0.0f,8);
        if(count){Abs(w[8192],w,count);PipeBarrier<PIPE_V>();ReduceMax(scalar,w[8192],tmp,count,false);}
        PipeBarrier<PIPE_ALL>();DataCopy(mem[STATS+lane*8],scalar,8);PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Normalize() {
        auto stats=w[20000];DataCopy(stats,mem[STATS],teams*8);PipeBarrier<PIPE_ALL>();float maximum=0.0f;
        for(uint32_t i=0;i<teams;++i){float v=stats.GetValue(i*8);if(v>maximum)maximum=v;}
        union{float f;uint32_t u;} bits;bits.f=maximum;int32_t e=(int32_t)((bits.u>>23)&255)-127;
        int32_t se=maximum>0?-e:0;se=se>126?126:(se<-126?-126:se);bits.u=(uint32_t)(127+se)<<23;scale=bits.f;
        uint32_t panel=(N+teams*8-1)/(teams*8)*8,first=lane*panel;
        uint32_t rows=first<N?(N-first<panel?N-first:panel):0,count=rows*CAP;
        if(count) {
            Muls(w,w,scale,count);Duplicate(w[16384],0.0f,count);PipeBarrier<PIPE_ALL>();
            for(uint32_t row=0;row<rows;++row) {
                w.SetValue(16384+row*CAP+first+row,1.0f);
                if(first+row>=n)w.SetValue(row*CAP+first+row,1.0f);
            }
            PipeBarrier<PIPE_ALL>();
            DataCopy(mem[W+first*CAP],w,count);DataCopy(mem[A+first*CAP],w,count);
            if(N<CAP)DataCopy(mem[X+first*CAP],w,count);
            DataCopy(mem[T+first*CAP],w[16384],count);PipeBarrier<PIPE_ALL>();
        }
        auto sc=w[22000],tmp=w[23000];Duplicate(sc,0.0f,8);PipeBarrier<PIPE_V>();
        if(count){Mul(w[8192],w,w,count);PipeBarrier<PIPE_V>();ReduceSum(sc,w[8192],tmp,count);}
        PipeBarrier<PIPE_ALL>();normA2=sc.GetValue(0);
    }
    __aicore__ inline void NewtonPrepare() {
        GlobalTensor<half> packed;packed.SetGlobalBuffer((__gm__ half*)mem.GetPhyAddr()+PACK*2);auto gram=mem[G];
        if(lane>=8)return;
        auto value=w,ab=w[4096],sum=w[8192],eye=w[12288];auto h=w[16384].ReinterpretCast<half>();
        DataCopy(value,gram,4096);PipeBarrier<PIPE_ALL>();Abs(ab,value,4096);PipeBarrier<PIPE_V>();
        WholeReduceSum(sum,ab,64,64,1,1,8);PipeBarrier<PIPE_V>();WholeReduceMax(sum[128],sum,64,1,1,1,8);PipeBarrier<PIPE_V>();
        PipeBarrier<PIPE_ALL>();float alpha=1.8f/sum.GetValue(128);
        for(uint32_t row=lane*8;row<64;row+=teams*8) {
            DataCopy(value,gram[row*64],512);PipeBarrier<PIPE_ALL>();
            Cast(h,value,RoundMode::CAST_RINT,512);PipeBarrier<PIPE_V>();
            Cast(ab,h,RoundMode::CAST_NONE,512);PipeBarrier<PIPE_V>();Sub(ab,value,ab,512);PipeBarrier<PIPE_V>();Cast(h[2048],ab,RoundMode::CAST_RINT,512);PipeBarrier<PIPE_V>();
            Muls(eye,w[36864],1.0f,512);PipeBarrier<PIPE_V>();
            Cast(h[1024],eye,RoundMode::CAST_RINT,512);PipeBarrier<PIPE_V>();Muls(ab,value,-alpha,512);PipeBarrier<PIPE_V>();Add(ab,eye,ab,512);PipeBarrier<PIPE_V>();Cast(h[1536],ab,RoundMode::CAST_RINT,512);PipeBarrier<PIPE_V>();Muls(value,eye,-alpha,512);PipeBarrier<PIPE_V>();Cast(h[512],value,RoundMode::CAST_RINT,512);PipeBarrier<PIPE_V>();
            Muls(value,eye,2.0f,512);PipeBarrier<PIPE_V>();Cast(h[2560],value,RoundMode::CAST_RINT,512);PipeBarrier<PIPE_ALL>();DataCopy(packed[row*64],h,DataCopyParams{4,32,0,224});DataCopy(packed[16384+row*64],h[2048],512);DataCopy(packed[20480+row*64],h[2560],512);PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ZeroDecision(GlobalTensor<int32_t> decision) {
        if(batch || lane)return;auto flag=w.ReinterpretCast<int32_t>();Duplicate(flag,(int32_t)0,8);PipeBarrier<PIPE_ALL>();DataCopy(decision,flag,8);PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void NewtonGuard(GlobalTensor<int32_t> decision) {
        auto dst=mem[H];
        auto ab=w[4096],sum=w[8192];
        for(uint32_t row=lane*8;row<64;row+=teams*8) {
            DataCopy(w,dst[row*64],512);PipeBarrier<PIPE_ALL>();
            Abs(ab,w,512);PipeBarrier<PIPE_V>();WholeReduceSum(sum,ab,64,8,1,1,8);PipeBarrier<PIPE_V>();
            WholeReduceMax(sum[128],sum,8,1,1,1,8);PipeBarrier<PIPE_V>();WholeReduceSum(sum[192],sum,8,1,1,1,8);PipeBarrier<PIPE_ALL>();
            float bound=sum.GetValue(128),total=sum.GetValue(192);
            if(!(bound<=0.015625f && total>=0.0f && total<=2.0f)) {
                auto flag=sum[128].ReinterpretCast<int32_t>();Duplicate(flag,(int32_t)0,8);PipeBarrier<PIPE_ALL>();flag.SetValue(0,1);PipeBarrier<PIPE_ALL>();
                SetAtomicAdd<int32_t>();DataCopy(decision,flag,8);PipeBarrier<PIPE_ALL>();SetAtomicNone();
            }
        }
    }
    __aicore__ inline void InvertBlock(uint32_t width){if(lane)return;if(width<=16)Invert<16>(width);else if(width<=32)Invert<32>(width);else Invert<64>(width);}
    __aicore__ inline void MeasureResidual(uint32_t active) {
        if(!(active&(1u<<batch)))return;
        normX2=0.0f;normR2=0.0f;
        uint32_t groups=lane*8<N?(N-lane*8+teams*8-1)/(teams*8):0,count=groups*8*CAP;
        if(!count)return;
        auto sc=w[24576],tmp=w[25088];
        DataCopyExtParams read{(uint16_t)groups,8*CAP*4,(teams-1)*8*CAP*4,0,0};
        DataCopyPad(w,mem[W+lane*8*CAP],read,DataCopyPadExtParams<float>{false,0,0,0});
        DataCopyPad(w[8192],mem[X+lane*8*CAP],read,DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
        PipeBarrier<PIPE_ALL>();
        Mul(w[16384],w,w,count);PipeBarrier<PIPE_V>();ReduceSum(sc,w[16384],tmp,count);PipeBarrier<PIPE_ALL>();normR2=sc.GetValue(0);
        Mul(w[16384],w[8192],w[8192],count);PipeBarrier<PIPE_V>();ReduceSum(sc,w[16384],tmp,count);PipeBarrier<PIPE_ALL>();normX2=sc.GetValue(0);
    }
    __aicore__ inline void Update(uint32_t active) {
        if(!(active&(1u<<batch)))return;
        uint32_t groups=lane*8<N?(N-lane*8+teams*8-1)/(teams*8):0,count=groups*8*CAP;
        float d2=0.0f;auto sc=w[24576],tmp=w[25088];
        if(count) {
            DataCopyExtParams read{(uint16_t)groups,8*CAP*4,(teams-1)*8*CAP*4,0,0};
            DataCopyPad(w,mem[X+lane*8*CAP],read,DataCopyPadExtParams<float>{false,0,0,0});
            DataCopyPad(w[8192],mem[T+lane*8*CAP],read,DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
            Mul(w[16384],w[8192],w[8192],count);PipeBarrier<PIPE_V>();ReduceSum(sc,w[16384],tmp,count);PipeBarrier<PIPE_ALL>();d2=sc.GetValue(0);
            Add(w,w,w[8192],count);PipeBarrier<PIPE_ALL>();
            DataCopyPad(mem[X+lane*8*CAP],w,DataCopyExtParams{(uint16_t)groups,8*CAP*4,0,(teams-1)*8*CAP*4,0});PipeBarrier<PIPE_ALL>();
        }
        Duplicate(sc,0.0f,8);PipeBarrier<PIPE_ALL>();sc.SetValue(0,normA2);sc.SetValue(1,normX2);sc.SetValue(2,normR2);sc.SetValue(3,d2);
        PipeBarrier<PIPE_ALL>();DataCopy(mem[STATS+lane*8],sc,8);PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Decide(uint32_t active,const LinalgPinvTilingData &t) {
        if(lane || !(active&(1u<<batch)))return;
        auto st=w[20000];DataCopy(st,mem[STATS],teams*8);PipeBarrier<PIPE_ALL>();float a2=0,x2=0,r2=0,d2=0;
        for(uint32_t i=0;i<teams;++i){a2+=st.GetValue(i*8);x2+=st.GetValue(i*8+1);r2+=st.GetValue(i*8+2);d2+=st.GetValue(i*8+3);}
        a2*=1.002001f;x2*=1.002001f;r2*=1.002001f;d2*=1.0201f;
        float s2=scale*scale,cut2=t.rtol*t.rtol*a2,absolute=t.atol*scale;if(absolute*absolute>cut2)cut2=absolute*absolute;
        // rho < 1/8. The promised discarded spectrum is <= 0.1*cutoff;
        // cutoff*||X|| < 9*(1-rho) certifies that none is discarded.
        // After this Newton step, ||error|| <= ||delta||*rho/(1-rho).
        // These computed-norm estimates retain the same rounding margins as
        // the previous checker; final tolerance is tested on actual outputs.
        bool ok=r2<0.015625f && x2>0.0f && x2*s2<1e6f && cut2*x2<62.015625f && d2*s2*r2<4.9e-9f;
        auto flag=w[24576].ReinterpretCast<uint32_t>();Duplicate(flag,uint32_t(0),8);PipeBarrier<PIPE_ALL>();flag.SetValue(0,ok?1u:0u);
        GlobalTensor<uint32_t> dst;dst.SetGlobalBuffer((__gm__ uint32_t*)mem.GetPhyAddr()+STATS+400);
        PipeBarrier<PIPE_ALL>();DataCopy(dst,flag,8);PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void StoreOutput() {
        for(uint32_t row=lane*8;row<n;row+=teams*8) {
            DataCopy(w,mem[X+row*CAP],8*CAP);PipeBarrier<PIPE_ALL>();Muls(w,w,-scale,8*CAP);PipeBarrier<PIPE_ALL>();
            uint32_t rows=n-row<8?n-row:8;
            DataCopyPad(output[row*n],w,DataCopyExtParams{(uint16_t)rows,n*4,(CAP-(n+7)/8*8)/8,0,0});PipeBarrier<PIPE_ALL>();
        }
    }
};

template<bool IsCube> __aicore__ inline uint32_t Run(GM_ADDR aa,GM_ADDR yy,GM_ADDR workspace,const LinalgPinvTilingData &t) {
    const uint32_t n=t.n,N=(n+15)/16*16,batches=t.batches;
    std::conditional_t<IsCube,Cube,Vector> op;GlobalTensor<float> mem;mem.SetGlobalBuffer((__gm__ float*)workspace);
    GlobalTensor<half> packed;packed.SetGlobalBuffer((__gm__ half*)workspace+PACK*2);
    GlobalTensor<int32_t> decision;decision.SetGlobalBuffer((__gm__ int32_t*)workspace+STATS);
    if constexpr(IsCube)op.Init(workspace);else op.Init(aa,yy,workspace,n,batches);
    if constexpr(!IsCube){op.Load();SyncAll<true>();op.Normalize();}SyncAll<false>();
    for(uint32_t p=0;p<N;p+=BLOCK) {
        uint32_t b=N-p<BLOCK?N-p:BLOCK;
        if constexpr(IsCube)op.template Product<false,true>(mem[W+p*CAP],mem[W+p*CAP],mem[G],batches,b,b,N,CAP,CAP,64);

        SyncAll<false>();bool exact=N-p<2*BLOCK;
        if(!exact) {
            if constexpr(!IsCube){op.ZeroDecision(decision);op.NewtonPrepare();}SyncAll<false>();
            uint32_t remaining=N-p,steps=remaining>=4*BLOCK?6:remaining>=3*BLOCK?7:remaining>=2*BLOCK?9:14;
            if constexpr(IsCube)op.Newton64(packed,mem[GI],mem[H],batches,STRIDE,steps);
            SyncAll<false>();if constexpr(!IsCube)op.NewtonGuard(decision);SyncAll<false>();
            DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(decision);
            exact=decision.GetValue(0)!=0;
        }
        if constexpr(!IsCube){if(exact)op.InvertBlock(b);}SyncAll<false>();
        if constexpr(IsCube)op.template Product<>(mem[GI],mem[W+p*CAP],mem[D+p*CAP],batches,b,N,b,64,CAP,CAP);
        if constexpr(IsCube)op.Barrier();
        if(p+b==N) {
            // Refine the dual row block before forming the full inverse.
            for(uint32_t step=0;step<2;++step) {
                if constexpr(IsCube)op.template Product<false,true,false,true>(mem[D+p*CAP],mem[W+p*CAP],mem[G],batches,b,b,N,CAP,CAP,64);
                if constexpr(IsCube)op.Barrier();
                // One row tile owns every row of this block for each column tile.
                if constexpr(IsCube)op.template Product<false,false,true>(mem[G],mem[D+p*CAP],mem[D+p*CAP],batches,b,N,b,64,CAP,CAP);
                if constexpr(IsCube)op.Barrier();
            }
            // Cancel the last dual panel's response to earlier row panels.
            // Earlier negative duals satisfy D_prev * W_prev^T ~= -I.
            if constexpr(IsCube)if(p) {
                op.template Product<false,true>(mem[D+p*CAP],mem[W],mem[H],batches,b,p,N,CAP,CAP,CAP);
                op.Barrier();
                op.template Product<false,false,true>(mem[H],mem[D],mem[D+p*CAP],batches,b,N,p,CAP,CAP,CAP);
                op.Barrier();
            }
        } else {
            if constexpr(IsCube)op.template Product<false,true>(mem[W+(p+b)*CAP],mem[D+p*CAP],mem[H],batches,N-p-b,b,N,CAP,CAP,64);
            if constexpr(IsCube)op.Barrier();
            if constexpr(IsCube) {
                op.template Product<false,false,true>(mem[H],mem[W+p*CAP],mem[W+(p+b)*CAP],batches,N-p-b,N,b,64,CAP,CAP);
                op.template Product<false,false,true>(mem[H],mem[T+p*CAP],mem[T+(p+b)*CAP],batches,N-p-b,p+b,b,64,CAP,CAP);
                op.Barrier();
            }
        }
    }
    if constexpr(IsCube)op.template Product<true>(mem[D],mem[T],mem[X],batches,N,N,N,CAP,CAP,CAP);
    if constexpr(IsCube)op.Barrier();
    uint32_t accepted=0,active=(1u<<batches)-1;
    for(uint32_t iteration=0;iteration<4 && active;++iteration) {
        if constexpr(IsCube)op.template Product<false,false,false,true>(mem[A],mem[X],mem[W],batches,N,N,N,CAP,CAP,CAP,active);
        SyncAll<false>();
        if constexpr(IsCube)op.template Product<>(mem[X],mem[W],mem[T],batches,N,N,N,CAP,CAP,CAP,active);
        else op.MeasureResidual(active);
        SyncAll<false>();if constexpr(!IsCube){op.Update(active);SyncAll<true>();op.Decide(active,t);}SyncAll<false>();
        for(uint32_t b=0;b<batches;++b) {
            GlobalTensor<uint32_t> flag;flag.SetGlobalBuffer((__gm__ uint32_t*)workspace+(uint64_t)b*STRIDE+STATS+400);
            DataCacheCleanAndInvalid<uint32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(flag);
            if(flag.GetValue(0)==1)accepted|=1u<<b;
        }
        active=((1u<<batches)-1)&~accepted;
    }
    if constexpr(!IsCube)op.StoreOutput();SyncAll<false>();op.End();return accepted;
}
}

template<uint32_t Capacity> __aicore__ inline void RunRowBlockEntry(GM_ADDR a,GM_ADDR y,GM_ADDR workspace,GM_ADDR tiling) {
    static_assert(Capacity==RowBlock::CAP);
    GET_TILING_DATA_WITH_STRUCT(LinalgPinvTilingData,t,tiling);
    // The generated CANN wrapper has already selected the user workspace.
    GM_ADDR user=workspace;
    uint32_t accepted=0;
    if ASCEND_IS_AIC {accepted=RowBlock::Run<true>(a,y,user,t);}
    if ASCEND_IS_AIV {accepted=RowBlock::Run<false>(a,y,user,t);}
    if(accepted==((1u<<t.batches)-1))return;
    if ASCEND_IS_AIV {
        // The general vector implementation rebuilds its own workspace from A.
        // Its barriers involve only Vector cores; Cube cores have finished.
        Pinv<float,false,true> fallback;fallback.Run(a,y,user,t);
    }
}

#ifndef PINV_DIRECT_BENCH
template<uint32_t Capacity> __aicore__ __attribute__((noinline,aligned(4096))) void RunTinyKernel(
    GM_ADDR a,GM_ADDR y,GM_ADDR workspace,GM_ADDR tiling) {
    static_assert(Capacity==16);
    ICachePreLoad(5);
    GET_TILING_DATA_WITH_STRUCT(LinalgPinvTilingData,t,tiling);
    RunResidentTiny<Capacity>(a,y,workspace,t,tiling);
}

template<typename DT_A,uint32_t KERNEL_PATH>
__global__ __aicore__ void linalg_pinv(GM_ADDR A,GM_ADDR A_pinv,GM_ADDR workspace,GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    InitSocState();
    REGISTER_TILING_DEFAULT(LinalgPinvMaxTilingData);
    if constexpr(KERNEL_PATH==6) {
        GET_TILING_DATA_WITH_STRUCT(LinalgPinvTilingData,t,tiling);RunColumnCapacity<PINV_COLUMN_CAPACITY>(A,A_pinv,workspace,t);
    } else if constexpr(KERNEL_PATH==5) {
        RunTinyKernel<16>(A,A_pinv,workspace,tiling);
    } else if constexpr(KERNEL_PATH==4) {
        RunRowBlockEntry<PINV_ROW_BLOCK_CAPACITY>(A,A_pinv,workspace,tiling);
    } else {
        GET_TILING_DATA_WITH_STRUCT(LinalgPinvTilingData,t,tiling);
        Pinv<DT_A,KERNEL_PATH==0> op;op.Run(A,A_pinv,workspace,t);
    }
}

#endif
