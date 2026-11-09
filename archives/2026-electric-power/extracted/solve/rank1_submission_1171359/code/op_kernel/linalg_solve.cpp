#include "kernel_operator.h"
#include "linalg_solve_tiling.h"
#include "tiling_key_linalg_solve.h"
using namespace AscendC;
template<class DT_A, uint32_t N_VALUE> class KernelLinalgSolve {
public:
    template<HardEvent event> __aicore__ inline void Fence() {
        const event_t id=static_cast<event_t>(pipe.FetchEventID(event));
        SetFlag<event>(id);WaitFlag<event>(id);
    }
    __aicore__ inline void Init(GM_ADDR A,GM_ADDR B,GM_ADDR X,GM_ADDR work,const LinalgSolveTilingData& t) {
        n=N_VALUE?N_VALUE:t.n;k=t.k;batches=t.batches;broadcast=t.broadcastB;cores=t.cores;resident=t.resident;grouped=t.grouped;
        na=(n+7)/8*8;nk=(k+7)/8*8;width=na+nk;
        refine=sizeof(DT_A)==4 && n>64 && k<=8;
        a.SetGlobalBuffer((__gm__ DT_A*)A);b.SetGlobalBuffer((__gm__ DT_A*)B);x.SetGlobalBuffer((__gm__ DT_A*)X);
        if(!resident) w.SetGlobalBuffer((__gm__ float*)work+uint64_t(GetBlockIdx())*n*width);
        pipe.InitBuffer(matrixBuf,resident?n*width*4:(grouped?32*width*4:32));
        pipe.InitBuffer(pivotBuf,width*4);pipe.InitBuffer(rowBuf,width*4);pipe.InitBuffer(tempBuf,width*4);
        pipe.InitBuffer(colBuf,(n+7)/8*32);
        pipe.InitBuffer(ioBuf,(((n>k?n:k)+15)/16*16+16)*sizeof(DT_A));
        mat=matrixBuf.Get<float>();pivot=pivotBuf.Get<float>();row=rowBuf.Get<float>();
        temp=tempBuf.Get<float>();column=colBuf.Get<float>();io=ioBuf.Get<DT_A>();
        if(refine) {
            pipe.InitBuffer(solBuf,n*nk*4);pipe.InitBuffer(rhsBuf,n*nk*4);pipe.InitBuffer(xtBuf,k*na*4);
            sol=solBuf.Get<float>();rhs=rhsBuf.Get<float>();xt=xtBuf.Get<float>();
        }
    }
    __aicore__ inline void ReadInput(LocalTensor<float> dst,GlobalTensor<DT_A> src,uint32_t count) {
        DataCopyExtParams cp{1,count*uint32_t(sizeof(DT_A)),0,0,0};
        DataCopyPadExtParams<DT_A> pad{false,0,0,0};
        Fence<HardEvent::V_MTE2>();
        DataCopyPad(io,src,cp,pad);Fence<HardEvent::MTE2_V>();
        if constexpr(sizeof(DT_A)==4) Adds(dst,io.template ReinterpretCast<float>(),0.0f,count);
        else Cast(dst,io,RoundMode::CAST_NONE,count);
        Fence<HardEvent::V_S>();
    }
    __aicore__ inline void ReadRow(LocalTensor<float> dst,uint32_t i) {
        Fence<HardEvent::S_V>();
        if(resident) {
            Adds(dst,mat[i*width],0.0f,width);Fence<HardEvent::V_S>();
        } else {
            Fence<HardEvent::V_MTE2>();DataCopy(dst,w[uint64_t(i)*width],width);
            Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
        }
    }
    __aicore__ inline void WriteRow(LocalTensor<float> src,uint32_t i) {
        Fence<HardEvent::S_V>();
        if(resident) {
            Adds(mat[i*width],src,0.0f,width);PipeBarrier<PIPE_V>();
        } else {
            Fence<HardEvent::V_MTE3>();DataCopy(w[uint64_t(i)*width],src,width);
            Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_MTE2>();
        }
    }
    __aicore__ inline void Process() {
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=cores) {
            uint64_t aoff=uint64_t(batch)*n*n,boff=broadcast?0:uint64_t(batch)*n*k;
            for(uint32_t pass=0;pass<(refine?2u:1u);++pass) {
            for(uint32_t i=0;i<n;++i) {
                Duplicate(row,0.0f,width);PipeBarrier<PIPE_ALL>();
                ReadInput(row,a[aoff+uint64_t(i)*n],n);
                if(pass==0) ReadInput(row[na],b[boff+uint64_t(i)*k],k);
                else {Adds(row[na],rhs[i*nk],0.0f,nk);Fence<HardEvent::V_S>();}
                column.SetValue(i,row.GetValue(0));WriteRow(row,i);
            }
            for(uint32_t p=0;p<n;++p) {
                uint32_t best=p;float maxabs=0.0f;
                for(uint32_t i=p;i<n;++i) {
                    float v=column.GetValue(i),av=v<0?-v:v;
                    if(av>maxabs){maxabs=av;best=i;}
                }
                ReadRow(pivot,best);
                if(best!=p){ReadRow(row,p);WriteRow(row,best);}
                WriteRow(pivot,p);
                float inv=1.0f/pivot.GetValue(p);
                if(grouped) ForwardGrouped(p,inv);
                else for(uint32_t i=p+1;i<n;++i) {
                    ReadRow(row,i);
                    float factor=-row.GetValue(p)*inv;
                    Axpy(row,pivot,factor,width);Fence<HardEvent::V_S>();
                    if(p+1<n) column.SetValue(i,row.GetValue(p+1));
                    WriteRow(row,i);
                }
            }
            for(int32_t p=int32_t(n)-1;p>=0;--p) {
                ReadRow(pivot,p);float inv=1.0f/pivot.GetValue(p);
                Muls(pivot[na],pivot[na],inv,nk);PipeBarrier<PIPE_ALL>();
                if(refine) {
                    if(pass==0) Adds(sol[p*nk],pivot[na],0.0f,nk);
                    else Add(sol[p*nk],sol[p*nk],pivot[na],nk);
                    PipeBarrier<PIPE_V>();
                }
                if constexpr(sizeof(DT_A)==4) {
                    if(refine && pass==1) Adds(io.template ReinterpretCast<float>(),sol[p*nk],0.0f,k);
                    else Adds(io.template ReinterpretCast<float>(),pivot[na],0.0f,k);
                } else Cast(io,pivot[na],RoundMode::CAST_RINT,k);
                Fence<HardEvent::V_MTE3>();
                DataCopyExtParams cp{1,k*uint32_t(sizeof(DT_A)),0,0,0};
                DataCopyPad(x[uint64_t(batch)*n*k+uint64_t(p)*k],io,cp);Fence<HardEvent::MTE3_V>();
                if(grouped) BackGrouped(p);
                else for(uint32_t i=0;i<uint32_t(p);++i) {
                    ReadRow(row,i);float factor=-row.GetValue(p);
                    Axpy(row[na],pivot[na],factor,nk);PipeBarrier<PIPE_ALL>();
                    WriteRow(row,i);
                }
            }
            if(refine && pass==0) ComputeResidual(aoff,boff);
            }
        }
    }
private:
    __aicore__ inline LocalTensor<float> ReadTile(uint32_t base,uint32_t rows) {
        if(resident) return mat[base*width];
        Fence<HardEvent::V_MTE2>();
        DataCopy(mat,w[uint64_t(base)*width],rows*width);
        Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
        return mat;
    }
    __aicore__ inline void WriteTile(uint32_t base,uint32_t rows) {
        if(!resident) {
            Fence<HardEvent::V_MTE3>();
            DataCopy(w[uint64_t(base)*width],mat,rows*width);
            Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_MTE2>();
        }
    }
    __aicore__ inline void ForwardGrouped(uint32_t p,float inv) {
        for(uint32_t base=p+1;base<n;base+=32) {
            uint32_t rows=n-base<32?n-base:32;
            auto tile=ReadTile(base,rows);
            for(uint32_t j=0;j<rows;++j) {
                auto current=tile[j*width];
                float factor=-current.GetValue(p)*inv;
                Axpy(current,pivot,factor,width);Fence<HardEvent::V_S>();
                if(p+1<n) column.SetValue(base+j,current.GetValue(p+1));
            }
            WriteTile(base,rows);
        }
    }
    __aicore__ inline void BackGrouped(uint32_t p) {
        for(uint32_t base=0;base<p;base+=32) {
            uint32_t rows=p-base<32?p-base:32;
            auto tile=ReadTile(base,rows);
            for(uint32_t j=0;j<rows;++j) {
                auto current=tile[j*width];
                float factor=-current.GetValue(p);
                Axpy(current[na],pivot[na],factor,nk);Fence<HardEvent::V_S>();
            }
            WriteTile(base,rows);
        }
    }
    union FloatBits {float number;uint32_t bits;};
    __aicore__ inline float HighPart(float value) {
        FloatBits v;v.number=value;v.bits &= 0xfffff000u;return v.number;
    }
    __aicore__ inline void AddResidualTerm(float term,float& sum,float& comp) {
        // Error-free TwoSum plus accumulated low parts; volatile forbids reassociation.
        volatile float updated=sum+term;
        volatile float displaced=updated-sum;
        volatile float kept=updated-displaced;
        volatile float lowSum=sum-kept;
        volatile float lowTerm=term-displaced;
        volatile float error=lowSum+lowTerm;
        comp=comp+error;sum=updated;
    }
    __aicore__ inline void ComputeResidual(uint64_t aoff,uint64_t boff) {
        Fence<HardEvent::V_S>();
        for(uint32_t q=0;q<k;++q)
            for(uint32_t j=0;j<n;++j) xt.SetValue(q*na+j,sol.GetValue(j*nk+q));
        Fence<HardEvent::S_V>();
        Duplicate(rhs,0.0f,n*nk);Fence<HardEvent::V_S>();
        // Form B-A X in original row order; second solve refactors A from scratch.
        for(uint32_t i=0;i<n;++i) {
            uint32_t original=i;
            ReadInput(row,a[aoff+uint64_t(original)*n],n);
            ReadInput(row[na],b[boff+uint64_t(original)*k],k);
            for(uint32_t q=0;q<k;++q) {
                Fence<HardEvent::S_V>();
                Mul(temp,row,xt[q*na],n);Fence<HardEvent::V_S>();
                float sum=row.GetValue(na+q),comp=0.0f;
                for(uint32_t j=0;j<n;++j) {
                    float av=row.GetValue(j),xv=xt.GetValue(q*na+j);
                    float ah=HighPart(av),xh=HighPart(xv);
                    volatile float al=av-ah,xl=xv-xh;
                    float product=temp.GetValue(j);
                    volatile float hh=ah*xh;
                    volatile float first=hh-product;
                    volatile float hl=ah*xl;
                    volatile float second=first+hl;
                    volatile float lh=al*xh;
                    volatile float third=second+lh;
                    volatile float ll=al*xl;
                    volatile float productError=third+ll;
                    AddResidualTerm(-product,sum,comp);
                    AddResidualTerm(-productError,sum,comp);
                }
                rhs.SetValue(i*nk+q,sum+comp);
            }
        }
        Fence<HardEvent::S_V>();
    }
    TPipe pipe;
    TBuf<TPosition::VECCALC> matrixBuf,pivotBuf,rowBuf,tempBuf,colBuf,ioBuf;
    TBuf<TPosition::VECCALC> solBuf,rhsBuf,xtBuf;
    LocalTensor<float> sol,rhs,xt;
    bool refine;
    GlobalTensor<DT_A> a,b,x;GlobalTensor<float> w;
    LocalTensor<float> mat,pivot,row,temp,column;LocalTensor<DT_A> io;
    uint32_t n,k,batches,broadcast,cores,resident,grouped,na,nk,width;
};
template<uint32_t FAMILY=0,uint32_t STATIC_N=0,uint32_t STATIC_WORKERS=0,uint32_t STATIC_K=0,bool PARTIAL_CAPACITY=false> class KernelColumnSolve {
public:
    __aicore__ inline ~KernelColumnSolve() {PipeBarrier<PIPE_ALL>();}
    template<HardEvent event> __aicore__ inline void Fence() {
        const event_t id=static_cast<event_t>(0);
        SetFlag<event>(id);WaitFlag<event>(id);
    }
    __aicore__ inline void CreateVecIndex(LocalTensor<int32_t> target,int32_t first,uint32_t count) {
        uint32_t seed=count<8?count:8;
        for(uint32_t i=0;i<seed;++i)target.SetValue(i,first+int32_t(i));
        Fence<HardEvent::S_V>();
        for(uint32_t filled=8;filled<count;filled*=2) {
            uint32_t len=count-filled<filled?count-filled:filled;
            Adds(target[filled],target,int32_t(filled),len);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Init(GM_ADDR A,GM_ADDR B,GM_ADDR X,GM_ADDR work,const LinalgSolveTilingData& t) {
        n=STATIC_N?STATIC_N:t.n;k=STATIC_K?STATIC_K:t.k;na=(n+7)/8*8;batches=t.batches;cores=t.cores;
        logicalN=PARTIAL_CAPACITY?t.n:n;logicalK=PARTIAL_CAPACITY?t.k:k;
        parallel=STATIC_WORKERS?true:t.resident==3;workers=STATIC_WORKERS?STATIC_WORKERS:parallel?t.cores/t.batches:1;
        cp=parallel?na+8+(FAMILY==1 && (na/8)%2==1?8:0):na;
        firstResident=t.grouped;initialResident=firstResident;broadcast=t.broadcastB;
        a.SetGlobalBuffer((__gm__ float*)A);b.SetGlobalBuffer((__gm__ float*)B);x.SetGlobalBuffer((__gm__ float*)X);
        w.SetGlobalBuffer((__gm__ float*)work+(parallel?uint64_t(GetBlockIdx()/workers)*na*((FAMILY==1 || (FAMILY==3 && STATIC_N==256))?2*n+(FAMILY==1?48:((STATIC_WORKERS==10 && STATIC_K==3)?41:40)):n+40):uint64_t(GetBlockIdx())*na*(n+k)));
        if constexpr(FAMILY==1 || (FAMILY==3 && STATIC_N==256)) {
            publication.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(
                const_cast<__gm__ float*>(w.GetPhyAddr()))+(2*n+32)*na);
        }
        uint32_t matrixCols=parallel?(n<=200?192:144):n+k-firstResident;if(matrixCols<(parallel?144u:72u))matrixCols=parallel?144:72;
        if constexpr(FAMILY==3){if(parallel && n<=200)matrixCols=n<144?144:n;}
        spillGroup=n>=224?16:8;
        SetMaskNorm();SetAtomicNone();
        uint32_t offset=0;
        mat=LocalTensor<float>(TPosition::VECCALC,offset,na*matrixCols);offset+=na*matrixCols*4;
        uint32_t inputRows=parallel?(FAMILY==3?12:16):spillGroup;
        uint32_t inputSize=inputRows*na;
        if constexpr(FAMILY==3 && STATIC_N>0 && STATIC_N%16==0)inputSize=16*cp;
        input=LocalTensor<float>(TPosition::VECCALC,offset,inputSize);offset+=inputSize*4;
        transposed=LocalTensor<float>(TPosition::VECCALC,offset,8*na);offset+=8*na*4;
        indices=LocalTensor<uint32_t>(TPosition::VECCALC,offset,8*cp);offset+=8*cp*4;
        baseFactor=LocalTensor<float>(TPosition::VECCALC,offset,na+8);offset+=(na+8)*4;
        factor=baseFactor;
        column=LocalTensor<float>(TPosition::VECCALC,offset,na);offset+=na*4;
        order=LocalTensor<uint32_t>(TPosition::VECCALC,offset,na);offset+=na*4;
        output=LocalTensor<float>(TPosition::VECCALC,offset,n*8);offset+=n*8*4;
        inverse=LocalTensor<float>(TPosition::VECCALC,offset,na);
    }
    __aicore__ inline void LoadMatrix(GlobalTensor<float> source,uint64_t offset,uint32_t cols,uint32_t firstColumn) {
        const uint32_t stride=(cols+7)/8*8;
        auto sequence=input.ReinterpretCast<int32_t>();
        auto columnIndex=transposed.ReinterpretCast<int32_t>();
        auto offsets=indices.ReinterpretCast<int32_t>();
        CreateVecIndex(sequence,int32_t(0),cols*8);PipeBarrier<PIPE_V>();
        ShiftRight(columnIndex,sequence,int32_t(3),cols*8);PipeBarrier<PIPE_V>();
        Muls(offsets,columnIndex,int32_t(8),cols*8);PipeBarrier<PIPE_V>();
        Sub(offsets,sequence,offsets,cols*8);PipeBarrier<PIPE_V>();
        Muls(offsets,offsets,int32_t(stride*4),cols*8);
        Muls(columnIndex,columnIndex,int32_t(4),cols*8);PipeBarrier<PIPE_V>();
        Add(offsets,offsets,columnIndex,cols*8);PipeBarrier<PIPE_V>();
        for(uint32_t base=0;base<n;base+=8) {
            uint32_t rows=n-base<8?n-base:8;
            Duplicate(input,0.0f,8*stride);Fence<HardEvent::V_MTE2>();
            DataCopyExtParams cp{static_cast<uint16_t>(rows),cols*4,0,0,0};
            DataCopyPadExtParams<float> pad{false,0,0,0};
            DataCopyPad(input,source[offset+uint64_t(base)*cols],cp,pad);
            Fence<HardEvent::MTE2_V>();
            Gather(transposed,input,indices,0,cols*8);
            uint32_t spilled=firstResident>firstColumn?firstResident-firstColumn:0;
            if(spilled>cols) spilled=cols;
            if(spilled<cols) {
                DataCopyParams tp{static_cast<uint16_t>(cols-spilled),1,0,static_cast<uint16_t>((na-8)/8)};
                PipeBarrier<PIPE_V>();
                DataCopy(mat[(firstColumn+spilled-firstResident)*na+base],transposed[spilled*8],tp);
                PipeBarrier<PIPE_V>();
            }
            if(spilled) {
                DataCopyParams tp{static_cast<uint16_t>(spilled),1,0,static_cast<uint16_t>((na-8)/8)};
                Fence<HardEvent::V_MTE3>();DataCopy(w[firstColumn*na+base],transposed,tp);
                Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_MTE2>();
            }
        }
    }
    __aicore__ inline LocalTensor<float> ReadColumn(uint32_t j) {
        if(j>=firstResident) return mat[(j-firstResident)*na];
        Fence<HardEvent::V_MTE2>();DataCopy(column,w[j*na],na);
        Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
        return column;
    }
    __aicore__ inline void WriteColumn(uint32_t j,LocalTensor<float> value) {
        if(j<firstResident) {
            Fence<HardEvent::V_MTE3>();DataCopy(w[j*na],value,na);
            Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_MTE2>();
        }
    }
    __aicore__ inline void Process() {
        if(parallel){ProcessCooperative();return;}
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=cores) {
            firstResident=initialResident;
            LoadMatrix(a,uint64_t(batch)*n*n,n,0);
            LoadMatrix(b,broadcast?0:uint64_t(batch)*n*k,k,n);
            auto active=indices[na].ReinterpretCast<float>();
            auto absolute=indices[2*na].ReinterpretCast<float>();
            auto reduced=indices[3*na].ReinterpretCast<float>();
            auto baseIndex=indices.ReinterpretCast<int32_t>();
            CreateVecIndex(baseIndex,int32_t(0),64);PipeBarrier<PIPE_V>();
            Muls(baseIndex,baseIndex,int32_t(na*4),64);
            Duplicate(active,0.0f,na);PipeBarrier<PIPE_V>();Duplicate(active,1.0f,n);
            for(uint32_t p=0;p<n;++p) {
                auto current=ReadColumn(p);
                Fence<HardEvent::S_V>();Adds(factor,current,0.0f,na);PipeBarrier<PIPE_V>();
                Abs(absolute,factor,na);PipeBarrier<PIPE_V>();
                Mul(absolute,absolute,active,na);PipeBarrier<PIPE_V>();
                WholeReduceMax(reduced,absolute,64,na/64,4,1,8,ReduceOrder::ORDER_VALUE_INDEX);
                if(na%64) WholeReduceMax(reduced[(na/64)*8],absolute[(na/64)*64],na%64,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
                Fence<HardEvent::V_S>();
                float largest=-1.0f;uint32_t pivotRow=0;
                for(uint32_t base=0;base<na;base+=64) {
                    float value=reduced.GetValue(base/8);
                    if(value>largest){largest=value;pivotRow=base+reduced.ReinterpretCast<uint32_t>().GetValue(base/8+1);}
                }
                order.SetValue(p,pivotRow);active.SetValue(pivotRow,0.0f);
                float inv=1.0f/factor.GetValue(pivotRow);inverse.SetValue(p,inv);
                factor.SetValue(pivotRow,0.0f);Fence<HardEvent::S_V>();
                for(uint32_t j=p+1;j<firstResident;j+=spillGroup) {
                    uint32_t cols=firstResident-j<spillGroup?firstResident-j:spillGroup;
                    Fence<HardEvent::V_MTE2>();DataCopy(input,w[j*na],cols*na);Fence<HardEvent::MTE2_V>();
                    UpdateColumns(input,cols,pivotRow,inv);
                    Fence<HardEvent::V_MTE3>();DataCopy(w[j*na],input,cols*na);
                    Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_MTE2>();
                }
                uint32_t first=p+1>firstResident?p+1:firstResident;
                for(uint32_t j=first;j<n+k;j+=64) {
                    uint32_t cols=n+k-j<64?n+k-j:64;
                    UpdateColumns(mat[(j-firstResident)*na],cols,pivotRow,inv);
                }
            }
            Fence<HardEvent::V_S>();
            Assemble(mat[(n-firstResident)*na],output,false);
            Refine(batch);
            Fence<HardEvent::V_MTE3>();
            DataCopyExtParams cp{static_cast<uint16_t>(n),k*4,0,0,0};
            DataCopyPad(x[uint64_t(batch)*n*k],output,cp);Fence<HardEvent::MTE3_S>();
        }
    }

    __aicore__ inline void LoadRange(GlobalTensor<float> source,uint64_t offset,uint32_t stride,uint32_t count,uint32_t dstColumn) {
        if constexpr(FAMILY==3 && STATIC_N>0 && STATIC_N%16==0) {
            LoadAlignedWideRange(source,offset,stride,count,dstColumn);return;
        }
        for(uint32_t col=0;col<count;col+=8) {
            uint32_t cols=count-col<8?count-col:8;
            Duplicate(input,0.0f,8*cp);Fence<HardEvent::V_MTE2>();
            DataCopyPad(input,source[offset+col],
                        DataCopyExtParams{static_cast<uint16_t>(n),cols*4,(stride-cols)*4,0,0},
                        DataCopyPadExtParams<float>{false,0,0,0});
            Fence<HardEvent::MTE2_V>();
            uint64_t sourceAddresses[16],targetAddresses[16];
            #pragma unroll
            for(uint32_t i=0;i<16;++i) {
                sourceAddresses[i]=reinterpret_cast<uint64_t>(input[i*8].GetPhyAddr());
                targetAddresses[i]=reinterpret_cast<uint64_t>(mat[(dstColumn+col+i/2)*cp+(i%2)*8].GetPhyAddr());
            }
            TransDataTo5HD<float>(targetAddresses,sourceAddresses,
                                  TransDataTo5HDParams{false,false,uint8_t((na+15)/16),2,16});
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void LoadAlignedWideRange(GlobalTensor<float> source,uint64_t offset,
                                               uint32_t stride,uint32_t count,uint32_t dstColumn) {
        for(uint32_t col=0;col<count;) {
            uint32_t group=count-col>=16?16:8;
            uint32_t cols=count-col<group?count-col:group;
            if(cols<group)Duplicate(input,0.0f,group*cp);
            Fence<HardEvent::V_MTE2>();
            DataCopyPad(input,source[offset+col],
                        DataCopyExtParams{static_cast<uint16_t>(n),cols*4,(stride-cols)*4,0,0},
                        DataCopyPadExtParams<float>{false,0,0,0});
            Fence<HardEvent::MTE2_V>();
            for(uint32_t part=0;part<group;part+=8) {
                uint64_t sourceAddresses[16],targetAddresses[16];
                #pragma unroll
                for(uint32_t i=0;i<16;++i) {
                    sourceAddresses[i]=reinterpret_cast<uint64_t>(input[i*group+part].GetPhyAddr());
                    targetAddresses[i]=reinterpret_cast<uint64_t>(mat[(dstColumn+col+part+i/2)*cp+(i%2)*8].GetPhyAddr());
                }
                TransDataTo5HD<float>(targetAddresses,sourceAddresses,
                                      TransDataTo5HDParams{false,false,uint8_t(na/16),2,uint8_t(2*group)});
            }
            PipeBarrier<PIPE_V>();col+=cols;
        }
    }
    __aicore__ inline void FactorPaired() {
        if constexpr(FAMILY==2 || FAMILY==3)FactorWindow();
        else if constexpr(FAMILY==1)FactorUnpivoted();
        else {if(n>=224)FactorWindow();else FactorUnpivoted();}
    }
    // Build the fixed private layout directly from bounded GM reads. Missing
    // rows/columns form diag(A,4I); extra RHS columns are zero. The added block
    // is independent, so its solution is zero and the original AX=B is retained.
    __aicore__ inline void LoadCapacityRange(GlobalTensor<float> source,uint64_t offset,
                                            uint32_t stride,uint32_t count,uint32_t dstColumn,
                                            uint32_t validCols,uint32_t firstDiagonal) {
        if(logicalN==n && validCols==count) {
            LoadAlignedWideRange(source,offset,stride,count,dstColumn);return;
        }
        for(uint32_t col=0;col<count;) {
            uint32_t group=count-col>=16?16:8;
            uint32_t cols=col<validCols?(validCols-col<group?validCols-col:group):0;
            Duplicate(input,0.0f,group*cp);Fence<HardEvent::V_MTE2>();
            if(cols) {
                DataCopyPad(input,source[offset+col],
                    DataCopyExtParams{uint16_t(logicalN),cols*4,(stride-cols)*4,
                                      (group-(cols+7)/8*8)/8,0},
                    DataCopyPadExtParams<float>{true,0,uint8_t((8-cols%8)%8),0});
                Fence<HardEvent::MTE2_V>();
            } else PipeBarrier<PIPE_V>();
            for(uint32_t part=0;part<group;part+=8) {
                uint64_t sourceAddresses[16],targetAddresses[16];
                #pragma unroll
                for(uint32_t i=0;i<16;++i) {
                    sourceAddresses[i]=reinterpret_cast<uint64_t>(input[i*group+part].GetPhyAddr());
                    targetAddresses[i]=reinterpret_cast<uint64_t>(mat[(dstColumn+col+part+i/2)*cp+(i%2)*8].GetPhyAddr());
                }
                TransDataTo5HD<float>(targetAddresses,sourceAddresses,
                    TransDataTo5HDParams{false,false,uint8_t(na/16),2,uint8_t(2*group)});
            }
            PipeBarrier<PIPE_V>();col+=group;
        }
        if(firstDiagonal<n && validCols<count) {
            Fence<HardEvent::V_S>();
            for(uint32_t col=validCols;col<count;++col)
                mat.SetValue((dstColumn+col)*cp+firstDiagonal+col,4.0f);
            Fence<HardEvent::S_V>();
        }
    }
    __aicore__ inline void FactorUnpivoted() {
        constexpr bool persistent=FAMILY==1;
        if constexpr(persistent)InitializePublication();
        uint32_t batch=GetBlockIdx()/workers,lane=GetBlockIdx()%workers,split=(n+31)/32*8;
        uint32_t rhsLane=FAMILY==1?workers-1:0;
        uint32_t blocks=(n+7)/8;
        uint32_t begin=workers==4?lane*split:(blocks*lane/workers)*8;
        uint32_t end=workers==4?begin+split:(blocks*(lane+1)/workers)*8;if(end>n)end=n;
        uint32_t localCols=end-begin;
        {LoadRange(a,uint64_t(batch)*n*n+begin,n,localCols,0);
        if(lane==rhsLane)LoadRange(b,broadcast?0:uint64_t(batch)*n*k,k,k,localCols);}
        auto baseIndex=indices.ReinterpretCast<int32_t>();
        CreateVecIndex(baseIndex,int32_t(0),72);PipeBarrier<PIPE_V>();
        Muls(baseIndex,baseIndex,int32_t(cp*4),72);
        CreateVecIndex(order.ReinterpretCast<int32_t>(),int32_t(0),na);
        PipeBarrier<PIPE_V>();
        constexpr uint32_t panelWidth=FAMILY==1?4:8;
        for(uint32_t panel=0;panel<n;panel+=panelWidth) {
            factor=baseFactor;
            uint32_t panelEnd=n-panel<panelWidth?n:panel+panelWidth,steps=panelEnd-panel;
            bool owner=panel>=begin && panel<end;
            uint32_t packet=(n+16)*na+(persistent?panel:((panel/8)%2)*8)*cp;
            if(owner) {
                Fence<HardEvent::S_V>();
                SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,na);
                for(uint32_t p=panel;p<panelEnd;++p) {
                    factor=mat[(p-begin)*cp];
                    uint32_t row=p;
                    Fence<HardEvent::V_S>();
                    float diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[row];
                    if constexpr(persistent) {
                        factor.SetValue(row,0.0f);
                        SetFlag<HardEvent::S_V>(static_cast<event_t>(1));
                    }
                    float inv=1.0f/diagonal;
                    auto packetView=factor;
                    packetView.SetValue(na,inv);
                    if constexpr(persistent)WaitFlag<HardEvent::S_V>(static_cast<event_t>(1));
                    else {factor.SetValue(row,0.0f);Fence<HardEvent::S_V>();}
                    NormalizeOwnedPanel(mat[(p+1-begin)*cp],panelEnd-p-1,row,inv);
                }
                ResetMask();SetMaskNorm();
                // Deferred inverse stores feed the owner's Gather and packet DMA.
                if constexpr(persistent)Fence<HardEvent::S_V>();
                Fence<HardEvent::V_MTE3>();DataCopy(w[packet],mat[(panel-begin)*cp],steps*cp);
            }
            if constexpr(persistent) {
                // Each packet has private GM storage through the final join;
                // immutable readiness slots avoid all-matrix rendezvous.
                if(owner) {
                    if(panelEnd==end) {
                        // This owner has written every local inverse. Its last
                        // immutable packet flag also publishes this disjoint GM
                        // metadata slice before the RHS owner can assemble X.
                        Gather(inverse,mat,indices,na*4,localCols);
                        Fence<HardEvent::V_MTE3>();
                        DataCopyPad(w[(n+8)*na+begin],inverse,
                                    DataCopyExtParams{1,localCols*4,0,0,0});
                    }
                    PublishPersistentPanel(panel);
                }
                if(end<=panel && lane!=rhsLane)continue;
                if(!owner)WaitPersistentPanel(panel);
            } else SyncAll();
            if constexpr(FAMILY==1) {
                if(end<=panel && lane!=rhsLane)continue;
            }
            if(!owner){
                if constexpr(persistent) {
                    // Receives are the contiguous prefix [0,begin). Owners
                    // then retire, and the RHS owner is the final owner.
                    if(panel>=8)
                        WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(2+((panel/4)&1u)));
                    else if(panel==0)Fence<HardEvent::V_MTE2>();
                }
                DataCopy(input[(persistent?panel%8:0)*cp],w[packet],steps*cp);Fence<HardEvent::MTE2_V>();
            }
            if(!persistent && lane==rhsLane) {
                uint32_t metadataBegin=persistent?panel/8*8:panel;
                auto packetView=owner?mat[(metadataBegin-begin)*cp]:input;
                Gather(inverse[metadataBegin],packetView,indices,na*4,panelEnd-metadataBegin);
                PipeBarrier<PIPE_V>();
            }
            for(uint32_t p=panel;p<panelEnd;++p) {
                factor=owner?mat[(p-begin)*cp]:input[((persistent?panel%8:0)+p-panel)*cp];
                uint32_t row=p;float inv=0.0f;
                uint32_t first=panelEnd>begin?panelEnd:begin;
                uint32_t rhsCols=lane==rhsLane?k:0;
                if(first<end)UpdateColumns(mat[(first-begin)*cp],end-first+rhsCols,row,inv);
                else if(rhsCols)UpdateColumns(mat[localCols*cp],rhsCols,row,inv);
            }
            if constexpr(persistent) {
                if(!owner)SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(2+((panel/4)&1u)));
            }
        }
        if constexpr(persistent) {
            // Every nonempty receive prefix is 8-column aligned, using both slots.
            if(begin>0) {
                WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(2));
                WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(3));
            }
        }
        // Canonical pair composition overwrites the last published source columns.
        if constexpr(persistent)Fence<HardEvent::MTE3_V>();
        factor=baseFactor;
        if constexpr(FAMILY==1) {
            if(k<=workers) {
                // Compose retired factors only; initial RHS and factorization stay sequential.
                Fence<HardEvent::V_S>();
                SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,na);
                for(uint32_t p=begin;p+1<end;p+=2) {
                    auto left=mat[(p-begin)*cp];
                    float cross=reinterpret_cast<__ubuf__ float*>(left.GetPhyAddr())[p+1];
                    Axpy<float,float,false>(left,mat[(p+1-begin)*cp],cross,uint64_t(0),1,
                                            UnaryRepeatParams{1,1,8,8});
                }
                ResetMask();SetMaskNorm();
            }
        }
        {Fence<HardEvent::V_MTE3>();DataCopyPad(w[begin*na],mat,DataCopyExtParams{static_cast<uint16_t>(localCols),na*4,(cp-na)/8,0,0});}
        if(lane==rhsLane) {
            if constexpr(FAMILY==1) {
                // All prior owners' final flags were observed. Join this owner's
                // own inverse store before overwriting its UB source with the
                // complete vector. The first solution join precedes residual
                // reuse of the same GM metadata region.
                Fence<HardEvent::MTE3_MTE2>();
                DataCopy(inverse,w[(n+8)*na],na);
                Fence<HardEvent::MTE2_V>();
            }
            Assemble(mat[localCols*cp],output,false,cp);
        }
    }
    __aicore__ inline void InitializePublication() {
        auto ready=mat[128*na].ReinterpretCast<uint32_t>();
        if(GetBlockIdx()%workers==0) {
            constexpr uint32_t flagWords=FAMILY==1?2048:STATIC_WORKERS==10 && STATIC_K==3?2304:STATIC_WORKERS==8 && STATIC_K==3?2048:1024;
            Duplicate(ready,uint32_t(0),flagWords);
            Fence<HardEvent::V_MTE3>();DataCopy(publication,ready,flagWords);
        }
        // Global initialization joins the zero DMA before its source is reused.
        SyncAll();
        // Discard prior-invocation external ready lines after the zero DMA is joined.
        DataCacheCleanAndInvalid<uint32_t,CacheLine::ENTIRE_DATA_CACHE,DcciDst::CACHELINE_OUT>(publication);
        Duplicate(ready,uint32_t(1),8);Fence<HardEvent::V_MTE3>();
        // This 32B payload remains immutable through the final publication join.
    }
    __aicore__ inline void PublishPersistentPanel(uint32_t panel) {
        uint32_t index=(panel/(FAMILY==1?4:((STATIC_WORKERS==8 || STATIC_WORKERS==10) && STATIC_K==3?4:8)))*32;
        auto ready=mat[128*na].ReinterpretCast<uint32_t>();
        // Previous packet bytes finish before its flag; immutable UB source.
        PipeBarrier<PIPE_MTE3>();DataCopy(publication[index],ready,8);
    }
    __aicore__ inline void WaitPersistentPanel(uint32_t panel) {
        auto flag=publication[(panel/(FAMILY==1?4:((STATIC_WORKERS==8 || STATIC_WORKERS==10) && STATIC_K==3?4:8)))*32];
        auto ready=reinterpret_cast<volatile const __gm__ uint32_t*>(flag.GetPhyAddr());
        // Each slot only advances 0->1 here; a cached 1 is already current and final.
        while(*ready==0) {
            DataCacheCleanAndInvalid<uint32_t,CacheLine::SINGLE_CACHE_LINE>(flag);
        }
    }
    __aicore__ inline void FactorWindow() {
        constexpr bool persistent=FAMILY==3 && STATIC_N==256;
        constexpr bool retiredMetadata=persistent && (STATIC_WORKERS==8 || STATIC_WORKERS==10) && STATIC_K==3;
        constexpr uint32_t rhsLane=retiredMetadata?STATIC_WORKERS-1:0;
        constexpr uint32_t panelWidth=retiredMetadata?4:8;
        if constexpr(persistent)InitializePublication();
        constexpr uint32_t window=64;
        uint32_t batch=GetBlockIdx()/workers,lane=GetBlockIdx()%workers,split=(n+31)/32*8;
        uint32_t blocks=(n+7)/8;
        uint32_t begin=workers==4?lane*split:(blocks*lane/workers)*8;
        uint32_t end=workers==4?begin+split:(blocks*(lane+1)/workers)*8;if(end>n)end=n;
        if constexpr(STATIC_WORKERS==10 && STATIC_K==3) {
            // Eight owners cover the192-column prefix; the two tail owners
            // retain32 columns each, keeping their trailing-update load fixed.
            begin=lane<8?lane*24:192+(lane-8)*32;
            end=begin+(lane<8?24:32);
        }
        uint32_t localCols=end-begin;
        if constexpr(FAMILY==3 && STATIC_N==256 && STATIC_K==3) {
            uint32_t validCols=begin<logicalN?(logicalN-begin<localCols?logicalN-begin:localCols):0;
            LoadCapacityRange(a,uint64_t(batch)*logicalN*logicalN+begin,logicalN,localCols,0,validCols,begin);
            if(lane==rhsLane)
                LoadCapacityRange(b,broadcast?0:uint64_t(batch)*logicalN*logicalK,logicalK,k,localCols,logicalK,n);
        } else {
            LoadRange(a,uint64_t(batch)*n*n+begin,n,localCols,0);
            if(lane==rhsLane)LoadRange(b,broadcast?0:uint64_t(batch)*n*k,k,k,localCols);
        }
        auto absolute=indices[2*na].ReinterpretCast<float>();
        auto reduced=indices[3*na].ReinterpretCast<float>();
        auto baseIndex=indices.ReinterpretCast<int32_t>();
        CreateVecIndex(baseIndex,int32_t(0),72);PipeBarrier<PIPE_V>();
        Muls(baseIndex,baseIndex,int32_t(cp*4),72);
        PipeBarrier<PIPE_V>();
        constexpr bool boundaryReady=retiredMetadata;
        bool pendingBoundary=false;
        uint64_t activeRows[2]={0,0};
        for(uint32_t panel=0;panel<(persistent?192:n);panel+=panelWidth) {
            uint32_t rowBase=panel/window*window;
            if(panel%window==0)activeRows[0]=n-rowBase>=window?((~uint64_t(0))>>(64-window)):(uint64_t(1)<<(n-rowBase))-1;
            factor=baseFactor;
            uint32_t panelEnd=n-panel<panelWidth?n:panel+panelWidth,steps=panelEnd-panel;
            bool owner=panel>=begin && panel<end;
            uint32_t packet=(n+16)*na+(persistent?panel:((panel/8)%2)*8)*(na+8);
            if(owner) {
                Fence<HardEvent::S_V>();
                SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,FAMILY==3?na-rowBase:na);
                for(uint32_t p=panel;p<panelEnd;++p) {
                    factor=mat[(p-begin)*cp];
                    uint32_t rowCount=n-rowBase<window?n-rowBase:window;
                    if constexpr(persistent) {
                        if(p==panel) {
                            if constexpr(boundaryReady) {
                                if(pendingBoundary)WaitFlag<HardEvent::V_S>(static_cast<event_t>(3));
                                else Fence<HardEvent::V_S>();
                            } else Fence<HardEvent::V_S>();
                        }
                        else WaitFlag<HardEvent::V_S>(static_cast<event_t>(1));
                    } else Fence<HardEvent::V_S>();
                    uint64_t firstBit=activeRows[0]&(~activeRows[0]+1);
                    uint32_t row=rowBase+63-static_cast<uint32_t>(CountLeadingZero(firstBit));
                    float diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[row];
                    bool search;
                    if constexpr(persistent) {
                        union {float value;uint32_t bits;} word;word.value=diagonal;
                        search=(word.bits&0x7fffffffu)<0x40000000u;
                    } else {
                        float magnitude=diagonal<0.0f?-diagonal:diagonal;
                        search=magnitude<2.0f;
                    }
                    if(search) {
                        Abs(absolute,factor[rowBase],rowCount);PipeBarrier<PIPE_V>();
                        WholeReduceMax(reduced,absolute,activeRows,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
                        Fence<HardEvent::V_S>();
                        row=rowBase+reinterpret_cast<__ubuf__ uint32_t*>(reduced.GetPhyAddr())[1];
                        diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[row];
                        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,FAMILY==3?na-rowBase:na);
                    }
                    if constexpr(persistent) {
                        factor.SetValue(row,0.0f);
                        SetFlag<HardEvent::S_V>(static_cast<event_t>(1));
                    }
                    float inv=1.0f/diagonal;
                    auto packetView=factor;
                    packetView.SetValue(na,inv);packetView.ReinterpretCast<uint32_t>().SetValue(na+1,row);
                    activeRows[0]&=~(uint64_t(1)<<(row-rowBase));
                    if constexpr(!persistent)factor.SetValue(row,0.0f);
                    // Other columns may still be completing the preceding pivot.
                    // Join them before reading their new scalar coefficients.
                    if constexpr(persistent) {
                        if(p!=panel)WaitFlag<HardEvent::V_S>(static_cast<event_t>(2));
                        else if constexpr(boundaryReady) {
                            if(pendingBoundary) {
                                // Scalar coefficients read only the completed pivot window.
                                WaitFlag<HardEvent::V_S>(static_cast<event_t>(4));
                                // Normalization reads the entire factor suffix on V.
                                PipeBarrier<PIPE_V>();
                                pendingBoundary=false;
                            }
                        }
                    }
                    if constexpr(persistent)WaitFlag<HardEvent::S_V>(static_cast<event_t>(1));
                    else Fence<HardEvent::S_V>();
                    if constexpr(persistent)
                        NormalizeOwnedPanelEarly(mat[(p+1-begin)*cp],panelEnd-p-1,row,inv);
                    else NormalizeOwnedPanel(mat[(p+1-begin)*cp],panelEnd-p-1,row,inv);
                }
                ResetMask();SetMaskNorm();
                // Metadata feeds both this owner's inverse/order Gather and packet DMA.
                // S_V orders Gather;the following V_MTE3 also orders its DMA read.
                if constexpr(persistent)Fence<HardEvent::S_V>();
                Fence<HardEvent::V_MTE3>();DataCopy(w[packet],mat[(panel-begin)*cp],steps*cp);
                if constexpr(persistent)PublishPersistentPanel(panel);
            }
            if constexpr(!persistent)SyncAll();
            if constexpr(FAMILY==3) {
                // The collapsed-tail owner publishes the complete inverse/order copy.
                // Other lanes reload it before backsolve and need no retired packets.
                if constexpr(retiredMetadata) {if(end<=panel)break;}
                else if constexpr(persistent) {if(end<=panel && lane!=0)continue;}
                else {if(end<=panel && lane>=k)continue;}
            }
            if(!owner){
                if constexpr(persistent) {
                    WaitPersistentPanel(panel);
                    // The single receiver slot must outlive its preceding V reads.
                    Fence<HardEvent::V_MTE2>();
                }
                DataCopy(input,w[packet],steps*(na+8));Fence<HardEvent::MTE2_V>();
            }
            auto packetView=owner?mat[(panel-begin)*cp]:input;
            if((!retiredMetadata && lane==0) || (!persistent && FAMILY==3 && lane<k))
                Gather(inverse[panel],packetView,indices,na*4,steps);
            if(!retiredMetadata && (persistent?lane==0:lane<k))
                Gather(order[panel].ReinterpretCast<float>(),packetView,indices,(na+1)*4,steps);
            Fence<HardEvent::V_S>();
            for(uint32_t p=panel;p<panelEnd;++p) {
                factor=owner?mat[(p-begin)*cp]:input[(p-panel)*cp];
                uint32_t row=reinterpret_cast<__ubuf__ uint32_t*>(factor.GetPhyAddr())[na+1];
                activeRows[0]&=~(uint64_t(1)<<(row-rowBase));
                uint32_t first=panelEnd>begin?panelEnd:begin;
                uint32_t rhsCols=lane==rhsLane?k:0;
                if constexpr(boundaryReady) {
                    if(p+1==panelEnd && panelEnd>=begin && panelEnd<end && panelEnd<192) {
                        UpdateColumnsWindowReady(mat[(first-begin)*cp],end-first+rhsCols,row,panelEnd/64*64);
                        pendingBoundary=true;
                    } else if(first<end)UpdateColumns(mat[(first-begin)*cp],end-first+rhsCols,row,0.0f);
                    else if(rhsCols)UpdateColumns(mat[localCols*cp],rhsCols,row,0.0f);
                } else {
                    if(first<end)UpdateColumns(mat[(first-begin)*cp],end-first+rhsCols,row,0.0f);
                    else if(rhsCols)UpdateColumns(mat[localCols*cp],rhsCols,row,0.0f);
                }
            }
        }
        if constexpr(persistent)Fence<HardEvent::MTE3_V>();
        factor=baseFactor;
        if constexpr(FAMILY==3 && STATIC_N==256) {
            // Publication DMA is joined above; only retired canonical factors change.
            Fence<HardEvent::V_S>();SetMaskCount();
            uint32_t currentStart=na;
            for(uint32_t p=begin;p<end && p<192;p+=4) {
                uint32_t start=p/64*64;
                if(start!=currentStart) {
                    SetVectorMask<float,MaskMode::COUNTER>(0,na-start);
                    currentStart=start;
                }
                auto f0=mat[(p-begin)*cp],f1=mat[(p+1-begin)*cp];
                auto f2=mat[(p+2-begin)*cp],f3=mat[(p+3-begin)*cp];
                uint32_t r1=reinterpret_cast<__ubuf__ uint32_t*>(f1.GetPhyAddr())[na+1];
                uint32_t r2=reinterpret_cast<__ubuf__ uint32_t*>(f2.GetPhyAddr())[na+1];
                uint32_t r3=reinterpret_cast<__ubuf__ uint32_t*>(f3.GetPhyAddr())[na+1];
                float c01=reinterpret_cast<__ubuf__ float*>(f0.GetPhyAddr())[r1];
                float c02=reinterpret_cast<__ubuf__ float*>(f0.GetPhyAddr())[r2];
                float c03=reinterpret_cast<__ubuf__ float*>(f0.GetPhyAddr())[r3];
                float c12=reinterpret_cast<__ubuf__ float*>(f1.GetPhyAddr())[r2];
                float c13=reinterpret_cast<__ubuf__ float*>(f1.GetPhyAddr())[r3];
                float c23=reinterpret_cast<__ubuf__ float*>(f2.GetPhyAddr())[r3];
                Axpy<float,float,false>(f2[start],f3[start],c23,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
                PipeBarrier<PIPE_V>();
                Axpy<float,float,false>(f1[start],f2[start],c12,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
                PipeBarrier<PIPE_V>();
                Axpy<float,float,false>(f1[start],f3[start],c13,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
                PipeBarrier<PIPE_V>();
                Axpy<float,float,false>(f0[start],f1[start],c01,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
                PipeBarrier<PIPE_V>();
                Axpy<float,float,false>(f0[start],f2[start],c02,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
                PipeBarrier<PIPE_V>();
                Axpy<float,float,false>(f0[start],f3[start],c03,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
                PipeBarrier<PIPE_V>();
            }
            ResetMask();SetMaskNorm();
        } else if constexpr(FAMILY==3) {
            Fence<HardEvent::V_S>();
            SetMaskCount();
            uint32_t currentStart=na;
            for(uint32_t p=begin;p+1<end && (!persistent || p+1<192);p+=2) {
                uint32_t start=p/64*64;
                if(start!=currentStart) {
                    SetVectorMask<float,MaskMode::COUNTER>(0,na-start);
                    currentStart=start;
                }
                auto left=mat[(p-begin)*cp],right=mat[(p+1-begin)*cp];
                uint32_t row1=reinterpret_cast<__ubuf__ uint32_t*>(right.GetPhyAddr())[na+1];
                float cross=reinterpret_cast<__ubuf__ float*>(left.GetPhyAddr())[row1];
                Axpy<float,float,false>(left[start],right[start],cross,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
            }
            ResetMask();SetMaskNorm();
        }
        if constexpr(retiredMetadata) {
            if(lane<(STATIC_WORKERS==10?8:6)) {
                // Retired local columns still contain their inverse/order words.
                // Staging n+12/n+13 is disjoint from live prefix packets at n+16.
                Gather(inverse[begin],mat,indices,na*4,localCols);
                Gather(order[begin].ReinterpretCast<float>(),mat,indices,(na+1)*4,localCols);
                Fence<HardEvent::V_MTE3>();
                DataCopy(w[(n+12)*na+begin],inverse[begin],localCols);
                DataCopy(w[(n+13)*na+begin],order[begin].ReinterpretCast<float>(),localCols);
            }
        }
        {Fence<HardEvent::V_MTE3>();DataCopyPad(w[begin*na],mat,DataCopyExtParams{static_cast<uint16_t>(localCols),na*4,1,0,0});}
        if(lane==rhsLane) {
            if constexpr(FAMILY==3) {
                Fence<HardEvent::V_MTE3>();
                DataCopyPad(w[(n+8)*na],mat[localCols*cp],
                            DataCopyExtParams{static_cast<uint16_t>(k),na*4,1,0,0});
            } else Assemble(mat[localCols*cp],output,false,cp);
        }
        if constexpr(persistent) {
            if constexpr((STATIC_WORKERS==8 || STATIC_WORKERS==10) && STATIC_K==3) {
                if constexpr(retiredMetadata) {
                    // One ordered flag joins each nonzero lane's canonical data,
                    // compact metadata and (on lane7) the complete RHS stream.
                    if(lane>0)PublishPersistentPanel(192+lane*8);
                } else if(lane>=(STATIC_WORKERS==10?8:6))PublishPersistentPanel(192+lane*8);
                if(lane==0) {
                    if constexpr(retiredMetadata)PrepareCompactThreeIndices();
                    Fence<HardEvent::MTE3_MTE2>();
                    if constexpr(retiredMetadata) {
                        for(uint32_t producer=1;producer<STATIC_WORKERS;++producer)
                            WaitPersistentPanel(192+producer*8);
                        DataCopy(inverse,w[(n+12)*na],192);
                        DataCopy(order.ReinterpretCast<float>(),w[(n+13)*na],192);
                        Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
                    } else {WaitPersistentPanel(240);WaitPersistentPanel(248);}
                    FactorCollapsedTail<192>();
                    // Every prefix producer was already joined above. Slot48
                    // now covers the complete tail/RHS/inverse/order stores.
                    if constexpr(retiredMetadata)PublishPersistentPanel(192);
                }
            } else {
                SyncAll();
                if(lane==0)FactorCollapsedTail<192>();
            }
        }
    }
    template<uint32_t PREFIX>
    __aicore__ inline void FactorCollapsedTail() {
        constexpr uint32_t rows=256-PREFIX,pitch=rows+8;
        // PREFIX192: (64+8)*72*4=20736B. Existing scratch and metadata remain disjoint.
        auto matrix=LocalTensor<float>(TPosition::VECCALC,0,(rows+8)*pitch);
        auto absolute=indices[2*na].ReinterpretCast<float>();
        auto reduced=indices[3*na].ReinterpretCast<float>();
        auto pivot=indices[4*na].ReinterpretCast<float>();
        auto baseIndex=indices.ReinterpretCast<int32_t>();
        Fence<HardEvent::V_MTE2>();
        DataCopyPad(matrix,w[PREFIX*na+PREFIX],
            DataCopyExtParams{uint16_t(rows),rows*4,PREFIX*4,1,0},
            DataCopyPadExtParams<float>{false,0,0,0});
        DataCopyPad(matrix[rows*pitch],w[(n+8)*na+PREFIX],
            DataCopyExtParams{uint16_t(k),rows*4,PREFIX*4,1,0},
            DataCopyPadExtParams<float>{false,0,0,0});
        Fence<HardEvent::MTE2_V>();
        CreateVecIndex(baseIndex,0,rows+8);PipeBarrier<PIPE_V>();
        Muls(baseIndex,baseIndex,int32_t(pitch*4),rows+8);PipeBarrier<PIPE_V>();
        uint64_t active[2]={0,0};
        for(uint32_t p=PREFIX;p<n;++p) {
            uint32_t rowBase=p/64*64,firstRow=rowBase-PREFIX;
            if(p%64==0)active[0]=~uint64_t(0);
            auto f=matrix[(p-PREFIX)*pitch];
            if constexpr(rows==64) {
                if(p==PREFIX)Fence<HardEvent::V_S>();
                else WaitFlag<HardEvent::V_S>(static_cast<event_t>(1));
            } else Fence<HardEvent::V_S>();
            uint64_t firstBit=active[0]&(~active[0]+1);
            uint32_t row=rowBase+63-static_cast<uint32_t>(CountLeadingZero(firstBit));
            float diagonal=reinterpret_cast<__ubuf__ float*>(f.GetPhyAddr())[row-PREFIX];
            union {float value;uint32_t bits;} word;word.value=diagonal;
            if((word.bits&0x7fffffffu)<0x40000000u) {
                Abs(absolute,f[firstRow],64);PipeBarrier<PIPE_V>();
                WholeReduceMax(reduced,absolute,active,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
                Fence<HardEvent::V_S>();
                row=rowBase+reinterpret_cast<__ubuf__ uint32_t*>(reduced.GetPhyAddr())[1];
                diagonal=reinterpret_cast<__ubuf__ float*>(f.GetPhyAddr())[row-PREFIX];
            }
            // This column's first update is complete;its zero can become visible
            // while Gather and scalar reciprocal/metadata are issued.
            f.SetValue(row-PREFIX,0.0f);
            SetFlag<HardEvent::S_V>(static_cast<event_t>(1));
            // Pivot column is ready. Join the prior later-column updates on V
            // before Gather reads them and Brcb overwrites their coefficient scratch.
            if constexpr(rows==64)PipeBarrier<PIPE_V>();
            uint32_t cols=n+k-p-1;
            auto trailing=matrix[(p+1-PREFIX)*pitch];
            Gather(pivot,trailing,indices,(row-PREFIX)*4,cols);PipeBarrier<PIPE_V>();
            Brcb(transposed,pivot,(cols+7)/8,BrcbRepeatParams{1,8});
            float inv=1.0f/diagonal;
            inverse.SetValue(p,inv);order.SetValue(p,row);
            active[0]&=~(uint64_t(1)<<(row-rowBase));
            WaitFlag<HardEvent::S_V>(static_cast<event_t>(1));
            if constexpr(rows==64) {
                // Explicit normal-mode 64-lane normalization leaves the same mask
                // for the independent-column updates; no count-mode transition.
                Muls<float>(f,f,-inv,uint64_t(64),1,UnaryRepeatParams{1,1,8,8});
                PipeBarrier<PIPE_V>();
                if(p+1<n) {
                    MulAddDst<float,float,false>(trailing,f,transposed,uint64_t(64),1,
                        BinaryRepeatParams{1,1,0,uint8_t(pitch/8),0,1});
                    SetFlag<HardEvent::V_S>(static_cast<event_t>(1));
                    MulAddDst<float,float,false>(trailing[pitch],f,transposed[8],uint64_t(64),cols-1,
                        BinaryRepeatParams{1,1,0,uint8_t(pitch/8),0,1});
                } else {
                    MulAddDst<float,float,false>(trailing,f,transposed,uint64_t(64),cols,
                        BinaryRepeatParams{1,1,0,uint8_t(pitch/8),0,1});
                }
            } else {
                Muls(f[firstRow],f[firstRow],-inv,rows-firstRow);PipeBarrier<PIPE_V>();
                for(uint32_t base=firstRow;base<rows;base+=64) {
                    MulAddDst(trailing[base],f[base],transposed,64,cols,
                        BinaryRepeatParams{1,1,0,uint8_t(pitch/8),0,1});
                }
            }
            if constexpr(rows!=64)PipeBarrier<PIPE_V>();
        }
        Fence<HardEvent::V_S>();
        SetMaskCount();uint32_t currentStart=rows;
        for(uint32_t p=PREFIX;p+1<n;p+=2) {
            uint32_t start=p/64*64-PREFIX;
            if(start!=currentStart) {
                SetVectorMask<float,MaskMode::COUNTER>(0,rows-start);currentStart=start;
            }
            auto left=matrix[(p-PREFIX)*pitch],right=matrix[(p+1-PREFIX)*pitch];
            uint32_t row1=order.GetValue(p+1)-PREFIX;
            float cross=reinterpret_cast<__ubuf__ float*>(left.GetPhyAddr())[row1];
            Axpy<float,float,false>(left[start],right[start],cross,uint64_t(0),1,
                UnaryRepeatParams{1,1,8,8});
        }
        ResetMask();SetMaskNorm();
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(w[PREFIX*na+PREFIX],matrix,
            DataCopyExtParams{uint16_t(rows),rows*4,1,PREFIX*4,0});
        DataCopyPad(w[(n+8)*na+PREFIX],matrix[rows*pitch],
            DataCopyExtParams{uint16_t(k),rows*4,1,PREFIX*4,0});
        // Retired packet space is disjoint from RHS n+8..n+15.
        Fence<HardEvent::S_MTE3>();
        DataCopy(w[(n+16)*na],inverse,na);
        DataCopy(w[(n+17)*na],order.ReinterpretCast<float>(),na);
    }
    __aicore__ inline void ProcessCooperative() {
        if constexpr(FAMILY==3){ProcessBlocked();return;}
        uint32_t batch=GetBlockIdx()/workers,lane=GetBlockIdx()%workers;
        uint32_t rhsLane=FAMILY==1?workers-1:0;
        FactorPaired();
        if(lane==rhsLane){Fence<HardEvent::V_MTE3>();DataCopy(w[n*na],output,n*8);}
        SyncAll();
        if(lane!=rhsLane){DataCopy(output,w[n*na],n*8);Fence<HardEvent::MTE2_V>();}
        firstResident=n+k;
        auto xt=input;auto rhs=transposed;
        Duplicate(xt,0.0f,k*na);
        CreateVecIndex(indices.ReinterpretCast<int32_t>(),int32_t(0),na);PipeBarrier<PIPE_V>();
        Muls(indices.ReinterpretCast<int32_t>(),indices.ReinterpretCast<int32_t>(),int32_t(32),na);PipeBarrier<PIPE_V>();
        for(uint32_t q=0;q<k;++q)Gather(xt[q*na],output,indices,q*4,n);
        PipeBarrier<PIPE_V>();Duplicate(rhs,0.0f,k*na);PipeBarrier<PIPE_V>();
        PrepareResidualIndices();
        uint32_t tileRows=n>=224?32:16;
        for(uint32_t base=lane*tileRows;base<n;base+=workers*tileRows) {
            uint32_t rows=n-base<tileRows?n-base:tileRows;
            ResidualTile(batch,base,rows,tileRows);Fence<HardEvent::V_MTE3>();
            for(uint32_t q=0;q<k;++q)
                DataCopyPad(w[(n+8+q)*na+base],rhs[q*na+base],DataCopyExtParams{1,rows*4,0,0,0});
            Fence<HardEvent::MTE3_V>();
        }
        SyncAll();
        if(k<=workers) {
            if(lane<k) {
                DataCopy(rhs,w[(n+8+lane)*na],na);Fence<HardEvent::MTE2_V>();
                CorrectSingleRhs();Fence<HardEvent::V_MTE3>();
                DataCopy(w[(n+8+lane)*na],rhs,na);
            }
            SyncAll();
        }
        if(lane==rhsLane) {
            DataCopy(rhs,w[(n+8)*na],k*na);Fence<HardEvent::MTE2_V>();
            if(k<=workers)Assemble(rhs,mat,true);else ApplyCorrection();
            StoreCooperativeOutput(batch);
            Fence<HardEvent::MTE3_S>();
        }
    }
    __aicore__ inline void StoreCooperativeOutput(uint32_t batch) {
        if(n%8==0) {
            Duplicate(indices,uint32_t(((1u<<k)-1)*0x01010101u),8);PipeBarrier<PIPE_V>();
            GatherMask(mat,output,indices.ReinterpretCast<float>(),uint8_t(0),
                       GatherMaskParams{1,static_cast<uint16_t>(n/8),8,0});
            Fence<HardEvent::V_MTE3>();
            DataCopyPad(x[uint64_t(batch)*n*k],mat,DataCopyExtParams{1,n*k*4,0,0,0});
        } else {
            Fence<HardEvent::V_MTE3>();
            DataCopyPad(x[uint64_t(batch)*n*k],output,DataCopyExtParams{static_cast<uint16_t>(n),k*4,0,0,0});
        }
    }
    __aicore__ inline void PrepareCompactThreeIndices() {
        // Initial solution uses output[0:3*na]; immutable compact gather indices
        // occupy output[3*na:6*na], within the existing8*n allocation.
        auto offsets=output[3*na].ReinterpretCast<int32_t>();
        #pragma unroll
        for(uint32_t i=0;i<24;++i)offsets.SetValue(i,int32_t(((i%3)*na+i/3)*4));
        Fence<HardEvent::S_V>();
        for(uint32_t filled=24;filled<3*na;filled*=2) {
            Adds(offsets[filled],offsets,int32_t((filled/3)*4),filled);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void StoreCompactThreeOutput(uint32_t batch) {
        Add(transposed,output,transposed,3*na);PipeBarrier<PIPE_V>();
        if(logicalK<k) {
            // Reuse the dead initial-solution buffer for dense output indices.
            auto offsets=output.ReinterpretCast<uint32_t>();
            Fence<HardEvent::V_S>();
            for(uint32_t row=0;row<logicalN;++row)
                for(uint32_t q=0;q<logicalK;++q)offsets.SetValue(row*logicalK+q,(q*na+row)*4);
            Fence<HardEvent::S_V>();
            Gather(mat,transposed,offsets,0,logicalN*logicalK);
        } else Gather(mat,transposed,output[3*na].ReinterpretCast<uint32_t>(),0,3*logicalN);
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(x[uint64_t(batch)*logicalN*logicalK],mat,DataCopyExtParams{1,logicalN*logicalK*4,0,0,0});
    }
    __aicore__ inline void ProcessBlocked() {
        uint32_t batch=GetBlockIdx()/workers,lane=GetBlockIdx()%workers;
        constexpr bool localJoins=FAMILY==3 && STATIC_N==256 && (STATIC_WORKERS==8 || STATIC_WORKERS==10) && STATIC_K==3;
        FactorPaired();
        if constexpr(localJoins) {
            if(lane==0)Fence<HardEvent::MTE3_MTE2>();
            else if(lane<k)WaitPersistentPanel(192);
        } else SyncAll();
        if constexpr(FAMILY==3 && STATIC_N==256) {
            if constexpr((STATIC_WORKERS==8 || STATIC_WORKERS==10) && STATIC_K==3) {
                // Prefix metadata was reloaded before tail generation; the tail
                // filled the remaining64 entries. Order scalar stores before V.
                if(lane==0)Fence<HardEvent::S_V>();
            }
            if(lane<k && (!((STATIC_WORKERS==8 || STATIC_WORKERS==10) && STATIC_K==3) || lane!=0)) {
                DataCopy(inverse,w[(n+16)*na],na);
                DataCopy(order.ReinterpretCast<float>(),w[(n+17)*na],na);
                Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
            }
        }
        if(lane<k) {
            DataCopy(transposed,w[(n+8+lane)*na],na);Fence<HardEvent::MTE2_V>();
            BacksolveBlocked();Fence<HardEvent::V_MTE3>();
            DataCopy(w[(n+8+lane)*na],transposed,na);
        }
        SyncAll();
        DataCopy(input,w[(n+8)*na],k*na);Fence<HardEvent::MTE2_V>();
        if(lane==0) {
            if constexpr(localJoins) {DataCopy(output,input,k*na);PipeBarrier<PIPE_V>();}
            else Assemble(input,output,false,na,false,false);
        }
        Duplicate(transposed,0.0f,k*na);PipeBarrier<PIPE_V>();
        PrepareResidualIndices();
        uint32_t tileRows=n>=224?32:16;
        for(uint32_t base=lane*tileRows;base<n;base+=workers*tileRows) {
            uint32_t rows=n-base<tileRows?n-base:tileRows;
            if(n>=224)ResidualTileRounded(batch,base,rows,tileRows);
            else ResidualTileAccurate(batch,base,rows,tileRows);
            Fence<HardEvent::V_MTE3>();
            if constexpr(FAMILY==3 && STATIC_N==256) {
                // 32-row aligned segments; equal 256-element source/destination pitch.
                DataCopy(w[(n+8)*na+base],transposed[base],
                    DataCopyParams{uint16_t(k),uint16_t(rows/8),uint16_t((na-rows)/8),uint16_t((na-rows)/8)});
            } else {
                for(uint32_t q=0;q<k;++q)
                    DataCopyPad(w[(n+8+q)*na+base],transposed[q*na+base],DataCopyExtParams{1,rows*4,0,0,0});
            }
            Fence<HardEvent::MTE3_V>();
        }
        SyncAll();
        if(lane<k) {
            DataCopy(transposed,w[(n+8+lane)*na],na);Fence<HardEvent::MTE2_V>();
            ForwardBlocked();BacksolveBlocked(true);
            if constexpr(localJoins) {
                // Residual scratch overwrote the original seed. Restore it after
                // correction; the existing V_MTE3 orders both result and seed.
                if(lane)Duplicate(mat[128*na].ReinterpretCast<uint32_t>(),uint32_t(1),8);
            }
            Fence<HardEvent::V_MTE3>();
            DataCopy(w[(n+8+lane)*na],transposed,na);
            if constexpr(localJoins) {
                if(lane)PublishPersistentPanel(196+lane*8);
            }
        }
        if constexpr(localJoins) {
            if(lane==0) {
                Fence<HardEvent::MTE3_MTE2>();
                WaitPersistentPanel(204);WaitPersistentPanel(212);
            }
        } else SyncAll();
        if(lane==0) {
            DataCopy(transposed,w[(n+8)*na],k*na);Fence<HardEvent::MTE2_V>();
            if constexpr(localJoins)StoreCompactThreeOutput(batch);
            else {
                Assemble(transposed,mat,true,na,false,false);
                StoreCooperativeOutput(batch);
            }
            Fence<HardEvent::MTE3_S>();
        }
    }
    __aicore__ inline void ForwardBlocked() {
        auto rhs=transposed;
        bool cached=n<=200;
        Fence<HardEvent::V_MTE2>();DataCopy(mat,w,(cached?n:(n<64?n:64))*na);
        for(uint32_t block=0;block<n;block+=64) {
            uint32_t stop=block+64<n?block+64:n,slot=(block/64)&1u;
            Fence<HardEvent::MTE2_V>();
            if(!cached && stop<n) {
                uint32_t count=n-stop<64?n-stop:64;
                Fence<HardEvent::V_MTE2>();DataCopy(mat[(slot^1u)*64*na],w[stop*na],count*na);
            }
            if(FAMILY==3 && STATIC_N==256 && block<192) {
                SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,na-block);
                for(uint32_t p=block;p<stop;p+=4) {
                    uint32_t r0=order.GetValue(p),r1=order.GetValue(p+1);
                    uint32_t r2=order.GetValue(p+2),r3=order.GetValue(p+3);
                    Fence<HardEvent::V_S>();
                    float v0=rhs.GetValue(r0),v1=rhs.GetValue(r1);
                    float v2=rhs.GetValue(r2),v3=rhs.GetValue(r3);
                    auto f=mat[(slot*64+p-block)*na+block];
                    Axpy<float,float,false>(rhs[block],f,v0,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
                    PipeBarrier<PIPE_V>();
                    Axpy<float,float,false>(rhs[block],f[na],v1,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
                    PipeBarrier<PIPE_V>();
                    Axpy<float,float,false>(rhs[block],f[2*na],v2,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
                    PipeBarrier<PIPE_V>();
                    Axpy<float,float,false>(rhs[block],f[3*na],v3,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
                    PipeBarrier<PIPE_V>();
                }
            } else for(uint32_t p=block;p<stop;p+=2) {
                uint32_t start=p/64*64;
                if(p%64==0){SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,na-start);}
                uint32_t row=order.GetValue(p);
                uint32_t row1=p+1<stop?order.GetValue(p+1):row;
                Fence<HardEvent::V_S>();float value=rhs.GetValue(row);
                float value1=p+1<stop?rhs.GetValue(row1):0.0f;
                Axpy<float,float,false>(rhs[start],mat[(cached?p:slot*64+p-block)*na+start],value,uint64_t(0),1,
                                        UnaryRepeatParams{1,1,8,8});
                PipeBarrier<PIPE_V>();
                if(p+1<stop) {
                    Axpy<float,float,false>(rhs[start],mat[(cached?p+1:slot*64+p+1-block)*na+start],value1,uint64_t(0),1,
                                            UnaryRepeatParams{1,1,8,8});
                    PipeBarrier<PIPE_V>();
                }
            }
        }
        ResetMask();SetMaskNorm();
    }
    __aicore__ inline void BacksolveBlocked(bool coefficientsReady=false) {
        auto rhs=transposed;
        bool cached=n<=200;
        if(cached && !coefficientsReady) {
            Fence<HardEvent::V_MTE2>();DataCopy(mat,w,n*na);Fence<HardEvent::MTE2_V>();
        }
        uint32_t last=(n-1)/64*64;
        if(!cached && !coefficientsReady) {
            uint32_t count=n-last,slot=(last/64)&1u;
            Fence<HardEvent::V_MTE2>();DataCopy(mat[slot*64*na],w[last*na],count*na);
        }
        for(int32_t block=int32_t(last);block>=0;block-=64) {
            uint32_t steps=n-uint32_t(block)<64?n-uint32_t(block):64;
            uint32_t slot=(uint32_t(block)/64)&1u;
            if(!cached) {
                if(block && !(coefficientsReady && uint32_t(block)+64>=last))
                    Fence<HardEvent::MTE2_V>();
                // Forward replay leaves its final two coefficient blocks in these slots.
                if(block>64 && !(coefficientsReady && uint32_t(block)==last)) {
                    Fence<HardEvent::V_MTE2>();
                    DataCopy(mat[(slot^1u)*64*na],w[(uint32_t(block)-64)*na],64*na);
                }
            }
            Muls(indices.ReinterpretCast<int32_t>(),order[uint32_t(block)].ReinterpretCast<int32_t>(),int32_t(4),steps);
            PipeBarrier<PIPE_V>();Gather(column,rhs,indices,0,steps);PipeBarrier<PIPE_V>();
            Mul(column,column,inverse[uint32_t(block)],steps);PipeBarrier<PIPE_V>();
            Adds(baseFactor[uint32_t(block)],column,0.0f,steps);PipeBarrier<PIPE_V>();
            if(block) {
                // Retired upper columns are dead after this block's back-update.
                auto coefficients=mat[cached?uint32_t(block)*na:slot*64*na];
                Brcb(input,column,(steps+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                BinaryRepeatParams productParams{1,1,0,static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8),1};
                BinaryRepeatParams sumParams{1,1,1,static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8)};
                for(uint32_t base=0;base<uint32_t(block);base+=64) {
                    uint32_t count=uint32_t(block)-base<64?uint32_t(block)-base:64;
                    Mul(coefficients[base],coefficients[base],input,count,steps,productParams);
                }
                PipeBarrier<PIPE_V>();
                for(uint32_t remain=steps;remain>1;) {
                    uint32_t half=(remain+1)/2,countCols=remain-half;
                    for(uint32_t base=0;base<uint32_t(block);base+=64) {
                        uint32_t count=uint32_t(block)-base<64?uint32_t(block)-base:64;
                        Add(coefficients[base],coefficients[base],coefficients[half*na+base],count,countCols,sumParams);
                    }
                    PipeBarrier<PIPE_V>();remain=half;
                }
                Sub(rhs,rhs,coefficients,uint32_t(block));PipeBarrier<PIPE_V>();
            }
        }
        Adds(rhs,baseFactor,0.0f,n);PipeBarrier<PIPE_V>();factor=baseFactor;
    }
    __aicore__ inline void NormalizeOwnedPanelEarly(LocalTensor<float> tile,uint32_t cols,uint32_t row,float inv) {
        uint32_t start=row/64*64;
        Muls<float,false>(factor[start],factor[start],-inv,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
        PipeBarrier<PIPE_V>();
        if(cols) {
            float first=reinterpret_cast<__ubuf__ float*>(tile.GetPhyAddr())[row];
            Axpy<float,float,false>(tile[start],factor[start],first,uint64_t(0),1,
                                    UnaryRepeatParams{1,1,8,8});
            SetFlag<HardEvent::V_S>(static_cast<event_t>(1));
            for(uint32_t col=1;col<cols;++col) {
                float coefficient=reinterpret_cast<__ubuf__ float*>(tile.GetPhyAddr())[col*cp+row];
                Axpy<float,float,false>(tile[col*cp+start],factor[start],coefficient,uint64_t(0),1,
                                        UnaryRepeatParams{1,1,8,8});
            }
            SetFlag<HardEvent::V_S>(static_cast<event_t>(2));
        }
    }
    __aicore__ inline void NormalizeOwnedPanel(LocalTensor<float> tile,uint32_t cols,uint32_t row,float inv) {
        // The pivot-selection V_S already completed all preceding tile writes.
        // Factor normalization touches only the retired column; tile is disjoint.
        // The owner panel retains count mode, restoring it after pivot reductions.
        uint32_t start=FAMILY==3?row/64*64:0;
        Muls<float,false>(factor[start],factor[start],-inv,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
        PipeBarrier<PIPE_V>();
        for(uint32_t col=0;col<cols;++col) {
            float coefficient=reinterpret_cast<__ubuf__ float*>(tile.GetPhyAddr())[col*cp+row];
            Axpy<float,float,false>(tile[col*cp+start],factor[start],coefficient,uint64_t(0),1,
                                    UnaryRepeatParams{1,1,8,8});
        }
        // FAMILY1 joins all updates at the next pivot V_S or final packet V_MTE3.
        if constexpr(FAMILY!=1)PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void UpdateColumnsWindowReady(LocalTensor<float> tile,uint32_t cols,
                                                     uint32_t row,uint32_t readyBase) {
        auto pivot=indices[4*na].ReinterpretCast<float>();
        Gather(pivot,tile,indices,row*4,cols);PipeBarrier<PIPE_V>();
        Brcb(transposed,pivot,(cols+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
        // Only the next pivot window is split; the remaining bands keep their full fanout.
        MulAddDst(tile[readyBase],factor[readyBase],transposed,64,1,
                  BinaryRepeatParams{1,1,0,static_cast<uint8_t>(cp/8),0,1});
        SetFlag<HardEvent::V_S>(static_cast<event_t>(3));
        MulAddDst(tile[cp+readyBase],factor[readyBase],transposed[8],64,cols-1,
                  BinaryRepeatParams{1,1,0,static_cast<uint8_t>(cp/8),0,1});
        SetFlag<HardEvent::V_S>(static_cast<event_t>(4));
        for(uint32_t base=row/64*64;base<na;base+=64) {
            if(base==readyBase)continue;
            MulAddDst(tile[base],factor[base],transposed,64,cols,
                      BinaryRepeatParams{1,1,0,static_cast<uint8_t>(cp/8),0,1});
        }
    }
    __aicore__ inline void UpdateColumns(LocalTensor<float> tile,uint32_t cols,uint32_t row,float inv) {
        auto pivot=indices[4*na].ReinterpretCast<float>();
        Gather(pivot,tile,indices,row*4,cols);PipeBarrier<PIPE_V>();
        if(!parallel){Muls(pivot,pivot,-inv,cols);PipeBarrier<PIPE_V>();}
        Brcb(transposed,pivot,(cols+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
        if constexpr(FAMILY==1) {
            for(uint32_t col=0;col<cols;col+=8) {
                uint32_t count=cols-col<8?cols-col:8;
                MulAddDst(tile[col*cp],factor,transposed[col*8],count*8,na/8,
                          BinaryRepeatParams{static_cast<uint8_t>(cp/8),0,1,1,1,0});
            }
            PipeBarrier<PIPE_V>();return;
        }
        for(uint32_t base=FAMILY==3?row/64*64:0;base<na;base+=64) {
            uint32_t count=na-base<64?na-base:64;
            MulAddDst(tile[base],factor[base],transposed,count,cols,
                      BinaryRepeatParams{1,1,0,static_cast<uint8_t>(cp/8),0,1});
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RowMul(LocalTensor<float> dst,LocalTensor<float> a0,LocalTensor<float> b0,uint32_t rows) {
        for(uint32_t j=0;j<na;j+=64) {
            uint32_t count=na-j<64?na-j:64;
            Mul(dst[j],a0[j],b0[j],count,rows,
                BinaryRepeatParams{1,1,1,static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8),0});
        }
    }
    __aicore__ inline void PrepareResidualIndices() {
        CreateVecIndex(indices[352].ReinterpretCast<int32_t>(),int32_t(0),32);PipeBarrier<PIPE_V>();
        Muls(indices[352].ReinterpretCast<int32_t>(),indices[352].ReinterpretCast<int32_t>(),int32_t(32),32);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void ResidualTileOriginal(uint32_t batch,uint32_t base,uint32_t rows,uint32_t tileRows) {
        const uint32_t size=tileRows*na;
        auto original=mat;auto ah=mat[size];auto al=mat[2*size];
        auto product=mat[3*size];auto low=mat[4*size];auto temp=mat[5*size];
        auto z=mat[6*size];auto total=mat[7*size];auto other=mat[8*size];
        auto xh=factor;auto xl=column;
        auto leftIndex=indices;auto rightIndex=indices[64];auto integerTemp=indices[128].ReinterpretCast<int32_t>();
        uint64_t aoff=uint64_t(batch)*n*n+uint64_t(base)*n;
        uint64_t boff=(broadcast?0:uint64_t(batch)*n*k)+uint64_t(base)*k;
        auto savedB=input[8*na];
        if(parallel) {
            Fence<HardEvent::V_MTE2>();
            DataCopyPad(savedB,b[boff],DataCopyExtParams{static_cast<uint16_t>(rows),k*4,0,0,0},
                        DataCopyPadExtParams<float>{false,0,0,0});Fence<HardEvent::MTE2_V>();
        }
        // Reuse immutable gather tables and the complete RHS tile across its columns.
        for(uint32_t q=0;q<k;++q) {
            leftIndex=indices;rightIndex=indices[64];
            Duplicate(original,0.0f,rows*na);Fence<HardEvent::V_MTE2>();
            DataCopyPad(original,a[aoff],DataCopyExtParams{static_cast<uint16_t>(rows),n*4,0,0,0},
                        DataCopyPadExtParams<float>{false,0,0,0});Fence<HardEvent::MTE2_V>();
            RowMul(product,original,input[q*na],rows);
            ShiftRight(ah.ReinterpretCast<int32_t>(),original.ReinterpretCast<int32_t>(),int32_t(12),rows*na);
            ShiftRight(xh.ReinterpretCast<int32_t>(),input[q*na].ReinterpretCast<int32_t>(),int32_t(12),na);PipeBarrier<PIPE_V>();
            ShiftLeft(ah.ReinterpretCast<int32_t>(),ah.ReinterpretCast<int32_t>(),int32_t(12),rows*na);
            ShiftLeft(xh.ReinterpretCast<int32_t>(),xh.ReinterpretCast<int32_t>(),int32_t(12),na);PipeBarrier<PIPE_V>();
            Sub(al,original,ah,rows*na);Sub(xl,input[q*na],xh,na);PipeBarrier<PIPE_V>();
            RowMul(low,ah,xh,rows);PipeBarrier<PIPE_V>();
            Sub(low,low,product,rows*na);RowMul(temp,ah,xl,rows);PipeBarrier<PIPE_V>();
            Add(low,low,temp,rows*na);RowMul(temp,al,xh,rows);PipeBarrier<PIPE_V>();
            Add(low,low,temp,rows*na);RowMul(temp,al,xl,rows);PipeBarrier<PIPE_V>();
            Add(low,low,temp,rows*na);PipeBarrier<PIPE_V>();
            BinaryRepeatParams rr{1,1,1,static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8)};
            UnaryRepeatParams ur{1,1,static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8)};
            for(uint32_t remain=na;remain>8;) {
                uint32_t half=(remain+15)/16*8,count=remain-half;
                for(uint32_t off=0;off<count;off+=64) {
                    uint32_t len=count-off<64?count-off:64;
                    Add(total[off],product[off],product[off+half],len,rows,rr);PipeBarrier<PIPE_V>();
                    Sub(z[off],total[off],product[off],len,rows,rr);PipeBarrier<PIPE_V>();
                    Sub(temp[off],total[off],z[off],len,rows,rr);PipeBarrier<PIPE_V>();
                    Sub(temp[off],product[off],temp[off],len,rows,rr);
                    Sub(other[off],product[off+half],z[off],len,rows,rr);PipeBarrier<PIPE_V>();
                    Add(temp[off],temp[off],other[off],len,rows,rr);
                    Add(low[off],low[off],low[off+half],len,rows,rr);PipeBarrier<PIPE_V>();
                    Add(low[off],low[off],temp[off],len,rows,rr);
                    Adds(product[off],total[off],0.0f,len,rows,ur);PipeBarrier<PIPE_V>();
                }
                remain=half;
            }
            // Gather the final8 terms per row, then perform three packed TwoSum levels.
            if(!parallel) {
            CreateVecIndex(leftIndex.ReinterpretCast<int32_t>(),int32_t(0),rows*8);PipeBarrier<PIPE_V>();
            ShiftRight(integerTemp,leftIndex.ReinterpretCast<int32_t>(),int32_t(3),rows*8);PipeBarrier<PIPE_V>();
            Muls(integerTemp,integerTemp,int32_t(na-8),rows*8);PipeBarrier<PIPE_V>();
            Add(leftIndex.ReinterpretCast<int32_t>(),leftIndex.ReinterpretCast<int32_t>(),integerTemp,rows*8);PipeBarrier<PIPE_V>();
            Muls(leftIndex.ReinterpretCast<int32_t>(),leftIndex.ReinterpretCast<int32_t>(),int32_t(4),rows*8);PipeBarrier<PIPE_V>();
            }
            Gather(original,product,leftIndex,0,rows*8);Gather(ah,low,leftIndex,0,rows*8);PipeBarrier<PIPE_V>();
            for(int32_t shift=2;shift>=0;--shift) {
                uint32_t half=uint32_t(1)<<shift,count=rows*half;
                if(parallel) {
                    uint32_t offset=shift==2?128:(shift==1?256:320);
                    leftIndex=indices[offset];rightIndex=indices[offset+16*half];
                } else {
                CreateVecIndex(leftIndex.ReinterpretCast<int32_t>(),int32_t(0),count);PipeBarrier<PIPE_V>();
                ShiftRight(integerTemp,leftIndex.ReinterpretCast<int32_t>(),shift,count);PipeBarrier<PIPE_V>();
                Muls(integerTemp,integerTemp,int32_t(half),count);PipeBarrier<PIPE_V>();
                Add(leftIndex.ReinterpretCast<int32_t>(),leftIndex.ReinterpretCast<int32_t>(),integerTemp,count);PipeBarrier<PIPE_V>();
                Muls(leftIndex.ReinterpretCast<int32_t>(),leftIndex.ReinterpretCast<int32_t>(),int32_t(4),count);PipeBarrier<PIPE_V>();
                Adds(rightIndex.ReinterpretCast<int32_t>(),leftIndex.ReinterpretCast<int32_t>(),int32_t(half*4),count);PipeBarrier<PIPE_V>();
                }
                Gather(al,original,leftIndex,0,count);Gather(product,original,rightIndex,0,count);PipeBarrier<PIPE_V>();
                Add(total,al,product,count);PipeBarrier<PIPE_V>();Sub(z,total,al,count);PipeBarrier<PIPE_V>();
                Sub(temp,total,z,count);PipeBarrier<PIPE_V>();Sub(temp,al,temp,count);Sub(other,product,z,count);PipeBarrier<PIPE_V>();
                Add(temp,temp,other,count);
                Gather(al,ah,leftIndex,0,count);Gather(product,ah,rightIndex,0,count);PipeBarrier<PIPE_V>();
                Add(ah,al,product,count);PipeBarrier<PIPE_V>();Add(ah,ah,temp,count);Adds(original,total,0.0f,count);PipeBarrier<PIPE_V>();
            }
            auto originalB=other;
            if(parallel) {leftIndex=indices[352];originalB=savedB;}
            else {
            Fence<HardEvent::V_MTE2>();
            DataCopyPad(other,b[boff],DataCopyExtParams{static_cast<uint16_t>(rows),k*4,0,0,0},
                        DataCopyPadExtParams<float>{false,0,0,0});Fence<HardEvent::MTE2_V>();
            CreateVecIndex(leftIndex.ReinterpretCast<int32_t>(),int32_t(0),rows);PipeBarrier<PIPE_V>();
            Muls(leftIndex.ReinterpretCast<int32_t>(),leftIndex.ReinterpretCast<int32_t>(),int32_t(32),rows);PipeBarrier<PIPE_V>();
            }
            Gather(al,originalB,leftIndex,q*4,rows);Muls(original,original,-1.0f,rows);PipeBarrier<PIPE_V>();
            Add(total,al,original,rows);PipeBarrier<PIPE_V>();Sub(z,total,al,rows);PipeBarrier<PIPE_V>();
            Sub(temp,total,z,rows);PipeBarrier<PIPE_V>();Sub(temp,al,temp,rows);Sub(other,original,z,rows);PipeBarrier<PIPE_V>();
            Add(temp,temp,other,rows);PipeBarrier<PIPE_V>();Sub(temp,temp,ah,rows);PipeBarrier<PIPE_V>();
            Add(transposed[q*na+base],total,temp,rows);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void ResidualTile(uint32_t batch,uint32_t base,uint32_t rows,uint32_t tileRows) {
        if constexpr(FAMILY==2)ResidualTileRounded(batch,base,rows,tileRows);
        else if constexpr(FAMILY==1)ResidualTileSplitProduct(batch,base,rows,tileRows);
        else if constexpr(FAMILY==3)ResidualTileAccurate(batch,base,rows,tileRows);
        else {
            if(parallel && n>=224)ResidualTileRounded(batch,base,rows,tileRows);
            else ResidualTileAccurate(batch,base,rows,tileRows);
        }
    }
    __aicore__ inline void ResidualPlainThreeRhs32(uint32_t batch,uint32_t base) {
        constexpr uint32_t tile=32*256,rows=96;
        auto sourceA=mat,products=mat[tile];
        auto sums=mat[4*tile],originalB=mat[4*tile+128];
        auto savedB=input[8*na];
        uint64_t aoff=uint64_t(batch)*logicalN*logicalN+uint64_t(base)*logicalN;
        uint64_t boff=(broadcast?0:uint64_t(batch)*logicalN*logicalK)+uint64_t(base)*logicalK;
        uint32_t validRows=logicalN-base<32?logicalN-base:32;
        if(logicalN!=n || logicalK!=k) {
            Duplicate(savedB,0.0f,32*8);PipeBarrier<PIPE_V>();
        }
        Fence<HardEvent::V_MTE2>();
        if constexpr(PARTIAL_CAPACITY)
            DataCopyPad(savedB,b[boff],DataCopyExtParams{uint16_t(validRows),logicalK*4,0,0,0},
                        DataCopyPadExtParams<float>{true,0,uint8_t(8-logicalK),0});
        else DataCopyPad(savedB,b[boff],DataCopyExtParams{32,12,0,0,0},
                         DataCopyPadExtParams<float>{false,0,0,0});
        Duplicate(sourceA,0.0f,tile);Fence<HardEvent::V_MTE2>();
        if(logicalN==n)DataCopy(sourceA,a[aoff],tile);
        else DataCopyPad(sourceA,a[aoff],
            DataCopyExtParams{uint16_t(validRows),logicalN*4,0,(na-(logicalN+7)/8*8)/8,0},
            DataCopyPadExtParams<float>{true,0,uint8_t((8-logicalN%8)%8),0});
        Fence<HardEvent::MTE2_V>();
        for(uint32_t q=0;q<3;++q)RowMul(products[q*tile],sourceA,input[q*na],32);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams rr{1,1,1,32,32,32};
        for(uint32_t remain=256;remain>8;) {
            uint32_t half=(remain+15)/16*8,count=remain-half;
            for(uint32_t off=0;off<count;off+=64) {
                uint32_t len=count-off<64?count-off:64;
                Add(products[off],products[off],products[off+half],len,rows,rr);
            }
            PipeBarrier<PIPE_V>();remain=half;
        }
        WholeReduceSum(sums,products,8,rows,1,1,32);
        for(uint32_t q=0;q<3;++q)Gather(originalB[q*32],savedB,indices[352],q*4,32);
        PipeBarrier<PIPE_V>();
        Sub(sums,originalB,sums,rows);PipeBarrier<PIPE_V>();
        DataCopy(transposed[base],sums,DataCopyParams{3,4,0,28});PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void ResidualTileRounded(uint32_t batch,uint32_t base,uint32_t rows,uint32_t tileRows) {
        if(!parallel){ResidualTileOriginal(batch,base,rows,tileRows);return;}
        if constexpr(FAMILY==3 && STATIC_N==256 && (STATIC_WORKERS==8 || STATIC_WORKERS==10) && STATIC_K==3) {
            ResidualPlainThreeRhs32(batch,base);return;
        }
        const uint32_t size=tileRows*na;
        auto sourceA=mat;auto product=mat[size];auto low=mat[2*size];auto ah=mat[3*size];
        auto original=mat[4*size];auto al=mat[4*size+64];auto z=mat[4*size+128];
        auto total=mat[4*size+192];auto other=mat[4*size+256];auto temp=mat[4*size+320];
        auto xh=factor;auto xl=column;
        auto leftIndex=indices;auto rightIndex=indices[64];auto integerTemp=indices[128].ReinterpretCast<int32_t>();
        uint64_t aoff=uint64_t(batch)*n*n+uint64_t(base)*n;
        uint64_t boff=(broadcast?0:uint64_t(batch)*n*k)+uint64_t(base)*k;
        auto savedB=input[8*na];
        if(parallel) {
            Fence<HardEvent::V_MTE2>();
            DataCopyPad(savedB,b[boff],DataCopyExtParams{static_cast<uint16_t>(rows),k*4,0,0,0},
                        DataCopyPadExtParams<float>{false,0,0,0});Fence<HardEvent::MTE2_V>();
        }
        Duplicate(sourceA,0.0f,rows*na);Fence<HardEvent::V_MTE2>();
        DataCopyPad(sourceA,a[aoff],DataCopyExtParams{static_cast<uint16_t>(rows),n*4,0,0,0},
                    DataCopyPadExtParams<float>{false,0,0,0});Fence<HardEvent::MTE2_V>();
        // Reuse immutable gather tables and the complete RHS tile across its columns.
        for(uint32_t q=0;q<k;++q) {
            leftIndex=indices;rightIndex=indices[64];
            RowMul(product,sourceA,input[q*na],rows);
            PipeBarrier<PIPE_V>();
            // Full tiles stay live through reduction; row outputs reuse a compact tail.
            ShiftRight(ah.ReinterpretCast<int32_t>(),product.ReinterpretCast<int32_t>(),int32_t(15),rows*na);PipeBarrier<PIPE_V>();
            ShiftLeft(ah.ReinterpretCast<int32_t>(),ah.ReinterpretCast<int32_t>(),int32_t(15),rows*na);PipeBarrier<PIPE_V>();
            Sub(low,product,ah,rows*na);PipeBarrier<PIPE_V>();
            Adds(product,ah,0.0f,rows*na);PipeBarrier<PIPE_V>();
            BinaryRepeatParams rr{1,1,1,static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8)};
            for(uint32_t remain=na;remain>8;) {
                uint32_t half=(remain+15)/16*8,count=remain-half;
                for(uint32_t off=0;off<count;off+=64) {
                    uint32_t len=count-off<64?count-off:64;
                    Add(product[off],product[off],product[off+half],len,rows,rr);
                    Add(low[off],low[off],low[off+half],len,rows,rr);
                }
                PipeBarrier<PIPE_V>();remain=half;
            }
            // The remaining eight terms already occupy one block per row.
            // Reduce them in place in the row layout, avoiding compaction tables.
            WholeReduceSum(original,product,8,rows,1,1,na/8);
            WholeReduceSum(ah,low,8,rows,1,1,na/8);PipeBarrier<PIPE_V>();
            auto originalB=other;
            if(parallel) {leftIndex=indices[352];originalB=savedB;}
            else {
            Fence<HardEvent::V_MTE2>();
            DataCopyPad(other,b[boff],DataCopyExtParams{static_cast<uint16_t>(rows),k*4,0,0,0},
                        DataCopyPadExtParams<float>{false,0,0,0});Fence<HardEvent::MTE2_V>();
            CreateVecIndex(leftIndex.ReinterpretCast<int32_t>(),int32_t(0),rows);PipeBarrier<PIPE_V>();
            Muls(leftIndex.ReinterpretCast<int32_t>(),leftIndex.ReinterpretCast<int32_t>(),int32_t(32),rows);PipeBarrier<PIPE_V>();
            }
            Gather(al,originalB,leftIndex,q*4,rows);Muls(original,original,-1.0f,rows);PipeBarrier<PIPE_V>();
            Add(total,al,original,rows);PipeBarrier<PIPE_V>();Sub(z,total,al,rows);PipeBarrier<PIPE_V>();
            Sub(temp,total,z,rows);PipeBarrier<PIPE_V>();Sub(temp,al,temp,rows);Sub(other,original,z,rows);PipeBarrier<PIPE_V>();
            Add(temp,temp,other,rows);PipeBarrier<PIPE_V>();Sub(temp,temp,ah,rows);PipeBarrier<PIPE_V>();
            Add(transposed[q*na+base],total,temp,rows);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void ResidualTileAccurate(uint32_t batch,uint32_t base,uint32_t rows,uint32_t tileRows) {
        if(!parallel){ResidualTileOriginal(batch,base,rows,tileRows);return;}
        const uint32_t size=tileRows*na;
        auto sourceA=mat;auto sourceHigh=mat[size];auto sourceLow=mat[2*size];
        auto product=mat[3*size];auto low=mat[4*size];auto temp=mat[5*size];auto ah=mat[6*size];
        auto original=mat[7*size];auto al=mat[7*size+64];auto z=mat[7*size+128];
        auto total=mat[7*size+192];auto other=mat[7*size+256];
        auto xh=factor;auto xl=column;
        auto leftIndex=indices;auto rightIndex=indices[64];auto integerTemp=indices[128].ReinterpretCast<int32_t>();
        uint64_t aoff=uint64_t(batch)*n*n+uint64_t(base)*n;
        uint64_t boff=(broadcast?0:uint64_t(batch)*n*k)+uint64_t(base)*k;
        auto savedB=input[8*na];
        if(parallel) {
            Fence<HardEvent::V_MTE2>();
            DataCopyPad(savedB,b[boff],DataCopyExtParams{static_cast<uint16_t>(rows),k*4,0,0,0},
                        DataCopyPadExtParams<float>{false,0,0,0});Fence<HardEvent::MTE2_V>();
        }
        {
            Duplicate(sourceA,0.0f,rows*na);Fence<HardEvent::V_MTE2>();
            DataCopyPad(sourceA,a[aoff],DataCopyExtParams{static_cast<uint16_t>(rows),n*4,0,0,0},
                        DataCopyPadExtParams<float>{false,0,0,0});Fence<HardEvent::MTE2_V>();
            ShiftRight(sourceHigh.ReinterpretCast<int32_t>(),sourceA.ReinterpretCast<int32_t>(),int32_t(12),rows*na);PipeBarrier<PIPE_V>();
            ShiftLeft(sourceHigh.ReinterpretCast<int32_t>(),sourceHigh.ReinterpretCast<int32_t>(),int32_t(12),rows*na);PipeBarrier<PIPE_V>();
            Sub(sourceLow,sourceA,sourceHigh,rows*na);PipeBarrier<PIPE_V>();
        }
        // Reuse immutable gather tables and the complete RHS tile across its columns.
        for(uint32_t q=0;q<k;++q) {
            leftIndex=indices;rightIndex=indices[64];
            RowMul(product,sourceA,input[q*na],rows);
            ShiftRight(xh.ReinterpretCast<int32_t>(),input[q*na].ReinterpretCast<int32_t>(),int32_t(12),na);PipeBarrier<PIPE_V>();
            ShiftLeft(xh.ReinterpretCast<int32_t>(),xh.ReinterpretCast<int32_t>(),int32_t(12),na);PipeBarrier<PIPE_V>();
            Sub(xl,input[q*na],xh,na);PipeBarrier<PIPE_V>();
            RowMul(low,sourceHigh,xh,rows);PipeBarrier<PIPE_V>();
            Sub(low,low,product,rows*na);RowMul(temp,sourceHigh,xl,rows);PipeBarrier<PIPE_V>();
            Add(low,low,temp,rows*na);RowMul(temp,sourceLow,xh,rows);PipeBarrier<PIPE_V>();
            Add(low,low,temp,rows*na);RowMul(temp,sourceLow,xl,rows);PipeBarrier<PIPE_V>();
            Add(low,low,temp,rows*na);PipeBarrier<PIPE_V>();
            // Split the rounded product before reductions; retain its low part and product error.
            ShiftRight(ah.ReinterpretCast<int32_t>(),product.ReinterpretCast<int32_t>(),int32_t(15),rows*na);PipeBarrier<PIPE_V>();
            ShiftLeft(ah.ReinterpretCast<int32_t>(),ah.ReinterpretCast<int32_t>(),int32_t(15),rows*na);PipeBarrier<PIPE_V>();
            Sub(temp,product,ah,rows*na);PipeBarrier<PIPE_V>();
            Add(low,low,temp,rows*na);Adds(product,ah,0.0f,rows*na);PipeBarrier<PIPE_V>();
            BinaryRepeatParams rr{1,1,1,static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8)};
            for(uint32_t remain=na;remain>8;) {
                uint32_t half=(remain+15)/16*8,count=remain-half;
                for(uint32_t off=0;off<count;off+=64) {
                    uint32_t len=count-off<64?count-off:64;
                    Add(product[off],product[off],product[off+half],len,rows,rr);
                    Add(low[off],low[off],low[off+half],len,rows,rr);
                }
                PipeBarrier<PIPE_V>();remain=half;
            }
            // The remaining eight terms already occupy one block per row.
            // Reduce them in place in the row layout, avoiding compaction tables.
            WholeReduceSum(original,product,8,rows,1,1,na/8);
            WholeReduceSum(ah,low,8,rows,1,1,na/8);PipeBarrier<PIPE_V>();
            auto originalB=other;
            if(parallel) {leftIndex=indices[352];originalB=savedB;}
            else {
            Fence<HardEvent::V_MTE2>();
            DataCopyPad(other,b[boff],DataCopyExtParams{static_cast<uint16_t>(rows),k*4,0,0,0},
                        DataCopyPadExtParams<float>{false,0,0,0});Fence<HardEvent::MTE2_V>();
            CreateVecIndex(leftIndex.ReinterpretCast<int32_t>(),int32_t(0),rows);PipeBarrier<PIPE_V>();
            Muls(leftIndex.ReinterpretCast<int32_t>(),leftIndex.ReinterpretCast<int32_t>(),int32_t(32),rows);PipeBarrier<PIPE_V>();
            }
            Gather(al,originalB,leftIndex,q*4,rows);Muls(original,original,-1.0f,rows);PipeBarrier<PIPE_V>();
            Add(total,al,original,rows);PipeBarrier<PIPE_V>();Sub(z,total,al,rows);PipeBarrier<PIPE_V>();
            Sub(temp,total,z,rows);PipeBarrier<PIPE_V>();Sub(temp,al,temp,rows);Sub(other,original,z,rows);PipeBarrier<PIPE_V>();
            Add(temp,temp,other,rows);PipeBarrier<PIPE_V>();Sub(temp,temp,ah,rows);PipeBarrier<PIPE_V>();
            Add(transposed[q*na+base],total,temp,rows);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void ResidualJointFour(uint32_t batch,uint32_t base,uint32_t rows) {
        const uint32_t size=rows*na,allRows=rows*k;
        auto sourceA=mat;auto sourceHigh=mat[size];auto sourceLow=mat[2*size];
        auto product=mat[3*size];auto low=mat[(3+k)*size];auto temp=mat[(3+2*k)*size];
        auto xh=factor;auto xl=column;auto savedB=input[8*na];
        uint64_t aoff=uint64_t(batch)*n*n+uint64_t(base)*n;
        uint64_t boff=(broadcast?0:uint64_t(batch)*n*k)+uint64_t(base)*k;
        Fence<HardEvent::V_MTE2>();
        DataCopyPad(savedB,b[boff],DataCopyExtParams{static_cast<uint16_t>(rows),k*4,0,0,0},
                    DataCopyPadExtParams<float>{false,0,0,0});Fence<HardEvent::MTE2_V>();
        Duplicate(sourceA,0.0f,size);Fence<HardEvent::V_MTE2>();
        DataCopyPad(sourceA,a[aoff],DataCopyExtParams{static_cast<uint16_t>(rows),n*4,0,0,0},
                    DataCopyPadExtParams<float>{false,0,0,0});Fence<HardEvent::MTE2_V>();
        ShiftRight(sourceHigh.ReinterpretCast<int32_t>(),sourceA.ReinterpretCast<int32_t>(),int32_t(20),size);
        PipeBarrier<PIPE_V>();
        ShiftLeft(sourceHigh.ReinterpretCast<int32_t>(),sourceHigh.ReinterpretCast<int32_t>(),int32_t(20),size);
        PipeBarrier<PIPE_V>();Sub(sourceLow,sourceA,sourceHigh,size);PipeBarrier<PIPE_V>();
        for(uint32_t q=0;q<k;++q) {
            ShiftRight(xh.ReinterpretCast<int32_t>(),input[q*na].ReinterpretCast<int32_t>(),int32_t(20),na);
            PipeBarrier<PIPE_V>();
            ShiftLeft(xh.ReinterpretCast<int32_t>(),xh.ReinterpretCast<int32_t>(),int32_t(20),na);
            PipeBarrier<PIPE_V>();Sub(xl,input[q*na],xh,na);PipeBarrier<PIPE_V>();
            RowMul(product[q*size],sourceHigh,xh,rows);
            RowMul(low[q*size],sourceLow,xh,rows);
            RowMul(temp,sourceA,xl,rows);PipeBarrier<PIPE_V>();
            Add(low[q*size],low[q*size],temp,size);PipeBarrier<PIPE_V>();
        }
        BinaryRepeatParams rr{1,1,1,static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8)};
        for(uint32_t remain=na;remain>8;) {
            uint32_t half=(remain+15)/16*8,count=remain-half;
            for(uint32_t off=0;off<count;off+=64) {
                uint32_t len=count-off<64?count-off:64;
                Add(product[off],product[off],product[off+half],len,allRows,rr);
                Add(low[off],low[off],low[off+half],len,allRows,rr);
            }
            PipeBarrier<PIPE_V>();remain=half;
        }
        // Input A has no remaining readers; reuse its storage for the short finish.
        auto highSum=mat;auto lowSum=mat[64];auto originalB=mat[128];
        auto total=mat[192];auto z=mat[256];auto other=mat[320];
        WholeReduceSum(highSum,product,8,allRows,1,1,na/8);
        WholeReduceSum(lowSum,low,8,allRows,1,1,na/8);
        auto offsets=indices.ReinterpretCast<int32_t>();
        for(uint32_t q=0;q<k;++q)Adds(offsets[q*rows],indices[352].ReinterpretCast<int32_t>(),int32_t(q*4),rows);
        PipeBarrier<PIPE_V>();
        Gather(originalB,savedB,indices,0,allRows);
        Muls(highSum,highSum,-1.0f,allRows);PipeBarrier<PIPE_V>();
        Add(total,originalB,highSum,allRows);PipeBarrier<PIPE_V>();
        Sub(z,total,originalB,allRows);PipeBarrier<PIPE_V>();
        Sub(temp,total,z,allRows);PipeBarrier<PIPE_V>();
        Sub(temp,originalB,temp,allRows);Sub(other,highSum,z,allRows);PipeBarrier<PIPE_V>();
        Add(temp,temp,other,allRows);PipeBarrier<PIPE_V>();
        Sub(temp,temp,lowSum,allRows);PipeBarrier<PIPE_V>();
        Add(z,total,temp,allRows);PipeBarrier<PIPE_V>();
        for(uint32_t q=0;q<k;++q)DataCopy(transposed[q*na+base],z[q*rows],rows);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void ResidualTileSplitProduct(uint32_t batch,uint32_t base,uint32_t rows,uint32_t tileRows) {
        if(parallel && n<=200 && n%8==0 && k<=4){ResidualJointFour(batch,base,rows);return;}
        if(!parallel){ResidualTileOriginal(batch,base,rows,tileRows);return;}
        const uint32_t size=tileRows*na;
        auto sourceA=mat;auto sourceHigh=mat[size];auto sourceLow=mat[2*size];
        auto product=mat[3*size];auto low=mat[4*size];auto temp=mat[5*size];auto ah=mat[6*size];
        auto original=mat[7*size];auto al=mat[7*size+64];auto z=mat[7*size+128];
        auto total=mat[7*size+192];auto other=mat[7*size+256];
        auto xh=factor;auto xl=column;
        auto leftIndex=indices;auto rightIndex=indices[64];auto integerTemp=indices[128].ReinterpretCast<int32_t>();
        uint64_t aoff=uint64_t(batch)*n*n+uint64_t(base)*n;
        uint64_t boff=(broadcast?0:uint64_t(batch)*n*k)+uint64_t(base)*k;
        auto savedB=input[8*na];
        if(parallel) {
            Fence<HardEvent::V_MTE2>();
            DataCopyPad(savedB,b[boff],DataCopyExtParams{static_cast<uint16_t>(rows),k*4,0,0,0},
                        DataCopyPadExtParams<float>{false,0,0,0});Fence<HardEvent::MTE2_V>();
        }
        {
            Duplicate(sourceA,0.0f,rows*na);Fence<HardEvent::V_MTE2>();
            DataCopyPad(sourceA,a[aoff],DataCopyExtParams{static_cast<uint16_t>(rows),n*4,0,0,0},
                        DataCopyPadExtParams<float>{false,0,0,0});Fence<HardEvent::MTE2_V>();
            ShiftRight(sourceHigh.ReinterpretCast<int32_t>(),sourceA.ReinterpretCast<int32_t>(),int32_t(20),rows*na);PipeBarrier<PIPE_V>();
            ShiftLeft(sourceHigh.ReinterpretCast<int32_t>(),sourceHigh.ReinterpretCast<int32_t>(),int32_t(20),rows*na);PipeBarrier<PIPE_V>();
            Sub(sourceLow,sourceA,sourceHigh,rows*na);PipeBarrier<PIPE_V>();
        }
        // Reuse immutable gather tables and the complete RHS tile across its columns.
        for(uint32_t q=0;q<k;++q) {
            leftIndex=indices;rightIndex=indices[64];
            ShiftRight(xh.ReinterpretCast<int32_t>(),input[q*na].ReinterpretCast<int32_t>(),int32_t(20),na);PipeBarrier<PIPE_V>();
            ShiftLeft(xh.ReinterpretCast<int32_t>(),xh.ReinterpretCast<int32_t>(),int32_t(20),na);PipeBarrier<PIPE_V>();
            Sub(xl,input[q*na],xh,na);PipeBarrier<PIPE_V>();
            // HH uses four significant bits per operand. Keep both remaining
            // contributions as Al*Xh + A*Xl, including the low*low term.
            RowMul(product,sourceHigh,xh,rows);
            RowMul(low,sourceLow,xh,rows);
            RowMul(temp,sourceA,xl,rows);PipeBarrier<PIPE_V>();
            Add(low,low,temp,rows*na);PipeBarrier<PIPE_V>();
            BinaryRepeatParams rr{1,1,1,static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8)};
            for(uint32_t remain=na;remain>8;) {
                uint32_t half=(remain+15)/16*8,count=remain-half;
                for(uint32_t off=0;off<count;off+=64) {
                    uint32_t len=count-off<64?count-off:64;
                    Add(product[off],product[off],product[off+half],len,rows,rr);
                    Add(low[off],low[off],low[off+half],len,rows,rr);
                }
                PipeBarrier<PIPE_V>();remain=half;
            }
            // The remaining eight terms already occupy one block per row.
            // Reduce them in place in the row layout, avoiding compaction tables.
            WholeReduceSum(original,product,8,rows,1,1,na/8);
            WholeReduceSum(ah,low,8,rows,1,1,na/8);PipeBarrier<PIPE_V>();
            auto originalB=other;
            if(parallel) {leftIndex=indices[352];originalB=savedB;}
            else {
            Fence<HardEvent::V_MTE2>();
            DataCopyPad(other,b[boff],DataCopyExtParams{static_cast<uint16_t>(rows),k*4,0,0,0},
                        DataCopyPadExtParams<float>{false,0,0,0});Fence<HardEvent::MTE2_V>();
            CreateVecIndex(leftIndex.ReinterpretCast<int32_t>(),int32_t(0),rows);PipeBarrier<PIPE_V>();
            Muls(leftIndex.ReinterpretCast<int32_t>(),leftIndex.ReinterpretCast<int32_t>(),int32_t(32),rows);PipeBarrier<PIPE_V>();
            }
            Gather(al,originalB,leftIndex,q*4,rows);Muls(original,original,-1.0f,rows);PipeBarrier<PIPE_V>();
            Add(total,al,original,rows);PipeBarrier<PIPE_V>();Sub(z,total,al,rows);PipeBarrier<PIPE_V>();
            Sub(temp,total,z,rows);PipeBarrier<PIPE_V>();Sub(temp,al,temp,rows);Sub(other,original,z,rows);PipeBarrier<PIPE_V>();
            Add(temp,temp,other,rows);PipeBarrier<PIPE_V>();Sub(temp,temp,ah,rows);PipeBarrier<PIPE_V>();
            Add(transposed[q*na+base],total,temp,rows);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void AddWide(float term,float& sum,float& low) {
        volatile float total=sum+term;
        volatile float z=total-sum;
        volatile float kept=total-z;
        volatile float left=sum-kept;
        volatile float right=term-z;
        volatile float error=left+right;
        low=low+error;sum=total;
    }
    __aicore__ inline void Assemble(LocalTensor<float> rhs,LocalTensor<float> target,bool add,uint32_t rhsPitch=0,bool scale=true,bool useOrder=true) {
        auto offsets=indices.ReinterpretCast<int32_t>();
        auto pattern=factor.ReinterpretCast<int32_t>();
        for(uint32_t q=0;q<8;++q)pattern.SetValue(q,(q<k?q:0)*(rhsPitch?rhsPitch:na)*4);
        Fence<HardEvent::S_V>();
        if(useOrder)Brcb(offsets.ReinterpretCast<float>(),order.ReinterpretCast<float>(),na/8,BrcbRepeatParams{1,8});
        else {
            CreateVecIndex(column.ReinterpretCast<int32_t>(),int32_t(0),na);PipeBarrier<PIPE_V>();
            Brcb(offsets.ReinterpretCast<float>(),column,na/8,BrcbRepeatParams{1,8});
        }
        PipeBarrier<PIPE_V>();Muls(offsets,offsets,int32_t(4),n*8);PipeBarrier<PIPE_V>();
        Add(offsets,offsets,pattern,64,na/8,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();
        Gather(target,rhs,indices,0,n*8);PipeBarrier<PIPE_V>();
        if(scale) {
            Brcb(input,inverse,na/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            Mul(target,target,input,n*8);PipeBarrier<PIPE_V>();
        }
        if(add){Add(output,output,target,n*8);PipeBarrier<PIPE_V>();}
    }
    __aicore__ inline void CorrectSingleRhs() {
        auto rhs=transposed;
        Fence<HardEvent::V_MTE2>();DataCopy(mat,w,(n<64?n:64)*na);
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,na);
        for(uint32_t block=0;block<n;block+=64) {
            uint32_t stop=block+64<n?block+64:n;
            uint32_t slot=(block/64)&1u;
            Fence<HardEvent::MTE2_V>();
            if(stop<n) {
                uint32_t count=n-stop<64?n-stop:64;
                Fence<HardEvent::V_MTE2>();
                DataCopy(mat[(slot^1u)*64*na],w[stop*na],count*na);
            }
            for(uint32_t p=block;p<stop;) {
                if constexpr(FAMILY==1) {
                    Fence<HardEvent::V_S>();float first=rhs.GetValue(p);
                    float second=p+1<stop?rhs.GetValue(p+1):0.0f;
                    Axpy<float,float,false>(rhs,mat[(slot*64+p-block)*na],first,uint64_t(0),1,
                                            UnaryRepeatParams{1,1,8,8});
                    PipeBarrier<PIPE_V>();
                    if(p+1<stop) {
                        Axpy<float,float,false>(rhs,mat[(slot*64+p+1-block)*na],second,uint64_t(0),1,
                                                UnaryRepeatParams{1,1,8,8});
                        PipeBarrier<PIPE_V>();
                    }
                    p+=2;
                } else {
                uint32_t row;
                if constexpr(FAMILY==2)row=order.GetValue(p);
                else row=n<224?p:order.GetValue(p);
                Fence<HardEvent::V_S>();float value=rhs.GetValue(row);
                Axpy<float,float,false>(rhs,mat[(slot*64+p-block)*na],value,uint64_t(0),1,
                                        UnaryRepeatParams{1,1,8,8});
                PipeBarrier<PIPE_V>();
                ++p;
                }
            }
        }
        ResetMask();SetMaskNorm();
        factor=baseFactor;
    }
    __aicore__ inline void ApplyCorrection() {
        auto rhs=transposed;
        // Reuse the known inverse and update all RHS columns with one outer product.
        auto baseIndex=indices.ReinterpretCast<int32_t>();
        CreateVecIndex(baseIndex,int32_t(0),8);PipeBarrier<PIPE_V>();
        Muls(baseIndex,baseIndex,int32_t(na*4),8);Fence<HardEvent::V_S>();
        auto pivot=indices[na].ReinterpretCast<float>();
        auto spread=indices[2*na].ReinterpretCast<float>();
        for(uint32_t block=0;block<n;block+=64) {
            uint32_t stop=block+64<n?block+64:n;
            if(parallel) {
                Fence<HardEvent::V_MTE2>();DataCopy(mat,w[block*na],(stop-block)*na);
                Fence<HardEvent::MTE2_V>();
            }
            for(uint32_t p=block;p<stop;++p) {
            uint32_t pivotRow=order.GetValue(p);float inv=inverse.GetValue(p);
            if(parallel) factor=mat[(p-block)*na];
            else {
                Fence<HardEvent::V_MTE2>();DataCopy(factor,w[p*na],na);
                Fence<HardEvent::MTE2_S>();
                factor.SetValue(pivotRow,0.0f);Fence<HardEvent::S_V>();
            }
            Gather(pivot,rhs,indices,pivotRow*4,k);PipeBarrier<PIPE_V>();
            if(!parallel){Muls(pivot,pivot,-inv,k);PipeBarrier<PIPE_V>();}
            Brcb(spread,pivot,1,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            for(uint32_t base=0;base<na;base+=64) {
                uint32_t count=na-base<64?na-base:64;
                MulAddDst(rhs[base],factor[base],spread,count,k,
                          BinaryRepeatParams{1,1,0,static_cast<uint8_t>(na/8),0,1});
            }
            PipeBarrier<PIPE_V>();
            }
        }
        factor=baseFactor;
        Assemble(rhs,mat,true);
    }
    __aicore__ inline void Refine(uint32_t batch) {
        // Spill retained coefficients once, reusing the resident matrix UB for batched residuals.
        Fence<HardEvent::S_V>();
        Fence<HardEvent::V_MTE3>();DataCopy(w[firstResident*na],mat,(n+k-firstResident)*na);
        Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_MTE2>();
        uint32_t tileRows=n+k-firstResident>=144?16:8;
        firstResident=n+k;
        auto xt=input;auto rhs=transposed;
        Duplicate(xt,0.0f,k*na);
        CreateVecIndex(indices.ReinterpretCast<int32_t>(),int32_t(0),na);PipeBarrier<PIPE_V>();
        Muls(indices.ReinterpretCast<int32_t>(),indices.ReinterpretCast<int32_t>(),int32_t(32),na);PipeBarrier<PIPE_V>();
        for(uint32_t q=0;q<k;++q) Gather(xt[q*na],output,indices,q*4,n);
        PipeBarrier<PIPE_V>();Duplicate(rhs,0.0f,k*na);PipeBarrier<PIPE_V>();
        for(uint32_t base=0;base<n;base+=tileRows)
            ResidualTile(batch,base,n-base<tileRows?n-base:tileRows,tileRows);
        ApplyCorrection();

    }
private:
    GlobalTensor<uint32_t> publication;
    LocalTensor<float> mat,input,transposed,factor,column,output,inverse,baseFactor;LocalTensor<uint32_t> indices,order;
    GlobalTensor<float> a,b,x,w;
    uint32_t n,k,na,cp,logicalN,logicalK,batches,cores,firstResident,initialResident,spillGroup,workers;bool broadcast,parallel;
};
// FP32 transpose: 16 source rows x 8 columns per repeat.
__aicore__ inline void TransposeSmallColumns(LocalTensor<float> dst,LocalTensor<float> src,
                                            uint32_t rows,uint32_t sourcePitch,uint32_t targetPitch,
                                            uint32_t columns) {
    for(uint32_t block=0;block<rows;block+=16) {
        uint64_t source[16],target[16];
        #pragma unroll
        for(uint32_t i=0;i<16;++i) {
            source[i]=reinterpret_cast<uint64_t>(src[(block+i)*sourcePitch].GetPhyAddr());
            target[i]=reinterpret_cast<uint64_t>(dst[(i/2)*targetPitch+block+(i%2)*8].GetPhyAddr());
        }
        TransDataTo5HD<float>(target,source,TransDataTo5HDParams{false,false,
            static_cast<uint8_t>(columns/8),static_cast<uint16_t>(targetPitch),1});
    }
}
template<bool BLOCK_TRANSPOSE,bool FULL_WIDTH=false> class KernelSmallOuterT {
public:
    __aicore__ inline ~KernelSmallOuterT() {PipeBarrier<PIPE_ALL>();}
    template<HardEvent event> __aicore__ inline void Fence() {
        const event_t id=static_cast<event_t>(0);
        SetFlag<event>(id);WaitFlag<event>(id);
    }
    __aicore__ inline void Init(GM_ADDR A,GM_ADDR B,GM_ADDR X,const LinalgSolveTilingData& t) {
        n=t.n;k=t.k;pitch=FULL_WIDTH?64:8;if constexpr(!FULL_WIDTH){while(pitch<n)pitch*=2;}stride=pitch+8;cols=(n+k+7)/8*8;batches=t.batches;cores=t.cores;broadcast=t.broadcastB;
        a.SetGlobalBuffer((__gm__ float*)A);b.SetGlobalBuffer((__gm__ float*)B);x.SetGlobalBuffer((__gm__ float*)X);
        SetMaskNorm();SetAtomicNone();
        mat=LocalTensor<float>(TPosition::VECCALC,0,5184);
        input=LocalTensor<float>(TPosition::VECCALC,20736,4096);
        transposed=LocalTensor<float>(TPosition::VECCALC,37120,4096);
        transposeIndex=LocalTensor<uint32_t>(TPosition::VECCALC,53504,4096);
        pivot=LocalTensor<float>(TPosition::VECCALC,69888,72);
        spread=LocalTensor<float>(TPosition::VECCALC,70176,576);
        baseIndex=LocalTensor<int32_t>(TPosition::VECCALC,72480,72);
        absolute=LocalTensor<float>(TPosition::VECCALC,72768,64);
        reduced=LocalTensor<float>(TPosition::VECCALC,73024,8);
        order=LocalTensor<uint32_t>(TPosition::VECCALC,73056,64);
        inverse=LocalTensor<float>(TPosition::VECCALC,73312,64);
        output=LocalTensor<float>(TPosition::VECCALC,73568,512);
    }
    __aicore__ inline void MakeIndex(LocalTensor<int32_t> target,uint32_t count) {
        for(uint32_t i=0;i<8;++i)target.SetValue(i,int32_t(i));
        Fence<HardEvent::S_V>();
        for(uint32_t filled=8;filled<count;filled*=2) {
            uint32_t len=count-filled<filled?count-filled:filled;
            Adds(target[filled],target,int32_t(filled),len);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void LoadMatrix(GlobalTensor<float> source,uint64_t offset,uint32_t count,uint32_t first) {
        if constexpr(BLOCK_TRANSPOSE) {
            if(first==0 && count==n) {
                uint32_t sourcePitch=(n+7)/8*8;
                DataCopyPad(input,source[offset],DataCopyExtParams{static_cast<uint16_t>(n),n*4,0,0,0},
                            DataCopyPadExtParams<float>{false,0,0,0});
                Fence<HardEvent::MTE2_V>();
                TransposeSmallColumns(mat,input,pitch,sourcePitch,stride,sourcePitch);
                PipeBarrier<PIPE_V>();return;
            }
        }
        uint32_t size=count*pitch;
        Fence<HardEvent::V_MTE2>();
        DataCopyPad(input,source[offset],DataCopyExtParams{1,n*count*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
        auto offsets=transposeIndex.ReinterpretCast<int32_t>();
        Fence<HardEvent::V_S>();
        for(uint32_t i=0;i<8;++i)reinterpret_cast<__ubuf__ int32_t*>(offsets.GetPhyAddr())[i]=int32_t(i*count*4);
        Fence<HardEvent::S_V>();
        for(uint32_t filled=8;filled<pitch;filled*=2) {
            Adds(offsets[filled],offsets,int32_t(filled*count*4),filled);PipeBarrier<PIPE_V>();
        }
        for(uint32_t filled=pitch;filled<size;filled*=2) {
            uint32_t len=size-filled<filled?size-filled:filled;
            Adds(offsets[filled],offsets,int32_t(filled/pitch*4),len);PipeBarrier<PIPE_V>();
        }
        Fence<HardEvent::MTE2_V>();Gather(transposed,input,transposeIndex,0,size);PipeBarrier<PIPE_V>();
        Adds(mat[first*stride],transposed,0.0f,pitch,count,
             UnaryRepeatParams{1,1,static_cast<uint8_t>(stride/8),static_cast<uint8_t>(pitch/8)});
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Process() {
        uint32_t owners=cores<batches?cores:batches;
        for(uint32_t batch=GetBlockIdx()%owners;batch<batches;batch+=owners) {
            MakeIndex(baseIndex,cols);PipeBarrier<PIPE_V>();
            Muls(baseIndex,baseIndex,static_cast<int32_t>(stride*4),cols);PipeBarrier<PIPE_V>();
            LoadMatrix(a,uint64_t(batch)*n*n,n,0);
            LoadMatrix(b,broadcast?0:uint64_t(batch)*n*k,k,n);
            uint64_t activeRows[2]={n==64?~uint64_t(0):(uint64_t(1)<<n)-1,0};
            for(uint32_t p=0;p<n;++p) {
                factor=mat[p*stride];
                if constexpr(FULL_WIDTH && BLOCK_TRANSPOSE) {
                    if(p==0) {
                        WholeReduceMax(reduced,factor,activeRows,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
                        SetFlag<HardEvent::V_S>(static_cast<event_t>(1));
                    }
                } else if constexpr(FULL_WIDTH)
                    WholeReduceMax(reduced,factor,activeRows,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
                else {
                    Abs(absolute,factor,pitch);PipeBarrier<PIPE_V>();
                    WholeReduceMax(reduced,absolute,activeRows,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
                }
                if constexpr(FULL_WIDTH && BLOCK_TRANSPOSE)
                    WaitFlag<HardEvent::V_S>(static_cast<event_t>(1));
                else Fence<HardEvent::V_S>();
                uint32_t row;float diagonal;
                if constexpr(FULL_WIDTH) {
                    // ORDER_VALUE_INDEX already contains the signed pivot value.
                    uint64_t packet=reinterpret_cast<__ubuf__ uint64_t*>(reduced.GetPhyAddr())[0];
                    union {uint32_t bits;float value;} word;
                    word.bits=static_cast<uint32_t>(packet);
                    row=static_cast<uint32_t>(packet>>32);diagonal=word.value;
                } else {
                    row=reinterpret_cast<__ubuf__ uint32_t*>(reduced.GetPhyAddr())[1];
                    diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[row];
                }
                uint32_t remaining=n+k-p-1;
                // The reduced next column is ready independently of later FMA
                // destinations. Join those updates before Gather reads them.
                if constexpr(FULL_WIDTH && BLOCK_TRANSPOSE)PipeBarrier<PIPE_V>();
                if constexpr(FULL_WIDTH)
                    Gather(pivot,mat[(p+1)*stride],baseIndex.ReinterpretCast<uint32_t>(),row*4,remaining);
                if constexpr(FULL_WIDTH) {
                    if(diagonal==0.0f) {
                        Abs(absolute,factor,pitch);PipeBarrier<PIPE_V>();
                        WholeReduceMax(reduced,absolute,activeRows,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
                        Fence<HardEvent::V_S>();row=reinterpret_cast<__ubuf__ uint32_t*>(reduced.GetPhyAddr())[1];
                        diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[row];
                        Gather(pivot,mat[(p+1)*stride],baseIndex.ReinterpretCast<uint32_t>(),row*4,remaining);
                    }
                }
                reinterpret_cast<__ubuf__ uint32_t*>(order.GetPhyAddr())[p]=row;activeRows[0]&=~(uint64_t(1)<<row);
                float inv=1.0f/diagonal;reinterpret_cast<__ubuf__ float*>(inverse.GetPhyAddr())[p]=inv;reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[row]=0.0f;
                Fence<HardEvent::S_V>();
                if constexpr(BLOCK_TRANSPOSE) {
                    if constexpr(FULL_WIDTH)Muls<float>(factor,factor,-inv,uint64_t(64),1,UnaryRepeatParams{1,1,8,8});
                    else Muls(factor,factor,-inv,pitch);
                    if constexpr(!FULL_WIDTH)
                        Gather(pivot,mat[(p+1)*stride],baseIndex.ReinterpretCast<uint32_t>(),row*4,remaining);
                    PipeBarrier<PIPE_V>();
                    Brcb(spread,pivot,(remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                    if constexpr(FULL_WIDTH) {
                        if(p+1<n) {
                            auto next=mat[(p+1)*stride];
                            MulAddDst<float,float,false>(next,factor,spread,uint64_t(64),1,BinaryRepeatParams{1,1,0,9,0,1});
                            PipeBarrier<PIPE_V>();
                            WholeReduceMax(reduced,next,activeRows,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
                            SetFlag<HardEvent::V_S>(static_cast<event_t>(1));
                            // Reduction changes the mask. Restore all64 rows
                            // for the disjoint remaining-column updates.
                            MulAddDst<float,float,true>(next[stride],factor,spread[8],uint64_t(64),remaining-1,BinaryRepeatParams{1,1,0,9,0,1});
                        } else MulAddDst<float,float,false>(mat[(p+1)*stride],factor,spread,uint64_t(64),remaining,BinaryRepeatParams{1,1,0,9,0,1});
                    }
                    else MulAddDst(mat[(p+1)*stride],factor,spread,pitch,remaining,BinaryRepeatParams{1,1,0,static_cast<uint8_t>(stride/8),0,1});
                } else {
                if constexpr(!FULL_WIDTH)
                    Gather(pivot,mat,baseIndex.ReinterpretCast<uint32_t>(),row*4,cols);
                PipeBarrier<PIPE_V>();
                if constexpr(FULL_WIDTH) {
                    if(remaining<=64)Muls<float,false>(pivot,pivot,-inv,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
                    else Muls(pivot,pivot,-inv,remaining);
                } else Muls(pivot,pivot,-inv,cols);
                PipeBarrier<PIPE_V>();
                Brcb(spread,pivot,FULL_WIDTH?(remaining+7)/8:cols/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                if constexpr(FULL_WIDTH)MulAddDst<float,float,false>(mat[(p+1)*stride],factor,spread,uint64_t(64),remaining,BinaryRepeatParams{1,1,0,9,0,1});
                else MulAddDst(mat[(p+1)*stride],factor,spread[(p+1)*8],pitch,remaining,BinaryRepeatParams{1,1,0,static_cast<uint8_t>(stride/8),0,1});
                }
                if constexpr(!(FULL_WIDTH && BLOCK_TRANSPOSE))PipeBarrier<PIPE_V>();
            }
            Fence<HardEvent::V_S>();
            auto offsets=transposeIndex.ReinterpretCast<int32_t>();
            auto pattern=baseIndex;
            for(uint32_t q=0;q<8;++q) reinterpret_cast<__ubuf__ int32_t*>(pattern.GetPhyAddr())[q]=(n+(q<k?q:0))*stride*4;
            Fence<HardEvent::S_V>();
            Brcb(offsets.ReinterpretCast<float>(),order.ReinterpretCast<float>(),(n+7)/8,BrcbRepeatParams{1,8});
            PipeBarrier<PIPE_V>();Muls(offsets,offsets,int32_t(4),pitch*8);PipeBarrier<PIPE_V>();
            Add(offsets,offsets,pattern,64,(n+7)/8,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();
            Gather(output,mat,offsets.ReinterpretCast<uint32_t>(),0,n*8);PipeBarrier<PIPE_V>();
            Brcb(input,inverse,(n+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            Mul(output,output,input,n*8);PipeBarrier<PIPE_V>();
            // Keep the first k floats of each padded output row, then one contiguous DMA.
            Duplicate(transposeIndex,uint32_t(((1u<<k)-1)*0x01010101u),8);PipeBarrier<PIPE_V>();
            SetMaskNorm();
            GatherMask(input,output,transposeIndex.ReinterpretCast<float>(),uint8_t(0),
                       GatherMaskParams{1,static_cast<uint16_t>((n+7)/8),8,0});
            Fence<HardEvent::V_MTE3>();
            if(GetBlockIdx()<owners){DataCopyPad(x[uint64_t(batch)*n*k],input,DataCopyExtParams{1,n*k*4,0,0,0});}
            if(batch+owners<batches)Fence<HardEvent::MTE3_S>();
        }
    }
private:
    LocalTensor<float> mat,input,transposed,factor,pivot,spread,inverse,output,absolute,reduced;
    LocalTensor<int32_t> baseIndex;LocalTensor<uint32_t> transposeIndex,order;
    GlobalTensor<float> a,b,x;
    uint32_t n,k,pitch,stride,cols,batches,cores;bool broadcast;
};
using KernelSmallOuter=KernelSmallOuterT<false>;

template<uint32_t SN,uint32_t SK,bool PARTIAL_CAPACITY=false> class KernelSmallFixed {
public:
    __aicore__ inline ~KernelSmallFixed() {PipeBarrier<PIPE_ALL>();}
    template<HardEvent event> __aicore__ inline void Fence() {
        const event_t id=static_cast<event_t>(0);
        SetFlag<event>(id);WaitFlag<event>(id);
    }
    __aicore__ inline void Init(GM_ADDR A,GM_ADDR B,GM_ADDR X,const LinalgSolveTilingData& t) {
        batches=t.batches;cores=t.cores;broadcast=t.broadcastB;logicalN=PARTIAL_CAPACITY?t.n:n;
        a.SetGlobalBuffer((__gm__ float*)A);b.SetGlobalBuffer((__gm__ float*)B);x.SetGlobalBuffer((__gm__ float*)X);
        SetMaskNorm();SetAtomicNone();
        constexpr uint32_t offset0=0;
        mat=LocalTensor<float>(TPosition::VECCALC,offset0,cols*pitch);
        constexpr uint32_t offset1=offset0+(cols*pitch)*4;
        input=LocalTensor<float>(TPosition::VECCALC,offset1,pitch*pitch);
        constexpr uint32_t offset2=offset1+(pitch*pitch)*4;
        transposed=LocalTensor<float>(TPosition::VECCALC,offset2,pitch*pitch);
        constexpr uint32_t offset3=offset2+(pitch*pitch)*4;
        transposeIndex=LocalTensor<uint32_t>(TPosition::VECCALC,offset3,pitch*pitch);
        constexpr uint32_t offset4=offset3+(pitch*pitch)*4;
        pivot=LocalTensor<float>(TPosition::VECCALC,offset4,cols);
        constexpr uint32_t offset5=offset4+(cols)*4;
        spread=LocalTensor<float>(TPosition::VECCALC,offset5,cols*8);
        constexpr uint32_t offset6=offset5+(cols*8)*4;
        baseIndex=LocalTensor<int32_t>(TPosition::VECCALC,offset6,cols);
        constexpr uint32_t offset7=offset6+(cols)*4;
        absolute=LocalTensor<float>(TPosition::VECCALC,offset7,64);
        constexpr uint32_t offset8=offset7+(64)*4;
        reduced=LocalTensor<float>(TPosition::VECCALC,offset8,8);
        constexpr uint32_t offset9=offset8+(8)*4;
        order=LocalTensor<uint32_t>(TPosition::VECCALC,offset9,64);
        constexpr uint32_t offset10=offset9+(64)*4;
        inverse=LocalTensor<float>(TPosition::VECCALC,offset10,64);
        constexpr uint32_t offset11=offset10+(64)*4;
        output=LocalTensor<float>(TPosition::VECCALC,offset11,n*8);
    }
    __aicore__ inline void MakeIndex(LocalTensor<int32_t> target,uint32_t count) {
        uint32_t seed=count<8?count:8;
        for(uint32_t i=0;i<seed;++i)target.SetValue(i,int32_t(i));
        Fence<HardEvent::S_V>();
        for(uint32_t filled=8;filled<count;filled*=2) {
            uint32_t len=count-filled<filled?count-filled:filled;
            Adds(target[filled],target,int32_t(filled),len);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void LoadMatrix(GlobalTensor<float> source,uint64_t offset,uint32_t count,uint32_t first) {
        uint32_t shift=3;while((uint32_t(1)<<shift)<pitch)++shift;
        uint32_t size=count*pitch;
        auto sequence=input.ReinterpretCast<int32_t>();
        auto columnIndex=transposed.ReinterpretCast<int32_t>();
        auto offsets=transposeIndex.ReinterpretCast<int32_t>();
        MakeIndex(sequence,size);PipeBarrier<PIPE_V>();
        ShiftRight(columnIndex,sequence,int32_t(shift),size);PipeBarrier<PIPE_V>();
        Muls(offsets,columnIndex,static_cast<int32_t>(pitch),size);PipeBarrier<PIPE_V>();
        Sub(offsets,sequence,offsets,size);PipeBarrier<PIPE_V>();
        Muls(offsets,offsets,static_cast<int32_t>(count*4),size);
        Muls(columnIndex,columnIndex,int32_t(4),size);PipeBarrier<PIPE_V>();
        Add(offsets,offsets,columnIndex,size);PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE2>();
        DataCopyPad(input,source[offset],DataCopyExtParams{1,n*count*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
        Fence<HardEvent::MTE2_V>();Gather(mat[first*pitch],input,transposeIndex,0,size);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void LoadPartial(uint32_t batch) {
        // n15 + one RHS fill16 private columns; rows use a16-lane capacity.
        // Only external dimensions vary. Extend A by an independent4I block.
        Duplicate(input,0.0f,pitch*pitch);Fence<HardEvent::V_MTE2>();
        DataCopyPad(input,a[uint64_t(batch)*logicalN*logicalN],
            DataCopyExtParams{uint16_t(logicalN),logicalN*4,0,0,0},
            DataCopyPadExtParams<float>{true,0,uint8_t(pitch-logicalN),0});
        DataCopyPad(mat[n*pitch],b[broadcast?0:uint64_t(batch)*logicalN],
            DataCopyExtParams{1,logicalN*4,0,0,0},
            DataCopyPadExtParams<float>{true,0,uint8_t(pitch-logicalN),0});
        Fence<HardEvent::MTE2_S>();
        for(uint32_t i=logicalN;i<n;++i)input.SetValue(i*pitch+i,4.0f);
        auto offsets=transposeIndex.ReinterpretCast<int32_t>();
        for(uint32_t i=0;i<8;++i) {
            offsets.SetValue(i,int32_t(i*pitch*4));
            baseIndex.SetValue(i,int32_t(i*pitch*4));
        }
        Fence<HardEvent::S_V>();
        Adds(offsets[8],offsets,int32_t(8*pitch*4),8);
        Adds(baseIndex[8],baseIndex,int32_t(8*pitch*4),8);PipeBarrier<PIPE_V>();
        for(uint32_t filled=pitch;filled<n*pitch;filled*=2) {
            uint32_t len=n*pitch-filled<filled?n*pitch-filled:filled;
            Adds(offsets[filled],offsets,int32_t(filled/pitch*4),len);PipeBarrier<PIPE_V>();
        }
        Fence<HardEvent::MTE2_V>();
        Gather(mat,input,transposeIndex,0,n*pitch);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Process() {
        uint32_t owners=cores<batches?cores:batches;
        for(uint32_t batch=GetBlockIdx()%owners;batch<batches;batch+=owners) {
            if constexpr(PARTIAL_CAPACITY)LoadPartial(batch);
            else if constexpr(n>8 && n<=16 && k==1) {
                Fence<HardEvent::V_MTE2>();
                DataCopyPad(input,a[uint64_t(batch)*n*n],DataCopyExtParams{1,n*n*4,0,0,0},
                    DataCopyPadExtParams<float>{true,0,(8-n*n%8)%8,0});
                DataCopyPad(mat[n*pitch],b[broadcast?0:uint64_t(batch)*n],DataCopyExtParams{1,n*4,0,0,0},
                    DataCopyPadExtParams<float>{true,0,pitch-n,0});
                auto offsets=transposeIndex.ReinterpretCast<int32_t>();
                #pragma unroll
                for(uint32_t i=0;i<8;++i){reinterpret_cast<__ubuf__ int32_t*>(offsets.GetPhyAddr())[i]=int32_t(i*n*4);reinterpret_cast<__ubuf__ int32_t*>(baseIndex.GetPhyAddr())[i]=int32_t(i*pitch*4);}
                Fence<HardEvent::S_V>();
                Adds(offsets[8],offsets,int32_t(8*n*4),8);
                Adds(baseIndex[8],baseIndex,int32_t(8*pitch*4),8);PipeBarrier<PIPE_V>();
                if constexpr(cols>16){Adds(baseIndex[16],baseIndex,int32_t(16*pitch*4),cols-16);}
                for(uint32_t filled=pitch;filled<n*pitch;filled*=2) {
                    uint32_t len=n*pitch-filled<filled?n*pitch-filled:filled;
                    Adds(offsets[filled],offsets,int32_t(filled/pitch*4),len);PipeBarrier<PIPE_V>();
                }
                Fence<HardEvent::MTE2_V>();
                Gather(mat,input,transposeIndex,0,n*pitch);PipeBarrier<PIPE_V>();
            } else {
                MakeIndex(baseIndex,cols);PipeBarrier<PIPE_V>();
                Muls(baseIndex,baseIndex,static_cast<int32_t>(pitch*4),cols);PipeBarrier<PIPE_V>();
                Duplicate(mat,0.0f,cols*pitch);PipeBarrier<PIPE_V>();
                LoadMatrix(a,uint64_t(batch)*n*n,n,0);
                LoadMatrix(b,broadcast?0:uint64_t(batch)*n*k,k,n);
            }
            uint64_t activeRows[2]={n==64?~uint64_t(0):(uint64_t(1)<<n)-1,0};
            uint32_t pairFirst=0;
            float pivotThreshold=0.0f;
            #pragma unroll
            for(uint32_t p=0;p<n;++p) {
                factor=mat[p*pitch];
                uint32_t row;float diagonal;
                if constexpr(n==16) {
                    if((p&1)==0) {
                        Fence<HardEvent::V_S>();
                        uint64_t packet=reinterpret_cast<__ubuf__ uint64_t*>(factor.GetPhyAddr())[p/2];
                        uint32_t left=static_cast<uint32_t>(packet),right=static_cast<uint32_t>(packet>>32);
                        bool chooseRight=(left&0x7fffffffu)<(right&0x7fffffffu);
                        row=p+uint32_t(chooseRight);pairFirst=row;
                        union {uint32_t bits;float value;} word;
                        word.bits=chooseRight?right:left;diagonal=word.value;
                    } else {
                        Fence<HardEvent::V_S>();row=pairFirst^1u;
                        diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[row];
                    }
                } else if constexpr(n==15) {
                    if(p+2>=n) {
                        uint64_t first=activeRows[0]&(~activeRows[0]+1);
                        uint32_t left=63-static_cast<uint32_t>(CountLeadingZero(first));
                        Fence<HardEvent::V_S>();
                        row=left;diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[left];
                        if(p+2==n) {
                            uint32_t right=63-static_cast<uint32_t>(CountLeadingZero(activeRows[0]^first));
                            float other=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[right];
                            union {uint32_t bits;float value;} aWord,bWord;
                            aWord.value=diagonal;bWord.value=other;
                            if((aWord.bits&0x7fffffffu)<(bWord.bits&0x7fffffffu)) {
                                row=right;diagonal=other;
                            }
                        }
                    } else {
                        bool search=p==0;
                        if(p) {
                            uint64_t first=activeRows[0]&(~activeRows[0]+1);
                            row=63-static_cast<uint32_t>(CountLeadingZero(first));
                            Fence<HardEvent::V_S>();diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[row];
                            float magnitude=diagonal<0.0f?-diagonal:diagonal;
                            search=magnitude<pivotThreshold;
                        }
                        if(search) {
                            auto candidates=absolute;
                            if(p==0) {
                                Muls(candidates,factor,-1.0f,pitch);
                                Adds(candidates[pitch],factor,0.0f,pitch);
                            } else {
                                // Retired tiny GJ columns are dead; the current column follows this copy.
                                candidates=mat[(p-1)*pitch];
                                Muls(candidates,factor,-1.0f,pitch);
                            }
                            PipeBarrier<PIPE_V>();
                            uint64_t signedRows[2]={activeRows[0]|(activeRows[0]<<pitch),0};
                            WholeReduceMax(reduced,candidates,signedRows,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
                            Fence<HardEvent::V_S>();
                            uint64_t packet=reinterpret_cast<__ubuf__ uint64_t*>(reduced.GetPhyAddr())[0];
                            uint32_t index=static_cast<uint32_t>(packet>>32);
                            union {uint32_t bits;float value;} word;
                            word.bits=static_cast<uint32_t>(packet)|(uint32_t(index<pitch)<<31);
                            diagonal=word.value;row=index&(pitch-1);
                        }
                        if(p==0)pivotThreshold=(diagonal<0.0f?-diagonal:diagonal)*0.5f;
                    }
                    activeRows[0]&=~(uint64_t(1)<<row);
                } else {
                    Abs(absolute,factor,pitch);PipeBarrier<PIPE_V>();
                    WholeReduceMax(reduced,absolute,activeRows,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
                    Fence<HardEvent::V_S>();
                    row=reinterpret_cast<__ubuf__ uint32_t*>(reduced.GetPhyAddr())[1];
                    activeRows[0]&=~(uint64_t(1)<<row);
                    diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[row];
                }
                reinterpret_cast<__ubuf__ uint32_t*>(order.GetPhyAddr())[p]=row;
                float inv=1.0f/diagonal;reinterpret_cast<__ubuf__ float*>(inverse.GetPhyAddr())[p]=inv;reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[row]=0.0f;
                Fence<HardEvent::S_V>();
                Gather(pivot,mat,baseIndex.ReinterpretCast<uint32_t>(),row*4,cols);PipeBarrier<PIPE_V>();
                Muls(pivot,pivot,-inv,cols);PipeBarrier<PIPE_V>();
                Brcb(spread,pivot,cols/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                uint32_t remaining=n+k-p-1;
                MulAddDst(mat[(p+1)*pitch],factor,spread[(p+1)*8],pitch,remaining,BinaryRepeatParams{1,1,0,static_cast<uint8_t>(pitch/8),0,1});
                PipeBarrier<PIPE_V>();
            }
            auto offsets=transposeIndex.ReinterpretCast<int32_t>();
            Muls(offsets,order.ReinterpretCast<int32_t>(),int32_t(4),n);PipeBarrier<PIPE_V>();
            Gather(output,mat[n*pitch],transposeIndex,0,n);PipeBarrier<PIPE_V>();
            Mul(output,output,inverse,n);Fence<HardEvent::V_MTE3>();
            if(GetBlockIdx()<owners){DataCopyPad(x[uint64_t(batch)*logicalN],output,DataCopyExtParams{1,logicalN*4,0,0,0});if(batch+owners<batches)Fence<HardEvent::MTE3_S>();}
        }
    }
private:
    LocalTensor<float> mat,input,transposed,factor,pivot,spread,inverse,output,absolute,reduced;
    LocalTensor<int32_t> baseIndex;LocalTensor<uint32_t> transposeIndex,order;
    GlobalTensor<float> a,b,x;
    static constexpr uint32_t n=SN,k=SK,pitch=SN<=8?8:SN<=16?16:SN<=32?32:64,cols=(SN+SK+7)/8*8;
    uint32_t batches,cores,logicalN;bool broadcast;
};
template<typename DT_A, uint32_t TASK_MODE, uint32_t N_VALUE>
__global__ __aicore__ void linalg_solve(GM_ADDR A,GM_ADDR B,GM_ADDR X,GM_ADDR workspace,GM_ADDR tiling) {
    if constexpr(TASK_MODE>=64 && TASK_MODE<80) {
        // Full private row/column tiles bind batch metadata for one8-core wave.
        LinalgSolveTilingData data{};
        data.n=N_VALUE;data.k=1;data.batches=(TASK_MODE-64)/2+1;
        data.cores=8;data.broadcastB=(TASK_MODE-64)&1u;
        KernelSmallFixed<N_VALUE,1> op;
        op.Init(A,B,X,data);op.Process();
    } else {
    REGISTER_TILING_DEFAULT(LinalgSolveTilingData);
    GET_TILING_DATA_WITH_STRUCT(LinalgSolveTilingData,tiling_data,tiling);
    if constexpr(sizeof(DT_A)==4 && TASK_MODE==3) {
        if constexpr(N_VALUE==15) {
            // One16-column capacity covers n9..15 and one RHS. Full-capacity
            // inputs retain static extents; smaller inputs use bounded GM IO.
            if(tiling_data.n==N_VALUE) {
                KernelSmallFixed<15,1> op;
                op.Init(A,B,X,tiling_data);op.Process();
            } else {
                KernelSmallFixed<15,1,true> op;
                op.Init(A,B,X,tiling_data);op.Process();
            }
        } else if constexpr(N_VALUE>0 && N_VALUE<=16) {
            KernelSmallFixed<N_VALUE,1> op;
            op.Init(A,B,X,tiling_data);op.Process();
        } else {
            KernelSmallOuter op;op.Init(A,B,X,tiling_data);op.Process();
        }
    } else if constexpr(sizeof(DT_A)==4 && TASK_MODE==2) {
        if(tiling_data.n>32) {
            KernelSmallOuterT<N_VALUE==32 || N_VALUE==64,true> op;
            op.Init(A,B,X,tiling_data);op.Process();
        } else {
        KernelSmallOuterT<N_VALUE==32 || N_VALUE==64> op;
        op.Init(A,B,X,tiling_data);op.Process();
        }
    } else if constexpr(sizeof(DT_A)==4 && (TASK_MODE==7 || TASK_MODE==8)) {
        // One registered capacity template handles both full and partial tiles.
        // Static full-tile extents remove tail bookkeeping from the hot path.
        if(tiling_data.n==256 && tiling_data.k==3) {
            KernelColumnSolve<3,256,TASK_MODE==8?10:8,3,false> op;
            op.Init(A,B,X,GetUserWorkspace(workspace),tiling_data);op.Process();
        } else {
            KernelColumnSolve<3,256,TASK_MODE==8?10:8,3,true> op;
            op.Init(A,B,X,GetUserWorkspace(workspace),tiling_data);op.Process();
        }
    } else if constexpr(sizeof(DT_A)==4 && (TASK_MODE==1 || TASK_MODE==4 || TASK_MODE==5 || TASK_MODE==6)) {
        KernelColumnSolve<(TASK_MODE==5 || TASK_MODE==6)?3:TASK_MODE==4?2:1,
                          (TASK_MODE==5 || TASK_MODE==6)?N_VALUE:0,TASK_MODE==6?8:0> op;
        op.Init(A,B,X,GetUserWorkspace(workspace),tiling_data);
        op.Process();
    } else {
        if constexpr(sizeof(DT_A)==4) {
            if(tiling_data.n<=256 && tiling_data.k<=8) {
                KernelColumnSolve<> op;
                op.Init(A,B,X,GetUserWorkspace(workspace),tiling_data);
                op.Process();return;
            }
        }
        KernelLinalgSolve<DT_A, N_VALUE> op;
        op.Init(A,B,X,GetUserWorkspace(workspace),tiling_data);
        op.Process();
    }
    }
}

// Small/medium AIV entries use only explicit A/B/X pointers and private UB.
// Retain generated system initialization for every other mode and debug build.
// CANN9 appends its wrapper after this source; TEMPLATE_PARAMS is defined there.
namespace AscendC {
template<class DT_A, uint32_t TASK_MODE, uint32_t N_VALUE>
__aicore__ inline void SolveSetSysWorkspaceForMode(GM_ADDR workspace) {
#if defined(ASCENDC_DUMP) || defined(ASCENDC_TIME_STAMP_ON) || defined(ASCENDC_OOM) || defined(ASCENDC_CPU_DEBUG)
    SetSysWorkspaceForce(workspace);
#else
    if constexpr(!(sizeof(DT_A)==4 && (TASK_MODE==2 || TASK_MODE==3 || (TASK_MODE>=64 && TASK_MODE<80)))) SetSysWorkspaceForce(workspace);
#endif
}
}
template<class DT_A, uint32_t TASK_MODE, uint32_t N_VALUE>
__aicore__ inline void SolveSetFftsForMode(uint64_t address) {
#if defined(ASCENDC_DUMP) || defined(ASCENDC_TIME_STAMP_ON) || defined(ASCENDC_OOM) || defined(ASCENDC_CPU_DEBUG)
    set_ffts_base_addr(address);
#else
    if constexpr(!(sizeof(DT_A)==4 && (TASK_MODE==2 || TASK_MODE==3 || (TASK_MODE>=64 && TASK_MODE<80)))) set_ffts_base_addr(address);
#endif
}
template<class DT_A, uint32_t TASK_MODE, uint32_t N_VALUE>
__aicore__ inline void SolveGeneratedPreloadForMode(uint64_t mode) {
    (void)mode;
#if defined(ASCENDC_DUMP) || defined(ASCENDC_TIME_STAMP_ON) || defined(ASCENDC_OOM) || defined(ASCENDC_CPU_DEBUG)
    icache_preload(1);
#else
    if constexpr(!(sizeof(DT_A)==4 && (TASK_MODE==2 || TASK_MODE==3 || (TASK_MODE>=64 && TASK_MODE<80)))) icache_preload(1);
#endif
}
#define SetSysWorkspaceForce(workspace) SolveSetSysWorkspaceForMode<TEMPLATE_PARAMS>(workspace)
#define set_ffts_base_addr(address) SolveSetFftsForMode<TEMPLATE_PARAMS>(address)
#define icache_preload(mode) SolveGeneratedPreloadForMode<TEMPLATE_PARAMS>(mode)
