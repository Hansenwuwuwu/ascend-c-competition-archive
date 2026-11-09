#pragma once
#include "spline_conv_sparse_output.h"

template <typename T,uint32_t C>
class SplineCompactSparse {
public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR edges,GM_ADDR pseudo,GM_ADDR weight,GM_ADDR out,const SplineConvTilingData &td,TPipe *pipe) {
        t=td; xg.SetGlobalBuffer((__gm__ T*)x); eg.SetGlobalBuffer((__gm__ int64_t*)edges);
        pg.SetGlobalBuffer((__gm__ T*)pseudo); wg.SetGlobalBuffer((__gm__ T*)weight); yg.SetGlobalBuffer((__gm__ T*)out);
        if constexpr(C==128 || C==32) {
            constexpr uint32_t WO=2560,WF=sizeof(T)==2?WO+4*C*16*sizeof(T):WO;
            constexpr uint32_t XO=WF+4*C*16*4,XF=sizeof(T)==2?XO+C*sizeof(T):XO;
            constexpr uint32_t AO=XF+C*4,BO=AO+4*C*4,DO=BO+4*C*8*4,YO=DO+4*C*16*4;
            el=LocalTensor<int64_t>(TPosition::VECCALC,0,128);
            pr=LocalTensor<T>(TPosition::VECCALC,1024,128);pb=pr.template ReinterpretCast<uint16_t>();
            head=LocalTensor<int32_t>(TPosition::VECCALC,1536,128);
            links=LocalTensor<int32_t>(TPosition::VECCALC,2048,64);
            occupied=LocalTensor<uint32_t>(TPosition::VECCALC,2304,64);
            wr=LocalTensor<T>(TPosition::VECCALC,WO,4*C*16);
            weights=LocalTensor<float>(TPosition::VECCALC,WF,4*C*16);
            xr=LocalTensor<T>(TPosition::VECCALC,XO,C);xv=LocalTensor<float>(TPosition::VECCALC,XF,C);
            agg=LocalTensor<float>(TPosition::VECCALC,AO,4*C);
            broadcast=LocalTensor<float>(TPosition::VECCALC,BO,4*C*8);
            dot=LocalTensor<float>(TPosition::VECCALC,DO,4*C*16);
            outLocal=LocalTensor<T>(TPosition::VECCALC,YO,128*16);
            if constexpr(C==128 || C==32) prefetched=LocalTensor<T>(TPosition::VECCALC,YO+128*16*sizeof(T),64*C);
        } else {
        uint32_t off=0;
        uint32_t eo=Reserve(off,t.e*16),po=Reserve(off,t.e*2*sizeof(T)),ho=Reserve(off,t.nodesPerBlock*4),no=Reserve(off,t.e*4);
        uint32_t ro=C==128?Reserve(off,t.e*4):0;
        uint32_t wo=Reserve(off,4*C*16*sizeof(T));
        uint32_t wf=sizeof(T)==2?Reserve(off,4*C*16*4):wo;
        uint32_t xo=Reserve(off,C*sizeof(T)),xf=sizeof(T)==2?Reserve(off,C*4):xo;
        uint32_t ao=Reserve(off,4*C*4),bo=Reserve(off,4*C*8*4),doff=Reserve(off,4*C*16*4),yo=Reserve(off,t.nodesPerBlock*16*sizeof(T));
        pipe->InitBuffer(buf,off);
        el=buf.Get<int64_t>()[eo/8]; pr=buf.Get<T>()[po/sizeof(T)]; pb=pr.template ReinterpretCast<uint16_t>();
        head=buf.Get<int32_t>()[ho/4]; links=buf.Get<int32_t>()[no/4];
        if constexpr(C==128) occupied=buf.Get<uint32_t>()[ro/4];
        wr=buf.Get<T>()[wo/sizeof(T)]; weights=buf.Get<float>()[wf/4];
        xr=buf.Get<T>()[xo/sizeof(T)]; xv=buf.Get<float>()[xf/4];
        agg=buf.Get<float>()[ao/4]; broadcast=buf.Get<float>()[bo/4]; dot=buf.Get<float>()[doff/4]; outLocal=buf.Get<T>()[yo/sizeof(T)];
        }
    }
    __aicore__ inline uint32_t Reserve(uint32_t &off,uint32_t bytes) {
        uint32_t result=off; off+=(bytes+31)/32*32; return result;
    }
    __aicore__ inline float Pseudo(uint32_t i) {
        if constexpr(sizeof(T)==2) {union {uint32_t u;float f;} b;b.u=uint32_t(pb.GetValue(i))<<16;return b.f;}
        else return pr.GetValue(i);
    }
    __aicore__ inline bool SingleDot(uint32_t e) {
        int64_t src=el.GetValue(e);if(src<0 || src>=t.n) return false;
        DataCopyExtParams xp{1,C*uint32_t(sizeof(T)),0,0,0};DataCopyPadExtParams<T> pad{false,0,0,{}};
        if constexpr(C==128 || C==32) {
            xr=prefetched[e*C];
            if constexpr(sizeof(T)==4) xv=xr.template ReinterpretCast<float>();
        } else DataCopyPad(xr,xg[uint32_t(src)*C],xp,pad);
        if constexpr(C==128) {SetFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);}
        else {SetFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);}
        if constexpr(sizeof(T)==2) {Cast(xv,xr,RoundMode::CAST_NONE,C);}
        float p0=Pseudo(e*2)*int32_t(t.scale[0]),p1=Pseudo(e*2+1)*int32_t(t.scale[1]);
        int32_t b0=int32_t(p0),b1=int32_t(p1);p0-=float(b0);p1-=float(b1);
        float f[4]={(1-p0)*(1-p1),p0*(1-p1),(1-p0)*p1,p0*p1};
        uint32_t wi=(b0&1)+2*(b1&1);
        if constexpr(C==128) {
            Muls(dot,weights[wi*C*16],f[0],C*16);
            for(uint32_t slot=1;slot<4;++slot) {
                wi=((b0+(slot&1))&1)+2*((b1+(slot>>1))&1);
                Axpy(dot,weights[wi*C*16],f[slot],C*16);
            }
            Brcb(broadcast,xv,uint8_t(C/8),BrcbRepeatParams{1,8});
            Mul(dot,dot,broadcast,uint64_t(16),uint8_t(C),BinaryRepeatParams{1,1,0,2,2,1});
            for(uint32_t rows=C;rows>1;rows/=2) {Add(dot,dot,dot[rows/2*16],rows/2*16);}
        } else {
            Muls(dot,weights[wi*C*16],f[0],C*16);
            for(uint32_t slot=1;slot<4;++slot) {
                wi=((b0+(slot&1))&1)+2*((b1+(slot>>1))&1);
                Axpy(dot,weights[wi*C*16],f[slot],C*16);
            }
            Brcb(broadcast,xv,uint8_t(C/8),BrcbRepeatParams{1,8});
            Mul(dot,dot,broadcast,uint64_t(16),uint8_t(C),BinaryRepeatParams{1,1,0,2,2,1});
            for(uint32_t rows=C;rows>1;rows/=2) {Add(dot,dot,dot[rows/2*16],rows/2*16);}
        }
        return true;
    }
    __aicore__ inline void Dot() {
        Brcb(broadcast,agg,uint8_t(4*C/8),BrcbRepeatParams{1,8});
        BinaryRepeatParams params{1,1,0,2,2,1};
        for(uint32_t row=0;row<4*C;row+=255) {
            uint32_t count=4*C-row;if(count>255) count=255;
            Mul(dot[row*16],weights[row*16],broadcast[row*8],uint64_t(16),uint8_t(count),params);
        }
        for(uint32_t rows=4*C;rows>1;rows/=2) {Add(dot,dot,dot[rows/2*16],rows/2*16);}
    }
    template <uint32_t Step=0>
    __aicore__ inline void Process() {
        uint32_t start=GetBlockIdx(),step=Step?Step:t.blocks,rows=(t.n-start+step-1)/step;
        uint32_t zeroBound=(t.n+step-1)/step,zeroBegin=GetBlockIdx()*zeroBound,zeroRows=0,occupiedZero=0;
        splitZero=C==128 && zeroBound<=32;nodeMask=0;
        if(splitZero && zeroBegin<t.n) {zeroRows=t.n-zeroBegin;if(zeroRows>zeroBound) zeroRows=zeroBound;}
        DataCopyExtParams ep{1,t.e*16,0,0,0};DataCopyPadExtParams<int64_t> ez{false,0,0,0};DataCopyPad(el,eg,ep,ez);
        DataCopyExtParams pp{1,t.e*2*uint32_t(sizeof(T)),0,0,0};DataCopyPadExtParams<T> pz{false,0,0,{}};
        DataCopyPad(pr,pg,pp,pz);
        if constexpr(C==128 || C==32) {
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            if((uint64_t(wg.GetPhyAddr())&31)==0) DataCopy(wr,wg,4*C*16);
            else {DataCopyExtParams wp{1,4*C*16*uint32_t(sizeof(T)),0,0,0};DataCopyPad(wr,wg,wp,pz);}
            Duplicate(outLocal.template ReinterpretCast<int32_t>(),int32_t(0),(C==128?zeroBound:rows)*16*sizeof(T)/4);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        } else {
            Duplicate(outLocal.template ReinterpretCast<int32_t>(),int32_t(0),(C==128?zeroBound:rows)*16*sizeof(T)/4); PipeBarrier<PIPE_ALL>();
        }
        if constexpr(C==8) for(uint32_t i=0;i<rows;++i) head.SetValue(i,-1);
        uint32_t activeCount=0,ownedCount=0,singleEdge=0; bool found=false;
        for(uint32_t e=0;e<t.e;++e) {
            int64_t dst=el.GetValue(t.e+e);
            if constexpr(C==128) {
                if(splitZero && dst>=zeroBegin && dst<int64_t(zeroBegin+zeroRows)) occupiedZero|=1u<<(uint32_t(dst)-zeroBegin);
            }
            if(dst>=start && dst<t.n && (uint32_t(dst)-start)%step==0) {
                uint32_t j=(uint32_t(dst)-start)/step;
                if constexpr(C==128) {if(splitZero) nodeMask|=1u<<j;}
                if constexpr(C==128 || C==32) {
                    int64_t src=el.GetValue(e);
                    if(src>=0 && src<t.n) {
                        DataCopyExtParams xp{1,C*uint32_t(sizeof(T)),0,0,0};
                        DataCopyPad(prefetched[e*C],xg[uint32_t(src)*C],xp,pz);
                    }
                }
                if constexpr(C==128) {++ownedCount; singleEdge=e;}
                else if constexpr(C==32) {++ownedCount;singleEdge=e;}
                else {links.SetValue(e,head.GetValue(j));head.SetValue(j,e);found=true;}
            }
        }
        if constexpr(C==32) {
            if(ProcessSingleOwner(ownedCount,singleEdge,start,step,rows)) return;
        }
        if constexpr(C==32) {
            for(uint32_t i=0;i<rows;++i) head.SetValue(i,-1);
            for(uint32_t e=0;e<t.e;++e) {
                int64_t dst=el.GetValue(t.e+e);
                if(dst>=start && dst<t.n && (uint32_t(dst)-start)%step==0) {
                    uint32_t j=(uint32_t(dst)-start)/step;
                    links.SetValue(e,head.GetValue(j));head.SetValue(j,e);found=true;
                }
            }
        }
        if constexpr(C==128) {
            if(splitZero) spline_output::ClearInactiveRows(yg,outLocal,zeroBegin,zeroRows,occupiedZero);
            if(ownedCount<=1) {
                {
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                }
                if(ownedCount==1) {
                    if constexpr(sizeof(T)==2) {Cast(weights,wr,RoundMode::CAST_NONE,4*C*16);}
                    if(SingleDot(singleEdge)) {
                        uint32_t row=(uint32_t(el.GetValue(t.e+singleEdge))-start)/step;
                        if constexpr(sizeof(T)==4) {
                            if(splitZero) {
                                SetFlag<HardEvent::V_MTE3>(EVENT_ID5);WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
                                spline_output::CopyRows(yg,dot,start+row*step,1);directMask|=1u<<row;
                            } else DataCopy(outLocal[row*16],dot,16);
                        } else {
                            if(splitZero) {SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);}
                            Cast(outLocal[row*16],dot,RoundMode::CAST_RINT,16);
                        }
                    }
                }
                StoreOutput(start,step,rows);
                return;
            }
            for(uint32_t e=0;e<t.e;++e) {
                int64_t dst=el.GetValue(t.e+e);
                if(dst>=start && dst<t.n && (uint32_t(dst)-start)%step==0) {
                    uint32_t j=(uint32_t(dst)-start)/step,index=0;
                    while(index<activeCount && occupied.GetValue(index)!=j) ++index;
                    if(index==activeCount) {occupied.SetValue(index,j);head.SetValue(index,-1);++activeCount;}
                    links.SetValue(e,head.GetValue(index));head.SetValue(index,e);found=true;
                }
            }
        }
        if(found) {
            if constexpr(C==8) {DataCopyExtParams wp{1,4*C*16*uint32_t(sizeof(T)),0,0,0};DataCopyPad(wr,wg,wp,pz);}
            SetFlag<HardEvent::MTE2_V>(EVENT_ID4);WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
            if constexpr(sizeof(T)==2) {Cast(weights,wr,RoundMode::CAST_NONE,4*C*16);}
            for(uint32_t j=0;j<(C==128?activeCount:rows);++j) {
                if(head.GetValue(j)<0) continue;
                uint32_t outputRow=C==128?occupied.GetValue(j):j;
                if constexpr(C==128 || C==32) {
                    if(links.GetValue(head.GetValue(j))<0) {
                        if(SingleDot(head.GetValue(j))) {
                            if constexpr(sizeof(T)==2) Cast(outLocal[outputRow*16],dot,RoundMode::CAST_RINT,16);
                            else DataCopy(outLocal[outputRow*16],dot,16);
                            PipeBarrier<PIPE_ALL>();
                        }
                        continue;
                    }
                }
                Duplicate(agg,0.0f,4*C);PipeBarrier<PIPE_ALL>();uint32_t count=0;
                for(int32_t e=head.GetValue(j);e>=0;e=links.GetValue(e)) {
                    int64_t src=el.GetValue(e);if(src<0 || src>=t.n) continue;
                    if constexpr(C==128 || C==32) {
                        xr=prefetched[uint32_t(e)*C];
                        if constexpr(sizeof(T)==4) xv=xr.template ReinterpretCast<float>();
                    } else {
                        DataCopyExtParams xp{1,C*uint32_t(sizeof(T)),0,0,0};DataCopyPad(xr,xg[uint32_t(src)*C],xp,pz);
                    }
                    PipeBarrier<PIPE_ALL>();
                    if constexpr(sizeof(T)==2) {Cast(xv,xr,RoundMode::CAST_NONE,C);}
                    float p0=Pseudo(e*2)*int32_t(t.scale[0]),p1=Pseudo(e*2+1)*int32_t(t.scale[1]);
                    int32_t b0=int32_t(p0),b1=int32_t(p1);p0-=float(b0);p1-=float(b1);
                    float factors[4]={(1-p0)*(1-p1),p0*(1-p1),(1-p0)*p1,p0*p1};
                    SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(C);
                    for(uint32_t slot=0;slot<4;++slot) {
                        uint32_t wi=((b0+(slot&1))&1)+2*((b1+(slot>>1))&1);
                        Axpy<float,float,false>(agg[wi*C],xv,factors[slot],MASK_PLACEHOLDER,1,UnaryRepeatParams{1,1,8,8});
                    }
                    SetMaskNorm();ResetMask();++count;
                }
                if(count==0) continue;
                Dot();
                if(t.mean && count!=1) {Muls(dot,dot,1.0f/count,16);}
                if constexpr(sizeof(T)==2) Cast(outLocal[outputRow*16],dot,RoundMode::CAST_RINT,16);
                else DataCopy(outLocal[outputRow*16],dot,16);
                PipeBarrier<PIPE_ALL>();
            }
        }
        StoreOutput(start,step,rows);
    }
    __aicore__ inline void StoreOutput(uint32_t start,uint32_t step,uint32_t rows) {
        if constexpr(C==128) {
            if(splitZero) {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID3);WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
                uint32_t mask=nodeMask&~directMask;
                while(mask) {
                    uint32_t row=spline_output::FirstBit(mask);
                    spline_output::CopyRows(yg,outLocal[row*16],start+row*step,1);
                    mask&=mask-1;
                }
                return;
            }
        }
        DataCopyExtParams yp{uint16_t(rows),16*uint32_t(sizeof(T)),0,(step-1)*16*uint32_t(sizeof(T)),0};
        SetFlag<HardEvent::V_MTE3>(EVENT_ID3);WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
        if constexpr(C==128) {
            if((uint64_t(yg.GetPhyAddr())&31)==0) {
                DataCopyParams basic{uint16_t(rows),uint16_t(16*sizeof(T)/32),0,uint16_t((step-1)*16*sizeof(T)/32)};
                DataCopy(yg[start*16],outLocal,basic);
            } else DataCopyPad(yg[start*16],outLocal,yp);
        } else DataCopyPad(yg[start*16],outLocal,yp);
    }
private:
    __aicore__ inline bool ProcessSingleOwner(uint32_t ownedCount,uint32_t singleEdge,uint32_t start,uint32_t step,uint32_t rows);
    SplineConvTilingData t;TBuf<TPosition::VECCALC> buf;
    bool splitZero=false;uint32_t nodeMask=0,directMask=0;
    LocalTensor<int64_t> el;LocalTensor<int32_t> head,links;LocalTensor<uint32_t> occupied;
    LocalTensor<T> pr,wr,xr,outLocal,prefetched;LocalTensor<uint16_t> pb;
    LocalTensor<float> weights,xv,agg,broadcast,dot;
    GlobalTensor<T> xg,pg,wg,yg;GlobalTensor<int64_t> eg;
};

template <typename T,uint32_t C>
__aicore__ inline bool SplineCompactSparse<T,C>::ProcessSingleOwner(uint32_t ownedCount,uint32_t singleEdge,uint32_t start,uint32_t step,uint32_t rows) {
    if(ownedCount<=1) {
        if(ownedCount==1) {
            SetFlag<HardEvent::MTE2_V>(EVENT_ID4);WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
            if constexpr(sizeof(T)==2) {Cast(weights,wr,RoundMode::CAST_NONE,4*C*16);}
            if(SingleDot(singleEdge)) {
                uint32_t row=(uint32_t(el.GetValue(t.e+singleEdge))-start)/step;
                if constexpr(sizeof(T)==2) Cast(outLocal[row*16],dot,RoundMode::CAST_RINT,16);
                else DataCopy(outLocal[row*16],dot,16);
            }
        }
        StoreOutput(start,step,rows);
        return true;
    }
    return false;
}


template <typename T>
__aicore__ inline bool RunCompactSparse(GM_ADDR x,GM_ADDR edges,GM_ADDR pseudo,GM_ADDR weight,GM_ADDR out,const SplineConvTilingData &t) {
    if(t.streaming || t.hasRoot || t.hasBias || !t.cacheGraph || !t.cacheWeights || !t.fusedDot ||
       t.e==0 || t.e>=t.n || t.e>64 || uint64_t(t.nodesPerBlock)*t.e>512 || t.nodesPerBlock>t.ci ||
       t.ci!=t.ciPad || t.co!=16 || t.dim!=2 || t.degree!=1 || t.ks[0]!=2 || t.ks[1]!=2) return false;
    if(t.ci==8) {TPipe pipe;SplineCompactSparse<T,8> op;op.Init(x,edges,pseudo,weight,out,t,&pipe);op.Process();}
    else if(t.ci==32) {SplineCompactSparse<T,32> op;op.Init(x,edges,pseudo,weight,out,t,nullptr);op.Process();}
    else if(t.ci==128) {SplineCompactSparse<T,128> op;op.Init(x,edges,pseudo,weight,out,t,nullptr);if(t.blocks==8) op.template Process<8>();else op.Process();}
    else return false;
    return true;
}
