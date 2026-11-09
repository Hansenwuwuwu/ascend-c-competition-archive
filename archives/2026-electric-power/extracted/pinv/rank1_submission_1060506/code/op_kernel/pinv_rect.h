#pragma once
#include "kernel_operator.h"
#include "linalg_pinv_tiling.h"
using namespace AscendC;

// Direct one-sided Jacobi for B=A (tall) or B=A^T (wide).
// Each GM row holds a column of B followed by the corresponding column of V.
class RectPinv {
    LinalgPinvTilingData t;
    uint32_t head, basis, width, batchOffset;
    float power;
    TBuf<TPosition::VECCALC> mem;
    LocalTensor<float> xh,xl,yh,yl,ah,al,bh,bl,calc[8],red,gather,sigma,sigmaLow,certH,certL,certT,certAccumulator;
    LocalTensor<uint32_t> retained;
    GlobalTensor<float> input,result,high,low;
    __aicore__ inline void VS(){SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);}
    __aicore__ inline void SV(){SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);}
    __aicore__ inline float AbsScalar(float x){return x<0?-x:x;}
#include "pinv_wide.h"
    __aicore__ inline float Value(Wide x){return x.h+x.l;}
    __aicore__ inline float Root(float x){
        red.SetValue(0,x>0?x:0);SV();Sqrt(red,red,1);VS();return red.GetValue(0);
    }
    __aicore__ inline Wide RootPair(Wide x){
        float h=Root(x.h);
        if(!(h>0))return {0,0};
        return Plus({h,0},Divide(Minus(x,Times({h,0},{h,0})),{2*h,0}));
    }
    __aicore__ inline void Load(LocalTensor<float> dst,GlobalTensor<float> src,uint32_t count){
        SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
        DataCopy(dst,src,count);SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
        SetFlag<HardEvent::MTE2_S>(0);WaitFlag<HardEvent::MTE2_S>(0);
    }
    __aicore__ inline void Store(GlobalTensor<float> dst,LocalTensor<float> src,uint32_t count){
        SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
        SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);DataCopy(dst,src,count);
        SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);
    }
    __aicore__ inline void Row(uint32_t i,LocalTensor<float> h,LocalTensor<float> l){
        Load(h,high[i*width],width);Load(l,low[i*width],width);
    }
    __aicore__ inline void Original(uint32_t column,LocalTensor<float> h,LocalTensor<float> l){
        Duplicate(h,0.0f,width);Duplicate(l,0.0f,width);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
        if(t.m>=t.n){
            DataCopyPad(gather,input[batchOffset+column],
                DataCopyExtParams{static_cast<uint16_t>(t.rows),4,(t.n-1)*4,0,0},
                DataCopyPadExtParams<float>{true,0,7,0});
            SetFlag<HardEvent::MTE2_S>(0);WaitFlag<HardEvent::MTE2_S>(0);
            for(uint32_t j=0;j<t.rows;++j)h.SetValue(j,gather.GetValue(j*8));SV();
        }else{
            DataCopyPad(h,input[batchOffset+column*t.n],DataCopyExtParams{1,t.n*4,0,0,0},
                DataCopyPadExtParams<float>{true,0,static_cast<uint8_t>((8-t.n%8)%8),0});
            SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
        }
        Muls(h,h,1.0f/power,head);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void PairAdd(LocalTensor<float> h,LocalTensor<float> l,
        LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c,LocalTensor<float> d,uint32_t count){
        auto sum=calc[0],bv=calc[1],av=calc[2],e=calc[3],v=calc[4];
        Add(sum,a,c,count);PipeBarrier<PIPE_V>();Sub(bv,sum,a,count);PipeBarrier<PIPE_V>();
        Sub(av,sum,bv,count);Sub(e,c,bv,count);PipeBarrier<PIPE_V>();
        Sub(v,a,av,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Add(e,e,b,count);PipeBarrier<PIPE_V>();Add(e,e,d,count);PipeBarrier<PIPE_V>();
        Add(h,sum,e,count);PipeBarrier<PIPE_V>();Sub(v,h,sum,count);PipeBarrier<PIPE_V>();
        Sub(l,e,v,count);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Scale(LocalTensor<float> h,LocalTensor<float> l,
        LocalTensor<float> a,LocalTensor<float> b,Wide factor,uint32_t count){
        auto hi=calc[0],lo=calc[1],p=calc[2],e=calc[3],v=calc[4],z=calc[5];
        float split=factor.h*pinv::kSplitFactor,fh=split-(split-factor.h),fl=factor.h-fh;
        Muls(hi,a,pinv::kSplitFactor,count);PipeBarrier<PIPE_V>();Sub(lo,hi,a,count);PipeBarrier<PIPE_V>();
        Sub(hi,hi,lo,count);PipeBarrier<PIPE_V>();Sub(lo,a,hi,count);
        Muls(p,a,factor.h,count);Muls(e,hi,fh,count);PipeBarrier<PIPE_V>();Sub(e,e,p,count);PipeBarrier<PIPE_V>();
        Muls(v,hi,fl,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Muls(v,lo,fh,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Muls(v,lo,fl,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Muls(v,a,factor.l,count);Muls(z,b,factor.h,count);PipeBarrier<PIPE_V>();
        Add(e,e,v,count);PipeBarrier<PIPE_V>();Add(e,e,z,count);PipeBarrier<PIPE_V>();
        Muls(v,b,factor.l,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Add(h,p,e,count);PipeBarrier<PIPE_V>();Sub(v,h,p,count);PipeBarrier<PIPE_V>();Sub(l,e,v,count);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline Wide Dot(LocalTensor<float> a,LocalTensor<float> b,
        LocalTensor<float> c,LocalTensor<float> d,uint32_t count){
        auto p=calc[0],e=calc[1],hi=calc[2],lo=calc[3],ch=calc[4],cl=calc[5],v=calc[6],z=calc[7];
        Muls(hi,a,pinv::kSplitFactor,count);Muls(ch,c,pinv::kSplitFactor,count);PipeBarrier<PIPE_V>();
        Sub(lo,hi,a,count);Sub(cl,ch,c,count);PipeBarrier<PIPE_V>();
        Sub(hi,hi,lo,count);Sub(ch,ch,cl,count);PipeBarrier<PIPE_V>();
        Sub(lo,a,hi,count);Sub(cl,c,ch,count);Mul(p,a,c,count);PipeBarrier<PIPE_V>();
        Mul(e,hi,ch,count);PipeBarrier<PIPE_V>();Sub(e,e,p,count);PipeBarrier<PIPE_V>();
        Mul(v,hi,cl,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Mul(v,lo,ch,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Mul(v,lo,cl,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Mul(v,a,d,count);Mul(z,b,c,count);PipeBarrier<PIPE_V>();
        Add(e,e,v,count);PipeBarrier<PIPE_V>();Add(e,e,z,count);PipeBarrier<PIPE_V>();
        Mul(v,b,d,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        // Pairwise TwoSum keeps the FP32 reduction error in the low component.
        for(uint32_t size=count;size>8;size/=2){
            uint32_t half=size/2;
            Add(hi,p,p[half],half);PipeBarrier<PIPE_V>();Sub(lo,hi,p,half);PipeBarrier<PIPE_V>();
            Sub(ch,hi,lo,half);Sub(cl,p[half],lo,half);PipeBarrier<PIPE_V>();
            Sub(v,p,ch,half);PipeBarrier<PIPE_V>();Add(cl,cl,v,half);PipeBarrier<PIPE_V>();
            Add(cl,cl,e,half);PipeBarrier<PIPE_V>();Add(cl,cl,e[half],half);PipeBarrier<PIPE_V>();
            Add(p,hi,cl,half);PipeBarrier<PIPE_V>();Sub(v,p,hi,half);PipeBarrier<PIPE_V>();
            Sub(e,cl,v,half);PipeBarrier<PIPE_V>();
        }
        VS();Wide sum={0,0};for(uint32_t i=0;i<8;++i)sum=Plus(sum,{p.GetValue(i),e.GetValue(i)});
        return sum;
    }
// Three-component arithmetic is confined to posterior residual measurements.
// Error-free FP32 products and TwoSum leave a third-order accumulation remainder.
    __aicore__ inline void CertTwoSum(LocalTensor<float> sum,LocalTensor<float> error,
        LocalTensor<float> a,LocalTensor<float> b,uint32_t count){
        auto v=calc[6],w=calc[7];
        Add(sum,a,b,count);PipeBarrier<PIPE_V>();Sub(v,sum,a,count);PipeBarrier<PIPE_V>();
        Sub(w,sum,v,count);PipeBarrier<PIPE_V>();Sub(w,a,w,count);Sub(v,b,v,count);PipeBarrier<PIPE_V>();
        Add(error,w,v,count);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void CertAdd(LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c,
        LocalTensor<float> value,uint32_t count){
        CertTwoSum(calc[0],calc[1],a,value,count);
        CertTwoSum(calc[2],calc[3],b,calc[1],count);
        CertTwoSum(calc[4],calc[5],c,calc[3],count);
        CertTwoSum(a,calc[2],calc[0],calc[2],count);
        CertTwoSum(b,c,calc[2],calc[4],count);
    }
    __aicore__ inline void CertProduct(LocalTensor<float> p,LocalTensor<float> error,
        LocalTensor<float> a,LocalTensor<float> b,uint32_t count){
        auto hi=calc[0],lo=calc[1],ch=calc[2],cl=calc[3],v=calc[4];
        Muls(hi,a,pinv::kSplitFactor,count);Muls(ch,b,pinv::kSplitFactor,count);PipeBarrier<PIPE_V>();
        Sub(lo,hi,a,count);Sub(cl,ch,b,count);PipeBarrier<PIPE_V>();
        Sub(hi,hi,lo,count);Sub(ch,ch,cl,count);PipeBarrier<PIPE_V>();
        Sub(lo,a,hi,count);Sub(cl,b,ch,count);Mul(p,a,b,count);PipeBarrier<PIPE_V>();
        Mul(error,hi,ch,count);PipeBarrier<PIPE_V>();Sub(error,error,p,count);PipeBarrier<PIPE_V>();
        Mul(v,hi,cl,count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
        Mul(v,lo,ch,count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
        Mul(v,lo,cl,count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void CertProductScalar(LocalTensor<float> p,LocalTensor<float> error,
        LocalTensor<float> a,float b,uint32_t count){
        auto hi=calc[0],lo=calc[1],v=calc[2];
        volatile float split=b*pinv::kSplitFactor;
        volatile float temp=split-b;
        volatile float ch=split-temp;
        volatile float cl=b-ch;
        Muls(hi,a,pinv::kSplitFactor,count);PipeBarrier<PIPE_V>();Sub(lo,hi,a,count);PipeBarrier<PIPE_V>();
        Sub(hi,hi,lo,count);PipeBarrier<PIPE_V>();Sub(lo,a,hi,count);Muls(p,a,b,count);PipeBarrier<PIPE_V>();
        Muls(error,hi,static_cast<float>(ch),count);PipeBarrier<PIPE_V>();Sub(error,error,p,count);PipeBarrier<PIPE_V>();
        Muls(v,hi,static_cast<float>(cl),count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
        Muls(v,lo,static_cast<float>(ch),count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
        Muls(v,lo,static_cast<float>(cl),count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline Wide CertScalarSum(float a,float b){
        volatile float sum=a+b;
        volatile float v=sum-a;
        volatile float w=sum-v;
        volatile float ar=a-w;
        volatile float br=b-v;
        return {sum,ar+br};
    }
    __aicore__ inline void CertScalarAdd(float &a,float &b,float &c,float value){
        Wide first=CertScalarSum(a,value),second=CertScalarSum(b,first.l),third=CertScalarSum(c,second.l);
        first=CertScalarSum(first.h,second.h);
        second=CertScalarSum(first.l,third.h);
        a=first.h;b=second.h;c=second.l;
    }
    __aicore__ inline void CertScale(LocalTensor<float> a,Wide factor,uint32_t count){
        Duplicate(certH,0.0f,count);Duplicate(certL,0.0f,count);Duplicate(certT,0.0f,count);PipeBarrier<PIPE_V>();
        CertProductScalar(bh,bl,a,factor.h,count);
        CertAdd(certH,certL,certT,bh,count);CertAdd(certH,certL,certT,bl,count);
        CertProductScalar(bh,bl,a,factor.l,count);
        CertAdd(certH,certL,certT,bh,count);CertAdd(certH,certL,certT,bl,count);
    }
    __aicore__ inline Wide CertDot(LocalTensor<float> a,LocalTensor<float> b,
        LocalTensor<float> c,LocalTensor<float> d,uint32_t count){
        Duplicate(certH,0.0f,count);Duplicate(certL,0.0f,count);Duplicate(certT,0.0f,count);PipeBarrier<PIPE_V>();
        LocalTensor<float> left[2]={a,b},right[2]={c,d};
        for(uint32_t i=0;i<2;++i)for(uint32_t j=0;j<2;++j){
            CertProduct(bh,bl,left[i],right[j],count);
            CertAdd(certH,certL,certT,bh,count);CertAdd(certH,certL,certT,bl,count);
        }
        for(uint32_t size=count;size>8;size/=2){
            uint32_t half=size/2;
            CertAdd(certH,certL,certT,certH[half],half);
            CertAdd(certH,certL,certT,certL[half],half);
            CertAdd(certH,certL,certT,certT[half],half);
        }
        VS();float sh=0,sl=0,st=0;
        for(uint32_t i=0;i<8;++i){
            CertScalarAdd(sh,sl,st,certH.GetValue(i));
            CertScalarAdd(sh,sl,st,certL.GetValue(i));
            CertScalarAdd(sh,sl,st,certT.GetValue(i));
        }
        return Plus({sh,sl},{st,0});
    }

    __aicore__ inline Wide Sigma(uint32_t i){VS();return {sigma.GetValue(i),sigmaLow.GetValue(i)};}
    template<bool PRECISE>
    __aicore__ inline void Solve(){
        for(uint32_t i=0;i<t.cols;++i){
            Original(i,xh,xl);VS();xh.SetValue(head+i,1.0f);SV();
            Store(high[i*width],xh,width);Store(low[i*width],xl,width);
        }
        constexpr uint32_t limit=PRECISE?pinv::kRectRobustSweeps:pinv::kJacobiSweeps;
        constexpr float orthogonality=PRECISE?pinv::kRectWideOrthogonality:pinv::kFp32Orthogonality;
        for(uint32_t sweep=0;sweep<limit;++sweep){
            uint32_t changed=0;
            for(uint32_t p=0;p<t.cols;++p){
                Row(p,xh,xl);
                for(uint32_t q=p+1;q<t.cols;++q){
                    Row(q,yh,yl);
                    Wide a=Dot(xh,xl,xh,xl,head),d=Dot(yh,yl,yh,yl,head),h=Dot(xh,xl,yh,yl,head);
                    if(!(a.h>0 && d.h>0) || AbsScalar(Value(h))<=orthogonality*Root(a.h*d.h))continue;
                    Wide tau=Divide(Minus(d,a),Times({2,0},h)),z;
                    if(AbsScalar(tau.h)>pinv::kJacobiLargeTau)z=Divide({0.5f,0},tau);
                    else{
                        Wide root=RootPair(Plus({1,0},Times(tau,tau)));
                        Wide absolute=tau.h<0?Wide{-tau.h,-tau.l}:tau;
                        z=Divide({tau.h<0?-1.0f:1.0f,0},Plus(absolute,root));
                    }
                    Wide c=Divide({1,0},RootPair(Plus({1,0},Times(z,z)))),sn=Times(c,z);
                    if constexpr(PRECISE){
                        Scale(ah,al,xh,xl,c,width);Scale(bh,bl,yh,yl,{-sn.h,-sn.l},width);
                        PairAdd(ah,al,ah,al,bh,bl,width);Store(high[p*width],ah,width);Store(low[p*width],al,width);
                        Scale(ah,al,xh,xl,sn,width);Scale(bh,bl,yh,yl,c,width);
                        PairAdd(ah,al,ah,al,bh,bl,width);Store(high[q*width],ah,width);Store(low[q*width],al,width);
                    }else{
                        Muls(ah,xh,c.h,width);Muls(bh,yh,-sn.h,width);PipeBarrier<PIPE_V>();
                        Add(ah,ah,bh,width);Duplicate(al,0.0f,width);PipeBarrier<PIPE_V>();
                        Store(high[p*width],ah,width);Store(low[p*width],al,width);
                        Muls(ah,xh,sn.h,width);Muls(bh,yh,c.h,width);PipeBarrier<PIPE_V>();
                        Add(ah,ah,bh,width);PipeBarrier<PIPE_V>();Store(high[q*width],ah,width);Store(low[q*width],al,width);
                    }
                    Row(p,xh,xl);++changed;
                }
            }
            if(changed==0)break;
        }
    }
    // See tests/README.md for the block-diagonal perturbation certificate.
    __aicore__ inline bool Certify(float &error){
        float maximum=0,minimum=pinv::kScaleLimit,discarded=0,total=0;
        uint32_t rank=0;
        for(uint32_t i=0;i<t.cols;++i){
            Row(i,xh,xl);Wide pair=RootPair(CertDot(xh,xl,xh,xl,head));float n=Value(pair);
            sigma.SetValue(i,pair.h);sigmaLow.SetValue(i,pair.l);maximum=n>maximum?n:maximum;total+=n*n;
        }
        float cutoff=t.atol/power,relative=t.rtol*maximum;cutoff=cutoff>relative?cutoff:relative;
        for(uint32_t i=0;i<t.cols;++i){
            float n=Value(Sigma(i));bool keep=n>cutoff;retained.SetValue(i,keep?1:0);
            if(keep){++rank;minimum=n<minimum?n:minimum;}
        }
        SV();Wide ov={0,0},ou={0,0},r1={0,0},r2={0,0};
        float discardedGram=0,vGramTotal=0,uGramTotal=0;
        for(uint32_t i=0;i<t.cols;++i){
            Row(i,xh,xl);Wide ni=Sigma(i);float rowBound=0;
            for(uint32_t j=0;j<t.cols;++j){
                Row(j,yh,yl);
                Wide vv=CertDot(xh[head],xl[head],yh[head],yl[head],basis);
                if(i==j)vGramTotal+=Value(vv);
                Wide v=Minus(vv,{i==j?1.0f:0.0f,0});
                ov=Plus(ov,Times(v,v));
                Wide cc=CertDot(xh,xl,yh,yl,head);
                if(retained.GetValue(i) && retained.GetValue(j)){
                    Wide uu=Divide(cc,Times(ni,Sigma(j)));
                    if(i==j)uGramTotal+=Value(uu);
                    Wide u=Minus(uu,{i==j?1.0f:0.0f,0});
                    ou=Plus(ou,Times(u,u));
                }else if(!retained.GetValue(i) && !retained.GetValue(j))rowBound+=AbsScalar(Value(cc));
            }
            discardedGram=rowBound>discardedGram?rowBound:discardedGram;
            Duplicate(ah,0.0f,width);Duplicate(al,0.0f,width);Duplicate(certAccumulator,0.0f,width);PipeBarrier<PIPE_V>();
            for(uint32_t j=0;j<t.cols;++j){
                Original(j,yh,yl);VS();Wide vij={xh.GetValue(head+j),xl.GetValue(head+j)};SV();
                CertScale(yh,vij,head);
                CertAdd(ah,al,certAccumulator,certH,head);
                CertAdd(ah,al,certAccumulator,certL,head);
                CertAdd(ah,al,certAccumulator,certT,head);
                if(retained.GetValue(i)){
                    Wide rhs=Minus(Divide(CertDot(yh,yl,xh,xl,head),ni),Times(vij,ni));
                    r2=Plus(r2,Times(rhs,rhs));
                }
            }
            Muls(bh,xh,-1.0f,head);Muls(bl,xl,-1.0f,head);PipeBarrier<PIPE_V>();
            CertAdd(ah,al,certAccumulator,bh,head);CertAdd(ah,al,certAccumulator,bl,head);
            Add(al,al,certAccumulator,head);PipeBarrier<PIPE_V>();
            r1=Plus(r1,CertDot(ah,al,ah,al,head));
        }
        float norm=Root(total)*pinv::kNormSlack;
        uint32_t headDepth=0,basisDepth=0;
        for(uint32_t size=head;size>1;size/=2)++headDepth;
        for(uint32_t size=basis;size>1;size/=2)++basisDepth;
        // TwoSum reduction depth, normalization, and sequential matrix accumulation
        // have different operation counts. Their error factors use the 2^-48 unit.
        float dotGamma=pinv::kRectPairRoundOps*pinv::kWideRoundoff+pinv::kRectOpFactor*(headDepth+pinv::kRectDotOverhead)*pinv::kTripleRoundoff;
        float basisGamma=pinv::kRectDotCompressOps*pinv::kWideRoundoff+pinv::kRectOpFactor*(basisDepth+pinv::kRectDotOverhead)*pinv::kTripleRoundoff;
        float normGamma=pinv::kRectPairRoundOps*pinv::kWideRoundoff+pinv::kRectOpFactor*(headDepth+pinv::kRectNormOverhead)*pinv::kTripleRoundoff;
        float productGamma=pinv::kRectOpFactor*(t.cols+pinv::kRectDotOverhead)*pinv::kTripleRoundoff;
        float outputGamma=pinv::kRectOpFactor*(t.cols+pinv::kRectDotOverhead)*pinv::kWideRoundoff;
        dotGamma=dotGamma/(1-dotGamma)*pinv::kNormSlack;
        basisGamma=basisGamma/(1-basisGamma)*pinv::kNormSlack;
        normGamma=normGamma/(1-normGamma)*pinv::kNormSlack;
        productGamma=productGamma/(1-productGamma)*pinv::kNormSlack;
        float etaV=Root(Value(ov))*pinv::kNormSlack+basisGamma*vGramTotal*pinv::kNormSlack;
        float etaU=Root(Value(ou))*pinv::kNormSlack+normGamma*uGramTotal*pinv::kNormSlack;
        if(!(etaV<pinv::kRectMaxGramDefect && etaU<pinv::kRectMaxGramDefect))return false;
        float dv=etaV/(1+Root(1-etaV)),du=etaU/(1+Root(1-etaU));
        float firstResidual=Root(Value(r1))*pinv::kNormSlack;
        float rootCols=Root(static_cast<float>(static_cast<int32_t>(t.cols)));
        float basisDenominator=Root(1-etaV)-productGamma*rootCols;
        if(!(basisDenominator>0))return false;
        norm=(norm+firstResidual)/basisDenominator*pinv::kNormSlack;
        float e1=firstResidual+productGamma*norm*rootCols;
        discarded=Root(discardedGram)*pinv::kNormSlack+e1+norm*dv;
        float delta=e1+Root(Value(r2))*pinv::kNormSlack+dotGamma*norm*Root(static_cast<float>(static_cast<int32_t>(rank)))
            +(norm+maximum)*(du+dv);
        float lower=maximum*(1-2*pinv::kUnitRoundoff)-delta,upper=maximum*(1+2*pinv::kUnitRoundoff)+delta;
        float thresholdLow=t.atol/power,thresholdHigh=thresholdLow;
        if(t.rtol*lower>thresholdLow)thresholdLow=t.rtol*lower;
        if(t.rtol*upper>thresholdHigh)thresholdHigh=t.rtol*upper;
        if(rank==0){error=0;return discarded+delta<=thresholdLow;}
        float s=minimum*(1-2*pinv::kUnitRoundoff)-delta,d=rank==t.cols?0:discarded+delta;
        if(!(s>thresholdHigh && (rank==t.cols || d<=thresholdLow) && s>d))return false;
        float lipschitz=1/(s*(s-d));
        error=(lipschitz*delta+(du+dv)/minimum
            +outputGamma*Root(static_cast<float>(static_cast<int32_t>(rank)))/minimum)/power*pinv::kRectBoundSlack;
        return error>=0 && error<pinv::kScaleLimit;
    }
    __aicore__ inline bool Emit(float error){
        bool valid=true;uint32_t count=t.m>=t.n?head:basis,length=t.m;
        for(uint32_t row=0;row<t.n;++row){
            Duplicate(ah,0.0f,width);Duplicate(al,0.0f,width);PipeBarrier<PIPE_V>();
            for(uint32_t i=0;i<t.cols;++i){
                if(!retained.GetValue(i))continue;
                Row(i,xh,xl);Wide n=Sigma(i);VS();
                uint32_t position=t.m>=t.n?head+row:row;
                Wide factor=Divide({xh.GetValue(position),xl.GetValue(position)},Times(n,n));SV();
                uint32_t first=t.m>=t.n?0:head;
                Scale(bh,bl,xh[first],xl[first],factor,count);PairAdd(ah,al,ah,al,bh,bl,count);
            }
            Muls(ah,ah,1/power,count);Muls(al,al,1/power,count);PipeBarrier<PIPE_V>();
            Add(bh,ah,al,count);VS();
            for(uint32_t col=0;col<length;++col){
                float value=bh.GetValue(col);
                float bound=error+pinv::kUnitRoundoff/(1-pinv::kUnitRoundoff)*AbsScalar(value);
                float budget=(pinv::kOutputAtol+pinv::kOutputRtol*AbsScalar(value))/(1+pinv::kOutputRtol);
                if(!(bound<=budget))valid=false;
            }
            SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);
            SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
            DataCopyPad(result[batchOffset+row*t.m],bh,DataCopyExtParams{1,length*4,0,0,0});
            SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
        }
        return valid;
    }
public:
    __aicore__ inline void Init(GM_ADDR src,GM_ADDR dst,GM_ADDR ws,const LinalgPinvTilingData &data,TPipe &pipe){
        t=data;head=pinv::RectAxis(t.rows);basis=pinv::RectAxis(t.cols);width=pinv::RectWidth(t.rows,t.cols);
        uint32_t count=pinv::RectLocalFloats(t.rows,t.cols);
        pipe.InitBuffer(mem,count*4);auto all=mem.Get<float>();uint32_t off=0;
        xh=all;off+=width;xl=all[off];off+=width;yh=all[off];off+=width;yl=all[off];off+=width;
        ah=all[off];off+=width;al=all[off];off+=width;bh=all[off];off+=width;bl=all[off];off+=width;
        for(uint32_t i=0;i<8;++i){calc[i]=all[off];off+=width;}
        certH=all[off];off+=width;certL=all[off];off+=width;certT=all[off];off+=width;certAccumulator=all[off];off+=width;
        red=all[off];off+=32;gather=all[off];off+=8*t.rows;sigma=all[off];off+=basis;sigmaLow=all[off];off+=basis;
        retained=all[off].ReinterpretCast<uint32_t>();
        input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src));result.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dst));
        high.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws));low.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws));
    }
    __aicore__ inline void Process(GM_ADDR ws){
        for(uint32_t batch=GetBlockIdx();batch<t.batch;batch+=AscendC::GetBlockNum()){
            batchOffset=batch*t.m*t.n;
            high.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws)+uint64_t(batch)*2*t.cols*width);
            low.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws)+uint64_t(batch)*2*t.cols*width+t.cols*width);
            power=1;float peak=0;
            for(uint32_t i=0;i<t.cols;++i){
                Original(i,xh,xl);Abs(ah,xh,head);PipeBarrier<PIPE_V>();ReduceMax(red,ah,calc[0],head,false);VS();
                float value=red.GetValue(0);peak=value>peak?value:peak;
            }
            if(peak==0){
                Duplicate(bh,0.0f,width);PipeBarrier<PIPE_V>();
                for(uint32_t row=0;row<t.n;++row){
                    SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
                    DataCopyPad(result[batchOffset+row*t.m],bh,DataCopyExtParams{1,t.m*4,0,0,0});
                    SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
                }continue;
            }
            while(power<peak && power<pinv::kScaleLimit)power*=2;
            while(power*0.5f>=peak)power*=0.5f;
            Solve<false>();float error=0;
            if(Certify(error) && Emit(error))continue;
            Solve<true>();
            if(Certify(error) && Emit(error))continue;
            // An uncertified result is visible to the normal finite-output check.
            Duplicate(bh,0.0f,width);PipeBarrier<PIPE_V>();Div(bh,bh,bh,width);PipeBarrier<PIPE_V>();
            for(uint32_t row=0;row<t.n;++row){
                SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
                DataCopyPad(result[batchOffset+row*t.m],bh,DataCopyExtParams{1,t.m*4,0,0,0});
                SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
            }
        }
    }
};
