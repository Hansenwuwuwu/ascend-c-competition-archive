#pragma once
#include "pinv_cube_residual.h"
#include "kernel_operator.h"
#include "linalg_pinv_tiling.h"
using namespace AscendC;
class WindowPivotProducer {
    TBuf<TPosition::VECCALC> mem;
    LocalTensor<float> metaData, a, scratch, factors, v, prod, tmp, red;
    LocalTensor<uint32_t> indices, perm;
    GlobalTensor<float> input, stage;
    uint32_t n,k,wide,window;
    __aicore__ inline void VS(){SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);}
    __aicore__ inline void SV(){SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);}
public:
    __aicore__ inline LocalTensor<float> Flags(){return scratch;}
    __aicore__ inline void Init(GM_ADDR in,GM_ADDR workspace,const LinalgPinvTilingData &data,TPipe &pipe,uint32_t batch){
        n=data.cols;k=(n+7)/8*8;wide=(k+63)/64*64;
        window=data.local;
        uint32_t trailing=n-window,trailingPitch=(trailing+7)/8*8;
        uint32_t capacity=n*window>trailing*trailingPitch?n*window:trailing*trailingPitch;
        uint32_t count=capacity+4096+10*k+2*wide+64+(n+15)/16*64;
        pipe.InitBuffer(mem,count*4);auto all=mem.Get<float>();uint32_t off=0;
        a=all;off+=capacity;scratch=all[off];off+=4096;factors=all[off];off+=8*k;
        v=all[off];off+=wide;prod=all[off];off+=wide;tmp=all[off];off+=32;red=all[off];off+=32;
        indices=all[off].ReinterpretCast<uint32_t>();off+=k;perm=all[off].ReinterpretCast<uint32_t>();off+=k;metaData=all[off];
        input.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(in));
        stage.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace)+uint64_t(batch)*n*data.stride);
    }
    __aicore__ inline bool Run(uint32_t batch,GlobalTensor<float> metadata){

        uint32_t offset=batch*n*n;
        float scale=0.0f;
        constexpr uint32_t chunkSize=4096;
        for(uint32_t begin=0;begin<n*n;begin+=chunkSize){
            uint32_t count=n*n-begin<chunkSize?n*n-begin:chunkSize;
            SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
            DataCopyPad(scratch,input[offset+begin],DataCopyExtParams{1,count*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
            SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
            Abs(scratch,scratch,count);PipeBarrier<PIPE_V>();
            ReduceMax(red,scratch,prod,count,false);VS();
            float value=red.GetValue(0);scale=value>scale?value:scale;
        }
        bool valid=scale>=0.1f && scale<pinv::kScaleLimit;
        float power=1.0f;if(valid){while(power<scale)power*=2;while(power*0.5f>=scale)power*=0.5f;}
        float total=0.0f;
        for(uint32_t begin=0;begin<n*n;begin+=chunkSize){
            uint32_t count=n*n-begin<chunkSize?n*n-begin:chunkSize;
            SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
            DataCopyPad(scratch,input[offset+begin],DataCopyExtParams{1,count*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
            SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
            Muls(scratch,scratch,1.0f/power,count);PipeBarrier<PIPE_V>();
            Mul(scratch,scratch,scratch,count);PipeBarrier<PIPE_V>();
            ReduceSum(red,scratch,prod,count);VS();total+=red.GetValue(0);
        }
        DataCopyPad(a,input[offset],DataCopyExtParams{static_cast<uint16_t>(n),window*4,(n-window)*4,0,0},
            DataCopyPadExtParams<float>{false,0,0,0});
        SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
        for(uint32_t begin=0;begin<n*window;begin+=16320){
            uint32_t count=n*window-begin<16320?n*window-begin:16320;
            Muls(a[begin],a[begin],1.0f/power,count);
        }
        PipeBarrier<PIPE_V>();
        for(uint32_t row=0;row<k;++row){indices.SetValue(row,row*k*4);perm.SetValue(row,row);}SV();


        uint32_t size=n,pitch=window;
        for(uint32_t first=0;first<n;first+=16){
            if(first==window){
                PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<0,PIPE_MTE3>(3);CrossCoreWaitFlag(3);
                CrossCoreSetFlag<0,PIPE_MTE3>(2);CrossCoreWaitFlag(2);
                if(valid){
                    for(uint32_t row=0;row<size;++row){
                        uint32_t original=perm.GetValue(row);
                        DataCopyPad(a[row*pitch],stage[original*k+window],
                            DataCopyExtParams{1,size*4,0,0,0},
                            DataCopyPadExtParams<float>{true,0,static_cast<uint8_t>(pitch-size),0});
                    }
                    SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
                }
                PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<0,PIPE_MTE3>(4);CrossCoreWaitFlag(4);
            }
            if(valid){
            auto offsets=indices.ReinterpretCast<int32_t>();CreateVecIndex(offsets,int32_t(0),size);PipeBarrier<PIPE_V>();Muls(offsets,offsets,static_cast<int32_t>(pitch*4),size);PipeBarrier<PIPE_V>();
            uint32_t block=size<16?size:16;
            for(uint32_t p=0;p<block;++p){
                uint32_t remaining=size-p;
                Gather(v,a,indices,(p*pitch+p)*4,remaining);PipeBarrier<PIPE_V>();
                Abs(prod,v,remaining);PipeBarrier<PIPE_V>();
                uint32_t full=remaining/64,tail=remaining%64,chunks=full+(tail!=0);
                if(full)WholeReduceMax(red,prod,uint64_t(64),static_cast<uint8_t>(full),1,1,8);
                if(tail)WholeReduceMax(red[full*2],prod[full*64],static_cast<uint64_t>(tail),1,1,1,8);
                uint64_t summary[2]={uint64_t(0x5555)&((uint64_t(1)<<(chunks*2))-1),0};
                PipeBarrier<PIPE_V>();WholeReduceMax(tmp,red,summary,1,1,1,8);VS();
                float best,bits;GetReduceRepeatMaxMinSpr<float>(best,bits);
                uint32_t chunk=(*reinterpret_cast<uint32_t *>(&bits))/2;
                uint32_t selected=p+chunk*64+red.ReinterpretCast<uint32_t>().GetValue(chunk*2+1);
                if(!(best>1.0e-6f)){valid=false;break;}
                float divisor=v.GetValue(selected-p);
                if(selected!=p){
                    uint32_t old=perm.GetValue(p);perm.SetValue(p,perm.GetValue(selected));perm.SetValue(selected,old);
                }
                metaData.SetValue(first/16*64+p,static_cast<float>(static_cast<int32_t>(perm.GetValue(p))));
                metaData.SetValue(first/16*64+16+p,1.0f/divisor);
                if(remaining<=1)continue;
                v.SetValue(selected-p,0.0f);
                DataCopy(scratch,a[selected*pitch],pitch);PipeBarrier<PIPE_V>();
                Muls(scratch,scratch,-(1.0f/divisor),pitch);PipeBarrier<PIPE_V>();
                SV();Brcb(factors,v,(remaining+7)/8,{1,8});PipeBarrier<PIPE_V>();
                for(uint32_t col=(p/8)*8;col<pitch;col+=64){
                    uint32_t count=pitch-col<64?pitch-col:64;
                    for(uint32_t row=0;row<remaining;row+=255){
                        uint32_t repeats=remaining-row<255?remaining-row:255;
                        MulAddDst(a[(p+row)*pitch+col],scratch[col],factors[row*8],static_cast<uint64_t>(count),static_cast<uint8_t>(repeats),{1,1,0,static_cast<uint8_t>(pitch/8),0,1});
                    }
                }
                PipeBarrier<PIPE_V>();
                if(selected!=p){DataCopy(a[selected*pitch],a[p*pitch],pitch);PipeBarrier<PIPE_V>();}
            }
            uint32_t remaining=size-block;
            uint32_t columns=first+block<window?window-first-block:n-first-block;
            uint32_t newPitch=(columns+7)/8*8;
            VS();
            if(first+block!=window)for(uint32_t row=0;row<remaining;row+=16){
                uint32_t count=remaining-row<16?remaining-row:16;
                DataCopy(a[row*newPitch],a[(row+block)*pitch+block],DataCopyParams{static_cast<uint16_t>(count),static_cast<uint16_t>(newPitch/8),static_cast<uint16_t>((pitch-newPitch)/8),0});
                PipeBarrier<PIPE_V>();
            }
            if(remaining){
                auto order=perm.ReinterpretCast<int32_t>();auto temporary=scratch.ReinterpretCast<int32_t>();
                uint32_t count=(remaining+7)/8*8;SV();DataCopy(temporary,order[block],count);PipeBarrier<PIPE_V>();
                DataCopy(order,temporary,count);VS();
            }
            SV();size=remaining;pitch=newPitch;
            }
            // 两个标记分别保护主元元数据和主元数据包；延后等待以重叠下一面板。
            if(first){CrossCoreWaitFlag(0);CrossCoreWaitFlag(1);}
            auto record=metaData[first/16*64];
            record.SetValue(32,power);record.SetValue(33,total);record.SetValue(34,valid?1.0f:0.0f);
            SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);
            DataCopy(metadata[first/16*64],record,64);
            CrossCoreSetFlag<0,PIPE_MTE3>(0);CrossCoreSetFlag<0,PIPE_MTE3>(1);
        }
        CrossCoreWaitFlag(0);CrossCoreWaitFlag(1);


        return valid;
    }
};
class WindowPipelineRows {
    TBuf<TPosition::VECCALC> mem;
    LocalTensor<float> a, packets, out, factors, meta, norms, v, prod, acc, tmp, red;
    LocalTensor<uint32_t> rowIndices, packetIndices, mapping;
    GlobalTensor<float> input, result, work;
    uint32_t n,k,stride,owned,core,cores,packetStride,wide,factorRows,window;
    __aicore__ inline void VS(){SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);}
    __aicore__ inline void SV(){SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);}
public:
    float inverseTotal=0;
    __aicore__ inline LocalTensor<float> Flags(){return out;}
    __aicore__ inline void Init(GM_ADDR in,GM_ADDR dst,GM_ADDR workspace,const LinalgPinvTilingData &t,TPipe &pipe,uint32_t batch,uint32_t c,uint32_t count){
        window=t.local;n=t.cols;k=(n+7)/8*8;stride=t.stride;core=c;cores=count;owned=(n+cores-1-core)/cores;
        wide=(k+63)/64*64;factorRows=(owned+7)/8*8;packetStride=k+16;
        uint32_t outSize=owned*k>512?owned*k:512;
        uint32_t size=owned*stride+16*packetStride+outSize+8*(factorRows>16?factorRows:16)+64+5*wide+32+factorRows+16+k;
        pipe.InitBuffer(mem,size*4);auto all=mem.Get<float>();uint32_t off=0;
        a=all;off+=owned*stride;packets=all[off];off+=16*packetStride;out=all[off];off+=outSize;
        factors=all[off];off+=8*(factorRows>16?factorRows:16);meta=all[off];off+=64;
        norms=all[off];off+=wide;v=all[off];off+=wide;prod=all[off];off+=wide;acc=all[off];off+=wide;tmp=all[off];off+=wide;
        red=all[off];off+=32;rowIndices=all[off].ReinterpretCast<uint32_t>();off+=factorRows;
        packetIndices=all[off].ReinterpretCast<uint32_t>();off+=16;mapping=all[off].ReinterpretCast<uint32_t>();
        input.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(in));result.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dst));
        work.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace)+uint64_t(batch)*n*stride);
        Duplicate(a,0.0f,owned*stride);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
        DataCopyPad(a,input[batch*n*n+core*n],DataCopyExtParams{static_cast<uint16_t>(owned),n*4,(cores-1)*n*4,(stride-k)/8,0},DataCopyPadExtParams<float>{true,0,static_cast<uint8_t>(k-n),0});
        SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
        Duplicate(norms,1.0f,k);VS();
        for(uint32_t row=0;row<factorRows;++row)rowIndices.SetValue(row,row<owned?row*stride*4:0);
        for(uint32_t row=0;row<16;++row)packetIndices.SetValue(row,row*packetStride*4);SV();
    }
    __aicore__ inline bool Run(uint32_t batch,GlobalTensor<float> metadata){
        bool valid=true;float power=1;
        for(uint32_t first=0;first<n;first+=16){
            uint32_t rows=n-first<16?n-first:16;
            PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<0,PIPE_MTE3>(0);CrossCoreWaitFlag(0);
            DataCopy(meta,metadata[first/16*64],64);
            SetFlag<HardEvent::MTE2_S>(0);WaitFlag<HardEvent::MTE2_S>(0);
            SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
            valid=valid && meta.GetValue(34)>0.5f;power=meta.GetValue(32);
            if(valid){
                if(first==0){
                    for(uint32_t begin=0;begin<owned*stride;begin+=16320){uint32_t count=owned*stride-begin<16320?owned*stride-begin:16320;Muls(a[begin],a[begin],1.0f/power,count);}PipeBarrier<PIPE_V>();
                }
                SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
                for(uint32_t column=0;column<rows;++column){
                    uint32_t selected=static_cast<uint32_t>(static_cast<int32_t>(meta.GetValue(column)));
                    norms.SetValue(selected,-static_cast<float>(static_cast<int32_t>(first+column+1)));
                    if(selected%cores==core){
                        a.SetValue((selected/cores)*stride+k+first+column,1.0f);
                        SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);
                        DataCopyPad(work[(first+column)*packetStride],a[(selected/cores)*stride+first],DataCopyExtParams{1,packetStride*4,0,0,0});
                        SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
                    }
                }
            }
            CrossCoreSetFlag<0,PIPE_MTE3>(1);CrossCoreWaitFlag(1);
            if(valid){
                DataCopy(packets,work[first*packetStride],rows*packetStride);
                SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
                for(uint32_t column=0;column<rows;++column){
                    Gather(prod,packets,packetIndices,column*4,rows);PipeBarrier<PIPE_V>();
                    float reciprocal=meta.GetValue(16+column);
                    Muls(packets[column*packetStride],packets[column*packetStride],reciprocal,packetStride);PipeBarrier<PIPE_V>();
                    if(column+1<rows){
                        Muls(prod,prod,-1.0f,16);PipeBarrier<PIPE_V>();Brcb(factors,prod,2,{1,8});PipeBarrier<PIPE_V>();
                        for(uint32_t col=0;col<packetStride;col+=64){uint32_t count=packetStride-col<64?packetStride-col:64;
                            MulAddDst(packets[(column+1)*packetStride+col],packets[column*packetStride+col],factors[(column+1)*8],static_cast<uint64_t>(count),static_cast<uint8_t>(rows-column-1),{1,1,0,static_cast<uint8_t>(packetStride/8),0,1});
                        }
                        PipeBarrier<PIPE_V>();
                    }
                }
                for(uint32_t column=0;column<rows;++column){
                    uint32_t selected=static_cast<uint32_t>(static_cast<int32_t>(meta.GetValue(column)));
                    if(selected%cores==core)DataCopy(a[(selected/cores)*stride+first],packets[column*packetStride],packetStride);
                    PipeBarrier<PIPE_V>();
                    Gather(v,a,rowIndices,(first+column)*4,factorRows);PipeBarrier<PIPE_V>();
                    if(selected%cores==core){VS();v.SetValue(selected/cores,0.0f);SV();}
                    Muls(v,v,-1.0f,factorRows);PipeBarrier<PIPE_V>();Brcb(factors,v,factorRows/8,{1,8});PipeBarrier<PIPE_V>();
                    uint32_t pivotColumn=first+column;bool consumed=(pivotColumn&7)==7 || pivotColumn+1==n;
                    uint32_t liveBegin=pivotColumn/8*8+(consumed?8:0),liveLimit=liveBegin+k+(consumed?0:8);
                    for(uint32_t col=liveBegin;col<liveLimit;col+=64){uint32_t count=liveLimit-col<64?liveLimit-col:64;
                        MulAddDst(a[col],packets[column*packetStride+col-first],factors,static_cast<uint64_t>(count),static_cast<uint8_t>(owned),{1,1,0,static_cast<uint8_t>(stride/8),0,1});
                    }
                    PipeBarrier<PIPE_V>();
                }
            }
            if(first+rows==window){
                PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<0,PIPE_MTE3>(3);CrossCoreWaitFlag(3);
                if(valid){
                    SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
                    for(uint32_t row=0;row<owned;++row){
                        DataCopyPad(work[(core+row*cores)*k+window],a[row*stride+window],
                            DataCopyExtParams{1,(n-window)*4,0,0,0});
                    }
                }
                PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<0,PIPE_MTE3>(2);CrossCoreWaitFlag(2);
                CrossCoreSetFlag<0,PIPE_MTE3>(4);CrossCoreWaitFlag(4);
            }
        }
        if(!valid)return false;
        VS();for(uint32_t j=0;j<n;++j)mapping.SetValue(j,(static_cast<uint32_t>(static_cast<int32_t>(-norms.GetValue(j)))-1)*4);SV();
        Duplicate(acc,0.0f,wide);PipeBarrier<PIPE_V>();
        for(uint32_t row=0;row<owned;++row){
            Gather(out[row*k],a[row*stride+k],mapping,0,n);PipeBarrier<PIPE_V>();
            Mul(prod,out[row*k],out[row*k],n);PipeBarrier<PIPE_V>();Add(acc,acc,prod,n);PipeBarrier<PIPE_V>();
        }
        ReduceSum(red,acc,tmp,k);VS();inverseTotal=red.GetValue(0);
        for(uint32_t row=0;row<owned;++row){
            Muls(out[row*k],out[row*k],1.0f/power,k);PipeBarrier<PIPE_V>();
            uint32_t target=static_cast<uint32_t>(static_cast<int32_t>(-norms.GetValue(core+row*cores)))-1;
            SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
            DataCopyPad(result[batch*n*n+target*n],out[row*k],DataCopyExtParams{1,n*4,0,0,0});
        }
        PipeBarrier<PIPE_ALL>();return true;
    }
};
__aicore__ inline uint32_t TryWindowPipelinePinv(GM_ADDR in,GM_ADDR dst,GM_ADDR workspace,const LinalgPinvTilingData &data){
    uint32_t all=PinvVectorBlockCount(),physical=GetBlockIdx(),groupSize=all/data.batch,workers=groupSize-1,batch=physical/groupSize,core=physical%groupSize;
    TPipe pipe;LocalTensor<float> local;bool valid;float inverse=0;
    GlobalTensor<float> metadata,result;
    metadata.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace)+pinv::MetadataOffset(data.batch,data.cols,data.stride,batch));
    result.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dst));
    if(core==workers){WindowPivotProducer op;op.Init(in,workspace,data,pipe,batch);valid=op.Run(batch,metadata);local=op.Flags();}
    else{WindowPipelineRows op;op.Init(in,dst,workspace,data,pipe,batch,core,workers);valid=op.Run(batch,metadata);inverse=op.inverseTotal;local=op.Flags();}
    PipeBarrier<PIPE_ALL>();local.SetValue(0,valid?1.0f:0.0f);local.SetValue(1,inverse);
    SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);
    DataCopyPad(metadata[pinv::kGroupStatus+core*8],local,DataCopyExtParams{1,8,0,0,0});SyncAll();
    uint32_t accepted=0,prepared=0,k=(data.cols+7)/8*8,owned=(data.cols+all-1)/all;
    for(uint32_t b=0;b<data.batch;++b){
        metadata.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace)+pinv::MetadataOffset(data.batch,data.cols,data.stride,b));
        DataCopy(local,metadata[pinv::kGroupStatus],groupSize*8);
        SetFlag<HardEvent::MTE2_S>(0);WaitFlag<HardEvent::MTE2_S>(0);
        bool sound=true;float inverseTotal=0;
        for(uint32_t c=0;c<groupSize;++c){if(local.GetValue(c*8)<0.5f)sound=false;inverseTotal+=local.GetValue(c*8+1);}
        DataCopy(local,metadata,64);
        SetFlag<HardEvent::MTE2_S>(0);WaitFlag<HardEvent::MTE2_S>(0);
        float power=local.GetValue(32),total=local.GetValue(33);
        local.SetValue(256,total);
        SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);Sqrt(local[256],local[256],1);
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
        float cutoff=local.GetValue(256)*data.rtol,absolute=data.atol/power;cutoff=cutoff>absolute?cutoff:absolute;
        sound=sound && inverseTotal>0 && inverseTotal*cutoff*cutoff<pinv::kGapGateSquared;
        if(sound && total*inverseTotal<pinv::kDirectConditionLimit)accepted|=uint32_t(1)<<b;
        else if(sound && total*inverseTotal<pinv::kRefineConditionLimit && owned*k<=8*(k+k)
                && data.inverseTile*data.stride>=owned*8 && 2*all*k<=data.cols*data.stride){
            prepared|=uint32_t(1)<<b;
            for(uint32_t row=physical;row<data.cols;row+=all){
                SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
                DataCopyPad(local,result[b*data.cols*data.cols+row*data.cols],
                    DataCopyExtParams{1,data.cols*4,0,0,0},
                    DataCopyPadExtParams<float>{true,0,static_cast<uint8_t>(k-data.cols),0});
                SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
                Muls(local,local,power,k);
                SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
                DataCopyPad(result[b*data.cols*data.cols+row*data.cols],local,
                    DataCopyExtParams{1,data.cols*4,0,0,0});
                SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
            }
            PipeBarrier<PIPE_ALL>();
            if(physical==0){
                local.SetValue(0,total);local.SetValue(1,inverseTotal);local.SetValue(2,power);
                SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);
                DataCopyPad(metadata[pinv::kPrepared],local,DataCopyExtParams{1,12,0,0,0});
                SetFlag<HardEvent::MTE3_S>(0);WaitFlag<HardEvent::MTE3_S>(0);
            }
        }
    }
    PipeBarrier<PIPE_ALL>();SyncAll();pipe.Destroy();
    return accepted|(prepared<<4);
}

