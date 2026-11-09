#pragma once
#include <algorithm>
#include <array>
#include <vector>
#include <cstdint>
#include "../op_kernel/fire_candidate_decode_fusion_tiling.h"

struct FireHostPlan {
    FireCandidateDecodeFusionPackedTilingData packet{};
    uint32_t launchBlocks=0, tile=0;
};
inline FireHostPlan fire_make_plan(uint32_t batch,uint32_t n,bool halfInput,uint32_t available) {
    FireHostPlan result;
        FireCandidateDecodeFusionTilingData expanded{};
        auto *tiling = &expanded;
        auto *packet = &result.packet;
        tiling->batch = batch;
        tiling->n = n;
        const uint32_t totalRows=tiling->batch*tiling->n;
        const uint32_t cores=available;
        const bool compactSingleTile=totalRows<=8192;
        const uint32_t batchAlign=halfInput?16:8;
        const uint32_t batchUnits=(n+batchAlign-1)/batchAlign;
        // Amortize preparation with one bounded 1024-row resource per owner.
        // The original aligned ownership still handles each partial tile.
        const uint32_t desiredOwners=n/1024+uint32_t(n%1024!=0);
        const bool balanced1024Owners=!compactSingleTile && totalRows<cores*1664*3 &&
            uint64_t(batch)*desiredOwners<=cores;
        const uint32_t batchWorkers=balanced1024Owners?desiredOwners:
            std::min({cores/batch,batchUnits,std::max(1u,n/512)});
        const uint32_t maxBatchRows=batchWorkers?
            (batchUnits+batchWorkers-1)/batchWorkers*batchAlign:0;
        const bool batchOne=!compactSingleTile && totalRows<cores*1664*3 &&
            batchWorkers && maxBatchRows<=1024;
        if(compactSingleTile) {
            tiling->tile=tiling->n<=32?32:(tiling->n<=64?64:128);
            uint32_t tiles=tiling->batch*((tiling->n+tiling->tile-1)/tiling->tile);
            tiling->blocks=std::min(cores,tiles);
        } else if(totalRows>=cores*1664*3) {
            const uint32_t originalTiles=(tiling->n+1663)/1664;
            const uint32_t balancedRows=(tiling->n+originalTiles-1)/originalTiles;
            const uint32_t rowAlign=halfInput?16:8;
            const uint32_t alignedRows=(balancedRows+rowAlign-1)/rowAlign*rowAlign;
            tiling->tile=!halfInput && alignedRows<=1640?1640:1664;
            tiling->blocks=std::min(cores,tiling->batch*((tiling->n+tiling->tile-1)/tiling->tile));
        } else if(batchOne) {
            tiling->tile=maxBatchRows<=512?512:(maxBatchRows<=528?528:(maxBatchRows<=640?640:((halfInput && maxBatchRows<=896)?896:1024)));
            tiling->blocks=batchWorkers*batch;
        } else {
            uint32_t rowAlign=halfInput?16:8;
            uint32_t workers=std::min(cores,std::max(1u,totalRows/512));
            if(workers>=8)workers=workers/8*8;
            uint32_t units=(totalRows+rowAlign-1)/rowAlign;
            uint32_t maxWorkerRows=(units+workers-1)/workers*rowAlign;
            tiling->blocks=workers;
            tiling->tile=maxWorkerRows<=512?512:(maxWorkerRows<=528?528:(maxWorkerRows<=640?640:1024));
        }
        uint32_t TILE=tiling->tile;
        uint32_t launchBlocks=tiling->blocks;
        if(compactSingleTile)launchBlocks=std::min(cores,std::max(launchBlocks,8u));
        else launchBlocks=std::min(cores,((launchBlocks+7)/8)*8);
        const uint32_t rowAlign=halfInput?16:8;
        const uint32_t units=(totalRows+rowAlign-1)/rowAlign;
        tiling->unitsPerWorker=batchOne?batchUnits/batchWorkers:units/tiling->blocks;
        tiling->extraUnits=batchOne?batchUnits%batchWorkers:units%tiling->blocks;
        tiling->totalRows=totalRows;
        tiling->denseOne=(tiling->tile<=128 && tiling->n<=tiling->tile &&
            tiling->blocks==tiling->batch) ||
            (tiling->tile>128 && tiling->tile<=1024 && tiling->n>=tiling->tile &&
             (tiling->unitsPerWorker+(tiling->extraUnits!=0))*rowAlign<=tiling->tile);
        result.launchBlocks=launchBlocks;
        // Reuse the existing unused unit field for an exactly uniform
        // large-Float tile partition. Zero keeps the original partition.
        // Bound every original uint32 multiplication before applying the
        // arithmetic identity; the fallback retains its original behavior.
        uint32_t packedUnits=tiling->tile>1024?0u:tiling->unitsPerWorker;
        if(!halfInput && TILE==1640) {
            const uint64_t rounded=uint64_t(n)+TILE-1;
            const uint64_t tiles=(rounded/TILE)*batch;
            if(rounded<=UINT32_MAX && tiles<=UINT32_MAX/tiling->blocks &&
                tiles%tiling->blocks==0 && tiles/tiling->blocks<=65535)
                packedUnits=static_cast<uint32_t>(tiles/tiling->blocks);
        }
        packet->n=tiling->n;packet->totalRows=tiling->totalRows;
        packet->batch=tiling->batch;
        // A full-capacity partition has the same constant count on every
        // owner. All other aligned/tail partitions retain the generic path.
        // The new bounded resource retains the original partial-entry flag.
        const bool uniformFull=batchOne && TILE!=896 && n%batchAlign==0 &&
            batchUnits%batchWorkers==0 && maxBatchRows==(TILE+127)/128*128;
        const uint32_t packedDense=batchOne?uint32_t(uniformFull):tiling->denseOne;
        packet->control=packedUnits|(tiling->extraUnits<<16)|
            (tiling->blocks<<22)|(packedDense<<28)|(uint32_t(batchOne)<<29);
        if(batchOne && !(batch&(batch-1))) {
            const uint32_t shift=(batch>1)+(batch>2)+(batch>4);
            packet->control|=shift<<30;
        }
        result.tile=TILE;
        return result;
}

// Byte offsets for the two existing Gather address streams.
// Capacity is a tiling resource; no tensor values participate here.
inline std::vector<uint32_t> fire_small_index_offsets(uint32_t capacity) {
    std::vector<uint32_t> words(2*capacity);
    for(uint32_t i=0;i<capacity;++i) {
        words[i]=i*6*sizeof(float);words[capacity+i]=i*5*sizeof(float);
    }
    return words;
}

// Generic UB address metadata; this depends only on the compiled capacity.
// It contains no tensor values or numerical outputs.
inline std::vector<uint64_t> fire_address_contexts(uint32_t tile,bool nativeFullHalf=false) {
    const uint32_t G=8,Q=(tile+127)/128*16,P=G*Q+(nativeFullHalf?16:24);
    const uint64_t S0=5*P*4+128,V0=S0+6*P*4+128;
    const uint64_t O0=V0+6*P*4+128,F0=O0+6*P*4+128;
    std::vector<uint64_t> words;words.reserve(128);
    auto emit=[&](const std::array<uint64_t,16>& source,const std::array<uint64_t,16>& destination) {
        for(const auto* addresses:{&source,&destination})for(uint32_t part=0;part<4;++part) {
            uint64_t word=0;
            for(uint32_t lane=0;lane<4;++lane)
                word|=(((*addresses)[part*4+lane]>>5)&0x1fffULL)<<(16*lane);
            words.push_back(word);
        }
    };
    if(nativeFullHalf) {
        for(uint32_t block=0;block<3;++block) {
            std::array<uint64_t,16> source{},destination{};
            for(uint32_t i=0;i<16;++i) {
                source[i]=V0+(i*6*G+block*16)*2;
                const uint32_t j=block*16+i;
                destination[i]=O0+((j%6)*P+(j/6)*Q)*2;
            }
            emit(source,destination);
        }
        // Preserve the original16-context packet layout. Unused raw slots
        // are skipped by the native Half shuffle, not by input values.
        for(uint32_t word=0;word<3*8;++word)words.push_back(0);
    } else for(uint32_t block=0;block<6;++block) {
        std::array<uint64_t,16> source{},destination{};
        for(uint32_t i=0;i<16;++i)source[i]=O0+(i*6*G+block*G)*4;
        for(uint32_t i=0;i<8;++i) {
            uint32_t j=block*8+i,offset=(j%6)*P+(j/6)*Q;
            destination[2*i]=S0+offset*4;destination[2*i+1]=S0+(offset+8)*4;
        }
        emit(source,destination);
    }
    for(uint32_t block=0;block<5;++block) {
        std::array<uint64_t,16> source{},destination{};
        for(uint32_t i=0;i<16;++i)source[i]=F0+(i*5*G+block*8)*4;
        for(uint32_t i=0;i<8;++i) {
            uint32_t j=block*8+i,offset=(j%5)*P+(j/5)*Q;
            destination[2*i]=offset*4;destination[2*i+1]=(offset+8)*4;
        }
        emit(source,destination);
    }
    for(uint32_t block=0;block<5;++block) {
        std::array<uint64_t,16> source{},destination{};
        for(uint32_t i=0;i<8;++i) {
            uint32_t j=block*8+i,offset=(j%5)*P+(j/5)*Q;
            source[i]=S0+offset*4;source[i+8]=S0+(offset+8)*4;
            destination[2*i]=F0+(i*5*G+block*8)*4;
            destination[2*i+1]=F0+((i+8)*5*G+block*8)*4;
        }
        emit(source,destination);
    }
    return words;
}

inline std::vector<uint64_t> fire_mid_low_address_contexts(uint32_t tile,bool nativeHalf) {
    const auto original=fire_address_contexts(tile,nativeHalf);
    std::vector<uint64_t> result;result.reserve(256);
    for(uint32_t i=0;i<128;i+=2) {
        result.push_back(original[i]);result.push_back(original[i+1]);
        result.push_back(0);result.push_back(0);
    }
    return result;
}

inline std::vector<uint64_t> fire_large_low_address_contexts(uint32_t tile) {
    auto compact=fire_address_contexts(tile,false);
    const uint32_t G=8,Q=(tile+127)/128*16,P=G*Q+24;
    const uint64_t S0=5*P*4+128,V0=S0+6*P*4+128;
    for(uint32_t block=0;block<5;++block) {
        std::array<uint64_t,16> source{},destination{};
        for(uint32_t i=0;i<8;++i) {
            const uint32_t j=block*8+i,offset=(j%5)*P+(j/5)*Q;
            source[i]=S0+offset*4;source[i+8]=S0+(offset+8)*4;
            destination[2*i]=V0+(i*5*G+block*8)*4;
            destination[2*i+1]=V0+((i+8)*5*G+block*8)*4;
        }
        for(const auto* addresses:{&source,&destination})for(uint32_t part=0;part<4;++part) {
            uint64_t word=0;
            for(uint32_t lane=0;lane<4;++lane)word|=(((*addresses)[part*4+lane]>>5)&0x1fffULL)<<(16*lane);
            compact.push_back(word);
        }
    }
    std::vector<uint64_t> result;result.reserve(336);
    for(uint32_t i=0;i<168;i+=2) {
        result.push_back(compact[i]);result.push_back(compact[i+1]);
        result.push_back(0);result.push_back(0);
    }
    return result;
}

inline std::vector<uint64_t> fire_large_half_low_address_contexts(uint32_t tile) {
    const uint32_t G=8,Q=(tile+127)/128*16,P=G*Q+16;
    const uint64_t S0=5*P*4+448,V0=S0+6*P*4+128;
    const uint64_t O0=V0+6*P*4+128,F0=O0+6*P*4+128;
    std::vector<uint64_t> result;result.reserve(288);
    auto append=[&](const std::array<uint64_t,16>& src,const std::array<uint64_t,16>& dst) {
        for(const auto* addresses:{&src,&dst})for(uint32_t part=0;part<2;++part) {
            for(uint32_t word=0;word<2;++word) {
                uint64_t packed=0;
                for(uint32_t lane=0;lane<4;++lane)
                    packed|=(((*addresses)[part*8+word*4+lane]>>5)&0x1fffULL)<<(16*lane);
                result.push_back(packed);
            }
            result.push_back(0);result.push_back(0);
        }
    };
    for(uint32_t block=0;block<3;++block) {
        std::array<uint64_t,16> src{},dst{};
        for(uint32_t i=0;i<16;++i) {
            const uint32_t j=block*16+i;
            src[i]=O0+2*(i*6*G+block*16);
            dst[i]=V0+2*((j%6)*P+(j/6)*Q);
        }
        append(src,dst);
    }
    for(uint32_t block=0;block<5;++block) {
        std::array<uint64_t,16> src{},dst{};
        for(uint32_t i=0;i<16;++i)src[i]=F0+4*(i*40+block*8);
        for(uint32_t i=0;i<8;++i) {
            const uint32_t j=block*8+i,offset=(j%5)*P+(j/5)*Q;
            dst[2*i]=4*offset;dst[2*i+1]=4*(offset+8);
        }
        append(src,dst);
    }
    for(uint64_t sink:{F0,V0})for(uint32_t block=0;block<5;++block) {
        std::array<uint64_t,16> src{},dst{};
        for(uint32_t i=0;i<8;++i) {
            const uint32_t j=block*8+i,offset=(j%5)*P+(j/5)*Q;
            src[i]=S0+4*offset;src[i+8]=S0+4*(offset+8);
            dst[2*i]=sink+4*(i*5*G+block*8);
            dst[2*i+1]=sink+4*((i+8)*5*G+block*8);
        }
        append(src,dst);
    }
    return result;
}

