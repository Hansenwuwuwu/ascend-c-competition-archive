// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

#include "../op_kernel/swi_glu_tiling.h"
#include "../op_kernel/tiling_key_swi_glu.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const auto &shape = context->GetInputShape(0)->GetStorageShape();
        if (shape.GetDimNum() == 0) {
            return ge::GRAPH_FAILED;
        }
        const int64_t dim = shape.GetDim(shape.GetDimNum() - 1);
        if (dim < 1 || dim > 8192 || shape.GetShapeSize() <= 0) {
            return ge::GRAPH_FAILED;
        }
        const int64_t rows = shape.GetShapeSize() / dim;
        if (rows > 1048576) {
            return ge::GRAPH_FAILED;
        }
        uint32_t DT_INPUT_X = static_cast<uint32_t>(ge::DT_FLOAT16);
        const uint32_t vectorCores = platform.GetCoreNumAiv();
        uint32_t vectorCols = 16;
        uint32_t bestWaves = (rows * ((dim + 15) / 16) + vectorCores - 1) / vectorCores;
        const uint32_t maxVectorCols = dim <= 64 ? ((dim + 15) / 16) * 16 : 32;
        for (uint32_t c = 32; c <= maxVectorCols; c += 16) {
            uint32_t waves = (rows * ((dim + c - 1) / c) + vectorCores - 1) / vectorCores;
            if (waves < bestWaves) { vectorCols = c; bestWaves = waves; }
        }
        const bool vectorSmall = (dim <= 64 && rows * dim * dim <= (dim <= 16 ? 65536 : 32768)) ||
            (dim > 64 && dim <= 256 && rows * ((dim + vectorCols - 1) / vectorCols) <= vectorCores);
        uint32_t CALC_MODE = dim == 1 ? 1 : (vectorSmall ? 2 : 0);
        // Reuse each weight tile across rows once the saved transfers outweigh
        // the additional vector work per core.
        const uint32_t minBatchRows = dim == 16 ? 7 :
            (dim % 16 == 0 ? 4 : (dim < 16 ? 12 : (dim < 32 ? 8 : 4)));
        if (CALC_MODE == 2 && dim > 3 && dim <= 128 && rows >= minBatchRows) {
            CALC_MODE = 6;
        }
        if (CALC_MODE == 2 && dim > 128 && dim <= 256 && rows >= 4 &&
            (rows % 2 == 0 || (dim % 16 != 0 && dim < 193))) {
            CALC_MODE = 7;
        }
        if (CALC_MODE == 6 && dim % 16 != 0 &&
            std::min<int64_t>(dim, 16) * ((dim + 15) / 16 * 16) > SWIGLU_GATHER_CAPACITY) {
            CALC_MODE = 0;
        }
        uint32_t DIM_TAG = CALC_MODE != 2 ? 0 :
            (dim == 3 ? 1 : (dim == 13 ? 2 : (dim == 64 && vectorCols == 16 ? 3 : 0)));
        uint32_t TAIL_CODE = 0, PAD_BLOCKS = 0, ROW_WAVES = 0;
        if (rows == 1) {
            if (CALC_MODE == 1) DIM_TAG = 7;
            else if (DIM_TAG != 0) DIM_TAG += 3;
        }
        if (CALC_MODE == 2 && DIM_TAG == 3 &&
            rows * ((dim + vectorCols - 1) / vectorCols) > vectorCores) DIM_TAG = 0;
        if (CALC_MODE == 2 && DIM_TAG == 0 &&
            rows * ((dim + vectorCols - 1) / vectorCols) <= vectorCores) DIM_TAG = 7;
        if (CALC_MODE == 6) {
            if (dim > 64) DIM_TAG = dim % 16 != 0 ? 5 : 4;
            else if (dim % 16 != 0) {
                const uint32_t waves8 = ((rows + 7) / 8 + vectorCores - 1) / vectorCores;
                const uint32_t waves16 = ((rows + 15) / 16 + vectorCores - 1) / vectorCores;
                const bool useRows16 = dim >= 8 && dim < 16 && rows >= 256 &&
                    (rows <= 640 || waves16 * 16 <= waves8 * 8);
                DIM_TAG = useRows16 ? 6 : (rows >= 64 ? 3 : 2);
            }
            else if (dim == 16) DIM_TAG = rows >= 64 ? 9 : 8;
            else if (dim == 32) DIM_TAG = rows >= 15 ? 10 : 0;
            else if (dim == 48) DIM_TAG = 11;
            else if (dim == 64) DIM_TAG = rows >= 7 ? 12 : 0;
        }
        if (CALC_MODE == 7) DIM_TAG = dim % 16 != 0 ? 1 :
            (dim == 224 ? 13 : (dim == 256 ? 14 : 0));
        if (CALC_MODE == 2 && dim == 4 && rows * dim <= vectorCores) DIM_TAG = 24;
        else if (CALC_MODE == 2 && dim == 32 && rows == 1) DIM_TAG = 19;
        else if (CALC_MODE == 6 && dim == 13 && rows >= 12 && rows <= 63 &&
            (rows + 2) / 3 <= vectorCores) DIM_TAG = 23;
        // Share generic implementations when a static dimension does not change
        // the row grouping, memory layout, or accumulation order.
        if (CALC_MODE == 2) {
            if (DIM_TAG == 1 || DIM_TAG == 2) DIM_TAG = 0;
            else if (DIM_TAG == 6) DIM_TAG =
                rows * ((dim + vectorCols - 1) / vectorCols) <= vectorCores ? 3 : 0;
            else if (DIM_TAG >= 4 && DIM_TAG <= 6) DIM_TAG = 7;
        } else if (CALC_MODE == 6) {
            if (DIM_TAG == 8 || (DIM_TAG >= 10 && DIM_TAG <= 12)) DIM_TAG = 0;
            else if (DIM_TAG == 15 || DIM_TAG == 17 || DIM_TAG == 18) DIM_TAG = 4;
        } else if (CALC_MODE == 7 && (DIM_TAG == 13 || DIM_TAG == 14)) {
            DIM_TAG = 0;
        }
        if (CALC_MODE == 6 && DIM_TAG == 23) {
            if (rows % 3 == 0) DIM_TAG = 15;
            else if (rows <= vectorCores) DIM_TAG = 12;
        }
        // One wave of three-row groups with a one-row tail fits the wide-vector UB layout.
        if (dim == 33 && rows >= 31 && rows % 3 == 1 && (rows + 2) / 3 <= vectorCores) {
            CALC_MODE = 6;
            DIM_TAG = 11;
        }
        if (dim == 33 && rows >= 31 && rows % 2 == 1 && (rows + 1) / 2 <= vectorCores) {
            CALC_MODE = 6;
            DIM_TAG = 13;
        }
        // Favor three rows per AIV when two-row groups nearly fill a wave.
        // This reduces replicated weight loads; the existing three-row tail
        // path requires one remaining row and handles it explicitly.
        if (dim == 33 && rows >= vectorCores * 3 / 2 && rows % 3 == 1 &&
            (rows + 2) / 3 <= vectorCores) {
            CALC_MODE = 6; DIM_TAG = 11;
        }
        // Keep one owner for the complete tiny row while launching a short AIV wave.
        if (CALC_MODE == 2 && DIM_TAG == 24 && rows == 1) DIM_TAG = 16;
        if (CALC_MODE == 6 && DIM_TAG == 15 && (rows / 3) % 8 != 0 &&
            ((rows / 3 + 7) / 8 * 8) <= vectorCores) {
            DIM_TAG = 15 + (rows / 3) % 8;
        }
        if (dim == 13 && rows >= 14 && rows <= 56 && rows % 7 == 0) {
            CALC_MODE = 6; DIM_TAG = 23 + rows / 7;
        }
        SwiGluTilingData *tiling = context->GetTilingData<SwiGluTilingData>();
        if (tiling == nullptr) return ge::GRAPH_FAILED;
        context->GetRawTilingData()->SetDataSize(sizeof(SwiGluVectorTilingData));
        tiling->rows = rows;
        tiling->dim = dim;
        // Reuse each unaligned weight tile across eight input rows. Round the
        // launch to an AIV wave; kernel bounds use the actual tiling row count.
        const uint32_t vectorRowTile = 8;
        const uint32_t vectorRowTiles = (rows + vectorRowTile - 1) / vectorRowTile;
        const uint32_t vectorLaunch = (vectorRowTiles + 7) / 8 * 8;
        if (dim == 33 && rows >= vectorRowTile * vectorRowTile && vectorLaunch <= vectorCores) {
            CALC_MODE=6; DIM_TAG=14;
            TAIL_CODE = rows % vectorRowTile;
            PAD_BLOCKS = vectorLaunch - vectorRowTiles;
            tiling->tileRows=vectorRowTile; tiling->tileCols=40; tiling->cores=vectorLaunch;
            ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X, CALC_MODE, DIM_TAG, TAIL_CODE, PAD_BLOCKS, ROW_WAVES);
            context->SetBlockDim(tiling->cores); context->GetWorkspaceSizes(1)[0]=32;
            return ge::GRAPH_SUCCESS;
        }
        if (CALC_MODE != 0) {
            ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X, CALC_MODE, DIM_TAG, TAIL_CODE, PAD_BLOCKS, ROW_WAVES);
            if (CALC_MODE == 6 && DIM_TAG >= 25 && DIM_TAG <= 31) {
                tiling->tileRows=7; tiling->tileCols=16; tiling->cores=8;
                context->SetBlockDim(8); context->GetWorkspaceSizes(1)[0]=32;
                return ge::GRAPH_SUCCESS;
            }
            if (CALC_MODE == 6) {
                tiling->tileRows = DIM_TAG == 13 ? 2 : (DIM_TAG == 12 ? 1 : (DIM_TAG == 11 || DIM_TAG == 15 || DIM_TAG == 23 || (DIM_TAG >= 16 && DIM_TAG <= 22) ? 3 :
                    (DIM_TAG == 7 ? 32 : (DIM_TAG == 6 ? 16 : (DIM_TAG == 3 || DIM_TAG == 9 ? 8 : 4)))));
                tiling->tileCols = DIM_TAG == 11 || DIM_TAG == 13 ? 40 : 16;
                const uint32_t tiles = ((rows + tiling->tileRows - 1) / tiling->tileRows) *
                    ((dim + tiling->tileCols - 1) / tiling->tileCols);
                tiling->cores = std::min<uint32_t>(tiles, vectorCores);
                if (DIM_TAG >= 16 && DIM_TAG <= 22) tiling->cores = (tiling->cores + 7) / 8 * 8;
                context->SetBlockDim(tiling->cores);
                context->GetWorkspaceSizes(1)[0] = 32;
                return ge::GRAPH_SUCCESS;
            }
            if (CALC_MODE == 7) {
                tiling->tileRows = 2;
                tiling->tileCols = 16;
                const uint32_t tiles = ((rows + 1) / 2) * ((dim + 15) / 16);
                tiling->cores = std::min<uint32_t>(tiles, vectorCores);
                context->SetBlockDim(tiling->cores);
                context->GetWorkspaceSizes(1)[0] = 32;
                return ge::GRAPH_SUCCESS;
            }
            tiling->tileCols = CALC_MODE == 2 && DIM_TAG == 19 ? 32 : vectorCols;
            tiling->tileRows = std::min<uint32_t>(16384,
                ((rows + vectorCores * 128 - 1) / (vectorCores * 128)) * 128);
            const uint32_t tiles = CALC_MODE == 1 ? (rows + tiling->tileRows - 1) / tiling->tileRows :
                rows * ((dim + vectorCols - 1) / vectorCols);
            tiling->cores = std::min<uint32_t>(tiles, platform.GetCoreNumAiv());
            if (DIM_TAG == 24) tiling->cores = rows;
            if ((CALC_MODE == 1 && DIM_TAG == 7) || (CALC_MODE == 2 && (DIM_TAG == 16 || DIM_TAG == 19))) {
                tiling->cores = std::min<uint32_t>(8, vectorCores);
            }
            context->SetBlockDim(tiling->cores);
            // Static small-vector paths consume no user workspace.
            const bool noWorkspace=(CALC_MODE==1 && DIM_TAG==7) ||
                (CALC_MODE==2 && (DIM_TAG==3 || DIM_TAG==16 || DIM_TAG==19 || DIM_TAG==24));
            context->GetWorkspaceSizes(1)[0] = noWorkspace ? 0 : 32;
            return ge::GRAPH_SUCCESS;
        }
        const uint32_t cubeCoresAvailable = platform.GetCoreNumAic();
        constexpr uint32_t projectionCols = 128;
        // Large aligned matrices use a full-row double-buffered pipeline once
        // at least four row waves amortize its setup. Tail rows remain dynamic.
        constexpr uint32_t fullRowCores = 20;
        const uint32_t fullRowWave = fullRowCores / 2 * 128;
        if (dim == 2 * projectionCols && cubeCoresAvailable >= fullRowCores && rows >= 4 * fullRowWave) {
            CALC_MODE=3; DIM_TAG=18;
            // Pad the final row wave to a half/full tile capacity. Both buckets
            // cover every remainder, including an empty final wave.
            TAIL_CODE = rows % fullRowWave <= fullRowWave / 2 ? 8 : 9;
            const uint32_t tailCapacity = TAIL_CODE == 8 ? 64 : 128;
            // Complete row tiles need no dynamic per-AIV boundary checks.
            if (rows % tailCapacity == 0) {
                TAIL_CODE += 2;
                // Statically schedule a bounded number of complete waves.
                // Compile every wave/tail combination in this resource class;
                // larger schedules and partial tiles keep runtime tiling.
                const uint32_t fullWaves = rows / fullRowWave;
                constexpr uint32_t maxStaticWaves = 8;
                if (fullWaves <= maxStaticWaves) {
                    ROW_WAVES = fullWaves;
                    PAD_BLOCKS = fullRowCores / 2 - rows % fullRowWave / tailCapacity;
                }
            }
            tiling->tileRows=128; tiling->tileCols=projectionCols; tiling->cores=fullRowCores;
            ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X, CALC_MODE, DIM_TAG, TAIL_CODE, PAD_BLOCKS, ROW_WAVES);
            context->SetBlockDim(tiling->cores);
            context->GetWorkspaceSizes(1)[0]=platform.GetLibApiWorkSpaceSize()+
                static_cast<size_t>(tiling->cores)*8*tiling->tileRows*projectionCols;
            return ge::GRAPH_SUCCESS;
        }
        // Two complete row waves fit the separate A0/C0 slots. Use this path
        // when at least half of the Cube cores can own one complete row tile.
        constexpr uint32_t rowWave = 128, rowTile = 2 * rowWave;
        const uint32_t completeRowTiles = rows / rowTile;
        if (dim == projectionCols && rows % rowTile == 0 &&
                completeRowTiles <= cubeCoresAvailable &&
                completeRowTiles >= (cubeCoresAvailable + 1) / 2) {
            CALC_MODE=3; DIM_TAG=19;
            tiling->tileRows=rowTile; tiling->tileCols=projectionCols; tiling->cores=completeRowTiles;
            ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X, CALC_MODE, DIM_TAG, TAIL_CODE, PAD_BLOCKS, ROW_WAVES);
            context->SetBlockDim(tiling->cores);
            context->GetWorkspaceSizes(1)[0]=platform.GetLibApiWorkSpaceSize()+
                static_cast<size_t>(tiling->cores)*2*rowTile*projectionCols*sizeof(uint16_t);
            return ge::GRAPH_SUCCESS;
        }
        // Complete 64-row tiles, two N128 projections per row tile, fit a
        // single Cube wave. Each paired AIV owns the complete row tile.
        constexpr uint32_t fullKRowTile = 64;
        const uint32_t fullKTiles = rows / fullKRowTile * 2;
        if (dim == 2 * projectionCols && rows % fullKRowTile == 0 &&
                fullKTiles >= 2 && fullKTiles <= cubeCoresAvailable) {
            CALC_MODE=3; DIM_TAG=21;
            tiling->tileRows=fullKRowTile; tiling->tileCols=projectionCols; tiling->cores=fullKTiles;
            ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X, CALC_MODE, DIM_TAG, TAIL_CODE, PAD_BLOCKS, ROW_WAVES);
            context->SetBlockDim(tiling->cores);
            context->GetWorkspaceSizes(1)[0]=platform.GetLibApiWorkSpaceSize()+
                static_cast<size_t>(tiling->cores)*fullKRowTile*projectionCols*4;
            return ge::GRAPH_SUCCESS;
        }
        if(dim==256 && rows>=384 && rows<=platform.GetCoreNumAic()/2*64) {
            CALC_MODE=3; DIM_TAG=22;
            tiling->tileRows=64; tiling->tileCols=128; tiling->cores=(rows+63)/64*2;
            ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X, CALC_MODE, DIM_TAG, TAIL_CODE, PAD_BLOCKS, ROW_WAVES);
            context->SetBlockDim(tiling->cores);
            context->GetWorkspaceSizes(1)[0]=platform.GetLibApiWorkSpaceSize()+
                static_cast<size_t>(tiling->cores)*64*128*4;
            return ge::GRAPH_SUCCESS;
        }
        if(dim==128 && rows>platform.GetCoreNumAic()*128 && rows<=platform.GetCoreNumAic()*256) {
            CALC_MODE=3; DIM_TAG=23;
            tiling->tileRows=128; tiling->tileCols=128; tiling->cores=platform.GetCoreNumAic();
            ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X, CALC_MODE, DIM_TAG, TAIL_CODE, PAD_BLOCKS, ROW_WAVES);
            context->SetBlockDim(tiling->cores);
            context->GetWorkspaceSizes(1)[0]=platform.GetLibApiWorkSpaceSize()+
                static_cast<size_t>(tiling->cores)*4*128*128*sizeof(uint16_t);
            return ge::GRAPH_SUCCESS;
        }
        uint32_t cubeCols = 128;
        if (dim > 128 && dim < 256) {
            const bool unaligned = dim % 16 != 0 && rows <= 384 &&
                !(dim >= 208 && rows >= 48 && rows <= 80);
            const bool shortAligned = dim % 16 == 0 && rows <= 16;
            const bool narrowAligned = dim % 16 == 0 && dim <= 176 && rows <= 64;
            const bool balancedAligned = dim % 16 == 0 && rows == 128;
            const bool wideAligned = dim % 16 == 0 && dim >= 208 &&
                rows >= 96 && rows <= 384;
            if (unaligned || shortAligned || narrowAligned || balancedAligned || wideAligned) {
                cubeCols = 64;
            }
        }
        if ((dim >= 129 && dim <= 144 && rows >= 385 && rows <= 2048) ||
            (dim >= 145 && dim <= 160 && rows >= 385 && rows <= 768) ||
            (dim == 177 && rows >= 385 && rows <= 2048)) cubeCols = 96;
        if (dim == 256 && ((rows >= 81 && rows <= 96) ||
            (rows >= 128 && rows <= 160) || (rows >= 241 && rows <= 256))) {
            cubeCols = rows == 128 ? 96 : 64;
        }
        if (dim >= 257 && dim <= 320 && (rows <= 16 || rows >= 512)) cubeCols = 64;
        if (dim >= 321 && dim <= 511 && rows <= 16) cubeCols = 64;
        if (dim == 384 && rows <= 16) cubeCols = 96;
        if ((dim >= 321 && dim <= 383 && rows >= 49 && rows <= 384) ||
            (dim == 384 && rows >= 49 && rows <= 96) ||
            (dim >= 385 && dim <= 503 && rows >= 49 && rows <= 256) ||
            (dim >= 504 && dim <= 511 && rows >= 49 && rows <= 224)) cubeCols = 64;
        if ((dim >= 321 && dim <= 352 && rows >= 512 && rows <= 8192) ||
            (dim >= 385 && dim <= 448 && rows >= 384 && rows <= 2048) ||
            (dim == 480 && rows >= 384 && rows <= 4096)) cubeCols = 96;
        if (dim == 512 && ((rows >= 49 && rows <= 80) ||
            (rows >= 97 && rows <= 256))) cubeCols = rows == 64 ? 96 : 64;
        if (dim >= 513 && dim <= 767 && rows <= 256) {
            const bool unstable576 = dim == 576 &&
                ((rows >= 33 && rows <= 48) || (rows >= 65 && rows <= 96));
            const bool unstable640 = dim == 640 && rows <= 32;
            const bool unstable641 = dim == 641 && rows >= 225;
            if (!unstable576 && !unstable640 && !unstable641) cubeCols = 96;
        }
        if ((dim >= 769 && dim <= 832 && rows <= 256) ||
            (((dim >= 833 && dim <= 895) || (dim >= 897 && dim <= 943)) &&
             (rows <= 16 || (rows >= 33 && rows <= 192))) ||
            (dim >= 944 && dim <= 959 &&
             (rows <= 16 || (rows >= 33 && rows <= 128)))) cubeCols = 96;
        if (dim >= 6144 && rows <= 256) cubeCols = 64;
        if (dim == 224 && rows >= 1792) cubeCols = 112;
        if ((dim == 159 && rows >= 385 && rows <= 768) ||
            (dim == 160 && rows >= 385)) cubeCols = 80;
        if (dim == 33 && rows >= 161 && rows <= 896) cubeCols = 32;
        if (dim >= 34 && dim <= 47 &&
            ((rows >= 192 && rows <= 320) ||
             (rows >= 512 && rows <= 640) ||
             (dim >= 43 && rows >= 768 && rows <= 896))) cubeCols = 32;
        if (dim >= 49 && dim <= 63 &&
            (rows <= 23 || (rows >= 192 && rows <= 2048))) cubeCols = 32;
        if (dim >= 65 && dim <= 127 && dim % 16 != 0 &&
            (rows <= 32 || (dim >= 81 && rows <= 48) ||
             (dim >= 95 && rows <= 72) || (dim >= 103 && rows <= 96) ||
             (dim >= 111 && rows <= 112) ||
             (rows >= 192 && rows <= 2048))) cubeCols = 64;
        tiling->tileCols = std::min<uint32_t>(cubeCols, ((dim + 15) / 16) * 16);
        const uint32_t columnTiles = (dim + tiling->tileCols - 1) / tiling->tileCols;
        const uint32_t rowCores = std::max<uint32_t>(1, platform.GetCoreNumAic() / columnTiles);
        tiling->tileRows = std::min<uint32_t>(128, ((rows + rowCores * 16 - 1) / (rowCores * 16)) * 16);
        // Use the available cores in the last wave by balancing its row tiles.
        // Keep the wave count unchanged to avoid extra per-tile overhead.
        const uint32_t cubeCores = platform.GetCoreNumAic();
        const uint32_t initialTiles = ((rows + tiling->tileRows - 1) / tiling->tileRows) * columnTiles;
        const uint32_t waves = (initialTiles + cubeCores - 1) / cubeCores;
        const uint32_t rowTileCapacity = waves * cubeCores / columnTiles;
        const uint32_t balancedRows = ((rows + rowTileCapacity * 16 - 1) / (rowTileCapacity * 16)) * 16;
        // A very short final row tile can already balance the critical core.
        // Compare actual padded rows, including that tail, before shrinking.
        const auto criticalRows = [&](uint32_t tileRows) {
            const uint32_t rowTiles = (rows + tileRows - 1) / tileRows;
            const uint32_t fullTiles = (rowTiles - 1) * columnTiles;
            const uint32_t tailRows = ((rows - (rowTiles - 1) * tileRows + 15) / 16) * 16;
            const uint32_t fullRemainder = fullTiles % cubeCores;
            const uint32_t tailRemainder = columnTiles % cubeCores;
            const uint32_t extra = fullRemainder + tailRemainder > cubeCores ?
                tileRows + tailRows : std::max(fullRemainder ? tileRows : 0,
                                               tailRemainder ? tailRows : 0);
            return (fullTiles / cubeCores) * tileRows + (columnTiles / cubeCores) * tailRows + extra;
        };
        if (balancedRows < tiling->tileRows && criticalRows(balancedRows) < criticalRows(tiling->tileRows)) {
            tiling->tileRows = balancedRows;
        }
        // A 64-row single wave loads fewer redundant weight stripes than
        // the smallest full-occupancy tile, while still fitting all cores.
        if (dim == 256 && tiling->tileCols == 128 && rows >= 384 &&
            rows <= cubeCores / 2 * 64) tiling->tileRows = 64;
        const uint32_t tiles = ((rows + tiling->tileRows - 1) / tiling->tileRows) * columnTiles;
        tiling->cores = std::min<uint32_t>(tiles, platform.GetCoreNumAic());

        // K-step pipelining requires full 128-wide K chunks. The prefetch-only
        // path keeps the original 128-step accumulation order and handles a
        // partial final K chunk.
        if (tiling->tileRows >= 32 && dim % 128 == 0) CALC_MODE = 3;
        if (dim > 512 && (dim % 128 == 0 ||
            (rows >= 384 && (dim >= 575 || rows >= 512)))) {
            // D=640 with 96 columns has a measured-safe 64-column tail and
            // benefits substantially from keeping the L0 pipeline enabled.
            const bool fullNTiles = dim % tiling->tileCols == 0 ||
                (dim == 640 && tiling->tileCols == 96);
            CALC_MODE = dim % 128 == 0 && tiling->tileRows >= 32 && fullNTiles ? 4 : 5;
        }
        // A core that owns one tile never needs to reuse its projection buffer.
        if (DIM_TAG == 0 && tiles <= platform.GetCoreNumAic() && CALC_MODE < 4) {
            DIM_TAG = 25;
        }
        if (CALC_MODE < 4 && tiling->tileCols == 128 && (dim == 128 || dim == 256)) {
            DIM_TAG = (dim == 128 ? 26 : 27) + (tiles <= platform.GetCoreNumAic() ? 2 : 0);
        }
        if (dim == 33 && tiling->tileCols == 48 && tiles <= platform.GetCoreNumAic()) DIM_TAG = 30;
        if (DIM_TAG == 30 && tiling->tileRows == 16) DIM_TAG = 31;
        if (CALC_MODE == 3 && dim == 256 && tiling->tileCols == 128 &&
            tiling->tileRows == 128 && rows % 128 == 0 && tiles > tiling->cores &&
            tiling->cores % columnTiles == 0) DIM_TAG = 14;
        if (DIM_TAG == 14 && rows % (tiling->cores / columnTiles * tiling->tileRows) != 0) DIM_TAG = 17;
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X, CALC_MODE, DIM_TAG, TAIL_CODE, PAD_BLOCKS, ROW_WAVES);
        context->SetBlockDim(tiling->cores);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = platform.GetLibApiWorkSpaceSize() + static_cast<size_t>(tiling->cores) * tiling->tileRows * tiling->tileCols *
            ((DIM_TAG == 14 || DIM_TAG == 17 || DIM_TAG == 26 || DIM_TAG == 27) ? 8 : 4);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        *context->GetOutputShape(0) = *context->GetInputShape(0);
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class SwiGlu : public OpDef {
    public:
        explicit SwiGlu(const char *name) : OpDef(name) {
            this->Input("input_x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND});
            this->Input("weight_1")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND});
            this->Input("bias_1")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND});
            this->Input("weight_2")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND});
            this->Input("bias_2")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND});
            this->Output("output")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(SwiGlu);
}  // namespace ops
