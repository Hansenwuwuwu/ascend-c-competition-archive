#pragma once
// Inlined local header: include/elec/template_host.h
#ifndef CANNJUDGE_INLINE_530D7EE08E66F759A4063EDF
#define CANNJUDGE_INLINE_530D7EE08E66F759A4063EDF

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
// Inlined local header: include/elec/template_contract.h
#ifndef CANNJUDGE_INLINE_5E30E70DE78CAE8F8FF1D946
#define CANNJUDGE_INLINE_5E30E70DE78CAE8F8FF1D946

// Inlined local header: include/elec/metadata.h
#ifndef CANNJUDGE_INLINE_7AB04A479FE30651BCB07257
#define CANNJUDGE_INLINE_7AB04A479FE30651BCB07257

// Inlined local header: include/elec/config.h
#ifndef CANNJUDGE_INLINE_690E6E24C5AB526862289F78
#define CANNJUDGE_INLINE_690E6E24C5AB526862289F78

#include <cstdint>

namespace elec {
enum DType : uint32_t { F32 = 0, F16 = 1, BF16 = 2, I32 = 3, I64 = 4 };
enum Op : uint32_t { SAFETY = 0, FIRE, TREE, SWIGLU, SPLINE, DET, SOLVE, LSTSQ, EIGH, PINV };
enum Flags : uint32_t { ROOT = 1, BIAS = 2, MEAN = 4, UPPER = 8, RTOL_SET = 16 };
// Fixed-width, naturally aligned ABI shared by Python, host and device.
// Shapes are flattened to a batch count; all arrays are contiguous ND.
struct Config {
    uint64_t batch = 1, m = 1, n = 1, k = 1, edges = 0, dim = 2;
    uint64_t rhs_batch_stride = 0;
    uint32_t dtype = F32, flags = 0, degree = 1;
    uint32_t kernel_size[3] = {2, 2, 2};
    uint32_t open_spline[3] = {1, 1, 1};
    uint32_t max_sweeps = 100;
    float waist_ratio = 1.0f, threshold = 0.0f;
    float atol = 0.0f, rtol = 0.0f, tolerance = 1.0e-7f;
};
enum Status : int32_t { SUCCESS = 0, SINGULAR = 1, NO_CONVERGENCE = 2, INVALID = 3, NUMERIC_ERROR = 4 };
}

#endif // CANNJUDGE_INLINE_690E6E24C5AB526862289F78

#include <array>
#include <cstddef>
#include <stdexcept>
#include <cmath>
namespace elec {
struct Sizes { std::array<size_t,6> input{}; std::array<size_t,2> output{}; size_t workspace = 0; };
inline void require(bool ok, const char* message) { if (!ok) throw std::invalid_argument(message); }
inline Sizes sizes(uint32_t op, const Config& c) {
    Sizes s;
    require(op <= PINV, "unknown operator");
    require(c.dtype <= BF16, "invalid floating dtype");
    require(c.batch >= 1 && c.batch <= 4096, "batch out of range");
    require(c.m >= 1 && c.m <= 1048576 && c.n >= 1 && c.n <= 32768 && c.k >= 1 && c.k <= 256, "dimensions out of range");
    require(std::isfinite(c.atol) && c.atol >= 0 && std::isfinite(c.rtol) && c.rtol >= 0, "invalid tolerances");
    require(c.max_sweeps >= 1 && c.max_sweeps <= 10000 && c.tolerance > 0 && std::isfinite(c.tolerance), "invalid iteration controls");
    const size_t z = c.dtype == F32 ? 4 : 2, b=c.batch, m=c.m,n=c.n,k=c.k;
    if (op <= TREE) require(c.dtype == F16 || c.dtype == F32, "operator supports float16/float32 only");
    if (op >= DET) require(c.dtype == F32, "linear algebra operators require float32");
    switch (op) {
    case SAFETY:
        require(c.waist_ratio > 0 && c.waist_ratio <= 1 && std::isfinite(c.threshold) && c.threshold >= 0, "invalid safety attributes");
        s.input={b*4*z,b*2*z,4,0,0,0}; s.output={b*4,0}; break;
    case FIRE:
        require(b<=8 && n<=32768,"fire dimensions out of range");
        s.input={b*n*6*z,n*5*4,0,0,0,0}; s.output={b*n*5*z,0}; break;
    case TREE:
        require(b<=8 && m<=16384 && n<=1024,"tree dimensions out of range");
        s.input={b*m*3*z,b*n*6*z,0,0,0,0}; s.output={b*m*z,b*m*4}; break;
    case SWIGLU:
        require(c.dtype==F16 && b==1 && n<=8192,"invalid SwiGLU configuration");
        s.input={m*n*z,n*n*z,n*z,n*n*z,n*z,0}; s.output={m*n*z,0}; break;
    case SPLINE: {
        require((c.dtype==F32||c.dtype==BF16) && b==1 && m>=16 && m<=65536 && n>=8 && n<=128 && k>=8 && k<=128,"invalid SplineConv shape/type");
        require(c.edges>=64 && c.edges<=262144 && (c.dim==2||c.dim==3) && c.degree>=1 && c.degree<=3,"invalid SplineConv attributes");
        size_t kernels=1, slots=1;
        for(uint32_t d=0; d<c.dim; ++d) {
            require(c.kernel_size[d]>=2 && c.kernel_size[d]<=4 && c.open_spline[d]<=1,"invalid spline kernel size");
            require(!c.open_spline[d] || c.kernel_size[d]>=c.degree+1,"open spline kernel too small");
            kernels*=c.kernel_size[d]; slots*=c.degree+1;
        }
        require(c.edges*slots*n*k <= (uint64_t(1)<<31),"SplineConv workload exceeds specification");
        s.input={m*n*z,2*c.edges*8,c.edges*c.dim*z,kernels*n*k*z,(c.flags&ROOT)?n*k*z:0,(c.flags&BIAS)?k*z:0};
        s.output={m*k*z,0}; s.workspace=(2*m*k+m)*4; break;
    }
    case DET:
        require(b<=256&&n<=512,"det dimensions out of range");
        s.input[0]=b*n*n*4; s.output[0]=b*4; s.workspace=(n*n+n)*4; break;
    case SOLVE:
        require(b<=64&&n<=512,"solve dimensions out of range");
        require(c.rhs_batch_stride==0||c.rhs_batch_stride==n*k,"invalid RHS stride");
        s.input[0]=b*n*n*4; s.input[1]=(c.rhs_batch_stride?b:1)*n*k*4;
        s.output[0]=b*n*k*4; s.workspace=(n*n+n+2*n*k)*4; break;
    case LSTSQ:
        require(b<=32&&m<=1024&&n<=m&&n<=512&&k<=64,"lstsq dimensions out of range");
        require(c.rhs_batch_stride==0||c.rhs_batch_stride==m*k,"invalid RHS stride");
        s.input[0]=b*m*n*4; s.input[1]=(c.rhs_batch_stride?b:1)*m*k*4;
        s.output={b*n*k*4,m>n?b*k*4:0}; s.workspace=(m*n+2*m*k+m)*4; break;
    case EIGH:
        require(b<=64&&n<=512,"eigh dimensions out of range");
        s.input[0]=b*n*n*4; s.output={b*n*4,b*n*n*4}; s.workspace=2*n*n*4; break;
    case PINV: {
        require(b<=64&&m<=512&&n<=512,"pinv dimensions out of range");
        size_t q=m<n?m:n;
        s.input[0]=b*m*n*4; s.output[0]=b*n*m*4; s.workspace=(m*n+q*q+q)*4; break;
    }
    }
    return s;
}
}

#endif // CANNJUDGE_INLINE_7AB04A479FE30651BCB07257

// Inlined local header: include/elec/template_tiling.h
#ifndef CANNJUDGE_INLINE_728A063E3555C4F12FC632AC
#define CANNJUDGE_INLINE_728A063E3555C4F12FC632AC

// Inlined local header: include/elec/config.h
#ifndef CANNJUDGE_INLINE_690E6E24C5AB526862289F78
#define CANNJUDGE_INLINE_690E6E24C5AB526862289F78

#include <cstdint>

namespace elec {
enum DType : uint32_t { F32 = 0, F16 = 1, BF16 = 2, I32 = 3, I64 = 4 };
enum Op : uint32_t { SAFETY = 0, FIRE, TREE, SWIGLU, SPLINE, DET, SOLVE, LSTSQ, EIGH, PINV };
enum Flags : uint32_t { ROOT = 1, BIAS = 2, MEAN = 4, UPPER = 8, RTOL_SET = 16 };
// Fixed-width, naturally aligned ABI shared by Python, host and device.
// Shapes are flattened to a batch count; all arrays are contiguous ND.
struct Config {
    uint64_t batch = 1, m = 1, n = 1, k = 1, edges = 0, dim = 2;
    uint64_t rhs_batch_stride = 0;
    uint32_t dtype = F32, flags = 0, degree = 1;
    uint32_t kernel_size[3] = {2, 2, 2};
    uint32_t open_spline[3] = {1, 1, 1};
    uint32_t max_sweeps = 100;
    float waist_ratio = 1.0f, threshold = 0.0f;
    float atol = 0.0f, rtol = 0.0f, tolerance = 1.0e-7f;
};
enum Status : int32_t { SUCCESS = 0, SINGULAR = 1, NO_CONVERGENCE = 2, INVALID = 3, NUMERIC_ERROR = 4 };
}

#endif // CANNJUDGE_INLINE_690E6E24C5AB526862289F78


namespace elec {
// Plain fixed-width ABI copied by GET_TILING_DATA_WITH_STRUCT on the device.
struct TemplateTilingData {
    Config config;
    uint64_t scratch_bytes = 0;
    uint32_t input_types[6] = {};
    uint32_t output_types[2] = {};
};
}

#endif // CANNJUDGE_INLINE_728A063E3555C4F12FC632AC

#include <vector>
#include <initializer_list>
#include <limits>

namespace elec::template_contract {
using Shape = std::vector<int64_t>;
struct Plan {
    TemplateTilingData tiling;
    std::vector<Shape> outputs;
    size_t workspace_bytes = 0;
};
inline uint64_t product(const Shape& s, size_t end) {
    uint64_t p = 1;
    for (size_t i=0; i<end; ++i) {
        require(s[i]>0 && p <= (uint64_t(1)<<40)/uint64_t(s[i]), "invalid or oversized shape");
        p *= uint64_t(s[i]);
    }
    return p;
}
inline void exact(const Shape& actual, const Shape& expected) {
    require(actual==expected, "input shape does not match the operator template");
}
inline bool empty(const Shape& shape) {
    bool zero=false;
    for (const auto d:shape) {
        if (d<0) return false;
        zero=zero||d==0;
    }
    return zero;
}
inline Shape append(Shape s, std::initializer_list<int64_t> tail) {
    s.insert(s.end(),tail); return s;
}
inline Plan make(uint32_t op, const std::vector<Shape>& inputs, uint32_t type, Config c = {}, bool matrix_rhs = false) {
    require(!inputs.empty() && type<=BF16, "invalid template inputs");
    const auto& a=inputs[0];
    product(a,a.size());
    Plan p;
    c.dtype = op>=DET ? uint32_t(F32) : type;
    for (auto& t:p.tiling.input_types) t=type;
    for (auto& t:p.tiling.output_types) t=type;
    const auto count = [&](size_t n) { require(inputs.size()==n,"wrong number of template inputs"); };
    if(op==SAFETY) {
        require(inputs.size()==2||inputs.size()==3,"safety expects two or three inputs");
        require(a.size()==2&&a[1]==4,"person_boxes must be [B,4]");
        c.batch=a[0]; exact(inputs[1],{a[0],2});
        if(inputs.size()==3) exact(inputs[2],{1});
        p.outputs={{a[0],1}};
    } else if(op==FIRE) {
        count(2); require(a.size()==3&&a[2]==6,"raw_pred must be [B,N,6]");
        c.batch=a[0]; c.n=a[1]; exact(inputs[1],{a[1],5});
        p.outputs={{a[0],a[1],5}};
    } else if(op==TREE) {
        count(2); require(a.size()==3&&a[2]==3,"tree_points must be [B,N,3]");
        const auto& s=inputs[1];
        require(s.size()==4&&s[0]==a[0]&&s[2]==2&&s[3]==3,"line_segments must be [B,M,2,3]");
        c.batch=a[0]; c.m=a[1]; c.n=s[1];
        p.outputs={{a[0],a[1]},{a[0],a[1]}}; p.tiling.output_types[1]=I32;
    } else if(op==SWIGLU) {
        count(5); require(!a.empty(),"input_x needs a feature dimension");
        c.m=product(a,a.size()-1); c.n=a.back();
        exact(inputs[1],{a.back(),a.back()}); exact(inputs[3],inputs[1]);
        exact(inputs[2],{a.back()}); exact(inputs[4],inputs[2]); p.outputs={a};
    } else if(op==SPLINE) {
        count(4); require(type==F32,"SplineConv template accepts float32");
        require(a.size()==2&&inputs[1].size()==2&&inputs[1][0]==2,"invalid x or edge_index");
        c.m=a[0]; c.n=a[1]; c.edges=inputs[1][1];
        exact(inputs[2],{int64_t(c.edges),int64_t(c.dim)});
        uint64_t kernels=1;
        require(c.dim==2||c.dim==3,"dim must be 2 or 3");
        for(uint32_t i=0;i<c.dim;++i) kernels*=c.kernel_size[i];
        const auto& w=inputs[3];
        require(w.size()==3&&w[0]==int64_t(kernels)&&w[1]==a[1],"weight must be [K,Cin,Cout]");
        c.k=w[2]; c.flags&=~uint32_t(ROOT|BIAS); p.outputs={{a[0],w[2]}};
        p.tiling.input_types[1]=I64;
    } else {
        require(a.size()>=2,"A must have at least two dimensions");
        Shape batch(a.begin(),a.end()-2);
        c.batch=product(batch,batch.size()); c.m=a[a.size()-2]; c.n=a.back();
        if(op==DET||op==SOLVE||op==EIGH) require(c.m==c.n,"A must be square");
        if(op==SOLVE||op==LSTSQ) {
            count(2); const auto& b=inputs[1];
            const Shape vector_shape=append(batch,{int64_t(c.m)});
            const bool vector=!matrix_rhs&&(b==Shape{int64_t(c.m)}||b==vector_shape);
            if(vector) c.k=1;
            else { require(b.size()>=2,"B must be a matrix"); c.k=b.back(); }
            Shape tail=vector?Shape{int64_t(c.m)}:Shape{int64_t(c.m),int64_t(c.k)};
            if(b==tail) c.rhs_batch_stride=0;
            else { exact(b,append(batch, vector?std::initializer_list<int64_t>{int64_t(c.m)}:
                           std::initializer_list<int64_t>{int64_t(c.m),int64_t(c.k)}));
                   c.rhs_batch_stride=c.m*c.k; }
            p.outputs={vector?append(batch,{int64_t(c.n)}):append(batch,{int64_t(c.n),int64_t(c.k)})};
            // The residual keeps its RHS axis even when B is a vector. Only
            // the solution removes that singleton dimension.
            if(op==LSTSQ) p.outputs.push_back(c.m>c.n?append(batch,{int64_t(c.k)}):Shape{0});
        } else {
            count(1);
            if(op==DET) { require(type==F32,"LinalgDet template accepts float32"); p.outputs={batch}; }
            else if(op==EIGH) p.outputs={append(batch,{int64_t(c.n)}),a};
            else if(op==PINV) p.outputs={append(batch,{int64_t(c.n),int64_t(c.m)})};
            else require(false,"unknown template operator");
        }
        require(type==F32||type==BF16,"linear algebra template accepts float32/bfloat16");
    }
    const auto capacities=sizes(op,c); // Reuse limits and exact scratch capacity of the compute baseline.
    p.tiling.config=c;
    p.tiling.scratch_bytes=capacities.workspace;
    p.workspace_bytes=capacities.workspace;
    if(op==LSTSQ&&c.m>c.n) p.workspace_bytes+=c.batch*c.k*(type==F32?4:2);
    // GetUserWorkspace is required even for algorithms with zero scratch.
    p.workspace_bytes=(p.workspace_bytes+63)/64*64;
    if(!p.workspace_bytes) p.workspace_bytes=64;
    return p;
}
}

#endif // CANNJUDGE_INLINE_5E30E70DE78CAE8F8FF1D946

#include <cstring>

namespace elec::template_host {
inline uint32_t dtype(ge::DataType t) {
    if(t==ge::DT_FLOAT) return F32;
    if(t==ge::DT_FLOAT16) return F16;
    if(t==ge::DT_BF16) return BF16;
    if(t==ge::DT_INT64) return I64;
    throw std::invalid_argument("unsupported template dtype");
}
inline uint32_t input_count(uint32_t op) {
    if(op==SWIGLU) return 5;
    if(op==SPLINE) return 4;
    return op==SAFETY||op==FIRE||op==TREE||op==SOLVE||op==LSTSQ?2:1;
}
inline Config attributes(uint32_t op, const gert::RuntimeAttrs* attrs) {
    Config c;
    if(op==SAFETY) {
        require(attrs && attrs->GetFloat(0),"waist_ratio is required");
        c.waist_ratio=*attrs->GetFloat(0);
    } else if(op==EIGH) {
        const char* u=attrs?attrs->GetStr(0):nullptr;
        require(!u||std::strcmp(u,"L")==0||std::strcmp(u,"l")==0||std::strcmp(u,"U")==0||std::strcmp(u,"u")==0,"UPLO must be L or U");
        if(u&&(u[0]=='U'||u[0]=='u')) c.flags|=UPPER;
    } else if(op==SOLVE) {
        require(!attrs||!attrs->GetBool(0)||*attrs->GetBool(0),"only left=True is supported");
    } else if(op==LSTSQ) {
        const char* driver=attrs?attrs->GetStr(0):nullptr;
        require(!driver||std::strcmp(driver,"gels")==0,"only driver=gels is supported");
        // rcond is deliberately unused by full-rank gels, as specified in the problem.
    } else if(op==PINV) {
        if(attrs&&attrs->GetFloat(0)) c.atol=*attrs->GetFloat(0);
        if(attrs&&attrs->GetFloat(1)&&*attrs->GetFloat(1)!=-1.0f) {
            c.rtol=*attrs->GetFloat(1); c.flags|=RTOL_SET;
        }
        require(!attrs||!attrs->GetBool(2)||!*attrs->GetBool(2),"only hermitian=False is supported");
    } else if(op==SPLINE) {
        if(attrs&&attrs->GetInt(0)) {
            const int64_t d=*attrs->GetInt(0);
            require(d==2||d==3,"dim must be 2 or 3"); c.dim=d;
        }
        if(attrs&&attrs->GetInt(3)) {
            const int64_t d=*attrs->GetInt(3);
            require(d>=1&&d<=3,"degree must be 1, 2 or 3"); c.degree=d;
        }
        if(attrs) for(uint32_t slot=1;slot<=2;++slot) {
            const auto* list=attrs->GetListInt(slot);
            require(list&&list->GetSize()==c.dim,"spline attribute list length mismatch");
            const int64_t* data=list->GetData();
            for(uint32_t d=0;d<c.dim;++d) {
                require(slot==1?(data[d]>=2&&data[d]<=4):(data[d]==0||data[d]==1),"invalid spline attribute");
                if(slot==1) c.kernel_size[d]=data[d]; else c.open_spline[d]=data[d];
            }
        }
        const char* aggr=attrs?attrs->GetStr(4):nullptr;
        require(!aggr||std::strcmp(aggr,"add")==0||std::strcmp(aggr,"mean")==0,"aggr must be add or mean");
        if(!aggr||std::strcmp(aggr,"mean")==0) c.flags|=MEAN;
    }
    return c;
}
inline template_contract::Shape shape(const gert::Shape* s) {
    require(s,"missing shape"); template_contract::Shape result;
    for(size_t i=0;i<s->GetDimNum();++i) result.push_back(s->GetDim(i));
    return result;
}
template<uint32_t OpId>
ge::graphStatus infer_shape(gert::InferShapeContext* context) {
    try {
        std::vector<template_contract::Shape> inputs;
        for(uint32_t i=0;i<input_count(OpId);++i) inputs.push_back(shape(context->GetInputShape(i)));
        if(OpId==SAFETY&&context->GetOptionalInputShape(2)) inputs.push_back(shape(context->GetOptionalInputShape(2)));
        // Shape inference precedes dtype inference. Choose a supported representative dtype.
        const uint32_t type=OpId==SWIGLU?F16:F32;
        const auto plan=template_contract::make(OpId,inputs,type,attributes(OpId,context->GetAttrs()));
        for(size_t i=0;i<plan.outputs.size();++i) {
            auto* out=context->GetOutputShape(i);
            if(!out) { require(OpId==LSTSQ&&i==1,"missing output shape"); continue; }
            out->SetDimNum(plan.outputs[i].size());
            for(size_t d=0;d<plan.outputs[i].size();++d) out->SetDim(d,plan.outputs[i][d]);
        }
        return ge::GRAPH_SUCCESS;
    } catch(const std::exception&) { return ge::GRAPH_FAILED; }
}
template<uint32_t OpId>
ge::graphStatus infer_type(gert::InferDataTypeContext* context) {
    const auto t=context->GetInputDataType(0);
    if(context->SetOutputDataType(0,t)!=ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if(OpId==TREE) return context->SetOutputDataType(1,ge::DT_INT32);
    if(OpId==EIGH) return context->SetOutputDataType(1,t);
    if(OpId==LSTSQ && context->GetComputeNodeInfo()->GetOutputsNum()>1)
        return context->SetOutputDataType(1,t);
    return ge::GRAPH_SUCCESS;
}
template<uint32_t OpId,typename Tiling,typename Select>
ge::graphStatus tile(gert::TilingContext* context, Select select) {
    try {
        std::vector<template_contract::Shape> inputs;
        const auto* first=context->GetRequiredInputDesc(0);
        require(first,"missing input descriptor");
        const uint32_t type=dtype(first->GetDataType());
        for(uint32_t i=0;i<input_count(OpId);++i) {
            const auto* s=context->GetRequiredInputShape(i);
            const auto* desc=context->GetRequiredInputDesc(i);
            require(s&&desc,"missing required input");
            require(dtype(desc->GetDataType())==(OpId==SPLINE&&i==1?uint32_t(I64):type),"template input dtype mismatch");
            inputs.push_back(shape(&s->GetStorageShape()));
        }
        if(OpId==SAFETY) {
            const auto* s=context->GetOptionalInputShape(2);
            if(s) {
                const auto* desc=context->GetOptionalInputDesc(2);
                require(desc&&dtype(desc->GetDataType())==type,"threshold dtype must match person_boxes");
                inputs.push_back(shape(&s->GetStorageShape()));
            }
        }
        const auto* primary_output=context->GetOutputShape(0);
        const bool matrix_rhs=(OpId==SOLVE||OpId==LSTSQ)&&primary_output&&
            primary_output->GetStorageShape().GetDimNum()==inputs[0].size();
        const auto plan=template_contract::make(OpId,inputs,type,attributes(OpId,context->GetAttrs()),matrix_rhs);
        for(size_t i=0;i<plan.outputs.size();++i) {
            const auto* output=context->GetOutputShape(i);
            if(!output) { require(OpId==LSTSQ&&i==1,"missing required output"); continue; }
            const auto actual_shape=shape(&output->GetStorageShape());
            // A square least-squares problem has no residual tensor values.
            // ACL callers may represent that empty output with any zero-size
            // shape; its metadata must not prevent computing the solution.
            if(OpId==LSTSQ&&i==1&&template_contract::empty(plan.outputs[i]))
                require(template_contract::empty(actual_shape),"residuals must be empty for square A");
            else if(OpId==LSTSQ&&i==1&&plan.outputs[0].size()+1==inputs[0].size()) {
                // Retain compatibility with callers of the earlier scalar
                // residual form while inferring the standard [...,1] shape.
                auto scalar_form=plan.outputs[i]; scalar_form.pop_back();
                require(actual_shape==plan.outputs[i]||actual_shape==scalar_form,
                        "vector RHS residual shape mismatch");
            } else template_contract::exact(actual_shape,plan.outputs[i]);
            const auto* desc=context->GetOutputDesc(i);
            const uint32_t expected=plan.tiling.output_types[i];
            require(desc&&(expected==I32?desc->GetDataType()==ge::DT_INT32:
                          dtype(desc->GetDataType())==expected),"template output dtype mismatch");
        }
        auto* tiling=context->GetTilingData<Tiling>();
        require(tiling,"missing tiling storage"); *tiling=plan.tiling;
        select(static_cast<uint32_t>(first->GetDataType()));
        context->SetBlockDim(1);
        auto* workspace=context->GetWorkspaceSizes(1);
        require(workspace,"missing workspace metadata");
        // GetUserWorkspace skips CANN's reserved system area; reserve it on host too.
        auto platform=platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        workspace[0]=platform.GetLibApiWorkSpaceSize()+plan.workspace_bytes;
        return ge::GRAPH_SUCCESS;
    } catch(const std::exception&) { return ge::GRAPH_FAILED; }
}
}

#endif // CANNJUDGE_INLINE_530D7EE08E66F759A4063EDF

// Inlined local header: include/elec/template_diagnostics.h
#ifndef CANNJUDGE_INLINE_D8D4738AF379FE520A09598F
#define CANNJUDGE_INLINE_D8D4738AF379FE520A09598F

#include <base/err_msg.h>
#include <toolchain/slog.h>

namespace elec::template_host {
inline void report_error(const char* stage, const char* message) noexcept {
    const char* where = stage ? stage : "host";
    const char* reason = message ? message : "unknown exception";
    try {
        if (DlogRecord != nullptr)
            DlogRecord(static_cast<int32_t>(OP | STDOUT_LOG_MASK), DLOG_ERROR,
                       "%s: %s", where, reason);
    } catch (...) {}
    try {
        if (error_message::ReportInnerErrMsg != nullptr)
            REPORT_INNER_ERR_MSG("EZ9999", "%s: %s", where, reason);
    } catch (...) {}
}
}

#endif // CANNJUDGE_INLINE_D8D4738AF379FE520A09598F

#include "../op_kernel/lstsq_options.h"
#include <string>

namespace elec::lstsq_host {
using Shape = template_contract::Shape;
constexpr uint64_t memory_limit = uint64_t(1) << 30;

inline uint64_t multiply(uint64_t a, uint64_t b) {
    require(a == 0 || b <= memory_limit / a, "lstsq allocation exceeds 1 GiB");
    return a * b;
}
inline uint64_t add(uint64_t a, uint64_t b) {
    require(a <= memory_limit && b <= memory_limit - a, "lstsq allocation exceeds 1 GiB");
    return a + b;
}
inline uint64_t elements(const Shape& shape) {
    uint64_t result = 1;
    for (const int64_t dimension : shape) {
        require(dimension > 0, "lstsq input dimensions must be positive");
        result = multiply(result, static_cast<uint64_t>(dimension));
    }
    return result;
}
inline uint64_t output_elements(const Shape& shape) {
    if (template_contract::empty(shape)) return 0;
    return elements(shape);
}
inline uint64_t align64(uint64_t bytes) {
    return add(bytes, (64 - bytes % 64) % 64);
}
struct Plan {
    TemplateTilingData tiling;
    std::vector<Shape> outputs;
    Shape batch_shape;
    bool vector_rhs = false;
    uint64_t tensor_bytes = 0;
    uint64_t per_core_bytes = 0;
};

inline Plan make(const std::vector<Shape>& inputs, uint32_t type, Config c,
                 bool matrix_rhs = false) {
    require(inputs.size() == 2 && (type == F32 || type == BF16), "invalid lstsq inputs/type");
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    require(a.size() >= 2, "lstsq A must have at least two dimensions");
    const uint64_t input_a = elements(a), input_b = elements(b);
    Plan plan;
    plan.batch_shape.assign(a.begin(), a.end() - 2);
    c.batch = elements(plan.batch_shape);
    c.m = a[a.size() - 2]; c.n = a.back();
    require(c.m >= c.n, "lstsq requires m >= n");
    const Shape vector_batch = template_contract::append(plan.batch_shape, {int64_t(c.m)});
    plan.vector_rhs = !matrix_rhs && (b == Shape{int64_t(c.m)} || b == vector_batch);
    if (plan.vector_rhs) c.k = 1;
    else {
        require(b.size() >= 2, "lstsq B must be a vector or matrix");
        c.k = b.back();
    }
    const Shape tail = plan.vector_rhs ? Shape{int64_t(c.m)} : Shape{int64_t(c.m), int64_t(c.k)};
    if (b == tail) c.rhs_batch_stride = 0;
    else {
        Shape batched = plan.batch_shape;
        batched.insert(batched.end(), tail.begin(), tail.end());
        template_contract::exact(b, batched);
        c.rhs_batch_stride = multiply(c.m, c.k);
    }
    c.dtype = F32;  // QR and all refinement work are FP32.
    plan.tiling.config = c;
    for (auto& input_type : plan.tiling.input_types) input_type = type;
    for (auto& output_type : plan.tiling.output_types) output_type = type;
    plan.outputs.push_back(plan.vector_rhs
        ? template_contract::append(plan.batch_shape, {int64_t(c.n)})
        : template_contract::append(plan.batch_shape, {int64_t(c.n), int64_t(c.k)}));
    const bool residual_requested = c.m > c.n || (c.flags & LSTSQ_SQUARE_RESIDUAL);
    plan.outputs.push_back(residual_requested
        ? template_contract::append(plan.batch_shape, {int64_t(c.k)}) : Shape{0});

    const uint64_t bytes = type == F32 ? 4 : 2;
    const uint64_t output_x = multiply(multiply(c.batch, c.n), c.k);
    const uint64_t output_r = residual_requested ? multiply(c.batch, c.k) : 0;
    plan.tensor_bytes = multiply(add(add(input_a, input_b), add(output_x, output_r)), bytes);
    if (c.m == c.n && c.n <= 64 && c.k <= 8) {
        plan.tiling.scratch_bytes = 0;
        plan.per_core_bytes = 0;
        return plan;
    }
    const uint64_t pad_n = (c.n + 7) & ~uint64_t(7);
    if (c.m == c.n && c.n <= 256 && c.k <= 8) {
        plan.tiling.scratch_bytes = multiply(add(multiply(c.n, pad_n), 4096), 4);
        plan.per_core_bytes = align64(plan.tiling.scratch_bytes);
        add(plan.tensor_bytes, plan.per_core_bytes);
        return plan;
    }
    const uint64_t scratch_elements = add(add(multiply(c.m, pad_n), multiply(multiply(2, c.m), c.k)), c.m);
    plan.tiling.scratch_bytes = multiply(scratch_elements, 4);
    // Scalar stores stay in cacheline-private staging. Exact DMA writes then
    // publish each batch independently, including narrow residual outputs.
    const uint64_t staged_x = align64(multiply(multiply(c.n, c.k), bytes));
    const uint64_t staged_r = residual_requested ? align64(multiply(c.k, bytes)) : 0;
    plan.per_core_bytes = add(add(align64(plan.tiling.scratch_bytes), staged_x), staged_r);
    add(plan.tensor_bytes, plan.per_core_bytes);
    return plan;
}

inline bool concrete_hint(const gert::Shape* shape) {
    if (!shape || shape->GetDimNum() == 0) return false;
    for (size_t dimension = 0; dimension < shape->GetDimNum(); ++dimension)
        if (shape->GetDim(dimension) < 0) return false;
    return true;
}

inline std::string shape_text(const Shape& shape) {
    std::string result = "[";
    for (size_t dimension = 0; dimension < shape.size(); ++dimension) {
        if (dimension) result += ",";
        result += std::to_string(shape[dimension]);
    }
    return result + "]";
}

inline void append_tensor_shapes(std::string& message, const char* name,
                                 const gert::StorageShape* shape) {
    message += std::string(" ") + name;
    if (!shape) { message += "=missing"; return; }
    message += "_origin=" + shape_text(template_host::shape(&shape->GetOriginShape()));
    message += std::string(" ") + name + "_storage=" +
        shape_text(template_host::shape(&shape->GetStorageShape()));
}

inline std::string output_mismatch(gert::TilingContext* context, const Plan& plan,
                                   size_t index, const Shape& actual,
                                   uint64_t actual_count, uint64_t expected_count) {
    std::string message = "lstsq output view element count mismatch output_index=" +
        std::to_string(index) + " actual_elements=" + std::to_string(actual_count) +
        " expected_elements=" + std::to_string(expected_count) +
        " actual_shape=" + shape_text(actual) +
        " expected_shape=" + shape_text(plan.outputs[index]);
    append_tensor_shapes(message, "A", context->GetRequiredInputShape(0));
    append_tensor_shapes(message, "B", context->GetRequiredInputShape(1));
    append_tensor_shapes(message, "output0", context->GetOutputShape(0));
    append_tensor_shapes(message, "output1", context->GetOutputShape(1));
    const auto& config = plan.tiling.config;
    return message + " m=" + std::to_string(config.m) + " n=" + std::to_string(config.n) +
        " k=" + std::to_string(config.k) + " batch=" + std::to_string(config.batch) +
        " rhs=" + (plan.vector_rhs ? "vector" : "matrix");
}

inline ge::graphStatus infer_shape(gert::InferShapeContext* context) {
    try {
        auto* solution = context->GetOutputShape(0);
        require(solution, "missing lstsq solution shape");
        // A concrete caller hint disambiguates batched vector B from a shared
        // matrix B. Tiling validates its capacity against both interpretations.
        if (concrete_hint(solution)) return ge::GRAPH_SUCCESS;
        const auto* input_a = context->GetInputShape(0);
        const auto* input_b = context->GetInputShape(1);
        if (!input_a || !input_b) return ge::GRAPH_SUCCESS;
        const Shape a = template_host::shape(input_a), b = template_host::shape(input_b);
        if (a.size() < 2 || b.empty()) return ge::GRAPH_SUCCESS;
        for (const auto& input : {a, b})
            for (const int64_t dimension : input)
                if (dimension <= 0) return ge::GRAPH_SUCCESS;

        const Shape batch(a.begin(), a.end() - 2);
        const int64_t m = a[a.size() - 2], n = a.back();
        const Shape matrix_tail{m, b.back()};
        Shape matrix_batch = batch;
        matrix_batch.insert(matrix_batch.end(), matrix_tail.begin(), matrix_tail.end());
        const bool matrix_rhs = b.size() >= 2 && (b == matrix_tail || b == matrix_batch);
        const bool vector_rhs = b == Shape{m} || b == template_contract::append(batch, {m});
        if (!matrix_rhs && !vector_rhs) return ge::GRAPH_SUCCESS;
        if (matrix_rhs && vector_rhs) return ge::GRAPH_SUCCESS;
        // Only unambiguous shape propagation belongs here. Tiling has the
        // concrete output capacity needed to resolve the remaining cases.
        const int64_t k = matrix_rhs ? b.back() : 1;
        const std::vector<Shape> outputs{
            matrix_rhs ? template_contract::append(batch, {n, k})
                       : template_contract::append(batch, {n}),
            m > n ? template_contract::append(batch, {k}) : Shape{0}};
        for (size_t index = 0; index < outputs.size(); ++index) {
            auto* output = context->GetOutputShape(index);
            if (!output || concrete_hint(output)) continue;
            output->SetDimNum(outputs[index].size());
            for (size_t dimension = 0; dimension < outputs[index].size(); ++dimension)
                output->SetDim(dimension, outputs[index][dimension]);
        }
        return ge::GRAPH_SUCCESS;
    } catch (const std::exception& error) {
        template_host::report_error("lstsq infer", error.what());
        return ge::GRAPH_FAILED;
    }
}

template <typename Select>
ge::graphStatus tile(gert::TilingContext* context, Select select) {
    try {
        std::vector<Shape> inputs;
        const auto* first = context->GetRequiredInputDesc(0);
        require(first, "missing lstsq input descriptor");
        const uint32_t type = template_host::dtype(first->GetDataType());
        for (uint32_t index = 0; index < 2; ++index) {
            const auto* shape = context->GetRequiredInputShape(index);
            const auto* description = context->GetRequiredInputDesc(index);
            require(shape && description && template_host::dtype(description->GetDataType()) == type,
                    "lstsq input shape/type mismatch");
            inputs.push_back(template_host::shape(&shape->GetStorageShape()));
        }
        const auto* primary = context->GetOutputShape(0);
        require(primary, "missing lstsq solution");
        auto attributes = template_host::attributes(LSTSQ, context->GetAttrs());
        const auto* residual = context->GetOutputShape(1);
        if (inputs[0].size() >= 2 && inputs[0][inputs[0].size() - 2] == inputs[0].back() &&
            residual && output_elements(template_host::shape(&residual->GetStorageShape())) != 0)
            attributes.flags |= LSTSQ_SQUARE_RESIDUAL;
        auto plan = make(inputs, type, attributes);
        const auto primary_shape = template_host::shape(&primary->GetStorageShape());
        // Resolve the existing vector/matrix RHS ambiguity by storage count,
        // so flattening the output descriptor cannot change the RHS meaning.
        if (output_elements(primary_shape) != output_elements(plan.outputs[0]))
            plan = make(inputs, type, attributes, true);
        for (size_t index = 0; index < plan.outputs.size(); ++index) {
            const auto* output = context->GetOutputShape(index);
            if (!output) { require(index == 1, "missing lstsq solution"); continue; }
            const auto actual = template_host::shape(&output->GetStorageShape());
            // ND output storage is indexed linearly by the kernel. Dense
            // reshapes, including flat and singleton-axis views, preserve
            // every written value and must have exactly the same capacity.
            const uint64_t actual_count = output_elements(actual);
            const uint64_t expected_count = output_elements(plan.outputs[index]);
            if (actual_count != expected_count)
                throw std::invalid_argument(output_mismatch(context, plan, index, actual,
                                                           actual_count, expected_count));
            const auto* description = context->GetOutputDesc(index);
            require(description && template_host::dtype(description->GetDataType()) == type,
                    "lstsq output dtype mismatch");
        }
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const uint64_t available = platform.GetCoreNumAiv();
        require(available > 0, "lstsq requires vector cores");
        const uint64_t batches = plan.tiling.config.batch;
        const auto& cfg = plan.tiling.config;
        const bool use_p8 = (cfg.m == cfg.n && (cfg.n == 200 || cfg.n == 256) && cfg.k <= 8 &&
                             multiply(batches, 8) <= available);
        const bool use_p4 = (!use_p8 && cfg.m == cfg.n && (cfg.n == 200 || cfg.n == 256) && cfg.k <= 8 &&
                             multiply(batches, 4) <= available);
        const uint64_t desired_blocks = use_p8 ? multiply(batches, 8) : (use_p4 ? multiply(batches, 4) : batches);
        const uint64_t blocks = desired_blocks < available ? desired_blocks : available;
        const uint64_t user_ws = plan.per_core_bytes == 0 ? 0ULL : add(16384ULL, multiply(blocks, plan.per_core_bytes));
        const uint64_t workspace_bytes = add(platform.GetLibApiWorkSpaceSize(), user_ws);
        add(plan.tensor_bytes, workspace_bytes);
        auto* tiling = context->GetTilingData<TemplateTilingData>();
        auto* workspace = context->GetWorkspaceSizes(1);
        require(tiling && workspace, "missing lstsq tiling/workspace storage");
        *tiling = plan.tiling; workspace[0] = workspace_bytes;
        select(static_cast<uint32_t>(first->GetDataType()));
        context->SetBlockDim(static_cast<uint32_t>(blocks));
        return ge::GRAPH_SUCCESS;
    } catch (const std::exception& error) {
        template_host::report_error("lstsq tiling", error.what());
        return ge::GRAPH_FAILED;
    }
}
}
