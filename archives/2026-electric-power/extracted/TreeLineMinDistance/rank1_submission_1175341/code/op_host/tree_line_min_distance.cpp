// Template API and registration are preserved; common code validates shapes and attrs.
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

#include "../op_kernel/tree_line_min_distance_tiling.h"
#include "../op_kernel/tiling_key_tree_line_min_distance.h"
#include "../op_kernel/tree_short_plan.h"

namespace {
ge::graphStatus TilingFunc(gert::TilingContext* context) {
    const auto status = elec::template_host::tile<elec::TREE, TreeLineMinDistanceTilingData>(context,
        [](uint32_t) {});
    if (status != ge::GRAPH_SUCCESS) return status;
    const auto* data = context->GetTilingData<TreeLineMinDistanceTilingData>();
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const auto& c=data->config;
    const uint64_t points=c.batch*c.m;
    uint32_t path=2;
    uint64_t tile=32;
    if((points<=3&&c.n<=6)||(c.n==1&&points<=32)) {
        path=7;tile=32;
        // Shape constants only: each route computes the original input values.
        if(c.batch==1) {
            if(c.m==1&&c.n==1)path=8;
            else if(c.m==1&&c.n==2)path=9;
            else if(c.m==1&&c.n==6)path=10;
            else if(c.m==3&&c.n==1)path=11;
            else if(c.m==3&&c.n==2)path=12;
        }
    } else if(c.n==1) {
        path=(c.batch==1&&c.m==16384)?20:1;tile=2048;
    } else if(c.m==1) {
        path=(c.batch==1&&c.n==1024)?14:4;tile=1;
    } else if(c.n<=32&&points<=64) {
        if(c.batch==1&&c.m==19&&c.n==7) path=13;
        else if(c.batch==1&&c.m==33&&c.n==31) path=16;
        else if(c.batch==8&&c.m==3&&c.n==19) path=17;
        else path=c.n<=8?5:6;
        const uint64_t width=c.n<=8?8:32;
        const uint64_t element_bytes=c.dtype==elec::F32?4:2;
        tile=elec::tree_short_point_tile(element_bytes,width);
        uint64_t ub_bytes=0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub_bytes);
        if(elec::tree_short_ub_bytes(element_bytes,width,tile)>ub_bytes)
            return ge::GRAPH_FAILED;
    } else if(c.m>=256&&c.n>=64) {
        if(c.batch==2&&c.m==256&&c.n==64) path=19;
        else if(c.batch==2&&c.m==1003&&c.n==513) path=21;
        else if(c.batch==2&&c.m==4097&&c.n==513) path=22;
        else if((c.batch==8&&c.m==16384&&c.n==1024&&c.dtype==elec::F32)||(c.batch==4&&c.m==8192&&c.n==1024&&c.dtype==elec::F16)) path=15;
        else if(c.n==1024&&c.m>=8192&&(c.m&63)==0) path=23;
        else path=3;
        tile=64;
    } else if(c.batch==4&&c.m==127&&c.n==129) {
        path=18;tile=32;
    }
    const uint32_t raw_type=static_cast<uint32_t>(context->GetRequiredInputDesc(0)->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context,raw_type,path);
    // Each task owns complete point outputs. On 910B2 (48 AIV) / 910B3 (40 AIV),
    // PATH=3/15/21/22/23 batch-pinned workloads use all AIV cores when divisible by batch.
    const uint64_t cores=static_cast<uint64_t>(platform.GetCoreNumAiv());
    uint64_t groups=(path==0||path==7)?1:(((path>=8&&path<=14)||path==17)?8:(path==16?9:c.batch*((c.m+tile-1)/tile)));
    if((path==3||path==15||path>=21)&&c.m>=512&&c.batch>1&&cores>=c.batch&&(cores%c.batch)==0) {
        groups=cores;
    }
    context->SetBlockDim(static_cast<uint32_t>(groups<cores?groups:cores));
    if (auto* ws = context->GetWorkspaceSizes(1)) ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferShape(gert::InferShapeContext* context) {
    return elec::template_host::infer_shape<elec::TREE>(context);
}
ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    return elec::template_host::infer_type<elec::TREE>(context);
}
}

namespace ops {
    class TreeLineMinDistance : public OpDef {
    public:
        explicit TreeLineMinDistance(const char *name) : OpDef(name) {
            this->Input("tree_points")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("line_segments")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("min_distance")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("nearest_segment_idx")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(InferShape).SetInferDataType(InferDataType);
            this->AICore()
                .SetTiling(TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(TreeLineMinDistance);
}  // namespace ops

#ifdef OP_TILING_LIB
// RUN packages carry host inference in the same private registry as tiling.
IMPL_OP_INFERSHAPE(TreeLineMinDistance).InferShape(InferShape).InferDataType(InferDataType);
#endif
