#pragma once

template <bool Closed0,bool Closed1>
__aicore__ inline void SplineLinearFactors(float q0,float q1,float *f) {
    if constexpr(Closed0) {q0*=2.0f;q0=q0>1.0f?2.0f-q0:q0;}
    if constexpr(Closed1) {q1*=2.0f;q1=q1>1.0f?2.0f-q1:q1;}
    f[0]=(1.0f-q0)*(1.0f-q1);
    f[1]=q0*(1.0f-q1);
    f[2]=(1.0f-q0)*q1;
    f[3]=q0*q1;
}

template <bool Closed0,bool Closed1,bool ExplicitBarrier=true>
__aicore__ inline void SplineSingleInterpolate(const LocalTensor<float>& dot,
        const LocalTensor<float>& weights,float q0,float q1) {
    if constexpr(Closed0) {q0*=2.0f;q0=q0>1.0f?2.0f-q0:q0;}
    if constexpr(Closed1) {q1*=2.0f;q1=q1>1.0f?2.0f-q1:q1;}
    Muls(dot,weights,(1-q0)*(1-q1),128);if constexpr(ExplicitBarrier) PipeBarrier<PIPE_V>();
    Axpy(dot,weights[128],q0*(1-q1),128);if constexpr(ExplicitBarrier) PipeBarrier<PIPE_V>();
    Axpy(dot,weights[256],(1-q0)*q1,128);if constexpr(ExplicitBarrier) PipeBarrier<PIPE_V>();
    Axpy(dot,weights[384],q0*q1,128);if constexpr(ExplicitBarrier) PipeBarrier<PIPE_V>();
}
