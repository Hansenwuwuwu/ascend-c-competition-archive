    __aicore__ inline void VectorEliminate(Wide factor) {
        auto qh = calc[0], ql = calc[1], bh = calc[2], bl = calc[3], e = calc[4], part = calc[5], product = calc[6], aux = calc[7];
        volatile float ac = factor.h * pinv::kSplitFactor;
        volatile float az = ac - factor.h;
        volatile float ah = ac - az;
        volatile float al = factor.h - ah;
        Muls(bh, u, pinv::kSplitFactor, s); PipeBarrier<PIPE_V>();
        Sub(bl, bh, u, s); PipeBarrier<PIPE_V>(); Sub(bh, bh, bl, s); PipeBarrier<PIPE_V>();
        Sub(bl, u, bh, s); Muls(product, u, factor.h, s); PipeBarrier<PIPE_V>();
        Muls(e, bh, static_cast<float>(ah), s); PipeBarrier<PIPE_V>(); Sub(e, e, product, s); PipeBarrier<PIPE_V>();
        Muls(part, bl, static_cast<float>(ah), s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, bh, static_cast<float>(al), s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, bl, static_cast<float>(al), s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, v, factor.h, s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, u, factor.l, s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Add(qh, product, e, s); PipeBarrier<PIPE_V>(); Sub(ql, qh, product, s); PipeBarrier<PIPE_V>(); Sub(ql, e, ql, s); PipeBarrier<PIPE_V>();
        Sub(bh, x, qh, s); PipeBarrier<PIPE_V>(); Sub(bl, bh, x, s); PipeBarrier<PIPE_V>();
        Sub(e, bh, bl, s); PipeBarrier<PIPE_V>(); Sub(e, x, e, s); Muls(part, qh, -1.0f, s); PipeBarrier<PIPE_V>();
        Sub(part, part, bl, s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Add(e, e, y, s); PipeBarrier<PIPE_V>(); Sub(e, e, ql, s); PipeBarrier<PIPE_V>();
        Add(x, bh, e, s); PipeBarrier<PIPE_V>(); Sub(aux, x, bh, s); PipeBarrier<PIPE_V>(); Sub(y, e, aux, s); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void VectorNormalize(Wide divisor) {
        float reciprocal = 1.0f / divisor.h;
        Muls(u, x, reciprocal, s); Duplicate(v, 0.0f, s); PipeBarrier<PIPE_V>();
        VectorEliminate(divisor);
        Muls(calc[0], x, reciprocal, s); PipeBarrier<PIPE_V>();
        Add(x, u, calc[0], s); PipeBarrier<PIPE_V>();
        Sub(y, x, u, s); PipeBarrier<PIPE_V>();
        Sub(y, calc[0], y, s); PipeBarrier<PIPE_V>();
    }
