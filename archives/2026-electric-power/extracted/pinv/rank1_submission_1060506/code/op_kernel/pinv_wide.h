    struct Wide { float h, l; };
    __aicore__ inline Wide Plus(Wide a, Wide b) {
        volatile float sum = a.h + b.h;
        volatile float bv = sum - a.h;
        volatile float av = sum - bv;
        volatile float br = b.h - bv;
        volatile float ar = a.h - av;
        volatile float e = ar + br;
        e = e + a.l; e = e + b.l;
        volatile float h = sum + e;
        volatile float z = h - sum;
        return {h, e - z};
    }
    __aicore__ inline Wide Minus(Wide a, Wide b) { return Plus(a, {-b.h, -b.l}); }
    __aicore__ inline Wide Times(Wide a, Wide b) {
        volatile float ahc = pinv::kSplitFactor * a.h, bhc = pinv::kSplitFactor * b.h;
        volatile float ahv = ahc - a.h, bhv = bhc - b.h;
        volatile float ah = ahc - ahv, bh = bhc - bhv;
        volatile float al = a.h - ah, bl = b.h - bh;
        volatile float p = a.h * b.h;
        volatile float e = ah * bh; e = e - p;
        volatile float x1 = ah * bl; e = e + x1;
        volatile float x2 = al * bh; e = e + x2;
        volatile float x3 = al * bl; e = e + x3;
        volatile float x4 = a.h * b.l; e = e + x4;
        volatile float x5 = a.l * b.h; e = e + x5;
        return Plus({p, 0}, {e, 0});
    }
    __aicore__ inline Wide Divide(Wide a, Wide b) {
        float q = a.h / b.h;
        Wide residual = Minus(a, Times(b, {q, 0}));
        return Plus({q, 0}, {residual.h / b.h, 0});
    }
    __aicore__ inline Wide Entry(LocalTensor<float> h, LocalTensor<float> l, uint32_t pos) {
        return {h.GetValue(pos), l.GetValue(pos)};
    }
    __aicore__ inline void SaveEntry(LocalTensor<float> h, LocalTensor<float> l, uint32_t pos, Wide val) {
        h.SetValue(pos, val.h); l.SetValue(pos, val.l);
    }
