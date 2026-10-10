// LLMario native engine: Metal kernels, part 2 (decode GEMV, `y = W x` for one token).
//
// Bandwidth-tuned structure: a threadgroup of GEMV_NSG simdgroups, each simdgroup owns GEMV_NR
// consecutive rows and streams their blocks with all 32 lanes reading adjacent bytes; the x values
// a lane needs are loaded once per block position and reused across the simdgroup's rows; the
// per-row partial sums are reduced with `simd_sum` at the end. Lane mappings per block type are
// chosen so that one block (or a few) is read with contiguous, aligned loads.
//
// One template kernel per (type, variant): `gemv_<type>` writes `y = W x`, `gemv_acc_<type>`
// accumulates `y += W x` (the residual add fused into the output projections) and
// `gemv_glu_<type>` computes `y = silu(W1 x) * (W2 x)` (gate and up in one pass over x).

#define GEMV_NSG 2
#define GEMV_NR 2

struct GemvParams {
    uint rows;
    uint cols;
};

// Per-type lane work: accumulate this lane's partial dot products of rows row0..row0+GEMV_NR with x.

static inline void gemv_rows(TagF32, device const uchar* W, device const float* x, uint row0, uint rows,
                             uint cols, ushort tiisg, thread float* acc) {
    for (uint i = tiisg * 4; i < cols; i += 128) {
        const float4 xv = *(device const float4*)(x + i);
        for (int r = 0; r < GEMV_NR; r++) {
            const uint row = row0 + r;
            if (row < rows) {
                const float4 w = *(device const float4*)((device const float*)W + (ulong)row * cols + i);
                acc[r] += dot(w, xv);
            }
        }
    }
}

static inline void gemv_rows(TagF16, device const uchar* W, device const float* x, uint row0, uint rows,
                             uint cols, ushort tiisg, thread float* acc) {
    for (uint i = tiisg * 4; i < cols; i += 128) {
        const float4 xv = *(device const float4*)(x + i);
        for (int r = 0; r < GEMV_NR; r++) {
            const uint row = row0 + r;
            if (row < rows) {
                const half4 w = *(device const half4*)((device const half*)W + (ulong)row * cols + i);
                acc[r] += dot(float4(w), xv);
            }
        }
    }
}

// Q4_0: one 18-byte block per lane per iteration (32 adjacent blocks per simdgroup).
static inline void gemv_rows(TagQ4_0, device const uchar* W, device const float* x, uint row0, uint rows,
                             uint cols, ushort tiisg, thread float* acc) {
    const uint nb = cols / 32;
    for (uint ib = tiisg; ib < nb; ib += 32) {
        device const float4* xb = (device const float4*)(x + ib * 32);
        float4 xl[4], xh[4];
        for (int k = 0; k < 4; k++) { xl[k] = xb[k]; xh[k] = xb[4 + k]; }
        for (int r = 0; r < GEMV_NR; r++) {
            const uint row = row0 + r;
            if (row >= rows) break;
            device const uchar* b = W + ((ulong)row * nb + ib) * 18;
            const float d = float(*(device const half*)b);
            device const packed_uchar4* qs = (device const packed_uchar4*)(b + 2);
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                const uchar4 q = uchar4(qs[k]);
                const float4 lo = float4(int4(q & 0xF) - 8);
                const float4 hi = float4(int4(q >> 4) - 8);
                sum += dot(lo, xl[k]) + dot(hi, xh[k]);
            }
            acc[r] += d * sum;
        }
    }
}

// Q8_0: one 34-byte block per lane per iteration.
static inline void gemv_rows(TagQ8_0, device const uchar* W, device const float* x, uint row0, uint rows,
                             uint cols, ushort tiisg, thread float* acc) {
    const uint nb = cols / 32;
    for (uint ib = tiisg; ib < nb; ib += 32) {
        device const float4* xb = (device const float4*)(x + ib * 32);
        float4 xv[8];
        for (int k = 0; k < 8; k++) xv[k] = xb[k];
        for (int r = 0; r < GEMV_NR; r++) {
            const uint row = row0 + r;
            if (row >= rows) break;
            device const uchar* b = W + ((ulong)row * nb + ib) * 34;
            const float d = float(*(device const half*)b);
            device const packed_char4* qs = (device const packed_char4*)(b + 2);
            float sum = 0.0f;
            for (int k = 0; k < 8; k++) sum += dot(float4(char4(qs[k])), xv[k]);
            acc[r] += d * sum;
        }
    }
}

// Q4_K: four super-blocks in flight per simdgroup (ix = lane / 8); the 8 lanes of a super-block
// each take 16 bytes of `qs` (chunk c = il / 2, half hf = il % 2), which hold 16 low nibbles
// (sub-block 2c) and 16 high nibbles (sub-block 2c + 1).
static inline void gemv_rows(TagQ4_K, device const uchar* W, device const float* x, uint row0, uint rows,
                             uint cols, ushort tiisg, thread float* acc) {
    const uint nb = cols / QK_K;
    const uint ix = tiisg / 8;
    const uint il = tiisg % 8;
    const uint c = il / 2;
    const uint hf = il % 2;
    for (uint ib = ix; ib < nb; ib += 4) {
        device const float4* xb = (device const float4*)(x + ib * QK_K + c * 64 + hf * 16);
        float4 xl[4], xh[4];
        float sxl = 0.0f, sxh = 0.0f;
        for (int k = 0; k < 4; k++) {
            xl[k] = xb[k];
            xh[k] = xb[8 + k];
            sxl += xl[k].x + xl[k].y + xl[k].z + xl[k].w;
            sxh += xh[k].x + xh[k].y + xh[k].z + xh[k].w;
        }
        for (int r = 0; r < GEMV_NR; r++) {
            const uint row = row0 + r;
            if (row >= rows) break;
            device const block_q4_K* b = (device const block_q4_K*)(W + ((ulong)row * nb + ib) * 144);
            const float d = float(b->d);
            const float dmin = float(b->dmin);
            uchar sc0, m0, sc1, m1;
            get_scale_min_k4(2 * c, b->scales, sc0, m0);
            get_scale_min_k4(2 * c + 1, b->scales, sc1, m1);
            const uint4 qq = *(device const uint4*)(b->qs + c * 32 + hf * 16);
            const uchar4 q0 = as_type<uchar4>(qq.x);
            const uchar4 q1 = as_type<uchar4>(qq.y);
            const uchar4 q2 = as_type<uchar4>(qq.z);
            const uchar4 q3 = as_type<uchar4>(qq.w);
            float sql = dot(float4(q0 & 0xF), xl[0]) + dot(float4(q1 & 0xF), xl[1])
                      + dot(float4(q2 & 0xF), xl[2]) + dot(float4(q3 & 0xF), xl[3]);
            float sqh = dot(float4(q0 >> 4), xh[0]) + dot(float4(q1 >> 4), xh[1])
                      + dot(float4(q2 >> 4), xh[2]) + dot(float4(q3 >> 4), xh[3]);
            acc[r] += d * (float(sc0) * sql + float(sc1) * sqh) - dmin * (float(m0) * sxl + float(m1) * sxh);
        }
    }
}

// Q5_K: as Q4_K plus the fifth bit from `qh` (bit s of qh[l] for sub-block s).
static inline void gemv_rows(TagQ5_K, device const uchar* W, device const float* x, uint row0, uint rows,
                             uint cols, ushort tiisg, thread float* acc) {
    const uint nb = cols / QK_K;
    const uint ix = tiisg / 8;
    const uint il = tiisg % 8;
    const uint c = il / 2;
    const uint hf = il % 2;
    const uchar s0 = 2 * c;
    const uchar s1 = 2 * c + 1;
    for (uint ib = ix; ib < nb; ib += 4) {
        device const float4* xb = (device const float4*)(x + ib * QK_K + c * 64 + hf * 16);
        float4 xl[4], xh[4];
        float sxl = 0.0f, sxh = 0.0f;
        for (int k = 0; k < 4; k++) {
            xl[k] = xb[k];
            xh[k] = xb[8 + k];
            sxl += xl[k].x + xl[k].y + xl[k].z + xl[k].w;
            sxh += xh[k].x + xh[k].y + xh[k].z + xh[k].w;
        }
        for (int r = 0; r < GEMV_NR; r++) {
            const uint row = row0 + r;
            if (row >= rows) break;
            device const block_q5_K* b = (device const block_q5_K*)(W + ((ulong)row * nb + ib) * 176);
            const float d = float(b->d);
            const float dmin = float(b->dmin);
            uchar sc0, m0, sc1, m1;
            get_scale_min_k4(2 * c, b->scales, sc0, m0);
            get_scale_min_k4(2 * c + 1, b->scales, sc1, m1);
            const uint4 qq = *(device const uint4*)(b->qs + c * 32 + hf * 16);
            const uint4 hh = *(device const uint4*)(b->qh + hf * 16);
            uchar4 q[4] = {as_type<uchar4>(qq.x), as_type<uchar4>(qq.y), as_type<uchar4>(qq.z), as_type<uchar4>(qq.w)};
            uchar4 h[4] = {as_type<uchar4>(hh.x), as_type<uchar4>(hh.y), as_type<uchar4>(hh.z), as_type<uchar4>(hh.w)};
            float sql = 0.0f, sqh = 0.0f;
            for (int k = 0; k < 4; k++) {
                const uchar4 lo = (q[k] & 0xF) | (((h[k] >> s0) & 1) << 4);
                const uchar4 hi = (q[k] >> 4) | (((h[k] >> s1) & 1) << 4);
                sql += dot(float4(lo), xl[k]);
                sqh += dot(float4(hi), xh[k]);
            }
            acc[r] += d * (float(sc0) * sql + float(sc1) * sqh) - dmin * (float(m0) * sxl + float(m1) * sxh);
        }
    }
}

// Q6_K: two super-blocks in flight per simdgroup (ix = lane / 16); 16 lanes per super-block,
// lane il covers half h = il / 8 and l0 = (il % 8) * 4, i.e. 4 consecutive `l` values in each of
// the four 32-element quarters of that half (16 elements per lane).
static inline void gemv_rows(TagQ6_K, device const uchar* W, device const float* x, uint row0, uint rows,
                             uint cols, ushort tiisg, thread float* acc) {
    const uint nb = cols / QK_K;
    const uint ix = tiisg / 16;
    const uint il = tiisg % 16;
    const uint h = il / 8;
    const uint l0 = (il % 8) * 4;
    const uint is = l0 / 16;
    for (uint ib = ix; ib < nb; ib += 2) {
        device const float* xb = x + ib * QK_K + h * 128 + l0;
        const float4 x1 = *(device const float4*)(xb);
        const float4 x2 = *(device const float4*)(xb + 32);
        const float4 x3 = *(device const float4*)(xb + 64);
        const float4 x4 = *(device const float4*)(xb + 96);
        for (int r = 0; r < GEMV_NR; r++) {
            const uint row = row0 + r;
            if (row >= rows) break;
            device const uchar* b = W + ((ulong)row * nb + ib) * 210;
            const uchar4 qla = uchar4(*(device const packed_uchar4*)(b + h * 64 + l0));
            const uchar4 qlb = uchar4(*(device const packed_uchar4*)(b + h * 64 + l0 + 32));
            const uchar4 qh = uchar4(*(device const packed_uchar4*)(b + 128 + h * 32 + l0));
            device const char* sc = (device const char*)(b + 192 + h * 8);
            const float d = float(*(device const half*)(b + 208));
            const float4 q1 = float4(int4((qla & 0xF) | ((qh & 3) << 4)) - 32);
            const float4 q2 = float4(int4((qlb & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
            const float4 q3 = float4(int4((qla >> 4) | (((qh >> 4) & 3) << 4)) - 32);
            const float4 q4 = float4(int4((qlb >> 4) | (((qh >> 6) & 3) << 4)) - 32);
            acc[r] += d * (float(sc[is]) * dot(q1, x1) + float(sc[is + 2]) * dot(q2, x2)
                         + float(sc[is + 4]) * dot(q3, x3) + float(sc[is + 6]) * dot(q4, x4));
        }
    }
}

// GELU of ggml-cpu (fp16 table semantics; misc.metal `gelu_fp16`, repeated here because the GEMV
// source comes first).
static inline float gemv_gelu(float x) {
    if (x <= -10.0f) return 0.0f;
    if (x >= 10.0f) return x;
    const float h = float(half(x));
    const float g = 0.5f * h * (1.0f + precise::tanh(0.7978845608f * h * (1.0f + 0.044715f * h * h)));
    return float(half(g));
}

// Variants: 0 = store, 1 = accumulate into y, 2 = SwiGLU (silu(W x) * (W2 x)), 3 = GeGLU
// (gelu(W x) * (W2 x)).
template <typename Tag, int VARIANT>
kernel void gemv_t(device const uchar* W [[buffer(0)]],
                   device const float* x [[buffer(1)]],
                   device float* y [[buffer(2)]],
                   constant GemvParams& p [[buffer(3)]],
                   device const uchar* W2 [[buffer(4)]],
                   uint tgpig [[threadgroup_position_in_grid]],
                   ushort tiisg [[thread_index_in_simdgroup]],
                   ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint row0 = (tgpig * GEMV_NSG + sgitg) * GEMV_NR;
    float acc[GEMV_NR] = {0.0f, 0.0f};
    gemv_rows(Tag(), W, x, row0, p.rows, p.cols, tiisg, acc);
    if (VARIANT == 2 || VARIANT == 3) {
        float acc2[GEMV_NR] = {0.0f, 0.0f};
        gemv_rows(Tag(), W2, x, row0, p.rows, p.cols, tiisg, acc2);
        for (int r = 0; r < GEMV_NR; r++) {
            const float g = simd_sum(acc[r]);
            const float u = simd_sum(acc2[r]);
            if (tiisg == 0 && row0 + r < p.rows) {
                y[row0 + r] = (VARIANT == 2 ? g / (1.0f + precise::exp(-g)) : gemv_gelu(g)) * u;
            }
        }
    } else {
        for (int r = 0; r < GEMV_NR; r++) {
            const float s = simd_sum(acc[r]);
            if (tiisg == 0 && row0 + r < p.rows) {
                if (VARIANT == 1) y[row0 + r] += s; else y[row0 + r] = s;
            }
        }
    }
}

#define GEMV_INSTANCE(name, Tag, V) \
    template [[host_name(name)]] kernel void gemv_t<Tag, V>(device const uchar*, device const float*, device float*, constant GemvParams&, device const uchar*, uint, ushort, ushort);

GEMV_INSTANCE("gemv_f32", TagF32, 0)
GEMV_INSTANCE("gemv_f16", TagF16, 0)
GEMV_INSTANCE("gemv_q4_0", TagQ4_0, 0)
GEMV_INSTANCE("gemv_q8_0", TagQ8_0, 0)
GEMV_INSTANCE("gemv_q4_k", TagQ4_K, 0)
GEMV_INSTANCE("gemv_q5_k", TagQ5_K, 0)
GEMV_INSTANCE("gemv_q6_k", TagQ6_K, 0)
GEMV_INSTANCE("gemv_acc_f32", TagF32, 1)
GEMV_INSTANCE("gemv_acc_f16", TagF16, 1)
GEMV_INSTANCE("gemv_acc_q4_0", TagQ4_0, 1)
GEMV_INSTANCE("gemv_acc_q8_0", TagQ8_0, 1)
GEMV_INSTANCE("gemv_acc_q4_k", TagQ4_K, 1)
GEMV_INSTANCE("gemv_acc_q5_k", TagQ5_K, 1)
GEMV_INSTANCE("gemv_acc_q6_k", TagQ6_K, 1)
GEMV_INSTANCE("gemv_glu_f32", TagF32, 2)
GEMV_INSTANCE("gemv_glu_f16", TagF16, 2)
GEMV_INSTANCE("gemv_glu_q4_0", TagQ4_0, 2)
GEMV_INSTANCE("gemv_glu_q8_0", TagQ8_0, 2)
GEMV_INSTANCE("gemv_glu_q4_k", TagQ4_K, 2)
GEMV_INSTANCE("gemv_glu_q5_k", TagQ5_K, 2)
GEMV_INSTANCE("gemv_glu_q6_k", TagQ6_K, 2)
GEMV_INSTANCE("gemv_geglu_f32", TagF32, 3)
GEMV_INSTANCE("gemv_geglu_f16", TagF16, 3)
GEMV_INSTANCE("gemv_geglu_q4_0", TagQ4_0, 3)
GEMV_INSTANCE("gemv_geglu_q8_0", TagQ8_0, 3)
GEMV_INSTANCE("gemv_geglu_q4_k", TagQ4_K, 3)
GEMV_INSTANCE("gemv_geglu_q5_k", TagQ5_K, 3)
GEMV_INSTANCE("gemv_geglu_q6_k", TagQ6_K, 3)
