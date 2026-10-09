// LLMario native engine: Metal kernels, part 3 (prefill GEMM, `Y = X Wᵀ` for n tokens).
//
// Threadgroup tile: BM = 64 weight rows × BN = 32 tokens, stepping BK = 32 along the shared
// dimension. Per step every thread dequantises 16 weight elements to f16 in threadgroup memory and
// converts 8 activations to f16; the four simdgroups then each multiply a 16-row × 32-token
// sub-tile with `simdgroup_matrix` 8×8 multiply-accumulates (f16 inputs, f32 accumulation).
// Output is written transposed into the token-major activation layout `y[token][row]`.

#define GEMM_BM 64
#define GEMM_BN 32
#define GEMM_BK 32
#define GEMM_THREADS 128

struct GemmParams {
    uint rows;       // output features (rows of W)
    uint cols;       // shared dimension (row length of W; multiple of 16)
    uint n;          // tokens
    uint row_bytes;  // bytes per W row
    uint accumulate; // 1: y += X Wᵀ (fused residual add)
};

template <typename Tag>
kernel void gemm_t(device const uchar* W [[buffer(0)]],
                   device const float* x [[buffer(1)]],
                   device float* y [[buffer(2)]],
                   constant GemmParams& p [[buffer(3)]],
                   uint2 tgpig [[threadgroup_position_in_grid]],
                   ushort tiitg [[thread_index_in_threadgroup]],
                   ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[GEMM_BM * GEMM_BK];   // [row][k]
    threadgroup half sb[GEMM_BN * GEMM_BK];   // [token][k]
    threadgroup float sc[GEMM_BM * GEMM_BN];  // [row][token] staging for edge tiles

    const uint tok0 = tgpig.x * GEMM_BN;
    const uint row0 = tgpig.y * GEMM_BM;

    const bool full = row0 + GEMM_BM <= p.rows && tok0 + GEMM_BN <= p.n;
    simdgroup_float8x8 c[8];
    if (p.accumulate != 0 && full) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                device const float* src = y + (ulong)(tok0 + j * 8) * p.rows + row0 + sgitg * 16 + i * 8;
                simdgroup_load(c[i * 4 + j], src, p.rows, ulong2(0, 0), true);
            }
        }
    } else {
        for (int i = 0; i < 8; i++) c[i] = simdgroup_float8x8(0.0f);
    }

    const uint arow = tiitg / 2;
    const uint akseg = (tiitg % 2) * 16;
    const uint btok = tiitg / 4;
    const uint bkseg = (tiitg % 4) * 8;

    for (uint k0 = 0; k0 < p.cols; k0 += GEMM_BK) {
        {
            const uint row = row0 + arow;
            const uint e0 = k0 + akseg;
            float tmp[16];
            if (row < p.rows && e0 < p.cols) {
                dq16(Tag(), W + (ulong)row * p.row_bytes, e0, tmp);
            } else {
                for (int j = 0; j < 16; j++) tmp[j] = 0.0f;
            }
            threadgroup half* dst = sa + arow * GEMM_BK + akseg;
            for (int j = 0; j < 16; j++) dst[j] = half(tmp[j]);
        }
        {
            const uint tok = tok0 + btok;
            const uint e0 = k0 + bkseg;
            threadgroup half* dst = sb + btok * GEMM_BK + bkseg;
            if (tok < p.n && e0 + 8 <= p.cols) {
                device const float* src = x + (ulong)tok * p.cols + e0;
                for (int j = 0; j < 8; j++) dst[j] = half(src[j]);
            } else {
                for (int j = 0; j < 8; j++) dst[j] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < GEMM_BK; kk += 8) {
            simdgroup_half8x8 a0, a1, b[4];
            simdgroup_load(a0, sa + (sgitg * 16 + 0) * GEMM_BK + kk, GEMM_BK);
            simdgroup_load(a1, sa + (sgitg * 16 + 8) * GEMM_BK + kk, GEMM_BK);
            for (int j = 0; j < 4; j++) {
                simdgroup_load(b[j], sb + (j * 8) * GEMM_BK + kk, GEMM_BK, ulong2(0, 0), true);
            }
            for (int j = 0; j < 4; j++) {
                simdgroup_multiply_accumulate(c[j], a0, b[j], c[j]);
                simdgroup_multiply_accumulate(c[4 + j], a1, b[j], c[4 + j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (full) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                device float* dst = y + (ulong)(tok0 + j * 8) * p.rows + row0 + sgitg * 16 + i * 8;
                simdgroup_store(c[i * 4 + j], dst, p.rows, ulong2(0, 0), true);
            }
        }
    } else {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                simdgroup_store(c[i * 4 + j], sc + (sgitg * 16 + i * 8) * GEMM_BN + j * 8, GEMM_BN);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tiitg; idx < GEMM_BM * GEMM_BN; idx += GEMM_THREADS) {
            const uint r = idx / GEMM_BN;
            const uint t = idx % GEMM_BN;
            if (row0 + r < p.rows && tok0 + t < p.n) {
                device float* dst = y + (ulong)(tok0 + t) * p.rows + row0 + r;
                *dst = (p.accumulate != 0 ? *dst : 0.0f) + sc[r * GEMM_BN + t];
            }
        }
    }
}

#define GEMM_INSTANCE(name, Tag) \
    template [[host_name(name)]] kernel void gemm_t<Tag>(device const uchar*, device const float*, device float*, constant GemmParams&, uint2, ushort, ushort);

GEMM_INSTANCE("gemm_f32", TagF32)
GEMM_INSTANCE("gemm_f16", TagF16)
GEMM_INSTANCE("gemm_q4_0", TagQ4_0)
GEMM_INSTANCE("gemm_q8_0", TagQ8_0)
GEMM_INSTANCE("gemm_q4_k", TagQ4_K)
GEMM_INSTANCE("gemm_q5_k", TagQ5_K)
GEMM_INSTANCE("gemm_q6_k", TagQ6_K)
