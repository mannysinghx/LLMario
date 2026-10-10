// LLMario native engine: Metal kernels, part 3 (prefill GEMM, `Y = X Wᵀ` for n tokens).
//
// `gemmf_t` (below) is the main kernel; `gemm_t` serves matrices whose row length is not a
// multiple of 32.
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

// Prefill GEMM after ggml-metal's `kernel_mul_mm` (MIT): 64 rows × 32 tokens per threadgroup,
// 8 KiB of threadgroup memory, the output computed as tokens × rows so no load or store is
// transposed, each 8×8 operand tile contiguous in threadgroup memory, and the next weight tile
// dequantised into registers *before* the barrier so its memory latency overlaps the other
// simdgroups' multiply-accumulates. Simdgroup s computes rows 32·(s%2).. and tokens 16·(s/2)..
// Needs `cols % 32 == 0`; rows and tokens past the matrix are clamped on load and skipped on
// store.
template <typename Tag>
kernel void gemmf_t(device const uchar* W [[buffer(0)]],
                    device const float* x [[buffer(1)]],
                    device float* y [[buffer(2)]],
                    constant GemmParams& p [[buffer(3)]],
                    uint2 tgpig [[threadgroup_position_in_grid]],
                    ushort tiitg [[thread_index_in_threadgroup]],
                    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half shmem[4096];                    // 8 KiB
    threadgroup half* sa = shmem;                    // 32 tiles [k block][row block], [k][row]
    threadgroup half* sb = shmem + 2048;             // 16 tiles [k block][token block], [token][k]

    const uint r0 = tgpig.y * GEMM_BM;
    const uint r1 = tgpig.x * GEMM_BN;
    const short nr0 = min((uint)GEMM_BM, p.rows - r0);
    const short nr1 = min((uint)GEMM_BN, p.n - r1);
    const short lr0 = min((short)(tiitg / 2), (short)(nr0 - 1));  // weight row this thread loads
    const short lr1 = min((short)(tiitg / 4), (short)(nr1 - 1));  // token this thread loads
    const short il0 = tiitg % 2;                                   // which 16 of the 32 k
    const short ib4 = tiitg % 4;                                   // which 8 of the 32 k

    device const uchar* wrow = W + (ulong)(r0 + lr0) * p.row_bytes;
    device const float* xrow = x + (ulong)(r1 + lr1) * p.cols + 8 * ib4;

    // Destinations in the blocked layouts.
    const short arow = tiitg / 2;
    threadgroup half* adst = sa + (8 * (2 * il0) + arow / 8) * 64 + arow % 8;
    const short btok = tiitg / 4;
    threadgroup half* bdst = sb + (4 * ib4 + btok / 8) * 64 + 8 * (btok % 8);

    simdgroup_float8x8 mc[8];
    device float* C = y + (ulong)(r1 + 16 * (sgitg >> 1)) * p.rows + r0 + 32 * (sgitg & 1);
    const bool full = nr0 == GEMM_BM && nr1 == GEMM_BN;
    if (p.accumulate != 0 && full) {
        for (short i = 0; i < 8; i++) {
            simdgroup_load(mc[i], C + 8 * (i % 4) + 8 * p.rows * (i / 4), p.rows, 0, false);
        }
    } else {
        for (short i = 0; i < 8; i++) mc[i] = simdgroup_float8x8(0.0f);
    }

    for (uint k0 = 0; k0 < p.cols; k0 += GEMM_BK) {
        float tmp[16];
        dq16(Tag(), wrow, k0 + 16 * il0, tmp);
        const float4 xa = *(device const float4*)(xrow + k0);
        const float4 xb = *(device const float4*)(xrow + k0 + 4);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        // A tile (k block 2·il0 + i/8, row block arow/8): element [k % 8][row % 8].
        for (short i = 0; i < 16; i++) {
            adst[(i / 8) * 8 * 64 + 8 * (i % 8)] = half(tmp[i]);
        }
        *(threadgroup half4*)(bdst) = half4(xa);
        *(threadgroup half4*)(bdst + 4) = half4(xb);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < GEMM_BK / 8; ik++) {
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mb[2];
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (full) {
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * p.rows * (i / 4), p.rows, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // [token][row] staging, 32 × 64 floats = the whole 8 KiB.
        threadgroup float* st = (threadgroup float*)shmem + 32 * (sgitg & 1) + 16 * (sgitg >> 1) * GEMM_BM;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], st + 8 * (i % 4) + 8 * GEMM_BM * (i / 4), GEMM_BM, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const float* all = (threadgroup const float*)shmem;
        for (uint idx = tiitg; idx < GEMM_BM * GEMM_BN; idx += GEMM_THREADS) {
            const uint t = idx / GEMM_BM;
            const uint r = idx % GEMM_BM;
            if (r < (uint)nr0 && t < (uint)nr1) {
                device float* dst = y + (ulong)(r1 + t) * p.rows + r0 + r;
                *dst = (p.accumulate != 0 ? *dst : 0.0f) + all[t * GEMM_BM + r];
            }
        }
    }
}

#define GEMMF_INSTANCE(name, Tag) \
    template [[host_name(name)]] kernel void gemmf_t<Tag>(device const uchar*, device const float*, device float*, constant GemmParams&, uint2, ushort, ushort);

GEMMF_INSTANCE("gemmf_f32", TagF32)
GEMMF_INSTANCE("gemmf_f16", TagF16)
GEMMF_INSTANCE("gemmf_q4_0", TagQ4_0)
GEMMF_INSTANCE("gemmf_q8_0", TagQ8_0)
GEMMF_INSTANCE("gemmf_q4_k", TagQ4_K)
GEMMF_INSTANCE("gemmf_q5_k", TagQ5_K)
GEMMF_INSTANCE("gemmf_q6_k", TagQ6_K)
