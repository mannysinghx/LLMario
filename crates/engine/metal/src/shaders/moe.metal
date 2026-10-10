// LLMario native engine: Metal kernels, part 6 (routed mixture-of-experts FFN).
//
// Same math as the CPU path (`crates/engine/model/src/moe.rs`, after llama.cpp build_moe_ffn for
// qwen3moe): softmax router, top-k experts in descending order (ties keep the lower expert id),
// weights renormalised with the sum clamped at the smallest normal f16, SwiGLU experts, outputs
// weighted and summed in selection order. A (row t, selection slot j) pair is `t * k + j`.
//
// - `moe_route`: one simdgroup per row picks the experts and their weights.
// - Few rows (decode): `gemv_glu_id_*` / `gemv_id_*` run one expert matvec per pair.
// - Many rows (prefill): `moe_group` lists the pairs of each expert, then `gemm_id_*` multiplies
//   every expert once over the rows that chose it (one threadgroup grid per expert).
// - `moe_combine`: x += Σ_j w_j · out_j.
// Expert e's matrix starts `e * expert_bytes` into the 3-D tensor.

#define MOE_MAX_EXPERTS 1024
#define MOE_GROUP_TG 256

struct MoeRouteParams {
    uint n_expert;
    uint k;
    uint norm;
    uint pad;
};

kernel void moe_route(device const float* logits [[buffer(0)]],
                      device uint* sel [[buffer(1)]],
                      device float* wout [[buffer(2)]],
                      constant MoeRouteParams& p [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]],
                      ushort lane [[thread_index_in_simdgroup]]) {
    device const float* l = logits + (ulong)row * p.n_expert;
    float m = -INFINITY;
    for (uint e = lane; e < p.n_expert; e += 32) m = max(m, l[e]);
    m = simd_max(m);
    float s = 0.0f;
    for (uint e = lane; e < p.n_expert; e += 32) s += precise::exp(l[e] - m);
    s = simd_sum(s);
    // Expert e belongs to lane e % 32, bit e / 32 of that lane's `taken` mask.
    uint taken = 0;
    float wsum = 0.0f;
    for (uint j = 0; j < p.k; j++) {
        float bv = -INFINITY;
        uint bi = 0xFFFFFFFFu;
        for (uint i = 0, e = lane; e < p.n_expert; i++, e += 32) {
            if ((taken >> i) & 1u) continue;
            const float v = l[e];
            if (v > bv || (v == bv && e < bi)) {
                bv = v;
                bi = e;
            }
        }
        const float mv = simd_max(bv);
        const uint best = simd_min(bv == mv ? bi : 0xFFFFFFFFu);
        if (best % 32 == lane) taken |= 1u << (best / 32);
        const float pj = precise::exp(mv - m) / s;
        wsum += pj;
        if (lane == 0) {
            sel[row * p.k + j] = best;
            wout[row * p.k + j] = pj;
        }
    }
    if (p.norm != 0 && lane == 0) {
        const float d = max(wsum, 6.103515625e-5f);
        for (uint j = 0; j < p.k; j++) wout[row * p.k + j] /= d;
    }
}

struct GemvIdParams {
    uint rows;          // rows of one expert's matrix
    uint cols;
    uint expert_bytes;  // bytes of one expert's matrix
    uint k;             // experts per row
    uint x_per_pair;    // 1: x has one row per pair; 0: one row per token (pair / k)
    uint pad0;
    uint pad1;
    uint pad2;
};

// y[pair] = W[sel[pair]] x (VARIANT 0) or silu(W x) * (W2 x) (VARIANT 2), one pair per grid row.
template <typename Tag, int VARIANT>
kernel void gemv_id_t(device const uchar* W [[buffer(0)]],
                      device const float* x [[buffer(1)]],
                      device float* y [[buffer(2)]],
                      constant GemvIdParams& p [[buffer(3)]],
                      device const uchar* W2 [[buffer(4)]],
                      device const uint* sel [[buffer(5)]],
                      uint2 tgpig [[threadgroup_position_in_grid]],
                      ushort tiisg [[thread_index_in_simdgroup]],
                      ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint pair = tgpig.y;
    const ulong eoff = (ulong)sel[pair] * p.expert_bytes;
    const uint xrow = p.x_per_pair != 0 ? pair : pair / p.k;
    device const float* xv = x + (ulong)xrow * p.cols;
    device float* yv = y + (ulong)pair * p.rows;
    const uint row0 = (tgpig.x * GEMV_NSG + sgitg) * GEMV_NR;
    float acc[GEMV_NR] = {0.0f, 0.0f};
    gemv_rows(Tag(), W + eoff, xv, row0, p.rows, p.cols, tiisg, acc);
    if (VARIANT == 2) {
        float acc2[GEMV_NR] = {0.0f, 0.0f};
        gemv_rows(Tag(), W2 + eoff, xv, row0, p.rows, p.cols, tiisg, acc2);
        for (int r = 0; r < GEMV_NR; r++) {
            const float g = simd_sum(acc[r]);
            const float u = simd_sum(acc2[r]);
            if (tiisg == 0 && row0 + r < p.rows) yv[row0 + r] = g / (1.0f + precise::exp(-g)) * u;
        }
    } else {
        for (int r = 0; r < GEMV_NR; r++) {
            const float s = simd_sum(acc[r]);
            if (tiisg == 0 && row0 + r < p.rows) yv[row0 + r] = s;
        }
    }
}

#define GEMV_ID_INSTANCE(name, Tag, V) \
    template [[host_name(name)]] kernel void gemv_id_t<Tag, V>(device const uchar*, device const float*, device float*, constant GemvIdParams&, device const uchar*, device const uint*, uint2, ushort, ushort);

GEMV_ID_INSTANCE("gemv_id_f32", TagF32, 0)
GEMV_ID_INSTANCE("gemv_id_f16", TagF16, 0)
GEMV_ID_INSTANCE("gemv_id_q4_0", TagQ4_0, 0)
GEMV_ID_INSTANCE("gemv_id_q8_0", TagQ8_0, 0)
GEMV_ID_INSTANCE("gemv_id_q4_k", TagQ4_K, 0)
GEMV_ID_INSTANCE("gemv_id_q5_k", TagQ5_K, 0)
GEMV_ID_INSTANCE("gemv_id_q6_k", TagQ6_K, 0)
GEMV_ID_INSTANCE("gemv_glu_id_f32", TagF32, 2)
GEMV_ID_INSTANCE("gemv_glu_id_f16", TagF16, 2)
GEMV_ID_INSTANCE("gemv_glu_id_q4_0", TagQ4_0, 2)
GEMV_ID_INSTANCE("gemv_glu_id_q8_0", TagQ8_0, 2)
GEMV_ID_INSTANCE("gemv_glu_id_q4_k", TagQ4_K, 2)
GEMV_ID_INSTANCE("gemv_glu_id_q5_k", TagQ5_K, 2)
GEMV_ID_INSTANCE("gemv_glu_id_q6_k", TagQ6_K, 2)

struct MoeGroupParams {
    uint n_pairs;
    uint n_expert;
    uint cap;  // slots per expert in `ids`
    uint pad;
};

// counts[e] = pairs that chose expert e; ids[e * cap + i] = those pairs (order within an expert
// is arbitrary: every output row is computed independently, so results do not depend on it).
kernel void moe_group(device const uint* sel [[buffer(0)]],
                      device uint* counts [[buffer(1)]],
                      device uint* ids [[buffer(2)]],
                      constant MoeGroupParams& p [[buffer(3)]],
                      ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint cnt[MOE_MAX_EXPERTS];
    for (uint e = tid; e < p.n_expert; e += MOE_GROUP_TG) {
        atomic_store_explicit(&cnt[e], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < p.n_pairs; i += MOE_GROUP_TG) {
        const uint e = sel[i];
        const uint slot = atomic_fetch_add_explicit(&cnt[e], 1u, memory_order_relaxed);
        ids[e * p.cap + slot] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < p.n_expert; e += MOE_GROUP_TG) {
        counts[e] = atomic_load_explicit(&cnt[e], memory_order_relaxed);
    }
}

struct GemmIdParams {
    uint rows;          // rows of one expert's matrix
    uint cols;
    uint row_bytes;
    uint expert_bytes;
    uint k;
    uint x_per_pair;
    uint cap;
    uint pad;
};

// For expert e = grid z: Y[pair] = W[e] · X[row of pair] for the pairs listed in ids[e], as the
// prefill GEMM (`gemm_t`) with the token rows gathered and the outputs scattered.
template <typename Tag>
kernel void gemm_id_t(device const uchar* W [[buffer(0)]],
                      device const float* x [[buffer(1)]],
                      device float* y [[buffer(2)]],
                      constant GemmIdParams& p [[buffer(3)]],
                      device const uint* counts [[buffer(4)]],
                      device const uint* ids [[buffer(5)]],
                      uint3 tgpig [[threadgroup_position_in_grid]],
                      ushort tiitg [[thread_index_in_threadgroup]],
                      ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint e = tgpig.z;
    const uint cnt = counts[e];
    const uint tok0 = tgpig.x * GEMM_BN;
    if (tok0 >= cnt) return;  // uniform across the threadgroup
    const uint row0 = tgpig.y * GEMM_BM;
    device const uint* list = ids + (ulong)e * p.cap;
    device const uchar* We = W + (ulong)e * p.expert_bytes;

    threadgroup half sa[GEMM_BM * GEMM_BK];
    threadgroup half sb[GEMM_BN * GEMM_BK];
    threadgroup float sc[GEMM_BM * GEMM_BN];

    simdgroup_float8x8 c[8];
    for (int i = 0; i < 8; i++) c[i] = simdgroup_float8x8(0.0f);

    const uint arow = tiitg / 2;
    const uint akseg = (tiitg % 2) * 16;
    const uint btok = tiitg / 4;
    const uint bkseg = (tiitg % 4) * 8;
    const bool bvalid = tok0 + btok < cnt;
    device const float* xrow = x;
    if (bvalid) {
        const uint pair = list[tok0 + btok];
        xrow = x + (ulong)(p.x_per_pair != 0 ? pair : pair / p.k) * p.cols;
    }

    for (uint k0 = 0; k0 < p.cols; k0 += GEMM_BK) {
        {
            const uint row = row0 + arow;
            const uint e0 = k0 + akseg;
            float tmp[16];
            if (row < p.rows && e0 < p.cols) {
                dq16(Tag(), We + (ulong)row * p.row_bytes, e0, tmp);
            } else {
                for (int j = 0; j < 16; j++) tmp[j] = 0.0f;
            }
            threadgroup half* dst = sa + arow * GEMM_BK + akseg;
            for (int j = 0; j < 16; j++) dst[j] = half(tmp[j]);
        }
        {
            const uint e0 = k0 + bkseg;
            threadgroup half* dst = sb + btok * GEMM_BK + bkseg;
            if (bvalid && e0 + 8 <= p.cols) {
                for (int j = 0; j < 8; j++) dst[j] = half(xrow[e0 + j]);
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

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            simdgroup_store(c[i * 4 + j], sc + (sgitg * 16 + i * 8) * GEMM_BN + j * 8, GEMM_BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tiitg; idx < GEMM_BM * GEMM_BN; idx += GEMM_THREADS) {
        const uint r = idx / GEMM_BN;
        const uint t = idx % GEMM_BN;
        if (row0 + r < p.rows && tok0 + t < cnt) {
            const uint pair = list[tok0 + t];
            y[(ulong)pair * p.rows + row0 + r] = sc[r * GEMM_BN + t];
        }
    }
}

#define GEMM_ID_INSTANCE(name, Tag) \
    template [[host_name(name)]] kernel void gemm_id_t<Tag>(device const uchar*, device const float*, device float*, constant GemmIdParams&, device const uint*, device const uint*, uint3, ushort, ushort);

GEMM_ID_INSTANCE("gemm_id_f32", TagF32)
GEMM_ID_INSTANCE("gemm_id_f16", TagF16)
GEMM_ID_INSTANCE("gemm_id_q4_0", TagQ4_0)
GEMM_ID_INSTANCE("gemm_id_q8_0", TagQ8_0)
GEMM_ID_INSTANCE("gemm_id_q4_k", TagQ4_K)
GEMM_ID_INSTANCE("gemm_id_q5_k", TagQ5_K)
GEMM_ID_INSTANCE("gemm_id_q6_k", TagQ6_K)

struct MoeCombineParams {
    uint n;  // rows
    uint d;
    uint k;
    uint pad;
};

// x[t] += Σ_j w[t·k + j] · out[t·k + j], summed in selection order.
kernel void moe_combine(device float* x [[buffer(0)]],
                        device const float* out [[buffer(1)]],
                        device const float* w [[buffer(2)]],
                        constant MoeCombineParams& p [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n * p.d) return;
    const uint t = gid / p.d;
    const uint i = gid % p.d;
    const ulong base = (ulong)t * p.k;
    float s = out[base * p.d + i] * w[base];
    for (uint j = 1; j < p.k; j++) s += out[(base + j) * p.d + i] * w[base + j];
    x[gid] += s;
}
