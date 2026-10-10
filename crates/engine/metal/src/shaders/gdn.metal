// LLMario native engine: Metal kernels, part 7 (Gated DeltaNet, the recurrent layers of the
// Qwen3.5 / Qwen3-Next hybrid family).
//
// Equations follow `crates/engine/model/src/gdn.rs` (the CPU reference): causal depthwise conv +
// SiLU over the per-sequence history, per-head L2 norm of q/k, decay g = A · softplus(a + dt_bias),
// β = sigmoid(b), the delta rule per value head over an fp32 [head_k][head_v] state, and the gated
// RMSNorm rms_norm(o, w) · silu(z). All recurrent math is fp32.
//
// Several sequences share one call: `segs[s] = (seq, first row, rows)` lists each sequence's
// contiguous rows, and `rs_tab[seq]` is the GPU address of that sequence's recurrent buffer, which
// holds per DeltaNet layer the conv history `[conv_dim][d_conv − 1]` (oldest first) followed by
// the state `[n_v_heads][head_k][head_v]`; a layer's part starts `off` floats in.

static inline float gdn_silu(float x) {
    return x / (1.0f + precise::exp(-x));
}

static inline float gdn_softplus(float x) {
    return x > 20.0f ? x : precise::log(1.0f + precise::exp(x));
}

struct GdnConvParams {
    uint conv_dim;
    uint off;  // float offset of this layer's conv history in a sequence's recurrent buffer
};

// One thread per (channel, sequence) walks the sequence's rows in order with the last DC − 1
// pre-activation inputs in registers, then leaves them in the history for the next call.
// `x`, `out`: `[rows][conv_dim]`; `w`: `[conv_dim][DC]`.
template <int DC>
kernel void gdn_conv_silu_t(device const float* x [[buffer(0)]],
                            device const float* w [[buffer(1)]],
                            device const ulong* rs_tab [[buffer(2)]],
                            device float* out [[buffer(3)]],
                            constant GdnConvParams& p [[buffer(4)]],
                            device const uint4* segs [[buffer(5)]],
                            uint2 gid [[thread_position_in_grid]]) {
    const uint ch = gid.x;
    if (ch >= p.conv_dim) return;
    const uint4 sg = segs[gid.y];
    constexpr int HIST = DC - 1;
    device float* st = reinterpret_cast<device float*>(rs_tab[sg.x]) + p.off + (ulong)ch * HIST;
    float win[DC];
    float wc[DC];
    for (int j = 0; j < HIST; j++) win[j] = st[j];
    for (int j = 0; j < DC; j++) wc[j] = w[(ulong)ch * DC + j];
    for (uint r = sg.y; r < sg.y + sg.z; r++) {
        win[HIST] = x[(ulong)r * p.conv_dim + ch];
        float acc = 0.0f;
        for (int j = 0; j < DC; j++) acc += win[j] * wc[j];
        out[(ulong)r * p.conv_dim + ch] = gdn_silu(acc);
        for (int j = 0; j < HIST; j++) win[j] = win[j + 1];
    }
    for (int j = 0; j < HIST; j++) st[j] = win[j];
}

template [[host_name("gdn_conv_silu_k4")]] kernel void gdn_conv_silu_t<4>(device const float*, device const float*, device const ulong*, device float*, constant GdnConvParams&, device const uint4*, uint2);

// ---- The gated delta rule (gdn.rs equations 3–6): one threadgroup per (value head, sequence),
// that sequence's rows in order. The DV value columns are spread over DV·GDN_R threads; thread
// (j, r) keeps rows [r·DK/GDN_R, (r+1)·DK/GDN_R) of state column j in registers for the whole
// call, so the state is read and written once per call. Per row:
//
//   q̂, k̂  = L2-normalised q/k of QK head h % n_k (threadgroup memory; q̂ also scaled by 1/√DK)
//   decay  = exp(A[h] · softplus(alpha[t][h] + dt_bias[h])),  β = sigmoid(beta[t][h])
//   S[i][j] ← S[i][j] · decay;  kv[j] = Σ_i S[i][j] k̂[i]          (partial per r, simd shuffles)
//   Δ[j] = (v[t][h][j] − kv[j]) · β
//   S[i][j] ← S[i][j] + k̂[i] Δ[j];  o[j] = Σ_i S[i][j] q̂[i]
//   out[t][h][j] = o[j] / sqrt(mean_j o² + eps) · w[j] · silu(z[t][h][j])
//
// `qkv` is the conv output `[rows][conv_dim]` = `[q (key_dim) | k (key_dim) | v (value_dim)]`;
// `alpha`, `beta`: `[rows][n_v_heads]`; `z`, `out`: `[rows][value_dim]`.
#define GDN_R 4

struct GdnScanParams {
    uint n_k_heads;
    uint n_v_heads;
    uint conv_dim;
    uint key_dim;
    uint value_dim;
    uint off;  // float offset of this layer's state in a sequence's recurrent buffer
    float eps;
    float scale;
};

template <int DK, int DV>
kernel void gdn_scan_t(device const float* qkv [[buffer(0)]],
                       device const float* alpha [[buffer(1)]],
                       device const float* beta [[buffer(2)]],
                       device const float* z [[buffer(3)]],
                       device const float* A [[buffer(4)]],
                       device const float* dt_bias [[buffer(5)]],
                       device const float* w [[buffer(6)]],
                       device const ulong* rs_tab [[buffer(7)]],
                       device float* out [[buffer(8)]],
                       constant GdnScanParams& p [[buffer(9)]],
                       device const uint4* segs [[buffer(10)]],
                       uint2 tgpig [[threadgroup_position_in_grid]],
                       ushort tiitg [[thread_index_in_threadgroup]],
                       ushort tiisg [[thread_index_in_simdgroup]],
                       ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int NT = DV * GDN_R;      // threads per threadgroup
    constexpr int NSG = NT / 32;        // simdgroups
    constexpr int ROWS = DK / GDN_R;    // state rows per thread
    constexpr int COLS_PER_SG = 32 / GDN_R;
    threadgroup float qs[DK];
    threadgroup float ks[DK];
    threadgroup float red_q[NSG];
    threadgroup float red_k[NSG];
    threadgroup float red_o[NSG];

    const uint h = tgpig.x;
    const uint4 sg = segs[tgpig.y];
    const uint kh = h % p.n_k_heads;
    const int jj = tiisg % COLS_PER_SG;
    const int r = tiisg / COLS_PER_SG;
    const int j = sgitg * COLS_PER_SG + jj;   // value column of this thread
    const int row0 = r * ROWS;

    device float* S = reinterpret_cast<device float*>(rs_tab[sg.x]) + p.off + (ulong)h * DK * DV;
    float s[ROWS];
    for (int ii = 0; ii < ROWS; ii++) s[ii] = S[(ulong)(row0 + ii) * DV + j];

    const float a_h = A[h];
    const float dtb = dt_bias[h];
    const float w_j = w[j];

    for (uint t = sg.y; t < sg.y + sg.z; t++) {
        // L2-normalise q and k of QK head kh into threadgroup memory.
        float qv = 0.0f;
        float kv = 0.0f;
        if (tiitg < DK) {
            device const float* row = qkv + (ulong)t * p.conv_dim;
            qv = row[kh * DK + tiitg];
            kv = row[p.key_dim + kh * DK + tiitg];
        }
        const float sq = simd_sum(qv * qv);
        const float sk = simd_sum(kv * kv);
        if (tiisg == 0 && sgitg < DK / 32) {
            red_q[sgitg] = sq;
            red_k[sgitg] = sk;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiitg < DK) {
            float tq = 0.0f;
            float tk = 0.0f;
            for (int g = 0; g < DK / 32; g++) {
                tq += red_q[g];
                tk += red_k[g];
            }
            qs[tiitg] = qv * (1.0f / precise::sqrt(tq + p.eps)) * p.scale;
            ks[tiitg] = kv * (1.0f / precise::sqrt(tk + p.eps));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float decay = precise::exp(a_h * gdn_softplus(alpha[(ulong)t * p.n_v_heads + h] + dtb));
        const float bt = 1.0f / (1.0f + precise::exp(-beta[(ulong)t * p.n_v_heads + h]));

        // S ← S·decay; kv = Sᵀ k̂ (this thread's rows, then across the GDN_R row slices).
        float kvp = 0.0f;
        for (int ii = 0; ii < ROWS; ii++) {
            s[ii] *= decay;
            kvp += s[ii] * ks[row0 + ii];
        }
        kvp += simd_shuffle_xor(kvp, COLS_PER_SG);
        kvp += simd_shuffle_xor(kvp, 2 * COLS_PER_SG);
        const float vj = qkv[(ulong)t * p.conv_dim + 2 * p.key_dim + h * DV + j];
        const float delta = (vj - kvp) * bt;

        // S ← S + k̂ Δᵀ; o = Sᵀ q̂.
        float op = 0.0f;
        for (int ii = 0; ii < ROWS; ii++) {
            s[ii] += ks[row0 + ii] * delta;
            op += s[ii] * qs[row0 + ii];
        }
        op += simd_shuffle_xor(op, COLS_PER_SG);
        op += simd_shuffle_xor(op, 2 * COLS_PER_SG);

        // Gated RMSNorm over the DV columns: y = o / sqrt(mean(o²) + eps) · w · silu(z).
        const float so = simd_sum(r == 0 ? op * op : 0.0f);
        if (tiisg == 0) red_o[sgitg] = so;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (r == 0) {
            float tot = 0.0f;
            for (int g = 0; g < NSG; g++) tot += red_o[g];
            const float sc = 1.0f / precise::sqrt(tot / float(DV) + p.eps);
            const ulong oi = (ulong)t * p.value_dim + h * DV + j;
            out[oi] = op * sc * w_j * gdn_silu(z[oi]);
        }
    }

    for (int ii = 0; ii < ROWS; ii++) S[(ulong)(row0 + ii) * DV + j] = s[ii];
}

template [[host_name("gdn_scan_128")]] kernel void gdn_scan_t<128, 128>(device const float*, device const float*, device const float*, device const float*, device const float*, device const float*, device const float*, device const ulong*, device float*, constant GdnScanParams&, device const uint4*, uint2, ushort, ushort, ushort);
