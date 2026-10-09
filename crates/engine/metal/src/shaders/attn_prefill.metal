// LLMario native engine: Metal kernels, part 5 (prefill attention with simdgroup matrices).
//
// Flash-attention structure: one threadgroup per (block of FA_BQ queries, head); each of the four
// simdgroups owns 8 query rows and walks the keys in blocks of FA_BK: S = Q·Kᵀ (8×32, f16 inputs,
// f32 accumulate) straight from the f16 cache, an fp32 online softmax on the scores in threadgroup
// memory (causal mask: key ≤ query position), O = diag(corr)·O + P·V with P in f16. Head dims 64
// and 128 (hdv == hd); other geometries use the per-query kernels in misc.metal.
#define FA_BQ 32
#define FA_BK 32

template <int HD>
kernel void attn_prefill_t(device const float* q [[buffer(0)]],
                           device const half* K [[buffer(1)]],
                           device const half* V [[buffer(2)]],
                           device float* out [[buffer(3)]],
                           constant AttnParams& p [[buffer(5)]],
                           uint3 tgpig [[threadgroup_position_in_grid]],
                           ushort tiitg [[thread_index_in_threadgroup]],
                           ushort tiisg [[thread_index_in_simdgroup]],
                           ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int NF = HD / 8;
    threadgroup half sq[FA_BQ * HD];
    threadgroup float ss[ATTN_NSG * 8 * FA_BK];
    threadgroup half sp[ATTN_NSG * 8 * FA_BK];
    threadgroup float sdiag[ATTN_NSG * 64];

    const uint q0 = tgpig.x * FA_BQ;
    const uint h = tgpig.y;
    const uint kvh = h / (p.n_head / p.n_kv_head);
    const uint ostride = p.n_head * HD;

    // Q tile (scaled) as f16.
    for (uint i = tiitg; i < FA_BQ * HD; i += 128) {
        const uint r = i / HD;
        const uint d = i % HD;
        const uint qi = q0 + r;
        const float v = (qi < p.n_q) ? q[((ulong)qi * p.n_head + h) * HD + d] * p.scale : 0.0f;
        sq[i] = half(v);
    }
    for (uint i = tiitg; i < ATTN_NSG * 64; i += 128) sdiag[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* my_ss = ss + sgitg * 8 * FA_BK;
    threadgroup half* my_sp = sp + sgitg * 8 * FA_BK;
    threadgroup float* my_diag = sdiag + sgitg * 64;

    simdgroup_half8x8 Qf[NF];
    for (int i = 0; i < NF; i++) simdgroup_load(Qf[i], sq + (sgitg * 8) * HD + i * 8, HD);
    simdgroup_float8x8 O[NF];
    for (int i = 0; i < NF; i++) O[i] = simdgroup_float8x8(0.0f);

    const uint r = tiisg / 4;          // this lane's query row within the simdgroup's 8
    const uint c0 = (tiisg % 4) * 8;   // this lane's 8 score columns
    const uint qi = q0 + sgitg * 8 + r;
    const uint qpos = p.pos0 + qi;     // keys ≤ qpos are visible
    float m = -INFINITY;
    float l = 0.0f;

    device const half* Kb = K + p.k_off + kvh * HD;
    device const half* Vb = V + p.v_off + kvh * HD;
    const uint kv_end = p.pos0 + q0 + sgitg * 8 + 8; // keys needed by this simdgroup's last row

    for (uint kb = 0; kb < kv_end; kb += FA_BK) {
        // S = Q Kᵀ
        simdgroup_float8x8 S[4];
        for (int j = 0; j < 4; j++) S[j] = simdgroup_float8x8(0.0f);
        for (int j = 0; j < 4; j++) {
            device const half* kp = Kb + (ulong)(kb + j * 8) * p.kv_dim;
            for (int i = 0; i < NF; i++) {
                simdgroup_half8x8 Kf;
                simdgroup_load(Kf, kp + i * 8, p.kv_dim, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S[j], Qf[i], Kf, S[j]);
            }
        }
        for (int j = 0; j < 4; j++) simdgroup_store(S[j], my_ss + j * 8, FA_BK);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax on this lane's 8 scores of row r.
        float sv[8];
        float mx = -INFINITY;
        for (int c = 0; c < 8; c++) {
            const uint key = kb + c0 + c;
            sv[c] = (key <= qpos) ? my_ss[r * FA_BK + c0 + c] : -INFINITY;
            mx = max(mx, sv[c]);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        const float m_new = max(m, mx);
        const float corr = (m_new == -INFINITY) ? 1.0f : exp(m - m_new);
        float sum = 0.0f;
        for (int c = 0; c < 8; c++) {
            const float pr = (sv[c] == -INFINITY) ? 0.0f : exp(sv[c] - m_new);
            my_sp[r * FA_BK + c0 + c] = half(pr);
            sum += pr;
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        l = l * corr + sum;
        m = m_new;
        if ((tiisg % 4) == 0) my_diag[r * 8 + r] = corr;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(corr) · O + P · V
        simdgroup_float8x8 D;
        simdgroup_load(D, my_diag, 8);
        for (int i = 0; i < NF; i++) simdgroup_multiply(O[i], D, O[i]);
        for (int j = 0; j < 4; j++) {
            simdgroup_half8x8 Pf;
            simdgroup_load(Pf, my_sp + j * 8, FA_BK);
            device const half* vp = Vb + (ulong)(kb + j * 8) * p.v_dim;
            for (int i = 0; i < NF; i++) {
                simdgroup_half8x8 Vf;
                simdgroup_load(Vf, vp + i * 8, p.v_dim);
                simdgroup_multiply_accumulate(O[i], Pf, Vf, O[i]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // O / l per row, then store rows q0 + sgitg*8 .. +8 (the out buffer is padded to FA_BQ rows).
    if ((tiisg % 4) == 0) my_diag[r * 8 + r] = (l > 0.0f) ? 1.0f / l : 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 D;
    simdgroup_load(D, my_diag, 8);
    for (int i = 0; i < NF; i++) simdgroup_multiply(O[i], D, O[i]);
    device float* op = out + (ulong)(q0 + sgitg * 8) * ostride + h * HD;
    for (int i = 0; i < NF; i++) simdgroup_store(O[i], op + i * 8, ostride);
}

template [[host_name("attn_prefill_hd64")]] kernel void attn_prefill_t<64>(device const float*, device const half*, device const half*, device float*, constant AttnParams&, uint3, ushort, ushort, ushort);
template [[host_name("attn_prefill_hd128")]] kernel void attn_prefill_t<128>(device const float*, device const half*, device const half*, device float*, constant AttnParams&, uint3, ushort, ushort, ushort);
