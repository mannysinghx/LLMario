// LLMario native engine: Metal kernels, part 5 (prefill attention with simdgroup matrices).
//
// Flash-attention structure: one threadgroup per (block of FA_BQ queries, head) of one
// sequence; each of the four simdgroups owns 8 query rows and walks the keys in tiles of FA_BK:
// S = Q·Kᵀ (8×32, f16 inputs, f32 accumulate), an fp32 online softmax on the scores in
// threadgroup memory (causal mask: key ≤ query position), O = diag(corr)·O + P·V with P in f16.
// Keys come from the paged cache (misc.metal, `KvPage`): a tile of 32 keys never straddles a
// block because blocks hold a multiple of 32 positions and tiles start at multiples of 32.
// f16 rows are read straight from the block; q8_0 rows are dequantised tile by tile into
// threadgroup memory by all four simdgroups together (K first, then V in the same buffer).
// The q/out buffers start at this sequence's first row (bound with an offset); `tab` starts at
// this sequence's table. Rows past `n_q` are padding: never stored.
// Head dims 64 and 128 (hdv == hd); other geometries use the per-query kernels.
#define FA_BQ 32
#define FA_BK 32

template <int HD, int KVT>
kernel void attn_prefill_t(device const float* q [[buffer(0)]],
                           device const ulong* tab [[buffer(1)]],
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
    // q8_0 only: one dequantised 32-key tile (K, then V).
    threadgroup half skv[KVT == KVT_Q8_0 ? FA_BK * HD : 1];

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

    // Keys this simdgroup needs (its last row's position + 1), never past the last cached one;
    // the q8_0 path walks the threadgroup's range so every simdgroup reaches every barrier.
    const uint n_ctx = p.pos0 + p.n_q;
    const uint kv_end = (KVT == KVT_Q8_0) ? min(p.pos0 + q0 + FA_BQ, n_ctx)
                                          : min(p.pos0 + q0 + sgitg * 8 + 8, n_ctx);
    const uint k_stride = (KVT == KVT_Q8_0) ? HD : p.kv.k_row / 2;
    const uint v_stride = (KVT == KVT_Q8_0) ? HD : p.kv.v_row / 2;

    for (uint kb = 0; kb < kv_end; kb += FA_BK) {
        if (KVT == KVT_Q8_0) {
            // Dequantise K rows kb..kb+31 of head kvh (HD/32 blocks per row) into skv.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tiitg; i < FA_BK * (HD / 32); i += 128) {
                const uint rr = i / (HD / 32);
                const uint b = i % (HD / 32);
                device const uchar* blk =
                    kv_row_ptr(tab, kb + rr, p.kv.k_base, p.kv.k_row, p.kv) + (kvh * (HD / 32) + b) * Q8_BYTES;
                const float d = float(*reinterpret_cast<device const half*>(blk));
                device const char* qs = reinterpret_cast<device const char*>(blk + 2);
                for (int j = 0; j < 32; j++) skv[rr * HD + b * 32 + j] = half(d * float(qs[j]));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // S = Q Kᵀ
        simdgroup_float8x8 S[4];
        for (int j = 0; j < 4; j++) S[j] = simdgroup_float8x8(0.0f);
        for (int j = 0; j < 4; j++) {
            for (int i = 0; i < NF; i++) {
                simdgroup_half8x8 Kf;
                if (KVT == KVT_Q8_0) {
                    simdgroup_load(Kf, skv + (j * 8) * HD + i * 8, k_stride, ulong2(0, 0), true);
                } else {
                    device const half* kp = reinterpret_cast<device const half*>(
                        kv_row_ptr(tab, kb + j * 8, p.kv.k_base, p.kv.k_row, p.kv)) + kvh * HD;
                    simdgroup_load(Kf, kp + i * 8, k_stride, ulong2(0, 0), true);
                }
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

        if (KVT == KVT_Q8_0) {
            // Every simdgroup is done with the K tile; replace it with the V tile.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tiitg; i < FA_BK * (HD / 32); i += 128) {
                const uint rr = i / (HD / 32);
                const uint b = i % (HD / 32);
                device const uchar* blk =
                    kv_row_ptr(tab, kb + rr, p.kv.v_base, p.kv.v_row, p.kv) + (kvh * (HD / 32) + b) * Q8_BYTES;
                const float d = float(*reinterpret_cast<device const half*>(blk));
                device const char* qs = reinterpret_cast<device const char*>(blk + 2);
                for (int j = 0; j < 32; j++) skv[rr * HD + b * 32 + j] = half(d * float(qs[j]));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // O = diag(corr) · O + P · V
        simdgroup_float8x8 D;
        simdgroup_load(D, my_diag, 8);
        for (int i = 0; i < NF; i++) simdgroup_multiply(O[i], D, O[i]);
        for (int j = 0; j < 4; j++) {
            simdgroup_half8x8 Pf;
            simdgroup_load(Pf, my_sp + j * 8, FA_BK);
            for (int i = 0; i < NF; i++) {
                simdgroup_half8x8 Vf;
                if (KVT == KVT_Q8_0) {
                    simdgroup_load(Vf, skv + (j * 8) * HD + i * 8, v_stride);
                } else {
                    device const half* vp = reinterpret_cast<device const half*>(
                        kv_row_ptr(tab, kb + j * 8, p.kv.v_base, p.kv.v_row, p.kv)) + kvh * HD;
                    simdgroup_load(Vf, vp + i * 8, v_stride);
                }
                simdgroup_multiply_accumulate(O[i], Pf, Vf, O[i]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // O / l per row, then store this simdgroup's rows that are real queries.
    if ((tiisg % 4) == 0) my_diag[r * 8 + r] = (l > 0.0f) ? 1.0f / l : 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 D;
    simdgroup_load(D, my_diag, 8);
    for (int i = 0; i < NF; i++) simdgroup_multiply(O[i], D, O[i]);
    const uint row0 = q0 + sgitg * 8;
    if (row0 >= p.n_q) return;
    device float* op = out + (ulong)row0 * ostride + h * HD;
    if (row0 + 8 <= p.n_q) {
        for (int i = 0; i < NF; i++) simdgroup_store(O[i], op + i * 8, ostride);
    } else {
        // A partial group of rows: stage each 8×8 tile and copy only the real rows.
        const uint rows = p.n_q - row0;
        for (int i = 0; i < NF; i++) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(O[i], my_ss, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = tiisg; e < 64; e += 32) {
                const uint rr = e / 8;
                const uint cc = e % 8;
                if (rr < rows) op[(ulong)rr * ostride + i * 8 + cc] = my_ss[e];
            }
        }
    }
}

#define ATTN_PREFILL_INSTANCE(name, HD, KVT) \
    template [[host_name(name)]] kernel void attn_prefill_t<HD, KVT>(device const float*, device const ulong*, device float*, constant AttnParams&, uint3, ushort, ushort, ushort);

ATTN_PREFILL_INSTANCE("attn_prefill_hd64_f16", 64, KVT_F16)
ATTN_PREFILL_INSTANCE("attn_prefill_hd128_f16", 128, KVT_F16)
ATTN_PREFILL_INSTANCE("attn_prefill_hd64_q8_0", 64, KVT_Q8_0)
ATTN_PREFILL_INSTANCE("attn_prefill_hd128_q8_0", 128, KVT_Q8_0)
