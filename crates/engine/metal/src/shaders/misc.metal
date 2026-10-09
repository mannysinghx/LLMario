// LLMario native engine: Metal kernels, part 4 (norms, fused QK-norm/RoPE/KV-store, decode
// attention, element-wise ops).
//
// Semantics follow `crates/engine/cpu/src/ops.rs` and `crates/engine/model/src/forward.rs`.

// ---- RMSNorm: one threadgroup of 256 threads (8 simdgroups) per row; `cols % 4 == 0`.
#define NORM_TG 256

struct NormParams {
    uint cols;
    float eps;
};

kernel void rms_norm(device const float* x [[buffer(0)]],
                     device const float* w [[buffer(1)]],
                     device float* y [[buffer(2)]],
                     constant NormParams& p [[buffer(3)]],
                     uint tg [[threadgroup_position_in_grid]],
                     ushort tiitg [[thread_index_in_threadgroup]],
                     ushort tiisg [[thread_index_in_simdgroup]],
                     ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[NORM_TG / 32];
    x += (ulong)tg * p.cols;
    y += (ulong)tg * p.cols;
    float ss = 0.0f;
    for (uint i = tiitg * 4; i < p.cols; i += NORM_TG * 4) {
        const float4 v = *(device const float4*)(x + i);
        ss += dot(v, v);
    }
    ss = simd_sum(ss);
    if (tiisg == 0) part[sgitg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int g = 0; g < NORM_TG / 32; g++) tot += part[g];
    const float scale = 1.0f / precise::sqrt(tot / float(p.cols) + p.eps);
    for (uint i = tiitg * 4; i < p.cols; i += NORM_TG * 4) {
        const float4 v = *(device const float4*)(x + i);
        const float4 wv = *(device const float4*)(w + i);
        *(device float4*)(y + i) = v * scale * wv;
    }
}

// ---- Fused per-head QK-norm + RoPE + KV store. One simdgroup (32 threads) per (token, head):
// heads 0..n_head are Q (normalised and rotated in place), the next n_kv are K (normalised,
// rotated, written to the f16 cache) and the last n_kv are V (written to the cache).
struct QkRopeParams {
    uint n_tokens;
    uint n_head;
    uint n_kv_head;
    uint hd;
    uint hdv;
    uint rot_dim;
    uint mode;      // 0 = normal (adjacent pairs), 1 = neox (halves)
    uint pos0;
    uint k_off;     // element offset of this layer in the K cache
    uint v_off;
    uint kv_dim;
    uint v_dim;
    uint q_norm;    // 1 when `qn` holds the per-head Q norm weight
    uint k_norm;
    uint rope;      // 0 on NoPE layers
    float eps;
    float theta;
    float freq_scale;
    float attn_factor;
};

kernel void qk_rope_kv(device float* q [[buffer(0)]],
                       device float* k [[buffer(1)]],
                       device const float* v [[buffer(2)]],
                       device const float* qn [[buffer(3)]],
                       device const float* kn [[buffer(4)]],
                       device half* kc [[buffer(5)]],
                       device half* vc [[buffer(6)]],
                       constant QkRopeParams& p [[buffer(7)]],
                       uint tg [[threadgroup_position_in_grid]],
                       ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup float hb[256];
    const uint per_tok = p.n_head + 2 * p.n_kv_head;
    const uint t = tg / per_tok;
    const uint hh = tg % per_tok;
    if (t >= p.n_tokens) return;
    const uint pos = p.pos0 + t;

    if (hh >= p.n_head + p.n_kv_head) {
        // V: plain copy into the cache.
        const uint h = hh - p.n_head - p.n_kv_head;
        device const float* src = v + (ulong)t * p.v_dim + h * p.hdv;
        device half* dst = vc + (ulong)p.v_off + (ulong)pos * p.v_dim + h * p.hdv;
        for (uint i = tiisg; i < p.hdv; i += 32) dst[i] = half(src[i]);
        return;
    }
    const bool is_q = hh < p.n_head;
    const uint h = is_q ? hh : hh - p.n_head;
    device float* src = is_q ? q + ((ulong)t * p.n_head + h) * p.hd : k + ((ulong)t * p.n_kv_head + h) * p.hd;
    device const float* nw = is_q ? qn : kn;
    const bool norm = is_q ? (p.q_norm != 0) : (p.k_norm != 0);

    // Normalise into threadgroup memory (or copy when the layer has no QK-norm).
    float ss = 0.0f;
    for (uint i = tiisg; i < p.hd; i += 32) ss += src[i] * src[i];
    ss = simd_sum(ss);
    const float scale = norm ? 1.0f / precise::sqrt(ss / float(p.hd) + p.eps) : 1.0f;
    for (uint i = tiisg; i < p.hd; i += 32) hb[i] = norm ? src[i] * scale * nw[i] : src[i];
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Rotate pairs (lane handles pair indices tiisg, tiisg + 32, ...), copy the rest.
    const uint half_rot = p.rope != 0 ? p.rot_dim / 2 : 0;
    device half* kdst = kc + (ulong)p.k_off + (ulong)pos * p.kv_dim + h * p.hd;
    for (uint i = tiisg; i < half_rot; i += 32) {
        const float freq = precise::pow(p.theta, -(2.0f * float(i)) / float(p.rot_dim));
        const float angle = float(pos) * p.freq_scale * freq;
        const float s = precise::sin(angle) * p.attn_factor;
        const float c = precise::cos(angle) * p.attn_factor;
        const uint a = (p.mode == 0) ? 2 * i : i;
        const uint b = (p.mode == 0) ? 2 * i + 1 : i + half_rot;
        const float x0 = hb[a];
        const float x1 = hb[b];
        const float ra = x0 * c - x1 * s;
        const float rb = x0 * s + x1 * c;
        if (is_q) {
            src[a] = ra;
            src[b] = rb;
        } else {
            src[a] = ra;
            src[b] = rb;
            kdst[a] = half(ra);
            kdst[b] = half(rb);
        }
    }
    for (uint i = 2 * half_rot + tiisg; i < p.hd; i += 32) {
        src[i] = hb[i];
        if (!is_q) kdst[i] = half(hb[i]);
    }
}

// ---- Decode attention over the cache: one threadgroup (4 simdgroups) per (query, head, split).
// The threadgroup's key range is cut into four contiguous sub-ranges, one per simdgroup; each
// simdgroup scores ATTN_KB keys at a time (independent loads and reductions in flight) and keeps
// an fp32 online softmax. Lane `l` owns head dimensions [l·ND, l·ND + ND) so a key row is one
// contiguous 2·hd-byte load across the simdgroup. Simdgroups are merged through threadgroup memory;
// with `n_split > 1` the (o, m, l) partials go to `part` and `attn_reduce` merges them.
#define ATTN_MAX_HD 256
#define ATTN_NSG 4
#define ATTN_KB 8

struct AttnParams {
    uint n_q;
    uint n_head;
    uint n_kv_head;
    uint hd;
    uint hdv;
    uint kv_dim;
    uint v_dim;
    uint pos0;
    uint k_off;
    uint v_off;
    uint n_split;
    uint split_len;
    float scale;
};

template <int ND>
static inline void load_nd(device const half* p, thread float* o) {
    for (int i = 0; i < ND; i++) o[i] = float(p[i]);
}
template <>
inline void load_nd<2>(device const half* p, thread float* o) {
    const half2 v = *(device const half2*)p;
    o[0] = v.x; o[1] = v.y;
}
template <>
inline void load_nd<4>(device const half* p, thread float* o) {
    const half4 v = *(device const half4*)p;
    o[0] = v.x; o[1] = v.y; o[2] = v.z; o[3] = v.w;
}
template <>
inline void load_nd<8>(device const half* p, thread float* o) {
    const half4 v0 = *(device const half4*)p;
    const half4 v1 = *(device const half4*)(p + 4);
    o[0] = v0.x; o[1] = v0.y; o[2] = v0.z; o[3] = v0.w;
    o[4] = v1.x; o[5] = v1.y; o[6] = v1.z; o[7] = v1.w;
}

// Merge the four simdgroups' (m, l, o) and write the result (or the split partial).
static inline void attn_finish(threadgroup float* sm, threadgroup float* sl, threadgroup float* so,
                               device float* out, device float* part, constant AttnParams& p,
                               uint t, uint h, uint sp, ushort tiisg, ushort sgitg) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg != 0) return;
    float M = sm[0];
    for (uint g = 1; g < ATTN_NSG; g++) M = max(M, sm[g]);
    float w[ATTN_NSG];
    float L = 0.0f;
    for (uint g = 0; g < ATTN_NSG; g++) {
        w[g] = (sm[g] == -INFINITY) ? 0.0f : exp(sm[g] - M);
        L += w[g] * sl[g];
    }
    const bool empty = (M == -INFINITY);
    const uint ndv = (p.hdv + 31) / 32;
    if (p.n_split == 1) {
        device float* dst = out + ((ulong)t * p.n_head + h) * p.hdv;
        const float inv = empty ? 0.0f : 1.0f / L;
        for (uint i = 0; i < ndv; i++) {
            const uint d = tiisg + i * 32;
            if (d < p.hdv) {
                float v = 0.0f;
                for (uint g = 0; g < ATTN_NSG; g++) v += w[g] * so[g * ATTN_MAX_HD + d];
                dst[d] = v * inv;
            }
        }
    } else {
        device float* dst = part + (((ulong)t * p.n_head + h) * p.n_split + sp) * (p.hdv + 2);
        for (uint i = 0; i < ndv; i++) {
            const uint d = tiisg + i * 32;
            if (d < p.hdv) {
                float v = 0.0f;
                for (uint g = 0; g < ATTN_NSG; g++) v += w[g] * so[g * ATTN_MAX_HD + d];
                dst[d] = v;
            }
        }
        if (tiisg == 0) {
            dst[p.hdv] = M;
            dst[p.hdv + 1] = empty ? 0.0f : L;
        }
    }
}

template <int ND>
kernel void attn_vec_t(device const float* q [[buffer(0)]],
                       device const half* K [[buffer(1)]],
                       device const half* V [[buffer(2)]],
                       device float* out [[buffer(3)]],
                       device float* part [[buffer(4)]],
                       constant AttnParams& p [[buffer(5)]],
                       uint3 tgpig [[threadgroup_position_in_grid]],
                       ushort tiisg [[thread_index_in_simdgroup]],
                       ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sm[ATTN_NSG];
    threadgroup float sl[ATTN_NSG];
    threadgroup float so[ATTN_NSG * ATTN_MAX_HD];

    const uint t = tgpig.x;
    const uint h = tgpig.y;
    const uint sp = tgpig.z;
    const uint hd = ND * 32;
    const uint kvh = h / (p.n_head / p.n_kv_head);
    const uint n_pos = p.pos0 + t + 1;
    const uint p_begin = sp * p.split_len;
    const uint p_end = min(n_pos, p_begin + p.split_len);
    // Contiguous sub-range per simdgroup.
    const uint span = (p_end > p_begin) ? (p_end - p_begin + ATTN_NSG - 1) / ATTN_NSG : 0;
    const uint r0 = p_begin + sgitg * span;
    const uint r1 = min(p_end, r0 + span);

    float qr[ND];
    {
        device const float* qp = q + ((ulong)t * p.n_head + h) * hd + tiisg * ND;
        for (int i = 0; i < ND; i++) qr[i] = qp[i] * p.scale;
    }
    float o[ND];
    for (int i = 0; i < ND; i++) o[i] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    device const half* Kb = K + p.k_off + kvh * hd + tiisg * ND;
    device const half* Vb = V + p.v_off + kvh * hd + tiisg * ND;
    for (uint base = r0; base < r1; base += ATTN_KB) {
        float s[ATTN_KB];
        for (int j = 0; j < ATTN_KB; j++) {
            const uint pos = base + j;
            float acc = 0.0f;
            if (pos < r1) {
                float kr[ND];
                load_nd<ND>(Kb + (ulong)pos * p.kv_dim, kr);
                for (int i = 0; i < ND; i++) acc += qr[i] * kr[i];
            }
            s[j] = acc;
        }
        for (int j = 0; j < ATTN_KB; j++) s[j] = simd_sum(s[j]);
        float mx = -INFINITY;
        for (int j = 0; j < ATTN_KB; j++) {
            if (base + j < r1) mx = max(mx, s[j]);
        }
        const float m_new = max(m, mx);
        const float corr = exp(m - m_new);
        l *= corr;
        for (int i = 0; i < ND; i++) o[i] *= corr;
        for (int j = 0; j < ATTN_KB; j++) {
            const uint pos = base + j;
            if (pos < r1) {
                const float pr = exp(s[j] - m_new);
                l += pr;
                float vr[ND];
                load_nd<ND>(Vb + (ulong)pos * p.v_dim, vr);
                for (int i = 0; i < ND; i++) o[i] += pr * vr[i];
            }
        }
        m = m_new;
    }

    if (tiisg == 0) {
        sm[sgitg] = m;
        sl[sgitg] = l;
    }
    for (int i = 0; i < ND; i++) so[sgitg * ATTN_MAX_HD + tiisg * ND + i] = o[i];
    attn_finish(sm, sl, so, out, part, p, t, h, sp, tiisg, sgitg);
}

#define ATTN_VEC_INSTANCE(name, ND) \
    template [[host_name(name)]] kernel void attn_vec_t<ND>(device const float*, device const half*, device const half*, device float*, device float*, constant AttnParams&, uint3, ushort, ushort);

ATTN_VEC_INSTANCE("attn_vec_hd32", 1)
ATTN_VEC_INSTANCE("attn_vec_hd64", 2)
ATTN_VEC_INSTANCE("attn_vec_hd128", 4)
ATTN_VEC_INSTANCE("attn_vec_hd256", 8)

// Generic fallback for any hd/hdv ≤ 256 (lane `l` owns dimensions l, l + 32, ...; one key at a time).
kernel void attn_vec_generic(device const float* q [[buffer(0)]],
                             device const half* K [[buffer(1)]],
                             device const half* V [[buffer(2)]],
                             device float* out [[buffer(3)]],
                             device float* part [[buffer(4)]],
                             constant AttnParams& p [[buffer(5)]],
                             uint3 tgpig [[threadgroup_position_in_grid]],
                             ushort tiisg [[thread_index_in_simdgroup]],
                             ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sm[ATTN_NSG];
    threadgroup float sl[ATTN_NSG];
    threadgroup float so[ATTN_NSG * ATTN_MAX_HD];

    const uint t = tgpig.x;
    const uint h = tgpig.y;
    const uint sp = tgpig.z;
    const uint kvh = h / (p.n_head / p.n_kv_head);
    const uint n_pos = p.pos0 + t + 1;
    const uint p_begin = sp * p.split_len;
    const uint p_end = min(n_pos, p_begin + p.split_len);
    const uint nd = (p.hd + 31) / 32;
    const uint ndv = (p.hdv + 31) / 32;

    float qr[ATTN_MAX_HD / 32];
    for (uint i = 0; i < nd; i++) {
        const uint d = tiisg + i * 32;
        qr[i] = d < p.hd ? q[((ulong)t * p.n_head + h) * p.hd + d] * p.scale : 0.0f;
    }
    float o[ATTN_MAX_HD / 32];
    for (uint i = 0; i < ATTN_MAX_HD / 32; i++) o[i] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    device const half* Kb = K + p.k_off + kvh * p.hd;
    device const half* Vb = V + p.v_off + kvh * p.hdv;
    for (uint pos = p_begin + sgitg; pos < p_end; pos += ATTN_NSG) {
        float s = 0.0f;
        device const half* kr = Kb + (ulong)pos * p.kv_dim;
        for (uint i = 0; i < nd; i++) {
            const uint d = tiisg + i * 32;
            if (d < p.hd) s += qr[i] * float(kr[d]);
        }
        s = simd_sum(s);
        const float m_new = max(m, s);
        const float corr = exp(m - m_new);
        const float pr = exp(s - m_new);
        l = l * corr + pr;
        device const half* vr = Vb + (ulong)pos * p.v_dim;
        for (uint i = 0; i < ndv; i++) {
            const uint d = tiisg + i * 32;
            if (d < p.hdv) o[i] = o[i] * corr + pr * float(vr[d]);
        }
        m = m_new;
    }

    if (tiisg == 0) {
        sm[sgitg] = m;
        sl[sgitg] = l;
    }
    for (uint i = 0; i < ndv; i++) {
        const uint d = tiisg + i * 32;
        if (d < p.hdv) so[sgitg * ATTN_MAX_HD + d] = o[i];
    }
    attn_finish(sm, sl, so, out, part, p, t, h, sp, tiisg, sgitg);
}

// Merge split-K partials: one thread per (token, head, dim).
kernel void attn_reduce(device const float* part [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant AttnParams& p [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    const uint total = p.n_q * p.n_head * p.hdv;
    if (gid >= total) return;
    const uint th = gid / p.hdv;
    const uint d = gid % p.hdv;
    device const float* base = part + (ulong)th * p.n_split * (p.hdv + 2);
    float M = -INFINITY;
    for (uint s = 0; s < p.n_split; s++) M = max(M, base[s * (p.hdv + 2) + p.hdv]);
    float num = 0.0f, den = 0.0f;
    for (uint s = 0; s < p.n_split; s++) {
        device const float* ps = base + s * (p.hdv + 2);
        const float ms = ps[p.hdv];
        if (ms == -INFINITY) continue;
        const float w = exp(ms - M);
        num += w * ps[d];
        den += w * ps[p.hdv + 1];
    }
    out[gid] = den > 0.0f ? num / den : 0.0f;
}

// ---- Element-wise.
struct ElemParams {
    uint n;
    uint rows; // for add_bias: row length
};

// gate[i] = silu(gate[i]) * up[i]
kernel void swiglu(device float* gate [[buffer(0)]],
                   device const float* up [[buffer(1)]],
                   constant ElemParams& p [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    const float g = gate[gid];
    gate[gid] = g / (1.0f + precise::exp(-g)) * up[gid];
}

// x[i] += y[i]
kernel void add(device float* x [[buffer(0)]],
                device const float* y [[buffer(1)]],
                constant ElemParams& p [[buffer(2)]],
                uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    x[gid] += y[gid];
}

// y[t * rows + r] += bias[r]
kernel void add_bias(device float* y [[buffer(0)]],
                     device const float* bias [[buffer(1)]],
                     constant ElemParams& p [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    y[gid] += bias[gid % p.rows];
}
