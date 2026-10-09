// LLMario native engine: Metal kernels, part 1 (block layouts and dequantisation helpers).
//
// Compiled at runtime from this source by the OS Metal compiler; no Xcode is needed. Block layouts
// are written from `crates/engine/core/src/dequant.rs` (the scalar reference these kernels must
// match), which in turn transcribes ggml-common.h (MIT).

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define QK_K 256

// Block structs: byte-exact with the file layout (alignment 2, no padding).
struct block_q4_0 { half d; uchar qs[16]; };                                  // 18 bytes
struct block_q8_0 { half d; char qs[32]; };                                   // 34 bytes
struct block_q4_K { half d; half dmin; uchar scales[12]; uchar qs[128]; };    // 144 bytes
struct block_q5_K { half d; half dmin; uchar scales[12]; uchar qh[32]; uchar qs[128]; }; // 176 bytes
struct block_q6_K { uchar ql[128]; uchar qh[64]; char scales[16]; half d; };  // 210 bytes

// Type tags for the templated GEMM / embedding kernels.
struct TagF32 {};
struct TagF16 {};
struct TagQ4_0 {};
struct TagQ8_0 {};
struct TagQ4_K {};
struct TagQ5_K {};
struct TagQ6_K {};

// 6-bit scale/min unpacking for Q4_K / Q5_K (dequant.rs `get_scale_min_k4`).
static inline void get_scale_min_k4(int j, device const uchar* q, thread uchar& d, thread uchar& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Dequantise 16 consecutive elements starting at element `e0` (a multiple of 16) of one row.
// Used by the GEMM tile loader and the embedding gather. `row` points at the row's first byte.

static inline void dq16(TagF32, device const uchar* row, uint e0, thread float* o) {
    device const float* f = (device const float*)row + e0;
    for (int j = 0; j < 16; j++) o[j] = f[j];
}

static inline void dq16(TagF16, device const uchar* row, uint e0, thread float* o) {
    device const half* f = (device const half*)row + e0;
    for (int j = 0; j < 16; j++) o[j] = float(f[j]);
}

static inline void dq16(TagQ4_0, device const uchar* row, uint e0, thread float* o) {
    device const uchar* b = row + (e0 / 32) * 18;
    const float d = float(*(device const half*)b);
    device const packed_uchar4* qs = (device const packed_uchar4*)(b + 2);
    const bool hi = (e0 & 31) != 0;
    for (int k = 0; k < 4; k++) {
        const uchar4 q = uchar4(qs[k]);
        const float4 v = float4(int4(hi ? (q >> 4) : (q & 0xF)) - 8) * d;
        o[k * 4 + 0] = v.x; o[k * 4 + 1] = v.y; o[k * 4 + 2] = v.z; o[k * 4 + 3] = v.w;
    }
}

static inline void dq16(TagQ8_0, device const uchar* row, uint e0, thread float* o) {
    device const uchar* b = row + (e0 / 32) * 34;
    const float d = float(*(device const half*)b);
    device const packed_char4* qs = (device const packed_char4*)(b + 2 + (e0 & 31));
    for (int k = 0; k < 4; k++) {
        const float4 v = float4(char4(qs[k])) * d;
        o[k * 4 + 0] = v.x; o[k * 4 + 1] = v.y; o[k * 4 + 2] = v.z; o[k * 4 + 3] = v.w;
    }
}

static inline void dq16(TagQ4_K, device const uchar* row, uint e0, thread float* o) {
    device const block_q4_K* b = (device const block_q4_K*)row + e0 / QK_K;
    const uint r = e0 % QK_K;
    const int s = r / 32;
    const uint l0 = r % 32;
    uchar sc, mn;
    get_scale_min_k4(s, b->scales, sc, mn);
    const float dl = float(b->d) * float(sc);
    const float ml = float(b->dmin) * float(mn);
    const uint4 qq = *(device const uint4*)(b->qs + (s / 2) * 32 + l0);
    const uchar4 q[4] = {as_type<uchar4>(qq.x), as_type<uchar4>(qq.y), as_type<uchar4>(qq.z), as_type<uchar4>(qq.w)};
    const bool hi = (s & 1) != 0;
    for (int k = 0; k < 4; k++) {
        const float4 v = dl * float4(hi ? (q[k] >> 4) : (q[k] & 0xF)) - ml;
        o[k * 4 + 0] = v.x; o[k * 4 + 1] = v.y; o[k * 4 + 2] = v.z; o[k * 4 + 3] = v.w;
    }
}

static inline void dq16(TagQ5_K, device const uchar* row, uint e0, thread float* o) {
    device const block_q5_K* b = (device const block_q5_K*)row + e0 / QK_K;
    const uint r = e0 % QK_K;
    const int s = r / 32;
    const uint l0 = r % 32;
    uchar sc, mn;
    get_scale_min_k4(s, b->scales, sc, mn);
    const float dl = float(b->d) * float(sc);
    const float ml = float(b->dmin) * float(mn);
    const uint4 qq = *(device const uint4*)(b->qs + (s / 2) * 32 + l0);
    const uint4 hh = *(device const uint4*)(b->qh + l0);
    const uchar4 q[4] = {as_type<uchar4>(qq.x), as_type<uchar4>(qq.y), as_type<uchar4>(qq.z), as_type<uchar4>(qq.w)};
    const uchar4 h[4] = {as_type<uchar4>(hh.x), as_type<uchar4>(hh.y), as_type<uchar4>(hh.z), as_type<uchar4>(hh.w)};
    const bool hi = (s & 1) != 0;
    const uchar sh = (uchar)s;
    for (int k = 0; k < 4; k++) {
        const uchar4 lo = hi ? (q[k] >> 4) : (q[k] & 0xF);
        const uchar4 qv = lo | (((h[k] >> sh) & 1) << 4);
        const float4 v = dl * float4(qv) - ml;
        o[k * 4 + 0] = v.x; o[k * 4 + 1] = v.y; o[k * 4 + 2] = v.z; o[k * 4 + 3] = v.w;
    }
}

static inline void dq16(TagQ6_K, device const uchar* row, uint e0, thread float* o) {
    device const uchar* b = row + (e0 / QK_K) * 210;
    const uint r = e0 % QK_K;
    const uint h = r / 128;
    const uint rr = r % 128;
    const uint quarter = rr / 32;
    const uint l0 = rr % 32;
    const float d = float(*(device const half*)(b + 208));
    device const char* sc = (device const char*)(b + 192 + h * 8);
    // 16 consecutive `l` values: l0..l0+16 → scale index l/16 is constant (l0 is 0 or 16).
    const float dsc = d * float(sc[l0 / 16 + 2 * quarter]);
    device const packed_uchar4* ql = (device const packed_uchar4*)(b + h * 64 + l0 + (quarter & 1) * 32);
    device const packed_uchar4* qh = (device const packed_uchar4*)(b + 128 + h * 32 + l0);
    const uchar shift = (uchar)(2 * quarter);
    const bool hi = quarter >= 2;
    for (int k = 0; k < 4; k++) {
        const uchar4 lq = uchar4(ql[k]);
        const uchar4 hq = uchar4(qh[k]);
        const uchar4 low = hi ? (lq >> 4) : (lq & 0xF);
        const int4 q = int4(low | (((hq >> shift) & 3) << 4)) - 32;
        const float4 v = dsc * float4(q);
        o[k * 4 + 0] = v.x; o[k * 4 + 1] = v.y; o[k * 4 + 2] = v.z; o[k * 4 + 3] = v.w;
    }
}

// Embedding gather: one thread per 16 elements; `tokens[t]` selects the row.
struct EmbedParams {
    uint cols;
    uint n_tokens;
    uint row_bytes;
};

template <typename Tag>
kernel void embed_t(device const uchar* W [[buffer(0)]],
                    device const uint* tokens [[buffer(1)]],
                    device float* out [[buffer(2)]],
                    constant EmbedParams& p [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
    const uint chunks = p.cols / 16;
    const uint t = gid / chunks;
    const uint c = gid % chunks;
    if (t >= p.n_tokens) return;
    float tmp[16];
    dq16(Tag(), W + (ulong)tokens[t] * p.row_bytes, c * 16, tmp);
    device float* o = out + t * p.cols + c * 16;
    for (int j = 0; j < 16; j++) o[j] = tmp[j];
}

template [[host_name("embed_f32")]] kernel void embed_t<TagF32>(device const uchar*, device const uint*, device float*, constant EmbedParams&, uint);
template [[host_name("embed_f16")]] kernel void embed_t<TagF16>(device const uchar*, device const uint*, device float*, constant EmbedParams&, uint);
template [[host_name("embed_q4_0")]] kernel void embed_t<TagQ4_0>(device const uchar*, device const uint*, device float*, constant EmbedParams&, uint);
template [[host_name("embed_q8_0")]] kernel void embed_t<TagQ8_0>(device const uchar*, device const uint*, device float*, constant EmbedParams&, uint);
template [[host_name("embed_q4_k")]] kernel void embed_t<TagQ4_K>(device const uchar*, device const uint*, device float*, constant EmbedParams&, uint);
template [[host_name("embed_q5_k")]] kernel void embed_t<TagQ5_K>(device const uchar*, device const uint*, device float*, constant EmbedParams&, uint);
template [[host_name("embed_q6_k")]] kernel void embed_t<TagQ6_K>(device const uchar*, device const uint*, device float*, constant EmbedParams&, uint);
