# llmario-engine-cpu

CPU backend of the native engine: quantised matmul kernels, the element-wise primitives and the
worker pool. This file documents the matmul kernel sets (`src/simd.rs`, `src/simd/`); the public
API is in `src/lib.rs` (`QMat`, `matvec`, `matmul`, `ThreadPool`).

## Kernel sets

| set      | where               | activations                                  | types                                                |
|----------|---------------------|----------------------------------------------|------------------------------------------------------|
| `scalar` | `simd.rs`           | f32 (weights dequantised per row)            | everything the reference dequantizer supports        |
| `int8`   | `simd/int8.rs`      | Q8_K (256-element types), Q8_0 (32-element)  | Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q4_K Q5_K Q6_K F16 BF16 F32 |
| `neon`   | `simd/neon.rs`      | as `int8`                                    | all of the above, plus two-row Q4_K / Q6_K / Q8_0    |
| `avx2`   | `simd/avx2.rs`      | as `int8`                                    | Q4_0 Q8_0 Q4_K Q6_K (others fall back to `int8`)     |

`simd::detect()` picks `neon` on aarch64 with FEAT_DotProd, `avx2` on x86_64 with AVX2 + F16C,
`int8` otherwise. `LLMARIO_CPU_KERNELS=scalar|int8|neon|avx2` overrides (an unavailable set falls
back to detection with a warning). `Kernels::name` reports the active set.

The recipe is ggml's quantised-activation one, reimplemented from the block layouts: the
activation row is quantised once per op (`quantize_row_q8_k` / `quantize_row_q8_0` from the core
crate), each weight row is consumed in its native block layout with exact integer dot products,
and the block deltas scale the result in f32 (Q4_K/Q5_K: `Σ_blocks d·d8·Σ_s scale_s·dot_s −
dmin·d8·Σ_s min_s·bsum_s`; Q6_K: `Σ_blocks d·d8·Σ_g scale_g·Σ(q−32)·a`).

**Bit-exactness.** `int8` fixes the integer sums (exact) and the order of every f32 operation
(no FMA; 32-element types accumulate block `b` into lane `b % 4`, reduced as
`(l0+l1)+(l2+l3)`; float types use a 4×4 lane grid). `neon` and `avx2` perform the same f32
operations in the same order, so their outputs equal `int8`'s bit for bit; the tests assert it.

**Stable toolchain workarounds (aarch64).** `vdotq_s32` and the f16 conversion intrinsics are
nightly-only on Rust 1.92, so `sdot` is emitted through `asm!` (one instruction, `#[target_feature]`
gated) and f16 → f32 uses an exact integer-domain conversion verified for all 65536 patterns.

## Drivers (`simd/common.rs`)

- `matvec`: quantise the activation once, then contiguous row chunks on the pool (a few per
  thread, ≥ 16 rows each); two-row kernels where the set provides them.
- `matmul`: quantise all `n` activation rows once (in parallel), then per row chunk walk the
  tokens in blocks of 32 so a token block's Q8_K rows and the weight row stay in L1; every weight
  row is read from memory once per token block.

## Tests

`cargo test -p llmario-engine-cpu` (0.2 s): every (type, set) against `scalar` with a tolerance
derived from the activation-quantisation model; a strong oracle (`W · Q(x)` in f64 from the
dequantised quantised activation, matched to 2e-5); `neon` vs `int8` bit-exact; edge shapes
(1 block, odd rows, tails of 1–3 blocks, 1/2/7/64 tokens, float lengths that are not multiples
of 16); page-boundary over-read detection; the f16 conversion sweep.

Gated extras:

- `LLMARIO_TEST_GGUF=<file>` — real Q4_K / Q6_K rows from a GGUF: reports RMS and max error
  against the f32 reference and the ratio of measured to predicted quantisation error.
- `LLMARIO_BENCH=1 cargo test -p llmario-engine-cpu --release -- --nocapture --test-threads=1 bench_`
  — µs/call and GB/s per shape and thread count, a Qwen3-1.7B-shaped decode step (≈ 1.15 GB per
  token), prefill GMAC/s, and calibrations (achievable read bandwidth, pool round trip,
  quantisation cost, kernel cycles per block).

## Measurements (Apple M4 Max, 12 performance cores, shared and heavily loaded machine)

Calibration: a plain 1 GB streaming read reaches ~280–290 GB/s with 4 or more threads (the
546 GB/s SoC figure is the fabric peak shared with the GPU; the CPU cluster cannot reach it).
That is the decode bound. Pool round trip 3–6 µs at 12 threads; Q8_K quantisation 1.5 µs for 2048
columns.

Kernel-only, one core, cache-resident (cycles per super-block at 4.4 GHz):

| type | `int8` | `neon` | `neon` two-row | GB/s per core (`neon`) |
|------|-------:|-------:|---------------:|-----------------------:|
| Q4_K |   72   |  18.8  |      18.1      |            34–35       |
| Q6_K |   61   |  30.9  |      31.2      |            30          |
| Q5_K |   86   |  35.9  |       –        |            22          |
| Q8_0 |   4.6  |   3.0  |       2.5      |            50–60       |
| Q4_0 |   7.7  |   3.9  |       –        |            20          |

Multi-thread numbers were taken with other sessions' builds and system daemons running (load
average 8–75), so they are lower bounds; see the report accompanying the change for the table.
Single-thread Q4_K / Q6_K matvec streams at 26–27 / 21 GB/s; a Q4_K 6144×2048 matvec took 58–63 µs
at 12 threads (110–120 GB/s) under that load. Prefill (`matmul`, 6144×2048, n = 512, 12 threads)
reached 290 GMAC/s (Q4_K) and 370 GMAC/s (Q6_K) in the quietest run.

## Known limits and follow-ups

- The AVX2 set is compile- and clippy-checked for `x86_64-apple-darwin` only; it has not been
  executed (no x86 machine here).
- No i8mm (`smmla`) 4×4 prefill tile yet; `matmul` is the row-dot loop with L1 token blocking.
- `pool.rs` (owned by the lead) has a lost wake-up between `Drop` and a worker about to park
  (`stop` is stored and `notify_all` called without holding the `job` mutex); it hung one
  benchmark run that creates and drops pools in a loop.
