# Phase 7: GPU/CPU split on a simulated 16 GB Mac (M4 Max, 64 GB)

gpt-oss-20b MXFP4 (11.28 GiB of weights, MoE) through LLMario, suite `benchmarks/suites/decode.toml`
(one prose prompt, 256 tokens), unique prompts, one request at a time, 3 runs, `--no-quality`.
The scratch home's config simulates a 16 GB Mac's GPU limit:

```toml
[runtime]
gpu_memory_limit_gb = 10.67   # a 16 GB Mac's default GPU working set
memory_profile = "small"

[backends.llamacpp]
offload = "auto"
```

The planner keeps the MoE experts of 8 of 24 layers (3.16 GiB) in RAM for the CPU
(`--n-cpu-moe 8`). Development machine, not 16 GB hardware.

| Run | Loading | Estimate | Peak | Decode | Load at start |
|---|---|---:|---:|---:|---:|
| [044929Z](20261008T044929Z-gpt-oss-20b-gguf-mxfp4.md) | memory-mapped (llama.cpp default) | 13.29 GiB | **14.75 GiB** | 69.5 tok/s | 4.2 |
| [060948Z](20261008T060948Z-gpt-oss-20b-gguf-mxfp4.md) | `--load-mode none` (LLMario now) | 13.29 GiB | 12.56 GiB | 70.7 tok/s | 5.0 |

With `offload` off, the same configuration refuses the model (13.29 GiB over the 10.67 GiB GPU
limit). All on the GPU with no limit (Phase 6, standard profile): 105.6 tok/s, peak 11.51 GiB.

## Read before comparing

- **The first run exceeded its estimate.** Memory-mapped, Metal maps the whole file as one GPU
  buffer (11.5 GiB, over the 10.67 GiB limit the split was meant to respect), and the CPU keeps a
  repacked copy of its experts on top. Direct `llama-server` runs with the same split:
  memory-mapped 65.8 tok/s and peak 14.74 GiB; memory-mapped with `--no-repack` 59.6 tok/s and
  11.71 GiB; `--load-mode none` 74.1 tok/s, 12.50 GiB and a 7.53 GiB GPU buffer. LLMario now
  passes `--load-mode none` for every split (commit 31fcea5), and the second run is that build.
- **Cold start:** 36.7 s and 4.7 s. Loading the weights took 35.8 s memory-mapped and 3.6 s with
  `--load-mode none` (engine log). The file sits on an external SSD and whether it was in the OS
  file cache for each run was not checked, so this is not a clean comparison.
- **Speed prediction:** the CPU's 404 MiB per token at 15% of 546 GB/s gives 70.0 tok/s, against
  70.7 measured.
- **Swap:** this machine has 64 GB, so these runs cannot show swapping. On a real 16 GB Mac the
  plan's budget is 14 GiB (RAM minus 2 GiB of headroom), and a 12.6 GiB peak leaves little for
  other apps. The run on 16 GB hardware is still to do.
