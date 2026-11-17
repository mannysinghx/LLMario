# CPU baseline and before/after, 2026-11-16

Machine: Apple M4 Max (12 performance + 4 efficiency cores), 64 GB, macOS 27.0.1, AC power.
Models read from an external PCIe SSD (APFS). Native engine on the CPU (`--device cpu`, NEON
kernels), llama.cpp build 11146 strictly on the CPU (`-ngl 0 -dev none -nopo 1 -nkvo 1`).
"Before" is commit 7eeafe6 (`target/baseline`), "after" is this branch. Driver:
`scripts/engine/bench_cpu.py` (every figure below is in a `report.json` here);
`scripts/engine/bench_pressure.py` for the pressure table.

## Read this first: the machine was shared

Other programs ran throughout (an Ollama server holding an 18 GB model on the GPU, Chrome, a
Docker VM, an MLX server). For the first hour the machine was also short of memory (about 5 GiB
reclaimable, swap 40 of 42 GB used, pressure level "warning") and a pure-compute test scaled
only 6.4× over 12 processes. `baseline/qwen3-1.7b` was recorded in that state: its `raw` sweep
and its llama.cpp figures are far below what either engine does on a free machine and are kept
as a record of a busy desktop, not as a baseline for speed. From 21:40 the machine was free
again (11.2× over 12 processes, 34–44 GiB reclaimable); everything under `ab/` and the pressure
table were measured then, old and new builds interleaved round by round.

## Before and after (free machine, interleaved, CPU, 12 threads unless stated)

| Measurement | Model | Before | After | Change |
|---|---|---|---|---|
| Cold load (file not in the page cache) | Qwen3-1.7B Q4_K_M, 1.03 GiB | 0.537, 0.538, 0.544 s | 0.349, 0.373, 0.379 s | −32 % |
| | Qwen3.5-9B Q4_K_M, 5.29 GiB | 1.80, 1.82 s | 1.29, 1.31 s | −28 % |
| | Gemma 4 12B QAT Q4_0, 6.50 GiB | 2.48, 2.50 s | 1.59, 1.63 s | −35 % |
| | Qwen3-Coder-30B-A3B Q4_K_M, 17.3 GiB (MoE) | 0.89, 0.91 s | 0.91, 0.91 s | none (read-ahead is for dense models) |
| Warm load | all four | 0.12–0.35 s | 0.12–0.34 s | none |
| Decode at the engine's default thread count | Qwen3-1.7B (512-token prompt, 64 tokens) | 77–86 tok/s (16 threads) | 93–100 tok/s (12 threads) | +15–20 % |
| Prompt at the engine's default thread count | Qwen3-1.7B (512 tokens) | 269–273 tok/s | 246–249 tok/s | −9 % |
| Decode, same thread count | Qwen3-1.7B, 8 / 12 threads | 94–96 / 90–99 tok/s | 91–95 / 95–99 tok/s | none |
| | Qwen3.5-9B / Gemma 4 12B / Qwen3-Coder-30B | 32–34 / 18.6–18.8 / 41–49 | 33–34 / 19.1 / 49 | none |
| Prompt, same thread count | Qwen3-1.7B, 8 / 12 / 16 threads | 199–206 / 239–247 / 257–269 | 200–206 / 237–245 / 250–265 | none |
| Memory after load (physical footprint) | 1.7B / 9B / 12B / 30B | 75 / 174 / 167 / 82 MiB | same | none |
| Idle 3,159-token conversation while the system reports a memory warning | Qwen3-1.7B, f16 KV | 478 MiB held | 127 MiB (350 MiB freed, saved to disk) | −73 % |
| Next turn of that conversation | | 0.83 s (cache in memory) | 1.01 s (read back from disk) | +0.18 s |
| Correctness set (12 fixed questions, temperature 0) | Qwen3-1.7B | 10 of 12 | 10 of 12, identical answers | none |

The app starts the engine with 12 threads on this machine already, so the default-thread row
changes only an engine started by hand. The Qwen3-Coder raw prompt figure of the first round
(46 tok/s before, 97 after) is an order effect (the first run reads the experts from disk);
the second round is 97 against 94.

## Memory-pressure relief (Qwen3-1.7B, 3,159-token prompt, 8K context, one slot)

`scripts/engine/bench_pressure.py <gguf> 2800 1`, free machine, simulated pressure level.

| Configuration | After load | After turn 1 | After idling under pressure | Turn 2 | Prompt tokens reused |
|---|---|---|---|---|---|
| Before (no relief), disk tier on | 75 MiB | 478 MiB | 478 MiB | 0.83 s | 3,159 of 3,176 |
| Relief on, no pressure | 73 MiB | 475 MiB | 475 MiB | 0.85 s | 3,159 |
| Relief on, warning, disk tier on | 75 MiB | 476 MiB | 127 MiB | 1.01 s | 3,159 (read from disk) |
| Relief on, warning, no disk tier | 75 MiB | 477 MiB | 477 MiB (kept: it could not be saved) | 0.83 s | 3,159 |
| Relief on, critical, no disk tier | 74 MiB | 476 MiB | 128 MiB | 20.9 s | 0 (recomputed) |
| Static alternative: 8-bit KV, relief off | 75 MiB | 313 MiB | 313 MiB | 0.77 s | 3,159 |

Turn 1 took 20.7–21.5 s in every row (`pressure-relief-qwen3-1.7b.txt`). A first run of the
same script while the machine was busy gave the same memory figures with slower turns (turn 1
41–47 s, turn 2 2.2–2.7 s from memory or disk, 43.8 s recomputed).

## What a busy, memory-short desktop did (first hour, not a controlled test)

- Gemma 4 12B (6.5 GiB) with about 5 GiB reclaimable: loaded in 8.7 s, then read 13 GiB from
  disk to answer one 16-token question (first token after 12 s, 23 s in all): the weights did
  not stay in memory. The plan, made against 64 GB of installed memory, had no objection. This
  is what the new "paging" figure in `/engine/stats`, the plan note and the supervisor's
  "available right now" note are for.
- Read-ahead in that state made the cold load slower (11.3 s against 8.7 s, the file read
  twice), which is why it now runs only when the system has 1.25× the file size available.
- Qwen3-1.7B: llama.cpp's CPU path fell to 0.3–33 tok/s decode depending on the thread count
  (4 threads best, 16 worst); the native engine held 30–55 tok/s at every thread count from 4 to
  16. One machine, one afternoon: an observation, not a ranking.

## Baseline details (`baseline/qwen3-1.7b`, later part recorded on the freer machine)

Serve mode, engine default of 16 threads, three rounds: prompt 215–278 tok/s (176- and
644-token prompts), 104–154 tok/s at 2,371 tokens; decode 66–94 tok/s after the short prompts,
31–36 after 2,371 tokens (one run 9.7 under load); time to first token 0.64–0.68 s / 2.65–3.0 s
/ 15.4–22.9 s; average cores used 7.6–13.1 of 16; physical footprint 75 MiB after load,
121 / 224 / 484 MiB after the three prompts; resident size 1,137 MiB (weights mapped);
1,056 MiB read from disk on a cold load, 0 warm; 19 OS threads. Concurrent requests (12
threads): one request 85–98 tok/s; two 20–27 each; four 14–24 each; the total stays near one
request's speed, as expected for CPU decode.

## Not tested

- Windows and x86-64 (no AVX2 hardware here); Linux (the pressure source there is PSI and is
  covered by unit tests only).
- A real 8 or 16 GB machine. The planner results for small machines are plans, not runs.
- Metal: nothing in this change touches the GPU path; its figures were not re-measured.
- Read-ahead on an internal SSD or a hard disk.
- llama.cpp and MLX engines under memory pressure (relief is in LLMario's engine only).
