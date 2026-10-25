# Chart data

The numbers behind the charts in the main README; `python tools/make_charts.py` draws `docs/img/` from these files.
All runs: the box and the model build described in the main README, the `default` preset.

## decode-token.csv

One decode token on the GPU's timeline, averaged over 223 tokens in a 12 s window of Nsight Systems, short
context. The GPU's work is split four ways: kernels running while no expert copy runs, both at once, expert copies
(host to device) while no kernel runs, and nothing at all. `of_which_nvme_wait_ms` is the part of the idle time when no
copy was pending because the expert was still being read from NVMe. Under the profiler a token took 54 ms
(18.45 tok/s); without it, 49-50 ms. 23 expert slots per layer in VRAM, a 46 GiB RAM tier, 17 I/O threads.

| column | meaning |
|---|---|
| `segment` | the part of the token |
| `ms` | milliseconds per token |
| `of_which_nvme_wait_ms` | for `idle` only: waiting for an NVMe read |

## turn-layers.csv

The main pass of a warm agent turn under Nsight: turn 7 of the session in `agent-turns.csv`, 7,530 new tokens on top
of a 16.8K-token conversation; 7,526 rows go through all 40 layers (the server runs the last 4 tokens in a second
call). A layer's time is split at its expert waves (NVTX ranges of the waves carry the layer id): the part before the
waves (attention, the router, dense layers, norms and the previous layer's tail) and the waves (expert loads,
copies and compute). The whole turn took 13.36 s: 1.91 s before the first kernel (1.29 s of it reading Engram rows),
10.85 s in this pass, 0.34 s in the 4-token call, 0.26 s moving the output head back to VRAM. The profile predates two
changes that took most of the time before the first kernel away (the prompt's Engram rows now load under the compute
of the earlier layers, and the pinned host blocks of the compute buffer are kept between prompts); the pass itself is
the same.

| column | meaning |
|---|---|
| `layer` | 0-39; layers 20-39 are the decoder half |
| `outside_waves_ms` | wall time before the layer's waves |
| `outside_waves_gpu_ms` | GPU kernel time in that part |
| `waves_ms` | wall time of the layer's expert waves |
| `waves_waiting_ms` | of the waves: the graph waits for expert loads |
| `waves_compute_ms` / `waves_copies_ms` | of the waves: GPU kernels / expert copies (they overlap) |
| `waves` | number of waves |

## agent-turns.csv

An 8-turn agent session replayed against the server (32K context, batch 8192, 43 GiB RAM tier, 22 expert slots,
q8_0 KV cache, a copy of the expert files on a second NVMe, the patches' defaults), two runs `a` and `b`. Turn 1
starts cold; turns 2-8 reuse the cached conversation and read only their new tokens
(`python bench/agent_turns.py TAG` makes this session; two of its sentences were reworded for publishing, a few
tokens' difference).

| column | meaning |
|---|---|
| `turn`, `new_tokens` | the turn and the tokens it adds |
| `read_s_*` | time to read the new tokens (the server's prompt timing) |
| `wave_wait_s_*` | of it: waiting for expert loads |
| `answer_tok_s_*` | generation speed of the turn's answer |
| `nvme_read_gb_*` | bytes read from NVMe during the turn (prompt and answer) |

## drives.csv

Synthetic reads with diskspd in the pattern of the fork's expert reads: one unbuffered read of one expert slab of
the `cd` build (3,420,160 bytes, 512-byte aligned) at a time per thread, reads only, the given number in flight per
drive: the Gen5 drive that holds the model (12), the Gen4 drive with a copy of the expert parts (4), and the system
NVMe on a x2 link (4). Run together, no drive slows down next to the others; the server uses the first two
(`LLAMA_MOE_STREAM_MIRROR`), a third copy was not kept.

| column | meaning |
|---|---|
| `config` | which drives read together and how many reads each had in flight |
| `drive`, `reads_in_flight`, `gb_s` | each drive's share |
