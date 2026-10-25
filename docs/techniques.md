# Techniques

What we changed, why, and what each change measured. Labels are the names the patches use: **G** decode, **P**
prefill, **D** long context, **M** memory, **C** changes that cost quality. Each gain is against the configuration of
its day, on this box (RTX 5070 Ti 16 GB on PCIe 4.0, 64 GB RAM, Gen5 + Gen4 NVMe, Windows 11), so the percentages do
not add up. Switches and their defaults are in [switches.md](switches.md); the commit message of each patch has the
details of its measurement.

## The ladder: VRAM, RAM, NVMe

DeepSeek-V4.1-Flash routes each token to 6 of 384 experts in each of 40 layers: 240 expert choices per token, ~10 MB
each at ~2.4 bits. None of the 373 GB build fits; the experts stream.

- **VRAM expert cache** (JigSawPT's): ~23 slots per layer next to the dense part. Of the 240 choices of a decode
  token, ~160 hit VRAM, ~61 come from RAM, ~10 from the drive and ~10 are dropped (the drop rule below).
- **A pinned RAM tier** (`--moe-stream-l2`, JigSawPT's): we sized it to 43-46 GiB. Its ceiling is not Windows'
  needs but the WDDM budget of "shared GPU memory", which is RAM - 16 GB (48,245 MB on 64 GB); past ~46 GiB the
  pinning silently fails and decode drops 15-20% (M1, M6). The tier's slots are sized per expert size (M3: 17% more
  experts in 46 GiB, drive reads -14%, +2.2%).
- **Unbuffered NVMe reads, no memory mapping of the expert files.** An NTFS file that is also memory-mapped reads
  about 3x slower unbuffered; the loader no longer maps or prefetches the expert files (M5: load 32 s -> 12-16 s).
- **A copy of the expert files on a second NVMe** (P16): a read goes to the copy when the main drive has 4 reads in
  flight. The two drives together read 17.5-18.9 GB/s against 13.5 GB/s for one; cold prompts +11-22%. Decode does
  not gain: it waits for the latency of one critical read, not for bandwidth.
- **17 I/O threads** (were 9): agent turns read +2.8%.

## Decode

A decode token is ~49-50 ms. Under Nsight: the GPU computes 19 ms, copies experts from RAM 30 ms, does both at once
for only 3 ms, and idles 8 ms - 7 of them waiting for the drive. The bus is the wall: ~30 ms of copies caps decode at
~32 tok/s with perfect overlap.

### Off the critical path

- **G16, the router's mailbox:** the GPU posts the router's choices into pinned host memory; the host remaps the
  experts without a synchronizing device-to-host copy (+6.2%).
- **G22, the reply:** the host's answer (the VRAM slot of each expert) goes back the same way, read by the GPU at the
  next call: 43 -> 8 graph splits with CPU inputs, hand-offs 4.2 -> 1.9 ms per token (+2-5.5%).
- **G24, one wake-up:** the I/O thread pool is woken once after the remap loop instead of at every miss; the first
  copy starts 36 us earlier (+1.2%).
- **G26 / G23, Engram rows:** the token's Engram rows are faulted in on a helper thread while the graph is built, then
  gathered right before the layer that reads them (+2.2%), and the decode graph is reused between tokens (+1.7%).
- Together the host's part of a token went from ~7 ms to ~1 ms.

### Hiding the loads

- **G2, runahead** (lossless): a layer's expert remap is split in two. Between the halves an approximate pass - the
  experts already resident, the others at weight 0 - runs the next layer's attention and router and stages its top 6
  experts into VRAM; then the exact layer runs unchanged. It caught ~91% of the misses at 0.1 wrong guesses per call
  (+11% at the time). It costs the GPU ~8 ms of compute per token, so it is the default only on the exact path.
- **G31, split calls:** each MoE call computes the pairs whose experts are resident while the copies of the missing
  ones run (+2.8%).
- **G35, heavy experts ahead:** the router of layer L+2 applied to layer L's input - "the lens", nearly free because
  the GPU computes it in the mailbox - predicts the heavy experts that would come from the drive, and they are read
  into the RAM tier two layers early. +3-4% (+6.7% past 30K context); it gets ~55-60% of what a perfect predictor of
  the same experts would.
- **G29 / G30:** a CSA mask view no longer forces a synchronous copy in every layer (+3.5% at 30K), and the exact
  expert GEMMs reuse the approximate pass's outputs (+1.3% on the exact path). Both bit-exact.

### The drop rule (lossy)

An expert that is in neither VRAM nor the RAM tier and carries less than 14% of its token's routing weight is not
waited for: it gets weight 0 and is read into the tier for later. It was worth +8-9% over the exact path; it costs up
to +0.006 nat/token on our code-and-agent quality set (significant there, within error on the two general sets). It
is on in the `default` preset; `exact` turns it off.

### The fast preset (lossy)

- **C14, late experts:** up to one light pair per token (the drive's first) is not waited for; its copy is queued
  last and its weighted sum enters after the next layer's attention, which does not see it (+4.8-5.2%, about +0.005
  nat on general-100).
- **G40:** G35's lens stages up to one confident guess per call into VRAM two calls ahead (+3%). The math is
  untouched, but next to the drop rule ~5% more experts get dropped.
- On the lighter `qd` build the fast preset decodes 23-23.7 tok/s; 5 experts per token instead of 6 (`K5`, a separate
  switch) gives 24.7 for another +0.010-0.013 nat.

### Long context

- **D11, sparse attention in decode over a quantized KV cache:** the sparse path (top-512 indexer blocks plus the
  window) used to require an f16 cache; with q8_0 / q4_0 it now gathers the same rows. 64K +32-41%, 128K x2.0-2.1;
  decode stays at 16-20 tok/s from short context to 512K.

## Prefill

A prompt runs in micro-batches through all 40 layers; every expert a batch routes to is copied to the GPU in
"waves", and a long prompt touches nearly all 384 experts of every layer in every batch.

- **P1, the head yields:** the output head gives its VRAM to the compute buffers of a long prompt and is read back
  from its file afterwards (bit-exact).
- **P9, the tier keeps the prompt's experts:** the RAM tier no longer evicts experts the prompt has touched; a ring
  takes what it reads past the tier's holding. A 23K prompt +25.5%, drive reads 1,473 -> 745 GB.
- **P10, `-b` a multiple of `-ub`:** the server cut each chunk into 1536 + 512 at `-b 2048`: 64K +18.5%, drive -24%.
- **P11, a shared pool:** a long prompt takes the VRAM of the expert cache and the waves of all layers share a pool
  of 64 slots: 64K and 128K +51%, cold 2.2K / 8.9K +37% / +79%.
- **D10, sparse attention in prefill** as in decode (attention was 64% of the GPU's time at the end of a 128K
  document); batch 8192 up to 32K.
- **P4, encoder-only positions:** V4.1's decoder half (layers 20-39) reads nothing of a position but its window
  rings, so the positions of a call more than its reach before the first output run through the encoder only - half
  the experts, copies and compute of a long prompt. Bit-exact.
- **C7, YoCo truncation** (DeepInfer's default for V4.1; lossy, within error on our three quality sets): the decoder
  half computes only the last 140 positions. We skip its experts for the earlier rows (attention and dense layers
  still run on all of them): agent turns +13%, cold 2.2K / 8.9K +20% / +60%.
- **P8 and P20 for agent turns:** a prompt's Engram rows are faulted in layer by layer under the compute of the
  earlier layers instead of all before the first kernel (+5.6%; cold +9% / +6.3%), and pinned host blocks of 64 MiB
  and more are kept between prompts instead of 1.66 GiB being allocated and freed on every turn (+3.7%).
- **P17, a thinner compute buffer:** expert outputs are weighted in place (-215 MiB at batch 6144).
- **D12, 512K:** the dense prefill plan is reserved only for the positions where a sequence can take it, so 512K
  loads (a 451K-token document at 482 tok/s, answers 16-18 tok/s). 1M loads at batch 512 only.

Reading agent turns went from 333 tok/s (before C7) to ~418 tok/s on 25 October; a short turn still costs ~5 s, 2-3 s
of it waiting for expert loads.

## Quantization

- **The build (`cd`):** prometheusAIR's IQ2_XS gate/up and Q2_K down for the experts, ISTA-DASLab's Q5_K dense part.
  The lighter dense part frees 3.0 GiB of VRAM - 6-8 more expert slots per layer - so `cd` decodes +8.5% against our
  first plain 2-bit mix and is no farther from the API (pooled over three sets ~0.008 nat closer, within error).
- **`qd`** (IQ2_XXS gate/up, the same dense part): +16-18% decode for +0.004-0.005 nat - the fast preset's build.
- prometheusAIR's whole IQ2_XS experts (IQ3_XXS down) were 0.031 nat closer to the API (~28% of the Q2 -> Q4 gap)
  at -11-13% decode; gate/up alone gave most of it, hence `cd`.
- ISTA-DASLab's non-uniform GSQ-RCO experts lost to the plain mix on our API sets: +0.02 nat at "2.25" bits (+0.034
  on Russian prompts), +0.12 at "2.00".
- Across all the builds and lossy switches we measured, speed and quality trade at about 0.002 nat per 1% of decode.

## What did not work

| idea | result |
|---|---|
| read a slab in 2-8 parallel chunks | slower: 4 / 8 chunks 3.9 / 2.9 GB/s against 4.25 whole |
| better eviction policies | LRU 58%, LRFU 62%, the fork's own 62.1% hits; hotness halving every 16 tokens instead of 64 within noise |
| a better victim when evicting | 85% of misses are experts evicted earlier, but only 5.4% within one token: the cache is short of places, not of choice |
| prefetch a per-request expert profile | -3-5% drive reads; a perfect (Belady) cache would read 40-56% less, but that takes knowing the future |
| read the next layer's experts ahead during prefill | ≈0 even with a perfect oracle: the tier cannot hold a prompt's working set |
| "free" guesses in the bus's idle time | a wrong staging costs 0.18-0.26 ms wherever it goes: the bus has no idle windows in decode |
| speculative reads of dropped experts off the critical drive queue | within noise: decode waits for one heavy miss's latency, not for the queue |
| copies in the compute stream, inputs hoisted into one copy, `GGML_CUDA_GRAPH_OPT` | no gain: copy -> kernel costs ~31 us under WDDM either way |
| lossless compression of the 2-bit weights | at most 4.46% (zlib 0.8%) |
| computing RAM-resident experts on the 6-core CPU | a probe gave +3-5% at best; left for a bigger CPU |
| a direct NVMe -> GPU path, TCC / MCDM, WSL2, a bigger page file, power plans | not available on Windows with a GeForce, or no change |

## Where the time is left

Perfect expert prediction in this pipeline would give 27.5-29.3 tok/s against our 20.3; perfect staging into VRAM
two calls ahead, 27.4. The lens reaches ~76% precision at ~14% recall; staging pays off from about 65-70% precision
at 50% recall. See the main README's "Where we're stuck".
