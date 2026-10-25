# Environment switches

The patches read 116 environment variables. Most are measurement knobs: the defaults are the configuration we
measured best. Each switch is documented in the source where it is read, and the commit messages say what each
change measured. The labels in brackets (G35, C7, P11 ...) are the names of the changes in the patches and in
[techniques.md](techniques.md).

## Presets

`LLAMA_MOE_STREAM_PRESET` sets several switches at once. A preset only fills switches that are not set, so an
explicit switch still wins. Unset means `default`.

| preset | sets | what you get |
|---|---|---|
| `exact` | `LLAMA_MOE_STREAM_DROP_W=0`, `LLAMA_DSV4_YOCO=0` | no approximation beyond the quantized weights: every routed expert is waited for, the prompt runs through all layers; runahead (below) turns itself on |
| `default` | nothing | the defaults in the tables below: the drop rule, G31, G35, C7 and the rest |
| `fast` | `LLAMA_MOE_STREAM_C14=0.2`, `LLAMA_MOE_STREAM_G40=1` | `default` plus two lossy changes for speed (small measured quality cost) |

## Decode: the expert ladder

| switch | default | what it does |
|---|---|---|
| `LLAMA_MOE_STREAM_DROP_W` | `0.14` (with `DROP_ALL=1`) | the drop rule: an expert in neither VRAM nor the RAM tier carrying less than this share of the token's routing weight is not waited for. `0` is exact. Lossy |
| `LLAMA_MOE_STREAM_RUNAHEAD` | `6` on the exact path, off otherwise | G2: between a layer's two halves of expert remapping, an approximate pass (the experts already resident, the others at weight 0) runs the next layer's attention and router and stages its top K experts; the exact layer then runs unchanged, so the output is the same. Off by default next to the drop rule, an oracle or the prediction modes |
| `LLAMA_MOE_STREAM_G35` | `2` | G35, heavy experts ahead: layer L+d's router applied to layer L's input ("the lens") predicts the heavy experts that would come from the drive, and they are read into the RAM tier d layers early. `_G35_M` (8) candidates, `_G35_H` (0.14) share threshold; `0` is off. Lossless |
| `LLAMA_MOE_STREAM_G31` | `1` | G31: each MoE call is split - the pairs whose experts are resident compute while the copies of the missing ones run. Needs the mailbox and its reply |
| `LLAMA_MOE_STREAM_MAILBOX` | `1` | G16: the GPU posts the router's choices into pinned host memory, the host remaps without a synchronizing copy |
| `LLAMA_MOE_STREAM_MBOX_REPLY` | `1` | G22: the host's answer (VRAM slots of the experts) goes back the same way, read by the GPU at the next call |
| `LLAMA_MOE_STREAM_WAKE_ONCE` | `1` | G24: one host wake-up per token instead of one per layer |
| `LLAMA_MOE_STREAM_C14` | off (`0.2` in `fast`) | C14: up to `_C14_N` (1) pairs per token with less than this share of the routing weight - the ones that would come from the drive first, then the lightest - are not waited for: their copies are queued last, and their weighted sum enters after the next layer's attention. Lossy |
| `LLAMA_MOE_STREAM_G40` | `0` (`1` in `fast`) | G40: G35's lens stages up to k of its best guesses (share of at least `_G40_H`, 0.18) into VRAM slots two calls ahead. The math is untouched, but next to the drop rule ~5% more experts end up dropped |
| `LLAMA_MOE_STREAM_MIRROR` | unset | P16: up to three `;`-separated folders with a copy of the expert parts on other drives. A read goes to a copy when the primary drive has `_MIRROR_AT` (4) reads in flight and the copy fewer than `_MIRROR_MAX` (4) |
| `--moe-stream-io-threads` (flag) | 17 | reader threads for the drives |
| `LLAMA_ENGRAM_AHEAD` | `1` | G26: the Engram rows of a token are faulted in on a helper thread while the graph is built |
| `LLAMA_ENGRAM_LATE` | `1` | G23: at decode an Engram layer's rows are gathered right before the layer that reads them |
| `LLAMA_DSV4_SPARSE_DECODE`, `LLAMA_DSV4_SPARSE_QUANT` | `1`, `1` | sparse attention in decode, also over a q8_0 / q4_0 KV cache (D11) - decode stays flat from short context to 512K |
| `LLAMA_MOE_STREAM_KEEP_MAPPED` | unset | `1` keeps the expert files memory-mapped (an NTFS file that is also mapped reads ~3x slower unbuffered) |

## Prefill

| switch | default | what it does |
|---|---|---|
| `LLAMA_DSV4_ENC_ONLY` | `1` | P4: positions the decoder half (layers 20-39) never reads run through the encoder layers only. Exact |
| `LLAMA_DSV4_YOCO` | `140` | C7: DeepSeek's YoCo truncation - the decoder half computes the last K prompt positions (its experts are skipped for the others). `0` is off (and `exact` sets it). Lossy, within error on our quality sets |
| `LLAMA_MOE_STREAM_P11` | `64` | P11: a long prompt takes the VRAM of the expert cache; the waves of all layers go through a shared pool of this many slots |
| `LLAMA_DSV4_SPARSE_PREFILL` | `1` | D10: sparse attention in prefill |
| `LLAMA_HEAD_YIELD` | `64` | P1: the output head gives its VRAM to prompts longer than N tokens and is read back from its file afterwards. Exact |
| `LLAMA_ENGRAM_LATE_PP` | `1` | P8: a prompt's Engram rows are faulted in layer by layer under the compute of the earlier layers. Exact |
| `GGML_CUDA_HOST_CACHE` | `1` | P20: pinned host blocks of 64 MiB and more are kept for reuse between prompts |
| `LLAMA_MOE_WEIGHT_INPLACE` | `1` | P17: expert outputs are weighted in place (a smaller compute buffer at a large batch) |
| `LLAMA_MOE_STREAM_WAVE_SKIP` | `1` | waves skip the pairs outside them instead of parking them in extra slots |
| `LLAMA_MOE_STREAM_L2_SCAN` | `1` | P9: a ring in the RAM tier for what a prompt reads past what the tier holds |
| `LLAMA_DSV4_SWIGLU_FUSED` | `1` | V4's clamped SwiGLU as one op |

## Diagnostics

| switch | what it does |
|---|---|
| `LLAMA_MOE_STREAM_SLOW_READ_MS=N` | log every drive read of N ms or more (D14: the drive's stalls) |
| `LLAMA_MOE_STREAM_TRACE=file` / `_ORACLE=file` / `_AHEAD=N` | JigSawPT's: record a run's expert sequence / replay it as a perfect predictor, N calls ahead; the patches add `_ORACLE_*` knobs for the ceiling studies |
| `LLAMA_GRAPH_TOP=N`, `LLAMA_GRAPH_PEAK=N`, `GGML_ALLOC_TRACE` | what fills the compute buffer |
| `LLAMA_HEAD_YIELD_DEBUG=1` | a hash of every output row's logits: runs with and without P1 must print the same lines |
| `LLAMA_MOE_STREAM_PROBE_VRAM=1` | how far past the reported budget the driver really allocates |

## All switches

The rest are measurement knobs of the changes above and of experiments that did not become defaults; each is
explained in the source where it is read.

| switch | read in |
|---|---|
| `GGML_ALLOC_TRACE` (above) | `ggml/src/ggml-alloc.c` |
| `GGML_CUDA_HOST_CACHE` (above) | `ggml/src/ggml-cuda/ggml-cuda.cu` |
| `GGML_SCHED_HOIST_INPUTS` | `ggml/src/ggml-backend.cpp` |
| `LLAMA_DSV41_NO_HC_THREAD` | `src/llama-context.cpp`, `src/models/deepseek4.cpp` |
| `LLAMA_DSV4_ATTN_QCHUNK` | `src/llama-graph.cpp` |
| `LLAMA_DSV4_ATTN_QCHUNK_IDENTITY` | `src/llama-graph.cpp` |
| `LLAMA_DSV4_ATTN_QCHUNK_MIN` | `src/llama-graph.cpp` |
| `LLAMA_DSV4_ATTN_SPLIT` | `src/models/deepseek4.cpp` |
| `LLAMA_DSV4_ATTN_SPLIT_MIN` | `src/models/deepseek4.cpp` |
| `LLAMA_DSV4_CAND_COMPACT` | `src/models/deepseek4.cpp` |
| `LLAMA_DSV4_DENSE_SHIFT` | `src/models/deepseek4.cpp` |
| `LLAMA_DSV4_ENC_ONLY` (above) | `src/llama-context.cpp` |
| `LLAMA_DSV4_ENC_ONLY_MIN` | `src/llama-context.cpp` |
| `LLAMA_DSV4_ENC_ONLY_REACH` | `src/llama-context.cpp` |
| `LLAMA_DSV4_Q_ROPE_INPLACE` | `src/models/deepseek4.cpp` |
| `LLAMA_DSV4_SPARSE_DECODE` (above) | `src/models/deepseek4.cpp` |
| `LLAMA_DSV4_SPARSE_FA_NCOLS2` | `ggml/src/ggml-cuda/fattn.cu` |
| `LLAMA_DSV4_SPARSE_MASK_VIEW` | `src/models/deepseek4.cpp` |
| `LLAMA_DSV4_SPARSE_MIN` | `src/llama-graph.cpp` |
| `LLAMA_DSV4_SPARSE_PREFILL` (above) | `src/llama-graph.cpp` |
| `LLAMA_DSV4_SPARSE_QC` | `src/llama-graph.cpp` |
| `LLAMA_DSV4_SPARSE_QUANT` (above) | `src/models/deepseek4.cpp` |
| `LLAMA_DSV4_SWIGLU_FUSED` (above) | `src/llama-graph.cpp` |
| `LLAMA_DSV4_YOCO` (above) | `src/llama-context.cpp` |
| `LLAMA_DSV4_YOCO_MODE` | `src/llama-context.cpp` |
| `LLAMA_DSV4_YOCO_NOBOUND` | `src/llama-context.cpp` |
| `LLAMA_DSV4_YOCO_ROWS` | `src/llama-context.cpp` |
| `LLAMA_ENGRAM_AHEAD` (above) | `src/models/deepseek4.cpp` |
| `LLAMA_ENGRAM_LATE` (above) | `src/models/deepseek4.cpp` |
| `LLAMA_ENGRAM_LATE_PP` (above) | `src/models/deepseek4.cpp` |
| `LLAMA_GRAPH_PEAK` (above) | `src/llama-context.cpp` |
| `LLAMA_GRAPH_TOP` (above) | `src/llama-context.cpp` |
| `LLAMA_HEAD_YIELD` (above) | `src/llama-context.cpp`, `src/llama-model.cpp` |
| `LLAMA_HEAD_YIELD_DEBUG` (above) | `src/llama-context.cpp` |
| `LLAMA_MOE_STREAM_ASYNC_UPLOAD` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_C14` (above) | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_C14_DEPTH` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_C14_DROP` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_C14_N` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_C14_Q` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_C14_SEL` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_C14_SKIP` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_DROP_ALL` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_DROP_LATE` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_DROP_ONLY_LAYER` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_DROP_PACK` | `src/llama-graph.cpp` |
| `LLAMA_MOE_STREAM_DROP_RENORM` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_DROP_SPEC` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_DROP_VRAM_ONLY` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_DROP_W` (above) | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_G31` (above) | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_G32` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_G32_CHECK` | `src/models/deepseek4.cpp` |
| `LLAMA_MOE_STREAM_G35` (above) | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_G35_H` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_G35_LAZY` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_G35_M` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_G40` (above) | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_G40_H` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_G40_LOW` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_G40_SRC` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_G40_TIER` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_G4_K` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_G4_PROBE` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_G4_T_US` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_HOT_DECAY` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_KEEP_FIRST` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_KEEP_MAPPED` (above) | `src/llama-model.cpp` |
| `LLAMA_MOE_STREAM_L2_CLASSES` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_L2_MARGIN` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_L2_SCAN` (above) | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_L2_SCAN_MIN` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_L2_WAIT` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_MAILBOX` (above) | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_MASK_LATE_SPLIT` | `src/llama-graph.cpp` |
| `LLAMA_MOE_STREAM_MBOX_REPLY` (above) | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_MIRROR` (above) | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_MIRROR_ADAPT` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_MIRROR_ADAPT_AT` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_MIRROR_ADAPT_MS` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_MIRROR_AT` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_MIRROR_MAX` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_ORACLE_FALSE` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_ORACLE_FALSE_LOW` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_ORACLE_ONLY` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_ORACLE_RECALL` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_P11` (above) | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_P11_AHEAD` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_PRED_RAM` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_PRED_STAGE` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_PRED_VRAM` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_PREFETCH_ALL` | `src/llama-model.cpp` |
| `LLAMA_MOE_STREAM_PRESET` (above) | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_PROBE_VRAM` (above) | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_RA_DEBUG` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_RA_GROUPS` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_RA_GROUPS_SCALE` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_RA_REUSE` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_RA_SKIP` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_RUNAHEAD` (above) | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_SHIFT_RANKS` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_SHIFT_TAU` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_SKIP_MIN` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_SLOT_FENCE` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_SLOW_READ_MS` (above) | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_SPEC_IDLE` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_SPEC_QUEUE` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_SPEC_VRAM` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_SPEC_WORKERS` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_STAGE_INFLIGHT` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_SUBST` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_SUBST_W` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_VRAM_AHEAD` | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_WAKE_ONCE` (above) | `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_STREAM_WAVE_SKIP` (above) | `src/llama-model.cpp`, `src/llama-moe-stream.cpp` |
| `LLAMA_MOE_WEIGHT_INPLACE` (above) | `src/llama-graph.cpp` |
