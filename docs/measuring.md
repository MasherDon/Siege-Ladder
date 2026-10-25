# How we measure

The scripts are in [bench/](../bench) (paths in `bench/config.py`). Everything here ran on Windows 11 with the
patched llama.cpp; the numbers in the main README come from these methods.

## Decode speed: teacher forcing

Generation speed depends on the text: which experts the tokens route to decides the hits and misses. To compare two
configurations on the same tokens we do not sample; we feed a fixed text one token at a time and time it:

    llama-perplexity -m <model> -ngl 99 -t 1 -c 512 -b 512 -ub 1 --moe-stream-direct --moe-stream-cache 23s
                     --moe-stream-l2 46 --moe-stream-io-threads 17 -f <text>

with `OMP_WAIT_POLICY=PASSIVE`, `LLAMA_MOE_STREAM_SLOW_READ_MS=16` and `LLAMA_MOE_STREAM_L2_SCAN=0`. Every token is
one decode step with the whole expert machinery (`-ub 1`); 512-token chunks keep the context short. The time per
token is the span between the log's "calculating perplexity" and "Final estimate" lines over the token count
(`bench/tf_run.py` runs a series, `bench/tf_series.py` summarises it).

- **The text.** ~3,000 tokens of the model's own answers (ours were answers to a private prompt and are not
  published; any text of that length works - absolute speeds move a little with it, comparisons hold).
- **Order.** A warm-up run, then the arms in A B B A order: the RAM tier and the drive's own caches warm up over a
  series, and ABBA cancels the trend.
- **Drive stalls (D14).** Our Gen5 drive freezes for 35-50 ms about once a second in some runs, for minutes at a time
  (all reads in flight end within the same millisecond, apparently on the device's own clock; the copy on the second
  drive never stalls). The stalls land unevenly in the arms and cost ~2 ms per token, so raw means mislead. We count them
  (`SLOW_READ_MS` logs every read of 16 ms or more; reads ending within 5 ms of each other are one stall), fit one
  line "ms per token against stalls per 1,000 tokens" through all runs of a series (slope 0.08-0.10 ms), and compare
  the arms at zero stalls. Corrected and raw differences are both reported.

## Reading speed: server timings

Prompt processing is measured on the server (`bench/harness.py`), from its own timings (`prompt_n /
prompt_per_second`), greedy, thinking off:

- **Cold prompts** (2.2K and 8.9K tokens of C source from the patched tree, `harness.py prefill`): a fresh server per
  run, so the RAM tier starts empty.
- **Agent turns** (`agent_turns.py`): one server for an 8-turn session; each turn adds 0.5-7.5K tokens of tool output
  (CUDA sources of ggml) and a short question to the growing conversation, the prompt cache keeps the earlier turns,
  so a turn reads only its new tokens on a warm tier. Turn 1 is cold and reported apart.
- **Long documents** (`consult.py`): a document cut from the patched tree's C sources, then questions about it; the
  KV slot can be saved and restored to time a return to the document without re-reading it.

Every run also records the bytes the process read from the drive, pages faulted in, and the fork's own counters
(VRAM hits and misses, the wait for expert loads, RAM-tier hits, prefill waves).

## Quality: distance to the full model

We score how likely our build finds the full model's own answers:

1. `api_collect.py` sends each prompt of a set to DeepSeek's API (V4.1 Flash 1017, thinking off) and keeps the
   continuation with its log-probabilities.
2. `ds4_score.py` feeds each continuation to our build token by token after the prompt, exactly as antirez's ds4
   does (`score_official.c`: BOS, user turn, assistant turn with an empty think block, no system prompt), and
   averages the negative log-likelihood (NLL) over all continuation tokens. `llama-logits` of the patched build only
   tokenizes and runs the passes; one model load for all prompts.
3. The **distance to the full model** is our NLL minus the API's own NLL on the same tokens: smaller is closer.
   `ds4_compare.py` compares two runs on the same tokens with a 95% bootstrap interval over prompts. For scale: going
   from a Q2 to a Q4 quant of this model is ~0.11 nat; running the same Q2 file through different engines moves it by
   ~0.004.

Prompt sets: **general-100** (100 short prompts from ds4's quality tests), **router-112** (112 code and agent prompts
from ds4, up to 23K tokens), **general-ru** (ours: Russian prompts, not published) and **agent-long** (ours: 40 prompts
of 0.5-8K tokens of llama.cpp sources plus a question). The prompt sets of ds4 live in its repository; the API's
answers are not redistributed here - collect your own with `api_collect.py` and a DeepSeek API key.

A lossy change becomes a default only when the quality sets show no significant loss (general-100 within ±0.004 nat);
otherwise it goes to the `fast` preset. HumanEval (`humaneval.py`: the 164 tasks of openai/human-eval, greedy,
thinking off, the answer's function run against the task's tests, pass@1) is a check on top: 157 for our default
against 158 for the API.

## Profiles: Nsight Systems

`nsys_server.py`, `prefill_nsys.py`, `turn_nsys.py` and `oracle_nsys.py` run the server or `llama-perplexity` under
`nsys` with the fork's NVTX ranges (each expert wave carries its layer); `nsys_split.py`, `nsys_waves.py`,
`nsys_layers.py` and `nsys_gaps.py` read the `.sqlite` export: compute, copies, overlap and idle time of the GPU per
token, waves per layer, and what each gap in the GPU's timeline waited for. The charts in the README come from these
tables ([docs/data](data)).

## Bit-exactness

Lossless changes are checked to give the same logits as before: `p4_check.py` and `c7_check.py` compare the outputs
of the prefill paths bit for bit (with a negative control that must differ), and a run with a change and one without
must print the same per-row logit hashes (`LLAMA_HEAD_YIELD_DEBUG=1`) or the same text and counters.
