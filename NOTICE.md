# Notices

This repository's own code and notes (the scripts in `bench/` and `tools/`, the docs, the charts) are under
[MIT-0](LICENSE): no attribution needed. Parts that come from other projects keep their own licenses, and whoever
redistributes them keeps their notices:

| what | origin | license |
|---|---|---|
| `patches/` - changes to llama.cpp source files (their context lines are llama.cpp's code) | llama.cpp / ggml, through JigSawPT's DeepSeek-V4.1 fork (`github.com/JigSawPT/llama.cpp`, branch `dsv41-porte`) | MIT, © 2023-2026 the ggml authors - [patches/LICENSE-llama.cpp](patches/LICENSE-llama.cpp) |
| the expert streaming path that many patches change (`src/llama-moe-stream.*`) | Crow by nibor1896 (`github.com/nibor1896/Crow`), applied to JigSawPT's fork as a patch series | MIT, © 2026 nibor1896 - [patches/LICENSE-Crow](patches/LICENSE-Crow) |
| `template/` - the GGUF headers of our builds and their first part (metadata, the tokenizer, four small Engram hash tables) | derived from DeepSeek's DeepSeek-V4.1-Flash-1017 release | MIT, © 2023 DeepSeek - [template/LICENSE](template/LICENSE) |

So the patched llama.cpp you build carries llama.cpp's and Crow's MIT notices, not MIT-0.

Some changes follow ideas of other projects: DeepInfer's YoCo prefill truncation, Strata's fused SwiGLU op and ds4's
logit and token-text conventions (named in the patches' comments), and Project Maya's late experts (C14). These are
our own implementations; no code from those projects is included.

No model weights are included. The assembly script downloads them from their own repositories, each under its own
license: DeepSeek's official DeepSeek-V4.1-Flash-1017 release (`deepseek-ai/DeepSeek-V4.1-Flash-1017`), prometheusAIR's
and ISTA-DASLab's GGUF quants of it. All three were MIT when we checked on 2026-10-26; check the model cards when you
download.
