# Notices

This repository's own code and notes are under [MIT-0](LICENSE): no attribution needed. Parts that come from other
projects keep their own licenses:

| what | origin | license |
|---|---|---|
| `patches/` — changes to llama.cpp source files | llama.cpp / ggml, through JigSawPT's DeepSeek-V4.1 fork (`github.com/JigSawPT/llama.cpp`, branch `dsv41-porte`) | MIT, © 2023-2026 the ggml authors — keep `LICENSE` of llama.cpp with the patched files |
| code adapted from Project Maya (marked in the file headers) | `github.com/mw00/project-maya` | MIT, © 2026 Peasant Smith and the Project Maya contributors |

No model weights are included. The assembly script downloads them from their own repositories, each under its own
license (see the model cards; all three were MIT on 2026-10-25): DeepSeek's official DeepSeek-V4.1-Flash-1017 release
(`deepseek-ai/DeepSeek-V4.1-Flash-1017`), prometheusAIR's and ISTA-DASLab's GGUF quants of it. If the GGUF part headers
of our build are included as the assembly template (metadata and tensor tables only), they are derived from DeepSeek's
MIT-licensed release and keep its notice.
