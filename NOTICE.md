# Notices

This repository's own code and notes are under [MIT-0](LICENSE): no attribution needed. Parts that come from other
projects keep their own licenses:

| what | origin | license |
|---|---|---|
| `patches/` — changes to llama.cpp source files | llama.cpp / ggml, through JigSawPT's DeepSeek-V4.1 fork (`github.com/JigSawPT/llama.cpp`, branch `dsv41-porte`) | MIT, © 2023-2026 the ggml authors — keep `LICENSE` of llama.cpp with the patched files |

Some changes follow ideas of other projects, named in the patches' comments: DeepInfer's YoCo prefill truncation,
Strata's fused SwiGLU op, ds4's logit and token-text conventions. These are our own implementations; no code from those
projects (or from Project Maya) is included.

No model weights are included. The assembly script downloads them from their own repositories, each under its own
license (see the model cards; all three were MIT on 2026-10-25): DeepSeek's official DeepSeek-V4.1-Flash-1017 release
(`deepseek-ai/DeepSeek-V4.1-Flash-1017`), prometheusAIR's and ISTA-DASLab's GGUF quants of it.

`template/` holds the GGUF headers of our builds (metadata and tensor tables) and their first part whole (metadata,
the tokenizer and four small Engram hash tables). They are derived from DeepSeek's MIT-licensed release, whose
license is kept beside them as [template/LICENSE](template/LICENSE) (MIT, © 2023 DeepSeek).
