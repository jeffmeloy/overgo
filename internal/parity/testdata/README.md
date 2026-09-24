# OLMoE fixture boundary

`olmoe_real_reference.json` is the 5-token prompt and 12-token greedy
continuation generated with Hugging Face `OlmoeForCausalLM` by Colibri's
`c/tools/make_olmoe_real_oracle.py` at commit
`f028d26b422144ed4a69ad9aeaee2553ce0f9572` (Colibri is Apache-2.0).
The script SHA-256 is
`b6be54f74aa55c2767bf46c37efbb9fda96a915589e4d2269b60030553`;
the reference JSON SHA-256 is
`8f8f3140615b0033dd3d31624bab979b0737b91cda0300a64756bd039a459037`.
This records an independent output target. The real model weights, tokenizer,
and exact execution environment are not present in this fixture, so these
tests do not establish Overgo real-model numerical parity.

`make_olmoe_tiny.py` independently serializes `olmoe_tiny.gguf` in GGUF v3.
Its 15 deterministic F32 tensors model a one-block **structural** inventory,
not trained weights. `olmoe_tiny.json` records every tensor's shape, payload
hash, and a small Q-data sum probe. The GGUF SHA-256 is
`191e65fe8477cd1e50a9d14ee832aa6bf10b39980b61189831f2f44f2a05a5d1`.
The tests require all files and compare Overgo's parser and writer against
those independent bytes. Real OLMoE validation remains in the
`colibri-moe-oracle/do` plan step.
