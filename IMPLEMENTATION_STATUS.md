# Implementation status

## Compatibility baseline

- llama.cpp commit: `42fc243060709331ff9b158a9ed2cbe37219ae83`
- initial host: Windows amd64
- initial device: NVIDIA compute capability 8.9
- cgo: disabled

## Status

| Area | State | Notes |
| --- | --- | --- |
| Repository foundation | Complete | Go module and compatibility contract created |
| CUDA driver loading | Complete | Direct `nvcuda.dll` calls, device inventory validated |
| CUDA execution | In progress | Persistent resources, cuBLAS F32, native quantized embedding/matmul/routed-expert decode, fused Qwen3-Next/Qwen3.5 recurrent kernels, and correctness-first routed MoE validated on RTX 4090 |
| GGUF format | In progress | Bounds-checked parser, automatic validated split-file loading, bounded tensor ranges, streamed device weights, and canonical streaming single-file writer validated |
| Tensor graph | In progress | Typed IR, layout transforms, head broadcasting, clamp, RMSNorm/affine LayerNorm, token/learned-position embeddings, scaled and YaRN normal/NeoX RoPE, scaled multi-axis RoPE, ALiBi, causal and symmetric sliding/softcapped/gated GQA with per-head sink logits, bidirectional T5 relative-position attention, MLA decomposition, softmax/sigmoid top-k routed gated or ungated SiLU/ReLU/GELU MoE with separate or fused gate/up expert storage, split router inputs, grouped expert-bank indices, correction bias, and per-expert output scales, GELU/xIELU/SwiGLU/squared-ReLU, SSM convolution, fused gated delta net, reference/CUDA executors, and arena planner |
| Quantization | In progress | F32/F16/BF16/F64, I8/I16/I32/I64, Q8_0, Q2_K-Q6_K, every pinned IQ1/IQ2/IQ3/IQ4 layout, Q1_0/Q2_0, TQ1_0/TQ2_0, MXFP4/NVFP4, Q4_0/Q4_1, and Q5_0/Q5_1 decoding |
| Model runtime | In progress | Incremental dense Qwen 2/3, bounded-host/F32-preload/native-quantized Mixtral, Arctic, BailingMoE/BailingMoE2, Cohere2-MoE, DBRX, Deci, DOTS1, DeepSeek v1/DeepSeek2-OCR, ERNIE 4.5-MoE, Gemma4, GLM4-MoE, GraniteMoE, GroveMoE, Grok, Hunyuan-Dense/Hunyuan-MoE, HY-V3, Mellum, MiMo2, MiniMax-M2, SmallThinker, Qwen2-MoE, Qwen3-MoE, Qwen3-VL-MoE, AFMoE, Laguna MoE, OLMoE, PhiMoE, EXAONE-MoE, and LLaDA-MoE, text-only hybrid Qwen3-Next/Qwen3.5/Qwen3.5-MoE, non-causal no-cache Dream, LLaDA, and RND1 MoE, hybrid LFM2/LFM2-MoE, PLM/MiniCPM3 MLA, BERT/EuroBERT/Gemma Embedding/JinaBERT v2/v3/Llama Embed/ModernBERT/NeoBERT/NomicBERT/NomicBERT-MoE encoders, Chameleon decoders with projected soft-token overrides, Hunyuan-VL/PaddleOCR/Qwen2-VL/Qwen3-VL text-coordinate decoding, CogVLM token-input decoding, Apertus, Arcee, Baichuan 7B, BitNet, Bloom, CodeShell, dense Cohere2/Command R/ERNIE 4.5, Falcon, Gemma 1/2/3, GLM4, GPT-2/GPT-NeoX, Granite, InternLM2, EXAONE/EXAONE 4, XVERSE, Jais/Jais2, Maincoder, MiniCPM, compatible MPT, dense Mistral 3, Nemotron, OLMo/OLMo2/OLMoE, OpenELM, Orion, Pangu Embedded, Phi-2/Phi-3, PLaMo/PLaMo 3, dense Refact, Seed-OSS, StableLM, StarCoder/StarCoder2, SmolLM3, Talkie, T5 encoder, and constrained Llama-family CUDA execution with serializable, prefix-editable attention/recurrent cache |
| Tokenizer and sampling | In progress | Seven tokenizer corpora match 326 upstream cases; Gemma4 raw UTF-8 BPE, BERT WordPiece, and real-model T5 UGM are validated; ordered/repeatable top-k/p, min-p, typical, top-n-sigma, XTC, penalties, DRY, infill, Mirostat v1/v2, GBNF, and JSON-Schema conversion implemented |
| CLI and server | In progress | Inspect/tokenize/block-check/generate/perplexity/embedding/benchmark/JSON-Schema CLIs plus bounded completion, streaming, embedding, literal-choice, GBNF, and JSON-Schema HTTP APIs |
| Local verification | Complete | Unit and optional CUDA integration script |

## Validated milestones

- The local Qwen3 4B GGUF loads a 151,936-token GPT-2/Qwen2 vocabulary and
  tokenizes representative text.
- The benchmark CLI records JSON load, TTFT, post-first-token decode,
  end-to-end, host-heap, model, and device metrics with bounded warmup/run/token
  controls. A real Qwen3 Q8 native-quant smoke run on the RTX 4090 D loaded in
  4.09 s, reached first token in 181 ms, and decoded the post-first token at
  10.5 tokens/s. A subsequent instrumented run measured 6,706,837,504
  persistent Runner-owned CUDA bytes and a 6,707,455,488-byte generation
  high-water mark.
- CUDA driver allocation accounting keeps a concurrency-safe pointer-size
  ledger for every successful `MemAlloc`/`MemFree` on the Runner's shared
  worker. Current allocation count/bytes and lifetime peak bytes therefore
  cover persistent device weights, graph arenas, feeds, and auxiliaries.
- Bounded fuzz harnesses cover GGUF parsing, tokenizer encode/decode, GBNF
  compilation, JSON-Schema conversion, sampler restoration, cache/session
  restoration, and server JSON routes. Short live campaigns completed millions
  of mutations without a panic. `scripts/fuzz-smoke.ps1` makes the eight-target
  campaign repeatable.
- A deterministic CycloneDX 1.6 SBOM inventories the Go module graph, pinned
  llama.cpp baseline, Go standard library, CUDA driver/toolkit, and SHA-256
  hashes for the versioned CUDA manifest, sources, and embedded PTX. The
  kernel-manifest verifier also checks schema/ABI metadata, the complete PTX
  entry set, and every function's ordered parameter types/alignment/array
  extents. Unit tests, verification, release creation, and CI reject stale
  provenance. The project/kernel license remains `NOASSERTION` rather than
  assuming distribution rights.
- CUDA module loading fails closed unless the embedded PTX matches the
  ABI-versioned hashes compiled into the Go host. The manifest verifier ties
  those runtime pins to the checked-in assets, providing content-based NVIDIA
  JIT-cache invalidation without a separate application cache.
- Driver-level execution telemetry counts successful custom-kernel launches,
  stream/context barriers, and H2D/D2H operations and bytes. Benchmark runs
  report deltas and per-token rates; server metrics expose monotonic totals.
- Preloaded dense generation composes embedding lookup, every layer, final
  normalization, last-token slicing, and output projection into one executor
  graph per token. KV outputs have explicit device lifetimes and identical
  RoPE attributes share one upload. On the local Qwen3 Q8 three-token
  benchmark this reduced stream synchronizations from 117 to 3 and H2D/D2H
  calls from 477/333 to 6/3, leaving only logits on the host; greedy Q8/Q6_K
  output and allocation-release invariants pass.
- Direct Qwen3.5 generation uses the same retained-output lifetime for mixed
  attention KV and convolution/SSM state. Its local three-token run uses one
  barrier and one logits copy per token; 52.69 MB of initial recurrent zeros
  are created by 48 stream-ordered device memsets instead of host transfers.
- Embeddings share one Runner implementation for mean, last-token, and
  unpooled per-token output. Normalization matches pinned
  `common_embd_normalize` (`-1`, max-absolute int16 range, and general p-norm);
  OpenAI responses can encode little-endian float32 vectors as base64. Real
  UMT5 per-token output matches its validated final hidden state.
- The release builder cross-compiles sixteen Windows-amd64 tools with
  `CGO_ENABLED=0`, `-trimpath`, no VCS stamp, and no Go build ID; it builds
  twice and requires byte-identical stored ZIPs with fixed timestamps and
  internal/external SHA-256 manifests. A real two-build check produced
  byte-identical archives; the checksum is kept outside the archive to avoid a
  self-referential manifest.
- The tokenizer matches every checked-in `ggml-vocab-qwen2` and
  `ggml-vocab-gpt-2` oracle case from the pinned llama.cpp checkout.
- BERT WordPiece matches all 46 checked-in `ggml-vocab-bert-bge` oracle cases.
  The Go path ports the pinned default/metadata-controlled NFD accent stripping
  and lowercasing, Unicode whitespace/control/mark handling,
  punctuation/ASCII-symbol/Chinese segmentation, `▁` longest-token matching,
  whole-word unknown fallback, CLS/SEP insertion, and three-pass decode-space
  cleanup. `golang.org/x/text` supplies the pinned pure-Go Unicode NFD tables.
- Llama SPM and Llama 3 BPE tokenization each match all 46 checked-in upstream
  oracle cases, including whitespace, Unicode, byte fallback, and punctuation.
- The Qwen3.5 pre-tokenizer includes combining marks in letter runs and matches
  all 46 pinned upstream `ggml-vocab-qwen35` cases. The real local 9B
  vocabulary loads and tokenizes successfully.
- Raw GGUF tensors can be streamed into persistent CUDA allocations without a
  full host-side weight copy.
- The pinned IQ1_S/IQ1_M, IQ2_XXS/IQ2_XS/IQ2_S, and IQ3_XXS/IQ3_S
  codebooks are deterministically generated from audited `ggml-common.h`.
  Their host decoders byte-match complete 256-value C-oracle blocks, while
  native CUDA `get_rows` and `mul_mat` paths match the Go reference on the
  RTX 4090 D. The kernel manifest hashes the generated include dependency.
- Local split GGUF models opened through
  `<prefix>-00001-of-XXXXX.gguf` load as one logical file while every tensor
  retains its shard-local source, data offset, and bounds. The reader validates
  pinned `uint16` split indices/counts, the non-negative `int32` global tensor
  count, shard naming, missing parts, duplicates, aggregate limits, and closes
  all part handles together. `inspect-gguf` exposes the split count and tensor
  shard index.
- The pure-Go GGUF v2/v3 writer validates every scalar and array metadata
  representation, tensor dimensions/types/block alignment, duplicate names,
  offsets, and short payloads. It streams canonical aligned data without
  buffering weights. `gguf-merge` uses the same path to merge validated split
  sources, removes split bookkeeping, and never overwrites an existing output.
  The inverse `gguf-split` path partitions on tensor count and aligned payload
  bytes, supports a metadata-only first shard, writes pinned upstream names and
  split metadata, and cleans up only shards it created after a failure. Pinned
  `llama-gguf-hash` accepts Go writer output, and pinned
  `llama-gguf-split --merge` accepts and merges the Go-produced shard set.
- The pure-Go GGUF hasher streams raw logical tensor bytes and matches pinned
  `llama-gguf-hash` for per-tensor and concatenated-model XXH64, SHA-1,
  SHA-256, and llama.cpp-namespaced UUIDv5 output. Manifest checks preserve
  model/tensor fallback and distinct mismatch, missing-entry, unknown-format,
  and file-error exit states.
- The pure-Go model quantizer streams source blocks through the complete host
  dequantizer into pinned-layout Q1_0, Q2_0, Q2_K-Q6_K, Q4_0/Q4_1,
  Q5_0/Q5_1, Q8_0, TQ1_0/TQ2_0, IQ2_S, IQ3_XXS/IQ3_S,
  IQ4_NL/IQ4_XS, or MXFP4/NVFP4 encoders, with F32/F16/BF16 destinations as
  well. All 23 packed encoders, including internal Q8_1 and Q8_K layouts,
  byte-match exported reference routines from the pinned
  `ggml-base.dll`; multi-chunk requantization, "mostly" matrix selection,
  metadata rewriting, split input, exclusive output creation, and upstream
  `llama-gguf-hash` acceptance are covered.
- A complete small Qwen3 dense block executes on CPU and CUDA with matching
  results: Q/K/V projections, per-head Q/K RMSNorm, NeoX RoPE, causal GQA,
  attention output, residuals, and SwiGLU feed-forward.
- Layer 0 of the real local Qwen3 4B model matches the CPU reference over four
  tokens with maximum absolute error `1.43e-6`.
- The append-only KV cache matches full-forward hidden states for the checked
  two-token real-model integration case.
- Composite session state includes token history, KV tensors, deterministic RNG
  position, and adaptive Mirostat state. It validates payload bounds, sampler
  configuration, and a sampled GGUF fingerprint before restoring. A real
  Qwen3 native-Q8 save/load/continue run exactly reproduces uninterrupted
  greedy token output.
- Mirostat v1 and v2 both have deterministic adaptive implementations with
  reset and serialization coverage. DRY accepts token sequence breakers and
  expands string breakers across overlapping vocabulary-token boundaries like
  the pinned llama.cpp implementation.
- The implemented non-Mirostat stages use llama.cpp names and execute in a
  caller-selected, repeatable chain. The default is
  `penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature`; both
  the generation CLI and HTTP `samplers` array expose ordering. Order-sensitive
  candidate-set tests cover penalties before/after top-k, and custom chains
  participate in the resumable sampler signature without changing legacy
  default signatures.
- Top-n-sigma and XTC reproduce the pinned upstream probability-vector corpus.
  Top-p, min-p, typical, and XTC share `min_keep` behavior. XTC probability,
  threshold, and random-stream position are validated and resumable.
- The temperature stage supports the pinned dynamic-temperature entropy
  normalization, range, and exponent formula. Zero range remains byte-for-byte
  on the fixed-temperature path, and dynamic configuration is state-bound.
- Adaptive-p implements the pinned target-distance transform, always executes
  as the terminal selector, and updates its original-probability EMA only after
  token acceptance. Reset and sampler state v4 preserve its RNG and EMA;
  compatible v2/v3 state readers remain.
- Infill reproduces the pinned sorted-softmax EOG weighting, unrendered
  token-piece prefix merging, EOT/EOS fallback, and two probability thresholds.
  The runner builds its immutable descriptor from GGUF tokenizer metadata; CLI
  and HTTP ordered sampler chains populate it automatically, and its topology
  participates in the resumable sampler signature.
- Native `/infill` reproduces the pinned PSM/SPM repository-context formatter,
  3:1 prefix/suffix truncation, batch/context budgeting, BOS placement, and
  exact-token prompt handoff. The tokenizer reads all six current FIM token
  metadata keys, three deprecated aliases, and pinned text fallbacks while
  promoting auto-detected controls.
- Text-only chat formatting executes the GGUF `tokenizer.chat_template`
  through a pure-Go, in-memory-only Jinja loader with bounded source/output.
  Qwen3, Qwen3.5, Gemma 3, and Bonsai prompts are byte-exact against pinned
  `/apply-template`, including the injected `enable_thinking` default and
  single-BOS behavior. Boundary-token native formatters remain the fallback
  for models without metadata.
- Tool-aware Jinja formatting supplies OpenAI function schemas, assistant
  calls, tool results, named `tool_use` metadata templates, and Python-style
  ordered `tojson`. Qwen3 JSON-in-XML plus Qwen3.5/Bonsai Hermes prompt and
  history vectors are byte-exact against pinned `/apply-template`. Buffered
  and streaming OpenAI chat support auto, required, none, and named choices;
  auto uses lazy delimiter-triggered GBNF, required/named use forced
  schema-derived GBNF, and parsed calls receive validated arguments, IDs, and
  `tool_calls` finish reasons. Tool-enabled SSE buffers the template output
  through validation and emits complete structured call deltas without
  leaking XML/JSON wrapper syntax.
- A real Bonsai-27B-Q1_0 named-tool request on the RTX 4090 D generated the
  Hermes call for `weather`, parsed `{"city":"Boston"}`, assigned a call ID,
  returned null content, and finished with `tool_calls`. Disabling optional
  thinking kept the forced call to 26 output tokens.
- Ordered logit biases run before filtering and grammar acceptance, support
  additive finite adjustments and negative-infinity bans, and participate in
  the sampler signature. CLI `-ignore-eos` and HTTP `ignore_eos` ban every EOG
  token recognized by the loaded GGUF vocabulary; HTTP bias objects and pair
  arrays support numeric IDs, tokenized text keys, and `false` bans.
- Completion/chat stop strings are validated as a string or bounded array.
  The runner halts at the matching token boundary, and a cross-token output
  filter withholds possible prefixes so neither buffered nor SSE responses
  leak the matched marker. Unmatched prefixes flush at natural completion.
- Completion/chat `n` supports one through eight sequential choices. Each
  choice starts from fresh sampler, grammar, RNG, and adaptive state with a
  deterministic seed offset. Buffered and SSE responses preserve indices;
  usage counts prompt tokens once and aggregates generated choice tokens.
- The server process handles SIGINT/SIGTERM with a bounded 30-second graceful
  drain. Optional end-to-end request deadlines propagate through inference and
  return HTTP 408 on expiry; negative timeout configuration is rejected.
- Gemma turn-template formatting matches the pinned `/apply-template` oracle
  for user-only, system-prefix, and alternating conversations. It maps
  assistant to `model`, trims message bodies like the GGUF Jinja template,
  rejects role-order violations/boundary injection, and emits the final model
  generation marker. ChatML selection remains unchanged for Qwen vocabularies.
- Llama 3 header formatting matches the pinned
  `test-chat-template.cpp` vector: system/user/assistant headers, trimmed
  content, EOT delimiters, and the final assistant header. It validates role
  alternation and rejects all Llama 3 boundary tokens in message content.
- Authenticated `POST /apply-template` exposes the selected native formatter
  without inference, with bounded/strict JSON parsing and llama.cpp-compatible
  `{"prompt": ...}` output.
- Authenticated `/tokenize` and `/detokenize` endpoints use the loaded Go
  vocabulary. Tokenization accepts text or a flat mixed integer/string
  sequence, applies `add_special` only when the first component is textual,
  defaults `parse_special` to true, and can return pinned `{id,piece}` objects.
  Non-UTF-8 pieces become integer byte arrays. Detokenization concatenates
  exact pieces with special-token rendering enabled. Both reject malformed
  JSON, oversized arrays, and out-of-range IDs.
- Authenticated `GET /props` mirrors the pinned llama.cpp property envelope
  with normalized generation defaults, total slots, capability flags, model
  identity/path/file type, original GGUF chat template, BOS/EOS pieces, and a
  bounded architecture summary. Its values come from an immutable Runner
  snapshot; unsupported mutable `POST /props` is rejected with HTTP 405.
- Public `GET /models` and `/v1/models` return the pinned dual discovery
  envelope: Ollama-style `models` plus llama.cpp/OpenAI-style `data`. The
  immutable Runner snapshot supplies GGUF family, vocabulary type/count,
  train/runtime context, embedding width, quantization, overflow-safe
  parameter count, and tensor bytes while omitting the local model path.
- Authenticated native `POST /embedding` and `/embeddings` return the pinned
  non-OpenAI array with a nested pooled vector per indexed input. Both native
  and `/v1/embeddings` accept strings, exact token sequences, mixed
  token/string sequences, and bounded heterogeneous batches. Exact token IDs
  enter `EmbedTokens` directly; on real Qwen3 Q8, `[9707]` is byte-for-byte
  identical to textual `Hello` and retains unit L2 norm.
- Chat messages accept both string bodies and bounded OpenAI text content-part
  arrays; non-text parts remain explicit errors. Public `/v1/health` aliases
  the existing health probe. Authenticated `/responses` and
  `/v1/responses` convert text or text-message inputs through the same native
  formatter and return pinned Responses output/usage objects. Streaming emits
  the named created/in-progress/item/content/delta/done/completed SSE
  lifecycle without a `[DONE]` marker. Flat function definitions,
  auto/none/required/named selection, replayable `function_call` and
  `function_call_output` items, schema-constrained generation, buffered
  function-call output, single/parallel call constraints, and call-complete
  argument SSE events are supported.
  `/responses/input_tokens` and `/v1/responses/input_tokens` share that
  tool-aware conversion without running inference.
- Text-only Anthropic `/v1/messages` generation supports buffered responses
  and the named message/content-block SSE lifecycle, including native stop
  reasons/sequences and cache/input/output usage without llama timing fields.
  `/v1/messages/count_tokens` supports string and multipart text system/message
  content through the identical prompt formatter. Anthropic tool definitions
  and auto/any/named selection use the same schema grammar and GGUF Jinja tool
  context as OpenAI chat. Assistant `tool_use` and user `tool_result` history,
  single/parallel call constraints, buffered tool blocks, call-complete
  streaming `input_json_delta` blocks, and tool-aware counting are supported.
- Authenticated `/lora-adapters` exposes a truthful empty adapter list for this
  no-LoRA runtime. Empty disable-all POST requests succeed with the pinned
  envelope; non-empty activation is rejected explicitly.
- Native authenticated `POST /completion` and `/completions` reuse the same
  bounded sampler and generation core while exposing llama.cpp's `n_predict`,
  `n_cmpl`, raw-token, stop-type/word, generation-settings, prompt/token
  accounting, and timing envelope. Buffered single/multi-choice results and
  native SSE are covered; a live pinned-server differential confirmed partial
  token events, the final empty-content `stop:true` metadata event, and absence
  of OpenAI's `[DONE]` marker.
- OpenAI `/v1/completions` now shares the validated prompt parser: strings,
  exact/mixed token sequences, flat string batches, and nested heterogeneous
  batches up to 64 prompts. Exact IDs bypass tokenization, choices use stable
  prompt-major indices in buffered/SSE output, and usage counts each unique
  batch prompt once while aggregating all generated choices.
- Native prompt parsing accepts a string, an exact integer token sequence, or a
  mixed integer/string sequence. Exact IDs bypass re-tokenization in normal and
  resumable Runner generation, are cloned and range-checked, and preserve
  control/byte-token identity. A leading string segment alone applies
  configured BOS insertion. The response renders processed prompt text like
  the pinned oracle; Qwen3 Q8 exact ID `[9707]` reproduces the same
  `[9707, 27, 18, 198]` greedy sequence as textual `Hello`.
- Native multiple-prompt input supports flat string batches and heterogeneous
  batches containing nested exact/mixed token sequences, capped at 64 prompts.
  Buffered and SSE output flatten prompt-count × `n_cmpl` in prompt-major
  order with stable global indices. A pinned live oracle confirmed the order
  `Hello`, `Hello`, `Hello<`, `Hello<` for two prompts and two choices.
- Native `response_fields` projects final object, batch, and SSE metadata with
  the pinned full slash-path keys and silent omission of missing paths.
  Duplicate requests are idempotent, partial SSE token events remain intact,
  and field count/path length/component depth are bounded before generation.
- Admission uses a bounded pool of stable integer slot IDs rather than an
  anonymous semaphore. Authenticated `GET /slots` reports each slot's atomic
  busy/task state, context length, prompt processed/cache counts, raw generated
  text, zero-safe prompt/decode timing rates, and the authoritative effective
  sampler, stop, cache, and context-shift parameters passed to generation.
  `next_token` reports decoded and remaining counts; pending-token/newline flags
  are false because the synchronous Runner never retains a sampled token
  between steps.
  Completed request metrics remain available on the idle slot until its next
  acquisition.
  `fail_on_no_slot` returns HTTP 503.
  Native `id_slot` acquires the exact requested slot and returns HTTP 429 if it
  is occupied. Slot allocation is capped at 65,536 entries.
- Authenticated `/chat/completions` aliases the OpenAI-prefixed route.
  `/chat/completions/input_tokens` and `/v1/chat/completions/input_tokens`
  reuse the selected native formatter and tokenize with the same BOS/special
  policy as generation, returning the pinned `response.input_tokens` object.
  The real Qwen3 chat prompt produces an identical ID sequence through the
  generation and count paths.
- A sparse token-DFA constrains generation to exact textual alternatives,
  permits EOS only after an accepting choice, and serializes its current state
  with a topology-bound sampler signature. Repeated CLI `-grammar-choice`
  flags and the HTTP `grammar_choices` request field use the loaded model's
  tokenizer. A real Qwen3.5 run constrained `Hello` to produce `Hello there`.
- A bounded pushdown GBNF parser supports literals/escapes, Unicode character
  ranges and negation, wildcard characters, groups, alternation, named
  nonterminals, comments, right recursion, and all llama.cpp repetition
  operators. Candidate matching carries partial UTF-8 across byte tokens,
  gates EOG on root acceptance, rejects growing left recursion, and safely
  deduplicates nullable epsilon cycles. Its acceptance corpus mirrors pinned
  upstream simple/recursive/expression/repetition cases.
- Numeric `<[id]>`, vocabulary-named `<|special|>`, and inverse `!<...>` token
  terminals match the pinned upstream pushdown behavior. On Qwen3.5, both Go
  and llama.cpp constrained `Hello` to `<|im_start|><|im_end|>`; Go asserts
  generated IDs `[248045, 248046]` while omitting control pieces from decoded
  user text.
- Lazy GBNF can activate on token IDs or regex matches. It samples
  unconstrained while waiting, buffers token-piece byte spans, begins replay
  at the first participating capture group, preserves partial-token overlap,
  and serializes/restores the waiting buffer by deterministic token-history
  replay. CLI and HTTP expose triggers. A real Qwen3.5 check waits for comma,
  then constrains `Hello` to IDs `[9419, 11, 353, 1044, 248046]`
  (`Hello, I am`); the pinned llama.cpp server independently produced the
  identical `", I am"` continuation from the same prompt, grammar, and
  capture trigger.
- Sampler state v4 records variable-length accepted-token replay for GBNF and
  adaptive-p EMA state, validates it before mutation, and loads compatible
  v2/v3 state. CLI
  `-grammar`/`-grammar-file` and HTTP `grammar` fields bind source to the real
  model vocabulary. On Qwen3.5, Go and pinned llama.cpp both greedily produced
  `Return a boolean object:{"ok":true}` under the same recursive whitespace
  and boolean-alternative grammar, with identical Go token IDs asserted in
  the CUDA integration suite.
- Ordered JSON-Schema conversion matches all 70 shared cases in the pinned
  llama.cpp `test-json-schema-to-grammar.cpp` corpus plus its two C++-only
  non-capturing regular-expression cases. The fixture preserves source object
  order and covers references (including array JSON pointers and cycles),
  unions, constants/enums, bounded integers/strings/arrays, object required and
  additional-property rules, `allOf`, and anchored regular expressions. Every
  successful oracle grammar also compiles through the Go GBNF parser. The
  `json-schema-grammar` CLI accepts a file or stdin; native `/completion` and
  `/completions` compile `json_schema` with root `root` before generation and
  reject conflicts with explicit grammar controls. OpenAI `/v1/completions`
  accepts the same top-level schema. Chat completions accept top-level
  `json_schema` plus `response_format` modes `text`, `json_object`, and the
  wrapped OpenAI `json_schema.schema` form; all structured-output paths share
  the converter, compile before generation, and reject grammar conflicts and
  malformed envelopes.
- Teacher-forced perplexity uses batched CUDA output projection and stable
  log-sum-exp scoring, with optional per-token NLL output. Its `-ctx` mode
  reproduces llama.cpp's disjoint-window/latter-half convention: on the local
  Qwen3.5 fixture and a 121-token repeated-text corpus, Go reports `1.079212`
  versus the pinned CPU oracle's `1.0785` (0.066% relative difference).
- KV cache state has a versioned little-endian save/load format with strict
  rank/shape/count/layer/truncation checks. A real Qwen3 save→load→incremental
  decode matches full forward within `3.0517578125e-05` maximum absolute error.
- Cache state v2 separates active attention length from the absolute
  next-token position and loads v1 state compatibly. `ShiftCache` removes a
  dense-attention prefix without aliasing its input, retains Qwen3.5 recurrent
  state, and preserves absolute RoPE positions. Opt-in rolling context is
  available through generation/session options and the CLI/server
  `-context-shift` flag. A real Qwen3 CUDA session continued beyond an
  artificially reduced two-token context with active/history/position
  invariants intact.
- Native Q1_0/Q2_0, Q4_0/Q4_1, Q5_0/Q5_1, Q8_0/Q8_1/Q8_K, Q2_K-Q6_K,
  TQ1_0/TQ2_0, IQ4_NL/IQ4_XS, and MXFP4/NVFP4 CUDA
  embedding and matrix multiplication consume persistent raw GGUF block
  layouts directly; other tensors share the F32 device bridge. Device results
  match the host decoders on synthetic packed block fixtures.
- Greedy generation on the local Qwen3 4B model matches the pinned llama.cpp
  CPU oracle for three generated tokens: prompt `Hello` produces IDs
  `[9707, 27, 18, 198]`, or `Hello<3\n`.
- The native-Q8 path completes this three-token check in about 4.6 seconds
  including `go run` startup, versus about 8.4 seconds for full F32 preload.
- Q6_K's 210-byte super-block layout is decoded on the host and consumed
  directly by native CUDA embedding/matrix kernels. The local Qwen3 4B Q6_K
  fixture produces the same `[9707, 27, 18, 198]` oracle sequence in about
  3.8 seconds including `go run`, versus about 10.5 seconds through F32 preload.
- Q2_K, Q3_K, Q4_K, and Q5_K host decoders implement the pinned 256-value
  super-block layouts with exact packed scale/min/high-bit tests.
- The HTTP surface provides health/model discovery and bounded non-streaming or
  SSE-streaming `/v1/completions`, with cancellation, admission control, and
  graceful process shutdown.
- `/metrics` emits dependency-free Prometheus text for uptime, total and active
  HTTP requests, generation attempts/errors, and generated tokens. Accounting
  is atomic across concurrent streaming and non-streaming requests. Runner
  instances also export current/peak CUDA bytes and live allocation count;
  failed optional snapshots do not fail the metrics endpoint.
- Optional bearer authentication protects generation/embedding `/v1/*` routes and the
  llama.cpp-compatible `/apply-template`, `/tokenize`, `/detokenize`, and
  `/props`, `/slots`, `/completion`, `/completions`, `/chat/completions`,
  chat token-count, `/embedding`, and `/embeddings` utility routes with
  constant-time token comparison. Secrets are loaded from
  `LLAMACPP2GO_API_KEY` or `-api-key-file`; health and metrics remain available
  to orchestration probes. Native and OpenAI model-discovery routes remain
  public like the pinned server.
- `/v1/embeddings` accepts bounded string batches and returns mean-pooled,
  L2-normalized final hidden states with token usage. A real Qwen3 native-Q8
  integration check validates embedding width and unit norm.
- `/v1/chat/completions` provides non-streaming and SSE responses for ChatML
  vocabularies. Formatting validates roles, rejects boundary injection, and
  tokenizes boundary markers as control tokens; a real Qwen3 decode passes.
- Dense Llama graph construction uses consecutive-pair normal RoPE rather
  than Qwen3's split-half NeoX layout. Converted Llama 3
  `rope_freqs.weight` tensors apply one validated divisor per rotary pair on
  both the CPU reference and CUDA paths, including streamed, preloaded-F32,
  and native-quantized execution. Unsupported long/short RoPE factors,
  YaRN, and LongRoPE modes are rejected before execution. Metadata-driven
  linear scaling is applied to Llama normal RoPE and Qwen/Gemma NeoX RoPE and
  composes with per-pair factors.
- Dense attention Q/K/V/output and feed-forward gate/up/down projection biases
  are shape-validated, loaded through streamed, preloaded-F32, and
  native-quantized paths, and applied with broadcast additions on CPU and
  CUDA.
- Qwen3.5 prerequisite graph operations Sigmoid, numerically stable Softplus,
  and row-wise L2Norm match between the CPU reference and CUDA executors.
- Qwen3.5 projection layout extraction, rank-2 transposition, axis-zero
  concatenation, channel-wise SSM convolution, and the fused K=1 gated delta
  net match the CPU reference on CUDA for scalar/vector gates and shared Q/K
  heads.
- Multi-axis RoPE implements the pinned split-half temporal/height/width/extra
  section mapping on CPU and CUDA. Qwen3.5 also honors an explicit
  `attention.recurrent_layers` boolean array when present, with the
  full-attention interval retained as the upstream-compatible fallback.
- The local Qwen3.5 9B Q8_0 hybrid model executes all 32 layers: 24 recurrent
  gated-delta-net layers and eight gated full-attention layers. Greedy
  generation exactly matches the pinned llama.cpp oracle for three generated
  tokens: `Hello` produces IDs `[9419, 11, 353, 1044]`, or `Hello, I am`.
- The local Bonsai 27B Q1_0 Qwen3.5 hybrid executes all 64 layers with 498
  tensors retained in native one-bit CUDA storage. For two greedy tokens,
  both Go and the pinned CPU `llama-completion` oracle produce `Hello, I`;
  Go exposes exact IDs `[9419, 11, 353]`. The optional
  `LLAMACPP2GO_BONSAI_MODEL` integration test preserves this differential.
- Qwen3.5 hybrid cache serialization carries fixed convolution/delta-net state
  for recurrent layers and append-only KV tensors for full-attention layers.
  A save/load session resumed after the first generated token exactly
  reproduces the oracle sequence.
- Qwen3.5-MoE reuses both hybrid cores and replaces every dense FFN with
  normalized softmax top-k routed SwiGLU experts plus a sigmoid-gated shared
  SwiGLU expert. Separate and fused expert gate/up catalogs are accepted;
  metadata, strict catalog, graph semantics, and full-attention/recurrent
  reference/CUDA differentials pass. MTP remains external, and real-model
  validation awaits a local fixture.
- Qwen3-Next executes gated NeoX full attention and recurrent GDN layers with
  optimized QKV-plus-gate or legacy grouped QKVZ projection layouts. Its fused
  beta/alpha projection, adjacent value-head key-group repetition, normalized
  routed SwiGLU experts, and sigmoid-gated shared expert match the pinned
  graph. Metadata, both recurrent catalogs, host/F32/native tensor propagation,
  graph topology, and full-attention/recurrent reference/CUDA differentials
  pass. Real-model validation awaits a local fixture.
- The real Gemma 3 12B IQ4_XS file now passes strict metadata and all 48-layer
  weight catalog validation. Its graph executes with embedding scaling,
  per-head Q/K norms, alternating local/global scaled RoPE, sliding attention,
  attention/FFN post norms, GEGLU, and optional metadata-driven final-logit
  softcapping on native IQ4_XS/Q6_K CUDA weights. Softcapping is applied
  consistently to streamed, preloaded/device-cache, and perplexity logits.
  The oracle-compatible evaluator now resets BOS at every disjoint window, as
  llama.cpp does for BOS-enabled vocabularies. On a diverse 126-token probe,
  Go reports PPL `33.2276` versus the pinned CPU oracle's `33.1217` (0.32%
  relative delta). The pinned CPU server now provides raw generation IDs:
  for `Hello`, both runtimes produce generated IDs
  `[255999, 1018, 3689]`, yielding `Hello**What` after the prompt IDs.
- Gemma 2 metadata and dense graphs are supported with the upstream defaults
  for alternating 4096-token sliding attention, split-half NeoX RoPE,
  embedding and query scaling, attention-logit and final-logit softcapping,
  post-attention/post-FFN norms, and GEGLU. Attention softcapping has CPU/CUDA
  differential coverage and kernel ABI v3 validation. The tensor catalog and
  graph currently have synthetic coverage; real-model oracle validation is
  pending a local Gemma 2 fixture.
- Original Gemma metadata and dense graphs are supported with tied output
  embeddings, embedding/query scaling, split-half NeoX RoPE, and GEGLU without
  the later Gemma post norms. Metadata, tensor-catalog, and graph coverage is
  synthetic pending a local real-model fixture.
- Dense Qwen 2 metadata, tensor catalogs, and graphs are supported with
  separate Q/K/V biases, NeoX RoPE, RMSNorm, and SwiGLU. Optional
  vocabulary-wide output bias is applied before final-logit softcapping in
  streamed, preloaded/device-cache, and perplexity execution. Real-model
  differential validation remains pending a local Qwen 2 fixture.
- InternLM2 and EXAONE metadata, dense tensor catalogs, and graphs are
  supported. Both use RMSNorm, separate Q/K/V projections, and SwiGLU;
  InternLM2 uses normal consecutive-pair RoPE and requires an untied output
  projection, while EXAONE uses NeoX RoPE with optional per-pair factors and
  an optional tied output fallback. Real-model validation is pending fixtures.
- EXAONE 4 decoders support post-only attention/FFN RMSNorm, fused or separate
  QKV with optional bias, per-head Q/K RMSNorm, NeoX RoPE with optional
  factors, SwiGLU, and tied or untied output. The 64-layer cadence uses a 4096
  sliding window with RoPE on sliding layers and periodic full attention
  without RoPE. NextN/MTP layers are rejected.
- XVERSE uses the same strict RMSNorm/SwiGLU dense catalog and normal RoPE as
  InternLM2, including a required untied output projection. Metadata, catalog,
  and graph coverage is synthetic pending a real-model fixture.
- OLMo2 post-normalized dense blocks are supported with full-projection Q/K
  RMSNorm, attention and FFN post norms, NeoX RoPE, and optional four-layer
  sliding-attention cadence. Streamed and preloaded weight adapters accept its
  intentionally absent attention/FFN pre-norm tensors. Real-model validation
  is pending a local fixture.
- SmolLM3 dense decoders support metadata-driven attention scale and the
  upstream cadence that leaves every fourth block without RoPE. Its remaining
  layers use normal consecutive-pair RoPE with the standard RMSNorm/SwiGLU
  catalog. Real-model validation is pending a local fixture.
- MiniCPM dense decoders support their backward-compatible and metadata-
  overridden embedding, residual-branch, and inverse-logit scales across host
  and retained-device inference. Standard and linear-scaled normal RoPE are
  supported; LongRoPE variants remain pending. Real-model validation is pending
  a local fixture.
- Text-only Granite and GraniteMoE decoders support metadata-driven embedding,
  residual-branch, attention, and inverse-logit scales, optional no-RoPE mode,
  and LongRoPE short/long factor selection. GraniteMoE adds normalized softmax
  top-k gated or ungated SiLU experts plus an optional shared SwiGLU expert.
  Vision deepstack remains rejected explicitly. Real-model validation is
  pending a local fixture.
- Maincoder dense decoders support normal consecutive-pair RoPE followed by
  per-head Q/K RMSNorm, preserving the upstream operation order, with the
  standard tied-output RMSNorm/SwiGLU catalog. Real-model validation is pending
  a local fixture.
- Dense Mistral 3 decoders support normal consecutive-pair RoPE, optional
  projection/MLP biases, optional tied output, and metadata attention scale.
  Expert, attention-temperature, YaRN, and LongRoPE variants are rejected by
  metadata or the shared advanced-RoPE guard. Real-model validation is pending
  a local fixture.
- Orion dense decoders support affine LayerNorm (including attention, FFN, and
  final-output biases), split-half NeoX RoPE, parallel SwiGLU, and the
  required untied output projection. Host and preloaded-device weight adapters
  carry every norm bias. Real-model validation is pending a local fixture.
- StarCoder2 dense decoders support affine LayerNorm, split-half NeoX RoPE,
  required attention-output/up/down projection biases, sequential GELU FFNs
  without a gate tensor, and optional tied output embeddings. Real-model
  validation is pending a local fixture.
- CodeShell reuses the affine-LayerNorm, NeoX-RoPE, sequential-GELU path with
  required projection biases and an untied output projection. Its optional
  token embedding correctly falls back to the output table when absent.
  Real-model validation is pending a local fixture.
- Baichuan 7B supports the 32-layer RMSNorm/SwiGLU graph, normal
  consecutive-pair RoPE, and required untied output projection. The 40-layer
  Baichuan 13B ALiBi variant is rejected explicitly until ALiBi attention is
  available. Real-model validation is pending a local fixture.
- Arcee dense decoders support RMSNorm, metadata attention scaling, optional
  tied output embeddings, normal RoPE with optional per-pair factors, and the
  gate-free squared-ReLU FFN. Real-model validation is pending a local fixture.
- Nemotron dense decoders support affine LayerNorm, split-half NeoX RoPE,
  gate-free squared-ReLU FFNs, required untied output weights, and optional
  attention-output/up/down biases. Real-model validation is pending a local
  fixture.
- Jais2 dense decoders support affine LayerNorm, split-half NeoX RoPE, required
  Q/K/V/output and MLP biases, gate-free squared-ReLU FFNs, and optional tied
  output embeddings. Non-MHA metadata is rejected because upstream Jais2 bias
  tensors use the full embedding width. Real-model validation is pending.
- Original OLMo dense decoders support tensor-less LayerNorm before attention,
  FFN, and output, plus normal RoPE, SwiGLU, and optional tied output. Nonzero
  QKV-clamp metadata is rejected until a clamp primitive is available.
- Seed-OSS dense decoders support RMSNorm, metadata attention scaling, NeoX
  RoPE, SwiGLU, and optional tied output. Its `post_attention_norm.weight` is
  mapped to the pre-FFN normalization slot matching upstream execution.
- Dense Cohere2 decoders support weight-only LayerNorm, parallel attention and
  SwiGLU residual branches, partial normal RoPE on sliding layers, periodic
  full attention without RoPE, direct logit scaling, and tied output.
- Cohere2-MoE decoder trunks support RMSNorm or weight-only LayerNorm, leading
  dense SwiGLU blocks, per-layer or periodic sliding attention, dense-prefix
  and sliding-layer normal RoPE, metadata-normalized sigmoid top-k routing, fused or
  separate expert gate/up storage, optional shared-expert half averaging,
  direct logit scaling, and optional tied output. Declared NextN/MTP layers are
  excluded from ordinary decoder execution; speculative prediction is pending.
- Command R decoders support weight-only LayerNorm, parallel attention and
  SwiGLU residual branches, normal RoPE, optional direct logit scaling, tied
  output, and the 64-layer variant's per-head weight-only Q/K norms.
- Original PLaMo decoders support RMSNorm, parallel attention and SwiGLU
  residual branches, NeoX RoPE, grouped-query attention, and untied output.
- PLaMo 3 decoders support per-layer attention/FFN widths, fused QKV,
  pre-RoPE per-head Q/K RMSNorm, NeoX RoPE with scalar or per-layer sliding
  attention, attention/FFN post-norms, fused gate/up SwiGLU, and optional tied
  output. Metadata, strict catalog, graph, and reference/CUDA block
  differentials pass; real-model validation remains pending a local fixture.
- StableLM decoders support affine LayerNorm, partial NeoX RoPE, SwiGLU,
  optional per-head weight-only Q/K norms, and both sequential-normalized and
  parallel-residual FFN layouts selected from the validated tensor catalog.
- Phi-2 decoders support affine LayerNorm, contiguous fused or separate Q/K/V
  projections with optional biases, partial NeoX RoPE, query pre-scaling,
  parallel GELU FFNs, required projection/output biases, and untied output.
- GPT-NeoX decoders support affine LayerNorm, required contiguous fused QKV
  projections and biases, partial NeoX RoPE, GELU FFNs, and metadata-selected
  parallel or sequential residual topology.
- Falcon 7B/40B decoders support affine LayerNorm, required contiguous fused
  QKV, full NeoX RoPE, parallel GELU FFNs, tied or untied output, and the 40B
  variant's optional second attention normalization.
- Dense Phi-3 decoders support RMSNorm, fused or separate QKV, query
  pre-scaling, fused gate/up SwiGLU FFNs, tied or untied output, and validated
  LongRoPE short/long factors selected from configured versus original context.
- PhiMoE reuses the Phi-3 attention path with affine RMSNorm biases, mandatory
  output/attention biases, and normalized softmax top-k packed SwiGLU experts.
- BitNet decoders support RMSNorm, attention/FFN sub-layer RMSNorms, optional
  scalar projection scales, SwiGLU, tied output, and existing ternary tensor
  decoding across streamed host and device execution paths.
- Apertus decoders support RMSNorm, fused or separate QKV with optional biases,
  per-head Q/K RMSNorm, full NeoX RoPE with ordinary or LongRoPE factors,
  metadata attention scaling, ungated xIELU FFNs with scalar-or-per-layer
  parameters, optional attention-output bias, and required untied output.
- Text-only GLM4 decoders support pre/post RMSNorm, fused or separate QKV with
  optional bias, partial normal RoPE, fused gate/up SwiGLU, and tied or untied
  output. Multimodal RoPE sections and NextN/MTP layers are rejected.
- The real UMT5 XXL encoder passes strict 24-layer T5 metadata and tensor
  catalog validation. Its SentencePiece UGM tokenizer produces
  `[23231, 3914, 332]` for `Hello world!`, exactly matching the pinned
  tokenizer oracle, and adds EOS token `1` for encoder execution.
- The T5 graph executes full bidirectional no-cache attention with per-layer
  32-bucket relative-position bias, RMSNorm, GEGLU, residuals, and final
  encoder normalization on native Q5_K/Q6_K CUDA weights. For `Hello world!`,
  the 4,096-dimensional mean-pooled vector has cosine similarity `0.9994314`,
  RMSE `0.0012743`, and a `0.17%` norm delta versus pinned llama.cpp CPU
  execution. The oracle command must specify `--attention non-causal`; its
  default for this fixture is causal.

## Current runtime limitations

- Preloaded dense generation retains KV state on-device between ordinary
  decoding calls and device-backed prompt reuse can require zero submissions.
  Qwen3.5 direct generation likewise retains hybrid state on-device. Editable
  sessions and streamed-weight execution use the serialized host cache.
  Ordinary rolling context shift advances attention-cache device pointers and
  shrinks logical token dimensions while preserving absolute positions and
  recurrent state. Combined immutable prompt caching plus shifting clones the
  retained device ranges before the first edit. Native completion `n_keep` performs exact prefix-preserving
  middle-range compaction in owned CUDA allocations using direct
  device-to-device range copies; zero-prefix shifts remain pointer views.
  `n_discard` accepts an explicit bounded removal size, with zero using the
  pinned half-window default. Device paging, multiple sequences, and
  continuous batching remain pending.
- Native device execution currently supports Q1_0/Q2_0, Q4_0/Q4_1,
  Q5_0/Q5_1, Q8_0/Q8_1/Q8_K, Q2_K-Q6_K, TQ1_0/TQ2_0,
  every pinned IQ1/IQ2/IQ3/IQ4 layout, and MXFP4/NVFP4 matrices. This covers
  every quantized tensor type currently declared by the pinned GGML format
  map; scalar F16/BF16/F64 and integer storage still use F32 conversion.
- Default-mode tied-output logits stream bounded GGUF chunks on the CPU;
  preload modes compute logits on the GPU.
- CUDA memory metrics cover allocations owned by this runtime, not driver
  overhead or allocations made by other processes in the same device context.
- Character, token-terminal, and lazy-trigger GBNF are supported. Trigger
  patterns use Go RE2; llama.cpp ECMAScript constructs such as lookaround and
  backreferences remain pending. General and infill sampler stages have
  arbitrary configurable ordering.
- Text and function-tool GGUF Jinja templates are supported. Custom
  llama.cpp-only Jinja extensions not implemented by gonja, token-incremental
  streaming tool-call argument deltas, and continuous-batching server
  scheduling remain pending. Tool-enabled streams currently wait for complete
  validated output before emitting structured call deltas.
- Chat generation/counting accepts string message bodies, text content-part
  arrays, tool schemas, assistant calls/results, and the implemented
  system/user/assistant/tool role families. Buffered and complete-call SSE
  OpenAI function calls plus tool-aware token counting are supported;
  images/audio/files remain pending.
- Text-only OpenAI Responses generation, streaming, token counting, and
  function tools/history are supported. Continuation IDs, hosted/custom
  tools, reasoning items, multimodal/file inputs, and token-incremental
  function-argument deltas remain pending.
- Text-only Anthropic generation, streaming, counting, tool use/results, and
  tool-aware Jinja contexts are supported. Thinking blocks, images, and
  token-incremental tool-input deltas remain pending.
- LoRA model loading, tensor application, and per-request/global scaling remain
  pending; the control-plane endpoint does not claim adapters are loaded.
- Native completion supports strings, exact/mixed token sequences, bounded
  batches of either, and opt-in reuse from the best retained prefix meeting
  `n_cache_reuse`. It reports cached/evaluated token counts and
  separate prompt/decode timing. Context shifting accepts `n_keep`, including
  `-1` for the largest safe initial-prompt prefix, plus `n_discard`, and reports
  both in generation settings and slot state. Pure-attention prompt caches
  reuse their longest common prefix through host or zero-copy device suffix
  rollback; recurrent hybrids retain exact-prefix-only reuse. A configurable
  bounded LRU retains independent host or CUDA prompt states. Multimodal prompt
  objects and per-request LoRA are rejected explicitly rather
  than silently approximated. Positive `n_probs`
  reports selected and top-N raw-logit softmax log probabilities with token
  pieces and byte arrays in buffered and SSE responses.
  `post_sampling_probs` instead reports normalized `prob`/`top_probs` from the
  sampler's actual filtered candidate set, including greedy, Mirostat, and
  adaptive paths. Native SSE
  supports the pinned 30-second default comment heartbeat, positive integer
  `sse_ping_interval` overrides, and `-1`/`0` disabling. `return_progress`
  emits exact prompt-start and prompt-complete envelopes; internal per-batch
  increments are unavailable because the Runner evaluates a prompt as one
  synchronous graph. `timings_per_token` adds measured cache/prompt/decode
  timings and zero-safe rates to each native token event. Positive
  `t_max_predict_ms` uses the Runner's graceful post-token stop hook to end on
  the first newline after the prediction budget, preserving final IDs/text;
  `0` and `-1` disable the limit. Positive `n_indent` stops when a generated
  post-newline line has insufficient leading spaces/tabs and trims the
  offending non-whitespace suffix.
- `/slots` exposes authoritative admission state, task ID while busy, context
  length, prompt/cache counters, prompt and generated text, and measured timing
  rates. Its `params` object retains the effective sampler, stop, cache, and
  context-shift settings. Its `next_token` object reports decoded/remaining
  counts and truthfully reports no pending token in the synchronous runtime.
- Llama execution currently covers dense models with optional projection
  biases and ordinary RoPE
  or converted Llama 3 per-pair frequency factors. LongRoPE/YaRN tensor
  selection, MoE, and model-specific attention variants are rejected.
- The Qwen3.5 graph supports true multi-axis positions, but the text generation
  API currently supplies the same sequential position to every MRoPE axis.
  Image/video input plumbing, MTP heads, rollback snapshots, and
  multi-sequence recurrent batching remain pending.
- T5 support currently covers encoder-only models and embeddings. T5 decoder,
  encoder-decoder cross-attention, padding masks for multi-sequence batches,
  and token generation remain pending.

## Blockers

### Deferred: importance-matrix-only IQ encoders

The pinned IQ1_S, IQ1_M, IQ2_XXS, and IQ2_XS row quantizers do not expose
values-only reference encoders. Their production paths require per-element
importance weights and, for the IQ2 variants, assert when those weights are
absent. The current `Quantize` and `gguf-quantize` contracts accept only tensor
values, so these four destinations are rejected rather than silently using
invented weights. Adding them requires an explicit weighted quantization API
and a compatible importance-matrix input format; all values-only reference
layouts continue independently.

### Deferred: LoRA graph-wide projection application and oracle

Pinned LoRA application is graph-time
`base*x + adapter_scale*B*(A*x)` for every adapted projection, including
embeddings and output. The current streamed, preloaded-F32, and
native-quantized paths expose only base tensors and would each need adapter
graph inputs plus cache/session topology binding. Destructively merging deltas
into quantized weights would change rounding and prevent per-request scaling.
No local LoRA GGUF adapter is available for a differential oracle, so
loading/application remains deferred while independent work continues.

### Deferred: Go vet cannot prove CUDA C-string pointer provenance

The CUDA error-name/error-string APIs return driver-owned `const char *`
values through the Windows DLL syscall boundary, whose Go representation is
necessarily `uintptr`. The bounded `readCString` conversion is annotated
`//go:nocheckptr` and live driver tests pass, but `go vet ./...` still reports
`possible misuse of unsafe.Pointer` because its analyzer cannot track pointer
provenance across `Proc.Call`. Package tests and scoped vet checks remain
enabled; replacing this ABI conversion solely to silence the analyzer would
require an extra native shim and violate the no-cgo design.

### Deferred: release signing and project license declaration

Reproducible unsigned archives are implemented. Authenticode signing requires
a user-controlled code-signing certificate/private key and timestamping
policy, neither of which is present or safe to invent. The repository owner
also has not declared a license for the Go/kernel code; SBOM and license
inventory therefore use `NOASSERTION`. Public distribution should wait for
both decisions.

### Deferred: Go race detector under the no-cgo contract

The standard Go race detector requires cgo on this host, while this port's
deployment contract requires `CGO_ENABLED=0`. `go test -race` therefore fails
before compilation. Enabling cgo solely for that check would test a different
binary contract. Deterministic concurrency/contention tests remain enabled;
race instrumentation is deferred until Go supports it for the no-cgo Windows
target or a suitable alternative is integrated.

### Deferred: CUDA 12.9 with Visual Studio 2026

`nvcc` 12.9 rejects the installed Visual Studio 2026 compiler, including with
`-allow-unsupported-compiler`. Kernel compilation uses the installed Visual
Studio 2019 Build Tools until CUDA 13.2 or a supported newer toolkit is
installed. Runtime execution is unaffected.

### Deferred: local ternary Q2_0 fixture layout mismatch

`Ternary-Bonsai-27B-Q2_0.gguf` encodes a Q2_0 tensor with an effective 17-byte
block, while the pinned llama.cpp commit defines Q2_0 as an 18-byte block. The
reader rejects the resulting non-contiguous offsets. Do not add a heuristic
layout variant without a metadata/version discriminator. Other tested F32,
BF16, Q8_0, Q6_K, IQ4_XS, and Q1_0 models parse successfully.

### Deferred: Go race detector under no-cgo builds

The Go race detector requires cgo on Windows, while the product build requires
`CGO_ENABLED=0`. Normal unit and CUDA integration tests remain no-cgo. A
separate cgo-enabled diagnostic test job can be added later without changing
release binaries.

### Deferred: real Llama end-to-end fixture

The available local GGUF inventory has Qwen3, Qwen3.5, Gemma, T5, diffusion,
and ternary fixtures but no real Llama weight file. The pinned upstream
Llama SPM/BPE vocabulary fixtures and synthetic dense-block CPU/CUDA tests,
including converted Llama 3 frequency factors, pass. Final logits, greedy
tokens, and performance for the Llama path remain deferred until a compatible
dense Llama GGUF is available.

### Deferred: real-model infill oracle

The pinned infill probability sampler, vocabulary-piece plumbing, FIM metadata
resolution, PSM/SPM formatter, and native endpoint are implemented. None of
the local GGUF fixtures contains FIM metadata or recognized FIM control-token
spellings, so end-to-end token/output differential validation remains
deferred until a compatible model is available.

### Deferred: continuous-batching graph and cache architecture

The HTTP layer can admit multiple slots, but `Runner.Generate` intentionally
holds one model-wide lock across prompt evaluation, decode, callbacks, and
cache mutation, and the CUDA worker owns one serial stream. True continuous
batching requires sequence-tagged device KV/recurrent caches, variable active
batch graph shapes, per-sequence sampling/stop state, and cancellation-safe
compaction. Concurrent admission is not reported as batching; this tranche is
deferred while independent compatibility work continues.

### Deferred: RWKV and PLaMo2 tokenizer oracle fixtures

The pinned source contains RWKV and PLaMo2 tokenizer implementations but no
checked-in GGUF vocabulary/oracle pair for either algorithm, and the local
model inventory has neither family. WordPiece work proceeded against the
available BERT fixture. RWKV/PLaMo2 tokenization remains deferred rather than
claiming an unverified port; independent compatibility work continues.

### Deferred: remaining fused-QKV decoder families

Contiguous fused QKV projection and bias loading/slicing is shared by the
dense host, preloaded-device, and cache execution paths and is enabled for
Phi-2, GPT-NeoX, and Falcon. Remaining fused-QKV families stay
architecture-gated until their distinct position encoding, residual topology,
normalization, and tensor-layout rules are implemented and tested; the
presence of `attn_qkv` alone is not treated as proof of compatibility.

BERT now adds optional sentence-A token-type row 0 and learned absolute-position
rows before affine embedding LayerNorm. Its bidirectional no-cache encoder uses
optional fused QKV and projection/MLP biases, GELU, and affine post-attention
and post-FFN LayerNorm without a decoder output projection. Metadata, strict
catalog, graph ordering, and a complete reference/CUDA block differential pass.
The optional classifier/pooler tensors remain outside the hidden-state embedding
boundary; real-model execution is pending a local weight fixture.

NeoBERT now executes fused-QKV bidirectional attention with normal RoPE,
pre-RMSNorm residual blocks, fused SwiGLU storage, and final encoder RMSNorm.
Metadata, strict catalog, graph semantics, and a complete reference/CUDA block
differential pass. Optional classifier tensors remain outside the hidden-state
embedding boundary; real-model execution is pending a local weight fixture.

Dense NomicBERT now executes optional fused QKV, bidirectional NeoX RoPE,
affine embedding/post norms, and gated SwiGLU without a decoder projection.
Metadata, strict catalog, graph semantics, and a complete reference/CUDA block
differential pass.

NomicBERT-MoE now follows its exact alternating cadence, with dense GELU blocks
and gate-free softmax top-k GELU expert banks. Metadata, strict per-layer
catalogs, graph semantics, and a complete reference/CUDA differential pass.

JinaBERT v2 now executes fixed bidirectional ALiBi, optional fused QKV and
affine Q/K norms, optional secondary attention residual normalization, and
plain GELU or separate/fused GEGLU FFNs. Metadata, strict catalog, topology
variants, and a complete reference/CUDA block differential pass.

Dense JinaBERT v3 now executes optional fused QKV, bidirectional NeoX RoPE,
affine embedding/post norms, and GELU FFNs without a decoder projection.
Metadata, strict catalog, graph semantics, and a complete reference/CUDA block
differential pass. Expert metadata and cadence are rejected explicitly.

Llama Embed now reuses the dense and Mixtral-style Llama tensor catalogs and
normal-RoPE transformer blocks with bidirectional no-cache attention. Final
RMS-normalized hidden states form the encoder boundary; vocabulary logits are
rejected even when an optional output projection is present. Metadata, dense
and MoE catalogs, graph semantics, and a reference/CUDA differential pass.

Pangu Embedded now executes its causal RMSNorm/SwiGLU decoder and exposes the
same final normalized states used by its optional tied vocabulary projection.
Separate or fused QKV, required attention-output bias, NeoX RoPE, and optional
LongRoPE factors are covered by strict catalogs and a reference/CUDA block
differential. Real-model execution is pending a local Pangu GGUF fixture.

ModernBERT now executes embedding LayerNorm, layer-0 identity attention norm,
fused QKV, NeoX RoPE, dense-first periodic symmetric attention, and fused
GEGLU or SwiGLU selected by metadata. Final weight-only LayerNorm exposes
hidden states without a vocabulary projection. Metadata, strict catalog,
graph variants, and a reference/CUDA differential pass; real-model execution
is pending a local ModernBERT GGUF fixture.

Gemma Embedding now executes scaled token embeddings, per-head Q/K RMSNorm,
NeoX RoPE, periodic full and symmetric local attention, branch post-RMSNorm,
and parallel GEGLU. Optional sentence-transformer dense-2/dense-3 matrices run
after pooling and before normalization on bounded-host and preloaded CUDA
paths. Metadata, strict fused/separate-QKV catalogs, graph variants, host
projection tests, and a reference/CUDA block differential pass; real-model
execution is pending a local Gemma Embedding GGUF fixture.

Bloom now applies its input embedding LayerNorm and llama.cpp-compatible ALiBi.
GPT-2 and StarCoder gather learned absolute-position rows before the first
block. Compatible MPT variants support optional learned positions and ALiBi;
Q/K-LayerNorm, activation-scale, and clamped-QKV variants remain rejected.
Dense Refact uses ALiBi. Qwen3-MoE loads its router and packed expert gate/up/down
tensors and executes normalized softmax top-k routing with routed weight scaling
through the bounded-host, F32-preload, and native-quantized expert paths. This
matches the complete Qwen3-MoE definition in the pinned llama.cpp source.

Mixtral now detects expert-bearing `llama` GGUF metadata without changing the
dense Llama path, loads the router and packed SwiGLU expert bank, and executes
normalized softmax top-k routing through bounded-host, F32-preload, and
native-quantized CUDA expert paths. Metadata, strict catalog, graph semantics,
and complete reference/CUDA block differential tests pass. Real-model
validation remains pending because no Mixtral GGUF is available locally.

The fused routed-expert kernel accepts F32 or any native-quantized gate/up/down
bank supported by the device decoder set. Packed blocks remain resident in
their GGUF layout and decode inside the kernel; router, activations,
accumulation, selection bias, and output remain F32. Softmax and sigmoid
routing, selected-probability normalization, and routed scaling share this
path. CUDA differentials cover classic Q4, Q8, K-quant, IQ, ternary, float4,
and small-block layouts against their dequantized F32 references.

BailingMoE now supports its dedicated BPE pre-tokenizer and loads its required output projection, standard Q/K/V attention,
packed routed experts, and always-on shared expert bank. Its normal RoPE,
optional selected-probability normalization, routed scaling, and parallel shared
SwiGLU branch execute through bounded-host, F32-preload, and native-quantized
expert paths. Metadata, strict catalog, graph semantics, and complete
reference/CUDA block differential tests pass. Real-model validation remains
pending because no BailingMoE GGUF is available locally.

DeepSeek v1 now supports its exact `deepseek-llm` BPE pre-tokenizer, including
the pinned tokenizer's restricted alphabet and punctuation ranges. The model
loads optional leading dense SwiGLU blocks followed by unnormalized softmax
top-k routed experts, routed scaling, and the always-on shared SwiGLU expert
bank. Output projection remains optional with tied token-embedding fallback.
The complete 46-case upstream tokenizer oracle, metadata, strict mixed-layer
catalog, graph semantics, and reference/CUDA block differential pass. The local
fixture inventory contains only tokenizer data, so real-model logit validation
remains pending-fixture.

GraniteMoE now loads normalized softmax top-k packed experts with either gated
SwiGLU or ungated SiLU activation, plus its optional shared SwiGLU expert. It
inherits Granite's embedding, residual, attention, inverse-logit, no-RoPE, and
LongRoPE behavior. Metadata, strict catalog, graph semantics, primitive
reference coverage, and complete reference/CUDA block differential tests pass.
Real-model validation remains pending because no GraniteMoE GGUF is available
locally.

DBRX now maps its `dbrx` BPE pre-tokenizer to the pinned Llama 3 segmentation,
loads required fused QKV and untied output tensors, applies its symmetric QKV
clamp before projection slicing, and executes weight-only LayerNorm, NeoX RoPE,
and normalized softmax top-k packed SwiGLU experts. Metadata, strict catalog,
graph ordering, clamp primitive coverage, and complete reference/CUDA block
differential tests pass. Real-model validation remains pending because no DBRX
GGUF is available locally.

Grok now loads optional fused or separate QKV projections, optional gated
expert and dense GEGLU branches, required attention/FFN post-norm weights, and
optional tied output projection. Its graph applies metadata/default embedding,
attention, and direct-logit scales; NeoX or YaRN RoPE; softcapped attention;
normalized softmax top-k packed GELU/GEGLU experts; and the pinned dense/MoE
blend. Reference and CUDA differential tests cover ungated-only and
gated-plus-dense blocks. Real-model validation remains pending because no Grok
GGUF is available locally.

Mellum now loads required untied output, separate QKV and per-head Q/K RMSNorm
tensors, plus normalized softmax top-k packed SwiGLU experts. Scalar or
per-layer sliding-window patterns select bounded attention; sliding layers use
unscaled NeoX RoPE while full layers retain linear or YaRN scaling. Metadata,
catalog, alternating graph, and reference/CUDA differential tests pass.
Real-model validation remains pending because no Mellum GGUF is available
locally.

SmallThinker now loads optional fused or separate QKV projections and packed
ReGLU experts, computes router logits from the pre-attention residual while
feeding normalized post-attention state to the experts, and honors metadata
selected softmax or sigmoid routing. Its dense-first periodic sliding schedule,
4096-token effective window, SWA RoPE base, and no-RoPE full-attention layers
match the pinned implementation. Metadata, strict catalog, graph semantics,
split-router primitive tests, and complete reference/CUDA block differentials
pass. Real-model validation remains pending because no SmallThinker GGUF is
available locally.

DOTS1 now loads required untied output, optional fused or separate QKV,
per-head Q/K RMSNorm, leading dense SwiGLU blocks, and later packed routed
experts. Its metadata-selected softmax or sigmoid routing supports optional
selection correction bias, selected-weight normalization, routed scaling, and
the required shared SwiGLU bank. Metadata, strict mixed-layer catalog, graph
semantics, and a complete reference/CUDA block differential pass. Real-model
validation remains pending because no DOTS1 GGUF is available locally.

MiniMax-M2 now loads required untied output, optional fused or separate QKV,
full-projection Q/K RMSNorm, partial NeoX RoPE, and packed SwiGLU experts. Its
required F32 selection correction bias works with metadata-selected softmax or
sigmoid routing, normalized selected weights, and routed scaling. Metadata,
strict catalog, graph semantics, and a complete reference/CUDA block
differential pass. Real-model validation remains pending because no MiniMax-M2
GGUF is available locally.

Deci now preserves scalar-or-array per-layer query-head, KV-head, and
feed-forward schedules. Full-attention layers support optional fused or
separate QKV, normal or LongRoPE factors, optional projection/MLP biases, and
ordinary KV caching. Projection-only linear-attention and attention-free
layers execute their pinned residual ordering, while zero-width dummy layers
remain no-ops. Non-KV layers use compact serializable sentinel caches so cached
decode and context shifting retain a uniform layer table. Metadata, strict
mixed-layer catalog, all sparse graph modes, cache editing, and a four-mode
reference/CUDA differential pass. Real-model validation remains pending
because no Deci GGUF is available locally.

BailingMoE2 adds fused QKV projection, per-head Q/K RMSNorm, NeoX RoPE, leading
dense SwiGLU blocks, optional selection correction bias, metadata-selected
softmax or sigmoid routing, configurable shared-expert width, and always-on
shared experts. Preserved NextN/MTP layers are removed from the executable block
count. Metadata, strict dense/MoE catalog, both routing graphs, and a complete
sigmoid-bias reference/CUDA block differential pass. Real-model validation is
pending because no BailingMoE2 GGUF is available locally.

Dream now has an explicit non-causal, no-cache full-sequence execution path
for bounded-host and preloaded-device weights. It returns either all hidden
states or vocabulary logits for every position, and rejects KV-cache and
autoregressive-generation entry points. The iterative mask-transfer sampler
and diffusion CLI remain separate work.

Laguna now preserves scalar-or-array per-layer query/KV head metadata and its
full/SWA cadence. Full layers use NeoX YaRN while sliding layers use their
separate plain-RoPE dimensions and frequency base. Its attention output is
softplus-gated in either per-head or per-element layout; leading blocks use a
dense SwiGLU FFN and later blocks combine sigmoid top-k routed experts,
selection correction bias, optional selected-weight normalization, and one
always-on shared expert. Synthetic catalog/graph tests and a complete
reference/CUDA block differential pass. No Laguna GGUF exists in the local
fixture inventory, so real-model logits and throughput remain pending-fixture.

AFMoE now reuses the same sigmoid/correction-bias routed expert primitive and
supports its optional leading dense blocks and zero-or-more shared experts.
Its distinct graph behavior is preserved: embeddings receive MuP square-root
scaling, attention and FFN branches have both pre- and post-RMSNorm, the
attention output uses a per-element sigmoid gate before projection, and the
four-layer cadence combines periodic no-RoPE layers with interleaved sliding
attention. Synthetic metadata/catalog tests and a complete reference/CUDA
block differential pass. No AFMoE GGUF is present locally, so real-model
logits and throughput remain pending-fixture.

Qwen2-MoE now executes unnormalized softmax top-k routed experts in parallel
with its SwiGLU shared expert. The shared branch's learned scalar-per-token
sigmoid gate is represented explicitly in the graph, including its rank-one
GGUF tensor. Metadata and complete graph tests pass; real-model validation is
pending because the local fixture inventory has no Qwen2-MoE GGUF.

OLMoE now loads its packed SwiGLU expert bank, applies learned RMSNorm across
the complete Q/K projections before head reshaping, and executes unnormalized
softmax top-k routing. Metadata, strict catalog, graph-ordering, and complete
reference/CUDA block differential tests pass. Real-model validation remains
pending because no OLMoE GGUF is available locally.

PhiMoE now loads affine RMSNorm and projection/output biases, selects its
LongRoPE factor set from configured versus original context, pre-scales the
rotated query, and executes normalized softmax top-k packed SwiGLU experts.
Metadata, strict catalog, graph-semantics, and complete reference/CUDA block
differential tests pass. Real-model validation remains pending because no
PhiMoE GGUF is available locally.

EXAONE-MoE now preserves scalar-period or explicit per-layer sliding-attention
metadata, applies per-head Q/K RMSNorm and RoPE only on local layers, supports
leading dense blocks, and combines metadata-selected softmax/sigmoid routed
experts with optional selection correction bias and an always-on shared
expert. Preserved NextN/MTP layers are removed from the executable block count.
Metadata, strict catalog, graph, and complete reference/CUDA block differential
tests pass; real-model validation is pending because no EXAONE-MoE GGUF is
available locally.

RND1 now reuses the normalized softmax top-k MoE executor inside the explicit
full-sequence non-causal path. Its Q/K-normalized NeoX-RoPE attention, expert
tensors, hidden-state output, and all-position vocabulary logits run through
bounded-host and F32-preload CUDA execution; cache and autoregressive entry
points reject it. An iterative diffusion sampler/CLI remains separate work.

LFM2 now consumes its per-layer KV-head schedule, uses Q/K-normalized attention
on transformer layers, and runs gated channel-wise short convolution on
recurrent layers. Its fixed convolution window participates in host cache
serialization and prefix-edit operations; bounded-host and F32-preload CUDA
execution are covered. Native quantized convolution kernels and centered
non-causal LFM2 convolution remain deferred.

LFM2-MoE composes that hybrid cache with optional leading dense SwiGLU blocks
and later normalized softmax or sigmoid top-k routed experts. Selection
correction bias is required and remains F32 while expert banks use the native
quantized device path. Metadata, mixed dense/MoE and recurrent/attention
catalogs, graph semantics, and a complete recurrent convolution-plus-MoE
reference/CUDA differential pass. Real-model validation remains pending because
no LFM2-MoE GGUF is available locally.

PLM now executes its low-rank KV-A/KV-B decomposition, weighted compressed-KV
normalization, shared positional-key expansion, partial normal RoPE, and squared-ReLU
FFN through bounded-host and F32-preload CUDA paths. Its decompressed K/V cache
is serializable and prefix-editable; native quantized MLA fusion remains
performance work.

MiniCPM3 now reuses the MLA decomposition with its query LoRA A/RMSNorm/B
chain, partial NeoX RoPE with optional short/long factors, scaled attention and
FFN residual branches, and dense SwiGLU. Embedding and inverse-logit scaling
reuse the MiniCPM path; real-model validation remains pending a local fixture.

Hunyuan-Dense now executes optional fused or separate Q/K/V projections,
XDRoPE-adjusted normal RoPE, post-RoPE per-head Q/K RMSNorm, and dense SwiGLU
through bounded-host and F32-preload CUDA paths. Text-only zeroed multidimensional
RoPE sections are accepted; nonzero multimodal sections remain explicit errors.

Hunyuan-VL now executes the corresponding text decoder with optional fused or
separate Q/K/V, XDRoPE-adjusted normal or four-axis MRoPE, post-RoPE per-head
Q/K RMSNorm, dense SwiGLU, projected embedding overrides, and serializable KV
caching. Text tokens repeat their coordinate across all axes. Vision encoding,
projection, and image-grid coordinate construction remain external. Metadata,
strict catalog, graph ordering, and a reference/CUDA block differential pass;
real-model validation is pending a local fixture.

CogVLM now executes its token-input decoder with mandatory contiguous fused
QKV, normal RoPE with optional per-pair factors, RMSNorm, SwiGLU, optional tied
output, and serializable KV caching. The strict GGUF catalog also validates the
complete parallel visual attention/FFN expert bank without loading it into the
token graph. Embedding overrides are rejected so visual input cannot silently
use text weights. Visual-embedding execution and image encoding remain external;
metadata, catalog, graph topology, and reference/CUDA block differentials pass.

ERNIE 4.5 dense and MoE decoders now execute normal-RoPE attention followed by
sequential dense or metadata-scheduled expert residuals. The MoE path supports
optional expert gates, selection correction bias, normalized softmax top-k
routing, routed-weight scaling, and an optional shared SwiGLU expert. The
catalog preserves the pinned graph's ignored attention-output-bias behavior;
metadata, mixed-layer catalogs, graph semantics, and reference/CUDA block
differentials pass. Real-model validation remains pending a local fixture.

DeepSeek2-OCR now executes its text decoder with full split Q/K/V projections,
NeoX RoPE, leading dense SwiGLU blocks, and later fused or separate routed
experts. Softmax or sigmoid routing supports optional F32 selection bias,
normalization, scaling, and the required shared SwiGLU expert. Metadata, strict
mixed-layer catalog, graph semantics, and a reference/CUDA block differential
pass. Image encoding, projection, and soft-embedding construction remain the
external multimodal boundary; real-model validation is pending a local fixture.

PaddleOCR now reuses the ERNIE 4.5 dense catalog with its pinned split-half
four-axis MRoPE graph and optional attention-output bias. Text decoding repeats
the token coordinate across all axes and passes reference/CUDA block
differentials. The vision projector and image-grid coordinate construction
remain explicit multimodal blockers; real-model validation is pending fixtures.

Qwen2-VL now executes its dense text decoder with fused or separate Q/K/V,
optional projection/output biases, scaled four-axis split-half MRoPE, SwiGLU,
and serializable KV caching. Projected visual token embeddings can enter through
the embedding-override API; text tokens use repeated coordinates across all
axes. Image encoding and image-grid coordinate construction remain external
multimodal boundaries. Metadata, strict catalog, graph semantics, and a complete
reference/CUDA block differential pass; real-model validation is pending a
local text-model fixture.

Qwen3-VL now adds mandatory per-head Q/K RMSNorm before scaled four-axis MRoPE,
plus its optional deepstack-layer metadata, over the same dense text-decoder and
KV-cache paths. Ordinary token input exactly matches upstream's zero-filled
deepstack stream. Projected visual deepstack injection, image-grid coordinate
construction, vision encoding, and the optional classification/rerank head
remain external boundaries. Metadata, strict catalog, graph topology, and a
complete reference/CUDA block differential pass; real-model validation is
pending a local fixture.

Qwen3-VL-MoE now composes the Qwen3-VL text attention path with packed routed
SwiGLU experts. It supports fused or separate Q/K/V, per-head Q/K RMSNorm,
scaled four-axis MRoPE, normalized softmax top-k routing, routed-weight scaling,
native-quantized experts, zero-filled token deepstack, and serializable KV
caching. Projected visual deepstack injection, vision encoding, image-grid
coordinates, and the optional classification/rerank head remain external.
Metadata, strict catalog, graph topology, and reference/CUDA block differential
tests pass; real-model validation is pending a local fixture.

Chameleon now loads per-head affine Q/K LayerNorm weights and optional biases,
honors standard or sandwich RMS-normalization topology, uses normal RoPE, and
suppresses image-code logits 4 through 8195 for text output. Causal decoder
entry points accept caller-provided embedding overrides before learned
positions/scaling/embedding normalization and return an ordinary continuable
KV cache. This is a complete projected-soft-token decoder boundary. Image
preprocessing, a Chameleon vision encoder/projector, and image-token generation
remain deferred because neither the pinned source nor the local model inventory
provides a validated Chameleon vision GGUF/oracle fixture; external encoders can
already enter through the override API.

Talkie now executes its normalized token-embedding skip across every decoder
layer, unweighted RMSNorm before attention and FFN, post-RoPE learned query
gain, unweighted key RMSNorm, causal NeoX attention, parallel SwiGLU, per-layer
scalar skip injection, final unweighted RMSNorm, untied output projection, and
direct logit scaling. Bounded-host, F32-preload, native-quantized, and retained
device-cache paths share the graph. Metadata, strict catalog, cache topology,
and complete reference/CUDA block differentials pass; real-model validation is
pending because no Talkie GGUF is present locally.

## Working rules

- A blocked task is recorded here and deferred while independent work continues.
- Unsupported model/type combinations fail explicitly.
- Upstream llama.cpp is an oracle and test dependency, not a shipped runtime
  dependency.
- Correctness gates precede performance work.
