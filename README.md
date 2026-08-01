# llamacpp2go

`llamacpp2go` is a no-cgo Go reimplementation of the llama.cpp CUDA runtime.
Its current executable model subset is dense Qwen 1/2/3, Mixtral, BailingMoE/BailingMoE2, DeepSeek v1/2/3.2/4, DeepSeek2-OCR, GLM-DSA, and Mistral 4 text decoders, Qwen2-MoE/Qwen3-MoE/Qwen3-VL-MoE
through the bounded-host, F32-preload, and native-quantized expert paths, AFMoE, Arctic, text-only Qwen3-Next/Qwen3.5/Qwen3.5-MoE hybrid
gated-delta-net models, Apertus, Arcee, Baichuan 7B/13B, BitNet, Bloom, ChatGLM, CogVLM text/projected-visual decoding, CodeShell,
dense Cohere2, Cohere2-MoE decoder trunks, Command R, DBRX, Deci, DOTS1, Dream and LLaDA/LLaDA-MoE non-causal diffusion generation, Falcon/Falcon-H1, Gemma 1/2/3/4 and Gemma Embedding,
ERNIE 4.5/ERNIE 4.5-MoE, BERT/EuroBERT/JinaBERT v2/v3/Llama Embed/ModernBERT/NeoBERT/NomicBERT/NomicBERT-MoE encoders, GLM4/GLM4-MoE multimodal-coordinate decoding, GPT-2/GPT-NeoX, Granite/GraniteMoE with projected vision deepstack injection, GroveMoE, Grok, Hunyuan-Dense/Hunyuan-MoE and Hunyuan-VL text-coordinate decoding, HY-V3 decoder trunks, Chameleon decoders with projected soft-token input,
InternLM2, EXAONE/EXAONE 4/EXAONE-MoE, XVERSE, Jais/Jais2, Jamba, Granite Hybrid, Maincoder, Mamba v1/v2, RWKV6/RWKV6-Qwen2/RWKV7/ARWKV7, Mellum, MiMo2, MiniCPM/MiniCPM3, MPT,
Mistral 3 dense/MoE, Laguna hybrid-attention MoE, hybrid LFM2/LFM2-MoE, MiniMax-M2, SmallThinker, Nemotron, OLMo/OLMo2/OLMoE, OpenELM,
Orion, PaddleOCR and Qwen2-VL/Qwen3-VL text-coordinate decoders, Pangu Embedded, Phi-2/Phi-3/PhiMoE, PLaMo/PLaMo 2/PLaMo 3/PLM MLA, dense Refact, Talkie,
RND1 non-causal MoE diffusion generation, Seed-OSS, StableLM, StarCoder/StarCoder2 and SmolLM3 decoders, T5 encoder-decoder models and UMT5
encoders, and dense or Mixtral Llama-family decoders, including projection biases and converted Llama 3
per-pair RoPE factors plus metadata-driven linear RoPE scaling. Optional output
projection biases and Gemma attention/final-logit softcapping are honored in
every inference mode. MPT includes fused-QKV clamping, optional full-projection
affine Q/K LayerNorm, and optional AWQ post-GELU activation scaling.

The target is a Go-owned model and tensor runtime that calls NVIDIA's installed
Windows DLLs directly. CUDA kernels are reproducible binary assets and are the
only project-owned non-Go runtime components.

The implementation is currently experimental. See:

- `PORT_PLAN.md` for the architecture and phased plan;
- `IMPLEMENTATION_STATUS.md` for completed work, active work, and blockers;
- `compatibility.yaml` for the machine-readable compatibility target.
- `SBOM.cdx.json` and `LICENSES.md` for dependency/kernel provenance and the
  explicit license inventory.

## Development

```powershell
go test ./...
go run ./cmd/cuda-info
.\scripts\verify.ps1 -CUDA
.\scripts\fuzz-smoke.ps1 -Duration 5s
go run ./cmd/kernel-manifest
go run ./cmd/sbom -check
go run ./cmd/release -out dist -verify-reproducible
```

Regenerate the pinned llama.cpp IQ codebook tables for both Go and CUDA with:

```powershell
go run ./cmd/gen-iq-tables `
  -source C:\path\to\llama.cpp\ggml\src\ggml-common.h `
  -out internal\quant\iq_tables_generated.go `
  -cuda-out kernels\cuda\iq_tables_generated.cuh `
  -commit 42fc243060709331ff9b158a9ed2cbe37219ae83
```

`cmd/benchmark` reports per-run custom-kernel launches, stream
synchronizations, host/device copy counts and bytes, plus launches and barriers
per output token. These counters are snapshots from the Runner-owned CUDA
driver instance; cuBLAS launches are not misreported as custom kernels.

Inspect or tokenize a GGUF model with:

```powershell
go run ./cmd/inspect-gguf -metadata -tensors <model.gguf>
go run ./cmd/gguf-hash -all -uuid <model.gguf>
go run ./cmd/gguf-merge -out merged.gguf <first-split-or-single.gguf>
go run ./cmd/gguf-quantize <input.gguf> <output.gguf> q4_0
go run ./cmd/gguf-split -out-prefix model-part -max-tensors 128 -max-size 4G <model.gguf>
go run ./cmd/json-schema-grammar <schema.json>
go run ./cmd/tokenize <model.gguf> "Hello, world!"
go run ./cmd/block-check -tokens 4 <model.gguf>
go run ./cmd/generate -n 1 <model.gguf> "Hello"
go run ./cmd/generate -native-quant -n 16 <supported-model.gguf> "Hello"
go run ./cmd/generate -native-quant -context-shift -n 8192 <supported-model.gguf> "Hello"
go run ./cmd/diffusion -native-quant -length 512 -steps 128 -eps 0.001 <dream.gguf> "Hello"
go run ./cmd/diffusion -native-quant -length 512 -steps 128 -block-length 32 <llada.gguf> "Hello"
go run ./cmd/perplexity -native-quant <supported-model.gguf> "evaluation text"
go run ./cmd/embedding -model <t5-encoder.gguf> -prompt "Hello world!"
go run ./cmd/rerank -model <qwen3-reranker.gguf> -query "search terms" -document "candidate text"
go run ./cmd/benchmark -native-quant -tokens 32 -runs 5 <supported-model.gguf> "Hello"
go run ./cmd/server -native-quant -listen 127.0.0.1:8080 <supported-model.gguf>
```

`cmd/diffusion` implements the pinned iterative mask-transfer loop for Dream,
LLaDA, LLaDA-MoE, and RND1. Select exactly one schedule with `-eps` or
`-block-length`; confidence, entropy, margin, random, and origin ranking,
classifier-free guidance, shifted logits, Gumbel transformation, and visual
step progress are available.

Opening a first shard named `<prefix>-00001-of-XXXXX.gguf` automatically loads
the complete local split set. The reader validates shard indices/counts,
global tensor count, duplicate names, and each shard's independent tensor
bounds; `inspect-gguf` reports both `splitCount` and each tensor's shard.
`gguf-merge` streams a logical split model into one canonical GGUF without
buffering tensor payloads, strips split bookkeeping metadata, and refuses to
overwrite an existing output.
`gguf-split` performs the inverse operation with upstream shard names and
metadata, tensor-count and aligned-payload limits, an optional metadata-only
first shard, and cleanup of only the new files it created if a later shard
fails.
`gguf-hash` streams logical tensor payloads across single or split files and
matches pinned llama.cpp XXH64, SHA-1, SHA-256, UUIDv5, per-tensor/model output,
and manifest-check exit semantics.
`gguf-quantize` converts matrix tensors in bounded blocks through F32 and
preserves one-dimensional tensors by default. Its Q1_0, Q2_0, Q2_K-Q6_K,
Q8_K, Q4_0/Q4_1, Q5_0/Q5_1, Q8_0/Q8_1, TQ1_0/TQ2_0,
IQ1_S/IQ1_M, IQ2_XXS/IQ2_XS/IQ2_S, IQ3_XXS/IQ3_S, IQ4_NL/IQ4_XS, and
MXFP4/NVFP4 encoders byte-match the pinned llama.cpp reference routines; F32,
F16, and BF16 outputs are also supported. Q8_1 and Q8_K are internal
dot-product layouts, while the other listed packed model layouts are available
as CLI destinations. IQ1_S/IQ1_M and IQ2_XXS/IQ2_XS use `-imatrix` with pinned
GGUF or legacy importance matrices, including per-expert normalization and
llama.cpp-compatible provenance metadata. The tool updates GGUF quantization
metadata, accepts split input, never overwrites an existing output, and offers
`-all` for compatible one-dimensional tensors.

Generation modes are:

- default: bounded host weight loading with CUDA graph execution;
- `-preload`: dequantize the complete model to persistent F32 CUDA weights;
- `-native-quant`: retain supported Q1_0/Q2_0, Q4_0/Q4_1, Q5_0/Q5_1,
  Q8_0/Q8_1/Q8_K,
  Q2_K through Q6_K, TQ1_0/TQ2_0, IQ1_S/IQ1_M, IQ2_XXS/IQ2_XS/IQ2_S,
  IQ3_XXS/IQ3_S, IQ4_NL/IQ4_XS, or MXFP4/NVFP4 matrices and embeddings in CUDA memory and
  execute them with native quantized kernels. This is the preferred mode for
  those models. Operations whose pinned CUDA kernels require F32 coefficients,
  including LFM2 short convolution, promote only those source tensors during
  preload. `-native-q8` remains an alias.

The Go runner's `ForwardCachedWithMultimodalInputs` accepts projected visual
token embeddings plus distinct temporal, height, width, and extra MRoPE
coordinates. It returns the ordinary continuable cache; image encoding,
projection, and grid construction remain caller-owned boundaries.
CogVLM projected-visual calls use `ForwardCachedWithEmbeddingOverrides` and
must replace every token in the chunk; the runtime then selects its visual
attention and FFN bank for the complete batch.

Sampling supports temperature, top-k, top-p, min-p, locally typical filtering,
top-n-sigma, probabilistic XTC, shared `min_keep` floors, repetition windows,
presence/frequency penalties, token-history DRY, and adaptive Mirostat v1/v2.
Dynamic temperature uses the pinned entropy-normalized range/exponent formula
through `-dynatemp-range` and `-dynatemp-exp`, or the matching HTTP fields.
DRY restart strings use llama.cpp-style
overlapping-token expansion so breakers embedded in larger tokens are honored.
The non-Mirostat stages can be reordered or repeated with llama.cpp-style
`-samplers "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature"`;
use `none` for an empty transform chain. HTTP completion requests accept the
same names in an ordered `samplers` array. Mirostat, as upstream, replaces
that chain. Adding `adaptive_p` makes adaptive-p the terminal selector
regardless of where it appears in the list; configure it with
`-adaptive-target` and `-adaptive-decay` or matching HTTP fields.
Adding `infill` applies llama.cpp's sorted-softmax EOG weighting,
token-piece prefix merging, and two-stage probability thresholds using the
loaded GGUF vocabulary. The CLI and HTTP sampler paths populate that
vocabulary automatically. FIM prefix/suffix/middle prompt construction is
still caller-managed.
Repeated `-logit-bias TOKEN=BIAS` flags adjust or ban (`-inf`) individual
tokens, while `-ignore-eos` bans every EOG token recognized from GGUF
metadata. HTTP `logit_bias` accepts llama.cpp-compatible pair arrays or
objects, including textual keys and `false` bans, plus `ignore_eos`.
Tokenization covers Llama SentencePiece BPE, GPT-2/Qwen byte BPE, T5 unigram,
and BERT WordPiece. Six pinned upstream vocabulary corpora pass all 280 exact
token-ID cases; WordPiece includes NFD accent stripping, lowercasing,
punctuation/Chinese segmentation, greedy matching, and CLS/SEP handling.
Text-only chat formatting executes `tokenizer.chat_template` with a pure-Go,
memory-only Jinja loader. It supplies llama.cpp-compatible message,
BOS/EOS, generation-prompt, thinking, tool-schema, assistant tool-call, and
tool-result variables while bounding template and rendered output sizes.
Default and named `tool_use` metadata templates are selected as appropriate.
Local Qwen3, Qwen3.5, Gemma 3, and Bonsai text and tool-history prompts match
pinned `/apply-template` output byte-for-byte; token-boundary native formatters
remain the fallback when GGUF metadata has no template.
Buffered and streaming OpenAI chat support `auto`, `required`, `none`, and
named function choices. Auto calls use delimiter-triggered lazy GBNF;
required/named calls use schema-derived forced GBNF. Qwen JSON-in-XML and
Qwen3.5/Bonsai Hermes XML outputs are parsed into validated `tool_calls` with
generated IDs and `finish_reason:"tool_calls"`. Tool-enabled streams buffer
template syntax until the complete output is validated, then emit one
structured delta per call; token-incremental argument deltas remain pending.
Exact literal completion alternatives can be enforced with repeated
`-grammar-choice` flags; the equivalent experimental HTTP request field is
`"grammar_choices":[" first"," second"]`. Grammar state is included in
resumable sampler/session state.
General character GBNF is available through `-grammar`, `-grammar-file`, and
`-grammar-root`, or HTTP fields `"grammar"` and `"grammar_root"`. Supported
syntax includes literals and escapes, Unicode ranges and negated classes,
wildcards, named rules, groups, alternation, right recursion, comments, and
`*`, `+`, `?`, or `{m,n}` repetition. Explicit numeric `<[id]>`, named
`<|special|>`, and inverse `!<...>` token terminals are also supported.
Matching follows decoded token bytes,
including UTF-8 sequences split across byte-fallback tokens, and permits EOG
only after the root rule accepts. Lazy activation is available with
`-grammar-lazy` plus repeated `-grammar-trigger-pattern` or
`-grammar-trigger-token` flags. Regex capture groups select where buffered
token-piece replay begins, including matches that start partway through a
token. HTTP uses `grammar_lazy`, `grammar_trigger_patterns`, and
`grammar_trigger_tokens`. Trigger regexes currently use Go RE2 syntax, so
ECMAScript lookaround and backreferences are rejected explicitly.
`cmd/json-schema-grammar` converts an ordered JSON Schema from a file or stdin
to deterministic GBNF. Native `/completion` and `/completions` requests accept
the same schema in `json_schema`; it is compiled with root `root` before
generation. OpenAI `/v1/completions` accepts the same top-level field.
`/v1/chat/completions` and its unprefixed alias additionally accept
`response_format.type` values `text`, `json_object`, and `json_schema`; the
last form reads the schema from `response_format.json_schema.schema`, while
`json_object` optionally reads `response_format.schema` and otherwise
constrains output to a JSON object. Explicit grammar and JSON Schema controls
are mutually exclusive on every route.
The converter matches all 70 shared cases plus the two C++-only regular
expression cases from the pinned upstream converter suite.
Run `go run ./cmd/generate -help` for the complete flag set.
`-context-shift` enables a rolling context window: once active attention KV
reaches the model limit, the oldest entries are discarded while absolute RoPE
positions and Qwen3.5 recurrent state continue forward. It is opt-in for both
the generation CLI and server. `generate -keep N` and native completion
`n_keep` preserve the requested initial prompt prefix while compacting the
middle of a full cache; `-1` keeps as much of the prompt as the context safety
margin permits. `generate -discard N` and native `n_discard` select the
discard size; zero follows llama.cpp's half-of-discardable-window default.
Ordinary prefix removal uses zero-copy CUDA views, while prefix-preserving
middle compaction creates bounded device-owned copies of the retained ranges.
In Go, set `GenerateOptions.ContextShift`, `KeepTokens`, and `DiscardTokens`.

The perplexity CLI scores every next token by default. Set `-ctx N` to use
llama.cpp's disjoint-window convention (full windows, latter half scored) for
differential comparisons, and `-token-scores` to include per-token negative
log-likelihoods.

The experimental HTTP server provides public `GET /health`, `/healthz`, and
`/v1/health`, Prometheus-format
`GET /metrics`, public native `GET /models`, `GET /v1/models`,
`POST /v1/completions`, `POST /v1/chat/completions`, and
`POST /v1/embeddings`. Reranker models expose `/rerank`, `/reranking`,
`/v1/rerank`, and `/v1/reranking` with Jina and TEI response formats,
descending stable sorting, `top_n`, usage totals, and optional TEI text return.
Native llama.cpp-compatible `POST /embedding` and
`/embeddings` return the pinned non-OpenAI array with one nested normalized
vector per pooled input. Native and OpenAI embedding inputs accept strings,
exact token sequences, mixed token/string sequences, and heterogeneous
batches; exact IDs bypass text round trips. Native embeddings support mean,
last-token, and unpooled per-token vectors plus `embd_normalize` modes `-1`,
`0`, and general p-norms. OpenAI embeddings support float arrays and
little-endian float32 `base64`. Gemma Embedding applies optional converted
sentence-transformer dense-2/dense-3 projections after pooling and before
normalization.
Qwen3 and dense Qwen3-VL rerank heads execute through `Runner.RankPair`,
`Rank`, `RankTokens`, or `RankTokensWithProjectedInputs`. The rank path selects
the last normalized token, applies `cls.output.weight`, returns labeled softmax
scores, and honors the named rerank template or configured EOS/SEP separator.
`cmd/rerank` exposes the text-pair path for bounded-host, F32-preloaded, and
native-quantized execution.
OpenAI `/v1/completions` likewise accepts strings, exact or mixed token
sequences, flat string batches, and nested heterogeneous prompt batches of up
to 64. Choices flatten in prompt-major order and exact IDs enter the Runner
without re-tokenization in buffered and SSE modes.
`POST /completion` and
`/completions` accept string prompts, exact token-ID sequences, or mixed
token/string sequences plus `n_predict`, `n_cmpl`, stop strings, raw token
returns, and the implemented sampler/grammar fields. Exact IDs enter the
Runner directly without a lossy text round trip; a leading string segment
applies the vocabulary's configured BOS policy. Flat string arrays and arrays
containing nested token sequences form batches of up to 64 prompts. Results
are flattened in prompt-major order, with `n_cmpl` choices per prompt and
stable global indices. Buffered replies
carry native stop/timing/generation metadata. Native SSE sends token chunks
followed by the pinned empty-content `stop:true` metadata event without an
OpenAI `[DONE]` marker. While a native stream is silent,
`sse_ping_interval` emits the pinned `:\n\n` SSE comment heartbeat; `-1`
disables it and the default is 30 seconds. `response_fields` can project up
to 64 fields from each final response; slash paths such as
`generation_settings/n_predict`
retain their full path as the output key, and absent paths are omitted like
the pinned server. With `cache_prompt:true`, the Runner retains evaluated
prompt states in a bounded LRU. The server defaults to one entry and exposes
`-prompt-cache-entries`; Go callers use `OpenOptions.PromptCacheEntries`.
Single-entry pure-attention caches reuse the longest common token prefix,
including suffix rollback and divergent continuations. Multiple device entries
remain independent, while recurrent models reuse only exact
cached-prefix extensions because their summarized state cannot be reversed
safely.
`POST /infill` accepts required mixed token/string `input_prefix` and
`input_suffix` values, an optional special-token-aware string `prompt`, and
optional `{filename,text}` repository chunks. It reproduces the pinned 3:1
prefix/suffix truncation and repository-context layout, then uses the native
completion response/streaming path. Server `-batch-size` controls the logical
FIM prompt window and `-spm-infill` selects suffix/prefix/middle ordering.
The loaded vocabulary supplies current or deprecated FIM metadata, with the
pinned token-spelling fallbacks when metadata is absent.
`return_progress:true` emits pinned `prompt_progress` envelopes at prompt
evaluation start and completion, including total/cache/processed counts and
elapsed milliseconds. `timings_per_token:true` adds the current cache,
prompt-evaluation, and decode timing/rate object to every emitted token event.
Positive `t_max_predict_ms` gracefully ends generation on the first newline
token emitted after the prediction budget expires; `0` and `-1` disable it.
Positive `n_indent` ends code completion when a post-newline line has fewer
leading spaces/tabs than requested and trims the offending non-whitespace
suffix, matching the pinned native endpoint.
Positive `n_probs` returns the selected token plus the requested top-N
pre-sampling softmax log probabilities, token pieces, and exact byte arrays in
both buffered and per-token SSE responses. With
`post_sampling_probs:true`, the same envelopes use `prob`/`top_probs` from the
sampler's actual normalized candidate set after its configured filtering
chain; fewer than N entries are returned when filtering leaves fewer choices.
`n_cache_reuse` sets the minimum matching length. Responses report real
cached/evaluated counts and separate prompt/decode timing. Authenticated
`GET /slots` reports stable slot IDs, busy/task state, usable context length,
prompt processed/cache counts, generated text, and measured prompt/decode
timings. The effective sampler parameters, stop sequences, prompt-cache
controls, `n_keep`, `n_discard`, and context-shift setting are included in
`params`. Completed
snapshots remain visible while the slot is idle.
`fail_on_no_slot=1` returns 503 when capacity is exhausted. Native `id_slot`
requests acquire that exact slot.
`next_token` reports decoded/remaining counts; its pending-token flags remain
false because this runtime samples synchronously and does not retain a sampled
token between generation steps.
Authenticated `/chat/completions` aliases `/v1/chat/completions`.
`/chat/completions/input_tokens` and `/v1/chat/completions/input_tokens`
format and tokenize the request exactly like generation, returning the pinned
`response.input_tokens` envelope without running the model.
Text-only OpenAI content-part arrays are flattened through the same formatter.
Authenticated `/responses` and `/v1/responses` convert text inputs and message
arrays into chat prompts and return the pinned Responses object/output/usage
envelope. Streaming emits the named Responses lifecycle events through
`response.completed` without an OpenAI `[DONE]` marker. Flat function tools,
auto/none/required/named choice, replayable call/output history, constrained
generation, single/parallel call constraints, buffered function-call items,
and call-complete argument SSE events are supported. `/responses/input_tokens` and
`/v1/responses/input_tokens` expose the corresponding tool-aware
no-generation count. Continuation IDs, hosted/custom tools, reasoning items,
multimodal/file inputs, and token-incremental function-argument deltas remain
explicit exclusions.
Text-only Anthropic-compatible `/v1/messages` supports buffered and named-SSE
streaming replies with Anthropic text/tool-use content blocks, stop fields,
and usage. Tool definitions, auto/any/named choice, assistant `tool_use`,
user `tool_result`, schema-constrained generation, call-complete streaming
`input_json_delta`, single/parallel call constraints, and tool-aware token
counting share the GGUF Jinja formatter used by OpenAI chat.
`/v1/messages/count_tokens` accepts the same string or multipart text
system/message forms. Anthropic thinking and image blocks remain rejected
until their template/runtime semantics are available.
Repeatable `--lora <adapter.gguf>` loads pinned-format LoRA adapters for the
server, generator, diffusion, perplexity, embedding, and benchmark commands. Server
adapters start at scale 1 unless `--lora-init-without-apply` is set.
Authenticated `GET /lora-adapters` reports `{id,path,scale}` entries;
`POST /lora-adapters` atomically replaces global scales and disables omitted
adapters. Alpha/rank scaling applies graph-wide to dense projections, grouped
expert banks, token embeddings, and output projections across streamed,
F32-preloaded, and native-quantized model paths. Native completion `lora`
arrays override scales for one serialized generation and restore global state;
prompt caches are isolated by adapter content and scale. A single enabled
aLoRA activates at the last matching invocation-token sequence; prompt state
before that sequence is evaluated with scale zero. Diffusion accepts regular
adapters and rejects aLoRA because non-causal attention has no isolated
pre-invocation prefix. Fused-MoE execution forms
ephemeral adapted expert weights on device; this preserves request-local
scales but adds a full expert-bank merge pass per graph.
llama.cpp-compatible `POST /apply-template` returns
the selected native chat prompt without inference. Authenticated
`POST /tokenize` accepts text or a flat mixed token/string sequence, defaults
`parse_special` to true, and optionally returns `{id,piece}` objects. Invalid
UTF-8 pieces use llama.cpp's integer-byte-array representation. `/detokenize`
concatenates exact per-token pieces with special tokens rendered. Both routes
are bounded and strictly validate token IDs. Authenticated `GET /props`
returns llama.cpp-compatible global properties: normalized generation
defaults, slot/capability flags, model path/type, original GGUF chat template,
BOS/EOS text, and a bounded `model_metadata` summary of the loaded
architecture. `POST /props` is intentionally disabled because this server has
no mutable global-property mode. Text and chat completion streaming use OpenAI-style SSE
with `{"stream":true}`. Chat selects native ChatML, Gemma
`<start_of_turn>`, or Llama 3 header-token formatting from the loaded
vocabulary. Embeddings are
mean-pooled over final token hidden states and L2-normalized. Requests are size/token/batch
bounded. Completion and chat `stop` accept a string or string array; matching
halts the runner at the token boundary, while the HTTP output filter withholds
partial prefixes so stop markers never appear in buffered or streamed text.
Both completion endpoints accept `n` from 1 through 8. Choices receive fresh
sampler/grammar state and deterministic seed offsets; buffered and SSE
responses carry stable choice indices.

For preloaded dense models, embedding lookup, all decoder layers, final
normalization, last-token slicing, and output projection execute as one CUDA
submission per token. KV state uses explicitly owned device outputs between
steps; only the vocabulary logits cross to Go for sampling. Identical RoPE
attribute buffers are uploaded once per graph. Editable sessions,
and streamed-weight compatibility paths retain their host-cache behavior.
Prompt-cache-plus-context-shift requests preserve the immutable device prompt
with a range copy before the first destructive edit. Ordinary rolling context
shift uses zero-copy device-pointer views over the retained attention cache,
while prefix-preserving `n_keep` compaction copies only retained device ranges. Direct
Qwen3.5 generation retains both attention
KV and recurrent convolution/SSM state on-device; its initial zero state uses
stream-ordered device memsets rather than host uploads.
Cancellation propagates through execution, and overload returns HTTP 429.
Metrics expose request concurrency/counts, generation attempts and errors,
generated tokens, uptime, and readiness without an external package.
`-request-timeout <duration>` applies an end-to-end handler deadline; zero
keeps long generations unbounded. SIGINT/SIGTERM stop admission and allow a
30-second graceful HTTP drain before the model is closed.
Set `LLAMACPP2GO_API_KEY` or pass `-api-key-file <path>` to require a
constant-time checked bearer token on generation/embedding `/v1/*` routes,
`/apply-template`,
`/completion`, `/completions`, `/embedding`, `/embeddings`, `/rerank`,
`/reranking`, `/tokenize`,
`/detokenize`, `/slots`, `/lora-adapters`, `/chat/completions`, `/responses`, `/v1/messages`,
chat/Responses/Anthropic token-count routes, and
`/props` endpoint. Health, metrics, `/models`, and
`/v1/models` remain unauthenticated for orchestration and model discovery. The
model-discovery responses omit the local path but include both pinned native
`models` and richer `data` arrays with GGUF family,
vocabulary/context/embedding sizes, parameter count, tensor bytes, and
quantization level.
Prometheus metrics include optional Runner-owned CUDA current bytes, lifetime
peak bytes, live allocation count, custom launches, synchronization totals,
and host/device plus device/device transfer bytes when the generator exposes a device
memory snapshot; snapshot failures omit only those gauges.

Set `LLAMACPP2GO_LLAMA_CPP` to a pinned llama.cpp checkout to enable the
tokenizer oracle corpus during `go test ./...`.

Set `LLAMACPP2GO_QWEN3_MODEL` and/or `LLAMACPP2GO_QWEN3_Q6K_MODEL` to the
validated local Qwen3 files to run optional cache and native-quantized
end-to-end integration tests. Set `LLAMACPP2GO_QWEN35_MODEL` to a compatible
Qwen3.5 hybrid GGUF to validate gated full attention, recurrent convolution,
fused delta-net state, serialization, and resumed generation. Qwen3.5-MoE
catalog, graph, and CUDA differential tests use synthetic fixtures until a
compatible local GGUF is available. Qwen3-Next optimized and legacy recurrent
catalogs plus both hybrid block types are likewise validated synthetically.
Qwen3.5 files declaring the pinned single NextN layer expose
`NewQwen35MTPSession`/`AdvanceQwen35MTP`; the catalog, normalized token/hidden
fusion, dense gated-attention draft block, private/shared embedding and head
fallbacks, independent KV/hidden state, and CPU/CUDA execution are synthetic-fixture validated.
MTP-only sidecars use `NewQwen35MTPPairedSession` with an exact compatible
target vocabulary/shape profile. `DraftQwen35MTPGreedy` and
`VerifyQwen35MTPGreedy` build bounded probability-filtered proposals, verify
the prefix without committing a rejected suffix, and resynchronize target
hidden state at the accepted boundary.
`DraftQwen35MTPSampled` and `VerifyQwen35MTPSampled` add stochastic verification
with full post-filter draft distributions, probability-ratio acceptance,
positive-residual correction, and transactional draft/target sampler state.
`SaveQwen35MTPSession`/`LoadQwen35MTPSession` preserve the target trunk cache,
independent MTP KV, pending hidden row, and absolute positions in a bounded
payload fingerprint-bound to both draft and target models.
Set `LLAMACPP2GO_QWEN35_MTP_MODEL` to a bundled trunk-plus-MTP fixture to run
the optional two-step native-quantized session integration test.
Step3.5 files declaring multiple trained NextN heads expose
`NewStep35MTPSession`/`AdvanceStep35MTP`. Each head retains its full decoder
block, per-layer head/sliding/clamp schedule, optional private embedding and
output head, and independent prompt KV. Drafting rebuilds the growing draft
prefix under successive heads while carrying the target's pre-output-norm
hidden rows, matching the pinned chain semantics. `DraftStep35MTPGreedy` and
`VerifyStep35MTPGreedy` provide bounded confidence-filtered proposals and
target resynchronization; `DraftStep35MTPSampled` and
`VerifyStep35MTPSampled` add probability-ratio acceptance and transactional
sampler state. `SaveStep35MTPSession`/`LoadStep35MTPSession` preserve all head
caches, active draft rows, trunk cache, positions, and model binding. Set
`LLAMACPP2GO_STEP35_MTP_MODEL` to run the optional native-quantized multi-head
session and coordinator integration test.
HY-V3 exposes the parallel `NewHYV3MTPSession`/`AdvanceHYV3MTP`, greedy and
sampled draft/verify, and save/load APIs. It reuses the per-head growing-prefix
coordinator while retaining HY-V3's distinct post-final-norm target and draft
hidden states. Set `LLAMACPP2GO_HYV3_MTP_MODEL` to run its optional
native-quantized session and coordinator integration test.
Cohere2-MoE files declaring the pinned single NextN block expose
`NewCohere2MTPSession`/`AdvanceCohere2MTP`, with
`NewCohere2MTPPairedSession` for an MTP sidecar. The draft graph uses the
pinned full-attention RoPE block, parallel attention/expert residuals,
optional shared-expert averaging, LayerNorm-or-RMSNorm selection, post-norm
hidden state, and direct logit scaling. Greedy and sampled draft/verify plus
`SaveCohere2MTPSession`/`LoadCohere2MTPSession` provide the same transactional,
target-bound coordination guarantees as the Qwen3.5 single-block path. Set
`LLAMACPP2GO_COHERE2_MTP_MODEL` to run the optional native-quantized session,
state, and coordinator integration test.
GroveMoE metadata, grouped chunk-expert routing, catalog, graph, and CUDA
differentials use synthetic fixtures until a compatible local GGUF is available.
GLM4-MoE dense-leading and expert catalogs, text-coordinate MRoPE, norm ordering,
routing, shared experts, and CUDA differentials are likewise synthetic-fixture validated.
MiMo2 mixed dense/expert layers, per-layer KV heads and sliding selection, attention
sinks, value scaling, MTP trimming, and CUDA differentials are synthetic-fixture validated.
Gemma4 full/sliding head widths, shared KV, absent-V fallback, mixed dense/GELU-MoE
blocks, expert output scales, projected per-layer inputs, and raw BPE are synthetic-fixture validated.
Llama4 chunk-aligned attention, periodic temperature-scaled full attention,
post-RoPE Q/K normalization, interleaved sigmoid MoE/shared-expert blocks, and
GPT-4o pre-tokenization are synthetic-fixture and CUDA-differential validated.
GPT-OSS alternating sliding attention with sinks, residual post-attention norm,
selected-logit softmax routing, biased OpenAI SwiGLU experts, and MXFP4-native
expert storage are synthetic-fixture and CUDA-differential validated.
DeepSeek2 and Mistral 4 query LoRA, legacy and absorbed MLA, YaRN scaling,
temperature tuning, dense-prefix/MoE switching, correction bias, and shared
experts are synthetic-fixture and CUDA-differential validated.
GLM-DSA full/shared lightning indexers, FWHT key rotation, named indexer cache,
top-k reuse, sparse absorbed MLA, YaRN scaling, and sigmoid MoE/shared experts
are synthetic-fixture and CUDA-differential validated.
Mamba v1 convolution state, selective-scan state, optional FalconMamba dt/B/C
RMSNorm, D skip, and SiLU gate are synthetic-fixture and CUDA-differential validated.
Mamba2 grouped B/C state, scalar-per-head A/D, grouped output RMSNorm, and
expanded convolution state are likewise synthetic-fixture and CUDA-differential validated.
Falcon-H1 parallel GQA/Mamba2 mixing, NeoX RoPE, projection biases, and mixed
token/fixed cache state are synthetic-fixture and CUDA-differential validated.
T5 encoder-decoder execution includes bidirectional encoder attention, causal
decoder relative buckets, fixed cross-attention K/V, classic ReLU or gated
GELU FFNs, and model-bound resumable session serialization. The public path is
`GenerateT5` for complete source-to-text sampling, or `NewT5Session` followed
by chunked `DecodeT5`; `SaveT5Session` and `LoadT5Session` preserve both
encoder output and decoder cache. Active-range edits compact decoder self K/V
without changing fixed cross-attention state. The `generate` CLI selects this
source-to-text path automatically for T5 models. Generic `Generate` dispatches
to the same coordinator and retains its combined input/output token contract,
covering completion-compatible HTTP routes and usage accounting as well.
DFlash uses the paired-runner path: `PrimeDFlash` or `SyncDFlashPrefix`
extracts configured target-layer inputs, fuses them, and injects committed K/V;
`DraftDFlashBlock` evaluates the last-token plus MASK noise block with
non-causal cache-aware attention and the target model's embedding/output
tables. Lower-level fusion, injection, extraction, and explicit noise-block
methods remain public for schedulers that own verification and acceptance.
Eagle3 uses `NewEagle3Session` to extract and fuse exactly three configured
target-layer inputs while constructing the shifted draft cache. Each
`AdvanceEagle3` call pairs the next token with the pending feature, returns
target-vocabulary logits plus the next pre-norm feature, and supports optional
draft-owned embeddings/output weights and `d2t` vocabulary remapping.
Gemma4 Assistant uses `NewGemma4AssistantSession` to run the target Gemma4
prefix and retain its final two shared KV layers plus the last normalized target
hidden row. `AdvanceGemma4Assistant` reads those fixed target caches, reuses the
target-cache position for every draft token, and carries only the projected
target-width hidden row between steps.
WavTokenizer decoder execution maps semantic token IDs to audio-feature frames
through the strict upstream tensor catalog, six-stage PosNet, full non-causal
single-head attention, dense/depthwise same-padding convolutions, GroupNorm,
and ConvNeXt projection blocks. `DecodeWavTokenizer` is the explicit public
entry point; `Forward` returns the same feature matrix for this architecture.
RWKV6 dual token shifts, WKV6 state, affine normalization, time-first mixing,
per-head group normalization, and squared-ReLU channel mixing are
synthetic-fixture and CUDA-differential validated.
RWKV6-Qwen2 five-way low-rank token mixing, grouped-KV gated linear attention,
cached normalized token shift and WKV state, optional projection biases, and
periodic residual rescaling are synthetic-fixture and CUDA-differential validated.
RWKV7/ARWKV7 vector-decay recurrence, in-context learning-rate mixing,
cross-layer first-value residuals, key normalization/correction, optional gates
and time group normalization, and architecture-specific channel/FFN branches
are synthetic-fixture and CUDA-differential validated.
Jamba mixed no-RoPE attention/Mamba layers, learned dt/B/C norms, and optional
dense or routed-MoE FFNs are synthetic-fixture and CUDA-differential validated.
Granite Hybrid mixed optional-RoPE attention/Mamba2 layers, metadata scaling,
optional projection biases, and dense or routed/shared-expert FFNs are
synthetic-fixture and CUDA-differential validated.
PLaMo2 mixed fused-QKV attention/Mamba layers, learned B/C/dt normalization,
post-normalized mixer and fused SwiGLU branches, and hybrid recurrent/KV cache
are synthetic-fixture and CUDA-differential validated.
Set `LLAMACPP2GO_BONSAI_MODEL` to the local Bonsai 27B Q1_0 Qwen3.5 fixture
to validate real-model native one-bit weights against the pinned CPU oracle.
Set `LLAMACPP2GO_GEMMA3_MODEL` and `LLAMACPP2GO_UMT5_MODEL` to run the
optional real-model Gemma greedy-ID/perplexity and UMT5
tokenizer/catalog/encoder checks. The UMT5 differential oracle is llama-embedding with
`--attention non-causal`; its default attention selection for the validated
fixture is causal.

The Go inference API exposes `SaveCache`/`LoadCache` for raw attention KV and
hybrid recurrent state, `ShiftCache` for deep-copy prefix removal, and
`StartSession`/`ContinueSession` plus `SaveSession`/`LoadSession` for resumable
generation. Cache state v2 records active cache length separately from the
absolute next-token position and remains able to load v1 append-only files.
Session files
contain token history, model cache tensors, sampler RNG and grammar state, and
Mirostat adaptive state. They are versioned, little-endian, bounds-checked,
sampler-configuration checked, and fingerprint-bound to the loaded GGUF
model. Sampler state v4 stores variable-length GBNF replay history plus
adaptive-p EMA state and loads compatible v2/v3 states.

The initial supported host is Windows amd64 with an NVIDIA CUDA driver.

`cmd/benchmark` emits machine-readable JSON containing model/device identity,
load time, per-run prompt/output token counts, TTFT, post-first-token decode
rate, end-to-end rate, p50/min summaries, host heap snapshots, and current/peak
Runner-owned CUDA allocation bytes. CUDA allocation accounting is maintained
at the shared driver boundary, so it includes persistent weights and temporary
graph arenas but intentionally excludes other processes and driver overhead.

The fuzz smoke script exercises bounded hostile-input targets for GGUF,
tokenization, GBNF, sampler state, cache/session state, and server JSON. Go's
race detector is unavailable under the required `CGO_ENABLED=0` build; enabling
it would test a different runtime contract, so concurrency is covered with
deterministic contention tests until a no-cgo race instrumenter is available.

`cmd/release` builds the seventeen user-facing Windows-amd64 executables twice
with no cgo, source paths, VCS stamp, or Go build ID and rejects any byte
difference. It creates a stable stored ZIP with fixed timestamps, embedded
documentation/SBOM/license/kernel manifest, an internal `SHA256SUMS`, and an
external archive checksum. Kernel-manifest and SBOM freshness are release
gates. The tag/manual GitHub workflow uploads this unsigned artifact; signing
requires a user-controlled certificate and is not simulated.

Regenerate the embedded smoke-test PTX with:

```powershell
.\scripts\build-kernels.ps1
```
