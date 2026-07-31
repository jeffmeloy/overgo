# llamacpp2go

`llamacpp2go` is a no-cgo Go reimplementation of the llama.cpp CUDA runtime.
Its current executable model subset is dense Qwen3, text-only Qwen3.5 hybrid
gated-delta-net models, Gemma 3 decoders, T5/UMT5 encoders, and bias-free
Llama-family decoders, including converted Llama 3 per-pair RoPE factors.

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
go run ./cmd/perplexity -native-quant <supported-model.gguf> "evaluation text"
go run ./cmd/embedding -model <t5-encoder.gguf> -prompt "Hello world!"
go run ./cmd/benchmark -native-quant -tokens 32 -runs 5 <supported-model.gguf> "Hello"
go run ./cmd/server -native-quant -listen 127.0.0.1:8080 <supported-model.gguf>
```

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
IQ2_S/IQ3_XXS/IQ3_S/IQ4_NL/IQ4_XS, and MXFP4/NVFP4 encoders byte-match the
pinned llama.cpp reference routines; F32, F16, and BF16 outputs are also
supported. Q8_1 and Q8_K are internal dot-product layouts, while the other
listed packed model layouts are available as CLI destinations. The tool
updates GGUF quantization metadata, accepts split input, never overwrites an
existing output, and offers `-all` for compatible one-dimensional tensors.
IQ1_S/IQ1_M and IQ2_XXS/IQ2_XS encoding require an importance matrix in the
pinned implementation and are not exposed by the values-only quantizer API.

Generation modes are:

- default: bounded host weight loading with CUDA graph execution;
- `-preload`: dequantize the complete model to persistent F32 CUDA weights;
- `-native-quant`: retain supported Q1_0/Q2_0, Q4_0/Q4_1, Q5_0/Q5_1,
  Q8_0/Q8_1/Q8_K,
  Q2_K through Q6_K, TQ1_0/TQ2_0, IQ1_S/IQ1_M, IQ2_XXS/IQ2_XS/IQ2_S,
  IQ3_XXS/IQ3_S, IQ4_NL/IQ4_XS, or MXFP4/NVFP4 matrices and embeddings in CUDA memory and
  execute them with native quantized kernels. This is the preferred mode for
  those models. `-native-q8` remains an alias.

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
`POST /v1/embeddings`. Native llama.cpp-compatible `POST /embedding` and
`/embeddings` return the pinned non-OpenAI array with one nested normalized
vector per pooled input. Native and OpenAI embedding inputs accept strings,
exact token sequences, mixed token/string sequences, and heterogeneous
batches; exact IDs bypass text round trips. Native embeddings support mean,
last-token, and unpooled per-token vectors plus `embd_normalize` modes `-1`,
`0`, and general p-norms. OpenAI embeddings support float arrays and
little-endian float32 `base64`. Native llama.cpp-compatible
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
remain independent, while hybrid recurrent models reuse only exact
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
`response.completed` without an OpenAI `[DONE]` marker. `/responses/input_tokens` and
`/v1/responses/input_tokens` expose the corresponding no-generation count.
Continuation IDs, tools, and multimodal/file inputs are rejected explicitly.
Text-only Anthropic-compatible `/v1/messages` supports buffered and named-SSE
streaming replies with Anthropic content blocks, stop fields, and usage.
`/v1/messages/count_tokens` accepts the same string or multipart text
system/message forms. Anthropic tools, thinking, and image blocks are rejected
until their template/runtime semantics are available.
Authenticated `GET /lora-adapters` truthfully reports an empty loaded-adapter
list. `POST /lora-adapters` accepts the empty disable-all list and rejects
non-empty activation because LoRA tensor execution is not implemented.
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
`/completion`, `/completions`, `/embedding`, `/embeddings`, `/tokenize`,
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
fused delta-net state, serialization, and resumed generation.
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

`cmd/release` builds the sixteen user-facing Windows-amd64 executables twice
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
