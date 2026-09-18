# Colibri transfer review

Reviewed local Colibri `f028d26b422144ed4a69ad9aeaee2553ce0f9572`
(v1.11.0) against Overgo master `f2953499d7e2a5627f61e7a0081e09648e0c82c1`.
This is source analysis, not execution authority. `docs/plan.json` owns this
worktree's queue and workflow. No Colibri performance result is an Overgo result.

## Findings

Colibri's useful transfer is its treatment of inference as a complete memory
and execution pipeline. Its implementation is organized around model-specific
C engines; Overgo should transfer behavior through its existing recipe,
inference, sampling, tensor, runrecord and workbench owners. No C/Python runtime,
React build, private weight format or parallel scheduler is required.

| Source inspected in the local Colibri checkout | Transfer into Overgo | Acceptance boundary |
| --- | --- | --- |
| README.md; docs/benchmarks.md; c/tools/datapoint.py | Cold, repeated and rotating workloads; full environment and realized residency | Same artifacts, token digests, workload and quality protocol; paired end-to-end measurements |
| c/download_fp8.py (`download_file_curl`); Apache-2.0 LICENSE | Retain interrupted transfer progress in the existing Go hub client | Immutable source identity, prefix rehash, range/validator checks, exact declared size/digest, caller cancellation and serialized publication; no source translation |
| docs/experiments/glm52-decode-failure-ledger-2026-07-31.md | Retain failed experiments and topology-specific reopen conditions | Never generalize six-5090 or H200 results to this Windows machine |
| c/colibri.c prefix reuse; c/kv_persist.h | Repeated-turn prefix reuse, recurrent boundaries and exact restart | Shared cache/session owner; cold-versus-reused token/logit equality, bounded memory, cancellation |
| c/sample.h; docs/corpus-draft.md; docs/grammar-draft.md | Shared target-verified MTP, context, corpus and grammar draft sources | Span/quantization invariance first; forced rejection; unchanged output; positive end-to-end gain before default enablement |
| c/hybrid_split.h; c/route_trace.h; c/backend_loader.c | Layer offload, routing counts, held-out expert placement and overlap | Real small MoE oracle before larger offload; resource fit and unchanged semantics |
| c/kv_fp8.h; c/tests/test_absorb_determinism.cu | Cache representation experiments and adversarial kernel shapes | Independent numerical oracle, quality floors, race/lifetime and constrained-memory checks |
| c/openai_server.py; web/src/lib/api.ts | Explicit overflow, cached usage and output-limit outcomes | Consistent streaming, non-streaming, stored replay and restart behavior |
| web/src/Profiling.tsx | Turn profile, time series and tokens per forward | Counter-based records; exclusive phases plus unattributed time; overlaps drawn separately |
| web/src/Brain.tsx | Expert tier and routing heat visibility | Recipe-derived dimensions, exact counts, accessible table fallback and authenticated stream |
| web/src/App.tsx | Clear turn state, hardware and cache feedback | Extend current vanilla workbench; browser acceptance and actual model journeys separately |

## Important limits

The failure ledger records faster isolated work becoming slower complete
inference. Counts explain a hypothesis; they cannot approve an optimization by
themselves. Corpus replay's reduced forward count also greatly exceeds its wall
time gain. Repeated-prompt results are cache upper bounds, not mixed-workload
performance. Held-out placement can reverse an apparent training-set gain.

Resume downloads, truthful terminal states, prompt reuse and immutable grammar
bindings have immediate consumers in Overgo. Detailed profiling and MoE policy
should arrive with the first consumer that needs them. Conditional experiments
may close without promotion when their predicate or benefit is absent.

Output-identity receipts may retain numerical evidence. Historical timing,
memory and capacity observations keep their original source and environment.
Network cancellation and caller budgets remain effective when fixed timers are
replaced. Missing or skipped device/model checks never count as acceptance.

Colibri is an external behavioral and experimental reference. Reimplement
concepts in existing Go/CUDA components; record source revision and applicable
license before any source translation. This review copies no Colibri source.

Protocol admission reread `c/openai_server.py` (`_engine_error` and deferred
stream acceptance), `c/colibri.c` and `c/qwen38.c` at the pinned revision above,
under Apache-2.0. Overgo checks the fully prepared prompt, including projected
media, before generation, stream headers or durable Responses reservation.
Cached prefixes count within that prompt. Unknown provider counts remain with
the provider; decode budgets and context shifting remain with inference because
encoder-decoder source and decode contexts are separate. This transfers the
early-refusal behavior without translating source or importing engine margins.

Terminal outcomes reread `c/openai_server.py` (length-limited finish reasons
and valid tool-call precedence) and `web/src/lib/api.ts` (retained stream finish
reason) at the pinned Apache-2.0 source revision. Overgo uses its existing
generation pump and stop filter, persists the output-limit reason in the
interaction owner, and projects incomplete Responses through live delivery,
follow, restart and conversation history. Browser acceptance uses synthetic
generation and makes no model quality or performance claim. No source was
translated.

Protocol prompt reuse reread `c/openai_server.py` (`conversation_cache_slot`),
`c/kv_prefix.h` (fed-token identity and media taint), and the GLM-5.2 failure
ledger at the pinned Apache-2.0 revision. Overgo retains its existing bounded
prompt pool and projection/LoRA signatures. Ordinary chat, Responses and
Anthropic turns request reuse only from the selected execution owner when it
declares support; continuous generation retains its admission policy. Usage
comes from prefill observations. Cold/reused host and CUDA token/logit tests
cover multi-token decode, divergence, shortening and cancellation. Matched
ABBA host total-generation observations describe the hermetic fixture only, not a
production-model speed claim; device-greedy/speculative defaults are unchanged.
No source was translated.

Grammar vocabulary binding reread `c/colibri.c` (`grammar_setup_text`,
`grammar_reset`, and the request cache), `c/tests/test_grammar_cache.c`, and
`docs/grammar-draft.md` at the pinned Apache-2.0 revision. The reusable compiled
data and fresh per-request state contract informs a shared immutable vocabulary
binding in Overgo's sampling owner. Each runner decodes once; grammars share
owned pieces, terminal flags and names while samplers keep their own state.
No source was translated. Pre-change saved sampler states pin exact signature
compatibility. The legacy FNV stream hashes grammar source/root before vocabulary
bytes, so this slice preserves that stream; hash-once remains an explicit plan
obligation. Allocation counts establish removal of per-compile vocabulary copies,
not an end-to-end inference speed claim.

The hash compatibility study reread the same pinned `grammar_setup_text`,
`grammar_reset`, request cache, cache tests and grammar-draft limitations.
It tests an independently derived exact FNV-1a suffix transform: byte XOR only
changes the incoming low byte, while multiplication propagates higher bits
linearly. A table covering every possible low byte and a power of the FNV prime
therefore preserve the old stream for arbitrary grammar prefixes, using fixed
storage independent of grammar count. The standard-library sequential hash is
the independent oracle, including all incoming byte states, high-bit carries,
empty and binary pieces, terminal flags and distinct grammar prefixes.

This candidate remains test-only. Its single vocabulary traversal updates all
256 hash lanes, imposing a substantial setup cost on the first grammar even
though it amortizes over many distinct grammars. The acceptance receipt retains
paired ABBA hash-only observations including table construction on identical
serialized inputs, separately for one and many compilations. Those observations
exclude parsing, model execution and decoding; they are neither total-generation
evidence nor a production speed claim. Runtime promotion requires matched
complete inference evidence covering cold, repeated and rotating grammars with
setup and caller budgets included, or a cheaper exact construction. The original
hash-once row, exact state compatibility and required hash-reuse regression
remain open. No grammar cache, state format or production behavior changed.

The continuation repair retains accepted completion facts in the existing plan
authority and checks them through the loop's existing world adapter. It adds no
new evidence ledger, scheduler, or authority package. The observed failure was
two accepted prerequisite commits being counted as two failed attempts because
the parent step stayed open. Regression tests require those accepted results to
reset only the consecutive nonprogress count, while unchanged or invalid
receipts, merges, and pending validation earn no credit and the total invocation
ceiling remains fixed.

The reviewed harness baseline increases by one focused command adapter file,
374 syntax nodes, 40 nodes in the largest file, and 13 nodes in the largest
function. Its structural repeated-policy census rises by one group and five
sites. Package count, guarded scalar fields and layer violations do not change.
These measured additions bind progress to existing receipt, ownership and
validation checks; replacing them with HEAD movement would restore false
progress credit. The baseline records this bounded correctness cost without
claiming a runtime speed improvement. Moving the adapter into an unrelated file
merely to preserve the file count would not remove its complexity. This repo
repair does not authorize restarting the Colibri supervisor: the user selected
direct execution in the current task, and its background timer and campaign
were stopped on 2026-09-17.

Download integrity reread `c/download_fp8.py` at the same pinned Apache-2.0
revision. Its curl fallback illustrates interrupted-file reuse but size-based
publication is not a content-integrity oracle. The independent local HTTP
regressions reproduced a corrupted ETag-only prefix being published and complete
prefixes failing repeatedly with HTTP 416. A strong ETag validates the server
representation, not retained local bytes. Overgo now requires a declared SHA-256
for partial reuse; otherwise it restarts from the full remote representation.
After rehashing, a complete matching digest avoids another content request while
retaining the common cancellation, declared-size and publication checks.

The exact-byte fixtures require one full transfer for ETag-only recovery and
zero transfer requests for complete digest-verified content, including unknown
size and empty content. Incorrect declared size still refuses publication.
Malformed and wrong-total 416 cases use corrupt complete prefixes so the response
validation actually executes; every range-response case asserts a request was
made. Existing interrupted digest-backed transfers retain their served-byte
comparison. Files without a digest pay the explicit cost of a full restart.
These are integrity and transfer-count results, not a model-performance claim.
Existing final-file caching without a digest still relies on declared size;
staging inventory contamination, cross-revision reuse and transport timers are
separate outstanding work. No Colibri source was translated.

Live Activity reread `web/src/Profiling.tsx` and `web/src/lib/api.ts` at the
pinned Apache-2.0 revision. Colibri's turn view motivates the capability;
Overgo delivers it through the existing authenticated runtime SSE route and
shared browser subscription, without adopting the poller or React components.
No Colibri source was translated and no persistent record format changed.

Successful durable publication now sends the already-identified observation.
The record codec owns its slices; normal delivery needs no store visit. A
handler-local cursor ties initial and recovery snapshots to publication, while
artifact IDs retain observation identity across restarts. Subscriber buffers
and visible history use MaxStoredResponses; existing operation subscription
admission bounds the paired runtime streams. Overflow explicitly requests a
snapshot instead of silently losing state. Publication and snapshot capture
share a mutex, so connection/recovery snapshot reads can delay publication;
this is a consistency tradeoff, not a model-performance promotion.

The shared browser projection rejects old cursors, deduplicates record IDs,
retains its declared history limit, and carries forward current operation
states. Channel-rendezvous tests cover live publication, failed publication,
snapshot overlap, slow readers, cancellation, reconnect and handler restart.
The browser fixture exercises actual local completions, late Activity mounting,
bounded rows, reconnect, bearer authentication and zero activity-polling GETs.
These establish delivery and UI behavior on a synthetic serving fixture; they
do not establish production-model quality or throughput.

Both exact acceptance commands remain mandatory in a verification batch. The
first combined gate attempt failed evidence parsing: its selector scanner
counted the browser lane's `-run` as a second direct Go test, while the lane
reports its own checked browser verdict. The server test is now a required
checkpoint and the unchanged browser command is the final integration check.

The user reprioritized basic model correctness before further MoE work. At the
pinned Apache-2.0 Colibri revision, `c/olmoe.c` documents the checkpoint's exact
embedded chat template and unusual shared BOS/EOS token. This motivates preserving
artifact declarations through common components, without translating Colibri's
family-specific formatter. Overgo already executes GGUF Jinja in its bounded Go
interpreter; the HF converter previously ignored `tokenizer_config.json` templates.

The shared dense/Qwen importer now reads either an embedded string or a named
template list. A standalone `chat_template.jinja` overrides the default while
retaining named variants. Source bytes are preserved, metadata order is stable,
and malformed, duplicate, empty or default-less declarations refuse conversion
instead of falling through to an unrelated prompt format. `tool_use` maps to the
existing GGUF runtime selector. Other names are retained as metadata; arbitrary
named selection is not added. This follows the tokenizer loading precedence in
[Transformers documentation](https://huggingface.co/docs/transformers/chat_templating_writing#storing-and-loading-chat-templates).
Modern additional-template directories, legacy processor template JSON, and
convergence of the separate Gemma converter remain outside this change.

Negative source tests reproduced dropped embedded templates before the change.
The public-conversion test uses the existing tiny dense safetensors builder,
writes GGUF, reloads its tokenizer and checks exact rendered messages, tools,
generation/reasoning flags and unchanged BOS/EOS IDs. Existing pinned Jinja byte
parity remains a separate mandatory check. These prove metadata and prompt
semantics, not numerical quality, full Python/Jinja compatibility or throughput.

The next ready step preserves every declared end-of-generation token. Independent
probes found truncated generation arrays, missing dense model-config fallback,
unchecked invalid trailing IDs and Go's null-to-zero integer decoding pitfall.
The corrected plan retains explicit generation-config precedence rather than
blindly unioning model and generation sources. Master synchronization against
`69ffcc7423fa498c5df1dd4aaee269838626fe27` could not prepare because Windows denied
renaming the retained evidence snapshot; its evidence remains intact and the
merge retry remains due. No master plan was imported.

The stop-token transfer reread `c/colibri.c` at the same pinned Apache-2.0 source.
Its complete-stop rationale is useful, but its cross-source union and eight-ID
limit are not adopted. A non-null generation declaration takes precedence over
model EOS; otherwise the existing unresolved-value fallback applies. The importer
validates every integer and range, deduplicates without reordering the primary
EOS, and retains the complete selected list in `overgo.tokenizer.eos_token_ids`.
This is an Overgo extension; the usual scalar GGUF EOS remains the first ID.
The runtime retains its existing recognized-name and FIM stopping conventions.
The import concerns token identities rather than every Transformers generation
setting, including its separate treatment of a generation config file.

The existing end-marker map also retains which IDs were explicitly declared EOS.
Semantic vocabulary artifacts preserve those EOS roles, and exact row mapping
rejects differing roles. Existing scalar vocabularies and token encoding retain
their behavior. Public dense conversion/reload tests include ten declared stop
IDs, source precedence, scalar zero, null/unset fallback, duplicates, absent EOS
and invalid trailing entries. Nested Qwen tests isolate configuration decoding
and shared tokenizer metadata; they are not a full Qwen numerical execution.

GGUF named variants now also publish `tokenizer.chat_templates`, the sorted
non-default name index described by llama.cpp's `gguf_writer.py` at
`42fc243060709331ff9b158a9ed2cbe37219ae83`. A negative test reproduced the missing
index. Existing Overgo default/tool-use selection and template bytes are retained;
exact rendering and pinned Jinja byte parity remain required. No additional
renderer, artifact acquisition or numerical/throughput promotion is involved.

The reasoning parser transfer uses the existing buffered parser and tool stream.
The retained Gemma Jinja template advertises `<|channel>thought` / `<channel|>`;
Colibri's pinned `c/openai_server.py` advertises `<|content_thinking|>` /
`<|content_text|>` for its content-channel wire. The source pin and Apache-2.0
license above apply. The transferred facts are these delimiters and the need to
retain split markers, rather than its Python layer, architecture switches or
fixed prefix-length limit. Legacy `<think>` handling remains the fallback.

Negative probes returned Gemma's thought channel as visible content in buffered
and split-marker parsing, and made a tool example inside an explicitly opened
think block fail as an unterminated block. Supported syntax is now selected
from the loaded default or tool-use template, ambiguity refuses, and buffered
parsing and tool-stream prefix scanning share the marker splitter. Streams
resolve syntax once and withhold explicitly opened reasoning until closed.
Tests cover every split boundary, unchanged plain bytes, declaration selection,
completed malformed blocks, and actual tool deltas following reasoning.

The chat completion API now uses the existing output stream for buffered and SSE
responses when the generator provides prompt-aware parsing. Native Runner uses
the actual prepared token IDs, rendered with control spellings, to establish a
reasoning opener already supplied by the template; this also covers projected
prompts whose token sequence differs from their original text. Formatting and
output parsing share default/tool-use template selection. The legacy complete
parser and providers without the new capability retain their existing paths.

The channel splitter preserves partial markers and UTF-8, emits ordinary text
and reasoning before generation finishes, and keeps tool examples in reasoning
out of the tool scanner. No active tools, including `tool_choice: none`, leaves
tool-like visible prose intact. Buffered and streamed results preserve the same
visible prefix bytes before a tool call. Pure channel deltas create no empty
tool events; finalization checks already-sent prefixes and sends only remaining
text. Unfinished declared reasoning stays in its own field when a budget or stop
ends generation. Eager final-output grammars keep their constrained result as
visible content; lazy grammars can still carry declared reasoning.

A native token-delivery regression found that control-typed channel markers
were stripped before reaching the parser. A request-scoped token-event decoder
now preserves only pieces of declared reasoning delimiters and active tool-call
delimiters. Unrelated controls and EOG controls remain suppressed. The shared
generation adapter applies this rendering to callbacks and stop filtering;
token IDs, sampling and the returned detokenized sequence remain unchanged.

Public HTTP checks cover buffered/SSE output at every split boundary, bytewise
Unicode, prompt-opened/closed states, tool examples, budgets/stops, immediate
progress, provider fallback and structured output. A cancellation regression
also required checking context before routing each piece, preventing one extra
content chunk after cancellation. Runner tests exercise actual prepared control
tokens, named-template selection and native token delivery; these complement
the existing parser, tool grammar, Jinja, Responses and Anthropic contracts.
This does not claim general Jinja grammar inference, native tool syntax for
every model, numerical model accuracy or throughput.

## Declared NFC normalization

The HF tokenizer census found a concrete Krea mismatch: its tokenizer declares
NFC, but composed `café` produced `[924, 58858]` while decomposed `café`
produced `[924, 1859, 53839]`. Krea tokenizer SHA256 is
`be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506`.
Fractale and Granite declare no normalizer; Nemotron declares a precompiled
Sequence with Metaspace and is used for decoding. SenseNova's split-file
tokenizer path has no changed normalization declaration.

The shared `internal/tokenizer.NormalizeNFC` owner now supplies Unicode 16
canonical ordering and composition to the HF encoder when NFC is declared.
It reuses the existing decomposition table and adds only its version delta,
combining classes and composition data from pinned Colibri. WPM accent
stripping and native GGUF tokenizer profiles are unchanged. The source,
regeneration procedure and licenses are recorded in
[the normalization contract](../internal/tokenizer/NFC.md).

Raw added tokens remain outside normalization. Unsupported normalizers and
unsupported NFC added-token matching now refuse encoding; loading still permits
valid decoding. General Sequence pipelines, pre-tokenizer compilation and
normalized added-token matching remain open in the broader text-semantics step.

Acceptance is frozen against the independent Unicode 16 normalization corpus,
not outputs generated by this implementation. It covers 19,965 rows, every NFC
column invariant and unlisted scalars, with separate long combining-run,
invalid-byte, token-boundary and legacy-mode checks. Actual Krea now gives the
same `[924, 58858]` for both spellings. The existing fox-prompt IDs and mask
remain byte-identical; that prompt is also the frozen Krea request
`file:sha256:1d840594046f2981e9fb7c335808a4f3399101072ab0a8ab2768492642f9b6e2`.
Other prompts whose IDs change still require affected output reacquisition;
this slice makes no model-numerical or throughput claim.
