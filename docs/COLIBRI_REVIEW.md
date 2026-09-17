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
