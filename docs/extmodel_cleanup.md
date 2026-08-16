# Model Runtime Cleanup

Status: active  
Updated: 2026-08-15
Scope: `internal/model`, `internal/inference`, model recipes, direct callers

## Endpoint

```text
active recipe + resolved artifact facts
-> sealed forward/layer/cache/projection programs
-> neutral shared operators
-> output + evidence
```

- Recipe identity owns topology, sequencing, policies, and lifetimes.
- Execution branches on typed operators, never model or architecture identity.
- Artifact compatibility ends at parsing and tensor binding.
- No runtime wrapper, alias, fallback, or parallel family implementation survives migration.
- Unique math becomes a neutral operator even with one current consumer.
- Each wave migrates every compatible caller and deletes the displaced authority.
- Numerical order stays fixed unless host and CUDA parity prove a deliberate change.
- Production code is net-negative by default; retained growth must remove execution work, copies, or residency.

## Current Census

| Surface | Count | Treatment |
|---|---:|---|
| Repo production files / AST nodes | 749 / 1,024,876 | Structural baseline |
| Repo test files / AST nodes | 586 / 703,108 | Structural baseline |
| Exported declarations / import edges | 4,371 / 2,231 | Reduce single-consumer exports |
| Exact duplicate excess AST nodes | 11,455 | Profiler-ranked deletion queue |
| Model/inference production files | 143 | Diagnostic; ownership matters, not file count |
| Model/inference production functions | 943 | Typed plans and operators own execution |
| Family-named production files | 0 | Hold at zero |
| Family-identity-bearing production files | 21 | Restrict to parsing/wire facts, then remove runtime uses |
| Post-resolution architecture catalog lookups in execution compilation | 0 | Hold at zero |
| Family switches in compiled layer execution | 0 | Hold at zero |
| Family-owned alternate-state production files | 0 | Hold at zero |
| Alternate-state facts on generic forward program | 0 | Hold at zero |
| Alternate-state magnitude-rescale copies | 0 | Hold at zero |
| Global audio waveform codec facts | 0 | Hold at zero |
| Hardcoded alternate/draft metadata defaults | 0 | Hold at zero |
| Host-load tensor-topology probes | 0 | Compiled recipe/model admission owns topology |
| Device-bind tensor-topology probes | 0 | Compiled recipe/model admission owns topology |
| Runner graph-binding pass-through helpers | 0 | Runtime owns feeds and placement branch |
| Inference-owned sequence tensor catalogs | 0 | Model binder consumes compiled catalog |
| Invocation-time graph schema discovery | 0 | Reflect once; bind by indexed descriptors |
| Registry-backed model-plan compilation APIs | 0 | Commands pass resolved profile explicitly |
| Unbound `Spec` policy fallback lookups | 0 | Metadata admission binds the exact profile |
| Host cached-generation loops | 1 | Generate and resumable sessions share one loop |
| Continuous token-acceptance paths | 1 | Greedy, top-K, and full-logit sampling converge |
| Continuous scheduler cohort allocations per decode step | 0 | Reuse bounded scheduler scratch |
| Default runner lock/open admission implementations | 1 | Shared nil, lock, closed, and unlock-on-rejection contract |
| Scalar capability request-admission paths | 1 | Decode, validate, and identity-check once for resident and ephemeral execution |
| Image stage topology declarations | 1 | Shared prepare, integrate, and decode schema; modules supply math |
| Resident capability construction/failure adapters | 1 | One typed executor adapter owns ready/error conversion |
| Dead graph-runtime feed mutators | 0 | Weight/layer binders own feed registration |
| Graph-runtime executor callback types | 0 | Runtime selects reference, CUDA host-feed, or CUDA device-feed execution |
| Hardcoded Gemma3N validation facts | 0 | Serialized typed profile facts |
| Exported registry-backed profile resolution APIs | 0 | Bootstrap stays inside model package |
| Hardcoded RWKV execution/validation scalars | 0 | Serialized typed recurrent profile facts |
| Test-only backward/oracle files compiled into production | 0 | Hold at zero |
| Generic projector artifact reopens | 0 | Probe and build share one admitted handle |
| Projector runner lifecycle implementations | 1 | Embedded resource owner closes file and device state |
| Manual variable-length auxiliary tensor checks | 0 | Relational tensor requirements own rank/type/nonempty |
| Invalid layer-program sentinels | 0 | Compilation returns explicit errors |
| Duplicated tensor-descriptor maps per model load | 0 | Borrowed indexed catalog owns descriptors |
| Parallel weight requirement lookup authorities | 0 | Indexed catalog owns presence, lookup, and validation |
| Ordinary CUDA launch tensor-pointer lookups | 0 | Flat compiled operand slots |
| CUDA launch tensor-pointer lookups | 0 | Ordinary and fused launches use compiled slots |
| Runtime CUDA fusion descriptor maps | 0 | Rewrite map released after indexed launch compilation |
| Runtime CUDA rewrite decision maps | 0 | Node and fusion frames retain compiled decisions |
| Compiled CUDA tensor-pointer map APIs | 0 | Indexed input slabs are the sole compiled contract |
| Retained compiled CUDA execution methods | 1 | Inputs, targets, and attributes share one indexed entry point |
| Single-axis RoPE builder entry points | 1 | Typed options own layout and scaling policy |
| Routed-expert builder entry points | 1 | Typed options own routing, activation, and storage policy |
| General attention builder entry points | 1 | Typed options own masking, window, bias, and sink policy |
| Routed per-layer device-binding maps | 0 | Branch input programs retain compiled slots |
| Duplicate dense FFN builders | 0 | Feed-forward policy owns layout, activation, bias, and projection |
| Routed/denoiser allocation owners | 1 | Shared ordered allocation set; VAE keeps distinct batched/workspace ownership |
| Duplicated CUDA launch frames | 0 | Node frames plus compact launch indices |
| Resident image per-execution device-feed maps | 0 | Compiled input slots; reusable pointer slabs |
| Resident image parallel weight ownership maps | 0 | Indexed pointer/refcount/byte slots |
| Hybrid-local Q/K/V projection implementations | 0 | Shared projection stage owns fused/separate binding |
| Resident video full host-weight owners after upload | 0 | Retain timestep bundle only |
| Resident video named weight/feed maps after binding | 0 | Compiled input slots own execution bindings |
| General recipe VQA runtime adapters | 0 | Extract command-local processor/device pipeline |
| Production video recipe activation paths | 1 | Shared capability preparation and active resolution |
| Projector no-policy constructor wrappers | 0 | Fifteen callers migrated to option-bearing constructors |
| Advisory workflow recipe implementations | 0 | Deleted; executable model recipes remain |
| Capability command family switches | 0 | Compiled entry module selects implementation |
| Repeated image/video artifact classifiers | 0 | Inventory and definition share one source resolution |
| Coarse capability modules | 31/34 | Replace staged family shells with neutral operators |
| Family capability runtime files | 91 / 21,953 lines | Separate unique math from displaced orchestration |
| Family capability exported functions | 295 | Delete wrappers and direct orchestration after caller migration |
| Production projector recipe consumers | 2 | Server and generate require active projection plans |
| Active projection lifecycle dimensions | model + projector + modules | Exact bundle authority |
| Projection command lifecycle implementations | 1 | Shared capability prepare, verify, and activate path |
| Duplicate projection modality enums | 0 | Standard recipe data kinds own media identity |
| Projection inspection runtime constructions | 0 | Artifact descriptor supplies typed media facts |
| Training authority timing | pre-allocation | Training plan resolves before runtime construction |
| Typed architecture profile facts | 166 | Profile/artifact facts own omitted-value policy |
| Model-specific `optionalOr` defaults in `model/spec.go` | 0 | Three remaining calls use multiplicative identity |
| Production `legacy` / `fallback` line hits | 37 / 93 | Format, tokenizer, schema, and explicit policy vocabulary |
| Implicit runtime compatibility fallbacks | 0 | Recipe residency owns device-OOM host recovery |
| Media codec program contracts | 1 | Shared typed operations; resident/streamed bindings stay explicit |

Family-named production files: none.

## Ordered Work

1. Replace coarse prepare/integrate/decode shells with neutral operator programs.
2. Move remaining codec topology declarations from artifact-specific compilers into recipe facts.

## Completed Waves

- Recipe module dispatch authoritative; family capability switch deleted.
- Recipe v1, old KV readers, sampler readers, and undeclared tokenizer policy deleted.
- Training authority compiled before execution allocation.
- Advisory workflow recipes deleted: `-443` net lines.
- Scalar GGUF metadata construction centralized: `-67` net lines.
- Stack scratch pooling mandatory: `-31` net lines; compatibility branch deleted.
- BF16 slice rounding centralized: `-18` net lines.
- Flow-time shifting centralized: `-20` net lines including touched-comment cleanup.
- Latent-attention RoPE policy compiled once: `-28` net lines; YaRN wrappers deleted.
- JSON artifact identity centralized across four domains: `-5` net lines.
- Resolved normalization policy handed back to model-plan compilation; stale-profile execution fixed.
- Rotary omitted-value policy moved into 166 typed profile facts: two waves, `-132` net lines.
- Duplicate Step35 metadata object removed; one serialized policy owner remains.
- Duplicate projector inventory maps removed: `-16` net lines.
- Indexed CUDA execution made authoritative: two map-execution APIs deleted.
- Test-only MoE builder facade deleted: `-218` net lines.
- Typed single-axis RoPE options made authoritative: `-119` net lines.
- Typed general attention options made authoritative: `-202` net lines.
- Active projection bundle binds exact model, projector, and modality modules; server/generate direct opens deleted.
- Projection lifecycle consolidated; command forks and modality inference deleted: `-105` production lines.
- GGUF inventory requires explicit model/projector kind; implicit identity fallback deleted.
- Inventory-only VQA capability registration deleted; verified VQA runtime remains recipe-bound.
- T5/Qwen/Step/Grove omitted metadata moved into typed profile facts: four literals removed.
- Image/video codec channel norm and spatial attention unified in `hostmath`; two local implementations deleted.
- Expert width derivation is exact; shared-expert fallback relationships are typed profile facts.
- Oscillator image/video share one runtime registration owner; video-only facade deleted.
- Packed-batch remainder and recipe-owned host recovery named precisely; runtime fallback ambiguity removed.
- Image/video/source codecs share one typed program; local enums, validators, arity switch, and plan wrappers deleted.
- Context-pipeline-tail abstraction rejected: `+26` lines for three consumers; no commit.

## Per-Wave Gate

- Old family symbol and fallback absent by `rg`.
- Recipe/profile admission and identity mismatch fail before allocation.
- Focused numerical parity, `go test ./...`, and `go vet ./...` pass.
- CUDA/device gate passes for graph, layout, cache, or residency changes.
- Production declaration/line delta and transfer/allocation delta recorded in the commit.

## Stop Conditions

- Neutral primitive lands without all compatible callers.
- Old and new runtime authorities coexist.
- Runtime reads model identity or repairs an incomplete recipe.
- Model fact becomes a code literal.
- Numerical behavior changes without evidence.
