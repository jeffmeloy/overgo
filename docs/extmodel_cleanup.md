# Model Runtime Cleanup

Status: active  
Updated: 2026-08-14
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
| Model/inference production files | 143 | Diagnostic; ownership matters, not file count |
| Model/inference production functions | 952 | Reduce by caller migration and deletion |
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
| Duplicated CUDA launch frames | 0 | Node frames plus compact launch indices |
| Resident image per-execution device-feed maps | 0 | Compiled input slots; reusable pointer slabs |
| Resident image parallel weight ownership maps | 0 | Indexed pointer/refcount/byte slots |
| Hybrid-local Q/K/V projection implementations | 0 | Shared projection stage owns fused/separate binding |
| Resident video full host-weight owners after upload | 0 | Retain timestep bundle only |
| Resident video named weight maps after binding | 0 | Node-to-device feeds own execution bindings |
| General recipe VQA runtime adapters | 0 | Extract command-local processor/device pipeline |
| Production video recipe activation paths | 0 | Bind profile artifact; require active recipe |

Family-named production files: none.

## Ordered Work

1. Decompose latent attention, hyper attention/feed-forward, recurrent, and hybrid operators only where an existing stage replaces displaced code.
2. Move VQA prepare/generate execution from `cmd/vqaparity` into the general capability runtime; delete command-local orchestration.
3. Bind latent-video config/policy as a profile artifact; activate and resolve production video through its recipe.
4. Compile video artifact names and codec selection from the active profile; remove command/runtime literals.

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
