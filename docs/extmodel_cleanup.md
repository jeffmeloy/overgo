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
| Model/inference production files | 140 | Diagnostic |
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
| Dead graph-runtime feed mutators | 0 | Weight/layer binders own feed registration |
| Hardcoded Gemma3N validation facts | 0 | Serialized typed profile facts |
| Exported registry-backed profile resolution APIs | 0 | Bootstrap stays inside model package |
| Hardcoded RWKV execution/validation scalars | 0 | Serialized typed recurrent profile facts |
| Test-only backward/oracle files compiled into production | 0 | Hold at zero |

Family-named production files: none.

## Ordered Work

1. Consolidate resident image-session construction and failure executors.
2. Separate specialized runner availability errors from shared lock admission.

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
