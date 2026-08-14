# Model Runtime Cleanup

Status: active  
Updated: 2026-08-13  
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
| Family-named production files | 7 | Delete through vertical migration |
| Family-identity-bearing production files | 21 | Restrict to parsing/wire facts, then remove runtime uses |
| Post-resolution architecture catalog lookups in execution compilation | 0 | Hold at zero |
| Family switches in compiled layer execution | 0 | Hold at zero |
| Family-owned alternate-state forward files | 1 | Next migration |
| Test-only backward/oracle files compiled into production | 0 | Hold at zero |

Family-named files still live:

- `internal/model/dflash.go`
- `internal/model/eagle3.go`
- `internal/model/gemma4_assistant.go`
- `internal/model/wavtokenizer.go`
- `internal/inference/gemma3n.go`
- `internal/inference/runner_noncausal_t5.go`
- `internal/inference/wavtokenizer_audio.go`

## Ordered Work

1. Compile alternate-state initialization, prediction, correction, injection, activation, and merge as neutral operators; migrate serving and training; delete Gemma-specific host orchestration.
2. Fold feature-draft, paired-cache, DFlash, and audio-token leaves into typed projection/session programs; delete family wrappers and filenames.
3. Make encoder/decoder and audio-token coordinators consume sealed session programs; remove T5 and WavTokenizer runtime ownership.
4. Move remaining architecture validation constants into resolved profile facts or derived artifact relationships.
5. Remove duplicate host/reference probes once operator-level host/CUDA evidence owns the same contract.

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
