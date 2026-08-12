# Overgo runtime-authority tightening plan

Goal: recipes and compiled neutral operator programs are the only production runtime authorities. Each wave migrates every caller in its scope, deletes displaced orchestration in the same commit, verifies numerical behavior, then merges `master`.

## Gates

- Host: focused package tests, numerical graph parity tests, `go test -count=1 ./...`, `go vet ./...`.
- CUDA: affected Windows/CUDA executor tests plus resident/device parity lanes when hardware is available. Environment-gated tests must report skips, never silently substitute host execution.
- Change control: commit one complete vertical wave; merge `master` immediately afterward; resolve toward the newest neutral runtime.
- Paydown: production family-builder/direct-dispatch count and code surface must fall in every migration wave.

## Runtime-authority ledger

| Wave | Scope | Remaining production authority | Exit condition | Status |
|---|---|---|---|---|
| 1 | Draft/MTP layers | `compiledDraftBlock`; direct NextN and Step3.5 calls to `BuildArchitectureBlockCached` | `DraftLayerProgram` executes all draft blocks; adapter spec argument and direct calls deleted | Complete |
| 2 | Trunk host graphs | `buildLayerBlockFromPlan`; three direct runner dispatches | compiled trunk program owns construction; helper and direct dispatches deleted | In progress |
| 3 | Continuous/device cache | direct dispatch in `device_cache.go` | decode session invokes compiled program with indexed cache bindings | Pending |
| 4 | Auxiliary decoders | direct Eagle3 and Gemma 4 assistant dispatch | compiled auxiliary programs own topology and bindings | Pending |
| 5 | Encoder/decoder | exported T5 encoder/decoder builders | typed encoder/cross-attention programs own ordered stages; builders deleted | Pending |
| 6 | Diffusion/video | `BuildConditionedDiffusionBlock` | compiled diffusion program owns block/head stages; direct builder deleted | Pending |
| 7 | Compatibility dispatch | exported `BuildArchitectureBlockCached` and direct-construction tests | all callers execute sealed programs; compatibility entry point deleted | Pending |
| 8 | Leaf operators | family-named attention/recurrent/FFN leaves selected inside neutral instructions | mathematical operator policies replace family naming where semantics match; redundant leaves deleted | Pending |
| 9 | Recipe boundaries | remaining command/server task switches or runtime reconstruction | active compiled recipe selected once at assembly; no serving fallbacks/probes | Pending |

## Current inventory

- Production direct generic block-dispatch callers: 6.
- Production T5 builder calls: 2.
- Production conditioned-diffusion builder calls: 1.
- Draft programs already carry executable `Spec` and `LayerPlan`; execution authority is the first narrow deletion seam.

## Wave log

- Baseline: branch includes current `master`; worktree clean; full repository tests and vet passed before this campaign.
- Wave 1: sealed `DraftLayerProgram.Build` owns Qwen3.5, Cohere2, NextN, Step3.5, and HYV3 draft block construction. Deleted `compiledDraftBlock`, its translation result, the adapter callback, ignored spec input, and three direct generic dispatches. Production delta: +40/-50. Host model/inference tests and vet pass. CUDA parity gates compile and report explicit `OVERGO_CUDA_TEST=1` skips on this environment.
