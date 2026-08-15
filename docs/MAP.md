# Reuse map — capability → existing owner

Read this BEFORE grepping for "does X exist" or writing a new helper. If a row
covers your need, use that owner; do not re-derive it. If a row is missing, add
it here when you find or build the owner. This map exists because the recurring
failure mode is agents re-discovering (or reinventing) code that already exists.

## Models and datasets (RepoDB is the catalog)
| Need | Use |
| --- | --- |
| List available models (present, active recipe) | `discovery.Servable(ctx, store, limit)` |
| List available datasets | `store.Query(ctx, repodb.Query{Kind: artifact.KindDataset})`; mirror `discovery.Servable` if a typed lister is wanted |
| Resolve a model's active definition + tensor inventory | `modelrecipe.ActiveRecord` → `activation.Definition.Dependency(recipe.DependencyDefinition, 0)` → `modelrecipe.ResolveModelDefinition` (`.Tensors`) |
| Load a model for inference (identity-bound) | `clioptions.OpenRunner` / `modelrecipe.ResolveActiveGGUF` |
| A model/dataset's on-disk location | `store.Locations(ctx, id)` (see `discovery.presence` for the stat pattern) |
| Data roots (store, models, datasets, checkpoints) | `dataroot.ResolveCurrent()` → `Roots` |

## Tensor / model artifacts
| Need | Use |
| --- | --- |
| Build a model's tensor inventory | `modelartifact.FromGGUF` / `modelartifact.FromHFRepository` |
| Characterize a model's tensors (sampled, distribution-free) | `modelartifact.MeasureGGUF` / `MeasureSafetensors`; format-agnostic by location: `modelartifact.MeasureAtLocation` |
| The per-tensor profile itself (L-moments, energy, quartiles) | `internal/tensorstats.Characterize` (embedded in `modelartifact.TensorMeasurement`) |
| Effective rank (spectral) | `tensorstats.EffectiveRank` / `EffectiveRankOf`; singular values via `tensorstats.SingularValues` |
| Shape similarity (rank-based, no metric prior) | `tensorstats.ShapeFeatures` + `tensorstats.Nearest` |
| Open weights by format | `gguf.Open` (file) / `safetensors.OpenSource` (dir) — or just `MeasureAtLocation` |

## Media execution

| Need | Use |
| --- | --- |
| Execute SenseNova routed-image denoising | `routedlm.NewDeviceGenerationSession` retains prefix/body state; `routedlm.NewDeviceFlowPeripheralSession` retains vision, condition, and flow-head projections; `sensenovarecipe.LoadGenerator` binds the production recipe |
| Compile and execute a Wan/LiveEdit causal VAE source encoder | `latentvideo.CompileVAEEncoderPlan` + `CompileSourceCodecBoundary`; host oracle `EncodeSourceVideo`; resident device path `NewVAEEncoderCUDASession` |
| Execute LiveEdit chunks with retained text and bounded self-attention K/V | `latentvideo.NewReferenceEditDenoiserCUDASession`; neutral history graph: `model.ConditionedDiffusionProgram.BuildBlockWithSelfHistory` |
| Run typed LiveEdit source-to-frames device composition | `latentvideo.NewReferenceEditRuntime` → `ReferenceEditRuntime.Run` |
| Activate Wan or LiveEdit through a retained recipe session | `cmd/recipe.videoCapability` routes `model.latent-video-*` / `model.reference-video-*`; `capabilityruntime.NewScalarSessionCache` / `NewMappedSessionCache` owns residency |
| Publish decoded Wan or LiveEdit frames | `latentvideo.EncodedVideo` + `GIFContent`; `PixelRange` owns signed Wan versus unit-range LiveEdit quantization; `MP4Encoder` streams full-clip evidence through FFmpeg without retaining float frames |

## HTTP serving

| Need | Use |
| --- | --- |
| Serve native, OpenAI, or Anthropic protocols | `server.Handler`; protocol files translate into shared generation options |
| Prove cross-protocol behavior | `server.TestAdaptiveServingContractMatrix`; focused endpoint tests remain the implementation evidence |
| Edit request cache windows | Native completion `cache_prompt`, `n_cache_reuse`, `n_keep`, and `n_discard`; `inference.Runner` owns retained state |

## Training state

| Need | Use |
| --- | --- |
| Publish or load an exact-resume checkpoint | `trainingprogram.PublishCheckpoint` / `LoadCheckpoint`; one staged, non-overwriting directory |
| Resume dense host or resident CUDA Muon | `densecausal.TrainBatchesResume` / `TrainDeviceResidentBatches` |
| Load and train a real gated-delta recurrent layer | `hybridtrain.LoadRecurrentLayerArtifact` / `Model.TrainDeviceResident`; execution order is `Model.Program()` |
| Train an artifact-declared per-layer residual adapter | `adaptertrain.LoadArtifact` / `Model.BuildExample` / `Model.Step`; shared `hostmath.PerLayerAdapter*`, device Muon, and `trainingprogram` checkpoint state |
| Compile corpus-bound multimodal training authority | `trainingprogram.NewObjective` / `CompileObjectiveMatrix` / `CompileTrainingRunPlanFromRepository`; absent RepoDB objective evidence yields a refused row |
| Compile objective semantics through one update order | `trainingprogram.CompileObjectiveProgram` / `CompileTrainingObjectiveMatrix`; programs bind objective kind plus shared forward/backward/Muon operators, absent program bindings are refused |
| Evaluate typed training outputs | `trainingprogram.EvaluateNative`; objective documents select token accuracy, signed/unit-range image/video PSNR, audio SNR, forecast MAE, or table accuracy |
| Train existing latent-sequence or image-flow objectives | `speechsynth.NewJointTrainer` / `diffusionimage.NewTrainer`; both expose their compiled `TrainingProgram` |
| Compile, train, evaluate, and externally promote a scratch workflow controller | `controllertrain.Compile` / `TrainSeed` / `Decide`; actions come from `workflowrecipe`, execution stays in `scratchmodel.ResidentTrainer`, and RepoDB stores corpus, run, evaluation, decision, and rollback lineage |
| Compile and admit recursive improvement trials | `trainingprogram.CompileImprovementProposal` / `runrecord.AdmitImprovement` / `DecideImprovement`; proposals cannot authorize admission, evaluation, promotion, or rollback |
| Compile a measured training-memory schedule | `trainingprogram.CompileMemorySchedule`; `scratchmodel.TrainingMemorySegments` supplies exact flat parameter, gradient, and Muon-state geometry |
| Bind dataset progress | `trainingdata.StreamState`; checkpoint form is `trainingprogram.DatasetState` |
| Decode real image/audio training records | `trainingdata.ImageProcessor` / `Image`; `trainingdata.AudioProcessor` / `Audio` |

## Commands / process (Go owns policy; scripts are bash or Go)
| Need | Use |
| --- | --- |
| Run a subprocess, capture combined output | `clioptions.CombinedOutput(env, name, args...)` |
| Pass/fail label, bounded output tail | `clioptions.Verdict(err)`, `clioptions.Tail(s, n)` |
| A `cmd/*` main entrypoint | `clioptions.MainNamed("<name>", run)` |
| The current task of record | `go run ./cmd/plan -next` (trust it; do not carry the goal in memory) |
| Park an out-of-scope finding | `go run ./cmd/finding -title <t> -severity <s> -owner <surface> -evidence <fact> -closure <path> -check <failable-check>` |
| Rank repeated raw policy literals | `go run ./cmd/closure-scan -raw` (advisory ownership evidence; math/structure facts excluded) |
| Eliminate an exact internal forwarding wrapper | `go run ./cmd/tighten` lists mechanically safe proposals; `-apply <id>` migrates all direct callers, deletes the wrapper, and retains the transaction only when package tests pass and AST surface falls |
| Validate or propose RSI roadmap work | `go run ./cmd/roadmap` derives states from reachable gate results, the live plan, leases, and exact-verifier probe evidence; output is `PROPOSED` only and never mutates `docs/plan.json`; `-probe-row <id>` executes the exact named verifier and records only a healthy target-absent run binding |
| Inject / advance a plan task | `go run ./cmd/plan -add -title <t> -vcmd <verify> <id>`; `-advance <id> <step>` |
| Synchronize local master | `go run ./cmd/plan -sync-master`; if a merge is prepared, finalize with the plan-bound gate `-merge` |
| Commit (required; raw `git commit` is guard-blocked) | `go run ./cmd/gate -message-file <f> -paths <csv> -plan <item>/<step>` |
| Finalize a merge | `git merge --no-ff --no-commit <branch>` then `go run ./cmd/gate -merge -plan <item>/do` (long: run backgrounded to avoid the 2-min shell timeout) |
| RepoDB store | `repodb.Open(root)`; `store.Query` / `Content` / `Locations` / `Commit` |

## Tests
| Need | Use |
| --- | --- |
| Write a GGUF fixture | `testutil.TempGGUF(t, name, metadata, tensors)` / `testutil.WriteGGUF` |
| float32 → little-endian tensor bytes | `testutil.Float32LE(values)` |
| Artifact/id/repo fixtures | `internal/testutil` (`ArtifactID`, `PublishArtifact`, `MonoPCM16WAV`, …) |
