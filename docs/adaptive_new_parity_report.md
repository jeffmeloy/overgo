# adaptive_new Parity and Performance Report

Validation base: Overgo `1a5074c` plus the E4B image-language parity slice;
adaptive_new `214950b3b`; reviewed 2026-08-13. adaptive_new working-tree changes
remain observations, not landed evidence.

Scope: adaptive_new Go runtime and training code, model-native Python oracles,
Overgo host/CUDA runtime, recipes, training paths, compatibility claims, and
current automation gates. Model, dataset, and checkpoint directories were
read-only.

## Verdict

Overgo is the better destination architecture. It already beats adaptive_new
on several real inference paths. It does not yet match adaptive_new's training
breadth or media breadth, and its production trainer does not yet exercise its
best resident implementation.

- Runtime lead: compiled programs, neutral capability packages, cgo-free CUDA,
  GGUF breadth, native quantization, retained device state, smaller Python
  surface.
- Proven wins: Qwen3.5-4B decode, Gemma E4B text decode, RxBrain VQA,
  SimpleDiffusion host forward, MiniCPM decode, Wan exact-quality generation,
  and Krea 256/2048 generation on the recorded protocols.
- Training lead remains adaptive_new: more objective types, more artifact
  trainers, more promotion/evidence machinery, and real media-training paths.
- Critical correction: Overgo's E4B item is marked done although the real E4B
  model has not trained end to end. Device VJPs and a synthetic hybrid trainer
  are prerequisites, not E4B closure.
- Production dense training beats adaptive's retained Carbon causal protocol;
  Qwen, hybrid, E4B, checkpoint, and promotion remain open.
- Critical evidence gap: 34 compatibility claims are marked `implemented`, but
  the current file does not encode the doctrine's evidence tiers. All 135
  architecture entries are `experimental`; only four name a validated fixture.

The target is not code-for-code parity. The target is capability parity through
Overgo's own contracts, then strict performance leadership on matched artifacts.

## Evidence Rules

- `measured`: current or retained numeric evidence with artifact/protocol named.
- `exact`: hashes, tokens, or bounded tensor comparisons pass.
- `implemented`: code and focused tests exist; real-artifact promotion may not.
- `partial`: at least one required modality, production caller, or evidence leg
  is absent.
- `unverified`: a plan status or skipped/fixture-gated test is the only evidence.
- Model-native Python is the quality oracle when adaptive_new has no equivalent
  exact Go oracle.
- Legacy `done` rows without `step.verify` are historical notes, not proof under
  the current gate contract.
- Cross-repo speed claims require the same artifact, inputs, seed, precision,
  output contract, warm/cold policy, and uncontended GPU admission.

## Structural Comparison

| Surface | adaptive_new | Overgo | Assessment |
| --- | ---: | ---: | --- |
| Production Go files | 643 | 624 | Similar file count |
| Production Go nonblank lines | 181,048 | 164,257 | Overgo 9.3% smaller |
| Go test files | 507 | 420 | adaptive_new has broader test volume |
| Go test nonblank lines | 128,957 | 86,544 | Overgo 32.9% smaller |
| `go/extmodel` production | 72 files / 84,029 lines | No extmodel package | adaptive_new concentration remains the main transfer tax |
| Repo-owned Python, excluding protected model/data roots | 76 files / 10,471 lines | 2 files / 367 lines | Overgo is effectively Go-native |
| Model-family compatibility inventory | Focused local artifact set | 135 experimental architectures | Overgo breadth is much larger; evidence depth is shallower |
| Compatibility claims | Capability budgets, RepoDB, findings, recipes | 34 `implemented` claims | Overgo needs evidence tiers and live validators |

Counts include generated Go. They measure surface, not capability quality.
Overgo's tokenizer total includes a large generated Unicode table.

### Authority shape

| Concern | adaptive_new | Overgo | Preferred owner |
| --- | --- | --- | --- |
| Model/artifact facts | RepoDB plus extmodel/config scans | RepoDB, recipes, GGUF/safetensors catalogs | Overgo RepoDB + compiled recipe |
| Runtime topology | Shared primitives plus large extmodel assembly | `recipe.Program`, model plans, neutral packages; image/audio/video projection modules and decoded tensor kinds are typed independently of placement | Overgo compiled program |
| CUDA | Go loaders plus family-heavy extmodel kernels | Manifested binaries, driver API, executor graph | Overgo executor + kernel manifest |
| Serving | Mature multimodal endpoints and evidence | llama-compatible server plus growing typed workflows | Overgo, after API parity matrix |
| Training | Broad objective/trainer set; uneven authority | Strong dense/hybrid primitives; thin production assembly | Overgo `TrainingRunPlan` + `TrainingProgram` |
| Evidence | RepoDB, capability budgets, media artifacts, fixtures | RepoDB, compatibility claims, plan fixtures | Overgo RepoDB; Git owns chronology |

## Inference and Generation Matrix

| Capability | adaptive_new Go / Python | Overgo | Verdict | Required promotion gate |
| --- | --- | --- | --- | --- |
| Dense causal text: Carbon, Qwen2.5, MiniCPM | Active CUDA inference, real artifacts, capability floors | Weight and golden identities are live-gated; exact-token parity for all three; MiniCPM 3.109 vs 3.400 ms/token | Parity; MiniCPM wall lead | Refresh Carbon/Qwen2.5 wall and all three matched peaks |
| Qwen3.5-4B hybrid text | Active CUDA; 16.76 ms/token retained baseline | 10.96 ms/token; same BF16 weights; about 2.2 GB more peak | Gated tradeoff: wall lead, memory loss | Retain speed while matching or beating adaptive peak |
| Qwen3.5-9B GGUF | adaptive_new cannot serve the bare Q8_0 GGUF | 11.6 ms/token, 9.3 GB recorded | Overgo-only capability | Re-run real artifact; register oracle-backed claim |
| Gemma E4B text | adaptive host spine; 341 ms/token comparison | 18.5 ms/token device; exact 3-case serving fixtures; 21.79 GiB self-baseline peak | Large speed lead | Matched peak-memory comparison; keep exact tokens |
| Gemma E4B image/video/audio | Adaptive image path passes the fingerprinted 2,520-patch oracle in 14.56 s focused-test wall. Its audio fixture runs wave -> mel -> tower in 1.03 s | Retained CUDA vision graph matches all 16 stages and 280x2,560 soft tokens; 0.210-0.215 s resident body. The exact 300-token adaptive image prompt with 280 projected tokens produces `23910` (`Pattern`) with 1.017 s language prefill. Resize matches 912x672/266 tokens. Video preserves frame-major order at 70 tokens/frame. Audio waveform -> log-mel -> 12-layer tower matches: frontend max error `2.72e-4`, worst stage relative `1.73e-4`, final `1.90e-5`, 0.129 s resident body. Its missing attention RoPE fact is now a content-addressed audio profile in recipe dependency slot 1 | Image projector-to-language parity and wall lead; audio frontend/tower parity; video ordering contract | Measure image/audio peaks; gate audio/video language output on real media |
| Gemma4 12B FP8 text | Active native FP8 CUDA | 24/24 exact tokens; 29.81 ms/token, 18.89 GiB versus Overgo BF16 baseline | Functional; cross-repo performance open | Same FP8 artifact and prompt against adaptive native FP8 |
| Gemma4 12B image | Active unified multimodal path | Real FP8-native model + BF16 device projector first-token oracle passes; 32.36 s focused-test wall | First-token parity | Multi-token corpus, request-only wall, peak |
| Gemma4 12B audio/video | Adaptive declares audio/video input coverage | Real audio projector-to-language top-ID oracle passes; 29.80 s focused-test wall. Video remains unproved | Audio first-token parity; video gap | Exact audio envelope + request-only wall/peak; real video case |
| RxBrain VQA | Native path; retained 18.4-23.1 s and 11.97 GB comparison | Exact answer; 9.5 s, 10.59 GB, 6.19 ms/token | Lead on both axes | Re-run nonfixture corpus cases; stop-contract and semantic scoring |
| Unlimited OCR | Active adaptive multimodal recipe | No promoted Overgo parity row found | Gap | Port typed OCR contract; exact text and layout metrics |
| TimesFM forecasting | Active host recipe | Fingerprinted real weights/golden; 684 vs 873 ms matched host wall | Exact parity; wall lead | Matched peak measurement |
| TabFM prediction | Active host classification/regression | Both 6.5 GB heads and shared golden fingerprinted; all four warm cases 0.65-0.81x reference wall | Exact parity; wall lead | Matched peak measurement |
| Needle seq2seq | Partial adaptive forward/backward | Fingerprinted real weights/JAX oracle; retained load+generate envelope <=250 vs 420 ms | Exact parity; wall lead | Matched peak measurement |
| Text embedding | No evidence-qualified encoder artifact in configured roots | Runtime and recipe exist; causal hidden-state substitution refused | Explicit refusal | Add a real encoder artifact, corpus, golden, wall, and peak |
| Text rerank | No evidence-qualified classifier-head artifact in configured roots | Runtime and recipe exist; causal LM substitution refused | Explicit refusal | Add a real reranker artifact, pair corpus, scores, wall, and peak |
| Pocket-TTS speech | Active adaptive host audio recipe; codec and training scaffolds | Backbone/codec/recipe rows recorded done | Unverified production audio quality | Real text-to-waveform: exact stages, audio metrics, wall, peak |
| Un-0 image/video | Active deterministic Go route | Exact fixture within `3.13e-7`; about 1.25 ms warm generation | Parity | Non-vacuous golden in normal gate; image and video outputs |
| SimpleDiffusion image | Host generation and real-checkpoint training | Forward max error `3.99e-6`; recorded 0.17 s vs adaptive 0.21 s | Host lead | Device forward/backward; real output quality and peak |
| Wan text-to-video | Native adaptive exact path; Python oracle | Fresh session+denoise 331.96 s / 7.875 GB; decode 38.31 s / 10.330 GB; exact G3/G4 bounded, BF16 G3 cosine 0.999899, frame-0 CUDA/host max error 8.04e-6 | 370.27 s staged wall beats retained 463.4 s Python wall | Fresh same-revision Python/adaptive run; semantic clip gate |
| Krea text-to-image | Matched 256 fixture: 25.430 s; exact u8 SHA `b257e244`; full 2048 record: 158.144 s, 33.47 GB device | 256: 13.804 s / 25.030 GiB, MAE 0.02401. 2048: 63.510 s / 32.732 GB, MAE 0.03910; production caller and phased residency gated | Wall/peak lead at both sizes; bounded image quality | Exact SHA remains numerically unstable; retain robust pixel oracle and RMSE advisory |
| SenseNova image/edit | Adaptive exact Go replay: 206.010 s, 19.50 GB; Python 366.6 s, 47.58 GiB | Fingerprinted native two-step core: boundary cosine >=0.99914, velocity >=0.99863, final state >=0.99832; reusable 42-layer body 5.64-5.69 s / 0.623 GiB versus adaptive 6.88 s | Core lead; production gap | Bind production decode/publication and session reuse; edit route still needs its absent source PNG |
| LiveEdit video edit | Adaptive complete experimental route; best 109.5 s, 20.61 GB; Python 60.1 s | No complete Overgo route found | Gap | Port shared video primitives; beat Python without temporal degradation |
| Server/API/session/cache | Mature adaptive-specific multimodal surface | Broad llama-compatible server, resumable caches, tools | Different strengths | Contract matrix: streaming, tools, embeddings, media, sessions, refusal |

## Training Matrix

| Training concern | adaptive_new | Overgo | Verdict |
| --- | --- | --- | --- |
| Training authority | Typed examples, objectives, admission, memory, progress, evidence; recipes not yet universal | `TrainingRunPlan` and `TrainingProgram` exist only in the design document | Overgo gap |
| Optimizer | Adaptive optimizer machinery and model-specific use; historical SGD paths remain | Matrix/vector/scalar groups use one compiled Muon path; sign, BF16-SGD and family-local production updates deleted | Muon-only production authority |
| Dense causal LM | Retained Carbon causal CUDA at `c72b6595d`: four 31-token DNA windows, LR `1.33179e-5`, Muon 0.95, loss `8.43767 -> 7.29179`, 6.816 s loop, 11.47 GiB admitted peak | Same ordered windows and settings: loss `8.437673 -> 7.293611`, 5.05-5.13 s loop, 4.620 GiB measured peak | Matched Carbon wall/peak lead; Qwen bias path open |
| Production dense command | Multiple adaptive entry points, uneven recipe authority | `cmd/train -freeze-lexical` selects the no-fallback resident contract; full-parameter mode remains explicit | Carbon route reachable; exact checkpoint contract open |
| Checkpoint/resume | Fine-tune evidence and persistence exist, uneven by trainer | Dense exact-resume tests exist; production save overwrites files and omits optimizer/RNG/data cursor | Test capability only |
| Qwen3.5 hybrid | Inference active; training unsupported in adaptive inventory | Host/device hybrid layer VJPs and synthetic resident stack train | Overgo primitive lead; real-model gap |
| Gemma E4B | Adaptive declares streamed CUDA LM scope and supplies real semantics/oracles | Host E4B VJPs plus device primitives; no real E4B end-to-end training | Open; current done status is false closure |
| Gemma4 12B | Streamed CUDA LM scope | Shared backward prerequisites; no reported real 12B training | Gap |
| Forecast/latent/FNS | CUDA objectives for LM, FNS, latent L2/sequence, forecast | TimesFM/Un-0 ports; no single compiled production program | Breadth gap |
| OCR | Host objective; CUDA train/eval absent | No promoted training path | Both incomplete |
| SimpleDiffusion/UViT | Real-checkpoint OT-flow trainer and revive experiments; device path incomplete | Shared Muon plan covers the real artifact; tiny Muon loss `0.026695 -> 0.002327`; real device update open | Optimizer authority fixed; performance parity incomplete |
| Pocket-TTS | Latent bridge, flow-net, backbone paths | Port row says scoped, but no current production proof | Unverified |
| Controller model | Strategic adaptive plan; no promoted controller | Overgo training design targets it | Not started |
| Tier-1 memory scaling | Adaptive streamed/checkpointed components | Resident dense session and scratch pool exist; no production E4B run | Partial |

### Current Overgo training evidence

- Dense resident: 12 steps; recorded loss difference `1.545e-5`, final-weight
  difference `3.613e-4`; loss `3.48 -> 2.17`.
- Dense resident scale fixture (12 layers, hidden 256, sequence 64): two pooled
  CUDA Muon steps stay below 2 s total; live allocation peak 120.1 MiB versus
  143.5 MiB unpooled. This is an Overgo ratchet, not adaptive parity.
- Carbon-500M, 64 tokens, four frozen-lexical Muon steps: loss
  `4.124540 -> 2.630955`, 1.476 s/step, 4.714 GiB peak. Ratchets are
  1.8 s/step and 6 GiB. Adaptive retained 2.018 s/step and later measured a
  15.06 GB Carbon training peak. A Git-history replay recovered adaptive's
  causal protocol at `c72b6595d`; current adaptive master refuses that gate at
  admission.
- Matched Carbon causal: identical artifact, four ordered DNA windows, LR and
  Muon setting. Overgo loss `8.437673 -> 7.293611`, resident loop 5.05-5.13 s,
  total public-call wall 5.89-6.55 s, peak 4.620 GiB. Adaptive loss
  `8.43767 -> 7.29179`, loop 6.816 s, admitted peak 11.47 GiB.
- Hybrid resident synthetic stack: loss `15.92 -> 2.56`; host/device trajectory
  difference `2.856e-6`.
- Synthetic dense medium benchmark: host `85.6 s/step`; device full
  `1.89 s/step`; `45.4x`. This is an internal backend comparison, not
  adaptive_new parity and not a production-model result.
- Device layer backward, cached forward, loss/gradients, full trainer, and
  resident trainer focused tests pass on the reviewed machine.

These results promote matched Carbon frozen-lexical leadership. They do not
justify Qwen, E4B, 12B, controller, or system-wide training closure.

## Recorded Performance Scoreboard

| Workload | adaptive_new / Python | Overgo | Current verdict |
| --- | ---: | ---: | --- |
| Qwen3.5-4B decode | adaptive 16.76 ms/token | 10.96 ms/token; +about 2.2 GB peak | Speed win; memory loss |
| E4B decode | adaptive host 341 ms/token | 18.5 ms/token | About 18x speed win |
| E4B image tower | adaptive 14.56 s focused test | 1.65 s fingerprint+load+run; 0.215 s resident encode; exact sampled stages | 8.8x focused-test wall lead; peak open |
| E4B audio input | adaptive wave -> mel -> tower focused test 1.03 s | 0.129 s resident wave -> mel -> tower; fingerprinted frontend/stages | Functional parity; wall lead unclaimed until matched lifecycle and peak |
| RxBrain full VQA | adaptive 18.4-23.1 s / 11.97 GB | 9.5 s / 10.59 GB | About 2x wall win; lower peak |
| SimpleDiffusion host forward | adaptive median 0.21 s | median 0.17 s | 0.81x wall |
| MiniCPM decode | adaptive 294.6-295.8 token/s | 321.7-363.0 token/s | Overgo faster |
| Un-0 generation | 0.01 s test resolution | 0.01 s test resolution | Tie; resolution-limited |
| Wan full generation | Python baselines vary by retained report: 399.1 or 463.4 s; adaptive exact Go 795.1 s warm in the current media report | fresh 331.96 s session+denoise + 38.31 s decode = 370.27 s; stage peaks 7.875 / 10.330 GB | 20.1% wall lead vs retained 463.4 s Python; fresh matched reference rerun still open |
| Krea 256 generation | adaptive exact Go 25.430 s | 13.804 s / 25.030 GiB; MAE 0.02401, RMSE 0.05459 | 46% wall lead; lower peak; bounded quality lead |
| Krea 2048 generation | Python 97.5-135.4 s; adaptive exact Go 158.144 s / 33.47 GB | 63.510 s / 32.732 GB; MAE 0.03910 | 35% lead vs best Python; 60% vs adaptive; 2.2% lower peak |
| Carbon-500M causal training | adaptive `c72b6595d`: 6.816 s loop / 11.47 GiB admitted; loss 8.43767 -> 7.29179 | 5.05-5.13 s loop / 4.620 GiB measured; loss 8.437673 -> 7.293611 | 25% loop-wall lead; about 60% lower peak |
| SenseNova generation core | Adaptive matched reusable body 6.88 s | Overgo 5.64-5.69 s / 0.623 GiB, two native steps gated | 17% reusable-body wall lead; full image route open |
| LiveEdit full edit | Python 60.1 s / 23.50 GiB; adaptive best 109.5 s / 20.61 GB | No complete Overgo result | Open |
| Gemma4 12B FP8 | adaptive comparison not yet recorded in Overgo plan | 29.81 ms/token / 18.89 GiB | Cross-repo open |

Wan's Overgo row is fresh on the reviewed machine. Python and adaptive numbers
remain retained, cross-revision evidence until one harness reruns all three.
Krea's Overgo rows are fresh real-artifact runs. The 2048 gate compares the
hash-verified adaptive PNG with robust pixel MAE; RMSE remains advisory.

The text/structured matrix is now a v2 content-addressed snapshot. Its normal
gate hashes every referenced GGUF/safetensors file plus its corpus/golden, so a
path-compatible artifact substitution cannot retain a promotion. Verdicts are
typed: `lead` requires matched wall and peak, `wall-lead` cannot carry an
unmeasured peak, `tradeoff` requires opposite-axis results, and `refused`
cannot carry support evidence. Current rows intentionally contain no full
wall+peak lead: structured peak measurements remain open, Qwen3.5 is a measured
memory loss, and encoder/reranker artifacts are absent.

## Corrective Findings

### P0: gate skip detection is not effective for ordinary Go skips

`cmd/gate` scans test output for `UNAVAILABLE` and `parity NOT verified`, but
invokes `go test` without `-v`. A skipped test in a successful package normally
prints only `ok`; its skip reason is hidden. Add verbose JSON/event parsing and
prove the commit gate refuses a real skipped capability test. `cmd/plan` has the
same general risk when a verify command omits verbose output.

### P0: root cleanup currently breaks live references

The working tree deletes `PORT_PLAN.md`, `IMPLEMENTATION_STATUS.md`, and other
root documents. `README.md`, `docs/IMPLEMENTATION_LOG.md`, `kernels/README.md`,
and `cmd/release` still reference the deleted files. Complete caller migration
and release-contract cleanup in the same slice; do not restore compatibility
stubs.

### P0: production training leadership is Carbon-only

Carbon now has a reachable frozen-lexical CUDA contract and matched wall/peak
lead. Repeat for Qwen, hybrid, and E4B; add held-out promotion and exact
checkpoint/resume evidence before broader claims.

### P0: E4B training is falsely closed

`rung13-e4b/training-leg` says the remaining work is real device training, but
the item and step are `done`. Reopen against the actual Gemma E4B artifact.
Synthetic Qwen-style hybrid training is not a substitute for AltUp, PLE,
Laurel, SharedKV, and mixed-window execution together.

### P1: production checkpoint contract is unsafe and incomplete

The command creates the output directory and writes `model.safetensors`
directly. It may overwrite an existing artifact, is not atomic, and does not
persist Muon momentum, RNG, dataset cursor, compiled plan identity, or RepoDB
lineage. Exact resume tests exist below the command but are not reachable.

### P1: Muon-only production authority needs a scanner

Dense and SimpleDiffusion training now express matrix/vector/scalar groups
through the shared compiled Muon path; sign, BF16-SGD and the image-local SGD
loop are deleted. Add a production-code scanner so new local weight-update loops
cannot bypass the optimizer package.

### P1: plan and training docs contain stale completion prose

`docs/plan.json` carries 38 done items and many legacy steps without runnable
verification. `docs/training_plan.md` still describes device backward as
host-only and names types that do not exist. Git owns chronology. Keep only
open executable work plus current contracts.

### P1: compatibility breadth outruns evidence depth

All 135 model entries are experimental; only Gemma3, Qwen3, Qwen3.5, and T5
encoder name validated fixtures. Add evidence tiers, source commits, artifact
hashes, failable commands, and expiry/reverification rules. An architecture
parser is not a promoted model capability.

### P1: media breadth remains SenseNova and LiveEdit

Krea and Wan now have production performance gates. SenseNova's neutral
generation core is native-gated and faster than adaptive; production decode,
publication, and request-session binding remain. LiveEdit is still the largest
video gap; edit-input evidence stays explicitly blocked when its source artifact
is absent.

## Multimodal Completion Contract

Multimodality is derived from active recipe and objective declarations. It is
not satisfied by one image smoke test or by separate family-specific code.

| Dimension | Inference requirement | Training requirement |
| --- | --- | --- |
| Signature | Ordered typed inputs and outputs: text, image, audio, video, time series, table | Ordered input/target modalities plus objective identity |
| Front end | Real tokenizer, image processor, audio frontend, video sampler, table/series adapter | Same processor identity and differentiable/frozen boundary recorded |
| Model boundary | Projector, merger, positional/mask semantics, modality routing | Gradients cover every declared trainable projector, connector, trunk, and head |
| Output | Tokens, waveform, image, clip, forecast, table result | Modality-native loss and held-out evaluation; scalar loss alone is insufficient |
| Composition | Single and combined inputs where declared; ordering and temporal structure preserved | Cross-modal pairs run only with adaptive evidence or an approved objective |
| Persistence | Recipe, artifact, processor/codec, and corpus identities recorded | Checkpoint includes Muon, RNG/augmentation, data cursor, processor/codec identities |
| Performance | Same artifact/input; wall, peak, quality | Same initial state/data/update contract; step wall, peak, trajectory, held-out quality |
| Absence | Explicit unsupported/refused row | Explicit non-trainable/refused row; no synthetic substitute |

Applicable inference coverage includes text, image, audio, and video input;
text, speech, image, and video output; forecast and tabular results; and combined
inputs declared by Qwen, Gemma, RxBrain, OCR, and edit recipes. Applicable
training coverage is narrower: only adaptive-backed or explicitly approved
objectives advance. Krea, Wan, LiveEdit, RxBrain, or any other path recorded as
serving-only in the parity snapshot does not acquire a training lane merely
because inference exists.

## Promotion Contract

For each adaptive capability, one machine-readable parity row owns:

1. Adaptive source commit and model-native Python source commit when used.
2. Artifact identity and immutable input/evaluation corpus identity.
3. Ordered input/output modality signature, typed capability, and output contract.
4. Adaptive Go result, Python oracle result, and Overgo result.
5. Exact hashes/tokens/tensors or an approved quality metric with threshold
   derived from oracle repeatability.
6. Cold wall, warm wall, peak device memory, peak host memory, and contention
   facts.
7. Production entry point and active recipe identity.
8. Gate command that fails on skip, missing fixture, stale artifact, regression,
   or absent production caller.

Promotion states:

| State | Meaning |
| --- | --- |
| `absent` | No Overgo capability |
| `reference` | Host implementation matches an oracle |
| `device` | CUDA implementation matches reference |
| `reachable` | Active recipe and production caller execute it |
| `parity` | Behavior/quality matches adaptive or Python |
| `lead` | Parity holds and Overgo is no worse on wall and peak memory |
| `promoted` | Lead repeats across the declared corpus and gate remains live |

No aggregate percentage may hide a missing modality or an unavailable test.

## Execution Program

The authoritative steps live in `docs/plan.json`. Order:

1. Repair non-vacuous gate evidence.
2. Finish root-document authority cleanup; remove stale release callers.
3. Compact the live plan to open work; Git and this report retain history.
4. Wire and exclusively retain the resident production trainer.
5. Build a Go-owned neutral adaptive parity snapshot/import and scoreboard.
6. Derive inference modality signatures from recipes; close text, structured,
   combined multimodal input, speech, image, video, then API/session behavior.
7. Compile `TrainingRunPlan` and `TrainingProgram`; bind immutable RepoDB facts.
8. Make Muon the only optimizer family; delete SGD/sign compatibility paths.
9. Make checkpoint/resume atomic, exact, lineage-bound, and production-reachable.
10. Prove dense real-artifact training parity; then real Qwen hybrid and E4B,
    exercising every adaptive-declared trainable modality.
11. Compile the multimodal training matrix; port remaining adaptive objective
    families through shared programs and explicitly refuse absent objectives.
12. Train and promote the small workflow controller; scale only after evidence.

## Definition of Exceeds adaptive_new

Overgo exceeds adaptive_new only when:

- Every promoted adaptive inference and training capability has a reachable
  Overgo recipe and non-vacuous real-artifact gate.
- Every applicable ordered modality combination is exercised end to end;
  missing or non-trainable combinations are explicit refusals.
- Output is exact where the reference is deterministic; otherwise quality is
  no worse across a declared corpus and repeatability envelope.
- Median wall and peak memory are both no worse on the same hardware and
  protocol. A tradeoff requires an explicit product decision; it is not a lead.
- Training comparisons use the same initial weights, ordered data, objective,
  update semantics, precision contract, and checkpoint boundary; loss alone is
  insufficient.
- Model-specific code does not grow when a recipe row or shared processing
  primitive can express the difference.
- Displaced adaptive-style orchestration and compatibility paths are deleted.
- The plan contains open work only; Git and RepoDB retain evidence and history.

Current state: inference leadership is real but incomplete. Training leadership
is not yet established.
