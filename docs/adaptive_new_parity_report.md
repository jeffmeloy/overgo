# adaptive_new Parity and Performance Report

Validation base: Overgo `dfcebd9` plus the final-cross training slice;
adaptive_new `214950b3b`; reviewed 2026-08-14. adaptive_new working-tree changes
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
- Corpus-to-model construction now matches the pinned adaptive_new scratch
  oracle. The resident three-step program leads the matched reference on wall
  and runtime-owned peak memory; production checkpoint/publication remains open.
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
| Training | Broad objective/trainer set plus a working corpus-to-new-model path; uneven authority | Compiled `TrainingRunPlan`, corpus-derived `ScratchConstruction`, shared tensor VJP, resident CUDA and Muon; objective/model breadth remains narrower | Overgo compiled authority + shared runtime |
| Evidence | RepoDB, capability budgets, media artifacts, fixtures | RepoDB, compatibility claims, plan fixtures | Overgo RepoDB; Git owns chronology |

## Inference and Generation Matrix

| Capability | adaptive_new Go / Python | Overgo | Verdict | Required promotion gate |
| --- | --- | --- | --- | --- |
| Dense causal text: Carbon, Qwen2.5, MiniCPM | Active CUDA inference, real artifacts, capability floors | Weight and golden identities are live-gated; exact-token parity for all three; MiniCPM 3.109 vs 3.400 ms/token | Parity; MiniCPM wall lead | Refresh Carbon/Qwen2.5 wall and all three matched peaks |
| Qwen3.5-4B hybrid text | Active CUDA; 16.76 ms/token retained baseline | 10.96 ms/token; same BF16 weights; about 2.2 GB more peak | Gated tradeoff: wall lead, memory loss | Retain speed while matching or beating adaptive peak |
| Qwen3.5-4B image/video | Python processor and Go path own pinned image plus 16-frame video goldens | Direct HF projector export; CUDA image/video processor and projector probes pass; exact prompt IDs, MRoPE axes, and first language tokens pass for both routes | Real image/video input parity | Add combined-media corpus, request-only wall, and peak |
| Qwen3.5-9B GGUF | adaptive_new cannot serve the bare Q8_0 GGUF | 11.6 ms/token, 9.3 GB recorded | Overgo-only capability | Re-run real artifact; register oracle-backed claim |
| Gemma E4B text | adaptive host spine; 341 ms/token comparison | 18.5 ms/token device; exact 3-case serving fixtures; 21.79 GiB self-baseline peak | Large speed lead | Matched peak-memory comparison; keep exact tokens |
| Gemma E4B image/video/audio | Adaptive image path passes the fingerprinted 2,520-patch oracle in 14.56 s focused-test wall. Its audio fixture runs wave -> mel -> tower in 1.03 s | Retained CUDA vision graph matches all 16 stages and 280x2,560 soft tokens; 0.210-0.215 s resident body. The exact 300-token adaptive image prompt with 280 projected tokens produces `23910` (`Pattern`) with 1.017 s language prefill. Resize matches 912x672/266 tokens. Video preserves frame-major order at 70 tokens/frame. Audio waveform -> log-mel -> 12-layer tower matches: frontend max error `2.72e-4`, worst stage relative `1.73e-4`, final `1.90e-5`, 0.129 s resident body. Its missing attention RoPE fact is now a content-addressed audio profile in recipe dependency slot 1 | Image projector-to-language parity and wall lead; audio frontend/tower parity; video ordering contract | Measure image/audio peaks; gate audio/video language output on real media |
| Gemma4 12B FP8 text | Active native FP8 CUDA | 24/24 exact tokens; 29.81 ms/token, 18.89 GiB versus Overgo BF16 baseline | Functional; cross-repo performance open | Same FP8 artifact and prompt against adaptive native FP8 |
| Gemma4 12B image | Active unified multimodal path | Real FP8-native model + BF16 CUDA projector exact first-token oracle passes inside the promoted matrix | First-token parity | Multi-token corpus, request-only wall, peak |
| Gemma4 12B audio/video | Adaptive declares audio/video input coverage | Real audio projector-to-language top-ID oracle passes. Mixed image→audio and audio→image prompts retain exact modality chunks, image-only attention spans, and order-sensitive language tokens 107/108. A two-frame real-image video preserves frame order and block masks through language generation; no adaptive video output oracle exists | Audio first-token parity; mixed-media and video execution contracts | Mint a native video output oracle; exact audio envelope; request-only wall/peak |
| RxBrain VQA | Native path; retained 18.4-23.1 s and 11.97 GB comparison | Real processor and image execute through compiled recipe, publish a bound verifier gate/run, activate, then replay bit-exactly; two gates measured 10.211-10.273 s verify and 9.529-9.982 s active, 6.06-6.28 ms/token, 3.940 GiB coarse device peak | Exact golden prefix and coherent EOS answer; lead on both recorded axes | Re-run nonfixture corpus cases; semantic scoring |
| Unlimited OCR | Native BF16 image/text golden: 277 prompt IDs, 811 generated IDs, 29 OCR rows; `crop_mode=false`; 35-gram/128-window repetition policy | Shared DeepSeek OCR CUDA projector plus shared DeepSeek2/MoE inference and shared sliding-window no-repeat sampler: exact prompt IDs, 273 finite projected tokens, exact 200-token prefix, terminal EOG, and all 29 rows; every coordinate is within one and every row text is within one edit. Fresh projection 10.02 s; full generation 4m22.482s | Real production OCR parity | Resolve the token-200 BF16 numerical swap for exact full-sequence identity; measure matched full-output peak |
| TimesFM forecasting | Active host recipe | Fingerprinted real weights/golden; 684 vs 873 ms matched host wall. A held-out Supernova light curve now runs through the shared typed stream: 32 context samples, 31 targets, point MAE 72.908 versus 127.428 for persistence | Exact parity; wall lead; real-data baseline win | Matched peak measurement; native forecast on the same light curve |
| TabFM prediction | Active host classification/regression | Both 6.5 GB heads and shared golden fingerprinted; all four warm cases 0.65-0.81x reference wall | Exact parity; wall lead | Matched peak measurement |
| Needle seq2seq | Inference only; no adaptive Needle training owner | Fingerprinted real weights/JAX oracle; retained load+generate envelope <=250 vs 420 ms. The typed text recipe uses the shared SentencePiece tokenizer and emits grounded JSON for the model-card weather tool-call example. Candidate execution published a recipe-bound gate/run pair; activation and active replay passed at the experimental tier. Native BF16 matrix ownership preserves the prior parity metrics and grounded generation while three matched cold processes measure 66.758-67.363 MiB versus adaptive_new's 120.918-121.328 MiB. A real GSM8K record crosses the shared materializer, typed paired-text processor, deterministic stream, tokenizer, shifted decoder targets, and shared cross-entropy. Compiled training now owns the decoder final norm and complete last cross-attention block: input/Q/K norms, raw residual gate, and Q/K/V/output projections. The shared attention VJP is finite-difference gated. Real-record analytic gradients for all six newly bound projection and norm groups agree with parameter finite differences; the BF16 V probe differs by 11.8%, within its quantization-aware bound. Three device-Muon updates change 18,849/262,144 Q, 13,626/131,072 K, and 11,000/131,072 V words. Training loss falls from 6.541863 to 6.415488 and a separate held-out record from 8.629535 to 8.541487. The combined residual and cross-query gradient reaches the final causal self-attention: Q/K/V core gradient norms are 0.134667/0.042098/0.001092, and its raw-gate analytic gradient 0.142948 matches the 0.143119 parameter finite difference. Exact JAX numeric parity, 65,536-value incremental/full identity, and the grounded tool-call generation remain unchanged | Exact numeric parity; real text generation; active recipe; wall and process-memory lead; real-data complete-final-cross device training with analytic-gradient and held-out evidence; final self-attention core backward | Bind the final self-attention gate, output, Q/K/V, and norm gradients to Muon, then propagate through the remaining decoder and encoder; evaluate a larger fixed GSM8K split; do not claim adaptive training parity because adaptive_new has no Needle trainer |
| Text embedding | No evidence-qualified encoder artifact in configured roots | Runtime and recipe exist; causal hidden-state substitution refused | Explicit refusal | Add a real encoder artifact, corpus, golden, wall, and peak |
| Text rerank | No evidence-qualified classifier-head artifact in configured roots | Runtime and recipe exist; causal LM substitution refused | Explicit refusal | Add a real reranker artifact, pair corpus, scores, wall, and peak |
| Pocket-TTS speech | Active adaptive host audio recipe and real-model generation fixture | Compiled production recipe; every generated latent/EOS value is bounded against the adaptive oracle; PCM max error `3.38e-4`, RMS `0.089438`, encoded WAV `24 kHz` mono; warm matched synthesis `0.48-0.53 s`; peak heap `0.540-0.543 GiB` | Production output parity; warm-wall lead versus adaptive's retained `2.0 s` optimized route; matched process peak still open | Preserve exact stage/audio gates; measure both implementations under one cold/process-memory harness |
| Un-0 image/video | Active deterministic Go route | Exact fixture within `3.13e-7`; about 1.25 ms warm generation | Parity | Non-vacuous golden in normal gate; image and video outputs |
| SimpleDiffusion image | Host generation and real-checkpoint training | Forward max error `3.99e-6`; recorded 0.17 s vs adaptive 0.21 s | Host lead | Device forward/backward; real output quality and peak |
| Wan text-to-video | Native adaptive exact path; Python oracle | Fresh session+denoise 331.96 s / 7.875 GB; decode 38.31 s / 10.330 GB; exact G3/G4 bounded, BF16 G3 cosine 0.999899, frame-0 CUDA/host max error 8.04e-6 | 370.27 s staged wall beats retained 463.4 s Python wall | Fresh same-revision Python/adaptive run; semantic clip gate |
| Krea text-to-image | Matched 256 fixture: 25.430 s; exact u8 SHA `b257e244`; full 2048 record: 158.144 s, 33.47 GB device | 256: 13.804 s / 25.030 GiB, MAE 0.02401. 2048: 63.510 s / 32.732 GB, MAE 0.03910; production caller and phased residency gated | Wall/peak lead at both sizes; bounded image quality | Exact SHA remains numerically unstable; retain robust pixel oracle and RMSE advisory |
| SenseNova image/edit | Adaptive exact Go replay: 206.010 s, 19.50 GB; Python 366.6 s, 47.58 GiB | Compiled routed-image recipe owns tokenizer/prompt, retained prefix/body, shifted integration, and planar PNG publication. Pinned 256px/2-step case: boundary cosine >=0.99914, velocity >=0.99863, final state >=0.99832, exact PNG `d439b8ce...`, 13.56 s total, 0.708 GiB peak | Production text-to-image route gated; warm-body lead | Run the full-size 50-step quality case; edit route still needs its absent source PNG |
| LiveEdit video edit | Adaptive complete experimental route; best 109.5 s, 20.61 GB; Python 60.1 s | The shared source-codec boundary derives causal `[1,stride,...]` frame chunks, latent geometry, channel-major frame spans, and recipe-owned latent normalization. The real 507 MB Wan VAE compiles into 17 ordered encoder operations with 86 owned tensors, 204.456 MiB of weights, 3 input channels, 32 moment channels, 16 latent channels, and stride `[4,8,8]`. Device execution, denoise integration, and decode remain incomplete | Real source graph and boundary compiled; execution gap | Execute the compiled causal VAE encoder; retain source latent through denoise; beat Python without temporal degradation |
| Server/API/session/cache | Mature adaptive-specific multimodal surface | Broad llama-compatible server, resumable caches, tools | Different strengths | Contract matrix: streaming, tools, embeddings, media, sessions, refusal |

## Training Matrix

| Training concern | adaptive_new | Overgo | Verdict |
| --- | --- | --- | --- |
| Training authority | Typed examples, objectives, admission, memory, progress, evidence; recipes not yet universal | `TrainingRunPlan` and `TrainingProgram` compile immutable construction, dataset, operator and Muon authority; production checkpoint/promotion breadth remains open | Core authority ported; breadth gap |
| Model from scratch | `RunFromProviderRich` derives split, rune vocabulary, topology, initialization, batching, causal training and metrics from documents | Pinned source/config/oracle parity; direct flat initialization; shared tensor forward/VJP; resident CUDA weights, gradients, momentum, compiled graphs and pooled scratch | Matched three-step wall/peak lead; artifact publication and exact resume open |
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
| Pocket-TTS | Latent bridge, flow-net, backbone paths | Inference recipe and reference waveform are production-proven; training gradients exist but no matched real training trajectory | Inference promoted; training open |
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
- Needle 26M, first real GSM8K training record, three compiled device-Muon
  steps over the decoder final norm and complete last cross-attention block:
  train loss `6.541863 -> 6.415488`; separate held-out record
  `8.629535 -> 8.541487`. Q/K/V BF16 words changed
  `18,849/13,626/11,000`; projection and normalization gradients match direct
  parameter finite differences. adaptive_new has no Needle training owner, so
  this is Overgo capability evidence rather than cross-repository parity.
- Needle's combined final-cross residual and query-path gradient reaches the
  last causal self-attention block. Real GSM8K Q/K/V core gradient norms are
  `0.134667/0.042098/0.001092`; its raw-gate analytic gradient `0.142948`
  matches the direct `0.143119` parameter finite difference. The final self
  parameters are not yet in the Muon plan.
- Synthetic dense medium benchmark: host `85.6 s/step`; device full
  `1.89 s/step`; `45.4x`. This is an internal backend comparison, not
  adaptive_new parity and not a production-model result.
- Device layer backward, cached forward, loss/gradients, full trainer, and
  resident trainer focused tests pass on the reviewed machine.
- Adaptive scratch profile, seed 7, batch 1, three steps: adaptive_new pinned
  runs span 221.4-231.8 ms and 3.13-3.18 MB peak Go heap. Overgo reports
  process launch 7.7-303.0 ms, driver/context 37.7-150.6 ms, model initialization
  0.5-2.6 ms, PTX/program preparation 77.5-85.4 ms, first step 38.4-44.5 ms,
  warm steps 9.1-11.6 ms, 58.0-65.4 ms for all three steps, and 1.15-1.17 MB
  combined host plus runtime-owned device peak. Loss/validation maximum delta is
  `1.738e-3`; final weights, cleared gradients, and Muon momentum are also gated.

These results promote matched Carbon frozen-lexical leadership. They do not
justify Qwen, E4B, 12B, controller, or system-wide training closure.

## Recorded Performance Scoreboard

| Workload | adaptive_new / Python | Overgo | Current verdict |
| --- | ---: | ---: | --- |
| Qwen3.5-4B decode | adaptive 16.76 ms/token | 10.96 ms/token; +about 2.2 GB peak | Speed win; memory loss |
| E4B decode | adaptive host 341 ms/token | 18.5 ms/token | About 18x speed win |
| E4B image tower | adaptive 14.56 s focused test | 1.65 s fingerprint+load+run; 0.215 s resident encode; exact sampled stages | 8.8x focused-test wall lead; peak open |
| E4B audio input | adaptive wave -> mel -> tower focused test 1.03 s | 0.129 s resident wave -> mel -> tower; fingerprinted frontend/stages | Functional parity; wall lead unclaimed until matched lifecycle and peak |
| RxBrain full VQA | adaptive 18.4-23.1 s / 11.97 GB | 10.211-10.273 s verifier / 9.529-9.982 s active replay; 3.940 GiB coarse device peak | 1.8-2.4x wall lead; lower recorded peak; exact deterministic answer |
| Unlimited OCR image/text | Native BF16 811-token/29-row golden | 10.02 s projection + 4m22.482s full generation; exact 200-token prefix; complete 29-row output within one coordinate/text edit; 273 image tokens | Functional full-output parity; exact sequence and matched reference wall/peak open |
| SimpleDiffusion host forward | adaptive median 0.21 s | median 0.17 s | 0.81x wall |
| MiniCPM decode | adaptive 294.6-295.8 token/s | 321.7-363.0 token/s | Overgo faster |
| Un-0 generation | 0.01 s test resolution | 0.01 s test resolution | Tie; resolution-limited |
| Wan full generation | Python baselines vary by retained report: 399.1 or 463.4 s; adaptive exact Go 795.1 s warm in the current media report | fresh 331.96 s session+denoise + 38.31 s decode = 370.27 s; stage peaks 7.875 / 10.330 GB | 20.1% wall lead vs retained 463.4 s Python; fresh matched reference rerun still open |
| Krea 256 generation | adaptive exact Go 25.430 s | 13.804 s / 25.030 GiB; MAE 0.02401, RMSE 0.05459 | 46% wall lead; lower peak; bounded quality lead |
| Krea 2048 generation | Python 97.5-135.4 s; adaptive exact Go 158.144 s / 33.47 GB | 63.510 s / 32.732 GB; MAE 0.03910 | 35% lead vs best Python; 60% vs adaptive; 2.2% lower peak |
| Carbon-500M causal training | adaptive `c72b6595d`: 6.816 s loop / 11.47 GiB admitted; loss 8.43767 -> 7.29179 | 5.05-5.13 s loop / 4.620 GiB measured; loss 8.437673 -> 7.293611 | 25% loop-wall lead; about 60% lower peak |
| Corpus-derived scratch causal training | adaptive pinned run: 221.4-231.8 ms / 3.13-3.18 MB peak heap | 58.0-65.4 ms matched steps / 1.15-1.17 MB combined runtime peak; six lifecycle phases and complete state gated | At least 3.38x step-wall lead; at least 62% lower peak; trajectory delta `1.738e-3` |
| SenseNova generation core | Adaptive matched reusable body 6.88 s | Overgo retained session: fresh cold body 8.117 s, warm body 4.048 s / 0.710 GiB peak; cold <=8.6 s and warm <=6.2 s gated twice | 41.1% warm-body lead at the fresh measurement; cold is slower than adaptive and is not claimed as leadership; full image route open |
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
7. Publish scratch initialization and training results as complete RepoDB
   artifacts; bind exact resume and production command reachability.
8. Make checkpoint/resume atomic, exact, lineage-bound, and production-reachable.
9. Prove dense real-artifact training parity; then real Qwen hybrid and E4B,
    exercising every adaptive-declared trainable modality.
10. Compile the multimodal training matrix; port remaining adaptive objective
    families through shared programs and explicitly refuse absent objectives.
11. Train and promote the scratch workflow controller; keep recursive candidate
    proposal inside the model but data admission and promotion external.

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
is established for matched Carbon and the pinned scratch profile, not yet for
the broader adaptive_new objective/model matrix.
