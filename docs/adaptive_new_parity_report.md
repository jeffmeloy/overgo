# adaptive_new Parity and Performance Report

<!-- overgo-document: historical-ranking -->
<!-- overgo-current-work: docs/plan.json -->

> Evidence snapshot reviewed 2026-08-15. Its priorities are not current work
> authority; use `docs/plan.json` and generated compatibility evidence for
> current work and status.

Validation base: current Overgo master plus the `dense-real-models` slice;
adaptive_new references are named per measurement. Uncommitted changes in
either repository are not reference evidence.

Scope: adaptive_new Go runtime and training code, model-native Python oracles,
Overgo host/CUDA runtime, recipes, training paths, compatibility claims, and
current automation gates. Model, dataset, and checkpoint directories were
read-only.

## Verdict

Overgo is the better destination architecture. It beats adaptive_new on several
real inference and dense-training paths. It does not yet match adaptive_new's
training or media breadth.

- Runtime lead: compiled programs, neutral capability packages, cgo-free CUDA,
  GGUF breadth, native quantization, retained device state, smaller Python
  surface.
- Proven wins: Qwen3.5-4B decode, Gemma E4B text decode, RxBrain VQA,
  SimpleDiffusion host forward, MiniCPM decode, Wan exact-quality generation,
  Krea 256/2048 generation, and SenseNova 1536x2720 generation on the recorded
  protocols.
- Training lead remains adaptive_new: more objective types, more artifact
  trainers, more promotion/evidence machinery, and real media-training paths.
- E4B now has real, exact-resume layer-0 adapter training over text, image, and
  audio representations. Complete 42-layer language/tower training remains open.
- Production dense training beats the retained Carbon and Qwen2.5-0.5B
  protocols; hybrid and E4B selected-layer programs now run real artifacts,
  while complete-stack training and broad checkpoint adoption remain open.
- Corpus-to-model construction now matches the pinned adaptive_new scratch
  oracle. The resident three-step program leads the matched reference on wall
  and runtime-owned peak memory; the shared exact checkpoint is now published
  and consumed by host and resident CUDA dense training.
- The scratch-controller contract now uses an immutable Git/RepoDB corpus,
  external holdout, three resident CUDA seeds, common action/modality scoring,
  and an external promotion decision with a rollback target. This is lifecycle
  evidence on eight actions, not production controller quality.
- Recursive trials now compile four bounded proposal classes: corpus, recipe,
  evaluator, and component composition. Admission requires separate authority
  and development/promotion splits. Promotion requires an independently bound
  evaluator, run, evaluation, decider, rollback target, and complete decision
  aggregate with acyclic child construction lineage. No recursive improvement
  outcome has been demonstrated yet.
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
| Serving | Mature adaptive-specific multimodal endpoints and evidence | Native llama.cpp-style, OpenAI, and Anthropic protocols behind one handler and one compatibility matrix | Overgo shared server contracts |
| Training | Broad objective/trainer set plus a working corpus-to-new-model path; uneven authority | Compiled `TrainingRunPlan`, corpus-derived `ScratchConstruction`, shared tensor VJP, resident CUDA and Muon; objective/model breadth remains narrower | Overgo compiled authority + shared runtime |
| Evidence | RepoDB, capability budgets, media artifacts, fixtures | RepoDB, compatibility claims, plan fixtures | Overgo RepoDB; Git owns chronology |

## Inference and Generation Matrix

| Capability | adaptive_new Go / Python | Overgo | Verdict | Required promotion gate |
| --- | --- | --- | --- | --- |
| Dense causal text: Carbon, Qwen2.5, MiniCPM | Active CUDA inference, real artifacts, capability floors | Weight and golden identities are live-gated; exact-token parity for all three; MiniCPM 3.109 vs 3.400 ms/token | Parity; MiniCPM wall lead | Refresh Carbon/Qwen2.5 wall and all three matched peaks |
| Fractale-350M ThoughtBank | Shipped PyTorch checkpoint and tokenizer; retained reference completion for `def fibonacci(n):` | PyTorch tensor facts, tokenizer, checkpoint loader, incremental decode, and a retained host session execute through one typed generation recipe. The real 1.5 GB artifact produces the exact eight-token completion, publishes recipe-bound evidence, activates, and resolves the active program. Focused end-to-end test wall is 4.46 s, including inventory hashing and model load | Exact real-output parity; active generic recipe | Separate load, cold generation, warm generation, and peak memory; compare with the adaptive route |
| Qwen3.5-4B hybrid text | Active CUDA; 16.76 ms/token retained baseline | 10.96 ms/token; same BF16 weights; about 2.2 GB more peak | Gated tradeoff: wall lead, memory loss | Retain speed while matching or beating adaptive peak |
| Qwen3.5-4B image/video | Python processor and Go path own pinned image plus 16-frame video goldens | Direct HF projector export; CUDA image/video processor and projector probes pass; exact prompt IDs, MRoPE axes, and first language tokens pass for both routes | Real image/video input parity | Add combined-media corpus, request-only wall, and peak |
| Qwen3.5-9B GGUF | adaptive_new cannot serve the bare Q8_0 GGUF | 11.6 ms/token, 9.3 GB recorded | Overgo-only capability | Re-run real artifact; register oracle-backed claim |
| Gemma E4B text | adaptive host spine; 341 ms/token comparison | 18.5 ms/token device; exact 3-case serving fixtures; 21.79 GiB self-baseline peak | Large speed lead | Matched peak-memory comparison; keep exact tokens |
| Gemma E4B image/video/audio | Adaptive image path passes the fingerprinted 2,520-patch oracle in 14.56 s focused-test wall. Its audio fixture runs wave -> mel -> tower in 1.03 s | Retained CUDA vision graph matches all 16 stages and 280x2,560 soft tokens; 0.210-0.215 s resident body. The exact 300-token adaptive image prompt with 280 projected tokens produces `23910` (`Pattern`) with 1.017 s language prefill. Resize matches 912x672/266 tokens. Video preserves frame-major order at 70 tokens/frame. Audio waveform -> log-mel -> 12-layer tower matches: frontend max error `2.72e-4`, worst stage relative `1.73e-4`, final `1.90e-5`, 0.129 s resident body. Its missing attention RoPE fact is now a content-addressed audio profile in recipe dependency slot 1 | Image projector-to-language parity and wall lead; audio frontend/tower parity; video ordering contract | Measure image/audio peaks; gate audio/video language output on real media |
| Gemma4 12B FP8 text | Native FP8 CUDA; pinned case-0 generation 488.886-497.603 ms and 14.363 GiB runtime-owned peak | Same source-weight conversion and exact prompt IDs produce `[50429, 106]`; two gates measure 231.979-248.041 ms generation and 14.353 GiB peak with a 32-token demand page | Exact-token parity; 49.7-53.0% request-wall lead and 10.4 MiB peak lead | Reduce model-open wall; extend the matched gate across all four golden prompts |
| Gemma4 12B image | Active unified multimodal path | Real FP8-native model + BF16 CUDA projector exact first-token oracle passes inside the promoted matrix | First-token parity | Multi-token corpus, request-only wall, peak |
| Gemma4 12B audio/video | Adaptive declares audio/video input coverage | Real audio projector-to-language top-ID oracle passes. Mixed image→audio and audio→image prompts retain exact modality chunks, image-only attention spans, and order-sensitive language tokens 107/108. A two-frame real-image video preserves frame order and block masks through language generation; no adaptive video output oracle exists | Audio first-token parity; mixed-media and video execution contracts | Mint a native video output oracle; exact audio envelope; request-only wall/peak |
| RxBrain VQA | Native path; retained 18.4-23.1 s and 11.97 GB comparison | Real processor and image execute through compiled recipe, publish a bound verifier gate/run, activate, then replay bit-exactly; two gates measured 10.211-10.273 s verify and 9.529-9.982 s active, 6.06-6.28 ms/token, 3.940 GiB coarse device peak | Exact golden prefix and coherent EOS answer; lead on both recorded axes | Re-run nonfixture corpus cases; semantic scoring |
| Unlimited OCR | Native BF16 image/text golden: 277 prompt IDs, 811 generated IDs, 29 OCR rows; `crop_mode=false`; 35-gram/128-window repetition policy | Shared DeepSeek OCR CUDA projector plus shared DeepSeek2/MoE inference and shared sliding-window no-repeat sampler: exact prompt IDs, 273 finite projected tokens, exact 200-token prefix, terminal EOG, and all 29 rows; every coordinate is within one and every row text is within one edit. Fresh projection 10.02 s; full generation 4m22.482s | Real production OCR parity | Resolve the token-200 BF16 numerical swap for exact full-sequence identity; measure matched full-output peak |
| TimesFM forecasting | Active host recipe | Fingerprinted real weights/golden; 684 vs 873 ms matched host wall. A held-out Supernova light curve now runs through the shared typed stream: 32 context samples, 31 targets, point MAE 72.908 versus 127.428 for persistence | Exact parity; wall lead; real-data baseline win | Matched peak measurement; native forecast on the same light curve |
| TabFM prediction | Active host classification/regression | Both 6.5 GB heads and shared golden fingerprinted; all four warm cases 0.65-0.81x reference wall | Exact parity; wall lead | Matched peak measurement |
| Needle seq2seq | Inference only; no adaptive Needle training owner | Fingerprinted real weights/JAX oracle; retained load+generate envelope <=250 vs 420 ms. The typed text recipe uses the shared SentencePiece tokenizer and emits grounded JSON for the model-card weather tool-call example. Candidate execution published a recipe-bound gate/run pair; activation and active replay passed at the experimental tier. Native BF16 matrix ownership preserves the prior parity metrics and grounded generation while three matched cold processes measure 66.758-67.363 MiB versus adaptive_new's 120.918-121.328 MiB. A real GSM8K record crosses the shared materializer, typed paired-text processor, deterministic stream, tokenizer, shifted decoder targets, and shared cross-entropy. Shared reverse traversal owns both final norms, every self/cross-attention block across all eight decoder layers, and every self-attention block across all 12 encoder layers. Decoder cross-attention K/V paths accumulate one encoder-memory gradient before the encoder VJP. One tied embedding parameter receives source-input, decoder-input, and output-projection gradients. All 227 compiled groups are trainable through the same Muon program. The tied BF16 analytic gradient `-0.0750179` matches the direct `-0.0750161` finite difference. Three updates lower one-record training loss from 6.541863 to 5.503946 and held-out loss from 8.629535 to 8.335628. A deterministic 4-train/4-held-out GSM8K split over two ordered epochs lowers mean train loss from 7.491162 to 5.274749 and mean held-out loss from 7.011798 to 6.123598. Exact JAX numeric parity, 65,536-value incremental/full identity, and the grounded tool-call generation remain unchanged | Exact numeric parity; real text generation; active recipe; wall and process-memory lead; complete real-data parameter training with analytic-gradient and multi-record held-out evidence | Add exact production checkpoint/resume and multi-epoch generation-quality evaluation; do not claim adaptive training parity because adaptive_new has no Needle trainer |
| Text embedding | No evidence-qualified encoder artifact in configured roots | Runtime and recipe exist; causal hidden-state substitution refused | Explicit refusal | Add a real encoder artifact, corpus, golden, wall, and peak |
| Text rerank | No evidence-qualified classifier-head artifact in configured roots | Runtime and recipe exist; causal LM substitution refused | Explicit refusal | Add a real reranker artifact, pair corpus, scores, wall, and peak |
| Pocket-TTS speech | Active adaptive host audio recipe and real-model generation fixture | Compiled production recipe; every generated latent/EOS value is bounded against the adaptive oracle; PCM max error `3.38e-4`, RMS `0.089438`, encoded WAV `24 kHz` mono; warm matched synthesis `0.48-0.53 s`; peak heap `0.540-0.543 GiB`. A compiled joint backbone+flow trainer consumes the native `Hello world.` generated codec latents through shared platform Muon. One real-artifact step at derived LR `3.383e-5` lowers loss `0.505204 -> 0.285938` and changes 16,358/16,384 sampled final-projection weights | Production output parity; warm-wall lead versus adaptive's retained `2.0 s` optimized route; real-artifact native-latent Muon trajectory | Measure both inference implementations under one cold/process-memory harness; add corpus audio encoding and held-out speech-training evidence |
| Un-0 image/video | Active deterministic Go route; synthetic bootstrap training was deleted | Exact fixture within `3.13e-7`; about 1.25 ms warm generation. Typed image and video recipes publish encoded media instead of float JSON. The real class-1/seed-42 PNG is 8x8, 173 bytes, and pixel-exact against adaptive's retained PNG. The real class-1/seed-202 route advances the seed for six frames, applies the request-owned 8x publication scale, and emits a 64x64, 10,028-byte GIF that is byte-identical to adaptive's retained artifact; 319 source pixels change between adjacent frames. Full generator VJPs remain finite-difference and adaptive-golden gated. The prior constant color-ramp objective and its public `TrainDrift` entry point were deleted; training now refuses until a recipe supplies real class/image authority | Exact image and video inference/publication parity; honest training refusal | Add real class/image training authority before restoring a trainer |
| SimpleDiffusion image | Real resident CUDA generation and real-checkpoint training | The retained seed-7, two-step, 64x64 host PNG remains byte-identical to adaptive_new. The typed production recipe now caches a geometry-keyed resident CUDA generator. Three repeated generation gates report 17.89-18.04 ms warm median versus 269.88-318.30 ms host median, float maximum difference `2.265e-6`, and one of 12,288 decoded channels differing by one byte (`0.000081` byte MAE). The complete forward retains 388.456 MiB of weights and peaks at 516.660 MiB. Final-projection and one complete decoder residual-block VJP now execute through shared resident CUDA owners | Real retained graph and recipe activation; at least 4x warm generation leadership without material pixel loss; two exact device-backward boundaries | Remaining backward blocks and matched whole-process generation-memory evidence |
| Wan text-to-video | Native adaptive exact path; Python oracle | Fresh session+denoise 331.96 s / 7.875 GB; decode 38.31 s / 10.330 GB; exact G3/G4 bounded, BF16 G3 cosine 0.999899, frame-0 CUDA/host max error 8.04e-6 | 370.27 s staged wall beats retained 463.4 s Python wall | Fresh same-revision Python/adaptive run; semantic clip gate |
| Krea text-to-image | Matched 256 fixture: 25.430 s; exact u8 SHA `b257e244`; full 2048 record: 158.144 s, 33.47 GB device | 256: 13.804 s / 25.030 GiB, MAE 0.02401. 2048: 63.510 s / 32.732 GB, MAE 0.03910; production caller and phased residency gated | Wall/peak lead at both sizes; bounded image quality | Exact SHA remains numerically unstable; retain robust pixel oracle and RMSE advisory |
| SenseNova image/edit | Adaptive exact Go replay: 206.010 s, 19.50 GB; Python 366.6 s, 47.58 GiB | Compiled routed-image recipe owns tokenizer/prompt, retained prefix/body weights, device vision/condition/head projections, shifted integration, and PNG publication. The native 1536x2720 infographic prompt runs for 50 steps in 136.603-136.829 s at 17.069 GiB. Both cold runs reproduce the reviewed PNG and pass its spatial, contrast, edge, and palette signature | 33.6% wall lead and 6.0% peak lead over adaptive on the full production route | Edit route still needs the fingerprinted source PNG absent from both repositories |
| LiveEdit video edit | Adaptive complete experimental route: best 109.5 s, 20.61 GB; Python: 60.1 s, 23.50 GiB | Shared Wan VAE encoding/decoding, tensor graph operators, and recipe/session owners execute the complete 30-block edit. Text projection, compiled graphs, cumulative per-layer K/V history, output buffers, and same-source latents remain resident. Exact bounded latent/frame replay remains gated. Three full 81-frame currant-to-grape gates take 81.6-81.9 s cold and 51.0-51.5 s with the source latent cached, at 15.624 GiB peak. Final latent cosine is 0.999890-0.999966 with 0.008276-0.014835 NRMS versus adaptive. Streamed MP4 comparison measures 0.00731-0.00753 MAE / 37.29-38.29 dB versus adaptive and 0.01012-0.01024 MAE / 32.31-32.58 dB versus Python; temporal-delta ratios are 1.021 and 1.008-1.009. LiveEdit uses its validated `[0,1]` output range; Wan uses `[-1,1]`. | Full numerical, conditioned-output, temporal, warm-wall, and peak leadership proven; cold wall beats adaptive | Python retains cold-wall leadership; recipe output remains GIF while the leadership gate streams MP4 |
| Server/API/session/cache | Mature adaptive-specific multimodal surface | One passing contract matrix covers native, OpenAI, and Anthropic streaming; structured tools; exact embedding inputs; ordered image/audio/video projection; stored response continuation; prompt-cache reuse and `n_keep`/`n_discard` context editing; client/timeout cancellation; and explicit missing-capability refusal | Broader protocol surface with consolidated behavioral evidence | Keep model-quality claims in recipe gates; add external protocol fixtures only when behavior changes |

## Training Matrix

| Training concern | adaptive_new | Overgo | Verdict |
| --- | --- | --- | --- |
| Training authority | Typed examples, nine objective kinds, admission, memory, progress, evidence; recipes not universal. Production dense admission still accepts LM only; test oracles wire FNS, latent L2/sequence, forecast and OCR, while generic flow matching, image-latent and distillation remain unwired in the dense owner | `TrainingRunPlan` and every `TrainingProgram` bind objective identity, dataset, operators and Muon authority. RepoDB v2 objective documents bind kind, modality pair, corpus, processors, projection/codec, loss, evaluation and evidence. Eight adaptive non-LM contracts compile through one forward/backward/Muon program shape; a missing program is refused. Separately, the 36-row real-record modality matrix approves three E4B input-to-text objectives and refuses 33 rows | Adaptive objective taxonomy ported without importing its family loop; executor and modality evidence remain separate |
| Model from scratch | `RunFromProviderRich` derives split, rune vocabulary, topology, initialization, batching, causal training and metrics from documents | Pinned source/config/oracle parity; direct flat initialization; shared tensor forward/VJP; resident CUDA weights, gradients, momentum, compiled graphs and pooled scratch | Matched three-step wall/peak lead; artifact publication and exact resume open |
| Fractale fast-weight bank | Adaptive has a complete analytic ThoughtBank backward and family-local update enumeration | One real Fractale checkpoint bank consumes the ordered WikiText-2 stream through the shared materializer. `TrainingProgram` owns forward, backward, and optimize order; the common tensor pack and Muon backend update all 9,879,476 affected parameters. Device and host updates match within `1.43e-7` at loss `0.01152465`; the full checkpoint-load plus dual-oracle gate takes 161.66 s | First real-data training organ proven without a parallel model runtime; complete block and model backward remain open |
| Optimizer | Adaptive optimizer machinery and model-specific use; historical SGD paths remain | Matrix/vector/scalar groups use one compiled Muon path; sign, BF16-SGD and family-local production updates deleted | Muon-only production authority |
| Dense causal LM | Carbon at `c72b6595d`: four 31-token DNA windows, loss `8.43767 -> 7.29179`, 6.816 s loop, 11.47 GiB. Qwen2.5-0.5B at `de547a363`: 505 ms warm step | Carbon on the same windows: loss `8.438149 -> 7.290705-7.293696`, 1.510-1.550 s loop, 4.985 GiB. Qwen on the pinned 512-token adaptive corpus: loss `4.592842 -> 2.958702`; 552-753 ms cold execution, 469-480 ms retained warm execution, 6.592 GiB peak | Carbon wall/peak lead; Qwen retained warm-wall lead with measured peak |
| Production dense command | Multiple adaptive entry points, uneven recipe authority | `cmd/train -freeze-lexical` selects the no-fallback resident contract; full-parameter mode remains explicit; `-resume` restores a validated checkpoint and advances the same stream | Carbon route and exact resume reachable |
| Checkpoint/resume | Fine-tune evidence and persistence exist, uneven by trainer | Atomic directory publication refuses overwrite and binds weight content, full Muon state, data/augmentation RNG counters, stream position, processor/projector/codec identities, compiled run/program identities, and lineage parents. Host and resident CUDA tests match uninterrupted weights and momentum exactly. | Dense production contract complete; adoption by scratch, seq2seq, speech, and diffusion trainers remains open |
| Qwen3.5 hybrid | Inference active; no Qwen3.5 training oracle at `214950b3b` | Model-native per-head GDN norm matches the local PyTorch gradient oracle. Real 4B GGUF recurrent layer 0 trains 112,885,760 matrix and 38,080 vector parameters through the compiled resident program on all five token IDs from a recorded prompt; loss `13.185789 -> 11.006020` | Overgo-only real-layer capability; complete 32-layer training remains open |
| Gemma E4B | Adaptive declares streamed CUDA LM scope and supplies real inference semantics/oracles; no retained adapter-training trajectory | Fingerprinted real language/projector artifacts and adaptive image/audio inputs feed one compiled layer-0 residual-adapter program. Device Muon updates 1,313,280 parameters on text, image, and audio examples; shared checkpoint resume reproduces weights and momentum exactly. The bound model plan verifies PLE, sliding/full layers, and shared-KV ownership. | Real multimodal adapter training proven; complete 42-layer language and projector training remains open |
| Gemma4 12B | Streamed CUDA LM scope | Shared backward prerequisites; no reported real 12B training | Gap |
| Forecast/latent/FNS | Test-oracle CUDA objectives for FNS, latent L2/sequence and forecast; production dense admission remains LM-only | All four semantics are objective-owned. Pocket-TTS executes latent-sequence training through the shared program; TimesFM has forward/loss/backward primitives but no production trainer; FNS and latent-mean lack production model bindings | Contract parity; Pocket execution proven; remaining executors open |
| OCR | Host objective; CUDA train/eval absent | OCR token objective and image-to-text signature compile through shared authority; no production executor | Contract parity; execution remains incomplete in both repositories |
| SimpleDiffusion/UViT | Real-checkpoint OT-flow trainer and resident CUDA VJP boundaries | The shared materializer selects structured 32x32 crops from the artifact's four real sample grids, then the common stream, normalized-image processor, seeded OT objective, full model VJP, and platform Muon stepper train the 101,828,450-parameter checkpoint. The trainer now executes the shared flow-matching forward/backward/Muon program. One train/held-out gate lowers matched train loss `1.331198 -> 1.156815` and held-out loss `1.396862 -> 1.346165`; 6,143 sampled final-projection values change. The final transpose projection and one complete 128x8x8 decoder residual block use shared resident CUDA VJPs. The residual block matches input gradients within `1.863e-9` and all nine parameter gradients within `4.768e-7` | Real-data full-checkpoint host update, shared objective program and two exact device-backward boundaries proven; remaining blocks and matched training performance remain open |
| Pocket-TTS | Latent bridge, flow-net, backbone paths | Inference recipe and reference waveform are production-proven. The production compiled trainer binds all 85,282,848 joint backbone+flow parameters to shared Muon; native generated codec latents lower loss `0.505204 -> 0.285938` in one step | Real-artifact Muon trajectory proven; corpus audio encoding and held-out evaluation open |
| Controller model | Strategic adaptive plan; no promoted controller | Git-pinned eight-action corpus, external holdout, three scratch CUDA seeds, RepoDB run/evaluation evidence, and external reversible decision. Held-out action accuracy improves from 12.5% to 37.5-50.0%; modality accuracy improves from 12.5-25.0% to 37.5-50.0%. | Contract proven; production activation, broader tasks, and compatible pretrained scorers remain open |
| Tier-1 memory scaling | Adaptive streamed/checkpointed components | A real step through the promoted 800-step controller construction supplies runtime peak and the live 2 MiB allocation granularity. The compiled forced-cap plan lowers logical device state from 8,649,248 to 8,200,648 bytes, lowers physical reservation from 10 MiB to 8 MiB, and places 530,520 bytes in host RAM across 11 segments. No connector state applies to this text controller. Checkpointing, layer-major accumulation, and paging remain disabled without measurements. | Scheduling contract and physical fit reduction proven; streamed training execution remains open |

### Current Overgo training evidence

- Dense resident: 12 steps; recorded loss difference `1.545e-5`, final-weight
  difference `3.613e-4`; loss `3.48 -> 2.17`.
- Dense resident scale fixture (12 layers, hidden 256, sequence 64): two pooled
  CUDA Muon steps must stay within 5% of the same-run unpooled control; live
  allocation peak is 120.1 MiB versus 143.5 MiB unpooled. The normalized wall
  gate replaces a 2 s absolute threshold that the original ratchet commit also
  misses under current device state. This is an Overgo ratchet, not adaptive parity.
- Carbon-500M, 64 tokens, four frozen-lexical Muon steps: loss
  `4.124540 -> 2.630955`, 1.476 s/step, 4.714 GiB peak. Ratchets are
  1.8 s/step and 6 GiB. Adaptive retained 2.018 s/step and later measured a
  15.06 GB Carbon training peak. A Git-history replay recovered adaptive's
  causal protocol at `c72b6595d`; current adaptive master refuses that gate at
  admission.
- Matched Carbon causal: identical artifact, four ordered DNA windows, LR and
  Muon setting. Repeated Overgo loss is `8.438149 -> 7.290705-7.293696`,
  resident loop is 1.510-1.550 s, and peak is 4.985 GiB. Adaptive loss
  `8.43767 -> 7.29179`, loop 6.816 s, admitted peak 11.47 GiB.
- Qwen2.5-0.5B: checkpoint, configuration, adaptive corpus, and execution
  protocol are hash-gated. Two retained 512-token steps report loss
  `4.592842 -> 2.958702`; cold execution is 552-753 ms and warm execution is
  469-480 ms versus adaptive's 505 ms warm reference. Peak use is 6.592 GiB.
- Hybrid resident synthetic stack: loss `15.92 -> 2.56`; host/device trajectory
  difference `2.856e-6`.
- Qwen3.5-4B recurrent layer 0: checkpoint `c4e8dc8f...`, local gradient oracle
  `76d9880f...`, serving data `6dcce666...`, and compiled program `cc95d667...`
  are pinned. Four real prompt transitions train 112,885,760 matrix and 38,080
  vector parameters; loss is `13.185789 -> 11.006020`, training wall is 1.552 s,
  and peak device allocation is 1.775 GiB. adaptive_new `214950b3b` has no Qwen3.5 training
  oracle; no parity or full-model claim is made.
- Gemma E4B layer-0 adapter: language GGUF `cd4ada47...`, projector
  `185786ec...`, and adaptive vision/audio inputs are fingerprinted. The bound
  model plan verifies layer 0 sliding attention, layer 5 full attention, and
  layer 24 shared KV from layer 22; it declares PLE and no AltUp/Laurel. Real
  text rows and CUDA-projected image/audio representations drive three compiled
  Muon updates over 1,313,280 adapter parameters. Losses are
  `14.315110/16.071185/16.074883`; update wall is 68.989 ms. Checkpointed and
  uninterrupted weights and Muon state are bit-identical. adaptive_new has no
  matched adapter-training trajectory; no complete-model claim is made.
- Multimodal objective matrix: fingerprinted Wikitext (`9e9fa1ad...`), P2 image
  (`026362cc...`), and LongSpeech audio/label (`ac135a0d...`/`8749bb8c...`)
  records pass the shared materializer and typed processors,
  the real E4B tokenizer or CUDA image/audio projector, the 1,313,280-parameter
  adapter, device Muon, and token-accuracy evaluation. Losses are
  `11.091556/11.795913/5.802830`; repeated three-update runs take
  60.045-61.414 ms. RepoDB approves
  text-to-text, image-to-text, and audio-to-text. The other 33 single-modality
  pairs are explicit refusals, not inferred training support.
- Adaptive objective-program matrix at `214950b3b`: FNS, latent L2,
  latent-sequence L2, forecast, OCR token prediction, flow matching,
  image-latent and logit distillation each bind stored objective evidence to a
  compiled forward/backward/Muon program. Removing the distillation program
  produces a refused row. This proves contract breadth and refusal, not eight
  production model executors. Pocket-TTS and SimpleDiffusion are the currently
  bound latent-sequence and flow-matching production consumers.
- Needle 26M, first real GSM8K training record, three compiled device-Muon
  steps over the decoder final norm and complete last cross-attention block:
  train loss `6.541863 -> 6.415488`; separate held-out record
  `8.629535 -> 8.541487`. Q/K/V BF16 words changed
  `18,849/13,626/11,000`; projection and normalization gradients match direct
  parameter finite differences. adaptive_new has no Needle training owner, so
  this is Overgo capability evidence rather than cross-repository parity.
- Needle's combined final-cross residual and query-path gradient reaches the
  last causal self-attention block. Its gate, output, Q/K/V, input norm, and
  Q/K norms now share the compiled Muon plan. Three complete-final-layer steps
  lower real GSM8K train loss `6.541863 -> 6.405072` and the held-out record
  `8.629535 -> 8.534937`; end-to-end Q/K/V finite differences include inverse
  RoPE and query-score scaling.
- Indexed reverse traversal now reaches decoder layer 6 cross-attention on the
  same GSM8K record. Q/K/V core gradient norms are
  `0.309892/0.131402/0.001113`; raw-gate analytic `-0.023812` matches the
  direct `-0.023562` finite difference. Its complete cross block now shares
  the Muon plan. A four-word central BF16 stencil verifies the earlier-depth
  Q/K/V projections; input/Q/K norm gradients use direct float32 differences.
  Three steps lower train loss `6.541863 -> 6.394826` and held-out loss
  `8.629535 -> 8.533603`, changing 21,334/262,144 Q, 13,886/131,072 K, and
  11,389/131,072 V words. Layer 6 causal self-attention now shares the same
  compiled plan and inverse-RoPE projection VJP. Three complete-pair steps
  lower train loss `6.541863 -> 6.387569` and held-out loss
  `8.629535 -> 8.531155`, changing 21,048/262,144 Q, 15,945/131,072 K, and
  13,403/131,072 V words. Its Q/K/V and input/Q/K norm finite differences all
  pass.
- The same indexed traversal now covers decoder layers 5 through 0 instead of
  adding layer-specific trainers. All 129 final-norm and decoder-attention
  parameter groups have finite nonzero gradients on the real GSM8K record.
  Three complete-decoder steps lower train loss `6.541863 -> 6.285377` and
  held-out loss `8.629535 -> 8.506213`; every self/cross Q/K/V block updates,
  changing 803,956 BF16 words. Encoder-memory and lexical-embedding gradients
  remain open.
- Every decoder cross block now contributes its K/V source gradient to one
  encoder-memory boundary. The encoder final norm and all 12 bidirectional
  attention blocks reuse the same traced attention VJPs, indexed parameter
  layout, compiled program, and Muon owner. All 226 attention-stack parameter
  groups are finite and nonzero; all encoder Q/K/V blocks update. Three steps
  lower train loss `6.541863 -> 6.083478` and held-out loss
  `8.629535 -> 8.451201`. The encoder final-norm gradient is
  `0.104070` analytic versus `0.103963` direct. Tied embeddings remain open.
- The single tied BF16 embedding now accumulates output-head, decoder-input,
  and encoder-input gradients before Muon. Its analytic gradient
  `-0.0750179` matches the radius-4 direct difference `-0.0750161`. Three
  complete-model steps lower train loss `6.541863 -> 5.503946` and held-out
  loss `8.629535 -> 8.335628`; 139,010 embedding words change, including both
  source and decoder token rows. Needle's complete parameter set is now bound;
  broader split evaluation and production resume remain open.
- A deterministic GSM8K corpus gate trains records 0-3 and holds out records
  4-7. Two ordered epochs lower mean train loss `7.491162 -> 5.274749` and
  mean held-out loss `7.011798 -> 6.123598`. This replaces the single-example
  trajectory as the breadth claim; production resume and longer generation-
  quality evaluation remain open.
- SimpleDiffusion consumes six structured training crops and two held-out crops
  selected from the real checkpoint's sample grids through the shared binary-
  record materializer and normalized-image processor. The adaptive-seeded OT
  path and full 101,828,450-parameter VJP feed the shared platform Muon
  stepper. One update lowers matched train loss `1.331198 -> 1.156815` and
  held-out loss `1.396862 -> 1.346165`; 6,143 final-projection values change.
  The 3.34-4.81 s focused gates are capability evidence, not a matched performance
  claim. Forward is resident CUDA. The final projection and one complete
  decoder residual block are migrated backward boundaries; remaining blocks
  remain host-owned.
- The real final transpose projection now compiles to the shared resident
  linear VJP layout. Against a real forward activation and an upstream MSE
  gradient derived from the retained real image, CUDA input-gradient maximum
  error is `6.985e-10` and checkpoint-layout weight-gradient maximum error is
  `2.980e-8`. The resident session retains the reordered checkpoint weight;
  dynamic activation and loss gradients reuse fixed device buffers.
- One real final-decoder residual block now composes shared resident Conv2D,
  GroupNorm, SiLU, scale and add VJPs. The 128x8x8 block retains both
  convolution and normalization weights plus one fixed workspace. Against the
  host oracle, input-gradient maximum error is `1.863e-9`; the maximum across
  all convolution, normalization, bias and residual-scale gradients is
  `4.768e-7`. The current boundary uploads retained host forward activations;
  full device-forward activation retention remains open.
- SimpleDiffusion's typed prepare/integrate/decode recipe publishes the real
  checkpoint's seed-7, two-step, 64x64 sample as a 10,506-byte PNG identical
  to adaptive_new's retained artifact. That host result remains the exact
  publication oracle; the resident CUDA result and leadership evidence are
  reported below.
- SimpleDiffusion and adaptive_new now run the same real-checkpoint forward in
  isolated child processes. The shared safetensors reader decodes native F32
  directly into final slabs and the diffusion loader traverses tensors
  deterministically. Two complete three-process gates put Overgo's median wall
  at `0.602-1.162 s` versus adaptive_new's `0.710-1.263 s`; median working-set
  peak is `407.949-408.906 MiB` versus `411.195-411.836 MiB`. Each child also
  enforces its vendor golden before contributing evidence.
- The first real SimpleDiffusion encoder residual block compiles from the
  checkpoint-owned weights into shared GroupNorm, SiLU, Conv2D, scale and add
  operators. The reference and CUDA graph outputs each match the existing host
  block within `4.768e-7` at 128x8x8. This proves the reusable outer-block
  device path.
- The first real 512-channel SimpleDiffusion middle transformer composes CP
  factor projections, reverse split-half RoPE, dense attention, xATGLU,
  LayerNorm, MLP projections and scaled residuals through shared tensor and
  CUDA owners. Reference and CUDA outputs match the host block within
  `1.490e-8`. Full graph residency remains.
- The real patch projection and both encoder transitions now compose shared
  Conv2D operations; downsampling is a depthwise 2x2 average convolution.
  CUDA maximum error is `9.537e-7` for the 128x8x8 patch output, `2.384e-7`
  for the 256x4x4 transition and `5.960e-8` for the 512x2x2 transition.
  Decoder upsampling and final unpatching are covered below.
- Both real decoder transitions now compose 1x1 Conv2D, Concat and the shared
  `PixelShuffle2D` operation; CUDA maximum errors are `1.192e-7` and
  `1.907e-6`. The final stride-4 transpose convolution compiles to a reordered
  MulMat projection plus pixel shuffle and matches within `3.338e-6` at 32x32.
- All 46 real encoder, middle and decoder blocks pass a `1e-5` CUDA boundary
  ratchet. The observed maximum is `2.861e-6`; all 16 middle transformers stay
  at or below `2.980e-8`. This closes shape-specific block coverage, not full
  graph residency.
- One compiled graph now owns the complete 101.8M-parameter forward and retains
  388.456 MiB of static F32 inputs. Three fresh runs report 220-297 ms compile
  and upload, 32.1-35.5 ms cold execution, and 6.30-6.62 ms warm median versus
  53.8-61.3 ms host median. Peak device allocation is 516.660 MiB. Four
  executions transfer only 0.049 MiB host-to-device after binding, proving
  weights remain resident. Golden maximum error is `5.722e-6`. Gates require
  cold leadership, at least 4x warm leadership, error at most `2e-5`, and
  bounded resident memory.
- The Windows typed image recipe now leases a geometry-keyed
  `ResidentGenerator`; seed and step changes reuse the same compiled graph.
  Three real seed-7, two-step, 64x64 gates report 17.89-18.04 ms warm sampling
  versus 269.88-318.30 ms host medians. CUDA and host floats differ by at most
  `2.265e-6`; decoded PNGs differ in one of 12,288 color channels by one byte,
  with byte MAE `0.000081`. The gate requires at least 4x leadership, float
  error at most `3e-5`, decoded-channel maximum one, and byte MAE at most
  `0.001`.
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

These results promote matched Carbon and Qwen2.5-0.5B frozen-lexical
leadership and real E4B adapter capability. They do not justify complete E4B,
12B, controller, or system-wide training closure.

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
| Un-0 generation | Retained PNG and class-1/seed-202 six-frame GIF | Pixel-exact PNG; byte-identical 64x64 GIF, 10,028 bytes, 319 changed source pixels | Exact image and video publication parity; matched peak remains open |
| Wan full generation | Python baselines vary by retained report: 399.1 or 463.4 s; adaptive exact Go 795.1 s warm in the current media report | fresh 331.96 s session+denoise + 38.31 s decode = 370.27 s; stage peaks 7.875 / 10.330 GB | 20.1% wall lead vs retained 463.4 s Python; fresh matched reference rerun still open |
| Krea 256 generation | adaptive exact Go 25.430 s | 13.804 s / 25.030 GiB; MAE 0.02401, RMSE 0.05459 | 46% wall lead; lower peak; bounded quality lead |
| Krea 2048 generation | Python 97.5-135.4 s; adaptive exact Go 158.144 s / 33.47 GB | 63.510 s / 32.732 GB; MAE 0.03910 | 35% lead vs best Python; 60% vs adaptive; 2.2% lower peak |
| Carbon-500M causal training | adaptive `c72b6595d`: 6.816 s loop / 11.47 GiB admitted; loss 8.43767 -> 7.29179 | 1.510-1.550 s loop / 4.985 GiB measured; loss 8.438149 -> 7.290705-7.293696 | At least 77.3% loop-wall lead; 56.5% lower peak |
| Qwen2.5-0.5B causal training | adaptive `de547a363`: 505 ms retained warm step | 469-480 ms retained warm execution; 552-753 ms cold execution; 6.592 GiB peak; loss 4.592842 -> 2.958702 | At least 5.0% warm-wall lead; cold and warm lifecycles gated separately |
| Corpus-derived scratch causal training | adaptive pinned run: 221.4-231.8 ms / 3.13-3.18 MB peak heap | 58.0-65.4 ms matched steps / 1.15-1.17 MB combined runtime peak; six lifecycle phases and complete state gated | At least 3.38x step-wall lead; at least 62% lower peak; trajectory delta `1.738e-3` |
| Gemma E4B per-layer adapter training | No retained adaptive adapter-training trajectory | Three real text/image/audio updates over 1,313,280 layer-0 parameters in 68.989 ms; exact checkpoint resume | Overgo-only real adapter capability; complete stack open |
| SenseNova full generation | Adaptive exact Go replay 206.010 s / 18.162 GiB | Two cold production runs at 1536x2720 and 50 steps: 136.603-136.829 s / 17.069 GiB; reviewed PNG signature passes and both measured runs are byte-identical to the reviewed output | 33.6% wall lead and 6.0% peak lead; edit input oracle remains absent |
| LiveEdit full edit | Python 60.1 s / 23.50 GiB; adaptive best 109.5 s / 20.61 GB | 81.6-81.9 s cold; 51.0-51.5 s same-source warm; 15.624 GiB peak. Final latent cosine 0.999890-0.999966 and NRMS 0.008276-0.014835 versus adaptive. MP4 output MAE/PSNR: 0.00731-0.00753/37.29-38.29 dB versus adaptive and 0.01012-0.01024/32.31-32.58 dB versus Python. Temporal-delta ratios: 1.021 and 1.008-1.009. | Cold wall and peak lead versus adaptive; warm wall and peak lead versus Python; Python retains cold-wall lead |
| Gemma4 12B FP8 | 488.886-497.603 ms case-0 generation / 14.363 GiB runtime-owned peak | 231.979-248.041 ms / 14.353 GiB; exact `[50429, 106]` | 49.7-53.0% request-wall lead; 10.4 MiB peak lead; Overgo model-open wall remains 23.77-24.03 s versus adaptive 2.82-3.21 s |

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

### P0: production training leadership remains narrow

Carbon and Qwen2.5-0.5B have reachable frozen-lexical CUDA contracts and matched
leadership evidence. Qwen3.5 has one real recurrent-layer training gate without
an adaptive oracle. Extend that compiled path across the complete stack, then
E4B; add held-out promotion before broader claims. Exact dense checkpoint/resume
evidence is already gated.

### P0: E4B complete-stack training remains open

The real artifact now drives layer-0 adapter training and the bound topology
verifies PLE, SharedKV, and mixed sliding/full attention. The artifact declares
no AltUp or Laurel tensors, so those mechanisms are not part of this E4B claim.
Training all 42 language layers and the vision/audio towers, with retained
device state and held-out promotion, remains open.

### P1: checkpoint adoption is incomplete

`cmd/train` now publishes one staged directory atomically, refuses overwrite,
validates weight content on load, and resumes exact host or resident CUDA Muon
state with data/RNG/compiled-authority/lineage bindings. Scratch, seq2seq,
speech, and diffusion trainers still need to consume this shared owner.

### P1: Muon-only production authority needs a scanner

Dense and SimpleDiffusion training now express matrix/vector/scalar groups
through the shared compiled Muon path; sign, BF16-SGD and the image-local SGD
loop are deleted. Add a production-code scanner so new local weight-update loops
cannot bypass the optimizer package.

### P1: training documentation must track executable evidence

`docs/plan.json` contains only open executable work, while Git owns chronology.
Training claims must continue to name the tested artifact, data, lifecycle,
backend boundaries, verifier, and remaining gaps.

### P1: compatibility breadth outruns evidence depth

All 135 model entries are experimental; only Gemma3, Qwen3, Qwen3.5, and T5
encoder name validated fixtures. Add evidence tiers, source commits, artifact
hashes, failable commands, and expiry/reverification rules. An architecture
parser is not a promoted model capability.

### P1: media breadth remains SenseNova edit and LiveEdit publication

Krea, Wan, and full-size SenseNova have production performance gates. SenseNova
editing still lacks its source PNG oracle. LiveEdit has typed production activation,
exact bounded replay, a full native quality comparison, and lifecycle-separated
performance evidence. Its remaining gaps are cold-wall leadership versus Python
and recipe-configured MP4 publication.

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
9. Extend the proven real Qwen recurrent-layer path to the complete hybrid
    stack; then train E4B, exercising every adaptive-declared modality.
10. Compile the multimodal training matrix; port remaining adaptive objective
    families through shared programs and explicitly refuse absent objectives.
11. Exercise the recursive proposal and admission boundary with measured
    descendants; require a child to beat its parent under matched evaluator and
    split identities before recording improvement.

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
