# Training plan — scratch-controller-first, measured, memory-hierarchy-aware

This plan targets one measured workstation:

- GPU: NVIDIA GeForce RTX 4090 D, 49,140 MiB reported VRAM.
- Host RAM: approximately 94 GB available to the system.
- GPU link: PCIe 4 x16, subject to measured pinned-transfer throughput.
- Local storage: Gen4 NVMe with approximately 1.1 TB free.

The immediate product objective is not full training of every supported model. It
is a trustworthy model-from-scratch learning loop for a small workflow controller.
Adaptive_new supplies the verified construction and training semantics; Overgo
supplies compiled recipes, resident CUDA execution, Muon, artifact lineage and
promotion. Larger-model offload follows after that loop works.

## 1. First product milestone

A small causal controller can be derived from an immutable Git/RepoDB corpus,
initialized from a sealed recipe, trained on-device, checkpointed and resumed
exactly, and promoted or rolled back entirely through RepoDB identities. It must
improve across multiple seeds on immutable held-out workflow tasks and pass
regression plus typed-action validity gates. A pretrained 350M–1B candidate remains
a comparison rung, not a prerequisite for proving the scratch pipeline.

This milestone requires all of the following:

1. Corpus-derived vocabulary, shape and batch facts with immutable identities.
2. Deterministic parameter manifest and initialization.
3. Device forward, backward, loss and Muon execution.
4. Exact step-boundary and accumulation-boundary checkpoint contracts.
5. Immutable dataset splits and complete artifact lineage.
6. Fixed-seed adaptive/host/device trajectory evidence.
7. Multi-seed held-out task improvement.
8. Candidate, evaluated, promoted and rollback lifecycle.

E4B, 12B, dynamic quantization and NVMe optimizer paging are not prerequisites
for this milestone.

### Current implementation boundary (2026-08-13)

Implemented substrate:

- dense host and CUDA forward, loss, backward and Muon matrix updates;
- device Newton–Schulz and resident dense weights/momentum across steps;
- host/device parity and descending-loss fixtures for dense and synthetic
  hybrid stacks;
- lower-level dense checkpoint/resume trajectory tests;
- host Gemma3n AltUp, PLE, Laurel and mixed-window VJPs plus shared device
  operator VJPs.

Production gaps:

- `TrainingRunPlan` and `TrainingProgram` are design contracts, not implemented
  Go types;
- no production type derives a new model vocabulary, topology, parameter manifest
  and initialized checkpoint from a dataset artifact;
- adaptive_new's scratch path is not yet captured as fingerprinted construction,
  initialization, batch-order and training-trajectory oracles;
- `cmd/train` selects the resident dense device loop; the displaced per-step
  weight scatter/gradient gather training adapters are deleted;
- production checkpoints are not atomic complete-state resumes;
- matrix, vector and scalar groups share the Muon Newton–Schulz path;
- no real Qwen3.5, Gemma E4B or Gemma4 12B artifact has completed the compiled
  resident training contract;
- compiled multimodal training authority is not implemented as a sole runtime
  owner;
- real multimodal processor/projector/codec gradient and held-out quality gates
  remain open.
- SimpleDiffusion compiles every real tensor into shared Muon geometry; its
  real update remains blocked on a device/resident Muon program because host
  Newton–Schulz at artifact scale is not an admissible production path.

Device backward and device Newton–Schulz are implemented. Production
reachability and real-model integration are now the long poles.

## 2. Authority and compiled contracts

RepoDB owns the identities and lineage for:

- construction profile, tokenizer, parameter manifest and initializer;
- initial model or scratch-initialized checkpoint;
- dataset source, normalization and immutable split;
- training recipe and compiled program;
- objective mixture and sampling policy;
- optimizer, schedule and precision policy;
- run, measurement and evaluation;
- checkpoint and resume lineage;
- candidate, promotion and rollback decision.

A training workflow recipe currently describes orchestration. It must compile to
a sealed `TrainingRunPlan` before production training begins. Its initial-state
union names exactly one pretrained artifact, resume checkpoint or sealed scratch
construction. The plan owns:

- model, dataset, split and recipe identities;
- a compiled model-level `TrainingProgram`;
- optimizer groups and typed update policies;
- precision, memory and transfer policies;
- objective and sampling configuration;
- checkpoint boundaries and retention policy;
- evaluation and promotion gates.

The model-level `TrainingProgram` contains ordered semantic forward and backward
operators, indexed tensor and parameter bindings, saved-versus-recomputed tensor
liveness, gradient destinations and optimizer-group bindings. CUDA launch details
remain executor concerns. Missing compiled facts are initialization errors; no
production fallback reconstructs training policy from model-family predicates.

### 2.1 Scratch construction contract

The first port preserves the behavior of adaptive_new
`training.RunFromProviderRich` at explicit boundaries:

1. normalize and fingerprint the document provider;
2. derive deterministic train, validation and test membership;
3. derive the character vocabulary, context, width, heads, MLP width, depth,
   learning rate, initialization scale and batch facts from the train partition;
4. compile the adaptive causal topology and ordered parameter manifest;
5. initialize each tensor from the named seed and initializer algorithm;
6. compile causal forward, cross-entropy backward and Muon groups;
7. execute ordered batches, validation and shift evaluation;
8. publish checkpoint, run, evaluation and promotion lineage.

Overgo adds a sealed `ScratchConstruction` to `TrainingRunPlan`. It contains the
source dataset/split identities, derivation-policy identity, topology profile,
tokenizer contract, tensor roles/shapes, tying rules, initializer algorithm and
independent split/init/data/augmentation RNG stream identities. Derived values are
stored as facts in the compiled plan; production callers do not expose them as a
bag of hyperparameter flags.

The parity profile first reproduces adaptive_new's sorted rune vocabulary,
corpus/horizon laws, uniform `[-InitStd,+InitStd]` weights, zero-initialized
positional terms, fused QKV layout, attention/MLP math and batch order. It is a
neutral model program, not a copy of adaptive_new autograd or an approximation to
Llama. Exact reference mode may retain adaptive_new's coupled RNG sequence for
oracle replay. Production profiles use versioned independent RNG streams so a
data-order change cannot silently change initial weights.

Construction emits a canonical initialized model artifact plus tokenizer,
parameter-manifest and compiled-program identities before the first update. This
makes scratch initialization an inspectable checkpoint boundary and lets serving,
training, evaluation and descendants consume the same artifact contract.

### 2.2 Port and performance boundary

Adaptive_new is an offline oracle, not a runtime dependency. One fingerprinted
export records source commit, dataset/split IDs, derived facts, parameter probes or
digests, token batches, logits, loss, gradients, first Muon update, multi-step
trajectory and resume boundary. Overgo must match that host oracle before CUDA or
fusion work can claim correctness.

After parity, Overgo compiles the same semantic graph onto its tensor executor and
retains weights, gradients, Muon momentum, activations/recompute state and reusable
scratch for the session. Optimization order is:

1. compile once; retain topology, buffers and CUDA graphs;
2. keep initialization, token batches, forward/backward and Muon state on device
   where measured beneficial;
3. reuse liveness-planned activation and Newton–Schulz scratch by capacity class;
4. fuse only oracle-covered operator boundaries;
5. admit BF16 or alternate initializer streams only as new fingerprinted profiles
   with matched convergence and quality evidence.

Leadership compares identical corpus, split, derived configuration, initial
weights, batches, objective, seed, update count and validation cadence. Record
compile/init wall, cold and warm step wall, peak VRAM/host RAM, transfers, launches,
loss trajectory, gradient error and final-weight error. Faster initialization does
not compensate for a changed training trajectory.

### 2.3 Multimodal contract

Each plan compiles an ordered input/target modality signature from RepoDB:
text, image, audio, video, time series or table. It also binds processor,
projector, merger, codec, mask/position, augmentation, objective and
modality-native evaluation identities. Train/freeze boundaries are explicit;
every declared trainable connector, trunk and head receives gradient evidence.

Applicable objective pairs include text-to-text, image/audio/video-to-text,
text-to-audio, text/image-to-image, text/image/video-to-video, forecast and
table prediction. A pair enters training only with adaptive_new evidence or an
approved objective and real corpus. Inference support alone does not create a
training claim. Missing pairs compile to explicit non-trainable/refused rows;
synthetic tensors cannot substitute for real modality evidence.

Checkpoint identity includes processors/codecs and RNG/augmentation state.
Promotion requires end-to-end processor-to-output evidence: exact boundaries
where deterministic, modality-native held-out quality otherwise, plus matched
wall and peak memory. Scalar loss descent alone is insufficient.

## 3. Memory models

Memory claims must name the representation they describe. Current and target
state must not share one estimate.

### 3.1 Current host optimizer

The current optimizer holds:

| Component | Representation | Bytes/parameter |
|---|---:|---:|
| Weights | FP32 | 4 |
| Gradients | FP32 | 4 |
| Momentum | FP64 | 8 |
| **Persistent minimum** |  | **16** |

Newton–Schulz input, output and Gram buffers add group-dependent scratch. This is
a correctness reference and small-model implementation, not the state layout used
by the future 8-byte estimate.

### 3.2 Initial device baseline

Bring-up proceeds in two explicit precision stages:

1. FP32 weights, gradients, accumulation and momentum for the smallest fixture.
2. BF16 compute weights with FP32 gradient accumulation and FP32 momentum.

If BF16 weights are authoritative, update accuracy must be measured. If an FP32
master is retained, its additional 4 bytes/parameter must be budgeted. Individual
gradient transfer may later use BF16 or FP8; accumulated gradients remain FP32
until convergence evidence supports another representation.

### 3.3 Target optimized layouts

Candidate layouts are compiled policies, not implicit assumptions:

| Layout | Persistent state | Approximate bytes/parameter |
|---|---|---:|
| BF16 master + BF16 gradient + FP32 momentum | 2 + 2 + 4 | 8 |
| FP32 master + BF16 compute + BF16 gradient + FP32 momentum | 4 + 2 + 2 + 4 | 12 |
| BF16 master + FP32 accumulated gradient + FP32 momentum | 2 + 4 + 4 | 10 |

Derived forward copies, pinned transfer buffers, optimizer scratch, saved
activations, CUDA workspaces, allocator fragmentation, runtime memory and operating
system headroom are budgeted separately. A model fits only when its measured peak
stays below a configured safe high-water mark. Because OOM is a hard halt, the
high-water mark is a **tail bound (p99 of measured peak), not a mean** — measured
allocation has run-to-run variance (fragmentation, workspace sizing, allocation
order) and no stationarity guarantee across driver or thermal state, so budgeting
to the average peak OOMs on the tail.

## 4. Muon-only optimizer policy

Muon is the only optimizer family. The compiled plan assigns an explicit Muon
geometry to every trainable parameter group rather than routing exceptional shapes
to a second optimizer:

- `MuonMatrix`: Newton–Schulz orthogonalized Nesterov momentum over the compiled
  two-dimensional view, including dense embeddings and output projections.
- `MuonVector`: the same update over a compiled `1xN` or `Nx1` view.
- `MuonScalar`: the same update over a compiled `1x1` view.
- `Frozen`: no state and no update for parameters excluded by the recipe.

Higher-rank tensors receive a semantic matrix view compiled from their tensor
role; they are not flattened by an unexplained runtime convention. Tied tensors
have one optimizer binding and one state allocation. Weight decay, clipping and
loss scaling are orthogonal transforms inside the Muon step, independently typed
by parameter role, not alternate optimizers.

The sign and BF16-SGD paths are deleted. Every geometry shares one Muon
configuration, schedule, checkpoint schema and plan identity. Matrix and vector
CPU/device trajectories are gated; scalar coverage remains part of production
program promotion. Newton–Schulz streams per optimizer
group; its memory plan includes the current matrix, output, Gram, polynomial
scratch and conversion buffers. The planner rejects a group whose peak scratch
cannot fit its assigned capacity class.

## 5. Memory hierarchy and execution schedule

### Tier 0 — VRAM

Active weights, live activations, gradient workspaces, optimizer scratch and
double-buffered transfer slots. All math occurs here unless a compiled host
operator explicitly says otherwise.

### Tier 1 — host RAM

Pinned staging buffers, offloaded master weights, accumulated gradients, optimizer
state and explicitly spilled checkpoint boundaries. The planner reserves host and
operating-system headroom rather than treating all installed RAM as allocatable.

### Tier 2 — NVMe

Dataset shards, durable checkpoints and cold overflow. Optimizer-state paging is
a last-resort experimental policy because it adds read/write latency, synchronization
and endurance cost.

### Scheduling rules

1. Stream layer weights through reusable pinned double buffers.
2. Prefetch only when measured transfer can overlap useful compute.
3. Store only compiled checkpoint boundaries; recompute internal activations.
4. Accumulate gradients in FP32 by default.
5. To amortize weight transfer across microbatches, use an explicit layer-major
   schedule that retains a layer while processing several microbatches.
6. Account for the additional activation storage or spill induced by layer-major
   scheduling.
7. Release or reuse buffers according to compiled liveness, not garbage-collector
   timing.

## 6. Measured scheduling model

Theoretical peak FLOP/s and link bandwidth are hypotheses only. The planner uses
calibration records keyed by device, operator, shape, dtype and transfer direction.

Per layer and capacity class, measure:

- forward execution time;
- backward execution time;
- recomputation time;
- CPU-to-GPU and GPU-to-CPU transfer time;
- quantization, dequantization and conversion time;
- pinned-buffer setup and synchronization overhead;
- achievable overlap and peak allocated memory.

The scheduler compares complete candidate timelines rather than only
`FLOPs/peak > bytes/link`. Gradient accumulation affects transfer amortization only
when the selected execution schedule reuses loaded weights. Every compiled memory
schedule records its calibration identity and rejects incompatible hardware or
capacity classes.

Calibration records are distributions, not point estimates. A memory-fit or
overlap decision that must not fail (peak memory, whether a prefetch hides behind
compute) uses a tail statistic (p95/p99); a throughput estimate that only affects
expected wall-clock may use the median. Each calibrated timing carries its sample
count and spread so the planner can tell a stable measurement from a noisy one
rather than trusting a single sample.

## 7. Checkpoint and resume contracts

Two checkpoint schemas are required:

### Step-boundary checkpoint

- authoritative weights;
- optimizer state and step;
- learning-rate and objective schedules;
- RNG streams;
- dataset shard, cursor and sampler state;
- precision and memory policies;
- compiled recipe and program identities.

No accumulated gradient is required after a committed optimizer step.

### Accumulation-boundary checkpoint

Includes every step-boundary field plus:

- accumulated gradients;
- microstep index and target accumulation depth;
- pending loss-scaling state;
- scheduler state needed to resume the exact transfer/recompute boundary.

Exact-resume tests compare the uninterrupted and resumed trajectories, not only
their next loss value.

## 8. Quantized training transport

Quantization follows a working BF16 Tier-1 baseline. Existing inference codecs
prove encoding and device-consumption mechanics, not training convergence.

Admit one state category at a time:

1. FP8 forward-weight transport or resident compute copy.
2. FP8/BF16 gradient transport with FP32 accumulation.
3. Quantized activation spill.
4. Optimizer-state compression.
5. Q6/Q4 forward copies only after multi-seed convergence evidence.

Each policy specifies:

- scale granularity and update cadence;
- deterministic or stochastic rounding;
- saturation accounting;
- straight-through estimator behavior for a quantized forward copy;
- error-feedback or residual accumulation for lossy gradient transport;
- authoritative master representation;
- stability and rollback thresholds.

Precision choices form a measured Pareto catalog over memory, transfer time,
kernel time and error. They are not ordered only by nominal bit width and do not
change opportunistically during a run unless the recipe defines a validated phase
transition.

Quality gates include gradient relative error, update cosine similarity, sampled
directional derivatives, fixed-seed trajectory bounds and multi-seed held-out
evaluation. Monotonic minibatch loss is not required or sufficient.

## 9. Controller learning plan

The first controller campaign trains a small model from scratch through the ported
adaptive construction semantics and Overgo execution. Its exact size is derived
from the admitted corpus, training horizon and memory policy; the compiled facts,
not a marketing size label, define the candidate. The first three evidence rungs
are a tiny deterministic fixture, the matched adaptive scratch profile and the
full workflow corpus.

Fractale-350M, Carbon-500M and Qwen2.5-0.5B remain pretrained comparison
candidates. They use the same dataset, objective and promotion suite where their
tokenizer/context contracts permit a fair comparison. A scratch candidate is not
promoted merely for being self-constructed; it must beat the current controller
on held-out utility per unit wall, memory and inference cost. Conversely, a
pretrained candidate cannot displace construction authority by requiring a
family-local trainer.

### Dataset composition

- successful workflow and tool-use traces;
- tool selection and argument construction;
- repairs after tool, validation and execution failures;
- evaluation decisions and evidence interpretation;
- negative examples, rejected actions and failed trajectories;
- concise completion and escalation behavior.

### Split and lineage policy

- Split before augmentation or self-generation.
- Keep repository, task family and near-duplicate groups within one split.
- Preserve an immutable external holdout unavailable to data generation.
- Record source, license, normalization and generator lineage.
- Quarantine generated examples until quality and contamination checks pass.
- Never promote evaluation traces directly into training without a new versioned
  dataset and split audit.

### Objective mixture

The recipe defines weights and sampling policy for:

- next-token behavior cloning;
- structured tool-call validity;
- workflow completion;
- repair and recovery;
- preference or ranking examples;
- optional auxiliary state/value prediction.

The scratch milestone uses full-model causal training. Auxiliary structured-action
and state/value heads enter only after the causal baseline is reproducible.
Pretrained comparisons may use supervised fine-tuning or adapters, then selective
or full-model fine-tuning only when the same evaluation suite shows a useful
increment.

### Evaluation

- syntax and schema validity of tool calls;
- correct tool and argument selection;
- workflow completion rate;
- recovery after injected failures;
- held-out repository/task performance;
- token, latency and tool-call efficiency;
- general-language and prior-capability regression;
- contamination and memorization checks.

Promotion requires multiple seeds, predefined thresholds and comparison against
the current promoted controller. RepoDB records candidate, evaluation, decision
and rollback lineage. A single descending training-loss curve cannot promote a
model.

## 10. Implementation ladder

`docs/plan.json` (driven by `cmd/plan`) is the single execution owner: it contains
only dispatchable or externally blocked work. Completed evidence belongs in Git
and RepoDB, not the live queue. This section and the §11 model
ladder are rationale and sequencing only — they explain *why* the rungs are
ordered this way and *what* each proves; they do not record completion. When a
training rung lands, it leaves `plan.json`. If this ladder
and `plan.json` ever disagree on scope, `plan.json` wins and this section is
corrected to match. Add or rename a training rung in `plan.json` first, then
reflect the rationale here.

Each rung produces a runnable artifact and an evidence record. A later rung does
not redefine an earlier rung's correctness contract.

1. **Capture the adaptive scratch oracle.** Export one content-addressed corpus,
   split, derived config, initialization, token-batch, forward/backward, Muon,
   trajectory and resume evidence bundle from a pinned adaptive_new commit.
2. **Compile training and construction authority.** Add sealed `TrainingRunPlan`,
   `ScratchConstruction` and model-level `TrainingProgram`; bind RepoDB dataset,
   split, topology, initializer, ordered input/target modalities,
   processor/projector/codec, objective and policy IDs. Derive the
   applicable/refused multimodal matrix from those facts.
3. **Match the adaptive host path.** Compile the adaptive causal topology without
   importing scalar autograd; match derived facts, parameters, logits, loss,
   gradients, first update, multi-step trajectory and reload.
4. **Resident scratch Muon path.** Route the matched program through device
   forward/backward, matrix/vector/scalar Muon, retained model/optimizer state,
   pooled scratch and production command routing.
5. **Exact recovery.** Lower-level dense trajectory tests exist. Implement both
   checkpoint schemas at the production boundary: atomic publication, complete
   Muon/RNG/data/program/construction state and uninterrupted-versus-resumed
   equality.
6. **Beat adaptive_new on the matched scratch run.** Separate compile/init, cold
   and warm training phases; require equal quality with lower repeated wall and
   peak memory before recording leadership.
7. **Mixed-precision resident training.** BF16 compute with FP32 accumulation;
   establish loss scaling, clipping and convergence envelopes.
8. **Muon scale-up.** Stream all Muon geometry groups, reuse Newton–Schulz scratch
   by capacity class and compare complete CPU/device update trajectories.
9. **Scratch controller proof.** Build the workflow corpus explicitly from Git
   and RepoDB evidence, train multiple scratch seeds, and pass the immutable
   held-out promotion suite, including typed text/image/audio/video component and
   workflow actions. Compare against pretrained candidates under the same suite.
10. **Recursive improvement boundary.** Let the promoted controller propose data,
    recipe and component-composition candidates, but keep corpus admission,
    evaluation and promotion outside the model. Every descendant names parent,
    data, recipe, evaluator and decision identities; no model self-promotes.
11. **Forced Tier-1 streaming.** Artificially cap VRAM on the small model; prove
   double-buffered weight/gradient transfer, overlap and exact results.
12. **Checkpointed activations.** Prove recomputation independently, then compose
   it with Tier-1 streaming.
13. **Layer-major accumulation.** Reuse each loaded layer across microbatches;
   measure the activation-versus-transfer tradeoff.
14. **E4B Tier-1 training.** Compile a measured BF16 schedule; train only after
    peak memory and recovery gates pass.
15. **Quantized transport.** Admit FP8 and other policies incrementally against
    the BF16 E4B baseline.
16. **Hybrid backward.** Add SSM/GDN and gated-attention VJPs only for a concrete
    Qwen training objective.
17. **12B scale-up.** Attempt RAM-resident or compressed Tier-1 training only
    after measured headroom exists.
18. **NVMe optimizer paging.** Last-resort experiment with explicit latency,
    write-volume and endurance budgets.

## 11. Model ladder

| Model or family | Near-term role | Initial tier | Admission condition |
|---|---|---:|---|
| Dense fixture | Numerical/device plumbing | VRAM | Forward/backward/update parity |
| Adaptive causal scratch profile | Construction/parity oracle | VRAM | Derived-fact, initialization and complete trajectory parity |
| Scratch workflow controller | Primary controller candidate | VRAM | Resident leadership, exact resume and multi-seed promotion suite |
| Fractale-350M, Carbon-500M or Qwen2.5-0.5B | Pretrained controller comparison | VRAM | Same corpus/objective/evaluation contract; compiled resident execution |
| SimpleDiffusion, Un-0, pocket-tts | Image and speech training validation | VRAM | Real modality corpus, processor/codec gradients and native-quality evaluation |
| MiniCPM5-1B | Resident scale validation | VRAM | Measured peak below safe capacity class |
| Gemma3n E4B | Tier-1 multimodal scale target | RAM offload | Device backward, checkpointing and every adaptive-declared trainable text/image/audio modality |
| Qwen3.5 4B/9B | Optional hybrid/multimodal training | RAM offload | Concrete text/image/video objective plus SSM/GDN and gated-attention VJPs |
| Gemma4 12B | Late scale target | RAM, optional NVMe | Measured state layout and safe host headroom |
| Wan, RxBrain, SenseNova, Krea | Serving/generation | N/A | No training work without an approved learning objective |
| TimesFM, TabFM, needle | Serving/evaluation | N/A | Add only with a concrete fine-tuning recipe and dataset |

Parameter counts and fit decisions come from compiled model inventory and measured
state schemas. This table expresses sequence, not authoritative byte counts.

## 12. Evidence and promotion gates

Every rung records:

- exact artifact and source revision identities;
- ordered input/target modality signature and applicable/refused matrix;
- processor, projector, merger, codec and augmentation identities;
- hardware, driver and capacity class;
- compiled program and calibration identities;
- peak VRAM and host-RAM usage;
- bytes transferred by direction and tier;
- forward, backward, recompute, transfer and optimizer timing;
- numerical parity and convergence measurements;
- checkpoint/resume evidence;
- modality-native evaluation results across required seeds;
- explicit pass, halt or rollback decision.

A result is not promoted when evidence is missing, stale, hardware-incompatible or
derived from mutable datasets. Performance claims compare the same model, data,
seed, precision, objective and stopping rule.

## 13. Non-goals for the first milestone

- Training every model that Overgo can serve.
- Pretraining a broad general-purpose foundation model; the scratch target is the
  bounded workflow controller.
- Treating inference quantization as proven training quantization.
- Hiding NVMe traffic behind theoretical peak-compute arithmetic.
- Mixing optimizer families or silently routing parameter shapes to another
  optimizer.
- Using model-family fallbacks outside compiled training authority.
- Accepting one seed, one batch or monotonic training loss as promotion evidence.

The plan succeeds first when the controller learning loop is trustworthy. Larger
models then validate that the same compiled contracts and evidence gates scale
through the memory hierarchy.
