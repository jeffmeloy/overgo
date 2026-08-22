# Cross-Model Representation Bridging for Overgo

Research date: 2026-08-21

Scope: capability evaluation and incorporation design; no production implementation

Constraint: Go implementation using Overgo's own artifact, tensor, training, and execution systems; no Python, PyTorch, Candle, foreign runtime, or subprocess dependency

## Executive conclusion

Overgo does not lack every form of structural model composition. It already has:

- model-specific vision and audio towers that project media features into language-model inputs;
- exact hidden-state and attention-boundary capture for supported compiled layer programs;
- architecture-declared per-layer side inputs and alternate-state injection;
- paired-model component sessions with explicit model slots and lifetimes;
- artifact-declared residual-adapter training; and
- a real experimental graft that inserts a frozen donor MLP into a frozen target model through trainable width-changing matrices.

The missing capability is narrower and more consequential: **a general, artifact-declared representation interface and bridge graph that can connect two independently loaded model artifacts without either model having been authored as a predefined multimodal architecture**.

The recommended first deliverable is not arbitrary layer stitching. It is a generic encoder-to-decoder bridge with three initial operators:

1. normalized two-layer GELU projection;
2. AlignVLM-style language-embedding-basis projection; and
3. fixed-latent Perceiver resampling.

The first injection boundary should be target input embeddings. Interleaved gated cross-attention should follow only after the bridge artifact, masks, positions, cache interaction, and frozen-gradient invariants are proven. Tensor-conditioned LoRA should be a later research track. Offline weight merging and layer stacking should remain a separate capability because they have fundamentally different compatibility and failure rules.

## Correcting the capability description

Several statements in the supplied capability description are directionally useful but too broad for a system design decision.

- A linear or MLP projector can handle both width mismatch and unchanged sequence length, but it does not by itself establish semantic alignment. That comes from the training data, objective, and frozen/trainable policy.
- A Perceiver resampler is appropriate when the source sequence is long or variable and the target needs a bounded token budget. It is not automatically better than pooling or an MLP for small, already aligned sequences.
- AlignVLM's distinctive connector is not merely a multi-stage MLP. It predicts a distribution over the target language vocabulary and maps that distribution through the frozen language embedding table, constraining bridge outputs to the target embedding basis.
- `representation-engineering` and `ster` demonstrate activation capture and additive steering. They do not connect one model's representation space to another model's space.
- `mergekit` passthrough copies selected tensors/layers. It does not learn an interface between incompatible hidden widths, normalization conventions, attention layouts, or residual semantics. Its default behavior rejects differing known architectures unless an explicitly dangerous override is supplied.
- Tokenizer union or embedding fallback makes an output vocabulary mechanically constructible; it does not make unrelated token embeddings or hidden representations semantically equivalent.
- The supplied claim about a Perceiver bridge in Flamingo is consistent with the inspected architecture catalog. The local evidence inspected here does not establish the broader claim about early Gemini, so this report does not rely on it.

## What Overgo has today

### Model-specific multimodal projection

`internal/projector` contains substantial vision and audio implementations. Its catalog validates exact tensor names and shapes, including standard two-layer projector tensors (`mm.0` and `mm.2`) for several supported families. `MultimodalPrompt` can carry replacement embeddings, deep-stack embeddings, selected token indices, multi-axis positions, and attention/visual blocks.

This is strong execution machinery, but it is intentionally model-specific. `OpenActiveAs` requires the loaded projector artifact to match the active projection recipe and the descriptor's exact supported modalities. There is no neutral contract saying that an arbitrary model A emits a representation which an arbitrary model B accepts.

### Hidden-state capture and declared injection paths

`internal/inference/layer_input_capture.go` exposes full-sequence pre-layer rows and exact attention replay boundaries for compiled architectures that declare capture support. Other paths support paired inputs, deep-stack features, and hidden-state sessions.

`internal/inference/alternate_state.go` implements a specialized architecture's predicted/corrected state flow and per-layer injection. `internal/model/block.go` also executes architecture-declared per-layer input gates and projections. These prove that Overgo can represent capture and injection in its compiled runtime, but the seams are selected by model policy rather than by a general external bridge contract.

### Artifact-declared adapter training

`internal/adaptertrain/model.go` trains per-layer gate, projection, and normalization tensors against examples containing an input, side representation, and target. Model-provided inputs remain fixed and only the declared adapter tensors are trainable. This is close to the training discipline a generic bridge needs, but it only consumes the per-layer adapter geometry already present in a supported model artifact.

### Actual cross-model grafting

`internal/densecausal/graft.go` computes, at one target layer:

```text
y = x + Up * donorMLP(Down * x)
```

The target and donor MLP are frozen. `Down` maps target hidden width to donor hidden width, `Up` maps back, and only those matrices receive gradients. `Up` is zero-initialized, making the untrained graft exactly equal to the baseline target. `internal/composition/viability.go` binds the target and donor to separate recipe model slots, selects one donor MLP, trains across requested seeds, evaluates a held-out batch, and refuses promotion unless the worst seed beats baseline.

This is genuine weight-function grafting across width differences. It is still a narrow host-side experiment for dense causal safetensors models. The donor does not independently encode another modality or produce an external sequence; the branch invokes one donor MLP on a projection of the target residual stream. It therefore does not solve generic encoder-to-decoder representation bridging.

### Reusable infrastructure

The following foundations should be reused rather than replaced:

- content-addressed artifact identity and dependency lineage;
- recipe dependencies with independently bound model slots;
- compiled `ComponentSessionPlan` lifetimes and resource extents;
- compiled immutable model/layer programs;
- host and CUDA tensor graphs and differential tests;
- frozen/trainable tensor declarations and optimizer plans;
- exact checkpoint/resume authority; and
- multi-seed, held-out, fail-closed promotion from composition viability.

## Reference repository findings

All findings below refer to the inspected local checkout and commit, not to an assumed upstream state.

| Repository | Inspected commit | What the code actually contributes | What it does not establish |
|---|---:|---|---|
| `dlimp_openvla` | `040105d` | A TensorFlow `tf.data`/RLDS trajectory loading and transformation layer useful as a VLA dataset-schema reference. | No model merger, independent tower attachment, projector, Perceiver, representation bridge, or Go-runtime design. |
| `MergeVLA` | `5ef0e0d` | Same-base cross-skill VLA merging: component decomposition, task arithmetic/masks for LoRA drift, routing, preservation of nonmergeable action experts, and ordinary projection modules for vision/proprioception/actions. | Not arbitrary cross-architecture or cross-modality fusion. Its experts share a Qwen2.5 base. A Perceiver class exists in utilities but no call site was found in the inspected main tree, so it is not evidence of the core method. |
| `AlignVLM` | `47f1893` | Frozen vision/LLM alignment, ordinary configurable MLP projectors, an embedding-basis Align connector, a learned-latent Perceiver resampler, and independently tunable tower/projector/resampler/LLM stages. | Not a generic systems runtime. The implementation is coupled to the Python/Hugging Face/LLaVA training stack and the README describes a low-resource implementation. |
| `awesome-vlm-architectures` | `74897b3` | A broad citation and architecture taxonomy that locates linear-projector, Q-Former, Perceiver, and gated cross-attention designs. | No executable engine, interoperability contract, training implementation, or validation evidence. It should route research to primary sources, not serve as executable specification. |
| `representation-engineering` | `5455d8a` | Hidden-state capture, layer/token selection, learned directions (including PCA and mean differences), masks, additive control, and optional norm preservation. | No cross-model mapper. Runtime control relies on Hugging Face model wrappers and model-structure assumptions; one advertised projection operator is unimplemented in the inspected code. |
| `ster` | `7f808a60` | A compact native activation-steering contract: last-token layer capture, CAA/PCA/logistic directions, held-out selection, immutable JSON artifacts, strict model/width/layer validation, and additive residual steering in an owned decoder loop. | It supports only Llama-family safetensors in its current contract, captures only the last token for training, and adds fixed vectors rather than another model's transformed sequence. It uses Rust/Candle, which is reference behavior rather than an incorporable runtime dependency. |
| `mergekit` | `a6e4028` | Offline tensor algebra, architecture catalogs, slices, passthrough layer copying, vocabulary construction, embedding fallback, and tokenizer transplantation. | Passthrough accepts exactly one source tensor per output tensor and learns no boundary adapter. Known differing architectures are rejected by default. The dangerous override can attempt a merge but cannot create missing semantic or geometric compatibility. |

### AlignVLM details worth reproducing in Go

The Align connector in `llava/model/multimodal_projector/align_connector.py` performs approximately:

```text
z = Linear(source_width -> target_width)(source)
z = ReLU(LayerNorm(z))
token_logits = LayerNorm(Linear(target_width -> target_vocab)(z))
p = Softmax(token_logits)
bridge_output = LayerNorm(p * target_embedding_table)
```

Its preparation script initializes the vocabulary projection from the target language model's `lm_head`. This gives the connector a useful inductive bias: before final normalization, every output row is a probability-weighted mixture of actual target token embeddings. The tradeoff is material: the vocabulary-sized logits and softmax are much more expensive than an `Hs -> Hb -> Ht` MLP, especially for large vocabularies and long source sequences.

The repository's Perceiver resampler uses learned latent queries and repeated cross-attention/feed-forward blocks to reduce variable source sequences to a fixed latent count. The independently frozen/tunable tower, projector, resampler, and language-model switches are a good basis for Overgo's trainability policy, but not for its runtime API.

### MergeVLA details worth retaining

MergeVLA's core contribution to this design is not arbitrary model fusion. It is the discipline of splitting a composite artifact into parts with different merge rules:

- merge only parameters that share a valid base and geometry;
- route or mask task-specific low-rank changes rather than averaging all of them;
- retain structurally nonmergeable experts as separately selected components; and
- initialize newly introduced paths so the base behavior remains defined.

That maps well to an Overgo offline same-base merge tool, but not to the generic representation bridge runtime.

### Activation-control details worth retaining

`representation-engineering` makes layer, token, mask, operator, and norm policy explicit, while `ster` makes artifact/model/width/layer compatibility fail closed. Combined, they suggest the right Overgo control contract:

```text
capture point + tensor layout + selector + immutable control/bridge artifact
+ injection operator + coefficient/gate + magnitude policy
```

Overgo should compile those facts into its layer program. It should not reproduce Python-style runtime monkey-patching or Rust/Candle model ownership.

## Required new abstractions

### 1. Representation contract

A representation must be more than a float array and hidden width. A versioned, content-addressed `RepresentationContract` should include:

```text
Producer
  model artifact ID and resolved model-definition ID
  output/tap point: embedding, encoder final, layer residual, attention input, etc.
  layer index and whether the state is pre- or post-residual/normalization

Tensor
  dtype, rank, dimensions or bounded dynamic axes
  logical layout: batch, sequence, channel, spatial, temporal
  channel width and sequence/token limits
  normalization convention and expected magnitude policy

Sequence semantics
  validity mask and padding rule
  position convention and axes
  pooling/CLS/BOS/EOS treatment
  modality and sampling/patch/frame metadata

Authority
  tokenizer/processor/projector IDs where applicable
  exact source artifact revision and tensor inventory
```

Width equality alone must never authorize injection. Two `[tokens, 4096]` tensors may have incompatible normalization, position, token, residual, or semantic conventions.

### 2. Bridge definition and weights

A `BridgeDefinition` should bind exact source and target representation contracts and declare an ordered operator graph. It should record:

- source and target model slots;
- accepted source contract ID and produced target contract ID;
- operator types, dimensions, masks, normalization, activation, and residual/gate policy;
- target injection point(s);
- trainable versus frozen tensor names;
- initialization policy, including an exact no-op path where possible;
- objective, dataset/split, optimizer, precision, and promotion-policy dependencies; and
- resource bounds such as maximum source tokens, latent count, target injections, and vocabulary projection size.

Bridge weights should be ordinary Overgo tensor artifacts with a validated inventory. A checkpoint must bind the definition, both model artifacts, processors/tokenizers, dataset split, RNG counters, optimizer plan, and every trainable tensor. Loading must reject any mismatch.

### 3. Runtime representation batch

The internal value passed between independently loaded components should be a typed batch, conceptually:

```go
type RepresentationBatch struct {
    Contract artifact.ID
    Values   tensor.Value
    Mask     tensor.Value
    Position PositionEncoding
    Metadata BoundedMetadata
}
```

This is a conceptual interface, not a proposed public API. The important point is that mask and positional semantics travel with values and are checked against the bridge contract.

### 4. Compiled bridge graph

The bridge should compile to the same tensor IR used by Overgo's model execution. An initial sealed operator catalog is preferable to arbitrary callbacks:

- `normalize`: RMSNorm or LayerNorm with explicit epsilon;
- `linear` and `mlp_gelu`: one or more validated dense projections;
- `pool`: bounded mean/max/spatial pooling baseline;
- `perceiver_resample`: learned latents, source cross-attention, FFN, fixed output count;
- `align_vocab`: vocabulary logits, softmax, and multiplication by the exact target embedding table;
- `residual_gate`: scalar/per-channel gated addition with declared initialization; and
- later, `cross_attention` and `conditional_lora_scale`.

Keeping this catalog sealed preserves compile-time shape checking, memory planning, host/CUDA parity, and artifact reproducibility.

### 5. Injection contract

Initial support should admit only target input-embedding insertion or replacement. A later layer-injection contract needs all of:

- exact target layer and stage within the compiled layer program;
- whether the input is normalized and by which norm;
- attention query versus key/value source;
- residual add/replace/concatenate semantics;
- token selector and mask behavior;
- position and cache policy during prompt ingestion and autoregressive continuation;
- gate initialization and magnitude/norm policy; and
- proof that the target architecture exposes that seam.

Unknown injection stages must fail during recipe compilation, before model weights are loaded.

## Bridge algorithms

### Normalized MLP projection

Recommended first operator:

```text
B = Norm_out(W2 * GELU(W1 * Norm_in(A) + b1) + b2)
```

It is cheap, maps arbitrary widths, fits Overgo's existing tensor operations, and provides the simplest end-to-end training test. It does not reduce sequence length. Its acceptance bar must be a held-out gain over linear projection, pooled projection, and an untrained/no-media baseline--not merely decreasing training loss.

### Align embedding-basis projection

Recommended as an alternative target head when target vocabulary size is within configured resource limits:

```text
p = softmax(VocabHead(Transform(A)))
B = Norm(p * E_target)
```

The target embedding table and vocabulary are exact dependencies. Tied and untied embedding/LM-head cases must be explicit. Full-vocabulary logits may dominate memory, so the Go implementation should first support bounded chunked vocabulary multiplication/softmax or a declared vocabulary subset. Any approximation must be part of the artifact identity.

### Perceiver resampler

Recommended second sequence operator:

```text
Q0 = learned_latents[L, Hb]
Qi+1 = Qi + CrossAttention(Norm(Qi), Norm(A), source_mask)
Qi+1 = Qi+1 + FFN(Norm(Qi+1))
B = Project(QN -> target_width)
```

The fixed latent count `L`, head geometry, number of blocks, position/media embeddings, and masking rules belong in the bridge definition. Source key/value projections can map from source width without first materializing a target-width source sequence. The output token budget is fixed and can be inserted into the target prompt.

### Interleaved gated cross-attention

Later-stage operator:

```text
x_l = target decoder state at admitted layer l
m   = cached bridged source representation
x'_l = x_l + tanh(g_l) * CrossAttention(Norm(x_l), K(m), V(m))
```

Zero-initialized gates provide an exact target-only baseline. Source K/V should be computed once per request and cached. Every admitted decoder layer affects cache scheduling and memory, so this is not just another projector. The model plan needs an explicit external cross-attention stage; arbitrary insertion into existing blocks would undermine Overgo's compiled-program guarantees.

This mechanism is one-way cooperation--target queries source memory. Bidirectional alternating communication would require re-running or recurrently updating the source and should not be implied by the initial feature.

### Tensor-conditioned LoRA

Treat this as a high-risk research feature. Generating complete LoRA matrices per request is expensive, complicates batching and caching, and creates input-dependent weights that current artifact identities do not fully describe. A safer first experiment is fixed low-rank bases with source-conditioned scales:

```text
delta(x, m) = B * diag(s(m)) * A * x
```

Here `A` and `B` are trained bridge tensors and `s(m)` is a bounded vector produced from pooled source representation `m`. The artifact remains fixed while per-request state is limited to scales. Even this needs strict range controls, batching rules, deterministic reduction, and a no-op initialization.

### Offline weight merging and layer stacking

This should be a different recipe task and executable path. Safe initial scope:

- exact same model definition or an explicitly declared tensor-name conversion;
- identical tensor shapes for arithmetic methods;
- exact common base identity for task-vector methods;
- tokenizer/embedding handling recorded as a separate transformation;
- passthrough slices only when the output model plan validates every boundary and tensor inventory; and
- mandatory evaluation of the resulting artifact as a new model.

Do not market an `allow incompatible architectures` flag as representation alignment. If adjacent copied layers expose different widths, attention layouts, normalization, or residual semantics, a trained runtime bridge is required; copying cannot repair the interface.

## Go-native execution design

No reference repository should be imported as a runtime dependency. The mathematical behavior should be reimplemented using Overgo's Go packages and tensor IR.

Proposed package boundaries:

```text
internal/representation   versioned contracts, batches, tap/injection enums
internal/bridgeartifact   definitions, tensor inventories, lineage, checkpoints
internal/bridgegraph      validation and compilation of sealed bridge operators
internal/bridgetrain      frozen-source/frozen-target training and objectives
internal/inference        admitted capture/injection execution seams
internal/modelrecipe      composite recipe compilation and component lifetimes
internal/workflowrecipe   encode, bridge, inject, train, evaluate modules
```

A request would execute as:

```text
source input
  -> source model slot: encode/capture exact RepresentationContract A
  -> compiled bridge graph: A -> B
  -> target model slot: inject B at an admitted boundary
  -> target generation/evaluation
```

The component session director should load and lease source, target, and bridge artifacts according to recipe lifetimes and byte extents. The runtime must not guess source/target models from filenames or inspect arbitrary Python configuration at execution time.

For training, source and target weights remain read-only. Gradients flow through the target activation path into bridge tensors and, when necessary, back through bridge operations to their trainable inputs, but no optimizer slot may exist for frozen model tensors. Initial training can use host reference operations; production readiness requires the bridge graph to use the shared tensor builder so the same definition executes on supported host/CUDA paths.

## Recommended delivery sequence

### Rung 0 -- contracts, capture, and replay

Deliver a representation artifact contract and exact capture/replay fixtures before training a bridge.

- Export full-sequence states, masks, positions, model/definition IDs, tap point, and dtype.
- Add additive fixed-vector control as a narrow injection operator to validate the seam, following `ster`'s strict artifact checks and the selector/norm ideas in `representation-engineering`.
- Prove identity replay and deterministic serialization.

This rung converts existing hidden-state capture into a safe external contract without claiming cross-model alignment.

### Rung 1 -- generic embedding-boundary MLP bridge

- Two independently bound model slots.
- Frozen source encoder and target decoder.
- Linear and normalized two-layer GELU bridge variants.
- Input-embedding insertion/replacement only.
- Exact no-media and zero-gated baseline.
- Bridge-only training and held-out multi-seed promotion.

This is the minimum useful cross-model capability.

### Rung 2 -- Align and Perceiver operators

- Add embedding-basis alignment with explicit target vocabulary authority and resource cap.
- Add masked fixed-latent Perceiver resampling.
- Validate long/variable audio or image sequences and fixed target token budgets.
- Compare quality, memory, latency, and bridge parameter count against pooling and MLP baselines.

### Rung 3 -- interleaved gated cross-attention

- Extend only explicitly supported target layer programs with external cross-attention stages.
- Cache source K/V per request.
- Start with every-N-layer placement and zero-initialized scalar gates.
- Prove prompt/generation cache correctness, multi-request isolation, and target-only identity.

### Rung 4 -- conditional low-rank adaptation

- First test source-conditioned scales over fixed low-rank bases.
- Do not generate arbitrary per-request full adapter matrices initially.
- Require strong ablations against cross-attention and static adapters before productization.

### Separate track -- same-base offline merging

Port only clearly specified algorithms needed by Overgo, using its tensor artifact reader/writer. Begin with weighted average, task arithmetic, TIES-like masking, and passthrough under exact architecture/tensor compatibility. Keep this track out of the representation-bridge runtime.

## Validation program

### Contract and shape tests

- Reject wrong source/target artifact, model definition, tap point, layer, width, dtype, rank, mask layout, position convention, processor, tokenizer, and vocabulary.
- Bound dynamic sequence, vocabulary, latent, head, and layer counts before allocation.
- Reject unrecognized operators and tensor names.
- Prove serialized definition and checkpoint IDs are stable and content-addressed.

### Numerical tests

- Hand-computed fixtures for normalization, MLP, attention, masking, vocabulary mixture, and conditional low-rank scale.
- Finite-difference gradients for every trainable bridge tensor.
- Host/CUDA forward and gradient differential tests within declared tolerances.
- Deterministic attention masking and chunked-vocabulary equivalence.

### Isolation and identity tests

- Hash all source and target tensors before and after training; require bit identity.
- Require zero-gated cross-attention and conditional LoRA to reproduce target-only logits exactly or within an explicitly justified backend tolerance.
- Verify no optimizer state is allocated for frozen tensors.
- Verify one request's source cache or conditional scales cannot affect another request.

### Behavioral experiments

Each approach should be evaluated against cheap baselines, not only against the unmodified target:

| Experiment | Baselines | Required measurements |
|---|---|---|
| Vision encoder -> text decoder | mean pool + linear; two-layer MLP; target-only | held-out CE/task score, bridge parameters, source/target token counts, latency, peak memory |
| Audio encoder -> text decoder | temporal pool + linear; MLP; target-only | transcription/task score, robustness across duration, fixed-latent scaling, latency/memory |
| Align head | ordinary MLP; randomly initialized vocabulary head | quality, vocabulary-softmax cost, embedding-neighbor behavior, calibration |
| Perceiver | mean/spatial/temporal pooling; equal-token MLP | quality versus latent count, source-length scaling, mask robustness |
| Interleaved cross-attention | input-only best bridge; equal-parameter adapter | gain by injection layer/frequency, cache cost, generation throughput |
| Conditional LoRA scale | static LoRA; cross-attention; pooled MLP | gain, stability, batch fragmentation, scale distribution, adversarial source sensitivity |

Promotion should preserve the existing composition standard: multiple seeds, disjoint held-out data, finite outputs, and refusal when the worst accepted seed does not beat the declared baseline. For practical model work, also require task-specific metrics and regression suites; a loss delta alone is insufficient.

## Principal risks

1. **Semantic compatibility is not inferable from shapes.** Contracts can prevent mechanical misuse but cannot prove that training learned a useful alignment.
2. **Position and mask errors can look plausible.** These are more dangerous than obvious shape errors because they may produce fluent but incorrectly grounded outputs.
3. **Bridge training may exploit the target language prior.** Ablations must demonstrate that outputs actually depend on source representations.
4. **Cross-attention changes cache and memory economics.** Source K/V and per-layer projections can dominate long generation sessions.
5. **Vocabulary alignment is expensive.** AlignVLM-style full-vocabulary projection can be prohibitive without bounded/chunked execution.
6. **Dynamic adapters fragment batching.** Per-request weights or scales reduce reuse and complicate deterministic scheduling.
7. **Architecture-name compatibility is insufficient for weight merging.** Exact base, tensor inventory, conversion, and output-plan validation are required.
8. **Research code is not production evidence.** Most inspected repositories assume Python/Hugging Face; even the native Rust reference is intentionally limited to Llama.

## Licensing and provenance

This report recommends independent mathematical reimplementation, not source translation.

- `AlignVLM`: Apache-2.0 top-level license.
- `awesome-vlm-architectures`: CC0-1.0 top-level license; its README separately warns about rights in architecture images, which are unnecessary here.
- `representation-engineering`: MIT top-level license.
- `ster`: MIT top-level license.
- `mergekit`: LGPL-3.0-only in the inspected source headers/top-level license. Treat code copying or close translation as a legal review item; behavioral reimplementation from documented algorithms is preferable.
- `dlimp_openvla` and `MergeVLA`: no top-level license file was found in the inspected checkouts. Do not copy code from them without separate permission/provenance confirmation.

Any imported pretrained model, connector checkpoint, tokenizer, or dataset also has its own terms and must be represented as external artifact provenance rather than inheriting the repository's code license.

## Decision

Proceed with a general representation contract and Go-native bridge graph, beginning with embedding-boundary MLP, Align, and Perceiver operators. Reuse Overgo's artifact lineage, recipes, component session director, tensor graph, adapter-training discipline, and fail-closed composition evaluation.

Do not begin with arbitrary cross-family layer stacking, model monkey-patching, or per-request generated LoRA matrices. Those paths combine the highest semantic risk with the weakest current validation story. Keep compatible offline weight merging separate from runtime representation fusion.

## Inspected evidence index

### Overgo

- `internal/composition/viability.go`
- `internal/densecausal/graft.go`
- `internal/projector/catalog.go`
- `internal/projector/prompt.go`
- `internal/inference/layer_input_capture.go`
- `internal/inference/alternate_state.go`
- `internal/model/block.go`
- `internal/model/layer_plan.go`
- `internal/adaptertrain/model.go`
- `internal/modelrecipe/compile.go`
- `internal/modelrecipe/session_resources.go`
- `internal/capabilityruntime/session_director.go`

### Local reference checkouts

- `C:\Users\jeffm\dlimp_openvla\README.md`
- `C:\Users\jeffm\MergeVLA\model_merging\mergy.py`
- `C:\Users\jeffm\MergeVLA\prismatic\extern\hf\modeling_prismatic.py`
- `C:\Users\jeffm\MergeVLA\prismatic\models\projectors.py`
- `C:\Users\jeffm\MergeVLA\prismatic\models\transformer_utils.py`
- `C:\Users\jeffm\AlignVLM\llava\model\multimodal_projector\align_connector.py`
- `C:\Users\jeffm\AlignVLM\llava\model\multimodal_projector\builder.py`
- `C:\Users\jeffm\AlignVLM\llava\model\multimodal_resampler\perceiver.py`
- `C:\Users\jeffm\AlignVLM\scripts\prepare_align_weights.py`
- `C:\Users\jeffm\awesome-vlm-architectures\README.md`
- `C:\Users\jeffm\representation-engineering\repe\rep_reading_pipeline.py`
- `C:\Users\jeffm\representation-engineering\repe\rep_control_pipeline.py`
- `C:\Users\jeffm\representation-engineering\repe\rep_control_reading_vec.py`
- `C:\Users\jeffm\representation-engineering\repe\rep_control_contrast_vec.py`
- `C:\Users\jeffm\ster\src\artifact.rs`
- `C:\Users\jeffm\ster\src\model.rs`
- `C:\Users\jeffm\ster\src\representation.rs`
- `C:\Users\jeffm\ster\src\runtime.rs`
- `C:\Users\jeffm\mergekit\mergekit\merge_methods\passthrough.py`
- `C:\Users\jeffm\mergekit\mergekit\architecture\__init__.py`
- `C:\Users\jeffm\mergekit\mergekit\architecture\auto.py`
- `C:\Users\jeffm\mergekit\mergekit\tokenizer\build.py`
