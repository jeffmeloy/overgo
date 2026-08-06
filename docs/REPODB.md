# RepoDB

RepoDB is the optional provenance and control plane. Runtime packages do not
depend on its storage implementation.

Current foundation:

- Kind-qualified SHA-256 artifact identities.
- Immutable artifact descriptors.
- Canonical multi-file manifests with typed component roles and slots.
- Acyclic, typed lineage.
- Compare-and-set aliases.
- Path-independent identity with append-only location availability.
- Canonical atomic batches with idempotency keys.
- Versioned CRC32C frames.
- Hash-chained commit identities and sequence validation.
- Torn-tail recovery without accepting complete corrupt frames.
- Fail-closed single-writer locking.
- Concurrent read/query safety and read-only store access.
- GGUF single/split and Hugging Face Safetensors inventory adapters.
- Canonical logical tensor inventories with ordered names, shapes, storage, and
  payload sizes, derived from model manifests and independent of paths, shard
  offsets, and runtime tensor loading.
- Canonical model definitions binding an exact model manifest, architecture
  profile, tensor inventory, and validated runtime metadata spec. Recipe v2 can
  depend on and compile this definition without bootstrap-registry lookup. One
  batch atomically publishes the physical model inventory and resolved facts.
- Bounded inline content records for independently owned typed schemas.
- Canonical typed recipe DAGs persisted as content-addressed documents.
- Recipe v2 identities include a canonical role/slot dependency set for models,
  profiles, tokenizers, projectors, adapters, datasets, and checkpoints; v1
  documents remain readable.
- Append-only candidate, validated, active, refused, and superseded lifecycle
  evidence with compare-and-set status and active-recipe aliases.
- Model recipe binding that compiles through the existing `ModelPlan` and
  `LayerPlan` authority.
- Canonical content-addressed architecture profiles seeded from a strict data
  catalog, embedded in exact recipe identities, and activated only with typed
  parity evidence. RepoDB may publish replacement catalogs without mutating the
  bootstrap fallback; mutable profile side aliases are legacy-read-only.
- Runtime profile injection across metadata parsing, weight catalogs, and graph
  planning. Dispatch, catalog loading, and persistent-cache policy preserve the
  exact bound profile; unbound models retain the bootstrap path.
- Typed runtime profile policies own input/logit scaling, normalization
  placement/bias/fallback, and layer-level RoPE schedules; model methods retain
  only the reusable execution mechanisms.
- Typed validation selectors own base rotary and encoder-family invariants;
  shared validators consume the resolved policy without architecture-name
  dispatch.
- Thirty-two attention validation contracts now serialize with profiles,
  covering rotary, sliding, per-layer, multimodal, and expert-shape invariants
  without compiled architecture-name selection.
- MLA/DSA validation selectors now carry DeepSeek, Kimi Linear, Mistral, and
  MiniCPM contracts; shared MLA mechanisms consume only typed profile policy.
- Recurrent validation selectors now carry WavTokenizer, draft, Mamba, RWKV,
  hybrid SSM, and Nemotron contracts with named dimensional facts.
- Hybrid validation selectors now carry Qwen, GroveMoE, MiMo2, and Step3.5
  scheduling, rotary, and expert contracts.
- Thirty-six hybrid/MoE contracts now cover dense/MoE variants, shared experts,
  routing, scaling, position policy, and hybrid convolution schedules without
  compiled architecture-name selection.
- Full-indexer scheduling now travels in layer cadence policy. Production
  `Spec` and shared validators consume resolved policy without name predicates.
- Specialized executable guards consume typed forward policy, and speculative
  sidecars require exact resolved-profile equality instead of name equality.
- Canonical generation, embedding, rerank, image/audio/video projection, and
  training orchestration DAGs sharing one module catalog and topological plan
  compiler. Runtime-backed and orchestration-only plans are distinguished.
- Immutable successful/failed/cancelled run records, finite typed evaluation
  metrics, and queryable input/output/dataset provenance. Evaluation identities
  can serve directly as recipe promotion evidence.

Storage layout:

```text
<root>/
  repodb.log
  repodb.lock
```

Bulk tensors, datasets, checkpoints, and generated media remain external.
RepoDB stores their identities, immutable facts, logical tensor inventories,
resolved model definitions, lineage, aliases, recipes, evidence, and promotion
decisions.

Dependency direction:

```text
artifact contracts <- recipe compiler <- runtime adapters
        ^
        |
RepoDB persistence
```

The model adapters hash exact component bytes after the existing loaders have
validated their formats. Logical names and roles enter manifest identity;
absolute paths enter only the mutable location set. Copying an unchanged model
repository therefore preserves its manifest identity.

Planned layers:

1. Dataset versions, assets, views, splits, and mixtures.
2. Runtime adapters that execute compiled workflow steps and emit run records.
3. Immutable segments plus versioned snapshots and rebuildable indexes.

The event log remains authoritative. Snapshots may accelerate replay but must
carry a verified commit-chain anchor and never discard required provenance.
