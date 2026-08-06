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
  depend on and compile this definition without bootstrap-registry lookup.
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
  planning; unbound models retain the bootstrap path.
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
