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
- Bounded inline content records for independently owned typed schemas.
- Canonical typed recipe DAGs persisted as content-addressed documents.
- Append-only candidate, validated, active, refused, and superseded lifecycle
  evidence with compare-and-set status and active-recipe aliases.
- Model recipe binding that compiles through the existing `ModelPlan` and
  `LayerPlan` authority.
- Canonical content-addressed architecture profiles seeded from the bootstrap
  registry, bound to exact recipes, and activated only with typed parity
  evidence.
- Runtime profile injection across metadata parsing, weight catalogs, and graph
  planning; unbound models retain the bootstrap path.

Storage layout:

```text
<root>/
  repodb.log
  repodb.lock
```

Bulk tensors, datasets, checkpoints, and generated media remain external.
RepoDB stores their identities, immutable facts, lineage, aliases, recipes,
evidence, and promotion decisions.

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

1. Generation, embedding, rerank, projector, and training module bindings.
2. Dataset versions, assets, views, splits, and mixtures.
3. Run, evaluation, evidence, and promotion records.
4. Immutable segments plus versioned snapshots and rebuildable indexes.

The event log remains authoritative. Snapshots may accelerate replay but must
carry a verified commit-chain anchor and never discard required provenance.
