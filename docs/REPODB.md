# RepoDB

RepoDB is the optional provenance and control plane. Runtime packages do not
depend on its storage implementation.

Current foundation:

- Kind-qualified SHA-256 artifact identities.
- Immutable artifact descriptors.
- Acyclic, typed lineage.
- Compare-and-set aliases.
- Canonical atomic batches with idempotency keys.
- Versioned CRC32C frames.
- Hash-chained commit identities and sequence validation.
- Torn-tail recovery without accepting complete corrupt frames.
- Fail-closed single-writer locking.
- Concurrent read/query safety and read-only store access.

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

Planned layers:

1. Typed recipe DAGs with named ports, schemas, and placement.
2. Dataset versions, assets, views, splits, and mixtures.
3. Run, evaluation, evidence, and promotion records.
4. Immutable segments plus versioned snapshots and rebuildable indexes.

The event log remains authoritative. Snapshots may accelerate replay but must
carry a verified commit-chain anchor and never discard required provenance.
