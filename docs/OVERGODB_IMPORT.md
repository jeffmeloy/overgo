# OvergoDB import contract

`overgodb-import` consumes canonical legacy RepoDB JSONL from standard input. The producer may
live in another repository; overgo imports no producer packages and requires no
permanent source dependency.

The first line is the export manifest:

```json
{"type":"manifest","version":1,"source_commit":"0123456789abcdef0123456789abcdef01234567","counts":{"alias":1,"kind:model":1,"kind:tensor-set":1}}
```

Counts are exact. Artifact and manifest rows count under `kind:<kind>`;
lineage and alias rows count under their type. A mismatch rejects the stream.

Artifact rows choose exactly one source:

```json
{"type":"artifact","name":"weights","kind":"tensor-set","path":"models/weights.gguf"}
{"type":"artifact","name":"profile","kind":"profile","document":{"version":2},"media_type":"application/vnd.overgo.model-profile+json","schema":"overgo/model-profile/v2"}
```

Paths are relative to `-root`. Absolute, non-clean, and escaping paths fail.
The importer hashes the real file and records its resolved location. JSON
documents are canonicalized and hashed after logical references resolve:

```json
{"$artifact":"weights"}
```

An object containing only `$artifact` becomes the referenced kind-qualified
artifact ID. References and manifests resolve topologically; absent references
and cycles fail.

Manifest rows bind logical components:

```json
{"type":"manifest","name":"model","kind":"model","components":[{"role":"weights","ordinal":0,"name":"weights","artifact":"weights"}]}
```

Lineage and aliases use logical names:

```json
{"type":"lineage","child":"profile","parent":"model","relation":"depends-on"}
{"type":"alias","name":"models/active","target":"profile"}
```

Every line must be compact canonical JSON. The stream is bounded to 512 MiB,
one million records, and 64 MiB per line. Known overgo document media types are
parsed by their domain validators before commit; a media/schema mismatch fails.
The importer resolves all facts in memory and submits one normal atomic OvergoDB
batch. It also creates source evidence containing the producer commit, exact
per-kind counts, and SHA-256 of the complete JSONL stream, then links every
imported artifact to that evidence.

Example:

```bash
overgodb-import -repo /data/overgodb-store -root /data/adaptive-export < export.jsonl
```

Repeated exports are separate evidence-bearing batches. The source store may
remain a read-only answer key; no source log bytes enter the overgo log.
