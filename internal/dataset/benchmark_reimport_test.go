package dataset

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
)

// TestABenchmarkImportedAgainIsTheImportHeld imports a benchmark into a store that already holds its records and its
// import, committed under keys of an earlier writer's, as the store's own BBH records were (catalog-reimport/
// held-records): the import gives the one the store holds. go run ./cmd/evaluate -import-hf-cache failed at the first
// such dataset with no effective change, so a split added to a family (MMLU-Pro's validation split) never reached the
// catalog.
func TestABenchmarkImportedAgainIsTheImportHeld(t *testing.T) {
	t.Parallel()
	data := []byte("{\"query\":\"first\",\"label\":\"one\"}\n{\"query\":\"second\",\"label\":\"two\"}\n")
	path := filepath.Join(t.TempDir(), "records.jsonl")
	testcheck.NoError(t, os.WriteFile(path, data, 0o600))
	digest := sha256.Sum256(data)
	spec := BenchmarkImportSpec{
		Source: "benchmark/source", Revision: "source-revision", SHA256: hex.EncodeToString(digest[:]), Split: "test",
		Format: BenchmarkFormatJSONL, Conversion: "source-schema-to-canonical/v1",
		Fields: []FieldBinding{{Target: "answer", Source: "label"}, {Target: "prompt", Source: "query"}},
	}
	earlier := overgodbtest.Open(t, t.TempDir())
	first := testcheck.Must(ImportBenchmark(t.Context(), earlier, path, spec))(t)

	store := overgodbtest.Open(t, t.TempDir())
	for ordinal, id := range first.Records {
		record, found, err := benchmarkRecordCodec.Read(t.Context(), earlier, id)
		if err != nil || !found {
			t.Fatalf("record %d is unreadable: %v", ordinal, err)
		}
		batch := testcheck.Must(benchmarkRecordCodec.Batch("earlier/record/"+strconv.Itoa(ordinal), record, nil, nil))(t)
		testcheck.Must(artifact.CommitBatch(t.Context(), store, batch))(t)
	}
	held := testcheck.Must(benchmarkImportCodec.Batch("earlier/import", first, artifact.DependencyLineage(first.ID, first.Records...), nil))(t)
	testcheck.Must(artifact.CommitBatch(t.Context(), store, held))(t)

	again, err := ImportBenchmark(t.Context(), store, path, spec)
	if err != nil || again.ID != first.ID {
		t.Fatalf("the benchmark imported into a store holding it gives import %s (%v); want the import held, %s", again.ID, err, first.ID)
	}
}
