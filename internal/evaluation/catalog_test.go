package evaluation

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
)

func TestBenchmarkCatalogResolvesPinnedLocalDatasets(t *testing.T) {
	t.Parallel()
	names := []string{"bbh", "ifeval", "math", "mmlu-pro", "musr", "truthfulqa"}
	root := t.TempDir()
	data := []byte("{\"input\":\"question\",\"target\":\"answer\"}\n")
	digest := sha256.Sum256(data)
	manifest := benchmarkManifest{Datasets: make([]benchmarkDeclaration, len(names))}
	for index, name := range names {
		path := name + ".jsonl"
		testcheck.NoError(t, os.WriteFile(filepath.Join(root, path), data, 0o600))
		manifest.Datasets[index] = benchmarkDeclaration{
			Name: name, Path: path,
			Spec: dataset.BenchmarkImportSpec{
				Source: name, Revision: "pinned-revision", SHA256: hex.EncodeToString(digest[:]), Split: "test",
				Format: dataset.BenchmarkFormatJSONL, Conversion: "fixture/v1",
				Fields: []dataset.FieldBinding{{Target: "answer", Source: "target"}, {Target: "prompt", Source: "input"}},
			},
		}
	}
	encoded := testcheck.Must(json.Marshal(manifest))(t)
	manifestPath := filepath.Join(root, "catalog.json")
	testcheck.NoError(t, os.WriteFile(manifestPath, encoded, 0o600))
	store := overgodbtest.Open(t, t.TempDir())
	id := testcheck.Must(CatalogLocalBenchmarks(t.Context(), store, manifestPath))(t)
	content, ok, err := artifact.ReadContent(t.Context(), store, id)
	if err != nil || !ok {
		t.Fatalf("catalog content = %v, %v", ok, err)
	}
	catalog := testcheck.Must(benchmarkCatalogCodec.Parse(content.Data))(t)
	if len(catalog.Entries) != len(names) {
		t.Fatalf("catalog entries = %d, want %d", len(catalog.Entries), len(names))
	}
	for index, entry := range catalog.Entries {
		if entry.Name != names[index] || entry.Split != "test" {
			t.Fatalf("catalog entry %d = %+v", index, entry)
		}
		importedContent, found, err := artifact.ReadContent(t.Context(), store, entry.Dataset)
		if err != nil || !found || importedContent.Descriptor.ID != entry.Dataset {
			t.Fatalf("dataset %q = %v, %v", entry.Name, found, err)
		}
	}
	active, ok, err := store.ResolveAlias(t.Context(), benchmarkCatalogAlias)
	if err != nil || !ok || active != id {
		t.Fatalf("active catalog = %s, %v, %v", active, ok, err)
	}
	repeated, err := CatalogLocalBenchmarks(t.Context(), store, manifestPath)
	if err != nil || repeated != id {
		t.Fatalf("repeated catalog = %s, %v", repeated, err)
	}
}
