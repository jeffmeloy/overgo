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
	"overgo/internal/overgodb"
)

func TestBenchmarkCatalogResolvesPinnedLocalDatasets(t *testing.T) {
	names := []string{"bbh", "ifeval", "math", "mmlu-pro", "musr", "truthfulqa"}
	root := t.TempDir()
	data := []byte("{\"input\":\"question\",\"target\":\"answer\"}\n")
	digest := sha256.Sum256(data)
	manifest := benchmarkManifest{Datasets: make([]benchmarkDeclaration, len(names))}
	for index, name := range names {
		path := name + ".jsonl"
		if err := os.WriteFile(filepath.Join(root, path), data, 0o600); err != nil {
			t.Fatal(err)
		}
		manifest.Datasets[index] = benchmarkDeclaration{
			Name: name, Path: path,
			Spec: dataset.BenchmarkImportSpec{
				Source: name, Revision: "pinned-revision", SHA256: hex.EncodeToString(digest[:]), Split: "test",
				Format: dataset.BenchmarkFormatJSONL, Conversion: "fixture/v1",
				Fields: []dataset.FieldBinding{{Target: "answer", Source: "target"}, {Target: "prompt", Source: "input"}},
			},
		}
	}
	encoded, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	manifestPath := filepath.Join(root, "catalog.json")
	if err := os.WriteFile(manifestPath, encoded, 0o600); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	id, err := CatalogLocalBenchmarks(t.Context(), store, manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	content, ok, err := artifact.ReadContent(t.Context(), store, id)
	if err != nil || !ok {
		t.Fatalf("catalog content = %v, %v", ok, err)
	}
	catalog, err := benchmarkCatalogCodec.Parse(content.Data)
	if err != nil {
		t.Fatal(err)
	}
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
