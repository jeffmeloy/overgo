package repodbimport

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
)

const importFixtureCommit = "0123456789abcdef0123456789abcdef01234567"

func TestImportResolvesFilesDocumentsManifestsAndLineage(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "weights.bin"), []byte("weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	records := []wireRecord{
		{Type: "artifact", Name: "weights", Kind: "tensor-set", Path: "weights.bin"},
		{Type: "manifest", Name: "model", Kind: "model", Components: []componentRecord{{
			Role: "weights", Name: "weights", Artifact: "weights",
		}}},
		{Type: "artifact", Name: "definition", Kind: "model-definition",
			Document:  json.RawMessage(`{"version":1,"model":{"$artifact":"model"}}`),
			MediaType: "application/test+json", Schema: "test/v1"},
		{Type: "lineage", Child: "definition", Parent: "model", Relation: "depends-on"},
		{Type: "alias", Name: "models/active", Target: "definition"},
	}
	input := importStream(t, records, nil)
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	result, err := Import(context.Background(), store, root, bytes.NewReader(input))
	if err != nil {
		t.Fatal(err)
	}
	if !result.Commit.Valid() || result.Source.Kind() != artifact.KindEvidence || len(result.Names) != 3 {
		t.Fatalf("import result = %+v", result)
	}
	resolved, ok, err := store.ResolveAlias(context.Background(), "models/active")
	if err != nil || !ok || resolved != result.Names["definition"] {
		t.Fatalf("resolved import = (%s, %t, %v)", resolved, ok, err)
	}
	parents, err := store.Parents(context.Background(), result.Names["definition"])
	if err != nil || len(parents) != 2 {
		t.Fatalf("definition parents = (%+v, %v)", parents, err)
	}
}

func TestImportRejectsCountDriftPathEscapeAndReferenceCycle(t *testing.T) {
	records := []wireRecord{{Type: "artifact", Name: "weights", Kind: "file", Path: "../weights.bin"}}
	input := importStream(t, records, nil)
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := Import(context.Background(), store, t.TempDir(), bytes.NewReader(input)); err == nil {
		t.Fatal("escaping path accepted")
	}
	cycle := []wireRecord{
		{Type: "artifact", Name: "a", Kind: "evidence", Document: json.RawMessage(`{"ref":{"$artifact":"b"}}`), MediaType: "application/test+json", Schema: "test/v1"},
		{Type: "artifact", Name: "b", Kind: "evidence", Document: json.RawMessage(`{"ref":{"$artifact":"a"}}`), MediaType: "application/test+json", Schema: "test/v1"},
	}
	if _, err := Import(context.Background(), store, t.TempDir(), bytes.NewReader(importStream(t, cycle, nil))); err == nil {
		t.Fatal("reference cycle accepted")
	}
	counts := map[string]uint64{"kind:file": 2}
	if _, err := Import(context.Background(), store, t.TempDir(), bytes.NewReader(importStream(t, records, counts))); err == nil {
		t.Fatal("count drift accepted")
	}
}

func importStream(t *testing.T, records []wireRecord, counts map[string]uint64) []byte {
	t.Helper()
	if counts == nil {
		counts = make(map[string]uint64)
		for _, record := range records {
			counts[recordCountKey(record)]++
		}
	}
	header := Header{Type: "manifest", Version: Version, SourceCommit: importFixtureCommit, Counts: counts}
	var output bytes.Buffer
	encode := func(value any) {
		content, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		output.Write(content)
		output.WriteByte('\n')
	}
	encode(header)
	for _, record := range records {
		encode(record)
	}
	return output.Bytes()
}
