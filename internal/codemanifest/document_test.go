package codemanifest

import (
	"bytes"
	"encoding/json"
	"slices"
	"testing"
)

const fixtureDigest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

func TestSchemaCanonicalIdentity(t *testing.T) {
	first, err := codec.New(fixtureManifest())
	if err != nil {
		t.Fatal(err)
	}
	reordered := fixtureManifest()
	slices.Reverse(reordered.Files)
	slices.Reverse(reordered.Symbols)
	slices.Reverse(reordered.References)
	reordered.BuildContexts[0].Tags = []string{"cuda", "cuda"}
	second, err := codec.New(reordered)
	if err != nil {
		t.Fatal(err)
	}
	if first.ID != second.ID {
		t.Fatalf("canonical IDs differ: %s != %s", first.ID, second.ID)
	}
	content, err := first.Content()
	if err != nil {
		t.Fatal(err)
	}
	replayed, err := codec.Parse(content.Data)
	if err != nil {
		t.Fatal(err)
	}
	if replayed.ID != first.ID {
		t.Fatalf("replayed ID = %s, want %s", replayed.ID, first.ID)
	}
}

func TestStrictDecode(t *testing.T) {
	document, err := codec.New(fixtureManifest())
	if err != nil {
		t.Fatal(err)
	}
	content, err := document.Content()
	if err != nil {
		t.Fatal(err)
	}
	unknown := bytes.Replace(content.Data, []byte(`"version":1`), []byte(`"version":1,"surprise":true`), 1)
	if _, err := codec.Parse(unknown); err == nil {
		t.Fatal("unknown field was accepted")
	}
	var raw map[string]any
	if err := json.Unmarshal(content.Data, &raw); err != nil {
		t.Fatal(err)
	}
	pretty, err := json.MarshalIndent(raw, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := codec.Parse(pretty); err == nil {
		t.Fatal("non-canonical encoding was accepted")
	}
}

func TestSchemaRejectsInvalidRelations(t *testing.T) {
	value := fixtureManifest()
	value.References[0].To.Name = "absent"
	if _, err := codec.New(value); err == nil {
		t.Fatal("reference to absent symbol was accepted")
	}
	value = fixtureManifest()
	value.Files[0].SelectedContexts = []string{"unknown"}
	if _, err := codec.New(value); err == nil {
		t.Fatal("unknown build context was accepted")
	}
}

func fixtureManifest() Manifest {
	caller := SymbolID{Package: "overgo/internal/example", Name: "Run", Kind: SymbolFunction}
	callee := SymbolID{Package: "overgo/internal/example", Receiver: "Worker", Name: "Apply", Kind: SymbolMethod}
	return Manifest{
		Version: Version, SourceIdentity: fixtureDigest,
		Analyzer:      Analyzer{Name: "overgo-code-profile", Version: "fixture-v1"},
		BuildContexts: []BuildContext{{ID: "windows/amd64+cgo+cuda", GOOS: "windows", GOARCH: "amd64", Tags: []string{"cuda"}, Cgo: true}},
		Files: []File{
			{Path: "internal/example/worker.go", ContentID: fixtureDigest, Package: "overgo/internal/example", SelectedContexts: []string{"windows/amd64+cgo+cuda"}},
			{Path: "internal/example/run.go", ContentID: fixtureDigest, Package: "overgo/internal/example", SelectedContexts: []string{"windows/amd64+cgo+cuda"}},
		},
		Symbols: []Symbol{
			{ID: caller, File: "internal/example/run.go", SignatureSHA256: fixtureDigest, BodySHA256: fixtureDigest, Exported: true},
			{ID: callee, File: "internal/example/worker.go", SignatureSHA256: fixtureDigest, BodySHA256: fixtureDigest, Exported: true},
		},
		References:     []Reference{{From: caller, To: callee, Kind: ReferenceCall}},
		ExternalInputs: []ExternalInput{{Path: "kernels/manifest.json", ContentID: fixtureDigest, Kind: "kernel-manifest", Owner: "internal/cuda/kernel"}},
		Uncertainty:    []Uncertainty{{Kind: "reflection", Path: "internal/example/run.go", Symbol: &caller, Reason: "reflective target cannot be resolved syntactically"}},
	}
}
