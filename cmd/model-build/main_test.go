package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/scratchmodel"
)

func TestCommandRunsSharedModelBuilder(t *testing.T) {
	directory := t.TempDir()
	dataset := filepath.Join(directory, "corpus.json")
	if err := os.WriteFile(dataset, []byte(`["abcd","bcda","cdab","dabc"]`), 0o600); err != nil {
		t.Fatal(err)
	}
	repository := filepath.Join(directory, "repo")
	arguments := []string{"-repo", repository, "-dataset", dataset, "-steps", "1"}
	if err := run(arguments, &bytes.Buffer{}); err == nil || !strings.Contains(err.Error(), "active derivation profile absent") {
		t.Fatalf("unseeded model build error = %v", err)
	}
	store, err := overgodb.Open(repository)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := scratchmodel.PublishDerivationProfileCatalog(t.Context(), store); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := run(arguments, &output); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{`"model":`, `"checkpoint":`, `"run":`, `"evaluation":`, `"decision":`} {
		if !bytes.Contains(output.Bytes(), []byte(field)) {
			t.Fatalf("output lacks %s: %s", field, output.String())
		}
	}
}
