package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/scratchmodel"
	"overgo/internal/testcheck"
)

func TestCommandRunsSharedModelBuilder(t *testing.T) {
	directory := t.TempDir()
	dataset := filepath.Join(directory, "corpus.json")
	testcheck.NoError(t, os.WriteFile(dataset, []byte(`["abcd","bcda","cdab","dabc"]`), 0o600))
	repository := t.TempDir()
	arguments := []string{"-repo", repository, "-dataset", dataset, "-steps", "1"}
	if err := run(arguments, &bytes.Buffer{}); err == nil || !strings.Contains(err.Error(), "active derivation profile absent") {
		t.Fatalf("unseeded model build error = %v", err)
	}
	store := testcheck.Must(overgodb.Create(repository))(t)
	if _, err := scratchmodel.PublishDerivationProfileCatalog(t.Context(), store); err != nil {
		t.Fatal(err)
	}
	testcheck.NoError(t, store.Close())
	var output bytes.Buffer
	testcheck.NoError(t, run(arguments, &output))
	for _, field := range []string{`"model":`, `"checkpoint":`, `"run":`, `"evaluation":`, `"decision":`} {
		if !bytes.Contains(output.Bytes(), []byte(field)) {
			t.Fatalf("output lacks %s: %s", field, output.String())
		}
	}
}
