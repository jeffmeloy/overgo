package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestCommandRunsSharedModelBuilder(t *testing.T) {
	directory := t.TempDir()
	dataset := filepath.Join(directory, "corpus.json")
	if err := os.WriteFile(dataset, []byte(`["abcd","bcda","cdab","dabc"]`), 0o600); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := run([]string{"-repo", filepath.Join(directory, "repo"), "-dataset", dataset, "-steps", "1"}, &output); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{`"model":`, `"checkpoint":`, `"run":`, `"evaluation":`, `"decision":`} {
		if !bytes.Contains(output.Bytes(), []byte(field)) {
			t.Fatalf("output lacks %s: %s", field, output.String())
		}
	}
}
