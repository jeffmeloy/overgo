package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadInputFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "schema.json")
	if err := os.WriteFile(path, []byte(`{"type":"boolean"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := readInput([]string{path})
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != `{"type":"boolean"}` {
		t.Fatalf("input = %q", got)
	}
}

func TestReadInputRejectsExtraArguments(t *testing.T) {
	if _, err := readInput([]string{"one", "two"}); err == nil {
		t.Fatal("expected usage error")
	}
}
