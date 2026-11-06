package main

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

func TestReadInputFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "schema.json")
	testcheck.NoError(t, os.WriteFile(path, []byte(`{"type":"boolean"}`), 0o600))
	got := testcheck.Must(readInput([]string{path}))(t)
	if string(got) != `{"type":"boolean"}` {
		t.Fatalf("input = %q", got)
	}
}

func TestReadInputRejectsExtraArguments(t *testing.T) {
	if _, err := readInput([]string{"one", "two"}); err == nil {
		t.Fatal("expected usage error")
	}
}
