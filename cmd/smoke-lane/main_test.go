package main

import (
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
)

func TestLaneOutcomeContract(t *testing.T) {
	root := t.TempDir()
	t.Setenv(dataroot.Env, root)
	store, err := overgodb.Open(filepath.Join(root, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	err = run()
	if err == nil || !strings.Contains(err.Error(), "outcome=empty") {
		t.Fatalf("empty smoke outcome = %v", err)
	}
}
