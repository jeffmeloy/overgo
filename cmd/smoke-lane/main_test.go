package main

import (
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/repodb"
)

func TestLaneOutcomeContract(t *testing.T) {
	root := t.TempDir()
	t.Setenv(dataroot.Env, root)
	store, err := repodb.Open(filepath.Join(root, "repodb-store"))
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
