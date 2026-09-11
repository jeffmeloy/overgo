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
	err = run(nil)
	if err == nil || !strings.Contains(err.Error(), "outcome=empty") {
		t.Fatalf("empty smoke outcome = %v", err)
	}
}

func TestSelectedModelPublicationAdmission(t *testing.T) {
	root := t.TempDir()
	t.Setenv(dataroot.Env, root)
	store, err := overgodb.Open(filepath.Join(root, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	// A selected publication reaches the same catalog admission as a full
	// pass. An empty catalog still fails; selecting a model grants no credit.
	model := "model:sha256:3007c05b8ece556726a37980069cf6c0f1f966a48572b1c130c5643810c23a32"
	if err := run([]string{"-model", model, "-budget", "1m"}); err == nil || !strings.Contains(err.Error(), "outcome=empty") {
		t.Fatalf("selected publication admission: %v", err)
	}
	for _, value := range []string{"MiniCPM", strings.Replace(model, "model:", "recipe:", 1)} {
		if err := run([]string{"-model", value}); err == nil || !strings.Contains(err.Error(), "exact model identity") {
			t.Fatalf("invalid selected model %q: %v", value, err)
		}
	}
}
