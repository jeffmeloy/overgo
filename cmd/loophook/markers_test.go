package main

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/processlock"
)

// TestLoopMarkersLiveInTheRuntimeDirectory holds the hook's turn snapshot to
// the checkout's process state: a snapshot an older binary left under docs
// is carried across once, and it is then read from the state directory, so
// docs holds no process state and the tree shows none as untracked.
func TestLoopMarkersLiveInTheRuntimeDirectory(t *testing.T) {
	t.Chdir(t.TempDir())
	legacy := filepath.Join("docs", ".loop_state")
	if err := os.MkdirAll("docs", 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(legacy, []byte(`{"head":"abc123","facts":[]}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if path := turnBaseFile(); path != filepath.Join(processlock.StateDirectory, "loop_state") {
		t.Fatalf("turn snapshot at %q, want the state directory", path)
	}
	if _, err := os.Stat(legacy); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("the legacy snapshot stayed under docs: %v", err)
	}
	if base, ok := readTurnBase(); !ok || base.Head != "abc123" {
		t.Fatalf("the legacy snapshot was not carried across: %+v %v", base, ok)
	}
}
