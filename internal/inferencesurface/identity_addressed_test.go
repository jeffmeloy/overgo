package inferencesurface

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/processcontrol"
)

// TestStoreEditLeavesTheSurfaceUnchanged holds the inference surface to the
// code a decode runs. The runner reaches the store and the run records only
// to resolve and record artifacts by content identity, so an edit there
// cannot change what a decode computes over -- yet those two packages moved
// the surface alone on 21 of its 67 moves in 400 commits, each time
// expiring every text model's evidence. The surface stops at them; the code
// the decode executes -- the CUDA executor, the model and tensor math --
// must not reach them, so the stop cannot hide compute, and an edit to the
// executor still moves the surface.
func TestStoreEditLeavesTheSurfaceUnchanged(t *testing.T) {
	ctx := t.Context()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	packages, err := Packages(ctx, root)
	if err != nil {
		t.Fatal(err)
	}
	for _, pkg := range packages {
		relative, err := filepath.Rel(root, pkg.Dir)
		if err != nil {
			t.Fatal(err)
		}
		if slices.Contains(identityAddressed, filepath.ToSlash(relative)) {
			t.Errorf("the surface descends into identity-addressed %s", relative)
		}
	}
	moves, err := Moves(ctx, root, []string{"internal/overgodb/store.go", "internal/runrecord/run.go"})
	if err != nil || len(moves) != 0 {
		t.Fatalf("a store or record edit moved the surface: %v %v", moves, err)
	}
	if moves, err := Moves(ctx, root, []string{"internal/cuda/executor/executor.go"}); err != nil || len(moves) != 1 {
		t.Fatalf("an executor edit did not move the surface: %v %v", moves, err)
	}
	var stdout, stderr bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "go", Args: []string{"list", "-deps", "./internal/cuda/executor", "./internal/model", "./internal/tensor"},
		Dir: root, Env: os.Environ(), Stdout: &stdout, Stderr: &stderr,
	})
	if err != nil || receipt.ExitCode != 0 {
		t.Fatalf("go list: %v %s", err, stderr.String())
	}
	for line := range strings.SplitSeq(stdout.String(), "\n") {
		for _, name := range identityAddressed {
			if strings.TrimSpace(line) == "overgo/"+name {
				t.Errorf("decode compute reaches identity-addressed %s, so the surface would hide it", name)
			}
		}
	}
}
