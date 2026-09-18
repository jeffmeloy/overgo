//go:build windows

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/testutil"
)

func TestCompositeGenerationParity(t *testing.T) {
	cudatest.Require(t)
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	// Promotion trials are fixtures; keep their publications out of live authority.
	directory := t.TempDir()
	roots.Store = filepath.Join(directory, "store")
	data, err := json.Marshal(roots)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, dataroot.ConfigFile), data, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Chdir(root)
	t.Setenv(dataroot.Env, directory)
	if err := run(); err != nil {
		t.Fatal(err)
	}
}
