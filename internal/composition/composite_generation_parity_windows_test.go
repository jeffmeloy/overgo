//go:build windows

package composition_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestCompositeGenerationParity(t *testing.T) {
	cudatest.Require(t)
	root := testutil.RepoRoot(t)
	roots := testcheck.Must(dataroot.Resolve(root))(t)
	// Promotion trials are fixtures; keep their publications out of live authority.
	directory := t.TempDir()
	roots.Store = filepath.Join(directory, "store")
	data := testcheck.Must(json.Marshal(roots))(t)
	testcheck.NoError(t, os.WriteFile(filepath.Join(directory, dataroot.ConfigFile), data, 0o600))
	t.Setenv(dataroot.Env, directory)
	testcheck.NoError(t, runCompositeGeneration(root))
}
