package main

import (
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

// TestKernelManifestIsCurrent holds the committed kernel manifest to the
// sources, assets and host ABI it hashes, as `kernel-manifest` does without
// -update, so go test answers for it.
func TestKernelManifestIsCurrent(t *testing.T) {
	t.Parallel()
	testcheck.NoError(t, run(filepath.Join("..", ".."), false))
}
