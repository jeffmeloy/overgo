package main

import (
	"path/filepath"
	"testing"
)

// TestKernelManifestIsCurrent holds the committed kernel manifest to the
// sources, assets and host ABI it hashes, as `kernel-manifest` does without
// -update, so go test answers for it.
func TestKernelManifestIsCurrent(t *testing.T) {
	t.Parallel()
	if err := run(filepath.Join("..", ".."), false); err != nil {
		t.Fatal(err)
	}
}
