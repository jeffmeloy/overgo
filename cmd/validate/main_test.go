package main

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
)

// TestCompatibilityCommandIsGone holds the fold of cmd/compatibility into
// validate: the model and media reports, the media samples, the capability
// census and the model-verification records it served are validate's flags,
// and its acceptance suite runs as validate's, so no second command remains.
func TestCompatibilityCommandIsGone(t *testing.T) {
	t.Parallel()
	if _, err := os.Stat(filepath.Join("..", "compatibility")); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("cmd/compatibility is present (%v); its reports, samples, census and verification records are validate's", err)
	}
}
