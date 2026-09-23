package modeltest

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// TestRegisteredModelResolvesThroughTheStore holds a model's directory to
// the one its registration records, including a model the configured models
// root does not hold: FireRedVAD sits beside the audio reference store while
// this checkout's models root names another home, and a name joined to that
// root found nothing. A name nothing registers resolves to nothing.
func TestRegisteredModelResolvesThroughTheStore(t *testing.T) {
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(roots.Store); errors.Is(err, fs.ErrNotExist) {
		t.Skip(testskip.Inapplicable + ": no store holds model registrations here")
	}
	directory, err := registeredDirectory(t.Context(), roots.Store, "FireRedVAD")
	if err != nil {
		t.Fatal(err)
	}
	if directory == "" {
		t.Skip(testskip.Inapplicable + ": FireRedVAD is not registered in this store")
	}
	if filepath.Dir(filepath.Join(roots.Models, "FireRedVAD")) == filepath.Dir(directory) {
		t.Logf("the models root holds FireRedVAD too; the registration still decided it")
	}
	if _, err := os.Stat(filepath.Join(directory, "VAD", "model.pth.tar")); err != nil {
		t.Fatalf("the registered directory %s does not hold the registered weights: %v", directory, err)
	}
	if missing, err := registeredDirectory(t.Context(), roots.Store, "no-model-is-registered-under-this-name"); err != nil || missing != "" {
		t.Fatalf("an unregistered name resolved to %q: %v", missing, err)
	}
}
