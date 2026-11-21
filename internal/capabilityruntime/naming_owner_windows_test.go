//go:build windows

package capabilityruntime

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/heldfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestANamingThatCannotListItsFilesOwnsNothing holds a request's naming (Contents) to owning nothing of the set it names
// through until its files are listed (held-release, SQA-454): a request whose listing fails, as one naming a preset the
// model has none of does, is answered by a fresh run, to which Reading gives no release, so an ownership taken before
// the listing was never given back, and the process's set then held every file named after it, a writer refused for the
// process's life. A file named with no owner after such a request is the writer's.
func TestANamingThatCannotListItsFilesOwnsNothing(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	testutil.WriteTextFile(t, root, "adapters/a.safetensors", "the adapter's weights", 0o600)
	adapter := filepath.Join(root, "adapters", "a.safetensors")
	set := &heldfile.Set{}
	unlisted := Contents(set, func(string, string) ([]string, error) {
		return nil, errors.New(`latent image: the model has no preset "unknown"`)
	})
	fresh := func(context.Context, artifact.Repository, string, modelrecipe.CapabilityEvidenceSelection, string) (any, error) {
		return "a fresh run", nil
	}
	testcheck.Must(Reading(fresh, unlisted)(t.Context(), nil, root, modelrecipe.CapabilityEvidenceSelection{}, `{"preset":"unknown"}`))(t)
	testcheck.Must(set.Identity(adapter))(t)
	testutil.WriteTextFile(t, root, "adapters/written", "weights written after the request", 0o600)
	if err := os.Rename(filepath.Join(root, "adapters", "written"), adapter); err != nil {
		t.Errorf("a file named after a request whose files could not be listed is held: %v; want it the writer's, no owner left behind", err)
	}
}
