package capabilityruntime

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/testcheck"
)

// TestAnAbsentFileARequestReadsIsRefused holds served-answers/absent (SQA-406): a request whose answer reads a file
// beside its model (a preset's adapter, the depth estimator) that is absent is refused, naming the file, before its
// executor runs, so no stored answer is sought for it and none is kept under it; with the file in place the request is
// executed under the file's identity. Contents had written "absent" into the naming and gone on, so a draw that found
// the file installed by the time it opened it was kept under that naming, and once the file was removed again the
// request was answered from that run.
func TestAnAbsentFileARequestReadsIsRefused(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	held, adapter := filepath.Join(directory, "held.safetensors"), filepath.Join(directory, "adapter.safetensors")
	testcheck.NoError(t, os.WriteFile(held, []byte("held"), 0o600))
	var executed []string
	serve := Reading(func(ctx context.Context, _ artifact.Repository, _ string, _ modelrecipe.CapabilityEvidenceSelection, _ string) (any, error) {
		named, _ := ctx.Value(besideContextKey{}).(string)
		executed = append(executed, named)
		return nil, nil
	}, Contents(nil, func(context.Context, artifact.Repository, string, modelrecipe.CapabilityEvidenceSelection, string) ([]string, error) {
		return []string{held, adapter}, nil
	}))
	_, err := serve(t.Context(), nil, directory, modelrecipe.CapabilityEvidenceSelection{}, `{}`)
	if !errors.Is(err, fs.ErrNotExist) || !strings.Contains(err.Error(), adapter) {
		t.Fatalf("a request reading the absent %s gave %v; want it refused naming the file", adapter, err)
	}
	if len(executed) != 0 {
		t.Fatalf("a request reading an absent file was executed %d times", len(executed))
	}
	testcheck.NoError(t, os.WriteFile(adapter, []byte("adapter"), 0o600))
	testcheck.Must(serve(t.Context(), nil, directory, modelrecipe.CapabilityEvidenceSelection{}, `{}`))(t)
	if len(executed) != 1 || executed[0] == "" {
		t.Fatalf("with the file in place the request was executed under %q; want once, under the files' identities", executed)
	}
}
