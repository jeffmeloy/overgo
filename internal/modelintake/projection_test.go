//go:build modeltest

package modelintake

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// storedProjectionPair finds a model whose store activation binds a
// projector with bytes on disk: the pair the intake must reproduce. The
// store is the authority for which pair that is; no name is typed here.
func storedProjectionPair(t *testing.T) (modelPath, projectorPath string, projection recipe.Definition) {
	t.Helper()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatalf("projection pair unavailable: %v", err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatalf("projection pair unavailable: %v", err)
	}
	defer store.Close()
	ctx := t.Context()
	memo := discovery.LoadMemo(ctx, store)
	entries, _, err := discovery.CapabilityCatalog(ctx, store, 256, memo)
	testcheck.NoError(t, err)
	for _, entry := range entries {
		if !entry.Present || entry.Location == "" {
			continue
		}
		declared, ok, err := discovery.ActiveProjector(ctx, store, entry.Model, memo)
		if err != nil || !ok {
			continue
		}
		if _, err := os.Stat(entry.Location); err != nil {
			continue
		}
		_, program, err := modelrecipe.ResolveActiveCapability(ctx, store, entry.Model, recipe.TaskProjection)
		testcheck.NoError(t, err)
		return entry.Location, declared, program.Definition()
	}
	t.Fatal("projection pair unavailable: the store activates no projector with bytes on disk")
	return "", "", recipe.Definition{}
}

// TestPrepareProjectionCandidateReproducesStoredBinding proves the intake
// defines, from the two files alone, the same projection recipe the
// store activated for that pair: the same model, the same projector
// artifact, the same media.
func TestPrepareProjectionCandidateReproducesStoredBinding(t *testing.T) {
	testskip.Short(t, "reading model files is integration")
	modelPath, projectorPath, stored := storedProjectionPair(t)
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	testcheck.NoError(t, err)
	store := overgodbtest.OpenReadOnly(t, roots.Store)
	candidate, err := PrepareProjectionCandidate(t.Context(), store, modelPath, projectorPath)
	testcheck.NoError(t, err)
	if candidate.Definition.ID != stored.ID {
		t.Fatalf("intake defined %s for %s + %s; the store activated %s", candidate.Definition.ID, filepath.Base(modelPath), filepath.Base(projectorPath), stored.ID)
	}
	if len(candidate.Media) == 0 || candidate.ProjectorPath != projectorPath {
		t.Fatalf("candidate = %+v", candidate)
	}
	t.Logf("projection intake: %s + %s -> %s media=%v", filepath.Base(modelPath), filepath.Base(projectorPath), candidate.Definition.ID, candidate.Media)
}
