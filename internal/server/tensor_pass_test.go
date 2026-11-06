package server

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
)

func tensorFixtureInventory(t *testing.T, path string) modelartifact.TensorInventoryDocument {
	t.Helper()
	file := testcheck.Must(gguf.Open(path))(t)
	defer file.Close()
	inventory := testcheck.Must(modelartifact.FromGGUF(file, artifact.KindModel))(t)
	return inventory.TensorInventory
}

// TestTensorPassSurvivesPublishFailure keeps a finished full measurement
// answering when its commit fails (the pass once reported the ranks failed
// until the server restarted), inventories an unchanged model file once
// however often the tab asks, names each profile's model the way the store's
// pool does so the served model can be found in it, and links the published
// inventory to a model the store knows.
func TestTensorPassSurvivesPublishFailure(t *testing.T) {
	t.Parallel()
	path := writeTensorFixture(t)
	inventory := tensorFixtureInventory(t, path)

	store := overgodbtest.Open(t, t.TempDir())
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key: "server/tensor-pass-model", Artifacts: []artifact.Descriptor{{ID: inventory.Owner, Size: 1}},
	}); err != nil {
		t.Fatal(err)
	}
	serving := newTestHandlerForRepository(t, store, tensorPathGenerator{fakeGenerator: &fakeGenerator{}, path: path})
	defer serving.Close()
	first := requestTensors(t, serving, "?wait=spectra")
	again := requestTensors(t, serving, "")
	if first.Spectra != spectraComplete || len(first.Tensors) == 0 || again.Spectra != spectraComplete {
		t.Fatalf("answers = %s then %s with %d profiles", first.Spectra, again.Spectra, len(first.Tensors))
	}
	if served := serving.servedTensorModel(); served == "" || first.Tensors[0].Model != served {
		t.Fatalf("profile model %q, served model %q: the pool and the served model name models alike", first.Tensors[0].Model, served)
	}
	inventories := 0
	serving.tensorInventories.Range(func(any, any) bool { inventories++; return true })
	if inventories != 1 {
		t.Fatalf("two requests for one model file built %d inventories", inventories)
	}
	parents := testcheck.Must(store.Parents(t.Context(), inventory.ID))(t)
	linked := false
	for _, parent := range parents {
		linked = linked || parent.Parent == inventory.Owner
	}
	if !linked {
		t.Fatalf("the published inventory names no link to its model: %+v", parents)
	}

	closed := testcheck.Must(overgodb.Open(t.TempDir()))(t)
	failing := newTestHandlerForRepository(t, closed, tensorPathGenerator{fakeGenerator: &fakeGenerator{}, path: path})
	defer failing.Close()
	testcheck.NoError(t, closed.Close())
	unpublished := requestTensors(t, failing, "?wait=spectra")
	if unpublished.Spectra != spectraComplete || unpublished.SpectraFailure != "" || len(unpublished.Tensors) != len(first.Tensors) {
		t.Fatalf("a failed commit answered %s (%q) with %d profiles; the measurement it finished must still answer",
			unpublished.Spectra, unpublished.SpectraFailure, len(unpublished.Tensors))
	}
	held, ok := failing.tensorPasses.Load(inventory.ID)
	if !ok || held.(*tensorPass).published {
		t.Fatal("a commit into a closed store was recorded as published")
	}
}
