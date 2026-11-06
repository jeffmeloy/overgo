package modelcli

import (
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/discovery"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/pathidentity"
	"overgo/internal/recipe"
	"overgo/internal/servingattempt"
	"overgo/internal/testcheck"
)

// TestAServedOpenRegistersWhatItsRecordsNeed: once a file the store has never seen has opened, an attempt on it records,
// the store holding it as a candidate and no more, and the next open claims its identity instead of hashing it.
func TestAServedOpenRegistersWhatItsRecordsNeed(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	storeRoot, path := filepath.Join(root, "store"), filepath.Join(root, "model.gguf")
	writeServingFixture(t, path, 300)
	created := testcheck.Must(overgodb.Create(storeRoot))(t)
	testcheck.NoError(t, created.Close())
	reader := testcheck.Must(overgodb.OpenReadOnly(storeRoot))(t)
	loaded := testcheck.Must(modelrecipe.ResolveClaimedGGUF(t.Context(), reader, path, nil))(t)
	derivation, derived := loaded.Derived()
	testcheck.NoError(t, loaded.Close())
	testcheck.NoError(t, reader.Close())
	if !derived {
		t.Fatal("a file the store has never seen was not derived")
	}
	testcheck.NoError(t, registerDerived(t.Context(), storeRoot, derivation))

	store := overgodbtest.Open(t, storeRoot)
	if _, err := servingattempt.RecordWholeOutput(t.Context(), store, servingattempt.WholeOutput{
		Model: derivation.Inventory.Manifest.ID, Recipe: derivation.Definition.ID, Task: recipe.TaskInference,
		Started: time.Now(), Wall: time.Millisecond,
	}); err != nil {
		t.Fatalf("an attempt on a derived model did not record: %v", err)
	}
	_, state, err := modelrecipe.ReadModelState(t.Context(), store, derivation.Inventory.Manifest.ID, recipe.TaskInference)
	if err != nil || state.Verification != modelrecipe.StateNotVerified {
		t.Fatalf("registering a run verified it: %+v, %v", state, err)
	}
	canonical := testcheck.Must(pathidentity.Canonical(path))(t)
	memo := discovery.LoadMemo(t.Context(), store)
	claimed, size, remembered := memo.Claim(canonical)
	if !remembered || claimed != derivation.Inventory.Components[0].ID || size != derivation.Inventory.Components[0].Size {
		t.Fatalf("claim = %v, %d, %t, want the open's identity", claimed, size, remembered)
	}
	again := testcheck.Must(modelrecipe.ResolveClaimedGGUF(t.Context(), store, path, memo.Claim))(t)
	defer again.Close()
	if identity, err := again.Identity(); err != nil || identity.Model != derivation.Inventory.Manifest.ID {
		t.Fatalf("the next open resolved %+v, %v", identity, err)
	}
}
