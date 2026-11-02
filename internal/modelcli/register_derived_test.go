package modelcli

import (
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/discovery"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/pathidentity"
	"overgo/internal/recipe"
	"overgo/internal/servingattempt"
)

// TestAServedOpenRegistersWhatItsRecordsNeed: once a file the store has never seen has opened, an attempt on it records,
// the store holding it as a candidate and no more, and the next open claims its identity instead of hashing it.
func TestAServedOpenRegistersWhatItsRecordsNeed(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	storeRoot, path := filepath.Join(root, "store"), filepath.Join(root, "model.gguf")
	writeServingFixture(t, path, 300)
	created, err := overgodb.Create(storeRoot)
	if err != nil {
		t.Fatal(err)
	}
	if err := created.Close(); err != nil {
		t.Fatal(err)
	}
	reader, err := overgodb.OpenReadOnly(storeRoot)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := modelrecipe.ResolveClaimedGGUF(t.Context(), reader, path, nil)
	if err != nil {
		t.Fatal(err)
	}
	derivation, derived := loaded.Derived()
	if err := loaded.Close(); err != nil {
		t.Fatal(err)
	}
	if err := reader.Close(); err != nil {
		t.Fatal(err)
	}
	if !derived {
		t.Fatal("a file the store has never seen was not derived")
	}
	if err := registerDerived(t.Context(), storeRoot, derivation); err != nil {
		t.Fatal(err)
	}

	store, err := overgodb.Open(storeRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
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
	canonical, err := pathidentity.Canonical(path)
	if err != nil {
		t.Fatal(err)
	}
	memo := discovery.LoadMemo(t.Context(), store)
	claimed, size, remembered := memo.Claim(canonical)
	if !remembered || claimed != derivation.Inventory.Components[0].ID || size != derivation.Inventory.Components[0].Size {
		t.Fatalf("claim = %v, %d, %t, want the open's identity", claimed, size, remembered)
	}
	again, err := modelrecipe.ResolveClaimedGGUF(t.Context(), store, path, memo.Claim)
	if err != nil {
		t.Fatal(err)
	}
	defer again.Close()
	if identity, err := again.Identity(); err != nil || identity.Model != derivation.Inventory.Manifest.ID {
		t.Fatalf("the next open resolved %+v, %v", identity, err)
	}
}
