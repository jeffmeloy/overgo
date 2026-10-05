package capabilityruntime

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestRefreshRefusesARemovedSelectionAgainAtAnUnmovedHead holds a director's skip of
// alias re-resolution to the proof of the selection it skips: once a selection's alias
// is removed it is refused on every attempt, including the ones at the head the first
// refusal read, and a removed selection cannot borrow the proof another selection earned
// at that head, while a selection validated at the current head is not validated again.
func TestRefreshRefusesARemovedSelectionAgainAtAnUnmovedHead(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	weights := []byte("refresh-weights")
	weightsID := testutil.ArtifactBytesID(t, artifact.KindTensorSet, weights)
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{Role: artifact.ComponentWeights, Name: "weights", Artifact: weightsID}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "refresh/model", Artifacts: []artifact.Descriptor{{ID: weightsID, Size: uint64(len(weights))}}, Manifests: []artifact.Manifest{manifest},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskGeneration, manifest.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "refresh/candidate", definition); err != nil {
		t.Fatal(err)
	}
	verification, err := modelrecipetest.PublishVerification(ctx, store, "refresh/verification", definition.ID)
	if err != nil {
		t.Fatal(err)
	}
	if err := modelrecipe.ActivateCapability(ctx, store, definition, verification, recipe.EvidenceParity, "refresh fixture"); err != nil {
		t.Fatal(err)
	}
	aliases := []string{"capability/generation/primary", "capability/generation/other", "capability/generation/third"}
	var bindings []artifact.AliasBinding
	for _, alias := range aliases {
		bindings = append(bindings, artifact.AliasBinding{Name: alias, Target: manifest.ID})
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "refresh/aliases", Aliases: bindings}); err != nil {
		t.Fatal(err)
	}
	resolve := func(alias string) modelrecipe.CapabilityEvidenceSelection {
		t.Helper()
		selection, err := modelrecipe.ResolveCapabilityEvidenceSelector(ctx, store, modelrecipe.CapabilityEvidenceSelector{
			Alias: alias, Task: recipe.TaskGeneration, Session: modelrecipe.SessionWarm,
		})
		if err != nil {
			t.Fatal(err)
		}
		return selection
	}
	director := &ModelSessionDirector[struct{}, struct{}, struct{}]{}
	primary, other, third := resolve(aliases[0]), resolve(aliases[1]), resolve(aliases[2])
	for _, selection := range []modelrecipe.CapabilityEvidenceSelection{primary, other, third} {
		if _, err := director.refreshExecution(ctx, store, selection); err != nil {
			t.Fatalf("a current selection was refused: %v", err)
		}
	}
	previous := manifest.ID
	if _, err := store.Commit(ctx, artifact.Batch{Key: "refresh/remove", Aliases: []artifact.AliasBinding{
		{Name: aliases[0], Target: manifest.ID, Previous: &previous, Remove: true},
		{Name: aliases[2], Target: manifest.ID, Previous: &previous, Remove: true},
	}}); err != nil {
		t.Fatal(err)
	}
	for attempt := range 3 {
		if _, err := director.refreshExecution(ctx, store, primary); err == nil {
			t.Fatalf("attempt %d admitted a selection whose alias was removed", attempt+1)
		}
	}
	// The independent selection earned its proof at this head; the other removed one did not and cannot borrow it.
	if _, err := director.refreshExecution(ctx, store, other); err != nil {
		t.Fatalf("the independent selection was refused: %v", err)
	}
	if _, err := director.refreshExecution(ctx, store, third); err == nil {
		t.Fatal("a removed selection borrowed the proof another selection earned at the head")
	}
	if _, err := director.refreshExecution(ctx, store, other); err != nil {
		t.Fatalf("the independent selection was refused again: %v", err)
	}
}
