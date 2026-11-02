package modelrecipe

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestModelStateReadsFromItsRecords: each state a model can stand in reads from what the store holds of it, a record
// that cannot be trusted being a state and not a failure.
func TestModelStateReadsFromItsRecords(t *testing.T) {
	t.Parallel()
	path := t.TempDir() + "/model.gguf"
	writeServingDefinitionGGUF(t, path, false)
	open := func(t *testing.T) artifact.Repository {
		t.Helper()
		store := openProgramStore(t)
		t.Cleanup(func() { store.Close() })
		return store
	}
	read := func(t *testing.T, store artifact.Reader, modelID artifact.ID) (Activation, ModelState) {
		t.Helper()
		activation, state, err := ReadModelState(t.Context(), store, modelID, recipe.TaskInference)
		if err != nil {
			t.Fatal(err)
		}
		return activation, state
	}
	t.Run("nothing recorded", func(t *testing.T) {
		t.Parallel()
		store := open(t)
		file, err := gguf.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		defer file.Close()
		inventory, err := modelartifact.FromGGUF(file, artifact.KindModel)
		if err != nil {
			t.Fatal(err)
		}
		if _, state := read(t, store, inventory.Manifest.ID); state != (ModelState{Verification: StateNotVerified}) {
			t.Fatalf("state = %+v", state)
		}
	})
	t.Run("a candidate nothing activated", func(t *testing.T) {
		t.Parallel()
		store := open(t)
		inventory, resolved := publishProgramFacts(t, store, path)
		if _, _, err := PublishCandidate(t.Context(), store, "fixture/state/candidate", definitionRecipe(t, inventory, resolved)); err != nil {
			t.Fatal(err)
		}
		if _, state := read(t, store, inventory.Manifest.ID); state.Verification != StateNotVerified {
			t.Fatalf("state = %+v", state)
		}
	})
	t.Run("an activation against the compiled profile", func(t *testing.T) {
		t.Parallel()
		store := open(t)
		inventory, resolved := publishProgramFacts(t, store, path)
		definition := definitionRecipe(t, inventory, resolved)
		activateProgram(t, store, definition)
		activation, state := read(t, store, inventory.Manifest.ID)
		if state != (ModelState{Verification: StateVerified}) || activation.Definition.ID != definition.ID {
			t.Fatalf("state = %+v, activation of %v", state, activation.Definition.ID)
		}
	})
	t.Run("an activation against an earlier profile", func(t *testing.T) {
		t.Parallel()
		store := open(t)
		inventory, resolved := publishProgramFacts(t, store, path)
		earlier := testutil.ArtifactID(t, artifact.KindProfile, "a profile an earlier build compiled")
		testutil.PublishArtifact(t, store, earlier)
		definition, err := InferenceWithModelDefinition(
			inventory.Manifest.ID, earlier, resolved.Document.ID, recipe.PlacementHybrid, DecodeSessionCapacity, recipe.ResidencyHybridNative,
		)
		if err != nil {
			t.Fatal(err)
		}
		activateProgram(t, store, definition)
		activation, state := read(t, store, inventory.Manifest.ID)
		if state.Verification != StateOutOfDate || state.Reason == "" || activation.Definition.ID != definition.ID {
			t.Fatalf("state = %+v, activation of %v", state, activation.Definition.ID)
		}
	})
	t.Run("an activation whose profile is no architecture's", func(t *testing.T) {
		t.Parallel()
		store := open(t)
		inventory, _ := publishProgramFacts(t, store, path)
		contract := testutil.ArtifactID(t, artifact.KindProfile, "a contract no build compiles")
		testutil.PublishArtifact(t, store, contract)
		definition, err := inference([]recipe.Dependency{
			{Role: recipe.DependencyModel, Artifact: inventory.Manifest.ID}, {Role: recipe.DependencyProfile, Artifact: contract},
		}, recipe.PlacementHybrid, DecodeSessionCapacity, recipe.ResidencyHybridNative)
		if err != nil {
			t.Fatal(err)
		}
		activateProgram(t, store, definition)
		if _, state := read(t, store, inventory.Manifest.ID); state.Verification != StateVerified {
			t.Fatalf("state = %+v", state)
		}
	})
	t.Run("an activation with no evidence behind it", func(t *testing.T) {
		t.Parallel()
		store := open(t)
		inventory, resolved := publishProgramFacts(t, store, path)
		definition := definitionRecipe(t, inventory, resolved)
		if _, _, err := PublishCandidate(t.Context(), store, "fixture/state/unactivated", definition); err != nil {
			t.Fatal(err)
		}
		batch, err := artifact.NewDocumentBatch("fixture/state/alias", nil, nil, []artifact.AliasBinding{
			{Name: activeAlias(definition.Model, definition.Task), Target: definition.ID},
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
			t.Fatal(err)
		}
		activation, state := read(t, store, inventory.Manifest.ID)
		if state.Verification != StateBroken || state.Reason == "" || activation.Definition.ID != definition.ID {
			t.Fatalf("state = %+v, activation of %v", state, activation.Definition.ID)
		}
	})
	t.Run("a canceled read", func(t *testing.T) {
		t.Parallel()
		store := open(t)
		inventory, _ := publishProgramFacts(t, store, path)
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		if _, _, err := ReadModelState(ctx, store, inventory.Manifest.ID, recipe.TaskInference); !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v", err)
		}
	})
}
