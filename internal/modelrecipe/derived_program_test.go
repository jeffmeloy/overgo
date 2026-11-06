package modelrecipe

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAGGUFNothingRecordsResolvesFromItsFile: a model no trusted recipe names resolves from its own file and the
// prototype compiled for its architecture, its state saying what the store holds of it; a recipe that stands still wins.
func TestAGGUFNothingRecordsResolvesFromItsFile(t *testing.T) {
	t.Parallel()
	path := t.TempDir() + "/model.gguf"
	writeServingDefinitionGGUF(t, path, false)
	file := testcheck.Must(gguf.Open(path))(t)
	defer file.Close()
	inventory := testcheck.Must(modelartifact.FromGGUF(file, artifact.KindModel))(t)
	catalog, found, err := CatalogRuntimePolicy(recipe.TaskInference)
	if err != nil || !found {
		t.Fatalf("catalog policy: %v, %v", found, err)
	}
	resolve := func(t *testing.T, store artifact.Reader) (ProgramIdentity, ModelState, RuntimePolicy) {
		t.Helper()
		loaded := testcheck.Must(ResolveActiveGGUF(t.Context(), store, path))(t)
		t.Cleanup(func() { loaded.Close() })
		identity := testcheck.Must(loaded.Identity())(t)
		state := testcheck.Must(loaded.State())(t)
		policy := testcheck.Must(loaded.RuntimePolicy())(t)
		return identity, state, policy
	}
	open := openedProgramStore
	t.Run("nothing recorded", func(t *testing.T) {
		t.Parallel()
		identity, state, policy := resolve(t, open(t))
		if state.Verification != StateNotVerified || identity.Model != inventory.Manifest.ID {
			t.Fatalf("state %+v, model %v, want not verified %v", state, identity.Model, inventory.Manifest.ID)
		}
		if policy.Serving.OutputTokens != catalog.Serving.OutputTokens {
			t.Fatalf("policy %+v, want the catalog's %+v", policy.Serving, catalog.Serving)
		}
	})
	t.Run("a candidate nothing activated", func(t *testing.T) {
		t.Parallel()
		store := open(t)
		facts, resolved := publishProgramFacts(t, store, path)
		if _, _, err := PublishCandidate(t.Context(), store, "fixture/derived/candidate", definitionRecipe(t, facts, resolved)); err != nil {
			t.Fatal(err)
		}
		if _, state, _ := resolve(t, store); state.Verification != StateNotVerified {
			t.Fatalf("state %+v", state)
		}
	})
	t.Run("an activation with no evidence behind it", func(t *testing.T) {
		t.Parallel()
		store := open(t)
		facts, resolved := publishProgramFacts(t, store, path)
		definition := definitionRecipe(t, facts, resolved)
		if _, _, err := PublishCandidate(t.Context(), store, "fixture/derived/unactivated", definition); err != nil {
			t.Fatal(err)
		}
		batch := testcheck.Must(artifact.NewDocumentBatch("fixture/derived/alias", nil, nil, []artifact.AliasBinding{
			{Name: activeAlias(definition.Model, definition.Task), Target: definition.ID},
		}))(t)
		if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
			t.Fatal(err)
		}
		if _, state, _ := resolve(t, store); state.Verification != StateBroken || state.Reason == "" {
			t.Fatalf("state %+v", state)
		}
	})
	t.Run("a recipe whose policy cannot be read", func(t *testing.T) {
		t.Parallel()
		store := open(t)
		facts, resolved := publishProgramFacts(t, store, path)
		definition := definitionRecipe(t, facts, resolved)
		activateProgram(t, store, definition)
		bogus := testutil.ArtifactID(t, artifact.KindProfile, "a policy the store does not hold")
		if _, err := store.Commit(t.Context(), artifact.Batch{
			Key: "fixture/derived/policy", Artifacts: []artifact.Descriptor{{ID: bogus}},
			Aliases: []artifact.AliasBinding{{Name: runtimePolicyAlias(definition.ID), Target: bogus, Previous: &catalog.ID}},
		}); err != nil {
			t.Fatal(err)
		}
		identity, state, policy := resolve(t, store)
		if state.Verification != StateVerified || identity.Recipe != definition.ID || policy.Serving.OutputTokens != catalog.Serving.OutputTokens {
			t.Fatalf("state %+v, recipe %v, policy %+v", state, identity.Recipe, policy.Serving)
		}
	})
	t.Run("a recipe that stands", func(t *testing.T) {
		t.Parallel()
		store := open(t)
		facts, resolved := publishProgramFacts(t, store, path)
		definition := definitionRecipe(t, facts, resolved)
		activateProgram(t, store, definition)
		identity, state, _ := resolve(t, store)
		if state.Verification != StateVerified || identity.Recipe != definition.ID {
			t.Fatalf("state %+v, recipe %v, want %v", state, identity.Recipe, definition.ID)
		}
	})
}
