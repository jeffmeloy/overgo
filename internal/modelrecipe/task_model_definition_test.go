package modelrecipe

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// publishedGenerationRecipe publishes a model manifest and the generation
// recipe over it, a recipe that depends on its model alone.
func publishedGenerationRecipe(t *testing.T, store *overgodb.Store, name string) recipe.Definition {
	t.Helper()
	weights := []byte(name + "-weights")
	weightsID := testutil.ArtifactBytesID(t, artifact.KindTensorSet, weights)
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: weightsID,
	}})
	testcheck.NoError(t, err)
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key:       "test/task-definition-model/" + manifest.ID.String(),
		Artifacts: []artifact.Descriptor{{ID: weightsID, Size: uint64(len(weights))}}, Manifests: []artifact.Manifest{manifest},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := CapabilityDefinition(recipe.TaskGeneration, manifest.ID)
	testcheck.NoError(t, err)
	content, err := Content(definition)
	testcheck.NoError(t, err)
	batch, err := artifact.NewDocumentBatch("test/task-definition-recipe/"+definition.ID.String(), []artifact.Content{content}, nil, nil)
	testcheck.NoError(t, err)
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	return definition
}

// TestTaskModelDefinitionOfAModelOnlyRecipeBindsItsModelAndRecipe holds the
// definition of a recipe that names no tensor inventory to the model and the
// recipe it was published for: it binds them, and binds no other.
func TestTaskModelDefinitionOfAModelOnlyRecipeBindsItsModelAndRecipe(t *testing.T) {
	t.Parallel()
	store := overgodbtest.Open(t, t.TempDir())
	definition := publishedGenerationRecipe(t, store, "bound")
	id, err := PublishTaskModelDefinition(t.Context(), store, definition.ID)
	testcheck.NoError(t, err)
	if err := RequireModelDefinitionBinding(t.Context(), store, id, definition.Model, definition.ID); err != nil {
		t.Fatalf("the definition does not bind the model and recipe it was published for: %v", err)
	}
	foreignModel := testutil.ArtifactID(t, artifact.KindModel, "foreign-model")
	if err := RequireModelDefinitionBinding(t.Context(), store, id, foreignModel, definition.ID); err == nil {
		t.Fatal("the definition binds a model it was not published for")
	}
	foreignRecipe := testutil.ArtifactID(t, artifact.KindRecipe, "foreign-recipe")
	if err := RequireModelDefinitionBinding(t.Context(), store, id, definition.Model, foreignRecipe); err == nil {
		t.Fatal("the definition binds a recipe it was not published for")
	}
}

// TestTaskModelDefinitionRefusesARecipeOverAModelNoManifestHolds holds the
// definition to the model manifest it binds: a recipe over an unpublished
// model has no definition.
func TestTaskModelDefinitionRefusesARecipeOverAModelNoManifestHolds(t *testing.T) {
	t.Parallel()
	store := overgodbtest.Open(t, t.TempDir())
	definition, err := CapabilityDefinition(recipe.TaskGeneration, testutil.ArtifactID(t, artifact.KindModel, "unpublished-model"))
	testcheck.NoError(t, err)
	content, err := Content(definition)
	testcheck.NoError(t, err)
	batch, err := artifact.NewDocumentBatch("test/task-definition-recipe/"+definition.ID.String(), []artifact.Content{content}, nil, nil)
	testcheck.NoError(t, err)
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	if _, err := PublishTaskModelDefinition(t.Context(), store, definition.ID); err == nil {
		t.Fatal("a definition was published for a model no manifest holds")
	}
}
