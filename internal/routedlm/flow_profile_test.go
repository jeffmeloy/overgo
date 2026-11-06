package routedlm

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestFlowProfileResolvesFromRecipe(t *testing.T) {
	directory := t.TempDir()
	const metadata = `{"architectures":["NEOChatModel"],"model_type":"neo_chat"}`
	testcheck.NoError(t, os.WriteFile(filepath.Join(directory, "config.json"), []byte(metadata), 0o600))
	profile, err := InspectFlowProfile(directory)
	testcheck.NoError(t, err)
	content, err := profile.Content()
	testcheck.NoError(t, err)
	store := overgodbtest.Open(t, t.TempDir())
	ctx := t.Context()
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/flow-profile", Contents: []artifact.Content{content},
	}); err != nil {
		t.Fatal(err)
	}
	modelID := testutil.ArtifactID(t, artifact.KindModel, "flow-profile-model")
	definition, err := modelrecipe.GenerationDefinition(modelrecipe.ModuleRoutedImagePrepare, modelID, profile.ID)
	testcheck.NoError(t, err)
	resolved, err := ResolveFlowProfile(ctx, store, definition)
	testcheck.NoError(t, err)
	if resolved != profile {
		t.Fatalf("resolved profile = %+v, want %+v", resolved, profile)
	}
	withoutProfile, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskImageGen,
		[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}},
		definition.Nodes, definition.Edges, definition.Inputs, definition.Outputs,
	)
	testcheck.NoError(t, err)
	if _, err := ResolveFlowProfile(ctx, store, withoutProfile); err == nil {
		t.Fatal("missing flow profile accepted")
	}
}
