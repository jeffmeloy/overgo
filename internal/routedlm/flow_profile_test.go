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
	profile := testcheck.Must(InspectFlowProfile(directory))(t)
	content := testcheck.Must(profile.Content())(t)
	store := overgodbtest.Open(t, t.TempDir())
	ctx := t.Context()
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/flow-profile", Contents: []artifact.Content{content},
	}); err != nil {
		t.Fatal(err)
	}
	modelID := testutil.ArtifactID(t, artifact.KindModel, "flow-profile-model")
	definition := testcheck.Must(modelrecipe.GenerationDefinition(modelrecipe.ModuleRoutedImagePrepare, modelID, profile.ID))(t)
	resolved := testcheck.Must(ResolveFlowProfile(ctx, store, definition))(t)
	if resolved != profile {
		t.Fatalf("resolved profile = %+v, want %+v", resolved, profile)
	}
	withoutProfile := testcheck.Must(recipe.NewDefinitionWithDependencies(
		recipe.TaskImageGen,
		[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}},
		definition.Nodes, definition.Edges, definition.Inputs, definition.Outputs,
	))(t)
	if _, err := ResolveFlowProfile(ctx, store, withoutProfile); err == nil {
		t.Fatal("missing flow profile accepted")
	}
}
