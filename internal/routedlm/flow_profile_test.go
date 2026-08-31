package routedlm

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestFlowProfileResolvesFromRecipe(t *testing.T) {
	directory := t.TempDir()
	const metadata = `{"architectures":["NEOChatModel"],"model_type":"neo_chat"}`
	if err := os.WriteFile(filepath.Join(directory, "config.json"), []byte(metadata), 0o600); err != nil {
		t.Fatal(err)
	}
	profile, err := InspectFlowProfile(directory)
	if err != nil {
		t.Fatal(err)
	}
	content, err := profile.Content()
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/flow-profile", Contents: []artifact.Content{content},
	}); err != nil {
		t.Fatal(err)
	}
	modelID := testutil.ArtifactID(t, artifact.KindModel, "flow-profile-model")
	definition, err := modelrecipe.GenerationDefinition(modelrecipe.ModuleRoutedImagePrepare, modelID, profile.ID)
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := ResolveFlowProfile(ctx, store, definition)
	if err != nil {
		t.Fatal(err)
	}
	if resolved != profile {
		t.Fatalf("resolved profile = %+v, want %+v", resolved, profile)
	}
	withoutProfile, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskImageGen,
		[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}},
		definition.Nodes, definition.Edges, definition.Inputs, definition.Outputs,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ResolveFlowProfile(ctx, store, withoutProfile); err == nil {
		t.Fatal("missing flow profile accepted")
	}
}
