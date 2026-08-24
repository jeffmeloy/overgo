package server

import (
	"context"
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestModelBuilderWorkspaceUsesSharedCampaign(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	workspace, err := NewModelBuilderWorkspace(store)
	if err != nil {
		t.Fatal(err)
	}
	capabilities, err := workspace.WorkflowCapabilities(context.Background(), WorkflowModelBuild)
	if err != nil || len(capabilities) != 1 {
		t.Fatalf("capabilities=%+v err=%v", capabilities, err)
	}
	input, err := json.Marshal(modelBuildInput{Corpus: "abcd\nbcda\ncdab\ndabc", Seed: 1, Steps: 1})
	if err != nil {
		t.Fatal(err)
	}
	completion, err := workspace.ExecuteWorkflow(context.Background(), WorkflowModelBuild, recipe.TaskTraining,
		capabilities[0].Recipe, input, testReporter{id: testutil.ArtifactID(t, artifact.KindEvidence, "model-builder-operation")})
	if err != nil {
		t.Fatal(err)
	}
	if completion.Run.Kind() != artifact.KindRun || len(completion.Outputs) != 5 {
		t.Fatalf("completion=%+v", completion)
	}
}
