package server

import (
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/scratchmodel"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestModelBuilderWorkspaceUsesSharedCampaign(t *testing.T) {
	t.Parallel()
	store := overgodbtest.Open(t, t.TempDir())
	ctx := t.Context()
	if _, err := scratchmodel.PublishDerivationProfileCatalog(ctx, store); err != nil {
		t.Fatal(err)
	}
	workspace, err := NewModelBuilderWorkspace(ctx, store)
	testcheck.NoError(t, err)
	capabilities, err := workspace.WorkflowCapabilities(ctx, workflowModelBuild)
	if err != nil || len(capabilities) != 1 {
		t.Fatalf("capabilities=%+v err=%v", capabilities, err)
	}
	input, err := json.Marshal(modelBuildInput{Corpus: "abcd\nbcda\ncdab\ndabc", Seed: 1, Steps: 1})
	testcheck.NoError(t, err)
	completion, err := workspace.ExecuteWorkflow(ctx, workflowModelBuild,
		capabilities[0], input, testReporter{id: testutil.ArtifactID(t, artifact.KindEvidence, "model-builder-operation")})
	testcheck.NoError(t, err)
	if completion.Run.Kind() != artifact.KindRun || len(completion.Outputs) != 5 {
		t.Fatalf("completion=%+v", completion)
	}
}
