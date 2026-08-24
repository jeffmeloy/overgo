package workflowruntime

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

type modelBuildFixture struct {
	state                                        ModelBuildState
	checkpoint, model, run, evaluation, evidence artifact.ID
	decision                                     artifact.ID
	phase                                        int
}

func (fixture *modelBuildFixture) Initialize(context.Context) (ModelBuildState, error) {
	fixture.phase++
	return fixture.state, nil
}

func (fixture *modelBuildFixture) Train(_ context.Context, state ModelBuildState) (ModelBuildState, error) {
	fixture.phase++
	state.Checkpoint = fixture.checkpoint
	state.Model = fixture.model
	return state, nil
}

func (fixture *modelBuildFixture) Evaluate(_ context.Context, state ModelBuildState) (ModelBuildState, error) {
	fixture.phase++
	state.Run = fixture.run
	state.Evaluation = fixture.evaluation
	return state, nil
}

func (fixture *modelBuildFixture) Record(_ context.Context, state ModelBuildState) (ModelBuildState, error) {
	fixture.phase++
	state.Evidence = fixture.evidence
	return state, nil
}

func (fixture *modelBuildFixture) Promote(_ context.Context, state ModelBuildState) (ModelBuildState, error) {
	fixture.phase++
	state.Decision = fixture.decision
	return state, nil
}

func TestModelBuildUsesRecipeStageReceipts(t *testing.T) {
	fixture := &modelBuildFixture{state: ModelBuildState{
		Recipe:       testutil.ArtifactID(t, artifact.KindRecipe, "builder-recipe"),
		Dataset:      testutil.ArtifactID(t, artifact.KindDataset, "builder-dataset"),
		Construction: testutil.ArtifactID(t, artifact.KindRecipe, "builder-construction"),
		Model:        testutil.ArtifactID(t, artifact.KindModel, "builder-initial-model"),
	},
		checkpoint: testutil.ArtifactID(t, artifact.KindCheckpoint, "builder-checkpoint"),
		model:      testutil.ArtifactID(t, artifact.KindModel, "builder-trained-model"),
		run:        testutil.ArtifactID(t, artifact.KindRun, "builder-run"),
		evaluation: testutil.ArtifactID(t, artifact.KindEvaluation, "builder-evaluation"),
		evidence:   testutil.ArtifactID(t, artifact.KindEvidence, "builder-evidence"),
		decision:   testutil.ArtifactID(t, artifact.KindEvidence, "builder-decision"),
	}
	operation := testutil.ArtifactID(t, artifact.KindEvidence, "builder-operation")
	store := testRepository(t)
	result, err := ExecuteModelBuild(context.Background(), store, operation, fixture)
	if err != nil {
		t.Fatal(err)
	}
	if fixture.phase != 5 || result.Decision.Kind() != artifact.KindEvidence || result.Model == fixture.state.Model {
		t.Fatalf("builder result=%+v phases=%d", result, fixture.phase)
	}
	for _, stage := range ModelBuildStages() {
		receipt, found, err := runrecord.ResolveStageReceipt(context.Background(), store, operation, stage.Node.ID)
		if err != nil || !found || receipt.State != runrecord.StageCompleted {
			t.Fatalf("stage %s receipt = (%+v, %t, %v)", stage.Node.ID, receipt, found, err)
		}
	}
}

func testRepository(t testing.TB) artifact.Repository {
	t.Helper()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	return store
}
