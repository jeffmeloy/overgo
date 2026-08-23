package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

const stageReceiptTestFailure = "execution_failed"

func TestStageReceiptIdentity(t *testing.T) {
	fixture := StageReceipt{
		Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "stage-receipt-recipe"),
		Node:   "execute", Operation: testutil.ArtifactID(t, artifact.KindEvidence, "stage-receipt-operation"),
		Attempt: 1, State: StageCompleted,
		Inputs: []StageBinding{{Port: recipe.PortName("input"), Artifacts: []artifact.ID{
			testutil.ArtifactID(t, artifact.KindFile, "stage-receipt-input"),
		}}},
	}
	first, err := NewStageReceipt(fixture)
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewStageReceipt(fixture)
	if err != nil || first.ID != second.ID {
		t.Fatalf("stage receipt identity = (%s, %s, %v)", first.ID, second.ID, err)
	}
	fixture.State, fixture.Failure = StageFailed, stageReceiptTestFailure
	if failed, err := NewStageReceipt(fixture); err != nil || failed.ID == first.ID {
		t.Fatalf("failed stage identity = (%s, %v)", failed.ID, err)
	}
}
