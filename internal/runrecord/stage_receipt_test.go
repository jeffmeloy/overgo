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

// TestStageAdmittedCrashRecovers closes the admitted-crash finding: a
// stage persisted as admitted whose runner died before persisting
// running must accept a next-attempt admission, exactly as running,
// waiting, and failed stages do.
func TestStageAdmittedCrashRecovers(t *testing.T) {
	base := StageReceipt{
		Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "stage-admitted-recipe"),
		Node:   "execute", Operation: testutil.ArtifactID(t, artifact.KindEvidence, "stage-admitted-operation"),
		Attempt: 1, State: StageAdmitted,
	}
	previous, err := NewStageReceipt(base)
	if err != nil {
		t.Fatal(err)
	}
	retry := base
	retry.Attempt, retry.State = 2, StageAdmitted
	if !stageTransition(previous, retry) {
		t.Fatal("admitted stage refused a recovery admission")
	}
	same := base
	same.State = StageCompleted
	if stageTransition(previous, same) {
		t.Fatal("admitted stage completed without running")
	}
	skipped := base
	skipped.Attempt, skipped.State = 3, StageAdmitted
	if stageTransition(previous, skipped) {
		t.Fatal("recovery skipped an attempt number")
	}
}
