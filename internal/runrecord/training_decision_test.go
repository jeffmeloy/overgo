package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestTrainingDecisionWithholdsPromotionUntilEvaluation(t *testing.T) {
	id := func(kind artifact.Kind, label string) artifact.ID { return testutil.ArtifactID(t, kind, label) }
	decision := testcheck.Must(NewTrainingDecision(
		id(artifact.KindRun, "run"), id(artifact.KindRecipe, "recipe"), id(artifact.KindModel, "parent"),
		id(artifact.KindCheckpoint, "checkpoint"), id(artifact.KindEvidence, "trace"),
	))(t)
	if decision.State != TrainingEvaluationRequired || decision.Rollback != decision.Parent || len(decision.Lineage()) != 5 {
		t.Fatalf("decision=%+v lineage=%+v", decision, decision.Lineage())
	}
}
