// Package capabilityruntime_test verifies runtime-facing evaluation contracts
// without adding a production dependency cycle.
package capabilityruntime_test

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestActivationCaseRegistry(t *testing.T) {
	value, err := evaluation.NewActivationCase(evaluation.ActivationCase{
		Name: "generation.contract", Task: recipe.TaskGeneration,
		Contract:      testutil.ArtifactID(t, artifact.KindRecipe, "runtime-task-contract"),
		Input:         testutil.ArtifactID(t, artifact.KindDataset, "runtime-case-input"),
		EvidenceCheck: testutil.ArtifactID(t, artifact.KindProfile, "runtime-evidence-check"),
	})
	if err != nil || value.ID.Kind() != artifact.KindRecipe {
		t.Fatalf("activation case = (%+v, %v)", value, err)
	}
}
