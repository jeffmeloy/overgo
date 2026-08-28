// Package loop_test verifies loop-facing activation admission contracts.
package loop_test

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestActivationMatrixCoverage(t *testing.T) {
	profile, err := evaluation.NewActivationProfile(evaluation.ActivationProfile{
		Name: "loop-local", Kind: evaluation.ActivationProfileLocalModel,
		Capability: testutil.ArtifactID(t, artifact.KindProfile, "loop-capability"),
		Tasks:      []recipe.Task{recipe.TaskGeneration},
	})
	if err != nil || profile.ID.Kind() != artifact.KindProfile {
		t.Fatalf("activation profile = (%+v, %v)", profile, err)
	}
}
