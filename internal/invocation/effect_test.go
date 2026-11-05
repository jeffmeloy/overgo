package invocation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestMutationRequiresBoundPreflightInspection(t *testing.T) {
	manual := testutil.ArtifactID(t, artifact.KindRecipe, "inspection-manual")
	arguments := testutil.ArtifactID(t, artifact.KindEvidence, "inspection-arguments")
	inspection, err := NewEffect(Effect{
		Manual: manual, Arguments: arguments, Class: ClassInspection, Known: true,
		Targets: []Target{{Scope: ScopeWorkspace, Value: "internal/modelrecipe"}},
	})
	testcheck.NoError(t, err)
	mutation, err := NewEffect(Effect{
		Manual: manual, Arguments: arguments, Class: ClassMutation, Known: true,
		Targets: []Target{{Scope: ScopeWorkspace, Value: "internal/modelrecipe/lifecycle.go"}},
	})
	testcheck.NoError(t, err)
	if !InspectionCovers(inspection, mutation) {
		t.Fatal("relevant parent inspection did not cover mutation")
	}
	unrelated := inspection
	unrelated.ID = artifact.ID{}
	unrelated.Targets[0].Value = "internal/dataset"
	unrelated, err = NewEffect(unrelated)
	testcheck.NoError(t, err)
	if InspectionCovers(unrelated, mutation) {
		t.Fatal("unrelated inspection covered mutation")
	}
}
