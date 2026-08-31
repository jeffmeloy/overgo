package invocation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestMutationRequiresBoundPreflightInspection(t *testing.T) {
	manual := testutil.ArtifactID(t, artifact.KindRecipe, "inspection-manual")
	arguments := testutil.ArtifactID(t, artifact.KindEvidence, "inspection-arguments")
	inspection, err := NewEffect(Effect{
		Manual: manual, Arguments: arguments, Class: ClassInspection, Known: true,
		Targets: []Target{{Scope: ScopeWorkspace, Value: "internal/modelrecipe"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	mutation, err := NewEffect(Effect{
		Manual: manual, Arguments: arguments, Class: ClassMutation, Known: true,
		Targets: []Target{{Scope: ScopeWorkspace, Value: "internal/modelrecipe/lifecycle.go"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !InspectionCovers(inspection, mutation) {
		t.Fatal("relevant parent inspection did not cover mutation")
	}
	unrelated := inspection
	unrelated.ID = artifact.ID{}
	unrelated.Targets[0].Value = "internal/dataset"
	unrelated, err = NewEffect(unrelated)
	if err != nil {
		t.Fatal(err)
	}
	if InspectionCovers(unrelated, mutation) {
		t.Fatal("unrelated inspection covered mutation")
	}
}
