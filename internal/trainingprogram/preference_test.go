package trainingprogram

import (
	"math"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/hostoptimizer"
	"overgo/internal/testutil"
)

func TestDPOProgramRequiresReferenceAndScale(t *testing.T) {
	plan, err := hostoptimizer.CompilePlan(1, []hostoptimizer.GroupSpec{{Name: "weight", End: 1, Rows: 1, Cols: 1}})
	if err != nil {
		t.Fatal(err)
	}
	reference := testutil.ArtifactID(t, artifact.KindModel, "reference")
	spec := ProgramSpec{
		Objective: ObjectiveDPO,
		Operators: []OperatorSpec{
			{ID: ObjectiveOperatorForward, Phase: PhaseForward},
			{ID: ObjectiveOperatorBackward, Phase: PhaseBackward},
			{ID: ObjectiveOperatorMuon, Phase: PhaseOptimize},
		},
		Parameters: []ParameterSpec{{Name: "weight", Rows: 1, Cols: 1, Trainable: true}},
		Optimizer:  plan,
		Preference: &PreferencePolicy{Reference: reference, Scale: 0.1},
	}
	program, err := CompileTrainingProgram(spec)
	if err != nil {
		t.Fatal(err)
	}
	policy, ok := program.Preference()
	if !ok || policy.Reference != reference || policy.Scale != spec.Preference.Scale {
		t.Fatalf("preference policy = (%+v, %v)", policy, ok)
	}
	for _, invalid := range []*PreferencePolicy{
		nil,
		{Reference: reference},
		{Reference: reference, Scale: math.Inf(1)},
		{Reference: testutil.ArtifactID(t, artifact.KindProfile, "profile"), Scale: 1},
	} {
		spec.Preference = invalid
		if _, err := CompileTrainingProgram(spec); err == nil {
			t.Fatalf("accepted preference policy %+v", invalid)
		}
	}
}
