package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestWorkflowEvaluationBindsTrace(t *testing.T) {
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "evaluation trace recipe")
	modelID := testutil.ArtifactID(t, artifact.KindModel, "evaluation trace model")
	request := testutil.ArtifactID(t, artifact.KindEvidence, "evaluation trace request")
	trace := func(input artifact.ID) runrecord.InteractionTrace {
		value, err := runrecord.NewInteractionTrace(runrecord.Interaction{Recipe: recipeID, Model: modelID}, input,
			[]runrecord.InteractionMessage{{Role: "assistant", Content: "answer"}}, nil, "")
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	baseline, candidate := trace(request), trace(request)
	comparison, err := CompareInteractionTraces(baseline, candidate)
	if err != nil || !comparison.Exact || comparison.Baseline != baseline.ID || comparison.Candidate != candidate.ID {
		t.Fatalf("comparison = (%+v, %v)", comparison, err)
	}
	candidate = trace(testutil.ArtifactID(t, artifact.KindEvidence, "drift"))
	comparison, err = CompareInteractionTraces(baseline, candidate)
	if err != nil || comparison.Exact || comparison.RequestExact {
		t.Fatalf("drift comparison = (%+v, %v)", comparison, err)
	}
}
