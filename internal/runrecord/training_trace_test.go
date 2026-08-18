package runrecord

import (
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

func TestDPOTraceBindsMeasuredObservations(t *testing.T) {
	id := func(kind artifact.Kind, value string) artifact.ID { return testutil.ArtifactID(t, kind, value) }
	observation := trainingprogram.DPOObservation{
		Step: 1, Loss: 0.6,
		PolicyChosen: -1, PolicyRejected: -2, ReferenceChosen: -1.5, ReferenceRejected: -2,
		PolicyMargin: 1, ReferenceMargin: 0.5, RelativeMargin: 0.5,
		ChosenTokens: 2, RejectedTokens: 3, LearningRate: 0.01, GradientL2: 2, UpdateL2: 0.1,
	}
	trace, err := NewDPOTrace(
		id(artifact.KindRun, "run"), id(artifact.KindRecipe, "recipe"), id(artifact.KindDataset, "dataset"),
		id(artifact.KindModel, "policy"), id(artifact.KindModel, "reference"), []trainingprogram.DPOObservation{observation},
	)
	if err != nil {
		t.Fatal(err)
	}
	content, err := trace.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := trainingTraceCodec.Parse(content.Data)
	if err != nil || parsed.ID != trace.ID || len(parsed.Observations) != 1 || parsed.Observations[0] != observation || len(trace.Lineage()) != 5 {
		t.Fatalf("parsed=%+v lineage=%+v err=%v", parsed, trace.Lineage(), err)
	}
}

func TestDPOTraceRejectsInventedReward(t *testing.T) {
	raw := map[string]any{
		"version":      trainingTraceVersion,
		"run":          testutil.ArtifactID(t, artifact.KindRun, "run"),
		"recipe":       testutil.ArtifactID(t, artifact.KindRecipe, "recipe"),
		"dataset":      testutil.ArtifactID(t, artifact.KindDataset, "dataset"),
		"policy":       testutil.ArtifactID(t, artifact.KindModel, "policy"),
		"reference":    testutil.ArtifactID(t, artifact.KindModel, "reference"),
		"observations": []map[string]any{{"step": 1, "reward": 1}},
	}
	data, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := trainingTraceCodec.Parse(data); err == nil {
		t.Fatal("invented reward field accepted")
	}
}
