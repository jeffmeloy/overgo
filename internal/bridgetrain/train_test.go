package bridgetrain

import (
	"context"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func TestBridgeOnlyTrainingPreservesFrozenModels(t *testing.T) {
	ctx := context.Background()
	sourceID := testutil.ArtifactID(t, artifact.KindModel, "frozen-source")
	targetID := testutil.ArtifactID(t, artifact.KindModel, "frozen-target")
	source := artifact.Descriptor{ID: sourceID, Size: uint64(len("frozen-source"))}
	target := artifact.Descriptor{ID: targetID, Size: uint64(len("frozen-target"))}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "bridge-training/frozen-models", Artifacts: []artifact.Descriptor{source, target},
	}); err != nil {
		t.Fatal(err)
	}
	weights := []float32{1, 2, 3, 4}
	before := slices.Clone(weights)
	gradients := []float32{1, -1, 2, -2}
	plan, err := optimizer.CompilePlan(len(weights), []optimizer.GroupSpec{{
		Name: "bridge.first", Start: 0, End: len(weights), Rows: 2, Cols: 2,
	}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := (Trainer{}).Step(ctx, Request{
		Reader: store, Source: sourceID, Target: targetID,
		Weights: weights, Gradients: gradients, Plan: plan,
		Config: optimizer.Config{
			BaseLearningRate: 0.1, Momentum: 0.9, Steps: 1,
			Schedule: optimizer.ScheduleConstant,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Source.Before != source || result.Source.After != source ||
		result.Target.Before != target || result.Target.After != target {
		t.Fatalf("frozen evidence changed: source=%+v target=%+v", result.Source, result.Target)
	}
	if result.BridgeBefore == result.BridgeAfter || slices.Equal(weights, before) || result.Optimizer.UpdateL2 <= 0 {
		t.Fatalf("bridge was not updated: result=%+v weights=%v", result, weights)
	}
	if len(result.Lineage) != plan.GroupCount()+len([]artifact.ID{sourceID, targetID}) {
		t.Fatalf("bridge lineage=%+v", result.Lineage)
	}
	sourceAfter, found, err := store.Artifact(ctx, sourceID)
	if err != nil || !found || sourceAfter != source {
		t.Fatalf("stored source changed: descriptor=%+v found=%t err=%v", sourceAfter, found, err)
	}
	targetAfter, found, err := store.Artifact(ctx, targetID)
	if err != nil || !found || targetAfter != target {
		t.Fatalf("stored target changed: descriptor=%+v found=%t err=%v", targetAfter, found, err)
	}
}
