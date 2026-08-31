package bridgetrain

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

type datasetTrainingFixture struct {
	store   *overgodb.Store
	request Request
	source  *testutil.FrozenForwardStub
	target  *testutil.FrozenForwardStub
}

func TestDatasetBridgeTraining(t *testing.T) {
	fixture := newDatasetTrainingFixture(t)
	before := slices.Clone(fixture.request.Weights)
	result, err := (Trainer{}).Step(t.Context(), fixture.request)
	if err != nil {
		t.Fatal(err)
	}
	if slices.Equal(before, fixture.request.Weights) || result.BridgeBefore == result.BridgeAfter ||
		result.Checkpoint.Kind() != artifact.KindCheckpoint || len(result.Metrics) != 2 ||
		fixture.source.Calls != len(fixture.request.Examples)*fixture.request.Epochs ||
		fixture.target.Calls != fixture.source.Calls {
		t.Fatalf("dataset training result=%+v weights=%v calls=%d/%d", result, fixture.request.Weights, fixture.source.Calls, fixture.target.Calls)
	}
	if _, err := fixture.store.Commit(t.Context(), result.Batch); err != nil {
		t.Fatal(err)
	}
	if _, found, err := artifact.ReadContent(t.Context(), fixture.store, result.Checkpoint); err != nil || !found {
		t.Fatalf("checkpoint content found=%t err=%v", found, err)
	}
}

func TestBridgeCheckpointResume(t *testing.T) {
	fixture := newDatasetTrainingFixture(t)
	first, err := (Trainer{}).Step(t.Context(), fixture.request)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.store.Commit(t.Context(), first.Batch); err != nil {
		t.Fatal(err)
	}
	fixture.request.Resume = first.Checkpoint
	fixture.request.Epochs = 1
	second, err := (Trainer{}).Step(t.Context(), fixture.request)
	if err != nil {
		t.Fatal(err)
	}
	if second.BridgeBefore != first.BridgeAfter || second.BridgeAfter == first.BridgeAfter ||
		second.Optimizer.Step != first.Optimizer.Step+fixture.request.Epochs {
		t.Fatalf("resume result first=%+v second=%+v", first, second)
	}
}

func TestBridgeTrainingFreezesModels(t *testing.T) {
	fixture := newDatasetTrainingFixture(t)
	ctx := t.Context()
	sourceBefore, _, err := fixture.store.Artifact(ctx, fixture.request.Source)
	if err != nil {
		t.Fatal(err)
	}
	targetBefore, _, err := fixture.store.Artifact(ctx, fixture.request.Target)
	if err != nil {
		t.Fatal(err)
	}
	result, err := (Trainer{}).Step(ctx, fixture.request)
	if err != nil {
		t.Fatal(err)
	}
	sourceAfter, _, _ := fixture.store.Artifact(ctx, fixture.request.Source)
	targetAfter, _, _ := fixture.store.Artifact(ctx, fixture.request.Target)
	if sourceBefore != sourceAfter || targetBefore != targetAfter ||
		result.Source.Before != result.Source.After || result.Target.Before != result.Target.After {
		t.Fatalf("frozen model evidence changed: source=%+v target=%+v", result.Source, result.Target)
	}
}

func newDatasetTrainingFixture(t *testing.T) datasetTrainingFixture {
	t.Helper()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	sourceID := testutil.ArtifactID(t, artifact.KindModel, "dataset-source")
	targetID := testutil.ArtifactID(t, artifact.KindModel, "dataset-target")
	trainingPolicy := testutil.ArtifactID(t, artifact.KindProfile, "dataset-training-policy")
	examples := []Example{
		{SourceInput: []float32{1, 0}, TargetInput: []float32{1, 1}},
		{SourceInput: []float32{0, 1}, TargetInput: []float32{1, -1}},
	}
	datasetID, err := artifact.JSONID(artifact.KindDataset, examples)
	if err != nil {
		t.Fatal(err)
	}
	weights := []float32{0.5, 0, 0, 0.5}
	bridgeID, err := artifact.JSONID(artifact.KindAdapter, weights)
	if err != nil {
		t.Fatal(err)
	}
	descriptors := []artifact.Descriptor{
		{ID: sourceID}, {ID: targetID}, {ID: datasetID}, {ID: trainingPolicy}, {ID: bridgeID},
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "bridge-training/dataset-fixture", Artifacts: descriptors}); err != nil {
		t.Fatal(err)
	}
	plan, err := optimizer.CompilePlan(len(weights), []optimizer.GroupSpec{{
		Name: "bridge.linear", Start: 0, End: len(weights), Rows: 2, Cols: 2,
	}})
	if err != nil {
		t.Fatal(err)
	}
	source := &testutil.FrozenForwardStub{Model: sourceID}
	target := &testutil.FrozenForwardStub{Model: targetID}
	return datasetTrainingFixture{
		store: store, source: source, target: target,
		request: Request{
			Reader: store, Source: sourceID, Target: targetID, Weights: weights,
			Plan: plan, Config: optimizer.Config{
				BaseLearningRate: 0.05, Momentum: 0.9, Steps: 4,
				Schedule: optimizer.ScheduleConstant,
			},
			Dataset: datasetID, Examples: examples, SourceForward: source, TargetForward: target,
			TrainingPolicy: trainingPolicy, Epochs: 2,
		},
	}
}
