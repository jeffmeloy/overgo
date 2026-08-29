package runrecord

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestTrainingHealthObservationsBindPhaseStepAndAuthority(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	observation := func(ordinal uint64, phase trainingHealthPhase, name string) trainingHealthObservation {
		return trainingHealthObservation{Ordinal: ordinal, Step: min(ordinal, 1), Phase: phase, Authority: id(artifact.KindEvidence, name)}
	}
	loading := observation(0, trainingHealthLoading, "loading observation")
	loading.Loading = trainingHealthLoadingValue{Observed: true, Bytes: 1024, WallNS: 20}
	forward := observation(1, trainingHealthForward, "forward observation")
	forward.Loss = trainingHealthFloat{Observed: true, Value: 2.5}
	backward := observation(2, trainingHealthBackward, "backward observation")
	backward.GradientL2 = trainingHealthFloat{Observed: true, Value: 0.75}
	optimize := observation(3, trainingHealthOptimize, "optimize observation")
	optimize.Throughput = trainingHealthThroughput{Observed: true, Tokens: 32, WallNS: 100}
	checkpoint := observation(4, trainingHealthCheckpoint, "checkpoint observation")
	checkpoint.Checkpoint, checkpoint.CheckpointEvidence = id(artifact.KindCheckpoint, "checkpoint"), id(artifact.KindEvidence, "checkpoint evidence")
	health, err := trainingHealthCodec.New(trainingHealth{
		Version: artifact.InitialDocumentVersion, Run: id(artifact.KindRun, "run"),
		Program: id(artifact.KindRecipe, "program"), Recipe: id(artifact.KindRecipe, "recipe"), Model: id(artifact.KindModel, "model"),
		Dataset: id(artifact.KindDataset, "dataset"), Split: id(artifact.KindDatasetShard, "split"),
		Code: id(artifact.KindEvidence, "code"), Environment: id(artifact.KindEvidence, "environment"),
		Hardware: id(artifact.KindEvidence, "hardware"), Plan: id(artifact.KindProfile, "health plan"), PlannedSteps: 1,
		Observations: []trainingHealthObservation{checkpoint, backward, loading, optimize, forward},
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := trainingHealthCodec.Content(health)
	if err != nil || content.Descriptor.ID != health.ID || health.Observations[0].Phase != trainingHealthLoading || len(trainingHealthLineage(health)) != 17 {
		t.Fatalf("training health = (%+v, %v)", health, err)
	}
	overrun := health
	overrun.ID, overrun.Observations = artifact.ID{}, slices.Clone(health.Observations)
	overrun.Observations[4].Step = health.PlannedSteps + 1
	if _, err := trainingHealthCodec.New(overrun); err == nil {
		t.Fatal("progress beyond the exact plan admitted")
	}
	missingGradient := health
	missingGradient.ID, missingGradient.Observations = artifact.ID{}, slices.Clone(health.Observations)
	missingGradient.Observations[2].GradientL2 = trainingHealthFloat{}
	if _, err := trainingHealthCodec.New(missingGradient); err == nil {
		t.Fatal("missing gradient signal admitted")
	}
	badThroughput := health
	badThroughput.ID, badThroughput.Observations = artifact.ID{}, slices.Clone(health.Observations)
	badThroughput.Observations[3].Throughput.WallNS = 0
	if _, err := trainingHealthCodec.New(badThroughput); err == nil {
		t.Fatal("throughput without an exact time denominator admitted")
	}
}
