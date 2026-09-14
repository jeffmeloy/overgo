package trainingprogram

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/hostoptimizer"
)

func TestProductionCheckpointResumeExact(t *testing.T) {
	id := func(kind artifact.Kind, value string) artifact.ID {
		result, err := artifact.IdentifyBytes(kind, []byte(value))
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	plan, err := hostoptimizer.CompilePlan(2, []hostoptimizer.GroupSpec{{Name: "weight", Start: 0, End: 2, Rows: 1, Cols: 2}})
	if err != nil {
		t.Fatal(err)
	}
	muon, err := hostoptimizer.New([]float32{1, 2}, []float32{3, 4}, plan, hostoptimizer.Config{BaseLearningRate: 0.01, Momentum: 0.9})
	if err != nil {
		t.Fatal(err)
	}
	muon.Step()
	model := id(artifact.KindModel, "model")
	spec := CheckpointSpec{
		RunPlan: id(artifact.KindRecipe, "run"), Program: id(artifact.KindRecipe, "program"), Model: model,
		Dataset: id(artifact.KindDataset, "dataset"), Split: id(artifact.KindDatasetShard, "split"),
		Stream:    DatasetState{Identity: id(artifact.KindDatasetShard, "stream"), Position: 7},
		Optimizer: muon.Snapshot(), ParameterCount: 2,
		RNG: []RNGState{
			{Name: "data", Algorithm: id(artifact.KindProfile, "rng"), Seed: 1, Counter: 7},
			{Name: "augmentation", Algorithm: id(artifact.KindProfile, "rng"), Seed: 2, Counter: 3},
		},
		Processors: []artifact.ID{id(artifact.KindProfile, "processor")},
		Projectors: []artifact.ID{id(artifact.KindProjector, "projector")},
		Codecs:     []artifact.ID{id(artifact.KindProfile, "codec")},
		Lineage:    []LineageParent{{Artifact: model, Relation: artifact.RelationTrainedFrom}},
	}
	target := filepath.Join(t.TempDir(), "checkpoint")
	written, err := PublishCheckpoint(target, spec, func(stage string) error {
		return os.WriteFile(filepath.Join(stage, CheckpointWeights), []byte("weights"), 0o644)
	})
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadCheckpoint(target)
	if err != nil || loaded.ID() != written.ID() || loaded.Stream != spec.Stream || loaded.Optimizer.Step != 1 || len(loaded.ArtifactLineage()) != 1 {
		t.Fatalf("round trip = (%+v, %v)", loaded, err)
	}
	if _, err := PublishCheckpoint(target, spec, func(string) error { return nil }); err == nil {
		t.Fatal("checkpoint overwrite accepted")
	}
	if err := os.WriteFile(filepath.Join(target, CheckpointWeights), []byte("tampered"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCheckpoint(target); err == nil {
		t.Fatal("modified weights accepted")
	}
	failed := filepath.Join(filepath.Dir(target), "failed")
	if _, err := PublishCheckpoint(failed, spec, func(string) error { return errors.New("stop") }); err == nil {
		t.Fatal("failed publication accepted")
	}
	if _, err := os.Stat(failed); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("failed target exists: %v", err)
	}
	spec.Accumulation = 1
	if _, err := NewCheckpoint(spec, id(artifact.KindTensorSet, "weights")); err == nil {
		t.Fatal("mid-accumulation checkpoint accepted")
	}
}
