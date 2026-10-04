// Package checkpointtest holds a trainer to the checkpoint it publishes.
package checkpointtest

import (
	"math"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/hostoptimizer"
	"overgo/internal/safetensors"
	"overgo/internal/trainingprogram"
)

// Trainer is what a trainer hands its checkpoint: the optimizer state its
// next step continues from and the weights that state pairs with.
type Trainer interface {
	Snapshot() hostoptimizer.State
	SaveWeights(path string) error
}

// RequireRoundTrip publishes what trainer trained under program as a
// checkpoint under stand-in authority and reads it back: the optimizer state
// must be the trainer's snapshot bit for bit, in the dtype the trainer
// steps, stepped and carrying momentum under the program's plan, and each
// program parameter's tensor must hold trained's values bit for bit at the
// parameter's [rows, cols].
func RequireRoundTrip(t testing.TB, program trainingprogram.TrainingProgram, trainer Trainer, trained map[string][]float32) {
	t.Helper()
	state := trainer.Snapshot()
	if state.Step == 0 || state.PlanIdentity != program.OptimizerIdentity() ||
		!slices.ContainsFunc(state.Momentum, func(value float64) bool { return value != 0 }) &&
			!slices.ContainsFunc(state.MomentumF32, func(value float32) bool { return value != 0 }) {
		t.Fatalf("snapshot = step %d of plan %s, want a step of %s with momentum", state.Step, state.PlanIdentity, program.OptimizerIdentity())
	}
	id := func(kind artifact.Kind, name string) artifact.ID {
		value, err := artifact.IdentifyBytes(kind, []byte(name))
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	model, rng := id(artifact.KindModel, "model"), id(artifact.KindProfile, "rng")
	target := filepath.Join(t.TempDir(), "checkpoint")
	if _, err := trainingprogram.PublishCheckpoint(target, trainingprogram.CheckpointSpec{
		RunPlan: id(artifact.KindRecipe, "run plan"), Program: program.ID(), Model: model,
		Dataset: id(artifact.KindDataset, "dataset"), Split: id(artifact.KindDatasetShard, "split"),
		Stream:    trainingprogram.DatasetState{Identity: id(artifact.KindDataset, "stream"), Position: uint64(state.Step)},
		Optimizer: state, ParameterCount: state.Parameters(),
		RNG:        []trainingprogram.RNGState{{Name: "augmentation", Algorithm: rng}, {Name: "data", Algorithm: rng}},
		Processors: []artifact.ID{id(artifact.KindProfile, "processor")},
		Lineage:    []trainingprogram.LineageParent{{Artifact: model, Relation: artifact.RelationTrainedFrom}},
	}, func(stage string) error {
		return trainer.SaveWeights(filepath.Join(stage, trainingprogram.CheckpointWeights))
	}); err != nil {
		t.Fatal(err)
	}
	loaded, err := trainingprogram.LoadCheckpoint(target)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Optimizer.Step != state.Step || loaded.Optimizer.Config != state.Config || loaded.Optimizer.PlanIdentity != state.PlanIdentity ||
		!slices.EqualFunc(loaded.Optimizer.Momentum, state.Momentum, func(left, right float64) bool { return math.Float64bits(left) == math.Float64bits(right) }) ||
		!slices.EqualFunc(loaded.Optimizer.MomentumF32, state.MomentumF32, func(left, right float32) bool { return math.Float32bits(left) == math.Float32bits(right) }) {
		t.Fatal("the checkpoint's optimizer state differs from the trainer's")
	}
	source, err := safetensors.OpenFile(filepath.Join(target, trainingprogram.CheckpointWeights))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	catalog, err := source.MaterializeF32(safetensors.F32Selection{RetainShapes: true})
	if err != nil {
		t.Fatal(err)
	}
	parameters := program.Parameters()
	if len(catalog.Values) != len(parameters) || len(trained) != len(parameters) {
		t.Fatalf("checkpoint holds %d tensors of %d trained, want the program's %d", len(catalog.Values), len(trained), len(parameters))
	}
	for _, parameter := range parameters {
		if !slices.Equal(catalog.Shapes[parameter.Name], []int{parameter.Rows, parameter.Cols}) ||
			!slices.EqualFunc(catalog.Values[parameter.Name], trained[parameter.Name], func(left, right float32) bool { return math.Float32bits(left) == math.Float32bits(right) }) {
			t.Fatalf("checkpoint tensor %q differs from the trained weights", parameter.Name)
		}
	}
}
