package seriesforecast

import (
	"math"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/optimizer"
	"overgo/internal/safetensors"
	"overgo/internal/trainingprogram"
)

// tinyTrainableModel builds a complete synthetic forecast model exercising
// every trainable tensor family: tokenizer block with a bias, one decoder
// layer, and a bias-free point head.
func tinyTrainableModel() *Model {
	dims := Dims{
		PatchLen: 4, Hidden: 8, Layers: 1, Heads: 2, HeadDim: 4,
		Horizon: 2, Quantiles: 3, RopeTheta: 10000, RMSEps: 1e-6,
	}
	model := &Model{
		Dims:    dims,
		Weights: map[string][]float32{},
		Shapes:  map[string][]int{},
		Levels:  []float64{0.1, 0.9},
	}
	seed := uint32(2463534242)
	fill := func(name string, shape ...int) {
		count := 1
		for _, extent := range shape {
			count *= extent
		}
		values := make([]float32, count)
		for i := range values {
			seed ^= seed << 13
			seed ^= seed >> 17
			seed ^= seed << 5
			values[i] = (float32(seed%2000)/1000 - 1) * 0.2
		}
		model.Weights[name] = values
		model.Shapes[name] = shape
	}
	fill("tokenizer.hidden_layer.weight", 8, 2*dims.PatchLen)
	fill("tokenizer.hidden_layer.bias", 8)
	fill("tokenizer.output_layer.weight", dims.Hidden, 8)
	fill("tokenizer.residual_layer.weight", dims.Hidden, 2*dims.PatchLen)
	fill("stacked_xf.0.pre_attn_ln.scale", dims.Hidden)
	fill("stacked_xf.0.post_attn_ln.scale", dims.Hidden)
	fill("stacked_xf.0.pre_ff_ln.scale", dims.Hidden)
	fill("stacked_xf.0.post_ff_ln.scale", dims.Hidden)
	fill("stacked_xf.0.attn.qkv_proj.weight", 3*dims.Hidden, dims.Hidden)
	fill("stacked_xf.0.attn.out.weight", dims.Hidden, dims.Hidden)
	fill("stacked_xf.0.attn.query_ln.scale", dims.HeadDim)
	fill("stacked_xf.0.attn.key_ln.scale", dims.HeadDim)
	fill("stacked_xf.0.attn.per_dim_scale.per_dim_scale", dims.HeadDim)
	fill("stacked_xf.0.ff0.weight", dims.Hidden, dims.Hidden)
	fill("stacked_xf.0.ff1.weight", dims.Hidden, dims.Hidden)
	fill("output_projection_point.hidden_layer.weight", 8, dims.Hidden)
	fill("output_projection_point.output_layer.weight", dims.Horizon*dims.Quantiles, 8)
	fill("output_projection_point.residual_layer.weight", dims.Horizon*dims.Quantiles, dims.Hidden)
	return model
}

func testOptimizerConfig(t *testing.T, model *Model) optimizer.Config {
	t.Helper()
	config, err := trainingprogram.BuiltinOptimizerPolicy().Config(model.TrainableParameterCount())
	if err != nil {
		t.Fatal(err)
	}
	return config
}

func TestTrainerDecreasesForecastLossTiny(t *testing.T) {
	model := tinyTrainableModel()
	trainer, err := NewTrainer(model, testOptimizerConfig(t, model))
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	input := []float32{0.4, 0.9, 1.3, 1.1, 0.8, 1.6, 2.1, 1.9, 1.4, 2.3}
	target := []float32{2.6, 2.2}
	before, err := trainer.Loss(input, target)
	if err != nil {
		t.Fatal(err)
	}
	var first, last TrainStepResult
	for step := range 6 {
		result, err := trainer.Step(input, target)
		if err != nil {
			t.Fatal(err)
		}
		if math.IsNaN(result.Total) || math.IsInf(result.Total, 0) {
			t.Fatalf("step %d loss is non-finite: %+v", step, result)
		}
		if step == 0 {
			first = result
		}
		last = result
	}
	if !(last.Total < first.Total) {
		t.Fatalf("forecast loss did not descend: first=%.6f last=%.6f", first.Total, last.Total)
	}
	if first.Step != 1 || last.Step != 6 || last.GradientL2 <= 0 {
		t.Fatalf("optimizer step facts are wrong: first=%+v last=%+v", first, last)
	}
	after, err := trainer.Loss(input, target)
	if err != nil {
		t.Fatal(err)
	}
	if first.Total != before || !(after < before) {
		t.Fatalf("loss evaluation disagrees with training: before=%.6f first=%.6f after=%.6f", before, first.Total, after)
	}
}

func TestTrainerPublishesCompiledParameterAuthority(t *testing.T) {
	model := tinyTrainableModel()
	trainer, err := NewTrainer(model, testOptimizerConfig(t, model))
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	parameters := trainer.Program().Parameters()
	if trainer.ParameterCount() == 0 || len(parameters) != len(model.trainableNames()) {
		t.Fatalf("program parameters=%d packed=%d", len(parameters), trainer.ParameterCount())
	}
	for _, parameter := range parameters {
		if values := model.Weights[parameter.Name]; len(values) != parameter.Rows*parameter.Cols || cap(values) != len(values) {
			t.Fatalf("parameter %q shape=%dx%d storage=%d/%d", parameter.Name, parameter.Rows, parameter.Cols, len(values), cap(values))
		}
	}
}

// TestTrainerCheckpointRoundTripIsBitExact trains one step, publishes the
// trainer's weights and optimizer state as a checkpoint, and reads both back
// bit for bit.
func TestTrainerCheckpointRoundTripIsBitExact(t *testing.T) {
	model := tinyTrainableModel()
	trainer, err := NewTrainer(model, testOptimizerConfig(t, model))
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	if _, err := trainer.Step([]float32{0.4, 0.9, 1.3, 1.1, 0.8, 1.6, 2.1, 1.9, 1.4, 2.3}, []float32{2.6, 2.2}); err != nil {
		t.Fatal(err)
	}
	state := trainer.Snapshot()
	loaded, catalog := checkpointRoundTrip(t, trainer.Program(), state, trainer.SaveWeights)
	if state.Step != 1 || state.PlanIdentity != trainer.Program().OptimizerIdentity() ||
		!slices.ContainsFunc(state.Momentum, func(value float64) bool { return value != 0 }) {
		t.Fatalf("snapshot after one step = step %d plan %s, want step 1 of %s with momentum", state.Step, state.PlanIdentity, trainer.Program().OptimizerIdentity())
	}
	if loaded.Step != state.Step || loaded.Config != state.Config || loaded.PlanIdentity != state.PlanIdentity ||
		!slices.EqualFunc(loaded.Momentum, state.Momentum, func(left, right float64) bool { return math.Float64bits(left) == math.Float64bits(right) }) {
		t.Fatal("the checkpoint's optimizer state differs from the trainer's")
	}
	parameters := trainer.Program().Parameters()
	if len(catalog.Values) != len(parameters) {
		t.Fatalf("checkpoint holds %d tensors, want the %d trained", len(catalog.Values), len(parameters))
	}
	for _, parameter := range parameters {
		if !slices.Equal(catalog.Shapes[parameter.Name], []int{parameter.Rows, parameter.Cols}) ||
			!slices.EqualFunc(catalog.Values[parameter.Name], model.Weights[parameter.Name], func(left, right float32) bool { return math.Float32bits(left) == math.Float32bits(right) }) {
			t.Fatalf("checkpoint tensor %q differs from the trained weights", parameter.Name)
		}
	}
}

// checkpointRoundTrip publishes program's state and the weights write
// stages as a checkpoint under stand-in authority, and reads the state and
// weights back.
func checkpointRoundTrip(t *testing.T, program trainingprogram.TrainingProgram, state optimizer.State, write func(string) error) (optimizer.State, safetensors.F32Catalog) {
	t.Helper()
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
		Optimizer: state, ParameterCount: len(state.Momentum),
		RNG:        []trainingprogram.RNGState{{Name: "augmentation", Algorithm: rng}, {Name: "data", Algorithm: rng}},
		Processors: []artifact.ID{id(artifact.KindProfile, "processor")},
		Lineage:    []trainingprogram.LineageParent{{Artifact: model, Relation: artifact.RelationTrainedFrom}},
	}, func(stage string) error { return write(filepath.Join(stage, trainingprogram.CheckpointWeights)) }); err != nil {
		t.Fatal(err)
	}
	loaded, err := trainingprogram.LoadCheckpoint(target)
	if err != nil {
		t.Fatal(err)
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
	return loaded.Optimizer, catalog
}

func TestTrainerRefusesGradientsOutsidePack(t *testing.T) {
	model := tinyTrainableModel()
	trainer, err := NewTrainer(model, testOptimizerConfig(t, model))
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	trainer.grads["stacked_xf.0.attn.stray.weight"] = make([]float32, 4)
	_, err = trainer.Step([]float32{1, 2, 3, 4, 5}, []float32{1, 2})
	if err == nil || !strings.Contains(err.Error(), "outside the trainable pack") {
		t.Fatalf("stray gradient slot was not refused: %v", err)
	}
}
