package seriesforecast

import (
	"math"
	"strings"
	"testing"

	"overgo/internal/optimizer"
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
	for step := 0; step < 6; step++ {
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
