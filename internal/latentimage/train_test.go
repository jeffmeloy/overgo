package latentimage

import (
	"math"
	"testing"

	"overgo/internal/checkpointtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

func tinyFinalLayer(hidden, out int) FinalLayerWeights {
	seed := uint32(2463534242)
	next := testutil.XorShift(seed)
	fill := func(n int) []float32 {
		values := make([]float32, n)
		for i := range values {
			values[i] = next()
		}
		return values
	}
	return FinalLayerWeights{
		Hidden: hidden, Out: out,
		Norm: fill(hidden), Table: fill(2 * hidden), Linear: fill(out * hidden), Bias: fill(out),
	}
}

func finalStimulus(rows, hidden, out int) (x, temb, target []float64) {
	seed := uint32(88172645)
	next := func() float64 {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		return float64(seed%2000)/1000 - 1
	}
	x = make([]float64, rows*hidden)
	for i := range x {
		x[i] = next() * 0.5
	}
	temb = make([]float64, hidden)
	for i := range temb {
		temb[i] = next() * 0.2
	}
	target = make([]float64, rows*out)
	for i := range target {
		target[i] = next()
	}
	return x, temb, target
}

func TestFinalLayerTrainerDescends(t *testing.T) {
	trainer := testcheck.Must(NewFinalLayerTrainer(tinyFinalLayer(8, 4), 1e-6, trainingprogram.BuiltinOptimizerPolicy()))(t)
	defer trainer.Close()
	if got, want := trainer.Config().Momentum, 29.0/31.0; math.Abs(got-want) > 1e-12 {
		t.Fatalf("momentum %g, want derived %g", got, want)
	}
	x, temb, target := finalStimulus(5, 8, 4)
	before := testcheck.Must(trainer.Loss(x, temb, target))(t)
	var first FinalLayerStepResult
	for step := range 12 {
		result := testcheck.Must(trainer.Step(x, temb, target))(t)
		if math.IsNaN(result.Loss) || math.IsInf(result.Loss, 0) {
			t.Fatalf("step %d non-finite loss", step)
		}
		if step == 0 {
			first = result
			if math.Abs(result.Loss-before) > 1e-9 || result.GradientL2 <= 0 {
				t.Fatalf("first step loss %.9g vs evaluation %.9g grad=%g", result.Loss, before, result.GradientL2)
			}
		}
	}
	after := testcheck.Must(trainer.Loss(x, temb, target))(t)
	if !(after < first.Loss) {
		t.Fatalf("final layer loss did not descend: %.6f -> %.6f", first.Loss, after)
	}
}

// TestFinalLayerTrainerCheckpointRoundTripIsBitExact trains one step,
// publishes the trainer's organ and optimizer state as a checkpoint, and
// reads both back bit for bit under every head tensor name.
func TestFinalLayerTrainerCheckpointRoundTripIsBitExact(t *testing.T) {
	trainer := testcheck.Must(NewFinalLayerTrainer(tinyFinalLayer(8, 4), 1e-6, trainingprogram.BuiltinOptimizerPolicy()))(t)
	defer trainer.Close()
	x, temb, target := finalStimulus(5, 8, 4)
	if _, err := trainer.Step(x, temb, target); err != nil {
		t.Fatal(err)
	}
	program := testcheck.Must(trainingprogram.CompileObjectiveProgram(trainingprogram.ObjectiveFlowMatching, nil, trainer.OptimizerPlan()))(t)
	view := trainer.pack.Weights
	checkpointtest.RequireRoundTrip(t, program, trainer, map[string][]float32{
		"final.norm": view(trainer.norm), "final.table": view(trainer.table),
		"final.linear": view(trainer.linear), "final.bias": view(trainer.bias),
	})
}

func TestFinalLayerLayerNormVariantDescendsAndMatchesFiniteDifference(t *testing.T) {
	weights := tinyFinalLayer(7, 3)
	weights.Norm = nil // the Wan2.1 head: non-parametric LayerNorm
	trainer := testcheck.Must(NewFinalLayerTrainer(weights, 1e-6, trainingprogram.BuiltinOptimizerPolicy()))(t)
	defer trainer.Close()
	x, temb, target := finalStimulus(3, 7, 3)
	baseline := testcheck.Must(trainer.Loss(x, temb, target))(t)
	stepLoss, gradientL2, err := trainer.lossAndGradients(x, temb, target)
	testcheck.NoError(t, err)
	if math.Abs(stepLoss-baseline) > 1e-9 || gradientL2 <= 0 {
		t.Fatalf("layernorm gradient pass loss %.9g vs %.9g", stepLoss, baseline)
	}
	const epsilonFD, limitFD = 1e-4, 2e-2
	groups := trainer.OptimizerPlan().Groups()
	_, gradients := trainer.pack.Slabs()
	for _, index := range []int{groups[trainer.table].Start + 2, groups[trainer.table].Start + 7 + 2, groups[trainer.linear].Start + 4, groups[trainer.bias].Start} {
		fresh := testcheck.Must(NewFinalLayerTrainer(weights, 1e-6, trainingprogram.BuiltinOptimizerPolicy()))(t)
		freshWeights, _ := fresh.pack.Slabs()
		original := freshWeights[index]
		freshWeights[index] = original + epsilonFD
		plus := testcheck.Must(fresh.Loss(x, temb, target))(t)
		freshWeights[index] = original - epsilonFD
		minus := testcheck.Must(fresh.Loss(x, temb, target))(t)
		_ = fresh.Close()
		numeric := (plus - minus) / (2 * epsilonFD)
		got := float64(gradients[index])
		scale := max(1, max(math.Abs(numeric), math.Abs(got)))
		if math.Abs(numeric-got)/scale > limitFD {
			t.Fatalf("layernorm gradient[%d]: analytic=%.6g numeric=%.6g", index, got, numeric)
		}
	}
	for range 10 {
		if _, err := trainer.Step(x, temb, target); err != nil {
			t.Fatal(err)
		}
	}
	after := testcheck.Must(trainer.Loss(x, temb, target))(t)
	if !(after < baseline) {
		t.Fatalf("layernorm head did not descend: %.6f -> %.6f", baseline, after)
	}
}

func TestFinalLayerGradientMatchesFiniteDifference(t *testing.T) {
	const epsilon, limit = 1e-4, 2e-2
	weights := tinyFinalLayer(7, 3)
	x, temb, target := finalStimulus(3, 7, 3)

	grader := testcheck.Must(NewFinalLayerTrainer(weights, 1e-6, trainingprogram.BuiltinOptimizerPolicy()))(t)
	defer grader.Close()
	baseline := testcheck.Must(grader.Loss(x, temb, target))(t)
	stepLoss, gradientL2, err := grader.lossAndGradients(x, temb, target)
	testcheck.NoError(t, err)
	if math.Abs(stepLoss-baseline) > 1e-9 || gradientL2 <= 0 {
		t.Fatalf("gradient pass loss %.9g vs %.9g", stepLoss, baseline)
	}

	// One index from every group: norm, table scale row, table shift row,
	// linear, bias.
	groups := grader.OptimizerPlan().Groups()
	indices := []int{
		groups[grader.norm].Start + 2,
		groups[grader.table].Start + 3,
		groups[grader.table].Start + 7 + 3,
		groups[grader.linear].Start + 5,
		groups[grader.bias].Start + 1,
	}
	_, gradients := grader.pack.Slabs()
	for _, index := range indices {
		fresh := testcheck.Must(NewFinalLayerTrainer(weights, 1e-6, trainingprogram.BuiltinOptimizerPolicy()))(t)
		freshWeights, _ := fresh.pack.Slabs()
		original := freshWeights[index]
		freshWeights[index] = original + epsilon
		plus := testcheck.Must(fresh.Loss(x, temb, target))(t)
		freshWeights[index] = original - epsilon
		minus := testcheck.Must(fresh.Loss(x, temb, target))(t)
		_ = fresh.Close()
		numeric := (plus - minus) / (2 * epsilon)
		got := float64(gradients[index])
		scale := max(1, max(math.Abs(numeric), math.Abs(got)))
		if math.Abs(numeric-got)/scale > limit {
			t.Fatalf("gradient[%d]: analytic=%.6g numeric=%.6g", index, got, numeric)
		}
	}
}
