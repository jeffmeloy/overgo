package latentimage

import (
	"math"
	"testing"
)

func tinyFinalLayer(hidden, out int) FinalLayerWeights {
	seed := uint32(2463534242)
	next := func() float32 {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		return (float32(seed%2000)/1000 - 1) * 0.3
	}
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
	trainer, err := NewFinalLayerTrainer(tinyFinalLayer(8, 4), 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	if got, want := trainer.Config().Momentum, 29.0/31.0; math.Abs(got-want) > 1e-12 {
		t.Fatalf("momentum %g, want derived %g", got, want)
	}
	x, temb, target := finalStimulus(5, 8, 4)
	before, err := trainer.Loss(x, temb, target)
	if err != nil {
		t.Fatal(err)
	}
	var first FinalLayerStepResult
	for step := 0; step < 12; step++ {
		result, err := trainer.Step(x, temb, target)
		if err != nil {
			t.Fatal(err)
		}
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
	after, err := trainer.Loss(x, temb, target)
	if err != nil {
		t.Fatal(err)
	}
	if !(after < first.Loss) {
		t.Fatalf("final layer loss did not descend: %.6f -> %.6f", first.Loss, after)
	}
}

func TestFinalLayerLayerNormVariantDescendsAndMatchesFiniteDifference(t *testing.T) {
	weights := tinyFinalLayer(7, 3)
	weights.Norm = nil // the Wan2.1 head: non-parametric LayerNorm
	trainer, err := NewFinalLayerTrainer(weights, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	x, temb, target := finalStimulus(3, 7, 3)
	baseline, err := trainer.Loss(x, temb, target)
	if err != nil {
		t.Fatal(err)
	}
	stepLoss, gradientL2, err := trainer.lossAndGradients(x, temb, target)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(stepLoss-baseline) > 1e-9 || gradientL2 <= 0 {
		t.Fatalf("layernorm gradient pass loss %.9g vs %.9g", stepLoss, baseline)
	}
	const epsilonFD, limitFD = 1e-4, 2e-2
	for _, index := range []int{trainer.table.start + 2, trainer.table.start + 7 + 2, trainer.linear.start + 4, trainer.bias.start} {
		fresh, err := NewFinalLayerTrainer(weights, 1e-6)
		if err != nil {
			t.Fatal(err)
		}
		original := fresh.weights[index]
		fresh.weights[index] = original + epsilonFD
		plus, err := fresh.Loss(x, temb, target)
		if err != nil {
			t.Fatal(err)
		}
		fresh.weights[index] = original - epsilonFD
		minus, err := fresh.Loss(x, temb, target)
		if err != nil {
			t.Fatal(err)
		}
		_ = fresh.Close()
		numeric := (plus - minus) / (2 * epsilonFD)
		got := float64(trainer.gradients[index])
		scale := math.Max(1, math.Max(math.Abs(numeric), math.Abs(got)))
		if math.Abs(numeric-got)/scale > limitFD {
			t.Fatalf("layernorm gradient[%d]: analytic=%.6g numeric=%.6g", index, got, numeric)
		}
	}
	for step := 0; step < 10; step++ {
		if _, err := trainer.Step(x, temb, target); err != nil {
			t.Fatal(err)
		}
	}
	after, err := trainer.Loss(x, temb, target)
	if err != nil {
		t.Fatal(err)
	}
	if !(after < baseline) {
		t.Fatalf("layernorm head did not descend: %.6f -> %.6f", baseline, after)
	}
}

func TestFinalLayerGradientMatchesFiniteDifference(t *testing.T) {
	const epsilon, limit = 1e-4, 2e-2
	weights := tinyFinalLayer(7, 3)
	x, temb, target := finalStimulus(3, 7, 3)

	grader, err := NewFinalLayerTrainer(weights, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	defer grader.Close()
	baseline, err := grader.Loss(x, temb, target)
	if err != nil {
		t.Fatal(err)
	}
	stepLoss, gradientL2, err := grader.lossAndGradients(x, temb, target)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(stepLoss-baseline) > 1e-9 || gradientL2 <= 0 {
		t.Fatalf("gradient pass loss %.9g vs %.9g", stepLoss, baseline)
	}

	// One index from every group: norm, table scale row, table shift row,
	// linear, bias.
	indices := []int{
		grader.norm.start + 2,
		grader.table.start + 3,
		grader.table.start + 7 + 3,
		grader.linear.start + 5,
		grader.bias.start + 1,
	}
	for _, index := range indices {
		fresh, err := NewFinalLayerTrainer(weights, 1e-6)
		if err != nil {
			t.Fatal(err)
		}
		original := fresh.weights[index]
		fresh.weights[index] = original + epsilon
		plus, err := fresh.Loss(x, temb, target)
		if err != nil {
			t.Fatal(err)
		}
		fresh.weights[index] = original - epsilon
		minus, err := fresh.Loss(x, temb, target)
		if err != nil {
			t.Fatal(err)
		}
		_ = fresh.Close()
		numeric := (plus - minus) / (2 * epsilon)
		got := float64(grader.gradients[index])
		scale := math.Max(1, math.Max(math.Abs(numeric), math.Abs(got)))
		if math.Abs(numeric-got)/scale > limit {
			t.Fatalf("gradient[%d]: analytic=%.6g numeric=%.6g", index, got, numeric)
		}
	}
}
