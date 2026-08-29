package routedlm

import (
	"math"
	"testing"

	"overgo/internal/trainingprogram"

	"overgo/internal/tensor/dtype"
)

// tinyFlowHead builds a deterministic small head in BF16 serving form.
func tinyFlowHead(hidden, flowDim int) (FlowPlan, FlowMLPWeights) {
	plan := FlowPlan{Hidden: hidden, FlowDim: flowDim, TEps: 1e-3}
	seed := uint32(362436069)
	next := func() float32 {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		return (float32(seed%2000)/1000 - 1) * 0.3
	}
	bf16 := func(n int) []uint16 {
		values := make([]uint16, n)
		for i := range values {
			values[i] = dtype.Float32ToBF16(next())
		}
		return values
	}
	f32 := func(n int) []float32 {
		values := make([]float32, n)
		for i := range values {
			values[i] = next() * 0.1
		}
		return values
	}
	head := FlowMLPWeights{
		W0: BF16Matrix{Data: bf16(hidden * hidden), In: hidden, Out: hidden},
		B0: f32(hidden),
		W2: BF16Matrix{Data: bf16(flowDim * hidden), In: hidden, Out: flowDim},
		B2: f32(flowDim),
	}
	return plan, head
}

func flowStimulus(rows, hidden, flowDim int) (x, z, target []float32) {
	seed := uint32(521288629)
	next := func() float32 {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		return float32(seed%2000)/1000 - 1
	}
	x = make([]float32, rows*hidden)
	for i := range x {
		x[i] = next() * 0.5
	}
	z = make([]float32, rows*flowDim)
	target = make([]float32, rows*flowDim)
	for i := range z {
		z[i] = next() * 0.2
		target[i] = next()
	}
	return x, z, target
}

func TestFlowHeadTrainerDescendsAndDerivesHyperparameters(t *testing.T) {
	plan, head := tinyFlowHead(6, 4)
	trainer, err := NewFlowHeadTrainer(plan, head, trainingprogram.BuiltinOptimizerPolicy())
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	if got, want := trainer.Config().Momentum, 29.0/31.0; math.Abs(got-want) > 1e-12 {
		t.Fatalf("momentum %g, want derived %g", got, want)
	}
	if trainer.Config().BaseLearningRate <= 0 {
		t.Fatal("base learning rate was not derived")
	}
	x, z, target := flowStimulus(3, plan.Hidden, plan.FlowDim)
	var first, last FlowHeadStepResult
	for step := range 12 {
		result, err := trainer.Step(x, z, target, 0.25)
		if err != nil {
			t.Fatal(err)
		}
		if math.IsNaN(result.Loss) || math.IsInf(result.Loss, 0) {
			t.Fatalf("step %d loss non-finite: %+v", step, result)
		}
		if step == 0 {
			first = result
			if result.GradientL2 <= 0 {
				t.Fatal("first step carried no gradient")
			}
		}
		last = result
	}
	if !(last.Loss < first.Loss) {
		t.Fatalf("flow head loss did not descend: %.6f -> %.6f", first.Loss, last.Loss)
	}
}

func TestFlowHeadStepGradientMatchesFiniteDifference(t *testing.T) {
	const epsilon, limit = 1e-3, 2e-2
	plan, head := tinyFlowHead(5, 3)
	x, z, target := flowStimulus(2, plan.Hidden, plan.FlowDim)
	const timestep = 0.4

	loss := func(tr *FlowHeadTrainer) float64 {
		v, err := tr.Velocity(x, z, timestep)
		if err != nil {
			t.Fatal(err)
		}
		var total float64
		invN := 1 / float64(len(target))
		for i := range v {
			d := float64(v[i]) - float64(target[i])
			total += d * d * invN
		}
		return total
	}

	// Analytic gradients captured before any optimizer step consumes them.
	probe, err := NewFlowHeadTrainer(plan, head, trainingprogram.BuiltinOptimizerPolicy())
	if err != nil {
		t.Fatal(err)
	}
	defer probe.Close()
	baseline := loss(probe)
	stepLoss, gradientL2, err := probe.lossAndGradients(x, z, target, timestep)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(stepLoss-baseline) > 1e-9 || gradientL2 <= 0 {
		t.Fatalf("loss %.9g disagrees with evaluation %.9g (grad_l2=%g)", stepLoss, baseline, gradientL2)
	}
	analytic := append([]float32(nil), probe.gradients...)

	for _, index := range []int{0, 7, len(analytic)/2 + 1, len(analytic) - 2, len(analytic) - 1} {
		fresh, err := NewFlowHeadTrainer(plan, head, trainingprogram.BuiltinOptimizerPolicy())
		if err != nil {
			t.Fatal(err)
		}
		original := fresh.weights[index]
		fresh.weights[index] = original + epsilon
		plus := loss(fresh)
		fresh.weights[index] = original - epsilon
		minus := loss(fresh)
		fresh.weights[index] = original
		numeric := (plus - minus) / (2 * epsilon)
		_ = fresh.Close()
		got := float64(analytic[index])
		scale := max(1, max(math.Abs(numeric), math.Abs(got)))
		if math.Abs(numeric-got)/scale > limit {
			t.Fatalf("gradient[%d]: analytic=%.6g numeric=%.6g", index, got, numeric)
		}
	}
}
