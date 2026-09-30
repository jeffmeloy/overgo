package e4btrain

import (
	"math"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/optimizer"
	"overgo/internal/trainingprogram"
)

// Parity bounds, each relative to the largest magnitude in its tensor: the
// device attention VJP accumulates in f32 where the host kit accumulates in
// f64, and the device Newton-Schulz iterates in f32.
const (
	gradientTolerance = 1e-5
	updateTolerance   = 1e-2
)

// TestE4BFullParameterStepMatchesHost takes one full-parameter step of the
// tiny E4B-shaped decoder on the device and holds it to the host kit: every
// trained tensor's device gradient matches the host kit's within
// gradientTolerance of its largest magnitude, the step moves every tensor
// by the host kit's Muon update within updateTolerance, and the window's
// loss descends.
func TestE4BFullParameterStepMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(device.DefaultOrdinal())
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	host := tinyModel(t, 1)
	hostLoss, err := host.differentiate(hostAttention)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := trainingprogram.CompileProbeSessionPlan(trainingprogram.ProbeSpec{
		Objective: trainingprogram.ObjectiveTokenPrediction, Updates: 1, MaximumSequence: tinyTokens,
		Parameters: host.ParameterCount(), Optimizer: trainingprogram.BuiltinOptimizerPolicy(),
	})
	if err != nil {
		t.Fatal(err)
	}
	config := plan.Optimizer()
	model := tinyModel(t, 1)
	trainer, err := NewTrainer(worker, model, config)
	if err != nil {
		t.Fatal(err)
	}
	deviceLoss, err := model.differentiate(trainer.attention)
	if err != nil {
		t.Fatal(err)
	}
	if deviceLoss != hostLoss {
		t.Fatalf("device pass loss %v, host kit %v: the forward is shared", deviceLoss, hostLoss)
	}
	worst := 0.0
	for _, group := range model.plan.Groups() {
		ratio := relativeDifference(model.gradients[group.Start:group.End], host.gradients[group.Start:group.End])
		worst = max(worst, ratio)
		if ratio > gradientTolerance {
			t.Errorf("%s: device gradient differs from the host kit by %.3g of its largest magnitude", group.Name, ratio)
		}
	}
	t.Logf("gradients: %d tensors, %d parameters, worst relative difference %.3g", model.plan.GroupCount(), model.ParameterCount(), worst)

	before := slices.Clone(model.weights)
	result, err := trainer.Step()
	if err != nil {
		t.Fatal(err)
	}
	if result.Loss != hostLoss || result.Step != 1 || !(result.GradientL2 > 0) {
		t.Fatalf("step result %+v, want the host kit's loss %v", result, hostLoss)
	}
	stepper, err := optimizer.New(host.weights, host.gradients, host.plan, config)
	if err != nil {
		t.Fatal(err)
	}
	stepper.Step()
	worst = 0
	for _, group := range model.plan.Groups() {
		span := func(values []float32) []float32 { return values[group.Start:group.End] }
		deviceUpdate, hostUpdate := difference(span(model.weights), span(before)), difference(span(host.weights), span(before))
		if slices.Max(absolute(deviceUpdate)) == 0 {
			t.Errorf("%s: the step left the tensor unchanged", group.Name)
		}
		ratio := relativeDifference(deviceUpdate, hostUpdate)
		worst = max(worst, ratio)
		if ratio > updateTolerance {
			t.Errorf("%s: device update differs from the host kit's Muon update by %.3g of its largest magnitude", group.Name, ratio)
		}
	}
	after, err := trainer.Loss()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("step: loss %.6f -> %.6f, gradient L2 %.4g, learning rate %.4g, worst update difference %.3g", result.Loss, after, result.GradientL2, result.LearningRate, worst)
	if !(after < result.Loss) {
		t.Errorf("the step did not descend: %.6f -> %.6f", result.Loss, after)
	}
}

func difference(left, right []float32) []float32 {
	result := make([]float32, len(left))
	for index := range left {
		result[index] = left[index] - right[index]
	}
	return result
}

func absolute(values []float32) []float32 {
	result := make([]float32, len(values))
	for index, value := range values {
		result[index] = float32(math.Abs(float64(value)))
	}
	return result
}

// relativeDifference is the largest element difference of got from want
// over want's largest magnitude.
func relativeDifference(got, want []float32) float64 {
	var largest, worst float64
	for index := range want {
		largest = max(largest, math.Abs(float64(want[index])))
		worst = max(worst, math.Abs(float64(got[index])-float64(want[index])))
	}
	if largest == 0 {
		return worst
	}
	return worst / largest
}
